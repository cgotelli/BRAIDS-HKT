"""Nonblocking QProcess runner for the native BRAIDS executable."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from qgis.PyQt.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from .project import (RUN_MARKER, update_run_status, verified_native_executable,
                      verified_prepared_inputs)


class BraidsRunner(QObject):
    started=pyqtSignal();stdout=pyqtSignal(str);stderr=pyqtSignal(str)
    finished=pyqtSignal(int,bool);progress=pyqtSignal(float);live_frame=pyqtSignal(int,float)

    def __init__(self,parent=None):
        super().__init__(parent);self.process=QProcess(self);self._run_root=None;self._stop_requested=False
        self.process.readyReadStandardOutput.connect(self._stdout)
        self.process.readyReadStandardError.connect(self._stderr)
        self.process.started.connect(self.started);self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error);self._finalized=True
        self._stop_file=None
        self._terminate_timer=QTimer(self);self._terminate_timer.setSingleShot(True);self._terminate_timer.timeout.connect(self._terminate)
        self._kill_timer=QTimer(self);self._kill_timer.setSingleShot(True);self._kill_timer.timeout.connect(self._kill)
        # The native solver writes a tiny completion marker, so polling once a
        # second is cheap and makes the t=0 preview visible immediately.
        self._progress_timer=QTimer(self);self._progress_timer.setInterval(1000);self._progress_timer.timeout.connect(self._progress)
        self._end_time_s=None;self._last_output_time_s=0.;self._progress_text_tail="";self._live_frame_id=-1

    @property
    def is_running(self):return self.process.state()!=QProcess.ProcessState.NotRunning

    def start(self,run_root,checkpoint=None,profile=False):
        if self.is_running:raise RuntimeError("A BRAIDS simulation is already running")
        root=Path(run_root).resolve();marker=json.loads((root/RUN_MARKER).read_text(encoding="utf-8"))
        if marker.get("preparation_error"):raise ValueError("Run preparation failed: "+marker["preparation_error"])
        if marker.get("status") not in {"prepared","failed","cancelled"}:raise ValueError("BRAIDS run is not prepared")
        verified_prepared_inputs(root,marker)
        executable=verified_native_executable(root,marker)
        arguments=["case.data"]
        if checkpoint:arguments.append(str(Path(checkpoint).expanduser().resolve()))
        environment=QProcessEnvironment.systemEnvironment()
        # Time integration is selected by the prepared deck/solver's guarded
        # auto policy.  Never let a parent shell silently override it.
        environment.remove("BRAIDS_TIME_INTEGRATOR")
        environment.insert("OMP_NUM_THREADS",str(marker["threads"]));environment.insert("OMP_DYNAMIC","FALSE")
        # MinGW libgomp on Windows can put workers to sleep between the solver's
        # short RK kernels and does not support the requested affinity binding.
        # Keep the selected worker pool warm; this changes latency only and is
        # byte-identical to passive execution.
        environment.insert("OMP_PROC_BIND","FALSE");environment.insert("OMP_WAIT_POLICY","ACTIVE");environment.insert("GOMP_SPINCOUNT","1000000")
        environment.insert("BRAIDS_PROGRESS_STEPS","250");environment.insert("BRAIDS_CHECKPOINT_STEPS","10000")
        # A unique absolute marker lets the native solver stop only between
        # accepted steps. Remove abandoned markers from an earlier launcher;
        # they must never cancel a clean run or restart implicitly.
        for stale in root.glob(".braids-native-stop-*.request"):
            try:stale.unlink()
            except FileNotFoundError:pass
        self._stop_file=(root/f".braids-native-stop-{uuid4().hex}.request").resolve()
        environment.insert("BRAIDS_STOP_FILE",str(self._stop_file))
        if profile:environment.insert("BRAIDS_PROFILE","1")
        else:environment.remove("BRAIDS_PROFILE")
        self.process.setProcessEnvironment(environment);self.process.setWorkingDirectory(str(root))
        self.process.setProgram(str(executable));self.process.setArguments(arguments)
        self._run_root=root;self._stop_requested=False;self._last_output_time_s=0.;self._progress_text_tail="";self._live_frame_id=-1
        try:self._end_time_s=float(json.loads((root/"project.json").read_text(encoding="utf-8"))["numerics"]["end_time_s"])
        except (OSError,ValueError,KeyError,json.JSONDecodeError):self._end_time_s=None
        header=(f"BRAIDS execution started {datetime.now(timezone.utc).isoformat()}\n"
                f"Run folder: {root}\nExecutable: {executable}\nOpenMP threads: {marker['threads']}\n"
                f"Native performance profile: {'enabled' if profile else 'disabled'}\n\n")
        (root/"native.log").write_text(header,encoding="utf-8");self.stdout.emit(header)
        update_run_status(root,"running",checkpoint=str(checkpoint) if checkpoint else None)
        self._finalized=False;self._progress_timer.start();self.process.start()

    def stop(self):
        if not self.is_running or self._stop_requested:return
        self._stop_requested=True
        try:
            # Exclusive creation makes this request durable and prevents an
            # accidental overwrite from disguising an unrelated path error.
            with self._stop_file.open("x",encoding="ascii") as handle:
                handle.write("stop\n");handle.flush()
        except (OSError,AttributeError):
            # If the cooperative channel itself fails, retain the historical
            # immediate terminate plus timed hard-kill behavior. That fallback
            # cannot promise a new accepted-step restart checkpoint.
            self.process.terminate();self._kill_timer.start(3000);return
        message="BRAIDS stop requested; waiting for an accepted-step boundary.\n"
        try:self._append_log(message)
        except OSError:pass
        self.stdout.emit(message)
        # Keep the UI nonblocking while allowing a large-domain accepted step
        # and its transactional checkpoint to finish. A genuinely blocked
        # process still receives terminate and then kill as last resorts.
        self._terminate_timer.start(30000)

    def _terminate(self):
        if self.is_running:self.process.terminate();self._kill_timer.start(3000)

    def _kill(self):
        if self.is_running:self.process.kill()

    def _append_log(self,text):
        if text and self._run_root is not None:
            with (self._run_root/"native.log").open("a",encoding="utf-8") as handle:handle.write(text)

    def _stdout(self):
        output=bytes(self.process.readAllStandardOutput()).decode("utf-8",errors="replace");self._append_log(output);self.stdout.emit(output)
        if self._end_time_s:
            combined=self._progress_text_tail+output
            matches=re.findall(r"progress:\s*time=\s*([0-9Ee+.-]+)",combined)
            self._progress_text_tail=combined[-256:]
            if matches:
                self._last_output_time_s=float(matches[-1]);self.progress.emit(min(100.,100*self._last_output_time_s/self._end_time_s))

    def _stderr(self):
        output=bytes(self.process.readAllStandardError()).decode("utf-8",errors="replace");self._append_log(output);self.stderr.emit(output)

    def _progress(self):
        if self._run_root is None:return
        try:
            # Native stdout drives progress. The timer only repeats the last
            # parsed value; it never opens the growing result frame file.
            if self._end_time_s and self._last_output_time_s:
                self.progress.emit(min(100.,100*self._last_output_time_s/self._end_time_s))
            latest=self._run_root/"live_preview"/"latest.txt"
            if latest.is_file():
                frame,time_s,_=latest.read_text(encoding="utf-8").split()
                frame=int(frame)
                if frame>self._live_frame_id:
                    self._live_frame_id=frame;self.live_frame.emit(frame,float(time_s))
        except (OSError,ValueError,KeyError,IndexError,json.JSONDecodeError):pass

    def _process_error(self,error):
        # FailedToStart does not emit QProcess.finished. Complete the UI lifecycle
        # explicitly so Run can be retried after fixing the executable problem.
        if error==QProcess.ProcessError.FailedToStart:
            message=f"Could not start the native solver: {self.process.errorString()}\n"
            self.stderr.emit(message)
            try:self._append_log(message)
            except OSError:pass
            self._finished(-1,QProcess.ExitStatus.CrashExit)

    def _finished(self,exit_code,exit_status):
        if self._finalized:return
        self._finalized=True;self._progress_timer.stop();self._terminate_timer.stop();self._kill_timer.stop()
        normal=exit_status==QProcess.ExitStatus.NormalExit;complete=bool(self._run_root and ((self._run_root/"frames.txt").is_file() or (self._run_root/"live_preview"/"frames.csv").is_file()) and (self._run_root/"checkpoint.bin").is_file())
        succeeded=not self._stop_requested and exit_code==0 and normal and complete
        footer=(f"\nBRAIDS execution finished {datetime.now(timezone.utc).isoformat()}\n"
                f"Status: {'completed' if succeeded else ('cancelled' if self._stop_requested else 'failed')}\nReturn code: {int(exit_code)}\n")
        try:
            self._stdout();self._stderr();self._append_log(footer);self.stdout.emit(footer)
            if self._run_root:update_run_status(self._run_root,"completed" if succeeded else ("cancelled" if self._stop_requested else "failed"),returncode=int(exit_code))
            if succeeded:self.progress.emit(100.)
        except Exception as exc:
            self.stderr.emit(f"Could not finalize the run log or status: {exc}\n")
        finally:
            try:
                if self._stop_file is not None:self._stop_file.unlink(missing_ok=True)
            except OSError:pass
            self._stop_file=None
            self.finished.emit(int(exit_code),succeeded)
