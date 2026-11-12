"""One QProcess lifecycle for the explicitly selected CPU/GPU river backend."""
import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

from qgis.PyQt.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from braids_core.river_project import load_project
from braids_core.windows_paths import filesystem_path


class RiverRunner(QObject):
    log = pyqtSignal(str)
    progress = pyqtSignal(float, str)
    live_frame = pyqtSignal(str)
    finished = pyqtSignal(str, bool, str)
    busy_changed = pyqtSignal(bool)

    def __init__(self, parent=None, *, development_runtime=False):
        super().__init__(parent)
        self.development_runtime = development_runtime
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.poll)
        self.action = None
        self.output = self.stop_file = None
        self.cancel_requested = False
        self._frame_stamp = None
        self._last_progress = None
        self.log_path = None

    @property
    def is_running(self):
        return self.action is not None

    def prepare(self, project, output, *, python, runtime, log_path=None):
        # Full new-format validation before any process or target directory.
        load_project(project)
        self._launch("prepare", ["--project", str(filesystem_path(project).resolve())], output, python, runtime, log_path)

    def run(self, prepared, output, *, python, runtime, restart=None, log_path=None):
        prepared = filesystem_path(prepared).resolve()
        receipt = json.loads((prepared/"PREPARED.json").read_text(encoding="utf-8"))
        from braids_core.river_worker import PREPARED_SCHEMA
        if receipt.get("schema") != PREPARED_SCHEMA:
            raise ValueError("Old prepared folders cannot run. Prepare a new BRAIDS project.")
        args = ["--input", str(prepared)]
        if restart:
            args += ["--restart", str(filesystem_path(restart).resolve())]
        self._launch("run", args, output, python, runtime, log_path)

    def _launch(self, action, arguments, output, python, runtime, log_path=None):
        if self.is_running:
            raise RuntimeError("Preparation or execution is already active")
        python, runtime, output = filesystem_path(python), filesystem_path(runtime).resolve(), filesystem_path(output).resolve()
        if not python.is_file():
            raise ValueError("Choose a Python interpreter with the solver dependencies")
        if not (runtime/"braids_core/river_worker.py").is_file():
            raise ValueError("The unified BRAIDS runtime is missing")
        if output.exists():
            raise FileExistsError(output)
        if self.development_runtime:
            invocation = ["-m", "braids_core.river_worker"]
        else:
            if not (runtime/"RUNTIME.json").is_file() or not (runtime/"braids_core/river_launcher.py").is_file():
                raise ValueError("A complete authenticated unified runtime is required")
            invocation = [str(runtime/"braids_core/river_launcher.py")]
        output.parent.mkdir(parents=True, exist_ok=True)
        self.stop_file = output.parent/(".braids_stop_"+uuid4().hex)
        args = invocation+[action, *arguments, "--output", str(output), "--stop-file", str(self.stop_file)]
        environment = QProcessEnvironment.systemEnvironment()
        # QGIS ships its own GDAL/PROJ/Python DLLs. Do not contaminate the
        # explicitly selected external interpreter with QGIS runtime paths.
        for name in ("PYTHONHOME", "QT_PLUGIN_PATH", "QGIS_PREFIX_PATH",
                     "GDAL_DRIVER_PATH", "GDAL_DATA", "PROJ_LIB", "PROJ_DATA"):
            environment.remove(name)
        search = [str(python.parent/"Library/bin"), str(python.parent), str(python.parent/"Scripts"),
                  os.environ.get("SystemRoot", "C:/Windows")+"/System32"]
        if self.development_runtime:
            # Checkout tests compile the BRAIDS native libraries. Installed
            # releases must carry authenticated DLLs instead of using a compiler.
            compiler = shutil.which("gfortran")
            if compiler:
                search.insert(0, str(Path(compiler).parent))
        if os.environ.get("CUDA_PATH"):
            search.append(str(Path(os.environ["CUDA_PATH"])/"bin"))
        environment.insert("PATH", os.pathsep.join(search))
        environment.insert("PYTHONPATH", str(runtime))
        environment.insert("PYTHONNOUSERSITE", "1")
        environment.insert("PYTHONDONTWRITEBYTECODE", "1")
        environment.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(runtime))
        self.process.setProgram(str(python))
        self.process.setArguments(args)
        self.log_path = filesystem_path(log_path).resolve() if log_path else output.with_suffix('.log')
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(action=action, python=str(python), runtime=str(runtime),
                output=str(output), arguments=args), ensure_ascii=False)+'\n')
        self.action, self.output = action, output
        self.cancel_requested = False
        self._frame_stamp = self._last_progress = None
        self.busy_changed.emit(True)
        self.progress.emit(0., "Validating project and runtime")
        self.process.start()
        self.timer.start()

    def stop(self):
        if self.is_running and not self.cancel_requested:
            self.cancel_requested = True
            # Exclusive marker, closed immediately; solver checks preparation
            # loops and accepted-step boundaries. No forced process kill.
            with self.stop_file.open("x", encoding="utf-8") as stream:
                stream.write("Cancel requested by QGIS\n")
            self._log("Cancellation requested; waiting for a safe boundary.")

    def _log(self, text):
        if self.log_path is not None:
            with self.log_path.open('a', encoding='utf-8') as stream:
                stream.write(text+'\n')
        self.log.emit(text)

    def _read_output(self):
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self._log(text.rstrip())

    def poll(self):
        if not self.is_running:
            return
        progress = self.output/"PROGRESS.json"
        try:
            values = json.loads(progress.read_text(encoding="utf-8"))
            phase=values.get("phase", values.get("status", ""))
            hydraulic=values.get('hydraulic',{})
            if phase=='running' and hydraulic.get('next_substep_hint_s') is not None:
                phase+=(f" (explicit hydraulic step target {hydraulic['next_substep_hint_s']:.4g} s; "
                    f"{hydraulic.get('cfl_retries',0)} CFL reductions, "
                    f"{hydraulic.get('spatial_retries',0)} spatial/positivity retries)")
            observed = (float(values["percent"]), phase)
            if observed != self._last_progress:
                self._last_progress = observed
                self.progress.emit(*observed)
        except (OSError, ValueError, KeyError):
            pass  # An unpublished progress file is not process completion.
        index = self.output/"frames/INDEX.json"
        try:
            stat = index.stat()
            stamp = stat.st_mtime_ns, stat.st_size
            if stamp != self._frame_stamp:
                json.loads(index.read_text(encoding="utf-8"))
                self._frame_stamp = stamp
                self.live_frame.emit(str(self.output))
        except (OSError, ValueError):
            pass

    def _error(self, error):
        if error == QProcess.FailedToStart and self.action is not None:
            self._complete(False, self.process.errorString())

    def _finished(self, code, exit_status):
        if self.action is None:
            return
        self._read_output()
        self.poll()
        try:
            if self.action == "prepare":
                receipt = json.loads((self.output/"PREPARED.json").read_text(encoding="utf-8"))
                from braids_core.river_worker import PREPARED_SCHEMA
                okay = code == 0 and receipt.get("schema") == PREPARED_SCHEMA
                message = "Preparation complete" if okay else "Preparation failed"
                if okay:
                    slopes=json.loads((self.output/'NORMAL_SLOPES.json').read_text(encoding='utf-8'))
                    for name,row in slopes['boundaries'].items():
                        self._log(f"Normal depth — {name}: automatic slope {row['energy_slope']:.8g} m/m; "
                            f"{row['sample_cells']} finest cells, bed-fit RMS {row['bed_fit_rms_m']:.6g} m. "
                            "Uniform-flow energy-slope proxy from initial bed; fixed across AMR/restart.")
            else:
                status = json.loads((self.output/"STATUS.json").read_text(encoding="utf-8"))
                okay = code == 0 and status["status"] in ("complete", "segment_complete")
                message = status.get("error") or status["status"].replace("_", " ")
        except (OSError, ValueError, KeyError):
            okay, message = False, "Preparation or execution did not publish a complete receipt; see the log"
        if self.cancel_requested and not okay:
            message = "Cancelled; any accepted checkpoint has been retained"
        self._complete(okay, message)

    def _complete(self, okay, message):
        self._log(message)
        action, self.action = self.action, None
        self.timer.stop()
        self.busy_changed.emit(False)
        self.finished.emit(action, okay, message)
