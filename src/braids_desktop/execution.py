"""Nonblocking standalone execution through the authenticated packaged launcher."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
import codecs
import shutil
from pathlib import Path
import sys
from uuid import uuid4

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from braids_core import river_folder
from braids_core.river_project import identity, load_project
from .services import project_member, update_project_session
from .presentation import display_number


def remove_old_runs(project_root, *, keep=None, protected=None):
    """Remove finished BRAIDS run directories inside this project."""
    root=Path(project_root).resolve();runs=(root/'runs').resolve()
    if runs.parent!=root or not runs.is_dir() or (root/'runs').is_symlink():return []
    keep=Path(keep).resolve() if keep else None
    protected=Path(protected).resolve() if protected else None
    removed=[]
    for path in runs.iterdir():
        if not path.name.startswith('run_') or not path.is_dir() or path.is_symlink() or getattr(path,'is_junction',lambda:False)():continue
        resolved=path.resolve()
        if resolved.parent!=runs or resolved==keep or (protected and protected.is_relative_to(resolved)):continue
        try:
            status=json.loads((resolved/'STATUS.json').read_text(encoding='utf-8'))
            if status.get('status') not in ('complete','segment_complete','failed','cancelled'):continue
            shutil.rmtree(resolved);removed.append(path.name)
        except (OSError,ValueError,TypeError):continue
    return removed


class DesktopRunner(QObject):
    log = Signal(str)
    progress = Signal(float, str)
    live_frame = Signal(str)
    finished = Signal(str, bool, str)
    busy_changed = Signal(bool)

    def __init__(self, parent=None, *, runtime, python=None):
        super().__init__(parent)
        self.runtime = Path(runtime).resolve()
        self.python = Path(python or sys.executable).resolve()
        if self.python.name.lower() in ('pythonw.exe','braids.exe'):self.python=self.python.with_name('python.exe')
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.poll)
        self.action = None
        self.output = self.log_path = self.stop_file = self.project_root = None
        self.cancel_requested = False
        self._frame_stamp = self._last_progress = None
        self._checkpoint_stamp=None
        self._event = None
        self._output_decoder=codecs.getincrementaldecoder('utf-8')(errors='replace')

    @property
    def is_running(self):
        return self.action is not None

    def _check_idle(self):
        if self.is_running:
            raise RuntimeError("Preparation or execution is already active")

    def _target(self, project_root, action):
        root = Path(project_root).resolve()
        document = load_project(root / river_folder.PROJECT_FILE)
        from .workspace import check_runnable
        check_runnable(root,document)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        parent = "prepared" if action == "prepare" else "runs"
        target = project_member(root, Path(parent) / f"{action}_{stamp}_{uuid4().hex[:8]}")
        return root, document, target

    def prepare(self, project_root):
        self._check_idle()
        root, document, output = self._target(project_root, "prepare")
        self._launch("prepare", root, document, output,
                     ["--project", str(root / river_folder.PROJECT_FILE),'--asset-cache',str(project_member(root,'model_layers/preparation_cache'))], prepared=output)
        return output

    def run(self, project_root, prepared, *, restart=None, until=None,initial_state=None):
        self._check_idle()
        root, document, output = self._target(project_root, "run")
        if restart is not None and initial_state is not None:raise ValueError('Choose exact continuation or initialized saved fields')
        continuation=False
        if restart is not None:
            restart=project_member(root,restart)
            sidecar=restart.with_suffix('.json')
            if sidecar.is_file():
                from braids_core.river_worker import validate_continuation
                from braids_core.sediment.river_leaf_io import sha
                location=json.loads(sidecar.read_text(encoding='utf-8'))
                if location.get('schema')!='braids.checkpoint-location.v1' or location.get('sha256')!=sha(restart):raise ValueError('Checkpoint location or checksum differs')
                prepared=project_member(root,restart.parent/location['prepared'])
                validate_continuation(load_project(prepared/'PROJECT.json'),document)
                continuation=True
        prepared = project_member(root, prepared)
        receipt = json.loads((prepared / "PREPARED.json").read_text(encoding="utf-8"))
        from braids_core.river_worker import PREPARED_SCHEMA
        if receipt.get("schema") != PREPARED_SCHEMA:
            raise ValueError("A current prepared folder is required; prepare this project again")
        original=load_project(prepared/'PROJECT.json') if continuation else document
        if receipt.get("project_id") != identity(original):
            raise ValueError("The project changed after preparation; prepare the saved settings again")
        arguments = ["--input", str(prepared)]
        arguments+=['--checkpoint-dir',str(project_member(root,Path('checkpoints')/output.name))]
        if continuation:arguments+=['--continuation-project',str(root/river_folder.PROJECT_FILE)]
        if restart is not None:
            restart = project_member(root, restart)
            if not restart.is_file():
                raise FileNotFoundError(restart)
            from braids_core.river_worker import checkpoint_identities
            try:
                # Fast UI preflight reads only the small identity member. The
                # authenticated worker still verifies the complete payload.
                import numpy as np
                from braids_core.river_project import read_json
                with np.load(restart,allow_pickle=False) as archive:
                    if set(archive.files)!={'identity','payload','state','initialization'} or archive['identity'].shape!=():raise ValueError('Current exact-checkpoint format required; use saved-state initialization for older data')
                    metadata=read_json(str(archive['identity']))
                checksum=metadata.pop('payload_sha256',None)
                for name in ('state_sha256','initialization_sha256'):
                    digest=metadata.pop(name,None)
                    if not isinstance(digest,str) or len(digest)!=64:raise ValueError('Checkpoint state checksum missing')
                if not isinstance(checksum,str) or len(checksum)!=64 or metadata not in checkpoint_identities(original,receipt):raise ValueError('Checkpoint identity differs')
            except Exception as exc:
                raise ValueError('The selected resume checkpoint does not match this project, backend, settings or runtime. '
                    'Clear "Checkpoint to resume" to start a NEW run, or select a checkpoint from exactly these saved settings.\n'+str(exc)) from exc
            arguments += ["--restart", str(restart)]
        if initial_state is not None:
            initial_state=project_member(root,initial_state)
            from braids_core.river_state_initialization import load
            state_meta,_=load(initial_state,document,receipt)
            arguments+=['--initial-state',str(initial_state)]
        if until is not None:
            if type(until) not in (float, int) or not math.isfinite(until) or until <= 0:
                raise ValueError("Run segment end must be positive finite seconds")
            arguments += ["--until", str(until)]
        self._launch("run", root, document, output, arguments, prepared=prepared, restart=restart)
        return output

    def _launch(self, action, root, document, output, arguments, *, prepared, restart=None):
        launcher = self.runtime / "braids_core/river_launcher.py"
        if not self.python.is_file():
            raise ValueError("The desktop Python interpreter is unavailable")
        if not launcher.is_file() or not (self.runtime / "RUNTIME.json").is_file():
            raise ValueError("The independent authenticated BRAIDS Desktop runtime is missing")
        if output.exists():
            raise FileExistsError(output)
        # No checkout/development route. The child verifies every runtime artifact
        # before importing the worker and loads the packaged native libraries.
        stop = project_member(root, output.parent / (".braids_stop_" + uuid4().hex))
        bootstrap=Path(__file__).resolve().parents[1]/'desktop/worker.py'
        args = ["-B", str(bootstrap), str(launcher), action, *arguments, "--output", str(output), "--stop-file", str(stop)]
        environment = QProcessEnvironment.systemEnvironment()
        for name in ("PYTHONHOME", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QGIS_PREFIX_PATH",
                     "GDAL_DRIVER_PATH", "GDAL_DATA", "PROJ_LIB", "PROJ_DATA"):
            environment.remove(name)
        search = [str(self.python.parent / "Library/bin"), str(self.python.parent),
                  str(self.python.parent / "Scripts"), os.environ.get("SystemRoot", "C:/Windows") + "/System32"]
        if self.python.parent == Path(sys.executable).resolve().parent:
            # A venv interpreter lives in .venv/Scripts, while its scientific
            # DLLs remain in the base conda installation. Never use QGIS paths.
            base = Path(sys.base_prefix)
            search = [str(base / "Library/bin"), str(base), str(base / "Scripts"), *search]
        if os.environ.get("CUDA_PATH"):
            search.append(str(Path(os.environ["CUDA_PATH"]) / "bin"))
        environment.insert("PATH", os.pathsep.join(search))
        environment.insert("PYTHONPATH", str(self.runtime))
        environment.insert("PYTHONNOUSERSITE", "1")
        environment.insert("PYTHONDONTWRITEBYTECODE", "1")
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(self.runtime))
        self.process.setProgram(str(self.python))
        self.process.setArguments(args)
        output.parent.mkdir(parents=True, exist_ok=True)
        log_path = project_member(root, Path("history") / (output.name + ".log"))
        event = dict(action=action, output=output.relative_to(root).as_posix(),
                     prepared=prepared.relative_to(root).as_posix(), log=log_path.relative_to(root).as_posix(),
                     project_id=identity(document), backend=document["model"]["request"]["execution"]["backend"])
        if restart is not None:
            event["restart"] = restart.relative_to(root).as_posix()
        with log_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(action=action, python=str(self.python), runtime=str(self.runtime),
                                         output=str(output), arguments=args), ensure_ascii=False) + "\n")
        river_folder.record_event(root, "desktop_worker", status="started", **event)
        self.action, self.output, self.stop_file = action, output, stop
        self.project_root, self.log_path, self._event = root, log_path, event
        self.cancel_requested = False
        self._output_decoder.reset()
        self._frame_stamp = self._last_progress = None
        self._checkpoint_stamp=None
        from .project_retention import active_marker
        river_folder._atomic_json(active_marker(root,output.name),dict(pid=os.getpid(),action=action,output=str(output)))
        self.busy_changed.emit(True)
        self.progress.emit(0., "Validating project and authenticated runtime")
        self.timer.start()
        self.process.start()

    def stop(self):
        if self.is_running and not self.cancel_requested:
            with self.stop_file.open("x", encoding="utf-8") as stream:
                stream.write("Cancel requested by BRAIDS Desktop\n")
            self.cancel_requested = True
            if self.action=='prepare':
                phase=self._last_progress[1] if self._last_progress else 'validating dependencies'
                message=f'Preparation cancellation requested. Waiting for the current preparation operation to reach its cancellation check ({phase}). No hydraulic timestep is running; incomplete preparation will not be used.'
            else:
                message='Simulation cancellation requested. The current trial is stopped/rolled back at its next cancellation check; the last accepted state is retained for a checkpoint.'
            self._log(message)
            self.progress.emit(self._last_progress[0] if self._last_progress else 0.,message)

    def _log(self, text):
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(text + "\n")
        self.log.emit(text)

    def _read_output(self):
        text = self._output_decoder.decode(bytes(self.process.readAllStandardOutput()))
        if text:
            self._log(text.rstrip())

    def poll(self):
        if not self.is_running:
            return
        if self.action=='run':
            try:
                folder=self.project_root/'checkpoints'/self.output.name
                stamp=folder.stat().st_mtime_ns
                if stamp!=self._checkpoint_stamp:
                    from .project_retention import prune_live_checkpoints
                    selected=river_folder.folder_info(self.project_root)['session'].get('restart')
                    count=prune_live_checkpoints(self.project_root,self.output.name,protected=(selected,self._event.get('restart')))
                    self._checkpoint_stamp=folder.stat().st_mtime_ns
                    if count:self._log(f'Removed {count} superseded checkpoint(s); latest accepted checkpoint retained.')
            except (OSError,ValueError,KeyError):pass
        try:
            values = json.loads((self.output / "PROGRESS.json").read_text(encoding="utf-8"))
            phase = values.get("phase", values.get("status", ""))
            if self.cancel_requested:phase=('Cancelling preparation: ' if self.action=='prepare' else 'Cancelling simulation: ')+phase
            hydraulic = values.get("hydraulic", {})
            if phase == "running" and hydraulic.get("next_substep_hint_s") is not None:
                phase += (f"; hydraulic step target {display_number(hydraulic['next_substep_hint_s'])} s, "
                          f"{hydraulic.get('cfl_retries', 0)} CFL reductions")
            observed = (float(values["percent"]), phase)
            if observed != self._last_progress:
                self._last_progress = observed
                self.progress.emit(*observed)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        try:
            index = self.output / "frames/INDEX.json"
            stat = index.stat()
            stamp = stat.st_mtime_ns, stat.st_size
            if stamp != self._frame_stamp:
                json.loads(index.read_text(encoding="utf-8"))
                self._frame_stamp = stamp
                self.live_frame.emit(str(self.output))
        except (OSError, ValueError):
            pass  # This is a UI notification, never an integrity verification cache.

    def _error(self, error):
        if error == QProcess.FailedToStart and self.is_running:
            self._complete(False, self.process.errorString())

    def _finished(self, code, exit_status):
        if not self.is_running:
            return
        self._read_output()
        self.poll()
        try:
            if self.action == "prepare":
                from braids_core.river_worker import PREPARED_SCHEMA
                receipt = json.loads((self.output / "PREPARED.json").read_text(encoding="utf-8"))
                okay = code == 0 and receipt.get("schema") == PREPARED_SCHEMA
                message = "Preparation complete" if okay else "Preparation failed"
                if okay:
                    slopes = json.loads((self.output / "NORMAL_SLOPES.json").read_text(encoding="utf-8"))
                    for name, row in slopes["boundaries"].items():
                        self._log(f"Normal depth — {name}: mesh energy-slope proxy {row['energy_slope']:.8g} m/m; "
                                  f"{row['sample_cells']} finest cells, bed-fit RMS {row['bed_fit_rms_m']:.6g} m.")
            else:
                status = json.loads((self.output / "STATUS.json").read_text(encoding="utf-8"))
                okay = code == 0 and status["status"] in ("complete", "segment_complete")
                message = status.get("error") or status["status"].replace("_", " ")
        except (OSError, ValueError, KeyError, TypeError):
            okay, message = False, "Worker did not publish a complete receipt; see the execution log"
            try:
                status = json.loads((self.output / "STATUS.json").read_text(encoding="utf-8"))
                message = status.get("error") or message
            except (OSError, ValueError, TypeError):
                pass
        if self.cancel_requested and not okay:
            message = ('Preparation cancelled; incomplete mesh/preparation is not runnable.' if self.action=='prepare' else 'Simulation cancelled; any accepted checkpoint has been retained.')
        self._complete(okay, message)

    def _complete(self, okay, message):
        action = self.action
        self.timer.stop()
        from .project_retention import active_marker,cleanup_project
        active_marker(self.project_root,self.output.name).unlink(missing_ok=True)
        status = "complete" if okay else "cancelled" if self.cancel_requested else "failed"
        try:
            river_folder.record_event(self.project_root, "desktop_worker", status=status,
                                      message=message, **self._event)
            session = {"log": self._event["log"]}
            if action=='run':session['output']=self._event['output']
            if action == "prepare" and okay:
                session["prepared"] = self._event["prepared"]
                session['output']=''
            checkpoint = self.output / "checkpoint.npz"
            # A produced checkpoint is recovery evidence, not a user decision to
            # resume the next run. Never auto-populate the resume editor.
            update_project_session(self.project_root, **session)
            # Cleaning generated storage must not change the solver outcome.
            try:
                restart=self._event.get('restart') or river_folder.folder_info(self.project_root)['session'].get('restart')
                result=cleanup_project(self.project_root,protected=(restart,) if restart else ())
                if result.get('removed'):
                    self._log(f'Project cleanup: removed {len(result["removed"])} older generated items ({result["removed_bytes"]/1e9:.3f} GB). Latest run, preparation and checkpoint retained; selected restart dependencies protected.')
                if result.get('errors'):self._log('Some obsolete files are still in use; cleanup will retry after the next operation.')
            except Exception as exc:self._log('Project cleanup deferred: '+str(exc))
        except Exception as exc:
            message += f"; folder history/session publication failed: {exc}"
            okay = False  # Keep all solver output; never pretend history was persisted.
        self._log(message)
        self.action = None
        self.busy_changed.emit(False)
        self.finished.emit(action, okay, message)
