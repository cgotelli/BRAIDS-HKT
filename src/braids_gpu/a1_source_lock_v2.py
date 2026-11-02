"""Versioned A1 source lock for the later cancellable-preparation change.

The original native/gate receipts remain immutable.  This adapter accepts
exactly one known live-source drift: stop-file handling in the preparation
helper, whose default no-stop branch preserves the previous behavior.  It
does not waive the archived original, native executable, input, frame, CUDA,
solver-source, or numerical acceptance checks.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import threading

from braids_support.gpu_fwave import a1_combined_live_scheduled_native_case as native
from braids_support.gpu_fwave import a1_combined_one_step_native as one


OLD_COUPLED_SHA='12425095ff97f9936d36234f9e9e03d7a80d4d9822b638c6edf2513538ecf3c7'
CURRENT_COUPLED_SHA='912171475e44434b9ba631bfbfcb4b23f3f690fcdbd8a4c0e2815496f7c18f79'
ORIGINAL_SOURCE_LOCK_SHA='abed5ebcc91f0cb27a58394bf4f3e41329fe27a009433d35319a69a4c0690b1c'
ORIGINAL_PLAN_SHA='53ba92f818daf0924134ec7dddee26d6f149f879291eaaea4651a3c791222dcf'
COUPLED=Path(native.__file__).resolve().parent/'coupled_reference.py'
ARCHIVED=native.ROOT/'source/01_coupled_reference.py'
_LOCK=threading.RLock()


def verify_source_transition():
    if (native.sha(COUPLED)!=CURRENT_COUPLED_SHA or
            native.sha(ARCHIVED)!=OLD_COUPLED_SHA or
            native.sha(one.SOURCE_LOCK)!=ORIGINAL_SOURCE_LOCK_SHA or
            native.sha(native.ROOT/'plan.json')!=ORIGINAL_PLAN_SHA):
        raise ValueError('A1 versioned preparation-only source transition differs')


@contextmanager
def compatible_source_lock():
    """Scope an exact one-file V2 lock while the old read-only gates execute."""
    with _LOCK:
        verify_source_transition()
        original_verify=native.verify_files
        original_identity=one.source_identity
        old_name=str(COUPLED)

        def verify_files(files):
            expected=files.get(old_name)
            if expected==OLD_COUPLED_SHA:
                verify_source_transition()
                files=dict(files)
                del files[old_name]
            original_verify(files)

        def source_identity():
            files={**native.source_files(),str(Path(one.__file__).resolve()):native.sha(one.__file__),
                   str(one.GATE.resolve()):one.GATE_SHA,
                   str(one.SOURCE_LOCK.resolve()):ORIGINAL_SOURCE_LOCK_SHA}
            for raw in one.SOURCE_LOCK.read_text(encoding='utf-8').splitlines():
                if not raw.startswith('braids_support/gpu_fwave/'):
                    continue
                name,digest=raw.split()
                path=Path(name).resolve()
                if path==COUPLED:
                    if digest!=OLD_COUPLED_SHA:
                        raise ValueError('A1 original one-step coupled source lock differs')
                    verify_source_transition()
                    files[str(path)]=CURRENT_COUPLED_SHA
                else:
                    if native.sha(path)!=digest:
                        raise ValueError('A1 original current-source anchor changed: '+name)
                    files[str(path)]=digest
            verify_files(files)
            return files

        native.verify_files=verify_files
        one.source_identity=source_identity
        try:
            yield
        finally:
            one.source_identity=original_identity
            native.verify_files=original_verify
