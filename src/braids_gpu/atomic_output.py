"""Atomic GPU-worker output publication resilient to brief Windows locks.

Only WinError 5 (access denied) and 32 (sharing violation) are retried, and
only while replacing a fully closed temporary file. Other errors retain their
original meaning. The old target stays intact until replacement succeeds.
"""
from __future__ import annotations

import json
import io
import math
import os
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
from braids_core.windows_paths import filesystem_path


WINDOWS_TRANSIENT_REPLACE_ERRORS = frozenset((5, 32))
DEFAULT_ATTEMPTS = 8
INITIAL_DELAY_S = 0.02
MAXIMUM_DELAY_S = 1.0


def replace_with_windows_retry(source, target, *, attempts=DEFAULT_ATTEMPTS,
                               initial_delay_s=INITIAL_DELAY_S,
                               maximum_delay_s=MAXIMUM_DELAY_S):
    """Publish a complete file with a bounded retry of Windows rename locks."""
    if not isinstance(attempts, int) or attempts < 1:
        raise ValueError('Positive integer replacement attempt count required')
    if (not math.isfinite(initial_delay_s) or initial_delay_s < 0 or
            not math.isfinite(maximum_delay_s) or maximum_delay_s < 0):
        raise ValueError('Finite nonnegative replacement retry delays required')
    source=filesystem_path(source);target=filesystem_path(target)
    delay=float(initial_delay_s)
    for attempt in range(attempts):
        try:
            os.replace(source,target)
            return
        except OSError as error:
            if (getattr(error,'winerror',None) not in WINDOWS_TRANSIENT_REPLACE_ERRORS or
                    attempt+1==attempts):
                raise
            time.sleep(delay)
            delay=min(float(maximum_delay_s),max(float(initial_delay_s),2*delay))


def _temporary_for(path):
    path=filesystem_path(path)
    # A copied destination name plus PID/thread/UUID can exceed Windows'
    # legacy path limit even when the destination itself is perfectly valid.
    # UUID plus exclusive creation retains concurrent-writer isolation without
    # lengthening the name according to the destination or thread identifier.
    return path.with_name(f'.{uuid4().hex}.partial')


def _cleanup(temporary):
    try:
        temporary.unlink(missing_ok=True)
    except OSError:
        # If a scanner still holds the failed staging file, never mask the
        # original publication error or alter the intact destination.
        pass


def write_json_atomic(path, value):
    """Use the runner's JSON format and publish with a unique staging file."""
    path=filesystem_path(path)
    payload=json.dumps(value,indent=2,allow_nan=False)+'\n'
    temporary=_temporary_for(path)
    try:
        with temporary.open('x',encoding='utf-8') as stream:
            stream.write(payload)
        replace_with_windows_retry(temporary,path)
    finally:
        _cleanup(temporary)


def save_arrays_atomic(path, **arrays):
    """Close the NPZ handle before atomic publication, as required on Windows."""
    path=filesystem_path(path)
    temporary=_temporary_for(path)
    try:
        with temporary.open('xb') as stream:
            np.savez(stream,**arrays)
        replace_with_windows_retry(temporary,path)
    finally:
        _cleanup(temporary)


def arrays_to_bytes(**arrays):
    """Encode the existing uncompressed NPZ format without a temporary file.

    Checkpoint owners nest authenticated payloads. Keeping intermediate layers
    in memory avoids writing and reading each layer before its parent is saved.
    The array inventory, metadata and NumPy archive representation are unchanged.
    """
    stream=io.BytesIO()
    np.savez(stream,**arrays)
    return stream.getvalue()


def write_bytes_atomic(path, payload):
    """Publish an already encoded checkpoint with the same Windows guarantees."""
    path=filesystem_path(path)
    temporary=_temporary_for(path)
    try:
        with temporary.open('xb') as stream:
            stream.write(payload)
        replace_with_windows_retry(temporary,path)
    finally:
        _cleanup(temporary)
