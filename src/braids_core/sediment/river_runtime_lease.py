"""Run-scoped, byte-authenticated immutable file leases for AMR construction.

On Windows a live read handle denies writes and deletion, rather than trusting
mtime/size as evidence that a cached checksum is still correct. Rebuilt owners
reuse the checksum while that handle remains open. All handles close when the
worker exits, including cancellation/failure; a later run hashes fresh bytes.
Non-Windows callers retain full reads (no advisory-lock integrity shortcut).
Only source and native library files qualify, never mutable frames/checkpoints.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import ctypes
import functools
import hashlib
import os
from pathlib import Path
import threading
from braids_core.windows_paths import filesystem_path

_current = ContextVar('braids_river_runtime_lease', default=None)
_suffixes = frozenset(('.py', '.h', '.cu', '.cpp', '.f90', '.f', '.inc', '.dll', '.so', '.pyd'))


class RuntimeFileLease:
    def __init__(self):
        self.files = {}
        self.lock = threading.RLock()
        self.reads = 0
        self.hits = 0
        self.closed = False
        self.verified = {}
        self.observers = []

    def digest(self, path, *, immutable=False):
        path = filesystem_path(path)
        if os.name != 'nt' or (not immutable and path.suffix.lower() not in _suffixes):
            return hashlib.sha256(path.read_bytes()).hexdigest()
        with self.lock:
            if self.closed:
                raise RuntimeError('Runtime file lease is closed')
            key = os.path.normcase(str(path))
            for observer in self.observers:observer.add(key)
            if key in self.files:
                state = os.stat(self.files[key][3])
                if (state.st_dev, state.st_ino) != self.files[key][2]:
                    raise ValueError('Leased runtime path now identifies another file')
                self.hits += 1
                return self.files[key][1]
            # FILE_SHARE_READ only: a writer/deleter cannot coexist, even if
            # it attempts to preserve timestamps and file length.
            import msvcrt
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            create = kernel.CreateFileW
            create.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                               ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                               ctypes.c_void_p]
            create.restype = ctypes.c_void_p
            handle = create(os.fspath(path), 0x80000000, 1, None, 3, 0x80, None)
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
            except BaseException:
                kernel.CloseHandle.argtypes = [ctypes.c_void_p]
                kernel.CloseHandle(handle)
                raise
            stream = os.fdopen(descriptor, 'rb')
            try:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            except BaseException:
                stream.close()
                raise
            state = os.fstat(stream.fileno())
            self.files[key] = (stream, digest, (state.st_dev, state.st_ino), os.fspath(path))
            self.reads += 1
            return digest

    def verify(self, key, receipt, validator, inventory):
        """Reuse a full check only while all its byte dependencies stay leased."""
        with self.lock:
            if key in self.verified:
                result,dependencies,previous_inventory=self.verified[key]
                for path in dependencies:
                    state=os.stat(self.files[path][3])
                    if (state.st_dev,state.st_ino)!=self.files[path][2]:
                        raise ValueError('Verified runtime dependency now identifies another file')
                if inventory(result)!=previous_inventory:
                    raise ValueError('Verified runtime source inventory changed')
                return deepcopy(result)
            dependencies=set();self.observers.append(dependencies)
            try:
                self.digest(receipt,immutable=True)
                result=validator()
                self.verified[key]=(deepcopy(result),frozenset(dependencies),inventory(result))
                return result
            finally:self.observers.pop()

    def close(self):
        with self.lock:
            for stream, _, _, _ in self.files.values():
                stream.close()
            self.files.clear()
            self.verified.clear()
            self.closed = True


def runtime_lease_active():
    """Archive I/O must keep active lease dependencies on the calling thread."""
    return _current.get() is not None


def digest(path):
    lease = _current.get()
    return (hashlib.sha256(filesystem_path(path).read_bytes()).hexdigest() if lease is None
            else lease.digest(path))


def verified_sources(folder, validator, inventory):
    """Memoize a source-validator contract, never mutable solver/input state."""
    lease=_current.get()
    folder=filesystem_path(folder).resolve()
    if lease is None or os.name!='nt':return validator(folder)
    return lease.verify(('transport_sources',str(folder)),
        folder/'general_live_transport_sources.json',lambda:validator(folder),inventory)


@contextmanager
def runtime_file_lease():
    existing = _current.get()
    if existing is not None:
        yield existing
        return
    lease = RuntimeFileLease()
    token = _current.set(lease)
    try:
        yield lease
    finally:
        _current.reset(token)
        lease.close()


def leased_runtime(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with runtime_file_lease():
            return function(*args, **kwargs)
    return wrapped
