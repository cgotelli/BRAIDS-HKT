"""Byte-checked temporary I/O for native Windows runtimes with MAX_PATH limits.

User projects, receipts and outputs remain in their requested directories. Only
the native file-open operation uses temporary local files; no physical setting,
backend, source identity or numerical array is changed by this adapter.
"""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

from .windows_paths import filesystem_path


def _hash(path):
    with filesystem_path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@contextmanager
def native_directory(folder, inputs, *, outputs=(), cancelled=lambda: None):
    folder = filesystem_path(folder)
    names = tuple(inputs)
    for name in (*names, *outputs):
        if not isinstance(name, str) or Path(name).name != name or not name.isascii():
            raise ValueError('Native I/O requires local ASCII basenames')
    if set(names) & set(outputs):
        raise ValueError('Native outputs may not overwrite inputs')
    expected = {}
    for name in names:
        path = folder / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Native input must be an existing regular file: ' + name)
        expected[name] = _hash(path)
    if os.name != 'nt':
        yield folder
        return
    # A read-only native load already in a short ASCII directory needs no
    # second copy. Output-producing invocations retain the private staging and
    # exclusive publication below. Keep pre/post authentication even here:
    # input decks are mutable files, not cached runtime source material.
    ordinary = str(folder)
    if (not outputs and ordinary.isascii() and not ordinary.startswith('\\\\')
            and all(len(str(folder / name)) < 240 for name in names)
            and len(ordinary) < 240):
        cancelled()
        yield folder
        cancelled()
        for name, digest in expected.items():
            if _hash(folder / name) != digest:
                raise ValueError('Native input changed during execution: ' + name)
        return
    # CreateProcessW rejects an extended (>MAX_PATH) working directory even
    # though Python can open its files. Relative native filenames also avoid
    # passing UTF-8 absolute paths to a narrow Fortran runtime.
    with tempfile.TemporaryDirectory(prefix='braids-native-') as temporary:
        stage = filesystem_path(temporary)
        if len(str(stage)) >= 240:
            raise OSError('Windows native temporary directory is too long; select a shorter TEMP directory')
        for name in names:
            cancelled()
            shutil.copyfile(folder / name, stage / name)
            if _hash(stage / name) != expected[name]:
                raise ValueError('Native input copy differs: ' + name)
        cancelled()
        yield stage
        cancelled()
        for name, digest in expected.items():
            if _hash(folder / name) != digest or _hash(stage / name) != digest:
                raise ValueError('Native input changed during execution: ' + name)
        for name in outputs:
            cancelled()
            source, target = stage / name, folder / name
            if source.is_symlink() or not source.is_file():
                raise ValueError('Native output missing or linked: ' + name)
            with source.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            if _hash(source) != _hash(target):
                raise ValueError('Native output copy differs: ' + name)
