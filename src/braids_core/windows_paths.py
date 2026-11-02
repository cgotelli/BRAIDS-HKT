"""Long Windows paths for filesystem I/O without changing receipt identities.

``str(path)`` remains the ordinary absolute spelling used by JSON and users.
``os.fspath(path)`` uses the extended Windows namespace, independently of the
machine's LongPathsEnabled registry setting. Child paths retain this behavior.
No global pathlib monkey patch, registry change, or dependency is required.
"""
from __future__ import annotations

import os
from pathlib import Path


def _ordinary(value: str) -> str:
    if value.startswith("\\\\?\\UNC\\"):
        return "\\\\" + value[8:]
    if value.startswith("\\\\?\\"):
        return value[4:]
    return value


def _extended(value: str) -> str:
    if value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


class IOPath(type(Path())):
    """A Path whose OS representation supports long paths on Windows."""

    def __fspath__(self):
        value = str(self)
        if os.name != "nt":
            return value
        # Resolve dot segments lexically before the extended namespace, where
        # Windows intentionally stops normalizing them. Do not resolve links:
        # callers must still be able to detect and reject a symlink itself.
        return _extended(os.path.abspath(_ordinary(value)))

    def resolve(self, strict=False):
        resolved = super().resolve(strict=strict)
        return type(self)(_ordinary(str(resolved)) if os.name == "nt" else str(resolved))

    def iterdir(self):
        # Python 3.13 pathlib passes str(self), not self, to scandir.
        # Keep the canonical spelling while using the OS spelling for I/O.
        with os.scandir(self) as entries:
            names = [entry.name for entry in entries]
        return (self / name for name in names)

    def glob(self, pattern, **kwargs):
        for path in Path(os.fspath(self)).glob(pattern, **kwargs):
            yield filesystem_path(path)

    def rglob(self, pattern, **kwargs):
        for path in Path(os.fspath(self)).rglob(pattern, **kwargs):
            yield filesystem_path(path)


def filesystem_path(path) -> IOPath:
    """Return an absolute I/O-capable path; keep final-link checks possible."""
    value = str(path)
    if os.name == "nt":
        value = _ordinary(value)
    return IOPath(os.path.abspath(value))


def canonical_path(path) -> Path:
    """Canonical unprefixed spelling for persisted identities and display."""
    return Path(str(filesystem_path(path).resolve()))
