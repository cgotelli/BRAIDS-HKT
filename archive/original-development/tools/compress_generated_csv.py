#!/usr/bin/env python3
"""Transparently compress a completed, authenticated CSV on macOS HFS+/APFS.

Call only after its producing process has finished. The required expected
SHA256 identifies the completed artifact. A sibling copy is compressed and
verified before atomic replacement; unsupported compression leaves the
original alone. Readers keep using the same path and ordinary open()/CSV APIs.
No archive format or evidence-content changes are introduced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time

DITTO = "/usr/bin/ditto"
UF_COMPRESSED = 0x20


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_state(path: Path) -> dict:
    s = path.lstat()
    return {"logical_bytes": s.st_size, "allocated_bytes": s.st_blocks * 512,
            "device": s.st_dev, "inode": s.st_ino, "links": s.st_nlink,
            "mode": s.st_mode, "uid": s.st_uid, "gid": s.st_gid,
            "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns,
            "flags": getattr(s, "st_flags", 0)}


def compress_csv(path: str | Path, *, expected_sha256: str) -> dict:
    """Compress one completed CSV, or retain it byte-for-byte when unsupported.

    Callers must exclude active producers. Stability checks detect changes
    during the copy; this helper is not synchronization with another writer.
    """
    path = Path(path).absolute()
    if path.suffix.lower() != ".csv":
        raise ValueError("Only explicitly selected generated CSV files are supported")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Expected SHA256 must be 64 lowercase hexadecimal characters")
    before = file_state(path)
    if not stat.S_ISREG(before["mode"]) or before["links"] != 1:
        raise ValueError("Source must be a regular file with exactly one link; symlinks are unsupported")
    if sys.platform != "darwin":
        raise RuntimeError("Transparent ditto compression requires macOS HFS+ or APFS")
    started = time.monotonic()
    if sha256(path) != expected_sha256 or file_state(path) != before:
        raise ValueError("Source does not match the stable completed artifact SHA256")
    result = {"path": str(path), "sha256": expected_sha256, "before": before,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "method": "ditto --hfsCompression --noclone; verify then atomic replace",
              "helper_sha256": sha256(Path(__file__)),
              "scope": "Completed generated CSV only; unchanged logical bytes and ordinary readers"}
    with tempfile.TemporaryDirectory(prefix=".csv-compression-", dir=path.parent) as staging:
        candidate = Path(staging) / path.name
        subprocess.run([DITTO, "--hfsCompression", "--noclone", str(path), str(candidate)],
                       check=True, capture_output=True, text=True)
        after = file_state(candidate)
        if not stat.S_ISREG(after["mode"]) or after["links"] != 1:
            raise RuntimeError("Compressed candidate is not an independent regular file")
        if sha256(candidate) != expected_sha256 or after["logical_bytes"] != before["logical_bytes"]:
            raise RuntimeError("Compressed candidate changed logical bytes; original retained")
        for key in ("mode", "uid", "gid", "mtime_ns"):
            if after[key] != before[key]:
                raise RuntimeError(f"Compressed candidate changed {key}; original retained")
        if after["flags"] & ~UF_COMPRESSED != before["flags"] & ~UF_COMPRESSED:
            raise RuntimeError("Compressed candidate changed unrelated file flags; original retained")
        if file_state(path) != before or sha256(path) != expected_sha256 or file_state(path) != before:
            raise RuntimeError("Source changed during compression; replacement cancelled")
        result["candidate"] = after
        if not after["flags"] & UF_COMPRESSED or after["allocated_bytes"] >= before["allocated_bytes"]:
            result.update(status="UNCHANGED", reason="No verified filesystem-compression space saving", after=before)
        else:
            with candidate.open("rb") as stream:
                os.fsync(stream.fileno())
            # Source remains in place through every integrity check above.
            os.replace(candidate, path)
            result.update(status="COMPRESSED", after=file_state(path))
            if result["after"]["inode"] != after["inode"] or sha256(path) != expected_sha256:
                raise RuntimeError("Atomic replacement identity verification failed")
    result["saved_allocated_bytes"] = before["allocated_bytes"] - result["after"]["allocated_bytes"]
    result["elapsed_s"] = time.monotonic() - started
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--record", type=Path, help="Optional JSON compression record; never the input CSV")
    args = parser.parse_args(argv)
    if args.record is not None and args.record.absolute() == args.path.absolute():
        parser.error("Record path cannot overwrite the CSV")
    try:
        result = compress_csv(args.path, expected_sha256=args.expected_sha256)
    except Exception as exc:
        result = {"status": "ERROR", "path": str(args.path.absolute()), "error": str(exc)}
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.record is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(text)
    print(text, end="")
    return 1 if result["status"] == "ERROR" else 0


if __name__ == "__main__":
    raise SystemExit(main())
