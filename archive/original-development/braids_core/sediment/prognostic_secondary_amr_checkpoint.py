"""Atomic on-disk schema-10 checkpoint for prognostic-secondary AMR.

The lifecycle module owns the physical schema.  This module only serializes a
fully committed :class:`Schema10Checkpoint`, authenticates it on the host, and
optionally copies the accepted arrays into already allocated host storage.
It deliberately has no CPU/GPU dispatch and cannot select or fall back to a
backend.

Publication is write-once.  Both files are completed in a sibling staging
directory before one directory rename makes the checkpoint visible.  An
existing checkpoint is never replaced.
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import MappingProxyType

import numpy as np

from braids_core.windows_paths import filesystem_path

from .prognostic_secondary_amr_execution import (
    CHECKPOINT_SCHEMA,
    validate_prognostic_secondary_amr_execution_document,
)
from .prognostic_secondary_amr_lifecycle import (
    PrognosticSecondaryAMRLifecycleError,
    RegisterAudit,
    Schema10Checkpoint,
    register_audit,
    validate_schema10_checkpoint,
)


ARTIFACT_SCHEMA = 1
KIND = "braids_prognostic_secondary_amr_schema10"
PAYLOAD_FILE = "accepted_state.npz"
RECEIPT_FILE = "checkpoint.json"
_WINDOWS_TRANSIENT_ERRORS = frozenset((5, 32, 33))
_RECEIPT_KEYS = frozenset({
    "artifact_schema",
    "kind",
    "complete",
    "checkpoint_schema",
    "payload_file",
    "payload_file_sha256",
    "payload_digest",
    "execution_identity",
    "source_bindings",
    "document",
    "array_storage",
    "register_storage",
})
_SOURCE_BINDING_KEYS = (
    "parent_execution_fingerprint",
    "source_contract_sha256",
    "canonical_fixture_sha256",
    "betac_operator_amendment_sha256",
    "implementation_source_sha256",
)


def _fail(message: str, error: Exception | None = None):
    failure = PrognosticSecondaryAMRLifecycleError(
        "Q3-H3/schema-10 disk " + message
    )
    if error is None:
        raise failure
    raise failure from error


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        _fail("JSON differs", error)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _temporary_file(path: Path) -> Path:
    # The fixed short name avoids pushing a valid Windows destination over the
    # legacy path limit.  The enclosing staging directory is already unique.
    return path.with_name("." + path.name + ".partial")


def _replace_file(source: Path, target: Path) -> None:
    delay = 0.02
    for attempt in range(8):
        try:
            os.replace(source, target)
            return
        except OSError as error:
            if (
                os.name != "nt"
                or getattr(error, "winerror", None)
                not in _WINDOWS_TRANSIENT_ERRORS
                or attempt == 7
            ):
                raise
            time.sleep(delay)
            delay = min(2.0 * delay, 0.5)


def _write_json(path: Path, value: object) -> None:
    temporary = _temporary_file(path)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace_file(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _write_npz(path: Path, fields: Mapping[str, np.ndarray]) -> None:
    temporary = _temporary_file(path)
    try:
        with temporary.open("xb") as stream:
            np.savez(stream, **fields)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_file(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _publish_directory(staging: Path, target: Path) -> None:
    delay = 0.02
    for attempt in range(8):
        if target.exists() or target.is_symlink():
            _fail("checkpoint is write-once")
        try:
            os.rename(staging, target)
            return
        except OSError as error:
            if (
                os.name != "nt"
                or getattr(error, "winerror", None)
                not in _WINDOWS_TRANSIENT_ERRORS
                or attempt == 7
            ):
                raise
            time.sleep(delay)
            delay = min(2.0 * delay, 0.5)


def _storage_rows(checkpoint: Schema10Checkpoint):
    arrays = []
    fields: dict[str, np.ndarray] = {}
    for index, (name, value) in enumerate(sorted(checkpoint.arrays.items())):
        field = f"accepted_{index:04d}"
        arrays.append({"name": name, "field": field})
        fields[field] = value
    registers = []
    for index, item in enumerate(checkpoint.completed_registers):
        ids_field = f"register_{index:04d}_face_source_ids"
        values_field = f"register_{index:04d}_values"
        registers.append({
            "index": index,
            "component": item.component,
            "face_source_ids_field": ids_field,
            "values_field": values_field,
        })
        fields[ids_field] = item.face_source_ids
        fields[values_field] = item.values
    return arrays, registers, fields


def write_schema10_checkpoint(
    path: str | os.PathLike[str], checkpoint: Schema10Checkpoint
) -> Mapping[str, object]:
    """Atomically publish one complete, immutable schema-10 directory."""
    validate_schema10_checkpoint(checkpoint)
    document = json.loads(_canonical(dict(checkpoint.document)).decode("ascii"))
    identity = validate_prognostic_secondary_amr_execution_document(
        document["execution_identity"]
    )
    # Keep the final path unresolved so a caller cannot hide a symlink behind
    # ``Path.resolve`` and bypass the write-once/artifact checks.
    target = filesystem_path(path)
    if target.exists() or target.is_symlink():
        _fail("checkpoint is write-once")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = filesystem_path(tempfile.mkdtemp(
        prefix=".q3h3-schema10-", dir=target.parent
    ))
    try:
        array_storage, register_storage, fields = _storage_rows(checkpoint)
        payload = staging / PAYLOAD_FILE
        _write_npz(payload, fields)
        receipt = {
            "artifact_schema": ARTIFACT_SCHEMA,
            "kind": KIND,
            "complete": True,
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "payload_file": PAYLOAD_FILE,
            "payload_file_sha256": _sha256_file(payload),
            "payload_digest": checkpoint.payload_digest,
            "execution_identity": identity,
            "source_bindings": {
                key: identity[key] for key in _SOURCE_BINDING_KEYS
            },
            "document": document,
            "array_storage": array_storage,
            "register_storage": register_storage,
        }
        _write_json(staging / RECEIPT_FILE, receipt)
        _publish_directory(staging, target)
        # Return detached JSON data; callers cannot mutate the disk receipt
        # through a nested object retained by this function.
        return MappingProxyType(json.loads(_canonical(receipt).decode("ascii")))
    finally:
        if staging.exists():
            if (
                staging.parent != target.parent
                or not staging.name.startswith(".q3h3-schema10-")
                or staging.is_symlink()
            ):
                _fail("staging ownership differs")
            try:
                shutil.rmtree(staging)
            except OSError:
                # A Windows scanner may briefly retain a failed staging file.
                # The target remains absent or already-complete either way.
                pass


def _read_receipt(folder: Path) -> dict[str, object]:
    members = {item.name for item in folder.iterdir()} if folder.is_dir() else set()
    if (
        not folder.is_dir()
        or folder.is_symlink()
        or members != {PAYLOAD_FILE, RECEIPT_FILE}
        or any(
            not (folder / name).is_file() or (folder / name).is_symlink()
            for name in members
        )
    ):
        _fail("artifact set differs")
    try:
        value = json.loads((folder / RECEIPT_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail("receipt differs", error)
    if not isinstance(value, dict) or set(value) != _RECEIPT_KEYS:
        _fail("receipt inventory differs")
    if (
        value["artifact_schema"] != ARTIFACT_SCHEMA
        or value["kind"] != KIND
        or value["complete"] is not True
        or value["checkpoint_schema"] != CHECKPOINT_SCHEMA
        or value["payload_file"] != PAYLOAD_FILE
        or value["payload_file_sha256"] != _sha256_file(folder / PAYLOAD_FILE)
    ):
        _fail("receipt header or file digest differs")
    return value


def _validate_storage_rows(value: object, names: tuple[str, ...]):
    if not isinstance(value, list) or len(value) != len(names):
        _fail("array storage inventory differs")
    result = []
    for index, (row, name) in enumerate(zip(value, names, strict=True)):
        expected = {"name": name, "field": f"accepted_{index:04d}"}
        if row != expected:
            _fail("array storage order differs")
        result.append(expected)
    return result


def read_schema10_checkpoint(
    path: str | os.PathLike[str],
    *,
    expected_execution_identity: Mapping[str, object],
) -> Schema10Checkpoint:
    """Authenticate a complete checkpoint before route/device allocation."""
    expected = validate_prognostic_secondary_amr_execution_document(
        expected_execution_identity
    )
    folder = filesystem_path(path)
    receipt = _read_receipt(folder)
    try:
        stored_identity = validate_prognostic_secondary_amr_execution_document(
            receipt["execution_identity"]
        )
    except (TypeError, ValueError) as error:
        _fail("execution identity differs", error)
    if (
        _canonical(stored_identity) != _canonical(expected)
        or receipt["source_bindings"] != {
            key: expected[key] for key in _SOURCE_BINDING_KEYS
        }
    ):
        _fail("expected execution/profile/source identity differs")
    document = receipt["document"]
    if (
        not isinstance(document, dict)
        or document.get("schema") != CHECKPOINT_SCHEMA
        or _canonical(document.get("execution_identity"))
        != _canonical(expected)
        or receipt["payload_digest"] != hashlib.sha256(
            _canonical(document)
        ).hexdigest()
    ):
        _fail("checkpoint document digest differs")
    array_document = document.get("arrays")
    if not isinstance(array_document, dict):
        _fail("checkpoint array document differs")
    names = tuple(sorted(array_document))
    storage = _validate_storage_rows(receipt["array_storage"], names)
    register_document = document.get("completed_registers")
    register_storage = receipt["register_storage"]
    if (
        not isinstance(register_document, list)
        or not isinstance(register_storage, list)
        or len(register_document) != len(register_storage)
    ):
        _fail("register storage inventory differs")
    expected_fields = {row["field"] for row in storage}
    for index, (metadata, row) in enumerate(
        zip(register_document, register_storage, strict=True)
    ):
        expected_row = {
            "index": index,
            "component": metadata.get("component")
            if isinstance(metadata, dict) else None,
            "face_source_ids_field": f"register_{index:04d}_face_source_ids",
            "values_field": f"register_{index:04d}_values",
        }
        if row != expected_row:
            _fail("register storage order differs")
        expected_fields.update((
            expected_row["face_source_ids_field"], expected_row["values_field"]
        ))
    try:
        with np.load(folder / PAYLOAD_FILE, allow_pickle=False) as archive:
            if (len(archive.files) != len(expected_fields) or
                    set(archive.files) != expected_fields):
                _fail("payload field inventory differs")
            loaded = {
                name: np.array(archive[name], copy=True, order="C")
                for name in archive.files
            }
    except PrognosticSecondaryAMRLifecycleError:
        raise
    except (OSError, ValueError, EOFError) as error:
        _fail("payload archive differs", error)
    arrays: dict[str, np.ndarray] = {}
    for row in storage:
        value = loaded[row["field"]]
        value.setflags(write=False)
        arrays[row["name"]] = value
    registers: list[RegisterAudit] = []
    for metadata, row in zip(register_document, register_storage, strict=True):
        try:
            item = register_audit(
                metadata["component"],
                topology_fingerprint=metadata["topology_fingerprint"],
                mesh_epoch=metadata["mesh_epoch"],
                face_source_ids=loaded[row["face_source_ids_field"]],
                values=loaded[row["values_field"]],
                pending=metadata["pending"],
            )
        except (KeyError, TypeError, ValueError) as error:
            _fail("register payload differs", error)
        if item.digest != metadata.get("digest"):
            _fail("register digest differs")
        registers.append(item)
    checkpoint = Schema10Checkpoint(
        MappingProxyType(json.loads(_canonical(document).decode("ascii"))),
        MappingProxyType(arrays),
        tuple(registers),
        str(receipt["payload_digest"]),
    )
    validate_schema10_checkpoint(checkpoint)
    return checkpoint


def restore_schema10_checkpoint(
    path: str | os.PathLike[str],
    *,
    expected_execution_identity: Mapping[str, object],
    destination_arrays: MutableMapping[str, np.ndarray] | None = None,
) -> Schema10Checkpoint:
    """Read first, then atomically copy accepted host arrays if requested.

    With no ``destination_arrays`` this returns the immutable, authenticated
    restore image.  A supplied destination must contain the exact schema-10
    array inventory.  Every destination is validated and snapshotted before
    any copy; an unexpected copy failure restores all original bytes.
    Controller metadata is returned in ``checkpoint.document`` and remains the
    caller's responsibility because this backend-neutral layer cannot mutate a
    CPU or CUDA controller.
    """
    checkpoint = read_schema10_checkpoint(
        path, expected_execution_identity=expected_execution_identity
    )
    if destination_arrays is None:
        return checkpoint
    if not isinstance(destination_arrays, MutableMapping) or set(
        destination_arrays
    ) != set(checkpoint.arrays):
        _fail("restore destination inventory differs")
    targets: dict[str, np.ndarray] = {}
    for name, source in checkpoint.arrays.items():
        target = np.asarray(destination_arrays[name])
        if (
            target.dtype != source.dtype
            or target.shape != source.shape
            or not target.flags.c_contiguous
            or not target.flags.writeable
        ):
            _fail("restore destination differs: " + name)
        targets[name] = target
    target_items = list(targets.items())
    for index, (left_name, left) in enumerate(target_items):
        for right_name, right in target_items[index + 1:]:
            if np.shares_memory(left, right):
                _fail(
                    "restore destinations alias: " + left_name + "/" + right_name
                )
    before = {name: value.copy(order="C") for name, value in targets.items()}
    try:
        for name in sorted(targets):
            np.copyto(targets[name], checkpoint.arrays[name], casting="no")
    except Exception as error:
        for name, value in before.items():
            np.copyto(targets[name], value, casting="no")
        _fail("restore copy failed and was rolled back", error)
    return checkpoint


__all__ = [
    "ARTIFACT_SCHEMA",
    "KIND",
    "PAYLOAD_FILE",
    "RECEIPT_FILE",
    "read_schema10_checkpoint",
    "restore_schema10_checkpoint",
    "write_schema10_checkpoint",
]
