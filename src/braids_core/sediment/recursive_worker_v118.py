"""Production lifecycle contract for the generalized recursive bank owner.

This module does not duplicate any numerical operator.  It binds the v115--
v117 transaction to an explicitly selected Q10 CPU/GPU profile and owns the
parts that a real worker must own: cancellation, immutable checkpoints,
restart authentication, result frames, and atomic topology replacement.

The numerical advance remains injected by the hydraulic/AMR worker because it
owns the accepted augmented-f-wave state and face registers.  The adapter
verifies that an injected advance publishes exactly one accepted macro and
never changes the selected backend silently.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
from collections.abc import Callable, Mapping

import numpy as np

from braids_core.sediment.qgis_product_v109 import (
    CPU_BACKEND, GPU_BACKEND, ProductResolutionV109,
    RECURSIVE_BANK_WORKER_VARIANT,
)
from .bed_columns_v101 import host
from .distributed_banks_v117 import DistributedBankOwnerV117
from .recursive_cross_physics_v115 import (
    CoupledAMRRecursiveCrossPhysicsV115,
)
from .recursive_physical_toe_v116 import (
    CoupledAMRRecursivePhysicalToeV116,
)


SCHEMA = "braids.recursive_worker.v118"
PREPARED_SCHEMA = "braids.recursive_worker_prepared.v118"
CHECKPOINT_SCHEMA = "braids.recursive_worker_checkpoint.v118"
RESULT_SCHEMA = "braids.recursive_worker_result.v118"
PREPARED_SIDECAR = "recursive_worker_v118.json"
LATEST_CHECKPOINT = "recursive_checkpoint_latest_v118.json"
RUN_MANIFEST = "run.json"
ROUTES = {2: "physical_toe_mode2", 3: "distributed_bank_mode3"}


class RecursiveWorkerV118Error(ValueError):
    """The prepared identity, live owner, checkpoint, or result differs."""


def _canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _cancelled(value) -> bool:
    return bool(value() if callable(value) else value)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Publish a small file without reusing a data-file pathname.

    Checkpoint payloads themselves are immutable and uniquely named.  Only
    this small pointer/manifest is replaced, avoiding the Windows raster-style
    lock failure which originally motivated unique staging files.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial-" + secrets.token_hex(8))
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _pack(value, arrays: dict[str, np.ndarray], prefix="root"):
    if isinstance(value, np.ndarray) or (
            hasattr(value, "get") and hasattr(value, "shape") and
            hasattr(value, "dtype")):
        key = f"a{len(arrays):08d}"
        arrays[key] = np.ascontiguousarray(host(value))
        return {"__array__": key}
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _pack(asdict(value), arrays, prefix)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise RecursiveWorkerV118Error(
                f"{prefix} contains a non-string mapping key")
        return {key: _pack(child, arrays, f"{prefix}.{key}")
                for key, child in value.items()}
    if isinstance(value, tuple):
        return {"__tuple__": [_pack(child, arrays, prefix)
                              for child in value]}
    if isinstance(value, list):
        return [_pack(child, arrays, prefix) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            raise RecursiveWorkerV118Error(
                f"{prefix} contains a non-finite scalar")
        return value
    raise RecursiveWorkerV118Error(
        f"{prefix} contains unsupported checkpoint type {type(value).__name__}")


def _unpack(value, arrays: Mapping[str, np.ndarray]):
    if isinstance(value, Mapping) and set(value) == {"__array__"}:
        key = value["__array__"]
        if key not in arrays:
            raise RecursiveWorkerV118Error("Checkpoint array inventory differs")
        return arrays[key].copy()
    if isinstance(value, Mapping) and set(value) == {"__tuple__"}:
        return tuple(_unpack(child, arrays) for child in value["__tuple__"])
    if isinstance(value, Mapping):
        return {key: _unpack(child, arrays) for key, child in value.items()}
    if isinstance(value, list):
        return [_unpack(child, arrays) for child in value]
    return value


def _write_archive(path: Path, value) -> str:
    arrays: dict[str, np.ndarray] = {}
    tree = _pack(value, arrays)
    manifest = _canonical({"tree": tree, "arrays": {
        key: {"dtype": array.dtype.str, "shape": list(array.shape)}
        for key, array in arrays.items()}}).decode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial-" + secrets.token_hex(8))
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, __manifest__=np.asarray(manifest),
                                **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _file_sha(path)


def _read_archive(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        if "__manifest__" not in archive.files:
            raise RecursiveWorkerV118Error("Checkpoint manifest is missing")
        raw = archive["__manifest__"]
        if raw.shape != () or raw.dtype.kind != "U":
            raise RecursiveWorkerV118Error("Checkpoint manifest type differs")
        try:
            manifest = json.loads(raw.item())
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RecursiveWorkerV118Error(
                "Checkpoint manifest is invalid") from error
        specifications = manifest.get("arrays")
        if (not isinstance(specifications, Mapping) or
                set(archive.files) != set(specifications) | {"__manifest__"}):
            raise RecursiveWorkerV118Error("Checkpoint array inventory differs")
        arrays = {}
        for key, specification in specifications.items():
            array = archive[key]
            invalid = (not isinstance(specification, Mapping) or
                       specification.get("dtype") != array.dtype.str or
                       specification.get("shape") != list(array.shape))
            if array.dtype.kind in "fc":
                invalid = invalid or not np.isfinite(array).all()
            if invalid:
                raise RecursiveWorkerV118Error(
                    "Checkpoint array specification differs")
            arrays[key] = array.copy()
    return _unpack(manifest.get("tree"), arrays)


def _backend_is_gpu(value: str) -> bool:
    text = str(value).casefold()
    return "gpu" in text or "cuda" in text


def _owner_cross(owner):
    if isinstance(owner, CoupledAMRRecursivePhysicalToeV116):
        return owner.cross
    if isinstance(owner, CoupledAMRRecursiveCrossPhysicsV115):
        return owner
    raise RecursiveWorkerV118Error("A v115/v116 recursive owner is required")


def _owner_route(owner) -> tuple[int, str]:
    if isinstance(owner, CoupledAMRRecursivePhysicalToeV116):
        return 2, ROUTES[2]
    cross = _owner_cross(owner)
    if all(isinstance(value, DistributedBankOwnerV117)
           for value in cross.banks.values()):
        return 3, ROUTES[3]
    raise RecursiveWorkerV118Error(
        "The recursive owner is neither qualified mode 2 nor mode 3")


def _profile_document(profile):
    if isinstance(profile, ProductResolutionV109):
        return profile.as_dict()
    if not isinstance(profile, Mapping):
        raise RecursiveWorkerV118Error("A resolved QGIS product is required")
    return dict(profile)


@dataclass(frozen=True)
class RecursivePreparedV118:
    schema: str
    worker_variant: str
    route: str
    bank_mode: int
    backend: str
    integrator: str
    automatic_backend_fallback: bool
    product_fingerprint: str
    selected_method_ids: tuple[str, ...]
    topology_policy: str
    checkpoint_policy: str
    result_policy: str
    identity: str

    def as_dict(self):
        value = asdict(self)
        value["selected_method_ids"] = list(value["selected_method_ids"])
        return value

    @classmethod
    def create(cls, profile, *, bank_mode: int):
        document = _profile_document(profile)
        if bank_mode not in ROUTES:
            raise RecursiveWorkerV118Error("Bank mode must be 2 or 3")
        required = document.get("worker_requirements")
        methods = document.get("selected_method_ids")
        if (document.get("backend") not in (CPU_BACKEND, GPU_BACKEND) or
                not isinstance(required, Mapping) or
                required.get("required_worker_variant") !=
                    RECURSIVE_BANK_WORKER_VARIANT or
                required.get("recursive_bank_contract") != SCHEMA or
                not isinstance(methods, list) or
                not all(isinstance(value, str) for value in methods) or
                "redistribution.braids_toe_collapse" not in methods or
                not {"bed.layered", "bedform.enabled"} <= set(methods)):
            raise RecursiveWorkerV118Error(
                "QGIS product is not the admitted recursive bank bundle")
        body = {
            "schema": PREPARED_SCHEMA,
            "worker_variant": RECURSIVE_BANK_WORKER_VARIANT,
            "route": ROUTES[bank_mode], "bank_mode": bank_mode,
            "backend": document["backend"],
            "integrator": document["integrator"],
            "automatic_backend_fallback": False,
            "product_fingerprint": document.get("fingerprint"),
            "selected_method_ids": methods,
            "topology_policy": "private_rebuild_then_atomic_install_v117",
            "checkpoint_policy": "immutable_unique_archive_v118",
            "result_policy": "accepted_macro_frames_v118",
        }
        if not isinstance(body["product_fingerprint"], str):
            raise RecursiveWorkerV118Error("QGIS product fingerprint is absent")
        return cls(**{**body,
                      "selected_method_ids": tuple(methods),
                      "identity": _digest(body)})

    @classmethod
    def load(cls, value):
        if not isinstance(value, Mapping):
            raise RecursiveWorkerV118Error("Prepared worker document is invalid")
        raw = dict(value); identity = raw.pop("identity", None)
        methods = raw.get("selected_method_ids")
        if (raw.get("schema") != PREPARED_SCHEMA or
                raw.get("worker_variant") != RECURSIVE_BANK_WORKER_VARIANT or
                raw.get("bank_mode") not in ROUTES or
                raw.get("route") != ROUTES.get(raw.get("bank_mode")) or
                raw.get("automatic_backend_fallback") is not False or
                not isinstance(methods, list) or _digest(raw) != identity):
            raise RecursiveWorkerV118Error(
                "Prepared recursive worker identity differs")
        try:
            return cls(**{**raw, "selected_method_ids": tuple(methods),
                          "identity": identity})
        except (TypeError, ValueError) as error:
            raise RecursiveWorkerV118Error(
                "Prepared recursive worker fields differ") from error


def _snapshot(owner, parent) -> dict[str, object]:
    cross = _owner_cross(owner)
    arrays = {}
    for level in ("coarse", "fine"):
        column = cross.surface.levels[level].columns
        bank = cross.banks[level]
        arrays[f"{level}_layers_kg_m2"] = np.ascontiguousarray(
            host(column.mass))
        leaf = getattr(parent, f"{level}_numerical_parent")
        arrays[f"{level}_bed_elevation_m"] = np.ascontiguousarray(
            host(leaf.bed_elevation_m))
        if hasattr(bank, "history"):
            arrays[f"{level}_toe_cumulative_m"] = np.ascontiguousarray(
                host(bank.history["toe_cumulative"]))
            arrays[f"{level}_collapse_count"] = np.ascontiguousarray(
                host(bank.history["collapse_count"]))
        else:
            arrays[f"{level}_bank_undercut_m"] = np.ascontiguousarray(
                host(bank.state["undercut_m"]))
            arrays[f"{level}_collapse_count"] = np.ascontiguousarray(
                host(bank.state["collapse_count"]))
        if isinstance(owner, CoupledAMRRecursivePhysicalToeV116):
            arrays[f"{level}_toe_face_signed_kg"] = np.ascontiguousarray(
                host(owner.ledgers[level].signed_kg))
            arrays[f"{level}_toe_face_gross_kg"] = np.ascontiguousarray(
                host(owner.ledgers[level].gross_kg))
    return arrays


class RecursiveWorkerV118:
    """Lifecycle owner attached to one prepared v118 CPU/GPU transaction."""

    def __init__(self, prepared: RecursivePreparedV118 | Mapping,
                 owner, parent, *, worker_steps=0):
        self.prepared = (prepared if isinstance(prepared, RecursivePreparedV118)
                         else RecursivePreparedV118.load(prepared))
        self.owner = owner; self.parent = parent
        self.worker_steps = int(worker_steps)
        self.receipts: list[str] = []
        self.frames: list[dict] = []
        self._validate_live()

    def _validate_live(self):
        mode, route = _owner_route(self.owner)
        gpu = _backend_is_gpu(self.owner.backend)
        if (mode != self.prepared.bank_mode or route != self.prepared.route or
                gpu != (self.prepared.backend == GPU_BACKEND) or
                self.prepared.automatic_backend_fallback is not False or
                int(self.owner.updates) < 0 or
                not np.isfinite(float(self.owner.time_s))):
            raise RecursiveWorkerV118Error(
                "Live recursive owner differs from prepared backend/route")
        if (float(getattr(self.parent, "time_s", np.nan)) !=
                float(self.owner.time_s) or
                int(getattr(self.parent, "accepted_macro_updates", -1)) !=
                int(self.owner.updates)):
            raise RecursiveWorkerV118Error(
                "Accepted hydraulic parent and sediment owner clocks differ")

    def checkpoint_document(self):
        self._validate_live()
        checkpoint = self.owner.checkpoint()
        body = {
            "schema": CHECKPOINT_SCHEMA,
            "prepared_identity": self.prepared.identity,
            "backend": self.prepared.backend,
            "route": self.prepared.route,
            "worker_steps": self.worker_steps,
            "time_s": float(self.owner.time_s),
            "owner_updates": int(self.owner.updates),
            "owner_identity": self.owner.identity,
            "owner_checkpoint": checkpoint,
            "owner_checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "parent_state_sha256": getattr(self.parent, "state_sha256", None),
            "receipt_sha256": list(self.receipts),
            "frames": list(self.frames),
        }
        return {**body, "identity": _digest({
            key: value for key, value in body.items()
            if key != "owner_checkpoint"})}

    def publish_checkpoint(self, folder: str | Path) -> Path:
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        document = self.checkpoint_document()
        name = (f"recursive_checkpoint_v118_{self.worker_steps:08d}_"
                f"{secrets.token_hex(6)}.npz")
        path = folder / name
        archive_sha = _write_archive(path, document)
        pointer = {
            "schema": CHECKPOINT_SCHEMA, "archive": name,
            "archive_sha256": archive_sha,
            "prepared_identity": self.prepared.identity,
            "worker_steps": self.worker_steps,
            "time_s": float(self.owner.time_s),
            "identity": document["identity"],
        }
        _atomic_bytes(folder / LATEST_CHECKPOINT, _canonical(pointer))
        return path

    def _validate_latest(self, folder: Path) -> None:
        path = folder / LATEST_CHECKPOINT
        try:
            pointer = json.loads(path.read_text(encoding="ascii"))
            archive = folder / str(pointer["archive"])
        except (OSError, KeyError, TypeError, ValueError,
                json.JSONDecodeError) as error:
            raise RecursiveWorkerV118Error(
                "Latest recursive checkpoint pointer is invalid") from error
        if (pointer.get("schema") != CHECKPOINT_SCHEMA or
                pointer.get("prepared_identity") != self.prepared.identity or
                pointer.get("worker_steps") != self.worker_steps or
                not archive.is_file() or
                _file_sha(archive) != pointer.get("archive_sha256")):
            raise RecursiveWorkerV118Error(
                "Existing result folder belongs to another recursive state")
        document = _read_archive(archive)
        if (document.get("identity") != pointer.get("identity") or
                document.get("owner_checkpoint_sha256") !=
                    self.owner.checkpoint()["checkpoint_sha256"]):
            raise RecursiveWorkerV118Error(
                "Existing result checkpoint differs from the live owner")

    @classmethod
    def resume(cls, prepared, owner, parent, checkpoint: str | Path):
        prepared = (prepared if isinstance(prepared, RecursivePreparedV118)
                    else RecursivePreparedV118.load(prepared))
        path = Path(checkpoint)
        if path.name == LATEST_CHECKPOINT:
            pointer = json.loads(path.read_text(encoding="ascii"))
            archive = path.parent / str(pointer.get("archive"))
            if (pointer.get("schema") != CHECKPOINT_SCHEMA or
                    not archive.is_file() or
                    _file_sha(archive) != pointer.get("archive_sha256") or
                    pointer.get("prepared_identity") != prepared.identity):
                raise RecursiveWorkerV118Error(
                    "Latest recursive checkpoint pointer differs")
            path = archive
        document = _read_archive(path)
        excluded = {"owner_checkpoint"}
        seal = _digest({key: value for key, value in document.items()
                        if key not in excluded | {"identity"}})
        if (document.get("schema") != CHECKPOINT_SCHEMA or
                document.get("prepared_identity") != prepared.identity or
                document.get("backend") != prepared.backend or
                document.get("route") != prepared.route or
                document.get("identity") != seal or
                document.get("owner_identity") != owner.identity):
            raise RecursiveWorkerV118Error(
                "Recursive checkpoint envelope differs")
        old = owner.checkpoint()
        try:
            owner.restore(document["owner_checkpoint"])
            restored_parent = parent(owner) if callable(parent) else parent
            worker = cls(prepared, owner, restored_parent,
                         worker_steps=int(document["worker_steps"]))
            worker.receipts = list(document.get("receipt_sha256", ()))
            worker.frames = list(document.get("frames", ()))
            if (float(owner.time_s) != float(document["time_s"]) or
                    int(owner.updates) != int(document["owner_updates"])):
                raise RecursiveWorkerV118Error(
                    "Restored recursive checkpoint clock differs")
            return worker
        except BaseException:
            owner.restore(old)
            raise

    def _publish_frame(self, folder: Path, receipt) -> None:
        fields = _snapshot(self.owner, self.parent)
        metadata = {
            "schema": RESULT_SCHEMA, "prepared_identity": self.prepared.identity,
            "worker_steps": self.worker_steps, "time_s": float(self.owner.time_s),
            "owner_updates": int(self.owner.updates),
            "receipt_sha256": receipt.receipt_sha256,
            "backend": receipt.backend, "automatic_cpu_fallback": False,
        }
        path = folder / (f"recursive_frame_v118_{self.worker_steps:08d}_"
                         f"{secrets.token_hex(6)}.npz")
        archive_sha = _write_archive(path, {"metadata": metadata,
                                            "fields": fields})
        self.frames.append({**metadata, "file": path.name,
                            "sha256": archive_sha})

    def _manifest(self, status, error=None):
        return {
            "schema": RESULT_SCHEMA, "status": status,
            "prepared_identity": self.prepared.identity,
            "product_fingerprint": self.prepared.product_fingerprint,
            "backend": self.prepared.backend,
            "integrator": self.prepared.integrator,
            "route": self.prepared.route,
            "automatic_backend_fallback": False,
            "worker_steps": self.worker_steps,
            "time_s": float(self.owner.time_s),
            "owner_updates": int(self.owner.updates),
            "receipts": list(self.receipts), "frames": list(self.frames),
            "latest_checkpoint": LATEST_CHECKPOINT,
            "error": error,
        }

    def run(self, count: int, advance: Callable, output: str | Path, *,
            cancelled=False):
        if type(count) is not int or count < 0:
            raise RecursiveWorkerV118Error("Accepted macro count is invalid")
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        _atomic_bytes(output / PREPARED_SIDECAR,
                      _canonical(self.prepared.as_dict()))
        if not (output / LATEST_CHECKPOINT).is_file():
            self.publish_checkpoint(output)
        else:
            self._validate_latest(output)
        try:
            for _ in range(count):
                if _cancelled(cancelled):
                    raise InterruptedError("Recursive worker cancelled")
                accepted = self.owner.checkpoint()
                old_parent = self.parent
                old_steps = self.worker_steps
                old_receipts = len(self.receipts)
                old_frames = len(self.frames)
                before_updates = int(self.owner.updates)
                try:
                    receipt, state = advance(
                        self.owner, self.parent, self.worker_steps + 1,
                        cancelled)
                    joined = state.get("joined") if isinstance(state, Mapping) else None
                    if joined is None:
                        raise RecursiveWorkerV118Error(
                            "Numerical advance omitted the accepted joined state")
                    self.parent = joined
                    if _cancelled(cancelled):
                        raise InterruptedError("Recursive worker cancelled")
                    if (getattr(receipt, "automatic_cpu_fallback", None) is not False or
                            _backend_is_gpu(receipt.backend) !=
                            (self.prepared.backend == GPU_BACKEND) or
                            int(self.owner.updates) != before_updates + 1):
                        raise RecursiveWorkerV118Error(
                            "Accepted macro receipt/backend/update count differs")
                    self.worker_steps += 1
                    self.receipts.append(receipt.receipt_sha256)
                    self._validate_live()
                    self._publish_frame(output, receipt)
                    self.publish_checkpoint(output)
                except BaseException:
                    self.owner.restore(accepted)
                    self.parent = old_parent; self.worker_steps = old_steps
                    del self.receipts[old_receipts:]
                    del self.frames[old_frames:]
                    raise
            manifest = self._manifest("completed")
            _atomic_bytes(output / RUN_MANIFEST, _canonical(manifest))
            return manifest
        except InterruptedError as error:
            manifest = self._manifest("cancelled", str(error))
            _atomic_bytes(output / RUN_MANIFEST, _canonical(manifest))
            raise
        except BaseException as error:
            manifest = self._manifest("failed", str(error))
            _atomic_bytes(output / RUN_MANIFEST, _canonical(manifest))
            raise

    def install_regrid(self, new_owner, new_parent, receipt, output,
                       *, cancelled=False):
        """Atomically install an already validated private v117 rebuild."""
        if _cancelled(cancelled):
            raise InterruptedError("Recursive regrid cancelled")
        previous = self.owner, self.parent
        self.owner, self.parent = new_owner, new_parent
        try:
            self._validate_live()
            if (receipt.source_owner_identity != previous[0].identity or
                    receipt.target_owner_identity != new_owner.identity or
                    receipt.automatic_cpu_fallback is not False or
                    receipt.route != self.prepared.route):
                raise RecursiveWorkerV118Error(
                    "Recursive AMR receipt differs from installed owners")
            if _cancelled(cancelled):
                raise InterruptedError("Recursive regrid cancelled")
            return self.publish_checkpoint(output)
        except BaseException:
            self.owner, self.parent = previous
            raise


def validate_recursive_result_v118(folder: str | Path,
                                   prepared: RecursivePreparedV118 | Mapping):
    folder = Path(folder)
    prepared = (prepared if isinstance(prepared, RecursivePreparedV118)
                else RecursivePreparedV118.load(prepared))
    sidecar = RecursivePreparedV118.load(json.loads(
        (folder / PREPARED_SIDECAR).read_text(encoding="ascii")))
    manifest = json.loads((folder / RUN_MANIFEST).read_text(encoding="ascii"))
    if (sidecar != prepared or manifest.get("schema") != RESULT_SCHEMA or
            manifest.get("prepared_identity") != prepared.identity or
            manifest.get("backend") != prepared.backend or
            manifest.get("integrator") != prepared.integrator or
            manifest.get("automatic_backend_fallback") is not False or
            manifest.get("status") not in ("completed", "cancelled")):
        raise RecursiveWorkerV118Error("Recursive result identity differs")
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise RecursiveWorkerV118Error("Recursive result frame index differs")
    previous_step = -1
    for row in frames:
        path = folder / str(row.get("file"))
        step = row.get("worker_steps")
        if (type(step) is not int or step <= previous_step or
                not path.is_file() or
                _file_sha(path) != row.get("sha256")):
            raise RecursiveWorkerV118Error("Recursive result frame differs")
        previous_step = step
        frame = _read_archive(path)
        if frame.get("metadata", {}).get("prepared_identity") != prepared.identity:
            raise RecursiveWorkerV118Error(
                "Recursive result frame identity differs")
    return manifest


__all__ = [
    "CHECKPOINT_SCHEMA", "LATEST_CHECKPOINT", "PREPARED_SCHEMA",
    "PREPARED_SIDECAR", "RESULT_SCHEMA", "ROUTES", "RUN_MANIFEST", "SCHEMA",
    "RecursivePreparedV118", "RecursiveWorkerV118",
    "RecursiveWorkerV118Error", "validate_recursive_result_v118",
]
