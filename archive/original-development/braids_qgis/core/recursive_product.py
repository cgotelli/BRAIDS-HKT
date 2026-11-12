"""Host-only QGIS binding for the v118 recursive bank worker.

No CUDA module is imported here.  Preparation resolves the already saved Q10
identity and writes a small authenticated sidecar before numerical allocation.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping

from braids_core.sediment.recursive_worker_v118 import (
    PREPARED_SIDECAR, RecursivePreparedV118, RecursiveWorkerV118Error,
    validate_recursive_result_v118,
)
from .sediment_product import validate_qgis_product


_TOE_MODES = {
    "noncohesive_transport_capacity": 2,
    "distributed_slope_transport": 3,
}


def resolve_recursive_product(project: Mapping) -> RecursivePreparedV118:
    """Resolve the qualified recursive bank bundle from an ordinary project."""
    if not isinstance(project, Mapping):
        raise RecursiveWorkerV118Error("BRAIDS project must be a mapping")
    profile = validate_qgis_product(project)
    if profile is None:
        raise RecursiveWorkerV118Error(
            "Recursive preparation requires a saved Q10 product identity")
    numerics = project.get("numerics")
    internal = project.get("internal_conditions")
    if not isinstance(numerics, Mapping) or not isinstance(internal, Mapping):
        raise RecursiveWorkerV118Error(
            "Recursive project numerics/internal conditions are missing")
    amr = numerics.get("amr", {})
    bank = internal.get("bank_failure", {})
    avalanche = internal.get("general_avalanche", {})
    bedform = internal.get("bedform_roughness", {})
    if (not isinstance(amr, Mapping) or amr.get("subcycling") is not True or
            not isinstance(bank, Mapping) or bank.get("enabled") is not True or
            not isinstance(avalanche, Mapping) or
            avalanche.get("enabled") is not True or
            not isinstance(bedform, Mapping) or
            bedform.get("enabled") is not True):
        raise RecursiveWorkerV118Error(
            "v118 requires recursive AMR, bank failure, parent avalanche and "
            "dynamic bedforms to be explicitly enabled")
    mode = _TOE_MODES.get(bank.get("toe_erosion_mode"))
    if mode is None:
        raise RecursiveWorkerV118Error(
            "Select surveyed physical-toe mode 2 or distributed-bank mode 3")
    return RecursivePreparedV118.create(profile, bank_mode=mode)


def write_recursive_prepared_product(run_root: str | Path,
                                     project: Mapping) -> Path:
    """Write the v118 sidecar once; an existing differing file is rejected."""
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    prepared = resolve_recursive_product(project)
    payload = (json.dumps(prepared.as_dict(), sort_keys=True,
                          separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False) + "\n").encode("ascii")
    path = root / PREPARED_SIDECAR
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RecursiveWorkerV118Error(
                "Existing recursive prepared identity differs")
        return path
    with path.open("xb") as stream:
        stream.write(payload)
    return path


def read_recursive_prepared_product(run_root: str | Path) -> RecursivePreparedV118:
    path = Path(run_root) / PREPARED_SIDECAR
    if not path.is_file():
        raise RecursiveWorkerV118Error("Recursive prepared sidecar is missing")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RecursiveWorkerV118Error(
            "Recursive prepared sidecar is invalid") from error
    return RecursivePreparedV118.load(value)


def validate_recursive_qgis_result(run_root: str | Path,
                                   prepared_root: str | Path):
    prepared = read_recursive_prepared_product(prepared_root)
    return validate_recursive_result_v118(run_root, prepared)


__all__ = [
    "read_recursive_prepared_product", "resolve_recursive_product",
    "validate_recursive_qgis_result", "write_recursive_prepared_product",
]
