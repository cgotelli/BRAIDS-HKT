"""Authenticate and index GPU AMR preparation for QGIS result display.

This is a host-only geometry/identity reader. It does not run a solver or
imply that the current experimental AMR route is a QGIS GPU release.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np
from braids_core.windows_paths import filesystem_path

from .gpu_frame_io import _geometry
from .gpu_prepared_io import read_gpu_prepared


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_TOPOLOGY = re.compile(r"topology_[0-9]{4}\Z")
_TREE_KEYS = frozenset({"parent", "node_level", "maximum_level",
                        "offsets", "area", "xy"})


def _sha256(path: Path) -> str:
    with filesystem_path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _artifact(root: Path, name: str) -> Path:
    root = filesystem_path(root)
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name or
            PurePosixPath(name).is_absolute() or
            PurePosixPath(name).as_posix() != name):
        raise ValueError("Invalid GPU AMR artifact path")
    parts = PurePosixPath(name).parts
    if (not 1 <= len(parts) <= 2 or any(part in ("", ".", "..") for part in parts) or
            (len(parts) == 2 and not _TOPOLOGY.fullmatch(parts[0]))):
        raise ValueError("GPU AMR artifact is outside a topology folder")
    path = root.joinpath(*parts)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("GPU AMR artifact is missing or linked: " + name)
    return path


def _tree(path: Path, ratio: int) -> dict[str, np.ndarray]:
    path = filesystem_path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != _TREE_KEYS:
                raise ValueError("Unsupported GPU AMR hierarchy schema")
            tree = {name: archive[name] for name in archive.files}
    except (OSError, EOFError, KeyError) as exc:
        raise ValueError("Cannot read GPU AMR hierarchy") from exc
    parent = tree["parent"]
    level = tree["node_level"]
    maximum = tree["maximum_level"]
    offsets = tree["offsets"]
    area = tree["area"]
    xy = tree["xy"]
    n = len(parent)
    if (ratio != 2 or n < 5 or
            any(value.dtype != np.int64 for value in (parent, level, maximum, offsets)) or
            parent.shape != (n,) or level.shape != (n,) or maximum.shape != (n,) or
            area.shape != (n,) or area.dtype != np.float64 or
            xy.shape != (n, 2) or xy.dtype != np.float64 or
            offsets.ndim != 1 or len(offsets) < 3 or offsets[0] != 0 or
            offsets[-1] != n or np.any(np.diff(offsets) <= 0) or
            np.any(parent < -1) or np.any(parent >= np.arange(n)) or
            not np.any(parent == -1) or
            np.any(level < 0) or np.any(maximum < level) or
            np.any(maximum >= len(offsets) - 1) or
            not np.isfinite(area).all() or np.any(area <= 0) or
            not np.isfinite(xy).all()):
        raise ValueError("Invalid GPU AMR hierarchy geometry or ancestry")
    expected_levels = np.searchsorted(offsets[1:], np.arange(n), side="right")
    if not np.array_equal(level, expected_levels):
        raise ValueError("GPU AMR hierarchy level offsets differ")
    # Validate all parent groups in linear time.  Scanning the full parent
    # array once per node made opening a large-reach AMR result quadratic.
    child_nodes = np.flatnonzero(parent >= 0)
    parents = parent[child_nodes]
    child_counts = np.bincount(parents, minlength=n)
    if np.any((child_counts != 0) & (child_counts != 4)):
        raise ValueError("GPU AMR hierarchy has partial child groups")
    if np.any(level[child_nodes] != level[parents] + 1):
        raise ValueError("GPU AMR children have incorrect levels")
    child_area = np.bincount(parents, weights=area[child_nodes], minlength=n)
    internal = child_counts == 4
    if not np.allclose(area[internal], child_area[internal],
                       rtol=1e-11, atol=1e-11):
        raise ValueError("GPU AMR children do not partition their parent")
    return tree


def _selected_nodes(path: Path, tree: dict[str, np.ndarray]) -> np.ndarray:
    path = filesystem_path(path)
    try:
        nodes = np.load(path, allow_pickle=False)
    except (OSError, EOFError, ValueError) as exc:
        raise ValueError("Cannot read GPU AMR selected nodes") from exc
    parent = tree["parent"]
    if (nodes.dtype != np.int64 or nodes.ndim != 1 or nodes.size < 1 or
            np.any(nodes < 0) or np.any(nodes >= len(parent)) or
            np.any(np.diff(nodes) <= 0)):
        raise ValueError("Invalid GPU AMR selected nodes")
    selected = set(map(int, nodes))
    for node in nodes:
        ancestor = int(parent[node])
        while ancestor >= 0:
            if ancestor in selected:
                raise ValueError("GPU AMR selected leaf overlaps its ancestor")
            ancestor = int(parent[ancestor])
    roots = np.flatnonzero(parent == -1)
    if not np.isclose(tree["area"][nodes].sum(), tree["area"][roots].sum(),
                      rtol=1e-11, atol=1e-11):
        raise ValueError("GPU AMR selected areas do not cover the domain")
    return nodes


def inspect_gpu_amr_prepared(folder: str | Path, *,
                             expected_manifest_sha256: str | None = None) -> dict:
    """Return authenticated selected topologies, global IDs and QGIS geometry.

    Numerical input is decoded lazily by :func:`read_gpu_amr_topology` so a
    result viewer need not load every operator merely to list mesh layouts.
    """
    root = filesystem_path(Path(folder).expanduser()).resolve()
    manifest_path = root / "amr_prepared.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("GPU AMR preparation manifest is missing or linked")
    manifest_sha = _sha256(manifest_path)
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise ValueError("GPU AMR preparation manifest differs from run identity")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or
            manifest.get("kind") != "braids-gpu-amr-prepared" or
            manifest.get("schema") not in (1, 2) or manifest.get("ratio") != 2 or
            manifest.get("CPU_numerical_fallback") is not False or
            manifest.get("solver_steps_performed_by_exporter") != 0):
        raise ValueError("Unsupported GPU AMR preparation manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("GPU AMR preparation has no artifact inventory")
    for name, digest in artifacts.items():
        if not isinstance(digest, str) or _SHA.fullmatch(digest) is None:
            raise ValueError("Invalid GPU AMR artifact identity")
        if _sha256(_artifact(root, name)) != digest:
            raise ValueError("GPU AMR artifact differs: " + name)
    if "tree.npz" not in artifacts:
        raise ValueError("GPU AMR hierarchy identity is missing")
    tree = _tree(root / "tree.npz", manifest["ratio"])
    records = manifest.get("topologies")
    if not isinstance(records, list) or not records:
        raise ValueError("GPU AMR preparation has no selected topology")
    topologies = {}
    for row in records:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("Invalid GPU AMR topology record")
        name = row["name"]
        if not name or name in topologies:
            raise ValueError("Duplicate GPU AMR topology")
        for key in ("node_ids", "geometry", "operator", "receipt"):
            if not isinstance(row.get(key), str) or row[key] not in artifacts:
                raise ValueError("GPU AMR topology lacks authenticated " + key)
        node_path = _artifact(root, row["node_ids"])
        geometry_path = _artifact(root, row["geometry"])
        receipt_path = _artifact(root, row["receipt"])
        topology_root = node_path.parent
        if any(path.parent != topology_root for path in
               (geometry_path, receipt_path, _artifact(root, row["operator"]))):
            raise ValueError("GPU AMR topology artifacts have different folders")
        sidecars = row.get("sidecars", {})
        if not isinstance(sidecars, dict):
            raise ValueError("Invalid GPU AMR topology sidecar inventory")
        for kind, relative in sidecars.items():
            if (not isinstance(kind, str) or not isinstance(relative, str) or
                    relative not in artifacts or
                    _artifact(root, relative).parent != topology_root):
                raise ValueError("GPU AMR topology has an unauthenticated sidecar")
        nodes = _selected_nodes(node_path, tree)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        geometry = _geometry(topology_root, receipt)
        if (geometry["cell_count"] != len(nodes) or
                not np.allclose(geometry["area_m2"], tree["area"][nodes],
                                rtol=1e-11, atol=1e-11) or
                not np.allclose(geometry["centroid_xy"], tree["xy"][nodes],
                                rtol=1e-11, atol=1e-11)):
            raise ValueError("GPU AMR selected nodes and geometry differ")
        topologies[name] = {"name": name, "folder": topology_root,
                            "node_ids": nodes, "geometry": geometry,
                            "receipt_sha256": artifacts[row["receipt"]],
                            "operator_sha256": artifacts[row["operator"]]}
    return {"root": root, "manifest_sha256": manifest_sha,
            "tree": tree, "topologies": topologies}


def read_gpu_amr_topology(index: dict, name: str) -> dict:
    """Decode one selected topology's initial numerical state on demand."""
    try:
        row = index["topologies"][name]
    except (KeyError, TypeError) as exc:
        raise ValueError("Unknown GPU AMR topology") from exc
    data = read_gpu_prepared(row["folder"],
                             receipt_sha256=row["receipt_sha256"])
    if int(data["nc"]) != len(row["node_ids"]):
        raise ValueError("GPU AMR topology state and selected nodes differ")
    return {**row, "initial_data": data}
