"""Append-only, content-addressed cache for zero-step bank-AMR operators.

The producer is a trusted *initialization-only* host callback.  This store
never invokes a native timestep, CUDA, or a reference trajectory.  A complete
schema-2 :class:`PreparedAMR` and its global bank/toe lineage are validated in
an unpublished sibling directory before a single atomic directory rename.
Reopening a store rechecks every published byte against its immutable receipt.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Callable, Mapping, Sequence

import numpy as np
from braids_core.windows_paths import filesystem_path

from .amr_prepared import PreparedAMR, load_amr_prepared, _read_coupled


STORE_FILE = "store.json"
ENTRY_FILE = "entry.json"
PACKAGE = "package"
SCHEMA = 1
EVOLVING_SCHEMA = 2
FACE_IDENTITY = "face_identity"
_FACE_IDENTITY_FILES = ("amr_selected_identity.bin", "amr_tree_input.bin",
                        "selection_receipt.json")
_WINDOWS_TRANSIENT_RENAME_ERRORS = frozenset((5, 32, 33))
_PUBLISH_ATTEMPTS = 12
_GLOBAL_KEYS = {"segment_id", "bank_node", "toe_node",
                "midpoint_xy_m", "length_m"}


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        letter in "0123456789abcdef" for letter in value)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _ids(selected_ids: Sequence[int] | np.ndarray) -> np.ndarray:
    value = np.asarray(selected_ids)
    _require(value.ndim == 1 and value.size > 0 and
             np.issubdtype(value.dtype, np.integer),
             "AMR selected IDs must be a nonempty integer vector")
    ids = np.sort(value.astype(np.int64, copy=False))
    _require(np.all(ids >= 0) and len(np.unique(ids)) == len(ids),
             "AMR selected IDs must be unique nonnegative global nodes")
    return ids


def _lineage(rows: object, *, with_face: bool) -> list[dict]:
    _require(isinstance(rows, list) and len(rows) > 0,
             "Global bank/toe lineage must be a nonempty JSON list")
    normalized = []
    for row in rows:
        expected = _GLOBAL_KEYS | ({"local_face"} if with_face else set())
        _require(isinstance(row, dict) and set(row) == expected,
                 "Global bank/toe lineage row keys differ")
        item = {}
        for key in ("segment_id", "bank_node", "toe_node") + (
                ("local_face",) if with_face else ()):
            value = row[key]
            _require(type(value) is int and value >= 0,
                     "Global bank/toe lineage IDs must be nonnegative integers")
            item[key] = value
        point = row["midpoint_xy_m"]
        _require(isinstance(point, list) and len(point) == 2 and
                 all(type(x) in (int, float) and np.isfinite(x) for x in point),
                 "Global bank/toe midpoint differs")
        length = row["length_m"]
        _require(type(length) in (int, float) and np.isfinite(length) and length > 0,
                 "Global bank/toe length differs")
        item["midpoint_xy_m"] = [float(x) for x in point]
        item["length_m"] = float(length)
        normalized.append(item)
    normalized.sort(key=lambda row: row["segment_id"])
    _require(len({row["segment_id"] for row in normalized}) == len(normalized) and
             (not with_face or len({row["local_face"] for row in normalized}) ==
              len(normalized)), "Duplicate global toe segment or local face")
    return normalized


def _global_lineage(rows: list[dict]) -> list[dict]:
    return [{key: row[key] for key in _GLOBAL_KEYS} for row in rows]


def _package_files(package: Path) -> dict[str, str]:
    _require(package.is_dir() and not package.is_symlink(),
             "AMR operator package is missing or linked")
    files: dict[str, str] = {}
    for item in package.rglob("*"):
        _require(not item.is_symlink() and item.resolve().is_relative_to(package.resolve()),
                 "Linked or escaping AMR package member")
        if item.is_file():
            relative = item.relative_to(package).as_posix()
            files[relative] = _sha(item)
        else:
            _require(item.is_dir(), "Non-file AMR package member")
    manifest = package / "amr_prepared.json"
    _require(manifest.is_file(), "AMR package lacks its manifest")
    record = json.loads(manifest.read_text("utf-8"))
    _require(set(files) == set(record.get("artifacts", {})) | {manifest.name},
             "AMR package contains an unlisted or missing file")
    return dict(sorted(files.items()))


def _validate_operator(prepared: PreparedAMR, package: Path, ids: np.ndarray,
                       identities: Mapping[str, str], expected_lineage: list[dict],
                       rows: object, *, evolving_toe_variant: bool = False,
                       tree_input_sha256: str | None = None
                       ) -> tuple[PreparedAMR, list[dict], dict[str, str]]:
    _require(isinstance(prepared, PreparedAMR) and
             prepared.folder.resolve() == package.resolve(),
             "Producer must return the staged PreparedAMR package")
    # Reopen from bytes; do not trust an in-memory producer result.
    checked = load_amr_prepared(package)
    _require(checked.schema == 2 and len(checked.topologies) == 1 and
             np.array_equal(checked.topologies[0].node_ids, ids),
             "Operator is not one schema-2 selected-leaf package")
    _require(all(checked.source.get(key) == value for key, value in identities.items()),
             "AMR package source/hierarchy/base/runtime identity differs")
    _require(_sha(package / "runtime_build.json") ==
             identities["runtime_build_sha256"],
             "AMR operator native runtime build differs")
    if evolving_toe_variant:
        _require(checked.source.get("on_demand_initialization") is True,
                 "Evolving AMR operator is not a selected native initialization")
        lineage = _face_lineage(checked, filesystem_path(rows), tree_input_sha256)
        return checked, lineage, _package_files(package)
    _require(not any('toe_geometry_seed_only' in top.data for top in checked.topologies),
             'Geometry-only toe seed operator needs the evolving bank contract')
    lineage = _lineage(rows, with_face=True)
    _require(_global_lineage(lineage) == expected_lineage,
             "AMR operator changes immutable global toe lineage")
    top = checked.topologies[0]
    data = top.data
    active = np.flatnonzero(np.asarray(data["is_toe_face"]) != 0)
    _require(np.array_equal(np.sort([row["local_face"] for row in lineage]), active),
             "AMR operator toe-face set differs from lineage")
    for row in lineage:
        face = row["local_face"]
        _require(face < int(data["nf"]) and
                 int(top.node_ids[int(data["candidate_bank"][face])]) ==
                 row["bank_node"] and
                 int(top.node_ids[int(data["candidate_toe"][face])]) ==
                 row["toe_node"] and
                 np.array_equal(np.asarray(data["mid"])[:, face],
                                np.asarray(row["midpoint_xy_m"])) and
                 float(np.asarray(data["length"])[face]) == row["length_m"],
                 "AMR global toe segment does not match selected operator geometry")
    return checked, lineage, _package_files(package)


def _face_identity_files(folder: Path) -> dict[str, str]:
    _require(folder.is_dir() and not folder.is_symlink(),
             "Missing or linked AMR face identity folder")
    return {name: _sha(_identity_file(folder, name))
            for name in _FACE_IDENTITY_FILES}


def _identity_file(folder: Path, name: str) -> Path:
    path = folder / name
    _require(path.is_file() and not path.is_symlink() and
             path.resolve().is_relative_to(folder.resolve()),
             "Missing or linked AMR face identity file: " + name)
    return path


def _face_lineage(prepared: PreparedAMR, folder: Path,
                  tree_input_sha256: str | None) -> list[dict]:
    """Authenticate every native face, independently of the initial toe mask.

    The unchanged initializer writes one-based source-face IDs. Reproduce its
    *geometry-only* cover/use-face predicate, not a numerical operation, to
    reject missing, reordered, unrelated or duplicate source interfaces.
    The source owner side remains the operator owner side; normals are retained
    exactly from its separately authenticated native operator. No centroid-dot
    heuristic is imposed on non-convex cut cells.
    """
    files = _face_identity_files(folder)
    _require(_hex(tree_input_sha256) and
             files["amr_tree_input.bin"] == tree_input_sha256,
             "AMR original tree identity differs")
    selection = _read_coupled(folder / "amr_selected_identity.bin")
    tree = _read_coupled(folder / "amr_tree_input.bin")
    receipt = json.loads((folder / "selection_receipt.json").read_text("utf-8"))
    top = prepared.topologies[0]
    data = top.data
    ids = top.node_ids
    nf = int(data["nf"])
    _require(set(selection) == {"node_ids", "face_source_ids"} and
             np.asarray(selection["node_ids"]).dtype == np.dtype("int32") and
             np.array_equal(selection["node_ids"], ids),
             "AMR native selected global node identity differs")
    sources = np.asarray(selection["face_source_ids"])
    _require(sources.dtype == np.dtype("int32") and sources.shape == (nf,),
             "AMR native all-face identity dimensions differ")
    _require(isinstance(receipt, dict) and
             receipt.get("node_ids") == ids.tolist() and
             receipt.get("tree_input_sha256") == tree_input_sha256 and
             receipt.get("selection_identity_sha256") ==
             files["amr_selected_identity.bin"] and
             receipt.get("original_input_sha256") ==
             prepared.source["base_input_sha256"] and
             type(receipt.get("native_solver_steps")) is int and
             receipt["native_solver_steps"] == 0 and
             receipt.get("full_state_transfer_on_GPU") is True,
             "AMR native face selection receipt differs")
    for name in ("parent", "node_level", "maximum_level", "area"):
        _require(name in tree and
                 np.array_equal(tree[name], prepared.tree[name]),
                 "AMR original tree/operator mismatch: " + name)
    _require("xy" in tree and np.array_equal(tree["xy"].T, prepared.tree["xy"]),
             "AMR original tree/operator coordinates differ")
    parent = np.asarray(tree["parent"])
    nn = len(parent)
    source_owner = np.asarray(tree.get("face_owner"))
    source_other = np.asarray(tree.get("face_neighbour"))
    source_mid = np.asarray(tree.get("face_mid"))
    source_length = np.asarray(tree.get("face_length"))
    fn = source_owner.size
    _require(source_owner.dtype == np.dtype("int32") and
             source_owner.shape == (fn,) and fn > 0 and
             source_other.dtype == np.dtype("int32") and
             source_other.shape == (fn,) and
             np.all((source_owner >= 0) & (source_owner < nn)) and
             np.all((source_other >= -1) & (source_other < nn)) and
             np.all(source_other != source_owner) and
             source_mid.dtype == np.dtype("float64") and
             source_mid.shape == (2, fn) and np.isfinite(source_mid).all() and
             source_length.dtype == np.dtype("float64") and
             source_length.shape == (fn,) and np.all(source_length > 0),
             "AMR original all-face geometry is invalid")
    cover = np.full(nn, -1, dtype=np.int64)
    cover[ids] = ids
    # Parent IDs precede their children, as already checked by PreparedAMR.
    for node, ancestor in enumerate(parent):
        if ancestor >= 0 and cover[ancestor] >= 0:
            cover[node] = cover[ancestor]
    owner_cover = cover[source_owner]
    internal = source_other >= 0
    other_cover = np.full(fn, -1, dtype=np.int64)
    other_cover[internal] = cover[source_other[internal]]
    used = (owner_cover >= 0) & np.where(internal,
        (other_cover >= 0) & (owner_cover != other_cover) &
        ((source_owner == owner_cover) | (source_other == other_cover)),
        source_owner == owner_cover)
    _require(np.array_equal(sources.astype(np.int64), np.flatnonzero(used) + 1),
             "AMR all-face source IDs differ from selected hierarchy interfaces")
    source_index = sources.astype(np.int64) - 1
    owner = np.asarray(data["owner"])
    other = np.asarray(data["neighbour"])
    actual_other = np.full(nf, -1, dtype=np.int64)
    actual_other[other >= 0] = ids[other[other >= 0]]
    normal = np.asarray(data["normal"])
    _require(np.array_equal(ids[owner], owner_cover[source_index]) and
             np.array_equal(actual_other, other_cover[source_index]) and
             np.array_equal(data["mid"], source_mid[:, source_index]) and
             np.array_equal(data["length"], source_length[source_index]) and
             normal.dtype == np.dtype("float64") and normal.shape == (2, nf) and
             np.isfinite(normal).all() and
             np.allclose(np.sum(normal * normal, axis=0), 1., rtol=0., atol=1e-12),
             "AMR all-face lineage differs from operator geometry/orientation")
    return [dict(local_face=face, source_face_id=int(sources[face]),
        owner_node=int(ids[owner[face]]), neighbour_node=int(actual_other[face]),
        midpoint_xy_m=source_mid[:, source_index[face]].tolist(),
        length_m=float(source_length[source_index[face]]),
        normal_xy=normal[:, face].tolist()) for face in range(nf)]


def _stage_face_identity(source: object, pending: Path) -> Path:
    _require(isinstance(source, (str, Path)),
             "Evolving AMR producer must return its native identity folder")
    source = filesystem_path(source)
    files = _face_identity_files(source)
    destination = pending / FACE_IDENTITY
    destination.mkdir()
    for name, digest in files.items():
        shutil.copy2(_identity_file(source, name), destination / name)
        _require(_sha(destination / name) == digest,
                 "AMR face identity changed while being copied")
    return destination


def _cleanup_pending(path: Path, root: Path) -> None:
    _require(path.resolve().is_relative_to(root.resolve()) and
             path.name.startswith(".pending-") and not path.is_symlink(),
             "Refusing to clean an unexpected AMR staging directory")
    shutil.rmtree(path)


@dataclass(frozen=True)
class StoredOperator:
    selected_ids: np.ndarray
    request_key: str
    package_identity: str
    entry_sha256: str
    package: PreparedAMR
    lineage: tuple[dict, ...]
    schema: int = SCHEMA

    @property
    def face_lineage(self) -> tuple[dict, ...]:
        """Complete owner-oriented lineage, present only for schema 2."""
        _require(self.schema == EVOLVING_SCHEMA,
                 "Legacy AMR operator does not carry all-face lineage")
        return self.lineage


class AMROperatorStore:
    """Immutable input scope plus append-only selected-leaf operator objects."""

    def __init__(self, root: str | Path, *, hierarchy_sha256: str,
                 base_input_sha256: str, runtime_build_sha256: str,
                 global_toe_lineage: list[dict] | None = None,
                 evolving_toe_variant: bool = False,
                 tree_input_sha256: str | None = None):
        self.root = filesystem_path(root).resolve()
        values = dict(hierarchy_sha256=hierarchy_sha256,
                      base_input_sha256=base_input_sha256,
                      runtime_build_sha256=runtime_build_sha256)
        _require(all(_hex(value) for value in values.values()),
                 "AMR store identities must be SHA-256 digests")
        _require(type(evolving_toe_variant) is bool,
                 "AMR evolving-interface selection must be explicit boolean")
        self.evolving_toe_variant = evolving_toe_variant
        self.schema = EVOLVING_SCHEMA if evolving_toe_variant else SCHEMA
        self.tree_input_sha256 = tree_input_sha256
        if evolving_toe_variant:
            _require(global_toe_lineage is None and _hex(tree_input_sha256),
                     "Evolving AMR store needs original tree identity, not fixed toe lineage")
            self.lineage = []
        else:
            _require(tree_input_sha256 is None,
                     "Legacy AMR store cannot accept an evolving tree identity")
            self.lineage = _lineage(global_toe_lineage, with_face=False)
        self.identities = values
        base = dict(schema=self.schema, kind="braids-bank-amr-operator-store", **values)
        if evolving_toe_variant:
            base.update(tree_input_sha256=tree_input_sha256,
                        face_lineage_contract="native-all-faces-owner-oriented-v1")
        else:
            base["global_toe_lineage"] = self.lineage
        self.identity = _digest(base)
        self.manifest = dict(base, identity_sha256=self.identity)
        self._ensure_root()
        self._verify_all()

    def _ensure_root(self) -> None:
        _require(not self.root.is_symlink(), "Linked AMR operator store root")
        if self.root.exists():
            return
        self.root.parent.mkdir(parents=True, exist_ok=True)
        pending = filesystem_path(tempfile.mkdtemp(prefix=".amr-store-init-",
                                        dir=self.root.parent))
        try:
            (pending / "operators").mkdir()
            (pending / STORE_FILE).write_bytes(_canonical(self.manifest) + b"\n")
            try:
                os.rename(pending, self.root)
            except OSError:
                if not self.root.exists():
                    raise
        finally:
            if pending.exists():
                _require(pending.resolve().is_relative_to(self.root.parent.resolve()) and
                         pending.name.startswith(".amr-store-init-") and
                         not pending.is_symlink(),
                         "Refusing to clean unexpected AMR store initializer")
                shutil.rmtree(pending)

    def _verify_all(self) -> None:
        _require(self.root.is_dir() and not self.root.is_symlink() and
                 (self.root / "operators").is_dir() and
                 not (self.root / "operators").is_symlink(),
                 "AMR operator store directory differs")
        manifest = self.root / STORE_FILE
        _require(manifest.is_file() and not manifest.is_symlink() and
                 json.loads(manifest.read_text("utf-8")) == self.manifest,
                 "AMR operator store identity changed")
        seen: set[str] = set()
        for entry in (self.root / "operators").iterdir():
            _require(entry.is_dir() and not entry.is_symlink(),
                     "Unexpected AMR operator store member")
            stored = self._read_entry(entry)
            _require(stored.request_key not in seen,
                     "Multiple AMR operators for the same selected-leaf request")
            seen.add(stored.request_key)

    def _request(self, selected_ids: Sequence[int] | np.ndarray) -> tuple[np.ndarray, str]:
        ids = _ids(selected_ids)
        key = _digest(dict(store_identity_sha256=self.identity,
                           selected_node_ids=ids.tolist()))
        return ids, key

    def _matching(self, request_key: str) -> list[Path]:
        # The directory name is the full package-content digest.  A separate
        # full request digest in its receipt makes lookup deterministic while
        # keeping Windows paths below MAX_PATH for nested schema-2 sidecars.
        result = [entry for entry in (self.root / "operators").iterdir()
                  if self._read_entry(entry).request_key == request_key]
        _require(len(result) <= 1, "Multiple AMR operator contents for one leaf set")
        return result

    def _read_entry(self, folder: Path) -> StoredOperator:
        _require(folder.parent == self.root / "operators" and folder.is_dir() and
                 not folder.is_symlink(), "Invalid AMR operator object path")
        receipt_path = folder / ENTRY_FILE
        _require(receipt_path.is_file() and not receipt_path.is_symlink(),
                 "Published AMR operator lacks entry receipt")
        receipt = json.loads(receipt_path.read_text("utf-8"))
        expected_keys = {"schema", "kind", "store_identity_sha256",
                         "request_key", "selected_node_ids", "package_identity",
                         "package_files", "lineage"}
        if self.evolving_toe_variant:
            expected_keys |= {"face_identity_files", "object_identity"}
        _require(set(receipt) == expected_keys and receipt["schema"] == self.schema and
                 receipt["kind"] == "zero-step-bank-amr-operator" and
                 receipt["store_identity_sha256"] == self.identity and
                 isinstance(receipt["selected_node_ids"], list),
                 "Published AMR operator receipt differs")
        ids, key = self._request(receipt["selected_node_ids"])
        _require(receipt["selected_node_ids"] == ids.tolist() and
                 key == receipt["request_key"] and
                 _hex(receipt["package_identity"]),
                 "AMR operator content address or selected IDs differ")
        expected_members = {PACKAGE, ENTRY_FILE}
        if self.evolving_toe_variant:
            expected_members.add(FACE_IDENTITY)
            face_folder = folder / FACE_IDENTITY
            face_files = _face_identity_files(face_folder)
            _require({child.name for child in face_folder.iterdir()} ==
                     set(_FACE_IDENTITY_FILES) and
                     face_files == receipt["face_identity_files"],
                     "AMR face identity artifact digest/inventory differs")
            object_identity = _digest(dict(package_identity=receipt["package_identity"],
                face_identity_files=receipt["face_identity_files"]))
            _require(receipt["object_identity"] == object_identity and
                     folder.name == object_identity,
                     "AMR evolving operator content address differs")
        else:
            _require(folder.name == receipt["package_identity"],
                     "AMR operator content address differs")
        _require({child.name for child in folder.iterdir()} == expected_members,
                 "Published AMR operator contains unlisted members")
        package = folder / PACKAGE
        _require(receipt["package_files"] == _package_files(package) and
                 receipt["package_identity"] == _sha(package / "amr_prepared.json"),
                 "Published AMR operator file digest differs")
        checked = load_amr_prepared(package)
        checked, lineage, _ = _validate_operator(
            checked, package, ids, self.identities, self.lineage,
            face_folder if self.evolving_toe_variant else receipt["lineage"],
            evolving_toe_variant=self.evolving_toe_variant,
            tree_input_sha256=self.tree_input_sha256)
        _require(_canonical(lineage) == _canonical(receipt["lineage"]),
                 "Published AMR face lineage differs from authenticated operator")
        return StoredOperator(ids, key, checked.identity, _sha(receipt_path),
                              checked, tuple(lineage), self.schema)

    def get(self, selected_ids: Sequence[int] | np.ndarray) -> StoredOperator | None:
        """Return a fully reauthenticated object; never trust an index alone."""
        _, key = self._request(selected_ids)
        matches = self._matching(key)
        return self._read_entry(matches[0]) if matches else None

    def get_or_create(self, selected_ids: Sequence[int] | np.ndarray,
                      producer: Callable[[Path, np.ndarray],
                                         tuple[PreparedAMR, object]],
                      *, cancelled: Callable[[], bool] = lambda: False) -> StoredOperator:
        """Publish only a complete, authenticated package in one rename.

        ``producer(staging_package_dir, sorted_ids)`` must return its validated
        single-topology schema-2 ``PreparedAMR`` and JSON-safe lineage rows
        for the legacy contract. An evolving store instead requires the native
        output folder containing the three unchanged face-identity artifacts.
        The producer owns initialization, not any CPU numerical evolution.
        """
        ids, key = self._request(selected_ids)
        existing = self.get(ids)
        if existing is not None:
            return existing
        _require(callable(producer) and callable(cancelled),
                 "AMR producer/cancellation callbacks must be callable")
        if cancelled():
            raise InterruptedError("AMR operator construction cancelled")
        pending = filesystem_path(tempfile.mkdtemp(prefix=".pending-", dir=self.root))
        try:
            package = pending / PACKAGE
            product = producer(package, ids.copy())
            _require(isinstance(product, tuple) and len(product) == 2,
                     "AMR producer must return (PreparedAMR, lineage)")
            if cancelled():
                raise InterruptedError("AMR operator construction cancelled")
            rows_source = (_stage_face_identity(product[1], pending)
                           if self.evolving_toe_variant else product[1])
            prepared, rows, files = _validate_operator(
                product[0], package, ids, self.identities, self.lineage, rows_source,
                evolving_toe_variant=self.evolving_toe_variant,
                tree_input_sha256=self.tree_input_sha256)
            receipt = dict(schema=self.schema, kind="zero-step-bank-amr-operator",
                store_identity_sha256=self.identity, request_key=key,
                selected_node_ids=ids.tolist(), package_identity=prepared.identity,
                package_files=files, lineage=rows)
            object_identity = prepared.identity
            if self.evolving_toe_variant:
                receipt["face_identity_files"] = _face_identity_files(rows_source)
                object_identity = _digest(dict(package_identity=prepared.identity,
                    face_identity_files=receipt["face_identity_files"]))
                receipt["object_identity"] = object_identity
            (pending / ENTRY_FILE).write_bytes(_canonical(receipt) + b"\n")
            expected_members = {PACKAGE, ENTRY_FILE} | (
                {FACE_IDENTITY} if self.evolving_toe_variant else set())
            _require({child.name for child in pending.iterdir()} == expected_members,
                     "Unlisted AMR producer staging member")
            if cancelled():
                raise InterruptedError("AMR operator construction cancelled")
            target = self.root / "operators" / object_identity
            prior = self._matching(key)
            if prior:
                _require(prior[0] == target,
                         "Conflicting AMR operator content for the same leaf set")
                return self._read_entry(prior[0])
            for attempt in range(_PUBLISH_ATTEMPTS):
                if cancelled():
                    raise InterruptedError("AMR operator construction cancelled")
                try:
                    os.rename(pending, target)
                    break
                except OSError as error:
                    # Another process may have completed the same request
                    # between lookup and rename.  The existing object must
                    # reauthenticate and have *identical* content identity.
                    prior = self._matching(key)
                    if prior:
                        _require(prior[0] == target,
                                 "Conflicting concurrent AMR operator publication")
                        return self._read_entry(prior[0])
                    _require(not target.exists(),
                             "Conflicting AMR operator content target already exists")
                    if (os.name != "nt" or
                            getattr(error, "winerror", None) not in
                            _WINDOWS_TRANSIENT_RENAME_ERRORS or
                            attempt + 1 == _PUBLISH_ATTEMPTS):
                        raise
                    # WinError 5/32/33 can be a short-lived handle/scanner
                    # lock on the staged directory.  No published target is
                    # replaced, and the total delay is bounded (<5 seconds).
                    time.sleep(min(.05 * (attempt + 1), .5))
            return self._read_entry(target)
        finally:
            if pending.exists():
                _cleanup_pending(pending, self.root)
