"""Portable, current-format river projects with immutable parameter history.

The project document remains the numerical contract. Folder metadata holds UI
preferences and provenance separately; it never adds defaults or migrates an old
project. Prepared inputs, results and authenticated checkpoints are left intact.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
import uuid

from .river_project import canonical, identity, read_json, validate_project
from .windows_paths import filesystem_path

SCHEMA = "braids.river-folder.v1"
PROJECT_FILE = "braids_project.json"
MANIFEST_FILE = "BRAIDS_FOLDER.json"
REVISION_SCHEMA = "braids.river-folder-revision.v1"
EVENT_SCHEMA = "braids.river-folder-event.v1"
DIRECTORIES = ("inputs", "prepared", "runs", "history")
_SESSION_PATHS = ("prepared", "output", "restart", "log")
_UNSET = object()
_PENDING = ".braids-folder-publication.json"
_LOCK = ".braids-folder.lock"
_GUARD = ".braids-folder.guard"
_PUBLICATION_SCHEMA = "braids.river-folder-publication.v1"


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _bytes(value):
    return (canonical(value) + "\n").encode("utf-8")


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member(root, relative):
    """Reject lexical and linked escapes, including Windows paths on any OS."""
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or PureWindowsPath(relative).drive or PurePosixPath(relative).is_absolute()
            or any(p in ("", ".", "..") for p in relative.split("/"))):
        raise ValueError("Project folder path must be a contained relative path: " + str(relative))
    path = root.joinpath(*PurePosixPath(relative).parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Project folder path escapes through a link: " + relative)
    # Even an internal link makes the immutable inventory ambiguous after a move.
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
            raise ValueError("Linked project folder members are not supported: " + relative)
    return path


def _atomic_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".braids-write-", dir=path.parent)
    temporary = filesystem_path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path, value):
    _atomic_bytes(path, _bytes(value))


@contextmanager
def _guard(root):
    """An OS-owned guard is released even when a process terminates abruptly."""
    path = _member(root, _GUARD)
    with path.open("a+b") as stream:
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Project folder is being saved by another session") from exc
        try:
            if path.stat().st_size == 0:
                stream.write(b"\0")
                stream.flush()
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def _lock(root):
    """Serialize publication and recover only known interrupted transactions."""
    with _guard(root):
        _recover_locked(root)
        with _writer_marker(root) as marker:
            yield marker


@contextmanager
def _writer_marker(root):
    path = _member(root, _LOCK)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError("Project folder is being saved by another session. If an interrupted "
                         "application left .braids-folder.lock, close it before removing that lock.") from exc
    try:
        with os.fdopen(descriptor, "w") as stream:
            marker = canonical(dict(pid=os.getpid(), token=uuid.uuid4().hex))
            stream.write(marker)
        yield marker
    finally:
        path.unlink(missing_ok=True)


def _session(root, value):
    if not isinstance(value, dict):
        raise ValueError("Project session must be a JSON object")
    result = read_json(canonical(value))
    for key in _SESSION_PATHS:
        if key in result and result[key] != "":
            _member(root, result[key])
    if "model_layers" in result:
        if not isinstance(result["model_layers"], dict):
            raise ValueError("Saved model layer selections must be an object")
        for layer in result["model_layers"].values():
            if not isinstance(layer, dict) or "path" not in layer:
                raise ValueError("Saved model layers require contained file paths")
            _member(root, layer["path"])
    return result


def _check_cancel(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError("Project folder save cancelled")


def _raster_files(source):
    """Use GDAL's actual file inventory, preserving same-directory sidecars."""
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".vrt":
        raise ValueError("Export the DEM virtual raster to a self-contained GeoTIFF before saving a project folder")
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise ValueError("GDAL is required to identify all DEM files before saving a project folder") from exc
    dataset = gdal.Open(os.fspath(source), gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError("Cannot open the DEM raster: " + str(source))
    try:
        if dataset.GetDriver().ShortName == "VRT":
            raise ValueError("Export the DEM virtual raster to a self-contained GeoTIFF before saving a project folder")
        paths = {filesystem_path(p).resolve() for p in (dataset.GetFileList() or ())}
    finally:
        dataset = None
    paths.add(source)
    # Include auxiliary files even when the driver has not opened their content.
    candidates = [source.with_name(source.name + suffix) for suffix in (".aux.xml", ".ovr", ".msk")]
    candidates += [source.with_suffix(suffix) for suffix in (".tfw", ".tifw", ".wld", ".prj", ".hdr", ".ige")]
    paths.update(p.resolve() for p in candidates if p.is_file())
    for path in paths:
        if path.parent != source.parent or not path.is_file():
            raise ValueError("DEM depends on files outside its directory; export a self-contained GeoTIFF: " + str(path))
    names = [p.name.casefold() for p in paths]
    if len(set(names)) != len(names):
        raise ValueError("DEM filenames collide on a case-insensitive filesystem")
    return sorted(paths, key=lambda p: p.name)


def _copy_raster(source, staging, cancelled):
    files = _raster_files(source)
    copied = staging / "dem-copy"
    copied.mkdir()
    rows = []
    for path in files:
        _check_cancel(cancelled)
        before = _sha(path)
        target = copied / path.name
        with path.open("rb") as src, target.open("xb") as dst:
            while True:
                _check_cancel(cancelled)
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        if _sha(target) != before or _sha(path) != before:
            raise ValueError("DEM input changed while saving: " + str(path))
        rows.append(dict(name=path.name, sha256=before, size=target.stat().st_size))
    if files != _raster_files(source):
        raise ValueError("DEM file inventory changed while saving")
    if any(_sha(path) != row["sha256"] for path, row in zip(files, rows)):
        raise ValueError("DEM input changed while saving its dependent files")
    bundle = identity(rows)
    relative = "inputs/" + bundle
    assets = [dict(path=relative + "/" + row["name"], sha256=row["sha256"], size=row["size"]) for row in rows]
    return copied, relative, relative + "/" + source.name, assets


def _validate_assets(root, assets):
    if not isinstance(assets, list) or not assets:
        raise ValueError("Project folder has no DEM inventory")
    seen = set()
    for row in assets:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "size"}:
            raise ValueError("Project folder asset inventory differs")
        path = _member(root, row["path"])
        if not row["path"].startswith("inputs/") or row["path"].casefold() in seen:
            raise ValueError("Project folder asset path differs")
        seen.add(row["path"].casefold())
        if not path.is_file() or path.stat().st_size != row["size"] or _sha(path) != row["sha256"]:
            raise ValueError("Missing or changed project input: " + row["path"])


def _read_record(path, schema):
    record = read_json(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("schema") != schema:
        raise ValueError("Unsupported project history format: " + path.name)
    if path.stem.rsplit("_", 1)[-1] != identity(record):
        raise ValueError("Project history record changed: " + path.name)
    return record


def _validate_heads(root, manifest_payload, project_payload, *, assets=True):
    for name in (*DIRECTORIES, PROJECT_FILE, MANIFEST_FILE):
        _member(root, name)
    manifest = read_json(manifest_payload.decode("utf-8"))
    expected = {"schema", "folder_id", "created_at", "saved_at", "project_file", "project_id",
                "project_sha256", "revision", "assets", "session", "geometry_review"}
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA or set(manifest) != expected:
        raise ValueError("Only the current BRAIDS project-folder format can open; old folders are not migrated")
    if manifest["project_file"] != PROJECT_FILE:
        raise ValueError("Project folder document name differs")
    if hashlib.sha256(project_payload).hexdigest() != manifest["project_sha256"]:
        raise ValueError("Project document changed outside its saved revision")
    project = validate_project(read_json(project_payload.decode("utf-8")))
    if identity(project) != manifest["project_id"]:
        raise ValueError("Project identity differs from its folder manifest")
    revision_path = _member(root, manifest["revision"])
    if not manifest["revision"].startswith("history/revisions/"):
        raise ValueError("Project revision is outside history")
    revision = _read_record(revision_path, REVISION_SCHEMA)
    if (revision["project"] != project or revision["assets"] != manifest["assets"]
            or revision["geometry_review"] != manifest["geometry_review"]):
        raise ValueError("Project folder differs from its immutable revision")
    _session(root, manifest["session"])
    dem = project["terrain"]["dem_path"]
    _member(root, dem)
    if dem not in [row["path"] for row in manifest["assets"]]:
        raise ValueError("Project DEM is not an inventoried input")
    if assets:
        _validate_assets(root, manifest["assets"])
    return manifest, project


def _recover_locked(root):
    """Recover a hash-bound head transition while holding the OS guard.

    The guard proves no participating writer is live. An unrelated lock or any
    head bytes outside the journal's exact before/after states fail closed.
    """
    pending = _member(root, _PENDING)
    if not pending.exists():
        return
    envelope = read_json(pending.read_text(encoding="utf-8"))
    if (not isinstance(envelope, dict) or set(envelope) != {"record", "sha256"}
            or identity(envelope["record"]) != envelope["sha256"]):
        raise ValueError("Interrupted project publication journal changed")
    record = envelope["record"]
    if (not isinstance(record, dict) or set(record) != {"schema", "writer", "before", "after"}
            or record["schema"] != _PUBLICATION_SCHEMA or not isinstance(record["writer"], str)):
        raise ValueError("Unsupported interrupted project publication")
    lock = _member(root, _LOCK)
    if lock.exists() and lock.read_text(encoding="utf-8") != record["writer"]:
        raise ValueError("Project publication has an unknown writer lock; no files were recovered")
    states = {}
    for state in ("before", "after"):
        row = record[state]
        if (not isinstance(row, dict) or set(row) != {"project", "manifest"}
                or not all(isinstance(value, str) for value in row.values())):
            raise ValueError("Interrupted project publication heads differ")
        project_payload, manifest_payload = row["project"].encode("utf-8"), row["manifest"].encode("utf-8")
        manifest, project = _validate_heads(root, manifest_payload, project_payload)
        states[state] = (manifest, project, project_payload, manifest_payload)
    before, after = states["before"], states["after"]
    after_revision = _read_record(_member(root, after[0]["revision"]), REVISION_SCHEMA)
    if (before[0]["folder_id"] != after[0]["folder_id"]
            or before[0]["created_at"] != after[0]["created_at"]
            or after_revision["parent"] != before[0]["revision"]):
        raise ValueError("Interrupted project publication is not a consecutive revision")
    project_path, manifest_path = _member(root, PROJECT_FILE), _member(root, MANIFEST_FILE)
    current = (project_path.read_bytes(), manifest_path.read_bytes())
    if current[0] not in (before[2], after[2]) or current[1] not in (before[3], after[3]):
        raise ValueError("Project heads differ from the interrupted publication; no files were recovered")
    # Both after-heads mean the save committed. Any exact mixed state restores
    # the earlier saved revision; the uncommitted immutable revision is retained.
    if current != (after[2], after[3]):
        _atomic_bytes(project_path, before[2])
        _atomic_bytes(manifest_path, before[3])
    lock.unlink(missing_ok=True)
    pending.unlink()


def _recover(root):
    if _member(root, _PENDING).exists():
        with _guard(root):
            _recover_locked(root)


def _read_folder(root, *, assets=True):
    _recover(root)
    return _validate_heads(root, _member(root, MANIFEST_FILE).read_bytes(),
                           _member(root, PROJECT_FILE).read_bytes(), assets=assets)


def folder_info(root):
    """Read verified folder metadata, including current UI session selections."""
    return _read_folder(filesystem_path(root).resolve())[0]


def load_folder(root):
    """Return a fresh runtime/editor document resolved against the moved folder."""
    root = filesystem_path(root).resolve()
    _, project = _read_folder(root)
    project["terrain"]["dem_path"] = str(_member(root, project["terrain"]["dem_path"]))
    return validate_project(project)


def _revision_chain(root, manifest):
    seen = set()
    relative = manifest["revision"]
    while relative is not None:
        if (not isinstance(relative, str) or relative in seen
                or not relative.startswith("history/revisions/")):
            raise ValueError("Project revision history is cyclic or outside its directory")
        seen.add(relative)
        row = _read_record(_member(root, relative), REVISION_SCHEMA)
        yield relative, row
        relative = row["parent"]


def revision_info(root, revision):
    """Read an authenticated committed revision and verify its own input assets."""
    root = filesystem_path(root).resolve()
    _member(root, revision)
    manifest, _ = _read_folder(root, assets=False)
    chain = dict(_revision_chain(root, manifest))
    if revision not in chain:
        raise ValueError("Only a committed project revision can be opened")
    row = chain[revision]
    project = validate_project(row["project"])
    if row["project_id"] != identity(project):
        raise ValueError("Saved revision project identity differs")
    _session(root, row["session"])
    _validate_assets(root, row["assets"])
    dem = project["terrain"]["dem_path"]
    _member(root, dem)
    if dem not in [asset["path"] for asset in row["assets"]]:
        raise ValueError("Saved revision DEM is not an inventoried input")
    return deepcopy(row)


def load_revision(root, revision):
    """Resolve a committed revision for the editor without changing any files."""
    root = filesystem_path(root).resolve()
    project = revision_info(root, revision)["project"]
    project["terrain"]["dem_path"] = str(_member(root, project["terrain"]["dem_path"]))
    return validate_project(project)


def _record_path(directory, record):
    timestamp = record["timestamp"].replace("-", "").replace(":", "").replace(".", "")
    return directory / (timestamp + "_" + identity(record) + ".json")


def save_folder(root, document, *, source_base=None, session=None, geometry_review=_UNSET, cancelled=None):
    """Copy DEM assets and atomically publish a current project and revision.

    ``source_base`` resolves a relative input DEM. Cancellation and copy errors
    leave the last committed project untouched. Existing results are never moved,
    replaced or deleted. New folders must be empty or contain only the current
    UI's model_layers geometry drafts; arbitrary/old workspaces are not adopted.
    The returned document uses portable relative paths.
    """
    project = validate_project(document)
    root = filesystem_path(root).resolve()
    source = Path(project["terrain"]["dem_path"])
    if not source.is_absolute():
        source = filesystem_path(source_base if source_base is not None else root) / source
    source = filesystem_path(source).resolve()
    chosen_session = None if session is None else _session(root, session)
    review = None if geometry_review is _UNSET else read_json(canonical(geometry_review))
    if review is not None and not isinstance(review, dict):
        raise ValueError("Geometry review must be a JSON object or null")
    if root.exists() and not root.is_dir():
        raise ValueError("Project folder is not a directory")
    existing = (root / MANIFEST_FILE).is_file()
    previous, previous_project = _read_folder(root) if existing else (None, None)
    if root.exists() and not existing and any(not _draft_member(p) for p in root.iterdir()):
        raise ValueError("Choose an empty folder, a folder containing only current model_layers drafts, "
                         "or an existing current BRAIDS project folder")
    _check_cancel(cancelled)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".braids-save-", dir=root.parent) as temporary:
        staging = filesystem_path(temporary)
        local_dem = None if previous is None else _member(root, previous_project["terrain"]["dem_path"])
        if (source == local_dem and set(_raster_files(source)) ==
                {_member(root, row["path"]) for row in previous["assets"]}):
            # A reopened project's DEM is already an immutable, verified asset.
            # Reuse it without copying gigabytes again on Save/Prepare/Run.
            copied = None
            dem_relative = previous_project["terrain"]["dem_path"]
            bundle_relative = PurePosixPath(dem_relative).parent.as_posix()
            assets = deepcopy(previous["assets"])
        else:
            copied, bundle_relative, dem_relative, assets = _copy_raster(source, staging, cancelled)
        project["terrain"]["dem_path"] = dem_relative
        validate_project(project)
        timestamp = _now()
        selected_session = chosen_session if chosen_session is not None else (previous["session"] if previous else {})
        if geometry_review is _UNSET and previous is not None:
            review = previous["geometry_review"]
        revision = dict(schema=REVISION_SCHEMA, timestamp=timestamp, kind="saved",
                        parent=None if previous is None else previous["revision"], project=project,
                        project_id=identity(project), assets=assets, session=selected_session, geometry_review=review)
        revision_relative = _record_path(Path("history/revisions"), revision).as_posix()
        manifest = dict(schema=SCHEMA, folder_id=previous["folder_id"] if previous else uuid.uuid4().hex,
                        created_at=previous["created_at"] if previous else timestamp, saved_at=timestamp,
                        project_file=PROJECT_FILE, project_id=identity(project),
                        project_sha256=hashlib.sha256(_bytes(project)).hexdigest(),
                        revision=revision_relative, assets=assets, session=selected_session, geometry_review=review)
        _check_cancel(cancelled)
        if not existing:
            # Stage the complete initial folder before publishing any member.
            ready = staging / "ready"
            ready.mkdir()
            for name in DIRECTORIES:
                (ready / name).mkdir()
            destination = _member(ready, bundle_relative)
            os.replace(copied, destination)
            _atomic_json(_member(ready, revision_relative), revision)
            _atomic_json(ready / PROJECT_FILE, project)
            _atomic_json(ready / MANIFEST_FILE, manifest)
            _check_cancel(cancelled)
            if not root.exists():
                os.rename(ready, root)
            else:
                # The UI may already have saved geometry drafts in model_layers.
                # Restore all newly published members on a handled I/O failure,
                # leaving those original drafts untouched.
                with _lock(root):
                    if any(p.name != _LOCK and not _draft_member(p) for p in root.iterdir()):
                        raise ValueError("Project folder changed while saving; reopen it and retry")
                    published = []
                    try:
                        for name in (*DIRECTORIES, PROJECT_FILE, MANIFEST_FILE):
                            os.rename(ready / name, _member(root, name))
                            published.append(name)
                    except BaseException:
                        for name in reversed(published):
                            os.rename(_member(root, name), ready / name)
                        raise
            return deepcopy(project)
        with _lock(root) as writer:
            # Detect an intervening save before publishing a stale parent revision.
            current = _read_folder(root)[0] if existing else None
            if current != previous:
                raise ValueError("Project folder changed while saving; reopen it and retry")
            for name in DIRECTORIES:
                _member(root, name).mkdir(exist_ok=True)
            destination = _member(root, bundle_relative)
            if destination.exists():
                _validate_assets(root, assets)
            else:
                os.rename(copied, destination)
            revision_path = _member(root, revision_relative)
            _atomic_json(revision_path, revision)
            old_project = (root / PROJECT_FILE).read_bytes() if existing else None
            old_manifest = (root / MANIFEST_FILE).read_bytes() if existing else None
            transition = dict(schema=_PUBLICATION_SCHEMA, writer=writer,
                before=dict(project=old_project.decode("utf-8"), manifest=old_manifest.decode("utf-8")),
                after=dict(project=_bytes(project).decode("utf-8"), manifest=_bytes(manifest).decode("utf-8")))
            _atomic_json(_member(root, _PENDING), dict(record=transition, sha256=identity(transition)))
            try:
                _atomic_json(root / PROJECT_FILE, project)
                _atomic_json(root / MANIFEST_FILE, manifest)
            except BaseException:
                # A handled I/O failure restores both old heads. Published immutable
                # inputs/history remain as evidence; no results are touched.
                if existing:
                    _atomic_bytes(root / PROJECT_FILE, old_project)
                    _atomic_bytes(root / MANIFEST_FILE, old_manifest)
                else:
                    (root / PROJECT_FILE).unlink(missing_ok=True)
                    (root / MANIFEST_FILE).unlink(missing_ok=True)
                _member(root, _PENDING).unlink()
                raise
            _member(root, _PENDING).unlink()
        return deepcopy(project)


def _draft_member(path):
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
        return False
    return ((path.name == "model_layers" and path.is_dir())
            or (path.name == _GUARD and path.is_file() and path.stat().st_size == 1))


def _tree_inventory(root, cancelled=None):
    """Content inventory, including empty directories, with no followed links."""
    inventory = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        _check_cancel(cancelled)
        for path in sorted(directory.iterdir()):
            relative = path.relative_to(root).as_posix()
            if relative in (_LOCK, _GUARD):
                continue
            _member(root, relative)
            if path.is_dir():
                inventory[relative] = dict(kind="directory")
                pending.append(path)
            elif path.is_file():
                inventory[relative] = dict(kind="file", size=path.stat().st_size, sha256=_sha(path))
            else:
                raise ValueError("Unsupported project folder member: " + relative)
    return inventory


def _copy_tree(source, ready, inventory, cancelled):
    for relative, row in sorted(inventory.items(), key=lambda item: (item[0].count("/"), item[0])):
        _check_cancel(cancelled)
        destination = _member(ready, relative)
        if row["kind"] == "directory":
            destination.mkdir()
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _member(source, relative).open("rb") as src, destination.open("xb") as dst:
            while True:
                _check_cancel(cancelled)
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        if destination.stat().st_size != row["size"] or _sha(destination) != row["sha256"]:
            raise ValueError("Project file changed while copying: " + relative)


def copy_folder(source, target, *, cancelled=None):
    """Copy a complete project, results and history without altering the source.

    A staged, verified copy is published only after the entire source inventory
    is checked again. Lock/guard state is local to a process and is not copied.
    Both inputs and outputs keep their bytes and restart identities unchanged.
    """
    source, target = filesystem_path(source), filesystem_path(target)
    if any(path.is_symlink() or getattr(path, "is_junction", lambda: False)() for path in (source, target)):
        raise ValueError("Linked source or destination project folders are not supported")
    source, target = source.resolve(), target.resolve()
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("Source and destination project folders must be separate, non-nested directories")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("Save As requires a new or empty destination folder")
    _read_folder(source)
    _check_cancel(cancelled)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _lock(source):
        _read_folder(source)
        list_history(source)
        inventory = _tree_inventory(source, cancelled)
        with tempfile.TemporaryDirectory(prefix=".braids-copy-", dir=target.parent) as temporary:
            staging = filesystem_path(temporary)
            ready = staging / "ready"
            ready.mkdir()
            _copy_tree(source, ready, inventory, cancelled)
            if _tree_inventory(source, cancelled) != inventory:
                raise ValueError("Project source changed during Save As; the copy was not published")
            if _tree_inventory(ready, cancelled) != inventory:
                raise ValueError("Staged project copy differs; the copy was not published")
            _read_folder(ready)
            _check_cancel(cancelled)
            if not target.exists():
                os.rename(ready, target)
            else:
                if not target.is_dir() or any(target.iterdir()):
                    raise ValueError("Save As destination changed while copying")
                # Keep an already chosen empty directory in place. Publish the
                # manifest last; restore staged members if any rename fails.
                names = sorted(p.name for p in ready.iterdir() if p.name != MANIFEST_FILE) + [MANIFEST_FILE]
                published = []
                try:
                    for name in names:
                        os.rename(ready / name, _member(target, name))
                        published.append(name)
                except BaseException:
                    for name in reversed(published):
                        os.rename(_member(target, name), ready / name)
                    raise
    return target


def update_session(root, session):
    """Persist UI/tool selections without creating a numerical revision."""
    root = filesystem_path(root).resolve()
    selected = _session(root, session)
    with _lock(root):
        manifest, _ = _read_folder(root)
        manifest["session"] = selected
        _atomic_json(root / MANIFEST_FILE, manifest)
    return manifest


def record_event(root, kind, **fields):
    """Append one immutable lifecycle record without rewriting earlier history."""
    root = filesystem_path(root).resolve()
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("History event kind must be nonempty")
    if {"schema", "kind", "timestamp", "revision"} & fields.keys():
        raise ValueError("Reserved history fields cannot be replaced")
    values = read_json(canonical(fields))
    for name in ("prepared", "output", "restart", "log"):
        if name in values and values[name] not in (None, ""):
            _member(root, values[name])
    with _lock(root):
        manifest, _ = _read_folder(root, assets=False)
        record = dict(values, schema=EVENT_SCHEMA, timestamp=_now(), kind=kind, revision=manifest["revision"])
        record.setdefault("project_id", manifest["project_id"])
        _atomic_json(_record_path(_member(root, "history/events"), record), record)
    return deepcopy(record)


def list_history(root):
    """Return verified revision summaries and lifecycle events in time order."""
    root = filesystem_path(root).resolve()
    manifest, _ = _read_folder(root, assets=False)
    records = []
    seen = set()
    revision = manifest["revision"]
    while revision is not None:
        if revision in seen or not revision.startswith("history/revisions/"):
            raise ValueError("Project revision history is cyclic or outside its directory")
        seen.add(revision)
        row = _read_record(_member(root, revision), REVISION_SCHEMA)
        records.append(dict(kind="saved", timestamp=row["timestamp"], revision=revision,
                            project_id=row["project_id"], name=row["project"]["name"]))
        revision = row["parent"]
    directory = _member(root, "history/events")
    if directory.is_dir():
        for path in directory.glob("*.json"):
            row = _read_record(_member(root, path.relative_to(root).as_posix()), EVENT_SCHEMA)
            if row["revision"] not in seen:
                raise ValueError("History event refers to an uncommitted revision")
            records.append(row)
    return sorted(records, key=lambda row: (row["timestamp"], row["kind"]))
