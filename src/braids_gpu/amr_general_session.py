"""Write-once nonbank AMR sessions and on-demand schema-3 operators.

The native executable only prepares selected geometry. Accepted physical state
belongs to the CUDA route and is never obtained from a later CPU initializer.
This separate contract deliberately does not reinterpret bank-AMR sessions.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from braids_core.windows_paths import filesystem_path

from .amr_general_native_geometry import DEFAULT_RUNTIME, verify_runtime
from .amr_general_fast_export import export_selected
from .amr_native_geometry import hierarchy_from_sidecar
from .amr_general_prepared import (GLOBAL_CONTROLS,
    load_general_amr_prepared)
from .amr_prepared import PreparedAMR, _read_coupled
from .amr_on_demand import cold_indicator_state, selector_inputs
from .amr_general_selector import GeneralAMRStateSelector
from .atomic_output import write_json_atomic


KIND = 'on-demand-general-transport-amr'
SCHEMA = 3
SLOPE_SELECTOR_POLICY = 'local_eta_plane_v1'


def sha(path):
    """Hash nested operator artifacts without the Windows MAX_PATH limit."""
    return hashlib.sha256(filesystem_path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def validate_selector_policy(policy):
    if policy is not None:
        require(type(policy) is str and policy == SLOPE_SELECTOR_POLICY,
                'Unknown general AMR selector policy')
    return policy


def selector_source_hashes(policy):
    validate_selector_policy(policy)
    if policy is None:
        return None
    root = Path(__file__).resolve().parent
    return {name: sha(root / name) for name in
            ('amr_slope_indicator_v2.py', 'amr_slope_indicator_research.py')}


def state_selector(prepared, arguments, policy):
    original = GeneralAMRStateSelector(prepared, **arguments)
    if validate_selector_policy(policy) is None:
        return original
    from .amr_slope_indicator_v2 import VersionedSlopeAwareSelector
    return VersionedSlopeAwareSelector(original)


def _input_files(package):
    manifest = json.loads((package.folder / 'amr_prepared.json').read_text('utf-8'))
    receipt = json.loads((package.folder / manifest['topologies'][0]['receipt']).read_text('utf-8'))
    files = receipt['origin']['files']
    return {key: value for key, value in files.items() if key != 'selected_nodes.txt'}


def _ids(value):
    ids = np.asarray(value)
    require(ids.ndim == 1 and ids.size > 0 and ids.dtype.kind in 'iu',
            'General AMR request needs integer selected nodes')
    ids = np.sort(ids.astype(np.int64))
    require(np.all(ids >= 0) and len(np.unique(ids)) == len(ids),
            'General AMR selected nodes must be unique and nonnegative')
    return ids


def _files(folder):
    folder = filesystem_path(folder).resolve()
    require(folder.is_dir() and not folder.is_symlink(),
            'General AMR operator package is missing or linked')
    result = {}
    for path in folder.rglob('*'):
        require(not path.is_symlink() and path.resolve().is_relative_to(folder),
                'General AMR operator package contains a linked/escaping member')
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = sha(path)
        else:
            require(path.is_dir(), 'General AMR package contains a non-file member')
    return dict(sorted(result.items()))


def _loaded_package_files(package):
    """Reuse digests already verified by ``load_general_amr_prepared``.

    The loader has just hashed every listed artifact and rejected any extra
    file.  Re-reading the same 40--70 MB package merely to reconstruct that
    mapping made every AMR cache miss pay the authentication cost repeatedly.
    """
    manifest_path = package.folder / 'amr_prepared.json'
    manifest = json.loads(manifest_path.read_text('utf-8'))
    require(package.identity == sha(manifest_path) and
            isinstance(manifest.get('artifacts'), dict) and
            all(isinstance(name, str) and isinstance(value, str) and
                len(value) == 64 for name, value in manifest['artifacts'].items()),
            'Loaded general AMR package identity differs')
    return dict(sorted({**manifest['artifacts'],
                        'amr_prepared.json': package.identity}.items()))


@dataclass(frozen=True)
class GeneralStoredOperator:
    selected_ids: np.ndarray
    request_key: str
    package_identity: str
    entry_sha256: str
    package: PreparedAMR


@dataclass(frozen=True)
class _AuthenticatedOperator:
    selected_ids: np.ndarray
    request_key: str
    package_identity: str
    entry_sha256: str
    package_files: dict[str, str]
    entry_stamp: tuple[int, int, int, int]
    package_stamps: dict[str, tuple[int, int, int, int]]


@dataclass(frozen=True)
class _ProducedGeneralOperator:
    """Internal proof that the staged destination already passed the loader."""
    package: PreparedAMR
    package_files: dict[str, str]


def _file_stamp(path):
    """Cheap change detector used only after a full cryptographic read.

    A fresh process still hashes every byte.  During one live session this
    stamp lets repeated lookups avoid rereading tens of megabytes.  Any
    metadata change triggers the original full SHA-256 authentication.
    """
    path = filesystem_path(path)
    require(path.is_file() and not path.is_symlink(),
            'General AMR authenticated file is missing or linked')
    value = path.stat(follow_symlinks=False)
    return (int(value.st_size), int(value.st_mtime_ns), int(value.st_ctime_ns),
            int(value.st_ino))


def _file_stamps(folder):
    folder = filesystem_path(folder).resolve()
    require(folder.is_dir() and not folder.is_symlink(),
            'General AMR operator package is missing or linked')
    result = {}
    for path in folder.rglob('*'):
        require(not path.is_symlink() and path.resolve().is_relative_to(folder),
                'General AMR operator package contains a linked/escaping member')
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = _file_stamp(path)
        else:
            require(path.is_dir(),
                    'General AMR package contains a non-file member')
    return dict(sorted(result.items()))


class GeneralOperatorStore:
    """Append-only schema-3 operator objects, including native face identity."""

    _parsed_cache_size = 2

    def __init__(self, root, *, hierarchy_sha256, base_input_sha256,
                 runtime_build_sha256, tree_input_sha256):
        self.root = filesystem_path(root).resolve()
        self.identities = dict(hierarchy_sha256=hierarchy_sha256,
            base_input_sha256=base_input_sha256,
            runtime_build_sha256=runtime_build_sha256)
        self.tree_input_sha256 = tree_input_sha256
        require(all(isinstance(value, str) and len(value) == 64 for value in
                    (*self.identities.values(), tree_input_sha256)),
                'General AMR store identities differ')
        self.manifest = dict(schema=SCHEMA, kind='general-transport-amr-operator-store',
            **self.identities, tree_input_sha256=tree_input_sha256)
        self.identity = digest(self.manifest)
        self.manifest['identity'] = self.identity
        if not self.root.exists():
            require(not self.root.is_symlink(), 'General AMR store root is linked')
            self.root.mkdir(parents=True)
            (self.root / 'operators').mkdir()
            write_json_atomic(self.root / 'store.json', self.manifest)
        require(self.root.is_dir() and not self.root.is_symlink() and
                (self.root / 'operators').is_dir() and
                not (self.root / 'operators').is_symlink() and
                not (self.root / 'store.json').is_symlink() and
                json.loads((self.root / 'store.json').read_text('utf-8')) == self.manifest,
                'General AMR store identity differs')
        # Every object is fully authenticated on open. Later lookups still hash
        # every stored byte, but reuse the parsed operator while it is resident.
        self._authenticated = {}
        self._parsed = OrderedDict()
        self._all()

    def _request(self, ids):
        ids = _ids(ids)
        return ids, digest(dict(store_identity=self.identity,
                                selected_node_ids=ids.tolist()))

    def _checked_package(self, folder, ids, package_identity):
        package = load_general_amr_prepared(folder / 'package')
        require(package.identity == package_identity and
                package.schema == SCHEMA and len(package.topologies) == 1 and
                np.array_equal(package.topologies[0].node_ids, ids) and
                all(package.source.get(key) == value for key, value in
                    self.identities.items()) and
                sha(package.folder / 'topology_0000' / 'amr_tree_input.bin') ==
                    self.tree_input_sha256,
                'General AMR operator selected leaves or source changed')
        return package

    def _remember(self, identity, package):
        self._parsed[identity] = package
        self._parsed.move_to_end(identity)
        if len(self._parsed) > self._parsed_cache_size:
            self._parsed.popitem(last=False)

    def _read(self, folder):
        require(folder.parent == self.root / 'operators' and folder.is_dir() and
                not folder.is_symlink() and
                {p.name for p in folder.iterdir()} == {'entry.json', 'package'},
                'General AMR operator object differs')
        receipt_path = folder / 'entry.json'
        require(receipt_path.is_file() and not receipt_path.is_symlink(),
                'General AMR operator receipt is missing or linked')
        prior = self._authenticated.get(folder.name)
        if prior is not None:
            entry_stamp = _file_stamp(receipt_path)
            package_stamps = _file_stamps(folder / 'package')
            if (prior.entry_stamp == entry_stamp and
                    prior.package_stamps == package_stamps):
                return prior
            # A timestamp, size, file ID, or inventory changed.  Re-run the
            # original byte-level authentication before accepting a benign
            # metadata-only touch; altered content remains fail-closed.
            require(prior.entry_sha256 == sha(receipt_path),
                    'General AMR operator changed after authentication')
            require(prior.package_files == _files(folder / 'package'),
                    'General AMR operator package bytes changed')
            refreshed = _AuthenticatedOperator(prior.selected_ids,
                prior.request_key, prior.package_identity,
                prior.entry_sha256, prior.package_files,
                entry_stamp, package_stamps)
            self._authenticated[folder.name] = refreshed
            return refreshed
        receipt = json.loads(receipt_path.read_text('utf-8'))
        require(set(receipt) == {'schema', 'kind', 'store_identity', 'request_key',
                'selected_node_ids', 'package_identity', 'package_files'} and
                receipt['schema'] == SCHEMA and
                receipt['kind'] == 'zero-step-general-amr-operator' and
                receipt['store_identity'] == self.identity and
                folder.name == receipt['package_identity'],
                'General AMR operator receipt differs')
        ids, key = self._request(receipt['selected_node_ids'])
        package_files = _files(folder / 'package')
        require(key == receipt['request_key'] and
                receipt['package_files'] == package_files,
                'General AMR operator request or package bytes changed')
        receipt_sha256 = sha(receipt_path)
        package = self._checked_package(folder, ids,
                                        receipt['package_identity'])
        entry = _AuthenticatedOperator(ids, key, package.identity,
            receipt_sha256, package_files, _file_stamp(receipt_path),
            _file_stamps(folder / 'package'))
        self._authenticated[folder.name] = entry
        self._remember(package.identity, package)
        return entry

    def _materialize(self, entry):
        package = self._parsed.get(entry.package_identity)
        if package is None:
            folder = self.root / 'operators' / entry.package_identity
            package = self._checked_package(folder, entry.selected_ids,
                                            entry.package_identity)
            require(sha(folder / 'entry.json') == entry.entry_sha256 and
                    _files(folder / 'package') == entry.package_files,
                    'General AMR operator changed while loading')
            self._remember(entry.package_identity, package)
        else:
            self._parsed.move_to_end(entry.package_identity)
        # Callers receive independent mutable arrays; they cannot alter the
        # authenticated copy that later requests will use.
        return GeneralStoredOperator(entry.selected_ids.copy(),
            entry.request_key, entry.package_identity, entry.entry_sha256,
            copy.deepcopy(package))

    def _all(self):
        entries = []
        for path in (self.root / 'operators').iterdir():
            entries.append(self._read(path))
        require(len({entry.request_key for entry in entries}) == len(entries),
                'Duplicate general AMR request in store')
        return entries

    def get(self, ids):
        _, key = self._request(ids)
        entries = [entry for entry in self._all() if entry.request_key == key]
        require(len(entries) <= 1, 'Ambiguous general AMR selected operator')
        return self._materialize(entries[0]) if entries else None

    def get_or_create(self, ids, producer, *, cancelled=lambda: False):
        ids, key = self._request(ids)
        existing = self.get(ids)
        if existing is not None:
            return existing
        if cancelled():
            raise InterruptedError('General AMR operator request cancelled')
        staging = filesystem_path(tempfile.mkdtemp(prefix='.pending-', dir=self.root))
        try:
            product = producer(staging / 'package', ids)
            package = (product.package if isinstance(product,
                       _ProducedGeneralOperator) else product)
            require(isinstance(package, PreparedAMR) and
                    package.folder.resolve() == (staging / 'package').resolve(),
                    'General AMR producer returned a different package')
            if isinstance(product, _ProducedGeneralOperator):
                checked = package
                package_files = _loaded_package_files(checked)
                require(package_files == product.package_files,
                        'General AMR producer authentication proof differs')
            else:
                checked = load_general_amr_prepared(staging / 'package')
                package_files = _loaded_package_files(checked)
            require(checked.identity == package.identity and
                    len(checked.topologies) == 1 and
                    np.array_equal(checked.topologies[0].node_ids, ids) and
                    all(checked.source.get(name) == value for name, value in
                        self.identities.items()) and
                    sha(checked.folder / 'topology_0000' / 'amr_tree_input.bin') ==
                        self.tree_input_sha256,
                    'General AMR producer changed original inputs or tree')
            receipt = dict(schema=SCHEMA, kind='zero-step-general-amr-operator',
                store_identity=self.identity, request_key=key,
                selected_node_ids=ids.tolist(), package_identity=checked.identity,
                package_files=package_files)
            write_json_atomic(staging / 'entry.json', receipt)
            entry_sha256 = sha(staging / 'entry.json')
            if cancelled():
                raise InterruptedError('General AMR operator request cancelled')
            target = self.root / 'operators' / checked.identity
            for attempt in range(12):
                prior = self.get(ids)
                if prior is not None:
                    require(prior.package_identity == checked.identity,
                            'Conflicting general AMR operator publication')
                    return prior
                try:
                    os.rename(staging, target)
                    break
                except OSError as error:
                    if os.name != 'nt' or getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 11:
                        raise
                    time.sleep(min(.05 * (attempt + 1), .5))
            # ``checked`` has already passed the complete byte-level loader.
            # The directory rename is atomic and changes no file contents, so
            # publish that authenticated object directly instead of hashing
            # and parsing the entire package for a third time.
            published = replace(checked, folder=target / 'package')
            entry = _AuthenticatedOperator(ids.copy(), key, checked.identity,
                entry_sha256, package_files, _file_stamp(target / 'entry.json'),
                _file_stamps(target / 'package'))
            self._authenticated[target.name] = entry
            self._remember(checked.identity, published)
            return self._materialize(entry)
        finally:
            if staging.exists():
                require(staging.resolve().is_relative_to(self.root.resolve()) and
                        staging.name.startswith('.pending-') and not staging.is_symlink(),
                        'Refusing to clean unexpected general AMR pending object')
                shutil.rmtree(staging)


class GeneralGeometrySession:
    """One ordinary, no-bank live AMR run family and its lazy geometry cache."""

    @classmethod
    def create(cls, folder, deck, *, runtime=DEFAULT_RUNTIME,
               cancelled=lambda: False, progress=None, selector_policy=None):
        folder, deck, runtime = (filesystem_path(path).resolve() for path in
                                 (folder, deck, runtime))
        validate_selector_policy(selector_policy)
        policy_sources = selector_source_hashes(selector_policy)
        require(not folder.exists(), 'General AMR session is write-once')
        require(progress is None or callable(progress),
                'General AMR progress callback must be callable')

        def update(fraction, message):
            if cancelled():
                raise InterruptedError('General AMR session creation cancelled')
            if progress is not None:
                progress(fraction, message)
            if cancelled():
                raise InterruptedError('General AMR session creation cancelled')

        update(0., 'Checking original general AMR inputs')
        hierarchy = hierarchy_from_sidecar(deck.parent / 'amr_hierarchy.npz')
        finest = np.flatnonzero(hierarchy.node_level ==
                               np.max(hierarchy.node_level))
        folder.parent.mkdir(parents=True, exist_ok=True)
        staging = filesystem_path(tempfile.mkdtemp(prefix='.general-amr-',
                                                  dir=folder.parent))
        try:
            update(.1, 'Exporting general AMR cold geometry')
            bootstrap, _ = export_selected(deck, staging / 'bootstrap', finest,
                                           runtime=runtime, cancelled=cancelled)
            tree_path = staging / 'bootstrap_native' / 'amr_tree_input.bin'
            tree = _read_coupled(tree_path)
            require(int(bootstrap.topologies[0].data['bank_toe_erosion_mode']) == 0,
                    'General AMR session cannot contain bank physics')
            arguments = selector_inputs(tree)
            selector = state_selector(bootstrap, arguments, selector_policy)
            initial_tree_state = np.ascontiguousarray(tree['state'].T)
            proposal = selector.propose(bootstrap.topologies[0],
                bootstrap.topologies[0].data['state'].T,
                cached_tree_state=initial_tree_state,
                delay_state=np.zeros(len(tree['parent']), dtype=np.int64),
                time_s=0., initial=True,
                **cold_indicator_state(selector, bootstrap.topologies[0],
                                       initial_tree_state))
            update(.5, 'Authenticating first accepted mesh')
            source = dict(schema=SCHEMA, kind=KIND, deck=str(deck),
                runtime=str(runtime), original_input_files=_input_files(bootstrap),
                bootstrap_identity=bootstrap.identity,
                tree_input_sha256=sha(tree_path),
                initial_node_ids=list(proposal.node_ids),
                selector_contract=selector.contract_identity,
                initial_proposal=proposal.receipt,
                original_controls=selector.settings,
                native_solver_steps=0, evolving_banks=False,
                live_boundaries=bool(np.array_equal(
                    bootstrap.topologies[0].data['mode'], [9, 4])))
            if selector_policy is not None:
                source.update(selector_policy=selector_policy,
                              selector_source_sha256=policy_sources)
            source['identity'] = digest(source)
            write_json_atomic(staging / 'session.json', source)
            session = cls(staging, cancelled=cancelled)
            session.get(proposal.node_ids)
            update(.9, 'Publishing general AMR session')
            for attempt in range(12):
                require(not folder.exists(), 'General AMR session is write-once')
                try:
                    os.rename(staging, folder)
                    break
                except OSError as error:
                    if os.name != 'nt' or getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 11:
                        raise
                    time.sleep(min(.05 * (attempt + 1), .5))
            ready = cls(folder, cancelled=cancelled)
            ready.requests = session.requests
            if progress is not None:
                try:
                    progress(1., 'General AMR session ready')
                except Exception:
                    pass
            return ready
        finally:
            if staging.exists():
                require(staging.resolve().is_relative_to(folder.parent.resolve()) and
                        staging.name.startswith('.general-amr-') and
                        not staging.is_symlink(),
                        'Refusing to clean unexpected general AMR staging')
                shutil.rmtree(staging)

    def __init__(self, folder, *, cancelled=lambda: False):
        self.folder = filesystem_path(folder).resolve()
        self.manifest = json.loads((self.folder / 'session.json').read_text('utf-8'))
        m = self.manifest
        require(m.get('schema') == SCHEMA and m.get('kind') == KIND and
                m.get('native_solver_steps') == 0 and
                m.get('evolving_banks') is False and
                m.get('identity') == digest({key: value for key, value in
                                           m.items() if key != 'identity'}),
                'General AMR session identity differs')
        self.identity = m['identity']
        self.cancelled = cancelled
        self.deck, self.runtime = filesystem_path(m['deck']), filesystem_path(m['runtime'])
        self.evolving_banks = False
        self.general_transport = True
        self.selector_policy = validate_selector_policy(m.get('selector_policy'))
        if self.selector_policy is None:
            require('selector_source_sha256' not in m,
                    'Legacy general AMR session has selector source hashes')
        else:
            require(m.get('selector_source_sha256') ==
                    selector_source_hashes(self.selector_policy),
                    'General AMR selector source differs')
        self.bank_mode = 0
        self.live_boundaries = m['live_boundaries']
        self.checkpoint_schema = 5
        self.verify_originals()
        self.bootstrap = load_general_amr_prepared(self.folder / 'bootstrap')
        tree_path = self.folder / 'bootstrap_native' / 'amr_tree_input.bin'
        require(self.bootstrap.identity == m['bootstrap_identity'] and
                sha(tree_path) == m['tree_input_sha256'] and
                _input_files(self.bootstrap) == m['original_input_files'] and
                int(self.bootstrap.topologies[0].data['bank_toe_erosion_mode']) == 0 and
                self.live_boundaries == bool(np.array_equal(
                    self.bootstrap.topologies[0].data['mode'], [9, 4])),
                'General AMR session bootstrap differs')
        self.tree = _read_coupled(tree_path)
        self.initial_tree_state = np.ascontiguousarray(self.tree['state'].T)
        self.selector_arguments = selector_inputs(self.tree)
        selector = state_selector(self.bootstrap, self.selector_arguments,
                                  self.selector_policy)
        require(selector.contract_identity == m['selector_contract'] and
                selector.settings == m['original_controls'],
                'General AMR selection controls changed')
        self.store = GeneralOperatorStore(self.folder / 'store',
            hierarchy_sha256=self.bootstrap.source['hierarchy_sha256'],
            base_input_sha256=self.bootstrap.source['base_input_sha256'],
            runtime_build_sha256=self.bootstrap.source['runtime_build_sha256'],
            tree_input_sha256=m['tree_input_sha256'])
        self.requests = []

    def verify_originals(self):
        if (Path(__file__).resolve().parents[1] / 'PACKAGE-MANIFEST.json').exists():
            from .amr_portable_runtime import verify_packaged_runtime
            packaged = verify_packaged_runtime()
            require(self.runtime.resolve() == packaged.get('general_runtime'),
                    'General AMR initializer is outside selected worker package')
        verify_runtime(self.runtime)
        for name, value in self.manifest['original_input_files'].items():
            actual = 'geometry.npz' if name == 'original_geometry.npz' else name
            require(Path(actual).name == actual and
                    sha(self.deck.parent / actual) == value,
                    'Original general AMR input changed: ' + name)

    def initial_selection(self):
        selector = state_selector(self.bootstrap, self.selector_arguments,
                                  self.selector_policy)
        proposal = selector.propose(self.bootstrap.topologies[0],
            self.bootstrap.topologies[0].data['state'].T,
            cached_tree_state=self.initial_tree_state,
            delay_state=np.zeros(len(self.tree['parent']), dtype=np.int64),
            time_s=0., initial=True,
            **cold_indicator_state(selector, self.bootstrap.topologies[0],
                                   self.initial_tree_state))
        require(list(proposal.node_ids) == self.manifest['initial_node_ids'] and
                proposal.receipt == self.manifest['initial_proposal'],
                'Original general AMR cold selection differs')
        return proposal

    def _produce(self, folder, ids):
        self.verify_originals()
        jobs = self.folder / 'initializations'
        jobs.mkdir(exist_ok=True)
        job = filesystem_path(tempfile.mkdtemp(prefix='request-', dir=jobs))
        package, native = export_selected(self.deck, job / 'prepared', ids,
                                          runtime=self.runtime,
                                          cancelled=self.cancelled)
        require(_input_files(package) == self.manifest['original_input_files'] and
                sha(native / 'amr_tree_input.bin') ==
                self.manifest['tree_input_sha256'],
                'New general AMR operator changed original inputs')
        self.verify_originals()
        shutil.copytree(package.folder, folder)
        checked = load_general_amr_prepared(folder)
        return _ProducedGeneralOperator(checked, _loaded_package_files(checked))

    def get(self, ids):
        if self.cancelled():
            raise InterruptedError('General AMR mesh request cancelled')
        self.verify_originals()
        started = time.perf_counter()
        hit = self.store.get(ids)
        result = hit if hit is not None else self.store.get_or_create(
            ids, self._produce, cancelled=self.cancelled)
        require(_input_files(result.package) == self.manifest['original_input_files'],
                'Cached general AMR operator changed original inputs')
        self.requests.append(dict(node_count=len(result.selected_ids),
            request_key=result.request_key, cache_hit=hit is not None,
            elapsed_s=time.perf_counter() - started,
            package_identity=result.package_identity,
            native_solver_steps=0))
        return result

    def view(self, entries):
        require(bool(entries), 'Empty general AMR selected-operator family')
        first = entries[0].package
        require(first.schema == SCHEMA, 'General AMR view requires schema 3')
        tops = []
        names = set()
        base = first.topologies[0].data
        # These fields authenticate the topology-local subset of one common
        # hierarchy raw-roughness binding, so their array hash/range may differ
        # after regridding.  The common AMR fingerprint below must not differ.
        local_typed_roughness = {
            'delft_forcing_binding', 'delft_forcing_fingerprint',
            'roughness_parameter', 'amr_roughness_node_ids'}
        table_keys = {key for key in base if key.startswith((
            'series_', 'hydro_series_', 'fraction_series_'))}
        for entry in entries:
            package = entry.package
            top = package.topologies[0]
            data = top.data
            require(package.schema == SCHEMA and
                    _input_files(package) == self.manifest['original_input_files'] and
                    set(package.tree) == set(first.tree) and
                    all(np.array_equal(package.tree[key], first.tree[key])
                        for key in first.tree) and
                    set(data) == set(base) and
                    all(np.array_equal(data[key], base[key]) for key in
                        GLOBAL_CONTROLS) and
                    {key for key in data if key.startswith((
                        'series_', 'hydro_series_', 'fraction_series_'))} == table_keys and
                    all(np.array_equal(data[key], base[key]) for key in
                        table_keys) and
                    all(np.array_equal(data[key], base[key]) for key in base
                        if key not in {'nc', 'nf', *local_typed_roughness} and
                        np.asarray(base[key]).shape == ()) and
                    data.get('amr_typed_roughness_fingerprint') ==
                    base.get('amr_typed_roughness_fingerprint') and
                    top.name not in names,
                    'General AMR operator changes hierarchy, controls or identity')
            names.add(top.name)
            tops.append(top)
        return replace(first, folder=self.folder, identity=self.identity,
                       topologies=tuple(tops),
                       source=dict(session_identity=self.identity))
