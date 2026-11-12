"""Isolated, equivalent execution prototypes; never a supported solver variant.

Selected compiler executables are pinned for one run by an immutable byte lease.
All constitutive/hydraulic operators, compiled keys and input bytes are unchanged.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


class PinnedToolchain:
    """Resolve a run's compiler once, retain an immutable Windows read handle.

    PATH selection is explicitly fixed for the run, not periodically re-searched.
    Environment changes are re-resolved against the chosen executables before
    reuse; any changed executable selection rejects. GDAL/CUDA may add DLL-only
    paths, and the native deck reader changes cwd temporarily.
    Non-Windows keeps original lookups;
    there is no advisory-lock shortcut to the Windows immutability guarantee.
    """
    def __init__(self, lease, which=None, check_output=None):
        self.lease = lease
        self.which_original = which or shutil.which
        self.output_original = check_output or subprocess.check_output
        self.environment = (os.getcwd(), os.environ.get('PATH'), os.environ.get('PATHEXT'))
        self.checked_environments = {self.environment}
        self.checked_directories = {os.getcwd()}
        self.paths = {}
        self.versions = {}
        self.identities = {}
        self.lookup_hits = self.version_hits = 0

    def check_environment(self):
        if self.lease.closed:
            raise RuntimeError('Run-pinned compiler byte lease closed')
        current = (os.getcwd(), os.environ.get('PATH'), os.environ.get('PATHEXT'))
        if current not in self.checked_environments:
            for cmd, chosen in self.paths.items():
                found = self.which_original(cmd)
                if found is None or os.path.normcase(str(Path(found).resolve())) != os.path.normcase(chosen):
                    raise RuntimeError('Run-pinned compiler environment changed executable selection')
            self.checked_directories.add(os.getcwd())
            self.checked_environments.add(current)

    def which(self, cmd, mode=os.F_OK | os.X_OK, path=None):
        if os.name != 'nt' or cmd not in ('g++', 'gfortran') or mode != os.F_OK | os.X_OK or path is not None:
            return self.which_original(cmd, mode=mode, path=path)
        self.check_environment()
        if cmd not in self.paths:
            result = self.which_original(cmd, mode=mode, path=path)
            if result is None:
                return None  # Absence is not an immutable executable identity.
            absolute = str(Path(result).resolve())
            digest = self.lease.digest(absolute, immutable=True)
            self.paths[cmd] = absolute
            self.identities[cmd] = dict(path=absolute, sha256=digest)
        else:
            self.lookup_hits += 1
        return self.paths[cmd]

    def check_output(self, args, **kwargs):
        if (os.name != 'nt' or not isinstance(args, (list, tuple)) or len(args) != 2
                or args[1] != '--version' or kwargs != {'text': True}):
            return self.output_original(args, **kwargs)
        name = os.path.normcase(os.path.abspath(os.fspath(args[0])))
        chosen = {os.path.normcase(path): cmd for cmd, path in self.paths.items()}
        if name not in chosen:
            return self.output_original(args, **kwargs)
        self.check_environment()
        if name not in self.versions:
            self.versions[name] = self.output_original(args, **kwargs)
            self.identities[chosen[name]]['version_output'] = self.versions[name]
        else:
            self.version_hits += 1
        return self.versions[name]

    def evidence(self):
        return dict(environment=self.environment, checked_directories=sorted(self.checked_directories), compilers=self.identities,
            verified_environments=sorted(self.checked_environments),
            lookup_hits=self.lookup_hits, version_hits=self.version_hits,
            definition='Explicit within-run toolchain selection under immutable byte lease; fresh next-run lookup; no physical/model change')


def cached_epoch_write(self, controller, fields):
    """Same epoch identity and fields; no geometry reconstruction on cache hit."""
    from braids_core.sediment import river_leaf_io as io
    leaves = controller.leaves
    key = io.fingerprint(dict(geometry=controller.geometry_sha256, leaves=list(leaves)))
    name = 'mesh_' + key + '.npz'
    path = self.root / name
    if name not in self.epochs:
        if path.exists():
            raise FileExistsError('Unindexed mesh epoch exists')
        mesh = controller._mesh_cache.get(tuple(leaves))
        if mesh is None:
            mesh = controller.hierarchy.select(leaves)
        offsets = np.r_[0, np.cumsum([len(c.vertices_xy) for c in mesh.cells])]
        io.save_arrays_atomic(path, leaves=np.asarray(leaves), levels=controller.hierarchy.node_level[list(leaves)],
            offsets=offsets, vertices=np.concatenate([c.vertices_xy for c in mesh.cells]),
            area=np.array([c.area_m2 for c in mesh.cells]))
        self.epochs[name] = io.sha(path)
    frame = f'frame_{controller.owner.steps:08d}.npz'
    if (self.root / frame).exists():
        raise FileExistsError('Accepted frame already exists')
    if any(np.asarray(v).shape[-1:] != (len(leaves),) for k, v in fields.items() if k in ('q', 'z', 'mass')):
        raise ValueError('Frame shape differs from mesh epoch')
    io.save_arrays_atomic(self.root / frame, mesh_epoch=np.array(name), **fields)
    self.frames.append(dict(file=frame, sha256=io.sha(self.root / frame), epoch=name,
        time_s=controller.owner.time, steps=controller.owner.steps))
    io.write_json_atomic(self.index, dict(epochs=self.epochs, frames=self.frames))


@contextmanager
def install(*, mesh=True, toolchain=True):
    from braids_core.sediment.river_runtime_lease import RuntimeFileLease
    from braids_core.sediment import river_leaf_io as io
    lease = RuntimeFileLease()
    original_write, original_which, original_output = io.MeshEpochWriter.write, shutil.which, subprocess.check_output
    pinned = PinnedToolchain(lease, original_which, original_output)
    try:
        if mesh:
            io.MeshEpochWriter.write = cached_epoch_write
        if toolchain:
            # Resolve in the caller's initial environment, before temporary
            # native-reader working directories may be entered.
            pinned.which('g++')
            pinned.which('gfortran')
            shutil.which, subprocess.check_output = pinned.which, pinned.check_output
        yield pinned
    finally:
        io.MeshEpochWriter.write = original_write
        shutil.which, subprocess.check_output = original_which, original_output
        lease.close()
