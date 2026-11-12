"""Isolated, reversible checkpoint-in-memory candidate; never a public option.

The maintained source and checkpoint schemas remain untouched. The same arrays,
metadata and hashes are serialized in the same nested uncompressed NPZ layout.
Only intermediate temporary files/readbacks are replaced by BytesIO payloads.
Validation must identify this hook separately from the frozen runtime identity.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path

import numpy as np


def _npz_bytes(**arrays):
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MemoryCheckpointCandidate:
    """Context manager with explicit undo, suitable for isolated worker tests."""
    def __init__(self):
        self.replacements = []
        self.stats = dict(solver=0, mixture=0, leaves=0, policy=0,
                          published_files=0, published_bytes=0)
        self.identity = dict(method='validation_only_nested_checkpoint_bytes134',
                             hook_sha256=_file_sha(__file__),
                             changes='Intermediate checkpoint I/O only; original inventory and schemas',
                             runtime_sources={})

    def _set(self, obj, name, value):
        previous = obj.__dict__.get(name, _MISSING)
        self.replacements.append((obj, name, previous))
        setattr(obj, name, value)

    def install(self):
        from braids_core.sediment import shoreline_solver as shoreline
        from braids_core.sediment import river_mixtures as mixture
        from braids_core.sediment import river_leaf_controller as leaf
        from braids_core.sediment import river_field_policy as policy
        from braids_core.sediment.bed_columns_v101 import host, array_sha
        from braids_core.windows_paths import filesystem_path
        from braids_gpu.atomic_output import _temporary_for, _cleanup, replace_with_windows_retry

        if self.replacements:
            raise RuntimeError('Candidate is already installed')
        classes = (shoreline.ShorelineSolver, mixture.RiverMixtureSolver,
                   leaf.RiverLeafController, policy.AutomaticRiver)
        if any('checkpoint_bytes' in cls.__dict__ for cls in classes):
            raise RuntimeError('Checkpoint candidate conflicts with an existing bytes implementation')
        self.identity['runtime_sources'] = {m.__name__: _file_sha(m.__file__)
                                            for m in (shoreline, mixture, leaf, policy)}
        stats = self.stats

        def solver_bytes(a):
            if a.physics._open:
                raise ValueError('Checkpoint needs accepted state')
            arrays, meta = a.snapshot()
            arrays = {k: a.host(v) for k, v in arrays.items()}
            arrays.update(saved_q=a.host(a.q), saved_z=a.host(a.z),
                          water_ledger=a.host(a.water_ledger))
            header = dict(identity=a.identity, physics=meta, time=a.time,
                          steps=a.steps, morphology=a.morphology,
                          hashes={k: hashlib.sha256(v.tobytes()).hexdigest()
                                  for k, v in arrays.items()})
            result = _npz_bytes(metadata=np.array(json.dumps(header, sort_keys=True)), **arrays)
            stats['solver'] += 1
            return result

        def mixture_bytes(a):
            payload = a.solver.checkpoint_bytes()
            result = _npz_bytes(river=np.frombuffer(payload, dtype=np.uint8),
                                columns=np.array(json.dumps(a.columns.checkpoint(), sort_keys=True)))
            stats['mixture'] += 1
            return result

        def leaf_bytes(c):
            a = c.owner
            payload = a.checkpoint_bytes()
            meta = dict(scope=leaf.SCOPE, geometry_sha256=c.geometry_sha256,
                        leaves=list(c.leaves), wrapper_sha256=_file_sha(leaf.__file__),
                        transfer_sha256=_file_sha(Path(leaf.__file__).with_name('river_amr_transfer.py')),
                        bank_transfer_sha256=_file_sha(Path(leaf.__file__).with_name('river_bank_remap.py')),
                        bank_transfer_native_sha256=_file_sha(Path(leaf.__file__).with_name('river_bank_remap.h')),
                        terrain_sha256=_file_sha(Path(leaf.__file__).with_name('river_terrain_transfer.py')),
                        identity=a.identity, payload_sha256=hashlib.sha256(payload).hexdigest())
            result = _npz_bytes(metadata=np.array(json.dumps(meta, sort_keys=True)),
                                owner=np.frombuffer(payload, dtype=np.uint8))
            stats['leaves'] += 1
            return result

        def policy_bytes(d):
            payload = d.controller.checkpoint_bytes()
            meta = dict(scope=policy.SCOPE, settings=asdict(d.settings), next_time=d.next_time,
                        events=d.events, code_sha256=_file_sha(policy.__file__),
                        runtime=d.indicators.identity,
                        history_sha256=array_sha(d.reference, d.delay, host(d.previous)),
                        payload_sha256=hashlib.sha256(payload).hexdigest())
            result = _npz_bytes(metadata=np.array(json.dumps(meta, sort_keys=True)),
                                payload=np.frombuffer(payload, np.uint8), reference=d.reference,
                                delay=d.delay, previous=host(d.previous))
            stats['policy'] += 1
            return result

        def checkpoint(a, path):
            payload = a.checkpoint_bytes()
            path = filesystem_path(path)
            temporary = _temporary_for(path)
            try:
                with temporary.open('xb') as stream:
                    stream.write(payload)
                replace_with_windows_retry(temporary, path)
            finally:
                _cleanup(temporary)
            stats['published_files'] += 1
            stats['published_bytes'] += len(payload)

        for cls, encode in zip(classes, (solver_bytes, mixture_bytes, leaf_bytes, policy_bytes)):
            self._set(cls, 'checkpoint_bytes', encode)
            self._set(cls, 'checkpoint', checkpoint)
        return self

    def close(self):
        while self.replacements:
            obj, name, previous = self.replacements.pop()
            if previous is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, previous)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


_MISSING = object()


def install():
    """Install the isolated candidate and return its identity/statistics/undo."""
    return MemoryCheckpointCandidate().install()
