"""Validation-only lossless output-format microbenchmark on saved run arrays.

This does not modify any source run. Timings include an atomic publication and
all byte checksums on writes, and full integrity verification on reads. The OS
file cache is warm; no fsync is added because the maintained writer has none.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from braids_core.windows_paths import filesystem_path


def sha(path):
    return hashlib.sha256(filesystem_path(path).read_bytes()).hexdigest()


def equal(actual, expected):
    if set(actual) != set(expected):
        raise AssertionError('Array inventory differs')
    for name, a in actual.items():
        b = expected[name]
        if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
            raise AssertionError('Lossless array mismatch: '+name)


def tree_bytes(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.is_dir() else path.stat().st_size


def remove_owned(path, root):
    path = filesystem_path(path)
    if path.resolve().parent != root.resolve() or not path.name.startswith('sample_'):
        raise ValueError('Benchmark cleanup target is not an owned sample')
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def write_payload(path, method, arrays):
    """Return external checksum receipt, matching INDEX's role for NPZ."""
    pending = path.with_name('sample_'+uuid4().hex+'.partial')
    if method == 'npy_directory':
        pending.mkdir()
        files = {}
        for name, a in arrays.items():
            target = pending/(name+'.npy')
            with target.open('xb') as stream:
                np.save(stream, a, allow_pickle=False)
            files[target.name] = sha(target)
        (pending/'MANIFEST.json').write_text(json.dumps(files, sort_keys=True), encoding='utf-8')
        receipt = sha(pending/'MANIFEST.json')
        os.replace(pending, path)
        return receipt
    if method.startswith('npz'):
        with pending.open('xb') as stream:
            (np.savez_compressed if method == 'npz_compressed' else np.savez)(stream, **arrays)
    else:
        import h5py
        with h5py.File(pending, 'x') as f:
            for name, a in arrays.items():
                # HDF5 has no native NumPy Unicode dtype. Exact code units plus
                # dtype/shape metadata preserve strings byte-for-byte.
                value = a.reshape(-1).view(np.uint32) if a.dtype.kind == 'U' else a
                kwargs = {'compression': 'lzf'} if method == 'hdf5_lzf' and value.ndim and value.size else {}
                d = f.create_dataset(name, data=value, **kwargs)
                d.attrs['numpy_dtype'] = a.dtype.str
                d.attrs['numpy_shape'] = a.shape
    os.replace(pending, path)
    return sha(path)


def read_payload(path, method, receipt, names=None):
    if method == 'npy_directory':
        if sha(path/'MANIFEST.json') != receipt:
            raise ValueError('NPY manifest changed')
        files = json.loads((path/'MANIFEST.json').read_text(encoding='utf-8'))
        for filename, digest in files.items():
            if sha(path/filename) != digest:
                raise ValueError('NPY payload changed')
        wanted = [Path(n).stem for n in files] if names is None else names
        return {name: np.load(path/(name+'.npy'), allow_pickle=False) for name in wanted}
    if sha(path) != receipt:
        raise ValueError('Payload changed')
    if method.startswith('npz'):
        with np.load(path, allow_pickle=False) as f:
            return {name: f[name] for name in (f.files if names is None else names)}
    import h5py
    with h5py.File(path, 'r') as f:
        result = {}
        for name in (f.keys() if names is None else names):
            d = f[name]; a = np.asarray(d[()]); dtype = np.dtype(d.attrs['numpy_dtype'])
            shape = tuple(int(v) for v in d.attrs['numpy_shape'])
            result[name] = a.view(dtype).reshape(shape) if dtype.kind == 'U' else a.reshape(shape)
        return result


def arrays_from(path, kind):
    raw = filesystem_path(path).read_bytes()
    if kind == 'frame':
        with np.load(io.BytesIO(raw), allow_pickle=False) as f:
            return {name: f[name] for name in f.files}
    # Public checkpoint -> policy -> leaves -> mixture -> solver. This extracts
    # the actual solver arrays only for a format benchmark, never for execution.
    for field in ('payload', 'payload', 'owner', 'river'):
        with np.load(io.BytesIO(raw), allow_pickle=False) as f:
            raw = f[field].tobytes()
    with np.load(io.BytesIO(raw), allow_pickle=False) as f:
        return {name: f[name] for name in f.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', action='append', default=[])
    parser.add_argument('--checkpoint', action='append', default=[])
    parser.add_argument('--output', default='validation/_performance134/formats')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 2 or not (args.frame or args.checkpoint):
        parser.error('At least two repeats and one input are required')
    root = filesystem_path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    methods = ['npz_uncompressed', 'npz_compressed', 'npy_directory']
    try:
        import h5py
        methods += ['hdf5_uncompressed', 'hdf5_lzf']
        hdf5_version = h5py.__version__
    except ImportError:
        hdf5_version = None
    sources = [('frame', Path(p).resolve()) for p in args.frame] + [('checkpoint_solver', Path(p).resolve()) for p in args.checkpoint]
    report = dict(scope='Warm-cache validation-only format benchmark; no production change',
                  numpy=np.__version__, h5py=hdf5_version, repeats=args.repeats,
                  write_includes='serialize, close, atomic publish, full SHA-256 receipt',
                  read_includes='full SHA-256 verification and selected or all decode; no cache bypass',
                  fsync=False, cases=[])
    rng = random.Random(134)
    for number, (kind, source) in enumerate(sources):
        source_hash = sha(source)
        arrays = arrays_from(source, 'frame' if kind == 'frame' else 'checkpoint')
        wanted = ('q', 'z') if kind == 'frame' else ('saved_q', 'saved_z')
        selected = {k: arrays[k] for k in wanted}
        rows = {method: [] for method in methods}
        case = dict(kind=kind, source=str(source), source_sha256=source_hash,
                    source_bytes=source.stat().st_size, cells=int(selected[wanted[0]].shape[-1]),
                    array_count=len(arrays), array_bytes=sum(a.nbytes for a in arrays.values()),
                    selected_array_bytes=sum(a.nbytes for a in selected.values()))
        for repeat in range(args.repeats+1):
            order = methods.copy(); rng.shuffle(order)
            for method in order:
                path = root/f'sample_{number}_{repeat}_{method}'
                began = time.perf_counter(); receipt = write_payload(path, method, arrays); write_s = time.perf_counter()-began
                began = time.perf_counter(); actual = read_payload(path, method, receipt); read_all_s = time.perf_counter()-began
                equal(actual, arrays)
                began = time.perf_counter(); actual = read_payload(path, method, receipt, wanted); read_selected_s = time.perf_counter()-began
                equal(actual, selected)
                row = dict(write_s=write_s, read_all_s=read_all_s, read_selected_s=read_selected_s, bytes=tree_bytes(path))
                if repeat: rows[method].append(row)
                remove_owned(path, root)
        if sha(source) != source_hash:
            raise ValueError('Source changed during benchmark')
        case['methods'] = {method: dict(samples=samples, median={key: statistics.median(r[key] for r in samples) for key in samples[0]}) for method, samples in rows.items()}
        report['cases'].append(case)
        (root/'BENCHMARK.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(case), flush=True)
    print(root/'BENCHMARK.json')


if __name__ == '__main__':
    main()
