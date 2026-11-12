"""Bounded isolated old-executable versus in-process geometry record timings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from braids_core.sediment import river_mesh_records
from braids_core.sediment.river_mesh_records import initialize, build
from braids_core.windows_native_io import native_directory
from braids_gpu.general_live_transport import read_records
from braids_gpu.graded_native_exporter import EXPORTER_SHA256, export_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deck', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--precompiled', action='store_true',
                        help='Bind the already authenticated build as a compiler-free release would')
    args = parser.parse_args()
    deck = Path(args.deck).resolve()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    executable = ROOT / 'braids_gpu/native_exporter_v1/xbraids_normal_export.exe'
    assert hashlib.sha256(executable.read_bytes()).hexdigest() == EXPORTER_SHA256
    built = build()  # compilation is not a runtime speed claim
    library, identity, _ = built
    if args.precompiled:
        river_mesh_records.build = lambda: built
    inputs = [deck, deck.with_suffix('.data.json'), deck.with_suffix('.data.bin')]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    rows = []
    expected = None
    for repeat, variant in enumerate(('old', 'new', 'new', 'old', 'old', 'new', 'new', 'old')):
        with tempfile.TemporaryDirectory(prefix='braids-record-bench-') as tmp:
            folder = Path(tmp)
            for path in inputs:
                shutil.copy2(path, folder/path.name)
            began = time.perf_counter()
            if variant == 'old':
                with native_directory(folder, hashes,
                        outputs=('coupled_input.bin', 'normal_metadata.bin')) as native:
                    subprocess.run([str(executable), 'case.data'], cwd=native,
                        env=export_environment(executable.parent), stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, check=True, timeout=60,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                records = read_records(folder)
            else:
                records, _ = initialize(folder/'case.data')
            elapsed = time.perf_counter() - began
            if expected is None:
                expected = records
            exact = (set(records) == set(expected) and all(
                np.asarray(records[k]).dtype == np.asarray(expected[k]).dtype and
                np.asarray(records[k]).shape == np.asarray(expected[k]).shape and
                np.asarray(records[k]).tobytes() == np.asarray(expected[k]).tobytes()
                for k in records))
            if not exact:
                raise ValueError('Isolated record comparison differs')
            rows.append(dict(repeat=repeat, variant=variant, seconds=elapsed,
                             all_records_byte_exact=exact))
            print(rows[-1], flush=True)
    result = dict(scope='initializer only; no deck writing, solver stepping or AMR transfer',
        source_deck=str(deck), input_sha256=hashes, cells=expected['nc'], faces=expected['nf'],
        native_reference_sha256=EXPORTER_SHA256, candidate_identity=identity, rows=rows,
        precompiled_binding=args.precompiled,
        mean_seconds={v: float(np.mean([r['seconds'] for r in rows if r['variant'] == v]))
                      for v in ('old', 'new')})
    (output/'BENCHMARK.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
