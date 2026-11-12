"""Isolated source-archive preparation candidate; never patches product code.

The candidate authenticates bytes using a private mandatory read-handle lease.
All handles close before its staging directory is renamed. Reference and
candidate use the same unmodified released prepare_sources implementation.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


@contextmanager
def fresh_lease(module):
    """Validation-only private scope, independent of an enclosing run lease."""
    lease = module.RuntimeFileLease()
    token = module._current.set(lease)
    try:
        yield lease
    finally:
        module._current.reset(token)
        lease.close()


@contextmanager
def original_only_lease(transport, module, originals):
    """Lease immutable package inputs; always reread newly copied artifacts."""
    lease = module.RuntimeFileLease()
    original_sha = transport.SHA
    def key(path):
        return os.path.normcase(os.path.abspath(path)).removeprefix('\\\\?\\')
    eligible = frozenset(key(path) for path in originals)
    def digest(path):
        # The production runtime uses long-path-normalized filesystem Paths.
        return lease.digest(path) if key(path) in eligible else sha(Path(path))
    transport.SHA = digest
    try:
        yield lease
    finally:
        transport.SHA = original_sha
        lease.close()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--candidate', choices=('full', 'originals'), default='full')
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(runtime))
    import numpy as np
    from braids_gpu import general_live_transport as transport
    from braids_core.sediment import river_runtime_lease as leasing
    assert Path(transport.__file__).resolve().is_relative_to(runtime)
    with np.load(args.records, allow_pickle=False) as archive:
        data = {key: archive[key].item() if archive[key].ndim == 0 else archive[key]
                for key in archive.files}
    transport._validate_prepared(data, integrator='ssp33')
    results = []
    expected = None
    inventory = None
    for index, candidate in enumerate([False, True, True, False] * args.cycles):
        folder = output / f'{index:02d}_{"leased" if candidate else "reference"}.pending'
        start = time.perf_counter()
        if candidate:
            scope = (fresh_lease(leasing) if args.candidate == 'full' else
                     original_only_lease(transport, leasing,
                                         transport._source_files(transport.SPATIAL)))
            with scope as lease:
                receipt = transport.prepare_sources(folder, data)
                statistics_ = dict(reads=lease.reads, hits=lease.hits)
        else:
            receipt = transport.prepare_sources(folder, data)
            statistics_ = {}
        elapsed = time.perf_counter() - start
        # Publication must succeed after all private handles have closed.
        final = folder.with_suffix('')
        folder.rename(final)
        verified = transport.verify_sources(final)
        files = {p.relative_to(final).as_posix(): sha(p)
                 for p in sorted(final.rglob('*')) if p.is_file()}
        if expected is None:
            expected, inventory = receipt, files
        assert receipt == verified == expected
        assert files == inventory
        row = dict(index=index, candidate=candidate, seconds=elapsed,
                   exact_archive=True, published=True, **statistics_)
        results.append(row)
        print(json.dumps(row), flush=True)
    reference = statistics.median(r['seconds'] for r in results if not r['candidate'])
    candidate = statistics.median(r['seconds'] for r in results if r['candidate'])
    report = dict(runtime=str(runtime), records=str(args.records.resolve()),
                  records_sha256=sha(args.records), sources=len(expected['sources']),
                  archive_files=len(inventory), archive_bytes=sum(
                      p.stat().st_size for p in final.rglob('*') if p.is_file()),
                  transport_sha256=sha(Path(transport.__file__)),
                  lease_sha256=sha(Path(leasing.__file__)), results=results,
                  candidate_kind=args.candidate,
                  reference_median_s=reference, candidate_median_s=candidate,
                  speedup=reference/candidate, improvement_pct=100*(1-candidate/reference),
                  schema=1)
    (output/'BENCHMARK.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='results'}), flush=True)


if __name__ == '__main__':
    main()
