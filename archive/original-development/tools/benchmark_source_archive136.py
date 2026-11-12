"""Measure unchanged source checks with independent parallel file operations."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.source_parallel_candidate136 import install


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, nargs='+', default=[1,4,8,16])
    parser.add_argument('--cycles', type=int, default=1)
    args = parser.parse_args()
    out = args.output.resolve();out.mkdir(parents=True, exist_ok=False)
    runtime = args.runtime.resolve();sys.path.insert(0,str(runtime));sys.dont_write_bytecode=True
    import numpy as np
    from braids_gpu import general_live_transport as transport
    assert Path(transport.__file__).resolve().is_relative_to(runtime)
    with np.load(args.records,allow_pickle=False) as archive:
        data={key:archive[key].item() if archive[key].ndim==0 else archive[key]
              for key in archive.files}
    order = (args.workers + list(reversed(args.workers))) * args.cycles
    expected = None; inventory = None; results=[]
    for i,workers in enumerate(order):
        pending=out/f'{i:02d}_w{workers}.pending'
        # Zero selects the genuinely unmodified released function; one is the
        # equivalent serial transform with isolated per-loop measurements.
        scope = nullcontext([]) if workers == 0 else install(transport,workers)
        with scope as phases:
            start=time.perf_counter();receipt=transport.prepare_sources(pending,data)
            elapsed=time.perf_counter()-start
        final=pending.with_suffix('');pending.rename(final)
        assert transport.verify_sources(final)==receipt
        files={p.relative_to(final).as_posix():sha(p)
               for p in sorted(final.rglob('*')) if p.is_file()}
        if expected is None:expected,inventory=receipt,files
        assert receipt==expected and files==inventory
        row=dict(index=i,workers=workers,seconds=elapsed,phases=phases,
                 exact_archive=True,independent_reference_verification=True)
        results.append(row);print(json.dumps(row),flush=True)
    summaries={str(w):dict(median_s=statistics.median(r['seconds'] for r in results if r['workers']==w),
                    copy_median_s=(None if w == 0 else statistics.median(p['seconds'] for r in results if r['workers']==w
                                               for p in r['phases'] if p['phase']=='copy')),
                    verify_median_s=(None if w == 0 else statistics.median(p['seconds'] for r in results if r['workers']==w
                                                 for p in r['phases'] if p['phase']=='verify')))
              for w in args.workers}
    result=dict(schema=1,runtime=str(runtime),records=str(args.records.resolve()),
                records_sha256=sha(args.records),source_count=len(expected['sources']),
                archive_files=len(inventory),transport_sha256=sha(Path(transport.__file__)),
                prototype_sha256=sha(ROOT/'tools/source_parallel_candidate136.py'),
                results=results,summary=summaries)
    (out/'BENCHMARK.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':main()
