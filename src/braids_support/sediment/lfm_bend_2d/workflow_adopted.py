"""Snapshot and benchmark the actual rebuilt development executable."""
import argparse
import json
from pathlib import Path
import shutil
from . import workflow_production as candidate

p=candidate.p
TAG='workflow_adopted_20260930_v1'
VARIANTS=('reference','adopted')


def paths():return p.paths(TAG)


def snapshot():
    root,report=paths();root.mkdir(exist_ok=False);report.mkdir(exist_ok=False)
    build_dir=p.h.r.ROOT/'solver/cut_cells/build'
    build=json.loads((build_dir/'build.json').read_text())
    original=(candidate.paths()[0]/'runtime_reference/cutcell_solver.f90').read_text()
    assert (p.h.r.ROOT/p.h.SOURCES[-1]).read_text()==candidate.transform(original,'optimized')
    shutil.copytree(candidate.paths()[0]/'runtime_reference',root/'runtime_reference')
    target=root/'runtime_adopted';target.mkdir()
    for name in p.h.SOURCES:
        source=p.h.r.ROOT/name
        assert p.h.r.sha256(source)==build['source_sha256'][name]
        shutil.copy2(source,target/source.name)
    for name in p.h.SOURCES[:-1]:
        assert build['source_sha256'][name]==p.h.runtime(root,'reference')[1]['source_sha256'][name]
    exe=build_dir/'xbraids_cutcell.exe'
    assert p.h.r.sha256(exe)==build['executable_sha256']
    assert p.h.r.sha256(build_dir/'xbraids_cutcell')==build['executable_sha256']
    shutil.copy2(exe,target/'xbraids_candidate.exe')
    p.h.r.write_json(target/'build.json',dict(build,variant='adopted',
        qualification_candidate_sha256=p.h.runtime(candidate.paths()[0],'optimized')[1]['executable_sha256'],
        production_source_adopted=True,installed_plugin_updated=False))
    p.h.runtime(root,'adopted')
    print(build['executable_sha256'])


def benchmark():p.benchmark(TAG,variants=VARIANTS)


def warm_startup():
    """Retain, do not replace, the first batch's anomalous process-start times."""
    root,report=paths();rows=[]
    source=p.p.paths()[0]/'input_small_reference'
    for repeat in range(3):
        order=VARIANTS if repeat%2==0 else VARIANTS[::-1]
        for variant in order:
            row=p.h.execute_native(root,source,f'warm_small_{variant}_r{repeat+1}',8,variant)
            row['outside_engine_s']=row['process_wall_s']-row['profile']['profile: measured run wall seconds']
            rows.append(row);p.h.r.write_json(report/'warm_startup.json',rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('snapshot','benchmark','warm_startup'))
    globals()[parser.parse_args().stage]()
