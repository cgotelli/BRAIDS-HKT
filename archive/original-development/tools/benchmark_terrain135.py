"""Bounded exact terrain-integral benchmark; no hydraulic solver or user writes.

Run only in an agreed clean timing slot. Cold first calls are separate from
the ABBA warm comparisons; frozen alpha134 source is never changed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def load_surface(path,label):
    spec=importlib.util.spec_from_file_location('_terrain135_'+label,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,default=ROOT/'validation/_performance134/current_inputs/cpu8_clock1_output1.json')
    p.add_argument('--before',type=Path,default=ROOT/'dist/unified-alpha134/braids_qgis/river_runtime/braids_qgis/core/raster_surface.py')
    p.add_argument('--after',type=Path,default=ROOT/'braids_qgis/core/raster_surface.py')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,choices=(1,8,32),default=8)
    p.add_argument('--cycles',type=int,default=1,help='ABBA cycles, each two observations per variant')
    p.add_argument('--maximum-error-m',type=float,default=1e-9)
    a=p.parse_args()
    if a.cycles<1 or a.cycles>4:raise ValueError('Use one to four bounded ABBA cycles')
    # Fix ancillary BLAS threading before NumPy import. Native preparation
    # thread count is passed explicitly to the candidate function below.
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    os.environ['OMP_NUM_THREADS']=str(a.threads)
    sys.path.insert(0,str(ROOT))
    import numpy as np
    from braids_core.river_project import load_project,worker_profile
    from braids_core.sediment.adaptive_river_worker import hierarchy

    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    project=load_project(a.project);profile=worker_profile(project)
    began=time.perf_counter();mesh=hierarchy(profile).finest;mesh_s=time.perf_counter()-began
    dem=Path(project['terrain']['dem_path'])
    if not dem.is_absolute():dem=a.project.resolve().parent/dem
    dem=dem.resolve()
    sources={'before':a.before.resolve(),'after':a.after.resolve()}
    identities={key:dict(path=str(path),sha256=sha(path)) for key,path in sources.items()}
    native_paths=[ROOT/'braids_core/boundaries'/name for name in ('raster_quadrature.py','raster_quadrature.h')]
    native_sources={str(path):sha(path) for path in native_paths if path.is_file()}
    modules={key:load_surface(path,key) for key,path in sources.items()}
    rows=[];reference=None;arrays={}
    def measure(variant,phase,order):
        nonlocal reference
        progress=[];extra={'threads':a.threads} if variant=='after' else {}
        began=time.perf_counter()
        values,audit=modules[variant].surface_averages(mesh,dem,progress=progress.append,**extra)
        elapsed=time.perf_counter()-began
        if reference is None:reference=values.copy()
        difference=np.asarray(values)-reference
        record=dict(variant=variant,phase=phase,order=order,wall_s=elapsed,
            maximum_abs_difference_m=float(np.max(np.abs(difference),initial=0.)),
            rms_difference_m=float(np.sqrt(np.mean(difference*difference))),
            array_exact=bool(np.array_equal(values,reference)),audit=audit,
            progress_calls=len(progress),completed_progress=bool(progress and progress[-1]==100.))
        if not np.isfinite(values).all() or record['maximum_abs_difference_m']>a.maximum_error_m:
            raise AssertionError(record)
        rows.append(record);arrays[f'{phase}_{order}_{variant}']=values
        print(phase,order,variant,round(elapsed,6),record['maximum_abs_difference_m'],flush=True)
    for index,variant in enumerate(('before','after')):measure(variant,'first_call',index)
    for index,variant in enumerate(('before','after','after','before')*a.cycles):measure(variant,'warm_abba',index)
    summary={}
    for variant in ('before','after'):
        values=[row['wall_s'] for row in rows if row['variant']==variant and row['phase']=='warm_abba']
        summary[variant]=dict(samples=len(values),median_s=statistics.median(values),minimum_s=min(values),maximum_s=max(values))
    for key,path in sources.items():
        if sha(path)!=identities[key]['sha256']:raise AssertionError('Terrain source changed during benchmark')
    for path,digest in native_sources.items():
        if sha(path)!=digest:raise AssertionError('Native terrain source changed during benchmark')
    np.savez(out/'AVERAGES.npz',**arrays)
    doc=dict(schema='braids.terrain135.benchmark.v1',sources=identities,native_sources=native_sources,
        project=str(a.project.resolve()),project_sha256=sha(a.project),dem=str(dem),dem_sha256=sha(dem),
        requested_candidate_threads=a.threads,ancillary_blas_threads=1,mesh_cells=len(mesh.cells),mesh_construction_s=mesh_s,
        rows=rows,summary=summary,warm_speedup=summary['before']['median_s']/summary['after']['median_s'],
        maximum_accepted_difference_m=a.maximum_error_m,
        scope='Isolated surface_averages on one actual mesh/DEM, including raster load and callback collection; excluding hydraulic computation and full public preparation. First calls recorded separately; warm ABBA is clean, sequential and unprofiled.')
    (out/'BENCHMARK.json').write_text(json.dumps(doc,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
