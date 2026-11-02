"""Build and run an isolated BRAIDS curvature-regularization candidate.

Production sources/runtime and all Delft3D sources/runtime remain unchanged.
"""
import argparse
import json
import math
import re
import warnings
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from . import run_case as r
from . import analyze_runs as a
from tools import build_cutcell as build

MODULE=r.HERE/'native/secondary_curvature_regularization.f90'
FAST_MODULE=r.HERE/'native/secondary_curvature_fast.f90'
MULTILEVEL_MODULE=r.HERE/'native/secondary_curvature_hybrid.f90'
EXACT_MODULE=r.HERE/'native/secondary_curvature_exact.f90'
PARALLEL_MODULE=r.HERE/'native/secondary_curvature_parallel.f90'
PRODUCTION_MODULE='solver/wave/braids_curvature_regularization_module.f90'


def candidate_source(source, factor, *, filter_rtol=None):
    # The alpha-81 runtime already has the tested filter. Replace that one
    # integration point when building a research variant; never apply twice.
    use=' use secondary_curvature_regularization, only: regularize_curvature\n'
    if use in source:
        block=(r'  filter_raw=secondary_curvature\n'
               r'  call regularize_curvature\(nc,nf,owner,neighbour,area,xy,normal,length,state\(1,:\), &\n'
               r'   dry_tolerance,[^\n]+\n'
               r"  if\(\.not\.ok\) error stop 'secondary curvature regularization failed'\n")
        source,count=re.subn(block,'',source)
        if count!=1 or source.count(use)!=1:
            raise ValueError('Production curvature integration context changed')
        source=source.replace(use,'').replace(
            '  integer :: work,c,nwork,invalid,si,filter_iterations\n'
            '  real(8) :: filter_raw(nc),filter_residual\n',
            '  integer :: work,c,nwork,invalid,si\n')
    tolerance_args=''
    if filter_rtol is not None:
        if not math.isfinite(filter_rtol) or not 0 < filter_rtol <= .01:
            raise ValueError('Filter relative tolerance must be finite and in (0, 0.01]')
        literal=lambda x:format(x,'.16e').replace('e','d')
        tolerance_args=f',relative_tolerance={literal(filter_rtol)},absolute_tolerance={literal(max(1e-16,filter_rtol*1e-6))}'
    changes={
        ' use geoclaw_module, only:':
            ' use secondary_curvature_regularization, only: regularize_curvature\n use geoclaw_module, only:',
        '  integer :: work,c,nwork,invalid,si\n  real(8) :: h,u,v,equilibrium,time_a,updated,increment':
            '  integer :: work,c,nwork,invalid,si,filter_iterations\n'
            '  real(8) :: filter_raw(nc),filter_residual\n'
            '  real(8) :: h,u,v,equilibrium,time_a,updated,increment',
        '  call secondary_flow_curvatures(state,secondary_curvature)\n  secondary_source_work=0.d0;':
            '  call secondary_flow_curvatures(state,secondary_curvature)\n'
            '  filter_raw=secondary_curvature\n'
            '  call regularize_curvature(nc,nf,owner,neighbour,area,xy,normal,length,state(1,:), &\n'
            f'   dry_tolerance,{float(factor):.8f}d0,filter_raw,secondary_curvature,filter_iterations,filter_residual,ok{tolerance_args})\n'
            "  if(.not.ok) error stop 'secondary curvature regularization failed'\n"
            '  secondary_source_work=0.d0;'}
    for old,new in changes.items():
        if source.count(old)!=1:raise ValueError(f'Candidate source context changed: {old}')
        source=source.replace(old,new)
    return source


def compile_candidate(tag, factor, *, optimized=True, exact_order=False, parallel_filter=False,
                      filter_rtol=None, second_order_bedload=False, multilevel_filter=False):
    if exact_order and not optimized:
        raise ValueError('Choose either the original reference or the exact-order optimized filter')
    if parallel_filter and (exact_order or not optimized):
        raise ValueError('Parallel research filter cannot be combined with a reference-filter selection')
    practical = optimized and not exact_order and not parallel_filter
    if multilevel_filter and not practical:
        raise ValueError('The size-aware multilevel filter cannot be combined with a reference, exact-order or research parallel filter')
    if filter_rtol is not None and not practical:
        raise ValueError('Explicit tolerance is supported by the MIC filter, not the historical/reference alternatives')
    if practical and filter_rtol is None:filter_rtol=1e-4
    runtime=r.HERE/'runs'/f'{tag}_runtime'
    runtime.mkdir(parents=True,exist_ok=False)
    override=runtime/'cutcell_solver.f90'
    original=(r.ROOT/'solver/cut_cells/cutcell_solver.f90').read_text()
    source=candidate_source(original,factor,filter_rtol=filter_rtol)
    if second_order_bedload:
        from .performance_speedfirst import muscl_source
        warnings.warn('Experimental sediment MUSCL2: bend regression passes, but coupled outlet bed-wave errors are 31-48%; not a generally qualified method.',RuntimeWarning)
        source=muscl_source(source)
    from .performance_auxiliary import hydraulic_guard
    guard=hydraulic_guard(source,original)
    r.write(override,source)
    module=(EXACT_MODULE if exact_order else (MULTILEVEL_MODULE if multilevel_filter else FAST_MODULE)) if optimized else MODULE
    if parallel_filter:module=PARALLEL_MODULE
    module_copy=runtime/module.name
    shutil.copy2(module,module_copy)
    names=[*[name for name in build.SOURCES[:-1] if name!=PRODUCTION_MODULE],
           str(module.relative_to(r.ROOT)),build.SOURCES[-1]]
    sources=[override if name==build.SOURCES[-1] else
             module_copy if name==str(module.relative_to(r.ROOT)) else r.ROOT/name for name in names]
    # A validation runtime must remain reproducible after production evolves.
    # Freeze every dependency, not only the patched main program and filter.
    frozen=[]
    for source in sources:
        target=runtime/source.name
        if source.resolve()!=target.resolve():shutil.copy2(source,target)
        frozen.append(target)
    sources=frozen
    exe=runtime/'xbraids_candidate.exe'
    command=[shutil.which('gfortran'),'-O3','-flto','-fopenmp','-ffree-line-length-none',
             '-ffixed-line-length-none',*[str(p) for p in sources],'-o',str(exe)]
    hashes={name:r.sha256(path) for name,path in zip(names,sources)}
    compiled=subprocess.run(command,cwd=runtime,capture_output=True,text=True,timeout=120)
    r.write(runtime/'build.log',compiled.stdout+compiled.stderr);compiled.check_returncode()
    if hashes!={name:r.sha256(path) for name,path in zip(names,sources)}:
        raise RuntimeError('Sources changed during candidate compilation')
    identity=dict(source_sha256=hashes,executable_sha256=r.sha256(exe),command=command,
                  depth_factor=factor,role='Unpromoted native curvature regularization candidate',
                  curvature_linear_solver=('Packed cached Jacobi' if exact_order else 'MIC(0)-PCG') if optimized else 'Reference Jacobi',
                  curvature_relative_tolerance=filter_rtol if practical else (1e-13 if parallel_filter else 1e-10),
                  curvature_absolute_area_scaled_tolerance=max(1e-16,filter_rtol*1e-6) if practical else (1e-16 if parallel_filter else 1e-14),
                  bedload_reconstruction='Bounded least-squares MUSCL2' if second_order_bedload else 'WENO / bounded least-squares',
                  bedload_reconstruction_qualification='Experimental: fails coupled bed-wave regressions' if second_order_bedload else 'Retained reference reconstruction',
                  hydrodynamic_routines_unchanged=guard)
    if parallel_filter:identity['curvature_linear_solver']='Explicit parallel alternatives: BRAIDS_CURVATURE_MODE=matvec or block'
    identity['multilevel_filter']=multilevel_filter
    if multilevel_filter:
        identity.update(curvature_linear_solver='Size-aware MIC / symmetric aggregation V-cycle PCG',
            multilevel_minimum_cells=65536,multilevel_team_limits={'below_131072':4,'below_524288':8,'otherwise':16},
            selection_independent_of_thread_count=True,
            qualification='Measured bend and synthetic-grid performance at ell=2h; other configured factors need separate qualification, not a QGIS production-physics promotion')
    r.write_json(runtime/'build.json',identity)
    return exe,identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('tag');p.add_argument('--factor',type=float,default=2.)
    p.add_argument('--end',type=float,default=60.)
    p.add_argument('--dx',type=float,default=.10625)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--cold',action='store_true')
    p.add_argument('--runtime-tag')
    p.add_argument('--thread-check',action='store_true')
    p.add_argument('--filter-rtol',type=float,help='MIC relative residual target (default 1e-4; use 1e-13 for the historical strict solve)')
    p.add_argument('--second-order-bedload',action='store_true',help='EXPERIMENTAL: sediment MUSCL2; fails coupled bed-wave regressions. Hydraulic WENO and SSPRK3 unchanged.')
    filters=p.add_mutually_exclusive_group()
    filters.add_argument('--reference-filter',action='store_true',help='Build the preserved original Jacobi filter')
    filters.add_argument('--exact-order-filter',action='store_true',help='Build v4 packed Jacobi for bitwise reference comparisons instead of MIC-PCG')
    filters.add_argument('--parallel-filter',action='store_true',help='Build research parallel alternatives; select BRAIDS_CURVATURE_MODE=matvec or block')
    filters.add_argument('--multilevel-filter',action='store_true',help='Use the measured size-aware MIC/multilevel solver for the existing experimental curvature filter; useful on larger meshes.')
    args=p.parse_args()
    report=r.HERE/'reports'/args.tag;report.mkdir(parents=True,exist_ok=False)
    if args.runtime_tag:
        runtime=r.HERE/'runs'/f'{args.runtime_tag}_runtime'
        exe=runtime/'xbraids_candidate.exe';identity=json.loads((runtime/'build.json').read_text())
        if r.sha256(exe)!=identity['executable_sha256']:raise ValueError('Runtime hash mismatch')
        if args.factor!=identity['depth_factor']:raise ValueError('Candidate factor mismatch')
        if args.multilevel_filter and not identity.get('multilevel_filter',identity.get('variant')=='hybrid'):
            raise ValueError('Requested multilevel filter does not match the frozen runtime')
    else:exe,identity=compile_candidate(args.tag,args.factor,optimized=not args.reference_filter,exact_order=args.exact_order_filter,
                                       parallel_filter=args.parallel_filter,filter_rtol=args.filter_rtol,second_order_bedload=args.second_order_bedload,
                                       multilevel_filter=args.multilevel_filter)
    r.write_json(report/'build.json',identity)
    original=r.shared.verified_braids_build
    try:
        r.shared.verified_braids_build=lambda:(exe,identity)
        folders=[]
        for threads in ([1,8] if args.thread_check else [args.threads]):
            folder=r.HERE/'runs'/f'{args.tag}_t{threads}'
            r.prepare_native(folder,dx=args.dx,end=args.end,output=args.end/6,dt=.0125,morfac=1,
                hydraulic=True,seed=None if args.cold else r.HERE/'runs/fine3600_m1_v1_hydraulic_braids')
            provenance=json.loads((folder/'input_provenance.json').read_text())
            provenance.update(engine_source_modified=True,production_source_modified=False,
                candidate='Helmholtz regularization of curvature forcing; not promoted',
                curvature_regularization_depth_factor=args.factor)
            r.write_json(folder/'input_provenance.json',provenance)
            folders.append((folder,threads))
        def one(item):
            folder,threads=item
            r.write_json(report/f'status_t{threads}.json',dict(status='running',folder=str(folder)))
            try:
                result=r.execute_native(folder,threads)
                metrics=a.audit(a.load(folder))
                row=dict(execution=result,metrics=metrics,candidate_only=True,delft_source_modified=False)
                r.write_json(report/f'metrics_t{threads}.json',row)
                r.write_json(report/f'status_t{threads}.json',dict(status='complete',folder=str(folder)))
                print(json.dumps(metrics,indent=2),flush=True)
                return row
            except Exception as exc:
                r.write_json(report/f'status_t{threads}.json',dict(status='failed',error=str(exc)))
                raise
        with ThreadPoolExecutor(max_workers=len(folders)) as pool:list(pool.map(one,folders))
    finally:r.shared.verified_braids_build=original


if __name__=='__main__':main()
