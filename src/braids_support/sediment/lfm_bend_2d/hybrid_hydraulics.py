"""Isolated alternative hydraulic fluxes on the frozen SSP43 candidate.

No production changes. Delft3D is read-only. All methods keep the same SSP43,
Euler CFL 0.8, WENO reconstruction, sediment closures and source integration.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import numpy as np
from . import larger_timesteps as p
from . import hydraulic_performance as h

TAG='hybrid_hydraulics_20261001_v2'
VARIANTS=('reference','hlle','fwave')
PARENT=p.paths()[0]/'runtime_ssp43'


def paths(tag=TAG):return p.paths(tag)


def transform(original,variant):
    method=VARIANTS.index(variant)
    text=p.replace(original,' implicit none\n integer :: nc,nf',f''' implicit none
 integer,parameter :: research_flux_method={method}
 integer(kind=8) :: research_fast_calls=0,research_fallback_calls=0
 integer :: nc,nf''')
    before=p.methods.routine(text,'euler');after=before
    after=p.replace(after,'  logical :: valid,parallel_faces',
        '  integer(kind=8) :: fast_count,fallback_count\n  logical :: used_fast\n  logical :: valid,parallel_faces')
    after=p.replace(after,'  secondary_invalid=0','  secondary_invalid=0;fast_count=0;fallback_count=0')
    after=p.replace(after,'stress_active_left,stress_active_right) reduction(max:secondary_invalid)',
        'stress_active_left,stress_active_right,used_fast) reduction(max:secondary_invalid) reduction(+:fast_count,fallback_count)')
    after=p.replace(after,'''     call wave_flux(l,r,left(bed_primitive_index()),right(bed_primitive_index()), &
      normal(:,f),fl(1:3,f),fr(1:3,f),speed(f))''','''     call research_flux(l,r,left(bed_primitive_index()),right(bed_primitive_index()), &
      normal(:,f),fl(1:3,f),fr(1:3,f),speed(f),used_fast)
     if(used_fast) then
      fast_count=fast_count+1
     else
      fallback_count=fallback_count+1
     endif''')
    after=p.replace(after,"  if(secondary_invalid/=0) error stop 'invalid prognostic secondary-flow face closure'",
        "  research_fast_calls=research_fast_calls+fast_count;research_fallback_calls=research_fallback_calls+fallback_count\n  if(secondary_invalid/=0) error stop 'invalid prognostic secondary-flow face closure'")
    text=p.replace(text,before,after)
    text=p.replace(text,' subroutine profile_report()',
        (p.r.HERE/'native/hybrid_flux.inc').read_text()+'\n subroutine profile_report()')
    # Insert executable statements after profile_report's declarations.
    report=p.methods.routine(text,'profile_report')
    anchor='  if(.not.profile_enabled) return'
    report=p.replace(report,anchor,anchor+'''
  write(*,'(a,i0)') 'research: fast interior flux calls=',research_fast_calls
  write(*,'(a,i0)') 'research: fallback interior flux calls=',research_fallback_calls''')
    text=p.replace(text,p.methods.routine(text,'profile_report'),report)
    for name in re.findall(r'(?im)^ *(?:subroutine|function) (\w+)\(',original):
        if name not in ('euler','profile_report'):
            assert p.methods.routine(text,name)==p.methods.routine(original,name),name
    return text


def build(tag):
    root,report=paths(tag);root.mkdir(parents=True,exist_ok=False);report.mkdir(parents=True,exist_ok=False)
    parent=json.loads((PARENT/'build.json').read_text());p.runtime(PARENT.parent,'ssp43')
    shutil.copy2(p.paths()[1]/'production_before.json',report/'production_before.json')
    p.r.write_json(report/'plan.json',dict(variants=VARIANTS,limits=p.LIMITS,
        minimum_compute_speedup=1.05,phase='screen, independent physics, repeated timings and developed bed continuation',
        baseline='Frozen SSP43 at Euler CFL 0.8; no credit for the preceding RK3-to-SSP43 gain',
        parent_runtime_sha256=parent['executable_sha256'],production_promoted=False,delft_modified=False))
    for variant in VARIANTS:
        folder=root/('runtime_'+variant);folder.mkdir();hashes={}
        for name in p.build_cutcell.SOURCES:
            source=PARENT/Path(name).name;target=folder/source.name
            assert p.r.sha256(source)==parent['source_sha256'][name]
            if name==p.build_cutcell.SOURCES[-1]:target.write_text(transform(source.read_text(),variant))
            else:shutil.copy2(source,target)
            hashes[name]=p.r.sha256(target)
        exe=folder/'xbraids_candidate.exe'
        command=[shutil.which('gfortran') or r'C:\Strawberry\c\bin\gfortran.exe','-O3','-flto','-fopenmp',
                 '-ffree-line-length-none','-ffixed-line-length-none',
                 *[str(folder/Path(s).name) for s in p.build_cutcell.SOURCES],'-o',str(exe)]
        started=time.perf_counter();proc=subprocess.run(command,cwd=folder,capture_output=True,text=True,timeout=240)
        (folder/'build.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
        p.r.write_json(folder/'build.json',dict(variant=variant,command=command,source_sha256=hashes,
            executable_sha256=p.r.sha256(exe),parent_executable_sha256=parent['executable_sha256'],
            compile_wall_s=time.perf_counter()-started,production_promoted=False,delft_modified=False))
        print('BUILT '+variant,flush=True)


def counters(folder,row):
    log=(folder/'native.log').read_text()
    for key,kind in (('fast_calls','fast'),('fallback_calls','fallback')):
        match=re.search('research: '+kind+r' interior flux calls=(\d+)',log)
        assert match;row[key]=int(match[1])
    row['fast_fraction']=row['fast_calls']/max(row['fast_calls']+row['fallback_calls'],1)
    p.r.write_json(folder/'execution.json',row)
    return row


def qualified_variants(report):
    checks=json.loads((report/'screen.json').read_text())['checks']
    return ('reference',)+tuple(v for v in VARIANTS[1:]
        if all(checks[f'{case}_r1_{v}']['passed'] for case in ('developed','recent')))


def bend(tag,stage):
    root,report=paths(tag);rows=[];checks={}
    duration={'screen':2,'benchmark':6,'long':24}[stage];repeats=3 if stage=='benchmark' else 1
    cases=('developed','recent') if stage!='benchmark' else ('developed',)
    variants=VARIANTS if stage=='screen' else qualified_variants(report)
    for case in cases:
        for repeat in range(repeats):
            labels={};shift=repeat%len(variants);order=variants[shift:]+variants[:shift]
            for variant in order:
                label=f'{stage}_{case}_r{repeat+1}_{variant}';labels[variant]=label
                row=p.bend_run(root,label,variant,.8,p.SOURCES[case],duration)
                rows.append(counters(root/label,row))
                p.r.write_json(report/(stage+'.json'),dict(rows=rows,checks=checks))
            for variant in variants[1:]:checks[f'{case}_r{repeat+1}_{variant}']=p.compare(root/labels['reference'],root/labels[variant])
            p.r.write_json(report/(stage+'.json'),dict(rows=rows,checks=checks))
            print(json.dumps({k:v['passed'] for k,v in checks.items()}),flush=True)


def fixture(kind):
    if kind!='smooth_subcritical':return p.physics.fixture(kind)
    from braids_core.boundaries.domain import PolygonDomain
    from braids_core.boundaries.inlet_io import NativeGrid
    from braids_core.boundaries.cut_cells import map_cut_cells
    domain=PolygonDomain(kind,[(0,0),(20,0),(20,.2),(0,.2)],'EPSG:2056')
    mesh=map_cut_cells(domain,[],NativeGrid(400,2,0,0,.05,.1))
    x=np.array([c.centroid_xy[0] for c in mesh.cells]);eps=1e-5;c=np.sqrt(9.81)
    from numpy.polynomial.legendre import leggauss
    nodes,weights=leggauss(16);xx=x[:,None]+.025*nodes
    pulse=lambda time:eps*np.exp(-.5*((xx-5-c*time)/.6)**2)@weights/2
    q=np.zeros((len(x),5));q[:,0]=1+pulse(0);q[:,1]=c*pulse(0)
    truth=1+pulse(1);core=(x>3)&(x<14)
    return mesh,q,np.zeros(len(x)),{},1.,truth,core,float(abs(truth[core]-1).sum())


def physics(tag):
    root,report=paths(tag);rows=[]
    for kind in ('smooth_subcritical','steep_wave','dry_dam_break','cutcell_lake'):
        mesh,q,bed,bc,end,truth,core,scale=fixture(kind);area=np.array([c.area_m2 for c in mesh.cells]);small=None
        for variant,cfl in [('reference',.2),*[(v,.8) for v in VARIANTS]]:
            folder=root/f'physics_{kind}_{variant}_{cfl:g}';folder.mkdir()
            p.r.write_cut_case(folder/'case.data',mesh,q,bed,bc,end_time_s=end,dt_max_s=1.,output_interval_s=end,
                cfl=.4,scheme=6,bedload=0,exchange=0,settling_velocity_m_s=0,dry_tolerance_m=.0001,manning=0,
                source_sha256={'test':'hybrid-hydraulic-physics'})
            row=counters(folder,p.execute(folder,p.runtime(root,variant),cfl,threads=4))
            frames,state=p.fields(folder);state=state[-1]
            if small is None:small=state.copy()
            error=float(abs(state[core,0]-truth[core]).sum()/scale)
            diff=float(abs(state[core,0]-small[core,0]).sum()/scale)
            budget=float(area@(state[:,0]-q[:,0])-frames[-1]['ledger'][:,0].sum())
            row.update(case=kind,variant=variant,cfl=cfl,analytic_depth_L1=error,small_step_depth_L1=diff,
                water_budget_m3=budget,minimum_depth_m=float(state[:,0].min()),
                passed=bool(error<.02 and diff<.01 and abs(budget)<1e-8 and np.isfinite(state).all() and state[:,0].min()>=0))
            if kind=='cutcell_lake':row['passed'] &= bool(abs(state-q).max()<1e-10)
            np.savez(folder/'analytic.npz',truth=truth,core=core,initial=q)
            rows.append(row);p.r.write_json(report/'physics.json',rows)


def hydraulic_bench(tag):
    root,report=paths(tag);rows=[];checks={}
    variants=qualified_variants(report)
    for size,dx,end,threads in (('small',.10625,20,8),('large',.053125,2,16)):
        for variant in variants:h.prepare_native(root,root/(f'hydraulic_input_{size}_'+variant),dx=dx,end=end,output=end/4,cfl=.4,variant=variant)
        for repeat in range(3):
            shift=repeat%len(variants)
            for variant in variants[shift:]+variants[:shift]:
                label=f'hydraulic_{size}_r{repeat+1}_{variant}';folder=root/label
                h.clone(root/(f'hydraulic_input_{size}_'+variant),folder)
                rows.append(counters(folder,p.execute(folder,p.runtime(root,variant),.8,threads=threads)))
            for variant in variants[1:]:
                left=root/f'hydraulic_{size}_r{repeat+1}_reference';right=root/f'hydraulic_{size}_r{repeat+1}_{variant}'
                check=h.native_comparison(left,right)
                check['passed']=bool(check['depth_rms_m']<=p.LIMITS['depth_rms_m'] and check['velocity_rms_m_s']<=p.LIMITS['velocity_rms_m_s']
                    and check['water_residual_max_m3']<=p.LIMITS['water_budget_m3'] and check['minimum_depth_m']>=0)
                checks[f'{size}_r{repeat+1}_{variant}']=check
            p.r.write_json(report/'hydraulic.json',dict(rows=rows,checks=checks))


def same_binary(tag):
    """Distinguish algorithmic effects from separately compiled binary layouts."""
    import os
    root,report=paths(tag);folder=root/'runtime_switch';folder.mkdir()
    base=root/'runtime_fwave';identity=json.loads((base/'build.json').read_text());p.runtime(root,'fwave')
    hashes={}
    for name in p.build_cutcell.SOURCES:
        source=base/Path(name).name;target=folder/source.name
        if name==p.build_cutcell.SOURCES[-1]:
            code=p.replace(source.read_text(),' integer,parameter :: research_flux_method=2',' integer :: research_flux_method=2')
            anchor=' ! Recorded research setting; not a production restart contract.'
            code=p.replace(code,anchor,''' call get_environment_variable('BRAIDS_RESEARCH_FLUX',profile_setting,status=environment_status)
 if(environment_status/=0) error stop 'same-binary control requires BRAIDS_RESEARCH_FLUX'
 read(profile_setting,*,iostat=ios) research_flux_method
 if(ios/=0) error stop 'invalid research flux method'
 if(research_flux_method/=0.and.research_flux_method/=2) error stop 'research flux method must be 0 or 2'
 write(*,'(a,i0)') 'research: selected flux method=',research_flux_method
'''+anchor)
            target.write_text(code)
        else:shutil.copy2(source,target)
        hashes[name]=p.r.sha256(target)
    exe=folder/'xbraids_candidate.exe'
    command=[shutil.which('gfortran'),'-O3','-flto','-fopenmp','-ffree-line-length-none','-ffixed-line-length-none',
             *[str(folder/Path(s).name) for s in p.build_cutcell.SOURCES],'-o',str(exe)]
    proc=subprocess.run(command,cwd=folder,capture_output=True,text=True,timeout=240)
    (folder/'build.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
    p.r.write_json(folder/'build.json',dict(source_sha256=hashes,executable_sha256=p.r.sha256(exe),command=command,
        parent_executable_sha256=identity['executable_sha256'],production_promoted=False,delft_modified=False))
    rows=[];checks={};old=os.environ.get('BRAIDS_RESEARCH_FLUX')
    try:
        for repeat in range(3):
            for mode in ((0,2) if repeat%2==0 else (2,0)):
                os.environ['BRAIDS_RESEARCH_FLUX']=str(mode)
                label=f'same_binary_r{repeat+1}_{mode}'
                row=p.bend_run(root,label,'switch',.8,p.SOURCES['developed'],6)
                row['environment']['BRAIDS_RESEARCH_FLUX']=str(mode)
                rows.append(counters(root/label,row))
            checks[f'r{repeat+1}']=p.compare(root/f'same_binary_r{repeat+1}_0',root/f'same_binary_r{repeat+1}_2')
            p.r.write_json(report/'same_binary.json',dict(rows=rows,checks=checks))
    finally:
        if old is None:os.environ.pop('BRAIDS_RESEARCH_FLUX',None)
        else:os.environ['BRAIDS_RESEARCH_FLUX']=old


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['build','screen','benchmark','long','physics','hydraulic','same_binary'])
    parser.add_argument('--tag',default=TAG);args=parser.parse_args()
    if args.stage=='build':build(args.tag)
    elif args.stage=='physics':physics(args.tag)
    elif args.stage=='hydraulic':hydraulic_bench(args.tag)
    elif args.stage=='same_binary':same_binary(args.tag)
    else:bend(args.tag,args.stage)


if __name__=='__main__':main()
