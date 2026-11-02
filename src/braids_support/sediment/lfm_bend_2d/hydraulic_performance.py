"""Clean, matched fixed-bed hydraulic benchmarks; locked stock Delft3D only.

No sediment, secondary flow, avalanching, or turbulent closure is active. The
optional constant horizontal viscosity is identical in both input decks. Native
five-state storage has two zero legacy sediment slots, not an active sediment law.
All builds and inputs are frozen under a unique tag, never an installed runtime.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from netCDF4 import Dataset

from . import run_case as r
from .profile_current import parse_profile
from tools.build_cutcell import SOURCES

TAG = 'hydraulics_only_20260930_v1'


def build(root, variant='baseline', reference=None):
    folder = root/('runtime_'+variant)
    folder.mkdir(parents=True, exist_ok=False)
    command = [shutil.which('gfortran') or r'C:\Strawberry\c\bin\gfortran.exe',
               '-O3', '-flto=8', '-fopenmp', '-ffree-line-length-none',
               '-ffixed-line-length-none']
    hashes = {}
    prior=json.loads((reference/'build.json').read_text()) if reference else None
    for name in SOURCES:
        target = folder/Path(name).name
        source=reference/Path(name).name if reference else r.ROOT/name
        if prior:assert r.sha256(source)==prior['source_sha256'][name]
        shutil.copy2(source, target)
        hashes[name] = r.sha256(target)
        command.append(str(target))
    exe = folder/'xbraids_candidate.exe'
    command += ['-o', str(exe)]
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=240)
    (folder/'build.log').write_text(proc.stdout+proc.stderr)
    proc.check_returncode()
    r.write_json(folder/'build.json', dict(command=command,source_sha256=hashes,
        executable_sha256=r.sha256(exe),compile_wall_s=time.perf_counter()-start,
        compiler=subprocess.check_output([command[0],'--version'],text=True).splitlines()[0],
        variant=variant,delft_source_modified=False))
    print('BUILT '+variant, flush=True)


def runtime(root, variant='baseline'):
    folder=root/('runtime_'+variant)
    identity=json.loads((folder/'build.json').read_text())
    exe=folder/'xbraids_candidate.exe'
    assert r.sha256(exe)==identity['executable_sha256']
    for name, expected in identity['source_sha256'].items():
        assert r.sha256(folder/Path(name).name)==expected
    return exe, identity


def record_input(folder, engine, **options):
    record=dict(engine=engine, Q_m3_s=r.e.Q, manning=r.e.MANNING,
        initial_depth_m=r.e.DEPTH, gravity_m_s2=9.81, dry_depth_m=.0001,
        sediment=False, secondary_flow=False, bed_updates=False,
        inlet='constant total discharge',outlet='constant stage 0.2 m',
        initial_velocity_m_s=r.e.Q/r.e.WIDTH/r.e.DEPTH,
        grid_note='Same nominal resolution and geometry; curvilinear vs Cartesian cut cells are not identical meshes',
        files={p.name:r.sha256(p) for p in folder.iterdir()
               if p.is_file() and p.name!='input_provenance.json'},
        runner_sha256=r.sha256(Path(__file__)),**options)
    r.write_json(folder/'input_provenance.json',record)


def prepare_native(root, folder, *, dx=.10625, end=10., output=5., cfl=.4, viscosity=0., variant='baseline'):
    folder.mkdir(parents=True,exist_ok=False)
    _, identity=runtime(root,variant)
    mesh,_=r.native_mesh(dx)
    xy=np.array([c.centroid_xy for c in mesh.cells]);s,n=r.e.sn(*xy.T)
    bed=r.e.initial_bed(s)
    q=np.zeros((len(s),5));q[:,0]=r.e.DEPTH
    theta=np.clip((s-r.e.APPROACH)/r.e.RADIUS,0,r.e.ANGLE)
    q[:,1]=r.e.Q/r.e.WIDTH*np.cos(theta);q[:,2]=-r.e.Q/r.e.WIDTH*np.sin(theta)
    boundaries=dict(inlet=r.CutBoundary('total_discharge',discharge_m3_s=r.e.Q),
                    outlet=r.CutBoundary('stage',stage_m=r.e.DEPTH))
    meta=r.write_cut_case(folder/'case.data',mesh,q,bed,boundaries,
        end_time_s=end,dt_max_s=.0125,output_interval_s=output,cfl=cfl,scheme=6,
        dry_tolerance_m=.0001,bedload=0,exchange=0,settling_velocity_m_s=0,
        manning=r.e.MANNING,horizontal_viscosity_m2_s=viscosity,
        horizontal_viscosity_formulation='velocity_laplacian' if viscosity else 'depth_conservative',
        source_sha256=identity['source_sha256'],compact_sidecars=True)
    mesh.write_geometry(folder/'geometry.npz')
    np.savez_compressed(folder/'analysis_geometry.npz',xy=xy,s=s,n=n,bed=bed,
                        area=np.array([c.area_m2 for c in mesh.cells]))
    r.write_json(folder/'preparation.json',meta)
    record_input(folder,'braids',dx_m=dx,end_s=end,output_s=output,cfl=cfl,
                 viscosity_m2_s=viscosity,cells=len(s),variant=variant)


def prepare_delft(folder, *, dx=.10625, end=10., output=5., viscosity=0., dt=.0125):
    # Reuse only the checked geometric/input writer; remove every optional
    # physics input before execution. dimpro.f90: blank Sub1/Sub2 = no I/C.
    r.prepare_delft(folder,dx=dx,end=end,output=output,dt=dt,morfac=1.,hydraulic=True)
    mdf=(folder/'basin.mdf').read_text()
    changes=dict(Sub1='#    #',Sub2='#   #',Betac='0',Vicouv=str(viscosity),
                 Dpuopt='#MIN#',Runtxt='#LFM fixed-bed hydrodynamics ONLY#')
    for key,value in changes.items():
        mdf,count=re.subn(r'(?m)^'+key+r'\s*=.*$',f'{key} = {value}',mdf)
        assert count==1,key
    # ck_dpopt.f90 maps MOR to MIN even without morphology; MIN preserves
    # the previous face-depth rule explicitly, not a new reference method.
    mdf=re.sub(r'(?m)^(Namc\d+|C\d+|Filsed|Filmor|FilbcC|FmtbcC|Rettis|Rettib)\s*=.*\n','',mdf)
    (folder/'basin.mdf').write_text(mdf)
    geom=np.load(folder/'analysis_geometry.npz');z=geom['bed'];pad=lambda a:np.pad(a,1,mode='edge')
    initial=[pad(z+r.e.DEPTH),pad(np.full_like(z,r.e.Q/r.e.WIDTH/r.e.DEPTH)),pad(np.zeros_like(z))]
    np.savetxt(folder/'initial.ini',np.concatenate([a.T for a in initial]),fmt='%.17e')
    # Generated and never-run sediment files are retained in an explicit
    # unused subdirectory for provenance, not referenced by the hydraulic deck.
    unused=folder/'unused_generated_sediment_inputs';unused.mkdir()
    for name in ('basin.bcc','basin.bcm','basin.sed','basin.mor','formula.tra','input_provenance.json'):
        shutil.move(str(folder/name),str(unused/name))
    record_input(folder,'delft',dx_m=dx,end_s=end,output_s=output,dt_s=dt,
                 viscosity_m2_s=viscosity,cells=z.size,source_modified=False,
                 input_changes=changes,initial_fields=['surface','alongstream_velocity','crossstream_velocity'])


def clone(source, target):
    target.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((source/'input_provenance.json').read_text())
    for name, expected in receipt['files'].items():
        assert r.sha256(source/name)==expected
        shutil.copy2(source/name,target/name)
    shutil.copy2(source/'input_provenance.json',target/'input_provenance.json')


def execute_native(root, source, label, threads, variant='baseline'):
    folder=root/label;clone(source,folder)
    exe,_=runtime(root,variant)
    env=dict(os.environ,OMP_NUM_THREADS=str(threads),OMP_DYNAMIC='FALSE',
             OMP_WAIT_POLICY='PASSIVE',BRAIDS_PROFILE='1',BRAIDS_PROGRESS_STEPS='20000')
    start=time.perf_counter()
    with (folder/'native.log').open('w') as log:
        proc=subprocess.run([str(exe),'case.data'],cwd=folder,env=env,stdout=log,
                            stderr=subprocess.STDOUT,timeout=1800)
    wall=time.perf_counter()-start
    proc.check_returncode()
    profile=parse_profile((folder/'native.log').read_text())
    row=dict(label=label,engine='braids',variant=variant,threads=threads,
        process_wall_s=wall,profile=profile,executable_sha256=r.sha256(exe),
        stepping_wall_s=profile['profile: measured run wall seconds']-profile['profile: result output wall seconds'],
        steps=profile['steps'],rejected=profile['rejected'])
    r.write_json(folder/'execution.json',row)
    print(json.dumps(row),flush=True)
    return row


def execute_delft(root, source, label, ranks):
    folder=root/label;clone(source,folder)
    if ranks==1:
        result=r.shared.delft_runtime.run_case(folder,timeout_s=1800)
    else:
        sys.path.insert(0,str(r.shared.HERE))
        from braids_support.sediment.synthetic_expansion.screen_delft_parallel import execute
        result=execute(folder,ranks,timeout=1800)
    receipt=json.loads((folder/'input_provenance.json').read_text())
    row=dict(label=label,engine='delft',ranks=ranks,
             process_wall_s=result['elapsed_wall_s'],steps=round(receipt['end_s']/receipt['dt_s']),
             runtime=result)
    r.write_json(folder/'hydraulic_timing.json',row)
    # Check the output actually contains only hydraulics, not just MorUpd=false.
    with Dataset(folder/'trim-basin.nc') as d:
        assert not any(k.startswith(('R1','SBTR','SBUU','MSED','MORFT')) for k in d.variables),list(d.variables)
        assert len(d['time'])==round(receipt['end_s']/receipt['output_s'])+1
    print(json.dumps({k:v for k,v in row.items() if k!='runtime'}),flush=True)
    return row


def delft_timers(folder):
    """Stock AddTim output: wall time per MPI partition, NOT summed CPU time."""
    partitions={}
    for path in sorted(folder.glob('tri-diag.basin*')):
        timers={}
        for name,wall,cpu in re.findall(r'^\|([^|]+)\|\s*([\d.Ee+-]+)\s*\|[^|]+\|\s*([\d.Ee+-]+)\s*\|',path.read_text(),re.M):
            timers[name.strip()]=dict(wall_s=float(wall),cpu_s=float(cpu))
        if 'Simulation' in timers:partitions[path.name]=timers
    if not partitions:raise ValueError(f'Missing stock engine timers: {folder}')
    return dict(partitions=partitions,
        simulation_wall_s=max(x['Simulation']['wall_s'] for x in partitions.values()),
        initialization_wall_s=max(x['Initialization']['wall_s'] for x in partitions.values()),
        transport_wall_s=max(x['Transport eq.']['wall_s'] for x in partitions.values()),
        morphology_wall_s=max(x['3D Morphology']['wall_s'] for x in partitions.values()))


def native_comparison(left, right):
    a=r.shared.read_frames(r.shared.native_frames_path(left))
    b=r.shared.read_frames(r.shared.native_frames_path(right))
    np.testing.assert_allclose([f['time_s'] for f in a],[f['time_s'] for f in b])
    qa,qb=[np.stack([f['q'] for f in frames]) for frames in (a,b)]
    assert np.isfinite(qa).all() and np.isfinite(qb).all()
    ha,hb=qa[:,:,0],qb[:,:,0]
    va=qa[:,:,1:3]/np.maximum(ha[:,:,None],1e-12)
    vb=qb[:,:,1:3]/np.maximum(hb[:,:,None],1e-12)
    area=np.load(left/'analysis_geometry.npz')['area']
    water_change=np.sum((hb-hb[0])*area,axis=1)
    ledger=np.stack([f['ledger'] for f in b])
    residual=water_change-ledger[:,:,0].sum(axis=1)
    return dict(state_bitwise_equal=bool(np.array_equal(qa,qb)),
        ledger_bitwise_equal=all(np.array_equal(x['ledger'],y['ledger']) for x,y in zip(a,b)),
        depth_max_m=float(abs(ha-hb).max()),depth_rms_m=float(np.sqrt(np.mean((ha-hb)**2))),
        velocity_component_max_m_s=float(abs(va-vb).max()),
        velocity_rms_m_s=float(np.sqrt(np.mean((va-vb)**2))),
        depth_relative_rms=float(np.sqrt(np.mean((ha-hb)**2))/r.e.DEPTH),
        velocity_relative_rms=float(np.sqrt(np.mean((va-vb)**2))/(r.e.Q/r.e.WIDTH/r.e.DEPTH)),
        water_residual_max_m3=float(abs(residual).max()),minimum_depth_m=float(hb.min()))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','build_equivalent','prepare','screen','optimized_screen','benchmark','delft_dt','scale'])
    p.add_argument('--tag',default=TAG)
    p.add_argument('--baseline-runtime',type=Path,help='Frozen baseline sources/build.json; otherwise snapshot current source')
    args=p.parse_args()
    root=r.HERE/'runs'/args.tag;report=r.HERE/'reports'/args.tag
    if args.stage=='build':
        root.mkdir(parents=True,exist_ok=False);report.mkdir(parents=True,exist_ok=False)
        build(root,reference=args.baseline_runtime);return
    if args.stage=='build_equivalent':
        build(root,'equivalent');return
    if args.stage=='prepare':
        prepare_native(root,root/'input_braids');prepare_delft(root/'input_delft');return
    if args.stage=='screen':
        rows=[]
        for threads in (1,4,8,16,32):
            rows.append(execute_native(root,root/'input_braids',f'screen_b_t{threads}',threads))
            r.write_json(report/'screen.json',rows)
        for ranks in (1,2,4,8):
            label=f'screen_d_p{ranks}'+('_mpi' if ranks>1 else '')
            rows.append(execute_delft(root,root/'input_delft',label,ranks))
            r.write_json(report/'screen.json',rows)
        return
    if args.stage=='optimized_screen':
        rows=[]
        for threads in (4,8,16):
            rows.append(execute_native(root,root/'input_braids',f'screen_equivalent_t{threads}',threads,'equivalent'))
            r.write_json(report/'optimized_screen.json',rows)
        for cfl in (.2,.6,.8):
            source=root/f'input_cfl{cfl:g}'
            prepare_native(root,source,cfl=cfl,variant='equivalent')
            rows.append(execute_native(root,source,f'screen_cfl{cfl:g}',8,'equivalent'))
            r.write_json(report/'optimized_screen.json',rows)
        checks={str(cfl):native_comparison(root/'screen_cfl0.2',root/f'screen_cfl{cfl:g}') for cfl in (.6,.8)}
        checks['0.4']=native_comparison(root/'screen_cfl0.2',root/'screen_equivalent_t8')
        r.write_json(report/'cfl_screen_accuracy.json',checks)
        return
    if args.stage=='delft_dt':
        rows=[]
        for dt in (.00625,.025,.05):
            source=root/f'input_delft_dt{dt:g}'
            prepare_delft(source,end=40.,dt=dt)
            row=execute_delft(root,source,f'delft_dt{dt:g}',4)
            rows.append(dict(row,dt_s=dt));r.write_json(report/'delft_dt.json',rows)
        return
    if args.stage=='scale':
        prepare_native(root,root/'input_large_braids',dx=.053125,end=2.,output=1.)
        prepare_native(root,root/'input_large_cfl08',dx=.053125,end=2.,output=1.,cfl=.8,variant='equivalent')
        prepare_delft(root/'input_large_delft',dx=.053125,end=2.,output=1.,dt=.00625)
        rows=[]
        for repeat,order in enumerate((('baseline','equivalent','cfl0.8','delft'),('delft','cfl0.8','equivalent','baseline')),1):
            for variant in order:
                label=f'large_r{repeat}_{variant}'
                if variant=='delft':
                    row=execute_delft(root,root/'input_large_delft',label,4)
                    row['timers']=delft_timers(root/label)
                else:
                    source=root/('input_large_cfl08' if variant=='cfl0.8' else 'input_large_braids')
                    row=execute_native(root,source,label,16,'baseline' if variant=='baseline' else 'equivalent')
                    row['variant']=variant
                rows.append(dict(row,repeat=repeat));r.write_json(report/'scale.json',rows)
        return
    if args.stage=='benchmark':
        screen=json.loads((report/'screen.json').read_text())
        optimized=json.loads((report/'optimized_screen.json').read_text())
        bt=min((x for x in screen if x['engine']=='braids'),key=lambda x:x['stepping_wall_s'])['threads']
        et=min(optimized[:3],key=lambda x:x['stepping_wall_s'])['threads']
        dp=min((x for x in screen if x['engine']=='delft'),key=lambda x:x['process_wall_s'])['ranks']
        prepare_native(root,root/'input_braids40',end=40.)
        prepare_native(root,root/'input_cfl0.8_40',end=40.,cfl=.8,variant='equivalent')
        prepare_delft(root/'input_delft40',end=40.)
        cfl_check=json.loads((report/'cfl_screen_accuracy.json').read_text())['0.8']
        assert cfl_check['depth_relative_rms']<.01 and cfl_check['velocity_relative_rms']<.02
        rows=[]
        # Sequential runs, rotated engine order; never benchmark alongside
        # builds or other solver runs. Two durations expose fixed startup cost.
        for repeat,order in enumerate((('baseline','equivalent','cfl0.8','delft'),('delft','cfl0.8','equivalent','baseline'),('equivalent','baseline','delft','cfl0.8')),1):
            for seconds in (10,40):
                for variant in order:
                    label=f'paired_r{repeat}_{seconds}_{variant}'
                    suffix='' if seconds==10 else '40'
                    if variant=='delft':
                        row=execute_delft(root,root/('input_delft'+suffix),label,dp)
                    elif variant=='cfl0.8':
                        row=execute_native(root,root/('input_cfl0.8'+('_40' if seconds==40 else '')),label,et,'equivalent')
                        row['variant']='cfl0.8'
                    else:
                        row=execute_native(root,root/('input_braids'+suffix),label,bt if variant=='baseline' else et,variant)
                    rows.append(dict(row,duration_s=seconds,repeat=repeat))
                    r.write_json(report/'benchmark.json',rows)
        return


if __name__=='__main__':main()
