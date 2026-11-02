"""Isolated hydraulic-method experiments; NEVER edit/install production sources.

GeoClaw augmented Riemann sources, wave_flux, boundary closures, Euler/FCT,
sources, wall reconstruction and sediment routines remain byte-for-byte equal
to the frozen parent. Only BRAIDS's fixed-mesh RK wrapper and/or regular-interior
reconstruction change. Unsupported coupled physics/AMR fail explicitly.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

from . import hydraulic_performance as h
from .qualify_hydraulic_cfl import fixture

TAG = 'hydraulic_methods_20260930_v3'
PARENT = h.r.HERE/'runs'/h.TAG/'runtime_equivalent'
VARIANTS = ('reference', 'rk2', 'muscl', 'combined')


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:90]
    return text.replace(old, new, 1)


def routine(text, name):
    match = re.search(r'(?im)^ *(?:subroutine|function) '+name+r'\(', text)
    assert match, name
    end = re.search(r'(?im)^ *end (?:subroutine|function)\b[^\n]*', text[match.start():])
    return text[match.start():match.start()+end.end()]


GUARD = """
 ! Research-only fixed-bed path. Do not silently change coupled/AMR methods.
 if(amr_enabled.or.bedlaw/=0.or.exchange_mode/=0.or.secondary_flow.or. &
    bank_failure.or.bank_avalanche.or.general_avalanche.or.bedform_roughness.or. &
    bedload_adaptation.or.bed_slope/=0.or.horizontal_viscosity/=0.or. &
    nstate/=5.or.nfrac/=1.or.any(ws_fraction/=0.d0).or.any(q(4,:)/=0.d0).or. &
    any(forcing(3:4,:)/=0.d0).or.any(series_concentration/=0.d0).or. &
    any(series_bed_input/=0.d0).or.any(fraction_concentration/=0.d0).or. &
    any(fraction_bed_input/=0.d0).or.any(fraction_series_concentration/=0.d0).or. &
    any(fraction_series_bed_input/=0.d0)) &
    error stop 'hydraulic research build requires fixed mesh and inactive auxiliary physics'
"""


def transform(original, variant):
    assert variant in VARIANTS
    if variant == 'reference':
        return original
    text = replace_once(original, ' call load_case(trim(deck))',
                        ' call load_case(trim(deck))'+GUARD)
    changed = set()
    if variant in ('rk2', 'combined'):
        before = routine(text, 'advance')
        after = before.replace('toe_flux_weight=1.d0/6.d0', 'toe_flux_weight=.5d0')
        start = after.index('  !$omp parallel do schedule(static,256)')
        end = after.index('  call rebalance_active_layer(c)', start)
        after = after[:start]+"""  ! SSPRK(2,2), using the same conservative Euler operator twice.
  !$omp parallel do schedule(static,256) if(nc*nstate>=131072) default(shared) private(target)
  do target=1,nc
   c(:,target)=.5d0*q(:,target)+.5d0*b(:,target)
  enddo
  !$omp end parallel do
"""+after[end:]
        after = replace_once(after, 'ledger=ledger+(fa+fb)/6.d0+2.d0*fc/3.d0',
                             'ledger=ledger+.5d0*(fa+fb)')
        text = replace_once(text, before, after)
        changed.add('advance')
    if variant in ('muscl', 'combined'):
        before = routine(text, 'face_value')
        after = replace_once(before, 'value(m)=reconstruct(stencil(:,m),5)',
            '''if(m<=3.or.m==bed_primitive_index()) then
    value(m)=reconstruct(stencil(:,m),2)
   else
    value(m)=reconstruct(stencil(:,m),5)
   endif''')
        text = replace_once(text, before, after)
        before = routine(text, 'face_elevations')
        after = before.replace('reconstruct(stencil(:,1),5)','reconstruct(stencil(:,1),2)')
        after = after.replace('reconstruct(stencil(:,2),5)','reconstruct(stencil(:,2),2)')
        text = replace_once(text, before, after)
        # Scheme 6 caches ordinary interior WENO in this routine as well as
        # actual reflected/triangular wall values. Keep the latter on WENO5.
        before = routine(text, 'prepare_wall_traces')
        after = replace_once(before,'integer :: c,a,side,slot,k,kk,s,kind,m,f,other',
            'integer :: c,a,side,slot,k,kk,s,kind,m,f,other,reconstruction_scheme')
        after = replace_once(after,'private(c,a,side,slot,k,kk,s,kind,m,f,other,',
            'private(c,a,side,slot,k,kk,s,kind,m,f,other,reconstruction_scheme,')
        after = replace_once(after,'    if(.not.wet) cycle',
            '''    if(.not.wet) cycle
    reconstruction_scheme=5
    if(.not.wall_triangle(c).and.all(wall_stencil_kind(:,a,c)==0)) reconstruction_scheme=2''')
        after = replace_once(after,'wall_values(m,slot,c)=reconstruct(stencil(:,m),5)',
            '''if(m<=3.or.m==bed_primitive_index()) then
       wall_values(m,slot,c)=reconstruct(stencil(:,m),reconstruction_scheme)
      else
       wall_values(m,slot,c)=reconstruct(stencil(:,m),5)
      endif''')
        text = replace_once(text, before, after)
        changed.update(('face_value', 'face_elevations', 'prepare_wall_traces'))
    # Exhaustive routine-level protection, not just selected named kernels.
    names = re.findall(r'(?im)^ *(?:subroutine|function) (\w+)\(', original)
    for name in names:
        if name not in changed:
            assert routine(original, name) == routine(text, name), name
    return text


def paths(tag):
    return h.r.HERE/'runs'/tag, h.r.HERE/'reports'/tag


def build(tag):
    root, report = paths(tag)
    root.mkdir(parents=True, exist_ok=False); report.mkdir(parents=True, exist_ok=False)
    _, parent = h.runtime(PARENT.parent, 'equivalent')
    for variant in VARIANTS:
        folder = root/('runtime_'+variant); folder.mkdir()
        hashes = {}
        for name in h.SOURCES:
            source = PARENT/Path(name).name
            assert h.r.sha256(source) == parent['source_sha256'][name]
            target = folder/source.name
            shutil.copy2(source, target)
            if name == h.SOURCES[-1] and variant != 'reference':
                old = source.read_text(); new = transform(old, variant)
                target.write_text(new)
                (folder/'experiment.patch').write_text(''.join(difflib.unified_diff(
                    old.splitlines(True), new.splitlines(True), fromfile='frozen_reference', tofile=variant)))
            hashes[name] = h.r.sha256(target)
            if name != h.SOURCES[-1]:
                assert hashes[name] == parent['source_sha256'][name]
        exe = folder/'xbraids_candidate.exe'
        reuse = h.r.HERE/'runs'/'hydraulic_methods_20260930_v2'/('runtime_'+variant)
        if variant in ('reference','rk2') and (reuse/'build.json').exists():
            previous_exe, previous = h.runtime(reuse.parent,variant)
            assert previous['source_sha256']==hashes
            shutil.copy2(previous_exe,exe)
            h.r.write_json(folder/'build.json',dict(previous,reused_from=str(reuse),
                builder_sha256=h.r.sha256(Path(__file__))))
            print('REUSED identical '+variant,flush=True)
            continue
        command = [shutil.which('gfortran') or r'C:\Strawberry\c\bin\gfortran.exe',
            '-O3', '-flto=8', '-fopenmp', '-ffree-line-length-none', '-ffixed-line-length-none',
            *[str(folder/Path(s).name) for s in h.SOURCES], '-o', str(exe)]
        start = time.perf_counter()
        result = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=240)
        (folder/'build.log').write_text(result.stdout+result.stderr); result.check_returncode()
        h.r.write_json(folder/'build.json', dict(variant=variant, command=command,
            source_sha256=hashes, executable_sha256=h.r.sha256(exe),
            parent_executable_sha256=parent['executable_sha256'],
            compile_wall_s=time.perf_counter()-start, core_solver_unchanged=True,
            builder_sha256=h.r.sha256(Path(__file__)),
            compiler=subprocess.check_output([command[0], '--version'],text=True).splitlines()[0],
            research_only=True, installed=False, delft_source_modified=False))
        print('BUILT '+variant, flush=True)


def screen(tag):
    root, report = paths(tag); rows = []
    # Same output/cfl/threads; physics preparation is outside timed sections.
    for variant in VARIANTS:
        h.prepare_native(root, root/('input_'+variant), cfl=.8, variant=variant)
    for variant in VARIANTS:
        rows.append(h.execute_native(root, root/('input_'+variant), 'screen_'+variant, 8, variant))
        h.r.write_json(report/'screen.json', rows)
    checks = {v:h.native_comparison(root/'screen_reference', root/('screen_'+v)) for v in VARIANTS[1:]}
    h.r.write_json(report/'screen_accuracy.json', checks)
    print(json.dumps(checks), flush=True)


def physics(tag):
    root, report = paths(tag); rows = []
    for kind in ('steep_wave', 'dry_dam_break', 'cutcell_lake'):
        mesh, q, bed, bc, end, ref, core, scale = fixture(kind)
        area = np.array([c.area_m2 for c in mesh.cells]); small = None
        for variant, cfl in [('reference', .2), *[(v, .8) for v in VARIANTS]]:
            exe, identity = h.runtime(root, variant)
            folder = root/f'physics_{kind}_{variant}_{cfl:g}'; folder.mkdir()
            h.r.write_cut_case(folder/'case.data', mesh, q, bed, bc, end_time_s=end,
                dt_max_s=1., output_interval_s=end, cfl=cfl, scheme=6,
                bedload=0, exchange=0, settling_velocity_m_s=0,
                dry_tolerance_m=.0001, manning=0, source_sha256=identity['source_sha256'])
            result = subprocess.run([str(exe),'case.data'], cwd=folder, capture_output=True,
                text=True, timeout=120, env=dict(os.environ, OMP_NUM_THREADS='4',
                OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',BRAIDS_PROFILE='1'))
            (folder/'native.log').write_text(result.stdout+result.stderr); result.check_returncode()
            final = h.r.shared.read_frames(folder/'frames.txt')[-1]; state = final['q']
            if small is None: small = state.copy()
            error = float(np.sum(abs(state[core,0]-ref[core]))/max(scale,1e-30))
            diff = float(np.sum(abs(state[core,0]-small[core,0]))/max(scale,1e-30))
            budget = float(np.dot(area,state[:,0]-q[:,0])-final['ledger'][:,0].sum())
            row = dict(case=kind, variant=variant, cfl=cfl, depth_relative_L1=error,
                vs_small_step_relative_L1=diff, water_residual_m3=budget,
                minimum_depth_m=float(state[:,0].min()),
                max_momentum=float(abs(state[:,1:3]).max()),
                profile=h.parse_profile(result.stdout), executable_sha256=identity['executable_sha256'],
                case_sha256=h.r.sha256(folder/'case.data'),
                passed=bool(error<.02 and diff<.01 and abs(budget)<1e-8 and
                            np.isfinite(state).all() and state[:,0].min()>=0))
            if kind == 'cutcell_lake': row['passed'] &= bool(abs(state-q).max()<1e-10)
            np.savez(folder/'reference.npz',initial=q,reference_depth=ref,core=core,area=area)
            rows.append(row); h.r.write_json(report/'physics.json',rows)
            print(json.dumps({k:v for k,v in row.items() if k!='profile'}),flush=True)


def benchmark(tag):
    root, report = paths(tag); rows = []
    for variant in VARIANTS:
        h.prepare_native(root, root/('input40_'+variant), end=40., cfl=.8, variant=variant)
        h.prepare_native(root, root/('input_large_'+variant), dx=.053125, end=2., output=1., cfl=.8, variant=variant)
    for repeat, order in enumerate((VARIANTS, tuple(reversed(VARIANTS)),
                                   ('muscl','reference','combined','rk2')),1):
        for size, threads in (('40',8), ('_large',16)):
            for variant in order:
                row = h.execute_native(root,root/('input'+size+'_'+variant),
                                       f'bench{size}_r{repeat}_{variant}',threads,variant)
                rows.append(dict(row,size=size,repeat=repeat))
                h.r.write_json(report/'benchmark.json',rows)
    # Repeat the already accuracy-screened stock dt=.05 comparison; no source changes.
    h.prepare_delft(root/'input_delft40',end=40.,dt=.05)
    for repeat in range(1,4):
        label=f'delft40_r{repeat}'
        row=h.execute_delft(root,root/'input_delft40',label,4)
        row['timers']=h.delft_timers(root/label)
        rows.append(dict(row,size='40',repeat=repeat,variant='delft'))
        h.r.write_json(report/'benchmark.json',rows)


def threads(tag):
    root, report=paths(tag);rows=[];checks={}
    for variant in ('reference','rk2'):
        source=root/('thread_input_'+variant)
        h.prepare_native(root,source,end=10.,cfl=.8,variant=variant)
        for count in (1,4,8,16,32):
            label=f'threads_{variant}_{count}'
            rows.append(h.execute_native(root,source,label,count,variant))
            if count!=1:
                checks[f'{variant}_{count}']=h.native_comparison(root/f'threads_{variant}_1',root/label)
            h.r.write_json(report/'threads.json',dict(rows=rows,checks=checks))


def convergence(tag):
    """Same exact smooth simple-wave family, cell-average errors on three grids."""
    from braids_support.sediment.reservoir_polygon.geometry_resolution import slope_wave as wave
    from braids_core.boundaries.domain import PolygonDomain,BoundaryAssignment
    from braids_core.boundaries.inlet_io import NativeGrid
    from braids_core.boundaries.cut_cells import map_cut_cells
    root,report=paths(tag);rows=[]
    for dx in (20.,10.,5.):
        domain=PolygonDomain('wave',[(-800,0),(1200,0),(1200,1),(-800,1)],'EPSG:2056')
        assign=[BoundaryAssignment(domain.edges[3].edge_id,'inlet','inlet'),
                BoundaryAssignment(domain.edges[1].edge_id,'outlet','outlet')]
        mesh=map_cut_cells(domain,assign,NativeGrid(round(2000/dx),1,-800,0,dx,1))
        x=np.array([c.centroid_xy[0] for c in mesh.cells]);q=np.zeros((len(x),5))
        q[:,0],q[:,1]=wave.means(x,dx,0)
        _,_,_,bc,end,_,_,_=fixture('steep_wave')
        ref,_=wave.means(x,dx,end);core=(x>=150)&(x<=650)
        for variant in VARIANTS:
            exe,identity=h.runtime(root,variant);folder=root/f'convergence_{variant}_{dx:g}';folder.mkdir()
            h.r.write_cut_case(folder/'case.data',mesh,q,-wave.S*x,bc,end_time_s=end,
                dt_max_s=1,output_interval_s=end,cfl=.8,scheme=6,bedload=0,exchange=0,
                settling_velocity_m_s=0,dry_tolerance_m=.0001,manning=0,
                source_sha256=identity['source_sha256'])
            proc=subprocess.run([str(exe),'case.data'],cwd=folder,capture_output=True,text=True,
                timeout=120,env=dict(os.environ,OMP_NUM_THREADS='1',OMP_DYNAMIC='FALSE',
                OMP_WAIT_POLICY='PASSIVE',BRAIDS_PROFILE='1'))
            (folder/'native.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
            state=h.r.shared.read_frames(folder/'frames.txt')[-1]['q']
            error=float(np.sum(abs(state[core,0]-ref[core]))/np.sum(abs(ref[core]-wave.H)))
            np.savez(folder/'reference.npz',x=x,depth=ref,initial=q[:,0],core=core)
            rows.append(dict(variant=variant,dx_m=dx,depth_pulse_relative_L1=error,
                profile=h.parse_profile(proc.stdout),case_sha256=h.r.sha256(folder/'case.data'),
                executable_sha256=identity['executable_sha256']))
            h.r.write_json(report/'convergence.json',rows)
            print(variant,dx,error,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','screen','physics','benchmark','threads','convergence'])
    p.add_argument('--tag',default=TAG);args=p.parse_args()
    globals()[args.stage](args.tag)
