"""Equal square-cell hydraulics and isolated OpenMP size-threshold experiments.

Only validation copies change. GeoClaw and Delft3D remain byte-identical.
No performance measurements execute concurrently with another solver.
"""
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

from braids_core.boundaries.cut_cells import map_cut_cells
from braids_core.boundaries.domain import BoundaryAssignment, PolygonDomain
from braids_core.boundaries.inlet_io import NativeGrid
from . import hydraulic_methods as m
from . import hydraulic_performance as h

TAG = 'rectangular_openmp_20260930_v2'
PARENT = h.r.HERE/'runs'/m.TAG
VARIANTS = ('reference', 'hotloops', 'unconditional')
CASES = {
    'small': dict(nx=320, ny=16, dx=.10625, end=5., dt=.01, fast_dt=.05),
    'large': dict(nx=1280, ny=64, dx=.0265625, end=.5, dt=.0025, fast_dt=.0125),
}
DEPTH, VELOCITY, MANNING = .2, .5, 1/36


def paths(tag=TAG):
    return h.r.HERE/'runs'/tag, h.r.HERE/'reports'/tag


def transform(original, variant):
    assert variant in VARIANTS
    if variant == 'reference':
        return original
    text = original
    if variant == 'hotloops':
        for name, condition in (('source', 'nwork*nfrac>=8192'), ('euler', 'nc*nfrac>=8192')):
            before = m.routine(text, name)
            # Only the first Euler occurrence is the water/momentum/CFL loop;
            # the later occurrence belongs to inactive scalar bookkeeping.
            after = before.replace(' if('+condition+')', '', 1)
            assert before != after
            text = m.replace_once(text, before, after)
    else:
        def directive(match):
            line = match[0]
            return re.sub(r' if\([^()]*>=\d+[^()]*\)', '', line)
        text, count = re.subn(r'(?m)^ *!\$omp[^\n]*', directive, text)
        assert count > 30
        text = m.replace_once(text, 'parallel_faces=nwork*nfrac>=8192', 'parallel_faces=.true.')
        assert not re.search(r'(?m)^ *!\$omp.*if\([^()]*>=', text)
    text = m.replace_once(text, ' call load_case(trim(deck))', ' call load_case(trim(deck))'+m.GUARD)
    assert strip_execution_policy(text.replace(m.GUARD, '')) == strip_execution_policy(original)
    assert m.routine(text, 'wave_flux') == m.routine(original, 'wave_flux')
    return text


def strip_execution_policy(text):
    text = re.sub(r'(?m)^ *!\$omp[^\n]*\n?', '', text)
    return re.sub(r'(?m)^ *parallel_faces=.*\n', '', text)


def build(tag):
    root, report = paths(tag)
    root.mkdir(parents=True, exist_ok=False); report.mkdir(parents=True, exist_ok=False)
    _, parent = h.runtime(PARENT, 'reference')
    for source, digest in parent['source_sha256'].items():
        assert h.r.sha256(h.r.ROOT/source) == digest
    shutil.copytree(PARENT/'runtime_reference', root/'runtime_reference')
    for variant in VARIANTS[1:]:
        folder = root/('runtime_'+variant); folder.mkdir()
        hashes = {}
        for name in h.SOURCES:
            source = PARENT/'runtime_reference'/Path(name).name
            target = folder/source.name
            if name == h.SOURCES[-1]:
                old = source.read_text(); new = transform(old, variant)
                target.write_text(new)
                (folder/'threshold.patch').write_text(''.join(difflib.unified_diff(
                    old.splitlines(True),new.splitlines(True),fromfile='reference',tofile=variant)))
            else:
                shutil.copy2(source, target)
            hashes[name] = h.r.sha256(target)
        command = [parent['command'][0], '-O3', '-flto=8', '-fopenmp',
                   '-ffree-line-length-none', '-ffixed-line-length-none',
                   *[str(folder/Path(s).name) for s in h.SOURCES], '-o', str(folder/'xbraids_candidate.exe')]
        start = time.perf_counter()
        proc = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=240)
        (folder/'build.log').write_text(proc.stdout+proc.stderr); proc.check_returncode()
        h.r.write_json(folder/'build.json', dict(command=command, source_sha256=hashes,
            executable_sha256=h.r.sha256(folder/'xbraids_candidate.exe'), variant=variant,
            compile_wall_s=time.perf_counter()-start, parent_executable_sha256=parent['executable_sha256'],
            numerical_formulas_unchanged=True, fixed_mesh_hydraulics_only=True,
            builder_sha256=h.r.sha256(Path(__file__))))
        shutil.copy2(Path(__file__), folder/'builder_snapshot.py')
        print('BUILT '+variant, flush=True)


def reuse_build(tag, previous):
    """Reuse verified executables after an input-writer-only correction."""
    root,report=paths(tag); old,_=paths(previous)
    root.mkdir(parents=True,exist_ok=False); report.mkdir(parents=True,exist_ok=False)
    original=(PARENT/'runtime_reference'/Path(h.SOURCES[-1]).name).read_text()
    for variant in VARIANTS:
        h.runtime(old,variant)
        frozen=old/('runtime_'+variant)/Path(h.SOURCES[-1]).name
        assert frozen.read_text()==transform(original,variant)
        shutil.copytree(old/('runtime_'+variant),root/('runtime_'+variant))
    h.r.write_json(report/'build_reuse.json',dict(previous=previous,
        reason='Delft3D legacy numeric-key columns corrected in input writer; compiled sources unchanged',
        source_verified=True))


def mesh_for(case):
    nx, ny, dx = (case[k] for k in ('nx', 'ny', 'dx'))
    length, width = nx*dx, ny*dx
    domain = PolygonDomain('Uniform rectangular channel', ((0.,0.),(length,0.),(length,width),(0.,width)), 'EPSG:2056')
    assignments = []
    for edge in domain.edges:
        if edge.start_xy[0] == edge.end_xy[0] == 0.:
            assignments.append(BoundaryAssignment(edge.edge_id, 'inlet', 'inlet'))
        if edge.start_xy[0] == edge.end_xy[0] == length:
            assignments.append(BoundaryAssignment(edge.edge_id, 'outlet', 'outlet'))
    assert len(assignments) == 2
    mesh = map_cut_cells(domain, assignments, NativeGrid(nx,ny,0.,0.,dx,dx))
    assert len(mesh.cells) == nx*ny
    assert mesh.metrics()['partial_cell_count'] == 0
    assert len(mesh.faces) == (nx+1)*ny+(ny+1)*nx
    np.testing.assert_allclose([c.area_m2 for c in mesh.cells], dx*dx, rtol=0, atol=1e-12)
    assert np.all(np.diff(mesh.incidence()[0]) == 4)
    return mesh


def depth(x, y):
    # Small 2D disturbance, identical physical function on both grids. Flat bed.
    return DEPTH + .002*np.exp(-((x-13.6)/1.2)**2-((y-.85)/.35)**2)


def record(folder, engine, case, **options):
    h.r.write_json(folder/'input_provenance.json', dict(engine=engine, cells=case['nx']*case['ny'],
        nx=case['nx'], ny=case['ny'], dx_m=case['dx'], end_s=case['end'], output_s=case['end']/2,
        dt_s=case['dt'], Q_m3_s=VELOCITY*DEPTH*case['ny']*case['dx'], manning=MANNING,
        initial_depth_m=DEPTH, initial_velocity_m_s=VELOCITY, initial_perturbation_m=.002,
        sediment=False, secondary_flow=False, viscosity_m2_s=0., bed_updates=False,
        geometry='Identical square cells on [0,34] x [0,1.7] m; no cuts or agglomeration',
        inlet='constant total discharge', outlet='constant stage 0.2 m',
        runner_sha256=h.r.sha256(Path(__file__)),
        files={p.name:h.r.sha256(p) for p in folder.iterdir() if p.is_file() and p.name!='input_provenance.json'},
        **options))


def prepare_native(root, folder, case, mesh, variant):
    folder.mkdir()
    _, identity = h.runtime(root, variant)
    xy = np.array([c.centroid_xy for c in mesh.cells]); x,y = xy.T
    q = np.zeros((len(x),5)); q[:,0] = depth(x,y); q[:,1] = VELOCITY*q[:,0]
    bc = dict(inlet=h.r.CutBoundary('total_discharge', discharge_m3_s=VELOCITY*DEPTH*case['ny']*case['dx']),
              outlet=h.r.CutBoundary('stage', stage_m=DEPTH))
    meta = h.r.write_cut_case(folder/'case.data',mesh,q,np.zeros(len(x)),bc,
        end_time_s=case['end'],output_interval_s=case['end']/2,dt_max_s=case['dt'],cfl=.8,scheme=6,
        dry_tolerance_m=.0001,bedload=0,exchange=0,settling_velocity_m_s=0.,manning=MANNING,
        horizontal_viscosity_m2_s=0.,source_sha256=identity['source_sha256'],compact_sidecars=True)
    mesh.write_geometry(folder/'geometry.npz')
    np.savez_compressed(folder/'analysis_geometry.npz',xy=xy,s=x,n=y,bed=np.zeros(len(x)),
                        area=np.array([c.area_m2 for c in mesh.cells]))
    h.r.write_json(folder/'preparation.json',meta)
    record(folder,'braids',case,variant=variant,cfl=.8)


def prepare_delft(folder, case, dt):
    folder.mkdir()
    nx,ny,dx = (case[k] for k in ('nx','ny','dx'))
    sx=np.arange(nx+1)*dx; sy=np.arange(ny+1)*dx; gx,gy=np.meshgrid(sx,sy)
    x,y=np.meshgrid((sx[1:]+sx[:-1])/2,(sy[1:]+sy[:-1])/2,indexing='ij')
    pad=lambda value:np.pad(value,1,mode='edge')
    w=lambda name,text:h.r.write(folder/name,text)
    w('basin.grd','\n'.join(('Coordinate System = Cartesian',f'{nx+1:8d}{ny+1:8d}',
        '       0       0       0',*h.r.shared.common._grid_rows(gx),*h.r.shared.common._grid_rows(gy))))
    w('basin.enc','\n'.join(f'{i:6d}{j:6d}' for i,j in ((1,1),(nx+2,1),(nx+2,ny+2),(1,ny+2),(1,1))))
    np.savetxt(folder/'basin.dep',-pad(np.zeros_like(x)).T,fmt='%.17e')
    initial=[pad(depth(x,y)),pad(np.full_like(x,VELOCITY)),pad(np.zeros_like(x))]
    np.savetxt(folder/'initial.ini',np.concatenate([a.T for a in initial]),fmt='%.17e')
    w('basin.bnd',f'inlet                T T {1:5d} {2:5d} {1:5d} {ny+1:5d} 0.0 Uniform\n'
        f'outlet               Z T {nx+2:5d} {2:5d} {nx+2:5d} {ny+1:5d} 0.0 Uniform')
    w('basin.bct',h.r.shared.common._boundary_table(1,'inlet','total discharge (t)','m3/s',VELOCITY*DEPTH*ny*dx,case['end']/60)+'\n'+
        h.r.shared.common._boundary_table(2,'outlet','water elevation (z)','m',DEPTH,case['end']/60))
    w('basin.crs',f'inlet                1 2 1 {ny+1}\noutlet               {nx+1} 2 {nx+1} {ny+1}')
    template=PARENT/'input_delft40'
    mdf=(template/'basin.mdf').read_text()
    changes=dict(Runtxt='#Uniform rectangular fixed-bed hydraulics ONLY#', MNKmax=f'{nx+2} {ny+2} 1',
        Dt=f'{dt/60:.17e}',Tstop=f'{case["end"]/60:.17e}',
        Flmap=f'0 {case["end"]/120:.17e} {case["end"]/60:.17e}',
        Flhis=f'0 {case["end"]/120:.17e} {case["end"]/60:.17e}',
        Ccofu=f'{MANNING:.17e}',Ccofv=f'{MANNING:.17e}')
    for key,value in changes.items():
        # Legacy read2r scans numeric values after the six-character key field.
        mdf,count=re.subn(r'(?m)^'+key+r'\s*=.*$',f'{key:<6}= {value}',mdf)
        assert count==1,key
    w('basin.mdf',mdf)
    shutil.copy2(template/'config_d_hydro.xml',folder/'config_d_hydro.xml')
    np.savez_compressed(folder/'analysis_geometry.npz',s=x,n=y,bed=np.zeros_like(x),gx=gx,gy=gy)
    actual=dict(case,dt=dt)
    record(folder,'delft',actual,input_changes=changes,source_modified=False)


def prepare(tag):
    root,report=paths(tag); geometry={}
    for size,case in CASES.items():
        mesh=mesh_for(case)
        for variant in VARIANTS:
            prepare_native(root,root/f'input_{size}_{variant}',case,mesh,variant)
        for name,dt in (('matched',case['dt']),('fast',case['fast_dt'])):
            prepare_delft(root/f'input_{size}_delft_{name}',case,dt)
        geometry[size]=dict(mesh.metrics(),nx=case['nx'],ny=case['ny'],spacing_m=case['dx'],
            incident_faces_per_cell=4,delft_cells=case['nx']*case['ny'],identical_physical_cells=True)
        h.r.write_json(report/'geometry.json',geometry)
        print('PREPARED '+size+' '+str(len(mesh.cells))+' cells',flush=True)


def screen(tag):
    root,report=paths(tag); rows=[]
    for size in CASES:
        for i,threads in enumerate((1,4,8,16,32)):
            order=VARIANTS[i%3:]+VARIANTS[:i%3]
            for variant in order:
                row=h.execute_native(root,root/f'input_{size}_{variant}',f'screen_{size}_{variant}_t{threads}',threads,variant)
                rows.append(dict(row,size=size));h.r.write_json(report/'screen.json',rows)


def benchmark(tag):
    root,report=paths(tag); rows=[]
    # Common thread counts are independent of the noisy single-run screen.
    # Rotate order; compare identical thread counts before selecting any optimum.
    for size in CASES:
        for repeat in range(1,4):
            for threads in (8,32):
                order=VARIANTS[(repeat-1)%3:]+VARIANTS[:(repeat-1)%3]
                for variant in order:
                    label=f'bench_{size}_{variant}_t{threads}_r{repeat}'
                    row=h.execute_native(root,root/f'input_{size}_{variant}',label,threads,variant)
                    rows.append(dict(row,size=size,repeat=repeat));h.r.write_json(report/'benchmark.json',rows)
            for name in (('matched','fast') if repeat%2 else ('fast','matched')):
                label=f'bench_{size}_delft_{name}_r{repeat}'
                row=h.execute_delft(root,root/f'input_{size}_delft_{name}',label,4)
                row['timers']=h.delft_timers(root/label)
                rows.append(dict(row,size=size,repeat=repeat,variant='delft_'+name))
                h.r.write_json(report/'benchmark.json',rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['build','prepare','screen','benchmark','reuse_build'])
    parser.add_argument('--tag',default=TAG)
    parser.add_argument('--previous')
    args=parser.parse_args()
    if args.stage=='reuse_build':
        assert args.previous
        reuse_build(args.tag,args.previous)
    else:
        globals()[args.stage](args.tag)
