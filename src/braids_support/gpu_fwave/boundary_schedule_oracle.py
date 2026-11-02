"""Private native boundary-table export and zero-step forcing oracle.

CPU21b7, its original parser/guards and all forcing routines remain unchanged.
This is initialization plus function evaluation, NOT coupled time evolution.
Native-shaped records are read as C-contiguous arrays by coupled_reference.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from . import coupled_reference as cr
from . import cpu_filter_cold20h as parent
from . import curved_wetdry_readiness as util

HERE = Path(__file__).resolve().parent
TAG = 'boundary_schedule_oracle_20261002_v1'
TABLES = {
    'hydro_series_time': ('size(hydro_series_time,1)', 'nb'),
    'hydro_series_q': ('size(hydro_series_q,1)', 'nb'),
    'hydro_series_stage': ('size(hydro_series_stage,1)', 'nb'),
    'series_time': ('size(series_time,1)', 'nb'),
    'series_concentration': ('size(series_concentration,1)', 'nb'),
    'series_bed_input': ('size(series_bed_input,1)', 'nb'),
    'fraction_series_time': ('size(fraction_series_time,1)', 'nfrac', 'nb'),
    'fraction_series_concentration': ('size(fraction_series_concentration,1)', 'nfrac', 'nb'),
    'fraction_series_bed_input': ('size(fraction_series_bed_input,1)', 'nfrac', 'nb'),
}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def export_source():
    text = '''
 subroutine schedule_export_oracle()
  integer :: gpu_unit,input_unit,ns,s
  real(8),allocatable :: sample_time(:),sample_hydraulic(:,:,:),sample_concentration(:,:,:),sample_bed_input(:,:,:)
  real(8),allocatable :: sample_next_knot(:)
  integer,allocatable :: sample_ok(:)
  real(8) :: schedule_bed(nc),original_q(nstate,nc),original_ledger(nledger,nb),original_time
  logical :: schedule_ok
  if(any(section_control/=0)) error stop 'Schedule oracle does not qualify live section roots'
  original_q=q;original_ledger=ledger;original_time=time
  call effective_bed_elevation(q,schedule_bed)
  call gpu_open(gpu_unit,'schedules.bin')
'''
    for name in ('nb', 'nfrac', 'deck_version'):
        text += cr.rec(name, kind='i')
    for name, expr in (('maxhydro', 'size(hydro_series_time,1)'),
                       ('maxscalar', 'size(series_time,1)'),
                       ('maxfraction', 'size(fraction_series_time,1)')):
        text += cr.rec(name, expr, kind='i')
    for name, dims in (('forcing', ('6','nb')), ('fraction_concentration', ('nfrac','nb')),
                       ('fraction_bed_input', ('nfrac','nb'))):
        text += cr.rec(name, dims=dims)
    for name, dims in (('nhydro_series', ('nb',)), ('nseries', ('nb',)),
                       ('nfraction_series', ('nfrac','nb')), ('mode', ('nb',)),
                       ('section_control', ('nb',))):
        text += cr.rec(name, dims=dims, kind='i')
    for name, dims in TABLES.items():
        text += cr.rec(name, dims=dims)
    text += '''  call gpu_finish(gpu_unit)
  open(newunit=input_unit,file='schedule_samples.txt',status='old',action='read')
  read(input_unit,*) ns
  if(ns<1.or.ns>4096) error stop 'Invalid schedule oracle sample count'
  allocate(sample_time(ns),sample_hydraulic(2,nb,ns),sample_concentration(nfrac,nb,ns), &
   sample_bed_input(nfrac,nb,ns),sample_next_knot(ns),sample_ok(ns))
  read(input_unit,*) sample_time
  close(input_unit)
  if(.not.all(ieee_is_finite(sample_time))) error stop 'Nonfinite schedule oracle time'
  do s=1,ns
   call boundary_hydraulic_at(sample_time(s),schedule_bed,sample_hydraulic(:,:,s),schedule_ok)
   sample_ok(s)=merge(1,0,schedule_ok)
   call boundary_sediment_at(sample_time(s),sample_concentration(:,:,s),sample_bed_input(:,:,s))
   sample_next_knot(s)=next_boundary_knot(sample_time(s))
  enddo
  if(any(sample_ok/=1)) error stop 'Native hydraulic schedule evaluation failed'
  if(any(q/=original_q).or.any(ledger/=original_ledger).or.time/=original_time) &
   error stop 'Schedule observation altered accepted state, clock or ledgers'
  call gpu_open(gpu_unit,'schedule_samples.bin')
'''
    for name, expr, dims in (
        ('evaluation_time','sample_time',('ns',)),
        ('hydraulic','sample_hydraulic',('2','nb','ns')),
        ('concentration','sample_concentration',('nfrac','nb','ns')),
        ('bed_input','sample_bed_input',('nfrac','nb','ns')),
        ('next_boundary_knot','sample_next_knot',('ns',)),
    ):
        text += cr.rec(name, expr, dims)
    text += cr.rec('ok', 'sample_ok', ('ns',), 'i')
    text += '''  call gpu_finish(gpu_unit)
 end subroutine
'''
    return text


def transform(source):
    """Only append exports to the frozen export-only instrumentation pattern."""
    baseline = cr.instrument(source)
    anchor = '  call gpu_export_input()\n  stop '
    if baseline.count(anchor) != 1:
        raise ValueError('Unique zero-step export driver required')
    changed = baseline.replace(anchor, '  call gpu_export_input()\n  call schedule_export_oracle()\n  stop ', 1)
    marker = 'end module\n\nprogram run_cutcell'
    if changed.count(marker) != 1:
        raise ValueError('Unique module/program boundary required')
    changed = changed.replace(marker, export_source()+marker, 1)
    # All original forcing/parser/RK bodies must remain text-identical, not a
    # port of the equations or a Python imitation called the native oracle.
    for name in ('boundary_sediment_at', 'boundary_hydraulic_at', 'cpu_ssp43'):
        start = source.index(' subroutine '+name+'(')
        end = source.index('\n end subroutine', start)
        body = source[start:end]
        if changed.count(body) != 1:
            raise ValueError('Original native function changed: '+name)
    return changed


def verify_transform(source, changed):
    if changed != transform(source):
        raise ValueError('Unexpected private source edits')
    restored = changed.replace(export_source(), '', 1).replace('  call schedule_export_oracle()\n', '', 1)
    if restored != cr.instrument(source):
        raise ValueError('Schedule transformation is not exactly reversible')
    return True


def sample_spec():
    knots = [0., .5, 1., 1.25, 1.5, 2., 3.]
    samples = [-.25, 0., .125]
    for t in knots[1:]:
        samples.extend([float(np.nextafter(t, -np.inf)), t, float(np.nextafter(t, np.inf))])
    samples.append(4.)
    stages = []
    for start, dt in ((.25,.5), (.75,.5), (1.25,.5), (2.5,.5)):
        offset = len(samples)
        times = [start, start+.5*dt, start+dt, start+.5*dt]
        samples.extend(times)
        stages.append(dict(start=start, dt=dt, indices=list(range(offset,offset+4)), times=times))
    return dict(times=samples, SSPRK43=stages,
                stage_order=[0.,.5,1.,.5], endpoint_hold=True,
                stage_samples_are_function_queries_not_coupled_steps=True)


def prepare_fixture(name, folder):
    """Small real writer/parser fixtures; no patched decks or disabled guards."""
    from braids_core.boundaries.cut_cell_io import (CutBoundary, CutSedimentFraction,
        CutHydraulicBoundarySeries as Hydro, CutSedimentBoundarySeries as Sediment, write_cut_case)
    from braids_core.boundaries.cut_cells import map_cut_cells
    from braids_core.boundaries.domain import BoundaryAssignment, PolygonDomain
    from braids_core.boundaries.inlet_io import NativeGrid
    domain = PolygonDomain('Boundary schedule oracle', [(0.,0.),(2.,0.),(2.,1.),(0.,1.)], 'EPSG:2056')
    assignments = [BoundaryAssignment(domain.edges[3].edge_id,'in','inlet'),
                   BoundaryAssignment(domain.edges[1].edge_id,'out','outlet')]
    mesh = map_cut_cells(domain, assignments, NativeGrid(4,2,0.,0.,.5,.5))
    if name not in ('constant', 'scalar', 'fraction', 'fraction_partial'):
        raise ValueError('Unknown oracle fixture')
    hydro = Hydro((.5,1.,2.,3.),(.04,.08,.12,.04))
    outlet = CutBoundary('stage', hydraulic_series=Hydro((0.,1.25,3.),(0.,0.,0.),(.08,.09,.07)))
    kw = {}
    if name == 'scalar':
        q = np.zeros((len(mesh.cells),5)); q[:,0]=.1; q[:,1]=.04; q[:,4]=159.
        inlet = CutBoundary('total_discharge',hydraulic_series=hydro,
            sediment_series=Sediment((.5,1.5,3.),(0.,0.,0.),(.001,.003,.002)))
    else:
        q = np.zeros((len(mesh.cells),9)); q[:,0]=.1; q[:,1]=.04; q[:,5:9]=39.75
        fractions = tuple(CutSedimentFraction(fraction_id='sand_'+str(k), d50_m=.0002, d90_m=.0003,
            grain_density_kg_m3=2650., porosity=.4, initial_volume_fraction=.5,
            initial_active_volume_fraction=.5, settling_velocity_m_s=0.) for k in range(2))
        kw.update(sediment_fractions=fractions, active_layer_thickness_m=.05,
                  substrate_layer_count=1, substrate_layer_thickness_m=.05,
                  mpm_friction_mode=1, secondary_flow=True, secondary_flow_model='prognostic',
                  secondary_flow_transport_direction=True, secondary_flow_momentum_beta=1.)
        if name == 'constant':
            inlet = CutBoundary('total_discharge',discharge_m3_s=.08,
                fraction_bed_input_kg_s={'sand_0':.001,'sand_1':.002})
            outlet = CutBoundary('stage',stage_m=.08)
        else:
            series = {'sand_0':Sediment((.5,1.5,3.),(0.,0.,0.),(.001,.003,.002))}
            extra = {}
            if name == 'fraction':
                series['sand_1'] = Sediment((0.,1.,1.25,2.),(0.,0.,0.,0.),(.004,.001,.006,.002))
            else:
                extra['fraction_bed_input_kg_s'] = {'sand_1':.007}
            inlet = CutBoundary('total_discharge',hydraulic_series=hydro,
                                fraction_sediment_series=series, **extra)
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=False)
    write_cut_case(folder/'case.data',mesh,q,np.full(len(q),-.1),{'in':inlet,'out':outlet},
        end_time_s=4.,dt_max_s=.002,output_interval_s=.25,cfl=.4,scheme=6,bedload=2,exchange=0,
        manning=1/36,mpm_critical_shields=.047,morphological_factor=1.,compact_sidecars=True,
        source_sha256={'fixture':util.sha(__file__)},**kw)
    return dict(name=name,input=str(folder/'case.data'),cells=len(q),
                files={str(p):util.sha(p) for p in folder.iterdir() if p.is_file()})


def prepare(tag=TAG):
    root=HERE/'runs'/tag; report=HERE/'reports'/tag
    if root.exists() or report.exists(): raise FileExistsError('Preserve previous attempt')
    identity, protected = parent.runtime_identity()
    if identity['executable_sha256'] != util.CPU_SHA: raise ValueError('Require original CPU21b7')
    root.mkdir(parents=True); report.mkdir(parents=True); runtime=root/'runtime'; runtime.mkdir()
    sources=[]
    for name in identity['source_sha256']:
        original=parent.TIGHT_RUNTIME/name; target=runtime/name
        if name=='cutcell_solver.f90':
            source=original.read_text(); target.write_text(transform(source)); verify_transform(source,target.read_text())
        else: shutil.copy2(original,target)
        sources.append(target)
    for original in parent.TIGHT_RUNTIME.glob('*.dll'):
        shutil.copy2(original,runtime/original.name)
    cases=[prepare_fixture(name,root/'inputs'/name) for name in ('constant','scalar','fraction','fraction_partial')]
    compiler=Path(identity['command'][0]); flags=identity['command'][1:6]
    if flags!=['-O3','-flto','-fopenmp','-ffree-line-length-none','-ffixed-line-length-none']:
        raise ValueError('Original compiler flags changed')
    exe=runtime/'xbraids_schedule_oracle.exe'
    dependencies=[Path(__file__),Path(cr.__file__),Path(parent.__file__),Path(util.__file__)]
    for p in dependencies: shutil.copy2(p,report/p.name)
    plan=dict(schema=1,tag=tag,root=str(root),runtime=str(runtime),executable=str(exe),
        parent_executable_sha256=util.CPU_SHA,protected_files=protected,
        private_files={str(p):util.sha(p) for p in runtime.iterdir() if p.is_file()},
        dependencies={str(p.resolve()):util.sha(p) for p in dependencies},
        compiler_sha256=util.sha(compiler),command=[str(compiler),*flags,*map(str,sources),'-o',str(exe)],
        cases=cases,samples=sample_spec(),tables=list(TABLES),
        arrays='Native dimensions preserved; little-endian float64/int32; cr.read_records gives C-contiguous arrays',
        environment=dict(OMP_NUM_THREADS='8',OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',
                         BRAIDS_GPU_INITIAL_EXPORT='1'),
        numerical_steps=0,live_section_roots_qualified=False,suspended_exchange_qualified=False,
        source_equations_unchanged=True,no_guard_bypasses=True,production_modified=False,Delft3D_modified=False)
    save(report/'plan.json',plan)
    return dict(plan=str(report/'plan.json'),sha256=util.sha(report/'plan.json'))


def verify(plan):
    for mapping in (plan['protected_files'],plan['private_files'],plan['dependencies']):
        for path,digest in mapping.items():
            if util.sha(path)!=digest: raise ValueError('Changed authenticated file: '+path)
    for row in plan['cases']:
        for path,digest in row['files'].items():
            if util.sha(path)!=digest: raise ValueError('Changed fixture: '+path)
    if util.sha(plan['command'][0])!=plan['compiler_sha256']: raise ValueError('Compiler changed')
    verify_transform((parent.TIGHT_RUNTIME/'cutcell_solver.f90').read_text(),
                     (Path(plan['runtime'])/'cutcell_solver.f90').read_text())


def validate_arrays(data):
    nb,ng=int(data['nb']),int(data['nfrac'])
    for prefix,count_name,tail in (('hydro_series','nhydro_series',(nb,)),
        ('series','nseries',(nb,)),('fraction_series','nfraction_series',(ng,nb))):
        count=np.asarray(data[count_name]); table=np.asarray(data[prefix+'_time'])
        if count.shape!=tail or count.dtype!=np.int32 or table.shape[1:]!=tail:
            raise ValueError('Native count/table shape differs')
        if np.any(count<0) or np.any(count==1) or np.any(count>len(table)):
            raise ValueError('Invalid native schedule count')
        for index in np.ndindex(tail):
            values=table[(slice(0,int(count[index])),)+index]
            if np.any(np.diff(values)<=0) or np.any(values<0): raise ValueError('Invalid native knots')
    for name in TABLES:
        a=np.asarray(data[name])
        if a.dtype!=np.float64 or not a.flags.c_contiguous or not np.isfinite(a).all():
            raise ValueError('Native table dtype/layout/finite check failed')
    return True


def expected(data, times):
    """Independent NumPy oracle for validating exported native evaluations."""
    times=np.asarray(times,dtype=np.float64); nb,ng=data['nb'],data['nfrac']
    h=np.broadcast_to(data['forcing'][:2,:,None],(2,nb,len(times))).copy()
    c=np.broadcast_to(data['fraction_concentration'][:,:,None],(ng,nb,len(times))).copy()
    b=np.broadcast_to(data['fraction_bed_input'][:,:,None],(ng,nb,len(times))).copy()
    for boundary in range(nb):
        n=int(data['nhydro_series'][boundary])
        if n:
            for field,dest in (('q',h[0,boundary]),('stage',h[1,boundary])):
                dest[:]=np.interp(times,data['hydro_series_time'][:n,boundary],data['hydro_series_'+field][:n,boundary])
        n=int(data['nseries'][boundary])
        if n:
            for field,dest in (('concentration',c[0,boundary]),('bed_input',b[0,boundary])):
                dest[:]=np.interp(times,data['series_time'][:n,boundary],data['series_'+field][:n,boundary])
        for grain in range(ng):
            n=int(data['nfraction_series'][grain,boundary])
            if n:
                for field,dest in (('concentration',c[grain,boundary]),('bed_input',b[grain,boundary])):
                    dest[:]=np.interp(times,data['fraction_series_time'][:n,grain,boundary],
                                     data['fraction_series_'+field][:n,grain,boundary])
    return dict(hydraulic=h,concentration=c,bed_input=b)


def execute(plan_path, approved):
    plan_path=Path(plan_path)
    if util.sha(plan_path)!=approved: raise ValueError('Exact reviewed plan required')
    plan=json.loads(plan_path.read_text()); verify(plan)
    receipt=plan_path.parent/'execution.json'
    if receipt.exists(): raise FileExistsError('Preserve earlier execution')
    result=dict(status='running',plan_sha256=approved,cases=[],numerical_steps=0)
    save(receipt,result)
    try:
        p=subprocess.run(plan['command'],cwd=plan['runtime'],capture_output=True,text=True,timeout=300)
        (Path(plan['runtime'])/'build.log').write_text(p.stdout+p.stderr);p.check_returncode()
        result['executable_sha256']=util.sha(plan['executable'])
        env={k:v for k,v in os.environ.items() if not k.upper().startswith(('OMP_','BRAIDS_'))}
        env.update(plan['environment'])
        for row in plan['cases']:
            folder=Path(plan['root'])/row['name']; copied=cr.copy_inputs(row['input'],folder)
            times=np.array(plan['samples']['times'],dtype=np.float64)
            (folder/'schedule_samples.txt').write_text(str(len(times))+'\n'+' '.join(format(x,'.17g') for x in times)+'\n')
            p=subprocess.run([plan['executable'],'case.data'],cwd=folder,env=env,capture_output=True,text=True,timeout=60,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            (folder/'native.log').write_text(p.stdout+p.stderr);p.check_returncode()
            data=cr.read_records(folder/'schedules.bin'); samples=cr.read_records(folder/'schedule_samples.bin')
            validate_arrays(data)
            if not np.array_equal(samples['evaluation_time'],times) or not np.all(samples['ok']==1):
                raise ValueError('Native sample times/status differ')
            comparison={key:float(np.max(np.abs(value-samples[key]),initial=0.)) for key,value in expected(data,times).items()}
            if any(error>5e-16 for error in comparison.values()): raise ValueError('Native interpolation differs: '+str(comparison))
            for group in plan['samples']['SSPRK43']:
                for key in ('hydraulic','concentration','bed_input'):
                    if not np.array_equal(samples[key][...,group['indices'][1]],samples[key][...,group['indices'][3]]):
                        raise ValueError('Repeated half-time values differ')
            np.savez_compressed(folder/'schedules.npz',**data)
            np.savez_compressed(folder/'schedule_samples.npz',**samples)
            result['cases'].append(dict(name=row['name'],folder=str(folder),cells=row['cells'],
                deck_version=int(data['deck_version']),nfrac=int(data['nfrac']),nb=int(data['nb']),
                nhydro_series=data['nhydro_series'].tolist(),nseries=data['nseries'].tolist(),
                nfraction_series=data['nfraction_series'].tolist(),max_error=comparison,
                input_identity=copied,artifacts={p.name:util.sha(p) for p in folder.iterdir() if p.is_file()},
                sample_count=len(times),source_equations_unchanged=True))
            save(receipt,result)
        verify(plan);result.update(status='completed',passed=True,live_section_roots_qualified=False,
            component_only=True,source_and_array_identities_retained=True)
    except Exception as exc:
        result.update(status='failed',error=repr(exc));raise
    finally: save(receipt,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('prepare');a.add_argument('--tag',default=TAG)
    a=s.add_parser('execute');a.add_argument('--plan',type=Path,required=True);a.add_argument('--approved-sha',required=True)
    a=p.parse_args()
    print(json.dumps(prepare(a.tag) if a.action=='prepare' else execute(a.plan,a.approved_sha),indent=2))
