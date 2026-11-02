"""Real mode9/control1 coupled fixtures and an export-only CPU21b7 initializer.

The original parser, boundary sorting, seed checks and numerical routines are
unchanged. This new exporter writes tables/seed metadata without invoking the
older schedule-only oracle's deliberate no-live-sections guard. Full reference
evolution, when requested, uses the original executable, not this exporter.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np

from . import coupled_reference as cr
from . import boundary_schedule_oracle as schedules
from . import cpu_filter_cold20h as parent

HERE=Path(__file__).resolve().parent
TAG='normal_coupled_oracle_20261002_v1'
ROOT=HERE/'runs'/TAG
RUNTIME=ROOT/'runtime'
sha=cr.r.sha
ENV=dict(OMP_NUM_THREADS='8',OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',
         BRAIDS_TIME_INTEGRATOR='ssp43',BRAIDS_PROFILE='1')


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def extra_export():
    text="\n subroutine normal_export_metadata()\n  integer :: gpu_unit\n"
    text+="  call gpu_open(gpu_unit,'normal_metadata.bin')\n"
    for key in ('nb','nfrac','deck_version'):
        text+=cr.rec(key,kind='i')
    for key in ('nhydro_series','nseries','mode','section_control'):
        text+=cr.rec(key,dims=('nb',),kind='i')
    text+=cr.rec('nfraction_series',dims=('nfrac','nb'),kind='i')
    # Keep raw parser values in this file. Conversion belongs to read_case,
    # with both versions retained to make the two different index spaces clear.
    for key in ('section_seed_node','boundary_seed_position'):
        text+=cr.rec('native_'+key,key,('nb',),'i')
    text+=cr.rec('section_energy_slope',dims=('nb',))
    for key,dims in schedules.TABLES.items():text+=cr.rec(key,dims=dims)
    text+='  call gpu_finish(gpu_unit)\n end subroutine\n'
    return text


def transform(source):
    original=cr.instrument(source)
    anchor='  call gpu_export_input()\n  stop '
    marker='end module\n\nprogram run_cutcell'
    if original.count(anchor)!=1 or original.count(marker)!=1:
        raise ValueError('Unexpected zero-step export anchors')
    changed=original.replace(anchor,'  call gpu_export_input()\n  call normal_export_metadata()\n  stop ',1)
    return changed.replace(marker,extra_export()+marker,1)


def verify_transform(source,changed):
    restored=changed.replace(extra_export(),'',1).replace('  call normal_export_metadata()\n','',1)
    if changed!=transform(source) or restored!=cr.instrument(source):
        raise ValueError('Only the declared export additions are allowed')
    for name in ('load_case','section_nodes','live_section_stage','boundary_hydraulic_at','traces'):
        start=source.index(' subroutine '+name+'(')
        end=source.index('\n end subroutine',start)
        if changed.count(source[start:end])!=1:raise ValueError('Native numerical/parser body changed: '+name)
    return True


def verify_files(mapping):
    for path,digest in mapping.items():
        if sha(path)!=digest:raise ValueError('Changed identity: '+str(path))


def build(runtime=RUNTIME):
    runtime=Path(runtime).resolve()
    if runtime.exists():return verify_build(runtime)
    identity,protected=parent.runtime_identity()
    runtime.mkdir(parents=True)
    sources=[]
    for name in identity['source_sha256']:
        original=parent.TIGHT_RUNTIME/name;target=runtime/name
        if name=='cutcell_solver.f90':
            source=original.read_text();changed=transform(source)
            verify_transform(source,changed);target.write_text(changed)
        else:shutil.copy2(original,target)
        sources.append(str(target))
    for path in parent.TIGHT_RUNTIME.glob('*.dll'):shutil.copy2(path,runtime/path.name)
    dependencies=[Path(__file__),Path(cr.__file__),Path(schedules.__file__),Path(parent.__file__)]
    for path in dependencies:shutil.copy2(path,runtime/path.name)
    flags=identity['command'][1:6]
    if flags!=['-O3','-flto','-fopenmp','-ffree-line-length-none','-ffixed-line-length-none']:
        raise ValueError('Parent flags differ')
    compiler=Path(identity['command'][0]);exe=runtime/'xbraids_normal_export.exe'
    command=[str(compiler),*flags,*sources,'-o',str(exe)]
    process=subprocess.run(command,cwd=runtime,capture_output=True,text=True,timeout=300)
    (runtime/'build.log').write_text(process.stdout+process.stderr);process.check_returncode()
    record=dict(schema=1,runtime=str(runtime),executable=str(exe),command=command,
        executable_sha256=sha(exe),parent_executable_sha256=parent.TIGHT_SHA,
        protected_files=protected,dependencies={str(p.resolve()):sha(p) for p in dependencies},
        compiler_sha256=sha(compiler),compiler_dlls={str(p):sha(p) for p in compiler.parent.glob('*.dll')},
        private_files={str(p):sha(p) for p in runtime.iterdir() if p.is_file()},
        source_equations_unchanged=True,parser_sorting_seed_validation_unchanged=True,
        transformation='cr.instrument I/O hooks plus zero-step raw tables/metadata exporter; no schedule oracle evaluation or guard change',
        evolution_runtime=str(parent.TIGHT_EXE),production_modified=False,Delft3D_modified=False)
    save(runtime/'build.json',record);return verify_build(runtime)


def verify_build(runtime=RUNTIME):
    runtime=Path(runtime).resolve();record=json.loads((runtime/'build.json').read_text())
    parent.runtime_identity()
    for key in ('protected_files','dependencies','compiler_dlls','private_files'):verify_files(record[key])
    verify_files({record['command'][0]:record['compiler_sha256']})
    verify_transform((parent.TIGHT_RUNTIME/'cutcell_solver.f90').read_text(),(runtime/'cutcell_solver.f90').read_text())
    return record


def fixture(name):
    """128 Cartesian cells, eight true sampled section faces, mild curved flow."""
    from braids_core.boundaries.cut_cell_io import (CutBoundary,CutHydraulicBoundarySeries as Hydro,
        CutSedimentBoundarySeries as Sediment,CutSedimentFraction)
    from braids_core.boundaries.cut_cells import map_cut_cells
    from braids_core.boundaries.domain import BoundaryAssignment,PolygonDomain
    from braids_core.boundaries.inlet_io import NativeGrid
    from braids_core.boundaries.section import Section
    if name not in ('prescribed','equilibrium'):raise ValueError('Unknown coupled fixture')
    domain=PolygonDomain('Live normal coupled '+name,[(0.,0.),(4.,0.),(4.,2.),(0.,2.)],'EPSG:2056')
    assignments=[BoundaryAssignment(domain.edges[3].edge_id,'in','inlet'),BoundaryAssignment(domain.edges[1].edge_id,'out','outlet')]
    mesh=map_cut_cells(domain,assignments,NativeGrid(16,8,0.,0.,.25,.25))
    xy=np.array([c.centroid_xy for c in mesh.cells]);stations=np.linspace(0.,2.,9)
    def elevation(x,y):return np.interp(y,[0.,.75,1.25,2.],[.12,0.,0.,.12])-.002*x
    bed=elevation(xy[:,0],xy[:,1]);faces=[]
    for f in mesh.faces:
        if f.bc_id=='in':
            z0,z1=(float(elevation(*point)) for point in (f.start_xy,f.end_xy))
            centre=.5*(np.array(f.start_xy)+f.end_xy)
            f=replace(f,bed_elevation_m=float(elevation(*centre)),section_bed_start_m=z0,section_bed_end_m=z1)
        faces.append(f)
    mesh=replace(mesh,faces=tuple(faces))
    section=Section('in',stations,elevation(0.,stations),0.,(1.,0.))
    q=np.zeros((len(xy),9));q[:,0]=np.maximum(.14-.002*xy[:,0]-bed,0.)
    q[:,1]=.35*q[:,0];q[:,2]=.015*np.sin(np.pi*xy[:,0]/4)*np.sin(np.pi*xy[:,1]/2)*q[:,0]
    q[:,5:9]=39.75;rock=bed-.1
    fractions=tuple(CutSedimentFraction(fraction_id='sand_'+str(k),d50_m=.0002,d90_m=.0003,
        grain_density_kg_m3=2650.,porosity=.4,initial_volume_fraction=.5,
        initial_active_volume_fraction=.5,settling_velocity_m_s=0.) for k in range(2))
    feed={}
    if name=='prescribed':
        feed['fraction_sediment_series']={f'sand_{k}':Sediment((0.,.02,.04),(0.,0.,0.),(.0005,.0006,.0005)) for k in range(2)}
    else:feed['bedload_boundary_mode']='equilibrium_capacity'
    inlet=CutBoundary('inlet_normal',energy_slope=.002,boundary_manning_n=.03,
        hydraulic_series=Hydro((0.,.02,.04),(.08,.09,.08)),**feed)
    return dict(mesh=mesh,q=q,rock=rock,bed=bed,xy=xy,fractions=fractions,section=section,
        boundaries={'in':inlet,'out':CutBoundary('stage',stage_m=.132)})


def prepare_fixture(name,folder):
    from braids_core.boundaries import cut_cell_io as writer
    data=fixture(name);folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=False)
    writer.write_cut_case(folder/'case.data',data['mesh'],data['q'],data['rock'],data['boundaries'],
        sections={'in':data['section']},end_time_s=.04,dt_max_s=.001,output_interval_s=.005,
        cfl=.4,scheme=6,bedload=2,exchange=0,manning=.03,mpm_friction_mode=1,mpm_critical_shields=.047,
        sediment_fractions=data['fractions'],active_layer_thickness_m=.05,substrate_layer_count=1,
        substrate_layer_thickness_m=.05,secondary_flow=True,secondary_flow_model='prognostic',
        secondary_flow_transport_direction=True,secondary_flow_momentum_beta=1.,morphological_factor=1.,
        compact_sidecars=True,source_sha256={'fixture':sha(__file__),'writer':sha(writer.__file__)})
    np.savez(folder/'initial_analysis.npz',q=data['q'],rock=data['rock'],bed=data['bed'],xy=data['xy'],
        area=np.array([c.area_m2 for c in data['mesh'].cells]))
    record=dict(name=name,input=str(folder/'case.data'),cells=len(data['q']),inlet_faces=8,
        duration=.04,dt=.001,output_times_s=(np.arange(9)*.005).tolist(),
        inlet='Live normal mode9/control1, eight all-pool sampled faces, positive Manning',
        outlet='Stage mode2; explicit no-reversal policy1',bed_is_mobile=True,
        sediment='Two identical sand labels with .05 m active plus .05 m substrate; no suspended exchange',
        initial_secondary='hI initially zero, nonuniform transverse velocity; prognostic source enabled',
        numerical_scope='Short component/coupled port check; not steady normal-flow, morphology or experimental validation',
        source_files={str(Path(__file__).resolve()):sha(__file__),str(Path(writer.__file__).resolve()):sha(writer.__file__)},
        files={str(p):sha(p) for p in folder.iterdir() if p.is_file()})
    save(folder/'fixture.json',record);return record


def read_case(folder):
    folder=Path(folder);data=cr.read_snapshot(folder);extra=cr.read_records(folder/'normal_metadata.bin')
    schedules.validate_arrays(extra)
    for key,value in extra.items():
        if key in data and not np.array_equal(data[key],value):raise ValueError('Duplicate export differs: '+key)
        data[key]=value
    nb=int(data['nb']);counts=np.diff(data['boundary_offset'])
    for key in ('section_seed_node','boundary_seed_position'):
        raw=data['native_'+key]
        if raw.dtype!=np.int32 or raw.shape!=(nb,) or np.any(raw<0):raise ValueError('Invalid native seed metadata')
        data[key]=np.ascontiguousarray(raw-1)
    live=np.flatnonzero(data['section_control']==1)
    if not data['production_section_geometry'] or len(live)!=1 or np.any(data['mode'][live]!=9):
        raise ValueError('Require one production normal section')
    for b in live:
        if counts[b]<4 or not 1<=data['section_seed_node'][b]<counts[b]:raise ValueError('No interior normal seed')
        if not data['boundary_offset'][b]<=data['boundary_seed_position'][b]<data['boundary_offset'][b+1]:
            raise ValueError('Boundary seed is not a CSR position in its section')
    if np.any(~np.isin(data['mode'],[2,9])) or np.any(~np.isin(data['section_control'],[0,1])):
        raise ValueError('Unqualified boundary mode/control')
    return data


def environment(initial=False,runtime=RUNTIME):
    env={k:v for k,v in os.environ.items() if not k.upper().startswith(('BRAIDS_','OMP_'))}
    env.update(ENV)
    if initial:env['BRAIDS_GPU_INITIAL_EXPORT']='1'
    env['PATH']=str(Path(runtime))+os.pathsep+env.get('PATH','')
    return env


def prepare_case(deck,folder,runtime=RUNTIME):
    built=verify_build(runtime);folder=Path(folder).resolve();origin=cr.copy_inputs(deck,folder)
    command=[built['executable'],Path(deck).name]
    process=subprocess.run(command,cwd=folder,env=environment(True,runtime),capture_output=True,text=True,timeout=60)
    (folder/'initial.log').write_text(process.stdout+process.stderr);process.check_returncode()
    data=read_case(folder)
    if int(data['cpu_time_integrator'])!=43 or int(data['cpu_outlet_extension'])!=1:
        raise ValueError('Expected exact CPU21b7 initialized controls')
    np.savez_compressed(folder/'coupled_input.npz',**data)
    verify_build(runtime)
    save(folder/'preparation_receipt.json',dict(schema=1,status='completed',origin=origin,
        parent_executable_sha256=parent.TIGHT_SHA,export_runtime_sha256=built['executable_sha256'],
        build=str(Path(runtime)/'build.json'),build_sha256=sha(Path(runtime)/'build.json'),
        helper_sha256=sha(__file__),solver_steps_performed=0,initialization_CPU_only=True,
        metadata_conversion={'section_seed_node':'Native local1-based node /0 absent ->0-based /-1 absent',
            'boundary_seed_position':'Native global boundary_faces CSR position1-based /0 absent ->0-based /-1 absent'},
        real_native_schedule_tables=list(schedules.TABLES),native_parser_sorting_seed_checks_unchanged=True,
        artifacts={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
    return data


def reference(deck,folder):
    """Run only frozen original CPU21b7; no exporter hooks in this evolution."""
    _,protected=parent.runtime_identity();folder=Path(folder).resolve();origin=cr.copy_inputs(deck,folder)
    command=[str(parent.TIGHT_EXE),Path(deck).name];start=time.perf_counter()
    process=subprocess.run(command,cwd=folder,env=environment(runtime=parent.TIGHT_RUNTIME),capture_output=True,text=True,timeout=120)
    (folder/'native.log').write_text(process.stdout+process.stderr)
    record=dict(status='completed' if process.returncode==0 else 'failed',return_code=process.returncode,
        command=command,origin=origin,executable_sha256=parent.TIGHT_SHA,helper_sha256=sha(__file__),
        elapsed_s=time.perf_counter()-start,timing_is_benchmark=False,environment=ENV,
        output_files={str(p):sha(p) for p in folder.iterdir() if p.is_file()})
    save(folder/'execution.json',record);verify_files(protected);process.check_returncode();return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['build','fixtures','prepare','reference'])
    p.add_argument('--deck',type=Path);p.add_argument('--folder',type=Path);a=p.parse_args()
    if a.action=='build':
        result=build();print(json.dumps({'runtime':result['runtime'],'executable_sha256':result['executable_sha256']}))
    elif a.action=='fixtures':
        result=[prepare_fixture(n,ROOT/'inputs'/n) for n in ('prescribed','equilibrium')]
        print(json.dumps([{'name':r['name'],'input':r['input']} for r in result]))
    else:
        if a.deck is None or a.folder is None:p.error('deck and folder required')
        if a.action=='prepare':
            result=prepare_case(a.deck,a.folder);print(json.dumps({'cells':result['nc'],'folder':str(a.folder)}))
        else:print(json.dumps(reference(a.deck,a.folder)))


if __name__=='__main__':main()
