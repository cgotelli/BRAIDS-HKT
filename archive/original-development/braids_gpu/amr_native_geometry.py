"""Initialization-only selected-leaf operators from an ordinary native AMR deck.

The copied BRAIDS initializer preserves the original parser and geometry
routines. It loads original initial conditions, selects requested global leaves,
exports immutable geometry/material controls, and stops before time integration.
Evolving solution transfer remains exclusively in the GPU route.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from .amr_prepared import (_read_coupled, _leaves, _tree_arrays,
                          export_bank_amr_prepared_v2)
from .atomic_output import write_json_atomic


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value,message):
    if not value:
        raise ValueError(message)


SELECT_SOURCE = '''
 subroutine gpu_selected_amr_initialization()
  integer :: unit,ios,count,c,n,f,b,gpu_unit
  integer,allocatable :: selected(:)
  character(64) :: request
  call get_environment_variable('BRAIDS_GPU_SELECTED_AMR_EXPORT',request)
  if(trim(request)/='1') return
  if(.not.amr_enabled.or.scheme/=6.or.time/=0.d0.or.steps/=0) &
   error stop 'Selected GPU initializer needs original AMR scheme6 initial state'
  if(bank_front_tracking.or.bank_toe_erosion_mode/=2.or..not.bank_failure) &
   error stop 'Selected GPU initializer requires mode2 bank without persistent front'
  ! This record is the original complete initial tree, not a CPU trajectory.
  call gpu_open(gpu_unit,'amr_tree_input.bin')
TREE_RECORDS
  call gpu_finish(gpu_unit)
  open(newunit=unit,file='selected_nodes.txt',status='old',action='read')
  read(unit,*,iostat=ios) count
  if(ios/=0.or.count<=0.or.count>amr_nn) error stop 'Invalid selected leaf count'
  allocate(selected(count))
  do c=1,count
   read(unit,*,iostat=ios) n
   if(ios/=0.or.n<0.or.n>=amr_nn) error stop 'Invalid selected global leaf'
   selected(c)=n+1
   if(c>1) then
    if(selected(c)<=selected(c-1)) error stop 'Selected global leaves are not ordered'
   endif
  enddo
  read(unit,*,iostat=ios) n
  if(ios>=0) error stop 'Unexpected selected leaf trailer'
  close(unit)
  deallocate(at_leaf);allocate(at_leaf(count));at_leaf=selected
  call amr_update_cover()
  do n=1,amr_nn
   if(at_cover(n)>0) then
    if(at_level(at_cover(n))<at_min(n)) error stop 'Selected leaves violate minimum level'
   endif
  enddo
  if(any(at_cover(amr_fine_offset+1:amr_nn)<=0)) error stop 'Incomplete selected leaf cover'
  do f=1,amr_fn
   if(af_neighbour(f)<=0) cycle
   c=at_cover(af_owner(f));n=at_cover(af_neighbour(f))
   if(c<=0.or.n<=0.or.c==n) cycle
   if(abs(at_level(c)-at_level(n))>1) error stop 'Selected leaves violate 2to1 balance'
  enddo
  call amr_rebuild_mesh()
  call build_physical_toe_map(q)
  call gpu_open(gpu_unit,'amr_selected_identity.bin')
SELECTED_RECORDS
  call gpu_finish(gpu_unit)
  call i4_export_physical_toe_metadata()
  call i4_export_avalanche_metadata()
  call g4_export_bank_history_metadata()
  ! The exported operator is a fixed selected graph. The outer GPU driver
  ! owns adaptation, while the separate tree record retains original controls.
  amr_enabled=.false.
  ! Same post-load invariant resolution as the unchanged native main program.
  ! Do not export the module's conservative default (.true.) as resolved data.
  suspended_transport_active=exchange_mode>0.or.any(fraction_concentration/=0.d0).or. &
   any(series_concentration/=0.d0).or.any(fraction_series_concentration/=0.d0)
  if(.not.suspended_transport_active) then
   do b=1,nfrac
    if(any(q(suspended_index(b),:)/=0.d0)) then
     suspended_transport_active=.true.;exit
    endif
   enddo
  endif
  call resolve_cpu_time_integrator(.false.)
  call gpu_export_input()
  stop 'GPU selected geometry exported; zero native solver timesteps'
 end subroutine
'''


def selection_source():
    from validation.gpu_fwave.coupled_reference import rec
    records=''
    for key,expr,shape,kind in (
        ('parent','at_parent-1',('amr_nn',),'i'),
        ('node_level','at_level',('amr_nn',),'i'),
        ('minimum_level','at_min',('amr_nn',),'i'),
        ('maximum_level','at_max',('amr_nn',),'i'),
        ('area','at_area',('amr_nn',),'r'),
        ('xy','at_xy',('2','amr_nn'),'r'),
        ('rock','at_rock',('amr_nn',),'r'),
        ('state','at_q',('nstate','amr_nn'),'r'),
        ('face_owner','af_owner-1',('amr_fn',),'i'),
        ('face_neighbour','af_neighbour-1',('amr_fn',),'i'),
        ('face_mid','af_mid',('2','amr_fn'),'r'),
        ('face_length','af_length',('amr_fn',),'r'),
        ('flags','amr_flags',('size(amr_flags)',),'i')):
        records+=rec(key,expr,shape,kind)
    for key in ('amr_levels','amr_ratio','amr_delay','amr_wet_level','amr_buffer'):
        records+=rec(key,kind='i')
    for key in ('amr_interval','amr_wet_depth','amr_eta_tol','amr_u_tol',
                'amr_bed_tol','amr_c_tol','amr_mix_tol','amr_dz_tol'):
        records+=rec(key)
    records+=rec('subcycling','merge(1,0,amr_subcycling)',kind='i')
    selected=rec('node_ids','at_leaf-1',('nc',),'i')
    selected+=rec('face_source_ids','at_face_source',('nf',),'i')
    return SELECT_SOURCE.replace('TREE_RECORDS',records.rstrip()).replace('SELECTED_RECORDS',selected.rstrip())


_LAW9_SNAPSHOT_NATIVE_SYMBOLS = (
    'fixed_boundary_bed_mode',
    'fixed_boundary_bed',
    'gaia_skin_roughness_ratio',
    'gaia_slope_beta',
    'gaia_deviation_beta2',
    'gaia_secondary_alpha',
)
_HISTORICAL_BEDLAW2_PARENT_TEXT_SHA256 = (
    'a9ff651cec270eb57d7076033bd7a0b64f01e1879f72aca0f1e534bc106ed312')
_HISTORICAL_BEDLAW2_PARENT_RAW_SHA256 = (
    '75a5451cce5d11e2aec244330aa91724e514dbe16dfdfef97fb9b50572020340')


def _source_has_identifier(source, name):
    return re.search(r'(?<![A-Za-z0-9_])'+re.escape(name)+
                     r'(?![A-Za-z0-9_])', source) is not None


def _snapshot_export_view(source, changed):
    """Match snapshot records to the immutable parent source capability.

    The current fixed-grid exporter adds explicit law-9 fields.  Historical
    AMR parents predate those native symbols, so their export-only derivative
    must omit the complete law-9 record group rather than manufacture values
    or inject new physics into a bedlaw-2 parent.  A partial native capability
    is never accepted.  Current law-9 parents are returned byte-for-byte.
    """
    present = tuple(_source_has_identifier(source, name)
                    for name in _LAW9_SNAPSHOT_NATIVE_SYMBOLS)
    require(all(present) or not any(present),
            'Native snapshot source has a partial law9 export capability')
    if all(present):
        return changed
    source_sha = hashlib.sha256(source.encode('utf-8')).hexdigest()
    require(source_sha == _HISTORICAL_BEDLAW2_PARENT_TEXT_SHA256,
            'Law9-absent snapshot source is not the frozen historical bedlaw2 parent')

    replacements = (
        ('  integer :: gpu_fixed_boundary_bed_mode(nb)\n', ''),
        (',gpu_fixed_boundary_bed(nf)\n', '\n'),
        ('  gpu_fixed_boundary_bed_mode=0;gpu_fixed_boundary_bed=0.d0\n'
         '  if(bedlaw==9) then\n'
         '   if(.not.allocated(fixed_boundary_bed_mode).or..not.allocated(fixed_boundary_bed)) &\n'
         "    error stop 'GPU TELEMAC-GAIA fixed boundary bed is not initialized'\n"
         '   gpu_fixed_boundary_bed_mode=fixed_boundary_bed_mode\n'
         '   gpu_fixed_boundary_bed=fixed_boundary_bed\n'
         '  endif\n', ''),
        ("  call gpu_put_r(gpu_unit,'gaia_skin_roughness_ratio',[gaia_skin_roughness_ratio],[integer::])\n", ''),
        ("  call gpu_put_r(gpu_unit,'gaia_slope_beta',[gaia_slope_beta],[integer::])\n", ''),
        ("  call gpu_put_r(gpu_unit,'gaia_deviation_beta2',[gaia_deviation_beta2],[integer::])\n", ''),
        ("  call gpu_put_r(gpu_unit,'gaia_secondary_alpha',[gaia_secondary_alpha],[integer::])\n", ''),
        ("  call gpu_put_r(gpu_unit,'fixed_boundary_bed',reshape(gpu_fixed_boundary_bed,[size(gpu_fixed_boundary_bed)]),[nf])\n", ''),
        ("  call gpu_put_i(gpu_unit,'fixed_boundary_bed_mode',reshape(gpu_fixed_boundary_bed_mode,[size(gpu_fixed_boundary_bed_mode)]),[nb])\n", ''),
    )
    for old, new in replacements:
        require(changed.count(old) == 1,
                'Historical snapshot compatibility anchor differs')
        changed = changed.replace(old, new, 1)
    require(not any(_source_has_identifier(changed, name)
                    for name in _LAW9_SNAPSHOT_NATIVE_SYMBOLS),
            'Historical snapshot retained a law9-only export symbol')
    return changed


def transform(source):
    from validation.gpu_fwave import g4_bank_active_front_export as baseline
    changed=_snapshot_export_view(source,baseline.transform(source))
    anchor=' call load_case(trim(deck))\n'
    marker='end module\n\nprogram run_cutcell'
    require(changed.count(anchor)==1 and changed.count(marker)==1,
            'Selected initialization anchors differ')
    return changed.replace(anchor,anchor+' call gpu_selected_amr_initialization()\n',1).replace(
        marker,selection_source()+marker,1)


def verify_transform(source, changed):
    from validation.gpu_fwave import g4_bank_active_front_export as baseline
    require(changed==transform(source),'Selected initialization transform differs')
    restored=changed.replace(selection_source(),'',1).replace(
        ' call gpu_selected_amr_initialization()\n','',1)
    raw=baseline.transform(source)
    require(restored==_snapshot_export_view(source,raw),
            'Selected initialization snapshot capability view differs')
    baseline.verify_transform(source,raw)
    for name in ('amr_rebuild_mesh','amr_initialize_tree','amr_prolong_node','amr_regrid'):
        start=source.index('subroutine '+name+'(')
        end=source.index('end subroutine',start)
        require(changed.count(source[start:end])==1,'Original AMR routine changed: '+name)


def build(runtime):
    from validation.gpu_fwave import cpu_filter_cold20h as parent
    runtime=Path(runtime).resolve()
    require(not runtime.exists(),'Selected AMR initialization runtime is write-once')
    identity,protected=parent.runtime_identity()
    require(identity['source_sha256'].get('cutcell_solver.f90')==
            _HISTORICAL_BEDLAW2_PARENT_RAW_SHA256,
            'Selected AMR initializer parent receipt differs')
    runtime.mkdir(parents=True)
    sources=[]
    for name,digest in identity['source_sha256'].items():
        original=parent.TIGHT_RUNTIME/name
        require(sha(original)==digest,'Frozen native source changed')
        target=runtime/name
        if name=='cutcell_solver.f90':
            source=original.read_text(encoding='utf-8')
            changed=transform(source)
            verify_transform(source,changed)
            # Generated build input, never a patch to the reference checkout.
            target.write_text(changed,encoding='utf-8')
        else:
            shutil.copy2(original,target)
        sources.append(str(target))
    for original in parent.TIGHT_RUNTIME.glob('*.dll'):
        shutil.copy2(original,runtime/original.name)
    compiler=Path(identity['command'][0])
    flags=identity['command'][1:6]
    require(flags==['-O3','-flto','-fopenmp','-ffree-line-length-none','-ffixed-line-length-none'],
            'Frozen compiler options differ')
    exe=runtime/'xbraids_amr_geometry.exe'
    command=[str(compiler),*flags,*sources,'-o',str(exe)]
    result=subprocess.run(command,cwd=runtime,capture_output=True,text=True,timeout=300,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    (runtime/'build.log').write_text(result.stdout+result.stderr)
    require(result.returncode==0,'Selected initializer build failed: '+result.stderr[-3000:])
    record=dict(schema=1,command=command,executable=exe.name,
        executable_sha256=sha(exe),parent_executable_sha256=parent.TIGHT_SHA,
        compiler_sha256=sha(compiler),generator_sha256=sha(__file__),
        source_sha256={Path(path).name:sha(path) for path in sources},
        parent_protected_files=protected,export_only=True,numerical_routines_unchanged=True,
        source_original_modified=False,Delft3D_source_modified=False)
    write_json_atomic(runtime/'build.json',record)
    return record


def verify_runtime(runtime):
    runtime=Path(runtime).resolve()
    record=json.loads((runtime/'build.json').read_text())
    require(record['schema']==1 and record['export_only'] is True and
            record['numerical_routines_unchanged'] is True and
            record['generator_sha256']==sha(__file__) and
            Path(record['executable']).name==record['executable'] and
            sha(runtime/record['executable'])==record['executable_sha256'] and
            all(sha(runtime/name)==digest for name,digest in record['source_sha256'].items()),
            'Selected initialization runtime identity differs')
    return record


def hierarchy_from_sidecar(path):
    with np.load(path,allow_pickle=False) as data:
        metadata=json.loads(str(data['metadata_json']))
        scalar=data['cell_scalar']
        hierarchy=SimpleNamespace(parent=data['parent'].copy(),
            node_level=data['node_level'].copy(),maximum_level=data['maximum_level'].copy(),
            offsets=data['offsets'].copy(),refinement_ratio=metadata['refinement_ratio'],
            cells=tuple(SimpleNamespace(area_m2=float(row[0]),centroid_xy=tuple(row[1:3]))
                        for row in scalar))
    _tree_arrays(hierarchy)
    return hierarchy


def _selected_geometry(hierarchy_path, output, ids):
    from braids_core.boundaries.geometry_io import load_geometry,write_packed_geometry
    content=load_geometry(hierarchy_path,cell_ids=ids,include_faces=False)
    rows=content.pop('cells')
    content['semantic_format']='braids-cut-cell-mesh-v1'
    for index,row in enumerate(rows):
        row['cell_id']=index
    write_packed_geometry(output,cell_count=len(ids),rows=lambda:iter(rows),metadata=content)


def export_selected(deck, folder, ids, *, runtime, hierarchy_path=None, cancelled=lambda:False):
    """Return schema-2 package plus immutable toe lineage; cancellation is explicit."""
    from validation.gpu_fwave import coupled_reference as core
    deck,folder,runtime=Path(deck).resolve(),Path(folder).resolve(),Path(runtime).resolve()
    hierarchy_path=Path(hierarchy_path or deck.parent/'amr_hierarchy.npz').resolve()
    hierarchy=hierarchy_from_sidecar(hierarchy_path)
    ids=_leaves(np.asarray(ids,dtype=np.int64),_tree_arrays(hierarchy))
    record=verify_runtime(runtime)
    require(not folder.exists(),'Selected operator destination is write-once')
    if cancelled():
        raise InterruptedError('AMR geometry preparation cancelled')
    work=folder.with_name(folder.name+'_native')
    origin=core.copy_inputs(deck,work)
    # Retain a copied original geometry if present; never overwrite its provenance.
    if (work/'geometry.npz').exists():
        (work/'geometry.npz').rename(work/'original_geometry.npz')
        origin['files']['original_geometry.npz']=origin['files'].pop('geometry.npz')
    shutil.copy2(hierarchy_path,work/'amr_hierarchy.npz')
    origin['files']['amr_hierarchy.npz']=sha(work/'amr_hierarchy.npz')
    np.savetxt(work/'selected_nodes.txt',ids,fmt='%d',header=str(len(ids)),comments='')
    origin['files']['selected_nodes.txt']=sha(work/'selected_nodes.txt')
    environment={key:value for key,value in os.environ.items()
                 if not key.upper().startswith(('BRAIDS_','OMP_'))}
    environment.update(BRAIDS_GPU_SELECTED_AMR_EXPORT='1',BRAIDS_TIME_INTEGRATOR='legacy',
        OMP_NUM_THREADS='1',OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE')
    environment['PATH']=str(runtime)+os.pathsep+environment.get('PATH','')
    started=time.perf_counter()
    process=subprocess.Popen([str(runtime/record['executable']),deck.name],cwd=work,env=environment,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    try:
        while True:
            if cancelled():
                raise InterruptedError('AMR geometry preparation cancelled')
            if time.perf_counter()-started>120:
                raise TimeoutError('AMR geometry initialization exceeded 120 seconds')
            try:
                stdout,stderr=process.communicate(timeout=.05)
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        process.terminate()
        stdout,stderr=process.communicate(timeout=10)
        (work/'initial.log').write_text(stdout+stderr)
        raise
    (work/'initial.log').write_text(stdout+stderr)
    require(process.returncode==0,'Selected initializer failed: '+(stdout+stderr)[-2000:])
    require(not (work/'frames.txt').exists() and not (work/'checkpoint.bin').exists(),
            'Geometry preparation unexpectedly advanced a native simulation')
    data=_read_coupled(work/'coupled_input.bin')
    for sidecar in ('bank_toe_metadata.bin','avalanche_metadata.bin','bank_history_metadata.bin'):
        extra=_read_coupled(work/sidecar)
        require(all(key not in data or np.array_equal(data[key],value) for key,value in extra.items()),
                'Initializer sidecar conflicts with native core')
        data.update(extra)
    selected=_read_coupled(work/'amr_selected_identity.bin')
    require(np.array_equal(selected['node_ids'],ids),'Native initializer selected different leaves')
    _selected_geometry(hierarchy_path,work/'geometry.npz',ids)
    artifact_names=('coupled_input.bin','bank_toe_metadata.bin','avalanche_metadata.bin','bank_history_metadata.bin')
    receipt=dict(schema=1,status='completed',origin=origin,
        runtime_build_sha256=sha(runtime/'build.json'),runtime_executable_sha256=record['executable_sha256'],
        native_cpu_parent_sha256=record['parent_executable_sha256'],exporter_source_sha256=sha(__file__),
        initialization_CPU_only=True,solver_steps_performed=0,numerical_routines_unchanged=True,
        GPU_solver_executed=False,Delft3D_source_modified=False,cells=int(data['nc']),faces=int(data['nf']),
        controls={key:int(data[key]) for key in ('scheme','bedlaw','exchange_mode',
            'bank_toe_erosion_mode','general_avalanche','secondary_flow_model','cpu_time_integrator')},
        bank_cells=int(np.count_nonzero(data['bank_material_mask'])),
        toe_faces=int(np.count_nonzero(data['is_toe_face'])),
        artifacts={name:sha(work/name) for name in artifact_names})
    write_json_atomic(work/'preparation_receipt.json',receipt)
    if cancelled():
        raise InterruptedError('AMR geometry preparation cancelled before publication')
    package=export_bank_amr_prepared_v2(folder,hierarchy,
        [('leaves_'+hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest(),ids,work)],
        runtime_build=runtime/'build.json',source=dict(
            hierarchy_sha256=sha(hierarchy_path),base_input_sha256=sha(deck),
            runtime_build_sha256=sha(runtime/'build.json'),on_demand_initialization=True))
    lineage=[]
    for face in np.flatnonzero(data['is_toe_face']):
        lineage.append(dict(segment_id=int(selected['face_source_ids'][face]),local_face=int(face),
            bank_node=int(ids[int(data['candidate_bank'][face])]),
            toe_node=int(ids[int(data['candidate_toe'][face])]),
            midpoint_xy_m=data['mid'][:,face].tolist(),length_m=float(data['length'][face])))
    write_json_atomic(work/'selection_receipt.json',dict(
        node_ids=ids.tolist(),tree_input_sha256=sha(work/'amr_tree_input.bin'),
        selection_identity_sha256=sha(work/'amr_selected_identity.bin'),
        elapsed_s=time.perf_counter()-started,original_input_sha256=sha(deck),
        native_solver_steps=0,full_state_transfer_on_GPU=True))
    return package,lineage
