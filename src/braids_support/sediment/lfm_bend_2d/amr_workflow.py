"""Frozen adaptive hydraulic performance controls; no reference-engine changes."""
import argparse
import ctypes
from ctypes import wintypes
from dataclasses import asdict
import difflib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import numpy as np
from braids_core.boundaries.amr import AMRSettings,build_hierarchy
from braids_core.boundaries.domain import PolygonDomain
from braids_core.boundaries.inlet_io import NativeGrid
from . import face_workflow as prior

h=prior.h;m=prior.m
TAG='amr_workflow_20260930_v1'


def paths():return prior.p.paths(TAG)


def snapshot():
    root,report=paths();root.mkdir(exist_ok=False);report.mkdir(exist_ok=False)
    old,_=prior.paths()
    for src,name in (('reference','previous'),('delivery','reference')):
        h.runtime(old,src)
        shutil.copytree(old/('runtime_'+src),root/('runtime_'+name))
    _,identity=h.runtime(root,'reference')
    manifest=json.loads((h.r.ROOT/'solver/cut_cells/build/build.json').read_text())
    assert manifest['executable_sha256']==identity['executable_sha256']
    for name,value in manifest['source_sha256'].items():assert h.r.sha256(h.r.ROOT/name)==value
    h.r.write_json(report/'baseline.json',dict(current=identity,local_manifest=manifest,
        previous=h.runtime(root,'previous')[1],installed_plugin_modified=False,delft_modified=False))


def prepare(levels=4):
    root,report=paths();_,identity=h.runtime(root,'reference')
    domain=PolygonDomain('adaptive_control',[(0,0),(64,0),(64,16),(0,16)],'EPSG:2056')
    settings=AMRSettings(enabled=True,levels=levels,wet_level=levels-3,regrid_interval_s=.1,coarsen_delay=2,
        refine_walls=False,refine_bathymetry=False,refine_sediment=False,
        surface_tolerance_m=.002,velocity_tolerance_m_s=.03)
    start=time.perf_counter()
    scale=2**(levels-4)
    tree=build_hierarchy(domain,[],NativeGrid(64//scale,16//scale,0,0,scale,scale),levels=levels)
    preparation_geometry_s=time.perf_counter()-start
    xy=np.array([c.centroid_xy for c in tree.finest.cells]);x,y=xy.T
    areas=np.array([c.area_m2 for c in tree.cells])
    for case in (('pulse','front') if levels==4 else ('pulse6',)):
        folder=root/('input_'+case);folder.mkdir(exist_ok=False)
        q=np.zeros((len(x),5));rock=np.zeros(len(x))
        if case.startswith('pulse'):q[:,0]=.5+.15*np.exp(-((x-24)/3)**2-((y-8)/3)**2)
        else:q[:,0]=np.where(x<16.,.5,0.)
        h.r.write_cut_case(folder/'case.data',tree.finest,q,rock,{},end_time_s=.8,
            dt_max_s=.01*scale,output_interval_s=.2,scheme=6,bedload=0,exchange=0,
            manning=.025,cfl=.8,source_sha256=identity['source_sha256'],
            dry_tolerance_m=.0001,amr_hierarchy=tree,amr_settings=settings,compact_sidecars=True)
        np.savez(folder/'analysis_geometry.npz',area=areas,level=tree.node_level,
            xy=np.array([c.centroid_xy for c in tree.cells]))
        h.r.write_json(folder/'input_provenance.json',dict(case=case,domain_m=[64,16],
            finest_dx_m=.125,finest_cells=len(x),tree_nodes=tree.count,settings=asdict(settings),
            end_s=.8,dt_max_s=.01*scale,output_s=.2,scheme=6,hydraulic_only=True,
            geometry_preparation_s=preparation_geometry_s,
            files={f.name:h.r.sha256(f) for f in folder.iterdir() if f.is_file()}))
        print('PREPARED '+case,flush=True)


class MemoryCounters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
        (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
            'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]


def measure(command,folder,env):
    # OS high-water marks sampled every 50 ms, not instantaneous-RSS maxima.
    # Sampling can miss a final allocation after the last successful query.
    getmemory=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
    getmemory.argtypes=[wintypes.HANDLE,ctypes.POINTER(MemoryCounters),wintypes.DWORD]
    getmemory.restype=wintypes.BOOL
    peak_ws=peak_commit=queries=0;last_query_s=0.;start=time.perf_counter()
    with (folder/'native.log').open('w') as log:
        process=subprocess.Popen(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
        while True:
            info=MemoryCounters();info.cb=ctypes.sizeof(info)
            if getmemory(int(process._handle),ctypes.byref(info),info.cb):
                queries+=1;last_query_s=time.perf_counter()-start
                peak_ws=max(peak_ws,info.PeakWorkingSetSize);peak_commit=max(peak_commit,info.PeakPagefileUsage)
            if process.poll() is not None:break
            if time.perf_counter()-start>600:
                process.terminate();process.wait();raise TimeoutError('bounded AMR benchmark exceeded 600 s')
            time.sleep(.05)
        assert process.returncode==0,(folder/'native.log').read_text()[-6000:]
    return dict(process_wall_s=time.perf_counter()-start,peak_working_set_bytes=peak_ws,
        peak_commit_bytes=peak_commit,memory_queries=queries,last_memory_query_s=last_query_s,
        memory_method='Win32 OS high-water marks, queried every 50 ms; may miss final-query tail')


def execute(case,variant,threads,label):
    root,report=paths();folder=root/label;h.clone(root/('input_'+case),folder)
    exe,identity=h.runtime(root,variant)
    env=dict(os.environ,OMP_NUM_THREADS=str(threads),OMP_DYNAMIC='FALSE',
             OMP_WAIT_POLICY='PASSIVE',BRAIDS_PROFILE='1',BRAIDS_PROGRESS_STEPS='20000')
    env.pop('BRAIDS_AMR_DIAGNOSTICS',None)
    row=measure([str(exe),'case.data'],folder,env)
    log=(folder/'native.log').read_text();profile=h.parse_profile(log)
    counts=[int(x) for x in re.search(r'AMR meshes, regrid evaluations, active leaves:\s*([^\n]+)',log)[1].split()]
    levels=[int(x) for x in re.search(r'AMR accepted steps by level L0 through finest:\s*([^\n]+)',log)[1].split()]
    row.update(label=label,case=case,variant=variant,threads=threads,profile=profile,
        meshes=counts[0],regrids=counts[1],final_cells=counts[2],level_steps=levels,
        executable_sha256=identity['executable_sha256'],input_sha256=h.r.sha256(folder/'input_provenance.json'))
    row['detail']={k:[float(x) for x in v.split()] for k,v in re.findall(r'^detail: (.*?)=\s*([^\n]+)',log,re.M)}
    h.r.write_json(folder/'execution.json',row)
    print(json.dumps({k:row[k] for k in ('label','process_wall_s','peak_working_set_bytes','meshes','level_steps')}),flush=True)
    return row


def run(variants=('previous','reference'),threads=(1,8,16,32),cases=('pulse','front'),repeats=1,label='scaling'):
    _,report=paths();rows=[]
    assert not (report/(label+'.json')).exists()
    for case in cases:
        for repeat in range(repeats):
            for nt in threads:
                rotated=variants[repeat%len(variants):]+variants[:repeat%len(variants)]
                for variant in rotated:
                    row=execute(case,variant,nt,f'{label}_{case}_{variant}_t{nt}_r{repeat+1}')
                    rows.append(dict(row,repeat=repeat+1));h.r.write_json(report/(label+'.json'),rows)


def compare(left,right):
    a=h.r.shared.read_frames(left/'frames.txt');b=h.r.shared.read_frames(right/'frames.txt')
    assert len(a)==len(b)
    geometry=np.load(left/'analysis_geometry.npz');area=geometry['area'];level=geometry['level']
    initial=a[0]['q'].T@area[a[0]['node_ids']];budgets=[];shapes=[]
    for x,y in zip(a,b):
        assert x['time_s']==y['time_s'] and x['mesh_id']==y['mesh_id']
        for field in ('node_ids','q','ledger'):np.testing.assert_array_equal(x[field],y[field])
        assert np.isfinite(y['q']).all() and np.min(y['q'][:,0])>=0
        budgets.append(y['q'].T@area[y['node_ids']]-initial)
        shapes.append(dict(time_s=y['time_s'],cells=len(y['q']),
            levels=np.bincount(level[y['node_ids']],minlength=4).tolist()))
    residual=float(np.max(abs(np.array(budgets)[:,0])))
    assert residual<1e-8
    assert len({r['cells'] for r in shapes})>1,'Case must really change mesh'
    assert (left/'checkpoint.bin').read_bytes()==(right/'checkpoint.bin').read_bytes()
    return dict(saved_state_and_mesh_exact=True,checkpoint_exact=True,water_residual_m3=residual,frames=shapes)


def fused_transform(original):
    text=m.replace_once(original,' integer,allocatable :: amr_level_eval_cells(:,:),amr_level_face_cells(:,:)',
        ''' integer,allocatable :: amr_level_eval_cells(:,:),amr_level_face_cells(:,:)
 ! Bit 0: evaluate wet/dry traces; bit 1: reconstruct a participating face cell.
 integer,allocatable :: amr_prepare_role(:,:)''')
    before=m.routine(text,'amr_cache_level_work');after=before
    after=m.replace_once(after,'  amr_level_cells=0;',
        '''  if(allocated(amr_prepare_role)) deallocate(amr_prepare_role)
  allocate(amr_prepare_role(nc,0:amr_levels-1));amr_prepare_role=0
  amr_level_cells=0;''')
    after=m.replace_once(after,'    if(face_marked(c)) then',
        '    if(face_marked(c)) then\n     amr_prepare_role(c,l)=ibset(amr_prepare_role(c,l),1)')
    after=m.replace_once(after,'    if(marked(c)) then',
        '    if(marked(c)) then\n     amr_prepare_role(c,l)=ibset(amr_prepare_role(c,l),0)')
    text=m.replace_once(text,before,after)
    fused=m.routine(original,'prepare_hydraulic_traces_fused')
    fused=m.replace_once(fused,'subroutine prepare_hydraulic_traces_fused(',
        'subroutine prepare_adaptive_traces_fused(')
    fused=fused.replace('  ! Same fixed-grid scheme-6 formulas and per-cell accumulation order.',
        '  ! Same scheme-6 kernels, preserving the distinct AMR cell subsets.')
    fused=m.replace_once(fused,'   if(.not.active_cell(c)) cycle',
        '''   if(amr_step_level>=0) then
    if(.not.btest(amr_prepare_role(c,amr_step_level),1)) cycle
   endif
   if(.not.active_cell(c)) cycle''')
    fused=m.replace_once(fused,'   h=state(1,c);flatten=h<=dry_tolerance',
        '''   if(amr_step_level>=0) then
    if(.not.btest(amr_prepare_role(c,amr_step_level),0)) cycle
   endif
   h=state(1,c);flatten=h<=dry_tolerance''')
    text=text.replace('\ncontains\n','\ncontains\n'+fused+'\n',1)
    before=m.routine(text,'euler');after=before
    anchor='''  else
   call gradients(v,bc_concentration,grad,vlo,vhi)'''
    after=m.replace_once(after,anchor,'''  else if(amr_enabled.and.scheme==6.and.nc>=4096.and.nfrac==1.and.nstate==5.and. &
     bedlaw==0.and.exchange_mode==0.and..not.secondary_flow.and. &
     .not.bank_front_tracking.and..not.bank_failure.and..not.bank_avalanche.and. &
     .not.general_avalanche.and.bank_erosion_fraction==0.d0.and. &
     .not.bedform_roughness.and..not.bedload_adaptation) then
   downslope=0
   if(bed_slope==2) call bed_downslopes(bed,downslope)
   call prepare_adaptive_traces_fused(state,v,bc_concentration,grad,vlo,vhi)
   if(profile_enabled) profile_hydraulic_workflow_calls=profile_hydraulic_workflow_calls+1
  else
   call gradients(v,bc_concentration,grad,vlo,vhi)''')
    return m.replace_once(text,before,after)


def instrument(original):
    text=original.replace('\ncontains\n','''
 integer(8) :: detail_level_calls(-1:max_amr_levels-1)=0
 integer(8) :: detail_level_cells(-1:max_amr_levels-1)=0,detail_level_faces(-1:max_amr_levels-1)=0
 integer(8) :: detail_level_grad(-1:max_amr_levels-1)=0,detail_level_eval(-1:max_amr_levels-1)=0
 real(8) :: detail_rebuild_s=0.d0
contains
''',1)
    before=m.routine(text,'euler');after=m.replace_once(before,
        '  if(amr_step_level>=0) ntarget=amr_level_cell_count(amr_step_level)',
        '''  if(amr_step_level>=0) ntarget=amr_level_cell_count(amr_step_level)
  detail_level_calls(amr_step_level)=detail_level_calls(amr_step_level)+1
  detail_level_cells(amr_step_level)=detail_level_cells(amr_step_level)+ntarget
  detail_level_faces(amr_step_level)=detail_level_faces(amr_step_level)+nwork
  if(amr_step_level>=0) then
   detail_level_grad(amr_step_level)=detail_level_grad(amr_step_level)+amr_level_face_cell_count(amr_step_level)
   detail_level_eval(amr_step_level)=detail_level_eval(amr_step_level)+amr_level_eval_cell_count(amr_step_level)
  else
   detail_level_grad(-1)=detail_level_grad(-1)+nc;detail_level_eval(-1)=detail_level_eval(-1)+nc
  endif''')
    text=m.replace_once(text,before,after)
    before=m.routine(text,'amr_regrid');after=m.replace_once(before,
        ' real(8) :: profile_start',' real(8) :: profile_start,detail_rebuild_start')
    after=m.replace_once(after,'   call amr_rebuild_mesh();amr_mesh_id=amr_mesh_id+1',
        '''   detail_rebuild_start=omp_get_wtime()
   call amr_rebuild_mesh();amr_mesh_id=amr_mesh_id+1
   if(.not.initial) detail_rebuild_s=detail_rebuild_s+omp_get_wtime()-detail_rebuild_start''')
    text=m.replace_once(text,before,after)
    anchor="  write(*,'(a,f12.4)') 'profile: authoritative output wall seconds='"
    out=''
    for key,variable in (('level calls','calls'),('level update cells','cells'),('level faces','faces'),
                         ('level gradient cells','grad'),('level evaluation cells','eval')):
        out+=f"  write(*,'(a,*(i0,1x))') 'detail: {key}=',detail_level_{variable}(-1:amr_levels-1)\n"
    out+="  write(*,'(a,f12.6)') 'detail: regrid rebuild seconds=',detail_rebuild_s\n"
    return m.replace_once(text,anchor,out+anchor)


def build(variants):
    root,_=paths();_,parent=h.runtime(root,'reference')
    original=(root/'runtime_reference/cutcell_solver.f90').read_text()
    for variant in variants:
        folder=root/('runtime_'+variant);folder.mkdir(exist_ok=False);hashes={}
        changed=transform(original,variant)
        for name in h.SOURCES:
            source=root/'runtime_reference'/Path(name).name;target=folder/source.name
            if name==h.SOURCES[-1]:target.write_text(changed)
            else:shutil.copy2(source,target)
            hashes[name]=h.r.sha256(target)
        (folder/'candidate.patch').write_text(''.join(difflib.unified_diff(
            original.splitlines(True),changed.splitlines(True),fromfile='reference',tofile=variant)))
        command=[parent['command'][0],'-O3','-flto','-fopenmp','-ffree-line-length-none','-ffixed-line-length-none',
                 *[str(folder/Path(name).name) for name in h.SOURCES],'-o',str(folder/'xbraids_candidate.exe')]
        start=time.perf_counter();proc=subprocess.run(command,cwd=folder,capture_output=True,text=True,timeout=240)
        (folder/'build.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
        h.r.write_json(folder/'build.json',dict(command=command,source_sha256=hashes,variant=variant,
            executable_sha256=h.r.sha256(folder/'xbraids_candidate.exe'),parent_executable_sha256=parent['executable_sha256'],
            compile_wall_s=time.perf_counter()-start,builder_sha256=h.r.sha256(Path(__file__))))
        shutil.copy2(__file__,folder/'builder_snapshot.py');print('BUILT '+variant,flush=True)


def transform(original,variant):
    if variant=='diagnostic':return instrument(original)
    if variant=='faces_all_teams':
        return m.replace_once(transform(original,'faces_all'),
            '    omp_get_max_threads()<=8.and.nfrac==1.and.nstate==5.and.bedlaw==0.and.exchange_mode==0.and. &',
            '    nfrac==1.and.nstate==5.and.bedlaw==0.and.exchange_mode==0.and. &')
    assert variant in ('fused','faces4096','faces_all')
    text=fused_transform(original)
    if variant=='fused':return text
    threshold='4096' if variant=='faces4096' else '1'
    # Test short AMR face loops independently of the fixed-grid policy.
    return m.replace_once(text,'  parallel_faces=nwork*nfrac>=8192',
        f'''  parallel_faces=nwork*nfrac>=8192.or.(amr_step_level>=0.and.nc>=4096.and.nwork>={threshold}.and. &
    omp_get_max_threads()<=8.and.nfrac==1.and.nstate==5.and.bedlaw==0.and.exchange_mode==0.and. &
    .not.secondary_flow.and..not.bank_front_tracking.and..not.bank_failure.and..not.bank_avalanche.and. &
    .not.general_avalanche.and.bank_erosion_fraction==0.d0.and..not.bedform_roughness.and..not.bedload_adaptation)''')


def fixed(variants,label='fixed',selected=('small','large','bend')):
    root,report=paths();old,_=prior.p.p.paths();rows=[]
    assert not (report/(label+'.json')).exists()
    cases=dict(small=(old/'input_small_reference',8),large=(old/'input_large_reference',32),
               bend=(prior.p.p.PARENT/'input40_reference',8))
    for case,(source,threads) in cases.items():
        if case not in selected:continue
        for repeat in range(3):
            for variant in variants[repeat%len(variants):]+variants[:repeat%len(variants)]:
                row=h.execute_native(root,source,f'{label}_{case}_{variant}_r{repeat+1}',threads,variant)
                rows.append(dict(row,case=case,repeat=repeat+1));h.r.write_json(report/(label+'.json'),rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('snapshot','prepare','prepare6','run','build','fixed'))
    parser.add_argument('--variants',nargs='+',default=['previous','reference'])
    parser.add_argument('--threads',nargs='+',type=int,default=[1,8,16,32])
    parser.add_argument('--cases',nargs='+')
    parser.add_argument('--repeats',type=int,default=1);parser.add_argument('--label',default='scaling')
    args=parser.parse_args()
    if args.stage=='run':run(tuple(args.variants),tuple(args.threads),tuple(args.cases or ('pulse','front')),args.repeats,args.label)
    elif args.stage=='build':build(tuple(args.variants))
    elif args.stage=='prepare6':prepare(6)
    elif args.stage=='fixed':fixed(tuple(args.variants),'fixed' if args.label=='scaling' else args.label,tuple(args.cases or ('small','large','bend')))
    else:globals()[args.stage]()
