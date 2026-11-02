"""Isolated GPU Exner post-flux port, not a coupled sediment GPU solver.

Keeps original per-fraction donor limiting, MorFac placement and shared flux
gather. CPU-calculated sediment flux is an explicit input to this bounded proof;
capacity, reconstruction, slope and boundary physics are NOT claimed here.
No automatic GPU dispatch or timing claim follows from these correctness tests.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from . import resident_update as r

HERE=Path(__file__).resolve().parent
TAG='gpu_exner_update_20261001_v4'
PARENT=r.ROOT/'braids_support/sediment/lfm_bend_2d/runs/fast_cpu_default_20261001_v1/runtime_mf25'
INPUT=r.ROOT/'braids_support/sediment/lfm_bend_2d/runs/fast_cpu_cold20h_20261001_v1_mobile_braids'


def instrument(source):
    start=source.index(' subroutine euler(');end=source.index(' end subroutine',start)
    block=source[start:end]
    block=r.replace_once(block,'  if(any(workspace_shape/=',
        '  real(8),allocatable :: gpu_bed_raw(:,:),gpu_bed_next(:,:),gpu_bed_rock(:)\n  integer :: gpu_bed_unit,gpu_bed_g,gpu_bed_layer\n  if(any(workspace_shape/=')
    block=r.replace_once(block,'  morfac=step_morfac', '''  ! Export only; every native formula below remains unchanged.
  if(amr_enabled.or.nfrac<1.or.bank_failure.or.bank_erosion_fraction/=0.d0.or. &
     bank_avalanche.or.general_avalanche.or.exchange_mode/=0) error stop 'GPU Exner export requires simple fixed-grid bedload'
  allocate(gpu_bed_raw(nfrac,nf),gpu_bed_next(nfrac,nc),gpu_bed_rock(nc))
  gpu_bed_rock=rock
  do gpu_bed_g=1,nfrac
   gpu_bed_raw(gpu_bed_g,:)=fl(bed_index(gpu_bed_g),:)
   do gpu_bed_layer=2,nbed_layers
    gpu_bed_rock=gpu_bed_rock+state(bed_layer_index(gpu_bed_layer,gpu_bed_g),:)/bulk_fraction(gpu_bed_g)
   enddo
  enddo
  morfac=step_morfac''')
    block=r.replace_once(block,'  call rebalance_active_layer(next)',
        '  gpu_bed_next=next(bed_index(1):bed_index(nfrac),:)\n  call rebalance_active_layer(next)')
    block=r.replace_once(block,'  ok=.true.', '''  open(newunit=gpu_bed_unit,file='bed_snapshot.bin',access='stream',form='unformatted',status='new')
  write(gpu_bed_unit) nc,nf,size(faces),nb,nfrac
  write(gpu_bed_unit) dt,step_morfac
  write(gpu_bed_unit) area,length,gpu_bed_rock,bulk_fraction
  write(gpu_bed_unit) owner-1,neighbour-1,offset-1,faces-1,signs,bid-1,merge(1,0,active_cell)
  write(gpu_bed_unit) state(bed_index(1):bed_index(nfrac),:),gpu_bed_raw,theta
  write(gpu_bed_unit) fl(bed_index(1):bed_index(nfrac),:),gpu_bed_next
  write(gpu_bed_unit) accepted(1+nfrac+1:1+nfrac+nfrac,:)
  close(gpu_bed_unit)
  stop 'GPU Exner snapshot complete, not a complete simulation'
  ok=.true.''')
    return source[:start]+block+source[end:]


def build(tag=TAG):
    root,report=r.paths(tag);root.mkdir(parents=True,exist_ok=False);report.mkdir(parents=True,exist_ok=False)
    runtime=root/'runtime_export';runtime.mkdir()
    identity=json.loads((PARENT/'build.json').read_text());sources=[];hashes={}
    for name,digest in identity['source_sha256'].items():
        source=PARENT/Path(name).name
        if r.sha(source)!=digest:raise ValueError('Frozen CPU source changed: '+name)
        dest=runtime/source.name
        if source.name=='cutcell_solver.f90':dest.write_text(instrument(source.read_text()))
        else:shutil.copy2(source,dest)
        sources.append(str(dest));hashes[dest.name]=r.sha(dest)
    cmd=[r'C:\Strawberry\c\bin\gfortran.exe','-O3','-flto','-fopenmp','-ffree-line-length-none',
         '-ffixed-line-length-none',*sources,'-o',str(runtime/'xbraids_export.exe')]
    proc=subprocess.run(cmd,cwd=runtime,capture_output=True,text=True,timeout=240)
    (runtime/'build.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
    for name in ('exner_update.py','exner_update.cu'):shutil.copy2(HERE/name,root/name)
    r.write(root/'build.json',dict(command=cmd,source_sha256=hashes,
        parent_executable_sha256=identity['executable_sha256'],executable_sha256=r.sha(runtime/'xbraids_export.exe'),
        cuda_sha256=r.sha(root/'exner_update.cu'),python_sha256=r.sha(root/'exner_update.py'),
        production_modified=False,delft_modified=False,scope='Read-only Euler export instrumentation'))
    folder=root/'bend';folder.mkdir()
    names=('case.data','case.data.bin','geometry.npz','case.data.json')
    for name in names:
        if (INPUT/name).exists():shutil.copy2(INPUT/name,folder/name)
    r.write(folder/'input_identity.json',{name:r.sha(folder/name) for name in names if (folder/name).exists()})
    proc=subprocess.run([str(runtime/'xbraids_export.exe'),'case.data'],cwd=folder,
        env=dict(os.environ,OMP_NUM_THREADS='2',OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',BRAIDS_TIME_INTEGRATOR='ssp43'),
        capture_output=True,text=True,timeout=120)
    (folder/'snapshot.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
    data=read_snapshot(folder/'bed_snapshot.bin')
    r.write(report/'snapshot.json',dict(cells=data['mass'].shape[1],faces=len(data['owner']),morfac=data['morfac'],
        snapshot_sha256=r.sha(folder/'bed_snapshot.bin'),source_build_sha256=r.sha(root/'build.json')))


def read_snapshot(path):
    with Path(path).open('rb') as stream:
        nc,nf,na,nb,ng=map(int,np.fromfile(stream,dtype='<i4',count=5))
        if min(nc,nf,na,ng)<1 or nb<0:raise ValueError('Invalid snapshot sizes')
        dt,mf=np.fromfile(stream,dtype='<f8',count=2);data=dict(dt=float(dt),morfac=float(mf),nb=nb)
        def read(name,n,parts=1,integer=False):
            v=np.fromfile(stream,dtype='<i4' if integer else '<f8',count=n*parts)
            if v.size!=n*parts:raise ValueError('Truncated '+name)
            data[name]=np.ascontiguousarray(v.reshape(n,parts).T) if parts>1 else v
        for key,n in (('area',nc),('length',nf),('rock',nc),('bulk',ng)):read(key,n)
        for key,n in (('owner',nf),('neighbour',nf),('offset',nc+1),('faces',na),('signs',na),('bid',nf),('active',nc)):
            read(key,n,integer=True)
        for key,n in (('mass',nc),('raw',nf),('theta',nc),('flux',nf),('next',nc),('ledger',nb)):
            read(key,n,ng);data[key]=data[key].reshape(ng,n)
        if stream.read(1):raise ValueError('Trailing snapshot bytes')
    validate_input(data);return data


def refreeze(prior='gpu_exner_update_20261001_v3',tag=TAG):
    """Reuse an authenticated untouched CPU oracle; freeze only the new CUDA/driver."""
    old,_=r.paths(prior);root,report=r.paths(tag)
    manifest=json.loads((old/'build.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        if r.sha(old/'runtime_export'/name)!=digest:raise ValueError('Oracle source changed')
    if r.sha(old/'runtime_export/xbraids_export.exe')!=manifest['executable_sha256']:raise ValueError('Oracle runtime changed')
    _,old_report=r.paths(prior);receipt=json.loads((old_report/'snapshot.json').read_text())
    if r.sha(old/'bend/bed_snapshot.bin')!=receipt['snapshot_sha256']:raise ValueError('Oracle snapshot changed')
    inputs=json.loads((old/'bend/input_identity.json').read_text())
    for name,digest in inputs.items():
        if r.sha(old/'bend'/name)!=digest:raise ValueError('Oracle input changed')
    root.mkdir(parents=True,exist_ok=False);report.mkdir(parents=True,exist_ok=False)
    shutil.copytree(old/'runtime_export',root/'runtime_export');shutil.copytree(old/'bend',root/'bend')
    for name in ('exner_update.py','exner_update.cu'):shutil.copy2(HERE/name,root/name)
    manifest.update(reused_oracle_from=str(old),prior_manifest_sha256=r.sha(old/'build.json'),
        cuda_sha256=r.sha(root/'exner_update.cu'),python_sha256=r.sha(root/'exner_update.py'))
    r.write(root/'build.json',manifest)
    receipt['source_build_sha256']=r.sha(root/'build.json');r.write(report/'snapshot.json',receipt)


def validate_input(data):
    mass=np.asarray(data['mass']);raw=np.asarray(data['raw'])
    if mass.ndim!=2 or raw.ndim!=2 or mass.shape[0]!=raw.shape[0] or min(mass.shape)<1 or raw.shape[1]<1:raise ValueError('Invalid mass/flux shape')
    ng,nc=mass.shape;nf=raw.shape[1];nb=data['nb']
    if not np.isfinite([data['dt'],data['morfac']]).all() or data['dt']<=0 or data['morfac']<=0:raise ValueError('Invalid time controls')
    for k in ('mass','raw','area','length','rock','bulk'):
        if not np.isfinite(data[k]).all():raise ValueError('Nonfinite '+k)
    if np.any(mass<0) or any(np.any(data[k]<=0) for k in ('area','length','bulk')):raise ValueError('Invalid positive field')
    shapes={'area':(nc,),'length':(nf,),'rock':(nc,),'bulk':(ng,),'owner':(nf,),'neighbour':(nf,),
            'offset':(nc+1,),'bid':(nf,),'active':(nc,)}
    for k,s in shapes.items():
        if np.asarray(data[k]).shape!=s:raise ValueError('Invalid shape '+k)
    for k in ('owner','neighbour','offset','faces','signs','bid','active'):
        values=np.asarray(data[k])
        if values.dtype.kind not in 'iu' or np.any(values>np.iinfo(np.int32).max) or np.any(values<np.iinfo(np.int32).min):
            raise ValueError('Topology must fit integer32')
    if np.any((data['owner']<0)|(data['owner']>=nc)) or np.any((data['neighbour']< -1)|(data['neighbour']>=nc)):raise ValueError('Invalid owner/neighbour')
    off=data['offset'];faces=data['faces'];sign=data['signs']
    if off[0]!=0 or off[-1]!=len(faces) or np.any(np.diff(off)<0) or sign.shape!=faces.shape:raise ValueError('Invalid adjacency')
    if np.any((faces<0)|(faces>=nf)) or not np.isin(sign,[-1,1]).all():raise ValueError('Invalid face or sign')
    if np.any((data['bid']< -1)|(data['bid']>=nb)) or not np.isin(data['active'],[0,1]).all():raise ValueError('Invalid boundary/activity')
    if np.any(data['owner']==data['neighbour']) or np.any((data['neighbour']>=0)&(data['bid']>=0)):
        raise ValueError('Self-neighbour or interior boundary ID')
    if not np.array_equal(np.bincount(faces[sign>0],minlength=nf),np.ones(nf,dtype=int)):
        raise ValueError('Every face requires one owner incidence')
    if not np.array_equal(np.bincount(faces[sign<0],minlength=nf),(data['neighbour']>=0).astype(int)):
        raise ValueError('Every interior face requires one neighbour incidence')
    closed=(data['neighbour']<0)&(data['bid']<0)
    if np.any(raw[:,closed]!=0):raise ValueError('Impermeable wall has nonzero bed flux')
    for c in range(nc):
        ff=faces[off[c]:off[c+1]];ss=sign[off[c]:off[c+1]]
        if np.any(np.where(ss>0,data['owner'][ff],data['neighbour'][ff])!=c):raise ValueError('Adjacency belongs to another cell')
    # This standalone fixed-grid operator may not skip a cell exchanging mass.
    inactive=np.flatnonzero(data['active']==0)
    for c in inactive:
        if np.any(raw[:,faces[off[c]:off[c+1]]]!=0):raise ValueError('Inactive cell has a nonzero bed flux')


class ResidentBedUpdate:
    def __init__(self,data,source):
        import cupy as cp
        validate_input(data);self.cp=cp;self.ng,self.nc=data['mass'].shape;self.nf=data['raw'].shape[1];self.nb=data['nb']
        self.geometry={k:cp.asarray(data[k],dtype=cp.float64) for k in ('area','length','rock','bulk')}
        self.geometry.update({k:cp.asarray(data[k],dtype=cp.int32) for k in ('owner','neighbour','offset','faces','signs','bid','active')})
        lists=[np.flatnonzero(data['bid']==b) for b in range(self.nb)]
        offsets=np.r_[0,np.cumsum([len(v) for v in lists])].astype(np.int32)
        self.geometry['boundary_offset']=cp.asarray(offsets)
        self.geometry['boundary_faces']=cp.asarray(np.concatenate(lists).astype(np.int32) if lists else np.array([],np.int32))
        names=('bed_scale','bed_donors','bed_limit','bed_gather','bed_ledger')
        self.module=cp.RawModule(code=Path(source).read_text(),options=r.OPTIONS,name_expressions=names)
        self.k={name:self.module.get_function(name) for name in names}
        self.buffers={'theta':cp.empty((self.ng,self.nc)), 'flux':cp.empty((self.ng,self.nf)),
            'next':cp.empty((self.ng,self.nc)), 'bed':cp.empty(self.nc), 'ledger':cp.empty((self.ng,self.nb)),
            'status':cp.empty(self.nc,dtype=cp.int32),'face_status':cp.empty((self.ng,self.nf),dtype=cp.int32)}

    def step(self,mass,raw,dt,morfac):
        cp=self.cp
        for name,array,shape in (('mass',mass,(self.ng,self.nc)),('raw',raw,(self.ng,self.nf))):
            if not isinstance(array,cp.ndarray) or array.dtype!=cp.float64 or array.shape!=shape or not array.flags.c_contiguous:raise ValueError('Invalid device '+name)
        if not np.isfinite([dt,morfac]).all() or min(dt,morfac)<=0:raise ValueError('Invalid time controls')
        a,b=self.geometry,self.buffers;nc,nf,ng=np.int64(self.nc),np.int64(self.nf),np.int32(self.ng)
        dt,mf=np.float64(dt),np.float64(morfac)
        def launch(name,n,args):self.k[name](((n+127)//128,),(128,),args)
        launch('bed_scale',self.nf*self.ng,(raw,a['owner'],a['neighbour'],a['bid'],a['active'],nf,ng,mf,b['flux'],b['face_status']))
        launch('bed_donors',self.nc*self.ng,(mass,a['area'],a['length'],a['offset'],a['faces'],a['signs'],a['active'],b['flux'],nc,nf,ng,dt,b['theta']))
        launch('bed_limit',self.nf*self.ng,(a['owner'],a['neighbour'],b['theta'],nc,nf,ng,b['flux']))
        launch('bed_gather',self.nc,(mass,a['area'],a['length'],a['offset'],a['faces'],a['signs'],a['active'],b['flux'],a['rock'],a['bulk'],b['face_status'],nc,nf,ng,dt,b['next'],b['bed'],b['status']))
        if self.nb:self.k['bed_ledger']((self.nb,self.ng),(128,),(b['flux'],a['length'],a['boundary_offset'],a['boundary_faces'],nf,ng,np.int32(self.nb),dt,b['ledger']))
        return b


def validate(tag=TAG):
    root,report=r.paths(tag);data=read_snapshot(root/'bend/bed_snapshot.bin')
    identity=json.loads((root/'build.json').read_text())
    if r.sha(root/'exner_update.cu')!=identity['cuda_sha256']:raise ValueError('Frozen CUDA source changed')
    engine=ResidentBedUpdate(data,root/'exner_update.cu');cp=engine.cp
    out=engine.step(cp.asarray(data['mass']),cp.asarray(data['raw']),data['dt'],data['morfac'])
    actual={k:cp.asnumpy(v) for k,v in out.items()}
    errors={k:float(np.max(abs(actual[k]-data[k]))) if actual[k].size else 0. for k in ('theta','flux','next','ledger')}
    residual=(actual['next']-data['mass'])@data['area']-actual['ledger'].sum(axis=1)
    result=dict(errors=errors,max_mass_residual_kg=float(abs(residual).max()),
        min_mass_kg_m2=float(actual['next'].min()),status=int(actual['status'].max()),
        all_passed=bool(all(v<2e-11 for v in errors.values()) and abs(residual).max()<1e-8 and actual['status'].max()==0),
        input_flux_origin='Unchanged frozen CPU sediment face calculations; explicit boundary of this proof',
        active_layer_rebalancing_ported=False,compared_inventory='Active inventory after face update, before native rebalancing',
        not_complete_coupled_GPU_solver=True,performance_qualified=False,production_modified=False,delft_modified=False,
        source_manifest_sha256=r.sha(root/'build.json'),snapshot_sha256=r.sha(root/'bend/bed_snapshot.bin'),
        validation_driver_sha256=r.sha(__file__))
    archive=root/'gpu_bed_arrays.npz'
    if archive.exists():
        saved=np.load(archive)
        if any(not np.array_equal(saved[k],v) for k,v in actual.items()):raise ValueError('Previous GPU arrays differ')
    else:np.savez_compressed(archive,**actual)
    r.write(report/'validation_complete.json',result);print(json.dumps(result),flush=True)
    if not result['all_passed']:raise RuntimeError('GPU Exner post-flux qualification failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('build','validate','refreeze'));p.add_argument('--tag',default=TAG)
    args=p.parse_args();globals()[args.action](**{'tag':args.tag})
