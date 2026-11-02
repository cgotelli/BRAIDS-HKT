"""Isolated fixed-grid FP64 resident hydraulic pipeline.

Production/QGIS is not changed. This pipeline rejects unported physics rather
than substituting simpler formulas. Reconstruction, supported boundary controls,
augmented f-wave fluxes, draining, conservative gather, SSPRK43 and Manning are
GPU resident. Python orchestrates kernels and reads only small control scalars;
saved output/validation is an explicit device-to-host operation.
"""
from pathlib import Path
import json
import shutil
import numpy as np

from . import resident_update as old
from . import reconstruction as recon
from . import boundaries as bc
from .time_integration import SSPRK43

HERE=Path(__file__).resolve().parent
FLUX_REFERENCE=HERE/'runs/gpu_resident_update_20261001_v2'


def freeze(destination):
    """Freeze the exact current driver and CUDA inputs; never overwrite a run."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((FLUX_REFERENCE/'resident_build.json').read_text())
    for name in ('augmented_fwave.cu','resident_update.cu'):
        source=FLUX_REFERENCE/name
        assert old.sha(source)==manifest['source_sha256'][name],name
        shutil.copy2(source,destination/name)
    names=('reconstruction.cu','reconstruction.py','boundaries.cu','boundaries.py',
           'time_integration.cu','time_integration.py','hydraulic_pipeline.py','hydraulic_braids_support.py',
           'resident_update.py','reconstruction_oracle.py')
    for name in names:shutil.copy2(HERE/name,destination/name)
    names=names+('augmented_fwave.cu','resident_update.cu')
    receipt=dict(files={name:old.sha(destination/name) for name in names},
        cuda_options=list(old.OPTIONS),fixed_grid_only=True,hydraulic_only=True,
        time_integrator='SSPRK(4,3); existing split Manning',
        production_modified=False,delft_modified=False)
    old.write(destination/'pipeline_build.json',receipt)
    return receipt


def verify_frozen(folder):
    folder=Path(folder);receipt=json.loads((folder/'pipeline_build.json').read_text())
    for name,digest in receipt['files'].items():
        if old.sha(folder/name)!=digest:raise ValueError('Frozen GPU source changed: '+name)
        if name.endswith('.py') and old.sha(HERE/name)!=digest:
            raise ValueError('Live GPU driver differs from frozen source; create a new qualification: '+name)
    return receipt


def validate_scope(data):
    """Do not silently drop physics when used outside the controlled oracle."""
    physics=data.get('physics')
    expected=dict(scheme=6,amr=False,bedlaw=0,exchange='off',secondary=False,
                  viscosity=0.,dynamic_roughness=False,banks=False,sediment_scalars=False)
    if not isinstance(physics,dict) or set(physics)!=set(expected):
        raise ValueError('Explicit hydraulic-only physics identity required')
    for key,value in expected.items():
        if physics[key]!=value:raise ValueError('GPU hydraulic port excludes '+key)
    for key in ('amr_enabled','bedlaw','exchange_mode','secondary_flow','horizontal_viscosity'):
        if np.any(data.get(key,0)):raise ValueError('GPU hydraulic port excludes '+key)
    if np.any(data.get('nmorfac',0)):raise ValueError('GPU hydraulic port excludes MF series')
    for key in ('area','length'):
        values=np.asarray(data[key])
        if not np.isfinite(values).all() or np.any(values<=0):raise ValueError('Invalid '+key)
    manning=np.asarray(data['manning_cell'])
    if not np.isfinite(manning).all() or np.any(manning<0):raise ValueError('Invalid Manning roughness')


def load_snapshot(folder):
    """Authenticate the controlled hydraulic oracle and its input physics."""
    from . import reconstruction_oracle as oracle
    folder=Path(folder)
    receipt=json.loads((folder/'snapshot_receipt.json').read_text())
    for name,key in (('gpu_snapshot.bin','snapshot_sha256'),
                     ('reconstruction_snapshot.bin','reconstruction_snapshot_sha256'),
                     ('input_provenance.json','input_provenance_sha256')):
        if old.sha(folder/name)!=receipt[key]:raise ValueError('Oracle artifact changed: '+name)
    provenance=json.loads((folder/'input_provenance.json').read_text())
    if not {'case.data','case.data.json'}<=set(provenance['input_files']):
        raise ValueError('Oracle manifest must authenticate the input deck and physics metadata')
    for name,digest in provenance['input_files'].items():
        if old.sha(folder/name)!=digest:raise ValueError('Oracle input changed: '+name)
    metadata=json.loads((folder/'case.data.json').read_text())
    binary=metadata.get('sidecars',{}).get('native_binary')
    if binary and binary not in provenance['input_files']:
        raise ValueError('Oracle manifest must authenticate the native binary sidecar')
    data=oracle.read_snapshot(folder)
    data['physics']=dict(scheme=metadata['reconstruction_scheme'],amr=metadata['amr'] is not None,
        bedlaw=metadata['bedload_law'],exchange=metadata['suspended_exchange_name'],
        secondary=metadata['secondary_flow']['enabled'],viscosity=metadata['horizontal_viscosity_m2_s'],
        dynamic_roughness=metadata['bedform_roughness']['enabled'] or metadata['materials']['roughness_mode']!='fixed',
        banks=any(metadata[key]['enabled'] for key in ('bank_morphology','bank_failure','bank_front_tracking','general_avalanche')),
        sediment_scalars=bool(metadata['fraction_ids']))
    validate_scope(data)
    return data


def boundary_data(data):
    values=dict(data)
    # Six native forcing slots include unused sediment controls. The strict
    # hydraulic oracle excludes their physics; only Q and stage/depth enter BCs.
    values['forcing']=np.ascontiguousarray(data['forcing'][:2])
    values['manning']=np.ascontiguousarray(data['manning_cell'])
    return values


class ResidentFlux(old.ResidentUpdate):
    """Reuse the frozen tested GPU face/update methods without oracle upload."""
    def __init__(self,data,reconstruction,source_folder):
        import cupy as cp
        self.cp=cp;self.nc=reconstruction.nc;self.nf=reconstruction.nf
        self.inputs={k:reconstruction.geometry[k] for k in
            ('bed','normal','owner','neighbour','offset','faces','signs','active','boundary_mode')}
        for k in ('length','area','sampled','bedoffset','distance'):
            self.inputs[k]=cp.asarray(data[k])
        for k in ('trace','hydro_constant','kind'):
            self.inputs[k]=reconstruction.buffers[k]
        self.inputs['q']=None
        self.controls={key:np.float64(data[key]) for key in ('dt','grav','dry','cfl_target')}
        from braids_core.windows_paths import filesystem_path
        folder=filesystem_path(source_folder)
        code=(folder/'augmented_fwave.cu').read_text()+'\n'+(folder/'resident_update.cu').read_text()
        names=('resident_prepare_faces','augmented_fwave_local','resident_assemble_flux',
               'resident_draining','resident_limit_flux','resident_gather_update')
        self.module=cp.RawModule(code=code,options=old.OPTIONS,name_expressions=names)
        self.kernels={name:self.module.get_function(name) for name in names}
        nf,nc=self.nf,self.nc
        self.buffers={key:cp.empty(shape,dtype=cp.float64) for key,shape in {
            'local':(8,nf),'riemann':(18,nf),'theta':(nc,), 'fl':(3,nf),'fr':(3,nf),
            'raw_fl':(3,nf),'raw_fr':(3,nf),'speed':(nf,),'next':(3,nc),
            'cfl_ratio':(nc,),'maxdepth':()}.items()}
        for key,size in (('closed',nf),('augmented_status',nf),('face_status',nf),('cell_status',nc)):
            self.buffers[key]=cp.empty(size,dtype=cp.int32)


class HydraulicPipeline:
    def __init__(self,data,source_folder,*,outlet_extension=None):
        import cupy as cp
        self.cp=cp;self.folder=Path(source_folder);verify_frozen(self.folder)
        validate_scope(data)
        self.data=data;self.nc=data['q'].shape[1];self.nf=len(data['owner'])
        self.reconstruction=recon.ResidentReconstruction(data,source_path=self.folder/'reconstruction.cu')
        if outlet_extension is not None:self.reconstruction.outlet_extension=int(outlet_extension)
        self.flux=ResidentFlux(data,self.reconstruction,self.folder)
        self.boundary=bc.BoundaryControls(boundary_data(data),self.flux.inputs,source=self.folder/'boundaries.cu')
        self.nb=self.boundary.nb
        self.q=cp.asarray(data['q']);self.ledger=cp.zeros(self.nb,dtype=cp.float64)
        self.status=cp.zeros(max(self.nc,self.nf,self.nb),dtype=cp.int32)
        self.status_kernel=cp.RawModule(code=(self.folder/'boundaries.cu').read_text(),options=old.OPTIONS,
            name_expressions=('gpu_pipeline_status',)).get_function('gpu_pipeline_status')
        self.stepper=SSPRK43(self.nc,self.nb,self.boundary.arrays['manning'],grav=data['grav'],
            dry_tolerance=data['dry'],speed_limit=data['speed_limit'],
            chezy_minimum_depth=data['manning_friction_chezy_minimum_depth'],
            source_path=self.folder/'time_integration.cu')
        self.euler_calls=0

    def euler(self,q,dt,time,*,capture_raw=False):
        """No field transfers: refresh reconstruction/BCs from this stage state."""
        r=self.reconstruction.reconstruct(q)
        self.boundary.apply(q,r['trace'])
        self.flux.inputs['q']=q;self.flux.controls['dt']=np.float64(dt)
        f=self.flux.step(capture_raw=capture_raw)
        ledger=self.boundary.integrate_ledger(f['fl'],dt)
        self.status_kernel(((self.status.size+127)//128,),(128,),
            (r['primitive_status'],r['reconstruction_status'],f['augmented_status'],f['face_status'],f['cell_status'],
             self.boundary.status,self.boundary.face_status,np.int64(self.nc),np.int64(self.nf),np.int32(self.nb),self.status))
        self.euler_calls+=1
        return dict(next=f['next'],ledger=ledger,status=self.status,cfl=f['cfl_ratio'])

    def step(self,dt,time,*,commit=True):
        # SSP43 has no outlet-bound extension; that spatial repair belongs to
        # the separately qualified CPU multistep method, not this integrator.
        if self.reconstruction.outlet_extension:
            raise ValueError('Full SSP43 GPU stepping requires outlet_extension=0')
        return self.stepper.step(self.q,self.ledger,dt,time,self.euler,commit=commit)

    def fields(self):
        """Explicit output/analysis boundary; not called by any stage kernel."""
        return self.cp.asnumpy(self.q),self.cp.asnumpy(self.ledger)
