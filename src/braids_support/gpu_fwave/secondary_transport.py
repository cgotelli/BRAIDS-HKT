"""Fixed-grid native secondary-flow branches for a coupled GPU stepper.

The caller owns SSP43/Strang ordering and transactional acceptance. In model 2,
never call relax() on accepted state until the whole step is accepted; it mutates
hI. Its curvature is refreshed only by relax(), as in native alpha82; all Euler
stages use the retained first-half-source curvature for boundary inflow.
The legacy equilibrium branch instead derives raw curvature and hI from the
current Euler stage solely for bedload steering; it transports no hI inventory.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from .secondary_curvature import ResidentCurvature,OPTIONS

HERE=Path(__file__).resolve().parent
OUTLET_ABI = '#define BRAIDS_SECONDARY_OUTLET_ABI 1'
LEGACY_EXTENSION_ONE_SHA256 = '5b6836851b646b432ac1d212ed2f752b2b35fcff908020285cadcf8393ca3a13'
# Authenticated October 2 fixed-grid source, before the recursive extension.
LEGACY_FIXED_GRID_EXTENSION_ONE_SHA256 = 'c5b9c26ad48adca908ff2eb9a6b85b9ce9b078f68430f76f8413e1473197e3ac'


def outlet_control(data, reconstruction=None):
    """Bind secondary limiter bounds to the same explicit hydraulic control."""
    values=[]
    for name in ('outlet_extension','cpu_outlet_extension'):
        if name not in data:continue
        value=np.asarray(data[name])
        if value.shape!=() or value.dtype.kind not in 'iu' or int(value) not in (0,1):
            raise ValueError('Secondary outlet extension must be integer 0 or 1')
        values.append(int(value))
    if values and len(set(values))!=1:
        raise ValueError('Secondary CPU/GPU outlet extension controls disagree')
    value=values[0] if values else 0
    if reconstruction is not None:
        hydraulic=np.asarray(reconstruction.outlet_extension)
        if hydraulic.shape!=() or hydraulic.dtype.kind not in 'iu' or int(hydraulic)!=value:
            raise ValueError('Secondary/hydraulic outlet extension controls disagree')
    return value


def outlet_kernel_interface(code, extension):
    """Historical unmarked kernel represents extension one only, by exact SHA."""
    text=code.decode('utf-8')
    markers=[line.strip() for line in text.splitlines() if 'BRAIDS_SECONDARY_OUTLET_ABI' in line]
    if markers==[OUTLET_ABI]:return True
    if not markers and extension==1 and hashlib.sha256(code).hexdigest() in (
            LEGACY_EXTENSION_ONE_SHA256, LEGACY_FIXED_GRID_EXTENSION_ONE_SHA256):
        return False
    raise ValueError('Secondary outlet CUDA ABI/control pairing differs')


def scalar(data,name,default=None):
    if name not in data:
        if default is None:raise ValueError('Missing secondary control '+name)
        return default
    value=np.asarray(data[name])
    if value.size!=1:raise ValueError('Expected scalar '+name)
    return value.item()


def validate_scope(data):
    model=int(scalar(data,'secondary_flow_model'))
    if model not in (0,1,2,3):raise ValueError('Unsupported GPU secondary-flow model')
    if int(scalar(data,'scheme',6))!=6:raise ValueError('GPU secondary reconstruction requires scheme 6')
    if bool(scalar(data,'amr_enabled',False)):raise ValueError('GPU secondary transport does not support AMR')
    if model==2 and int(scalar(data,'secondary_flow_inflow_mode',1))!=1:
        raise ValueError('Only native equilibrium secondary inflow is supported')
    controls=[scalar(data,'grav',9.81),scalar(data,'dry_tolerance',scalar(data,'dry',1e-4)),
              scalar(data,'secondary_flow_espir',1.),scalar(data,'secondary_flow_momentum_beta',0.)]
    if not np.isfinite(controls).all() or controls[0]<=0 or controls[1]<=0 or min(controls[2:])<0:
        raise ValueError('Invalid secondary-flow controls')
    enabled=bool(scalar(data,'secondary_flow',model!=0))
    direction=bool(scalar(data,'secondary_flow_transport_direction',model!=0))
    if enabled!=(model!=0):raise ValueError('Secondary-flow enable flag contradicts model')
    if model==0 and (direction or controls[3]):
        raise ValueError('OFF secondary flow cannot request direction or momentum feedback')
    if model==1 and (not direction or controls[3]):
        raise ValueError('Equilibrium secondary flow requires direction-only coupling')
    if model==3 and (not direction or controls[3]):
        raise ValueError('TELEMAC-GAIA secondary flow requires stress-vector transport coupling and zero momentum Beta')
    return controls


class SecondaryTransport:
    """One fixed mesh; arrays remain GPU-resident. No backend auto-selection.

    Hook order each Euler: hydraulic reconstruct; secondary reconstruct;
    hydraulic+sediment face flux assembly; add_stress; hydraulic water draining
    and gather; advect using final shared water flux. All statuses must feed
    the stepper's rejection gate. Source and boundary ledgers are signed.
    """
    def __init__(self,data,reconstruction=None,*,source_path=None,curvature_source_path=None):
        import cupy as cp
        self.outlet_extension=outlet_control(data,reconstruction)
        self.cp=cp;controls=validate_scope(data);self.grav,self.dry,self.espir,self.beta=map(np.float64,controls)
        self.model=int(scalar(data,'secondary_flow_model'))
        self.constant_chezy=np.float64(scalar(data,'engelund_hansen_chezy',0.)
            if int(scalar(data,'bedlaw',0))==8 else 0.)
        if not np.isfinite(self.constant_chezy) or self.constant_chezy<0.:
            raise ValueError('Invalid secondary-flow constant Chezy coefficient')
        # Law 8 owns its capacity Chezy even when secondary flow is disabled.
        # Model 0 returns before constructing any steering kernel; the value
        # is inert here and must not reject an otherwise admitted OFF route.
        if self.constant_chezy>0. and self.model not in (0,1,2):
            raise ValueError('Constant-Chezy secondary steering requires Delft3D model 1 or 2')
        self.direction=bool(scalar(data,'secondary_flow_transport_direction',self.model!=0))
        self.nc=len(data['area']);self.nf=len(data['owner']);self.ng=int(scalar(data,'nfrac',1));self.reconstruction=reconstruction
        self.nadj=len(data['faces']);self.nb=len(data.get('boundary_offset',[]))-1
        if self.nb<0:self.nb=len(data.get('mode',[]))
        if self.model!=2:
            # CoupledRK43 retains a neutral secondary array in its checkpoint
            # schema, but these two CPU branches have no transported hI state.
            self.inputs={};self.kernels={};self.filter=None;self.last_filter_receipt=None
            self.buffers={'curvature':cp.zeros(self.nc), 'equilibrium_intensity':cp.zeros(self.nc),
                          'equilibrium_mass':cp.zeros(self.nc),
                          'raw_curvature':cp.zeros(self.nc), 'raw_status':cp.zeros(self.nc,dtype=cp.int32),
                          'equilibrium_status':cp.zeros(self.nc,dtype=cp.int32),
                          'rotation_status':cp.zeros(self.nc*self.ng,dtype=cp.int32),
                          'constant_chezy_manning':cp.empty(self.nc),
                          'ledger':cp.zeros(self.nb), 'zero_status':cp.zeros((),dtype=cp.int32),
                          'source_integral':cp.zeros(()), 'traction':cp.zeros((2,self.nf))}
            if self.model==1:
                for name in ('owner','neighbour','offset','faces','signs'):
                    self.inputs[name]=cp.asarray(np.ascontiguousarray(data[name],dtype=np.int32))
                self.inputs['xy']=cp.asarray(np.ascontiguousarray(data['xy'],dtype=np.float64))
                manning=np.asarray(data.get('manning_cell',data.get('manning')),dtype=np.float64)
                if manning.shape!=(self.nc,) or not np.isfinite(manning).all() or np.any(manning<=0):
                    raise ValueError('Positive finite cell Manning coefficients required')
                self.inputs['manning']=cp.asarray(manning)
                names=('raw_curvature','equilibrium','rotate')
                self.module=cp.RawModule(code=Path(source_path or HERE/'secondary_transport.cu').read_text(),
                                         options=OPTIONS,name_expressions=tuple('st_'+name for name in names))
                self.kernels={name:self.module.get_function('st_'+name) for name in names}
            return
        integer=('owner','neighbour','bid','offset','faces','signs','weno_cells','wall_stencil_ids','wall_slot','boundary_offset','boundary_faces')
        real=('area','xy','mid','normal','length','hydraulic_ls_matrix','hydraulic_ls_weight','bed_ls_delta','face_displacement')
        self.inputs={name:cp.asarray(np.asarray(data[name],dtype=np.int32),order='C') for name in integer}
        self.inputs.update({name:cp.asarray(np.asarray(data[name],dtype=np.float64),order='C') for name in real})
        if 'boundary_mode' in data:bmode=np.asarray(data['boundary_mode'],dtype=np.int32)
        else:
            bid=np.asarray(data['bid']);bmode=np.zeros(self.nf,dtype=np.int32);valid=bid>=0;bmode[valid]=np.asarray(data['mode'])[bid[valid]]
        self.inputs['boundary_mode']=cp.asarray(bmode)
        self.inputs['manning']=cp.asarray(np.asarray(data.get('manning_cell',data.get('manning')),dtype=np.float64))
        if self.inputs['manning'].shape!=(self.nc,):raise ValueError('Missing cell Manning coefficients')
        if not np.isfinite(np.asarray(data.get('manning_cell',data.get('manning')))).all() or np.any(np.asarray(data.get('manning_cell',data.get('manning')))<=0):
            raise ValueError('Positive finite Manning coefficients required')
        names=('raw_curvature','relax','primitive','reconstruct_cells','reconstruct_faces','add_stress','low_flux','fct_bounds','fct_faces','fct_update','ledger','rotate')
        code=Path(source_path or HERE/'secondary_transport.cu').read_bytes()
        self.outlet_dispatch=outlet_kernel_interface(code,self.outlet_extension)
        self.module=cp.RawModule(code=code.decode('utf-8'),options=OPTIONS,
                                 name_expressions=tuple('st_'+name for name in names))
        self.kernels={name:self.module.get_function('st_'+name) for name in names}
        self.buffers={name:cp.empty(self.nc,dtype=cp.float64) for name in ('raw_curvature','curvature','intensity','source_work','lowstate','plus','minus','next')}
        # The transport-law Chezy coefficient belongs to sediment steering,
        # not to hydraulic friction, spiral adaptation or momentum stress.
        # A separate stage-local equivalent n preserves the existing CUDA ABI
        # and cannot overwrite the configured hydraulic Manning coefficients.
        self.buffers['constant_chezy_manning']=cp.empty(self.nc,dtype=cp.float64)
        self.buffers['curvature'].fill(0.)
        self.buffers.update({name:cp.empty(self.nf,dtype=cp.float64) for name in ('low_flux','anti_flux','boundary_intensity','flux')})
        self.buffers['gradient']=cp.empty((2,self.nc),dtype=cp.float64);self.buffers['face_intensity']=cp.empty((2,self.nf),dtype=cp.float64)
        self.buffers['traction']=cp.empty((2,self.nf),dtype=cp.float64);self.buffers['ledger']=cp.zeros(self.nb,dtype=cp.float64)
        for name in ('raw_status','source_status','primitive_status','fct_status'):self.buffers[name]=cp.zeros(self.nc,dtype=cp.int32)
        for name in ('reconstruction_status','stress_status','flux_status'):self.buffers[name]=cp.zeros(self.nf,dtype=cp.int32)
        self.buffers['rotation_status']=cp.zeros(self.nc*self.ng,dtype=cp.int32)
        self.filter=ResidentCurvature(data,curvature_source_path,parallel_coarse=True)
        self.last_filter_receipt=None

    def _array(self,array,shape):
        if not isinstance(array,self.cp.ndarray) or array.dtype!=self.cp.float64 or array.shape!=shape or not array.flags.c_contiguous:
            raise ValueError(f'Expected resident C-contiguous FP64 array {shape}')

    def launch(self,name,n,args):
        self.kernels[name](((max(n,1)+127)//128,),(128,),tuple(args))

    def _hydro(self,buffers=None):
        if buffers is not None:return buffers
        if self.reconstruction is None:raise ValueError('Hydraulic reconstruction buffers required')
        return self.reconstruction.buffers

    def relax(self,q,secondary,dt):
        self._array(q,(3,self.nc));self._array(secondary,(self.nc,))
        if not np.isfinite(dt) or dt<0:raise ValueError('Nonnegative source timestep required')
        if self.model!=2:
            return dict(source_integral=self.buffers['source_integral'],status=self.buffers['zero_status'],
                        curvature=self.buffers['curvature'],filter_receipt=None)
        a=self.inputs;b=self.buffers;nc=np.int32(self.nc);nf=np.int32(self.nf)
        self.launch('raw_curvature',self.nc,(nc,nf,q,a['xy'],a['owner'],a['neighbour'],a['offset'],a['faces'],a['signs'],self.dry,b['raw_curvature'],b['raw_status']))
        if int(self.cp.max(b['raw_status']).get()):raise RuntimeError('Invalid raw secondary curvature')
        filtered,self.last_filter_receipt=self.filter.solve(q[0],b['raw_curvature'],dry=float(self.dry),depth_factor=2.,rtol=1.e-4,atol=1.e-10,graph_batch=5)
        self.cp.copyto(b['curvature'],filtered)
        self.launch('relax',self.nc,(nc,q,secondary,b['curvature'],a['manning'],a['area'],self.dry,self.grav,np.float64(dt),b['source_work'],b['source_status']))
        return dict(source_integral=self.cp.sum(b['source_work']),status=self.cp.max(b['source_status']),curvature=b['curvature'],filter_receipt=self.last_filter_receipt)

    def reconstruct(self,q,bed,secondary,hydraulic_buffers=None):
        self._array(q,(3,self.nc));self._array(bed,(self.nc,));self._array(secondary,(self.nc,))
        if self.model!=2:
            b=self.buffers
            if self.model==1:
                a=self.inputs;nc=np.int32(self.nc);nf=np.int32(self.nf)
                self.launch('raw_curvature',self.nc,(nc,nf,q,a['xy'],a['owner'],a['neighbour'],
                    a['offset'],a['faces'],a['signs'],self.dry,b['raw_curvature'],b['raw_status']))
                self.cp.copyto(b['curvature'],b['raw_curvature'])
                self.launch('equilibrium',self.nc,(nc,q,b['curvature'],self.dry,
                    b['equilibrium_intensity'],b['equilibrium_mass'],b['equilibrium_status']))
                status=self.cp.maximum(self.cp.max(b['raw_status']),self.cp.max(b['equilibrium_status']))
            else:status=b['zero_status']
            return dict(intensity=b['equilibrium_intensity'],equilibrium_mass=b['equilibrium_mass'],status=status)
        a=self.inputs;b=self.buffers;h=self._hydro(hydraulic_buffers);nc=np.int32(self.nc);nf=np.int32(self.nf)
        self.launch('primitive',self.nc,(nc,q,secondary,b['intensity'],b['primitive_status']))
        args=(nc,nf,np.int32(self.nadj),q,bed,b['intensity'],a['xy'],a['mid'],a['normal'],a['owner'],a['neighbour'],a['bid'],a['boundary_mode'],a['offset'],a['faces'],a['signs'],h['active'],a['hydraulic_ls_matrix'],a['hydraulic_ls_weight'],a['bed_ls_delta'],self.dry,self.grav)
        if self.outlet_dispatch:args+=(np.int32(self.outlet_extension),)
        self.launch('reconstruct_cells',self.nc,args+(b['gradient'],))
        self.launch('reconstruct_faces',self.nf,(nc,nf,b['intensity'],b['gradient'],a['owner'],a['neighbour'],a['weno_cells'],a['face_displacement'],a['wall_stencil_ids'],a['wall_slot'],h['wall_ready'],b['face_intensity'],b['reconstruction_status']))
        return dict(intensity=b['intensity'],face_intensity=b['face_intensity'],status=self.cp.maximum(self.cp.max(b['primitive_status']),self.cp.max(b['reconstruction_status'])))

    def add_stress(self,trace,kind,closed,fl,fr,speed):
        if self.model!=2:
            return dict(traction=self.buffers['traction'],status=self.buffers['zero_status'])
        a=self.inputs;b=self.buffers
        self.launch('add_stress',self.nf,(np.int32(self.nc),np.int32(self.nf),trace,b['face_intensity'],a['manning'],a['normal'],a['owner'],a['neighbour'],kind,closed,self.dry,self.grav,self.beta,fl,fr,speed,b['traction'],b['stress_status']))
        return dict(traction=b['traction'],status=self.cp.max(b['stress_status']))

    def advect(self,q,next_q,secondary,water_flux,dt,hydraulic_buffers=None):
        self._array(q,(3,self.nc));self._array(next_q,(3,self.nc));self._array(secondary,(self.nc,))
        if not np.isfinite(dt) or dt<=0:raise ValueError('Positive Euler timestep required')
        if water_flux.shape==(3,self.nf):water_flux=water_flux[0]
        if self.model!=2:
            return dict(next=secondary,ledger=self.buffers['ledger'],flux=None,status=self.buffers['zero_status'])
        self._array(water_flux,(self.nf,));a=self.inputs;b=self.buffers;h=self._hydro(hydraulic_buffers)
        nc=np.int32(self.nc);nf=np.int32(self.nf);step=np.float64(dt)
        self.launch('low_flux',self.nf,(nc,nf,q,b['intensity'],b['face_intensity'],b['curvature'],a['owner'],a['neighbour'],water_flux,self.dry,b['low_flux'],b['anti_flux'],b['boundary_intensity'],b['flux_status']))
        self.launch('fct_bounds',self.nc,(nc,nf,b['intensity'],secondary,next_q,a['area'],a['length'],a['owner'],a['neighbour'],a['bid'],a['offset'],a['faces'],a['signs'],h['active'],water_flux,b['low_flux'],b['anti_flux'],b['boundary_intensity'],step,b['lowstate'],b['plus'],b['minus'],b['fct_status']))
        self.launch('fct_faces',self.nf,(nf,a['owner'],a['neighbour'],b['low_flux'],b['plus'],b['minus'],b['anti_flux'],b['flux']))
        self.launch('fct_update',self.nc,(nc,next_q,secondary,b['lowstate'],a['area'],a['length'],a['offset'],a['faces'],a['signs'],h['active'],b['anti_flux'],step,b['next'],b['fct_status']))
        if self.nb:self.kernels['ledger']((self.nb,),(128,),(np.int32(self.nb),a['boundary_offset'],a['boundary_faces'],a['length'],b['flux'],step,b['ledger']))
        return dict(next=b['next'],ledger=b['ledger'],flux=b['flux'],status=self.cp.maximum(self.cp.max(b['fct_status']),self.cp.max(b['flux_status'])))

    def rotate_capacity(self,q,secondary,capacity):
        self._array(q,(3,self.nc));self._array(secondary,(self.nc,));self._array(capacity,(2,self.ng,self.nc))
        if not self.direction:
            self.buffers['rotation_status'].fill(0)
            return self.buffers['rotation_status']
        mass=self.buffers['equilibrium_mass'] if self.model==1 else secondary
        manning=self.inputs['manning']
        if self.constant_chezy>0.:
            # Preserve the qualified CUDA ABI.  For either Delft3D branch,
            # n=h^(1/6)/C makes the existing kernel evaluate exactly the
            # configured constant Chezy at every Euler stage.
            manning=self.buffers['constant_chezy_manning']
            self.cp.power(q[0],1./6.,out=manning);manning/=self.constant_chezy
        self.launch('rotate',self.nc*self.ng,(np.int32(self.nc),np.int32(self.ng),q,mass,manning,self.dry,self.grav,self.espir,capacity,self.buffers['rotation_status']))
        return self.buffers['rotation_status']
