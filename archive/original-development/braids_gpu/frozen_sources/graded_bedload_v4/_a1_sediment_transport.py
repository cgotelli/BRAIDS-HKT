"""Resident MPM flux and shared multilayer bed-state operations.

The fixed-grid path supports distinct noncohesive MPM grains and mapped
hiding/exposure. One substrate keeps its qualified original kernels; ordered
multiple substrates use the separately qualified resident exchange kernels.
Unsupported physics is rejected, never silently disabled.
MorFac, donor limiting and time integration belong to the coupled caller.
Bedlaw 0 may use only bed elevation and conservative layer rebalance; its
horizontal flux is identically zero and must not call the MPM flux methods.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .normal_outlet_boundary import validate_boundaries

HERE=Path(__file__).resolve().parent
OPTIONS=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true','--ftz=false')


def _value(data,*names,default=None):
    for name in names:
        if name in data:return data[name]
    if default is not None:return default
    raise ValueError('Missing sediment control '+names[0])


def configuration(data,config=None,*,external_suspension=False,allow_scheduled=False,
                  allow_bedform=False,allow_general_avalanche=False,
                  allow_physical_toe_mode2=False,
                  allow_distributed_bank_mode3=False,allow_adaptation=False):
    """Normalize native-export names; defaults cover only absent inert switches."""
    source=dict(data);source.update(config or {})
    validate_boundaries(source)
    if 'sediment_type_fraction' in source and np.any(np.asarray(source['sediment_type_fraction'])==1):
        raise ValueError('Cohesive fractions are not ported')
    if bool(source.get('bank_front_tracking',False)):
        raise ValueError('Bank-front tracking is not ported')
    if not external_suspension:
        has_initial=('suspended_mass' in source and np.any(source['suspended_mass']))
        if 'state' in source and 'suspended_indices' in source:
            has_initial=has_initial or np.any(np.asarray(source['state'])[np.asarray(source['suspended_indices'],dtype=int)])
        has_boundary=('fraction_concentration' in source and np.any(source['fraction_concentration']))
        if bool(source.get('suspended_transport_active',False)) or has_initial or has_boundary:
            raise ValueError('Suspended transport requires a coupled suspension operator')
    if int(_value(source,'deck_version',default=13))<13:raise ValueError('Explicit active/substrate layers required')
    out={name:np.ascontiguousarray(_value(source,native,name),dtype=np.float64).reshape(-1)
         for name,native in [('d50','d50_fraction'),('d90','d90_fraction'),('rhos','rhos_fraction'),
                             ('bulk','bulk_fraction'),('min_depth','bedload_min_depth_fraction')]}
    aliases={
        'rho':('rho','water_density'),'grav':('grav','gravity'),'dry':('dry_tolerance','dry'),
        'critical':('mpm_critical','critical'),'cf':('mpm_cf','cf'),'alfabs':('bed_slope_alfabs','alfabs'),
        'alfabn':('bed_slope_alfabn','alfabn'),'angle':('bed_friction_angle_deg','angle'),
        'target':('active_layer_thickness','target'),'dx':('dx',),'dy':('dy',)}
    for key,names in aliases.items():out[key]=float(_value(source,*names))
    out['morfac']=float(_value(source,'morphological_factor','morfac',default=1.))
    for key in ('scheme','bedlaw','mpm_friction_mode','bed_slope','secondary_flow_model','active_layer_mode','nsubstrate_layers'):
        out[key]=int(_value(source,key))
    if out['nsubstrate_layers']>1:
        out['substrate_thickness']=float(_value(source,'substrate_layer_thickness','substrate_thickness'))
    out['secondary_direction']=bool(_value(source,'secondary_flow_transport_direction','secondary_direction',default=False))
    for key in ('amr_enabled','bedload_adaptation','bedform_roughness','bank_failure','bank_avalanche','general_avalanche'):
        out[key]=bool(_value(source,key,default=False))
    for key in ('exchange_mode','bank_erosion_fraction'):
        out[key]=float(_value(source,key,default=0.))
    hiding=np.asarray(_value(source,'hiding_exposure',default=0))
    if hiding.size!=1 or hiding.dtype.kind=='b' or not np.isfinite(hiding).all() or not float(hiding.item()).is_integer():
        raise ValueError('Invalid hiding/exposure mode')
    out['hiding_exposure']=int(hiding.item())
    out['hiding_parker_alpha']=float(_value(source,'hiding_parker_alpha',default=.6))
    out['hiding_wu_exponent']=float(_value(source,'hiding_wu_exponent',default=-.6))
    out['mode']=np.ascontiguousarray(_value(source,'mode'),dtype=np.int32).reshape(-1)
    out['bed_boundary_mode']=np.ascontiguousarray(_value(source,'bed_boundary_mode'),dtype=np.int32).reshape(-1)
    out['feed']=np.ascontiguousarray(_value(source,'bc_bed_input','fraction_bed_input','feed'),dtype=np.float64)
    out['bc_hydraulic']=np.ascontiguousarray(_value(source,'bc_hydraulic'),dtype=np.float64)
    # Match the native deck's mutually exclusive equilibrium/prescribed inlet
    # contract even for callers that supply a prepared snapshot directly.
    equilibrium=np.flatnonzero(out['bed_boundary_mode']==1)
    if len(equilibrium):
        for b in equilibrium:
            if b>=len(out['mode']) or out['mode'][b] not in (1,6,7,8,9):
                raise ValueError('Equilibrium bedload requires an inlet boundary')
            if np.any(out['feed'][:,b]!=0):
                raise ValueError('Equilibrium bedload conflicts with prescribed class feed')
            for key in ('fraction_bed_input','bc_bed_input','fraction_series_bed_input',
                        'series_bed_input'):
                if key in source and np.any(np.asarray(source[key])[...,b]!=0):
                    raise ValueError('Equilibrium bedload conflicts with prescribed input: '+key)
            if ('forcing' in source and np.asarray(source['forcing']).ndim==2 and
                    np.asarray(source['forcing']).shape[0]>=4 and
                    np.asarray(source['forcing'])[3,b]!=0):
                raise ValueError('Equilibrium bedload conflicts with prescribed total input')
            if 'nseries' in source and np.asarray(source['nseries'])[b]!=0:
                raise ValueError('Equilibrium bedload conflicts with sediment time series')
    for key in ('nseries','nhydro_series','nfraction_series'):
        if allow_scheduled and key in ('nhydro_series','nfraction_series'):continue
        if key in source and np.any(np.asarray(source[key])!=0):raise ValueError('Time-dependent '+key+' is not ported')
    validate_config(out,external_suspension=external_suspension,allow_bedform=allow_bedform,
                    allow_general_avalanche=allow_general_avalanche,
                    allow_physical_toe_mode2=allow_physical_toe_mode2,
                    allow_distributed_bank_mode3=allow_distributed_bank_mode3,
                    allow_adaptation=allow_adaptation)
    return out


def validate_config(c,*,external_suspension=False,allow_bedform=False,
                    allow_general_avalanche=False,allow_physical_toe_mode2=False,
                    allow_distributed_bank_mode3=False,allow_adaptation=False):
    ng=len(c['d50']);nb=len(c['mode'])
    if ng<1 or any(c[k].shape!=(ng,) for k in ('d90','rhos','bulk','min_depth')):
        raise ValueError('Invalid per-fraction arrays')
    if any(not np.isfinite(c[k]).all() for k in ('d50','d90','rhos','bulk','min_depth','feed','bc_hydraulic')):
        raise ValueError('Nonfinite sediment parameters')
    if any(np.any(c[k]<=0) for k in ('d50','d90','bulk')) or np.any(c['rhos']<=c['rho']) or np.any(c['min_depth']<0):
        raise ValueError('Invalid grain/volume parameters')
    if ng>32:raise ValueError('Native mixtures support at most 32 grains')
    if any(not np.isfinite(c[k]) for k in ('rho','grav','dry','critical','cf','alfabs','alfabn','angle','target','dx','dy')):
        raise ValueError('Nonfinite controls')
    if min(c[k] for k in ('rho','grav','target','dx','dy','morfac'))<=0 or min(c[k] for k in ('dry','critical','cf','alfabs','alfabn'))<0 or not 0<c['angle']<90:
        raise ValueError('Invalid physical controls')
    if c['scheme']!=6 or c['bedlaw'] not in (0,2):
        raise ValueError('Only native scheme-6 no-bedload or MPM transport is ported')
    if c['bedlaw']==2:
        if c['mpm_friction_mode'] not in (0,1) or c['bed_slope'] not in (0,2):
            raise ValueError('Only native MPM friction modes 0/1 with slope 0/2 are ported')
    elif (c['mpm_friction_mode']!=0 or c['bed_slope']!=0 or c['hiding_exposure']!=0 or
          c['secondary_flow_model']!=0 or c['secondary_direction'] or
          np.any(c['feed']) or np.any(c['bed_boundary_mode'])):
        raise ValueError('No-bedload mode requires zero bed feed and inactive MPM, slope, hiding and secondary controls')
    if c['bedlaw']==0 and c['exchange_mode'] in (1.,2.) and c['morfac']!=1.:
        raise ValueError('No-bedload suspended exchange requires MorFac 1')
    if c['secondary_flow_model'] not in (0,1,2):
        raise ValueError('Unsupported secondary-flow model')
    if c['active_layer_mode']!=0 or not 1<=c['nsubstrate_layers']<=32:
        raise ValueError('Fixed active target and 1-32 ordered substrate layers required')
    if c['nsubstrate_layers']>1 and ('substrate_thickness' not in c or
            not np.isfinite(c['substrate_thickness']) or c['substrate_thickness']<=0):
        raise ValueError('Positive finite substrate-layer thickness required for multilayer transport')
    if c['bedform_roughness'] and not allow_bedform:
        raise ValueError('Bedform roughness requires the explicit coupled GPU variant')
    if c['general_avalanche'] and not allow_general_avalanche:
        raise ValueError('General avalanche requires the explicit coupled GPU variant')
    if (any(c[k] for k in ('amr_enabled','bank_avalanche',
                           'bank_erosion_fraction')) or
            (c['bedload_adaptation'] and not allow_adaptation) or
            (c['bank_failure'] and not (allow_physical_toe_mode2 or allow_distributed_bank_mode3))):
        raise ValueError('Unsupported AMR, adaptation, roughness or bank physics')
    if external_suspension:
        if c['exchange_mode'] not in (0.,1.,2.):
            raise ValueError('Coupled suspension supports exchange modes 0, 1 and 2')
    elif c['exchange_mode']:
        raise ValueError('Exchange requires a coupled suspension operator')
    if c['hiding_exposure'] not in (0,2,3,4,5) or not np.isfinite(c.get('hiding_parker_alpha',.6)) or not np.isfinite(c.get('hiding_wu_exponent',-.6)):
        raise ValueError('Unsupported hiding/exposure controls')
    if c['bed_boundary_mode'].shape!=(nb,) or c['feed'].shape!=(ng,nb) or c['bc_hydraulic'].shape!=(2,nb):
        raise ValueError('Invalid sediment boundary shapes')
    allowed_modes=[2,3,4,6,8,9]+([1] if allow_bedform else [])
    if not np.isin(c['mode'],allowed_modes).all() or not np.isin(c['bed_boundary_mode'],[0,1]).all() or np.any(c['feed']<0):
        raise ValueError('Unsupported sediment/hydraulic boundary mode')


def validate_topology(data,nc,nf,nb):
    if min(nc,nf)<1:raise ValueError('Nonempty transport grid required')
    names=('owner','neighbour','bid','offset','faces','signs')
    arrays={k:np.asarray(data[k]) for k in names}
    for k,v in arrays.items():
        if v.ndim!=1 or v.dtype.kind not in 'iu' or np.any(v>np.iinfo(np.int32).max) or np.any(v<np.iinfo(np.int32).min):
            raise ValueError('Invalid integer topology '+k)
    owner,neighbour,bid,offset,faces,signs=(arrays[k] for k in names)
    if np.any((owner<0)|(owner>=nc)) or np.any((neighbour< -1)|(neighbour>=nc)) or np.any(owner==neighbour):
        raise ValueError('Invalid owner/neighbour')
    if np.any((bid< -1)|(bid>=nb)) or np.any((bid>=0)&(neighbour>=0)):raise ValueError('Invalid boundary topology')
    if offset.shape!=(nc+1,) or offset[0]!=0 or offset[-1]!=len(faces) or np.any(np.diff(offset.astype(np.int64))<0):
        raise ValueError('Invalid adjacency offsets')
    if signs.shape!=faces.shape or np.any((faces<0)|(faces>=nf)) or not np.isin(signs,[-1,1]).all():
        raise ValueError('Invalid adjacency incidence')
    if not np.array_equal(np.bincount(faces[signs>0],minlength=nf),np.ones(nf,int)) or not np.array_equal(
            np.bincount(faces[signs<0],minlength=nf),(neighbour>=0).astype(int)):
        raise ValueError('Incomplete or duplicate incidence')
    for cell in range(nc):
        ff=faces[offset[cell]:offset[cell+1]];ss=signs[offset[cell]:offset[cell+1]]
        if np.any(np.where(ss>0,owner[ff],neighbour[ff])!=cell):raise ValueError('Incidence belongs to another cell')


class ResidentSedimentTransport:
    """Persistent SoA buffers; all stage work remains on the selected CUDA stream.

    Set ``rotate_capacity`` to SecondaryTransport.rotate_capacity for enabled
    secondary transport steering. Callback modifies capacity(2,ng,nc) in-place
    using current hI, and returns its device status. Reconstruction follows
    rotation exactly as in native.
    Layers layout is (1+nsubstrate,ng,nc), active first. Inputs are not
    mutated by fluxes(); rebalance() intentionally mutates supplied layers.
    """
    def __init__(self,data,sediment_config=None,*,source_path=None,augmented_source=None,
                 stratigraphy_source=None,external_suspension=False,allow_scheduled=False,
                 allow_bedform=False,allow_general_avalanche=False,
                 allow_physical_toe_mode2=False,
                 allow_distributed_bank_mode3=False,allow_adaptation=False):
        import cupy as cp
        self.cp=cp;self.config=configuration(data,sediment_config,
            external_suspension=external_suspension,allow_scheduled=allow_scheduled,
            allow_bedform=allow_bedform,allow_general_avalanche=allow_general_avalanche,
            allow_physical_toe_mode2=allow_physical_toe_mode2,
            allow_distributed_bank_mode3=allow_distributed_bank_mode3,
            allow_adaptation=allow_adaptation);c=self.config
        self.nc=int(np.asarray(data['xy']).shape[1]);self.nf=len(data['owner']);self.ng=len(c['d50']);self.nb=len(c['mode'])
        self.nl=1+c['nsubstrate_layers']
        self.coupled_mpm=(c['bedlaw']==2 and not c['bedload_adaptation'] and
                          c['secondary_flow_model']==0 and not c['bedform_roughness'])
        self.na=len(data['faces']);self.rotate_capacity=None
        self.adaptation=None;self.adaptation_result=None
        nc,nf,na=self.nc,self.nf,self.na
        validate_topology(data,nc,nf,self.nb)
        shapes={'xy':(2,nc),'mid':(2,nf),'normal':(2,nf),'rectangle_width':(2,nc),'bed_ls_matrix':(4,nc),
                'bed_ls_delta':(2,na),'bed_ls_weight':(na,),'face_displacement':(2,2,nf),
                'manning_cell':(nc,),'owner':(nf,),'neighbour':(nf,),'bid':(nf,),
                'offset':(nc+1,),'faces':(na,),'signs':(na,), 'weno_cells':(5,2,nf),
                'open_boundary_distance':(nc,),'uncoupled_boundary_distance':(nc,)}
        integer={'owner','neighbour','bid','offset','faces','signs','weno_cells',
                 'open_boundary_distance','uncoupled_boundary_distance'}
        self.geometry={}
        for name,shape in shapes.items():
            value=np.asarray(data[name])
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Invalid sediment geometry '+name)
            if name in integer and (value.dtype.kind not in 'iu' or np.any(value>np.iinfo(np.int32).max)):
                raise ValueError('Integer topology required: '+name)
            self.geometry[name]=cp.asarray(np.ascontiguousarray(value,dtype=np.int32 if name in integer else np.float64))
        for name in ('d50','d90','rhos','bulk','min_depth','mode','bed_boundary_mode','feed','bc_hydraulic'):
            self.geometry[name]=cp.asarray(c[name])
        manning=np.asarray(data['manning_cell'])
        if np.any(manning<0) or (c['bedlaw']==2 and c['mpm_friction_mode']==1 and np.any(manning<=0)):
            raise ValueError('Nonnegative Manning required; roughness MPM requires positive effective roughness')
        w=np.asarray(data['weno_cells']);present=w[0]>=0
        if np.any(w[:,present]<0) or np.any(w>=nc):raise ValueError('Invalid WENO capacity stencil')
        source=Path(augmented_source or HERE/'augmented_fwave.cu').read_text()+'\n'+Path(source_path or HERE/'sediment_transport.cu').read_text()
        names=('st_capacity','st_capacity_graded','st_gradients','st_faces','st_coupled_stage','st_coupled_faces',
               'st_rebalance','st_bed')
        self.module=cp.RawModule(code=source,options=OPTIONS,name_expressions=names)
        self.k={name:self.module.get_function(name) for name in names}
        self.buffers={'capacity':cp.empty((2,self.ng,nc)), 'gradient':cp.empty((2,2,self.ng,nc)),
                     'downslope':cp.empty((2,nc)), 'availability':cp.empty((self.ng,nc)),
                     'mixture_d90':cp.empty(nc),'hiding':cp.ones((self.ng,nc)),
                     'raw':cp.empty((self.ng,nf)),
                     'face_left':cp.empty((2,self.ng,nf)), 'face_right':cp.empty((2,self.ng,nf)),
                     'cell_status':cp.empty(nc,cp.int32),'face_status':cp.empty(nf,cp.int32),
                     'coupled_stage_status':cp.zeros(nf,cp.int32),
                     'coupled_stage_applied':cp.zeros(nf,cp.int32),
                     'coupled_face_status':cp.zeros(nf,cp.int32),
                     'unrotated_status':cp.zeros(nc,cp.int32),
                     'rebalance_status':cp.empty(nc,cp.int32),'bed':cp.empty(nc),
                     'adaptation_status':cp.zeros((),cp.int32),
                     'cfl':cp.zeros((),cp.float64)}
        self.stratigraphy=None
        if c['nsubstrate_layers']>1:
            from .sediment_stratigraphy import ResidentMultilayerStratigraphy
            # A frozen transport CUDA source must use the stratigraphy CUDA
            # source from that same frozen source directory, not live HEAD.
            if stratigraphy_source is None and source_path is not None:
                stratigraphy_source=Path(source_path).with_name('sediment_stratigraphy.cu')
            self.stratigraphy=ResidentMultilayerStratigraphy(
                nc,c['bulk'],c['target'],c['substrate_thickness'],c['nsubstrate_layers'],
                source_path=stratigraphy_source)
            self.buffers['rebalance_status']=self.stratigraphy.status
            self.buffers['bed_status']=self.stratigraphy.bed_status
        self.graded_capacity=bool(np.any(c['d50']!=c['d50'][0]) or
                                  np.any(c['min_depth']!=c['min_depth'][0]) or c['hiding_exposure'])

    def _array(self,a,shape,dtype=None):
        cp=self.cp
        if not isinstance(a,cp.ndarray) or a.shape!=shape or a.dtype!=(dtype or cp.float64) or not a.flags.c_contiguous:
            raise ValueError('Expected contiguous device array '+str(shape))

    def _launch(self,name,n,args):self.k[name](((n+127)//128,),(128,),args)

    def prepare(self,q,bed,layers,secondary_mass,active):
        """Resident nonlinear capacity and gradients before the model-0 stage BC."""
        cp=self.cp;c=self.config;a=self.geometry;b=self.buffers;nc,nf,ng,nb=map(np.int32,(self.nc,self.nf,self.ng,self.nb))
        if c['bedlaw']!=2:raise ValueError('No-bedload mode has no MPM capacity to prepare')
        self._array(q,(3,self.nc));self._array(bed,(self.nc,));self._array(layers,(self.nl,self.ng,self.nc));self._array(secondary_mass,(self.nc,))
        self._array(active,(self.nc,),cp.int32)
        capacity_args=(nc,ng,q,layers,a['manning_cell'],a['d50'],a['d90'],a['rhos'],a['bulk'],a['min_depth'],
                       active,*[np.float64(c[k]) for k in ('rho','grav','dry','critical','target','cf')],
                       np.int32(c['mpm_friction_mode']))
        if self.graded_capacity:
            self._launch('st_capacity_graded',self.nc,(*capacity_args,np.int32(c['hiding_exposure']),
                         np.float64(c['hiding_parker_alpha']),np.float64(c['hiding_wu_exponent']),
                         b['capacity'],b['availability'],b['mixture_d90'],b['hiding'],b['cell_status']))
        else:
            self._launch('st_capacity',self.nc,(*capacity_args,b['capacity'],b['availability'],b['mixture_d90'],b['cell_status']))
        steer=c['secondary_flow_model']==1 or (c['secondary_flow_model']==2 and c['secondary_direction'])
        if steer:
            if not callable(self.rotate_capacity):raise ValueError('Current secondary-direction callback required')
            secondary_status=self.rotate_capacity(q,secondary_mass,b['capacity'])
        else:secondary_status=b['unrotated_status']
        if c['bedload_adaptation']:
            if self.adaptation is None:
                raise ValueError('Enabled bed-load adaptation requires an attached resident operator')
            self.adaptation_result=self.adaptation.relax(q,b['capacity'],a['feed'])
            if not isinstance(self.adaptation_result,dict) or not all(
                    key in self.adaptation_result for key in ('iterations','residual','tolerance')):
                raise ValueError('Bed-load adaptation returned incomplete convergence diagnostics')
            b['adaptation_status'].fill(0)
        self._launch('st_gradients',self.nc,(nc,nf,np.int32(self.na),ng,b['capacity'],bed,a['bed_ls_matrix'],a['bed_ls_delta'],a['bed_ls_weight'],
                     a['xy'],a['mid'],a['owner'],a['neighbour'],a['offset'],a['faces'],a['signs'],active,b['gradient'],b['downslope']))
        return secondary_status

    def stage_characteristic(self,q,bed,face_value,trace):
        """Native coupled SWE/Exner mode-0 stage outlet characteristic."""
        if not self.coupled_mpm:raise ValueError('Coupled stage characteristic requires native model-0 MPM trigger')
        cp=self.cp;c=self.config;a=self.geometry;b=self.buffers
        self._array(q,(3,self.nc));self._array(bed,(self.nc,))
        self._array(face_value,(4,2,self.nf));self._array(trace,(10,self.nf))
        self._launch('st_coupled_stage',self.nf,(np.int32(self.nc),np.int32(self.nf),np.int32(self.ng),np.int32(self.nb),
            q,face_value,bed,a['normal'],a['owner'],a['neighbour'],a['bid'],a['mode'],a['bc_hydraulic'],
            a['manning_cell'],a['d50'],a['rhos'],a['bulk'],a['min_depth'],b['mixture_d90'],b['hiding'],
            b['availability'],np.float64(c['rho']),np.float64(c['grav']),np.float64(c['critical']),
            np.float64(c['cf']),np.int32(c['mpm_friction_mode']),
            np.float64(c['dry']),np.float64(c['morfac']),trace,b['coupled_stage_status'],b['coupled_stage_applied']))
        return b['coupled_stage_status']

    def fluxes(self,q,bed,layers,secondary_mass,hydraulic,*,prepared=False):
        cp=self.cp;c=self.config;a=self.geometry;b=self.buffers;nc,nf,ng,nb=map(np.int32,(self.nc,self.nf,self.ng,self.nb))
        if c['bedlaw']!=2:raise ValueError('No-bedload mode has no horizontal bed flux')
        self._array(q,(3,self.nc));self._array(bed,(self.nc,));self._array(layers,(self.nl,self.ng,self.nc));self._array(secondary_mass,(self.nc,))
        for name,shape,dtype in [('trace',(10,self.nf),cp.float64),('raw_fl',(3,self.nf),cp.float64),
                                 ('closed',(self.nf,),cp.int32),('active',(self.nc,),cp.int32)]:
            self._array(hydraulic[name],shape,dtype)
        if prepared:
            if not self.coupled_mpm:raise ValueError('Precomputed capacity is only admitted for native coupled MPM')
            secondary_status=b['unrotated_status']
        else:secondary_status=self.prepare(q,bed,layers,secondary_mass,hydraulic['active'])
        self._launch('st_faces',self.nf,(nc,nf,ng,nb,b['capacity'],b['gradient'],bed,b['downslope'],b['availability'],a['xy'],a['mid'],
                     a['rectangle_width'],a['normal'],a['face_displacement'],a['weno_cells'],a['open_boundary_distance'],
                     a['uncoupled_boundary_distance'],a['owner'],a['neighbour'],a['bid'],a['offset'],a['faces'],a['signs'],hydraulic['active'],
                     a['mode'],a['bed_boundary_mode'],a['feed'],a['bc_hydraulic'],hydraulic['trace'],hydraulic['raw_fl'],hydraulic['closed'],
                     *[np.float64(c[k]) for k in ('dry','grav','dx','dy')],np.int32(c['bed_slope']),
                     *[np.float64(c[k]) for k in ('alfabs','alfabn','angle')],np.int32(self.coupled_mpm),q,
                     a['manning_cell'],a['d50'],b['mixture_d90'],a['rhos'],b['hiding'],a['min_depth'],
                     np.float64(c['rho']),np.float64(c['critical']),np.float64(c['cf']),
                     np.int32(c['mpm_friction_mode']),b['face_left'],b['face_right'],b['raw'],b['face_status']))
        status=(b['cell_status'],b['face_status'],secondary_status)
        if c['bedload_adaptation']:status+=(b['adaptation_status'],)
        if self.stratigraphy is not None:status+= (b['bed_status'],)
        return {'raw':b['raw'],'cfl':b['cfl'],'status':status}

    def coupled_faces(self,hydraulic,flux):
        """Insert native model-0 coupled SWE/Exner face waves before draining."""
        if not self.coupled_mpm:raise ValueError('Coupled face waves require native model-0 MPM trigger')
        c=self.config;a=self.geometry;b=self.buffers
        for name,shape,dtype in [('local',(8,self.nf),self.cp.float64),
                                 ('riemann',(18,self.nf),self.cp.float64),
                                 ('closed',(self.nf,),self.cp.int32)]:
            self._array(hydraulic[name],shape,dtype)
        for name,shape in [('fl',(3,self.nf)),('fr',(3,self.nf)),('speed',(self.nf,))]:
            self._array(flux[name],shape)
        self._launch('st_coupled_faces',self.nf,(np.int32(self.nc),np.int32(self.nf),np.int32(self.ng),
            hydraulic['local'],hydraulic['riemann'],hydraulic['kind'],hydraulic['closed'],
            a['owner'],a['neighbour'],a['uncoupled_boundary_distance'],a['normal'],b['availability'],
            b['face_left'],b['face_right'],a['manning_cell'],a['d50'],a['rhos'],a['bulk'],
            a['min_depth'],b['mixture_d90'],b['hiding'],np.float64(c['rho']),np.float64(c['grav']),
            np.float64(c['critical']),np.float64(c['cf']),np.int32(c['mpm_friction_mode']),
            np.float64(c['dry']),np.float64(c['morfac']),
            flux['fl'],flux['fr'],flux['speed'],b['raw'],b['coupled_face_status']))
        return b['coupled_face_status']

    def rebalance(self,layers):
        self._array(layers,(self.nl,self.ng,self.nc))
        if self.stratigraphy is not None:return self.stratigraphy.rebalance(layers)
        self._launch('st_rebalance',self.nc,(np.int32(self.nc),np.int32(self.ng),layers,self.geometry['bulk'],
                     np.float64(self.config['target']),self.buffers['rebalance_status']))
        return self.buffers['rebalance_status']

    def bed_elevation(self,layers,rock,out=None):
        self._array(layers,(self.nl,self.ng,self.nc));self._array(rock,(self.nc,))
        out=self.buffers['bed'] if out is None else out;self._array(out,(self.nc,))
        if self.stratigraphy is not None:return self.stratigraphy.bed_elevation(layers,rock,out)['bed']
        self._launch('st_bed',self.nc,(np.int32(self.nc),np.int32(self.ng),layers,rock,self.geometry['bulk'],out))
        return out
