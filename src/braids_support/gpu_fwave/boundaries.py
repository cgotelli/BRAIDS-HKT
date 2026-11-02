"""GPU constant nonsampled stage/total-Q/depth controls; strict scope guard.

Hydrographs, surveyed sections, critical/weir controls, sediment characteristics
and viscosity-equilibrated inlets are deliberately rejected, not approximated.
"""
from pathlib import Path
import numpy as np
from .resident_update import OPTIONS


def validate_scope(data, *, allow_scheduled=False, allow_bedform_mode1=False):
    modes=np.asarray(data['mode'])
    supported=[2,3,6,8]+([1] if allow_bedform_mode1 else [])
    if not np.isin(modes,supported).all():
        raise ValueError('GPU boundary port supports only modes 2, 3, 6 and 8; mode 1 requires the bedform variant')
    if allow_bedform_mode1:
        # The stock mode-1 root below implements physical owner-bed faces.
        # Surveyed subcell section spans/offsets are a separate GPU port.
        if not np.any(modes==1):
            raise ValueError('Explicit bedform boundary variant requires a mode-1 total-discharge inlet')
        if np.any(np.asarray(data.get('sampled',0))) or np.any(np.asarray(data.get('bedoffset',0))):
            raise ValueError('Bedform mode-1 inlet requires unsampled zero-offset faces')
    if allow_scheduled:
        # This is validation input only, not a substituted numerical deck.
        data=dict(data);data.pop('nhydro_series',None)
    for key in ('nhydro_series','section_control','sampled'):
        if key in data and np.any(data[key]):
            raise ValueError('GPU boundary port excludes '+key)
    if 'reversal_policy' in data and np.any(np.asarray(data['reversal_policy'])!=1):
        raise ValueError('GPU boundary port requires native no-reversal policy')
    if data.get('horizontal_viscosity',0)!=0:
        raise ValueError('GPU boundary port excludes viscosity-equilibrated inlets')
    forcing=np.asarray(data['forcing'])
    if forcing.shape!=(2,len(modes)) or not np.isfinite(forcing).all():
        raise ValueError('Invalid constant boundary forcing')
    if np.any(forcing[0,np.isin(modes,[1,6,8])]<0):raise ValueError('Negative prescribed inlet Q')
    nf=len(data['owner']);nb=len(modes)
    bid=np.asarray(data['bid']);offset=np.asarray(data['boundary_offset']);faces=np.asarray(data['boundary_faces'])
    if any(v.dtype.kind not in 'iu' for v in (modes,bid,offset,faces)):
        raise ValueError('Boundary topology must use integer indices')
    if bid.shape!=(nf,) or np.any((bid< -1)|(bid>=nb)):
        raise ValueError('Invalid boundary ID')
    if offset.shape!=(nb+1,) or offset[0]!=0 or offset[-1]!=len(faces) or np.any(np.diff(offset)<0):
        raise ValueError('Invalid boundary CSR offsets')
    if faces.ndim!=1 or np.any((faces<0)|(faces>=nf)) or len(np.unique(faces))!=len(faces):
        raise ValueError('Invalid boundary CSR faces')
    if not np.array_equal(np.sort(faces),np.flatnonzero(bid>=0)):
        raise ValueError('Boundary CSR must cover every open face once')
    for b in range(nb):
        selected=faces[offset[b]:offset[b+1]]
        if np.any(bid[selected]!=b) or np.any(np.asarray(data['neighbour'])[selected]>=0):
            raise ValueError('Boundary faces must be exterior faces of their own boundary')


def validate_gaia_fixed_inlet(data):
    """Admit only the exact native law-9 imposed-face-bed envelope.

    TELEMAC-GAIA LIEBOR=KENT fixes the inlet boundary bed while its CONLIT
    setting independently selects either free/equilibrium capacity (public bed
    boundary mode 1) or a prescribed per-fraction feed (mode 2).  Both modes
    use the same immutable hydraulic face elevations.  The GPU receives those
    elevations explicitly; it must never infer them from a subsequently mobile
    owner cell.
    """
    law=int(np.asarray(data.get('bedlaw',0)).item())
    mode=np.asarray(data['mode']);nb=len(mode);nf=len(data['owner'])
    if law==9 and not {'fixed_boundary_bed_mode','fixed_boundary_bed'}<=set(data):
        raise ValueError('Law 9 requires explicit fixed boundary-bed arrays from the native exporter')
    fixed_mode=np.asarray(data.get('fixed_boundary_bed_mode',np.zeros(nb,dtype=np.int32)))
    fixed_bed=np.asarray(data.get('fixed_boundary_bed',np.zeros(nf,dtype=np.float64)))
    if law!=9:
        if np.any(fixed_mode) or np.any(fixed_bed):
            raise ValueError('GAIA fixed boundary bed is inert outside law 9')
        return False
    bed_mode=np.asarray(data.get('bed_boundary_mode'))
    if (fixed_mode.shape!=(nb,) or fixed_mode.dtype.kind not in 'iu' or
            not np.isin(fixed_mode,(0,1)).all() or bed_mode.shape!=(nb,) or
            bed_mode.dtype.kind not in 'iu' or
            not np.isin(bed_mode,(0,1,2)).all()):
        raise ValueError('Law 9 requires explicit fixed boundary-bed modes')
    expected=np.isin(bed_mode,(1,2)).astype(fixed_mode.dtype)
    if not np.array_equal(fixed_mode,expected):
        raise ValueError('Law 9 fixed-face boundary modes must use fixed face beds')
    if np.any(mode[fixed_mode==1]!=6):
        raise ValueError('GAIA fixed face bed requires a mode-6 total-discharge inlet')
    if (fixed_bed.shape!=(nf,) or fixed_bed.dtype.kind!='f' or
            not np.isfinite(fixed_bed).all()):
        raise ValueError('Invalid GAIA fixed boundary-bed elevations')
    bid=np.asarray(data['bid']);owner=np.asarray(data['owner'])
    face_fixed=np.zeros(nf,dtype=bool);exterior=bid>=0
    face_fixed[exterior]=fixed_mode[bid[exterior]]==1
    offset=np.asarray(data.get('bedoffset',data.get('boundary_bed_offset')))
    bed=np.asarray(data['bed'])
    if offset.shape!=(nf,) or bed.shape!=(len(data['q'][0]),):
        raise ValueError('GAIA fixed face-bed geometry is incomplete')
    if (not np.array_equal(fixed_bed[face_fixed],bed[owner[face_fixed]]+offset[face_fixed]) or
            np.any(fixed_bed[~face_fixed]!=0.) or
            np.any(np.signbit(fixed_bed[~face_fixed]))):
        raise ValueError('GAIA fixed face bed differs from the native initial geometry')
    return bool(np.any(fixed_mode))


class BoundaryControls:
    def __init__(self,data,inputs,source=None,*,allow_scheduled=False,allow_bedform_mode1=False):
        import cupy as cp
        validate_scope(data,allow_scheduled=allow_scheduled,
            allow_bedform_mode1=allow_bedform_mode1)
        self.gaia_fixed=validate_gaia_fixed_inlet(data)
        self.cp=cp;self.nc=data['q'].shape[1];self.nf=len(data['owner']);self.nb=len(data['mode'])
        self.inputs=inputs
        self.arrays={k:cp.asarray(data[k]) for k in
            ('boundary_offset','boundary_faces','bid','mode','forcing','manning')}
        for key in ('boundary_offset','boundary_faces','bid','mode'):
            self.arrays[key]=self.arrays[key].astype(cp.int32,copy=False)
        from braids_core.windows_paths import filesystem_path
        source=filesystem_path(source) if source else filesystem_path(__file__).with_suffix('.cu')
        names=('gpu_section_traces','gpu_stage_traces','gpu_boundary_ledger')+(
            ('gpu_mode1_traces',) if allow_bedform_mode1 else ())
        self.module=cp.RawModule(code=source.read_text(),options=OPTIONS,name_expressions=names)
        self.kernels={name:self.module.get_function(name) for name in names}
        self.typed_section_kernel=None
        binding=data.get('delft_forcing_binding')
        if isinstance(binding,dict) and binding.get('hydraulic_roughness_mode')!='manning':
            roughness_source=source.with_name('hydraulic_roughness.cu')
            typed_source=source.with_name('typed_boundary_roughness.cu')
            if (not roughness_source.is_file() or roughness_source.is_symlink() or
                    not typed_source.is_file() or typed_source.is_symlink()):
                raise ValueError('Archived typed boundary roughness sources differ')
            code=(roughness_source.read_text(encoding='utf-8')+'\n'+
                  typed_source.read_text(encoding='utf-8'))
            module=cp.RawModule(code=code,options=OPTIONS,
                                name_expressions=('typed_section_traces',))
            self.typed_section_kernel=module.get_function('typed_section_traces')
            modes={'chezy':1,'manning':2,'white_colebrook':3}
            try:self.roughness_mode=np.int32(modes[binding['hydraulic_roughness_mode']])
            except KeyError as error:raise ValueError('Typed boundary roughness mode differs') from error
            self.roughness_waqua=np.int32(bool(binding['white_colebrook_chzk2d']))
            self.roughness_work_depth=np.float64(binding['effective_minimum_work_depth_m'])
        self.fixed_kernels={}
        if self.gaia_fixed:
            fixed_source=source.with_name('gaia_fixed_inlet.cu')
            if not fixed_source.is_file():
                raise ValueError('Authenticated GAIA fixed-inlet CUDA source is missing')
            fixed_names=('gpu_gaia_fixed_inlet_traces','gpu_gaia_fixed_inlet_pressure')
            fixed_module=cp.RawModule(code=fixed_source.read_text(),options=OPTIONS,
                name_expressions=fixed_names)
            self.fixed_kernels={name:fixed_module.get_function(name) for name in fixed_names}
            self.arrays['fixed_boundary_bed_mode']=cp.asarray(
                data['fixed_boundary_bed_mode'],dtype=cp.int32)
            self.arrays['fixed_boundary_bed']=cp.asarray(
                data['fixed_boundary_bed'],dtype=cp.float64)
        self.status=cp.zeros(self.nb,dtype=cp.int32)
        self.face_status=cp.zeros(self.nf,dtype=cp.int32)
        self.ledger=cp.zeros(self.nb,dtype=cp.float64)
        self.dry=np.float64(data['dry']);self.grav=np.float64(data['grav'])
        self.speed=np.float64(data.get('speed_limit',50.))
        self.chezy=np.float64(data.get('manning_chezy_minimum_depth',.01))
        self.mode1=allow_bedform_mode1

    def apply(self,q,trace):
        if not self.nb:return
        a,b=self.inputs,self.arrays;nc,nf,nb=np.int64(self.nc),np.int64(self.nf),np.int32(self.nb)
        if self.typed_section_kernel is None:
            self.kernels['gpu_section_traces']((self.nb,),(128,),
                (q,a['bed'],b['manning'],a['normal'],a['length'],a['owner'],b['boundary_offset'],
                 b['boundary_faces'],b['mode'],b['forcing'],nc,nf,nb,self.dry,self.speed,self.chezy,trace,self.status))
        else:
            self.typed_section_kernel((self.nb,),(128,),
                (q,a['bed'],b['manning'],a['normal'],a['length'],a['owner'],b['boundary_offset'],
                 b['boundary_faces'],b['mode'],b['forcing'],nc,nf,nb,self.dry,self.speed,
                 self.roughness_mode,self.grav,self.roughness_work_depth,
                 self.roughness_waqua,trace,self.status))
        self.kernels['gpu_stage_traces'](((self.nf+127)//128,),(128,),
            (q,a['bed'],a['normal'],a['owner'],b['bid'],b['mode'],b['forcing'],nc,nf,nb,
             self.grav,self.dry,trace,self.face_status))
        if self.gaia_fixed:
            self.fixed_kernels['gpu_gaia_fixed_inlet_traces']((self.nb,),(128,),
                (q,a['bed'],b['manning'],a['normal'],a['length'],a['owner'],
                 b['boundary_offset'],b['boundary_faces'],b['mode'],b['forcing'],
                 b['fixed_boundary_bed_mode'],b['fixed_boundary_bed'],nc,nf,nb,
                 self.dry,self.speed,self.chezy,trace,self.status,self.face_status))
        if self.mode1:
            self.kernels['gpu_mode1_traces']((self.nb,),(128,),
                (q,a['bed'],a['normal'],a['length'],a['owner'],b['boundary_offset'],
                 b['boundary_faces'],b['mode'],b['forcing'],nc,nf,nb,
                 self.grav,self.dry,self.speed,trace,self.status,self.face_status))

    def apply_pressure(self,q,bed,trace,fl,fr,face_status):
        """Add only the dynamic cell-to-fixed-face hydrostatic correction."""
        if not self.gaia_fixed:return
        a,b=self.inputs,self.arrays
        self.fixed_kernels['gpu_gaia_fixed_inlet_pressure'](((self.nf+127)//128,),(128,),
            (q,bed,trace,a['normal'],a['owner'],b['bid'],
             b['fixed_boundary_bed_mode'],b['fixed_boundary_bed'],np.int64(self.nc),
             np.int64(self.nf),np.int32(self.nb),self.grav,self.dry,fl,fr,face_status))

    def integrate_ledger(self,flux,dt):
        if self.nb:
            a,b=self.inputs,self.arrays
            self.kernels['gpu_boundary_ledger']((self.nb,),(128,),
                (flux,a['length'],b['boundary_offset'],b['boundary_faces'],np.int32(self.nb),np.float64(dt),self.ledger))
        return self.ledger
