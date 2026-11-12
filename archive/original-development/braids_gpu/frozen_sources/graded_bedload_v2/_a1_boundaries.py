"""GPU constant nonsampled stage/total-Q/depth controls; strict scope guard.

Hydrographs, surveyed sections, critical/weir controls, sediment characteristics
and viscosity-equilibrated inlets are deliberately rejected, not approximated.
"""
from pathlib import Path
import numpy as np
from .normal_outlet_boundary import validate_boundaries
from .resident_update import OPTIONS


def validate_scope(data, *, allow_scheduled=False, allow_bedform_mode1=False):
    modes=np.asarray(data['mode'])
    supported=[2,3,4,6,8,9]+([1] if allow_bedform_mode1 else [])
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
    validate_boundaries(data)
    for key in ('nhydro_series',):
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


class BoundaryControls:
    def __init__(self,data,inputs,source=None,*,allow_scheduled=True,allow_bedform_mode1=False):
        import cupy as cp
        validate_scope(data,allow_scheduled=allow_scheduled,
            allow_bedform_mode1=allow_bedform_mode1)
        self.cp=cp;self.nc=data['q'].shape[1];self.nf=len(data['owner']);self.nb=len(data['mode'])
        self.inputs=inputs
        self.arrays={k:cp.asarray(data[k]) for k in
            ('boundary_offset','boundary_faces','bid','mode','forcing','manning')}
        for key in ('boundary_offset','boundary_faces','bid','mode'):
            self.arrays[key]=self.arrays[key].astype(cp.int32,copy=False)
        source=Path(source) if source else Path(__file__).with_suffix('.cu')
        names=('gpu_section_traces','gpu_stage_traces','gpu_boundary_ledger')+(
            ('gpu_mode1_traces',) if allow_bedform_mode1 else ())
        self.module=cp.RawModule(code=source.read_text(),options=OPTIONS,name_expressions=names)
        self.kernels={name:self.module.get_function(name) for name in names}
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
        self.kernels['gpu_section_traces']((self.nb,),(128,),
            (q,a['bed'],b['manning'],a['normal'],a['length'],a['owner'],b['boundary_offset'],
             b['boundary_faces'],b['mode'],b['forcing'],nc,nf,nb,self.dry,self.speed,self.chezy,trace,self.status))
        self.kernels['gpu_stage_traces'](((self.nf+127)//128,),(128,),
            (q,a['bed'],a['normal'],a['owner'],b['bid'],b['mode'],b['forcing'],nc,nf,nb,
             self.grav,self.dry,trace,self.face_status))
        if self.mode1:
            self.kernels['gpu_mode1_traces']((self.nb,),(128,),
                (q,a['bed'],a['normal'],a['length'],a['owner'],b['boundary_offset'],
                 b['boundary_faces'],b['mode'],b['forcing'],nc,nf,nb,
                 self.grav,self.dry,self.speed,trace,self.status,self.face_status))

    def integrate_ledger(self,flux,dt):
        if self.nb:
            a,b=self.inputs,self.arrays
            self.kernels['gpu_boundary_ledger']((self.nb,),(128,),
                (flux,a['length'],b['boundary_offset'],b['boundary_faces'],np.int32(self.nb),np.float64(dt),self.ledger))
        return self.ledger
