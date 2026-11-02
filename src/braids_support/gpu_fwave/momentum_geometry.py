"""Isolated fully GPU v35 geometry rebuild after wet/perched mask changes.

The resident diffusion wrapper retains the native-cache fast path and lazily
uses this module after a mask change. No CPU numerical fallback or automatic
CPU/GPU backend selection exists. Complete coupled wet/dry behavior requires
separate qualification beyond these native-equivalent diffusion operators.
"""
from pathlib import Path
import numpy as np
from .resident_update import OPTIONS


class ResidentDiffusionGeometry:
    def __init__(self,data,source_path=None):
        import cupy as cp
        from .momentum_diffusion import validate_scope
        validate_scope(data);self.cp=cp;self.nc=len(data['area']);self.nf=len(data['owner'])
        self.dry=float(data['dry']);self.nu=float(data['horizontal_viscosity']);self.ready=False;self.rebuilds=0
        self.a={name:cp.asarray(data[name]) for name in ('area','xy','mid','normal','length','owner','neighbour','offset','faces','signs')}
        nc,nf=self.nc,self.nf
        shapes=dict(conductance=(nf,),face_diffusivity=(nf,),tangent_offset=(nf,),normal_asymmetry=(nf,),face_bound=(nf,),
            rate=(nc,),recovery_inverse=(5,5,nc),recovery_affine_inverse=(2,2,nc))
        self.cache={name:cp.empty(shape,dtype=cp.float64) for name,shape in shapes.items()}
        for name,shape in dict(corrected_face=(nf,),recovery_slot=(nc,),recovery_order=(nc,),recovery_valid=(nc,),
            recovery_stencils=(256,nc),recovery_stencil_counts=(nc,)).items():self.cache[name]=cp.empty(shape,dtype=cp.int32)
        self.cache['recovery_cells']=cp.arange(nc,dtype=cp.int32)
        self.boundary=cp.empty(nc,dtype=cp.int32);self.face_status=cp.empty(nf,dtype=cp.int32)
        self.topology=cp.empty(nf,dtype=cp.bool_);self.maximum_rate=cp.empty(())
        names=('mg_faces','mg_boundary','mg_corrected','mg_slots','mg_recovery','mg_bound','mg_rate')
        self.source_path=Path(source_path or Path(__file__).with_suffix('.cu'))
        self.module=cp.RawModule(code=self.source_path.read_text(),options=OPTIONS,name_expressions=names)
        self.k={name:self.module.get_function(name) for name in names}

    def launch(self,name,count,args):
        if count:self.k[name](((count+127)//128,),(128,),args)

    def update(self,q,bed):
        cp=self.cp;a=self.a;b=self.cache;nc,nf=map(np.int64,(self.nc,self.nf));dry=np.float64(self.dry)
        if not isinstance(q,cp.ndarray) or q.shape!=(3,self.nc) or q.dtype!=cp.float64 or not q.flags.c_contiguous:raise ValueError('Resident hydraulic FP64 state required')
        if not isinstance(bed,cp.ndarray) or bed.shape!=(self.nc,) or bed.dtype!=cp.float64 or not bed.flags.c_contiguous:raise ValueError('Resident bed required')
        self.launch('mg_faces',self.nf,(q,bed,a['xy'],a['mid'],a['normal'],a['length'],a['owner'],a['neighbour'],nc,nf,dry,np.float64(self.nu),
            b['conductance'],b['face_diffusivity'],b['tangent_offset'],b['normal_asymmetry'],self.face_status))
        if int(cp.max(self.face_status).item()):raise ValueError('Invalid internal face geometry or nonfinite diffusion conductance')
        changed=not self.ready or bool(cp.any(self.topology!=(b['conductance']>0)).item())
        if changed:
            self.launch('mg_boundary',self.nc,(q,b['conductance'],a['owner'],a['neighbour'],a['offset'],a['faces'],a['signs'],nc,dry,self.boundary))
            self.launch('mg_corrected',self.nf,(b['conductance'],b['tangent_offset'],b['normal_asymmetry'],a['owner'],a['neighbour'],self.boundary,nf,b['corrected_face']))
            self.launch('mg_slots',self.nc,(b['corrected_face'],a['offset'],a['faces'],nc,b['recovery_slot']))
            self.launch('mg_recovery',self.nc,(a['area'],a['xy'],a['mid'],a['normal'],a['length'],b['conductance'],a['owner'],a['neighbour'],
                a['offset'],a['faces'],a['signs'],b['recovery_slot'],nc,nf,b['recovery_stencils'],b['recovery_stencil_counts'],
                b['recovery_inverse'],b['recovery_affine_inverse'],b['recovery_order'],b['recovery_valid']))
            cp.greater(b['conductance'],0,out=self.topology);self.rebuilds+=1;self.ready=True
        self.launch('mg_bound',self.nf,(a['area'],a['xy'],a['mid'],a['normal'],b['conductance'],b['face_diffusivity'],a['owner'],a['neighbour'],
            b['recovery_slot'],b['recovery_stencils'],b['recovery_stencil_counts'],b['recovery_inverse'],b['recovery_affine_inverse'],
            b['recovery_order'],b['recovery_valid'],nc,nf,b['corrected_face'],b['face_bound']))
        self.launch('mg_rate',self.nc,(q,a['area'],b['face_bound'],a['offset'],a['faces'],nc,dry,b['rate']))
        cp.max(b['rate'],out=self.maximum_rate)
        return self.cache

    def install(self,operator):
        """Bind GPU arrays without downloading any cache fields."""
        if not self.ready:raise ValueError('Geometry must be built before installation')
        changed=any(operator.a[name].data.ptr!=value.data.ptr for name,value in self.cache.items())
        operator.a.update(self.cache);operator.maximum_rate=float(self.maximum_rate.item())
        if operator.nr!=self.nc:
            operator.nr=self.nc;operator.polynomial=self.cp.empty((2,5,self.nc));operator.recovery_status=self.cp.empty(self.nc,dtype=self.cp.int32)
            changed=True
        if changed:operator.graph=None


# Imported after defining the geometry class so either module import order works.
from .momentum_diffusion import ResidentMomentumDiffusion


class ExperimentalDynamicMomentumDiffusion(ResidentMomentumDiffusion):
    """Preserved explicit wrapper used by the first isolated geometry proof."""
    def __init__(self,data,source_path=None,*,execution='graph'):
        super().__init__(data,source_path=source_path,execution=execution,dynamic_geometry=False)
        self.dynamic_geometry=ResidentDiffusionGeometry(data,source_path=self.source_path.with_name('momentum_geometry.cu'))

    def apply(self,q,bed,dt):
        try:return super().apply(q,bed,dt)
        except NotImplementedError:
            self.dynamic_geometry.update(q,bed);self.dynamic_geometry.install(self)
            return super().apply(q,bed,dt)
