"""Resident scheme-6 suspended concentration face reconstruction.

This is the scalar dependency of native cutcell_solver gradients/face_value.
It consumes the resident hydraulic reconstruction's immutable geometry and
stage-specific active/wall-ready/hydro-constant buffers. It does not perform
hydraulic reconstruction, sediment FCT, exchange or a coupled RK commit.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OPTIONS=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true','--ftz=false')


class ResidentSuspendedFaces:
    """Supply (grain,face) left/right concentration without a host stage copy."""

    def __init__(self,reconstruction,grain_count,boundary_count,source_path=None):
        import cupy as cp
        if (not isinstance(grain_count,(int,np.integer)) or not 1<=grain_count<=32 or
            not isinstance(boundary_count,(int,np.integer)) or boundary_count<0):
            raise ValueError('Invalid grain/boundary count')
        if reconstruction.cp is not cp:
            raise ValueError('A resident hydraulic reconstruction is required')
        self.cp=cp;self.reconstruction=reconstruction
        self.nc=reconstruction.nc;self.nf=reconstruction.nf;self.ng=int(grain_count)
        self.nb=int(boundary_count);self.nadj=reconstruction.nadj
        self.dry=reconstruction.controls['dry'];self.grav=reconstruction.controls['grav']
        self.outlet_extension=np.int32(reconstruction.outlet_extension)
        a=reconstruction.geometry
        bids=cp.asnumpy(a['bid'])
        if np.any(bids>=self.nb):
            raise ValueError('Boundary concentration count does not match geometry')
        names=('sc_primitive','sc_gradients','sc_wall_values','sc_faces','sc_status_merge')
        module=cp.RawModule(code=Path(source_path or HERE/'suspended_faces.cu').read_text(),
                            options=OPTIONS,name_expressions=names)
        self.k={name:module.get_function(name) for name in names}
        ng,nc,nf=self.ng,self.nc,self.nf
        self.buffers={'primitive':cp.empty((ng,nc),cp.float64),
            'grad':cp.empty((ng,2,nc),cp.float64),
            'vlo':cp.empty((ng,nc),cp.float64),'vhi':cp.empty((ng,nc),cp.float64),
            'wall_values':cp.empty((ng,4,nc),cp.float64),
            'left':cp.empty((ng,nf),cp.float64),'right':cp.empty((ng,nf),cp.float64),
            'primitive_status':cp.empty(nc,cp.int32),
            'gradient_status':cp.empty(nc,cp.int32),'face_status':cp.empty(nf,cp.int32),
            'status':cp.zeros((),cp.int32)}

    def _check(self,value,shape,name,dtype):
        if (not isinstance(value,self.cp.ndarray) or value.shape!=shape or
            value.dtype!=dtype or not value.flags.c_contiguous):
            raise ValueError(f'{name} must be contiguous device {dtype} {shape}')

    def _launch(self,name,n,args):
        self.k[name](((n+127)//128,),(128,),args)

    def step(self,depth,suspended_mass,boundary_concentration,hydraulic=None):
        """Reconstruct one Euler-stage trial; outputs overwritten at next call.

        `hydraulic` is the current ResidentReconstruction.reconstruct() result.
        Every state and boundary value stays on device. Global commit/reject
        remains the coupled stepper's responsibility.
        """
        cp=self.cp;nc=np.int32(self.nc);nf=np.int32(self.nf);ng=np.int32(self.ng)
        nadj=np.int32(self.nadj);nb=np.int32(self.nb)
        hydraulic=self.reconstruction.buffers if hydraulic is None else hydraulic
        self._check(depth,(self.nc,),'depth',cp.float64)
        self._check(suspended_mass,(self.ng,self.nc),'suspended_mass',cp.float64)
        self._check(boundary_concentration,(self.ng,self.nb),'boundary_concentration',cp.float64)
        self._check(hydraulic['primitive'],(4,self.nc),'hydraulic primitive',cp.float64)
        self._check(hydraulic['active'],(self.nc,),'hydraulic active',cp.int32)
        self._check(hydraulic['wall_ready'],(4,self.nc),'hydraulic wall_ready',cp.int32)
        self._check(hydraulic['hydro_constant'],(self.nc,),'hydro_constant',cp.int32)
        a=self.reconstruction.geometry;b=self.buffers
        self._launch('sc_primitive',self.nc,(nc,ng,self.dry,depth,suspended_mass,
            b['primitive'],b['primitive_status']))
        self._launch('sc_gradients',self.nc,(nc,nf,nadj,ng,nb,self.dry,self.grav,
            self.outlet_extension,b['primitive'],hydraulic['primitive'],
            boundary_concentration,a['xy'],a['mid'],a['normal'],a['owner'],
            a['neighbour'],a['bid'],a['boundary_mode'],a['offset'],a['faces'],
            a['signs'],hydraulic['active'],a['hydraulic_ls_matrix'],
            a['hydraulic_ls_weight'],a['bed_ls_delta'],b['grad'],b['vlo'],
            b['vhi'],b['gradient_status']))
        self._launch('sc_wall_values',self.nc,(nc,ng,b['primitive'],
            a['wall_stencil_ids'],hydraulic['wall_ready'],b['wall_values']))
        self._launch('sc_faces',self.nf,(nc,nf,ng,nb,self.dry,depth,b['primitive'],
            b['grad'],b['wall_values'],hydraulic['wall_ready'],
            boundary_concentration,a['owner'],a['neighbour'],a['bid'],
            a['wall_slot'],a['weno_cells'],a['face_displacement'],b['left'],
            b['right'],b['face_status']))
        b['status'].fill(0)
        self._launch('sc_status_merge',max(self.nc,self.nf),(nc,nf,
            b['primitive_status'],b['gradient_status'],b['face_status'],b['status']))
        return b
