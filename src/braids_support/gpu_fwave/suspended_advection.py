"""Isolated resident GPU suspended advection using accepted hydraulic flux.

The caller supplies reconstructed face concentrations, current and accepted
next water depths, the post-draining water flux, and current boundary forcing.
This module advances only suspended mass. Exchange, timestep integration and
checkpointing belong to the later coupled caller.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .sediment_transport import validate_topology

HERE=Path(__file__).resolve().parent
OPTIONS=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true','--ftz=false')


class ResidentSuspendedAdvection:
    def __init__(self,data,ng,*,source_path=None):
        import cupy as cp
        nc=int(np.asarray(data['area']).size);nf=len(data['owner'])
        nb=int(data['nb']) if 'nb' in data else len(data['mode'])
        if not isinstance(ng,(int,np.integer)) or not 1<=ng<=32 or nc<1 or nf<1 or nb<0:
            raise ValueError('Invalid fixed-grid suspended dimensions')
        if int(data.get('scheme',6))!=6 or bool(data.get('amr_enabled',False)):
            raise ValueError('Isolated suspended FCT requires fixed-grid scheme 6')
        validate_topology(data,nc,nf,nb)
        if 'active' in data and not np.all(np.asarray(data['active'])==1):
            raise ValueError('Inactive/AMR cells are not ported by isolated suspended advection')
        self.cp=cp;self.nc=nc;self.nf=nf;self.ng=int(ng);self.nb=nb
        self.dry=float(data['dry_tolerance'] if 'dry_tolerance' in data else data['dry'])
        if not np.isfinite(self.dry) or self.dry<0:raise ValueError('Invalid dry tolerance')
        shapes={'area':(nc,),'length':(nf,),'normal':(2,nf),'rectangle_width':(2,nc),
                'weno_cells':(5,2,nf),'owner':(nf,),'neighbour':(nf,),'bid':(nf,),
                'offset':(nc+1,),'faces':(len(data['faces']),),'signs':(len(data['faces']),)}
        integer={'weno_cells','owner','neighbour','bid','offset','faces','signs'}
        self.geometry={}
        for name,shape in shapes.items():
            value=np.asarray(data[name])
            if value.shape!=shape or (name not in integer and not np.isfinite(value).all()):
                raise ValueError('Invalid suspended geometry '+name)
            if name in integer and (value.dtype.kind not in 'iu' or
                                    np.any(value>np.iinfo(np.int32).max) or np.any(value<np.iinfo(np.int32).min)):
                raise ValueError('Invalid suspended integer geometry '+name)
            self.geometry[name]=cp.asarray(np.ascontiguousarray(value,dtype=np.int32 if name in integer else np.float64))
        if np.any(np.asarray(data['area'])<=0) or np.any(np.asarray(data['length'])<=0) or np.any(np.asarray(data['rectangle_width'])<0):
            raise ValueError('Positive area/length and nonnegative rectangle widths required')
        weno=np.asarray(data['weno_cells']);present=weno[0]>=0
        if np.any(weno[:,present]<0) or np.any(weno>=nc):
            raise ValueError('Invalid suspended smooth-bound stencil')
        bid=np.asarray(data['bid'])
        lists=[np.flatnonzero(bid==b).astype(np.int32) for b in range(nb)]
        boundary_faces=np.concatenate(lists) if lists else np.empty(0,np.int32)
        boundary_offset=np.r_[0,np.cumsum([len(indices) for indices in lists])].astype(np.int32)
        self.geometry['boundary_faces']=cp.asarray(boundary_faces)
        self.geometry['boundary_offset']=cp.asarray(boundary_offset)
        names=('sa_concentration','sa_face_low','sa_cell_bounds','sa_limit_faces','sa_update','sa_boundary_ledger')
        code=Path(source_path or HERE/'suspended_advection.cu').read_text()
        module=cp.RawModule(code=code,options=OPTIONS,name_expressions=names)
        self.k={name:module.get_function(name) for name in names}
        self.buffers={
            'concentration':cp.empty((ng,nc)),'low_flux':cp.empty((ng,nf)),
            'anti':cp.empty((ng,nf)),'low_state':cp.empty((ng,nc)),
            'rplus':cp.empty((ng,nc)),'rminus':cp.empty((ng,nc)),
            'scaled_anti':cp.empty((ng,nf)),'flux':cp.empty((ng,nf)),
            'next_mass':cp.empty((ng,nc)),'next_concentration':cp.empty((ng,nc)),
            'ledger':cp.empty((ng,nb)),
            'concentration_status':cp.empty((ng,nc),cp.int32),
            'face_status':cp.empty((ng,nf),cp.int32),
            'cell_status':cp.empty((ng,nc),cp.int32),
            'flux_status':cp.empty((ng,nf),cp.int32),
            'update_status':cp.empty((ng,nc),cp.int32),
            'ledger_status':cp.empty((ng,nb),cp.int32)}

    def _array(self,value,shape,dtype):
        if not isinstance(value,self.cp.ndarray) or value.shape!=shape or value.dtype!=dtype or not value.flags.c_contiguous:
            raise ValueError('Expected contiguous device array '+str(shape))

    def _launch(self,name,n,args):
        if n:self.k[name](((n+127)//128,),(128,),args)

    def step(self,depth,mass,next_depth,accepted_water_flux,left,right,boundary_concentration,
             hydro_constant,dt):
        """Return trial mass, concentration, face flux, signed boundary ledger and statuses."""
        cp=self.cp;a=self.geometry;b=self.buffers;nc,nf,ng,nb=map(np.int32,(self.nc,self.nf,self.ng,self.nb))
        for value,shape in ((depth,(self.nc,)),(mass,(self.ng,self.nc)),(next_depth,(self.nc,)),
                            (accepted_water_flux,(self.nf,)),(left,(self.ng,self.nf)),
                            (right,(self.ng,self.nf)),(boundary_concentration,(self.ng,self.nb))):
            self._array(value,shape,cp.float64)
        self._array(hydro_constant,(self.nc,),cp.int32)
        if not np.isfinite(dt) or dt<=0:raise ValueError('Positive finite timestep required')
        dt=np.float64(dt)
        self._launch('sa_concentration',self.nc*self.ng,
            (nc,ng,depth,mass,np.float64(self.dry),b['concentration'],b['concentration_status']))
        self._launch('sa_face_low',self.nf*self.ng,
            (nc,nf,ng,nb,accepted_water_flux,b['concentration'],left,right,boundary_concentration,
             a['owner'],a['neighbour'],a['bid'],b['concentration_status'],
             b['low_flux'],b['anti'],b['face_status']))
        self._launch('sa_cell_bounds',self.nc*self.ng,
            (nc,nf,ng,nb,mass,next_depth,b['concentration'],boundary_concentration,accepted_water_flux,
             b['low_flux'],b['anti'],a['area'],a['length'],a['normal'],a['rectangle_width'],
             hydro_constant,a['weno_cells'],a['owner'],a['neighbour'],a['bid'],a['offset'],
             a['faces'],a['signs'],b['concentration_status'],b['face_status'],dt,
             b['low_state'],b['rplus'],b['rminus'],b['cell_status']))
        self._launch('sa_limit_faces',self.nf*self.ng,
            (nc,nf,ng,b['low_flux'],b['anti'],b['rplus'],b['rminus'],a['owner'],a['neighbour'],
             b['cell_status'],b['scaled_anti'],b['flux'],b['flux_status']))
        self._launch('sa_update',self.nc*self.ng,
            (nc,nf,ng,b['low_state'],b['scaled_anti'],next_depth,a['length'],a['area'],
             a['offset'],a['faces'],a['signs'],b['cell_status'],b['flux_status'],dt,
             np.float64(self.dry),b['next_mass'],b['next_concentration'],b['update_status']))
        self._launch('sa_boundary_ledger',self.nb*self.ng,
            (nf,ng,nb,a['boundary_offset'],a['boundary_faces'],b['flux'],a['length'],
             b['flux_status'],dt,b['ledger'],b['ledger_status']))
        return {'next_mass':b['next_mass'],'next_concentration':b['next_concentration'],
                'flux':b['flux'],'ledger':b['ledger'],
                'status':(b['concentration_status'],b['face_status'],b['cell_status'],
                          b['flux_status'],b['update_status'],b['ledger_status'])}
