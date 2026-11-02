"""Resident general-avalanche proposal for a single bounded diffusion substep.

Mirrors the current native BRAIDS `avalanche_erodible_banks` geometry and wet/dry
mobility. The caller supplies the effective bed elevation (including any bank
front), depth and exposed active inventory. This component returns proposed
signed total kg per (grain, face); `ResidentSurfaceTransfer` must limit/gather
that proposal, followed by active-layer rebalance and additional substeps.
It does not mutate the supplied state or claim a complete avalanche time step.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .surface_transfer import validate_geometry

HERE=Path(__file__).resolve().parent
OPTIONS=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true','--ftz=false')


class ResidentGeneralAvalancheProposal:
    def __init__(self,area,owner,neighbour,offset,faces,signs,xy,mid,normal,length,
                 bulk,*,dry_tolerance,wet_slope,dry_slope,wet_depth_switch,
                 relaxation_length,relaxation_time,source_path=None):
        import cupy as cp
        topology,area=validate_geometry(area,owner,neighbour,offset,faces,signs)
        owner,neighbour,offset,faces,signs=topology
        nc=len(area);nf=len(owner);ni=len(faces)
        bulk=np.asarray(bulk,dtype=np.float64)
        controls=np.asarray((dry_tolerance,wet_slope,dry_slope,wet_depth_switch,
                             relaxation_length,relaxation_time),dtype=np.float64)
        if bulk.ndim!=1 or not 1<=len(bulk)<=32 or not np.isfinite(bulk).all() or np.any(bulk<=0):
            raise ValueError('Expected one to 32 positive bulk densities')
        if not np.isfinite(controls).all() or controls[0]<0 or controls[1]<=0 or controls[2]<=0 or \
                controls[3]<0 or controls[4]<=0 or controls[5]<=0:
            raise ValueError('Invalid general-avalanche controls')
        geometry={}
        for name,value,shape in [('xy',xy,(2,nc)),('mid',mid,(2,nf)),
                                 ('normal',normal,(2,nf)),('length',length,(nf,))]:
            values=np.asarray(value,dtype=np.float64)
            if values.shape!=shape or not np.isfinite(values).all():
                raise ValueError('Invalid avalanche geometry '+name)
            geometry[name]=np.ascontiguousarray(values)
        if np.any(geometry['length']<=0) or np.any(np.abs(np.sum(geometry['normal']**2,axis=0)-1)>1e-10):
            raise ValueError('Face lengths must be positive and normals unit length')
        delta=np.zeros((2,ni),np.float64);weight=np.zeros(ni,np.float64)
        matrix=np.zeros((4,nc),np.float64)
        for c in range(nc):
            a11=a12=a22=0.
            for k in range(int(offset[c]),int(offset[c+1])):
                f=int(faces[k]);n=int(neighbour[f] if signs[k]>0 else owner[f])
                if n<0:continue
                r=geometry['xy'][:,n]-geometry['xy'][:,c]
                rr=float(np.dot(r,r))
                if rr<=0 or not np.isfinite(rr):raise ValueError('Coincident connected cell centroids')
                w=1./rr;delta[:,k]=r;weight[k]=w
                a11+=w*r[0]*r[0];a12+=w*r[0]*r[1];a22+=w*r[1]*r[1]
            matrix[:,c]=(a11,a12,a22,a11*a22-a12*a12)
        self.cp=cp;self.nc=nc;self.nf=nf;self.ni=ni;self.ng=len(bulk)
        self.diffusivity=np.float64(.5*relaxation_length**2/relaxation_time)
        self.controls=(np.float64(wet_slope),np.float64(dry_slope),
                       np.float64(wet_depth_switch),np.float64(dry_tolerance))
        self.area=cp.asarray(area);self.bulk=cp.asarray(np.ascontiguousarray(bulk))
        self.owner,self.neighbour,self.offset,self.faces,self.signs=(cp.asarray(v) for v in topology)
        self.xy=cp.asarray(geometry['xy']);self.mid=cp.asarray(geometry['mid'])
        self.normal=cp.asarray(geometry['normal']);self.length=cp.asarray(geometry['length'])
        self.delta=cp.asarray(delta);self.weight_ls=cp.asarray(weight);self.matrix=cp.asarray(matrix)
        source=Path(source_path or HERE/'avalanche_proposal.cu').read_text()
        names=('av_downslopes','av_depth_intercept','av_face_rate','av_cell_rate','av_subdt','av_propose')
        module=cp.RawModule(code=source,options=OPTIONS,name_expressions=names)
        self.k={name:module.get_function(name) for name in names}
        self.down=cp.empty((2,nc));self.intercept=cp.empty(nc)
        self.rate=cp.empty(nf);self.conductance=cp.empty(nf)
        self.cell_rate=cp.empty(nc);self.maximum_rate=cp.empty((),cp.float64)
        self.subdt=cp.empty((),cp.float64);self.proposed=cp.empty((self.ng,nf))
        self.zero_front=cp.zeros(nc,cp.int32)
        self.status={name:cp.empty(size,cp.int32) for name,size in
                     [('down',nc),('intercept',nc),('face',nf),('cell_rate',nc),
                      ('subdt',1),('proposal',nf)]}

    def _array(self,value,shape,dtype):
        if not isinstance(value,self.cp.ndarray) or value.shape!=shape or \
                value.dtype!=dtype or not value.flags.c_contiguous:
            raise ValueError('Expected contiguous device '+str(dtype)+' array '+str(shape))

    def _launch(self,name,n,args):
        if n:self.k[name](((n+127)//128,),(128,),args)

    def step(self,bed,depth,active,remaining,*,front_active=None):
        """Return one face proposal and its internally bounded substep dt."""
        cp=self.cp;nc=np.int32(self.nc);nf=np.int32(self.nf);ng=np.int32(self.ng)
        self._array(bed,(self.nc,),cp.float64)
        self._array(depth,(self.nc,),cp.float64)
        self._array(active,(self.ng,self.nc),cp.float64)
        if front_active is None:front_active=self.zero_front
        self._array(front_active,(self.nc,),cp.int32)
        if not np.isfinite(remaining) or remaining<=0:
            raise ValueError('Positive remaining relaxation time required')
        self._launch('av_downslopes',self.nc,(nc,nf,np.int32(self.ni),bed,self.owner,self.neighbour,
            self.offset,self.faces,self.signs,self.delta,self.weight_ls,self.matrix,
            self.down,self.status['down']))
        self._launch('av_depth_intercept',self.nc,(nc,nf,depth,self.down,self.xy,self.mid,
            self.normal,self.length,self.area,self.offset,self.faces,self.signs,
            self.intercept,self.status['intercept']))
        self._launch('av_face_rate',self.nf,(nc,nf,bed,self.down,self.intercept,self.xy,self.mid,
            self.normal,self.length,self.owner,self.neighbour,front_active,self.diffusivity,
            *self.controls,self.rate,self.conductance,self.status['face']))
        self._launch('av_cell_rate',self.nc,(nc,self.conductance,self.area,self.offset,self.faces,
            self.cell_rate,self.status['cell_rate']))
        cp.max(self.cell_rate,out=self.maximum_rate)
        self._launch('av_subdt',1,(self.maximum_rate,np.float64(remaining),
            self.subdt,self.status['subdt']))
        self._launch('av_propose',self.nf,(nc,nf,ng,active,self.bulk,self.rate,
            self.owner,self.neighbour,self.subdt,self.proposed,self.status['proposal']))
        return {'proposed':self.proposed,'subdt':self.subdt,'face_rate':self.rate,
                'conductance':self.conductance,'cell_rate':self.cell_rate,
                'down':self.down,'depth_intercept':self.intercept,
                'status':tuple(self.status.values())}
