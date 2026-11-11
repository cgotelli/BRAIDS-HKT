"""Call-local GPU batching for magnitude-referenced fixed-axis yaw/device rotors.

Group only identical physical controls and diameter; use the original operators.
Exceptional groups fall back to the owner so error ordering and shutdown remain
unchanged. No analytical wake correction or state is cached here.
"""
import json
import copy
import numpy as np


def eligible(r):
    from .river_turbine_yaw import FixedAxisRotor
    return type(r) is FixedAxisRotor and r.xp is not np


class YawRotorBatch:
    def __init__(self, rotors):
        self.rotors=tuple(rotors);self.xp=rotors[0].xp;xp=self.xp
        self.off=np.r_[0,np.cumsum([len(r.ids) for r in rotors])]
        self.poff=np.r_[0,np.cumsum([len(r.plane) for r in rotors])]
        self.ids=xp.concatenate([r.ids for r in rotors]);self.plane=xp.concatenate([r.plane for r in rotors])
        self.delta=xp.concatenate([r.delta for r in rotors],axis=1)
        self.weights=xp.concatenate([r.normalized[r.ids] for r in rotors])
        self.den=xp.concatenate([r.density_integrals[r.ids] for r in rotors])
        self.area=xp.concatenate([r.area[r.ids] for r in rotors])
        self.starts=xp.asarray(self.off[:-1]);self.pstarts=xp.asarray(self.poff[:-1]);self._key=None

    def evaluate(self,q,z,grad):
        xp=self.xp
        from .river_turbine_yaw import yaw_terms,bounds,device_yaw
        # Controls may be replaced/edited; immutable geometry alone survives calls.
        key=tuple((r.dry,tuple(r.normal),r.c['D'],json.dumps(r.control,sort_keys=True,default=lambda v: np.asarray(v).tolist()),json.dumps(r.device_setting,sort_keys=True,default=lambda v: np.asarray(v).tolist()),r.device_yaw) for r in self.rotors)
        if key!=self._key:
            sizes=np.diff(self.off)
            self.dry=xp.asarray(np.repeat([r.dry for r in self.rotors],sizes))
            self.nx=xp.asarray(np.repeat([r.normal[0] for r in self.rotors],sizes));self.ny=xp.asarray(np.repeat([r.normal[1] for r in self.rotors],sizes))
            grouped={}
            for j,r in enumerate(self.rotors):grouped.setdefault(key[j][2:],[]).append(j)
            self.groups=[]
            for indices in grouped.values():
                idx=xp.asarray(np.concatenate([np.arange(self.off[j],self.off[j+1]) for j in indices]))
                self.groups.append((indices,idx,copy.deepcopy(self.rotors[indices[0]].control)))
            self._key=key
        h=q[0,self.ids];w=self.weights
        u=q[1,self.ids]/xp.maximum(h,self.dry);v=q[2,self.ids]/xp.maximum(h,self.dry)
        un=u*self.nx+v*self.ny;ut=-u*self.ny+v*self.nx
        eta=q[0,self.plane]+z[self.plane]
        surface=eta+grad[4,self.plane]*self.delta[0]+grad[5,self.plane]*self.delta[1]
        bed=z[self.plane]+grad[6,self.plane]*self.delta[0]+grad[7,self.plane]*self.delta[1]
        stats=_reduce(xp,xp.stack((h<=self.dry,w*un,w*h,w*ut,un)),self.starts,[1,0,0,0,-1])
        plane_stats=_reduce(xp,xp.stack((xp.minimum(surface,eta),xp.maximum(bed,z[self.plane]))),self.pstarts,[-1,1])
        stats=xp.concatenate((plane_stats,stats),axis=0)
        K=xp.zeros_like(h);P=xp.zeros_like(h);U=xp.zeros_like(h);Ct=xp.zeros_like(h);Cp=xp.zeros_like(h)
        low=xp.zeros_like(h);high=xp.zeros_like(h);fallback=set()
        for indices,idx,control in self.groups:
            r=self.rotors[indices[0]];hh=h[idx];nn=un[idx];tt=ut[idx]
            if r.control['mode']!='constant_ct':low[idx],high[idx]=bounds(r.control,r.c['D'],hh,nn,tt,xp)
            try:
                kk,pp,uu,ct,cp=yaw_terms(control,r.c['D'],hh,nn,tt,xp)
                if r.device_yaw:
                    # Low speed and invalid table states must be adjudicated by
                    # each rotor after coverage/reverse/curve checks, in order.
                    pp,cp=device_yaw.correct(r.device_setting,pp,uu,tt,cp,xp)
                K[idx]=kk;P[idx]=pp;U[idx]=uu;Ct[idx]=ct;Cp[idx]=cp
            except ValueError:
                fallback.update(indices)
        values=_reduce(xp,xp.stack((K*self.den*un**2,K*self.den*abs(un)**3,P*self.den,w*U,w*Ct,w*Cp,low,high)),self.starts,[0,0,0,0,0,0,1,1])
        host=xp.concatenate((stats,values),axis=0).get()
        k=K*self.den/self.area
        result=[]
        for j,r in enumerate(self.rotors):
            if j in fallback:result.append(r._sample_packed(q,z,grad,None));continue
            a,b=self.off[j:j+2]
            packed=(h[a:b],w[a:b],un[a:b],ut[a:b],host[:7,j],k[a:b],host[7:13,j],host[13:15,j])
            result.append(r._sample_packed(q,z,grad,packed))
        return result


_REDUCE=None
_MODES={}
def _reduce(xp,values,starts,modes):
    global _REDUCE
    if _REDUCE is None:
        _REDUCE=xp.RawKernel(r'''extern "C" __global__ void reduce_rows(int n,int nr,int rows,const double* v,const long long* starts,const int* mode,double* out){
          int j=blockDim.x*blockIdx.x+threadIdx.x;if(j>=nr*rows)return;
          int row=j/nr,r=j%nr;long long a=starts[r],b=r+1<nr?starts[r+1]:n;
          double s=mode[row]==0?0.:(mode[row]<0?1./0.:-1./0.);
          for(long long k=a;k<b;k++){double x=v[row*n+k];s=mode[row]==0?s+x:(isnan(s)||isnan(x)?0./0.:(mode[row]<0?fmin(s,x):fmax(s,x)));}
          out[j]=s;
        }''','reduce_rows',options=('--fmad=false',))
    key=(int(xp.cuda.Device().id),tuple(modes))
    if key not in _MODES:_MODES[key]=xp.asarray(modes,dtype=xp.int32)
    nr=len(starts);out=xp.empty((len(modes),nr));v=xp.ascontiguousarray(values,dtype=xp.float64)
    _REDUCE(((out.size+127)//128,),(128,),(np.int32(v.shape[1]),np.int32(nr),np.int32(len(modes)),v,starts,_MODES[key],out))
    return out
