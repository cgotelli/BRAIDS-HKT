"""Physical-time tables for zero-area exterior bed and composition history.

Bed elevation and surface fractions are endpoint states. Elevation rate is
sampled at the outer midpoint and integrated with morphological dt. These
inputs never overwrite a physical cell or change the hydraulic integrator.
"""
from copy import deepcopy
import math
from types import MethodType
import numpy as np
from .river_schedule import CONSTANT, LINEAR, bind_breakpoints, check_interval

BED_FIELDS={'bed.free':None,'bed.fixed':None,'bed.elevation':'elevation_m',
            'bed.elevation_rate':'elevation_rate_m_s'}
COMPOSITION_FIELDS={'composition.free':None,'composition.fixed':None,
    'composition.mass_fraction':'fractions','composition.reference_volume_fraction':'fractions'}


class ExteriorForcing:
    def __init__(self,xp,methods,schedules,end_time,bids,initial,*,composition=False):
        if not math.isfinite(end_time) or end_time<=0:raise ValueError('Positive exterior schedule run end required')
        fields=COMPOSITION_FIELDS if composition else BED_FIELDS
        if schedules is None:schedules={b:dict(method=CONSTANT) for b in methods}
        if not isinstance(schedules,dict) or set(schedules)!=set(methods):
            raise ValueError('Explicit exterior schedule for every boundary required')
        self.document={};self.tables=[];knots=[]
        bids=np.asarray(bids);initial=np.asarray(initial)
        for bid,row in schedules.items():
            if type(bid) is not int or not isinstance(row,dict) or methods[bid] not in fields:
                raise ValueError('Invalid exterior schedule boundary or physical method')
            field=fields[methods[bid]];method=row.get('method')
            if method==CONSTANT:
                if set(row)!={'method'}:raise ValueError('Constant exterior forcing cannot contain a table')
            elif method==LINEAR:
                if field is None:raise ValueError('This exterior method has no prescribed time-table fields')
                if set(row)!={'method','time_s',field}:raise ValueError('Exterior table fields differ from its method')
                times=row['time_s'];values=row[field]
                if (not isinstance(times,list) or len(times)<2 or
                    any(type(t) not in (int,float) or not math.isfinite(t) for t in times) or
                    times[0]!=0 or times[-1]<end_time or any(b<=a for a,b in zip(times,times[1:]))):
                    raise ValueError('Exterior table must start at zero, increase strictly and cover the run')
                if not isinstance(values,list) or len(values)!=len(times):
                    raise ValueError('Exterior table values must match its times')
                if composition:
                    ng=initial.shape[0]
                    if (any(not isinstance(v,list) or len(v)!=ng for v in values) or
                        any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for v in values for x in v) or
                        not np.allclose(np.sum(values,axis=1),1.,rtol=0,atol=1e-12)):
                        raise ValueError('Normalized nonnegative time-by-fraction table required')
                elif any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
                    raise ValueError('Finite signed exterior elevation/rate values required')
                faces=np.flatnonzero(bids==bid)
                if not len(faces):raise ValueError('Scheduled exterior boundary has no faces')
                first=np.asarray(values[0]);expected=first[:,None] if composition else first
                if not np.all(initial[...,faces]==expected):
                    raise ValueError('Initial exterior forcing must equal first table value on every selected face')
                time=np.array(times,float);time.setflags(write=False)
                self.tables.append((xp.asarray(faces),time,xp.asarray(values,dtype=xp.float64).copy(),
                                    not composition and methods[bid]=='bed.elevation_rate'))
                knots.extend(times)
            else:raise ValueError('Explicit constant or linear physical-time exterior forcing required')
            self.document[str(bid)]=deepcopy(row)
        self.knots=tuple(sorted(set(knots)))

    def sample(self,start,dt,target):
        if not math.isfinite(start) or not math.isfinite(dt) or dt<0:
            raise ValueError('Finite exterior evaluation clock required')
        for faces,times,values,rate in self.tables:
            time=start+(.5*dt if rate else dt)
            if time<times[0] or time>times[-1]:raise ValueError('Exterior table does not cover evaluation time')
            i=min(int(np.searchsorted(times,time,side='right'))-1,len(times)-2)
            weight=(time-times[i])/(times[i+1]-times[i])
            v=(1.-weight)*values[i]+weight*values[i+1]
            target[...,faces]=v[:,None] if target.ndim==2 else v

    def bind(self,a,name):
        if not self.tables:return
        bind_breakpoints(a,name,self.knots);original=a.step
        def step(owner,dt,*args,**kwargs):
            check_interval(owner,dt)
            return original(dt,*args,**kwargs)
        a.step=MethodType(step,a)
