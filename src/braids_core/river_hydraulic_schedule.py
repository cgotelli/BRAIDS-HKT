"""Immutable named hydraulic tables evaluated at the actual residual clock."""
from copy import deepcopy
import math
from types import MethodType
import numpy as np
from .river_schedule import CONSTANT,LINEAR,bind_breakpoints,check_interval

from .river_open_boundary import HYDRAULICS
FIELDS={name:row[3] for name,row in HYDRAULICS.items()}


def validate(controls,schedules,end_time):
    """Controls use resolved string IDs; a depth datum is fixed, not a table."""
    if not math.isfinite(end_time) or end_time<=0:raise ValueError('Positive hydraulic schedule run end required')
    if schedules is None:schedules={int(b):dict(method=CONSTANT) for b in controls}
    if not isinstance(schedules,dict) or set(schedules)!={int(b) for b in controls}:
        raise ValueError('Explicit hydraulic schedule for every boundary required')
    result={}
    for bid,row in schedules.items():
        if type(bid) is not int or not isinstance(row,dict):raise ValueError('Invalid hydraulic schedule boundary')
        control=controls[str(bid)];fields=FIELDS[control['method']]
        if row.get('method')==CONSTANT:
            if set(row)!={'method'}:raise ValueError('Constant hydraulic forcing cannot contain a table')
        elif row.get('method')==LINEAR:
            if not fields:raise ValueError('This hydraulic method has no prescribed time-table fields')
            if set(row)!={'method','time_s',*fields}:raise ValueError('Hydraulic table fields differ from its method')
            times=row['time_s']
            if (not isinstance(times,list) or len(times)<2 or
                any(type(t) not in (int,float) or not math.isfinite(t) for t in times) or
                times[0]!=0 or times[-1]<end_time or any(b<=a for a,b in zip(times,times[1:]))):
                raise ValueError('Hydraulic table must start at zero, increase strictly and cover the run')
            for key in fields:
                values=row[key]
                if (not isinstance(values,list) or len(values)!=len(times) or
                    any(type(v) not in (int,float) or not math.isfinite(v) or
                        (key!='stage_m' and v<0) for v in values)):
                    raise ValueError('Finite hydraulic table values required; Q and depth must be nonnegative')
                if values[0]!=control[key]:raise ValueError('Initial hydraulic control must equal first table value')
            if 'depth_m' in fields and any(not math.isfinite(v+control['datum_m']) for v in row['depth_m']):
                raise ValueError('Resolved scheduled stage is not finite')
        else:raise ValueError('Explicit constant or linear physical-time hydraulic forcing required')
        result[str(bid)]=deepcopy(row)
    return result


class HydraulicForcing:
    def __init__(self,xp,controls,schedules,end_time):
        self.document=validate(controls,schedules,end_time);self.tables=[];knots=[]
        for bid,row in self.document.items():
            if row['method']==CONSTANT:continue
            control=controls[bid];times=np.array(row['time_s'],float);times.setflags(write=False)
            values={key:xp.asarray(row[key],dtype=xp.float64).copy() for key in FIELDS[control['method']]}
            self.tables.append((int(bid),times,values,float(control.get('datum_m',0.))))
            knots.extend(times.tolist())
        self.knots=tuple(sorted(set(knots)))

    def sample(self,time,forcing):
        if not math.isfinite(time):raise ValueError('Finite hydraulic evaluation time required')
        for bid,times,values,datum in self.tables:
            if time<times[0] or time>times[-1]:raise ValueError('Hydraulic table does not cover evaluation time')
            i=min(int(np.searchsorted(times,time,side='right'))-1,len(times)-2)
            weight=(time-times[i])/(times[i+1]-times[i])
            for key,series in values.items():
                sample=(1.-weight)*series[i]+weight*series[i+1]
                forcing[0 if key=='discharge_m3_s' else 1,bid]=sample+(datum if key=='depth_m' else 0.)

    def bind(self,a):
        if not self.tables:return
        bind_breakpoints(a,'hydraulics',self.knots);original=a.step
        def step(owner,dt,*args,**kwargs):
            check_interval(owner,dt)
            return original(dt,*args,**kwargs)
        a.step=MethodType(step,a)
