"""Explicit constant or linear sediment forcing on the physical clock.

Tables are immutable inputs, not evolving state. Every retried interval is
sampled anew at its midpoint; the run controller ends steps at table knots.
"""
from copy import deepcopy
import math
import numpy as np
from .river_boundary import FEED_FIELDS,feed_field

CONSTANT='forcing.constant'
LINEAR='forcing.linear_physical_time'


def bind_breakpoints(a,name,knots):
    """Compose immutable forcing clocks on the outer physical step owner."""
    clocks=vars(a).setdefault('_river_forcing_breakpoints',{})
    if name in clocks:raise ValueError('Forcing clock already installed: '+name)
    values=tuple(sorted(set(float(t) for t in knots)))
    if any(not math.isfinite(t) or t<0 for t in values):raise ValueError('Invalid physical forcing breakpoint')
    clocks[name]=values
    def next_time(now):
        tolerance=8*np.spacing(max(1.,abs(now)))
        return min((t for times in clocks.values() for t in times if t>now+tolerance),default=float('inf'))
    a.next_forcing_time=next_time


def check_interval(a,dt):
    if not math.isfinite(dt) or dt<=0:raise ValueError('Positive finite forcing interval required')
    end=a.time+dt
    if end>a.next_forcing_time(a.time)+8*np.spacing(max(1.,abs(end))):
        raise ValueError('Step crosses a forcing breakpoint')


def constants(boundaries):
    return {name:dict(method=CONSTANT) for name in boundaries}


def validate(schedules,boundaries,ng,end_time,*,sediment,suspended):
    if not isinstance(schedules,dict) or set(schedules)!=set(boundaries):
        raise ValueError('Explicit forcing schedule required for every open boundary')
    for name,row in schedules.items():
        if not isinstance(row,dict):raise ValueError(name+': forcing schedule must be an object')
        method=row.get('method')
        if method==CONSTANT:
            if set(row)!={'method'}:raise ValueError(name+': constant forcing cannot contain a table')
            continue
        if method!=LINEAR:raise ValueError(name+': explicit constant or linear physical-time method required')
        if not sediment:raise ValueError(name+': sediment is off but a schedule is active')
        key=feed_field(boundaries[name]['kind'])
        fields={'concentration_kg_m3'}|({key} if key else set())
        if set(row)!={'method','time_s'}|fields:raise ValueError(name+': schedule fields differ from the boundary method')
        times=row['time_s']
        if (not isinstance(times,list) or len(times)<2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in times)
            or times[0]!=0 or any(b<=a for a,b in zip(times,times[1:])) or times[-1]<end_time):
            raise ValueError(name+': table must start at zero, increase strictly and cover the run end in seconds')
        for key in fields:
            values=row[key]
            if (not isinstance(values,list) or len(values)!=len(times) or any(not isinstance(v,list) or len(v)!=ng for v in values)
                or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for v in values for x in v)):
                raise ValueError(name+': finite nonnegative time-by-fraction table required for '+key)
            if values[0]!=boundaries[name][key]:raise ValueError(name+': initial forcing must equal the first table row')
            if key=='concentration_kg_m3' and not suspended and any(x for v in values for x in v):
                raise ValueError(name+': resident suspension is off but scheduled concentration is nonzero')
    return schedules


class ScheduleDraft:
    """Named inactive UI text, kept separate from active numerical tables."""
    def __init__(self,value=None):
        self.data=deepcopy(value) if value is not None else dict(schema=2,boundaries={})
        self.snapshot()

    def ensure(self,ids,boundaries):
        for name in boundaries:
            row=self.data['boundaries'].setdefault(name,dict(method=CONSTANT,rows=[]))
            for record in row['rows']:
                for grain in ids:record['fractions'].setdefault(grain,{key:'0' for key in (*FEED_FIELDS,'concentration_kg_m3')})

    def snapshot(self):
        v=self.data
        if (not isinstance(v,dict) or set(v)!={'schema','boundaries'} or type(v['schema']) is not int or v['schema']!=2
            or not isinstance(v['boundaries'],dict)):raise ValueError('Current forcing schedule draft required')
        for name,row in v['boundaries'].items():
            if (not isinstance(name,str) or not name or not isinstance(row,dict) or set(row)!={'method','rows'}
                or row['method'] not in (CONSTANT,LINEAR) or not isinstance(row['rows'],list)):
                raise ValueError('Invalid named schedule draft')
            for record in row['rows']:
                if (not isinstance(record,dict) or set(record)!={'time_s','fractions'} or type(record['time_s']) is not str
                    or not isinstance(record['fractions'],dict)):raise ValueError('Invalid schedule row draft')
                for grain,f in record['fractions'].items():
                    if (type(grain) is not str or not grain or not isinstance(f,dict)
                        or set(f)!=set(FEED_FIELDS)|{'concentration_kg_m3'} or any(type(x) is not str for x in f.values())):
                        raise ValueError('Invalid named schedule fraction draft')
        return deepcopy(v)

    def active(self,ids,boundaries,*,sediment,suspended):
        from .river_forcing import nonnegative
        result=constants(boundaries)
        if not sediment:return result
        self.ensure(ids,boundaries)
        for name,boundary in boundaries.items():
            draft=self.data['boundaries'][name]
            if draft['method']==CONSTANT:continue
            records=draft['rows'];row=dict(method=LINEAR,time_s=[nonnegative(r['time_s'],name+' time') for r in records])
            key=feed_field(boundary['kind'])
            for key,enabled in ((key,key is not None),('concentration_kg_m3',True)):
                if not enabled:continue
                row[key]=[[nonnegative(r['fractions'][g][key],name+' / '+g+' '+key)
                    if key!='concentration_kg_m3' or suspended else 0. for g in ids] for r in records]
                if records:boundary[key]=row[key][0].copy()
            result[name]=row
        return result

    def load_active(self,project):
        if project['model']['physics']['sediment'] is None:return
        ids=project['settings']['columns']['fraction_ids'];bounds=project['settings']['sediment_boundaries']
        self.ensure(ids,bounds)
        for name,schedule in project['sediment_schedules'].items():
            target=self.data['boundaries'][name];target['method']=schedule['method']
            if schedule['method']==CONSTANT:continue
            # Preserve inactive fields/IDs if the table topology is unchanged.
            old=target['rows'];rows=[]
            for i,time in enumerate(schedule['time_s']):
                record=deepcopy(old[i]) if i<len(old) else dict(time_s='',fractions={})
                record['time_s']=str(time)
                for k,grain in enumerate(ids):
                    f=record['fractions'].setdefault(grain,{key:'0' for key in (*FEED_FIELDS,'concentration_kg_m3')})
                    for key in (*FEED_FIELDS,'concentration_kg_m3'):
                        if key in schedule and (key!='concentration_kg_m3' or project['model']['physics']['sediment']['suspension']!='off'):
                            f[key]=str(schedule[key][i][k])
                rows.append(record)
            target['rows']=rows


def install(solver,schedules,data):
    """Map immutable schedules to the current mesh, on the selected backend."""
    selected={name:s for name,s in schedules.items() if s['method']==LINEAR}
    if not selected:return
    from types import MethodType
    from .sediment.bend_methods_v1 import digest
    a=solver;p=a.physics;xp=a.xp
    tables=[];knots=[]
    for bid,name in enumerate(sorted(schedules)):
        if name not in selected:continue
        table=selected[name];time=np.asarray(table['time_s'],float)
        faces=np.flatnonzero((data['neighbour']<0)&(data['bid']==bid))
        if not len(faces):raise ValueError('Scheduled boundary has no prepared faces: '+name)
        tables.append((time,xp.asarray(faces),{key:xp.asarray(table[key],dtype=xp.float64)
            for key in ('solid_flux_m2_s','concentration_kg_m3') if key in table}))
        knots.extend(time.tolist())
    bind_breakpoints(a,'sediment',knots)
    original=a.step
    def step(owner,dt,*args,**kwargs):
        at=owner.time
        check_interval(owner,dt)
        sample=at+.5*dt;restore=[]
        try:
            for times,faces,values in tables:
                if sample<times[0] or sample>times[-1]:raise ValueError('Forcing table does not cover the interval')
                i=min(int(np.searchsorted(times,sample,side='right'))-1,len(times)-2)
                weight=(sample-times[i])/(times[i+1]-times[i])
                for key,series in values.items():
                    target=owner.feed if key=='solid_flux_m2_s' else p.suspended_boundary
                    restore.append((target,faces,target[:,faces].copy()))
                    target[:,faces]=((1.-weight)*series[i]+weight*series[i+1])[:,None]
            return original(dt,*args,**kwargs)
        finally:
            # Forcing is immutable input, not accepted mutable solver state.
            # Restore even after a rejected or cancelled physical transaction.
            for target,faces,old in restore:target[:,faces]=old
    a.step=MethodType(step,a)
    a.identity['sediment_schedule']=dict(method=LINEAR,source_sha256=digest(__file__),tables=deepcopy(schedules),
        clock='physical seconds; interval midpoint; split at knots',state='immutable inputs; no cursor',
        backend=a.backend)
