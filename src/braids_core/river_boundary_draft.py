"""Named boundary editor text; only explicit active projections can execute."""
from copy import deepcopy
from .river_open_boundary import (HYDRAULICS, PARAMETERS, BED_METHODS, COMPOSITION_METHODS,
    STAGE_POLICIES, INITIALS, new_state, resolve, exact, finite)
from .river_exterior_schedule import BED_FIELDS, COMPOSITION_FIELDS
from .river_schedule import CONSTANT, LINEAR


def slot(fields):
    return dict(values={k:PARAMETERS[k][1] for k in fields},
                schedule=dict(method=CONSTANT,rows=[]))


def new_row(role):
    return dict(hydraulic='inlet.total_discharge' if role=='inlet' else 'outlet.stage',
        controls={m:slot(v[2]) for m,v in HYDRAULICS.items() if v[1]==role},
        stage_policy='stage.outflow_only',secondary_intensity_m_s='0',bed='bed.free',
        beds={m:dict(slot([f] if f else []),initial='initial.prepared_bed',initial_elevation_m='0')
              for m,f in BED_FIELDS.items()},composition='composition.free',
        compositions={m:slot([]) for m in COMPOSITION_METHODS})


def numeric(text,label):
    try:value=float(text)
    except (TypeError,ValueError) as exc:raise ValueError(label+': finite numeric value required') from exc
    return finite(value,label)


def project_slot(source,fields,table_fields,label):
    """Tables replace their constant fields, without parsing inactive text."""
    schedule=source['schedule'];linear=schedule['method']==LINEAR
    values={k:('mesh' if k=='energy_slope' and source['values'][k]=='mesh' else
               numeric(source['values'][k],label+' '+k)) for k in fields if not linear or k not in table_fields}
    if not linear:return values,dict(method=CONSTANT)
    if not table_fields:raise ValueError(label+': this method has no time-table fields')
    if not schedule['rows']:raise ValueError(label+': add at least two time rows')
    result=dict(method=LINEAR,time_s=[numeric(r['time_s'],label+' time') for r in schedule['rows']])
    for key in table_fields:
        result[key]=[numeric(r['values'][key],label+' '+key) for r in schedule['rows']]
        values[key]=result[key][0]
    return values,result


class BoundaryDraft:
    def __init__(self,value=None):
        self.data=deepcopy(value) if value is not None else dict(schema=2,boundaries={})
        self.snapshot()

    def ensure(self,ids,kinds):
        for name,role in kinds.items():
            if role not in ('inlet','outlet'):raise ValueError('Open boundary role required')
            roles=self.data['boundaries'].setdefault(name,{})
            row=roles.setdefault(role,new_row(role))
            for method,target in row['compositions'].items():
                if COMPOSITION_FIELDS[method] is None:continue
                for g in ids:target['values'].setdefault(g,str(1./len(ids)))
                for record in target['schedule']['rows']:
                    for g in ids:record['values'].setdefault(g,'0')

    def snapshot(self):
        v=self.data;exact(v,('schema','boundaries'),'Boundary editor')
        if type(v['schema']) is not int or v['schema']!=2 or not isinstance(v['boundaries'],dict):
            raise ValueError('Current boundary editor format required; no migration')
        def text(value):
            if type(value) is not str:raise ValueError('Boundary draft values must be text')
        def table(value,keys,*,fractions=False,bed=False,table_keys=None):
            exact(value,('values','schedule',*(['initial','initial_elevation_m'] if bed else [])),'Boundary draft slot')
            if not isinstance(value['values'],dict):raise ValueError('Named boundary draft values required')
            if not fractions and set(value['values'])!=set(keys):raise ValueError('Boundary draft fields differ from method')
            for k,t in value['values'].items():
                if type(k) is not str or not k:raise ValueError('Named boundary draft field required')
                text(t)
            s=value['schedule'];exact(s,('method','rows'),'Boundary draft schedule')
            if s['method'] not in (CONSTANT,LINEAR) or not isinstance(s['rows'],list):raise ValueError('Invalid boundary draft schedule')
            for r in s['rows']:
                exact(r,('time_s','values'),'Boundary draft time row');text(r['time_s'])
                if not isinstance(r['values'],dict):raise ValueError('Named boundary time values required')
                if not fractions and set(r['values'])!=set(keys if table_keys is None else table_keys):
                    raise ValueError('Boundary time fields differ from method')
                if any(type(k) is not str or not k for k in r['values']):raise ValueError('Named boundary time field required')
                for t in r['values'].values():text(t)
            if bed:
                if value['initial'] not in INITIALS:raise ValueError('Named initial bed method required')
                text(value['initial_elevation_m'])
        for name,roles in v['boundaries'].items():
            if type(name) is not str or not name or not isinstance(roles,dict) or not set(roles)<= {'inlet','outlet'}:
                raise ValueError('Named inlet/outlet drafts required')
            for role,row in roles.items():
                exact(row,('hydraulic','controls','stage_policy','secondary_intensity_m_s','bed','beds','composition','compositions'),'Boundary draft')
                text(row['secondary_intensity_m_s'])
                methods={m for m,d in HYDRAULICS.items() if d[1]==role}
                exact(row['controls'],methods,'Hydraulic drafts')
                exact(row['beds'],BED_METHODS,'Bed drafts');exact(row['compositions'],COMPOSITION_METHODS,'Composition drafts')
                if row['hydraulic'] not in methods or row['stage_policy'] not in STAGE_POLICIES or row['bed'] not in BED_METHODS or row['composition'] not in COMPOSITION_METHODS:
                    raise ValueError('Explicit named boundary draft methods required')
                for m,s in row['controls'].items():table(s,HYDRAULICS[m][2],table_keys=HYDRAULICS[m][3])
                for m,s in row['beds'].items():table(s,[BED_FIELDS[m]] if BED_FIELDS[m] else [],bed=True)
                for m,s in row['compositions'].items():table(s,[],fractions=COMPOSITION_FIELDS[m] is not None)
        return deepcopy(v)

    def active(self,ids,kinds,*,sediment,end_time):
        self.ensure(ids,kinds);hydraulics={};states={}
        for name,role in kinds.items():
            row=self.data['boundaries'][name][role];method=row['hydraulic']
            values,schedule=project_slot(row['controls'][method],HYDRAULICS[method][2],HYDRAULICS[method][3],name)
            hydraulics[name]=dict(method=method,**values)
            state=new_state(method,sediment=sediment is not None)
            state['secondary_intensity_m_s']=numeric(row['secondary_intensity_m_s'],name+' incoming secondary intensity')
            state['hydraulic_schedule']=schedule
            if method=='outlet.stage':state['stage_policy']=row['stage_policy']
            if sediment is not None:
                m=row['bed'];s=row['beds'][m];f=BED_FIELDS[m];fields=[f] if f else []
                values,schedule=project_slot(s,fields,fields,name+' bed')
                initial=dict(method=s['initial'])
                if m=='bed.elevation':initial=dict(method='initial.elevation',elevation_m=values[f])
                elif s['initial']=='initial.elevation':initial['elevation_m']=numeric(s['initial_elevation_m'],name+' initial bed')
                state['bed']=dict(method=m,initial=initial,**values);state['bed_schedule']=schedule
                m=row['composition'];s=row['compositions'][m];fields=ids if COMPOSITION_FIELDS[m] else []
                values,schedule=project_slot(s,fields,fields,name+' composition')
                state['composition']=dict(method=m)
                if fields:state['composition']['fractions']=[values[g] for g in ids]
                if schedule['method']==LINEAR:
                    schedule=dict(method=LINEAR,time_s=schedule['time_s'],
                        fractions=[[schedule[g][i] for g in ids] for i in range(len(schedule['time_s']))])
                state['composition_schedule']=schedule
            states[name]=state
        resolved=resolve(hydraulics,states,kinds,sediment=sediment,ng=len(ids),end_time=end_time)
        return hydraulics,states,resolved

    def load_active(self,project):
        """Validated active values win over a saved, possibly stale UI draft."""
        hydraulics=project['settings']['hydraulics'];ids=project['settings']['columns']['fraction_ids']
        kinds={name:HYDRAULICS[r['method']][1] for name,r in hydraulics.items()};self.ensure(ids,kinds)
        def overlay(slot,values,schedule,fields):
            for k,v in values.items():slot['values'][k]=str(v)
            slot['schedule']['method']=schedule['method']
            if schedule['method']==CONSTANT:return
            old=slot['schedule']['rows'];rows=[]
            for i,t in enumerate(schedule['time_s']):
                r=deepcopy(old[i]) if i<len(old) else dict(time_s='',values={})
                r['time_s']=str(t)
                for k in fields:r['values'][k]=str(schedule[k][i])
                rows.append(r)
            slot['schedule']['rows']=rows
        for name,control in hydraulics.items():
            row=self.data['boundaries'][name][kinds[name]];state=project['boundary_states'][name]
            m=control['method'];row['hydraulic']=m
            row['secondary_intensity_m_s']=str(state['secondary_intensity_m_s'])
            overlay(row['controls'][m],{k:control[k] for k in HYDRAULICS[m][2]},state['hydraulic_schedule'],HYDRAULICS[m][3])
            if m=='outlet.stage':row['stage_policy']=state['stage_policy']
            if project['model']['physics']['sediment'] is None:continue
            b=state['bed'];m=b['method'];row['bed']=m;s=row['beds'][m];f=BED_FIELDS[m]
            s['initial']=b['initial']['method']
            if s['initial']=='initial.elevation':s['initial_elevation_m']=str(b['initial']['elevation_m'])
            overlay(s,{f:b[f]} if f else {},state['bed_schedule'],[f] if f else [])
            c=state['composition'];m=c['method'];row['composition']=m;fields=ids if COMPOSITION_FIELDS[m] else []
            schedule=state['composition_schedule']
            if schedule['method']==LINEAR:
                schedule=dict(method=LINEAR,time_s=schedule['time_s'],
                              **{g:[r[k] for r in schedule['fractions']] for k,g in enumerate(ids)})
            overlay(row['compositions'][m],dict(zip(ids,c['fractions'])) if fields else {},schedule,fields)
