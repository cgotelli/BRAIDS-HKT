"""Named bedload boundary contracts and one conversion to solid volume."""
from copy import deepcopy
import math

FLOW_SOLID='flow_prescribed_solid_volume'
FLOW_BULK='flow_prescribed_bulk_volume'
FLOW_MASS='flow_prescribed_mass'
FLOW_FREE='flow_free_equilibrium'
TOTAL='inlet_total_solid_discharge'
RATIO='inlet_solid_water_ratio'
NATIVE=(FLOW_SOLID,FLOW_BULK,FLOW_MASS)
FIELDS={'feed':'solid_flux_m2_s','outflow':None,FLOW_SOLID:'solid_flux_m2_s',
        FLOW_BULK:'bulk_flux_m2_s',FLOW_MASS:'mass_flux_kg_m_s',FLOW_FREE:None,
        TOTAL:'solid_discharge_m3_s',RATIO:'solid_water_ratio'}
FEED_FIELDS=('solid_flux_m2_s','bulk_flux_m2_s','mass_flux_kg_m_s','solid_discharge_m3_s','solid_water_ratio')


def feed_field(kind):
    if kind not in FIELDS:raise ValueError('Explicit named bedload boundary method required')
    return FIELDS[kind]


def factors(kind,columns):
    solid=columns['solid_density_kg_m3'];dry=columns['dry_density_kg_m3']
    if (len(solid)!=len(dry) or len(solid)!=len(columns['fraction_ids']) or
        any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in (*solid,*dry))):
        raise ValueError('Finite positive class solid and dry bulk densities required')
    return [d/s if kind==FLOW_BULK else 1./s if kind==FLOW_MASS else 1. for d,s in zip(dry,solid)]


def carrier_boundaries(rows,columns):
    """Private old-shaped mesh carrier, never an old-project import path."""
    if not isinstance(rows,dict):raise ValueError('Named sediment boundaries required')
    ng=len(columns['fraction_ids']);result={}
    for name,row in rows.items():
        if not isinstance(row,dict):raise ValueError(name+': boundary object required')
        kind=row.get('kind');key=feed_field(kind)
        expected={'kind','concentration_kg_m3'}|({key} if key else set())
        if set(row)!=expected:raise ValueError(name+': fields differ from selected bedload units/method')
        for field in expected-{'kind'}:
            if (not isinstance(row[field],list) or len(row[field])!=ng
                or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in row[field])):
                raise ValueError(name+': finite nonnegative per-fraction '+field+' required')
        converted=dict(kind='outflow' if key is None else 'feed',concentration_kg_m3=list(row['concentration_kg_m3']))
        if key:
            converted['solid_flux_m2_s']=[v*f for v,f in zip(row[key],factors(kind,columns))]
            if not all(math.isfinite(v) for v in converted['solid_flux_m2_s']):raise ValueError(name+': converted feed is nonfinite')
        result[name]=converted
    return result


def carrier_schedules(tables,rows,columns):
    result=deepcopy(tables)
    for name,table in result.items():
        kind=rows[name]['kind'];key=feed_field(kind)
        if table['method']=='forcing.constant' or key in (None,'solid_flux_m2_s'):continue
        table['solid_flux_m2_s']=[[v*f for v,f in zip(row,factors(kind,columns))] for row in table.pop(key)]
    return result


def install(solver,rows,data):
    """Native inlet policy before the common finite-stock limiter."""
    from pathlib import Path
    import numpy as np
    from .sediment.bend_methods_v1 import digest
    from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
    a=solver;p=a.physics;xp=a.xp
    mask=np.zeros(p.nf,np.int32);total_sections=[]
    for bid,name in enumerate(sorted(rows)):
        kind=rows[name]['kind']
        if kind in (*NATIVE,FLOW_FREE):
            mask[(data['neighbour']<0)&(data['bid']==bid)]=2 if kind==FLOW_FREE else 1
        if kind in ('outflow',TOTAL,RATIO):
            faces=(data['neighbour']<0)&(data['bid']==bid)
            mask[faces]={'outflow':3,TOTAL:1,RATIO:4}[kind]
            if kind==TOTAL:total_sections.append(xp.asarray(np.flatnonzero(faces)))
    if not mask.any():return
    path=Path(__file__).with_suffix('.h')
    engine=WaveEngine(a.backend,p.flow.threads,source=path,names=('river_prescribed_boundary',))
    resident=xp.asarray(mask);p.bedbc[resident!=0]=3 # signed computed transport before inflow replacement
    def apply(grain,water):
        if water is None:raise ValueError('Native transport boundary requires accepted hydraulic face flux')
        feed=a.feed[grain]
        if total_sections:
            feed=feed.copy()
            for faces in total_sections:
                incoming=xp.maximum(-water[faces],0.)
                total=xp.sum(incoming*p.g['length'][faces])
                feed[faces]=xp.where(total>0,feed[faces]*incoming/xp.maximum(total,1e-300),0.)
        engine.launch('river_prescribed_boundary',p.nf,p.nf,resident,p.hydraulic_face_active,water,
            feed,p.class_flux[grain])
    a._transport_boundary=apply
    a.identity['transport_boundary']=dict(source_sha256=digest(__file__),runtime=engine.identity,
        methods={name:row['kind'] for name,row in rows.items()},
        direction='accepted time-averaged water flux per true face; outward positive',
        zero_flow='native non-outward condition; prescribe only on hydraulically open faces',
        input='one class-density conversion in preparation/schedules; no extra surface fraction',
        export='outflow clamps negative transport and closes sediment on hydraulic inflow; same stock limiter and ledger',
        total='solid m3/s distributed by accepted incoming water flux; zero while no inward water',
        ratio='solid sediment volume / water volume times local accepted inward water discharge',
        free='signed reconstructed capacity; extrapolated interior bed and composition; no imposed feed or interior clamp')
