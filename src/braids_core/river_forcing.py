"""Named sediment forcing drafts and explicit current execution inputs.

Drafts retain text for inactive fractions/processes. Only active, validated
values become the existing ordered worker arrays; no numerical method changes.
"""
from copy import deepcopy
import math
from .river_boundary import FIELDS,FEED_FIELDS,feed_field


def validate_draft(value):
    if (not isinstance(value,dict) or set(value)!={'schema','boundaries','initial'}
        or type(value['schema']) is not int or value['schema']!=2):
        raise ValueError('Current sediment forcing editor schema required')
    if not isinstance(value['boundaries'],dict) or not isinstance(value['initial'],dict):
        raise ValueError('Named forcing drafts required')
    def names(values):
        if any(type(k) is not str or not k for k in values):
            raise ValueError('Explicit nonempty forcing identities required')
    names(value['boundaries']);names(value['initial'])
    if any(type(v) is not str for v in value['initial'].values()):
        raise ValueError('Initial forcing drafts must be text')
    for row in value['boundaries'].values():
        if not isinstance(row,dict) or set(row)!={'kind','fractions'} or row['kind'] not in FIELDS:
            raise ValueError('Explicit bedload boundary method required')
        if not isinstance(row['fractions'],dict):raise ValueError('Named fraction drafts required')
        names(row['fractions'])
        for v in row['fractions'].values():
            if (not isinstance(v,dict) or set(v)!=set(FEED_FIELDS)|{'concentration_kg_m3'}
                or any(type(x) is not str for x in v.values())):
                raise ValueError('Typed per-fraction forcing text required')


def nonnegative(text,path):
    try:value=float(text)
    except (ValueError,TypeError):raise ValueError(path+': enter a finite nonnegative number') from None
    if not math.isfinite(value) or value<0:
        raise ValueError(path+': enter a finite nonnegative number')
    return value


class ForcingDraft:
    def __init__(self,value=None):
        self.data=deepcopy(value) if value is not None else dict(schema=2,boundaries={},initial={})
        validate_draft(self.data)

    def ensure(self,ids,boundaries):
        if len(set(ids))!=len(ids) or any(type(k) is not str or not k for k in ids):
            raise ValueError('Give each sediment fraction a unique nonempty ID')
        for name,kind in boundaries.items():
            if kind not in ('inlet','outlet'):raise ValueError('Only open boundaries carry sediment forcing')
            row=self.data['boundaries'].setdefault(name,dict(kind='feed' if kind=='inlet' else 'outflow',fractions={}))
            for grain in ids:
                row['fractions'].setdefault(grain,{key:'0' for key in (*FEED_FIELDS,'concentration_kg_m3')})
        for grain in ids:self.data['initial'].setdefault(grain,'0')

    def active(self,ids,boundaries,*,sediment,suspended,inactive_boundaries=()):
        if suspended and not sediment:raise ValueError('Suspension requires enabled sediment')
        if not sediment:
            return {name:dict(kind='feed' if kind=='inlet' else 'outflow',
                **({'solid_flux_m2_s':[0.]*len(ids)} if kind=='inlet' else {}),
                concentration_kg_m3=[0.]*len(ids)) for name,kind in boundaries.items()},[0.]*len(ids)
        self.ensure(ids,boundaries);rows={}
        for name in boundaries:
            draft=self.data['boundaries'][name]
            row=dict(kind=draft['kind'],concentration_kg_m3=[nonnegative(
                draft['fractions'][g]['concentration_kg_m3'],f'{name} / {g} incoming concentration')
                if suspended and name not in inactive_boundaries else 0. for g in ids])
            key=feed_field(draft['kind'])
            if key:
                row[key]=[nonnegative(draft['fractions'][g][key],
                    f'{name} / {g} '+('solid-volume feed' if key=='solid_flux_m2_s' else key))
                    if name not in inactive_boundaries else 0. for g in ids]
            rows[name]=row
        initial=[nonnegative(self.data['initial'][g],g+' initial concentration') if suspended else 0. for g in ids]
        return rows,initial

    def load_active(self,project,*,inactive_boundaries=()):
        """Authoritative active inputs override stale drafts, never vice versa."""
        s=project['model']['physics']['sediment']
        if s is None:return
        ids=project['settings']['columns']['fraction_ids']
        bounds={a['bc_id']:a['kind'] for a in project['settings']['assignments'] if a['kind']!='wall'}
        self.ensure(ids,bounds)
        def put(dst,key,value):
            try:equal=nonnegative(dst[key],'draft')==value
            except ValueError:equal=False
            if not equal:dst[key]=str(value)
        for name,row in project['settings']['sediment_boundaries'].items():
            draft=self.data['boundaries'][name];draft['kind']=row['kind']
            if name in inactive_boundaries:continue
            for k,g in enumerate(ids):
                key=feed_field(row['kind'])
                if key:put(draft['fractions'][g],key,row[key][k])
                if s['suspension']!='off':put(draft['fractions'][g],'concentration_kg_m3',row['concentration_kg_m3'][k])
        if s['suspension']!='off':
            for k,g in enumerate(ids):put(self.data['initial'],g,project['initial']['concentration_kg_m3'][k])

    def snapshot(self):
        validate_draft(self.data)
        return deepcopy(self.data)
