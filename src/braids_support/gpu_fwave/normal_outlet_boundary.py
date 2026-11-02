"""Validation-only composition of frozen normal-inlet and critical-outlet bridges.

Private Python copies rename bookkeeping methods, not CUDA mathematics. Both
consume the identical native deck. The inherited parent reset executes once.
"""
from __future__ import annotations
import inspect
from pathlib import Path
from types import MappingProxyType
import numpy as np
from . import live_normal_boundary as normal
from . import outlet_boundary as outlet

HERE=Path(__file__).resolve().parent
require=normal.require
ANCESTORS={
    'live_normal_boundary.py':'de5cc0ed6c5ff35b72ebe408a2488d48e6fe83ee8e151d80cf86ecc23679fea4',
    'outlet_boundary.py':'257a1bdf54f46822304225c4b72ab257fe410cd86f3cb87c46a3087a801463df'}
PRIVATE_NAMES={'live_normal_boundary.py':'_normal_outlet_normal_bridge',
               'outlet_boundary.py':'_normal_outlet_outlet_bridge'}


def verify_ancestors():
    for name,value in ANCESTORS.items():
        require(normal.sha(HERE/name)==value,'Frozen bridge ancestor differs: '+name)
    return True


def replace_once(source,before,after):
    require(source.count(before)==1 and after not in source,'Absent/ambiguous combined-boundary anchor')
    return source.replace(before,after,1)


def _normal_validator():
    """Reuse the exact detached normal validator, explicitly admitting mode4."""
    verify_ancestors()
    source=inspect.getsource(normal.validate_sections)
    source=replace_once(source,'np.isin(mode,[2,3,6,8,9])','np.isin(mode,[2,3,4,6,8,9])')
    source=replace_once(source,'Only existing modes2/3/6/8 and NORMAL9 supported',
                        'Only existing modes2/3/6/8, critical OUTLET4 and NORMAL9 supported')
    namespace=dict(normal.__dict__)
    exec(compile(source,str(HERE/'normal_outlet_boundary.py')+'::normal_validator','exec'),namespace)
    return namespace['validate_sections']


_VALIDATE_NORMAL=_normal_validator()


def validate_boundaries(data):
    """Normal sampled/control1 and mode4 unsampled/control0, with real flags."""
    verify_ancestors();section=_VALIDATE_NORMAL(data)
    mode=np.asarray(data['mode']);selected=np.flatnonzero(mode==4).astype(np.int32)
    require(len(selected)>0,'Combined pipeline requires at least one critical OUTLET4')
    # The original normal validator already enforces control0 and no sampled
    # faces on every non-mode9 boundary. Absent seeds must also remain explicit.
    for key in ('section_seed_node','boundary_seed_position'):
        require(np.all(np.asarray(data[key])[mode!=9]==-1),'Nonnormal seed metadata must remain absent')
    raw_nc=np.asarray(data['nc'])
    require(raw_nc.shape==() and raw_nc.dtype.kind in 'iu','Canonical integer nc required')
    nc=int(raw_nc);require(np.asarray(data['q']).shape==(3,nc),'Hydraulic q/nc mismatch')
    offsets=np.asarray(data['boundary_offset']);faces=np.asarray(data['boundary_faces'])
    counts=np.array([int(offsets[b+1])-int(offsets[b]) for b in selected],dtype=np.int64)
    require(np.all(counts>0) and counts.sum()<2**31-1,'Nonempty int32-addressable outlet sections required')
    compact=np.concatenate([faces[offsets[b]:offsets[b+1]] for b in selected]).astype(np.int32)
    geometry=dict(nc=nc,owner=data['owner'],faces=compact,offsets=np.r_[0,np.cumsum(counts)].astype(np.int32),
                  length=data['length'],normal=data['normal'])
    host,count=outlet.component.validate_geometry(geometry)
    out=dict(geometry=MappingProxyType(dict(host,nc=count)),global_boundary_ids=outlet.freeze(selected),
             native_metadata=MappingProxyType({key:outlet.freeze(np.asarray(data[key])) for key in
                 ('mode','section_control','sampled','boundary_offset','boundary_faces','bid','neighbour')}))
    require(not np.intersect1d(section['geometry']['faces'],out['geometry']['faces']).size,
            'Normal and outlet faces overlap')
    return {'normal':section,'outlet':out}


def validate_sections(data):return validate_boundaries(data)['normal']
def validate_outlets(data):return validate_boundaries(data)['outlet']


def transform_bridge(name,source):
    verify_ancestors();require(name in PRIVATE_NAMES,'Unknown combined bridge ancestor')
    require(source==(HERE/name).read_text(),'Bridge source does not match frozen ancestor')
    is_normal=name=='live_normal_boundary.py'
    function=normal.validate_sections if is_normal else outlet.validate_outlets
    key='validate_sections' if is_normal else 'validate_outlets'
    source=replace_once(source,inspect.getsource(function),
        'def '+key+'(data):\n    from .normal_outlet_boundary import '+key+' as validate_actual\n    return validate_actual(data)\n')
    label='normal' if is_normal else 'outlet'
    source=source.replace('self.stage_records','self.'+label+'_records')
    methods=('check_storage','begin_trial','apply','record_stage','audit','receipt','bind_stage_state')
    for method in methods:
        new='_'+method+'_'+label
        source=replace_once(source,'        def '+method+'(', '        def '+new+'(')
        source=source.replace('self.'+method+'(', 'self.'+new+'(')
    if is_normal:
        source=replace_once(source,'super().apply(q,trace)','self._apply_outlet(q,trace)')
    else:
        source=source.replace('def _storage_identity(', 'def _outlet_storage_identity(')
        source=source.replace('self._storage_identity(', 'self._outlet_storage_identity(')
    return source


def make_boundary_controls(ParentControls,normal_module,outlet_module):
    """Compose independent bookkeeping; shared status arrays are reset once."""
    Outlet=outlet_module.make_boundary_controls(ParentControls)
    Normal=normal_module.make_boundary_controls(Outlet)
    class NormalOutletBoundary(Normal):
        def bind_stage_state(self,q):
            self._bind_stage_state_outlet(q);self._bind_stage_state_normal(q)
        def check_storage(self):
            self._check_storage_outlet();self._check_storage_normal();return True
        def begin_trial(self):
            self.check_storage();self._begin_trial_outlet();self._begin_trial_normal()
        def apply(self,q,trace):
            self.check_storage();return self._apply_normal(q,trace)
        def record_stage(self,index):
            self.check_storage();self._record_stage_normal(index);self._record_stage_outlet(index)
        def audit(self):
            self.check_storage();return dict(normal=self._audit_normal(),outlet=self._audit_outlet())
        def receipt(self):
            return dict(normal=self._receipt_normal(),outlet=self._receipt_outlet(),
                identical_native_data=True,surrogate_controls=False,parent_reset_count_per_apply=1,
                disjoint_trace_face_sets=True,separate_failure_latches=True,separate_stage_records=True,
                current_Euler_state=True,automatic_CPU_fallback=False,physical_validation=False)
    return NormalOutletBoundary
