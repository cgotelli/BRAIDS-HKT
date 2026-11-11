"""Accepted-step common-clock adaptation of the current combined river owner.

The caller proposes hierarchy leaves (an indicator policy is separate) and
provides a factory for fresh mesh-specific owners. Publication is atomic;
inlet/outlet finest protection is optional. Mixed wet/dry groups retain current resolution.
Walls alone do not impose refinement; cropped wall cells may remain coarse.
No recursive subcycling, automatic backend choice or physics fallback.
"""
from pathlib import Path
import copy
import hashlib
import json
from dataclasses import fields
from functools import lru_cache
from operator import attrgetter
import tempfile
import numpy as np
from .bed_columns_v101 import host
from .river_amr_transfer import CellMap,owner_polygons,transfer
from braids_gpu.atomic_output import arrays_to_bytes,write_bytes_atomic

SCOPE='accepted-step-river-leaves-common-clock-v4-open-boundaries'
DYNAMIC_KEYS=('river_regrid','river_dynamic_leaf')


def canonical(value):return json.loads(json.dumps(value,sort_keys=True))


@lru_cache(maxsize=16)
def _geometry_values(kind,excluded=()):
    # Capture values, not object references to mutable dataclass dictionaries.
    return attrgetter(*(f.name for f in fields(kind) if f.name not in excluded))


def _geometry_snapshot(hierarchy,mesh):
    """Exact in-memory construction guard; derived metric caches are not inputs."""
    def values(obj,excluded=()):return _geometry_values(type(obj),excluded)(obj)
    def selected(m):
        cells=_geometry_values(type(m.cells[0]))
        faces=_geometry_values(type(m.faces[0]))
        return (values(m,('cells','faces','grid','_metrics_cache')),values(m.grid),
                tuple(cells(c) for c in m.cells),tuple(faces(f) for f in m.faces))
    def array(a):return a.dtype.str,a.shape,a.tobytes()
    return (values(hierarchy,('meshes','parent','node_level','maximum_level')),
            array(hierarchy.parent),array(hierarchy.node_level),array(hierarchy.maximum_level),
            tuple(selected(m) for m in hierarchy.meshes),selected(mesh))


class RiverLeafController:
    def __init__(self,hierarchy,leaves,owner):
        leaves=tuple(sorted(leaves))
        self._initialize(hierarchy,leaves,owner,hierarchy.select(leaves))

    @classmethod
    def from_factory(cls,hierarchy,leaves,factory,*,cancelled=lambda:False):
        """Select once, then retain that verified mesh across owner construction.

        No caller-supplied mesh/approval token is accepted. The factory may
        read the selected mesh, but must not mutate it or its hierarchy. A
        exact value snapshot covers geometry fields and hierarchy arrays, not
        merely polygons or object IDs. Derived mesh metric caches may populate
        during construction; they are not geometry inputs. No serialization,
        persisted approval flag or caller-supplied cached mesh is trusted.
        """
        def stop():
            if cancelled():raise InterruptedError('Initial mesh construction cancelled')
        stop();leaves=tuple(sorted(leaves));mesh=hierarchy.select(leaves)
        before=_geometry_snapshot(hierarchy,mesh);stop()
        owner=factory(leaves,mesh);stop()
        if _geometry_snapshot(hierarchy,mesh)!=before:
            raise ValueError('Selected mesh or hierarchy changed during owner construction')
        controller=cls.__new__(cls)
        controller._initialize(hierarchy,leaves,owner,mesh)
        stop()
        return controller

    def _initialize(self,hierarchy,leaves,owner,mesh):
        self.hierarchy=hierarchy;self.leaves=tuple(sorted(leaves));self.owner=owner
        self.open_boundary_refinement=getattr(owner,'_open_boundary_refinement',True)
        if type(self.open_boundary_refinement) is not bool:raise ValueError('Explicit inlet/outlet refinement switch required')
        if self.open_boundary_refinement and any(f.neighbour<0 and f.kind in ('inlet','outlet') and
               mesh.cells[f.owner].level!=hierarchy.finest_level for f in mesh.faces):
            raise ValueError('Inlet/outlet cells must be finest leaves')
        if any(f.breakline_ids and
               (mesh.cells[f.owner].level!=hierarchy.finest_level or
                (f.neighbour>=0 and mesh.cells[f.neighbour].level!=hierarchy.finest_level)) for f in mesh.faces):
            raise ValueError('Both sides of a breakline must be finest leaves')
        m=CellMap.polygons([c.vertices_xy for c in mesh.cells],owner_polygons(owner),
            [c.area_m2 for c in mesh.cells],host(owner.columns.area))
        if len(m.source_area)!=len(m.target_area) or not np.array_equal(m.indices,np.arange(len(self.leaves))) or not np.all(np.diff(m.offsets)==1):
            raise ValueError('Leaf IDs do not match the solver cell order')
        self.geometry_sha256=hashlib.sha256(b''.join(np.asarray(c.vertices_xy,float).tobytes() for c in hierarchy.cells)).hexdigest()
        # Geometry is immutable during a run. Keep maps in RAM when a moving
        # patch revisits a leaf pattern; physical states are NEVER cached here.
        self._mesh_cache={self.leaves:mesh}
        self._map_cache={}
        self._pinned_cache={}
        self.survey_leaves=tuple(getattr(owner,'_survey_required_leaves',()))
        if any(type(v) is not int or v not in self.leaves or
               hierarchy.node_level[v]!=hierarchy.finest_level for v in self.survey_leaves):
            raise ValueError('Survey corridor requires explicit finest current leaves')

    def _geometry_map(self,leaves):
        key=(self.leaves,tuple(leaves))
        if leaves not in self._mesh_cache:self._mesh_cache[leaves]=self.hierarchy.select(leaves)
        mesh=self._mesh_cache[leaves]
        if key not in self._map_cache:
            self._map_cache[key]=CellMap.hierarchy(self.hierarchy,self.leaves,leaves)
        return mesh,self._map_cache[key]

    @staticmethod
    def pin_boundaries(hierarchy,leaves,required=(),*,refine_open_boundaries=True):
        hierarchy.leaf_cover(leaves)
        if type(refine_open_boundaries) is not bool:raise ValueError('Explicit inlet/outlet refinement switch required')
        if any(type(v) is not int or v<0 or v>=hierarchy.count or
               hierarchy.node_level[v]!=hierarchy.finest_level for v in required):
            raise ValueError('Survey protection requires explicit finest hierarchy node IDs')
        wanted=np.zeros(hierarchy.count,dtype=int)
        wanted[list(leaves)]=hierarchy.node_level[list(leaves)]
        base=hierarchy.offsets[-2]
        if refine_open_boundaries:
            for f in hierarchy.finest.faces:
                if f.neighbour<0 and f.kind in ('inlet','outlet'):wanted[base+f.owner]=hierarchy.finest_level
        wanted[list(required)]=hierarchy.finest_level
        result=hierarchy.balanced_leaves(wanted)
        cover=hierarchy.leaf_cover(result)
        if refine_open_boundaries and any(cover[base+f.owner]!=base+f.owner for f in hierarchy.finest.faces
               if f.neighbour<0 and f.kind in ('inlet','outlet')):
            raise ValueError('Hierarchy refinement cap prevents finest inlet/outlet cells')
        if not set(required).issubset(result):
            raise ValueError('Hierarchy refinement cap prevents finest surveyed bank corridor')
        return result

    def plan(self,requested):
        tree=self.hierarchy;requested=tuple(requested)
        if requested not in self._pinned_cache:
            self._pinned_cache[requested]=self.pin_boundaries(tree,requested,self.survey_leaves,
                refine_open_boundaries=self.open_boundary_refinement)
        leaves=self._pinned_cache[requested]
        q=None;dry=self.owner.physics.parameters.dry;retained=set()
        for _ in range(tree.levels+1):
            mesh,m=self._geometry_map(leaves)
            unsafe=[]
            for j in np.flatnonzero(np.diff(m.offsets)>1):
                if q is None:q=host(self.owner.q)
                sl=slice(m.offsets[j],m.offsets[j+1]);ids=m.indices[sl];h=q[0,ids]
                mean=float(h@m.weights[sl])
                if (0<mean<=dry) or (np.any(h<=dry) and np.any(h>dry)):
                    unsafe.extend(self.leaves[i] for i in ids);retained.add(leaves[j])
            if not unsafe:return leaves,mesh,m,sorted(retained)
            wanted=np.zeros(tree.count,dtype=int)
            wanted[list(leaves)]=tree.node_level[list(leaves)]
            wanted[unsafe]=tree.node_level[unsafe]
            leaves=self.pin_boundaries(tree,tree.balanced_leaves(wanted),self.survey_leaves,
                refine_open_boundaries=self.open_boundary_refinement)
        raise ValueError('Wet/dry coarsening guard did not converge')

    def adapt(self,requested,factory,*,cancelled=lambda:False,reject_after_candidate=False):
        a=self.owner
        if a.columns.pending is not None or a.physics._open:raise ValueError('Adaptation requires an accepted timestep')
        if cancelled():raise InterruptedError('Adaptation cancelled')
        leaves,mesh,m,retained=self.plan(requested)
        event=dict(scope=SCOPE,from_cells=len(self.leaves),to_cells=len(leaves),time_s=a.time,
            accepted_updates=a.steps,retained_wetdry_groups=retained,
            refine_open_boundaries=self.open_boundary_refinement,
            open_boundaries_finest=all(mesh.cells[f.owner].level==self.hierarchy.finest_level
                for f in mesh.faces if f.neighbour<0 and f.kind in ('inlet','outlet')),
            changed=leaves!=self.leaves,leaves=list(leaves),wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if self.survey_leaves:event['finest_survey_corridor_leaves']=list(self.survey_leaves)
        if leaves==self.leaves:return event
        b=factory(leaves,mesh)
        if getattr(b,'_open_boundary_refinement',True)!=self.open_boundary_refinement:
            raise ValueError('Fresh owner changed inlet/outlet refinement policy')
        if tuple(getattr(b,'_survey_required_leaves',()))!=self.survey_leaves:
            raise ValueError('Fresh owner changed surveyed corridor refinement')
        b,receipt=transfer(a,b,m,cancelled=cancelled,reject_after_candidate=reject_after_candidate,
            hierarchy_geometry=(self.hierarchy,self.leaves,leaves))
        if cancelled():raise InterruptedError('Adaptation cancelled before owner publication')
        event['transfer']=receipt
        b.identity['river_dynamic_leaf']=dict(scope=SCOPE,history=copy.deepcopy(a.identity.get('river_dynamic_leaf',{}).get('history',[]))+[event])
        self.owner=b;self.leaves=leaves
        return event

    def checkpoint_bytes(self):
        a=self.owner
        payload=a.checkpoint_bytes()
        meta=dict(scope=SCOPE,geometry_sha256=self.geometry_sha256,leaves=list(self.leaves),
            refine_open_boundaries=self.open_boundary_refinement,
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            transfer_sha256=hashlib.sha256(Path(__file__).with_name('river_amr_transfer.py').read_bytes()).hexdigest(),
            bank_transfer_sha256=hashlib.sha256(Path(__file__).with_name('river_bank_remap.py').read_bytes()).hexdigest(),
            bank_transfer_native_sha256=hashlib.sha256(Path(__file__).with_name('river_bank_remap.h').read_bytes()).hexdigest(),
            terrain_sha256=hashlib.sha256(Path(__file__).with_name('river_terrain_transfer.py').read_bytes()).hexdigest(),
            identity=a.identity,payload_sha256=hashlib.sha256(payload).hexdigest())
        return arrays_to_bytes(metadata=np.array(json.dumps(meta,sort_keys=True)),owner=np.frombuffer(payload,dtype=np.uint8))

    def checkpoint(self,path):
        write_bytes_atomic(path,self.checkpoint_bytes())

    @classmethod
    def restart(cls,path,hierarchy,factory):
        """Rebuild the current mesh only; authenticate base physics before load."""
        with np.load(path,allow_pickle=False) as f:
            if set(f.files)!={'metadata','owner'}:raise ValueError('Dynamic checkpoint schema differs')
            meta=json.loads(str(f['metadata']));payload=f['owner'].tobytes()
        for key,name in [('wrapper_sha256','river_leaf_controller.py'),('transfer_sha256','river_amr_transfer.py'),
                         ('bank_transfer_sha256','river_bank_remap.py'),('bank_transfer_native_sha256','river_bank_remap.h'),
                         ('terrain_sha256','river_terrain_transfer.py')]:
            if meta.get(key)!=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest():raise ValueError('Dynamic restart code identity differs')
        if meta['scope']!=SCOPE or meta['payload_sha256']!=hashlib.sha256(payload).hexdigest():raise ValueError('Dynamic restart payload identity differs')
        leaves=tuple(meta['leaves']);c=cls.from_factory(hierarchy,leaves,factory);a=c.owner
        if meta['refine_open_boundaries']!=c.open_boundary_refinement:raise ValueError('Dynamic restart inlet/outlet refinement differs')
        if meta['geometry_sha256']!=c.geometry_sha256:raise ValueError('Dynamic restart geometry differs')
        def base(d):return canonical({k:v for k,v in d.items() if k not in DYNAMIC_KEYS})
        if base(a.identity)!=base(meta['identity']):raise ValueError('Dynamic restart base identity differs')
        for key in DYNAMIC_KEYS:
            if key in meta['identity']:a.identity[key]=copy.deepcopy(meta['identity'][key])
        with tempfile.TemporaryDirectory(prefix='braids-leaves-') as tmp:
            p=Path(tmp)/'owner.npz';p.write_bytes(payload);a.restart(p)
        return c
