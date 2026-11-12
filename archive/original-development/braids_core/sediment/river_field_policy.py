"""Field-driven common-clock policy for RiverLeafController.

Dynamic adaptation uses water depth/halo only. Indicators run on the chosen
backend; only integer tags go to host topology planning. No solver substitution.
The outer checkpoint includes coarsening memory and the next decision time.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib
import json
import tempfile
import time
import numpy as np
from .bed_columns_v101 import host,array_sha
from .river_leaf_controller import RiverLeafController
from .river_native_lifecycle import release_loaded_search_directories
from . import river_sediment_clock
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
from braids_gpu.atomic_output import arrays_to_bytes,write_bytes_atomic

SCOPE='river-exclusive-mesh-policies-common-clock-v4'
CAUSES={1:'wet'}
# Exact predecessor files, before failure-cap checkpoint support. Both source
# and packaged forms differ only in the established validation import rewrite.
LEGACY_FIELD_HASHES={
    '4c215303f82512d28e47d9d794d134cb00c699d58380c9e6a7286a76121dcf77',
    'bde8d175361ba1ad00edb33a6728c1d200d24aa38a210ef5c7ee80a25b69893e',
}


@dataclass(frozen=True)
class FieldSettings:
    interval:float=1.
    coarsen_delay:int=2
    buffer_cells:int=2
    transition_layers:int=1
    terrain_enabled:bool=False
    domain_border_enabled:bool=False
    open_boundary_enabled:bool=True
    terrain_angle_deg:float=5.
    terrain_detail_m:float=.1

    def __post_init__(self):
        for key,value in asdict(self).items():
            if key in ('terrain_enabled','domain_border_enabled','open_boundary_enabled'):
                if type(value) is not bool:raise ValueError('Explicit boolean refinement switch required')
                continue
            if isinstance(value,bool) or not np.isfinite(value) or value<=0:raise ValueError('Positive finite field policy settings required')
        if self.terrain_angle_deg>=90:raise ValueError('Terrain slope-change angle must be below 90 degrees')
        for key in ('coarsen_delay','buffer_cells','transition_layers'):
            if not isinstance(getattr(self,key),int):raise ValueError('Integer delay and buffer required')
        if self.transition_layers>32:raise ValueError('Transition layers must be from 1 to 32')


def refinement_leaves(tree,requested,settings,required=()):
    """Grade physical requests, including open sections and surveyed corridors."""
    wanted=np.asarray(requested,dtype=int).copy();base=tree.offsets[-2]
    if any(type(n) is not int or not base<=n<tree.count for n in required):
        raise ValueError('Refinement protection requires finest hierarchy node IDs')
    if settings.open_boundary_enabled:
        for face in tree.finest.faces:
            if face.neighbour<0 and face.kind in ('inlet','outlet'):wanted[base+face.owner]=tree.finest_level
    wanted[list(required)]=tree.finest_level
    leaves=tree.balanced_leaves(wanted,transition_layers=settings.transition_layers)
    cover=tree.leaf_cover(leaves)
    if (settings.open_boundary_enabled and any(cover[base+f.owner]!=base+f.owner for f in tree.finest.faces
           if f.neighbour<0 and f.kind in ('inlet','outlet'))) or not set(required).issubset(leaves):
        raise ValueError('Hierarchy refinement cap prevents mandatory finest protection')
    return leaves


class FieldIndicators:
    def __init__(self,backend,threads=4):
        if backend not in ('cpu','gpu'):raise ValueError('Explicit CPU/GPU required')
        self.backend=backend;self.xp=np
        if backend=='gpu':
            import cupy as cp
            self.xp=cp
        self.engine=WaveEngine(backend,threads,source=Path(__file__).with_suffix('.h'),
            names=('depth_tags','depth_buffer'))
        release_loaded_search_directories(self.engine)
        self.identity=dict(self.engine.identity,loader_sha256=hashlib.sha256(Path(__file__).with_name('river_native_lifecycle.py').read_bytes()).hexdigest());self.cache={}

    def geometry(self,tree,leaves):
        key=tuple(leaves)
        if key in self.cache:return self.cache[key]
        mesh=tree.select(leaves);n=len(leaves);rows=[set() for _ in leaves]
        for f in mesh.faces:
            if f.neighbour>=0:rows[f.owner].add(f.neighbour);rows[f.neighbour].add(f.owner)
        off=np.r_[0,np.cumsum([len(r) for r in rows])].astype(np.int32)
        adj=np.array([j for r in rows for j in sorted(r)],np.int32)
        xp=self.xp
        # Open boundaries can introduce water between two scheduled assessments,
        # including dry start and reversal. Protect their entrance before that
        # happens; walls alone are not wet-front seeds.
        entrance=np.array(sorted({f.owner for f in mesh.faces if f.neighbour<0 and f.kind!='wall'}),np.int32)
        g=dict(off=xp.asarray(off),adj=xp.asarray(adj),entrance=xp.asarray(entrance))
        self.cache[key]=g;return g

    def evaluate(self,controller,reference,previous,settings):
        a=controller.owner;p=a.physics;xp=self.xp;t=controller.hierarchy;leaves=controller.leaves
        if p.engine.backend!=self.backend:raise ValueError('Field policy backend differs')
        n=len(leaves);g=self.geometry(t,leaves);bits=xp.empty(n,dtype=xp.int32)
        self.engine.launch('depth_tags',n,n,float(p.parameters.dry),a.q,bits)
        active=bits.copy()
        if settings.open_boundary_enabled:active[g['entrance']]=1
        other=xp.empty_like(active)
        # Fixed-count parallel dilation. At coarse/fine interfaces a ring may
        # be wider than one finest cell: conservative protection, never thinner.
        # Terrain pins are computed separately once, not in this loop. No GPU
        # convergence synchronization inside the buffer loop.
        for _ in range(settings.buffer_cells):
            self.engine.launch('depth_buffer',n,n,g['off'],g['adj'],active,other)
            active,other=other,active
        return host(bits),host(active).astype(bool),previous,settings.buffer_cells


class AutomaticRiver:
    def __init__(self,controller,reference_bed,settings=FieldSettings()):
        self.controller=controller;self.settings=settings
        release_loaded_search_directories(controller.owner)
        self.reference=np.asarray(reference_bed,float).copy()
        if self.reference.shape!=(controller.hierarchy.count,) or not np.isfinite(self.reference).all():raise ValueError('Prepared hierarchy bed required')
        a=controller.owner;self.indicators=FieldIndicators(a.physics.engine.backend,a.physics.flow.threads)
        from .river_terrain_refinement import terrain_features
        self.terrain=terrain_features(controller.hierarchy,self.reference[controller.hierarchy.offsets[-2]:],
            settings,backend=a.physics.engine.backend,threads=a.physics.flow.threads)
        self.indicators.identity['terrain']=self.terrain['identity']
        self.delay=np.zeros(controller.hierarchy.count,np.int32)
        self.previous=self.indicators.xp.zeros(controller.hierarchy.count)
        # Initialize the reference at the actual accepted state, including warm-up.
        _,_,self.previous,_=self.indicators.evaluate(controller,self.reference,self.previous,settings)
        self.next_time=a.time;self.events=[];self.sediment_step_hint=None

    def assess(self,factory,*,cancelled=lambda:False,reject_after_candidate=False,force=False):
        c=self.controller;a=c.owner;t=c.hierarchy
        if a.columns.pending is not None or a.physics._open:raise ValueError('Field adaptation requires accepted timestep')
        if a.time+1e-12<self.next_time and not force:return None
        if cancelled():raise InterruptedError('Field adaptation cancelled')
        began=time.perf_counter();bits,active,history,iterations=self.indicators.evaluate(c,self.reference,self.previous,self.settings)
        raw=np.zeros(t.count,np.int32);raw[np.asarray(c.leaves)[active]]=t.finest_level
        raw[self.terrain['nodes']]=t.finest_level
        if self.settings.transition_layers>1:
            graded=refinement_leaves(t,raw,self.settings,c.survey_leaves)
            raw.fill(0);raw[list(graded)]=t.node_level[list(graded)]
        # Temporary coarsening holds are NOT new physical refinement seeds:
        # otherwise their bands would expand farther into dry terrain each time.
        delay=self.delay.copy();ids=np.asarray(c.leaves);level=t.node_level[ids]
        delay[ids]=np.where(raw[ids]<level,np.minimum(delay[ids]+1,self.settings.coarsen_delay),0)
        hold=ids[(raw[ids]<level)&(delay[ids]<self.settings.coarsen_delay)];raw[hold]=t.node_level[hold]
        requested=t.balanced_leaves(raw);tag_seconds=time.perf_counter()-began
        if not set(self.terrain['nodes']).issubset(requested):
            raise ValueError('Hierarchy refinement cap prevents finest terrain protection')
        def prepared(leaves,mesh):
            candidate=factory(leaves,mesh)
            release_loaded_search_directories(candidate)
            return candidate
        event=c.adapt(requested,prepared,cancelled=cancelled,reject_after_candidate=reject_after_candidate)
        # Nothing persistent in the policy changes before atomic owner publication.
        # Reactivated children/parents must not inherit quiet counts from an old
        # epoch. The delay counts consecutive quiet observations while active.
        delay[np.setdiff1d(np.asarray(c.leaves),ids,assume_unique=True)]=0
        self.delay=delay;self.previous=history
        self.next_time=(np.floor((a.time+1e-12)/self.settings.interval)+1)*self.settings.interval
        event.update(causes={name:int(np.count_nonzero(bits&bit)) for bit,name in CAUSES.items()},
            terrain_detected_cells=self.terrain['identity']['detected_cells'],
            terrain_pinned_cells=len(self.terrain['nodes']),
            directly_tagged=int(np.count_nonzero(bits)),buffered=int(np.count_nonzero(active)),buffer_iterations=iterations,
            indicator_planning_s=tag_seconds,adaptation_total_s=time.perf_counter()-began,next_decision_s=self.next_time,
            active_levels=np.bincount(t.node_level[list(c.leaves)],minlength=t.levels).tolist())
        self.events.append(event);return event

    def checkpoint_bytes(self):
        payload=self.controller.checkpoint_bytes()
        cap=getattr(self.sediment_step_hint,'failure_cap',None)
        meta=dict(scope=SCOPE,settings=asdict(self.settings),next_time=self.next_time,events=self.events,
            turbine_low_speed_warning=getattr(self,'turbine_low_speed_warning',None),
            sediment_clock=dict(method=river_sediment_clock.METHOD,
                code_sha256=hashlib.sha256(Path(river_sediment_clock.__file__).read_bytes()).hexdigest(),
                hint_s=self.sediment_step_hint,failure_cap_s=cap),
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runtime=self.indicators.identity,
            history_sha256=array_sha(self.reference,self.delay,host(self.previous),
                np.array([self.sediment_step_hint or 0.,cap or 0.])),
            payload_sha256=hashlib.sha256(payload).hexdigest())
        return arrays_to_bytes(metadata=np.array(json.dumps(meta,sort_keys=True)),payload=np.frombuffer(payload,np.uint8),
            reference=self.reference,delay=self.delay,previous=host(self.previous))

    def checkpoint(self,path):
        write_bytes_atomic(path,self.checkpoint_bytes())

    @classmethod
    def restart(cls,path,hierarchy,factory,reference_bed,settings):
        with np.load(path,allow_pickle=False) as f:
            if set(f.files)!={'metadata','payload','reference','delay','previous'}:raise ValueError('Field checkpoint schema differs')
            m=json.loads(str(f['metadata']));payload=f['payload'].tobytes();reference=f['reference'];delay=f['delay'];previous=f['previous']
        current_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        legacy=m.get('code_sha256') in LEGACY_FIELD_HASHES
        if m['scope']!=SCOPE or (m.get('code_sha256')!=current_hash and not legacy):raise ValueError('Field checkpoint code differs')
        if m['settings']!=asdict(settings) or not np.array_equal(reference,reference_bed):raise ValueError('Field checkpoint policy differs')
        if m['payload_sha256']!=hashlib.sha256(payload).hexdigest():raise ValueError('Field checkpoint payload differs')
        clock=m['sediment_clock'];hint=clock['hint_s']
        current_clock=(clock.get('method')==river_sediment_clock.METHOD and
                clock.get('code_sha256')==hashlib.sha256(Path(river_sediment_clock.__file__).read_bytes()).hexdigest())
        legacy_clock=(legacy and clock.get('method')==river_sediment_clock.LEGACY_METHOD and
                clock.get('code_sha256')==river_sediment_clock.LEGACY_SHA256 and 'failure_cap_s' not in clock)
        if not ((not legacy and current_clock) or legacy_clock):
            raise ValueError('Sediment step controller identity differs')
        cap=None if legacy_clock else clock.get('failure_cap_s')
        restored_hint=river_sediment_clock.restore_hint(hint,cap)
        history=[hint or 0.] if legacy_clock else [hint or 0.,cap or 0.]
        if m['history_sha256']!=array_sha(reference,delay,previous,np.array(history)):raise ValueError('Field checkpoint history checksum differs')
        if delay.shape!=(hierarchy.count,) or previous.shape!=delay.shape or not np.isfinite(previous).all() or np.any(delay<0):raise ValueError('Field checkpoint history differs')
        with tempfile.TemporaryDirectory(prefix='braids-fields-') as tmp:
            p=Path(tmp)/'leaves.npz';p.write_bytes(payload);c=RiverLeafController.restart(p,hierarchy,factory)
        result=cls(c,reference,settings)
        if m['runtime']!=result.indicators.identity:raise ValueError('Field checkpoint runtime differs')
        if not np.isfinite(m['next_time']) or not c.owner.time-1e-12<=m['next_time']<=c.owner.time+settings.interval+1e-12:raise ValueError('Field checkpoint decision clock differs')
        result.delay=delay.copy();result.previous=result.indicators.xp.asarray(previous);result.next_time=float(m['next_time']);result.events=m['events']
        result.sediment_step_hint=restored_hint
        warning=m.get('turbine_low_speed_warning')
        if warning is not None:
            if (not isinstance(warning,dict) or set(warning)!={'first_time_s','turbine_ids'} or
                type(warning['first_time_s']) not in (int,float) or not np.isfinite(warning['first_time_s']) or
                not 0<=warning['first_time_s']<=c.owner.time or
                not isinstance(warning['turbine_ids'],list) or not warning['turbine_ids'] or
                any(not isinstance(name,str) or not name for name in warning['turbine_ids'])):
                raise ValueError('Invalid turbine low-speed checkpoint diagnostic')
        result.turbine_low_speed_warning=warning
        return result


class FixedRiver(AutomaticRiver):
    """Checkpoint-compatible fixed mesh; no indicator kernels or runtime tags.

    Manual constraints or slope breaks selected the leaves during preparation. A fixed
    run must not instantiate/evaluate FieldIndicators or rescan terrain, even
    on restart. The actual mesh and all physical arrays remain authenticated by
    RiverLeafController's payload. No automatic switch to dynamic refinement.
    """
    def __init__(self,controller,reference_bed,settings):
        from types import SimpleNamespace
        self.controller=controller;self.settings=settings
        self.reference=np.asarray(reference_bed,float).copy()
        if self.reference.shape!=(controller.hierarchy.count,) or not np.isfinite(self.reference).all():
            raise ValueError('Prepared hierarchy bed required')
        self.indicators=SimpleNamespace(xp=np,identity=dict(method=('fixed_initial_terrain_mesh_v1' if settings.terrain_enabled else 'fixed_initial_manual_mesh_v1'),
            leaves_sha256=array_sha(np.asarray(controller.leaves,dtype=np.int64))))
        self.delay=np.zeros(controller.hierarchy.count,np.int32)
        self.previous=np.zeros(controller.hierarchy.count)
        self.next_time=controller.owner.time;self.events=[];self.sediment_step_hint=None

    def assess(self,*args,**kwargs):
        raise RuntimeError('Static mesh must never enter an AMR assessment')
