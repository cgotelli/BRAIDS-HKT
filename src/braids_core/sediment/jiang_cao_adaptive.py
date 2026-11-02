"""Accepted-state depth AMR and strict mesh-epoch restart for the JC research route.

No physical kernel changes, old-checkpoint migration, or Desktop admission.
Factories return fresh 203 owners from immutable prepared hierarchy material.
Only a fully validated private owner is published. Hashes detect corruption,
not malicious replacement of both data and checksums.
"""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
import copy, hashlib, io, json, os, tempfile, threading, zipfile
import numpy as np
from .bend_methods_v1 import digest
from .jiang_cao_checkpoint import (JiangCaoRestart, canonical, array_hash, encode,
    decode, _unique, _invalid_constant, MAX_METADATA)
from .jiang_cao_remesh import JiangCaoRemesh, transfer, validate, contract
from .river_field_policy import FieldSettings, FieldIndicators
from .river_leaf_controller import RiverLeafController, _geometry_snapshot
from .river_amr_transfer import CellMap
from .river_native_lifecycle import release_loaded_search_directories
from braids_core.windows_paths import filesystem_path

SCHEMA='braids.jiang_cao.adaptive_checkpoint204.v1'
MAX_BYTES=256*1024*1024
MAX_EVENTS=4096


def sha(value):return hashlib.sha256(value).hexdigest()


def metadata(raw):
    if len(raw)>MAX_METADATA:raise ValueError('Adaptive metadata exceeds bounded research limit')
    value=json.loads(raw.decode('utf-8'),object_pairs_hook=_unique,parse_constant=_invalid_constant)
    if not isinstance(value,dict):raise ValueError('Adaptive metadata must be an object')
    return value


def archive_read(payload):
    if len(payload)>MAX_BYTES:raise ValueError('Adaptive checkpoint exceeds bounded research limit')
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            names=[i.filename for i in z.infolist()]
            if len(names)!=len(set(names)) or len(names)>MAX_EVENTS+2:raise ValueError('Duplicate or excessive checkpoint members')
            if sum(i.file_size for i in z.infolist())>MAX_BYTES:raise ValueError('Oversized adaptive archive')
            if any(i.compress_type!=zipfile.ZIP_STORED or i.flag_bits&1 for i in z.infolist()):
                raise ValueError('Unsupported adaptive archive encoding')
            return {name:z.read(name) for name in names}
    except (zipfile.BadZipFile, OSError, EOFError) as exc:raise ValueError('Corrupt adaptive archive') from exc


def epoch_read(payload,owner):
    """Authenticate historical geometry and ledgers before allocating NPY arrays."""
    expected={k:owner.a.host(v) for k,v in owner.mesh.g.items()}
    expected.update(collapse_integral=np.zeros(owner.nf),secondary_fields=np.zeros((9,owner.n)),
                    dispersion_flux=np.zeros((2,owner.nf)))
    raw=archive_read(payload)
    if set(raw)!={k+'.npy' for k in (*expected,'header')}:raise ValueError('Epoch inventory differs')
    arrays={}
    for name,value in raw.items():
        k=name[:-4];stream=io.BytesIO(value)
        if np.lib.format.read_magic(stream)!=(1,0):raise ValueError('Epoch NPY version differs')
        shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream,max_header_size=512)
        valid=(dtype==np.uint8 and len(shape)==1 and 0<shape[0]<=MAX_METADATA) if k=='header' else (
            shape==expected[k].shape and dtype==expected[k].dtype)
        if not valid or fortran:raise ValueError('Epoch shape or dtype differs')
        offset=stream.tell();count=int(np.prod(shape))
        if len(value)-offset!=count*dtype.itemsize:raise ValueError('Epoch array length differs')
        arrays[k]=np.frombuffer(value,dtype=dtype,count=count,offset=offset).reshape(shape)
    h=metadata(arrays.pop('header').tobytes())
    if set(h)!={'identity','time_s','steps','gross_start_m3','gross_end_m3','last','arrays'}:
        raise ValueError('Epoch header differs')
    if canonical(h['identity'])!=canonical(owner.identity):raise ValueError('Epoch runtime/geometry identity differs')
    if h['arrays']!={k:array_hash(v) for k,v in arrays.items()}:raise ValueError('Epoch array checksum failed')
    for k in owner.mesh.g:
        if not np.array_equal(arrays[k],expected[k]):raise ValueError('Epoch geometry differs')
    if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite epoch array')
    gross=h['gross_end_m3']-h['gross_start_m3']
    if gross<0 or gross+1e-12<abs(arrays['collapse_integral']).sum():raise ValueError('Epoch collapse budget differs')
    if np.any(arrays['collapse_integral'][expected['neighbour']<0]):raise ValueError('Exterior epoch collapse')
    if h['steps'] and (h['last'].get('time_s')!=h['time_s'] or
        h['last'].get('cumulative_collapse_gross_solid_m3')!=h['gross_end_m3']):raise ValueError('Epoch diagnostics differ')
    return h


class JiangCaoAdaptive:
    def __init__(self,hierarchy,initial_leaves,factory,*,prepared_identity,settings=FieldSettings(),protected_finest=(),start_time_s=0.):
        if settings.terrain_enabled or settings.domain_border_enabled:
            raise ValueError('JC research AMR uses depth plus buffer only')
        if (not isinstance(protected_finest,tuple) or
            protected_finest!=tuple(sorted(set(protected_finest))) or
            any(type(i) is not int or i<hierarchy.offsets[-2] or i>=hierarchy.count for i in protected_finest)):
            raise ValueError('Protection requires sorted unique explicit finest node IDs')
        self.protected_finest=protected_finest
        self.hierarchy=hierarchy;self.initial_leaves=self._leaves(initial_leaves)
        self.leaves=self.initial_leaves;self.factory=factory;self.settings=settings
        self._lock=threading.Lock();self._geometry=self._geometry_hash()
        self._meshes={}
        self.owner=self._fresh(self.leaves)
        if not np.isfinite(start_time_s) or start_time_s<0:raise ValueError('Nonnegative initial clock required')
        self._origin_time=float(start_time_s)
        if start_time_s:
            from braids_core.river_state_initialization import set_clock
            set_clock(self.owner,start_time_s,jiang_cao=True)
        self._totals=self.owner._initial_totals
        self.indicators=FieldIndicators(self.owner.backend,self.owner.a.physics.flow.threads)
        self.identity=dict(schema=SCHEMA,method='depth-buffer-jc204',source=digest(__file__),
            geometry_sha256=self._geometry,prepared=copy.deepcopy(prepared_identity),
            origin=self.owner.identity,initial_leaves=list(self.initial_leaves),settings=asdict(settings),clock_origin_s=self._origin_time,
            protection=dict(method='explicit.initial_repose_patch205_v1' if protected_finest else 'none',
                            finest_nodes=list(protected_finest)),
            indicators=self.indicators.identity,planner=digest(Path(__file__).with_name('river_leaf_controller.py')),
            field_policy_python=digest(Path(__file__).with_name('river_field_policy.py')),
            transfer_python=digest(Path(__file__).with_name('river_amr_transfer.py')),
            transfer_native=digest(Path(__file__).with_name('river_amr_transfer.h')),
            terrain_python=digest(Path(__file__).with_name('river_terrain_transfer.py')),
            terrain_native=digest(Path(__file__).with_name('river_terrain_transfer.h')))
        self._identity=canonical(self.identity)
        self.delay=np.zeros(hierarchy.count,np.int32);self.next_time=self._origin_time+settings.interval
        self._events=();self._maps={}

    @property
    def events(self):return tuple(json.loads(e) for e in self._events)

    def _leaves(self,leaves):
        if not isinstance(leaves,(tuple,list)) or any(type(i) is not int for i in leaves):raise ValueError('Explicit integer leaf list required')
        leaves=tuple(leaves)
        if leaves!=tuple(sorted(set(leaves))):raise ValueError('Sorted unique leaves required')
        self.hierarchy.leaf_cover(leaves)
        if RiverLeafController.pin_boundaries(self.hierarchy,leaves,self.protected_finest)!=leaves:
            raise ValueError('Open boundaries and explicit process protection must be finest leaves')
        return leaves

    def _geometry_hash(self):return sha(repr(_geometry_snapshot(self.hierarchy,self.hierarchy.finest)).encode('utf-8'))

    def _fresh(self,leaves):
        mesh=self.hierarchy.select(leaves);before=_geometry_snapshot(self.hierarchy,mesh)
        owner=self.factory(leaves,mesh)
        # Regridding/restart may construct many private epochs in one process.
        # Windows does not GC AddDllDirectory registrations. All libraries are
        # eagerly loaded; release only their redundant compiler search cookies.
        release_loaded_search_directories(owner)
        if type(owner) is not JiangCaoRemesh or owner.time or owner.steps or owner.epochs:
            raise ValueError('Factory must supply fresh JC remesh owner')
        if _geometry_snapshot(self.hierarchy,mesh)!=before:raise ValueError('Factory changed prepared geometry')
        m=CellMap.polygons([c.vertices_xy for c in mesh.cells],owner.polygons,
            [c.area_m2 for c in mesh.cells],owner.a.host(owner.mesh.g['area']))
        if not np.array_equal(m.indices,np.arange(len(leaves))) or not np.all(np.diff(m.offsets)==1):
            raise ValueError('Selected leaves and solver cell order differ')
        self._meshes[tuple(leaves)]=mesh
        return owner

    def _guard(self):
        if (self._identity!=canonical(self.identity) or self._geometry_hash()!=self._geometry or
            asdict(self.settings)!=self.identity['settings'] or self.leaves!=self._leaves(self.leaves) or
            list(self.protected_finest)!=self.identity['protection']['finest_nodes']):
            raise ValueError('Live adaptive identity/geometry changed')
        validate(self.owner)

    def step(self,dt):
        # Adaptation is an independent accepted-state transaction. A failed
        # adaptation never undoes an already accepted physical timestep.
        with self._exclusive():
            if self.owner.time>=self.next_time-1e-12:raise ValueError('AMR assessment due before next step')
            if dt>self.next_time-self.owner.time+1e-12:raise ValueError('Step crosses AMR decision clock')
            return self.owner.step(dt)

    def _exclusive(self):
        from contextlib import contextmanager
        @contextmanager
        def lock():
            if not self._lock.acquire(False):raise RuntimeError('Adaptive transaction already active')
            try:yield
            finally:self._lock.release()
        return lock()

    def assess(self,*,check=lambda:None):
        with self._exclusive():
            a=self.owner;t=self.hierarchy
            if a.time<self.next_time-1e-12:return None
            if len(self._events)>=MAX_EVENTS:raise ValueError('Adaptive event limit reached')
            check();self._guard()
            # Reuse qualified depth tagging and wet/dry-safe planner. Its
            # hydraulic carrier is only a read-only view of the JC state.
            c=RiverLeafController.__new__(RiverLeafController)
            c.hierarchy=t;c.leaves=self.leaves;c.owner=a.a;c.survey_leaves=self.protected_finest
            c._mesh_cache=self._meshes;c._map_cache=self._maps;c._pinned_cache={}
            bits,active,_,_=self.indicators.evaluate(c,None,None,self.settings)
            raw=np.zeros(t.count,np.int32);ids=np.array(self.leaves);levels=t.node_level[ids]
            raw[ids[active]]=t.finest_level
            delay=self.delay.copy()
            delay[ids]=np.where(raw[ids]<levels,np.minimum(delay[ids]+1,self.settings.coarsen_delay),0)
            hold=ids[(raw[ids]<levels)&(delay[ids]<self.settings.coarsen_delay)];raw[hold]=t.node_level[hold]
            leaves,_,_,retained=c.plan(t.balanced_leaves(raw))
            event=dict(time_s=a.time,steps=a.steps,from_leaves=list(self.leaves),leaves=list(leaves),
                wet_cells=int(np.count_nonzero(bits)),buffered_cells=int(active.sum()),
                retained_wetdry_groups=retained,changed=leaves!=self.leaves)
            target=a
            if event['changed']:
                target=self._fresh(leaves);check()
                event['transfer']=transfer(a,target,check=check)
            delay[np.setdiff1d(leaves,ids)]=0
            next_time=self._origin_time+(np.floor((a.time-self._origin_time+1e-12)/self.settings.interval)+1)*self.settings.interval
            event['next_time_s']=next_time;serialized=canonical(event);check()
            self.owner=target;self.leaves=tuple(leaves);self.delay=delay;self.next_time=next_time
            self._events+=(serialized,)
            return copy.deepcopy(event)

    def _payload(self):
        self._guard();a=self.owner
        if (self.delay.shape!=(self.hierarchy.count,) or self.delay.dtype!=np.int32 or
            np.any((self.delay<0)|(self.delay>self.settings.coarsen_delay)) or
            self.next_time!=self._origin_time+(len(self._events)+1)*self.settings.interval or
            not self.next_time-self.settings.interval-1e-12<=a.time<=self.next_time+1e-12):
            raise ValueError('Invalid live adaptation memory/clock')
        changed=[e for e in self.events if e['changed']]
        if len(changed)!=len(a.epochs) or len(changed)!=len(a.receipts):raise ValueError('Live mesh epoch inventory differs')
        for e,r,epoch in zip(changed,a.receipts,a.epochs):
            if canonical(e['transfer'])!=canonical(r) or r['archived_epoch_sha256']!=sha(epoch):
                raise ValueError('Live mesh epoch checksum differs')
        h,arrays=a._snapshot();a._validate_payload(h,arrays)
        pieces={'accepted.npz':encode(h,arrays)}
        pieces.update({f'epoch_{i}.npz':e for i,e in enumerate(a.epochs)})
        meta=dict(schema=SCHEMA,identity=self.identity,leaves=list(self.leaves),
            next_time_s=self.next_time,delay=self.delay.tolist(),events=list(self.events),
            receipts=a.receipts,epoch_gross_start_m3=a._epoch_gross_start,
            members={k:sha(v) for k,v in pieces.items()})
        meta['content_sha256']=sha(canonical(meta))
        raw=canonical(meta)
        if len(raw)>MAX_METADATA:raise ValueError('Adaptive metadata exceeds bounded research limit')
        pieces['metadata.json']=raw
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as z:
            for k,v in pieces.items():z.writestr(k,v)
        payload=stream.getvalue()
        if len(payload)>MAX_BYTES:raise ValueError('Adaptive checkpoint exceeds bounded research limit')
        return payload

    def checkpoint(self,path,*,check=lambda:None):
        with self._exclusive():
            check();payload=self._payload();check();path=filesystem_path(path)
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():raise FileExistsError('Checkpoint already exists; choose a new name')
            temp=path.with_name('.'+uuid4().hex+'.partial')
            try:
                with temp.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
                check();os.link(temp,path)
            finally:
                try:temp.unlink(missing_ok=True)
                except OSError:pass
            return dict(path=str(path),sha256=sha(payload),bytes=len(payload),time_s=self.owner.time,epochs=len(self.owner.epochs))

    def restart(self,path,*,check=lambda:None):
        with self._exclusive():
            if self.owner.time!=self._origin_time or self.owner.steps or self._events:raise ValueError('Adaptive restart requires fresh controller at its declared initial clock')
            check();self._guard()
            with filesystem_path(path).open('rb') as f:payload=f.read(MAX_BYTES+1)
            pieces=archive_read(payload)
            if 'metadata.json' not in pieces:raise ValueError('Unsupported adaptive schema; no migration')
            m=metadata(pieces.pop('metadata.json'))
            expected=m.pop('content_sha256',None)
            if expected!=sha(canonical(m)):raise ValueError('Adaptive metadata checksum failed')
            required={'schema','identity','leaves','next_time_s','delay','events','receipts','epoch_gross_start_m3','members'}
            if set(m)!=required or m['schema']!=SCHEMA:raise ValueError('Unsupported adaptive schema; no migration')
            if canonical(m['identity'])!=self._identity:raise ValueError('Adaptive model/backend/input/runtime/policy differs')
            if m['members']!={k:sha(v) for k,v in pieces.items()}:raise ValueError('Adaptive member checksum failed')
            leaves=self._leaves(m['leaves']);events=m['events'];receipts=m['receipts']
            if not isinstance(events,list) or len(events)>MAX_EVENTS or not isinstance(receipts,list):raise ValueError('Invalid adaptive history')
            if set(pieces)!={'accepted.npz',*(f'epoch_{i}.npz' for i in range(len(receipts)))}:
                raise ValueError('Adaptive epoch inventory differs')
            owners={self.initial_leaves:self.owner}
            def fresh(ids):
                if ids not in owners:check();owners[ids]=self._fresh(ids)
                return owners[ids]
            # Never publish into the caller's origin, even when its leaf pattern
            # is revisited after coarsening or the file fails later checks.
            target=self._fresh(leaves);previous=self.initial_leaves;gross=0.;nepoch=0;lasttime=self._origin_time;laststeps=0
            for i,e in enumerate(events):
                base={'time_s','steps','from_leaves','leaves','wet_cells','buffered_cells','retained_wetdry_groups','changed','next_time_s'}
                if not isinstance(e,dict) or type(e.get('changed')) is not bool or set(e)!=base|({'transfer'} if e['changed'] else set()):
                    raise ValueError('Invalid adaptive event')
                nxt=self._leaves(e['leaves'])
                if tuple(e['from_leaves'])!=previous or e['changed']!=(nxt!=previous):raise ValueError('Broken adaptive leaf chain')
                expected_time=self._origin_time+(i+1)*self.settings.interval
                if (type(e['steps']) is not int or e['steps']<=laststeps or abs(e['time_s']-expected_time)>1e-10 or
                    abs(e['next_time_s']-(self._origin_time+(i+2)*self.settings.interval))>1e-10):raise ValueError('Invalid adaptive event clock')
                if e['changed']:
                    if nepoch>=len(receipts):raise ValueError('Missing transfer receipt')
                    old=fresh(previous);new=fresh(nxt);epoch=pieces[f'epoch_{nepoch}.npz'];eh=epoch_read(epoch,old)
                    r=e['transfer']
                    if nepoch>=len(receipts) or canonical(r)!=canonical(receipts[nepoch]):raise ValueError('Transfer receipt chain differs')
                    mapping=CellMap.polygons(old.polygons,new.polygons,old.a.host(old.mesh.g['area']),new.a.host(new.mesh.g['area']))
                    if (r.get('archived_epoch_sha256')!=sha(epoch) or r.get('mapping_sha256')!=mapping.identity or
                        r.get('time_s')!=e['time_s'] or r.get('steps')!=e['steps'] or
                        eh['time_s']!=e['time_s'] or eh['steps']!=e['steps'] or eh['gross_start_m3']!=gross or
                        r.get('gross_collapse_m3')!=eh['gross_end_m3']):raise ValueError('Epoch/transfer history differs')
                    if canonical(contract(old))!=canonical(contract(new)):raise ValueError('Epoch physical methods differ')
                    gross=eh['gross_end_m3'];nepoch+=1
                previous=nxt;lasttime=e['time_s'];laststeps=e['steps']
            if previous!=leaves or nepoch!=len(receipts) or m['epoch_gross_start_m3']!=gross:raise ValueError('Incomplete adaptive epoch chain')
            delay=m['delay']
            if (not isinstance(delay,list) or len(delay)!=self.hierarchy.count or
                any(type(v) is not int or not 0<=v<=self.settings.coarsen_delay for v in delay)):
                raise ValueError('Invalid coarsening memory')
            if m['next_time_s']!=self._origin_time+(len(events)+1)*self.settings.interval:raise ValueError('Adaptive next decision clock differs')
            # The trusted prepared origin owns global initial budgets, not a
            # freshly constructed current mesh. Rebase ONLY the private target
            # after authenticating the origin, prepared geometry and epoch chain.
            target._initial_totals=self._totals
            target.solver.initial_solid,target.solver.initial_water=self._totals
            with tempfile.TemporaryDirectory(prefix='braids-jc204-') as tmp:
                inner=Path(tmp)/'accepted.npz';inner.write_bytes(pieces['accepted.npz'])
                JiangCaoRestart.restart(target,inner,check=check)
            if (not lasttime<=target.time<=m['next_time_s']+1e-12 or target.steps<laststeps or target.collapse_gross<gross or
                abs(target.a.host(target.collapse_integral)).sum()>target.collapse_gross-gross+1e-12):
                raise ValueError('Current state and adaptive epoch history differ')
            target._epochs=tuple(pieces[f'epoch_{i}.npz'] for i in range(nepoch))
            target._receipts=tuple(copy.deepcopy(receipts));target._epoch_gross_start=gross
            validate(target);check()
            self.owner=target;self.leaves=leaves;self.delay=np.array(delay,np.int32)
            self.next_time=m['next_time_s'];self._events=tuple(canonical(e) for e in events)
            return dict(time_s=target.time,steps=target.steps,epochs=nepoch,sha256=sha(payload))
