"""Strict research restart of the JC wet/dry composition, not Desktop admission.

SHA256 binds canonical metadata, array shape/dtype and bytes. This detects
corruption and incompatible inputs; it is not a keyed signature against an
attacker who can rewrite both the payload and its checksums. No pickle, legacy
migration, backend substitution, or partial-state restart is supported.
"""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4
import copy
import hashlib
import io
import json
import os
import threading
import zipfile
import numpy as np
from .bend_methods_v1 import digest
from .jiang_cao_wetdry import JiangCaoShoreline
from braids_core.windows_paths import filesystem_path

SCHEMA='braids.jiang_cao.accepted_checkpoint202.v1'
ARRAYS=('state','solid_integral','water_integral','collapse_integral','secondary_fields','dispersion_flux')
MAX_METADATA=2*1024*1024


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def array_hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(canonical(dict(shape=list(a.shape),dtype=a.dtype.str))+a.tobytes()).hexdigest()


def _unique(pairs):
    out={}
    for key,value in pairs:
        if key in out:raise ValueError('Duplicate checkpoint metadata key')
        out[key]=value
    return out


def _invalid_constant(value):raise ValueError('Nonfinite checkpoint metadata')


def decode(payload,shapes):
    """Bound archive sizes and NPY headers before allocating declared arrays."""
    arrays={}
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            names=[i.filename for i in z.infolist()]
            if len(names)!=len(set(names)) or set(names)!={k+'.npy' for k in (*ARRAYS,'header')}:
                raise ValueError('Checkpoint archive inventory differs')
            for info in z.infolist():
                key=info.filename[:-4];limit=MAX_METADATA if key=='header' else 8*int(np.prod(shapes[key]))
                if info.file_size>limit+512 or info.compress_type!=zipfile.ZIP_STORED or info.flag_bits&1:
                    raise ValueError('Unsupported or oversized checkpoint member')
                raw=z.read(info);stream=io.BytesIO(raw)
                version=np.lib.format.read_magic(stream)
                if version!=(1,0):raise ValueError('Unsupported checkpoint NPY version')
                shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream,max_header_size=512)
                valid=(dtype==np.dtype('uint8') and len(shape)==1 and 0<shape[0]<=MAX_METADATA) if key=='header' else (
                    dtype.str=='<f8' and shape==shapes[key])
                if fortran or not valid:raise ValueError('Checkpoint shape or dtype differs')
                count=int(np.prod(shape));offset=stream.tell()
                if len(raw)-offset!=count*dtype.itemsize:raise ValueError('Checkpoint array length differs')
                arrays[key]=np.frombuffer(raw,dtype=dtype,count=count,offset=offset).reshape(shape).copy()
        header=json.loads(arrays.pop('header').tobytes().decode('utf-8'),object_pairs_hook=_unique,parse_constant=_invalid_constant)
        if not isinstance(header,dict):raise ValueError('Checkpoint header must be an object')
        return header,arrays
    except (OSError,EOFError,zipfile.BadZipFile,UnicodeError,KeyError) as exc:
        raise ValueError('Unreadable or corrupt checkpoint') from exc


def encode(header,arrays):
    h=copy.deepcopy(header);h['arrays']={k:array_hash(v) for k,v in arrays.items()}
    h['content_sha256']=hashlib.sha256(canonical(h)).hexdigest()
    data=canonical(h)
    if len(data)>MAX_METADATA:raise ValueError('Checkpoint metadata too large')
    stream=io.BytesIO();np.savez(stream,header=np.frombuffer(data,dtype=np.uint8),**arrays)
    return stream.getvalue()


class JiangCaoRestart:
    """Lifecycle owner around the unchanged 201 physical stepping implementation.

    Fresh owner required. Sediment feed is an explicit constant scalar/per-face
    concentration bound to restart identity. Callable/changed feed is rejected.
    Hydraulic time-series settings are already bound by the input-deck identity.
    """
    def __init__(self,solver,*,inlet_concentration=0.):
        if type(solver) is not JiangCaoShoreline or solver.time!=0 or solver.steps!=0:
            raise ValueError('Fresh qualified JiangCaoShoreline owner required')
        if callable(inlet_concentration):raise ValueError('Restart requires declared constant inlet concentration, not a callable')
        self.solver=solver;self._lock=threading.Lock()
        feed=np.ascontiguousarray(np.broadcast_to(np.asarray(inlet_concentration,dtype=np.float64),(solver.nf,))).copy()
        if not np.isfinite(feed).all() or np.any((feed<0)|(feed>=solver.B)):
            raise ValueError('Inlet concentration must be finite and below packing')
        self._feed=solver.xp.asarray(feed).copy()
        self._initial_state=solver.a.host(solver.state).copy()
        self._bedrock=self._initial_state[5]-self._initial_state[4]/solver.B
        self._initial_totals=(solver.initial_solid,solver.initial_water)
        self._contract=self._configuration()
        self._contract_bytes=canonical(self._contract)
        self.identity=dict(method='lifecycle.jc2026_checkpoint202_v1',schema=SCHEMA,
            lifecycle_sha256=digest(__file__),physical_owner=copy.deepcopy(solver.identity),
            configuration_sha256=hashlib.sha256(self._contract_bytes).hexdigest(),
            public_support=False,restart_scope='same backend, runtime, geometry, initial state and declared constant sediment feed')
        # z0 is a residual input mirror (the physical owner rewrites it before
        # every rate), not an additional bed history. Initialize it also at t=0.
        solver.a.physics.z0[...]=solver.state[5]
        self._assert_carrier()

    def __getattr__(self,key):return getattr(self.solver,key)

    @contextmanager
    def _exclusive(self):
        if not self._lock.acquire(blocking=False):raise RuntimeError('Step/checkpoint/restart transaction already active')
        try:yield
        finally:self._lock.release()

    def _configuration(self):
        s=self.solver
        result = dict(owner=s.identity,lifecycle_sha256=digest(__file__),backend=s.backend,
            clock_origin_s=float(getattr(s,'initial_clock_origin_s',0.)),
            material=asdict(s.exchange.material),roughness=asdict(s.roughness.parameters),
            shallow=asdict(s.shallow),controls=asdict(s.a.controls),
            switches=[s.secondary_enabled,s.collapse_enabled,s.roughness.bedforms_enabled],
            collapse_controls=[s.collapse.target_slope,s.collapse.tolerance,s.collapse.max_iterations],
            cfl=s.a.midpoint.cfl,geometry={k:array_hash(s.a.host(v)) for k,v in s.mesh.g.items()},
            initial_state_sha256=array_hash(self._initial_state),
            feed_sha256=array_hash(s.a.host(self._feed)))
        if 'direction_resolution' in s.identity:
            d=s.direction
            result['direction_resolution'] = None if d is None else dict(identity=d.identity,
                offsets=array_hash(s.a.host(d.offsets)),neighbours=array_hash(s.a.host(d.neighbours)))
        if 'secondary_adjustment' in s.identity:
            result['secondary_adjustment']=None if s.adjustment is None else s.adjustment.identity
        if 'desktop_bundle' in s.identity and getattr(s.a,'exterior_bed',None) is not None:
            from .jiang_cao_desktop import FreeExterior
            if type(getattr(s,'desktop_exterior',None)) is not FreeExterior:
                raise ValueError('Explicit verified stateless Desktop exterior adapter required')
            result['desktop_exterior']=s.desktop_exterior.verify()
        return result

    def _assert_configuration(self):
        if canonical(self._configuration())!=self._contract_bytes:
            raise ValueError('Live checkpoint configuration changed')

    def _assert_carrier(self):
        s=self.solver;a=s.a;p=a.physics
        exterior=getattr(a,'exterior_bed',None)
        if exterior is not None and 'desktop_bundle' in s.identity:
            from .jiang_cao_desktop import FreeExterior
            if type(getattr(s,'desktop_exterior',None)) is not FreeExterior or s.desktop_exterior.ext is not exterior:
                raise ValueError('Unverified JC Desktop exterior')
            s.desktop_exterior.verify();exterior=None
        if (a.time!=0 or a.steps!=0 or p.time!=0 or p.steps!=0 or p._open or a.morphology or
            a.midpoint.secondary.active or exterior is not None or
            a.midpoint.state()!=dict(step_hint=None,easy_steps=0)):
            raise ValueError('Unadmitted extra carrier history; cannot checkpoint this route')
        for x,y in ((a.q,s.state[:3]),(a.z,s.state[5]),(p.z0,s.state[5])):
            if not bool(s.xp.array_equal(x,y)):raise ValueError('Accepted physical and carrier states differ')

    def step(self,dt):
        with self._exclusive():
            # These diagnostic fields are assigned during stage evaluation;
            # the physical owner's rollback only restores physical inventories.
            diagnostics={k:getattr(self.solver,k).copy() if hasattr(self.solver,k) else None
                         for k in ('secondary_fields','dispersion_flux')}
            try:result=self.solver.step(dt,inlet_concentration=self._feed)
            except BaseException:
                for k,v in diagnostics.items():
                    if v is not None:setattr(self.solver,k,v)
                    elif hasattr(self.solver,k):delattr(self.solver,k)
                raise
            # The parent has already committed these trial transfers. They
            # must never be replayed from a checkpoint or counted twice.
            self._clear_trial()
            return result

    def _clear_trial(self):
        s=self.solver;s._pending.fill(0.);s._gross=0.;s._iterations=0;s._virtual_iterations=0

    def _shapes(self):
        s=self.solver
        return dict(state=(7 if s.adjustment is not None else 6,s.n),solid_integral=(s.nf,),water_integral=(s.nf,),
            collapse_integral=(s.nf,),secondary_fields=(9,s.n),dispersion_flux=(2,s.nf))

    def _snapshot(self):
        s=self.solver;shapes=self._shapes()
        arrays={k:np.ascontiguousarray(s.a.host(getattr(s,k)) if hasattr(s,k) else np.zeros(shapes[k]),dtype='<f8').copy() for k in ARRAYS}
        header=dict(schema=SCHEMA,identity=self.identity,configuration=self._contract,
            time_s=s.time,steps=s.steps,initial_solid_m3=s.initial_solid,initial_water_pore_m3=s.initial_water,
            collapse_gross_m3=s.collapse_gross,last=copy.deepcopy(s.last),
            counters=dict(fwave_evaluations=s.a.midpoint.evaluations,matvecs=s.a.midpoint.matvecs))
        return header,arrays

    def _validate_payload(self,h,arrays):
        required={'schema','identity','configuration','time_s','steps','initial_solid_m3','initial_water_pore_m3',
                  'collapse_gross_m3','last','counters'}
        if set(h)!=required or h['schema']!=SCHEMA:raise ValueError('Unsupported checkpoint schema; no migration')
        if canonical(h['identity'])!=canonical(self.identity) or canonical(h['configuration'])!=self._contract_bytes:
            raise ValueError('Checkpoint model/backend/runtime/input/forcing identity differs')
        s=self.solver
        if type(h['steps']) is not int or h['steps']<0:raise ValueError('Invalid checkpoint step count')
        for key in ('time_s','initial_solid_m3','initial_water_pore_m3','collapse_gross_m3'):
            if type(h[key]) not in (int,float) or not np.isfinite(h[key]) or h[key]<0:raise ValueError('Invalid checkpoint clock or budget')
        origin=float(getattr(s,'initial_clock_origin_s',0.))
        if not np.isfinite(origin) or origin<0 or h['time_s']<origin or ((h['time_s']==origin)!=(h['steps']==0)):
            raise ValueError('Checkpoint time and steps inconsistent with its initial clock')
        if (h['initial_solid_m3'],h['initial_water_pore_m3'])!=self._initial_totals:
            raise ValueError('Checkpoint original phase totals differ')
        counters=h['counters']
        if not isinstance(counters,dict) or set(counters)!={'fwave_evaluations','matvecs'} or any(type(v) is not int or v<0 for v in counters.values()):
            raise ValueError('Invalid checkpoint counters')
        if counters['fwave_evaluations']<2*h['steps'] or counters['matvecs']!=0:
            raise ValueError('Checkpoint counters inconsistent with explicit stepping')
        if not isinstance(h['last'],dict):raise ValueError('Invalid last accepted diagnostics')
        if h['steps']==0:
            if h['last']!={}:raise ValueError('Initial checkpoint cannot have accepted-step diagnostics')
        elif (h['last'].get('time_s')!=h['time_s'] or h['last'].get('cumulative_collapse_gross_solid_m3')!=h['collapse_gross_m3']):
            raise ValueError('Checkpoint diagnostics and accepted clock differ')
        if set(arrays)!=set(ARRAYS):raise ValueError('Checkpoint array inventory differs')
        for key,value in arrays.items():
            if value.shape!=self._shapes()[key] or value.dtype.str!='<f8' or not np.isfinite(value).all():
                raise ValueError('Invalid checkpoint array')
        st=arrays['state'];hh,mm,bb=st[0],st[3],st[4]
        if np.any((hh<0)|(mm<0)|(bb<0)|((hh>0)&(mm>=s.B*hh))|((hh==0)&(mm!=0))):
            raise ValueError('Checkpoint phase bounds violated')
        dry=hh<=s.dry
        if np.any(st[1:4,dry]!=0):raise ValueError('Checkpoint is not a cleaned accepted dry state')
        if st.shape[0]==7 and np.any(st[6,dry]!=0):raise ValueError('Dry checkpoint carries secondary intensity')
        tol=1e-11*max(1.,float(np.max(abs(self._bedrock))))
        if np.max(abs(st[5]-bb/s.B-self._bedrock))>tol:raise ValueError('Checkpoint bedrock datum differs')
        outside=s.a.host(s.mesh.g['neighbour'])<0;wall=s.a.host(s.mesh.g['kind'])==1
        if any(np.any(arrays[k][~outside|wall]!=0) for k in ('solid_integral','water_integral')):
            raise ValueError('Checkpoint has internal/wall boundary ledger')
        if np.any(arrays['collapse_integral'][outside]!=0):raise ValueError('Collapse crosses exterior boundary')
        if h['collapse_gross_m3']+1e-12<np.sum(abs(arrays['collapse_integral'])):
            raise ValueError('Collapse gross history smaller than net transfers')
        area=s.a.host(s.mesh.g['area']);p=s.exchange.material.porosity
        solid=float((mm+bb)@area)+float(arrays['solid_integral'].sum())
        water=float((hh-mm+p*bb/s.B)@area)+float(arrays['water_integral'].sum())
        scale=max(1.,*self._initial_totals)
        if max(abs(solid-h['initial_solid_m3']),abs(water-h['initial_water_pore_m3']))>1e-10*scale:
            raise ValueError('Checkpoint phase budget failed')
        if h['steps']==0 and (not np.array_equal(st,self._initial_state) or h['collapse_gross_m3']!=0 or
            any(np.any(arrays[k]) for k in ARRAYS[1:])):raise ValueError('Initial checkpoint state differs')

    def checkpoint(self,path,*,check=lambda:None):
        """Publish a new, complete checkpoint exclusively; never replace evidence."""
        with self._exclusive():
            check();self._assert_configuration();self._assert_carrier()
            h,arrays=self._snapshot();self._validate_payload(h,arrays);payload=encode(h,arrays);check()
            path=filesystem_path(path);path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():raise FileExistsError('Checkpoint already exists; choose a new name')
            temp=path.with_name('.'+uuid4().hex+'.partial')
            try:
                with temp.open('xb') as stream:
                    stream.write(payload);stream.flush();os.fsync(stream.fileno())
                check()
                # An exclusive same-filesystem link is atomic and cannot
                # overwrite a file published concurrently by another writer.
                os.link(temp,path)
            finally:
                try:temp.unlink(missing_ok=True)
                except OSError:pass # A locked staging file must not hide the original error.
            return dict(path=str(path),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),time_s=h['time_s'],steps=h['steps'])

    def restart(self,path,*,check=lambda:None):
        """Validate and stage everything before publishing into a fresh owner."""
        with self._exclusive():
            if self.solver.time not in (0.,float(getattr(self.solver,'initial_clock_origin_s',0.))) or self.solver.steps!=0:raise ValueError('Restart requires a fresh owner')
            check();self._assert_configuration();self._assert_carrier()
            path=filesystem_path(path);limit=MAX_METADATA+sum(8*int(np.prod(v))+1024 for v in self._shapes().values())+4096
            with path.open('rb') as stream:
                payload=stream.read(limit+1)
            if len(payload)>limit:raise ValueError('Checkpoint archive too large')
            h,arrays=decode(payload,self._shapes())
            if set(h)!={'schema','identity','configuration','time_s','steps','initial_solid_m3','initial_water_pore_m3',
                        'collapse_gross_m3','last','counters','arrays','content_sha256'}:
                raise ValueError('Unsupported checkpoint header; no migration')
            expected=h.pop('content_sha256')
            if not isinstance(expected,str) or hashlib.sha256(canonical(h)).hexdigest()!=expected:
                raise ValueError('Checkpoint metadata checksum failed')
            hashes=h.pop('arrays')
            if hashes!={k:array_hash(v) for k,v in arrays.items()}:raise ValueError('Checkpoint array checksum failed')
            self._validate_payload(h,arrays);check()
            s=self.solver;xp=s.xp
            staged={k:xp.asarray(v).copy() for k,v in arrays.items()}
            if s.backend=='gpu':xp.cuda.get_current_stream().synchronize()
            check() # cancellation before any live state is changed
            oldq=s.a.q.copy();oldz=s.a.z.copy();oldz0=s.a.physics.z0.copy()
            try:
                s.a.q[...]=staged['state'][:3];s.a.z[...]=staged['state'][5];s.a.physics.z0[...]=staged['state'][5]
                if s.backend=='gpu':xp.cuda.get_current_stream().synchronize()
            except BaseException:
                s.a.q[...]=oldq;s.a.z[...]=oldz;s.a.physics.z0[...]=oldz0
                raise
            for k,v in staged.items():setattr(s,k,v)
            s.time=h['time_s'];s.steps=h['steps'];s.collapse_gross=h['collapse_gross_m3'];s.last=copy.deepcopy(h['last'])
            s.a.midpoint.evaluations=h['counters']['fwave_evaluations'];s.a.midpoint.matvecs=h['counters']['matvecs']
            self._clear_trial()
            return dict(time_s=s.time,steps=s.steps,sha256=hashlib.sha256(payload).hexdigest())
