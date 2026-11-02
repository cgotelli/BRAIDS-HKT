"""Q6b Delft3D4-style sediment boundaries and internal point loads.

Parsing and schedule resolution are host preparation.  Per-face and per-cell
updates use the shared native CPU/CUDA equations.  Classic FLOW modes 6/7 are
deliberately fail-closed: the locked source parses them but has no bed-update
consumer.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "solver/wave/braids_sediment_boundaries_v103.h"
SOURCE = HEADER.with_suffix(".cpp")
SCOPE = "braids-q6b-sediment-boundaries-v103"
SUPPORTED_BED_MODES = frozenset((0, 1, 2, 3, 4, 5, 8))
PARSED_UNUSED_BED_MODES = frozenset((6, 7))


def resolve_boundary_selection_v103(bed_mode,composition_mode,*,has_bcfil,coarse_layer=False):
    """Resolve Q6b payload dependencies without silently changing physics."""
    if bed_mode in PARSED_UNUSED_BED_MODES:
        raise NotImplementedError("Locked classic FLOW parses IBedCond 6/7 but has no runtime consumer")
    if bed_mode not in SUPPORTED_BED_MODES or composition_mode not in range(4):
        raise ValueError("Invalid Q6b bed/composition selection")
    required=bed_mode in (2,3,4,5,8) or composition_mode in (2,3)
    if required and not has_bcfil: raise ValueError("Selected boundary methods require a typed BcFil payload")
    reasons=[]
    if required: reasons.append("BcFil required by the selected prescribed boundary method")
    if composition_mode==1 and coarse_layer:
        reasons.append("classic FLOW fixed composition copies the interior column when CrsLyr is enabled")
    return {"bed_mode":bed_mode,"composition_mode":composition_mode,"bcfil_required":required,
            "coarse_layer":bool(coarse_layer),"reasons":tuple(reasons)}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _array_sha(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(a.dtype.str.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _host(value):
    return value.get() if hasattr(value, "get") else np.asarray(value)


_UNIT_SECONDS = {"s": 1., "sec": 1., "second": 1., "seconds": 1.,
                 "min": 60., "minute": 60., "minutes": 60.,
                 "h": 3600., "hr": 3600., "hour": 3600., "hours": 3600.,
                 "d": 86400., "day": 86400., "days": 86400.}


@dataclass(frozen=True)
class FlowTableV103:
    name: str
    location: str
    parameters: tuple[str, ...]
    units: tuple[str, ...]
    time_s: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    reference_time: str = ""
    interpolation: str = "linear"
    contents: str = "Uniform"

    def __post_init__(self):
        if (not self.name or not self.location or len(self.parameters) < 2 or
                len(self.parameters) != len(self.units)):
            raise ValueError("BcFil table requires a name, location and typed parameters")
        t = np.asarray(self.time_s, dtype=float)
        v = np.asarray(self.values, dtype=float)
        if (t.ndim != 1 or len(t) < 1 or v.shape != (len(t), len(self.parameters)-1) or
                not np.isfinite(t).all() or not np.isfinite(v).all() or
                np.any(np.diff(t) <= 0) or self.interpolation.lower() != "linear"):
            raise ValueError("BcFil requires finite increasing time and linear records")

    @property
    def fingerprint(self):
        return _digest(self.__dict__)

    def evaluate(self, time_s, cursor=0):
        if not np.isfinite(time_s):
            raise ValueError("Finite schedule time required")
        t = np.asarray(self.time_s); v = np.asarray(self.values)
        if time_s <= t[0]:
            return v[0].copy(), 0
        if time_s >= t[-1]:
            return v[-1].copy(), len(t)-1
        i = int(np.clip(cursor, 0, len(t)-2))
        while i and time_s < t[i]: i -= 1
        while i+1 < len(t)-1 and time_s > t[i+1]: i += 1
        f = (time_s-t[i])/(t[i+1]-t[i])
        return (1.-f)*v[i]+f*v[i+1], i


def resolve_table_parameters_v103(table,time_s,names,*,alpha=None,cursor=0):
    """Resolve uniform or FLOW ``end A/end B`` values along one boundary."""
    values,next_cursor=table.evaluate(time_s,cursor)
    mapping={name.strip().lower():values[i] for i,name in enumerate(table.parameters[1:])}
    alpha=np.array([0.]) if alpha is None else np.asarray(alpha,dtype=float)
    if alpha.ndim!=1 or not np.isfinite(alpha).all() or np.any((alpha<0)|(alpha>1)):
        raise ValueError("Boundary interpolation coordinate must be in [0,1]")
    result=[]
    for raw in names:
        name=raw.strip().lower()
        if name in mapping: result.append(np.full(len(alpha),mapping[name]));continue
        a=name+" end a";b=name+" end b"
        if a not in mapping or b not in mapping: raise ValueError("BcFil parameter not found: "+raw)
        result.append((1.-alpha)*mapping[a]+alpha*mapping[b])
    return np.ascontiguousarray(result),next_cursor


@dataclass(frozen=True)
class PointLoadPayloadV103:
    """Prepared point geometry and constituent-to-fraction identity."""
    name: str
    cell_index: int
    table_name: str
    discharge_parameter: str
    fraction_ids: tuple[str,...]
    concentration_parameters: tuple[str,...]
    def __post_init__(self):
        if (not self.name or not self.table_name or not self.discharge_parameter or
                type(self.cell_index) is not int or self.cell_index<0 or not self.fraction_ids or
                len(self.fraction_ids)!=len(self.concentration_parameters) or
                len(set(self.fraction_ids))!=len(self.fraction_ids)):
            raise ValueError("Invalid named point-load geometry or fraction mapping")
    @property
    def fingerprint(self): return _digest(self.__dict__)
    def evaluate(self,table,time_s,nc,*,cursor=0):
        if table.name!=self.table_name or self.cell_index>=nc: raise ValueError("Point table/geometry identity differs")
        metadata={name.strip().lower():unit.strip().lower().replace(" ","")
                  for name,unit in zip(table.parameters[1:],table.units[1:])}
        qkey=self.discharge_parameter.strip().lower()
        if metadata.get(qkey) not in {"m3/s","m^3/s","m3s-1"}:
            raise ValueError("Point discharge unit must be m3/s")
        for raw in self.concentration_parameters:
            if metadata.get(raw.strip().lower()) not in {"kg/m3","kg/m^3","kgm-3"}:
                raise ValueError("Point sediment concentration unit must be kg/m3")
        values,next_cursor=resolve_table_parameters_v103(table,time_s,
            (self.discharge_parameter,)+self.concentration_parameters,cursor=cursor)
        q=np.zeros(nc);c=np.zeros((len(self.fraction_ids),nc));q[self.cell_index]=values[0,0]
        c[:,self.cell_index]=values[1:,0]
        return q,c,next_cursor


def parse_bcfil_v103(path):
    """Parse classic FLOW table blocks without guessing names or units."""
    path = Path(path); lines = path.read_text(encoding="utf-8-sig").splitlines()
    tables=[]; i=0
    quoted = re.compile(r"^\s*([^']*?)\s*'([^']*)'\s*(?:unit\s*'([^']*)')?\s*$", re.I)
    while i < len(lines):
        row = lines[i].split("#", 1)[0].strip(); i += 1
        if not row or not row.lower().startswith("table-name"):
            continue
        match=quoted.match(row)
        if not match: raise ValueError("Malformed BcFil table-name")
        data={"table-name": match.group(2)}; parameters=[]; units=[]; records=None
        while i < len(lines):
            raw=lines[i].split("#",1)[0].strip(); i += 1
            if not raw: continue
            low=raw.lower()
            if low.startswith("records-in-table"):
                parts=raw.split()
                try: records=int(parts[-1])
                except (ValueError, IndexError) as exc: raise ValueError("Invalid records-in-table") from exc
                break
            m=quoted.match(raw)
            if m:
                key=m.group(1).strip().lower(); value=m.group(2).strip(); unit_value=m.group(3)
            else:
                plain=raw.split(None,1)
                if len(plain)!=2: raise ValueError("Malformed BcFil metadata: "+raw)
                key=plain[0].strip().lower();value=plain[1].strip().strip("'\"");unit_value=None
            if key == "parameter":
                parameters.append(value); units.append((unit_value or "").strip("[] "))
            else:
                if key in data: raise ValueError("Duplicate BcFil key: "+key)
                data[key]=value
        if records is None or records < 1: raise ValueError("BcFil table lacks records")
        rows=[]
        while len(rows)<records and i<len(lines):
            raw=lines[i].split("#",1)[0].strip(); i+=1
            if raw: rows.append([float(x) for x in raw.replace(","," ").split()])
        if len(rows)!=records or any(len(x)!=len(parameters) for x in rows):
            raise ValueError("BcFil record cardinality differs from parameters")
        if not parameters or parameters[0].strip().lower() != "time":
            raise ValueError("First BcFil parameter must be time")
        unit=(data.get("time-unit") or units[0]).strip("[] ").lower()
        if unit not in _UNIT_SECONDS: raise ValueError("Unsupported BcFil time unit: "+unit)
        a=np.asarray(rows,dtype=float); factor=_UNIT_SECONDS[unit]
        tables.append(FlowTableV103(data["table-name"],data.get("location",""),
            tuple(parameters),tuple(units),tuple((a[:,0]*factor).tolist()),
            tuple(map(tuple,a[:,1:].tolist())),data.get("reference-time",""),
            data.get("interpolation","linear"),data.get("contents","Uniform")))
    if not tables: raise ValueError("BcFil contains no tables")
    keys=[(x.location,x.name) for x in tables]
    if len(set(keys))!=len(keys): raise ValueError("Duplicate BcFil table identity")
    return tuple(tables), {"schema": 1, "scope": SCOPE,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tables": [x.fingerprint for x in tables]}


def prescribed_bedload_v103(mode, scheduled, dry_density, solid_density):
    """Convert BcFil transport units to the BRAIDS kg/(m s) face contract."""
    if mode not in (4,5,8): raise ValueError("Prescribed bedload requires IBedCond 4, 5 or 8")
    q=np.asarray(scheduled,dtype=float); dry=np.asarray(dry_density,dtype=float)
    solid=np.asarray(solid_density,dtype=float)
    if (q.shape != dry.shape or q.shape != solid.shape or not np.isfinite(q).all() or
            not np.isfinite(dry).all() or not np.isfinite(solid).all() or
            np.any(dry<=0) or np.any(solid<=0)):
        raise ValueError("Prescribed bedload arrays differ")
    return np.ascontiguousarray(q*(dry if mode==4 else solid if mode==5 else 1.))


def resolve_prescribed_bedload_table_v103(table,mode,fraction_ids,time_s,*,alpha=None,
                                          dry_density,solid_density,cursor=0):
    """Resolve a typed BcFil bedload table and convert its declared basis."""
    prefix={4:"transport incl pores",5:"transport excl pores",8:"transport mass"}.get(mode)
    if prefix is None: raise ValueError("Prescribed table requires IBedCond 4, 5 or 8")
    names=tuple(f"{prefix} {fraction}" for fraction in fraction_ids)
    metadata={name.strip().lower():unit.strip().lower().replace(" ","")
              for name,unit in zip(table.parameters[1:],table.units[1:])}
    allowed_volume={"m2/s","m^2/s","m2s-1"};allowed_mass={"kg/m/s","kg/s/m","kg/(ms)","kgm-1s-1"}
    allowed=allowed_mass if mode==8 else allowed_volume
    for name in names:
        keys=(name.lower(),name.lower()+" end a",name.lower()+" end b")
        present=[k for k in keys if k in metadata]
        if not present or any(metadata[k] not in allowed for k in present):
            raise ValueError(f"BcFil unit/basis differs for {name}")
    values,next_cursor=resolve_table_parameters_v103(table,time_s,names,alpha=alpha,cursor=cursor)
    dry=np.asarray(dry_density,dtype=float)[:,None];solid=np.asarray(solid_density,dtype=float)[:,None]
    converted=np.ascontiguousarray(values*(dry if mode==4 else solid if mode==5 else 1.))
    return converted,next_cursor


def resolve_bedload_face_v103(mode, inward, computed, *, scheduled=None,
                              dry_density=None, solid_density=None,
                              equilibrium=None):
    computed=np.asarray(computed,dtype=float)
    if not inward: return computed.copy(), "free_export_on_reversal"
    if mode in (4,5,8):
        return prescribed_bedload_v103(mode,scheduled,dry_density,solid_density), "prescribed_inflow"
    if mode==0 and equilibrium is not None:
        eq=np.asarray(equilibrium,dtype=float)
        if eq.shape!=computed.shape or not np.isfinite(eq).all(): raise ValueError("Equilibrium feed shape differs")
        return eq.copy(), "equilibrium_inflow"
    if mode in PARSED_UNUSED_BED_MODES:
        raise NotImplementedError("Locked classic FLOW parses IBedCond 6/7 but its bed update does not consume them")
    if mode not in SUPPORTED_BED_MODES: raise ValueError("Invalid bed boundary mode")
    return computed.copy(), "computed_inflow"


@lru_cache(maxsize=1)
def native_boundary_runtime_v103():
    compiler=shutil.which("g++")
    if not compiler: raise RuntimeError("Q6b native route requires an OpenMP C++ compiler")
    directory=Path(tempfile.mkdtemp(prefix="braids_q6b_v103_"))
    library=directory/("q6b.dll" if os.name=="nt" else "q6b.so")
    flags=["-O3","-fopenmp","-shared","-ffp-contract=off","-std=c++11"]
    if os.name=="nt": flags += ["-static-libstdc++","-static-libgcc"]
    build=subprocess.run([compiler,*flags,str(SOURCE),"-o",str(library)],cwd=directory,
                         capture_output=True,text=True,timeout=120)
    if build.returncode: raise RuntimeError(build.stdout+build.stderr)
    dependency=os.add_dll_directory(str(Path(compiler).parent)) if os.name=="nt" else None
    dll=ctypes.CDLL(str(library)); real=np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
    integer=np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
    bed=dll.braids_q6b_bed_v103
    bed.argtypes=[ctypes.c_int]*4+[ctypes.c_double]+[real]*5+[integer]; bed.restype=None
    comp=dll.braids_q6b_composition_v103
    comp.argtypes=[ctypes.c_int]*6+[real]*4+[integer]; comp.restype=None
    load=dll.braids_q6b_load_v103
    load.argtypes=[ctypes.c_int]*3+[ctypes.c_double]+[real]*5+[integer,real,real,integer]; load.restype=None
    receipt={"source":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
             "equations":hashlib.sha256(HEADER.read_bytes()).hexdigest(),
             "compiler":hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),
             "binary":hashlib.sha256(library.read_bytes()).hexdigest(),"flags":flags}
    return bed,comp,load,dll,dependency,receipt


class NativeSedimentBoundaryRuntimeV103:
    backend="native_cpu_openmp"
    def __init__(self, fraction_ids, solid_density, *, threads=4):
        self.fraction_ids=tuple(fraction_ids); self.rho=np.ascontiguousarray(solid_density,dtype=float)
        if (not self.fraction_ids or len(set(self.fraction_ids))!=len(self.fraction_ids) or
                self.rho.shape!=(len(self.fraction_ids),) or np.any(self.rho<=0) or
                not np.isfinite(self.rho).all() or type(threads) is not int or not 1<=threads<=256):
            raise ValueError("Invalid Q6b fractions, density or thread count")
        self.threads=threads; self.bed_fn,self.comp_fn,self.load_fn,_,_,self.receipt=native_boundary_runtime_v103()
        self.identity=_digest({"scope":SCOPE,"backend":self.backend,"fractions":self.fraction_ids,
            "rho":self.rho.tolist(),"threads":threads,"runtime":self.receipt})

    def bed_change(self,mode,inward,dt,computed,interior,depth,prescribed):
        arrays=[np.ascontiguousarray(x,dtype=float) for x in (computed,interior,depth,prescribed)]
        nc=len(arrays[0])
        if any(a.shape!=(nc,) or not np.isfinite(a).all() for a in arrays) or not np.isfinite(dt) or dt<0:
            raise ValueError("Invalid bed-boundary fields")
        out=np.empty(nc); status=np.empty(nc,dtype=np.int32)
        self.bed_fn(nc,self.threads,int(mode),int(bool(inward)),dt,*arrays,out,status)
        if np.any(status):
            if mode in PARSED_UNUSED_BED_MODES: raise NotImplementedError("IBedCond 6/7 is parsed but unused in locked classic FLOW")
            raise ValueError("Bed-boundary update failed")
        return out

    def composition(self,mode,boundary,interior,values,*,coarse_layer=False):
        out=np.ascontiguousarray(boundary,dtype=float).copy(); inside=np.ascontiguousarray(interior,dtype=float)
        val=np.ascontiguousarray(values,dtype=float); nl,ng,nc=out.shape
        if (ng!=len(self.rho) or inside.shape!=out.shape or val.shape!=(ng,nc) or
                not all(np.isfinite(a).all() for a in (out,inside,val)) or
                np.any(out<0) or np.any(inside<0) or np.any(val<0)):
            raise ValueError("Invalid composition-boundary fields")
        status=np.empty(nc,dtype=np.int32)
        self.comp_fn(nc,ng,nl,self.threads,int(mode),int(bool(coarse_layer)),self.rho,inside,val,out,status)
        if np.any(status): raise ValueError("Composition-boundary update failed")
        return out

    def point_load(self,mass,dt,discharge,concentration,initial_volume,current_volume,area,wet):
        out=np.ascontiguousarray(mass,dtype=float).copy(); ng,nc=out.shape
        q=np.ascontiguousarray(discharge,dtype=float); cin=np.ascontiguousarray(concentration,dtype=float)
        v0=np.ascontiguousarray(initial_volume,dtype=float); v=np.ascontiguousarray(current_volume,dtype=float)
        a=np.ascontiguousarray(area,dtype=float); w=np.ascontiguousarray(wet,dtype=np.int32)
        if (ng!=len(self.rho) or cin.shape!=(ng,nc) or any(x.shape!=(nc,) for x in (q,v0,v,a,w)) or
                not all(np.isfinite(x).all() for x in (out,q,cin,v0,v,a)) or np.any(out<0) or
                np.any(cin<0) or np.any(v0<0) or np.any(v<0) or np.any(a<=0) or
                np.any((w!=0)&(w!=1)) or not np.isfinite(dt) or dt<0):
            raise ValueError("Invalid point-load fields")
        delta=np.empty_like(out); status=np.empty(nc,dtype=np.int32)
        self.load_fn(nc,ng,self.threads,dt,q,cin,v0,v,a,w,out,delta,status)
        if np.any(status==2): raise ValueError("Positive suspended source has zero initial cell volume")
        if np.any(status): raise ValueError("Point suspended load failed")
        return out,delta


def apply_point_load_operations_v103(runtime,mass,dt,operations):
    """Apply up to four named point loads sequentially without sign aggregation."""
    operations=tuple(operations)
    if not 1<=len(operations)<=4: raise ValueError("Bounded Q6b transaction requires 1--4 point loads")
    current=mass;total=None
    required=("discharge","concentration","initial_volume","current_volume","area","wet")
    for operation in operations:
        if set(operation)!=set(required): raise ValueError("Point-load operation payload differs")
        current,delta=runtime.point_load(current,dt,*(operation[k] for k in required))
        total=delta.copy() if total is None else total+delta
    return current,total


class SedimentBoundaryOwnerV103:
    """One accepted Q6b owner with rollback and exact event-cursor restart."""
    def __init__(self,runtime,boundary_mass,suspended_mass,bed_change,*,time_s=0.,cursors=None,
                 coarse_layer=False):
        self.runtime=runtime; self.xp=getattr(runtime,"cp",np)
        self.boundary_mass=self.xp.array(boundary_mass,dtype=self.xp.float64,order="C",copy=True)
        self.suspended_mass=self.xp.array(suspended_mass,dtype=self.xp.float64,order="C",copy=True)
        self.bed_change=self.xp.array(bed_change,dtype=self.xp.float64,order="C",copy=True)
        ng=len(runtime.fraction_ids)
        if (self.boundary_mass.ndim!=3 or self.boundary_mass.shape[1]!=ng or
                self.suspended_mass.ndim!=2 or self.suspended_mass.shape[0]!=ng or
                self.bed_change.ndim!=1 or self.boundary_mass.shape[2]!=len(self.bed_change) or
                not all(np.isfinite(_host(x)).all() for x in (self.boundary_mass,self.suspended_mass,self.bed_change)) or
                np.any(_host(self.boundary_mass)<0) or np.any(_host(self.suspended_mass)<0) or
                not np.isfinite(time_s) or time_s<0):
            raise ValueError("Invalid initial Q6b accepted state")
        if not isinstance(coarse_layer,(bool,np.bool_)): raise ValueError("Explicit coarse-layer topology must be boolean")
        self.coarse_layer=bool(coarse_layer)
        self.time_s=float(time_s); self.updates=0; self.cursors=dict(cursors or {})
        self.source_kg=np.zeros(ng); self.sink_kg=np.zeros(ng); self.pending=None; self._seal()

    def _token(self):
        return _digest({"runtime":self.runtime.identity,"time":self.time_s,"updates":self.updates,
            "cursors":self.cursors,"coarse_layer":self.coarse_layer,"arrays":_array_sha(*(_host(x) for x in
             (self.boundary_mass,self.suspended_mass,self.bed_change,self.source_kg,self.sink_kg)))})
    def _seal(self):
        if self.xp is np:
            for x in (self.boundary_mass,self.suspended_mass,self.bed_change): x.setflags(write=False)
        self.state_sha256=self._token()
    def validate(self):
        if self._token()!=self.state_sha256: raise ValueError("Accepted Q6b state was mutated")

    def begin(self,dt,*,bed_mode,inward,computed_change,interior_change,current_depth,prescribed,
              composition_mode,interior_boundary_mass,composition_values,point_discharge,
              point_concentration,initial_volume,current_volume,area,wet,cursors=None,cancelled=False,
              point_load_operations=None):
        self.validate()
        if self.pending is not None: raise ValueError("An uncommitted Q6b trial exists")
        if cancelled: raise InterruptedError("Q6b update cancelled before launch")
        if self.updates>=12: raise ValueError("Bounded Q6b route permits at most 12 accepted updates")
        bed=self.runtime.bed_change(bed_mode,inward,dt,computed_change,interior_change,current_depth,prescribed)
        bmass=self.runtime.composition(composition_mode,self.boundary_mass,interior_boundary_mass,
                                       composition_values,coarse_layer=self.coarse_layer)
        if point_load_operations is None:
            smass,delta=self.runtime.point_load(self.suspended_mass,dt,point_discharge,point_concentration,
                                                 initial_volume,current_volume,area,wet)
        else:
            smass,delta=apply_point_load_operations_v103(self.runtime,self.suspended_mass,dt,
                                                          point_load_operations)
        dh=_host(delta); cell_area=np.asarray(_host(area),dtype=float)
        source=(np.maximum(dh,0)*cell_area[None,:]).sum(axis=1)
        sink=(np.maximum(-dh,0)*cell_area[None,:]).sum(axis=1)
        self.pending=(bed,bmass,smass,source,sink,float(dt),dict(self.cursors if cursors is None else cursors))
        self.pending_sha256=_digest({"arrays":_array_sha(*(_host(x) for x in self.pending[:5])),
                                    "dt":self.pending[5],"cursors":self.pending[6]})
        return self.pending

    def commit(self,*,cancelled=False):
        self.validate()
        if self.pending is None: raise ValueError("No Q6b trial")
        if cancelled:
            self.rollback(); raise InterruptedError("Q6b update cancelled before publication")
        token=_digest({"arrays":_array_sha(*(_host(x) for x in self.pending[:5])),
                       "dt":self.pending[5],"cursors":self.pending[6]})
        if token!=self.pending_sha256: raise ValueError("Q6b trial was mutated")
        self.bed_change,self.boundary_mass,self.suspended_mass,source,sink,dt,self.cursors=self.pending
        self.source_kg+=source; self.sink_kg+=sink; self.time_s+=dt; self.updates+=1
        self.pending=None; self._seal(); return source-sink
    def rollback(self): self.pending=None; self.pending_sha256=None

    def checkpoint(self):
        self.validate()
        if self.pending is not None: raise ValueError("Checkpoint requires committed Q6b state")
        data={"schema":1,"scope":SCOPE,"runtime":self.runtime.identity,"time_s":self.time_s,
              "updates":self.updates,"cursors":self.cursors,"coarse_layer":self.coarse_layer,
              **{k:_host(getattr(self,k)).tolist() for k in
                 ("boundary_mass","suspended_mass","bed_change","source_kg","sink_kg")},
              "state_sha256":self.state_sha256}
        return {**data,"checkpoint_sha256":_digest(data)}

    def result_fields(self):
        self.validate()
        return {"SourceSinkTerms":{"source_kg":self.source_kg.copy(),
                                    "sink_kg":self.sink_kg.copy(),
                                    "net_kg":self.source_kg-self.sink_kg,
                                    "producer_scope":SCOPE}}

    def restore(self,data):
        raw={k:v for k,v in data.items() if k!="checkpoint_sha256"}
        if (_digest(raw)!=data.get("checkpoint_sha256") or data.get("schema")!=1 or
                data.get("scope")!=SCOPE or data.get("runtime")!=self.runtime.identity):
            raise ValueError("Q6b restart identity differs")
        candidate=SedimentBoundaryOwnerV103(self.runtime,data["boundary_mass"],data["suspended_mass"],
            data["bed_change"],time_s=data["time_s"],cursors=data["cursors"],
            coarse_layer=data["coarse_layer"])
        candidate.updates=int(data["updates"])
        if not 0<=candidate.updates<=12: raise ValueError("Invalid Q6b restart count")
        for key in ("source_kg","sink_kg"):
            value=np.asarray(data[key],dtype=float)
            if value.shape!=getattr(candidate,key).shape or not np.isfinite(value).all(): raise ValueError("Invalid Q6b ledger")
            setattr(candidate,key,value)
        candidate._seal()
        if candidate.state_sha256!=data["state_sha256"]: raise ValueError("Q6b restart state differs")
        self.__dict__.update(candidate.__dict__)


__all__=["FlowTableV103","NativeSedimentBoundaryRuntimeV103","PARSED_UNUSED_BED_MODES",
         "PointLoadPayloadV103",
         "SUPPORTED_BED_MODES","SedimentBoundaryOwnerV103","parse_bcfil_v103",
         "apply_point_load_operations_v103",
         "prescribed_bedload_v103","resolve_bedload_face_v103","resolve_prescribed_bedload_table_v103",
         "resolve_table_parameters_v103",
         "resolve_boundary_selection_v103",
         "native_boundary_runtime_v103"]
