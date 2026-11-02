"""Q8b cohesive-bank/fine-reservoir composition (v107).

The transaction gives one owner to bank removal, adjacent-toe release,
optional parent-owned repose, and the accepted Q9 water/bed/fluff exchange.
It is a BRAIDS extension: the locked Delft3D sources remain read-only and are
used only for the already qualified fine and dry-bank reference operators.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

import numpy as np

from .fines_v106 import FineMethodsV106, FineParametersV106, FineRuntimeV106
from .planform_v105 import (
    NativePlanformRuntimeV105, PlanformMethodsV105, PlanformParametersV105,
)

SCOPE = "braids-q8b-cohesive-bank-v107"
STATE_KEYS = (
    "layers", "suspended_mass_areal", "fluff_mass_areal", "solid_fraction",
    "preload", "deposit_time", "area", "depth", "shear_pa",
    "bulk_density", "sediment_type", "rock", "bank_mask", "toe_target",
    "bank_height_m", "bank_face_length_m", "undercut_m", "collapse_count",
)
FINE_STATE_KEYS = (
    "layers", "suspended_mass_areal", "fluff_mass_areal", "solid_fraction",
    "preload", "deposit_time", "area", "depth", "shear_pa",
    "bulk_density", "sediment_type",
)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class CohesiveBankMethodsV107:
    toe_erosion: str = "cohesive_excess_shear"
    undercut: str = "retained_threshold"
    collapse: str = "threshold_block"
    deposition: str = "adjacent_toe"
    fine_release: str = "suspended_toe"
    avalanche: bool = True
    integrator: str = "implicit_fwave_midpoint"

    def __post_init__(self):
        required = (
            (self.toe_erosion, "cohesive_excess_shear"),
            (self.undercut, "retained_threshold"),
            (self.collapse, "threshold_block"),
            (self.deposition, "adjacent_toe"),
            (self.fine_release, "suspended_toe"),
        )
        if any(actual != expected for actual, expected in required):
            raise ValueError("Q8b exposes one inseparable, physically compatible bank bundle")
        if type(self.avalanche) is not bool:
            raise ValueError("Q8b avalanche selection must be boolean")
        if self.integrator != "implicit_fwave_midpoint":
            raise ValueError("Q8b GPU composition requires implicit_fwave_midpoint")

    @property
    def method_ids(self) -> tuple[str, ...]:
        result = (
            "bank_toe.cohesive_excess_shear",
            "bank_undercut.retained_threshold",
            "bank_collapse.threshold_block",
            "bank_release.suspended_toe",
            "bank_deposition.adjacent_toe",
        )
        if self.avalanche:
            result += ("redistribution.braids_conservative_wet_dry_avalanche",)
        return result

    @property
    def fingerprint(self) -> str:
        return digest({"scope": SCOPE, **asdict(self), "method_ids": self.method_ids})


@dataclass(frozen=True)
class CohesiveBankParametersV107:
    erodibility_m_pa_s: float = 1.e-3
    critical_shear_pa: float = 5.0
    erosion_exponent: float = 1.0
    critical_undercut_m: float = 5.e-5

    def __post_init__(self):
        values = np.asarray(tuple(asdict(self).values()), dtype=float)
        if not np.isfinite(values).all() or self.erodibility_m_pa_s <= 0 or \
                self.critical_shear_pa < 0 or self.erosion_exponent <= 0 or \
                self.critical_undercut_m <= 0:
            raise ValueError("Invalid cohesive-bank material parameters")


def resolve_cohesive_bank_methods_v107(sediment_types, *, enabled=True, avalanche=True):
    """Resolve the mandatory auxiliary bundle from one high-level choice."""
    if type(enabled) is not bool or type(avalanche) is not bool:
        raise ValueError("Q8b resolver controls must be boolean")
    if not enabled:
        return None
    types = np.asarray(sediment_types, dtype=np.int32)
    if types.ndim != 1 or types.size == 0 or np.any((types < 0) | (types > 2)):
        raise ValueError("Sediment types must be sand=0, silt=1 or clay=2")
    if not np.any(types > 0):
        raise ValueError("Cohesive-bank physics requires at least one silt/clay fraction")
    return CohesiveBankMethodsV107(avalanche=avalanche)


def cohesive_bank_catalogue_v107() -> dict:
    """Canonical zero-step Q8b contract; QGIS admission remains Q10."""
    methods = [
        {"method_id": "bank_toe.cohesive_excess_shear", "component": "bank_toe",
         "owner": "cohesive_bank_v107", "requires": ["fraction_type.silt_or_clay"]},
        {"method_id": "bank_undercut.retained_threshold", "component": "bank_undercut",
         "owner": "cohesive_bank_v107", "requires": ["bank_toe.cohesive_excess_shear"]},
        {"method_id": "bank_collapse.threshold_block", "component": "bank_collapse",
         "owner": "cohesive_bank_v107", "requires": ["bank_undercut.retained_threshold"]},
        {"method_id": "bank_release.suspended_toe", "component": "bank_release",
         "owner": "cohesive_bank_v107", "requires": ["suspension.cohesive_mixed"]},
        {"method_id": "bank_deposition.adjacent_toe", "component": "bank_deposition",
         "owner": "cohesive_bank_v107", "requires": ["bank_collapse.threshold_block"]},
    ]
    parameters = [
        {"parameter_id": key, "units": units, "positive": positive}
        for key, units, positive in (
            ("erodibility_m_pa_s", "m/(Pa s)", True),
            ("critical_shear_pa", "Pa", False),
            ("erosion_exponent", "1", True),
            ("critical_undercut_m", "m", True),
            ("bank_height_m", "m", True),
            ("bank_face_length_m", "m", True),
        )
    ]
    outputs = [
        "output.bank_toe_release_mass", "output.bank_collapse_mass",
        "output.bank_suspended_release_mass", "output.bank_bed_deposition_mass",
        "output.bank_undercut", "output.bank_collapse_count",
    ]
    body = {"schema": "braids.q8b.catalogue.v107", "methods": methods,
            "parameters": parameters, "outputs": outputs,
            "automatic_bundle": True, "qgis_admission": "deferred_to_q10",
            "managed_fildad": "excluded", "fine_mass_owner": "braids.fines.v106"}
    body["identity"] = digest(body)
    return body


class CohesiveBankRuntimeV107:
    """Backend-neutral array transaction; NumPy is CPU and CuPy is CUDA."""

    def __init__(self, methods=CohesiveBankMethodsV107(),
                 parameters=CohesiveBankParametersV107(), *, fine_runtime=None,
                 planform_runtime=None, threads=4):
        self.methods, self.parameters = methods, parameters
        self.fine = fine_runtime or FineRuntimeV106(FineMethodsV106(), FineParametersV106())
        self.xp = self.fine.xp
        self.backend = self.fine.backend
        if methods.avalanche and planform_runtime is None:
            planform_runtime = NativePlanformRuntimeV105(
                PlanformMethodsV105("braids_conservative_wet_dry_avalanche"),
                PlanformParametersV105(), threads=threads)
        self.planform = planform_runtime
        if methods.avalanche:
            if self.planform is None or self.planform.methods.relaxation != \
                    "braids_conservative_wet_dry_avalanche":
                raise ValueError("Q8b avalanche requires the accepted Q8a repose owner")
            if self.planform.methods.dry_bank or self.planform.methods.toe_collapse:
                raise ValueError("Q8b owns bank transfer; Q8a may own repose only")
            plan_xp = getattr(self.planform, "xp", np)
            if plan_xp is not self.xp:
                raise ValueError("Q8b components must execute on the same selected backend")
        self.catalogue = cohesive_bank_catalogue_v107()
        self.identity = digest({"scope": SCOPE, "backend": self.backend,
            "methods": methods.fingerprint, "parameters": asdict(parameters),
            "fine_runtime": self.fine.identity,
            "planform_runtime": None if self.planform is None else self.planform.identity,
            "catalogue": self.catalogue["identity"]})

    def _host(self, value):
        return self.fine._host(value)

    def _bool(self, value):
        return self.fine._bool(value)

    def _array(self, value, shape, name, *, positive=False, nonnegative=False, dtype=None):
        xp = self.xp
        dtype = xp.float64 if dtype is None else dtype
        a = xp.asarray(value, dtype=dtype)
        if a.shape == ():
            a = xp.full(shape, a, dtype=dtype)
        elif a.shape != shape:
            a = xp.broadcast_to(a, shape).copy()
        a = xp.ascontiguousarray(a)
        if self._bool(xp.any(~xp.isfinite(a))) or \
                (positive and self._bool(xp.any(a <= 0))) or \
                (nonnegative and self._bool(xp.any(a < 0))):
            raise ValueError(f"Invalid Q8b {name}")
        return a

    def _validated(self, state):
        missing = tuple(key for key in STATE_KEYS if key not in state)
        if missing:
            raise ValueError("Q8b state is missing " + ", ".join(missing))
        xp = self.xp
        q = xp.ascontiguousarray(xp.asarray(state["layers"], dtype=xp.float64))
        if q.ndim != 4 or min(q.shape) < 1:
            raise ValueError("Q8b layers require (layer,grain,y,x)")
        nl, ng, ny, nx = q.shape; shape = (ny, nx); nc = ny*nx
        typ = xp.asarray(state["sediment_type"], dtype=xp.int32)
        if typ.shape != (ng,) or self._bool(xp.any((typ < 0) | (typ > 2))) or \
                not self._bool(xp.any(typ > 0)):
            raise ValueError("Q8b requires a valid mixed/cohesive fraction profile")
        result = {key: xp.asarray(state[key]).copy() for key in STATE_KEYS}
        result["layers"] = q
        result["area"] = self._array(state["area"], shape, "area", positive=True)
        result["depth"] = self._array(state["depth"], shape, "depth", nonnegative=True)
        result["shear_pa"] = self._array(state["shear_pa"], shape, "shear", nonnegative=True)
        result["bulk_density"] = self._array(state["bulk_density"], (ng,), "density", positive=True)
        result["solid_fraction"] = self._array(state["solid_fraction"], (nl,ny,nx),
                                                "solid fraction", positive=True)
        if self._bool(xp.any(result["solid_fraction"] > 1)):
            raise ValueError("Q8b solid fraction cannot exceed one")
        result["bank_height_m"] = self._array(state["bank_height_m"], shape,
                                               "bank height", positive=True)
        result["bank_face_length_m"] = self._array(state["bank_face_length_m"], shape,
                                                    "bank face length", positive=True)
        result["undercut_m"] = self._array(state["undercut_m"], shape, "undercut", nonnegative=True)
        result["collapse_count"] = xp.ascontiguousarray(xp.asarray(
            state["collapse_count"], dtype=xp.int64))
        result["bank_mask"] = xp.ascontiguousarray(xp.asarray(state["bank_mask"], dtype=xp.bool_))
        result["toe_target"] = xp.ascontiguousarray(xp.asarray(state["toe_target"], dtype=xp.int64))
        if result["bank_mask"].shape != shape or result["toe_target"].shape != shape or \
                result["collapse_count"].shape != shape:
            raise ValueError("Q8b bank topology shape differs")
        active = result["bank_mask"].ravel(); target = result["toe_target"].ravel()
        invalid = active & ((target < 0) | (target >= nc) | (target == xp.arange(nc)))
        if self._bool(xp.any(invalid)):
            raise ValueError("Every active bank cell needs a distinct in-patch toe target")
        active_target = target[active]
        if active_target.size and self._bool(xp.any(result["depth"].ravel()[active_target] <= 0)):
            raise ValueError("Cohesive bank release requires an adjacent wet toe")
        for key in ("suspended_mass_areal", "fluff_mass_areal"):
            result[key] = self._array(state[key], (ng,ny,nx), key, nonnegative=True)
        for key in ("preload", "deposit_time"):
            result[key] = self._array(state[key], (nl,ny,nx), key,
                                      nonnegative=(key == "preload"))
        result["rock"] = self._array(state["rock"], shape, "rock")
        if self._bool(xp.any(~xp.isfinite(q))) or self._bool(xp.any(q < 0)):
            raise ValueError("Q8b layer inventory must be finite and nonnegative")
        return result

    def _available_bulk_volume(self, layers, area, bulk, solid_fraction):
        xp = self.xp
        solid = layers / bulk[None,:,None,None]
        return xp.sum(solid, axis=1) / solid_fraction * area[None]

    def _extract(self, layers, request_bulk_volume, area, bulk, solid_fraction):
        """Remove requested physical bank volume top-down, returning cell mass."""
        xp = self.xp; q = layers.copy(); remaining = request_bulk_volume.copy()
        removed = xp.zeros(q.shape[1:], dtype=xp.float64); actual = xp.zeros_like(remaining)
        for level in range(q.shape[0]):
            available_solid = xp.sum(q[level]/bulk[:,None,None], axis=0)*area
            available_bulk = available_solid/xp.maximum(solid_fraction[level], 1.e-15)
            take = xp.minimum(remaining, available_bulk)
            fraction = xp.where(available_bulk > 0, take/xp.maximum(available_bulk,1.e-15), 0.)
            mass = q[level]*area[None]*fraction[None]
            q[level] -= mass/area[None]; removed += mass
            remaining -= take; actual += take
        return q, removed, actual, remaining

    def _deposit(self, layers, water, removed, active, targets, area, typ, *, fines_to_water):
        xp = self.xp; q = layers; w = water
        flat_q = q[0].reshape((q.shape[1], -1)); flat_w = w.reshape((w.shape[0], -1))
        target_area = area.ravel()[targets]
        for grain in range(q.shape[1]):
            values = removed[grain].ravel()[active]
            destination = flat_w[grain] if fines_to_water and int(self._host(typ[grain])) > 0 else flat_q[grain]
            xp.add.at(destination, targets, values/target_area)
        return q, w

    @staticmethod
    def _cancelled(cancelled):
        return bool(cancelled() if callable(cancelled) else cancelled)

    def advance(self, *, dt_s, time_s=0., morfac=1., dx_m=1., dy_m=1.,
                fine_forcing=None, transfer_tokens=(), cancelled=False, **state):
        if not np.isfinite([dt_s,time_s,morfac,dx_m,dy_m]).all() or dt_s <= 0 or \
                morfac < 0 or min(dx_m,dy_m) <= 0:
            raise ValueError("Invalid Q8b clock/grid")
        xp = self.xp; s = self._validated(state); p = self.parameters
        q = s["layers"].copy(); water = s["suspended_mass_areal"].copy()
        area = s["area"]; typ = s["sediment_type"]; active = s["bank_mask"].ravel()
        targets = s["toe_target"].ravel()[active]
        depth_toe = s["depth"].ravel()[targets]
        shear = s["shear_pa"].ravel()[active]
        height = s["bank_height_m"].ravel()[active]
        face = s["bank_face_length_m"].ravel()[active]
        retreat = p.erodibility_m_pa_s*xp.maximum(shear-p.critical_shear_pa,0.)**p.erosion_exponent
        retreat *= dt_s*morfac
        toe_height = xp.minimum(depth_toe, height)
        toe_request_flat = xp.zeros(area.size, dtype=xp.float64)
        toe_request_flat[active] = retreat*toe_height*face
        q, toe_removed, toe_actual, _ = self._extract(q, toe_request_flat.reshape(area.shape),
            area, s["bulk_density"], s["solid_fraction"])
        denominator = toe_height*face
        actual_retreat = xp.where(denominator > 0,
            toe_actual.ravel()[active]/xp.maximum(denominator,1.e-15), 0.)
        total_undercut = s["undercut_m"].ravel()[active] + actual_retreat
        # A collapse threshold is a discrete physical event.  Bias exact
        # multiples by a few ulps so CPU/CUDA reduction noise cannot select
        # different event counts after repeated accepted updates.
        threshold_tol=64.*np.finfo(np.float64).eps*xp.maximum(
            total_undercut,p.critical_undercut_m)
        requested_events = xp.floor((total_undercut+threshold_tol)/
                                    p.critical_undercut_m).astype(xp.int64)
        upper_height = xp.maximum(height-toe_height,0.)
        event_volume = p.critical_undercut_m*upper_height*face
        available = xp.sum(self._available_bulk_volume(q,area,s["bulk_density"],
                           s["solid_fraction"]), axis=0).ravel()[active]
        possible = xp.where(event_volume > 0,
            xp.floor(available/xp.maximum(event_volume,1.e-15)).astype(xp.int64), 0)
        events = xp.minimum(requested_events, possible)
        collapse_request_flat = xp.zeros(area.size, dtype=xp.float64)
        collapse_request_flat[active] = events*event_volume
        q, collapse_removed, collapse_actual, collapse_remaining = self._extract(
            q,collapse_request_flat.reshape(area.shape),area,s["bulk_density"],s["solid_fraction"])
        if self._bool(xp.any(collapse_remaining > 2.e-12)):
            raise ValueError("Q8b collapse extraction violated its finite-stock bound")
        undercut = s["undercut_m"].copy().ravel()
        undercut[active] = xp.maximum(total_undercut-events*p.critical_undercut_m,0.)
        collapse_count = s["collapse_count"].copy().ravel();collapse_count[active] += events
        q,water = self._deposit(q,water,toe_removed,active,targets,area,typ,fines_to_water=True)
        q,water = self._deposit(q,water,collapse_removed,active,targets,area,typ,fines_to_water=False)
        token = digest({"owner": "cohesive_bank_v107",
            "toe": hashlib.sha256(np.ascontiguousarray(self._host(toe_removed)).tobytes()).hexdigest(),
            "collapse": hashlib.sha256(np.ascontiguousarray(self._host(collapse_removed)).tobytes()).hexdigest()})
        if token in transfer_tokens:
            raise ValueError("Cohesive bank material transfer applied twice")
        tokens = tuple(transfer_tokens)+(token,)
        if self._cancelled(cancelled):
            raise InterruptedError("Q8b cancelled after bank proposal")
        avalanche = None
        if self.methods.avalanche and morfac > 0:
            avalanche = self.planform.advance(q,s["rock"],s["depth"],area,s["bulk_density"],
                dt_s=dt_s*morfac,dx_m=dx_m,dy_m=dy_m)
            q = avalanche["layers"]
        if self._cancelled(cancelled):
            raise InterruptedError("Q8b cancelled after parent-owned avalanche")
        fine_state = {key:(q if key == "layers" else water if key == "suspended_mass_areal" else s[key])
                      for key in FINE_STATE_KEYS}
        forcing = {} if fine_forcing is None else dict(fine_forcing)
        forbidden = set(forcing).intersection({"dt_s","time_s","morfac"})
        if forbidden:
            raise ValueError("Q8b owns fine clock fields: " + ", ".join(sorted(forbidden)))
        fine = self.fine.advance(**fine_state,dt_s=dt_s,time_s=time_s,morfac=morfac,**forcing)
        if self._cancelled(cancelled):
            raise InterruptedError("Q8b cancelled after fine exchange")
        result = {key:s[key].copy() for key in STATE_KEYS}
        for key in ("layers","suspended_mass_areal","fluff_mass_areal","solid_fraction",
                    "preload","deposit_time"):
            result[key] = fine[key].copy()
        result["undercut_m"] = undercut.reshape(area.shape)
        result["collapse_count"] = collapse_count.reshape(area.shape)
        result.update({
            # Retain cell-resolved accepted-transfer candidates for the
            # surveyed physical-toe owner.  These are diagnostics only: the
            # layer and water arrays above remain the sole mass owners.  A
            # caller can therefore accumulate signed/gross face ledgers
            # without reconstructing transfers from rounded bed changes.
            "toe_removed_mass_by_cell_kg": toe_removed.copy(),
            "collapse_removed_mass_by_cell_kg": collapse_removed.copy(),
            "toe_release_mass_kg": xp.sum(toe_removed,axis=(1,2)),
            "collapse_mass_kg": xp.sum(collapse_removed,axis=(1,2)),
            "suspended_release_mass_kg": xp.sum(xp.where((typ>0)[:,None,None],toe_removed,0.),axis=(1,2)),
            "bed_deposition_mass_kg": xp.sum(collapse_removed+xp.where((typ==0)[:,None,None],toe_removed,0.),axis=(1,2)),
            "toe_volume_m3": xp.sum(toe_actual), "collapse_volume_m3": xp.sum(collapse_actual),
            "collapse_events": xp.sum(events), "transfer_tokens": tokens,
            "fine_diagnostics": fine, "avalanche_diagnostics": avalanche,
            "method_ids": self.methods.method_ids+self.fine.methods.method_ids,
            "runtime_identity": self.identity, "dt_s": float(dt_s), "morfac": float(morfac),
        })
        return result


class CohesiveBankOwnerV107:
    def __init__(self, runtime:CohesiveBankRuntimeV107, *, time_s=0., updates=0, **state):
        self.runtime=runtime;self.time_s=float(time_s);self.updates=int(updates);self._trial=None
        self.state=runtime._validated(state)

    @property
    def state_sha256(self):
        h=hashlib.sha256();h.update(np.asarray([self.time_s,self.updates],np.float64).tobytes())
        for key in sorted(self.state):
            h.update(key.encode());h.update(np.ascontiguousarray(self.runtime._host(self.state[key])).tobytes())
        return h.hexdigest()

    def propose(self, **forcing):
        if self._trial is not None:raise RuntimeError("Q8b trial already pending")
        self._trial=self.runtime.advance(**self.state,time_s=self.time_s,**forcing);return self._trial

    def propose_from_column_state(self, layers, solid_fraction, **forcing):
        """Propose Q8b after the unpublished Q5 transport-layer update.

        ``layers`` and ``solid_fraction`` are the resident output of the
        authoritative Q5 owner.  They replace only the corresponding inputs
        of this trial; every bank/fines auxiliary field still comes from the
        accepted Q8b state.  Publication remains owned by :meth:`commit`.
        """
        if self._trial is not None:raise RuntimeError("Q8b trial already pending")
        xp=self.runtime.xp
        expected_layers=self.state["layers"].shape
        expected_solid=self.state["solid_fraction"].shape
        trial_layers=xp.ascontiguousarray(xp.asarray(layers,dtype=xp.float64))
        trial_solid=xp.ascontiguousarray(xp.asarray(solid_fraction,dtype=xp.float64))
        if trial_layers.shape!=expected_layers or trial_solid.shape!=expected_solid:
            raise ValueError("Q8b/Q5 split-process layer shapes differ")
        state=dict(self.state);state["layers"]=trial_layers
        state["solid_fraction"]=trial_solid
        self._trial=self.runtime.advance(**state,time_s=self.time_s,**forcing)
        return self._trial

    def commit(self, *, cancelled=False, accept=True):
        if self._trial is None:raise RuntimeError("No Q8b trial pending")
        trial=self._trial;self._trial=None
        if cancelled:raise InterruptedError("Q8b transaction cancelled before publication")
        if not accept:return False
        for key in STATE_KEYS:self.state[key]=trial[key].copy()
        self.time_s+=trial["dt_s"];self.updates+=1;return True

    def discard(self):self._trial=None

    def checkpoint(self):
        if self._trial is not None:raise RuntimeError("Cannot checkpoint pending Q8b trial")
        return {"schema":"braids.cohesive_bank.v107","runtime_identity":self.runtime.identity,
            "time_s":self.time_s,"updates":self.updates,
            "state":{k:self.runtime._host(v).copy() for k,v in self.state.items()},
            "state_sha256":self.state_sha256}

    def restore(self, document:Mapping):
        if document.get("schema")!="braids.cohesive_bank.v107" or \
                document.get("runtime_identity")!=self.runtime.identity:
            raise ValueError("Q8b checkpoint identity differs")
        candidate=CohesiveBankOwnerV107(self.runtime,time_s=document["time_s"],
            updates=document["updates"],**document["state"])
        if candidate.state_sha256!=document.get("state_sha256"):
            raise ValueError("Q8b checkpoint digest differs")
        self.__dict__.update(candidate.__dict__)


def remap_cohesive_bank_owner_v107(owner, offsets, sources, weights, target_area, *,
                                   bank_mask, toe_target, bank_height_m,
                                   bank_face_length_m):
    """Parent-owned AMR remap with topology rebuilt by the new mesh owner."""
    xp=owner.runtime.xp;off=np.asarray(offsets,int);src=np.asarray(sources,int);w=np.asarray(weights,float)
    old_area=np.asarray(owner.runtime._host(owner.state["area"])).ravel();ta=np.asarray(target_area,float);nt=ta.size
    if off.shape!=(nt+1,) or off[0]!=0 or off[-1]!=src.size or src.size!=w.size or \
            np.any(src<0) or np.any(src>=old_area.size) or np.any(w<0) or np.any(ta<=0):
        raise ValueError("Invalid Q8b remap CSR")
    row=np.repeat(np.arange(nt),np.diff(off))
    if not np.allclose(np.bincount(row,weights=w,minlength=nt),1.,atol=1e-13,rtol=0) or \
            not np.allclose(np.bincount(src,weights=w*ta.ravel()[row],minlength=old_area.size),
                            old_area,atol=1e-12,rtol=0):
        raise ValueError("Q8b remap is not area conservative")
    shape=ta.shape if ta.ndim==2 else (1,nt);new={};spatial=owner.state["area"].shape
    supplied={"bank_mask":bank_mask,"toe_target":toe_target,"bank_height_m":bank_height_m,
              "bank_face_length_m":bank_face_length_m}
    for key,value in owner.state.items():
        if key in supplied:continue
        a=np.asarray(owner.runtime._host(value))
        if a.shape[-2:]!=spatial:new[key]=xp.asarray(a);continue
        flat=a.reshape((-1,old_area.size));out=np.empty((flat.shape[0],nt),float)
        for t in range(nt):
            ids=src[off[t]:off[t+1]];ww=w[off[t]:off[t+1]]
            if key=="collapse_count":out[:,t]=np.max(flat[:,ids],axis=1)
            else:out[:,t]=np.sum(flat[:,ids]*ww,axis=1)
        dtype=np.int64 if key=="collapse_count" else a.dtype
        new[key]=xp.asarray(out.reshape(a.shape[:-2]+shape),dtype=dtype)
    new["area"]=xp.asarray(ta.reshape(shape))
    new.update({key:xp.asarray(value) for key,value in supplied.items()})
    result=CohesiveBankOwnerV107(owner.runtime,time_s=owner.time_s,updates=owner.updates,**new)
    for field in ("layers","suspended_mass_areal","fluff_mass_areal"):
        # Preserve the spatial axes when applying cell areas.  A flattened
        # area vector only broadcasts accidentally for one-row patches and
        # previously made genuine 2-D refinement fail before conservation
        # could even be evaluated.
        before=np.sum(owner.runtime._host(owner.state[field])*
                      old_area.reshape(spatial),axis=(-2,-1))
        after=np.sum(owner.runtime._host(result.state[field])*
                     ta.reshape(shape),axis=(-2,-1))
        if not np.allclose(before,after,atol=2e-10,rtol=0):
            raise ValueError(f"Q8b remap changed {field}")
    return result
