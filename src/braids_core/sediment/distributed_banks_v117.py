"""Accepted-flux distributed-slope bank owner for the v115 transaction.

This is the existing BRAIDS/Delft3D-mode-3 operator contract expressed as a
Q5-compatible owner.  The CPU path is a direct deterministic implementation
of the accepted CUDA operator; the CUDA path reuses the qualified resident
kernel.  The implicit f-wave integrator supplies one completed, equal-trace
face-mass register, so bank physics is independent of hydraulic stage count.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from collections.abc import Mapping

import numpy as np

from .bed_columns_v101 import host
from .cohesive_banks_v107 import STATE_KEYS
from .planform_v105 import (
    NativePlanformRuntimeV105, PlanformMethodsV105, PlanformParametersV105,
)


SCOPE = "braids-distributed-bank-v117"
LOCAL_FACE_SLOPE = "accepted_flux_distributed_slope"
PUBLISHED_NET_GRADIENT = "accepted_flux_published_net_gradient"
TRANSPORT_METHODS = (LOCAL_FACE_SLOPE, PUBLISHED_NET_GRADIENT)
FLOAT_HISTORY = (
    "undercut", "reference_height", "toe_height", "reference_length",
    "toe_cumulative", "collapse_cumulative",
)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("ascii")).hexdigest()


def _sha_array(*values):
    h = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(host(value))
        h.update(str(array.dtype).encode("ascii"))
        h.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        h.update(array.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class DistributedBankMethodsV117:
    transport: str = LOCAL_FACE_SLOPE
    collapse: str = "threshold_block"
    avalanche: bool = True
    integrator: str = "implicit_fwave_midpoint"

    def __post_init__(self):
        if (self.transport not in TRANSPORT_METHODS or
                self.collapse != "threshold_block" or
                type(self.avalanche) is not bool or
                self.integrator != "implicit_fwave_midpoint"):
            raise ValueError("Invalid distributed-bank v117 method bundle")

    @property
    def fingerprint(self):
        return _digest({"scope": SCOPE, **asdict(self)})


@dataclass(frozen=True)
class DistributedBankParametersV117:
    transport_fraction: float = 0.1
    critical_slope: float = 0.35
    critical_undercut_m: float = 0.05
    dry_tolerance_m: float = 1.e-6

    def __post_init__(self):
        values = np.asarray(tuple(asdict(self).values()), dtype=float)
        if (not np.isfinite(values).all() or
                not 0.0 <= self.transport_fraction <= 1.0 or
                self.critical_slope <= 0.0 or
                self.critical_undercut_m <= 0.0 or
                self.dry_tolerance_m < 0.0):
            raise ValueError("Invalid distributed-bank v117 parameters")


@dataclass(frozen=True)
class DistributedBankTopologyV117:
    area: np.ndarray
    owner: np.ndarray
    neighbour: np.ndarray
    offset: np.ndarray
    faces: np.ndarray
    signs: np.ndarray
    xy: np.ndarray
    normal: np.ndarray
    length: np.ndarray
    bank_material_mask: np.ndarray
    source_identity: str
    topology_epoch: int
    identity: str

    @classmethod
    def create(cls, area, owner, neighbour, offset, faces, signs, xy,
               normal, length, bank_material_mask, *, source_identity,
               topology_epoch=0):
        # Reuse the already-qualified geometry validator without creating a
        # CUDA object.  It returns compact int32 topology and float64 area.
        from braids_support.gpu_fwave.surface_transfer import validate_geometry
        topology, area = validate_geometry(
            area, owner, neighbour, offset, faces, signs)
        owner, neighbour, offset, faces, signs = topology
        nc, nf = area.size, owner.size
        xy = np.ascontiguousarray(xy, dtype=np.float64)
        normal = np.ascontiguousarray(normal, dtype=np.float64)
        length = np.ascontiguousarray(length, dtype=np.float64)
        mask = np.ascontiguousarray(bank_material_mask, dtype=np.int32)
        if (xy.shape != (2, nc) or normal.shape != (2, nf) or
                length.shape != (nf,) or mask.shape != (nc,) or
                not np.isfinite(xy).all() or not np.isfinite(normal).all() or
                not np.isfinite(length).all() or np.any(length <= 0.0) or
                not np.isin(mask, (0, 1)).all() or
                np.any(np.abs(np.sum(normal*normal, axis=0)-1.0) > 1.e-12) or
                type(source_identity) is not str or not source_identity or
                int(topology_epoch) < 0):
            raise ValueError("Invalid distributed-bank v117 topology")
        arrays = _sha_array(area, owner, neighbour, offset, faces, signs,
                            xy, normal, length, mask)
        identity = _digest({
            "scope": SCOPE, "source_identity": source_identity,
            "topology_epoch": int(topology_epoch), "arrays": arrays,
        })
        readonly = []
        for value in (area, owner, neighbour, offset, faces, signs, xy,
                      normal, length, mask):
            value.setflags(write=False); readonly.append(value)
        return cls(*readonly, source_identity, int(topology_epoch), identity)

    @property
    def nc(self):
        return self.area.size

    @property
    def nf(self):
        return self.owner.size


def published_net_gradient_v117(topology, bed):
    """Return the rotation-invariant extension of the published net slope.

    On an orthogonal structured grid this is exactly
    ``sqrt(S_m**2 + S_n**2)`` using positive top-to-toe slopes.  On a curved
    or locally nonorthogonal topology, the second directional component is
    projected perpendicular to the first.  Opposite neighbours therefore
    remain one logical direction while genuinely transverse neighbours add a
    second component.  Only downhill, internal neighbours participate.
    """
    t = topology
    value = np.ascontiguousarray(bed, dtype=np.float64).ravel()
    if value.shape != (t.nc,) or not np.isfinite(value).all():
        raise ValueError("Published net-gradient bed is invalid")
    result = np.zeros(t.nc, dtype=np.float64)
    for cell in range(t.nc):
        slopes: list[float] = []
        directions: list[tuple[float, float]] = []
        for slot in range(t.offset[cell], t.offset[cell + 1]):
            face = int(t.faces[slot])
            neighbour = int(t.neighbour[face] if t.owner[face] == cell
                            else t.owner[face])
            if neighbour < 0 or value[cell] <= value[neighbour]:
                continue
            delta = t.xy[:, cell] - t.xy[:, neighbour]
            distance = float(np.linalg.norm(delta))
            if not np.isfinite(distance) or distance <= 0.0:
                raise ValueError("Published net-gradient topology is invalid")
            slopes.append(float((value[cell] - value[neighbour]) / distance))
            directions.append((float(delta[0] / distance),
                               float(delta[1] / distance)))
        if not slopes:
            continue
        best_squared = max(slope * slope for slope in slopes)
        for first in range(len(slopes)):
            ux, uy = directions[first]
            for second in range(first + 1, len(slopes)):
                vx, vy = directions[second]
                transverse = abs(ux * vy - uy * vx)
                # Either component may be chosen as the primary logical
                # direction.  The max makes the extension symmetric.
                candidate = max(
                    slopes[first] ** 2 + (slopes[second] * transverse) ** 2,
                    slopes[second] ** 2 + (slopes[first] * transverse) ** 2,
                )
                best_squared = max(best_squared, candidate)
        result[cell] = math.sqrt(best_squared)
    return result


class DistributedBankRuntimeV117:
    """Backend-neutral mode-3 runtime with an exact CUDA implementation."""

    def __init__(self, topology, bulk_density, nlayer,
                 methods=DistributedBankMethodsV117(),
                 parameters=DistributedBankParametersV117(), *, gpu=False,
                 threads=4, planform_runtime=None):
        self.topology = topology
        self.bulk = np.ascontiguousarray(bulk_density, dtype=np.float64)
        self.nl = int(nlayer)
        self.ng = self.bulk.size
        if (self.bulk.ndim != 1 or not 1 <= self.ng <= 32 or
                np.any(~np.isfinite(self.bulk)) or np.any(self.bulk <= 0) or
                not 2 <= self.nl <= 33):
            raise ValueError("Invalid distributed-bank fractions/layers")
        self.methods, self.parameters = methods, parameters
        self.gpu = bool(gpu)
        self.threads = int(threads)
        self.backend = ("implicit_fwave_midpoint_gpu" if self.gpu
                        else "native_cpu_reference_v117")
        self.xp = np
        self.operator = None
        if self.gpu:
            import cupy as cp
            from braids_support.gpu_fwave.distributed_bank import (
                ResidentDistributedBank,
            )
            self.xp = cp
            t = topology; p = parameters
            self.operator = ResidentDistributedBank(
                t.area, t.owner, t.neighbour, t.offset, t.faces, t.signs,
                t.xy, t.normal, t.length, self.bulk,
                t.bank_material_mask, self.nl,
                transport_fraction=p.transport_fraction,
                critical_slope=p.critical_slope,
                critical_undercut=p.critical_undercut_m,
                dry_tolerance=p.dry_tolerance_m,
                slope_method=methods.transport)
        if methods.avalanche and planform_runtime is None:
            if self.gpu:
                from braids_gpu.planform_v105 import (
                    ResidentPlanformRuntimeV105,
                )
                planform_runtime = ResidentPlanformRuntimeV105(
                    PlanformMethodsV105(
                        "braids_conservative_wet_dry_avalanche"),
                    PlanformParametersV105())
            else:
                planform_runtime = NativePlanformRuntimeV105(
                    PlanformMethodsV105(
                        "braids_conservative_wet_dry_avalanche"),
                    PlanformParametersV105(), threads=self.threads)
        self.planform = planform_runtime
        if methods.avalanche and (
                self.planform is None or self.planform.xp is not self.xp):
            raise ValueError("Distributed bank and avalanche backends differ")
        self.identity = _digest({
            "scope": SCOPE, "backend": self.backend,
            "topology": topology.identity, "bulk": self.bulk.tolist(),
            "nlayer": self.nl, "methods": methods.fingerprint,
            "parameters": asdict(parameters),
            "planform": None if self.planform is None
            else self.planform.identity,
        })

    def _validated_state(self, state):
        missing = [key for key in STATE_KEYS if key not in state]
        if missing:
            raise ValueError("Distributed-bank state is missing " +
                             ", ".join(missing))
        xp = self.xp; t = self.topology
        result = {key: xp.asarray(state[key]).copy() for key in STATE_KEYS}
        layers = xp.ascontiguousarray(
            xp.asarray(state["layers"], dtype=xp.float64))
        if (layers.ndim != 4 or layers.shape[0] != self.nl or
                layers.shape[1] != self.ng or
                layers.shape[2]*layers.shape[3] != t.nc or
                bool(xp.any(~xp.isfinite(layers))) or
                bool(xp.any(layers < 0.0))):
            raise ValueError("Invalid distributed-bank layer inventory")
        shape = layers.shape[2:]; result["layers"] = layers
        expected = {
            "area": shape, "depth": shape, "shear_pa": shape,
            "rock": shape, "bank_mask": shape, "toe_target": shape,
            "bank_height_m": shape, "bank_face_length_m": shape,
            "undercut_m": shape, "collapse_count": shape,
            "solid_fraction": (self.nl,)+shape,
            "preload": (self.nl,)+shape,
            "deposit_time": (self.nl,)+shape,
            "suspended_mass_areal": (self.ng,)+shape,
            "fluff_mass_areal": (self.ng,)+shape,
        }
        for key, required in expected.items():
            dtype = xp.int64 if key == "collapse_count" else (
                xp.bool_ if key == "bank_mask" else xp.float64)
            if key == "toe_target": dtype = xp.int64
            value = xp.ascontiguousarray(xp.asarray(state[key], dtype=dtype))
            if value.shape != required:
                raise ValueError("Distributed-bank state shape differs: "+key)
            result[key] = value
        if (not np.array_equal(host(result["area"]).ravel(), t.area) or
                not np.array_equal(
                    host(result["bank_mask"]).ravel().astype(np.int32),
                    t.bank_material_mask) or
                np.any(host(result["depth"]) < 0.0) or
                np.any(host(result["solid_fraction"]) <= 0.0) or
                np.any(host(result["solid_fraction"]) > 1.0) or
                np.any(host(result["suspended_mass_areal"]) != 0.0) or
                np.any(host(result["fluff_mass_areal"]) != 0.0)):
            raise ValueError(
                "Mode 3 requires matched geometry and suspension exchange off")
        bulk = np.asarray(host(state["bulk_density"]), dtype=np.float64)
        types = np.asarray(host(state["sediment_type"]), dtype=np.int32)
        if (bulk.shape != (self.ng,) or not np.array_equal(bulk, self.bulk) or
                types.shape != (self.ng,) or np.any(types != 0)):
            raise ValueError("Mode 3 requires noncohesive fractions")
        result["bulk_density"] = xp.asarray(bulk)
        result["sediment_type"] = xp.asarray(types)
        return result

    def _initial_history(self):
        xp = self.xp; nc = self.topology.nc
        result = {key: xp.zeros(nc, xp.float64) for key in FLOAT_HISTORY}
        result["collapse_count"] = xp.zeros(nc, xp.int32)
        return result

    @staticmethod
    def _surface_transfer(active, proposed, t):
        ng, nc = active.shape; nf = t.nf
        scale = np.ones((ng, nc), dtype=np.float64)
        for g in range(ng):
            for c in range(nc):
                outgoing = 0.0
                for k in range(t.offset[c], t.offset[c+1]):
                    outgoing += max(t.signs[k]*proposed[g, t.faces[k]], 0.0)
                available = active[g, c]*t.area[c]
                if outgoing > available and outgoing > 0.0:
                    scale[g, c] = max(0.0, available/outgoing)*(
                        1.0-64.0*np.finfo(float).eps)
        limited = np.zeros((ng, nf), dtype=np.float64)
        for g in range(ng):
            for f in range(nf):
                value = proposed[g, f]
                donor = t.owner[f] if value > 0.0 else t.neighbour[f]
                if value != 0.0:
                    limited[g, f] = value*scale[g, donor]
        next_active = active.copy()
        for g in range(ng):
            for c in range(nc):
                change = sum(-t.signs[k]*limited[g, t.faces[k]]
                             for k in range(t.offset[c], t.offset[c+1]))
                next_active[g, c] += change/t.area[c]
                if (next_active[g, c] < 0.0 and
                        abs(next_active[g, c])*t.area[c] <=
                        128.0*np.finfo(float).eps):
                    next_active[g, c] = 0.0
        return next_active, limited

    def _advance_cpu(self, layers, depth, bed, history, integral, dt_s):
        t = self.topology; p = self.parameters
        ng, nc, nf = self.ng, t.nc, t.nf
        mass = integral[:, 0, :]
        q = np.zeros((2, ng, nc), dtype=np.float64)
        rank = np.zeros(nc, dtype=np.int32)
        for c in range(nc):
            mxx = mxy = myy = 0.0
            for k in range(t.offset[c], t.offset[c+1]):
                f = t.faces[k]; nx, ny = t.normal[:, f]; length = t.length[f]
                mxx += length*nx*nx; mxy += length*nx*ny
                myy += length*ny*ny
            determinant = mxx*myy-mxy*mxy
            if determinant <= 1.e-14*max(mxx*myy, 1.0):
                rank[c] = 1; continue
            for g in range(ng):
                bx = by = 0.0
                for k in range(t.offset[c], t.offset[c+1]):
                    f = t.faces[k]; r = mass[g, f]/dt_s
                    bx += t.normal[0, f]*r; by += t.normal[1, f]*r
                q[0, g, c] = (myy*bx-mxy*by)/determinant
                q[1, g, c] = (mxx*by-mxy*bx)/determinant
        net_slope = (published_net_gradient_v117(t, bed)
                     if self.methods.transport == PUBLISHED_NET_GRADIENT
                     else np.zeros(nc, dtype=np.float64))
        proposal = np.zeros((ng, nf)); eligible = np.zeros(nf, np.int32)
        donor = np.full(nf, -1, np.int32); recipient = donor.copy()
        tangent = np.zeros(nf); local_slope = np.zeros(nf)
        active = layers[0]
        for f in range(nf):
            o, n = int(t.owner[f]), int(t.neighbour[f])
            if n < 0 or bed[o] == bed[n]: continue
            high, low, orientation = ((o, n, 1) if bed[o] > bed[n]
                                      else (n, o, -1))
            delta = t.xy[:, high]-t.xy[:, low]
            distance = float(np.linalg.norm(delta))
            slope = (bed[high]-bed[low])/distance
            local_slope[f] = slope
            criterion_slope = (net_slope[high]
                               if self.methods.transport == PUBLISHED_NET_GRADIENT
                               else slope)
            if (not t.bank_material_mask[high] or
                    criterion_slope <= p.critical_slope):
                continue
            eligible[f] = 1; donor[f] = high; recipient[f] = low
            total = np.sum(q[:, :, low], axis=1)
            nx, ny = t.normal[:, f]
            transport = abs(-ny*total[0]+nx*total[1])
            tangent[f] = transport
            total_active = np.sum(active[:, high])
            if total_active > 0.0:
                amount = p.transport_fraction*dt_s*t.length[f]*transport
                proposal[:, f] = orientation*amount*active[:, high]/total_active
        next_active, limited = self._surface_transfer(active, proposal, t)
        out = layers.copy(); out[0] = next_active
        h = {key: np.asarray(host(value)).copy() for key, value in history.items()}
        eps = np.finfo(float).eps
        for c in range(nc):
            exposed = False; request = weight = down_length = down_height = 0.0
            toe_max = h["toe_height"][c]
            cumulative = h["toe_cumulative"][c]
            tolerance = 512.0*eps*max(abs(bed[c]), 1.0)
            for k in range(t.offset[c], t.offset[c+1]):
                f = t.faces[k]
                if not eligible[f] or donor[f] != c: continue
                r = recipient[f]; height = max(bed[c]-bed[r], 0.0)
                wet = min(max(depth[r], 0.0), height)
                above = max(height-wet, 0.0)
                exposed = exposed or above > tolerance
                volume = np.sum(np.abs(limited[:, f])/self.bulk)
                cumulative += volume/t.area[c]
                if volume <= 0.0 or above <= tolerance or wet <= p.dry_tolerance_m:
                    continue
                request += volume; weight += t.length[f]*wet
                down_length += t.length[f]
                down_height = max(down_height, height); toe_max = max(toe_max, wet)
            if not exposed:
                for key in ("undercut", "reference_height", "toe_height",
                            "reference_length"): h[key][c] = 0.0
            elif request > 0.0:
                h["undercut"][c] += request/weight
                h["reference_height"][c] = max(
                    h["reference_height"][c], down_height)
                h["reference_length"][c] = max(
                    h["reference_length"][c], down_length)
                h["toe_height"][c] = toe_max
            h["toe_cumulative"][c] = cumulative
        next_bed = bed+np.sum((out[0]-active)/self.bulk[:, None], axis=0)
        collapse_removed = np.zeros((ng, nc)); collapse_gain = np.zeros((ng, nc))
        collapse_events = 0
        # Ascending donor order is the native deterministic conflict rule.
        for c in range(nc):
            if h["undercut"][c] < p.critical_undercut_m: continue
            candidates = [int(t.faces[k]) for k in range(t.offset[c], t.offset[c+1])
                          if eligible[t.faces[k]] and donor[t.faces[k]] == c]
            if not candidates: continue
            face = max(candidates, key=lambda f: (tangent[f]*t.length[f], -f))
            r = int(recipient[face])
            height = max(next_bed[c]-next_bed[r], 0.0)
            above = max(height-h["toe_height"][c], 0.0)
            if above <= 0.0 or h["reference_length"][c] <= 0.0: continue
            events = int(h["undercut"][c]/p.critical_undercut_m)
            request = events*p.critical_undercut_m*h["reference_length"][c]*above
            request = min(request, height/(1.0/t.area[c]+1.0/t.area[r]))
            remaining = request; actual = 0.0
            for level in range(self.nl):
                if remaining <= 64.0*eps*max(request, 1.0): break
                volume = np.sum(out[level, :, c]*t.area[c]/self.bulk)
                if volume <= 0.0: continue
                take = min(remaining, volume); factor = min(1.0, take/volume)
                amount = out[level, :, c]*factor
                out[level, :, c] -= amount
                collapse_removed[:, c] += amount*t.area[c]
                actual += take; remaining = max(0.0, remaining-take)
            if actual <= 0.0: continue
            out[0, :, r] += collapse_removed[:, c]/t.area[r]
            collapse_gain[:, r] += collapse_removed[:, c]
            next_bed[c] -= actual/t.area[c]; next_bed[r] += actual/t.area[r]
            h["collapse_cumulative"][c] += actual/t.area[c]
            h["collapse_count"][c] += events
            h["undercut"][c] = max(
                0.0, h["undercut"][c]-events*p.critical_undercut_m)
            collapse_events += events
            if h["undercut"][c] <= 64.0*eps*p.critical_undercut_m:
                for key in ("undercut", "reference_height", "toe_height",
                            "reference_length"): h[key][c] = 0.0
        residual = np.sum(collapse_removed-collapse_gain, axis=1)
        return {"layers": out, "history": h, "limited": limited,
                "rank_deficient": rank, "collapse_events": collapse_events,
                "collapse_budget_residual_kg": residual,
                "bed": next_bed, "eligible": eligible, "donor": donor,
                "recipient": recipient, "local_slope": local_slope,
                "net_slope": net_slope, "tangential": tangent}

    def advance(self, state, history, *, dt_s, accepted_face_integral_kg,
                dx_m=1.0, dy_m=1.0, cancelled=False):
        if not np.isfinite([dt_s, dx_m, dy_m]).all() or dt_s <= 0.0 or \
                min(dx_m, dy_m) <= 0.0:
            raise ValueError("Invalid distributed-bank clock/grid")
        s = self._validated_state(state); xp = self.xp; t = self.topology
        integral = xp.ascontiguousarray(xp.asarray(
            accepted_face_integral_kg, dtype=xp.float64))
        if (integral.shape != (self.ng, 2, t.nf) or
                bool(xp.any(~xp.isfinite(integral))) or
                not bool(xp.all(integral[:, 0] == integral[:, 1]))):
            raise ValueError(
                "Distributed bank requires finite equal-trace face integral")
        if bool(cancelled() if callable(cancelled) else cancelled):
            raise InterruptedError("Distributed-bank proposal cancelled")
        shape = s["area"].shape
        layers = s["layers"].reshape((self.nl, self.ng, t.nc))
        depth = s["depth"].ravel()
        bed = s["rock"].ravel()+xp.sum(
            layers/xp.asarray(self.bulk)[None, :, None], axis=(0, 1))
        if self.gpu:
            op = self.operator
            op.begin_trial(); op.set_accepted_face_integral(integral)
            result = op.step(layers, depth, bed, history, dt_s)
            if int(result["status"].item()):
                raise ValueError("CUDA distributed-bank trial rejected")
            bank_layers = result["trial_layers"].copy()
            next_history = {key: value.copy()
                            for key, value in result["trial_history"].items()}
            events = int(xp.sum(
                next_history["collapse_count"]-
                history["collapse_count"]).item())
            residual = result["collapse_budget"][:, 0]-result[
                "collapse_budget"][:, 1]
            limited = result["limited"].copy()
            eligible = result["eligible"].copy()
            donor = result["donor"].copy()
            recipient = result["recipient"].copy()
            local_slope = result["local_slope"].copy()
            net_slope = result["net_slope"].copy()
            tangential = result["tangential"].copy()
        else:
            result = self._advance_cpu(
                np.asarray(layers), np.asarray(depth), np.asarray(bed),
                history, np.asarray(integral), dt_s)
            bank_layers = result["layers"]
            next_history = result["history"]
            events = int(result["collapse_events"])
            residual = result["collapse_budget_residual_kg"]
            limited = result["limited"]
            eligible = result["eligible"]
            donor = result["donor"]
            recipient = result["recipient"]
            local_slope = result["local_slope"]
            net_slope = result["net_slope"]
            tangential = result["tangential"]
        if self.methods.avalanche:
            avalanche = self.planform.advance(
                bank_layers.reshape((self.nl, self.ng)+shape),
                s["rock"], s["depth"], s["area"], s["bulk_density"],
                dt_s=dt_s, dx_m=dx_m, dy_m=dy_m)
            bank_layers = avalanche["layers"].reshape(
                (self.nl, self.ng, t.nc))
        else:
            avalanche = None
        if bool(cancelled() if callable(cancelled) else cancelled):
            raise InterruptedError("Distributed-bank proposal cancelled")
        out = {key: value.copy() for key, value in s.items()}
        out["layers"] = bank_layers.reshape((self.nl, self.ng)+shape)
        out.update({
            "history": next_history,
            "accepted_face_mass_kg": integral[:, 0].copy(),
            "limited_bank_mass_kg": limited.copy(),
            "bank_eligible_faces": eligible.copy(),
            "bank_donor_cells": donor.copy(),
            "bank_recipient_cells": recipient.copy(),
            "bank_local_face_slope": local_slope.copy(),
            "bank_net_cell_slope": net_slope.copy(),
            "bank_tangential_transport_kg_m_s": tangential.copy(),
            "collapse_events": xp.asarray(events, dtype=xp.int64),
            "collapse_budget_residual_kg": xp.asarray(residual),
            "avalanche_diagnostics": avalanche,
            "dt_s": float(dt_s), "runtime_identity": self.identity,
        })
        return out


class DistributedBankOwnerV117:
    """Q5-compatible accepted/trial owner for distributed mode 3."""

    def __init__(self, runtime, *, time_s=0.0, updates=0, history=None,
                 **state):
        self.runtime = runtime; self.time_s = float(time_s)
        self.updates = int(updates); self._trial = None
        self.state = runtime._validated_state(state)
        self.history = (runtime._initial_history() if history is None else
                        self._validated_history(history))

    def _validated_history(self, value):
        xp = self.runtime.xp; nc = self.runtime.topology.nc
        if set(value) != set(FLOAT_HISTORY+("collapse_count",)):
            raise ValueError("Incomplete distributed-bank history")
        result = {}
        for key in FLOAT_HISTORY+("collapse_count",):
            dtype = xp.int32 if key == "collapse_count" else xp.float64
            item = xp.ascontiguousarray(xp.asarray(value[key], dtype=dtype))
            if (item.shape != (nc,) or bool(xp.any(item < 0)) or
                    (key != "collapse_count" and
                     bool(xp.any(~xp.isfinite(item))))):
                raise ValueError("Invalid distributed-bank history: "+key)
            result[key] = item.copy()
        return result

    @property
    def state_sha256(self):
        arrays = [np.asarray([self.time_s, self.updates], np.float64)]
        arrays += [self.state[key] for key in sorted(self.state)]
        arrays += [self.history[key] for key in sorted(self.history)]
        return _sha_array(*arrays)

    def propose_from_column_state(self, layers, solid_fraction, **forcing):
        if self._trial is not None:
            raise RuntimeError("Distributed-bank trial already pending")
        xp = self.runtime.xp
        state = dict(self.state)
        state["layers"] = xp.ascontiguousarray(xp.asarray(
            layers, dtype=xp.float64))
        state["solid_fraction"] = xp.ascontiguousarray(xp.asarray(
            solid_fraction, dtype=xp.float64))
        self._trial = self.runtime.advance(
            state, self.history, **forcing)
        return self._trial

    def commit(self, *, cancelled=False, accept=True):
        if self._trial is None:
            raise RuntimeError("No distributed-bank trial pending")
        trial = self._trial; self._trial = None
        if cancelled:
            raise InterruptedError(
                "Distributed-bank transaction cancelled before publication")
        if not accept:
            return False
        for key in STATE_KEYS:
            self.state[key] = trial[key].copy()
        self.history = {key: value.copy()
                        for key, value in trial["history"].items()}
        self.time_s += trial["dt_s"]; self.updates += 1
        return True

    def discard(self):
        self._trial = None

    def checkpoint(self):
        if self._trial is not None:
            raise RuntimeError("Cannot checkpoint pending distributed bank")
        raw = {
            "schema": "braids.distributed_bank.v117",
            "runtime_identity": self.runtime.identity,
            "time_s": self.time_s, "updates": self.updates,
            "state": {key: host(value).tolist()
                      for key, value in self.state.items()},
            "history": {key: host(value).tolist()
                        for key, value in self.history.items()},
            "state_sha256": self.state_sha256,
        }
        return {**raw, "checkpoint_sha256": _digest(raw)}

    def restore(self, value: Mapping):
        raw = {key: item for key, item in value.items()
               if key != "checkpoint_sha256"}
        if (_digest(raw) != value.get("checkpoint_sha256") or
                value.get("schema") != "braids.distributed_bank.v117" or
                value.get("runtime_identity") != self.runtime.identity):
            raise ValueError("Distributed-bank checkpoint identity differs")
        candidate = DistributedBankOwnerV117(
            self.runtime, time_s=value["time_s"], updates=value["updates"],
            history=value["history"], **value["state"])
        if candidate.state_sha256 != value.get("state_sha256"):
            raise ValueError("Distributed-bank checkpoint digest differs")
        self.__dict__.update(candidate.__dict__)
        return self


def remap_distributed_bank_owner_v117(owner, target_runtime, offsets,
                                      indices, weights, target_area):
    """Area-conservative mode-3 remap; target topology is rebuilt, not copied."""
    if owner._trial is not None:
        raise ValueError("Distributed-bank remap requires committed state")
    if (owner.runtime.methods.fingerprint !=
            target_runtime.methods.fingerprint or
            owner.runtime.parameters != target_runtime.parameters or
            owner.runtime.backend != target_runtime.backend or
            owner.runtime.nl != target_runtime.nl or
            not np.array_equal(owner.runtime.bulk, target_runtime.bulk)):
        raise ValueError("Distributed-bank regrid cannot change physics/backend")
    original_target = np.asarray(target_area, dtype=np.float64)
    area = np.ascontiguousarray(original_target.ravel())
    off = np.ascontiguousarray(offsets, dtype=np.int32)
    src = np.ascontiguousarray(indices, dtype=np.int32)
    w = np.ascontiguousarray(weights, dtype=np.float64)
    old_area = owner.runtime.topology.area
    nc = target_runtime.topology.nc
    if (area.shape != (nc,) or not np.array_equal(
            area, target_runtime.topology.area) or
            off.shape != (nc+1,) or off[0] != 0 or off[-1] != src.size or
            w.shape != src.shape or np.any(np.diff(off) <= 0) or
            np.any(src < 0) or np.any(src >= old_area.size) or
            np.any(w < 0.0) or not np.isfinite(w).all()):
        raise ValueError("Invalid distributed-bank conservative remap")
    row = np.repeat(np.arange(nc), np.diff(off))
    if (not np.allclose(np.bincount(row, weights=w), 1.0,
                        rtol=0.0, atol=1.e-13) or
            not np.allclose(np.bincount(
                src, weights=w*area[row], minlength=old_area.size),
                old_area, rtol=0.0, atol=1.e-12)):
        raise ValueError("Distributed-bank remap does not conserve area")
    xp = target_runtime.xp
    spatial = owner.state["area"].shape
    target_shape = (original_target.shape if original_target.ndim == 2
                    else (1, nc))

    def mapped(value, *, maximum=False, dtype=None):
        array = np.asarray(host(value))
        if array.shape[-2:] != spatial:
            return xp.asarray(array, dtype=dtype)
        flat = array.reshape((-1, old_area.size))
        out = np.empty((flat.shape[0], nc), dtype=np.float64)
        for target in range(nc):
            ids = src[off[target]:off[target+1]]
            ww = w[off[target]:off[target+1]]
            out[:, target] = (np.max(flat[:, ids], axis=1) if maximum
                              else np.sum(flat[:, ids]*ww, axis=1))
        return xp.asarray(out.reshape(array.shape[:-2]+target_shape),
                          dtype=dtype or array.dtype)

    state = {}
    for key, value in owner.state.items():
        if key in ("area", "bank_mask", "toe_target",
                   "bank_face_length_m"):
            continue
        state[key] = mapped(
            value, maximum=(key == "collapse_count"),
            dtype=(xp.int64 if key == "collapse_count" else None))
    state["area"] = xp.asarray(area.reshape(target_shape))
    state["bank_mask"] = xp.asarray(
        target_runtime.topology.bank_material_mask.reshape(target_shape),
        dtype=xp.bool_)
    state["toe_target"] = xp.full(target_shape, -1, dtype=xp.int64)
    lengths = np.ones(nc, dtype=np.float64)
    for cell in np.flatnonzero(target_runtime.topology.bank_material_mask):
        incident = target_runtime.topology.faces[
            target_runtime.topology.offset[cell]:
            target_runtime.topology.offset[cell+1]]
        if incident.size:
            lengths[cell] = np.sum(target_runtime.topology.length[incident])
    state["bank_face_length_m"] = xp.asarray(lengths.reshape(target_shape))
    history = {}
    for key, value in owner.history.items():
        flat = np.asarray(host(value)).reshape(-1)
        out = np.empty(nc, dtype=np.float64)
        for target in range(nc):
            ids = src[off[target]:off[target+1]]
            ww = w[off[target]:off[target+1]]
            out[target] = (np.max(flat[ids]) if key == "collapse_count"
                           else np.sum(flat[ids]*ww))
        history[key] = xp.asarray(
            out, dtype=xp.int32 if key == "collapse_count" else xp.float64)
    result = DistributedBankOwnerV117(
        target_runtime, time_s=owner.time_s, updates=owner.updates,
        history=history, **state)
    for field in ("layers", "suspended_mass_areal", "fluff_mass_areal"):
        before = np.sum(host(owner.state[field])*old_area.reshape(spatial),
                        axis=(-2, -1))
        after = np.sum(host(result.state[field])*area.reshape(target_shape),
                       axis=(-2, -1))
        if not np.allclose(before, after, rtol=0.0, atol=2.e-10):
            raise ValueError("Distributed-bank remap changed "+field)
    return result


__all__ = [
    "SCOPE", "FLOAT_HISTORY", "LOCAL_FACE_SLOPE",
    "PUBLISHED_NET_GRADIENT", "TRANSPORT_METHODS",
    "published_net_gradient_v117", "DistributedBankMethodsV117",
    "DistributedBankParametersV117", "DistributedBankTopologyV117",
    "DistributedBankRuntimeV117", "DistributedBankOwnerV117",
    "remap_distributed_bank_owner_v117",
]
