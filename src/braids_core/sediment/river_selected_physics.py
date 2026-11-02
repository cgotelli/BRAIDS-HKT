"""Bind resolved physical switches to the one common river owner."""
from types import MethodType

from braids_core.river_model import verify_resolved
from .bend_methods_v1 import digest


def install(a, resolved):
    doc = verify_resolved(resolved).document()
    if a.time or a.steps or a.physics._open:
        raise ValueError("Select the river model before execution")
    p = a.physics
    h, s = doc["physics"]["hydrodynamics"], doc["physics"]["sediment"]
    a.morphology = s is not None
    if h["secondary_flow"] == "secondary.off":
        previous = p.local
        def local(owner, *args):
            out = previous(*args)
            # All spiral sources, equilibrium values and stress coefficients;
            # retain the hydraulic Chezy row, including selected bedforms.
            out[:9] = 0.
            out[10:] = 0.
            return out
        p.local = MethodType(local, p)
        if h["turbulence"] == "off":
            p.add_momentum = MethodType(lambda owner, q, water, rate: None, p)
    if s is not None and s["bed_slope"] == "bed_slope.off":
        def slope_off(capacity, fields):
            if fields.ndim != 2 or fields.shape[0] != 10:
                raise ValueError("Ten slope-input rows required")
            # Inputs 0:2 are already reconstructed/secondary-directed capacity.
            # Do not overwrite those directions or invoke the inherited KF law.
            return fields[:2].copy()
        for capacity in a.capacity:
            capacity.slope = MethodType(slope_off, capacity)
            capacity.identity["slope"] = "bed_slope.off"
    if s is not None and a.columns.composition.method != s["composition_fraction"]:
        raise ValueError("Column composition differs from resolved physical fraction identity")
    from .river_column_methods import topology
    p.identity["river_mixtures"]["layer_roles"] = topology(
        a.methods, a.columns.runtime.nl, coarse=bool(s and s["bed"]["coarse_layer"]))
    p.identity["resolved_river_model"] = dict(model=doc, binding_sha256=digest(__file__))
    a.identity["resolved_river_model"] = dict(model=doc, binding_sha256=digest(__file__))
    p.identity["river_mixtures"]["capacity"] = [c.identity for c in a.capacity]
    return a
