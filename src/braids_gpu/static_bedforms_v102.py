"""Resident CUDA Q7a static bedform/capacity route; no CPU fallback."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import numpy as np

from braids_core.sediment.static_bedforms_v102 import (
    HEADER, StaticMethodsV102, StaticParametersV102, _digest, _modes,
    resolve_static_methods_v102,
)


class ResidentStaticRuntimeV102:
    backend = "implicit_fwave_midpoint_gpu"

    def __init__(self, methods: StaticMethodsV102, parameters=StaticParametersV102()):
        import cupy as cp
        self.cp = cp; self.resolved = resolve_static_methods_v102(methods); self.parameters = parameters
        self.mode_host = _modes(self.resolved, parameters); self.params_host = parameters.wire()
        self.mode = cp.asarray(self.mode_host); self.params = cp.asarray(self.params_host)
        path = Path(__file__).with_suffix(".cu")
        source = "#define BRAIDS_HD __device__\n" + HEADER.read_text() + path.read_text()
        options = ("--std=c++11", "--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false")
        self.module = cp.RawModule(code=source, options=options, name_expressions=(
            "braids_static_bedforms_v102", "braids_static_capacity_v102"))
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        self.receipt = {"equations_sha256": hashlib.sha256(HEADER.read_bytes()).hexdigest(),
                        "entry_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "device": props["name"].decode(), "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                        "cupy": cp.__version__, "options": options}
        self.identity = _digest({"backend": self.backend, "methods": self.resolved.fingerprint,
                                 "parameters": asdict(parameters), "runtime": self.receipt})

    def _field(self, value, nc, name, *, positive=False, nonnegative=False):
        cp = self.cp
        if isinstance(value, cp.ndarray):
            a = value
            if a.shape == (): a = cp.full(nc, a, cp.float64)
            elif a.shape != (nc,): raise ValueError(f"Invalid CUDA {name} shape")
            if a.dtype != cp.float64 or not a.flags.c_contiguous: a = cp.ascontiguousarray(a, dtype=cp.float64)
        else:
            a = cp.asarray(np.broadcast_to(np.asarray(value, dtype=np.float64), (nc,)).copy())
        if not bool(cp.isfinite(a).all()) or (positive and bool((a <= 0).any())) or (nonnegative and bool((a < 0).any())):
            raise ValueError(f"Invalid CUDA {name}")
        return a

    def bedforms(self, depth, u, v, chezy, d50, d90):
        cp = self.cp; nc = int(depth.size)
        values = ((depth, "depth", True), (u, "u", False), (v, "v", False),
                  (chezy, "chezy", True), (d50, "d50", True), (d90, "d90", True))
        fields = [self._field(x, nc, n, positive=p) for x, n, p in values]
        out = cp.empty((9, nc), cp.float64); status = cp.empty(nc, cp.int32)
        self.module.get_function("braids_static_bedforms_v102")(((nc+127)//128,), (128,),
            (np.int32(nc), self.mode, self.params, *fields, out, status))
        if bool((status != 0).any()): raise ValueError("Nonphysical CUDA Q7a bedform cell")
        names = ("height", "length", "ripple_roughness", "mega_ripple_roughness",
                 "dune_roughness", "total_roughness", "critical_shields", "mobility", "dstar")
        return {name: out[i] for i, name in enumerate(names)}

    def capacity(self, depth, u, v, d10, d50, d90, roughness, *, skin_roughness=None, mud_fraction=0.,
                 concentration_kg_m3=0., salinity=0., slope_x=0., slope_y=0.):
        cp = self.cp; nc = int(depth.size)
        skin_roughness = roughness if skin_roughness is None else skin_roughness
        values = ((depth, "depth", True, False), (u, "u", False, False), (v, "v", False, False),
                  (d10, "d10", True, False), (d50, "d50", True, False), (d90, "d90", True, False),
                  (roughness, "roughness", True, False), (skin_roughness, "skin_roughness", True, False),
                  (mud_fraction, "mud", False, True),
                  (concentration_kg_m3, "concentration", False, True), (salinity, "salinity", False, True),
                  (slope_x, "slope_x", False, False), (slope_y, "slope_y", False, False))
        fields = [self._field(x, nc, n, positive=p, nonnegative=nn) for x, n, p, nn in values]
        out = cp.empty((10, nc), cp.float64); status = cp.empty(nc, cp.int32)
        self.module.get_function("braids_static_capacity_v102")(((nc+127)//128,), (128,),
            (np.int32(nc), self.mode, self.params, *fields, out, status))
        if bool((status != 0).any()): raise ValueError("Nonphysical CUDA Q7a capacity cell")
        names = ("qx_kg_m_s", "qy_kg_m_s", "reference_concentration", "settling_diameter_m",
                 "settling_velocity_m_s", "stress_ratio", "transport_shear", "reference_height_m",
                 "magnitude_kg_m_s", "critical_shields")
        return {name: out[i] for i, name in enumerate(names)}


__all__ = ["ResidentStaticRuntimeV102"]
