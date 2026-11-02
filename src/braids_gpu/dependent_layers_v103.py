"""Resident CUDA Q5b dependent layers; no CPU fallback."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np

from braids_core.sediment.bed_columns_v101 import HEADER as BASE_HEADER, array_sha, digest
from braids_core.sediment.dependent_layers_v103 import (
    DependentLayerMethodsV103, HEADER, SCOPE, SORTING_CODES, dependent_target_controls,
)
from .bed_columns_v101 import ResidentColumnRuntimeV101


class ResidentDependentLayerRuntimeV103:
    backend = "implicit_fwave_midpoint_gpu"

    def __init__(self, methods, nc, nl, **q5a_options):
        import cupy as cp
        if not isinstance(methods, DependentLayerMethodsV103):
            raise TypeError("Versioned Q5b methods required")
        base_methods = replace(methods.q5a, transport_layer="constant")
        self.cp, self.methods, self.nc, self.nl = cp, methods, nc, nl
        self.target_controls = dependent_target_controls(methods, nl)
        self.base = ResidentColumnRuntimeV101(base_methods, nc, nl, **q5a_options)
        self.rho = cp.asarray(np.ascontiguousarray(methods.density))
        self.solid_rho = cp.asarray(np.ascontiguousarray(methods.q5a.solid_density_kg_m3))
        self.diameter = cp.asarray(np.ascontiguousarray(methods.q5a.diameter_m))
        path = Path(__file__).with_suffix(".cu")
        source = ("#define BRAIDS_HD __device__\n" + BASE_HEADER.read_text() +
                  HEADER.read_text() + path.read_text())
        options = ("--std=c++11", "--fmad=false", "--prec-div=true",
                   "--prec-sqrt=true", "--ftz=false")
        self.module = cp.RawModule(code=source, options=options,
                                   name_expressions=("braids_dependent_sort_v103", "braids_dependent_targets_v103"))
        self.kernel = self.module.get_function("braids_dependent_sort_v103")
        self.receipt = {"equations": hashlib.sha256(HEADER.read_bytes()).hexdigest(),
            "entry": hashlib.sha256(path.read_bytes()).hexdigest(), "options": options,
            "device": cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(), "cupy": cp.__version__}
        self.identity = digest({"scope": SCOPE, "backend": self.backend,
            "methods": methods.fingerprint, "base": self.base.identity,
            "runtime": self.receipt, "nc": nc, "nl": nl})

    def _resident(self, name, value, shape):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.float64 or
                value.shape != shape or not value.flags.c_contiguous or
                not bool(cp.isfinite(value).all())):
            raise ValueError(f"{name} must be resident contiguous FP64 {shape}")
        return value

    def targets(self, depth, *, dune_height=None):
        cp = self.cp
        depth = self._resident("depth", cp.asarray(depth), (self.nc,))
        if self.methods.transport_layer == "dune_height" and dune_height is None:
            raise ValueError("TTLForm=3 requires accepted dune height")
        dune = cp.zeros(self.nc) if dune_height is None else self._resident(
            "dune_height", cp.asarray(dune_height), (self.nc,))
        if bool((depth < 0).any()) or bool((dune < 0).any()):
            raise ValueError("Nonnegative depth/dune fields required")
        mode, coarse, exchange, nlag, *values = self.target_controls
        out = cp.empty((self.nl, self.nc))
        self.module.get_function("braids_dependent_targets_v103")(((self.nc+127)//128,), (128,),
            (np.int32(self.nc), np.int32(self.nl), np.int32(mode), np.int32(coarse),
             np.int32(exchange), np.int32(nlag), *map(np.float64, values), depth, dune, out))
        return out

    def evaluate(self, mass, sv, delta, depth, dt, *, dune_height=None,
                 dune_length=None, shear_pa=None, bedload_mass_flux=None,
                 hiding_factor=None):
        target = self.targets(depth, dune_height=dune_height)
        return self.evaluate_targets(mass, sv, delta, target, dt,
            dune_length=dune_length, shear_pa=shear_pa,
            bedload_mass_flux=bedload_mass_flux, hiding_factor=hiding_factor)

    def evaluate_targets(self, mass, sv, delta, target, dt, *, dune_length,
                         shear_pa, bedload_mass_flux, hiding_factor, sorting_dt=None):
        cp=self.cp; m=self.methods; ng=len(m.fraction_ids)
        self._resident("target", target, (self.nl, self.nc))
        interval = dt if sorting_dt is None else sorting_dt
        if bool((target <= 0).any()) or not np.isfinite(interval) or interval < 0:
            raise ValueError("Invalid dependent target or sorting interval")
        self._resident("mass",mass,(self.nl,ng,self.nc));self._resident("sv",sv,(self.nl,self.nc));self._resident("delta",delta,(ng,self.nc))
        shape=(ng,self.nc)
        tau = cp.zeros(self.nc, dtype=cp.float64) if shear_pa is None else self._resident("shear_pa",shear_pa,(self.nc,))
        length = cp.ones(self.nc, dtype=cp.float64) if dune_length is None else self._resident("dune_length",dune_length,(self.nc,))
        bedload = cp.zeros(shape, dtype=cp.float64) if bedload_mass_flux is None else self._resident("bedload_mass_flux",bedload_mass_flux,shape)
        hiding = cp.ones(shape, dtype=cp.float64) if hiding_factor is None else self._resident("hiding_factor",hiding_factor,shape)
        out=cp.empty_like(mass);sout=cp.empty_like(sv)
        mobility=cp.empty(shape,cp.float64); transfer=cp.empty(self.nc,cp.float64)
        status=cp.empty(self.nc,cp.int32)
        self.kernel(((self.nc+127)//128,),(128,),(
            np.int32(self.nc),np.int32(ng),np.int32(self.nl),np.int32(SORTING_CODES[m.sorting]),
            np.int32(m.q5a.top_update),np.int32(m.q5a.porosity),
            np.float64(interval),np.float64(m.water_density_kg_m3),np.float64(m.gravity_m_s2),
            np.float64(m.viscosity_m2_s),np.float64(m.a_max),np.float64(m.sink_fraction_max),
            np.float64(m.asfm),np.float64(m.bsfm),np.float64(m.sigma_sfm),
            np.float64(m.q5a.solid_fraction),np.float64(m.q5a.sand_solid_fraction),
            np.float64(m.q5a.mud_solid_fraction),tau,length,target[0],self.rho,
            self.solid_rho,self.diameter,hiding,bedload,mass,sv,delta,self.base.phi,
            self.base.sigma,self.base.mud,out,sout,mobility,transfer,status))
        if bool((status != 0).any()):
            raise ValueError("Q5b CUDA sorting rejected a nonphysical state")
        final,final_sv,basal=self.base.evaluate(out,sout,cp.zeros_like(delta),target,dt,
            coarse_mobility=mobility if m.coarse_layer else None)
        return final,final_sv,basal,mobility,transfer,target


__all__ = ["ResidentDependentLayerRuntimeV103"]
