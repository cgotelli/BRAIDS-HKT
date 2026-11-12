"""Dependent layers within the one continuous river column transaction."""
from dataclasses import asdict
from pathlib import Path
from types import MethodType
import numpy as np

from .bed_columns_v101 import digest
from .bend_methods_v1 import digest as file_digest
from .dependent_layers_v103 import DependentLayerMethodsV103, NativeDependentLayerRuntimeV103
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine

PARAMETERS = dict(coarse_thickness_m=.02, a_max=20., sink_fraction_max=.05,
                  asfm=.523, bsfm=.67, sigma_sfm=.2, viscosity_m2_s=1e-6,
                  hiding_scope="active")


def methods_for(columns, bed, parameters):
    if set(parameters) != set(PARAMETERS):
        raise ValueError("Explicit complete dependent-layer parameters required")
    return DependentLayerMethodsV103(columns, coarse_layer=bed["coarse_layer"],
        sorting=bed["sorting"].removeprefix("sorting."),
        transport_layer={"transport_layer.constant": "constant",
                         "transport_layer.water_depth": "depth",
                         "transport_layer.dune_height": "dune_height"}[bed["transport_layer"]],
        ttl_alpha=columns.depth_fraction, ttl_min_m=columns.minimum_thickness_m, **parameters)


class SortingForcing:
    def __init__(self, backend, threads=4):
        self.engine = WaveEngine(backend, threads, source=Path(__file__).with_suffix(".h"),
                                names=("sorting_face_stress", "sorting_cell_forcing"))
        self.identity = self.engine.identity

    def evaluate(self, q, z, gradient, chezy, geometry, flux, face, shear, bedload, dry, solid_density):
        g = geometry; n, nf, ng = q.shape[1], flux.shape[1], flux.shape[0]
        self.engine.launch("sorting_face_stress", nf, nf, n, float(dry), 1000., 9.81,
            q, z, gradient, chezy, g["owner"], g["neighbour"], g["bid"],
            g["distance"], g["displacement"], face)
        self.engine.launch("sorting_cell_forcing", n, n, nf, ng, solid_density, float(dry),
            q, face, g["normal"], g["length"], g["offset"], g["faces"], flux, shear, bedload)


class RiverDependentColumns:
    """Runtime adapter, not a second inventory or accepted-state owner."""
    def __init__(self, base, process, physics, bedforms_enabled):
        self.base, self.process, self.physics = base, process, physics
        self.methods, self.nc, self.nl, self.rho = base.methods, base.nc, base.nl, base.rho
        self.backend = base.backend
        if hasattr(base, "cp"):
            self.cp = base.cp
        self.bedforms_enabled = bedforms_enabled
        self.receipt = dict(base=base.receipt, dependent=process.receipt)
        self.identity = digest(dict(scope="river-dependent-columns-v1", base=base.identity,
            dependent=process.identity, binding=file_digest(__file__),
            bedforms_enabled=bedforms_enabled))

    def targets(self, depth):
        return self.process.targets(depth, dune_height=self.physics.bedform_state[0])

    def evaluate(self, mass, sv, delta, targets, dt, *, sorting_dt=0.):
        p = self.physics
        xp = getattr(self, "cp", np)
        # The no-bedform value is the explicit FLOW bott3d fallback, not the
        # old standalone helper's one-metre default.
        length = (xp.ascontiguousarray(xp.maximum(p.bedform_state[1], .001))
                  if self.bedforms_enabled else xp.full(self.nc, 1e10))
        result = self.process.evaluate_targets(mass, sv, delta, targets, dt,
            sorting_dt=sorting_dt, dune_length=length, shear_pa=p.sorting_shear,
            bedload_mass_flux=p.sorting_bedload, hiding_factor=p.mpm_hiding)
        p.sorting_mobility[:] = result[3]
        # Post-collapse rebalance must not replace the nonzero sorting trace
        # or apply a second time-integrated transfer.
        if sorting_dt > 0:
            p.sorting_transfer[:] = result[4]
        return result[:3]

    def remap(self, *args, **kwargs):
        return self.base.remap(*args, **kwargs)

    def limits(self, *args, **kwargs):
        return self.base.limits(*args, **kwargs)


def install(a, selection, parameters):
    bed = selection["bed"]
    needed = bed["coarse_layer"] or bed["transport_layer"] == "transport_layer.dune_height"
    if not needed:
        return a
    p, xp, c = a.physics, a.xp, a.columns
    if a.time or a.steps or p._open or c.pending is not None:
        raise ValueError("Install dependent layers before execution")
    methods = methods_for(a.methods, bed, parameters)
    if methods.hiding_scope == "active_coarse" and not methods.coarse_layer:
        raise ValueError("Active-plus-coarse hiding requires the selected coarse layer")
    if a.backend == "cpu":
        process = NativeDependentLayerRuntimeV103(methods, p.n, c.runtime.nl, threads=p.flow.threads)
    else:
        from braids_gpu.dependent_layers_v103 import ResidentDependentLayerRuntimeV103
        process = ResidentDependentLayerRuntimeV103(methods, p.n, c.runtime.nl)
    force = SortingForcing(a.backend, p.flow.threads)
    ng = len(a.methods.fraction_ids)
    p.sorting_shear = xp.zeros(p.n)
    p.sorting_bedload = xp.zeros((ng, p.n))
    p.sorting_mobility = xp.ones((ng, p.n))
    p.sorting_transfer = xp.zeros(p.n)
    p.sorting_faces = xp.zeros((2, p.nf))
    p.sorting_constant_chezy = xp.full(p.nf, p.chezy)
    c.runtime = RiverDependentColumns(c.runtime, process, p, selection["bedforms"] != "off")
    c._seal()

    def begin(owner, delta, qnew, dt):
        # bed_faces already refreshed effective hydraulic fields and hiding
        # once for this exact end state. class_flux has been donor-limited.
        face_chezy = p.bedform_faces[0] if hasattr(p, "bedform_faces") else p.sorting_constant_chezy
        force.evaluate(qnew, p.z0, p.effective_gradient, face_chezy, p.g, p.class_flux,
            p.sorting_faces, p.sorting_shear, p.sorting_bedload, p.parameters.dry,a.material.solid)
        return owner.columns.begin(delta, qnew[0], dt, sorting_dt=dt*p.morfac)
    a.begin_columns = MethodType(begin, a)
    p.identity["river_dependent_layers"] = dict(methods=asdict(methods),
        binding=file_digest(__file__), forcing=force.identity,
        inventory="one continuous owner; sorting once before accepted commit",
        column_runtime=c.runtime.identity, no_bedform_dune_length_m=1e10,
        flux="accepted donor-limited normal flux; face-length weighted vector fit",
        shear="current-only physical-face stress; length-weighted FLOW caltmx adaptation",
        topology="active, coarse when selected, ordered substrate; no direct substrate bypass",
        public_support=False)
    p.identity["river_mixtures"]["column_runtime"] = c.runtime.identity
    return a
