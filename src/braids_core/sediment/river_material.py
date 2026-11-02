"""Explicit material conversions for the common river inventory.

Mass is kg/m2; transport face flux is solid m2/s. Column volume uses the
column method's reference density and its separate solid-fraction state.
These quantities must not be interchanged when grain densities differ.
"""
from dataclasses import dataclass
import numpy as np
from .bend_methods_v1 import digest


def composition_method(bed):
    if bed == "mixed":
        return "composition.flow_mixed_mass"
    if bed == "layered":
        return "composition.flow_layered_reference_volume"
    raise ValueError("Explicit mixed or layered bed required")


@dataclass(frozen=True)
class RiverMaterial:
    fraction_ids: tuple[str, ...]
    solid_density: tuple[float, ...]
    column_density: tuple[float, ...]
    water_density: float = 1000.

    def __post_init__(self):
        for name in ("fraction_ids", "solid_density", "column_density"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        n = len(self.fraction_ids)
        if n < 1 or len(set(self.fraction_ids)) != n or any(not k for k in self.fraction_ids):
            raise ValueError("Unique ordered material identities required")
        if isinstance(self.water_density, (bool, np.bool_)) or not np.isfinite(self.water_density) or self.water_density <= 0:
            raise ValueError("Positive water density required")
        for values in (self.solid_density, self.column_density):
            if (len(values) != n or any(isinstance(v, (bool, np.bool_)) for v in values)
                    or not np.isfinite(values).all() or min(values) <= 0):
                raise ValueError("Positive ordered per-class densities required")
        if any(r <= self.water_density for r in self.solid_density):
            raise ValueError("River sediment must be denser than water")
        if any(c > s for c, s in zip(self.column_density, self.solid_density)):
            raise ValueError("Column reference density cannot exceed solid density")

    @classmethod
    def from_columns(cls, methods):
        return cls(tuple(methods.fraction_ids), tuple(methods.solid_density_kg_m3),
                   tuple(float(r) for r in methods.density))

    def resident(self, xp):
        return ResidentMaterial(self, xp)

    def input_column_fractions(self, fractions, basis):
        """Public mass/true-solid-volume input to private reference volume.

        This is a units conversion, not selection of transport weighting.
        Initial layer thickness remains bulk thickness in both input bases.
        """
        f = np.asarray(fractions, dtype=float)
        if (f.ndim != 2 or f.shape[1] != len(self.fraction_ids)
                or not np.isfinite(f).all() or np.any(f < 0)
                or not np.allclose(f.sum(1), 1., atol=1e-12, rtol=0)):
            raise ValueError("Normalized layer-by-class fractions required")
        if basis == "fraction_basis.mass":
            scale = 1./np.asarray(self.column_density)
        elif basis == "fraction_basis.solid_volume":
            scale = np.asarray(self.solid_density)/np.asarray(self.column_density)
        else:
            raise ValueError("Explicit mass or true solid-volume input basis required")
        if np.all(scale == scale[0]):
            return f.copy()
        weights = f*(scale/scale[0])
        return weights/weights.sum(1, keepdims=True)


class RiverComposition:
    """One FLOW physical weighting for all consumers of the same stock."""
    def __init__(self, methods, xp):
        self.xp = xp
        self.method = composition_method(methods.bed)
        self.density = xp.asarray(methods.density, dtype=xp.float64)
        if xp is np:
            self.density.setflags(write=False)
        self.identity = dict(method=self.method, source_sha256=digest(__file__),
                             source="FLOW bedcomposition set_default_fractions/getfrac",
                             input_units="independent of physical transport weighting")

    def weights(self, stock):
        return stock if self.method == "composition.flow_mixed_mass" else stock/self.density[:, None]

    def fractions(self, stock):
        weights = self.weights(stock)
        total = weights.sum(0)
        return self.xp.where(total[None, :] > 0.,
            weights/self.xp.maximum(total[None, :], 1e-300), 0.)


class ResidentMaterial:
    def __init__(self, definition, xp):
        self.definition, self.xp = definition, xp
        self.solid = xp.asarray(definition.solid_density, dtype=xp.float64)
        self.column = xp.asarray(definition.column_density, dtype=xp.float64)
        self.relative = self.solid/definition.water_density-1.
        if xp is np:
            for array in (self.solid, self.column, self.relative):
                array.setflags(write=False)
        self.identity = dict(
            fraction_ids=list(definition.fraction_ids),
            solid_density_kg_m3=list(definition.solid_density),
            column_reference_density_kg_m3=list(definition.column_density),
            water_density_kg_m3=definition.water_density,
            source_sha256=digest(__file__),
            face_flux="solid volume per width and time",
            inventory="mass per actual cell area",
            bed="sum(class_mass / column_reference_density) / layer_solid_fraction")

    def bed(self, mass, sv, rock):
        return rock+(mass/self.column[None, :, None]/sv[:, None, :]).sum(axis=(0, 1))

    def solid_volume(self, mass):
        """Per-class/per-cell solid thickness, summed through ordered strata."""
        return (mass/self.solid[None, :, None]).sum(axis=0)

    def boundary_mass(self, solid_integrals, length, outside):
        return -self.xp.sum(solid_integrals*length*outside, axis=1)*self.solid

    def bulk_increment(self, class_mass, solid_fraction=None):
        """Frozen surface packing for the transport-only increment guard.

        The accepted bed always uses bed(), including recomputed packing.
        """
        volume = (class_mass/self.column[:, None]).sum(axis=0)
        return volume if solid_fraction is None else volume/solid_fraction
