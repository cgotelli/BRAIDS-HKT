"""Bounded classic-FLOW morphology-control resolution for Q6a.

This module records parser controls whose numerical owners are in earlier or
later stages. It never substitutes an FM-only consumer into classic FLOW 2DH.
"""
from dataclasses import dataclass, asdict
import math
from .bed_columns_v101 import digest


@dataclass(frozen=True)
class MorphologyControlsV101:
    update_bed_at_inflow: bool = False
    maximum_waterdepth: bool = False
    maximum_waterdepth_fraction: float = 1.
    flux_limiter: str = 'none'

    def __post_init__(self):
        if type(self.update_bed_at_inflow) is not bool or type(self.maximum_waterdepth) is not bool:
            raise ValueError('Morphology controls must use explicit booleans')
        if (not math.isfinite(self.maximum_waterdepth_fraction)
                or self.maximum_waterdepth_fraction < 0):
            raise ValueError('MaximumWaterdepthFraction must be finite and nonnegative')
        if self.flux_limiter.lower() not in ('none', 'minmod', 'mc'):
            raise ValueError('Classic parser accepts none, minmod or mc')

    @property
    def fingerprint(self):
        return digest({'scope': 'braids-q6a-morphology-controls-v101', **asdict(self)})

    def resolve_bed_boundary_modes(self, explicit_modes=()):
        """Resolve the FLOW UpdInf default without claiming Q6b modes 1--8."""
        modes = tuple(explicit_modes)
        if any(type(x) is not int or not 0 <= x <= 8 for x in modes):
            raise ValueError('Bed-boundary modes must be integers from zero through eight')
        if self.update_bed_at_inflow:
            return {'modes': (0,)*len(modes), 'reason': 'UpdInf_true_forces_free_boundaries'}
        return {'modes': modes, 'reason': 'explicit_or_UpdInf_false_default_fixed'}

    @property
    def comparison_dispositions(self):
        return {
            'UpdInf': {'value': self.update_bed_at_inflow,
                'classic_flow_2dh': 'forces all bed boundary conditions to mode 0 (free)'},
            'MaximumWaterdepth': {'value': self.maximum_waterdepth,
                'owner': 'Q3 settling characteristic-depth control'},
            'MaximumWaterdepthFraction': {'value': self.maximum_waterdepth_fraction,
                'classic_flow_2dh': 'parsed but unused; classic uses max(hs,hu) when MaximumWaterdepth=true',
                'dflow_fm': 'active multiplier; not imported into classic comparison'},
            'FluxLimiter': {'value': self.flux_limiter.lower(),
                'classic_flow_2dh': 'parsed to enum but has no numerical consumer'},
        }
