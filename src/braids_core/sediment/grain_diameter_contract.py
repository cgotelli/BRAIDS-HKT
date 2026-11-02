"""Bulk grain descriptors are measurements, not independently transported classes."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DiameterContract:
    definition: str
    label: str
    note: str


LABELS = dict(d30='D30 (30%)', d50='D50 (50%)', d84='D84 (84%)',
              d90='D90 (90%)', dm='Dm (mean)')
REQUIRED = {
    'capacity.van_rijn_1993': ('d50', 'd90'),
    'capacity.mpm': ('d50', 'd90'),
    'capacity.wong_parker_2006': ('dm',),
    'capacity.ashida_michiue_generalized': ('d50',),
    'capacity.wang_fredsoe': ('d50', 'd90'),
    'capacity.smart_jaggi': ('dm', 'd30', 'd90'),
    'capacity.ancey_recking_2023': ('d50',),
    'capacity.recking_2010': ('d84', 'd50'),
    'capacity.camenen_larson_2005': ('d50',),
    'capacity.mpm_jc2026_p15_reconstruction_v1': ('d50', 'd90'),
}
NOTES = {
    'capacity.van_rijn_1993': 'Measured bed D50 sets mobility and reference concentration; measured bed D90 sets current drag efficiency. Suspended Ds is supplied independently in the suspended setup.',
    'capacity.mpm': 'D50 sets Shields stress and transport scale; measured D90 sets the FLOW grain/form-drag correction.',
    'capacity.wong_parker_2006': 'Arithmetic mean Dm, not a percentile. Uniform-sediment plane-bed relation; no MPM form-drag correction.',
    'capacity.ashida_michiue_generalized': 'D50 sets Shields stress and transport scale.',
    'capacity.wang_fredsoe': 'D50 sets particle size and threshold; measured D90 sets grain friction.',
    'capacity.smart_jaggi': 'Mean Dm sets the threshold; measured D30 and D90 set (D90/D30)^0.2. Dm is not a percentile.',
    'capacity.ancey_recking_2023': 'Bed-surface D50 sets Shields stress and dimensionless transport scale.',
    'capacity.recking_2010': 'Measured surface D84 sets Shields stress and transport scale. D84/D50 enters the threshold and low-transport branch (2010 Eq. 12).',
    'capacity.camenen_larson_2005': 'D50 sets Shields stress, dimensionless grain size, critical Shields and transport scale.',
    'capacity.mpm_jc2026_p15_reconstruction_v1': 'D50 sets capacity and exchange; measured D90 sets automatic Sun/van Rijn roughness and shallow/profile gates. Fixed 3/2 exponent.',
}
CONTRACTS = {m: DiameterContract(ds[0], LABELS[ds[0]], NOTES[m] +
    ' Enter every required diameter independently. One bulk flux; no size-class sorting.')
    for m, ds in REQUIRED.items()}
CLASS = DiameterContract('class_representative', 'Class size',
    'Representative class diameter, not a percentile of the whole mixture. '
    'The internal multi-fraction route is not offered in Desktop.')


def diameter_contract(method, count=1):
    return CONTRACTS.get(method, CLASS) if count == 1 else CLASS


def require_diameter_definition(method, count, definition):
    contract = diameter_contract(method, count)
    if definition != contract.definition:
        raise ValueError(f'{method} requires {contract.label}, not {definition!r}. '
                         'Enter the required measured grain descriptors in 04 Sediment transport.')
    return contract


def validate_grain_sizes(capacity, *, primary=None):
    """No inferred missing percentile, numerical conversion or legacy adapter."""
    method = capacity['method']
    required = REQUIRED.get(method, ())
    sizes = capacity.get('grain_sizes_m')
    if not isinstance(sizes, dict) or set(sizes) != set(required):
        labels = ', '.join(LABELS[d] for d in required) or 'no bulk descriptors'
        raise ValueError(f'{method} requires explicit {labels} in grain_sizes_m. '
                         'Enter these sizes in 04 Sediment transport / Grain sizes; '
                         'missing diameters are never derived from another diameter.')
    for d, value in sizes.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Enter a finite positive {LABELS[d]} in Grain sizes (UI: mm; JSON: metres).')
    ordered = [d for d in ('d30', 'd50', 'd84', 'd90') if d in sizes]
    for low, high in zip(ordered, ordered[1:]):
        if sizes[low] > sizes[high]:
            raise ValueError(f'{LABELS[low]} cannot exceed {LABELS[high]}. Correct the measured grain sizes.')
    if required and primary is not None and sizes[required[0]] != primary:
        raise ValueError(f'Bulk carrier diameter must equal the explicitly supplied {LABELS[required[0]]}. '
                         'Re-enter the formula grain sizes; no second transport class is needed.')
    return sizes
