"""Public supplied suspension: measured Ds, clear-water settling, no entrainment.

Uses the retained SettleFrm2/exponential source kernel. Infinite hindering
reference is an internal exact switch (C/inf=0), not a user-supplied number.
The project records the clear-water choice explicitly; no nominal-size fallback.
"""
import math

METHOD='suspension.settling_only'
EDITOR='suspended.settling_only'
BUNDLE=dict(method=METHOD,settling='FLOW_SettleFrm2_clear_water_measured_Ds',
    profile='well_mixed_depth_mean',entrainment='off',
    exchange='exact_exponential_physical_clock_wet_bed_MF',hindering='off',
    advection='accepted_water_flux_first_order',
    shallow='physical_settling_all_wet_depths_dry_stranding_once')

def parameters(ds):
    return dict(validate(dict(BUNDLE,suspended_diameter_m=ds)))

def validate(values):
    if not isinstance(values,dict) or set(values)!=set(BUNDLE)|{'suspended_diameter_m'}:
        raise ValueError('Supplied suspension requires its complete named clear-water bundle and measured Ds; no diameter is inferred.')
    for k,v in BUNDLE.items():
        if type(values[k]) is not type(v) or values[k]!=v:
            raise ValueError(f'Supplied suspension requires {k}={v!r}. Restore the selected no-entrainment bundle.')
    d=values['suspended_diameter_m']
    if type(d) not in (int,float) or not math.isfinite(d) or d<62.5e-6:
        raise ValueError('Enter measured suspended Ds of at least 0.0625 mm. Finer/cohesive transport is not admitted.')
    return values
