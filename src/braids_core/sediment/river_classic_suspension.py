"""Named current-only S6 closures; shared CPU/CUDA arithmetic.

Published equilibrium closures are distinguished from the BRAIDS conservative
2DH exchange, local shear and shoreline adaptations. No inferred grain sizes.
"""
from pathlib import Path
import math
import numpy as np

VR84='suspension.van_rijn_1984'
ZF94='suspension.zyserman_fredsoe_1994'
CL08='suspension.camenen_larson_2008'
METHODS=(VR84,ZF94,CL08)
EDITOR=tuple(x.replace('suspension.','suspended.') for x in METHODS)
FIELDS={'ds':'suspended_diameter_m','d50':'bed_d50_m','d90':'bed_d90_m'}
COMMON=dict(settling='van_rijn1984_clear_water_eq11_13_measured_Ds',
    total_shear='physical_hydraulic_resistance_local_v1',
    advection='depth_mean_water_velocity',
    exchange='exponential_physical_clock_finite_stock_bed_clock_adapter',
    shallow='h_below_20_max_Ds_D50_or_reference_above_half_depth_no_entrainment_well_mixed_v1')
BUNDLES={
 VR84:dict(COMMON,entrainment='van_rijn1984_eq38',coefficient=.015,exponent=1.5,
    skin_shear='chezy_18log10_12h_over_3D90_bounded_total_v1',
    threshold='van_rijn1984_Shields_piecewise',
    reference_height='max_30z0_0p01h',profile='van_rijn1984_eq19_33_34_parabolic_constant',
    beta='eq22_1plus2ratio_squared_ratio_capped_1',
    damping='eq34_ratio_capped_1_packing_0p65',
    depth_average='integral_a_to_h_no_below_reference_inventory',packing_limit=.65),
 ZF94:dict(COMMON,entrainment='zyserman_fredsoe1994',coefficient=.331,exponent=1.75,
    settling='rubey1933_clear_water_measured_Ds_analytic_no_HEC_cap',
    saturation=.46,threshold=.045,skin_shear='zf1994_eq9_11_boundary_layer_ks_2p5D50_bounded_total_v1',
    reference_height='2D50',profile='rouse_total_shear_kappa_0p4',
    depth_average='integral_a_to_h_no_below_reference_inventory'),
 CL08:dict(COMMON,entrainment='camenen_larson2008_eq16_17_current_cr_over_theta',
    coefficient=.0015,grain_exponent=-.2,threshold_exponent=-4.5,
    threshold='soulsby_whitehouse1997',skin_shear='not_used_total_shear',
    reference_height='bed_zero',profile='exponential_constant_diffusivity_eq4_9_11_12',
    schmidt_low=.4,schmidt_amplitude=3.5,kappa=.4,
    depth_average='analytic_exponential_integral_0_to_h',packing_limit=.65),
}
NAMES=('classic_probe','classic_exchange')

def parameters(method,sizes):
    keys=('ds','d50','d90') if method==VR84 else ('ds','d50')
    if method not in METHODS:raise ValueError('Choose a named S6 suspended closure.')
    return dict(validate(dict(BUNDLES[method],method=method,**{FIELDS[k]:sizes[k] for k in keys})))

def validate(values):
    if not isinstance(values,dict) or values.get('method') not in METHODS:
        raise ValueError('S6 requires its complete named suspension bundle.')
    method=values['method'];keys=('ds','d50','d90') if method==VR84 else ('ds','d50')
    if set(values)!=set(BUNDLES[method])|{'method'}|{FIELDS[k] for k in keys}:
        raise ValueError('Enter exactly the required measured diameters and named S6 auxiliaries; no diameter is inferred.')
    for k,v in BUNDLES[method].items():
        if type(values[k]) is not type(v) or values[k]!=v:raise ValueError(f'S6 requires {k}={v!r}; restore the selected published bundle.')
    for k in keys:
        x=values[FIELDS[k]]
        if type(x) not in (int,float) or not math.isfinite(x) or x<=0:
            raise ValueError('Enter a positive finite measured '+k.upper()+'.')
    if values['suspended_diameter_m']<62.5e-6:
        raise ValueError('S6 is admitted for noncohesive sand: measured Ds must be at least 0.0625 mm. Finer/cohesive transport is not qualified.')
    if method==VR84 and values['bed_d90_m']<values['bed_d50_m']:
        raise ValueError('Measured D90 cannot be smaller than D50.')
    return values

class ClassicSuspension:
    def __init__(self,backend,threads,values):
        from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine
        self.parameters=dict(validate(values));self.code=METHODS.index(values['method'])
        self.engine=WaveEngine(backend,threads,source=Path(__file__).with_suffix('.h'),names=NAMES)
        self.xp=np if backend=='cpu' else __import__('cupy')
        x,w=np.polynomial.legendre.leggauss(64)
        self.quadrature=self.xp.asarray(np.concatenate(((x+1)/2,w/2)))

    @property
    def arguments(self):
        p=self.parameters
        return (self.code,float(p['suspended_diameter_m']),float(p['bed_d50_m']),float(p.get('bed_d90_m',0.)))

    def probe(self,h,speed,z0,*,density=2650.,water_density=1000.,temperature=20.):
        xp=self.xp;arrays=[xp.ascontiguousarray(xp.asarray(v,dtype=xp.float64)) for v in (h,speed,z0)]
        if any(v.ndim!=1 or v.shape!=arrays[0].shape for v in arrays):raise ValueError('Equal 1D arrays required')
        if any(bool(xp.any(~xp.isfinite(v))) for v in arrays) or bool(xp.any(arrays[0]<=0)) or bool(xp.any(arrays[2]<=0)):
            raise ValueError('Positive finite depth/roughness required')
        if not np.isfinite([density,water_density,temperature]).all() or not density>water_density>0 or temperature<=-20:
            raise ValueError('Positive submerged density and temperature above -20 C required')
        n=arrays[0].size;out=xp.empty((10,n))
        self.engine.launch('classic_probe',n,n,*self.arguments,float(density/water_density-1),float(temperature),*arrays,self.quadrature,out)
        return out

    def exchange(self,n,dt,dry,rho,r,temp,q,z0,mass,stock,next_mass,delta,diag):
        self.engine.launch('classic_exchange',n,n,dt,dry,*self.arguments,rho,r,temp,q,z0,mass,stock,self.quadrature,next_mass,delta,diag)
