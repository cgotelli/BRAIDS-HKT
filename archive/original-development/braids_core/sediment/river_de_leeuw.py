"""de Leeuw et al. 2020 Eq.26a with the section 3.3 concentration-profile fit.

Explicit single-bulk 2DH adaptation, not the paper's full grain-resolved dataset.
The local friction-slope and shallow-front extensions are saved method identities.
"""
from pathlib import Path
import math
import numpy as np

METHOD = 'suspension.de_leeuw_2020'
DEFAULTS = dict(entrainment='de_leeuw2020_eq26a', coefficient=4.74e-4,
    mobility_exponent=1.5, threshold=0.015, exponent=1.18, denominator_factor=3.0,
    reference_height_fraction=0.1, below_reference_multiplier=1.66,
    skin_shear='manning_strickler_8p1_3D84_local_friction_slope_bounded_total_v1',
    settling='ferguson_church2004_natural_C1_18_C2_1',
    profile='de_leeuw2020_section3p3_P_total_ratio_power_minus0p45',
    depth_average='rouse_above_0p1h_constant_1p66Ca_below_v1',
    advection='depth_mean_water_velocity',
    shallow='no_entrainment_below_max_10Ds_3D84_well_mixed_settling_v1')
NAMES = ('dl_probe', 'dl_exchange')


def parameters(ds,d84):
    return dict(validate(dict(DEFAULTS,suspended_diameter_m=ds,bed_d84_m=d84)))


def validate(values):
    if not isinstance(values,dict) or set(values)!=set(DEFAULTS)|{'suspended_diameter_m','bed_d84_m'}:
        raise ValueError('de Leeuw 2020 requires its complete named bundle and independent measured Ds and bed D84.')
    for k,v in DEFAULTS.items():
        if type(values[k]) is not type(v) or values[k]!=v:
            raise ValueError(f'de Leeuw 2020 requires {k}={v!r}; restore the qualified bundle.')
    for key in ('suspended_diameter_m','bed_d84_m'):
        x=values[key]
        if type(x) not in (float,int) or not math.isfinite(x) or x<=0:
            raise ValueError('Enter a positive finite measured '+key+'. No diameter is inferred.')
    if values['suspended_diameter_m']<62.5e-6:
        raise ValueError('de Leeuw 2020 is admitted for noncohesive sand/gravel: Ds must be at least 0.0625 mm. Choose a different law for finer material.')
    return values


class DeLeeuw:
    def __init__(self,backend,threads,values):
        from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
        self.parameters=dict(validate(values))
        self.engine=WaveEngine(backend,threads,source=Path(__file__).with_suffix('.h'),names=NAMES)
        self.xp=np if backend=='cpu' else __import__('cupy')
        x,w=np.polynomial.legendre.leggauss(48)
        self.quadrature=self.xp.asarray(np.concatenate(((x+1)/2,w/2)))

    def probe(self,h,speed,z0,*,density=2650.,water_density=1000.,temperature=20.):
        xp=self.xp
        arrays=[xp.ascontiguousarray(xp.asarray(v,dtype=xp.float64)) for v in (h,speed,z0)]
        if any(v.ndim!=1 or v.shape!=arrays[0].shape for v in arrays):raise ValueError('Equal 1D arrays required')
        if any(bool(xp.any(~xp.isfinite(v))) for v in arrays) or bool(xp.any(arrays[0]<=0)) or bool(xp.any(arrays[2]<=0)):
            raise ValueError('Positive finite depth/roughness required')
        if not np.isfinite([density,water_density,temperature]).all() or not density>water_density>0 or temperature<=-20:
            raise ValueError('Positive submerged density and temperature above -20 C required')
        n=arrays[0].size;out=xp.empty((8,n))
        self.engine.launch('dl_probe',n,n,float(self.parameters['suspended_diameter_m']),
            float(self.parameters['bed_d84_m']),float(density/water_density-1),float(temperature),
            *arrays,self.quadrature,out)
        return out
