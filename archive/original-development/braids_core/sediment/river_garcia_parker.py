"""GP1991 uniform-material entrainment with explicitly named 2DH auxiliaries.

This is not a reproduction of a complete published hydraulic solver. See S4
report for the bounded EH skin partition and full-depth Rouse extension.
"""
from pathlib import Path
import math
import numpy as np

METHOD = 'suspension.garcia_parker_1991'
DEFAULTS = dict(entrainment='garcia_parker_1991_uniform', coefficient=1.3e-7,
    saturation=0.3, reference_height_fraction=0.05, kappa=0.4,
    skin_shear='engelund_hansen1967_bounded_total_v1',
    settling='flow_settlefrm2_clear_water',
    profile='rouse_depth_mean_constant_below_reference_v1',
    advection='depth_mean_water_velocity',
    shallow='no_entrainment_below_h_20Ds_well_mixed_settling_v1')
NAMES = ('gp_probe', 'gp_exchange')
ZERO_CAPACITY = dict(ACal=0., PowerB=0., PowerC=1.5, RipFac=1., ThetaC=0.,
    SusACal=0., SusPowerB=0., SusPowerC=1., SusRipFac=1., SusThetaC=0.)


def parameters(ds):
    result=dict(DEFAULTS, suspended_diameter_m=ds)
    validate(result)
    return result


def validate(values):
    if not isinstance(values,dict) or set(values)!=set(DEFAULTS)|{'suspended_diameter_m'}:
        raise ValueError('GP1991 requires the complete named auxiliary bundle and an independent measured Ds.')
    for k,v in DEFAULTS.items():
        if type(values[k]) is not type(v) or values[k]!=v:
            raise ValueError(f'GP1991 requires {k}={v!r}; this is a fixed qualified bundle, not a calibration control.')
    ds=values['suspended_diameter_m']
    if type(ds) not in (float,int) or not math.isfinite(ds) or ds<=0:
        raise ValueError('Enter a positive finite measured suspended diameter Ds.')
    return values


class GarciaParker:
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
        n=arrays[0].size;out=xp.empty((7,n))
        self.engine.launch('gp_probe',n,n,float(self.parameters['suspended_diameter_m']),
            float(density/water_density-1),float(temperature),*arrays,self.quadrature,out)
        return out
