"""Classic FLOW wet-slope bedload flux; not a post-update collapse owner."""
import math
from pathlib import Path
from .bend_methods_v1 import digest
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine

PARAMETERS = dict(wet_slope=10., relaxation_time_s=86400.)
METHOD = "redistribution.avalanche"


def validate_parameters(value):
    if not isinstance(value, dict) or set(value) != set(PARAMETERS):
        raise ValueError("Explicit wet slope and avalanche relaxation time required")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
           for v in value.values()):
        raise ValueError("Wet slope and avalanche relaxation time must be finite and positive")


class WetSlopeFlux:
    def __init__(self, owner, parameters):
        validate_parameters(parameters)
        p=owner.physics;g=p.g;xp=p.xp
        self.p=p;self.parameters=dict(parameters)
        self.engine=WaveEngine(owner.backend,p.flow.threads,
            source=Path(__file__).with_suffix('.h'),
            names=('wet_avalanche_geometry','wet_avalanche_flux'))
        self.weight=xp.zeros(p.nf);self.flux=xp.zeros(p.nf)
        self.engine.launch('wet_avalanche_geometry',p.nf,p.nf,p.n,
            g['owner'],g['neighbour'],g['offset'],g['faces'],g['normal'],
            g['length'],g['area'],g['xy'],self.weight)
        if not bool(xp.all(xp.isfinite(self.weight))) or bool(xp.any(self.weight<0)):
            raise ValueError('Invalid wet-avalanche face geometry')
        self.identity=dict(method=METHOD,parameters=self.parameters,wrapper_sha256=digest(__file__),
            runtime=self.engine.identity,
            native_source='FLOW4 adjust_bedload.f90 wet-slope branch',
            units='pure solid-volume m2/s; no extra porosity factor',
            ownership='add before combined-flux donor fraction; one shared finite-stock limiter and column update',
            geometry='projected-side area shares; harmonic pair area; true-normal secant-corrected gradient',
            activation='internal hydraulic-open face with two physically wet cells; no exterior avalanche flux',
            time='relaxation seconds in morphological clock; MorFac applied once by class update',
            state='mesh-owned weights and overwritten per-call scratch; no independent accepted inventory')

    def evaluate(self,q):
        p=self.p;g=p.g
        self.engine.launch('wet_avalanche_flux',p.nf,p.nf,p.n,p.parameters.dry,
            float(self.parameters['wet_slope']),float(self.parameters['relaxation_time_s']),
            g['owner'],g['neighbour'],p.hydraulic_face_active,q,g['normal'],p.bedface,
            self.weight,self.flux)
        return self.flux


def install(owner, selected, parameters):
    validate_parameters(parameters)
    if selected is None or selected['avalanche'] != METHOD:
        return owner
    if owner.time or owner.steps or owner.physics._open or owner._wet_avalanche is not None:
        raise ValueError('Select wet avalanching once before execution')
    if owner.physics.identity['slope_relaxation']['method'] != 'off':
        raise ValueError('Wet-slope flux and post-update collapse cannot both own redistribution')
    owner._wet_avalanche=WetSlopeFlux(owner,parameters)
    owner.physics.identity['slope_relaxation']=owner._wet_avalanche.identity
    return owner
