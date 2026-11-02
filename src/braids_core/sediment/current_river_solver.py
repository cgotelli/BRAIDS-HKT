"""One construction path for the accepted CPU/CUDA fixed-grid river bundle.

No laboratory geometry or physical coefficient is selected here. Callers own
prepared inputs and the GPU source context. This is the shared integration
entry point, not yet admission to the wider QGIS worker or wet/dry/AMR physics.
Historical constructors/checkpoints remain unchanged.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .bend_methods_v1 import Parameters, digest


@dataclass(frozen=True)
class CurrentRiverControls:
    chezy: float
    diameter: float
    morfac: float
    betac: float
    viscosity: float
    diffusivity: float
    threads: int = 4
    cfl: float = .5
    nonlinear_atol: float = 1e-7
    nonlinear_rtol: float = 1e-6
    temporal_depth_atol: float = 4e-4
    temporal_momentum_atol: float = 4e-3
    temporal_rtol: float = 4e-3

    def validate(self):
        values = asdict(self)
        if not np.isfinite(list(values.values())).all():
            raise ValueError('Current river controls must be finite')
        if (self.chezy < 1.e-6 or self.diameter <= 0 or self.morfac <= 0
                or not 0 <= self.betac <= 1
                or min(self.viscosity, self.diffusivity) < 0
                or isinstance(self.threads, bool) or int(self.threads) != self.threads
                or self.threads < 1
                or isinstance(self.cfl, bool) or not 0 < self.cfl <= 1.
                or not 0 < self.nonlinear_atol <= 1e-6
                or not 0 < self.nonlinear_rtol < 1
                or self.temporal_depth_atol <= 0
                or self.temporal_momentum_atol <= 0
                or not 0 < self.temporal_rtol < 1):
            raise ValueError('Invalid current river controls')


def validate_current_river_input(data, controls):
    """Host-only admission, shared by preparation and either backend."""
    controls.validate()
    if (float(data.get('time', 0)) != 0 or data.get('amr_enabled', False)
            or np.any(np.asarray(data['bed']) != 0)):
        raise ValueError('Current river entry requires fresh flat-bed fixed-grid input')
    depth=np.asarray(data['state'])[0]
    if not np.isfinite(depth).all() or np.any(depth <= 20 * float(data['dry'])):
        raise ValueError('Current river wet/dry integration is not yet qualified')
    for key in ('bedlaw','exchange_mode','secondary_flow_model','secondary_flow','bed_slope',
                'horizontal_viscosity','bedload_adaptation','bedform_roughness','general_avalanche',
                'bank_failure','bank_toe_erosion_mode','morphological_acceleration',
                'nsubstrate_layers','nhydro_series','nseries','fraction_bed_input','bc_bed_input',
                'fraction_concentration','series_bed_input','fraction_series_bed_input','secondary_mass'):
        if np.any(data.get(key,0)):
            raise ValueError('Current river cannot duplicate or ignore '+key)
    if (int(data['nfrac'])!=1 or int(data['nbed_layers'])!=1
            or np.any(data['state'][3]) or float(data['morphological_factor'])!=1
            or np.any(np.asarray(data['forcing'])[2:4])):
        raise ValueError('Current river requires one inert grain/layer carrier and explicit physical owner')
    if not np.allclose((data['grav'],data['dry'],data['d50_fraction'][0],data['rho'],
                        data['rhos_fraction'][0],data['bulk_fraction'][0]),
                       (9.81,1e-6,controls.diameter,1000.,2650.,1590.),rtol=1e-12,atol=0):
        raise ValueError('Prepared material differs from qualified river material')


def create_current_river_solver(deck, data, *, backend, controls, bedbc, feed,
                                morphology=False, gpu_solver_type=None,
                                gpu_source_folder=None):
    """Build all accepted corrections together; never select/fallback backend.

    The input must be a fresh hydraulic-only, fully wet, flat-bed, fixed-grid
    carrier. Sediment boundaries are face-indexed solid-volume feed/outflow
    definitions, supplied explicitly. The bed then evolves normally. Native
    input physics that would be duplicated is rejected by the paired owners.

    Returns (hydraulic solver, transaction/checkpoint owner, physical owner).
    CPU and GPU have identical constitutive controls. Restart must go through
    the returned owner and retains its exact method/runtime identity.
    """
    if backend not in ('cpu', 'gpu'):
        raise ValueError('Choose backend explicitly: cpu or gpu')
    # Check inexpensive argument errors before loading/allocating a backend.
    controls.validate()
    if backend == 'gpu':
        if gpu_solver_type is None or gpu_source_folder is None:
            raise ValueError('GPU constructor and authenticated source folder are required')
        if not (Path(gpu_source_folder) / 'reconstruction.cu').is_file():
            raise ValueError('GPU reconstruction source is missing')
    elif gpu_solver_type is not None or gpu_source_folder is not None:
        raise ValueError('CPU selection cannot consume GPU construction options')
    validate_current_river_input(data,controls)

    # No imports from experiment drivers and no case-specific globals. Every
    # accepted numerical correction is installed below without runtime switches.
    from .bend_cpu_solver import CPUBendSolver
    from .flow2dh_current_bed import Flow2DHCurrentBed
    from .flow2dh_fixed_bed import bind
    from .wall_bed_hydraulics import install

    parameters = Parameters(manning=float(data['manning_cell'][0]),
                            diameter=controls.diameter, ripple_factor=0.,
                            wall_sand_roughness=0.)
    options = dict(method='lv_odgaard', parameters=parameters)
    if backend == 'cpu':
        solver = owner = CPUBendSolver(Path(deck), data, morphology=morphology,
            threads=int(controls.threads), nonlinear_atol=controls.nonlinear_atol,
            nonlinear_rtol=controls.nonlinear_rtol, **options)
    else:
        from braids_gpu.bend_methods_v1 import ProductionBendOwner
        solver = gpu_solver_type(data)
        controller = solver.implicit_midpoint
        if controller is None:
            raise ValueError('Current river GPU requires implicit_fwave_midpoint')
        controller.nonlinear_atol = controls.nonlinear_atol
        controller.nonlinear_rtol = controls.nonlinear_rtol
        controller.nonlinear_maxiter = 30
        owner = ProductionBendOwner(solver, morphology=morphology, **options)

    physics = Flow2DHCurrentBed(data, backend=backend,
        chezy=controls.chezy, diameter=controls.diameter,
        bedbc=bedbc, feed=feed, morfac=controls.morfac, betac=controls.betac,
        viscosity=controls.viscosity, diffusivity=controls.diffusivity,
        threads=int(controls.threads))
    bind(solver, owner, physics, backend)
    install(solver, owner, backend, Path(deck), data, gpu_source_folder)
    owner.identity['current_river_construction'] = dict(
        schema=1, source_sha256=digest(__file__), backend=backend,
        controls=asdict(controls), corrections='single_maintained_bundle',
        scope='fully_wet_flat_initial_bed_fixed_grid_single_grain_EH',
        public_qgis_admission=False)
    return solver, owner, physics
