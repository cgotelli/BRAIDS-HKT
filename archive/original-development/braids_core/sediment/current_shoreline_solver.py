"""Single maintained shoreline bundle for semi-implicit CPU/CUDA runs.

Frozen validation constructors remain reference evidence, not supported
numerical choices. Installed-QGIS admission is tracked separately by release.
The reconstruction/Exner adapters remain development dependencies under
validation until their wider worker integration is qualified.
"""
from pathlib import Path
from .shoreline_solver import ShorelineSolver
from .shoreline_viscosity import install as viscosity
from .bend_methods_v1 import digest


METHOD='current_shoreline_positive_conservative_v1'


def build_current_shoreline(deck,data,*,backend,controls,bedbc,feed,
                            gpu_solver_type=None,source_folder=None,native_hydraulics=None):
    """Compose the fixed bundle once; caller separately owns bank physics.

    Shared augmented f-wave midpoint, positive bounded hydraulic traces,
    conservative depth-integrated viscosity, matrix sediment-wave bed flux,
    physical shallow-depth conversion and face-based transverse correction.
    No CPU/GPU auto-selection, implicit fallback, or altered physical controls.
    """
    from validation.sediment.river_morphodynamics.shoreline_hydraulic_probe import install as hydraulic
    from validation.sediment.river_morphodynamics.sediment_wave import install as wave
    from validation.sediment.river_morphodynamics.shallow_face_order import install as shallow
    a=ShorelineSolver(deck,data,backend=backend,controls=controls,bedbc=bedbc,feed=feed,
        gpu_solver_type=gpu_solver_type,source_folder=source_folder,native_hydraulics=native_hydraulics)
    from braids_core.windows_paths import filesystem_path
    a._shoreline_source_folder=filesystem_path(source_folder) if source_folder else filesystem_path(deck).parent/'gpu_sources'
    if native_hydraulics is None:
        hydraulic(a,'positive_bounded')
    else:
        if not getattr(a.hydraulics,'_river_activity_ready',False):raise ValueError('Final river hydraulic constructor required')
        a.identity['hydraulic_probe']=a.hydraulics.identity
    wave(a.physics,matrix=True);shallow(a.physics,face_tangent=True);viscosity(a)
    a.identity['current_shoreline_bundle']=dict(method=METHOD,source_sha256=digest(__file__),
        supported_development_scope='fixed_grid_uniform_grain_implicit_midpoint',
        implicit_wet_dry=True,installed_QGIS=False,numerical_options=False)
    return a
