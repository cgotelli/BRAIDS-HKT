"""Transport-only extension of the selected river arithmetic to cropped cells.

No Cartesian bank/Scour adapter is installed or silently used on irregular
polygons. The same physical areas, faces, normals and shared fluxes own Exner.
"""
from pathlib import Path
from .current_shoreline_solver import build_current_shoreline
from .river_morphodynamics import METHOD
from .bend_methods_v1 import digest


def build_transport_river(deck,data,*,backend,controls,bedbc,feed,work_directory,
                          gpu_solver_type=None,source_folder=None):
    from validation.sediment.river_morphodynamics import face_activity,hr_wave_coupling,mean_state_wave
    a=build_current_shoreline(deck,data,backend=backend,controls=controls,bedbc=bedbc,feed=feed,
        gpu_solver_type=gpu_solver_type,source_folder=source_folder,
        native_hydraulics=face_activity.native_hydraulics if backend=='cpu' else None)
    face_activity.install_hydraulic(a)
    from braids_core.windows_paths import filesystem_path
    folder=filesystem_path(work_directory);folder.mkdir(parents=True,exist_ok=True)
    hr_wave_coupling.install(a,folder);face_activity.install_sediment(a);mean_state_wave.install(a,folder)
    p=a.physics;xp=a.xp
    p.transport_trial_mass=xp.zeros(p.n);p.transport_cumulative_mass=xp.zeros(p.n)
    p.identity['dry_bank']=dict(theta=0.,geometry_not_installed=True)
    p.identity['slope_relaxation']=dict(method='off',geometry_not_installed=True)
    a.identity['selected_river_bundle']=dict(method=METHOD,wrapper_sha256=digest(__file__),
        scope='transport_only_polygonal_fixed_grid_implicit_midpoint',
        banks=False,collapse=False,implicit_wet_dry=True,installed_QGIS=False,backend_explicit=True,
        geometry='actual clipped area/centroid/face length/normal; no nominal-cell replacement')
    return a
