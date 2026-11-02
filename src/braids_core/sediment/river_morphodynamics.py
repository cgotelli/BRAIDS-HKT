"""Selected river-development composition; one numerical method on CPU/CUDA.

The 2026-10-18 trend/peak decision accepts the conservative wet-bank route for
continued feature development. Historical constructors stay immutable replay
evidence. This is not a change to saved projects or the installed QGIS worker.
"""
from pathlib import Path
import numpy as np
from .current_shoreline_solver import build_current_shoreline
from .flow_bank_relaxation import install_shoreline_bank_relaxation
from .bend_methods_v1 import digest


METHOD='river_conservative_fwave_wet_bank_v1'


def build_river_morphodynamics(deck,data,*,backend,controls,bedbc,feed,xy,
                              work_directory,bank_theta=.5,bank_slope=9/16,
                              relaxation=False,gpu_solver_type=None,source_folder=None):
    """Use the selected composition, without competing numerical switches.

    Bank sharing and collapse are physical controls, not solver variants.
    Uniform full square cells/single grain are the present coupled scope.
    Mixture/layer admission requires the subsequent authoritative-stock bridge.
    """
    if backend not in ('cpu','gpu'):raise ValueError('Choose cpu or gpu explicitly')
    if not np.isfinite(bank_theta) or not 0<=bank_theta<=1:
        raise ValueError('Bank sharing fraction must be in [0,1]')
    if type(relaxation) is not bool:raise ValueError('Explicit collapse control required')
    if not np.isfinite(bank_slope) or bank_slope<=0:raise ValueError('Positive bank slope required')
    dx=float(data['dx'])
    if not np.allclose(data['area'],dx*dx,rtol=0,atol=1e-12):
        raise ValueError('Selected river development route requires full square cells')
    # Runtime-generated sources belong in new work, never beside a frozen deck.
    folder=Path(work_directory);folder.mkdir(parents=True,exist_ok=True)
    from braids_support.sediment.river_morphodynamics import face_activity,wet_bank_handoff
    from braids_support.sediment.river_morphodynamics.hr_wave_coupling import install as hrwave
    from braids_support.sediment.river_morphodynamics.mean_state_wave import install as meanwave
    a=build_current_shoreline(deck,data,backend=backend,controls=controls,bedbc=bedbc,feed=feed,
        gpu_solver_type=gpu_solver_type,source_folder=source_folder)
    face_activity.install_hydraulic(a)
    install_shoreline_bank_relaxation(a,xy,theta=bank_theta,slope=bank_slope,relaxation=relaxation)
    hrwave(a,folder);face_activity.install_sediment(a);meanwave(a,folder)
    if bank_theta>0:wet_bank_handoff.install(a)
    a.identity['selected_river_bundle']=dict(method=METHOD,wrapper_sha256=digest(__file__),
        selection='trend, extrema, spatial behavior, water/solid conservation, CPU/GPU agreement',
        source_evidence='bank_opening_v1; common numerical face cutoff comparison',
        water_bed_update='preserve water volume, no bed-volume source',
        dry_depth_m=float(a.physics.parameters.dry),one_owner_per_process=True,
        numerical_choices=False,backend_explicit=True,
        scope='fixed_grid_single_grain_implicit_midpoint',
        mixtures_integrated=False,implicit_wet_dry=True,installed_QGIS=False,
        collapse_scope='existing isolated/composed legacy Scour evidence; no new long-time combined qualification')
    return a
