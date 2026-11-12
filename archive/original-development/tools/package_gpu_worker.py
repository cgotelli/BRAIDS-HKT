"""Package the authenticated Windows GPU SOURCE worker for QGIS.

No Python environment, CUDA wheel, validation input/output, Delft3D checkout or
user project is collected. The native executable only initializes a deck for
the GPU; the public CLI explicitly selects the computation backend.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath,PureWindowsPath
import re
import runpy
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,''):
    sys.path.insert(0,str(ROOT))
# These variants need frozen validation input/output or an archived numerical
# source tree.  This source-only package deliberately contains neither.  Keep
# the public choices visible but reject them before output/CUDA initialization.
ARCHIVE_BOUND_VARIANTS={
    'live_mode9_section_choice_ssp33':'requires the current-source surveyed stage/critical inlet route; portable numerical bundle qualification pending',
    'authenticated_composed_amr_ssp33':'requires the frozen native composed AMR input, private Gate-B numerical adapter and reference archive',
    'scheduled_normal9_critical4':'requires the frozen qualified source plan and live-section reference archive',
    'combined_live_scheduled_a1_ssp43':'requires the frozen 128-cell A1 input, native reference and current-source archive',
    'live_normal_critical_secondary_mpm_ssp43':'requires the guarded live native exporter and its source-locked reference gate',
    'live_normal_critical_suspension_mpm_ssp33':'requires the frozen live suspension/exchange input pair, native reference and source-locked numerical gate',
    'live_normal_critical_van_rijn_mpm_ssp33':'requires the frozen live Van Rijn exchange-2 input pair, native reference and source-locked numerical gate',
    'live_normal_critical_van_rijn_graded_ssp33':'requires the frozen seven-class/four-layer live Van Rijn input pair, native reference and source-locked numerical gate',
    'live_normal_critical_graded_adaptation_ssp33':'requires the frozen seven-class/four-layer target-B adaptation ON/OFF pair, native reference and source-locked numerical gate',
    'physical_toe_mode2_avalanche_secondary_ssp33':'requires the frozen four-case bank/avalanche/secondary inputs, process gate and prepared sidecars',
    'g3_live_bank_mode2_ssp33':'requires the frozen live mode-2 bank/avalanche native pair, combined sidecars and source-locked process gate',
    'g3_live_bank_mode3_ssp33':'requires the frozen live mode-3 bank/avalanche native pair, combined sidecars and source-locked process gate',
    'qualified_curved_wetdry':'requires the frozen bar256 input, plan and numerical source archive',
    'distributed_slope_mode3_plain_ssp33':'requires frozen native reference plans, gates and run outputs',
    'distributed_slope_mode3_avalanche_ssp33':'requires frozen native reference plans, gates and run outputs',
}
ARCHIVE_BOUND_SIDECARS={
    'normal_metadata.bin':'live normal/critical boundaries use the archived combined source route',
    'distributed_bank_metadata.bin':'mode-3 bank physics requires frozen native reference artifacts',
    'bank_history_metadata.bin':'live bank and bank-AMR histories require authenticated combined preparation and a qualified runtime',
    'delft_forcing_binding.json':'typed Delft forcing belongs only to the authenticated common transport route',
    'execution_profile_v2.json':'execution-profile admission belongs only to the authenticated common transport route',
}
EXTERNAL_PREPARATION_ONLY={
    'general_avalanche_plain_ssp33':['avalanche_metadata.bin'],
    'physical_toe_mode2_plain_ssp33':['bank_toe_metadata.bin'],
    'physical_toe_mode2_avalanche_ssp33':['bank_toe_metadata.bin','avalanche_metadata.bin'],
}
# The qualified graded route prepares this sidecar from an ordinary deck using
# the shipped pinned zero-step native exporter. The exception is variant-scoped;
# all historical routes still reject this sidecar.
VARIANT_PREPARED_SIDECARS={
    'live_normal_critical_graded_bedload_ssp33':['normal_metadata.bin'],
    'live_normal_critical_transport_ssp33':['avalanche_metadata.bin','coupled_diffusion.bin',
        'delft_forcing_binding.json','execution_profile_v2.json','normal_metadata.bin'],
    'live_normal_critical_bank_ssp33':['avalanche_metadata.bin','bank_history_metadata.bin',
        'bank_toe_metadata.bin','coupled_diffusion.bin','distributed_bank_metadata.bin','normal_metadata.bin'],
}
# Relative worker payload location for the qualified common numerical bundle;
# never point a package at a validation run folder.
VARIANT_SOURCE_ARCHIVES={
    'live_normal_critical_graded_bedload_ssp33':'braids_gpu/frozen_sources/graded_bedload_v8',
    'on_demand_bank_amr_ssp33':'braids_gpu/frozen_sources/on_demand_amr_v9',
    'on_demand_transport_amr_ssp33':'braids_gpu/frozen_sources/on_demand_transport_amr_v8',
}
AMR_VARIANT='on_demand_bank_amr_ssp33'
RECURSIVE_AMR_CONTRACT={
    'schema':2,
    'stepper':'recursive_ssprk33_fine_replaced_register_v1',
    'routes':[
        {'bank_mode':2,'evolving_banks':False,
         'checkpoint_kind':'recursive_bank_amr','checkpoint_schema':2},
        {'bank_mode':2,'evolving_banks':True,
         'checkpoint_kind':'recursive_bank_amr','checkpoint_schema':2},
        {'bank_mode':3,'evolving_banks':True,
         'checkpoint_kind':'recursive_bank_amr','checkpoint_schema':1},
    ],
}
RECURSIVE_GENERAL_AMR_CONTRACT={
    'schema':1,'bank_mode':0,
    'stepper':'recursive_ssprk33_fine_replaced_register_v1',
    'checkpoint_kind':'recursive_general_amr','checkpoint_schema':6,
}
RECURSIVE_ADAPTATION_CONTRACT={
    'schema':1,'policy':'full_leaf_steady_v1','bank_mode':3,
    'secondary_flow_model':2,'boundary_modes':[9,4],'section_controls':[0,1,2],
}
RECURSIVE_GENERAL_ADAPTATION_CONTRACT={
    'schema':1,'policy':'full_leaf_steady_v1','bank_mode':0,
    'secondary_flow_model':2,'boundary_modes':[9,4],'section_controls':[0,1,2],
}
RECURSIVE_GENERAL_AVALANCHE_CONTRACT={
    'schema':1,'policy':'parent_post_reflux_once_per_macro_v1','bank_mode':0,
    'secondary_flow_model':2,'boundary_modes':[9,4],'checkpoint_schema':6,
}
RECURSIVE_GENERAL_VISCOSITY_CONTRACT={
    'schema':1,
    'policy':'parent_post_reflux_ssprk2_once_per_macro_v1',
    'placement':'external_parent_post_reflux_before_avalanche',
    'calls_per_macro':1,
    'integrator':'SSPRK2_subcycled_area_weighted_velocity_L2_limited',
    'limited_norm':'area_weighted_velocity_L2',
    'area_weighted_velocity_l2_nonincrease':True,
    'pairwise_velocity_face_flux':True,
    'momentum_conservation':'variable_depth_not_exact',
    'dynamic_wet_geometry':True,
    'general_avalanche_modes':[0,1],
    'composition_order':'viscosity_then_avalanche',
    'bank_mode':0,
    'secondary_flow_model':2,
    'boundary_modes':[9,4],
    'viscosity_mode':1,
    'viscosity_geometry':1,
    'formulation':'velocity_laplacian',
    'checkpoint_schema':6,
}
RECURSIVE_GENERAL_BEDFORM_CONTRACT={
    'schema':1,
    'policy':'level_ssprk33_stage_capacity_and_pre_friction_v1',
    'stage_capacity_evaluations_per_level_step':3,
    'pre_friction_evaluations_per_level_step':1,
    'accepted_diagnostics':'lazy_from_accepted_state',
    'checkpointed_diagnostics':False,
    'bank_mode':0,
    'boundary_mode_counts':{'1':1,'2':1},
    'bedform_modes':[0,1],
    'unsampled_zero_offset':True,
    'scheduled_forcing':False,
    'bedlaw':2,
    'mpm_friction_mode':1,
    'bed_slope':0,
    'hiding_exposure':0,
    'exchange_mode':0,
    'secondary_flow_model':0,
    'bedload_adaptation':0,
    'horizontal_viscosity':0.,
    'general_avalanche':0,
    'morphological_factor':1.,
    'nsubstrate_layers':1,
    'checkpoint_schema':6,
}
RECURSIVE_BANK_BEDFORM_AVALANCHE_CONTRACT={
    'schema':1,
    'policy':'recursive_bedform_mode3_parent_avalanche_v1',
    'bedform_order':'level_ssprk33_stage_capacity_and_pre_friction_v1',
    'stage_capacity_evaluations_per_level_step':3,
    'pre_friction_evaluations_per_level_step':1,
    'accepted_diagnostics':'lazy_from_accepted_state',
    'checkpointed_diagnostics':False,
    'bank_transaction':'distributed_mode3_post_reflux_once_per_macro_v1',
    'avalanche_transaction':'parent_post_bank_once_per_macro_v1',
    'calls_per_macro':{'bank':1,'avalanche':1},
    'bank_mode':3,'evolving_banks':True,
    'boundary_mode_counts':{'2':1},'unsampled_zero_offset':True,
    'scheduled_forcing':False,'bank_failure':1,'bedform_roughness':1,
    'general_avalanche':1,'bedlaw':2,'mpm_friction_mode':1,
    'bed_slope':0,'hiding_exposure':0,'exchange_mode':0,
    'secondary_flow_model':0,'bedload_adaptation':0,
    'horizontal_viscosity':0.,'morphological_factor':1.,
    'nsubstrate_layers':1,'checkpoint_kind':'recursive_bank_amr',
    'checkpoint_schema':1,'frame_schema':7,
}
RECURSIVE_PHYSICAL_TOE_BEDFORM_AVALANCHE_CONTRACT={
    'schema':1,
    'policy':'recursive_bedform_mode2_physical_toe_parent_avalanche_v1',
    'bedform_order':'level_ssprk33_stage_capacity_and_pre_friction_v1',
    'stage_capacity_evaluations_per_level_step':3,
    'pre_friction_evaluations_per_level_step':1,
    'accepted_diagnostics':'lazy_from_accepted_state',
    'checkpointed_diagnostics':False,
    'accepted_flux_owner':'completed_root_exner_register_v1',
    'bank_transaction':
        'physical_toe_supply_undercut_collapse_post_reflux_once_per_macro_v1',
    'avalanche_transaction':'parent_post_bank_once_per_macro_v1',
    'mapping_transaction':'reconcile_then_evolving_toe_rebuild_before_commit_v1',
    'calls_per_macro':{'bank':1,'avalanche':1},
    'bank_mode':2,'evolving_banks':True,'toe_geometry_seed_only':True,
    'boundary_mode_counts':{'2':1},'unsampled_zero_offset':True,
    'scheduled_forcing':False,'bank_failure':1,'bedform_roughness':1,
    'general_avalanche':1,'bedlaw':2,'mpm_friction_mode':1,
    'bed_slope':0,'hiding_exposure':0,'exchange_mode':0,
    'secondary_flow_model':0,'bedload_adaptation':0,
    'horizontal_viscosity':0.,'morphological_factor':1.,
    'nsubstrate_layers':1,'checkpoint_kind':'recursive_bank_amr',
    'checkpoint_schema':2,'frame_schema':8,
}
AMR_RUNTIME_SOURCE='validation/g4_bank_amr/runs/on_demand_geometry_runtime_20261008_v3'
AMR_RUNTIME_TARGET='braids_gpu/native_amr_v1'
AMR_LIVE_RUNTIME_SOURCE='validation/g4_bank_amr/runs/on_demand_bank_mode23_geometry_runtime_20261008_v2'
AMR_LIVE_RUNTIME_TARGET='braids_gpu/native_amr_live_v1'
AMR_BEDFORM_BANK_RUNTIME_SOURCE='validation/g4_bank_amr/runs/on_demand_bank_mode3_bedform_runtime_20261013_v9'
AMR_BEDFORM_BANK_RUNTIME_TARGET='braids_gpu/native_amr_bedform_bank_v3'
AMR_BEDFORM_PHYSICAL_TOE_RUNTIME_SOURCE='validation/g4_bank_amr/runs/on_demand_bank_mode2_physical_toe_bedform_runtime_20261013_v6'
AMR_BEDFORM_PHYSICAL_TOE_RUNTIME_TARGET='braids_gpu/native_amr_bedform_physical_toe_v1'
AMR_GENERAL_RUNTIME_SOURCE='braids_gpu/native_amr_general_v10'
AMR_GENERAL_RUNTIME_TARGET='braids_gpu/native_amr_general_v1'
AMR_GENERAL_VARIANT='on_demand_transport_amr_ssp33'
AMR_GENERAL_QUALIFIED_SOURCE='validation/g5_consolidated/runs/regrid_fast_path_v12/source/runtime_manifest.json'
AMR_GENERAL_QUALIFIED_SOURCE_SHA='549b0154fb2d88685bc3609c0f3a019f781a0e658fc9d062841ab8fd201b6e5c'
AMR_KERNEL_SOURCE='validation/g4_bank_amr/runs/common_worker_lifecycle_20261005_v3/full/source'
AMR_KERNEL_MANIFEST_SHA='335d3ef64e99264ff41b29ad7acf8f9304ae44bfb7b1bd0234f43bdb1354b16e'
GRADED_VARIANT='live_normal_critical_graded_bedload_ssp33'
GRADED_EXPORT_SOURCE='braids_gpu/native_exporter_v1'
GRADED_EXPORT_TARGET='braids_gpu/native_exporter_v1'
GRADED_EXPORT_SHA='a50f91dd99e5b48af94535761a7ec2f5b933e1e21ca9479370e78d03275eac33'
GRADED_CRITICAL_REFERENCE='validation/gpu_fwave/runs/cpu_filter_convergence_control_20261002_v1/runtime/cutcell_solver.f90'
GRADED_CRITICAL_REFERENCE_SHA='75a5451cce5d11e2aec244330aa91724e514dbe16dfdfef97fb9b50572020340'
INTEGRATED_VARIANT='integrated'
TRANSPORT_VARIANT='live_normal_critical_transport_ssp33'
BANK_VARIANT='live_normal_critical_bank_ssp33'
IMPLICIT_INTEGRATOR='implicit_fwave_midpoint'
IMPLICIT_INTEGRATOR_CONTRACT={
    'label':'IMPLICIT-F-WAVE-MIDPOINT',
    'schema':1,
    'variants':[TRANSPORT_VARIANT],
    'launch_modes':['plain'],
    'checkpoint_schema':2,
    'checkpoint_schemas':[2,3,4,5,6,7,8],
    'component_coupling_schemas':[1,2,8,9,10,11],
    'bedload_laws':[0,2,8],
    'execution_profile_v2_bedload_laws':[2,8,11],
    'bed_slope_modes':[0,2,3,5],
    'secondary_flow_models':[0,1,2],
}
TRANSPORT_EXPORT_SOURCE='braids_gpu/native_exporter_v3'
TRANSPORT_EXPORT_TARGET='braids_gpu/native_exporter_v3'
TRANSPORT_EXPORT_SHA='898f467518789453ee2be202095eef2a81725c4e262f0c1c22b0625be56970fd'
GAIA_TRANSPORT_EXPORT_SOURCE='braids_gpu/native_exporter_v6'
GAIA_TRANSPORT_EXPORT_TARGET='braids_gpu/native_exporter_v6'
GAIA_TRANSPORT_EXPORT_SHA='1c1c8503b7e297e6df07fa53685c41635937ea982254f6719bfe205d253be838'
DELFT_TRANSPORT_EXPORT_SOURCE='braids_gpu/native_exporter_v14'
DELFT_TRANSPORT_EXPORT_TARGET='braids_gpu/native_exporter_v14'
DELFT_TRANSPORT_EXPORT_SHA='d803c052cfc3db3de71ef387dc7e77f1458c1d11a691a1b44b429af4ba7f699f'
DELFT_TRANSPORT_EXPORT_V13_SOURCE='braids_gpu/native_exporter_v13'
DELFT_TRANSPORT_EXPORT_V13_TARGET='braids_gpu/native_exporter_v13'
DELFT_TRANSPORT_EXPORT_V13_SHA='1828a447ea08e5311828128cca005581f44d14b88dc409a63ff9f48faacbacb8'
DELFT_TRANSPORT_EXPORT_V15_SOURCE='braids_gpu/native_exporter_v15'
DELFT_TRANSPORT_EXPORT_V15_TARGET='braids_gpu/native_exporter_v15'
DELFT_TRANSPORT_EXPORT_V15_SHA='9ccdd9cb457140eda9ad418a2843047666c40ea1d8fe6031d52a02a4da88fbd8'
DELFT_TRANSPORT_EXPORT_V16_SOURCE='braids_gpu/native_exporter_v16'
DELFT_TRANSPORT_EXPORT_V16_TARGET='braids_gpu/native_exporter_v16'
DELFT_TRANSPORT_EXPORT_V16_SHA='4e76d84516ba9fc24cf66154de35b548d8214932fa785cc365a818692e4cb266'
DELFT_TRANSPORT_EXPORT_V17_SOURCE='braids_gpu/native_exporter_v17'
DELFT_TRANSPORT_EXPORT_V17_TARGET='braids_gpu/native_exporter_v17'
DELFT_TRANSPORT_EXPORT_V17_SHA='75faae530e2d8a19144b62aa8e18aff7620b5f0b158c0e60a19e769770a2ae7d'
DELFT_TRANSPORT_EXPORT_V18_SOURCE='braids_gpu/native_exporter_v18'
DELFT_TRANSPORT_EXPORT_V18_TARGET='braids_gpu/native_exporter_v18'
DELFT_TRANSPORT_EXPORT_V18_SHA='fb6c9778e1b21400ed6a8bbce0acce4fcecbf3babda55f1dd17262b322f8f467'
BANK_EXPORT_SOURCE='validation/g3_live_bank/runs/live_combined_export_20261004_v2/runtime'
BANK_EXPORT_TARGET='braids_gpu/native_bank_exporter_v1'
BANK_EXPORT_SHA='f9dc86987a04328e778ce790bb89133aa7c97943b550b2b335508aa3a8195ab5'
RUNTIME_DLLS=frozenset(('libgfortran-5.dll','libgomp-1.dll','libgcc_s_seh-1.dll',
    'libquadmath-0.dll','libwinpthread-1.dll','libdl.dll'))
SYSTEM_DLLS=frozenset(('kernel32.dll','advapi32.dll','ntdll.dll','user32.dll','msvcrt.dll',
    'ucrtbase.dll','shell32.dll','ole32.dll','oleaut32.dll','ws2_32.dll','secur32.dll','bcrypt.dll'))
CLAWPACK_SHA='668317944192d8217d63c0fd0730f889bd07f759b69ed6b657e1fb9902145ebe'
EXECUTION_PROFILE_SOURCE='braids_core/sediment/execution_profile_v2.py'
EXECUTION_PROFILE_COMPLETENESS_SOURCE=(
    'braids_core/sediment/fixed_grid_profile_completeness.py')
EXECUTION_PROFILE_LEDGER='braids_core/sediment/flow2dh_selector_ledger_v2.json'
QGIS_PRODUCT_SOURCE='braids_core/sediment/qgis_product_v109.py'
QGIS_PRODUCT_OUTPUT_OVERLAY='braids_core/sediment/flow2dh_scope_overlay_v1.json'
RECURSIVE_WORKER_SOURCE='braids_core/sediment/recursive_worker_v118.py'
Q3H2_LIVE_CUDA_SOURCE='braids_gpu/flow2dh_secondary_live.cu'
Q3H3_PROGNOSTIC_PYTHON_SOURCE='braids_gpu/flow2dh_prognostic_secondary.py'
Q3H3_PROGNOSTIC_CUDA_SOURCE='braids_gpu/flow2dh_prognostic_secondary.cu'
Q3H3_PROGNOSTIC_PYTHON_SHA=(
    '5637a5513ef615f74bc0823bda8826c0aa14c1f3648279b6cdfa579e4658f2ac')
Q3H3_PROGNOSTIC_CUDA_SHA=(
    '90e0b16986048250b74edcd29a1aa43c6e7a3a8dbfa166b20cd315c64db8a082')
Q3H3_PROGNOSTIC_SOURCE_CONTRACT={
    'schema':1,
    'execution_scope_id':'braids-q3h3-flow2dh-prognostic-secondary-composition-v1',
    'method_id':'secondary.prognostic',
    'coupling':'weak_midpoint_q3h3_classic_flow_prognostic_secondary_composition_v1',
    'component_coupling_schema':11,
    'checkpoint_schema':8,
    'python_source':Q3H3_PROGNOSTIC_PYTHON_SOURCE,
    'python_sha256':Q3H3_PROGNOSTIC_PYTHON_SHA,
    'cuda_source':Q3H3_PROGNOSTIC_CUDA_SOURCE,
    'cuda_sha256':Q3H3_PROGNOSTIC_CUDA_SHA,
}

QUICKSTART=r'''BRAIDS authenticated Windows GPU SOURCE worker
==================================================
This is NOT an installable QGIS plugin, NOT a bundled Python environment, and
NOT a replacement for the production CPU executable. Python/CuPy/CUDA drivers
are not included. The QGIS plugin authenticates and launches this external
process only for an explicitly selected supported GPU product profile.
On Windows, extract the ZIP into a short directory (for example C:\BRAIDS_GPU)
to avoid legacy long-path limits in some archive tools.

Requirements
------------
64-bit Windows; an NVIDIA CUDA-capable GPU and compatible NVIDIA driver;
64-bit CPython 3.13 (tested 3.13.5); the pinned packages in requirements-gpu.txt.
The tested GPU is an RTX 5080, using FP64 and CUDA 12.9 runtime/NVRTC. Other
hardware is not promised to be faster. The CPU backend requires your own
compatible native BRAIDS executable; the bundled executable is initialization
only and must not be selected as the CPU simulation executable.

Do NOT install these packages into QGIS Python or an existing project environment.
First verify that "python --version" reports CPython 3.13 x64; if it does not,
select a full path to a compatible CPython executable. The Windows "py"
launcher is optional and is not present on every PC. From this extracted
directory, create a separate GPU environment with the verified interpreter:
  python -m venv .venv_gpu
Or use an explicit path when Python is not on PATH:
  "C:\Path\To\Python313\python.exe" -m venv .venv_gpu
  .venv_gpu\Scripts\python.exe -m pip install -r requirements-gpu.txt
No installer is run automatically. Internet access is needed for that optional
installation; this ZIP does not contain Python, CuPy wheels, the CUDA driver or
an offline installer. An existing compatible GPU environment can be used instead.
In QGIS, explicitly select this extracted worker and that environment's
python.exe, then use Verify Worker before choosing the GPU backend. Pair them
in Run: choose GPU, browse to this extracted worker folder, browse to
.venv_gpu\Scripts\python.exe, press Verify, select Integrated (selected
physics), then Prepare Run. The worker/Python pins are session-only: repeat
Verify after restarting QGIS or changing either path. This is not offline
plug-and-play while Python/CuPy/wheels and the NVIDIA driver are external.

Use the source-worker directory as the working directory in all commands:
  .venv_gpu\Scripts\python.exe -m braids_gpu capabilities
  .venv_gpu\Scripts\python.exe -m braids_gpu prepare --input "C:\case\case.data" --output "C:\case\gpu_prepared" --gpu-preparation-variant integrated --stop-file "C:\case\STOP_PREP"
  .venv_gpu\Scripts\python.exe -m braids_gpu run --backend gpu --input "C:\case\gpu_prepared" --output "C:\case\gpu_run" --gpu-numerical-variant integrated --gpu-integrator ssp33 --gpu-launch-mode plain --until 20 --output-every 1 --stop-file "C:\case\STOP_GPU"
For eligible fixed-grid, nonbank MPM bedlaw-2 cases with exchange off and
bedform feedback, bedload adaptation and general avalanche off, select the
qualified four-stage method explicitly:
  .venv_gpu\Scripts\python.exe -m braids_gpu run --backend gpu --input "C:\case\gpu_prepared" --output "C:\case\gpu_run_ssp43" --gpu-numerical-variant integrated --gpu-integrator ssp43 --gpu-launch-mode plain --until 20 --output-every 1
Other integrated combinations retain plain SSPRK(3,3) and reject SSPRK(4,3)
instead of dropping physics. The historical preparation/schema identifier
`live_normal_critical_transport_ssp33` is not the executed method label; each
run authenticates its actual integrator and stage map in the method identity.
The guarded implicit f-wave midpoint is a separate, explicitly selected method:
  .venv_gpu\Scripts\python.exe -m braids_gpu run --backend gpu --input "C:\case\gpu_prepared" --output "C:\case\gpu_run_implicit" --gpu-numerical-variant live_normal_critical_transport_ssp33 --gpu-integrator implicit_fwave_midpoint --gpu-launch-mode plain --until 20 --output-every 1
It is admitted only when this exact source worker advertises
`IMPLICIT-F-WAVE-MIDPOINT` schema 1/checkpoint schema 2. Its initial scope is
fixed-grid, safely fully wet, low-Froude flow with constant supported boundaries,
MorFac 1, and either static bed, the narrowly weakly coupled one-class MPM
law-2 route, or the exact schema-v2 one-class law-11 total-load profile. The
law-11 profile requires total hydraulic shear with separate suspension,
slope, hiding and secondary flow all off. AMR, wet/dry fronts, suspended
exchange, secondary flow, bank
processes, avalanching, bedform roughness, adaptation, slope/hiding corrections,
viscosity and other optional morphology are rejected. Existing qualification
evidence is limited to the documented 1 mm perturbation envelope. A missing or
incompatible capability fails before launch; there is no explicit-method or CPU
fallback. Use the matching source worker rather than an older worker package.
The integrated route reads the authenticated ordinary deck and selects its
packaged fixed-grid transport/bank or qualified on-demand AMR operator.
For nonbank AMR, it selects the versioned zero-step general initializer and
the dynamic GPU transport route; both layered transport and hydraulic-only
ordinary decks have passed extracted-worker engineering smokes.
For a new general-AMR preparation, add
  --gpu-amr-selector-policy local_eta_plane_v1
to avoid refining a smooth longitudinal free-surface slope merely because
adjacent cells have different elevations. Omit the flag only to reproduce
the legacy selector for an existing/reference run. The chosen policy is
authenticated in the prepared session and cannot change on restart.

For a supported graded live-bedload deck, select both its preparation and
numerical route explicitly (the authenticated numerical bundle is included):
  .venv_gpu\Scripts\python.exe -m braids_gpu prepare --input "C:\case\case.data" --output "C:\case\graded_prepared" --gpu-preparation-variant live_normal_critical_graded_bedload_ssp33
  .venv_gpu\Scripts\python.exe -m braids_gpu run --backend gpu --input "C:\case\graded_prepared" --output "C:\case\graded_gpu_run" --gpu-numerical-variant live_normal_critical_graded_bedload_ssp33 --gpu-integrator ssp33 --gpu-launch-mode plain

For the supported synchronous bank-AMR composition, the original native deck
must include amr_hierarchy.npz and its declared sidecars:
  .venv_gpu\Scripts\python.exe -m braids_gpu prepare --input "C:\case\case.data" --output "C:\case\amr_prepared" --gpu-preparation-variant on_demand_bank_amr_ssp33 --stop-file "C:\case\STOP_PREP"
  .venv_gpu\Scripts\python.exe -m braids_gpu run --backend gpu --input "C:\case\amr_prepared" --output "C:\case\amr_run" --gpu-numerical-variant on_demand_bank_amr_ssp33 --gpu-integrator ssp33 --gpu-launch-mode plain --stop-file "C:\case\STOP_GPU"
Its geometry-only initializer and revision-paired CUDA bundle are included. It
supports mode-2 bank erosion, MPM and prognostic secondary flow with one
constant stage outlet and the qualified AMR indicators. Add --evolving-banks
to preparation for the versioned evolving-interface candidate. New evolving
preparations also export live-section geometry and hydraulic/class-feed tables;
the current candidate can combine a surveyed inlet (normal, critical or stage
control) and critical outlet, clipping timesteps at forcing knots. This route
requires positive inlet Q and its declared wet-section support; it must not be
used to claim that entirely dry/zero-Q inlet handling has been qualified.
For an admitted AMR deck, subcycling selects recursive SSPRK33 level stepping
with conservative fine-replaced face registers. The bank route supports static
and evolving physical-toe mode 2 and evolving distributed-bank mode 3. The
nonbank general route has its own recursive checkpoint contract. Subcycling does
not select a different CPU/GPU backend. Recursive bedload adaptation is
explicitly supported only by each route's advertised full-leaf steady GPU
closure with surveyed mode-9 inlet, critical mode-4 outlet and prognostic
secondary flow (model 2). The nonbank route may also apply conservative
domain-wide avalanching once on the complete selected-leaf graph after each
completed macro-step reflux. It may apply native mode-1/geometry-1 positive
velocity-Laplacian viscosity once on that graph before optional avalanching.
The viscosity operator limits area-weighted velocity L2 and preserves the
pairwise velocity-face divergence; variable-depth integrated momentum is not
exact. A separate recursive bedform-feedback branch supports the qualified
legacy MPM-friction-1 envelope only: one unsampled mode-1 discharge inlet,
one constant mode-2 stage outlet, suspended exchange/secondary flow/
adaptation/viscosity/avalanche/banks off, MorFac 1 and one substrate layer.
The paired bedform-OFF control uses that same branch and worker capability;
accepted bedform diagnostics are reconstructed lazily for frames and are not
checkpoint state. Within this recursive nonbank route, bedform feedback cannot
be composed with recursive viscosity or avalanche, and bank physics remains
excluded.

Separately, on-demand bank AMR admits exactly one recursive composition:
bedform feedback with evolving distributed-bank mode 3 followed by one
parent-owned avalanche after the bank transaction. It is authenticated as
`recursive_bedform_mode3_parent_avalanche_v1`; preparation must explicitly add
`--evolving-banks --recursive-bank-bedform-avalanche`. This does not relax the
nonbank exclusions or admit any other bank/bedform/avalanche combination.
All surveyed inlet controls
(stage, normal and critical) retain their existing scope. Older OFF-only
workers do not admit adaptation, avalanche ON, positive recursive viscosity or
the discharge/stage recursive bedform pair. The package authenticates
these capabilities separately from older synchronous workers; never reuse a
synchronous-only worker for a recursive prepared session.
Suspended bank exchange and other unported AMR combinations reject.
This is the external numerical worker used by the Q10 QGIS GPU product.  It is
not itself an installable QGIS plugin and still requires a separate pinned
Python/CuPy/CUDA environment.

All destination folders must be new. Preparation copies only the native deck
and its declared sidecars, then builds initialized geometry on CPU. Native DLLs
and their notices are beside/included with the initializer; no compiler is
required to run it. The simulation's supported numerical state evolves on GPU;
Python controls launches, convergence, timestep adaptation and output.

There is no automatic CPU/GPU selection and no CPU fallback on GPU failure.
Use --backend cpu --cpu-executable "C:\BRAIDS\xbraids_cutcell.exe" with a native
case.data only when you explicitly want the separately supplied CPU engine.

Current supported scope
-----------------------
The integrated fixed-grid route includes ordinary hydraulic-only, bedload,
suspended transport, secondary flow, supported viscosity and bedform feedback,
and supported mode-2/3 bank processes. The package contains their pinned
zero-step native exporters, authenticated sidecars and source files. It also
includes the separately scoped on-demand bank-AMR, nonbank transport-AMR and
graded-bedload routes. Run "capabilities" for exact supported and excluded
choices. Other historical archived experiments and complete physical
validation remain outside this package. Frozen validation inputs and results
are not bundled. This source worker is selected and authenticated by the QGIS
plugin; it is not itself an installable QGIS plugin.

Output / stop / restart
-----------------------
Create STOP_PREP to cancel CPU-side initialization and its direct native
initializer child. Preparation reports cancelled with exit code 130; the new
folder keeps preparation_status.json for diagnosis but has no valid receipt,
so it cannot be run or resumed. Remove/use a different stop marker before
another preparation. Completed prepared folders are never overwritten.
status.json reports progress. frame_*.npz contain water, momentum, layered
sediment inventory, secondary-flow state, bed elevation and cumulative budgets.
checkpoint.npz is an atomic recoverable checkpoint. The worker writes NPZ frames;
the QGIS plugin can materialize authenticated completed runs as temporal rasters.
Create the chosen STOP_GPU file to request a stop at an accepted-step boundary.
Remove/use a different stop marker before starting another run.
Resume into a NEW folder with --resume "C:\case\gpu_run\checkpoint.npz" and
the same prepared input and exact worker sources. A changed numerical-source
identity is rejected. --dt-max and output controls are explicit run overrides.
The on-demand AMR route instead saves four-file checkpoint directories. Read
the current relative checkpoint.path from run.json and pass that DIRECTORY to
--resume. Never choose a folder by its name alone; rejected-trial metadata can
publish an immutable revision. Retain the original deck/sidecars/hierarchy,
prepared session and selected worker directory for continuation and later
mesh construction. Viewing copied AMR results is independent of those inputs.

Integrity and notices
---------------------
PACKAGE-MANIFEST.json hashes every payload file (except itself) and records
native initializer, source and DLL provenance. The adjacent packaging receipt
hashes the ZIP and manifest. Paths in these receipts are package-relative;
the original host build receipt is identified by SHA-256 but is not shipped.
Validate this manifest after extraction. Do not
edit runtime sources in an active worker directory.
Clawpack BSD and GCC/MinGW/runtime/library notices are in licenses/. No Delft3D
source/runtime, user model, validation result, Python environment or GPU wheel
is distributed in this archive. No existing plugin/default is modified.
'''


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_name(value):
    name=str(value)
    if not name or name in ('.','..') or Path(name).name!=name or '/' in name or '\\' in name or ':' in name:
        raise ValueError('Expected safe direct-child name: '+name)
    return name


def forbidden_payload_members(names):
    """Identify data/reference trees and Python environments in a worker.

    The worker is assembled from authenticated allowlists, but this final
    independent audit prevents a later source-closure change from quietly
    turning it into a model-data bundle or Python distribution.  Native
    executables and their audited runtime DLLs are allowed; environments,
    wheels, bytecode, user decks/results and every Delft3D checkout are not.
    """
    raw_names={
        'case.data','boundary_ledger.txt','checkpoint.bin','frames.bin',
        'frames.txt','fort.q0000','fort.t0000','trim-basin.nc','trih-basin.nc',
    }
    raw_suffixes={
        '.asc','.csv','.feather','.h5','.hdf5','.nc','.npy','.npz','.parquet',
        '.pickle','.pkl','.pyc','.pyo','.tif','.tiff','.whl',
    }
    environment_parts={
        '.conda','.tox','.venv','conda-meta','dist-packages','env','envs',
        'site-packages','venv','__pycache__',
    }
    forbidden=[]
    for original in names:
        normalized=str(original).replace('\\','/')
        path=PurePosixPath(normalized)
        parts=tuple(part.lower() for part in path.parts)
        basename=parts[-1] if parts else ''
        if (path.is_absolute() or '..' in path.parts or
                any(part.startswith('delft3d') for part in parts) or
                'resources' in parts or
                any(part in environment_parts or part.startswith('.venv')
                    for part in parts) or
                basename in raw_names or Path(basename).suffix.lower() in raw_suffixes):
            forbidden.append(str(original))
    return forbidden


def _check_portable_json(value,member):
    """Reject machine paths in copied JSON rather than invalidating its pins."""
    if isinstance(value,dict):
        for key,item in value.items():
            _check_portable_json(key,member)
            _check_portable_json(item,member)
    elif isinstance(value,list):
        for item in value:_check_portable_json(item,member)
    elif isinstance(value,str):
        schema_pointer=(member==QGIS_PRODUCT_OUTPUT_OVERLAY and
            re.fullmatch(r'/[a-z][a-z0-9_]*(?:/[a-z][a-z0-9_]*)*',value))
        if (not schema_pointer and (PureWindowsPath(value).is_absolute() or
                re.search(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]',value) or
                value.startswith(('/',r'\\')))):
            raise ValueError('Machine-absolute path in packaged JSON: '+member)


def portable_initializer_receipt(path,identity):
    """Keep verified runtime identity, not the host-specific compiler command.

    The historical build receipt remains frozen at the source. Its SHA-256
    permits provenance audit without shipping the old compiler/checkout paths.
    The native runtime reads only the source/executable/parent identity fields.
    """
    allowed={'command','source_sha256','executable_sha256','parent_path',
             'parent_executable_sha256','driver_sha256','transformation',
             'production_modified','delft_modified'}
    if set(identity)-allowed:
        raise ValueError('Unaudited initializer build receipt fields')
    required={'source_sha256','executable_sha256','parent_executable_sha256',
              'driver_sha256'}
    if not required<=identity.keys():
        raise ValueError('Incomplete initializer build receipt')
    if (not isinstance(identity['source_sha256'],dict) or
            not identity['source_sha256']):
        raise ValueError('Incomplete initializer source identity')
    for name,digest in identity['source_sha256'].items():
        if (safe_name(name)!=name or not isinstance(digest,str) or
                not re.fullmatch(r'[0-9a-f]{64}',digest)):
            raise ValueError('Invalid initializer source identity')
    for key in ('executable_sha256','parent_executable_sha256','driver_sha256'):
        if (not isinstance(identity[key],str) or
                not re.fullmatch(r'[0-9a-f]{64}',identity[key])):
            raise ValueError('Invalid initializer '+key)
    portable=dict(schema=2,kind='portable native initializer build receipt',
        source_sha256=identity['source_sha256'],
        executable_sha256=identity['executable_sha256'],
        parent_executable_sha256=identity['parent_executable_sha256'],
        driver_sha256=identity['driver_sha256'],
        original_build_receipt_sha256=sha(path),
        historical_compiler_and_parent_runtime_not_bundled=True)
    _check_portable_json(portable,'initializer build receipt')
    return portable


def _runner_freeze_members(runner):
    """Read every branch of the public runner's frozen-source inventory.

    This deliberately uses the runner's AST, not a second hand-maintained
    subset.  A computed or renamed inventory fails packaging until audited.
    """
    tree=ast.parse(Path(runner).read_text(encoding='utf-8'))
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='freeze_sources']
    if len(functions)!=1:raise ValueError('Cannot locate one GPU runner freeze_sources inventory')
    groups={'names':set(),'runtime_modules':set()};assigned=set()
    for node in ast.walk(functions[0]):
        if isinstance(node,ast.Assign):
            targets=[target.id for target in node.targets if isinstance(target,ast.Name)]
        elif isinstance(node,ast.AugAssign) and isinstance(node.target,ast.Name):
            targets=[node.target.id]
        else:continue
        for target in targets:
            if target not in groups:continue
            try:value=ast.literal_eval(node.value)
            except (ValueError,TypeError,MemoryError,RecursionError) as exc:
                raise ValueError('Computed GPU freeze_sources inventory: '+target) from exc
            if not isinstance(value,(tuple,list)) or not all(isinstance(v,str) for v in value):
                raise ValueError('Nonliteral GPU freeze_sources inventory: '+target)
            expected='.cu' if target=='names' else '.py'
            for name in value:
                if safe_name(name)!=name or not name.endswith(expected):
                    raise ValueError('Unexpected frozen GPU source member: '+name)
            groups[target].update(value);assigned.add(target)
    if assigned!=set(groups) or not groups['names'] or not groups['runtime_modules']:
        raise ValueError('Incomplete GPU runner freeze_sources inventory')
    return groups['runtime_modules'],groups['names']


_WORKER_LOCAL_MODULE_PREFIXES=(
    'braids_gpu.',
    'validation.gpu_fwave.',
    # The public fixed-grid worker executes the schema-v2 validators directly.
    # Follow their headless sediment dependencies as ordinary source instead
    # of relying on a development checkout that will not exist after extract.
    'braids_core.sediment.',
)
_WORKER_LOCAL_MODULES=frozenset((
    'braids_core.gpu_capabilities',
    'braids_core.gpu_live_boundary_metadata',
    'braids_core.sediment_method_catalog',
    'braids_core.windows_paths',
))


def _worker_local_module_name(module):
    return (module in _WORKER_LOCAL_MODULES or
            module.startswith(_WORKER_LOCAL_MODULE_PREFIXES))


def _local_module(root,module):
    if not _worker_local_module_name(module):return None
    path=Path(root).joinpath(*module.split('.')).with_suffix('.py')
    return path if path.is_file() else None


def _module_name(root,path):
    return '.'.join(Path(path).relative_to(root).with_suffix('').parts)


def _local_imports(root,path):
    module=_module_name(root,path)
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    found=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            candidates=[alias.name for alias in node.names]
        elif isinstance(node,ast.ImportFrom):
            if node.level:
                parts=module.split('.')
                if node.level>len(parts):raise ValueError('Invalid local relative import: '+str(path))
                base='.'.join(parts[:-node.level])
                base=base+('.'+node.module if node.module else '')
            else:base=node.module or ''
            candidates=[base]
            if node.module is None or base in ('braids_gpu','validation.gpu_fwave'):
                candidates += [base+'.'+alias.name for alias in node.names if alias.name!='*']
        else:continue
        for candidate in candidates:
            if not _worker_local_module_name(candidate):continue
            resolved=_local_module(root,candidate)
            if resolved is not None:found.add(resolved)
            elif (isinstance(node,ast.Import) or
                  (isinstance(node,ast.ImportFrom) and
                   (candidate==base or node.module is None or
                    base in ('braids_gpu','validation.gpu_fwave')))):
                raise ValueError('Missing local GPU import '+candidate+' in '+str(path))
    return found,tree


def _module_local_cuda_literals(tree):
    """Resolve explicit module-directory CUDA paths without a global fallback.

    Only literal ``HERE / 'kernel.cu'`` expressions whose HERE is assigned the
    module's directory are admitted.  Missing adjacent files must remain errors
    even when an unrelated kernel with the same basename exists elsewhere.
    """
    directory_expressions={ast.dump(ast.parse(value,mode='eval').body)
        for value in ('Path(__file__).resolve().parent','Path(__file__).parent')}
    directory_names=set()
    for node in tree.body:
        if (isinstance(node,ast.Assign) and len(node.targets)==1 and
                isinstance(node.targets[0],ast.Name) and
                ast.dump(node.value) in directory_expressions):
            directory_names.add(node.targets[0].id)
    found={}
    for node in ast.walk(tree):
        if (isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div) and
                isinstance(node.left,ast.Name) and node.left.id in directory_names and
                isinstance(node.right,ast.Constant) and isinstance(node.right.value,str) and
                re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*\.cu',node.right.value)):
            found[id(node.right)]=node.right.value
    return found


def source_closure(root):
    """Overapproximate all public CLI branches without collecting run data.

    Includes the union of stock/optional `freeze_sources` branches, all local
    Python imports reachable from the public package, and top-level CUDA file
    literals used by those modules.  This is a source closure, not a claim that
    archived validation plans/inputs are available in the source worker.
    """
    root=Path(root).resolve();gpu=root/'validation/gpu_fwave';backend=root/'braids_gpu'
    modules,kernels=_runner_freeze_members(backend/'runner.py')
    pending=list(backend.glob('*.py'))+[gpu/name for name in modules]
    # NumPy-only shared native boundary admission, without importing QGIS/GDAL.
    pending.append(root/'braids_core/gpu_live_boundary_metadata.py')
    pending.append(root/'braids_core/gpu_capabilities.py')
    pending.append(root/'braids_core/sediment_method_catalog.py')
    pending.append(root/QGIS_PRODUCT_SOURCE)
    # Package the v118 lifecycle and everything it imports.  The worker still
    # runs outside QGIS and the selected backend remains explicit.
    pending.append(root/RECURSIVE_WORKER_SOURCE)
    # The worker validates the complete, canonical schema-v2 execution
    # preimage before CUDA starts.  Keep the executable validator in the
    # ordinary Python source closure; its generated JSON ledger is collected
    # separately because this function deliberately returns source files only.
    pending.append(root/EXECUTION_PROFILE_SOURCE)
    pending.append(root/EXECUTION_PROFILE_COMPLETENESS_SOURCE)
    # The first Q3-H3 route is admitted by an exact, post-FCT CPU/GPU source
    # pair.  Keep both members explicit even though ordinary import/companion
    # discovery also reaches them; the package contract must not depend on a
    # future refactor retaining those incidental links.
    pending.append(root/Q3H3_PROGNOSTIC_PYTHON_SOURCE)
    # Q3-C automatic compatibility resolution is part of the portable
    # source closure even though normal worker execution receives the already
    # resolved profile from QGIS.  Shipping the resolver keeps clean-extract
    # audit/replay independent of a development checkout.
    pending.append(root/'braids_core/sediment/capacity_foundation_profile.py')
    pending.append(root/'braids_core/windows_paths.py')
    pending.append(root/'braids_core/__init__.py')
    missing=[path for path in pending if not path.is_file()]
    if missing:raise ValueError('Missing GPU frozen Python source: '+str(missing[0]))
    python_files=set();cuda_files={gpu/name for name in kernels}
    # Q3-H2 loads this auxiliary live kernel through the authenticated profile
    # source inventory rather than a same-stem Python companion.  Keep it in
    # the clean worker closure explicitly so packaging cannot pass and then
    # fail only when classic equilibrium secondary flow compiles on CUDA.
    cuda_files.add(root/Q3H2_LIVE_CUDA_SOURCE)
    cuda_files.add(root/Q3H3_PROGNOSTIC_CUDA_SOURCE)
    # ``general_live_transport._source_files`` addresses this shared kernel
    # through ROOT / "braids_gpu/typed_boundary_roughness.cu" rather than a
    # module-local basename.  Keep it explicit so a clean extracted worker
    # cannot authenticate successfully and then fail at first compilation.
    cuda_files.add(root/'braids_gpu/typed_boundary_roughness.cu')
    while pending:
        path=pending.pop().resolve()
        if path in python_files:continue
        python_files.add(path)
        imports,tree=_local_imports(root,path)
        pending.extend(imports-python_files)
        module_local=_module_local_cuda_literals(tree)
        cuda_files.update(path.parent/name for name in module_local.values())
        # Every top-level runtime module is public worker source.  Include its
        # same-stem CUDA companion from either source namespace; otherwise a
        # clean extract can import the Python dispatcher but fails only when it
        # tries to compile a kernel that the manifest silently omitted.
        if path.parent in (gpu,backend):
            companion=path.with_suffix('.cu')
            if companion.is_file():cuda_files.add(companion)
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant) and isinstance(node.value,str):
                name=node.value
                # Archived variants construct additional CUDA files inside a
                # frozen run-source folder.  Such generated filenames are not
                # top-level worker sources and their variants are excluded.
                if (id(node) not in module_local and
                        re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*\.cu',name) and (gpu/name).is_file()):
                    cuda_files.add(gpu/name)
    missing=sorted(path for path in cuda_files if not path.is_file())
    if missing:raise ValueError('Missing referenced GPU CUDA source: '+str(missing[0]))
    return python_files,cuda_files


def _module_string_constants(tree):
    """Resolve only module-level literal strings, never execute package code."""
    constants={}
    for node in tree.body:
        if (isinstance(node,ast.Assign) and len(node.targets)==1 and
                isinstance(node.targets[0],ast.Name) and
                isinstance(node.value,ast.Constant) and
                isinstance(node.value.value,str)):
            name=node.targets[0].id
            if name in constants:raise ValueError('Repeated package CLI constant: '+name)
            constants[name]=node.value.value
    return constants


def _literal_string_choices(node,constants):
    if not isinstance(node,(ast.Tuple,ast.List)):
        raise ValueError('Computed GPU numerical-variant choices')
    result=[]
    for value in node.elts:
        if isinstance(value,ast.Constant) and isinstance(value.value,str):
            result.append(value.value)
        elif isinstance(value,ast.Name) and value.id in constants:
            result.append(constants[value.id])
        else:
            raise ValueError('Computed GPU numerical-variant choice')
    return result


def _variant_choices(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    constants=_module_string_constants(tree)
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute) or node.func.attr!='add_argument':continue
        if not node.args or not isinstance(node.args[0],ast.Constant) or node.args[0].value!='--gpu-numerical-variant':continue
        choices=[arg.value for arg in node.keywords if arg.arg=='choices']
        if len(choices)!=1:break
        return set(_literal_string_choices(choices[0],constants))
    raise ValueError('Cannot derive public GPU numerical-variant choices')


def _integrator_choices(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    constants=_module_string_constants(tree)
    for node in ast.walk(tree):
        if (not isinstance(node,ast.Call) or
                not isinstance(node.func,ast.Attribute) or
                node.func.attr!='add_argument' or not node.args or
                not isinstance(node.args[0],ast.Constant) or
                node.args[0].value!='--gpu-integrator'):
            continue
        choices=[arg.value for arg in node.keywords if arg.arg=='choices']
        if len(choices)==1:
            return set(_literal_string_choices(choices[0],constants))
        break
    raise ValueError('Cannot derive public GPU integrator choices')


def _runner_variants(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    constants=_module_string_constants(tree)
    for node in ast.walk(tree):
        if not isinstance(node,ast.Compare) or not isinstance(node.left,ast.Name) or node.left.id!='numerical_variant':continue
        if len(node.ops)!=1 or not isinstance(node.ops[0],ast.NotIn):continue
        try:result=_literal_string_choices(node.comparators[0],constants)
        except ValueError:continue
        if 'auto' in result:return set(result)
    raise ValueError('Cannot derive runner GPU numerical-variant choices')


def _runner_integrators(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    constants=_module_string_constants(tree)
    for node in ast.walk(tree):
        if (not isinstance(node,ast.Compare) or
                not isinstance(node.left,ast.Name) or node.left.id!='integrator' or
                len(node.ops)!=1 or not isinstance(node.ops[0],ast.NotIn)):
            continue
        try:
            result=set(_literal_string_choices(node.comparators[0],constants))
        except ValueError:
            continue
        if {'ssp43','ssp33'}<=result:
            return result
    raise ValueError('Cannot derive runner GPU integrator choices')


def _implicit_runtime_contract(root):
    """Authenticate the package declaration against literal runtime identities."""
    root=Path(root)
    capabilities=ast.parse((root/'braids_core/gpu_capabilities.py').read_text(
        encoding='utf-8'))
    labels=[]
    for node in capabilities.body:
        if (isinstance(node,ast.Assign) and len(node.targets)==1 and
                isinstance(node.targets[0],ast.Name) and
                node.targets[0].id=='GPU_INTEGRATOR_LABELS'):
            try:labels.append(ast.literal_eval(node.value))
            except (ValueError,TypeError,MemoryError,RecursionError) as exc:
                raise ValueError('Computed GPU integrator-label contract') from exc
    if len(labels)!=1 or labels[0].get(IMPLICIT_INTEGRATOR)!=(
            IMPLICIT_INTEGRATOR_CONTRACT['label']):
        raise ValueError('Implicit package/runtime label contract differs')

    runtime=ast.parse((root/'braids_gpu/implicit_midpoint_fwave.py').read_text(
        encoding='utf-8'))
    constants={}
    label_expression=None
    for node in runtime.body:
        if (isinstance(node,ast.Assign) and len(node.targets)==1 and
                isinstance(node.targets[0],ast.Name)):
            name=node.targets[0].id
            if isinstance(node.value,ast.Constant):
                constants[name]=node.value.value
            elif name=='LABEL':
                label_expression=node.value
    if (constants.get('TOKEN')!=IMPLICIT_INTEGRATOR or
            constants.get('METHOD_SCHEMA')!=IMPLICIT_INTEGRATOR_CONTRACT['schema'] or
            constants.get('CHECKPOINT_SCHEMA')!=
                IMPLICIT_INTEGRATOR_CONTRACT['checkpoint_schema']):
        raise ValueError('Implicit package/runtime schema contract differs')
    if not (isinstance(label_expression,ast.Call) and
            isinstance(label_expression.func,ast.Name) and
            label_expression.func.id=='gpu_integrator_label' and
            len(label_expression.args)==1 and
            isinstance(label_expression.args[0],ast.Name) and
            label_expression.args[0].id=='TOKEN' and
            not label_expression.keywords):
        raise ValueError('Implicit runtime label must use the shared label contract')
    return dict(IMPLICIT_INTEGRATOR_CONTRACT)


def _q3h3_prognostic_source_contract(root):
    """Authenticate the exact resident Q3-H3 Python/CUDA implementation.

    This is intentionally independent of the native v18 initializer.  The
    initializer authenticates the zero-step CPU export while this declaration
    binds the code that owns accepted prognostic state after CUDA starts.
    """
    root=Path(root)
    python_source=root/Q3H3_PROGNOSTIC_PYTHON_SOURCE
    cuda_source=root/Q3H3_PROGNOSTIC_CUDA_SOURCE
    if (not python_source.is_file() or python_source.is_symlink() or
            not cuda_source.is_file() or cuda_source.is_symlink()):
        raise ValueError('Q3-H3 prognostic source pair is absent or linked')
    if (sha(python_source)!=Q3H3_PROGNOSTIC_PYTHON_SHA or
            sha(cuda_source)!=Q3H3_PROGNOSTIC_CUDA_SHA):
        raise ValueError('Q3-H3 prognostic source identity differs')
    return dict(Q3H3_PROGNOSTIC_SOURCE_CONTRACT)


def _sediment_catalog_contract(root):
    """Read the packaged catalogue from ``root`` without importing checkout code."""
    root=Path(root)
    relative='braids_core/sediment_method_catalog.py'
    source=root/relative
    if not source.is_file() or source.is_symlink():
        raise ValueError('Sediment method catalogue source is absent or linked')
    namespace=runpy.run_path(str(source),run_name='_braids_packaged_sediment_catalog')
    schema=namespace.get('CATALOG_SCHEMA')
    scope_id=namespace.get('CATALOG_SCOPE_ID')
    integrator=namespace.get('GPU_INTEGRATOR_ID')
    semantic=namespace.get('semantic_catalog_id')
    if (type(schema) is not int or schema<=0 or
            not isinstance(scope_id,str) or not scope_id or
            integrator!=IMPLICIT_INTEGRATOR or not callable(semantic)):
        raise ValueError('Sediment method catalogue contract differs')
    semantic_id=semantic()
    if not isinstance(semantic_id,str) or not re.fullmatch(r'[0-9a-f]{64}',semantic_id):
        raise ValueError('Sediment method catalogue identity differs')
    return dict(schema=schema,scope_id=scope_id,
        semantic_catalog_id=semantic_id,source=relative,
        source_sha256=sha(source))


def _execution_profile_contract(root):
    """Authenticate the packaged schema-v2 validator and generated ledger.

    Run the source by path in an isolated interpreter so a packaging fixture
    cannot accidentally resolve ``braids_core`` from this development
    checkout.  This is identity validation only; it performs no solver or GPU
    work.
    """
    root=Path(root).resolve()
    source=root/EXECUTION_PROFILE_SOURCE
    completeness_source=root/EXECUTION_PROFILE_COMPLETENESS_SOURCE
    ledger=root/EXECUTION_PROFILE_LEDGER
    if (not source.is_file() or source.is_symlink() or
            not completeness_source.is_file() or completeness_source.is_symlink() or
            not ledger.is_file() or ledger.is_symlink()):
        raise ValueError(
            'Execution-profile validator/completeness policy or selector ledger is absent or linked')
    probe=(
        "import json,runpy,sys\n"
        "from pathlib import Path\n"
        f"root=Path({str(root)!r})\n"
        "sys.path.insert(0,str(root))\n"
        f"ns=runpy.run_path(str(root/{EXECUTION_PROFILE_SOURCE!r}),"
        "run_name='_braids_packaged_execution_profile_v2')\n"
        f"ledger=ns['load_selector_ledger'](root/{EXECUTION_PROFILE_LEDGER!r})\n"
        "print(json.dumps({"
        "'schema':ns['EXECUTION_PROFILE_SCHEMA'],"
        "'scope_id':ns['EXECUTION_PROFILE_SCOPE_ID'],"
        "'semantic_id':ns['execution_profile_semantic_id'](),"
        "'ledger_schema':ns['SELECTOR_LEDGER_SCHEMA'],"
        "'ledger_scope_id':ns['SELECTOR_LEDGER_SCOPE_ID'],"
        "'ledger_semantic_id':ledger['semantic_id'],"
        "'admission_registry_id':ledger['admission_registry_id'],"
        "'ledger_counts':ledger['counts'],"
        "'backends':[ns['BACKEND_CPU'],ns['BACKEND_GPU']]"
        "},sort_keys=True,separators=(',',':')))\n")
    completed=subprocess.run([sys.executable,'-I','-c',probe],cwd=root,
        capture_output=True,text=True,timeout=30)
    if completed.returncode!=0:
        raise ValueError('Execution-profile package contract differs: '+
                         completed.stderr.strip())
    try:
        contract=json.loads(completed.stdout)
    except (TypeError,ValueError,json.JSONDecodeError) as exc:
        raise ValueError('Execution-profile package contract is not JSON') from exc
    if (contract.get('schema')!=2 or contract.get('ledger_schema')!=2 or
            contract.get('backends')!=['cpu','implicit_fwave_midpoint_gpu'] or
            any(not isinstance(contract.get(key),str) or
                not re.fullmatch(r'[0-9a-f]{64}',contract[key])
                for key in ('semantic_id','ledger_semantic_id',
                            'admission_registry_id'))):
        raise ValueError('Execution-profile package identity differs')
    contract.update(source=EXECUTION_PROFILE_SOURCE,
        source_sha256=sha(source),
        completeness_source=EXECUTION_PROFILE_COMPLETENESS_SOURCE,
        completeness_source_sha256=sha(completeness_source),
        ledger=EXECUTION_PROFILE_LEDGER,
        ledger_sha256=sha(ledger))
    return contract


def _qgis_product_contract(root):
    """Authenticate the importable Q10 pre-allocation product resolver."""
    root=Path(root).resolve()
    source=root/QGIS_PRODUCT_SOURCE
    overlay=root/QGIS_PRODUCT_OUTPUT_OVERLAY
    if (not source.is_file() or source.is_symlink() or
            not overlay.is_file() or overlay.is_symlink()):
        raise ValueError('Q10 product resolver/output overlay is absent or linked')
    probe=(
        "import json,sys\n"
        "from pathlib import Path\n"
        f"root=Path({str(root)!r})\n"
        "sys.path.insert(0,str(root))\n"
        "from braids_core.sediment.qgis_product_v109 import ("
        "GPU_BACKEND,PRODUCT_ID,SCHEMA,product_catalogue_v109)\n"
        "from braids_core.sediment.cross_physics_v108 import ("
        "INTEGRATOR,cross_physics_catalogue_v108,output_registry_v108)\n"
        "catalogue=product_catalogue_v109()\n"
        "print(json.dumps({'schema':SCHEMA,'product_id':PRODUCT_ID,"
        "'product_catalogue_id':catalogue['identity'],"
        "'semantic_catalog_id':catalogue['semantic_catalog_id'],"
        "'cross_physics_catalogue_id':cross_physics_catalogue_v108()['identity'],"
        "'output_registry_id':output_registry_v108()['identity'],"
        "'backend':GPU_BACKEND,'integrator':INTEGRATOR,"
        "'automatic_backend_fallback':False},sort_keys=True,separators=(',',':')))\n")
    completed=subprocess.run([sys.executable,'-I','-c',probe],cwd=root,
        capture_output=True,text=True,timeout=30)
    if completed.returncode!=0:
        raise ValueError('Q10 product package contract differs: '+
                         completed.stderr.strip())
    try:
        contract=json.loads(completed.stdout)
    except (TypeError,ValueError,json.JSONDecodeError) as exc:
        raise ValueError('Q10 product package contract is not JSON') from exc
    if (contract.get('schema')!='braids.qgis_product.v109' or
            contract.get('backend')!='implicit_fwave_midpoint_gpu' or
            contract.get('integrator')!=IMPLICIT_INTEGRATOR or
            contract.get('automatic_backend_fallback') is not False or
            any(not isinstance(contract.get(key),str) or
                not re.fullmatch(r'[0-9a-f]{64}',contract[key]) for key in
                ('product_catalogue_id','semantic_catalog_id',
                 'cross_physics_catalogue_id','output_registry_id'))):
        raise ValueError('Q10 product package identity differs')
    return contract


def source_worker_scope(root, *, qgis_product_only=False):
    root=Path(root);public=_variant_choices(root/'braids_gpu/__main__.py')
    runner=_runner_variants(root/'braids_gpu/runner.py')
    if public!=runner:raise ValueError('Public and runner GPU numerical variants differ')
    public_integrators=_integrator_choices(root/'braids_gpu/__main__.py')
    runner_integrators=_runner_integrators(root/'braids_gpu/runner.py')
    if public_integrators!=runner_integrators:
        raise ValueError('Public and runner GPU integrator choices differ')
    if IMPLICIT_INTEGRATOR not in public_integrators:
        raise ValueError('Implicit GPU integrator is absent from public worker')
    implicit_contract=_implicit_runtime_contract(root)
    q3h3_prognostic_sources=_q3h3_prognostic_source_contract(root)
    sediment_catalog=_sediment_catalog_contract(root)
    execution_profile=_execution_profile_contract(root)
    qgis_product=_qgis_product_contract(root)
    recursive_worker_source=root/RECURSIVE_WORKER_SOURCE
    if not recursive_worker_source.is_file() or recursive_worker_source.is_symlink():
        raise ValueError('v118 recursive worker source is absent or linked')
    recursive_worker=dict(
        schema='braids.recursive_worker.v118',
        required_worker_variant='recursive_bank_amr_v118',
        source=RECURSIVE_WORKER_SOURCE,
        source_sha256=sha(recursive_worker_source),
        automatic_backend_fallback=False)
    if not set(ARCHIVE_BOUND_VARIANTS)<=public:
        raise ValueError('Archive-bound GPU variant was renamed without package policy update')
    # A newly introduced selectable path requires a deliberate source/archive
    # classification before a source worker may be packaged.
    admitted={'auto',INTEGRATED_VARIANT,TRANSPORT_VARIANT,BANK_VARIANT,
        AMR_GENERAL_VARIANT,
        'scheduled_inlet_ssp33','bedform_feedback_ssp33',
        'general_avalanche_plain_ssp33','physical_toe_mode2_plain_ssp33',
        'physical_toe_mode2_avalanche_ssp33','bedload_adaptation_plain_ssp33',
        GRADED_VARIANT,AMR_VARIANT}
    if public!=admitted|set(ARCHIVE_BOUND_VARIANTS):
        raise ValueError('Unclassified selectable GPU variant in source-worker package')
    if not set(EXTERNAL_PREPARATION_ONLY)<=admitted:
        raise ValueError('External-preparation-only variant was renamed without package policy update')
    delft_generations={
        'v42_law8':TRANSPORT_EXPORT_TARGET,
        'delft_components_v13_historical_q3sp':
            DELFT_TRANSPORT_EXPORT_V13_TARGET,
        'delft_components_v14':DELFT_TRANSPORT_EXPORT_TARGET,
        'v43_law9_coupled':GAIA_TRANSPORT_EXPORT_TARGET,
    }
    # Q3-H1 is additive.  A deliberately reduced fixture/package root that
    # does not contain v15 retains the exact historical generation map; the
    # production root must advertise the independently authenticated v15.
    if (root/DELFT_TRANSPORT_EXPORT_V15_SOURCE/'build.json').is_file():
        delft_generations['delft_components_v15_q3h1'] = \
            DELFT_TRANSPORT_EXPORT_V15_TARGET
    if (root/DELFT_TRANSPORT_EXPORT_V16_SOURCE/'build.json').is_file():
        delft_generations['delft_components_v16_q3h2'] = \
            DELFT_TRANSPORT_EXPORT_V16_TARGET
    if (root/DELFT_TRANSPORT_EXPORT_V17_SOURCE/'build.json').is_file():
        delft_generations['delft_components_v17_q3h2'] = \
            DELFT_TRANSPORT_EXPORT_V17_TARGET
    if (root/DELFT_TRANSPORT_EXPORT_V18_SOURCE/'build.json').is_file():
        delft_generations['delft_components_v18_q3h3'] = \
            DELFT_TRANSPORT_EXPORT_V18_TARGET
    classified_admitted=set(admitted)
    excluded=dict(ARCHIVE_BOUND_VARIANTS)
    if qgis_product_only:
        # M20 adds the qualified recursive bank/general AMR paths to the QGIS
        # desktop product while continuing to exclude unrelated archived
        # experiments and the stale graded-bed source archive.
        admitted={'auto',INTEGRATED_VARIANT,TRANSPORT_VARIANT,AMR_VARIANT}
        for name in sorted(classified_admitted-admitted):
            excluded[name]=(
                'not admitted by the QGIS v109 fixed-grid product worker')
    result=dict(supported_gpu_variants=sorted(admitted),
        native_preparation_integrators=['ssp43','legacy'],
        sediment_method_catalog=sediment_catalog,
        sediment_execution_profile_v2=execution_profile,
        qgis_product_v109=qgis_product,
        recursive_worker_v118=recursive_worker,
        implicit_integrators={
            IMPLICIT_INTEGRATOR:implicit_contract},
        q3h3_prognostic_secondary_sources=q3h3_prognostic_sources,
        excluded_gpu_variants=dict(sorted(excluded.items())),
        excluded_prepared_sidecars=dict(sorted(ARCHIVE_BOUND_SIDECARS.items())),
        allowed_prepared_sidecars_by_variant={name:sidecars for name,sidecars in
            VARIANT_PREPARED_SIDECARS.items() if name in admitted},
        source_archives_by_variant={name:archive for name,archive in
            VARIANT_SOURCE_ARCHIVES.items() if name in admitted},
        general_amr_selector_policies_by_variant=(
            {AMR_GENERAL_VARIANT:['local_eta_plane_v1']}
            if AMR_GENERAL_VARIANT in admitted else {}),
        native_exporters_by_variant={name:target for name,target in {
            AMR_VARIANT:AMR_RUNTIME_TARGET,
            AMR_GENERAL_VARIANT:AMR_GENERAL_RUNTIME_TARGET,
            TRANSPORT_VARIANT:GAIA_TRANSPORT_EXPORT_TARGET,
            BANK_VARIANT:BANK_EXPORT_TARGET}.items() if name in admitted},
        native_exporter_generations_by_variant=(
            {TRANSPORT_VARIANT:delft_generations}
            if TRANSPORT_VARIANT in admitted else {}),
        recursive_amr_by_variant={name:contract for name,contract in {
            AMR_VARIANT:dict(RECURSIVE_AMR_CONTRACT),
            AMR_GENERAL_VARIANT:dict(RECURSIVE_GENERAL_AMR_CONTRACT)}.items()
            if name in admitted},
        recursive_bedload_adaptation_by_variant={name:contract
            for name,contract in {
                AMR_VARIANT:dict(RECURSIVE_ADAPTATION_CONTRACT),
                AMR_GENERAL_VARIANT:dict(RECURSIVE_GENERAL_ADAPTATION_CONTRACT)}.items()
            if name in admitted},
        recursive_general_avalanche_by_variant=(
            {AMR_GENERAL_VARIANT:dict(RECURSIVE_GENERAL_AVALANCHE_CONTRACT)}
            if AMR_GENERAL_VARIANT in admitted else {}),
        recursive_general_viscosity_by_variant=(
            {AMR_GENERAL_VARIANT:dict(RECURSIVE_GENERAL_VISCOSITY_CONTRACT)}
            if AMR_GENERAL_VARIANT in admitted else {}),
        recursive_general_bedform_by_variant=(
            {AMR_GENERAL_VARIANT:dict(RECURSIVE_GENERAL_BEDFORM_CONTRACT)}
            if AMR_GENERAL_VARIANT in admitted else {}),
        recursive_bank_bedform_avalanche_by_variant=(
            {AMR_VARIANT:dict(RECURSIVE_BANK_BEDFORM_AVALANCHE_CONTRACT)}
            if AMR_VARIANT in admitted else {}),
        recursive_bank_bedform_native_exporters_by_variant=(
            {AMR_VARIANT:AMR_BEDFORM_BANK_RUNTIME_TARGET}
            if AMR_VARIANT in admitted else {}),
        requires_external_prepared_sidecars=dict(sorted(
            (name,sidecar) for name,sidecar in EXTERNAL_PREPARATION_ONLY.items()
            if name in admitted)),
        restriction=(
            'QGIS v109 desktop product worker; fixed-grid plus qualified '
            'recursive bank AMR; validation inputs/outputs excluded'
            if qgis_product_only else
            'source-only worker; integrated fixed-grid transport/bank and '
            'on-demand bank/general AMR routes; validation inputs/outputs and '
            'other research archives excluded'))
    if (not qgis_product_only and AMR_VARIANT in admitted and
            (root/AMR_LIVE_RUNTIME_SOURCE/'build.json').is_file()):
        result['evolving_native_exporters_by_variant']={AMR_VARIANT:AMR_LIVE_RUNTIME_TARGET}
    if (AMR_VARIANT in admitted and
            (root/AMR_BEDFORM_PHYSICAL_TOE_RUNTIME_SOURCE/'build.json').is_file()):
        result['recursive_physical_toe_bedform_avalanche_by_variant']={
            AMR_VARIANT:dict(RECURSIVE_PHYSICAL_TOE_BEDFORM_AVALANCHE_CONTRACT)}
        result['recursive_physical_toe_bedform_native_exporters_by_variant']={
            AMR_VARIANT:AMR_BEDFORM_PHYSICAL_TOE_RUNTIME_TARGET}
    return result


def declared_source_archive(root, relative, variant):
    """Authenticate a fixture-independent numerical bundle for packaging.

    The current source-worker scope has no admitted archive.  When the graded
    route is promoted, this guard requires a pinned, complete payload under
    the worker's source namespace rather than silently copying a validation
    run or any host-specific path into the release.
    """
    root=Path(root).resolve()
    if not isinstance(relative,str):
        raise ValueError('Invalid packaged GPU source archive path')
    name=PurePosixPath(relative)
    if ('\\' in relative or ':' in relative or
            name.is_absolute() or name.as_posix()!=relative or
            len(name.parts)<3 or name.parts[:2]!=('braids_gpu','frozen_sources') or
            any(part in ('','.','..') for part in name.parts)):
        raise ValueError('Invalid packaged GPU source archive path')
    folder=root.joinpath(*name.parts)
    if (folder.is_symlink() or not folder.is_dir() or
            not folder.resolve().is_relative_to(root)):
        raise ValueError('Packaged GPU source archive is absent or linked')
    manifest=folder/'runtime_manifest.json'
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError('Packaged GPU source archive has no regular runtime manifest')
    identity=json.loads(manifest.read_text(encoding='utf-8'))
    if (identity.get('schema')!=1 or identity.get('variant')!=variant or
            identity.get('preparation_contract_schema')!=1):
        raise ValueError('Packaged GPU source archive contract differs')
    pins={'runner_sha256':root/'braids_gpu/runner.py',
          'adapter_sha256':root/'braids_gpu/live_bedload_adaptation.py',
          'prebuilt_adapter_sha256':root/'braids_gpu/graded_bedload_prebuilt.py',
          'ancestor_sha256':root/'validation/gpu_fwave/a1_combined_current_adapter.py',
          'transaction_sha256':root/'braids_gpu/a1_full_array_transaction.py',
          'candidate_receipt_sha256':folder/'candidate_sources.json'}
    if any(not source.is_file() or identity.get(key)!=sha(source)
           for key,source in pins.items()):
        raise ValueError('Packaged GPU source archive is stale against current code')
    files=identity.get('source_files')
    transaction=identity.get('transaction_sources')
    if (not isinstance(files,dict) or not files or
            not isinstance(transaction,dict) or not transaction or
            'candidate_sources.json' not in files):
        raise ValueError('Packaged GPU source archive has an incomplete inventory')
    actual={path.relative_to(folder).as_posix() for path in folder.rglob('*')
            if path.is_file()}
    if any(path.is_symlink() for path in folder.rglob('*')) or actual!=set(files)|{'runtime_manifest.json'}:
        raise ValueError('Packaged GPU source archive inventory differs')
    for member,digest in files.items():
        path=PurePosixPath(member)
        if (not isinstance(member,str) or not member or '\\' in member or ':' in member or
                path.is_absolute() or path.as_posix()!=member or
                any(part in ('','.','..') for part in path.parts) or
                not isinstance(digest,str) or not re.fullmatch(r'[0-9a-f]{64}',digest) or
                sha(folder.joinpath(*path.parts))!=digest):
            raise ValueError('Packaged GPU source archive file differs: '+str(member))
    for member,digest in transaction.items():
        if safe_name(member)!=member or files.get('transaction_python/'+member)!=digest:
            raise ValueError('Packaged GPU transaction source differs: '+str(member))
    return {folder/member for member in actual}


def default_initializer_tag(root):
    tree=ast.parse((Path(root)/'validation/gpu_fwave/coupled_reference.py').read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='TAG' for v in node.targets):
            return safe_name(ast.literal_eval(node.value))
    raise ValueError('Native initializer default TAG is missing')


def dll_imports(binary,objdump):
    result=subprocess.run([str(objdump),'-p',str(binary)],capture_output=True,text=True,check=True,timeout=30)
    return sorted(set(re.findall(r'DLL Name:\s*([^\s]+)',result.stdout)),key=str.lower)


def runtime_dependencies(executable,dll_dir,objdump,inspect=dll_imports):
    """Resolve the entire PE dependency closure; reject unknown redistributions."""
    pending=[Path(executable)];graph={};dependencies={}
    while pending:
        binary=pending.pop();imports=inspect(binary,objdump);graph[binary.name]=imports
        for imported in imports:
            name=safe_name(imported);key=name.lower()
            if key in SYSTEM_DLLS or key.startswith(('api-ms-win-','ext-ms-win-')):continue
            if key not in RUNTIME_DLLS:raise ValueError('Unapproved runtime DLL: '+name)
            if key in dependencies:continue
            source=Path(dll_dir)/key
            if not source.is_file():raise ValueError('Missing required runtime DLL: '+str(source))
            dependencies[key]=source;pending.append(source)
    return dependencies,graph


def graded_exporter_payloads(root,*,dll_dir,objdump,inspect=dll_imports):
    """Pin the no-compiler, I/O-only native exporter and its source/DLL closure.

    The historical build receipt contains host-absolute compiler paths.  Its
    content hashes are used here, but only verified local source files and the
    executable enter the portable worker; that absolute-path receipt does not.
    """
    root=Path(root).resolve();runtime=root/GRADED_EXPORT_SOURCE
    built=json.loads((runtime/'build.json').read_text(encoding='utf-8'))
    executable=runtime/'xbraids_normal_export.exe'
    if (built.get('executable_sha256')!=GRADED_EXPORT_SHA or
            sha(executable)!=GRADED_EXPORT_SHA):
        raise ValueError('Frozen graded native exporter executable differs')
    private=built.get('private_files')
    if not isinstance(private,dict):
        raise ValueError('Frozen graded native exporter has no source provenance')
    private_by_name={}
    for original,digest in private.items():
        name=Path(original).name
        if name in private_by_name and private_by_name[name]!=digest:
            raise ValueError('Ambiguous graded native exporter source name: '+name)
        private_by_name[name]=digest
    source_names={Path(part).name for part in built.get('command',())
                  if isinstance(part,str) and Path(part).suffix.lower() in
                  ('.f','.f90','.f95')}
    if not source_names or 'cutcell_solver.f90' not in source_names:
        raise ValueError('Frozen graded native exporter has no complete source list')
    sources={}
    for name in sorted(source_names):
        safe_name(name)
        source=runtime/name
        if (source.is_symlink() or not source.is_file() or
                private_by_name.get(name)!=sha(source)):
            raise ValueError('Frozen graded native exporter source differs: '+name)
        sources[f'{GRADED_EXPORT_TARGET}/source/{name}']=source
    dependencies,imports=runtime_dependencies(executable,dll_dir,objdump,inspect)
    payloads={f'{GRADED_EXPORT_TARGET}/{executable.name}':executable,**sources}
    for name,path in dependencies.items():
        payloads[f'{GRADED_EXPORT_TARGET}/{name}']=path
    return payloads,dict(executable_sha256=GRADED_EXPORT_SHA,
        source_sha256={name:sha(source) for name,source in sources.items()},
        runtime_imports=imports,
        dll_sha256={name:sha(path) for name,path in sorted(dependencies.items())})


def integrated_exporter_payloads(root, *, source, target, executable_name,
                                 expected_sha256, source_hash_key, unchanged_key,
                                 steps_key, dll_dir, objdump, inspect=dll_imports):
    """Copy one pinned BRAIDS zero-step exporter into a package-local runtime."""
    runtime=Path(root).resolve()/source
    built=json.loads((runtime/'build.json').read_text(encoding='utf-8'))
    executable=runtime/executable_name
    source_hashes=built.get(source_hash_key)
    names={Path(part).name for part in built.get('command',())
           if isinstance(part,str) and Path(part).suffix.lower() in ('.f','.f90','.f95')}
    if (built.get('executable_sha256')!=expected_sha256 or sha(executable)!=expected_sha256 or
            built.get(unchanged_key) is not True or built.get(steps_key)!=0 or
            built.get('Delft3D_source_modified') is not False or
            not isinstance(source_hashes,dict) or not names or
            names!=set(source_hashes) or 'cutcell_solver.f90' not in names):
        raise ValueError('Pinned integrated native exporter provenance differs: '+source)
    payloads={target+'/'+executable_name:executable}
    for name,digest in sorted(source_hashes.items()):
        safe_name(name)
        member=runtime/name
        if (member.is_symlink() or not member.is_file() or
                not re.fullmatch(r'[0-9a-f]{64}',digest) or sha(member)!=digest):
            raise ValueError('Pinned integrated native source differs: '+name)
        payloads[target+'/source/'+name]=member
    dependencies,imports=runtime_dependencies(executable,dll_dir,objdump,inspect)
    for name,path in dependencies.items():
        payloads[target+'/'+name]=path
    receipt=dict(schema=1,kind='portable integrated native initializer',
        executable=executable_name,executable_sha256=expected_sha256,
        source_sha256=dict(sorted(source_hashes.items())),
        parent_executable_sha256=built['parent_executable_sha256'],
        original_build_receipt_sha256=sha(runtime/'build.json'),
        numerical_source_unchanged=True,native_solver_steps_performed=0,
        Delft3D_source_modified=False,
        runtime_imports=imports,
        dll_sha256={name:sha(path) for name,path in sorted(dependencies.items())})
    _check_portable_json(receipt,target+'/build.json')
    return payloads,receipt


def amr_exporter_payloads(root,*,dll_dir,objdump,inspect=dll_imports):
    """Ship only frozen geometry initializer/code, never its test inputs/results."""
    from braids_gpu.amr_portable_runtime import portable_build_receipt,EXECUTABLE
    from braids_gpu.amr_portable_runtime import current_interface_kernel_sources,REVISED_KERNEL_ABIS
    from braids_gpu.amr_evolving_runtime import current_sources,KERNEL_ABIS
    root=Path(root)
    runtime=root/AMR_RUNTIME_SOURCE
    build=portable_build_receipt(runtime)
    dependencies,imports=runtime_dependencies(runtime/EXECUTABLE,dll_dir,objdump,inspect)
    payloads={AMR_RUNTIME_TARGET+'/'+name:runtime/name for name in
        (*build['source_sha256'],EXECUTABLE)}
    for name,path in dependencies.items():
        payloads[AMR_RUNTIME_TARGET+'/'+name]=path
    source=root/AMR_KERNEL_SOURCE
    if sha(source/'runtime_manifest.json')!=AMR_KERNEL_MANIFEST_SHA:
        raise ValueError('AMR qualified numerical archive changed')
    archived=json.loads((source/'runtime_manifest.json').read_text())
    kernels={name.removeprefix('kernels/'):digest for name,digest in archived['source_files'].items()
             if name.startswith('kernels/')}
    if not kernels or {p.name for p in (source/'kernels').iterdir()}!=set(kernels):
        raise ValueError('AMR qualified CUDA inventory differs')
    for name,digest in kernels.items():
        if safe_name(name)!=name or not name.endswith('.cu') or sha(source/'kernels'/name)!=digest:
            raise ValueError('AMR qualified CUDA source differs')
    selected,pair=current_interface_kernel_sources(root,source/'kernels')
    added,evolving_pair=current_sources(root)
    if set(added)&set(selected):
        raise ValueError('Evolving AMR added kernels overlap the immutable baseline inventory')
    selected.update(added)
    for name,path in selected.items():
        payloads[VARIANT_SOURCE_ARCHIVES[AMR_VARIANT]+'/'+name]=path
    kernel_manifest=dict(schema=2,variant=AMR_VARIANT,
        source_files={name:sha(path) for name,path in selected.items()},
        numerical_cuda_unchanged=False,intentional_cuda_revisions=sorted(REVISED_KERNEL_ABIS),
        interface_source_pair=pair,parent_source_manifest_sha256=AMR_KERNEL_MANIFEST_SHA,
        added_cuda_kernels=sorted(KERNEL_ABIS),
        evolving_interface_source_pair=evolving_pair,
        parent_cuda_sha256=kernels,
        qualification='Revised source pairing; coupled AMR requalification is recorded separately')
    return payloads,dict(executable_sha256=build['executable_sha256'],runtime_imports=imports,
        dll_sha256={name:sha(path) for name,path in sorted(dependencies.items())},
        build=build,kernel_manifest=kernel_manifest,
        geometry_namespace='standalone NumPy geometry I/O; QGIS boundary package initializer not imported')


def amr_general_exporter_payloads(root,*,dll_dir,objdump,inspect=dll_imports):
    """Ship the pinned schema-3 nonbank AMR zero-step native initializer."""
    from braids_gpu.amr_portable_runtime import (
        GENERAL_EXECUTABLE, GENERAL_EXPORTER_SHA256, portable_general_build_receipt)
    runtime=Path(root).resolve()/AMR_GENERAL_RUNTIME_SOURCE
    build=portable_general_build_receipt(runtime)
    if build['executable_sha256']!=GENERAL_EXPORTER_SHA256:
        raise ValueError('General AMR native executable identity differs')
    payloads={AMR_GENERAL_RUNTIME_TARGET+'/'+name:runtime/name for name in
        (*build['source_sha256'],GENERAL_EXECUTABLE)}
    dependencies,imports=runtime_dependencies(runtime/GENERAL_EXECUTABLE,
        dll_dir,objdump,inspect)
    for name,path in dependencies.items():
        payloads[AMR_GENERAL_RUNTIME_TARGET+'/'+name]=path
    return payloads,dict(executable_sha256=build['executable_sha256'],
        runtime_imports=imports,
        dll_sha256={name:sha(path) for name,path in sorted(dependencies.items())},
        build=build,geometry_namespace='standalone NumPy schema-3 general geometry I/O')


def general_amr_source_revision(root):
    """Authenticate the additive v8 numerical revision and both gate receipts."""
    from braids_gpu.amr_portable_runtime import GENERAL_SOURCE_ARCHIVE_REVISION
    root=Path(root).resolve();revision=GENERAL_SOURCE_ARCHIVE_REVISION
    if (revision.get('archive')!=VARIANT_SOURCE_ARCHIVES[AMR_GENERAL_VARIANT] or
            revision.get('qualified_source_manifest_sha256')!=
                AMR_GENERAL_QUALIFIED_SOURCE_SHA):
        raise ValueError('General AMR v8 archive/source qualification identity differs')
    documents={}
    for kind in ('route_receipt','component_receipt'):
        identity=revision.get(kind,{});relative=identity.get('path')
        path=root/relative if isinstance(relative,str) else root/'__invalid__'
        if (not path.is_file() or path.is_symlink() or
                sha(path)!=identity.get('sha256')):
            raise ValueError('General AMR v8 '+kind.replace('_',' ')+' differs')
        documents[kind]=json.loads(path.read_text(encoding='utf-8'))
    route=documents['route_receipt'];component=documents['component_receipt']
    route_provenance=route.get('provenance',{})
    component_sources=component.get('provenance',{}).get(
        'braids_gpu',{}).get('sources',{})
    revised=revision['numerical_source_revisions']['sediment_transport.cu']['sha256']
    if (route.get('status')!='passed' or not all(route.get('gates',{}).values()) or
            route_provenance.get('source_manifest_sha256')!=
                AMR_GENERAL_QUALIFIED_SOURCE_SHA or
            route_provenance.get('cuda_source_sha256')!=revised or
            route_provenance.get('component_receipt_sha256')!=
                revision['component_receipt']['sha256'] or
            component.get('passed') is not True or
            component_sources.get('validation/gpu_fwave/sediment_transport.cu')!=revised):
        raise ValueError('General AMR v8 qualification receipts are incomplete')
    return revision


def amr_general_source_payloads(root, worker_files, bank_payloads, bank_pair):
    """Pin the reusable schema-3 CUDA closure to the qualified public run.

    Per-deck live-transport kernels are generated again from packaged source;
    the 767 generated files in the qualification run are not distributed.
    Only explicitly listed package-admission revisions may differ from the
    checkout qualification snapshot; the extracted worker is run separately.
    """
    root=Path(root).resolve()
    archive_revision=general_amr_source_revision(root)
    reference=root/AMR_GENERAL_QUALIFIED_SOURCE
    if sha(reference)!=AMR_GENERAL_QUALIFIED_SOURCE_SHA:
        raise ValueError('General AMR qualification source receipt changed')
    qualified=json.loads(reference.read_text(encoding='utf-8'))
    if (qualified.get('schema')!=2 or qualified.get('variant')!=AMR_GENERAL_VARIANT or
            not isinstance(qualified.get('source_files'),dict)):
        raise ValueError('General AMR qualification source contract differs')
    archive=VARIANT_SOURCE_ARCHIVES[AMR_GENERAL_VARIANT]
    static={};pins={};revisions={}
    # A newly qualified public source snapshot must already contain every
    # selector-policy and runtime admission change. Do not silently ship code
    # different from that snapshot under an old qualification identity.
    allowed_package_revisions={
            'braids_gpu/amr_native_geometry.py':
                'Adds a fail-closed export-schema compatibility view for the exact authenticated historical bedlaw-2 parent after the law-9 fixed-face records were added; native numerical routines remain byte-identical.',
            'braids_gpu/amr_live_native_geometry.py':
                'Selects a freshly authenticated dependency-refrozen zero-step live-boundary initializer; its export and numerical method are unchanged.',
            'braids_gpu/amr_bank_native_geometry.py':
                'Selects a freshly authenticated dependency-refrozen mode-2/3 zero-step initializer; its export and numerical method are unchanged.',
            'braids_gpu/amr_bank_bedform_native_geometry.py':
                'Selects a freshly authenticated dependency-refrozen recursive mode-3 bedform/bank zero-step initializer; its physics and numerical method are unchanged.',
            'braids_gpu/amr_bank_physical_toe_bedform_native_geometry.py':
                'Selects a freshly authenticated dependency-refrozen recursive mode-2 physical-toe zero-step initializer; its physics and numerical method are unchanged.',
            'braids_gpu/amr_general_native_geometry.py':
                'Selects a freshly authenticated dependency-refrozen nonbank zero-step initializer; its export and numerical method are unchanged.',
        'braids_gpu/__main__.py':
            'Selects the versioned graded archive and exposes the explicit implicit-f-wave-midpoint token; the separate implicit capability gate owns admission and AMR numerical operators are unchanged.',
        'braids_gpu/runner.py':
            'Routes the explicit implicit-f-wave-midpoint token only to common fixed-grid transport and rejects it on every historical route; AMR numerical operators are unchanged.',
            'braids_gpu/fixed_grid_lifecycle.py':
                'Fixed-grid lifecycle records the selected explicit integrator; AMR owns a separate lifecycle.',
            'braids_core/gpu_capabilities.py':
                'Adds the explicit implicit-f-wave-midpoint selector label shared by the host and worker; route selection and all AMR numerical operators are unchanged.',
            'braids_gpu/general_live_transport.py':
            'Retains fixed-grid stage ownership and adds the exact shared legacy bedform ON/OFF envelope plus authenticated recursive stage/pre-friction method identity.',
        'braids_gpu/g1_inlet_solver.py':
            'Updates only the authenticated source-transform anchor for the current boundary constructor; the G1 numerical inlet implementation is unchanged.',
        'braids_gpu/general_live_transport_runner.py':
            'Binds the qualified fixed-grid method identity and restart contract; AMR runners are separate.',
        'braids_gpu/amr_bank_subcycle.py':
            'Uses the shared arbitrary-level clock/register scheduler and admits the external accepted-integral physical-toe mode-2 transaction; the bank operator remains one conservative macro-step commit.',
        'braids_gpu/amr_general_checkpoint.py':
            'Adds fail-closed schema-6 persistence and exact restoration of recursive nonbank level clocks, accepted registers and source/exchange diagnostics; historical synchronous schema 5 is unchanged.',
        'braids_gpu/amr_general_route.py':
            'Factors the accepted-clock topology handoff so the recursive subclass can carry its closed-epoch clock and conservative register diagnostics without changing synchronous remap behavior.',
        'braids_gpu/amr_general_prepared.py':
            'Admits authenticated topology-local avalanche metadata for the recursive nonbank parent transaction while preserving bank exclusions.',
        'braids_gpu/amr_general_selector.py':
            'Admits the exact unscheduled mode-1 discharge/mode-2 stage legacy bedform ON/OFF envelope without expanding recursive viscosity, avalanche or bank composition.',
        'braids_gpu/amr_general_stepper.py':
            'Exposes the resident viscosity and general-avalanche operators to their recursive owner without changing synchronous execution order.',
        'braids_gpu/amr_general_subcycle_stepper.py':
            'Adds level-local bedform capacity and prepared pre-friction evaluations with explicit per-level counts while retaining the qualified parent viscosity/avalanche transaction and rollback.',
        'braids_gpu/amr_level_euler.py':
            'Admits externally owned physical-toe mode-2 trials to the same unsuspended level-local Euler stage used by recursive distributed banks; no CUDA interface flux is changed.',
        'braids_gpu/amr_on_demand_runner.py':
            'Admits and identifies the exact recursive legacy bedform ON/OFF pair, publishes its explicit trace contract and retains prior recursive viscosity/avalanche routes.',
        'braids_gpu/amr_on_demand.py':
            'Dispatches the separately versioned exact mode-3 bedform/bank initializer while preserving all historical initializer identities.',
        'braids_gpu/amr_prepared.py':
            'Authenticates the separately versioned exact mode-3 bedform/bank initializer and sidecar inventory without broadening historical mode dispatch.',
        'braids_gpu/integrated.py':
            'Forwards only the explicit exact bedform-bank initializer selection on the evolving bank AMR route.',
        'braids_gpu/amr_portable_runtime.py':
            'Selects the separately qualified general v8 numerical archive while retaining read-only compatibility with all earlier archive names.',
        'braids_gpu/amr_recursive_route.py':
            'Adds static/evolving physical-toe mode-2 routing, schema-2 exact restart fields and closed-epoch regrid handoff while retaining the mode-3 schema-1 route.',
            'validation/gpu_fwave/coupled_solver.py':
                'Adds explicit all-face ownership for the bounded recursive bedform/mode-3-bank/parent-avalanche composition and keeps its derived Manning buffer transactionally independent.',
            'validation/gpu_fwave/boundaries.py':
                'Adds the accepted law-9 fixed initial face-bed inlet trace and dynamic hydrostatic correction, qualified by the paired cold-1200 and definitive Yen runs.',
            'validation/gpu_fwave/coupled_reference.py':
                'Exports and validates the explicit native-v5 law-9 fixed-face arrays while retaining authenticated historical snapshots without inventing the new semantics.',
            'validation/gpu_fwave/i4_mode3_avalanche_current_gate.py':
                'Removes a validation-audit write side effect; no numerical or runtime path is changed.',
            'validation/gpu_fwave/i4_mode3_avalanche_gpu_readback.py':
                'Authenticates the archived historical input-writer identity for read-only validation evidence without relabelling it as current.',
            'validation/gpu_fwave/bedform_roughness.py':
                'Admits only the explicit recursive mode-3 bank and parent-avalanche composition; the bedform equations and CUDA kernel are unchanged.',
            'validation/gpu_fwave/avalanche_operator.py':
                'Corrects the stale integration-status docstring after fixed-grid and recursive QGIS/GPU admission; numerical implementation is unchanged.',
            'validation/gpu_fwave/toe_face_diagnostic.py':
                'Adds a validated conservative seam that consumes an already SSPRK-weighted root Exner face integral without reweighting it.',
            'validation/gpu_fwave/secondary_transport.py':
                'Adds constant-Chezy steering only for fixed-grid Engelund-Hansen equilibrium model 1; the qualified AMR MPM branches retain their original controls and CUDA ABI.',
            'validation/gpu_fwave/sediment_transport.py':
                'Adds the fail-closed fixed-grid Engelund-Hansen constant-Chezy contract and corrected post-rotation donor ownership; qualified AMR MPM selection and CUDA interfaces remain unchanged.',
    }
    for name,digest in sorted(qualified['source_files'].items()):
        if name.startswith('python/'):
            local=name.removeprefix('python/')
            if local in worker_files:
                current=sha(worker_files[local])
                if current!=digest:
                    if local not in allowed_package_revisions:
                        raise ValueError('Qualified general AMR source changed: '+local)
                    revisions[local]=dict(qualified_sha256=digest,
                        packaged_sha256=current,reason=allowed_package_revisions[local])
                pins[local]=current
        elif name.startswith('kernels/') and not name.startswith('kernels/live_transport/'):
            kernel=name.removeprefix('kernels/')
            safe_name(kernel)
            source=bank_payloads.get(VARIANT_SOURCE_ARCHIVES[AMR_VARIANT]+'/'+kernel)
            if source is None or not source.is_file() or sha(source)!=digest:
                # A current fixed-grid kernel may advance independently of the
                # already-qualified AMR archive. Prefer the immutable source
                # beside the qualification manifest; never relabel live HEAD
                # as the older qualified AMR implementation.
                qualified_source=reference.parent/name
                source=(qualified_source if qualified_source.is_file() and
                        sha(qualified_source)==digest else root/'validation/gpu_fwave'/kernel)
            if not source.is_file() or sha(source)!=digest:
                raise ValueError('Qualified general AMR CUDA source changed: '+kernel)
            static[archive+'/'+kernel]=source
    if (len(static)<50 or len(pins)<100 or
            archive+'/amr_general_indicators.cu' not in static):
        raise ValueError('General AMR static source closure is incomplete')
    manifest=dict(schema=1,variant=AMR_GENERAL_VARIANT,
        native_preparation_schema=3,source_files={
            name.removeprefix(archive+'/'):sha(source) for name,source in sorted(static.items())},
        archive_revision=archive_revision,
        interface_source_pair=bank_pair,
        worker_python_sha256=dict(sorted(pins.items())),
        qualified_public_source_manifest_sha256=AMR_GENERAL_QUALIFIED_SOURCE_SHA,
        qualified_public_source_identity=AMR_GENERAL_QUALIFIED_SOURCE_SHA,
        packaging_source_revisions=dict(sorted(revisions.items())),
        generated_live_transport_sources_bundled=False,
        general_transport_source_generated_per_deck=True)
    return static,manifest


def collect(root,*,dll_dir,objdump,license_dir,inspect=dll_imports,
            qgis_product_only=False):
    root=Path(root).resolve();gpu=root/'validation/gpu_fwave';tag=default_initializer_tag(root)
    scope=(source_worker_scope(root,qgis_product_only=True)
           if qgis_product_only else source_worker_scope(root))
    python_files,cuda_files=source_closure(root)
    runtime=gpu/'runs'/tag/'runtime_export';identity=json.loads((runtime/'build.json').read_text())
    files={};expected={}
    def add(source,target,digest=None):
        source=Path(source)
        if source.is_symlink():raise ValueError('Linked payload file: '+str(source))
        source=source.resolve();target=PurePosixPath(target).as_posix()
        if not source.is_file():raise ValueError('Missing payload file: '+str(source))
        if PurePosixPath(target).is_absolute() or '..' in PurePosixPath(target).parts or target in files:
            raise ValueError('Invalid/duplicate package member: '+target)
        measured=sha(source)
        if digest is not None and measured!=digest:raise ValueError('Authenticated source changed: '+str(source))
        if target.endswith('.json'):
            _check_portable_json(json.loads(source.read_text(encoding='utf-8')),target)
        files[target]=source;expected[target]=measured
    for source in sorted(python_files):add(source,source.relative_to(root).as_posix())
    # JSON is executable configuration data for the schema-v2 validator, not
    # validation evidence.  Package and hash the exact generated ledger.
    add(root/EXECUTION_PROFILE_LEDGER,EXECUTION_PROFILE_LEDGER)
    add(root/QGIS_PRODUCT_OUTPUT_OVERLAY,QGIS_PRODUCT_OUTPUT_OVERLAY)
    if not {'braids_gpu/__init__.py','braids_gpu/__main__.py','braids_gpu/runner.py','braids_gpu/backend.py'}<=files.keys():
        raise ValueError('Incomplete public worker package')
    for source in sorted(cuda_files):add(source,source.relative_to(root).as_posix())
    for variant,relative in scope['source_archives_by_variant'].items():
        if variant in (AMR_VARIANT,AMR_GENERAL_VARIANT):
            continue  # Collected from the qualified immutable source archive below.
        for source in sorted(declared_source_archive(root,relative,variant)):
            add(source,source.relative_to(root).as_posix())
    add(gpu/'requirements-gpu.txt','validation/gpu_fwave/requirements-gpu.txt')
    add(gpu/'Clawpack-LICENSE.txt','licenses/Clawpack-LICENSE.txt',CLAWPACK_SHA)
    relative=runtime.relative_to(root).as_posix()
    initializer_receipt=portable_initializer_receipt(runtime/'build.json',identity)
    executable=runtime/'xbraids_coupled_export.exe'
    add(executable,relative+'/'+executable.name,identity['executable_sha256'])
    for name,digest in identity['source_sha256'].items():
        safe_name(name)
        if Path(name).suffix.lower() not in ('.f','.f90','.f95'):raise ValueError('Unexpected initializer source: '+name)
        add(runtime/name,relative+'/'+name,digest)
    add(runtime/'coupled_reference.py',relative+'/coupled_reference.py',identity['driver_sha256'])
    notices=json.loads((Path(license_dir)/'manifest.json').read_text())
    add(Path(license_dir)/'manifest.json','licenses/native-runtime-provenance.json')
    for row in notices['files']:
        name=safe_name(row['path']);add(Path(license_dir)/name,'licenses/'+name,row['sha256'])
    dependencies,imports=runtime_dependencies(executable,dll_dir,objdump,inspect)
    for name,path in dependencies.items():
        if name not in notices['libraries']:raise ValueError('DLL has no matching license provenance: '+name)
        add(path,relative+'/'+name,notices['libraries'][name])
    graded_exporter=None
    if GRADED_VARIANT in scope['supported_gpu_variants']:
        # The current critical-outlet component still authenticates this
        # frozen BRAIDS/GeoClaw source at construction.  It is source, not a
        # validation case/output or a Delft3D source checkout.
        add(root/GRADED_CRITICAL_REFERENCE,GRADED_CRITICAL_REFERENCE,
            GRADED_CRITICAL_REFERENCE_SHA)
        helper=root/'braids_gpu/graded_native_exporter.py'
        if 'braids_gpu/graded_native_exporter.py' not in files:
            add(helper,'braids_gpu/graded_native_exporter.py')
        payloads,graded_exporter=graded_exporter_payloads(root,dll_dir=dll_dir,
            objdump=objdump,inspect=inspect)
        for name,source in payloads.items():
            if name.lower().endswith('.dll'):
                dll_name=Path(name).name.lower()
                if dll_name not in notices['libraries']:
                    raise ValueError('Graded exporter DLL has no matching license provenance: '+dll_name)
                add(source,name,notices['libraries'][dll_name])
            else:add(source,name)
    transport_exporter=None
    gaia_transport_exporter=None
    delft_transport_exporter_v13=None
    delft_transport_exporter=None
    delft_transport_exporter_v15=None
    delft_transport_exporter_v16=None
    delft_transport_exporter_v17=None
    delft_transport_exporter_v18=None
    bank_exporter=None
    for variant,source,target,executable_name,expected_sha256,source_hash_key,unchanged_key,steps_key in (
            (TRANSPORT_VARIANT,TRANSPORT_EXPORT_SOURCE,TRANSPORT_EXPORT_TARGET,
             'xbraids_eh_export.exe',TRANSPORT_EXPORT_SHA,'source_sha256',
             'numerical_source_unchanged','native_solver_steps_performed'),
            (TRANSPORT_VARIANT,GAIA_TRANSPORT_EXPORT_SOURCE,GAIA_TRANSPORT_EXPORT_TARGET,
             'xbraids_gaia_export.exe',GAIA_TRANSPORT_EXPORT_SHA,'source_sha256',
             'numerical_source_unchanged','native_solver_steps_performed'),
            (TRANSPORT_VARIANT,DELFT_TRANSPORT_EXPORT_V13_SOURCE,
             DELFT_TRANSPORT_EXPORT_V13_TARGET,
             'xbraids_delft_export.exe',DELFT_TRANSPORT_EXPORT_V13_SHA,
             'snapshot_source_sha256',
             'numerical_source_unchanged_by_instrumentation',
             'native_solver_steps_performed'),
            (TRANSPORT_VARIANT,DELFT_TRANSPORT_EXPORT_SOURCE,DELFT_TRANSPORT_EXPORT_TARGET,
             'xbraids_delft_export.exe',DELFT_TRANSPORT_EXPORT_SHA,'snapshot_source_sha256',
             'numerical_source_unchanged_by_instrumentation','native_solver_steps_performed'),
            (TRANSPORT_VARIANT,DELFT_TRANSPORT_EXPORT_V15_SOURCE,
             DELFT_TRANSPORT_EXPORT_V15_TARGET,
             'xbraids_delft_export.exe',DELFT_TRANSPORT_EXPORT_V15_SHA,
             'snapshot_source_sha256',
             'numerical_source_unchanged_by_instrumentation',
             'native_solver_steps_performed'),
            (TRANSPORT_VARIANT,DELFT_TRANSPORT_EXPORT_V16_SOURCE,
             DELFT_TRANSPORT_EXPORT_V16_TARGET,
             'xbraids_delft_export.exe',DELFT_TRANSPORT_EXPORT_V16_SHA,
             'snapshot_source_sha256',
             'numerical_source_unchanged_by_instrumentation',
             'native_solver_steps_performed'),
            (TRANSPORT_VARIANT,DELFT_TRANSPORT_EXPORT_V17_SOURCE,
             DELFT_TRANSPORT_EXPORT_V17_TARGET,
             'xbraids_delft_export.exe',DELFT_TRANSPORT_EXPORT_V17_SHA,
             'snapshot_source_sha256',
             'numerical_source_unchanged_by_instrumentation',
             'native_solver_steps_performed'),
            (TRANSPORT_VARIANT,DELFT_TRANSPORT_EXPORT_V18_SOURCE,
             DELFT_TRANSPORT_EXPORT_V18_TARGET,
             'xbraids_delft_export.exe',DELFT_TRANSPORT_EXPORT_V18_SHA,
             'snapshot_source_sha256',
             'numerical_source_unchanged_by_instrumentation',
             'native_solver_steps_performed'),
            (BANK_VARIANT,BANK_EXPORT_SOURCE,BANK_EXPORT_TARGET,
             'xbraids_live_bank_export.exe',BANK_EXPORT_SHA,'copied_source_sha256',
             'numerical_routines_unchanged','solver_steps_performed')):
        if (source in (DELFT_TRANSPORT_EXPORT_V15_SOURCE,
                       DELFT_TRANSPORT_EXPORT_V16_SOURCE,
                       DELFT_TRANSPORT_EXPORT_V17_SOURCE,
                       DELFT_TRANSPORT_EXPORT_V18_SOURCE) and
                not (root/source/'build.json').is_file()):
            continue
        if variant not in scope['supported_gpu_variants']:
            continue
        payloads,identity=integrated_exporter_payloads(root,source=source,target=target,
            executable_name=executable_name,expected_sha256=expected_sha256,
            source_hash_key=source_hash_key,unchanged_key=unchanged_key,
            steps_key=steps_key,dll_dir=dll_dir,objdump=objdump,inspect=inspect)
        for name,payload in payloads.items():
            if name.lower().endswith('.dll'):
                dll_name=Path(name).name.lower()
                if dll_name not in notices['libraries']:
                    raise ValueError('Integrated initializer DLL has no license provenance: '+dll_name)
                add(payload,name,notices['libraries'][dll_name])
            else:
                add(payload,name)
        if source==TRANSPORT_EXPORT_SOURCE:
            transport_exporter=identity
        elif source==GAIA_TRANSPORT_EXPORT_SOURCE:
            gaia_transport_exporter=identity
        elif source==DELFT_TRANSPORT_EXPORT_V13_SOURCE:
            delft_transport_exporter_v13=identity
        elif source==DELFT_TRANSPORT_EXPORT_SOURCE:
            delft_transport_exporter=identity
        elif source==DELFT_TRANSPORT_EXPORT_V15_SOURCE:
            delft_transport_exporter_v15=identity
        elif source==DELFT_TRANSPORT_EXPORT_V16_SOURCE:
            delft_transport_exporter_v16=identity
        elif source==DELFT_TRANSPORT_EXPORT_V17_SOURCE:
            delft_transport_exporter_v17=identity
        elif source==DELFT_TRANSPORT_EXPORT_V18_SOURCE:
            delft_transport_exporter_v18=identity
        else:
            bank_exporter=identity
    amr_exporter=None
    amr_live_exporter=None
    amr_bedform_bank_exporter=None
    amr_bedform_physical_toe_exporter=None
    bank_payloads={}
    if AMR_VARIANT in scope['supported_gpu_variants']:
        payloads,amr_exporter=amr_exporter_payloads(root,dll_dir=dll_dir,objdump=objdump,inspect=inspect)
        bank_payloads=payloads
        for name,source in payloads.items():
            if name.lower().endswith('.dll'):
                dll_name=Path(name).name.lower()
                if dll_name not in notices['libraries']:
                    raise ValueError('AMR initializer DLL has no matching license provenance: '+dll_name)
                add(source,name,notices['libraries'][dll_name])
            else:
                add(source,name)
        for name in ('braids_core/boundaries/geometry_io.py',):
            add(root/name,name)
        if scope.get('evolving_native_exporters_by_variant'):
            from braids_gpu.amr_portable_runtime import portable_live_build_receipt
            live=root/AMR_LIVE_RUNTIME_SOURCE
            live_build=portable_live_build_receipt(live)
            live_dependencies,live_imports=runtime_dependencies(live/live_build['executable'],dll_dir,objdump,inspect)
            for name in (*live_build['source_sha256'],live_build['executable']):
                add(live/name,AMR_LIVE_RUNTIME_TARGET+'/'+name)
            for name,path in live_dependencies.items():
                if name not in notices['libraries']:
                    raise ValueError('Live AMR initializer DLL has no license provenance: '+name)
                add(path,AMR_LIVE_RUNTIME_TARGET+'/'+name,notices['libraries'][name])
            amr_live_exporter=dict(build=live_build,executable_sha256=live_build['executable_sha256'],
                runtime_imports=live_imports,
                dll_sha256={name:sha(path) for name,path in sorted(live_dependencies.items())})
        if scope.get('recursive_bank_bedform_native_exporters_by_variant'):
            from braids_gpu.amr_portable_runtime import portable_live_build_receipt
            exact=root/AMR_BEDFORM_BANK_RUNTIME_SOURCE
            exact_build=portable_live_build_receipt(exact)
            exact_dependencies,exact_imports=runtime_dependencies(
                exact/exact_build['executable'],dll_dir,objdump,inspect)
            for name in (*exact_build['source_sha256'],exact_build['executable']):
                add(exact/name,AMR_BEDFORM_BANK_RUNTIME_TARGET+'/'+name)
            for name,path in exact_dependencies.items():
                if name not in notices['libraries']:
                    raise ValueError('Bedform-bank AMR initializer DLL has no license provenance: '+name)
                add(path,AMR_BEDFORM_BANK_RUNTIME_TARGET+'/'+name,
                    notices['libraries'][name])
            amr_bedform_bank_exporter=dict(build=exact_build,
                executable_sha256=exact_build['executable_sha256'],
                runtime_imports=exact_imports,
                dll_sha256={name:sha(path) for name,path in
                             sorted(exact_dependencies.items())})
        if scope.get('recursive_physical_toe_bedform_native_exporters_by_variant'):
            from braids_gpu.amr_portable_runtime import portable_live_build_receipt
            exact=root/AMR_BEDFORM_PHYSICAL_TOE_RUNTIME_SOURCE
            exact_build=portable_live_build_receipt(exact)
            exact_dependencies,exact_imports=runtime_dependencies(
                exact/exact_build['executable'],dll_dir,objdump,inspect)
            for name in (*exact_build['source_sha256'],exact_build['executable']):
                add(exact/name,AMR_BEDFORM_PHYSICAL_TOE_RUNTIME_TARGET+'/'+name)
            for name,path in exact_dependencies.items():
                if name not in notices['libraries']:
                    raise ValueError(
                        'Physical-toe bedform AMR initializer DLL has no license provenance: '+name)
                add(path,AMR_BEDFORM_PHYSICAL_TOE_RUNTIME_TARGET+'/'+name,
                    notices['libraries'][name])
            amr_bedform_physical_toe_exporter=dict(build=exact_build,
                executable_sha256=exact_build['executable_sha256'],
                runtime_imports=exact_imports,
                dll_sha256={name:sha(path) for name,path in
                            sorted(exact_dependencies.items())})
    amr_general_exporter=None
    amr_general_source=None
    if AMR_GENERAL_VARIANT in scope['supported_gpu_variants']:
        payloads,amr_general_exporter=amr_general_exporter_payloads(root,
            dll_dir=dll_dir,objdump=objdump,inspect=inspect)
        for name,source in payloads.items():
            if name.lower().endswith('.dll'):
                dll_name=Path(name).name.lower()
                if dll_name not in notices['libraries']:
                    raise ValueError('General AMR initializer DLL lacks license provenance: '+dll_name)
                add(source,name,notices['libraries'][dll_name])
            else:
                add(source,name)
        source_payloads,amr_general_source=amr_general_source_payloads(root,files,
            bank_payloads,amr_exporter['kernel_manifest']['interface_source_pair'])
        for name,source in source_payloads.items():
            add(source,name)
    return files,expected,dict(initializer_tag=tag,initializer=initializer_receipt,runtime_imports=imports,
        source_worker_scope=scope,source_closure=dict(python_files=len(python_files),cuda_files=len(cuda_files)),
        graded_native_exporter=graded_exporter,
        transport_native_exporter=transport_exporter,
        delft_transport_native_exporter_v13=delft_transport_exporter_v13,
        delft_transport_native_exporter=delft_transport_exporter,
        delft_transport_native_exporter_v15=delft_transport_exporter_v15,
        delft_transport_native_exporter_v16=delft_transport_exporter_v16,
        delft_transport_native_exporter_v17=delft_transport_exporter_v17,
        delft_transport_native_exporter_v18=delft_transport_exporter_v18,
        gaia_transport_native_exporter=gaia_transport_exporter,
        bank_native_exporter=bank_exporter,
        on_demand_amr=amr_exporter,
        on_demand_amr_live=amr_live_exporter,
        on_demand_amr_bedform_bank=amr_bedform_bank_exporter,
        on_demand_amr_bedform_physical_toe=amr_bedform_physical_toe_exporter,
        on_demand_amr_general=amr_general_exporter,
        on_demand_amr_general_source=amr_general_source,
        dll_sha256={name:sha(path) for name,path in sorted(dependencies.items())},
        runtime_license_manifest='licenses/native-runtime-provenance.json')


def build(destination,*,root=ROOT,dll_dir=Path('C:/Strawberry/c/bin'),objdump=None,
          license_dir=None,inspect=dll_imports,qgis_product_only=False):
    root=Path(root).resolve();destination=Path(destination).resolve()
    archive=destination.parent/(destination.name+'.zip');receipt_path=destination.parent/(destination.name+'.package.json')
    if any(p.exists() for p in (destination,archive,receipt_path)):raise FileExistsError('Package destination/archive/receipt already exists')
    files,expected,provenance=collect(root,dll_dir=dll_dir,objdump=objdump or Path(dll_dir)/'objdump.exe',
        license_dir=license_dir or root/'dist/windows-x86_64/braids_qgis/licenses',inspect=inspect,
        qgis_product_only=qgis_product_only)
    forbidden=forbidden_payload_members(files)
    if forbidden:
        raise ValueError('Forbidden data/reference/environment worker payload: '+
                         forbidden[0])
    destination.mkdir(parents=True,exist_ok=False)
    origins={}
    for name,source in files.items():
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if sha(target)!=expected[name]:raise ValueError('Source changed while copying: '+str(source))
        origins[name]='package-local:'+name
    generated={'QUICKSTART.txt':QUICKSTART,'validation/__init__.py':'"""BRAIDS experimental worker modules."""\n',
        'validation/gpu_fwave/__init__.py':'"""Qualified experimental resident GPU operators."""\n',
        f"validation/gpu_fwave/runs/{provenance['initializer_tag']}/runtime_export/build.json":
            json.dumps(provenance['initializer'],indent=2,allow_nan=False)+'\n',
        'requirements-gpu.txt':(root/'validation/gpu_fwave/requirements-gpu.txt').read_text()+'\n# Tested source-worker NumPy version.\nnumpy==2.5.3\n'}
    for key,target in (('transport_native_exporter',TRANSPORT_EXPORT_TARGET),
                       ('delft_transport_native_exporter_v13',
                        DELFT_TRANSPORT_EXPORT_V13_TARGET),
                       ('delft_transport_native_exporter',DELFT_TRANSPORT_EXPORT_TARGET),
                       ('delft_transport_native_exporter_v15',
                        DELFT_TRANSPORT_EXPORT_V15_TARGET),
                       ('delft_transport_native_exporter_v16',
                        DELFT_TRANSPORT_EXPORT_V16_TARGET),
                       ('delft_transport_native_exporter_v17',
                        DELFT_TRANSPORT_EXPORT_V17_TARGET),
                       ('delft_transport_native_exporter_v18',
                        DELFT_TRANSPORT_EXPORT_V18_TARGET),
                       ('gaia_transport_native_exporter',GAIA_TRANSPORT_EXPORT_TARGET),
                       ('bank_native_exporter',BANK_EXPORT_TARGET)):
        if provenance[key] is not None:
            generated[target+'/build.json']=json.dumps(provenance[key],indent=2,allow_nan=False)+'\n'
    if provenance['on_demand_amr'] is not None:
        amr=provenance['on_demand_amr']
        generated.update({
            AMR_RUNTIME_TARGET+'/build.json':json.dumps(amr['build'],indent=2,allow_nan=False)+'\n',
            VARIANT_SOURCE_ARCHIVES[AMR_VARIANT]+'/runtime_manifest.json':
                json.dumps(amr['kernel_manifest'],indent=2,allow_nan=False)+'\n',
            'braids_core/boundaries/__init__.py':
                '"""GPU worker namespace for standalone NumPy geometry I/O only."""\n'})
    if provenance['on_demand_amr_live'] is not None:
        generated[AMR_LIVE_RUNTIME_TARGET+'/build.json']=json.dumps(
            provenance['on_demand_amr_live']['build'],indent=2,allow_nan=False)+'\n'
    if provenance['on_demand_amr_bedform_bank'] is not None:
        generated[AMR_BEDFORM_BANK_RUNTIME_TARGET+'/build.json']=json.dumps(
            provenance['on_demand_amr_bedform_bank']['build'],indent=2,
            allow_nan=False)+'\n'
    if provenance['on_demand_amr_bedform_physical_toe'] is not None:
        generated[AMR_BEDFORM_PHYSICAL_TOE_RUNTIME_TARGET+'/build.json']=json.dumps(
            provenance['on_demand_amr_bedform_physical_toe']['build'],indent=2,
            allow_nan=False)+'\n'
    if provenance['on_demand_amr_general'] is not None:
        generated[AMR_GENERAL_RUNTIME_TARGET+'/build.json']=json.dumps(
            provenance['on_demand_amr_general']['build'],indent=2,allow_nan=False)+'\n'
        generated[VARIANT_SOURCE_ARCHIVES[AMR_GENERAL_VARIANT]+'/runtime_manifest.json']=json.dumps(
            provenance['on_demand_amr_general_source'],indent=2,allow_nan=False)+'\n'
    for name,content in generated.items():
        target=destination/name
        if target.exists():raise ValueError('Generated member conflicts with source: '+name)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text(content,encoding='utf-8',newline='\n');origins[name]='package-local:'+name
    forbidden=forbidden_payload_members(
        path.relative_to(destination).as_posix()
        for path in destination.rglob('*') if path.is_file())
    if forbidden:
        raise ValueError('Forbidden data/reference/environment worker payload: '+
                         forbidden[0])
    # Fail if another development process changed any source during collection.
    for name,source in files.items():
        if sha(source)!=expected[name]:raise ValueError('Source changed during package build: '+str(source))
    hashes={p.relative_to(destination).as_posix():sha(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    for name in hashes:
        if name.endswith('.json'):
            _check_portable_json(json.loads((destination/name).read_text(encoding='utf-8')),name)
    manifest=dict(schema=1,kind='experimental Windows source worker',created_utc=datetime.now(timezone.utc).isoformat(),
        builder_sha256=sha(__file__),build_python=sys.version,platform='Windows x86_64',
        python_environment_bundled=False,GPU_wheels_bundled=False,QGIS_release=True,
        qgis_product_only=bool(qgis_product_only),automatic_backend_selection=False,
        data_or_Delft3D_bundled=False,native_initializer_only=True,tested_python='CPython 3.13.5 x64',
        provenance=provenance,source_worker_scope=provenance['source_worker_scope'],origins=origins,files=hashes,
        manifest_note='All payload files except this manifest are hashed; adjacent packaging receipt hashes manifest and ZIP; paths in receipts are package-local')
    _check_portable_json(manifest,'PACKAGE-MANIFEST.json')
    manifest_path=destination/'PACKAGE-MANIFEST.json';manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as bundle:
        for path in sorted(destination.rglob('*')):
            if path.is_file():bundle.write(path,(Path(destination.name)/path.relative_to(destination)).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        expected_members={destination.name+'/'+name for name in (*hashes,'PACKAGE-MANIFEST.json')}
        if set(bundle.namelist())!=expected_members or bundle.testzip() is not None:raise RuntimeError('Archive payload verification failed')
        for name,digest in hashes.items():
            if hashlib.sha256(bundle.read(destination.name+'/'+name)).hexdigest()!=digest:raise RuntimeError('Archive payload hash mismatch: '+name)
    receipt=dict(package=destination.name,archive=archive.name,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        manifest_sha256=sha(manifest_path),files=len(hashes),initializer_tag=provenance['initializer_tag'],
        no_GPU_or_solver_execution_during_packaging=True,no_environment_installation=True)
    _check_portable_json(receipt,receipt_path.name)
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--destination',required=True,type=Path)
    parser.add_argument('--dll-dir',type=Path,default=Path('C:/Strawberry/c/bin'))
    parser.add_argument('--qgis-product-only',action='store_true',
        help='package only the admitted QGIS v109 fixed-grid GPU route')
    args=parser.parse_args();print(json.dumps(build(args.destination,dll_dir=args.dll_dir,
        qgis_product_only=args.qgis_product_only),indent=2))
