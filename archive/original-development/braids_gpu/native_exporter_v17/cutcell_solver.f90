! Fixed cut-cell WAVE kernel integration. Sparse control volumes are independent
! of the legacy Cartesian AMR driver. All time stepping and reconstruction are
! native. Physical momentum flux has separate left/right bed-source corrections;
! mass and sediment always have ONE shared accepted face flux.
module cutcell_solver
 use secondary_curvature_regularization, only: regularize_curvature
 use geoclaw_module, only: grav,dry_tolerance,rho,speed_limit
 use braids_exner_mol_module, only: reconstruct
 use braids_van_rijn_1993_module, only: braids_vr1993_grain_parameters,braids_vr1993_coefficients
 use braids_bedload_mpm_module, only: braids_mpm_bedload,braids_mpm_bedload_roughness,braids_mpm_bedload_roughness_batch, &
  braids_delft_mpm_bedload_roughness,braids_delft_mpm_bedload_roughness_batch, &
  braids_mixture_percentile_default,braids_mixture_d90_default
 use braids_bedform_roughness_module, only: braids_fredsoe_vanrijn84_roughness
 use braids_bedload_current_module, only: braids_current_bedload,braids_surface_statistics
 use braids_hydraulic_roughness_module, only: braids_roughness_state,braids_roughness_shear, &
  braids_roughness_friction_rate,braids_roughness_chezy,braids_roughness_manning, &
  braids_roughness_white_colebrook
 use braids_sediment_shear_module, only: braids_soulsby_clarke_skin_shear
 use braids_soulsby_van_rijn_module, only: braids_soulsby_van_rijn_current
 use braids_delft_capacity_module, only: braids_general_capacity, &
  braids_modified_van_rijn_1984,braids_wang_fredsoe_capacity
 use braids_scalar_capacity_runtime_module, only: braids_scalar_capacity_dispatch, &
  braids_capacity_mpm,braids_capacity_engelund_hansen,braids_capacity_general
 use braids_delft_channel_composer_module, only: braids_compose_delft_channels, &
  BRAIDS_TRA_BEDLOAD,BRAIDS_TRA_COMBINE,BRAIDS_COMPOSE_OK
 use braids_secondary_flow_module, only: braids_streamline_curvature,braids_delft3d_spiral_transport, &
  braids_spiral_equilibrium,braids_spiral_adaptation_time,braids_spiral_relax_exact, &
  braids_spiral_transport_from_intensity,braids_spiral_stress,braids_gaia_secondary_stress, &
  braids_gaia_open_boundary_capacity
 use braids_flow2dh_secondary_operator_module, only: braids_flow2dh_secondary_source_sink, &
  braids_flow2dh_open_boundary_equilibrium,braids_flow2dh_scalar_bed_rotation
 use braids_cohesive_exchange_module, only: braids_partheniades_krone,braids_sand_mud_erosion
 use braids_hiding_exposure_module, only: braids_hiding_exposure_factors
 use braids_bedload_vr1993_module, only: braids_vr1993_bedload_cell
 use braids_bed_slope_module, only: braids_delft3d_bagnold_ikeda,braids_delft3d_koch_flokstra, &
  braids_delft3d_total_shields, &
  braids_gaia_talmon_slope
 use braids_morphodynamic_flux_module, only: coupled_exner_flux,mpm_mass_jacobian,gaia_mass_jacobian, &
  gaia_fixed_inlet_face_terms,gaia_fixed_inlet_trace,gaia_fixed_inlet_pressure
 use braids_bank_toe_geometry_module, only: braids_map_bank_toes,BANK_TOE_MAPPED, &
  BANK_TOE_SUCCESS
 use braids_bank_toe_supply_module, only: braids_exner_demand_from_inventory, &
  braids_exner_demand_from_face_integral,braids_allocate_physical_toe_requests, &
  braids_route_extracted_toe_mass
 use braids_bank_toe_flux_module, only: braids_bank_toe_add_face_rates, &
   braids_bank_toe_add_face_integrals,BANK_TOE_FLUX_SUCCESS
 use braids_accepted_flux_reconstruction_module, only: braids_reconstruct_accepted_bedload, &
   BRAIDS_ACCEPTED_FLUX_SUCCESS
 use braids_distributed_bank_module, only: braids_distributed_bank_face_transfers, &
   BRAIDS_DISTRIBUTED_BANK_SUCCESS
use braids_bank_front_module, only: BANK_FRONT_INACTIVE,BANK_FRONT_ACTIVE,BANK_FRONT_PROMOTED, &
   BANK_FRONT_SUCCESS,braids_bank_front_activate,braids_bank_front_effective_bed, &
   braids_bank_front_capacity,braids_bank_front_apply_actual_mass, &
   braids_bank_front_exclude_incoming,braids_bank_front_validate
 use braids_bank_reconstruction_module, only: bank_cell_depth_intercept,bank_face_mobility
 use braids_viscous_inlet_module, only: braids_viscous_roughness_profile, &
  BRAIDS_VISCOUS_INLET_SUCCESS
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 use, intrinsic :: iso_fortran_env, only: int32,int64,iostat_eor
 use omp_lib, only: omp_get_wtime,omp_get_max_threads,omp_get_num_procs,omp_get_num_threads
 implicit none
 ! Guarded CPU time integration; input CFL is never overridden here.
 integer :: cpu_time_integrator=33,cpu_outlet_extension=0
 character(len=64) :: cpu_time_integrator_requested='auto'
 logical :: cpu_multistep_eligible=.false.,cpu_ssp43_eligible=.false.
 integer :: explicit_history_method=0
 integer :: explicit_history_count=0,explicit_history_position=1
 integer :: explicit_trial_reset_kind=0
 integer(kind=8) :: explicit_history_steps=0,explicit_history_startup_steps=0,explicit_history_failures=0
 integer(kind=8) :: explicit_history_invalid_span_attempts=0,explicit_history_friction_rejections=0
 integer(kind=8) :: explicit_history_knot_resets=0,explicit_history_growth_resets=0
 real(8),allocatable :: explicit_history_state(:,:,:),explicit_history_ledger(:,:,:),explicit_history_time(:)
 real(8),allocatable :: explicit_friction_rate(:)
 real(8) :: explicit_trial_span=0,explicit_trial_ratio=0,explicit_trial_friction_peak=0,explicit_trial_physical_cfl=0
 real(8) :: explicit_history_minimum_span=huge(1.d0),explicit_history_maximum_span=0,explicit_history_maximum_ratio=0
 real(8) :: explicit_history_maximum_friction=0,explicit_history_maximum_physical_cfl=0
 real(8) :: explicit_history_physical_time=0,explicit_history_startup_time=0,explicit_history_source_seconds=0
 real(8) :: explicit_last_accepted_dt=0
 integer :: nc,nf,nb,nx,ny,scheme,bedlaw,exchange_mode,bed_slope=0,deck_version
 integer :: nfrac=1,nstate=5,nprimitive=5,nledger=3,nbed_layers=1,nsubstrate_layers=0
 integer :: nphysical_state=5,nphysical_primitive=5,nphysical_ledger=3
 integer, allocatable :: owner(:),neighbour(:),bid(:),offset(:),faces(:),signs(:),ii(:),jj(:),grid(:,:),mode(:)
 integer, allocatable :: boundary_offset(:),boundary_faces(:),reversal_policy(:),nseries(:),nhydro_series(:),bed_boundary_mode(:)
 integer, allocatable :: nfraction_series(:,:)
 integer, allocatable :: open_boundary_distance(:),uncoupled_boundary_distance(:)
 real(8), allocatable :: initial_signature(:)
 real(8), allocatable :: rectangle_width(:,:)
 integer, allocatable :: weno_cells(:,:,:)
 real(8), allocatable :: face_displacement(:,:,:)
 real(8), allocatable :: hydraulic_ls_weight(:),hydraulic_ls_matrix(:,:)
 ! Sediment-only least-squares geometry, rebuilt with every mesh epoch.
 real(8), allocatable :: bed_ls_delta(:,:),bed_ls_weight(:),bed_ls_matrix(:,:)
 logical, allocatable :: wall_triangle(:),wall_ready(:,:)
 logical, allocatable :: hydro_constant(:)
 logical, allocatable :: viscosity_equilibrated_inlet(:)
 logical, allocatable :: active_cell(:)
 logical, allocatable :: bank_erodible(:)
 integer, allocatable :: wall_stencil_ids(:,:,:),wall_stencil_kind(:,:,:),wall_slot(:,:)
 real(8), allocatable :: wall_stencil_normal(:,:,:,:),wall_box_center(:,:),wall_normal(:,:),wall_mean_distance(:),wall_values(:,:,:)
 real(8), allocatable :: area(:),xy(:,:),normal(:,:),length(:),mid(:,:),rock(:),q(:,:),forcing(:,:),ledger(:,:),manning_cell(:)
 real(8), allocatable :: bedq_cell(:,:,:),bedq_equilibrium(:,:,:),bedq_relax_work(:,:,:),bedq_grad(:,:,:,:)
 real(8), allocatable :: suspended_capacity_cell(:,:,:),suspended_capacity_equilibrium(:,:,:)
 real(8), allocatable :: channel_raw_bed_cell(:,:,:),channel_raw_suspended_cell(:,:,:), &
  channel_composed_bed_cell(:,:,:),channel_retained_suspended_cell(:,:,:), &
  channel_signed_suspended_rate_cell(:,:),channel_equilibrium_concentration_cell(:,:)
 real(8), allocatable :: suspended_exchange_cell_trial(:,:),suspended_exchange_cumulative(:,:), &
  suspended_exchange_step_trial(:,:)
 real(8), allocatable :: suspended_boundary_return_time(:)
 real(8), allocatable :: suspended_boundary_last_outflow(:,:), &
  suspended_boundary_return_clock(:,:)
 real(8), allocatable :: capacity_absolute_shields(:,:),capacity_critical_shields(:,:), &
  capacity_transport_stage(:,:)
 integer,allocatable :: split_boundary_role(:)
 real(8),allocatable :: split_boundary_bed(:,:,:),split_boundary_suspended(:,:,:), &
  split_boundary_absolute_shields(:,:),split_boundary_critical_shields(:,:), &
  split_boundary_transport_stage(:,:),split_boundary_stress_ratio(:,:)
 real(8),allocatable :: channel_raw_bed_boundary(:,:,:),channel_raw_suspended_boundary(:,:,:), &
  channel_composed_bed_boundary(:,:,:),channel_retained_suspended_boundary(:,:,:), &
  channel_signed_suspended_rate_boundary(:,:),channel_equilibrium_concentration_boundary(:,:)
 real(8), allocatable :: gaia_bed_down(:,:),gaia_eta_gradient(:,:)
 real(8), allocatable :: bedq_gaia_base(:,:,:),bedq_gaia_base_grad(:,:,:,:)
 ! Existing laws store tau/tau_cr; Engelund-Hansen law 8 stores absolute
 ! Shields stress for the Koch-Flokstra (ISlope=3) direction law.
 real(8), allocatable :: bedtau_cell(:,:)
 real(8), allocatable :: secondary_curvature(:)
 real(8), allocatable :: secondary_classic_intensity(:)
 real(8), allocatable :: secondary_face_left(:),secondary_face_right(:),secondary_boundary_intensity(:)
 real(8), allocatable :: secondary_low_flux(:),secondary_anti_flux(:),secondary_stress_face(:,:)
 real(8), allocatable :: secondary_rplus(:),secondary_rminus(:),secondary_low_state(:),secondary_source_work(:)
real(8), allocatable :: bank_transfer(:,:),bank_scale(:,:),bank_bed(:),bank_request(:),bank_weight(:),bank_down(:,:)
 real(8), allocatable :: bank_previous(:,:),bank_depth(:)
 real(8), allocatable :: bank_depth_intercept(:),bank_rate(:),bank_conductance(:)
 real(8),allocatable :: distributed_face_mass_trial(:,:,:),distributed_cell_bedload(:,:,:)
 logical,allocatable :: distributed_rank_deficient(:)
 real(8), allocatable :: boundary_bed_offset(:)
 real(8), allocatable :: boundary_bed_left_offset(:),boundary_bed_right_offset(:)
 real(8), allocatable :: fixed_boundary_bed(:)
 integer, allocatable :: sampled_boundary_bed(:)
 integer, allocatable :: fixed_boundary_bed_mode(:)
 integer, allocatable :: boundary_seed_position(:)
 logical, allocatable :: sampled_boundary_open(:)
 real(8), allocatable :: series_time(:,:),series_concentration(:,:),series_bed_input(:,:)
 real(8), allocatable :: hydro_series_time(:,:),hydro_series_q(:,:),hydro_series_stage(:,:)
 real(8), allocatable :: fraction_concentration(:,:),fraction_bed_input(:,:)
 real(8), allocatable :: fraction_series_time(:,:,:),fraction_series_concentration(:,:,:),fraction_series_bed_input(:,:,:)
 real(8), allocatable :: morfac_time(:),morfac_value(:)
 real(8), allocatable :: bulk_fraction(:),ws_fraction(:),d50_fraction(:),d90_fraction(:),rhos_fraction(:),z0_fraction(:)
 real(8), allocatable :: sedthr_fraction(:),bedload_min_depth_fraction(:),reference_height_fraction(:),dstar_fraction(:),taucr_fraction(:)
 real(8), allocatable :: critical_erosion_fraction(:),critical_deposition_fraction(:),erosion_parameter_fraction(:),erosion_exponent_fraction(:)
 integer, allocatable :: sediment_roughness_fraction(:),sediment_type_fraction(:)
 integer, allocatable :: section_control(:),section_seed_node(:)
 real(8), allocatable :: section_energy_slope(:)
 real(8), allocatable :: section_cached_stage(:),section_cached_discharge(:),section_cached_bed(:)
 logical, allocatable :: section_cache_valid(:)
real(8) :: bulk,manning,ws,d50,d90,rhos,z0,sedthr,bedload_min_depth,dstar,taucr,dx,dy,time,dtmax,endtime,output_dt,step_hint,cfl_target=.45d0
 real(8) :: trial_cfl_peak=0.d0
 real(8) :: horizontal_viscosity=0.d0,profile_diffusion_s=0.d0
 real(8) :: manning_friction_chezy_minimum_depth=0.d0
 integer :: hydraulic_roughness_mode=braids_roughness_manning,sediment_shear_mode=0
 logical :: white_colebrook_chzk2d=.false.
 real(8) :: roughness_minimum_work_depth=0.d0
 real(8) :: skin_kinematic_viscosity=1.d-6,skin_sand_roughness=.003d0
 real(8) :: skin_silt_roughness=.0003d0,skin_critical_mud_fraction_low=.2d0
 real(8) :: skin_critical_mud_fraction_high=.8d0,skin_fluff_fraction=0.d0
 integer :: horizontal_viscosity_mode=0,horizontal_viscosity_geometry=0,diffusion_substeps=0
 real(8) :: diffusion_minimum_correction_factor=1.d0
 integer :: diffusion_recovered_faces=0,diffusion_recovery_fallbacks=0
 integer :: diffusion_affine_recovered_faces=0
 integer :: steps=0,retries=0,failure_reason=0
 integer :: failed_speed_boundary=0
 real(8) :: failed_trace_speed=0
 integer :: failed_trace_boundary=0,failed_trace_face=0,failed_trace_condition=0
 real(8) :: failed_trace_values(6)=0
 real(8) :: mpm_cf=.01d0,mpm_critical=0.d0
 real(8) :: reference_height_factor=1.d0
 real(8) :: active_layer_thickness=0.d0
 integer :: active_layer_mode=0
 real(8) :: active_layer_depth_fraction=0.d0,active_layer_minimum_thickness=0.d0
 real(8) :: hiding_parker_alpha=.6d0,hiding_wu_exponent=-.6d0
 real(8) :: substrate_layer_thickness=0.d0
 real(8) :: critical_mud_fraction=.5d0,transport_calibration=1.d0
 real(8) :: engelund_hansen_chezy=0.d0
 real(8) :: gaia_skin_roughness_ratio=0.d0,gaia_slope_beta=0.d0
 real(8) :: gaia_deviation_beta2=0.d0,gaia_secondary_alpha=0.d0
 logical :: gaia_interior_slope_talmon=.true.
 logical :: gaia_interior_secondary_stress=.true.
 logical :: gaia_open_boundary_secondary_stress=.true.
 real(8) :: bank_erosion_fraction=0.d0,bank_wet_slope=10.d0,bank_dry_slope=1.d0
 real(8) :: bank_avalanche_time=86400.d0,bank_wet_depth_switch=.1d0,bank_relaxation_length=1.d0
 logical :: bank_avalanche=.false.
 real(8) :: general_avalanche_wet_slope=10.d0,general_avalanche_dry_slope=1.d0
 real(8) :: general_avalanche_time=86400.d0,general_avalanche_wet_depth_switch=.1d0
 real(8) :: general_avalanche_relaxation_length=1.d0
 logical :: general_avalanche=.false.
 logical :: secondary_flow=.false.,secondary_flow_transport_direction=.false.
 integer :: secondary_flow_model=0,secondary_flow_inflow_mode=1
 real(8) :: secondary_flow_espir=1.d0,secondary_flow_momentum_beta=0.d0
 logical :: secondary_foundation_identity_declaration=.false.
 integer :: secondary_foundation_base_envelope=0
 real(8) :: secondary_foundation_rmincf=0.d0,secondary_foundation_chzmin=0.d0, &
  secondary_foundation_espir=0.d0,secondary_foundation_coriolis=0.d0, &
  secondary_foundation_betac=0.d0
 character(len=64) :: secondary_foundation_execution_fingerprint=''
 character(len=:),allocatable :: secondary_foundation_execution_preimage
 real(8) :: secondary_initial_integral=0.d0,secondary_source_cumulative=0.d0
 real(8) :: secondary_momentum_initial(2)=0.d0,secondary_momentum_boundary_cumulative(2)=0.d0
 real(8) :: secondary_stress_boundary_cumulative(2)=0.d0,secondary_momentum_remainder_cumulative(2)=0.d0
 real(8) :: secondary_momentum_boundary_trial(2,2)=0.d0
 logical :: bedload_adaptation=.false.
 real(8) :: bedload_adaptation_length_diameters=100.d0
 integer :: bedload_adaptation_iterations=0
 real(8) :: bedload_adaptation_residual=0.d0
 integer :: nmorfac=0
 real(8) :: morphological_factor=1.d0
 logical :: morphological_acceleration=.false.
 logical :: bedform_roughness=.false.
 logical :: bank_failure=.false.
 integer :: bank_toe_erosion_mode=0
 real(8) :: bank_toe_erodibility=0.d0,bank_toe_critical_shear=0.d0
 real(8) :: bank_toe_exponent=1.d0,bank_critical_undercut=0.d0
 real(8) :: bank_toe_capacity_fraction=1.d0
 real(8) :: bank_distributed_transport_fraction=.1d0,bank_distributed_critical_slope=.35d0
 real(8),allocatable :: bank_undercut(:),bank_reference_height(:),bank_toe_height(:),bank_reference_length(:)
 real(8),allocatable :: bank_toe_cumulative(:),bank_collapse_cumulative(:)
 integer,allocatable :: bank_collapse_count(:)
 logical :: bank_front_tracking=.false.
 integer,allocatable :: bank_front_status(:),bank_front_seed_face(:)
 real(8),allocatable :: bank_front_solid_area(:),bank_front_base(:),bank_front_height(:)
 real(8),allocatable :: bank_front_removed_mass(:,:)
 real(8) :: bank_front_retry_scale=1.d0
 logical :: bank_toe_mapping_ready=.false.
 integer,allocatable :: physical_toe_cell(:),physical_toe_donor(:),physical_toe_face(:), &
  physical_toe_status(:),physical_candidate_bank(:),physical_candidate_toe(:)
 logical,allocatable :: physical_is_toe_cell(:),physical_is_toe_donor(:),physical_is_toe_face(:)
 real(8),allocatable :: physical_toe_distance(:),physical_face_request(:),physical_donor_request(:), &
  physical_demand_volume(:),physical_before_inventory(:,:),physical_after_inventory(:,:), &
  physical_demand_mass(:,:),physical_removed_mass(:,:),physical_face_mass(:,:)
 real(8),allocatable :: physical_toe_flux_trial(:,:),physical_toe_flux_cumulative(:,:), &
  physical_toe_flux_absolute_cumulative(:,:)
 integer,allocatable :: physical_toe_flux_bank_id(:),physical_toe_flux_toe_id(:), &
  physical_toe_flux_local_face(:)
 logical,allocatable :: physical_toe_flux_active(:)
 logical :: production_section_geometry=.false.
 integer :: sediment_roughness_mode=0,hiding_exposure=0,mpm_friction_mode=0,settling_mode=0
 real(8) :: constitutive_kinematic_viscosity=1.d-6,suspended_adaptation_factor=1.d0
 integer :: capacity_fraction_basis=1
 integer :: settlefrm_code=0,settling_temperature_mode_code=0
 integer :: sediment_activity_contract_code=0,wet_mask_provider_code=0
 real(8) :: settling_fac_dss=1.d0,settling_temperature_c=0.d0
 real(8) :: settling_reference_concentration_declaration=1.d4
 real(8) :: suspended_calibration_factor=1.d0
 logical :: typed_constitutive_declaration=.false.
 logical :: flow2dh_boundary_diffusion_declaration=.false.
 logical :: flow2dh_amr_ownership_declaration=.false.
 logical :: flow2dh_neubc_sand=.false.
 integer :: flow2dh_horizontal_diffusivity_mode=0
 real(8) :: flow2dh_horizontal_diffusivity=0.d0
 real(8) :: flow2dh_diffusion_dry_threshold=0.d0
 real(8),parameter :: flow2dh_horizontal_schmidt=.7d0
 integer,parameter :: capacity_parameter_maximum=10
 character(len=64),parameter :: capacity_parameter_schema_expected= &
  '6a9237f904d32258afacfb4c0949316509e29403b1154cc712beb3e8717edcc4'
 character(len=64),parameter :: capacity_profile_schema_expected= &
  'c60c0ec96b946c5b9ef33f117ce0733deccd7d02137ffeb4f3fba11fa8f29bba'
 character(len=64),parameter :: capacity_diagnostic_schema_expected= &
  'f21b3228c09d105ea0f7986c1ac8d9725576edfdbce872c2ed094139da620a3a'
 character(len=64),parameter :: flow2dh_exchange_schema_expected= &
  '2064b1e5d879812812946e428b2cb77c6cf2712cfbc091d6061045398c56f346'
 character(len=64),parameter :: suspension_profile_schema_expected= &
  'e833adfb99f0831b59a57b2ac5ea2b41af9db79e627215fe71cb97e65297b012'
 character(len=64),parameter :: flow2dh_settling_schema_expected= &
  '870467c03fd8e96f5eb5b2ba2917f5763cf92d996d89bcc4a36e41469324f317'
 character(len=64),parameter :: flow2dh_open_boundary_schema_expected= &
  'f4f75c9012b8d3b8fd47f4f721bd5894dc0a73e294b5294acc3d07a5088edcaf'
 character(len=64),parameter :: flow2dh_horizontal_diffusion_schema_expected= &
  '9868ee69f3e95a27d64ec5b801ee447cfa410f6561fe3f28bc07a4ae7df97cdf'
 character(len=64),parameter :: suspended_amr_schema_expected= &
  '32b1648e8bada94cb8735ddcad656dfa4c1c88b5186c0b609f0d75a2192f7e80'
 character(len=64),parameter :: capacity_execution_schema_expected= &
  '2262a1f59fe1960774902a5b4c7e5253123cf3729b8cbc2cf7417ba2698d25f3'
 character(len=64),parameter :: capacity_foundation_execution_schema_expected= &
  'd25822d9801a748849d8bee6f4d9d31ee04547a52f9bea9b349d007a507ea677'
 character(len=*),parameter :: capacity_foundation_execution_scope_expected= &
  'braids-q3c-capacity-forcing-execution-v1'
 character(len=64),parameter :: hiding_foundation_execution_schema_expected= &
  '590cd2fa8091eb82826f65eb0944378a11d18695321f9add58d155ef6c4dc327'
 character(len=*),parameter :: hiding_foundation_execution_scope_expected= &
  'braids-q3h1-hiding-exposure-composition-v1'
 character(len=64),parameter :: secondary_foundation_execution_schema_expected= &
  '593790edba81303a0890d9ae006719f304298b0578997e63ae2ca3697fa4c9be'
 character(len=*),parameter :: secondary_foundation_execution_scope_expected= &
  'braids-q3h2-flow2dh-equilibrium-secondary-composition-v1'
 character(len=64),parameter :: capacity_route_profile_schema_expected= &
  '1fc85b4bfa3f1aa8065144879602ab703611908b1c8e64e1a77cc69c4115fc76'
 character(len=64),parameter :: channel_composition_scope_expected= &
  'delft3d4-flow-2dh-post-density-channel-composition-v1'
 character(len=64),parameter :: channel_composition_semantic_expected= &
  '69518e4129052727dbebee431c0f7846743612ec21a45ccadf25d28d4d25456b'
 integer,parameter :: channel_native_wire_schema_expected=2
 integer,parameter :: channel_native_max_preimage_ascii=32768
 integer,parameter :: channel_native_max_identifier_token_ascii=1536
 integer,parameter :: channel_native_hex_chunk=128
 integer,parameter :: execution_profile_schema_expected=2
 integer,parameter :: execution_profile_max_preimage_ascii=262144
 character(len=64),parameter :: execution_profile_scope_expected= &
  'delft3d4-flow-2dh-execution-profile-v2'
 character(len=64),parameter :: execution_profile_semantic_expected= &
  '4611257ccfa28c8d5b0134f6e39a63e65a26961d4b3fa9de47c1bd1cd89cf35a'
character(len=64),parameter :: execution_profile_ledger_expected= &
 '2d9346640e7235baf0be9ca99294328dca2b36f0f12d439e3734d6a6068a8611'
 character(len=64),parameter :: sediment_catalog_expected= &
  'f0fc1ae109dcbd1254a573ad87edaf76384cb74c821e089b7a428d3a0e2ab7f8'
 character(len=64),parameter :: capacity_parameter_catalog_expected= &
  'c8d95abb6f27c127b98f1ce058f415fc57df496608c83e7f79026cbb30ab23bd'
 integer,parameter :: component_native_maximum=16
 integer,parameter :: component_native_identifier_maximum=128
 type :: native_component_document
  character(len=component_native_identifier_maximum) :: component_id=''
  character(len=64) :: document_sha256=''
  character(len=:),allocatable :: preimage
 end type
 integer :: capacity_parameter_law=0,capacity_parameter_count=0
 logical :: capacity_split_retained=.false.
 logical :: capacity_foundation_runtime_declaration=.false.
 character(len=64) :: capacity_parameter_schema=''
 character(len=64) :: capacity_profile_schema=''
 character(len=64) :: capacity_diagnostic_schema='',flow2dh_exchange_schema='', &
  suspension_profile_schema='',flow2dh_settling_schema=''
 character(len=64) :: suspension_profile_fingerprint=''
 character(len=64) :: flow2dh_open_boundary_schema='', &
  flow2dh_horizontal_diffusion_schema=''
 character(len=64) :: suspended_amr_schema='',suspended_amr_fingerprint='', &
  suspended_amr_profile_fingerprint=''
 logical :: capacity_route_identity_declaration=.false.
 logical :: capacity_foundation_route_identity_declaration=.false.
 logical :: capacity_foundation_forcing_identity_declaration=.false.
 logical :: hiding_foundation_identity_declaration=.false.
 integer :: capacity_route_identity_schema=0
 character(len=64) :: capacity_route_method_id='',capacity_route_execution_schema='', &
  capacity_route_profile_schema='',capacity_route_execution_fingerprint='', &
  capacity_route_profile_fingerprint=''
 character(len=64) :: capacity_foundation_forcing_fingerprint='', &
  capacity_foundation_forcing_raw_sha256='', &
  capacity_foundation_native_execution_fingerprint=''
 character(len=:),allocatable :: capacity_foundation_forcing_preimage, &
  capacity_foundation_execution_preimage,hiding_foundation_execution_preimage
 character(len=64) :: hiding_foundation_execution_fingerprint=''
 real(8) :: capacity_route_acal=0.d0
 logical :: channel_composition_declaration=.false.
 logical :: channel_native_identity_declaration=.false.
 logical :: execution_profile_declaration=.false.
 logical :: component_native_identity_declaration=.false.
 integer :: component_native_count=0
 type(native_component_document),allocatable :: component_native_documents(:)
 integer :: channel_composition_schema=0,channel_composition_fraction_count=0
 integer :: channel_native_wire_schema=0,channel_native_preimage_length=0
 integer :: execution_profile_schema=0,execution_profile_preimage_length=0
 integer :: channel_native_identifier_token_length(32)=0
 integer :: channel_transport_class_input(32)=0
 real(8) :: channel_bed_factor=1.d0,channel_suspended_factor=1.d0, &
  channel_speed_epsilon=1.d-6
 character(len=64) :: channel_composition_scope='',channel_composition_semantic='', &
  channel_composition_profile_fingerprint=''
 character(len=64) :: channel_native_preimage_sha256=''
 character(len=64) :: execution_profile_semantic='',execution_profile_fingerprint=''
 character(len=channel_native_max_identifier_token_ascii) :: &
  channel_native_identifier_token(32)=''
 character(len=:),allocatable :: channel_native_preimage
 character(len=:),allocatable :: execution_profile_preimage
 real(8),allocatable :: capacity_parameter(:,:)
 real(8) :: hindered_settling_reference_concentration=1.d4
 real(8) :: bed_slope_alfabs=1.d0,bed_slope_alfabn=1.5d0,bed_friction_angle_deg=30.d0
 character(64) :: fingerprint
 ! Configurable nested adaptive tree. Immutable candidate geometry is separate from the
 ! active leaf arrays consumed by the existing finite-volume kernels.
 integer,parameter :: max_amr_levels=8
 integer,parameter :: max_diffusion_recovery_stencil=256
 real(8),parameter :: diffusion_sqrt_two=1.414213562373095048801688724209698d0
 real(8),parameter :: manning_chezy_minimum_depth=1.d-2
 real(8),parameter :: van_rijn_calculation_minimum_depth=1.d-2
 logical :: amr_enabled=.false.
 integer :: amr_nn=0,amr_fn=0,amr_fine_offset=0,amr_mesh_id=0,amr_regrid_count=0
 integer :: amr_levels=4,amr_ratio=2
 integer :: amr_delay=3,amr_wet_level=1,amr_buffer=3,amr_flags(5)=1
 real(8) :: amr_interval=1,amr_next=0,amr_wet_depth=1.d-6,amr_eta_tol=.02d0,amr_u_tol=.2d0
 real(8) :: amr_bed_tol=.02d0,amr_c_tol=.01d0,amr_mix_tol=.05d0,amr_dz_tol=.002d0
 integer, allocatable :: at_parent(:),at_level(:),at_i(:),at_j(:),at_stride(:),at_min(:),at_max(:),at_delay(:)
 integer, allocatable :: at_offset(:),at_child(:),at_leaf(:),at_cover(:),at_required(:)
 integer, allocatable :: af_owner(:),af_neighbour(:),af_bid(:),af_level(:),af_source(:),af_solid(:),at_face_source(:)
 ! ``at_manning`` stores the raw user field for the selected ROUMET branch
 ! (C, n, or ks).  It is a legacy external/deck storage name, not permission
 ! to apply Manning-only transfer semantics to Chezy/Nikuradse bytes.
 real(8), allocatable :: at_area(:),at_xy(:,:),at_rock(:),at_manning(:),at_q(:,:),at_initial(:,:),at_last_bed(:)
 integer :: at_roughness_mode=0
 real(8), allocatable :: af_normal(:,:),af_length(:),af_mid(:,:),af_bed(:),af_left(:),af_right(:)
 integer, allocatable :: af_sampled(:)
 logical, allocatable :: at_bank(:)
 real(8),allocatable :: at_bank_undercut(:),at_bank_reference_height(:),at_bank_toe_height(:),at_bank_reference_length(:)
 real(8),allocatable :: at_bank_toe_cumulative(:),at_bank_collapse_cumulative(:)
 integer,allocatable :: at_bank_collapse_count(:)
 integer,allocatable :: at_bank_front_status(:),at_bank_front_seed_face(:)
 real(8),allocatable :: at_bank_front_solid_area(:),at_bank_front_base(:),at_bank_front_height(:)
 real(8),allocatable :: at_bank_front_removed_mass(:,:)
logical :: amr_subcycling=.false.
 logical :: amr_absolute_boundary_bed=.false.
 logical :: amr_diagnostics=.false.
 integer :: amr_diagnostic_unit=-1,amr_diagnostic_summary_unit=-1
 integer :: amr_step_level=-1,amr_failed_level=-1,amr_level_steps(0:max_amr_levels-1)=0
 integer :: amr_failure_cell=0,amr_failure_state=0
 real(8) :: amr_failure_value=0,amr_failure_aux=0
 real(8),allocatable :: amr_dense_old(:,:),amr_dense_new(:,:),amr_dense_t0(:),amr_dense_t1(:)
 integer :: amr_level_cell_count(0:max_amr_levels-1)=0,amr_level_face_count(0:max_amr_levels-1)=0
 integer :: amr_reflux_face_count(0:max_amr_levels-1)=0,amr_subtree_face_count(0:max_amr_levels-1)=0
 integer :: amr_level_eval_cell_count(0:max_amr_levels-1)=0,amr_level_face_cell_count(0:max_amr_levels-1)=0
 integer,allocatable :: amr_level_cells(:,:),amr_level_faces(:,:),amr_reflux_faces(:,:),amr_subtree_faces(:,:)
 integer,allocatable :: amr_level_eval_cells(:,:),amr_level_face_cells(:,:)
 ! Bit 0: evaluate wet/dry traces; bit 1: reconstruct a participating face cell.
 integer,allocatable :: amr_prepare_role(:,:)
 real(8),allocatable :: amr_rk_state(:,:,:),amr_rk_flux(:,:,:,:)
 real(8),allocatable :: amr_level_flux(:,:,:,:),amr_subtree_flux(:,:,:,:),amr_state_work(:,:),amr_accepted_work(:,:)
 ! Optional wall-clock profiling.  These counters are deliberately outside the
 ! numerical state and are only updated when BRAIDS_PROFILE=1 is requested.
 logical :: profile_enabled=.false.
 ! Whether any suspended equation can depart from its exact zero solution.
 ! This is immutable after the resolved initial/restart state is known: only
 ! exchange, prescribed boundary concentration, or carried inventory can
 ! activate it. Caching the answer avoids seven full-state ANY reductions at
 ! every SSPRK stage in bedload-only braided-river calculations.
 logical :: suspended_transport_active=.true.
 logical :: hydraulic_workflow_fast=.false.
 integer :: profile_hydraulic_workflow_calls=0
 real(8) :: profile_started=0.d0,profile_regrid_s=0.d0,profile_prepare_s=0.d0
 real(8) :: profile_euler_s=0.d0,profile_source_s=0.d0,profile_output_s=0.d0
 real(8) :: profile_output_authoritative_s=0.d0,profile_output_preview_pack_s=0.d0
 real(8) :: profile_output_preview_io_s=0.d0,profile_output_checkpoint_s=0.d0
 real(8) :: profile_euler_setup_s=0.d0,profile_euler_flux_s=0.d0
 real(8) :: profile_euler_limit_s=0.d0,profile_euler_finalize_s=0.d0
 real(8) :: profile_euler_boundary_s=0.d0,profile_euler_bedload_s=0.d0
 real(8) :: profile_euler_hiding_s=0.d0,profile_euler_mixture_s=0.d0
 real(8) :: profile_euler_capacity_cell_s=0.d0,profile_euler_capacity_gradient_s=0.d0
 real(8) :: profile_euler_reconstruction_s=0.d0,profile_euler_trace_s=0.d0
 integer :: profile_regrid_calls=0,profile_prepare_calls=0,profile_euler_calls=0,profile_source_calls=0,profile_output_calls=0
 logical :: gpu_export_diffusion=.false.

contains
 subroutine prepare_adaptive_traces_fused(state,v,bc_concentration,grad,vlo,vhi)
  ! Same scheme-6 kernels, preserving the distinct AMR cell subsets.
  ! Only immutable v/geometry cross cell boundaries in these three kernels.
  real(8), intent(in) :: state(nstate,nc),v(nprimitive,nc),bc_concentration(nfrac,nb)
  real(8), intent(out) :: grad(nprimitive,2,nc),vlo(nprimitive,nc),vhi(nprimitive,nc)
  real(8) :: a11,a12,a22,b(nprimitive,2),r(2),w,det,delta(nprimitive),theta(nprimitive),eps_guard(nprimitive),z,adjacent(nprimitive)
  real(8) :: stencil(5,nprimitive),tangential(5),n(2),tangent(2),un,coef,tlo,thi,ut,h,point(2),ratio,normal_q,eta,zb
  integer :: c,k,f,ncell,m,g,once,a,side,slot,kk,s,kind,other
  logical :: wet,flatten
  !$omp parallel do schedule(static,256) default(shared) &
  !$omp& private(c,k,f,ncell,m,g,once,a,side,slot,kk,s,kind,other,a11,a12,a22,b,r,w,det,delta,theta,eps_guard,z,adjacent,stencil,tangential,n,tangent,un,coef,tlo,thi,ut,h,point,ratio,normal_q,eta,zb,wet,flatten)
  do c=1,nc
   grad(:,:,c)=0;vlo(:,c)=v(:,c);vhi(:,c)=v(:,c)
   wall_ready(:,c)=.false.;wall_values(:,:,c)=0;hydro_constant(c)=.false.
   do once=1,1
   if(amr_step_level>=0) then
    if(.not.btest(amr_prepare_role(c,amr_step_level),1)) cycle
   endif
   if(.not.active_cell(c)) cycle
   a11=hydraulic_ls_matrix(1,c);a12=hydraulic_ls_matrix(2,c);a22=hydraulic_ls_matrix(3,c);b=0
   do k=offset(c),offset(c+1)-1
    f=faces(k);ncell=neighbour(f)
    if(signs(k)<0) ncell=owner(f)
    if(ncell==0) then
     if(bid(f)>0) then
      do g=1,nfrac
       m=concentration_index(g)
       ! The v48 concentration at an open face is resolved only after the
       ! accepted hydraulic face direction is known (prescribed/return-time/
       ! NeuBcSand).  A raw per-boundary schedule is therefore not a valid
       ! reconstruction bound.  Keep the frozen generic/v47 bound unchanged,
       ! and let v48 use its resolved face value in the FCT boundary mapping.
       if(.not.flow2dh_boundary_diffusion_declaration) then
        vlo(m,c)=min(vlo(m,c),bc_concentration(g,bid(f)))
        vhi(m,c)=max(vhi(m,c),bc_concentration(g,bid(f)))
       endif
      enddo
     endif
     cycle
    endif
    r=bed_ls_delta(:,k)
    ! Integrate the neighbour equation over its aperture.  A geometric face
    ! split must not count the same neighbour twice in the normal equations.
    w=hydraulic_ls_weight(k)
    adjacent=v(:,ncell)
    ! High dry terrain is not a higher water surface. Extend this wet cell's
    ! eta into that dry stencil point; otherwise a stationary shoreline can
    ! reconstruct a fictitious uphill water column. Lower dry land remains
    ! available for inundation. Do not alter terrain or conserved inventory.
    if(v(1,c)-v(bed_primitive_index(),c)>dry_tolerance.and. &
       v(1,ncell)-v(bed_primitive_index(),ncell)<=dry_tolerance.and. &
       v(bed_primitive_index(),ncell)>=v(1,c)) &
     adjacent(1)=v(1,c)
    b(:,1)=b(:,1)+w*r(1)*(adjacent-v(:,c));b(:,2)=b(:,2)+w*r(2)*(adjacent-v(:,c))
    vlo(:,c)=min(vlo(:,c),adjacent);vhi(:,c)=max(vhi(:,c),adjacent)
   enddo
   det=hydraulic_ls_matrix(4,c)
   if(det<=1.d-12*(a11+a22)**2) cycle
   grad(:,1,c)=(a22*b(:,1)-a12*b(:,2))/det
   grad(:,2,c)=(a11*b(:,2)-a12*b(:,1))/det
   eps_guard=0.d0
   do m=1,3
    eps_guard(m)=64*epsilon(1.d0)* &
      max(1.d0,abs(v(m,c)),abs(vlo(m,c)),abs(vhi(m,c)))
   enddo
   m=bed_primitive_index()
   eps_guard(m)=64*epsilon(1.d0)* &
     max(1.d0,abs(v(m,c)),abs(vlo(m,c)),abs(vhi(m,c)))
   theta=1
   do k=offset(c),offset(c+1)-1
    f=faces(k);r=mid(:,f)-xy(:,c);delta=grad(:,1,c)*r(1)+grad(:,2,c)*r(2)
    ! A sloping stage outlet lies beyond the last cell-centre extremum.
    ! Limiting against that extremum flattens both terrain and water surface
    ! throughout the end cell and removes its downstream half-cell gravity
    ! contribution. Keep the one-sided slope here; internal faces still
    ! supply all ordinary bounds. Flat-bed and other boundary modes retain
    ! their previous reconstruction exactly.
    if(neighbour(f)==0.and.bid(f)>0) then
     if(mode(bid(f))==2.and.v(1,c)-v(bed_primitive_index(),c)>dry_tolerance) then
      if(abs(dot_product(v(2:3,c),normal(:,f)))<sqrt(grav*(v(1,c)-v(bed_primitive_index(),c)))) then
       if(abs(delta(bed_primitive_index()))>64*epsilon(1.d0)*max(1.d0,abs(v(bed_primitive_index(),c)))) cycle
      endif
     endif
    endif
    do m=1,nprimitive
     if(abs(delta(m))<=eps_guard(m)) cycle
     if(delta(m)>0) theta(m)=min(theta(m),(vhi(m,c)-v(m,c))/delta(m))
     if(delta(m)<0) theta(m)=min(theta(m),(vlo(m,c)-v(m,c))/delta(m))
    enddo
   enddo
   do m=1,nprimitive
    grad(m,:,c)=grad(m,:,c)*max(0.d0,theta(m))
   enddo
   enddo
   do once=1,1
   if(state(1,c)<=dry_tolerance) cycle
   if(wall_triangle(c)) then
    if(any(wall_stencil_ids(:,:,c)==0)) cycle
    n=wall_normal(:,c);tangent=[-n(2),n(1)]
    tlo=dot_product(v(2:3,c),tangent);thi=tlo
    do k=offset(c),offset(c+1)-1
     f=faces(k);other=neighbour(f)
     if(signs(k)<0) other=owner(f)
     if(other<=0) cycle
     ut=dot_product(v(2:3,other),tangent);tlo=min(tlo,ut);thi=max(thi,ut)
    enddo
   endif
   do a=1,2
    if(any(wall_stencil_ids(:,a,c)==0)) cycle
    wet=.true.
    do k=1,5
     if(state(1,wall_stencil_ids(k,a,c))<=dry_tolerance) wet=.false.
    enddo
    if(.not.wet) cycle
    do side=1,2
     slot=2*(a-1)+side
     do k=1,5
      kk=k
      if(side==1) kk=6-k
      s=wall_stencil_ids(kk,a,c);kind=wall_stencil_kind(kk,a,c)
      stencil(k,:)=v(:,s)
      if(kind>0) then
       n=wall_stencil_normal(:,kk,a,c);coef=real(kind,8)
       un=dot_product(stencil(k,2:3),n);stencil(k,2:3)=stencil(k,2:3)-coef*un*n
      endif
      if(wall_triangle(c)) tangential(k)=dot_product(stencil(k,2:3),tangent)
     enddo
     do m=1,nprimitive
      wall_values(m,slot,c)=reconstruct(stencil(:,m),5)
     enddo
     wall_values(1,slot,c)=max(vlo(1,c),min(vhi(1,c),wall_values(1,slot,c)))
     wall_values(bed_primitive_index(),slot,c)=max(vlo(bed_primitive_index(),c), &
      min(vhi(bed_primitive_index(),c),wall_values(bed_primitive_index(),slot,c)))
     if(wall_triangle(c)) then
      ut=max(tlo,min(thi,reconstruct(tangential,5)))
      wall_values(2:3,slot,c)=ut*tangent
     else
      wall_values(2:3,slot,c)=max(vlo(2:3,c),min(vhi(2:3,c),wall_values(2:3,slot,c)))
     endif
     wall_ready(slot,c)=.true.
    enddo
   enddo
   if(.not.wall_triangle(c)) cycle
   if(.not.all(wall_ready(:,c)).or. &
      any(wall_values(1,:,c)-wall_values(bed_primitive_index(),:,c)<=dry_tolerance)) then
    wall_ready(:,c)=.false.;cycle
   endif
   n=wall_normal(:,c);normal_q=dot_product(state(2:3,c),n)
   do a=1,2
    do side=1,2
     slot=2*(a-1)+side;h=wall_values(1,slot,c)-wall_values(bed_primitive_index(),slot,c)
     point=wall_box_center(:,c);point(a)=point(a)+(side-1.5d0)*dx
     ratio=dot_product(point-wall_box_center(:,c),n)/wall_mean_distance(c)
     wall_values(2:3,slot,c)=wall_values(2:3,slot,c)+normal_q*ratio/h*n
    enddo
   enddo
   enddo
   if(amr_step_level>=0) then
    if(.not.btest(amr_prepare_role(c,amr_step_level),0)) cycle
   endif
   h=state(1,c);flatten=h<=dry_tolerance
   if(.not.flatten) then
    do k=offset(c),offset(c+1)-1
     f=faces(k);call face_elevations(c,f,v,grad,vlo,vhi,eta,zb)
     if(eta-zb<=dry_tolerance.or. &
        (any(rectangle_width(:,c)<=0).and. &
         abs(zb-v(bed_primitive_index(),c))>=h)) then
      flatten=.true.;exit
     endif
    enddo
   endif
   hydro_constant(c)=flatten
  enddo
  !$omp end parallel do
 end subroutine

 subroutine prepare_hydraulic_traces_fused(state,v,bc_concentration,grad,vlo,vhi)
  ! Same fixed-grid scheme-6 formulas and per-cell accumulation order.
  ! Only immutable v/geometry cross cell boundaries in these three kernels.
  real(8), intent(in) :: state(nstate,nc),v(nprimitive,nc),bc_concentration(nfrac,nb)
  real(8), intent(out) :: grad(nprimitive,2,nc),vlo(nprimitive,nc),vhi(nprimitive,nc)
  real(8) :: a11,a12,a22,b(nprimitive,2),r(2),w,det,delta(nprimitive),theta(nprimitive),eps_guard(nprimitive),z,adjacent(nprimitive)
  real(8) :: stencil(5,nprimitive),tangential(5),n(2),tangent(2),un,coef,tlo,thi,ut,h,point(2),ratio,normal_q,eta,zb
  integer :: c,k,f,ncell,m,g,once,a,side,slot,kk,s,kind,other
  logical :: wet,flatten
  !$omp parallel do schedule(static,256) default(shared) &
  !$omp& private(c,k,f,ncell,m,g,once,a,side,slot,kk,s,kind,other,a11,a12,a22,b,r,w,det,delta,theta,eps_guard,z,adjacent,stencil,tangential,n,tangent,un,coef,tlo,thi,ut,h,point,ratio,normal_q,eta,zb,wet,flatten)
  do c=1,nc
   grad(:,:,c)=0;vlo(:,c)=v(:,c);vhi(:,c)=v(:,c)
   wall_ready(:,c)=.false.;wall_values(:,:,c)=0;hydro_constant(c)=.false.
   do once=1,1
   if(.not.active_cell(c)) cycle
   a11=hydraulic_ls_matrix(1,c);a12=hydraulic_ls_matrix(2,c);a22=hydraulic_ls_matrix(3,c);b=0
   do k=offset(c),offset(c+1)-1
    f=faces(k);ncell=neighbour(f)
    if(signs(k)<0) ncell=owner(f)
    if(ncell==0) then
     if(bid(f)>0) then
      do g=1,nfrac
       m=concentration_index(g)
       if(.not.flow2dh_boundary_diffusion_declaration) then
        vlo(m,c)=min(vlo(m,c),bc_concentration(g,bid(f)))
        vhi(m,c)=max(vhi(m,c),bc_concentration(g,bid(f)))
       endif
      enddo
     endif
     cycle
    endif
    r=bed_ls_delta(:,k)
    ! Integrate the neighbour equation over its aperture.  A geometric face
    ! split must not count the same neighbour twice in the normal equations.
    w=hydraulic_ls_weight(k)
    adjacent=v(:,ncell)
    ! High dry terrain is not a higher water surface. Extend this wet cell's
    ! eta into that dry stencil point; otherwise a stationary shoreline can
    ! reconstruct a fictitious uphill water column. Lower dry land remains
    ! available for inundation. Do not alter terrain or conserved inventory.
    if(v(1,c)-v(bed_primitive_index(),c)>dry_tolerance.and. &
       v(1,ncell)-v(bed_primitive_index(),ncell)<=dry_tolerance.and. &
       v(bed_primitive_index(),ncell)>=v(1,c)) &
     adjacent(1)=v(1,c)
    b(:,1)=b(:,1)+w*r(1)*(adjacent-v(:,c));b(:,2)=b(:,2)+w*r(2)*(adjacent-v(:,c))
    vlo(:,c)=min(vlo(:,c),adjacent);vhi(:,c)=max(vhi(:,c),adjacent)
   enddo
   det=hydraulic_ls_matrix(4,c)
   if(det<=1.d-12*(a11+a22)**2) cycle
   grad(:,1,c)=(a22*b(:,1)-a12*b(:,2))/det
   grad(:,2,c)=(a11*b(:,2)-a12*b(:,1))/det
   call cpu_outlet_bounds(c,v,grad,vlo,vhi)
   eps_guard=0.d0
   do m=1,3
    eps_guard(m)=64*epsilon(1.d0)* &
      max(1.d0,abs(v(m,c)),abs(vlo(m,c)),abs(vhi(m,c)))
   enddo
   m=bed_primitive_index()
   eps_guard(m)=64*epsilon(1.d0)* &
     max(1.d0,abs(v(m,c)),abs(vlo(m,c)),abs(vhi(m,c)))
   theta=1
   do k=offset(c),offset(c+1)-1
    f=faces(k);r=mid(:,f)-xy(:,c);delta=grad(:,1,c)*r(1)+grad(:,2,c)*r(2)
    ! A sloping stage outlet lies beyond the last cell-centre extremum.
    ! Limiting against that extremum flattens both terrain and water surface
    ! throughout the end cell and removes its downstream half-cell gravity
    ! contribution. Keep the one-sided slope here; internal faces still
    ! supply all ordinary bounds. Flat-bed and other boundary modes retain
    ! their previous reconstruction exactly.
    if(neighbour(f)==0.and.bid(f)>0) then
     if(mode(bid(f))==2.and.v(1,c)-v(bed_primitive_index(),c)>dry_tolerance) then
      if(abs(dot_product(v(2:3,c),normal(:,f)))<sqrt(grav*(v(1,c)-v(bed_primitive_index(),c)))) then
       if(abs(delta(bed_primitive_index()))>64*epsilon(1.d0)*max(1.d0,abs(v(bed_primitive_index(),c)))) cycle
      endif
     endif
    endif
    do m=1,nprimitive
     if(abs(delta(m))<=eps_guard(m)) cycle
     if(delta(m)>0) theta(m)=min(theta(m),(vhi(m,c)-v(m,c))/delta(m))
     if(delta(m)<0) theta(m)=min(theta(m),(vlo(m,c)-v(m,c))/delta(m))
    enddo
   enddo
   do m=1,nprimitive
    grad(m,:,c)=grad(m,:,c)*max(0.d0,theta(m))
   enddo
   enddo
   do once=1,1
   if(state(1,c)<=dry_tolerance) cycle
   if(wall_triangle(c)) then
    if(any(wall_stencil_ids(:,:,c)==0)) cycle
    n=wall_normal(:,c);tangent=[-n(2),n(1)]
    tlo=dot_product(v(2:3,c),tangent);thi=tlo
    do k=offset(c),offset(c+1)-1
     f=faces(k);other=neighbour(f)
     if(signs(k)<0) other=owner(f)
     if(other<=0) cycle
     ut=dot_product(v(2:3,other),tangent);tlo=min(tlo,ut);thi=max(thi,ut)
    enddo
   endif
   do a=1,2
    if(any(wall_stencil_ids(:,a,c)==0)) cycle
    wet=.true.
    do k=1,5
     if(state(1,wall_stencil_ids(k,a,c))<=dry_tolerance) wet=.false.
    enddo
    if(.not.wet) cycle
    do side=1,2
     slot=2*(a-1)+side
     do k=1,5
      kk=k
      if(side==1) kk=6-k
      s=wall_stencil_ids(kk,a,c);kind=wall_stencil_kind(kk,a,c)
      stencil(k,:)=v(:,s)
      if(kind>0) then
       n=wall_stencil_normal(:,kk,a,c);coef=real(kind,8)
       un=dot_product(stencil(k,2:3),n);stencil(k,2:3)=stencil(k,2:3)-coef*un*n
      endif
      if(wall_triangle(c)) tangential(k)=dot_product(stencil(k,2:3),tangent)
     enddo
     do m=1,nprimitive
      wall_values(m,slot,c)=reconstruct(stencil(:,m),5)
     enddo
     wall_values(1,slot,c)=max(vlo(1,c),min(vhi(1,c),wall_values(1,slot,c)))
     wall_values(bed_primitive_index(),slot,c)=max(vlo(bed_primitive_index(),c), &
      min(vhi(bed_primitive_index(),c),wall_values(bed_primitive_index(),slot,c)))
     if(wall_triangle(c)) then
      ut=max(tlo,min(thi,reconstruct(tangential,5)))
      wall_values(2:3,slot,c)=ut*tangent
     else
      wall_values(2:3,slot,c)=max(vlo(2:3,c),min(vhi(2:3,c),wall_values(2:3,slot,c)))
     endif
     wall_ready(slot,c)=.true.
    enddo
   enddo
   if(.not.wall_triangle(c)) cycle
   if(.not.all(wall_ready(:,c)).or. &
      any(wall_values(1,:,c)-wall_values(bed_primitive_index(),:,c)<=dry_tolerance)) then
    wall_ready(:,c)=.false.;cycle
   endif
   n=wall_normal(:,c);normal_q=dot_product(state(2:3,c),n)
   do a=1,2
    do side=1,2
     slot=2*(a-1)+side;h=wall_values(1,slot,c)-wall_values(bed_primitive_index(),slot,c)
     point=wall_box_center(:,c);point(a)=point(a)+(side-1.5d0)*dx
     ratio=dot_product(point-wall_box_center(:,c),n)/wall_mean_distance(c)
     wall_values(2:3,slot,c)=wall_values(2:3,slot,c)+normal_q*ratio/h*n
    enddo
   enddo
   enddo
   h=state(1,c);flatten=h<=dry_tolerance
   if(.not.flatten) then
    do k=offset(c),offset(c+1)-1
     f=faces(k);call face_elevations(c,f,v,grad,vlo,vhi,eta,zb)
     if(eta-zb<=dry_tolerance.or. &
        (any(rectangle_width(:,c)<=0).and. &
         abs(zb-v(bed_primitive_index(),c))>=h)) then
      flatten=.true.;exit
     endif
    enddo
   endif
   hydro_constant(c)=flatten
  enddo
  !$omp end parallel do
 end subroutine
 pure integer function suspended_index(fraction) result(index)
  integer, intent(in) :: fraction
  index=3+fraction
 end function

 pure integer function bed_index(fraction) result(index)
  integer, intent(in) :: fraction
  index=3+nfrac+fraction
 end function

 pure integer function substrate_index(fraction) result(index)
  integer, intent(in) :: fraction
  index=bed_layer_index(2,fraction)
 end function

 pure integer function bed_layer_index(layer,fraction) result(index)
  integer, intent(in) :: layer,fraction
  index=3+nfrac+(layer-1)*nfrac+fraction
 end function

 pure integer function concentration_index(fraction) result(index)
  integer, intent(in) :: fraction
  index=3+fraction
 end function

 pure integer function bed_primitive_index() result(index)
  index=nprimitive
 end function

 function capacity_surface_fraction(state,fraction,cell) result(value)
  ! FLOW getfrac supplies either dry-mass or solid-volume fractions according
  ! to IFractions.  Keep this separate from bed_availability: the latter also
  ! contains BRAIDS' thin-layer transport limiter and is not the eqtran
  ! concentration multiplier.
  real(8),intent(in) :: state(nstate,nc)
  integer,intent(in) :: fraction,cell
  real(8) :: value,total
  integer :: g
  value=0.d0;total=0.d0
  if(capacity_fraction_basis==1) then
   do g=1,nfrac
    total=total+state(bed_index(g),cell)
   enddo
   if(total>0.d0) value=state(bed_index(fraction),cell)/total
  else
   do g=1,nfrac
    total=total+state(bed_index(g),cell)/bulk_fraction(g)
   enddo
   if(total>0.d0) value=(state(bed_index(fraction),cell)/bulk_fraction(fraction))/total
  endif
 end function

 pure logical function prognostic_secondary_flow() result(enabled)
  enabled=secondary_flow.and.secondary_flow_model==2
 end function

 pure logical function classic_secondary_foundation() result(enabled)
  ! The authenticated outer v56 envelope is deliberately distinct from the
  ! historical numeric model-1 route carried by v28.
  enabled=secondary_foundation_identity_declaration
 end function

 pure logical function secondary_bedload_direction() result(enabled)
  enabled=classic_secondary_foundation().or.secondary_flow_model==1.or. &
   (prognostic_secondary_flow().and.secondary_flow_transport_direction)
 end function

 pure integer function secondary_state_index() result(index)
  index=nphysical_state+1
 end function

 pure integer function secondary_primitive_index() result(index)
  index=nphysical_primitive
 end function

 pure integer function secondary_ledger_index() result(index)
  index=nphysical_ledger+1
 end function

 pure integer function suspended_advective_ledger_index(fraction) result(index)
  integer,intent(in) :: fraction
  index=nphysical_ledger+merge(1,0,prognostic_secondary_flow())+fraction
 end function

 pure integer function suspended_diffusive_ledger_index(fraction) result(index)
  integer,intent(in) :: fraction
  index=nphysical_ledger+merge(1,0,prognostic_secondary_flow())+nfrac+fraction
 end function

 integer function flow2dh_memory_face(face) result(index)
  ! Return the restart-stable storage identity for an active mesh face.
  ! Fixed-grid decks retain their historical one-to-one face numbering.
  integer,intent(in) :: face
  if(face<1.or.face>nf) error stop 'invalid active FLOW 2DH memory face'
  index=face
  if(amr_enabled) then
   if(.not.allocated(at_face_source)) &
    error stop 'adaptive FLOW 2DH memory requested before face identity exists'
   index=at_face_source(face)
   if(index<1.or.index>amr_fn) &
    error stop 'invalid stable adaptive FLOW 2DH memory face'
  endif
 end function

 pure integer function capacity_diagnostic_applicability(critical_shields) result(mask)
  real(8),intent(in) :: critical_shields
  ! Bits: 1 absolute Shields, 2 critical Shields, 4 stress ratio,
  ! 8 formula-specific transport stage.  Zero remains the numeric N/A value.
  mask=1+2
  if(critical_shields>0.d0) mask=mask+4
  if(bedlaw==12) mask=mask+8
 end function

 subroutine domain_momentum(state,value)
  real(8),intent(in) :: state(nstate,nc)
  real(8),intent(out) :: value(2)
  integer :: c
  value=0.d0
  do c=1,nc
   value=value+area(c)*state(2:3,c)
  enddo
 end subroutine

 subroutine allocate_secondary_work()
  if(allocated(secondary_curvature)) deallocate(secondary_curvature)
  if(allocated(secondary_classic_intensity)) deallocate(secondary_classic_intensity)
  if(allocated(secondary_face_left)) deallocate(secondary_face_left,secondary_face_right, &
   secondary_boundary_intensity,secondary_low_flux,secondary_anti_flux,secondary_stress_face,secondary_rplus, &
   secondary_rminus,secondary_low_state,secondary_source_work)
  if(.not.secondary_flow.and..not.classic_secondary_foundation()) return
  allocate(secondary_curvature(nc));secondary_curvature=0.d0
  if(classic_secondary_foundation()) then
   allocate(secondary_classic_intensity(nc));secondary_classic_intensity=0.d0
  endif
  if(.not.prognostic_secondary_flow()) return
  allocate(secondary_face_left(nf),secondary_face_right(nf),secondary_boundary_intensity(nf), &
   secondary_low_flux(nf),secondary_anti_flux(nf),secondary_stress_face(2,nf),secondary_rplus(nc),secondary_rminus(nc), &
   secondary_low_state(nc),secondary_source_work(nc))
  secondary_face_left=0.d0;secondary_face_right=0.d0;secondary_boundary_intensity=0.d0
  secondary_low_flux=0.d0;secondary_anti_flux=0.d0
  secondary_stress_face=0.d0
  secondary_rplus=1.d0;secondary_rminus=1.d0;secondary_low_state=0.d0;secondary_source_work=0.d0
 end subroutine

 subroutine release_physical_toe_work()
  if(allocated(physical_toe_cell)) then
   deallocate(physical_toe_cell,physical_toe_donor,physical_toe_face,physical_toe_status, &
    physical_candidate_bank,physical_candidate_toe,physical_is_toe_cell, &
    physical_is_toe_donor,physical_is_toe_face,physical_toe_distance, &
    physical_face_request,physical_donor_request,physical_demand_volume, &
    physical_before_inventory,physical_after_inventory,physical_demand_mass, &
    physical_removed_mass,physical_face_mass)
  endif
  if(allocated(physical_toe_flux_trial)) deallocate(physical_toe_flux_trial)
  bank_toe_mapping_ready=.false.
 end subroutine

 subroutine allocate_physical_toe_work()
  call release_physical_toe_work()
  allocate(physical_toe_cell(nc),physical_toe_donor(nc),physical_toe_face(nc), &
   physical_toe_status(nc),physical_candidate_bank(nf),physical_candidate_toe(nf), &
   physical_is_toe_cell(nc),physical_is_toe_donor(nc),physical_is_toe_face(nf), &
   physical_toe_distance(nc),physical_face_request(nf),physical_donor_request(nc), &
   physical_demand_volume(nc),physical_before_inventory(nfrac,nc), &
   physical_after_inventory(nfrac,nc),physical_demand_mass(nfrac,nc), &
   physical_removed_mass(nfrac,nc),physical_face_mass(nfrac,nf),physical_toe_flux_trial(nfrac,nf))
  physical_toe_cell=0;physical_toe_donor=0;physical_toe_face=0;physical_toe_status=0
  physical_candidate_bank=0;physical_candidate_toe=0
  physical_is_toe_cell=.false.;physical_is_toe_donor=.false.;physical_is_toe_face=.false.
  physical_toe_distance=0.d0;physical_face_request=0.d0;physical_donor_request=0.d0
  physical_demand_volume=0.d0;physical_before_inventory=0.d0;physical_after_inventory=0.d0
  physical_demand_mass=0.d0;physical_removed_mass=0.d0;physical_face_mass=0.d0
  physical_toe_flux_trial=0.d0
 end subroutine

 subroutine refresh_physical_toe_flux_metadata()
  integer :: f,source
  if(.not.allocated(physical_toe_flux_active)) return
  physical_toe_flux_active=.false.;physical_toe_flux_local_face=0
  do f=1,nf
   if(.not.physical_is_toe_face(f)) cycle
   source=f
   if(amr_enabled) source=at_face_source(f)
   if(source<1.or.source>size(physical_toe_flux_active)) &
    error stop 'invalid stable physical bank-toe segment ID'
   physical_toe_flux_active(source)=.true.;physical_toe_flux_local_face(source)=f
   if(amr_enabled) then
    physical_toe_flux_bank_id(source)=at_leaf(physical_candidate_bank(f))
    physical_toe_flux_toe_id(source)=at_leaf(physical_candidate_toe(f))
   else
    physical_toe_flux_bank_id(source)=physical_candidate_bank(f)
    physical_toe_flux_toe_id(source)=physical_candidate_toe(f)
   endif
  enddo
 end subroutine

 subroutine build_physical_toe_map(state)
  real(8),intent(in) :: state(nstate,nc)
  real(8) :: elevation_tolerance,distance_tolerance,bed_scale
  integer :: mapping_error
  if(bank_toe_erosion_mode/=2) return
  call allocate_physical_toe_work()
  call bank_bed_elevation(state,bank_bed)
  bed_scale=max(1.d0,maxval(abs(bank_bed)))
  elevation_tolerance=512.d0*epsilon(1.d0)*bed_scale
  distance_tolerance=512.d0*epsilon(1.d0)*max(1.d0,max(dx,dy))
  call braids_map_bank_toes(nc,nf,xy,area,bank_bed,bank_erodible,owner,neighbour, &
   mid,normal,length,elevation_tolerance,distance_tolerance,physical_toe_cell, &
   physical_toe_donor,physical_toe_face,physical_toe_distance, &
   physical_is_toe_cell,physical_is_toe_donor,physical_is_toe_face, &
   physical_candidate_bank,physical_candidate_toe,physical_toe_status,mapping_error)
  if(mapping_error/=BANK_TOE_SUCCESS.or. &
     any(bank_erodible.and.physical_toe_status/=BANK_TOE_MAPPED)) &
   error stop 'noncohesive bank material does not map uniquely to a physical toe'
  call refresh_physical_toe_flux_metadata()
  bank_toe_mapping_ready=.true.
 end subroutine

 subroutine commit_physical_toe_flux()
  integer :: f,source
  if(bank_toe_erosion_mode/=2) return
  if(.not.bank_toe_mapping_ready.or..not.allocated(physical_toe_flux_trial)) &
   error stop 'physical bank-toe flux committed before geometry mapping'
  if(.not.all(ieee_is_finite(physical_toe_flux_trial))) &
   error stop 'nonfinite accepted physical bank-toe flux'
  do f=1,nf
   if(.not.physical_is_toe_face(f)) cycle
   source=f
   if(amr_enabled) source=at_face_source(f)
   if(source<1.or.source>size(physical_toe_flux_cumulative,2)) &
    error stop 'invalid stable physical bank-toe flux segment'
   physical_toe_flux_cumulative(:,source)=physical_toe_flux_cumulative(:,source)+ &
    physical_toe_flux_trial(:,f)
   ! Gross is the sum of absolute accepted SSPRK/reflux step-net transport on
   ! each immutable physical segment. It is not reconstructed from cell change.
   physical_toe_flux_absolute_cumulative(:,source)=physical_toe_flux_absolute_cumulative(:,source)+ &
    abs(physical_toe_flux_trial(:,f))
  enddo
 end subroutine

 subroutine read_v51_record(unit,record,valid)
  ! Consume one complete v51 record without allowing a fixed-width CHARACTER
  ! value to hide a nonblank suffix.  Harmless trailing blanks are accepted;
  ! every nonblank character must fit in the caller's record buffer.
  integer,intent(in) :: unit
  character(len=*),intent(out) :: record
  logical,intent(out) :: valid
  character(len=128) :: chunk
  integer :: io_status,transferred,position,index
  record='';valid=.true.;position=0
  do
   chunk='';transferred=0
   read(unit,'(a)',advance='no',size=transferred,iostat=io_status) chunk
   do index=1,transferred
    position=position+1
    if(position<=len(record)) then
     record(position:position)=chunk(index:index)
    else if(chunk(index:index)/=' ') then
     valid=.false.
    endif
   enddo
   if(io_status==iostat_eor) exit
   if(io_status/=0) then
    valid=.false.;exit
   endif
  enddo
 end subroutine

 pure integer function v51_token_count(record) result(count)
  character(len=*),intent(in) :: record
  integer :: index
  logical :: in_token
  count=0;in_token=.false.
  do index=1,len_trim(record)
   if(record(index:index)==' '.or.iachar(record(index:index))==9) then
    in_token=.false.
   else if(.not.in_token) then
    count=count+1;in_token=.true.
   endif
  enddo
 end function

 pure logical function v51_numeric_record_syntax(record,integer_only) result(valid)
  ! V51 records deliberately use a small, host-emitted numeric grammar.  In
  ! particular, reject list-directed repeat counts, comma separators and '/'
  ! terminators that could otherwise make trailing text invisible to READ.
  character(len=*),intent(in) :: record
  logical,intent(in) :: integer_only
  integer :: position
  character :: value
  valid=.true.
  do position=1,len_trim(record)
   value=record(position:position)
   if(value==' '.or.iachar(value)==9) cycle
   if(index('0123456789+-',value)/=0) cycle
   if(.not.integer_only.and.index('.eEdD',value)/=0) cycle
   valid=.false.;return
  enddo
 end function

 pure integer function v52_hex_nibble(value) result(nibble)
  character,intent(in) :: value
  integer :: code
  code=iachar(value)
  if(code>=iachar('0').and.code<=iachar('9')) then
   nibble=code-iachar('0')
  else if(code>=iachar('a').and.code<=iachar('f')) then
   nibble=code-iachar('a')+10
  else
   nibble=-1
  endif
 end function

 pure logical function v52_lower_hex(record) result(valid)
  character(len=*),intent(in) :: record
  integer :: position
  valid=len(record)>0.and.mod(len(record),2)==0
  if(.not.valid) return
  do position=1,len(record)
   if(v52_hex_nibble(record(position:position))<0) then
    valid=.false.;return
   endif
  enddo
 end function

 subroutine v52_decode_hex(record,decoded,valid)
  character(len=*),intent(in) :: record
  character(len=*),intent(out) :: decoded
  logical,intent(out) :: valid
  integer :: position,high,low
  decoded='';valid=v52_lower_hex(record).and.len(record)==2*len(decoded)
  if(.not.valid) return
  do position=1,len(decoded)
   high=v52_hex_nibble(record(2*position-1:2*position-1))
   low=v52_hex_nibble(record(2*position:2*position))
   decoded(position:position)=achar(16*high+low)
  enddo
 end subroutine

 pure logical function v52_canonical_json_number(token) result(valid)
  ! Exact JSON spellings emitted by CPython's finite-float repr are a strict
  ! subset of this grammar.  The additional shortest-form checks reject the
  ! alternate spellings relevant to this wire (leading/trailing zeroes,
  ! uppercase exponents, missing exponent signs and unnormalised exponents).
  character(len=*),intent(in) :: token
  integer :: n,p,start_integer,start_fraction,start_exponent,exponent_value,ios
  logical :: has_fraction,has_exponent
  character(len=64) :: exponent_text
  n=len_trim(token);valid=.false.
  if(n<3.or.n>64) return
  if(n<len(token)) then
   if(any([(token(p:p)/=' ',p=n+1,len(token))])) return
  endif
  p=1
  if(token(p:p)=='-') then
   p=p+1;if(p>n) return
  endif
  start_integer=p
  if(token(p:p)=='0') then
   p=p+1
   if(p<=n) then
    if(index('0123456789',token(p:p))/=0) return
   endif
  else
   if(index('123456789',token(p:p))==0) return
   do while(p<=n)
    if(index('0123456789',token(p:p))==0) exit
    p=p+1
   enddo
  endif
  has_fraction=.false.;start_fraction=0
  if(p<=n) then
   if(token(p:p)=='.') then
    has_fraction=.true.;p=p+1;start_fraction=p
    if(p>n) return
    if(index('0123456789',token(p:p))==0) return
    do while(p<=n)
     if(index('0123456789',token(p:p))==0) exit
     p=p+1
    enddo
    if(token(p-1:p-1)=='0'.and.p-start_fraction>1) return
   endif
  endif
  has_exponent=.false.;start_exponent=0
  if(p<=n) then
   if(token(p:p)=='e') then
    has_exponent=.true.;p=p+1
    if(p>n) return
    if(token(p:p)/='+'.and.token(p:p)/='-') return
    p=p+1;start_exponent=p
    if(n-p+1<2) return
    do while(p<=n)
     if(index('0123456789',token(p:p))==0) exit
     p=p+1
    enddo
    if(p<=n) return
    if(n-start_exponent+1>2) then
     if(token(start_exponent:start_exponent)=='0') return
    endif
    exponent_text='';exponent_text=token(start_exponent:n)
    read(exponent_text,*,iostat=ios) exponent_value
    if(ios/=0) return
    if(token(start_exponent-1:start_exponent-1)=='-') then
     if(exponent_value<5) return
    else
     if(exponent_value<16) return
    endif
    ! Exponential float repr is normalized to one integer digit.
    if(has_fraction) then
     if(index(token(start_integer:n),'.')/=2) return
    else
     if(index(token(start_integer:n),'e')/=2) return
    endif
   endif
  endif
  if(p<=n.or.(.not.has_fraction.and..not.has_exponent)) return
  valid=.true.
 end function

 pure logical function v52_canonical_json_string(token) result(valid)
  character(len=*),intent(in) :: token
  integer :: n,p,q,code,low_code
  character :: escaped
  n=len_trim(token);valid=.false.
  if(n<2) return
  if(token(1:1)/='"'.or.token(n:n)/='"') return
  if(n<len(token)) then
   if(any([(token(p:p)/=' ',p=n+1,len(token))])) return
  endif
  p=2
  do while(p<n)
   code=iachar(token(p:p))
   if(token(p:p)==achar(92)) then
    if(p+1>=n) return
    escaped=token(p+1:p+1)
    if(escaped=='"'.or.escaped==achar(92).or.escaped=='b'.or. &
       escaped=='f'.or.escaped=='n'.or.escaped=='r'.or.escaped=='t') then
     p=p+2;cycle
    endif
    if(escaped/='u'.or.p+5>=n) return
    code=0
    do q=p+2,p+5
     if(v52_hex_nibble(token(q:q))<0) return
     code=16*code+v52_hex_nibble(token(q:q))
    enddo
    ! Printable ASCII must be literal.  The five named controls use their
    ! two-byte escape; all other controls use lowercase \u00xx.
    if(code>=32.and.code<=126) return
    if(code==8.or.code==9.or.code==10.or.code==12.or.code==13) return
    if(code<32.and.(token(p+2:p+3)/='00')) return
    if(code>=55296.and.code<=56319) then
    if(p+11>=n) return
    if(token(p+6:p+6)/=achar(92).or.token(p+7:p+7)/='u') return
     low_code=0
     do q=p+8,p+11
      if(v52_hex_nibble(token(q:q))<0) return
      low_code=16*low_code+v52_hex_nibble(token(q:q))
     enddo
     if(low_code<56320.or.low_code>57343) return
     p=p+12;cycle
    else if(code>=56320.and.code<=57343) then
     return
    endif
    p=p+6
   else
    if(code<32.or.code>126.or.token(p:p)=='"') return
    p=p+1
   endif
  enddo
  valid=p==n
 end function

 pure logical function v52_canonical_json_integer(token) result(valid)
  character(len=*),intent(in) :: token
  integer :: n,p
  logical :: negative
  n=len(token);valid=.false.;p=1;negative=.false.
  if(n<1) return
  if(token(p:p)=='-') then
   negative=.true.;p=p+1
   if(p>n) return
  endif
  if(token(p:p)=='0') then
   ! Python never emits a negative integer zero and never emits leading zeroes.
   if(negative.or.p/=n) return
   valid=.true.;return
  endif
  if(index('123456789',token(p:p))==0) return
  p=p+1
  do while(p<=n)
   if(index('0123456789',token(p:p))==0) return
   p=p+1
  enddo
  valid=.true.
 end function

 pure logical function v52_ascii_strict_less(left,right) result(less)
  character(len=*),intent(in) :: left,right
  integer :: position,common,left_code,right_code
  less=.false.;common=min(len(left),len(right))
  do position=1,common
   left_code=iachar(left(position:position));right_code=iachar(right(position:position))
   if(left_code<right_code) then
    less=.true.;return
   else if(left_code>right_code) then
    return
   endif
  enddo
  less=len(left)<len(right)
 end function

 subroutine v52_scan_canonical_json_string(value,position,token,valid)
  character(len=*),intent(in) :: value
  integer,intent(inout) :: position
  character(len=:),allocatable,intent(out) :: token
  logical,intent(inout) :: valid
  integer :: start
  token=''
  if(.not.valid.or.position>len(value).or.value(position:position)/='"') then
   valid=.false.;return
  endif
  start=position;position=position+1
  do while(position<=len(value))
   if(value(position:position)==achar(92)) then
    if(position==len(value)) then
     valid=.false.;return
    endif
    position=position+2
   else if(value(position:position)=='"') then
    token=value(start:position);position=position+1
    valid=v52_canonical_json_string(token)
    return
   else
    position=position+1
   endif
  enddo
  valid=.false.
 end subroutine

 recursive subroutine v52_parse_canonical_json_value(value,position,depth,valid)
  character(len=*),intent(in) :: value
  integer,intent(inout) :: position
  integer,intent(in) :: depth
  logical,intent(inout) :: valid
  integer :: start
  character(len=:),allocatable :: token,key,previous_key
  if(.not.valid) return
  if(position>len(value).or.depth>64) then
   valid=.false.;return
  endif
  select case(value(position:position))
  case('{')
   position=position+1
   if(position<=len(value)) then
    if(value(position:position)=='}') then
     position=position+1;return
    endif
   endif
   do
    call v52_scan_canonical_json_string(value,position,key,valid)
    if(.not.valid) return
    ! Every schema-v2 object key is printable ASCII.  Rejecting escaped keys
    ! makes encoded-key ordering identical to Python's Unicode key ordering.
    if(index(key,achar(92))/=0) then
     valid=.false.;return
    endif
    if(allocated(previous_key)) then
     if(.not.v52_ascii_strict_less(previous_key,key)) then
      valid=.false.;return
     endif
    endif
    previous_key=key
    if(position>len(value)) then
     valid=.false.;return
    endif
    if(value(position:position)/=':') then
     valid=.false.;return
    endif
    position=position+1
    call v52_parse_canonical_json_value(value,position,depth+1,valid)
    if(.not.valid.or.position>len(value)) then
     valid=.false.;return
    endif
    if(value(position:position)=='}') then
     position=position+1;return
    endif
    if(value(position:position)/=',') then
     valid=.false.;return
    endif
    position=position+1
   enddo
  case('[')
   position=position+1
   if(position<=len(value)) then
    if(value(position:position)==']') then
     position=position+1;return
    endif
   endif
   do
    call v52_parse_canonical_json_value(value,position,depth+1,valid)
    if(.not.valid.or.position>len(value)) then
     valid=.false.;return
    endif
    if(value(position:position)==']') then
     position=position+1;return
    endif
    if(value(position:position)/=',') then
     valid=.false.;return
    endif
    position=position+1
   enddo
  case('"')
   call v52_scan_canonical_json_string(value,position,token,valid)
  case('t')
   if(position+3>len(value)) then
    valid=.false.;return
   endif
   if(value(position:position+3)/='true') then
    valid=.false.;return
   endif
   position=position+4
  case('f')
   if(position+4>len(value)) then
    valid=.false.;return
   endif
   if(value(position:position+4)/='false') then
    valid=.false.;return
   endif
   position=position+5
  case('n')
   if(position+3>len(value)) then
    valid=.false.;return
   endif
   if(value(position:position+3)/='null') then
    valid=.false.;return
   endif
   position=position+4
  case default
   start=position
   do while(position<=len(value))
    if(index(',]}',value(position:position))/=0) exit
    position=position+1
   enddo
   if(position==start) then
    valid=.false.;return
   endif
   token=value(start:position-1)
   valid=v52_canonical_json_integer(token).or.v52_canonical_json_number(token)
  end select
 end subroutine

 logical function v52_canonical_json_document(value) result(valid)
  character(len=*),intent(in) :: value
  integer :: position
  position=1;valid=.true.
  call v52_parse_canonical_json_value(value,position,0,valid)
  valid=valid.and.position==len(value)+1
 end function

 pure integer(int64) function sha256_u32(value) result(masked)
  integer(int64),intent(in) :: value
  masked=iand(value,int(z'ffffffff',int64))
 end function

 pure integer(int64) function sha256_rotr(value,amount) result(rotated)
  integer(int64),intent(in) :: value
  integer,intent(in) :: amount
  integer(int64) :: word
  word=sha256_u32(value)
  rotated=sha256_u32(ior(shiftr(word,amount),shiftl(word,32-amount)))
 end function

 subroutine sha256_octets(octets,input_length,digest)
  integer(int64),intent(in) :: octets(:)
  integer,intent(in) :: input_length
  character(len=64),intent(out) :: digest
  integer(int64),parameter :: round_constant(0:63)=[integer(int64) :: &
   int(z'428a2f98',int64),int(z'71374491',int64),int(z'b5c0fbcf',int64),int(z'e9b5dba5',int64), &
   int(z'3956c25b',int64),int(z'59f111f1',int64),int(z'923f82a4',int64),int(z'ab1c5ed5',int64), &
   int(z'd807aa98',int64),int(z'12835b01',int64),int(z'243185be',int64),int(z'550c7dc3',int64), &
   int(z'72be5d74',int64),int(z'80deb1fe',int64),int(z'9bdc06a7',int64),int(z'c19bf174',int64), &
   int(z'e49b69c1',int64),int(z'efbe4786',int64),int(z'0fc19dc6',int64),int(z'240ca1cc',int64), &
   int(z'2de92c6f',int64),int(z'4a7484aa',int64),int(z'5cb0a9dc',int64),int(z'76f988da',int64), &
   int(z'983e5152',int64),int(z'a831c66d',int64),int(z'b00327c8',int64),int(z'bf597fc7',int64), &
   int(z'c6e00bf3',int64),int(z'd5a79147',int64),int(z'06ca6351',int64),int(z'14292967',int64), &
   int(z'27b70a85',int64),int(z'2e1b2138',int64),int(z'4d2c6dfc',int64),int(z'53380d13',int64), &
   int(z'650a7354',int64),int(z'766a0abb',int64),int(z'81c2c92e',int64),int(z'92722c85',int64), &
   int(z'a2bfe8a1',int64),int(z'a81a664b',int64),int(z'c24b8b70',int64),int(z'c76c51a3',int64), &
   int(z'd192e819',int64),int(z'd6990624',int64),int(z'f40e3585',int64),int(z'106aa070',int64), &
   int(z'19a4c116',int64),int(z'1e376c08',int64),int(z'2748774c',int64),int(z'34b0bcb5',int64), &
   int(z'391c0cb3',int64),int(z'4ed8aa4a',int64),int(z'5b9cca4f',int64),int(z'682e6ff3',int64), &
   int(z'748f82ee',int64),int(z'78a5636f',int64),int(z'84c87814',int64),int(z'8cc70208',int64), &
   int(z'90befffa',int64),int(z'a4506ceb',int64),int(z'bef9a3f7',int64),int(z'c67178f2',int64)]
  integer(int64) :: state(0:7),work(0:63),a,b,c,d,e,f,g,h,choice,majority, &
   sigma0,sigma1,temp1,temp2,bit_length
  integer(int64),allocatable :: bytes(:)
  integer :: total_length,block,index,j,nibble
  character(len=16),parameter :: hexadecimal='0123456789abcdef'
  if(input_length<0.or.size(octets)<input_length.or. &
     (input_length>0.and.any(octets(:input_length)<0_int64.or. &
      octets(:input_length)>255_int64))) &
   error stop 'invalid SHA-256 octet input'
  total_length=((input_length+9+63)/64)*64
  allocate(bytes(total_length));bytes=0_int64
  if(input_length>0) bytes(:input_length)=octets(:input_length)
  bytes(input_length+1)=128_int64
  bit_length=int(input_length,int64)*8_int64
  do j=0,7
   bytes(total_length-j)=iand(shiftr(bit_length,8*j),255_int64)
  enddo
  state=[integer(int64) :: int(z'6a09e667',int64),int(z'bb67ae85',int64), &
   int(z'3c6ef372',int64),int(z'a54ff53a',int64),int(z'510e527f',int64), &
   int(z'9b05688c',int64),int(z'1f83d9ab',int64),int(z'5be0cd19',int64)]
  do block=0,total_length/64-1
   do j=0,15
    index=block*64+j*4+1
    work(j)=shiftl(bytes(index),24)+shiftl(bytes(index+1),16)+ &
     shiftl(bytes(index+2),8)+bytes(index+3)
   enddo
   do j=16,63
    sigma0=ieor(ieor(sha256_rotr(work(j-15),7),sha256_rotr(work(j-15),18)), &
     shiftr(work(j-15),3))
    sigma1=ieor(ieor(sha256_rotr(work(j-2),17),sha256_rotr(work(j-2),19)), &
     shiftr(work(j-2),10))
    work(j)=sha256_u32(work(j-16)+sigma0+work(j-7)+sigma1)
   enddo
   a=state(0);b=state(1);c=state(2);d=state(3)
   e=state(4);f=state(5);g=state(6);h=state(7)
   do j=0,63
    sigma1=ieor(ieor(sha256_rotr(e,6),sha256_rotr(e,11)),sha256_rotr(e,25))
    choice=ieor(iand(e,f),iand(not(e),g))
    temp1=sha256_u32(h+sigma1+choice+round_constant(j)+work(j))
    sigma0=ieor(ieor(sha256_rotr(a,2),sha256_rotr(a,13)),sha256_rotr(a,22))
    majority=ieor(ieor(iand(a,b),iand(a,c)),iand(b,c))
    temp2=sha256_u32(sigma0+majority)
    h=g;g=f;f=e;e=sha256_u32(d+temp1)
    d=c;c=b;b=a;a=sha256_u32(temp1+temp2)
   enddo
   state=[sha256_u32(state(0)+a),sha256_u32(state(1)+b), &
    sha256_u32(state(2)+c),sha256_u32(state(3)+d), &
    sha256_u32(state(4)+e),sha256_u32(state(5)+f), &
    sha256_u32(state(6)+g),sha256_u32(state(7)+h)]
  enddo
  digest=''
  do index=0,7
   do j=0,7
    nibble=int(iand(shiftr(state(index),4*(7-j)),15_int64))
    digest(index*8+j+1:index*8+j+1)=hexadecimal(nibble+1:nibble+1)
   enddo
  enddo
  deallocate(bytes)
 end subroutine

 subroutine sha256_ascii(value,digest)
  character(len=*),intent(in) :: value
  character(len=64),intent(out) :: digest
  integer(int64),allocatable :: octets(:)
  integer :: index
  allocate(octets(len(value)))
  do index=1,len(value)
   octets(index)=int(iachar(value(index:index)),int64)
  enddo
  call sha256_octets(octets,len(value),digest)
  deallocate(octets)
 end subroutine

 subroutine sha256_real64_little_endian(values,digest)
  real(8),intent(in) :: values(:)
  character(len=64),intent(out) :: digest
  integer(int64),allocatable :: octets(:)
  integer(int64) :: bits
  integer :: value_index,byte_index,position
  allocate(octets(8*size(values)))
  position=1
  do value_index=1,size(values)
   bits=transfer(values(value_index),bits)
   do byte_index=0,7
    octets(position)=iand(shiftr(bits,8*byte_index),255_int64)
    position=position+1
   enddo
  enddo
  call sha256_octets(octets,size(octets),digest)
  deallocate(octets)
 end subroutine

 pure integer function ascii_occurrences(value,needle) result(count)
  character(len=*),intent(in) :: value,needle
  integer :: position,found
  count=0;position=1
  if(len(needle)==0) return
  do while(position<=len(value)-len(needle)+1)
   found=index(value(position:),needle)
   if(found==0) exit
   count=count+1;position=position+found+len(needle)-1
  enddo
 end function

 subroutine canonical_json_string_field(document,key,value,valid)
  character(len=*),intent(in) :: document,key
  character(len=*),intent(out) :: value
  logical,intent(out) :: valid
  character(len=:),allocatable :: needle
  integer :: first,last
  value='';valid=.false.
  needle='"'//trim(key)//'":"'
  if(ascii_occurrences(document,needle)/=1) return
  first=index(document,needle)+len(needle)
  if(first<1.or.first>len(document)) return
  last=index(document(first:),'"')
  if(last<=1) return
  last=first+last-2
  if(last-first+1>len(value).or.index(document(first:last),'\')/=0) return
  if(last+2>len(document).or.document(last+1:last+1)/='"'.or. &
     index(',}',document(last+2:last+2))==0) return
  value=document(first:last);valid=.true.
 end subroutine

 subroutine canonical_json_number_field(document,key,value,token,valid)
  character(len=*),intent(in) :: document,key
  real(8),intent(out) :: value
  character(len=*),intent(out) :: token
  logical,intent(out) :: valid
  character(len=:),allocatable :: needle
  integer :: first,last,ios
  value=0.d0;token='';valid=.false.
  needle='"'//trim(key)//'":'
  if(ascii_occurrences(document,needle)/=1) return
  first=index(document,needle)+len(needle);last=first
  do while(last<=len(document).and. &
           index('-+0123456789.eE',document(last:last))/=0)
   last=last+1
  enddo
  if(last<=first.or.last>len(document).or. &
     index(',}]',document(last:last))==0.or.last-first>len(token)) return
  token=document(first:last-1)
  if(.not.(v52_canonical_json_number(trim(token)).or. &
           v52_canonical_json_integer(trim(token)))) return
  read(token,*,iostat=ios) value
  valid=ios==0.and.ieee_is_finite(value)
 end subroutine

 subroutine canonical_json_integer_field(document,key,value,valid)
  character(len=*),intent(in) :: document,key
  integer,intent(out) :: value
  logical,intent(out) :: valid
  character(len=128) :: token
  real(8) :: numeric
  integer :: ios
  call canonical_json_number_field(document,key,numeric,token,valid)
  if(.not.valid.or..not.v52_canonical_json_integer(trim(token))) then
   valid=.false.;return
  endif
  read(token,*,iostat=ios) value
  valid=ios==0.and.real(value,8)==numeric
 end subroutine

 subroutine canonical_json_boolean_field(document,key,value,valid)
  character(len=*),intent(in) :: document,key
  logical,intent(out) :: value,valid
  character(len=:),allocatable :: needle
  integer :: first
  value=.false.;valid=.false.
  needle='"'//trim(key)//'":'
  if(ascii_occurrences(document,needle)/=1) return
  first=index(document,needle)+len(needle)
  if(first+3<=len(document).and.document(first:first+3)=='true') then
   if(first+4<=len(document).and.index(',}',document(first+4:first+4))/=0) then
    value=.true.;valid=.true.
   endif
  else if(first+4<=len(document).and.document(first:first+4)=='false') then
   if(first+5<=len(document).and.index(',}',document(first+5:first+5))/=0) &
    valid=.true.
  endif
 end subroutine

 subroutine canonical_json_document_fingerprint(document,fingerprint,valid)
  character(len=*),intent(in) :: document
  character(len=64),intent(out) :: fingerprint
  logical,intent(out) :: valid
  character(len=*),parameter :: needle=',"fingerprint":"'
  character(len=:),allocatable :: body
  character(len=64) :: computed
  integer :: first,digest_first,digest_last
  fingerprint='';valid=.false.
  if(ascii_occurrences(document,'"fingerprint":')/=1.or. &
     ascii_occurrences(document,needle)/=1) return
  first=index(document,needle);digest_first=first+len(needle)
  digest_last=digest_first+63
  if(first<=1.or.digest_last+1>len(document).or. &
     document(digest_last+1:digest_last+1)/='"'.or. &
     verify(document(digest_first:digest_last),'0123456789abcdef')/=0) return
  body=document(:first-1)//document(digest_last+2:)
  if(.not.canonical_json_has_no_external_space(body).or. &
     .not.v52_canonical_json_document(body)) return
  call sha256_ascii(body,computed)
  fingerprint=document(digest_first:digest_last)
  valid=computed==fingerprint
 end subroutine

 subroutine validate_capacity_foundation_forcing_identity(outer_envelope_version)
  integer,intent(in) :: outer_envelope_version
  character(len=128) :: text_value,roughness_method,expected_method, &
   expected_mode,expected_parameter,expected_units,shear_method,shear_mode, &
   execution_forcing_fingerprint,execution_profile_fingerprint
  character(len=64) :: raw_digest
  character(len=128) :: numeric_token,shape_fragment
  integer :: integer_value,formula_code
  real(8) :: numeric_value,forcing_minimum,forcing_maximum
  logical :: valid,boolean_value
  if(.not.capacity_foundation_forcing_identity_declaration) return
  if((outer_envelope_version/=54.and.outer_envelope_version/=55.and. &
      outer_envelope_version/=56).or. &
     (outer_envelope_version==55.and..not.hiding_foundation_identity_declaration).or. &
     (outer_envelope_version==56.and. &
      (.not.secondary_foundation_identity_declaration.or. &
       all(secondary_foundation_base_envelope/=[54,55]))).or. &
     .not.capacity_foundation_runtime_declaration.or. &
     .not.capacity_foundation_route_identity_declaration.or. &
     .not.component_native_identity_declaration) &
   error stop 'v54 Q3-C forcing identity lacks its authenticated runtime chain'
  if(ascii_occurrences(capacity_foundation_forcing_preimage,'":')/=31.or. &
     ascii_occurrences(capacity_foundation_execution_preimage,'":')/=9) &
   error stop 'v54 Q3-C forcing/execution document inventory differs'

  call canonical_json_integer_field(capacity_foundation_forcing_preimage, &
   'schema',integer_value,valid)
  if(.not.valid.or.integer_value/=2) &
   error stop 'v54 forcing schema differs'
  call canonical_json_integer_field(capacity_foundation_forcing_preimage, &
   'native_hydraulic_roughness_mode',integer_value,valid)
  if(.not.valid.or.integer_value/=hydraulic_roughness_mode) &
   error stop 'v54 forcing hydraulic roughness mode differs'
  call canonical_json_integer_field(capacity_foundation_forcing_preimage, &
   'native_sediment_shear_mode',integer_value,valid)
  if(.not.valid.or.integer_value/=sediment_shear_mode) &
   error stop 'v54 forcing sediment shear mode differs'
  call canonical_json_boolean_field(capacity_foundation_forcing_preimage, &
   'white_colebrook_chzk2d',boolean_value,valid)
  if(.not.valid.or.boolean_value.neqv.white_colebrook_chzk2d) &
   error stop 'v54 forcing White-Colebrook branch differs'
  call canonical_json_boolean_field(capacity_foundation_forcing_preimage, &
   'hydraulic_friction_unchanged_by_skin',boolean_value,valid)
  if(.not.valid.or..not.boolean_value) &
   error stop 'v54 forcing hydraulic-friction ownership differs'
  call canonical_json_boolean_field(capacity_foundation_forcing_preimage, &
   'skin_shear_controls_active',boolean_value,valid)
  if(.not.valid.or.boolean_value.neqv.(sediment_shear_mode==1)) &
   error stop 'v54 forcing skin-shear activation differs'

  select case(hydraulic_roughness_mode)
  case(braids_roughness_chezy)
   expected_method='roughness.chezy';expected_mode='chezy'
   expected_parameter='chezy_C';expected_units='m^0.5/s'
  case(braids_roughness_manning)
   expected_method='roughness.manning';expected_mode='manning'
   expected_parameter='manning_n';expected_units='s/m^(1/3)'
  case(braids_roughness_white_colebrook)
   expected_method=merge('roughness.white_colebrook_waqua', &
    'roughness.white_colebrook      ',white_colebrook_chzk2d)
   expected_mode='white_colebrook';expected_parameter='nikuradse_ks'
   expected_units='m'
  case default
   error stop 'v54 forcing hydraulic roughness mode is unsupported'
  end select
  call canonical_json_string_field(capacity_foundation_forcing_preimage, &
   'roughness_method_id',roughness_method,valid)
  if(.not.valid.or.trim(roughness_method)/=trim(expected_method)) &
   error stop 'v54 forcing roughness method differs'
  call canonical_json_string_field(capacity_foundation_forcing_preimage, &
   'hydraulic_roughness_mode',text_value,valid)
  if(.not.valid.or.trim(text_value)/=trim(expected_mode)) &
   error stop 'v54 forcing roughness mode name differs'
  call canonical_json_string_field(capacity_foundation_forcing_preimage, &
   'roughness_parameter_name',text_value,valid)
  if(.not.valid.or.trim(text_value)/=trim(expected_parameter)) &
   error stop 'v54 forcing roughness parameter name differs'
  call canonical_json_string_field(capacity_foundation_forcing_preimage, &
   'roughness_parameter_units',text_value,valid)
  if(.not.valid.or.trim(text_value)/=trim(expected_units)) &
   error stop 'v54 forcing roughness parameter units differ'
  call canonical_json_string_field(capacity_foundation_forcing_preimage, &
   'roughness_storage_field',text_value,valid)
  if(.not.valid.or.trim(text_value)/='manning_cell') &
   error stop 'v54 forcing roughness storage differs'
  call canonical_json_string_field(capacity_foundation_forcing_preimage, &
   'dtype',text_value,valid)
  if(.not.valid.or.trim(text_value)/='<f8') &
   error stop 'v54 forcing raw roughness dtype differs'
  write(shape_fragment,'(a,i0,a)') '"shape":[',nc,']'
  if(ascii_occurrences(capacity_foundation_forcing_preimage, &
      trim(shape_fragment))/=1) &
   error stop 'v54 forcing raw roughness shape differs'
  call sha256_real64_little_endian(manning_cell,raw_digest)
  if(trim(raw_digest)/=trim(capacity_foundation_forcing_raw_sha256)) then
   write(*,'(a,a)') 'computed v54 raw FP64 SHA-256: ',trim(raw_digest)
   write(*,'(a,a)') 'expected v54 raw FP64 SHA-256: ', &
    trim(capacity_foundation_forcing_raw_sha256)
   error stop 'v54 forcing raw FP64 roughness fingerprint differs'
  endif

  call canonical_json_number_field(capacity_foundation_forcing_preimage, &
   'roughness_default_parameter',numeric_value,numeric_token,valid)
  if(.not.valid.or.numeric_value/=manning) &
   error stop 'v54 forcing default roughness differs'
  call canonical_json_number_field(capacity_foundation_forcing_preimage, &
   'roughness_parameter_minimum',forcing_minimum,numeric_token,valid)
  if(.not.valid.or.forcing_minimum/=minval(manning_cell)) &
   error stop 'v54 forcing minimum roughness differs'
  call canonical_json_number_field(capacity_foundation_forcing_preimage, &
   'roughness_parameter_maximum',forcing_maximum,numeric_token,valid)
  if(.not.valid.or.forcing_maximum/=maxval(manning_cell)) &
   error stop 'v54 forcing maximum roughness differs'
  call canonical_json_number_field(capacity_foundation_forcing_preimage, &
   'roughness_minimum_work_depth_m',numeric_value,numeric_token,valid)
  if(.not.valid.or.numeric_value/=roughness_minimum_work_depth) &
   error stop 'v54 forcing roughness work depth differs'
  call canonical_json_number_field(capacity_foundation_forcing_preimage, &
   'friction_minimum_work_depth_m',numeric_value,numeric_token,valid)
  if(.not.valid.or.numeric_value/=manning_friction_chezy_minimum_depth) &
   error stop 'v54 forcing friction work depth differs'
  call canonical_json_number_field(capacity_foundation_forcing_preimage, &
   'effective_minimum_work_depth_m',numeric_value,numeric_token,valid)
  if(.not.valid.or.numeric_value/=max(roughness_minimum_work_depth, &
      manning_friction_chezy_minimum_depth)) &
   error stop 'v54 forcing effective work depth differs'

  if(sediment_shear_mode==1) then
   shear_method='shear.soulsby_clarke_skin';shear_mode='soulsby_clarke_skin'
  else
   shear_method='shear.total_roughness';shear_mode='total'
  endif
  call canonical_json_string_field(capacity_foundation_forcing_preimage, &
   'sediment_shear_method_id',text_value,valid)
  if(.not.valid.or.trim(text_value)/=trim(shear_method)) &
   error stop 'v54 forcing sediment shear method differs'
  call canonical_json_string_field(capacity_foundation_forcing_preimage, &
   'sediment_shear_mode',text_value,valid)
  if(.not.valid.or.trim(text_value)/=trim(shear_mode)) &
   error stop 'v54 forcing sediment shear mode name differs'
  call canonical_json_number_field(capacity_foundation_forcing_preimage, &
   'kinematic_viscosity_m2_s',numeric_value,numeric_token,valid)
  if(.not.valid.or.numeric_value/=skin_kinematic_viscosity) &
   error stop 'v54 forcing skin viscosity differs'
  call canonical_json_number_field(capacity_foundation_forcing_preimage, &
   'sand_roughness_m',numeric_value,numeric_token,valid)
  if(.not.valid.or.numeric_value/=skin_sand_roughness) &
   error stop 'v54 forcing skin sand roughness differs'
  call canonical_json_number_field(capacity_foundation_forcing_preimage, &
   'silt_roughness_m',numeric_value,numeric_token,valid)
  if(.not.valid.or.numeric_value/=skin_silt_roughness) &
   error stop 'v54 forcing skin silt roughness differs'
  call canonical_json_number_field(capacity_foundation_forcing_preimage, &
   'critical_mud_fraction_low',numeric_value,numeric_token,valid)
  if(.not.valid.or.numeric_value/=skin_critical_mud_fraction_low) &
   error stop 'v54 forcing skin mud threshold differs'
  call canonical_json_number_field(capacity_foundation_forcing_preimage, &
   'critical_mud_fraction_high',numeric_value,numeric_token,valid)
  if(.not.valid.or.numeric_value/=skin_critical_mud_fraction_high) &
   error stop 'v54 forcing skin mud threshold differs'
  call canonical_json_number_field(capacity_foundation_forcing_preimage, &
   'fixed_fluff_fraction',numeric_value,numeric_token,valid)
  if(.not.valid.or.numeric_value/=skin_fluff_fraction) &
   error stop 'v54 forcing skin fluff fraction differs'

  call canonical_json_integer_field(capacity_foundation_execution_preimage, &
   'schema',integer_value,valid)
  if(.not.valid.or.integer_value/=1) &
   error stop 'v54 capacity execution schema differs'
  call canonical_json_integer_field(capacity_foundation_execution_preimage, &
   'formula_code',formula_code,valid)
  if(.not.valid.or.formula_code/=bedlaw) &
   error stop 'v54 capacity execution formula differs'
  select case(bedlaw)
  case(2);expected_method='capacity.mpm'
  case(8);expected_method='capacity.engelund_hansen'
  case(11);expected_method='capacity.general'
  case default;error stop 'v54 capacity execution formula is unsupported'
  end select
  call canonical_json_string_field(capacity_foundation_execution_preimage, &
   'method_id',text_value,valid)
  if(.not.valid.or.trim(text_value)/=trim(expected_method).or. &
     trim(text_value)/=trim(capacity_route_method_id)) &
   error stop 'v54 capacity execution method differs'
  call canonical_json_string_field(capacity_foundation_execution_preimage, &
   'scope_id',text_value,valid)
  if(.not.valid.or.trim(text_value)/= &
      capacity_foundation_execution_scope_expected) &
   error stop 'v54 capacity execution scope differs'
  call canonical_json_string_field(capacity_foundation_execution_preimage, &
   'semantic_id',text_value,valid)
  if(.not.valid.or.trim(text_value)/= &
      capacity_foundation_execution_schema_expected) &
   error stop 'v54 capacity execution semantic identity differs'
  call canonical_json_string_field(capacity_foundation_execution_preimage, &
   'forcing_fingerprint',execution_forcing_fingerprint,valid)
  if(.not.valid.or.trim(execution_forcing_fingerprint)/= &
      trim(capacity_foundation_forcing_fingerprint)) &
   error stop 'v54 capacity execution forcing fingerprint differs'
  call canonical_json_string_field(capacity_foundation_execution_preimage, &
   'capacity_profile_fingerprint',execution_profile_fingerprint,valid)
  if(.not.valid.or.trim(execution_profile_fingerprint)/= &
      trim(capacity_route_profile_fingerprint)) &
   error stop 'v54 capacity execution profile fingerprint differs'
  call canonical_json_string_field(capacity_foundation_execution_preimage, &
   'roughness_method_id',text_value,valid)
  if(.not.valid.or.trim(text_value)/=trim(roughness_method)) &
   error stop 'v54 capacity execution roughness method differs'
  if(trim(capacity_foundation_native_execution_fingerprint)/= &
       trim(capacity_route_execution_fingerprint)) &
   error stop 'v54 ordered capacity execution fingerprint differs'
 end subroutine

 subroutine validate_hiding_foundation_identity(outer_envelope_version)
  integer,intent(in) :: outer_envelope_version
  character(len=128) :: text_value,expected_method,numeric_token
  character(len=:),allocatable :: fraction_fragment
  integer :: integer_value
  real(8) :: numeric_value
  logical :: valid,boolean_value
  if(.not.hiding_foundation_identity_declaration) return
  if((outer_envelope_version/=55.and.outer_envelope_version/=56).or. &
     (outer_envelope_version==56.and. &
      (.not.secondary_foundation_identity_declaration.or. &
       secondary_foundation_base_envelope/=55)).or. &
     .not.capacity_foundation_forcing_identity_declaration.or. &
     .not.capacity_foundation_runtime_declaration.or. &
     .not.capacity_foundation_route_identity_declaration.or. &
     .not.component_native_identity_declaration) &
   error stop 'v55 Q3-H1 hiding identity lacks its authenticated Q3-C chain'
  if(ascii_occurrences(hiding_foundation_execution_preimage,'":')/=20) &
   error stop 'v55 Q3-H1 hiding document inventory differs'

  call canonical_json_integer_field(hiding_foundation_execution_preimage, &
   'schema',integer_value,valid)
  if(.not.valid.or.integer_value/=1) error stop 'v55 hiding schema differs'
  call canonical_json_string_field(hiding_foundation_execution_preimage, &
   'scope_id',text_value,valid)
  if(.not.valid.or.trim(text_value)/=hiding_foundation_execution_scope_expected) &
   error stop 'v55 hiding scope differs'
  call canonical_json_string_field(hiding_foundation_execution_preimage, &
   'semantic_id',text_value,valid)
  if(.not.valid.or.trim(text_value)/=hiding_foundation_execution_schema_expected) &
   error stop 'v55 hiding semantic identity differs'
  call canonical_json_integer_field(hiding_foundation_execution_preimage, &
   'mode',integer_value,valid)
  if(.not.valid.or.integer_value/=hiding_exposure.or. &
     all(integer_value/=[2,3,4,5])) error stop 'v55 hiding mode differs'
  select case(hiding_exposure)
  case(2);expected_method='hiding.egiazaroff'
  case(3);expected_method='hiding.ashida_michiue'
  case(4);expected_method='hiding.parker_klm'
  case(5);expected_method='hiding.wu_wang_jia'
  case default;error stop 'v55 hiding mode is unsupported'
  end select
  call canonical_json_string_field(hiding_foundation_execution_preimage, &
   'method_id',text_value,valid)
  if(.not.valid.or.trim(text_value)/=trim(expected_method)) &
   error stop 'v55 hiding method differs'
  call canonical_json_integer_field(hiding_foundation_execution_preimage, &
   'capacity_law',integer_value,valid)
  if(.not.valid.or.integer_value/=bedlaw.or.all(bedlaw/=[2,11])) &
   error stop 'v55 hiding capacity law differs'
  call canonical_json_number_field(hiding_foundation_execution_preimage, &
   'parker_alpha',numeric_value,numeric_token,valid)
  if(.not.valid.or.numeric_value/=hiding_parker_alpha) &
   error stop 'v55 Parker hiding control differs'
  call canonical_json_number_field(hiding_foundation_execution_preimage, &
   'wu_exponent',numeric_value,numeric_token,valid)
  if(.not.valid.or.numeric_value/=hiding_wu_exponent) &
   error stop 'v55 Wu hiding control differs'
  call canonical_json_string_field(hiding_foundation_execution_preimage, &
   'capacity_execution_fingerprint',text_value,valid)
  if(.not.valid.or.trim(text_value)/= &
      trim(capacity_foundation_native_execution_fingerprint)) &
   error stop 'v55 nested capacity execution fingerprint differs'
  call canonical_json_string_field(hiding_foundation_execution_preimage, &
   'capacity_profile_fingerprint',text_value,valid)
  if(.not.valid.or.trim(text_value)/=trim(capacity_route_profile_fingerprint)) &
   error stop 'v55 nested capacity profile fingerprint differs'

  if(nfrac/=2.or.channel_composition_fraction_count/=2.or. &
     any(channel_transport_class_input(:2)/=BRAIDS_TRA_BEDLOAD).or. &
     active_layer_thickness<=0.d0.or.any(sediment_type_fraction/=0).or. &
     exchange_mode/=0.or.bed_slope/=0.or.secondary_flow.or. &
     sediment_shear_mode/=0.or.amr_enabled) &
   error stop 'v55 Q3-H1 physical scope differs'
  fraction_fragment='"fraction_ids":['// &
   channel_native_identifier_token(1)(:channel_native_identifier_token_length(1))//','// &
   channel_native_identifier_token(2)(:channel_native_identifier_token_length(2))//']'
  if(ascii_occurrences(hiding_foundation_execution_preimage,fraction_fragment)/=1) &
   error stop 'v55 ordered fraction identities differ'

  call canonical_json_string_field(hiding_foundation_execution_preimage, &
   'active_volume_source',text_value,valid)
  if(.not.valid.or.trim(text_value)/= &
     'transactional_trial_surface_layer_mass_divided_by_bulk_density') &
   error stop 'v55 active-surface ownership differs'
  call canonical_json_string_field(hiding_foundation_execution_preimage, &
   'backend_selection',text_value,valid)
  if(.not.valid.or.trim(text_value)/='explicit_user_choice') &
   error stop 'v55 backend selection differs'
  call canonical_json_boolean_field(hiding_foundation_execution_preimage, &
   'automatic_cpu_fallback',boolean_value,valid)
  if(.not.valid.or.boolean_value) error stop 'v55 CPU fallback contract differs'
  call canonical_json_string_field(hiding_foundation_execution_preimage, &
   'commit_policy',text_value,valid)
  if(.not.valid.or.trim(text_value)/='only_whole_accepted_candidate_commits') &
   error stop 'v55 hiding transaction policy differs'
  call canonical_json_string_field(hiding_foundation_execution_preimage, &
   'exchange_operator',text_value,valid)
  if(.not.valid.or.trim(text_value)/='suspension.off') &
   error stop 'v55 exchange operator differs'
  call canonical_json_string_field(hiding_foundation_execution_preimage, &
   'slope_operator',text_value,valid)
  if(.not.valid.or.trim(text_value)/='bed_slope.off') &
   error stop 'v55 slope operator differs'
  call canonical_json_string_field(hiding_foundation_execution_preimage, &
   'secondary_flow_operator',text_value,valid)
  if(.not.valid.or.trim(text_value)/='secondary.off') &
   error stop 'v55 secondary-flow operator differs'
  if(ascii_occurrences(hiding_foundation_execution_preimage, &
     '"composition_order":["transactional_trial_surface_active_volume",'// &
     '"hiding_exposure_factors_once_per_trial_stage",'// &
     '"cell_and_owner_indexed_boundary_capacity","channel_composition",'// &
     '"face_reconstruction_and_exner"]')/=1) &
   error stop 'v55 hiding composition order differs'
 end subroutine

 subroutine validate_secondary_foundation_identity(outer_envelope_version)
  integer,intent(in) :: outer_envelope_version
  character(len=128) :: text_value,expected_method,numeric_token, &
   forcing_roughness_method
  integer :: integer_value
  real(8) :: numeric_value
  logical :: valid,boolean_value
  if(.not.secondary_foundation_identity_declaration) return
  if(outer_envelope_version/=56.or. &
     all(secondary_foundation_base_envelope/=[54,55]).or. &
     .not.capacity_foundation_forcing_identity_declaration.or. &
     .not.capacity_foundation_runtime_declaration.or. &
     .not.capacity_foundation_route_identity_declaration.or. &
     .not.component_native_identity_declaration.or. &
     (secondary_foundation_base_envelope==55.and. &
      .not.hiding_foundation_identity_declaration).or. &
     (secondary_foundation_base_envelope==54.and. &
      hiding_foundation_identity_declaration)) &
   error stop 'v56 Q3-H2 secondary identity lacks its exact v54/v55 foundation'
  if(ascii_occurrences(secondary_foundation_execution_preimage,'":')/=42) &
   error stop 'v56 Q3-H2 secondary document inventory differs'

  call canonical_json_integer_field(secondary_foundation_execution_preimage, &
   'schema',integer_value,valid)
  if(.not.valid.or.integer_value/=1) error stop 'v56 secondary schema differs'
  call canonical_json_integer_field(secondary_foundation_execution_preimage, &
   'envelope_version',integer_value,valid)
  if(.not.valid.or.integer_value/=56) error stop 'v56 secondary envelope differs'
  call canonical_json_string_field(secondary_foundation_execution_preimage, &
   'scope_id',text_value,valid)
  if(.not.valid.or.trim(text_value)/=secondary_foundation_execution_scope_expected) &
   error stop 'v56 secondary scope differs'
  call canonical_json_string_field(secondary_foundation_execution_preimage, &
   'semantic_id',text_value,valid)
  if(.not.valid.or.trim(text_value)/=secondary_foundation_execution_schema_expected) &
   error stop 'v56 secondary semantic identity differs'
  call canonical_json_string_field(secondary_foundation_execution_preimage, &
   'grid_scope',text_value,valid)
  if(.not.valid.or.trim(text_value)/='fixed_grid') &
   error stop 'v56 secondary grid scope differs'
  call canonical_json_string_field(secondary_foundation_execution_preimage, &
   'method_id',text_value,valid)
  if(.not.valid.or.trim(text_value)/='secondary.equilibrium') &
   error stop 'v56 secondary method differs'
  call canonical_json_integer_field(secondary_foundation_execution_preimage, &
   'model',integer_value,valid)
  if(.not.valid.or.integer_value/=1) error stop 'v56 secondary model differs'
  call canonical_json_string_field(secondary_foundation_execution_preimage, &
   'selector_provenance',text_value,valid)
  if(.not.valid.or.trim(text_value)/='Sub1=I;Equili=Y') &
   error stop 'v56 secondary selector provenance differs'
  call canonical_json_integer_field(secondary_foundation_execution_preimage, &
   'capacity_law',integer_value,valid)
  if(.not.valid.or.integer_value/=bedlaw.or.all(bedlaw/=[2,8,11])) &
   error stop 'v56 secondary capacity law differs'
  select case(bedlaw)
  case(2);expected_method='capacity.mpm'
  case(8);expected_method='capacity.engelund_hansen'
  case(11);expected_method='capacity.general'
  case default;error stop 'v56 secondary capacity law is unsupported'
  end select
  call canonical_json_string_field(secondary_foundation_execution_preimage, &
   'capacity_method_id',text_value,valid)
  if(.not.valid.or.trim(text_value)/=trim(expected_method)) &
   error stop 'v56 secondary capacity method differs'
  call canonical_json_integer_field(secondary_foundation_execution_preimage, &
   'base_envelope_version',integer_value,valid)
  if(.not.valid.or.integer_value/=secondary_foundation_base_envelope) &
   error stop 'v56 secondary base envelope differs'
  call canonical_json_integer_field(secondary_foundation_execution_preimage, &
   'capacity_foundation_envelope_version',integer_value,valid)
  if(.not.valid.or.integer_value/=54) &
   error stop 'v56 secondary capacity foundation envelope differs'
  call canonical_json_string_field(secondary_foundation_execution_preimage, &
   'capacity_execution_fingerprint',text_value,valid)
  if(.not.valid.or.trim(text_value)/= &
      trim(capacity_foundation_native_execution_fingerprint)) &
   error stop 'v56 nested capacity execution fingerprint differs'
  call canonical_json_string_field(secondary_foundation_execution_preimage, &
   'capacity_profile_fingerprint',text_value,valid)
  if(.not.valid.or.trim(text_value)/=trim(capacity_route_profile_fingerprint)) &
   error stop 'v56 nested capacity profile fingerprint differs'
  if(secondary_foundation_base_envelope==55) then
   call canonical_json_integer_field(secondary_foundation_execution_preimage, &
    'hiding_foundation_envelope_version',integer_value,valid)
   if(.not.valid.or.integer_value/=55) &
    error stop 'v56 hiding foundation envelope differs'
   call canonical_json_string_field(secondary_foundation_execution_preimage, &
    'hiding_execution_fingerprint',text_value,valid)
   if(.not.valid.or.trim(text_value)/=trim(hiding_foundation_execution_fingerprint)) &
    error stop 'v56 nested hiding fingerprint differs'
   call canonical_json_string_field(secondary_foundation_execution_preimage, &
    'h1_reference_policy',text_value,valid)
   if(.not.valid.or.trim(text_value)/= &
      'opaque_frozen_v55_reference_secondary.off_not_reinterpreted') &
    error stop 'v56 hiding reference policy differs'
  else
   if(ascii_occurrences(secondary_foundation_execution_preimage, &
      '"hiding_foundation_envelope_version":null')/=1.or. &
      ascii_occurrences(secondary_foundation_execution_preimage, &
      '"hiding_execution_fingerprint":null')/=1.or. &
      ascii_occurrences(secondary_foundation_execution_preimage, &
      '"h1_reference_policy":null')/=1) &
    error stop 'v56 absent hiding foundation is not canonical'
  endif
  call canonical_json_string_field(secondary_foundation_execution_preimage, &
   'base_reference_policy',text_value,valid)
  if(.not.valid.or.trim(text_value)/= &
     'opaque_frozen_foundation_secondary.off_not_reinterpreted') &
   error stop 'v56 base reference policy differs'

  call canonical_json_string_field(capacity_foundation_forcing_preimage, &
   'roughness_method_id',forcing_roughness_method,valid)
  if(.not.valid) error stop 'v56 typed roughness foundation is invalid'
  call canonical_json_string_field(secondary_foundation_execution_preimage, &
   'roughness_method_id',text_value,valid)
  if(.not.valid.or.trim(text_value)/=trim(forcing_roughness_method)) &
   error stop 'v56 typed roughness method differs'
  call canonical_json_string_field(secondary_foundation_execution_preimage, &
   'roughness_fingerprint',text_value,valid)
  if(.not.valid.or.trim(text_value)/=trim(capacity_foundation_forcing_fingerprint)) &
   error stop 'v56 typed roughness fingerprint differs'
  call canonical_json_string_field(secondary_foundation_execution_preimage, &
   'roughness_policy',text_value,valid)
  if(.not.valid.or.trim(text_value)/= &
     'typed_raw_chezy_source_floor_rotation_unfloored_v1') &
   error stop 'v56 typed roughness policy differs'

  call canonical_json_string_field(secondary_foundation_execution_preimage, &
   'coriolis_disposition',text_value,valid)
  if(.not.valid.or.trim(text_value)/='zero') &
   error stop 'v56 first live route requires zero Coriolis disposition'
  call canonical_json_number_field(secondary_foundation_execution_preimage, &
   'coriolis_constant_s_inverse',secondary_foundation_coriolis,numeric_token,valid)
  if(.not.valid.or.secondary_foundation_coriolis/=0.d0) &
   error stop 'v56 first live route requires zero Coriolis'
  call canonical_json_string_field(secondary_foundation_execution_preimage, &
   'coriolis_source',text_value,valid)
  if(.not.valid.or.trim(text_value)/='constant_zero_planar') &
   error stop 'v56 Coriolis source differs'
  call canonical_json_number_field(secondary_foundation_execution_preimage, &
   'rmincf',secondary_foundation_rmincf,numeric_token,valid)
  if(.not.valid.or.secondary_foundation_rmincf<=0.d0) &
   error stop 'v56 Rmincf differs'
  call canonical_json_number_field(secondary_foundation_execution_preimage, &
   'chzmin',secondary_foundation_chzmin,numeric_token,valid)
  if(.not.valid.or.secondary_foundation_chzmin<=0.d0) &
   error stop 'v56 Chzmin differs'
  call canonical_json_number_field(secondary_foundation_execution_preimage, &
   'espir',secondary_foundation_espir,numeric_token,valid)
  if(.not.valid.or.secondary_foundation_espir<0.d0) &
   error stop 'v56 Espir differs'
  call canonical_json_number_field(secondary_foundation_execution_preimage, &
   'betac',secondary_foundation_betac,numeric_token,valid)
  if(.not.valid.or.secondary_foundation_betac/=0.d0) &
   error stop 'v56 first live route requires zero Betac'

  do integer_value=1,9
   select case(integer_value)
   case(1);text_value='braids_fixed_grid_cell_lsq_streamline_curvature_v1'
   case(2);text_value='classic_flow2dh_rmincf_local_metric_cap_v1'
   case(3);text_value='fixed_cartesian_metric_sums_2dx_2dy_v1'
   case(4);text_value='same_transactional_trial_stage_v1'
   case(5);text_value='every_trial_stage_before_raw_bed_rotation'
   case(6);text_value='classic_flow2dh_adjacent_owner_open_boundary_equilibrium_v1'
   case(7);text_value='bedload_only_before_channel_composition'
   case(8);text_value='explicit_user_choice'
   case(9);text_value='separate_exact_route_gate'
   end select
   select case(integer_value)
   case(1);call canonical_json_string_field(secondary_foundation_execution_preimage, &
    'curvature_discretization',forcing_roughness_method,valid)
   case(2);call canonical_json_string_field(secondary_foundation_execution_preimage, &
    'curvature_cap_policy',forcing_roughness_method,valid)
   case(3);call canonical_json_string_field(secondary_foundation_execution_preimage, &
    'metric_policy',forcing_roughness_method,valid)
   case(4);call canonical_json_string_field(secondary_foundation_execution_preimage, &
    'depth_pair_policy',forcing_roughness_method,valid)
   case(5);call canonical_json_string_field(secondary_foundation_execution_preimage, &
    'refresh_timing',forcing_roughness_method,valid)
   case(6);call canonical_json_string_field(secondary_foundation_execution_preimage, &
    'boundary_policy',forcing_roughness_method,valid)
   case(7);call canonical_json_string_field(secondary_foundation_execution_preimage, &
    'raw_bed_rotation_scope',forcing_roughness_method,valid)
   case(8);call canonical_json_string_field(secondary_foundation_execution_preimage, &
    'backend_selection',forcing_roughness_method,valid)
   case(9);call canonical_json_string_field(secondary_foundation_execution_preimage, &
    'runtime_admission_authority',forcing_roughness_method,valid)
   end select
   if(.not.valid.or.trim(forcing_roughness_method)/=trim(text_value)) &
    error stop 'v56 secondary operator policy differs'
  enddo
  call canonical_json_boolean_field(secondary_foundation_execution_preimage, &
   'automatic_cpu_fallback',boolean_value,valid)
  if(.not.valid.or.boolean_value) error stop 'v56 CPU fallback contract differs'
  call canonical_json_boolean_field(secondary_foundation_execution_preimage, &
   'suspended_channel_rotation',boolean_value,valid)
  if(.not.valid.or.boolean_value) error stop 'v56 suspended rotation contract differs'
  call canonical_json_string_field(secondary_foundation_execution_preimage, &
   'downstream_numerics',text_value,valid)
  if(.not.valid.or.trim(text_value)/= &
     'existing_channel_composer_then_unchanged_face_reconstruction_availability_and_exner') &
   error stop 'v56 downstream numerics differ'
  if(ascii_occurrences(secondary_foundation_execution_preimage, &
     '"composition_order":["transactional_trial_hydraulic_state",'// &
     '"classic_flow2dh_equilibrium_refresh",'// &
     '"transactional_trial_surface_active_volume",'// &
     '"optional_hiding_exposure_factors_once_per_trial_stage",'// &
     '"raw_scalar_cell_and_owner_indexed_boundary_capacity",'// &
     '"classic_flow2dh_raw_bed_rotation","channel_composition_once",'// &
     '"face_reconstruction","bed_slope_off",'// &
     '"donor_and_finite_availability_limiting",'// &
     '"conservative_flux_and_exner"]')/=1) &
   error stop 'v56 secondary composition order differs'
  if(exchange_mode/=0.or.bed_slope/=0.or.secondary_flow.or.amr_enabled.or. &
     sediment_shear_mode/=0.or.horizontal_viscosity/=0.d0.or. &
     all(bedlaw/=[2,8,11]).or. &
     bedload_adaptation.or.bedform_roughness) &
   error stop 'v56 Q3-H2 physical scope differs'
 end subroutine

 pure logical function canonical_json_has_no_external_space(value) result(valid)
  character(len=*),intent(in) :: value
  integer :: position,code
  logical :: in_string,escaped
  valid=.false.;in_string=.false.;escaped=.false.
  do position=1,len(value)
   code=iachar(value(position:position))
   if(code<32.or.code>127) return
   if(in_string) then
    if(escaped) then
     escaped=.false.
    else if(value(position:position)==achar(92)) then
     escaped=.true.
    else if(value(position:position)=='"') then
     in_string=.false.
    endif
   else
    if(value(position:position)=='"') then
     in_string=.true.
    else if(value(position:position)==' '.or.iachar(value(position:position))==9) then
     return
    endif
   endif
  enddo
  valid=.not.in_string.and..not.escaped
 end function

 integer function native_component_index(component_id) result(found)
  character(len=*),intent(in) :: component_id
  integer :: item
  found=0
  if(.not.allocated(component_native_documents)) return
  do item=1,component_native_count
   if(trim(component_native_documents(item)%component_id)==component_id) then
    found=item;return
   endif
  enddo
 end function

 subroutine v53_execution_parameter_token(entry_id,fraction_token,token,value,valid)
  ! Extract one exact canonical inline floating-point token.  The token is
  ! reused when reconstructing the capacity component document, which proves
  ! that its coefficients and the selector profile describe the same bytes.
  character(len=*),intent(in) :: entry_id,fraction_token
  character(len=*),intent(out) :: token
  real(8),intent(out) :: value
  logical,intent(out) :: valid
  character(len=:),allocatable :: prefix
  integer :: start,finish,ios
  token='';value=0.d0;valid=.false.
  prefix='{"entry_id":"'//entry_id//'","representation":"inline","scope":'// &
   '{"fraction_id":'//fraction_token//'},"value":'
  if(ascii_occurrences(execution_profile_preimage,prefix)/=1) return
  start=index(execution_profile_preimage,prefix)+len(prefix)
  finish=index(execution_profile_preimage(start:),'}')
  if(finish<=1.or.finish-1>len(token)) return
  token=execution_profile_preimage(start:start+finish-2)
  if(.not.v52_canonical_json_number(trim(token))) return
  read(token,*,iostat=ios) value
  valid=ios==0.and.ieee_is_finite(value)
 end subroutine

 subroutine validate_v53_total_execution_profile(engelund_hansen_declaration, &
  capacity_parameter_declaration)
  ! Authenticate the complete Q3-C scalar-capacity chain.  V53 transports the
  ! exact component preimages; this routine independently reconstructs their
  ! native meaning and admits no defaulted coefficient or auxiliary selector.
  ! The historical seven-method law-11 profile remains readable, while the
  ! new common runtime requires an additional typed C/n/ks method selection.
  integer :: fraction,parameter,item,capacity_item,channel_item,insert_at
  integer :: expected_entry_count,parameter_count,roughness_count
  real(8) :: parameter_value
  logical :: valid,foundation_profile
  logical,intent(in) :: engelund_hansen_declaration,capacity_parameter_declaration
  character(len=32) :: coefficient_keyword(10),parameter_suffix(10)
  character(len=128) :: parameter_token(10,32)
  character(len=64) :: capacity_fingerprint,capacity_document_sha256, &
   channel_document_sha256,capacity_method,parameter_prefix
  character(len=:),allocatable :: capacity_payload,capacity_document, &
   channel_document,fragment,binding,fraction_tail,roughness_fragment

  if(.not.component_native_identity_declaration) return
  if(.not.execution_profile_declaration.or..not.channel_native_identity_declaration) &
   error stop 'v53 requires authenticated nested v52 identities'
  if(exchange_mode/=0.or.nfrac/=channel_composition_fraction_count.or. &
     any(channel_transport_class_input(:nfrac)/=BRAIDS_TRA_BEDLOAD)) &
   error stop 'v53 scalar capacity requires exact algebraic total-load routing'
  if(sediment_shear_mode/=0.or. &
     (hiding_exposure/=0.and..not.hiding_foundation_identity_declaration).or. &
     bed_slope/=0.or.secondary_flow) &
   error stop 'v53 scalar-capacity auxiliary operator chain differs'
  if(component_native_count/=2) &
   error stop 'v53 scalar-capacity component inventory must contain exactly two documents'
  capacity_item=native_component_index('capacity')
  channel_item=native_component_index('channel_composition')
  if(capacity_item==0.or.channel_item==0) &
   error stop 'v53 scalar-capacity component inventory differs'
  if(.not.allocated(capacity_parameter)) &
   error stop 'v53 scalar-capacity parameter storage is absent'

  coefficient_keyword='';parameter_suffix='';parameter_prefix=''
  select case(bedlaw)
  case(braids_capacity_mpm)
   if(.not.capacity_route_identity_declaration.or. &
      trim(capacity_route_method_id)/='capacity.mpm'.or.mpm_friction_mode/=1.or. &
      abs(mpm_critical-.047d0)>1.d-14) &
    error stop 'v53 corrected MPM identity differs from native physics'
   capacity_method='capacity.mpm';parameter_prefix='parameter.capacity.formula_2.'
   parameter_count=1;coefficient_keyword(1)='ACal';parameter_suffix(1)='acal'
  case(braids_capacity_engelund_hansen)
   if(.not.capacity_route_identity_declaration.or. &
      trim(capacity_route_method_id)/='capacity.engelund_hansen'.or. &
      engelund_hansen_declaration) &
    error stop 'v53 corrected Engelund-Hansen identity differs from native physics'
   capacity_method='capacity.engelund_hansen'
   parameter_prefix='parameter.capacity.formula_1.';parameter_count=3
   coefficient_keyword(1:3)=[character(len=32) :: 'ACal','RouKs','SusFac']
   parameter_suffix(1:3)=[character(len=32) :: 'acal','rouks','susfac']
  case(braids_capacity_general)
   if(.not.capacity_parameter_declaration.or.capacity_parameter_law/=11.or. &
      capacity_parameter_count/=10.or..not.capacity_split_retained) &
    error stop 'v53 general capacity differs from its nested v46 declaration'
   capacity_method='capacity.general';parameter_prefix='parameter.capacity.formula_4.'
   parameter_count=10
   coefficient_keyword=[character(len=32) :: 'ACal','PowerB','PowerC','RipFac', &
    'ThetaC','SusACal','SusPowerB','SusPowerC','SusRipFac','SusThetaC']
   parameter_suffix=[character(len=32) :: 'acal','powerb','powerc','ripfac', &
    'thetac','susacal','suspowerb','suspowerc','susripfac','susthetac']
  case default
   error stop 'v53 scalar-capacity profile selects an unsupported law'
  end select

  if(index(execution_profile_preimage,'{"backend":null,"component_profiles":[')/=1.or. &
     ascii_occurrences(execution_profile_preimage,'"backend":')/=1.or. &
     ascii_occurrences(execution_profile_preimage,'"ledger_semantic_id":"'// &
      execution_profile_ledger_expected//'"')/=1.or. &
     ascii_occurrences(execution_profile_preimage,'"legacy_catalog_id":"'// &
      sediment_catalog_expected//'"')/=1) &
   error stop 'v53 execution-profile root or ledger identity differs'

  select case(hydraulic_roughness_mode)
  case(braids_roughness_chezy)
   roughness_fragment='{"entry_id":"roughness.chezy","scope":{"model_id":"model"}}'
  case(braids_roughness_manning)
   roughness_fragment='{"entry_id":"roughness.manning","scope":{"model_id":"model"}}'
  case(braids_roughness_white_colebrook)
   if(white_colebrook_chzk2d) then
    roughness_fragment='{"entry_id":"roughness.white_colebrook_waqua","scope":{"model_id":"model"}}'
   else
    roughness_fragment='{"entry_id":"roughness.white_colebrook","scope":{"model_id":"model"}}'
   endif
  case default
   error stop 'v53 scalar capacity has an unknown typed roughness mode'
  end select
  roughness_count=ascii_occurrences(execution_profile_preimage,roughness_fragment)
  foundation_profile=roughness_count==1
  if(roughness_count>1.or.(any(bedlaw==[braids_capacity_mpm, &
     braids_capacity_engelund_hansen]).and..not.foundation_profile)) &
   error stop 'v53 scalar-capacity typed roughness selector differs'

  ! Every expected method is explicitly scoped.  Exact total counts prevent
  ! an unconsumed selector or coefficient from hiding beside this route.
  do fraction=1,nfrac
   fragment='{"entry_id":"'//trim(capacity_method)//'","scope":{"fraction_id":'// &
    channel_native_identifier_token(fraction)(:channel_native_identifier_token_length(fraction))//'}}'
   if(ascii_occurrences(execution_profile_preimage,fragment)/=1) &
    error stop 'v53 execution-profile capacity selector differs'
   fragment='{"entry_id":"transport_route.total","scope":{"fraction_id":'// &
    channel_native_identifier_token(fraction)(:channel_native_identifier_token_length(fraction))//'}}'
   if(ascii_occurrences(execution_profile_preimage,fragment)/=1) &
    error stop 'v53 execution-profile total route selector differs'
   fragment='{"entry_id":"suspension.off","scope":{"fraction_id":'// &
    channel_native_identifier_token(fraction)(:channel_native_identifier_token_length(fraction))//'}}'
   if(ascii_occurrences(execution_profile_preimage,fragment)/=1) &
    error stop 'v53 execution-profile suspended route selector differs'
  enddo
  do item=1,4
   select case(item)
   case(1);fragment='{"entry_id":"shear.total_roughness","scope":{"model_id":"model"}}'
   case(2);fragment='{"entry_id":"hiding.none","scope":{"model_id":"model"}}'
   case(3);fragment='{"entry_id":"bed_slope.off","scope":{"model_id":"model"}}'
   case default;fragment='{"entry_id":"secondary.off","scope":{"model_id":"model"}}'
   end select
   if(ascii_occurrences(execution_profile_preimage,fragment)/=1) &
    error stop 'v53 execution-profile auxiliary selector differs'
  enddo

  parameter_token=''
  do fraction=1,nfrac
   do parameter=1,parameter_count
    call v53_execution_parameter_token(trim(parameter_prefix)// &
     trim(parameter_suffix(parameter)), &
     channel_native_identifier_token(fraction)(:channel_native_identifier_token_length(fraction)), &
     parameter_token(parameter,fraction),parameter_value,valid)
    if(.not.valid) error stop 'v53 scalar-capacity coefficient is missing or invalid'
    if(bedlaw==braids_capacity_general.and. &
       parameter_value/=capacity_parameter(parameter,fraction)) &
     error stop 'v53 general coefficient differs from nested v46 deck'
    capacity_parameter(parameter,fraction)=parameter_value
   enddo
  enddo
  if(any(capacity_parameter(1,:)<0.d0)) &
   error stop 'v53 scalar-capacity ACal must be nonnegative'
  if(bedlaw==braids_capacity_engelund_hansen.and. &
     .not.all(ieee_is_finite(capacity_parameter(1:3,:)))) &
   error stop 'v53 Engelund-Hansen coefficients must be finite'
  if(any(bedlaw==[braids_capacity_mpm,braids_capacity_engelund_hansen, &
     braids_capacity_general])) then
   ! V50 has one compatibility scalar.  V53 authenticates the complete ordered
   ! per-fraction coefficient document, so only the first-fraction sentinel is
   ! cross-checked here; the other ACal values must remain distinct when given.
   if(capacity_parameter(1,1)/=capacity_route_acal) &
    error stop 'v53 first capacity ACal differs from its v50 sentinel'
  endif

  if(ascii_occurrences(execution_profile_preimage, &
     '{"entry_id":"control.flux_limiter","representation":"inline","scope":'// &
     '{"model_id":"model"},"value":"none"}')/=1.or. &
     ascii_occurrences(execution_profile_preimage, &
     '{"entry_id":"control.laterally_averaged_bedload","representation":"inline",'// &
     '"scope":{"model_id":"model"},"value":false}')/=1.or. &
     ascii_occurrences(execution_profile_preimage, &
     '{"entry_id":"control.maximum_waterdepth","representation":"inline",'// &
     '"scope":{"model_id":"model"},"value":false}')/=1.or. &
     ascii_occurrences(execution_profile_preimage, &
     '{"entry_id":"control.maximum_waterdepth_fraction","representation":"inline",'// &
     '"scope":{"model_id":"model"},"value":1.0}')/=1.or. &
     ascii_occurrences(execution_profile_preimage, &
     '{"entry_id":"control.upwind_bedload","representation":"inline",'// &
     '"scope":{"model_id":"model"},"value":true}')/=1) &
   error stop 'v53 execution-profile comparison controls differ'
  expected_entry_count=(3+parameter_count)*nfrac+9+merge(1,0,foundation_profile)
  if(ascii_occurrences(execution_profile_preimage,'"entry_id":')/=expected_entry_count) &
   error stop 'v53 execution profile contains an unconsumed entry'

  capacity_payload='{"backend":null,"bed_slope":{"method_id":"bed_slope.off",'// &
   '"persistence_code":0,"reference_selector":"ISlope=1","source_id":"d3d4_islope1_off"},'// &
   '"fractions":['
  do fraction=1,nfrac
   if(fraction>1) capacity_payload=capacity_payload//','
   capacity_payload=capacity_payload//'{"coefficients":['
   do parameter=1,parameter_count
    if(parameter>1) capacity_payload=capacity_payload//','
    capacity_payload=capacity_payload//'{"keyword":"'//trim(coefficient_keyword(parameter))// &
     '","value":'//trim(parameter_token(parameter,fraction))//'}'
   enddo
   select case(bedlaw)
   case(braids_capacity_mpm)
    fraction_tail='],"compatible_bed_slope_method_ids":["bed_slope.off",'// &
     '"bed_slope.delft_bagnold_ikeda","bed_slope.delft_koch_flokstra",'// &
     '"bed_slope.delft_parker_andrews"],"compatible_hiding_method_ids":['// &
     '"hiding.none","hiding.egiazaroff","hiding.ashida_michiue",'// &
     '"hiding.parker_klm","hiding.wu_wang_jia"],"compatible_secondary_method_ids":['// &
     '"secondary.off","secondary.equilibrium","secondary.prognostic"],"formula_code":2,'// &
     '"fraction_id":'//channel_native_identifier_token(fraction)(:channel_native_identifier_token_length(fraction))// &
     ',"generic_hiding_supported":true,"hiding_applied_to_capacity":false,'// &
     '"hiding_applied_to_slope":false,"method_id":"capacity.mpm","output":{'// &
     '"bed_channel":"scalar_magnitude","bed_direction":"external_scalar_direction",'// &
     '"source_shape":"scalar_split","suspended_channel":"source_identically_zero",'// &
     '"suspended_direction":"not_applicable","units":"solid_mass_per_time_per_width"},'// &
     '"reference_selector":2,"resolved_shear_source_method_id":"shear.total_roughness",'// &
     '"secondary_steering_applied":false,"secondary_steering_supported":true,'// &
     '"shear_consumer":"total_hydraulic_state",'// &
     '"source_id":"d3d4_mpm_tranb2_resolved_hydraulic_roughness"}'
   case(braids_capacity_engelund_hansen)
    fraction_tail='],"compatible_bed_slope_method_ids":["bed_slope.off",'// &
     '"bed_slope.delft_bagnold_ikeda","bed_slope.delft_koch_flokstra",'// &
     '"bed_slope.delft_parker_andrews"],"compatible_hiding_method_ids":["hiding.none"],'// &
     '"compatible_secondary_method_ids":["secondary.off","secondary.equilibrium",'// &
     '"secondary.prognostic"],"formula_code":8,"fraction_id":'// &
     channel_native_identifier_token(fraction)(:channel_native_identifier_token_length(fraction))// &
     ',"generic_hiding_supported":false,"hiding_applied_to_capacity":false,'// &
     '"hiding_applied_to_slope":false,"method_id":"capacity.engelund_hansen","output":{'// &
     '"bed_channel":"scalar_magnitude","bed_direction":"external_scalar_direction",'// &
     '"source_shape":"scalar_split","suspended_channel":"scalar_magnitude",'// &
     '"suspended_direction":"depth_averaged_flow_direction",'// &
     '"units":"solid_mass_per_time_per_width"},"reference_selector":1,'// &
     '"resolved_shear_source_method_id":"shear.total_roughness",'// &
     '"secondary_steering_applied":false,"secondary_steering_supported":true,'// &
     '"shear_consumer":"total_hydraulic_state",'// &
     '"source_id":"d3d4_engelund_hansen_resolved_hydraulic_chezy"}'
   case default
    fraction_tail='],"compatible_bed_slope_method_ids":["bed_slope.off",'// &
     '"bed_slope.delft_bagnold_ikeda","bed_slope.delft_koch_flokstra",'// &
     '"bed_slope.delft_parker_andrews"],"compatible_hiding_method_ids":['// &
     '"hiding.none","hiding.egiazaroff","hiding.ashida_michiue",'// &
     '"hiding.parker_klm","hiding.wu_wang_jia"],"compatible_secondary_method_ids":['// &
     '"secondary.off","secondary.equilibrium","secondary.prognostic"],"formula_code":11,'// &
     '"fraction_id":'//channel_native_identifier_token(fraction)(:channel_native_identifier_token_length(fraction))// &
     ',"generic_hiding_supported":true,"hiding_applied_to_capacity":false,'// &
     '"hiding_applied_to_slope":false,"method_id":"capacity.general","output":{'// &
     '"bed_channel":"scalar_magnitude","bed_direction":"external_scalar_direction",'// &
     '"source_shape":"scalar_split","suspended_channel":"scalar_magnitude",'// &
     '"suspended_direction":"depth_averaged_flow_direction",'// &
     '"units":"solid_mass_per_time_per_width"},"reference_selector":4,'// &
     '"resolved_shear_source_method_id":"shear.total_roughness",'// &
     '"secondary_steering_applied":false,"secondary_steering_supported":true,'// &
     '"shear_consumer":"total_hydraulic_state","source_id":"d3d4_general_bedload"}'
   end select
   capacity_payload=capacity_payload//fraction_tail
  enddo
  capacity_payload=capacity_payload//'],"hiding":{"method_id":"hiding.none",'// &
   '"persistence_code":0,"reference_selector":"IHidExp=1","source_id":"d3d4_hiding_none"},'// &
   '"parameter_catalog_id":"'//capacity_parameter_catalog_expected//'","schema":1,'// &
   '"scope_id":"delft3d4-flow-2dh-current-capacity-runtime-v1",'// &
   '"secondary_flow":{"method_id":"secondary.off","persistence_code":0,'// &
   '"reference_selector":"Sub1 excludes I","source_id":"d3d4_secondary_off"},'// &
   '"sediment_catalog_id":"'//sediment_catalog_expected//'",'// &
   '"sediment_shear":{"method_id":"shear.total_roughness","persistence_code":"total",'// &
   '"reference_selector":"BsSkin=false","source_id":"d3d4_total_bed_shear"},'// &
   '"semantic_id":"'//capacity_route_profile_schema_expected//'"}'
  call sha256_ascii(capacity_payload,capacity_fingerprint)
  insert_at=index(capacity_payload,',"fractions":')
  if(insert_at==0) error stop 'v53 internal capacity reconstruction failure'
  capacity_document=capacity_payload(:insert_at-1)//',"fingerprint":"'// &
   capacity_fingerprint//'"'//capacity_payload(insert_at:)
  call sha256_ascii(capacity_document,capacity_document_sha256)

  insert_at=index(channel_native_preimage,',"fractions":')
  if(insert_at==0) error stop 'v53 internal channel reconstruction failure'
  channel_document=channel_native_preimage(:insert_at-1)//',"fingerprint":"'// &
   trim(channel_native_preimage_sha256)//'"'//channel_native_preimage(insert_at:)
  call sha256_ascii(channel_document,channel_document_sha256)
  if(component_native_documents(capacity_item)%preimage/=capacity_document.or. &
     component_native_documents(capacity_item)%document_sha256/=capacity_document_sha256) &
   error stop 'v53 capacity component preimage differs from native deck'
  if(component_native_documents(channel_item)%preimage/=channel_document.or. &
     component_native_documents(channel_item)%document_sha256/=channel_document_sha256) &
   error stop 'v53 channel component preimage differs from native deck'

  binding='{"component_id":"capacity","document_sha256":"'// &
   capacity_document_sha256//'","fingerprint":"'//capacity_fingerprint// &
   '","schema":1,"scope_id":"delft3d4-flow-2dh-current-capacity-runtime-v1",'// &
   '"semantic_id":"'//capacity_route_profile_schema_expected//'"}'
  if(ascii_occurrences(execution_profile_preimage,binding)/=1) &
   error stop 'v53 execution profile does not bind the reconstructed capacity document'
  binding='{"component_id":"channel_composition","document_sha256":"'// &
   channel_document_sha256//'","fingerprint":"'//trim(channel_native_preimage_sha256)// &
   '","schema":1,"scope_id":"'//trim(channel_composition_scope_expected)// &
   '","semantic_id":"'//trim(channel_composition_semantic_expected)//'"}'
  if(ascii_occurrences(execution_profile_preimage,binding)/=1.or. &
     ascii_occurrences(execution_profile_preimage,'"component_id":')/=2) &
   error stop 'v53 execution profile component bindings differ'

  capacity_parameter_law=bedlaw;capacity_parameter_count=parameter_count
  if(foundation_profile) then
   capacity_foundation_runtime_declaration=.true.
   capacity_split_retained=.true.
  endif
 end subroutine

 subroutine load_case(path)
  character(*), intent(in) :: path
  integer :: unit,version,envelope_version,base_version,c,f,g,k,ninc,ios,maxseries,maxhydro,maxfractionseries,p,bank_flag,general_flag,secondary_flag,failure_flag,adaptation_flag,bedform_flag,toe_source_count,bank_front_flag
  integer :: native_wire_fraction_count,native_wire_preimage_length, &
   native_wire_chunk_count,native_wire_identifier_length,native_wire_position, &
   native_wire_chunk_bytes,native_wire_classes(32)
  integer :: execution_wire_position,execution_wire_chunk_bytes, &
   execution_component_position
  integer :: capacity_foundation_wire_schema,forcing_wire_preimage_length, &
   forcing_wire_chunk_count,forcing_wire_position,forcing_wire_chunk_bytes, &
   foundation_execution_wire_preimage_length,foundation_execution_wire_chunk_count, &
   foundation_execution_wire_position,foundation_execution_wire_chunk_bytes
  integer :: hiding_foundation_wire_schema,hiding_foundation_wire_preimage_length, &
   hiding_foundation_wire_chunk_count,hiding_foundation_wire_position, &
   hiding_foundation_wire_chunk_bytes
  integer :: secondary_foundation_wire_schema,secondary_foundation_wire_preimage_length, &
   secondary_foundation_wire_chunk_count,secondary_foundation_wire_position, &
   secondary_foundation_wire_chunk_bytes
  integer :: component_wire_identifier_length,component_wire_preimage_length, &
   component_wire_chunk_count,component_wire_position,component_wire_chunk_bytes, &
   component_index
  integer :: flow2dh_boundary_count,flow2dh_neubc_sand_input
  integer :: suspended_amr_fraction_count_input,suspended_amr_enabled_input, &
   suspended_amr_levels_input,suspended_amr_ratio_input
  integer :: nviscous_inlets
  integer,allocatable :: viscous_inlet_ids(:)
  integer,allocatable :: flow2dh_boundary_return_id_input(:)
  integer :: secondary_model_input,secondary_direction_input,secondary_inflow_input
  integer :: gaia_interior_slope_input,gaia_interior_secondary_input,gaia_open_secondary_input
  integer :: hydraulic_roughness_input,white_colebrook_input,sediment_shear_input
  integer :: capacity_parameter_fraction_count,capacity_split_input
  integer :: bank_toe_mode_input
  integer :: binary_unit,binary_header(9)
  integer,allocatable :: binary_bank(:)
  real(8),allocatable :: physical_q(:,:)
  real(8),allocatable :: flow2dh_boundary_return_time_input(:)
  real(8) :: capacity_parameter_input(capacity_parameter_maximum,32)
  real(8) :: expected_settling_velocity,settling_tolerance
  real(8) :: native_wire_bed_factor,native_wire_suspended_factor, &
   native_wire_epsilon
  real(8) :: suspended_amr_dry_tolerance_input, &
   suspended_amr_diffusion_dry_threshold_input,suspended_amr_wet_depth_input, &
   suspended_amr_regrid_interval_input
  character(16) :: binary_magic
   character(1024) :: binary_path
   character(4096) :: channel_v51_record,native_wire_hex
   character(128) :: native_wire_bed_token,native_wire_suspended_token, &
    native_wire_epsilon_token
   character(64) :: native_wire_scope,native_wire_semantic,native_wire_digest, &
    native_wire_computed_digest
   character(64) :: execution_wire_computed_digest
   character(64) :: foundation_document_fingerprint
   character(32) :: native_wire_class_token
   character(len=:),allocatable :: native_wire_expected_preimage,native_wire_chunk, &
    execution_fraction_fragment,execution_component_prefix,execution_component_suffix, &
    execution_root_suffix
  character(64) :: binary_digest
  character(64) :: flow2dh_v48_suspension_schema, &
   flow2dh_v48_suspension_fingerprint
   logical :: engelund_hansen_declaration,gaia_declaration,typed_roughness_declaration, &
    capacity_parameter_declaration,settling_valid,channel_identity_record_valid
  open(newunit=unit,file=path,status='old',action='read')
  binary_unit=0;binary_header=0;production_section_geometry=.false.
  general_avalanche=.false.
  secondary_flow=.false.;secondary_flow_model=0;secondary_flow_transport_direction=.false.
  secondary_flow_espir=1.d0;secondary_flow_momentum_beta=0.d0;secondary_flow_inflow_mode=1
  secondary_initial_integral=0.d0;secondary_source_cumulative=0.d0
  secondary_momentum_initial=0.d0;secondary_momentum_boundary_cumulative=0.d0
  secondary_stress_boundary_cumulative=0.d0;secondary_momentum_remainder_cumulative=0.d0
  secondary_momentum_boundary_trial=0.d0
  bedload_adaptation=.false.;bedload_adaptation_length_diameters=100.d0
  nmorfac=0;morphological_factor=1.d0
  morphological_acceleration=.false.
  bedform_roughness=.false.
  bank_failure=.false.;bank_toe_erosion_mode=0;bank_front_tracking=.false.
  bank_toe_erodibility=0.d0;bank_toe_critical_shear=0.d0
  bank_toe_exponent=1.d0;bank_critical_undercut=0.d0
  bank_toe_capacity_fraction=1.d0
  bank_distributed_transport_fraction=.1d0;bank_distributed_critical_slope=.35d0
   call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
   read(channel_v51_record,*,iostat=ios) version
   if(ios/=0) error stop 'invalid cut-cell deck version record'
   if(version==51) then
    if(.not.channel_identity_record_valid.or.v51_token_count(channel_v51_record)/=1.or. &
       .not.v51_numeric_record_syntax(channel_v51_record,.true.)) &
     error stop 'invalid v51 channel-composition identity declaration'
   else if(version==52.or.version==53.or.version==54.or.version==55.or.version==56) then
    if(.not.channel_identity_record_valid.or.v51_token_count(channel_v51_record)/=1.or. &
       .not.v51_numeric_record_syntax(channel_v51_record,.true.)) &
      error stop 'invalid v52/v53/v54/v55/v56 native identity declaration'
   endif
   if(version<1.or.version>56) error stop 'unsupported cut-cell deck version'
   envelope_version=version
   reference_height_factor=1;sediment_roughness_mode=0
   if(version==51.or.version==52.or.version==53.or.version==54.or.version==55.or.version==56) then
    call read_v51_record(unit,fingerprint,channel_identity_record_valid)
    if(.not.channel_identity_record_valid.or.len_trim(fingerprint)/=64.or. &
       verify(trim(fingerprint),'0123456789abcdef')/=0) then
     if(version==51) error stop 'invalid v51 channel-composition identity declaration'
     error stop 'invalid v52/v53/v54/v55/v56 native identity declaration'
    endif
   else
    read(unit,'(a)') fingerprint
    if(len_trim(fingerprint)/=64) error stop 'missing configuration fingerprint'
   endif
  horizontal_viscosity=0.d0
  horizontal_viscosity_mode=0
  horizontal_viscosity_geometry=0
  manning_friction_chezy_minimum_depth=0.d0
  hydraulic_roughness_mode=braids_roughness_manning
  white_colebrook_chzk2d=.false.;roughness_minimum_work_depth=0.d0
  sediment_shear_mode=0
  skin_kinematic_viscosity=1.d-6;skin_sand_roughness=.003d0
  skin_silt_roughness=.0003d0;skin_critical_mud_fraction_low=.2d0
  skin_critical_mud_fraction_high=.8d0;skin_fluff_fraction=0.d0
  engelund_hansen_declaration=.false.;gaia_declaration=.false.
  typed_roughness_declaration=.false.
  capacity_parameter_declaration=.false.;capacity_split_retained=.false.
  capacity_foundation_runtime_declaration=.false.
  capacity_parameter_law=0;capacity_parameter_count=0
  capacity_parameter_fraction_count=0;capacity_parameter_input=0.d0
  capacity_parameter_schema='';capacity_profile_schema=''
  capacity_diagnostic_schema='';flow2dh_exchange_schema='';suspension_profile_schema=''
  flow2dh_settling_schema='';suspension_profile_fingerprint=''
  constitutive_kinematic_viscosity=1.d-6;capacity_fraction_basis=1
  suspended_adaptation_factor=1.d0;typed_constitutive_declaration=.false.
  settlefrm_code=0;settling_fac_dss=1.d0;settling_temperature_mode_code=0
  settling_temperature_c=0.d0;settling_reference_concentration_declaration=1.d4
  suspended_calibration_factor=1.d0
  sediment_activity_contract_code=0;wet_mask_provider_code=0
  flow2dh_boundary_diffusion_declaration=.false.;flow2dh_neubc_sand=.false.
  flow2dh_horizontal_diffusivity_mode=0;flow2dh_horizontal_diffusivity=0.d0
  flow2dh_diffusion_dry_threshold=0.d0;flow2dh_boundary_count=0
  flow2dh_open_boundary_schema='';flow2dh_horizontal_diffusion_schema=''
  flow2dh_v48_suspension_schema='';flow2dh_v48_suspension_fingerprint=''
  flow2dh_amr_ownership_declaration=.false.
  suspended_amr_schema='';suspended_amr_fingerprint=''
  suspended_amr_profile_fingerprint=''
  suspended_amr_fraction_count_input=0;suspended_amr_enabled_input=0
  suspended_amr_levels_input=0;suspended_amr_ratio_input=0
  suspended_amr_dry_tolerance_input=0.d0
  suspended_amr_diffusion_dry_threshold_input=0.d0
  suspended_amr_wet_depth_input=0.d0;suspended_amr_regrid_interval_input=0.d0
   capacity_route_identity_declaration=.false.
   capacity_foundation_route_identity_declaration=.false.
   capacity_foundation_forcing_identity_declaration=.false.
   hiding_foundation_identity_declaration=.false.
   secondary_foundation_identity_declaration=.false.
  capacity_route_identity_schema=0
  capacity_route_method_id='';capacity_route_execution_schema=''
  capacity_route_profile_schema='';capacity_route_execution_fingerprint=''
  capacity_route_profile_fingerprint=''
  capacity_foundation_forcing_fingerprint=''
  capacity_foundation_forcing_raw_sha256=''
  capacity_foundation_native_execution_fingerprint=''
  hiding_foundation_execution_fingerprint=''
  secondary_foundation_execution_fingerprint=''
  secondary_foundation_base_envelope=0
  secondary_foundation_rmincf=0.d0;secondary_foundation_chzmin=0.d0
  secondary_foundation_espir=0.d0;secondary_foundation_coriolis=0.d0
  secondary_foundation_betac=0.d0
  if(allocated(capacity_foundation_forcing_preimage)) &
   deallocate(capacity_foundation_forcing_preimage)
  if(allocated(capacity_foundation_execution_preimage)) &
   deallocate(capacity_foundation_execution_preimage)
  if(allocated(hiding_foundation_execution_preimage)) &
   deallocate(hiding_foundation_execution_preimage)
  if(allocated(secondary_foundation_execution_preimage)) &
   deallocate(secondary_foundation_execution_preimage)
  capacity_route_acal=0.d0
  channel_composition_declaration=.false.
  channel_composition_schema=0;channel_composition_fraction_count=0
  channel_transport_class_input=0
  channel_bed_factor=1.d0;channel_suspended_factor=1.d0
  channel_speed_epsilon=1.d-6
  channel_composition_scope='';channel_composition_semantic=''
  channel_composition_profile_fingerprint=''
  channel_native_identity_declaration=.false.;channel_native_wire_schema=0
  channel_native_preimage_length=0;channel_native_preimage_sha256=''
  channel_native_identifier_token_length=0;channel_native_identifier_token=''
  if(allocated(channel_native_preimage)) deallocate(channel_native_preimage)
   execution_profile_declaration=.false.;execution_profile_schema=0
  execution_profile_preimage_length=0;execution_profile_semantic=''
   execution_profile_fingerprint=''
   if(allocated(execution_profile_preimage)) deallocate(execution_profile_preimage)
   component_native_identity_declaration=.false.;component_native_count=0
   if(allocated(component_native_documents)) deallocate(component_native_documents)
  diffusion_substeps=0;diffusion_minimum_correction_factor=1.d0
  diffusion_recovered_faces=0;diffusion_recovery_fallbacks=0
  diffusion_affine_recovered_faces=0
  nviscous_inlets=0
  engelund_hansen_chezy=0.d0
  gaia_skin_roughness_ratio=0.d0;gaia_slope_beta=0.d0
  gaia_deviation_beta2=0.d0;gaia_secondary_alpha=0.d0
   gaia_interior_slope_talmon=.true.
   gaia_interior_secondary_stress=.true.
   gaia_open_boundary_secondary_stress=.true.
    if(version==56) then
     call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
     if(.not.channel_identity_record_valid.or.v51_token_count(channel_v51_record)/=3.or. &
        .not.v51_numeric_record_syntax(channel_v51_record,.true.)) &
      error stop 'invalid v56 Q3-H2 secondary header'
     read(channel_v51_record,*,iostat=ios) secondary_foundation_wire_schema, &
      secondary_foundation_wire_preimage_length,secondary_foundation_wire_chunk_count
     if(ios/=0.or.secondary_foundation_wire_schema/=1.or. &
        secondary_foundation_wire_preimage_length<2.or. &
        secondary_foundation_wire_preimage_length>execution_profile_max_preimage_ascii.or. &
        secondary_foundation_wire_chunk_count/= &
         (2*secondary_foundation_wire_preimage_length+channel_native_hex_chunk-1)/ &
          channel_native_hex_chunk) &
      error stop 'invalid v56 Q3-H2 secondary header'
     allocate(character(len=secondary_foundation_wire_preimage_length) :: &
      secondary_foundation_execution_preimage)
     secondary_foundation_execution_preimage=repeat( &
      ' ',secondary_foundation_wire_preimage_length)
     secondary_foundation_wire_position=1
     do k=1,secondary_foundation_wire_chunk_count
      call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
      secondary_foundation_wire_chunk_bytes=min(channel_native_hex_chunk/2, &
       secondary_foundation_wire_preimage_length- &
       secondary_foundation_wire_position+1)
      if(.not.channel_identity_record_valid.or. &
         v51_token_count(channel_v51_record)/=1.or. &
         len_trim(channel_v51_record)/=2*secondary_foundation_wire_chunk_bytes.or. &
         .not.v52_lower_hex( &
          channel_v51_record(:2*secondary_foundation_wire_chunk_bytes))) &
       error stop 'invalid v56 secondary preimage chunk'
      allocate(character(len=secondary_foundation_wire_chunk_bytes)::native_wire_chunk)
      call v52_decode_hex( &
       channel_v51_record(:2*secondary_foundation_wire_chunk_bytes), &
       native_wire_chunk,channel_identity_record_valid)
      if(.not.channel_identity_record_valid) &
       error stop 'invalid v56 secondary preimage chunk'
      secondary_foundation_execution_preimage( &
       secondary_foundation_wire_position:secondary_foundation_wire_position+ &
       secondary_foundation_wire_chunk_bytes-1)=native_wire_chunk
      secondary_foundation_wire_position=secondary_foundation_wire_position+ &
       secondary_foundation_wire_chunk_bytes
      deallocate(native_wire_chunk)
     enddo
     if(secondary_foundation_wire_position/= &
          secondary_foundation_wire_preimage_length+1.or. &
        .not.canonical_json_has_no_external_space( &
         secondary_foundation_execution_preimage).or. &
        .not.v52_canonical_json_document( &
         secondary_foundation_execution_preimage)) &
      error stop 'noncanonical or truncated v56 secondary preimage'
     call canonical_json_document_fingerprint( &
      secondary_foundation_execution_preimage, &
      secondary_foundation_execution_fingerprint,channel_identity_record_valid)
     if(.not.channel_identity_record_valid) &
      error stop 'v56 secondary execution document fingerprint differs'
     secondary_foundation_identity_declaration=.true.
     call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
     if(.not.channel_identity_record_valid.or. &
        v51_token_count(channel_v51_record)/=1.or. &
        .not.v51_numeric_record_syntax(channel_v51_record,.true.)) &
      error stop 'invalid v56 nested foundation declaration'
     read(channel_v51_record,*,iostat=ios) version
     if(ios/=0.or.(version/=54.and.version/=55)) &
      error stop 'v56 must contain one nested v54 or v55 envelope'
     secondary_foundation_base_envelope=version
    endif
    if(version==55) then
     call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
     if(.not.channel_identity_record_valid.or.v51_token_count(channel_v51_record)/=3.or. &
        .not.v51_numeric_record_syntax(channel_v51_record,.true.)) &
      error stop 'invalid v55 Q3-H1 hiding header'
     read(channel_v51_record,*,iostat=ios) hiding_foundation_wire_schema, &
      hiding_foundation_wire_preimage_length,hiding_foundation_wire_chunk_count
     if(ios/=0.or.hiding_foundation_wire_schema/=1.or. &
        hiding_foundation_wire_preimage_length<2.or. &
        hiding_foundation_wire_preimage_length>execution_profile_max_preimage_ascii.or. &
        hiding_foundation_wire_chunk_count/= &
         (2*hiding_foundation_wire_preimage_length+channel_native_hex_chunk-1)/ &
          channel_native_hex_chunk) &
      error stop 'invalid v55 Q3-H1 hiding header'
     allocate(character(len=hiding_foundation_wire_preimage_length) :: &
      hiding_foundation_execution_preimage)
     hiding_foundation_execution_preimage=repeat( &
      ' ',hiding_foundation_wire_preimage_length)
     hiding_foundation_wire_position=1
     do k=1,hiding_foundation_wire_chunk_count
      call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
      hiding_foundation_wire_chunk_bytes=min(channel_native_hex_chunk/2, &
       hiding_foundation_wire_preimage_length-hiding_foundation_wire_position+1)
      if(.not.channel_identity_record_valid.or. &
         v51_token_count(channel_v51_record)/=1.or. &
         len_trim(channel_v51_record)/=2*hiding_foundation_wire_chunk_bytes.or. &
         .not.v52_lower_hex( &
          channel_v51_record(:2*hiding_foundation_wire_chunk_bytes))) &
       error stop 'invalid v55 hiding preimage chunk'
      allocate(character(len=hiding_foundation_wire_chunk_bytes)::native_wire_chunk)
      call v52_decode_hex( &
       channel_v51_record(:2*hiding_foundation_wire_chunk_bytes), &
       native_wire_chunk,channel_identity_record_valid)
      if(.not.channel_identity_record_valid) &
       error stop 'invalid v55 hiding preimage chunk'
      hiding_foundation_execution_preimage( &
       hiding_foundation_wire_position:hiding_foundation_wire_position+ &
       hiding_foundation_wire_chunk_bytes-1)=native_wire_chunk
      hiding_foundation_wire_position=hiding_foundation_wire_position+ &
       hiding_foundation_wire_chunk_bytes
      deallocate(native_wire_chunk)
     enddo
     if(hiding_foundation_wire_position/= &
          hiding_foundation_wire_preimage_length+1.or. &
        .not.canonical_json_has_no_external_space( &
         hiding_foundation_execution_preimage).or. &
        .not.v52_canonical_json_document( &
         hiding_foundation_execution_preimage)) &
      error stop 'noncanonical or truncated v55 hiding preimage'
     call canonical_json_document_fingerprint( &
      hiding_foundation_execution_preimage, &
      hiding_foundation_execution_fingerprint,channel_identity_record_valid)
     if(.not.channel_identity_record_valid) &
      error stop 'v55 hiding execution document fingerprint differs'
     hiding_foundation_identity_declaration=.true.
     call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
     if(.not.channel_identity_record_valid.or. &
        v51_token_count(channel_v51_record)/=1.or. &
        .not.v51_numeric_record_syntax(channel_v51_record,.true.)) &
      error stop 'invalid v55 nested v54 declaration'
     read(channel_v51_record,*,iostat=ios) version
     if(ios/=0.or.version/=54) &
      error stop 'v55 must contain one nested v54 envelope'
    endif
    if(version==54) then
     call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
     if(.not.channel_identity_record_valid.or.v51_token_count(channel_v51_record)/=5.or. &
        .not.v51_numeric_record_syntax(channel_v51_record,.true.)) &
      error stop 'invalid v54 Q3-C forcing/execution header'
     read(channel_v51_record,*,iostat=ios) capacity_foundation_wire_schema, &
      forcing_wire_preimage_length,forcing_wire_chunk_count, &
      foundation_execution_wire_preimage_length, &
      foundation_execution_wire_chunk_count
     if(ios/=0.or.capacity_foundation_wire_schema/=1.or. &
        forcing_wire_preimage_length<2.or. &
        forcing_wire_preimage_length>execution_profile_max_preimage_ascii.or. &
        forcing_wire_chunk_count/=(2*forcing_wire_preimage_length+ &
         channel_native_hex_chunk-1)/channel_native_hex_chunk.or. &
        foundation_execution_wire_preimage_length<2.or. &
        foundation_execution_wire_preimage_length>execution_profile_max_preimage_ascii.or. &
        foundation_execution_wire_chunk_count/= &
         (2*foundation_execution_wire_preimage_length+ &
          channel_native_hex_chunk-1)/channel_native_hex_chunk) &
      error stop 'invalid v54 Q3-C forcing/execution header'
     allocate(character(len=forcing_wire_preimage_length) :: &
      capacity_foundation_forcing_preimage)
     capacity_foundation_forcing_preimage=repeat(' ',forcing_wire_preimage_length)
     forcing_wire_position=1
     do k=1,forcing_wire_chunk_count
      call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
      forcing_wire_chunk_bytes=min(channel_native_hex_chunk/2, &
       forcing_wire_preimage_length-forcing_wire_position+1)
      if(.not.channel_identity_record_valid.or. &
         v51_token_count(channel_v51_record)/=1.or. &
         len_trim(channel_v51_record)/=2*forcing_wire_chunk_bytes.or. &
         .not.v52_lower_hex(channel_v51_record(:2*forcing_wire_chunk_bytes))) &
       error stop 'invalid v54 forcing preimage chunk'
      allocate(character(len=forcing_wire_chunk_bytes)::native_wire_chunk)
      call v52_decode_hex(channel_v51_record(:2*forcing_wire_chunk_bytes), &
       native_wire_chunk,channel_identity_record_valid)
      if(.not.channel_identity_record_valid) &
       error stop 'invalid v54 forcing preimage chunk'
      capacity_foundation_forcing_preimage(forcing_wire_position: &
       forcing_wire_position+forcing_wire_chunk_bytes-1)=native_wire_chunk
      forcing_wire_position=forcing_wire_position+forcing_wire_chunk_bytes
      deallocate(native_wire_chunk)
     enddo
     if(forcing_wire_position/=forcing_wire_preimage_length+1.or. &
        .not.canonical_json_has_no_external_space( &
         capacity_foundation_forcing_preimage).or. &
        .not.v52_canonical_json_document(capacity_foundation_forcing_preimage)) &
      error stop 'noncanonical or truncated v54 forcing preimage'
     call canonical_json_document_fingerprint( &
      capacity_foundation_forcing_preimage, &
      capacity_foundation_forcing_fingerprint,channel_identity_record_valid)
     if(.not.channel_identity_record_valid) &
      error stop 'v54 forcing document fingerprint differs'
     call canonical_json_string_field(capacity_foundation_forcing_preimage, &
      'sha256',capacity_foundation_forcing_raw_sha256,channel_identity_record_valid)
     if(.not.channel_identity_record_valid.or. &
        len_trim(capacity_foundation_forcing_raw_sha256)/=64.or. &
        verify(trim(capacity_foundation_forcing_raw_sha256), &
         '0123456789abcdef')/=0) &
      error stop 'invalid v54 raw roughness fingerprint'

     allocate(character(len=foundation_execution_wire_preimage_length) :: &
      capacity_foundation_execution_preimage)
     capacity_foundation_execution_preimage= &
      repeat(' ',foundation_execution_wire_preimage_length)
     foundation_execution_wire_position=1
     do k=1,foundation_execution_wire_chunk_count
      call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
      foundation_execution_wire_chunk_bytes=min(channel_native_hex_chunk/2, &
       foundation_execution_wire_preimage_length- &
       foundation_execution_wire_position+1)
      if(.not.channel_identity_record_valid.or. &
         v51_token_count(channel_v51_record)/=1.or. &
         len_trim(channel_v51_record)/=2*foundation_execution_wire_chunk_bytes.or. &
         .not.v52_lower_hex( &
          channel_v51_record(:2*foundation_execution_wire_chunk_bytes))) &
       error stop 'invalid v54 execution preimage chunk'
      allocate(character(len=foundation_execution_wire_chunk_bytes)::native_wire_chunk)
      call v52_decode_hex( &
       channel_v51_record(:2*foundation_execution_wire_chunk_bytes), &
       native_wire_chunk,channel_identity_record_valid)
      if(.not.channel_identity_record_valid) &
       error stop 'invalid v54 execution preimage chunk'
      capacity_foundation_execution_preimage( &
       foundation_execution_wire_position:foundation_execution_wire_position+ &
       foundation_execution_wire_chunk_bytes-1)=native_wire_chunk
      foundation_execution_wire_position=foundation_execution_wire_position+ &
       foundation_execution_wire_chunk_bytes
      deallocate(native_wire_chunk)
     enddo
     if(foundation_execution_wire_position/= &
          foundation_execution_wire_preimage_length+1.or. &
        .not.canonical_json_has_no_external_space( &
         capacity_foundation_execution_preimage).or. &
        .not.v52_canonical_json_document(capacity_foundation_execution_preimage)) &
      error stop 'noncanonical or truncated v54 execution preimage'
     call canonical_json_document_fingerprint( &
      capacity_foundation_execution_preimage, &
      capacity_foundation_native_execution_fingerprint, &
      channel_identity_record_valid)
     if(.not.channel_identity_record_valid) &
      error stop 'v54 capacity execution document fingerprint differs'
     capacity_foundation_forcing_identity_declaration=.true.
     call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
     if(.not.channel_identity_record_valid.or. &
        v51_token_count(channel_v51_record)/=1.or. &
        .not.v51_numeric_record_syntax(channel_v51_record,.true.)) &
      error stop 'invalid v54 nested v53 declaration'
     read(channel_v51_record,*,iostat=ios) version
     if(ios/=0.or.version/=53) &
      error stop 'v54 must contain one nested v53 envelope'
    endif
    if(version==53) then
     call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
     if(.not.channel_identity_record_valid.or.v51_token_count(channel_v51_record)/=1.or. &
        .not.v51_numeric_record_syntax(channel_v51_record,.true.)) &
      error stop 'invalid v53 component-document inventory declaration'
     read(channel_v51_record,*,iostat=ios) component_native_count
     if(ios/=0.or.component_native_count<1.or. &
        component_native_count>component_native_maximum) &
      error stop 'invalid v53 component-document inventory declaration'
     allocate(component_native_documents(component_native_count))
     do component_index=1,component_native_count
      call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
      if(.not.channel_identity_record_valid.or.v51_token_count(channel_v51_record)/=4) &
       error stop 'invalid v53 component-document header'
      read(channel_v51_record,*,iostat=ios) component_wire_identifier_length, &
       native_wire_hex,component_wire_preimage_length,component_wire_chunk_count
      if(ios/=0.or.component_wire_identifier_length<1.or. &
         component_wire_identifier_length>component_native_identifier_maximum.or. &
         len_trim(native_wire_hex)/=2*component_wire_identifier_length.or. &
         .not.v52_lower_hex(native_wire_hex(:2*component_wire_identifier_length)).or. &
         component_wire_preimage_length<2.or. &
         component_wire_preimage_length>execution_profile_max_preimage_ascii.or. &
         component_wire_chunk_count/=(2*component_wire_preimage_length+ &
          channel_native_hex_chunk-1)/channel_native_hex_chunk) &
       error stop 'invalid v53 component-document header'
      call v52_decode_hex(native_wire_hex(:2*component_wire_identifier_length), &
       component_native_documents(component_index)%component_id( &
        :component_wire_identifier_length),channel_identity_record_valid)
      if(.not.channel_identity_record_valid.or. &
         verify(component_native_documents(component_index)%component_id( &
          :component_wire_identifier_length), &
          'abcdefghijklmnopqrstuvwxyz0123456789_')/=0) &
       error stop 'invalid v53 component identifier'
      if(component_index>1) then
       if(trim(component_native_documents(component_index-1)%component_id)>= &
          trim(component_native_documents(component_index)%component_id)) &
        error stop 'v53 component identifiers are not unique and sorted'
      endif
      allocate(character(len=component_wire_preimage_length) :: &
       component_native_documents(component_index)%preimage)
      component_native_documents(component_index)%preimage= &
       repeat(' ',component_wire_preimage_length)
      component_wire_position=1
      do k=1,component_wire_chunk_count
       call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
       component_wire_chunk_bytes=min(channel_native_hex_chunk/2, &
        component_wire_preimage_length-component_wire_position+1)
       if(.not.channel_identity_record_valid.or. &
          v51_token_count(channel_v51_record)/=1.or. &
          len_trim(channel_v51_record)/=2*component_wire_chunk_bytes.or. &
          .not.v52_lower_hex(channel_v51_record(:2*component_wire_chunk_bytes))) &
        error stop 'invalid v53 component-document preimage chunk'
       allocate(character(len=component_wire_chunk_bytes)::native_wire_chunk)
       call v52_decode_hex(channel_v51_record(:2*component_wire_chunk_bytes), &
        native_wire_chunk,channel_identity_record_valid)
       if(.not.channel_identity_record_valid) &
        error stop 'invalid v53 component-document preimage chunk'
       component_native_documents(component_index)%preimage(component_wire_position: &
        component_wire_position+component_wire_chunk_bytes-1)=native_wire_chunk
       component_wire_position=component_wire_position+component_wire_chunk_bytes
       deallocate(native_wire_chunk)
      enddo
      if(component_wire_position/=component_wire_preimage_length+1.or. &
         .not.canonical_json_has_no_external_space( &
          component_native_documents(component_index)%preimage).or. &
         .not.v52_canonical_json_document( &
          component_native_documents(component_index)%preimage)) &
       error stop 'noncanonical or truncated v53 component-document preimage'
      call sha256_ascii(component_native_documents(component_index)%preimage, &
       component_native_documents(component_index)%document_sha256)
     enddo
     component_native_identity_declaration=.true.
     call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
     if(.not.channel_identity_record_valid.or.v51_token_count(channel_v51_record)/=1.or. &
        .not.v51_numeric_record_syntax(channel_v51_record,.true.)) &
      error stop 'invalid v53 nested v52 declaration'
     read(channel_v51_record,*,iostat=ios) version
     if(ios/=0.or.version/=52) error stop 'v53 must contain one nested v52 envelope'
    endif
    if(version==52) then
    call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
    if(.not.channel_identity_record_valid.or.v51_token_count(channel_v51_record)/=4.or. &
       .not.v51_numeric_record_syntax(channel_v51_record,.true.)) &
     error stop 'invalid v52 native channel-composition identity declaration'
    read(channel_v51_record,*,iostat=ios) channel_native_wire_schema, &
     native_wire_fraction_count,native_wire_preimage_length,native_wire_chunk_count
    if(ios/=0.or.channel_native_wire_schema/=channel_native_wire_schema_expected.or. &
       native_wire_fraction_count<1.or.native_wire_fraction_count>32.or. &
       native_wire_preimage_length<2.or. &
       native_wire_preimage_length>channel_native_max_preimage_ascii.or. &
       native_wire_chunk_count/=(2*native_wire_preimage_length+ &
        channel_native_hex_chunk-1)/channel_native_hex_chunk) &
     error stop 'invalid v52 native channel-composition identity declaration'
    call read_v51_record(unit,native_wire_scope,channel_identity_record_valid)
    if(.not.channel_identity_record_valid.or. &
       trim(native_wire_scope)/=channel_composition_scope_expected) &
     error stop 'invalid v52 native channel-composition identity declaration'
    call read_v51_record(unit,native_wire_semantic,channel_identity_record_valid)
    if(.not.channel_identity_record_valid.or. &
       trim(native_wire_semantic)/=channel_composition_semantic_expected) &
     error stop 'invalid v52 native channel-composition identity declaration'
    call read_v51_record(unit,native_wire_digest,channel_identity_record_valid)
    if(.not.channel_identity_record_valid.or.len_trim(native_wire_digest)/=64.or. &
       verify(trim(native_wire_digest),'0123456789abcdef')/=0) &
     error stop 'invalid v52 native channel-composition identity declaration'
    call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
    if(.not.channel_identity_record_valid.or.v51_token_count(channel_v51_record)/=3.or. &
       .not.v51_numeric_record_syntax(channel_v51_record,.false.)) &
     error stop 'invalid v52 native channel-composition identity declaration'
    native_wire_bed_token='';native_wire_suspended_token='';native_wire_epsilon_token=''
    read(channel_v51_record,*,iostat=ios) native_wire_bed_token, &
     native_wire_suspended_token,native_wire_epsilon_token
    if(ios/=0.or..not.v52_canonical_json_number(trim(native_wire_bed_token)).or. &
       .not.v52_canonical_json_number(trim(native_wire_suspended_token)).or. &
       .not.v52_canonical_json_number(trim(native_wire_epsilon_token))) &
     error stop 'noncanonical v52 channel-composition numeric token'
    read(native_wire_bed_token,*,iostat=ios) native_wire_bed_factor
    if(ios/=0) error stop 'invalid v52 native channel-composition identity declaration'
    read(native_wire_suspended_token,*,iostat=ios) native_wire_suspended_factor
    if(ios/=0) error stop 'invalid v52 native channel-composition identity declaration'
    read(native_wire_epsilon_token,*,iostat=ios) native_wire_epsilon
    if(ios/=0.or..not.all(ieee_is_finite([native_wire_bed_factor, &
       native_wire_suspended_factor,native_wire_epsilon])).or. &
       native_wire_bed_factor<0.d0.or.native_wire_suspended_factor<=0.d0.or. &
       native_wire_epsilon/=1.d-6) &
     error stop 'invalid v52 native channel-composition identity declaration'
    channel_native_identifier_token='';channel_native_identifier_token_length=0
    do k=1,native_wire_fraction_count
     call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
     if(.not.channel_identity_record_valid.or.v51_token_count(channel_v51_record)/=2) &
      error stop 'invalid v52 ordered fraction identifier record'
     native_wire_hex='';native_wire_identifier_length=0
     read(channel_v51_record,*,iostat=ios) native_wire_identifier_length,native_wire_hex
     if(ios/=0) error stop 'invalid v52 ordered fraction identifier record'
     if(native_wire_identifier_length<2.or. &
        native_wire_identifier_length>channel_native_max_identifier_token_ascii) &
      error stop 'invalid v52 ordered fraction identifier record'
     if(len_trim(native_wire_hex)/=2*native_wire_identifier_length) &
      error stop 'invalid v52 ordered fraction identifier record'
     if(.not.v52_lower_hex(native_wire_hex(:2*native_wire_identifier_length))) &
      error stop 'invalid v52 ordered fraction identifier record'
     call v52_decode_hex(native_wire_hex(:2*native_wire_identifier_length), &
      channel_native_identifier_token(k)(:native_wire_identifier_length), &
      channel_identity_record_valid)
     if(.not.channel_identity_record_valid.or. &
        .not.v52_canonical_json_string( &
         channel_native_identifier_token(k)(:native_wire_identifier_length))) &
      error stop 'noncanonical v52 ordered fraction identifier'
     do p=1,k-1
      if(channel_native_identifier_token_length(p)==native_wire_identifier_length.and. &
         channel_native_identifier_token(p)(:native_wire_identifier_length)== &
         channel_native_identifier_token(k)(:native_wire_identifier_length)) &
       error stop 'duplicate v52 ordered fraction identifier'
     enddo
     channel_native_identifier_token_length(k)=native_wire_identifier_length
    enddo
    call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
    if(.not.channel_identity_record_valid.or. &
       v51_token_count(channel_v51_record)/=native_wire_fraction_count.or. &
       .not.v51_numeric_record_syntax(channel_v51_record,.true.)) &
     error stop 'invalid v52 ordered transport-class record'
    native_wire_classes=0
    read(channel_v51_record,*,iostat=ios) native_wire_classes(:native_wire_fraction_count)
    if(ios/=0.or.any(native_wire_classes(:native_wire_fraction_count)/=BRAIDS_TRA_BEDLOAD.and. &
       native_wire_classes(:native_wire_fraction_count)/=BRAIDS_TRA_COMBINE)) &
     error stop 'invalid v52 ordered transport-class record'
    allocate(character(len=native_wire_preimage_length)::channel_native_preimage)
    channel_native_preimage=repeat(' ',native_wire_preimage_length);native_wire_position=1
    do k=1,native_wire_chunk_count
     call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
     native_wire_chunk_bytes=min(channel_native_hex_chunk/2, &
      native_wire_preimage_length-native_wire_position+1)
     if(.not.channel_identity_record_valid.or.v51_token_count(channel_v51_record)/=1.or. &
        len_trim(channel_v51_record)/=2*native_wire_chunk_bytes.or. &
        .not.v52_lower_hex(channel_v51_record(:2*native_wire_chunk_bytes))) &
      error stop 'invalid v52 canonical preimage chunk'
     allocate(character(len=native_wire_chunk_bytes)::native_wire_chunk)
     call v52_decode_hex(channel_v51_record(:2*native_wire_chunk_bytes), &
      native_wire_chunk,channel_identity_record_valid)
     if(.not.channel_identity_record_valid) &
      error stop 'invalid v52 canonical preimage chunk'
     channel_native_preimage(native_wire_position: &
      native_wire_position+native_wire_chunk_bytes-1)=native_wire_chunk
     native_wire_position=native_wire_position+native_wire_chunk_bytes
     deallocate(native_wire_chunk)
    enddo
    if(native_wire_position/=native_wire_preimage_length+1) &
     error stop 'truncated v52 canonical preimage'
    call sha256_ascii(channel_native_preimage,native_wire_computed_digest)
    if(native_wire_computed_digest/=native_wire_digest) &
     error stop 'v52 canonical preimage SHA-256 differs'
    native_wire_expected_preimage='{"bed_factor":'//trim(native_wire_bed_token)// &
     ',"epsilon":'//trim(native_wire_epsilon_token)//',"fractions":['
    do k=1,native_wire_fraction_count
     if(k>1) native_wire_expected_preimage=native_wire_expected_preimage//','
     write(native_wire_class_token,'(i0)') native_wire_classes(k)
     native_wire_expected_preimage=native_wire_expected_preimage// &
      '{"fraction_id":'// &
      channel_native_identifier_token(k)(:channel_native_identifier_token_length(k))// &
      ',"transport_class":'//trim(native_wire_class_token)//'}'
    enddo
    native_wire_expected_preimage=native_wire_expected_preimage// &
     '],"schema":1,"scope_id":"'//trim(native_wire_scope)// &
     '","semantic_id":"'//trim(native_wire_semantic)// &
     '","suspended_factor":'//trim(native_wire_suspended_token)//'}'
    if(len(native_wire_expected_preimage)/=native_wire_preimage_length.or. &
       native_wire_expected_preimage/=channel_native_preimage) &
     error stop 'noncanonical v52 channel-composition preimage'
    channel_native_identity_declaration=.true.
    channel_native_preimage_length=native_wire_preimage_length
    channel_native_preimage_sha256=native_wire_digest
    call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
    if(.not.channel_identity_record_valid.or.v51_token_count(channel_v51_record)/=1.or. &
       .not.v51_numeric_record_syntax(channel_v51_record,.true.)) &
     error stop 'invalid v52 execution-profile schema declaration'
    read(channel_v51_record,*,iostat=ios) execution_profile_schema
    if(ios/=0.or.execution_profile_schema/=execution_profile_schema_expected) &
     error stop 'invalid v52 execution-profile schema declaration'
    call read_v51_record(unit,execution_profile_semantic,channel_identity_record_valid)
    if(.not.channel_identity_record_valid.or. &
       execution_profile_semantic/=execution_profile_semantic_expected) &
     error stop 'invalid v52 execution-profile semantic identity'
    call read_v51_record(unit,execution_profile_fingerprint,channel_identity_record_valid)
    if(.not.channel_identity_record_valid.or.len_trim(execution_profile_fingerprint)/=64.or. &
       verify(trim(execution_profile_fingerprint),'0123456789abcdef')/=0) &
     error stop 'invalid v52 execution-profile fingerprint'
    call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
    if(.not.channel_identity_record_valid.or.v51_token_count(channel_v51_record)/=1.or. &
       .not.v51_numeric_record_syntax(channel_v51_record,.true.)) &
     error stop 'invalid v52 execution-profile preimage length'
    read(channel_v51_record,*,iostat=ios) execution_profile_preimage_length
    if(ios/=0.or.execution_profile_preimage_length<2.or. &
       execution_profile_preimage_length>execution_profile_max_preimage_ascii) &
     error stop 'invalid v52 execution-profile preimage length'
    native_wire_chunk_count=(2*execution_profile_preimage_length+ &
     channel_native_hex_chunk-1)/channel_native_hex_chunk
    allocate(character(len=execution_profile_preimage_length)::execution_profile_preimage)
    execution_profile_preimage=repeat(' ',execution_profile_preimage_length);execution_wire_position=1
    do k=1,native_wire_chunk_count
     call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
     execution_wire_chunk_bytes=min(channel_native_hex_chunk/2, &
      execution_profile_preimage_length-execution_wire_position+1)
     if(.not.channel_identity_record_valid.or.v51_token_count(channel_v51_record)/=1.or. &
        len_trim(channel_v51_record)/=2*execution_wire_chunk_bytes.or. &
        .not.v52_lower_hex(channel_v51_record(:2*execution_wire_chunk_bytes))) &
      error stop 'invalid v52 execution-profile preimage chunk'
     allocate(character(len=execution_wire_chunk_bytes)::native_wire_chunk)
     call v52_decode_hex(channel_v51_record(:2*execution_wire_chunk_bytes), &
      native_wire_chunk,channel_identity_record_valid)
     if(.not.channel_identity_record_valid) &
      error stop 'invalid v52 execution-profile preimage chunk'
     execution_profile_preimage(execution_wire_position: &
      execution_wire_position+execution_wire_chunk_bytes-1)=native_wire_chunk
     execution_wire_position=execution_wire_position+execution_wire_chunk_bytes
     deallocate(native_wire_chunk)
    enddo
    if(execution_wire_position/=execution_profile_preimage_length+1) &
     error stop 'truncated v52 execution-profile preimage'
    call sha256_ascii(execution_profile_preimage,execution_wire_computed_digest)
    if(execution_wire_computed_digest/=execution_profile_fingerprint) &
     error stop 'v52 execution-profile SHA-256 differs'
    if(.not.canonical_json_has_no_external_space(execution_profile_preimage).or. &
       .not.v52_canonical_json_document(execution_profile_preimage).or. &
       index(execution_profile_preimage,'{"backend":null,"component_profiles":[')/=1.or. &
       ascii_occurrences(execution_profile_preimage,'"backend":')/=1.or. &
       ascii_occurrences(execution_profile_preimage,'"fraction_layout":')/=1.or. &
       ascii_occurrences(execution_profile_preimage,'"schema":')<1.or. &
       ascii_occurrences(execution_profile_preimage,'"scope_id":')<1.or. &
       ascii_occurrences(execution_profile_preimage,'"semantic_id":')<1) &
     error stop 'noncanonical or incomplete v52 execution-profile preimage'
    execution_fraction_fragment='"fraction_layout":['
    do k=1,native_wire_fraction_count
     if(k>1) execution_fraction_fragment=execution_fraction_fragment//','
     write(native_wire_class_token,'(i0)') native_wire_classes(k)
     execution_fraction_fragment=execution_fraction_fragment// &
      '{"fraction_id":'// &
      channel_native_identifier_token(k)(:channel_native_identifier_token_length(k))// &
      ',"transport_class":'//trim(native_wire_class_token)//'}'
    enddo
    execution_fraction_fragment=execution_fraction_fragment//']'
    if(ascii_occurrences(execution_profile_preimage,execution_fraction_fragment)/=1) &
     error stop 'v52 execution-profile fraction layout differs'
    execution_component_prefix='{"component_id":"channel_composition",'// &
     '"document_sha256":"'
    if(ascii_occurrences(execution_profile_preimage,execution_component_prefix)/=1) &
     error stop 'v52 execution profile lacks one channel component'
    execution_component_position=index(execution_profile_preimage,execution_component_prefix)+ &
     len(execution_component_prefix)
    if(execution_component_position+63>len(execution_profile_preimage)) &
     error stop 'invalid v52 channel component document identity'
    if(verify(execution_profile_preimage(execution_component_position: &
       execution_component_position+63),'0123456789abcdef')/=0) &
     error stop 'invalid v52 channel component document identity'
    execution_component_suffix='","fingerprint":"'//trim(native_wire_digest)// &
     '","schema":1,"scope_id":"'//trim(native_wire_scope)// &
     '","semantic_id":"'//trim(native_wire_semantic)//'"}'
    if(execution_component_position+64+len(execution_component_suffix)-1> &
       len(execution_profile_preimage)) &
     error stop 'v52 execution-profile channel component differs'
    if(execution_profile_preimage(execution_component_position+64: &
        execution_component_position+63+len(execution_component_suffix))/= &
       execution_component_suffix) &
     error stop 'v52 execution-profile channel component differs'
    execution_root_suffix=',"schema":2,"scope_id":"'// &
     trim(execution_profile_scope_expected)//'","semantic_id":"'// &
     trim(execution_profile_semantic_expected)//'"}'
    if(len(execution_profile_preimage)<len(execution_root_suffix)) &
     error stop 'v52 execution-profile root identity differs'
    if(execution_profile_preimage(len(execution_profile_preimage)- &
        len(execution_root_suffix)+1:)/=execution_root_suffix) &
     error stop 'v52 execution-profile root identity differs'
    execution_profile_declaration=.true.
    call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
    if(.not.channel_identity_record_valid.or.v51_token_count(channel_v51_record)/=1.or. &
       .not.v51_numeric_record_syntax(channel_v51_record,.true.)) &
     error stop 'invalid v52 nested v51 declaration'
    read(channel_v51_record,*,iostat=ios) version
    if(ios/=0.or.version/=51) error stop 'v52 must contain one nested v51 envelope'
   endif
   if(version==51) then
    call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
    if(.not.channel_identity_record_valid.or.v51_token_count(channel_v51_record)/=5.or. &
       .not.v51_numeric_record_syntax(channel_v51_record,.false.)) &
     error stop 'invalid v51 channel-composition identity declaration'
    read(channel_v51_record,*,iostat=ios) channel_composition_schema, &
     channel_composition_fraction_count,channel_bed_factor, &
     channel_suspended_factor,channel_speed_epsilon
    if(ios/=0) error stop 'invalid v51 channel-composition identity declaration'
   call read_v51_record(unit,channel_composition_scope,channel_identity_record_valid)
   if(.not.channel_identity_record_valid.or. &
      trim(channel_composition_scope)/=channel_composition_scope_expected) &
    error stop 'invalid v51 channel-composition identity declaration'
   call read_v51_record(unit,channel_composition_semantic,channel_identity_record_valid)
   if(.not.channel_identity_record_valid.or. &
      trim(channel_composition_semantic)/=channel_composition_semantic_expected) &
    error stop 'invalid v51 channel-composition identity declaration'
   call read_v51_record(unit,channel_composition_profile_fingerprint, &
    channel_identity_record_valid)
   if(.not.channel_identity_record_valid) &
    error stop 'invalid v51 channel-composition identity declaration'
   if(channel_composition_schema/=1.or. &
      channel_composition_fraction_count<1.or.channel_composition_fraction_count>32.or. &
      .not.all(ieee_is_finite([channel_bed_factor,channel_suspended_factor, &
       channel_speed_epsilon])).or.channel_bed_factor<0.d0.or. &
      channel_suspended_factor<=0.d0.or.channel_speed_epsilon/=1.d-6.or. &
      trim(channel_composition_scope)/=channel_composition_scope_expected.or. &
      trim(channel_composition_semantic)/=channel_composition_semantic_expected.or. &
      len_trim(channel_composition_profile_fingerprint)/=64.or. &
      verify(trim(channel_composition_profile_fingerprint),'0123456789abcdef')/=0) &
    error stop 'invalid v51 channel-composition identity declaration'
   call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
   if(.not.channel_identity_record_valid.or. &
      v51_token_count(channel_v51_record)/=channel_composition_fraction_count.or. &
      .not.v51_numeric_record_syntax(channel_v51_record,.true.)) &
    error stop 'invalid v51 channel-composition identity declaration'
   read(channel_v51_record,*,iostat=ios) &
    channel_transport_class_input(:channel_composition_fraction_count)
   if(ios/=0) error stop 'invalid v51 channel-composition identity declaration'
   if(any(channel_transport_class_input(:channel_composition_fraction_count)/=BRAIDS_TRA_BEDLOAD.and. &
          channel_transport_class_input(:channel_composition_fraction_count)/=BRAIDS_TRA_COMBINE)) &
    error stop 'unsupported v51 transport class'
   channel_composition_declaration=.true.
   call read_v51_record(unit,channel_v51_record,channel_identity_record_valid)
   if(.not.channel_identity_record_valid.or.v51_token_count(channel_v51_record)/=1.or. &
      .not.v51_numeric_record_syntax(channel_v51_record,.true.)) &
    error stop 'invalid v51 channel-composition identity declaration'
   read(channel_v51_record,*,iostat=ios) version
   if(ios/=0) error stop 'invalid v51 channel-composition identity declaration'
   if(version<1.or.version>50) error stop 'unsupported v51 nested deck version'
   if(channel_native_identity_declaration) then
    if(channel_composition_fraction_count/=native_wire_fraction_count.or. &
       channel_composition_scope/=native_wire_scope.or. &
       channel_composition_semantic/=native_wire_semantic.or. &
       channel_composition_profile_fingerprint/=native_wire_digest.or. &
       channel_bed_factor/=native_wire_bed_factor.or. &
       channel_suspended_factor/=native_wire_suspended_factor.or. &
       channel_speed_epsilon/=native_wire_epsilon.or. &
       any(channel_transport_class_input(:native_wire_fraction_count)/= &
        native_wire_classes(:native_wire_fraction_count))) &
     error stop 'v52 native identity differs from nested v51 controls'
   endif
   endif
  if(version==50) then
   read(unit,'(a)') capacity_route_method_id
   read(unit,*) capacity_route_identity_schema
   read(unit,*) capacity_route_acal
   read(unit,'(a)') capacity_route_execution_schema
   read(unit,'(a)') capacity_route_profile_schema
   read(unit,'(a)') capacity_route_execution_fingerprint
   read(unit,'(a)') capacity_route_profile_fingerprint
   if(capacity_route_identity_schema/=1.or. &
      .not.ieee_is_finite(capacity_route_acal).or.capacity_route_acal<0.d0.or. &
       all(trim(capacity_route_execution_schema)/=[character(len=64) :: &
        capacity_execution_schema_expected, &
        capacity_foundation_execution_schema_expected]).or. &
      trim(capacity_route_profile_schema)/=capacity_route_profile_schema_expected.or. &
      len_trim(capacity_route_execution_fingerprint)/=64.or. &
      verify(trim(capacity_route_execution_fingerprint),'0123456789abcdef')/=0.or. &
      len_trim(capacity_route_profile_fingerprint)/=64.or. &
      verify(trim(capacity_route_profile_fingerprint),'0123456789abcdef')/=0) &
    error stop 'invalid v50 capacity execution/profile identity declaration'
   if(all(trim(capacity_route_method_id)/=[character(len=64) :: &
      'capacity.engelund_hansen_constant_chezy_legacy', &
      'capacity.engelund_hansen', &
       'capacity.mpm_specified_cf_legacy','capacity.mpm','capacity.general'])) &
     error stop 'unsupported v50 capacity method identity'
    capacity_foundation_route_identity_declaration= &
     trim(capacity_route_execution_schema)==capacity_foundation_execution_schema_expected
   capacity_route_identity_declaration=.true.
   read(unit,*) version
   if(version<1.or.version>49) error stop 'unsupported v50 nested deck version'
  endif
  if(version==49) then
   read(unit,'(a)') suspended_amr_schema
   read(unit,'(a)') suspended_amr_fingerprint
   read(unit,'(a)') suspended_amr_profile_fingerprint
   read(unit,*) suspended_amr_fraction_count_input, &
    suspended_amr_dry_tolerance_input, &
    suspended_amr_diffusion_dry_threshold_input,suspended_amr_enabled_input, &
    suspended_amr_levels_input,suspended_amr_ratio_input, &
    suspended_amr_wet_depth_input,suspended_amr_regrid_interval_input
   if(trim(suspended_amr_schema)/=suspended_amr_schema_expected.or. &
      len_trim(suspended_amr_fingerprint)/=64.or. &
      verify(trim(suspended_amr_fingerprint),'0123456789abcdef')/=0.or. &
      len_trim(suspended_amr_profile_fingerprint)/=64.or. &
      verify(trim(suspended_amr_profile_fingerprint), &
       '0123456789abcdef')/=0.or. &
      suspended_amr_fraction_count_input<1.or. &
      all(suspended_amr_enabled_input/=[0,1]).or. &
      suspended_amr_levels_input<1.or.suspended_amr_ratio_input/=2.or. &
      .not.all(ieee_is_finite([suspended_amr_dry_tolerance_input, &
       suspended_amr_diffusion_dry_threshold_input, &
       suspended_amr_wet_depth_input,suspended_amr_regrid_interval_input])).or. &
      suspended_amr_dry_tolerance_input<=0.d0.or. &
      min(suspended_amr_diffusion_dry_threshold_input, &
       suspended_amr_wet_depth_input,suspended_amr_regrid_interval_input)<0.d0) &
    error stop 'invalid v49 suspended wet/dry-AMR ownership declaration'
   if((suspended_amr_enabled_input==1.and. &
       (suspended_amr_levels_input<2.or. &
        suspended_amr_regrid_interval_input<=0.d0)).or. &
      (suspended_amr_enabled_input==0.and. &
       (suspended_amr_levels_input/=1.or. &
        suspended_amr_wet_depth_input/=0.d0.or. &
        suspended_amr_regrid_interval_input/=0.d0))) &
    error stop 'invalid v49 fixed/adaptive ownership scope'
   flow2dh_amr_ownership_declaration=.true.
   read(unit,*) version
   if(version/=48) error stop 'v49 ownership contract requires a nested v48 deck'
  endif
  if(version==48) then
   read(unit,*) flow2dh_neubc_sand_input,flow2dh_horizontal_diffusivity_mode, &
    flow2dh_horizontal_diffusivity,flow2dh_diffusion_dry_threshold, &
    expected_settling_velocity,flow2dh_boundary_count
   read(unit,'(a)') flow2dh_open_boundary_schema
   read(unit,'(a)') flow2dh_horizontal_diffusion_schema
   read(unit,'(a)') flow2dh_v48_suspension_schema
   read(unit,'(a)') flow2dh_v48_suspension_fingerprint
   if(flow2dh_boundary_count<0) error stop 'invalid v48 boundary count'
   allocate(flow2dh_boundary_return_id_input(flow2dh_boundary_count), &
    flow2dh_boundary_return_time_input(flow2dh_boundary_count))
   do k=1,flow2dh_boundary_count
    read(unit,*) flow2dh_boundary_return_id_input(k), &
     flow2dh_boundary_return_time_input(k)
   enddo
   if(all(flow2dh_neubc_sand_input/=[0,1]).or. &
      flow2dh_horizontal_diffusivity_mode/=1.or. &
      .not.all(ieee_is_finite([flow2dh_horizontal_diffusivity, &
       flow2dh_diffusion_dry_threshold,expected_settling_velocity])).or. &
      flow2dh_horizontal_diffusivity<0.d0.or. &
      flow2dh_diffusion_dry_threshold<0.d0.or. &
      expected_settling_velocity/=flow2dh_horizontal_schmidt.or. &
      trim(flow2dh_open_boundary_schema)/= &
       flow2dh_open_boundary_schema_expected.or. &
      trim(flow2dh_horizontal_diffusion_schema)/= &
       flow2dh_horizontal_diffusion_schema_expected.or. &
      trim(flow2dh_v48_suspension_schema)/= &
       suspension_profile_schema_expected.or. &
      len_trim(flow2dh_v48_suspension_fingerprint)/=64.or. &
      verify(trim(flow2dh_v48_suspension_fingerprint), &
       '0123456789abcdef')/=0.or. &
      .not.all(ieee_is_finite(flow2dh_boundary_return_time_input))) &
    error stop 'invalid v48 FLOW 2DH boundary/diffusion declaration'
   flow2dh_neubc_sand=flow2dh_neubc_sand_input==1
   flow2dh_boundary_diffusion_declaration=.true.
   read(unit,*) version
   if(version/=47) error stop 'v48 requires a nested v47 suspension deck'
  endif
  if(version==47) then
   read(unit,*) constitutive_kinematic_viscosity,capacity_fraction_basis, &
    suspended_adaptation_factor
   read(unit,*) settlefrm_code,settling_fac_dss,settling_temperature_mode_code, &
    settling_temperature_c,settling_reference_concentration_declaration, &
    suspended_calibration_factor,sediment_activity_contract_code,wet_mask_provider_code
   read(unit,'(a)') capacity_diagnostic_schema
   read(unit,'(a)') suspension_profile_schema
   read(unit,'(a)') flow2dh_exchange_schema
   read(unit,'(a)') flow2dh_settling_schema
   read(unit,'(a)') suspension_profile_fingerprint
   if(.not.all(ieee_is_finite([constitutive_kinematic_viscosity, &
       suspended_adaptation_factor,settling_fac_dss,settling_temperature_c, &
       settling_reference_concentration_declaration,suspended_calibration_factor])).or. &
      constitutive_kinematic_viscosity<=0.d0.or. &
      all(capacity_fraction_basis/=[1,2]).or.suspended_adaptation_factor<=0.d0.or. &
      all(settlefrm_code/=[0,2]).or.settling_fac_dss<=0.d0.or. &
      settling_reference_concentration_declaration<=0.d0.or. &
      suspended_calibration_factor<=0.d0.or. &
      sediment_activity_contract_code/=1.or.wet_mask_provider_code/=1.or. &
      trim(capacity_diagnostic_schema)/=capacity_diagnostic_schema_expected.or. &
      trim(suspension_profile_schema)/=suspension_profile_schema_expected.or. &
      trim(flow2dh_exchange_schema)/=flow2dh_exchange_schema_expected.or. &
      trim(flow2dh_settling_schema)/=flow2dh_settling_schema_expected.or. &
      len_trim(suspension_profile_fingerprint)/=64.or. &
      verify(trim(suspension_profile_fingerprint),'0123456789abcdef')/=0) &
    error stop 'invalid v47 typed suspension declaration'
   if(settlefrm_code==0.and.(settling_temperature_mode_code/=0.or. &
      settling_temperature_c/=0.d0.or.settling_fac_dss/=1.d0.or. &
      suspended_calibration_factor/=1.d0)) &
    error stop 'inactive v47 settling formula carries active controls'
   if(settlefrm_code==2.and.(settling_temperature_mode_code/=1.or. &
      settling_temperature_c<=-20.d0)) &
    error stop 'unsupported v47 SettleFrm=2 temperature declaration'
   if((settlefrm_code==0.and.suspension_profile_fingerprint/=&
       repeat('0',64)).or.(settlefrm_code==2.and.suspension_profile_fingerprint==&
       repeat('0',64))) &
    error stop 'v47 suspension profile instance fingerprint does not match formula activation'
   typed_constitutive_declaration=.true.
   if(flow2dh_boundary_diffusion_declaration) then
    if(trim(flow2dh_v48_suspension_schema)/=trim(suspension_profile_schema).or. &
       trim(flow2dh_v48_suspension_fingerprint)/= &
        trim(suspension_profile_fingerprint)) &
     error stop 'v48 and nested v47 suspension identities differ'
   endif
   if(flow2dh_amr_ownership_declaration.and. &
      trim(suspended_amr_profile_fingerprint)/= &
       trim(suspension_profile_fingerprint)) &
    error stop 'v49 and nested suspension-profile fingerprints differ'
   read(unit,*) version
   if(version<1.or.version>46) error stop 'unsupported v47 nested deck version'
  endif
  if(version==46) then
   read(unit,*) capacity_parameter_law,capacity_parameter_fraction_count, &
    capacity_parameter_count,capacity_split_input
   read(unit,'(a)') capacity_parameter_schema
   read(unit,'(a)') capacity_profile_schema
   if(all(capacity_parameter_law/=[11,12,13,14]).or. &
      capacity_parameter_fraction_count<1.or.capacity_parameter_fraction_count>32.or. &
      capacity_parameter_count<1.or.capacity_parameter_count>capacity_parameter_maximum.or. &
      capacity_split_input/=1.or.trim(capacity_parameter_schema)/=capacity_parameter_schema_expected.or. &
      trim(capacity_profile_schema)/=capacity_profile_schema_expected) &
    error stop 'invalid v46 split-capacity declaration'
   select case(capacity_parameter_law)
   case(11)
    if(capacity_parameter_count/=10) error stop 'law 11 requires ten capacity parameters per fraction'
   case(12)
    if(capacity_parameter_count/=5) error stop 'law 12 requires five capacity parameters per fraction'
   case(13)
    if(capacity_parameter_count/=4) error stop 'law 13 requires four capacity parameters per fraction'
   case(14)
    if(capacity_parameter_count/=2) error stop 'law 14 requires two capacity parameters per fraction'
   end select
   do k=1,capacity_parameter_fraction_count
    read(unit,*) capacity_parameter_input(:capacity_parameter_count,k)
   enddo
   if(.not.all(ieee_is_finite(capacity_parameter_input(:capacity_parameter_count, &
       :capacity_parameter_fraction_count)))) error stop 'nonfinite v46 capacity parameter'
   capacity_split_retained=.true.;capacity_parameter_declaration=.true.
   read(unit,*) version
   if(version<1.or.version>45) error stop 'unsupported split-capacity base deck version'
  endif
  if(version==45) then
   read(unit,*) hydraulic_roughness_input,white_colebrook_input,sediment_shear_input, &
    roughness_minimum_work_depth,skin_kinematic_viscosity,skin_sand_roughness, &
    skin_silt_roughness,skin_critical_mud_fraction_low, &
    skin_critical_mud_fraction_high,skin_fluff_fraction
   if(all(hydraulic_roughness_input/=[braids_roughness_chezy,braids_roughness_manning, &
       braids_roughness_white_colebrook]).or.white_colebrook_input<0.or. &
      white_colebrook_input>1.or.sediment_shear_input<0.or.sediment_shear_input>1.or. &
      .not.all(ieee_is_finite([roughness_minimum_work_depth,skin_kinematic_viscosity, &
       skin_sand_roughness,skin_silt_roughness,skin_critical_mud_fraction_low, &
       skin_critical_mud_fraction_high,skin_fluff_fraction])).or. &
      roughness_minimum_work_depth<0.d0.or. &
      ((hydraulic_roughness_input/=braids_roughness_manning.or.white_colebrook_input==1).and. &
       roughness_minimum_work_depth<=0.d0).or.skin_kinematic_viscosity<=0.d0.or. &
      min(skin_sand_roughness,skin_silt_roughness)<=0.d0.or. &
      skin_critical_mud_fraction_low<0.d0.or. &
      skin_critical_mud_fraction_high<=skin_critical_mud_fraction_low.or. &
      skin_critical_mud_fraction_high>1.d0.or.skin_fluff_fraction<0.d0.or. &
      skin_fluff_fraction>1.d0.or. &
      (white_colebrook_input==1.and. &
       hydraulic_roughness_input/=braids_roughness_white_colebrook)) &
    error stop 'invalid typed hydraulic roughness or sediment-shear configuration'
   hydraulic_roughness_mode=hydraulic_roughness_input
   white_colebrook_chzk2d=white_colebrook_input==1
   sediment_shear_mode=sediment_shear_input
   typed_roughness_declaration=.true.
   read(unit,*) version
   if(version<1.or.version>44) error stop 'unsupported typed-roughness base deck version'
  endif
  if(version==44) then
   read(unit,*) gaia_interior_slope_input,gaia_interior_secondary_input,gaia_open_secondary_input
   if(any([gaia_interior_slope_input,gaia_interior_secondary_input,gaia_open_secondary_input]<0).or. &
      any([gaia_interior_slope_input,gaia_interior_secondary_input,gaia_open_secondary_input]>1)) &
    error stop 'invalid TELEMAC-GAIA v44 diagnostic switches'
   gaia_interior_slope_talmon=gaia_interior_slope_input==1
   gaia_interior_secondary_stress=gaia_interior_secondary_input==1
   gaia_open_boundary_secondary_stress=gaia_open_secondary_input==1
   read(unit,*) version
   if(version/=43) error stop 'TELEMAC-GAIA v44 diagnostic envelope requires a v43 deck'
  endif
  if(version==43) then
   gaia_declaration=.true.
   read(unit,*) gaia_skin_roughness_ratio,gaia_slope_beta,gaia_deviation_beta2,gaia_secondary_alpha
   if(.not.all(ieee_is_finite([gaia_skin_roughness_ratio,gaia_slope_beta, &
      gaia_deviation_beta2,gaia_secondary_alpha])).or.gaia_skin_roughness_ratio<=0.d0.or. &
      gaia_slope_beta<0.d0.or.gaia_deviation_beta2<=0.d0.or.gaia_secondary_alpha<=0.d0) &
    error stop 'invalid TELEMAC-GAIA transport parameters'
   secondary_flow=.true.;secondary_flow_model=3;secondary_flow_transport_direction=.true.
   read(unit,*) version
   if(version<1.or.version>41) error stop 'unsupported TELEMAC-GAIA base deck version'
  endif
  if(version==42) then
   engelund_hansen_declaration=.true.
   read(unit,*) engelund_hansen_chezy
   if(.not.ieee_is_finite(engelund_hansen_chezy).or.engelund_hansen_chezy<=0.d0) &
    error stop 'invalid constant Engelund-Hansen Chezy coefficient'
   read(unit,*) version
   if(version<1.or.version>41) error stop 'unsupported Engelund-Hansen base deck version'
  endif
  if(version==41) then
   ! v41 froze the hydraulic bank top without representing the open part of
   ! a cell. Its checkpoints cannot be interpreted as moving-bank geometry.
   ! Require a fresh bed-coupled preparation rather than silently replaying it.
   error stop 'v41 bank front retired: prepare a new run with bed-elevation bank evolution'
  endif
  if(version==40) then
   read(unit,*) bank_toe_mode_input,bank_distributed_transport_fraction, &
                bank_distributed_critical_slope,bank_critical_undercut
   if(bank_toe_mode_input/=1.or. &
      .not.all(ieee_is_finite([bank_distributed_transport_fraction, &
                               bank_distributed_critical_slope,bank_critical_undercut])).or. &
      bank_distributed_transport_fraction<0.d0.or.bank_distributed_transport_fraction>1.d0.or. &
      bank_distributed_critical_slope<=0.d0.or.bank_critical_undercut<=0.d0) &
    error stop 'invalid distributed slope-bank configuration'
   bank_failure=.true.;bank_toe_erosion_mode=3
   read(unit,*) version
   if(version<1.or.version>39) error stop 'unsupported distributed slope-bank base deck version'
  endif
  if(version==39) then
   read(unit,*) nviscous_inlets
   if(nviscous_inlets<1) error stop 'viscosity-equilibrated inlet envelope requires at least one boundary ID'
   allocate(viscous_inlet_ids(nviscous_inlets));read(unit,*) viscous_inlet_ids
   if(any(viscous_inlet_ids<1)) error stop 'invalid viscosity-equilibrated inlet boundary ID'
   read(unit,*) version
   if(version<1.or.version>38) error stop 'unsupported viscosity-equilibrated inlet base deck version'
  endif
  if(version==38) then
   if(bank_toe_erosion_mode==3) &
    error stop 'v40 distributed slope-bank deck cannot contain a v38 physical-toe declaration'
   read(unit,*) bank_toe_mode_input,bank_toe_capacity_fraction,bank_critical_undercut
   if(bank_toe_mode_input/=1.or. &
      .not.all(ieee_is_finite([bank_toe_capacity_fraction,bank_critical_undercut])).or. &
      bank_toe_capacity_fraction<0.d0.or.bank_toe_capacity_fraction>1.d0.or. &
      bank_critical_undercut<=0.d0) &
    error stop 'invalid noncohesive physical-toe configuration'
   bank_failure=.true.;bank_toe_erosion_mode=2
   read(unit,*) version
   if(version<1.or.version>37) error stop 'unsupported noncohesive physical-toe base deck version'
  endif
  if(version==37) then
   read(unit,*) secondary_model_input,secondary_direction_input,secondary_flow_espir, &
                secondary_flow_momentum_beta,secondary_inflow_input
   if(secondary_model_input/=2.or.secondary_direction_input<0.or.secondary_direction_input>1.or. &
      secondary_inflow_input/=1.or. &
      .not.all(ieee_is_finite([secondary_flow_espir,secondary_flow_momentum_beta])).or. &
      min(secondary_flow_espir,secondary_flow_momentum_beta)<0.d0) &
    error stop 'invalid prognostic secondary-flow configuration'
   secondary_flow=.true.;secondary_flow_model=2
   secondary_flow_transport_direction=secondary_direction_input==1
   secondary_flow_inflow_mode=secondary_inflow_input
   read(unit,*) version
   if(version<1.or.version>36) error stop 'unsupported prognostic secondary-flow base deck version'
  endif
  if(version==36) then
   read(unit,*) manning_friction_chezy_minimum_depth
   if(.not.ieee_is_finite(manning_friction_chezy_minimum_depth).or. &
      manning_friction_chezy_minimum_depth<=0) &
    error stop 'invalid Manning friction Chezy minimum depth (m)'
   read(unit,*) version
   if(version<1.or.version>35) error stop 'unsupported Manning friction base deck version'
  endif
  if(version==35) then
   read(unit,*) horizontal_viscosity_geometry
   if(horizontal_viscosity_geometry/=1) error stop 'invalid velocity-Laplacian geometry formulation'
   read(unit,*) version
   if(version/=34) error stop 'nonorthogonal velocity-Laplacian envelope requires a v34 deck'
  endif
  if(version==34) then
   read(unit,*) horizontal_viscosity_mode
   if(horizontal_viscosity_mode<0.or.horizontal_viscosity_mode>1) &
    error stop 'invalid horizontal viscosity formulation'
   read(unit,*) version
   if(version/=33) error stop 'velocity-Laplacian envelope requires a v33 viscosity deck'
  endif
  if(version==33) then
   read(unit,*) horizontal_viscosity
   if(.not.ieee_is_finite(horizontal_viscosity).or.horizontal_viscosity<0) &
    error stop 'invalid horizontal viscosity (m2/s)'
   read(unit,*) version
   if(version<1.or.version>32) error stop 'unsupported viscosity base deck version'
  endif
  if(horizontal_viscosity_mode==1.and.horizontal_viscosity<=0) &
   error stop 'velocity-Laplacian formulation requires positive horizontal viscosity'
  if(horizontal_viscosity_geometry==1.and.horizontal_viscosity_mode/=1) &
   error stop 'nonorthogonal geometry requires velocity-Laplacian mode 1'
  if(version==32) then
   read(unit,*) bedform_flag
   bedform_roughness=bedform_flag==1
   if(bedform_flag<0.or.bedform_flag>1) error stop 'invalid bedform-roughness configuration'
   read(unit,*) version
   if(version<1.or.version>31) error stop 'unsupported bedform-roughness base deck version'
  endif
  if(version==31) then
   morphological_acceleration=.true.
   read(unit,*) nmorfac,morphological_factor
   if(nmorfac/=0.and.nmorfac<2) error stop 'morphological-factor series requires at least two knots'
   if(.not.ieee_is_finite(morphological_factor).or.morphological_factor<1.d0.or.morphological_factor>20.d0) &
    error stop 'morphological factor must be finite and within the qualified 1 to 20 range'
   if(nmorfac>0) then
    allocate(morfac_time(nmorfac),morfac_value(nmorfac))
    do k=1,nmorfac
     read(unit,*) morfac_time(k),morfac_value(k)
    enddo
    if(.not.all(ieee_is_finite(morfac_time)).or..not.all(ieee_is_finite(morfac_value)).or. &
       morfac_time(1)<0.d0.or.any(morfac_time(2:)<=morfac_time(:nmorfac-1)).or. &
       any(morfac_value<1.d0).or.any(morfac_value>20.d0)) &
     error stop 'invalid morphological-factor time series'
   endif
   read(unit,*) version
   if(version<1.or.version>30) error stop 'unsupported morphological-factor base deck version'
  endif
  if(version==30) then
   read(unit,*) adaptation_flag,bedload_adaptation_length_diameters
   bedload_adaptation=adaptation_flag==1
   if(adaptation_flag<0.or.adaptation_flag>1.or. &
      .not.ieee_is_finite(bedload_adaptation_length_diameters).or. &
      bedload_adaptation_length_diameters<=0) error stop 'invalid bed-load adaptation configuration'
   read(unit,*) version
   if(version<1.or.version>29) error stop 'unsupported adaptation base deck version'
  endif
  if(version==29) then
   if(bank_toe_erosion_mode>=2) &
    error stop 'noncohesive bank deck cannot contain a v29 cohesive declaration'
   read(unit,*) failure_flag,bank_toe_erodibility,bank_toe_critical_shear,bank_toe_exponent,bank_critical_undercut
   bank_failure=failure_flag==1
   if(failure_flag<0.or.failure_flag>1.or. &
      .not.all(ieee_is_finite([bank_toe_erodibility,bank_toe_critical_shear,bank_toe_exponent,bank_critical_undercut])).or. &
      bank_toe_erodibility<0.or.bank_toe_critical_shear<0.or.bank_toe_exponent<=0.or.bank_critical_undercut<0.or. &
      (bank_failure.and.(bank_toe_erodibility<=0.or.bank_critical_undercut<=0))) &
    error stop 'invalid process-separated bank-failure configuration'
   if(bank_failure) bank_toe_erosion_mode=1
   read(unit,*) version
   if(version<1.or.version>28) error stop 'unsupported bank-failure base deck version'
  endif
  if(version==28) then
   if(secondary_flow_model==2) error stop 'prognostic secondary flow cannot wrap a legacy v28 declaration'
   read(unit,*) secondary_flag,secondary_flow_espir
   secondary_flow=secondary_flag==1
   if(secondary_flow) then
    secondary_flow_model=1;secondary_flow_transport_direction=.true.
   endif
   if(secondary_flag<0.or.secondary_flag>1.or..not.ieee_is_finite(secondary_flow_espir).or.secondary_flow_espir<0) &
    error stop 'invalid secondary-flow configuration'
   read(unit,*) version
   if(version<1.or.version>27) error stop 'unsupported secondary-flow base deck version'
  endif
  if(version==27) then
   read(unit,*) general_flag,general_avalanche_wet_slope,general_avalanche_dry_slope, &
                general_avalanche_time,general_avalanche_wet_depth_switch,general_avalanche_relaxation_length
   general_avalanche=general_flag==1
   if(general_flag<0.or.general_flag>1.or. &
      .not.all(ieee_is_finite([general_avalanche_wet_slope,general_avalanche_dry_slope, &
                               general_avalanche_time,general_avalanche_wet_depth_switch, &
                               general_avalanche_relaxation_length])).or. &
      min(general_avalanche_wet_slope,general_avalanche_dry_slope,general_avalanche_time, &
          general_avalanche_relaxation_length)<=0.or.general_avalanche_wet_depth_switch<0) &
    error stop 'invalid general-avalanche configuration'
   read(unit,*) version
   if(version<1.or.version>26) error stop 'unsupported general-avalanche base deck version'
  endif
  if(version==26) then
   read(unit,*) active_layer_mode,active_layer_depth_fraction,active_layer_minimum_thickness
   if(active_layer_mode<0.or.active_layer_mode>1.or. &
      .not.all(ieee_is_finite([active_layer_depth_fraction,active_layer_minimum_thickness])).or. &
      active_layer_depth_fraction<0.or.active_layer_minimum_thickness<=0.or. &
      (active_layer_mode==1.and.active_layer_depth_fraction<=0)) &
    error stop 'invalid active-layer thickness mode'
   read(unit,*) version
   if(version<1.or.version>25) error stop 'unsupported active-layer base deck version'
  endif
  if(version==25) then
   read(unit,*) hiding_parker_alpha,hiding_wu_exponent
   if(.not.all(ieee_is_finite([hiding_parker_alpha,hiding_wu_exponent]))) &
    error stop 'invalid hiding/exposure parameters'
   read(unit,*) version
   if(version<1.or.version>24) error stop 'unsupported hiding/exposure base deck version'
  endif
  if(version==24) then
   read(unit,*) binary_path,binary_digest
   open(newunit=binary_unit,file=trim(binary_path),status='old',form='unformatted',access='stream', &
        action='read',convert='little_endian')
   read(binary_unit) binary_magic
   if(binary_magic/='BRAIDS_NATIVE_V1') error stop 'invalid native binary mesh signature'
   read(binary_unit) binary_header
   if(binary_header(1)/=1) error stop 'unsupported native binary mesh version'
   read(unit,*) version
  endif
  bank_relaxation_length=1.d0
  if(version==23) then
   read(unit,*) bank_relaxation_length
   if(.not.ieee_is_finite(bank_relaxation_length).or.bank_relaxation_length<=0) &
    error stop 'invalid bank relaxation length'
   read(unit,*) version
   if(version<1.or.version>22) error stop 'unsupported enveloped cut-cell deck version'
  endif
  amr_enabled=version==22
  if(amr_enabled) then
   read(unit,*) version
   if(version<1.or.version>21) error stop 'unsupported AMR base deck version'
  endif
  if(version==20.or.version==21) then
   ! Version 20 is an envelope around any existing state layout. This lets
   ! hindered settling reject older executables without forcing a one-material
   ! case to adopt multi-fraction/layer records.
   production_section_geometry=version==21
   read(unit,*) base_version
   if(base_version<1.or.base_version>19) error stop 'unsupported enveloped base deck version'
   version=base_version
  endif
  deck_version=version
  read(unit,*) nc,nf,nb,nx,ny
  if(min(nc,nf,nx,ny)<1.or.nb<0) error stop 'invalid mesh dimensions'
  allocate(suspended_boundary_return_time(nb));suspended_boundary_return_time=0.d0
  if(flow2dh_boundary_diffusion_declaration) then
   if(flow2dh_boundary_count/=nb) &
    error stop 'v48 return-time rows do not match logical boundaries'
   do k=1,flow2dh_boundary_count
    if(flow2dh_boundary_return_id_input(k)<1.or. &
       flow2dh_boundary_return_id_input(k)>nb) &
     error stop 'v48 return-time boundary ID is out of range'
   enddo
   do c=1,nb
    if(count(flow2dh_boundary_return_id_input==c)/=1) &
     error stop 'v48 return-time boundary IDs are not a permutation'
   enddo
   do k=1,flow2dh_boundary_count
    suspended_boundary_return_time(flow2dh_boundary_return_id_input(k))= &
     flow2dh_boundary_return_time_input(k)
   enddo
  endif
  if(binary_unit/=0) then
   if(binary_header(2)/=nc.or.binary_header(3)/=nf) error stop 'native binary mesh dimensions differ'
  endif
  read(unit,*) grav,dry_tolerance,bulk,manning,ws,d50,d90,rhos,rho,z0,sedthr,dx,dy
  bedload_min_depth=sedthr ! legacy v1/v2 decks used the exchange threshold for bedload too
  if(.not.all(ieee_is_finite([grav,dry_tolerance,bulk,manning,ws,d50,d90,rhos,rho,z0,sedthr,dx,dy]))) &
   error stop 'nonfinite configuration'
  if(min(grav,dry_tolerance,bulk,d50,d90,rho,z0,dx,dy)<=0.or.min(manning,ws,sedthr)<0.or.rhos<=rho) &
   error stop 'invalid physical configuration'
  if(version>=12) then
   read(unit,*) nfrac
   if(nfrac<2.or.nfrac>32) error stop 'multi-fraction deck requires 2 to 32 fractions'
  endif
  if(channel_composition_declaration.and.channel_composition_fraction_count/=nfrac) &
   error stop 'v51 channel-composition fraction count differs from nested deck'
  if(capacity_parameter_declaration) then
   if(capacity_parameter_fraction_count/=nfrac) &
    error stop 'v46 capacity parameter rows do not match sediment fractions'
   allocate(capacity_parameter(capacity_parameter_maximum,nfrac));capacity_parameter=0.d0
   capacity_parameter(:capacity_parameter_count,:)= &
    capacity_parameter_input(:capacity_parameter_count,:nfrac)
   select case(capacity_parameter_law)
   case(11)
    if(any(capacity_parameter(1,:)<0.d0).or.any(capacity_parameter(4:5,:)<0.d0).or. &
       any(capacity_parameter(6,:)<0.d0).or.any(capacity_parameter(9:10,:)<0.d0)) &
     error stop 'invalid law-11 capacity coefficients'
   case(12)
    if(any(capacity_parameter(1,:)<0.d0).or.any(capacity_parameter(3,:)<=0.d0).or. &
       any(capacity_parameter(4,:)<0.d0)) error stop 'invalid law-12 capacity coefficients'
   case(13)
    if(any(capacity_parameter(1,:)<0.d0).or.any(capacity_parameter(2:3,:)<=0.d0)) &
     error stop 'invalid law-13 capacity coefficients'
   case(14)
    if(any(capacity_parameter(1,:)<=0.d0).or.any(capacity_parameter(2,:)<0.d0)) &
     error stop 'invalid law-14 capacity coefficients'
    end select
   else if(component_native_identity_declaration.and.capacity_route_identity_declaration.and. &
      any(trim(capacity_route_method_id)==[character(len=64) :: &
       'capacity.mpm','capacity.engelund_hansen'])) then
    ! V53 carries the exact per-fraction coefficient preimage for corrected
    ! laws 2/8.  Allocate bounded storage now; the native SHA/profile validator
    ! below fills it only after proving every selector and coefficient byte.
    allocate(capacity_parameter(capacity_parameter_maximum,nfrac))
    capacity_parameter=0.d0
   endif
  nphysical_primitive=4+nfrac;nphysical_ledger=1+2*nfrac
  nprimitive=nphysical_primitive+merge(1,0,prognostic_secondary_flow())
  nledger=nphysical_ledger+merge(1,0,prognostic_secondary_flow())+ &
   merge(2*nfrac,0,flow2dh_boundary_diffusion_declaration)
  allocate(bulk_fraction(nfrac),ws_fraction(nfrac),d50_fraction(nfrac),d90_fraction(nfrac),rhos_fraction(nfrac),z0_fraction(nfrac))
  allocate(sedthr_fraction(nfrac),bedload_min_depth_fraction(nfrac),reference_height_fraction(nfrac),sediment_roughness_fraction(nfrac))
  allocate(dstar_fraction(nfrac),taucr_fraction(nfrac))
  allocate(sediment_type_fraction(nfrac),critical_erosion_fraction(nfrac),critical_deposition_fraction(nfrac), &
   erosion_parameter_fraction(nfrac),erosion_exponent_fraction(nfrac))
  bulk_fraction=bulk;ws_fraction=ws;d50_fraction=d50;d90_fraction=d90;rhos_fraction=rhos;z0_fraction=z0
  sedthr_fraction=sedthr;bedload_min_depth_fraction=sedthr;reference_height_fraction=1;sediment_roughness_fraction=0
  sediment_type_fraction=0;critical_erosion_fraction=.2d0;critical_deposition_fraction=.1d0
  erosion_parameter_fraction=1.d-4;erosion_exponent_fraction=1.d0
  if(version>=12) then
   do k=1,nfrac
    if(version>=16) then
     read(unit,*) bulk_fraction(k),ws_fraction(k),d50_fraction(k),d90_fraction(k),rhos_fraction(k),z0_fraction(k), &
      sedthr_fraction(k),bedload_min_depth_fraction(k),reference_height_fraction(k),sediment_roughness_fraction(k), &
      sediment_type_fraction(k),critical_erosion_fraction(k),critical_deposition_fraction(k), &
      erosion_parameter_fraction(k),erosion_exponent_fraction(k)
    else
     read(unit,*) bulk_fraction(k),ws_fraction(k),d50_fraction(k),d90_fraction(k),rhos_fraction(k),z0_fraction(k), &
      sedthr_fraction(k),bedload_min_depth_fraction(k),reference_height_fraction(k),sediment_roughness_fraction(k)
    endif
   enddo
   if(.not.all(ieee_is_finite([bulk_fraction,ws_fraction,d50_fraction,d90_fraction,rhos_fraction,z0_fraction, &
      sedthr_fraction,bedload_min_depth_fraction,reference_height_fraction])).or. &
      any(bulk_fraction<=0).or.any(ws_fraction<0).or.any(d50_fraction<=0).or.any(d90_fraction<d50_fraction).or. &
      any(rhos_fraction<=rho).or.any(z0_fraction<=0).or.any(sedthr_fraction<0).or. &
      any(bedload_min_depth_fraction<0).or.any(reference_height_fraction<=0).or. &
      any(sediment_roughness_fraction<0).or.any(sediment_roughness_fraction>1)) &
    error stop 'invalid sediment fraction properties'
   if(any(sediment_type_fraction<0).or.any(sediment_type_fraction>1).or. &
      .not.all(ieee_is_finite([critical_erosion_fraction,critical_deposition_fraction,erosion_parameter_fraction, &
      erosion_exponent_fraction])).or.any(critical_erosion_fraction<=0).or.any(critical_deposition_fraction<0).or. &
      any(erosion_parameter_fraction<0).or.any(erosion_exponent_fraction<=0)) &
    error stop 'invalid cohesive sediment fraction properties'
  endif
  if(version>=13) then
   read(unit,*) active_layer_thickness
   if(.not.ieee_is_finite(active_layer_thickness).or.active_layer_thickness<=0) &
    error stop 'active-layer thickness must be positive'
  endif
  if(version>=19) then
   read(unit,*) nsubstrate_layers,substrate_layer_thickness
   if(nsubstrate_layers<1.or.nsubstrate_layers>32.or. &
      .not.ieee_is_finite(substrate_layer_thickness).or.substrate_layer_thickness<=0) &
    error stop 'invalid preserved-substrate configuration'
  else if(version>=13) then
   nsubstrate_layers=1
   substrate_layer_thickness=active_layer_thickness
  endif
  nbed_layers=1+nsubstrate_layers
  nphysical_state=3+(1+nbed_layers)*nfrac
  nstate=nphysical_state+merge(1,0,prognostic_secondary_flow())
  if(version>=14) then
   read(unit,*) hiding_exposure
   if(hiding_exposure/=0.and.(hiding_exposure<2.or.hiding_exposure>5)) &
    error stop 'unsupported hiding/exposure mode'
  endif
  if(version>=16) then
   read(unit,*) critical_mud_fraction,transport_calibration
   if(.not.all(ieee_is_finite([critical_mud_fraction,transport_calibration])).or. &
      critical_mud_fraction<0.or.critical_mud_fraction>1.or.transport_calibration<0) &
    error stop 'invalid mixed-bed configuration'
  else if(channel_composition_declaration) then
   ! A v51 envelope is the native owner of global Bed when the older nested
   ! law-11 layout predates that otherwise unrelated legacy field.
   transport_calibration=channel_bed_factor
  endif
  if(version>=17) then
   read(unit,*) bank_erosion_fraction,bank_wet_slope,bank_dry_slope,bank_avalanche_time,bank_wet_depth_switch,bank_flag
   bank_avalanche=bank_flag==1
   if(bank_flag<0.or.bank_flag>1.or. &
      .not.all(ieee_is_finite([bank_erosion_fraction,bank_wet_slope,bank_dry_slope,bank_avalanche_time, &
                               bank_wet_depth_switch])).or. &
      bank_erosion_fraction<0.or.bank_erosion_fraction>1.or.min(bank_wet_slope,bank_dry_slope)<=0.or. &
      bank_avalanche_time<=0.or.bank_wet_depth_switch<0) error stop 'invalid bank-morphology configuration'
  endif
  if(bank_failure.and.(bank_erosion_fraction>0.or.bank_avalanche)) &
   error stop 'legacy and process-separated bank morphology cannot both be enabled'
  if(version>=18) then
   read(unit,*) mpm_friction_mode
   if(mpm_friction_mode<0.or.mpm_friction_mode>1) error stop 'unsupported MPM friction mode'
  endif
  read(unit,*) endtime,dtmax,output_dt
  if(.not.all(ieee_is_finite([endtime,dtmax,output_dt])).or.min(endtime,dtmax,output_dt)<=0) &
   error stop 'invalid time configuration'
  read(unit,*) scheme,bedlaw,exchange_mode
  ! The historical nested v18 record tied corrected MPM's friction selector
  ! to the multi-fraction active-layer layout.  Q3-C v54 instead binds that
  ! closure in its authenticated scalar-capacity profile and permits a real
  ! one-row reservoir.  Recover the selector for that compact route without
  ! changing the nested state layout; the full component/forcing validator
  ! below still rejects any unauthenticated or mismatched declaration.
  if(component_native_identity_declaration.and.channel_composition_fraction_count==1.and. &
     trim(capacity_route_method_id)=='capacity.mpm'.and.bedlaw==2) &
   mpm_friction_mode=1
  if((scheme/=1.and.scheme/=5.and.scheme/=6).or. &
     .not.(bedlaw>=0.and.bedlaw<=9.or.any(bedlaw==[11,12,13,14])).or. &
     exchange_mode<0.or.exchange_mode>4) &
   error stop 'unsupported scheme/law/exchange'
  if(any(bedlaw==[11,12,13,14])) then
   if(.not.capacity_parameter_declaration.or.capacity_parameter_law/=bedlaw.or. &
      .not.capacity_split_retained) &
    error stop 'laws 11-14 require a matching v46 retained split-capacity declaration'
   if(exchange_mode/=0.and..not.(bedlaw==11.and.exchange_mode==4)) &
    error stop 'split capacity requires diagnostic mode 0 or law-11 generic FLOW exchange mode 4'
  else if(capacity_parameter_declaration) then
   error stop 'v46 capacity parameters do not match the selected bed-load law'
  endif
   if(channel_composition_declaration) then
    if(all(bedlaw/=[2,8,11])) &
     error stop 'v51 channel composition requires scalar capacity law 2, 8, or 11'
    if(bedlaw==11.and.(.not.capacity_parameter_declaration.or. &
       capacity_parameter_law/=11.or..not.capacity_split_retained)) &
     error stop 'v51 law-11 composition requires its retained split-capacity path'
    if(any(bedlaw==[2,8]).and..not.component_native_identity_declaration) &
     error stop 'corrected law-2/law-8 composition requires authenticated v53 components'
     if(bedlaw==11.and..not.component_native_identity_declaration.and. &
        (channel_bed_factor/=transport_calibration.or. &
        channel_suspended_factor/=suspended_calibration_factor)) &
     error stop 'v51 law-11 factors differ from the nested global Bed/Sus controls'
    if(exchange_mode==0) then
    if(any(channel_transport_class_input(:nfrac)/=BRAIDS_TRA_BEDLOAD)) &
     error stop 'v51 bedload-only exchange requires transport class 1'
    else if(exchange_mode==4.and.bedlaw==11) then
    if(any(channel_transport_class_input(:nfrac)/=BRAIDS_TRA_COMBINE)) &
     error stop 'v51 combined exchange requires transport class 3'
   else
     error stop 'v51 scalar-capacity composition supports exchange 0; law 11 also supports 4'
   endif
  endif
  if(exchange_mode==4) then
   if(.not.typed_constitutive_declaration.or.bedlaw/=11.or.nfrac/=1.or. &
      any(sediment_type_fraction/=0).or.settlefrm_code/=2.or. &
      (amr_enabled.and..not.flow2dh_amr_ownership_declaration)) &
    error stop 'generic FLOW exchange requires law 11, one sand fraction, SettleFrm=2, and v49 for AMR'
   if(capacity_parameter(6,1)<=0.d0) &
    error stop 'generic FLOW exchange requires positive SusACal for every fraction'
  endif
  if(flow2dh_amr_ownership_declaration) then
   if(.not.flow2dh_boundary_diffusion_declaration.or.exchange_mode/=4.or. &
      suspended_amr_fraction_count_input/=nfrac.or. &
      suspended_amr_enabled_input/=merge(1,0,amr_enabled).or. &
      suspended_amr_dry_tolerance_input/=dry_tolerance.or. &
      suspended_amr_diffusion_dry_threshold_input/= &
       flow2dh_diffusion_dry_threshold) &
    error stop 'v49 ownership contract differs from the nested suspension configuration'
   if(prognostic_secondary_flow()) &
    error stop 'v49 suspended ownership is not qualified with prognostic secondary flow'
  endif
  if(flow2dh_boundary_diffusion_declaration.and.exchange_mode/=4) &
   error stop 'v48 boundary/diffusion requires generic FLOW exchange mode 4'
  if(bank_toe_erosion_mode>=2.and.(bedlaw==0.or.exchange_mode/=0)) &
   error stop 'noncohesive bank erosion requires bed load without suspended exchange'
  if(version<16.and..not.capacity_parameter_declaration.and. &
     ((bedlaw>2.and.bedlaw/=8).or.exchange_mode>2)) &
   error stop 'extended sediment laws require deck version 16'
  if(morphological_acceleration.and.bedlaw==0) &
   error stop 'morphological acceleration requires an active bed-load formulation'
  if(morphological_acceleration.and.exchange_mode/=0) &
   error stop 'bed-load morphological acceleration is not qualified with suspended exchange'
  if(morphological_acceleration.and.(bank_erosion_fraction>0.or.bank_avalanche.or.bank_failure)) &
   error stop 'morphological acceleration is not qualified with bank erosion or bank failure'
  if(version==2) read(unit,*) mpm_cf,mpm_critical
  if(version>=3) then
   read(unit,*) mpm_cf,mpm_critical
   read(unit,*) bed_slope,bed_slope_alfabs,bed_slope_alfabn,bed_friction_angle_deg,bedload_min_depth
  endif
  if(version==1.and.bedlaw==2) error stop 'MPM requires explicit v2 parameters'
  if(mpm_friction_mode/=0.and.bedlaw/=2) error stop 'roughness-corrected MPM mode requires MPM bed load'
  if(hiding_exposure/=0.and.all(bedlaw/=[2,11])) &
   error stop 'hiding/exposure requires MPM or v46 Delft3D general capacity'
  if(.not.all(ieee_is_finite([mpm_cf,mpm_critical])).or.min(mpm_cf,mpm_critical)<0) &
   error stop 'invalid MPM parameters'
  if(mpm_friction_mode==1.and.abs(mpm_critical-.047d0)>1.d-14) &
   error stop 'roughness-corrected MPM requires critical Shields 0.047'
  if(capacity_route_identity_declaration) then
   select case(trim(capacity_route_method_id))
   case('capacity.engelund_hansen_constant_chezy_legacy')
    if(bedlaw/=8.or.mpm_friction_mode/=0.or..not.engelund_hansen_declaration.or. &
       capacity_route_acal/=transport_calibration) &
     error stop 'v50 legacy Engelund-Hansen identity differs from nested physics'
    case('capacity.engelund_hansen')
     if(bedlaw/=8.or.mpm_friction_mode/=0.or..not.component_native_identity_declaration.or. &
        hiding_exposure/=0.or.sediment_shear_mode/=0) &
      error stop 'corrected Delft Engelund-Hansen requires authenticated v53 selected-Chezy physics'
   case('capacity.mpm_specified_cf_legacy')
    if(bedlaw/=2.or.mpm_friction_mode/=0.or.capacity_route_acal/=1.d0) &
     error stop 'v50 legacy specified-Cf MPM identity differs from nested physics'
    case('capacity.mpm')
    if(bedlaw/=2.or.mpm_friction_mode/=1.or.abs(mpm_critical-.047d0)>1.d-14) &
     error stop 'v50 corrected Delft MPM identity differs from nested physics'
     if((capacity_route_acal/=1.d0.and..not.component_native_identity_declaration).or. &
        (hiding_exposure/=0.and..not.hiding_foundation_identity_declaration)) &
      error stop 'v50 corrected Delft MPM requires authenticated ACal and hiding none'
    if(sediment_shear_mode/=0) &
     error stop 'corrected Delft MPM consumes total hydraulic roughness, not RP_TAUB'
    case('capacity.general')
     if(bedlaw/=11.or.mpm_friction_mode/=0.or. &
        (hiding_exposure/=0.and..not.hiding_foundation_identity_declaration).or. &
        sediment_shear_mode/=0) &
      error stop 'v50 Q3-C general identity differs from nested physics'
    case default
    error stop 'unsupported v50 capacity method identity'
   end select
   if(capacity_foundation_route_identity_declaration.and. &
      .not.component_native_identity_declaration) &
    error stop 'Q3-C forcing execution identity requires authenticated v53 components'
   if(capacity_foundation_route_identity_declaration.and. &
      (.not.capacity_foundation_forcing_identity_declaration.or. &
       (envelope_version/=54.and.envelope_version/=55.and. &
        envelope_version/=56))) &
    error stop 'Q3-C forcing execution identity requires authenticated v54 raw forcing'
   if((.not.capacity_foundation_route_identity_declaration).and. &
      trim(capacity_route_method_id)=='capacity.general') &
    error stop 'general Q3-C route cannot use a historical execution identity'
  endif
  if((bed_slope/=0.and.bed_slope/=2.and.bed_slope/=3.and.bed_slope/=4.and.bed_slope/=5).or. &
   .not.all(ieee_is_finite([bed_slope_alfabs,bed_slope_alfabn,bed_friction_angle_deg])).or. &
   min(bed_slope_alfabs,bed_slope_alfabn,bed_friction_angle_deg)<0.or. &
   (any(bed_slope==[2,5]).and.(bed_friction_angle_deg<=0.or.bed_friction_angle_deg>=90))) &
   error stop 'invalid bed-slope parameters'
  if(.not.ieee_is_finite(bedload_min_depth).or.bedload_min_depth<0) error stop 'invalid bedload depth threshold'
  if(bed_slope/=0.and.bedlaw==0) error stop 'bed-slope correction requires an active bed-load formulation'
  if(bed_slope==3.and.all(bedlaw/=[2,8])) &
   error stop 'Koch-Flokstra slope mode requires MPM or Engelund-Hansen absolute hydraulic Shields'
  if(bed_slope==5.and.all(bedlaw/=[2,8])) &
   error stop 'Delft3D FLOW slope mode 5 requires the positive MPM or Engelund-Hansen formula branch'
  if(bed_slope==4.and.bedlaw/=9) &
   error stop 'TELEMAC-GAIA slope mode 4 requires bed-load law 9'
  if(any(bedlaw==[11,12,13,14]).and.bed_slope/=0) &
   error stop 'v46 M1 split-capacity route requires bed-slope correction off'
   if(bedlaw==8.and.trim(capacity_route_method_id)/='capacity.engelund_hansen'.and. &
      (.not.engelund_hansen_declaration.or.engelund_hansen_chezy<=0.d0)) &
    error stop 'Engelund-Hansen law 8 requires a v42 constant Chezy coefficient'
  if(bedlaw==8.and.exchange_mode/=0) &
   error stop 'Engelund-Hansen law 8 is qualified only as bedload-only transport'
   if((bedlaw/=8.or.trim(capacity_route_method_id)=='capacity.engelund_hansen').and. &
      engelund_hansen_declaration) &
    error stop 'v42 constant Chezy is reserved for Engelund-Hansen law 8'
  if(bedlaw==9.and.(.not.gaia_declaration.or.scheme/=6.or.exchange_mode/=0.or.bed_slope/=4.or. &
     secondary_flow_model/=3.or..not.secondary_flow_transport_direction.or. &
     secondary_flow_momentum_beta/=0.d0.or.mpm_friction_mode/=0.or.hiding_exposure/=0.or. &
     bedload_adaptation.or.bedform_roughness.or.transport_calibration/=1.d0.or.amr_enabled.or. &
     morphological_acceleration.or.bank_failure.or.bank_avalanche.or.general_avalanche.or. &
     bank_erosion_fraction/=0.d0.or.horizontal_viscosity/=0.d0.or.active_layer_mode/=0.or. &
     nsubstrate_layers/=1.or.any(sediment_type_fraction/=0))) &
   error stop 'TELEMAC-GAIA law 9 requires the exact v43 bedload-only slope4/model3 scope'
  if(bedlaw/=9.and.gaia_declaration) &
   error stop 'v43/v44 TELEMAC-GAIA parameters are reserved for bed-load law 9'
  if(.not.typed_roughness_declaration.and. &
     (hydraulic_roughness_mode/=braids_roughness_manning.or. &
      white_colebrook_chzk2d.or.sediment_shear_mode/=0)) &
   error stop 'typed roughness state requires the v45 envelope'
  if(sediment_shear_mode==1.and.bedlaw==9) &
   error stop 'Delft3D BsSkin cannot replace the TELEMAC-GAIA skin correction'
  if(sediment_shear_mode==1.and.all(bedlaw/=[5,6,7]).and. &
     .not.(exchange_mode==3.and.any(sediment_type_fraction==1))) &
   error stop 'Soulsby/Clarke skin shear has no active RP_TAUB consumer'
  if(secondary_flow_model==3.and.bedlaw/=9) &
   error stop 'TELEMAC-GAIA secondary model 3 requires bed-load law 9'
  if(secondary_flow_model==1.and.all(bedlaw/=[2,3,5,6,7,8,11,12,14])) &
   error stop 'legacy secondary flow requires a magnitude-based bed-load formulation'
  if(prognostic_secondary_flow().and.secondary_flow_transport_direction.and. &
     all(bedlaw/=[1,2,3,5,6,7,8,11,12,14])) &
   error stop 'prognostic secondary-flow direction requires a supported bed-load formulation'
  if(bedload_adaptation.and.bedlaw==0) error stop 'bed-load adaptation requires an active bed-load formulation'
  if(bedform_roughness.and.(bedlaw/=2.or.mpm_friction_mode/=1)) &
   error stop 'bedform roughness requires roughness-corrected MPM bed load'
  if(bedform_roughness.and.exchange_mode/=0) &
   error stop 'bedform roughness is not qualified with suspended exchange'
  if(bedform_roughness.and.(secondary_flow.or.bedload_adaptation.or.morphological_acceleration.or. &
     bank_erosion_fraction>0.or.bank_avalanche.or.bank_failure.or.general_avalanche)) &
   error stop 'bedform roughness is not qualified with another experimental morphology process'
  if(bedform_roughness.and.any(sediment_type_fraction/=0)) &
   error stop 'bedform roughness requires noncohesive sediment fractions'
   if(any(sediment_type_fraction==1).and.exchange_mode/=3) &
    error stop 'cohesive fractions require mixed cohesive exchange mode'
   if(component_native_identity_declaration) call validate_v53_total_execution_profile( &
    engelund_hansen_declaration,capacity_parameter_declaration)
   if(capacity_foundation_runtime_declaration.and. &
      (.not.capacity_route_identity_declaration.or. &
       .not.capacity_foundation_route_identity_declaration)) &
    error stop 'v53 Q3-C capacity requires its ordered forcing/profile identity'
   allocate(area(nc),xy(2,nc),rock(nc),q(nstate,nc),ii(nc),jj(nc),grid(nx,ny),manning_cell(nc));q=0.d0
  manning_cell=manning
  allocate(hydro_constant(nc),active_cell(nc),bank_erodible(nc));hydro_constant=.false.;active_cell=.true.;bank_erodible=.false.
  allocate(owner(nf),neighbour(nf),bid(nf),normal(2,nf),length(nf),mid(2,nf))
  allocate(boundary_bed_offset(nf),boundary_bed_left_offset(nf),boundary_bed_right_offset(nf),sampled_boundary_bed(nf))
  boundary_bed_offset=0;boundary_bed_left_offset=0;boundary_bed_right_offset=0;sampled_boundary_bed=0
  allocate(sampled_boundary_open(nf),boundary_seed_position(nb));sampled_boundary_open=.true.;boundary_seed_position=0
  allocate(section_control(nb),section_seed_node(nb),section_energy_slope(nb))
  allocate(section_cached_stage(nb),section_cached_discharge(nb),section_cached_bed(nf),section_cache_valid(nb))
  section_control=0;section_seed_node=0;section_energy_slope=0
  section_cached_stage=0;section_cached_discharge=0;section_cached_bed=0;section_cache_valid=.false.
  allocate(mode(nb),forcing(6,nb),ledger(nledger,nb),offset(nc+1),boundary_offset(nb+1))
  allocate(viscosity_equilibrated_inlet(nb));viscosity_equilibrated_inlet=.false.
  if(nviscous_inlets>0) then
   if(any(viscous_inlet_ids>nb)) error stop 'viscosity-equilibrated inlet boundary ID exceeds boundary count'
   do k=1,nviscous_inlets
    if(count(viscous_inlet_ids==viscous_inlet_ids(k))/=1) &
     error stop 'duplicate viscosity-equilibrated inlet boundary ID'
    viscosity_equilibrated_inlet(viscous_inlet_ids(k))=.true.
   enddo
   deallocate(viscous_inlet_ids)
  endif
  allocate(reversal_policy(nb),nseries(nb),nhydro_series(nb),bed_boundary_mode(nb))
  grid=0;ledger=0;boundary_offset=0;reversal_policy=1;nseries=0;nhydro_series=0;bed_boundary_mode=0;forcing=0
  if(binary_unit/=0) then
   if(binary_header(4)/=nphysical_state.or.binary_header(7)/=version.or. &
      binary_header(8)/=merge(1,0,production_section_geometry).or. &
      binary_header(9)/=merge(1,0,amr_enabled)) error stop 'native binary mesh metadata differ'
    allocate(binary_bank(nc),physical_q(nphysical_state,nc))
   read(binary_unit) ii
   read(binary_unit) jj
   read(binary_unit) area
   read(binary_unit) xy
   read(binary_unit) rock
    read(binary_unit) physical_q
    q(1:nphysical_state,:)=physical_q
    deallocate(physical_q)
   read(binary_unit) manning_cell
   read(binary_unit) binary_bank
   if(any(binary_bank<0).or.any(binary_bank>1)) error stop 'invalid binary erodible-bank cell mask'
   bank_erodible=binary_bank==1
   deallocate(binary_bank)
  endif
  if(general_avalanche.and.(version<13.or.active_layer_thickness<=0)) &
   error stop 'general avalanching requires a layered multi-fraction bed'
  do c=1,nc
   if(binary_unit==0) then
    if(version>=17) then
      read(unit,*) ii(c),jj(c),area(c),xy(:,c),rock(c),q(1:nphysical_state,c),manning_cell(c),bank_flag
     if(bank_flag<0.or.bank_flag>1) error stop 'invalid erodible-bank cell mask'
     bank_erodible(c)=bank_flag==1
    else if(version>=5) then
      read(unit,*) ii(c),jj(c),area(c),xy(:,c),rock(c),q(1:nphysical_state,c),manning_cell(c)
    else
      read(unit,*) ii(c),jj(c),area(c),xy(:,c),rock(c),q(1:nphysical_state,c)
    endif
   endif
   if(ii(c)<1.or.ii(c)>nx.or.jj(c)<1.or.jj(c)>ny.or.area(c)<=0) error stop 'invalid cell'
   if(grid(ii(c),jj(c))/=0) then
    grid(ii(c),jj(c))=-1 ! multiple disconnected components: no Cartesian WENO stencil
   else
    grid(ii(c),jj(c))=c
   endif
  enddo
  if(capacity_foundation_forcing_identity_declaration) &
   call validate_capacity_foundation_forcing_identity(envelope_version)
  if(hiding_foundation_identity_declaration) &
   call validate_hiding_foundation_identity(envelope_version)
  if(secondary_foundation_identity_declaration) &
   call validate_secondary_foundation_identity(envelope_version)
  offset=0
  if(binary_unit/=0) then
   read(binary_unit) owner
   read(binary_unit) neighbour
   read(binary_unit) bid
   read(binary_unit) normal
   read(binary_unit) length
   read(binary_unit) mid
   read(binary_unit) sampled_boundary_bed
   read(binary_unit) boundary_bed_offset
   read(binary_unit) boundary_bed_left_offset
   read(binary_unit) boundary_bed_right_offset
  endif
  do f=1,nf
   if(binary_unit==0) then
    if(version>=11) then
     if(production_section_geometry) then
      read(unit,*) owner(f),neighbour(f),bid(f),normal(:,f),length(f),mid(:,f),sampled_boundary_bed(f), &
       boundary_bed_offset(f),boundary_bed_left_offset(f),boundary_bed_right_offset(f)
     else
      read(unit,*) owner(f),neighbour(f),bid(f),normal(:,f),length(f),mid(:,f),sampled_boundary_bed(f),boundary_bed_offset(f)
     endif
    else
     read(unit,*) owner(f),neighbour(f),bid(f),normal(:,f),length(f),mid(:,f)
    endif
   endif
   c=owner(f);k=neighbour(f)
   if(c<1.or.c>nc.or.k<0.or.k>nc.or.c==k.or.length(f)<=0) error stop 'invalid face'
   if(abs(sum(normal(:,f)**2)-1)>1.d-10) error stop 'non-unit face normal'
   if(bid(f)<0.or.bid(f)>nb.or.(k>0.and.bid(f)/=0)) error stop 'invalid boundary ID'
   if(bid(f)>0) boundary_offset(bid(f))=boundary_offset(bid(f))+1
   offset(c)=offset(c)+1
   if(k>0) offset(k)=offset(k)+1
  enddo
  do k=1,nb
   if(version>=6) then
    read(unit,*) mode(k),forcing(:,k) ! + free-weir coefficient and effective width
   else
    read(unit,*) mode(k),forcing(:4,k) ! v1/v2 bed: kg/(m s); v3+ bed: total kg/s for the ID
   endif
   if(mode(k)<1.or.(mode(k)>merge(8,merge(5,3,version>=6),version>=9).and. &
    .not.(version>=6.and.mode(k)==9)).or.forcing(1,k)<0.or.min(forcing(3,k),forcing(4,k))<0) &
    error stop 'invalid boundary forcing'
   if(mode(k)>1.and.mode(k)/=6.and.mode(k)/=7.and.mode(k)/=8.and.mode(k)/=9.and. &
    (forcing(1,k)/=0.or.forcing(4,k)/=0)) error stop 'outlet cannot prescribe Q/bed input'
   if(mode(k)==5.and.(forcing(5,k)<=0.or.forcing(6,k)<=0)) error stop 'weir needs positive coefficient and width'
   if(.not.any(bid==k)) error stop 'unused boundary ID'
   if(viscosity_equilibrated_inlet(k).and.mode(k)/=6) &
    error stop 'viscosity-equilibrated inlet profile requires mode 6 total discharge'
  enddo
  if(any(viscosity_equilibrated_inlet).and. &
     (horizontal_viscosity<=0.d0.or.horizontal_viscosity_mode/=1)) &
   error stop 'viscosity-equilibrated inlet requires positive velocity-Laplacian viscosity'
  allocate(fraction_concentration(nfrac,nb),fraction_bed_input(nfrac,nb))
  fraction_concentration=0;fraction_bed_input=0
  if(version>=12) then
   do k=1,nb
    read(unit,*) fraction_concentration(:,k)
    read(unit,*) fraction_bed_input(:,k)
   enddo
  else if(nb>0) then
   fraction_concentration(1,:)=forcing(3,:);fraction_bed_input(1,:)=forcing(4,:)
  endif
  allocate(nfraction_series(nfrac,nb));nfraction_series=0
  if(version>=15.and.nb>0) then
   read(unit,*) nfraction_series
   if(any(nfraction_series<0)) error stop 'negative fraction boundary series length'
  endif
  maxfractionseries=1
  if(nb>0) maxfractionseries=max(1,maxval(nfraction_series))
  allocate(fraction_series_time(maxfractionseries,nfrac,nb), &
           fraction_series_concentration(maxfractionseries,nfrac,nb), &
           fraction_series_bed_input(maxfractionseries,nfrac,nb))
  fraction_series_time=0;fraction_series_concentration=0;fraction_series_bed_input=0
  if(version>=15) then
   do k=1,nb
    do f=1,nfrac
     if(nfraction_series(f,k)>0.and.mode(k)/=1.and.mode(k)/=6.and.mode(k)/=7.and.mode(k)/=8.and.mode(k)/=9.and. &
        .not.flow2dh_boundary_diffusion_declaration) &
      error stop 'fraction sediment time series require an inlet ID'
     do p=1,nfraction_series(f,k)
      read(unit,*) fraction_series_time(p,f,k),fraction_series_concentration(p,f,k),fraction_series_bed_input(p,f,k)
      if(.not.all(ieee_is_finite([fraction_series_time(p,f,k),fraction_series_concentration(p,f,k), &
         fraction_series_bed_input(p,f,k)])).or.min(fraction_series_time(p,f,k), &
         fraction_series_concentration(p,f,k),fraction_series_bed_input(p,f,k))<0) &
       error stop 'invalid fraction boundary series value'
      if(p>1) then
       if(fraction_series_time(p,f,k)<=fraction_series_time(p-1,f,k)) &
        error stop 'fraction boundary series times are not increasing'
      endif
     enddo
     if(nfraction_series(f,k)==1) error stop 'fraction boundary series requires at least two points'
    enddo
   enddo
  endif
  if(version>=4) then
   if(version==4.and.nb==0) error stop 'v4 time series require a boundary ID'
   if(nb>0) then
    read(unit,*) reversal_policy
    read(unit,*) nseries
   endif
   if(flow2dh_boundary_diffusion_declaration) then
    if(any(reversal_policy<0).or.any(reversal_policy>1)) &
     error stop 'unsupported v48 local-reversal policy'
   else
    if(any(reversal_policy/=1)) error stop 'unsupported local-reversal policy'
   endif
   if(any(nseries<0)) error stop 'negative boundary series length'
  endif
  maxseries=1
  if(nb>0) maxseries=max(1,maxval(nseries))
  allocate(series_time(maxseries,nb),series_concentration(maxseries,nb),series_bed_input(maxseries,nb))
  series_time=0;series_concentration=0;series_bed_input=0
  if(version>=4) then
   do k=1,nb
    if(nseries(k)>0.and.mode(k)/=1.and.mode(k)/=6.and.mode(k)/=7.and.mode(k)/=8.and.mode(k)/=9) &
     error stop 'sediment time series require an inlet ID'
    do c=1,nseries(k)
     read(unit,*) series_time(c,k),series_concentration(c,k),series_bed_input(c,k)
     if(.not.all(ieee_is_finite([series_time(c,k),series_concentration(c,k),series_bed_input(c,k)])) &
      .or.min(series_time(c,k),series_concentration(c,k),series_bed_input(c,k))<0) &
      error stop 'invalid boundary series value'
     if(c>1) then
      if(series_time(c,k)<=series_time(c-1,k)) error stop 'boundary series times are not increasing'
     endif
    enddo
    if(nseries(k)==1) error stop 'boundary series requires at least two points'
    if(nseries(k)>0) then
     if(version<7.and.maxval(series_bed_input(:nseries(k),k))>0.and.forcing(1,k)<=0) &
      error stop 'positive bed series requires positive inlet discharge'
    endif
   enddo
  endif
  if(version>=12.and.any(nseries>0)) error stop 'multi-fraction sediment time series are not implemented'
  maxhydro=1
  if(version>=7) then
   if(nb>0) read(unit,*) nhydro_series
   if(any(nhydro_series<0)) error stop 'negative hydraulic series length'
   if(nb>0) maxhydro=max(1,maxval(nhydro_series))
  endif
  allocate(hydro_series_time(maxhydro,nb),hydro_series_q(maxhydro,nb),hydro_series_stage(maxhydro,nb))
  hydro_series_time=0;hydro_series_q=0;hydro_series_stage=0
  if(version>=7) then
   do k=1,nb
    if(nhydro_series(k)>0.and.mode(k)/=1.and.mode(k)/=2.and.mode(k)/=6.and.mode(k)/=7.and.mode(k)/=8.and.mode(k)/=9) &
     error stop 'hydraulic series require an inlet or stage outlet'
    do c=1,nhydro_series(k)
     read(unit,*) hydro_series_time(c,k),hydro_series_q(c,k),hydro_series_stage(c,k)
     if(.not.all(ieee_is_finite([hydro_series_time(c,k),hydro_series_q(c,k),hydro_series_stage(c,k)])) &
      .or.min(hydro_series_time(c,k),hydro_series_q(c,k))<0) error stop 'invalid hydraulic series value'
     if(c>1) then
      if(hydro_series_time(c,k)<=hydro_series_time(c-1,k)) error stop 'hydraulic series times are not increasing'
     endif
    enddo
    if(nhydro_series(k)==1) error stop 'hydraulic series requires at least two points'
    if(mode(k)==2.and.nhydro_series(k)>0) then
     if(any(hydro_series_q(:nhydro_series(k),k)/=0)) error stop 'stage-outlet series cannot prescribe discharge'
    endif
    if((forcing(4,k)>0.or.(nseries(k)>0.and.maxval(series_bed_input(:nseries(k),k))>0)).and. &
       max(forcing(1,k),maxval(hydro_series_q(:,k)))<=0) error stop 'positive bed forcing requires positive hydraulic discharge'
   enddo
  endif
  if(version>=8.and.nb>0) then
   read(unit,*) bed_boundary_mode
   if(any(bed_boundary_mode<0).or.any(bed_boundary_mode>2)) error stop 'invalid bed-load boundary mode'
   do k=1,nb
    if(bed_boundary_mode(k)==1) then
     if(mode(k)/=1.and.mode(k)/=6.and.mode(k)/=7.and.mode(k)/=8.and.mode(k)/=9) &
      error stop 'equilibrium bed-load boundary requires an inlet ID'
     if(bedlaw==0) error stop 'equilibrium bed-load boundary requires an enabled law'
     if(forcing(4,k)/=0.or.nseries(k)>0.or.any(fraction_bed_input(:,k)>0).or. &
       any(fraction_series_bed_input(:,:,k)>0)) &
      error stop 'equilibrium bed-load boundary conflicts with prescribed input'
    else if(bed_boundary_mode(k)==2) then
     if(bedlaw/=9.or.mode(k)/=6) &
      error stop 'prescribed fixed-face bed load requires TELEMAC-GAIA mode 6 inlet'
     if(.not.any(fraction_bed_input(:,k)>0).and. &
        .not.any(fraction_series_bed_input(:,:,k)>0)) &
      error stop 'prescribed fixed-face bed load requires positive class feed'
    endif
   enddo
  endif
  if(bedlaw==9) then
   allocate(fixed_boundary_bed_mode(nb),fixed_boundary_bed(nf))
   fixed_boundary_bed_mode=0;fixed_boundary_bed=0.d0
   do k=1,nb
    if(bed_boundary_mode(k)==0) cycle
    if(mode(k)/=6) error stop 'TELEMAC-GAIA fixed-face inlet requires mode 6 total discharge'
    fixed_boundary_bed_mode(k)=1
   enddo
  endif
  do k=1,nb
   if(mode(k)/=1.and.mode(k)/=6.and.mode(k)/=7.and.mode(k)/=8.and.mode(k)/=9.and. &
      (any(fraction_concentration(:,k)>0).or.any(fraction_bed_input(:,k)>0).or.any(nfraction_series(:,k)>0)).and. &
      .not.flow2dh_boundary_diffusion_declaration) &
    error stop 'outlet cannot prescribe sediment input'
   if(mode(k)/=1.and.mode(k)/=6.and.mode(k)/=7.and.mode(k)/=8.and.mode(k)/=9.and. &
      flow2dh_boundary_diffusion_declaration.and. &
      (any(fraction_bed_input(:,k)>0).or.any(fraction_series_bed_input(:,:,k)>0))) &
    error stop 'v48 outlet cannot prescribe bed-sediment input'
   if((any(fraction_bed_input(:,k)>0).or.any(fraction_series_bed_input(:,:,k)>0)).and. &
      max(forcing(1,k),maxval(hydro_series_q(:,k)))<=0) &
    error stop 'positive fraction bed forcing requires positive hydraulic discharge'
  enddo
  do k=1,nb
   if(mode(k)/=8) cycle
   if(nhydro_series(k)==0) then
    if(forcing(1,k)>0.and.forcing(2,k)<=dry_tolerance) &
     error stop 'positive discharge at a dry prescribed depth'
   else
    if(any(hydro_series_q(:nhydro_series(k),k)>0.and. &
           hydro_series_stage(:nhydro_series(k),k)<=dry_tolerance)) &
     error stop 'positive discharge at a dry prescribed depth'
   endif
  enddo
  if(version>=10) read(unit,*) reference_height_factor,sediment_roughness_mode
  if(.not.ieee_is_finite(reference_height_factor).or.reference_height_factor<=0) &
   error stop 'invalid reference-height factor'
  if(sediment_roughness_mode<0.or.sediment_roughness_mode>1) error stop 'invalid sediment roughness mode'
  if(sediment_roughness_mode==1.and.any(manning_cell<=0)) &
   error stop 'hydraulic-derived sediment roughness requires positive roughness parameters'
  if(any(bedlaw==[8,9,11,14]).and.any(manning_cell<=0)) &
   error stop 'selected capacity law requires positive hydraulic roughness parameters'
  if(version<12) then
   reference_height_fraction(1)=reference_height_factor
   sediment_roughness_fraction(1)=sediment_roughness_mode
   bedload_min_depth_fraction(1)=bedload_min_depth
  else if(any(sediment_roughness_fraction==1).and.any(manning_cell<=0)) then
   error stop 'hydraulic-derived fraction roughness requires positive roughness parameters'
  endif
  cfl_target=.45d0;settling_mode=0;hindered_settling_reference_concentration=1.d4
  ! Optional footer: CFL, settling treatment, and the total-concentration
  ! reference. Older decks may omit it or contain CFL alone.
  read(unit,*,iostat=ios) cfl_target,settling_mode,hindered_settling_reference_concentration
  if(ios>0.or..not.ieee_is_finite(cfl_target).or.cfl_target<=0.or.cfl_target>.9d0) &
   error stop 'invalid CFL limit'
  if(settling_mode<0.or.settling_mode>1.or. &
     .not.ieee_is_finite(hindered_settling_reference_concentration).or. &
     hindered_settling_reference_concentration<=0) error stop 'invalid hindered-settling configuration'
  if(typed_constitutive_declaration.and. &
     hindered_settling_reference_concentration/= &
      settling_reference_concentration_declaration) &
   error stop 'nested settling reference concentration differs from v47 declaration'
  if(exchange_mode==4) then
   if(settling_mode/=1) &
    error stop 'generic FLOW exchange requires mode-1 total-concentration hindering'
   call flow2dh_settlefrm2_clear_velocity(d50_fraction(1),settling_fac_dss, &
    rhos_fraction(1),rho,settling_temperature_c,grav,expected_settling_velocity, &
    settling_valid)
   if(.not.settling_valid) error stop 'invalid v47 SettleFrm=2 physical inputs'
   settling_tolerance=64.d0*epsilon(1.d0)*max(1.d0,abs(expected_settling_velocity))
   if(abs(ws_fraction(1)-expected_settling_velocity)>settling_tolerance) &
    error stop 'nested clear-water settling velocity differs from v47 SettleFrm=2 declaration'
  endif
  if(production_section_geometry) then
   do k=1,nb
    read(unit,*) section_control(k),section_energy_slope(k)
   enddo
   if(any(section_control<0).or.any(section_control>2).or. &
      .not.all(ieee_is_finite(section_energy_slope)).or. &
      any(section_control==1.and.section_energy_slope<=0)) error stop 'invalid live section control'
  endif
  if(bedform_roughness.and.production_section_geometry.and.any(section_control==1)) &
   error stop 'bedform roughness is not qualified with normal-depth inlet control'
  if(.not.all(ieee_is_finite(boundary_bed_offset)).or.any(sampled_boundary_bed<0).or.any(sampled_boundary_bed>1)) &
   error stop 'invalid sampled inlet terrain'
  if(.not.all(ieee_is_finite(boundary_bed_left_offset)).or. &
     .not.all(ieee_is_finite(boundary_bed_right_offset))) error stop 'invalid production section terrain'
  do f=1,nf
   if(sampled_boundary_bed(f)==0) then
    if(boundary_bed_offset(f)/=0.or.boundary_bed_left_offset(f)/=0.or. &
       boundary_bed_right_offset(f)/=0) error stop 'unselected boundary terrain offset'
    cycle
   endif
   if(neighbour(f)>0.or.bid(f)<=0) error stop 'sampled terrain requires an open inlet face'
   if(mode(bid(f))/=1.and.mode(bid(f))/=9) error stop 'sampled terrain requires a section-controlled inlet'
  enddo
  if(amr_enabled) call amr_read_tree(unit,binary_unit)
  if(flow2dh_amr_ownership_declaration) then
   if(amr_enabled) then
    if(suspended_amr_levels_input/=amr_levels.or. &
       suspended_amr_ratio_input/=amr_ratio.or. &
       suspended_amr_wet_depth_input/=amr_wet_depth.or. &
       suspended_amr_regrid_interval_input/=amr_interval) &
     error stop 'v49 ownership contract differs from the adaptive hierarchy'
   else
    if(suspended_amr_levels_input/=1.or. &
       suspended_amr_ratio_input/=2.or. &
       suspended_amr_wet_depth_input/=0.d0.or. &
       suspended_amr_regrid_interval_input/=0.d0) &
     error stop 'v49 fixed-grid ownership fields are not canonical'
   endif
  endif
  if(binary_unit/=0) close(binary_unit)
  close(unit)
  if(.not.all(ieee_is_finite(q)).or..not.all(ieee_is_finite(area)).or..not.all(ieee_is_finite(rock)) &
   .or..not.all(ieee_is_finite(xy)).or..not.all(ieee_is_finite(mid)).or..not.all(ieee_is_finite(length)) &
   .or..not.all(ieee_is_finite(normal)).or..not.all(ieee_is_finite(forcing)) &
   .or..not.all(ieee_is_finite(series_time)).or..not.all(ieee_is_finite(series_concentration)) &
   .or..not.all(ieee_is_finite(series_bed_input)).or..not.all(ieee_is_finite(hydro_series_time)) &
   .or..not.all(ieee_is_finite(hydro_series_q)).or..not.all(ieee_is_finite(hydro_series_stage)) &
   .or..not.all(ieee_is_finite(manning_cell)).or..not.all(ieee_is_finite(fraction_concentration)) &
   .or..not.all(ieee_is_finite(fraction_bed_input)).or..not.all(ieee_is_finite(fraction_series_time)) &
   .or..not.all(ieee_is_finite(fraction_series_concentration)).or. &
   .not.all(ieee_is_finite(fraction_series_bed_input))) error stop 'nonfinite deck'
  if(any(fraction_concentration<0).or.any(fraction_bed_input<0)) error stop 'negative fraction boundary forcing'
  if(minval(manning_cell)<0) error stop 'negative per-cell hydraulic roughness parameter'
  if(hydraulic_roughness_mode/=braids_roughness_manning.and.any(manning_cell<=0)) &
   error stop 'Chezy/White-Colebrook roughness requires a positive parameter in every cell'
  if(amr_enabled.and.all(hydraulic_roughness_mode/=[braids_roughness_chezy, &
     braids_roughness_manning,braids_roughness_white_colebrook])) &
   error stop 'adaptive roughness method identity is invalid'
  if(hydraulic_roughness_mode/=braids_roughness_manning.and.secondary_flow_model/=0) &
   error stop 'typed Chezy/White-Colebrook secondary-flow coupling is not yet qualified'
  if(hydraulic_roughness_mode/=braids_roughness_manning.and.exchange_mode/=0) &
   error stop 'typed Chezy/White-Colebrook suspended exchange is not yet qualified'
  if(secondary_flow.and.any(manning_cell<=0)) error stop 'secondary flow requires positive hydraulic roughness in every cell'
  if(classic_secondary_foundation().and.any(manning_cell<=0)) &
   error stop 'v56 classic secondary flow requires positive typed roughness in every cell'
  if(bedform_roughness.and.hydraulic_roughness_mode/=braids_roughness_manning) &
   error stop 'bedform roughness currently requires Manning hydraulic roughness'
  if(bedform_roughness.and.any(manning_cell<=0)) &
   error stop 'bedform roughness requires positive Manning roughness in every cell'
  if(minval(q(1,:))<0.or.minval(q(4:nphysical_state,:))<0) error stop 'negative initial inventory'
  if(flow2dh_amr_ownership_declaration) then
   if(any(q(1,:)<=dry_tolerance.and. &
          (q(2,:)/=0.d0.or.q(3,:)/=0.d0))) &
    error stop 'v49 initial dry cell carries hydraulic momentum'
   do g=1,nfrac
    if(any(q(1,:)<=dry_tolerance.and.q(suspended_index(g),:)>0.d0)) &
     error stop 'v49 initial dry cell carries suspended inventory'
   enddo
  endif
  if(version>=17) then
   if(active_layer_thickness<=0) error stop 'bank morphology requires an active layer'
   if((bank_erosion_fraction>0.or.bank_avalanche).and..not.any(bank_erodible)) &
    error stop 'enabled bank morphology requires at least one erodible-bank cell'
  endif
  if(bank_failure) then
   ! The cumulative deck version is not itself a physical prerequisite.
   ! A compact v24 binary can carry an explicit erodible-bank mask around a
   ! layered v13 state, and the v38/v40 envelopes intentionally reuse that
   ! state layout.  Validate the two data actually required by every bank
   ! failure mode: a layered active bed and at least one marked donor.  A
   ! text v13 deck has no mask, so it still fails through bank_erodible.
   if(active_layer_thickness<=0.or..not.any(bank_erodible)) &
    error stop 'process-separated bank failure requires layered erodible-bank cells'
   if(bank_toe_erosion_mode==1.and.any(manning_cell<=0)) &
    error stop 'cohesive bank failure requires positive hydraulic roughness'
  endif
  ninc=sum(offset);allocate(faces(ninc),signs(ninc))
  k=1
  do c=1,nc
   ninc=offset(c);offset(c)=k;k=k+ninc
  enddo
  offset(nc+1)=k
  block
   integer :: cursor(nc)
   cursor=offset(:nc)
   do f=1,nf
    c=owner(f);k=cursor(c);faces(k)=f;signs(k)=1;cursor(c)=k+1
    c=neighbour(f)
    if(c>0) then
     k=cursor(c);faces(k)=f;signs(k)=-1;cursor(c)=k+1
    endif
   enddo
  end block
  ! Trace/root calculations visit only the faces owned by their logical ID.
  ! Preserve ascending native face order so every floating-point sum is
  ! identical to the former full-face scan. nb=0 has one offset and no faces.
  k=1
  do c=1,nb
   ninc=boundary_offset(c);boundary_offset(c)=k;k=k+ninc
  enddo
  boundary_offset(nb+1)=k
  allocate(boundary_faces(k-1))
  block
   integer :: cursor(nb)
   cursor=boundary_offset(:nb)
   do f=1,nf
    c=bid(f)
    if(c==0) cycle
    k=cursor(c);boundary_faces(k)=f;cursor(c)=k+1
   enddo
  end block
  block
   integer :: bb,kk,jj,ff,first,last,gg,ll,local_seed
   real(8) :: tangent(2),position,zmin,z
   real(8), allocatable :: initial_bed(:),nodes(:)
   allocate(initial_bed(nc));initial_bed=rock
   do gg=1,nfrac
    initial_bed=initial_bed+q(bed_index(gg),:)/bulk_fraction(gg)
    if(version>=13) then
     do ll=2,nbed_layers
      initial_bed=initial_bed+q(bed_layer_index(ll,gg),:)/bulk_fraction(gg)
     enddo
   endif
  enddo
   if(allocated(fixed_boundary_bed)) then
    do ff=1,nf
     if(bid(ff)<=0) cycle
     if(fixed_boundary_bed_mode(bid(ff))/=1) cycle
     fixed_boundary_bed(ff)=initial_bed(owner(ff))+boundary_bed_offset(ff)
    enddo
    if(.not.all(ieee_is_finite(fixed_boundary_bed))) error stop 'invalid fixed TELEMAC-GAIA boundary bed'
   endif
  do bb=1,nb
    first=boundary_offset(bb);last=boundary_offset(bb+1)-1
    if(sampled_boundary_bed(boundary_faces(first))==0) cycle
    if(any(sampled_boundary_bed(boundary_faces(first:last))/=1)) error stop 'partly sampled inlet section'
    tangent=[-normal(2,boundary_faces(first)),normal(1,boundary_faces(first))]
    if(tangent(1)<0.or.(tangent(1)==0.and.tangent(2)<0)) tangent=-tangent
    ! Canonical order along the section; legacy face order is untouched.
    do kk=first+1,last
     ff=boundary_faces(kk);position=dot_product(mid(:,ff),tangent);jj=kk-1
     do while(jj>=first)
      if(dot_product(mid(:,boundary_faces(jj)),tangent)<=position) exit
      boundary_faces(jj+1)=boundary_faces(jj);jj=jj-1
     enddo
     boundary_faces(jj+1)=ff
    enddo
    zmin=huge(1.d0)
    do kk=first,last
     ff=boundary_faces(kk);z=rock(owner(ff))
     do gg=1,nfrac
      z=z+q(bed_index(gg),owner(ff))/bulk_fraction(gg)
      if(version>=13) then
       do ll=2,nbed_layers
        z=z+q(bed_layer_index(ll,gg),owner(ff))/bulk_fraction(gg)
       enddo
      endif
     enddo
     z=z+boundary_bed_offset(ff)
     if(z<zmin) then
      zmin=z;boundary_seed_position(bb)=kk
     endif
    enddo
    if(production_section_geometry) then
     allocate(nodes(last-first+2));call section_nodes(bb,initial_bed,nodes)
     local_seed=minloc(nodes,dim=1)
     if(local_seed==1.or.local_seed==size(nodes)) error stop 'section minimum lies at an open endpoint'
     section_seed_node(bb)=local_seed
     deallocate(nodes)
    endif
   enddo
   deallocate(initial_bed)
  end block
  call cache_open_boundary_distance()
  call classify_rectangles()
  call cache_face_geometry()
  if(scheme==6) call cache_wall_geometry()
  if(bedlaw>0) then
   allocate(bedq_cell(2,nfrac,nc),bedq_equilibrium(2,nfrac,nc),bedq_relax_work(2,nfrac,nc), &
    bedq_grad(2,2,nfrac,nc),bedtau_cell(nfrac,nc))
   bedq_cell=0;bedq_equilibrium=0;bedq_relax_work=0;bedq_grad=0;bedtau_cell=0
   if(capacity_split_retained) then
    allocate(suspended_capacity_cell(2,nfrac,nc),suspended_capacity_equilibrium(2,nfrac,nc), &
     capacity_absolute_shields(nfrac,nc),capacity_critical_shields(nfrac,nc), &
     capacity_transport_stage(nfrac,nc),split_boundary_role(nf), &
     split_boundary_bed(2,nfrac,nf),split_boundary_suspended(2,nfrac,nf), &
     split_boundary_absolute_shields(nfrac,nf),split_boundary_critical_shields(nfrac,nf), &
     split_boundary_transport_stage(nfrac,nf),split_boundary_stress_ratio(nfrac,nf))
    suspended_capacity_cell=0.d0;suspended_capacity_equilibrium=0.d0
    capacity_absolute_shields=0.d0;capacity_critical_shields=0.d0;capacity_transport_stage=0.d0
    split_boundary_role=0;split_boundary_bed=0.d0;split_boundary_suspended=0.d0
    split_boundary_absolute_shields=0.d0;split_boundary_critical_shields=0.d0
    split_boundary_transport_stage=0.d0;split_boundary_stress_ratio=0.d0
   endif
   if(channel_composition_declaration) then
    allocate(channel_raw_bed_cell(2,nfrac,nc),channel_raw_suspended_cell(2,nfrac,nc), &
     channel_composed_bed_cell(2,nfrac,nc),channel_retained_suspended_cell(2,nfrac,nc), &
     channel_signed_suspended_rate_cell(nfrac,nc), &
     channel_equilibrium_concentration_cell(nfrac,nc), &
     channel_raw_bed_boundary(2,nfrac,nf),channel_raw_suspended_boundary(2,nfrac,nf), &
     channel_composed_bed_boundary(2,nfrac,nf), &
     channel_retained_suspended_boundary(2,nfrac,nf), &
     channel_signed_suspended_rate_boundary(nfrac,nf), &
     channel_equilibrium_concentration_boundary(nfrac,nf))
    channel_raw_bed_cell=0.d0;channel_raw_suspended_cell=0.d0
    channel_composed_bed_cell=0.d0;channel_retained_suspended_cell=0.d0
    channel_signed_suspended_rate_cell=0.d0;channel_equilibrium_concentration_cell=0.d0
    channel_raw_bed_boundary=0.d0;channel_raw_suspended_boundary=0.d0
    channel_composed_bed_boundary=0.d0;channel_retained_suspended_boundary=0.d0
    channel_signed_suspended_rate_boundary=0.d0
    channel_equilibrium_concentration_boundary=0.d0
   endif
   if(bedlaw==9) then
     allocate(gaia_bed_down(2,nc),gaia_eta_gradient(2,nc), &
      bedq_gaia_base(2,nfrac,nc),bedq_gaia_base_grad(2,2,nfrac,nc))
     gaia_bed_down=0.d0;gaia_eta_gradient=0.d0
     bedq_gaia_base=0.d0;bedq_gaia_base_grad=0.d0
   endif
  endif
  if(exchange_mode==4) then
   allocate(suspended_exchange_cell_trial(nfrac,nc), &
    suspended_exchange_cumulative(2,nfrac),suspended_exchange_step_trial(2,nfrac))
   suspended_exchange_cell_trial=0.d0;suspended_exchange_cumulative=0.d0
   suspended_exchange_step_trial=0.d0
  endif
  if(flow2dh_boundary_diffusion_declaration) then
   ! Adaptive memory is keyed by the immutable candidate-face identity, not
   ! by the transient active-face number rebuilt at every mesh epoch.  Open
   ! boundaries are pinned to finest leaves; amr_rebuild_mesh verifies that
   ! this exact physical face set never changes before accepting a regrid.
   k=nf
   if(amr_enabled) k=amr_fn
   allocate(suspended_boundary_last_outflow(nfrac,k), &
    suspended_boundary_return_clock(nfrac,k))
   suspended_boundary_last_outflow=0.d0
   suspended_boundary_return_clock=0.d0
  endif
  call allocate_secondary_work()
  if(version>=17.or.general_avalanche.or.bank_failure) allocate(bank_transfer(nfrac,nf),bank_scale(nfrac,nc),bank_bed(nc),bank_request(nc),bank_weight(nc),bank_down(2,nc), &
                           bank_previous(nfrac,nc),bank_depth(nc),bank_depth_intercept(nc),bank_rate(nf),bank_conductance(nf))
  if(bank_failure) then
   allocate(bank_undercut(nc),bank_reference_height(nc),bank_toe_height(nc),bank_reference_length(nc), &
             bank_toe_cumulative(nc),bank_collapse_cumulative(nc),bank_collapse_count(nc))
   bank_undercut=0;bank_reference_height=0;bank_toe_height=0;bank_reference_length=0
   bank_toe_cumulative=0;bank_collapse_cumulative=0;bank_collapse_count=0
  endif
  if(bank_front_tracking) then
   allocate(bank_front_status(nc),bank_front_seed_face(nc),bank_front_solid_area(nc), &
            bank_front_base(nc),bank_front_height(nc),bank_front_removed_mass(nfrac,nc))
   bank_front_status=0;bank_front_seed_face=0;bank_front_solid_area=area
   bank_front_base=0.d0;bank_front_height=0.d0;bank_front_removed_mass=0.d0
  endif
  if(bank_failure.and.bank_toe_erosion_mode==3) then
   allocate(distributed_face_mass_trial(nfrac,2,nf),distributed_cell_bedload(2,nfrac,nc), &
            distributed_rank_deficient(nc))
   distributed_face_mass_trial=0.d0;distributed_cell_bedload=0.d0
   distributed_rank_deficient=.true.
  endif
  do k=1,nfrac
   call braids_vr1993_grain_parameters(d50_fraction(k),rhos_fraction(k),rho,grav, &
    constitutive_kinematic_viscosity,dstar_fraction(k),taucr_fraction(k))
  enddo
  ! Preserve the compact historical restart signature: legacy envelopes bind
  ! dimensions here and continue to use their caller-supplied deck fingerprint.
  ! Legacy checkpoints do not duplicate complete cell, face or AMR arrays;
  ! doing so made large-run startup and checkpoint I/O proportional to the
  ! complete candidate hierarchy twice.  The explicit typed-v45 exception
  ! below binds one raw per-cell roughness vector because no fixed-size native
  ! digest exists yet.  The identity-bound v46 envelope then adds its actual
  ! parsed controls, so a stale or tampered supplied fingerprint cannot
  ! authorize different split-capacity coefficients.
  if(prognostic_secondary_flow()) then
   initial_signature=[real(envelope_version,8),real(nc,8),real(nf,8),real(nb,8), &
    real(nstate,8),real(nfrac,8),real(nbed_layers,8),merge(1.d0,0.d0,amr_enabled), &
    real(amr_levels,8),real(amr_ratio,8),real(nphysical_state,8),real(nphysical_primitive,8), &
    real(nphysical_ledger,8),real(nledger,8),real(secondary_flow_model,8)]
  else
   ! Preserve the exact legacy signature shape for disabled and v28 decks.
   initial_signature=[real(envelope_version,8),real(nc,8),real(nf,8),real(nb,8), &
    real(nstate,8),real(nfrac,8),real(nbed_layers,8),merge(1.d0,0.d0,amr_enabled), &
    real(amr_levels,8),real(amr_ratio,8)]
  endif
  if(typed_roughness_declaration) then
   ! A v45 deck fingerprint is supplied by the caller, so bind the parsed
   ! constitutive choices independently as native restart authority.  Keep
   ! every pre-v45 signature byte-for-byte unchanged by making this suffix
   ! conditional on the explicit typed envelope.
   initial_signature=[initial_signature,4501.d0, &
    real(hydraulic_roughness_mode,8), &
    merge(1.d0,0.d0,white_colebrook_chzk2d), &
    real(sediment_shear_mode,8),roughness_minimum_work_depth, &
    skin_kinematic_viscosity,skin_sand_roughness,skin_silt_roughness, &
    skin_critical_mud_fraction_low,skin_critical_mud_fraction_high, &
    skin_fluff_fraction,manning_friction_chezy_minimum_depth, &
    ! The historical storage name is not a Manning type declaration.  These
    ! are the exact selected C/n/ks bytes on the native topology; include them
    ! so a caller cannot reuse a supplied fingerprint after changing a zone or
    ! cell value.  Only explicit v45+ typed decks pay this O(nc), 8*nc-byte
    ! signature cost; replacing it with an exact fixed-size native digest is
    ! recorded performance debt, not grounds for weakening restart identity.
    manning_cell]
  endif
  if(capacity_foundation_forcing_identity_declaration) then
   ! V54 binds the exact forcing and ordered execution documents as well as
   ! the raw FP64 C/n/ks vector independently of the caller-supplied deck
   ! fingerprint.  This also covers ordinary Manning decks that do not need
   ! the historical typed-v45 constitutive envelope.
   initial_signature=[initial_signature,5401.d0, &
    (real(iachar(capacity_foundation_forcing_fingerprint(k:k)),8), &
     k=1,len(capacity_foundation_forcing_fingerprint)), &
    (real(iachar(capacity_foundation_forcing_raw_sha256(k:k)),8), &
     k=1,len(capacity_foundation_forcing_raw_sha256)), &
    (real(iachar(capacity_foundation_native_execution_fingerprint(k:k)),8), &
     k=1,len(capacity_foundation_native_execution_fingerprint)), &
    real(hydraulic_roughness_mode,8), &
    merge(1.d0,0.d0,white_colebrook_chzk2d), &
    real(sediment_shear_mode,8),roughness_minimum_work_depth, &
    manning_friction_chezy_minimum_depth,skin_kinematic_viscosity, &
    skin_sand_roughness,skin_silt_roughness, &
    skin_critical_mud_fraction_low,skin_critical_mud_fraction_high, &
    skin_fluff_fraction,manning_cell]
  endif
  if(hiding_foundation_identity_declaration) then
   ! V55 adds no mutable field beyond the already checkpointed active layer.
   ! Bind the independently verified canonical identity and the exact physical
   ! controls so neither another hiding law nor another ordered fraction pair
   ! can resume this accepted state under a reused caller fingerprint.
   initial_signature=[initial_signature,5501.d0,real(hiding_exposure,8), &
    hiding_parker_alpha,hiding_wu_exponent, &
    (real(iachar(hiding_foundation_execution_fingerprint(k:k)),8), &
     k=1,len(hiding_foundation_execution_fingerprint))]
  endif
  if(secondary_foundation_identity_declaration) then
   ! V56 adds no prognostic field.  Bind the complete authenticated identity,
   ! its exact parent and every live scalar so restart cannot cross methods.
   initial_signature=[initial_signature,5601.d0, &
    real(secondary_foundation_base_envelope,8), &
    secondary_foundation_rmincf,secondary_foundation_chzmin, &
    secondary_foundation_espir,secondary_foundation_coriolis, &
    secondary_foundation_betac, &
    (real(iachar(secondary_foundation_execution_fingerprint(k:k)),8), &
     k=1,len(secondary_foundation_execution_fingerprint))]
  endif
  if(capacity_parameter_declaration) then
   ! The deck fingerprint is supplied by the caller and is therefore not a
   ! sufficient native restart authority by itself.  Bind every v46 envelope
   ! control, the exact schema identity, and every resolved per-fraction
   ! coefficient into the checkpoint signature.  Legacy signatures remain
   ! byte-for-byte unchanged because this suffix exists only for v46.
   initial_signature=[initial_signature,4601.d0,real(capacity_parameter_law,8), &
    real(capacity_parameter_fraction_count,8),real(capacity_parameter_count,8), &
    merge(1.d0,0.d0,capacity_split_retained), &
    (real(iachar(capacity_parameter_schema(k:k)),8),k=1,len(capacity_parameter_schema)), &
    (real(iachar(capacity_profile_schema(k:k)),8),k=1,len(capacity_profile_schema)), &
    reshape(capacity_parameter(:capacity_parameter_count,:), &
     [capacity_parameter_count*nfrac])]
  endif
  if(typed_constitutive_declaration) then
   ! Raw v47 fields are repeated in the native signature so restart identity
   ! does not depend only on a caller-provided fingerprint.  SedThr remains a
   ! per-fraction physical input in the nested deck and is bound here too.
   initial_signature=[initial_signature,4701.d0,constitutive_kinematic_viscosity, &
    real(capacity_fraction_basis,8),suspended_adaptation_factor, &
    real(settlefrm_code,8),settling_fac_dss,real(settling_temperature_mode_code,8), &
    settling_temperature_c,settling_reference_concentration_declaration, &
    suspended_calibration_factor,real(sediment_activity_contract_code,8), &
    real(wet_mask_provider_code,8),real(settling_mode,8), &
    hindered_settling_reference_concentration,grav,rho,dry_tolerance, &
    d50_fraction,rhos_fraction,ws_fraction,sedthr_fraction, &
    (real(iachar(capacity_diagnostic_schema(k:k)),8),k=1,len(capacity_diagnostic_schema)), &
    (real(iachar(suspension_profile_schema(k:k)),8),k=1,len(suspension_profile_schema)), &
    (real(iachar(flow2dh_exchange_schema(k:k)),8),k=1,len(flow2dh_exchange_schema)), &
    (real(iachar(flow2dh_settling_schema(k:k)),8),k=1,len(flow2dh_settling_schema)), &
    (real(iachar(suspension_profile_fingerprint(k:k)),8), &
     k=1,len(suspension_profile_fingerprint))]
  endif
  if(flow2dh_boundary_diffusion_declaration) then
   ! Bind the raw v48 controls and logical-boundary return-time order in
   ! addition to the case fingerprint.  Stateful per-face memory itself is
   ! checkpointed below and is never reconstructed from the current solution.
   initial_signature=[initial_signature,4801.d0, &
    merge(1.d0,0.d0,flow2dh_neubc_sand), &
    real(flow2dh_horizontal_diffusivity_mode,8), &
    flow2dh_horizontal_diffusivity,flow2dh_diffusion_dry_threshold, &
    flow2dh_horizontal_schmidt,real(nb,8), &
    suspended_boundary_return_time, &
    real(mode,8),real(reversal_policy,8),fraction_concentration, &
    real(nfraction_series,8), &
    reshape(fraction_series_time,[size(fraction_series_time)]), &
    reshape(fraction_series_concentration,[size(fraction_series_concentration)]), &
    real(nseries,8),reshape(series_time,[size(series_time)]), &
    reshape(series_concentration,[size(series_concentration)]), &
    (real(iachar(flow2dh_open_boundary_schema(k:k)),8), &
     k=1,len(flow2dh_open_boundary_schema)), &
     (real(iachar(flow2dh_horizontal_diffusion_schema(k:k)),8), &
      k=1,len(flow2dh_horizontal_diffusion_schema))]
  endif
  if(flow2dh_amr_ownership_declaration) then
   ! v49 is restart authority, not merely a preparation receipt.  Bind the
   ! raw semantic identity, instance identity, nested-profile identity and
   ! every explicit fixed/adaptive scope field so a caller-supplied case hash
   ! cannot authorize a different wet/dry-AMR ownership operator.
   initial_signature=[initial_signature,4901.d0, &
    (real(iachar(suspended_amr_schema(k:k)),8), &
     k=1,len(suspended_amr_schema)), &
    (real(iachar(suspended_amr_fingerprint(k:k)),8), &
     k=1,len(suspended_amr_fingerprint)), &
    (real(iachar(suspended_amr_profile_fingerprint(k:k)),8), &
     k=1,len(suspended_amr_profile_fingerprint)), &
    real(suspended_amr_fraction_count_input,8), &
    suspended_amr_dry_tolerance_input, &
    suspended_amr_diffusion_dry_threshold_input, &
    real(suspended_amr_enabled_input,8), &
    real(suspended_amr_levels_input,8), &
    real(suspended_amr_ratio_input,8), &
     suspended_amr_wet_depth_input,suspended_amr_regrid_interval_input]
  endif
  if(capacity_route_identity_declaration) then
   ! V50 carries no new constitutive coefficient.  It makes the exact method
   ! and the ordered execution->profile instance identities native restart
   ! authority, independently of the caller-supplied deck fingerprint.
   initial_signature=[initial_signature,5001.d0, &
    real(capacity_route_identity_schema,8),capacity_route_acal, &
    (real(iachar(capacity_route_method_id(k:k)),8), &
     k=1,len(capacity_route_method_id)), &
    (real(iachar(capacity_route_execution_schema(k:k)),8), &
     k=1,len(capacity_route_execution_schema)), &
    (real(iachar(capacity_route_profile_schema(k:k)),8), &
     k=1,len(capacity_route_profile_schema)), &
    (real(iachar(capacity_route_execution_fingerprint(k:k)),8), &
     k=1,len(capacity_route_execution_fingerprint)), &
    (real(iachar(capacity_route_profile_fingerprint(k:k)),8), &
     k=1,len(capacity_route_profile_fingerprint))]
  endif
  if(channel_composition_declaration) then
   ! V51 owns the exact post-density class composition.  The scratch channel
   ! arrays need no checkpoint record; these resolved controls make any
   ! cross-profile resume fail before accepted state is read.
   initial_signature=[initial_signature,5101.d0, &
    real(channel_composition_schema,8), &
    real(channel_composition_fraction_count,8),channel_bed_factor, &
    channel_suspended_factor,channel_speed_epsilon, &
    (real(iachar(channel_composition_scope(k:k)),8), &
     k=1,len(channel_composition_scope)), &
    (real(iachar(channel_composition_semantic(k:k)),8), &
     k=1,len(channel_composition_semantic)), &
    (real(iachar(channel_composition_profile_fingerprint(k:k)),8), &
     k=1,len(channel_composition_profile_fingerprint)), &
     real(channel_transport_class_input(:nfrac),8)]
  endif
  if(channel_native_identity_declaration) then
   ! V52 is additive to the byte-stable v51 identity.  Bind the independently
   ! recomputed channel preimage digest and the mandatory complete selector
   ! execution profile, so neither v51 nor another v52 profile can cross-resume
   ! even if a caller reuses its supplied deck hash.
   initial_signature=[initial_signature,5201.d0, &
    real(channel_native_wire_schema,8),real(channel_native_preimage_length,8), &
    (real(iachar(channel_native_preimage_sha256(k:k)),8), &
     k=1,len(channel_native_preimage_sha256)), &
    real(execution_profile_schema,8),real(execution_profile_preimage_length,8), &
    (real(iachar(execution_profile_semantic(k:k)),8), &
     k=1,len(execution_profile_semantic)), &
    (real(iachar(execution_profile_fingerprint(k:k)),8), &
     k=1,len(execution_profile_fingerprint))]
  endif
  if(component_native_identity_declaration) then
   ! V53 restart authority includes the sorted component inventory and every
   ! independently recomputed document digest.  Exact document contents were
   ! already reconstructed and checked above; fixed-size hashes keep restart
   ! signature growth independent of document length.
   initial_signature=[initial_signature,5301.d0,real(component_native_count,8), &
    ((real(iachar(component_native_documents(c)%component_id(k:k)),8), &
      k=1,len(component_native_documents(c)%component_id)), &
     (real(iachar(component_native_documents(c)%document_sha256(k:k)),8), &
      k=1,len(component_native_documents(c)%document_sha256)), &
     c=1,component_native_count)]
  endif
  time=0;step_hint=dtmax
  if(amr_enabled) call amr_initialize_tree()
  if(bank_toe_erosion_mode==2) then
   toe_source_count=nf
   if(amr_enabled) toe_source_count=amr_fn
   allocate(physical_toe_flux_cumulative(nfrac,toe_source_count), &
    physical_toe_flux_absolute_cumulative(nfrac,toe_source_count), &
    physical_toe_flux_bank_id(toe_source_count),physical_toe_flux_toe_id(toe_source_count), &
    physical_toe_flux_local_face(toe_source_count),physical_toe_flux_active(toe_source_count))
   physical_toe_flux_cumulative=0.d0;physical_toe_flux_absolute_cumulative=0.d0
   physical_toe_flux_bank_id=0;physical_toe_flux_toe_id=0;physical_toe_flux_local_face=0
   physical_toe_flux_active=.false.
  endif
 end subroutine

 subroutine amr_read_tree(unit,binary_unit)
  integer, intent(in) :: unit,binary_unit
  integer :: version,n,f,p,k,source_id,binary_tree_version
  read(unit,*) version
  if(version>=5) then
   backspace(unit);read(unit,*) version,amr_levels,amr_ratio,amr_nn,amr_fn,amr_fine_offset
  else
   backspace(unit);read(unit,*) version,amr_nn,amr_fn,amr_fine_offset
   amr_levels=4;amr_ratio=2
  endif
  if((version<1.or.version>6).or.amr_levels<2.or.amr_levels>max_amr_levels.or. &
   amr_ratio/=2.or.amr_nn<=nc.or.amr_fn<nf.or.amr_nn-amr_fine_offset/=nc) &
   error stop 'invalid AMR tree dimensions'
  amr_absolute_boundary_bed=version>=4
  if(version>=2) then
   read(unit,*) amr_interval,amr_delay,amr_wet_level,amr_buffer,amr_wet_depth,amr_eta_tol,amr_u_tol, &
    amr_bed_tol,amr_c_tol,amr_mix_tol,amr_dz_tol,amr_flags,k
   if(k/=0.and.k/=1) error stop 'invalid AMR time integration mode'
   amr_subcycling=k==1
  else
   read(unit,*) amr_interval,amr_delay,amr_wet_level,amr_buffer,amr_wet_depth,amr_eta_tol,amr_u_tol, &
    amr_bed_tol,amr_c_tol,amr_mix_tol,amr_dz_tol,amr_flags
  endif
  if(.not.all(ieee_is_finite([amr_interval,amr_wet_depth,amr_eta_tol,amr_u_tol,amr_bed_tol,amr_c_tol, &
    amr_mix_tol,amr_dz_tol])).or.min(amr_interval,amr_wet_depth,amr_eta_tol,amr_u_tol,amr_bed_tol, &
    amr_c_tol,amr_mix_tol,amr_dz_tol)<=0.or.amr_delay<1.or.amr_wet_level<1.or.amr_wet_level>=amr_levels.or. &
    amr_buffer<2.or.any(amr_flags<0).or.any(amr_flags>1)) error stop 'invalid AMR criteria'
  allocate(at_parent(amr_nn),at_level(amr_nn),at_i(amr_nn),at_j(amr_nn),at_stride(amr_nn), &
   at_min(amr_nn),at_max(amr_nn),at_delay(amr_nn),at_area(amr_nn),at_xy(2,amr_nn),at_rock(amr_nn),at_manning(amr_nn), &
   at_q(nstate,amr_nn),at_initial(nstate,amr_nn),at_last_bed(amr_nn),at_bank(amr_nn), &
   at_offset(amr_nn+1),at_cover(amr_nn),at_required(amr_nn))
  if(bank_failure) allocate(at_bank_undercut(amr_nn),at_bank_reference_height(amr_nn),at_bank_toe_height(amr_nn), &
   at_bank_reference_length(amr_nn),at_bank_toe_cumulative(amr_nn),at_bank_collapse_cumulative(amr_nn), &
   at_bank_collapse_count(amr_nn))
  if(bank_front_tracking) allocate(at_bank_front_status(amr_nn),at_bank_front_seed_face(amr_nn), &
   at_bank_front_solid_area(amr_nn),at_bank_front_base(amr_nn),at_bank_front_height(amr_nn), &
   at_bank_front_removed_mass(nfrac,amr_nn))
  at_offset=0;at_delay=0;at_required=0
  if(version==6) then
   if(binary_unit==0) error stop 'binary AMR tree has no native stream'
   read(binary_unit) binary_tree_version
   if(binary_tree_version/=1) error stop 'unsupported binary AMR tree version'
   read(binary_unit) at_parent
   read(binary_unit) at_level
   read(binary_unit) at_i
   read(binary_unit) at_j
   read(binary_unit) at_stride
   read(binary_unit) at_area
   read(binary_unit) at_xy
   read(binary_unit) at_min
   read(binary_unit) at_max
  endif
  do n=1,amr_nn
   if(version==6) then
    continue
   else if(version>=3) then
    read(unit,*) at_parent(n),at_level(n),at_i(n),at_j(n),at_stride(n),at_area(n),at_xy(:,n),at_min(n),at_max(n)
   else
    read(unit,*) at_parent(n),at_level(n),at_i(n),at_j(n),at_stride(n),at_area(n),at_xy(:,n),at_min(n)
    at_max(n)=amr_levels-1
   endif
   p=at_parent(n)
   if(p<0.or.p>=n.or.at_level(n)<0.or.at_level(n)>=amr_levels.or. &
    at_stride(n)/=amr_ratio**(amr_levels-1-at_level(n)).or. &
    at_min(n)<0.or.at_min(n)>=amr_levels.or.at_max(n)<0.or.at_max(n)>=amr_levels.or. &
    at_area(n)<=0.or..not.ieee_is_finite(at_area(n)).or. &
    .not.all(ieee_is_finite(at_xy(:,n))).or.at_i(n)<1.or.at_j(n)<1.or. &
    at_i(n)>nx.or.at_j(n)>ny) error stop 'invalid AMR node'
   if(p==0) then
    if(at_level(n)/=0) error stop 'AMR root is not at level zero'
   else
    if(at_level(n)/=at_level(p)+1) error stop 'AMR parent level mismatch'
    at_offset(p)=at_offset(p)+1
   endif
   if(n>amr_fine_offset) then
    k=n-amr_fine_offset
    if(at_level(n)/=amr_levels-1.or.at_i(n)/=ii(k).or.at_j(n)/=jj(k).or. &
     abs(at_area(n)-area(k))>1.d-10*area(k).or.any(abs(at_xy(:,n)-xy(:,k))>1.d-8*max(dx,dy))) &
     error stop 'AMR finest geometry differs from initial deck'
   endif
  enddo
  k=1
  do n=1,amr_nn
   p=at_offset(n);at_offset(n)=k;k=k+p
  enddo
  at_offset(amr_nn+1)=k;allocate(at_child(k-1))
  block
   integer :: cursor(amr_nn)
   cursor=at_offset(:amr_nn)
   do n=1,amr_nn
    p=at_parent(n)
    if(p==0) cycle
    k=cursor(p);at_child(k)=n;cursor(p)=k+1
   enddo
  end block
  do n=1,amr_fine_offset
   if(at_offset(n)==at_offset(n+1)) error stop 'AMR parent has no children'
   if(abs(sum(at_area(at_child(at_offset(n):at_offset(n+1)-1)))-at_area(n))> &
    1.d-10*at_area(n)) error stop 'AMR children do not partition parent'
  enddo
  allocate(af_owner(amr_fn),af_neighbour(amr_fn),af_bid(amr_fn),af_level(amr_fn),af_source(amr_fn), &
   af_solid(amr_fn),af_normal(2,amr_fn),af_length(amr_fn),af_mid(2,amr_fn), &
   af_bed(amr_fn),af_left(amr_fn),af_right(amr_fn),af_sampled(amr_fn))
  af_bed=0;af_left=0;af_right=0;af_sampled=0
  if(version==6) then
   read(binary_unit) af_owner
   read(binary_unit) af_neighbour
   read(binary_unit) af_bid
   read(binary_unit) af_level
   read(binary_unit) af_source
   read(binary_unit) af_solid
   read(binary_unit) af_normal
   read(binary_unit) af_length
   read(binary_unit) af_mid
   read(binary_unit) af_sampled
   read(binary_unit) af_bed
   read(binary_unit) af_left
   read(binary_unit) af_right
  endif
  do f=1,amr_fn
   if(version==6) then
    continue
   else if(version>=4) then
    read(unit,*) af_owner(f),af_neighbour(f),af_bid(f),af_level(f),af_source(f),af_solid(f), &
     af_normal(:,f),af_length(f),af_mid(:,f),af_sampled(f),af_bed(f),af_left(f),af_right(f)
   else
    read(unit,*) af_owner(f),af_neighbour(f),af_bid(f),af_level(f),af_source(f),af_solid(f), &
     af_normal(:,f),af_length(f),af_mid(:,f)
   endif
   if(af_owner(f)<1.or.af_owner(f)>amr_nn.or.af_neighbour(f)<0.or.af_neighbour(f)>amr_nn.or. &
    af_bid(f)<0.or.af_bid(f)>nb.or.af_length(f)<=0.or.af_level(f)<0.or.af_level(f)>=amr_levels.or. &
    af_source(f)<0.or.af_source(f)>nf.or.abs(sum(af_normal(:,f)**2)-1)>1.d-10.or. &
    .not.all(ieee_is_finite([af_normal(:,f),af_length(f),af_mid(:,f)]))) error stop 'invalid AMR face'
   if(at_level(af_owner(f))/=af_level(f)) error stop 'AMR face-owner level mismatch'
   if(af_neighbour(f)>0) then
    if(at_level(af_neighbour(f))/=af_level(f)) error stop 'AMR face-neighbour level mismatch'
   endif
   source_id=af_source(f)
   if(version<4.and.source_id>0) then
    if(af_level(f)/=amr_levels-1.or.owner(source_id)+amr_fine_offset/=af_owner(f).or. &
     abs(af_length(f)-length(source_id))>1.d-10*length(source_id)) error stop 'AMR finest face mismatch'
    af_bed(f)=boundary_bed_offset(source_id);af_left(f)=boundary_bed_left_offset(source_id)
    af_right(f)=boundary_bed_right_offset(source_id);af_sampled(f)=sampled_boundary_bed(source_id)
   else if(version>=4.and.source_id>0) then
    if(af_level(f)/=amr_levels-1.or.owner(source_id)+amr_fine_offset/=af_owner(f).or. &
     abs(af_length(f)-length(source_id))>1.d-10*length(source_id)) error stop 'AMR finest face mismatch'
   endif
   if(af_sampled(f)<0.or.af_sampled(f)>1.or. &
      .not.all(ieee_is_finite([af_bed(f),af_left(f),af_right(f)]))) error stop 'invalid AMR boundary terrain'
  enddo
 end subroutine

 subroutine amr_restrict_raw_roughness(mode,parameter,weight,value,valid)
  ! Restrict the immutable *raw* material parameter, never a value inferred
  ! from the legacy ``manning_cell`` storage name.  C [m**0.5/s], Manning n
  ! [s/m**(1/3)] and Nikuradse ks [m] are intensive spatial fields, so the
  ! hierarchy uses the same area-measure restriction as AMRHierarchy.restrict.
  ! It is deliberately independent of transient depth/velocity: otherwise a
  ! regrid would change the material map as the hydraulics evolved.  The
  ! explicit mode branch preserves the physical type and its admissible range.
  integer,intent(in) :: mode
  real(8),intent(in) :: parameter(:),weight(:)
  real(8),intent(out) :: value
  logical,intent(out) :: valid
  real(8) :: total
  value=0.d0;valid=.false.
  if(size(parameter)<=0.or.size(parameter)/=size(weight)) return
  if(.not.all(ieee_is_finite(parameter)).or. &
     .not.all(ieee_is_finite(weight)).or.any(weight<=0.d0)) return
  select case(mode)
  case(braids_roughness_manning)
   if(any(parameter<0.d0)) return
  case(braids_roughness_chezy,braids_roughness_white_colebrook)
   if(any(parameter<=0.d0)) return
  case default
   return
  end select
  total=sum(weight)
  if(.not.ieee_is_finite(total).or.total<=0.d0) return
  value=dot_product(weight,parameter)/total
  valid=ieee_is_finite(value).and.value>=minval(parameter).and. &
   value<=maxval(parameter).and. &
   (mode==braids_roughness_manning.or.value>0.d0)
  if(.not.valid) value=0.d0
 end subroutine

 subroutine amr_initialize_tree()
  integer :: c,n,k,child,f
  real(8) :: total
  logical :: roughness_valid
  at_q=0;at_rock=0;at_manning=0;at_bank=.false.
  if(all(hydraulic_roughness_mode/=[braids_roughness_chezy, &
     braids_roughness_manning,braids_roughness_white_colebrook])) &
   error stop 'AMR raw roughness method identity is invalid'
  at_roughness_mode=hydraulic_roughness_mode
  if(bank_failure) then
   at_bank_undercut=0;at_bank_reference_height=0;at_bank_toe_height=0;at_bank_reference_length=0
   at_bank_toe_cumulative=0;at_bank_collapse_cumulative=0;at_bank_collapse_count=0
  endif
  if(bank_front_tracking) then
   at_bank_front_status=0;at_bank_front_seed_face=0
   at_bank_front_solid_area=at_area;at_bank_front_base=0.d0;at_bank_front_height=0.d0
   at_bank_front_removed_mass=0.d0
  endif
  allocate(at_leaf(nc))
  do c=1,nc
   n=amr_fine_offset+c;at_leaf(c)=n
   at_q(:,n)=q(:,c);at_rock(n)=rock(c);at_manning(n)=manning_cell(c);at_bank(n)=bank_erodible(c)
  enddo
  do n=amr_fine_offset,1,-1
   total=0
   do k=at_offset(n),at_offset(n+1)-1
    child=at_child(k);total=total+at_area(child)
    at_q(:,n)=at_q(:,n)+at_area(child)*at_q(:,child)
    at_rock(n)=at_rock(n)+at_area(child)*at_rock(child)
    at_bank(n)=at_bank(n).or.at_bank(child)
   enddo
   at_q(:,n)=at_q(:,n)/total;at_rock(n)=at_rock(n)/total
   call amr_restrict_raw_roughness(at_roughness_mode, &
    at_manning(at_child(at_offset(n):at_offset(n+1)-1)), &
    at_area(at_child(at_offset(n):at_offset(n+1)-1)),at_manning(n),roughness_valid)
   if(.not.roughness_valid) error stop 'AMR raw roughness restriction failed'
  enddo
  at_initial=at_q
  do n=1,amr_nn
   at_last_bed(n)=amr_node_bed(n)
  enddo
  if(amr_absolute_boundary_bed) then
   do k=1,amr_fn
    if(af_sampled(k)==0) cycle
    af_bed(k)=af_bed(k)-amr_node_bed(af_owner(k))
    af_left(k)=af_left(k)-amr_node_bed(af_owner(k))
    af_right(k)=af_right(k)-amr_node_bed(af_owner(k))
   enddo
  endif
  call amr_update_cover()
  ! The initial hierarchy is already the finest mesh.  If initial AMR
  ! selection keeps every leaf, amr_rebuild_mesh is intentionally skipped;
  ! still provide the immutable candidate-face identity used by diagnostics.
  if(allocated(at_face_source)) deallocate(at_face_source)
  allocate(at_face_source(nf));at_face_source=0
  do f=1,amr_fn
   if(af_source(f)<=0) cycle
   if(at_face_source(af_source(f))/=0) error stop 'duplicate finest AMR face source'
   at_face_source(af_source(f))=f
  enddo
  if(any(at_face_source==0)) error stop 'incomplete finest AMR face source map'
 end subroutine

 real(8) function amr_node_bed(n) result(bed)
  integer,intent(in) :: n
  integer :: g,l
  bed=at_rock(n)
  do l=1,nbed_layers
   do g=1,nfrac
   bed=bed+at_q(bed_layer_index(l,g),n)/bulk_fraction(g)
   enddo
  enddo
  if(bank_front_tracking) then
   if(at_bank_front_status(n)==BANK_FRONT_ACTIVE) then
    block
     integer :: front_error
     real(8) :: effective_bed
     call braids_bank_front_effective_bed(bed,at_area(n),at_bank_front_status(n), &
      at_bank_front_solid_area(n),at_bank_front_base(n),effective_bed,front_error)
     if(front_error/=BANK_FRONT_SUCCESS) error stop 'invalid adaptive persistent bank-front bed closure'
     bed=effective_bed
    end block
   endif
  endif
 end function

 subroutine amr_update_cover()
  integer :: c,n,p
  at_cover=0
  do c=1,size(at_leaf)
   at_cover(at_leaf(c))=at_leaf(c)
  enddo
  do n=1,amr_nn
   p=at_parent(n)
   if(p==0) cycle
   if(at_cover(p)>0) at_cover(n)=at_cover(p)
  enddo
 end subroutine

 logical function amr_use_face(f) result(use_face)
  integer, intent(in) :: f
  integer :: o,n,a,b
  use_face=.false.;o=af_owner(f);n=af_neighbour(f);a=at_cover(o)
  if(a==0) return
  if(n==0) then
   use_face=o==a;return
  endif
  b=at_cover(n)
  if(b==0.or.a==b) return
  use_face=o==a.or.n==b
 end function

 subroutine amr_sync_state()
  integer :: c,n,k,child
  real(8) :: total
  do c=1,nc
   at_q(:,at_leaf(c))=q(:,c)
  enddo
  ! Parents above the active leaves are restrictions. Descendants below an
  ! active coarse cell retain subcell bed detail for its next refinement.
  do n=amr_fine_offset,1,-1
   if(at_cover(n)/=0) cycle
   at_q(:,n)=0;total=0
   do k=at_offset(n),at_offset(n+1)-1
    child=at_child(k);total=total+at_area(child)
    at_q(:,n)=at_q(:,n)+at_area(child)*at_q(:,child)
   enddo
   at_q(:,n)=at_q(:,n)/total
  enddo
  call amr_sync_bank_failure_state()
 end subroutine

 subroutine amr_sync_bank_failure_state()
  integer :: c,n,k,child
  real(8) :: total
  if(.not.bank_failure) return
  do c=1,nc
   n=at_leaf(c)
   at_bank_undercut(n)=bank_undercut(c)
   at_bank_reference_height(n)=bank_reference_height(c)
   at_bank_toe_height(n)=bank_toe_height(c)
   at_bank_reference_length(n)=bank_reference_length(c)
   at_bank_toe_cumulative(n)=bank_toe_cumulative(c)
   at_bank_collapse_cumulative(n)=bank_collapse_cumulative(c)
   at_bank_collapse_count(n)=bank_collapse_count(c)
   if(bank_front_tracking) then
    at_bank_front_status(n)=bank_front_status(c)
    at_bank_front_seed_face(n)=bank_front_seed_face(c)
    at_bank_front_solid_area(n)=bank_front_solid_area(c)
    at_bank_front_base(n)=bank_front_base(c)
    at_bank_front_height(n)=bank_front_height(c)
    at_bank_front_removed_mass(:,n)=bank_front_removed_mass(:,c)
   endif
  enddo
  do n=amr_fine_offset,1,-1
   if(at_cover(n)/=0) cycle
   total=0;at_bank_undercut(n)=0;at_bank_reference_height(n)=0
   at_bank_toe_height(n)=0;at_bank_reference_length(n)=0
   at_bank_toe_cumulative(n)=0;at_bank_collapse_cumulative(n)=0
   at_bank_collapse_count(n)=0
   if(bank_front_tracking) then
    at_bank_front_status(n)=0;at_bank_front_seed_face(n)=0
    at_bank_front_solid_area(n)=at_area(n);at_bank_front_base(n)=0.d0;at_bank_front_height(n)=0.d0
    at_bank_front_removed_mass(:,n)=0.d0
   endif
   do k=at_offset(n),at_offset(n+1)-1
    child=at_child(k);total=total+at_area(child)
    at_bank_undercut(n)=max(at_bank_undercut(n),at_bank_undercut(child))
    at_bank_reference_height(n)=max(at_bank_reference_height(n),at_bank_reference_height(child))
    at_bank_toe_height(n)=max(at_bank_toe_height(n),at_bank_toe_height(child))
    at_bank_reference_length(n)=at_bank_reference_length(n)+at_area(child)*at_bank_reference_length(child)
    at_bank_toe_cumulative(n)=at_bank_toe_cumulative(n)+at_area(child)*at_bank_toe_cumulative(child)
    at_bank_collapse_cumulative(n)=at_bank_collapse_cumulative(n)+at_area(child)*at_bank_collapse_cumulative(child)
    at_bank_collapse_count(n)=max(at_bank_collapse_count(n),at_bank_collapse_count(child))
   enddo
   if(total>0) then
    at_bank_reference_length(n)=at_bank_reference_length(n)/total
    at_bank_toe_cumulative(n)=at_bank_toe_cumulative(n)/total
    at_bank_collapse_cumulative(n)=at_bank_collapse_cumulative(n)/total
   endif
  enddo
 end subroutine

 logical function amr_updates_cell(c) result(updates)
  integer,intent(in) :: c
  updates=.true.
  if(amr_step_level>=0) updates=at_level(at_leaf(c))==amr_step_level
 end function

 subroutine amr_cache_level_work()
  ! Cache ascending local indices for the immutable active mesh. Rebuilding
  ! these lists only when the leaf set changes avoids rediscovering the same
  ! level membership in every recursive substep. Ascending insertion retains
  ! the original cell/face update order and therefore deterministic ledgers.
  integer :: c,f,l,lo,ln,k,n,side
  logical :: marked(nc),face_marked(nc),seed(nc)
  if(allocated(amr_level_cells)) deallocate(amr_level_cells,amr_level_faces,amr_reflux_faces,amr_subtree_faces, &
                                             amr_level_eval_cells,amr_level_face_cells)
  allocate(amr_level_cells(nc,0:amr_levels-1),amr_level_faces(nf,0:amr_levels-1), &
           amr_reflux_faces(nf,0:amr_levels-1),amr_subtree_faces(nf,0:amr_levels-1), &
           amr_level_eval_cells(nc,0:amr_levels-1),amr_level_face_cells(nc,0:amr_levels-1))
  if(allocated(amr_prepare_role)) deallocate(amr_prepare_role)
  allocate(amr_prepare_role(nc,0:amr_levels-1));amr_prepare_role=0
  amr_level_cells=0;amr_level_faces=0;amr_reflux_faces=0;amr_subtree_faces=0
  amr_level_eval_cells=0;amr_level_face_cells=0
  amr_level_cell_count=0;amr_level_face_count=0;amr_reflux_face_count=0;amr_subtree_face_count=0
  amr_level_eval_cell_count=0;amr_level_face_cell_count=0
  do c=1,nc
   l=at_level(at_leaf(c));amr_level_cell_count(l)=amr_level_cell_count(l)+1
   amr_level_cells(amr_level_cell_count(l),l)=c
  enddo
  do f=1,nf
   lo=at_level(at_leaf(owner(f)));ln=-1
   if(neighbour(f)>0) ln=at_level(at_leaf(neighbour(f)))
   do l=0,amr_levels-1
    if(lo==l.or.ln==l) then
     amr_level_face_count(l)=amr_level_face_count(l)+1
     amr_level_faces(amr_level_face_count(l),l)=f
    endif
    if(lo>=l.or.ln>=l) then
     amr_subtree_face_count(l)=amr_subtree_face_count(l)+1
     amr_subtree_faces(amr_subtree_face_count(l),l)=f
    endif
   enddo
   if(ln>=0.and.abs(lo-ln)==1) then
    l=min(lo,ln);amr_reflux_face_count(l)=amr_reflux_face_count(l)+1
    amr_reflux_faces(amr_reflux_face_count(l),l)=f
   endif
  enddo
  do l=0,amr_levels-1
   marked=.false.;face_marked=.false.
   do k=1,amr_level_face_count(l)
    f=amr_level_faces(k,l);face_marked(owner(f))=.true.;marked(owner(f))=.true.
    if(neighbour(f)>0) then
     face_marked(neighbour(f))=.true.;marked(neighbour(f))=.true.
    endif
    do side=1,2
     if(weno_cells(1,side,f)<=0) cycle
     marked(weno_cells(:,side,f))=.true.
    enddo
   enddo
   ! Capacity gradients at a participating face endpoint consume its direct
   ! neighbours. Include one deterministic halo around all reconstruction
   ! cells so no stale derived value can enter an indirect face trace.
   seed=marked
   do c=1,nc
    if(.not.seed(c)) cycle
    do k=offset(c),offset(c+1)-1
     f=faces(k);n=neighbour(f)
     if(signs(k)<0) n=owner(f)
     if(n>0) marked(n)=.true.
    enddo
   enddo
   do c=1,nc
    if(face_marked(c)) then
     amr_prepare_role(c,l)=ibset(amr_prepare_role(c,l),1)
     amr_level_face_cell_count(l)=amr_level_face_cell_count(l)+1
     amr_level_face_cells(amr_level_face_cell_count(l),l)=c
    endif
    if(marked(c)) then
     amr_prepare_role(c,l)=ibset(amr_prepare_role(c,l),0)
     amr_level_eval_cell_count(l)=amr_level_eval_cell_count(l)+1
     amr_level_eval_cells(amr_level_eval_cell_count(l),l)=c
    endif
   enddo
  enddo
 end subroutine

 subroutine amr_allocate_step_workspaces()
  ! One solver-lifetime allocation replaces automatic full-mesh arrays on
  ! every SSPRK stage and recursive call. Separate slices by recursion level
  ! keep sibling substeps independent without changing the numerical order.
  if(allocated(amr_dense_old)) deallocate(amr_dense_old,amr_dense_new,amr_dense_t0,amr_dense_t1)
  if(allocated(amr_rk_state)) deallocate(amr_rk_state,amr_rk_flux,amr_level_flux,amr_subtree_flux, &
                                          amr_state_work,amr_accepted_work)
  allocate(amr_dense_old(nstate,nc),amr_dense_new(nstate,nc),amr_dense_t0(nc),amr_dense_t1(nc), &
           amr_rk_state(nstate,nc,4),amr_rk_flux(nstate,2,nf,3), &
           amr_level_flux(nstate,2,nf,0:amr_levels-1),amr_subtree_flux(nstate,2,nf,0:amr_levels-1), &
           amr_state_work(nstate,nc),amr_accepted_work(nledger,nb))
 end subroutine

 subroutine amr_coarse_ghosts(state,evaluation_time,level)
  real(8),intent(inout) :: state(nstate,nc)
  real(8),intent(in) :: evaluation_time
  integer,intent(in) :: level
  integer :: c,l,k
  real(8) :: alpha
  do l=0,level-1
   do k=1,amr_level_cell_count(l)
    c=amr_level_cells(k,l);alpha=0
    if(amr_dense_t1(c)>amr_dense_t0(c)) &
     alpha=max(0.d0,min(1.d0,(evaluation_time-amr_dense_t0(c))/(amr_dense_t1(c)-amr_dense_t0(c))))
    state(:,c)=(1-alpha)*amr_dense_old(:,c)+alpha*amr_dense_new(:,c)
   enddo
  enddo
 end subroutine

 subroutine amr_level_rk(level,start_time,dt,state,integral,accepted,secondary_source_integral, &
                         momentum_boundary_impulse,ok)
  integer,intent(in) :: level
  real(8),intent(in) :: start_time,dt
 real(8),intent(inout) :: state(nstate,nc)
 real(8),intent(out) :: integral(nstate,2,nf),accepted(nledger,nb)
  real(8),intent(out) :: secondary_source_integral
  real(8),intent(out) :: momentum_boundary_impulse(2,2)
  logical,intent(out) :: ok
  real(8) :: fa(nledger,nb),fb(nledger,nb),fc(nledger,nb),profile_start,step_morfac,source_first,source_second
  real(8) :: momentum_a(2,2),momentum_b(2,2),momentum_c(2,2)
  integer :: c,k
  logical :: source_ok
  associate(initial=>amr_rk_state(:,:,1),a=>amr_rk_state(:,:,2), &
            b=>amr_rk_state(:,:,3),final=>amr_rk_state(:,:,4), &
            flux_a=>amr_rk_flux(:,:,:,1),flux_b=>amr_rk_flux(:,:,:,2),flux_c=>amr_rk_flux(:,:,:,3))
  amr_step_level=level;initial=state;integral=0;accepted=0;secondary_source_integral=0.d0
  momentum_boundary_impulse=0.d0
  source_first=0.d0;source_second=0.d0;step_morfac=morphological_factor_at(start_time)
  call amr_coarse_ghosts(initial,start_time,level)
  if(profile_enabled) profile_start=omp_get_wtime()
  call prepare_hydraulic_stage(initial)
  if(profile_enabled) then
   profile_prepare_s=profile_prepare_s+omp_get_wtime()-profile_start;profile_prepare_calls=profile_prepare_calls+1
   profile_start=omp_get_wtime()
  endif
  if(prognostic_secondary_flow()) then
   call secondary_relax_state(initial,.5d0*dt,level,source_first,source_ok)
   if(.not.source_ok) then
    failure_reason=6;ok=.false.;return
   endif
  endif
  call euler(initial,dt,start_time,step_morfac,a,fa,ok,flux_a,momentum_a)
  if(profile_enabled) then
   profile_euler_s=profile_euler_s+omp_get_wtime()-profile_start;profile_euler_calls=profile_euler_calls+1
  endif
  if(.not.ok) return
  call amr_coarse_ghosts(a,start_time+dt,level)
  if(profile_enabled) profile_start=omp_get_wtime()
  call prepare_hydraulic_stage(a)
  if(profile_enabled) then
   profile_prepare_s=profile_prepare_s+omp_get_wtime()-profile_start;profile_prepare_calls=profile_prepare_calls+1
   profile_start=omp_get_wtime()
  endif
  call euler(a,dt,start_time+dt,step_morfac,b,fb,ok,flux_b,momentum_b)
  if(profile_enabled) then
   profile_euler_s=profile_euler_s+omp_get_wtime()-profile_start;profile_euler_calls=profile_euler_calls+1
  endif
  if(.not.ok) return
  b=.75d0*initial+.25d0*b
  call rebalance_active_layer(b)
  call amr_coarse_ghosts(b,start_time+.5d0*dt,level)
  if(profile_enabled) profile_start=omp_get_wtime()
  call prepare_hydraulic_stage(b)
  if(profile_enabled) then
   profile_prepare_s=profile_prepare_s+omp_get_wtime()-profile_start;profile_prepare_calls=profile_prepare_calls+1
   profile_start=omp_get_wtime()
  endif
  call euler(b,dt,start_time+.5d0*dt,step_morfac,final,fc,ok,flux_c,momentum_c)
  if(profile_enabled) then
   profile_euler_s=profile_euler_s+omp_get_wtime()-profile_start;profile_euler_calls=profile_euler_calls+1
  endif
  if(.not.ok) return
  final=initial/3.d0+2.d0*final/3.d0
  call rebalance_active_layer(final)
  if(profile_enabled) profile_start=omp_get_wtime()
  call prepare_hydraulic_stage(final)
  if(profile_enabled) then
   profile_prepare_s=profile_prepare_s+omp_get_wtime()-profile_start;profile_prepare_calls=profile_prepare_calls+1
   profile_start=omp_get_wtime()
  endif
  call source(final,dt,initial(1,:))
  if(exchange_mode==4.and.flow2dh_amr_ownership_declaration) &
   call accumulate_suspended_exchange_trial(level)
  if(profile_enabled) then
   profile_source_s=profile_source_s+omp_get_wtime()-profile_start;profile_source_calls=profile_source_calls+1
  endif
  if(prognostic_secondary_flow()) then
   call secondary_relax_state(final,.5d0*dt,level,source_second,source_ok)
   if(.not.source_ok) then
    failure_reason=6;ok=.false.;return
   endif
  endif
  do k=1,amr_level_cell_count(level)
   c=amr_level_cells(k,level)
   if(.not.all(ieee_is_finite(final(:,c))).or.final(1,c)<0.or.any(final(4:nphysical_state,c)<0)) then
    amr_failure_cell=c;amr_failure_state=minloc(final(:,c),dim=1)
    amr_failure_value=final(amr_failure_state,c)
    failure_reason=9;ok=.false.;return
   endif
   amr_dense_old(:,c)=state(:,c);amr_dense_new(:,c)=final(:,c)
   amr_dense_t0(c)=start_time;amr_dense_t1(c)=start_time+dt
   state(:,c)=final(:,c)
  enddo
  integral=(flux_a+flux_b)/6.d0+2.d0*flux_c/3.d0
  accepted=(fa+fb)/6.d0+2.d0*fc/3.d0
  momentum_boundary_impulse=(momentum_a+momentum_b)/6.d0+2.d0*momentum_c/3.d0
  secondary_source_integral=source_first+source_second
  end associate
 end subroutine

 subroutine amr_repair_reflux_tails(level,state,ok)
  ! Fine substeps apply suspended/bed exchange before their integrated flux
  ! replaces the coarse flux.  The two phase flux corrections can therefore
  ! leave either member of a grain class negative even though their locally
  ! conserved total is admissible.  Restore the non-negative phase split in
  ! that same coarse cell before assessing individual roundoff tails.  This is
  ! local (no spatial smoothing), preserves every grain-class total exactly and
  ! rejects a negative class total instead of borrowing from another class.
  integer,intent(in) :: level
  real(8),intent(inout) :: state(nstate,nc)
  logical,intent(out) :: ok
  integer :: m,c,k,f,n,donor,g,si,bi
  real(8) :: deficit,total_deficit,positive_mass,maximum,available,candidate,tolerance,total_available,class_total,take
  integer :: donor_count
  logical :: resolved_reflux
  ok=.false.;tolerance=64*epsilon(1.d0)
  ! This applies at wet and dry coarse/fine interfaces.  Restricting it to dry
  ! cells caused resolved wetting fronts to reject forever as the exchange
  ! partition moved between the fine suspended and active-bed reservoirs.
  if(exchange_mode>0) then
   do g=1,nfrac
    si=suspended_index(g);bi=bed_index(g)
    tolerance=512*epsilon(1.d0)*max(maxval(abs(state(si,:))),maxval(abs(state(bi,:))),1.d0)
    do c=1,nc
     if(at_level(at_leaf(c))/=level) cycle
     class_total=state(si,c)+state(bi,c)
     if(class_total<0) then
      if(abs(class_total)<=tolerance) then
       state(si,c)=0;state(bi,c)=0;cycle
      endif
      amr_failure_cell=c;amr_failure_state=si;amr_failure_value=class_total;return
     endif
     if(state(si,c)<0) then
      state(si,c)=0;state(bi,c)=class_total
     else if(state(bi,c)<0) then
      state(si,c)=class_total;state(bi,c)=0
     endif
    enddo
   enddo
  endif
  do m=4,nphysical_state
   if(all(state(m,:)>=0)) cycle
   maximum=maxval(state(m,:));positive_mass=0;total_deficit=0
   do c=1,nc
    positive_mass=positive_mass+max(0.d0,state(m,c))*area(c)
    total_deficit=total_deficit-min(0.d0,state(m,c))*area(c)
   enddo
   resolved_reflux=.false.
   if(exchange_mode==0) then
    do g=1,nfrac
     if(m==bed_index(g)) resolved_reflux=.true.
    enddo
   endif
   if(.not.resolved_reflux.and. &
      (-minval(state(m,:))>tolerance*maximum.or.total_deficit>tolerance*positive_mass)) then
    amr_failure_cell=minloc(state(m,:),dim=1);amr_failure_state=m
    amr_failure_value=state(m,amr_failure_cell);return
   endif
   do c=1,nc
    if(state(m,c)>=0) cycle
    if(at_level(at_leaf(c))/=level) then
     amr_failure_cell=c;amr_failure_state=m;amr_failure_value=state(m,c);return
    endif
    deficit=-state(m,c)*area(c);donor=0;available=0;total_available=0;donor_count=0
    do k=offset(c),offset(c+1)-1
     f=faces(k);n=neighbour(f)
     if(signs(k)<0) n=owner(f)
     if(n==0) cycle
     if(at_level(at_leaf(n))/=level+1) cycle
     candidate=state(m,n)*area(n)
     if(candidate>0) then
      total_available=total_available+candidate;donor_count=donor_count+1
     endif
     if(candidate>available) then
      donor=n;available=candidate
     endif
    enddo
    if(donor==0.or.total_available<deficit) then
     amr_failure_cell=c;amr_failure_state=m;amr_failure_value=state(m,c)
     amr_failure_aux=total_available;return
    endif
    available=deficit
    do k=offset(c),offset(c+1)-1
     f=faces(k);n=neighbour(f)
     if(signs(k)<0) n=owner(f)
     if(n==0) cycle
     if(at_level(at_leaf(n))/=level+1) cycle
     candidate=max(0.d0,state(m,n)*area(n))
     if(candidate<=0) cycle
     take=min(candidate,available)
     state(m,n)=(candidate-take)/area(n);available=available-take
     if(available<=0) exit
    enddo
    state(m,c)=0
   enddo
  enddo
  ok=.true.
 end subroutine

 subroutine amr_repair_water_reflux(level,state,ok)
  ! A sum of stable fine substeps can overdraw a coarse wetting cell by a
  ! minute amount when it replaces the coarse interface flux. Return that
  ! overdraft from the adjacent fine recipients before the synchronized state
  ! is accepted. The operation is deterministic and conserves all three SWE
  ! inventories; it does not alter the Riemann flux or its time integration.
  integer,intent(in) :: level
  real(8),intent(inout) :: state(nstate,nc)
  logical,intent(out) :: ok
  real(8) :: donor_mass(nc),coarse_integral(3),secondary_integral,deficit,total,take,remaining,fraction
  integer :: c,k,f,n
  ok=.false.
  do c=1,nc
   if(level>=0) then
    if(at_level(at_leaf(c))/=level) cycle
   endif
   if(state(1,c)>=0) cycle
   deficit=-state(1,c)*area(c);donor_mass=0
   if(deficit==0) then
    state(1:3,c)=0
    if(prognostic_secondary_flow()) state(secondary_state_index(),c)=0.d0
    cycle
   endif
   do k=offset(c),offset(c+1)-1
    f=faces(k);n=neighbour(f)
    if(signs(k)<0) n=owner(f)
    if(n==0) cycle
    if(level>=0) then
     if(at_level(at_leaf(n))/=level+1) cycle
    endif
    donor_mass(n)=max(0.d0,state(1,n)*area(n))
   enddo
   total=sum(donor_mass)
   if(total<deficit.and.level<0) then
    do n=1,nc
     donor_mass(n)=max(0.d0,state(1,n)*area(n))
    enddo
    total=sum(donor_mass)
   endif
   if(total<deficit) then
    amr_failure_cell=c;amr_failure_state=1;amr_failure_value=state(1,c)
    amr_failure_aux=total;return
   endif
   coarse_integral=state(1:3,c)*area(c);secondary_integral=0.d0
   if(prognostic_secondary_flow()) secondary_integral=state(secondary_state_index(),c)*area(c)
   state(1:3,c)=0
   if(prognostic_secondary_flow()) state(secondary_state_index(),c)=0.d0
   remaining=deficit
   do n=1,nc
    if(donor_mass(n)<=0) cycle
    take=min(donor_mass(n),remaining);fraction=take/deficit
    state(1,n)=state(1,n)-take/area(n)
    state(2:3,n)=state(2:3,n)+fraction*coarse_integral(2:3)/area(n)
    if(prognostic_secondary_flow()) state(secondary_state_index(),n)= &
     state(secondary_state_index(),n)+fraction*secondary_integral/area(n)
    remaining=max(0.d0,remaining-take)
    if(remaining<=0) exit
   enddo
  enddo
  ok=.true.
 end subroutine

 recursive subroutine amr_subcycle_level(level,start_time,dt,state,integral,accepted,counts,secondary_source_integral, &
                                         momentum_boundary_impulse,ok)
  integer,intent(in) :: level
  real(8),intent(in) :: start_time,dt
  real(8),intent(inout) :: state(nstate,nc)
  real(8),intent(out) :: integral(nstate,2,nf),accepted(nledger,nb)
  real(8),intent(out) :: secondary_source_integral
  real(8),intent(out) :: momentum_boundary_impulse(2,2)
  integer,intent(inout) :: counts(0:max_amr_levels-1)
  logical,intent(out) :: ok
  real(8) :: own_ledger(nledger,nb),fine_ledger(nledger,nb),own_source,fine_source
  real(8) :: own_momentum(2,2),fine_momentum(2,2)
  integer :: c,f,n,coarse,side,substep,lc,ln,k
  logical :: any_here,any_finer
  integral=0;accepted=0;secondary_source_integral=0.d0;momentum_boundary_impulse=0.d0
  own_source=0.d0;fine_source=0.d0;own_momentum=0.d0;fine_momentum=0.d0;ok=.true.
  any_here=amr_level_cell_count(level)>0;any_finer=.false.
  if(level<amr_levels-1) any_finer=sum(amr_level_cell_count(level+1:amr_levels-1))>0
  if(any_here) then
   call amr_level_rk(level,start_time,dt,state,amr_level_flux(:,:,:,level),own_ledger,own_source,own_momentum,ok)
   if(.not.ok) return
   accepted=own_ledger;secondary_source_integral=own_source;momentum_boundary_impulse=own_momentum
   counts(level)=counts(level)+1
  endif
  if(any_finer) then
   do substep=0,amr_ratio-1
    call amr_subcycle_level(level+1,start_time+substep*dt/amr_ratio,dt/amr_ratio,state, &
                            amr_subtree_flux(:,:,:,level+1),fine_ledger,counts,fine_source,fine_momentum,ok)
    if(.not.ok) return
    do k=1,amr_subtree_face_count(level+1)
     f=amr_subtree_faces(k,level+1)
     integral(:,:,f)=integral(:,:,f)+amr_subtree_flux(:,:,f,level+1)
    enddo
    accepted=accepted+fine_ledger;secondary_source_integral=secondary_source_integral+fine_source
    momentum_boundary_impulse=momentum_boundary_impulse+fine_momentum
   enddo
  endif
  ! The fine time integral replaces the coarse one on both sides of each
  ! coarse/fine face. Substrate rows carry no horizontal flux. This uses the
  ! final limited water/suspended/bed flux, including both momentum traces.
  if(any_here.and.any_finer) then
   do k=1,amr_reflux_face_count(level)
    f=amr_reflux_faces(k,level)
    c=owner(f);n=neighbour(f)
    lc=at_level(at_leaf(c));ln=at_level(at_leaf(n));coarse=0;side=0
    if(lc==level.and.ln==level+1) then
     coarse=c;side=1
    else if(ln==level.and.lc==level+1) then
     coarse=n;side=2
    endif
    if(coarse==0) cycle
    if(side==1) then
     state(:,coarse)=state(:,coarse)+(amr_level_flux(:,side,f,level)-integral(:,side,f))/area(coarse)
    else
     state(:,coarse)=state(:,coarse)-(amr_level_flux(:,side,f,level)-integral(:,side,f))/area(coarse)
    endif
   enddo
   amr_step_level=level
   if(.not.all(ieee_is_finite(state))) then
    amr_failure_cell=minloc(state(1,:),dim=1);amr_failure_state=1
    amr_failure_value=state(1,amr_failure_cell)
    failure_reason=9;ok=.false.;return
   endif
   call amr_repair_water_reflux(level,state,ok)
   if(.not.ok) then
    failure_reason=9;return
   endif
   call amr_repair_reflux_tails(level,state,ok)
   if(.not.ok) then
    failure_reason=9;return
   endif
   call rebalance_active_layer(state)
  endif
  ! Return the physical time integral represented by the completed subtree.
  ! Deeper boundary faces used to disappear here because only this level's
  ! own_flux was copied into an initially zero array. At a coarse/fine
  ! interface the recursively accumulated fine integral must likewise remain
  ! the flux register seen by the parent.
  do k=1,amr_level_face_count(level)
   f=amr_level_faces(k,level)
   c=owner(f);n=neighbour(f);lc=at_level(at_leaf(c));ln=-1
   if(n>0) ln=at_level(at_leaf(n))
   if((lc==level.and.ln==level+1).or.(ln==level.and.lc==level+1)) cycle
   integral(:,:,f)=amr_level_flux(:,:,f,level)
  enddo
 end subroutine

 integer function series_interval(times,count,evaluation_time) result(knot)
  ! Return the first knot at or above an interior evaluation time.  Forcing
  ! tables from instruments can contain tens of thousands of rows; scanning
  ! them at every SSPRK stage made boundary evaluation O(steps * knots).
  real(8),intent(in) :: times(:),evaluation_time
  integer,intent(in) :: count
  integer :: low,high,middle
  low=2;high=count
  do while(low<high)
   middle=low+(high-low)/2
   if(evaluation_time<=times(middle)) then
    high=middle
   else
    low=middle+1
   endif
  enddo
  knot=low
 end function

 subroutine boundary_discharge_at(evaluation_time,discharge)
  ! Hydraulic schedules are piecewise linear and are evaluated independently
  ! of the stage/section solve. This is used by the synchronization reflux,
  ! where only the prescribed integral is required.
  real(8),intent(in) :: evaluation_time
  real(8),intent(out) :: discharge(nb)
  real(8) :: weight
  integer :: b,k
  discharge=forcing(1,:)
  do b=1,nb
   if(nhydro_series(b)==0) cycle
   if(evaluation_time<=hydro_series_time(1,b)) then
    discharge(b)=hydro_series_q(1,b)
   else if(evaluation_time>=hydro_series_time(nhydro_series(b),b)) then
    discharge(b)=hydro_series_q(nhydro_series(b),b)
   else
    k=series_interval(hydro_series_time(:,b),nhydro_series(b),evaluation_time)
    weight=(evaluation_time-hydro_series_time(k-1,b))/ &
           (hydro_series_time(k,b)-hydro_series_time(k-1,b))
    discharge(b)=(1-weight)*hydro_series_q(k-1,b)+weight*hydro_series_q(k,b)
   endif
  enddo
 end subroutine

 logical function amr_requires_synchronous_wet_front(evaluation_time) result(required)
  ! Dense-output time interpolation across asynchronously advancing levels is
  ! accurate for smooth wet flow, but it is not a conservative wet/dry Riemann
  ! state.  Advance all active leaves with the same SSPRK step whenever a wet
  ! front exists.  An entirely dry domain with prescribed inflow also needs a
  ! synchronized first step so that the front is created consistently.
  real(8),intent(in) :: evaluation_time
  real(8) :: discharge(nb)
  integer :: f,n,b,k,c
  logical :: any_wet_owner
  required=.false.
  ! A positive Thatcher--Harleman return time owns sequential face history.
  ! Resolving several fine substeps against one frozen root memory would be a
  ! different operator whenever flow reverses inside the root interval.
  ! Until transaction-local recursive memory is implemented, advance this
  ! bounded configuration synchronously; accepted memory still commits once
  ! and rejection/cancellation remains byte-exact.
  if(flow2dh_amr_ownership_declaration.and. &
     any(suspended_boundary_return_time>0.d0)) then
   required=.true.;return
  endif
  do f=1,nf
   n=neighbour(f)
   if(n<=0) cycle
   if((q(1,owner(f))>amr_wet_depth).neqv.(q(1,n)>amr_wet_depth)) then
    required=.true.;return
   endif
  enddo
  call boundary_discharge_at(evaluation_time,discharge)
  do b=1,nb
   if(mode(b)/=1.and.mode(b)/=6.and.mode(b)/=7.and.mode(b)/=8.and.mode(b)/=9) cycle
   if(discharge(b)<=0) cycle
   any_wet_owner=.false.
   do k=boundary_offset(b),boundary_offset(b+1)-1
    c=owner(boundary_faces(k))
    if(q(1,c)>amr_wet_depth) any_wet_owner=.true.
   enddo
   if(.not.any_wet_owner) then
    required=.true.;return
   endif
  enddo
 end function

 subroutine prescribed_boundary_rates(evaluation_time,rates,prescribed)
  real(8),intent(in) :: evaluation_time
  real(8),intent(out) :: rates(nledger,nb)
  logical,intent(out) :: prescribed(nledger,nb)
  real(8) :: discharge(nb),concentration(nfrac,nb),bed_input(nfrac,nb)
  integer :: b,g
  rates=0;prescribed=.false.
  call boundary_discharge_at(evaluation_time,discharge)
  call boundary_sediment_at(evaluation_time,concentration,bed_input)
  do b=1,nb
   if(mode(b)/=1.and.mode(b)/=6.and.mode(b)/=7.and.mode(b)/=8.and.mode(b)/=9) cycle
   rates(1,b)=discharge(b);prescribed(1,b)=.true.
   do g=1,nfrac
    ! NeuBcSand and Thatcher--Harleman resolve a face-local concentration
    ! from accepted state/memory.  Their asynchronous sum is physical, not a
    ! scheduled Q*C value, and must never be overwritten by synchronization.
    if(.not.flow2dh_boundary_diffusion_declaration.or. &
       (.not.flow2dh_neubc_sand.and. &
        suspended_boundary_return_time(b)==0.d0)) then
     rates(1+g,b)=discharge(b)*concentration(g,b)
     prescribed(1+g,b)=.true.
    endif
    if(bed_boundary_mode(b)/=1.and.deck_version>=3) then
     rates(1+nfrac+g,b)=bed_input(g,b);prescribed(1+nfrac+g,b)=.true.
    endif
   enddo
  enddo
 end subroutine

 subroutine amr_reconcile_prescribed_boundaries(start_time,dt,state,integral,accepted,ok)
  ! A logical inlet can contain leaves on several asynchronously subcycled
  ! levels. Each level advances its physical share correctly, but the sum of
  ! those shares is sampled at different RK times. Reflux the minute mismatch
  ! into the boundary owners at synchronization and record that same physical
  ! correction in the ledger. Simpson/SSPRK weights integrate Q, prescribed
  ! bed supply and the quadratic Q*C product exactly between forcing knots.
  real(8),intent(in) :: start_time,dt,integral(nstate,2,nf)
  real(8),intent(inout) :: state(nstate,nc),accepted(nledger,nb)
  logical,intent(out) :: ok
  real(8) :: ra(nledger,nb),rb(nledger,nb),rc(nledger,nb),expected(nledger,nb)
  real(8) :: weights(nc),amounts(nc),total,delta,residual
  logical :: pa(nledger,nb),pb(nledger,nb),pc(nledger,nb)
  integer :: b,m,g,k,f,c,si,largest
  ok=.false.;expected=accepted
  call prescribed_boundary_rates(start_time,ra,pa)
  call prescribed_boundary_rates(start_time+dt,rb,pb)
  call prescribed_boundary_rates(start_time+.5d0*dt,rc,pc)
  do b=1,nb
   do m=1,nledger
    if(.not.pa(m,b)) cycle
    if(.not.pb(m,b).or..not.pc(m,b)) error stop 'inconsistent prescribed AMR boundary mask'
    expected(m,b)=dt*((ra(m,b)+rb(m,b))/6.d0+2.d0*rc(m,b)/3.d0)
    if(flow2dh_boundary_diffusion_declaration.and. &
       m>=2.and.m<=1+nfrac) then
     ! The prescribed rate is advective.  Horizontal diffusion is already an
     ! accepted physical face contribution and remains additive in both the
     ! public total and its signed diagnostic decomposition.
     g=m-1
     expected(suspended_advective_ledger_index(g),b)=expected(m,b)
     expected(m,b)=expected(m,b)+ &
      accepted(suspended_diffusive_ledger_index(g),b)
    endif
   enddo
  enddo
  do b=1,nb
   if(.not.any(pa(:,b))) cycle
   weights=0
   do k=boundary_offset(b),boundary_offset(b+1)-1
    f=boundary_faces(k);c=owner(f)
    weights(c)=weights(c)+max(-integral(1,1,f),0.d0)
   enddo
   total=sum(weights)
   do m=1,nledger
    if(.not.pa(m,b)) cycle
    total=sum(weights)
    delta=expected(m,b)-accepted(m,b)
    if(delta==0) cycle
    if(total<=0) then
     amr_failure_cell=owner(boundary_faces(boundary_offset(b)))
     amr_failure_state=1;amr_failure_value=delta;amr_failure_aux=total
     failure_reason=9;return
    endif
    if(m==1) then
     si=1
    else if(m<=1+nfrac) then
     si=suspended_index(m-1)
    else
     si=bed_index(m-1-nfrac)
    endif
    if(delta>0) then
     amounts=delta*weights/total
    else
     ! Source/exchange follows every RK flux evaluation, so a small excess
     ! suspended influx may already have left one particular boundary owner.
     ! Remove a negative reflux in proportion to the available inventory of
     ! the same phase on boundary owners; this retains positivity and locality.
     amounts=0
     do c=1,nc
      if(weights(c)>0) amounts(c)=max(0.d0,state(si,c)*area(c))
     enddo
     total=sum(amounts)
     if(total+delta>0.d0) then
      amounts=delta*amounts/total
     else
      ! This is only reachable after rapid phase exchange. Preserve the exact
      ! phase ledger using its available domain inventory in fixed cell order.
      do c=1,nc
       amounts(c)=max(0.d0,state(si,c)*area(c))
      enddo
      total=sum(amounts)
      if(total+delta>0.d0) then
       amounts=delta*amounts/total
      else
       amr_failure_cell=maxloc(state(si,:)*area,dim=1);amr_failure_state=si
       amr_failure_value=delta;amr_failure_aux=total;failure_reason=9;return
      endif
     endif
    endif
    largest=maxloc(abs(amounts),dim=1)
    residual=delta-sum(amounts)
    amounts(largest)=amounts(largest)+residual
    do c=1,nc
     if(amounts(c)==0) cycle
     state(si,c)=state(si,c)+amounts(c)/area(c)
    enddo
    where(state(si,:)<0.d0.and.abs(state(si,:))<= &
          512*epsilon(1.d0)*max(maxval(abs(state(si,:))),1.d0)) state(si,:)=0.d0
    if(any(state(si,:)<0).or..not.all(ieee_is_finite(state(si,:)))) then
     amr_failure_cell=minloc(state(si,:),dim=1);amr_failure_state=si
     amr_failure_value=state(si,amr_failure_cell);amr_failure_aux=delta
     failure_reason=9;return
    endif
   enddo
  enddo
  accepted=expected
  call rebalance_active_layer(state)
  ok=.true.
 end subroutine

 subroutine amr_advance_subcycled(dt,ok)
  real(8),intent(in) :: dt
  logical,intent(out) :: ok
  integer :: counts(0:max_amr_levels-1)
  real(8) :: secondary_source_trial,momentum_boundary_trial(2,2)
  real(8),allocatable :: boundary_transport(:,:)
  integer :: toe_flux_error,f,g,bi
  if(bank_toe_erosion_mode==2.and..not.bank_toe_mapping_ready) call build_physical_toe_map(q)
  if(exchange_mode==4.and.flow2dh_amr_ownership_declaration) &
   suspended_exchange_step_trial=0.d0
  if(bank_toe_erosion_mode==2) physical_toe_flux_trial=0.d0
  if(bank_toe_erosion_mode==3) distributed_face_mass_trial=0.d0
  associate(state=>amr_state_work,integral=>amr_subtree_flux(:,:,:,0),accepted=>amr_accepted_work)
  amr_dense_old=q;amr_dense_new=q;amr_dense_t0=time;amr_dense_t1=time;state=q;counts=0
  amr_failure_cell=0;amr_failure_state=0;amr_failure_value=0;amr_failure_aux=0
  secondary_source_trial=0.d0;momentum_boundary_trial=0.d0;secondary_momentum_boundary_trial=0.d0
  call amr_subcycle_level(0,time,dt,state,integral,accepted,counts,secondary_source_trial, &
                          momentum_boundary_trial,ok)
  if(.not.ok) amr_failed_level=amr_step_level
  amr_step_level=-1
  if(ok) then
   call amr_repair_water_reflux(-1,state,ok)
  endif
  if(ok) then
   call amr_reconcile_prescribed_boundaries(time,dt,state,integral,accepted,ok)
  endif
  if(ok) then
   call deposit_dry_suspended_after_reflux(state)
   if(bank_toe_erosion_mode==3) then
    ! The root register already contains the final fine-replaced, refluxed
    ! physical face integrals. Recording stage fluxes as well would double
    ! count both time stages and coarse/fine interfaces.
    do f=1,nf
     do g=1,nfrac
      bi=bed_index(g)
      distributed_face_mass_trial(g,1,f)=integral(bi,1,f)
      distributed_face_mass_trial(g,2,f)=integral(bi,2,f)
     enddo
    enddo
   endif
   call redistribute_eroded_banks(q,state)
   if(bank_front_tracking) then
    ! Finish every potentially rejecting Eulerian redistribution before the
    ! persistent front history is advanced.  A retry must always see the
    ! original state and geometry.
    call avalanche_erodible_banks(state,dt)
    if(.not.all(ieee_is_finite(state)).or.any(state(1,:)<0).or. &
       any(state(4:nphysical_state,:)<0)) then
     failure_reason=9;ok=.false.;return
    endif
    call apply_bank_failure(q,state,dt,ok)
   else
    call apply_bank_failure(q,state,dt,ok)
    if(ok) call avalanche_erodible_banks(state,dt)
   endif
   if(.not.ok) return
   ! The persistent mode already rebalanced and validated inside the final
   ! transactional bank commit.  Repeating a fallible Eulerian operation here
   ! would create a retry path after irreversible front history was advanced.
   if(.not.bank_front_tracking) call rebalance_active_layer(state)
   if(.not.all(ieee_is_finite(state)).or.any(state(1,:)<0).or.any(state(4:nphysical_state,:)<0)) then
    if(bank_front_tracking) then
     error stop 'invalid adaptive state after persistent bank-front commit'
    else
     amr_failure_state=minloc(minval(state,dim=2),dim=1)
     amr_failure_cell=minloc(state(amr_failure_state,:),dim=1)
     amr_failure_value=state(amr_failure_state,amr_failure_cell)
     failure_reason=9;ok=.false.
    endif
   else
    if(bank_toe_erosion_mode==2) then
     call braids_bank_toe_add_face_integrals(nf,nfrac,nstate,bed_index(1),owner,neighbour, &
      physical_is_toe_face,physical_candidate_bank,physical_candidate_toe,1.d0,integral, &
      physical_toe_flux_trial,toe_flux_error)
     if(toe_flux_error/=BANK_TOE_FLUX_SUCCESS) error stop 'invalid accepted refluxed physical bank-toe flux'
     call commit_physical_toe_flux()
    endif
    if(flow2dh_boundary_diffusion_declaration) then
     allocate(boundary_transport(1+nfrac,nf));boundary_transport=0.d0
     boundary_transport(1,:)=integral(1,1,:)
     call commit_flow2dh_boundary_memory(dt,boundary_transport,q)
     deallocate(boundary_transport)
    endif
    q=state;ledger=ledger+accepted
    if(prognostic_secondary_flow()) secondary_source_cumulative=secondary_source_cumulative+secondary_source_trial
    if(prognostic_secondary_flow()) secondary_momentum_boundary_trial=momentum_boundary_trial
    time=time+dt;steps=steps+1;amr_level_steps=amr_level_steps+counts
   endif
  endif
  end associate
 end subroutine

 recursive subroutine amr_choose_node(n,wanted,selected)
  integer,intent(in) :: n,wanted(amr_nn)
  logical,intent(inout) :: selected(amr_nn)
  integer :: k
  if(at_level(n)<wanted(n).and.at_offset(n+1)>at_offset(n)) then
   do k=at_offset(n),at_offset(n+1)-1
    call amr_choose_node(at_child(k),wanted,selected)
   enddo
  else
   selected(n)=.true.
  endif
 end subroutine

 subroutine amr_positive_transfer(values,weights,target)
  ! Preserve target mass while limiting reconstructed child inventories.
  real(8),intent(inout) :: values(:)
  real(8),intent(in) :: weights(:),target
  real(8) :: total,residual
  integer :: largest
  if(target<=0) then
   values=0;return
  endif
  values=max(values,0.d0);total=sum(values*weights)
  if(total>0) then
   values=values*(target/total)
  else
   values=target/sum(weights)
  endif
  largest=maxloc(values*weights,dim=1);residual=target-sum(values*weights)
  values(largest)=max(0.d0,values(largest)+residual/weights(largest))
 end subroutine

 subroutine amr_fill_water(volume,bed,weights,depth)
  real(8),intent(in) :: volume,bed(:),weights(:)
  real(8),intent(out) :: depth(:)
  integer :: order(size(bed)),k,j,key,largest
  real(8) :: wet_area,moment,stage
  if(volume<=0) then
   depth=0;return
  endif
  do k=1,size(bed)
   order(k)=k
  enddo
  do k=2,size(bed)
   key=order(k);j=k-1
   do while(j>=1)
    if(bed(order(j))<=bed(key)) exit
    order(j+1)=order(j);j=j-1
   enddo
   order(j+1)=key
  enddo
  wet_area=0;moment=0;stage=0
  do k=1,size(bed)
   j=order(k);wet_area=wet_area+weights(j);moment=moment+weights(j)*bed(j)
   stage=(volume+moment)/wet_area
   if(k==size(bed)) exit
   if(stage<=bed(order(k+1))) exit
  enddo
  depth=max(stage-bed,0.d0);largest=maxloc(depth*weights,dim=1)
  depth(largest)=max(0.d0,depth(largest)+(volume-sum(depth*weights))/weights(largest))
 end subroutine

 subroutine amr_signed_water_transfer(parent_mass,parent_depth,child_depth,weights,raw_intensity,values)
  ! Conservative, bounded prolongation of M_I=h*I. Bounds are imposed on
  ! signed intensity while the exact integral is carried by child water.
  real(8),intent(in) :: parent_mass,parent_depth,child_depth(:),weights(:),raw_intensity(:)
  real(8),intent(out) :: values(:)
  real(8) :: parent_intensity,lower,upper,target,residual,total_capacity,share,tolerance
  real(8) :: intensity(size(values)),capacity(size(values))
  integer :: k,largest
  if(parent_depth<=dry_tolerance) then
   tolerance=512*epsilon(1.d0)*max(1.d0,abs(parent_mass))
   if(abs(parent_mass)>tolerance) error stop 'dry AMR parent carries secondary-flow inventory'
   values=0.d0;return
  endif
  parent_intensity=parent_mass/parent_depth
  lower=parent_intensity;upper=parent_intensity
  do k=1,size(values)
   if(child_depth(k)<=dry_tolerance) cycle
   lower=min(lower,raw_intensity(k));upper=max(upper,raw_intensity(k))
  enddo
  do k=1,size(values)
   if(child_depth(k)<=dry_tolerance) then
    intensity(k)=0.d0;values(k)=0.d0
   else
    intensity(k)=max(lower,min(upper,raw_intensity(k)))
    values(k)=child_depth(k)*intensity(k)
   endif
  enddo
  ! amr_fill_water preserves parent water exactly. Multiplying its child
  ! integral by I_parent therefore gives A_parent*M_parent.
  target=parent_intensity*sum(child_depth*weights)
  residual=target-sum(values*weights)
  tolerance=1024*epsilon(1.d0)*max(1.d0,abs(target),sum(abs(values*weights)))
  if(abs(residual)<=tolerance) then
   largest=maxloc(child_depth*weights,dim=1)
   if(child_depth(largest)>dry_tolerance) values(largest)=values(largest)+residual/weights(largest)
   return
  endif
  if(residual>0.d0) then
   capacity=weights*child_depth*max(upper-intensity,0.d0)
  else
   capacity=weights*child_depth*max(intensity-lower,0.d0)
  endif
  total_capacity=sum(capacity)
  if(total_capacity+tolerance<abs(residual)) error stop 'AMR secondary-flow prolongation has insufficient capacity'
  if(total_capacity<=0.d0) error stop 'AMR secondary-flow prolongation cannot close signed integral'
  do k=1,size(values)
   if(capacity(k)<=0.d0) cycle
   share=residual*capacity(k)/total_capacity
   values(k)=values(k)+share/weights(k)
  enddo
  residual=target-sum(values*weights);largest=maxloc(capacity,dim=1)
  values(largest)=values(largest)+residual/weights(largest)
  if(abs(target-sum(values*weights))>4*tolerance) error stop 'AMR secondary-flow prolongation did not conserve'
 end subroutine

 subroutine amr_deposit_selected_dry_suspension(selected,old_leaf,exchange)
  ! Restriction first forms exact area-weighted conserved rows.  Only after
  ! the new non-overlapping leaf set is known may a dry parent take ownership
  ! of its suspended phase; modifying every cached parent would book phantom
  ! exchange for nodes that never become part of the physical mesh.
  logical,intent(in) :: selected(amr_nn),old_leaf(amr_nn)
  real(8),intent(out) :: exchange(2,nfrac)
  integer :: n,g,si,bi
  real(8) :: suspended
  exchange=0.d0
  if(.not.flow2dh_amr_ownership_declaration) return
  do n=1,amr_nn
   if(.not.selected(n).or.old_leaf(n).or.at_cover(n)/=0) cycle
   if(at_q(1,n)>dry_tolerance) cycle
   ! Restriction can average shallow wet children into a hydraulically dry
   ! parent.  Momentum has no dry-phase reservoir and is discarded by the
   ! same accepted wet/dry safety rule used by the hydraulic source update.
   at_q(2:3,n)=0.d0
   do g=1,nfrac
    si=suspended_index(g);bi=bed_index(g);suspended=at_q(si,n)
    if(suspended<=0.d0) cycle
    at_q(si,n)=0.d0;at_q(bi,n)=at_q(bi,n)+suspended
    exchange(1,g)=exchange(1,g)-at_area(n)*suspended
    exchange(2,g)=exchange(2,g)+at_area(n)*suspended
   enddo
  enddo
 end subroutine

 subroutine amr_prolong_node(n,gradient)
  integer,intent(in) :: n
  real(8),intent(in) :: gradient(nprimitive,2)
  integer :: count,k,child,j,g,si,ci
  real(8) :: mean,target,water,velocity(2),concentration
  real(8),allocatable :: weight(:),bed(:),values(:),depth(:),position(:,:),data(:,:)
  count=at_offset(n+1)-at_offset(n)
  if(count==0) return
  allocate(weight(count),bed(count),values(count),depth(count),position(2,count),data(nstate,count))
  do k=1,count
   child=at_child(at_offset(n)+k-1);weight(k)=at_area(child);position(:,k)=at_xy(:,child)-at_xy(:,n)
   data(:,k)=at_q(:,child)
  enddo
  ! Preserve cached stratigraphy details and apply only the accumulated coarse
  ! inventory change. Limiting is conservative independently for every layer.
  do j=4+nfrac,nphysical_state
   mean=sum(data(j,:)*weight)/sum(weight)
   values=data(j,:)+(at_q(j,n)-mean)
   call amr_positive_transfer(values,weight,at_area(n)*at_q(j,n))
   data(j,:)=values
  enddo
  do k=1,count
   child=at_child(at_offset(n)+k-1)
   at_q(4+nfrac:nphysical_state,child)=data(4+nfrac:nphysical_state,k)
   bed(k)=amr_node_bed(child)-dot_product(gradient(1,:),position(:,k))
  enddo
  water=at_area(n)*at_q(1,n)
  call amr_fill_water(water,bed,weight,depth);data(1,:)=depth
  velocity=0
  if(at_q(1,n)>dry_tolerance) velocity=at_q(2:3,n)/at_q(1,n)
  do j=2,3
   do k=1,count
    data(j,k)=depth(k)*(velocity(j-1)+dot_product(gradient(j,:),position(:,k)))
   enddo
   if(water>0) data(j,:)=data(j,:)+depth*(at_area(n)*at_q(j,n)-sum(data(j,:)*weight))/water
  enddo
  do g=1,nfrac
   si=suspended_index(g);ci=concentration_index(g);concentration=0
   if(at_q(1,n)>dry_tolerance) concentration=at_q(si,n)/at_q(1,n)
   do k=1,count
    values(k)=depth(k)*max(0.d0,concentration+dot_product(gradient(ci,:),position(:,k)))
   enddo
   call amr_positive_transfer(values,weight,at_area(n)*at_q(si,n));data(si,:)=values
  enddo
  if(prognostic_secondary_flow()) then
   concentration=0.d0
   if(at_q(1,n)>dry_tolerance) concentration=at_q(secondary_state_index(),n)/at_q(1,n)
   do k=1,count
    values(k)=concentration+dot_product(gradient(secondary_primitive_index(),:),position(:,k))
   enddo
   call amr_signed_water_transfer(at_q(secondary_state_index(),n),at_q(1,n),depth,weight,values, &
    data(secondary_state_index(),:))
  endif
  do k=1,count
   at_q(:,at_child(at_offset(n)+k-1))=data(:,k)
  enddo
  deallocate(weight,bed,values,depth,position,data)
 end subroutine

 subroutine amr_coarsening_bed_detail(bed,near_water,raw)
  real(8),intent(in) :: bed(nc)
  logical,intent(in) :: near_water(nc)
  integer,intent(inout) :: raw(amr_nn)
  real(8) :: moments(5,amr_nn),slope(2,amr_nn),delta(2),dz,w,det,residual
  logical :: wet_parent(amr_nn)
  integer :: c,n
  ! Fine-cell neighbour indicators alone do not bound the terrain detail lost
  ! when several generations of leaves are replaced by one larger cell. Fit
  ! each prospective parent to its current leaves, then veto coarsening if the
  ! unresolved bed relief exceeds the same configured bathymetry tolerance.
  ! Centered moments avoid cancellation with projected GIS coordinates. A
  ! planar slope remains coarsenable; dry terrain away from water is untouched.
  moments=0;slope=0;wet_parent=.false.
  do c=1,nc
   n=at_parent(at_leaf(c))
   do while(n>0)
    delta=xy(:,c)-at_xy(:,n);dz=bed(c)-amr_node_bed(n);w=area(c)
    moments(:,n)=moments(:,n)+w*[delta(1)**2,delta(1)*delta(2),delta(2)**2,delta(1)*dz,delta(2)*dz]
    wet_parent(n)=wet_parent(n).or.near_water(c)
    n=at_parent(n)
   enddo
  enddo
  do n=1,amr_nn
   if(.not.wet_parent(n)) cycle
   det=moments(1,n)*moments(3,n)-moments(2,n)**2
   if(det>1.d-12*moments(1,n)*moments(3,n)) then
    slope(1,n)=(moments(3,n)*moments(4,n)-moments(2,n)*moments(5,n))/det
    slope(2,n)=(moments(1,n)*moments(5,n)-moments(2,n)*moments(4,n))/det
   else if(moments(1,n)+moments(3,n)>0) then
    ! Rank-one families (e.g. narrow cut strips): pseudoinverse along the
    ! resolved line, without spuriously treating a linear bed as roughness.
    slope(:,n)=moments(4:5,n)/(moments(1,n)+moments(3,n))
   endif
  enddo
  do c=1,nc
   n=at_parent(at_leaf(c))
   do while(n>0)
    if(wet_parent(n)) then
     residual=abs(bed(c)-amr_node_bed(n)-dot_product(slope(:,n),xy(:,c)-at_xy(:,n)))
     if(residual>amr_bed_tol) raw(n)=max(raw(n),at_level(n)+1)
    endif
    n=at_parent(n)
   enddo
  enddo
 end subroutine

 subroutine amr_propagate_forecast(reach)
  ! Resolve the maximum forecast distance from every flagged source without
  ! repeatedly sweeping every active face.  This is the same max-relaxation
  ! solved by the former synchronous scans: a node's key is the remaining
  ! distance and every face subtracts its geometric length.  A deterministic
  ! max heap visits a positive-reach node once, so isolated fronts no longer
  ! make a full-domain serial face pass at every AMR regrid.
  real(8),intent(inout) :: reach(nc)
  integer :: heap(nc),position(nc),heap_count,c,node,k,f,other
  real(8) :: candidate,distance
  position=0;heap_count=0
  do c=1,nc
   if(reach(c)>=0.d0) call push(c)
  enddo
  do while(heap_count>0)
   node=pop()
   do k=offset(node),offset(node+1)-1
    f=faces(k);other=neighbour(f)
    if(signs(k)<0) other=owner(f)
    if(other<=0) cycle
    distance=sqrt(sum((xy(:,node)-xy(:,other))**2))
    candidate=reach(node)-distance
    ! A negative remainder can never make a later cell positive because all
    ! face distances are positive.  It therefore has no bearing on the
    ! following reach>=0 selection while avoiding needless queue entries.
    if(candidate<0.d0.or.candidate<=reach(other)) cycle
    reach(other)=candidate
    if(position(other)==0) then
     call push(other)
    else
     call sift_up(position(other))
    endif
   enddo
  enddo
 contains
  logical function higher(left,right)
   integer,intent(in) :: left,right
   higher=reach(heap(left))>reach(heap(right))
   if(reach(heap(left))==reach(heap(right))) higher=heap(left)<heap(right)
  end function

  subroutine swap(left,right)
   integer,intent(in) :: left,right
   integer :: saved
   saved=heap(left);heap(left)=heap(right);heap(right)=saved
   position(heap(left))=left;position(heap(right))=right
  end subroutine

  subroutine sift_up(index)
   integer,intent(in) :: index
   integer :: child,parent
   child=index
   do while(child>1)
    parent=child/2
    if(.not.higher(child,parent)) exit
    call swap(child,parent);child=parent
   enddo
  end subroutine

  subroutine sift_down(index)
   integer,intent(in) :: index
   integer :: parent,child
   parent=index
   do while(2*parent<=heap_count)
    child=2*parent
    if(child<heap_count) then
     if(higher(child+1,child)) child=child+1
    endif
    if(.not.higher(child,parent)) exit
    call swap(child,parent);parent=child
   enddo
  end subroutine

  subroutine push(value)
   integer,intent(in) :: value
   heap_count=heap_count+1;heap(heap_count)=value;position(value)=heap_count
   call sift_up(heap_count)
  end subroutine

  integer function pop() result(value)
   value=heap(1);position(value)=0
   if(heap_count==1) then
    heap_count=0;return
   endif
   heap(1)=heap(heap_count);position(heap(1))=1;heap_count=heap_count-1
   call sift_down(1)
  end function
 end subroutine

 subroutine amr_regrid(initial)
  logical,intent(in) :: initial
  integer :: c,n,p,k,f,b,g,ci,a,other,ring,number,i,old_count,coarse,first_wet,last_wet,profile_iterations
  integer :: wanted(amr_nn),raw(amr_nn),buffered(amr_nn),cover(amr_nn),cause(nc)
  integer :: cause_counts(0:16),selected_counts(0:max_amr_levels-1)
  real(8) :: bed(nc),v(nprimitive,nc),grad(nprimitive,2,nc),vlo(nprimitive,nc),vhi(nprimitive,nc)
  real(8) :: bc_c(nfrac,nb),bc_b(nfrac,nb),bc_hydraulic(2,nb),front_speed(nb)
  real(8) :: speed,eta,child_bed,minbed,maxbed,change,ha,hb,sa,sb,total,weight,h,ub,zavg,wet_length
  real(8) :: profile_driving,profile_discharge_residual,profile_acceleration_residual
  real(8),allocatable :: mode6_profile_speed(:)
 real(8) :: tree_gradient(nprimitive,2,amr_nn),before(nstate),after(nstate),moment(nstate)
 real(8) :: regrid_exchange(2,nfrac)
  real(8) :: reach(nc),remaining,cause_area(0:16)
 real(8) :: profile_start
  logical :: selected(amr_nn),refine(amr_nn),change_mesh,wet,nearwet,old_leaf(amr_nn)
  logical :: wet_front(nc),hydraulic_ok,near_water(nc),moving_source,profile_ok
  if(.not.amr_enabled) return
  if(profile_enabled) profile_start=omp_get_wtime()
  call amr_sync_state()
  call primitive(q,v,bed)
  call boundary_sediment_at(time,bc_c,bc_b)
  call gradients(v,bc_c,grad,vlo,vhi)
  raw=at_min;tree_gradient=0;old_leaf=.false.;before=0;wet_front=.false.;cause=0
  do c=1,nc
   n=at_leaf(c);old_leaf(n)=.true.;tree_gradient(:,:,n)=grad(:,:,c)
   before=before+area(c)*q(:,c)
   wet=q(1,c)>amr_wet_depth;nearwet=wet
   if(at_min(n)>0) cause(c)=ibset(cause(c),0)
   if(bank_front_tracking) then
    ! Front state is cell-local and deliberately never restricted into a
    ! fictitious coarse prism.  Keep the complete erodible-bank support at
    ! the finest level, including promoted cells, so activation, restart and
    ! subsequent migration always retain one stable leaf identity.
    if(bank_erodible(c).or.bank_front_status(c)/=BANK_FRONT_INACTIVE) then
     raw(n)=amr_levels-1;cause(c)=ibset(cause(c),16)
    endif
   endif
   if(wet) then
    raw(n)=max(raw(n),amr_wet_level);cause(c)=ibset(cause(c),1)
   endif
   do k=offset(c),offset(c+1)-1
    f=faces(k);other=neighbour(f)
    if(signs(k)<0) other=owner(f)
    if(other>0) then
     if(q(1,other)>amr_wet_depth) nearwet=.true.
     if(amr_flags(1)==1.and.(wet.neqv.(q(1,other)>amr_wet_depth))) then
      raw(n)=amr_levels-1;raw(at_leaf(other))=amr_levels-1
      cause(c)=ibset(cause(c),2);cause(other)=ibset(cause(other),2)
      if(wet) wet_front(c)=.true.
      if(q(1,other)>amr_wet_depth) wet_front(other)=.true.
     endif
     if(amr_flags(4)==1.and.wet.and.q(1,other)>amr_wet_depth) then
      if(abs(v(1,c)-v(1,other))>amr_eta_tol.or. &
       sqrt(sum((v(2:3,c)-v(2:3,other))**2))>amr_u_tol) then
       raw(n)=amr_levels-1;cause(c)=ibset(cause(c),3)
      endif
     endif
     if(amr_flags(5)==1.and.nearwet) then
      if(any(abs(v(4:3+nfrac,c)-v(4:3+nfrac,other))>amr_c_tol)) then
       raw(n)=amr_levels-1;cause(c)=ibset(cause(c),4)
      endif
      if(at_bank(n).and.abs(bed(c)-bed(other))>amr_dz_tol) then
       raw(n)=amr_levels-1;cause(c)=ibset(cause(c),5)
      endif
      sa=sum(q(4+nfrac:3+2*nfrac,c)/bulk_fraction);sb=sum(q(4+nfrac:3+2*nfrac,other)/bulk_fraction)
      if(sa>0.and.sb>0) then
       if(any(abs(q(4+nfrac:3+2*nfrac,c)/(sa*bulk_fraction)- &
        q(4+nfrac:3+2*nfrac,other)/(sb*bulk_fraction))>amr_mix_tol)) then
        raw(n)=amr_levels-1;cause(c)=ibset(cause(c),6)
       endif
      endif
     endif
     if(nearwet.and.(manning_cell(c)/=manning_cell(other).or.(bank_erodible(c).neqv.bank_erodible(other)))) then
      raw(n)=max(raw(n),min(2,amr_levels-1));cause(c)=ibset(cause(c),7)
     endif
     if(bank_front_tracking) then
      if(bank_erodible(c).and. &
         ((.not.bank_erodible(other)).or.bank_front_status(other)==BANK_FRONT_PROMOTED)) then
       raw(n)=amr_levels-1;raw(at_leaf(other))=amr_levels-1
       cause(c)=ibset(cause(c),16);cause(other)=ibset(cause(other),16)
      endif
     endif
    else
     if(bid(f)>0) then
      raw(n)=amr_levels-1;cause(c)=ibset(cause(c),8)
     endif
     if(amr_flags(2)==1.and.wet.and.bid(f)==0) then
      raw(n)=amr_levels-1;cause(c)=ibset(cause(c),9)
     endif
    endif
   enddo
   near_water(c)=nearwet
   if(amr_flags(3)==1.and.nearwet) then
    change=0
    do k=at_offset(n),at_offset(n+1)-1
     p=at_child(k);child_bed=amr_node_bed(p)
     change=max(change,abs(child_bed-bed(c)-dot_product(grad(bed_primitive_index(),:,c),at_xy(:,p)-xy(:,c))))
    enddo
    do k=offset(c),offset(c+1)-1
     f=faces(k);other=neighbour(f)
     if(signs(k)<0) other=owner(f)
     if(other<=0) cycle
     change=max(change,abs(bed(other)-bed(c)-dot_product(grad(bed_primitive_index(),:,c),xy(:,other)-xy(:,c))))
    enddo
    if(change>amr_bed_tol) then
     raw(n)=amr_levels-1;cause(c)=ibset(cause(c),10)
    endif
   endif
   if(amr_flags(5)==1.and.abs(bed(c)-at_last_bed(n))>amr_dz_tol) then
    raw(n)=amr_levels-1;cause(c)=ibset(cause(c),11)
   endif
   if(amr_flags(1)==1.and.wet) then
    eta=v(1,c)
    do k=at_offset(n),at_offset(n+1)-1
     if(amr_node_bed(at_child(k))>=eta-amr_wet_depth) then
      raw(n)=amr_levels-1;cause(c)=ibset(cause(c),12)
     endif
    enddo
   endif
  enddo
  if(amr_flags(3)==1) then
   buffered=raw
   call amr_coarsening_bed_detail(bed,near_water,raw)
   do c=1,nc
    p=at_leaf(c)
    do while(p>0)
     if(raw(p)>buffered(p)) cause(c)=ibset(cause(c),13)
     p=at_parent(p)
    enddo
   enddo
  endif
  ! Protect a front's reconstruction stencil and its travel before the next
  ! adaptation. The dry-bed shallow-water front travels at u+2c, rather than
  ! the ordinary characteristic speed u+c. Prescribed inlet traces provide
  ! the same conservative forecast even when every interior owner is dry.
  ! The fine buffer is grown deterministically on active faces.
  front_speed=0
  call boundary_hydraulic_at(time,bed,bc_hydraulic,hydraulic_ok)
  if(hydraulic_ok) then
   do b=1,nb
    if(bc_hydraulic(1,b)<=0) cycle
    if(mode(b)==1.or.mode(b)==9) then
     call sampled_section_span(b,bc_hydraulic(2,b),bed,first_wet,last_wet)
     total=0
     do k=first_wet,last_wet
      f=boundary_faces(k);c=owner(f)
      h=max(bc_hydraulic(2,b)-bed(c)-boundary_bed_offset(f),0.d0)
      if(h<=dry_tolerance) cycle
      total=total+h*length(f)
     enddo
     if(total>0) then
      ub=bc_hydraulic(1,b)/total
      do k=first_wet,last_wet
       f=boundary_faces(k);c=owner(f)
       h=max(bc_hydraulic(2,b)-bed(c)-boundary_bed_offset(f),0.d0)
       if(h>dry_tolerance) front_speed(b)=max(front_speed(b),ub+2*sqrt(grav*h))
      enddo
     endif
    else if(mode(b)==7.or.mode(b)==8) then
     total=0
     do k=boundary_offset(b),boundary_offset(b+1)-1
      f=boundary_faces(k);c=owner(f)
      h=bc_hydraulic(2,b)
      if(mode(b)==7) h=max(bc_hydraulic(2,b)-bed(c),0.d0)
      if(h<=dry_tolerance) cycle
      if(hydraulic_roughness_mode==braids_roughness_manning.and. &
         roughness_minimum_work_depth==0.d0) then
       weight=h**1.5d0*max(h,manning_chezy_minimum_depth)**(1.d0/6.d0)
       if(manning_cell(c)>0) weight=weight/manning_cell(c)
      else
       weight=roughness_face_weight(h,h,manning_cell(c))
      endif
      total=total+weight*length(f)
     enddo
     if(total>0) then
      do k=boundary_offset(b),boundary_offset(b+1)-1
       f=boundary_faces(k);c=owner(f)
       h=bc_hydraulic(2,b)
       if(mode(b)==7) h=max(bc_hydraulic(2,b)-bed(c),0.d0)
       if(h<=dry_tolerance) cycle
       if(hydraulic_roughness_mode==braids_roughness_manning.and. &
          roughness_minimum_work_depth==0.d0) then
        weight=h**1.5d0*max(h,manning_chezy_minimum_depth)**(1.d0/6.d0)
        if(manning_cell(c)>0) weight=weight/manning_cell(c)
       else
        weight=roughness_face_weight(h,h,manning_cell(c))
       endif
       ub=bc_hydraulic(1,b)*weight/(total*h)
       front_speed(b)=max(front_speed(b),ub+2*sqrt(grav*h))
      enddo
    endif
   else if(mode(b)==6) then
     if(viscosity_equilibrated_inlet(b)) then
      if(.not.allocated(mode6_profile_speed)) allocate(mode6_profile_speed(nf))
      call viscosity_equilibrated_mode6_profile(b,q,bc_hydraulic(1,b),mode6_profile_speed, &
       profile_driving,profile_discharge_residual,profile_acceleration_residual, &
       profile_iterations,profile_ok)
      if(profile_ok) then
       do k=boundary_offset(b),boundary_offset(b+1)-1
        f=boundary_faces(k);c=owner(f);h=q(1,c)
        if(h<=dry_tolerance) cycle
        front_speed(b)=max(front_speed(b),mode6_profile_speed(f)+2*sqrt(grav*h))
       enddo
      endif
     else
      zavg=0;wet_length=0
      do k=boundary_offset(b),boundary_offset(b+1)-1
       f=boundary_faces(k);c=owner(f);h=q(1,c)
       if(h<=dry_tolerance) cycle
       zavg=zavg+(bed(c)+h)*length(f);wet_length=wet_length+length(f)
      enddo
      if(wet_length>0) zavg=zavg/wet_length
      total=0
      do k=boundary_offset(b),boundary_offset(b+1)-1
       f=boundary_faces(k);c=owner(f);h=q(1,c)
       if(h<=dry_tolerance) cycle
       if(hydraulic_roughness_mode==braids_roughness_manning.and. &
          roughness_minimum_work_depth==0.d0) then
        weight=max(zavg-bed(c),0.d0)**1.5d0* &
         max(h,manning_chezy_minimum_depth)**(1.d0/6.d0)
        if(manning_cell(c)>0) weight=weight/manning_cell(c)
       else
        weight=roughness_face_weight(max(zavg-bed(c),0.d0),h,manning_cell(c))
       endif
       total=total+weight*length(f)
      enddo
      if(total>0) then
       do k=boundary_offset(b),boundary_offset(b+1)-1
        f=boundary_faces(k);c=owner(f);h=q(1,c)
        if(h<=dry_tolerance) cycle
        if(hydraulic_roughness_mode==braids_roughness_manning.and. &
           roughness_minimum_work_depth==0.d0) then
         weight=max(zavg-bed(c),0.d0)**1.5d0* &
          max(h,manning_chezy_minimum_depth)**(1.d0/6.d0)
         if(manning_cell(c)>0) weight=weight/manning_cell(c)
        else
         weight=roughness_face_weight(max(zavg-bed(c),0.d0),h,manning_cell(c))
        endif
        ub=bc_hydraulic(1,b)*weight/(total*h)
        front_speed(b)=max(front_speed(b),ub+2*sqrt(grav*h))
       enddo
      endif
     endif
    endif
    wet=.false.
    do k=boundary_offset(b),boundary_offset(b+1)-1
     if(q(1,owner(boundary_faces(k)))>amr_wet_depth) wet=.true.
    enddo
    ! A dry start has no interior state from which to measure the very thin
    ! numerical leading edge. Reserve the configured admissible inlet-speed
    ! distance for its first adaptation interval; later forecasts use the
    ! resolved wet-front state above.
    if(.not.wet.and.front_speed(b)>0) front_speed(b)=max(front_speed(b),speed_limit)
   enddo
  endif
  reach=-huge(1.d0)
  do c=1,nc
   if(raw(at_leaf(c))/=amr_levels-1) cycle
   ! Every finest-level trigger keeps the configured geometric reconstruction
   ! stencil. Only fronts that can move before the next regrid receive the
   ! additional characteristic travel distance. Static bathymetry, material
   ! interfaces and solid walls do not advect through the mesh.
   reach(c)=amr_buffer*max(dx,dy)
   moving_source=btest(cause(c),2).or.btest(cause(c),3).or.btest(cause(c),4).or. &
    btest(cause(c),5).or.btest(cause(c),6).or.btest(cause(c),11).or.btest(cause(c),12)
   if(.not.moving_source) cycle
   speed=0
   if(q(1,c)>amr_wet_depth) then
    speed=sqrt(sum(v(2:3,c)**2))+sqrt(grav*q(1,c))
    if(wet_front(c)) speed=sqrt(sum(v(2:3,c)**2))+2*sqrt(grav*q(1,c))
   endif
   reach(c)=reach(c)+speed*amr_interval
  enddo
  do b=1,nb
   if(front_speed(b)<=0) cycle
   do k=boundary_offset(b),boundary_offset(b+1)-1
    f=boundary_faces(k);c=owner(f)
    reach(c)=max(reach(c),amr_buffer*max(dx,dy)+front_speed(b)*amr_interval)
   enddo
  enddo
  call amr_propagate_forecast(reach)
  do c=1,nc
   if(reach(c)>=0) then
    raw(at_leaf(c))=amr_levels-1;cause(c)=ibset(cause(c),14)
   endif
  enddo
  ! A finer level is kept until every affected leaf has remained unflagged for
  ! the requested number of accepted regrid events.
  if(.not.initial) then
   do c=1,nc
    n=at_leaf(c)
    if(raw(n)>=at_level(n)) then
     at_delay(n)=0
    else
     at_delay(n)=at_delay(n)+1
     if(at_delay(n)<amr_delay) then
      raw(n)=at_level(n);cause(c)=ibset(cause(c),15)
     endif
    endif
   enddo
  endif
  wanted=raw
  do n=1,amr_nn
   p=at_parent(n)
   if(p>0) wanted(n)=max(wanted(n),wanted(p))
  enddo
  do n=amr_nn,1,-1
   p=at_parent(n)
   if(p>0) wanted(p)=max(wanted(p),wanted(n))
  enddo
  wanted=min(wanted,at_max)
  selected=.false.
  do n=1,amr_nn
   if(at_parent(n)==0) call amr_choose_node(n,wanted,selected)
  enddo
  ring=0
  do
   ring=ring+1
   if(ring>amr_nn) error stop 'AMR 2:1 balance did not converge'
   cover=0
   do n=1,amr_nn
    if(selected(n)) cover(n)=n
    p=at_parent(n)
    if(p>0) then
     if(cover(p)>0) cover(n)=cover(p)
    endif
   enddo
   refine=.false.
   do f=1,amr_fn
    if(af_neighbour(f)==0) cycle
    a=cover(af_owner(f));b=cover(af_neighbour(f))
    if(a==0.or.b==0.or.a==b) cycle
    if(abs(at_level(a)-at_level(b))<=1) cycle
    coarse=a
    if(at_level(b)<at_level(a)) coarse=b
    if(at_level(coarse)>=at_max(coarse).or.at_offset(coarse+1)<=at_offset(coarse)) &
     error stop 'AMR safety cap is incompatible with 2:1 balance'
    refine(coarse)=.true.
   enddo
   if(.not.any(refine)) exit
   do n=1,amr_nn
    if(.not.refine(n)) cycle
    selected(n)=.false.
    selected(at_child(at_offset(n):at_offset(n+1)-1))=.true.
   enddo
  enddo
  if(bank_front_tracking) then
   do n=1,amr_nn
    if(.not.selected(n)) cycle
    if(at_bank_front_status(n)/=BANK_FRONT_INACTIVE.and. &
       at_level(n)/=amr_levels-1) &
     error stop 'persistent bank front cannot be represented on a coarse AMR leaf'
   enddo
  endif
  if(amr_diagnostics) then
   cause_counts=0;cause_area=0;selected_counts=0
   do i=0,16
    cause_counts(i)=count(btest(cause,i))
    cause_area(i)=sum(area,mask=btest(cause,i))
   enddo
   do n=1,amr_nn
    if(selected(n)) selected_counts(at_level(n))=selected_counts(at_level(n))+1
   enddo
   do c=1,nc
    n=at_leaf(c)
    write(amr_diagnostic_unit,*) time,amr_regrid_count+1,c,n,at_level(n),raw(n),wanted(n),cause(c),area(c),xy(:,c)
   enddo
   write(amr_diagnostic_summary_unit,*) time,amr_regrid_count+1,nc, &
    selected_counts(0:amr_levels-1),cause_counts,cause_area
   flush(amr_diagnostic_unit);flush(amr_diagnostic_summary_unit)
  endif
  change_mesh=any(selected.neqv.old_leaf);regrid_exchange=0.d0
  if(change_mesh) then
   ! Restriction has already constructed every candidate parent in at_q.  Close
   ! suspended/bed phase ownership only for parents that will actually replace
   ! their old fine leaves.  Keep the signed transfer local until the complete
   ! rebuilt hierarchy has passed its conservation check below.
   call amr_deposit_selected_dry_suspension(selected,old_leaf,regrid_exchange)
   ! Traverse old leaves toward any newly requested descendants. Reconstruct
   ! at each split so newly refined regions start from the evolved state.
    do n=1,amr_fine_offset
    if(cover(n)/=0) cycle
    if(at_cover(n)==0) cycle
    p=at_parent(n)
    if(.not.old_leaf(n).and.p>0) tree_gradient(:,:,n)=tree_gradient(:,:,p)
     call amr_prolong_node(n,tree_gradient(:,:,n))
    enddo
   if(bank_failure) then
    ! A newly exposed descendant inherits the effective undercut history of
    ! the old covering leaf.  A newly selected parent already contains the
    ! conservative restriction produced by amr_sync_bank_failure_state.
    do n=1,amr_nn
     if(.not.selected(n)) cycle
     p=at_cover(n)
     if(p<=0.or.p==n) cycle
     at_bank_undercut(n)=at_bank_undercut(p)
     at_bank_reference_height(n)=at_bank_reference_height(p)
     at_bank_toe_height(n)=at_bank_toe_height(p)
     at_bank_reference_length(n)=at_bank_reference_length(p)
     at_bank_toe_cumulative(n)=at_bank_toe_cumulative(p)
     at_bank_collapse_cumulative(n)=at_bank_collapse_cumulative(p)
     at_bank_collapse_count(n)=at_bank_collapse_count(p)
    enddo
   endif
   if(bank_front_tracking) then
    ! A newly refined bank leaf may inherit only an untouched material cell.
    ! Active and promoted fronts are pinned to finest leaves above, so
    ! prolonging either one would duplicate its prism history.
    do n=1,amr_nn
     if(.not.selected(n)) cycle
     p=at_cover(n)
     if(p<=0.or.p==n) cycle
     if(at_bank_front_status(p)/=BANK_FRONT_INACTIVE) &
      error stop 'AMR attempted to prolong a persistent bank front'
     at_bank_front_status(n)=BANK_FRONT_INACTIVE
     at_bank_front_seed_face(n)=0
     at_bank_front_solid_area(n)=at_area(n)
     at_bank_front_base(n)=0.d0;at_bank_front_height(n)=0.d0
     at_bank_front_removed_mass(:,n)=0.d0
    enddo
   endif
   deallocate(at_leaf);allocate(at_leaf(count(selected)))
   number=0
   do n=1,amr_nn
    if(.not.selected(n)) cycle
    number=number+1;at_leaf(number)=n
   enddo
   call amr_rebuild_mesh();amr_mesh_id=amr_mesh_id+1
  endif
  ! Initial selection can retain every finest leaf. In that case no rebuild
  ! occurred, but a fully wet step can immediately enter the subcycled path.
  if(.not.allocated(amr_state_work)) then
   call amr_cache_level_work()
   call amr_allocate_step_workspaces()
  endif
  after=0
  do c=1,nc
   after=after+area(c)*q(:,c);at_last_bed(at_leaf(c))=amr_node_bed(at_leaf(c))
  enddo
  moment=after-before
  if(flow2dh_amr_ownership_declaration) then
   do g=1,nfrac
    moment(suspended_index(g))=moment(suspended_index(g))-regrid_exchange(1,g)
    moment(bed_index(g))=moment(bed_index(g))-regrid_exchange(2,g)
   enddo
  endif
  if(any(abs(moment)>1.d-10*max(abs(before),1.d0))) &
   error stop 'AMR regrid did not conserve state and declared phase transfer'
  if(flow2dh_amr_ownership_declaration.and..not.initial) &
   suspended_exchange_cumulative=suspended_exchange_cumulative+regrid_exchange
  amr_regrid_count=amr_regrid_count+1
  amr_next=amr_regrid_count*amr_interval
  if(profile_enabled) then
   profile_regrid_s=profile_regrid_s+omp_get_wtime()-profile_start
   profile_regrid_calls=profile_regrid_calls+1
  endif
 end subroutine

 subroutine amr_rebuild_mesh()
  integer :: c,n,f,k,p,number(amr_nn),cursor(size(at_leaf)),bcursor(nb),old_nf
  real(8) :: closure(2),tol
  logical,allocatable :: old_open_memory_face(:),new_open_memory_face(:)
  if(at_roughness_mode/=hydraulic_roughness_mode.or. &
     all(at_roughness_mode/=[braids_roughness_chezy,braids_roughness_manning, &
      braids_roughness_white_colebrook])) &
   error stop 'AMR raw roughness method identity changed during transfer'
  old_nf=nf;nc=size(at_leaf);nf=0
  if(flow2dh_amr_ownership_declaration) then
   if(.not.allocated(at_face_source)) &
    error stop 'v49 adaptive boundary-memory topology has no source identity'
   allocate(old_open_memory_face(amr_fn),new_open_memory_face(amr_fn))
   old_open_memory_face=.false.;new_open_memory_face=.false.
   do f=1,old_nf
    if(bid(f)<=0.or.neighbour(f)>0) cycle
    if(at_face_source(f)<1.or.at_face_source(f)>amr_fn) &
     error stop 'v49 adaptive boundary-memory source identity is invalid'
    old_open_memory_face(at_face_source(f))=.true.
   enddo
  endif
  call amr_update_cover()
  do f=1,amr_fn
   if(amr_use_face(f)) nf=nf+1
  enddo
  ! Geometry caches are rebuilt only on a changed hierarchy, never in RK stages.
  deallocate(area,xy,rock,q,ii,jj,grid,manning_cell,hydro_constant,active_cell,bank_erodible)
  deallocate(owner,neighbour,bid,normal,length,mid,boundary_bed_offset,boundary_bed_left_offset, &
   boundary_bed_right_offset,sampled_boundary_bed,sampled_boundary_open,section_cached_bed, &
   offset,faces,signs,boundary_faces,rectangle_width,weno_cells,face_displacement, &
   open_boundary_distance,uncoupled_boundary_distance)
  if(allocated(at_face_source)) deallocate(at_face_source)
  if(allocated(wall_triangle)) deallocate(wall_triangle,wall_ready,wall_slot,wall_stencil_ids,wall_stencil_kind, &
   wall_stencil_normal,wall_box_center,wall_normal,wall_mean_distance,wall_values)
  if(allocated(bedq_cell)) deallocate(bedq_cell,bedq_equilibrium,bedq_relax_work,bedq_grad,bedtau_cell)
  if(allocated(suspended_capacity_cell)) &
   deallocate(suspended_capacity_cell,suspended_capacity_equilibrium,capacity_absolute_shields, &
    capacity_critical_shields,capacity_transport_stage)
  if(allocated(suspended_exchange_cell_trial)) &
   deallocate(suspended_exchange_cell_trial)
  if(allocated(split_boundary_role)) deallocate(split_boundary_role,split_boundary_bed, &
   split_boundary_suspended,split_boundary_absolute_shields,split_boundary_critical_shields, &
   split_boundary_transport_stage,split_boundary_stress_ratio)
  if(allocated(channel_raw_bed_cell)) deallocate(channel_raw_bed_cell, &
   channel_raw_suspended_cell,channel_composed_bed_cell,channel_retained_suspended_cell, &
   channel_signed_suspended_rate_cell,channel_equilibrium_concentration_cell, &
   channel_raw_bed_boundary,channel_raw_suspended_boundary,channel_composed_bed_boundary, &
   channel_retained_suspended_boundary,channel_signed_suspended_rate_boundary, &
   channel_equilibrium_concentration_boundary)
   if(allocated(gaia_bed_down)) deallocate(gaia_bed_down,gaia_eta_gradient, &
    bedq_gaia_base,bedq_gaia_base_grad)
  if(allocated(secondary_curvature)) deallocate(secondary_curvature)
  if(allocated(secondary_classic_intensity)) deallocate(secondary_classic_intensity)
  if(allocated(secondary_face_left)) deallocate(secondary_face_left,secondary_face_right, &
   secondary_boundary_intensity,secondary_low_flux,secondary_anti_flux,secondary_stress_face,secondary_rplus, &
   secondary_rminus,secondary_low_state,secondary_source_work)
  call release_physical_toe_work()
  if(allocated(distributed_face_mass_trial)) deallocate(distributed_face_mass_trial,distributed_cell_bedload, &
   distributed_rank_deficient)
  if(allocated(bank_transfer)) deallocate(bank_transfer,bank_scale,bank_bed,bank_request,bank_weight,bank_down,bank_previous,bank_depth, &
   bank_depth_intercept,bank_rate,bank_conductance)
  if(allocated(bank_undercut)) deallocate(bank_undercut,bank_reference_height,bank_toe_height,bank_reference_length, &
   bank_toe_cumulative,bank_collapse_cumulative,bank_collapse_count)
  if(allocated(bank_front_status)) deallocate(bank_front_status,bank_front_seed_face,bank_front_solid_area, &
   bank_front_base,bank_front_height,bank_front_removed_mass)
  allocate(area(nc),xy(2,nc),rock(nc),q(nstate,nc),ii(nc),jj(nc),grid(nx,ny),manning_cell(nc), &
   hydro_constant(nc),active_cell(nc),bank_erodible(nc),owner(nf),neighbour(nf),bid(nf), &
   normal(2,nf),length(nf),mid(2,nf),boundary_bed_offset(nf),boundary_bed_left_offset(nf), &
   boundary_bed_right_offset(nf),sampled_boundary_bed(nf),sampled_boundary_open(nf),section_cached_bed(nf), &
   offset(nc+1),at_face_source(nf))
  number=0;grid=0;offset=0;boundary_offset=0;hydro_constant=.false.;active_cell=.true.
  sampled_boundary_open=.true.;section_cache_valid=.false.;section_cached_bed=0
  boundary_seed_position=0;section_seed_node=0
  do c=1,nc
   n=at_leaf(c);number(n)=c;area(c)=at_area(n);xy(:,c)=at_xy(:,n);rock(c)=at_rock(n)
   q(:,c)=at_q(:,n);ii(c)=at_i(n);jj(c)=at_j(n);manning_cell(c)=at_manning(n);bank_erodible(c)=at_bank(n)
   if(grid(ii(c),jj(c))/=0) then
    grid(ii(c),jj(c))=-1
   else
    grid(ii(c),jj(c))=c
   endif
  enddo
  k=0
  do f=1,amr_fn
   if(.not.amr_use_face(f)) cycle
   k=k+1;at_face_source(k)=f
   owner(k)=number(at_cover(af_owner(f)));neighbour(k)=0
   if(af_neighbour(f)>0) neighbour(k)=number(at_cover(af_neighbour(f)))
   bid(k)=af_bid(f);normal(:,k)=af_normal(:,f);length(k)=af_length(f);mid(:,k)=af_mid(:,f)
   boundary_bed_offset(k)=af_bed(f);boundary_bed_left_offset(k)=af_left(f)
   boundary_bed_right_offset(k)=af_right(f);sampled_boundary_bed(k)=af_sampled(f)
   offset(owner(k))=offset(owner(k))+1
   if(neighbour(k)>0) offset(neighbour(k))=offset(neighbour(k))+1
   if(bid(k)>0) boundary_offset(bid(k))=boundary_offset(bid(k))+1
  enddo
  if(flow2dh_amr_ownership_declaration) then
   do f=1,nf
    if(bid(f)<=0.or.neighbour(f)>0) cycle
    new_open_memory_face(at_face_source(f))=.true.
   enddo
   if(any(new_open_memory_face.neqv.old_open_memory_face)) &
    error stop 'v49 AMR regrid changed an authenticated open-boundary segment'
   deallocate(old_open_memory_face,new_open_memory_face)
  endif
  k=1
  do c=1,nc
   p=offset(c);offset(c)=k;k=k+p
  enddo
  offset(nc+1)=k;allocate(faces(k-1),signs(k-1));cursor=offset(:nc)
  k=1
  do c=1,nb
   p=boundary_offset(c);boundary_offset(c)=k;k=k+p
  enddo
  boundary_offset(nb+1)=k;allocate(boundary_faces(k-1));bcursor=boundary_offset(:nb)
  do f=1,nf
   c=owner(f);k=cursor(c);faces(k)=f;signs(k)=1;cursor(c)=k+1
   c=neighbour(f)
   if(c>0) then
    k=cursor(c);faces(k)=f;signs(k)=-1;cursor(c)=k+1
   endif
   c=bid(f)
   if(c>0) then
    k=bcursor(c);boundary_faces(k)=f;bcursor(c)=k+1
   endif
  enddo
  tol=1.d-9*max(dx,dy)*8
  do c=1,nc
   closure=0
   do k=offset(c),offset(c+1)-1
    f=faces(k);closure=closure+signs(k)*length(f)*normal(:,f)
   enddo
   if(any(abs(closure)>tol)) error stop 'adaptive leaf face closure failed'
  enddo
  call amr_sort_boundary_sections()
  call cache_open_boundary_distance()
  call classify_rectangles()
  call cache_face_geometry()
  if(scheme==6) call cache_wall_geometry()
  if(bedlaw>0) then
   allocate(bedq_cell(2,nfrac,nc),bedq_equilibrium(2,nfrac,nc),bedq_relax_work(2,nfrac,nc), &
    bedq_grad(2,2,nfrac,nc),bedtau_cell(nfrac,nc))
   bedq_cell=0;bedq_equilibrium=0;bedq_relax_work=0;bedq_grad=0;bedtau_cell=0
   if(capacity_split_retained) then
    allocate(suspended_capacity_cell(2,nfrac,nc),suspended_capacity_equilibrium(2,nfrac,nc), &
     capacity_absolute_shields(nfrac,nc),capacity_critical_shields(nfrac,nc), &
     capacity_transport_stage(nfrac,nc),split_boundary_role(nf), &
     split_boundary_bed(2,nfrac,nf),split_boundary_suspended(2,nfrac,nf), &
     split_boundary_absolute_shields(nfrac,nf),split_boundary_critical_shields(nfrac,nf), &
     split_boundary_transport_stage(nfrac,nf),split_boundary_stress_ratio(nfrac,nf))
    suspended_capacity_cell=0.d0;suspended_capacity_equilibrium=0.d0
    capacity_absolute_shields=0.d0;capacity_critical_shields=0.d0;capacity_transport_stage=0.d0
    split_boundary_role=0;split_boundary_bed=0.d0;split_boundary_suspended=0.d0
    split_boundary_absolute_shields=0.d0;split_boundary_critical_shields=0.d0
    split_boundary_transport_stage=0.d0;split_boundary_stress_ratio=0.d0
   endif
   if(channel_composition_declaration) then
    allocate(channel_raw_bed_cell(2,nfrac,nc),channel_raw_suspended_cell(2,nfrac,nc), &
     channel_composed_bed_cell(2,nfrac,nc),channel_retained_suspended_cell(2,nfrac,nc), &
     channel_signed_suspended_rate_cell(nfrac,nc), &
     channel_equilibrium_concentration_cell(nfrac,nc), &
     channel_raw_bed_boundary(2,nfrac,nf),channel_raw_suspended_boundary(2,nfrac,nf), &
     channel_composed_bed_boundary(2,nfrac,nf), &
     channel_retained_suspended_boundary(2,nfrac,nf), &
     channel_signed_suspended_rate_boundary(nfrac,nf), &
     channel_equilibrium_concentration_boundary(nfrac,nf))
    channel_raw_bed_cell=0.d0;channel_raw_suspended_cell=0.d0
    channel_composed_bed_cell=0.d0;channel_retained_suspended_cell=0.d0
    channel_signed_suspended_rate_cell=0.d0;channel_equilibrium_concentration_cell=0.d0
    channel_raw_bed_boundary=0.d0;channel_raw_suspended_boundary=0.d0
    channel_composed_bed_boundary=0.d0;channel_retained_suspended_boundary=0.d0
    channel_signed_suspended_rate_boundary=0.d0
    channel_equilibrium_concentration_boundary=0.d0
   endif
   if(bedlaw==9) then
     allocate(gaia_bed_down(2,nc),gaia_eta_gradient(2,nc), &
      bedq_gaia_base(2,nfrac,nc),bedq_gaia_base_grad(2,2,nfrac,nc))
     gaia_bed_down=0.d0;gaia_eta_gradient=0.d0
     bedq_gaia_base=0.d0;bedq_gaia_base_grad=0.d0
   endif
  endif
  if(exchange_mode==4) then
   allocate(suspended_exchange_cell_trial(nfrac,nc))
   suspended_exchange_cell_trial=0.d0
  endif
  call allocate_secondary_work()
  if(deck_version>=17.or.general_avalanche.or.bank_failure) allocate(bank_transfer(nfrac,nf),bank_scale(nfrac,nc),bank_bed(nc),bank_request(nc),bank_weight(nc),bank_down(2,nc), &
   bank_previous(nfrac,nc),bank_depth(nc),bank_depth_intercept(nc),bank_rate(nf),bank_conductance(nf))
  if(bank_failure) then
   allocate(bank_undercut(nc),bank_reference_height(nc),bank_toe_height(nc),bank_reference_length(nc), &
            bank_toe_cumulative(nc),bank_collapse_cumulative(nc),bank_collapse_count(nc))
   do c=1,nc
    n=at_leaf(c);bank_undercut(c)=at_bank_undercut(n)
    bank_reference_height(c)=at_bank_reference_height(n);bank_toe_height(c)=at_bank_toe_height(n)
    bank_reference_length(c)=at_bank_reference_length(n);bank_toe_cumulative(c)=at_bank_toe_cumulative(n)
    bank_collapse_cumulative(c)=at_bank_collapse_cumulative(n);bank_collapse_count(c)=at_bank_collapse_count(n)
   enddo
  endif
  if(bank_front_tracking) then
   allocate(bank_front_status(nc),bank_front_seed_face(nc),bank_front_solid_area(nc), &
            bank_front_base(nc),bank_front_height(nc),bank_front_removed_mass(nfrac,nc))
   do c=1,nc
    n=at_leaf(c);bank_front_status(c)=at_bank_front_status(n)
    bank_front_seed_face(c)=at_bank_front_seed_face(n);bank_front_solid_area(c)=at_bank_front_solid_area(n)
    bank_front_base(c)=at_bank_front_base(n);bank_front_height(c)=at_bank_front_height(n)
    bank_front_removed_mass(:,c)=at_bank_front_removed_mass(:,n)
   enddo
  endif
  if(bank_failure.and.bank_toe_erosion_mode==3) then
   allocate(distributed_face_mass_trial(nfrac,2,nf),distributed_cell_bedload(2,nfrac,nc), &
            distributed_rank_deficient(nc))
   distributed_face_mass_trial=0.d0;distributed_cell_bedload=0.d0
   distributed_rank_deficient=.true.
  endif
  call amr_cache_level_work()
  call amr_allocate_step_workspaces()
 end subroutine

 subroutine amr_sort_boundary_sections()
  integer :: b,first,last,k,j,f,local_seed
  real(8) :: tangent(2),position,bed(nc)
  real(8),allocatable :: nodes(:)
  do k=1,nc
   bed(k)=amr_node_bed(at_leaf(k))
  enddo
  do b=1,nb
   first=boundary_offset(b);last=boundary_offset(b+1)-1
   if(last<first) error stop 'AMR lost a logical boundary ID'
   if(sampled_boundary_bed(boundary_faces(first))==0) cycle
   tangent=[-normal(2,boundary_faces(first)),normal(1,boundary_faces(first))]
   if(tangent(1)<0.or.(tangent(1)==0.and.tangent(2)<0)) tangent=-tangent
   do k=first+1,last
    f=boundary_faces(k);position=dot_product(mid(:,f),tangent);j=k-1
    do while(j>=first)
     if(dot_product(mid(:,boundary_faces(j)),tangent)<=position) exit
     boundary_faces(j+1)=boundary_faces(j);j=j-1
    enddo
    boundary_faces(j+1)=f
   enddo
   boundary_seed_position(b)=first
   do k=first+1,last
    f=boundary_faces(k);j=boundary_faces(boundary_seed_position(b))
    if(bed(owner(f))+boundary_bed_offset(f)<bed(owner(j))+boundary_bed_offset(j)) boundary_seed_position(b)=k
   enddo
   if(production_section_geometry) then
    allocate(nodes(last-first+2));call section_nodes(b,bed,nodes)
    local_seed=minloc(nodes,dim=1)
    if(local_seed==1.or.local_seed==size(nodes)) error stop 'AMR section minimum lies at an open endpoint'
    section_seed_node(b)=local_seed;deallocate(nodes)
   endif
  enddo
 end subroutine

 subroutine cache_open_boundary_distance()
  ! Mark the boundary owner and two interior layers: the radius of a
  ! five-cell WENO stencil.  Use the bounded capacity closure throughout
  ! this region before returning to the interior WENO/cut-cell operator.
  integer :: f,c,n,layer,b
  allocate(open_boundary_distance(nc));open_boundary_distance=nc+1
  allocate(uncoupled_boundary_distance(nc));uncoupled_boundary_distance=nc+1
  do f=1,nf
   b=bid(f)
   if(b>0.and.neighbour(f)==0) then
    open_boundary_distance(owner(f))=0
    ! Retain the legacy closure at discharge/critical/weir boundaries. At a
    ! fixed-stage boundary, however, midpoint capacity without the coupled
    ! SWE--Exner waves is unstable at accelerated morphological time scales.
    ! Keep its bounded capacity stencil, but allow the coupled interior flux.
    if(mode(b)/=2) uncoupled_boundary_distance(owner(f))=0
   endif
  enddo
  do layer=1,2
   do f=1,nf
    c=owner(f);n=neighbour(f)
    if(n==0) cycle
    if(open_boundary_distance(c)==layer-1) open_boundary_distance(n)=min(open_boundary_distance(n),layer)
    if(open_boundary_distance(n)==layer-1) open_boundary_distance(c)=min(open_boundary_distance(c),layer)
    if(uncoupled_boundary_distance(c)==layer-1) uncoupled_boundary_distance(n)=min(uncoupled_boundary_distance(n),layer)
    if(uncoupled_boundary_distance(n)==layer-1) uncoupled_boundary_distance(c)=min(uncoupled_boundary_distance(c),layer)
   enddo
  enddo
 end subroutine

 subroutine classify_rectangles()
  ! Verify actual face geometry, not just equal retained areas. A clipped
  ! rectangular strip has uniform FV averages along its uncut direction.
 real(8) :: lo(2),hi(2),width(2),totals(2,2),out(2),tol
 integer :: c,k,f,a,t,side
  logical :: valid
  allocate(rectangle_width(2,nc));rectangle_width=0
  tol=1.d-10*max(dx,dy)
  do c=1,nc
   lo=huge(1.d0);hi=-huge(1.d0);totals=0;valid=.true.
   do k=offset(c),offset(c+1)-1
    f=faces(k);out=signs(k)*normal(:,f);a=maxloc(abs(out),dim=1);t=3-a
    if(abs(out(t))>1.d-12.or.abs(abs(out(a))-1)>1.d-12) then
     valid=.false.;exit
    endif
    side=1
    if(out(a)>0) side=2
    if(side==1) lo(a)=min(lo(a),mid(a,f))
    if(side==2) hi(a)=max(hi(a),mid(a,f))
    totals(a,side)=totals(a,side)+length(f)
   enddo
   if(.not.valid.or.any(totals<=0)) cycle
   width=hi-lo
   if(any(width<=0).or.any(abs(xy(:,c)-.5d0*(lo+hi))>tol)) cycle
   if(abs(area(c)-product(width))>1.d-10*area(c)) cycle
   if(any(abs(totals(1,:)-width(2))>tol).or.any(abs(totals(2,:)-width(1))>tol)) cycle
   do k=offset(c),offset(c+1)-1
    f=faces(k);out=signs(k)*normal(:,f);a=maxloc(abs(out),dim=1);t=3-a
    if(out(a)<0.and.abs(mid(a,f)-lo(a))>tol) valid=.false.
    if(out(a)>0.and.abs(mid(a,f)-hi(a))>tol) valid=.false.
    if(mid(t,f)-.5d0*length(f)<lo(t)-tol.or.mid(t,f)+.5d0*length(f)>hi(t)+tol) valid=.false.
   enddo
   if(valid) rectangle_width(:,c)=width
  enddo
 end subroutine

 subroutine primitive(state,v,bed,include_suspended)
  real(8), intent(in) :: state(nstate,nc)
  real(8), intent(out) :: v(nprimitive,nc),bed(nc)
  logical, optional, intent(in) :: include_suspended
  integer :: c,f,l
  logical :: with_suspended
  with_suspended=.true.
  if(present(include_suspended)) with_suspended=include_suspended
  call effective_bed_elevation(state,bed)
  v=0;v(1,:)=state(1,:)+bed;v(bed_primitive_index(),:)=bed
  !$omp parallel do schedule(static,256) if(nc*nfrac>=8192) default(shared) private(c,f)
  do c=1,nc
   ! Keep the signed auxiliary concentration defined through a sub-tolerance
   ! wetting film.  The exact split source clears it at accepted dry states,
   ! but an SSPRK intermediate must carry M_I/h consistently with the same
   ! nonzero water inventory or its low-order FCT state cannot satisfy a local
   ! maximum principle at the moving shoreline.
   if(prognostic_secondary_flow().and.state(1,c)>0.d0) &
    v(secondary_primitive_index(),c)=state(secondary_state_index(),c)/state(1,c)
   if(state(1,c)<=dry_tolerance) cycle
   v(2:3,c)=state(2:3,c)/state(1,c)
   if(with_suspended) then
    do f=1,nfrac
     v(concentration_index(f),c)=state(suspended_index(f),c)/state(1,c)
    enddo
   endif
   if(prognostic_secondary_flow()) &
    v(secondary_primitive_index(),c)=state(secondary_state_index(),c)/state(1,c)
  enddo
  !$omp end parallel do
 end subroutine

 function bed_availability(state,fraction,cell) result(value)
  ! FLOW applies the upwind active-layer volume fraction and a common
  ! thin-layer fix factor to transport. Version 12 retains unit availability.
  real(8), intent(in) :: state(nstate,nc)
  integer, intent(in) :: fraction,cell
  real(8) :: value,thickness,target
  integer :: g
  value=1.d0
  if(deck_version<13) return
  thickness=0
  do g=1,nfrac
   thickness=thickness+state(bed_index(g),cell)/bulk_fraction(g)
  enddo
  if(thickness<=0) then
   value=0;return
  endif
  value=(state(bed_index(fraction),cell)/bulk_fraction(fraction))/thickness
  target=active_layer_target(state,cell)
  value=value*min(1.d0,thickness/target)
 end function

 pure function active_layer_target(state,cell) result(value)
  ! TTLForm=2 counterpart: optionally scale the well-mixed transport layer
  ! with instantaneous local water depth, with a strictly positive floor.
  real(8), intent(in) :: state(nstate,nc)
  integer, intent(in) :: cell
  real(8) :: value
  value=active_layer_thickness
  if(active_layer_mode==1) value=max(active_layer_depth_fraction*max(0.d0,state(1,cell)), &
                                     active_layer_minimum_thickness)
 end function

 subroutine mixture_statistics(state,dg,dgsd,sand_fraction,d90_mix,mud_fraction)
  ! Surface-composition statistics used by the Wilcock-Crowe family. With an
  ! active layer these are evaluated on that layer; otherwise the complete
  ! erodible bed is the surface mixture. Use the same volume basis as native
  ! bed_availability and Delft3D's active-layer getvfrac/IPorosity=0 route.
  real(8), intent(in) :: state(nstate,nc)
  real(8), intent(out) :: dg(nc),dgsd(nc),sand_fraction(nc),d90_mix(nc),mud_fraction(nc)
  real(8) :: fraction_mass(nfrac),fraction_volume(nfrac)
  real(8) :: total_mass,total_volume
  integer :: c,g,work,nwork,invalid_corrected_mixture
  logical :: valid
  dg=0;dgsd=1;sand_fraction=0;d90_mix=0;mud_fraction=0;nwork=nc
  invalid_corrected_mixture=0
  if(amr_step_level>=0) nwork=amr_level_eval_cell_count(amr_step_level)
  !$omp parallel do schedule(static,256) if(nwork*nfrac>=8192) default(shared) &
  !$omp& private(work,c,g,fraction_mass,fraction_volume,total_mass,total_volume,valid) &
  !$omp& reduction(max:invalid_corrected_mixture)
  do work=1,nwork
   c=work
   if(amr_step_level>=0) c=amr_level_eval_cells(work,amr_step_level)
   if(.not.active_cell(c)) cycle
   ! Bed-capacity closures return before consuming mixture statistics in a dry
   ! cell, so do not solve a percentile or sort a composition there.
   if(state(1,c)<=dry_tolerance) cycle
   do g=1,nfrac
    fraction_mass(g)=state(bed_index(g),c)
    fraction_volume(g)=fraction_mass(g)/bulk_fraction(g)
    if(sediment_type_fraction(g)==1) fraction_volume(g)=0.d0
   enddo
   ! FLOW getfrac returns mud content in the configured IFractions basis.
   ! Percentile statistics below retain their independent noncohesive solid-
   ! volume basis.
   total_mass=sum(fraction_mass)
   if(capacity_fraction_basis==1) then
    if(total_mass>0.d0) mud_fraction(c)= &
     sum(fraction_mass,mask=sediment_type_fraction==1)/total_mass
   else
    total_volume=sum(fraction_mass/bulk_fraction)
    if(total_volume>0.d0) mud_fraction(c)= &
     sum(fraction_mass/bulk_fraction,mask=sediment_type_fraction==1)/total_volume
   endif
   ! Each closure consumes only one of these two statistic families. Avoid
   ! sorting the surface mixture for MPM/Van Rijn, and avoid the percentile
   ! solve for closures that use only Dg, geometric spread and sand content.
   ! The guarded calls leave every value actually consumed by a closure
   ! bit-for-bit unchanged.
   if(bedlaw>2) then
    call braids_surface_statistics(nfrac,fraction_mass,bulk_fraction,d50_fraction,sediment_type_fraction, &
                                 dg(c),dgsd(c),sand_fraction(c))
    if(any(bedlaw==[12,14])) then
     ! FLOW supplies RP_D90MX, the D90 of the exposed mixture, to both
     ! TraFrm 7 and 13.  The original optimized selector populated D90 only
     ! for MPM roughness, which made the newly live current formulas consume
     ! zero and fail despite a valid bed composition.
     call braids_mixture_d90_default(fraction_volume,d50_fraction,d90_mix(c),valid)
     if(.not.valid) d90_mix(c)=maxval(d90_fraction)
    endif
   else if(bedlaw==2.and.mpm_friction_mode==1) then
    call braids_mixture_d90_default(fraction_volume,d50_fraction,d90_mix(c),valid)
    if(.not.valid) then
     if(capacity_route_identity_declaration.and.trim(capacity_route_method_id)=='capacity.mpm') then
      invalid_corrected_mixture=1
     else
      ! Preserve the historical mode-1 fallback for every pre-v50 deck.  An
      ! authenticated corrected route may not silently replace FLOW's
      ! cell-local mixture D90 with a configured class D90.
      d90_mix(c)=maxval(d90_fraction)
     endif
    endif
   endif
  enddo
  !$omp end parallel do
  if(invalid_corrected_mixture/=0) &
   error stop 'corrected Delft MPM requires a nonempty active noncohesive mixture'
 end subroutine

 subroutine surface_composition(state,cell,fraction_share,mud_fraction)
  real(8), intent(in) :: state(nstate,nc)
  integer, intent(in) :: cell
  real(8), intent(out) :: fraction_share(nfrac),mud_fraction
  real(8) :: total
  integer :: g
  do g=1,nfrac
   fraction_share(g)=state(bed_index(g),cell)
  enddo
  total=sum(fraction_share)
  if(total>0) then
   fraction_share=fraction_share/total
  else
   fraction_share=0
  endif
  mud_fraction=sum(fraction_share,mask=sediment_type_fraction==1)
 end subroutine

 pure subroutine hydraulic_roughness_at(h,parameter,chezy,z0,drag,valid)
  real(8),intent(in) :: h,parameter
  real(8),intent(out) :: chezy,z0,drag
  logical,intent(out) :: valid
  real(8) :: work_depth
  work_depth=max(manning_friction_chezy_minimum_depth,roughness_minimum_work_depth)
  call braids_roughness_state(hydraulic_roughness_mode,h,parameter,grav,work_depth, &
   white_colebrook_chzk2d,chezy,z0,drag,valid)
 end subroutine

 pure function hydraulic_equivalent_manning(h,parameter) result(value)
  ! Adapter for established sediment/secondary operators whose ABI accepts n
  ! and immediately reconstructs C=h^(1/6)/n.  It changes no constitutive
  ! physics: C/M/W first resolves the authoritative Chezy value here.
  real(8),intent(in) :: h,parameter
  real(8) :: value,chezy,z0,drag
  logical :: valid
  value=0.d0
  if(h<=0.d0.or.parameter<=0.d0) return
  if(hydraulic_roughness_mode==braids_roughness_manning.and. &
     roughness_minimum_work_depth==0.d0) then
   value=parameter
   return
  endif
  call hydraulic_roughness_at(h,parameter,chezy,z0,drag,valid)
  if(valid) value=h**(1.d0/6.d0)/chezy
 end function

 pure function hydraulic_bed_shear(h,hu,hv,parameter) result(tau)
  real(8), intent(in) :: h,hu,hv,parameter
  real(8) :: tau,speed,chezy,z0
  logical :: valid
  tau=0.d0
  if(h<=dry_tolerance.or.parameter<=0.d0) return
  speed=sqrt(hu*hu+hv*hv)/h
  call braids_roughness_shear(hydraulic_roughness_mode,h,speed,parameter,rho,grav, &
   max(manning_friction_chezy_minimum_depth,roughness_minimum_work_depth), &
   white_colebrook_chzk2d,tau,chezy,z0,valid)
  if(.not.valid) tau=0.d0
 end function

 pure function sediment_bed_shear(h,hu,hv,parameter,mud_fraction) result(tau)
  real(8),intent(in) :: h,hu,hv,parameter,mud_fraction
  real(8) :: tau,speed,z0skin,cdr,cds
  logical :: valid
  tau=hydraulic_bed_shear(h,hu,hv,parameter)
  if(sediment_shear_mode/=1.or.h<=dry_tolerance) return
  speed=sqrt(hu*hu+hv*hv)/h
  call braids_soulsby_clarke_skin_shear(h,speed,skin_kinematic_viscosity,rho, &
   skin_sand_roughness,skin_silt_roughness,mud_fraction, &
   skin_critical_mud_fraction_low,skin_critical_mud_fraction_high, &
   skin_fluff_fraction,tau,z0skin,cdr,cds,valid)
  if(.not.valid) tau=0.d0
 end function

 pure function roughness_conveyance(section_area,section_perimeter,parameter) result(value)
  real(8),intent(in) :: section_area,section_perimeter,parameter
  real(8) :: value,radius,chezy,z0,drag
  logical :: valid
  value=0.d0
  if(min(section_area,section_perimeter,parameter)<=0.d0) return
  radius=section_area/section_perimeter
  call hydraulic_roughness_at(radius,parameter,chezy,z0,drag,valid)
  if(valid) value=section_area*chezy*sqrt(radius)
 end function

 pure function roughness_face_weight(section_depth,face_depth,parameter) result(value)
  real(8),intent(in) :: section_depth,face_depth,parameter
  real(8) :: value,chezy,z0,drag
  logical :: valid
  value=0.d0
  if(min(section_depth,face_depth,parameter)<=0.d0) return
  call hydraulic_roughness_at(face_depth,parameter,chezy,z0,drag,valid)
  if(valid) value=section_depth**1.5d0*chezy
 end function

 subroutine hiding_factors(state,factors)
  real(8), intent(in) :: state(nstate,nc)
  real(8), intent(out) :: factors(nfrac,nc)
  real(8) :: active_volume(nfrac),mean_diameter
  integer :: c,g,work,nwork,invalid
  logical :: valid
  factors=1
  if(hiding_exposure==0) return
  invalid=0;nwork=nc
  if(amr_step_level>=0) nwork=amr_level_eval_cell_count(amr_step_level)
  !$omp parallel do schedule(static,256) if(nwork*nfrac>=8192) default(shared) &
  !$omp& private(work,c,g,active_volume,mean_diameter,valid) reduction(max:invalid)
  do work=1,nwork
   c=work
   if(amr_step_level>=0) c=amr_level_eval_cells(work,amr_step_level)
   if(.not.active_cell(c)) cycle
   do g=1,nfrac
    active_volume(g)=state(bed_index(g),c)/bulk_fraction(g)
   enddo
   if(sum(active_volume)<=0) then
    ! Dry/interface stencil cells retain neutral factors.  A genuinely wet
    ! transported cell with no active-surface stock is not a physical hiding
    ! state and must never be silently relabelled as hiding=1.
    if(state(1,c)>dry_tolerance) invalid=1
    cycle
   endif
   call braids_hiding_exposure_factors(hiding_exposure,active_volume,d50_fraction,factors(:,c), &
        mean_diameter,hiding_parker_alpha,hiding_wu_exponent,valid)
   if(.not.valid) invalid=1
  enddo
  !$omp end parallel do
  if(invalid/=0) error stop 'invalid active-surface hiding/exposure state'
 end subroutine

 subroutine repack_substrate_cell(state,c,deposited)
  ! Rebin the ordered substrate stack from the surface downward. Newly buried
  ! material is the first segment. Each preserved layer except the base has a
  ! fixed thickness; the base stores the remaining inventory. This is the same
  ! conservative ordering used by Delft3D's Lagrangian underlayer refill.
  real(8), intent(inout) :: state(nstate,nc)
  integer, intent(in) :: c
  real(8), intent(in) :: deposited(nfrac)
  real(8) :: segments(nfrac,nsubstrate_layers+1),packed(nfrac,nsubstrate_layers)
  real(8) :: thickness,capacity,take,factor
  integer :: source,layer,g
  segments=0;packed=0;segments(:,1)=deposited
  do layer=1,nsubstrate_layers
   do g=1,nfrac
    segments(g,layer+1)=state(bed_layer_index(layer+1,g),c)
   enddo
  enddo
  source=1
  do layer=1,nsubstrate_layers
   capacity=huge(1.d0)
   if(layer<nsubstrate_layers) capacity=substrate_layer_thickness
   do while(source<=nsubstrate_layers+1.and.capacity>0)
    thickness=sum(segments(:,source)/bulk_fraction)
    if(thickness<=64*epsilon(1.d0)) then
     source=source+1;cycle
    endif
    take=thickness
    if(layer<nsubstrate_layers) take=min(take,capacity)
    factor=min(1.d0,take/thickness)
    packed(:,layer)=packed(:,layer)+segments(:,source)*factor
    segments(:,source)=segments(:,source)*(1-factor)
    if(layer<nsubstrate_layers) capacity=max(0.d0,capacity-take)
    if(factor>=1-64*epsilon(1.d0)) source=source+1
    if(layer==nsubstrate_layers) cycle
   enddo
  enddo
  do layer=1,nsubstrate_layers
   do g=1,nfrac
    state(bed_layer_index(layer+1,g),c)=max(0.d0,packed(g,layer))
   enddo
  enddo
 end subroutine

 subroutine rebalance_active_layer(state)
  ! Keep a fixed-thickness active surface and an ordered, finite-resolution
  ! substrate. Burial inserts the active surplus at the top of the stack;
  ! exhumation removes material from the top down and compacts the remaining
  ! sequence. Every class transfer is conservative to roundoff.
  real(8), intent(inout) :: state(nstate,nc)
  real(8) :: active_thickness,substrate_thickness,total_thickness,target,move,factor,amount
  real(8) :: transferred(nfrac),empty(nfrac),layer_thickness,take
  integer :: c,g,l,ai,li,work,nwork
  if(deck_version<13) return
  empty=0
  nwork=nc
  if(amr_step_level>=0) nwork=amr_level_cell_count(amr_step_level)
  !$omp parallel do schedule(static,256) if(nwork*nfrac>=8192) default(shared) &
  !$omp& private(work,c,g,l,ai,li,active_thickness,substrate_thickness,total_thickness,target,move,factor,amount, &
  !$omp& transferred,layer_thickness,take)
  do work=1,nwork
   c=work
   if(amr_step_level>=0) c=amr_level_cells(work,amr_step_level)
   active_thickness=0;substrate_thickness=0
   do g=1,nfrac
    active_thickness=active_thickness+state(bed_index(g),c)/bulk_fraction(g)
    do l=2,nbed_layers
     substrate_thickness=substrate_thickness+state(bed_layer_index(l,g),c)/bulk_fraction(g)
    enddo
   enddo
   total_thickness=active_thickness+substrate_thickness
   target=min(active_layer_target(state,c),total_thickness)
   transferred=0
   if(active_thickness>target.and.active_thickness>0) then
    move=active_thickness-target;factor=min(1.d0,move/active_thickness)
    do g=1,nfrac
     ai=bed_index(g);amount=state(ai,c)*factor
     state(ai,c)=state(ai,c)-amount;transferred(g)=amount
     if(nsubstrate_layers==1) state(substrate_index(g),c)=state(substrate_index(g),c)+amount
    enddo
    if(nsubstrate_layers>1) call repack_substrate_cell(state,c,transferred)
   else if(active_thickness<target.and.substrate_thickness>0) then
    move=target-active_thickness
    if(nsubstrate_layers==1) then
     factor=min(1.d0,move/substrate_thickness)
     do g=1,nfrac
      li=substrate_index(g);amount=state(li,c)*factor
      state(li,c)=state(li,c)-amount;state(bed_index(g),c)=state(bed_index(g),c)+amount
     enddo
    else
     do l=2,nbed_layers
      if(move<=64*epsilon(1.d0)) exit
      layer_thickness=0
      do g=1,nfrac
       layer_thickness=layer_thickness+state(bed_layer_index(l,g),c)/bulk_fraction(g)
      enddo
      if(layer_thickness<=0) cycle
      take=min(move,layer_thickness);factor=min(1.d0,take/layer_thickness)
      do g=1,nfrac
       li=bed_layer_index(l,g);amount=state(li,c)*factor
       state(li,c)=state(li,c)-amount;state(bed_index(g),c)=state(bed_index(g),c)+amount
      enddo
      move=max(0.d0,move-take)
     enddo
     call repack_substrate_cell(state,c,empty)
    endif
   endif
  enddo
  !$omp end parallel do
 end subroutine

 subroutine raw_bed_elevation(state,bed)
  real(8), intent(in) :: state(nstate,nc)
  real(8), intent(out) :: bed(nc)
  integer :: g,l
  bed=rock
  do g=1,nfrac
   bed=bed+state(bed_index(g),:)/bulk_fraction(g)
   do l=2,nbed_layers
    bed=bed+state(bed_layer_index(l,g),:)/bulk_fraction(g)
   enddo
  enddo
 end subroutine

 subroutine effective_bed_elevation(state,bed)
  real(8),intent(in) :: state(nstate,nc)
  real(8),intent(out) :: bed(nc)
  integer :: c
  call raw_bed_elevation(state,bed)
  if(.not.bank_front_tracking) return
  do c=1,nc
   if(bank_front_status(c)/=BANK_FRONT_ACTIVE) cycle
   block
    integer :: front_error
    real(8) :: effective_bed
    call braids_bank_front_effective_bed(bed(c),area(c),bank_front_status(c), &
     bank_front_solid_area(c),bank_front_base(c),effective_bed,front_error)
    if(front_error/=BANK_FRONT_SUCCESS) error stop 'invalid persistent bank-front bed closure'
    bed(c)=effective_bed
   end block
  enddo
 end subroutine

 subroutine bank_bed_elevation(state,bed)
  real(8),intent(in) :: state(nstate,nc)
  real(8),intent(out) :: bed(nc)
  call effective_bed_elevation(state,bed)
 end subroutine

 subroutine apply_surface_transfer(state)
  ! bank_transfer(g,f) is a total mass moved during this stage. Positive is
  ! owner to neighbour. A cell-wise donor limiter prevents concurrent faces
  ! from removing more than the exposed active inventory. The final gather is
  ! deterministic and race-free for every OpenMP thread count.
  real(8), intent(inout) :: state(nstate,nc)
  call limit_surface_transfer(state)
  call apply_limited_surface_transfer(state)
 end subroutine

 subroutine limit_surface_transfer(state)
  ! Apply the shared donor limiter without changing inventory.  The
  ! persistent-front event detector consumes this already-limited transfer,
  ! so its event time is based on mass that can actually leave the donor.
  real(8), intent(in) :: state(nstate,nc)
  real(8) :: outgoing,available
  integer :: c,g,k,f,donor
  bank_scale=1
  !$omp parallel do collapse(2) schedule(static,256) if(nc>=4096) default(shared) private(c,g,k,f,outgoing,available)
  do c=1,nc
   do g=1,nfrac
    outgoing=0
    do k=offset(c),offset(c+1)-1
     f=faces(k)
     outgoing=outgoing+max(real(signs(k),8)*bank_transfer(g,f),0.d0)
    enddo
    available=state(bed_index(g),c)*area(c)
    if(outgoing>available.and.outgoing>0) bank_scale(g,c)=max(0.d0,available/outgoing)*(1-64*epsilon(1.d0))
   enddo
  enddo
  !$omp end parallel do
  !$omp parallel do collapse(2) schedule(static,256) if(nc>=4096) default(shared) private(f,g,donor)
  do f=1,nf
   do g=1,nfrac
    if(bank_transfer(g,f)>0) then
     donor=owner(f)
    else if(bank_transfer(g,f)<0) then
     donor=neighbour(f)
    else
     cycle
    endif
    bank_transfer(g,f)=bank_transfer(g,f)*bank_scale(g,donor)
   enddo
  enddo
  !$omp end parallel do
 end subroutine

 subroutine apply_limited_surface_transfer(state)
  ! Gather a transfer that has already passed the shared donor limiter.
  real(8), intent(inout) :: state(nstate,nc)
  real(8) :: change
  integer :: c,g,k,f,bi
  !$omp parallel do collapse(2) schedule(static,256) if(nc>=4096) default(shared) private(c,g,k,f,change,bi)
  do c=1,nc
   do g=1,nfrac
    change=0
    do k=offset(c),offset(c+1)-1
     f=faces(k);change=change-real(signs(k),8)*bank_transfer(g,f)
    enddo
    bi=bed_index(g);state(bi,c)=state(bi,c)+change/area(c)
    if(state(bi,c)<0.and.abs(state(bi,c))*area(c)<=128*epsilon(1.d0)) state(bi,c)=0
   enddo
  enddo
  !$omp end parallel do
 end subroutine

 subroutine redistribute_eroded_banks(reference,state)
  ! Replenish a configured fraction of bed erosion in a wet channel cell from
  ! adjacent higher, dry, explicitly erodible bank cells. This is the finite-
  ! volume counterpart of FLOW's ThetSD dry-bank redistribution. Only internal
  ! faces participate, so the calculation-domain perimeter remains fixed.
  real(8), intent(in) :: reference(nstate,nc)
  real(8), intent(inout) :: state(nstate,nc)
  integer :: g,l
  if(deck_version<17.or.bank_erosion_fraction<=0) return
  bank_depth=reference(1,:)
  do g=1,nfrac
   bank_previous(g,:)=reference(bed_index(g),:)
   do l=2,nbed_layers
    bank_previous(g,:)=bank_previous(g,:)+reference(bed_layer_index(l,g),:)
   enddo
  enddo
  call redistribute_bank_inventory(bank_depth,bank_previous,state)
 end subroutine

 subroutine redistribute_bank_inventory(reference_depth,previous_bed,state)
  real(8), intent(in) :: reference_depth(nc),previous_bed(nfrac,nc)
  real(8), intent(inout) :: state(nstate,nc)
  real(8) :: erosion,active_volume,volume,share,amount,sediment_depth,minimum_area
  integer :: c,g,k,f,n,wet,donor,l
  call bank_bed_elevation(state,bank_bed)
  sediment_depth=huge(1.d0)
  if(bedlaw>0) sediment_depth=min(sediment_depth,minval(bedload_min_depth_fraction))
  if(exchange_mode>0) sediment_depth=min(sediment_depth,minval(sedthr_fraction))
  if(bedlaw==0.and.exchange_mode==0) sediment_depth=dry_tolerance
  sediment_depth=max(dry_tolerance,sediment_depth)
  !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) private(c,g,l,erosion,amount)
  do c=1,nc
   bank_request(c)=0;bank_weight(c)=0
   if(reference_depth(c)<=sediment_depth) cycle
   erosion=0
   do g=1,nfrac
    amount=state(bed_index(g),c)
    do l=2,nbed_layers
     amount=amount+state(bed_layer_index(l,g),c)
    enddo
    erosion=erosion+(previous_bed(g,c)-amount)/bulk_fraction(g)
   enddo
   bank_request(c)=bank_erosion_fraction*max(0.d0,erosion)*area(c)
  enddo
  !$omp end parallel do
  !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) private(c,k,f,n,donor,minimum_area)
  do c=1,nc
   if(bank_request(c)<=0) cycle
   minimum_area=area(c)
   do k=offset(c),offset(c+1)-1
    f=faces(k);n=neighbour(f)
    if(n<=0) cycle
    if(signs(k)<0) n=owner(f)
    donor=n
    if(.not.bank_donor_link(c,donor,f,reference_depth,state,sediment_depth)) cycle
    if(bank_bed(donor)<=bank_bed(c)) cycle
    bank_weight(c)=bank_weight(c)+length(f)
    minimum_area=min(minimum_area,area(donor))
   enddo
   bank_request(c)=bank_request(c)*minimum_area/area(c)
  enddo
  !$omp end parallel do
  bank_transfer=0
  !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) private(f,c,n,wet,donor,g,active_volume,volume,share)
  do f=1,nf
   c=owner(f);n=neighbour(f)
   if(n<=0) cycle
   wet=0;donor=0
   if(reference_depth(c)>sediment_depth.and.reference_depth(n)<=sediment_depth) then
    wet=c;donor=n
   else if(reference_depth(n)>sediment_depth.and.reference_depth(c)<=sediment_depth) then
    wet=n;donor=c
   endif
   if(wet==0) cycle
   if(bank_weight(wet)<=0) cycle
   if(.not.bank_donor_link(wet,donor,f,reference_depth,state,sediment_depth)) cycle
   if(bank_bed(donor)<=bank_bed(wet)) cycle
   active_volume=0
   do g=1,nfrac
    active_volume=active_volume+state(bed_index(g),donor)/bulk_fraction(g)
   enddo
   if(active_volume<=0) cycle
   volume=bank_request(wet)*length(f)/bank_weight(wet)
   do g=1,nfrac
    share=(state(bed_index(g),donor)/bulk_fraction(g))/active_volume
    if(donor==c) then
     bank_transfer(g,f)=volume*share*bulk_fraction(g)
    else
     bank_transfer(g,f)=-volume*share*bulk_fraction(g)
    endif
   enddo
  enddo
  !$omp end parallel do
  call apply_surface_transfer(state)
 end subroutine

 logical function bank_donor_link(wet,donor,f,reference_depth,state,sediment_depth) result(eligible)
  integer, intent(in) :: wet,donor,f
  real(8), intent(in) :: reference_depth(nc),state(nstate,nc),sediment_depth
  integer :: g
  eligible=.false.
  if(reference_depth(donor)>sediment_depth.or..not.bank_erodible(donor)) return
  if(bank_bed(donor)<=bank_bed(wet)) return
  if(.not.any([(state(bed_index(g),donor)>0,g=1,nfrac)])) return
  ! Standard FLOW's kcu/kcv donor checks are grid-connectivity checks, not
  ! wet-face kfu/kfv masks. Dry-bank erosion must not require overtopping.
  eligible=.true.
 end function

subroutine avalanche_erodible_banks(state,dt)
  ! Conservative isotropic repose-slope diffusion.  The former single link
  ! relaxation used harmonic cell area directly, so its physical adjustment
  ! rate decreased with cell spacing and diagonal banks followed a different
  ! sequence of Cartesian link moves.  Here the full reconstructed slope sets
  ! one scalar excess, while each face receives the normal component of the
  ! same diffusion vector.  Internal diffusion substeps remove the parabolic
  ! mesh restriction from the hydraulic timestep. The diffusion coefficient
  ! uses the configured physical relaxation length. Its square divided by
  ! twice the relaxation time is the mesh-independent diffusion coefficient.
  ! The donor composition comes from the exposed active layer. Polygon boundary
  ! faces have no neighbour and are therefore excluded by construction.
  real(8), intent(inout) :: state(nstate,nc)
  real(8), intent(in) :: dt
  real(8) :: delta,distance,volume,active_volume,share,sx,sy,slope_magnitude
  real(8) :: alpha,conductance,diffusivity,remaining,subdt,rate_sum,maximum_rate,face_rate
  real(8) :: wet_slope,dry_slope,wet_depth_switch
  real(8) :: tangent(2),displacement(2),normal_slope,tangent_slope,depth_ends(2,2),r(2)
  integer :: f,c,n,donor,g,substeps
  if(.not.general_avalanche.and.(deck_version<17.or..not.bank_avalanche)) return
  if(general_avalanche) then
   diffusivity=.5d0*general_avalanche_relaxation_length**2/general_avalanche_time
   wet_slope=general_avalanche_wet_slope;dry_slope=general_avalanche_dry_slope
   wet_depth_switch=general_avalanche_wet_depth_switch
  else
   diffusivity=.5d0*bank_relaxation_length**2/bank_avalanche_time
   wet_slope=bank_wet_slope;dry_slope=bank_dry_slope;wet_depth_switch=bank_wet_depth_switch
  endif
  remaining=dt;substeps=0
  do while(remaining>64*epsilon(1.d0)*max(dt,1.d0))
   substeps=substeps+1
   if(substeps>100000) error stop 'bank relaxation exceeded internal substep limit'
   call bank_bed_elevation(state,bank_bed);call bed_downslopes(bank_bed,bank_down)
   bank_transfer=0;bank_rate=0;bank_conductance=0
   !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) private(c)
   do c=1,nc
    bank_depth_intercept(c)=bank_cell_depth_intercept(state(1,c),bank_down(:,c),xy(:,c),area(c), &
      faces(offset(c):offset(c+1)-1),signs(offset(c):offset(c+1)-1),mid,normal,length)
   enddo
   !$omp end parallel do
   !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) &
   !$omp& private(f,c,n,donor,delta,distance,sx,sy,slope_magnitude,alpha,conductance,face_rate, &
   !$omp& tangent,displacement,normal_slope,tangent_slope,depth_ends,r)
   do f=1,nf
    c=owner(f);n=neighbour(f)
    if(n<=0) cycle
    if(bank_front_tracking) then
     ! A partially traversed cell is an unresolved vertical bank prism.
     ! Ordinary repose diffusion may neither scour nor refill that prism;
     ! its inventory changes only through the conservative moving-front
     ! operator.  This also protects the effective-bed closure from a
     ! deposition update on the receiving side of an avalanche face.
     if(bank_front_status(c)==BANK_FRONT_ACTIVE.or. &
        bank_front_status(n)==BANK_FRONT_ACTIVE) cycle
    endif
    delta=bank_bed(c)-bank_bed(n)
    distance=abs(dot_product(xy(:,n)-xy(:,c),normal(:,f)))
    if(distance<=epsilon(1.d0)) distance=sqrt(sum((xy(:,n)-xy(:,c))**2))
    if(distance<=epsilon(1.d0)) cycle
    ! Correct the two-point normal difference for the tangential offset between
    ! cut-cell centroids. This is zero on orthogonal full cells and exact for a
    ! reconstructed plane, so the aligned link oracle remains unchanged.
    tangent=[-normal(2,f),normal(1,f)];displacement=xy(:,n)-xy(:,c)
    tangent_slope=.5d0*dot_product(bank_down(:,c)+bank_down(:,n),tangent)
    normal_slope=delta/distance-tangent_slope*dot_product(displacement,tangent)/distance
    sx=normal_slope*normal(1,f)+tangent_slope*tangent(1)
    sy=normal_slope*normal(2,f)+tangent_slope*tangent(2)
    slope_magnitude=sqrt(sx*sx+sy*sy)
    if(normal_slope==0) cycle
    donor=c;if(normal_slope<0) donor=n
    if(.not.general_avalanche.and..not.bank_erodible(donor)) cycle
    r=mid(:,f)-xy(:,c)
    depth_ends(:,1)=bank_depth_intercept(c)+[dot_product(bank_down(:,c),r-.5d0*length(f)*tangent), &
                                          dot_product(bank_down(:,c),r+.5d0*length(f)*tangent)]
    r=mid(:,f)-xy(:,n)
    depth_ends(:,2)=bank_depth_intercept(n)+[dot_product(bank_down(:,n),r-.5d0*length(f)*tangent), &
                                          dot_product(bank_down(:,n),r+.5d0*length(f)*tangent)]
    alpha=bank_face_mobility(slope_magnitude,dry_slope,wet_slope,wet_depth_switch,dry_tolerance,depth_ends)
    conductance=diffusivity*length(f)*alpha/distance
    face_rate=diffusivity*length(f)*alpha*normal_slope
    if(conductance>0.and.ieee_is_finite(conductance).and.ieee_is_finite(face_rate)) then
     bank_rate(f)=face_rate;bank_conductance(f)=conductance
    endif
   enddo
   !$omp end parallel do
   ! Bound the two-point diffusion number by 0.45. The explicit nonorthogonal
   ! correction is not an unconditional maximum-principle guarantee; the donor
   ! inventory limiter below separately enforces positivity. Fixed-order cell
   ! gathers and a serial maxval retain bitwise one/eight-thread identity.
   !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) private(c,g,f,rate_sum)
   do c=1,nc
    rate_sum=0
    do g=offset(c),offset(c+1)-1
     f=faces(g);rate_sum=rate_sum+bank_conductance(f)
    enddo
    bank_weight(c)=rate_sum/area(c)
   enddo
   !$omp end parallel do
   maximum_rate=maxval(bank_weight)
   if(maximum_rate<=0) exit
   subdt=min(remaining,.45d0/maximum_rate)
   !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) &
   !$omp& private(f,c,n,donor,g,face_rate,volume,active_volume,share)
   do f=1,nf
    face_rate=bank_rate(f)
    if(face_rate==0) cycle
    c=owner(f);n=neighbour(f)
    if(face_rate>0) then
     donor=c
    else
     donor=n
    endif
    active_volume=0
    do g=1,nfrac
     active_volume=active_volume+state(bed_index(g),donor)/bulk_fraction(g)
    enddo
    if(active_volume<=0) then
     bank_transfer(:,f)=0;cycle
    endif
    volume=subdt*abs(face_rate)
    do g=1,nfrac
     share=(state(bed_index(g),donor)/bulk_fraction(g))/active_volume
     if(donor==c) then
      bank_transfer(g,f)=volume*share*bulk_fraction(g)
     else
      bank_transfer(g,f)=-volume*share*bulk_fraction(g)
     endif
    enddo
   enddo
   !$omp end parallel do
   call apply_surface_transfer(state);call rebalance_active_layer(state)
   remaining=max(0.d0,remaining-subdt)
  enddo
 end subroutine

 subroutine prepare_hydraulic_stage(state)
  ! WAVE/GeoClaw clears momentum in cells classified as dry and caps velocity
  ! before every RK flux evaluation. Without this preparation, an intermediate
  ! state only infinitesimally above dry_tolerance can carry finite momentum;
  ! q/h then creates a nonphysical wave speed and collapses the global CFL.
  ! Water and both sediment inventories are unchanged.
  real(8), intent(inout) :: state(nstate,nc)
  real(8) :: speed
  integer :: c
  !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) private(c,speed)
  do c=1,nc
   if(state(1,c)<=dry_tolerance) then
    state(2:3,c)=0
   else
    speed=sqrt(sum(state(2:3,c)**2))/state(1,c)
    if(speed>speed_limit) then
     state(2:3,c)=state(2:3,c)*(speed_limit/speed)
    endif
   endif
  enddo
  !$omp end parallel do
 end subroutine

 subroutine gradients(v,bc_concentration,grad,vlo,vhi)
  real(8), intent(in) :: v(nprimitive,nc)
  real(8), intent(in) :: bc_concentration(nfrac,nb)
  real(8), intent(out) :: grad(nprimitive,2,nc),vlo(nprimitive,nc),vhi(nprimitive,nc)
  real(8) :: a11,a12,a22,b(nprimitive,2),r(2),w,det,delta(nprimitive),theta(nprimitive),eps_guard(nprimitive),z,adjacent(nprimitive)
  integer :: c,k,f,n,m,g,work,nwork
  grad=0;vlo=v;vhi=v
  if(scheme==1) return
  nwork=nc
  if(amr_step_level>=0) nwork=amr_level_face_cell_count(amr_step_level)
  !$omp parallel do schedule(static,256) if(nwork>=4096.or.nwork*nprimitive>=32768) default(shared) private(work,c,k,f,n,m,g,a11,a12,a22,b,r,w,det,delta,theta,eps_guard,z,adjacent)
  do work=1,nwork
   c=work
   if(amr_step_level>=0) c=amr_level_face_cells(work,amr_step_level)
   if(.not.active_cell(c)) cycle
   a11=hydraulic_ls_matrix(1,c);a12=hydraulic_ls_matrix(2,c);a22=hydraulic_ls_matrix(3,c);b=0
   do k=offset(c),offset(c+1)-1
    f=faces(k);n=neighbour(f)
    if(signs(k)<0) n=owner(f)
    if(n==0) then
     if(bid(f)>0) then
      do g=1,nfrac
       m=concentration_index(g)
       if(.not.flow2dh_boundary_diffusion_declaration) then
        vlo(m,c)=min(vlo(m,c),bc_concentration(g,bid(f)))
        vhi(m,c)=max(vhi(m,c),bc_concentration(g,bid(f)))
       endif
      enddo
     endif
     cycle
    endif
    r=bed_ls_delta(:,k)
    ! Integrate the neighbour equation over its aperture.  A geometric face
    ! split must not count the same neighbour twice in the normal equations.
    w=hydraulic_ls_weight(k)
    adjacent=v(:,n)
    ! High dry terrain is not a higher water surface. Extend this wet cell's
    ! eta into that dry stencil point; otherwise a stationary shoreline can
    ! reconstruct a fictitious uphill water column. Lower dry land remains
    ! available for inundation. Do not alter terrain or conserved inventory.
    if(v(1,c)-v(bed_primitive_index(),c)>dry_tolerance.and. &
       v(1,n)-v(bed_primitive_index(),n)<=dry_tolerance.and. &
       v(bed_primitive_index(),n)>=v(1,c)) &
     adjacent(1)=v(1,c)
    b(:,1)=b(:,1)+w*r(1)*(adjacent-v(:,c));b(:,2)=b(:,2)+w*r(2)*(adjacent-v(:,c))
    vlo(:,c)=min(vlo(:,c),adjacent);vhi(:,c)=max(vhi(:,c),adjacent)
   enddo
   det=hydraulic_ls_matrix(4,c)
   if(det<=1.d-12*(a11+a22)**2) cycle
   grad(:,1,c)=(a22*b(:,1)-a12*b(:,2))/det
   grad(:,2,c)=(a11*b(:,2)-a12*b(:,1))/det
   call cpu_outlet_bounds(c,v,grad,vlo,vhi)
   eps_guard=0.d0
   do m=1,3
    eps_guard(m)=64*epsilon(1.d0)* &
      max(1.d0,abs(v(m,c)),abs(vlo(m,c)),abs(vhi(m,c)))
   enddo
   m=bed_primitive_index()
   eps_guard(m)=64*epsilon(1.d0)* &
     max(1.d0,abs(v(m,c)),abs(vlo(m,c)),abs(vhi(m,c)))
   theta=1
   do k=offset(c),offset(c+1)-1
    f=faces(k);r=mid(:,f)-xy(:,c);delta=grad(:,1,c)*r(1)+grad(:,2,c)*r(2)
    ! A sloping stage outlet lies beyond the last cell-centre extremum.
    ! Limiting against that extremum flattens both terrain and water surface
    ! throughout the end cell and removes its downstream half-cell gravity
    ! contribution. Keep the one-sided slope here; internal faces still
    ! supply all ordinary bounds. Flat-bed and other boundary modes retain
    ! their previous reconstruction exactly.
    if(neighbour(f)==0.and.bid(f)>0) then
     if(mode(bid(f))==2.and.v(1,c)-v(bed_primitive_index(),c)>dry_tolerance) then
      if(abs(dot_product(v(2:3,c),normal(:,f)))<sqrt(grav*(v(1,c)-v(bed_primitive_index(),c)))) then
       if(abs(delta(bed_primitive_index()))>64*epsilon(1.d0)*max(1.d0,abs(v(bed_primitive_index(),c)))) cycle
      endif
     endif
    endif
    do m=1,nprimitive
     if(abs(delta(m))<=eps_guard(m)) cycle
     if(delta(m)>0) theta(m)=min(theta(m),(vhi(m,c)-v(m,c))/delta(m))
     if(delta(m)<0) theta(m)=min(theta(m),(vlo(m,c)-v(m,c))/delta(m))
    enddo
   enddo
   do m=1,nprimitive
    grad(m,:,c)=grad(m,:,c)*max(0.d0,theta(m))
   enddo
  enddo
  !$omp end parallel do
 end subroutine

 ! Immutable geometry is classified once; dynamic WENO arithmetic and
 ! ascending stencil order remain exactly as in the per-call checks.
 subroutine cache_face_geometry()
  integer :: f
  call cache_sediment_geometry()
  call cache_hydraulic_geometry()
  allocate(weno_cells(5,2,nf),face_displacement(2,2,nf))
  weno_cells=0;face_displacement=0
  do f=1,nf
   call cache_one_face(owner(f),f,1)
   if(neighbour(f)>0) call cache_one_face(neighbour(f),f,2)
  enddo
 end subroutine

 subroutine cache_hydraulic_geometry()
  ! Only fixed stencil geometry is cached. Rebuilt with every mesh topology.
  ! Preserve length/sum(r*r), not length times a rounded cached inverse, and
  ! preserve the per-cell face summation order. Wet/dry decisions stay dynamic.
  integer :: c,k,f,n
  real(8) :: r(2),w,a11,a12,a22
  if(allocated(hydraulic_ls_weight)) deallocate(hydraulic_ls_weight,hydraulic_ls_matrix)
  allocate(hydraulic_ls_weight(size(faces)),hydraulic_ls_matrix(4,nc))
  hydraulic_ls_weight=0;hydraulic_ls_matrix=0
  !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) private(c,k,f,n,r,w,a11,a12,a22)
  do c=1,nc
   a11=0;a12=0;a22=0
   do k=offset(c),offset(c+1)-1
    f=faces(k);n=neighbour(f)
    if(signs(k)<0) n=owner(f)
    if(n==0) cycle
    r=bed_ls_delta(:,k);w=length(f)/sum(r*r);hydraulic_ls_weight(k)=w
    a11=a11+w*r(1)**2;a12=a12+w*r(1)*r(2);a22=a22+w*r(2)**2
   enddo
   hydraulic_ls_matrix(:,c)=[a11,a12,a22,a11*a22-a12*a12]
  enddo
  !$omp end parallel do
 end subroutine

 subroutine cache_sediment_geometry()
  ! No solution, wet/dry mask, limiter or bed elevation is cached here.
  ! Keep the original sediment stencil arithmetic and face accumulation order.
  ! Hydrodynamic gradients and all GeoClaw-core routines are left unchanged.
  integer :: c,k,f,n
  real(8) :: r(2),w,a11,a12,a22
  if(allocated(bed_ls_delta)) deallocate(bed_ls_delta,bed_ls_weight,bed_ls_matrix)
  allocate(bed_ls_delta(2,size(faces)),bed_ls_weight(size(faces)),bed_ls_matrix(4,nc))
  bed_ls_delta=0;bed_ls_weight=0;bed_ls_matrix=0
  !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) private(c,k,f,n,r,w,a11,a12,a22)
  do c=1,nc
   do k=offset(c),offset(c+1)-1
    f=faces(k);n=neighbour(f)
    if(signs(k)<0) n=owner(f)
    if(n==0) cycle
    r=xy(:,n)-xy(:,c);bed_ls_delta(:,k)=r
    bed_ls_weight(k)=1.d0/sum(r*r)
   enddo
    a11=0;a12=0;a22=0
    do k=offset(c),offset(c+1)-1
     f=faces(k);n=neighbour(f)
     if(signs(k)<0) n=owner(f)
     if(n==0) cycle
     r=bed_ls_delta(:,k);w=bed_ls_weight(k)
     a11=a11+w*r(1)**2;a12=a12+w*r(1)*r(2);a22=a22+w*r(2)**2
    enddo
    bed_ls_matrix(:,c)=[a11,a12,a22,a11*a22-a12*a12]
  enddo
  !$omp end parallel do
 end subroutine

 subroutine cache_one_face(c,f,side)
  integer, intent(in) :: c,f,side
  integer :: axis,direction,k,i,j,n,indices(5),stride
  real(8) :: r(2),spacing,tol
  logical :: regular
  r=mid(:,f)-xy(:,c);face_displacement(:,side,f)=r
  axis=0
  if(abs(normal(1,f))==1.d0.and.abs(r(2))<1.d-10*dy) axis=1
  if(abs(normal(2,f))==1.d0.and.abs(r(1))<1.d-10*dx) axis=2
  if(axis==0) return
  spacing=dx
  if(axis==2) spacing=dy
  stride=1
  if(amr_enabled) then
   if(allocated(at_leaf)) stride=at_stride(at_leaf(c))
  endif
  spacing=spacing*stride
  tol=1.d-10*max(dx,dy)
  if(any(rectangle_width(:,c)<=0)) return
  if(abs(rectangle_width(axis,c)-spacing)>tol.or.abs(abs(r(axis))-.5d0*spacing)>tol) return
  if(abs(length(f)-rectangle_width(3-axis,c))>tol) return
  direction=1
  if(r(axis)<0) direction=-1
  regular=.true.
  do k=1,5
   i=ii(c);j=jj(c)
   if(axis==1) i=i+(k-3)*direction*stride
   if(axis==2) j=j+(k-3)*direction*stride
   if(i<1.or.i>nx.or.j<1.or.j>ny) then
    regular=.false.;exit
   endif
   n=grid(i,j)
   if(n<=0) then
    regular=.false.;exit
   endif
   if(any(rectangle_width(:,n)<=0).or.any(abs(rectangle_width(:,n)-rectangle_width(:,c))>tol)) then
    regular=.false.;exit
   endif
   if(abs(xy(axis,n)-xy(axis,c)-(k-3)*direction*spacing)>tol.or. &
      abs(xy(3-axis,n)-xy(3-axis,c))>tol) then
    regular=.false.;exit
   endif
   indices(k)=n
  enddo
  if(regular) weno_cells(:,side,f)=indices
 end subroutine

 subroutine cache_wall_geometry()
  ! Optional scheme 6: virtual full squares exist only beside certified
  ! half-square triangles and finite, uninterrupted 45-degree wall segments.
  integer :: c,f,k,a,side,slot,first,wf,legs,mask,nlines,nsegments,g,found,i,j,s,kind
  integer, allocatable :: segment_group(:),memo_id(:,:),memo_kind(:,:)
  real(8), allocatable :: line_normal(:,:),line_point(:,:),segment_lo(:),segment_hi(:),memo_normal(:,:,:)
  logical, allocatable :: full(:)
  real(8) :: tol,ntol,root2,origin(2),point(2),out(2),expected(2),tangent(2),r(2),mapped_normal(2),station,halfspan
  logical :: valid
  allocate(wall_triangle(nc),wall_ready(4,nc),wall_slot(2,nf),wall_stencil_ids(5,2,nc),wall_stencil_kind(5,2,nc))
  allocate(wall_stencil_normal(2,5,2,nc),wall_box_center(2,nc),wall_normal(2,nc),wall_mean_distance(nc),wall_values(nprimitive,4,nc))
  wall_triangle=.false.;wall_ready=.false.;wall_slot=0;wall_stencil_ids=0;wall_stencil_kind=0
  wall_stencil_normal=0;wall_box_center=xy;wall_normal=0;wall_mean_distance=0;wall_values=0
  tol=1.d-10*max(dx,dy);ntol=1.d-12;root2=sqrt(2.d0)
  if(abs(dx-dy)>tol) return
  allocate(full(nc));full=.false.;first=0
  do c=1,nc
   full(c)=all(abs(rectangle_width(:,c)-[dx,dy])<=tol)
   if(full(c).and.first==0) first=c
  enddo
  if(first==0) return
  origin=xy(:,first)-[(ii(first)-.5d0)*dx,(jj(first)-.5d0)*dy]
  do c=1,nc
   wall_box_center(:,c)=origin+[(ii(c)-.5d0)*dx,(jj(c)-.5d0)*dy]
   if(full(c).and.any(abs(xy(:,c)-wall_box_center(:,c))>tol)) full(c)=.false.
  enddo
  ! Three actual faces certify the triangle, rather than area alone.
  ! Two Cartesian legs must be full shared apertures; the diagonal must
  ! be a default wall, excluding inlet/outlet pieces and physical corners.
  do c=1,nc
   if(offset(c+1)-offset(c)/=3.or.abs(area(c)-.5d0*dx*dy)>1.d-10*dx*dy) cycle
   wf=0;legs=0;mask=0;valid=.true.
   do k=offset(c),offset(c+1)-1
    f=faces(k);out=signs(k)*normal(:,f)
    if(neighbour(f)==0.and.bid(f)==0.and.all(abs(abs(out)-1.d0/root2)<ntol)) then
     if(wf/=0) valid=.false.
     wf=f
     if(abs(length(f)-root2*dx)>tol.or.any(abs(mid(:,f)-wall_box_center(:,c))>tol)) valid=.false.
    else
     a=maxloc(abs(out),dim=1)
     if(neighbour(f)<=0.or.abs(abs(out(a))-1.d0)>ntol.or.abs(out(3-a))>ntol.or.abs(length(f)-dx)>tol) then
      valid=.false.;cycle
     endif
     legs=legs+1;mask=ibset(mask,a-1)
    endif
   enddo
   if(.not.valid.or.wf==0.or.legs/=2.or.mask/=3) cycle
   out=normal(:,wf)
   expected=wall_box_center(:,c)-out*dx/(3.d0*root2)
   if(any(abs(xy(:,c)-expected)>tol)) cycle
   do k=offset(c),offset(c+1)-1
    f=faces(k)
    if(f==wf) cycle
    r=signs(k)*normal(:,f);a=maxloc(abs(r),dim=1)
    if(r(a)*out(a)>=0) valid=.false.
    expected=wall_box_center(:,c);expected(a)=expected(a)+.5d0*dx*r(a)
    if(any(abs(mid(:,f)-expected)>tol)) valid=.false.
   enddo
   if(.not.valid) cycle
   wall_triangle(c)=.true.;wall_normal(:,c)=out
   wall_mean_distance(c)=dot_product(xy(:,c)-wall_box_center(:,c),out)
  enddo
  ! Group collinear diagonal walls, retaining every finite segment interval.
  ! Collinearity alone never extends a wall across an opening or a corner.
  allocate(line_normal(2,nf),line_point(2,nf),segment_group(nf),segment_lo(nf),segment_hi(nf))
  nlines=0;nsegments=0
  do f=1,nf
   if(neighbour(f)/=0.or.bid(f)/=0) cycle
   if(any(abs(abs(normal(:,f))-1.d0/root2)>ntol)) cycle
   found=0
   do g=1,nlines
    if(any(abs(normal(:,f)-line_normal(:,g))>ntol)) cycle
    if(abs(dot_product(mid(:,f)-line_point(:,g),line_normal(:,g)))<=tol) then
     found=g;exit
    endif
   enddo
   if(found==0) then
    nlines=nlines+1;found=nlines;line_normal(:,found)=normal(:,f);line_point(:,found)=mid(:,f)
   endif
   tangent=[-line_normal(2,found),line_normal(1,found)]
   station=dot_product(mid(:,f)-line_point(:,found),tangent)
   nsegments=nsegments+1;segment_group(nsegments)=found
   segment_lo(nsegments)=station-.5d0*length(f);segment_hi(nsegments)=station+.5d0*length(f)
  enddo
  allocate(memo_id(-1:nx+2,-1:ny+2),memo_kind(-1:nx+2,-1:ny+2),memo_normal(2,-1:nx+2,-1:ny+2))
  memo_id=-2;memo_kind=0;memo_normal=0
  do c=1,nc
   if(.not.full(c).and..not.wall_triangle(c)) cycle
   do a=1,2
    do k=1,5
     i=ii(c);j=jj(c)
     if(a==1) i=i+k-3
     if(a==2) j=j+k-3
     call virtual_cell(i,j,s,kind,mapped_normal)
     if(s==0) exit
     wall_stencil_ids(k,a,c)=s;wall_stencil_kind(k,a,c)=kind;wall_stencil_normal(:,k,a,c)=mapped_normal
    enddo
    if(any(wall_stencil_ids(:,a,c)==0)) then
     wall_stencil_ids(:,a,c)=0;wall_stencil_kind(:,a,c)=0;wall_stencil_normal(:,:,a,c)=0
    endif
   enddo
   if(wall_triangle(c).and.any(wall_stencil_ids(:,:,c)==0)) wall_stencil_ids(:,:,c)=0
   do k=offset(c),offset(c+1)-1
    f=faces(k);r=mid(:,f)-wall_box_center(:,c);a=0
    if(abs(normal(1,f))==1.d0.and.abs(r(2))<=tol) a=1
    if(abs(normal(2,f))==1.d0.and.abs(r(1))<=tol) a=2
    if(a==0) cycle
    if(abs(abs(r(a))-.5d0*dx)>tol.or.abs(length(f)-dx)>tol) cycle
    slot=2*a
    if(r(a)<0) slot=slot-1
    side=1
    if(c/=owner(f)) side=2
    wall_slot(side,f)=slot
   enddo
  enddo
 contains
  logical function interval_covered(group,lo,hi) result(covered)
   integer, intent(in) :: group
   real(8), intent(in) :: lo,hi
   real(8) :: reach,next_reach
   integer :: attempt,segment
   reach=lo;covered=.false.
   do attempt=1,nsegments+1
    if(reach>=hi-tol) then
     covered=.true.;return
    endif
    next_reach=reach
    do segment=1,nsegments
     if(segment_group(segment)/=group) cycle
     if(segment_lo(segment)<=reach+tol.and.segment_hi(segment)>=reach-tol) next_reach=max(next_reach,segment_hi(segment))
    enddo
    if(next_reach<=reach) return
    reach=next_reach
   enddo
  end function

  subroutine virtual_cell(i,j,source,kind,n)
   integer, intent(in) :: i,j
   integer, intent(out) :: source,kind
   real(8), intent(out) :: n(2)
   integer :: actual,group,is,js,candidate
   real(8) :: center(2),mirror(2),distance,radius,t(2),station
   source=0;kind=0;n=0
   if(memo_id(i,j)/=-2) then
    source=memo_id(i,j);kind=memo_kind(i,j);n=memo_normal(:,i,j);return
   endif
   actual=0
   if(i>=1.and.i<=nx.and.j>=1.and.j<=ny) actual=grid(i,j)
   if(actual/=0) then
    if(actual>0) then
     if(full(actual)) source=actual
     if(wall_triangle(actual)) then
      source=actual;kind=1;n=wall_normal(:,actual)
     endif
    endif
   else
    center=origin+[(i-.5d0)*dx,(j-.5d0)*dy]
    do group=1,nlines
     distance=dot_product(center-line_point(:,group),line_normal(:,group))
     radius=.5d0*(dx*abs(line_normal(1,group))+dy*abs(line_normal(2,group)))
     if(distance<=tol.or.distance<radius-tol) cycle
     mirror=center-2.d0*distance*line_normal(:,group)
     is=nint((mirror(1)-origin(1))/dx+.5d0);js=nint((mirror(2)-origin(2))/dy+.5d0)
     if(is<1.or.is>nx.or.js<1.or.js>ny) cycle
     candidate=grid(is,js)
     if(candidate<=0) cycle
     if(.not.full(candidate).or.any(abs(xy(:,candidate)-mirror)>tol)) cycle
     t=[-line_normal(2,group),line_normal(1,group)]
     station=dot_product(center-line_point(:,group),t)
     radius=.5d0*(dx*abs(t(1))+dy*abs(t(2)))
     if(.not.interval_covered(group,station-radius,station+radius)) cycle
     if(source/=0) then
      ! Ambiguous reflection near distinct walls/corners is not admitted.
      source=0;kind=0;n=0;exit
     endif
     source=candidate;kind=2;n=line_normal(:,group)
    enddo
   endif
   memo_id(i,j)=source;memo_kind(i,j)=kind;memo_normal(:,i,j)=n
  end subroutine
 end subroutine

 subroutine prepare_wall_traces(state,v,vlo,vhi)
  ! Read-only geometry; private per-cell preparation once per RK stage.
  ! Slots are -x,+x,-y,+y. A triangle either admits all four or falls back
  ! everywhere, preserving one common wall frame and its actual q_n mean.
  real(8), intent(in) :: state(nstate,nc),v(nprimitive,nc),vlo(nprimitive,nc),vhi(nprimitive,nc)
  integer :: c,a,side,slot,k,kk,s,kind,m,f,other
  real(8) :: stencil(5,nprimitive),tangential(5),n(2),tangent(2),un,coef,tlo,thi,ut,h,point(2),ratio,normal_q
  logical :: wet
  wall_ready=.false.;wall_values=0
  !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) private(c,a,side,slot,k,kk,s,kind,m,f,other,stencil,tangential,n,tangent,un,coef,tlo,thi,ut,h,point,ratio,normal_q,wet)
  do c=1,nc
   if(state(1,c)<=dry_tolerance) cycle
   if(wall_triangle(c)) then
    if(any(wall_stencil_ids(:,:,c)==0)) cycle
    n=wall_normal(:,c);tangent=[-n(2),n(1)]
    tlo=dot_product(v(2:3,c),tangent);thi=tlo
    do k=offset(c),offset(c+1)-1
     f=faces(k);other=neighbour(f)
     if(signs(k)<0) other=owner(f)
     if(other<=0) cycle
     ut=dot_product(v(2:3,other),tangent);tlo=min(tlo,ut);thi=max(thi,ut)
    enddo
   endif
   do a=1,2
    if(any(wall_stencil_ids(:,a,c)==0)) cycle
    wet=.true.
    do k=1,5
     if(state(1,wall_stencil_ids(k,a,c))<=dry_tolerance) wet=.false.
    enddo
    if(.not.wet) cycle
    do side=1,2
     slot=2*(a-1)+side
     do k=1,5
      kk=k
      if(side==1) kk=6-k
      s=wall_stencil_ids(kk,a,c);kind=wall_stencil_kind(kk,a,c)
      stencil(k,:)=v(:,s)
      if(kind>0) then
       n=wall_stencil_normal(:,kk,a,c);coef=real(kind,8)
       un=dot_product(stencil(k,2:3),n);stencil(k,2:3)=stencil(k,2:3)-coef*un*n
      endif
      if(wall_triangle(c)) tangential(k)=dot_product(stencil(k,2:3),tangent)
     enddo
     do m=1,nprimitive
      wall_values(m,slot,c)=reconstruct(stencil(:,m),5)
     enddo
     wall_values(1,slot,c)=max(vlo(1,c),min(vhi(1,c),wall_values(1,slot,c)))
     wall_values(bed_primitive_index(),slot,c)=max(vlo(bed_primitive_index(),c), &
      min(vhi(bed_primitive_index(),c),wall_values(bed_primitive_index(),slot,c)))
     if(wall_triangle(c)) then
      ut=max(tlo,min(thi,reconstruct(tangential,5)))
      wall_values(2:3,slot,c)=ut*tangent
     else
      wall_values(2:3,slot,c)=max(vlo(2:3,c),min(vhi(2:3,c),wall_values(2:3,slot,c)))
     endif
     wall_ready(slot,c)=.true.
    enddo
   enddo
   if(.not.wall_triangle(c)) cycle
   if(.not.all(wall_ready(:,c)).or. &
      any(wall_values(1,:,c)-wall_values(bed_primitive_index(),:,c)<=dry_tolerance)) then
    wall_ready(:,c)=.false.;cycle
   endif
   n=wall_normal(:,c);normal_q=dot_product(state(2:3,c),n)
   do a=1,2
    do side=1,2
     slot=2*(a-1)+side;h=wall_values(1,slot,c)-wall_values(bed_primitive_index(),slot,c)
     point=wall_box_center(:,c);point(a)=point(a)+(side-1.5d0)*dx
     ratio=dot_product(point-wall_box_center(:,c),n)/wall_mean_distance(c)
     wall_values(2:3,slot,c)=wall_values(2:3,slot,c)+normal_q*ratio/h*n
    enddo
   enddo
  enddo
  !$omp end parallel do
 end subroutine

 function face_value(c,f,v,grad,vlo,vhi) result(value)
  integer, intent(in) :: c,f
  real(8), intent(in) :: v(nprimitive,nc),grad(nprimitive,2,nc),vlo(nprimitive,nc),vhi(nprimitive,nc)
  real(8) :: value(nprimitive),r(2),stencil(5,nprimitive)
  integer :: side,k,slot,m
  value=v(:,c)
  if(scheme==1) return
  side=1
  if(c/=owner(f)) side=2
  if(scheme==6) then
   slot=wall_slot(side,f)
   if(slot>0) then
    if(wall_ready(slot,c)) then
     value=wall_values(:,slot,c)
     if(hydro_constant(c)) then
      value(1:3)=v(1:3,c)
      if(v(1,c)-v(bed_primitive_index(),c)<=dry_tolerance) &
       value(bed_primitive_index())=v(bed_primitive_index(),c)
     endif
     return
    endif
   endif
  endif
  r=face_displacement(:,side,f);value=value+grad(:,1,c)*r(1)+grad(:,2,c)*r(2)
  if(weno_cells(1,side,f)==0) then
   if(hydro_constant(c)) then
    value(1:3)=v(1:3,c)
    if(v(1,c)-v(bed_primitive_index(),c)<=dry_tolerance) &
     value(bed_primitive_index())=v(bed_primitive_index(),c)
   endif
   return
  endif
  do k=1,5
   stencil(k,:)=v(:,weno_cells(k,side,f))
   ! Apply the same dry-land surface extension as the linear reconstruction.
   if(v(1,c)-v(bed_primitive_index(),c)>dry_tolerance.and. &
      stencil(k,1)-stencil(k,bed_primitive_index())<=dry_tolerance.and. &
      stencil(k,bed_primitive_index())>=v(1,c)) &
    stencil(k,1)=v(1,c)
  enddo
  do m=1,nprimitive
   value(m)=reconstruct(stencil(:,m),5)
  enddo
  value(1:3)=max(vlo(1:3,c),min(vhi(1:3,c),value(1:3)))
  value(bed_primitive_index())=max(vlo(bed_primitive_index(),c), &
   min(vhi(bed_primitive_index(),c),value(bed_primitive_index())))
  if(hydro_constant(c)) then
   value(1:3)=v(1:3,c)
   if(v(1,c)-v(bed_primitive_index(),c)<=dry_tolerance) &
    value(bed_primitive_index())=v(bed_primitive_index(),c)
  endif
 end function


 subroutine face_elevations(c,f,v,grad,vlo,vhi,eta,zb)
  ! The wet/dry predicate consumes only these two face_value components.
  ! Same WENO/limiter arithmetic; no velocity or scalar reconstruction needed.
  integer,intent(in) :: c,f
  real(8),intent(in) :: v(nprimitive,nc),grad(nprimitive,2,nc),vlo(nprimitive,nc),vhi(nprimitive,nc)
  real(8),intent(out) :: eta,zb
  real(8) :: r(2),stencil(5,2),value(2)
  integer :: side,slot,k,s,bi
  bi=bed_primitive_index();value=[v(1,c),v(bi,c)]
  if(scheme/=1) then
   side=1
   if(c/=owner(f)) side=2
   slot=0
   if(scheme==6) slot=wall_slot(side,f)
   if(slot>0) then
    if(wall_ready(slot,c)) then
     value=[wall_values(1,slot,c),wall_values(bi,slot,c)]
     eta=value(1);zb=value(2)
     if(hydro_constant(c)) then
      eta=v(1,c)
      if(v(1,c)-v(bi,c)<=dry_tolerance) zb=v(bi,c)
     endif
     return
    endif
   endif
   r=face_displacement(:,side,f)
   value=value+[grad(1,1,c),grad(bi,1,c)]*r(1)+[grad(1,2,c),grad(bi,2,c)]*r(2)
   if(weno_cells(1,side,f)/=0) then
    do k=1,5
     s=weno_cells(k,side,f);stencil(k,:)=[v(1,s),v(bi,s)]
     if(v(1,c)-v(bi,c)>dry_tolerance.and.stencil(k,1)-stencil(k,2)<=dry_tolerance.and. &
        stencil(k,2)>=v(1,c)) stencil(k,1)=v(1,c)
    enddo
    value(1)=max(vlo(1,c),min(vhi(1,c),reconstruct(stencil(:,1),5)))
    value(2)=max(vlo(bi,c),min(vhi(bi,c),reconstruct(stencil(:,2),5)))
   endif
   if(hydro_constant(c)) then
    value(1)=v(1,c)
    if(v(1,c)-v(bi,c)<=dry_tolerance) value(2)=v(bi,c)
   endif
  endif
  eta=value(1);zb=value(2)
 end subroutine

 subroutine prepare_wet_dry_traces(state,v,grad,vlo,vhi)
  ! A partly dry reconstruction keeps cell-constant surface and velocity,
  ! while retaining the bounded reconstructed bed. Flattening the bed too
  ! inserts artificial Cartesian steps into a smooth oblique bank.
  ! The face source below integrates hydrostatic pressure to h=0.
  ! A steep bed alone is not a dry face: eta and bed may fall together.
  ! Fully wet rectangular cells retain reconstruction and the wet-column
  ! bed-source quadrature below. Keep the prior excursion guard on irregular
  ! cells; these now use the same constant-surface pressure balance.
  ! Scalar reconstruction and shared FCT are unchanged.
  real(8), intent(in) :: state(nstate,nc),v(nprimitive,nc),grad(nprimitive,2,nc),vlo(nprimitive,nc),vhi(nprimitive,nc)
  integer :: c,k,f,work,nwork
  real(8) :: eta,zb,h
  logical :: flatten
  hydro_constant=.false.
  if(scheme==1) return
  nwork=nc
  if(amr_step_level>=0) nwork=amr_level_eval_cell_count(amr_step_level)
  !$omp parallel do schedule(static,256) if(nwork>=4096) default(shared) private(work,c,k,f,eta,zb,h,flatten)
  do work=1,nwork
   c=work
   if(amr_step_level>=0) c=amr_level_eval_cells(work,amr_step_level)
   h=state(1,c);flatten=h<=dry_tolerance
   if(.not.flatten) then
    do k=offset(c),offset(c+1)-1
     f=faces(k);call face_elevations(c,f,v,grad,vlo,vhi,eta,zb)
     if(eta-zb<=dry_tolerance.or. &
        (any(rectangle_width(:,c)<=0).and. &
         abs(zb-v(bed_primitive_index(),c))>=h)) then
      flatten=.true.;exit
     endif
    enddo
   endif
   hydro_constant(c)=flatten
  enddo
  !$omp end parallel do
 end subroutine

 subroutine wave_flux(left,right,bl,br,n,fl,fr,speed,rightmost_speed)
  real(8), intent(in) :: left(3),right(3),bl,br,n(2)
  real(8), intent(out) :: fl(3),fr(3),speed
  real(8), optional, intent(out) :: rightmost_speed
  real(8) :: ql(3,0:2),qr(3,0:2),al(1,0:2),ar(1,0:2),fw(3,3,0:2),sw(3,0:2)
  real(8) :: am(3,0:2),ap(3,0:2),l(3),r(3),pl(3),pr(3),t(2),fluxl(3),fluxr(3)
  external rpn2
  t=[-n(2),n(1)]
  l=[left(1),dot_product(left(2:3),n),dot_product(left(2:3),t)]
  r=[right(1),dot_product(right(2:3),n),dot_product(right(2:3),t)]
  ql(:,0)=r;ql(:,1)=r;ql(:,2)=r
  qr(:,0)=l;qr(:,1)=l;qr(:,2)=l;al=br;ar=bl
  ! With mbc=1 the loop is i=1:mx+1. Only interface 1 is consumed,
  ! so mx=0 avoids an identical unused second solve; maxm/extents stay 1.
  call rpn2(1,1,3,3,1,1,0,ql,qr,al,ar,fw,sw,am,ap)
  pl=0;pr=0
  if(l(1)>dry_tolerance) pl=[l(2),l(2)**2/l(1)+.5d0*grav*l(1)**2,l(2)*l(3)/l(1)]
  if(r(1)>dry_tolerance) pr=[r(2),r(2)**2/r(1)+.5d0*grav*r(1)**2,r(2)*r(3)/r(1)]
  fluxl=pl+am(:,1);fluxr=pr-ap(:,1)
  fl=[fluxl(1),fluxl(2)*n(1)+fluxl(3)*t(1),fluxl(2)*n(2)+fluxl(3)*t(2)]
  fr=[fluxr(1),fluxr(2)*n(1)+fluxr(3)*t(1),fluxr(2)*n(2)+fluxr(3)*t(2)]
  if(abs(fl(1)-fr(1))>1.d-10*max(1.d0,abs(fl(1)),abs(fr(1)))) error stop 'non-shared Riemann water flux'
  fl(1)=.5d0*(fl(1)+fr(1));fr(1)=fl(1);speed=maxval(abs(sw(:,1)))
  if(present(rightmost_speed)) rightmost_speed=maxval(sw(:,1))
 end subroutine

 logical function dynamic_runup_opens_face(left,right,bl,br,n) result(open)
  ! A centre-stage wet/dry aperture is a low-velocity activation filter.  It
  ! must not suppress genuine bore run-up.  Use the same reflected Riemann
  ! height as the dry-step solver, but admit an override only when momentum is
  ! directed at the step and reflection raises the predicted surface beyond
  ! both the undisturbed depth and the higher bed by the dry tolerance.
  real(8), intent(in) :: left(3),right(3),bl,br,n(2)
  real(8) :: h,approach,hstar,s1m,s2m
  logical :: rare1,rare2
  open=.false.
  if(left(1)>dry_tolerance.and.right(1)<=dry_tolerance) then
   h=left(1);approach=dot_product(left(2:3),n)/h
   if(approach<=0) return
   call riemanntype(h,h,approach,-approach,hstar,s1m,s2m,rare1,rare2,1,dry_tolerance,grav)
   open=hstar>h+dry_tolerance.and.hstar+bl>br+dry_tolerance
  else if(right(1)>dry_tolerance.and.left(1)<=dry_tolerance) then
   h=right(1);approach=-dot_product(right(2:3),n)/h
   if(approach<=0) return
   call riemanntype(h,h,approach,-approach,hstar,s1m,s2m,rare1,rare2,1,dry_tolerance,grav)
   open=hstar>h+dry_tolerance.and.hstar+br>bl+dry_tolerance
  endif
 end function

 logical function bedload_face_open(left,right,bl,br,n) result(open)
  ! Match GeoClaw's dry-step wall decision, including the reflected wave
  ! height. A water-driven sediment capacity must not cross a face that
  ! the hydraulic Riemann problem treats as a wall. A low dry bed or an
  ! overtopped step stays open; this is not a blanket wet/wet restriction.
  real(8), intent(in) :: left(3),right(3),bl,br,n(2)
  real(8) :: h,u,hstar,s1m,s2m
  logical :: rare1,rare2
  open=.true.
  if(left(1)<=dry_tolerance.and.right(1)<=dry_tolerance) then
   open=.false.
  else if(right(1)<=dry_tolerance) then
   h=left(1);u=dot_product(left(2:3),n)/h
   call riemanntype(h,h,u,-u,hstar,s1m,s2m,rare1,rare2,1,dry_tolerance,grav)
   open=max(h,hstar)+bl>=br
  else if(left(1)<=dry_tolerance) then
   h=right(1);u=dot_product(right(2:3),n)/h
   call riemanntype(h,h,-u,u,hstar,s1m,s2m,rare1,rare2,1,dry_tolerance,grav)
   open=max(h,hstar)+br>=bl
  endif
 end function

 subroutine boundary_sediment_at(evaluation_time,concentration,bed_input)
  ! Piecewise-linear forcing is evaluated at the actual SSPRK stage time.
  ! Values before/after the declared range hold the first/last endpoint.
  real(8), intent(in) :: evaluation_time
  real(8), intent(out) :: concentration(nfrac,nb),bed_input(nfrac,nb)
  real(8) :: weight
  integer :: b,k,f
  concentration=fraction_concentration;bed_input=fraction_bed_input
  do b=1,nb
   if(nseries(b)>0) then
    if(evaluation_time<=series_time(1,b)) then
     concentration(1,b)=series_concentration(1,b);bed_input(1,b)=series_bed_input(1,b)
    else if(evaluation_time>=series_time(nseries(b),b)) then
     concentration(1,b)=series_concentration(nseries(b),b);bed_input(1,b)=series_bed_input(nseries(b),b)
    else
     k=series_interval(series_time(:,b),nseries(b),evaluation_time)
     weight=(evaluation_time-series_time(k-1,b))/(series_time(k,b)-series_time(k-1,b))
     concentration(1,b)=(1-weight)*series_concentration(k-1,b)+weight*series_concentration(k,b)
     bed_input(1,b)=(1-weight)*series_bed_input(k-1,b)+weight*series_bed_input(k,b)
    endif
   endif
   do f=1,nfrac
    if(nfraction_series(f,b)==0) cycle
    if(evaluation_time<=fraction_series_time(1,f,b)) then
     concentration(f,b)=fraction_series_concentration(1,f,b)
     bed_input(f,b)=fraction_series_bed_input(1,f,b)
    else if(evaluation_time>=fraction_series_time(nfraction_series(f,b),f,b)) then
     concentration(f,b)=fraction_series_concentration(nfraction_series(f,b),f,b)
     bed_input(f,b)=fraction_series_bed_input(nfraction_series(f,b),f,b)
    else
     k=series_interval(fraction_series_time(:,f,b),nfraction_series(f,b),evaluation_time)
     weight=(evaluation_time-fraction_series_time(k-1,f,b))/ &
            (fraction_series_time(k,f,b)-fraction_series_time(k-1,f,b))
     concentration(f,b)=(1-weight)*fraction_series_concentration(k-1,f,b)+ &
                         weight*fraction_series_concentration(k,f,b)
     bed_input(f,b)=(1-weight)*fraction_series_bed_input(k-1,f,b)+ &
                     weight*fraction_series_bed_input(k,f,b)
    endif
   enddo
  enddo
 end subroutine

 subroutine boundary_hydraulic_at(evaluation_time,bed,hydraulic,ok)
  ! Resolved total discharge and absolute water level are interpolated at the
  ! actual SSPRK stage time. Outside the series range, hold the nearest knot.
  real(8), intent(in) :: evaluation_time
  real(8), intent(in) :: bed(nc)
  real(8), intent(out) :: hydraulic(2,nb)
  logical, intent(out) :: ok
  real(8) :: weight
  integer :: b,k,f
  logical :: solved,cached
  hydraulic=forcing(1:2,:);ok=.false.
  do b=1,nb
   if(nhydro_series(b)==0) cycle
   if(evaluation_time<=hydro_series_time(1,b)) then
    hydraulic(:,b)=[hydro_series_q(1,b),hydro_series_stage(1,b)]
   else if(evaluation_time>=hydro_series_time(nhydro_series(b),b)) then
    hydraulic(:,b)=[hydro_series_q(nhydro_series(b),b),hydro_series_stage(nhydro_series(b),b)]
   else
    k=series_interval(hydro_series_time(:,b),nhydro_series(b),evaluation_time)
    weight=(evaluation_time-hydro_series_time(k-1,b))/(hydro_series_time(k,b)-hydro_series_time(k-1,b))
    hydraulic(1,b)=(1-weight)*hydro_series_q(k-1,b)+weight*hydro_series_q(k,b)
    hydraulic(2,b)=(1-weight)*hydro_series_stage(k-1,b)+weight*hydro_series_stage(k,b)
   endif
  enddo
  if(production_section_geometry) then
   do b=1,nb
    if(section_control(b)==0) cycle
    ! Normal/critical section roots depend only on the prescribed discharge
    ! and the current bed beneath that boundary.  A fixed-bed run previously
    ! rebuilt the section and repeated the same root search at every SSPRK
    ! stage.  Reuse the root while those exact inputs remain unchanged; an
    ! evolving boundary bed or hydrograph invalidates the entry immediately.
    cached=section_cache_valid(b).and.section_cached_discharge(b)==hydraulic(1,b)
    if(cached) then
     do k=boundary_offset(b),boundary_offset(b+1)-1
      f=boundary_faces(k)
      if(section_cached_bed(f)/=bed(owner(f))) then
       cached=.false.;exit
      endif
     enddo
    endif
    if(cached) then
     hydraulic(2,b)=section_cached_stage(b);solved=.true.
    else
     section_cache_valid(b)=.false.
     call live_section_stage(b,hydraulic(1,b),bed,hydraulic(2,b),solved)
     if(solved) then
      section_cached_stage(b)=hydraulic(2,b);section_cached_discharge(b)=hydraulic(1,b)
      do k=boundary_offset(b),boundary_offset(b+1)-1
       f=boundary_faces(k);section_cached_bed(f)=bed(owner(f))
      enddo
      section_cache_valid(b)=.true.
     endif
    endif
    if(.not.solved) then
     failure_reason=8;failed_trace_boundary=b;failed_trace_condition=6
     failed_trace_values=[hydraulic(1,b),hydraulic(2,b),real(section_control(b),8),0.d0,0.d0,0.d0]
     return
    endif
   enddo
  endif
  ok=.true.
 end subroutine

 function next_boundary_knot(current_time) result(next_time)
  ! Do not let an accepted timestep straddle a forcing knot. This
  ! makes the native SSPRK boundary ledger integrate every linear segment to
  ! roundoff while retaining adaptive CFL/retry behavior inside the segment.
  real(8), intent(in) :: current_time
  real(8) :: next_time,tolerance
  integer :: b,k,f
  next_time=huge(1.d0);tolerance=32*epsilon(1.d0)*max(1.d0,abs(current_time))
  do b=1,nb
   do k=1,nseries(b)
    if(series_time(k,b)>current_time+tolerance) next_time=min(next_time,series_time(k,b))
   enddo
   do f=1,nfrac
    do k=1,nfraction_series(f,b)
     if(fraction_series_time(k,f,b)>current_time+tolerance) &
      next_time=min(next_time,fraction_series_time(k,f,b))
    enddo
   enddo
   do k=1,nhydro_series(b)
    if(hydro_series_time(k,b)>current_time+tolerance) next_time=min(next_time,hydro_series_time(k,b))
   enddo
  enddo
 end function

 function morphological_factor_at(evaluation_time) result(value)
  real(8), intent(in) :: evaluation_time
  real(8) :: value
  integer :: k
  value=morphological_factor
  if(nmorfac==0) return
  if(evaluation_time<=morfac_time(1)) then
   value=morfac_value(1);return
  endif
  if(evaluation_time>=morfac_time(nmorfac)) then
   value=morfac_value(nmorfac);return
  endif
  do k=2,nmorfac
   if(evaluation_time<morfac_time(k)) then
    value=morfac_value(k-1);return
   endif
  enddo
 end function

 function next_morphological_knot(current_time) result(next_time)
  real(8), intent(in) :: current_time
  real(8) :: next_time,tolerance
  integer :: k
  next_time=huge(1.d0);tolerance=32*epsilon(1.d0)*max(1.d0,abs(current_time))
  do k=1,nmorfac
   if(morfac_time(k)>current_time+tolerance) next_time=min(next_time,morfac_time(k))
  enddo
 end function

 function morphological_time_at(hydraulic_time) result(value)
  ! Exact integral of Delft3D's block-interpolated acceleration curve.
  ! The clock is derived from immutable deck data and restored hydraulic time,
  ! so checkpoint/restart needs no additional history-dependent state.
  real(8), intent(in) :: hydraulic_time
  real(8) :: value,left,right,span
  integer :: k
  if(hydraulic_time<=0.d0) then
   value=0.d0;return
  endif
  if(nmorfac==0) then
   value=morphological_factor*hydraulic_time;return
  endif
  value=morfac_value(1)*min(hydraulic_time,morfac_time(1))
  if(hydraulic_time<=morfac_time(1)) return
  do k=2,nmorfac
   left=morfac_time(k-1);right=min(hydraulic_time,morfac_time(k));span=right-left
   if(span>0.d0) value=value+span*morfac_value(k-1)
   if(hydraulic_time<=morfac_time(k)) return
  enddo
  value=value+morfac_value(nmorfac)*(hydraulic_time-morfac_time(nmorfac))
 end function

 subroutine section_nodes(b,bed,nodes)
  ! Reconstruct a continuous piecewise-linear boundary profile. Adjacent
  ! solver owners can evolve by different amounts, so their shared node uses
  ! the arithmetic mean of the two represented endpoint elevations.
  integer, intent(in) :: b
  real(8), intent(in) :: bed(nc)
  real(8), intent(out) :: nodes(:)
  integer :: base,last,k,f
  real(8) :: previous_right,left,right
  base=boundary_offset(b);last=boundary_offset(b+1)-1
  if(size(nodes)/=last-base+2) error stop 'wrong production section node count'
  f=boundary_faces(base)
  nodes(1)=bed(owner(f))+boundary_bed_left_offset(f)
  previous_right=bed(owner(f))+boundary_bed_right_offset(f)
  do k=base+1,last
   f=boundary_faces(k);left=bed(owner(f))+boundary_bed_left_offset(f)
   nodes(k-base+1)=.5d0*(previous_right+left)
   previous_right=bed(owner(f))+boundary_bed_right_offset(f)
  enddo
  nodes(size(nodes))=previous_right
 end subroutine

 subroutine sort_real_ascending(values,count)
  ! In-place heapsort. Section nodes can evolve independently, so their
  ! elevation order cannot be cached; O(n log n) avoids the former insertion
  ! sort's quadratic cost without changing the exact sorted values.
  real(8), intent(inout) :: values(:)
  integer, intent(in) :: count
  integer :: root,child,end_position
  real(8) :: held
  if(count<2) return
  do root=count/2,1,-1
   held=values(root);child=2*root
   do while(child<=count)
    if(child<count.and.values(child)<values(child+1)) child=child+1
    if(held>=values(child)) exit
    values(child/2)=values(child);child=2*child
   enddo
   values(child/2)=held
  enddo
  do end_position=count,2,-1
   held=values(end_position);values(end_position)=values(1);child=2
   do while(child<end_position)
    if(child<end_position-1.and.values(child)<values(child+1)) child=child+1
    if(held>=values(child)) exit
    values(child/2)=values(child);child=2*child
   enddo
   values(child/2)=held
  enddo
 end subroutine

 subroutine sampled_section_span(b,stage,bed,first,last)
  integer, intent(in) :: b
  real(8), intent(in) :: stage,bed(nc)
  integer, intent(out) :: first,last
  integer :: f,k
  first=boundary_offset(b);last=boundary_offset(b+1)-1
  ! One assigned inlet supplies all its submerged portions at a common stage.
  ! A dry crest closes only its own faces, not another wet channel beyond it.
  ! Keep the connected selector below solely for legacy characteristic decks.
  if(production_section_geometry.or.mode(b)==9) return
  if(boundary_seed_position(b)==0) return
  k=boundary_seed_position(b);f=boundary_faces(k)
  if(stage<=bed(owner(f))+boundary_bed_offset(f)) then
   last=first-1;return
  endif
  first=k;last=k
  do k=first-1,boundary_offset(b),-1
   f=boundary_faces(k)
   if(stage<=bed(owner(f))+boundary_bed_offset(f)) exit
   first=k
  enddo
  do k=last+1,boundary_offset(b+1)-1
   f=boundary_faces(k)
   if(stage<=bed(owner(f))+boundary_bed_offset(f)) exit
   last=k
  enddo
 end subroutine

 subroutine section_metrics(b,stage,bed,wet_area,top_width,conveyance)
  ! Integrate the live piecewise-linear terrain. Equal adjacent Manning cells
  ! form one subsection; artificial vertical partition lines do not contribute
  ! to wetted perimeter.
  integer, intent(in) :: b
  real(8), intent(in) :: stage,bed(nc)
  real(8), intent(out) :: wet_area,top_width,conveyance
  integer :: base,last
  real(8), allocatable :: nodes(:)
  base=boundary_offset(b);last=boundary_offset(b+1)-1
  allocate(nodes(last-base+2));call section_nodes(b,bed,nodes)
  call section_metrics_nodes(b,stage,bed,nodes,wet_area,top_width,conveyance)
  deallocate(nodes)
 end subroutine

 subroutine section_metrics_nodes(b,stage,bed,nodes,wet_area,top_width,conveyance)
  ! Allocation-free form used repeatedly by the live section root search.
  integer, intent(in) :: b
  real(8), intent(in) :: stage,bed(nc),nodes(:)
  real(8), intent(out) :: wet_area,top_width,conveyance
  integer :: base,last,first_wet,last_wet,k,f,node
  real(8) :: h0,h1,width,fraction,segment_area,segment_perimeter,group_area,group_perimeter,group_n,nvalue
  wet_area=0;top_width=0;conveyance=0
  base=boundary_offset(b);last=boundary_offset(b+1)-1
  if(size(nodes)/=last-base+2) error stop 'wrong production section node count'
  call sampled_section_span(b,stage,bed,first_wet,last_wet)
  if(last_wet<first_wet) return
  group_area=0;group_perimeter=0;group_n=-1
  do k=first_wet,last_wet
   f=boundary_faces(k);node=k-base+1
   ! Roughness subsections are contiguous terrain segments, including dry
   ! segments. A dry roughness zone must not merge distinct wet subsections.
   nvalue=manning_cell(owner(f))
   if(group_n>0.and.nvalue/=group_n) then
    if(group_area>0.and.group_perimeter>0) then
     if(hydraulic_roughness_mode==braids_roughness_manning.and. &
        roughness_minimum_work_depth==0.d0) then
      conveyance=conveyance+group_area*(group_area/group_perimeter)**(2.d0/3.d0)/group_n
     else
      conveyance=conveyance+roughness_conveyance(group_area,group_perimeter,group_n)
     endif
    endif
    group_area=0;group_perimeter=0
   endif
   group_n=nvalue
   ! A production section terminates against the neighbouring reflective
   ! domain edges. If morphology lowers an endpoint below the controlled water
   ! line, that edge is a vertical sidewall rather than a new lateral outlet.
   ! Its wetted height belongs to the adjacent roughness subsection. Below the
   ! endpoint this is exactly zero, so established section solutions are
   ! unchanged bit for bit.
   if(production_section_geometry.and.k==base.and.stage>nodes(1)) &
    group_perimeter=group_perimeter+stage-nodes(1)
   h0=stage-nodes(node);h1=stage-nodes(node+1)
   if(max(h0,h1)<=0) cycle
   if(min(h0,h1)>0) then
    fraction=1
   else if(h0>0) then
    fraction=h0/(h0-h1)
   else
    fraction=h1/(h1-h0)
   endif
   width=length(f)*fraction
   segment_area=.5d0*(max(h0,0.d0)+max(h1,0.d0))*width
   segment_perimeter=hypot(length(f),nodes(node+1)-nodes(node))*fraction
   wet_area=wet_area+segment_area;top_width=top_width+width
   group_area=group_area+segment_area;group_perimeter=group_perimeter+segment_perimeter
  enddo
  if(production_section_geometry.and.last_wet==last.and.stage>nodes(size(nodes))) &
   group_perimeter=group_perimeter+stage-nodes(size(nodes))
  if(group_area>0.and.group_perimeter>0.and.group_n>0) then
   if(hydraulic_roughness_mode==braids_roughness_manning.and. &
      roughness_minimum_work_depth==0.d0) then
    conveyance=conveyance+group_area*(group_area/group_perimeter)**(2.d0/3.d0)/group_n
   else
    conveyance=conveyance+roughness_conveyance(group_area,group_perimeter,group_n)
   endif
  endif
 end subroutine

 function live_section_rate(b,stage,bed) result(rate)
  integer, intent(in) :: b
  real(8), intent(in) :: stage,bed(nc)
  real(8) :: rate,wet_area,top_width,conveyance
  call section_metrics(b,stage,bed,wet_area,top_width,conveyance)
  if(section_control(b)==1) then
   rate=sqrt(section_energy_slope(b))*conveyance
  else if(wet_area>0.and.top_width>0) then
   rate=sqrt(grav*wet_area**3/top_width)
  else
   rate=0
  endif
 end function

 function live_section_rate_nodes(b,stage,bed,nodes) result(rate)
  integer, intent(in) :: b
  real(8), intent(in) :: stage,bed(nc),nodes(:)
  real(8) :: rate,wet_area,top_width,conveyance
  call section_metrics_nodes(b,stage,bed,nodes,wet_area,top_width,conveyance)
  if(section_control(b)==1) then
   rate=sqrt(section_energy_slope(b))*conveyance
  else if(wet_area>0.and.top_width>0) then
   rate=sqrt(grav*wet_area**3/top_width)
  else
   rate=0
  endif
 end function

 subroutine live_section_stage(b,discharge,bed,stage,ok)
  ! Enumerate every terrain-elevation branch. Normal flow selects the lowest
  ! admissible root; critical flow selects the candidate with least specific
  ! energy. This mirrors the preparation calculator while using the live bed.
  integer, intent(in) :: b
  real(8), intent(in) :: discharge,bed(nc)
  real(8), intent(out) :: stage
  logical, intent(out) :: ok
  integer :: base,last,count,seed,k,j,nlevels,nroots,iteration,nknots,turn
  integer, allocatable :: root_valid(:)
  real(8) :: minimum,maximum,value,lo,hi,left_rate,right_rate,mid,residual,tol,energy,best_energy,area,width,conveyance
  real(8) :: quarter,lower_width,upper_width,top_derivative,qa,qb,qc,discriminant,stable,turns(2)
  real(8), allocatable :: nodes(:),levels(:),roots(:)
  ok=.false.;stage=0
  base=boundary_offset(b);last=boundary_offset(b+1)-1;count=last-base+1
  if(count<1.or.section_seed_node(b)<=0) return
  allocate(nodes(count+1))
  call section_nodes(b,bed,nodes)
  seed=minloc(nodes,dim=1)
  if(seed==1.or.seed==size(nodes)) then
   deallocate(nodes);return
  endif
  minimum=nodes(seed);maximum=min(nodes(1),nodes(size(nodes)))
  if(section_control(b)==1) then
   ! The sampled endpoints meet reflective boundary faces, so evolving bed
   ! elevations cannot turn them into open spill edges. Search the complete
   ! terrain first and, only if necessary, extend the vertical sidewalls until
   ! the requested normal-flow conveyance is bracketed. The former endpoint
   ! cap is still traversed first and therefore every previously admissible
   ! lowest root is numerically identical.
   maximum=maxval(nodes)
   iteration=0
   do while(live_section_rate_nodes(b,maximum,bed,nodes)<discharge.and.iteration<60)
    maximum=minimum+2.d0*(maximum-minimum);iteration=iteration+1
    if(.not.ieee_is_finite(maximum)) exit
   enddo
  endif
  if(maximum<=minimum) then
   deallocate(nodes);return
  endif
  if(discharge==0) then
   stage=minimum;ok=.true.;deallocate(nodes);return
  endif
  allocate(levels(3*count+6))
  nlevels=2;levels(1)=minimum;levels(2)=maximum
  do k=2,size(nodes)-1
   value=nodes(k)
   if(value<=minimum.or.value>=maximum) cycle
   nlevels=nlevels+1;levels(nlevels)=value
  enddo
  call sort_real_ascending(levels,nlevels)
  j=1
  do k=2,nlevels
   if(levels(k)==levels(j)) cycle
   j=j+1;levels(j)=levels(k)
  enddo
  nlevels=j
  if(section_control(b)==1) then
   ! Normal conveyance can turn in an irregular compound section, so a single
   ! global bisection is not generally safe. Search elevation branches in
   ! ascending order and stop as soon as the original lowest-root criterion is
   ! satisfied. Dry-start river inlets normally occupy only the deepest few
   ! branches; unlike the former all-branch parallel pass, work therefore scales
   ! with the actually wetted section. Parallelizing candidate branches is
   ! counterproductive here: only one branch owns the root and the others become
   ! speculative work. The large cell/face/fraction kernels remain OpenMP.
   do k=1,nlevels-1
     tol=max(1.d-12,128*epsilon(1.d0)*max(1.d0,abs(levels(k)),abs(levels(k+1))))
     lo=levels(k)+tol;hi=levels(k+1)-tol
     if(lo<hi) then
      left_rate=live_section_rate_nodes(b,lo,bed,nodes)-discharge
      right_rate=live_section_rate_nodes(b,hi,bed,nodes)-discharge
      if(left_rate*right_rate<=0) then
       do iteration=1,80
        mid=.5d0*(lo+hi);residual=live_section_rate_nodes(b,mid,bed,nodes)-discharge
        if(abs(residual)<=1.d-10*max(1.d0,discharge).or.hi-lo<=1.d-10) exit
        if((left_rate<=0.and.residual>=0).or.(left_rate>=0.and.residual<=0)) then
         hi=mid;right_rate=residual
        else
         lo=mid;left_rate=residual
        endif
       enddo
       stage=.5d0*(lo+hi);ok=.true.
       deallocate(nodes,levels);return
      endif
     endif
     value=levels(k+1)
     if(abs(live_section_rate_nodes(b,value,bed,nodes)-discharge)<=1.d-10*max(1.d0,discharge)) then
      stage=value;ok=.true.
      deallocate(nodes,levels);return
     endif
   enddo
   deallocate(nodes,levels);return
  endif
  allocate(roots(6*count+12),root_valid(3*count+6))
  if(section_control(b)==2) then
   ! All submerged portions contribute continuously to A. A newly wetted
   ! wide channel can still make Qcritical decrease and increase within one
   ! terrain interval. Split at its stationary rates before bracketing, as
   ! the Python calculator does; otherwise the least-energy root is missed.
   ! With delta=stage-mid: A=A0+T0*delta+Tprime*delta**2/2.
   ! d(Qcritical**2)/d(stage)=0 iff 3*T**2-A*Tprime=0.
   nknots=nlevels
   do k=1,nknots-1
    mid=.5d0*(levels(k)+levels(k+1));quarter=.25d0*(levels(k+1)-levels(k))
    if(quarter<=0.or.mid-quarter==mid+quarter) cycle
    call section_metrics_nodes(b,mid,bed,nodes,area,width,conveyance)
    call section_metrics_nodes(b,mid-quarter,bed,nodes,value,lower_width,conveyance)
    call section_metrics_nodes(b,mid+quarter,bed,nodes,value,upper_width,conveyance)
    top_derivative=(upper_width-lower_width)/(2*quarter)
    if(abs(top_derivative)<1.d-12*max(width,1.d0)/(4*quarter)) cycle
    qa=2.5d0*top_derivative**2;qb=5*width*top_derivative;qc=3*width**2-area*top_derivative
    value=max(abs(qa),abs(qb),abs(qc));qa=qa/value;qb=qb/value;qc=qc/value
    discriminant=qb*qb-4*qa*qc
    if(discriminant<0) cycle
    stable=-.5d0*(qb+sign(sqrt(discriminant),qb))
    if(stable==0) then
     turns=-qb/(2*qa)
    else
     turns=[stable/qa,qc/stable]
    endif
    do turn=1,2
     value=mid+turns(turn)
     if(.not.ieee_is_finite(value).or.value<=levels(k).or.value>=levels(k+1)) cycle
     if(any(levels(:nlevels)==value)) cycle
     nlevels=nlevels+1;levels(nlevels)=value
    enddo
   enddo
   call sort_real_ascending(levels,nlevels)
   j=1
   do k=2,nlevels
    if(levels(k)==levels(j)) cycle
    j=j+1;levels(j)=levels(k)
   enddo
   nlevels=j
  endif
  nroots=0;root_valid=0
  !$omp parallel do schedule(static) if(nlevels>=64) default(shared) &
  !$omp& private(k,tol,lo,hi,left_rate,right_rate,iteration,mid,residual)
  do k=1,nlevels-1
   tol=max(1.d-12,128*epsilon(1.d0)*max(1.d0,abs(levels(k)),abs(levels(k+1))))
   lo=levels(k)+tol;hi=levels(k+1)-tol
   if(lo>=hi) cycle
   left_rate=live_section_rate_nodes(b,lo,bed,nodes)-discharge
   right_rate=live_section_rate_nodes(b,hi,bed,nodes)-discharge
   if(section_control(b)==2.and.right_rate<left_rate) cycle
   if(left_rate*right_rate>0) cycle
   do iteration=1,80
    mid=.5d0*(lo+hi);residual=live_section_rate_nodes(b,mid,bed,nodes)-discharge
    if(abs(residual)<=1.d-10*max(1.d0,discharge).or.hi-lo<=1.d-10) exit
    if((left_rate<=0.and.residual>=0).or.(left_rate>=0.and.residual<=0)) then
     hi=mid;right_rate=residual
    else
     lo=mid;left_rate=residual
    endif
   enddo
   roots(k)=.5d0*(lo+hi);root_valid(k)=1
  enddo
  !$omp end parallel do
  do k=1,nlevels-1
   if(root_valid(k)==0) cycle
   nroots=nroots+1;roots(nroots)=roots(k)
  enddo
  do k=2,nlevels
   value=levels(k)
   if(value<=minimum) cycle
   if(abs(live_section_rate_nodes(b,value,bed,nodes)-discharge)<=1.d-10*max(1.d0,discharge)) then
    if(section_control(b)==2.and.value<maximum) then
     call section_metrics_nodes(b,value,bed,nodes,area,width,conveyance)
     ! Distinguish a true connectivity jump from the ordinary dA=T*deta
     ! increase on a continuous branch.  A fixed 1e-10 m probe made every
     ! smooth terrain knot look discontinuous; use the adjacent floating-point
     ! level, as the Python reference does.
     call section_metrics_nodes(b,min(maximum,nearest(value,1.d0)),bed,nodes,lo,hi,conveyance)
     if(lo-area>64*epsilon(1.d0)*max(1.d0,abs(area),abs(lo))) cycle
    endif
    nroots=nroots+1;roots(nroots)=value
   endif
  enddo
  if(nroots>0) then
   if(section_control(b)==1) then
    stage=minval(roots(:nroots))
   else
    best_energy=huge(1.d0);stage=roots(1)
    do k=1,nroots
     call section_metrics_nodes(b,roots(k),bed,nodes,area,width,conveyance)
     if(area<=0) cycle
     energy=roots(k)+discharge**2/(2*grav*area**2)
     if(energy<best_energy) then
      best_energy=energy;stage=roots(k)
     endif
    enddo
   endif
   ok=.true.
  endif
  deallocate(nodes,levels,roots,root_valid)
 end subroutine

 subroutine viscosity_equilibrated_mode6_profile(boundary,state,discharge,face_speed, &
   driving,discharge_residual,acceleration_residual,iterations,ok)
  integer,intent(in) :: boundary
  real(8),intent(in) :: state(nstate,nc),discharge
  real(8),intent(out) :: face_speed(nf),driving,discharge_residual,acceleration_residual
  integer,intent(out) :: iterations
  logical,intent(out) :: ok
  real(8),allocatable :: section_width(:),section_center(:),section_depth(:),section_roughness(:),section_speed(:)
  integer,allocatable :: section_component(:)
  real(8) :: tangent(2),work_depth
  integer :: first,last,count,k,f,c,status
  face_speed=0.d0;driving=0.d0;discharge_residual=0.d0
  acceleration_residual=0.d0;iterations=0;ok=.false.
  if(boundary<1.or.boundary>nb) return
  first=boundary_offset(boundary);last=boundary_offset(boundary+1)-1
  count=last-first+1
  if(count<1) return
  allocate(section_width(count),section_center(count),section_depth(count), &
   section_roughness(count),section_speed(count),section_component(count))
  f=boundary_faces(first);tangent=[-normal(2,f),normal(1,f)]
  if(tangent(1)<0.d0.or.(tangent(1)==0.d0.and.tangent(2)<0.d0)) tangent=-tangent
  do k=1,count
   f=boundary_faces(first+k-1);c=owner(f)
   section_width(k)=length(f);section_center(k)=dot_product(mid(:,f),tangent)
   section_depth(k)=state(1,c);section_roughness(k)=manning_cell(c)
  enddo
  work_depth=max(manning_friction_chezy_minimum_depth,roughness_minimum_work_depth)
  call braids_viscous_roughness_profile(count,section_width,section_center,section_depth, &
   section_roughness,hydraulic_roughness_mode,white_colebrook_chzk2d, &
   horizontal_viscosity,discharge,grav,dry_tolerance,work_depth, &
   section_speed,driving,section_component,iterations,discharge_residual, &
   acceleration_residual,status)
  if(status==BRAIDS_VISCOUS_INLET_SUCCESS) then
   do k=1,count
    face_speed(boundary_faces(first+k-1))=section_speed(k)
   enddo
   ok=.true.
  endif
  deallocate(section_width,section_center,section_depth,section_roughness,section_speed,section_component)
 end subroutine

 subroutine traces(state,bed,hydraulic,trace,ok,pv,pg,plo,phi)
  real(8), intent(in) :: state(nstate,nc),bed(nc),hydraulic(2,nb)
  real(8), intent(out) :: trace(3,nf)
  logical, intent(out) :: ok
  logical :: owner_incoming
  real(8) :: invariant(nf),target_u(nf),h,u,c,lo,hi,stage,total,targetarea,desired,ub,tangent(2),zavg,wet_length,weight
  real(8) :: incoming_left(3),incoming_right(3),incoming_speed,rightmost,tolerance
  real(8),optional,intent(in) :: pv(nprimitive,nc),pg(nprimitive,2,nc),plo(nprimitive,nc),phi(nprimitive,nc)
  real(8) :: edge(nprimitive),edge_state(3),edge_bed
  real(8) :: profile_driving,profile_discharge_residual,profile_acceleration_residual
  integer :: branch(nf),b,f,o,iter,k,first_wet,last_wet,profile_iterations
  logical :: profile_ok,gaia_fixed,face_valid
  ok=.false.;trace=0;branch=0;invariant=0;target_u=0;sampled_boundary_open=.true.
  failed_trace_boundary=0;failed_trace_face=0;failed_trace_condition=0;failed_trace_values=0
  if(present(pv)) then
   if(.not.present(pg).or..not.present(plo).or..not.present(phi)) error stop 'incomplete outlet reconstruction'
  endif
  do b=1,nb
   ok=.false. ! a preceding successful outlet must not license a failed inlet
   gaia_fixed=.false.
   if(bedlaw==9) gaia_fixed=fixed_boundary_bed_mode(b)==1
   if(mode(b)==4) then
    call critical_outlet(b,ok)
    if(.not.ok) return
    cycle
   else if(mode(b)==5) then
    call weir_outlet(b,ok)
    if(.not.ok) return
    cycle
   endif
   desired=hydraulic(1,b);targetarea=0
   if(mode(b)==6) then
    if(viscosity_equilibrated_inlet(b)) then
     call viscosity_equilibrated_mode6_profile(b,state,desired,target_u, &
      profile_driving,profile_discharge_residual,profile_acceleration_residual, &
      profile_iterations,profile_ok)
     if(.not.profile_ok) return
     do k=boundary_offset(b),boundary_offset(b+1)-1
      f=boundary_faces(k);o=owner(f);h=state(1,o);ub=target_u(f)
      if(h<=dry_tolerance.or.desired==0.d0) then
       trace(:,f)=0.d0
       if(h>dry_tolerance) then
        tangent=state(2:3,o)/h-dot_product(state(2:3,o),normal(:,f))/h*normal(:,f)
        trace(:,f)=[h,h*tangent]
       endif
       cycle
      endif
      tangent=state(2:3,o)/h-dot_product(state(2:3,o),normal(:,f))/h*normal(:,f)
      if(sqrt(ub**2+sum(tangent**2))>speed_limit) then
       failure_reason=6;failed_speed_boundary=b;failed_trace_speed=sqrt(ub**2+sum(tangent**2))
       failed_trace_boundary=b;failed_trace_face=f;failed_trace_condition=10
       failed_trace_values=[h,ub,sqrt(grav*h),desired,total,failed_trace_speed];return
      endif
      trace(:,f)=[h,h*(-ub*normal(:,f)+tangent)]
     enddo
     cycle
    endif
    ! Free-stage total-discharge boundary.  This follows the supplied FLOW
    ! source's T-boundary structure: a width-averaged adjacent water level
    ! supplies B*h^(3/2)*C weights (incbc.f90), the prescribed section Q is
     ! divided by each face width and depth (cucbp.f90), and the scalar water
     ! level has zero normal gradient. TAUBOT evaluates the Manning Chezy
     ! factor with its 0.01 m velocity-point depth floor; INcbc evaluates the
     ! section-depth factor from the current water line. Thus the weight is
     ! B*h_section^(3/2)*max(h_face,0.01)^(1/6)/n. The two physical depths
     ! coincide only for a laterally level surface. No target level is imposed.
    zavg=0;wet_length=0
    do k=boundary_offset(b),boundary_offset(b+1)-1
     f=boundary_faces(k);o=owner(f)
     if(state(1,o)<=dry_tolerance) cycle
     zavg=zavg+(bed(o)+state(1,o))*length(f);wet_length=wet_length+length(f)
    enddo
    if(desired>0.and.wet_length<=0) then
     failure_reason=8;failed_trace_boundary=b;failed_trace_condition=11
     failed_trace_values=[wet_length,desired,zavg,0.d0,0.d0,0.d0];return
    endif
    if(wet_length>0) zavg=zavg/wet_length
    total=0
    do k=boundary_offset(b),boundary_offset(b+1)-1
     f=boundary_faces(k);o=owner(f);target_u(f)=0
     if(gaia_fixed) then
     call gaia_fixed_inlet_face_terms(zavg,fixed_boundary_bed(f),state(1:3,o),normal(:,f), &
      manning_cell(o),manning_chezy_minimum_depth,dry_tolerance,h,tangent,weight,face_valid)
      if(face_valid.and.(hydraulic_roughness_mode/=braids_roughness_manning.or. &
         roughness_minimum_work_depth>0.d0)) &
       weight=roughness_face_weight(h,h,manning_cell(o))
      if(.not.face_valid) then
       failure_reason=8;failed_trace_boundary=b;failed_trace_face=f;failed_trace_condition=17
       failed_trace_values=[zavg,fixed_boundary_bed(f),state(1,o),manning_cell(o),0.d0,0.d0];return
      endif
     else
      h=state(1,o)
      if(h<=dry_tolerance) cycle
      if(hydraulic_roughness_mode==braids_roughness_manning.and. &
         roughness_minimum_work_depth==0.d0) then
       weight=max(zavg-bed(o),0.d0)**1.5d0* &
        max(h,manning_chezy_minimum_depth)**(1.d0/6.d0)
       if(manning_cell(o)>0) weight=weight/manning_cell(o)
      else
       weight=roughness_face_weight(max(zavg-bed(o),0.d0),h,manning_cell(o))
      endif
     endif
     if(h<=dry_tolerance) cycle
     target_u(f)=weight*length(f);total=total+target_u(f)
    enddo
    if(desired>0.and.total<=0) then
     failure_reason=8;failed_trace_boundary=b;failed_trace_condition=12
     failed_trace_values=[total,desired,zavg,wet_length,0.d0,0.d0];return
    endif
    do k=boundary_offset(b),boundary_offset(b+1)-1
     f=boundary_faces(k);o=owner(f)
     if(gaia_fixed) then
     call gaia_fixed_inlet_face_terms(zavg,fixed_boundary_bed(f),state(1:3,o),normal(:,f), &
      manning_cell(o),manning_chezy_minimum_depth,dry_tolerance,h,tangent,weight,face_valid)
      if(face_valid.and.(hydraulic_roughness_mode/=braids_roughness_manning.or. &
         roughness_minimum_work_depth>0.d0)) &
       weight=roughness_face_weight(h,h,manning_cell(o))
      if(face_valid) call gaia_fixed_inlet_trace(h,tangent,normal(:,f),weight,desired,total, &
       dry_tolerance,speed_limit,trace(:,f),face_valid)
      if(.not.face_valid) then
       ub=0.d0
       if(h>dry_tolerance.and.total>0.d0) ub=desired*weight/(total*h)
       failure_reason=6;failed_speed_boundary=b
       failed_trace_speed=sqrt(ub*ub+sum(tangent*tangent))
       failed_trace_boundary=b;failed_trace_face=f;failed_trace_condition=18
       failed_trace_values=[h,ub,sqrt(grav*h),desired,total,failed_trace_speed];return
      endif
      cycle
     endif
     h=state(1,o)
     if(h<=dry_tolerance.or.desired==0) then
      trace(:,f)=0
      if(h>dry_tolerance) then
       if(state(1,o)>dry_tolerance) then
        tangent=state(2:3,o)/state(1,o)-dot_product(state(2:3,o),normal(:,f))/state(1,o)*normal(:,f)
       else
        tangent=0
       endif
       trace(:,f)=[h,h*tangent]
      endif
      cycle
     endif
     ub=desired*target_u(f)/(total*h*length(f))
     if(state(1,o)>dry_tolerance) then
      tangent=state(2:3,o)/state(1,o)-dot_product(state(2:3,o),normal(:,f))/state(1,o)*normal(:,f)
     else
      tangent=0
     endif
     if(sqrt(ub**2+sum(tangent**2))>speed_limit) then
      failure_reason=6;failed_speed_boundary=b;failed_trace_speed=sqrt(ub**2+sum(tangent**2))
      failed_trace_boundary=b;failed_trace_face=f;failed_trace_condition=13
      failed_trace_values=[h,ub,sqrt(grav*h),desired,total,failed_trace_speed];return
     endif
     trace(:,f)=[h,h*(-ub*normal(:,f)+tangent)]
    enddo
    cycle
   endif
   if(mode(b)==7) then
    ! Diagnostic strong trace: prescribe both the horizontal section stage and
     ! total Q, while retaining the same Manning/Chezy face distribution as the
     ! free-stage total-discharge mode.  This is used to replay a saved external
    ! hydraulic trace and isolate morphology; it is not the characteristic BC.
    stage=hydraulic(2,b);total=0
    do k=boundary_offset(b),boundary_offset(b+1)-1
     f=boundary_faces(k);o=owner(f);h=max(stage-bed(o),0.d0);target_u(f)=0
     if(h<=dry_tolerance) cycle
     if(hydraulic_roughness_mode==braids_roughness_manning.and. &
        roughness_minimum_work_depth==0.d0) then
      weight=h**1.5d0*max(h,manning_chezy_minimum_depth)**(1.d0/6.d0)
      if(manning_cell(o)>0) weight=weight/manning_cell(o)
     else
      weight=roughness_face_weight(h,h,manning_cell(o))
     endif
     target_u(f)=weight*length(f);total=total+target_u(f)
    enddo
    if(desired>0.and.total<=0) return
    do k=boundary_offset(b),boundary_offset(b+1)-1
     f=boundary_faces(k);o=owner(f);h=max(stage-bed(o),0.d0)
     if(h<=dry_tolerance) then
      trace(:,f)=0;cycle
     endif
     ub=0
     if(desired>0) ub=desired*target_u(f)/(total*h*length(f))
     if(abs(ub)>speed_limit) return
     trace(:,f)=[h,-h*ub*normal(:,f)]
    enddo
    cycle
   endif
   if(mode(b)==8) then
    ! Diagnostic strong depth trace.  An evolving BRAIDS boundary cell need not
    ! have the same bed change as an external solver's fixed open-boundary
    ! point, so replay the saved hydraulic depth itself when exact h/u control
    ! is required.  This mode is intentionally limited to that diagnostic use.
    h=hydraulic(2,b);total=0
    if(desired>0.and.h<=dry_tolerance) return
    do k=boundary_offset(b),boundary_offset(b+1)-1
     f=boundary_faces(k);o=owner(f);target_u(f)=0
     if(h<=dry_tolerance) cycle
     if(hydraulic_roughness_mode==braids_roughness_manning.and. &
        roughness_minimum_work_depth==0.d0) then
      weight=h**1.5d0*max(h,manning_chezy_minimum_depth)**(1.d0/6.d0)
      if(manning_cell(o)>0) weight=weight/manning_cell(o)
     else
      weight=roughness_face_weight(h,h,manning_cell(o))
     endif
     target_u(f)=weight*length(f);total=total+target_u(f)
    enddo
    if(desired>0.and.total<=0) return
    do k=boundary_offset(b),boundary_offset(b+1)-1
     f=boundary_faces(k);o=owner(f)
     if(h<=dry_tolerance) then
      trace(:,f)=0;cycle
     endif
     ub=0
     if(desired>0) ub=desired*target_u(f)/(total*h*length(f))
     if(abs(ub)>speed_limit) return
     trace(:,f)=[h,-h*ub*normal(:,f)]
    enddo
    cycle
   endif
   if(mode(b)==9) then
    ! User-facing section controls prescribe both total Q and one horizontal
    ! water line. Normal and critical depth are resolved on the surveyed
    ! section before the run; depth and stage controls arrive in the same
    ! form. Retain that state as the interior evolves instead of changing the
    ! stage through an outgoing characteristic and eventually invalidating a
    ! face at wetting/drying transitions.
    stage=hydraulic(2,b)
    call sampled_section_span(b,stage,bed,first_wet,last_wet)
    total=0
    do k=first_wet,last_wet
     f=boundary_faces(k);o=owner(f)
     h=max(stage-bed(o)-boundary_bed_offset(f),0.d0)
     if(h>dry_tolerance) total=total+h*length(f)
    enddo
    if(desired>0.and.total<=0) then
     failure_reason=8;failed_trace_boundary=b;failed_trace_condition=2
     failed_trace_values=[total,desired,stage,0.d0,0.d0,0.d0];return
    endif
    ub=0
    if(desired>0) ub=desired/total
    do k=boundary_offset(b),boundary_offset(b+1)-1
     f=boundary_faces(k);o=owner(f)
     if(sampled_boundary_bed(f)==1.and.(k<first_wet.or.k>last_wet)) then
      sampled_boundary_open(f)=.false.
      h=0
      if(state(1,o)>dry_tolerance) h=max(state(1,o)-boundary_bed_offset(f),0.d0)
      trace(:,f)=[h,0.d0,0.d0]
      cycle
     endif
     h=max(stage-bed(o)-boundary_bed_offset(f),0.d0)
     if(h<=dry_tolerance) then
      trace(:,f)=0
      cycle
     endif
     if(abs(ub)>speed_limit) then
      failure_reason=6;failed_speed_boundary=b;failed_trace_speed=abs(ub)
      failed_trace_boundary=b;failed_trace_face=f;failed_trace_condition=5
      failed_trace_values=[h,ub,sqrt(grav*h),stage,state(1,o),0.d0];return
     endif
     trace(:,f)=[h,-h*ub*normal(:,f)]
    enddo
    cycle
   endif
   call sampled_section_span(b,hydraulic(2,b),bed,first_wet,last_wet)
   do k=boundary_offset(b),boundary_offset(b+1)-1
    f=boundary_faces(k)
    o=owner(f);edge_state=state(1:3,o);edge_bed=bed(o)
    if(mode(b)==2.and.present(pv).and.state(1,o)>dry_tolerance) then
     ! The stage and outgoing characteristic must refer to the same physical
     ! face as its bed source. This is a reconstruction, not a new tailwater
     ! condition. Do not change established flat-bed/dry-face behaviour.
     edge=face_value(o,f,pv,pg,plo,phi)
     if(abs(edge(bed_primitive_index())-bed(o))>64*epsilon(1.d0)*max(1.d0,abs(bed(o))).and. &
        abs(dot_product(state(2:3,o),normal(:,f)))<state(1,o)*sqrt(grav*state(1,o))) then
      edge_bed=edge(bed_primitive_index())
      edge_state(1)=max(edge(1)-edge_bed,0.d0)
      edge_state(2:3)=edge_state(1)*edge(2:3)
     endif
    endif
    h=edge_state(1);u=0
    if(h>dry_tolerance) u=-dot_product(edge_state(2:3),normal(:,f))/h
    ! Count the owner's incoming characteristics before a hydrostatic edge
    ! projection changes its depth. A thin supercritical stream over a bank
    ! does not acquire outgoing information because the sampled edge is lower.
    owner_incoming=h<=dry_tolerance.or.u>=sqrt(grav*h)*(1.d0-1.d-12)
    ! Evaluate the characteristic at the physical edge terrain, carrying the
    ! owner's bed increment. A dry owner supplies no hydrostatic column.
    if(sampled_boundary_bed(f)==1.and.h>dry_tolerance) h=max(h-boundary_bed_offset(f),0.d0)
    c=sqrt(grav*h);invariant(f)=u-2*c
    if(mode(b)==1) then
     if(k>=first_wet.and.k<=last_wet) &
      targetarea=targetarea+max(hydraulic(2,b)-bed(o)-boundary_bed_offset(f),0.d0)*length(f)
     branch(f)=1
     if(owner_incoming.or.h<=dry_tolerance.or.u>=c*(1.d0-1.d-12)) branch(f)=2
     if(h>dry_tolerance.and.u<=-c.and.k>=first_wet.and.k<=last_wet) then
      if(reversal_policy(b)==1) then
       failure_reason=8;failed_trace_boundary=b;failed_trace_face=f;failed_trace_condition=1
       failed_trace_values=[h,u,c,hydraulic(2,b),desired,0.d0];return
      endif
     endif
    else
     if(h<=dry_tolerance) then
      ! A numerically dry outlet face has no outgoing characteristic.
      ! Keep it inactive until interior water wets it, even if erosion has
      ! lowered its bed below the prescribed stage. A stage outlet does not
      ! supply incoming water to dry bank cells. Rejecting this local state
      ! instead pins morphology to stage-dry_tolerance and collapses dt.
      cycle
     endif
     if(u<=-c) then
      trace(:,f)=edge_state;cycle
     endif
     if(mode(b)==2.and.u>=c.and.reversal_policy(b)==1) then
      ! Extend the stage outlet's local no-inflow active set across the
      ! inward-critical transition. Both characteristics point into the
      ! domain here: a stage-only outlet cannot prescribe that inflow.
      ! Close this face with the existing hydrostatic no-flux trace, as for
      ! subcritical incoming traces below. Timestep halving cannot resolve
      ! an inadmissible boundary regime and must not pin the run at Fr=1.
      trace(:,f)=[edge_state(1),0.d0,0.d0];cycle
     endif
     if(mode(b)==3.or.u>=c) then
      if(reversal_policy(b)==1) then
       failure_reason=8;failed_trace_boundary=b;failed_trace_face=f;failed_trace_condition=14
       failed_trace_values=[h,u,c,hydraulic(2,b),edge_bed,invariant(f)];return
      endif
     endif
     h=max(hydraulic(2,b)-edge_bed,0.d0)
     if(mode(b)==2.and.(h<=dry_tolerance.or.invariant(f)<=-3*sqrt(grav*h))) then
      ! A dry or below-critical tailwater cannot supply a subcritical
      ! stage characteristic. This also covers continuous stage changes
      ! across the dry threshold without creating a new rejection there.
      ! Match the outgoing invariant at a critical free outfall instead:
      ! u_in=-sqrt(g*h), hence invariant=-3*sqrt(g*h). Supercritical
      ! owners have already been released unchanged above. If even this
      ! control is below the numerical wet depth, retain the owner's
      ! hydrostatic pressure with exactly zero water/sediment flux until
      ! the control becomes resolved. Halving dt cannot resolve it.
      h=(invariant(f)/3)**2/grav
      if(h<=dry_tolerance) then
       trace(:,f)=[edge_state(1),0.d0,0.d0];cycle
      endif
      ub=-sqrt(grav*h)
      tangent=edge_state(2:3)/edge_state(1)+u*normal(:,f)
      trace(:,f)=[h,h*(-ub*normal(:,f)+tangent)];cycle
     endif
     if(h<=dry_tolerance) then
      failure_reason=8;failed_trace_boundary=b;failed_trace_face=f;failed_trace_condition=15
      failed_trace_values=[h,u,c,hydraulic(2,b),edge_bed,invariant(f)];return
     endif
     ub=invariant(f)+2*sqrt(grav*h)
     if(mode(b)==2.and.ub>0.and.reversal_policy(b)==1) then
      ! The established policy-1 outlet rejects incoming transport locally,
      ! as the critical outlet's active set does. Newly wetted bank faces can
      ! initially predict inward flow from the prescribed stage. Close that
      ! face until an outward trace is admissible; other outlet faces continue
      ! exporting. Explicit v48 policy 0 deliberately retains this incoming
      ! trace so Thatcher--Harleman can represent a real return event.
      trace(:,f)=[edge_state(1),0.d0,0.d0];cycle
     endif
     if(ub>0.or.abs(ub)>=sqrt(grav*h)) then
      if(reversal_policy(b)==1) then ! explicit no-reversal subcritical outlet
       failure_reason=8;failed_trace_boundary=b;failed_trace_face=f;failed_trace_condition=16
       failed_trace_values=[h,ub,sqrt(grav*h),hydraulic(2,b),edge_bed,invariant(f)];return
      endif
     endif
     tangent=edge_state(2:3)/edge_state(1)+u*normal(:,f)
     trace(:,f)=[h,h*(-ub*normal(:,f)+tangent)]
    endif
   enddo
   if(mode(b)/=1) cycle
   if(desired>0.and.targetarea<=0) then
    failure_reason=8;failed_trace_boundary=b;failed_trace_condition=2
    failed_trace_values=[targetarea,desired,hydraulic(2,b),0.d0,0.d0,0.d0];return
   endif
   target_u=0
   if(targetarea>0) target_u=desired/targetarea
   ! A hydrostatic extrapolation below a nearly dry bank cell is not a
   ! resolved outgoing characteristic. Where the prescribed exterior jet is
   ! supercritical, ask the same augmented GeoClaw solver used in the interior
   ! whether its ENTIRE bed-step wave fan enters the domain. Only then may
   ! both exterior data control the trace. This is an admissibility check,
   ! not a velocity/depth clip or a change to the supplied section Q.
   do k=first_wet,last_wet
    f=boundary_faces(k);o=owner(f)
    if(sampled_boundary_bed(f)/=1.or.branch(f)/=1) cycle
    h=max(hydraulic(2,b)-bed(o)-boundary_bed_offset(f),0.d0)
    if(h<=dry_tolerance.or.target_u(f)<=sqrt(grav*h)) cycle
    call wave_flux(state(1:3,o),[h,-h*target_u(f)*normal(:,f)],bed(o), &
     bed(o)+boundary_bed_offset(f),normal(:,f),incoming_left,incoming_right,incoming_speed,rightmost)
    tolerance=64*epsilon(1.d0)*max(1.d0,incoming_speed)
    if(rightmost>tolerance) cycle
    if(abs(incoming_left(1)+h*target_u(f))>1.d-10*max(1.d0,h*target_u(f))) cycle
    branch(f)=2
   enddo
   lo=huge(1.d0);hi=-huge(1.d0)
   do k=boundary_offset(b),boundary_offset(b+1)-1
    f=boundary_faces(k)
    lo=min(lo,bed(owner(f))+boundary_bed_offset(f))
    hi=max(hi,bed(owner(f))+boundary_bed_offset(f)+state(1,owner(f))+1.d0,hydraulic(2,b))
   enddo
   do iter=1,60
    if(section_flux(hi,b)>=desired) exit
    hi=lo+2*(hi-lo)
   enddo
   if(iter>60) then
    failure_reason=8;failed_trace_boundary=b;failed_trace_condition=3
    failed_trace_values=[lo,hi,section_flux(hi,b),desired,0.d0,0.d0];return
   endif
   do iter=1,90
    stage=.5d0*(lo+hi)
    if(stage==lo.or.stage==hi) exit ! no representable refinement remains
    if(section_flux(stage,b)<desired) then
     lo=stage
    else
     hi=stage
    endif
   enddo
   stage=.5d0*(lo+hi)
   call sampled_section_span(b,stage,bed,first_wet,last_wet)
   do k=boundary_offset(b),boundary_offset(b+1)-1
    f=boundary_faces(k)
    if(sampled_boundary_bed(f)==1.and.(k<first_wet.or.k>last_wet)) then
     sampled_boundary_open(f)=.false.
     h=0
     if(state(1,owner(f))>dry_tolerance) h=max(state(1,owner(f))-boundary_bed_offset(f),0.d0)
     trace(:,f)=[h,0.d0,0.d0];cycle
    endif
   h=max(stage-bed(owner(f))-boundary_bed_offset(f),0.d0);ub=target_u(f)
   if(branch(f)==1) ub=invariant(f)+2*sqrt(grav*h)
   ! A face whose physical edge is dry contributes exactly zero area and zero
   ! discharge to section_flux.  Its characteristic velocity is undefined:
   ! retaining the neighbouring-cell invariant at h=0 can look like outlet
   ! reversal and no amount of timestep halving can change that geometric
   ! state.  Close only this zero-area face; the common stage root and all wet
   ! inlet faces retain the prescribed section discharge.
   ! Preserve every already-admissible dry trace exactly. Close the face only
   ! where the test below would otherwise reject an undefined velocity; this
   ! keeps established uniform trajectories bitwise unchanged.
   if(h<=dry_tolerance.and.(ub< -1.d-12.or.(branch(f)==1.and.ub>sqrt(grav*h)*(1.d0+1.d-12)))) then
    trace(:,f)=0;cycle
   endif
   ! Regime switches and reversal require explicit future policy, not silent clipping.
    if(ub< -1.d-12.or.(branch(f)==1.and.ub>sqrt(grav*h)*(1.d0+1.d-12))) then
     if(reversal_policy(b)==1) then
      failure_reason=8;failed_trace_boundary=b;failed_trace_face=f;failed_trace_condition=4
      failed_trace_values=[h,ub,sqrt(grav*h),stage,state(1,owner(f)),invariant(f)];return
     endif
    endif
    ! Apply the same operating limit as the hydraulic stage preparation and
    ! other inlet modes. Reject the trial; never clip a prescribed Q or stage.
    if(h>dry_tolerance.and.abs(ub)>speed_limit) then
     ok=.false.;failure_reason=6;failed_speed_boundary=b;failed_trace_speed=abs(ub)
     failed_trace_boundary=b;failed_trace_face=f;failed_trace_condition=5
     failed_trace_values=[h,ub,sqrt(grav*h),stage,state(1,owner(f)),invariant(f)];return
    endif
    trace(:,f)=[h,-h*ub*normal(:,f)]
   enddo
  enddo
  ok=.true.
 contains
  function section_flux(stage,b) result(total)
   real(8), intent(in) :: stage
   integer, intent(in) :: b
   real(8) :: total,hh,uu
   integer :: ff,kk,first,last
   total=0
   call sampled_section_span(b,stage,bed,first,last)
   do kk=first,last
    ff=boundary_faces(kk)
    hh=max(stage-bed(owner(ff))-boundary_bed_offset(ff),0.d0);uu=target_u(ff)
    if(branch(ff)==1) uu=invariant(ff)+2*sqrt(grav*hh)
   total=total+hh*uu*length(ff)
   enddo
  end function

  subroutine closed_outlet(bb)
   integer, intent(in) :: bb
   integer :: ff,kk,oo
   do kk=boundary_offset(bb),boundary_offset(bb+1)-1
    ff=boundary_faces(kk);oo=owner(ff);call close_outlet_face(ff,oo)
   enddo
  end subroutine

  subroutine close_outlet_face(ff,oo)
   integer, intent(in) :: ff,oo
   real(8) :: hh,un,ut(2)
   hh=state(1,oo)
   if(hh<=dry_tolerance) then
    trace(:,ff)=0
   else
    un=dot_product(state(2:3,oo),normal(:,ff))/hh
    ut=state(2:3,oo)/hh-un*normal(:,ff)
    trace(:,ff)=[hh,hh*ut]
   endif
  end subroutine

  subroutine critical_outlet(bb,valid)
   integer, intent(in) :: bb
   logical, intent(out) :: valid
   integer :: ff,kk,oo,it,active_iter
   real(8) :: hh,qn,aa,tt,qcritical,lower,upper,eta,ub,ut(2),tol,reslo,reshi,resmid,c
   logical :: changed
   valid=.false.;tol=128*epsilon(1.d0)
   do kk=boundary_offset(bb),boundary_offset(bb+1)-1
    ff=boundary_faces(kk);oo=owner(ff);hh=state(1,oo)
    if(hh<=dry_tolerance) cycle
    qn=dot_product(state(2:3,oo),normal(:,ff))
    if(qn < -tol*max(1.d0,abs(qn))) then
     call close_outlet_face(ff,oo);cycle
    endif
    c=sqrt(grav*hh);ub=-qn/hh
    if(ub < -c*(1.d0+1.d-10)) then
     trace(:,ff)=state(1:3,oo);branch(ff)=2;cycle
    endif ! supercritical outflow carries the interior state out unchanged
    branch(ff)=1
    invariant(ff)=ub-2*c
   enddo
   ! A common critical stage can predict local inflow over a high portion of
   ! an irregular section. Such faces cannot belong to an outlet. Remove them
   ! from the controlled opening and recompute the critical root over the
   ! remaining faces. Every repeated pass removes at least one face.
   do active_iter=1,boundary_offset(bb+1)-boundary_offset(bb)+1
    lower=huge(1.d0);upper=-huge(1.d0)
    do kk=boundary_offset(bb),boundary_offset(bb+1)-1
     ff=boundary_faces(kk);oo=owner(ff)
     if(branch(ff)/=1) cycle
     lower=min(lower,bed(oo));upper=max(upper,bed(oo)+state(1,oo))
    enddo
    if(lower==huge(1.d0)) then
     valid=.true.;return
    endif
    lower=lower+max(2*dry_tolerance,128*epsilon(1.d0)*max(1.d0,abs(lower)))
    reslo=critical_residual(bb,lower);reshi=critical_residual(bb,upper)
    if(reslo<0) then
     ! The characteristic critical depth is below the numerical dry depth.
     ! Keep the controlled portion closed until it is resolved; timestep
     ! halving cannot make an unresolved boundary trace admissible.
     do kk=boundary_offset(bb),boundary_offset(bb+1)-1
      ff=boundary_faces(kk)
      if(branch(ff)==1) then
       call close_outlet_face(ff,owner(ff));branch(ff)=0
      endif
     enddo
     valid=.true.;return
    endif
    do it=1,60
     if(reshi<=0) exit
     upper=lower+2*(upper-lower)
     reshi=critical_residual(bb,upper)
    enddo
    if(it>60) return
    do it=1,90
     eta=.5d0*(lower+upper);resmid=critical_residual(bb,eta)
     if(resmid>0) then
      lower=eta
     else
      upper=eta
     endif
    enddo
    eta=.5d0*(lower+upper);changed=.false.
    do kk=boundary_offset(bb),boundary_offset(bb+1)-1
     ff=boundary_faces(kk);oo=owner(ff)
     if(branch(ff)/=1) cycle
     hh=max(eta-bed(oo),0.d0)
     if(hh<=dry_tolerance) then
      call close_outlet_face(ff,oo);branch(ff)=0;changed=.true.;cycle
     endif
     ub=invariant(ff)+2*sqrt(grav*hh)
     if(ub>tol) then
      call close_outlet_face(ff,oo);branch(ff)=0;changed=.true.
     endif
    enddo
    if(changed) cycle
    call strip_geometry(bb,eta,aa,tt)
    if(aa<=0.or.tt<=0) return
    qcritical=sqrt(grav*aa**3/tt)
    if(abs(characteristic_outflow(bb,eta)-qcritical)>1.d-10*max(qcritical,1.d0)) return
    do kk=boundary_offset(bb),boundary_offset(bb+1)-1
     ff=boundary_faces(kk);oo=owner(ff);hh=max(eta-bed(oo),0.d0)
     if(branch(ff)/=1) cycle
     ub=invariant(ff)+2*sqrt(grav*hh)
     ut=0
     if(state(1,oo)>dry_tolerance) then
      qn=dot_product(state(2:3,oo),normal(:,ff))/state(1,oo)
      ut=state(2:3,oo)/state(1,oo)-qn*normal(:,ff)
     endif
     if(sqrt(ub**2+sum(ut**2))>speed_limit) then
      failure_reason=7;failed_speed_boundary=bb
      failed_trace_speed=sqrt(ub**2+sum(ut**2));return
     endif
     trace(:,ff)=[hh,hh*(-ub*normal(:,ff)+ut)]
    enddo
    valid=.true.;return
   enddo
   ! The loop bound is one larger than the number of faces, so reaching here
   ! indicates an implementation error rather than a hydraulic regime change.
   failure_reason=8;failed_trace_boundary=bb;failed_trace_condition=9
  end subroutine

  subroutine weir_outlet(bb,valid)
   integer, intent(in) :: bb
   logical, intent(out) :: valid
   integer :: ff,kk,oo
   real(8) :: hh,qn,eta,wet_length,head,requested,aa,tt,ub,ut(2),tol,c,lower,upper,reslo,reshi,resmid
   integer :: it
   valid=.false.;eta=0;wet_length=0;tol=128*epsilon(1.d0);lower=huge(1.d0);upper=-huge(1.d0)
   do kk=boundary_offset(bb),boundary_offset(bb+1)-1
    ff=boundary_faces(kk);oo=owner(ff);hh=state(1,oo)
    if(hh<=dry_tolerance) cycle
    branch(ff)=1
    qn=dot_product(state(2:3,oo),normal(:,ff))
    if(qn < -tol*max(1.d0,abs(qn))) return
    c=sqrt(grav*hh);ub=-qn/hh
    if(ub < -c*(1.d0+1.d-10)) return ! an undrowned weir is a subcritical upstream control
    invariant(ff)=ub-2*c
    eta=eta+(bed(oo)+hh)*length(ff);wet_length=wet_length+length(ff)
    lower=min(lower,bed(oo));upper=max(upper,bed(oo)+hh)
   enddo
   if(wet_length<=0) then
    call closed_outlet(bb);valid=.true.;return
   endif
   eta=eta/wet_length
   if(eta<=forcing(2,bb)+dry_tolerance) then
    call closed_outlet(bb);valid=.true.;return
   endif
   lower=max(lower,forcing(2,bb));reslo=weir_residual(bb,lower);reshi=weir_residual(bb,upper)
   if(reslo<0) return
   do it=1,60
    if(reshi<=0) exit
    upper=lower+2*(upper-lower);reshi=weir_residual(bb,upper)
   enddo
   if(it>60) return
   do it=1,90
    eta=.5d0*(lower+upper);resmid=weir_residual(bb,eta)
    if(resmid>0) then
     lower=eta
    else
     upper=eta
    endif
   enddo
   eta=.5d0*(lower+upper);head=max(eta-forcing(2,bb),0.d0)
   requested=forcing(5,bb)*forcing(6,bb)*head**1.5d0
   call strip_geometry(bb,eta,aa,tt)
   if(aa<=0.or.requested<=0) return
   if(abs(characteristic_outflow(bb,eta)-requested)>1.d-10*max(requested,1.d0)) return
   do kk=boundary_offset(bb),boundary_offset(bb+1)-1
    ff=boundary_faces(kk);oo=owner(ff);hh=max(eta-bed(oo),0.d0)
    if(branch(ff)==0) then
     trace(:,ff)=0;cycle
    endif
    if(hh<=dry_tolerance) then
     trace(:,ff)=0;cycle
    endif
    ub=invariant(ff)+2*sqrt(grav*hh)
    if(ub>tol) return
    ut=0
    if(state(1,oo)>dry_tolerance) then
     qn=dot_product(state(2:3,oo),normal(:,ff))/state(1,oo)
     ut=state(2:3,oo)/state(1,oo)-qn*normal(:,ff)
    endif
    if(sqrt(ub**2+sum(ut**2))>speed_limit) then
     failure_reason=7;failed_speed_boundary=bb
     failed_trace_speed=sqrt(ub**2+sum(ut**2));return
    endif
    trace(:,ff)=[hh,hh*(-ub*normal(:,ff)+ut)]
   enddo
   valid=.true.
  end subroutine

  function characteristic_outflow(bb,eta) result(total)
   integer, intent(in) :: bb
   real(8), intent(in) :: eta
   real(8) :: total,hh,ub
   integer :: ff,kk
   total=0
   do kk=boundary_offset(bb),boundary_offset(bb+1)-1
    ff=boundary_faces(kk)
    if(branch(ff)/=1) cycle
    hh=max(eta-bed(owner(ff)),0.d0)
    if(hh<=dry_tolerance) cycle
    ub=invariant(ff)+2*sqrt(grav*hh)
    total=total-hh*ub*length(ff)
   enddo
  end function

  function critical_residual(bb,eta) result(value)
   integer, intent(in) :: bb
   real(8), intent(in) :: eta
   real(8) :: value,aa,tt,qcritical
   call strip_geometry(bb,eta,aa,tt);qcritical=0
   if(tt>0) qcritical=sqrt(grav*aa**3/tt)
   value=characteristic_outflow(bb,eta)-qcritical
  end function

  function weir_residual(bb,eta) result(value)
   integer, intent(in) :: bb
   real(8), intent(in) :: eta
   real(8) :: value,head
   head=max(eta-forcing(2,bb),0.d0)
   value=characteristic_outflow(bb,eta)-forcing(5,bb)*forcing(6,bb)*head**1.5d0
  end function

  subroutine strip_geometry(bb,eta,aa,tt)
   integer, intent(in) :: bb
   real(8), intent(in) :: eta
   real(8), intent(out) :: aa,tt
   integer :: ff,kk
   real(8) :: hh
   aa=0;tt=0
   do kk=boundary_offset(bb),boundary_offset(bb+1)-1
    ff=boundary_faces(kk)
    if(branch(ff)/=1) cycle
    hh=max(eta-bed(owner(ff)),0.d0)
    if(hh<=dry_tolerance) cycle
    aa=aa+hh*length(ff);tt=tt+length(ff)
   enddo
  end subroutine
 end subroutine

 pure subroutine scalar_extremum_bound(values,kind,bound,flat_relative_tolerance)
  ! Curvature-checked finite-volume parabola: neighbouring CELL means are
  ! not pointwise extrema. Strict mean bounds repeatedly flatten a translating
  ! smooth peak. Adapt the geometric bound idea of Chaplin & Colella (2015),
  ! section 4.2.1, while retaining our stagewise positivity-preserving FCT.
  ! This is not their single-stage RK4/CTU algorithm. No bound is relaxed at
  ! a monotone front or inconsistent/underresolved curvature.
  real(8), intent(in) :: values(5)
  integer, intent(out) :: kind ! 0: retain old bound, 1: constant, 2: maximum, 3: minimum
  real(8), intent(out) :: bound
  real(8),optional,intent(in) :: flat_relative_tolerance
  real(8) :: z(5),scale,d2(3),slope,vertex,point,tol
  kind=0;bound=values(3);scale=maxval(abs(values))
  if(scale==0) then
   kind=1;return
  endif
  ! A scale-relative arithmetic band prevents accumulated RK/FCT roundoff
  ! in a constant transverse direction from toggling the extremum sensor.
  ! This is below 1e-12 of the local concentration, not an absolute C floor.
  z=values/scale;tol=4096*epsilon(1.d0)
  if(present(flat_relative_tolerance)) tol=max(tol,flat_relative_tolerance)
  if(maxval(z)-minval(z)<=tol) then
   kind=1;return
  endif
  if((z(3)-z(2))*(z(4)-z(3))>0) return
  d2=z(1:3)-2*z(2:4)+z(3:5)
  if(minval(abs(d2))<=tol) return
  if(.not.(all(d2>0).or.all(d2<0))) return
  if(minval(abs(d2))<.5d0*maxval(abs(d2))) return
  slope=.5d0*(z(4)-z(2))
  vertex=max(-.5d0,min(.5d0,-slope/d2(2)))
  point=z(3)+slope*vertex+.5d0*d2(2)*vertex**2-d2(2)/24.d0
  if(d2(2)<0) then
   kind=2;bound=scale*(z(3)+2*max(point-z(3),0.d0))
  else
   kind=3;bound=max(0.d0,scale*(z(3)-2*max(z(3)-point,0.d0)))
  endif
 end subroutine

 subroutine smooth_concentration_bounds(c,v,scalar,lower,upper)
  ! Only certified Cartesian rectangular wet stencils use this extension.
  ! Nonrectangular cuts, wet/dry stencils and open boundary cells retain
  ! their existing bounds. Each axis must be smooth-extremal or constant;
  ! saddles and unresolved curvature also retain the old limiter.
  integer, intent(in) :: c,scalar
  real(8), intent(in) :: v(nprimitive,nc)
  real(8), intent(inout) :: lower,upper
  integer :: a,k,f,n,side,ids(5),kind,extkind
  real(8) :: candidate,lo,hi,scale,tol,flat_relative_tolerance
  if(scheme<5.or.hydro_constant(c).or.any(rectangle_width(:,c)<=0)) return
  lo=lower;hi=upper;extkind=1
  scale=max(abs(lower),abs(upper),tiny(1.d0));tol=4096*epsilon(1.d0)*scale
  ! Coarse/fine remapping introduces harmless transverse arithmetic noise.
  ! Without a scale-relative flat-direction band, that noise can toggle the
  ! smooth-extremum limiter and repeatedly flatten a genuinely 1-D pulse.
  ! This is a concentration-bound sensor only: fluxes, positivity and all
  ! hydraulic reconstructions are unchanged. No absolute concentration floor.
  flat_relative_tolerance=4096*epsilon(1.d0)
  if(amr_enabled) flat_relative_tolerance=sqrt(epsilon(1.d0))
  tol=flat_relative_tolerance*scale
  do k=offset(c),offset(c+1)-1
   if(bid(faces(k))>0) return
  enddo
  do a=1,2
   ids=0
   do k=offset(c),offset(c+1)-1
    f=faces(k)
    if(abs(normal(a,f))/=1) cycle
    side=1
    if(c/=owner(f)) side=2
    if(weno_cells(1,side,f)>0) ids=weno_cells(:,side,f)
   enddo
   if(ids(1)==0) then
    ! A transverse constant at a default wall needs no extrapolated extremum.
    do k=offset(c),offset(c+1)-1
     f=faces(k)
     if(abs(normal(a,f))/=1) cycle
     n=neighbour(f)
     if(c/=owner(f)) n=owner(f)
     if(n<=0) cycle
     if(hydro_constant(n).or.abs(v(scalar,n)-v(scalar,c))>tol) return
    enddo
    cycle
   endif
   if(any(hydro_constant(ids))) return
   call scalar_extremum_bound(v(scalar,ids),kind,candidate,flat_relative_tolerance)
   if(kind==0) return
   if(kind==1) cycle
   if(extkind/=1.and.kind/=extkind) return
   extkind=kind
   if(kind==2) hi=max(hi,candidate)
   if(kind==3) lo=min(lo,candidate)
  enddo
  lower=max(0.d0,lo);upper=hi
 end subroutine

 subroutine euler(state,dt,evaluation_time,step_morfac,next,accepted,ok,integral,momentum_boundary_impulse,toe_flux_weight, &
                  instantaneous_face_flux,boundary_face_transport_integral)
  real(8), intent(in) :: state(nstate,nc),dt,evaluation_time,step_morfac
  real(8), intent(out) :: next(nstate,nc),accepted(nledger,nb)
  logical, intent(out) :: ok
  real(8),optional,intent(out) :: integral(nstate,2,nf)
  real(8),optional,intent(out) :: momentum_boundary_impulse(2,2)
  real(8),optional,intent(in) :: toe_flux_weight
  real(8),optional,intent(out) :: instantaneous_face_flux(nstate,nf)
  real(8),optional,intent(out) :: boundary_face_transport_integral(1+nfrac,nf)
  ! Reusable stage workspace: no recursive/concurrent Euler calls are permitted.
  ! AMR shape changes invalidate all buffers before any stage reads them.
  real(8),allocatable,save :: v(:,:)
  real(8),allocatable,save :: bed(:)
  real(8),allocatable,save :: grad(:,:,:)
  real(8),allocatable,save :: downslope(:,:)
  real(8),allocatable,save :: vlo(:,:)
  real(8),allocatable,save :: vhi(:,:)
  real(8),allocatable,save :: trace(:,:)
  real(8),allocatable,save :: bc_concentration(:,:)
  real(8),allocatable,save :: bc_bed_input(:,:)
  real(8),allocatable,save :: bc_hydraulic(:,:)
  real(8),allocatable,save :: bc_face_concentration(:,:)
  real(8),allocatable,save :: fl(:,:)
  real(8),allocatable,save :: fr(:,:)
  real(8),allocatable,save :: speed(:)
  real(8),allocatable,save :: loflux(:,:)
  real(8),allocatable,save :: anti(:,:)
  real(8),allocatable,save :: cl(:,:)
  real(8),allocatable,save :: cr(:,:)
  real(8),allocatable,save :: theta(:,:)
  real(8),allocatable,save :: rp(:,:)
  real(8),allocatable,save :: rm(:,:)
  real(8),allocatable,save :: lowstate(:,:)
  real(8),allocatable,save :: suspended_diffusive_flux(:,:)
  real(8),allocatable,save :: suspended_diffusion_theta(:,:)
  real(8),allocatable,save :: hydraulic_face_exchange_area(:)
  real(8),allocatable,save :: water_theta(:)
  real(8),allocatable,save :: hiding(:,:)
  real(8),allocatable,save :: mixture_dg(:)
  real(8),allocatable,save :: mixture_gsd(:)
  real(8),allocatable,save :: mixture_sand(:)
  real(8),allocatable,save :: mixture_d90(:)
  real(8),allocatable,save :: mixture_mud(:)
  integer,save :: workspace_shape(6)=[-1,-1,-1,-1,-1,-1]
  real(8) :: left(nprimitive),right(nprimitive),l(3),r(3),sumflux(nstate)
  real(8) :: pp,pm,z,scale,low,lower,upper,depth,cfl,cfl_ratio,stage_cfl_peak,correction,velocity(2),depth_tolerance,morfac
  real(8) :: boundary_stage_bed,shared_face_depth,owner_face_distance,neighbour_face_distance
  real(8) :: diffusion_distance,diffusion_left,diffusion_right,diffusion_rate
  real(8) :: secondary_equilibrium,secondary_high,stress_left(2,2),stress_right(2,2),stress_face(2,2)
  real(8) :: stress_flux(2),stress_signal_left,stress_signal_right,effective_momentum_flux(2)
  real(8) :: profile_phase_start,profile_setup_start,profile_detail_start
  real(8) :: blx,bly,brx,bry,qbx,qby,sblx,sbly,trl,trr,tratio,slx,sly,qsleft,qsright,outgoing
  real(8) :: boundary_absolute_shields,boundary_critical_shields,boundary_transport_stage
  real(8) :: boundary_spiral_intensity,boundary_spiral_alpha,boundary_formula_n
  real(8) :: raw_boundary_bed(2),raw_boundary_suspended(2), &
   composed_boundary_bed(2),retained_boundary_suspended(2), &
   boundary_signed_suspended_rate,boundary_equilibrium_concentration
  integer :: f,c,n,k,b,g,donor,stage_failure,water_failure,secondary_invalid,si,bi,ci,work,nwork,target,ntarget,toe_flux_error,composition_status
  logical :: valid,parallel_faces,suspended_active,hydraulic_closed,secondary_active,stress_active_left,stress_active_right
  logical :: fixed_bed_face
  if(any(workspace_shape/=[nprimitive,nstate,nc,nf,nfrac,nb])) then
   if(allocated(v)) then
    deallocate(v, &
     bed, &
     grad, &
     downslope, &
     vlo, &
     vhi, &
     trace, &
     bc_concentration, &
     bc_bed_input, &
     bc_hydraulic, &
     bc_face_concentration, &
     fl, &
     fr, &
     speed, &
     loflux, &
     anti, &
     cl, &
     cr, &
     theta, &
     rp, &
     rm, &
     lowstate, &
     suspended_diffusive_flux, &
     suspended_diffusion_theta, &
     hydraulic_face_exchange_area, &
     water_theta, &
     hiding, &
     mixture_dg, &
     mixture_gsd, &
     mixture_sand, &
     mixture_d90, &
     mixture_mud)
   endif
   allocate(v(nprimitive,nc), &
    bed(nc), &
    grad(nprimitive,2,nc), &
    downslope(2,nc), &
    vlo(nprimitive,nc), &
    vhi(nprimitive,nc), &
    trace(3,nf), &
    bc_concentration(nfrac,nb), &
    bc_bed_input(nfrac,nb), &
    bc_hydraulic(2,nb), &
    bc_face_concentration(nfrac,nf), &
    fl(nstate,nf), &
    fr(nstate,nf), &
    speed(nf), &
    loflux(nfrac,nf), &
    anti(nfrac,nf), &
    cl(nfrac,nf), &
    cr(nfrac,nf), &
    theta(nfrac,nc), &
    rp(nfrac,nc), &
    rm(nfrac,nc), &
    lowstate(nfrac,nc), &
    suspended_diffusive_flux(nfrac,nf), &
    suspended_diffusion_theta(nfrac,nc), &
    hydraulic_face_exchange_area(nf), &
    water_theta(nc), &
    hiding(nfrac,nc), &
    mixture_dg(nc), &
     mixture_gsd(nc), &
     mixture_sand(nc), &
     mixture_d90(nc), &
     mixture_mud(nc))
   workspace_shape=[nprimitive,nstate,nc,nf,nfrac,nb]
  endif
  ok=.false.;accepted=0;next=state;nwork=nf;ntarget=nc
  if(present(boundary_face_transport_integral)) &
   boundary_face_transport_integral=0.d0
  if(present(momentum_boundary_impulse)) momentum_boundary_impulse=0.d0
  if(profile_enabled) then
   profile_phase_start=omp_get_wtime();profile_setup_start=profile_phase_start
  endif
  if(amr_step_level>=0) nwork=amr_level_face_count(amr_step_level)
  if(amr_step_level>=0) ntarget=amr_level_cell_count(amr_step_level)
  ! Face work scales with the number of independently transported classes.
  ! A 4k-cell, seven-fraction flume is substantially heavier than the
  ! single-material crossover used by the old cell-count-only threshold.
  parallel_faces=nwork*nfrac>=8192.or.(amr_step_level>=0.and.nc>=4096.and.nwork>=1.and. &
    nfrac==1.and.nstate==5.and.bedlaw==0.and.exchange_mode==0.and. &
    .not.secondary_flow.and..not.bank_front_tracking.and..not.bank_failure.and..not.bank_avalanche.and. &
    .not.general_avalanche.and.bank_erosion_fraction==0.d0.and..not.bedform_roughness.and..not.bedload_adaptation)
  ! Recompute each RK stage: all wet cells, their dry neighbours, and every
  ! open-boundary owner participate. Other cells have only zero-transfer faces.
  !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) private(c,k,f,n)
  do c=1,nc
   active_cell(c)=state(1,c)>dry_tolerance
   if(active_cell(c)) cycle
   do k=offset(c),offset(c+1)-1
    f=faces(k);n=neighbour(f)
    if(signs(k)<0) n=owner(f)
    if(bid(f)>0) then
     active_cell(c)=.true.;exit
    endif
    if(n>0) then
     if(state(1,n)>dry_tolerance) then
      active_cell(c)=.true.;exit
     endif
    endif
   enddo
  enddo
  !$omp end parallel do
  call boundary_sediment_at(evaluation_time,bc_concentration,bc_bed_input)
  ! In a bedload-only calculation with zero suspended inventory and zero
  ! suspended boundary forcing, every suspended equation has the exact zero
  ! solution.  Avoid reconstructing and flux-limiting those seven (or more)
  ! identically-zero fields.  Re-enable the complete path immediately if an
  ! exchange law, boundary concentration, or carried inventory can make any
  ! suspended class nonzero.
  suspended_active=suspended_transport_active
  call primitive(state,v,bed,suspended_active)
  call boundary_hydraulic_at(evaluation_time,bed,bc_hydraulic,valid)
  if(.not.valid) return
  do b=1,nb
   if((mode(b)==1.or.mode(b)==6.or.mode(b)==7.or.mode(b)==8.or.mode(b)==9).and. &
      any(bc_bed_input(:,b)>0).and.bc_hydraulic(1,b)<=0) then
    failure_reason=4;return
   endif
  enddo
  if(profile_enabled) then
   profile_euler_boundary_s=profile_euler_boundary_s+omp_get_wtime()-profile_phase_start
   profile_phase_start=omp_get_wtime()
  endif
  if(classic_secondary_foundation()) &
   call refresh_classic_secondary_equilibrium(state)
  if(bedlaw>0) then
   if(profile_enabled) profile_detail_start=omp_get_wtime()
   call hiding_factors(state,hiding)
   if(profile_enabled) then
    profile_euler_hiding_s=profile_euler_hiding_s+omp_get_wtime()-profile_detail_start
    profile_detail_start=omp_get_wtime()
   endif
   call mixture_statistics(state,mixture_dg,mixture_gsd,mixture_sand,mixture_d90,mixture_mud)
   if(profile_enabled) profile_euler_mixture_s=profile_euler_mixture_s+omp_get_wtime()-profile_detail_start
  else
   ! No horizontal bed flux consumes these quantities in a hydraulic or
   ! suspended-only step. Avoid repeated sorting/statistics at every substep.
   hiding=1;mixture_dg=0;mixture_gsd=1;mixture_sand=0;mixture_d90=0;mixture_mud=0
  endif
  if(bedlaw>0) call prepare_bedload_capacity(state,hiding,mixture_dg,mixture_gsd,mixture_sand,mixture_d90,mixture_mud, &
   bc_bed_input,bedq_cell,bedtau_cell,bedq_grad)
  if(profile_enabled) then
   profile_euler_bedload_s=profile_euler_bedload_s+omp_get_wtime()-profile_phase_start
   profile_phase_start=omp_get_wtime()
  endif
  if(scheme==6.and.nc>=4096.and..not.amr_enabled.and. &
     .not.bank_front_tracking.and..not.bank_failure.and..not.bank_avalanche.and. &
     .not.general_avalanche.and.bank_erosion_fraction==0.d0.and. &
     .not.bedform_roughness.and..not.bedload_adaptation.and.exchange_mode==0.and. &
     (bedlaw==0.or.bedlaw==2)) then
   downslope=0
   if(bed_slope/=0) call bed_downslopes(bed,downslope)
   call prepare_hydraulic_traces_fused(state,v,bc_concentration,grad,vlo,vhi)
   if(profile_enabled) profile_hydraulic_workflow_calls=profile_hydraulic_workflow_calls+1
  else if(amr_enabled.and.scheme==6.and.nc>=4096.and.nfrac==1.and.nstate==5.and. &
     bedlaw==0.and.exchange_mode==0.and..not.secondary_flow.and. &
     .not.bank_front_tracking.and..not.bank_failure.and..not.bank_avalanche.and. &
     .not.general_avalanche.and.bank_erosion_fraction==0.d0.and. &
     .not.bedform_roughness.and..not.bedload_adaptation) then
   downslope=0
   if(bed_slope/=0) call bed_downslopes(bed,downslope)
   call prepare_adaptive_traces_fused(state,v,bc_concentration,grad,vlo,vhi)
   if(profile_enabled) profile_hydraulic_workflow_calls=profile_hydraulic_workflow_calls+1
  else
   call gradients(v,bc_concentration,grad,vlo,vhi)
   downslope=0
   if(bed_slope/=0) call bed_downslopes(bed,downslope)
   if(scheme==6) call prepare_wall_traces(state,v,vlo,vhi)
   call prepare_wet_dry_traces(state,v,grad,vlo,vhi)
  endif
  if(profile_enabled) then
   profile_euler_reconstruction_s=profile_euler_reconstruction_s+omp_get_wtime()-profile_phase_start
   profile_phase_start=omp_get_wtime()
  endif
  failure_reason=0
  call traces(state,bed,bc_hydraulic,trace,valid,v,grad,vlo,vhi)
  if(.not.valid) then
   if(failure_reason==0) failure_reason=1
   return
  endif
  if(profile_enabled) then
   profile_euler_trace_s=profile_euler_trace_s+omp_get_wtime()-profile_phase_start
   profile_euler_setup_s=profile_euler_setup_s+omp_get_wtime()-profile_setup_start
   profile_phase_start=omp_get_wtime()
  endif
  fl=0;fr=0;speed=0
  bc_face_concentration=0.d0
  suspended_diffusive_flux=0.d0
  hydraulic_face_exchange_area=0.d0
  if(capacity_split_retained) then
   ! A fresh ledger is assembled from this exact Euler evaluation.  Boundary
   ! reversal therefore cannot leave a stale constitutive vector from an
   ! earlier RK stage or output time.
   split_boundary_role=0
   split_boundary_bed=0.d0;split_boundary_suspended=0.d0
   split_boundary_absolute_shields=0.d0;split_boundary_critical_shields=0.d0
   split_boundary_stress_ratio=0.d0;split_boundary_transport_stage=0.d0
  endif
  if(channel_composition_declaration) then
   channel_raw_bed_boundary=0.d0;channel_raw_suspended_boundary=0.d0
   channel_composed_bed_boundary=0.d0;channel_retained_suspended_boundary=0.d0
   channel_signed_suspended_rate_boundary=0.d0
   channel_equilibrium_concentration_boundary=0.d0
  endif
  if(suspended_active) then
   cl=0;cr=0
  endif
  secondary_invalid=0
  if(prognostic_secondary_flow()) then
   secondary_face_left=0.d0;secondary_face_right=0.d0;secondary_boundary_intensity=0.d0
   secondary_low_flux=0.d0;secondary_anti_flux=0.d0;secondary_stress_face=0.d0
  endif
  !$omp parallel do schedule(static,256) if(parallel_faces) default(shared) &
  !$omp& private(work,f,c,n,b,g,si,bi,ci,donor,left,right,l,r,depth,blx,bly,brx,bry,qbx,qby,sblx,sbly,trl,trr,tratio,slx,sly,valid,qsleft,qsright,boundary_stage_bed,shared_face_depth,owner_face_distance,neighbour_face_distance,hydraulic_closed,fixed_bed_face,stress_left,stress_right,stress_face,stress_flux,stress_signal_left,stress_signal_right,stress_active_left,stress_active_right,boundary_absolute_shields,boundary_critical_shields,boundary_transport_stage,boundary_spiral_intensity,boundary_spiral_alpha,boundary_formula_n,raw_boundary_bed,raw_boundary_suspended,composed_boundary_bed,retained_boundary_suspended,boundary_signed_suspended_rate,boundary_equilibrium_concentration,composition_status) reduction(max:secondary_invalid)
  do work=1,nwork
   f=work
   if(amr_step_level>=0) f=amr_level_faces(work,amr_step_level)
   c=owner(f);n=neighbour(f);b=bid(f);hydraulic_closed=.false.;fixed_bed_face=.false.
   if(bedlaw==9) then
    if(b>0) fixed_bed_face=fixed_boundary_bed_mode(b)==1
   endif
   if(b==0.and.state(1,c)<=dry_tolerance) then
    if(n==0) cycle
    if(state(1,n)<=dry_tolerance) cycle
   endif
   left=face_value(c,f,v,grad,vlo,vhi);depth=max(left(1)-left(bed_primitive_index()),0.d0)
   if(prognostic_secondary_flow()) secondary_face_left(f)=left(secondary_primitive_index())
   if(state(1,c)<=dry_tolerance) depth=0
   if(hydro_constant(c).and.depth<=dry_tolerance) depth=0
   l=[depth,depth*left(2:3)]
   if(n>0) then
    right=face_value(n,f,v,grad,vlo,vhi);depth=max(right(1)-right(bed_primitive_index()),0.d0)
    if(prognostic_secondary_flow()) secondary_face_right(f)=right(secondary_primitive_index())
    if(state(1,n)<=dry_tolerance) depth=0
    if(hydro_constant(n).and.depth<=dry_tolerance) depth=0
    r=[depth,depth*right(2:3)]
    ! A dry internal face opens from the water column shared by its two
    ! centre stages, not merely because one reconstructed trace overtops the
    ! higher bed by an arbitrarily small amount.  This is the finite-volume
    ! analogue of the staggered wet/dry face aperture used by Delft3D.  Once
    ! both centres are wet the ordinary Riemann flux is recovered exactly.
    if(min(state(1,c),state(1,n))<=dry_tolerance) then
     owner_face_distance=abs(dot_product(mid(:,f)-xy(:,c),normal(:,f)))
     neighbour_face_distance=abs(dot_product(xy(:,n)-mid(:,f),normal(:,f)))
     if(owner_face_distance+neighbour_face_distance>tiny(1.d0)) then
      shared_face_depth=(neighbour_face_distance*(bed(c)+state(1,c))+ &
       owner_face_distance*(bed(n)+state(1,n)))/(owner_face_distance+neighbour_face_distance)- &
       max(bed(c),bed(n))
     else
      shared_face_depth=.5d0*((bed(c)+state(1,c))+(bed(n)+state(1,n)))-max(bed(c),bed(n))
     endif
     hydraulic_closed=shared_face_depth<=dry_tolerance
     ! A fast incoming bore may climb a nominally dry step even while the
     ! centre-stage aperture is closed.  Preserve that dynamic run-up; the
     ! shared-depth threshold is intended to suppress numerical seepage, not
     ! to replace the Riemann solver's reflected-wave overtopping physics.
     if(hydraulic_closed) hydraulic_closed=.not.dynamic_runup_opens_face( &
      l,r,left(bed_primitive_index()),right(bed_primitive_index()),normal(:,f))
    endif
    if(hydraulic_closed) then
     ! Block transport but retain each side's hydrostatic wall pressure.  The
     ! well-balanced correction below then completes this face value to the
     ! corresponding cell-centre pressure; zeroing momentum here would turn
     ! a lake at rest into a spurious bank-normal acceleration.
     fl(1,f)=0;fr(1,f)=0
     fl(2:3,f)=.5d0*grav*l(1)**2*normal(:,f)
     fr(2:3,f)=.5d0*grav*r(1)**2*normal(:,f)
     speed(f)=0
     if(l(1)>dry_tolerance) speed(f)=max(speed(f), &
      abs(dot_product(l(2:3),normal(:,f))/l(1))+sqrt(grav*l(1)))
     if(r(1)>dry_tolerance) speed(f)=max(speed(f), &
      abs(dot_product(r(2:3),normal(:,f))/r(1))+sqrt(grav*r(1)))
    else
     call wave_flux(l,r,left(bed_primitive_index()),right(bed_primitive_index()), &
      normal(:,f),fl(1:3,f),fr(1:3,f),speed(f))
    endif
    if(.not.hydraulic_closed) then
     ! This area is resolved from the accepted hydrostatic face traces that
     ! enter the f-wave flux.  The diffusion operator consumes this explicit
     ! hydraulic face quantity; it never reconstructs an area from endpoint
     ! cell depths.  The arithmetic trace-depth mapping mirrors FLOW's hu/hv
     ! staggered face area on this unstructured cut-cell route.
     hydraulic_face_exchange_area(f)=length(f)*max(0.d0,.5d0*(l(1)+r(1)))
    endif
    if(suspended_active) then
     do g=1,nfrac
      ci=concentration_index(g);cl(g,f)=left(ci);cr(g,f)=right(ci)
     enddo
    endif
   else if(b==0) then
    r=l;r(2:3)=l(2:3)-2*dot_product(l(2:3),normal(:,f))*normal(:,f)
    call wave_flux(l,r,left(bed_primitive_index()),left(bed_primitive_index()), &
     normal(:,f),fl(1:3,f),fr(1:3,f),speed(f))
    fl(1,f)=0;fl(2:3,f)=dot_product(fl(2:3,f),normal(:,f))*normal(:,f);fr(:,f)=fl(:,f)
    if(suspended_active) then
     do g=1,nfrac
      ci=concentration_index(g);cl(g,f)=left(ci);cr(g,f)=left(ci)
     enddo
    endif
   else
    if(mode(b)==2.and.bedlaw==2.and..not.capacity_foundation_runtime_declaration.and. &
       .not.bedload_adaptation.and. &
       .not.secondary_bedload_direction().and..not.bedform_roughness.and.state(1,c)>dry_tolerance) then
     if(dot_product(state(2:3,c),normal(:,f))>0.and. &
        dot_product(state(2:3,c),normal(:,f))<state(1,c)*sqrt(grav*state(1,c))) then
      call coupled_stage_characteristic(f,c,state,l,left(bed_primitive_index()),bc_hydraulic(2,b), &
       hiding(:,c),mixture_d90(c),step_morfac,trace(:,f),boundary_stage_bed,valid)
      if(valid) then
       left(bed_primitive_index())=boundary_stage_bed
       left(1)=boundary_stage_bed+trace(1,f)
      endif
     endif
    endif
    l=trace(:,f);depth=l(1);fl(1,f)=dot_product(l(2:3),normal(:,f))
    if(depth>dry_tolerance) fl(2:3,f)=fl(1,f)*l(2:3)/depth+.5d0*grav*depth**2*normal(:,f)
    fr(:,f)=fl(:,f)
    if(suspended_active) then
     do g=1,nfrac
      ci=concentration_index(g);cl(g,f)=v(ci,c)
      cr(g,f)=flow2dh_open_boundary_concentration(g,f,fl(1,f),dt, &
       bc_concentration(g,b),v(ci,c))
      bc_face_concentration(g,f)=cr(g,f)
     enddo
    endif
    hydraulic_face_exchange_area(f)=length(f)*max(0.d0,l(1))
    speed(f)=sqrt(grav*depth)
    if(depth>dry_tolerance) speed(f)=speed(f)+abs(fl(1,f))/depth
   endif
   ! Constant-surface wet/dry traces integrate pressure up to the shoreline:
   ! g/2*(h_cell**2-h_face**2). Extending g*h*db past h=0 produces
   ! a fictitious force. The same accepted face water flux still serves
   ! both cells, positivity limiting, every scalar and the boundary ledger.
   if(n>0.or.b==0) then
    if(state(1,c)>dry_tolerance) then
     if(hydro_constant(c)) then
      fl(2:3,f)=fl(2:3,f)+.5d0*grav*(state(1,c)**2-l(1)**2)*normal(:,f)
     else
      fl(2:3,f)=fl(2:3,f)+.5d0*grav* &
       (state(1,c)+max(left(1)-left(bed_primitive_index()),0.d0))* &
       (left(bed_primitive_index())-bed(c))*normal(:,f)
     endif
    endif
    if(n>0) then
     if(state(1,n)>dry_tolerance) then
      if(hydro_constant(n)) then
       fr(2:3,f)=fr(2:3,f)+.5d0*grav*(state(1,n)**2-r(1)**2)*normal(:,f)
      else
       fr(2:3,f)=fr(2:3,f)+.5d0*grav* &
        (state(1,n)+max(right(1)-right(bed_primitive_index()),0.d0))* &
        (right(bed_primitive_index())-bed(n))*normal(:,f)
      endif
     endif
    else
     fr(:,f)=fl(:,f)
    endif
   else if(mode(b)==2) then
    ! Complete the cell-centre-to-outlet bed force with the same owner-face
    ! reconstruction used for the characteristic above. Use the reconstructed
    ! interior depth, not an imposed tailwater depth, in this source integral.
    if(state(1,c)>dry_tolerance) then
     if(abs(left(bed_primitive_index())-bed(c))>64*epsilon(1.d0)*max(1.d0,abs(bed(c))).and. &
        abs(dot_product(state(2:3,c),normal(:,f)))<state(1,c)*sqrt(grav*state(1,c))) then
      depth=max(left(1)-left(bed_primitive_index()),0.d0)
      if(hydro_constant(c)) then
       fl(2:3,f)=fl(2:3,f)+.5d0*grav*(state(1,c)**2-depth**2)*normal(:,f)
      else
       fl(2:3,f)=fl(2:3,f)+.5d0*grav*(state(1,c)+depth)* &
        (left(bed_primitive_index())-bed(c))*normal(:,f)
      endif
     endif
    endif
    fr(:,f)=fl(:,f)
   else if(fixed_bed_face) then
    if(state(1,c)>dry_tolerance) then
     call gaia_fixed_inlet_pressure(state(1,c),trace(1,f),bed(c),fixed_boundary_bed(f),grav, &
      boundary_stage_bed,valid)
     if(.not.valid) error stop 'invalid TELEMAC-GAIA fixed inlet pressure correction'
     fl(2:3,f)=fl(2:3,f)+boundary_stage_bed*normal(:,f)
    endif
    fr(:,f)=fl(:,f)
   else if(sampled_boundary_bed(f)==1) then
    if(state(1,c)>dry_tolerance) fl(2:3,f)=fl(2:3,f)+.5d0*grav* &
     (state(1,c)+trace(1,f))*boundary_bed_offset(f)*normal(:,f)
    fr(:,f)=fl(:,f)
   endif
   ! Every class uses the same resolved hydraulics but its own grain properties,
   ! capacity, bed inventory and accepted boundary supply.
   if(bedlaw>0.and.n>0.and..not.hydraulic_closed) then
    do g=1,nfrac
     bi=bed_index(g)
     ! Reconstruct the cell-centred nonlinear capacity field.  This retains
     ! high spatial order without evaluating a nonlinear law independently on
     ! two hydraulic face traces, which can create an inlet-adjacent extremum.
     call bedload_capacity_face(c,f,g,bedq_cell,bedq_grad,blx,bly,coupled_stage_capacity(f,l,r))
     call bedload_capacity_face(n,f,g,bedq_cell,bedq_grad,brx,bry,coupled_stage_capacity(f,l,r))
     trl=bedtau_cell(g,c);trr=bedtau_cell(g,n)
     qsleft=dot_product([blx,bly],normal(:,f));qsright=dot_product([brx,bry],normal(:,f))
     if(qsleft>0.and.qsright>0) then
      qbx=blx;qby=bly;donor=c
     else if(qsleft<0.and.qsright<0) then
      qbx=brx;qby=bry;donor=n
     else
      ! Follow Delft3D's central fallback when neighbouring transport
      ! directions oppose.  Inventory limiting still uses the resulting
      ! shared face-flux direction to select its donor below.
      qbx=.5d0*(blx+brx);qby=.5d0*(bly+bry);donor=c
      if(dot_product([qbx,qby],normal(:,f))<0) donor=n
     endif
     if(any(bed_slope==[2,3,5])) then
      call face_downslope(f,bed,downslope,slx,sly,valid)
      if(.not.valid) error stop 'invalid face bed slope'
      tratio=.5d0*(trl+trr)
      if(bed_slope==2.or.bed_slope==5) then
       ! Positive FLOW EH/MPM branches leave erosed's taurat at zero.
       ! Keep that locked-runtime contract separate from legacy ratio mode 2.
       if(bed_slope==5) tratio=0.d0
       call braids_delft3d_bagnold_ikeda(qbx,qby,slx,sly,tratio,bed_slope_alfabs, &
        bed_slope_alfabn,bed_friction_angle_deg,blx,bly,valid)
      else
       call braids_delft3d_koch_flokstra(qbx,qby,slx,sly,tratio,bed_slope_alfabs, &
        bed_slope_alfabn,bed_friction_angle_deg,blx,bly,valid)
      endif
      if(.not.valid) error stop 'invalid corrected bedload capacity'
      qbx=blx;qby=bly
     endif
     ! Slope steering may reverse the normal transport direction. Select the
     ! inventory donor from the final corrected flux.
     if(dot_product([qbx,qby],normal(:,f))>=0.d0) then
      donor=c
     else
      donor=n
     endif
     qbx=qbx*bed_availability(state,g,donor);qby=qby*bed_availability(state,g,donor)
     fl(bi,f)=dot_product([qbx,qby],normal(:,f))
     if(.not.bedload_face_open(l,r,left(bed_primitive_index()), &
        right(bed_primitive_index()),normal(:,f))) fl(bi,f)=0
    enddo
    ! Flow-directed capacity upwinding is anti-diffusive for upstream-moving
    ! bed waves. Split the mobile-bed MPM face with the coupled SWE-Exner
    ! eigenstructure, including MorFac, while retaining physical slope fluxes.
    if(bedlaw==2.and..not.capacity_foundation_runtime_declaration.and. &
       .not.bedload_adaptation.and..not.secondary_bedload_direction().and..not.bedform_roughness) &
     call coupled_mpm_face(f,state,l,r,left(bed_primitive_index()),right(bed_primitive_index()), &
      hiding,mixture_d90,step_morfac,fl(:,f),fr(:,f),speed(f))
    if(bedlaw==9) call coupled_gaia_face(f,state,l,r,left(bed_primitive_index()), &
     right(bed_primitive_index()),step_morfac,fl(:,f),fr(:,f),speed(f))
   else if(bedlaw>0.and.b>0) then
    do g=1,nfrac
     bi=bed_index(g)
     raw_boundary_bed=0.d0;raw_boundary_suspended=0.d0
     composed_boundary_bed=0.d0;retained_boundary_suspended=0.d0
     boundary_signed_suspended_rate=0.d0
     boundary_equilibrium_concentration=0.d0;composition_status=BRAIDS_COMPOSE_OK
     if((mode(b)==1.or.mode(b)==6.or.mode(b)==7.or.mode(b)==8.or.mode(b)==9).and.fl(1,f)<0) then
      if(bed_boundary_mode(b)==1) then
       ! Monotone open-boundary closure: use the owner cell's capacity.  The
       ! prescribed hydraulic trace still controls water and momentum only.
       ! An equilibrium-capacity inlet is a constitutive boundary condition,
       ! not the lagged transport just inside the domain.  Using the local
       ! capacity here preserves that meaning when adaptation is enabled.
       sblx=0.d0;sbly=0.d0
       boundary_absolute_shields=0.d0;boundary_critical_shields=0.d0
       boundary_transport_stage=0.d0
       if(capacity_split_retained) then
        ! The split formulas are evaluated on the same hydraulic trace that
        ! supplies the inlet water flux.  Keep both constitutive channels and
        ! every unambiguous diagnostic; none may be silently substituted by
        ! the owner-cell result.
        call bed_capacity(trace(:,f),g,blx,bly,trl,valid,hiding(g,c),manning_cell(c), &
         mixture_dg(c),mixture_gsd(c),mixture_sand(c),mixture_d90(c),mixture_mud(c), &
         sblx,sbly,boundary_absolute_shields,boundary_critical_shields, &
         boundary_transport_stage)
        if(.not.valid) error stop 'invalid split-capacity equilibrium inlet'
        raw_boundary_suspended=[sblx,sbly]
        if(classic_secondary_foundation()) then
         call rotate_classic_secondary_bed(trace(:,f),c,[blx,bly], &
          raw_boundary_bed,boundary_spiral_alpha,valid)
         if(.not.valid) error stop 'invalid v56 equilibrium-inlet secondary direction'
         blx=raw_boundary_bed(1);bly=raw_boundary_bed(2)
        else if(any(bedlaw==[11,12,14]).and.secondary_bedload_direction()) then
         ! FLOW's spiral correction steers the scalar bed channel only.
         ! Suspended capacity remains signed and aligned with this inlet's
         ! depth-averaged hydraulic trace.  Apply the near-bed direction to
         ! the trace vector itself, so a superthreshold trace remains steered
         ! even when the owner-cell bed capacity is exactly zero.
         if(trace(1,f)>dry_tolerance.and.blx*blx+bly*bly>0.d0) then
          boundary_formula_n=hydraulic_equivalent_manning(trace(1,f),manning_cell(c))
          if(secondary_flow_model==1) then
           call braids_delft3d_spiral_transport(trace(1,f),trace(2,f)/trace(1,f), &
            trace(3,f)/trace(1,f),boundary_formula_n,grav,secondary_flow_espir, &
            secondary_curvature(c),blx,bly,boundary_spiral_intensity, &
            boundary_spiral_alpha,valid,engelund_hansen_chezy)
          else
           boundary_spiral_intensity=state(secondary_state_index(),c)/state(1,c)
           call braids_spiral_transport_from_intensity(trace(1,f),trace(2,f)/trace(1,f), &
            trace(3,f)/trace(1,f),boundary_formula_n,grav,secondary_flow_espir, &
            boundary_spiral_intensity,blx,bly,boundary_spiral_alpha,valid, &
            engelund_hansen_chezy)
          endif
          if(.not.valid) error stop 'invalid split-capacity inlet secondary direction'
         endif
        endif
         if(channel_composition_declaration) then
         ! Store the actual composer input after bed-only secondary steering.
         raw_boundary_bed=[blx,bly]
         call compose_live_law11_channels(trace(:,f),trace(1,f),g,[blx,bly],[sblx,sbly], &
          composed_boundary_bed,retained_boundary_suspended, &
          boundary_signed_suspended_rate,boundary_equilibrium_concentration, &
          valid,composition_status)
         if(.not.valid) error stop 'invalid v51 equilibrium-inlet channel composition'
         blx=composed_boundary_bed(1);bly=composed_boundary_bed(2)
         sblx=retained_boundary_suspended(1);sbly=retained_boundary_suspended(2)
        endif
       else if(bedlaw==9) then
        call gaia_open_boundary_capacity(trace(:,f),g,c,hiding(g,c),manning_cell(c), &
         mixture_dg(c),mixture_gsd(c),mixture_sand(c),mixture_d90(c),blx,bly,trl,valid)
        if(.not.valid) error stop 'invalid TELEMAC-GAIA equilibrium inlet capacity'
       else
        blx=bedq_equilibrium(1,g,c);bly=bedq_equilibrium(2,g,c);trl=bedtau_cell(g,c)
       endif
       if(capacity_split_retained) then
        split_boundary_role(f)=1
        split_boundary_bed(:,g,f)=[blx,bly]
        split_boundary_suspended(:,g,f)=[sblx,sbly]
        split_boundary_absolute_shields(g,f)=boundary_absolute_shields
        split_boundary_critical_shields(g,f)=boundary_critical_shields
        split_boundary_stress_ratio(g,f)=trl
        split_boundary_transport_stage(g,f)=boundary_transport_stage
       endif
       if(channel_composition_declaration) then
        channel_raw_bed_boundary(:,g,f)=raw_boundary_bed
        channel_raw_suspended_boundary(:,g,f)=raw_boundary_suspended
        channel_composed_bed_boundary(:,g,f)=[blx,bly]
        channel_retained_suspended_boundary(:,g,f)=[sblx,sbly]
        channel_signed_suspended_rate_boundary(g,f)=boundary_signed_suspended_rate
        channel_equilibrium_concentration_boundary(g,f)=boundary_equilibrium_concentration
       endif
       if(any(bed_slope==[2,3,5])) then
        call face_downslope(f,bed,downslope,slx,sly,valid)
        if(.not.valid) error stop 'invalid equilibrium inlet bed slope'
        if(bed_slope==2.or.bed_slope==5) then
         if(bed_slope==5) trl=0.d0
         call braids_delft3d_bagnold_ikeda(blx,bly,slx,sly,trl,bed_slope_alfabs, &
          bed_slope_alfabn,bed_friction_angle_deg,qbx,qby,valid)
        else
         call braids_delft3d_koch_flokstra(blx,bly,slx,sly,trl,bed_slope_alfabs, &
          bed_slope_alfabn,bed_friction_angle_deg,qbx,qby,valid)
        endif
        if(.not.valid) error stop 'invalid corrected equilibrium inlet capacity'
        blx=qbx;bly=qby
       endif
       blx=blx*bed_availability(state,g,c);bly=bly*bed_availability(state,g,c)
       fl(bi,f)=min(0.d0,dot_product([blx,bly],normal(:,f)))
      else if(deck_version>=3.and.bc_hydraulic(1,b)>0) then
       fl(bi,f)=bc_bed_input(g,b)*fl(1,f)/bc_hydraulic(1,b)
      else
       fl(bi,f)=-bc_bed_input(g,b)
      endif
     endif
     if(mode(b)>1.and.mode(b)/=6.and.mode(b)/=7.and.mode(b)/=8.and.mode(b)/=9.and.fl(1,f)>0) then
     blx=bedq_cell(1,g,c);bly=bedq_cell(2,g,c);trl=bedtau_cell(g,c)
      sblx=0.d0;sbly=0.d0
      boundary_absolute_shields=0.d0;boundary_critical_shields=0.d0
      boundary_transport_stage=0.d0
      if(capacity_split_retained) then
       sblx=suspended_capacity_equilibrium(1,g,c)
       sbly=suspended_capacity_equilibrium(2,g,c)
       boundary_absolute_shields=capacity_absolute_shields(g,c)
       boundary_critical_shields=capacity_critical_shields(g,c)
       boundary_transport_stage=capacity_transport_stage(g,c)
      endif
      if(channel_composition_declaration) then
       raw_boundary_bed=channel_raw_bed_cell(:,g,c)
       raw_boundary_suspended=channel_raw_suspended_cell(:,g,c)
       composed_boundary_bed=[blx,bly]
       retained_boundary_suspended=[sblx,sbly]
       boundary_signed_suspended_rate=channel_signed_suspended_rate_cell(g,c)
       boundary_equilibrium_concentration=channel_equilibrium_concentration_cell(g,c)
      endif
      ! Local-equilibrium stage export must use the SAME hydraulic trace as
      ! the water flux. Cell-centre capacity can fall as a boundary bed wave
      ! changes depth, while the imposed stage accelerates flow at the face;
      ! that inconsistent pairing traps sediment in an artificial outlet mound.
      ! Retain legacy supercritical release and non-equilibrium closures.
      if(mode(b)==2.and.((bedlaw==2.and..not.secondary_bedload_direction()).or. &
         any(bedlaw==[8,9,11,12,13,14])).and. &
         .not.bedload_adaptation.and..not.bedform_roughness.and.state(1,c)>dry_tolerance) then
       if(abs(dot_product(state(2:3,c),normal(:,f)))<state(1,c)*sqrt(grav*state(1,c))) then
       qbx=blx;qby=bly
        if(bedlaw==9) then
         call gaia_open_boundary_capacity(trace(:,f),g,c,hiding(g,c),manning_cell(c), &
          mixture_dg(c),mixture_gsd(c),mixture_sand(c),mixture_d90(c),blx,bly,trl,valid)
        else
         call bed_capacity(trace(:,f),g,blx,bly,trl,valid,hiding(g,c),manning_cell(c), &
          mixture_dg(c),mixture_gsd(c),mixture_sand(c),mixture_d90(c),mixture_mud(c), &
          sblx,sbly,boundary_absolute_shields,boundary_critical_shields, &
          boundary_transport_stage)
        endif
        if(.not.valid) error stop 'invalid stage-state export capacity'
        if(channel_composition_declaration) raw_boundary_suspended=[sblx,sbly]
        if(classic_secondary_foundation()) then
         call rotate_classic_secondary_bed(trace(:,f),c,[blx,bly], &
          raw_boundary_bed,boundary_spiral_alpha,valid)
         if(.not.valid) error stop 'invalid v56 stage-outlet secondary direction'
         blx=raw_boundary_bed(1);bly=raw_boundary_bed(2)
        else if(bedlaw==8.and.secondary_bedload_direction()) then
         ! The stage trace sets the scalar bed-formula magnitude; retain the
         ! legacy law-8 near-bed direction from the owner state.
         if(qbx*qbx+qby*qby>0.d0) then
          depth=sqrt(blx*blx+bly*bly)/sqrt(qbx*qbx+qby*qby)
          blx=depth*qbx;bly=depth*qby
         endif
         if(bedlaw==8.and.bed_slope==3.and.qbx*qbx+qby*qby>0.d0.and. &
            state(2,c)**2+state(3,c)**2>0.d0) then
          ! The boundary hydraulic trace sets total shear, while its transport
          ! direction retains the owner's spiral steering. Carry that same
          ! eqtran ust2 amplification into Koch-Flokstra's Shields input.
          depth=(qbx*state(2,c)+qby*state(3,c))/ &
           sqrt((qbx*qbx+qby*qby)*(state(2,c)**2+state(3,c)**2))
          if(depth<=0.d0) error stop 'invalid stage-state spiral slope stress'
          trl=trl/min(depth,1.d0)
         endif
        else if(any(bedlaw==[11,12,14]).and.secondary_bedload_direction()) then
         ! v46 scalar bed capacity uses the trace velocity with the owner's
         ! resolved spiral state/curvature.  Suspended capacity is deliberately
         ! untouched, signed, and aligned with the depth-averaged trace.
         if(trace(1,f)>dry_tolerance.and.blx*blx+bly*bly>0.d0) then
          boundary_formula_n=hydraulic_equivalent_manning(trace(1,f),manning_cell(c))
          if(secondary_flow_model==1) then
           call braids_delft3d_spiral_transport(trace(1,f),trace(2,f)/trace(1,f), &
            trace(3,f)/trace(1,f),boundary_formula_n,grav,secondary_flow_espir, &
            secondary_curvature(c),blx,bly,boundary_spiral_intensity, &
            boundary_spiral_alpha,valid,engelund_hansen_chezy)
          else
           boundary_spiral_intensity=state(secondary_state_index(),c)/state(1,c)
           call braids_spiral_transport_from_intensity(trace(1,f),trace(2,f)/trace(1,f), &
            trace(3,f)/trace(1,f),boundary_formula_n,grav,secondary_flow_espir, &
            boundary_spiral_intensity,blx,bly,boundary_spiral_alpha,valid, &
            engelund_hansen_chezy)
          endif
          if(.not.valid) error stop 'invalid split-capacity outlet secondary direction'
         endif
        endif
        if(channel_composition_declaration) then
         ! Store the actual trace composer input after bed-only steering.
         raw_boundary_bed=[blx,bly]
         call compose_live_law11_channels(trace(:,f),trace(1,f),g,[blx,bly],[sblx,sbly], &
          composed_boundary_bed,retained_boundary_suspended, &
          boundary_signed_suspended_rate,boundary_equilibrium_concentration, &
          valid,composition_status)
         if(.not.valid) error stop 'invalid v51 stage-outlet channel composition'
         blx=composed_boundary_bed(1);bly=composed_boundary_bed(2)
         sblx=retained_boundary_suspended(1);sbly=retained_boundary_suspended(2)
        endif
       endif
      endif
      if(capacity_split_retained) then
       split_boundary_role(f)=merge(2,3,mode(b)==2.and. &
        .not.bedload_adaptation.and..not.bedform_roughness.and. &
        state(1,c)>dry_tolerance.and. &
        abs(dot_product(state(2:3,c),normal(:,f)))<state(1,c)*sqrt(grav*state(1,c)))
       split_boundary_bed(:,g,f)=[blx,bly]
       split_boundary_suspended(:,g,f)=[sblx,sbly]
       split_boundary_absolute_shields(g,f)=boundary_absolute_shields
       split_boundary_critical_shields(g,f)=boundary_critical_shields
       split_boundary_stress_ratio(g,f)=trl
       split_boundary_transport_stage(g,f)=boundary_transport_stage
      endif
      if(channel_composition_declaration) then
       channel_raw_bed_boundary(:,g,f)=raw_boundary_bed
       channel_raw_suspended_boundary(:,g,f)=raw_boundary_suspended
       channel_composed_bed_boundary(:,g,f)=[blx,bly]
       channel_retained_suspended_boundary(:,g,f)=[sblx,sbly]
       channel_signed_suspended_rate_boundary(g,f)=boundary_signed_suspended_rate
       channel_equilibrium_concentration_boundary(g,f)=boundary_equilibrium_concentration
      endif
      if(any(bed_slope==[2,3,5])) then
       call face_downslope(f,bed,downslope,slx,sly,valid)
       if(.not.valid) error stop 'invalid boundary bed slope'
       if(bed_slope==2.or.bed_slope==5) then
        if(bed_slope==5) trl=0.d0
        call braids_delft3d_bagnold_ikeda(blx,bly,slx,sly,trl,bed_slope_alfabs, &
         bed_slope_alfabn,bed_friction_angle_deg,qbx,qby,valid)
       else
        call braids_delft3d_koch_flokstra(blx,bly,slx,sly,trl,bed_slope_alfabs, &
         bed_slope_alfabn,bed_friction_angle_deg,qbx,qby,valid)
       endif
       if(.not.valid) error stop 'invalid corrected export capacity'
       blx=qbx;bly=qby
      endif
      blx=blx*bed_availability(state,g,c);bly=bly*bed_availability(state,g,c)
      fl(bi,f)=max(0.d0,dot_product([blx,bly],normal(:,f)))
     endif
    enddo
   endif
   ! Assemble the optional stress after every hydraulic/Exner face closure.
   ! In particular, coupled_mpm_face may replace the hydraulic signal bound
   ! with max(speed,new_speed); adding c_spiral before that call could silently
   ! discard some or all of the required secondary-flow CFL contribution.
   if(prognostic_secondary_flow().and.secondary_flow_momentum_beta>0.d0.and..not.hydraulic_closed) then
    stress_left=0.d0;stress_right=0.d0;stress_signal_left=0.d0;stress_signal_right=0.d0
    stress_active_left=.false.;stress_active_right=.false.
    if(l(1)>dry_tolerance) then
     call braids_spiral_stress(l(1),l(2)/l(1),l(3)/l(1), &
      hydraulic_equivalent_manning(l(1),manning_cell(c)),grav, &
      secondary_flow_momentum_beta,secondary_face_left(f),stress_left,stress_signal_left, &
      stress_active_left,valid)
     if(.not.valid) secondary_invalid=1
    endif
    if(n>0.and.r(1)>dry_tolerance) then
     call braids_spiral_stress(r(1),r(2)/r(1),r(3)/r(1), &
      hydraulic_equivalent_manning(r(1),manning_cell(n)),grav, &
      secondary_flow_momentum_beta,secondary_face_right(f),stress_right,stress_signal_right, &
      stress_active_right,valid)
     if(.not.valid) secondary_invalid=1
     stress_face=.5d0*(stress_left+stress_right)
    else
     stress_face=stress_left
    endif
    ! The Lien secondary circulation transfers streamwise momentum from the
    ! inner side of a bend toward the outer side.  The annular-sector oracle
    ! and a frozen Jamieson impulse both show that this solver's signed
    ! curvature/intensity convention requires +T.n in the conservative flux;
    ! -T.n reverses that transfer while still looking algebraically plausible.
    stress_flux=matmul(stress_face,normal(:,f))
    secondary_stress_face(:,f)=stress_flux
    fl(2:3,f)=fl(2:3,f)+stress_flux;fr(2:3,f)=fr(2:3,f)+stress_flux
    speed(f)=speed(f)+max(stress_signal_left,stress_signal_right)
   endif
  enddo
  !$omp end parallel do
  if(secondary_invalid/=0) error stop 'invalid prognostic secondary-flow face closure'
  if(bank_front_tracking) then
   ! A partially traversed bank cell represents an unresolved solid/channel
   ! mixture. Ordinary Exner transport must not vertically refill or scour
   ! that cell while the conservative bank operator advances its lateral
   ! face. The cell is returned to ordinary Exner only after promotion.
   !$omp parallel do schedule(static,256) if(parallel_faces) default(shared) private(work,f,c,n,g,bi)
   do work=1,nwork
    f=work
    if(amr_step_level>=0) f=amr_level_faces(work,amr_step_level)
    c=owner(f);n=neighbour(f)
    if(bank_front_status(c)/=BANK_FRONT_ACTIVE) then
     if(n<=0) cycle
     if(bank_front_status(n)/=BANK_FRONT_ACTIVE) cycle
    endif
    do g=1,nfrac
     bi=bed_index(g);fl(bi,f)=0.d0;fr(bi,f)=0.d0
    enddo
   enddo
   !$omp end parallel do
  endif
  ! Match Delft3D's dtmor = dts * MorFac contract without altering the
  ! constitutive transport capacity.  Scaling the one shared conservative
  ! bed flux here also puts the accelerated rate through donor limiting, AMR
  ! reflux, boundary ledgers and every SSPRK stage consistently.
  morfac=step_morfac
  if(morfac/=1.d0) then
   !$omp parallel do schedule(static,256) if(parallel_faces) default(shared) private(work,f,g,bi)
   do work=1,nwork
    f=work
    if(amr_step_level>=0) f=amr_level_faces(work,amr_step_level)
    do g=1,nfrac
     bi=bed_index(g);fl(bi,f)=morfac*fl(bi,f)
    enddo
   enddo
   !$omp end parallel do
  endif
  if(profile_enabled) then
   profile_euler_flux_s=profile_euler_flux_s+omp_get_wtime()-profile_phase_start
   profile_phase_start=omp_get_wtime()
  endif
  ! Conservative water draining changes the shared advective flux on both sides.
  water_theta=1
  !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) private(c,k,f,outgoing)
  do c=1,nc
   ! A coarse cell is held as a time-interpolated ghost during a finer
   ! substep, but it can still donate through a coarse/fine face.  Include it
   ! in the draining limiter so the accumulated fine flux cannot draw water
   ! from an inventory that the refluxed coarse cell does not possess.
   if(.not.active_cell(c).and.amr_step_level<0) cycle
   outgoing=0
   do k=offset(c),offset(c+1)-1
    f=faces(k);outgoing=outgoing+max(signs(k)*fl(1,f),0.d0)*length(f)
   enddo
   if(dt*outgoing>=state(1,c)*area(c).and.outgoing>0) &
    water_theta(c)=min(1.d0,state(1,c)*area(c)/(dt*outgoing))*(1-32*epsilon(1.d0))
  enddo
  !$omp end parallel do
  water_failure=0
  !$omp parallel do schedule(static,256) if(parallel_faces) default(shared) &
  !$omp& private(work,f,donor,correction,velocity) reduction(max:water_failure)
  do work=1,nwork
   f=work
   if(amr_step_level>=0) f=amr_level_faces(work,amr_step_level)
   if(fl(1,f)==0) cycle
   donor=owner(f)
   if(fl(1,f)<0) donor=neighbour(f)
   if(donor==0) cycle
   if(water_theta(donor)==1) cycle
   if(bid(f)>0) then
    water_failure=2;cycle
   endif
   correction=(water_theta(donor)-1)*fl(1,f);velocity=0
   if(state(1,donor)>dry_tolerance) velocity=state(2:3,donor)/state(1,donor)
   fl(1,f)=fl(1,f)*water_theta(donor);fr(1,f)=fl(1,f)
   fl(2:3,f)=fl(2:3,f)+correction*velocity;fr(2:3,f)=fr(2:3,f)+correction*velocity
  enddo
  !$omp end parallel do
  if(water_failure>0) then
   if(profile_enabled) profile_euler_limit_s=profile_euler_limit_s+omp_get_wtime()-profile_phase_start
   failure_reason=water_failure;return
  endif
  ! CFL, hydraulic update, and a separate draining limiter for each bed class.
  if(.not.hydraulic_workflow_fast) theta=1
  stage_failure=0;stage_cfl_peak=0
  if(suspended_active) then
   loflux=0;anti=0
  endif
  depth_tolerance=64*epsilon(1.d0)*max(maxval(state(1,:)),1.d0)
  !$omp parallel do schedule(static,256) if(nc*nfrac>=8192) default(shared) &
  !$omp& private(c,k,f,g,bi,sumflux,cfl,cfl_ratio,outgoing) reduction(max:stage_failure,stage_cfl_peak)
  do c=1,nc
   if(.not.active_cell(c).and.amr_step_level<0) cycle
   sumflux=0;cfl=0
   do k=offset(c),offset(c+1)-1
    f=faces(k);cfl=cfl+length(f)*speed(f)
    if(signs(k)>0) then
     sumflux(1)=sumflux(1)+(length(f)/area(c))*fl(1,f)
     sumflux(2:3)=sumflux(2:3)+(length(f)/area(c))* &
      (fl(2:3,f)-.5d0*grav*state(1,c)**2*normal(:,f))
    else
     sumflux(1)=sumflux(1)-(length(f)/area(c))*fr(1,f)
     sumflux(2:3)=sumflux(2:3)-(length(f)/area(c))* &
      (fr(2:3,f)-.5d0*grav*state(1,c)**2*normal(:,f))
    endif
   enddo
   cfl_ratio=dt*cfl/area(c);stage_cfl_peak=max(stage_cfl_peak,cfl_ratio)
   if(amr_updates_cell(c)) then
    if(cfl_ratio>cfl_target) then
     stage_failure=max(stage_failure,2);cycle
    endif
    next(1:3,c)=state(1:3,c)-dt*sumflux(1:3)
    if(next(1,c)<0) then
     if(-next(1,c)<=depth_tolerance) then
      next(1:3,c)=0
     else
      stage_failure=max(stage_failure,3);cycle
     endif
    endif
   endif
   if(.not.hydraulic_workflow_fast) then
   do g=1,nfrac
    bi=bed_index(g);outgoing=0
    do k=offset(c),offset(c+1)-1
     f=faces(k);outgoing=outgoing+max(signs(k)*fl(bi,f),0.d0)*length(f)
    enddo
    if(dt*outgoing>state(bi,c)*area(c)) &
     theta(g,c)=state(bi,c)*area(c)/(dt*outgoing)*(1-16*epsilon(1.d0))
   enddo
   endif
  enddo
  !$omp end parallel do
  if(stage_failure>0) then
   if(profile_enabled) profile_euler_limit_s=profile_euler_limit_s+omp_get_wtime()-profile_phase_start
   if(stage_failure==3) then
    amr_failure_cell=minloc(next(1,:),dim=1);amr_failure_state=1
    amr_failure_value=next(1,amr_failure_cell)
   else if(stage_failure==2) then
    ! Populate the same diagnostics used by reflux failures.  This scan runs
    ! only on a rejected trial and does not alter the accepted parallel path.
    amr_failure_value=-huge(1.d0);amr_failure_aux=0;amr_failure_state=1
    do target=1,ntarget
     c=target
     if(amr_step_level>=0) c=amr_level_cells(target,amr_step_level)
     cfl=0
     do k=offset(c),offset(c+1)-1
      f=faces(k);cfl=cfl+length(f)*speed(f)
     enddo
     if(dt*cfl/area(c)>amr_failure_value) then
      amr_failure_value=dt*cfl/area(c);amr_failure_aux=area(c);amr_failure_cell=c
     endif
    enddo
   endif
   failure_reason=stage_failure;return
  endif
  ! With no bed flux or transported scalar, next=state already retains
  ! these inventories. Keep the complete original sweep for every other case.
  if(.not.hydraulic_workflow_fast) then
  !$omp parallel do schedule(static,256) if(parallel_faces) default(shared) &
  !$omp& private(work,f,c,n,b,g,bi,ci,donor,depth,velocity,secondary_equilibrium,secondary_high,secondary_active,valid) &
  !$omp& reduction(max:secondary_invalid)
  do work=1,nwork
   f=work
   if(amr_step_level>=0) f=amr_level_faces(work,amr_step_level)
   c=owner(f);n=neighbour(f);b=bid(f)
   do g=1,nfrac
    bi=bed_index(g);ci=concentration_index(g);donor=c
    if(fl(bi,f)<0) donor=n
    if(donor>0) fl(bi,f)=fl(bi,f)*theta(g,donor)
    fr(bi,f)=fl(bi,f)
    if(suspended_active) then
     ! First-order scalar reference and high-order candidate share accepted water.
     if(fl(1,f)>=0) then
      loflux(g,f)=fl(1,f)*v(ci,c);anti(g,f)=fl(1,f)*cl(g,f)-loflux(g,f)
     else if(n>0) then
      loflux(g,f)=fl(1,f)*v(ci,n);anti(g,f)=fl(1,f)*cr(g,f)-loflux(g,f)
     else
      loflux(g,f)=fl(1,f)*cr(g,f);anti(g,f)=0
     endif
     if(n==0) anti(g,f)=0
    endif
   enddo
   if(prognostic_secondary_flow()) then
    if(fl(1,f)>=0.d0) then
     secondary_low_flux(f)=fl(1,f)*v(secondary_primitive_index(),c)
     secondary_high=fl(1,f)*secondary_face_left(f)
     secondary_anti_flux(f)=secondary_high-secondary_low_flux(f)
    else if(n>0) then
     secondary_low_flux(f)=fl(1,f)*v(secondary_primitive_index(),n)
     secondary_high=fl(1,f)*secondary_face_right(f)
     secondary_anti_flux(f)=secondary_high-secondary_low_flux(f)
    else
     secondary_equilibrium=0.d0;secondary_active=.false.;valid=.true.;depth=state(1,c)
     if(depth>dry_tolerance) then
      velocity=state(2:3,c)/depth
      call braids_spiral_equilibrium(depth,velocity(1),velocity(2),secondary_curvature(c), &
       secondary_equilibrium,secondary_active,valid)
     endif
     if(.not.valid) secondary_invalid=1
     secondary_boundary_intensity(f)=secondary_equilibrium
     secondary_low_flux(f)=fl(1,f)*secondary_equilibrium
     secondary_anti_flux(f)=0.d0
    endif
   endif
  enddo
  !$omp end parallel do
  endif
  if(secondary_invalid/=0) error stop 'invalid prognostic secondary-flow boundary closure'
  trial_cfl_peak=max(trial_cfl_peak,stage_cfl_peak)
  ! Multidimensional FCT is applied independently to every suspended class.
  stage_failure=0
  if(suspended_active) then
   lowstate=0;rp=1;rm=1
   do g=1,nfrac
    lowstate(g,:)=state(suspended_index(g),:)
   enddo
   !$omp parallel do collapse(2) schedule(static,256) if(nc*nfrac>=8192) default(shared) &
   !$omp& private(c,g,k,f,n,si,ci,pp,pm,low,lower,upper,z) reduction(max:stage_failure)
   do c=1,nc
    do g=1,nfrac
     ! Bound antidiffusive scalar fluxes against coarse ghost donors too.  The
     ! ghost state itself is not advanced below; this only constrains the one
     ! shared interface flux that will replace the coarse flux during reflux.
     if(.not.active_cell(c).and.amr_step_level<0) cycle
     si=suspended_index(g);ci=concentration_index(g)
     pp=0;pm=0;low=state(si,c);lower=v(ci,c);upper=v(ci,c)
     do k=offset(c),offset(c+1)-1
      f=faces(k);n=neighbour(f)
      if(signs(k)<0) n=owner(f)
      if(n>0) then
       lower=min(lower,v(ci,n));upper=max(upper,v(ci,n))
      else if(bid(f)>0.and.fl(1,f)<0) then
       if(flow2dh_boundary_diffusion_declaration) then
        lower=min(lower,cr(g,f));upper=max(upper,cr(g,f))
       else
        lower=min(lower,bc_concentration(g,bid(f)))
        upper=max(upper,bc_concentration(g,bid(f)))
       endif
      endif
      z=-dt*signs(k)*length(f)/area(c)
      low=low+z*loflux(g,f);pp=pp+max(z*anti(g,f),0.d0);pm=pm+min(z*anti(g,f),0.d0)
     enddo
     if(low<0) then
      stage_failure=4;cycle
     endif
     call smooth_concentration_bounds(c,v,ci,lower,upper)
     lowstate(g,c)=low
     if(pp>0) rp(g,c)=min(1.d0,max(0.d0,upper*next(1,c)-low-64*tiny(1.d0))/pp)
     if(pm<0) rm(g,c)=min(1.d0,max(0.d0,low-lower*next(1,c)-64*tiny(1.d0))/(-pm))
     if(pp>0) rp(g,c)=rp(g,c)*(1-32*epsilon(1.d0))
     if(pm<0) rm(g,c)=rm(g,c)*(1-32*epsilon(1.d0))
    enddo
   enddo
   !$omp end parallel do
   if(stage_failure>0) then
    if(profile_enabled) profile_euler_limit_s=profile_euler_limit_s+omp_get_wtime()-profile_phase_start
    failure_reason=stage_failure;return
   endif
   !$omp parallel do schedule(static,256) if(parallel_faces) default(shared) private(work,f,g,c,n,si,scale)
   do work=1,nwork
    f=work
    if(amr_step_level>=0) f=amr_level_faces(work,amr_step_level)
    do g=1,nfrac
     c=owner(f);n=neighbour(f);scale=1
     if(anti(g,f)>0) then
      scale=rm(g,c)
      if(n>0) scale=min(scale,rp(g,n))
     else if(anti(g,f)<0) then
      scale=rp(g,c)
      if(n>0) scale=min(scale,rm(g,n))
     endif
     anti(g,f)=scale*anti(g,f);si=suspended_index(g)
     fl(si,f)=loflux(g,f)+anti(g,f);fr(si,f)=fl(si,f)
    enddo
   enddo
   !$omp end parallel do
  endif
  ! Signed FCT for the conserved spiral intensity M_I=h*I.  Its low-order
  ! flux uses the already accepted water flux; the antidiffusive correction
  ! is bounded in I, never by a sediment non-negativity constraint.
  if(prognostic_secondary_flow()) then
   secondary_low_state=state(secondary_state_index(),:)
   secondary_rplus=1.d0;secondary_rminus=1.d0;stage_failure=0
   !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) &
   !$omp& private(c,k,f,n,pp,pm,low,lower,upper,z,scale) reduction(max:stage_failure)
   do c=1,nc
    if(.not.active_cell(c).and.amr_step_level<0) cycle
    pp=0.d0;pm=0.d0;low=state(secondary_state_index(),c)
    lower=v(secondary_primitive_index(),c);upper=lower
    do k=offset(c),offset(c+1)-1
     f=faces(k);n=neighbour(f)
     if(signs(k)<0) n=owner(f)
     if(n>0) then
      lower=min(lower,v(secondary_primitive_index(),n))
      upper=max(upper,v(secondary_primitive_index(),n))
     else if(bid(f)>0.and.fl(1,f)<0.d0) then
      lower=min(lower,secondary_boundary_intensity(f))
      upper=max(upper,secondary_boundary_intensity(f))
     endif
     z=-dt*signs(k)*length(f)/area(c)
     low=low+z*secondary_low_flux(f)
     pp=pp+max(z*secondary_anti_flux(f),0.d0)
     pm=pm+min(z*secondary_anti_flux(f),0.d0)
    enddo
    if(next(1,c)<=0.d0) then
     lower=0.d0;upper=0.d0
    else
     lower=lower*next(1,c);upper=upper*next(1,c)
    endif
    scale=512*epsilon(1.d0)*max(1.d0,abs(low),abs(lower),abs(upper))
    if(low<lower-scale.or.low>upper+scale) then
     stage_failure=6;cycle
    endif
    secondary_low_state(c)=low
    if(pp>0.d0) secondary_rplus(c)=min(1.d0,max(0.d0,upper-low)/pp)
    if(pm<0.d0) secondary_rminus(c)=min(1.d0,max(0.d0,low-lower)/(-pm))
    if(pp>0.d0) secondary_rplus(c)=secondary_rplus(c)*(1.d0-32*epsilon(1.d0))
    if(pm<0.d0) secondary_rminus(c)=secondary_rminus(c)*(1.d0-32*epsilon(1.d0))
   enddo
   !$omp end parallel do
   if(stage_failure>0) then
    if(profile_enabled) profile_euler_limit_s=profile_euler_limit_s+omp_get_wtime()-profile_phase_start
    failure_reason=stage_failure;return
   endif
   !$omp parallel do schedule(static,256) if(parallel_faces) default(shared) private(work,f,c,n,scale)
   do work=1,nwork
    f=work
    if(amr_step_level>=0) f=amr_level_faces(work,amr_step_level)
    c=owner(f);n=neighbour(f);scale=1.d0
    if(secondary_anti_flux(f)>0.d0) then
     scale=secondary_rminus(c)
     if(n>0) scale=min(scale,secondary_rplus(n))
    else if(secondary_anti_flux(f)<0.d0) then
     scale=secondary_rplus(c)
     if(n>0) scale=min(scale,secondary_rminus(n))
    endif
    secondary_anti_flux(f)=scale*secondary_anti_flux(f)
    fl(secondary_state_index(),f)=secondary_low_flux(f)+secondary_anti_flux(f)
    fr(secondary_state_index(),f)=fl(secondary_state_index(),f)
   enddo
   !$omp end parallel do
  endif
  ! With no bed flux or transported scalar, next=state already retains
  ! these inventories. Keep the complete original sweep for every other case.
  if(.not.hydraulic_workflow_fast) then
  !$omp parallel do schedule(static,256) if(ntarget*nfrac>=8192) default(shared) private(target,c,g,k,f,si,bi,sumflux)
  do target=1,ntarget
   c=target
   if(amr_step_level>=0) c=amr_level_cells(target,amr_step_level)
   if(.not.active_cell(c)) cycle
   sumflux=0
   do k=offset(c),offset(c+1)-1
    f=faces(k)
    do g=1,nfrac
     bi=bed_index(g);sumflux(bi)=sumflux(bi)+signs(k)*(length(f)/area(c))*fl(bi,f)
    enddo
   enddo
   do g=1,nfrac
    si=suspended_index(g);bi=bed_index(g)
    next(bi,c)=state(bi,c)-dt*sumflux(bi)
    if(suspended_active) then
     next(si,c)=lowstate(g,c)
     do k=offset(c),offset(c+1)-1
      f=faces(k);next(si,c)=next(si,c)-dt*signs(k)*length(f)*anti(g,f)/area(c)
     enddo
    endif
   enddo
  enddo
  !$omp end parallel do
  endif
  ! FLOW 2DH horizontal suspended diffusion (difu.f90): an arithmetic dicuv
  ! face value divided by the locked 0.7 branch.  The exchange cross-section
  ! above is an explicit hydraulic face quantity formed from the accepted
  ! hydrostatic traces; this scalar operator must never reconstruct it from
  ! endpoint cell depths.  DryFlc belongs only to this diffusion gate.  The
  ! hydraulic/advection/FCT path above continues to use its existing kfs-like
  ! dry_tolerance mapping, so the two thresholds remain intentionally distinct.
  if(flow2dh_boundary_diffusion_declaration.and.suspended_active.and. &
     flow2dh_horizontal_diffusivity>0.d0.and.dt>0.d0) then
   suspended_diffusive_flux=0.d0
   suspended_diffusion_theta=1.d0
   !$omp parallel do schedule(static,256) if(parallel_faces) default(shared) &
   !$omp& private(work,f,c,n,b,g,si,diffusion_distance,diffusion_left,diffusion_right,diffusion_rate)
   do work=1,nwork
    f=work
    if(amr_step_level>=0) f=amr_level_faces(work,amr_step_level)
    c=owner(f);n=neighbour(f);b=bid(f)
    if(hydraulic_face_exchange_area(f)<=0.d0.or.length(f)<=0.d0) cycle
    if(next(1,c)<=flow2dh_diffusion_dry_threshold) cycle
    if(n>0) then
     if(next(1,n)<=flow2dh_diffusion_dry_threshold) cycle
     diffusion_distance=abs(dot_product(xy(:,n)-xy(:,c),normal(:,f)))
     if(diffusion_distance<=epsilon(1.d0)) &
      diffusion_distance=sqrt(sum((xy(:,n)-xy(:,c))**2))
    else
     if(b<=0) cycle
     ! FLOW applies the owner h1 value at an open boundary when forming the
     ! exterior wet mask.  The prescribed/TH/NeuBc concentration itself is
     ! already resolved per face in bc_face_concentration.
     diffusion_distance=abs(dot_product(mid(:,f)-xy(:,c),normal(:,f)))
     if(diffusion_distance<=epsilon(1.d0)) &
      diffusion_distance=sqrt(sum((mid(:,f)-xy(:,c))**2))
    endif
    if(diffusion_distance<=epsilon(1.d0)) cycle
    do g=1,nfrac
     si=suspended_index(g)
     diffusion_left=max(0.d0,next(si,c)/next(1,c))
     if(n>0) then
      diffusion_right=max(0.d0,next(si,n)/next(1,n))
     else
      diffusion_right=max(0.d0,bc_face_concentration(g,f))
     endif
     ! diffusion_rate is a signed whole-face mass rate, positive owner to
     ! neighbour/exterior.  Store the native per-width flux so the existing
     ! deterministic cell gathers and ledgers retain one common convention.
     diffusion_rate=hydraulic_face_exchange_area(f)* &
      (flow2dh_horizontal_diffusivity/flow2dh_horizontal_schmidt)* &
      (diffusion_left-diffusion_right)/diffusion_distance
     suspended_diffusive_flux(g,f)=diffusion_rate/length(f)
    enddo
   enddo
   !$omp end parallel do
   ! Bound explicit diffusion by each donor's post-advection inventory.
   ! Fixed-order incidence gathers are deterministic across thread counts.
   !$omp parallel do collapse(2) schedule(static,256) if(nc*nfrac>=8192) default(shared) &
   !$omp& private(c,g,k,f,outgoing)
   do c=1,nc
    do g=1,nfrac
     ! Coarse dense-output cells are donors to fine-level interface fluxes.
     ! They are not advanced in this substep, but their available inventory
     ! must still bound the shared flux later installed by reflux.
     outgoing=0.d0
     do k=offset(c),offset(c+1)-1
      f=faces(k)
      outgoing=outgoing+max(signs(k)*suspended_diffusive_flux(g,f),0.d0)*length(f)
     enddo
     if(dt*outgoing>=next(suspended_index(g),c)*area(c).and.outgoing>0.d0) &
      suspended_diffusion_theta(g,c)=min(1.d0, &
       next(suspended_index(g),c)*area(c)/(dt*outgoing))* &
       (1.d0-32.d0*epsilon(1.d0))
    enddo
   enddo
   !$omp end parallel do
   !$omp parallel do schedule(static,256) if(parallel_faces) default(shared) &
   !$omp& private(work,f,c,n,g,donor,scale)
   do work=1,nwork
    f=work
    if(amr_step_level>=0) f=amr_level_faces(work,amr_step_level)
    c=owner(f);n=neighbour(f)
    do g=1,nfrac
     scale=1.d0;donor=0
     if(suspended_diffusive_flux(g,f)>0.d0) donor=c
     if(suspended_diffusive_flux(g,f)<0.d0) donor=n
     if(donor>0) scale=suspended_diffusion_theta(g,donor)
     suspended_diffusive_flux(g,f)=scale*suspended_diffusive_flux(g,f)
    enddo
   enddo
   !$omp end parallel do
   !$omp parallel do collapse(2) schedule(static,256) if(ntarget*nfrac>=8192) default(shared) &
   !$omp& private(target,c,g,k,f,diffusion_rate)
   do target=1,ntarget
    do g=1,nfrac
     c=target
     if(amr_step_level>=0) c=amr_level_cells(target,amr_step_level)
     if(.not.amr_updates_cell(c)) cycle
     diffusion_rate=0.d0
     do k=offset(c),offset(c+1)-1
      f=faces(k)
      diffusion_rate=diffusion_rate+signs(k)*length(f)*suspended_diffusive_flux(g,f)
     enddo
     next(suspended_index(g),c)=next(suspended_index(g),c)- &
      dt*diffusion_rate/area(c)
    enddo
   enddo
   !$omp end parallel do
  endif
  if(prognostic_secondary_flow()) then
   stage_failure=0
   !$omp parallel do schedule(static,256) if(ntarget>=4096) default(shared) &
   !$omp& private(target,c,k,f,z,scale) reduction(max:stage_failure)
   do target=1,ntarget
    c=target
    if(amr_step_level>=0) c=amr_level_cells(target,amr_step_level)
    if(.not.active_cell(c)) cycle
    next(secondary_state_index(),c)=secondary_low_state(c)
    do k=offset(c),offset(c+1)-1
     f=faces(k);next(secondary_state_index(),c)=next(secondary_state_index(),c)- &
      dt*signs(k)*length(f)*secondary_anti_flux(f)/area(c)
    enddo
    scale=512*epsilon(1.d0)*max(1.d0,abs(next(secondary_state_index(),c)))
    if(next(1,c)<=0.d0) then
     if(abs(next(secondary_state_index(),c))<=scale) then
      next(secondary_state_index(),c)=0.d0
     else
      stage_failure=6
     endif
    endif
   enddo
   !$omp end parallel do
   if(stage_failure>0) then
    failure_reason=stage_failure;return
   endif
  endif
  if(profile_enabled) then
   profile_euler_limit_s=profile_euler_limit_s+omp_get_wtime()-profile_phase_start
   profile_phase_start=omp_get_wtime()
  endif
  if(amr_step_level<0) then
   call redistribute_eroded_banks(state,next)
   if(.not.bank_failure) call avalanche_erodible_banks(next,dt)
  endif
  call rebalance_active_layer(next)
  ! Positivity limiting can leave a negative subnormal after its final
  ! multiply/subtract on platforms that preserve denormals.  Such a value is
  ! arithmetic zero (far below any representable physical inventory), and
  ! timestep halving cannot change it.  Remove only this underflow band; every
  ! normal negative inventory remains a rejected step.
  where(next(4:nphysical_state,:)<0.d0.and.abs(next(4:nphysical_state,:))<=128*tiny(1.d0)) &
   next(4:nphysical_state,:)=0.d0
  if(.not.all(ieee_is_finite(next)).or.minval(next(4:nphysical_state,:))<0) then
   if(profile_enabled) profile_euler_finalize_s=profile_euler_finalize_s+omp_get_wtime()-profile_phase_start
   amr_failure_state=3+minloc(minval(next(4:nphysical_state,:),dim=2),dim=1)
   amr_failure_cell=minloc(next(amr_failure_state,:),dim=1)
   amr_failure_value=next(amr_failure_state,amr_failure_cell)
   amr_failure_aux=area(amr_failure_cell)
   failure_reason=5;return
  endif
  do work=1,nwork
   f=work
   if(amr_step_level>=0) f=amr_level_faces(work,amr_step_level)
   b=bid(f)
   if(b==0) cycle
   if(.not.amr_updates_cell(owner(f))) cycle
   accepted(1,b)=accepted(1,b)-dt*length(f)*fl(1,f)
   if(present(boundary_face_transport_integral)) &
    boundary_face_transport_integral(1,f)=dt*length(f)*fl(1,f)
   do g=1,nfrac
    ! Public physical ledgers are inward-positive and contain the accepted
    ! total (advective FCT plus horizontal diffusion).  The v48 diagnostic
    ! rows retain the two signed contributions separately.
    accepted(1+g,b)=accepted(1+g,b)-dt*length(f)* &
     (fl(suspended_index(g),f)+suspended_diffusive_flux(g,f))
    accepted(1+nfrac+g,b)=accepted(1+nfrac+g,b)-dt*length(f)*fl(bed_index(g),f)
    if(flow2dh_boundary_diffusion_declaration) then
     accepted(suspended_advective_ledger_index(g),b)= &
      accepted(suspended_advective_ledger_index(g),b)- &
      dt*length(f)*fl(suspended_index(g),f)
     accepted(suspended_diffusive_ledger_index(g),b)= &
      accepted(suspended_diffusive_ledger_index(g),b)- &
      dt*length(f)*suspended_diffusive_flux(g,f)
     if(present(boundary_face_transport_integral)) &
      boundary_face_transport_integral(1+g,f)= &
       dt*length(f)*fl(suspended_index(g),f)
    endif
   enddo
   if(prognostic_secondary_flow()) accepted(secondary_ledger_index(),b)= &
    accepted(secondary_ledger_index(),b)-dt*length(f)*fl(secondary_state_index(),f)
  enddo
  if(present(momentum_boundary_impulse)) then
   ! Column 1 is the ordinary finite-volume boundary impulse (including wall
   ! and open-boundary pressure/advective traction); column 2 isolates the
   ! optional secondary-stress traction. Internal faces are deliberately absent.
   do work=1,nwork
    f=work
    if(amr_step_level>=0) f=amr_level_faces(work,amr_step_level)
    if(neighbour(f)>0.or..not.amr_updates_cell(owner(f))) cycle
    c=owner(f)
    effective_momentum_flux=fl(2:3,f)-.5d0*grav*state(1,c)**2*normal(:,f)
    if(prognostic_secondary_flow()) then
     momentum_boundary_impulse(:,1)=momentum_boundary_impulse(:,1)-dt*length(f)* &
      (effective_momentum_flux-secondary_stress_face(:,f))
     momentum_boundary_impulse(:,2)=momentum_boundary_impulse(:,2)-dt*length(f)* &
      secondary_stress_face(:,f)
    else
     momentum_boundary_impulse(:,1)=momentum_boundary_impulse(:,1)-dt*length(f)*effective_momentum_flux
    endif
   enddo
  endif
  if(present(integral)) then
   integral=0
   do work=1,nwork
    f=work
    if(amr_step_level>=0) f=amr_level_faces(work,amr_step_level)
    integral(:,1,f)=dt*length(f)*fl(:,f);integral(:,2,f)=dt*length(f)*fr(:,f)
    if(flow2dh_boundary_diffusion_declaration) then
     do g=1,nfrac
      si=suspended_index(g)
      integral(si,1,f)=integral(si,1,f)+ &
       dt*length(f)*suspended_diffusive_flux(g,f)
      integral(si,2,f)=integral(si,2,f)+ &
       dt*length(f)*suspended_diffusive_flux(g,f)
     enddo
    endif
   enddo
  endif
  if(present(toe_flux_weight).and.bank_toe_erosion_mode==2) then
   if(.not.bank_toe_mapping_ready) error stop 'physical bank-toe flux requested before geometry mapping'
   call braids_bank_toe_add_face_rates(nf,nfrac,nstate,bed_index(1),owner,neighbour, &
    physical_is_toe_face,physical_candidate_bank,physical_candidate_toe,length,dt,toe_flux_weight, &
    fl,fr,physical_toe_flux_trial,toe_flux_error)
   if(toe_flux_error/=BANK_TOE_FLUX_SUCCESS) error stop 'invalid accepted physical bank-toe stage flux'
  endif
  if(present(toe_flux_weight).and.bank_toe_erosion_mode==3) then
   if(.not.allocated(distributed_face_mass_trial)) &
    error stop 'distributed bank face ledger is not allocated'
   ! Retain the signed, final donor-limited Exner mass from every SSPRK stage.
   ! Taking an absolute value here would define a different gross-transport
   ! law and could not be reconciled consistently across AMR levels.
   !$omp parallel do schedule(static,256) if(parallel_faces) default(shared) private(work,f,g,bi)
   do work=1,nwork
    f=work
    if(amr_step_level>=0) f=amr_level_faces(work,amr_step_level)
    do g=1,nfrac
     bi=bed_index(g)
     distributed_face_mass_trial(g,1,f)=distributed_face_mass_trial(g,1,f)+ &
      toe_flux_weight*dt*length(f)*fl(bi,f)
     distributed_face_mass_trial(g,2,f)=distributed_face_mass_trial(g,2,f)+ &
      toe_flux_weight*dt*length(f)*fr(bi,f)
    enddo
   enddo
   !$omp end parallel do
  endif
  if(present(instantaneous_face_flux)) instantaneous_face_flux=fl
  ok=.true.
  if(profile_enabled) profile_euler_finalize_s=profile_euler_finalize_s+omp_get_wtime()-profile_phase_start
 end subroutine

 function flow2dh_open_boundary_concentration(fraction,face,water_flux,dt, &
  prescribed,interior) result(value)
  ! Locked FLOW order: INCBCC prescription, Thatcher--Harleman return-time
  ! rule, then the global NeuBcSand adjacent-interior override.  The native
  ! face flux is positive outward from the owner cell, opposite to the
  ! inward-positive public oracle convention.
  integer,intent(in) :: fraction,face
  real(8),intent(in) :: water_flux,dt,prescribed,interior
  real(8) :: value,return_time,ratio
  integer :: boundary,memory_face
  boundary=bid(face);value=prescribed
  if(.not.flow2dh_boundary_diffusion_declaration.or.boundary<=0) return
  memory_face=flow2dh_memory_face(face)
  return_time=suspended_boundary_return_time(boundary)
  if(return_time<0.d0) then
   value=max(0.d0,interior)
  else if(return_time>0.d0) then
   if(water_flux>0.d0) then
    value=max(0.d0,interior)
   else if(water_flux<0.d0) then
    ratio=max(suspended_boundary_return_clock(fraction,memory_face)-dt,0.d0)/ &
     return_time
    ratio=max(0.d0,min(1.d0,ratio))
    value=suspended_boundary_last_outflow(fraction,memory_face)+ &
     .5d0*(cos(acos(-1.d0)*ratio)+1.d0)* &
     (prescribed-suspended_boundary_last_outflow(fraction,memory_face))
   endif
  endif
 if(flow2dh_neubc_sand) value=max(0.d0,interior)
 end function

 subroutine commit_flow2dh_boundary_memory(dt,face_transport_integral,base_state)
  ! Commit Thatcher--Harleman state exactly once, after the complete SSPRK
  ! macro step has been accepted.  Rejected Euler trials and cooperative
  ! cancellation therefore leave restart memory untouched.  Integrals use
  ! the native outward-positive orientation.  Direction is the accepted
  ! macro-step SSPRK face-volume direction, while FLOW's stored value remains
  ! the adjacent pre-transport r0 concentration from the accepted base state;
  ! it is deliberately not reconstructed from a limited mass/water ratio.
  real(8),intent(in) :: dt,face_transport_integral(1+nfrac,nf)
  real(8),intent(in) :: base_state(nstate,nc)
  real(8) :: water_volume,return_time
  integer :: f,g,b,memory_face
  if(.not.flow2dh_boundary_diffusion_declaration) return
  do f=1,nf
   b=bid(f)
   if(b<=0.or.neighbour(f)>0) cycle
   memory_face=flow2dh_memory_face(f)
   return_time=suspended_boundary_return_time(b)
   if(return_time<0.d0) then
    suspended_boundary_return_clock(:,memory_face)=0.d0
    cycle
   endif
   if(return_time==0.d0) cycle
   water_volume=face_transport_integral(1,f)
   ! FLOW branches on the signed normal transport without a dead band.  Keep
   ! that exact sign convention here as well: a small accepted outflow still
   ! refreshes memory, a small accepted return still advances its clock, and
   ! only an exactly zero macro-step transport resets the clock.
   if(water_volume>0.d0) then
    do g=1,nfrac
     suspended_boundary_last_outflow(g,memory_face)=0.d0
     if(base_state(1,owner(f))>dry_tolerance) &
      suspended_boundary_last_outflow(g,memory_face)=max(0.d0, &
       base_state(suspended_index(g),owner(f))/base_state(1,owner(f)))
     suspended_boundary_return_clock(g,memory_face)=return_time
    enddo
   else if(water_volume<0.d0) then
    suspended_boundary_return_clock(:,memory_face)=max( &
     suspended_boundary_return_clock(:,memory_face)-dt,0.d0)
   else
    ! FLOW resets the clock for zero normal boundary velocity while leaving
    ! the current prescribed concentration in force for this invocation.
    suspended_boundary_return_clock(:,memory_face)=0.d0
   endif
  enddo
 end subroutine

 subroutine validate_flow2dh_boundary_memory_checkpoint()
  ! A checkpoint is restart authority, so reject states the accepted FLOW
  ! return-time transaction cannot produce.  Positive return-time clocks may
  ! reach T exactly after outflow; the small upper tolerance only admits round
  ! off from serialization.  Disabled/reflected (T<=0) modes never own a
  ! running clock and therefore require exact zero.
  real(8) :: return_time,tolerance
  integer :: f,b,memory_count
  if(.not.flow2dh_boundary_diffusion_declaration) return
  if(.not.all(ieee_is_finite(suspended_boundary_last_outflow)).or. &
     .not.all(ieee_is_finite(suspended_boundary_return_clock)).or. &
     any(suspended_boundary_last_outflow<0.d0).or. &
     any(suspended_boundary_return_clock<0.d0)) &
   error stop 'invalid suspended boundary-memory checkpoint'
  memory_count=size(suspended_boundary_last_outflow,2)
  do f=1,memory_count
   if(amr_enabled) then
    b=af_bid(f)
    if(b<=0.or.af_neighbour(f)>0) then
     if(any(suspended_boundary_return_clock(:,f)/=0.d0)) &
      error stop 'suspended boundary-memory checkpoint clock is active on a non-open face'
     cycle
    endif
   else
    b=bid(f)
   endif
   if(b<=0.or.(.not.amr_enabled.and.neighbour(f)>0)) then
    if(any(suspended_boundary_return_clock(:,f)/=0.d0)) &
     error stop 'suspended boundary-memory checkpoint clock is active on a non-open face'
    cycle
   endif
   return_time=suspended_boundary_return_time(b)
   if(return_time>0.d0) then
    tolerance=256.d0*epsilon(1.d0)*max(1.d0,return_time)
    if(any(suspended_boundary_return_clock(:,f)>return_time+tolerance)) &
     error stop 'suspended boundary-memory checkpoint clock exceeds return time'
   else
    if(any(suspended_boundary_return_clock(:,f)/=0.d0)) &
     error stop 'suspended boundary-memory checkpoint clock is active for nonpositive return time'
   endif
  enddo
 end subroutine

 subroutine bedform_properties(state,cell,effective_n,dune_height,dune_length,total_ks,iterations,valid)
  real(8),intent(in) :: state(nstate,nc)
  integer,intent(in) :: cell
  real(8),intent(out) :: effective_n,dune_height,dune_length,total_ks
  integer,intent(out) :: iterations
  logical,intent(out) :: valid
  real(8) :: volume(nfrac),mixture_d50,mixture_d90,relative_density,grain_ks,dune_ks,total
  integer :: g
  logical :: percentile_valid
  effective_n=manning_cell(cell);dune_height=0;dune_length=0;total_ks=0;iterations=0;valid=.true.
  if(.not.bedform_roughness.or.state(1,cell)<=dry_tolerance) return
  do g=1,nfrac
   volume(g)=max(0.d0,state(bed_index(g),cell))/bulk_fraction(g)
  enddo
  total=sum(volume)
  if(total<=0) return
  call braids_mixture_percentile_default(volume,d50_fraction,.5d0,mixture_d50,percentile_valid)
  if(.not.percentile_valid) then
   valid=.false.;return
  endif
  call braids_mixture_percentile_default(volume,d50_fraction,.9d0,mixture_d90,percentile_valid)
  if(.not.percentile_valid) then
   valid=.false.;return
  endif
  relative_density=sum(volume*(rhos_fraction/rho-1.d0))/total
  call braids_fredsoe_vanrijn84_roughness(state(1,cell),state(2,cell)/state(1,cell), &
   state(3,cell)/state(1,cell),manning_cell(cell),mixture_d50,mixture_d90,relative_density, &
   mpm_critical,effective_n,dune_height,dune_length,grain_ks,dune_ks,total_ks,iterations,valid)
 end subroutine

 subroutine coupled_stage_characteristic(f,c,state,left,zleft,stage,hiding,d90,morfac,trace,ztrace,valid)
  integer,intent(in) :: f,c
  real(8),intent(in) :: state(nstate,nc),left(3),zleft,stage,hiding(nfrac),d90,morfac
  real(8),intent(inout) :: trace(3)
  real(8),intent(out) :: ztrace
  logical,intent(out) :: valid
  real(8) :: h,u,v,tangent(2),derivative(3),jacobian(3),b,cc,d,p,q,argument,angle,roots(3),lambda,ratio,dh,candidate(3)
  integer :: g,k
  valid=.false.;ztrace=zleft;h=left(1)
  if(h<=dry_tolerance.or.morfac<=0) return
  tangent=[-normal(2,f),normal(1,f)]
  u=dot_product(left(2:3),normal(:,f))/h;v=dot_product(left(2:3),tangent)/h
  if(u<=0.or.u>=sqrt(grav*h)) return
  derivative=0
  do g=1,nfrac
   call mpm_mass_jacobian(h,u,v,hydraulic_equivalent_manning(h,manning_cell(c)),d50_fraction(g),d90, &
    rhos_fraction(g)/rho-1.d0,rhos_fraction(g),grav,mpm_critical,hiding(g), &
    mpm_cf,mpm_friction_mode==1,bedload_min_depth_fraction(g),dry_tolerance,jacobian)
   derivative=derivative+morfac*bed_availability(state,g,c)/bulk_fraction(g)*jacobian
  enddo
  b=-2*u;cc=u*u-grav*h*(1+derivative(2));d=-grav*h*(derivative(1)+v*derivative(3))
  p=cc-b*b/3;q=2*b*b*b/27-b*cc/3+d
  if(p>=0) return
  argument=3*q/(2*p)*sqrt(-3/p)
  if(abs(argument)>1+1.d-10) return
  angle=acos(max(-1.d0,min(1.d0,argument)))/3
  do k=1,3
   roots(k)=2*sqrt(-p/3)*cos(angle-2*acos(-1.d0)*(k-1)/3)-b/3
  enddo
  lambda=minval(roots)
  if(lambda>=0) return
  ! Incoming eigenvector has [dh,dqn,dqt,dz]=[1,lambda,v,ratio-1].
  ! The only imposed datum is eta=h+z; outgoing modal content is retained.
  ratio=(lambda-u)**2/(grav*h);dh=(stage-h-zleft)/ratio
  if(h+dh<=dry_tolerance) return
  candidate=[h+dh,(h*u+lambda*dh)*normal(:,f)+(h*v+v*dh)*tangent]
  if(dot_product(candidate(2:3),normal(:,f))<=0) return
  if(.not.all(ieee_is_finite(candidate))) return
  trace=candidate;ztrace=zleft+(ratio-1)*dh;valid=.true.
 end subroutine

 subroutine coupled_mpm_face(f,state,l,r,bl,br,hiding,d90mix,morfac,fl,fr,speed)
  integer,intent(in) :: f
  real(8),intent(in) :: state(nstate,nc),l(3),r(3),bl,br,hiding(nfrac,nc),d90mix(nc),morfac
  real(8),intent(inout) :: fl(nstate),fr(nstate),speed
  real(8) :: nvec(2),tvec(2),rotl(3),rotr(3),meanstate(3),derivative(3)
  real(8) :: leftq(nfrac),rightq(nfrac),weight(nfrac),slope(nfrac),qx,qy,rx,ry,tr,donor_weight
  real(8) :: plus,minus,value,base,fluxl(4),fluxr(4),oldl(3),oldr(3),oldspeed,new_speed,qsleft,qsright,jacobian(3)
  integer :: c,n,g
  logical :: valid
  if(min(l(1),r(1))<=dry_tolerance.or.morfac<=0.or.bedlaw/=2) return
  if(.not.bedload_face_open(l,r,bl,br,normal(:,f))) return
  c=owner(f);n=neighbour(f);nvec=normal(:,f);tvec=[-nvec(2),nvec(1)]
  ! Preserve the prescribed-flux closure at non-stage open boundaries.
  ! Stage-adjacent faces retain distinct, bounded capacity traces and the
  ! coupled interior waves. Equal midpoint traces suppress the sediment-flux
  ! jump and can amplify grid-scale disturbances as a bed wave exits.
  if(min(uncoupled_boundary_distance(c),uncoupled_boundary_distance(n))<=2) return
  if(min(open_boundary_distance(c),open_boundary_distance(n))<=2) then
   ! A stage condition on supercritical outflow is a free release. Preserve
   ! its existing equivalence to the explicit supercritical boundary mode.
   if(abs(dot_product(l(2:3),nvec))>=l(1)*sqrt(grav*l(1)).and. &
      abs(dot_product(r(2:3),nvec))>=r(1)*sqrt(grav*r(1))) return
  endif
  rotl=[l(1),dot_product(l(2:3),nvec),dot_product(l(2:3),tvec)]
  rotr=[r(1),dot_product(r(2:3),nvec),dot_product(r(2:3),tvec)]
  meanstate=.5d0*(rotl+rotr)
  derivative=0;qsleft=0;qsright=0
  do g=1,nfrac
   call bedload_capacity_face(c,f,g,bedq_cell,bedq_grad,qx,qy,coupled_stage_capacity(f,l,r))
   call bedload_capacity_face(n,f,g,bedq_cell,bedq_grad,rx,ry,coupled_stage_capacity(f,l,r))
   plus=dot_product([qx,qy],nvec);minus=dot_product([rx,ry],nvec)
   if(plus>0.and.minus>0) then
    base=plus*bed_availability(state,g,c)
   else if(plus<0.and.minus<0) then
    base=minus*bed_availability(state,g,n)
   else
    base=.5d0*(plus+minus)
    donor_weight=bed_availability(state,g,c)
    if(base<0) donor_weight=bed_availability(state,g,n)
    base=base*donor_weight
   endif
   slope(g)=fl(bed_index(g))-base
   leftq(g)=plus*bed_availability(state,g,c)
   rightq(g)=minus*bed_availability(state,g,n)
   weight(g)=.5d0*(bed_availability(state,g,c)+bed_availability(state,g,n))
   qsleft=qsleft+leftq(g)/bulk_fraction(g)*morfac
   qsright=qsright+rightq(g)/bulk_fraction(g)*morfac
    call mpm_mass_jacobian(meanstate(1),meanstate(2)/meanstate(1),meanstate(3)/meanstate(1), &
     hydraulic_equivalent_manning(meanstate(1),.5d0*(manning_cell(c)+manning_cell(n))), &
     d50_fraction(g),.5d0*(d90mix(c)+d90mix(n)), &
    rhos_fraction(g)/rho-1.d0,rhos_fraction(g),grav,mpm_critical,.5d0*(hiding(g,c)+hiding(g,n)), &
    mpm_cf,mpm_friction_mode==1,bedload_min_depth_fraction(g),dry_tolerance,jacobian)
   derivative=derivative+morfac*weight(g)/bulk_fraction(g)*jacobian
  enddo
  if(maxval(abs(derivative))<1.d-16.or.sum(weight)<=0) return
  call coupled_exner_flux(rotl,rotr,bl,br,qsleft,qsright,derivative,grav,fluxl,fluxr,new_speed,valid)
  if(.not.valid) return
   call wave_flux(l,r,bl,br,nvec,oldl,oldr,oldspeed)
   fl(1:3)=fl(1:3)+[fluxl(1),fluxl(2)*nvec+fluxl(3)*tvec]-oldl
   fr(1:3)=fr(1:3)+[fluxr(1),fluxr(2)*nvec+fluxr(3)*tvec]-oldr
  speed=max(speed,new_speed)
  value=(fluxl(4)-.5d0*(qsleft+qsright))/morfac
  do g=1,nfrac
   fl(bed_index(g))=.5d0*(leftq(g)+rightq(g))+value*weight(g)/sum(weight)*bulk_fraction(g)+slope(g)
  enddo
 end subroutine

 subroutine coupled_gaia_face(f,state,l,r,bl,br,morfac,fl,fr,speed)
  ! Couple the formula-3 hydraulic base capacity to the SWE eigenstructure.
  ! SLOPEFF=1, DEVIA=2 and BEDLOAD_SECCURRENT remain the already evaluated
  ! physical residual, so this changes numerical wave splitting rather than
  ! the qualified TELEMAC-GAIA constitutive equations.
  integer,intent(in) :: f
  real(8),intent(in) :: state(nstate,nc),l(3),r(3),bl,br,morfac
  real(8),intent(inout) :: fl(nstate),fr(nstate),speed
  real(8) :: nvec(2),tvec(2),rotl(3),rotr(3),meanstate(3),derivative(3)
  real(8) :: leftq(nfrac),rightq(nfrac),weight(nfrac),residual(nfrac)
  real(8) :: qx,qy,rx,ry,plus,minus,base,donor_weight
  real(8) :: fluxl(4),fluxr(4),oldl(3),oldr(3),oldspeed,new_speed
  real(8) :: qsleft,qsright,jacobian(3),value
  integer :: c,n,g
  logical :: valid

  if(min(l(1),r(1))<=dry_tolerance.or.morfac<=0.d0.or.bedlaw/=9) return
  if(.not.bedload_face_open(l,r,bl,br,normal(:,f))) return
  c=owner(f);n=neighbour(f);nvec=normal(:,f);tvec=[-nvec(2),nvec(1)]
  ! Preserve the prescribed hydraulic/sediment inlet closure.  A stage
  ! outlet retains coupled interior waves until both traces are supercritical.
  if(min(uncoupled_boundary_distance(c),uncoupled_boundary_distance(n))<=2) return
  if(min(open_boundary_distance(c),open_boundary_distance(n))<=2) then
   if(abs(dot_product(l(2:3),nvec))>=l(1)*sqrt(grav*l(1)).and. &
      abs(dot_product(r(2:3),nvec))>=r(1)*sqrt(grav*r(1))) return
  endif
  rotl=[l(1),dot_product(l(2:3),nvec),dot_product(l(2:3),tvec)]
  rotr=[r(1),dot_product(r(2:3),nvec),dot_product(r(2:3),tvec)]
  meanstate=.5d0*(rotl+rotr)
  derivative=0.d0;qsleft=0.d0;qsright=0.d0
  do g=1,nfrac
   call bedload_capacity_face(c,f,g,bedq_gaia_base,bedq_gaia_base_grad,qx,qy,.false.)
   call bedload_capacity_face(n,f,g,bedq_gaia_base,bedq_gaia_base_grad,rx,ry,.false.)
   plus=dot_product([qx,qy],nvec);minus=dot_product([rx,ry],nvec)
   if(plus>0.d0.and.minus>0.d0) then
    base=plus*bed_availability(state,g,c)
   else if(plus<0.d0.and.minus<0.d0) then
    base=minus*bed_availability(state,g,n)
   else
    base=.5d0*(plus+minus)
    donor_weight=bed_availability(state,g,c)
    if(base<0.d0) donor_weight=bed_availability(state,g,n)
    base=base*donor_weight
   endif
   residual(g)=fl(bed_index(g))-base
   leftq(g)=plus*bed_availability(state,g,c)
   rightq(g)=minus*bed_availability(state,g,n)
   weight(g)=.5d0*(bed_availability(state,g,c)+bed_availability(state,g,n))
   qsleft=qsleft+morfac*leftq(g)/bulk_fraction(g)
   qsright=qsright+morfac*rightq(g)/bulk_fraction(g)
    call gaia_mass_jacobian(meanstate(1),meanstate(2)/meanstate(1),meanstate(3)/meanstate(1), &
     hydraulic_equivalent_manning(meanstate(1),.5d0*(manning_cell(c)+manning_cell(n))), &
     d50_fraction(g),rhos_fraction(g),rho,grav, &
    gaia_skin_roughness_ratio,bedload_min_depth_fraction(g),dry_tolerance,jacobian)
   derivative=derivative+morfac*weight(g)*jacobian/bulk_fraction(g)
  enddo
  if(maxval(abs(derivative))<1.d-16.or.sum(weight)<=0.d0) return
  call coupled_exner_flux(rotl,rotr,bl,br,qsleft,qsright,derivative,grav, &
   fluxl,fluxr,new_speed,valid)
  if(.not.valid) return
  call wave_flux(l,r,bl,br,nvec,oldl,oldr,oldspeed)
  fl(1:3)=fl(1:3)+[fluxl(1),fluxl(2)*nvec+fluxl(3)*tvec]-oldl
  fr(1:3)=fr(1:3)+[fluxr(1),fluxr(2)*nvec+fluxr(3)*tvec]-oldr
  speed=max(speed,new_speed)
  value=(fluxl(4)-.5d0*(qsleft+qsright))/morfac
  do g=1,nfrac
   fl(bed_index(g))=.5d0*(leftq(g)+rightq(g))+ &
    value*weight(g)/sum(weight)*bulk_fraction(g)+residual(g)
  enddo
 end subroutine

 subroutine compose_live_law11_channels(hydraulic_state,concentration_depth,fraction, &
  raw_bed,raw_suspended,composed_bed,retained_suspended,signed_suspended_rate, &
  equilibrium_concentration,valid,status)
  ! Single live adapter for the accepted post-density Delft channel ordering.
  ! `raw_bed` may already have received bed-only secondary-flow steering;
  ! `raw_suspended` must remain aligned with the depth-averaged velocity.
  real(8),intent(in) :: hydraulic_state(3),concentration_depth
  integer,intent(in) :: fraction
  real(8),intent(in) :: raw_bed(2),raw_suspended(2)
  real(8),intent(out) :: composed_bed(2),retained_suspended(2), &
   signed_suspended_rate,equilibrium_concentration
  logical,intent(out) :: valid
  integer,intent(out) :: status
  real(8) :: base_velocity(2)
  composed_bed=0.d0;retained_suspended=0.d0
  signed_suspended_rate=0.d0;equilibrium_concentration=0.d0
  valid=.false.;status=-1
  if(.not.channel_composition_declaration) return
  if(fraction<1.or.fraction>nfrac) return
  ! FLOW's class-3 algebraic composer is undefined at nonpositive depth, but
  ! an ordinary dry cell with identically-zero constitutive channels has the
  ! exact zero solution and is skipped before entering the strict operator.
  if(channel_transport_class_input(fraction)==BRAIDS_TRA_COMBINE.and. &
     hydraulic_state(1)<=dry_tolerance.and. &
     all(raw_bed==0.d0).and.all(raw_suspended==0.d0)) then
   valid=.true.;status=BRAIDS_COMPOSE_OK
   return
  endif
  base_velocity=0.d0
  if(hydraulic_state(1)>dry_tolerance) &
   base_velocity=hydraulic_state(2:3)/hydraulic_state(1)
  call braids_compose_delft_channels(raw_bed,raw_suspended,base_velocity, &
   concentration_depth,channel_bed_factor,channel_suspended_factor, &
   channel_transport_class_input(fraction),channel_speed_epsilon, &
   composed_bed,retained_suspended,signed_suspended_rate, &
   equilibrium_concentration,valid,status)
 end subroutine

 subroutine prepare_bedload_capacity(state,hiding,mixture_dg,mixture_gsd,mixture_sand,mixture_d90,mixture_mud, &
  boundary_bed_input,capacity,tau,gradient)
  ! Evaluate each nonlinear formula once at cell centres, as Delft3D does
  ! before transferring capacity to faces.  Reconstruction operates on this
  ! derived field and therefore cannot create a second nonlinear evaluation
  ! from independently reconstructed hydraulic traces.
  real(8), intent(in) :: state(nstate,nc),hiding(nfrac,nc)
  real(8), intent(in) :: mixture_dg(nc),mixture_gsd(nc),mixture_sand(nc),mixture_d90(nc),mixture_mud(nc)
  real(8), intent(in) :: boundary_bed_input(nfrac,nb)
  real(8), intent(out) :: capacity(2,nfrac,nc),tau(nfrac,nc),gradient(2,2,nfrac,nc)
  real(8) :: qx,qy,sqx,sqy,ratio,spiral_intensity,spiral_alpha, &
   effective_n,formula_n,dune_height,dune_length,total_ks
  real(8) :: absolute_shields_value,critical_shields_value,transport_stage_value
  real(8) :: raw_bed(2),raw_suspended(2),composed_bed(2),retained_suspended(2), &
   signed_suspended_rate,equilibrium_concentration
  real(8) :: batch_qx(nfrac),batch_qy(nfrac),batch_tau(nfrac),capacity_state(3),minimum_depth,wet_fraction,xi
  real(8) :: gaia_base_x,gaia_base_y,gaia_total_x,gaia_total_y,gaia_factor
  real(8) :: gaia_qmag,gaia_h,gaia_u,gaia_v
  real(8),allocatable :: gaia_bed(:),gaia_eta(:)
  real(8) :: profile_cell_start
  integer :: c,g,invalid,work,nwork,bedform_iterations,composition_status
  logical :: valid,batch_mpm
  capacity=0;tau=0;gradient=0
  if(capacity_split_retained) then
   suspended_capacity_cell=0.d0;suspended_capacity_equilibrium=0.d0
   capacity_absolute_shields=0.d0;capacity_critical_shields=0.d0;capacity_transport_stage=0.d0
  endif
  if(channel_composition_declaration) then
   channel_raw_bed_cell=0.d0;channel_raw_suspended_cell=0.d0
   channel_composed_bed_cell=0.d0;channel_retained_suspended_cell=0.d0
   channel_signed_suspended_rate_cell=0.d0
   channel_equilibrium_concentration_cell=0.d0
  endif
  if(bedlaw==0) return
  if(secondary_flow_model==1) call secondary_flow_curvatures(state,secondary_curvature)
  if(bedlaw==9) then
   if(.not.allocated(gaia_bed_down).or..not.allocated(gaia_eta_gradient)) &
    error stop 'TELEMAC-GAIA gradient work is not allocated'
    allocate(gaia_bed(nc),gaia_eta(nc))
    bedq_gaia_base=0.d0;bedq_gaia_base_grad=0.d0
   call effective_bed_elevation(state,gaia_bed)
   call bed_downslopes(gaia_bed,gaia_bed_down)
   gaia_eta=gaia_bed+state(1,:)
   call bed_downslopes(gaia_eta,gaia_eta_gradient)
   gaia_eta_gradient=-gaia_eta_gradient
  endif
  invalid=0;nwork=nc
  batch_mpm=bedlaw==2.and.mpm_friction_mode==1.and. &
   .not.capacity_foundation_runtime_declaration.and.all(sediment_type_fraction/=1).and. &
   hydraulic_roughness_mode==braids_roughness_manning.and.roughness_minimum_work_depth==0.d0.and. &
   all(bedload_min_depth_fraction==bedload_min_depth_fraction(1))
  if(amr_step_level>=0) nwork=amr_level_eval_cell_count(amr_step_level)
  if(profile_enabled) profile_cell_start=omp_get_wtime()
  ! A separate OpenMP team at every Runge--Kutta stage costs more than this
  ! work on compact validation meshes.  Keep the production path parallel,
  ! while executing this derived-field loop serially below the measured
  ! crossover.  OMP_NUM_THREADS still controls all larger loops/cases.
  !$omp parallel do schedule(static,256) if(nwork*nfrac>=4096) default(shared) &
  !$omp& private(work,c,g,qx,qy,sqx,sqy,ratio,absolute_shields_value,critical_shields_value,transport_stage_value,spiral_intensity,spiral_alpha,effective_n,formula_n,dune_height,dune_length,total_ks,bedform_iterations,valid,batch_qx,batch_qy,batch_tau,capacity_state,minimum_depth,wet_fraction,xi,gaia_base_x,gaia_base_y,gaia_total_x,gaia_total_y,gaia_factor,gaia_qmag,gaia_h,gaia_u,gaia_v,raw_bed,raw_suspended,composed_bed,retained_suspended,signed_suspended_rate,equilibrium_concentration,composition_status) reduction(max:invalid)
  do work=1,nwork
   c=work
   if(amr_step_level>=0) c=amr_level_eval_cells(work,amr_step_level)
   if(.not.active_cell(c)) cycle
   call bedform_properties(state,c,effective_n,dune_height,dune_length,total_ks,bedform_iterations,valid)
   if(.not.valid) then
     invalid=1;cycle
    endif
    formula_n=effective_n
    if(state(1,c)>dry_tolerance) formula_n=hydraulic_equivalent_manning(state(1,c),effective_n)
    if(batch_mpm) then
     if(state(1,c)<=dry_tolerance) cycle
     capacity_state=state(1:3,c);wet_fraction=1.d0
     minimum_depth=max(bedload_min_depth_fraction(1),dry_tolerance)
     if(state(1,c)<minimum_depth.and.minimum_depth>dry_tolerance) then
      xi=max(0.d0,min(1.d0,(state(1,c)-dry_tolerance)/(minimum_depth-dry_tolerance)))
      wet_fraction=xi*xi*(3.d0-2.d0*xi)
      capacity_state(1)=minimum_depth*(1.d0+128*epsilon(1.d0))
      capacity_state(2:3)=state(2:3,c)*capacity_state(1)/state(1,c)
     endif
     if(capacity_route_identity_declaration.and.trim(capacity_route_method_id)=='capacity.mpm') then
      call braids_delft_mpm_bedload_roughness_batch(capacity_state(1), &
       capacity_state(2)/capacity_state(1),capacity_state(3)/capacity_state(1),effective_n, &
       d50_fraction,mixture_d90(c),rhos_fraction/rho-1.d0,grav,hiding(:,c), &
       batch_qx,batch_qy,valid,batch_tau)
     else
      call braids_mpm_bedload_roughness_batch(capacity_state(1),capacity_state(2)/capacity_state(1), &
       capacity_state(3)/capacity_state(1),effective_n,d50_fraction,mixture_d90(c), &
       rhos_fraction/rho-1.d0,grav,mpm_critical,hiding(:,c),batch_qx,batch_qy,valid,batch_tau)
     endif
     do g=1,nfrac
      qx=rhos_fraction(g)*batch_qx(g);qy=rhos_fraction(g)*batch_qy(g)
      qx=wet_fraction*qx;qy=wet_fraction*qy;ratio=batch_tau(g)
      if(valid.and.bed_slope==3) &
       call braids_delft3d_total_shields(capacity_state(2)/capacity_state(1), &
        capacity_state(3)/capacity_state(1),capacity_state(1)**(1.d0/6.d0)/effective_n, &
        d50_fraction(g),rhos_fraction(g)/rho-1.d0,0.d0,ratio,valid)
      spiral_alpha=0.d0
      if(valid.and.secondary_bedload_direction().and.state(1,c)>dry_tolerance.and.qx*qx+qy*qy>0) then
       if(classic_secondary_foundation()) then
        call rotate_classic_secondary_bed(state(1:3,c),c,[qx,qy],raw_bed, &
         spiral_alpha,valid)
        if(valid) then
         qx=raw_bed(1);qy=raw_bed(2)
        endif
       else if(secondary_flow_model==1) then
        call braids_delft3d_spiral_transport(state(1,c),state(2,c)/state(1,c), &
         state(3,c)/state(1,c),effective_n,grav,secondary_flow_espir,secondary_curvature(c),qx,qy, &
         spiral_intensity,spiral_alpha,valid,engelund_hansen_chezy)
       else
        spiral_intensity=state(secondary_state_index(),c)/state(1,c)
        call braids_spiral_transport_from_intensity(state(1,c),state(2,c)/state(1,c), &
         state(3,c)/state(1,c),effective_n,grav,secondary_flow_espir,spiral_intensity,qx,qy, &
         spiral_alpha,valid,engelund_hansen_chezy)
       endif
      endif
      if(valid.and.bed_slope==3) ratio=ratio*sqrt(1.d0+spiral_alpha*spiral_alpha)
      if(.not.valid.or..not.all(ieee_is_finite([qx,qy,ratio]))) then
       invalid=1
      else
       capacity(:,g,c)=[qx,qy];tau(g,c)=ratio
      endif
     enddo
    else
     do g=1,nfrac
      sqx=0.d0;sqy=0.d0;absolute_shields_value=0.d0
      critical_shields_value=0.d0;transport_stage_value=0.d0
      call bed_capacity(state(1:3,c),g,qx,qy,ratio,valid,hiding(g,c),effective_n, &
       mixture_dg(c),mixture_gsd(c),mixture_sand(c),mixture_d90(c),mixture_mud(c),sqx,sqy, &
       absolute_shields_value,critical_shields_value,transport_stage_value)
      raw_suspended=[sqx,sqy]
      spiral_alpha=0.d0
      if(bedlaw==9) bedq_gaia_base(:,g,c)=[qx,qy]
      if(valid.and.bedlaw==9.and.state(1,c)>dry_tolerance.and.qx*qx+qy*qy>0.d0) then
       gaia_h=state(1,c);gaia_u=state(2,c)/state(1,c);gaia_v=state(3,c)/state(1,c)
       minimum_depth=max(bedload_min_depth_fraction(g),dry_tolerance)
       if(gaia_h<minimum_depth.and.minimum_depth>dry_tolerance) &
        gaia_h=minimum_depth*(1.d0+128*epsilon(1.d0))
       if(gaia_interior_slope_talmon) then
        call braids_gaia_talmon_slope(gaia_u,gaia_v,-gaia_bed_down(1,c),-gaia_bed_down(2,c),ratio, &
         rhos_fraction(g),rho,grav,d50_fraction(g),gaia_slope_beta,gaia_deviation_beta2, &
         gaia_base_x,gaia_base_y,gaia_factor,valid)
       else
        ! Stage-5b validation ablation: preserve the accepted transport law and
        ! normalization, but present an exactly flat interior bed to the Talmon
        ! operator.  This removes both its magnitude and directional steering.
        call braids_gaia_talmon_slope(gaia_u,gaia_v,0.d0,0.d0,ratio, &
         rhos_fraction(g),rho,grav,d50_fraction(g),gaia_slope_beta,gaia_deviation_beta2, &
         gaia_base_x,gaia_base_y,gaia_factor,valid)
       endif
       if(valid.and.gaia_interior_secondary_stress) then
        call braids_gaia_secondary_stress(gaia_h,gaia_u,gaia_v, &
         hydraulic_equivalent_manning(gaia_h,effective_n),grav,rho, &
         gaia_secondary_alpha,gaia_eta_gradient(1,c),gaia_eta_gradient(2,c),gaia_base_x,gaia_base_y, &
         gaia_total_x,gaia_total_y,valid)
       else if(valid) then
        gaia_total_x=gaia_base_x;gaia_total_y=gaia_base_y
       endif
       gaia_qmag=sqrt(qx*qx+qy*qy)
       if(valid.and.ratio>0.d0) then
        qx=gaia_qmag*gaia_total_x/ratio;qy=gaia_qmag*gaia_total_y/ratio
       else if(valid) then
        qx=0.d0;qy=0.d0
       endif
      endif
      ! The Delft scalar split is asymmetric: only bed load receives the
      ! near-bed spiral direction.  Suspended capacity follows depth-averaged
      ! flow and may be signed (notably Wang/Fredsoe total-minus-bed).
      if(valid.and.secondary_bedload_direction().and.state(1,c)>dry_tolerance.and. &
         qx*qx+qy*qy>0.d0) then
       if(classic_secondary_foundation()) then
        call rotate_classic_secondary_bed(state(1:3,c),c,[qx,qy],raw_bed, &
         spiral_alpha,valid)
        if(valid) then
         qx=raw_bed(1);qy=raw_bed(2)
        endif
       else if(secondary_flow_model==1) then
        call braids_delft3d_spiral_transport(state(1,c),state(2,c)/state(1,c), &
         state(3,c)/state(1,c),formula_n,grav,secondary_flow_espir,secondary_curvature(c),qx,qy, &
         spiral_intensity,spiral_alpha,valid,engelund_hansen_chezy)
       else
        spiral_intensity=state(secondary_state_index(),c)/state(1,c)
        call braids_spiral_transport_from_intensity(state(1,c),state(2,c)/state(1,c), &
         state(3,c)/state(1,c),formula_n,grav,secondary_flow_espir,spiral_intensity,qx,qy, &
         spiral_alpha,valid,engelund_hansen_chezy)
       endif
      endif
      if(valid.and.channel_composition_declaration) then
       ! This is the actual composer input after the optional bed-only
       ! secondary-flow steering; the suspended channel remains unrotated.
       raw_bed=[qx,qy]
       call compose_live_law11_channels(state(1:3,c),state(1,c),g,[qx,qy],[sqx,sqy], &
        composed_bed,retained_suspended,signed_suspended_rate,equilibrium_concentration, &
        valid,composition_status)
       if(valid) then
        qx=composed_bed(1);qy=composed_bed(2)
        sqx=retained_suspended(1);sqy=retained_suspended(2)
       endif
      endif
      if(valid.and.bed_slope==3) ratio=ratio*sqrt(1.d0+spiral_alpha*spiral_alpha)
      if(.not.valid.or..not.all(ieee_is_finite([qx,qy,ratio]))) then
       invalid=1
      else
       capacity(:,g,c)=[qx,qy];tau(g,c)=ratio
       if(capacity_split_retained) then
        suspended_capacity_cell(:,g,c)=[sqx,sqy]
        capacity_absolute_shields(g,c)=absolute_shields_value
        capacity_critical_shields(g,c)=critical_shields_value
        capacity_transport_stage(g,c)=transport_stage_value
       endif
       if(channel_composition_declaration) then
        channel_raw_bed_cell(:,g,c)=raw_bed
        channel_raw_suspended_cell(:,g,c)=raw_suspended
        channel_composed_bed_cell(:,g,c)=composed_bed
        channel_retained_suspended_cell(:,g,c)=retained_suspended
        channel_signed_suspended_rate_cell(g,c)=signed_suspended_rate
        channel_equilibrium_concentration_cell(g,c)=equilibrium_concentration
       endif
      endif
     enddo
    endif
  enddo
  !$omp end parallel do
  if(allocated(gaia_bed)) deallocate(gaia_bed,gaia_eta)
  if(invalid/=0) error stop 'invalid cell-centred bedload capacity'
  if(profile_enabled) profile_euler_capacity_cell_s=profile_euler_capacity_cell_s+omp_get_wtime()-profile_cell_start
  bedq_equilibrium=capacity
  if(capacity_split_retained) suspended_capacity_equilibrium=suspended_capacity_cell
  if(bedload_adaptation) call relax_bedload_capacity(state,boundary_bed_input,capacity)
  if(profile_enabled) profile_cell_start=omp_get_wtime()
  call bedload_capacity_gradients(capacity,gradient)
  if(bedlaw==9) call bedload_capacity_gradients(bedq_gaia_base,bedq_gaia_base_grad)
  if(profile_enabled) profile_euler_capacity_gradient_s=profile_euler_capacity_gradient_s+omp_get_wtime()-profile_cell_start
 end subroutine

 subroutine relax_bedload_capacity(state,boundary_bed_input,capacity)
  ! Steady first-order adaptation along the resolved streamline:
  !
  !        L_g dq_g/ds = q_eq,g - q_g,     L_g = C_L D50_g.
  !
  ! The exponential upwind graph solve below uses one vector value per cell,
  ! centroid-to-centroid (or boundary-face-to-centroid) streamline distances,
  ! and the same face connectivity as Exner.  It is a derived closure evaluated
  ! at every RK stage: no hidden history is introduced, so AMR regridding and
  ! checkpoint/restart remain exact.  A Jacobi solve keeps cell updates
  ! independent and therefore bitwise reproducible across OpenMP team sizes.
  real(8),intent(in) :: state(nstate,nc),boundary_bed_input(nfrac,nb)
  real(8),intent(inout) :: capacity(2,nfrac,nc)
  integer :: work,c,g,k,f,n,b,iteration,nwork,max_iterations
  real(8) :: cell_velocity(2),neighbour_velocity(2),face_velocity(2),outward(2)
  real(8) :: upstream(2),source(2),weight,total_weight,distance_sum,inflow,speed,face_speed,distance,segment_distance,attenuation
  real(8) :: adaptation_length,boundary_span(nb),change,max_change,scale,tolerance
  logical :: converged
  boundary_span=0
  do b=1,nb
   do k=boundary_offset(b),boundary_offset(b+1)-1
    boundary_span(b)=boundary_span(b)+length(boundary_faces(k))
   enddo
  enddo
  nwork=nc
  if(amr_step_level>=0) nwork=amr_level_eval_cell_count(amr_step_level)
  scale=max(maxval(abs(bedq_equilibrium)),1.d-20)
  tolerance=1.d-11*scale
  max_iterations=4096;converged=.false.;bedload_adaptation_iterations=0
  do iteration=1,max_iterations
   max_change=0
   !$omp parallel do schedule(static,256) if(nwork*nfrac>=4096) default(shared) &
   !$omp& private(work,c,g,k,f,n,b,cell_velocity,neighbour_velocity,face_velocity,outward,upstream,source, &
   !$omp& weight,total_weight,distance_sum,inflow,speed,face_speed,distance,segment_distance,attenuation,adaptation_length,change) reduction(max:max_change)
   do work=1,nwork
    c=work
    if(amr_step_level>=0) c=amr_level_eval_cells(work,amr_step_level)
    cell_velocity=0
    if(state(1,c)>dry_tolerance) cell_velocity=state(2:3,c)/state(1,c)
    speed=sqrt(sum(cell_velocity**2))
    do g=1,nfrac
     if(state(1,c)<=dry_tolerance.or.speed<=0) then
      bedq_relax_work(:,g,c)=bedq_equilibrium(:,g,c)
      max_change=max(max_change,maxval(abs(bedq_relax_work(:,g,c)-capacity(:,g,c))))
      cycle
     endif
     upstream=0;total_weight=0;distance_sum=0
     do k=offset(c),offset(c+1)-1
      f=faces(k);outward=signs(k)*normal(:,f);n=neighbour(f)
      if(signs(k)>0) then
       n=neighbour(f)
      else
       n=owner(f)
      endif
      if(n>0) then
       neighbour_velocity=0
       if(state(1,n)>dry_tolerance) neighbour_velocity=state(2:3,n)/state(1,n)
       face_velocity=.5d0*(cell_velocity+neighbour_velocity)
       inflow=-dot_product(face_velocity,outward)
       if(inflow>0) then
        weight=inflow*length(f);upstream=upstream+weight*capacity(:,g,n);total_weight=total_weight+weight
        face_speed=sqrt(sum(face_velocity**2));segment_distance=0
        if(face_speed>0) segment_distance=abs(dot_product(xy(:,c)-xy(:,n),face_velocity/face_speed))
        if(segment_distance<=64*epsilon(1.d0)*max(dx,dy)) segment_distance=sqrt(sum((xy(:,c)-xy(:,n))**2))
        distance_sum=distance_sum+weight*segment_distance
       endif
      else
       b=bid(f)
       if(b<=0) cycle
       if(.not.(mode(b)==1.or.mode(b)==6.or.mode(b)==7.or.mode(b)==8.or.mode(b)==9)) cycle
       inflow=-dot_product(cell_velocity,outward)
       if(inflow<=0) cycle
       if(bed_boundary_mode(b)==1) then
        source=bedq_equilibrium(:,g,c)
       else if(boundary_span(b)>0) then
        source=-outward*boundary_bed_input(g,b)/boundary_span(b)
       else
        source=0
       endif
       weight=inflow*length(f);upstream=upstream+weight*source;total_weight=total_weight+weight
       segment_distance=abs(dot_product(xy(:,c)-mid(:,f),cell_velocity/speed))
       if(segment_distance<=64*epsilon(1.d0)*max(dx,dy)) segment_distance=sqrt(sum((xy(:,c)-mid(:,f))**2))
       distance_sum=distance_sum+weight*segment_distance
      endif
     enddo
     if(total_weight<=0) then
      bedq_relax_work(:,g,c)=bedq_equilibrium(:,g,c)
     else
      distance=distance_sum/total_weight
      if(distance<=64*epsilon(1.d0)*max(dx,dy)) distance=area(c)*speed/total_weight
      adaptation_length=bedload_adaptation_length_diameters*d50_fraction(g)
      attenuation=exp(-distance/adaptation_length)
      bedq_relax_work(:,g,c)=(1.d0-attenuation)*bedq_equilibrium(:,g,c)+ &
                             attenuation*upstream/total_weight
     endif
     change=maxval(abs(bedq_relax_work(:,g,c)-capacity(:,g,c)))
     max_change=max(max_change,change)
    enddo
   enddo
   !$omp end parallel do
   do work=1,nwork
    c=work
    if(amr_step_level>=0) c=amr_level_eval_cells(work,amr_step_level)
    capacity(:,:,c)=bedq_relax_work(:,:,c)
   enddo
   bedload_adaptation_iterations=iteration;bedload_adaptation_residual=max_change
   if(max_change<=tolerance) then
    converged=.true.;exit
   endif
  enddo
  if(.not.converged) error stop 'bed-load adaptation solve did not converge'
  if(.not.all(ieee_is_finite(capacity))) error stop 'invalid adapted bedload capacity'
 end subroutine

 subroutine extract_bank_volume(state,cell,requested,removed,actual)
  ! Remove a three-dimensional solid volume from the resolved bank column,
  ! surface downward.  The returned class masses are totals (kg), whereas the
  ! native state stores kg/m2.  This preserves the actual layered composition
  ! of both continuously eroded toe material and a discrete failed block.
  real(8),intent(inout) :: state(nstate,nc)
  integer,intent(in) :: cell
  real(8),intent(in) :: requested
  real(8),intent(out) :: removed(nfrac),actual
  real(8) :: remaining,layer_volume,take,factor,amount
  integer :: layer,g,index
  removed=0;actual=0;remaining=max(0.d0,requested)
  do layer=1,nbed_layers
   if(remaining<=64*epsilon(1.d0)*max(requested,1.d0)) exit
   layer_volume=0
   do g=1,nfrac
    index=bed_layer_index(layer,g)
    layer_volume=layer_volume+state(index,cell)*area(cell)/bulk_fraction(g)
   enddo
   if(layer_volume<=0) cycle
   take=min(remaining,layer_volume);factor=min(1.d0,take/layer_volume)
   do g=1,nfrac
    index=bed_layer_index(layer,g);amount=state(index,cell)*factor
    state(index,cell)=state(index,cell)-amount
    removed(g)=removed(g)+amount*area(cell)
    if(state(index,cell)<0.and.abs(state(index,cell))*area(cell)<=128*epsilon(1.d0)) state(index,cell)=0
   enddo
   actual=actual+take;remaining=max(0.d0,remaining-take)
  enddo
 end subroutine

 subroutine deposit_bank_transfers(state)
  ! The donor was already excavated by extract_bank_volume.  Gather only the
  ! receiving side of each signed face transfer, avoiding atomics and keeping
  ! one/eight-thread arithmetic identical.
  real(8),intent(inout) :: state(nstate,nc)
  real(8) :: incoming
  integer :: c,g,k,f,bi
  !$omp parallel do collapse(2) schedule(static,256) if(nc>=4096) default(shared) private(c,g,k,f,bi,incoming)
  do c=1,nc
   do g=1,nfrac
    incoming=0
    do k=offset(c),offset(c+1)-1
     f=faces(k)
     if(signs(k)>0) then
      if(bank_transfer(g,f)<0) incoming=incoming-bank_transfer(g,f)
     else
      if(bank_transfer(g,f)>0) incoming=incoming+bank_transfer(g,f)
     endif
    enddo
    if(incoming<=0) cycle
    bi=bed_index(g);state(bi,c)=state(bi,c)+incoming/area(c)
   enddo
  enddo
  !$omp end parallel do
 end subroutine

 subroutine prepare_persistent_bank_front(state,face_transfer,eligible,donor_cell,recipient_cell,ok)
  ! Resolve a moving subcell bank face against the accepted transfer before
  ! changing either inventory or front history.  A step which would cross a
  ! whole-cell promotion event is retried at the exact event time; this avoids
  ! dropping residual erosion demand or making migration depend on dt.
  real(8),intent(in) :: state(nstate,nc)
  real(8),intent(inout) :: face_transfer(nfrac,nf)
  logical,intent(in) :: eligible(nf)
  integer,intent(in) :: donor_cell(nf),recipient_cell(nf)
  logical,intent(out) :: ok
  real(8) :: request_volume(nc),face_volume,capacity,ratio,tolerance
  real(8) :: prospective_base(nc),prospective_height(nc),largest_request(nc),scale(nc)
  integer :: seed(nc),c,f,g,k,recipient,front_error,candidate_seed
  ok=.true.;bank_front_retry_scale=1.d0
  if(.not.bank_front_tracking) return
  if(.not.allocated(bank_front_status)) error stop 'persistent bank-front state is not allocated'
  request_volume=0.d0;prospective_base=0.d0;prospective_height=0.d0
  largest_request=-1.d0;scale=1.d0;seed=0
  call effective_bed_elevation(state,bank_bed)
  do f=1,nf
   if(.not.eligible(f)) cycle
   c=donor_cell(f)
   if(c<1.or.c>nc.or.recipient_cell(f)<1.or.recipient_cell(f)>nc) cycle
   face_volume=0.d0
   do g=1,nfrac
    face_volume=face_volume+abs(face_transfer(g,f))/bulk_fraction(g)
   enddo
   if(face_volume<=0.d0) cycle
   request_volume(c)=request_volume(c)+face_volume
   if(face_volume>largest_request(c)) then
    largest_request(c)=face_volume;seed(c)=f
   endif
  enddo
  do c=1,nc
   if(request_volume(c)<=0.d0) cycle
   if(bank_front_status(c)==BANK_FRONT_PROMOTED) error stop 'promoted bank cell received a new front request'
   if(bank_front_status(c)==BANK_FRONT_INACTIVE) then
    f=seed(c);recipient=recipient_cell(f)
    prospective_base(c)=max(bank_bed(recipient),rock(c))
    prospective_height(c)=bank_bed(c)-prospective_base(c)
    tolerance=512.d0*epsilon(1.d0)*max(abs(bank_bed(c)),abs(prospective_base(c)),1.d0)
    if(prospective_height(c)<=tolerance) then
     do k=offset(c),offset(c+1)-1
      f=faces(k)
      if(donor_cell(f)==c) face_transfer(:,f)=0.d0
     enddo
     request_volume(c)=0.d0
     cycle
    endif
    capacity=area(c)*prospective_height(c)
   else
    if(bank_front_status(c)/=BANK_FRONT_ACTIVE) error stop 'invalid active persistent bank front'
    prospective_base(c)=bank_front_base(c);prospective_height(c)=bank_front_height(c)
    call braids_bank_front_capacity(bank_front_status(c),bank_front_solid_area(c), &
     bank_front_height(c),capacity,front_error)
    if(front_error/=BANK_FRONT_SUCCESS) error stop 'invalid persistent bank-front capacity'
   endif
   tolerance=1024.d0*epsilon(1.d0)*max(capacity,request_volume(c),1.d-30)
   if(request_volume(c)>capacity+tolerance) then
    ratio=capacity/request_volume(c)
    bank_front_retry_scale=min(bank_front_retry_scale,max(1.d-8,ratio))
    amr_failure_cell=c;amr_failure_state=0;amr_failure_value=request_volume(c);amr_failure_aux=capacity
    ok=.false.
   else if(request_volume(c)>capacity) then
    ! Roundoff-sized overshoot may be capped, but an already donor-limited
    ! request is never increased to manufacture an exact promotion event.
    scale(c)=capacity/request_volume(c)
   endif
  enddo
  if(.not.ok) then
   failure_reason=10
   return
  endif
  do c=1,nc
   if(request_volume(c)<=0.d0) cycle
   if(bank_front_status(c)==BANK_FRONT_INACTIVE) then
    candidate_seed=seed(c)
    if(amr_enabled) candidate_seed=at_face_source(candidate_seed)
    call braids_bank_front_activate(area(c),bank_bed(c),prospective_base(c),candidate_seed, &
     bank_front_status(c),bank_front_seed_face(c),bank_front_solid_area(c), &
     bank_front_height(c),bank_front_removed_mass(:,c),front_error)
    if(front_error/=BANK_FRONT_SUCCESS) error stop 'cannot activate persistent bank front'
    bank_front_base(c)=prospective_base(c)
   endif
   if(scale(c)/=1.d0) then
    do k=offset(c),offset(c+1)-1
     f=faces(k)
     if(donor_cell(f)==c) face_transfer(:,f)=scale(c)*face_transfer(:,f)
    enddo
   endif
  enddo
 end subroutine

 subroutine advance_persistent_bank_front(state,face_transfer,eligible,donor_cell)
  real(8),intent(in) :: state(nstate,nc),face_transfer(nfrac,nf)
  logical,intent(in) :: eligible(nf)
  integer,intent(in) :: donor_cell(nf)
  real(8) :: moved_mass(nfrac,nc),volume
  integer :: c,f,g
  if(.not.bank_front_tracking) return
  moved_mass=0.d0
  do f=1,nf
   if(.not.eligible(f)) cycle
   c=donor_cell(f)
   if(c<1.or.c>nc) cycle
   do g=1,nfrac
    moved_mass(g,c)=moved_mass(g,c)+abs(face_transfer(g,f))
   enddo
  enddo
  do c=1,nc
   if(bank_front_status(c)/=BANK_FRONT_ACTIVE.or. &
      all(moved_mass(:,c)==0.d0)) cycle
   volume=sum(moved_mass(:,c)/bulk_fraction)
   call advance_persistent_bank_front_cell(state,c,moved_mass(:,c),volume)
  enddo
 end subroutine

 subroutine advance_persistent_bank_front_cell(state,c,moved_mass,volume)
  real(8),intent(in) :: state(nstate,nc),moved_mass(nfrac),volume
  integer,intent(in) :: c
  real(8) :: raw_bed,tolerance,accepted_volume,volume_residual
  integer :: g,l,front_error
  logical :: promoted
  if(.not.bank_front_tracking.or.volume<=0.d0) return
  if(bank_front_status(c)/=BANK_FRONT_ACTIVE.or.volume<0.d0.or..not.ieee_is_finite(volume).or. &
     any(moved_mass<0.d0).or..not.all(ieee_is_finite(moved_mass))) &
   error stop 'invalid persistent bank-front transfer volume'
  call braids_bank_front_apply_actual_mass(bulk_fraction,moved_mass,area(c), &
   bank_front_status(c),bank_front_solid_area(c),bank_front_height(c), &
   bank_front_removed_mass(:,c),accepted_volume,promoted,front_error)
  if(front_error/=BANK_FRONT_SUCCESS) error stop 'persistent bank-front actual transfer exceeds its prism'
  tolerance=1024.d0*epsilon(1.d0)*max(volume,accepted_volume,1.d-30)
  if(abs(accepted_volume-volume)>tolerance) error stop 'persistent bank-front transfer-volume mismatch'
  if(promoted) then
   raw_bed=rock(c)
   do l=1,nbed_layers
    do g=1,nfrac
     raw_bed=raw_bed+state(bed_layer_index(l,g),c)/bulk_fraction(g)
    enddo
   enddo
   if(abs(raw_bed-bank_front_base(c))> &
      4096.d0*epsilon(1.d0)*max(abs(raw_bed),abs(bank_front_base(c)),1.d0)) &
    error stop 'persistent bank-front promotion does not close its sediment prism'
  endif
  call braids_bank_front_validate(bulk_fraction,area(c),bank_front_status(c), &
   bank_front_seed_face(c),bank_front_solid_area(c),bank_front_height(c), &
   bank_front_removed_mass(:,c),volume_residual,front_error)
  if(front_error/=BANK_FRONT_SUCCESS) error stop 'persistent bank-front invariant failed after transfer'
 end subroutine

 subroutine validate_persistent_bank_front_state()
  real(8) :: residual
  integer :: c,front_error,maximum_seed
  if(.not.bank_front_tracking) return
  maximum_seed=nf
  if(amr_enabled) maximum_seed=amr_fn
  if(.not.all(ieee_is_finite(bank_front_base)).or. &
     .not.all(ieee_is_finite(bank_front_height)).or. &
     .not.all(ieee_is_finite(bank_front_solid_area)).or. &
     .not.all(ieee_is_finite(bank_front_removed_mass))) &
   error stop 'nonfinite persistent bank-front checkpoint state'
  do c=1,nc
   if(bank_front_seed_face(c)<0.or.bank_front_seed_face(c)>maximum_seed) &
    error stop 'invalid persistent bank-front seed identity'
   call braids_bank_front_validate(bulk_fraction,area(c),bank_front_status(c), &
    bank_front_seed_face(c),bank_front_solid_area(c),bank_front_height(c), &
    bank_front_removed_mass(:,c),residual,front_error)
   if(front_error/=BANK_FRONT_SUCCESS) error stop 'invalid persistent bank-front checkpoint state'
  enddo
 end subroutine

 subroutine validate_persistent_bank_front_tree_state()
  real(8) :: residual
  integer :: n,front_error
  if(.not.bank_front_tracking.or..not.amr_enabled) return
  if(.not.all(ieee_is_finite(at_bank_front_base)).or. &
     .not.all(ieee_is_finite(at_bank_front_height)).or. &
     .not.all(ieee_is_finite(at_bank_front_solid_area)).or. &
     .not.all(ieee_is_finite(at_bank_front_removed_mass))) &
   error stop 'nonfinite adaptive persistent bank-front state'
  do n=1,amr_nn
   if(at_bank_front_seed_face(n)<0.or.at_bank_front_seed_face(n)>amr_fn) &
    error stop 'invalid adaptive persistent bank-front seed identity'
   call braids_bank_front_validate(bulk_fraction,at_area(n), &
    at_bank_front_status(n),at_bank_front_seed_face(n), &
    at_bank_front_solid_area(n),at_bank_front_height(n), &
    at_bank_front_removed_mass(:,n),residual,front_error)
   if(front_error/=BANK_FRONT_SUCCESS) &
    error stop 'invalid adaptive persistent bank-front state'
  enddo
 end subroutine

 subroutine total_bed_inventory(state,total)
  real(8),intent(in) :: state(nstate,nc)
  real(8),intent(out) :: total(nfrac,nc)
  integer :: g,layer
  total=0.d0
  do layer=1,nbed_layers
   do g=1,nfrac
    total(g,:)=total(g,:)+state(bed_layer_index(layer,g),:)
   enddo
  enddo
 end subroutine

 subroutine process_noncohesive_bank_failure(reference,state)
  ! Transfer a configured fraction of the accepted erosive Exner loss at a
  ! geometry-defined physical toe from adjacent bank material.  Uniform
  ! through-transport and depositional toes request no bank material.  This
  ! replaces vertical toe scour with equal-volume lateral supply and records
  ! the corresponding undercut; it never interprets raw capacity as erosion.
  real(8),intent(in) :: reference(nstate,nc)
  real(8),intent(inout) :: state(nstate,nc)
  real(8) :: removed(nfrac),actual,actual_volume,denominator,height,toe_height,interface_length
  real(8) :: requested,above_height,residual,best_distance
  integer :: supply_error,c,g,f,toe,events,collapse_cell,candidate
  logical :: has_inventory
  if(bank_toe_erosion_mode/=2) return
  if(.not.bank_toe_mapping_ready) call build_physical_toe_map(reference)
  call total_bed_inventory(reference,physical_before_inventory)
  call total_bed_inventory(state,physical_after_inventory)
  call braids_exner_demand_from_inventory(nc,nfrac,area,bulk_fraction, &
   physical_before_inventory,physical_after_inventory,physical_is_toe_cell, &
   physical_demand_mass,physical_demand_volume,supply_error)
  if(supply_error/=0) error stop 'invalid accepted Exner demand at physical bank toe'
  call braids_allocate_physical_toe_requests(nc,nf,bank_toe_capacity_fraction, &
   physical_demand_volume,physical_is_toe_face,physical_candidate_bank, &
   physical_candidate_toe,length,physical_face_request,physical_donor_request, &
   supply_error)
  if(supply_error/=0) error stop 'invalid physical bank-toe request allocation'

  call bank_bed_elevation(reference,bank_depth)
  physical_removed_mass=0.d0
  do c=1,nc
   if(physical_donor_request(c)<=0.d0) cycle
   call extract_bank_volume(state,c,physical_donor_request(c),removed,actual)
   physical_removed_mass(:,c)=removed
  enddo
  call braids_route_extracted_toe_mass(nc,nf,nfrac,owner,neighbour, &
   physical_is_toe_face,physical_candidate_bank,physical_candidate_toe, &
   physical_face_request,physical_donor_request,physical_removed_mass, &
   physical_face_mass,supply_error)
  if(supply_error/=0) error stop 'invalid physical bank-toe sediment routing'
  bank_transfer=physical_face_mass
  call deposit_bank_transfers(state)

  ! Convert the actual inventory-limited transfer into a lateral undercut.
  ! The denominator is the submerged material-interface area, so the update
  ! remains invariant under mesh rotation and duplicate upslope mappings.
  do c=1,nc
   if(.not.physical_is_toe_donor(c)) cycle
   actual_volume=0.d0
   do g=1,nfrac
    actual_volume=actual_volume+physical_removed_mass(g,c)/bulk_fraction(g)
   enddo
   if(actual_volume<=0.d0) cycle
   denominator=0.d0;interface_length=0.d0
   do f=1,nf
    if(.not.physical_is_toe_face(f).or.physical_candidate_bank(f)/=c) cycle
    toe=physical_candidate_toe(f)
    height=max(bank_depth(c)-bank_depth(toe),0.d0)
    toe_height=min(reference(1,toe),height)
    denominator=denominator+length(f)*toe_height
    interface_length=interface_length+length(f)
    bank_reference_height(c)=max(bank_reference_height(c),height)
    bank_toe_height(c)=max(bank_toe_height(c),toe_height)
   enddo
   if(denominator<=0.d0) error stop 'physical bank-toe transfer has zero submerged interface'
   bank_undercut(c)=bank_undercut(c)+actual_volume/denominator
   bank_reference_length(c)=max(bank_reference_length(c),interface_length)
   bank_toe_cumulative(c)=bank_toe_cumulative(c)+actual_volume/area(c)
  enddo

  ! Once the toe undercut crosses its physical threshold, release material
  ! from the nearest still-dry bank column mapped to that same toe.  Deposit
  ! the block in the mapped toe cell; internal transfer is exactly
  ! conservative and the repose operator runs immediately afterwards.
  do c=1,nc
   if(.not.physical_is_toe_donor(c).or.bank_undercut(c)<bank_critical_undercut) cycle
   events=int(bank_undercut(c)/bank_critical_undercut)
   toe=physical_toe_cell(c);collapse_cell=0;best_distance=huge(1.d0)
   do candidate=1,nc
    if(.not.bank_erodible(candidate).or.physical_toe_donor(candidate)/=c) cycle
    if(reference(1,candidate)>dry_tolerance) cycle
    has_inventory=.false.
    do g=1,nfrac
     if(physical_before_inventory(g,candidate)>0.d0) has_inventory=.true.
    enddo
    if(.not.has_inventory.or.bank_depth(candidate)<=bank_depth(toe)) cycle
    if(physical_toe_distance(candidate)<best_distance) then
     collapse_cell=candidate;best_distance=physical_toe_distance(candidate)
    endif
   enddo
   if(collapse_cell==0) cycle
   above_height=max(bank_depth(collapse_cell)-bank_depth(toe)-bank_toe_height(c),0.d0)
   if(above_height<=0.d0.or.bank_reference_length(c)<=0.d0) cycle
   requested=events*bank_critical_undercut*bank_reference_length(c)*above_height
   removed=0.d0;actual=0.d0
   if(requested>0.d0) call extract_bank_volume(state,collapse_cell,requested,removed,actual)
   if(actual>0.d0) then
    do g=1,nfrac
     state(bed_index(g),toe)=state(bed_index(g),toe)+removed(g)/area(toe)
    enddo
    bank_collapse_cumulative(c)=bank_collapse_cumulative(c)+actual/area(c)
   endif
   bank_collapse_count(c)=bank_collapse_count(c)+events
   residual=bank_undercut(c)-events*bank_critical_undercut
   bank_undercut(c)=max(0.d0,residual)
   if(bank_undercut(c)<=64.d0*epsilon(1.d0)*bank_critical_undercut) then
    bank_undercut(c)=0.d0;bank_reference_height(c)=0.d0
    bank_toe_height(c)=0.d0;bank_reference_length(c)=0.d0
   endif
  enddo
  call rebalance_active_layer(state)
 end subroutine

 subroutine process_distributed_bank_failure(state,dt,ok)
  ! Published local slope-bank supply driven by the conservative Exner flux
  ! that actually advanced this accepted SSPRK/AMR step:
  !   q_bank = alpha * abs(sum_g(q_toe,g) dot t_face)
  ! for an explicitly erodible higher cell whose local rise/run exceeds the
  ! configured critical slope.  Wet and dry bank donors are both valid.  The
  ! standalone constitutive module creates deterministic face requests;
  ! apply_surface_transfer performs the shared donor limiter and conservative
  ! per-fraction gather.  Actual transferred volume advances the retained
  ! undercut history and can release the remaining local overhang as a block.
  real(8),intent(inout) :: state(nstate,nc)
  real(8),intent(in) :: dt
  logical,intent(out) :: ok
  real(8),allocatable :: donor_active_mass(:,:),local_slope(:),tangential_transport(:)
  real(8),allocatable :: requested_transfer(:,:)
  logical,allocatable :: eligible(:),front_erodible(:),exposed_bank(:)
  integer,allocatable :: donor_cell(:),recipient_cell(:)
  real(8) :: actual_volume,height,toe_height,denominator,score,best_score
  real(8) :: requested,actual,above_height,residual,transport_duration
  real(8) :: removed(nfrac)
  integer :: f,c,g,recipient,events,best_face,bank_error,reconstruction_error
  integer :: suppressed_front_faces,front_error
  ok=.true.
  if(bank_toe_erosion_mode/=3) return
  if(dt<=0.d0) error stop 'distributed bank step duration must be positive'
  if(.not.allocated(distributed_face_mass_trial)) &
   error stop 'distributed bank face ledger is not allocated'
  call bank_bed_elevation(state,bank_bed)
  ! The trial ledger is no longer needed after this accepted-step process, so
  ! convert it in place from mass [kg] to mean face-integrated rate [kg/s].
  ! This avoids a second face-sized allocation on large meshes. Every retry
  ! resets the ledger before its first SSPRK stage.
  distributed_face_mass_trial=distributed_face_mass_trial/dt
  call braids_reconstruct_accepted_bedload(nc,nf,nfrac,owner,neighbour,normal,length, &
   distributed_face_mass_trial,distributed_cell_bedload,distributed_rank_deficient,reconstruction_error)
  if(reconstruction_error/=BRAIDS_ACCEPTED_FLUX_SUCCESS) &
   error stop 'invalid accepted distributed-bank face flux'
  allocate(donor_active_mass(nfrac,nc),local_slope(nf),tangential_transport(nf), &
   requested_transfer(nfrac,nf),eligible(nf),front_erodible(nc),exposed_bank(nc),donor_cell(nf),recipient_cell(nf))
  do g=1,nfrac
   donor_active_mass(g,:)=max(state(bed_index(g),:),0.d0)
  enddo
  front_erodible=bank_erodible
  if(bank_front_tracking) where(bank_front_status==2) front_erodible=.false.
  ! Morphological acceleration is already embedded in the accepted Exner
  ! flux. Multiplying the duration by MorFac here would apply it twice.
  transport_duration=dt
  call braids_distributed_bank_face_transfers(nc,nf,nfrac,owner,neighbour,xy,bank_bed, &
   front_erodible,normal,length,distributed_cell_bedload,donor_active_mass, &
   bank_distributed_transport_fraction,bank_distributed_critical_slope,transport_duration, &
   requested_transfer,eligible,donor_cell,recipient_cell,local_slope,tangential_transport,bank_error)
  if(bank_error/=BRAIDS_DISTRIBUTED_BANK_SUCCESS) &
   error stop 'invalid distributed slope-bank transfer request'
  bank_transfer=requested_transfer
  call limit_surface_transfer(state)
  requested_transfer=bank_transfer
  if(bank_front_tracking) then
   call braids_bank_front_exclude_incoming(nc,nf,nfrac,bank_front_status, &
    eligible,donor_cell,recipient_cell,requested_transfer,suppressed_front_faces,front_error)
   if(front_error/=BANK_FRONT_SUCCESS) &
    error stop 'invalid persistent bank-front incoming-transfer filter'
  endif
  call prepare_persistent_bank_front(state,requested_transfer,eligible,donor_cell,recipient_cell,ok)
  if(.not.ok) then
   deallocate(donor_active_mass,local_slope,tangential_transport,requested_transfer, &
    eligible,front_erodible,exposed_bank,donor_cell,recipient_cell)
   return
  endif
  bank_transfer=requested_transfer
  if(bank_front_tracking) then
   do f=1,nf
    if(.not.eligible(f).or.maxval(abs(bank_transfer(:,f)))<=0.d0) cycle
    recipient=recipient_cell(f)
    if(recipient>=1.and.recipient<=nc) then
     if(bank_front_status(recipient)==BANK_FRONT_ACTIVE) &
      error stop 'accepted bank transfer deposits into an active persistent front'
    endif
   enddo
  endif
  call apply_limited_surface_transfer(state)
  call advance_persistent_bank_front(state,bank_transfer,eligible,donor_cell)

  ! All accepted slope transfers change the inventory-derived hydraulic bed.
  ! Only a locally exposed bank can store an undercut: fully submerged slope
  ! redistribution must not accumulate a latent wall-collapse trigger.
  ! Serial face order keeps per-cell histories independent of thread count.
  bank_request=0.d0;bank_weight=0.d0;bank_down=0.d0;exposed_bank=.false.
  do f=1,nf
   if(.not.eligible(f)) cycle
   c=donor_cell(f);recipient=recipient_cell(f)
   height=max(bank_bed(c)-bank_bed(recipient),0.d0)
   toe_height=min(max(state(1,recipient),0.d0),height)
   above_height=max(height-toe_height,0.d0)
   if(above_height>512.d0*epsilon(1.d0)*max(abs(bank_bed(c)),1.d0)) &
    exposed_bank(c)=.true.
   actual_volume=0.d0
   do g=1,nfrac
    actual_volume=actual_volume+abs(bank_transfer(g,f))/bulk_fraction(g)
   enddo
   if(actual_volume<=0.d0) cycle
   bank_toe_cumulative(c)=bank_toe_cumulative(c)+actual_volume/area(c)
   if(above_height<=512.d0*epsilon(1.d0)*max(abs(bank_bed(c)),1.d0)) cycle
   ! Transport was integrated over the accepted hydraulic step. Its receiving
   ! cell can be dry at the endpoint; keep the conservative slope transfer,
   ! but do not interpret that as erosion of a submerged wall interface.
   if(toe_height<=dry_tolerance) cycle
   denominator=length(f)*toe_height
   if(denominator<=0.d0) error stop 'active distributed bank transfer has zero submerged interface'
   bank_request(c)=bank_request(c)+actual_volume
   bank_weight(c)=bank_weight(c)+denominator
   bank_down(1,c)=bank_down(1,c)+length(f)
   bank_down(2,c)=max(bank_down(2,c),height)
   bank_toe_height(c)=max(bank_toe_height(c),toe_height)
  enddo
  do c=1,nc
   if(.not.exposed_bank(c)) then
    bank_undercut(c)=0.d0;bank_reference_height(c)=0.d0
    bank_toe_height(c)=0.d0;bank_reference_length(c)=0.d0
   endif
   if(bank_request(c)<=0.d0) cycle
   if(bank_weight(c)<=0.d0) error stop 'distributed bank transfer has zero interface area'
   bank_undercut(c)=bank_undercut(c)+bank_request(c)/bank_weight(c)
   bank_reference_height(c)=max(bank_reference_height(c),bank_down(2,c))
   bank_reference_length(c)=max(bank_reference_length(c),bank_down(1,c))
  enddo

  ! Release an overhang locally across the strongest currently eligible face.
  ! If no dry/partly submerged overhang exists, continuous slope erosion still
  ! occurs and the undercut is retained rather than inventing a block.
  call bank_bed_elevation(state,bank_bed)
  do c=1,nc
   if(bank_undercut(c)<bank_critical_undercut) cycle
   if(bank_front_tracking) then
    if(bank_front_status(c)/=BANK_FRONT_ACTIVE) cycle
   endif
   best_face=0;best_score=-1.d0
   do f=1,nf
    if(.not.eligible(f).or.donor_cell(f)/=c) cycle
    score=tangential_transport(f)*length(f)
    if(score>best_score) then
     best_score=score;best_face=f
    endif
   enddo
   if(best_face==0) cycle
   recipient=recipient_cell(best_face)
   if(bank_front_tracking) then
    if(bank_front_status(recipient)==BANK_FRONT_ACTIVE) &
     error stop 'bank collapse deposits into an active persistent front'
   endif
   height=max(bank_bed(c)-bank_bed(recipient),0.d0)
   above_height=max(height-bank_toe_height(c),0.d0)
   if(above_height<=0.d0.or.bank_reference_length(c)<=0.d0) cycle
   events=int(bank_undercut(c)/bank_critical_undercut)
   requested=real(events,8)*bank_critical_undercut*bank_reference_length(c)*above_height
   ! Both beds move: the released block cannot invert their ordering merely
   ! to consume stored undercut (including unequal cut-cell areas).
   requested=min(requested,height/(1.d0/area(c)+1.d0/area(recipient)))
   if(bank_front_tracking) requested=min(requested,bank_front_solid_area(c)*bank_front_height(c))
   removed=0.d0;actual=0.d0
   if(requested>0.d0) call extract_bank_volume(state,c,requested,removed,actual)
   if(actual<=0.d0) cycle
   if(bank_front_tracking) call advance_persistent_bank_front_cell(state,c,removed,actual)
   do g=1,nfrac
    state(bed_index(g),recipient)=state(bed_index(g),recipient)+removed(g)/area(recipient)
   enddo
   bank_bed(c)=bank_bed(c)-actual/area(c)
   bank_bed(recipient)=bank_bed(recipient)+actual/area(recipient)
   bank_collapse_cumulative(c)=bank_collapse_cumulative(c)+actual/area(c)
   bank_collapse_count(c)=bank_collapse_count(c)+events
   residual=bank_undercut(c)-real(events,8)*bank_critical_undercut
   bank_undercut(c)=max(0.d0,residual)
   if(bank_undercut(c)<=64.d0*epsilon(1.d0)*bank_critical_undercut) then
    bank_undercut(c)=0.d0;bank_reference_height(c)=0.d0
    bank_toe_height(c)=0.d0;bank_reference_length(c)=0.d0
   endif
  enddo
  call rebalance_active_layer(state)
  if(.not.all(ieee_is_finite(state)).or.any(state(1,:)<0.d0).or. &
     any(state(4:nphysical_state,:)<0.d0)) &
   error stop 'distributed bank transfer produced an invalid conservative state'
  call validate_persistent_bank_front_state()
  deallocate(donor_active_mass,local_slope,tangential_transport,requested_transfer, &
   eligible,front_erodible,exposed_bank,donor_cell,recipient_cell)
 end subroutine

 subroutine apply_bank_failure(reference,state,dt,ok)
  real(8),intent(in) :: reference(nstate,nc),dt
  real(8),intent(inout) :: state(nstate,nc)
  logical,intent(out) :: ok
  ok=.true.
  if(.not.bank_failure) return
  if(bank_toe_erosion_mode==2) then
   call process_noncohesive_bank_failure(reference,state)
  else if(bank_toe_erosion_mode==3) then
   call process_distributed_bank_failure(state,dt,ok)
  else
   call process_bank_failure(state,dt)
  endif
 end subroutine

 subroutine process_bank_failure(state,dt)
  ! Experimental process-separated bank model:
  !   1. near-bank Manning shear drives toe retreat through the standard
  !      excess-shear law E=kd*max(tau-tauc,0)^a;
  !   2. the effective undercut is retained between accepted steps;
  !   3. crossing the configured critical undercut releases the remaining
  !      overhang as a discrete block;
  !   4. toe and block material are transferred, by fraction, to adjacent wet
  !      cells.  The qualified repose operator then acts on that deposit.
  ! A depth-averaged raster cannot represent a negative bank profile.  Toe
  ! excavation therefore lowers the donor column continuously; the separate
  ! undercut state controls only the additional overhang failure event.
  real(8),intent(inout) :: state(nstate,nc)
  real(8),intent(in) :: dt
  integer :: c,n,k,f,wet,events,g
  real(8) :: height,toe_depth,tau,retreat,volume,scale,actual,requested
  real(8) :: removed(nfrac),fraction_share(nfrac),mud_fraction,above_height,residual
  if(.not.bank_failure.or.bank_toe_erosion_mode/=1) return
  call bank_bed_elevation(state,bank_bed)
  bank_rate=0;bank_conductance=0;bank_request=0;bank_weight=0
  bank_depth_intercept=0;bank_down=0
  !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) &
  !$omp& private(c,k,f,n,wet,height,toe_depth,tau,retreat,volume,fraction_share,mud_fraction)
  do c=1,nc
   if(.not.bank_erodible(c).or.state(1,c)>dry_tolerance) cycle
   do k=offset(c),offset(c+1)-1
    f=faces(k);n=neighbour(f)
    if(signs(k)<0) n=owner(f)
    if(n<=0) cycle
    if(state(1,n)<=dry_tolerance) cycle
    wet=n;height=bank_bed(c)-bank_bed(wet)
    if(height<=0) cycle
    toe_depth=min(state(1,wet),height)
    if(toe_depth<=0) cycle
    call surface_composition(state,wet,fraction_share,mud_fraction)
    ! Delft BsSkin replaces RP_TAUB in its sediment formula consumers.  The
    ! BRAIDS toe-erosion law is not one of those locked Delft consumers, so
    ! retain total hydraulic shear until a separate extension is specified.
    tau=hydraulic_bed_shear(state(1,wet),state(2,wet),state(3,wet),manning_cell(wet))
    if(tau<=bank_toe_critical_shear) cycle
    retreat=bank_toe_erodibility*(tau-bank_toe_critical_shear)**bank_toe_exponent*dt
    if(.not.ieee_is_finite(retreat).or.retreat<0) error stop 'nonfinite bank-toe retreat'
    volume=retreat*length(f)*toe_depth
    if(volume<=0) cycle
    bank_rate(f)=volume;bank_conductance(f)=retreat;bank_request(c)=bank_request(c)+volume
    if(retreat>bank_depth_intercept(c)) then
     bank_depth_intercept(c)=retreat;bank_down(1,c)=height;bank_down(2,c)=toe_depth
    endif
   enddo
  enddo
  !$omp end parallel do
  bank_previous=0;bank_scale=0
  !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) private(c,removed,actual,scale)
  do c=1,nc
   if(bank_request(c)<=0) cycle
   call extract_bank_volume(state,c,bank_request(c),removed,actual)
   bank_previous(:,c)=removed;bank_weight(c)=actual
   scale=actual/bank_request(c)
   bank_undercut(c)=bank_undercut(c)+bank_depth_intercept(c)*scale
   bank_reference_height(c)=max(bank_reference_height(c),bank_down(1,c))
   bank_toe_height(c)=max(bank_toe_height(c),bank_down(2,c))
   do k=offset(c),offset(c+1)-1
    f=faces(k)
    if(bank_rate(f)>0.and.bank_conductance(f)==bank_depth_intercept(c)) &
     bank_reference_length(c)=max(bank_reference_length(c),length(f))
   enddo
   bank_toe_cumulative(c)=bank_toe_cumulative(c)+actual/area(c)
  enddo
  !$omp end parallel do
  bank_transfer=0
  !$omp parallel do schedule(static,256) if(nf>=4096) default(shared) private(f,c,n,g,scale)
  do f=1,nf
   if(bank_rate(f)<=0) cycle
   c=owner(f);n=neighbour(f)
   if(n<=0) cycle
   if(.not.bank_erodible(c).or.state(1,c)>dry_tolerance) c=n
   if(c<=0.or.bank_request(c)<=0) cycle
   scale=bank_rate(f)/bank_request(c)
   do g=1,nfrac
    if(c==owner(f)) then
     bank_transfer(g,f)=bank_previous(g,c)*scale
    else
     bank_transfer(g,f)=-bank_previous(g,c)*scale
    endif
   enddo
  enddo
  !$omp end parallel do
  call deposit_bank_transfers(state)

  ! A collapse is an event, not a continuously smeared source.  The failed
  ! block is the unresolved material above the submerged toe over the critical
  ! retreat width.  Multiple thresholds crossed in one accepted step remain
  ! deterministic and leave the exact residual undercut for restart.
  bank_rate=0;bank_request=0;bank_weight=0;bank_previous=0
  !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) &
  !$omp& private(c,events,above_height,requested,k,f,n)
  do c=1,nc
   if(bank_undercut(c)<bank_critical_undercut.or..not.bank_erodible(c)) cycle
   events=int(bank_undercut(c)/bank_critical_undercut)
   above_height=max(bank_reference_height(c)-bank_toe_height(c),0.d0)
   requested=events*bank_critical_undercut*bank_reference_length(c)*above_height
   bank_request(c)=requested
   do k=offset(c),offset(c+1)-1
    f=faces(k);n=neighbour(f)
    if(signs(k)<0) n=owner(f)
    if(n<=0) cycle
    if(state(1,n)>dry_tolerance) then
     bank_rate(f)=length(f);bank_weight(c)=bank_weight(c)+length(f)
    endif
   enddo
  enddo
  !$omp end parallel do
  bank_scale=0
  !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) private(c,removed,actual,events,residual)
  do c=1,nc
   if(bank_undercut(c)<bank_critical_undercut.or..not.bank_erodible(c)) cycle
   events=int(bank_undercut(c)/bank_critical_undercut)
   actual=0;removed=0
   if(bank_request(c)>0.and.bank_weight(c)>0) call extract_bank_volume(state,c,bank_request(c),removed,actual)
   bank_previous(:,c)=removed;bank_collapse_cumulative(c)=bank_collapse_cumulative(c)+actual/area(c)
   bank_collapse_count(c)=bank_collapse_count(c)+events
   residual=bank_undercut(c)-events*bank_critical_undercut
   bank_undercut(c)=max(0.d0,residual)
   if(bank_undercut(c)<=64*epsilon(1.d0)*bank_critical_undercut) then
    bank_undercut(c)=0;bank_reference_height(c)=0;bank_toe_height(c)=0;bank_reference_length(c)=0
   endif
  enddo
  !$omp end parallel do
  bank_transfer=0
  !$omp parallel do schedule(static,256) if(nf>=4096) default(shared) private(f,c,n,g,scale)
  do f=1,nf
   if(bank_rate(f)<=0) cycle
   c=owner(f);n=neighbour(f)
   if(n<=0) cycle
   if(.not.bank_erodible(c).or.state(1,c)>dry_tolerance) c=n
   if(c<=0.or.bank_weight(c)<=0) cycle
   scale=bank_rate(f)/bank_weight(c)
   do g=1,nfrac
    if(c==owner(f)) then
     bank_transfer(g,f)=bank_previous(g,c)*scale
    else
     bank_transfer(g,f)=-bank_previous(g,c)*scale
    endif
   enddo
  enddo
  !$omp end parallel do
  call deposit_bank_transfers(state)
  call rebalance_active_layer(state)
 end subroutine

 subroutine secondary_flow_curvatures(state,curvature)
  ! D-Flow FM's get_curvature least-squares reconstruction on the current
  ! cut-cell/AMR adjacency.  Dry neighbour velocities are zero, matching the
  ! reference routine.  Open boundaries require no ghost contribution because
  ! curvature is a cell-centred diagnostic and receives a Neumann trace later.
  real(8), intent(in) :: state(nstate,nc)
  real(8), intent(out) :: curvature(nc)
  real(8) :: a11,a12,a22,r(2),w,det,u,v,un,vn,dudx,dudy,dvdx,dvdy,distance
  integer :: c,k,f,n,work,nwork,invalid
  logical :: valid
  curvature=0;invalid=0
  nwork=nc
  !$omp parallel do schedule(static,256) if(nwork*nfrac>=8192) default(shared) &
  !$omp& private(work,c,k,f,n,a11,a12,a22,r,w,det,u,v,un,vn,dudx,dudy,dvdx,dvdy,distance,valid) reduction(max:invalid)
  do work=1,nwork
   c=work
   if(state(1,c)<dry_tolerance) cycle
   u=state(2,c)/state(1,c);v=state(3,c)/state(1,c)
   if(sqrt(u*u+v*v)<1.d-3) cycle
   a11=0;a12=0;a22=0;dudx=0;dudy=0;dvdx=0;dvdy=0
   do k=offset(c),offset(c+1)-1
    f=faces(k);n=neighbour(f)
    if(signs(k)<0) n=owner(f)
    if(n==0) cycle
    r=xy(:,n)-xy(:,c);distance=max(sqrt(sum(r*r)),1.d-6);w=1.d0/(distance*distance)
    un=0;vn=0
    if(state(1,n)>=dry_tolerance) then
     un=state(2,n)/state(1,n);vn=state(3,n)/state(1,n)
    endif
    a11=a11+w*r(1)*r(1);a12=a12+w*r(1)*r(2);a22=a22+w*r(2)*r(2)
    dudx=dudx+w*r(1)*(un-u);dudy=dudy+w*r(2)*(un-u)
    dvdx=dvdx+w*r(1)*(vn-v);dvdy=dvdy+w*r(2)*(vn-v)
   enddo
   det=a11*a22-a12*a12
   if(det<1.d-6) cycle
   call braids_streamline_curvature(u,v,(dudx*a22-a12*dudy)/det,(a11*dudy-dudx*a12)/det, &
    (dvdx*a22-a12*dvdy)/det,(a11*dvdy-dvdx*a12)/det,curvature(c),valid)
   if(.not.valid) invalid=1
  enddo
  !$omp end parallel do
  if(invalid/=0.or..not.all(ieee_is_finite(curvature))) error stop 'invalid secondary-flow curvature'
 end subroutine

 subroutine refresh_classic_secondary_equilibrium(state)
  ! Equili=Y: rebuild the classic FLOW intensity from the current
  ! transactional trial state at every stage.  The same stage depth owns both
  ! old/new slots; no hidden secondary state is advanced or committed.
  real(8),intent(in) :: state(nstate,nc)
  integer :: c,invalid
  real(8) :: h,u,v,raw_chezy,z0_value,drag_value,capped_radius,cap_limit, &
   equilibrium_target,source_coefficient,sink_coefficient,effective_chezy, &
   relaxation_alpha,boundary_equilibrium
  logical :: valid,was_capped,boundary_defined,boundary_valid
  if(.not.classic_secondary_foundation()) return
  if(.not.allocated(secondary_curvature).or. &
     .not.allocated(secondary_classic_intensity)) &
   error stop 'v56 classic secondary workspace is unavailable'
  call secondary_flow_curvatures(state,secondary_curvature)
  secondary_classic_intensity=0.d0;invalid=0
  !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) &
  !$omp& private(c,h,u,v,raw_chezy,z0_value,drag_value,capped_radius,cap_limit,equilibrium_target,source_coefficient,sink_coefficient,effective_chezy,relaxation_alpha,boundary_equilibrium,valid,was_capped,boundary_defined,boundary_valid) reduction(max:invalid)
  do c=1,nc
   h=state(1,c)
   if(h<=dry_tolerance) cycle
   u=state(2,c)/h;v=state(3,c)/h
   call hydraulic_roughness_at(h,manning_cell(c),raw_chezy,z0_value,drag_value,valid)
   if(.not.valid.or.raw_chezy<=0.d0) then
    invalid=1;cycle
   endif
   call braids_flow2dh_secondary_source_sink(.true.,h,h,u,v, &
    secondary_curvature(c),2.d0*dx,2.d0*dy,secondary_foundation_coriolis, &
    raw_chezy,dry_tolerance,secondary_foundation_rmincf, &
    secondary_foundation_chzmin,grav,.41d0,capped_radius,cap_limit, &
    equilibrium_target,source_coefficient,sink_coefficient,effective_chezy, &
    relaxation_alpha,was_capped,valid)
   if(.not.valid) then
    invalid=1;cycle
   endif
   call braids_flow2dh_open_boundary_equilibrium(.true.,h,h,dry_tolerance, &
    epsilon(1.d0),source_coefficient,sink_coefficient,boundary_equilibrium, &
    boundary_defined,boundary_valid)
   if(.not.boundary_valid) then
    invalid=1;cycle
   endif
   if(boundary_defined) secondary_classic_intensity(c)=boundary_equilibrium
  enddo
  !$omp end parallel do
  if(invalid/=0.or..not.all(ieee_is_finite(secondary_classic_intensity))) &
   error stop 'invalid v56 classic secondary equilibrium refresh'
 end subroutine

 subroutine rotate_classic_secondary_bed(hydraulic_state,owner_cell,raw_bed, &
  rotated_bed,spiral_alpha,valid)
  ! The scalar eqtran branch consumes physical raw Chezy.  Chzmin belongs to
  ! the source/sink refresh above and must not be applied a second time here.
  real(8),intent(in) :: hydraulic_state(3),raw_bed(2)
  integer,intent(in) :: owner_cell
  real(8),intent(out) :: rotated_bed(2),spiral_alpha
  logical,intent(out) :: valid
  real(8) :: h,u,v,raw_chezy,z0_value,drag_value
  rotated_bed=raw_bed;spiral_alpha=0.d0
  valid=owner_cell>=1.and.owner_cell<=nc.and. &
   all(ieee_is_finite([hydraulic_state,raw_bed]))
  if(.not.valid) return
  h=hydraulic_state(1)
  if(h<=dry_tolerance.or.dot_product(raw_bed,raw_bed)<=0.d0) return
  u=hydraulic_state(2)/h;v=hydraulic_state(3)/h
  call hydraulic_roughness_at(h,manning_cell(owner_cell),raw_chezy,z0_value, &
   drag_value,valid)
  if(.not.valid.or.raw_chezy<=0.d0) then
   valid=.false.;return
  endif
  call braids_flow2dh_scalar_bed_rotation(.true.,u,v,grav,raw_chezy, &
   secondary_foundation_espir,secondary_classic_intensity(owner_cell), &
   raw_bed(1),raw_bed(2),rotated_bed(1),rotated_bed(2),spiral_alpha,valid)
 end subroutine

 subroutine secondary_relax_state(state,dt_source,level,source_integral,ok)
  ! Exact local source integration for the signed conserved intensity h*I.
  ! The serial final sum fixes the accounting order across OpenMP team sizes.
  real(8),intent(inout) :: state(nstate,nc)
  real(8),intent(in) :: dt_source
  integer,intent(in) :: level
  real(8),intent(out) :: source_integral
  logical,intent(out) :: ok
  integer :: work,c,nwork,invalid,si,filter_iterations
  real(8) :: filter_raw(nc),filter_residual
  real(8) :: h,u,v,equilibrium,time_a,updated,increment
  logical :: equilibrium_active,time_active,valid
  source_integral=0.d0;ok=.true.
  if(.not.prognostic_secondary_flow()) return
  call secondary_flow_curvatures(state,secondary_curvature)
  filter_raw=secondary_curvature
  call regularize_curvature(nc,nf,owner,neighbour,area,xy,normal,length,state(1,:), &
   dry_tolerance,2.00000000d0,filter_raw,secondary_curvature,filter_iterations,filter_residual,ok,relative_tolerance=1.0000000000000000d-04,absolute_tolerance=1.0000000000000000d-10)
  if(.not.ok) error stop 'secondary curvature regularization failed'
  secondary_source_work=0.d0;invalid=0;si=secondary_state_index();nwork=nc
  if(level>=0) nwork=amr_level_cell_count(level)
  !$omp parallel do schedule(static,256) if(nwork>=4096) default(shared) &
  !$omp& private(work,c,h,u,v,equilibrium,time_a,updated,increment,equilibrium_active,time_active,valid) &
  !$omp& reduction(max:invalid)
  do work=1,nwork
   c=work
   if(level>=0) c=amr_level_cells(work,level)
   h=state(1,c)
   if(h<=dry_tolerance) then
    increment=-state(si,c);state(si,c)=0.d0
    secondary_source_work(c)=area(c)*increment
    cycle
   endif
   u=state(2,c)/h;v=state(3,c)/h
   call braids_spiral_equilibrium(h,u,v,secondary_curvature(c),equilibrium,equilibrium_active,valid)
   if(.not.valid) then
    invalid=1;cycle
   endif
   call braids_spiral_adaptation_time(h,u,v,hydraulic_equivalent_manning(h,manning_cell(c)), &
    grav,time_a,time_active,valid)
   if(.not.valid) then
    invalid=1;cycle
   endif
   if(.not.equilibrium_active.or..not.time_active.or.dt_source<=0.d0) cycle
   call braids_spiral_relax_exact(h,state(si,c),equilibrium,time_a,dt_source,updated,increment,valid)
   if(.not.valid) then
    invalid=1;cycle
   endif
   state(si,c)=updated;secondary_source_work(c)=area(c)*increment
  enddo
  !$omp end parallel do
  if(invalid/=0.or..not.all(ieee_is_finite(state(si,:)))) then
   ok=.false.;return
  endif
  do c=1,nc
   source_integral=source_integral+secondary_source_work(c)
  enddo
 end subroutine

 logical function coupled_stage_capacity(f,l,r) result(consistent)
  integer,intent(in) :: f
  real(8),intent(in) :: l(3),r(3)
  integer :: c,n
  consistent=.false.;c=owner(f);n=neighbour(f)
  if(n<=0.or.bedlaw/=2.or.bedload_adaptation.or.secondary_bedload_direction().or.bedform_roughness) return
  if(min(l(1),r(1))<=dry_tolerance) return
  if(min(open_boundary_distance(c),open_boundary_distance(n))>2) return
  if(min(uncoupled_boundary_distance(c),uncoupled_boundary_distance(n))<=2) return
  if(abs(dot_product(l(2:3),normal(:,f)))>=l(1)*sqrt(grav*l(1)).and. &
     abs(dot_product(r(2:3),normal(:,f)))>=r(1)*sqrt(grav*r(1))) return
  consistent=.true.
 end function

 subroutine bedload_capacity_gradients(capacity,gradient)
  ! Bounded least-squares slopes provide second order on arbitrary cut cells.
  ! The limiter uses only adjacent cell-centre capacities.  Open boundaries
  ! are closed separately by exact zero-gradient traces, while wall faces do
  ! not carry bedload.
  real(8), intent(in) :: capacity(2,nfrac,nc)
  real(8), intent(out) :: gradient(2,2,nfrac,nc)
  real(8) :: a11,a12,a22,r(2),w,det
  real(8) :: rhs(2,2,nfrac),lower(2,nfrac),upper(2,nfrac),delta(2,nfrac),theta(2,nfrac)
  integer :: c,k,f,n,g,m,work,nwork
  gradient=0
  if(scheme==1) return
  nwork=nc
  if(amr_step_level>=0) nwork=amr_level_face_cell_count(amr_step_level)
  !$omp parallel do schedule(static,256) if(nwork*nfrac>=4096) default(shared) &
  !$omp& private(work,c,k,f,n,g,m,a11,a12,a22,r,w,det,rhs,lower,upper,delta,theta)
  do work=1,nwork
   c=work
   if(amr_step_level>=0) c=amr_level_face_cells(work,amr_step_level)
   if(.not.active_cell(c)) cycle
   a11=bed_ls_matrix(1,c);a12=bed_ls_matrix(2,c);a22=bed_ls_matrix(3,c);rhs=0
   lower=capacity(:,:,c);upper=capacity(:,:,c)
   do k=offset(c),offset(c+1)-1
    f=faces(k);n=neighbour(f)
    if(signs(k)<0) n=owner(f)
    if(n==0) cycle
    r=bed_ls_delta(:,k);w=bed_ls_weight(k)
    do g=1,nfrac
     do m=1,2
      rhs(m,1,g)=rhs(m,1,g)+w*r(1)*(capacity(m,g,n)-capacity(m,g,c))
      rhs(m,2,g)=rhs(m,2,g)+w*r(2)*(capacity(m,g,n)-capacity(m,g,c))
     enddo
    enddo
    lower=min(lower,capacity(:,:,n));upper=max(upper,capacity(:,:,n))
   enddo
   det=bed_ls_matrix(4,c)
   if(det<=1.d-12*(a11+a22)**2) cycle
   do g=1,nfrac
    do m=1,2
     gradient(m,1,g,c)=(a22*rhs(m,1,g)-a12*rhs(m,2,g))/det
     gradient(m,2,g,c)=(a11*rhs(m,2,g)-a12*rhs(m,1,g))/det
    enddo
   enddo
   theta=1
   do k=offset(c),offset(c+1)-1
    f=faces(k);n=neighbour(f)
    if(signs(k)<0) n=owner(f)
    if(n==0) cycle
    r=mid(:,f)-xy(:,c)
    do g=1,nfrac
     do m=1,2
      delta(m,g)=dot_product(gradient(m,:,g,c),r)
      if(delta(m,g)>0) theta(m,g)=min(theta(m,g),(upper(m,g)-capacity(m,g,c))/delta(m,g))
      if(delta(m,g)<0) theta(m,g)=min(theta(m,g),(lower(m,g)-capacity(m,g,c))/delta(m,g))
     enddo
    enddo
   enddo
   do g=1,nfrac
    do m=1,2
     gradient(m,:,g,c)=gradient(m,:,g,c)*max(0.d0,min(1.d0,theta(m,g)))
    enddo
   enddo
  enddo
  !$omp end parallel do
 end subroutine

 subroutine stage_quadratic_capacity(c,f,g,capacity,value)
  integer,intent(in) :: c,f,g
  real(8),intent(in) :: capacity(2,nfrac,nc)
  real(8),intent(inout) :: value(2)
  real(8) :: r1(2),r2(2),disp(2),s,curvature(2)
  integer :: k,ff,n,a,b
  a=0;b=0;disp=mid(:,f)-xy(:,c)
  if(rectangle_width(1,c)<=0.or.rectangle_width(2,c)<=0) return
  do k=offset(c),offset(c+1)-1
   ff=faces(k);n=neighbour(ff)
   if(signs(k)<0) n=owner(ff)
   if(n<=0) cycle
   if(open_boundary_distance(n)<=open_boundary_distance(c)) cycle
   r1=xy(:,n)-xy(:,c)
   if(abs(r1(1)*disp(2)-r1(2)*disp(1))>1.d-8*sum(r1*r1)) cycle
   if(maxval(abs(rectangle_width(:,n)-rectangle_width(:,c)))>1.d-8*maxval(rectangle_width(:,c))) cycle
   a=n;exit
  enddo
  if(a==0) return
  r1=xy(:,a)-xy(:,c)
  do k=offset(a),offset(a+1)-1
   ff=faces(k);n=neighbour(ff)
   if(signs(k)<0) n=owner(ff)
   if(n<=0) cycle
   r2=xy(:,n)-xy(:,a)
   if(maxval(abs(r2-r1))>1.d-8*sqrt(sum(r1*r1))) cycle
   if(maxval(abs(rectangle_width(:,n)-rectangle_width(:,c)))>1.d-8*maxval(rectangle_width(:,c))) cycle
   b=n;exit
  enddo
  if(b==0) return
  s=dot_product(disp,r1)/sum(r1*r1)
  curvature=.5d0*(capacity(:,g,b)-2*capacity(:,g,a)+capacity(:,g,c))
  value=capacity(:,g,c)+s*(capacity(:,g,a)-capacity(:,g,c)-curvature)+(s*s-1.d0/12)*curvature
 end subroutine

 subroutine bedload_capacity_face(c,f,g,capacity,gradient,qx,qy,consistent_stage)
  ! WENO-Z is used on certified five-cell Cartesian stencils.  Cut faces use
  ! the bounded least-squares gradient above.  Both traces are finally bounded
  ! in the face-normal/tangential frame by the two adjacent cell values.
  integer, intent(in) :: c,f,g
  real(8), intent(in) :: capacity(2,nfrac,nc),gradient(2,2,nfrac,nc)
  logical, intent(in), optional :: consistent_stage
  logical :: midpoint
  real(8), intent(out) :: qx,qy
  real(8) :: value(2),stencil(5),r(2),tangent(2),qn,qt,lo,hi
  integer :: side,k,m,n
  side=1
  if(c/=owner(f)) side=2
  n=neighbour(f)
  if(c/=owner(f)) n=owner(f)
  value=capacity(:,g,c)
  if(n==0.or.scheme==1) then
   qx=value(1);qy=value(2);return
  endif
  midpoint=open_boundary_distance(c)<=2
  if(present(consistent_stage)) midpoint=midpoint.and..not.consistent_stage
  if(midpoint) then
   ! Use one bounded midpoint treatment through the five-cell stencil's
   ! boundary radius.  A constant second-cell trace instead creates a
   ! spurious 50%/150% Exner rate for affine capacity; switching to WENO
   ! while its stencil still includes the boundary owner also reintroduces
   ! inlet oscillations.  Interior reconstruction and physical BC fluxes
   ! retain their separate contracts; no bed smoothing is applied.
   value=.5d0*(capacity(:,g,c)+capacity(:,g,n))
   qx=value(1);qy=value(2);return
  endif
  if(weno_cells(1,side,f)>0) then
   do m=1,2
    do k=1,5
     stencil(k)=capacity(m,g,weno_cells(k,side,f))
    enddo
    value(m)=reconstruct(stencil,5)
   enddo
  else
   r=face_displacement(:,side,f)
   do m=1,2
    value(m)=value(m)+dot_product(gradient(m,:,g,c),r)
   enddo
  endif
  if(present(consistent_stage)) then
   if(consistent_stage) call stage_quadratic_capacity(c,f,g,capacity,value)
  endif
  tangent=[-normal(2,f),normal(1,f)]
  qn=dot_product(value,normal(:,f));qt=dot_product(value,tangent)
  lo=min(dot_product(capacity(:,g,c),normal(:,f)),dot_product(capacity(:,g,n),normal(:,f)))
  hi=max(dot_product(capacity(:,g,c),normal(:,f)),dot_product(capacity(:,g,n),normal(:,f)))
  qn=max(lo,min(hi,qn))
  lo=min(dot_product(capacity(:,g,c),tangent),dot_product(capacity(:,g,n),tangent))
  hi=max(dot_product(capacity(:,g,c),tangent),dot_product(capacity(:,g,n),tangent))
  qt=max(lo,min(hi,qt))
  value=qn*normal(:,f)+qt*tangent
  qx=value(1);qy=value(2)
 end subroutine

 function sediment_z0(h,roughness_n,fraction) result(value)
  ! FLOW taubot Manning-to-log-law conversion, evaluated on the supplied
  ! BRAIDS cell/face depth. FLOW's staggered spatial averaging is distinct.
  real(8), intent(in) :: h,roughness_n
  integer, intent(in) :: fraction
  real(8) :: value,x,e,chezy,drag
  logical :: valid
  value=z0_fraction(fraction)
  if(sediment_roughness_fraction(fraction)==0.or.h<=dry_tolerance) return
  if(hydraulic_roughness_mode==braids_roughness_manning.and. &
     roughness_minimum_work_depth==0.d0) then
   x=real(0.41,8)*h**(1.d0/6.d0)/(sqrt(grav)*roughness_n)
   ! Algebraically h/[e*(exp(x)-1)], without positive exponential overflow.
   e=exp(-x);value=max(tiny(1.d0),h*exp(-1.d0)*e/(1-e))
  else
   call hydraulic_roughness_at(h,roughness_n,chezy,value,drag,valid)
   if(.not.valid) value=z0_fraction(fraction)
  endif
 end function

 subroutine bed_capacity(state,fraction,qx,qy,tau_ratio,valid,hiding_factor,roughness_n,mixture_dg,mixture_gsd,mixture_sand,mixture_d90,mixture_mud,suspended_qx,suspended_qy,absolute_shields_output,critical_shields_output,transport_stage_output)
  real(8), intent(in) :: state(3)
  integer, intent(in) :: fraction
  real(8), intent(out) :: qx,qy,tau_ratio
  logical, intent(out) :: valid
  real(8), intent(in) :: hiding_factor
  real(8), optional, intent(in) :: roughness_n,mixture_dg,mixture_gsd,mixture_sand,mixture_d90,mixture_mud
  real(8), optional, intent(out) :: suspended_qx,suspended_qy
  real(8), optional, intent(out) :: absolute_shields_output,critical_shields_output,transport_stage_output
  real(8) :: us,mu,nvalue,formula_n,dgvalue,gsdvalue,sandvalue,d90value,slope_chezy,critical_shields
  real(8) :: formula_shear,chezy_value,z0_value,drag_value,mudvalue
  real(8) :: bed_mass(2),suspended_mass(2),absolute_shields_value,transport_stage,total_mass,current_ustar,critical_speed
  real(8) :: scalar_stress_ratio
  real(8) :: capacity_state(3),transport_state(3),minimum_depth,wet_fraction,xi
  nvalue=manning
  if(present(roughness_n)) nvalue=roughness_n
  qx=0;qy=0;tau_ratio=0;bed_mass=0.d0;suspended_mass=0.d0
  absolute_shields_value=0.d0;critical_shields=0.d0;transport_stage=0.d0
  scalar_stress_ratio=0.d0;valid=.true.
  if(present(suspended_qx)) suspended_qx=0.d0
  if(present(suspended_qy)) suspended_qy=0.d0
  if(present(absolute_shields_output)) absolute_shields_output=0.d0
  if(present(critical_shields_output)) critical_shields_output=0.d0
  if(present(transport_stage_output)) transport_stage_output=0.d0
  if(sediment_type_fraction(fraction)==1) then
   return
  endif
  ! The transport formula has a finite validity depth, but a hard switch at
  ! that depth creates a grid-dependent Exner discontinuity as a dry inlet
  ! wets.  Evaluate the closure at its shallowest valid state while retaining
  ! the resolved velocity, and use a C1 wetting ramp from the hydraulic dry
  ! tolerance to the declared transport depth.  This regularizes only the
  ! dry-start transition; resolved states above the threshold are unchanged.
  minimum_depth=max(bedload_min_depth_fraction(fraction),dry_tolerance)
  if(state(1)<=dry_tolerance) return
  capacity_state=state;wet_fraction=1.d0
  if(state(1)<minimum_depth.and.minimum_depth>dry_tolerance) then
   xi=max(0.d0,min(1.d0,(state(1)-dry_tolerance)/(minimum_depth-dry_tolerance)))
   wet_fraction=xi*xi*(3.d0-2.d0*xi)
   capacity_state(1)=minimum_depth*(1.d0+128*epsilon(1.d0))
   capacity_state(2:3)=state(2:3)*capacity_state(1)/state(1)
  endif
  if(wet_fraction<=0) return
  formula_n=nvalue
  if(hydraulic_roughness_mode/=braids_roughness_manning.or. &
     roughness_minimum_work_depth>0.d0) &
   formula_n=hydraulic_equivalent_manning(capacity_state(1),nvalue)
  mudvalue=0.d0
  if(present(mixture_mud)) mudvalue=mixture_mud
  formula_shear=sediment_bed_shear(capacity_state(1),capacity_state(2), &
   capacity_state(3),nvalue,mudvalue)
  if(capacity_foundation_runtime_declaration) then
   d90value=d90_fraction(fraction)
   if(present(mixture_d90).and.mixture_d90>0.d0) d90value=mixture_d90
   call hydraulic_roughness_at(capacity_state(1),nvalue,chezy_value,z0_value,drag_value,valid)
   if(valid) call braids_scalar_capacity_dispatch(bedlaw,capacity_state(1), &
    capacity_state(2)/capacity_state(1),capacity_state(3)/capacity_state(1), &
    chezy_value,d50_fraction(fraction),d90value,rhos_fraction(fraction),rho,grav, &
    hiding_factor,capacity_parameter(:capacity_parameter_count,fraction), &
    bed_mass,suspended_mass,absolute_shields_value,critical_shields, &
    scalar_stress_ratio,valid)
   qx=bed_mass(1);qy=bed_mass(2)
   ! Historical downstream slope consumers call this field ``tau_ratio``.
   ! EH has no capacity threshold and consumes absolute Shields; MPM/general
   ! consume the dimensionless stress ratio returned by the common ABI.
   if(valid) tau_ratio=merge(absolute_shields_value,scalar_stress_ratio, &
    bedlaw==braids_capacity_engelund_hansen)
  else if(bedlaw==2) then
   if(mpm_friction_mode==1) then
    d90value=d90_fraction(fraction)
    if(present(mixture_d90)) d90value=mixture_d90
    if(capacity_route_identity_declaration.and.trim(capacity_route_method_id)=='capacity.mpm') then
     call braids_delft_mpm_bedload_roughness(capacity_state(1), &
      capacity_state(2)/capacity_state(1),capacity_state(3)/capacity_state(1),formula_n, &
      d50_fraction(fraction),d90value,rhos_fraction(fraction)/rho-1.d0,grav, &
      hiding_factor,qx,qy,valid,tau_ratio)
    else
     call braids_mpm_bedload_roughness(capacity_state(1),capacity_state(2)/capacity_state(1), &
      capacity_state(3)/capacity_state(1),formula_n, &
      d50_fraction(fraction),d90value,rhos_fraction(fraction)/rho-1.d0,grav, &
      mpm_critical,hiding_factor,qx,qy,valid,tau_ratio)
    endif
   else
    call braids_mpm_bedload(capacity_state(2)/capacity_state(1),capacity_state(3)/capacity_state(1), &
     mpm_cf,d50_fraction(fraction), &
     rhos_fraction(fraction)/rho-1.d0,grav,mpm_critical*hiding_factor,qx,qy,valid,tau_ratio)
   endif
   ! MPM returns solid VOLUME per width; the shared bed inventory is MASS.
   qx=rhos_fraction(fraction)*qx;qy=rhos_fraction(fraction)*qy
  else if(bedlaw==1) then
   ! Van Rijn's current-only closure uses the same 0.01 m constitutive work
   ! depth as stock FLOW, but the resolved velocity remains defined at the
   ! physical/activation water-column height.  FLOW establishes that split in
   ! dwnvel before erosed floors its formula depth.  The optional observation
   ! depth below preserves that ordering instead of silently moving the
   ! velocity observation height to 0.01/e.  The already-qualified C1 factor
   ! still regularizes states below the user-declared transport depth.
   transport_state=capacity_state
   if(transport_state(1)<van_rijn_calculation_minimum_depth) then
    transport_state(2:3)=capacity_state(2:3)*van_rijn_calculation_minimum_depth/capacity_state(1)
    transport_state(1)=van_rijn_calculation_minimum_depth
   endif
   call braids_vr1993_bedload_cell(transport_state(1),transport_state(2),transport_state(3),d50_fraction(fraction), &
    d90_fraction(fraction),rhos_fraction(fraction),rho,sediment_z0(transport_state(1),nvalue,fraction), &
    dstar_fraction(fraction),taucr_fraction(fraction),bedload_min_depth_fraction(fraction), &
    qx,qy,us,mu,tau_ratio,valid,capacity_state(1))
  else if(bedlaw==11) then
   call hydraulic_roughness_at(capacity_state(1),nvalue,chezy_value,z0_value,drag_value,valid)
   if(valid) call braids_general_capacity(capacity_state(2)/capacity_state(1), &
    capacity_state(3)/capacity_state(1),d50_fraction(fraction),rhos_fraction(fraction), &
    rho,grav,chezy_value,hiding_factor,capacity_parameter(1,fraction), &
    capacity_parameter(2,fraction),capacity_parameter(3,fraction), &
    capacity_parameter(4,fraction),capacity_parameter(5,fraction), &
    capacity_parameter(6,fraction),capacity_parameter(7,fraction), &
    capacity_parameter(8,fraction),capacity_parameter(9,fraction), &
    capacity_parameter(10,fraction),bed_mass,suspended_mass,absolute_shields_value,valid)
   qx=bed_mass(1);qy=bed_mass(2)
   critical_shields=hiding_factor*capacity_parameter(5,fraction)
   if(critical_shields>0.d0) tau_ratio=absolute_shields_value/critical_shields
  else if(bedlaw==12) then
   d90value=d90_fraction(fraction)
   if(present(mixture_d90).and.mixture_d90>0.d0) d90value=mixture_d90
   call braids_modified_van_rijn_1984(capacity_state(1), &
    capacity_state(2)/capacity_state(1),capacity_state(3)/capacity_state(1), &
    d50_fraction(fraction),d90value,rhos_fraction(fraction),rho,grav, &
    constitutive_kinematic_viscosity, &
    capacity_parameter(1,fraction),capacity_parameter(3,fraction), &
    capacity_parameter(4,fraction),capacity_parameter(5,fraction),mudvalue, &
    bed_mass,suspended_mass,absolute_shields_value,critical_shields,transport_stage,valid)
   qx=bed_mass(1);qy=bed_mass(2)
   if(valid.and.critical_shields>0.d0) tau_ratio=absolute_shields_value/critical_shields
  else if(bedlaw==13) then
   call braids_soulsby_van_rijn_current(capacity_state(1), &
    capacity_state(2)/capacity_state(1),capacity_state(3)/capacity_state(1), &
    d50_fraction(fraction),rhos_fraction(fraction),rho,grav, &
    constitutive_kinematic_viscosity, &
    capacity_parameter(1,fraction),capacity_parameter(2,fraction), &
    capacity_parameter(3,fraction),bed_mass,suspended_mass,absolute_shields_value, &
    critical_shields,critical_speed,valid)
   qx=bed_mass(1);qy=bed_mass(2)
   if(valid.and.critical_shields>0.d0) tau_ratio=absolute_shields_value/critical_shields
  else if(bedlaw==14) then
   d90value=d90_fraction(fraction)
   if(present(mixture_d90).and.mixture_d90>0.d0) d90value=mixture_d90
   call hydraulic_roughness_at(capacity_state(1),nvalue,chezy_value,z0_value,drag_value,valid)
   current_ustar=sqrt(max(hydraulic_bed_shear(capacity_state(1),capacity_state(2), &
    capacity_state(3),nvalue)/rho,0.d0))
   if(valid) call braids_wang_fredsoe_capacity(capacity_state(1), &
    capacity_state(2)/capacity_state(1),capacity_state(3)/capacity_state(1), &
    d50_fraction(fraction),d90value,rhos_fraction(fraction),rho,grav, &
    capacity_parameter(1,fraction),chezy_value,current_ustar, &
    capacity_parameter(2,fraction),bed_mass,suspended_mass,absolute_shields_value, &
    critical_shields,total_mass,valid)
   qx=bed_mass(1);qy=bed_mass(2)
   if(valid.and.critical_shields>0.d0) tau_ratio=absolute_shields_value/critical_shields
  else
   dgvalue=d50_fraction(fraction);gsdvalue=1.d0;sandvalue=1.d0
   if(present(mixture_dg)) dgvalue=mixture_dg
   if(present(mixture_gsd)) gsdvalue=mixture_gsd
   if(present(mixture_sand)) sandvalue=mixture_sand
   if(bedlaw==8) then
    call braids_current_bedload(bedlaw,capacity_state(1),capacity_state(2)/capacity_state(1), &
     capacity_state(3)/capacity_state(1),formula_n, &
     sediment_z0(capacity_state(1),nvalue,fraction),d50_fraction(fraction),rhos_fraction(fraction),rho,grav,dgvalue,gsdvalue,sandvalue, &
     transport_calibration,qx,qy,tau_ratio,valid,engelund_hansen_chezy)
   else if(bedlaw==9) then
    call braids_current_bedload(bedlaw,capacity_state(1),capacity_state(2)/capacity_state(1), &
     capacity_state(3)/capacity_state(1),formula_n, &
     sediment_z0(capacity_state(1),nvalue,fraction),d50_fraction(fraction),rhos_fraction(fraction),rho,grav,dgvalue,gsdvalue,sandvalue, &
     transport_calibration,qx,qy,tau_ratio,valid,gaia_skin_roughness_ratio=gaia_skin_roughness_ratio)
   else
    if(bedlaw==3) then
     call braids_current_bedload(bedlaw,capacity_state(1),capacity_state(2)/capacity_state(1), &
      capacity_state(3)/capacity_state(1),formula_n, &
      sediment_z0(capacity_state(1),nvalue,fraction),d50_fraction(fraction),rhos_fraction(fraction),rho,grav,dgvalue,gsdvalue,sandvalue, &
      transport_calibration,qx,qy,tau_ratio,valid,hiding_factor=hiding_factor)
    else if(sediment_shear_mode==1.and.any(bedlaw==[5,6,7])) then
     call braids_current_bedload(bedlaw,capacity_state(1),capacity_state(2)/capacity_state(1), &
      capacity_state(3)/capacity_state(1),formula_n, &
      sediment_z0(capacity_state(1),nvalue,fraction),d50_fraction(fraction),rhos_fraction(fraction),rho,grav,dgvalue,gsdvalue,sandvalue, &
      transport_calibration,qx,qy,tau_ratio,valid,sediment_shear=formula_shear)
    else
     call braids_current_bedload(bedlaw,capacity_state(1),capacity_state(2)/capacity_state(1), &
      capacity_state(3)/capacity_state(1),formula_n, &
      sediment_z0(capacity_state(1),nvalue,fraction),d50_fraction(fraction),rhos_fraction(fraction),rho,grav,dgvalue,gsdvalue,sandvalue, &
      transport_calibration,qx,qy,tau_ratio,valid)
    endif
   endif
  endif
  if(valid.and.bedlaw==2.and.bed_slope==3) then
   if(mpm_friction_mode==1) then
    call hydraulic_roughness_at(capacity_state(1),nvalue,chezy_value,z0_value,drag_value,valid)
    if(valid) slope_chezy=chezy_value
   else if(mpm_cf>0.d0) then
    slope_chezy=sqrt(grav/mpm_cf)
   else
    tau_ratio=0.d0
   endif
   if(mpm_friction_mode==1.or.mpm_cf>0.d0) &
    call braids_delft3d_total_shields(capacity_state(2)/capacity_state(1), &
     capacity_state(3)/capacity_state(1),slope_chezy,d50_fraction(fraction), &
     rhos_fraction(fraction)/rho-1.d0,0.d0,tau_ratio,valid)
  else if(valid.and.bedlaw==8.and.bed_slope==2) then
   ! Legacy capacity-ratio mode is an explicit choice, distinct from FLOW
   ! mode 5. EH has no capacity threshold: use the physical grain critical
   ! stress, never the unrelated MPM calibration parameter.
   critical_shields=taucr_fraction(fraction)/ &
    ((rhos_fraction(fraction)-rho)*grav*d50_fraction(fraction))
   tau_ratio=tau_ratio/critical_shields
  endif
  qx=wet_fraction*qx;qy=wet_fraction*qy
  ! V51 authenticates model-global Sus in the shared post-density composer.
  ! Preserve every older deck's historical multiplication exactly.
  if(channel_composition_declaration.and.any(bedlaw==[2,8,11])) then
   suspended_mass=wet_fraction*suspended_mass
  else
   suspended_mass=suspended_calibration_factor*wet_fraction*suspended_mass
  endif
  if(present(suspended_qx)) suspended_qx=suspended_mass(1)
  if(present(suspended_qy)) suspended_qy=suspended_mass(2)
  if(present(absolute_shields_output)) absolute_shields_output=absolute_shields_value
  if(present(critical_shields_output)) critical_shields_output=critical_shields
  if(present(transport_stage_output)) transport_stage_output=transport_stage
 end subroutine

 subroutine gaia_open_boundary_capacity(state,fraction,cell,hiding_factor,roughness_n, &
  mixture_dg,mixture_gsd,mixture_sand,mixture_d90,qx,qy,tau_b,valid)
  ! GAIA treats LIQBOR=KSORT nodes as zero bed slope in BEDLOAD_EFFPNT, but
  ! still adds the free-surface-gradient secondary-current stress.  Evaluate
  ! the law on the resolved hydraulic boundary trace and apply exactly that
  ! open-boundary composition instead of inheriting law-8 or owner capacity.
  real(8),intent(in) :: state(3),hiding_factor,roughness_n
  real(8),intent(in) :: mixture_dg,mixture_gsd,mixture_sand,mixture_d90
  integer,intent(in) :: fraction,cell
  real(8),intent(out) :: qx,qy,tau_b
  logical,intent(out) :: valid
  real(8) :: u,v,formula_n

  call bed_capacity(state,fraction,qx,qy,tau_b,valid,hiding_factor,roughness_n, &
   mixture_dg,mixture_gsd,mixture_sand,mixture_d90)
  if(.not.valid.or.state(1)<=dry_tolerance.or.qx*qx+qy*qy<=0.d0) return
  u=state(2)/state(1);v=state(3)/state(1)
  if(gaia_open_boundary_secondary_stress) then
   formula_n=hydraulic_equivalent_manning(state(1),roughness_n)
   call braids_gaia_open_boundary_capacity(state(1),u,v,bedload_min_depth_fraction(fraction), &
    dry_tolerance,formula_n,grav,rho,gaia_secondary_alpha,gaia_eta_gradient(1,cell), &
    gaia_eta_gradient(2,cell),qx,qy,tau_b,valid)
  endif
 end subroutine gaia_open_boundary_capacity

 subroutine bed_downslopes(bed,down)
  ! Delft3D defines these components as positive down slope. Use a separate,
  ! unlimited least-squares reconstruction; the solution limiter used for the
  ! hydraulic face states must not silently flatten the morphology correction.
  real(8), intent(in) :: bed(nc)
  real(8), intent(out) :: down(2,nc)
  real(8) :: a11,a12,a22,b1,b2,r(2),w,det
  integer :: c,k,f,n,work,nwork
  down=0
  nwork=nc
  if(amr_step_level>=0) nwork=amr_level_face_cell_count(amr_step_level)
  !$omp parallel do schedule(static,256) if(nwork>=4096) default(shared) private(work,c,k,f,n,a11,a12,a22,b1,b2,r,w,det)
  do work=1,nwork
   c=work
   if(amr_step_level>=0) c=amr_level_face_cells(work,amr_step_level)
   if(.not.active_cell(c)) cycle
   a11=bed_ls_matrix(1,c);a12=bed_ls_matrix(2,c);a22=bed_ls_matrix(3,c);b1=0;b2=0
   do k=offset(c),offset(c+1)-1
    f=faces(k);n=neighbour(f)
    if(signs(k)<0) n=owner(f)
    if(n==0) cycle
    r=bed_ls_delta(:,k);w=bed_ls_weight(k)
    b1=b1+w*r(1)*(bed(n)-bed(c));b2=b2+w*r(2)*(bed(n)-bed(c))
   enddo
   det=bed_ls_matrix(4,c)
   if(det>1.d-12*(a11+a22)**2) then
    down(1,c)=-(a22*b1-a12*b2)/det
    down(2,c)=-(a11*b2-a12*b1)/det
   endif
  enddo
  !$omp end parallel do
 end subroutine

 subroutine face_downslope(f,bed,down,sx,sy,valid)
  ! Reproduce D-Flow FM's link construction: exact centre-to-centre normal
  ! difference, tangential component interpolated from cell gradients, and
  ! zero normal bed gradient at an open boundary.
  integer, intent(in) :: f
  real(8), intent(in) :: bed(nc),down(2,nc)
  real(8), intent(out) :: sx,sy
  logical, intent(out) :: valid
  real(8) :: tangent(2),dn,sn,st
  integer :: c,n
  c=owner(f);n=neighbour(f);tangent=[-normal(2,f),normal(1,f)]
  if(n>0) then
   dn=dot_product(xy(:,n)-xy(:,c),normal(:,f))
   valid=ieee_is_finite(dn).and.dn>1.d-12*max(dx,dy)
   if(.not.valid) then
    sx=0;sy=0;return
   endif
   sn=-(bed(n)-bed(c))/dn
   st=.5d0*dot_product(down(:,c)+down(:,n),tangent)
  else
   sn=0
   st=dot_product(down(:,c),tangent)
  endif
  sx=sn*normal(1,f)+st*tangent(1)
  sy=sn*normal(2,f)+st*tangent(2)
  valid=all(ieee_is_finite([sx,sy]))
 end subroutine

 pure subroutine flow2dh_settlefrm2_clear_velocity(d50,fac_dss,rhosol,rhow, &
  temperature_c,gravity,settling_velocity,valid)
  ! Locked FLOW rdsed/fallve/eqsettle SettleFrm=2 clear-water branch.  Runtime
  ! hindering is applied later from the total advected concentration; this
  ! routine independently verifies the derived velocity serialized in the
  ! nested legacy slot.
  real(8),intent(in) :: d50,fac_dss,rhosol,rhow,temperature_c,gravity
  real(8),intent(out) :: settling_velocity
  logical,intent(out) :: valid
  real(8),parameter :: dsand=64.d-6,dgravel=2.d-3
  real(8) :: dss,vcmol,relative_density,coefw
  settling_velocity=0.d0
  valid=all(ieee_is_finite([d50,fac_dss,rhosol,rhow,temperature_c,gravity])).and. &
   min(d50,fac_dss,rhow,gravity)>0.d0.and.rhosol>rhow.and.temperature_c>-20.d0
  if(.not.valid) return
  dss=d50*fac_dss
  vcmol=4.d-5/(20.d0+temperature_c)
  relative_density=rhosol/rhow
  if(dss<1.5d0*dsand) then
   settling_velocity=(relative_density-1.d0)*gravity*dss*dss/(18.d0*vcmol)
  else if(dss<.5d0*dgravel) then
   if(dss<2.d0*dsand) then
    coefw=(-2.9912d0/dsand)*dss+15.9824d0
   else
    coefw=10.d0
   endif
   settling_velocity=coefw*vcmol/dss* &
    (sqrt(1.d0+(relative_density-1.d0)*gravity*dss**3/(100.d0*vcmol*vcmol))-1.d0)
  else
   settling_velocity=1.1d0*sqrt((relative_density-1.d0)*gravity*dss)
  endif
  valid=ieee_is_finite(settling_velocity).and.settling_velocity>=0.d0
 end subroutine

 pure subroutine flow2dh_galappatti_terms(speed,ustar,settling_velocity,h0,h1, &
  equilibrium_concentration,explicit_source,total_implicit_sink,tsd,valid)
  ! Locked FLOW soursin_2d source order.  Coefficients are default-real
  ! literals promoted to binary64, matching the reference routine rather than
  ! silently granting the polynomial extra decimal precision.
  real(8),intent(in) :: speed,ustar,settling_velocity,h0,h1,equilibrium_concentration
  real(8),intent(out) :: explicit_source,total_implicit_sink,tsd
  logical,intent(out) :: valid
  real(8) :: wsl,u,w,x,x2,x3,hots,exponent
  explicit_source=0.d0;total_implicit_sink=0.d0;tsd=0.d0;valid=.true.
  if(.not.all(ieee_is_finite([speed,ustar,settling_velocity,h0,h1, &
      equilibrium_concentration])).or.min(speed,ustar,settling_velocity)<0.d0.or. &
     min(h0,h1)<=0.d0.or.equilibrium_concentration<0.d0.or. &
     suspended_adaptation_factor<=0.d0) then
   valid=.false.;return
  endif
  wsl=max(1.d-3,settling_velocity)
  if(speed>1.d-6.and.ustar>1.d-6) then
   u=min(ustar/speed,.15d0)
   if(ustar>wsl) then
    w=wsl/ustar
   else
    w=1.d0
   endif
   x=w;x2=x*x;x3=x2*x
   exponent=(real(1.547,8)-real(20.12,8)*u)*x3+ &
    (real(326.832,8)*u**real(2.2047,8)-real(.2,8))*x2+ &
    (real(.1385,8)*log(u)-real(6.4061,8))*x+ &
    real(.5467,8)*u+real(2.1963,8)
   tsd=x*exp(exponent)
   if(.not.ieee_is_finite(tsd).or.tsd<=0.d0) then
    valid=.false.;return
   endif
   hots=wsl/(tsd*suspended_adaptation_factor)
   explicit_source=equilibrium_concentration*hots/h0
   ! sink + sour_im = wsl/h1 + (hots-wsl)/h1.
   total_implicit_sink=hots/h1
  else
   ! soursin_2d leaves its adaptation source inactive and applies only the
   ! implicit settling sink when current or shear velocity is absent.
   total_implicit_sink=wsl/h1
  endif
  valid=all(ieee_is_finite([explicit_source,total_implicit_sink])).and. &
   min(explicit_source,total_implicit_sink)>=0.d0
 end subroutine

 subroutine source(state,dt,depth_before)
  real(8), intent(inout) :: state(nstate,nc)
  real(8), intent(in) :: dt
  real(8), intent(in) :: depth_before(nc)
  real(8) :: h,s,b,rate,eq,us,tau,caks,aks,tsd,total,speed,availability(nfrac)
  real(8) :: suspended_qx,suspended_qy,bed_qx,bed_qy,explicit_source
  real(8) :: composed_exchange_bed(2),retained_exchange_suspended(2), &
   signed_exchange_rate
  real(8) :: h0_actual,h0_ref,h1_ref,s_before
  real(8) :: erosion_velocity(nfrac),deposition_rate(nfrac),fraction_share(nfrac),mud_fraction,erosion_source,decay,bed_tau
  real(8) :: effective_ws(nfrac),total_concentration,hindered_factor,roughness_n,dune_height,dune_length,total_ks
  real(8) :: friction_denominator,friction_rate
 integer :: c,f,l,si,bi,work,nwork,bedform_iterations,composition_status
  logical :: valid
  if(deck_version>=17.and.bank_erosion_fraction>0.and.amr_step_level<0) then
   bank_depth=state(1,:)
   do f=1,nfrac
    bank_previous(f,:)=state(bed_index(f),:)
    do l=2,nbed_layers
     bank_previous(f,:)=bank_previous(f,:)+state(bed_layer_index(l,f),:)
    enddo
   enddo
  endif
  nwork=nc
  if(amr_step_level>=0) nwork=amr_level_cell_count(amr_step_level)
  if(exchange_mode==4) suspended_exchange_cell_trial=0.d0
  !$omp parallel do schedule(static,256) if(nwork*nfrac>=8192.or.(hydraulic_workflow_fast.and.omp_get_max_threads()<=8)) default(shared) &
  !$omp& private(work,c,f,si,bi,h,s,b,rate,eq,us,tau,caks,aks,tsd,total,speed,availability,erosion_velocity, &
  !$omp& deposition_rate,fraction_share,mud_fraction,erosion_source,decay,bed_tau,effective_ws, &
  !$omp& total_concentration,hindered_factor,roughness_n,dune_height,dune_length,total_ks,friction_denominator,friction_rate, &
  !$omp& suspended_qx,suspended_qy,bed_qx,bed_qy,explicit_source,h0_actual,h0_ref,h1_ref,s_before, &
  !$omp& composed_exchange_bed,retained_exchange_suspended,signed_exchange_rate,composition_status, &
  !$omp& bedform_iterations,valid)
  do work=1,nwork
   c=work
   if(amr_step_level>=0) c=amr_level_cells(work,amr_step_level)
   h=state(1,c)
   call bedform_properties(state,c,roughness_n,dune_height,dune_length,total_ks,bedform_iterations,valid)
   if(.not.valid) error stop 'invalid bedform roughness state'
   if(h<=dry_tolerance) then
    state(2:3,c)=0
   else
    speed=sqrt(sum(state(2:3,c)**2))/h
    if(hydraulic_roughness_mode/=braids_roughness_manning.or. &
       roughness_minimum_work_depth>0.d0) then
     call braids_roughness_friction_rate(hydraulic_roughness_mode,h,speed,roughness_n,grav, &
      max(manning_friction_chezy_minimum_depth,roughness_minimum_work_depth), &
      white_colebrook_chzk2d,friction_rate,valid)
     if(.not.valid) error stop 'invalid typed hydraulic friction state'
     state(2:3,c)=state(2:3,c)/(1+dt*friction_rate)
    else if(manning_friction_chezy_minimum_depth>0.d0.and.h<manning_friction_chezy_minimum_depth) then
     ! Stock FLOW computes Chezy from a 0.01 m work depth, but retains the
     ! physical depth in the momentum coefficient: g*n^2*|u|/[h*h_work^(1/3)].
     friction_denominator=h*manning_friction_chezy_minimum_depth**(1.d0/3.d0)
     state(2:3,c)=state(2:3,c)/(1+dt*grav*roughness_n**2*speed/friction_denominator)
    else
     ! Preserve the qualified legacy arithmetic exactly when the envelope is
     ! disabled or the physical depth is at/above the configured work depth.
     state(2:3,c)=state(2:3,c)/(1+dt*grav*roughness_n**2*speed/h**(4.d0/3))
    endif
   endif
   if(exchange_mode==0) cycle
   effective_ws=ws_fraction
   if(settling_mode==1.and.h>dry_tolerance) then
    total_concentration=0
    do f=1,nfrac
     total_concentration=total_concentration+max(0.d0,state(suspended_index(f),c))/h
    enddo
    ! Richardson-Zaki/Mehta closure used by the reference morphology source.
    ! The clamp prevents a negative fall velocity above the reference
    ! concentration and one shared factor couples all suspended fractions.
    hindered_factor=max(0.d0,1.d0-total_concentration/hindered_settling_reference_concentration)**5
    effective_ws=ws_fraction*hindered_factor
   endif
   do f=1,nfrac
    availability(f)=bed_availability(state,f,c)
   enddo
   if(exchange_mode==4) then
    ! The admitted v47 slice is fixed-grid and one-fraction.  Re-evaluate law
    ! 11 from the same accepted endpoint state used for speed and total-shear
    ! ustar, after the local hydraulic-friction update above.  Suspended mass
    ! has already completed conservative face advection.
    do f=1,nfrac
     si=suspended_index(f);bi=bed_index(f);s_before=state(si,c)
     s=state(si,c);b=state(bi,c);total=s+b
     ! M2-C gives a dry control volume one unambiguous owner for sediment:
     ! the active bed.  This is an exact local phase transfer, so class mass
     ! is unchanged and the accepted exchange ledger records both signs.  It
     ! is gated by the v49 ownership declaration; historical v47/v48 decks
     ! retain their frozen fixed-grid interpretation.
     if(h<=dry_tolerance.and.flow2dh_amr_ownership_declaration) then
      state(si,c)=0.d0;state(bi,c)=total
      suspended_exchange_cell_trial(f,c)=-s_before
      cycle
     endif
     ! In this bounded slice BRAIDS endpoint wetness is the explicit Boolean
     ! provider for FLOW kfs.  SedThr is an additional strict depth gate and
     ! must not be replaced by the hydraulic dry tolerance.
     if(h<=dry_tolerance.or.h<=sedthr_fraction(f)) cycle
     speed=sqrt(sum(state(2:3,c)**2))/h
     call bed_capacity(state(1:3,c),f,bed_qx,bed_qy,tau,valid,1.d0, &
      roughness_n=roughness_n,suspended_qx=suspended_qx,suspended_qy=suspended_qy)
     if(.not.valid) error stop 'invalid law-11 suspended capacity in generic FLOW exchange'
     h0_actual=depth_before(c);h0_ref=max(h0_actual,1.d-2);h1_ref=max(h,1.d-2)
     if(channel_composition_declaration) then
      call compose_live_law11_channels(state(1:3,c),h1_ref,f,[bed_qx,bed_qy], &
       [suspended_qx,suspended_qy],composed_exchange_bed,retained_exchange_suspended, &
       signed_exchange_rate,eq,valid,composition_status)
      if(.not.valid) error stop 'invalid v51 law-11 endpoint channel composition'
      ! The active-surface fraction remains downstream of channel composition
      ! and is applied exactly once to the signed equilibrium concentration.
      eq=eq*capacity_surface_fraction(state,f,c)
     else
      eq=sqrt(suspended_qx*suspended_qx+suspended_qy*suspended_qy)/ &
       (speed+1.d-6)/h1_ref*capacity_surface_fraction(state,f,c)
     endif
     us=sqrt(max(hydraulic_bed_shear(h,state(2,c),state(3,c),roughness_n)/rho,0.d0))
     call flow2dh_galappatti_terms(speed,us,effective_ws(f),h0_ref,h1_ref,eq, &
      explicit_source,rate,tsd,valid)
     if(.not.valid) error stop 'invalid generic FLOW Galappatti exchange state'
     ! FLOW treats the equilibrium source explicitly and the combined
     ! settling/adaptation sink implicitly in mass form.  The final cap is a
     ! BRAIDS finite-inventory safety limiter, not red_soursin equivalence.
     s=(s+dt*h0_actual*explicit_source)/(1.d0+dt*rate)
     state(si,c)=min(total,max(0.d0,s));state(bi,c)=total-state(si,c)
     suspended_exchange_cell_trial(f,c)=state(si,c)-s_before
    enddo
    cycle
   endif
   if(exchange_mode==3.and.h>dry_tolerance) then
    call surface_composition(state,c,fraction_share,mud_fraction)
    bed_tau=sediment_bed_shear(h,state(2,c),state(3,c),roughness_n,mud_fraction)
    erosion_velocity=0;deposition_rate=0
    do f=1,nfrac
     if(sediment_type_fraction(f)==1) then
      call braids_partheniades_krone(bed_tau,bulk_fraction(f),effective_ws(f),h,critical_erosion_fraction(f), &
       critical_deposition_fraction(f),erosion_parameter_fraction(f),erosion_exponent_fraction(f), &
       erosion_velocity(f),deposition_rate(f),valid)
      if(.not.valid) error stop 'invalid cohesive exchange state'
     else if(h>sedthr_fraction(f)) then
      call braids_vr1993_coefficients(h,state(2,c),state(3,c),effective_ws(f),rhos_fraction(f),rho, &
       constitutive_kinematic_viscosity,d50_fraction(f),d90_fraction(f),sediment_z0(h,roughness_n,f), &
       reference_height_fraction(f),.65d0,1.d0,.true.,dstar_fraction(f),taucr_fraction(f), &
       eq,rate,us,tau,caks,aks,tsd)
      erosion_velocity(f)=rate*h*eq/bulk_fraction(f);deposition_rate(f)=rate
     endif
    enddo
    call braids_sand_mud_erosion(nfrac,sediment_type_fraction,erosion_velocity,fraction_share,mud_fraction,critical_mud_fraction)
    do f=1,nfrac
     si=suspended_index(f);bi=bed_index(f);s=state(si,c);b=state(bi,c);total=s+b
     erosion_source=availability(f)*bulk_fraction(f)*erosion_velocity(f)
     if(deposition_rate(f)>0) then
      decay=exp(-deposition_rate(f)*dt)
      s=erosion_source/deposition_rate(f)+(s-erosion_source/deposition_rate(f))*decay
     else
      s=s+erosion_source*dt
     endif
     state(si,c)=min(total,max(0.d0,s));state(bi,c)=total-state(si,c)
    enddo
    cycle
   endif
   do f=1,nfrac
    si=suspended_index(f);bi=bed_index(f)
    s=state(si,c);b=state(bi,c);total=s+b
    if(h<=dry_tolerance) then
     s=0
    else if(exchange_mode==1) then
     s=s*exp(-effective_ws(f)*dt/h)
    else if(h>sedthr_fraction(f)) then
     call braids_vr1993_coefficients(h,state(2,c),state(3,c),effective_ws(f),rhos_fraction(f),rho, &
      constitutive_kinematic_viscosity,d50_fraction(f),d90_fraction(f),sediment_z0(h,roughness_n,f), &
      reference_height_fraction(f),.65d0,1.d0,.true.,dstar_fraction(f),taucr_fraction(f), &
      eq,rate,us,tau,caks,aks,tsd)
     eq=eq*availability(f)
     s=min(total,max(0.d0,h*eq+(s-h*eq)*exp(-rate*dt)))
    endif
    state(si,c)=s;state(bi,c)=total-s
   enddo
  enddo
  !$omp end parallel do
  if(deck_version>=17.and.bank_erosion_fraction>0.and.amr_step_level<0) &
   call redistribute_bank_inventory(bank_depth,bank_previous,state)
  call rebalance_active_layer(state)
 end subroutine

 subroutine accumulate_suspended_exchange_trial(level)
  ! Accumulate one accepted local source evaluation into the macro-step
  ! transaction.  Adaptive substeps call source independently by level; the
  ! old cell buffer is therefore only scratch and may not itself be used as
  ! the accepted global ledger.
  integer,intent(in) :: level
  integer :: f,c,k
  real(8) :: accepted
  if(exchange_mode/=4.or..not.flow2dh_amr_ownership_declaration) return
  do f=1,nfrac
   accepted=0.d0
   if(level<0) then
    do c=1,nc
     accepted=accepted+area(c)*suspended_exchange_cell_trial(f,c)
    enddo
   else
    do k=1,amr_level_cell_count(level)
     c=amr_level_cells(k,level)
     accepted=accepted+area(c)*suspended_exchange_cell_trial(f,c)
    enddo
   endif
   suspended_exchange_step_trial(1,f)= &
    suspended_exchange_step_trial(1,f)+accepted
   suspended_exchange_step_trial(2,f)= &
    suspended_exchange_step_trial(2,f)-accepted
  enddo
 end subroutine

 subroutine deposit_dry_suspended_after_reflux(state)
  ! Reflux and prescribed-boundary reconciliation occur after each level's
  ! local source.  Close the same dry-cell ownership rule once more on the
  ! synchronized macro state and account for this accepted phase transfer in
  ! the still-uncommitted source transaction.
  real(8),intent(inout) :: state(nstate,nc)
  integer :: f,c,si,bi
  real(8) :: suspended
  if(exchange_mode/=4.or..not.flow2dh_amr_ownership_declaration) return
  do c=1,nc
   if(state(1,c)>dry_tolerance) cycle
   state(2:3,c)=0.d0
   do f=1,nfrac
    si=suspended_index(f);bi=bed_index(f);suspended=state(si,c)
    if(suspended<=0.d0) cycle
    state(si,c)=0.d0;state(bi,c)=state(bi,c)+suspended
    suspended_exchange_step_trial(1,f)= &
     suspended_exchange_step_trial(1,f)-area(c)*suspended
    suspended_exchange_step_trial(2,f)= &
     suspended_exchange_step_trial(2,f)+area(c)*suspended
   enddo
  enddo
 end subroutine

 subroutine commit_suspended_exchange()
  integer :: f,c
  real(8) :: accepted
  if(exchange_mode/=4) return
  if(flow2dh_amr_ownership_declaration) then
   suspended_exchange_cumulative=suspended_exchange_cumulative+ &
    suspended_exchange_step_trial
   return
  endif
  do f=1,nfrac
   accepted=0.d0
   ! Fixed cell order makes the accepted source ledger independent of the
   ! OpenMP schedule used by the cell-local exchange operator.
   do c=1,nc
    accepted=accepted+area(c)*suspended_exchange_cell_trial(f,c)
   enddo
   suspended_exchange_cumulative(1,f)= &
    suspended_exchange_cumulative(1,f)+accepted
   suspended_exchange_cumulative(2,f)= &
    suspended_exchange_cumulative(2,f)-accepted
  enddo
 end subroutine

 ! Mode 0: d(hu)/dt = div(nu*h*grad(u)); mode 1: du/dt = div(nu*grad(u)).
 ! Pairwise conductances use harmonic wet depth only in conservative mode.
 ! The velocity Laplacian closes a perched shoreline face whose reconstructed
 ! centre-stage aperture is dry, with a half-tolerance linear transition.
 ! Zero traction on external and wet/dry faces (free slip, not no slip).
 ! Orthogonal-grid Laplacian; the positive two-point cut-cell extension is
 ! deliberately not claimed consistent on arbitrary nonorthogonal grids.
 ! Apply once to all synchronized leaves, never separately to AMR levels.
 ! SSPRK2 substeps obey the convex-combination bound independently of CFL.
 subroutine diffuse_momentum(state,dt)
  real(8),intent(inout) :: state(nstate,nc)
  real(8),intent(in) :: dt
  real(8) :: conductance(nf),rate(nc),flux(2,nf),initial(2,nc),stage(2,nc),bed_elevation(nc)
  real(8) :: h1,h2,distance,ds,maximum_rate,change(2),started,h_face,activation
  integer :: f,l,r,c,k,p,substep,count,g,layer
  if(horizontal_viscosity<=0.or.dt<=0) return
  if(horizontal_viscosity_geometry==1) then
   call diffuse_momentum_nonorthogonal(state,dt)
   return
  endif
  if(profile_enabled) started=omp_get_wtime()
  conductance=0;rate=0
  if(horizontal_viscosity_mode==1) then
   call effective_bed_elevation(state,bed_elevation)
  endif
  !$omp parallel do schedule(static) if(nf>=4096) default(shared) &
  !$omp& private(f,l,r,h1,h2,distance,h_face,activation)
  do f=1,nf
   l=owner(f);r=neighbour(f)
   if(r<=0) cycle
   h1=state(1,l);h2=state(1,r)
   if(min(h1,h2)<=dry_tolerance) cycle
   distance=sqrt(sum((xy(:,r)-xy(:,l))**2))
   if(distance<=0) cycle
   if(horizontal_viscosity_mode==0) then
    conductance(f)=horizontal_viscosity*length(f)*(2.d0/(1.d0/h1+1.d0/h2))/distance
   else
    h_face=.5d0*((bed_elevation(l)+h1)+(bed_elevation(r)+h2))- &
     max(bed_elevation(l),bed_elevation(r))
    if(h_face<=.5d0*dry_tolerance) cycle
    conductance(f)=horizontal_viscosity*length(f)/distance
    if(h_face<dry_tolerance) then
     activation=(h_face-.5d0*dry_tolerance)/(.5d0*dry_tolerance)
     conductance(f)=conductance(f)*max(0.d0,min(1.d0,activation))
    endif
   endif
  enddo
  !$omp end parallel do
  !$omp parallel do schedule(static) if(nc>=4096) default(shared) private(c,k,f)
  do c=1,nc
   if(state(1,c)<=dry_tolerance) cycle
   do k=offset(c),offset(c+1)-1
    f=faces(k);rate(c)=rate(c)+conductance(f)
   enddo
   if(horizontal_viscosity_mode==0) then
    rate(c)=rate(c)/(area(c)*state(1,c))
   else
    rate(c)=rate(c)/area(c)
   endif
  enddo
  !$omp end parallel do
  maximum_rate=maxval(rate)
  if(.not.ieee_is_finite(maximum_rate).or.dt*maximum_rate>1.d8) &
   error stop 'horizontal diffusion step is excessive; inspect cut-cell geometry'
  count=max(1,ceiling(dt*maximum_rate/.9d0));ds=dt/count
  do substep=1,count
   initial=state(2:3,:);stage=initial
   do p=1,2
    !$omp parallel do schedule(static) if(nf>=4096) default(shared) private(f,l,r)
    do f=1,nf
     flux(:,f)=0
     if(conductance(f)<=0) cycle
     l=owner(f);r=neighbour(f)
     flux(:,f)=conductance(f)*(stage(:,r)/state(1,r)-stage(:,l)/state(1,l))
    enddo
    !$omp end parallel do
    !$omp parallel do schedule(static) if(nc>=4096) default(shared) private(c,k,f,change)
    do c=1,nc
     change=0
     do k=offset(c),offset(c+1)-1
      f=faces(k);change=change+signs(k)*flux(:,f)
     enddo
     if(horizontal_viscosity_mode==0) then
      stage(:,c)=stage(:,c)+ds*change/area(c)
     else
      stage(:,c)=stage(:,c)+ds*state(1,c)*change/area(c)
     endif
     if(p==2) state(2:3,c)=.5d0*(initial(:,c)+stage(:,c))
    enddo
    !$omp end parallel do
   enddo
  enddo
  diffusion_substeps=diffusion_substeps+count
  if(profile_enabled) profile_diffusion_s=profile_diffusion_s+omp_get_wtime()-started
 end subroutine

 ! V35 velocity Laplacian on general cut-cell geometry. Certified symmetric,
 ! orthogonal interior faces retain the positive two-point flux. Faces which
 ! are skew, have unequal centroid-to-face normal distances, or touch a
 ! no-flux cell use a boundary-compatible quadratic recovery. For each trace,
 ! an unlimited Cartesian polynomial is fitted to unique active wet cells in
 ! a two-ring graph patch. External and closed wet/dry faces incident to the
 ! trace cell impose the homogeneous normal derivative at their physical
 ! midpoints. The recovered owner/neighbour normal derivatives are evaluated
 ! at the physical shared face and averaged.
 !
 ! One face flux is gathered with opposite signs, conserving area-integrated
 ! velocity (and physical momentum at constant depth). The v34 branch above
 ! remains the archived two-point implementation. Since multipoint recovery
 ! is not an M-matrix in general, the recovered-minus-two-point correction is
 ! scaled only when its global quadratic form would retain less than five
 ! percent of primary graph dissipation. The exact recovery coefficient L1
 ! norm enters the substep bound, and every Euler stage is checked against the
 ! area-weighted velocity energy. Rank-deficient quadratic patches retry the
 ! same constrained least-squares problem with an affine gradient. Only
 ! patches which cannot identify even that gradient (or oversized patches)
 ! fall back locally to the dissipative two-point flux. Quadratic recovery
 ! remains the first choice wherever it is valid.
 subroutine diffuse_momentum_nonorthogonal(state,dt)
  real(8),intent(inout) :: state(nstate,nc)
  real(8),intent(in) :: dt
  real(8) :: conductance(nf),face_diffusivity(nf),tangent_offset(nf),normal_asymmetry(nf),face_bound(nf)
  real(8) :: rate(nc),bed_elevation(nc),velocity(2,nc)
  real(8) :: primary(2,nf),correction(2,nf),flux(2,nf)
  real(8) :: initial(2,nc),stage(2,nc),candidate(2,nc)
  real(8),allocatable :: recovery_inverse(:,:,:),recovery_polynomial(:,:,:)
  real(8),allocatable :: recovery_affine_inverse(:,:,:),recovery_affine_polynomial(:,:,:)
  real(8) :: h1,h2,h_face,activation,d(2),tangent(2),dn,dlength,dl,dr,geometric_tolerance
  real(8) :: sl,sr,recovery_bound,recovered_flux(2),left_trace(2),right_trace(2)
  real(8) :: maximum_rate,base_step,remaining,ds
  real(8) :: change(2),du(2),d0,dc,correction_factor,energy_before,energy_after,tolerance,started
  integer,allocatable :: recovery_slot(:),recovery_cells(:),recovery_order(:)
  integer :: f,l,r,c,k,p,count,g,layer,attempt,nrecovery,slot,bad_geometry
  integer :: requested_recovery_faces,active_recovery_faces,active_affine_faces
  logical,allocatable :: boundary_cell(:),corrected_face(:),correction_cell(:),recovery_valid(:)
  logical :: accepted
  real(8),parameter :: dissipation_floor=.05d0
  integer :: gpu_unit,gpu_stencil(max_diffusion_recovery_stencil),gpu_nstencil
  integer,allocatable :: gpu_stencils(:,:),gpu_stencil_counts(:)
  logical :: gpu_complete
  if(horizontal_viscosity<=0.or.dt<=0) return
  if(profile_enabled) started=omp_get_wtime()
  conductance=0;face_diffusivity=0;tangent_offset=0;normal_asymmetry=0;face_bound=0;rate=0
  call effective_bed_elevation(state,bed_elevation)
  bad_geometry=0
  !$omp parallel do schedule(static) if(nf>=4096) default(shared) reduction(max:bad_geometry) &
  !$omp& private(f,l,r,h1,h2,h_face,activation,d,tangent,dn,dlength,dl,dr,geometric_tolerance)
  do f=1,nf
   l=owner(f);r=neighbour(f)
   if(r<=0) cycle
   d=xy(:,r)-xy(:,l);dlength=sqrt(sum(d*d));dn=dot_product(d,normal(:,f))
   if(.not.ieee_is_finite(dn).or.dn<=64*epsilon(1.d0)*max(dlength,tiny(1.d0))) then
    bad_geometry=1;cycle
   endif
   tangent=[-normal(2,f),normal(1,f)]
   tangent_offset(f)=dot_product(d,tangent)
   dl=dot_product(mid(:,f)-xy(:,l),normal(:,f))
   dr=dot_product(xy(:,r)-mid(:,f),normal(:,f))
   normal_asymmetry(f)=dl-dr
   ! Coordinate subtraction can lose ulps when a small local mesh is stored
   ! in a projected CRS with million-metre ordinates. Classify only resolved
   ! geometric skew/asymmetry; this also makes rigid rotations choose the
   ! identical correction envelope.
   geometric_tolerance=512*epsilon(1.d0)*max(1.d0,dlength,abs(dl)+abs(dr), &
    sqrt(sum(xy(:,l)*xy(:,l))),sqrt(sum(xy(:,r)*xy(:,r))),sqrt(sum(mid(:,f)*mid(:,f))))
   if(abs(tangent_offset(f))<=geometric_tolerance) tangent_offset(f)=0
   if(abs(normal_asymmetry(f))<=geometric_tolerance) normal_asymmetry(f)=0
   h1=state(1,l);h2=state(1,r)
   if(min(h1,h2)<=dry_tolerance) cycle
   h_face=.5d0*((bed_elevation(l)+h1)+(bed_elevation(r)+h2))- &
    max(bed_elevation(l),bed_elevation(r))
   if(h_face<=.5d0*dry_tolerance) cycle
   activation=1
   if(h_face<dry_tolerance) &
    activation=max(0.d0,min(1.d0,(h_face-.5d0*dry_tolerance)/(.5d0*dry_tolerance)))
   conductance(f)=horizontal_viscosity*length(f)*activation/dn
   face_diffusivity(f)=conductance(f)*dn
  enddo
  !$omp end parallel do
  if(bad_geometry/=0) error stop 'invalid internal-face normal orientation for nonorthogonal diffusion'

  ! The correction envelope is geometric except for the fixed wet/perched
  ! mask of this diffusion call. A cell touching any inactive face supplies
  ! physical homogeneous-Neumann rows to its local recovery.
  allocate(boundary_cell(nc),corrected_face(nf),correction_cell(nc),recovery_slot(nc))
  boundary_cell=.false.;corrected_face=.false.;correction_cell=.false.;recovery_slot=0
  !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) private(c,k,f,r)
  do c=1,nc
   if(state(1,c)<=dry_tolerance) cycle
   do k=offset(c),offset(c+1)-1
    f=faces(k);r=neighbour(f)
    if(signs(k)<0) r=owner(f)
    if(r<=0.or.conductance(f)<=0) then
     boundary_cell(c)=.true.;exit
    endif
   enddo
  enddo
  !$omp end parallel do
  !$omp parallel do schedule(static) if(nf>=4096) default(shared) private(f,l,r)
  do f=1,nf
   if(conductance(f)<=0) cycle
   l=owner(f);r=neighbour(f)
   corrected_face(f)=tangent_offset(f)/=0.or.normal_asymmetry(f)/=0.or. &
    boundary_cell(l).or.boundary_cell(r)
  enddo
  !$omp end parallel do
  requested_recovery_faces=0
  do f=1,nf
   if(corrected_face(f)) requested_recovery_faces=requested_recovery_faces+1
  enddo
  do f=1,nf
   if(.not.corrected_face(f)) cycle
   correction_cell(owner(f))=.true.;correction_cell(neighbour(f))=.true.
  enddo
  nrecovery=0
  do c=1,nc
   if(correction_cell(c)) nrecovery=nrecovery+1
  enddo
  allocate(recovery_cells(nrecovery),recovery_inverse(5,5,nrecovery), &
   recovery_polynomial(2,5,nrecovery),recovery_affine_inverse(2,2,nrecovery), &
   recovery_affine_polynomial(2,2,nrecovery),recovery_order(nrecovery), &
   recovery_valid(nrecovery))
  recovery_inverse=0;recovery_polynomial=0
  recovery_affine_inverse=0;recovery_affine_polynomial=0
  recovery_order=0;recovery_valid=.false.
  slot=0
  do c=1,nc
   if(.not.correction_cell(c)) cycle
   slot=slot+1;recovery_slot(c)=slot;recovery_cells(slot)=c
  enddo
  !$omp parallel do schedule(static,64) if(nrecovery>=512) default(shared) private(slot,c)
  do slot=1,nrecovery
   c=recovery_cells(slot)
   call diffusion_quadratic_matrix(c,conductance,recovery_inverse(:,:,slot),recovery_valid(slot))
   if(recovery_valid(slot)) then
    recovery_order(slot)=2
   else
    call diffusion_affine_matrix(c,conductance,recovery_affine_inverse(:,:,slot),recovery_valid(slot))
    if(recovery_valid(slot)) recovery_order(slot)=1
   endif
  enddo
  !$omp end parallel do
  ! A conservative infinity-norm bound for the complete corrected operator.
  ! The recovered flux is a convex blend with the primary graph flux, so the
  ! larger of the primary norm and the conservative sum of both recovered
  ! trace L1 norms bounds every limiter state.
  !$omp parallel do schedule(static) if(nf>=4096) default(shared) &
  !$omp& private(f,l,r,sl,sr,recovery_bound)
  do f=1,nf
   if(conductance(f)<=0) cycle
   l=owner(f);r=neighbour(f);face_bound(f)=2*conductance(f)
   if(.not.corrected_face(f)) cycle
   if(recovery_slot(l)<=0.or.recovery_slot(r)<=0) cycle
   if(.not.recovery_valid(recovery_slot(l)).or..not.recovery_valid(recovery_slot(r))) then
    corrected_face(f)=.false.;cycle
   endif
   if(recovery_order(recovery_slot(l))==2) then
    sl=diffusion_quadratic_l1(l,f,conductance,recovery_inverse(:,:,recovery_slot(l)))
   else
    sl=diffusion_affine_l1(l,f,conductance,recovery_affine_inverse(:,:,recovery_slot(l)))
   endif
   if(recovery_order(recovery_slot(r))==2) then
    sr=diffusion_quadratic_l1(r,f,conductance,recovery_inverse(:,:,recovery_slot(r)))
   else
    sr=diffusion_affine_l1(r,f,conductance,recovery_affine_inverse(:,:,recovery_slot(r)))
   endif
   if(.not.ieee_is_finite(sl).or..not.ieee_is_finite(sr)) then
    corrected_face(f)=.false.;cycle
   endif
   recovery_bound=.5d0*face_diffusivity(f)*(sl+sr)
   if(.not.ieee_is_finite(recovery_bound).or.recovery_bound<0) then
    corrected_face(f)=.false.;cycle
   endif
   face_bound(f)=max(face_bound(f),recovery_bound)
  enddo
  !$omp end parallel do
  active_recovery_faces=0;active_affine_faces=0
  do f=1,nf
   if(.not.corrected_face(f)) cycle
   active_recovery_faces=active_recovery_faces+1
   l=owner(f);r=neighbour(f)
   if(recovery_order(recovery_slot(l))==1.or.recovery_order(recovery_slot(r))==1) &
    active_affine_faces=active_affine_faces+1
  enddo
  diffusion_recovered_faces=diffusion_recovered_faces+active_recovery_faces
  diffusion_affine_recovered_faces=diffusion_affine_recovered_faces+active_affine_faces
  diffusion_recovery_fallbacks=diffusion_recovery_fallbacks+ &
   requested_recovery_faces-active_recovery_faces
  !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) private(c,k,f)
  do c=1,nc
   if(state(1,c)<=dry_tolerance) cycle
   do k=offset(c),offset(c+1)-1
    ! SSPRK2 is stable through two on the negative real axis. One half of
    ! the absolute row-sum bound therefore retains the established 0.9
    ! safety convention and exactly recovers the orthogonal v34 rate.
    f=faces(k);rate(c)=rate(c)+.5d0*face_bound(f)
   enddo
   rate(c)=rate(c)/area(c)
  enddo
  !$omp end parallel do
  maximum_rate=maxval(rate)
  if(.not.ieee_is_finite(maximum_rate).or.dt*maximum_rate>1.d8) &
   error stop 'nonorthogonal horizontal diffusion step is excessive; inspect cut-cell geometry'
  if(gpu_export_diffusion) then
   call gpu_open(gpu_unit,'coupled_diffusion.bin')
  call gpu_put_r(gpu_unit,'conductance',reshape(conductance,[size(conductance)]),[nf])
  call gpu_put_r(gpu_unit,'face_diffusivity',reshape(face_diffusivity,[size(face_diffusivity)]),[nf])
  call gpu_put_r(gpu_unit,'tangent_offset',reshape(tangent_offset,[size(tangent_offset)]),[nf])
  call gpu_put_r(gpu_unit,'normal_asymmetry',reshape(normal_asymmetry,[size(normal_asymmetry)]),[nf])
  call gpu_put_r(gpu_unit,'face_bound',reshape(face_bound,[size(face_bound)]),[nf])
  call gpu_put_r(gpu_unit,'rate',reshape(rate,[size(rate)]),[nc])
  call gpu_put_r(gpu_unit,'recovery_inverse',reshape(recovery_inverse,[size(recovery_inverse)]),[5,5,nrecovery])
  call gpu_put_r(gpu_unit,'recovery_affine_inverse',reshape(recovery_affine_inverse,[size(recovery_affine_inverse)]),[2,2,nrecovery])
  call gpu_put_i(gpu_unit,'corrected_face',reshape(merge(1,0,corrected_face),[size(merge(1,0,corrected_face))]),[nf])
  call gpu_put_i(gpu_unit,'recovery_slot',reshape(recovery_slot-1,[size(recovery_slot-1)]),[nc])
  call gpu_put_i(gpu_unit,'recovery_cells',reshape(recovery_cells-1,[size(recovery_cells-1)]),[nrecovery])
  call gpu_put_i(gpu_unit,'recovery_order',reshape(recovery_order,[size(recovery_order)]),[nrecovery])
  call gpu_put_i(gpu_unit,'recovery_valid',reshape(merge(1,0,recovery_valid),[size(merge(1,0,recovery_valid))]),[nrecovery])
   allocate(gpu_stencils(max_diffusion_recovery_stencil,nrecovery),gpu_stencil_counts(nrecovery))
   gpu_stencils=-1
   do slot=1,nrecovery
    call diffusion_recovery_stencil(recovery_cells(slot),conductance,gpu_stencil,gpu_nstencil,gpu_complete)
    if(.not.gpu_complete) error stop 'GPU diffusion recovery stencil incomplete'
    gpu_stencil_counts(slot)=gpu_nstencil
    gpu_stencils(:,slot)=gpu_stencil-1
   enddo
  call gpu_put_i(gpu_unit,'recovery_stencils',reshape(gpu_stencils,[size(gpu_stencils)]),[max_diffusion_recovery_stencil,nrecovery])
  call gpu_put_i(gpu_unit,'recovery_stencil_counts',reshape(gpu_stencil_counts,[size(gpu_stencil_counts)]),[nrecovery])
  call gpu_put_r(gpu_unit,'maximum_rate',[maximum_rate],[integer::])
   call gpu_finish(gpu_unit)
   return
  endif
  base_step=dt
  if(maximum_rate>0) base_step=min(dt,.9d0/maximum_rate)
  remaining=dt;count=0
  do while(remaining>0)
   ds=min(remaining,base_step);attempt=0
   do
    attempt=attempt+1
    if(attempt>60.or.ds<=epsilon(1.d0)*max(dt,tiny(1.d0))) &
     error stop 'nonorthogonal diffusion energy step reached machine precision'
    initial=state(2:3,:);stage=initial;accepted=.true.
    do p=1,2
     call diffusion_velocity_field(stage,state(1,:),velocity)
     recovery_polynomial=0;recovery_affine_polynomial=0
     !$omp parallel do schedule(static,64) if(nrecovery>=512) default(shared) private(slot,c)
     do slot=1,nrecovery
      if(.not.recovery_valid(slot)) cycle
      c=recovery_cells(slot)
      if(recovery_order(slot)==2) then
       call diffusion_quadratic_coefficients(c,conductance,recovery_inverse(:,:,slot), &
        velocity,recovery_polynomial(:,:,slot))
       if(.not.all(ieee_is_finite(recovery_polynomial(:,:,slot)))) then
        recovery_polynomial(:,:,slot)=0;recovery_valid(slot)=.false.
       endif
      else
       call diffusion_affine_coefficients(c,conductance,recovery_affine_inverse(:,:,slot), &
        velocity,recovery_affine_polynomial(:,:,slot))
       if(.not.all(ieee_is_finite(recovery_affine_polynomial(:,:,slot)))) then
        recovery_affine_polynomial(:,:,slot)=0;recovery_valid(slot)=.false.
       endif
      endif
     enddo
     !$omp end parallel do
     !$omp parallel do schedule(static) if(nf>=4096) default(shared) &
     !$omp& private(f,l,r,du,recovered_flux,left_trace,right_trace)
     do f=1,nf
      primary(:,f)=0;correction(:,f)=0
      if(conductance(f)<=0) cycle
      l=owner(f);r=neighbour(f)
      du=velocity(:,r)-velocity(:,l)
      primary(:,f)=conductance(f)*du
      if(.not.corrected_face(f)) cycle
      if(.not.recovery_valid(recovery_slot(l)).or..not.recovery_valid(recovery_slot(r))) cycle
      if(recovery_order(recovery_slot(l))==2) then
       call diffusion_quadratic_trace(l,f,recovery_polynomial(:,:,recovery_slot(l)),left_trace)
      else
       call diffusion_affine_trace(l,f,recovery_affine_polynomial(:,:,recovery_slot(l)),left_trace)
      endif
      if(recovery_order(recovery_slot(r))==2) then
       call diffusion_quadratic_trace(r,f,recovery_polynomial(:,:,recovery_slot(r)),right_trace)
      else
       call diffusion_affine_trace(r,f,recovery_affine_polynomial(:,:,recovery_slot(r)),right_trace)
      endif
      recovered_flux=.5d0*face_diffusivity(f)*(left_trace+right_trace)
      correction(:,f)=recovered_flux-primary(:,f)
     enddo
     !$omp end parallel do
     d0=0;dc=0
     do f=1,nf
      if(conductance(f)<=0) cycle
      l=owner(f);r=neighbour(f);du=velocity(:,r)-velocity(:,l)
      d0=d0+dot_product(du,primary(:,f));dc=dc+dot_product(du,correction(:,f))
     enddo
     if(.not.ieee_is_finite(d0).or..not.ieee_is_finite(dc).or.d0<0) &
      error stop 'nonfinite quadratic form in nonorthogonal diffusion'
     correction_factor=1
     if(dc<-(1.d0-dissipation_floor)*d0) then
      if(d0>0) then
       correction_factor=(1.d0-dissipation_floor)*d0/(-dc)
      else
       correction_factor=0
      endif
     endif
     if(.not.ieee_is_finite(correction_factor).or.correction_factor<0.or.correction_factor>1) &
      error stop 'invalid nonorthogonal diffusion correction factor'
     diffusion_minimum_correction_factor=min(diffusion_minimum_correction_factor,correction_factor)
     flux=primary+correction_factor*correction
     energy_before=diffusion_velocity_energy(stage,state(1,:))
     if(.not.ieee_is_finite(energy_before)) error stop 'nonfinite initial horizontal diffusion energy'
     candidate=stage
     !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) private(c,k,f,change)
     do c=1,nc
      if(state(1,c)<=dry_tolerance) cycle
      change=0
      do k=offset(c),offset(c+1)-1
       f=faces(k);change=change+signs(k)*flux(:,f)
      enddo
      candidate(:,c)=stage(:,c)+ds*state(1,c)*change/area(c)
     enddo
     !$omp end parallel do
     energy_after=diffusion_velocity_energy(candidate,state(1,:))
     if(.not.ieee_is_finite(energy_after).or..not.all(ieee_is_finite(candidate))) then
      accepted=.false.;exit
     endif
     tolerance=128*epsilon(1.d0)*max(energy_before,tiny(1.d0))
     if(energy_after>energy_before+tolerance) then
      accepted=.false.;exit
     endif
     stage=candidate
    enddo
    if(accepted) then
     candidate=.5d0*(initial+stage)
     energy_before=diffusion_velocity_energy(initial,state(1,:))
     energy_after=diffusion_velocity_energy(candidate,state(1,:))
     if(.not.ieee_is_finite(energy_before).or..not.ieee_is_finite(energy_after).or. &
        .not.all(ieee_is_finite(candidate))) accepted=.false.
     tolerance=128*epsilon(1.d0)*max(energy_before,tiny(1.d0))
     if(energy_after>energy_before+tolerance) accepted=.false.
    endif
    if(accepted) exit
    ds=.5d0*ds
   enddo
   state(2:3,:)=candidate
   remaining=max(0.d0,remaining-ds)
   if(remaining<=64*epsilon(1.d0)*max(dt,tiny(1.d0))) remaining=0
   count=count+1
  enddo
  if(.not.all(ieee_is_finite(state(2:3,:)))) error stop 'nonfinite horizontal diffusion result'
  diffusion_substeps=diffusion_substeps+count
  if(profile_enabled) profile_diffusion_s=profile_diffusion_s+omp_get_wtime()-started
 end subroutine

 subroutine diffusion_velocity_field(momentum,depth,velocity)
  real(8),intent(in) :: momentum(2,nc),depth(nc)
  real(8),intent(out) :: velocity(2,nc)
  integer :: c
  velocity=0
  !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) private(c)
  do c=1,nc
   if(depth(c)>dry_tolerance) velocity(:,c)=momentum(:,c)/depth(c)
  enddo
  !$omp end parallel do
 end subroutine

 subroutine diffusion_recovery_stencil(c,conductance,stencil,nstencil,complete)
  integer,intent(in) :: c
  real(8),intent(in) :: conductance(nf)
  integer,intent(out) :: stencil(max_diffusion_recovery_stencil),nstencil
  logical,intent(out) :: complete
  integer :: k,f,r,index,first_ring_end
  stencil=0;nstencil=0;complete=.true.
  do k=offset(c),offset(c+1)-1
   f=faces(k);r=neighbour(f)
   if(signs(k)<0) r=owner(f)
   if(r<=0.or.conductance(f)<=0) cycle
   call diffusion_append_stencil(c,r,stencil,nstencil,complete)
   if(.not.complete) return
  enddo
  first_ring_end=nstencil
  do index=1,first_ring_end
   do k=offset(stencil(index)),offset(stencil(index)+1)-1
    f=faces(k);r=neighbour(f)
    if(signs(k)<0) r=owner(f)
    if(r<=0.or.conductance(f)<=0) cycle
    call diffusion_append_stencil(c,r,stencil,nstencil,complete)
    if(.not.complete) return
   enddo
  enddo
 end subroutine

 subroutine diffusion_append_stencil(center,value,stencil,nstencil,complete)
  integer,intent(in) :: center,value
  integer,intent(inout) :: stencil(max_diffusion_recovery_stencil),nstencil
  logical,intent(inout) :: complete
  integer :: j
  if(value==center) return
  do j=1,nstencil
   if(stencil(j)==value) return
  enddo
  if(nstencil>=max_diffusion_recovery_stencil) then
   complete=.false.;return
  endif
  nstencil=nstencil+1;stencil(nstencil)=value
 end subroutine

 subroutine diffusion_quadratic_matrix(c,conductance,inverse,valid)
  integer,intent(in) :: c
  real(8),intent(in) :: conductance(nf)
  real(8),intent(out) :: inverse(5,5)
  logical,intent(out) :: valid
  real(8) :: matrix(5,5),row(5),z(2),h,w
  integer :: stencil(max_diffusion_recovery_stencil),nstencil,index,k,f,r,i,j
  logical :: complete
  inverse=0;valid=.false.
  if(area(c)<=0) return
  h=sqrt(area(c))
  call diffusion_recovery_stencil(c,conductance,stencil,nstencil,complete)
  if(.not.complete) return
  matrix=0
  do index=1,nstencil
   r=stencil(index);z=(xy(:,r)-xy(:,c))/h
   w=1.d0/sum(z*z)
   row=[z(1),z(2),z(1)*z(1)/diffusion_sqrt_two,z(1)*z(2), &
    z(2)*z(2)/diffusion_sqrt_two]
   do i=1,5
    do j=1,5
     matrix(i,j)=matrix(i,j)+w*row(i)*row(j)
    enddo
   enddo
  enddo
  do k=offset(c),offset(c+1)-1
   f=faces(k);r=neighbour(f)
   if(signs(k)<0) r=owner(f)
   if(r>0.and.conductance(f)>0) cycle
   z=(mid(:,f)-xy(:,c))/h;w=length(f)/h
   row=[normal(1,f),normal(2,f),diffusion_sqrt_two*normal(1,f)*z(1), &
    normal(1,f)*z(2)+normal(2,f)*z(1),diffusion_sqrt_two*normal(2,f)*z(2)]
   do i=1,5
    do j=1,5
     matrix(i,j)=matrix(i,j)+w*row(i)*row(j)
    enddo
   enddo
  enddo
  call diffusion_inverse5(matrix,inverse,valid)
 end subroutine

 subroutine diffusion_inverse5(matrix,inverse,valid)
  real(8),intent(in) :: matrix(5,5)
  real(8),intent(out) :: inverse(5,5)
  logical,intent(out) :: valid
  real(8) :: augmented(5,10),temporary(10),diagonal(5),residual(5,5),pivot_value,factor
  integer :: i,j,pivot
  inverse=0;valid=.false.
  do i=1,5
   if(.not.ieee_is_finite(matrix(i,i)).or.matrix(i,i)<=0) return
   diagonal(i)=sqrt(matrix(i,i))
  enddo
  augmented=0
  do i=1,5
   do j=1,5
    augmented(i,j)=matrix(i,j)/(diagonal(i)*diagonal(j))
   enddo
  enddo
  if(.not.all(ieee_is_finite(augmented(:,1:5)))) return
  do i=1,5
   augmented(i,5+i)=1
  enddo
  do i=1,5
   pivot=i-1+maxloc(abs(augmented(i:5,i)),dim=1)
   pivot_value=abs(augmented(pivot,i))
   if(.not.ieee_is_finite(pivot_value).or.pivot_value<=1.d-12) return
   if(pivot/=i) then
    temporary=augmented(i,:);augmented(i,:)=augmented(pivot,:);augmented(pivot,:)=temporary
   endif
   augmented(i,:)=augmented(i,:)/augmented(i,i)
   do j=1,5
    if(j==i) cycle
    factor=augmented(j,i);augmented(j,:)=augmented(j,:)-factor*augmented(i,:)
   enddo
  enddo
  do i=1,5
   do j=1,5
    inverse(i,j)=augmented(i,5+j)/(diagonal(i)*diagonal(j))
   enddo
  enddo
  inverse=.5d0*(inverse+transpose(inverse))
  if(.not.all(ieee_is_finite(inverse))) then
   inverse=0;return
  endif
  residual=matmul(matrix,inverse)
  do i=1,5
   residual(i,i)=residual(i,i)-1
  enddo
  if(.not.all(ieee_is_finite(residual)).or.maxval(abs(residual))>1.d-8) then
   inverse=0;return
  endif
  valid=.true.
 end subroutine

 subroutine diffusion_quadratic_coefficients(c,conductance,inverse,velocity,polynomial)
  integer,intent(in) :: c
  real(8),intent(in) :: conductance(nf),inverse(5,5),velocity(2,nc)
  real(8),intent(out) :: polynomial(2,5)
  real(8) :: rhs(5,2),row(5),z(2),h,w,difference(2)
  integer :: stencil(max_diffusion_recovery_stencil),nstencil,index,r
  logical :: complete
  polynomial=0;rhs=0;h=sqrt(area(c))
  call diffusion_recovery_stencil(c,conductance,stencil,nstencil,complete)
  if(.not.complete) return
  do index=1,nstencil
   r=stencil(index);z=(xy(:,r)-xy(:,c))/h;w=1.d0/sum(z*z)
   row=[z(1),z(2),z(1)*z(1)/diffusion_sqrt_two,z(1)*z(2), &
    z(2)*z(2)/diffusion_sqrt_two]
   difference=velocity(:,r)-velocity(:,c)
   rhs(:,1)=rhs(:,1)+w*row*difference(1)
   rhs(:,2)=rhs(:,2)+w*row*difference(2)
  enddo
  polynomial(1,:)=matmul(inverse,rhs(:,1))
  polynomial(2,:)=matmul(inverse,rhs(:,2))
 end subroutine

 subroutine diffusion_quadratic_face_flux(f,left_polynomial,right_polynomial,diffusivity,value)
  integer,intent(in) :: f
  real(8),intent(in) :: left_polynomial(2,5),right_polynomial(2,5),diffusivity
  real(8),intent(out) :: value(2)
  real(8) :: left(2),right(2)
  call diffusion_quadratic_trace(owner(f),f,left_polynomial,left)
  call diffusion_quadratic_trace(neighbour(f),f,right_polynomial,right)
  value=.5d0*diffusivity*(left+right)
 end subroutine

 subroutine diffusion_quadratic_trace(c,f,polynomial,value)
  integer,intent(in) :: c,f
  real(8),intent(in) :: polynomial(2,5)
  real(8),intent(out) :: value(2)
  real(8) :: row(5),z(2),h
  integer :: component
  h=sqrt(area(c));z=(mid(:,f)-xy(:,c))/h
  row=[normal(1,f),normal(2,f),diffusion_sqrt_two*normal(1,f)*z(1), &
   normal(1,f)*z(2)+normal(2,f)*z(1),diffusion_sqrt_two*normal(2,f)*z(2)]/h
  do component=1,2
   value(component)=dot_product(row,polynomial(component,:))
  enddo
 end subroutine

 function diffusion_quadratic_l1(c,f,conductance,inverse) result(value)
  integer,intent(in) :: c,f
  real(8),intent(in) :: conductance(nf),inverse(5,5)
  real(8) :: value,row(5),evaluation(5),z(2),h,w,coefficient,total
  integer :: stencil(max_diffusion_recovery_stencil),nstencil,index,r
  logical :: complete
  value=0;total=0;h=sqrt(area(c));z=(mid(:,f)-xy(:,c))/h
  evaluation=[normal(1,f),normal(2,f),diffusion_sqrt_two*normal(1,f)*z(1), &
   normal(1,f)*z(2)+normal(2,f)*z(1),diffusion_sqrt_two*normal(2,f)*z(2)]/h
  call diffusion_recovery_stencil(c,conductance,stencil,nstencil,complete)
  if(.not.complete) then
   value=huge(1.d0);return
  endif
  do index=1,nstencil
   r=stencil(index);z=(xy(:,r)-xy(:,c))/h;w=1.d0/sum(z*z)
   row=[z(1),z(2),z(1)*z(1)/diffusion_sqrt_two,z(1)*z(2), &
    z(2)*z(2)/diffusion_sqrt_two]
   coefficient=w*dot_product(evaluation,matmul(inverse,row))
   value=value+abs(coefficient);total=total+coefficient
  enddo
  value=value+abs(total)
 end function

 subroutine diffusion_affine_matrix(c,conductance,inverse,valid)
  integer,intent(in) :: c
  real(8),intent(in) :: conductance(nf)
  real(8),intent(out) :: inverse(2,2)
  logical,intent(out) :: valid
  real(8) :: matrix(2,2),row(2),z(2),h,w
  integer :: stencil(max_diffusion_recovery_stencil),nstencil,index,k,f,r,i,j
  logical :: complete
  inverse=0;valid=.false.
  if(area(c)<=0) return
  h=sqrt(area(c))
  call diffusion_recovery_stencil(c,conductance,stencil,nstencil,complete)
  ! Boundary equations constrain a fit; they must not create one from a
  ! single wet neighbour. Such minimal patches keep the established TPFA.
  if(.not.complete.or.nstencil<2) return
  matrix=0
  do index=1,nstencil
   r=stencil(index);z=(xy(:,r)-xy(:,c))/h
   w=1.d0/sum(z*z);row=z
   do i=1,2
    do j=1,2
     matrix(i,j)=matrix(i,j)+w*row(i)*row(j)
    enddo
   enddo
  enddo
  do k=offset(c),offset(c+1)-1
   f=faces(k);r=neighbour(f)
   if(signs(k)<0) r=owner(f)
   if(r>0.and.conductance(f)>0) cycle
   w=length(f)/h;row=normal(:,f)
   do i=1,2
    do j=1,2
     matrix(i,j)=matrix(i,j)+w*row(i)*row(j)
    enddo
   enddo
  enddo
  call diffusion_inverse2(matrix,inverse,valid)
 end subroutine

 subroutine diffusion_inverse2(matrix,inverse,valid)
  real(8),intent(in) :: matrix(2,2)
  real(8),intent(out) :: inverse(2,2)
  logical,intent(out) :: valid
  real(8) :: augmented(2,4),temporary(4),diagonal(2),residual(2,2),pivot_value,factor
  integer :: i,j,pivot
  inverse=0;valid=.false.
  do i=1,2
   if(.not.ieee_is_finite(matrix(i,i)).or.matrix(i,i)<=0) return
   diagonal(i)=sqrt(matrix(i,i))
  enddo
  augmented=0
  do i=1,2
   do j=1,2
    augmented(i,j)=matrix(i,j)/(diagonal(i)*diagonal(j))
   enddo
  enddo
  if(.not.all(ieee_is_finite(augmented(:,1:2)))) return
  do i=1,2
   augmented(i,2+i)=1
  enddo
  do i=1,2
   pivot=i-1+maxloc(abs(augmented(i:2,i)),dim=1)
   pivot_value=abs(augmented(pivot,i))
   if(.not.ieee_is_finite(pivot_value).or.pivot_value<=1.d-12) return
   if(pivot/=i) then
    temporary=augmented(i,:);augmented(i,:)=augmented(pivot,:);augmented(pivot,:)=temporary
   endif
   augmented(i,:)=augmented(i,:)/augmented(i,i)
   do j=1,2
    if(j==i) cycle
    factor=augmented(j,i);augmented(j,:)=augmented(j,:)-factor*augmented(i,:)
   enddo
  enddo
  do i=1,2
   do j=1,2
    inverse(i,j)=augmented(i,2+j)/(diagonal(i)*diagonal(j))
   enddo
  enddo
  inverse=.5d0*(inverse+transpose(inverse))
  if(.not.all(ieee_is_finite(inverse))) then
   inverse=0;return
  endif
  residual=matmul(matrix,inverse)
  do i=1,2
   residual(i,i)=residual(i,i)-1
  enddo
  if(.not.all(ieee_is_finite(residual)).or.maxval(abs(residual))>1.d-8) then
   inverse=0;return
  endif
  valid=.true.
 end subroutine

 subroutine diffusion_affine_coefficients(c,conductance,inverse,velocity,polynomial)
  integer,intent(in) :: c
  real(8),intent(in) :: conductance(nf),inverse(2,2),velocity(2,nc)
  real(8),intent(out) :: polynomial(2,2)
  real(8) :: rhs(2,2),z(2),h,w,difference(2)
  integer :: stencil(max_diffusion_recovery_stencil),nstencil,index,r
  logical :: complete
  polynomial=0;rhs=0;h=sqrt(area(c))
  call diffusion_recovery_stencil(c,conductance,stencil,nstencil,complete)
  if(.not.complete) return
  do index=1,nstencil
   r=stencil(index);z=(xy(:,r)-xy(:,c))/h;w=1.d0/sum(z*z)
   difference=velocity(:,r)-velocity(:,c)
   rhs(:,1)=rhs(:,1)+w*z*difference(1)
   rhs(:,2)=rhs(:,2)+w*z*difference(2)
  enddo
  polynomial(1,:)=matmul(inverse,rhs(:,1))
  polynomial(2,:)=matmul(inverse,rhs(:,2))
 end subroutine

 subroutine diffusion_affine_trace(c,f,polynomial,value)
  integer,intent(in) :: c,f
  real(8),intent(in) :: polynomial(2,2)
  real(8),intent(out) :: value(2)
  real(8) :: evaluation(2),h
  integer :: component
  h=sqrt(area(c));evaluation=normal(:,f)/h
  do component=1,2
   value(component)=dot_product(evaluation,polynomial(component,:))
  enddo
 end subroutine

 function diffusion_affine_l1(c,f,conductance,inverse) result(value)
  integer,intent(in) :: c,f
  real(8),intent(in) :: conductance(nf),inverse(2,2)
  real(8) :: value,evaluation(2),z(2),h,w,coefficient,total
  integer :: stencil(max_diffusion_recovery_stencil),nstencil,index,r
  logical :: complete
  value=0;total=0;h=sqrt(area(c));evaluation=normal(:,f)/h
  call diffusion_recovery_stencil(c,conductance,stencil,nstencil,complete)
  if(.not.complete) then
   value=huge(1.d0);return
  endif
  do index=1,nstencil
   r=stencil(index);z=(xy(:,r)-xy(:,c))/h;w=1.d0/sum(z*z)
   coefficient=w*dot_product(evaluation,matmul(inverse,z))
   value=value+abs(coefficient);total=total+coefficient
  enddo
  value=value+abs(total)
 end function

 function diffusion_velocity_energy(momentum,depth) result(energy)
  real(8),intent(in) :: momentum(2,nc),depth(nc)
  real(8) :: energy,velocity(2)
  integer :: c
  energy=0
  do c=1,nc
   if(depth(c)<=dry_tolerance) cycle
   velocity=momentum(:,c)/depth(c)
   energy=energy+.5d0*area(c)*sum(velocity*velocity)
  enddo
 end function

subroutine advance_prognostic_secondary_ssp33(dt,ok)
  real(8),intent(in) :: dt
  logical,intent(out) :: ok
  real(8) :: a(nstate,nc),b(nstate,nc),c(nstate,nc)
  real(8) :: fa(nledger,nb),fb(nledger,nb),fc(nledger,nb),profile_start,step_morfac
  real(8) :: source_first,source_second
  real(8) :: momentum_a(2,2),momentum_b(2,2),momentum_c(2,2)
  integer :: target
  logical :: source_ok
  trial_cfl_peak=0.d0;source_first=0.d0;source_second=0.d0
  secondary_momentum_boundary_trial=0.d0
  if(bank_toe_erosion_mode==2.and..not.bank_toe_mapping_ready) call build_physical_toe_map(q)
  if(bank_toe_erosion_mode==2) physical_toe_flux_trial=0.d0
  if(bank_toe_erosion_mode==3) distributed_face_mass_trial=0.d0
  if(profile_enabled) profile_start=omp_get_wtime()
  call prepare_hydraulic_stage(q)
  if(profile_enabled) then
   profile_prepare_s=profile_prepare_s+omp_get_wtime()-profile_start;profile_prepare_calls=profile_prepare_calls+1
  endif
  a=q
  call secondary_relax_state(a,.5d0*dt,-1,source_first,source_ok)
  if(.not.source_ok) then
   ok=.false.;failure_reason=6;return
  endif
  if(profile_enabled) profile_start=omp_get_wtime()
  step_morfac=morphological_factor_at(time)
  call euler(a,dt,time,step_morfac,b,fa,ok,momentum_boundary_impulse=momentum_a,toe_flux_weight=1.d0/6.d0)
  if(profile_enabled) then
   profile_euler_s=profile_euler_s+omp_get_wtime()-profile_start;profile_euler_calls=profile_euler_calls+1
  endif
  if(.not.ok) return
  if(profile_enabled) profile_start=omp_get_wtime()
  call prepare_hydraulic_stage(b)
  if(profile_enabled) then
   profile_prepare_s=profile_prepare_s+omp_get_wtime()-profile_start;profile_prepare_calls=profile_prepare_calls+1
   profile_start=omp_get_wtime()
  endif
  call euler(b,dt,time+dt,step_morfac,c,fb,ok,momentum_boundary_impulse=momentum_b,toe_flux_weight=1.d0/6.d0)
  if(profile_enabled) then
   profile_euler_s=profile_euler_s+omp_get_wtime()-profile_start;profile_euler_calls=profile_euler_calls+1
  endif
  if(.not.ok) return
  !$omp parallel do schedule(static,256) if(nc*nstate>=131072) default(shared) private(target)
  do target=1,nc
   b(:,target)=.75d0*a(:,target)+.25d0*c(:,target)
  enddo
  !$omp end parallel do
  call rebalance_active_layer(b)
  if(profile_enabled) profile_start=omp_get_wtime()
  call prepare_hydraulic_stage(b)
  if(profile_enabled) then
   profile_prepare_s=profile_prepare_s+omp_get_wtime()-profile_start;profile_prepare_calls=profile_prepare_calls+1
   profile_start=omp_get_wtime()
  endif
  call euler(b,dt,time+.5d0*dt,step_morfac,c,fc,ok,momentum_boundary_impulse=momentum_c,toe_flux_weight=2.d0/3.d0)
  if(profile_enabled) then
   profile_euler_s=profile_euler_s+omp_get_wtime()-profile_start;profile_euler_calls=profile_euler_calls+1
  endif
  if(.not.ok) return
  !$omp parallel do schedule(static,256) if(nc*nstate>=131072) default(shared) private(target)
  do target=1,nc
   c(:,target)=a(:,target)/3.d0+2.d0*c(:,target)/3.d0
  enddo
  !$omp end parallel do
  call rebalance_active_layer(c)
  if(profile_enabled) profile_start=omp_get_wtime()
  call prepare_hydraulic_stage(c)
  if(profile_enabled) then
   profile_prepare_s=profile_prepare_s+omp_get_wtime()-profile_start;profile_prepare_calls=profile_prepare_calls+1
   profile_start=omp_get_wtime()
  endif
  call source(c,dt,q(1,:))
  if(profile_enabled) then
   profile_source_s=profile_source_s+omp_get_wtime()-profile_start;profile_source_calls=profile_source_calls+1
  endif
  call secondary_relax_state(c,.5d0*dt,-1,source_second,source_ok)
  if(.not.source_ok) then
   ok=.false.;failure_reason=6;return
  endif
  if(.not.all(ieee_is_finite(c)).or.minval(c(1,:))<0.or.minval(c(4:nphysical_state,:))<0) then
   ok=.false.;return
  endif
  if(bank_failure) then
   if(bank_front_tracking) then
    ! Persistent geometry is committed last so a rejected Eulerian
    ! relaxation cannot leak front history into the retry.
    call avalanche_erodible_banks(c,dt)
    if(.not.all(ieee_is_finite(c)).or.minval(c(1,:))<0.or. &
       minval(c(4:nphysical_state,:))<0) then
     ok=.false.;return
    endif
    call apply_bank_failure(q,c,dt,ok)
   else
    call apply_bank_failure(q,c,dt,ok)
    if(ok) call avalanche_erodible_banks(c,dt)
   endif
   if(.not.ok) return
  endif
  if(.not.all(ieee_is_finite(c)).or.minval(c(1,:))<0.or.minval(c(4:nphysical_state,:))<0) then
   if(bank_front_tracking) error stop 'invalid fixed state after persistent bank-front commit'
   ok=.false.;return
  endif
  !$omp parallel do schedule(static,256) if(nc*nstate>=131072) default(shared) private(target)
  do target=1,nc
   q(:,target)=c(:,target)
  enddo
  !$omp end parallel do
  ledger=ledger+(fa+fb)/6.d0+2.d0*fc/3.d0
  if(bank_toe_erosion_mode==2) call commit_physical_toe_flux()
  secondary_source_cumulative=secondary_source_cumulative+source_first+source_second
  secondary_momentum_boundary_trial=(momentum_a+momentum_b)/6.d0+2.d0*momentum_c/3.d0
  time=time+dt;steps=steps+1
 end subroutine

subroutine advance_ssp33(dt,ok)
  real(8), intent(in) :: dt
  logical, intent(out) :: ok
  real(8) :: a(nstate,nc),b(nstate,nc),c(nstate,nc)
  real(8) :: fa(nledger,nb),fb(nledger,nb),fc(nledger,nb),profile_start,step_morfac
  real(8),allocatable :: transport_a(:,:),transport_b(:,:),transport_c(:,:)
  integer :: target
  if(prognostic_secondary_flow()) then
   call advance_prognostic_secondary_ssp33(dt,ok)
   return
  endif
  trial_cfl_peak=0
  if(exchange_mode==4.and.flow2dh_amr_ownership_declaration) &
   suspended_exchange_step_trial=0.d0
  if(bank_toe_erosion_mode==2.and..not.bank_toe_mapping_ready) call build_physical_toe_map(q)
  if(bank_toe_erosion_mode==2) physical_toe_flux_trial=0.d0
  if(bank_toe_erosion_mode==3) distributed_face_mass_trial=0.d0
  if(flow2dh_boundary_diffusion_declaration) &
   allocate(transport_a(1+nfrac,nf),transport_b(1+nfrac,nf), &
    transport_c(1+nfrac,nf))
  if(profile_enabled) profile_start=omp_get_wtime()
  call prepare_hydraulic_stage(q)
  if(profile_enabled) then
   profile_prepare_s=profile_prepare_s+omp_get_wtime()-profile_start;profile_prepare_calls=profile_prepare_calls+1
   profile_start=omp_get_wtime()
  endif
  step_morfac=morphological_factor_at(time)
  if(flow2dh_boundary_diffusion_declaration) then
   call euler(q,dt,time,step_morfac,a,fa,ok,toe_flux_weight=1.d0/6.d0, &
    boundary_face_transport_integral=transport_a)
  else
   call euler(q,dt,time,step_morfac,a,fa,ok,toe_flux_weight=1.d0/6.d0)
  endif
  if(profile_enabled) then
   profile_euler_s=profile_euler_s+omp_get_wtime()-profile_start;profile_euler_calls=profile_euler_calls+1
  endif
  if(.not.ok) return
  if(profile_enabled) profile_start=omp_get_wtime()
  call prepare_hydraulic_stage(a)
  if(profile_enabled) then
   profile_prepare_s=profile_prepare_s+omp_get_wtime()-profile_start;profile_prepare_calls=profile_prepare_calls+1
   profile_start=omp_get_wtime()
  endif
  if(flow2dh_boundary_diffusion_declaration) then
   call euler(a,dt,time+dt,step_morfac,b,fb,ok,toe_flux_weight=1.d0/6.d0, &
    boundary_face_transport_integral=transport_b)
  else
   call euler(a,dt,time+dt,step_morfac,b,fb,ok,toe_flux_weight=1.d0/6.d0)
  endif
  if(profile_enabled) then
   profile_euler_s=profile_euler_s+omp_get_wtime()-profile_start;profile_euler_calls=profile_euler_calls+1
  endif
  if(.not.ok) return
  !$omp parallel do schedule(static,256) if(nc*nstate>=131072) default(shared) private(target)
  do target=1,nc
   b(:,target)=.75d0*q(:,target)+.25d0*b(:,target)
  enddo
  !$omp end parallel do
  call rebalance_active_layer(b)
  if(profile_enabled) profile_start=omp_get_wtime()
  call prepare_hydraulic_stage(b)
  if(profile_enabled) then
   profile_prepare_s=profile_prepare_s+omp_get_wtime()-profile_start;profile_prepare_calls=profile_prepare_calls+1
   profile_start=omp_get_wtime()
  endif
  if(flow2dh_boundary_diffusion_declaration) then
   call euler(b,dt,time+.5d0*dt,step_morfac,c,fc,ok,toe_flux_weight=2.d0/3.d0, &
    boundary_face_transport_integral=transport_c)
  else
   call euler(b,dt,time+.5d0*dt,step_morfac,c,fc,ok,toe_flux_weight=2.d0/3.d0)
  endif
  if(profile_enabled) then
   profile_euler_s=profile_euler_s+omp_get_wtime()-profile_start;profile_euler_calls=profile_euler_calls+1
  endif
  if(.not.ok) return
  !$omp parallel do schedule(static,256) if(nc*nstate>=131072) default(shared) private(target)
  do target=1,nc
   c(:,target)=q(:,target)/3.d0+2.d0*c(:,target)/3.d0
  enddo
  !$omp end parallel do
  call rebalance_active_layer(c)
  if(profile_enabled) profile_start=omp_get_wtime()
  call prepare_hydraulic_stage(c)
  if(profile_enabled) then
   profile_prepare_s=profile_prepare_s+omp_get_wtime()-profile_start;profile_prepare_calls=profile_prepare_calls+1
   profile_start=omp_get_wtime()
  endif
  call source(c,dt,q(1,:))
  if(exchange_mode==4.and.flow2dh_amr_ownership_declaration) &
   call accumulate_suspended_exchange_trial(-1)
  if(profile_enabled) then
   profile_source_s=profile_source_s+omp_get_wtime()-profile_start;profile_source_calls=profile_source_calls+1
  endif
  if(.not.all(ieee_is_finite(c)).or.minval(c(1,:))<0.or.minval(c(4:nphysical_state,:))<0) then
   ok=.false.;return
  endif
  if(bank_failure) then
   if(bank_front_tracking) then
    ! Persistent geometry is committed last so a rejected Eulerian
    ! relaxation cannot leak front history into the retry.
    call avalanche_erodible_banks(c,dt)
    if(.not.all(ieee_is_finite(c)).or.minval(c(1,:))<0.or. &
       minval(c(4:nphysical_state,:))<0) then
     ok=.false.;return
    endif
    call apply_bank_failure(q,c,dt,ok)
   else
    call apply_bank_failure(q,c,dt,ok)
    if(ok) call avalanche_erodible_banks(c,dt)
   endif
   if(.not.ok) return
  endif
  if(.not.all(ieee_is_finite(c)).or.minval(c(1,:))<0.or.minval(c(4:nphysical_state,:))<0) then
   if(bank_front_tracking) error stop 'invalid fixed state after persistent bank-front commit'
   ok=.false.;return
  endif
  if(flow2dh_boundary_diffusion_declaration) then
   transport_a=(transport_a+transport_b)/6.d0+2.d0*transport_c/3.d0
   call commit_flow2dh_boundary_memory(dt,transport_a,q)
  endif
  !$omp parallel do schedule(static,256) if(nc*nstate>=131072) default(shared) private(target)
  do target=1,nc
   q(:,target)=c(:,target)
  enddo
  !$omp end parallel do
  ledger=ledger+(fa+fb)/6.d0+2.d0*fc/3.d0
  if(bank_toe_erosion_mode==2) call commit_physical_toe_flux()
  time=time+dt;steps=steps+1
 end subroutine

subroutine advance_ssp43(dt,ok)
  real(8), intent(in) :: dt
  logical, intent(out) :: ok
  real(8) :: a(nstate,nc),b(nstate,nc),c(nstate,nc)
  real(8) :: fa(nledger,nb),fb(nledger,nb),fc(nledger,nb),profile_start,step_morfac
  real(8) :: cpu_momentum(2,2)
  integer :: target
  if(prognostic_secondary_flow()) then
   call advance_prognostic_secondary_ssp43(dt,ok)
   return
  endif
  trial_cfl_peak=0
  if(bank_toe_erosion_mode==2.and..not.bank_toe_mapping_ready) call build_physical_toe_map(q)
  if(bank_toe_erosion_mode==2) physical_toe_flux_trial=0.d0
  if(bank_toe_erosion_mode==3) distributed_face_mass_trial=0.d0
  if(profile_enabled) profile_start=omp_get_wtime()
  call prepare_hydraulic_stage(q)
  if(profile_enabled) then
   profile_prepare_s=profile_prepare_s+omp_get_wtime()-profile_start;profile_prepare_calls=profile_prepare_calls+1
   profile_start=omp_get_wtime()
  endif
  step_morfac=morphological_factor_at(time)
  call cpu_ssp43(q,dt,step_morfac,c,fa,cpu_momentum,ok)
  if(.not.ok) return
  call rebalance_active_layer(c)
  if(profile_enabled) profile_start=omp_get_wtime()
  call prepare_hydraulic_stage(c)
  if(profile_enabled) then
   profile_prepare_s=profile_prepare_s+omp_get_wtime()-profile_start;profile_prepare_calls=profile_prepare_calls+1
   profile_start=omp_get_wtime()
  endif
  call source(c,dt,q(1,:))
  if(profile_enabled) then
   profile_source_s=profile_source_s+omp_get_wtime()-profile_start;profile_source_calls=profile_source_calls+1
  endif
  if(.not.all(ieee_is_finite(c)).or.minval(c(1,:))<0.or.minval(c(4:nphysical_state,:))<0) then
   ok=.false.;return
  endif
  if(bank_failure) then
   if(bank_front_tracking) then
    ! Persistent geometry is committed last so a rejected Eulerian
    ! relaxation cannot leak front history into the retry.
    call avalanche_erodible_banks(c,dt)
    if(.not.all(ieee_is_finite(c)).or.minval(c(1,:))<0.or. &
       minval(c(4:nphysical_state,:))<0) then
     ok=.false.;return
    endif
    call apply_bank_failure(q,c,dt,ok)
   else
    call apply_bank_failure(q,c,dt,ok)
    if(ok) call avalanche_erodible_banks(c,dt)
   endif
   if(.not.ok) return
  endif
  if(.not.all(ieee_is_finite(c)).or.minval(c(1,:))<0.or.minval(c(4:nphysical_state,:))<0) then
   if(bank_front_tracking) error stop 'invalid fixed state after persistent bank-front commit'
   ok=.false.;return
  endif
  !$omp parallel do schedule(static,256) if(nc*nstate>=131072) default(shared) private(target)
  do target=1,nc
   q(:,target)=c(:,target)
  enddo
  !$omp end parallel do
  ledger=ledger+fa
  if(bank_toe_erosion_mode==2) call commit_physical_toe_flux()
  time=time+dt;steps=steps+1
 end subroutine

subroutine advance_prognostic_secondary_ssp43(dt,ok)
  real(8),intent(in) :: dt
  logical,intent(out) :: ok
  real(8) :: a(nstate,nc),b(nstate,nc),c(nstate,nc)
  real(8) :: fa(nledger,nb),fb(nledger,nb),fc(nledger,nb),profile_start,step_morfac
  real(8) :: source_first,source_second
  real(8) :: momentum_a(2,2),momentum_b(2,2),momentum_c(2,2)
  integer :: target
  logical :: source_ok
  trial_cfl_peak=0.d0;source_first=0.d0;source_second=0.d0
  secondary_momentum_boundary_trial=0.d0
  if(bank_toe_erosion_mode==2.and..not.bank_toe_mapping_ready) call build_physical_toe_map(q)
  if(bank_toe_erosion_mode==2) physical_toe_flux_trial=0.d0
  if(bank_toe_erosion_mode==3) distributed_face_mass_trial=0.d0
  if(profile_enabled) profile_start=omp_get_wtime()
  call prepare_hydraulic_stage(q)
  if(profile_enabled) then
   profile_prepare_s=profile_prepare_s+omp_get_wtime()-profile_start;profile_prepare_calls=profile_prepare_calls+1
  endif
  a=q
  call secondary_relax_state(a,.5d0*dt,-1,source_first,source_ok)
  if(.not.source_ok) then
   ok=.false.;failure_reason=6;return
  endif
  if(profile_enabled) profile_start=omp_get_wtime()
  step_morfac=morphological_factor_at(time)
  call cpu_ssp43(a,dt,step_morfac,c,fa,momentum_a,ok)
  if(.not.ok) return
  call rebalance_active_layer(c)
  if(profile_enabled) profile_start=omp_get_wtime()
  call prepare_hydraulic_stage(c)
  if(profile_enabled) then
   profile_prepare_s=profile_prepare_s+omp_get_wtime()-profile_start;profile_prepare_calls=profile_prepare_calls+1
   profile_start=omp_get_wtime()
  endif
  call source(c,dt,q(1,:))
  if(profile_enabled) then
   profile_source_s=profile_source_s+omp_get_wtime()-profile_start;profile_source_calls=profile_source_calls+1
  endif
  call secondary_relax_state(c,.5d0*dt,-1,source_second,source_ok)
  if(.not.source_ok) then
   ok=.false.;failure_reason=6;return
  endif
  if(.not.all(ieee_is_finite(c)).or.minval(c(1,:))<0.or.minval(c(4:nphysical_state,:))<0) then
   ok=.false.;return
  endif
  if(bank_failure) then
   if(bank_front_tracking) then
    ! Persistent geometry is committed last so a rejected Eulerian
    ! relaxation cannot leak front history into the retry.
    call avalanche_erodible_banks(c,dt)
    if(.not.all(ieee_is_finite(c)).or.minval(c(1,:))<0.or. &
       minval(c(4:nphysical_state,:))<0) then
     ok=.false.;return
    endif
    call apply_bank_failure(q,c,dt,ok)
   else
    call apply_bank_failure(q,c,dt,ok)
    if(ok) call avalanche_erodible_banks(c,dt)
   endif
   if(.not.ok) return
  endif
  if(.not.all(ieee_is_finite(c)).or.minval(c(1,:))<0.or.minval(c(4:nphysical_state,:))<0) then
   if(bank_front_tracking) error stop 'invalid fixed state after persistent bank-front commit'
   ok=.false.;return
  endif
  !$omp parallel do schedule(static,256) if(nc*nstate>=131072) default(shared) private(target)
  do target=1,nc
   q(:,target)=c(:,target)
  enddo
  !$omp end parallel do
  ledger=ledger+fa
  if(bank_toe_erosion_mode==2) call commit_physical_toe_flux()
  secondary_source_cumulative=secondary_source_cumulative+source_first+source_second
  secondary_momentum_boundary_trial=momentum_a
  time=time+dt;steps=steps+1
 end subroutine

subroutine advance_multistep6(dt,ok)
  real(8), intent(in) :: dt
  logical, intent(out) :: ok
  real(8) :: a(nstate,nc),b(nstate,nc),c(nstate,nc)
  real(8) :: fa(nledger,nb),fb(nledger,nb),fc(nledger,nb),profile_start,step_morfac
  real(8) :: cpu_momentum(2,2)
  integer :: target
  logical :: used_history
  if(prognostic_secondary_flow()) then
   call advance_prognostic_secondary_ssp43(dt,ok)
   return
  endif
  trial_cfl_peak=0
  if(bank_toe_erosion_mode==2.and..not.bank_toe_mapping_ready) call build_physical_toe_map(q)
  if(bank_toe_erosion_mode==2) physical_toe_flux_trial=0.d0
  if(bank_toe_erosion_mode==3) distributed_face_mass_trial=0.d0
  if(profile_enabled) profile_start=omp_get_wtime()
  !$omp parallel do schedule(static,256) default(shared) private(target)
  do target=1,nc
   a(:,target)=q(:,target)
  enddo
  !$omp end parallel do
  call prepare_hydraulic_stage(a)
  if(profile_enabled) then
   profile_prepare_s=profile_prepare_s+omp_get_wtime()-profile_start;profile_prepare_calls=profile_prepare_calls+1
   profile_start=omp_get_wtime()
  endif
  step_morfac=morphological_factor_at(time)
  call cpu_history_step(a,dt,step_morfac,c,fa,cpu_momentum,used_history,ok)
  if(.not.ok) return
  call rebalance_active_layer(c)
  if(profile_enabled) profile_start=omp_get_wtime()
  call prepare_hydraulic_stage(c)
  if(profile_enabled) then
   profile_prepare_s=profile_prepare_s+omp_get_wtime()-profile_start;profile_prepare_calls=profile_prepare_calls+1
   profile_start=omp_get_wtime()
  endif
  if(.not.used_history) call source(c,dt,a(1,:))
  if(profile_enabled.and..not.used_history) then
   profile_source_s=profile_source_s+omp_get_wtime()-profile_start;profile_source_calls=profile_source_calls+1
  endif
  if(.not.all(ieee_is_finite(c)).or.minval(c(1,:))<0.or.minval(c(4:nphysical_state,:))<0) then
   ok=.false.;return
  endif
  if(bank_failure) then
   if(bank_front_tracking) then
    ! Persistent geometry is committed last so a rejected Eulerian
    ! relaxation cannot leak front history into the retry.
    call avalanche_erodible_banks(c,dt)
    if(.not.all(ieee_is_finite(c)).or.minval(c(1,:))<0.or. &
       minval(c(4:nphysical_state,:))<0) then
     ok=.false.;return
    endif
    call apply_bank_failure(q,c,dt,ok)
   else
    call apply_bank_failure(q,c,dt,ok)
    if(ok) call avalanche_erodible_banks(c,dt)
   endif
   if(.not.ok) return
  endif
  if(.not.all(ieee_is_finite(c)).or.minval(c(1,:))<0.or.minval(c(4:nphysical_state,:))<0) then
   if(bank_front_tracking) error stop 'invalid fixed state after persistent bank-front commit'
   ok=.false.;return
  endif
  call explicit_history_commit(a,ledger,time,dt,used_history)
  !$omp parallel do schedule(static,256) default(shared) private(target)
  do target=1,nc
   q(:,target)=c(:,target)
  enddo
  !$omp end parallel do
  ledger=ledger+fa
  if(bank_toe_erosion_mode==2) call commit_physical_toe_flux()
  time=time+dt;steps=steps+1
 end subroutine
! Qualified fixed-mesh SSPRK(4,3) transport update.
! Each Euler call advances dt/2 and keeps the original stage CFL rejection.
! Butcher weights (1/6,1/6,1/6,1/2), stage times (0,1/2,1,1/2).
! Euler returns time-integrated fluxes, hence ledger weights (1/3,1/3,1/3,1).
 subroutine cpu_ssp43(initial,dt,step_morfac,result,accepted,momentum,ok)
  real(8),intent(in) :: initial(nstate,nc),dt,step_morfac
  real(8),intent(out) :: result(nstate,nc),accepted(nledger,nb),momentum(2,2)
  logical,intent(out) :: ok
  real(8) :: first(nstate,nc),second(nstate,nc)
  real(8) :: flux(nledger,nb),impulse(2,2),started
  integer :: cell
  accepted=0;momentum=0;ok=.false.
  if(profile_enabled) started=omp_get_wtime()
  call euler(initial,.5d0*dt,time,step_morfac,first,flux,ok, &
   momentum_boundary_impulse=impulse,toe_flux_weight=1.d0/3.d0)
  if(profile_enabled) then
   profile_euler_s=profile_euler_s+omp_get_wtime()-started;profile_euler_calls=profile_euler_calls+1
  endif
  if(.not.ok) return
  accepted=flux/3.d0;momentum=impulse/3.d0
  call cpu_prepare_stage(first)
  if(profile_enabled) started=omp_get_wtime()
  call euler(first,.5d0*dt,time+.5d0*dt,step_morfac,second,flux,ok, &
   momentum_boundary_impulse=impulse,toe_flux_weight=1.d0/3.d0)
  if(profile_enabled) then
   profile_euler_s=profile_euler_s+omp_get_wtime()-started;profile_euler_calls=profile_euler_calls+1
  endif
  if(.not.ok) return
  accepted=accepted+flux/3.d0;momentum=momentum+impulse/3.d0
  call cpu_prepare_stage(second)
  if(profile_enabled) started=omp_get_wtime()
  call euler(second,.5d0*dt,time+dt,step_morfac,first,flux,ok, &
   momentum_boundary_impulse=impulse,toe_flux_weight=1.d0/3.d0)
  if(profile_enabled) then
   profile_euler_s=profile_euler_s+omp_get_wtime()-started;profile_euler_calls=profile_euler_calls+1
  endif
  if(.not.ok) return
  accepted=accepted+flux/3.d0;momentum=momentum+impulse/3.d0
  ! Independent cells: retain an outer OpenMP loop, not nested scalar teams.
  !$omp parallel do schedule(static,256) default(shared) private(cell)
  do cell=1,nc
   second(:,cell)=2.d0*initial(:,cell)/3.d0+first(:,cell)/3.d0
  enddo
  !$omp end parallel do
  call rebalance_active_layer(second)
  call cpu_prepare_stage(second)
  if(profile_enabled) started=omp_get_wtime()
  call euler(second,.5d0*dt,time+.5d0*dt,step_morfac,result,flux,ok, &
   momentum_boundary_impulse=impulse,toe_flux_weight=1.d0)
  if(profile_enabled) then
   profile_euler_s=profile_euler_s+omp_get_wtime()-started;profile_euler_calls=profile_euler_calls+1
  endif
  if(.not.ok) return
  accepted=accepted+flux;momentum=momentum+impulse
 end subroutine

 subroutine cpu_prepare_stage(state)
  real(8),intent(inout) :: state(nstate,nc)
  real(8) :: started
  if(profile_enabled) started=omp_get_wtime()
  call prepare_hydraulic_stage(state)
  if(profile_enabled) then
   profile_prepare_s=profile_prepare_s+omp_get_wtime()-started;profile_prepare_calls=profile_prepare_calls+1
  endif
 end subroutine

! Qualified hydraulic-only variable-step SSP multistep method, order two.
! Hadjimichael, Ketcheson, Loczi and Nemeth (2016), equation (3.2):
!   D=t_n-t_(n-k+1), r=dt/D, tau=dt/(1-r),
!   q_(n+1)=(1-r*r)*(q_n+tau*F(q_n,t_n))+r*r*q_(n-k+1).
! The finite-volume Euler leg, its augmented f-wave flux and its draining/CFL
! safeguards are unchanged. The Manning RHS belongs INSIDE that leg.
! State and cumulative boundary ledger use exactly the same recurrence.



 subroutine explicit_history_allocate()
  if(allocated(explicit_history_state)) return
  allocate(explicit_history_state(nstate,nc,explicit_history_method-1), &
   explicit_history_ledger(nledger,nb,explicit_history_method-1), &
   explicit_history_time(explicit_history_method-1),explicit_friction_rate(nc))
  ! Slots are written only after successful steps and read only when full.
  ! Avoid an otherwise unnecessary serial initialization of large arrays.
  explicit_history_count=0;explicit_history_position=1
 end subroutine

 subroutine cpu_history_step(initial,dt,step_morfac,result,accepted,momentum,used_history,ok)
  real(8),intent(in) :: initial(nstate,nc),dt,step_morfac
  real(8),intent(out) :: result(nstate,nc),accepted(nledger,nb),momentum(2,2)
  logical,intent(out) :: used_history,ok
  real(8) :: span,ratio,weight,tau,started,source_started,depth,speed,denominator,friction_peak,knot,tolerance
  integer :: cell,slot,newest
  logical :: invalid_friction,rough_valid
  used_history=.false.;ok=.false.;accepted=0;momentum=0;explicit_trial_reset_kind=0
  call explicit_history_allocate()
  if(explicit_history_count>0) then
   newest=mod(explicit_history_position-2+explicit_history_method-1,explicit_history_method-1)+1
   knot=next_boundary_knot(explicit_history_time(newest))
   tolerance=64*epsilon(1.d0)*max(1.d0,abs(time))
   if(knot<=time+tolerance) explicit_trial_reset_kind=1
   if(explicit_last_accepted_dt>0.d0.and.dt>2.d0*explicit_last_accepted_dt.and.explicit_trial_reset_kind==0) &
    explicit_trial_reset_kind=2
  endif
  if(explicit_trial_reset_kind/=0) then
   ! Never carry a multistep stencil across a forcing kink. A strongly growing
   ! step after a tiny output clip also restarts the stencil. Do not clear the
   ! accepted ring until this new startup step actually succeeds.
   call cpu_ssp43(initial,dt,step_morfac,result,accepted,momentum,ok)
   return
  endif
  if(explicit_history_count<explicit_history_method-1) then
   call cpu_ssp43(initial,dt,step_morfac,result,accepted,momentum,ok)
   return
  endif
  slot=explicit_history_position
  span=time-explicit_history_time(slot)
  if(.not.ieee_is_finite(span).or.span<=dt*(1.d0+64*epsilon(1.d0))) then
   explicit_history_invalid_span_attempts=explicit_history_invalid_span_attempts+1
   call cpu_ssp43(initial,dt,step_morfac,result,accepted,momentum,ok)
   return
  endif
  ratio=dt/span;weight=ratio*ratio;tau=dt/(1.d0-ratio)
  if(profile_enabled) source_started=omp_get_wtime()
  friction_peak=0;invalid_friction=.false.
   !$omp parallel do schedule(static,256) default(shared) private(cell,depth,speed,denominator,rough_valid) &
  !$omp& reduction(max:friction_peak) reduction(.or.:invalid_friction)
  do cell=1,nc
   explicit_friction_rate(cell)=0.d0
   depth=initial(1,cell)
   if(depth>dry_tolerance) then
    speed=sqrt(sum(initial(2:3,cell)**2))/depth
     if(hydraulic_roughness_mode/=braids_roughness_manning.or. &
        roughness_minimum_work_depth>0.d0) then
      call braids_roughness_friction_rate(hydraulic_roughness_mode,depth,speed,manning_cell(cell),grav, &
       max(manning_friction_chezy_minimum_depth,roughness_minimum_work_depth), &
       white_colebrook_chzk2d,explicit_friction_rate(cell),rough_valid)
      invalid_friction=invalid_friction.or..not.rough_valid
     else if(manning_friction_chezy_minimum_depth>0.d0.and.depth<manning_friction_chezy_minimum_depth) then
     denominator=depth*manning_friction_chezy_minimum_depth**(1.d0/3.d0)
    else
     denominator=depth**(4.d0/3.d0)
    endif
     if(hydraulic_roughness_mode==braids_roughness_manning.and. &
        roughness_minimum_work_depth==0.d0) &
      explicit_friction_rate(cell)=grav*manning_cell(cell)**2*speed/denominator
    invalid_friction=invalid_friction.or..not.ieee_is_finite(explicit_friction_rate(cell))
    friction_peak=max(friction_peak,tau*explicit_friction_rate(cell))
   endif
  enddo
  !$omp end parallel do
  if(profile_enabled) then
   explicit_history_source_seconds=explicit_history_source_seconds+omp_get_wtime()-source_started
   profile_source_s=profile_source_s+omp_get_wtime()-source_started
   profile_source_calls=profile_source_calls+1
  endif
  ! A nonnegative FE damping coefficient is mandatory. Failed attempts do not
  ! alter accepted history. The outer controller safely retries with smaller dt.
  if(invalid_friction.or.friction_peak>1.d0) then
   explicit_history_friction_rejections=explicit_history_friction_rejections+1
   failure_reason=3
   return
  endif
  if(profile_enabled) started=omp_get_wtime()
  call euler(initial,tau,time,step_morfac,result,accepted,ok, &
   momentum_boundary_impulse=momentum,toe_flux_weight=1.d0)
  if(profile_enabled) then
   profile_euler_s=profile_euler_s+omp_get_wtime()-started
   profile_euler_calls=profile_euler_calls+1
  endif
  if(.not.ok) return
  if(profile_enabled) source_started=omp_get_wtime()
  !$omp parallel do schedule(static,256) default(shared) private(cell)
  do cell=1,max(nc,nb)
   if(cell<=nc) then
    result(2:3,cell)=result(2:3,cell)-tau*explicit_friction_rate(cell)*initial(2:3,cell)
    result(:,cell)=(1.d0-weight)*result(:,cell)+weight*explicit_history_state(:,cell,slot)
    if(result(1,cell)<=dry_tolerance) result(2:3,cell)=0.d0
   endif
   if(cell<=nb) accepted(:,cell)=(1.d0-weight)*accepted(:,cell)+ &
    weight*(explicit_history_ledger(:,cell,slot)-ledger(:,cell))
  enddo
  !$omp end parallel do
  ! The hydraulic-only guard excludes secondary momentum/bank bookkeeping.
  momentum=(1.d0-weight)*momentum
  if(profile_enabled) then
   explicit_history_source_seconds=explicit_history_source_seconds+omp_get_wtime()-source_started
   profile_source_s=profile_source_s+omp_get_wtime()-source_started
  endif
  used_history=.true.
  explicit_trial_span=span;explicit_trial_ratio=ratio
  explicit_trial_friction_peak=friction_peak
  explicit_trial_physical_cfl=trial_cfl_peak*dt/tau
  ok=all(ieee_is_finite(result)).and.minval(result(1,:))>=0.d0.and. &
   minval(result(4:nphysical_state,:))>=0.d0
 end subroutine

 subroutine explicit_history_commit(initial,initial_ledger,initial_time,dt,used_history)
  real(8),intent(in) :: initial(nstate,nc),initial_ledger(nledger,nb),initial_time,dt
  logical,intent(in) :: used_history
  integer :: cell,slot
  if(explicit_trial_reset_kind/=0) then
   explicit_history_count=0;explicit_history_position=1
   if(explicit_trial_reset_kind==1) explicit_history_knot_resets=explicit_history_knot_resets+1
   if(explicit_trial_reset_kind==2) explicit_history_growth_resets=explicit_history_growth_resets+1
  endif
  slot=explicit_history_position
  !$omp parallel do schedule(static,256) default(shared) private(cell)
  do cell=1,max(nc,nb)
   if(cell<=nc) explicit_history_state(:,cell,slot)=initial(:,cell)
   if(cell<=nb) explicit_history_ledger(:,cell,slot)=initial_ledger(:,cell)
  enddo
  !$omp end parallel do
  explicit_history_time(slot)=initial_time
  explicit_history_count=min(explicit_history_method-1,explicit_history_count+1)
  explicit_history_position=mod(slot,explicit_history_method-1)+1
  explicit_last_accepted_dt=dt
  if(used_history) then
   explicit_history_steps=explicit_history_steps+1
   explicit_history_physical_time=explicit_history_physical_time+dt
   explicit_history_minimum_span=min(explicit_history_minimum_span,explicit_trial_span)
   explicit_history_maximum_span=max(explicit_history_maximum_span,explicit_trial_span)
   explicit_history_maximum_ratio=max(explicit_history_maximum_ratio,explicit_trial_ratio)
   explicit_history_maximum_friction=max(explicit_history_maximum_friction,explicit_trial_friction_peak)
   explicit_history_maximum_physical_cfl=max(explicit_history_maximum_physical_cfl,explicit_trial_physical_cfl)
  else
   explicit_history_startup_steps=explicit_history_startup_steps+1
   explicit_history_startup_time=explicit_history_startup_time+dt
  endif
 end subroutine

 subroutine cpu_outlet_bounds(c,v,grad,vlo,vhi)
  ! Qualified wet, subcritical stage-outlet bound extension. A one-sided sloping stage
  ! outlet already permits extrapolation at the open face. Its missing exterior
  ! stencil bound must also be represented when limiting the OTHER faces;
  ! otherwise a transverse perturbation can zero the whole longitudinal slope.
  ! The extrapolated virtual centre is two centre-to-face displacements away.
  ! No ghost state is fed to the Riemann solver and no physical BC is changed.
  integer,intent(in) :: c
  real(8),intent(in) :: v(nprimitive,nc),grad(nprimitive,2,nc)
  real(8),intent(inout) :: vlo(nprimitive,nc),vhi(nprimitive,nc)
  real(8) :: r(2),delta(nprimitive),exterior(nprimitive),h
  integer :: k,f,b
  if(cpu_outlet_extension==0) return
  h=v(1,c)-v(bed_primitive_index(),c)
  if(h<=dry_tolerance) return
  do k=offset(c),offset(c+1)-1
   f=faces(k);b=bid(f)
   if(neighbour(f)/=0.or.b<=0) cycle
   if(mode(b)/=2) cycle
   if(abs(dot_product(v(2:3,c),normal(:,f)))>=sqrt(grav*h)) cycle
   r=mid(:,f)-xy(:,c);delta=grad(:,1,c)*r(1)+grad(:,2,c)*r(2)
   if(abs(delta(bed_primitive_index()))<=64*epsilon(1.d0)*max(1.d0,abs(v(bed_primitive_index(),c)))) cycle
   exterior=v(:,c)+2*delta
   vlo(:,c)=min(vlo(:,c),exterior);vhi(:,c)=max(vhi(:,c),exterior)
  enddo
 end subroutine

 ! Resolve once after input loading and checkpoint restoration. The selected
 ! method is fixed for the run; source terms are never silently disabled.
 subroutine resolve_cpu_time_integrator(restarted)
  logical,intent(in) :: restarted
  integer :: status
  character(len=64) :: setting
  cpu_time_integrator_requested='auto'
  call get_environment_variable('BRAIDS_TIME_INTEGRATOR',setting,status=status)
  if(status==-1) error stop 'BRAIDS_TIME_INTEGRATOR value is too long'
  if(status==0.and.len_trim(setting)>0) cpu_time_integrator_requested=trim(setting)
  ! This is the complete prior hydraulic-research guard, with explicit bank
  ! redistribution and scheme checks added for conservative auto selection.
  cpu_multistep_eligible=.not.amr_enabled.and.bedlaw==0.and.exchange_mode==0.and. &
   .not.secondary_flow.and..not.bank_failure.and..not.bank_avalanche.and. &
   .not.general_avalanche.and..not.bedform_roughness.and..not.bedload_adaptation.and. &
   bed_slope==0.and.horizontal_viscosity==0.d0.and.bank_erosion_fraction==0.d0.and. &
   bank_toe_erosion_mode==0.and..not.bank_front_tracking.and.scheme==6.and. &
   nstate==5.and.nfrac==1.and.all(ws_fraction==0.d0).and.all(q(4,:)==0.d0).and. &
   all(forcing(3:4,:)==0.d0).and.all(series_concentration==0.d0).and. &
   all(series_bed_input==0.d0).and.all(fraction_concentration==0.d0).and. &
   all(fraction_bed_input==0.d0).and.all(fraction_series_concentration==0.d0).and. &
   all(fraction_series_bed_input==0.d0)
  ! Coupled SSP43 retains split friction, exact secondary relaxation and the
  ! existing viscosity update. Its MPM bend qualification must not imply that
  ! exchange, AMR, bank collapse or every other bedload law has been qualified.
  cpu_ssp43_eligible=.not.amr_enabled.and..not.bank_failure.and..not.bank_avalanche.and. &
   .not.general_avalanche.and..not.bedform_roughness.and..not.bedload_adaptation.and. &
   exchange_mode==0.and.nmorfac==0.and.bank_erosion_fraction==0.d0.and. &
   bank_toe_erosion_mode==0.and..not.bank_front_tracking.and.scheme==6.and. &
   (bedlaw==0.or.bedlaw==2).and..not.suspended_transport_active
  select case(trim(cpu_time_integrator_requested))
  case('auto')
   cpu_time_integrator=33
   if(cpu_ssp43_eligible) cpu_time_integrator=43
   if(cpu_multistep_eligible) cpu_time_integrator=6
  case('legacy')
   cpu_time_integrator=33
  case('ssp43')
   if(.not.cpu_ssp43_eligible) error stop 'SSP43 requested outside qualified fixed-grid scope'
   cpu_time_integrator=43
  case('multistep6')
   if(.not.cpu_multistep_eligible) error stop 'SSPMS6 requested outside qualified hydraulic-only scope'
   cpu_time_integrator=6
  case default
   error stop 'BRAIDS_TIME_INTEGRATOR must be auto, legacy, ssp43 or multistep6'
  end select
  explicit_history_method=0;cpu_outlet_extension=0
  if(cpu_time_integrator==6) explicit_history_method=6
  ! Only the hydraulic SSPMS qualification included the outlet-bound repair.
  ! Coupled SSP43 retains exactly its previously qualified spatial bounds.
  if(cpu_time_integrator==6) cpu_outlet_extension=1
  write(*,'(a,a)') 'time integrator: requested=',trim(cpu_time_integrator_requested)
  select case(cpu_time_integrator)
  case(6)
   write(*,'(a)') 'time integrator: active=SSPMS6 hydraulic-only; SSP43 startup/restart'
   if(restarted) write(*,'(a)') &
    'time integrator: restart resets multistep history; SSP43 rebuilds accepted history (not bitwise restart)'
  case(43)
   write(*,'(a)') 'time integrator: active=SSPRK(4,3) fixed-grid; existing coupled source operators retained'
  case(33)
   write(*,'(a)') 'time integrator: active=legacy SSPRK(3,3)'
   if(trim(cpu_time_integrator_requested)=='auto') write(*,'(a)') &
    'time integrator: legacy fallback because configuration is outside qualified accelerated scope'
  end select
  write(*,'(a,es22.14)') 'time integrator: input Euler-stage CFL (unchanged)=',cfl_target
  write(*,'(a,i0)') 'time integrator: wet stage-outlet bound extension=',cpu_outlet_extension
 end subroutine

 subroutine advance(dt,ok)
  real(8),intent(in) :: dt
  logical,intent(out) :: ok
  select case(cpu_time_integrator)
  case(6)
   call advance_multistep6(dt,ok)
   if(.not.ok) explicit_history_failures=explicit_history_failures+1
  case(43)
   call advance_ssp43(dt,ok)
  case(33)
   call advance_ssp33(dt,ok)
  case default
   error stop 'invalid resolved CPU time integrator'
  end select
 end subroutine

 subroutine profile_report()
  integer :: team
 if(.not.profile_enabled) return
 write(*,'(a,i0)') 'time integrator: active method=',cpu_time_integrator
  write(*,'(a,i0,a,l1,a,es22.14)') 'profile: hydraulic roughness mode=', &
   hydraulic_roughness_mode,' ChzK2D=',white_colebrook_chzk2d, &
   ' work depth=',roughness_minimum_work_depth
  write(*,'(a,i0)') 'profile: sediment RP_TAUB shear mode=',sediment_shear_mode

 write(*,'(a,i0)') 'explicit history: method=',explicit_history_method
  write(*,'(a,i0)') 'explicit history: accepted multistep steps=',explicit_history_steps
  write(*,'(a,i0)') 'explicit history: SSP43 startup/fallback steps=',explicit_history_startup_steps
  write(*,'(a,i0)') 'explicit history: failed attempts=',explicit_history_failures
  write(*,'(a,i0)') 'explicit history: invalid span fallback attempts=',explicit_history_invalid_span_attempts
  write(*,'(a,i0)') 'explicit history: friction rejected attempts=',explicit_history_friction_rejections
  write(*,'(a,i0)') 'explicit history: forcing knot resets=',explicit_history_knot_resets
  write(*,'(a,i0)') 'explicit history: excessive step growth resets=',explicit_history_growth_resets
  write(*,'(a,es22.14)') 'explicit history: multistep physical seconds=',explicit_history_physical_time
  write(*,'(a,es22.14)') 'explicit history: startup/fallback physical seconds=',explicit_history_startup_time
  write(*,'(a,es22.14)') 'explicit history: source/history combination wall seconds=',explicit_history_source_seconds
  write(*,'(a,es22.14)') 'explicit history: minimum accepted history span=', &
   merge(explicit_history_minimum_span,0.d0,explicit_history_steps>0)
  write(*,'(a,es22.14)') 'explicit history: maximum accepted history span=',explicit_history_maximum_span
  write(*,'(a,es22.14)') 'explicit history: maximum accepted dt/span=',explicit_history_maximum_ratio
  write(*,'(a,es22.14)') 'explicit history: maximum accepted friction FE factor=',explicit_history_maximum_friction
  write(*,'(a,es22.14)') 'explicit history: maximum accepted physical CFL=',explicit_history_maximum_physical_cfl

  team=1
  !$omp parallel default(shared)
  !$omp single
  team=omp_get_num_threads()
  !$omp end single
  !$omp end parallel
  write(*,'(a,f12.4)') 'profile: measured run wall seconds=',omp_get_wtime()-profile_started
  write(*,'(a,i0,a,i0,a,i0)') 'profile: OpenMP max threads=',omp_get_max_threads(), &
   ' observed team=',team,' logical processors=',omp_get_num_procs()
  write(*,'(a,f12.4,a,i0)') 'profile: AMR regrid wall seconds=',profile_regrid_s,' calls=',profile_regrid_calls
  write(*,'(a,f12.4,a,i0)') 'profile: hydraulic preparation wall seconds=',profile_prepare_s,' calls=',profile_prepare_calls
  write(*,'(a,f12.4,a,i0)') 'profile: Euler update wall seconds=',profile_euler_s,' calls=',profile_euler_calls
  write(*,'(a,i0)') 'profile: optimized hydraulic workflow calls=',profile_hydraulic_workflow_calls
  write(*,'(a,f12.4)') 'profile: Euler setup/reconstruction wall seconds=',profile_euler_setup_s
  write(*,'(a,f12.4)') 'profile: Euler boundary-control wall seconds=',profile_euler_boundary_s
  write(*,'(a,f12.4)') 'profile: Euler bedload-capacity wall seconds=',profile_euler_bedload_s
  write(*,'(a,f12.4)') 'profile: Euler hiding wall seconds=',profile_euler_hiding_s
  write(*,'(a,f12.4)') 'profile: Euler mixture statistics wall seconds=',profile_euler_mixture_s
  write(*,'(a,f12.4)') 'profile: Euler cell capacity wall seconds=',profile_euler_capacity_cell_s
  write(*,'(a,f12.4)') 'profile: Euler capacity gradients wall seconds=',profile_euler_capacity_gradient_s
  write(*,'(a,f12.4)') 'profile: Euler reconstruction wall seconds=',profile_euler_reconstruction_s
  write(*,'(a,f12.4)') 'profile: Euler boundary-trace wall seconds=',profile_euler_trace_s
  write(*,'(a,f12.4)') 'profile: Euler face-flux wall seconds=',profile_euler_flux_s
  write(*,'(a,f12.4)') 'profile: Euler limiting/update wall seconds=',profile_euler_limit_s
  write(*,'(a,f12.4)') 'profile: Euler finalization wall seconds=',profile_euler_finalize_s
  write(*,'(a,f12.4,a,i0)') 'profile: source-term wall seconds=',profile_source_s,' calls=',profile_source_calls
  write(*,'(a,f12.4,a,i0)') 'profile: horizontal diffusion wall seconds=',profile_diffusion_s,' substeps=',diffusion_substeps
  write(*,'(a,f10.6,a,i0,a,i0)') 'profile: diffusion minimum correction factor=', &
   diffusion_minimum_correction_factor,' recovered faces=',diffusion_recovered_faces, &
   ' recovery fallbacks=',diffusion_recovery_fallbacks
  write(*,'(a,f12.4,a,i0)') 'profile: result output wall seconds=',profile_output_s,' calls=',profile_output_calls
  write(*,'(a,f12.4)') 'profile: authoritative output wall seconds=',profile_output_authoritative_s
  write(*,'(a,f12.4)') 'profile: preview packing wall seconds=',profile_output_preview_pack_s
  write(*,'(a,f12.4)') 'profile: preview I/O wall seconds=',profile_output_preview_io_s
  write(*,'(a,f12.4)') 'profile: checkpoint output wall seconds=',profile_output_checkpoint_s
 end subroutine

 subroutine gpu_put_header(unit,name,kind,dimensions)
  integer,intent(in) :: unit,kind,dimensions(:)
  character(*),intent(in) :: name
  character(64) :: padded
  integer :: dims(4)
  if(size(dimensions)>4.or.len_trim(name)>64) error stop 'GPU export schema overflow'
  padded=name;dims=0;dims(1:size(dimensions))=dimensions
  write(unit) padded,kind,size(dimensions),dims
 end subroutine
 subroutine gpu_put_r(unit,name,values,dimensions)
  integer,intent(in) :: unit,dimensions(:)
  character(*),intent(in) :: name
  real(8),intent(in) :: values(:)
  call gpu_put_header(unit,name,2,dimensions);write(unit) values
 end subroutine
 subroutine gpu_put_i(unit,name,values,dimensions)
  integer,intent(in) :: unit,dimensions(:),values(:)
  character(*),intent(in) :: name
  call gpu_put_header(unit,name,1,dimensions);write(unit) values
 end subroutine
 subroutine gpu_finish(unit)
  integer,intent(in) :: unit
  call gpu_put_header(unit,'END',0,[integer::]);close(unit)
 end subroutine
 subroutine gpu_open(unit,path)
  integer,intent(out) :: unit
  character(*),intent(in) :: path
  open(newunit=unit,file=path,access='stream',form='unformatted',status='replace')
  write(unit) 'BRDGPU01',1
 end subroutine

 subroutine gpu_export_input()
  integer :: gpu_unit,c,f,g,l
  character(len=64) :: gpu_secondary_method_id, &
   gpu_secondary_preimage_sha
  integer :: gpu_bed_indices(nbed_layers,nfrac),gpu_suspended_indices(nfrac),gpu_face_mode(nf),gpu_active(nc)
  integer :: gpu_fixed_boundary_bed_mode(nb)
  real(8) :: gpu_bed(nc),gpu_distance(2,nf),gpu_layers(nbed_layers,nfrac,nc),gpu_fixed_boundary_bed(nf)
  real(8) :: gpu_state(nstate,nc)
  if(amr_enabled.or.scheme/=6) error stop 'GPU preparation requires fixed-grid scheme6'
  call effective_bed_elevation(q,gpu_bed)
  do g=1,nfrac
   gpu_suspended_indices(g)=suspended_index(g)-1
   do l=1,nbed_layers
    gpu_bed_indices(l,g)=bed_layer_index(l,g)-1
    gpu_layers(l,g,:)=q(bed_layer_index(l,g),:)
   enddo
  enddo
  gpu_face_mode=0;gpu_distance=0;gpu_active=1
  gpu_fixed_boundary_bed_mode=0;gpu_fixed_boundary_bed=0.d0
  if(bedlaw==9) then
   if(.not.allocated(fixed_boundary_bed_mode).or..not.allocated(fixed_boundary_bed)) &
    error stop 'GPU TELEMAC-GAIA fixed boundary bed is not initialized'
   gpu_fixed_boundary_bed_mode=fixed_boundary_bed_mode
   gpu_fixed_boundary_bed=fixed_boundary_bed
  endif
  do f=1,nf
   if(bid(f)>0) gpu_face_mode(f)=mode(bid(f))
   gpu_distance(1,f)=abs(dot_product(mid(:,f)-xy(:,owner(f)),normal(:,f)))
   if(neighbour(f)>0) gpu_distance(2,f)=abs(dot_product(xy(:,neighbour(f))-mid(:,f),normal(:,f)))
  enddo
  gpu_secondary_method_id='secondary.equilibrium'
  gpu_secondary_preimage_sha=''
  if(allocated(secondary_foundation_execution_preimage)) &
   call sha256_ascii(secondary_foundation_execution_preimage, &
    gpu_secondary_preimage_sha)
  call gpu_open(gpu_unit,'coupled_input.bin')
  call gpu_put_i(gpu_unit,'nc',[nc],[integer::])
  call gpu_put_i(gpu_unit,'nf',[nf],[integer::])
  call gpu_put_i(gpu_unit,'nb',[nb],[integer::])
  call gpu_put_i(gpu_unit,'nx',[nx],[integer::])
  call gpu_put_i(gpu_unit,'ny',[ny],[integer::])
  call gpu_put_i(gpu_unit,'scheme',[scheme],[integer::])
  call gpu_put_i(gpu_unit,'bedlaw',[bedlaw],[integer::])
  call gpu_put_i(gpu_unit,'exchange_mode',[exchange_mode],[integer::])
  call gpu_put_i(gpu_unit,'bed_slope',[bed_slope],[integer::])
  call gpu_put_i(gpu_unit,'deck_version',[deck_version],[integer::])
  call gpu_put_i(gpu_unit,'nfrac',[nfrac],[integer::])
  call gpu_put_i(gpu_unit,'nstate',[nstate],[integer::])
  call gpu_put_i(gpu_unit,'nprimitive',[nprimitive],[integer::])
  call gpu_put_i(gpu_unit,'nledger',[nledger],[integer::])
  call gpu_put_i(gpu_unit,'nbed_layers',[nbed_layers],[integer::])
  call gpu_put_i(gpu_unit,'nsubstrate_layers',[nsubstrate_layers],[integer::])
  call gpu_put_i(gpu_unit,'nphysical_state',[nphysical_state],[integer::])
  call gpu_put_i(gpu_unit,'nphysical_primitive',[nphysical_primitive],[integer::])
  call gpu_put_i(gpu_unit,'nphysical_ledger',[nphysical_ledger],[integer::])
  call gpu_put_i(gpu_unit,'cpu_time_integrator',[cpu_time_integrator],[integer::])
  call gpu_put_i(gpu_unit,'cpu_outlet_extension',[cpu_outlet_extension],[integer::])
  call gpu_put_i(gpu_unit,'active_layer_mode',[active_layer_mode],[integer::])
  call gpu_put_i(gpu_unit,'hiding_exposure',[hiding_exposure],[integer::])
  call gpu_put_i(gpu_unit,'mpm_friction_mode',[mpm_friction_mode],[integer::])
  call gpu_put_i(gpu_unit,'sediment_roughness_mode',[sediment_roughness_mode],[integer::])
  call gpu_put_i(gpu_unit,'settling_mode',[settling_mode],[integer::])
  call gpu_put_i(gpu_unit,'secondary_flow_model',[secondary_flow_model],[integer::])
  call gpu_put_i(gpu_unit,'secondary_flow_inflow_mode',[secondary_flow_inflow_mode],[integer::])
  call gpu_put_i(gpu_unit,'horizontal_viscosity_mode',[horizontal_viscosity_mode],[integer::])
  call gpu_put_i(gpu_unit,'horizontal_viscosity_geometry',[horizontal_viscosity_geometry],[integer::])
  call gpu_put_i(gpu_unit,'nmorfac',[nmorfac],[integer::])
  call gpu_put_i(gpu_unit,'bank_toe_erosion_mode',[bank_toe_erosion_mode],[integer::])
  call gpu_put_r(gpu_unit,'grav',[grav],[integer::])
  call gpu_put_r(gpu_unit,'dry_tolerance',[dry_tolerance],[integer::])
  call gpu_put_r(gpu_unit,'rho',[rho],[integer::])
  call gpu_put_r(gpu_unit,'speed_limit',[speed_limit],[integer::])
  call gpu_put_r(gpu_unit,'dx',[dx],[integer::])
  call gpu_put_r(gpu_unit,'dy',[dy],[integer::])
  call gpu_put_r(gpu_unit,'time',[time],[integer::])
  call gpu_put_r(gpu_unit,'dtmax',[dtmax],[integer::])
  call gpu_put_r(gpu_unit,'endtime',[endtime],[integer::])
  call gpu_put_r(gpu_unit,'output_dt',[output_dt],[integer::])
  call gpu_put_r(gpu_unit,'cfl_target',[cfl_target],[integer::])
  call gpu_put_r(gpu_unit,'mpm_cf',[mpm_cf],[integer::])
  call gpu_put_r(gpu_unit,'mpm_critical',[mpm_critical],[integer::])
  call gpu_put_r(gpu_unit,'bedload_adaptation_length_diameters',[bedload_adaptation_length_diameters],[integer::])
  call gpu_put_r(gpu_unit,'active_layer_thickness',[active_layer_thickness],[integer::])
  call gpu_put_r(gpu_unit,'active_layer_depth_fraction',[active_layer_depth_fraction],[integer::])
  call gpu_put_r(gpu_unit,'active_layer_minimum_thickness',[active_layer_minimum_thickness],[integer::])
  call gpu_put_r(gpu_unit,'substrate_layer_thickness',[substrate_layer_thickness],[integer::])
  call gpu_put_r(gpu_unit,'hiding_parker_alpha',[hiding_parker_alpha],[integer::])
  call gpu_put_r(gpu_unit,'hiding_wu_exponent',[hiding_wu_exponent],[integer::])
  call gpu_put_r(gpu_unit,'transport_calibration',[transport_calibration],[integer::])
  call gpu_put_r(gpu_unit,'bed_slope_alfabs',[bed_slope_alfabs],[integer::])
  call gpu_put_r(gpu_unit,'bed_slope_alfabn',[bed_slope_alfabn],[integer::])
  call gpu_put_r(gpu_unit,'bed_friction_angle_deg',[bed_friction_angle_deg],[integer::])
  call gpu_put_r(gpu_unit,'secondary_flow_espir',[secondary_flow_espir],[integer::])
  call gpu_put_r(gpu_unit,'secondary_flow_momentum_beta',[secondary_flow_momentum_beta],[integer::])
  call gpu_put_r(gpu_unit,'gaia_skin_roughness_ratio',[gaia_skin_roughness_ratio],[integer::])
  call gpu_put_r(gpu_unit,'gaia_slope_beta',[gaia_slope_beta],[integer::])
  call gpu_put_r(gpu_unit,'gaia_deviation_beta2',[gaia_deviation_beta2],[integer::])
  call gpu_put_r(gpu_unit,'gaia_secondary_alpha',[gaia_secondary_alpha],[integer::])
  call gpu_put_r(gpu_unit,'morphological_factor',[morphological_factor],[integer::])
  call gpu_put_r(gpu_unit,'horizontal_viscosity',[horizontal_viscosity],[integer::])
  call gpu_put_r(gpu_unit,'manning_chezy_minimum_depth',[manning_chezy_minimum_depth],[integer::])
  call gpu_put_r(gpu_unit,'manning_friction_chezy_minimum_depth',[manning_friction_chezy_minimum_depth],[integer::])
  call gpu_put_r(gpu_unit,'hindered_settling_reference_concentration',[hindered_settling_reference_concentration],[integer::])
  call gpu_put_r(gpu_unit,'bank_erosion_fraction',[bank_erosion_fraction],[integer::])
  call gpu_put_i(gpu_unit,'amr_enabled',[merge(1,0,amr_enabled)],[integer::])
  call gpu_put_i(gpu_unit,'secondary_flow',[merge(1,0,secondary_flow)],[integer::])
  call gpu_put_i(gpu_unit,'secondary_flow_transport_direction',[merge(1,0,secondary_flow_transport_direction)],[integer::])
  call gpu_put_i(gpu_unit,'bedload_adaptation',[merge(1,0,bedload_adaptation)],[integer::])
  call gpu_put_i(gpu_unit,'bedform_roughness',[merge(1,0,bedform_roughness)],[integer::])
  call gpu_put_i(gpu_unit,'bank_failure',[merge(1,0,bank_failure)],[integer::])
  call gpu_put_i(gpu_unit,'bank_front_tracking',[merge(1,0,bank_front_tracking)],[integer::])
  call gpu_put_i(gpu_unit,'bank_avalanche',[merge(1,0,bank_avalanche)],[integer::])
  call gpu_put_i(gpu_unit,'general_avalanche',[merge(1,0,general_avalanche)],[integer::])
  call gpu_put_i(gpu_unit,'production_section_geometry',[merge(1,0,production_section_geometry)],[integer::])
  call gpu_put_i(gpu_unit,'suspended_transport_active',[merge(1,0,suspended_transport_active)],[integer::])
  call gpu_put_i(gpu_unit,'morphological_acceleration',[merge(1,0,morphological_acceleration)],[integer::])
  call gpu_put_i(gpu_unit,'secondary_index',[merge(secondary_state_index()-1,-1,prognostic_secondary_flow())],[integer::])
  call gpu_put_i(gpu_unit,'secondary_ledger_index',[merge(secondary_ledger_index()-1,-1,prognostic_secondary_flow())],[integer::])
  call gpu_put_i(gpu_unit,'secondary_primitive_index',[merge(secondary_primitive_index()-1,-1,prognostic_secondary_flow())],[integer::])
  call gpu_put_r(gpu_unit,'state',reshape(q,[size(q)]),[nstate,nc])
  call gpu_put_r(gpu_unit,'ledger',reshape(ledger,[size(ledger)]),[nledger,nb])
  call gpu_put_r(gpu_unit,'bed',reshape(gpu_bed,[size(gpu_bed)]),[nc])
  call gpu_put_r(gpu_unit,'layer_mass',reshape(gpu_layers,[size(gpu_layers)]),[nbed_layers,nfrac,nc])
  call gpu_put_r(gpu_unit,'area',reshape(area,[size(area)]),[nc])
  call gpu_put_r(gpu_unit,'rock',reshape(rock,[size(rock)]),[nc])
  call gpu_put_r(gpu_unit,'xy',reshape(xy,[size(xy)]),[2,nc])
  call gpu_put_r(gpu_unit,'normal',reshape(normal,[size(normal)]),[2,nf])
  call gpu_put_r(gpu_unit,'length',reshape(length,[size(length)]),[nf])
  call gpu_put_r(gpu_unit,'mid',reshape(mid,[size(mid)]),[2,nf])
  call gpu_put_r(gpu_unit,'distance',reshape(gpu_distance,[size(gpu_distance)]),[2,nf])
  call gpu_put_r(gpu_unit,'forcing',reshape(forcing,[size(forcing)]),[6,nb])
  call gpu_put_r(gpu_unit,'manning_cell',reshape(manning_cell,[size(manning_cell)]),[nc])
  call gpu_put_r(gpu_unit,'hydraulic_ls_matrix',reshape(hydraulic_ls_matrix,[size(hydraulic_ls_matrix)]),[4,nc])
  call gpu_put_r(gpu_unit,'hydraulic_ls_weight',reshape(hydraulic_ls_weight,[size(hydraulic_ls_weight)]),[size(faces)])
  call gpu_put_r(gpu_unit,'bed_ls_matrix',reshape(bed_ls_matrix,[size(bed_ls_matrix)]),[4,nc])
  call gpu_put_r(gpu_unit,'bed_ls_delta',reshape(bed_ls_delta,[size(bed_ls_delta)]),[2,size(faces)])
  call gpu_put_r(gpu_unit,'bed_ls_weight',reshape(bed_ls_weight,[size(bed_ls_weight)]),[size(faces)])
  call gpu_put_r(gpu_unit,'rectangle_width',reshape(rectangle_width,[size(rectangle_width)]),[2,nc])
  call gpu_put_r(gpu_unit,'face_displacement',reshape(face_displacement,[size(face_displacement)]),[2,2,nf])
  call gpu_put_r(gpu_unit,'wall_stencil_normal',reshape(wall_stencil_normal,[size(wall_stencil_normal)]),[2,5,2,nc])
  call gpu_put_r(gpu_unit,'wall_box_center',reshape(wall_box_center,[size(wall_box_center)]),[2,nc])
  call gpu_put_r(gpu_unit,'wall_normal',reshape(wall_normal,[size(wall_normal)]),[2,nc])
  call gpu_put_r(gpu_unit,'wall_mean_distance',reshape(wall_mean_distance,[size(wall_mean_distance)]),[nc])
  call gpu_put_r(gpu_unit,'boundary_bed_offset',reshape(boundary_bed_offset,[size(boundary_bed_offset)]),[nf])
  call gpu_put_r(gpu_unit,'fixed_boundary_bed',reshape(gpu_fixed_boundary_bed,[size(gpu_fixed_boundary_bed)]),[nf])
  call gpu_put_r(gpu_unit,'boundary_bed_left_offset',reshape(boundary_bed_left_offset,[size(boundary_bed_left_offset)]),[nf])
  call gpu_put_r(gpu_unit,'boundary_bed_right_offset',reshape(boundary_bed_right_offset,[size(boundary_bed_right_offset)]),[nf])
  call gpu_put_r(gpu_unit,'fraction_concentration',reshape(fraction_concentration,[size(fraction_concentration)]),[nfrac,nb])
  call gpu_put_r(gpu_unit,'fraction_bed_input',reshape(fraction_bed_input,[size(fraction_bed_input)]),[nfrac,nb])
  call gpu_put_r(gpu_unit,'section_energy_slope',reshape(section_energy_slope,[size(section_energy_slope)]),[nb])
  call gpu_put_r(gpu_unit,'bulk_fraction',reshape(bulk_fraction,[size(bulk_fraction)]),[nfrac])
  call gpu_put_r(gpu_unit,'ws_fraction',reshape(ws_fraction,[size(ws_fraction)]),[nfrac])
  call gpu_put_r(gpu_unit,'d50_fraction',reshape(d50_fraction,[size(d50_fraction)]),[nfrac])
  call gpu_put_r(gpu_unit,'d90_fraction',reshape(d90_fraction,[size(d90_fraction)]),[nfrac])
  call gpu_put_r(gpu_unit,'rhos_fraction',reshape(rhos_fraction,[size(rhos_fraction)]),[nfrac])
  call gpu_put_r(gpu_unit,'z0_fraction',reshape(z0_fraction,[size(z0_fraction)]),[nfrac])
  call gpu_put_r(gpu_unit,'sedthr_fraction',reshape(sedthr_fraction,[size(sedthr_fraction)]),[nfrac])
  call gpu_put_r(gpu_unit,'bedload_min_depth_fraction',reshape(bedload_min_depth_fraction,[size(bedload_min_depth_fraction)]),[nfrac])
  call gpu_put_r(gpu_unit,'reference_height_fraction',reshape(reference_height_fraction,[size(reference_height_fraction)]),[nfrac])
  call gpu_put_r(gpu_unit,'dstar_fraction',reshape(dstar_fraction,[size(dstar_fraction)]),[nfrac])
  call gpu_put_r(gpu_unit,'taucr_fraction',reshape(taucr_fraction,[size(taucr_fraction)]),[nfrac])
  call gpu_put_i(gpu_unit,'bed_indices',reshape(gpu_bed_indices,[size(gpu_bed_indices)]),[nbed_layers,nfrac])
  call gpu_put_i(gpu_unit,'suspended_indices',reshape(gpu_suspended_indices,[size(gpu_suspended_indices)]),[nfrac])
  call gpu_put_i(gpu_unit,'owner',reshape(owner-1,[size(owner-1)]),[nf])
  call gpu_put_i(gpu_unit,'neighbour',reshape(neighbour-1,[size(neighbour-1)]),[nf])
  call gpu_put_i(gpu_unit,'bid',reshape(bid-1,[size(bid-1)]),[nf])
  call gpu_put_i(gpu_unit,'offset',reshape(offset-1,[size(offset-1)]),[nc+1])
  call gpu_put_i(gpu_unit,'faces',reshape(faces-1,[size(faces-1)]),[size(faces)])
  call gpu_put_i(gpu_unit,'signs',reshape(signs,[size(signs)]),[size(faces)])
  call gpu_put_i(gpu_unit,'active',reshape(gpu_active,[size(gpu_active)]),[nc])
  call gpu_put_i(gpu_unit,'boundary_mode',reshape(gpu_face_mode,[size(gpu_face_mode)]),[nf])
  call gpu_put_i(gpu_unit,'boundary_offset',reshape(boundary_offset-1,[size(boundary_offset-1)]),[nb+1])
  call gpu_put_i(gpu_unit,'boundary_faces',reshape(boundary_faces-1,[size(boundary_faces-1)]),[size(boundary_faces)])
  call gpu_put_i(gpu_unit,'mode',reshape(mode,[size(mode)]),[nb])
  call gpu_put_i(gpu_unit,'reversal_policy',reshape(reversal_policy,[size(reversal_policy)]),[nb])
  call gpu_put_i(gpu_unit,'bed_boundary_mode',reshape(bed_boundary_mode,[size(bed_boundary_mode)]),[nb])
  call gpu_put_i(gpu_unit,'fixed_boundary_bed_mode',reshape(gpu_fixed_boundary_bed_mode,[size(gpu_fixed_boundary_bed_mode)]),[nb])
  call gpu_put_i(gpu_unit,'section_control',reshape(section_control,[size(section_control)]),[nb])
  call gpu_put_i(gpu_unit,'sampled',reshape(sampled_boundary_bed,[size(sampled_boundary_bed)]),[nf])
  call gpu_put_i(gpu_unit,'viscosity_equilibrated_inlet',reshape(merge(1,0,viscosity_equilibrated_inlet),[size(merge(1,0,viscosity_equilibrated_inlet))]),[nb])
  call gpu_put_i(gpu_unit,'sampled_boundary_open',reshape(merge(1,0,sampled_boundary_open),[size(merge(1,0,sampled_boundary_open))]),[nf])
  call gpu_put_i(gpu_unit,'weno_cells',reshape(weno_cells-1,[size(weno_cells-1)]),[5,2,nf])
  call gpu_put_i(gpu_unit,'wall_triangle',reshape(merge(1,0,wall_triangle),[size(merge(1,0,wall_triangle))]),[nc])
  call gpu_put_i(gpu_unit,'wall_stencil_ids',reshape(wall_stencil_ids-1,[size(wall_stencil_ids-1)]),[5,2,nc])
  call gpu_put_i(gpu_unit,'wall_stencil_kind',reshape(wall_stencil_kind,[size(wall_stencil_kind)]),[5,2,nc])
  call gpu_put_i(gpu_unit,'wall_slot',reshape(wall_slot-1,[size(wall_slot-1)]),[2,nf])
  call gpu_put_i(gpu_unit,'open_boundary_distance',reshape(open_boundary_distance,[size(open_boundary_distance)]),[nc])
  call gpu_put_i(gpu_unit,'uncoupled_boundary_distance',reshape(uncoupled_boundary_distance,[size(uncoupled_boundary_distance)]),[nc])
  call gpu_put_i(gpu_unit,'nseries',reshape(nseries,[size(nseries)]),[nb])
  call gpu_put_i(gpu_unit,'nhydro_series',reshape(nhydro_series,[size(nhydro_series)]),[nb])
  call gpu_put_i(gpu_unit,'nfraction_series',reshape(nfraction_series,[size(nfraction_series)]),[nfrac,nb])
  call gpu_put_i(gpu_unit,'sediment_type_fraction',reshape(sediment_type_fraction,[size(sediment_type_fraction)]),[nfrac])
  call gpu_put_i(gpu_unit,'sediment_roughness_fraction',reshape(sediment_roughness_fraction,[size(sediment_roughness_fraction)]),[nfrac])
  call gpu_put_r(gpu_unit,'engelund_hansen_chezy',[engelund_hansen_chezy],[integer::])
  call gpu_put_i(gpu_unit,'native_hydraulic_roughness_mode',[hydraulic_roughness_mode],[integer::])
  call gpu_put_i(gpu_unit,'native_white_colebrook_chzk2d',[merge(1,0,white_colebrook_chzk2d)],[integer::])
  call gpu_put_i(gpu_unit,'native_sediment_shear_mode',[sediment_shear_mode],[integer::])
  call gpu_put_r(gpu_unit,'native_roughness_minimum_work_depth_m',[roughness_minimum_work_depth],[integer::])
  call gpu_put_r(gpu_unit,'native_skin_kinematic_viscosity_m2_s',[skin_kinematic_viscosity],[integer::])
  call gpu_put_r(gpu_unit,'native_skin_sand_roughness_m',[skin_sand_roughness],[integer::])
  call gpu_put_r(gpu_unit,'native_skin_silt_roughness_m',[skin_silt_roughness],[integer::])
  call gpu_put_r(gpu_unit,'native_skin_critical_mud_fraction_low',[skin_critical_mud_fraction_low],[integer::])
  call gpu_put_r(gpu_unit,'native_skin_critical_mud_fraction_high',[skin_critical_mud_fraction_high],[integer::])
  call gpu_put_r(gpu_unit,'native_skin_fluff_fraction',[skin_fluff_fraction],[integer::])
  call gpu_put_i(gpu_unit,'native_q3c_forcing_verified',[merge(1,0,capacity_foundation_forcing_identity_declaration)],[integer::])
  call gpu_put_i(gpu_unit,'native_q3c_forcing_envelope',[54],[integer::])
  call gpu_put_i(gpu_unit,'native_q3c_formula_code',[bedlaw],[integer::])
  call gpu_put_i(gpu_unit,'native_q3c_method_id_ascii',reshape([(iachar(capacity_route_method_id(c:c)),c=1,64)],[size([(iachar(capacity_route_method_id(c:c)),c=1,64)])]),[64])
  call gpu_put_i(gpu_unit,'native_q3c_execution_fingerprint_ascii',reshape([(iachar(capacity_route_execution_fingerprint(c:c)),c=1,64)],[size([(iachar(capacity_route_execution_fingerprint(c:c)),c=1,64)])]),[64])
  call gpu_put_i(gpu_unit,'native_q3c_profile_fingerprint_ascii',reshape([(iachar(capacity_route_profile_fingerprint(c:c)),c=1,64)],[size([(iachar(capacity_route_profile_fingerprint(c:c)),c=1,64)])]),[64])
  call gpu_put_i(gpu_unit,'native_q3c_forcing_fingerprint_ascii',reshape([(iachar(capacity_foundation_forcing_fingerprint(c:c)),c=1,64)],[size([(iachar(capacity_foundation_forcing_fingerprint(c:c)),c=1,64)])]),[64])
  call gpu_put_i(gpu_unit,'native_q3c_raw_roughness_sha256_ascii',reshape([(iachar(capacity_foundation_forcing_raw_sha256(c:c)),c=1,64)],[size([(iachar(capacity_foundation_forcing_raw_sha256(c:c)),c=1,64)])]),[64])
  call gpu_put_i(gpu_unit,'native_q3h1_identity_verified',[merge(1,0,hiding_foundation_identity_declaration)],[integer::])
  call gpu_put_i(gpu_unit,'native_q3h1_identity_envelope',[55],[integer::])
  call gpu_put_i(gpu_unit,'native_q3h1_formula_code',[bedlaw],[integer::])
  call gpu_put_i(gpu_unit,'native_q3h1_hiding_mode',[hiding_exposure],[integer::])
  call gpu_put_r(gpu_unit,'native_q3h1_parker_alpha',[hiding_parker_alpha],[integer::])
  call gpu_put_r(gpu_unit,'native_q3h1_wu_exponent',[hiding_wu_exponent],[integer::])
  call gpu_put_i(gpu_unit,'native_q3h1_execution_fingerprint_ascii',reshape([(iachar(hiding_foundation_execution_fingerprint(c:c)),c=1,64)],[size([(iachar(hiding_foundation_execution_fingerprint(c:c)),c=1,64)])]),[64])
  call gpu_put_i(gpu_unit,'native_q3h2_secondary_verified',[merge(1,0,secondary_foundation_identity_declaration)],[integer::])
  call gpu_put_i(gpu_unit,'native_q3h2_secondary_envelope',[56],[integer::])
  call gpu_put_i(gpu_unit,'native_q3h2_model',[1],[integer::])
  call gpu_put_i(gpu_unit,'native_q3h2_execution_fingerprint_ascii',reshape([(iachar(secondary_foundation_execution_fingerprint(c:c)),c=1,64)],[size([(iachar(secondary_foundation_execution_fingerprint(c:c)),c=1,64)])]),[64])
  call gpu_put_i(gpu_unit,'native_q3h2_capacity_execution_fingerprint_ascii',reshape([(iachar(capacity_route_execution_fingerprint(c:c)),c=1,64)],[size([(iachar(capacity_route_execution_fingerprint(c:c)),c=1,64)])]),[64])
  call gpu_put_i(gpu_unit,'native_q3h2_capacity_profile_fingerprint_ascii',reshape([(iachar(capacity_route_profile_fingerprint(c:c)),c=1,64)],[size([(iachar(capacity_route_profile_fingerprint(c:c)),c=1,64)])]),[64])
  call gpu_put_i(gpu_unit,'native_q3h2_hiding_execution_fingerprint_ascii',reshape([(iachar(hiding_foundation_execution_fingerprint(c:c)),c=1,64)],[size([(iachar(hiding_foundation_execution_fingerprint(c:c)),c=1,64)])]),[64])
  call gpu_put_i(gpu_unit,'native_q3h2_roughness_fingerprint_ascii',reshape([(iachar(capacity_foundation_forcing_fingerprint(c:c)),c=1,64)],[size([(iachar(capacity_foundation_forcing_fingerprint(c:c)),c=1,64)])]),[64])
  call gpu_put_i(gpu_unit,'native_q3h2_method_id_ascii',reshape([(iachar(gpu_secondary_method_id(c:c)),c=1,64)],[size([(iachar(gpu_secondary_method_id(c:c)),c=1,64)])]),[64])
  call gpu_put_i(gpu_unit,'native_q3h2_canonical_preimage_sha256_ascii',reshape([(iachar(gpu_secondary_preimage_sha(c:c)),c=1,64)],[size([(iachar(gpu_secondary_preimage_sha(c:c)),c=1,64)])]),[64])
  call gpu_finish(gpu_unit)
  if(horizontal_viscosity>0.and.horizontal_viscosity_geometry==1) then
   gpu_state=q;gpu_export_diffusion=.true.
   call diffuse_momentum(gpu_state,1.d-6)
   gpu_export_diffusion=.false.
  endif
 end subroutine

 subroutine normal_export_metadata()
  integer :: gpu_unit
  call gpu_open(gpu_unit,'normal_metadata.bin')
  call gpu_put_i(gpu_unit,'nb',[nb],[integer::])
  call gpu_put_i(gpu_unit,'nfrac',[nfrac],[integer::])
  call gpu_put_i(gpu_unit,'deck_version',[deck_version],[integer::])
  call gpu_put_i(gpu_unit,'nhydro_series',reshape(nhydro_series,[size(nhydro_series)]),[nb])
  call gpu_put_i(gpu_unit,'nseries',reshape(nseries,[size(nseries)]),[nb])
  call gpu_put_i(gpu_unit,'mode',reshape(mode,[size(mode)]),[nb])
  call gpu_put_i(gpu_unit,'section_control',reshape(section_control,[size(section_control)]),[nb])
  call gpu_put_i(gpu_unit,'nfraction_series',reshape(nfraction_series,[size(nfraction_series)]),[nfrac,nb])
  call gpu_put_i(gpu_unit,'native_section_seed_node',reshape(section_seed_node,[size(section_seed_node)]),[nb])
  call gpu_put_i(gpu_unit,'native_boundary_seed_position',reshape(boundary_seed_position,[size(boundary_seed_position)]),[nb])
  call gpu_put_r(gpu_unit,'section_energy_slope',reshape(section_energy_slope,[size(section_energy_slope)]),[nb])
  call gpu_put_r(gpu_unit,'hydro_series_time',reshape(hydro_series_time,[size(hydro_series_time)]),[size(hydro_series_time,1),nb])
  call gpu_put_r(gpu_unit,'hydro_series_q',reshape(hydro_series_q,[size(hydro_series_q)]),[size(hydro_series_q,1),nb])
  call gpu_put_r(gpu_unit,'hydro_series_stage',reshape(hydro_series_stage,[size(hydro_series_stage)]),[size(hydro_series_stage,1),nb])
  call gpu_put_r(gpu_unit,'series_time',reshape(series_time,[size(series_time)]),[size(series_time,1),nb])
  call gpu_put_r(gpu_unit,'series_concentration',reshape(series_concentration,[size(series_concentration)]),[size(series_concentration,1),nb])
  call gpu_put_r(gpu_unit,'series_bed_input',reshape(series_bed_input,[size(series_bed_input)]),[size(series_bed_input,1),nb])
  call gpu_put_r(gpu_unit,'fraction_series_time',reshape(fraction_series_time,[size(fraction_series_time)]),[size(fraction_series_time,1),nfrac,nb])
  call gpu_put_r(gpu_unit,'fraction_series_concentration',reshape(fraction_series_concentration,[size(fraction_series_concentration)]),[size(fraction_series_concentration,1),nfrac,nb])
  call gpu_put_r(gpu_unit,'fraction_series_bed_input',reshape(fraction_series_bed_input,[size(fraction_series_bed_input)]),[size(fraction_series_bed_input,1),nfrac,nb])
  call gpu_finish(gpu_unit)
 end subroutine

 subroutine integrated_export_avalanche_metadata()
  integer :: gpu_unit
  call gpu_open(gpu_unit,'avalanche_metadata.bin')
  call gpu_put_i(gpu_unit,'nc',[nc],[integer::])
  call gpu_put_i(gpu_unit,'nf',[nf],[integer::])
  call gpu_put_i(gpu_unit,'nfrac',[nfrac],[integer::])
  call gpu_put_i(gpu_unit,'nbed_layers',[nbed_layers],[integer::])
  call gpu_put_i(gpu_unit,'deck_version',[deck_version],[integer::])
  call gpu_put_i(gpu_unit,'general_avalanche',[merge(1,0,general_avalanche)],[integer::])
  call gpu_put_r(gpu_unit,'general_avalanche_wet_slope',[general_avalanche_wet_slope],[integer::])
  call gpu_put_r(gpu_unit,'general_avalanche_dry_slope',[general_avalanche_dry_slope],[integer::])
  call gpu_put_r(gpu_unit,'general_avalanche_time',[general_avalanche_time],[integer::])
  call gpu_put_r(gpu_unit,'general_avalanche_wet_depth_switch',[general_avalanche_wet_depth_switch],[integer::])
  call gpu_put_r(gpu_unit,'general_avalanche_relaxation_length',[general_avalanche_relaxation_length],[integer::])
  call gpu_finish(gpu_unit)
 end subroutine
end module

program run_cutcell
 use cutcell_solver
 implicit none
 character(1024) :: deck,restart,stop_file_setting
 character(32) :: progress_setting,checkpoint_setting,diagnostic_setting,adaptation_setting,profile_setting
 character(8) :: restart_integrator_magic
 integer :: restart_integrator=33
 character(64) :: restart_fingerprint
 real(8) :: dt,target,started,finished,knot,unclipped_dt,output_started,retry_scale
 real(8) :: checkpoint_published_time=-huge(1.d0)
 real(8) :: momentum_before(2),momentum_after(2)
 real(8), allocatable :: saved_signature(:)
 integer :: frame,unit,binary_unit=-1,section_unit=-1,ledger_unit=-1,secondary_unit=-1,gaia_secondary_unit=-1,secondary_budget_unit=-1, &
  secondary_momentum_unit=-1,bank_unit=-1,bank_front_unit=-1,toe_flux_unit=-1,adaptation_unit=-1,morfac_unit=-1, &
  bedform_unit=-1,gaia_fixed_unit=-1,split_capacity_unit=-1,split_boundary_unit=-1,c,b,attempt,ios, &
  channel_cell_unit=-1,channel_boundary_unit=-1,channel_face_unit=-1, &
  suspended_exchange_unit=-1,flow2dh_boundary_unit=-1,flow2dh_memory_unit=-1, &
  progress_steps=0,checkpoint_steps=0, &
  environment_status,restart_nc,tree_count, &
  checkpoint_published_steps=-1
 logical :: ok,scheduled_clip,section_output,initial_only,write_ascii_frames,write_binary_frames, &
  adaptation_diagnostics=.false.,stop_file_configured=.false.,cancelled_by_stop=.false.
 call get_command_argument(1,deck)
 if(len_trim(deck)==0) error stop 'usage: xbraids_cutcell case.data [checkpoint]'
 call load_case(trim(deck))
 if(capacity_route_identity_declaration) then
 write(*,'(a,a)') 'capacity route method: ',trim(capacity_route_method_id)
  write(*,'(a,es24.16)') 'capacity profile ACal: ',capacity_route_acal
  if(trim(capacity_route_method_id)=='capacity.mpm') &
   write(*,'(a)') 'capacity route native kernel: delft_mpm_fixed_0.047'
  write(*,'(a,a)') 'capacity execution fingerprint: ', &
   trim(capacity_route_execution_fingerprint)
  write(*,'(a,a)') 'capacity profile fingerprint: ', &
   trim(capacity_route_profile_fingerprint)
  if(capacity_foundation_forcing_identity_declaration) then
   write(*,'(a,i0)') 'capacity forcing native envelope: ',54
   write(*,'(a,a)') 'capacity forcing fingerprint: ', &
    trim(capacity_foundation_forcing_fingerprint)
   write(*,'(a,a)') 'capacity forcing raw FP64 SHA-256: ', &
    trim(capacity_foundation_forcing_raw_sha256)
  endif
  if(hiding_foundation_identity_declaration) then
   write(*,'(a,i0)') 'hiding foundation native envelope: ',55
   write(*,'(a,a)') 'hiding foundation execution fingerprint: ', &
    trim(hiding_foundation_execution_fingerprint)
   write(*,'(a,i0)') 'hiding foundation mode: ',hiding_exposure
  endif
  if(secondary_foundation_identity_declaration) then
   write(*,'(a,i0)') 'secondary foundation native envelope: ',56
   write(*,'(a,a)') 'secondary foundation execution fingerprint: ', &
    trim(secondary_foundation_execution_fingerprint)
   write(*,'(a,3(es24.16,1x))') 'secondary foundation Rmincf Chzmin Espir: ', &
    secondary_foundation_rmincf,secondary_foundation_chzmin, &
    secondary_foundation_espir
  endif
 endif
 if(channel_composition_declaration) then
  write(*,'(a,a)') 'channel composition scope: ',trim(channel_composition_scope)
  write(*,'(a,a)') 'channel composition semantic ID: ',trim(channel_composition_semantic)
  write(*,'(a,a)') 'channel composition profile fingerprint: ', &
   trim(channel_composition_profile_fingerprint)
  if(channel_native_identity_declaration) then
   write(*,'(a,i0)') 'channel composition native envelope: ',52
   write(*,'(a,a)') 'channel composition native preimage SHA-256: ', &
    trim(channel_native_preimage_sha256)
   write(*,'(a,a)') 'execution profile semantic ID: ',trim(execution_profile_semantic)
   write(*,'(a,a)') 'execution profile fingerprint: ',trim(execution_profile_fingerprint)
  endif
  if(component_native_identity_declaration) then
   write(*,'(a,i0)') 'component document native envelope: ',53
   do c=1,component_native_count
    write(*,'(a,a,1x,a)') 'component document SHA-256: ', &
     trim(component_native_documents(c)%component_id), &
     trim(component_native_documents(c)%document_sha256)
   enddo
  endif
 endif
 section_output=production_section_geometry
 ! Opt-in timing report for scalability investigations. This does not affect
 ! the solution, adaptive hierarchy, output cadence, or checkpoint contents.
 call get_environment_variable('BRAIDS_PROFILE',profile_setting,status=environment_status)
 if(environment_status==0.and.len_trim(profile_setting)>0) then
  if(trim(profile_setting)/='1') error stop 'BRAIDS_PROFILE must be 1 when set'
  profile_enabled=.true.
 endif
 call get_environment_variable('BRAIDS_AMR_DIAGNOSTICS',diagnostic_setting,status=environment_status)
 if(environment_status==0.and.len_trim(diagnostic_setting)>0) then
  if(trim(diagnostic_setting)/='1') error stop 'BRAIDS_AMR_DIAGNOSTICS must be 1 when set'
  if(.not.amr_enabled) error stop 'BRAIDS_AMR_DIAGNOSTICS requires AMR'
  amr_diagnostics=.true.
  open(newunit=amr_diagnostic_unit,file='amr_trigger_cells.txt',status='replace')
  write(amr_diagnostic_unit,'(a)') '# time_s regrid_index local_cell node level raw_level wanted_level cause_mask area_m2 x_m y_m'
  write(amr_diagnostic_unit,'(a)') '# cause bits: 0=min-level 1=wet-base 2=wet-dry 3=hydraulic 4=concentration 5=bank-bed-difference 6=composition 7=material-interface 8=open-boundary 9=wet-wall 10=bathymetry 11=bed-evolution 12=subcell-shoreline 13=coarsening-bed-veto 14=forecast-buffer 15=coarsen-delay 16=persistent-bank-front'
  open(newunit=amr_diagnostic_summary_unit,file='amr_trigger_summary.txt',status='replace')
  write(amr_diagnostic_summary_unit,'(a,i0,a)') &
   '# time_s regrid_index input_cells selected_counts_L0_to_L',amr_levels-1, &
   ' cause_count_0_to_16 cause_area_0_to_16'
  write(amr_diagnostic_summary_unit,'(a)') '# cause order: min-level wet-base wet-dry hydraulic concentration bank-bed-difference composition material-interface open-boundary wet-wall bathymetry bed-evolution subcell-shoreline coarsening-bed-veto forecast-buffer coarsen-delay persistent-bank-front'
 endif
 ! Optional observation only: no change to accepted steps or checkpoint state.
 call get_environment_variable('BRAIDS_PROGRESS_STEPS',progress_setting,status=environment_status)
 if(environment_status==0.and.len_trim(progress_setting)>0) then
  read(progress_setting,*,iostat=ios) progress_steps
  if(ios/=0.or.progress_steps<=0) error stop 'BRAIDS_PROGRESS_STEPS must be a positive integer'
 endif
 ! Long runs may refresh their restart state more frequently than their result
 ! frames. This bounds recoverable work without multiplying output volume.
 call get_environment_variable('BRAIDS_CHECKPOINT_STEPS',checkpoint_setting,status=environment_status)
 if(environment_status==0.and.len_trim(checkpoint_setting)>0) then
  read(checkpoint_setting,*,iostat=ios) checkpoint_steps
  if(ios/=0.or.checkpoint_steps<=0) error stop 'BRAIDS_CHECKPOINT_STEPS must be a positive integer'
 endif
 ! Cooperative CPU cancellation uses an absolute, launcher-owned marker.  It
 ! is observed only outside a trial step, after every accepted physical state,
 ! source ledger and morphology transfer have either all committed or none has.
 stop_file_setting=''
 call get_environment_variable('BRAIDS_STOP_FILE',stop_file_setting,status=environment_status)
 if(environment_status==-1) error stop 'BRAIDS_STOP_FILE exceeds the native path buffer'
 stop_file_configured=environment_status==0.and.len_trim(stop_file_setting)>0
 call get_command_argument(2,restart)
 initial_only=trim(restart)=='--initial-mesh'
 if(initial_only) restart=''
 if(len_trim(restart)>0) then
  call recover_checkpoint_replacement(trim(restart))
  open(newunit=unit,file=trim(restart),status='old',form='unformatted',access='stream')
  read(unit) restart_fingerprint,c,b
  if(restart_fingerprint/=fingerprint) error stop 'restart configuration/build fingerprint differs'
  restart_nc=c
  if((.not.amr_enabled.and.c/=nc).or.b/=nb) error stop 'restart mesh dimensions differ'
  read(unit) c
  if(c/=size(initial_signature)) error stop 'restart configuration dimensions differ'
  allocate(saved_signature(c));read(unit) saved_signature
  if(any(saved_signature/=initial_signature)) error stop 'restart resolved inputs differ'
  deallocate(saved_signature)
  if(amr_enabled) then
   read(unit) tree_count
   if(tree_count/=amr_nn.or.restart_nc<1.or.restart_nc>amr_nn) error stop 'restart adaptive hierarchy differs'
   read(unit) amr_mesh_id,amr_regrid_count,amr_next,amr_level_steps,at_delay,at_last_bed,at_q
   if(bank_failure) read(unit) at_bank_undercut,at_bank_reference_height,at_bank_toe_height, &
    at_bank_reference_length,at_bank_toe_cumulative,at_bank_collapse_cumulative,at_bank_collapse_count
   if(bank_front_tracking) read(unit) at_bank_front_status,at_bank_front_seed_face, &
    at_bank_front_solid_area,at_bank_front_base,at_bank_front_height,at_bank_front_removed_mass
   deallocate(at_leaf);allocate(at_leaf(restart_nc));read(unit) at_leaf
   if(any(at_leaf<1).or.any(at_leaf>amr_nn).or.amr_mesh_id<0.or.amr_regrid_count<1.or. &
    .not.ieee_is_finite(amr_next).or..not.all(ieee_is_finite(at_q)).or. &
    any(at_q(1,:)<0).or.any(at_q(4:nphysical_state,:)<0)) error stop 'invalid adaptive checkpoint'
   call validate_persistent_bank_front_tree_state()
   call amr_rebuild_mesh()
  endif
  read(unit) time,step_hint,steps,q,ledger
  if(prognostic_secondary_flow()) read(unit) secondary_initial_integral,secondary_source_cumulative, &
   secondary_momentum_initial,secondary_momentum_boundary_cumulative, &
   secondary_stress_boundary_cumulative,secondary_momentum_remainder_cumulative
  if(bank_failure.and..not.amr_enabled) read(unit) bank_undercut,bank_reference_height,bank_toe_height, &
   bank_reference_length,bank_toe_cumulative,bank_collapse_cumulative,bank_collapse_count
  if(bank_front_tracking.and..not.amr_enabled) read(unit) bank_front_status,bank_front_seed_face, &
   bank_front_solid_area,bank_front_base,bank_front_height,bank_front_removed_mass
   if(bank_toe_erosion_mode==2) then
    read(unit) physical_toe_flux_cumulative,physical_toe_flux_absolute_cumulative, &
     physical_toe_flux_bank_id,physical_toe_flux_toe_id
    call allocate_physical_toe_work()
    read(unit) physical_toe_cell,physical_toe_donor,physical_toe_face,physical_toe_status, &
     physical_candidate_bank,physical_candidate_toe,physical_is_toe_cell, &
     physical_is_toe_donor,physical_is_toe_face,physical_toe_distance
    if(any(physical_toe_cell<0).or.any(physical_toe_cell>nc).or. &
       any(physical_toe_donor<0).or.any(physical_toe_donor>nc).or. &
       any(physical_toe_face<0).or.any(physical_toe_face>nf).or. &
       any(physical_candidate_bank<0).or.any(physical_candidate_bank>nc).or. &
       any(physical_candidate_toe<0).or.any(physical_candidate_toe>nc).or. &
       .not.all(ieee_is_finite(physical_toe_distance)).or. &
       any(bank_erodible.and.physical_toe_status/=BANK_TOE_MAPPED).or. &
       any(physical_is_toe_face.neqv. &
           (physical_candidate_bank>0.and.physical_candidate_toe>0))) &
     error stop 'invalid physical bank-toe map checkpoint'
    bank_toe_mapping_ready=.true.
    call refresh_physical_toe_flux_metadata()
   endif
  if(exchange_mode==4) read(unit) suspended_exchange_cumulative
  if(flow2dh_boundary_diffusion_declaration) read(unit) &
   suspended_boundary_last_outflow,suspended_boundary_return_clock
  if(.not.all(ieee_is_finite(q)).or..not.all(ieee_is_finite(ledger)).or..not.ieee_is_finite(time)) &
   error stop 'nonfinite checkpoint'
  if(exchange_mode==4) then
   if(.not.all(ieee_is_finite(suspended_exchange_cumulative))) &
    error stop 'invalid suspended exchange ledger checkpoint'
  endif
  if(flow2dh_boundary_diffusion_declaration) then
   call validate_flow2dh_boundary_memory_checkpoint()
  endif
  if(bank_toe_erosion_mode==2) then
   if(.not.all(ieee_is_finite(physical_toe_flux_cumulative)).or. &
      .not.all(ieee_is_finite(physical_toe_flux_absolute_cumulative)).or. &
      any(physical_toe_flux_absolute_cumulative<0.d0).or. &
      any(physical_toe_flux_bank_id<0).or.any(physical_toe_flux_toe_id<0)) &
    error stop 'invalid physical bank-toe flux checkpoint'
  endif
  if(minval(q(1,:))<0.or.minval(q(4:nphysical_state,:))<0.or.time<0.or.time>endtime.or.steps<0 &
   .or..not.ieee_is_finite(step_hint).or.step_hint<=0.or.step_hint>dtmax) error stop 'invalid checkpoint state'
  read(unit,iostat=ios) restart_integrator_magic
  if(ios<0) then
   ! Pre-adoption checkpoint: never silently switch its integrator.
   restart_integrator=33
  else
   if(ios/=0.or.restart_integrator_magic/='BRDINT01') error stop 'invalid checkpoint integrator trailer'
   read(unit,iostat=ios) restart_integrator
   if(ios/=0.or.(restart_integrator/=33.and.restart_integrator/=43.and.restart_integrator/=6)) &
    error stop 'invalid checkpoint time integrator'
  endif
  close(unit)
  else if(amr_enabled) then
   call amr_regrid(.true.)
  endif
  if(bank_toe_erosion_mode==2.and..not.bank_toe_mapping_ready) call build_physical_toe_map(q)
  call validate_persistent_bank_front_state()
 if(prognostic_secondary_flow().and.len_trim(restart)==0) then
  secondary_initial_integral=0.d0;secondary_source_cumulative=0.d0
  do c=1,nc
   secondary_initial_integral=secondary_initial_integral+area(c)*q(secondary_state_index(),c)
  enddo
  call domain_momentum(q,secondary_momentum_initial)
  secondary_momentum_boundary_cumulative=0.d0;secondary_stress_boundary_cumulative=0.d0
  secondary_momentum_remainder_cumulative=0.d0;secondary_momentum_boundary_trial=0.d0
 endif
 ! Resolve the invariant suspended-transport switch only after a checkpoint
 ! has replaced the initial state and after the initial AMR hierarchy exists.
 ! Boundary series are inspected in full, so a later nonzero knot cannot be
 ! skipped by an initially clear-water stage.
 suspended_transport_active=exchange_mode>0.or.any(fraction_concentration/=0.d0).or. &
  any(series_concentration/=0.d0).or.any(fraction_series_concentration/=0.d0)
 if(.not.suspended_transport_active) then
  do b=1,nfrac
   if(any(q(suspended_index(b),:)/=0.d0)) then
    suspended_transport_active=.true.;exit
   endif
  enddo
 endif

 ! Resolve the guarded integrator only after input/restart state loading.
 call resolve_cpu_time_integrator(len_trim(restart)>0)
 if(len_trim(restart)>0.and.restart_integrator/=cpu_time_integrator) then
  error stop 'restart time integrator differs; resume using its original BRAIDS_TIME_INTEGRATOR'
 endif
 ! Qualified same-equation fixed-grid hydraulic workflow. Resolve after restart
 ! and inspect all forcing knots; dormant sediment inventories remain untouched.
 ! Keep the original implementation for coupled physics, AMR and small grids.
 hydraulic_workflow_fast=nc>=4096.and.scheme==6.and..not.amr_enabled.and. &
  bedlaw==0.and.exchange_mode==0.and..not.suspended_transport_active.and. &
  .not.secondary_flow.and..not.bank_failure.and..not.bank_avalanche.and. &
  .not.general_avalanche.and..not.bedform_roughness.and..not.bedload_adaptation.and. &
  bank_erosion_fraction==0.d0.and.bed_slope==0.and.horizontal_viscosity==0.d0.and. &
  nstate==5.and.nfrac==1.and.all(ws_fraction==0.d0).and. &
  all(forcing(3:4,:)==0.d0).and.all(series_concentration==0.d0).and. &
  all(series_bed_input==0.d0).and.all(fraction_concentration==0.d0).and. &
  all(fraction_bed_input==0.d0).and.all(fraction_series_concentration==0.d0).and. &
  all(fraction_series_bed_input==0.d0)
 call get_environment_variable('BRAIDS_ADAPTATION_DIAGNOSTICS',adaptation_setting,status=environment_status)
 if(environment_status==0.and.len_trim(adaptation_setting)>0) then
  if(trim(adaptation_setting)/='1') error stop 'BRAIDS_ADAPTATION_DIAGNOSTICS must be 1 when set'
  if(.not.bedload_adaptation) error stop 'BRAIDS_ADAPTATION_DIAGNOSTICS requires bed-load adaptation'
  adaptation_diagnostics=.true.
 endif
 if(.not.component_native_identity_declaration.or. &
    .not.capacity_foundation_runtime_declaration.or. &
    .not.capacity_foundation_route_identity_declaration.or. &
    .not.capacity_foundation_forcing_identity_declaration.or. &
    .not.secondary_foundation_identity_declaration.or. &
    all(bedlaw/=[braids_capacity_mpm,braids_capacity_engelund_hansen, &
                 braids_capacity_general]).or. &
    secondary_flow.or.bed_slope/=0.or.exchange_mode/=0) &
  error stop 'v17 exporter requires an exact Q3-H2 identity over Q3-C/Q3-H1'
 call gpu_export_input()
 call normal_export_metadata()
 if(general_avalanche) call integrated_export_avalanche_metadata()
 stop 'GPU native initialization exported; no solver timestep performed'
 call cpu_time(started)
 if(profile_enabled) then
  ! Initial hierarchy construction is intentionally outside the measured run,
  ! just like the existing CPU-seconds report. Clear its opt-in observations.
  profile_started=omp_get_wtime();profile_regrid_s=0;profile_prepare_s=0;profile_euler_s=0;profile_source_s=0
  profile_euler_setup_s=0;profile_euler_flux_s=0;profile_euler_limit_s=0;profile_euler_finalize_s=0
  profile_euler_boundary_s=0;profile_euler_bedload_s=0
  profile_euler_hiding_s=0;profile_euler_mixture_s=0
  profile_euler_capacity_cell_s=0;profile_euler_capacity_gradient_s=0
  profile_euler_reconstruction_s=0;profile_euler_trace_s=0
  profile_output_s=0;profile_output_authoritative_s=0;profile_output_preview_pack_s=0
  profile_output_preview_io_s=0;profile_output_checkpoint_s=0
  profile_regrid_calls=0;profile_prepare_calls=0;profile_euler_calls=0;profile_source_calls=0;profile_output_calls=0
  profile_hydraulic_workflow_calls=0
 endif
 ! Variable-topology AMR frames remain human-readable because every frame has
 ! a different leaf list. Fixed-grid state is large and rectangular: write it
 ! as one portable little-endian stream so output is bounded by memory/disk
 ! bandwidth instead of millions of formatted number conversions. A tiny
 ! frames.txt marker keeps old completion checks and directs current readers.
 write_binary_frames=.not.amr_enabled.and.nc>4096
 write_ascii_frames=.not.write_binary_frames
 if(write_ascii_frames) then
   open(newunit=unit,file='frames.txt',status='replace')
   if(amr_enabled) then
    write(unit,*) -22,nb,nfrac,nbed_layers
   else if(nbed_layers==1.and.nfrac==1) then
    write(unit,*) nc,nb
   else if(nbed_layers==1) then
    write(unit,*) nc,nb,nfrac
   else
    write(unit,*) nc,nb,nfrac,nbed_layers
   endif
  else
   open(newunit=unit,file='frames.txt',status='replace')
   write(unit,'(a)') '-23 1'
   close(unit)
   open(newunit=binary_unit,file='frames.bin',status='replace',form='unformatted',access='stream',action='write')
   write(binary_unit) 'BRDFRM01'
   write(binary_unit) int([1,nc,nb,nfrac,nbed_layers,nphysical_state,nphysical_ledger],int32)
   flush(binary_unit)
  endif
  open(newunit=ledger_unit,file='boundary_ledger.txt',status='replace')
  write(ledger_unit,*) nb,nphysical_ledger
 if(bedlaw==9.and.allocated(fixed_boundary_bed_mode)) then
  if(any(fixed_boundary_bed_mode==1)) then
   open(newunit=gaia_fixed_unit,file='gaia_fixed_inlet_flux_native.txt',status='replace')
   write(gaia_fixed_unit,'(a)') &
    '# time_s boundary_id_one_based aggregate_bedload_flux_kg_s_positive_into_domain owner_bed_max_abs_change_m'
   block
    integer :: raw_unit
    open(newunit=raw_unit,file='gaia_fixed_boundary_bed.f64',status='replace', &
     form='unformatted',access='stream',action='write',convert='little_endian')
    write(raw_unit) fixed_boundary_bed
    close(raw_unit)
   end block
  endif
 endif
 if(section_output) then
  open(newunit=section_unit,file='boundary_sections.txt',status='replace')
  write(section_unit,'(a)') 'time_s boundary_id control discharge_m3_s stage_m area_m2 top_width_m conveyance_m3_s_per_sqrt_slope'
 endif
 if(secondary_flow.or.classic_secondary_foundation()) then
  open(newunit=secondary_unit,file='secondary_flow.txt',status='replace')
  write(secondary_unit,'(a)') '# time_s cell_id fraction_id x_m y_m curvature_1_m spiral_intensity_m_s alpha bedload_x_kg_m_s bedload_y_kg_m_s'
  if(classic_secondary_foundation()) then
   write(secondary_unit,'(a)') '# secondary_foundation_identity_envelope 56'
   write(secondary_unit,'(a)') '# secondary_foundation_execution_fingerprint '// &
    trim(secondary_foundation_execution_fingerprint)
   write(secondary_unit,'(a)') &
    '# Q3-H2 Equili=Y: refreshed every transactional trial stage; fixed Cartesian LS curvature and typed raw Chezy rotation'
  endif
  if(secondary_flow_model==3) then
   write(secondary_unit,'(a)') &
    '# model 3 does not use curvature, spiral intensity, or legacy alpha; those three columns are not applicable and remain zero; see gaia_secondary_stress.txt'
   open(newunit=gaia_secondary_unit,file='gaia_secondary_stress.txt',status='replace')
   write(gaia_secondary_unit,'(a)') '# BRAIDS_GAIA_SECONDARY_STRESS_V1'
   write(gaia_secondary_unit,'(a)') &
    '# time_s cell_id fraction_id x_m y_m active eta_grad_x eta_grad_y uphill_bed_grad_x uphill_bed_grad_y tau_b_Pa slope_stress_x_Pa slope_stress_y_Pa secondary_delta_x_Pa secondary_delta_y_Pa total_stress_x_Pa total_stress_y_Pa secondary_turn_rad total_turn_from_flow_rad base_capacity_x_kg_m_s base_capacity_y_kg_m_s corrected_capacity_x_kg_m_s corrected_capacity_y_kg_m_s slope_magnitude_factor'
   write(gaia_secondary_unit,'(a)') &
    '# eta_grad=grad(z+h); uphill_bed_grad=grad(z); slope_stress is after Talmon/SLOPEFF and before secondary stress; secondary_delta=total-slope; signed angles are positive counter-clockwise in [-pi,pi]; capacities are cell-centred equilibrium capacities, not accepted face fluxes'
   write(gaia_secondary_unit,'(a,3(1x,i0))') &
    '# v44_process_flags interior_slope_talmon interior_secondary_stress open_boundary_secondary_stress', &
    merge(1,0,gaia_interior_slope_talmon),merge(1,0,gaia_interior_secondary_stress), &
    merge(1,0,gaia_open_boundary_secondary_stress)
  endif
 endif
  if(prognostic_secondary_flow()) then
   open(newunit=secondary_budget_unit,file='secondary_flow_budget.txt',status='replace')
  write(secondary_budget_unit,'(a)') &
    '# time_s integral_hI_m3_s initial_hI_m3_s boundary_hI_m3_s source_hI_m3_s residual_hI_m3_s'
   open(newunit=secondary_momentum_unit,file='secondary_momentum_budget.txt',status='replace')
   write(secondary_momentum_unit,'(a)') &
    '# time_s current_x_m4_s current_y_m4_s initial_x_m4_s initial_y_m4_s ordinary_boundary_x_m4_s ordinary_boundary_y_m4_s secondary_stress_boundary_x_m4_s secondary_stress_boundary_y_m4_s interior_source_reconciliation_x_m4_s interior_source_reconciliation_y_m4_s residual_x_m4_s residual_y_m4_s'
   write(secondary_momentum_unit,'(a)') &
    '# per-unit-density accounting; remainder includes bed/pressure, friction, wet-dry, diffusion, AMR/regrid, bank coupling, and accepted numerical reconciliation not represented by boundary traction'
 endif
  if(bank_failure) then
   open(newunit=bank_unit,file='bank_failure.txt',status='replace')
   write(bank_unit,'(a)') '# time_s cell_id tree_node x_m y_m undercut_m reference_height_m toe_height_m toe_eroded_volume_m3 collapse_volume_m3 collapse_count'
  endif
  if(bank_front_tracking) then
   open(newunit=bank_front_unit,file='bank_fronts.txt',status='replace')
   write(bank_front_unit,'(a)') &
    '# time_s cell_id tree_node status seed_segment_id x_m y_m cell_area_m2 solid_area_m2 remaining_fraction retreat_fraction base_m reference_height_m raw_bed_m effective_bed_m cumulative_removed_volume_m3'
  endif
  if(bank_toe_erosion_mode==2) then
   open(newunit=toe_flux_unit,file='physical_bank_toe_bedload_flux.txt',status='replace')
   write(toe_flux_unit,'(a)') '# time_s segment_id local_face_id active bank_cell_or_node_id toe_cell_or_node_id midpoint_x_m midpoint_y_m length_m fraction_id cumulative_signed_bank_to_channel_kg cumulative_abs_accepted_step_net_kg'
   write(toe_flux_unit,'(a)') '# segment_id is immutable (AMR candidate-face ID when adaptive); gross is sum(abs(accepted SSPRK/reflux step-net mass)), not a cell-change reconstruction'
  endif
 if(adaptation_diagnostics) then
  open(newunit=adaptation_unit,file='bedload_adaptation.txt',status='replace')
  write(adaptation_unit,'(a)') '# time_s cell_id fraction_id x_m y_m length_m equilibrium_x_kg_m_s equilibrium_y_kg_m_s adapted_x_kg_m_s adapted_y_kg_m_s iterations residual'
 endif
 if(morphological_acceleration) then
  open(newunit=morfac_unit,file='morphological_acceleration.txt',status='replace')
  write(morfac_unit,'(a)') '# hydraulic_time_s morphological_time_s instantaneous_factor'
 endif
 if(bedform_roughness) then
  open(newunit=bedform_unit,file='bedform_roughness.txt',status='replace')
  write(bedform_unit,'(a)') '# time_s cell_id x_m y_m effective_manning_n dune_height_m dune_length_m nikuradse_ks_m iterations'
 endif
 if(capacity_split_retained) then
  open(newunit=split_capacity_unit,file='capacity_split.txt',status='replace')
  if(typed_constitutive_declaration) then
   write(split_capacity_unit,'(a)') '# BRAIDS_DELFT3D_CURRENT_CAPACITY_SPLIT_V2'
   write(split_capacity_unit,'(a)') '# diagnostic_schema_sha256 '//trim(capacity_diagnostic_schema)
   write(split_capacity_unit,'(a)') &
    '# time_s cell_id fraction_id x_m y_m bed_x_kg_m_s bed_y_kg_m_s suspended_x_kg_m_s suspended_y_kg_m_s absolute_shields critical_shields stress_ratio transport_stage applicability_mask'
  else
   write(split_capacity_unit,'(a)') '# BRAIDS_DELFT3D_CURRENT_CAPACITY_SPLIT_V1'
   write(split_capacity_unit,'(a)') &
    '# time_s cell_id fraction_id x_m y_m bed_x_kg_m_s bed_y_kg_m_s suspended_x_kg_m_s suspended_y_kg_m_s absolute_shields critical_shields stress_ratio transport_stage'
  endif
  write(split_capacity_unit,'(a)') &
   '# Both formula channels are retained; exchange mode 4 connects law-11 suspended capacity, otherwise it remains diagnostic.'
  open(newunit=split_boundary_unit,file='capacity_split_boundary.txt',status='replace')
  if(typed_constitutive_declaration) then
   write(split_boundary_unit,'(a)') '# BRAIDS_DELFT3D_CURRENT_CAPACITY_BOUNDARY_SPLIT_V2'
   write(split_boundary_unit,'(a)') '# diagnostic_schema_sha256 '//trim(capacity_diagnostic_schema)
   write(split_boundary_unit,'(a)') &
    '# time_s face_id boundary_id owner_cell_id fraction_id role normal_x normal_y bed_x_kg_m_s bed_y_kg_m_s suspended_x_kg_m_s suspended_y_kg_m_s absolute_shields critical_shields stress_ratio transport_stage applicability_mask'
  else
   write(split_boundary_unit,'(a)') '# BRAIDS_DELFT3D_CURRENT_CAPACITY_BOUNDARY_SPLIT_V1'
   write(split_boundary_unit,'(a)') &
    '# time_s face_id boundary_id owner_cell_id fraction_id role normal_x normal_y bed_x_kg_m_s bed_y_kg_m_s suspended_x_kg_m_s suspended_y_kg_m_s absolute_shields critical_shields stress_ratio transport_stage'
  endif
  write(split_boundary_unit,'(a)') &
   '# role: 0=reversal_or_no_capacity_transfer, 1=equilibrium_inlet_trace, 2=stage_outlet_trace, 3=owner_capacity_outlet. Vectors are retained constitutive channels before bed inventory limiting; suspended is diagnostic until M2.'
 endif
 if(channel_composition_declaration) then
  open(newunit=channel_cell_unit,file='channel_composition.txt',status='replace')
  write(channel_cell_unit,'(a)') '# BRAIDS_DELFT3D_CHANNEL_COMPOSITION_V1'
  write(channel_cell_unit,'(a)') '# scope '//trim(channel_composition_scope)
  write(channel_cell_unit,'(a)') '# semantic_id '//trim(channel_composition_semantic)
  write(channel_cell_unit,'(a)') '# profile_fingerprint '// &
   trim(channel_composition_profile_fingerprint)
  if(channel_native_identity_declaration) then
   write(channel_cell_unit,'(a)') '# native_identity_envelope 52'
   write(channel_cell_unit,'(a)') '# canonical_preimage_sha256 '// &
    trim(channel_native_preimage_sha256)
   write(channel_cell_unit,'(a)') '# execution_profile_semantic_id '// &
    trim(execution_profile_semantic)
   write(channel_cell_unit,'(a)') '# execution_profile_fingerprint '// &
    trim(execution_profile_fingerprint)
    if(component_native_identity_declaration) then
     write(channel_cell_unit,'(a)') '# component_identity_envelope 53'
     do c=1,component_native_count
      write(channel_cell_unit,'(a)') '# component_document_sha256 '// &
       trim(component_native_documents(c)%component_id)//' '// &
       trim(component_native_documents(c)%document_sha256)
     enddo
     if(capacity_foundation_forcing_identity_declaration) then
      write(channel_cell_unit,'(a)') '# capacity_forcing_identity_envelope 54'
      write(channel_cell_unit,'(a)') '# capacity_forcing_fingerprint '// &
       trim(capacity_foundation_forcing_fingerprint)
      write(channel_cell_unit,'(a)') '# capacity_forcing_raw_sha256 '// &
       trim(capacity_foundation_forcing_raw_sha256)
      write(channel_cell_unit,'(a)') '# capacity_foundation_execution_fingerprint '// &
       trim(capacity_foundation_native_execution_fingerprint)
      if(hiding_foundation_identity_declaration) then
       write(channel_cell_unit,'(a)') '# hiding_foundation_identity_envelope 55'
       write(channel_cell_unit,'(a)') '# hiding_foundation_execution_fingerprint '// &
        trim(hiding_foundation_execution_fingerprint)
      endif
      if(secondary_foundation_identity_declaration) then
       write(channel_cell_unit,'(a)') '# secondary_foundation_identity_envelope 56'
       write(channel_cell_unit,'(a)') '# secondary_foundation_execution_fingerprint '// &
        trim(secondary_foundation_execution_fingerprint)
      endif
     endif
    endif
   endif
  write(channel_cell_unit,'(a)') &
   '# time_s cell_id fraction_id x_m y_m raw_bed_x raw_bed_y raw_sus_x raw_sus_y composed_bed_x composed_bed_y retained_sus_x retained_sus_y signed_sus_rate equilibrium_mass_concentration'
  open(newunit=channel_boundary_unit,file='channel_composition_boundary.txt',status='replace')
  write(channel_boundary_unit,'(a)') '# BRAIDS_DELFT3D_CHANNEL_COMPOSITION_BOUNDARY_V1'
  write(channel_boundary_unit,'(a)') '# scope '//trim(channel_composition_scope)
  write(channel_boundary_unit,'(a)') '# semantic_id '//trim(channel_composition_semantic)
  write(channel_boundary_unit,'(a)') '# profile_fingerprint '// &
   trim(channel_composition_profile_fingerprint)
  if(channel_native_identity_declaration) then
   write(channel_boundary_unit,'(a)') '# native_identity_envelope 52'
   write(channel_boundary_unit,'(a)') '# canonical_preimage_sha256 '// &
    trim(channel_native_preimage_sha256)
   write(channel_boundary_unit,'(a)') '# execution_profile_semantic_id '// &
    trim(execution_profile_semantic)
   write(channel_boundary_unit,'(a)') '# execution_profile_fingerprint '// &
    trim(execution_profile_fingerprint)
    if(component_native_identity_declaration) then
     write(channel_boundary_unit,'(a)') '# component_identity_envelope 53'
     do c=1,component_native_count
      write(channel_boundary_unit,'(a)') '# component_document_sha256 '// &
       trim(component_native_documents(c)%component_id)//' '// &
       trim(component_native_documents(c)%document_sha256)
     enddo
     if(capacity_foundation_forcing_identity_declaration) then
      write(channel_boundary_unit,'(a)') '# capacity_forcing_identity_envelope 54'
      write(channel_boundary_unit,'(a)') '# capacity_forcing_fingerprint '// &
       trim(capacity_foundation_forcing_fingerprint)
      write(channel_boundary_unit,'(a)') '# capacity_forcing_raw_sha256 '// &
       trim(capacity_foundation_forcing_raw_sha256)
      write(channel_boundary_unit,'(a)') '# capacity_foundation_execution_fingerprint '// &
       trim(capacity_foundation_native_execution_fingerprint)
      if(hiding_foundation_identity_declaration) then
       write(channel_boundary_unit,'(a)') '# hiding_foundation_identity_envelope 55'
       write(channel_boundary_unit,'(a)') '# hiding_foundation_execution_fingerprint '// &
        trim(hiding_foundation_execution_fingerprint)
      endif
      if(secondary_foundation_identity_declaration) then
       write(channel_boundary_unit,'(a)') '# secondary_foundation_identity_envelope 56'
       write(channel_boundary_unit,'(a)') '# secondary_foundation_execution_fingerprint '// &
        trim(secondary_foundation_execution_fingerprint)
      endif
     endif
    endif
   endif
  write(channel_boundary_unit,'(a)') &
   '# time_s face_id boundary_id owner_cell_id fraction_id role raw_bed_x raw_bed_y raw_sus_x raw_sus_y composed_bed_x composed_bed_y retained_sus_x retained_sus_y signed_sus_rate equilibrium_mass_concentration'
  open(newunit=channel_face_unit,file='channel_composition_face.txt',status='replace')
  write(channel_face_unit,'(a)') '# BRAIDS_DELFT3D_CHANNEL_COMPOSITION_FACE_V1'
  write(channel_face_unit,'(a)') '# scope '//trim(channel_composition_scope)
  write(channel_face_unit,'(a)') '# semantic_id '//trim(channel_composition_semantic)
  write(channel_face_unit,'(a)') '# profile_fingerprint '// &
   trim(channel_composition_profile_fingerprint)
  if(channel_native_identity_declaration) then
   write(channel_face_unit,'(a)') '# native_identity_envelope 52'
   write(channel_face_unit,'(a)') '# canonical_preimage_sha256 '// &
    trim(channel_native_preimage_sha256)
   write(channel_face_unit,'(a)') '# execution_profile_semantic_id '// &
    trim(execution_profile_semantic)
   write(channel_face_unit,'(a)') '# execution_profile_fingerprint '// &
    trim(execution_profile_fingerprint)
    if(component_native_identity_declaration) then
     write(channel_face_unit,'(a)') '# component_identity_envelope 53'
     do c=1,component_native_count
      write(channel_face_unit,'(a)') '# component_document_sha256 '// &
       trim(component_native_documents(c)%component_id)//' '// &
       trim(component_native_documents(c)%document_sha256)
     enddo
     if(capacity_foundation_forcing_identity_declaration) then
      write(channel_face_unit,'(a)') '# capacity_forcing_identity_envelope 54'
      write(channel_face_unit,'(a)') '# capacity_forcing_fingerprint '// &
       trim(capacity_foundation_forcing_fingerprint)
      write(channel_face_unit,'(a)') '# capacity_forcing_raw_sha256 '// &
       trim(capacity_foundation_forcing_raw_sha256)
      write(channel_face_unit,'(a)') '# capacity_foundation_execution_fingerprint '// &
       trim(capacity_foundation_native_execution_fingerprint)
      if(hiding_foundation_identity_declaration) then
       write(channel_face_unit,'(a)') '# hiding_foundation_identity_envelope 55'
       write(channel_face_unit,'(a)') '# hiding_foundation_execution_fingerprint '// &
        trim(hiding_foundation_execution_fingerprint)
      endif
      if(secondary_foundation_identity_declaration) then
       write(channel_face_unit,'(a)') '# secondary_foundation_identity_envelope 56'
       write(channel_face_unit,'(a)') '# secondary_foundation_execution_fingerprint '// &
        trim(secondary_foundation_execution_fingerprint)
      endif
     endif
    endif
   endif
  write(channel_face_unit,'(a)') &
   '# time_s face_id owner_cell_id neighbour_cell_id boundary_id fraction_id normal_x normal_y length_m bed_mass_flux_kg_m_s suspended_advective_mass_flux_kg_m_s'
 endif
 if(exchange_mode==4) then
  open(newunit=suspended_exchange_unit,file='suspended_exchange_ledger.txt',status='replace')
  write(suspended_exchange_unit,'(a)') &
   '# time_s fraction_id cumulative_suspended_gain_kg cumulative_bed_gain_kg balance_residual_kg'
  write(suspended_exchange_unit,'(a)') &
   '# positive suspended gain means bed-to-water transfer; accepted source steps only; boundary transport remains in boundary_ledger.txt'
 endif
 if(flow2dh_boundary_diffusion_declaration) then
  open(newunit=flow2dh_boundary_unit,file='suspended_boundary_ledger.txt',status='replace')
  write(flow2dh_boundary_unit,'(a)') &
   '# time_s boundary_id fraction_id cumulative_advective_mass_in_kg cumulative_diffusive_mass_in_kg cumulative_total_mass_in_kg closure_residual_kg'
  write(flow2dh_boundary_unit,'(a)') &
   '# inward-positive; total is the public boundary_ledger suspended column; advective is accepted FCT and diffusive uses the accepted hydraulic face exchange area'
  open(newunit=flow2dh_memory_unit,file='suspended_boundary_memory.txt',status='replace')
  write(flow2dh_memory_unit,'(a)') &
   '# time_s stable_candidate_face_id boundary_id fraction_id last_adjacent_outflow_concentration_kg_m3 return_clock_s'
  write(flow2dh_memory_unit,'(a)') &
   '# accepted macro-step state only; outflow stores adjacent pre-transport r0, return decrements once, exact signed integrated water transport selects direction'
 endif
 ! A capped prefix may end between scheduled outputs. Resume at the next
 ! future output, rather than rounding past it when time/output_dt > 0.5.
 frame=floor(time/output_dt+1.d-12)
 if(profile_enabled) output_started=omp_get_wtime()
 call output_frame()
 if(profile_enabled) then
  profile_output_s=profile_output_s+omp_get_wtime()-output_started;profile_output_calls=profile_output_calls+1
 endif
 simulation: do while(.not.initial_only.and.time<endtime-1.d-12)
  target=min(endtime,(frame+1)*output_dt)
  accepted_steps: do while(time<target-1.d-12)
   if(native_stop_requested()) then
    ! q, ledger and suspended_exchange_cumulative still describe one complete
    ! accepted-step boundary. Publish that transaction as the restart authority
    ! without advertising a partial result frame.
    if(checkpoint_published_steps/=steps.or.checkpoint_published_time/=time) &
     call write_checkpoint()
    cancelled_by_stop=.true.
    write(*,'(a,f14.6,a,i0)') 'cancelled: accepted time=',time,' steps=',steps
    flush(6)
    exit accepted_steps
   endif
   if(prognostic_secondary_flow()) call domain_momentum(q,momentum_before)
   unclipped_dt=step_hint;dt=min(unclipped_dt,target-time)
   knot=next_boundary_knot(time)
   if(knot<huge(1.d0)) dt=min(dt,knot-time)
   knot=next_morphological_knot(time)
   if(knot<huge(1.d0)) dt=min(dt,knot-time)
   if(amr_enabled) dt=min(dt,amr_next-time)
   scheduled_clip=dt<unclipped_dt
   do attempt=1,35
    if(time+dt<=time) then
     write(*,*) 'non-advancing timestep at time, step, reason:',time,steps,failure_reason
     if(failure_reason==8) write(*,*) 'boundary trace diagnostic:',failed_trace_boundary,failed_trace_face, &
      failed_trace_condition,failed_trace_values
     error stop 'timestep reached machine precision; boundary/state transition is unresolved'
    endif
    if(amr_enabled.and.amr_subcycling) then
     if(amr_requires_synchronous_wet_front(time)) then
      call advance(dt,ok)
     else
      call amr_advance_subcycled(dt,ok)
     endif
    else
     call advance(dt,ok)
    endif
    if(ok) exit
    if(progress_steps==1) then
     write(*,'(a,es16.8,a,es16.8,a,i0,a,i0)') 'rejected: time=',time,' dt=',dt, &
      ' reason=',failure_reason,' AMR_level=',amr_failed_level
     if(amr_failure_cell>0) then
      if(amr_enabled) then
       write(*,'(a,i0,a,i0,a,es24.16,a,es24.16,a,2es24.16,a,es24.16,a,i0)') &
        'AMR failure cell=',amr_failure_cell,' state=',amr_failure_state,' value=',amr_failure_value, &
        ' area=',area(amr_failure_cell),' xy=',xy(:,amr_failure_cell),' auxiliary=',amr_failure_aux, &
        ' node=',at_leaf(amr_failure_cell)
      else
       write(*,'(a,i0,a,i0,a,es24.16,a,es24.16,a,2es24.16,a,es24.16)') &
        'failure cell=',amr_failure_cell,' state=',amr_failure_state,' value=',amr_failure_value, &
        ' area=',area(amr_failure_cell),' xy=',xy(:,amr_failure_cell),' auxiliary=',amr_failure_aux
      endif
     endif
     flush(6)
    endif
    retries=retries+1
    if(failure_reason==10.and.bank_front_retry_scale>0.d0.and.bank_front_retry_scale<1.d0) then
     dt=bank_front_retry_scale*dt
    else if(failure_reason==2.and..not.amr_enabled.and.amr_failure_value>cfl_target) then
     ! The rejected Euler stage has already measured its exact maximum CFL.
     ! Move directly to 95% of the admissible step rather than discarding half
     ! of it and spending the next fourteen accepted steps climbing back.  A
     ! conservative one-percent growth rate then tracks the slowly changing
     ! hydraulic limit without the former saw-tooth sequence of retries.
     retry_scale=max(.1d0,min(.95d0,.95d0*cfl_target/amr_failure_value))
     dt=retry_scale*dt
    else
     dt=.5d0*dt
    endif
   enddo
   if(ok.and.attempt==1.and.scheduled_clip) then
    ! Landing exactly on an output time or forcing knot is not a stability
    ! restriction. Preserve the physical step estimate for the next interval.
    step_hint=unclipped_dt
   else if(.not.amr_enabled.and.trial_cfl_peak>0) then
    ! Each accepted RK step already measured the limiting cell in all three
    ! stages.  Reuse that information to target 95% of the CFL contract on the
    ! next step, capped at the historical five-percent growth rate.
    retry_scale=max(.5d0,min(1.05d0,.95d0*cfl_target/trial_cfl_peak))
    step_hint=min(dtmax,retry_scale*dt)
   else
    step_hint=min(dtmax,1.05d0*dt)
   endif
   if(.not.ok) then
    write(*,*) 'failed at time, step, reason:',time,steps,failure_reason
    if(failure_reason==6) then
     write(*,*) 'inlet numeric ID, trace speed (m/s), limit (m/s):',failed_speed_boundary,failed_trace_speed,speed_limit
     error stop 'inlet trace exceeds speed limit; inspect represented opening, discharge and level'
    else if(failure_reason==7) then
     write(*,*) 'outlet numeric ID, trace speed (m/s), limit (m/s):',failed_speed_boundary,failed_trace_speed,speed_limit
     error stop 'outlet trace exceeds speed limit; inspect selected face, water level and control'
    else if(failure_reason==8) then
     write(*,*) 'inlet numeric ID, face, condition:',failed_trace_boundary,failed_trace_face,failed_trace_condition
    write(*,*) 'inlet trace values:',failed_trace_values
     error stop 'inlet trace is incompatible with the evolving local hydraulic regime'
    endif
    error stop 'cut-cell step failed after 35 halvings; inspect regime/positivity'
   endif
   if(exchange_mode==4) call commit_suspended_exchange()
   if(horizontal_viscosity>0) call diffuse_momentum(q,dt)
   if(amr_enabled) then
    if(time>=amr_next-1.d-12) call amr_regrid(.false.)
   endif
   if(prognostic_secondary_flow()) then
    call domain_momentum(q,momentum_after)
    secondary_momentum_boundary_cumulative=secondary_momentum_boundary_cumulative+ &
     secondary_momentum_boundary_trial(:,1)
    secondary_stress_boundary_cumulative=secondary_stress_boundary_cumulative+ &
     secondary_momentum_boundary_trial(:,2)
    secondary_momentum_remainder_cumulative=secondary_momentum_remainder_cumulative+ &
     momentum_after-momentum_before-secondary_momentum_boundary_trial(:,1)- &
     secondary_momentum_boundary_trial(:,2)
   endif
   if(checkpoint_steps>0) then
    if(mod(steps,checkpoint_steps)==0) call write_checkpoint()
   endif
   if(progress_steps>0) then
    if(mod(steps,progress_steps)==0) then
     write(*,'(a,f14.6,a,f14.6,a,i0,a,es12.4,a,i0)') &
      'progress: time=',time,' end=',endtime,' steps=',steps,' dt=',dt,' retries=',retries
     flush(6)
    endif
   endif
  enddo accepted_steps
  if(cancelled_by_stop) exit simulation
  frame=frame+1
  if(profile_enabled) output_started=omp_get_wtime()
  call output_frame()
  if(profile_enabled) then
   profile_output_s=profile_output_s+omp_get_wtime()-output_started;profile_output_calls=profile_output_calls+1
  endif
 enddo simulation
 if(write_ascii_frames) close(unit)
 if(write_binary_frames) close(binary_unit)
 close(ledger_unit)
 if(section_output) close(section_unit)
 if(secondary_unit/=-1) close(secondary_unit)
 if(gaia_secondary_unit/=-1) close(gaia_secondary_unit)
 if(secondary_budget_unit/=-1) close(secondary_budget_unit)
 if(secondary_momentum_unit/=-1) close(secondary_momentum_unit)
 if(bank_unit/=-1) close(bank_unit)
 if(bank_front_unit/=-1) close(bank_front_unit)
 if(toe_flux_unit/=-1) close(toe_flux_unit)
 if(adaptation_unit/=-1) close(adaptation_unit)
 if(morfac_unit/=-1) close(morfac_unit)
 if(bedform_unit/=-1) close(bedform_unit)
 if(gaia_fixed_unit/=-1) close(gaia_fixed_unit)
 if(split_capacity_unit/=-1) close(split_capacity_unit)
 if(split_boundary_unit/=-1) close(split_boundary_unit)
 if(channel_cell_unit/=-1) close(channel_cell_unit)
 if(channel_boundary_unit/=-1) close(channel_boundary_unit)
 if(channel_face_unit/=-1) close(channel_face_unit)
 if(suspended_exchange_unit/=-1) close(suspended_exchange_unit)
 if(flow2dh_boundary_unit/=-1) close(flow2dh_boundary_unit)
 if(flow2dh_memory_unit/=-1) close(flow2dh_memory_unit)
 call cpu_time(finished)
 write(*,*) 'steps, rejected trials, CPU seconds:',steps,retries,finished-started
 call profile_report()
 if(amr_enabled) then
  write(*,*) 'AMR meshes, regrid evaluations, active leaves:',amr_mesh_id+1,amr_regrid_count,nc
  if(amr_subcycling) write(*,*) 'AMR accepted steps by level L0 through finest:', &
   amr_level_steps(0:amr_levels-1)
 endif
 if(amr_diagnostic_unit/=-1) close(amr_diagnostic_unit)
 if(amr_diagnostic_summary_unit/=-1) close(amr_diagnostic_summary_unit)
contains
 logical function native_stop_requested() result(requested)
  logical :: exists
  requested=.false.
  if(.not.stop_file_configured) return
  inquire(file=trim(stop_file_setting),exist=exists)
  requested=exists
 end function

 subroutine remove_checkpoint_file(path,status)
  character(len=*),intent(in) :: path
  integer,intent(out) :: status
  integer :: local_unit,attempt
  logical :: exists
  status=0
  do attempt=1,20
   inquire(file=trim(path),exist=exists)
   if(.not.exists) then
    status=0
    return
   endif
   open(newunit=local_unit,file=trim(path),status='old',action='readwrite',iostat=status)
   if(status==0) close(local_unit,status='delete',iostat=status)
   if(status==0) return
   call checkpoint_retry_pause(attempt)
  enddo
 end subroutine

 subroutine checkpoint_retry_pause(attempt)
  integer,intent(in) :: attempt
  real(8) :: until,delay
  ! Windows virus scanners and file indexers may retain a just-closed handle
  ! briefly. Retry publication without ever deleting the accepted old
  ! checkpoint first. This path is output-only and does not alter numerics.
  delay=min(.05d0,.001d0*2.d0**min(attempt-1,6))
  until=omp_get_wtime()+delay
  do while(omp_get_wtime()<until)
  enddo
 end subroutine

 subroutine rename_checkpoint_with_retry(source,destination,status)
  character(len=*),intent(in) :: source,destination
  integer,intent(out) :: status
  integer :: attempt
  do attempt=1,20
   call rename(trim(source),trim(destination),status)
   if(status==0) return
   call checkpoint_retry_pause(attempt)
  enddo
 end subroutine

 subroutine recover_checkpoint_replacement(destination)
  character(len=*),intent(in) :: destination
  character(len=:),allocatable :: partial,backup
  integer :: status
  logical :: destination_exists,partial_exists,backup_exists
  partial=trim(destination)//'.partial'
  backup=trim(destination)//'.replace.backup'
  inquire(file=trim(destination),exist=destination_exists)
  inquire(file=backup,exist=backup_exists)
  if(backup_exists) then
   if(destination_exists) then
    call remove_checkpoint_file(backup,status)
    if(status/=0) error stop 'cannot remove stale checkpoint replacement backup'
   else
    call rename_checkpoint_with_retry(backup,trim(destination),status)
    if(status/=0) error stop 'cannot recover previous checkpoint replacement'
   endif
  endif
  inquire(file=partial,exist=partial_exists)
  if(partial_exists) then
   call remove_checkpoint_file(partial,status)
   if(status/=0) error stop 'cannot remove stale partial checkpoint'
  endif
 end subroutine

 subroutine replace_checkpoint_file(temporary,destination)
  character(len=*),intent(in) :: temporary,destination
  character(len=:),allocatable :: backup
  integer :: status,rollback_status
  logical :: destination_exists,backup_exists
  backup=trim(destination)//'.replace.backup'
  inquire(file=trim(destination),exist=destination_exists)
  if(destination_exists) then
   call rename_checkpoint_with_retry(trim(destination),backup,status)
   if(status/=0) error stop 'cannot preserve previous checkpoint during replacement'
  endif
  call rename_checkpoint_with_retry(trim(temporary),trim(destination),status)
  if(status/=0) then
   inquire(file=backup,exist=backup_exists)
   if(backup_exists) call rename_checkpoint_with_retry(backup,trim(destination),rollback_status)
   error stop 'cannot publish complete checkpoint replacement'
  endif
  call remove_checkpoint_file(backup,status)
  if(status/=0) error stop 'cannot remove checkpoint replacement backup'
 end subroutine

 subroutine write_checkpoint()
  ! Keep the most recent completed output state restartable. Large validation
  ! and production runs must not lose every accepted interval when a later
  ! boundary/regime transition fails. The stream is replaced only after the
  ! corresponding frame has been committed, so it never advertises a future
   ! time. Legacy checkpoint records stay byte-identical; v38 additionally
   ! retains its accepted physical-toe flux and exact geometry-derived map.
   integer :: checkpoint_unit,status_unit
   if(bank_toe_erosion_mode==2.and..not.bank_toe_mapping_ready) call build_physical_toe_map(q)
   call recover_checkpoint_replacement('checkpoint.bin')
   open(newunit=checkpoint_unit,file='checkpoint.bin.partial',status='replace',form='unformatted',access='stream')
  write(checkpoint_unit) fingerprint,nc,nb,size(initial_signature),initial_signature
  if(amr_enabled) then
   call amr_sync_state()
   call validate_persistent_bank_front_tree_state()
   write(checkpoint_unit) amr_nn
   write(checkpoint_unit) amr_mesh_id,amr_regrid_count,amr_next,amr_level_steps,at_delay,at_last_bed,at_q
   if(bank_failure) write(checkpoint_unit) at_bank_undercut,at_bank_reference_height,at_bank_toe_height, &
    at_bank_reference_length,at_bank_toe_cumulative,at_bank_collapse_cumulative,at_bank_collapse_count
   if(bank_front_tracking) write(checkpoint_unit) at_bank_front_status,at_bank_front_seed_face, &
    at_bank_front_solid_area,at_bank_front_base,at_bank_front_height,at_bank_front_removed_mass
   write(checkpoint_unit) at_leaf
  endif
  write(checkpoint_unit) time,step_hint,steps,q,ledger
  if(prognostic_secondary_flow()) write(checkpoint_unit) secondary_initial_integral,secondary_source_cumulative, &
   secondary_momentum_initial,secondary_momentum_boundary_cumulative, &
   secondary_stress_boundary_cumulative,secondary_momentum_remainder_cumulative
  if(bank_failure.and..not.amr_enabled) write(checkpoint_unit) bank_undercut,bank_reference_height,bank_toe_height, &
   bank_reference_length,bank_toe_cumulative,bank_collapse_cumulative,bank_collapse_count
  if(bank_front_tracking.and..not.amr_enabled) write(checkpoint_unit) bank_front_status,bank_front_seed_face, &
   bank_front_solid_area,bank_front_base,bank_front_height,bank_front_removed_mass
   if(bank_toe_erosion_mode==2) then
    write(checkpoint_unit) physical_toe_flux_cumulative,physical_toe_flux_absolute_cumulative, &
     physical_toe_flux_bank_id,physical_toe_flux_toe_id
    write(checkpoint_unit) physical_toe_cell,physical_toe_donor,physical_toe_face,physical_toe_status, &
     physical_candidate_bank,physical_candidate_toe,physical_is_toe_cell, &
     physical_is_toe_donor,physical_is_toe_face,physical_toe_distance
   endif
  if(exchange_mode==4) write(checkpoint_unit) suspended_exchange_cumulative
  if(flow2dh_boundary_diffusion_declaration) write(checkpoint_unit) &
   suspended_boundary_last_outflow,suspended_boundary_return_clock
  write(checkpoint_unit) 'BRDINT01',cpu_time_integrator
  flush(checkpoint_unit)
  close(checkpoint_unit)
  call replace_checkpoint_file('checkpoint.bin.partial','checkpoint.bin')
  ! Human- and monitor-readable companion; the binary checkpoint remains the
  ! sole restart authority if interruption occurs between these two writes.
  call recover_checkpoint_replacement('checkpoint_status.txt')
  open(newunit=status_unit,file='checkpoint_status.txt.partial',status='replace')
  write(status_unit,'(es24.16,1x,i0)') time,steps
  flush(status_unit)
  close(status_unit)
  call replace_checkpoint_file('checkpoint_status.txt.partial','checkpoint_status.txt')
  checkpoint_published_time=time;checkpoint_published_steps=steps
 end subroutine

 subroutine output_frame()
  ! Fixed-grid runs use the native binary preview stream below.  Keeping an
  ! ASCII copy of every cell state made output and later result loading scale
  ! with text formatting/parsing rather than with the actual simulation.
  real(8) :: output_phase_started
   if(profile_enabled) output_phase_started=omp_get_wtime()
   if(write_ascii_frames) then
    if(amr_enabled) then
     write(unit,*) time,steps,nc,amr_mesh_id
     do c=1,nc
      write(unit,*) at_leaf(c)-1,q(1:nphysical_state,c)
     enddo
    else
     write(unit,*) time,steps
     do c=1,nc
      write(unit,*) q(1:nphysical_state,c)
     enddo
    endif
    do b=1,nb
     write(unit,*) ledger(1:nphysical_ledger,b)
    enddo
    flush(unit)
   endif
   if(write_binary_frames) then
    write(binary_unit) time,int(steps,int64),q(1:nphysical_state,:),ledger(1:nphysical_ledger,:)
    flush(binary_unit)
   endif
   write(ledger_unit,*) time
   do b=1,nb
    write(ledger_unit,*) ledger(1:nphysical_ledger,b)
   enddo
   flush(ledger_unit)
  if(profile_enabled) profile_output_authoritative_s=profile_output_authoritative_s+omp_get_wtime()-output_phase_started
  call output_live_preview()
  if(split_capacity_unit/=-1) call output_split_capacity_frame()
  if(channel_cell_unit/=-1) call output_channel_composition_frame()
  if(suspended_exchange_unit/=-1) call output_suspended_exchange_frame()
  if(flow2dh_boundary_unit/=-1) call output_flow2dh_boundary_frame()
  if(flow2dh_memory_unit/=-1) call output_flow2dh_memory_frame()
  if(section_output) call output_section_frame()
  if(profile_enabled) output_phase_started=omp_get_wtime()
  call write_checkpoint()
  if(profile_enabled) profile_output_checkpoint_s=profile_output_checkpoint_s+omp_get_wtime()-output_phase_started
  if(secondary_flow.or.classic_secondary_foundation()) call output_secondary_flow_frame()
  if(gaia_fixed_unit/=-1) call output_gaia_fixed_inlet_flux_frame()
  if(prognostic_secondary_flow()) call output_secondary_budget_frame()
  if(prognostic_secondary_flow()) call output_secondary_momentum_budget_frame()
  if(bank_failure) call output_bank_failure_frame()
  if(bank_front_tracking) call output_bank_front_frame()
  if(bank_toe_erosion_mode==2) call output_physical_toe_flux_frame()
   if(adaptation_diagnostics) call output_adaptation_frame()
  if(morfac_unit/=-1) then
    write(morfac_unit,*) time,morphological_time_at(time),morphological_factor_at(time)
    flush(morfac_unit)
  endif
   if(bedform_unit/=-1) call output_bedform_roughness_frame()
 end subroutine

 subroutine output_channel_composition_frame()
  integer :: local_cell,local_fraction,local_face
  ! `output_split_capacity_frame` immediately above has just re-entered the
  ! authoritative zero-dt Euler path, so these scratch arrays describe the
  ! same accepted state and independently recomputed open-boundary traces.
  do local_cell=1,nc
   do local_fraction=1,nfrac
    write(channel_cell_unit,*) time,local_cell-1,local_fraction-1,xy(:,local_cell), &
     channel_raw_bed_cell(:,local_fraction,local_cell), &
     channel_raw_suspended_cell(:,local_fraction,local_cell), &
     channel_composed_bed_cell(:,local_fraction,local_cell), &
     channel_retained_suspended_cell(:,local_fraction,local_cell), &
     channel_signed_suspended_rate_cell(local_fraction,local_cell), &
     channel_equilibrium_concentration_cell(local_fraction,local_cell)
   enddo
  enddo
  flush(channel_cell_unit)
  do local_face=1,nf
   if(bid(local_face)<=0) cycle
   do local_fraction=1,nfrac
    write(channel_boundary_unit,*) time,local_face-1,bid(local_face)-1, &
     owner(local_face)-1,local_fraction-1,split_boundary_role(local_face), &
     channel_raw_bed_boundary(:,local_fraction,local_face), &
     channel_raw_suspended_boundary(:,local_fraction,local_face), &
     channel_composed_bed_boundary(:,local_fraction,local_face), &
     channel_retained_suspended_boundary(:,local_fraction,local_face), &
     channel_signed_suspended_rate_boundary(local_fraction,local_face), &
     channel_equilibrium_concentration_boundary(local_fraction,local_face)
   enddo
  enddo
  flush(channel_boundary_unit)
 end subroutine output_channel_composition_frame

 subroutine output_split_capacity_frame()
  integer :: local_cell,local_fraction,local_face
  real(8),allocatable :: diagnostic_next(:,:),diagnostic_accepted(:,:),diagnostic_flux(:,:)
  logical :: diagnostic_ok
  ! Re-enter the authoritative native face path at zero time increment.  This
  ! evaluates cells and open boundaries from the same accepted state and the
  ! same time-dependent traces; the output layer does not reimplement a law.
  allocate(diagnostic_next(nstate,nc),diagnostic_accepted(nledger,nb),diagnostic_flux(nstate,nf))
  call euler(q,0.d0,time,1.d0,diagnostic_next,diagnostic_accepted,diagnostic_ok, &
   instantaneous_face_flux=diagnostic_flux)
  if(.not.diagnostic_ok) then
   write(*,'(a,i0)') 'split-capacity endpoint diagnostic failure reason=',failure_reason
   write(*,'(a,3(1x,i0),6(1x,es14.6))') 'split-capacity trace diagnostic=', &
    failed_trace_boundary,failed_trace_face,failed_trace_condition,failed_trace_values
   error stop 'invalid split-capacity endpoint diagnostic'
  endif
  do local_cell=1,nc
   do local_fraction=1,nfrac
    if(typed_constitutive_declaration) then
     write(split_capacity_unit,*) time,local_cell-1,local_fraction-1,xy(:,local_cell), &
      bedq_equilibrium(:,local_fraction,local_cell), &
      suspended_capacity_equilibrium(:,local_fraction,local_cell), &
      capacity_absolute_shields(local_fraction,local_cell), &
      capacity_critical_shields(local_fraction,local_cell), &
      bedtau_cell(local_fraction,local_cell),capacity_transport_stage(local_fraction,local_cell), &
      capacity_diagnostic_applicability(capacity_critical_shields(local_fraction,local_cell))
    else
     write(split_capacity_unit,*) time,local_cell-1,local_fraction-1,xy(:,local_cell), &
      bedq_equilibrium(:,local_fraction,local_cell), &
      suspended_capacity_equilibrium(:,local_fraction,local_cell), &
      capacity_absolute_shields(local_fraction,local_cell), &
      capacity_critical_shields(local_fraction,local_cell), &
      bedtau_cell(local_fraction,local_cell),capacity_transport_stage(local_fraction,local_cell)
    endif
   enddo
  enddo
  flush(split_capacity_unit)
  do local_face=1,nf
   if(bid(local_face)<=0) cycle
   do local_fraction=1,nfrac
    if(typed_constitutive_declaration) then
     write(split_boundary_unit,*) time,local_face-1,bid(local_face)-1,owner(local_face)-1, &
      local_fraction-1,split_boundary_role(local_face),normal(:,local_face), &
      split_boundary_bed(:,local_fraction,local_face), &
      split_boundary_suspended(:,local_fraction,local_face), &
      split_boundary_absolute_shields(local_fraction,local_face), &
      split_boundary_critical_shields(local_fraction,local_face), &
      split_boundary_stress_ratio(local_fraction,local_face), &
      split_boundary_transport_stage(local_fraction,local_face), &
      capacity_diagnostic_applicability(split_boundary_critical_shields(local_fraction,local_face))
    else
     write(split_boundary_unit,*) time,local_face-1,bid(local_face)-1,owner(local_face)-1, &
      local_fraction-1,split_boundary_role(local_face),normal(:,local_face), &
      split_boundary_bed(:,local_fraction,local_face), &
      split_boundary_suspended(:,local_fraction,local_face), &
      split_boundary_absolute_shields(local_fraction,local_face), &
      split_boundary_critical_shields(local_fraction,local_face), &
      split_boundary_stress_ratio(local_fraction,local_face), &
      split_boundary_transport_stage(local_fraction,local_face)
    endif
   enddo
  enddo
  flush(split_boundary_unit)
  if(channel_face_unit/=-1) then
   do local_face=1,nf
    do local_fraction=1,nfrac
     write(channel_face_unit,*) time,local_face-1,owner(local_face)-1, &
      neighbour(local_face)-1,bid(local_face)-1,local_fraction-1, &
      normal(:,local_face),length(local_face), &
      diagnostic_flux(bed_index(local_fraction),local_face), &
      diagnostic_flux(suspended_index(local_fraction),local_face)
    enddo
   enddo
   flush(channel_face_unit)
  endif
  deallocate(diagnostic_next,diagnostic_accepted,diagnostic_flux)
 end subroutine

 subroutine output_suspended_exchange_frame()
  integer :: local_fraction
  real(8) :: residual
  do local_fraction=1,nfrac
   residual=sum(suspended_exchange_cumulative(:,local_fraction))
   write(suspended_exchange_unit,*) time,local_fraction-1, &
    suspended_exchange_cumulative(:,local_fraction),residual
  enddo
  flush(suspended_exchange_unit)
 end subroutine

 subroutine output_flow2dh_boundary_frame()
  integer :: local_boundary,local_fraction
  real(8) :: advective,diffusive,total,residual
  do local_boundary=1,nb
   do local_fraction=1,nfrac
    advective=ledger(suspended_advective_ledger_index(local_fraction),local_boundary)
    diffusive=ledger(suspended_diffusive_ledger_index(local_fraction),local_boundary)
    total=ledger(1+local_fraction,local_boundary)
    residual=total-advective-diffusive
    write(flow2dh_boundary_unit,*) time,local_boundary-1,local_fraction-1, &
     advective,diffusive,total,residual
   enddo
  enddo
  flush(flow2dh_boundary_unit)
 end subroutine

 subroutine output_flow2dh_memory_frame()
  integer :: local_face,local_fraction,stable_face
  do local_face=1,nf
   if(bid(local_face)<=0.or.neighbour(local_face)>0) cycle
   stable_face=flow2dh_memory_face(local_face)
   do local_fraction=1,nfrac
    write(flow2dh_memory_unit,*) time,stable_face-1,bid(local_face)-1, &
     local_fraction-1,suspended_boundary_last_outflow(local_fraction,stable_face), &
     suspended_boundary_return_clock(local_fraction,stable_face)
   enddo
  enddo
  flush(flow2dh_memory_unit)
 end subroutine

 subroutine output_gaia_fixed_inlet_flux_frame()
  ! Evaluate the accepted state with the actual native Euler face path at
  ! zero time increment.  The returned face flux is downstream of capacity,
  ! secondary stress, coupled law-9 splitting and donor limiting; no
  ! constitutive equation is reimplemented by the output layer.
  real(8),allocatable :: diagnostic_next(:,:),diagnostic_accepted(:,:),diagnostic_flux(:,:),diagnostic_bed(:)
  real(8) :: aggregate_flux,owner_change,initial_owner_bed
  integer :: local_boundary,k,local_face,local_cell,local_fraction
  logical :: diagnostic_ok
  allocate(diagnostic_next(nstate,nc),diagnostic_accepted(nledger,nb), &
   diagnostic_flux(nstate,nf),diagnostic_bed(nc))
  call euler(q,0.d0,time,1.d0,diagnostic_next,diagnostic_accepted,diagnostic_ok, &
   instantaneous_face_flux=diagnostic_flux)
  if(.not.diagnostic_ok) error stop 'invalid native TELEMAC-GAIA endpoint face-flux diagnostic'
  call effective_bed_elevation(q,diagnostic_bed)
  do local_boundary=1,nb
   if(fixed_boundary_bed_mode(local_boundary)/=1) cycle
   aggregate_flux=0.d0;owner_change=0.d0
   do k=boundary_offset(local_boundary),boundary_offset(local_boundary+1)-1
    local_face=boundary_faces(k);local_cell=owner(local_face)
    do local_fraction=1,nfrac
     aggregate_flux=aggregate_flux-length(local_face)* &
      diagnostic_flux(bed_index(local_fraction),local_face)
    enddo
    initial_owner_bed=fixed_boundary_bed(local_face)-boundary_bed_offset(local_face)
    owner_change=max(owner_change,abs(diagnostic_bed(local_cell)-initial_owner_bed))
   enddo
   write(gaia_fixed_unit,*) time,local_boundary,aggregate_flux,owner_change
  enddo
  flush(gaia_fixed_unit)
  deallocate(diagnostic_next,diagnostic_accepted,diagnostic_flux,diagnostic_bed)
 end subroutine

 subroutine output_physical_toe_flux_frame()
  integer :: local_fraction,segment,local_face,bank_id,toe_id,active_flag
  real(8) :: segment_mid(2),segment_length
  ! A scheduled AMR regrid can occur immediately before an output frame.  In
  ! that case refresh only the geometry-derived map/metadata so ``active`` and
  ! ``local_face_id`` describe the hierarchy being written, not its predecessor.
  if(.not.bank_toe_mapping_ready) call build_physical_toe_map(q)
  do segment=1,size(physical_toe_flux_cumulative,2)
   if(.not.physical_toe_flux_active(segment).and. &
      all(physical_toe_flux_cumulative(:,segment)==0.d0).and. &
      all(physical_toe_flux_absolute_cumulative(:,segment)==0.d0)) cycle
   local_face=physical_toe_flux_local_face(segment)-1
   bank_id=physical_toe_flux_bank_id(segment)-1;toe_id=physical_toe_flux_toe_id(segment)-1
   active_flag=merge(1,0,physical_toe_flux_active(segment))
   if(amr_enabled) then
    segment_mid=af_mid(:,segment);segment_length=af_length(segment)
   else
    segment_mid=mid(:,segment);segment_length=length(segment)
   endif
   do local_fraction=1,nfrac
    write(toe_flux_unit,*) time,segment-1,local_face,active_flag,bank_id,toe_id, &
     segment_mid,segment_length,local_fraction,physical_toe_flux_cumulative(local_fraction,segment), &
     physical_toe_flux_absolute_cumulative(local_fraction,segment)
   enddo
  enddo
  flush(toe_flux_unit)
 end subroutine

 subroutine output_secondary_budget_frame()
  integer :: local_cell
  real(8) :: current_integral,boundary_integral,residual
  current_integral=0.d0
  do local_cell=1,nc
   current_integral=current_integral+area(local_cell)*q(secondary_state_index(),local_cell)
  enddo
  boundary_integral=sum(ledger(secondary_ledger_index(),:))
  residual=current_integral-secondary_initial_integral-boundary_integral-secondary_source_cumulative
  write(secondary_budget_unit,*) time,current_integral,secondary_initial_integral,boundary_integral, &
   secondary_source_cumulative,residual
  flush(secondary_budget_unit)
 end subroutine

 subroutine output_secondary_momentum_budget_frame()
  real(8) :: current(2),residual(2)
  call domain_momentum(q,current)
  residual=current-secondary_momentum_initial-secondary_momentum_boundary_cumulative- &
   secondary_stress_boundary_cumulative-secondary_momentum_remainder_cumulative
  write(secondary_momentum_unit,*) time,current,secondary_momentum_initial, &
   secondary_momentum_boundary_cumulative,secondary_stress_boundary_cumulative, &
   secondary_momentum_remainder_cumulative,residual
  flush(secondary_momentum_unit)
 end subroutine

 subroutine output_bedform_roughness_frame()
  integer :: local_cell,local_iterations
  real(8) :: effective_n,dune_height,dune_length,total_ks
  logical :: valid
  do local_cell=1,nc
   call bedform_properties(q,local_cell,effective_n,dune_height,dune_length,total_ks,local_iterations,valid)
   if(.not.valid) error stop 'invalid bedform roughness output diagnostic'
   write(bedform_unit,*) time,local_cell-1,xy(:,local_cell),effective_n,dune_height,dune_length,total_ks,local_iterations
  enddo
  flush(bedform_unit)
 end subroutine

 subroutine output_adaptation_frame()
  integer :: local_cell,local_fraction
  real(8),allocatable :: diagnostic_hiding(:,:),diagnostic_dg(:),diagnostic_gsd(:),diagnostic_sand(:),diagnostic_d90(:),diagnostic_mud(:)
  real(8) :: diagnostic_concentration(nfrac,nb),diagnostic_bed_input(nfrac,nb)
  allocate(diagnostic_hiding(nfrac,nc),diagnostic_dg(nc),diagnostic_gsd(nc),diagnostic_sand(nc), &
   diagnostic_d90(nc),diagnostic_mud(nc))
  call hiding_factors(q,diagnostic_hiding)
  call mixture_statistics(q,diagnostic_dg,diagnostic_gsd,diagnostic_sand,diagnostic_d90,diagnostic_mud)
  call boundary_sediment_at(time,diagnostic_concentration,diagnostic_bed_input)
  call prepare_bedload_capacity(q,diagnostic_hiding,diagnostic_dg,diagnostic_gsd,diagnostic_sand,diagnostic_d90,diagnostic_mud, &
   diagnostic_bed_input,bedq_cell,bedtau_cell,bedq_grad)
  do local_cell=1,nc
   do local_fraction=1,nfrac
    write(adaptation_unit,*) time,local_cell-1,local_fraction-1,xy(:,local_cell), &
     bedload_adaptation_length_diameters*d50_fraction(local_fraction), &
     bedq_equilibrium(:,local_fraction,local_cell),bedq_cell(:,local_fraction,local_cell), &
     bedload_adaptation_iterations,bedload_adaptation_residual
   enddo
  enddo
  deallocate(diagnostic_hiding,diagnostic_dg,diagnostic_gsd,diagnostic_sand,diagnostic_d90,diagnostic_mud)
  flush(adaptation_unit)
 end subroutine

 subroutine output_bank_failure_frame()
  integer :: local_cell,node
  do local_cell=1,nc
   if(.not.bank_erodible(local_cell).and.bank_undercut(local_cell)==0.and. &
      bank_collapse_count(local_cell)==0) cycle
   node=local_cell
   if(amr_enabled) node=at_leaf(local_cell)
   write(bank_unit,*) time,local_cell-1,node-1,xy(:,local_cell),bank_undercut(local_cell), &
    bank_reference_height(local_cell),bank_toe_height(local_cell), &
    bank_toe_cumulative(local_cell)*area(local_cell),bank_collapse_cumulative(local_cell)*area(local_cell), &
    bank_collapse_count(local_cell)
  enddo
  flush(bank_unit)
 end subroutine

 subroutine output_bank_front_frame()
  ! One row per erodible material cell makes lateral migration observable
  ! without changing the immutable computational topology.  The plan-area
  ! occupancy is authoritative; ``retreat_fraction`` is therefore valid for
  ! cut cells and AMR leaves even where a single Cartesian distance would be
  ! geometrically ambiguous.
  integer :: local_cell,node,local_fraction
  real(8) :: removed_volume,remaining_fraction
  real(8),allocatable :: raw_bed(:),effective_bed(:)
  allocate(raw_bed(nc),effective_bed(nc))
  call raw_bed_elevation(q,raw_bed)
  call effective_bed_elevation(q,effective_bed)
  do local_cell=1,nc
   if(.not.bank_erodible(local_cell).and. &
      bank_front_status(local_cell)==BANK_FRONT_INACTIVE) cycle
   node=local_cell
   if(amr_enabled) node=at_leaf(local_cell)
   removed_volume=0.d0
   do local_fraction=1,nfrac
    removed_volume=removed_volume+ &
     bank_front_removed_mass(local_fraction,local_cell)/bulk_fraction(local_fraction)
   enddo
   remaining_fraction=max(0.d0,min(1.d0, &
    bank_front_solid_area(local_cell)/area(local_cell)))
   write(bank_front_unit,*) time,local_cell-1,node-1,bank_front_status(local_cell), &
    bank_front_seed_face(local_cell)-1,xy(:,local_cell),area(local_cell), &
    bank_front_solid_area(local_cell),remaining_fraction,1.d0-remaining_fraction, &
    bank_front_base(local_cell),bank_front_height(local_cell),raw_bed(local_cell), &
    effective_bed(local_cell),removed_volume
  enddo
  flush(bank_front_unit)
  deallocate(raw_bed,effective_bed)
 end subroutine

 subroutine output_secondary_flow_frame()
  integer :: local_cell,local_fraction,gaia_active
  real(8) :: h,u,v,qx,qy,intensity,alpha,diagnostic_bedload(2), &
   raw_chezy,z0_value,drag_value,speed_value
  real(8) :: gaia_h,gaia_tau,gaia_bed_gradient(2),gaia_eta(2),gaia_base_capacity(2)
  real(8) :: gaia_slope_stress(2),gaia_total_stress(2),gaia_secondary_delta(2)
  real(8) :: gaia_factor,gaia_secondary_turn,gaia_total_turn,gaia_cross,gaia_dot
  real(8),allocatable :: diagnostic_hiding(:,:),diagnostic_dg(:),diagnostic_gsd(:),diagnostic_sand(:),diagnostic_d90(:),diagnostic_mud(:)
  real(8) :: diagnostic_concentration(nfrac,nb),diagnostic_bed_input(nfrac,nb)
  logical :: valid
  if(classic_secondary_foundation()) then
   call refresh_classic_secondary_equilibrium(q)
  else if(prognostic_secondary_flow()) then
   call secondary_flow_curvatures(q,secondary_curvature)
  endif
  if(bedlaw>0) then
   allocate(diagnostic_hiding(nfrac,nc),diagnostic_dg(nc),diagnostic_gsd(nc),diagnostic_sand(nc), &
    diagnostic_d90(nc),diagnostic_mud(nc))
   call hiding_factors(q,diagnostic_hiding)
   call mixture_statistics(q,diagnostic_dg,diagnostic_gsd,diagnostic_sand,diagnostic_d90,diagnostic_mud)
   call boundary_sediment_at(time,diagnostic_concentration,diagnostic_bed_input)
   call prepare_bedload_capacity(q,diagnostic_hiding,diagnostic_dg,diagnostic_gsd,diagnostic_sand,diagnostic_d90,diagnostic_mud, &
    diagnostic_bed_input,bedq_cell,bedtau_cell,bedq_grad)
  endif
  do local_cell=1,nc
   h=q(1,local_cell);u=0;v=0
   if(h>dry_tolerance) then
    u=q(2,local_cell)/h;v=q(3,local_cell)/h
   endif
   do local_fraction=1,nfrac
    qx=u;qy=v;intensity=0;alpha=0;valid=.true.;diagnostic_bedload=0.d0
    if(bedlaw>0) diagnostic_bedload=bedq_cell(:,local_fraction,local_cell)
    if(classic_secondary_foundation()) then
     intensity=secondary_classic_intensity(local_cell)
     speed_value=sqrt(u*u+v*v)
     if(h>dry_tolerance.and.speed_value>0.d0) then
      call hydraulic_roughness_at(h,manning_cell(local_cell),raw_chezy,z0_value, &
       drag_value,valid)
      if(valid) alpha=12.5d0*secondary_foundation_espir* &
       (1.d0-sqrt(grav)/(.4d0*raw_chezy))*intensity/speed_value
     endif
    else if(secondary_flow_model==1) then
     if(h>dry_tolerance.and.qx*qx+qy*qy>0) call braids_delft3d_spiral_transport( &
      h,u,v,hydraulic_equivalent_manning(h,manning_cell(local_cell)),grav, &
      secondary_flow_espir,secondary_curvature(local_cell), &
      qx,qy,intensity,alpha,valid,engelund_hansen_chezy)
    else if(secondary_flow_model==2.and.h>dry_tolerance) then
     intensity=q(secondary_state_index(),local_cell)/h
     if(qx*qx+qy*qy>0) call braids_spiral_transport_from_intensity(h,u,v, &
      hydraulic_equivalent_manning(h,manning_cell(local_cell)), &
      grav,secondary_flow_espir,intensity,qx,qy,alpha,valid,engelund_hansen_chezy)
    endif
    if(.not.valid) error stop 'invalid secondary-flow output diagnostic'
    write(secondary_unit,*) time,local_cell-1,local_fraction-1,xy(:,local_cell), &
     secondary_curvature(local_cell),intensity,alpha,diagnostic_bedload
    if(secondary_flow_model==3) then
     gaia_active=0;gaia_factor=0.d0;gaia_secondary_turn=0.d0;gaia_total_turn=0.d0
     gaia_slope_stress=0.d0;gaia_total_stress=0.d0;gaia_secondary_delta=0.d0
     gaia_eta=gaia_eta_gradient(:,local_cell)
     gaia_bed_gradient=-gaia_bed_down(:,local_cell)
     gaia_tau=bedtau_cell(local_fraction,local_cell)
     gaia_base_capacity=bedq_gaia_base(:,local_fraction,local_cell)
     if(h>dry_tolerance.and.dot_product(gaia_base_capacity,gaia_base_capacity)>0.d0) then
      gaia_h=h
      if(gaia_h<max(bedload_min_depth_fraction(local_fraction),dry_tolerance).and. &
         max(bedload_min_depth_fraction(local_fraction),dry_tolerance)>dry_tolerance) &
       gaia_h=max(bedload_min_depth_fraction(local_fraction),dry_tolerance)*(1.d0+128.d0*epsilon(1.d0))
      if(gaia_interior_slope_talmon) then
       call braids_gaia_talmon_slope(u,v,gaia_bed_gradient(1),gaia_bed_gradient(2),gaia_tau, &
        rhos_fraction(local_fraction),rho,grav,d50_fraction(local_fraction),gaia_slope_beta, &
        gaia_deviation_beta2,gaia_slope_stress(1),gaia_slope_stress(2),gaia_factor,valid)
      else
       call braids_gaia_talmon_slope(u,v,0.d0,0.d0,gaia_tau, &
        rhos_fraction(local_fraction),rho,grav,d50_fraction(local_fraction),gaia_slope_beta, &
        gaia_deviation_beta2,gaia_slope_stress(1),gaia_slope_stress(2),gaia_factor,valid)
      endif
      if(valid.and.gaia_interior_secondary_stress) then
       call braids_gaia_secondary_stress(gaia_h,u,v, &
        hydraulic_equivalent_manning(gaia_h,manning_cell(local_cell)),grav,rho, &
        gaia_secondary_alpha,gaia_eta(1),gaia_eta(2),gaia_slope_stress(1),gaia_slope_stress(2), &
        gaia_total_stress(1),gaia_total_stress(2),valid)
      else if(valid) then
       gaia_total_stress=gaia_slope_stress
      endif
      if(.not.valid) error stop 'invalid TELEMAC-GAIA secondary-stress output diagnostic'
      gaia_secondary_delta=gaia_total_stress-gaia_slope_stress
      gaia_cross=gaia_slope_stress(1)*gaia_total_stress(2)-gaia_slope_stress(2)*gaia_total_stress(1)
      gaia_dot=dot_product(gaia_slope_stress,gaia_total_stress)
      if(dot_product(gaia_slope_stress,gaia_slope_stress)>0.d0.and. &
         dot_product(gaia_total_stress,gaia_total_stress)>0.d0) gaia_secondary_turn=atan2(gaia_cross,gaia_dot)
      gaia_cross=u*gaia_total_stress(2)-v*gaia_total_stress(1)
      gaia_dot=u*gaia_total_stress(1)+v*gaia_total_stress(2)
      if(u*u+v*v>0.d0.and.dot_product(gaia_total_stress,gaia_total_stress)>0.d0) &
       gaia_total_turn=atan2(gaia_cross,gaia_dot)
      gaia_active=1
     endif
     write(gaia_secondary_unit,*) time,local_cell-1,local_fraction-1,xy(:,local_cell),gaia_active, &
      gaia_eta,gaia_bed_gradient,gaia_tau,gaia_slope_stress,gaia_secondary_delta,gaia_total_stress, &
      gaia_secondary_turn,gaia_total_turn,gaia_base_capacity,diagnostic_bedload,gaia_factor
    endif
   enddo
  enddo
  if(bedlaw>0) deallocate(diagnostic_hiding,diagnostic_dg,diagnostic_gsd,diagnostic_sand,diagnostic_d90,diagnostic_mud)
  flush(secondary_unit)
  if(gaia_secondary_unit/=-1) flush(gaia_secondary_unit)
 end subroutine

 subroutine output_live_preview()
  ! Fast, fixed-grid preview frames for QGIS.  These raw Float32 rasters avoid
  ! formatted-frame parsing and post-run polygon rasterisation.  The QGIS
  ! plugin supplies their VRT georeferencing and only reads a frame after the
  ! final marker below has been flushed.
  integer :: i,j,c,n,f,layer,preview_unit,index_unit,io_status,command_status,mkdir_status
  logical, save :: preview_directory_ready=.false.
  real(8) :: depth,speed,bed,preview_phase_started
  real(8),allocatable :: preview_bed(:)
  real(4), allocatable :: preview_depth(:,:),preview_speed(:,:),preview_stage(:,:),preview_hu(:,:),preview_hv(:,:)
  character(16) :: frame_text
  character(256) :: depth_path,speed_path,stage_path,hu_path,hv_path
  if(profile_enabled) preview_phase_started=omp_get_wtime()
  if(.not.preview_directory_ready) then
   ! A prepared QGIS run creates this directory, but the native executable is
   ! also a supported standalone artifact. Probe first so an existing folder
   ! is quiet; create it with the platform shell only when it is absent.
   open(newunit=preview_unit,file='live_preview/.braids_write_probe',status='replace', &
   action='write',iostat=io_status)
   if(io_status/=0) then
    mkdir_status=0;command_status=0
    call execute_command_line('mkdir live_preview',wait=.true.,exitstat=mkdir_status,cmdstat=command_status)
    if(command_status/=0.or.mkdir_status/=0) error stop 'cannot create live_preview directory'
    open(newunit=preview_unit,file='live_preview/.braids_write_probe',status='replace', &
     action='write',iostat=io_status)
   endif
   if(io_status/=0) error stop 'cannot write live_preview directory'
   close(preview_unit,status='delete');preview_directory_ready=.true.
  endif
  write(frame_text,'(i8.8)') frame
  depth_path='live_preview/depth_m_'//trim(adjustl(frame_text))//'.f32'
  speed_path='live_preview/velocity_m_s_'//trim(adjustl(frame_text))//'.f32'
  stage_path='live_preview/stage_m_'//trim(adjustl(frame_text))//'.f32'
  hu_path='live_preview/discharge_x_m2_s_'//trim(adjustl(frame_text))//'.f32'
  hv_path='live_preview/discharge_y_m2_s_'//trim(adjustl(frame_text))//'.f32'
  allocate(preview_depth(nx,ny),preview_speed(nx,ny),preview_stage(nx,ny),preview_hu(nx,ny),preview_hv(nx,ny), &
           preview_bed(nc))
  call effective_bed_elevation(q,preview_bed)
  preview_depth=-3.4028234e38;preview_speed=-3.4028234e38;preview_stage=-3.4028234e38
  preview_hu=-3.4028234e38;preview_hv=-3.4028234e38
  if(amr_enabled) then
   ! Active AMR leaves tile the finest native grid without overlap.  Expand
   ! each leaf to that grid only for this lightweight QGIS preview; the
   ! authoritative variable-leaf frame remains in frames.txt.
   !$omp parallel do schedule(static,256) if(nx*ny>=262144) default(shared) private(c,n,depth,speed,bed,f,layer,i,j)
   do c=1,nc
    n=at_leaf(c);depth=max(0.d0,q(1,c))
    if(depth>dry_tolerance) then
     speed=sqrt(q(2,c)**2+q(3,c)**2)/depth;bed=preview_bed(c)
    else
     speed=0.d0
    endif
    do j=at_j(n),at_j(n)+at_stride(n)-1
     do i=at_i(n),at_i(n)+at_stride(n)-1
      preview_depth(i,ny-j+1)=real(depth,4);preview_speed(i,ny-j+1)=real(speed,4)
      preview_hu(i,ny-j+1)=real(q(2,c),4);preview_hv(i,ny-j+1)=real(q(3,c),4)
      if(depth>dry_tolerance) preview_stage(i,ny-j+1)=real(depth+bed,4)
     enddo
    enddo
   enddo
   !$omp end parallel do
  else
   !$omp parallel do collapse(2) schedule(static,256) if(nx*ny>=262144) default(shared) private(i,j,c,depth,speed,bed,f,layer)
   do j=1,ny
    do i=1,nx
     c=grid(i,j)
     if(c<=0) cycle
     depth=max(0.d0,q(1,c));preview_depth(i,ny-j+1)=real(depth,4)
     preview_hu(i,ny-j+1)=real(q(2,c),4);preview_hv(i,ny-j+1)=real(q(3,c),4)
     if(depth>dry_tolerance) then
      speed=sqrt(q(2,c)**2+q(3,c)**2)/depth;preview_speed(i,ny-j+1)=real(speed,4)
      bed=preview_bed(c)
      preview_stage(i,ny-j+1)=real(depth+bed,4)
     else
      preview_speed(i,ny-j+1)=0.
     endif
    enddo
   enddo
   !$omp end parallel do
  endif
  if(profile_enabled) then
   profile_output_preview_pack_s=profile_output_preview_pack_s+omp_get_wtime()-preview_phase_started
   preview_phase_started=omp_get_wtime()
  endif
  open(newunit=preview_unit,file=trim(depth_path),status='replace',form='unformatted',access='stream',action='write');write(preview_unit) preview_depth;close(preview_unit)
  open(newunit=preview_unit,file=trim(speed_path),status='replace',form='unformatted',access='stream',action='write');write(preview_unit) preview_speed;close(preview_unit)
  open(newunit=preview_unit,file=trim(stage_path),status='replace',form='unformatted',access='stream',action='write');write(preview_unit) preview_stage;close(preview_unit)
  open(newunit=preview_unit,file=trim(hu_path),status='replace',form='unformatted',access='stream',action='write');write(preview_unit) preview_hu;close(preview_unit)
  open(newunit=preview_unit,file=trim(hv_path),status='replace',form='unformatted',access='stream',action='write');write(preview_unit) preview_hv;close(preview_unit)
  deallocate(preview_depth,preview_speed,preview_stage,preview_hu,preview_hv,preview_bed)
  open(newunit=index_unit,file='live_preview/frames.csv',status='unknown',position='append',action='write')
  write(index_unit,'(i0,1x,es26.17e3,1x,i0)') frame,time,steps
  close(index_unit)
  open(newunit=index_unit,file='live_preview/latest.txt',status='replace',action='write')
  write(index_unit,'(i0,1x,es26.17e3,1x,i0)') frame,time,steps
  flush(index_unit);close(index_unit)
  if(profile_enabled) profile_output_preview_io_s=profile_output_preview_io_s+omp_get_wtime()-preview_phase_started
 end subroutine

 subroutine output_section_frame()
  real(8), allocatable :: current_bed(:),hydraulic(:,:)
  real(8) :: wet_area,top_width,conveyance
  integer :: fraction,layer,boundary
  logical :: valid
  allocate(current_bed(nc),hydraulic(2,nb));call effective_bed_elevation(q,current_bed)
  call boundary_hydraulic_at(time,current_bed,hydraulic,valid)
  if(.not.valid) error stop 'cannot resolve live section control at saved output'
  do boundary=1,nb
   if(section_control(boundary)==0) cycle
   call section_metrics(boundary,hydraulic(2,boundary),current_bed,wet_area,top_width,conveyance)
   write(section_unit,'(es26.17e3,1x,i0,1x,i0,5(1x,es26.17e3))') time,boundary,section_control(boundary), &
    hydraulic(1,boundary),hydraulic(2,boundary),wet_area,top_width,conveyance
  enddo
  flush(section_unit);deallocate(current_bed,hydraulic)
 end subroutine
end program
