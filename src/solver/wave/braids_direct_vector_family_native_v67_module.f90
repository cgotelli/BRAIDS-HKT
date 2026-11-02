! SPDX-License-Identifier: AGPL-3.0-or-later
!
! Bounded native/OpenMP lifecycle for all six Q3-H6 direct-current
! producers added in v67.  Constitutive evaluation is deliberately separated
! from the immutable v65 face topology and Exner projection.  The producer
! adapter is always invoked in fraction-free face-lifecycle mode; accepted
! surface composition and finite inventory are therefore each owned exactly
! once by braids_scalar_flow_project_exner_v65.
!
! Scope: fixed grid, fully wet, first-order face reconstruction, upwind
! bedload and current-only forcing.  Bed slope remains off unless the additive
! positive-TraFrm v69 receipt and all ISlope=2 fields are supplied explicitly.
! TRA_COMBINE is rejected until the Q4 suspended-state owner is present.
module braids_direct_vector_family_native_v67_module
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use omp_lib, only: omp_get_num_threads
    use braids_direct_producer_adapter_v67_module, only: &
        BRAIDS_V67_WIRE_VAN_RIJN_1993, BRAIDS_V67_WIRE_BIJKER, &
        BRAIDS_V67_WIRE_VAN_THIEL_VAN_RIJN, &
        BRAIDS_V67_WIRE_XBEACH_SOULSBY_VAN_RIJN, &
        braids_direct_route_controls_v67, braids_vr1993_input_v67, &
        braids_bijker_input_v67, braids_1920_input_v67, &
        braids_direct_producer_result_v67, &
        braids_run_vr1993_adapter_v67, braids_run_bijker_adapter_v67, &
        braids_run_van_thiel_van_rijn_adapter_v67, &
        braids_run_xbeach_soulsby_van_rijn_adapter_v67
    use braids_bedload_current_module, only: braids_soulsby_current
    use braids_soulsby_van_rijn_module, only: &
        braids_soulsby_van_rijn_current
    use braids_direct_vector_route_composer_v66_module, only: &
        braids_compose_direct_vector_route_v66, BRAIDS_DV_ROUTE_OK
    use braids_direct_vector_maximum_waterdepth_v68_module, only: &
        BRAIDS_V68_FLOW_EPS, &
        braids_prepare_direct_vector_hydraulics_v68, &
        braids_select_direct_vector_velocity_v68
    use braids_scalar_flow_live_v65_module, only: &
        scalar_flow_projection_topology_v65, &
        scalar_live_face_internal, scalar_live_face_trace_inlet, &
        scalar_live_face_trace_outlet, scalar_live_face_owner_outlet, &
        scalar_live_face_closed, scalar_live_ok, &
        braids_prepare_scalar_flow_topology_v65, &
        braids_scalar_flow_face_candidates_v65, &
        braids_scalar_flow_project_exner_v65
    implicit none
    private

    integer, parameter, public :: direct_family_v67_ok = 0
    integer, parameter, public :: direct_family_v67_invalid_shape = 670
    integer, parameter, public :: direct_family_v67_invalid_input = 671
    integer, parameter, public :: direct_family_v67_invalid_topology = 672
    integer, parameter, public :: direct_family_v67_unsupported_method = 673
    integer, parameter, public :: direct_family_v67_producer_failure = 674
    integer, parameter, public :: direct_family_v67_route_failure = 675
    integer, parameter, public :: direct_family_v67_combine_requires_q4 = 676
    integer, parameter, public :: direct_family_v67_face_failure = 677
    integer, parameter, public :: direct_family_v67_exner_failure = 678
    integer, parameter, public :: direct_family_v67_state_mismatch = 679
    integer, parameter, public :: direct_family_v67_trial_not_pending = 680
    integer, parameter, public :: direct_family_v67_step_limit = 681
    integer, parameter, public :: direct_family_v67_not_fully_wet = 682
    integer, parameter, public :: direct_family_v67_missing_v68_receipt = 683
    integer, parameter, public :: direct_family_v67_v68_receipt_mismatch = 684
    integer, parameter, public :: direct_family_v67_v68_preprocess_failure = 685
    integer, parameter, public :: direct_family_v69_taurat_receipt_mismatch = 686
    integer, parameter, public :: direct_family_v69_live_taurat_required = 687

    integer, parameter, public :: direct_family_v67_tra_bedload = 1
    integer, parameter, public :: direct_family_v67_tra_advdiff = 2
    integer, parameter, public :: direct_family_v67_tra_combine = 3
    integer, parameter, public :: direct_family_v67_max_accepted_steps = 20
    integer, parameter, public :: direct_family_v67_wire_soulsby = 67012
    integer, parameter, public :: &
        direct_family_v67_wire_soulsby_van_rijn = 67011
    character(len=64), parameter, public :: &
        direct_family_v68_semantic_id = &
        'cc3c9037db1909b51f081631904cfd6fac3842d33b4db224602993a0e9bde550'
    character(len=64), parameter, public :: &
        direct_family_v69_taurat_contract_fingerprint = &
        'cc0c86f58a8d38513c3793f0c1cf8f45bc25713f9c8f8e6fa4a20763648d6599'

    ! Re-export the five authenticated v65 face roles without inventing a
    ! second numbering scheme.
    integer, parameter, public :: direct_family_v67_face_internal = &
        scalar_live_face_internal
    integer, parameter, public :: direct_family_v67_face_trace_inlet = &
        scalar_live_face_trace_inlet
    integer, parameter, public :: direct_family_v67_face_trace_outlet = &
        scalar_live_face_trace_outlet
    integer, parameter, public :: direct_family_v67_face_owner_outlet = &
        scalar_live_face_owner_outlet
    integer, parameter, public :: direct_family_v67_face_closed = &
        scalar_live_face_closed

    type, public :: braids_soulsby_input_v67
        real(real64) :: depth = 0.0_real64
        real(real64) :: u = 0.0_real64
        real(real64) :: v = 0.0_real64
        real(real64) :: d50 = 0.0_real64
        real(real64) :: solid_density = 0.0_real64
        real(real64) :: water_density = 0.0_real64
        real(real64) :: gravity = 0.0_real64
        real(real64) :: kinematic_viscosity = 0.0_real64
        real(real64) :: calibration = 0.0_real64
        integer :: model_index = 1
        real(real64) :: ratio_d50_z0 = 0.0_real64
        real(real64) :: von_karman = 0.0_real64
    end type braids_soulsby_input_v67

    type, public :: braids_soulsby_van_rijn_input_v67
        real(real64) :: depth = 0.0_real64
        real(real64) :: u = 0.0_real64
        real(real64) :: v = 0.0_real64
        real(real64) :: d50 = 0.0_real64
        real(real64) :: solid_density = 0.0_real64
        real(real64) :: water_density = 0.0_real64
        real(real64) :: gravity = 0.0_real64
        real(real64) :: kinematic_viscosity = 0.0_real64
        real(real64) :: calibration = 0.0_real64
        real(real64) :: d90_d50 = 0.0_real64
        real(real64) :: roughness_z0 = 0.0_real64
    end type braids_soulsby_van_rijn_input_v67

    type, public :: braids_direct_family_point_v67
        ! Exactly one member is read, according to the separately authenticated
        ! per-fraction wire ID.  Keeping the wire outside this point object
        ! prevents a cell or boundary trace from silently changing method.
        type(braids_soulsby_input_v67) :: soulsby
        type(braids_soulsby_van_rijn_input_v67) :: soulsby_van_rijn
        type(braids_vr1993_input_v67) :: van_rijn_1993
        type(braids_bijker_input_v67) :: bijker
        type(braids_1920_input_v67) :: modern
    end type braids_direct_family_point_v67

    type, public :: braids_direct_family_hydraulic_point_v68
        ! One source-order hydraulic point is shared by every fraction.  The
        ! physical depth is injected into every producer; dwnvel may only
        ! change the characteristic/effective velocity and VR93 height.
        real(real64) :: physical_depth = 0.0_real64
        real(real64) :: transport_u = 0.0_real64
        real(real64) :: transport_v = 0.0_real64
        real(real64) :: effective_speed = 0.0_real64
        integer :: active = 1
    end type braids_direct_family_hydraulic_point_v68

    type, public :: braids_direct_family_execution_v68
        ! Native consumers receive the already authenticated v68 envelope
        ! fingerprint and semantic ID from the Python control plane.  The
        ! exact switch and ordered support CSR are sealed here; a bare Boolean
        ! is never sufficient preparation.
        private
        logical :: prepared = .false.
        logical :: maximum_waterdepth_is_explicit = .false.
        logical :: maximum_waterdepth_enabled = .false.
        integer :: ncell = 0
        integer :: ntrace = 0
        character(len=64) :: semantic_id = ''
        character(len=64) :: execution_fingerprint = ''
        character(len=64) :: topology_fingerprint = ''
        integer(int64), allocatable :: cell_support_offset(:)
        integer(int64), allocatable :: cell_support_ids(:)
        integer(int64), allocatable :: trace_support_offset(:)
        integer(int64), allocatable :: trace_support_ids(:)
    end type braids_direct_family_execution_v68

    type, public :: braids_direct_family_method_identity_v67
        ! A wire ID is never a public method alias.  Preparation resolves and
        ! seals the public catalogue entry, the exact current-only adapter
        ! variant, and the forcing variant together.
        private
        logical :: prepared = .false.
        integer :: wire = 0
        character(len=48) :: catalogue_entry_id = ''
        character(len=64) :: implementation_variant_id = ''
        character(len=16) :: forcing_variant = ''
    end type braids_direct_family_method_identity_v67

    type, public :: braids_direct_family_topology_v67
        private
        logical :: prepared = .false.
        integer :: ncell = 0
        integer :: nface = 0
        integer :: nboundary = 0
        type(scalar_flow_projection_topology_v65) :: projection
        integer, allocatable :: owner(:), neighbour(:), role(:)
        real(real64), allocatable :: normal(:,:), length_m(:)
        type(braids_direct_family_execution_v68) :: execution_v68
    end type braids_direct_family_topology_v67

    type, public :: braids_direct_family_state_v67
        private
        logical :: initialized = .false.
        integer :: nfrac = 0
        integer :: ncell = 0
        integer :: accepted_step = 0
        real(real64) :: time_s = 0.0_real64
        real(real64), allocatable :: bed_mass_kg_m2(:,:)
        real(real64), allocatable :: bulk_density_kg_m3(:)
        real(real64), allocatable :: active_fraction(:,:)
        real(real64), allocatable :: boundary_signed_kg(:)
        real(real64), allocatable :: boundary_absolute_kg(:)
        logical :: maximum_waterdepth_is_explicit = .false.
        logical :: maximum_waterdepth_enabled = .false.
        character(len=64) :: execution_fingerprint = ''
        character(len=64) :: topology_fingerprint = ''
    end type braids_direct_family_state_v67

    type, public :: braids_direct_family_trial_v67
        private
        logical :: pending = .false.
        integer :: nfrac = 0
        integer :: ncell = 0
        integer :: nface = 0
        integer :: parent_step = -1
        real(real64) :: parent_time_s = -1.0_real64
        real(real64) :: dt_s = 0.0_real64
        integer :: producer_threads = 1
        integer :: trace_producer_threads = 1
        integer :: face_threads = 1
        integer :: exner_threads = 1
        integer :: bed_factor_application_count = 0
        integer :: suspended_factor_application_count = 0
        integer :: route_surface_fraction_application_count = -1
        integer :: accepted_active_fraction_application_count = 0
        integer :: inventory_fill_application_count = 0
        integer :: bed_slope_application_count = 0
        integer :: taurat_zero_application_count = 0
        integer :: cell_preprocessor_application_count = 0
        integer :: trace_preprocessor_application_count = 0
        real(real64), allocatable :: parent_mass(:,:)
        real(real64), allocatable :: parent_boundary_signed(:)
        real(real64), allocatable :: parent_boundary_absolute(:)
        real(real64), allocatable :: raw_face_candidate(:,:,:)
        real(real64), allocatable :: corrected_face_candidate(:,:,:)
        integer, allocatable :: raw_donor(:,:), donor(:,:), donor_cell(:,:)
        real(real64), allocatable :: face_flux_kg_m_s(:,:)
        real(real64), allocatable :: cell_delta_kg_m2(:,:)
        real(real64), allocatable :: active_fraction_factor(:,:)
        real(real64), allocatable :: inventory_fill(:,:), availability(:,:)
        real(real64), allocatable :: boundary_transfer_kg(:)
        real(real64), allocatable :: boundary_absolute_kg(:)
        real(real64), allocatable :: conservation_residual_kg(:)
        real(real64), allocatable :: cell_physical_depth(:)
        real(real64), allocatable :: cell_observation_depth(:)
        real(real64), allocatable :: cell_producer_velocity(:,:,:)
        real(real64), allocatable :: cell_producer_observation_height(:,:)
        type(braids_direct_family_state_v67) :: candidate
    end type braids_direct_family_trial_v67

    public :: braids_prepare_direct_family_topology_v67
    public :: braids_resolve_direct_family_execution_v68
    public :: braids_export_direct_family_execution_v68
    public :: braids_resolve_direct_family_method_v67
    public :: braids_export_direct_family_method_v67
    public :: braids_direct_family_produce_v67
    public :: braids_initialize_direct_family_state_v67
    public :: braids_restore_direct_family_state_v67
    public :: braids_export_direct_family_state_v67
    public :: braids_export_direct_family_state_identity_v68
    public :: braids_begin_direct_family_trial_v67
    public :: braids_commit_direct_family_trial_v67
    public :: braids_rollback_direct_family_trial_v67
    public :: braids_cancel_direct_family_trial_v67
    public :: braids_export_direct_family_trial_v67
    public :: braids_export_direct_family_trial_hydraulics_v68
    public :: braids_export_direct_family_trial_taurat_v69

contains

    pure logical function valid_sha256(value)
        character(len=*), intent(in) :: value
        integer :: item, code
        valid_sha256 = len_trim(value) == 64
        if (.not.valid_sha256) return
        do item = 1, 64
            code = iachar(value(item:item))
            if (.not.((code >= iachar('0') .and. code <= iachar('9')) .or. &
                      (code >= iachar('a') .and. code <= iachar('f')))) then
                valid_sha256 = .false.
                return
            end if
        end do
    end function valid_sha256

    pure logical function valid_support_csr(npoint, offset, ids)
        integer, intent(in) :: npoint
        integer(int64), intent(in) :: offset(:), ids(:)
        valid_support_csr = size(offset) == npoint+1
        if (.not.valid_support_csr) return
        valid_support_csr = offset(1) == 0_int64 .and. &
            offset(npoint+1) == int(size(ids),int64) .and. &
            all(ids >= 0_int64)
        if (.not.valid_support_csr .or. npoint == 0) return
        valid_support_csr = all(offset(2:) >= offset(:npoint))
    end function valid_support_csr

    pure logical function valid_support_offsets(npoint, nlink, offset)
        integer, intent(in) :: npoint, nlink
        integer(int64), intent(in) :: offset(:)
        valid_support_offsets = size(offset) == npoint+1
        if (.not.valid_support_offsets) return
        valid_support_offsets = offset(1) == 0_int64 .and. &
            offset(npoint+1) == int(nlink,int64)
        if (.not.valid_support_offsets .or. npoint == 0) return
        valid_support_offsets = all(offset(2:) >= offset(:npoint))
    end function valid_support_offsets

    subroutine braids_resolve_direct_family_execution_v68( &
            semantic_id, execution_fingerprint, topology_fingerprint, &
            maximum_waterdepth_is_explicit, maximum_waterdepth_enabled, &
            ncell, ntrace, execution, status, cell_support_offset, &
            cell_support_ids, trace_support_offset, trace_support_ids)
        character(len=*), intent(in) :: semantic_id, execution_fingerprint
        character(len=*), intent(in) :: topology_fingerprint
        logical, intent(in) :: maximum_waterdepth_is_explicit
        logical, intent(in) :: maximum_waterdepth_enabled
        integer, intent(in) :: ncell, ntrace
        type(braids_direct_family_execution_v68), intent(out) :: execution
        integer, intent(out) :: status
        integer(int64), intent(in), optional :: cell_support_offset(:)
        integer(int64), intent(in), optional :: cell_support_ids(:)
        integer(int64), intent(in), optional :: trace_support_offset(:)
        integer(int64), intent(in), optional :: trace_support_ids(:)
        logical :: all_support_present, any_support_present

        status = direct_family_v67_missing_v68_receipt
        if (.not.maximum_waterdepth_is_explicit) return
        status = direct_family_v67_v68_receipt_mismatch
        if (trim(semantic_id) /= direct_family_v68_semantic_id .or. &
            .not.valid_sha256(execution_fingerprint) .or. &
            .not.valid_sha256(topology_fingerprint) .or. &
            ncell <= 0 .or. ntrace < 0) return

        all_support_present = present(cell_support_offset) .and. &
            present(cell_support_ids) .and. present(trace_support_offset) .and. &
            present(trace_support_ids)
        any_support_present = present(cell_support_offset) .or. &
            present(cell_support_ids) .or. present(trace_support_offset) .or. &
            present(trace_support_ids)
        if (maximum_waterdepth_enabled) then
            if (.not.all_support_present) return
            if (.not.valid_support_csr(ncell,cell_support_offset, &
                    cell_support_ids) .or. &
                .not.valid_support_csr(ntrace,trace_support_offset, &
                    trace_support_ids)) return
            allocate(execution%cell_support_offset(ncell+1), &
                     execution%cell_support_ids(size(cell_support_ids)), &
                     execution%trace_support_offset(ntrace+1), &
                     execution%trace_support_ids(size(trace_support_ids)))
            execution%cell_support_offset = cell_support_offset
            execution%cell_support_ids = cell_support_ids
            execution%trace_support_offset = trace_support_offset
            execution%trace_support_ids = trace_support_ids
        else
            ! DEFAULT_ENABLED is false, but absence is not authorization: the
            ! explicit flag above remains mandatory and support descriptors
            ! are rejected exactly as in the v68 CPU/GPU control plane.
            if (any_support_present) return
            allocate(execution%cell_support_offset(ncell+1), &
                     execution%cell_support_ids(0), &
                     execution%trace_support_offset(ntrace+1), &
                     execution%trace_support_ids(0))
            execution%cell_support_offset = 0_int64
            execution%trace_support_offset = 0_int64
        end if
        execution%ncell = ncell
        execution%ntrace = ntrace
        execution%semantic_id = trim(semantic_id)
        execution%execution_fingerprint = trim(execution_fingerprint)
        execution%topology_fingerprint = trim(topology_fingerprint)
        execution%maximum_waterdepth_is_explicit = .true.
        execution%maximum_waterdepth_enabled = maximum_waterdepth_enabled
        execution%prepared = .true.
        status = direct_family_v67_ok
    end subroutine braids_resolve_direct_family_execution_v68

    subroutine braids_export_direct_family_execution_v68( &
            execution, semantic_id, execution_fingerprint, &
            topology_fingerprint, maximum_waterdepth_is_explicit, &
            maximum_waterdepth_enabled, ncell, ntrace, status)
        type(braids_direct_family_execution_v68), intent(in) :: execution
        character(len=*), intent(out) :: semantic_id, execution_fingerprint
        character(len=*), intent(out) :: topology_fingerprint
        logical, intent(out) :: maximum_waterdepth_is_explicit
        logical, intent(out) :: maximum_waterdepth_enabled
        integer, intent(out) :: ncell, ntrace, status

        semantic_id = ''
        execution_fingerprint = ''
        topology_fingerprint = ''
        maximum_waterdepth_is_explicit = .false.
        maximum_waterdepth_enabled = .false.
        ncell = 0
        ntrace = 0
        status = direct_family_v67_missing_v68_receipt
        if (.not.execution%prepared) return
        semantic_id = execution%semantic_id
        execution_fingerprint = execution%execution_fingerprint
        topology_fingerprint = execution%topology_fingerprint
        maximum_waterdepth_is_explicit = &
            execution%maximum_waterdepth_is_explicit
        maximum_waterdepth_enabled = execution%maximum_waterdepth_enabled
        ncell = execution%ncell
        ntrace = execution%ntrace
        status = direct_family_v67_ok
    end subroutine braids_export_direct_family_execution_v68

    pure logical function execution_matches_topology(state, topology)
        type(braids_direct_family_state_v67), intent(in) :: state
        type(braids_direct_family_topology_v67), intent(in) :: topology
        execution_matches_topology = state%initialized .and. &
            topology%prepared .and. topology%execution_v68%prepared .and. &
            state%maximum_waterdepth_is_explicit .and. &
            topology%execution_v68%maximum_waterdepth_is_explicit .and. &
            state%maximum_waterdepth_enabled .eqv. &
                topology%execution_v68%maximum_waterdepth_enabled
        if (.not.execution_matches_topology) return
        execution_matches_topology = &
            trim(state%execution_fingerprint) == &
                trim(topology%execution_v68%execution_fingerprint) .and. &
            trim(state%topology_fingerprint) == &
                trim(topology%execution_v68%topology_fingerprint)
    end function execution_matches_topology

    pure logical function supported_wire(wire)
        integer, intent(in) :: wire
        supported_wire = wire == direct_family_v67_wire_soulsby .or. &
            wire == direct_family_v67_wire_soulsby_van_rijn .or. &
            wire == BRAIDS_V67_WIRE_VAN_RIJN_1993 .or. &
            wire == BRAIDS_V67_WIRE_BIJKER .or. &
            wire == BRAIDS_V67_WIRE_VAN_THIEL_VAN_RIJN .or. &
            wire == BRAIDS_V67_WIRE_XBEACH_SOULSBY_VAN_RIJN
    end function supported_wire

    pure logical function supported_identity(identity)
        type(braids_direct_family_method_identity_v67), intent(in) :: identity
        supported_identity = identity%prepared .and. &
            trim(identity%forcing_variant) == 'current_only'
        if (.not.supported_identity) return
        select case (trim(identity%catalogue_entry_id))
        case ('capacity.soulsby')
            supported_identity = identity%wire == &
                direct_family_v67_wire_soulsby .and. &
                trim(identity%implementation_variant_id) == &
                    'capacity.soulsby'
        case ('capacity.soulsby_van_rijn')
            supported_identity = identity%wire == &
                direct_family_v67_wire_soulsby_van_rijn .and. &
                trim(identity%implementation_variant_id) == &
                    'capacity.soulsby_van_rijn'
        case ('capacity.van_rijn_1993')
            supported_identity = identity%wire == &
                BRAIDS_V67_WIRE_VAN_RIJN_1993 .and. &
                trim(identity%implementation_variant_id) == &
                    'capacity.van_rijn_1993'
        case ('capacity.bijker')
            supported_identity = identity%wire == BRAIDS_V67_WIRE_BIJKER .and. &
                trim(identity%implementation_variant_id) == &
                    'capacity.bijker_current'
        case ('capacity.van_thiel_van_rijn')
            supported_identity = identity%wire == &
                BRAIDS_V67_WIRE_VAN_THIEL_VAN_RIJN .and. &
                trim(identity%implementation_variant_id) == &
                    'capacity.van_thiel_van_rijn_2008_current'
        case ('capacity.xbeach_soulsby_van_rijn')
            supported_identity = identity%wire == &
                BRAIDS_V67_WIRE_XBEACH_SOULSBY_VAN_RIJN .and. &
                trim(identity%implementation_variant_id) == &
                    'capacity.xbeach_soulsby_van_rijn_current'
        case default
            supported_identity = .false.
        end select
    end function supported_identity

    subroutine braids_resolve_direct_family_method_v67( &
            catalogue_entry_id, implementation_variant_id, forcing_variant, &
            identity, status)
        character(len=*), intent(in) :: catalogue_entry_id
        character(len=*), intent(in) :: implementation_variant_id
        character(len=*), intent(in) :: forcing_variant
        type(braids_direct_family_method_identity_v67), intent(out) :: identity
        integer, intent(out) :: status

        status = direct_family_v67_unsupported_method
        if (trim(forcing_variant) /= 'current_only') return
        select case (trim(catalogue_entry_id))
        case ('capacity.soulsby')
            if (trim(implementation_variant_id) /= &
                'capacity.soulsby') return
            identity%wire = direct_family_v67_wire_soulsby
        case ('capacity.soulsby_van_rijn')
            if (trim(implementation_variant_id) /= &
                'capacity.soulsby_van_rijn') return
            identity%wire = direct_family_v67_wire_soulsby_van_rijn
        case ('capacity.van_rijn_1993')
            if (trim(implementation_variant_id) /= &
                'capacity.van_rijn_1993') return
            identity%wire = BRAIDS_V67_WIRE_VAN_RIJN_1993
        case ('capacity.bijker')
            if (trim(implementation_variant_id) /= &
                'capacity.bijker_current') return
            identity%wire = BRAIDS_V67_WIRE_BIJKER
        case ('capacity.van_thiel_van_rijn')
            if (trim(implementation_variant_id) /= &
                'capacity.van_thiel_van_rijn_2008_current') return
            identity%wire = BRAIDS_V67_WIRE_VAN_THIEL_VAN_RIJN
        case ('capacity.xbeach_soulsby_van_rijn')
            if (trim(implementation_variant_id) /= &
                'capacity.xbeach_soulsby_van_rijn_current') return
            identity%wire = BRAIDS_V67_WIRE_XBEACH_SOULSBY_VAN_RIJN
        case default
            return
        end select
        identity%catalogue_entry_id = trim(catalogue_entry_id)
        identity%implementation_variant_id = trim(implementation_variant_id)
        identity%forcing_variant = trim(forcing_variant)
        identity%prepared = .true.
        if (.not.supported_identity(identity)) return
        status = direct_family_v67_ok
    end subroutine braids_resolve_direct_family_method_v67

    subroutine braids_export_direct_family_method_v67( &
            identity, catalogue_entry_id, implementation_variant_id, &
            forcing_variant, wire, status)
        type(braids_direct_family_method_identity_v67), intent(in) :: identity
        character(len=*), intent(out) :: catalogue_entry_id
        character(len=*), intent(out) :: implementation_variant_id
        character(len=*), intent(out) :: forcing_variant
        integer, intent(out) :: wire, status
        catalogue_entry_id = ''
        implementation_variant_id = ''
        forcing_variant = ''
        wire = 0
        status = direct_family_v67_unsupported_method
        if (.not.supported_identity(identity)) return
        catalogue_entry_id = trim(identity%catalogue_entry_id)
        implementation_variant_id = trim(identity%implementation_variant_id)
        forcing_variant = trim(identity%forcing_variant)
        wire = identity%wire
        status = direct_family_v67_ok
    end subroutine braids_export_direct_family_method_v67

    pure real(real64) function point_depth(wire, point)
        integer, intent(in) :: wire
        type(braids_direct_family_point_v67), intent(in) :: point
        select case (wire)
        case (direct_family_v67_wire_soulsby)
            point_depth = point%soulsby%depth
        case (direct_family_v67_wire_soulsby_van_rijn)
            point_depth = point%soulsby_van_rijn%depth
        case (BRAIDS_V67_WIRE_VAN_RIJN_1993)
            point_depth = point%van_rijn_1993%depth
        case (BRAIDS_V67_WIRE_BIJKER)
            point_depth = point%bijker%depth
        case (BRAIDS_V67_WIRE_VAN_THIEL_VAN_RIJN, &
              BRAIDS_V67_WIRE_XBEACH_SOULSBY_VAN_RIJN)
            point_depth = point%modern%depth
        case default
            point_depth = -1.0_real64
        end select
    end function point_depth

    pure subroutine derive_active_fraction(state, status)
        type(braids_direct_family_state_v67), intent(inout) :: state
        integer, intent(out) :: status
        integer :: cell
        real(real64), allocatable :: volume(:)
        real(real64) :: total

        status = direct_family_v67_ok
        allocate(volume(state%nfrac))
        do cell = 1, state%ncell
            volume = state%bed_mass_kg_m2(:,cell)/state%bulk_density_kg_m3
            total = sum(volume)
            if (.not.ieee_is_finite(total) .or. total <= 0.0_real64) then
                status = direct_family_v67_invalid_input
                deallocate(volume)
                return
            end if
            state%active_fraction(:,cell) = volume/total
        end do
        deallocate(volume)
    end subroutine derive_active_fraction

    subroutine braids_prepare_direct_family_topology_v67( &
            ncell, nface, owner, neighbour, role, normal, length_m, ninc, &
            cell_face_offsets, cell_faces, nboundary, boundary_faces, &
            execution_v68, runtime_topology_fingerprint, topology, status)
        integer, intent(in) :: ncell, nface, ninc, nboundary
        integer, intent(in) :: owner(nface), neighbour(nface), role(nface)
        real(real64), intent(in) :: normal(2,nface), length_m(nface)
        integer, intent(in) :: cell_face_offsets(ncell+1), cell_faces(ninc)
        integer, intent(in) :: boundary_faces(nboundary)
        type(braids_direct_family_execution_v68), intent(in) :: execution_v68
        character(len=*), intent(in) :: runtime_topology_fingerprint
        type(braids_direct_family_topology_v67), intent(out) :: topology
        integer, intent(out) :: status
        integer :: projection_status, ntrace

        status = direct_family_v67_missing_v68_receipt
        if (.not.execution_v68%prepared .or. &
            .not.execution_v68%maximum_waterdepth_is_explicit) return
        status = direct_family_v67_v68_receipt_mismatch
        ntrace = count(role == scalar_live_face_trace_inlet .or. &
                       role == scalar_live_face_trace_outlet)
        if (execution_v68%ncell /= ncell .or. &
            execution_v68%ntrace /= ntrace .or. &
            trim(runtime_topology_fingerprint) /= &
                trim(execution_v68%topology_fingerprint)) return
        status = direct_family_v67_invalid_topology
        call braids_prepare_scalar_flow_topology_v65( &
            ncell, nface, owner, neighbour, role, normal, length_m, ninc, &
            cell_face_offsets, cell_faces, nboundary, boundary_faces, &
            topology%projection, projection_status)
        if (projection_status /= scalar_live_ok) return
        topology%ncell = ncell
        topology%nface = nface
        topology%nboundary = nboundary
        allocate(topology%owner(nface), topology%neighbour(nface), &
                 topology%role(nface), topology%normal(2,nface), &
                 topology%length_m(nface))
        topology%owner = owner
        topology%neighbour = neighbour
        topology%role = role
        topology%normal = normal
        topology%length_m = length_m
        topology%execution_v68 = execution_v68
        topology%prepared = .true.
        status = direct_family_v67_ok
    end subroutine braids_prepare_direct_family_topology_v67

    pure subroutine inject_source_order_hydraulics_v68( &
            wire, physical_depth, producer_u, producer_v, producer_height, &
            point, status)
        integer, intent(in) :: wire
        real(real64), intent(in) :: physical_depth, producer_u, producer_v
        real(real64), intent(in) :: producer_height
        type(braids_direct_family_point_v67), intent(inout) :: point
        integer, intent(out) :: status

        status = direct_family_v67_ok
        select case (wire)
        case (direct_family_v67_wire_soulsby)
            point%soulsby%depth = physical_depth
            point%soulsby%u = producer_u
            point%soulsby%v = producer_v
        case (direct_family_v67_wire_soulsby_van_rijn)
            point%soulsby_van_rijn%depth = physical_depth
            point%soulsby_van_rijn%u = producer_u
            point%soulsby_van_rijn%v = producer_v
        case (BRAIDS_V67_WIRE_VAN_RIJN_1993)
            point%van_rijn_1993%depth = physical_depth
            point%van_rijn_1993%velocity_x = producer_u
            point%van_rijn_1993%velocity_y = producer_v
            point%van_rijn_1993%observation_height = producer_height
        case (BRAIDS_V67_WIRE_BIJKER)
            point%bijker%depth = physical_depth
            point%bijker%u = producer_u
            point%bijker%v = producer_v
        case (BRAIDS_V67_WIRE_VAN_THIEL_VAN_RIJN, &
              BRAIDS_V67_WIRE_XBEACH_SOULSBY_VAN_RIJN)
            point%modern%depth = physical_depth
            point%modern%u = producer_u
            point%modern%v = producer_v
        case default
            status = direct_family_v67_unsupported_method
        end select
    end subroutine inject_source_order_hydraulics_v68

    subroutine prepare_family_points_v68( &
            nfrac, npoint, method_identity, input_point, hydraulic, &
            support_offset, support_depth, support_active, &
            maximum_waterdepth_enabled, direction_epsilon, &
            effective_direction_epsilon, output_point, physical_depth, &
            observation_depth, producer_velocity, producer_height, &
            preprocessor_application_count, status)
        integer, intent(in) :: nfrac, npoint
        type(braids_direct_family_method_identity_v67), intent(in) :: &
            method_identity(nfrac)
        type(braids_direct_family_point_v67), intent(in) :: &
            input_point(nfrac,npoint)
        type(braids_direct_family_hydraulic_point_v68), intent(in) :: &
            hydraulic(npoint)
        integer(int64), intent(in) :: support_offset(npoint+1)
        real(real64), intent(in) :: support_depth(:)
        integer, intent(in) :: support_active(:)
        logical, intent(in) :: maximum_waterdepth_enabled
        real(real64), intent(in) :: direction_epsilon
        real(real64), intent(in) :: effective_direction_epsilon
        type(braids_direct_family_point_v67), intent(out) :: &
            output_point(nfrac,npoint)
        real(real64), intent(out) :: physical_depth(npoint)
        real(real64), intent(out) :: observation_depth(npoint)
        real(real64), intent(out) :: producer_velocity(2,nfrac,npoint)
        real(real64), intent(out) :: producer_height(nfrac,npoint)
        integer, intent(out) :: preprocessor_application_count, status
        real(real64), allocatable :: transport_u(:), transport_v(:)
        real(real64), allocatable :: effective_speed_input(:)
        real(real64), allocatable :: characteristic_u(:), characteristic_v(:)
        real(real64), allocatable :: characteristic_speed(:)
        real(real64), allocatable :: observation_height(:)
        real(real64), allocatable :: effective_u(:), effective_v(:)
        real(real64), allocatable :: effective_speed_output(:)
        integer, allocatable :: active(:), point_status(:)
        integer :: point, fraction, selector_status, injection_status
        integer :: effect_code, switch_value
        real(real64) :: selected_u, selected_v, selected_height

        status = direct_family_v67_v68_preprocess_failure
        preprocessor_application_count = 0
        physical_depth = 0.0_real64
        observation_depth = 0.0_real64
        producer_velocity = 0.0_real64
        producer_height = 0.0_real64
        if (nfrac <= 0 .or. npoint <= 0 .or. &
            size(support_depth) /= size(support_active) .or. &
            .not.valid_support_offsets(npoint,size(support_depth), &
                support_offset)) return
        if (.not.ieee_is_finite(direction_epsilon) .or. &
            direction_epsilon <= 0.0_real64 .or. &
            .not.ieee_is_finite(effective_direction_epsilon) .or. &
            effective_direction_epsilon <= 0.0_real64) return

        allocate(transport_u(npoint),transport_v(npoint), &
                 effective_speed_input(npoint),active(npoint), &
                 characteristic_u(npoint),characteristic_v(npoint), &
                 characteristic_speed(npoint),observation_height(npoint), &
                 effective_u(npoint),effective_v(npoint), &
                 effective_speed_output(npoint),point_status(npoint))
        do point = 1, npoint
            physical_depth(point) = hydraulic(point)%physical_depth
            transport_u(point) = hydraulic(point)%transport_u
            transport_v(point) = hydraulic(point)%transport_v
            effective_speed_input(point) = hydraulic(point)%effective_speed
            active(point) = hydraulic(point)%active
        end do
        if (any(active /= 1) .or. any(.not.ieee_is_finite(physical_depth)) .or. &
            any(physical_depth <= 0.0_real64)) then
            status = direct_family_v67_not_fully_wet
            return
        end if
        switch_value = merge(1,0,maximum_waterdepth_enabled)

        ! This is the sole dwnvel/effective-direction call in the family
        ! lifecycle.  It operates over the hydraulic point batch before the
        ! fraction loop, exactly matching EROSED source order.
        call braids_prepare_direct_vector_hydraulics_v68( &
            npoint,size(support_depth),physical_depth,transport_u,transport_v, &
            effective_speed_input,active,support_offset,support_depth, &
            support_active,switch_value,direction_epsilon, &
            effective_direction_epsilon,observation_depth,characteristic_u, &
            characteristic_v,characteristic_speed,observation_height, &
            effective_u,effective_v,effective_speed_output,point_status, &
            preprocessor_application_count)
        if (preprocessor_application_count /= 1 .or. &
            any(point_status /= 0)) return

        output_point = input_point
        do point = 1, npoint
            do fraction = 1, nfrac
                call braids_select_direct_vector_velocity_v68( &
                    method_identity(fraction)%wire,characteristic_u(point), &
                    characteristic_v(point),effective_u(point), &
                    effective_v(point),observation_height(point),selected_u, &
                    selected_v,selected_height,effect_code,selector_status)
                if (selector_status /= 0) return
                call inject_source_order_hydraulics_v68( &
                    method_identity(fraction)%wire,physical_depth(point), &
                    selected_u,selected_v,selected_height, &
                    output_point(fraction,point),injection_status)
                if (injection_status /= direct_family_v67_ok) return
                producer_velocity(:,fraction,point) = [selected_u,selected_v]
                producer_height(fraction,point) = selected_height
            end do
        end do
        status = direct_family_v67_ok
    end subroutine prepare_family_points_v68

    pure subroutine run_soulsby_pair_for_face( &
            identity, point, bed_factor, suspended_factor, speed_epsilon, &
            composed_bed_mass, q4_concentration, status)
        type(braids_direct_family_method_identity_v67), intent(in) :: identity
        type(braids_direct_family_point_v67), intent(in) :: point
        real(real64), intent(in) :: bed_factor, suspended_factor, speed_epsilon
        real(real64), intent(out) :: composed_bed_mass(2),q4_concentration
        integer, intent(out) :: status
        real(real64) :: raw_bed_mass(2),raw_suspended_mass(2)
        real(real64) :: zero_concentration,tau_ratio
        real(real64) :: shields,critical_shields,critical_speed
        real(real64) :: volume_x,volume_y,velocity(2),depth
        integer :: output_kind,route_status
        logical :: producer_valid,route_valid
        character(len=32) :: composer_method

        composed_bed_mass = 0.0_real64
        q4_concentration = 0.0_real64
        raw_bed_mass = 0.0_real64
        raw_suspended_mass = 0.0_real64
        zero_concentration = 0.0_real64
        status = direct_family_v67_producer_failure
        select case (identity%wire)
        case (direct_family_v67_wire_soulsby)
            composer_method = 'soulsby_current'
            depth = point%soulsby%depth
            velocity = [point%soulsby%u,point%soulsby%v]
            call braids_soulsby_current( &
                point%soulsby%u,point%soulsby%v,point%soulsby%depth, &
                point%soulsby%d50,point%soulsby%solid_density, &
                point%soulsby%water_density,point%soulsby%gravity, &
                point%soulsby%kinematic_viscosity, &
                point%soulsby%calibration,point%soulsby%model_index, &
                point%soulsby%ratio_d50_z0,point%soulsby%von_karman, &
                volume_x,volume_y,tau_ratio,producer_valid)
            if (.not.producer_valid) return
            ! trab12 returns solid volume.  This is the one positive-TraFrm
            ! volume-to-mass conversion owned by eqtran in the reference.
            raw_bed_mass = point%soulsby%solid_density*[volume_x,volume_y]
        case (direct_family_v67_wire_soulsby_van_rijn)
            composer_method = 'soulsby_van_rijn_current'
            depth = point%soulsby_van_rijn%depth
            velocity = [point%soulsby_van_rijn%u, &
                        point%soulsby_van_rijn%v]
            call braids_soulsby_van_rijn_current( &
                point%soulsby_van_rijn%depth, &
                point%soulsby_van_rijn%u,point%soulsby_van_rijn%v, &
                point%soulsby_van_rijn%d50, &
                point%soulsby_van_rijn%solid_density, &
                point%soulsby_van_rijn%water_density, &
                point%soulsby_van_rijn%gravity, &
                point%soulsby_van_rijn%kinematic_viscosity, &
                point%soulsby_van_rijn%calibration, &
                point%soulsby_van_rijn%d90_d50, &
                point%soulsby_van_rijn%roughness_z0,raw_bed_mass, &
                raw_suspended_mass,shields,critical_shields, &
                critical_speed,producer_valid)
            if (.not.producer_valid) return
            ! trab11's established native producer already returns solid mass;
            ! applying density here would be a second, invalid conversion.
        case default
            status = direct_family_v67_unsupported_method
            return
        end select
        call braids_compose_direct_vector_route_v66( &
            trim(composer_method), &
            'tra_bedload',depth,velocity,raw_bed_mass,raw_suspended_mass, &
            zero_concentration,bed_factor,suspended_factor,1.0_real64, &
            speed_epsilon,output_kind,composed_bed_mass,q4_concentration, &
            route_valid,route_status)
        if (route_status /= BRAIDS_DV_ROUTE_OK .or. .not.route_valid) then
            status = direct_family_v67_route_failure
            return
        end if
        if (q4_concentration /= 0.0_real64) then
            status = direct_family_v67_combine_requires_q4
            return
        end if
        status = direct_family_v67_ok
    end subroutine run_soulsby_pair_for_face

    subroutine braids_direct_family_produce_v67( &
            nfrac, npoint, method_identity, point, bed_factor, suspended_factor, &
            speed_epsilon, composed_bed_mass, q4_concentration, &
            status, observed_threads, bed_factor_application_count, &
            suspended_factor_application_count, &
            surface_fraction_application_count)
        integer, intent(in) :: nfrac, npoint
        type(braids_direct_family_method_identity_v67), intent(in) :: &
            method_identity(nfrac)
        type(braids_direct_family_point_v67), intent(in) :: point(nfrac,npoint)
        real(real64), intent(in) :: bed_factor(nfrac)
        real(real64), intent(in) :: suspended_factor(nfrac)
        real(real64), intent(in) :: speed_epsilon(nfrac)
        real(real64), intent(out) :: composed_bed_mass(2,nfrac,npoint)
        real(real64), intent(out) :: q4_concentration(nfrac,npoint)
        integer, intent(out) :: status, observed_threads
        integer, intent(out) :: bed_factor_application_count
        integer, intent(out) :: suspended_factor_application_count
        integer, intent(out) :: surface_fraction_application_count
        integer, allocatable :: local_status(:,:)
        integer :: fraction, item
        type(braids_direct_route_controls_v67) :: controls
        type(braids_direct_producer_result_v67) :: result
        real(real64) :: direct_bed(2),direct_q4
        integer :: direct_status

        composed_bed_mass = 0.0_real64
        q4_concentration = 0.0_real64
        status = direct_family_v67_ok
        observed_threads = 1
        bed_factor_application_count = 0
        suspended_factor_application_count = 0
        surface_fraction_application_count = -1
        if (nfrac <= 0 .or. npoint <= 0) then
            status = direct_family_v67_invalid_shape
            return
        end if
        if (any(.not.ieee_is_finite(bed_factor)) .or. &
            any(.not.ieee_is_finite(suspended_factor)) .or. &
            any(.not.ieee_is_finite(speed_epsilon)) .or. &
            any(bed_factor < 0.0_real64) .or. &
            any(suspended_factor < 0.0_real64) .or. &
            any(speed_epsilon <= 0.0_real64)) then
            status = direct_family_v67_invalid_input
            return
        end if
        do fraction = 1, nfrac
            if (.not.supported_identity(method_identity(fraction))) then
                status = direct_family_v67_unsupported_method
                return
            end if
            do item = 1, npoint
                if (.not.ieee_is_finite(point_depth( &
                        method_identity(fraction)%wire, &
                        point(fraction,item))) .or. &
                    point_depth(method_identity(fraction)%wire, &
                                point(fraction,item)) <= 0.0_real64) then
                    status = direct_family_v67_not_fully_wet
                    return
                end if
            end do
        end do

        allocate(local_status(nfrac,npoint))
        local_status = direct_family_v67_ok
        !$omp parallel default(shared) private(fraction,item,controls,result, &
        !$omp& direct_bed,direct_q4,direct_status)
        !$omp single
        observed_threads = omp_get_num_threads()
        !$omp end single
        !$omp do collapse(2) schedule(static,256)
        do item = 1, npoint
            do fraction = 1, nfrac
                controls%route = 'tra_bedload'
                controls%bed_factor = bed_factor(fraction)
                controls%suspended_factor = suspended_factor(fraction)
                controls%active_fraction = 0.3141592653589793_real64
                controls%speed_epsilon = speed_epsilon(fraction)
                controls%defer_surface_fraction_to_face = .true.
                if (method_identity(fraction)%wire == &
                        direct_family_v67_wire_soulsby .or. &
                    method_identity(fraction)%wire == &
                        direct_family_v67_wire_soulsby_van_rijn) then
                    call run_soulsby_pair_for_face( &
                        method_identity(fraction),point(fraction,item), &
                        bed_factor(fraction),suspended_factor(fraction), &
                        speed_epsilon(fraction),direct_bed,direct_q4, &
                        direct_status)
                    local_status(fraction,item) = direct_status
                    if (direct_status == direct_family_v67_ok) then
                        composed_bed_mass(:,fraction,item) = direct_bed
                        q4_concentration(fraction,item) = direct_q4
                    end if
                    cycle
                end if
                select case (method_identity(fraction)%wire)
                case (BRAIDS_V67_WIRE_VAN_RIJN_1993)
                    call braids_run_vr1993_adapter_v67( &
                        point(fraction,item)%van_rijn_1993, controls, result)
                case (BRAIDS_V67_WIRE_BIJKER)
                    call braids_run_bijker_adapter_v67( &
                        point(fraction,item)%bijker, controls, result)
                case (BRAIDS_V67_WIRE_VAN_THIEL_VAN_RIJN)
                    call braids_run_van_thiel_van_rijn_adapter_v67( &
                        point(fraction,item)%modern, controls, result)
                case (BRAIDS_V67_WIRE_XBEACH_SOULSBY_VAN_RIJN)
                    call braids_run_xbeach_soulsby_van_rijn_adapter_v67( &
                        point(fraction,item)%modern, controls, result)
                case default
                    local_status(fraction,item) = &
                        direct_family_v67_unsupported_method
                    cycle
                end select
                if (result%producer_status /= 0) then
                    local_status(fraction,item) = &
                        direct_family_v67_producer_failure
                else if (result%route_status /= 0 .or. .not.result%valid) then
                    local_status(fraction,item) = direct_family_v67_route_failure
                else if (result%surface_fraction_application_count /= 0 .or. &
                    trim(result%surface_fraction_owner) /= &
                        'downstream_face_exner_lifecycle') then
                    local_status(fraction,item) = direct_family_v67_route_failure
                else if (result%q4_equilibrium_mass_concentration /= &
                         0.0_real64) then
                    local_status(fraction,item) = &
                        direct_family_v67_combine_requires_q4
                else
                    composed_bed_mass(:,fraction,item) = &
                        result%composed_bed_mass
                    q4_concentration(fraction,item) = &
                        result%q4_equilibrium_mass_concentration
                end if
            end do
        end do
        !$omp end do
        !$omp end parallel

        do item = 1, npoint
            do fraction = 1, nfrac
                if (local_status(fraction,item) /= direct_family_v67_ok) then
                    status = local_status(fraction,item)
                    deallocate(local_status)
                    return
                end if
            end do
        end do
        deallocate(local_status)
        bed_factor_application_count = 1
        suspended_factor_application_count = 1
        surface_fraction_application_count = 0
    end subroutine braids_direct_family_produce_v67

    subroutine braids_initialize_direct_family_state_v67( &
            nfrac, topology, bed_mass_kg_m2, bulk_density_kg_m3, state, status)
        integer, intent(in) :: nfrac
        type(braids_direct_family_topology_v67), intent(in) :: topology
        real(real64), intent(in) :: bed_mass_kg_m2(nfrac,topology%ncell)
        real(real64), intent(in) :: bulk_density_kg_m3(nfrac)
        type(braids_direct_family_state_v67), intent(out) :: state
        integer, intent(out) :: status
        real(real64), allocatable :: zero(:)

        allocate(zero(nfrac))
        zero = 0.0_real64
        call braids_restore_direct_family_state_v67( &
            nfrac, topology, 0, 0.0_real64, bed_mass_kg_m2, &
            bulk_density_kg_m3, zero, zero, &
            topology%execution_v68%execution_fingerprint, &
            topology%execution_v68%topology_fingerprint, &
            topology%execution_v68%maximum_waterdepth_is_explicit, &
            topology%execution_v68%maximum_waterdepth_enabled, state, status)
        deallocate(zero)
    end subroutine braids_initialize_direct_family_state_v67

    subroutine braids_restore_direct_family_state_v67( &
            nfrac, topology, accepted_step, time_s, bed_mass_kg_m2, &
            bulk_density_kg_m3, boundary_signed_kg, boundary_absolute_kg, &
            checkpoint_execution_fingerprint, checkpoint_topology_fingerprint, &
            checkpoint_maximum_waterdepth_is_explicit, &
            checkpoint_maximum_waterdepth_enabled, state, status)
        integer, intent(in) :: nfrac, accepted_step
        type(braids_direct_family_topology_v67), intent(in) :: topology
        real(real64), intent(in) :: time_s
        real(real64), intent(in) :: &
            bed_mass_kg_m2(nfrac,topology%ncell)
        real(real64), intent(in) :: bulk_density_kg_m3(nfrac)
        real(real64), intent(in) :: boundary_signed_kg(nfrac)
        real(real64), intent(in) :: boundary_absolute_kg(nfrac)
        character(len=*), intent(in) :: checkpoint_execution_fingerprint
        character(len=*), intent(in) :: checkpoint_topology_fingerprint
        logical, intent(in) :: checkpoint_maximum_waterdepth_is_explicit
        logical, intent(in) :: checkpoint_maximum_waterdepth_enabled
        type(braids_direct_family_state_v67), intent(out) :: state
        integer, intent(out) :: status
        real(real64) :: tolerance
        integer :: fraction

        status = direct_family_v67_v68_receipt_mismatch
        if (.not.topology%prepared .or. &
            .not.topology%execution_v68%prepared .or. &
            .not.checkpoint_maximum_waterdepth_is_explicit .or. &
            .not.topology%execution_v68%maximum_waterdepth_is_explicit .or. &
            (checkpoint_maximum_waterdepth_enabled .neqv. &
                topology%execution_v68%maximum_waterdepth_enabled) .or. &
            trim(checkpoint_execution_fingerprint) /= &
                trim(topology%execution_v68%execution_fingerprint) .or. &
            trim(checkpoint_topology_fingerprint) /= &
                trim(topology%execution_v68%topology_fingerprint)) return
        status = direct_family_v67_invalid_input
        if (nfrac <= 0 .or. topology%ncell <= 0 .or. accepted_step < 0 .or. &
            accepted_step > direct_family_v67_max_accepted_steps .or. &
            .not.ieee_is_finite(time_s) .or. time_s < 0.0_real64 .or. &
            any(.not.ieee_is_finite(bed_mass_kg_m2)) .or. &
            any(.not.ieee_is_finite(bulk_density_kg_m3)) .or. &
            any(.not.ieee_is_finite(boundary_signed_kg)) .or. &
            any(.not.ieee_is_finite(boundary_absolute_kg)) .or. &
            any(bed_mass_kg_m2 < 0.0_real64) .or. &
            any(bulk_density_kg_m3 <= 0.0_real64) .or. &
            any(boundary_absolute_kg < 0.0_real64)) return
        do fraction = 1, nfrac
            tolerance = 2048.0_real64*epsilon(1.0_real64)*max( &
                1.0_real64, boundary_absolute_kg(fraction))
            if (boundary_absolute_kg(fraction)+tolerance < &
                abs(boundary_signed_kg(fraction))) return
        end do
        state%nfrac = nfrac
        state%ncell = topology%ncell
        state%accepted_step = accepted_step
        state%time_s = time_s
        allocate(state%bed_mass_kg_m2(nfrac,topology%ncell), &
                 state%bulk_density_kg_m3(nfrac), &
                 state%active_fraction(nfrac,topology%ncell), &
                 state%boundary_signed_kg(nfrac), &
                 state%boundary_absolute_kg(nfrac))
        state%bed_mass_kg_m2 = bed_mass_kg_m2
        state%bulk_density_kg_m3 = bulk_density_kg_m3
        state%boundary_signed_kg = boundary_signed_kg
        state%boundary_absolute_kg = boundary_absolute_kg
        state%maximum_waterdepth_is_explicit = &
            checkpoint_maximum_waterdepth_is_explicit
        state%maximum_waterdepth_enabled = &
            checkpoint_maximum_waterdepth_enabled
        state%execution_fingerprint = trim(checkpoint_execution_fingerprint)
        state%topology_fingerprint = trim(checkpoint_topology_fingerprint)
        call derive_active_fraction(state, status)
        if (status /= direct_family_v67_ok) return
        state%initialized = .true.
    end subroutine braids_restore_direct_family_state_v67

    subroutine braids_export_direct_family_state_v67( &
            state, accepted_step, time_s, bed_mass_kg_m2, &
            bulk_density_kg_m3, active_fraction, boundary_signed_kg, &
            boundary_absolute_kg, status)
        type(braids_direct_family_state_v67), intent(in) :: state
        integer, intent(out) :: accepted_step, status
        real(real64), intent(out) :: time_s
        real(real64), allocatable, intent(out) :: bed_mass_kg_m2(:,:)
        real(real64), allocatable, intent(out) :: bulk_density_kg_m3(:)
        real(real64), allocatable, intent(out) :: active_fraction(:,:)
        real(real64), allocatable, intent(out) :: boundary_signed_kg(:)
        real(real64), allocatable, intent(out) :: boundary_absolute_kg(:)

        accepted_step = -1
        time_s = -1.0_real64
        status = direct_family_v67_invalid_input
        if (.not.state%initialized) return
        accepted_step = state%accepted_step
        time_s = state%time_s
        allocate(bed_mass_kg_m2(state%nfrac,state%ncell), &
                 bulk_density_kg_m3(state%nfrac), &
                 active_fraction(state%nfrac,state%ncell), &
                 boundary_signed_kg(state%nfrac), &
                 boundary_absolute_kg(state%nfrac))
        bed_mass_kg_m2 = state%bed_mass_kg_m2
        bulk_density_kg_m3 = state%bulk_density_kg_m3
        active_fraction = state%active_fraction
        boundary_signed_kg = state%boundary_signed_kg
        boundary_absolute_kg = state%boundary_absolute_kg
        status = direct_family_v67_ok
    end subroutine braids_export_direct_family_state_v67

    subroutine braids_export_direct_family_state_identity_v68( &
            state, execution_fingerprint, topology_fingerprint, &
            maximum_waterdepth_is_explicit, maximum_waterdepth_enabled, &
            status)
        type(braids_direct_family_state_v67), intent(in) :: state
        character(len=*), intent(out) :: execution_fingerprint
        character(len=*), intent(out) :: topology_fingerprint
        logical, intent(out) :: maximum_waterdepth_is_explicit
        logical, intent(out) :: maximum_waterdepth_enabled
        integer, intent(out) :: status

        execution_fingerprint = ''
        topology_fingerprint = ''
        maximum_waterdepth_is_explicit = .false.
        maximum_waterdepth_enabled = .false.
        status = direct_family_v67_invalid_input
        if (.not.state%initialized) return
        execution_fingerprint = state%execution_fingerprint
        topology_fingerprint = state%topology_fingerprint
        maximum_waterdepth_is_explicit = &
            state%maximum_waterdepth_is_explicit
        maximum_waterdepth_enabled = state%maximum_waterdepth_enabled
        status = direct_family_v67_ok
    end subroutine braids_export_direct_family_state_identity_v68

    subroutine braids_begin_direct_family_trial_v67( &
            nfrac, topology, method_identity, transport_class, cell_point, &
            trace_point, cell_hydraulic, trace_hydraulic, bed_factor, &
            suspended_factor, speed_epsilon, direction_epsilon, &
            effective_direction_epsilon, state, cell_area_m2, dt_s, &
            morphological_factor, trial, status, cell_support_depth, &
            cell_support_active, trace_support_depth, trace_support_active, &
            positive_taurat_contract_fingerprint, islope2_downhill, &
            islope2_alfabs, islope2_alfabn)
        integer, intent(in) :: nfrac
        type(braids_direct_family_topology_v67), intent(in) :: topology
        type(braids_direct_family_method_identity_v67), intent(in) :: &
            method_identity(nfrac)
        integer, intent(in) :: transport_class(nfrac)
        type(braids_direct_family_point_v67), intent(in) :: &
            cell_point(nfrac,topology%ncell)
        type(braids_direct_family_point_v67), intent(in) :: &
            trace_point(nfrac,topology%nface)
        type(braids_direct_family_hydraulic_point_v68), intent(in) :: &
            cell_hydraulic(topology%ncell)
        type(braids_direct_family_hydraulic_point_v68), intent(in) :: &
            trace_hydraulic(topology%nface)
        real(real64), intent(in) :: bed_factor(nfrac)
        real(real64), intent(in) :: suspended_factor(nfrac)
        real(real64), intent(in) :: speed_epsilon(nfrac)
        real(real64), intent(in) :: direction_epsilon
        real(real64), intent(in) :: effective_direction_epsilon
        type(braids_direct_family_state_v67), intent(in) :: state
        real(real64), intent(in) :: cell_area_m2(topology%ncell)
        real(real64), intent(in) :: dt_s, morphological_factor
        type(braids_direct_family_trial_v67), intent(out) :: trial
        integer, intent(out) :: status
        real(real64), intent(in), optional :: cell_support_depth(:)
        integer, intent(in), optional :: cell_support_active(:)
        real(real64), intent(in), optional :: trace_support_depth(:)
        integer, intent(in), optional :: trace_support_active(:)
        character(len=*), intent(in), optional :: &
            positive_taurat_contract_fingerprint
        real(real64), intent(in), optional :: islope2_downhill(:,:)
        real(real64), intent(in), optional :: islope2_alfabs, islope2_alfabn
        type(braids_direct_family_point_v67), allocatable :: prepared_cell(:,:)
        type(braids_direct_family_point_v67), allocatable :: packed_trace(:,:)
        type(braids_direct_family_point_v67), allocatable :: &
            packed_trace_input(:,:)
        type(braids_direct_family_hydraulic_point_v68), allocatable :: &
            packed_trace_hydraulic(:)
        real(real64), allocatable :: cell_bed(:,:,:), trace_bed(:,:,:)
        real(real64), allocatable :: q4(:,:), packed_bed(:,:,:), packed_q4(:,:)
        real(real64), allocatable :: cell_support_values(:)
        real(real64), allocatable :: trace_support_values(:)
        integer, allocatable :: cell_support_flags(:), trace_support_flags(:)
        real(real64), allocatable :: trace_physical_depth(:)
        real(real64), allocatable :: trace_observation_depth(:)
        real(real64), allocatable :: trace_producer_velocity(:,:,:)
        real(real64), allocatable :: trace_producer_height(:,:)
        real(real64), allocatable :: total_direction(:,:,:)
        real(real64), allocatable :: downhill(:,:), total_current_ust2(:)
        real(real64), allocatable :: water_density(:), face_depth(:,:)
        real(real64), allocatable :: d50(:,:), mean_diameter(:,:), hiding(:,:)
        integer, allocatable :: offsets(:), indices(:)
        real(real64), allocatable :: weights(:)
        integer :: ncell, nface, ntrace, face, item, fraction, islope
        integer :: producer_status, face_status, exner_status
        integer :: cell_bed_count, cell_sus_count, cell_surface_count
        integer :: trace_bed_count, trace_sus_count, trace_surface_count
        integer :: active_count, inventory_count, derive_status
        integer :: preprocess_status
        logical :: all_support_present, any_support_present
        logical :: all_slope_present, any_slope_present
        real(real64) :: alfabs_value, alfabn_value

        status = direct_family_v67_invalid_input
        ncell = topology%ncell
        nface = topology%nface
        if (.not.topology%prepared .or. nfrac <= 0 .or. &
            .not.state%initialized .or. state%nfrac /= nfrac .or. &
            state%ncell /= ncell) return
        if (.not.execution_matches_topology(state,topology)) then
            status = direct_family_v67_state_mismatch
            return
        end if
        if (state%accepted_step >= direct_family_v67_max_accepted_steps) then
            status = direct_family_v67_step_limit
            return
        end if
        if (any(transport_class == direct_family_v67_tra_combine)) then
            status = direct_family_v67_combine_requires_q4
            return
        end if
        if (any(transport_class /= direct_family_v67_tra_bedload)) then
            status = direct_family_v67_invalid_input
            return
        end if
        if (.not.ieee_is_finite(dt_s) .or. dt_s <= 0.0_real64 .or. &
            .not.ieee_is_finite(morphological_factor) .or. &
            morphological_factor < 0.0_real64 .or. &
            any(.not.ieee_is_finite(cell_area_m2)) .or. &
            any(cell_area_m2 <= 0.0_real64)) return

        all_slope_present = present(positive_taurat_contract_fingerprint) .and. &
            present(islope2_downhill) .and. present(islope2_alfabs) .and. &
            present(islope2_alfabn)
        any_slope_present = present(positive_taurat_contract_fingerprint) .or. &
            present(islope2_downhill) .or. present(islope2_alfabs) .or. &
            present(islope2_alfabn)
        islope = 0
        alfabs_value = 1.0_real64
        alfabn_value = 1.0_real64
        if (any_slope_present) then
            status = direct_family_v69_taurat_receipt_mismatch
            if (.not.all_slope_present) return
            if (trim(positive_taurat_contract_fingerprint) /= &
                    direct_family_v69_taurat_contract_fingerprint .or. &
                size(islope2_downhill,1) /= 2 .or. &
                size(islope2_downhill,2) /= nface .or. &
                any(.not.ieee_is_finite(islope2_downhill)) .or. &
                .not.ieee_is_finite(islope2_alfabs) .or. &
                .not.ieee_is_finite(islope2_alfabn)) return
            do fraction = 1, nfrac
                if (method_identity(fraction)%wire == &
                        BRAIDS_V67_WIRE_VAN_RIJN_1993) then
                    status = direct_family_v69_live_taurat_required
                    return
                end if
            end do
            if (state%accepted_step >= 4) then
                status = direct_family_v67_step_limit
                return
            end if
            islope = 2
            alfabs_value = islope2_alfabs
            alfabn_value = islope2_alfabn
        end if

        all_support_present = present(cell_support_depth) .and. &
            present(cell_support_active) .and. present(trace_support_depth) .and. &
            present(trace_support_active)
        any_support_present = present(cell_support_depth) .or. &
            present(cell_support_active) .or. present(trace_support_depth) .or. &
            present(trace_support_active)
        if (topology%execution_v68%maximum_waterdepth_enabled) then
            if (.not.all_support_present) then
                status = direct_family_v67_v68_preprocess_failure
                return
            end if
            if (size(cell_support_depth) /= &
                    size(topology%execution_v68%cell_support_ids) .or. &
                size(cell_support_active) /= &
                    size(topology%execution_v68%cell_support_ids) .or. &
                size(trace_support_depth) /= &
                    size(topology%execution_v68%trace_support_ids) .or. &
                size(trace_support_active) /= &
                    size(topology%execution_v68%trace_support_ids)) then
                status = direct_family_v67_v68_preprocess_failure
                return
            end if
            allocate(cell_support_values(size(cell_support_depth)), &
                     cell_support_flags(size(cell_support_active)), &
                     trace_support_values(size(trace_support_depth)), &
                     trace_support_flags(size(trace_support_active)))
            cell_support_values = cell_support_depth
            cell_support_flags = cell_support_active
            trace_support_values = trace_support_depth
            trace_support_flags = trace_support_active
        else
            if (any_support_present) then
                status = direct_family_v67_v68_receipt_mismatch
                return
            end if
            allocate(cell_support_values(0),cell_support_flags(0), &
                     trace_support_values(0),trace_support_flags(0))
        end if

        allocate(prepared_cell(nfrac,ncell),cell_bed(2,nfrac,ncell), &
                 q4(nfrac,ncell),trial%cell_physical_depth(ncell), &
                 trial%cell_observation_depth(ncell), &
                 trial%cell_producer_velocity(2,nfrac,ncell), &
                 trial%cell_producer_observation_height(nfrac,ncell))
        call prepare_family_points_v68( &
            nfrac,ncell,method_identity,cell_point,cell_hydraulic, &
            topology%execution_v68%cell_support_offset,cell_support_values, &
            cell_support_flags, &
            topology%execution_v68%maximum_waterdepth_enabled, &
            direction_epsilon,effective_direction_epsilon,prepared_cell, &
            trial%cell_physical_depth,trial%cell_observation_depth, &
            trial%cell_producer_velocity, &
            trial%cell_producer_observation_height, &
            trial%cell_preprocessor_application_count,preprocess_status)
        if (preprocess_status /= direct_family_v67_ok) then
            status = preprocess_status
            return
        end if
        call braids_direct_family_produce_v67( &
            nfrac, ncell, method_identity, prepared_cell, bed_factor, &
            suspended_factor, speed_epsilon, cell_bed, q4, producer_status, &
            trial%producer_threads, cell_bed_count, cell_sus_count, &
            cell_surface_count)
        if (producer_status /= direct_family_v67_ok) then
            status = producer_status
            return
        end if

        allocate(trace_bed(2,nfrac,nface))
        trace_bed = 0.0_real64
        ntrace = count(topology%role == scalar_live_face_trace_inlet .or. &
                       topology%role == scalar_live_face_trace_outlet)
        trace_bed_count = 1
        trace_sus_count = 1
        trace_surface_count = 0
        trial%trace_preprocessor_application_count = 0
        trial%trace_producer_threads = 1
        if (ntrace > 0) then
            allocate(packed_trace_input(nfrac,ntrace), &
                     packed_trace(nfrac,ntrace), &
                     packed_trace_hydraulic(ntrace), &
                     packed_bed(2,nfrac,ntrace),packed_q4(nfrac,ntrace), &
                     trace_physical_depth(ntrace), &
                     trace_observation_depth(ntrace), &
                     trace_producer_velocity(2,nfrac,ntrace), &
                     trace_producer_height(nfrac,ntrace))
            item = 0
            do face = 1, nface
                if (topology%role(face) == scalar_live_face_trace_inlet .or. &
                    topology%role(face) == scalar_live_face_trace_outlet) then
                    item = item+1
                    packed_trace_input(:,item) = trace_point(:,face)
                    packed_trace_hydraulic(item) = trace_hydraulic(face)
                end if
            end do
            call prepare_family_points_v68( &
                nfrac,ntrace,method_identity,packed_trace_input, &
                packed_trace_hydraulic, &
                topology%execution_v68%trace_support_offset, &
                trace_support_values,trace_support_flags, &
                topology%execution_v68%maximum_waterdepth_enabled, &
                direction_epsilon,effective_direction_epsilon,packed_trace, &
                trace_physical_depth,trace_observation_depth, &
                trace_producer_velocity,trace_producer_height, &
                trial%trace_preprocessor_application_count,preprocess_status)
            if (preprocess_status /= direct_family_v67_ok) then
                status = preprocess_status
                return
            end if
            call braids_direct_family_produce_v67( &
                nfrac, ntrace, method_identity, packed_trace, bed_factor, &
                suspended_factor, speed_epsilon, packed_bed, packed_q4, &
                producer_status, trial%trace_producer_threads, &
                trace_bed_count, trace_sus_count, trace_surface_count)
            if (producer_status /= direct_family_v67_ok) then
                status = producer_status
                return
            end if
            item = 0
            do face = 1, nface
                if (topology%role(face) == scalar_live_face_trace_inlet .or. &
                    topology%role(face) == scalar_live_face_trace_outlet) then
                    item = item+1
                    trace_bed(:,:,face) = packed_bed(:,:,item)
                end if
            end do
            deallocate(packed_trace_input,packed_trace, &
                       packed_trace_hydraulic,packed_bed,packed_q4, &
                       trace_physical_depth,trace_observation_depth, &
                       trace_producer_velocity,trace_producer_height)
        end if
        if (cell_bed_count /= 1 .or. cell_sus_count /= 1 .or. &
            cell_surface_count /= 0 .or. trace_bed_count /= 1 .or. &
            trace_sus_count /= 1 .or. trace_surface_count /= 0 .or. &
            trial%cell_preprocessor_application_count /= 1 .or. &
            (ntrace > 0 .and. &
             trial%trace_preprocessor_application_count /= 1)) then
            status = direct_family_v67_route_failure
            return
        end if

        allocate(total_direction(2,nfrac,ncell))
        total_direction = cell_bed
        allocate(offsets(ncell+1), indices(ncell), weights(ncell))
        do item = 1, ncell+1
            offsets(item) = item
        end do
        do item = 1, ncell
            indices(item) = item
            weights(item) = 1.0_real64
        end do
        allocate(downhill(2,nface), total_current_ust2(nface), &
                 water_density(nface), face_depth(nfrac,nface), &
                 d50(nfrac,nface), mean_diameter(nfrac,nface), &
                 hiding(nfrac,ncell))
        downhill = 0.0_real64
        if (islope == 2) downhill = islope2_downhill
        total_current_ust2 = 0.0_real64
        water_density = 1000.0_real64
        face_depth = 1.0_real64
        d50 = 1.0e-3_real64
        mean_diameter = 1.0e-3_real64
        hiding = 1.0_real64
        allocate(trial%raw_face_candidate(2,nfrac,nface), &
                 trial%corrected_face_candidate(2,nfrac,nface), &
                 trial%raw_donor(nfrac,nface), trial%donor(nfrac,nface), &
                 trial%donor_cell(nfrac,nface))
        call braids_scalar_flow_face_candidates_v65( &
            ncell, nface, nfrac, islope, .true., .false., topology%owner, &
            topology%neighbour, topology%role, topology%normal, cell_bed, &
            total_direction, trace_bed, ncell, offsets, indices, weights, &
            ncell, offsets, indices, weights, downhill, total_current_ust2, &
            9.81_real64, state%bulk_density_kg_m3, water_density, &
            face_depth, d50, mean_diameter, hiding, alfabs_value, &
            alfabn_value, 1.0_real64, 1.0_real64, 1.0_real64, 1.0_real64, &
            1.0_real64, 1.0_real64, trial%raw_face_candidate, &
            trial%corrected_face_candidate, trial%raw_donor, trial%donor, &
            trial%donor_cell, face_status, trial%face_threads)
        if (face_status /= scalar_live_ok) then
            status = direct_family_v67_face_failure
            return
        end if

        allocate(trial%active_fraction_factor(nfrac,ncell), &
                 trial%inventory_fill(nfrac,ncell), &
                 trial%availability(nfrac,ncell), &
                 trial%face_flux_kg_m_s(nfrac,nface), &
                 trial%cell_delta_kg_m2(nfrac,ncell), &
                 trial%boundary_transfer_kg(nfrac), &
                 trial%boundary_absolute_kg(nfrac), &
                 trial%conservation_residual_kg(nfrac))
        trial%candidate%nfrac = nfrac
        trial%candidate%ncell = ncell
        allocate(trial%candidate%bed_mass_kg_m2(nfrac,ncell), &
                 trial%candidate%bulk_density_kg_m3(nfrac), &
                 trial%candidate%active_fraction(nfrac,ncell), &
                 trial%candidate%boundary_signed_kg(nfrac), &
                 trial%candidate%boundary_absolute_kg(nfrac))
        call braids_scalar_flow_project_exner_v65( &
            ncell, nface, nfrac, topology%projection, &
            trial%corrected_face_candidate, trial%donor_cell, &
            state%bed_mass_kg_m2, state%bulk_density_kg_m3, cell_area_m2, &
            dt_s, morphological_factor, trial%active_fraction_factor, &
            trial%inventory_fill, trial%availability, &
            trial%face_flux_kg_m_s, trial%cell_delta_kg_m2, &
            trial%candidate%bed_mass_kg_m2, trial%boundary_transfer_kg, &
            trial%boundary_absolute_kg, trial%conservation_residual_kg, &
            exner_status, trial%exner_threads, active_count, inventory_count)
        if (exner_status /= scalar_live_ok) then
            status = direct_family_v67_exner_failure
            return
        end if

        trial%candidate%bulk_density_kg_m3 = state%bulk_density_kg_m3
        trial%candidate%boundary_signed_kg = state%boundary_signed_kg + &
            trial%boundary_transfer_kg
        trial%candidate%boundary_absolute_kg = state%boundary_absolute_kg + &
            trial%boundary_absolute_kg
        trial%candidate%accepted_step = state%accepted_step+1
        trial%candidate%time_s = state%time_s+dt_s
        trial%candidate%maximum_waterdepth_is_explicit = &
            state%maximum_waterdepth_is_explicit
        trial%candidate%maximum_waterdepth_enabled = &
            state%maximum_waterdepth_enabled
        trial%candidate%execution_fingerprint = state%execution_fingerprint
        trial%candidate%topology_fingerprint = state%topology_fingerprint
        call derive_active_fraction(trial%candidate, derive_status)
        if (derive_status /= direct_family_v67_ok) then
            status = direct_family_v67_exner_failure
            return
        end if
        trial%candidate%initialized = .true.
        trial%nfrac = nfrac
        trial%ncell = ncell
        trial%nface = nface
        trial%parent_step = state%accepted_step
        trial%parent_time_s = state%time_s
        trial%dt_s = dt_s
        allocate(trial%parent_mass(nfrac,ncell), &
                 trial%parent_boundary_signed(nfrac), &
                 trial%parent_boundary_absolute(nfrac))
        trial%parent_mass = state%bed_mass_kg_m2
        trial%parent_boundary_signed = state%boundary_signed_kg
        trial%parent_boundary_absolute = state%boundary_absolute_kg
        trial%bed_factor_application_count = 1
        trial%suspended_factor_application_count = 1
        trial%route_surface_fraction_application_count = 0
        trial%accepted_active_fraction_application_count = active_count
        trial%inventory_fill_application_count = inventory_count
        trial%bed_slope_application_count = merge(1,0,islope == 2)
        trial%taurat_zero_application_count = merge(1,0,islope == 2)
        trial%pending = .true.
        status = direct_family_v67_ok

        deallocate(prepared_cell,cell_support_values,cell_support_flags, &
                   trace_support_values,trace_support_flags,cell_bed,q4, &
                   trace_bed,total_direction,offsets, &
                   indices, weights, downhill, total_current_ust2, &
                   water_density, face_depth, d50, mean_diameter, hiding)
    end subroutine braids_begin_direct_family_trial_v67

    pure logical function parent_matches(state, trial)
        type(braids_direct_family_state_v67), intent(in) :: state
        type(braids_direct_family_trial_v67), intent(in) :: trial
        parent_matches = state%initialized .and. trial%pending .and. &
            state%nfrac == trial%nfrac .and. state%ncell == trial%ncell .and. &
            state%accepted_step == trial%parent_step .and. &
            state%time_s == trial%parent_time_s .and. &
            state%maximum_waterdepth_is_explicit .and. &
            trial%candidate%maximum_waterdepth_is_explicit .and. &
            (state%maximum_waterdepth_enabled .eqv. &
                trial%candidate%maximum_waterdepth_enabled) .and. &
            trim(state%execution_fingerprint) == &
                trim(trial%candidate%execution_fingerprint) .and. &
            trim(state%topology_fingerprint) == &
                trim(trial%candidate%topology_fingerprint)
        if (.not.parent_matches) return
        parent_matches = all(state%bed_mass_kg_m2 == trial%parent_mass) .and. &
            all(state%boundary_signed_kg == &
                trial%parent_boundary_signed) .and. &
            all(state%boundary_absolute_kg == &
                trial%parent_boundary_absolute)
    end function parent_matches

    subroutine braids_commit_direct_family_trial_v67(state, trial, status)
        type(braids_direct_family_state_v67), intent(inout) :: state
        type(braids_direct_family_trial_v67), intent(inout) :: trial
        integer, intent(out) :: status
        if (.not.trial%pending) then
            status = direct_family_v67_trial_not_pending
            return
        end if
        if (.not.parent_matches(state, trial)) then
            status = direct_family_v67_state_mismatch
            return
        end if
        state = trial%candidate
        trial%pending = .false.
        status = direct_family_v67_ok
    end subroutine braids_commit_direct_family_trial_v67

    subroutine braids_rollback_direct_family_trial_v67(state, trial, status)
        type(braids_direct_family_state_v67), intent(in) :: state
        type(braids_direct_family_trial_v67), intent(inout) :: trial
        integer, intent(out) :: status
        if (.not.trial%pending) then
            status = direct_family_v67_trial_not_pending
        else if (.not.parent_matches(state, trial)) then
            status = direct_family_v67_state_mismatch
        else
            trial%pending = .false.
            status = direct_family_v67_ok
        end if
    end subroutine braids_rollback_direct_family_trial_v67

    subroutine braids_cancel_direct_family_trial_v67(state, trial, status)
        type(braids_direct_family_state_v67), intent(in) :: state
        type(braids_direct_family_trial_v67), intent(inout) :: trial
        integer, intent(out) :: status
        call braids_rollback_direct_family_trial_v67(state, trial, status)
    end subroutine braids_cancel_direct_family_trial_v67

    subroutine braids_export_direct_family_trial_v67( &
            trial, pending, raw_face_candidate, corrected_face_candidate, &
            raw_donor, donor, donor_cell, face_flux_kg_m_s, &
            cell_delta_kg_m2, active_fraction_factor, inventory_fill, &
            availability, candidate_mass_kg_m2, boundary_transfer_kg, &
            boundary_absolute_kg, conservation_residual_kg, &
            producer_threads, trace_producer_threads, face_threads, &
            exner_threads, bed_factor_application_count, &
            suspended_factor_application_count, &
            route_surface_fraction_application_count, &
            accepted_active_fraction_application_count, &
            inventory_fill_application_count, status)
        type(braids_direct_family_trial_v67), intent(in) :: trial
        logical, intent(out) :: pending
        real(real64), allocatable, intent(out) :: raw_face_candidate(:,:,:)
        real(real64), allocatable, intent(out) :: corrected_face_candidate(:,:,:)
        integer, allocatable, intent(out) :: raw_donor(:,:), donor(:,:)
        integer, allocatable, intent(out) :: donor_cell(:,:)
        real(real64), allocatable, intent(out) :: face_flux_kg_m_s(:,:)
        real(real64), allocatable, intent(out) :: cell_delta_kg_m2(:,:)
        real(real64), allocatable, intent(out) :: active_fraction_factor(:,:)
        real(real64), allocatable, intent(out) :: inventory_fill(:,:)
        real(real64), allocatable, intent(out) :: availability(:,:)
        real(real64), allocatable, intent(out) :: candidate_mass_kg_m2(:,:)
        real(real64), allocatable, intent(out) :: boundary_transfer_kg(:)
        real(real64), allocatable, intent(out) :: boundary_absolute_kg(:)
        real(real64), allocatable, intent(out) :: conservation_residual_kg(:)
        integer, intent(out) :: producer_threads, trace_producer_threads
        integer, intent(out) :: face_threads, exner_threads
        integer, intent(out) :: bed_factor_application_count
        integer, intent(out) :: suspended_factor_application_count
        integer, intent(out) :: route_surface_fraction_application_count
        integer, intent(out) :: accepted_active_fraction_application_count
        integer, intent(out) :: inventory_fill_application_count, status

        status = direct_family_v67_invalid_input
        pending = .false.
        if (.not.allocated(trial%raw_face_candidate) .or. &
            .not.allocated(trial%candidate%bed_mass_kg_m2)) return
        pending = trial%pending
        allocate(raw_face_candidate(2,trial%nfrac,trial%nface), &
                 corrected_face_candidate(2,trial%nfrac,trial%nface), &
                 raw_donor(trial%nfrac,trial%nface), &
                 donor(trial%nfrac,trial%nface), &
                 donor_cell(trial%nfrac,trial%nface), &
                 face_flux_kg_m_s(trial%nfrac,trial%nface), &
                 cell_delta_kg_m2(trial%nfrac,trial%ncell), &
                 active_fraction_factor(trial%nfrac,trial%ncell), &
                 inventory_fill(trial%nfrac,trial%ncell), &
                 availability(trial%nfrac,trial%ncell), &
                 candidate_mass_kg_m2(trial%nfrac,trial%ncell), &
                 boundary_transfer_kg(trial%nfrac), &
                 boundary_absolute_kg(trial%nfrac), &
                 conservation_residual_kg(trial%nfrac))
        raw_face_candidate = trial%raw_face_candidate
        corrected_face_candidate = trial%corrected_face_candidate
        raw_donor = trial%raw_donor
        donor = trial%donor
        donor_cell = trial%donor_cell
        face_flux_kg_m_s = trial%face_flux_kg_m_s
        cell_delta_kg_m2 = trial%cell_delta_kg_m2
        active_fraction_factor = trial%active_fraction_factor
        inventory_fill = trial%inventory_fill
        availability = trial%availability
        candidate_mass_kg_m2 = trial%candidate%bed_mass_kg_m2
        boundary_transfer_kg = trial%boundary_transfer_kg
        boundary_absolute_kg = trial%boundary_absolute_kg
        conservation_residual_kg = trial%conservation_residual_kg
        producer_threads = trial%producer_threads
        trace_producer_threads = trial%trace_producer_threads
        face_threads = trial%face_threads
        exner_threads = trial%exner_threads
        bed_factor_application_count = trial%bed_factor_application_count
        suspended_factor_application_count = &
            trial%suspended_factor_application_count
        route_surface_fraction_application_count = &
            trial%route_surface_fraction_application_count
        accepted_active_fraction_application_count = &
            trial%accepted_active_fraction_application_count
        inventory_fill_application_count = &
            trial%inventory_fill_application_count
        status = direct_family_v67_ok
    end subroutine braids_export_direct_family_trial_v67

    subroutine braids_export_direct_family_trial_hydraulics_v68( &
            trial, execution_fingerprint, topology_fingerprint, &
            maximum_waterdepth_enabled, cell_preprocessor_application_count, &
            trace_preprocessor_application_count, cell_physical_depth, &
            cell_observation_depth, cell_producer_velocity, &
            cell_producer_observation_height, status)
        type(braids_direct_family_trial_v67), intent(in) :: trial
        character(len=*), intent(out) :: execution_fingerprint
        character(len=*), intent(out) :: topology_fingerprint
        logical, intent(out) :: maximum_waterdepth_enabled
        integer, intent(out) :: cell_preprocessor_application_count
        integer, intent(out) :: trace_preprocessor_application_count
        real(real64), allocatable, intent(out) :: cell_physical_depth(:)
        real(real64), allocatable, intent(out) :: cell_observation_depth(:)
        real(real64), allocatable, intent(out) :: &
            cell_producer_velocity(:,:,:)
        real(real64), allocatable, intent(out) :: &
            cell_producer_observation_height(:,:)
        integer, intent(out) :: status

        execution_fingerprint = ''
        topology_fingerprint = ''
        maximum_waterdepth_enabled = .false.
        cell_preprocessor_application_count = 0
        trace_preprocessor_application_count = 0
        status = direct_family_v67_invalid_input
        if (.not.trial%pending .or. &
            .not.trial%candidate%maximum_waterdepth_is_explicit .or. &
            .not.allocated(trial%cell_physical_depth) .or. &
            .not.allocated(trial%cell_observation_depth) .or. &
            .not.allocated(trial%cell_producer_velocity) .or. &
            .not.allocated(trial%cell_producer_observation_height)) return
        execution_fingerprint = trial%candidate%execution_fingerprint
        topology_fingerprint = trial%candidate%topology_fingerprint
        maximum_waterdepth_enabled = &
            trial%candidate%maximum_waterdepth_enabled
        cell_preprocessor_application_count = &
            trial%cell_preprocessor_application_count
        trace_preprocessor_application_count = &
            trial%trace_preprocessor_application_count
        allocate(cell_physical_depth(trial%ncell), &
                 cell_observation_depth(trial%ncell), &
                 cell_producer_velocity(2,trial%nfrac,trial%ncell), &
                 cell_producer_observation_height(trial%nfrac,trial%ncell))
        cell_physical_depth = trial%cell_physical_depth
        cell_observation_depth = trial%cell_observation_depth
        cell_producer_velocity = trial%cell_producer_velocity
        cell_producer_observation_height = &
            trial%cell_producer_observation_height
        status = direct_family_v67_ok
    end subroutine braids_export_direct_family_trial_hydraulics_v68

    subroutine braids_export_direct_family_trial_taurat_v69( &
            trial, taurat_contract_fingerprint, &
            bed_slope_application_count, taurat_zero_application_count, &
            status)
        type(braids_direct_family_trial_v67), intent(in) :: trial
        character(len=*), intent(out) :: taurat_contract_fingerprint
        integer, intent(out) :: bed_slope_application_count
        integer, intent(out) :: taurat_zero_application_count
        integer, intent(out) :: status

        taurat_contract_fingerprint = ''
        bed_slope_application_count = 0
        taurat_zero_application_count = 0
        status = direct_family_v67_invalid_input
        if (.not.trial%pending .or. &
            trial%bed_slope_application_count /= 1 .or. &
            trial%taurat_zero_application_count /= 1) return
        taurat_contract_fingerprint = &
            direct_family_v69_taurat_contract_fingerprint
        bed_slope_application_count = trial%bed_slope_application_count
        taurat_zero_application_count = trial%taurat_zero_application_count
        status = direct_family_v67_ok
    end subroutine braids_export_direct_family_trial_taurat_v69

end module braids_direct_vector_family_native_v67_module
