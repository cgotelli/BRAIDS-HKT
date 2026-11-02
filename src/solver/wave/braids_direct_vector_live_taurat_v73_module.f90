! SPDX-License-Identifier: AGPL-3.0-or-later
!
! Bounded native/OpenMP Van Rijn 1993 lifecycle carrying the live
! bedbc1993 taubcw/taucr1 output into classic FLOW ISlope=2.  The ratio and
! the bed vector are produced by the same adapter invocation.  No hydraulic
! shear diagnostic is accepted by this API, so none can be substituted.
module braids_direct_vector_live_taurat_v73_module
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use omp_lib, only: omp_get_num_threads
    use braids_direct_producer_adapter_v67_module, only: &
        braids_direct_route_controls_v67, braids_vr1993_input_v67, &
        braids_direct_producer_result_v67, braids_run_vr1993_adapter_v67
    use braids_flow_bed_slope_operator_module, only: &
        braids_flow_bed_slope_direction, flow_slope_ok
    use braids_scalar_flow_live_v65_module, only: &
        scalar_flow_projection_topology_v65, &
        scalar_live_face_internal, scalar_live_face_trace_inlet, &
        scalar_live_face_trace_outlet, scalar_live_face_owner_outlet, &
        scalar_live_face_closed, scalar_live_donor_owner, &
        scalar_live_donor_neighbour, scalar_live_donor_trace, &
        scalar_live_donor_none, scalar_live_ok, &
        braids_prepare_scalar_flow_topology_v65, &
        braids_scalar_flow_face_candidates_v65, &
        braids_scalar_flow_project_exner_v65
    implicit none
    private

    integer, parameter, public :: live_taurat_v73_ok = 0
    integer, parameter, public :: live_taurat_v73_invalid_shape = 730
    integer, parameter, public :: live_taurat_v73_invalid_input = 731
    integer, parameter, public :: live_taurat_v73_invalid_topology = 732
    integer, parameter, public :: live_taurat_v73_source_mismatch = 733
    integer, parameter, public :: live_taurat_v73_producer_failure = 734
    integer, parameter, public :: live_taurat_v73_face_failure = 735
    integer, parameter, public :: live_taurat_v73_exner_failure = 736
    integer, parameter, public :: live_taurat_v73_state_mismatch = 737
    integer, parameter, public :: live_taurat_v73_trial_not_pending = 738
    integer, parameter, public :: live_taurat_v73_step_limit = 739
    integer, parameter, public :: live_taurat_v73_cancelled = 740
    integer, parameter, public :: live_taurat_v73_max_steps = 4
    character(len=64), parameter, public :: live_taurat_v73_contract = &
        'cc0c86f58a8d38513c3793f0c1cf8f45bc25713f9c8f8e6fa4a20763648d6599'

    type, public :: braids_live_taurat_topology_v73
        private
        logical :: prepared = .false.
        integer :: ncell = 0, nface = 0
        integer, allocatable :: owner(:), neighbour(:), role(:)
        real(real64), allocatable :: normal(:,:), length_m(:), cell_area(:)
        type(scalar_flow_projection_topology_v65) :: projection
    end type braids_live_taurat_topology_v73

    type, public :: braids_live_taurat_state_v73
        private
        logical :: initialized = .false.
        integer :: nfrac = 0, ncell = 0, accepted_step = 0
        real(real64) :: time_s = 0.0_real64
        real(real64), allocatable :: bed_mass(:,:), bulk_density(:)
        real(real64), allocatable :: active_fraction(:,:)
        real(real64), allocatable :: boundary_signed(:), boundary_absolute(:)
    end type braids_live_taurat_state_v73

    type, public :: braids_live_taurat_trial_v73
        private
        logical :: pending = .false.
        integer :: nfrac = 0, ncell = 0, nface = 0
        integer :: parent_step = -1
        real(real64) :: parent_time_s = -1.0_real64
        integer :: cell_producer_threads = 1
        integer :: trace_producer_threads = 1
        integer :: raw_face_threads = 1
        integer :: slope_threads = 1
        integer :: exner_threads = 1
        integer :: producer_taurat_application_count = 0
        integer :: bed_slope_application_count = 0
        integer :: active_fraction_application_count = 0
        integer :: inventory_fill_application_count = 0
        real(real64), allocatable :: parent_mass(:,:)
        real(real64), allocatable :: parent_boundary_signed(:)
        real(real64), allocatable :: parent_boundary_absolute(:)
        real(real64), allocatable :: cell_taurat(:,:), trace_taurat(:,:)
        real(real64), allocatable :: face_taurat(:,:)
        real(real64), allocatable :: raw_face(:,:,:), corrected_face(:,:,:)
        integer, allocatable :: raw_donor(:,:), donor(:,:), donor_cell(:,:)
        real(real64), allocatable :: face_flux(:,:), cell_delta(:,:)
        real(real64), allocatable :: inventory_fill(:,:), availability(:,:)
        real(real64), allocatable :: boundary_transfer(:)
        real(real64), allocatable :: boundary_absolute_transfer(:)
        real(real64), allocatable :: conservation_residual(:)
        type(braids_live_taurat_state_v73) :: candidate
    end type braids_live_taurat_trial_v73

    public :: braids_prepare_live_taurat_topology_v73
    public :: braids_restore_live_taurat_state_v73
    public :: braids_initialize_live_taurat_state_v73
    public :: braids_begin_live_taurat_trial_v73
    public :: braids_commit_live_taurat_trial_v73
    public :: braids_rollback_live_taurat_trial_v73
    public :: braids_cancel_live_taurat_trial_v73
    public :: braids_export_live_taurat_state_v73
    public :: braids_export_live_taurat_trial_v73

contains

    subroutine braids_prepare_live_taurat_topology_v73( &
            ncell,nface,owner,neighbour,role,normal,length_m,cell_area, &
            ninc,cell_face_offsets,cell_faces,nboundary,boundary_faces, &
            topology,status)
        integer, intent(in) :: ncell,nface,ninc,nboundary
        integer, intent(in) :: owner(nface),neighbour(nface),role(nface)
        real(real64), intent(in) :: normal(2,nface),length_m(nface)
        real(real64), intent(in) :: cell_area(ncell)
        integer, intent(in) :: cell_face_offsets(ncell+1),cell_faces(ninc)
        integer, intent(in) :: boundary_faces(nboundary)
        type(braids_live_taurat_topology_v73), intent(out) :: topology
        integer, intent(out) :: status
        integer :: nested_status

        status = live_taurat_v73_invalid_input
        if (any(.not.ieee_is_finite(cell_area)) .or. &
                any(cell_area <= 0.0_real64)) return
        call braids_prepare_scalar_flow_topology_v65( &
            ncell,nface,owner,neighbour,role,normal,length_m,ninc, &
            cell_face_offsets,cell_faces,nboundary,boundary_faces, &
            topology%projection,nested_status)
        if (nested_status /= scalar_live_ok) then
            status = live_taurat_v73_invalid_topology
            return
        end if
        topology%ncell = ncell
        topology%nface = nface
        allocate(topology%owner(nface),topology%neighbour(nface), &
                 topology%role(nface),topology%normal(2,nface), &
                 topology%length_m(nface),topology%cell_area(ncell))
        topology%owner = owner
        topology%neighbour = neighbour
        topology%role = role
        topology%normal = normal
        topology%length_m = length_m
        topology%cell_area = cell_area
        topology%prepared = .true.
        status = live_taurat_v73_ok
    end subroutine braids_prepare_live_taurat_topology_v73

    subroutine derive_fraction(state,status)
        type(braids_live_taurat_state_v73), intent(inout) :: state
        integer, intent(out) :: status
        integer :: cell
        real(real64) :: total
        status = live_taurat_v73_exner_failure
        if (.not.allocated(state%active_fraction)) &
            allocate(state%active_fraction(state%nfrac,state%ncell))
        state%active_fraction = 0.0_real64
        do cell = 1,state%ncell
            total = sum(state%bed_mass(:,cell)/state%bulk_density)
            if (.not.ieee_is_finite(total) .or. total <= 0.0_real64) return
            state%active_fraction(:,cell) = &
                (state%bed_mass(:,cell)/state%bulk_density)/total
        end do
        status = live_taurat_v73_ok
    end subroutine derive_fraction

    subroutine braids_restore_live_taurat_state_v73( &
            nfrac,topology,accepted_step,time_s,bed_mass,bulk_density, &
            boundary_signed,boundary_absolute,state,status)
        integer, intent(in) :: nfrac,accepted_step
        type(braids_live_taurat_topology_v73), intent(in) :: topology
        real(real64), intent(in) :: time_s
        real(real64), intent(in) :: bed_mass(nfrac,topology%ncell)
        real(real64), intent(in) :: bulk_density(nfrac)
        real(real64), intent(in) :: boundary_signed(nfrac)
        real(real64), intent(in) :: boundary_absolute(nfrac)
        type(braids_live_taurat_state_v73), intent(out) :: state
        integer, intent(out) :: status

        status = live_taurat_v73_invalid_input
        if (.not.topology%prepared .or. nfrac <= 0 .or. accepted_step < 0 .or. &
                accepted_step > live_taurat_v73_max_steps .or. &
                .not.ieee_is_finite(time_s) .or. time_s < 0.0_real64 .or. &
                any(.not.ieee_is_finite(bed_mass)) .or. &
                any(.not.ieee_is_finite(bulk_density)) .or. &
                any(.not.ieee_is_finite(boundary_signed)) .or. &
                any(.not.ieee_is_finite(boundary_absolute)) .or. &
                any(bed_mass < 0.0_real64) .or. &
                any(bulk_density <= 0.0_real64) .or. &
                any(boundary_absolute < abs(boundary_signed))) return
        state%nfrac = nfrac
        state%ncell = topology%ncell
        state%accepted_step = accepted_step
        state%time_s = time_s
        allocate(state%bed_mass(nfrac,topology%ncell), &
                 state%bulk_density(nfrac), &
                 state%boundary_signed(nfrac), &
                 state%boundary_absolute(nfrac))
        state%bed_mass = bed_mass
        state%bulk_density = bulk_density
        state%boundary_signed = boundary_signed
        state%boundary_absolute = boundary_absolute
        call derive_fraction(state,status)
        if (status /= live_taurat_v73_ok) return
        state%initialized = .true.
    end subroutine braids_restore_live_taurat_state_v73

    subroutine braids_initialize_live_taurat_state_v73( &
            nfrac,topology,bed_mass,bulk_density,state,status)
        integer, intent(in) :: nfrac
        type(braids_live_taurat_topology_v73), intent(in) :: topology
        real(real64), intent(in) :: bed_mass(nfrac,topology%ncell)
        real(real64), intent(in) :: bulk_density(nfrac)
        type(braids_live_taurat_state_v73), intent(out) :: state
        integer, intent(out) :: status
        real(real64), allocatable :: zero(:)
        allocate(zero(nfrac)); zero = 0.0_real64
        call braids_restore_live_taurat_state_v73( &
            nfrac,topology,0,0.0_real64,bed_mass,bulk_density,zero,zero, &
            state,status)
        deallocate(zero)
    end subroutine braids_initialize_live_taurat_state_v73

    subroutine produce_points( &
            nfrac,npoint,points,bed_factor,suspended_factor,speed_epsilon, &
            active,bed,taurat,status,threads)
        integer, intent(in) :: nfrac,npoint
        type(braids_vr1993_input_v67), intent(in) :: points(nfrac,npoint)
        real(real64), intent(in) :: bed_factor(nfrac)
        real(real64), intent(in) :: suspended_factor(nfrac)
        real(real64), intent(in) :: speed_epsilon(nfrac)
        logical, intent(in) :: active(npoint)
        real(real64), intent(out) :: bed(2,nfrac,npoint)
        real(real64), intent(out) :: taurat(nfrac,npoint)
        integer, intent(out) :: status,threads
        integer, allocatable :: local_status(:,:)
        integer :: fraction,item
        type(braids_direct_route_controls_v67) :: controls
        type(braids_direct_producer_result_v67) :: result

        bed = 0.0_real64
        taurat = 0.0_real64
        status = live_taurat_v73_invalid_input
        threads = 1
        if (nfrac <= 0 .or. npoint <= 0 .or. &
                any(.not.ieee_is_finite(bed_factor)) .or. &
                any(.not.ieee_is_finite(suspended_factor)) .or. &
                any(.not.ieee_is_finite(speed_epsilon)) .or. &
                any(bed_factor < 0.0_real64) .or. &
                any(suspended_factor < 0.0_real64) .or. &
                any(speed_epsilon <= 0.0_real64)) return
        allocate(local_status(nfrac,npoint)); local_status = 0
        !$omp parallel default(shared) private(fraction,item,controls,result)
        !$omp single
        threads = omp_get_num_threads()
        !$omp end single
        !$omp do collapse(2) schedule(static,256)
        do item = 1,npoint
            do fraction = 1,nfrac
                if (.not.active(item)) cycle
                controls%route = 'tra_bedload'
                controls%bed_factor = bed_factor(fraction)
                controls%suspended_factor = suspended_factor(fraction)
                controls%active_fraction = 0.2718281828459045_real64
                controls%speed_epsilon = speed_epsilon(fraction)
                controls%defer_surface_fraction_to_face = .true.
                call braids_run_vr1993_adapter_v67( &
                    points(fraction,item),controls,result)
                if (result%producer_status /= 0 .or. &
                        result%route_status /= 0 .or. .not.result%valid .or. &
                        result%surface_fraction_application_count /= 0 .or. &
                        trim(result%surface_fraction_owner) /= &
                            'downstream_face_exner_lifecycle' .or. &
                        .not.ieee_is_finite(result%producer_auxiliary(3)) .or. &
                        result%producer_auxiliary(3) < 0.0_real64) then
                    local_status(fraction,item) = &
                        live_taurat_v73_producer_failure
                else
                    bed(:,fraction,item) = result%composed_bed_mass
                    ! Same invocation, exact producer-owned bedbc1993 output.
                    taurat(fraction,item) = result%producer_auxiliary(3)
                end if
            end do
        end do
        !$omp end do
        !$omp end parallel
        if (any(local_status /= 0)) then
            status = live_taurat_v73_producer_failure
        else
            status = live_taurat_v73_ok
        end if
        deallocate(local_status)
    end subroutine produce_points

    subroutine braids_begin_live_taurat_trial_v73( &
            nfrac,topology,cell_point,trace_point,bed_factor, &
            suspended_factor,speed_epsilon,downhill,alfabs,alfabn, &
            state,dt_s,morphological_factor,taurat_contract,cancelled, &
            trial,status)
        integer, intent(in) :: nfrac
        type(braids_live_taurat_topology_v73), intent(in) :: topology
        type(braids_vr1993_input_v67), intent(in) :: &
            cell_point(nfrac,topology%ncell)
        type(braids_vr1993_input_v67), intent(in) :: &
            trace_point(nfrac,topology%nface)
        real(real64), intent(in) :: bed_factor(nfrac)
        real(real64), intent(in) :: suspended_factor(nfrac)
        real(real64), intent(in) :: speed_epsilon(nfrac)
        real(real64), intent(in) :: downhill(2,topology%nface)
        real(real64), intent(in) :: alfabs,alfabn
        type(braids_live_taurat_state_v73), intent(in) :: state
        real(real64), intent(in) :: dt_s,morphological_factor
        character(len=*), intent(in) :: taurat_contract
        logical, intent(in) :: cancelled
        type(braids_live_taurat_trial_v73), intent(out) :: trial
        integer, intent(out) :: status
        real(real64), allocatable :: cell_bed(:,:,:),trace_bed(:,:,:)
        real(real64), allocatable :: total_direction(:,:,:)
        real(real64), allocatable :: dummy_downhill(:,:),dummy_stress(:)
        real(real64), allocatable :: dummy_water_density(:)
        real(real64), allocatable :: dummy_depth(:,:),dummy_d50(:,:)
        real(real64), allocatable :: dummy_mean(:,:),dummy_hiding(:,:)
        real(real64), allocatable :: weights(:)
        integer, allocatable :: offsets(:),indices(:)
        integer, allocatable :: local_status(:,:)
        logical, allocatable :: cell_active(:),trace_active(:)
        integer :: face,fraction,own,other,nested_status
        integer :: dummy_threads,raw_active,raw_inventory,derive_status
        real(real64) :: corrected(2),normal_value

        status = live_taurat_v73_invalid_input
        if (cancelled) then
            status = live_taurat_v73_cancelled
            return
        end if
        if (trim(taurat_contract) /= live_taurat_v73_contract) then
            status = live_taurat_v73_source_mismatch
            return
        end if
        if (.not.topology%prepared .or. .not.state%initialized .or. &
                state%nfrac /= nfrac .or. state%ncell /= topology%ncell .or. &
                .not.ieee_is_finite(dt_s) .or. dt_s <= 0.0_real64 .or. &
                .not.ieee_is_finite(morphological_factor) .or. &
                morphological_factor < 0.0_real64 .or. &
                any(.not.ieee_is_finite(downhill)) .or. &
                .not.ieee_is_finite(alfabs) .or. &
                .not.ieee_is_finite(alfabn)) return
        if (state%accepted_step >= live_taurat_v73_max_steps) then
            status = live_taurat_v73_step_limit
            return
        end if

        allocate(cell_bed(2,nfrac,topology%ncell), &
                 trace_bed(2,nfrac,topology%nface), &
                 trial%cell_taurat(nfrac,topology%ncell), &
                 trial%trace_taurat(nfrac,topology%nface), &
                 cell_active(topology%ncell),trace_active(topology%nface))
        cell_active = .true.
        trace_active = topology%role == scalar_live_face_trace_inlet .or. &
                       topology%role == scalar_live_face_trace_outlet
        call produce_points( &
            nfrac,topology%ncell,cell_point,bed_factor,suspended_factor, &
            speed_epsilon,cell_active,cell_bed,trial%cell_taurat, &
            status,trial%cell_producer_threads)
        if (status /= live_taurat_v73_ok) return
        call produce_points( &
            nfrac,topology%nface,trace_point,bed_factor,suspended_factor, &
            speed_epsilon,trace_active,trace_bed,trial%trace_taurat, &
            status,trial%trace_producer_threads)
        if (status /= live_taurat_v73_ok) return

        allocate(total_direction(2,nfrac,topology%ncell), &
                 offsets(topology%ncell+1),indices(topology%ncell), &
                 weights(topology%ncell), &
                 dummy_downhill(2,topology%nface), &
                 dummy_stress(topology%nface), &
                 dummy_water_density(topology%nface), &
                 dummy_depth(nfrac,topology%nface), &
                 dummy_d50(nfrac,topology%nface), &
                 dummy_mean(nfrac,topology%nface), &
                 dummy_hiding(nfrac,topology%ncell), &
                 trial%raw_face(2,nfrac,topology%nface), &
                 trial%corrected_face(2,nfrac,topology%nface), &
                 trial%raw_donor(nfrac,topology%nface), &
                 trial%donor(nfrac,topology%nface), &
                 trial%donor_cell(nfrac,topology%nface), &
                 trial%face_taurat(nfrac,topology%nface), &
                 local_status(nfrac,topology%nface))
        total_direction = cell_bed
        do face = 1,topology%ncell+1
            offsets(face) = face
        end do
        do face = 1,topology%ncell
            indices(face) = face; weights(face) = 1.0_real64
        end do
        dummy_downhill = 0.0_real64
        dummy_stress = 0.0_real64
        dummy_water_density = 1000.0_real64
        dummy_depth = 1.0_real64
        dummy_d50 = 1.0e-3_real64
        dummy_mean = 1.0e-3_real64
        dummy_hiding = 1.0_real64
        call braids_scalar_flow_face_candidates_v65( &
            topology%ncell,topology%nface,nfrac,0,.true.,.false., &
            topology%owner,topology%neighbour,topology%role,topology%normal, &
            cell_bed,total_direction,trace_bed,topology%ncell, &
            offsets,indices,weights,topology%ncell,offsets,indices,weights, &
            dummy_downhill,dummy_stress,9.81_real64,state%bulk_density, &
            dummy_water_density,dummy_depth,dummy_d50,dummy_mean, &
            dummy_hiding,alfabs,alfabn,1.0_real64,1.0_real64,1.0_real64, &
            1.0_real64,1.0_real64,1.0_real64,trial%raw_face, &
            trial%corrected_face,trial%raw_donor,trial%donor, &
            trial%donor_cell,nested_status,trial%raw_face_threads)
        if (nested_status /= scalar_live_ok) then
            status = live_taurat_v73_face_failure
            return
        end if

        local_status = live_taurat_v73_ok
        !$omp parallel default(shared) private(face,fraction,own,other, &
        !$omp& corrected,normal_value,nested_status)
        !$omp single
        trial%slope_threads = omp_get_num_threads()
        !$omp end single
        !$omp do collapse(2) schedule(static,256)
        do face = 1,topology%nface
            do fraction = 1,nfrac
                own = topology%owner(face)
                other = topology%neighbour(face)
                select case(topology%role(face))
                case(scalar_live_face_internal)
                    trial%face_taurat(fraction,face) = 0.5_real64*( &
                        trial%cell_taurat(fraction,own) + &
                        trial%cell_taurat(fraction,other))
                case(scalar_live_face_closed)
                    trial%face_taurat(fraction,face) = 0.0_real64
                case default
                    trial%face_taurat(fraction,face) = &
                        trial%cell_taurat(fraction,own)
                end select
                corrected = trial%raw_face(:,fraction,face)
                if (topology%role(face) /= scalar_live_face_closed) then
                    call braids_flow_bed_slope_direction( &
                        2,trial%raw_face(1,fraction,face), &
                        trial%raw_face(2,fraction,face),downhill(1,face), &
                        downhill(2,face),trial%face_taurat(fraction,face), &
                        1.0_real64,1.0e-3_real64,1.0e-3_real64, &
                        1.0_real64,1.0_real64,alfabs,alfabn,1.0_real64, &
                        1.0_real64,1.0_real64,1.0_real64,1.0_real64, &
                        1.0_real64,corrected(1),corrected(2),nested_status)
                    if (nested_status /= flow_slope_ok) then
                        local_status(fraction,face) = &
                            live_taurat_v73_face_failure
                    end if
                end if
                trial%corrected_face(:,fraction,face) = corrected
                normal_value = dot_product(corrected,topology%normal(:,face))
                select case(topology%role(face))
                case(scalar_live_face_internal)
                    if (normal_value < 0.0_real64) then
                        trial%donor(fraction,face) = &
                            scalar_live_donor_neighbour
                        trial%donor_cell(fraction,face) = other
                    else
                        trial%donor(fraction,face) = scalar_live_donor_owner
                        trial%donor_cell(fraction,face) = own
                    end if
                case(scalar_live_face_trace_inlet, &
                     scalar_live_face_trace_outlet)
                    trial%donor(fraction,face) = scalar_live_donor_trace
                    trial%donor_cell(fraction,face) = own
                case(scalar_live_face_owner_outlet)
                    trial%donor(fraction,face) = scalar_live_donor_owner
                    trial%donor_cell(fraction,face) = own
                case(scalar_live_face_closed)
                    trial%donor(fraction,face) = scalar_live_donor_none
                    trial%donor_cell(fraction,face) = 0
                end select
            end do
        end do
        !$omp end do
        !$omp end parallel
        if (any(local_status /= live_taurat_v73_ok)) then
            status = live_taurat_v73_face_failure
            return
        end if

        allocate(trial%inventory_fill(nfrac,topology%ncell), &
                 trial%availability(nfrac,topology%ncell), &
                 trial%face_flux(nfrac,topology%nface), &
                 trial%cell_delta(nfrac,topology%ncell), &
                 trial%boundary_transfer(nfrac), &
                 trial%boundary_absolute_transfer(nfrac), &
                 trial%conservation_residual(nfrac))
        trial%candidate%nfrac = nfrac
        trial%candidate%ncell = topology%ncell
        allocate(trial%candidate%bed_mass(nfrac,topology%ncell), &
                 trial%candidate%bulk_density(nfrac), &
                 trial%candidate%active_fraction(nfrac,topology%ncell), &
                 trial%candidate%boundary_signed(nfrac), &
                 trial%candidate%boundary_absolute(nfrac))
        call braids_scalar_flow_project_exner_v65( &
            topology%ncell,topology%nface,nfrac,topology%projection, &
            trial%corrected_face,trial%donor_cell,state%bed_mass, &
            state%bulk_density,topology%cell_area,dt_s,morphological_factor, &
            trial%candidate%active_fraction,trial%inventory_fill, &
            trial%availability,trial%face_flux,trial%cell_delta, &
            trial%candidate%bed_mass,trial%boundary_transfer, &
            trial%boundary_absolute_transfer,trial%conservation_residual, &
            nested_status,trial%exner_threads,raw_active,raw_inventory)
        if (nested_status /= scalar_live_ok) then
            status = live_taurat_v73_exner_failure
            return
        end if
        trial%candidate%bulk_density = state%bulk_density
        trial%candidate%boundary_signed = state%boundary_signed + &
            trial%boundary_transfer
        trial%candidate%boundary_absolute = state%boundary_absolute + &
            trial%boundary_absolute_transfer
        trial%candidate%accepted_step = state%accepted_step + 1
        trial%candidate%time_s = state%time_s + dt_s
        call derive_fraction(trial%candidate,derive_status)
        if (derive_status /= live_taurat_v73_ok) then
            status = live_taurat_v73_exner_failure
            return
        end if
        trial%candidate%initialized = .true.
        trial%nfrac = nfrac; trial%ncell = topology%ncell
        trial%nface = topology%nface
        trial%parent_step = state%accepted_step
        trial%parent_time_s = state%time_s
        allocate(trial%parent_mass(nfrac,topology%ncell), &
                 trial%parent_boundary_signed(nfrac), &
                 trial%parent_boundary_absolute(nfrac))
        trial%parent_mass = state%bed_mass
        trial%parent_boundary_signed = state%boundary_signed
        trial%parent_boundary_absolute = state%boundary_absolute
        trial%producer_taurat_application_count = 1
        trial%bed_slope_application_count = 1
        trial%active_fraction_application_count = raw_active
        trial%inventory_fill_application_count = raw_inventory
        trial%pending = .true.
        status = live_taurat_v73_ok
    end subroutine braids_begin_live_taurat_trial_v73

    pure logical function parent_matches(state,trial)
        type(braids_live_taurat_state_v73), intent(in) :: state
        type(braids_live_taurat_trial_v73), intent(in) :: trial
        parent_matches = state%initialized .and. trial%pending .and. &
            state%nfrac == trial%nfrac .and. state%ncell == trial%ncell .and. &
            state%accepted_step == trial%parent_step .and. &
            state%time_s == trial%parent_time_s
        if (.not.parent_matches) return
        parent_matches = all(state%bed_mass == trial%parent_mass) .and. &
            all(state%boundary_signed == trial%parent_boundary_signed) .and. &
            all(state%boundary_absolute == trial%parent_boundary_absolute)
    end function parent_matches

    subroutine braids_commit_live_taurat_trial_v73(state,trial,status)
        type(braids_live_taurat_state_v73), intent(inout) :: state
        type(braids_live_taurat_trial_v73), intent(inout) :: trial
        integer, intent(out) :: status
        if (.not.trial%pending) then
            status = live_taurat_v73_trial_not_pending
        else if (.not.parent_matches(state,trial)) then
            status = live_taurat_v73_state_mismatch
        else
            state = trial%candidate
            trial%pending = .false.
            status = live_taurat_v73_ok
        end if
    end subroutine braids_commit_live_taurat_trial_v73

    subroutine braids_rollback_live_taurat_trial_v73(state,trial,status)
        type(braids_live_taurat_state_v73), intent(in) :: state
        type(braids_live_taurat_trial_v73), intent(inout) :: trial
        integer, intent(out) :: status
        if (.not.trial%pending) then
            status = live_taurat_v73_trial_not_pending
        else if (.not.parent_matches(state,trial)) then
            status = live_taurat_v73_state_mismatch
        else
            trial%pending = .false.
            status = live_taurat_v73_ok
        end if
    end subroutine braids_rollback_live_taurat_trial_v73

    subroutine braids_cancel_live_taurat_trial_v73(state,trial,status)
        type(braids_live_taurat_state_v73), intent(in) :: state
        type(braids_live_taurat_trial_v73), intent(inout) :: trial
        integer, intent(out) :: status
        call braids_rollback_live_taurat_trial_v73(state,trial,status)
    end subroutine braids_cancel_live_taurat_trial_v73

    subroutine braids_export_live_taurat_state_v73( &
            state,initialized,accepted_step,time_s,bed_mass,active_fraction, &
            boundary_signed,boundary_absolute,status)
        type(braids_live_taurat_state_v73), intent(in) :: state
        logical, intent(out) :: initialized
        integer, intent(out) :: accepted_step,status
        real(real64), intent(out) :: time_s
        real(real64), allocatable, intent(out) :: bed_mass(:,:)
        real(real64), allocatable, intent(out) :: active_fraction(:,:)
        real(real64), allocatable, intent(out) :: boundary_signed(:)
        real(real64), allocatable, intent(out) :: boundary_absolute(:)
        initialized = state%initialized
        accepted_step = state%accepted_step
        time_s = state%time_s
        status = live_taurat_v73_invalid_input
        if (.not.state%initialized) return
        allocate(bed_mass(state%nfrac,state%ncell), &
                 active_fraction(state%nfrac,state%ncell), &
                 boundary_signed(state%nfrac),boundary_absolute(state%nfrac))
        bed_mass = state%bed_mass
        active_fraction = state%active_fraction
        boundary_signed = state%boundary_signed
        boundary_absolute = state%boundary_absolute
        status = live_taurat_v73_ok
    end subroutine braids_export_live_taurat_state_v73

    subroutine braids_export_live_taurat_trial_v73( &
            trial,pending,cell_taurat,trace_taurat,face_taurat,raw_face, &
            corrected_face,face_flux,conservation_residual,threads,counters, &
            status)
        type(braids_live_taurat_trial_v73), intent(in) :: trial
        logical, intent(out) :: pending
        real(real64), allocatable, intent(out) :: cell_taurat(:,:)
        real(real64), allocatable, intent(out) :: trace_taurat(:,:)
        real(real64), allocatable, intent(out) :: face_taurat(:,:)
        real(real64), allocatable, intent(out) :: raw_face(:,:,:)
        real(real64), allocatable, intent(out) :: corrected_face(:,:,:)
        real(real64), allocatable, intent(out) :: face_flux(:,:)
        real(real64), allocatable, intent(out) :: conservation_residual(:)
        integer, intent(out) :: threads(5),counters(4),status
        pending = trial%pending
        threads = [trial%cell_producer_threads,trial%trace_producer_threads, &
                   trial%raw_face_threads,trial%slope_threads, &
                   trial%exner_threads]
        counters = [trial%producer_taurat_application_count, &
                    trial%bed_slope_application_count, &
                    trial%active_fraction_application_count, &
                    trial%inventory_fill_application_count]
        status = live_taurat_v73_invalid_input
        if (.not.trial%pending) return
        allocate(cell_taurat(trial%nfrac,trial%ncell), &
                 trace_taurat(trial%nfrac,trial%nface), &
                 face_taurat(trial%nfrac,trial%nface), &
                 raw_face(2,trial%nfrac,trial%nface), &
                 corrected_face(2,trial%nfrac,trial%nface), &
                 face_flux(trial%nfrac,trial%nface), &
                 conservation_residual(trial%nfrac))
        cell_taurat = trial%cell_taurat
        trace_taurat = trial%trace_taurat
        face_taurat = trial%face_taurat
        raw_face = trial%raw_face
        corrected_face = trial%corrected_face
        face_flux = trial%face_flux
        conservation_residual = trial%conservation_residual
        status = live_taurat_v73_ok
    end subroutine braids_export_live_taurat_trial_v73

end module braids_direct_vector_live_taurat_v73_module
