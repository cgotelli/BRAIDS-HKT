! SPDX-License-Identifier: AGPL-3.0-or-later
!
! Bounded native/OpenMP Q4/v74 owner for Delft3D FLOW 2DH transport-route
! semantics.  Inputs are already mass-valued Q3 formula outputs.  The module
! owns only off/total/combine partition and combine-only soursin_2d exchange.
! It does not contain, compile, or modify Delft3D source.
module braids_suspended_route_live_v74_module
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use omp_lib, only: omp_get_num_threads
    implicit none
    private

    integer, parameter, public :: suspended_route_v74_ok = 0
    integer, parameter, public :: suspended_route_v74_invalid = 741
    integer, parameter, public :: suspended_route_v74_source_kind = 742
    integer, parameter, public :: suspended_route_v74_parent_mismatch = 743
    integer, parameter, public :: suspended_route_v74_trial_not_pending = 744
    integer, parameter, public :: suspended_route_v74_update_limit = 745
    integer, parameter, public :: suspended_route_v74_cancelled = 746
    integer, parameter, public :: suspended_route_v74_max_updates = 4
    integer, parameter, public :: suspended_route_v74_off = 0
    integer, parameter, public :: suspended_route_v74_total = 1
    integer, parameter, public :: suspended_route_v74_combine = 3
    integer, parameter, public :: suspended_route_v74_vector = 1
    integer, parameter, public :: suspended_route_v74_concentration = 2
    character(len=64), parameter, public :: suspended_route_v74_semantic = &
        '19c5c0d9a6e511f274e54704c88f33146ef5f67b425daca0dde4044b723211b1'

    type, public :: braids_suspended_route_state_v74
        logical :: initialized = .false.
        integer :: nlane = 0, ncell = 0, accepted_updates = 0
        real(real64) :: time_s = 0.0_real64
        real(real64), allocatable :: bed_mass(:,:), suspended_mass(:,:)
        real(real64), allocatable :: suspended_ledger(:), bed_ledger(:)
    end type braids_suspended_route_state_v74

    type, public :: braids_suspended_route_trial_v74
        logical :: pending = .false.
        integer :: parent_updates = -1
        real(real64) :: parent_time_s = -1.0_real64
        integer :: observed_parallel_workers = 1
        integer :: post_q3_density_conversion_count = 0
        type(braids_suspended_route_state_v74) :: candidate
        real(real64), allocatable :: parent_bed(:,:), parent_suspended(:,:)
        real(real64), allocatable :: parent_suspended_ledger(:)
        real(real64), allocatable :: parent_bed_ledger(:)
        real(real64), allocatable :: bed_capacity_x(:,:), bed_capacity_y(:,:)
        real(real64), allocatable :: equilibrium_concentration(:,:)
        real(real64), allocatable :: suspended_equivalent_x(:,:)
        real(real64), allocatable :: suspended_equivalent_y(:,:)
        real(real64), allocatable :: dimensionless_timescale(:,:)
        real(real64), allocatable :: adaptation_velocity(:,:)
        real(real64), allocatable :: transferred_to_suspension(:,:)
        real(real64), allocatable :: limiter_correction(:), mass_residual(:)
        integer, allocatable :: bed_suspended_owner_count(:)
        integer, allocatable :: water_suspended_owner_count(:)
        integer, allocatable :: suspended_factor_application_count(:)
        integer, allocatable :: active_fraction_application_count(:)
        integer, allocatable :: exchange_application_count(:)
        integer, allocatable :: duplicate_suspended_owner_count(:)
    end type braids_suspended_route_trial_v74

    public :: braids_initialize_suspended_route_state_v74
    public :: braids_begin_suspended_route_trial_v74
    public :: braids_commit_suspended_route_trial_v74
    public :: braids_rollback_suspended_route_trial_v74
    public :: braids_cancel_suspended_route_trial_v74
    public :: braids_checkpoint_suspended_route_v74
    public :: braids_restart_suspended_route_v74

contains

    pure logical function valid_route(route)
        integer, intent(in) :: route
        valid_route = route == suspended_route_v74_off .or. &
            route == suspended_route_v74_total .or. &
            route == suspended_route_v74_combine
    end function valid_route

    pure logical function valid_kind(kind)
        integer, intent(in) :: kind
        valid_kind = kind == suspended_route_v74_vector .or. &
            kind == suspended_route_v74_concentration
    end function valid_kind

    subroutine braids_initialize_suspended_route_state_v74( &
            nlane,ncell,bed_mass,suspended_mass,state,status)
        integer, intent(in) :: nlane,ncell
        real(real64), intent(in) :: bed_mass(nlane,ncell)
        real(real64), intent(in) :: suspended_mass(nlane,ncell)
        type(braids_suspended_route_state_v74), intent(out) :: state
        integer, intent(out) :: status

        status = suspended_route_v74_invalid
        if (nlane <= 0 .or. ncell <= 0) return
        if (any(.not.ieee_is_finite(bed_mass)) .or. &
                any(.not.ieee_is_finite(suspended_mass)) .or. &
                any(bed_mass < 0.0_real64) .or. &
                any(suspended_mass < 0.0_real64)) return
        state%initialized = .true.
        state%nlane = nlane
        state%ncell = ncell
        state%accepted_updates = 0
        state%time_s = 0.0_real64
        allocate(state%bed_mass(nlane,ncell), &
                 state%suspended_mass(nlane,ncell), &
                 state%suspended_ledger(nlane),state%bed_ledger(nlane))
        state%bed_mass = bed_mass
        state%suspended_mass = suspended_mass
        state%suspended_ledger = 0.0_real64
        state%bed_ledger = 0.0_real64
        status = suspended_route_v74_ok
    end subroutine braids_initialize_suspended_route_state_v74

    subroutine braids_begin_suspended_route_trial_v74( &
            nlane,ncell,output_kind,route,depth,velocity_x,velocity_y, &
            raw_bed_x,raw_bed_y,raw_suspended_x,raw_suspended_y, &
            raw_equilibrium,active_fraction,depth_before,depth_after, &
            speed,shear_velocity,settling_velocity,cell_area,bed_factor, &
            suspended_factor,speed_epsilon,fac_tsd,sediment_threshold, &
            minimum_reference_depth,dt_s,state,cancelled,trial,status)
        integer, intent(in) :: nlane,ncell
        integer, intent(in) :: output_kind(nlane),route(nlane)
        real(real64), intent(in) :: depth(ncell),velocity_x(ncell)
        real(real64), intent(in) :: velocity_y(ncell)
        real(real64), intent(in) :: raw_bed_x(nlane,ncell)
        real(real64), intent(in) :: raw_bed_y(nlane,ncell)
        real(real64), intent(in) :: raw_suspended_x(nlane,ncell)
        real(real64), intent(in) :: raw_suspended_y(nlane,ncell)
        real(real64), intent(in) :: raw_equilibrium(nlane,ncell)
        real(real64), intent(in) :: active_fraction(nlane,ncell)
        real(real64), intent(in) :: depth_before(ncell),depth_after(ncell)
        real(real64), intent(in) :: speed(ncell),shear_velocity(ncell)
        real(real64), intent(in) :: settling_velocity(nlane,ncell)
        real(real64), intent(in) :: cell_area(ncell)
        real(real64), intent(in) :: bed_factor,suspended_factor
        real(real64), intent(in) :: speed_epsilon,fac_tsd
        real(real64), intent(in) :: sediment_threshold
        real(real64), intent(in) :: minimum_reference_depth,dt_s
        type(braids_suspended_route_state_v74), intent(in) :: state
        logical, intent(in) :: cancelled
        type(braids_suspended_route_trial_v74), intent(out) :: trial
        integer, intent(out) :: status
        integer :: lane,cell
        integer, allocatable :: local_status(:,:)
        real(real64), allocatable :: limiter_cell(:,:),residual_cell(:,:)
        real(real64) :: h,ux,uy,flow_speed,fraction
        real(real64) :: scaled_nonbed_x,scaled_nonbed_y,derived_equilibrium
        real(real64) :: old_bed,old_suspended,total,raw_candidate
        real(real64) :: candidate_suspended,candidate_bed,transfer
        real(real64) :: wsl,ratio,w,x2,x3,exponent,timescale,hots
        real(real64) :: explicit_source,total_sink,safe_h0,safe_h1
        real(real64) :: before_mass,residual_scale
        logical :: active,moving

        status = suspended_route_v74_invalid
        if (cancelled) then
            status = suspended_route_v74_cancelled
            return
        end if
        if (.not.state%initialized .or. state%nlane /= nlane .or. &
                state%ncell /= ncell) return
        if (state%accepted_updates >= suspended_route_v74_max_updates) then
            status = suspended_route_v74_update_limit
            return
        end if
        if (any(.not.ieee_is_finite(depth)) .or. any(depth <= 0.0_real64) .or. &
                any(.not.ieee_is_finite(velocity_x)) .or. &
                any(.not.ieee_is_finite(velocity_y)) .or. &
                any(.not.ieee_is_finite(raw_bed_x)) .or. &
                any(.not.ieee_is_finite(raw_bed_y)) .or. &
                any(.not.ieee_is_finite(raw_suspended_x)) .or. &
                any(.not.ieee_is_finite(raw_suspended_y)) .or. &
                any(.not.ieee_is_finite(raw_equilibrium)) .or. &
                any(raw_equilibrium < 0.0_real64) .or. &
                any(.not.ieee_is_finite(active_fraction)) .or. &
                any(active_fraction < 0.0_real64) .or. &
                any(active_fraction > 1.0_real64) .or. &
                any(.not.ieee_is_finite(depth_before)) .or. &
                any(depth_before <= 0.0_real64) .or. &
                any(.not.ieee_is_finite(depth_after)) .or. &
                any(depth_after <= 0.0_real64) .or. &
                any(.not.ieee_is_finite(speed)) .or. &
                any(speed < 0.0_real64) .or. &
                any(.not.ieee_is_finite(shear_velocity)) .or. &
                any(shear_velocity < 0.0_real64) .or. &
                any(.not.ieee_is_finite(settling_velocity)) .or. &
                any(settling_velocity < 0.0_real64) .or. &
                any(.not.ieee_is_finite(cell_area)) .or. &
                any(cell_area <= 0.0_real64)) return
        if (.not.ieee_is_finite(bed_factor) .or. bed_factor < 0.0_real64 .or. &
                .not.ieee_is_finite(suspended_factor) .or. &
                suspended_factor < 0.0_real64 .or. &
                .not.ieee_is_finite(speed_epsilon) .or. &
                speed_epsilon <= 0.0_real64 .or. &
                .not.ieee_is_finite(fac_tsd) .or. fac_tsd <= 0.0_real64 .or. &
                .not.ieee_is_finite(sediment_threshold) .or. &
                sediment_threshold < 0.0_real64 .or. &
                .not.ieee_is_finite(minimum_reference_depth) .or. &
                minimum_reference_depth <= 0.0_real64 .or. &
                .not.ieee_is_finite(dt_s) .or. dt_s <= 0.0_real64) return
        do lane = 1,nlane
            if (.not.valid_route(route(lane)) .or. &
                    .not.valid_kind(output_kind(lane))) then
                status = suspended_route_v74_source_kind
                return
            end if
            if (output_kind(lane) == suspended_route_v74_vector) then
                if (any(raw_equilibrium(lane,:) /= 0.0_real64)) then
                    status = suspended_route_v74_source_kind
                    return
                end if
            else
                if (any(raw_suspended_x(lane,:) /= 0.0_real64) .or. &
                        any(raw_suspended_y(lane,:) /= 0.0_real64)) then
                    status = suspended_route_v74_source_kind
                    return
                end if
            end if
        end do

        trial%parent_updates = state%accepted_updates
        trial%parent_time_s = state%time_s
        trial%parent_bed = state%bed_mass
        trial%parent_suspended = state%suspended_mass
        trial%parent_suspended_ledger = state%suspended_ledger
        trial%parent_bed_ledger = state%bed_ledger
        trial%candidate = state
        trial%candidate%accepted_updates = state%accepted_updates+1
        trial%candidate%time_s = state%time_s+dt_s
        allocate(trial%bed_capacity_x(nlane,ncell), &
                 trial%bed_capacity_y(nlane,ncell), &
                 trial%equilibrium_concentration(nlane,ncell), &
                 trial%suspended_equivalent_x(nlane,ncell), &
                 trial%suspended_equivalent_y(nlane,ncell), &
                 trial%dimensionless_timescale(nlane,ncell), &
                 trial%adaptation_velocity(nlane,ncell), &
                 trial%transferred_to_suspension(nlane,ncell), &
                 trial%limiter_correction(nlane), &
                 trial%mass_residual(nlane), &
                 trial%bed_suspended_owner_count(nlane), &
                 trial%water_suspended_owner_count(nlane), &
                 trial%suspended_factor_application_count(nlane), &
                 trial%active_fraction_application_count(nlane), &
                 trial%exchange_application_count(nlane), &
                 trial%duplicate_suspended_owner_count(nlane), &
                 local_status(nlane,ncell),limiter_cell(nlane,ncell), &
                 residual_cell(nlane,ncell))
        trial%bed_capacity_x = 0.0_real64
        trial%bed_capacity_y = 0.0_real64
        trial%equilibrium_concentration = 0.0_real64
        trial%suspended_equivalent_x = 0.0_real64
        trial%suspended_equivalent_y = 0.0_real64
        trial%dimensionless_timescale = 0.0_real64
        trial%adaptation_velocity = 0.0_real64
        trial%transferred_to_suspension = 0.0_real64
        trial%limiter_correction = 0.0_real64
        trial%mass_residual = 0.0_real64
        trial%bed_suspended_owner_count = 0
        trial%water_suspended_owner_count = 0
        trial%suspended_factor_application_count = 0
        trial%active_fraction_application_count = 0
        trial%exchange_application_count = 0
        trial%duplicate_suspended_owner_count = 0
        local_status = 0
        limiter_cell = 0.0_real64
        residual_cell = 0.0_real64

        !$omp parallel default(shared) &
        !$omp& private(lane,cell,h,ux,uy,flow_speed,fraction, &
        !$omp& scaled_nonbed_x,scaled_nonbed_y,derived_equilibrium, &
        !$omp& old_bed,old_suspended,total,raw_candidate, &
        !$omp& candidate_suspended,candidate_bed,transfer,wsl,ratio,w,x2,x3, &
        !$omp& exponent,timescale,hots,explicit_source,total_sink,safe_h0, &
        !$omp& safe_h1,active,moving)
        !$omp single
        trial%observed_parallel_workers = omp_get_num_threads()
        !$omp end single
        !$omp do collapse(2) schedule(static)
        do cell = 1,ncell
            do lane = 1,nlane
                h = depth(cell)
                ux = velocity_x(cell)
                uy = velocity_y(cell)
                flow_speed = hypot(ux,uy)
                fraction = active_fraction(lane,cell)
                if (output_kind(lane) == suspended_route_v74_vector) then
                    scaled_nonbed_x = &
                        suspended_factor*raw_suspended_x(lane,cell)
                    scaled_nonbed_y = &
                        suspended_factor*raw_suspended_y(lane,cell)
                    derived_equilibrium = suspended_factor*hypot( &
                        raw_suspended_x(lane,cell), &
                        raw_suspended_y(lane,cell))/(flow_speed+speed_epsilon)/h
                else
                    derived_equilibrium = &
                        suspended_factor*raw_equilibrium(lane,cell)
                    scaled_nonbed_x = derived_equilibrium*h*ux
                    scaled_nonbed_y = derived_equilibrium*h*uy
                end if
                if (route(lane) /= suspended_route_v74_off) then
                    trial%suspended_equivalent_x(lane,cell) = &
                        fraction*scaled_nonbed_x
                    trial%suspended_equivalent_y(lane,cell) = &
                        fraction*scaled_nonbed_y
                    if (route(lane) == suspended_route_v74_total) then
                        trial%bed_capacity_x(lane,cell) = fraction*( &
                            bed_factor*raw_bed_x(lane,cell)+scaled_nonbed_x)
                        trial%bed_capacity_y(lane,cell) = fraction*( &
                            bed_factor*raw_bed_y(lane,cell)+scaled_nonbed_y)
                    else
                        trial%bed_capacity_x(lane,cell) = &
                            fraction*bed_factor*raw_bed_x(lane,cell)
                        trial%bed_capacity_y(lane,cell) = &
                            fraction*bed_factor*raw_bed_y(lane,cell)
                        trial%equilibrium_concentration(lane,cell) = &
                            fraction*derived_equilibrium
                    end if
                end if

                active = route(lane) == suspended_route_v74_combine .and. &
                    depth_after(cell) > sediment_threshold
                moving = active .and. speed(cell) > 1.0e-6_real64 .and. &
                    shear_velocity(cell) > 1.0e-6_real64
                wsl = max(1.0e-3_real64,settling_velocity(lane,cell))
                timescale = 0.0_real64
                hots = 0.0_real64
                explicit_source = 0.0_real64
                total_sink = 0.0_real64
                safe_h0 = max(minimum_reference_depth,depth_before(cell))
                safe_h1 = max(minimum_reference_depth,depth_after(cell))
                if (moving) then
                    ratio = min(shear_velocity(cell)/speed(cell),0.15_real64)
                    if (shear_velocity(cell) > wsl) then
                        w = wsl/shear_velocity(cell)
                    else
                        w = 1.0_real64
                    end if
                    x2 = w*w
                    x3 = x2*w
                    exponent = &
                        (real(1.547,real64)-real(20.12,real64)*ratio)*x3 + &
                        (real(326.832,real64)*ratio**real(2.2047,real64)- &
                         real(0.2,real64))*x2 + &
                        (real(0.1385,real64)*log(ratio)- &
                         real(6.4061,real64))*w + &
                        real(0.5467,real64)*ratio+real(2.1963,real64)
                    timescale = w*exp(exponent)
                    hots = wsl/(timescale*fac_tsd)
                    explicit_source = &
                        trial%equilibrium_concentration(lane,cell)*hots/safe_h0
                    total_sink = hots/safe_h1
                else if (active) then
                    total_sink = wsl/safe_h1
                end if
                old_bed = state%bed_mass(lane,cell)
                old_suspended = state%suspended_mass(lane,cell)
                total = old_bed+old_suspended
                if (active) then
                    raw_candidate = (old_suspended+ &
                        dt_s*depth_before(cell)*explicit_source)/ &
                        (1.0_real64+dt_s*total_sink)
                    candidate_suspended = min(max(raw_candidate,0.0_real64), &
                        total)
                else
                    raw_candidate = old_suspended
                    candidate_suspended = old_suspended
                end if
                candidate_bed = total-candidate_suspended
                transfer = candidate_suspended-old_suspended
                trial%candidate%bed_mass(lane,cell) = candidate_bed
                trial%candidate%suspended_mass(lane,cell) = &
                    candidate_suspended
                trial%transferred_to_suspension(lane,cell) = transfer
                trial%dimensionless_timescale(lane,cell) = timescale
                trial%adaptation_velocity(lane,cell) = hots
                limiter_cell(lane,cell) = &
                    abs(candidate_suspended-raw_candidate)*cell_area(cell)
                residual_cell(lane,cell) = &
                    ((candidate_suspended+candidate_bed)-total)*cell_area(cell)
                if (.not.ieee_is_finite(candidate_suspended) .or. &
                        .not.ieee_is_finite(candidate_bed) .or. &
                        .not.ieee_is_finite(timescale) .or. &
                        candidate_suspended < 0.0_real64 .or. &
                        candidate_bed < 0.0_real64 .or. &
                        (moving .and. timescale <= 0.0_real64)) then
                    local_status(lane,cell) = suspended_route_v74_invalid
                end if
            end do
        end do
        !$omp end do
        !$omp end parallel
        if (any(local_status /= 0)) then
            status = suspended_route_v74_invalid
            return
        end if
        do lane = 1,nlane
            trial%limiter_correction(lane) = sum(limiter_cell(lane,:))
            trial%mass_residual(lane) = sum(residual_cell(lane,:))
            before_mass = sum((state%bed_mass(lane,:)+ &
                state%suspended_mass(lane,:))*cell_area)
            residual_scale = max(1.0_real64,abs(before_mass))
            if (abs(trial%mass_residual(lane)) > &
                    4096.0_real64*epsilon(1.0_real64)*residual_scale) then
                status = suspended_route_v74_invalid
                return
            end if
            if (route(lane) /= suspended_route_v74_off) then
                trial%suspended_factor_application_count(lane) = 1
                trial%active_fraction_application_count(lane) = 1
            end if
            if (route(lane) == suspended_route_v74_total) then
                trial%bed_suspended_owner_count(lane) = 1
            else if (route(lane) == suspended_route_v74_combine) then
                trial%water_suspended_owner_count(lane) = 1
                trial%exchange_application_count(lane) = 1
            end if
            trial%duplicate_suspended_owner_count(lane) = &
                trial%bed_suspended_owner_count(lane)* &
                trial%water_suspended_owner_count(lane)
            transfer = sum( &
                trial%transferred_to_suspension(lane,:)*cell_area)
            trial%candidate%suspended_ledger(lane) = &
                state%suspended_ledger(lane)+transfer
            trial%candidate%bed_ledger(lane) = &
                state%bed_ledger(lane)-transfer
        end do
        if (any(trial%duplicate_suspended_owner_count /= 0)) then
            status = suspended_route_v74_invalid
            return
        end if
        trial%pending = .true.
        status = suspended_route_v74_ok
    end subroutine braids_begin_suspended_route_trial_v74

    logical function parent_matches(state,trial)
        type(braids_suspended_route_state_v74), intent(in) :: state
        type(braids_suspended_route_trial_v74), intent(in) :: trial
        parent_matches = state%initialized .and. trial%pending .and. &
            state%accepted_updates == trial%parent_updates .and. &
            state%time_s == trial%parent_time_s .and. &
            all(state%bed_mass == trial%parent_bed) .and. &
            all(state%suspended_mass == trial%parent_suspended) .and. &
            all(state%suspended_ledger == trial%parent_suspended_ledger) .and. &
            all(state%bed_ledger == trial%parent_bed_ledger)
    end function parent_matches

    subroutine braids_commit_suspended_route_trial_v74(state,trial,status)
        type(braids_suspended_route_state_v74), intent(inout) :: state
        type(braids_suspended_route_trial_v74), intent(inout) :: trial
        integer, intent(out) :: status
        if (.not.trial%pending) then
            status = suspended_route_v74_trial_not_pending
        else if (.not.parent_matches(state,trial)) then
            status = suspended_route_v74_parent_mismatch
        else
            state = trial%candidate
            trial%pending = .false.
            status = suspended_route_v74_ok
        end if
    end subroutine braids_commit_suspended_route_trial_v74

    subroutine braids_rollback_suspended_route_trial_v74(state,trial,status)
        type(braids_suspended_route_state_v74), intent(in) :: state
        type(braids_suspended_route_trial_v74), intent(inout) :: trial
        integer, intent(out) :: status
        if (.not.trial%pending) then
            status = suspended_route_v74_trial_not_pending
        else if (.not.parent_matches(state,trial)) then
            status = suspended_route_v74_parent_mismatch
        else
            trial%pending = .false.
            status = suspended_route_v74_ok
        end if
    end subroutine braids_rollback_suspended_route_trial_v74

    subroutine braids_cancel_suspended_route_trial_v74(state,trial,status)
        type(braids_suspended_route_state_v74), intent(in) :: state
        type(braids_suspended_route_trial_v74), intent(inout) :: trial
        integer, intent(out) :: status
        call braids_rollback_suspended_route_trial_v74(state,trial,status)
    end subroutine braids_cancel_suspended_route_trial_v74

    subroutine braids_checkpoint_suspended_route_v74(state,checkpoint,status)
        type(braids_suspended_route_state_v74), intent(in) :: state
        type(braids_suspended_route_state_v74), intent(out) :: checkpoint
        integer, intent(out) :: status
        if (.not.state%initialized .or. state%accepted_updates < 0 .or. &
                state%accepted_updates > suspended_route_v74_max_updates) then
            status = suspended_route_v74_invalid
            return
        end if
        checkpoint = state
        status = suspended_route_v74_ok
    end subroutine braids_checkpoint_suspended_route_v74

    subroutine braids_restart_suspended_route_v74(checkpoint,state,status)
        type(braids_suspended_route_state_v74), intent(in) :: checkpoint
        type(braids_suspended_route_state_v74), intent(out) :: state
        integer, intent(out) :: status
        if (.not.checkpoint%initialized .or. &
                checkpoint%accepted_updates < 0 .or. &
                checkpoint%accepted_updates > suspended_route_v74_max_updates .or. &
                any(.not.ieee_is_finite(checkpoint%bed_mass)) .or. &
                any(.not.ieee_is_finite(checkpoint%suspended_mass)) .or. &
                any(checkpoint%bed_mass < 0.0_real64) .or. &
                any(checkpoint%suspended_mass < 0.0_real64)) then
            status = suspended_route_v74_invalid
            return
        end if
        state = checkpoint
        status = suspended_route_v74_ok
    end subroutine braids_restart_suspended_route_v74

end module braids_suspended_route_live_v74_module
