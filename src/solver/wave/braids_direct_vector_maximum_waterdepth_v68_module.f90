! SPDX-License-Identifier: GPL-3.0-or-later
!
! Additive source-order bridge between FLOW dwnvel semantics and the six
! current-only direct-vector sediment producers.  The locked Delft3D source
! remains read-only.  Physical water depth is never replaced: only the
! characteristic velocity/height and the EROSED effective-current direction
! are prepared here.
module braids_direct_vector_maximum_waterdepth_v68_module
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use braids_maximum_waterdepth_v66_module, only: &
        braids_maximum_waterdepth_v66
    implicit none
    private

    integer, parameter, public :: BRAIDS_V68_WIRE_VAN_RIJN_1993 = 67001
    integer, parameter, public :: BRAIDS_V68_WIRE_BIJKER = 67005
    integer, parameter, public :: BRAIDS_V68_WIRE_SOULSBY_VAN_RIJN = 67011
    integer, parameter, public :: BRAIDS_V68_WIRE_SOULSBY = 67012
    integer, parameter, public :: BRAIDS_V68_WIRE_VAN_THIEL_VAN_RIJN = 67019
    integer, parameter, public :: BRAIDS_V68_WIRE_XBEACH_SOULSBY_VAN_RIJN = 67020

    integer, parameter, public :: BRAIDS_V68_EFFECT_EFFECTIVE_DIRECTION = 1
    integer, parameter, public :: BRAIDS_V68_EFFECT_CHARACTERISTIC = 2
    real(real64), parameter, public :: BRAIDS_V68_FLOW_EPS = &
        real(1.0e-6, real64)

    public :: braids_prepare_direct_vector_hydraulics_v68
    public :: braids_select_direct_vector_velocity_v68
    public :: braids_resolve_direct_vector_public_identity_v68

contains

    subroutine braids_prepare_direct_vector_hydraulics_v68( &
            npoint, nlink, cell_depth, transport_u, transport_v, &
            effective_speed_input, active, support_offset, support_depth, &
            support_active, maximum_waterdepth, direction_epsilon, &
            effective_direction_epsilon, observation_depth, characteristic_u, &
            characteristic_v, characteristic_speed, observation_height, &
            effective_u, effective_v, effective_speed_output, status, &
            preprocessor_application_count)
        integer, intent(in) :: npoint, nlink, maximum_waterdepth
        real(real64), intent(in) :: cell_depth(npoint), transport_u(npoint)
        real(real64), intent(in) :: transport_v(npoint)
        real(real64), intent(in) :: effective_speed_input(npoint)
        integer, intent(in) :: active(npoint), support_active(nlink)
        integer(int64), intent(in) :: support_offset(npoint+1)
        real(real64), intent(in) :: support_depth(nlink)
        real(real64), intent(in) :: direction_epsilon
        real(real64), intent(in) :: effective_direction_epsilon
        real(real64), intent(out) :: observation_depth(npoint)
        real(real64), intent(out) :: characteristic_u(npoint)
        real(real64), intent(out) :: characteristic_v(npoint)
        real(real64), intent(out) :: characteristic_speed(npoint)
        real(real64), intent(out) :: observation_height(npoint)
        real(real64), intent(out) :: effective_u(npoint), effective_v(npoint)
        real(real64), intent(out) :: effective_speed_output(npoint)
        integer, intent(out) :: status(npoint)
        integer, intent(out) :: preprocessor_application_count
        integer :: point
        real(real64) :: denominator

        preprocessor_application_count = 1
        if (maximum_waterdepth /= 0 .and. maximum_waterdepth /= 1) then
            observation_depth = 0.0_real64
            characteristic_u = 0.0_real64
            characteristic_v = 0.0_real64
            characteristic_speed = 0.0_real64
            observation_height = 0.0_real64
            effective_u = 0.0_real64
            effective_v = 0.0_real64
            effective_speed_output = 0.0_real64
            status = 1
            return
        end if

        ! Exactly one dwnvel-style preprocessing call for this hydraulic set.
        call braids_maximum_waterdepth_v66( &
            npoint, nlink, cell_depth, transport_u, transport_v, active, &
            support_offset, support_depth, support_active, &
            maximum_waterdepth, direction_epsilon, observation_depth, &
            characteristic_u, characteristic_v, characteristic_speed, &
            observation_height, status)

        !$omp parallel do default(none) schedule(static) &
        !$omp shared(npoint,active,effective_speed_input,effective_direction_epsilon, &
        !$omp        characteristic_u,characteristic_v,characteristic_speed,status, &
        !$omp        effective_u,effective_v,effective_speed_output) &
        !$omp private(point,denominator)
        do point = 1, npoint
            effective_u(point) = 0.0_real64
            effective_v(point) = 0.0_real64
            effective_speed_output(point) = 0.0_real64
            if (active(point) == 0) cycle
            if (status(point) /= 0) cycle
            if (.not.ieee_is_finite(effective_speed_input(point)) .or. &
                    effective_speed_input(point) < 0.0_real64 .or. &
                    .not.ieee_is_finite(effective_direction_epsilon) .or. &
                    effective_direction_epsilon <= 0.0_real64) then
                status(point) = 1
                cycle
            end if
            ! Locked EROSED: u=utot*uuu/(umod+eps), likewise for v.
            denominator = characteristic_speed(point) + &
                effective_direction_epsilon
            effective_u(point) = effective_speed_input(point) * &
                characteristic_u(point) / denominator
            effective_v(point) = effective_speed_input(point) * &
                characteristic_v(point) / denominator
            effective_speed_output(point) = sqrt( &
                effective_u(point)**2 + effective_v(point)**2)
            if (.not.all(ieee_is_finite([effective_u(point), &
                    effective_v(point), effective_speed_output(point)]))) then
                effective_u(point) = 0.0_real64
                effective_v(point) = 0.0_real64
                effective_speed_output(point) = 0.0_real64
                status(point) = 1
            end if
        end do
        !$omp end parallel do
    end subroutine braids_prepare_direct_vector_hydraulics_v68


    pure subroutine braids_select_direct_vector_velocity_v68( &
            internal_wire_id, characteristic_u, characteristic_v, &
            effective_u, effective_v, dwnvel_observation_height, &
            producer_u, producer_v, producer_observation_height, &
            effect_code, status)
        integer, intent(in) :: internal_wire_id
        real(real64), intent(in) :: characteristic_u, characteristic_v
        real(real64), intent(in) :: effective_u, effective_v
        real(real64), intent(in) :: dwnvel_observation_height
        real(real64), intent(out) :: producer_u, producer_v
        real(real64), intent(out) :: producer_observation_height
        integer, intent(out) :: effect_code, status

        producer_u = 0.0_real64
        producer_v = 0.0_real64
        producer_observation_height = 0.0_real64
        effect_code = 0
        status = 1
        select case (internal_wire_id)
        case (BRAIDS_V68_WIRE_VAN_RIJN_1993)
            producer_u = characteristic_u
            producer_v = characteristic_v
            producer_observation_height = dwnvel_observation_height
            effect_code = BRAIDS_V68_EFFECT_CHARACTERISTIC
        case (BRAIDS_V68_WIRE_BIJKER, BRAIDS_V68_WIRE_SOULSBY_VAN_RIJN, &
              BRAIDS_V68_WIRE_SOULSBY, BRAIDS_V68_WIRE_VAN_THIEL_VAN_RIJN, &
              BRAIDS_V68_WIRE_XBEACH_SOULSBY_VAN_RIJN)
            producer_u = effective_u
            producer_v = effective_v
            effect_code = BRAIDS_V68_EFFECT_EFFECTIVE_DIRECTION
        case default
            return
        end select
        if (.not.all(ieee_is_finite([producer_u, producer_v, &
                producer_observation_height]))) then
            producer_u = 0.0_real64
            producer_v = 0.0_real64
            producer_observation_height = 0.0_real64
            effect_code = 0
            return
        end if
        status = 0
    end subroutine braids_select_direct_vector_velocity_v68


    pure subroutine braids_resolve_direct_vector_public_identity_v68( &
            internal_wire_id, catalogue_entry_id, current_only, status)
        integer, intent(in) :: internal_wire_id
        character(len=*), intent(out) :: catalogue_entry_id
        logical, intent(out) :: current_only
        integer, intent(out) :: status

        catalogue_entry_id = ''
        current_only = .true.
        status = 0
        select case (internal_wire_id)
        case (BRAIDS_V68_WIRE_SOULSBY)
            catalogue_entry_id = 'capacity.soulsby'
        case (BRAIDS_V68_WIRE_SOULSBY_VAN_RIJN)
            catalogue_entry_id = 'capacity.soulsby_van_rijn'
        case (BRAIDS_V68_WIRE_VAN_RIJN_1993)
            catalogue_entry_id = 'capacity.van_rijn_1993'
        case (BRAIDS_V68_WIRE_BIJKER)
            catalogue_entry_id = 'capacity.bijker'
        case (BRAIDS_V68_WIRE_VAN_THIEL_VAN_RIJN)
            catalogue_entry_id = 'capacity.van_thiel_van_rijn'
        case (BRAIDS_V68_WIRE_XBEACH_SOULSBY_VAN_RIJN)
            catalogue_entry_id = 'capacity.xbeach_soulsby_van_rijn'
        case default
            current_only = .false.
            status = 1
        end select
    end subroutine braids_resolve_direct_vector_public_identity_v68

end module braids_direct_vector_maximum_waterdepth_v68_module
