! Generic post-formula route composer for current-only direct-vector methods.
!
! Inputs are already solid-mass-valued.  This module intentionally performs
! no density conversion and no external direction rotation.  It normalizes
! the three output-ownership patterns used by the locked Delft3D4 FLOW 2DH
! direct methods, applies Bed/Sus exactly once, then applies the active-bed
! fraction exactly once after algebraic folding (TRA_BEDLOAD) or to the two
! separately owned channels (TRA_COMBINE).
!
! Method identities are strings.  The integer values below form only the
! local output-kind/status enums; they are not Delft3D TraFrm/internal-law IDs.
module braids_direct_vector_route_composer_v66_module
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: BRAIDS_DV_KIND_ZERO_NONBED = 0
    integer, parameter, public :: BRAIDS_DV_KIND_EXPLICIT_SUSPENDED_VECTOR = 1
    integer, parameter, public :: BRAIDS_DV_KIND_EQUILIBRIUM_CONCENTRATION = 2

    integer, parameter, public :: BRAIDS_DV_ROUTE_OK = 0
    integer, parameter, public :: BRAIDS_DV_ROUTE_INVALID_METHOD = 1
    integer, parameter, public :: BRAIDS_DV_ROUTE_INVALID_ROUTE = 2
    integer, parameter, public :: BRAIDS_DV_ROUTE_NONFINITE = 3
    integer, parameter, public :: BRAIDS_DV_ROUTE_INVALID_ARGUMENT = 4
    integer, parameter, public :: BRAIDS_DV_ROUTE_SOURCE_KIND_MISMATCH = 5
    integer, parameter, public :: BRAIDS_DV_ROUTE_DRY_SOURCE_NONZERO = 6
    integer, parameter, public :: BRAIDS_DV_ROUTE_ARITHMETIC = 7

    character(len=32), parameter, public :: BRAIDS_DV_ROUTE_BEDLOAD = &
        'tra_bedload'
    character(len=32), parameter, public :: BRAIDS_DV_ROUTE_COMBINE = &
        'tra_combine'

    character(len=48), parameter, public :: BRAIDS_DV_METHOD_SOULSBY = &
        'soulsby_current'
    character(len=48), parameter, public :: BRAIDS_DV_METHOD_SOULSBY_VAN_RIJN = &
        'soulsby_van_rijn_current'
    character(len=48), parameter, public :: BRAIDS_DV_METHOD_BIJKER = &
        'bijker_current'
    character(len=48), parameter, public :: BRAIDS_DV_METHOD_VAN_RIJN_1993 = &
        'van_rijn_1993_current'
    character(len=48), parameter, public :: BRAIDS_DV_METHOD_VAN_THIEL_VAN_RIJN_2008 = &
        'van_thiel_van_rijn_2008_current'
    character(len=48), parameter, public :: BRAIDS_DV_METHOD_XBEACH_SOULSBY_VAN_RIJN = &
        'xbeach_soulsby_van_rijn_current'

    public :: braids_resolve_direct_vector_output_kind_v66
    public :: braids_compose_direct_vector_route_v66

contains

    pure subroutine braids_resolve_direct_vector_output_kind_v66( &
            method, output_kind, status)
        character(len=*), intent(in) :: method
        integer, intent(out) :: output_kind
        integer, intent(out) :: status

        output_kind = -1
        status = BRAIDS_DV_ROUTE_INVALID_METHOD
        select case (trim(method))
        case ('soulsby_current')
            output_kind = BRAIDS_DV_KIND_ZERO_NONBED
        case ('soulsby_van_rijn_current', 'bijker_current')
            output_kind = BRAIDS_DV_KIND_EXPLICIT_SUSPENDED_VECTOR
        case ('van_rijn_1993_current', &
              'van_thiel_van_rijn_2008_current', &
              'xbeach_soulsby_van_rijn_current')
            output_kind = BRAIDS_DV_KIND_EQUILIBRIUM_CONCENTRATION
        case default
            return
        end select
        status = BRAIDS_DV_ROUTE_OK
    end subroutine braids_resolve_direct_vector_output_kind_v66


    pure subroutine braids_compose_direct_vector_route_v66( &
            method, route, depth, velocity, raw_bed_mass, &
            raw_suspended_mass, raw_equilibrium_mass_concentration, &
            bed_factor, suspended_factor, active_fraction, speed_epsilon, &
            output_kind, composed_bed_mass, q4_equilibrium_mass_concentration, &
            valid, status)
        character(len=*), intent(in) :: method
        character(len=*), intent(in) :: route
        real(real64), intent(in) :: depth
        real(real64), intent(in) :: velocity(2)
        real(real64), intent(in) :: raw_bed_mass(2)
        real(real64), intent(in) :: raw_suspended_mass(2)
        real(real64), intent(in) :: raw_equilibrium_mass_concentration
        real(real64), intent(in) :: bed_factor
        real(real64), intent(in) :: suspended_factor
        real(real64), intent(in) :: active_fraction
        real(real64), intent(in) :: speed_epsilon
        integer, intent(out) :: output_kind
        real(real64), intent(out) :: composed_bed_mass(2)
        real(real64), intent(out) :: q4_equilibrium_mass_concentration
        logical, intent(out) :: valid
        integer, intent(out) :: status

        real(real64) :: scaled_bed(2)
        real(real64) :: nonbed_vector(2)
        real(real64) :: scaled_concentration
        real(real64) :: suspended_magnitude
        real(real64) :: speed
        logical :: vector_nonzero
        logical :: concentration_nonzero
        logical :: source_nonzero

        output_kind = -1
        composed_bed_mass = 0.0_real64
        q4_equilibrium_mass_concentration = 0.0_real64
        valid = .false.
        status = BRAIDS_DV_ROUTE_NONFINITE

        call braids_resolve_direct_vector_output_kind_v66( &
            method, output_kind, status)
        if (status /= BRAIDS_DV_ROUTE_OK) return
        if (trim(route) /= trim(BRAIDS_DV_ROUTE_BEDLOAD) .and. &
                trim(route) /= trim(BRAIDS_DV_ROUTE_COMBINE)) then
            status = BRAIDS_DV_ROUTE_INVALID_ROUTE
            return
        end if

        if (.not.ieee_is_finite(depth) .or. &
                .not.all(ieee_is_finite(velocity)) .or. &
                .not.all(ieee_is_finite(raw_bed_mass)) .or. &
                .not.all(ieee_is_finite(raw_suspended_mass)) .or. &
                .not.ieee_is_finite(raw_equilibrium_mass_concentration) .or. &
                .not.ieee_is_finite(bed_factor) .or. &
                .not.ieee_is_finite(suspended_factor) .or. &
                .not.ieee_is_finite(active_fraction) .or. &
                .not.ieee_is_finite(speed_epsilon)) return

        if (depth < 0.0_real64 .or. active_fraction < 0.0_real64 .or. &
                active_fraction > 1.0_real64 .or. speed_epsilon <= 0.0_real64 .or. &
                raw_equilibrium_mass_concentration < 0.0_real64) then
            status = BRAIDS_DV_ROUTE_INVALID_ARGUMENT
            return
        end if

        vector_nonzero = any(raw_suspended_mass /= 0.0_real64)
        concentration_nonzero = raw_equilibrium_mass_concentration /= 0.0_real64
        select case (output_kind)
        case (BRAIDS_DV_KIND_ZERO_NONBED)
            if (vector_nonzero .or. concentration_nonzero) then
                status = BRAIDS_DV_ROUTE_SOURCE_KIND_MISMATCH
                return
            end if
        case (BRAIDS_DV_KIND_EXPLICIT_SUSPENDED_VECTOR)
            if (concentration_nonzero) then
                status = BRAIDS_DV_ROUTE_SOURCE_KIND_MISMATCH
                return
            end if
        case (BRAIDS_DV_KIND_EQUILIBRIUM_CONCENTRATION)
            if (vector_nonzero) then
                status = BRAIDS_DV_ROUTE_SOURCE_KIND_MISMATCH
                return
            end if
        case default
            status = BRAIDS_DV_ROUTE_INVALID_METHOD
            return
        end select

        source_nonzero = any(raw_bed_mass /= 0.0_real64) .or. &
                         vector_nonzero .or. concentration_nonzero
        if (depth == 0.0_real64) then
            if (source_nonzero) then
                status = BRAIDS_DV_ROUTE_DRY_SOURCE_NONZERO
                return
            end if
            valid = .true.
            status = BRAIDS_DV_ROUTE_OK
            return
        end if

        ! These are the only Bed/Sus applications in the operator.
        scaled_bed = bed_factor*raw_bed_mass
        nonbed_vector = 0.0_real64
        scaled_concentration = 0.0_real64
        select case (output_kind)
        case (BRAIDS_DV_KIND_EXPLICIT_SUSPENDED_VECTOR)
            nonbed_vector = suspended_factor*raw_suspended_mass
        case (BRAIDS_DV_KIND_EQUILIBRIUM_CONCENTRATION)
            scaled_concentration = suspended_factor* &
                raw_equilibrium_mass_concentration
            nonbed_vector = scaled_concentration*depth*velocity
        end select

        if (trim(route) == trim(BRAIDS_DV_ROUTE_BEDLOAD)) then
            ! Fold first, then apply the active fraction once to total load.
            composed_bed_mass = active_fraction*(scaled_bed + nonbed_vector)
        else
            composed_bed_mass = active_fraction*scaled_bed
            select case (output_kind)
            case (BRAIDS_DV_KIND_ZERO_NONBED)
                q4_equilibrium_mass_concentration = 0.0_real64
            case (BRAIDS_DV_KIND_EXPLICIT_SUSPENDED_VECTOR)
                suspended_magnitude = safe_norm2(raw_suspended_mass)
                speed = safe_norm2(velocity)
                q4_equilibrium_mass_concentration = active_fraction* &
                    suspended_factor*suspended_magnitude/ &
                    (speed + speed_epsilon)/depth
            case (BRAIDS_DV_KIND_EQUILIBRIUM_CONCENTRATION)
                q4_equilibrium_mass_concentration = active_fraction* &
                    scaled_concentration
            end select
        end if

        if (.not.all(ieee_is_finite(composed_bed_mass)) .or. &
                .not.ieee_is_finite(q4_equilibrium_mass_concentration)) then
            composed_bed_mass = 0.0_real64
            q4_equilibrium_mass_concentration = 0.0_real64
            status = BRAIDS_DV_ROUTE_ARITHMETIC
            return
        end if
        valid = .true.
        status = BRAIDS_DV_ROUTE_OK
    end subroutine braids_compose_direct_vector_route_v66


    pure function safe_norm2(vector) result(norm)
        real(real64), intent(in) :: vector(2)
        real(real64) :: norm
        real(real64) :: scale

        scale = max(abs(vector(1)), abs(vector(2)))
        if (scale > 0.0_real64) then
            norm = scale*sqrt((vector(1)/scale)**2 + (vector(2)/scale)**2)
        else
            norm = 0.0_real64
        end if
    end function safe_norm2

end module braids_direct_vector_route_composer_v66_module
