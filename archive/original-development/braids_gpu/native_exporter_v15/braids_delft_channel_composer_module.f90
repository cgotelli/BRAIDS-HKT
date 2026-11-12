! Bounded CPU reference for the post-density Delft3D FLOW channel ordering.
!
! The scalar capacity producer has already converted both input channels to
! mass transport per unit width [kg/(m s)].  Any external secondary-flow
! steering has also already been applied to raw_bed_mass, and only to that
! channel.  This routine therefore performs no density conversion and no
! direction rotation.
!
! Locked-source anchors (read-only Delft3D checkout):
!   eqtran.f90:683-710  volume-to-mass conversion (upstream of this routine)
!   eqtran.f90:713-747  bed-only secondary-flow direction
!   eqtran.f90:749-762  global Bed/Sus factors
!   eqtran.f90:772-818  advdiff equilibrium concentration
!   eqtran.f90:819-858  total-load suspended-channel fold
!   erosed.f90:1070-1157  TRA_ADVDIFF uses erosilt and bypasses eqtran
!
! This is an isolated reference operator.  It is intentionally not wired into
! the production cut-cell solver by this change.
module braids_delft_channel_composer_module
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: BRAIDS_TRA_BEDLOAD = 1
    integer, parameter, public :: BRAIDS_TRA_ADVDIFF = 2
    integer, parameter, public :: BRAIDS_TRA_COMBINE = 3

    integer, parameter, public :: BRAIDS_COMPOSE_OK = 0
    integer, parameter, public :: BRAIDS_COMPOSE_NONFINITE = 1
    integer, parameter, public :: BRAIDS_COMPOSE_UNSUPPORTED_CLASS = 2
    integer, parameter, public :: BRAIDS_COMPOSE_ADVDIFF_REQUIRES_EROSILT = 3
    integer, parameter, public :: BRAIDS_COMPOSE_NONPOSITIVE_DEPTH = 4
    integer, parameter, public :: BRAIDS_COMPOSE_SUSPENDED_MISALIGNED = 5
    integer, parameter, public :: BRAIDS_COMPOSE_ZERO_FLOW_SUSPENDED = 6
    integer, parameter, public :: BRAIDS_COMPOSE_ARITHMETIC_OVERFLOW = 7
    integer, parameter, public :: BRAIDS_COMPOSE_NONPOSITIVE_EPSILON = 8

    real(real64), parameter, public :: BRAIDS_SUSPENDED_ALIGNMENT_RTOL = 1.0e-12_real64
    real(real64), parameter, public :: BRAIDS_SUSPENDED_ALIGNMENT_ATOL = 1.0e-15_real64

    public :: braids_compose_delft_channels

contains

    subroutine braids_compose_delft_channels(raw_bed_mass, raw_suspended_mass, &
            base_velocity, depth, bed_factor, suspended_factor, transport_class, &
            speed_epsilon, &
            composed_bed_mass, retained_suspended_mass, &
            signed_suspended_mass_rate, equilibrium_mass_concentration, valid, status)
        real(real64), intent(in) :: raw_bed_mass(2)
        real(real64), intent(in) :: raw_suspended_mass(2)
        real(real64), intent(in) :: base_velocity(2)
        real(real64), intent(in) :: depth
        real(real64), intent(in) :: bed_factor
        real(real64), intent(in) :: suspended_factor
        integer, intent(in) :: transport_class
        real(real64), intent(in) :: speed_epsilon
        real(real64), intent(out) :: composed_bed_mass(2)
        real(real64), intent(out) :: retained_suspended_mass(2)
        real(real64), intent(out) :: signed_suspended_mass_rate
        real(real64), intent(out) :: equilibrium_mass_concentration
        logical, intent(out) :: valid
        integer, intent(out) :: status

        real(real64) :: base_direction(2)
        real(real64) :: scaled_bed_mass(2)
        real(real64) :: scaled_suspended_mass(2)
        real(real64) :: speed
        real(real64) :: suspended_norm
        real(real64) :: raw_signed_rate
        real(real64) :: alignment_residual(2)

        composed_bed_mass = 0.0_real64
        retained_suspended_mass = 0.0_real64
        signed_suspended_mass_rate = 0.0_real64
        equilibrium_mass_concentration = 0.0_real64
        valid = .false.
        status = BRAIDS_COMPOSE_NONFINITE

        if (.not. all(ieee_is_finite(raw_bed_mass))) return
        if (.not. all(ieee_is_finite(raw_suspended_mass))) return
        if (.not. all(ieee_is_finite(base_velocity))) return
        if (.not. ieee_is_finite(depth)) return
        if (.not. ieee_is_finite(bed_factor)) return
        if (.not. ieee_is_finite(suspended_factor)) return
        if (.not. ieee_is_finite(speed_epsilon)) return

        if (speed_epsilon <= 0.0_real64) then
            status = BRAIDS_COMPOSE_NONPOSITIVE_EPSILON
            return
        end if

        if (transport_class == BRAIDS_TRA_ADVDIFF) then
            ! A pure TRA_ADVDIFF fraction is handled by erosilt in locked FLOW
            ! and exits erosed before eqtran.  There is no source-faithful
            ! algebraic raw-channel composition for class 2.
            status = BRAIDS_COMPOSE_ADVDIFF_REQUIRES_EROSILT
            return
        elseif (transport_class /= BRAIDS_TRA_BEDLOAD .and. &
                transport_class /= BRAIDS_TRA_COMBINE) then
            status = BRAIDS_COMPOSE_UNSUPPORTED_CLASS
            return
        end if

        if (transport_class == BRAIDS_TRA_COMBINE .and. depth <= 0.0_real64) then
            status = BRAIDS_COMPOSE_NONPOSITIVE_DEPTH
            return
        end if

        speed = safe_norm2(base_velocity)
        suspended_norm = safe_norm2(raw_suspended_mass)
        if (.not. ieee_is_finite(speed) .or. .not. ieee_is_finite(suspended_norm)) then
            status = BRAIDS_COMPOSE_ARITHMETIC_OVERFLOW
            return
        end if

        if (speed <= 0.0_real64) then
            if (suspended_norm > 0.0_real64) then
                status = BRAIDS_COMPOSE_ZERO_FLOW_SUSPENDED
                return
            end if
            base_direction = 0.0_real64
            raw_signed_rate = 0.0_real64
        else
            base_direction = base_velocity / speed
            if (suspended_norm > 0.0_real64) then
                raw_signed_rate = dot_product(raw_suspended_mass, base_direction)
                alignment_residual = raw_suspended_mass - &
                                     raw_signed_rate * base_direction
                if (safe_norm2(alignment_residual) > &
                    BRAIDS_SUSPENDED_ALIGNMENT_ATOL + &
                    BRAIDS_SUSPENDED_ALIGNMENT_RTOL*suspended_norm) then
                    status = BRAIDS_COMPOSE_SUSPENDED_MISALIGNED
                    return
                end if
            end if
            raw_signed_rate = dot_product(raw_suspended_mass, base_direction)
            if (.not. ieee_is_finite(raw_signed_rate)) then
                status = BRAIDS_COMPOSE_ARITHMETIC_OVERFLOW
                return
            end if
        end if

        ! These are the only global-factor applications in this routine.
        scaled_bed_mass = bed_factor * raw_bed_mass
        signed_suspended_mass_rate = suspended_factor * raw_signed_rate
        scaled_suspended_mass = signed_suspended_mass_rate * base_direction
        if (.not. all(ieee_is_finite(scaled_bed_mass)) .or. &
            .not. all(ieee_is_finite(scaled_suspended_mass)) .or. &
            .not. ieee_is_finite(signed_suspended_mass_rate)) then
            call reject_arithmetic(composed_bed_mass, retained_suspended_mass, &
                                   signed_suspended_mass_rate, &
                                   equilibrium_mass_concentration, status)
            return
        end if

        if (transport_class == BRAIDS_TRA_BEDLOAD) then
            ! Fold the scalar formula channel along the unrotated base-flow
            ! direction.  raw_bed_mass may already have a different direction.
            composed_bed_mass = scaled_bed_mass + &
                                signed_suspended_mass_rate * base_direction
            retained_suspended_mass = 0.0_real64
            equilibrium_mass_concentration = 0.0_real64
        else
            ! TRA_COMBINE keeps the scaled suspended channel separate.  The
            ! sequential division follows eqtran rather than forming a possibly
            ! overflowing denominator product.
            composed_bed_mass = scaled_bed_mass
            retained_suspended_mass = scaled_suspended_mass
            equilibrium_mass_concentration = signed_suspended_mass_rate / &
                (speed + speed_epsilon) / depth
        end if

        if (.not. all(ieee_is_finite(composed_bed_mass)) .or. &
            .not. all(ieee_is_finite(retained_suspended_mass)) .or. &
            .not. ieee_is_finite(equilibrium_mass_concentration)) then
            call reject_arithmetic(composed_bed_mass, retained_suspended_mass, &
                                   signed_suspended_mass_rate, &
                                   equilibrium_mass_concentration, status)
            return
        end if

        valid = .true.
        status = BRAIDS_COMPOSE_OK
    end subroutine braids_compose_delft_channels


    pure function safe_norm2(vector) result(norm)
        real(real64), intent(in) :: vector(2)
        real(real64) :: norm
        real(real64) :: scale

        scale = max(abs(vector(1)), abs(vector(2)))
        if (scale > 0.0_real64) then
            norm = scale * sqrt((vector(1)/scale)**2 + (vector(2)/scale)**2)
        else
            norm = 0.0_real64
        end if
    end function safe_norm2


    pure subroutine reject_arithmetic(composed_bed_mass, retained_suspended_mass, &
            signed_suspended_mass_rate, equilibrium_mass_concentration, status)
        real(real64), intent(out) :: composed_bed_mass(2)
        real(real64), intent(out) :: retained_suspended_mass(2)
        real(real64), intent(out) :: signed_suspended_mass_rate
        real(real64), intent(out) :: equilibrium_mass_concentration
        integer, intent(out) :: status

        composed_bed_mass = 0.0_real64
        retained_suspended_mass = 0.0_real64
        signed_suspended_mass_rate = 0.0_real64
        equilibrium_mass_concentration = 0.0_real64
        status = BRAIDS_COMPOSE_ARITHMETIC_OVERFLOW
    end subroutine reject_arithmetic

end module braids_delft_channel_composer_module
