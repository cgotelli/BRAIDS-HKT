! SPDX-License-Identifier: GPL-3.0-only
!
! Bounded, current-only transcriptions of three direct-vector Delft3D4 FLOW
! transport formulae.  This module is intentionally isolated from the public
! solver dispatch until its output ownership is composed with the live Q3-H6
! face/Exner path.
!
! Locked, unmodified source oracle (commit
! c05ed672f0a7fe81c5b9867ef354cfdd53c25379):
!   tranb5.f90  ca5933ccda6592174c627c81d2d10e977064dd47f96560d349b1a3fca50a2070
!   fgyint.f90  135138690f0b56a0f89f18cd9f9bc85149f29458f76b167f0ce54ec695730e19
!   termfy.f90  55bb8621b7d09e7a64d4c4e1a8b4b38697801c3fcfb7f11792c56e9e9e0e11e5
!   termgy.f90  6d6d1b81af9140f299fbd8a3e04cfe10163f2a9db1d235651995f4a9232c4618
!   trab19.f90  bfc530b573b95c4467a9de4bd34e25757f3990d489426db995b28d4b7ad34c63
!   trab20.f90  a7cbf17f232fc6edae5a1cf5f7c9b2ca1e1e0626aee6dbb5270efc1c245e4d19
!
! The classic FLOW call site supplies zero wave forcing and, for 19/20, zero
! D15MX/BLCHG/DZDX/DZDY/KWTUR.  Nonzero wave state and FM-derived
! dilatancy/initial-slope modes are rejected rather than silently discarded.
! Positive IFORM outputs are converted from volume to solid mass exactly once.
module braids_delft_direct_current_v66_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: BRAIDS_DIRECT_OK = 0
    integer, parameter, public :: BRAIDS_DIRECT_INVALID = 1
    integer, parameter, public :: BRAIDS_DIRECT_WAVE_UNSUPPORTED = 2
    integer, parameter, public :: BRAIDS_DIRECT_CLASSIC_STATE_UNSUPPORTED = 3
    integer, parameter, public :: BRAIDS_DIRECT_OPTION_UNSUPPORTED = 4

    real(8), parameter :: BIJKER_VON_KARMAN = real(0.41, 8)
    real(8), parameter :: D50_REFERENCE = 0.000225d0

    public :: braids_bijker_current_v66
    public :: braids_van_thiel_van_rijn_current_v66
    public :: braids_xbeach_soulsby_van_rijn_current_v66

contains

    pure logical function any_wave_forcing(wave_height, wave_period, wavelength, &
                                           wave_angle, orbital_velocity, breaker_turbulence)
        real(8), intent(in) :: wave_height, wave_period, wavelength
        real(8), intent(in) :: wave_angle, orbital_velocity, breaker_turbulence
        any_wave_forcing = wave_height /= 0.d0 .or. wave_period /= 0.d0 .or. &
                           wavelength /= 0.d0 .or. wave_angle /= 0.d0 .or. &
                           orbital_velocity /= 0.d0 .or. breaker_turbulence /= 0.d0
    end function any_wave_forcing

    pure real(8) function bijker_term_f(n, y, z) result(value)
        integer, intent(in) :: n
        real(8), intent(in) :: y, z
        integer :: i
        real(8) :: numerator, denominator, sign_value

        sign_value = merge(1.d0, -1.d0, mod(n + 1, 2) == 0)
        if (abs(real(n, 8) - z) <= 1.d-10) then
            value = sign_value * z * log(y)
        else if (n == 1) then
            value = y**(1.d0 - z) / (1.d0 - z)
        else
            numerator = 1.d0
            denominator = 1.d0
            do i = 2, n
                numerator = numerator * (z - real(i, 8) + 2.d0)
                denominator = denominator * real(i, 8)
            end do
            denominator = denominator / real(n, 8)
            value = sign_value * numerator * y**(real(n, 8) - z) / &
                    (denominator * (real(n, 8) - z))
        end if
    end function bijker_term_f

    pure real(8) function bijker_term_g(n, y, z) result(value)
        integer, intent(in) :: n
        real(8), intent(in) :: y, z
        integer :: i
        real(8) :: numerator, denominator, exponent, sign_value

        sign_value = merge(1.d0, -1.d0, mod(n + 1, 2) == 0)
        if (abs(real(n, 8) - z) <= 1.d-10) then
            value = sign_value * z * log(y) * log(y) * 0.5d0
        else if (n == 1) then
            exponent = 1.d0 - z
            value = y**exponent * (exponent * log(y) - 1.d0) / exponent**2
        else
            numerator = 1.d0
            denominator = 1.d0
            do i = 2, n
                numerator = numerator * (z - real(i, 8) + 2.d0)
                denominator = denominator * real(i, 8)
            end do
            denominator = denominator / real(n, 8)
            exponent = real(n, 8) - z
            value = sign_value * numerator * y**exponent / denominator * &
                    (exponent * log(y) - 1.d0) / exponent**2
        end if
    end function bijker_term_g

    pure real(8) function bijker_integral(a, b, z, use_g) result(value)
        real(8), intent(in) :: a, b, z
        logical, intent(in) :: use_g
        integer :: i
        real(8) :: rest, sum_value

        sum_value = 0.d0
        do i = 1, 100
            if (use_g) then
                rest = bijker_term_g(i, b, z) - bijker_term_g(i, a, z)
            else
                rest = bijker_term_f(i, b, z) - bijker_term_f(i, a, z)
            end if
            sum_value = sum_value + rest
            if (rest == 0.d0 .or. sum_value == 0.d0) exit
            if (abs(rest / sum_value) < 0.001d0) exit
        end do
        value = sum_value
    end function bijker_integral

    pure subroutine zero_direct_outputs(bed_mass, suspended_mass, concentration_mass, &
                                        auxiliary)
        real(8), intent(out) :: bed_mass(2), suspended_mass(2)
        real(8), intent(out) :: concentration_mass, auxiliary(3)
        bed_mass = 0.d0
        suspended_mass = 0.d0
        concentration_mass = 0.d0
        auxiliary = 0.d0
    end subroutine zero_direct_outputs

    pure subroutine braids_bijker_current_v66(depth, u, v, d50, d90, solid_density, &
            water_density, gravity, chezy, settling_velocity, cal_bs, cal_bd, &
            crit_cs, crit_cd, roughness_ks, formula_settling_velocity, porosity, &
            formula_wave_period, fac_a, fac_u, eps_sl, wave_height, wave_period, &
            wavelength, wave_angle, orbital_velocity, breaker_turbulence, &
            bed_slope_x, bed_slope_y, bed_mass, suspended_mass, concentration_mass, &
            auxiliary, status)
        ! FLOW TraFrm=5, wave-free subset.  concentration_mass is a diagnostic;
        ! Bijker owns explicit suspended vectors, not the equilibrium-
        ! concentration route used by TraFrm 19/20.
        real(8), intent(in) :: depth, u, v, d50, d90, solid_density, water_density
        real(8), intent(in) :: gravity, chezy, settling_velocity
        real(8), intent(in) :: cal_bs, cal_bd, crit_cs, crit_cd
        real(8), intent(in) :: roughness_ks, formula_settling_velocity, porosity
        real(8), intent(in) :: formula_wave_period, fac_a, fac_u, eps_sl
        real(8), intent(in) :: wave_height, wave_period, wavelength, wave_angle
        real(8), intent(in) :: orbital_velocity, breaker_turbulence
        real(8), intent(in) :: bed_slope_x, bed_slope_y
        real(8), intent(out) :: bed_mass(2), suspended_mass(2), concentration_mass
        real(8), intent(out) :: auxiliary(3)
        integer, intent(out) :: status
        real(8) :: speed, delta, roughness, fall_velocity, relhrms, bfactor
        real(8) :: c90, rmu, current_sq, exponent_arg, friction, shear_velocity
        real(8) :: z, bed_volume_scale, rkh, integral_f, integral_g
        real(8) :: concentration_volume

        call zero_direct_outputs(bed_mass, suspended_mass, concentration_mass, auxiliary)
        status = BRAIDS_DIRECT_INVALID
        if (.not. all(ieee_is_finite([depth, u, v, d50, d90, solid_density, &
                water_density, gravity, chezy, settling_velocity, cal_bs, cal_bd, &
                crit_cs, crit_cd, roughness_ks, formula_settling_velocity, porosity, &
                formula_wave_period, fac_a, fac_u, eps_sl, wave_height, wave_period, &
                wavelength, wave_angle, orbital_velocity, breaker_turbulence, &
                bed_slope_x, bed_slope_y]))) return
        if (any_wave_forcing(wave_height, wave_period, wavelength, wave_angle, &
                             orbital_velocity, breaker_turbulence)) then
            status = BRAIDS_DIRECT_WAVE_UNSUPPORTED
            return
        end if
        if (bed_slope_x /= 0.d0 .or. bed_slope_y /= 0.d0) then
            status = BRAIDS_DIRECT_CLASSIC_STATE_UNSUPPORTED
            return
        end if
        if (eps_sl > 0.d0) then
            status = BRAIDS_DIRECT_OPTION_UNSUPPORTED
            return
        end if
        if (depth < 0.d0 .or. d50 <= 0.d0 .or. d90 <= 0.d0 .or. &
            solid_density <= water_density .or. water_density <= 0.d0 .or. &
            gravity <= 0.d0 .or. chezy <= 0.d0 .or. settling_velocity < 0.d0 .or. &
            porosity < 0.d0 .or. porosity >= 1.d0 .or. formula_wave_period <= 0.d0) return

        status = BRAIDS_DIRECT_OK
        if (depth == 0.d0) return
        roughness = roughness_ks
        if (roughness < 0.d0) roughness = 12.d0 * depth * 10.d0**(-chezy / 18.d0)
        fall_velocity = formula_settling_velocity
        if (fall_velocity < 0.d0) fall_velocity = settling_velocity
        if (.not. all(ieee_is_finite([roughness, fall_velocity])) .or. &
            roughness <= 0.d0 .or. fall_velocity < 0.d0) return
        if (depth / roughness <= 1.33d0 .or. depth > 200.d0) return

        speed = sqrt(u*u + v*v)
        if (.not. ieee_is_finite(speed)) then
            status = BRAIDS_DIRECT_INVALID
            return
        end if
        delta = solid_density / water_density - 1.d0
        relhrms = 0.d0
        if (crit_cd < relhrms .and. relhrms < crit_cs) then
            bfactor = (cal_bs - cal_bd) / (crit_cs - crit_cd) * relhrms + &
                      cal_bd - (cal_bs - cal_bd) / (crit_cs - crit_cd) * crit_cd
        else if (relhrms <= crit_cd) then
            bfactor = cal_bd
        else
            bfactor = cal_bs
        end if
        c90 = 18.d0 * log10(12.d0 * depth / d90)
        rmu = chezy / c90
        if (.not. ieee_is_finite(rmu) .or. rmu < 0.d0) then
            status = BRAIDS_DIRECT_INVALID
            call zero_direct_outputs(bed_mass, suspended_mass, concentration_mass, auxiliary)
            return
        end if
        rmu = rmu * sqrt(rmu)
        current_sq = 0.d0
        if (speed > 1.d-10) current_sq = speed*speed
        friction = gravity / chezy**2
        if (current_sq > 1.d-20) then
            exponent_arg = -0.27d0 * delta * d50 * chezy**2 / (rmu * current_sq)
            exponent_arg = max(-50.d0, min(50.d0, exponent_arg))
            shear_velocity = sqrt(friction) * sqrt(current_sq)
            z = min(fall_velocity / BIJKER_VON_KARMAN / shear_velocity, 8.d0)
        else
            exponent_arg = -50.d0
            z = 8.d0
        end if
        bed_volume_scale = bfactor * d50 / chezy * sqrt(gravity) * &
                           exp(exponent_arg) * (1.d0 - porosity)
        rkh = roughness / depth
        integral_f = 0.216d0 * rkh**(z - 1.d0) / (1.d0 - rkh)**z * &
                     bijker_integral(rkh, 1.d0, z, .false.)
        integral_g = 0.216d0 * rkh**(z - 1.d0) / (1.d0 - rkh)**z * &
                     bijker_integral(rkh, 1.d0, z, .true.)
        concentration_volume = 1.83d0 * bed_volume_scale * &
                               (integral_f * log(33.d0 / rkh) + integral_g)
        concentration_mass = solid_density * concentration_volume
        auxiliary = [z, solid_density * bed_volume_scale, speed]
        if (speed > 1.d-10) then
            bed_mass = solid_density * bed_volume_scale * [u, v]
            suspended_mass = concentration_mass * [u, v]
        end if
        if (.not. all(ieee_is_finite([bed_mass, suspended_mass, concentration_mass, &
                                      auxiliary]))) then
            status = BRAIDS_DIRECT_INVALID
            call zero_direct_outputs(bed_mass, suspended_mass, concentration_mass, auxiliary)
        end if
    end subroutine braids_bijker_current_v66

    pure subroutine direct_1920_current(is_xbeach, depth, u, v, d15, d50, d90, &
            solid_density, water_density, gravity, kinematic_viscosity, chezy, &
            facua, facas, facsk, waveform, sws, lws, dilatancy, rheea, pormax, &
            bedslpini, smax, reposeangle, cmax, z0_input, alfad50, wave_height, &
            wave_period, wavelength, wave_angle, orbital_velocity, breaker_turbulence, &
            bed_change_rate, bed_slope_x, bed_slope_y, bed_mass, suspended_mass, &
            concentration_mass, auxiliary, status)
        logical, intent(in) :: is_xbeach
        real(8), intent(in) :: depth, u, v, d15, d50, d90
        real(8), intent(in) :: solid_density, water_density, gravity
        real(8), intent(in) :: kinematic_viscosity, chezy
        real(8), intent(in) :: facua, facas, facsk, rheea, pormax, smax
        real(8), intent(in) :: reposeangle, cmax, z0_input, alfad50
        integer, intent(in) :: waveform, sws, lws, dilatancy, bedslpini
        real(8), intent(in) :: wave_height, wave_period, wavelength, wave_angle
        real(8), intent(in) :: orbital_velocity, breaker_turbulence
        real(8), intent(in) :: bed_change_rate, bed_slope_x, bed_slope_y
        real(8), intent(out) :: bed_mass(2), suspended_mass(2), concentration_mass
        real(8), intent(out) :: auxiliary(3)
        integer, intent(out) :: status
        real(8) :: speed, delta, dstar, current_critical, wave_critical, critical
        real(8) :: b2, cf, drag, z0, asb, ass, effective_speed_sq, effective_speed
        real(8) :: bed_equilibrium, suspended_equilibrium, maximum_half_depth
        real(8) :: smax_limited, cmax_limited, alfa_limited, grain_multiplier

        call zero_direct_outputs(bed_mass, suspended_mass, concentration_mass, auxiliary)
        status = BRAIDS_DIRECT_INVALID
        if (.not. all(ieee_is_finite([depth, u, v, d15, d50, d90, solid_density, &
                water_density, gravity, kinematic_viscosity, chezy, facua, facas, &
                facsk, rheea, pormax, smax, reposeangle, cmax, z0_input, alfad50, &
                wave_height, wave_period, wavelength, wave_angle, orbital_velocity, &
                breaker_turbulence, bed_change_rate, bed_slope_x, bed_slope_y]))) return
        if (any_wave_forcing(wave_height, wave_period, wavelength, wave_angle, &
                             orbital_velocity, breaker_turbulence)) then
            status = BRAIDS_DIRECT_WAVE_UNSUPPORTED
            return
        end if
        if (bed_change_rate /= 0.d0 .or. bed_slope_x /= 0.d0 .or. bed_slope_y /= 0.d0) then
            status = BRAIDS_DIRECT_CLASSIC_STATE_UNSUPPORTED
            return
        end if
        if (dilatancy == 1 .or. bedslpini == 1 .or. bedslpini == 2) then
            status = BRAIDS_DIRECT_OPTION_UNSUPPORTED
            return
        end if
        if (depth < 0.d0 .or. d15 < 0.d0 .or. d50 <= 0.d0 .or. d90 <= 0.d0 .or. &
            solid_density <= water_density .or. water_density <= 0.d0 .or. &
            gravity <= 0.d0 .or. kinematic_viscosity <= 0.d0 .or. chezy <= 0.d0) return

        ! Keep source parameters present in the ABI even where zero wave state
        ! or rejected FM-only options make them inert.
        speed = sqrt(u*u + v*v)
        if (.not. ieee_is_finite(speed)) return
        status = BRAIDS_DIRECT_OK
        if (speed < 1.d-6 .or. depth > 200.d0 .or. depth < 0.01d0) return

        delta = solid_density / water_density - 1.d0
        dstar = (delta * gravity / 1.d-12)**(1.d0 / 3.d0) * d50
        if (d50 <= 0.0005d0) then
            current_critical = 0.19d0 * d50**0.1d0 * log10(4.d0 * depth / d90)
        else if (d50 <= 0.002d0) then
            current_critical = 8.5d0 * d50**0.6d0 * log10(4.d0 * depth / d90)
        else
            if (is_xbeach) then
                current_critical = 8.5d0 * d50**0.6d0 * log10(4.d0 * depth / d90)
            else
                current_critical = 1.3d0 * sqrt(delta * gravity * d50) * &
                                   (depth / d50)**(0.5d0 / 3.d0)
            end if
        end if

        if (is_xbeach) then
            critical = current_critical
        else
            ! Wave period is exactly zero in the classic current-only call.
            ! The resulting zero wave threshold remains in the low-speed B2
            ! blend and is a deliberate locked-source quirk.
            wave_critical = 0.d0
            b2 = speed / max(speed, 5.d-3)
            critical = b2 * current_critical + (1.d0 - b2) * wave_critical
        end if
        cf = gravity / chezy**2
        smax_limited = max(-1.d0, min(3.d0, smax))
        if (smax_limited < 0.d0) smax_limited = huge(0.d0) * 1.d-20
        cmax_limited = max(0.d0, min(1.d0, cmax))
        alfa_limited = max(0.d0, min(1.5d0, alfad50))

        if (is_xbeach) then
            z0 = max(0.0001d0, min(0.05d0, z0_input))
            drag = (0.40d0 / (log(max(depth, 10.d0*z0) / z0) - 1.d0))**2
            asb = 0.005d0 * depth * &
                  (d50 / depth / (delta * gravity * d50))**1.2d0
            ass = 0.012d0 * d50 * dstar**(-0.6d0) / &
                  (delta * gravity * d50)**1.2d0
            effective_speed_sq = speed**2
            effective_speed_sq = min(effective_speed_sq, &
                                     smax_limited * gravity / cf * d50 * delta)
        else
            asb = 0.015d0 * depth * (d50 / depth)**1.2d0 / &
                  (delta * gravity * d50)**0.75d0
            ass = 0.012d0 * d50 * dstar**(-0.6d0) / &
                  (delta * gravity * d50)**1.2d0
            effective_speed_sq = speed**2
            effective_speed_sq = min(effective_speed_sq, &
                                     smax_limited * gravity / max(cf, 1.d-10) * &
                                     d50 * delta)
        end if
        effective_speed = sqrt(effective_speed_sq)
        bed_equilibrium = 0.d0
        suspended_equilibrium = 0.d0
        if (effective_speed > critical) then
            if (is_xbeach) then
                bed_equilibrium = asb * (effective_speed - critical)**2.4d0
            else
                bed_equilibrium = asb * (effective_speed - critical)**1.5d0
            end if
            suspended_equilibrium = ass * (effective_speed - critical)**2.4d0
        end if
        if (alfa_limited > 0.d0) then
            grain_multiplier = (D50_REFERENCE / d50)**alfa_limited
            bed_equilibrium = bed_equilibrium * grain_multiplier
            suspended_equilibrium = suspended_equilibrium * grain_multiplier
        end if
        maximum_half_depth = cmax_limited * depth / 2.d0
        bed_equilibrium = min(bed_equilibrium, maximum_half_depth)
        concentration_mass = solid_density * &
                             min(suspended_equilibrium, maximum_half_depth) / depth
        bed_mass = solid_density * bed_equilibrium * [u, v]
        ! TraFrm 19/20 own an equilibrium concentration, not an explicit
        ! suspended vector.  Keep this channel exactly zero so a caller cannot
        ! accidentally apply both routes.
        suspended_mass = 0.d0
        auxiliary = [critical, critical, effective_speed]
        if (.not. all(ieee_is_finite([bed_mass, suspended_mass, concentration_mass, &
                                      auxiliary]))) then
            status = BRAIDS_DIRECT_INVALID
            call zero_direct_outputs(bed_mass, suspended_mass, concentration_mass, auxiliary)
        end if
    end subroutine direct_1920_current

    pure subroutine braids_van_thiel_van_rijn_current_v66(depth, u, v, d15, d50, d90, &
            solid_density, water_density, gravity, kinematic_viscosity, chezy, &
            facua, facas, facsk, waveform, sws, lws, dilatancy, rheea, pormax, &
            bedslpini, smax, reposeangle, cmax, alfad50, wave_height, wave_period, &
            wavelength, wave_angle, orbital_velocity, breaker_turbulence, &
            bed_change_rate, bed_slope_x, bed_slope_y, bed_mass, suspended_mass, &
            concentration_mass, auxiliary, status)
        real(8), intent(in) :: depth, u, v, d15, d50, d90, solid_density, water_density
        real(8), intent(in) :: gravity, kinematic_viscosity, chezy
        real(8), intent(in) :: facua, facas, facsk, rheea, pormax, smax
        real(8), intent(in) :: reposeangle, cmax, alfad50
        integer, intent(in) :: waveform, sws, lws, dilatancy, bedslpini
        real(8), intent(in) :: wave_height, wave_period, wavelength, wave_angle
        real(8), intent(in) :: orbital_velocity, breaker_turbulence
        real(8), intent(in) :: bed_change_rate, bed_slope_x, bed_slope_y
        real(8), intent(out) :: bed_mass(2), suspended_mass(2), concentration_mass
        real(8), intent(out) :: auxiliary(3)
        integer, intent(out) :: status
        call direct_1920_current(.false., depth, u, v, d15, d50, d90, &
            solid_density, water_density, gravity, kinematic_viscosity, chezy, &
            facua, facas, facsk, waveform, sws, lws, dilatancy, rheea, pormax, &
            bedslpini, smax, reposeangle, cmax, 0.006d0, alfad50, wave_height, &
            wave_period, wavelength, wave_angle, orbital_velocity, breaker_turbulence, &
            bed_change_rate, bed_slope_x, bed_slope_y, bed_mass, suspended_mass, &
            concentration_mass, auxiliary, status)
    end subroutine braids_van_thiel_van_rijn_current_v66

    pure subroutine braids_xbeach_soulsby_van_rijn_current_v66(depth, u, v, d15, d50, d90, &
            solid_density, water_density, gravity, kinematic_viscosity, chezy, &
            facua, facas, facsk, waveform, sws, lws, dilatancy, rheea, pormax, &
            bedslpini, smax, reposeangle, cmax, z0, alfad50, wave_height, wave_period, &
            wavelength, wave_angle, orbital_velocity, breaker_turbulence, &
            bed_change_rate, bed_slope_x, bed_slope_y, bed_mass, suspended_mass, &
            concentration_mass, auxiliary, status)
        real(8), intent(in) :: depth, u, v, d15, d50, d90, solid_density, water_density
        real(8), intent(in) :: gravity, kinematic_viscosity, chezy
        real(8), intent(in) :: facua, facas, facsk, rheea, pormax, smax
        real(8), intent(in) :: reposeangle, cmax, z0, alfad50
        integer, intent(in) :: waveform, sws, lws, dilatancy, bedslpini
        real(8), intent(in) :: wave_height, wave_period, wavelength, wave_angle
        real(8), intent(in) :: orbital_velocity, breaker_turbulence
        real(8), intent(in) :: bed_change_rate, bed_slope_x, bed_slope_y
        real(8), intent(out) :: bed_mass(2), suspended_mass(2), concentration_mass
        real(8), intent(out) :: auxiliary(3)
        integer, intent(out) :: status
        call direct_1920_current(.true., depth, u, v, d15, d50, d90, &
            solid_density, water_density, gravity, kinematic_viscosity, chezy, &
            facua, facas, facsk, waveform, sws, lws, dilatancy, rheea, pormax, &
            bedslpini, smax, reposeangle, cmax, z0, alfad50, wave_height, &
            wave_period, wavelength, wave_angle, orbital_velocity, breaker_turbulence, &
            bed_change_rate, bed_slope_x, bed_slope_y, bed_mass, suspended_mass, &
            concentration_mass, auxiliary, status)
    end subroutine braids_xbeach_soulsby_van_rijn_current_v66

end module braids_delft_direct_current_v66_module
