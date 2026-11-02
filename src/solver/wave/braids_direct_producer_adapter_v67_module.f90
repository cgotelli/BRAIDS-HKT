! SPDX-License-Identifier: GPL-3.0-or-later
!
! Additive native adapter joining the established v66 current-only direct
! producers to the generic v66 FLOW route composer.  Constitutive outputs are
! already solid-mass-valued: this module performs no density conversion and
! exposes no external secondary-flow rotation input.
module braids_direct_producer_adapter_v67_module
    use, intrinsic :: iso_fortran_env, only: real64
    use braids_vr1993_route_v66_module, only: &
        braids_vr1993_current_producer_v66
    use braids_delft_direct_current_v66_module, only: &
        braids_bijker_current_v66, &
        braids_van_thiel_van_rijn_current_v66, &
        braids_xbeach_soulsby_van_rijn_current_v66
    use braids_direct_vector_route_composer_v66_module, only: &
        braids_compose_direct_vector_route_v66
    implicit none
    private

    integer, parameter, public :: BRAIDS_V67_WIRE_VAN_RIJN_1993 = 67001
    integer, parameter, public :: BRAIDS_V67_WIRE_BIJKER = 67005
    integer, parameter, public :: BRAIDS_V67_WIRE_VAN_THIEL_VAN_RIJN = 67019
    integer, parameter, public :: BRAIDS_V67_WIRE_XBEACH_SOULSBY_VAN_RIJN = 67020

    character(len=52), parameter, public :: BRAIDS_V67_METHOD_VAN_RIJN_1993 = &
        'capacity.van_rijn_1993'
    character(len=52), parameter, public :: BRAIDS_V67_METHOD_BIJKER = &
        'capacity.bijker_current'
    character(len=52), parameter, public :: BRAIDS_V67_METHOD_VAN_THIEL_VAN_RIJN = &
        'capacity.van_thiel_van_rijn_2008_current'
    character(len=52), parameter, public :: BRAIDS_V67_METHOD_XBEACH_SOULSBY_VAN_RIJN = &
        'capacity.xbeach_soulsby_van_rijn_current'

    type, public :: braids_direct_route_controls_v67
        character(len=32) :: route = 'tra_bedload'
        real(real64) :: bed_factor = 1.0_real64
        real(real64) :: suspended_factor = 1.0_real64
        real(real64) :: active_fraction = 1.0_real64
        real(real64) :: speed_epsilon = 1.0e-6_real64
        logical :: defer_surface_fraction_to_face = .false.
    end type braids_direct_route_controls_v67

    type, public :: braids_vr1993_input_v67
        real(real64) :: depth = 0.0_real64
        real(real64) :: velocity_x = 0.0_real64
        real(real64) :: velocity_y = 0.0_real64
        real(real64) :: observation_height = 0.0_real64
        real(real64) :: d50 = 0.0_real64
        real(real64) :: d90 = 0.0_real64
        real(real64) :: solid_density = 0.0_real64
        real(real64) :: water_density = 0.0_real64
        real(real64) :: gravity = 0.0_real64
        real(real64) :: kinematic_viscosity = 0.0_real64
        real(real64) :: molecular_schmidt = 0.0_real64
        real(real64) :: settling_velocity = 0.0_real64
        real(real64) :: roughness_z0 = 0.0_real64
        real(real64) :: aks_factor = 0.0_real64
        real(real64) :: maximum_volume_concentration = 0.0_real64
        real(real64) :: additional_shear_stress = 0.0_real64
        real(real64) :: dstar = 0.0_real64
        real(real64) :: critical_shear_stress = 0.0_real64
        real(real64) :: minimum_depth = 0.0_real64
    end type braids_vr1993_input_v67

    type, public :: braids_bijker_input_v67
        real(real64) :: depth = 0.0_real64
        real(real64) :: u = 0.0_real64
        real(real64) :: v = 0.0_real64
        real(real64) :: d50 = 0.0_real64
        real(real64) :: d90 = 0.0_real64
        real(real64) :: solid_density = 0.0_real64
        real(real64) :: water_density = 0.0_real64
        real(real64) :: gravity = 0.0_real64
        real(real64) :: chezy = 0.0_real64
        real(real64) :: settling_velocity = 0.0_real64
        real(real64) :: cal_bs = 0.0_real64
        real(real64) :: cal_bd = 0.0_real64
        real(real64) :: crit_cs = 0.0_real64
        real(real64) :: crit_cd = 0.0_real64
        real(real64) :: roughness_ks = -999.0_real64
        real(real64) :: formula_settling_velocity = -999.0_real64
        real(real64) :: porosity = -999.0_real64
        real(real64) :: formula_wave_period = -999.0_real64
        real(real64) :: fac_a = 0.0_real64
        real(real64) :: fac_u = 0.0_real64
        real(real64) :: eps_sl = 0.0_real64
        real(real64) :: wave_height = 0.0_real64
        real(real64) :: wave_period = 0.0_real64
        real(real64) :: wavelength = 0.0_real64
        real(real64) :: wave_angle = 0.0_real64
        real(real64) :: orbital_velocity = 0.0_real64
        real(real64) :: breaker_turbulence = 0.0_real64
        real(real64) :: bed_slope_x = 0.0_real64
        real(real64) :: bed_slope_y = 0.0_real64
    end type braids_bijker_input_v67

    type, public :: braids_1920_input_v67
        real(real64) :: depth = 0.0_real64
        real(real64) :: u = 0.0_real64
        real(real64) :: v = 0.0_real64
        real(real64) :: d15 = 0.0_real64
        real(real64) :: d50 = 0.0_real64
        real(real64) :: d90 = 0.0_real64
        real(real64) :: solid_density = 0.0_real64
        real(real64) :: water_density = 0.0_real64
        real(real64) :: gravity = 0.0_real64
        real(real64) :: kinematic_viscosity = 0.0_real64
        real(real64) :: chezy = 0.0_real64
        real(real64) :: facua = 0.0_real64
        real(real64) :: facas = 0.2_real64
        real(real64) :: facsk = 0.15_real64
        integer :: waveform = 2
        integer :: sws = 1
        integer :: lws = 1
        integer :: dilatancy = 0
        real(real64) :: rheea = 0.75_real64
        real(real64) :: pormax = 0.5_real64
        integer :: bedslpini = 0
        real(real64) :: smax = -1.0_real64
        real(real64) :: reposeangle = 30.0_real64
        real(real64) :: cmax = 0.1_real64
        real(real64) :: z0 = 0.006_real64
        real(real64) :: alfad50 = 0.0_real64
        real(real64) :: wave_height = 0.0_real64
        real(real64) :: wave_period = 0.0_real64
        real(real64) :: wavelength = 0.0_real64
        real(real64) :: wave_angle = 0.0_real64
        real(real64) :: orbital_velocity = 0.0_real64
        real(real64) :: breaker_turbulence = 0.0_real64
        real(real64) :: bed_change_rate = 0.0_real64
        real(real64) :: bed_slope_x = 0.0_real64
        real(real64) :: bed_slope_y = 0.0_real64
    end type braids_1920_input_v67

    type, public :: braids_direct_producer_result_v67
        integer :: internal_wire_id = 0
        integer :: output_kind = -1
        real(real64) :: raw_bed_mass(2) = 0.0_real64
        real(real64) :: raw_suspended_mass(2) = 0.0_real64
        real(real64) :: raw_equilibrium_mass_concentration = 0.0_real64
        real(real64) :: producer_diagnostic_concentration = 0.0_real64
        real(real64) :: producer_auxiliary(5) = 0.0_real64
        real(real64) :: composed_bed_mass(2) = 0.0_real64
        real(real64) :: q4_equilibrium_mass_concentration = 0.0_real64
        logical :: valid = .false.
        integer :: producer_status = 1
        integer :: route_status = 1
        integer :: surface_fraction_application_count = -1
        character(len=40) :: surface_fraction_owner = 'invalid'
    end type braids_direct_producer_result_v67

    public :: braids_run_vr1993_adapter_v67
    public :: braids_run_bijker_adapter_v67
    public :: braids_run_van_thiel_van_rijn_adapter_v67
    public :: braids_run_xbeach_soulsby_van_rijn_adapter_v67
    public :: braids_resolve_direct_producer_identity_v67

contains

    pure subroutine clear_result(result, wire)
        type(braids_direct_producer_result_v67), intent(out) :: result
        integer, intent(in) :: wire
        result%internal_wire_id = wire
        result%output_kind = -1
        result%raw_bed_mass = 0.0_real64
        result%raw_suspended_mass = 0.0_real64
        result%raw_equilibrium_mass_concentration = 0.0_real64
        result%producer_diagnostic_concentration = 0.0_real64
        result%producer_auxiliary = 0.0_real64
        result%composed_bed_mass = 0.0_real64
        result%q4_equilibrium_mass_concentration = 0.0_real64
        result%valid = .false.
        result%producer_status = 1
        result%route_status = 1
        result%surface_fraction_application_count = -1
        result%surface_fraction_owner = 'invalid'
    end subroutine clear_result

    pure subroutine finish_result(method, depth, velocity, controls, result)
        character(len=*), intent(in) :: method
        real(real64), intent(in) :: depth, velocity(2)
        type(braids_direct_route_controls_v67), intent(in) :: controls
        type(braids_direct_producer_result_v67), intent(inout) :: result
        real(real64) :: route_fraction
        if (result%producer_status /= 0) return
        if (controls%defer_surface_fraction_to_face) then
            route_fraction = 1.0_real64
            result%surface_fraction_application_count = 0
            result%surface_fraction_owner = 'downstream_face_exner_lifecycle'
        else
            route_fraction = controls%active_fraction
            result%surface_fraction_application_count = 1
            result%surface_fraction_owner = 'direct_route_composer_v66'
        end if
        call braids_compose_direct_vector_route_v66( &
            method, controls%route, depth, velocity, result%raw_bed_mass, &
            result%raw_suspended_mass, &
            result%raw_equilibrium_mass_concentration, controls%bed_factor, &
            controls%suspended_factor, route_fraction, &
            controls%speed_epsilon, result%output_kind, &
            result%composed_bed_mass, &
            result%q4_equilibrium_mass_concentration, result%valid, &
            result%route_status)
    end subroutine finish_result

    pure subroutine braids_resolve_direct_producer_identity_v67( &
            method, wire, reference_selector, status)
        character(len=*), intent(in) :: method
        integer, intent(out) :: wire, reference_selector, status
        wire = 0
        reference_selector = 0
        status = 1
        select case (trim(method))
        case ('capacity.van_rijn_1993')
            wire = BRAIDS_V67_WIRE_VAN_RIJN_1993
            reference_selector = -1
        case ('capacity.bijker_current')
            wire = BRAIDS_V67_WIRE_BIJKER
            reference_selector = 5
        case ('capacity.van_thiel_van_rijn_2008_current')
            wire = BRAIDS_V67_WIRE_VAN_THIEL_VAN_RIJN
            reference_selector = 19
        case ('capacity.xbeach_soulsby_van_rijn_current')
            wire = BRAIDS_V67_WIRE_XBEACH_SOULSBY_VAN_RIJN
            reference_selector = 20
        case default
            return
        end select
        status = 0
    end subroutine braids_resolve_direct_producer_identity_v67

    pure subroutine braids_run_vr1993_adapter_v67(input, controls, result)
        type(braids_vr1993_input_v67), intent(in) :: input
        type(braids_direct_route_controls_v67), intent(in) :: controls
        type(braids_direct_producer_result_v67), intent(out) :: result
        real(real64) :: shear_velocity, efficiency_factor, shear_ratio
        real(real64) :: reference_concentration, reference_height

        call clear_result(result, BRAIDS_V67_WIRE_VAN_RIJN_1993)
        call braids_vr1993_current_producer_v66( &
            input%depth, input%velocity_x, input%velocity_y, &
            input%observation_height, input%d50, input%d90, &
            input%solid_density, input%water_density, input%gravity, &
            input%kinematic_viscosity, input%molecular_schmidt, &
            input%settling_velocity, input%roughness_z0, input%aks_factor, &
            input%maximum_volume_concentration, input%additional_shear_stress, &
            input%dstar, input%critical_shear_stress, input%minimum_depth, &
            result%raw_bed_mass(1), result%raw_bed_mass(2), &
            result%raw_equilibrium_mass_concentration, shear_velocity, &
            efficiency_factor, shear_ratio, reference_concentration, &
            reference_height, result%producer_status)
        result%producer_auxiliary = [shear_velocity, efficiency_factor, &
            shear_ratio, reference_concentration, reference_height]
        call finish_result('van_rijn_1993_current', input%depth, &
            [input%velocity_x, input%velocity_y], controls, result)
    end subroutine braids_run_vr1993_adapter_v67

    pure subroutine braids_run_bijker_adapter_v67(input, controls, result)
        type(braids_bijker_input_v67), intent(in) :: input
        type(braids_direct_route_controls_v67), intent(in) :: controls
        type(braids_direct_producer_result_v67), intent(out) :: result
        real(real64) :: auxiliary(3)

        call clear_result(result, BRAIDS_V67_WIRE_BIJKER)
        call braids_bijker_current_v66(input%depth, input%u, input%v, &
            input%d50, input%d90, input%solid_density, input%water_density, &
            input%gravity, input%chezy, input%settling_velocity, input%cal_bs, &
            input%cal_bd, input%crit_cs, input%crit_cd, input%roughness_ks, &
            input%formula_settling_velocity, input%porosity, &
            input%formula_wave_period, input%fac_a, input%fac_u, input%eps_sl, &
            input%wave_height, input%wave_period, input%wavelength, &
            input%wave_angle, input%orbital_velocity, input%breaker_turbulence, &
            input%bed_slope_x, input%bed_slope_y, result%raw_bed_mass, &
            result%raw_suspended_mass, &
            result%producer_diagnostic_concentration, auxiliary, &
            result%producer_status)
        result%producer_auxiliary(1:3) = auxiliary
        ! Bijker's concentration output is diagnostic: only its explicit
        ! suspended vector enters the normalized route ABI.
        result%raw_equilibrium_mass_concentration = 0.0_real64
        call finish_result('bijker_current', input%depth, [input%u, input%v], &
            controls, result)
    end subroutine braids_run_bijker_adapter_v67

    pure subroutine braids_run_van_thiel_van_rijn_adapter_v67( &
            input, controls, result)
        type(braids_1920_input_v67), intent(in) :: input
        type(braids_direct_route_controls_v67), intent(in) :: controls
        type(braids_direct_producer_result_v67), intent(out) :: result
        real(real64) :: auxiliary(3)

        call clear_result(result, BRAIDS_V67_WIRE_VAN_THIEL_VAN_RIJN)
        call braids_van_thiel_van_rijn_current_v66(input%depth, input%u, &
            input%v, input%d15, input%d50, input%d90, input%solid_density, &
            input%water_density, input%gravity, input%kinematic_viscosity, &
            input%chezy, input%facua, input%facas, input%facsk, input%waveform, &
            input%sws, input%lws, input%dilatancy, input%rheea, input%pormax, &
            input%bedslpini, input%smax, input%reposeangle, input%cmax, &
            input%alfad50, input%wave_height, input%wave_period, &
            input%wavelength, input%wave_angle, input%orbital_velocity, &
            input%breaker_turbulence, input%bed_change_rate, &
            input%bed_slope_x, input%bed_slope_y, result%raw_bed_mass, &
            result%raw_suspended_mass, &
            result%raw_equilibrium_mass_concentration, auxiliary, &
            result%producer_status)
        result%producer_auxiliary(1:3) = auxiliary
        call finish_result('van_thiel_van_rijn_2008_current', input%depth, &
            [input%u, input%v], controls, result)
    end subroutine braids_run_van_thiel_van_rijn_adapter_v67

    pure subroutine braids_run_xbeach_soulsby_van_rijn_adapter_v67( &
            input, controls, result)
        type(braids_1920_input_v67), intent(in) :: input
        type(braids_direct_route_controls_v67), intent(in) :: controls
        type(braids_direct_producer_result_v67), intent(out) :: result
        real(real64) :: auxiliary(3)

        call clear_result(result, BRAIDS_V67_WIRE_XBEACH_SOULSBY_VAN_RIJN)
        call braids_xbeach_soulsby_van_rijn_current_v66(input%depth, input%u, &
            input%v, input%d15, input%d50, input%d90, input%solid_density, &
            input%water_density, input%gravity, input%kinematic_viscosity, &
            input%chezy, input%facua, input%facas, input%facsk, input%waveform, &
            input%sws, input%lws, input%dilatancy, input%rheea, input%pormax, &
            input%bedslpini, input%smax, input%reposeangle, input%cmax, &
            input%z0, input%alfad50, input%wave_height, input%wave_period, &
            input%wavelength, input%wave_angle, input%orbital_velocity, &
            input%breaker_turbulence, input%bed_change_rate, &
            input%bed_slope_x, input%bed_slope_y, result%raw_bed_mass, &
            result%raw_suspended_mass, &
            result%raw_equilibrium_mass_concentration, auxiliary, &
            result%producer_status)
        result%producer_auxiliary(1:3) = auxiliary
        call finish_result('xbeach_soulsby_van_rijn_current', input%depth, &
            [input%u, input%v], controls, result)
    end subroutine braids_run_xbeach_soulsby_van_rijn_adapter_v67

end module braids_direct_producer_adapter_v67_module
