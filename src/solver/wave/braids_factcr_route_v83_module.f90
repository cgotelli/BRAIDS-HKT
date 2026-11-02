! SPDX-License-Identifier: GPL-3.0-or-later
!
! Additive Q4/v83 owner for locked Delft3D4 FLOW FactCr semantics.
!
! FactCr belongs to critical-shear formation and is consumed by the
! negative Van Rijn routes only when iTauCr=1.  This module exposes the
! common -1/-2/-4 threshold equation and one admitted full constitutive
! point: current-only, pure-sand TraFrm=-1/iTauCr=1.  The existing v66
! producer/composer is called without modification.
module braids_factcr_route_v83_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_fortran_env, only: real64
    use braids_vr1993_route_v66_module, only: &
        BRAIDS_VR1993_ROUTE_COMBINE_V66, &
        braids_vr1993_current_producer_v66, &
        braids_vr1993_compose_route_v66
    implicit none
    private

    integer, parameter, public :: BRAIDS_FACTCR_TRAFRM_V83 = -1
    integer, parameter, public :: BRAIDS_FACTCR_ITAUCR_V83 = 1
    integer, parameter, public :: BRAIDS_FACTCR_FORM_UNIFORM_V83 = 1
    integer, parameter, public :: BRAIDS_FACTCR_FORM_SPATIAL_V83 = 2

    public :: braids_factcr_threshold_v83
    public :: braids_factcr_vr1993_point_v83

contains

    pure subroutine braids_factcr_threshold_v83( &
            formula_code, formation_mode, factcr, d50, solid_density, &
            water_density, gravity, &
            kinematic_viscosity, dstar, critical_shields, unfactored_taucr, &
            effective_taucr, application_count, status)
        integer, intent(in) :: formula_code, formation_mode
        real(real64), intent(in) :: factcr, d50, solid_density
        real(real64), intent(in) :: water_density, gravity, kinematic_viscosity
        real(real64), intent(out) :: dstar, critical_shields
        real(real64), intent(out) :: unfactored_taucr, effective_taucr
        integer, intent(out) :: application_count, status
        real(real64) :: relative_density

        dstar = 0.0_real64
        critical_shields = 0.0_real64
        unfactored_taucr = 0.0_real64
        effective_taucr = 0.0_real64
        application_count = 0
        status = 1
        if (.not.all(ieee_is_finite([factcr, d50, solid_density, &
                water_density, gravity, kinematic_viscosity]))) return
        if (formula_code /= -1 .and. formula_code /= -2 .and. &
                formula_code /= -4) return
        if (formation_mode /= BRAIDS_FACTCR_FORM_UNIFORM_V83 .and. &
                formation_mode /= BRAIDS_FACTCR_FORM_SPATIAL_V83) return
        if (factcr <= 0.0_real64 .or. d50 <= 0.0_real64 .or. &
                solid_density <= water_density .or. water_density <= 0.0_real64 .or. &
                gravity <= 0.0_real64 .or. kinematic_viscosity <= 0.0_real64) return

        relative_density = (solid_density-water_density)/water_density
        dstar = d50*(relative_density*gravity/ &
                (kinematic_viscosity*kinematic_viscosity))**0.3333_real64
        if (dstar <= 4.0_real64) then
            if (formula_code /= -2 .and. .not.(formula_code == -4 .and. &
                    formation_mode == BRAIDS_FACTCR_FORM_SPATIAL_V83)) then
                critical_shields = 0.24_real64/dstar
            else
                critical_shields = 0.115_real64/sqrt(dstar)
            end if
        else if (dstar <= 10.0_real64) then
            critical_shields = 0.14_real64/dstar**0.64_real64
        else if (dstar <= 20.0_real64) then
            critical_shields = 0.04_real64/dstar**0.10_real64
        else if (dstar <= 150.0_real64) then
            critical_shields = 0.013_real64*dstar**0.29_real64
        else
            critical_shields = 0.055_real64
        end if
        unfactored_taucr = (solid_density-water_density)*gravity*d50* &
                           critical_shields
        effective_taucr = factcr*unfactored_taucr
        if (.not.all(ieee_is_finite([dstar, critical_shields, &
                unfactored_taucr, effective_taucr])) .or. &
                min(dstar, critical_shields, unfactored_taucr, &
                    effective_taucr) <= 0.0_real64) then
            dstar = 0.0_real64
            critical_shields = 0.0_real64
            unfactored_taucr = 0.0_real64
            effective_taucr = 0.0_real64
            return
        end if
        application_count = 1
        status = 0
    end subroutine braids_factcr_threshold_v83


    pure subroutine braids_factcr_vr1993_point_v83( &
            factcr, depth, velocity_x, velocity_y, observation_height, &
            d50, d90, solid_density, water_density, gravity, &
            kinematic_viscosity, molecular_schmidt, settling_velocity, &
            roughness_z0, aks_factor, maximum_volume_concentration, &
            additional_shear_stress, minimum_depth, bed_factor, &
            suspended_factor, surface_fraction, dstar, critical_shields, &
            unfactored_taucr, effective_taucr, raw_bed_x, raw_bed_y, &
            raw_concentration, bed_capacity_x, bed_capacity_y, &
            equilibrium_concentration, shear_velocity, efficiency_factor, &
            shear_ratio, reference_concentration, reference_height, &
            application_count, status)
        real(real64), intent(in) :: factcr, depth, velocity_x, velocity_y
        real(real64), intent(in) :: observation_height, d50, d90
        real(real64), intent(in) :: solid_density, water_density, gravity
        real(real64), intent(in) :: kinematic_viscosity, molecular_schmidt
        real(real64), intent(in) :: settling_velocity, roughness_z0, aks_factor
        real(real64), intent(in) :: maximum_volume_concentration
        real(real64), intent(in) :: additional_shear_stress, minimum_depth
        real(real64), intent(in) :: bed_factor, suspended_factor, surface_fraction
        real(real64), intent(out) :: dstar, critical_shields
        real(real64), intent(out) :: unfactored_taucr, effective_taucr
        real(real64), intent(out) :: raw_bed_x, raw_bed_y, raw_concentration
        real(real64), intent(out) :: bed_capacity_x, bed_capacity_y
        real(real64), intent(out) :: equilibrium_concentration, shear_velocity
        real(real64), intent(out) :: efficiency_factor, shear_ratio
        real(real64), intent(out) :: reference_concentration, reference_height
        integer, intent(out) :: application_count, status
        integer :: producer_status, compose_status

        raw_bed_x = 0.0_real64
        raw_bed_y = 0.0_real64
        raw_concentration = 0.0_real64
        bed_capacity_x = 0.0_real64
        bed_capacity_y = 0.0_real64
        equilibrium_concentration = 0.0_real64
        shear_velocity = 0.0_real64
        efficiency_factor = 0.0_real64
        shear_ratio = 0.0_real64
        reference_concentration = 0.0_real64
        reference_height = 0.0_real64
        call braids_factcr_threshold_v83( &
            BRAIDS_FACTCR_TRAFRM_V83, BRAIDS_FACTCR_FORM_UNIFORM_V83, &
            factcr, d50, solid_density, &
            water_density, gravity, kinematic_viscosity, dstar, &
            critical_shields, unfactored_taucr, effective_taucr, &
            application_count, status)
        if (status /= 0) return

        call braids_vr1993_current_producer_v66( &
            depth, velocity_x, velocity_y, observation_height, d50, d90, &
            solid_density, water_density, gravity, kinematic_viscosity, &
            molecular_schmidt, settling_velocity, roughness_z0, aks_factor, &
            maximum_volume_concentration, additional_shear_stress, dstar, &
            effective_taucr, minimum_depth, raw_bed_x, raw_bed_y, &
            raw_concentration, shear_velocity, efficiency_factor, shear_ratio, &
            reference_concentration, reference_height, producer_status)
        if (producer_status /= 0) then
            status = producer_status
            return
        end if
        call braids_vr1993_compose_route_v66( &
            BRAIDS_VR1993_ROUTE_COMBINE_V66, depth, velocity_x, velocity_y, &
            raw_bed_x, raw_bed_y, raw_concentration, bed_factor, &
            suspended_factor, surface_fraction, bed_capacity_x, &
            bed_capacity_y, equilibrium_concentration, compose_status)
        status = compose_status
    end subroutine braids_factcr_vr1993_point_v83

end module braids_factcr_route_v83_module
