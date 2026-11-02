! SPDX-License-Identifier: GPL-3.0-or-later
!
! Current-only Delft3D4 FLOW Van Rijn 1993 producer and Q3 ownership seam.
!
! The constitutive order follows the locked, read-only morphology routines
! bedbc1993, calseddf1993, tram1, bedtr1993 and eqtran.  The fixed 20-layer
! column is BRAIDS' 2DH approximation.  Wave terms, non-algebraic turbulence
! profiles and water-column time integration are intentionally outside this
! Q3 producer and must be rejected by its caller.
module braids_vr1993_route_v66_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_fortran_env, only: real64
    implicit none
    private

    integer, parameter, public :: BRAIDS_VR1993_ROUTE_BEDLOAD_V66 = 1
    integer, parameter, public :: BRAIDS_VR1993_ROUTE_COMBINE_V66 = 2
    integer, parameter :: vertical_layers = 20
    real(real64), parameter :: eps = 1.0e-6_real64
    real(real64), parameter :: von_karman = 0.41_real64
    real(real64), parameter :: sigma(vertical_layers) = [ &
        -0.0874_real64,-0.2472_real64,-0.3797_real64,-0.4897_real64, &
        -0.5809_real64,-0.6565_real64,-0.7193_real64,-0.7713_real64, &
        -0.8145_real64,-0.8503_real64,-0.8800_real64,-0.9046_real64, &
        -0.9250_real64,-0.9419_real64,-0.9560_real64,-0.9676_real64, &
        -0.9773_real64,-0.9854_real64,-0.9920_real64,-0.9975_real64]
    real(real64), parameter :: thickness(vertical_layers) = [ &
        0.1747_real64,0.1449_real64,0.1202_real64,0.0997_real64, &
        0.0827_real64,0.0686_real64,0.0569_real64,0.0472_real64, &
        0.0391_real64,0.0325_real64,0.0269_real64,0.0223_real64, &
        0.0185_real64,0.0154_real64,0.0127_real64,0.0106_real64, &
        0.0088_real64,0.0073_real64,0.0060_real64,0.0050_real64]

    public :: braids_vr1993_current_producer_v66
    public :: braids_vr1993_compose_route_v66

contains

    pure subroutine braids_vr1993_current_producer_v66( &
            depth, velocity_x, velocity_y, observation_height, d50, d90, &
            solid_density, water_density, gravity, kinematic_viscosity, &
            molecular_schmidt, settling_velocity, roughness_z0, aks_factor, &
            maximum_volume_concentration, additional_shear_stress, dstar, &
            critical_shear_stress, minimum_depth, raw_bed_x, raw_bed_y, &
            conc2d_unfractioned, shear_velocity, efficiency_factor, &
            shear_ratio, reference_concentration, reference_height, status)
        ! Outputs use Delft's pre-calibration/pre-fraction ownership:
        !   raw_bed_*              solid MASS transport [kg/(m s)]
        !   conc2d_unfractioned    dry MASS concentration [kg/m3]
        ! Both already contain solid density.  Neither may receive another
        ! density conversion.  conc2d_unfractioned deliberately excludes the
        ! active-surface fraction and Sus calibration; the route composer below
        ! owns those factors exactly once.
        real(real64), intent(in) :: depth, velocity_x, velocity_y
        real(real64), intent(in) :: observation_height, d50, d90
        real(real64), intent(in) :: solid_density, water_density, gravity
        real(real64), intent(in) :: kinematic_viscosity, molecular_schmidt
        real(real64), intent(in) :: settling_velocity, roughness_z0, aks_factor
        real(real64), intent(in) :: maximum_volume_concentration
        real(real64), intent(in) :: additional_shear_stress, dstar
        real(real64), intent(in) :: critical_shear_stress, minimum_depth
        real(real64), intent(out) :: raw_bed_x, raw_bed_y
        real(real64), intent(out) :: conc2d_unfractioned, shear_velocity
        real(real64), intent(out) :: efficiency_factor, shear_ratio
        real(real64), intent(out) :: reference_concentration, reference_height
        integer, intent(out) :: status

        integer :: k, kmaxsd
        real(real64) :: speed, rc, current_shear, f1c, fc, transport_stage
        real(real64) :: beta, maximum_diffusivity, zi, dz, bottom_diffusivity
        real(real64) :: factor, background_diffusivity, average_velocity
        real(real64) :: average_concentration_velocity, z, log_velocity
        real(real64) :: skin_shear_velocity, bed_magnitude
        real(real64) :: sediment_diffusivity(0:vertical_layers)
        real(real64) :: profile(vertical_layers)

        raw_bed_x = 0.0_real64
        raw_bed_y = 0.0_real64
        conc2d_unfractioned = 0.0_real64
        shear_velocity = 0.0_real64
        efficiency_factor = 0.0_real64
        shear_ratio = 0.0_real64
        reference_concentration = 0.0_real64
        reference_height = 0.0_real64
        status = 1

        if (.not.all(ieee_is_finite([depth, velocity_x, velocity_y, &
                observation_height, d50, d90, solid_density, water_density, &
                gravity, kinematic_viscosity, molecular_schmidt, &
                settling_velocity, roughness_z0, aks_factor, &
                maximum_volume_concentration, additional_shear_stress, &
                dstar, critical_shear_stress, minimum_depth]))) return
        if (depth < 0.0_real64 .or. minimum_depth < 0.0_real64) return
        if (observation_height < 0.0_real64 .or. d50 <= 0.0_real64 .or. &
                d90 < d50 .or. solid_density <= water_density .or. &
                water_density <= 0.0_real64 .or. gravity <= 0.0_real64 .or. &
                kinematic_viscosity <= 0.0_real64 .or. &
                molecular_schmidt <= 0.0_real64 .or. &
                settling_velocity < 0.0_real64 .or. roughness_z0 <= 0.0_real64 .or. &
                aks_factor <= 0.0_real64 .or. &
                maximum_volume_concentration <= 0.0_real64 .or. &
                maximum_volume_concentration > 1.0_real64 .or. &
                additional_shear_stress < 0.0_real64 .or. dstar <= 0.0_real64 .or. &
                critical_shear_stress <= 0.0_real64) return

        status = 0
        if (depth <= minimum_depth) return
        if (observation_height <= 0.0_real64) then
            status = 1
            return
        end if

        speed = sqrt(velocity_x*velocity_x + velocity_y*velocity_y)
        rc = 30.0_real64*roughness_z0
        reference_height = min(max(aks_factor*rc, 0.01_real64*depth), &
                               0.2_real64*depth)
        if (reference_height <= 0.0_real64) then
            status = 1
            return
        end if

        ! bedbc1993 current-only branch.  The observation height is RP_ZVLCH;
        ! it is not silently replaced by depth/e here.
        shear_velocity = speed*von_karman/ &
            log(1.0_real64 + observation_height/roughness_z0)
        current_shear = water_density*shear_velocity*shear_velocity
        if (additional_shear_stress > 0.0_real64) then
            current_shear = sqrt(current_shear*current_shear + &
                                 additional_shear_stress*additional_shear_stress)
            shear_velocity = sqrt(current_shear/water_density)
        end if

        if (12.0_real64*depth <= max(rc, 3.0_real64*d90)) then
            status = 1
            return
        end if
        f1c = 0.24_real64*log10(12.0_real64*depth/(3.0_real64*d90))**(-2)
        fc = 0.24_real64*log10(12.0_real64*depth/rc)**(-2)
        efficiency_factor = f1c/fc
        shear_ratio = efficiency_factor*current_shear/critical_shear_stress

        transport_stage = max(0.0_real64, shear_ratio - 1.0_real64)
        if (transport_stage > eps) then
            reference_concentration = min( &
                0.015_real64*d50*transport_stage**1.5_real64/ &
                (reference_height*dstar**0.3_real64), &
                maximum_volume_concentration)
        end if

        ! calseddf1993 algebraic/current-only branch on BRAIDS' fixed 20-layer
        ! 2DH integration column.  FLOW 2DH supplies zero dicww; the molecular
        ! background is vicmol/sigmol and therefore remains explicit here.
        if (shear_velocity > eps) then
            beta = min(max(1.0_real64 + &
                2.0_real64*(settling_velocity/shear_velocity)**2, &
                1.0_real64), 1.5_real64)
        else
            beta = 1.0_real64
        end if
        maximum_diffusivity = 0.25_real64*von_karman*beta* &
                              shear_velocity*depth
        sediment_diffusivity(0) = maximum_diffusivity
        do k = 1, vertical_layers
            zi = (1.0_real64 + sigma(k) - thickness(k)/2.0_real64)*depth
            if (zi >= 0.5_real64*depth) then
                sediment_diffusivity(k) = maximum_diffusivity
            else
                sediment_diffusivity(k) = maximum_diffusivity* &
                    (1.0_real64 - (1.0_real64 - 2.0_real64*zi/depth)**2)
            end if
        end do

        kmaxsd = 1
        do k = vertical_layers - 1, 1, -1
            zi = (1.0_real64 + sigma(k) - thickness(k)/2.0_real64)*depth
            if (zi >= reference_height) then
                kmaxsd = k
                exit
            end if
        end do

        profile = 0.0_real64
        background_diffusivity = kinematic_viscosity/molecular_schmidt
        if (reference_concentration > eps) then
            dz = depth*(1.0_real64 + sigma(kmaxsd)) - reference_height
            bottom_diffusivity = max(sediment_diffusivity(kmaxsd) + &
                background_diffusivity, 0.1_real64*settling_velocity*dz)
            factor = 1.0_real64 + dz*settling_velocity/bottom_diffusivity
            profile(kmaxsd) = reference_concentration*solid_density/factor
            do k = kmaxsd - 1, 1, -1
                dz = depth*(sigma(k) - sigma(k + 1))
                bottom_diffusivity = max(sediment_diffusivity(k) + &
                    background_diffusivity, 0.1_real64*settling_velocity*dz)
                factor = 1.0_real64 + dz*settling_velocity/bottom_diffusivity
                profile(k) = profile(k + 1)/factor
            end do
            do k = kmaxsd + 1, vertical_layers
                profile(k) = profile(k - 1)
            end do
        end if

        average_velocity = 0.0_real64
        average_concentration_velocity = 0.0_real64
        do k = 1, vertical_layers
            z = (1.0_real64 + sigma(k))*depth
            log_velocity = log(1.0_real64 + z/roughness_z0)
            average_velocity = average_velocity + log_velocity*thickness(k)
            average_concentration_velocity = average_concentration_velocity + &
                log_velocity*profile(k)*thickness(k)
        end do
        conc2d_unfractioned = average_concentration_velocity/ &
                             max(average_velocity, eps)

        ! bedtr1993 no-wave branch.  The 1e-10 stage floor is retained even
        ! below the threshold, exactly as in the locked source.
        transport_stage = max(1.0e-10_real64, shear_ratio - 1.0_real64)
        skin_shear_velocity = shear_velocity*sqrt(efficiency_factor)
        bed_magnitude = 0.5_real64*solid_density*d50*skin_shear_velocity* &
                        dstar**(-0.3_real64)*transport_stage
        if (speed > eps) then
            raw_bed_x = bed_magnitude*velocity_x/speed
            raw_bed_y = bed_magnitude*velocity_y/speed
        end if

        if (.not.all(ieee_is_finite([raw_bed_x, raw_bed_y, &
                conc2d_unfractioned, shear_velocity, efficiency_factor, &
                shear_ratio, reference_concentration, reference_height])) .or. &
                conc2d_unfractioned < 0.0_real64 .or. shear_velocity < 0.0_real64 .or. &
                efficiency_factor < 0.0_real64 .or. shear_ratio < 0.0_real64) then
            raw_bed_x = 0.0_real64
            raw_bed_y = 0.0_real64
            conc2d_unfractioned = 0.0_real64
            status = 1
        end if
    end subroutine braids_vr1993_current_producer_v66


    pure subroutine braids_vr1993_compose_route_v66( &
            route, depth, velocity_x, velocity_y, raw_bed_x, raw_bed_y, &
            conc2d_unfractioned, bed_factor, suspended_factor, surface_fraction, &
            bed_mass_x, bed_mass_y, equilibrium_concentration, status)
        ! Reproduce eqtran's special iform<=0 ownership without an intermediate
        ! divide by frac:
        !
        ! TRA_BEDLOAD: frac * (Bed*Qbed + Sus*Cbase*h*U)
        ! TRA_COMBINE: frac * Bed*Qbed, plus frac*Sus*Cbase owned by Q4.
        !
        ! Thus the fraction and the two calibration factors are each applied
        ! once, and the already-mass outputs receive no rhos conversion.
        integer, intent(in) :: route
        real(real64), intent(in) :: depth, velocity_x, velocity_y
        real(real64), intent(in) :: raw_bed_x, raw_bed_y
        real(real64), intent(in) :: conc2d_unfractioned
        real(real64), intent(in) :: bed_factor, suspended_factor, surface_fraction
        real(real64), intent(out) :: bed_mass_x, bed_mass_y
        real(real64), intent(out) :: equilibrium_concentration
        integer, intent(out) :: status

        bed_mass_x = 0.0_real64
        bed_mass_y = 0.0_real64
        equilibrium_concentration = 0.0_real64
        status = 1
        if (.not.all(ieee_is_finite([depth, velocity_x, velocity_y, raw_bed_x, &
                raw_bed_y, conc2d_unfractioned, bed_factor, suspended_factor, &
                surface_fraction]))) return
        if (depth < 0.0_real64 .or. conc2d_unfractioned < 0.0_real64 .or. &
                bed_factor < 0.0_real64 .or. suspended_factor < 0.0_real64 .or. &
                surface_fraction < 0.0_real64 .or. surface_fraction > 1.0_real64) return
        if (route == BRAIDS_VR1993_ROUTE_BEDLOAD_V66) then
            bed_mass_x = surface_fraction*(bed_factor*raw_bed_x + &
                suspended_factor*conc2d_unfractioned*depth*velocity_x)
            bed_mass_y = surface_fraction*(bed_factor*raw_bed_y + &
                suspended_factor*conc2d_unfractioned*depth*velocity_y)
        else if (route == BRAIDS_VR1993_ROUTE_COMBINE_V66) then
            bed_mass_x = surface_fraction*bed_factor*raw_bed_x
            bed_mass_y = surface_fraction*bed_factor*raw_bed_y
            equilibrium_concentration = surface_fraction*suspended_factor* &
                                        conc2d_unfractioned
        else
            return
        end if
        if (.not.all(ieee_is_finite([bed_mass_x, bed_mass_y, &
                                     equilibrium_concentration]))) then
            bed_mass_x = 0.0_real64
            bed_mass_y = 0.0_real64
            equilibrium_concentration = 0.0_real64
            return
        end if
        status = 0
    end subroutine braids_vr1993_compose_route_v66

end module braids_vr1993_route_v66_module
