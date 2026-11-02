! SPDX-License-Identifier: GPL-3.0-only
!
! Bounded, additive point adapter for six Delft3D4 FLOW transport formulae.
!
! This module is deliberately not connected to the live solver.  It gives the
! CPU and CUDA qualification paths one typed, pre-fraction point contract for
! TraFrm 7, 13, 14, 16, 17 and 18.  The caller owns FLOW's upstream hydraulic
! and mixture preprocessing (RP_D90MX, RP_TAUB, RP_DG, RP_DGSD and RP_SNDFR)
! and applies the exposed active-layer fraction exactly once downstream.
module braids_delft_scalar_capacity_v63_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use braids_delft_capacity_module, only: braids_modified_van_rijn_1984, &
                                              braids_wang_fredsoe_capacity
    use braids_bedload_current_module, only: braids_ashida_michiue_current
    implicit none
    private

    integer, parameter, public :: BRAIDS_SCALAR_CAPACITY_V63_SCHEMA = 1
    public :: braids_delft_scalar_capacity_v63

contains

    pure subroutine braids_delft_scalar_capacity_v63(selector,depth,u,v,d50,d90, &
            solid_density,water_density,gravity,formula_viscosity,chezy, &
            current_shear_velocity,bed_shear,hiding_factor,mud_fraction,dg,dgsd, &
            sand_fraction,acal,reference_height,settling_velocity,mud_power, &
            critical_shields,power_m,power_p,power_q,theta_c0,alpha0, &
            bed_mass,suspended_mass,absolute_shields,reference_shields, &
            stress_ratio,valid)
        integer, intent(in) :: selector
        real(8), intent(in) :: depth,u,v,d50,d90,solid_density,water_density
        real(8), intent(in) :: gravity,formula_viscosity,chezy
        real(8), intent(in) :: current_shear_velocity,bed_shear
        real(8), intent(in) :: hiding_factor,mud_fraction,dg,dgsd,sand_fraction
        real(8), intent(in) :: acal,reference_height,settling_velocity,mud_power
        real(8), intent(in) :: critical_shields,power_m,power_p,power_q
        real(8), intent(in) :: theta_c0,alpha0
        real(8), intent(out) :: bed_mass(2),suspended_mass(2)
        real(8), intent(out) :: absolute_shields,reference_shields,stress_ratio
        logical, intent(out) :: valid
        real(8) :: speed,relative,bed_volume,ratio_from_helper
        real(8) :: wistar,phi,taurm,tauri,b,total_ustar,denominator,sigma_phi

        bed_mass=0.d0
        suspended_mass=0.d0
        absolute_shields=0.d0
        reference_shields=0.d0
        stress_ratio=0.d0
        valid=.false.

        ! Validate only fields consumed by every selector here.  The union ABI
        ! deliberately carries unused placeholders, and locked eqtran does not
        ! read them.  Selector-specific finiteness checks below therefore own
        ! the rest of the contract.
        if (.not.all(ieee_is_finite([u,v,d50,solid_density,water_density, &
                                      gravity]))) return
        if (d50<=0.d0 .or. solid_density<=water_density .or. water_density<=0.d0 &
                .or. gravity<=0.d0) return
        speed=sqrt(u*u+v*v)
        if (.not.ieee_is_finite(speed)) return
        relative=solid_density/water_density-1.d0

        select case(selector)
        case(7)
            if (.not.all(ieee_is_finite([depth,d90,formula_viscosity,acal, &
                    reference_height,settling_velocity,mud_power,mud_fraction]))) return
            if (depth<0.d0 .or. d90<=0.d0 .or. formula_viscosity<=0.d0 .or. &
                    acal<0.d0 .or. reference_height<=0.d0 .or. &
                    settling_velocity<0.d0 .or. mud_fraction<0.d0) return
            call braids_modified_van_rijn_1984(depth,u,v,d50,d90,solid_density, &
                water_density,gravity,formula_viscosity,acal,reference_height, &
                settling_velocity,mud_power,mud_fraction,bed_mass,suspended_mass, &
                absolute_shields,reference_shields,stress_ratio,valid)
            if (valid .and. reference_shields>0.d0) then
                stress_ratio=absolute_shields/reference_shields
            else if (valid .and. absolute_shields>0.d0) then
                stress_ratio=huge(1.d0)
            else if (valid) then
                stress_ratio=0.d0
            endif
        case(13)
            if (.not.all(ieee_is_finite([depth,d90,formula_viscosity,chezy, &
                    current_shear_velocity,acal]))) return
            if (depth<=0.d0 .or. d90<=0.d0 .or. formula_viscosity<=0.d0 .or. &
                    chezy<0.d0 .or. current_shear_velocity<0.d0 .or. acal<0.d0) return
            call braids_wang_fredsoe_capacity(depth,u,v,d50,d90,solid_density, &
                water_density,gravity,formula_viscosity,chezy, &
                current_shear_velocity,acal,bed_mass,suspended_mass, &
                absolute_shields,reference_shields,stress_ratio,valid)
            if (valid .and. reference_shields>0.d0) then
                stress_ratio=absolute_shields/reference_shields
            else if (valid .and. absolute_shields>0.d0) then
                stress_ratio=huge(1.d0)
            else if (valid) then
                stress_ratio=0.d0
            endif
        case(14)
            if (.not.all(ieee_is_finite([chezy,hiding_factor,acal, &
                    critical_shields,power_m,power_p,power_q]))) return
            if (chezy<0.d0 .or. hiding_factor<0.d0 .or. acal<0.d0 .or. &
                    critical_shields<0.d0 .or. min(power_m,power_p,power_q)<0.d0) return
            call braids_ashida_michiue_current(speed,d50,chezy,gravity,relative, &
                hiding_factor,acal,critical_shields,power_m,power_p,power_q, &
                bed_volume,ratio_from_helper,valid)
            if (.not.valid) return
            if (speed>0.d0) bed_mass=solid_density*bed_volume*[u,v]/speed
            absolute_shields=0.d0
            if (chezy>0.d0) absolute_shields=(speed/chezy)**2/(relative*d50)
            reference_shields=hiding_factor*critical_shields
            stress_ratio=ratio_from_helper
            valid=all(ieee_is_finite([bed_mass,suspended_mass,absolute_shields, &
                                      reference_shields,stress_ratio]))
        case(16,17,18)
            if (.not.all(ieee_is_finite([chezy,bed_shear,dg]))) return
            if (chezy<0.d0 .or. bed_shear<0.d0 .or. dg<=0.d0) return
            if (selector==16) then
                ! trabwc consumes sand fraction and optional ACal, but not
                ! DgSD/ThetaC0/Alpha0.
                if (.not.all(ieee_is_finite([sand_fraction,acal]))) return
                if (sand_fraction<0.d0.or.sand_fraction>1.d0.or.acal<0.d0) return
            else
                ! trabwc2/trabg consume true DgSD and their two coefficients,
                ! but not sand fraction or ACal.  compdiam's spatial-D50 branch
                ! returns DgSD=0; evaluate the exact infinite-sigma limit below.
                if (.not.all(ieee_is_finite([dgsd,theta_c0,alpha0]))) return
                if (dgsd<0.d0.or.theta_c0<0.d0.or.alpha0<0.d0) return
            endif
            valid=.true.
            if (chezy<1.d-6 .or. dg<1.d-6 .or. speed==0.d0) return
            select case(selector)
            case(16)
                b=.67d0/(1.d0+exp(1.5d0-d50/dg))
                taurm=(.021d0+.015d0*exp(-20.d0*sand_fraction))* &
                    (solid_density-water_density)*gravity*dg
            case(17)
                b=(1.d0-alpha0)/(1.d0+exp(1.5d0-d50/dg))
                if(dgsd==0.d0) then
                    taurm=theta_c0*(solid_density-water_density)*gravity*dg
                else
                    sigma_phi=(log(dgsd)/log(2.d0))**2
                    taurm=(theta_c0+.015d0/(1.d0+exp(10.1d0*sigma_phi-14.14d0)))* &
                        (solid_density-water_density)*gravity*dg
                endif
            case(18)
                b=(1.d0-alpha0)/(1.d0+exp(1.9d0-d50/(3.d0*dg)))
                if(dgsd==0.d0) then
                    taurm=theta_c0*(solid_density-water_density)*gravity*dg
                else
                    sigma_phi=(log(dgsd)/log(2.d0))**2
                    taurm=(theta_c0+.022d0/(1.d0+exp(7.1d0*sigma_phi-11.786d0)))* &
                        (solid_density-water_density)*gravity*dg
                endif
            end select
            tauri=taurm*(d50/dg)**b
            if (.not.ieee_is_finite(tauri) .or. tauri<=0.d0) then
                valid=.false.
                return
            endif
            phi=bed_shear/tauri
            if (phi<1.35d0) then
                wistar=.002d0*phi**7.5d0
            else
                wistar=14.d0*(1.d0-.894d0/sqrt(phi))**4.5d0
            endif
            total_ustar=sqrt(gravity)*speed/chezy
            bed_volume=wistar*total_ustar**3/(relative*gravity)
            if (selector==16) bed_volume=acal*bed_volume
            bed_mass=solid_density*bed_volume*[u,v]/speed
            denominator=(solid_density-water_density)*gravity*d50
            absolute_shields=bed_shear/denominator
            reference_shields=tauri/denominator
            stress_ratio=phi
            valid=all(ieee_is_finite([bed_mass,suspended_mass,absolute_shields, &
                                      reference_shields,stress_ratio])) .and. &
                  min(absolute_shields,reference_shields,stress_ratio)>=0.d0
            if (.not.valid) then
                bed_mass=0.d0
                suspended_mass=0.d0
                absolute_shields=0.d0
                reference_shields=0.d0
                stress_ratio=0.d0
            endif
        case default
            valid=.false.
        end select
    end subroutine braids_delft_scalar_capacity_v63

end module braids_delft_scalar_capacity_v63_module
