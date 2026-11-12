! SPDX-License-Identifier: GPL-3.0-only
! Current-only Delft3D4 FLOW capacity operators not covered by BRAIDS' legacy
! bed-load selector.  These routines preserve the locked FLOW algebra and keep
! bed and suspended capacities separate in solid-mass units [kg/(m s)].
!
! They are intentionally pure scalar operators.  Production callers execute
! them from the existing OpenMP cell loop or CUDA counterpart; adding nested
! thread teams here would only add overhead.
module braids_delft_capacity_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    public :: braids_general_capacity
    public :: braids_modified_van_rijn_1984
    public :: braids_wang_fredsoe_capacity
    public :: braids_delft_shields_curve

contains

    pure real(8) function braids_delft_shields_curve(dstar) result(theta_cr)
        ! Locked sed_support_routines::shld piecewise Shields curve.
        real(8), intent(in) :: dstar
        if (.not.ieee_is_finite(dstar).or.dstar<=0.d0) then
            theta_cr=0.d0
        else if (dstar<=4.d0) then
            theta_cr=.240d0/dstar
        else if (dstar<=10.d0) then
            theta_cr=.140d0/dstar**.64d0
        else if (dstar<=20.d0) then
            theta_cr=.040d0/dstar**.10d0
        else if (dstar<=150.d0) then
            theta_cr=.013d0*dstar**.29d0
        else
            theta_cr=.055d0
        endif
    end function braids_delft_shields_curve

    pure subroutine braids_general_capacity(u,v,d50,solid_density,water_density, &
                                             gravity,chezy,hiding_factor, &
                                             bed_calibration,bed_shields_power, &
                                             bed_excess_power,bed_ripple_factor, &
                                             bed_critical_shields,sus_calibration, &
                                             sus_shields_power,sus_excess_power, &
                                             sus_ripple_factor,sus_critical_shields, &
                                             bed_mass,suspended_mass,shields,valid)
        ! FLOW TraFrm=4 / tranb4.  Formula parameters retain their Delft names
        ! and ordering, but are typed arguments so an invalid deck cannot shift
        ! one parameter silently into another slot.
        real(8), intent(in) :: u,v,d50,solid_density,water_density,gravity,chezy
        real(8), intent(in) :: hiding_factor,bed_calibration,bed_shields_power
        real(8), intent(in) :: bed_excess_power,bed_ripple_factor,bed_critical_shields
        real(8), intent(in) :: sus_calibration,sus_shields_power,sus_excess_power
        real(8), intent(in) :: sus_ripple_factor,sus_critical_shields
        real(8), intent(out) :: bed_mass(2),suspended_mass(2),shields
        logical, intent(out) :: valid
        real(8) :: speed,relative_density,scale,bed_excess,sus_excess
        real(8) :: bed_volume,sus_volume

        bed_mass=0.d0;suspended_mass=0.d0;shields=0.d0
        valid=all(ieee_is_finite([u,v,d50,solid_density,water_density,gravity,chezy, &
                                  hiding_factor,bed_calibration,bed_shields_power, &
                                  bed_excess_power,bed_ripple_factor,bed_critical_shields, &
                                  sus_calibration,sus_shields_power,sus_excess_power, &
                                  sus_ripple_factor,sus_critical_shields]))
        if(.not.valid) return
        valid=d50>0.d0.and.solid_density>water_density.and.water_density>0.d0.and. &
              gravity>0.d0.and.chezy>0.d0.and.hiding_factor>=0.d0.and. &
              min(bed_calibration,sus_calibration,bed_ripple_factor,sus_ripple_factor, &
                  bed_critical_shields,sus_critical_shields)>=0.d0
        if(.not.valid) return
        speed=sqrt(u*u+v*v)
        if(.not.ieee_is_finite(speed)) then
            valid=.false.;return
        endif
        if(speed==0.d0) return

        relative_density=solid_density/water_density-1.d0
        shields=(speed/chezy)**2/(relative_density*d50)
        bed_excess=max(bed_ripple_factor*shields-hiding_factor*bed_critical_shields,0.d0)
        sus_excess=max(sus_ripple_factor*shields-hiding_factor*sus_critical_shields,0.d0)
        scale=d50**1.5d0*sqrt(gravity*relative_density)
        bed_volume=bed_calibration*scale*shields**bed_shields_power*bed_excess**bed_excess_power
        sus_volume=sus_calibration*scale*shields**sus_shields_power*sus_excess**sus_excess_power
        bed_mass=solid_density*bed_volume*[u,v]/speed
        suspended_mass=solid_density*sus_volume*[u,v]/speed
        valid=all(ieee_is_finite(bed_mass)).and.all(ieee_is_finite(suspended_mass)).and. &
              ieee_is_finite(shields).and.shields>=0.d0
        if(.not.valid) then
            bed_mass=0.d0;suspended_mass=0.d0;shields=0.d0
        endif
    end subroutine braids_general_capacity

    pure subroutine braids_modified_van_rijn_1984(depth,u,v,d50,d90,solid_density, &
                                                   water_density,gravity,kinematic_viscosity, &
                                                   calibration,reference_height,settling_velocity, &
                                                   mud_power,mud_fraction,bed_mass,suspended_mass, &
                                                   shields,critical_shields,transport_stage,valid)
        ! FLOW TraFrm=7 / tranb7, current-only.  The locked formula deliberately
        ! floors a subcritical transport stage to 1e-6 rather than returning
        ! zero; preserving that historical behavior is part of equivalence.
        real(8), intent(in) :: depth,u,v,d50,d90,solid_density,water_density
        real(8), intent(in) :: gravity,kinematic_viscosity,calibration
        real(8), intent(in) :: reference_height,settling_velocity,mud_power,mud_fraction
        real(8), intent(out) :: bed_mass(2),suspended_mass(2)
        real(8), intent(out) :: shields,critical_shields,transport_stage
        logical, intent(out) :: valid
        ! FLOW initializes the runtime constant from an unsuffixed literal in
        ! iniphy.f90, so preserve its binary32 rounding before fp promotion.
        real(8), parameter :: von_karman=real(.41,8)
        real(8) :: speed,relative_density,dstar,roughness_factor,friction_factor
        real(8) :: shear,total_critical_shear,concentration,shear_velocity
        real(8) :: beta,psi,zc,ah,integral_factor,bed_volume,sus_volume

        bed_mass=0.d0;suspended_mass=0.d0
        shields=0.d0;critical_shields=0.d0;transport_stage=0.d0
        valid=all(ieee_is_finite([depth,u,v,d50,d90,solid_density,water_density,gravity, &
                                  kinematic_viscosity,calibration,reference_height, &
                                  settling_velocity,mud_power,mud_fraction]))
        if(.not.valid) return
        valid=depth>=0.d0.and.d50>0.d0.and.d90>0.d0.and.solid_density>water_density.and. &
              water_density>0.d0.and.gravity>0.d0.and.kinematic_viscosity>0.d0.and. &
              calibration>=0.d0.and.reference_height>0.d0.and.settling_velocity>=0.d0.and. &
              mud_fraction>=0.d0
        if(.not.valid) return
        speed=sqrt(u*u+v*v)
        if(.not.ieee_is_finite(speed)) then
            valid=.false.;return
        endif
        if(depth/reference_height<real(1.33,8).or.speed<real(1.e-3,8)) return

        relative_density=solid_density/water_density-1.d0
        dstar=d50*(relative_density*gravity/kinematic_viscosity/kinematic_viscosity)**real(1./3.,8)
        roughness_factor=(log10(real(12.,8)*depth/reference_height)/ &
                          log10(real(12.,8)*depth/real(3.,8)/d90))**2
        friction_factor=real(.24,8)*log10(real(12.,8)*depth/reference_height)**(-2)
        shear=real(.125,8)*water_density*friction_factor*speed**2
        shields=roughness_factor*shear/((solid_density-water_density)*gravity*d50)
        critical_shields=braids_delft_shields_curve(dstar)
        total_critical_shear=(solid_density-water_density)*gravity*d50*critical_shields* &
                             (1.d0+mud_fraction)**mud_power
        transport_stage=(roughness_factor*shear-total_critical_shear)/total_critical_shear
        transport_stage=max(transport_stage,real(.000001,8))
        concentration=real(.015,8)*calibration*d50/reference_height* &
                      transport_stage**real(1.5,8)/dstar**real(.3,8)
        shear_velocity=sqrt(real(.125,8)*friction_factor)*speed
        beta=min(real(1.,8)+real(2.,8)*(settling_velocity/shear_velocity)**2,real(1.5,8))
        psi=real(2.5,8)*(settling_velocity/shear_velocity)**real(.8,8)* &
            (concentration/real(.65,8))**real(.4,8)
        zc=min(settling_velocity/von_karman/shear_velocity/beta+psi,20.d0)
        ah=reference_height/depth
        if(abs(zc-real(1.2,8))>real(1.e-4,8)) then
            integral_factor=(ah**zc-ah**real(1.2,8))/(real(1.,8)-ah)**zc/ &
                            (real(1.2,8)-zc)
        else
            integral_factor=-(ah/(real(1.,8)-ah))**real(1.2,8)*log(ah)
        endif
        sus_volume=integral_factor*speed*depth*concentration
        if(transport_stage<real(3.,8)) then
            bed_volume=real(.053,8)*relative_density**real(.5,8)*sqrt(gravity)* &
                       d50**real(1.5,8)*dstar**(-real(.3,8))* &
                       transport_stage**real(2.1,8)
        else
            bed_volume=real(.100,8)*relative_density**real(.5,8)*sqrt(gravity)* &
                       d50**real(1.5,8)*dstar**(-real(.3,8))* &
                       transport_stage**real(1.5,8)
        endif
        bed_mass=solid_density*bed_volume*[u,v]/speed
        suspended_mass=solid_density*sus_volume*[u,v]/speed
        valid=all(ieee_is_finite(bed_mass)).and.all(ieee_is_finite(suspended_mass)).and. &
              all(ieee_is_finite([shields,critical_shields,transport_stage]))
        if(.not.valid) then
            bed_mass=0.d0;suspended_mass=0.d0
            shields=0.d0;critical_shields=0.d0;transport_stage=0.d0
        endif
    end subroutine braids_modified_van_rijn_1984

    pure subroutine braids_wang_fredsoe_capacity(depth,u,v,d50,d90,solid_density, &
                                                  water_density,gravity,kinematic_viscosity, &
                                                  chezy,current_shear_velocity,calibration, &
                                                  bed_mass,suspended_mass,shields, &
                                                  critical_shields,total_mass,valid)
        ! FLOW TraFrm=13 / tran9t historical test formula.  In the locked code
        ! the apparent grain-friction fixed-point loop executes only its first
        ! evaluation because the continuation condition is unreachable after
        ! the preceding branch.  We preserve the resulting one-pass cgrain.
        real(8), intent(in) :: depth,u,v,d50,d90,solid_density,water_density
        real(8), intent(in) :: gravity,kinematic_viscosity,chezy,current_shear_velocity
        real(8), intent(in) :: calibration
        real(8), intent(out) :: bed_mass(2),suspended_mass(2)
        real(8), intent(out) :: shields,critical_shields,total_mass
        logical, intent(out) :: valid
        real(8) :: speed,relative_density,cgrain,dstar,excess
        real(8) :: bed_volume,total_volume,suspended_volume

        bed_mass=0.d0;suspended_mass=0.d0
        shields=0.d0;critical_shields=0.d0;total_mass=0.d0
        valid=all(ieee_is_finite([depth,u,v,d50,d90,solid_density,water_density,gravity, &
                                  kinematic_viscosity,chezy,current_shear_velocity,calibration]))
        if(.not.valid) return
        valid=depth>0.d0.and.d50>0.d0.and.d90>0.d0.and.solid_density>water_density.and. &
              water_density>0.d0.and.gravity>0.d0.and.kinematic_viscosity>0.d0.and. &
              chezy>=0.d0.and.current_shear_velocity>=0.d0.and.calibration>=0.d0
        if(.not.valid) return
        speed=sqrt(u*u+v*v)
        if(.not.ieee_is_finite(speed)) then
            valid=.false.;return
        endif
        if(speed==0.d0.or.chezy<=0.d0) return

        relative_density=solid_density/water_density-1.d0
        total_volume=real(.05,8)*speed**2*current_shear_velocity**3/d50/ &
                     (gravity*relative_density)**2
        cgrain=real(6.,8)+real(2.5,8)*log(depth/(real(2.5,8)*d90))
        if(cgrain==0.d0) then
            valid=.false.;return
        endif
        shields=gravity*(speed/cgrain)**2/relative_density/gravity/d50
        dstar=d50*(relative_density*gravity/kinematic_viscosity**2)**real(.33333333,8)
        critical_shields=braids_delft_shields_curve(dstar)
        excess=shields-critical_shields
        bed_volume=0.d0
        if(excess>0.d0) bed_volume=calibration*real(4.8695,8)*d50*current_shear_velocity/ &
                         (real(1.,8)+(real(.2668,8)/excess)**4)**real(.25,8)* &
                         (real(1.,8)-real(.7,8)*sqrt(critical_shields/shields))
        suspended_volume=total_volume-bed_volume
        bed_mass=solid_density*bed_volume*[u,v]/speed
        suspended_mass=solid_density*suspended_volume*[u,v]/speed
        total_mass=solid_density*total_volume
        valid=all(ieee_is_finite(bed_mass)).and.all(ieee_is_finite(suspended_mass)).and. &
              all(ieee_is_finite([shields,critical_shields,total_mass]))
        if(.not.valid) then
            bed_mass=0.d0;suspended_mass=0.d0
            shields=0.d0;critical_shields=0.d0;total_mass=0.d0
        endif
    end subroutine braids_wang_fredsoe_capacity

end module braids_delft_capacity_module
