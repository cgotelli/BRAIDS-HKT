! SPDX-License-Identifier: GPL-3.0-only
! Current-only Delft3D4 FLOW TraFrm=11 Soulsby/Van Rijn capacity.
!
! This is a clean BRAIDS transcription of the current branch in the locked
! morphology_kernel/src/trab11.f90 reference.  It intentionally omits waves;
! when urms=0 the effective velocity in trab11 is exactly the current speed.
! Bed and suspended capacities remain separate directional mass vectors.
! Reference: Delft3D commit c05ed672f0a7fe81c5b9867ef354cfdd53c25379,
! trab11 SHA-256 f1bfce663f2851bd8fc1ec96783394562d016e44a8d3d5cbc5138e326782b1ab.
module braids_soulsby_van_rijn_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    ! trab11's historical unsuffixed literals are evaluated as default real
    ! before promotion to FLOW's double-precision fp.  Preserve that behavior
    ! so this reference operator can be compared directly with the locked
    ! executable rather than introducing a systematic ~1e-7 coefficient drift.
    real(8), parameter :: von_karman = real(0.41,8)
    real(8), parameter :: minimum_transport_depth = real(0.01,8)
    real(8), parameter :: maximum_transport_depth = real(200.,8)
    real(8), parameter :: minimum_transport_speed = real(0.000001,8)
    real(8), parameter :: grain_branch_diameter = real(0.0005,8)
    real(8), parameter :: exponent_one_third = real(1./3.,8)
    real(8), parameter :: exponent_one_tenth = real(0.1,8)
    real(8), parameter :: exponent_three_fifths = real(0.6,8)
    real(8), parameter :: exponent_six_fifths = real(1.2,8)
    real(8), parameter :: exponent_twelve_fifths = real(2.4,8)
    real(8), parameter :: exponent_one_half = real(0.5,8)

    public :: braids_soulsby_van_rijn_current

contains

    pure subroutine braids_soulsby_van_rijn_current(depth,u,v,d50,solid_density, &
                                                     water_density,gravity,kinematic_viscosity, &
                                                     calibration,d90_d50,roughness_z0, &
                                                     bed_mass,suspended_mass,shields, &
                                                     critical_shields,critical_speed,valid)
        ! Inputs correspond to the locked TraFrm=11 current-only parameters:
        ! calibration=ACal, d90_d50=RatioD90D50 and roughness_z0=RouZ0.
        ! u and v are FLOW's effective transport velocities (RP_EFUMN/EFVMN),
        ! not necessarily unadjusted depth-averaged hydraulic velocities.  The
        ! locked FLOW wrapper supplies its promoted default 1.0E-6 for viscosity.
        !
        ! bed_mass and suspended_mass are solid mass transport per unit width
        ! [kg/(m s)].  Multiplication by solid_density is the positive-IFORM
        ! volume-to-mass conversion performed by FLOW eqtran after trab11.
        !
        ! TraFrm=11 thresholds transport with critical_speed, not Shields.
        ! shields and critical_shields are the equivalent absolute diagnostics
        ! formed with trab11's current drag coefficient and FLOW's default
        ! von Karman value; they are not a replacement threshold and no hiding
        ! correction is applied here.
        real(8), intent(in) :: depth,u,v,d50,solid_density,water_density
        real(8), intent(in) :: gravity,kinematic_viscosity,calibration
        real(8), intent(in) :: d90_d50,roughness_z0
        real(8), intent(out) :: bed_mass(2),suspended_mass(2)
        real(8), intent(out) :: shields,critical_shields,critical_speed
        logical, intent(out) :: valid
        real(8) :: speed,effective_speed,d90,relative_density,dstar,drag,drag_denominator
        real(8) :: bed_coefficient,suspended_coefficient,excess_power
        real(8) :: bed_volume(2),suspended_volume(2)

        bed_mass=0.d0
        suspended_mass=0.d0
        shields=0.d0
        critical_shields=0.d0
        critical_speed=0.d0

        valid=all(ieee_is_finite([depth,u,v,d50,solid_density,water_density, &
                                  gravity,kinematic_viscosity,calibration, &
                                  d90_d50,roughness_z0]))
        if(.not.valid) return
        valid=depth>=0.d0.and.d50>0.d0.and.solid_density>water_density.and. &
              water_density>0.d0.and.gravity>0.d0.and.kinematic_viscosity>0.d0.and. &
              calibration>=0.d0.and.d90_d50>0.d0.and.roughness_z0>0.d0
        if(.not.valid) return

        ! Retain trab11's strict gates: both endpoint depths and exactly
        ! 1e-6 m/s remain eligible for the constitutive calculation.
        speed=sqrt(u*u+v*v)
        if(.not.ieee_is_finite(speed)) then
            valid=.false.
            return
        endif
        if(speed<minimum_transport_speed.or.depth>maximum_transport_depth.or. &
           depth<minimum_transport_depth) return

        ! Preserve EROSED's localpar(4) operation order as well as TRAB11.
        relative_density=(solid_density-water_density)/water_density
        d90=d90_d50*d50
        dstar=(gravity*relative_density/kinematic_viscosity**2)**exponent_one_third*d50
        if(.not.all(ieee_is_finite([relative_density,d90,dstar])).or. &
           min(relative_density,d90,dstar)<=0.d0) then
            valid=.false.
            return
        endif

        ! Soulsby (1997), equations 133d/e as selected by trab11.  The locked
        ! implementation deliberately uses the second expression above 2 mm
        ! as well as within the nominal 0.5--2 mm interval.
        if(d50<=grain_branch_diameter) then
            critical_speed=real(0.19,8)*d50**exponent_one_tenth* &
                           log10(real(4.,8)*depth/d90)
        else if(d50<0.002d0) then
            critical_speed=real(8.5,8)*d50**exponent_three_fifths* &
                           log10(real(4.,8)*depth/d90)
        else
            critical_speed=real(8.5,8)*d50**exponent_three_fifths* &
                           log10(real(4.,8)*depth/d90)
        endif

        ! Soulsby (1997), p.183.  This drag does not control current-only
        ! capacity, but maps the velocity threshold to absolute Shields values.
        drag_denominator=log(depth/roughness_z0)-1.d0
        if(.not.ieee_is_finite(drag_denominator).or. &
           abs(drag_denominator)<=tiny(drag_denominator)) then
            valid=.false.
            critical_speed=0.d0
            return
        endif
        drag=(von_karman/drag_denominator)**2
        shields=drag*speed**2/(relative_density*gravity*d50)
        critical_shields=drag*critical_speed**2/(relative_density*gravity*d50)
        if(.not.all(ieee_is_finite([critical_speed,drag,shields,critical_shields])).or. &
           critical_speed<0.d0.or.drag<0.d0) then
            valid=.false.
            shields=0.d0
            critical_shields=0.d0
            critical_speed=0.d0
            return
        endif
        ! trab11 recomputes its combined wave-current magnitude with a
        ! default-real 0.5 exponent.  With urms=0 this is nominally speed,
        ! while retaining the source evaluation is important to a direct
        ! compiled oracle comparison.
        effective_speed=(speed*speed)**exponent_one_half
        if(effective_speed<=critical_speed) return

        ! Preserve the operation sequence of trab11 instead of cancelling d50.
        bed_coefficient=real(0.005,8)*depth* &
                        (d50/depth/(relative_density*gravity*d50))**exponent_six_fifths
        suspended_coefficient=real(0.012,8)*d50*dstar**(-exponent_three_fifths)/ &
                              (relative_density*gravity*d50)**exponent_six_fifths
        excess_power=(effective_speed-critical_speed)**exponent_twelve_fifths
        bed_volume(1)=calibration*bed_coefficient*u*excess_power
        bed_volume(2)=calibration*bed_coefficient*v*excess_power
        suspended_volume(1)=calibration*suspended_coefficient*u*excess_power
        suspended_volume(2)=calibration*suspended_coefficient*v*excess_power
        ! Keep eqtran's later volume-to-mass multiplication in the same order.
        bed_mass=bed_volume*solid_density
        suspended_mass=suspended_volume*solid_density

        valid=all(ieee_is_finite(bed_mass)).and.all(ieee_is_finite(suspended_mass)).and. &
              all(ieee_is_finite([shields,critical_shields,critical_speed]))
        if(.not.valid) then
            bed_mass=0.d0
            suspended_mass=0.d0
            shields=0.d0
            critical_shields=0.d0
            critical_speed=0.d0
        endif
    end subroutine braids_soulsby_van_rijn_current

end module braids_soulsby_van_rijn_module
