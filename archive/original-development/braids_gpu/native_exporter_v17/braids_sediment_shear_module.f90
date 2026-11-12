! Sediment-only bed-shear closures from locked Delft3D4 FLOW.
!
! This module does not alter hydraulic momentum friction.  BsSkin in FLOW
! replaces only the sediment RP_TAUB value; formulas that consume Chezy or
! effective velocity continue to use the total hydraulic roughness chain.
module braids_sediment_shear_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private
    public :: braids_soulsby_clarke_skin_shear

contains

    pure subroutine braids_soulsby_clarke_skin_shear(depth,speed,viscosity,density, &
                                                     ks_sand,ks_silt,mud_factor, &
                                                     critical_mud_1,critical_mud_2, &
                                                     fluff_fraction,shear,z0_skin, &
                                                     cd_rough,cd_smooth,valid)
        ! Current-only branch of FLOW compbsskin.  The 0.40 coefficient is
        ! formula-specific and intentionally differs from FLOW's hydraulic
        ! von Karman constant 0.41.
        real(8), intent(in) :: depth,speed,viscosity,density,ks_sand,ks_silt
        real(8), intent(in) :: mud_factor,critical_mud_1,critical_mud_2,fluff_fraction
        real(8), intent(out) :: shear,z0_skin,cd_rough,cd_smooth
        logical, intent(out) :: valid
        real(8), parameter :: local_epsilon=1.d-12
        real(8) :: alpha,position,ks_bed,ks_effective,reynolds,denominator

        shear=0.d0;z0_skin=0.d0;cd_rough=0.d0;cd_smooth=0.d0
        valid=all(ieee_is_finite([depth,speed,viscosity,density,ks_sand,ks_silt, &
                                  mud_factor,critical_mud_1,critical_mud_2,fluff_fraction]))
        if(.not.valid) return
        valid=depth>0.d0.and.speed>=0.d0.and.viscosity>0.d0.and.density>0.d0.and. &
              ks_sand>=0.d0.and.ks_silt>=0.d0.and.critical_mud_2>critical_mud_1.and. &
              fluff_fraction>=0.d0.and.fluff_fraction<=1.d0
        if(.not.valid) return

        if(mud_factor>critical_mud_2) then
            alpha=1.d0
        else if(mud_factor>critical_mud_1) then
            position=(mud_factor-critical_mud_1)/(critical_mud_2-critical_mud_1)
            alpha=3.d0*position*position-2.d0*position*position*position
        else
            alpha=0.d0
        endif
        ks_bed=alpha*ks_silt+(1.d0-alpha)*ks_sand
        ks_effective=fluff_fraction*ks_silt+(1.d0-fluff_fraction)*ks_bed
        z0_skin=max(ks_effective/30.d0,local_epsilon)

        ! compbsskin floors the speed only for Reynolds-number evaluation.
        reynolds=max(speed,local_epsilon)*depth/viscosity
        cd_smooth=1.615d-4*exp(6.d0*reynolds**(-.08d0))
        denominator=log(depth/z0_skin)-1.d0
        if(denominator==0.d0) then
            valid=.false.;return
        endif
        cd_rough=(.40d0/denominator)**2
        if(speed>=1.d-6) shear=density*max(cd_rough,cd_smooth)*speed*speed
        valid=all(ieee_is_finite([shear,z0_skin,cd_rough,cd_smooth])).and. &
              min(shear,z0_skin,cd_rough,cd_smooth)>=0.d0
    end subroutine braids_soulsby_clarke_skin_shear

end module braids_sediment_shear_module
