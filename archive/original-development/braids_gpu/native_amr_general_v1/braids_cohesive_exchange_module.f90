! Depth-averaged cohesive exchange and sand-mud interaction.
!
! The Partheniades erosion, Krone deposition reduction and Van Ledden/Van
! Kessel mixture switch follow Delft3D's erosilt and sand_mud kernels.  Rates
! are returned separately so the cut-cell solver can integrate the linear
! deposition term exactly and cap erosion by its conservative bed inventory.
module braids_cohesive_exchange_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private
    public :: braids_partheniades_krone,braids_sand_mud_erosion
contains
    pure subroutine braids_partheniades_krone(tau,bulk_density,settling_velocity,h,critical_erosion, &
                                             critical_deposition,erosion_parameter,erosion_exponent, &
                                             erosion_velocity,deposition_rate,valid)
        real(8), intent(in) :: tau,bulk_density,settling_velocity,h,critical_erosion,critical_deposition
        real(8), intent(in) :: erosion_parameter,erosion_exponent
        real(8), intent(out) :: erosion_velocity,deposition_rate
        logical, intent(out) :: valid
        erosion_velocity=0.d0;deposition_rate=0.d0
        valid=all(ieee_is_finite([tau,bulk_density,settling_velocity,h,critical_erosion,critical_deposition, &
                                  erosion_parameter,erosion_exponent]))
        if(.not.valid) return
        valid=tau>=0.d0.and.bulk_density>0.d0.and.settling_velocity>=0.d0.and.h>0.d0.and. &
              critical_erosion>0.d0.and.critical_deposition>=0.d0.and.erosion_parameter>=0.d0.and.erosion_exponent>0.d0
        if(.not.valid) return
        erosion_velocity=erosion_parameter/bulk_density*max(0.d0,tau/critical_erosion-1.d0)**erosion_exponent
        if(critical_deposition>0.d0) deposition_rate=settling_velocity/h*max(0.d0,1.d0-tau/critical_deposition)
        valid=all(ieee_is_finite([erosion_velocity,deposition_rate]))
    end subroutine

    pure subroutine braids_sand_mud_erosion(nfrac,sediment_type,erosion_velocity,fraction_share,mud_fraction,critical_mud_fraction)
        integer, intent(in) :: nfrac,sediment_type(nfrac)
        real(8), intent(inout) :: erosion_velocity(nfrac)
        real(8), intent(in) :: fraction_share(nfrac),mud_fraction,critical_mud_fraction
        real(8) :: sand_average,mud_average,weight
        integer :: g
        if(mud_fraction<=0.d0.or.mud_fraction>=1.d0) return
        sand_average=0.d0
        do g=1,nfrac
            if(sediment_type(g)==0) sand_average=sand_average+fraction_share(g)*erosion_velocity(g)
        enddo
        sand_average=sand_average/(1.d0-mud_fraction)
        if(mud_fraction<=critical_mud_fraction) then
            if(sand_average<=0.d0) return
            do g=1,nfrac
                if(sediment_type(g)==1) erosion_velocity(g)=sand_average
            enddo
        else
            mud_average=0.d0;weight=(1.d0-mud_fraction)/(1.d0-critical_mud_fraction)
            do g=1,nfrac
                if(sediment_type(g)==1.and.erosion_velocity(g)>0.d0) then
                    if(sand_average>0.d0) erosion_velocity(g)=erosion_velocity(g)*(sand_average/erosion_velocity(g))**weight
                    mud_average=mud_average+fraction_share(g)*erosion_velocity(g)
                endif
            enddo
            mud_average=mud_average/mud_fraction
            do g=1,nfrac
                if(sediment_type(g)==0) erosion_velocity(g)=mud_average
            enddo
        endif
    end subroutine
end module
