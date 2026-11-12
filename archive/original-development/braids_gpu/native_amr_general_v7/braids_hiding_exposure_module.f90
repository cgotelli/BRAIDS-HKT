! Hiding/exposure factors for a noncohesive sediment mixture.
! Formulas and arithmetic mean diameter follow the supplied Delft3D
! comphidexp.f90 (ihidexp=2:5) and compdiam.f90 sources.
module braids_hiding_exposure_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private
    public :: braids_hiding_exposure_factors,braids_egiazaroff_factors
contains
    pure subroutine braids_hiding_exposure_factors(mode,active_volume,d50,factors,mean_diameter, &
                                                    parker_alpha,wu_exponent,valid)
        integer, intent(in) :: mode
        real(8), intent(in) :: active_volume(:),d50(:),parker_alpha,wu_exponent
        real(8), intent(out) :: factors(size(active_volume)),mean_diameter
        logical, intent(out) :: valid
        real(8), parameter :: log10_19=1.27875360095283d0
        real(8), parameter :: minimum_mean_diameter=1.d-4
        real(8) :: total_volume,diameter_ratio,denominator,local_mean,phi,pei
        real(8) :: fraction(size(active_volume))
        integer :: grain,other

        factors=1.d0;mean_diameter=0.d0;valid=.false.
        if(size(active_volume)/=size(d50).or.size(d50)<1) return
        if(.not.all(ieee_is_finite(active_volume)).or..not.all(ieee_is_finite(d50))) return
        if(.not.ieee_is_finite(parker_alpha).or..not.ieee_is_finite(wu_exponent)) return
        if(any(active_volume<0.d0).or.any(d50<=0.d0)) return
        if(mode/=0.and.(mode<2.or.mode>5)) return
        total_volume=sum(active_volume)
        if(total_volume<=0.d0) return
        fraction=active_volume/total_volume
        mean_diameter=dot_product(fraction,d50)
        local_mean=max(mean_diameter,minimum_mean_diameter)
        select case(mode)
        case(0)
            continue
        case(2,3)
            do grain=1,size(d50)
                diameter_ratio=d50(grain)/local_mean
                ! Preserve the source routine's default-real literals before
                ! promotion to double precision; this is observable at roundoff.
                if(mode==3.and.diameter_ratio<real(0.38889,8)) then
                    factors(grain)=real(0.8429,8)/diameter_ratio
                else
                    denominator=log10_19+log10(diameter_ratio)
                    if(denominator==0.d0) return
                    factors(grain)=(log10_19/denominator)**2
                endif
            enddo
        case(4)
            do grain=1,size(d50)
                diameter_ratio=d50(grain)/local_mean
                factors(grain)=diameter_ratio**(-parker_alpha)
            enddo
        case(5)
            do grain=1,size(d50)
                phi=0.d0
                do other=1,size(d50)
                    phi=phi+fraction(other)*d50(other)/(d50(grain)+d50(other))
                enddo
                pei=1.d0-phi
                if(phi<=0.d0.or.pei<=0.d0) return
                factors(grain)=(pei/phi)**wu_exponent
            enddo
        end select
        valid=all(ieee_is_finite(factors)).and.all(factors>0.d0)
    end subroutine braids_hiding_exposure_factors

    pure subroutine braids_egiazaroff_factors(active_volume,d50,factors,mean_diameter,valid)
        real(8), intent(in) :: active_volume(:),d50(:)
        real(8), intent(out) :: factors(size(active_volume)),mean_diameter
        logical, intent(out) :: valid
        call braids_hiding_exposure_factors(2,active_volume,d50,factors,mean_diameter,0.d0,0.d0,valid)
    end subroutine braids_egiazaroff_factors
end module braids_hiding_exposure_module
