! Meyer-Peter--Mueller volumetric bedload per unit width (m2/s).
! Zhu et al. (2018), Eq. 12; zero threshold is an explicit benchmark option.
! This formula is distinct from the Delft3D Van Rijn 1993 closure.
module braids_bedload_mpm_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private
    public :: braids_mpm_bedload, braids_mpm_bedload_roughness
    public :: braids_mpm_bedload_roughness_batch
    public :: braids_mixture_percentile_default, braids_mixture_d90_default
contains
    pure subroutine braids_mpm_bedload(u,v,cf,d50,relative_density,g,critical_shields,qx,qy,valid,tau_ratio)
        real(8), intent(in) :: u,v,cf,d50,relative_density,g,critical_shields
        real(8), intent(out) :: qx,qy
        logical, intent(out) :: valid
        real(8), intent(out), optional :: tau_ratio
        real(8) :: speed,theta,magnitude
        qx=0.d0; qy=0.d0
        if(present(tau_ratio)) tau_ratio=0.d0
        valid=all(ieee_is_finite([u,v,cf,d50,relative_density,g,critical_shields]))
        if(.not.valid) return
        valid=min(d50,relative_density,g)>0.d0.and.min(cf,critical_shields)>=0.d0
        if(.not.valid) return
        speed=sqrt(u*u+v*v)
        if(speed==0.d0) return
        theta=cf*speed**2/(g*relative_density*d50)
        if(present(tau_ratio)) then
            if(critical_shields>0.d0) then
                tau_ratio=theta/critical_shields
            else if(theta>0.d0) then
                tau_ratio=huge(1.d0)
            endif
        endif
        theta=max(0.d0,theta-critical_shields)
        magnitude=8.d0*sqrt(g*relative_density*d50**3)*theta**1.5d0
        qx=magnitude*u/speed; qy=magnitude*v/speed
        valid=all(ieee_is_finite([qx,qy]))
    end subroutine

    pure subroutine braids_mpm_bedload_roughness(h,u,v,manning_n,d50,d90,relative_density,g, &
                                                 critical_shields,hiding_factor,qx,qy,valid,tau_ratio)
        ! FLOW tranb2.f90 MPM closure: total-flow Chezy resistance with a
        ! grain-related Chezy ripple factor. This remains separate from the
        ! specified-friction Zhu benchmark closure above.
        real(8), intent(in) :: h,u,v,manning_n,d50,d90,relative_density,g
        real(8), intent(in) :: critical_shields,hiding_factor
        real(8), intent(out) :: qx,qy
        logical, intent(out) :: valid
        real(8), intent(out), optional :: tau_ratio
        real(8) :: speed,chezy,chezy_grain,mu_ripple,theta,excess_theta,magnitude
        qx=0.d0; qy=0.d0
        if(present(tau_ratio)) tau_ratio=0.d0
        valid=all(ieee_is_finite([h,u,v,manning_n,d50,d90,relative_density,g,critical_shields,hiding_factor]))
        if(.not.valid) return
        ! The mixture D90 can be smaller than an individual coarse class D50
        ! when that class has a small surface share; tranb2 accepts that case.
        valid=h>0.d0.and.manning_n>0.d0.and.d50>0.d0.and.d90>0.d0.and. &
              relative_density>0.d0.and.g>0.d0.and.critical_shields>=0.d0.and.hiding_factor>0.d0
        if(.not.valid) return
        speed=sqrt(u*u+v*v)
        if(speed==0.d0) return
        chezy=h**(1.d0/6.d0)/manning_n
        chezy_grain=18.d0*log10(12.d0*max(h/d90,1.d0))
        if(chezy<=0.d0.or.chezy_grain<=0.d0) then
            valid=.false.; return
        endif
        mu_ripple=min((chezy/chezy_grain)**1.5d0,1.d0)
        theta=(speed/chezy)**2/(relative_density*d50)
        if(present(tau_ratio)) then
            if(hiding_factor*critical_shields>0.d0) then
                tau_ratio=mu_ripple*theta/(hiding_factor*critical_shields)
            else if(mu_ripple*theta>0.d0) then
                tau_ratio=huge(1.d0)
            endif
        endif
        excess_theta=max(mu_ripple*theta-hiding_factor*critical_shields,0.d0)
        magnitude=8.d0*sqrt(g*relative_density*d50**3)*excess_theta**1.5d0
        qx=magnitude*u/speed; qy=magnitude*v/speed
        valid=all(ieee_is_finite([qx,qy]))
    end subroutine

    pure subroutine braids_mpm_bedload_roughness_batch(h,u,v,manning_n,d50,d90,relative_density,g, &
                                                       critical_shields,hiding_factor,qx,qy,valid,tau_ratio)
        ! Multi-fraction form of the FLOW MPM closure. Hydraulics, total-flow
        ! resistance, grain resistance and direction are common to every class
        ! in one cell; evaluate them once while retaining the scalar operation
        ! order within each class. This is an execution optimization only.
        real(8), intent(in) :: h,u,v,manning_n,d50(:),d90,relative_density(:),g
        real(8), intent(in) :: critical_shields,hiding_factor(:)
        real(8), intent(out) :: qx(size(d50)),qy(size(d50))
        logical, intent(out) :: valid
        real(8), intent(out), optional :: tau_ratio(size(d50))
        real(8) :: speed,chezy,chezy_grain,mu_ripple,theta,excess_theta,magnitude
        integer :: fraction
        qx=0.d0;qy=0.d0
        if(present(tau_ratio)) tau_ratio=0.d0
        valid=size(relative_density)==size(d50).and.size(hiding_factor)==size(d50).and.size(d50)>0
        if(.not.valid) return
        valid=all(ieee_is_finite([h,u,v,manning_n,d90,g,critical_shields])).and. &
              all(ieee_is_finite(d50)).and.all(ieee_is_finite(relative_density)).and. &
              all(ieee_is_finite(hiding_factor))
        if(.not.valid) return
        valid=h>0.d0.and.manning_n>0.d0.and.d90>0.d0.and.g>0.d0.and.critical_shields>=0.d0.and. &
              all(d50>0.d0).and.all(relative_density>0.d0).and.all(hiding_factor>0.d0)
        if(.not.valid) return
        speed=sqrt(u*u+v*v)
        if(speed==0.d0) return
        chezy=h**(1.d0/6.d0)/manning_n
        chezy_grain=18.d0*log10(12.d0*max(h/d90,1.d0))
        if(chezy<=0.d0.or.chezy_grain<=0.d0) then
            valid=.false.;return
        endif
        mu_ripple=min((chezy/chezy_grain)**1.5d0,1.d0)
        do fraction=1,size(d50)
            theta=(speed/chezy)**2/(relative_density(fraction)*d50(fraction))
            if(present(tau_ratio)) then
                if(hiding_factor(fraction)*critical_shields>0.d0) then
                    tau_ratio(fraction)=mu_ripple*theta/(hiding_factor(fraction)*critical_shields)
                else if(mu_ripple*theta>0.d0) then
                    tau_ratio(fraction)=huge(1.d0)
                endif
            endif
            excess_theta=max(mu_ripple*theta-hiding_factor(fraction)*critical_shields,0.d0)
            magnitude=8.d0*sqrt(g*relative_density(fraction)*d50(fraction)**3)*excess_theta**1.5d0
            qx(fraction)=magnitude*u/speed;qy(fraction)=magnitude*v/speed
        enddo
        valid=all(ieee_is_finite(qx)).and.all(ieee_is_finite(qy))
    end subroutine

    pure subroutine braids_mixture_percentile_default(volume,d50,percentile,diameter,valid)
        ! D90 of FLOW's backward-compatible piecewise-loguniform grain
        ! distributions (D10=.75*D50, D90=1.5*D50), combined by active-layer
        ! solid-volume fraction as in compdiam.f90.  The general percentile is
        ! also needed by the Fredsoe bedform predictor for the mixture D50.
        real(8), intent(in) :: volume(:),d50(:),percentile
        real(8), intent(out) :: diameter
        logical, intent(out) :: valid
        real(8) :: total,lo,hi,mid,cumulative,lower,upper,share,x
        real(8) :: log_d50(size(d50)),lower_bound(size(d50)),upper_bound(size(d50))
        integer :: fraction,iteration
        diameter=0.d0;valid=size(volume)==size(d50).and.size(volume)>0
        if(.not.valid) return
        valid=all(ieee_is_finite(volume)).and.all(ieee_is_finite(d50)).and. &
              ieee_is_finite(percentile).and.all(volume>=0.d0).and.all(d50>0.d0).and. &
              percentile>0.d0.and.percentile<1.d0
        if(.not.valid) return
        total=sum(volume)
        if(total<=0.d0.or..not.ieee_is_finite(total)) then
            valid=.false.;return
        endif
        ! Identical physical grains have the same distribution irrespective of
        ! the number or volume shares of their bookkeeping labels. Invert that
        ! piecewise-linear log-CDF directly, rather than bisect it 60 times per
        ! wet cell and RK stage. Keep validation/empty-mixture handling above;
        ! genuinely mixed diameters retain the original path below.
        if(all(d50==d50(1))) then
            x=log(d50(1))
            if(percentile<.5d0) then
                diameter=exp(x+1.25d0*log(.75d0)*(1.d0-2.d0*percentile))
            else
                diameter=exp(x+1.25d0*log(1.5d0)*(2.d0*percentile-1.d0))
            endif
            valid=ieee_is_finite(diameter).and.diameter>0.d0
            return
        endif
        ! Grain diameters do not change during the bracket solve.  Hoist their
        ! logarithms and distribution bounds out of the 60-iteration loop;
        ! this preserves the exact bisection/order while removing 420 repeated
        ! transcendental evaluations for a seven-fraction mixture.
        log_d50=log(d50)
        lower_bound=log_d50+1.25d0*log(.75d0)
        upper_bound=log_d50+1.25d0*log(1.5d0)
        lo=minval(lower_bound)
        hi=maxval(upper_bound)
        ! A double-precision bracket spanning the complete supported grain
        ! range has exhausted all representable midpoint refinement well before
        ! 60 halvings. The former 80 iterations repeated identical endpoints;
        ! retain a conservative margin while avoiding dead work in every wet
        ! cell and SSPRK stage.
        do iteration=1,60
            mid=.5d0*(lo+hi);cumulative=0.d0
            do fraction=1,size(volume)
                share=volume(fraction)/total
                lower=lower_bound(fraction)
                upper=upper_bound(fraction)
                x=log_d50(fraction)
                if(mid<=lower) then
                    continue
                else if(mid<x) then
                    cumulative=cumulative+share*.5d0*(mid-lower)/(x-lower)
                else if(mid<upper) then
                    cumulative=cumulative+share*(.5d0+.5d0*(mid-x)/(upper-x))
                else
                    cumulative=cumulative+share
                endif
            enddo
            if(cumulative<percentile) then
                lo=mid
            else
                hi=mid
            endif
        enddo
        diameter=exp(.5d0*(lo+hi));valid=ieee_is_finite(diameter).and.diameter>0.d0
    end subroutine

    pure subroutine braids_mixture_d90_default(volume,d50,d90,valid)
        real(8), intent(in) :: volume(:),d50(:)
        real(8), intent(out) :: d90
        logical, intent(out) :: valid
        call braids_mixture_percentile_default(volume,d50,.9d0,d90,valid)
    end subroutine
end module
