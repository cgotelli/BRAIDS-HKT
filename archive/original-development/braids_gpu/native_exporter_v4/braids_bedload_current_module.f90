! Current-only depth-averaged bed-load formulae used by BRAIDS.
!
! The equations and standard coefficients follow the corresponding Delft3D
! morphology kernels (tranb1, trab12, trab14, trabwc, trabwc2 and trabg).  This module
! deliberately accepts no wave variables: it exposes only the branches whose
! complete physical inputs are available in the present 2DH solver.
module braids_bedload_current_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private
    public :: braids_current_bedload,braids_surface_statistics,braids_gaia_engelund_cc
contains
    pure subroutine braids_surface_statistics(n,mass,bulk_density,diameter,sediment_type,dg,dgsd,sand_fraction)
        ! The active layer stores mass per bed area, but its composition and
        ! transport availability are volume fractions. With fixed class
        ! porosities this is Delft3D getvfrac/IPorosity=0: mass/CDryB.
        ! Mud is excluded and the remaining surface mixture is renormalized.
        integer, intent(in) :: n,sediment_type(n)
        real(8), intent(in) :: mass(n),bulk_density(n),diameter(n)
        real(8), intent(out) :: dg,dgsd,sand_fraction
        real(8) :: share(n),total,mean_log
        share=merge(max(mass,0.d0)/bulk_density,0.d0,sediment_type==0)
        total=sum(share)
        dg=exp(sum(log(diameter))/n);dgsd=1.d0;sand_fraction=0.d0
        if(total<=0.d0) return ! no transport availability in an empty/mud-only bed
        share=share/total
        mean_log=sum(share*log(diameter))
        dg=exp(mean_log)
        dgsd=exp(sqrt(max(0.d0,sum(share*(log(diameter)-mean_log)**2))))
        sand_fraction=sum(share,mask=sediment_type==0.and.diameter<2.d-3)
    end subroutine

    pure subroutine braids_current_bedload(law,h,u,v,manning_n,roughness_z0,di,rhos,rhow,g,dg,dgsd,sand_fraction, &
                                          calibration,qx,qy,tau_ratio,valid,constant_chezy,gaia_skin_roughness_ratio)
        integer, intent(in) :: law
        real(8), intent(in) :: h,u,v,manning_n,roughness_z0,di,rhos,rhow,g,dg,dgsd,sand_fraction,calibration
        real(8), intent(out) :: qx,qy,tau_ratio
        logical, intent(out) :: valid
        real(8), optional, intent(in) :: constant_chezy
        real(8), optional, intent(in) :: gaia_skin_roughness_ratio
        real(8) :: speed,chezy,tau,delta,theta,theta_c,dstar,magnitude
        real(8) :: b,taurm,tauri,phi,wistar,ustar,sigma_phi,cdrag,gaia_q,gaia_theta,gaia_mu

        qx=0.d0;qy=0.d0;tau_ratio=0.d0
        valid=all(ieee_is_finite([h,u,v,manning_n,roughness_z0,di,rhos,rhow,g,dg,dgsd,sand_fraction,calibration]))
        if(.not.valid) return
        valid=h>=0.d0.and.manning_n>=0.d0.and.di>0.d0.and.rhos>rhow.and.rhow>0.d0.and.g>0.d0.and. &
              roughness_z0>0.d0.and.dg>0.d0.and.dgsd>=1.d0.and.sand_fraction>=0.d0.and. &
              sand_fraction<=1.d0.and.calibration>=0.d0
        if(.not.valid.or.h<=0.d0.or.calibration==0.d0) return
        if(present(constant_chezy)) then
            valid=law==8.and.ieee_is_finite(constant_chezy).and.constant_chezy>0.d0
            if(.not.valid) return
        endif
        if(present(gaia_skin_roughness_ratio)) then
            valid=law==9.and.ieee_is_finite(gaia_skin_roughness_ratio).and.gaia_skin_roughness_ratio>0.d0
            if(.not.valid) return
        else if(law==9) then
            valid=.false.;return
        endif
        if(law/=4.and.manning_n==0.d0) return
        speed=sqrt(u*u+v*v)
        if(speed<=1.d-12) return
        chezy=huge(1.d0)
        if(manning_n>0.d0) chezy=h**(1.d0/6.d0)/manning_n
        if(law==8.and.present(constant_chezy)) chezy=constant_chezy
        if(chezy<=1.d-12) return
        delta=rhos/rhow-1.d0
        tau=rhow*g*speed*speed/(chezy*chezy)
        theta=tau/((rhos-rhow)*g*di)
        dstar=(g*delta/1.d-12)**(1.d0/3.d0)*di
        theta_c=.30d0/(1.d0+1.2d0*dstar)+.055d0*(1.d0-exp(-.020d0*dstar))
        tau_ratio=theta/max(theta_c,tiny(1.d0))
        magnitude=0.d0

        select case(law)
        case(3) ! Generalized Ashida-Michiue: A=17, theta_c=0.05, m=3/2, p=q=1.
            if(speed<1.d-6.or.chezy<1.d-6) return
            theta_c=.05d0;tau_ratio=theta/theta_c
            if(theta>theta_c) magnitude=17.d0*calibration*sqrt(g*delta*di**3)*theta**1.5d0* &
                (1.d0-theta_c/theta)*(1.d0-sqrt(theta_c/theta))
        case(4) ! Soulsby current-only branch (zero wave orbital velocity).
            if(speed<1.d-6.or.h<.01d0.or.h>200.d0) return
            cdrag=(.41d0/(1.d0+log(roughness_z0/h)))**2
            tau=rhow*cdrag*speed**2;theta=tau/((rhos-rhow)*g*di);tau_ratio=theta/max(theta_c,tiny(1.d0))
            if(theta>theta_c) magnitude=calibration*sqrt(g*delta*di**3)*12.d0* &
                sqrt(theta+1.d-4)*(theta-theta_c)
        case(5) ! Wilcock-Crowe (2003).
            if(chezy<1.d-6.or.dg<1.d-6) return
            b=.67d0/(1.d0+exp(1.5d0-di/dg))
            taurm=(.021d0+.015d0*exp(-20.d0*sand_fraction))*(rhos-rhow)*g*dg
            tauri=taurm*(di/dg)**b
            call wilcock_rate(tau,tauri,wistar,phi)
            ustar=sqrt(tau/rhow);magnitude=calibration*wistar*ustar**3/(delta*g)
            tau_ratio=phi
        case(6) ! Modified Wilcock-Crowe; standard theta_c0=.021, alpha_0=.33.
            if(chezy<1.d-6.or.dg<1.d-6) return
            b=(1.d0-.33d0)/(1.d0+exp(1.5d0-di/dg))
            sigma_phi=(log(dgsd)/log(2.d0))**2
            taurm=(.021d0+.015d0/(1.d0+exp(10.1d0*sigma_phi-14.14d0)))*(rhos-rhow)*g*dg
            tauri=taurm*(di/dg)**b
            call wilcock_rate(tau,tauri,wistar,phi)
            ustar=sqrt(tau/rhow);magnitude=calibration*wistar*ustar**3/(delta*g)
            tau_ratio=phi
        case(7) ! Gaeuman et al. (2009); standard theta_c0=.03, alpha_0=.30.
            if(chezy<1.d-6.or.dg<1.d-6) return
            b=(1.d0-.30d0)/(1.d0+exp(1.9d0-di/(3.d0*dg)))
            sigma_phi=(log(dgsd)/log(2.d0))**2
            taurm=(.03d0+.022d0/(1.d0+exp(7.1d0*sigma_phi-11.786d0)))*(rhos-rhow)*g*dg
            tauri=taurm*(di/dg)**b
            call wilcock_rate(tau,tauri,wistar,phi)
            ustar=sqrt(tau/rhow);magnitude=calibration*wistar*ustar**3/(delta*g)
            tau_ratio=phi
        case(8) ! Engelund-Hansen (1967), Delft3D tranb1 current-only total load.
            ! BRAIDS exposes the no-suspended split used by the Struiksma T2
            ! depth-averaged reference.  Keep the source formula in its
            ! original volume-transport units until the common mass conversion
            ! below.  Unlike every existing closure, mode-3 slope correction
            ! consumes the absolute Shields number rather than tau/tau_cr.
            if(chezy<1.d-6) return
            tau_ratio=theta
            magnitude=.05d0*calibration*(chezy**2/g)*di**1.5d0*sqrt(g*delta)*theta**2.5d0
        case(9) ! TELEMAC-GAIA formula 3: Engelund-Hansen + Chollet-Cunge.
            ! This is deliberately distinct from law 8.  GAIA applies the
            ! KSP=ratio*d skin correction only in TETAP, while the transport
            ! denominator retains the total hydraulic friction coefficient.
            call braids_gaia_engelund_cc(tau,2.d0*g/(chezy*chezy),h,di,rhos,rhow,g, &
             gaia_skin_roughness_ratio,gaia_q,gaia_theta,gaia_mu,valid)
            if(.not.valid) return
            tau_ratio=tau
            magnitude=calibration*gaia_q/rhos
        case default
            valid=.false.;return
        end select
        qx=rhos*magnitude*u/speed;qy=rhos*magnitude*v/speed
        valid=all(ieee_is_finite([qx,qy,tau_ratio])).and.min(qx*qx+qy*qy,tau_ratio)>=0.d0
    end subroutine

    pure subroutine braids_gaia_engelund_cc(tau_b,cf,h,d,rhos,rhow,g,skin_roughness_ratio, &
                                             q_mass,theta_cc,skin_factor,valid)
        ! Exact scalar transcription of TELEMAC v8p5r1
        ! gaia/bedload_engel_cc.f plus the ICR=1 skin correction in tob_gaia.f.
        ! q_mass is total bed-load mass transport per unit width [kg/(m s)].
        real(8), intent(in) :: tau_b,cf,h,d,rhos,rhow,g,skin_roughness_ratio
        real(8), intent(out) :: q_mass,theta_cc,skin_factor
        logical, intent(out) :: valid
        real(8) :: ksp,a,cf_skin,tetap,relative

        q_mass=0.d0;theta_cc=0.d0;skin_factor=0.d0
        valid=all(ieee_is_finite([tau_b,cf,h,d,rhos,rhow,g,skin_roughness_ratio]))
        if(.not.valid) return
        valid=tau_b>=0.d0.and.cf>0.d0.and.h>=0.d0.and.d>0.d0.and.rhos>rhow.and. &
              rhow>0.d0.and.g>0.d0.and.skin_roughness_ratio>0.d0
        if(.not.valid) return
        ksp=skin_roughness_ratio*d
        if(h>ksp) then
            a=2.5d0*log(11.036d0*max(h,ksp)/ksp)
            if(a<=0.d0.or..not.ieee_is_finite(a)) then
                valid=.false.;return
            endif
            cf_skin=2.d0/(a*a)
            skin_factor=cf_skin/cf
        endif
        tetap=tau_b*skin_factor/((rhos-rhow)*g*d)+1.d-6
        if(tetap<=.06d0) then
            theta_cc=0.d0
        else if(tetap<.384d0) then
            theta_cc=sqrt(2.5d0*(tetap-.06d0))
        else if(tetap<1.080d0) then
            theta_cc=1.066d0*tetap**.176d0
        else
            theta_cc=tetap
        endif
        relative=rhos/rhow-1.d0
        q_mass=.1d0*rhos*sqrt(relative*g*d**3)*theta_cc**2.5d0/max(cf,1.d-6)
        valid=all(ieee_is_finite([q_mass,theta_cc,skin_factor])).and. &
              min(q_mass,theta_cc,skin_factor)>=0.d0
    end subroutine braids_gaia_engelund_cc

    pure subroutine wilcock_rate(tau,reference_tau,wistar,phi)
        real(8), intent(in) :: tau,reference_tau
        real(8), intent(out) :: wistar,phi
        phi=max(0.d0,tau/reference_tau)
        if(phi<1.35d0) then
            wistar=.002d0*phi**7.5d0
        else
            wistar=14.d0*(1.d0-.894d0/sqrt(phi))**4.5d0
        endif
    end subroutine
end module
