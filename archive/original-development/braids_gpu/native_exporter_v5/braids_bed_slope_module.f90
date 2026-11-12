! SPDX-License-Identifier: AGPL-3.0-or-later
! Bed-load slope corrections matching Delft3D depth-averaged formulations.
! Derived from fm_adjust_bedload.f90, Stichting Deltares, 2017-2026.
module braids_bed_slope_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private
    public :: braids_delft3d_bagnold_ikeda,braids_delft3d_koch_flokstra,braids_gaia_talmon_slope
contains
    pure subroutine braids_delft3d_bagnold_ikeda(qx,qy,down_x,down_y,tau_ratio, &
                                                alfabs,alfabn,friction_angle_deg, &
                                                corrected_x,corrected_y,valid)
        ! q is the uncorrected bed-load mass transport [kg/(m s)].
        ! down is the dimensionless bed gradient, positive downhill, as in Delft3D.
        ! This follows D-Flow FM ISlope=2: Bagnold (1956) longitudinally and
        ! Ikeda/Van Rijn transversely.
        real(8), intent(in) :: qx,qy,down_x,down_y,tau_ratio,alfabs,alfabn,friction_angle_deg
        real(8), intent(out) :: corrected_x,corrected_y
        logical, intent(out) :: valid
        real(8) :: magnitude,along_x,along_y,normal_x,normal_y,dzds,dzdp
        real(8) :: tan_phi,bagnold,along_factor,normal_factor,pi

        corrected_x=qx; corrected_y=qy
        valid=all(ieee_is_finite([qx,qy,down_x,down_y,tau_ratio,alfabs,alfabn,friction_angle_deg]))
        if(.not.valid) return
        valid=tau_ratio>=0.d0.and.alfabs>=0.d0.and.alfabn>=0.d0.and. &
              friction_angle_deg>0.d0.and.friction_angle_deg<90.d0
        if(.not.valid) return
        magnitude=sqrt(qx*qx+qy*qy)
        ! Delft3D uses eps=1e-6 in fm_adjust_bedload.
        if(magnitude<=1.d-6) return
        along_x=qx/magnitude; along_y=qy/magnitude
        normal_x=-along_y; normal_y=along_x
        dzds=down_x*along_x+down_y*along_y
        dzdp=down_x*normal_x+down_y*normal_y
        pi=acos(-1.d0); tan_phi=tan(friction_angle_deg*pi/180.d0)
        dzds=min(0.9d0*tan_phi,dzds)
        bagnold=tan_phi/(cos(atan(dzds))*(tan_phi-dzds))
        along_factor=max(0.d0,1.d0+alfabs*(bagnold-1.d0))
        if(tau_ratio>=1.d0) then
            normal_factor=alfabn*dzdp/sqrt(tau_ratio)
        else
            normal_factor=alfabn*dzdp
        endif
        corrected_x=along_factor*magnitude*(along_x+normal_factor*normal_x)
        corrected_y=along_factor*magnitude*(along_y+normal_factor*normal_y)
        valid=all(ieee_is_finite([corrected_x,corrected_y]))
    end subroutine braids_delft3d_bagnold_ikeda

    pure subroutine braids_delft3d_koch_flokstra(qx,qy,down_x,down_y,shields, &
                                                 alfabs,ashld,bshld, &
                                                 corrected_x,corrected_y,valid)
        ! Van Bendegom (1947) / Koch & Flokstra (1980), Delft3D ISlope=3,
        ! specialized to the published Struiksma T2 settings CShld=DShld=0.
        ! q is mass transport [kg/(m s)], down is the dimensionless bed
        ! gradient positive downhill, and shields is ABSOLUTE Shields stress.
        ! The two grain/depth powers omitted here are exactly one only for
        ! CShld=DShld=0; callers must not represent nonzero values implicitly.
        real(8), intent(in) :: qx,qy,down_x,down_y,shields,alfabs,ashld,bshld
        real(8), intent(out) :: corrected_x,corrected_y
        logical, intent(out) :: valid
        real(8) :: magnitude,unit_x,unit_y,dzds,ftheta,direction_x,direction_y,norm

        corrected_x=qx;corrected_y=qy
        valid=all(ieee_is_finite([qx,qy,down_x,down_y,shields,alfabs,ashld,bshld]))
        if(.not.valid) return
        valid=shields>=0.d0.and.alfabs>=0.d0.and.ashld>=0.d0.and.bshld>=0.d0
        if(.not.valid) return
        magnitude=sqrt(qx*qx+qy*qy)
        if(magnitude<=1.d-6) return
        unit_x=qx/magnitude;unit_y=qy/magnitude
        dzds=down_x*unit_x+down_y*unit_y
        ! Delft3D leaves an exactly flat bed unchanged, including the limiting
        ! zero-stress case for which 0**0 would otherwise be ambiguous.
        if(down_x==0.d0.and.down_y==0.d0) return
        ftheta=0.d0
        if(shields>0.d0) ftheta=ashld*shields**bshld
        direction_x=ftheta*unit_x+down_x
        direction_y=ftheta*unit_y+down_y
        norm=sqrt(direction_x*direction_x+direction_y*direction_y)
        if(norm<=tiny(1.d0)) then
            valid=.false.;return
        endif
        magnitude=magnitude*(1.d0+alfabs*dzds)
        corrected_x=magnitude*direction_x/norm
        corrected_y=magnitude*direction_y/norm
        valid=all(ieee_is_finite([corrected_x,corrected_y]))
    end subroutine braids_delft3d_koch_flokstra

    pure subroutine braids_gaia_talmon_slope(u,v,gradz_x,gradz_y,tau_b,rhos,rhow,g,d, &
                                              beta,beta2,stress_x,stress_y,magnitude_factor,valid)
        ! TELEMAC v8p5r1 BEDLOAD_EFFPNT: DEVIA=2 and SLOPEFF=1.
        ! gradz is the uphill bed-elevation gradient.  The returned vector is
        ! the slope-corrected effective bed shear, before secondary current.
        real(8), intent(in) :: u,v,gradz_x,gradz_y,tau_b,rhos,rhow,g,d,beta,beta2
        real(8), intent(out) :: stress_x,stress_y,magnitude_factor
        logical, intent(out) :: valid
        real(8) :: speed,flow_x,flow_y,tt1,direction_x,direction_y,norm

        stress_x=0.d0;stress_y=0.d0;magnitude_factor=0.d0
        valid=all(ieee_is_finite([u,v,gradz_x,gradz_y,tau_b,rhos,rhow,g,d,beta,beta2]))
        if(.not.valid) return
        valid=tau_b>=0.d0.and.rhos>rhow.and.rhow>0.d0.and.g>0.d0.and.d>0.d0.and. &
              beta>=0.d0.and.beta2>0.d0
        if(.not.valid) return
        speed=sqrt(u*u+v*v)
        if(speed<=1.d-12.or.tau_b<=0.d0) return
        flow_x=u/speed;flow_y=v/speed
        tt1=sqrt(((rhos-rhow)*g*d/(beta2*beta2))/max(tau_b,1.d-10))
        direction_x=flow_x-tt1*gradz_x
        direction_y=flow_y-tt1*gradz_y
        norm=sqrt(direction_x*direction_x+direction_y*direction_y)
        if(norm<=tiny(1.d0)) then
            valid=.false.;return
        endif
        magnitude_factor=max(0.d0,1.d0-beta*(gradz_x*flow_x+gradz_y*flow_y))
        stress_x=tau_b*magnitude_factor*direction_x/norm
        stress_y=tau_b*magnitude_factor*direction_y/norm
        valid=all(ieee_is_finite([stress_x,stress_y,magnitude_factor]))
    end subroutine braids_gaia_talmon_slope
end module braids_bed_slope_module
