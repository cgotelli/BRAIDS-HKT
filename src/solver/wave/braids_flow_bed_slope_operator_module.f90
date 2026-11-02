! SPDX-License-Identifier: AGPL-3.0-or-later
!
! Isolated Delft3D4 FLOW 2DH bed-load direction operator.
!
! This module transcribes the constitutive arithmetic in the locked
! adjust_bedload.f90 ISlope=2/3/4 branches.  It is intentionally not wired into
! the accepted BRAIDS face path yet: the operator gate must pass before that
! integration changes any public profile.
module braids_flow_bed_slope_operator_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: flow_slope_ok = 0
    integer, parameter, public :: flow_slope_invalid_mode = 1
    integer, parameter, public :: flow_slope_invalid_input = 2
    integer, parameter, public :: flow_slope_singular = 3
    integer, parameter, public :: flow_slope_invalid_output = 4

    public :: braids_flow_bed_slope_direction
    public :: braids_flow_bagnold_ikeda
    public :: braids_flow_koch_flokstra
    public :: braids_flow_parker_andrews

contains

    pure subroutine braids_flow_bed_slope_direction( &
            islope,qx,qy,down_x,down_y,stress,depth,d50,mean_diameter, &
            hiding_left,hiding_right,alfabs,alfabn,ashld,bshld,cshld,dshld, &
            alfpa,thcrpa,corrected_x,corrected_y,status)
        ! Inputs use FLOW's conventions:
        !   * q is uncorrected bed-load mass transport [kg/(m s)];
        !   * down is the dimensionless bed gradient, positive downhill;
        !   * stress is taurat for ISlope=2 and absolute Shields for 3/4;
        !   * hiding_left/right are the selected generic hiding factors at the
        !     two cells adjoining a face (only ISlope=4 consumes them).
        !
        ! Output stays equal to input q on every rejected state.  A caller must
        ! not consume output when status is nonzero.
        integer, intent(in) :: islope
        real(8), intent(in) :: qx,qy,down_x,down_y,stress,depth,d50,mean_diameter
        real(8), intent(in) :: hiding_left,hiding_right
        real(8), intent(in) :: alfabs,alfabn,ashld,bshld,cshld,dshld,alfpa,thcrpa
        real(8), intent(out) :: corrected_x,corrected_y
        integer, intent(out) :: status
        real(8) :: sbedm,dzds,dzdn,tphi,bagnol,alfas,fnorm
        real(8) :: ftheta,sina,cosa,tnorm,hidexploc,denominator
        real(8), parameter :: transport_epsilon=1.d-6

        corrected_x=qx
        corrected_y=qy
        status=flow_slope_invalid_mode
        if(islope<2.or.islope>4) return

        status=flow_slope_invalid_input
        if(.not.all(ieee_is_finite([qx,qy,down_x,down_y,stress,alfabs]))) return
        if(stress<0.d0) return
        select case(islope)
        case(2)
            if(.not.ieee_is_finite(alfabn)) return
        case(3)
            if(.not.all(ieee_is_finite([depth,d50,mean_diameter,ashld,bshld,cshld,dshld]))) return
            if(depth<=0.d0.or.d50<=0.d0.or.mean_diameter<=0.d0) return
        case(4)
            if(.not.all(ieee_is_finite([hiding_left,hiding_right,alfpa,thcrpa]))) return
            if(hiding_left<0.d0.or.hiding_right<0.d0.or.thcrpa<0.d0) return
        end select

        sbedm=sqrt(qx*qx+qy*qy)
        if(.not.ieee_is_finite(sbedm)) return
        status=flow_slope_ok
        if(sbedm<=transport_epsilon) return

        ! adjust_bedload computes both projections before applying the one-sided
        ! fixed-30-degree longitudinal cap.  Preserve its default-real literal
        ! promotion: 30.0/180.0 and 0.9 are rounded in default real first.
        dzds=down_x*qx/sbedm+down_y*qy/sbedm
        dzdn=-down_x*qy/sbedm+down_y*qx/sbedm
        tphi=tan(real(30.0/180.0,8)*acos(-1.d0))
        dzds=min(real(0.9,8)*tphi,dzds)

        select case(islope)
        case(2)
            ! Bagnold longitudinal magnitude followed by Ikeda/Van Rijn
            ! transverse steering.  FLOW attenuates only when taurat >= 1.
            bagnol=tphi/(cos(atan(dzds))*(tphi-dzds))
            alfas=max(0.d0,1.d0+alfabs*(bagnol-1.d0))
            if(stress>=1.d0) then
                fnorm=alfabn*(1.d0/stress)**0.5d0*dzdn
            else
                fnorm=alfabn*dzdn
            endif
            corrected_x=alfas*qx-alfas*qy*fnorm
            corrected_y=alfas*qy+alfas*qx*fnorm

        case(3)
            ! Van Bendegom / Koch-Flokstra.  The CShld and DShld terms are
            ! required FLOW terms, not optional post-corrections.
            ftheta=0.d0
            if(stress/=0.d0) then
                ftheta=ashld*(stress**bshld)* &
                    ((d50/depth)**cshld)*((d50/mean_diameter)**dshld)
            endif
            if(down_x/=0.d0.or.down_y/=0.d0) then
                sina=ftheta*qx/sbedm+down_x
                cosa=ftheta*qy/sbedm+down_y
            else
                sina=qx/sbedm
                cosa=qy/sbedm
            endif
            tnorm=sqrt(sina*sina+cosa*cosa)
            if(.not.ieee_is_finite(tnorm).or.tnorm<=tiny(1.d0)) then
                corrected_x=qx
                corrected_y=qy
                status=flow_slope_singular
                return
            endif
            sbedm=sbedm*(1.d0+alfabs*dzds)
            corrected_x=sbedm*sina/tnorm
            corrected_y=sbedm*cosa/tnorm

        case(4)
            ! Parker-Andrews uses the generic hiding/exposure result selected
            ! elsewhere in FLOW.  It averages the two cell factors here and
            ! uses that average only in the critical-Shields denominator.
            hidexploc=(hiding_left+hiding_right)/2.d0
            ftheta=0.d0
            if(stress/=0.d0) then
                denominator=max(stress*0.1d0,hidexploc*thcrpa)
                if(.not.ieee_is_finite(denominator).or.denominator<=0.d0) then
                    corrected_x=qx
                    corrected_y=qy
                    status=flow_slope_singular
                    return
                endif
                ftheta=alfpa*sqrt(stress/denominator)
            endif
            if(down_x/=0.d0.or.down_y/=0.d0) then
                sina=ftheta*qx/sbedm+down_x
                cosa=ftheta*qy/sbedm+down_y
            else
                sina=qx/sbedm
                cosa=qy/sbedm
            endif
            tnorm=sqrt(sina*sina+cosa*cosa)
            if(.not.ieee_is_finite(tnorm).or.tnorm<=tiny(1.d0)) then
                corrected_x=qx
                corrected_y=qy
                status=flow_slope_singular
                return
            endif
            sbedm=sbedm*(1.d0+alfabs*dzds)
            corrected_x=sbedm*sina/tnorm
            corrected_y=sbedm*cosa/tnorm
        end select

        if(.not.all(ieee_is_finite([corrected_x,corrected_y]))) then
            corrected_x=qx
            corrected_y=qy
            status=flow_slope_invalid_output
        endif
    end subroutine braids_flow_bed_slope_direction

    pure subroutine braids_flow_bagnold_ikeda(qx,qy,down_x,down_y,tau_ratio, &
                                               alfabs,alfabn,corrected_x,corrected_y,status)
        real(8), intent(in) :: qx,qy,down_x,down_y,tau_ratio,alfabs,alfabn
        real(8), intent(out) :: corrected_x,corrected_y
        integer, intent(out) :: status
        call braids_flow_bed_slope_direction(2,qx,qy,down_x,down_y,tau_ratio, &
            1.d0,1.d0,1.d0,1.d0,1.d0,alfabs,alfabn,1.d0,1.d0,0.d0,0.d0, &
            1.d0,1.d0,corrected_x,corrected_y,status)
    end subroutine braids_flow_bagnold_ikeda

    pure subroutine braids_flow_koch_flokstra(qx,qy,down_x,down_y,shields,depth, &
                                               d50,mean_diameter,alfabs,ashld,bshld, &
                                               cshld,dshld,corrected_x,corrected_y,status)
        real(8), intent(in) :: qx,qy,down_x,down_y,shields,depth,d50,mean_diameter
        real(8), intent(in) :: alfabs,ashld,bshld,cshld,dshld
        real(8), intent(out) :: corrected_x,corrected_y
        integer, intent(out) :: status
        call braids_flow_bed_slope_direction(3,qx,qy,down_x,down_y,shields, &
            depth,d50,mean_diameter,1.d0,1.d0,alfabs,0.d0,ashld,bshld,cshld, &
            dshld,1.d0,1.d0,corrected_x,corrected_y,status)
    end subroutine braids_flow_koch_flokstra

    pure subroutine braids_flow_parker_andrews(qx,qy,down_x,down_y,shields, &
                                                hiding_left,hiding_right,alfabs, &
                                                alfpa,thcrpa,corrected_x,corrected_y,status)
        real(8), intent(in) :: qx,qy,down_x,down_y,shields,hiding_left,hiding_right
        real(8), intent(in) :: alfabs,alfpa,thcrpa
        real(8), intent(out) :: corrected_x,corrected_y
        integer, intent(out) :: status
        call braids_flow_bed_slope_direction(4,qx,qy,down_x,down_y,shields, &
            1.d0,1.d0,1.d0,hiding_left,hiding_right,alfabs,0.d0,1.d0,1.d0, &
            0.d0,0.d0,alfpa,thcrpa,corrected_x,corrected_y,status)
    end subroutine braids_flow_parker_andrews

end module braids_flow_bed_slope_operator_module
