! SPDX-License-Identifier: AGPL-3.0-or-later
! Bed-load slope correction matching Delft3D D-Flow FM ISlope=2.
! Derived from fm_adjust_bedload.f90, Stichting Deltares, 2017-2026.
module braids_bed_slope_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private
    public :: braids_delft3d_bagnold_ikeda
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
end module braids_bed_slope_module
