! SPDX-License-Identifier: GPL-3.0-only
! Derived from Delft3D bedtr1993 / bedbc1993, Stichting Deltares, 2011-2026.
! Pinned source hashes and scope: braids_support/sediment/bedload_vr1993/.
! Current-only, horizontal-bed, single noncohesive fraction (fraction=1).
! Formula API only: no bed update, face limiter, calibration or porosity factor.
module braids_bedload_vr1993_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private
    public :: braids_vr1993_bedload_raw, braids_vr1993_bedload_cell
contains
    pure subroutine braids_vr1993_bedload_raw(u,v,speed,d50,rhos,dstar,ustar,muc,tau_ratio,eps,qbx,qby)
        ! Preconditions: finite inputs, d50/rhos/dstar>0, speed/ustar/muc/eps>=0.
        ! speed is the magnitude associated with the supplied velocity vector.
        ! Output: solid MASS transport per unit width [kg/(m s)], already includes rhos.
        ! Preserve Delft3D's 1e-10 transport-stage floor, including below threshold.
        real(8), intent(in) :: u,v,speed,d50,rhos,dstar,ustar,muc,tau_ratio,eps
        real(8), intent(out) :: qbx,qby
        real(8) :: stage,skin_ustar,magnitude
        stage=max(1.d-10,tau_ratio-1.d0)
        skin_ustar=ustar*muc**0.5d0
        magnitude=0.5d0*rhos*d50*skin_ustar*dstar**(-0.3d0)*stage
        qbx=0.d0; qby=0.d0
        if (speed>eps) then
            qbx=magnitude*u/speed
            qby=magnitude*v/speed
        endif
    end subroutine braids_vr1993_bedload_raw

    pure subroutine braids_vr1993_bedload_cell(h,hu,hv,d50,d90,rhos,rhow,z0,dstar,taucr, &
                                               minimum_depth,qbx,qby,ustar,muc,tau_ratio,valid, &
                                               observation_depth)
        ! 2DH adapter: characteristic velocity at observation_depth/e,
        ! kappa=.41, iopkcw=1.  By default observation_depth=h.  The optional
        ! split is required when FLOW's finite constitutive work depth differs
        ! from the physical height at which the resolved velocity is defined.
        ! No waves, additional shear, mud, spiral-flow correction or bed slope.
        ! dstar/taucr come from braids_vr1993_grain_parameters at setup time.
        ! minimum_depth is explicit caller policy (e.g. existing SedThr=0.5 m).
        ! Dry/shallow cells return zero without evaluating logarithms. Invalid
        ! material/roughness inputs return valid=.false.; caller must reject them.
        real(8), intent(in) :: h,hu,hv,d50,d90,rhos,rhow,z0,dstar,taucr,minimum_depth
        real(8), intent(in), optional :: observation_depth
        real(8), intent(out) :: qbx,qby,ustar,muc,tau_ratio
        logical, intent(out) :: valid
        real(8) :: speed,u,v,rc,f1c,fc,tauc,velocity_height
        qbx=0.d0; qby=0.d0; ustar=0.d0; muc=0.d0; tau_ratio=0.d0
        valid=.false.
        if (.not.all(ieee_is_finite([h,hu,hv,d50,d90,rhos,rhow,z0,dstar,taucr,minimum_depth]))) return
        velocity_height=h
        if (present(observation_depth)) then
            if (.not.ieee_is_finite(observation_depth).or.observation_depth<=0.d0) return
            velocity_height=observation_depth
        endif
        if (min(d50,d90,rhow,z0,dstar,taucr)<=0.d0.or.rhos<=rhow.or.d90<d50) return
        if (minimum_depth<0.d0.or.h<0.d0) return
        valid=.true.
        if (h<=minimum_depth) return
        ! Require positive log-law arguments above unity. This adapter explicitly
        ! rejects unresolved roughness/grain scales rather than masking a singularity.
        rc=30.d0*z0
        if (12.d0*h<=max(rc,3.d0*d90)) then
            valid=.false.
            return
        endif
        u=hu/h; v=hv/h
        speed=sqrt(hu*hu+hv*hv)/h
        ustar=speed*0.41d0/log(1.d0+(velocity_height/exp(1.d0))/z0)
        tauc=rhow*ustar**2
        f1c=0.24d0*log10(12.d0*h/(3.d0*d90))**(-2)
        fc=0.24d0*log10(12.d0*h/rc)**(-2)
        muc=f1c/fc
        tau_ratio=muc*tauc/taucr
        call braids_vr1993_bedload_raw(u,v,speed,d50,rhos,dstar,ustar,muc,tau_ratio,1.d-6,qbx,qby)
        valid=all(ieee_is_finite([qbx,qby,ustar,muc,tau_ratio]))
    end subroutine braids_vr1993_bedload_cell
end module braids_bedload_vr1993_module
