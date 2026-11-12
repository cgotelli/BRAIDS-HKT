! Equilibrium secondary-flow correction for depth-averaged bed-load transport.
!
! The equations mirror the D-Flow FM curvature, equilibrium spiral-intensity,
! and eqtran direction branches.  This module intentionally changes direction
! only: the transport-law magnitude is preserved exactly.
module braids_secondary_flow_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private
    public :: braids_streamline_curvature,braids_delft3d_spiral_transport, &
              braids_spiral_equilibrium,braids_spiral_adaptation_time, &
              braids_spiral_relax_exact,braids_spiral_transport_from_intensity, &
              braids_spiral_stress
contains
    pure subroutine braids_streamline_curvature(u,v,dudx,dudy,dvdx,dvdy,curvature,valid)
        real(8), intent(in) :: u,v,dudx,dudy,dvdx,dvdy
        real(8), intent(out) :: curvature
        logical, intent(out) :: valid
        real(8) :: speed

        curvature=0.d0
        valid=all(ieee_is_finite([u,v,dudx,dudy,dvdx,dvdy]))
        if(.not.valid) return
        speed=sqrt(u*u+v*v)
        ! D-Flow FM get_curvature uses this absolute threshold.
        if(speed<1.d-3) return
        curvature=-(u*u*dvdx-v*v*dudy+u*v*(dvdy-dudx))/speed**3
        valid=ieee_is_finite(curvature)
        if(.not.valid) curvature=0.d0
    end subroutine

    pure subroutine braids_delft3d_spiral_transport(h,u,v,manning_n,g,espir,curvature,qx,qy, &
                                                     spiral_intensity,alpha,valid)
        real(8), intent(in) :: h,u,v,manning_n,g,espir,curvature
        real(8), intent(inout) :: qx,qy
        real(8), intent(out) :: spiral_intensity,alpha
        logical, intent(out) :: valid
        real(8) :: speed,chezy,magnitude,tx,ty,direction_norm

        spiral_intensity=0.d0;alpha=0.d0
        valid=all(ieee_is_finite([h,u,v,manning_n,g,espir,curvature,qx,qy]))
        if(.not.valid) return
        valid=h>=0.d0.and.manning_n>0.d0.and.g>0.d0.and.espir>=0.d0
        if(.not.valid.or.h<=0.d0) return
        speed=sqrt(u*u+v*v);magnitude=sqrt(qx*qx+qy*qy)
        if(speed<=0.d0.or.magnitude<=0.d0) return
        chezy=h**(1.d0/6.d0)/manning_n
        if(.not.ieee_is_finite(chezy).or.chezy<=0.d0) then
            valid=.false.;return
        endif
        ! Planar equilibrium intensity: I = h * curvature * |U|.  Coriolis is
        ! deliberately absent because BRAIDS currently uses Cartesian planar
        ! hydraulics and exposes no latitude-dependent secondary-flow model.
        spiral_intensity=h*curvature*speed
        alpha=12.5d0*espir*(1.d0-sqrt(g)/(0.4d0*chezy))*spiral_intensity/speed
        tx=u+alpha*v;ty=v-alpha*u
        direction_norm=sqrt(tx*tx+ty*ty)
        if(direction_norm<=epsilon(1.d0)) then
            qx=0.d0;qy=0.d0
        else
            qx=magnitude*tx/direction_norm;qy=magnitude*ty/direction_norm
        endif
        valid=all(ieee_is_finite([qx,qy,spiral_intensity,alpha]))
    end subroutine

    pure subroutine braids_spiral_equilibrium(h,u,v,curvature,spiral_intensity,active,valid)
        ! Planar equilibrium intensity.  Coriolis is deliberately absent from
        ! the Cartesian BRAIDS model.
        real(8), intent(in) :: h,u,v,curvature
        real(8), intent(out) :: spiral_intensity
        logical, intent(out) :: active,valid
        real(8) :: speed

        spiral_intensity=0.d0;active=.false.
        valid=all(ieee_is_finite([h,u,v,curvature]))
        if(.not.valid) return
        valid=h>=0.d0
        if(.not.valid.or.h<=0.d0) return
        speed=sqrt(u*u+v*v)
        if(.not.ieee_is_finite(speed)) then
            valid=.false.;return
        endif
        if(speed<1.d-3) return
        spiral_intensity=h*curvature*speed
        valid=ieee_is_finite(spiral_intensity)
        if(.not.valid) then
            spiral_intensity=0.d0;return
        endif
        active=.true.
    end subroutine

    pure subroutine braids_spiral_adaptation_time(h,u,v,manning_n,g,time_a,active,valid)
        ! Kalkwijk--Booij/Delft3D adaptation time for non-equilibrium spiral
        ! intensity.  The Chezy floor keeps alpha below one half for ordinary
        ! terrestrial gravity and makes the adaptation length positive.
        real(8), intent(in) :: h,u,v,manning_n,g
        real(8), intent(out) :: time_a
        logical, intent(out) :: active,valid
        real(8), parameter :: von_karman=0.41d0
        real(8) :: speed,chezy,alpha,length_a

        time_a=huge(1.d0);active=.false.
        valid=all(ieee_is_finite([h,u,v,manning_n,g]))
        if(.not.valid) return
        valid=h>=0.d0.and.manning_n>0.d0.and.g>0.d0
        if(.not.valid.or.h<=0.d0) return
        speed=sqrt(u*u+v*v)
        if(.not.ieee_is_finite(speed)) then
            valid=.false.;return
        endif
        if(speed<1.d-3) return
        chezy=h**(1.d0/6.d0)/manning_n
        if(.not.ieee_is_finite(chezy).or.chezy<=0.d0) then
            valid=.false.;return
        endif
        alpha=sqrt(g)/(von_karman*max(chezy,20.d0))
        if(.not.ieee_is_finite(alpha).or.alpha<=0.d0.or.alpha>=.5d0) then
            valid=.false.;return
        endif
        length_a=(1.d0-2.d0*alpha)*h/(2.d0*von_karman*von_karman*alpha)
        time_a=length_a/speed
        valid=ieee_is_finite(time_a).and.time_a>0.d0
        if(.not.valid) then
            time_a=huge(1.d0);return
        endif
        active=.true.
    end subroutine

    pure subroutine braids_spiral_relax_exact(h,stored_mass,equilibrium,time_a,dt, &
                                               updated_mass,source_increment,valid)
        ! Exact local integration of d(hI)/dt=h(Ieq-I)/time_a, with h fixed
        ! during this source substep.  The helper below avoids cancellation
        ! for a very small dt/time_a and directly takes the equilibrium limit
        ! for a very large ratio.
        real(8), intent(in) :: h,stored_mass,equilibrium,time_a,dt
        real(8), intent(out) :: updated_mass,source_increment
        logical, intent(out) :: valid
        real(8) :: target,ratio,fraction

        updated_mass=stored_mass;source_increment=0.d0
        valid=all(ieee_is_finite([h,stored_mass,equilibrium,time_a,dt]))
        if(.not.valid) return
        valid=h>=0.d0.and.time_a>0.d0.and.dt>=0.d0
        if(.not.valid.or.dt<=0.d0) return
        target=h*equilibrium
        if(.not.ieee_is_finite(target)) then
            valid=.false.;return
        endif
        if(time_a<=dt/50.d0) then
            source_increment=target-stored_mass
            updated_mass=target
        else
            ratio=dt/time_a
            fraction=braids_one_minus_exp_negative(ratio)
            source_increment=(target-stored_mass)*fraction
            updated_mass=stored_mass+source_increment
        endif
        valid=all(ieee_is_finite([updated_mass,source_increment]))
        if(.not.valid) then
            updated_mass=stored_mass;source_increment=0.d0
        endif
    end subroutine

    pure subroutine braids_spiral_transport_from_intensity(h,u,v,manning_n,g,espir, &
                                                            spiral_intensity,qx,qy,alpha,valid)
        ! Direction-only near-bed transport correction using a supplied,
        ! possibly signed, prognostic intensity.  This duplicates the legacy
        ! arithmetic intentionally; the legacy entry point above is left
        ! untouched so existing v28 calculations retain their operation order.
        real(8), intent(in) :: h,u,v,manning_n,g,espir,spiral_intensity
        real(8), intent(inout) :: qx,qy
        real(8), intent(out) :: alpha
        logical, intent(out) :: valid
        real(8) :: speed,chezy,magnitude,tx,ty,direction_norm

        alpha=0.d0
        valid=all(ieee_is_finite([h,u,v,manning_n,g,espir,spiral_intensity,qx,qy]))
        if(.not.valid) return
        valid=h>=0.d0.and.manning_n>0.d0.and.g>0.d0.and.espir>=0.d0
        if(.not.valid.or.h<=0.d0) return
        speed=sqrt(u*u+v*v);magnitude=sqrt(qx*qx+qy*qy)
        if(speed<=0.d0.or.magnitude<=0.d0) return
        chezy=h**(1.d0/6.d0)/manning_n
        if(.not.ieee_is_finite(chezy).or.chezy<=0.d0) then
            valid=.false.;return
        endif
        alpha=12.5d0*espir*(1.d0-sqrt(g)/(0.4d0*chezy))*spiral_intensity/speed
        tx=u+alpha*v;ty=v-alpha*u
        direction_norm=sqrt(tx*tx+ty*ty)
        if(direction_norm<=epsilon(1.d0)) then
            qx=0.d0;qy=0.d0
        else
            qx=magnitude*tx/direction_norm;qy=magnitude*ty/direction_norm
        endif
        valid=all(ieee_is_finite([qx,qy,alpha]))
    end subroutine

    pure subroutine braids_spiral_stress(h,u,v,manning_n,g,beta_spiral,spiral_intensity, &
                                          tensor,signal_speed,active,valid)
        ! Symmetric, trace-free secondary-flow dispersion stress used by the
        ! optional momentum coupling.  signal_speed is the conservative
        ! explicit face-speed bound proposed for the finite-volume caller.
        real(8), intent(in) :: h,u,v,manning_n,g,beta_spiral,spiral_intensity
        real(8), intent(out) :: tensor(2,2),signal_speed
        logical, intent(out) :: active,valid
        real(8), parameter :: von_karman=0.41d0
        real(8) :: speed,chezy,alpha,coefficient,beta

        tensor=0.d0;signal_speed=0.d0;active=.false.
        valid=all(ieee_is_finite([h,u,v,manning_n,g,beta_spiral,spiral_intensity]))
        if(.not.valid) return
        valid=h>=0.d0.and.manning_n>0.d0.and.g>0.d0.and.beta_spiral>=0.d0
        if(.not.valid.or.h<=0.d0.or.beta_spiral<=0.d0) return
        speed=sqrt(u*u+v*v)
        if(.not.ieee_is_finite(speed)) then
            valid=.false.;return
        endif
        if(speed<1.d-3) return
        chezy=h**(1.d0/6.d0)/manning_n
        if(.not.ieee_is_finite(chezy).or.chezy<=0.d0) then
            valid=.false.;return
        endif
        alpha=sqrt(g)/(von_karman*chezy)
        coefficient=beta_spiral*(5.d0*alpha-15.6d0*alpha**2+37.5d0*alpha**3)
        beta=coefficient*spiral_intensity/speed
        tensor(1,1)=-2.d0*h*beta*u*v
        tensor(1,2)=h*beta*(u*u-v*v)
        tensor(2,1)=tensor(1,2)
        tensor(2,2)=-tensor(1,1)
        signal_speed=2.d0*abs(coefficient*spiral_intensity)
        valid=all(ieee_is_finite([tensor(1,1),tensor(1,2),tensor(2,1),tensor(2,2),signal_speed]))
        if(.not.valid) then
            tensor=0.d0;signal_speed=0.d0;return
        endif
        active=signal_speed>0.d0
    end subroutine

    pure real(8) function braids_one_minus_exp_negative(value) result(result_value)
        real(8), intent(in) :: value
        if(value<=0.d0) then
            result_value=0.d0
        else if(value<1.d-5) then
            result_value=value*(1.d0+value*(-.5d0+value*(1.d0/6.d0+ &
                         value*(-1.d0/24.d0+value*(1.d0/120.d0-value/720.d0)))))
        else if(value>=50.d0) then
            result_value=1.d0
        else
            result_value=1.d0-exp(-value)
        endif
    end function
end module
