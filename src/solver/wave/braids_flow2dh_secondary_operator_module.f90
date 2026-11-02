! Isolated classic Delft3D4 FLOW 2DH secondary-flow constitutive operators.
!
! Locked, unmodified Delft3D source anchors (commit recorded by the Q3-H
! source contract):
!   secrhs.f90:172-200  inverse-radius cap, Coriolis equilibrium and the
!                       old-depth source/new-depth sink split;
!   secbou.f90:132-154  adjacent-interior equilibrium at an open boundary;
!   eqtran.f90:713-747  scalar bed-load direction correction;
!   vihsec.f90:137-207  optional Betac stress coefficient and FLOW clamp.
!
! Curvature differentiation is deliberately not implemented here.  The caller
! supplies a geometry-derived signed inverse radius and the two local metric
! sums used by FLOW's Rmincf cap.  Likewise, this module returns only the local
! stress basis; a grid-specific divergence and the structured-grid fsign are
! outside this component.
!
! This module is additive.  It does not alter braids_secondary_flow_module,
! whose existing entry points retain their historical D-Flow-FM semantics.
module braids_flow2dh_secondary_operator_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    public :: braids_flow2dh_curvature_cap
    public :: braids_flow2dh_secondary_source_sink
    public :: braids_flow2dh_open_boundary_equilibrium
    public :: braids_flow2dh_scalar_bed_rotation
    public :: braids_flow2dh_betac_coefficient
    public :: braids_flow2dh_secondary_state

contains

    pure subroutine braids_flow2dh_curvature_cap(raw_inverse_radius,metric_ksi_sum, &
                                                  metric_eta_sum,rmincf,capped_inverse_radius, &
                                                  cap_limit,was_capped,valid)
        ! `raw_inverse_radius` is already geometry-derived curvature [1/m].
        ! The metric sums are the explicit gksi2/geta2 values in secrhs.
        real(8), intent(in) :: raw_inverse_radius,metric_ksi_sum,metric_eta_sum,rmincf
        real(8), intent(out) :: capped_inverse_radius,cap_limit
        logical, intent(out) :: was_capped,valid
        real(8) :: metric_scale

        capped_inverse_radius=0.d0;cap_limit=0.d0;was_capped=.false.
        valid=all(ieee_is_finite([raw_inverse_radius,metric_ksi_sum,metric_eta_sum,rmincf]))
        if(.not.valid) return
        metric_scale=max(metric_ksi_sum,metric_eta_sum)
        valid=rmincf>0.d0.and.metric_scale>0.d0
        if(.not.valid) return
        cap_limit=1.d0/(rmincf*metric_scale)
        valid=ieee_is_finite(cap_limit).and.cap_limit>0.d0
        if(.not.valid) then
            cap_limit=0.d0;return
        endif
        if(abs(raw_inverse_radius)<cap_limit) then
            capped_inverse_radius=raw_inverse_radius
        else
            capped_inverse_radius=sign(cap_limit,raw_inverse_radius)
            was_capped=.true.
        endif
        valid=ieee_is_finite(capped_inverse_radius)
        if(.not.valid) then
            capped_inverse_radius=0.d0;cap_limit=0.d0;was_capped=.false.
        endif
    end subroutine

    pure subroutine braids_flow2dh_secondary_source_sink(active,depth_old,depth_new, &
                                                          u,v,raw_inverse_radius, &
                                                          metric_ksi_sum,metric_eta_sum, &
                                                          coriolis,raw_chezy,dryflc,rmincf, &
                                                          chzmin,g,von_karman, &
                                                          capped_inverse_radius,cap_limit, &
                                                          equilibrium_target,source_coefficient, &
                                                          sink_coefficient,effective_chezy,alpha, &
                                                          was_capped,valid)
        ! Physical Chezy [m**0.5/s] is supplied directly.  FLOW stores
        ! C/sqrt(g) internally, so max(czosg,Chzmin/sqrt(g)) is exactly
        ! max(C,Chzmin) before alpha=sqrt(g)/(von_karman*C).
        logical, intent(in) :: active
        real(8), intent(in) :: depth_old,depth_new,u,v,raw_inverse_radius
        real(8), intent(in) :: metric_ksi_sum,metric_eta_sum,coriolis,raw_chezy
        real(8), intent(in) :: dryflc,rmincf,chzmin,g,von_karman
        real(8), intent(out) :: capped_inverse_radius,cap_limit,equilibrium_target
        real(8), intent(out) :: source_coefficient,sink_coefficient,effective_chezy,alpha
        logical, intent(out) :: was_capped,valid
        real(8) :: speed,htrsh,h_old,h_new,tau_old,tau_new,rate_old
        logical :: cap_valid

        capped_inverse_radius=0.d0;cap_limit=0.d0;equilibrium_target=0.d0
        source_coefficient=0.d0;sink_coefficient=0.d0;effective_chezy=0.d0;alpha=0.d0
        was_capped=.false.
        valid=all(ieee_is_finite([depth_old,depth_new,u,v,raw_inverse_radius, &
            metric_ksi_sum,metric_eta_sum,coriolis,raw_chezy,dryflc,rmincf, &
            chzmin,g,von_karman]))
        if(.not.valid) return
        valid=dryflc>=0.d0.and.rmincf>0.d0.and.chzmin>0.d0.and.g>0.d0.and. &
              von_karman>0.d0.and.raw_chezy>0.d0
        if(.not.valid) return
        if(.not.active) return

        call braids_flow2dh_curvature_cap(raw_inverse_radius,metric_ksi_sum,metric_eta_sum, &
                                           rmincf,capped_inverse_radius,cap_limit, &
                                           was_capped,cap_valid)
        if(.not.cap_valid) then
            valid=.false.;return
        endif
        htrsh=.5d0*dryflc
        h_old=max(htrsh,depth_old)
        h_new=max(htrsh,depth_new)
        effective_chezy=max(raw_chezy,chzmin)
        alpha=sqrt(g)/(von_karman*effective_chezy)
        valid=ieee_is_finite(alpha).and.alpha>=0.d0.and.alpha<.5d0
        if(.not.valid) return
        tau_old=.5d0*(1.d0-2.d0*alpha)*h_old
        tau_new=.5d0*(1.d0-2.d0*alpha)*h_new
        valid=tau_old>0.d0.and.tau_new>0.d0.and. &
              all(ieee_is_finite([tau_old,tau_new]))
        if(.not.valid) return
        speed=sqrt(u*u+v*v)
        valid=ieee_is_finite(speed)
        if(.not.valid) return
        ! bendeq-corieq from secrhs.f90:178-179,198.
        equilibrium_target=(speed*capped_inverse_radius-.5d0*coriolis)*h_old
        rate_old=(von_karman*von_karman*alpha)*speed/tau_old
        source_coefficient=equilibrium_target*rate_old
        sink_coefficient=(von_karman*von_karman*alpha)*speed/tau_new
        valid=all(ieee_is_finite([equilibrium_target,source_coefficient, &
                                  sink_coefficient,effective_chezy,alpha]))
        if(.not.valid) then
            equilibrium_target=0.d0;source_coefficient=0.d0;sink_coefficient=0.d0
        endif
    end subroutine

    pure subroutine braids_flow2dh_open_boundary_equilibrium(active,depth_old,depth_new, &
                                                              dryflc,epsilon_value, &
                                                              source_coefficient, &
                                                              sink_coefficient,equilibrium, &
                                                              defined,valid)
        ! Exact secbou relation.  Unlike secrhs, secbou multiplies by the raw
        ! adjacent old/new depths, not by their htrsh-clamped values.
        logical, intent(in) :: active
        real(8), intent(in) :: depth_old,depth_new,dryflc,epsilon_value
        real(8), intent(in) :: source_coefficient,sink_coefficient
        real(8), intent(out) :: equilibrium
        logical, intent(out) :: defined,valid
        real(8) :: denominator,epsh

        equilibrium=0.d0;defined=.false.
        valid=all(ieee_is_finite([depth_old,depth_new,dryflc,epsilon_value, &
                                  source_coefficient,sink_coefficient]))
        if(.not.valid) return
        valid=dryflc>=0.d0.and.epsilon_value>=0.d0
        if(.not.valid.or..not.active) return
        denominator=sink_coefficient*depth_new
        epsh=.5d0*epsilon_value*dryflc
        if(abs(denominator)>epsh) then
            equilibrium=source_coefficient*depth_old/denominator
            valid=ieee_is_finite(equilibrium)
            if(.not.valid) then
                equilibrium=0.d0;return
            endif
            defined=.true.
        endif
    end subroutine

    pure subroutine braids_flow2dh_scalar_bed_rotation(active,u,v,g,raw_chezy,espir, &
                                                        spiral_intensity,bed_x,bed_y, &
                                                        rotated_x,rotated_y, &
                                                        spiral_alpha,valid)
        ! Direction-only eqtran scalar-capacity branch.  It applies to the raw
        ! bed channel only; channel composition and suspended load stay outside.
        logical, intent(in) :: active
        real(8), intent(in) :: u,v,g,raw_chezy,espir,spiral_intensity,bed_x,bed_y
        real(8), intent(out) :: rotated_x,rotated_y,spiral_alpha
        logical, intent(out) :: valid
        real(8) :: speed,magnitude,tx,ty,direction_norm

        rotated_x=0.d0;rotated_y=0.d0;spiral_alpha=0.d0
        valid=all(ieee_is_finite([u,v,g,raw_chezy,espir,spiral_intensity,bed_x,bed_y]))
        if(.not.valid) return
        valid=g>0.d0.and.raw_chezy>0.d0.and.espir>=0.d0
        if(.not.valid.or..not.active) return
        speed=sqrt(u*u+v*v)
        magnitude=sqrt(bed_x*bed_x+bed_y*bed_y)
        valid=all(ieee_is_finite([speed,magnitude]))
        if(.not.valid) return
        if(speed<=0.d0.or.magnitude<=0.d0) return
        spiral_alpha=12.5d0*espir*(1.d0-sqrt(g)/(.4d0*raw_chezy))* &
                     spiral_intensity/speed
        tx=u+spiral_alpha*v
        ty=v-spiral_alpha*u
        direction_norm=sqrt(tx*tx+ty*ty)
        valid=all(ieee_is_finite([spiral_alpha,tx,ty,direction_norm]))
        if(.not.valid) then
            spiral_alpha=0.d0;return
        endif
        if(direction_norm<=epsilon(1.d0)) return
        rotated_x=magnitude*tx/direction_norm
        rotated_y=magnitude*ty/direction_norm
        valid=all(ieee_is_finite([rotated_x,rotated_y]))
        if(.not.valid) then
            rotated_x=0.d0;rotated_y=0.d0;spiral_alpha=0.d0
        endif
    end subroutine

    pure subroutine braids_flow2dh_betac_coefficient(active,u,v,g,raw_chezy,chzmin, &
                                                      von_karman,betac,spiral_intensity, &
                                                      effective_chezy,alpha,betast,beta, &
                                                      stress_xx_basis,stress_xy_basis, &
                                                      enabled,was_clamped,valid)
        ! The returned stress basis omits vihsec's sweep-dependent `fsign` on
        ! RXX.  Applying that geometric sign and taking the stress divergence
        ! are deliberately caller responsibilities.
        logical, intent(in) :: active
        real(8), intent(in) :: u,v,g,raw_chezy,chzmin,von_karman,betac,spiral_intensity
        real(8), intent(out) :: effective_chezy,alpha,betast,beta
        real(8), intent(out) :: stress_xx_basis,stress_xy_basis
        logical, intent(out) :: enabled,was_clamped,valid
        real(8) :: speed,limit

        effective_chezy=0.d0;alpha=0.d0;betast=0.d0;beta=0.d0
        stress_xx_basis=0.d0;stress_xy_basis=0.d0
        enabled=.false.;was_clamped=.false.
        valid=all(ieee_is_finite([u,v,g,raw_chezy,chzmin,von_karman,betac, &
                                  spiral_intensity]))
        if(.not.valid) return
        valid=g>0.d0.and.raw_chezy>0.d0.and.chzmin>0.d0.and.von_karman>0.d0
        if(.not.valid.or..not.active) return
        ! vihsec.f90:137 disables the complete stress branch below this value.
        if(abs(betac)<1.d-4) return
        enabled=.true.
        speed=max(1.d-8,sqrt(u*u+v*v))
        effective_chezy=max(raw_chezy,chzmin)
        alpha=sqrt(g)/(von_karman*effective_chezy)
        betast=betac*(5.d0*alpha-15.6d0*alpha*alpha+37.5d0*alpha*alpha*alpha)
        beta=betast*spiral_intensity/speed
        limit=abs(.1d0*betast)
        if(abs(beta)>limit) then
            ! Preserve FLOW's signed assignment, including its behavior for a
            ! negative intensity or negative Betac.
            beta=.1d0*betast
            was_clamped=.true.
        endif
        stress_xx_basis=2.d0*beta*u*v
        stress_xy_basis=beta*(u*u-v*v)
        valid=all(ieee_is_finite([speed,effective_chezy,alpha,betast,beta, &
                                  stress_xx_basis,stress_xy_basis]))
        if(.not.valid) then
            effective_chezy=0.d0;alpha=0.d0;betast=0.d0;beta=0.d0
            stress_xx_basis=0.d0;stress_xy_basis=0.d0
            enabled=.false.;was_clamped=.false.
        endif
    end subroutine

    pure subroutine braids_flow2dh_secondary_state(active,depth_old,depth_new,u,v, &
                                                    raw_inverse_radius,metric_ksi_sum, &
                                                    metric_eta_sum,coriolis,raw_chezy, &
                                                    spiral_intensity,bed_x,bed_y,dryflc, &
                                                    rmincf,chzmin,g,von_karman,espir, &
                                                    betac,epsilon_value, &
                                                    capped_inverse_radius,cap_limit, &
                                                    equilibrium_target,source_coefficient, &
                                                    sink_coefficient,boundary_equilibrium, &
                                                    rotated_x,rotated_y,spiral_alpha, &
                                                    effective_chezy,relaxation_alpha, &
                                                    betast,beta,stress_xx_basis, &
                                                    stress_xy_basis,was_capped, &
                                                    boundary_defined,betac_enabled, &
                                                    beta_was_clamped,valid)
        logical, intent(in) :: active
        real(8), intent(in) :: depth_old,depth_new,u,v,raw_inverse_radius
        real(8), intent(in) :: metric_ksi_sum,metric_eta_sum,coriolis,raw_chezy
        real(8), intent(in) :: spiral_intensity,bed_x,bed_y,dryflc,rmincf,chzmin
        real(8), intent(in) :: g,von_karman,espir,betac,epsilon_value
        real(8), intent(out) :: capped_inverse_radius,cap_limit,equilibrium_target
        real(8), intent(out) :: source_coefficient,sink_coefficient,boundary_equilibrium
        real(8), intent(out) :: rotated_x,rotated_y,spiral_alpha,effective_chezy
        real(8), intent(out) :: relaxation_alpha,betast,beta,stress_xx_basis,stress_xy_basis
        logical, intent(out) :: was_capped,boundary_defined,betac_enabled
        logical, intent(out) :: beta_was_clamped,valid
        real(8) :: stress_chezy,stress_alpha
        logical :: source_valid,boundary_valid,rotation_valid,stress_valid

        call braids_flow2dh_secondary_source_sink(active,depth_old,depth_new,u,v, &
            raw_inverse_radius,metric_ksi_sum,metric_eta_sum,coriolis,raw_chezy, &
            dryflc,rmincf,chzmin,g,von_karman,capped_inverse_radius,cap_limit, &
            equilibrium_target,source_coefficient,sink_coefficient,effective_chezy, &
            relaxation_alpha,was_capped,source_valid)
        call braids_flow2dh_open_boundary_equilibrium(active,depth_old,depth_new, &
            dryflc,epsilon_value,source_coefficient,sink_coefficient, &
            boundary_equilibrium,boundary_defined,boundary_valid)
        call braids_flow2dh_scalar_bed_rotation(active,u,v,g,raw_chezy,espir, &
            spiral_intensity,bed_x,bed_y,rotated_x,rotated_y,spiral_alpha,rotation_valid)
        call braids_flow2dh_betac_coefficient(active,u,v,g,raw_chezy,chzmin,von_karman, &
            betac,spiral_intensity,stress_chezy,stress_alpha,betast,beta, &
            stress_xx_basis,stress_xy_basis,betac_enabled,beta_was_clamped,stress_valid)
        valid=source_valid.and.boundary_valid.and.rotation_valid.and.stress_valid
        if(valid.and.active.and.betac_enabled) valid=abs(stress_chezy-effective_chezy)<= &
            8.d0*epsilon(1.d0)*max(1.d0,abs(effective_chezy)).and. &
            abs(stress_alpha-relaxation_alpha)<= &
            8.d0*epsilon(1.d0)*max(1.d0,abs(relaxation_alpha))
    end subroutine

end module braids_flow2dh_secondary_operator_module
