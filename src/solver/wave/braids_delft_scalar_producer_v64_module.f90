! SPDX-License-Identifier: GPL-3.0-only
!
! Exact point producers needed before the six v63 scalar FLOW capacities can
! enter a live CPU route.  This module is additive and is not called by the
! production solver.  It mirrors the locked FLOW 2DH effective-velocity chain
! and the compdiam/compsandfrac point statistics from explicit distribution
! data.  The caller still owns RP_TAUB, settling, generic hiding, boundary
! traces, accepted-state availability and Exner transaction state.
module braids_delft_scalar_producer_v64_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: BRAIDS_SCALAR_PRODUCER_V64_SCHEMA = 1
    real(8), parameter :: FLOW_DGRAVEL = 2.d-3
    ! sediment_basics_module declares the lookup table with default-real
    ! literals before assignment to fp.  Preserve that rounding here.
    real(8), parameter :: FLOW_LOGNORMAL_90 = real(1.2816,8)
    public :: braids_flow2dh_effective_velocity_v64
    public :: braids_flow2dh_mixture_v64

contains

    pure subroutine braids_flow2dh_effective_velocity_v64( &
            dwnvel_observation_depth,physical_depth,u_characteristic, &
            v_characteristic,total_roughness_z0,gravity,vonkar, &
            direction_epsilon,wave,scour,chezy,rp_ustar, &
            rp_ustar_squared,effective_u,effective_v,valid)
        ! FLOW dwnvel and erosed own two different depths.  Dwnvel may raise
        ! its observation depth using adjacent face depths when
        ! MaximumWaterdepth is active, whereas erosed floors the physical
        ! cell depth at 0.01 m before calculating Chezy and RP_DEPTH.  The
        ! characteristic components are already normalized by the former.
        ! This current-only slice rejects wave and scour because the locked
        ! wave/scour branches can modify z0rou, RP_USTAR and RP_TAUB upstream.
        ! It resolves total-current RP_USTAR from z0rou, derives hydraulic
        ! Chezy from the same z0rou, and then constructs RP_EFUMN/RP_EFVMN.
        ! The 0.368 literal is intentionally rounded in default real before
        ! promotion, matching dwnvel.f90's source expression.
        real(8), intent(in) :: dwnvel_observation_depth,physical_depth
        real(8), intent(in) :: u_characteristic,v_characteristic
        real(8), intent(in) :: total_roughness_z0,gravity,vonkar
        real(8), intent(in) :: direction_epsilon
        logical, intent(in) :: wave,scour
        real(8), intent(out) :: chezy,rp_ustar,rp_ustar_squared
        real(8), intent(out) :: effective_u,effective_v
        logical, intent(out) :: valid
        real(8) :: speed,z_velocity,log_velocity,transport_speed,work_depth

        chezy=0.d0
        rp_ustar=0.d0
        rp_ustar_squared=0.d0
        effective_u=0.d0
        effective_v=0.d0
        valid=all(ieee_is_finite([dwnvel_observation_depth,physical_depth, &
            u_characteristic,v_characteristic,total_roughness_z0,gravity, &
            vonkar,direction_epsilon]))
        if(.not.valid) return
        valid=dwnvel_observation_depth>0.d0.and.physical_depth>0.d0.and. &
            total_roughness_z0>0.d0.and.gravity>0.d0.and.vonkar>0.d0.and. &
            direction_epsilon>0.d0.and..not.wave.and..not.scour
        if(.not.valid) return

        speed=sqrt(u_characteristic*u_characteristic+ &
                   v_characteristic*v_characteristic)
        z_velocity=real(0.368,8)*dwnvel_observation_depth
        work_depth=max(.01d0,physical_depth)
        log_velocity=log(1.d0+z_velocity/total_roughness_z0)
        if(.not.ieee_is_finite(speed).or..not.ieee_is_finite(log_velocity).or. &
                log_velocity<=0.d0) then
            valid=.false.
            return
        endif
        chezy=sqrt(gravity)*log(1.d0+work_depth/max(1.d-8, &
            exp(1.d0)*total_roughness_z0))/vonkar
        rp_ustar=speed*vonkar/log_velocity
        rp_ustar_squared=rp_ustar*rp_ustar
        transport_speed=rp_ustar*chezy/sqrt(gravity)
        effective_u=transport_speed*u_characteristic/(speed+direction_epsilon)
        effective_v=transport_speed*v_characteristic/(speed+direction_epsilon)
        valid=all(ieee_is_finite([chezy,rp_ustar,rp_ustar_squared, &
            effective_u,effective_v])).and.min(chezy,rp_ustar, &
            rp_ustar_squared)>=0.d0
        if(.not.valid) then
            chezy=0.d0
            rp_ustar=0.d0
            rp_ustar_squared=0.d0
            effective_u=0.d0
            effective_v=0.d0
        endif
    end subroutine braids_flow2dh_effective_velocity_v64

    pure subroutine braids_flow2dh_mixture_v64(fraction,mean_diameter, &
            median_diameter,sediment_type,max_mud_sedtyp,min_dxx_sedtyp, &
            distribution_count,distribution_probability, &
            distribution_log_diameter,spatial_d50,spatial_d50_value, &
            spatial_log_sigma,dm,dg,d90,dgsd,sand_fraction,mud_fraction, &
            has_exposed_nonmud,valid)
        ! One-point equivalent of locked compdiam + compsandfrac.  The full
        ! within-class percentile tables are explicit: substituting a class
        ! D50 list is not an admitted approximation.  Empty/mud-only surfaces
        ! retain the reference statistics but expose has_exposed_nonmud=false,
        ! so a later capacity owner must not consume the sentinel D90.
        real(8), intent(in) :: fraction(:),mean_diameter(:),median_diameter(:)
        integer, intent(in) :: sediment_type(:),max_mud_sedtyp,min_dxx_sedtyp
        integer, intent(in) :: distribution_count(:)
        real(8), intent(in) :: distribution_probability(:,:)
        real(8), intent(in) :: distribution_log_diameter(:,:)
        logical, intent(in) :: spatial_d50
        real(8), intent(in) :: spatial_d50_value,spatial_log_sigma
        real(8), intent(out) :: dm,dg,d90,dgsd,sand_fraction,mud_fraction
        logical, intent(out) :: has_exposed_nonmud,valid
        integer :: n,l,count_value
        real(8) :: total,fracdiam,fracnonmud,logdg,weight,sum_tolerance

        dm=0.d0
        dg=0.d0
        d90=0.d0
        dgsd=0.d0
        sand_fraction=0.d0
        mud_fraction=0.d0
        has_exposed_nonmud=.false.
        valid=.false.
        n=size(fraction)
        if(n<=0.or.size(mean_diameter)/=n.or.size(median_diameter)/=n.or. &
                size(sediment_type)/=n.or.size(distribution_count)/=n.or. &
                size(distribution_probability,2)/=n.or. &
                size(distribution_log_diameter,2)/=n.or. &
                size(distribution_probability,1)/= &
                    size(distribution_log_diameter,1)) return
        if(.not.all(ieee_is_finite(fraction)).or.any(fraction<0.d0)) return
        if(.not.all(ieee_is_finite(mean_diameter)).or. &
                .not.all(ieee_is_finite(median_diameter))) return
        ! The non-spatial locked FLOW4 erosed/initsedtra call sites pass
        ! sedd50 into both formal arguments of compdiam and compsandfrac.  The
        ! spatial branch instead owns a negative class sentinel plus the
        ! positive point field supplied separately below.
        if(.not.spatial_d50.and.any(mean_diameter/=median_diameter)) return
        if(spatial_d50.and.(mean_diameter(1)>=0.d0.or. &
                median_diameter(1)>=0.d0)) return
        if(max_mud_sedtyp>=min_dxx_sedtyp) return
        do l=1,n
            count_value=distribution_count(l)
            if(count_value<0.or.count_value>size(distribution_probability,1)) return
            if(.not.spatial_d50.and.sediment_type(l)>=min_dxx_sedtyp.and. &
                    median_diameter(l)<=0.d0) return
            if(.not.spatial_d50.and.sediment_type(l)>max_mud_sedtyp.and. &
                    mean_diameter(l)<=0.d0) return
            if(count_value>0) then
                if(.not.all(ieee_is_finite( &
                        distribution_probability(1:count_value,l))).or. &
                        .not.all(ieee_is_finite( &
                        distribution_log_diameter(1:count_value,l)))) return
                if(any(distribution_probability(1:count_value,l)<0.d0).or. &
                        any(distribution_probability(1:count_value,l)>1.d0)) return
                if(count_value>1) then
                    if(any(distribution_probability(2:count_value,l)< &
                           distribution_probability(1:count_value-1,l))) return
                    if(any(distribution_log_diameter(2:count_value,l)<= &
                           distribution_log_diameter(1:count_value-1,l))) return
                endif
            endif
        enddo

        total=sum(fraction)
        sum_tolerance=256.d0*epsilon(1.d0)*max(1.d0,real(n,8))
        ! getfrac is the owner of normalization.  Normalizing again here would
        ! authenticate a different state and could hide corrupt/depleted input.
        if(abs(total-1.d0)>sum_tolerance) return
        mud_fraction=sum(fraction,mask=sediment_type<=max_mud_sedtyp)
        fracnonmud=sum(fraction,mask=sediment_type>max_mud_sedtyp)
        has_exposed_nonmud=fracnonmud>0.d0

        if(spatial_d50) then
            if(.not.ieee_is_finite(spatial_d50_value).or. &
                    .not.ieee_is_finite(spatial_log_sigma).or. &
                    spatial_d50_value<=0.d0.or.spatial_log_sigma<0.d0) return
            if(sediment_type(1)<min_dxx_sedtyp) then
                dm=0.d0
                dg=0.d0
                d90=0.d0
                dgsd=0.d0
            else
                dm=spatial_d50_value*exp(.5d0*spatial_log_sigma**2)
                dg=spatial_d50_value
                d90=spatial_d50_value* &
                    exp(spatial_log_sigma)**FLOW_LOGNORMAL_90
                dgsd=0.d0
            endif
            if(sediment_type(1)>max_mud_sedtyp.and. &
                    spatial_d50_value>=64.d-6.and. &
                    spatial_d50_value<FLOW_DGRAVEL) sand_fraction=1.d0
            valid=all(ieee_is_finite([dm,dg,d90,dgsd,sand_fraction, &
                mud_fraction]))
            return
        endif

        fracdiam=sum(fraction,mask=sediment_type>=min_dxx_sedtyp)
        if(fracdiam>0.d0) then
            do l=1,n
                if(sediment_type(l)>=min_dxx_sedtyp) then
                    weight=fraction(l)/fracdiam
                    dm=dm+weight*mean_diameter(l)
                    dg=dg+weight*log(median_diameter(l))
                endif
            enddo
            logdg=dg
            dg=exp(logdg)
            dgsd=0.d0
            do l=1,n
                if(sediment_type(l)>=min_dxx_sedtyp.and.fraction(l)>0.d0) then
                    weight=fraction(l)/fracdiam
                    dgsd=dgsd+weight*(log(median_diameter(l))-logdg)**2
                endif
            enddo
            dgsd=exp(sqrt(dgsd))
        else
            dm=0.d0
            dg=0.d0
            dgsd=0.d0
        endif

        if(fracnonmud>0.d0) sand_fraction=sum(fraction, &
            mask=sediment_type>max_mud_sedtyp.and. &
                 mean_diameter<FLOW_DGRAVEL)/fracnonmud
        call mixture_percentile_v64(.9d0,fraction,sediment_type, &
            min_dxx_sedtyp,distribution_count,distribution_probability, &
            distribution_log_diameter,d90,valid)
        valid=valid.and.all(ieee_is_finite([dm,dg,d90,dgsd,sand_fraction, &
            mud_fraction])).and.min(dm,dg,d90,dgsd,sand_fraction, &
            mud_fraction)>=0.d0.and.sand_fraction<=1.d0.and. &
            mud_fraction<=1.d0
        if(.not.valid) then
            dm=0.d0
            dg=0.d0
            d90=0.d0
            dgsd=0.d0
            sand_fraction=0.d0
            mud_fraction=0.d0
            has_exposed_nonmud=.false.
        endif
    end subroutine braids_flow2dh_mixture_v64

    pure subroutine mixture_percentile_v64(requested,fraction,sediment_type, &
            min_dxx_sedtyp,distribution_count,distribution_probability, &
            distribution_log_diameter,diameter,valid)
        real(8), intent(in) :: requested,fraction(:)
        integer, intent(in) :: sediment_type(:),min_dxx_sedtyp
        integer, intent(in) :: distribution_count(:)
        real(8), intent(in) :: distribution_probability(:,:)
        real(8), intent(in) :: distribution_log_diameter(:,:)
        real(8), intent(out) :: diameter
        logical, intent(out) :: valid
        integer :: stage(size(fraction)),l,s,trigger,iterations,max_iterations
        real(8) :: fracdiam,fracfac,density,logdiam,logprevious
        real(8) :: accumulated,interval,remaining

        diameter=0.d0
        valid=requested>=0.d0.and.requested<=1.d0
        if(.not.valid) return
        fracdiam=sum(fraction,mask=sediment_type>=min_dxx_sedtyp)
        stage=0
        logprevious=0.d0
        accumulated=0.d0
        iterations=0
        max_iterations=sum(max(distribution_count,0))+size(fraction)+1
        do while(requested>accumulated)
            iterations=iterations+1
            if(iterations>max_iterations) then
                valid=.false.
                return
            endif
            logdiam=999.999d0
            trigger=0
            density=0.d0
            do l=1,size(fraction)
                s=stage(l)
                if(s<distribution_count(l)) then
                    if(s>0) then
                        if(fracdiam>0.d0) then
                            fracfac=fraction(l)/fracdiam
                        else
                            fracfac=1.d0
                        endif
                        density=density+fracfac* &
                            (distribution_probability(s+1,l)- &
                             distribution_probability(s,l))/ &
                            (distribution_log_diameter(s+1,l)- &
                             distribution_log_diameter(s,l))
                    endif
                    if(distribution_log_diameter(s+1,l)<logdiam) then
                        logdiam=distribution_log_diameter(s+1,l)
                        trigger=l
                    endif
                endif
            enddo
            if(trigger==0) then
                diameter=exp(logprevious)
                valid=ieee_is_finite(diameter)
                return
            endif
            interval=density*(logdiam-logprevious)
            remaining=requested-accumulated
            if(interval>=remaining.and.interval>0.d0) then
                logprevious=logprevious+(logdiam-logprevious)*remaining/interval
                diameter=exp(logprevious)
                valid=ieee_is_finite(diameter)
                return
            endif
            accumulated=accumulated+interval
            logprevious=logdiam
            if(stage(trigger)==0) then
                if(distribution_log_diameter(distribution_count(trigger),trigger)<= &
                        distribution_log_diameter(1,trigger)) then
                    stage(trigger)=max(5,distribution_count(trigger))
                    if(fraction(trigger)>requested-accumulated) then
                        diameter=exp(logdiam)
                        valid=ieee_is_finite(diameter)
                        return
                    endif
                    accumulated=accumulated+fraction(trigger)
                else
                    stage(trigger)=1
                endif
            else
                stage(trigger)=stage(trigger)+1
            endif
        enddo
        diameter=exp(logprevious)
        valid=ieee_is_finite(diameter)
    end subroutine mixture_percentile_v64

end module braids_delft_scalar_producer_v64_module
