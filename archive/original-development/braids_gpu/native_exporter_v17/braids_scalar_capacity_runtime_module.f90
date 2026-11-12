! SPDX-License-Identifier: GPL-3.0-only
!
! Common accepted-state operators for the first Q3 scalar-capacity slice.
!
! The nonlinear capacity is evaluated at cells or hydraulic boundary traces
! before this module's face selector is called.  Keeping that distinction
! explicit prevents a face consumer from silently re-evaluating a different
! formula or substituting owner capacity for a trace-owned boundary.  All
! transport vectors are solid mass per time per width [kg/(m s)].
module braids_scalar_capacity_runtime_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use braids_delft_capacity_module, only: braids_engelund_hansen_capacity, &
        braids_general_capacity
    implicit none
    private

    integer, parameter, public :: braids_capacity_mpm = 2
    integer, parameter, public :: braids_capacity_engelund_hansen = 8
    integer, parameter, public :: braids_capacity_general = 11
    integer, parameter, public :: braids_face_internal = 0
    integer, parameter, public :: braids_face_trace_inlet = 1
    integer, parameter, public :: braids_face_trace_outlet = 2
    integer, parameter, public :: braids_face_owner_outlet = 3

    public :: braids_scalar_capacity_dispatch
    public :: braids_scalar_capacity_face
    public :: braids_scalar_exner_candidate

contains

    pure subroutine braids_scalar_capacity_dispatch(law,depth,u,v,chezy,d50, &
        mixture_d90,solid_density,water_density,gravity,hiding,parameters, &
        bed_mass,suspended_mass,absolute_shields,critical_shields,stress_ratio,valid)
        ! One native CPU ABI for corrected FLOW TraFrm 1/2/4.  ``chezy`` is
        ! already resolved from the selected raw C/n/ks field by the forcing
        ! owner.  In particular, this routine never converts a Chezy or
        ! Nikuradse field to an equivalent Manning value.
        integer, intent(in) :: law
        real(8), intent(in) :: depth,u,v,chezy,d50,mixture_d90
        real(8), intent(in) :: solid_density,water_density,gravity,hiding
        real(8), intent(in) :: parameters(:)
        real(8), intent(out) :: bed_mass(2),suspended_mass(2)
        real(8), intent(out) :: absolute_shields,critical_shields,stress_ratio
        logical, intent(out) :: valid
        real(8), parameter :: mpm_critical_shields=0.047d0
        real(8) :: speed,relative_density,grain_chezy,ripple,threshold
        real(8) :: excess,magnitude

        bed_mass=0.d0;suspended_mass=0.d0
        absolute_shields=0.d0;critical_shields=0.d0;stress_ratio=0.d0
        valid=all(ieee_is_finite([depth,u,v,chezy,d50,mixture_d90, &
            solid_density,water_density,gravity,hiding])).and. &
            all(ieee_is_finite(parameters))
        if(.not.valid) return
        valid=depth>=0.d0.and.d50>0.d0.and.solid_density>water_density.and. &
            water_density>0.d0.and.gravity>0.d0.and.hiding>=0.d0
        if(.not.valid) return
        if(depth==0.d0) then
            valid=u==0.d0.and.v==0.d0
            return
        endif
        if(chezy<=0.d0) then
            valid=.false.;return
        endif
        speed=sqrt(u*u+v*v)
        if(.not.ieee_is_finite(speed)) then
            valid=.false.;return
        endif
        if(speed==0.d0) return
        relative_density=solid_density/water_density-1.d0

        select case(law)
        case(braids_capacity_mpm)
            valid=size(parameters)==1.and.parameters(1)>=0.d0.and. &
                mixture_d90>0.d0.and.hiding>0.d0
            if(.not.valid) return
            absolute_shields=(speed/chezy)**2/(relative_density*d50)
            grain_chezy=18.d0*log10(12.d0*max(depth/mixture_d90,1.d0))
            if(.not.ieee_is_finite(grain_chezy).or.grain_chezy<=0.d0) then
                valid=.false.;return
            endif
            ripple=min((chezy/grain_chezy)**1.5d0,1.d0)
            threshold=hiding*mpm_critical_shields
            excess=max(ripple*absolute_shields-threshold,0.d0)
            magnitude=parameters(1)*8.d0*sqrt(gravity*relative_density*d50**3)* &
                excess**1.5d0
            bed_mass=solid_density*magnitude*[u,v]/speed
            critical_shields=threshold
            stress_ratio=ripple*absolute_shields/threshold
        case(braids_capacity_engelund_hansen)
            valid=size(parameters)==3.and.parameters(1)>=0.d0
            if(.not.valid) return
            call braids_engelund_hansen_capacity(u,v,d50,solid_density, &
                water_density,gravity,chezy,parameters(1),parameters(2), &
                parameters(3),bed_mass,suspended_mass,absolute_shields,valid)
        case(braids_capacity_general)
            valid=size(parameters)==10
            if(.not.valid) return
            call braids_general_capacity(u,v,d50,solid_density,water_density, &
                gravity,chezy,hiding,parameters(1),parameters(2),parameters(3), &
                parameters(4),parameters(5),parameters(6),parameters(7), &
                parameters(8),parameters(9),parameters(10),bed_mass, &
                suspended_mass,absolute_shields,valid)
            critical_shields=hiding*parameters(5)
            if(valid.and.critical_shields>0.d0) then
                stress_ratio=absolute_shields/critical_shields
            else if(valid.and.absolute_shields>0.d0) then
                stress_ratio=huge(1.d0)
            endif
        case default
            valid=.false.
        end select
        valid=valid.and.all(ieee_is_finite(bed_mass)).and. &
            all(ieee_is_finite(suspended_mass)).and. &
            all(ieee_is_finite([absolute_shields,critical_shields,stress_ratio])).and. &
            absolute_shields>=0.d0.and.critical_shields>=0.d0.and.stress_ratio>=0.d0
        if(.not.valid) then
            bed_mass=0.d0;suspended_mass=0.d0
            absolute_shields=0.d0;critical_shields=0.d0;stress_ratio=0.d0
        endif
    end subroutine braids_scalar_capacity_dispatch

    pure subroutine braids_scalar_capacity_face(has_neighbour,role,normal, &
        left_bed,right_bed,owner_bed,trace_bed,owner_availability, &
        neighbour_availability,flux,donor,valid)
        ! Common face policy after reconstruction.  Internal faces follow the
        ! existing BRAIDS sign/upwind/central rule.  Open faces use an explicit
        ! trace-inlet, trace-outlet or owner-outlet role; no implicit owner/
        ! trace substitution is allowed.
        logical, intent(in) :: has_neighbour
        integer, intent(in) :: role
        real(8), intent(in) :: normal(2),left_bed(2),right_bed(2)
        real(8), intent(in) :: owner_bed(2),trace_bed(2)
        real(8), intent(in) :: owner_availability,neighbour_availability
        real(8), intent(out) :: flux
        integer, intent(out) :: donor
        logical, intent(out) :: valid
        real(8) :: left_normal,right_normal,value(2),availability

        flux=0.d0;donor=0
        valid=all(ieee_is_finite([normal,left_bed,right_bed,owner_bed, &
            trace_bed,owner_availability,neighbour_availability])).and. &
            owner_availability>=0.d0.and.neighbour_availability>=0.d0
        if(.not.valid) return
        if(has_neighbour) then
            if(role/=braids_face_internal) then
                valid=.false.;return
            endif
            left_normal=dot_product(left_bed,normal)
            right_normal=dot_product(right_bed,normal)
            if(left_normal>0.d0.and.right_normal>0.d0) then
                value=left_bed;availability=owner_availability;donor=1
            else if(left_normal<0.d0.and.right_normal<0.d0) then
                value=right_bed;availability=neighbour_availability;donor=2
            else
                value=.5d0*(left_bed+right_bed)
                if(dot_product(value,normal)<0.d0) then
                    availability=neighbour_availability;donor=2
                else
                    availability=owner_availability;donor=1
                endif
            endif
            flux=dot_product(value,normal)*availability
        else
            select case(role)
            case(braids_face_trace_inlet)
                flux=min(0.d0,dot_product(trace_bed,normal)*owner_availability)
                donor=3
            case(braids_face_trace_outlet)
                flux=max(0.d0,dot_product(trace_bed,normal)*owner_availability)
                donor=3
            case(braids_face_owner_outlet)
                flux=max(0.d0,dot_product(owner_bed,normal)*owner_availability)
                donor=1
            case(braids_face_internal)
                flux=0.d0;donor=0
            case default
                valid=.false.;return
            end select
        endif
        valid=ieee_is_finite(flux)
        if(.not.valid) then
            flux=0.d0;donor=0
        endif
    end subroutine braids_scalar_capacity_face

    pure subroutine braids_scalar_exner_candidate(old_mass,face_flux,face_length, &
        outward_sign,area,dt,morphological_factor,new_mass,delta_mass,valid)
        ! Candidate-only conservative update.  The caller owns acceptance and
        ! therefore retains an unchanged old state on any invalid candidate.
        real(8), intent(in) :: old_mass,face_flux(:),face_length(:)
        integer, intent(in) :: outward_sign(:)
        real(8), intent(in) :: area,dt,morphological_factor
        real(8), intent(out) :: new_mass,delta_mass
        logical, intent(out) :: valid
        real(8) :: divergence

        new_mass=old_mass;delta_mass=0.d0
        valid=size(face_flux)==size(face_length).and. &
            size(face_flux)==size(outward_sign).and. &
            all(abs(outward_sign)==1).and. &
            all(ieee_is_finite(face_flux)).and.all(ieee_is_finite(face_length)).and. &
            all(face_length>=0.d0).and. &
            all(ieee_is_finite([old_mass,area,dt,morphological_factor])).and. &
            old_mass>=0.d0.and.area>0.d0.and.dt>=0.d0.and. &
            morphological_factor>=0.d0
        if(.not.valid) return
        divergence=sum(real(outward_sign,8)*face_flux*face_length)/area
        delta_mass=-dt*morphological_factor*divergence
        new_mass=old_mass+delta_mass
        valid=ieee_is_finite(new_mass).and.new_mass>=-128.d0*epsilon(1.d0)* &
            max(1.d0,old_mass,abs(delta_mass))
        if(valid) then
            new_mass=max(0.d0,new_mass)
        else
            new_mass=old_mass;delta_mass=0.d0
        endif
    end subroutine braids_scalar_exner_candidate

end module braids_scalar_capacity_runtime_module
