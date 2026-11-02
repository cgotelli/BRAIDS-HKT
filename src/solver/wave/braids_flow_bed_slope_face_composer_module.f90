! SPDX-License-Identifier: AGPL-3.0-or-later
!
! Backend-neutral scalar face composer for the authenticated Q3-H4 classic
! Delft3D4 FLOW bed-slope direction methods.  The point formula lives in
! braids_flow_bed_slope_operator_module; this module owns the face roles,
! candidate selection, post-slope donor choice, one availability application,
! and signed open-boundary projection immediately upstream of Exner.
module braids_flow_bed_slope_face_composer_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use braids_flow_bed_slope_operator_module, only: &
        braids_flow_bed_slope_direction,flow_slope_ok,flow_slope_invalid_input
    implicit none
    private

    integer, parameter, public :: flow_face_internal = 0
    integer, parameter, public :: flow_face_trace_inlet = 1
    integer, parameter, public :: flow_face_trace_outlet = 2
    integer, parameter, public :: flow_face_owner_outlet = 3

    integer, parameter, public :: flow_donor_none = 0
    integer, parameter, public :: flow_donor_owner = 1
    integer, parameter, public :: flow_donor_neighbour = 2
    integer, parameter, public :: flow_donor_trace = 3

    integer, parameter, public :: flow_face_ok = 0
    integer, parameter, public :: flow_face_invalid_topology = 10
    integer, parameter, public :: flow_face_invalid_input = 11

    public :: braids_flow_bed_slope_face

contains

    pure subroutine braids_flow_bed_slope_face( &
            islope,role,has_neighbour,left_q,right_q,owner_q,trace_q, &
            normal,downhill,stress,depth,d50,mean_diameter, &
            hiding_left,hiding_right,owner_availability, &
            neighbour_availability,alfabs,alfabn,ashld,bshld,cshld,dshld, &
            alfpa,thcrpa,raw_q,corrected_q,flux,raw_donor,donor, &
            class_status,face_status)
        integer, intent(in) :: islope,role
        logical, intent(in) :: has_neighbour
        real(8), intent(in) :: left_q(2),right_q(2),owner_q(2),trace_q(2)
        real(8), intent(in) :: normal(2),downhill(2)
        real(8), intent(in) :: stress,depth,d50,mean_diameter
        real(8), intent(in) :: hiding_left,hiding_right
        real(8), intent(in) :: owner_availability,neighbour_availability
        real(8), intent(in) :: alfabs,alfabn,ashld,bshld,cshld,dshld
        real(8), intent(in) :: alfpa,thcrpa
        real(8), intent(out) :: raw_q(2),corrected_q(2),flux
        integer, intent(out) :: raw_donor,donor,class_status,face_status
        real(8) :: left_normal,right_normal,corrected_normal,availability
        real(8) :: safe_downhill(2),safe_stress,safe_depth,safe_mean
        real(8) :: safe_hiding_left,safe_hiding_right,magnitude
        logical :: participating

        raw_q=0.d0
        corrected_q=0.d0
        flux=0.d0
        raw_donor=flow_donor_none
        donor=flow_donor_none
        class_status=flow_slope_ok
        face_status=flow_face_ok

        if(role<flow_face_internal.or.role>flow_face_owner_outlet.or. &
           (role/=flow_face_internal.and.has_neighbour)) then
            class_status=flow_slope_invalid_input
            face_status=flow_face_invalid_topology
            return
        endif
        if(.not.all(ieee_is_finite([normal,owner_availability, &
                                     neighbour_availability])).or. &
           min(owner_availability,neighbour_availability)<0.d0) then
            class_status=flow_slope_invalid_input
            face_status=flow_face_invalid_input
            return
        endif

        participating=.false.
        select case(role)
        case(flow_face_internal)
            if(has_neighbour) then
                participating=.true.
                left_normal=dot_product(left_q,normal)
                right_normal=dot_product(right_q,normal)
                if(left_normal>0.d0.and.right_normal>0.d0) then
                    raw_q=left_q
                    raw_donor=flow_donor_owner
                else if(left_normal<0.d0.and.right_normal<0.d0) then
                    raw_q=right_q
                    raw_donor=flow_donor_neighbour
                else
                    raw_q=.5d0*(left_q+right_q)
                    if(dot_product(raw_q,normal)<0.d0) then
                        raw_donor=flow_donor_neighbour
                    else
                        raw_donor=flow_donor_owner
                    endif
                endif
            endif
        case(flow_face_trace_inlet,flow_face_trace_outlet)
            participating=.true.
            raw_q=trace_q
            raw_donor=flow_donor_trace
        case(flow_face_owner_outlet)
            participating=.true.
            raw_q=owner_q
            raw_donor=flow_donor_owner
        end select

        ! FLOW does not consume hydraulic/material auxiliaries for a class
        ! whose selected magnitude is at or below its direction epsilon.
        ! Neutralising those unused values also makes a closed face exact zero.
        corrected_q=raw_q
        if(participating) then
            magnitude=sqrt(raw_q(1)*raw_q(1)+raw_q(2)*raw_q(2))
            if(.not.ieee_is_finite(magnitude)) then
                class_status=flow_slope_invalid_input
            else
                if(magnitude>1.d-6) then
                    safe_downhill=downhill
                    safe_stress=stress
                    safe_depth=depth
                    safe_mean=mean_diameter
                    safe_hiding_left=hiding_left
                    safe_hiding_right=hiding_right
                else
                    safe_downhill=0.d0
                    safe_stress=0.d0
                    safe_depth=1.d0
                    safe_mean=1.d0
                    safe_hiding_left=1.d0
                    safe_hiding_right=1.d0
                endif
                call braids_flow_bed_slope_direction( &
                    islope,raw_q(1),raw_q(2),safe_downhill(1), &
                    safe_downhill(2),safe_stress,safe_depth,d50,safe_mean, &
                    safe_hiding_left,safe_hiding_right,alfabs,alfabn,ashld, &
                    bshld,cshld,dshld,alfpa,thcrpa,corrected_q(1), &
                    corrected_q(2),class_status)
            endif
        endif
        if(class_status/=flow_slope_ok) then
            donor=flow_donor_none
            flux=0.d0
            face_status=class_status
            return
        endif

        corrected_normal=dot_product(corrected_q,normal)
        select case(role)
        case(flow_face_internal)
            if(.not.has_neighbour) return
            if(corrected_normal<0.d0) then
                donor=flow_donor_neighbour
                availability=neighbour_availability
            else
                donor=flow_donor_owner
                availability=owner_availability
            endif
            flux=corrected_normal*availability
        case(flow_face_trace_inlet)
            donor=flow_donor_trace
            flux=min(0.d0,corrected_normal*owner_availability)
        case(flow_face_trace_outlet)
            donor=flow_donor_trace
            flux=max(0.d0,corrected_normal*owner_availability)
        case(flow_face_owner_outlet)
            donor=flow_donor_owner
            flux=max(0.d0,corrected_normal*owner_availability)
        end select
    end subroutine braids_flow_bed_slope_face

end module braids_flow_bed_slope_face_composer_module
