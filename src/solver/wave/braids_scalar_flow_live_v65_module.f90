! SPDX-License-Identifier: AGPL-3.0-or-later
!
! Native OpenMP reference for the bounded Q3-H5/v65 fixed-grid live seam.
! The six constitutive laws remain owned by the authenticated v64 producer;
! this module starts with its post-channel, post-secondary mass vectors.
module braids_scalar_flow_live_v65_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use omp_lib, only: omp_get_num_threads
    use braids_flow_bed_slope_operator_module, only: &
        braids_flow_bed_slope_direction,flow_slope_ok
    implicit none
    private

    integer, parameter, public :: scalar_live_face_internal = 0
    integer, parameter, public :: scalar_live_face_trace_inlet = 1
    integer, parameter, public :: scalar_live_face_trace_outlet = 2
    integer, parameter, public :: scalar_live_face_owner_outlet = 3
    integer, parameter, public :: scalar_live_face_closed = 4

    integer, parameter, public :: scalar_live_donor_none = 0
    integer, parameter, public :: scalar_live_donor_owner = 1
    integer, parameter, public :: scalar_live_donor_neighbour = 2
    integer, parameter, public :: scalar_live_donor_trace = 3

    integer, parameter, public :: scalar_live_ok = 0
    integer, parameter, public :: scalar_live_invalid_shape = 10
    integer, parameter, public :: scalar_live_invalid_topology = 11
    integer, parameter, public :: scalar_live_invalid_input = 12
    integer, parameter, public :: scalar_live_invalid_stencil = 13
    integer, parameter, public :: scalar_live_slope_failure = 14
    integer, parameter, public :: scalar_live_depleted_cell = 15
    integer, parameter, public :: scalar_live_inventory_failure = 16
    integer, parameter, public :: scalar_live_conservation_failure = 17

    type, public :: scalar_flow_projection_topology_v65
        ! Components are private so a hot-step receipt can only be created by
        ! the validating preparation routine.  The receipt owns immutable-in-
        ! practice copies of the fixed-grid catalog and geometry.
        private
        integer :: ncell=0,nface=0,ninc=0,nboundary=0
        logical :: prepared=.false.
        integer, allocatable :: owner(:),neighbour(:),role(:)
        integer, allocatable :: cell_face_offsets(:),cell_faces(:)
        integer, allocatable :: boundary_faces(:)
        real(8), allocatable :: normal(:,:),length_m(:)
    end type scalar_flow_projection_topology_v65

    public :: braids_scalar_flow_face_candidates_v65
    public :: braids_prepare_scalar_flow_topology_v65
    public :: braids_scalar_flow_project_exner_v65

contains

    subroutine braids_prepare_scalar_flow_topology_v65( &
            ncell,nface,owner,neighbour,role,normal,length_m,ninc, &
            cell_face_offsets,cell_faces,nboundary,boundary_faces, &
            topology,status)
        integer, intent(in) :: ncell,nface,ninc,nboundary
        integer, intent(in) :: owner(nface),neighbour(nface),role(nface)
        real(8), intent(in) :: normal(2,nface),length_m(nface)
        integer, intent(in) :: cell_face_offsets(ncell+1),cell_faces(ninc)
        integer, intent(in) :: boundary_faces(nboundary)
        type(scalar_flow_projection_topology_v65), intent(out) :: topology
        integer, intent(out) :: status
        integer, allocatable :: owner_seen(:),neighbour_seen(:),boundary_seen(:)
        integer :: cell,face,item,start,finish,previous,own,other,face_role

        status=scalar_live_ok
        if(ncell<=0.or.nface<=0.or.ninc<=0.or.nboundary<0) then
            status=scalar_live_invalid_shape
            return
        endif
        if(.not.all(ieee_is_finite(normal)).or. &
           .not.all(ieee_is_finite(length_m)).or.any(length_m<=0.d0)) then
            status=scalar_live_invalid_input
            return
        endif
        do face=1,nface
            if(abs(sum(normal(:,face)*normal(:,face))-1.d0)> &
               1024.d0*epsilon(1.d0)) then
                status=scalar_live_invalid_input
                return
            endif
            own=owner(face)
            other=neighbour(face)
            face_role=role(face)
            if(own<1.or.own>ncell.or.face_role<scalar_live_face_internal.or. &
               face_role>scalar_live_face_closed) then
                status=scalar_live_invalid_topology
                return
            endif
            if(face_role==scalar_live_face_internal) then
                if(other<1.or.other>ncell.or.other==own) then
                    status=scalar_live_invalid_topology
                    return
                endif
            else if(other/=0) then
                status=scalar_live_invalid_topology
                return
            endif
        enddo
        if(cell_face_offsets(1)/=1.or. &
           cell_face_offsets(ncell+1)/=ninc+1) then
            status=scalar_live_invalid_topology
            return
        endif
        allocate(owner_seen(nface),neighbour_seen(nface),boundary_seen(nface))
        owner_seen=0;neighbour_seen=0;boundary_seen=0
        do cell=1,ncell
            start=cell_face_offsets(cell)
            finish=cell_face_offsets(cell+1)-1
            if(start<1.or.finish<start.or.finish>ninc) then
                status=scalar_live_invalid_topology
                deallocate(owner_seen,neighbour_seen,boundary_seen)
                return
            endif
            previous=0
            do item=start,finish
                face=cell_faces(item)
                if(face<1.or.face>nface.or.face<=previous) then
                    status=scalar_live_invalid_topology
                    deallocate(owner_seen,neighbour_seen,boundary_seen)
                    return
                endif
                previous=face
                if(owner(face)==cell) then
                    owner_seen(face)=owner_seen(face)+1
                else if(role(face)==scalar_live_face_internal.and. &
                        neighbour(face)==cell) then
                    neighbour_seen(face)=neighbour_seen(face)+1
                else
                    status=scalar_live_invalid_topology
                    deallocate(owner_seen,neighbour_seen,boundary_seen)
                    return
                endif
            enddo
        enddo
        do face=1,nface
            if(owner_seen(face)/=1.or. &
               (role(face)==scalar_live_face_internal.and. &
                neighbour_seen(face)/=1).or. &
               (role(face)/=scalar_live_face_internal.and. &
                neighbour_seen(face)/=0)) then
                status=scalar_live_invalid_topology
                deallocate(owner_seen,neighbour_seen,boundary_seen)
                return
            endif
        enddo
        previous=0
        do item=1,nboundary
            face=boundary_faces(item)
            if(face<1.or.face>nface.or.face<=previous.or. &
               role(face)==scalar_live_face_internal) then
                status=scalar_live_invalid_topology
                deallocate(owner_seen,neighbour_seen,boundary_seen)
                return
            endif
            previous=face
            boundary_seen(face)=boundary_seen(face)+1
        enddo
        do face=1,nface
            if((role(face)==scalar_live_face_internal.and. &
                boundary_seen(face)/=0).or. &
               (role(face)/=scalar_live_face_internal.and. &
                boundary_seen(face)/=1)) then
                status=scalar_live_invalid_topology
                deallocate(owner_seen,neighbour_seen,boundary_seen)
                return
            endif
        enddo
        deallocate(owner_seen,neighbour_seen,boundary_seen)
        topology%ncell=ncell;topology%nface=nface;topology%ninc=ninc
        topology%nboundary=nboundary
        allocate(topology%owner(nface),topology%neighbour(nface), &
                 topology%role(nface),topology%normal(2,nface), &
                 topology%length_m(nface),topology%cell_face_offsets(ncell+1), &
                 topology%cell_faces(ninc),topology%boundary_faces(nboundary))
        topology%owner=owner;topology%neighbour=neighbour;topology%role=role
        topology%normal=normal;topology%length_m=length_m
        topology%cell_face_offsets=cell_face_offsets
        topology%cell_faces=cell_faces;topology%boundary_faces=boundary_faces
        topology%prepared=.true.
    end subroutine braids_prepare_scalar_flow_topology_v65

    pure logical function locked_stencil_component_v65( &
            ncell,nnz,offsets,indices,weights)
        integer, intent(in) :: ncell,nnz
        integer, intent(in) :: offsets(ncell+1),indices(nnz)
        real(8), intent(in) :: weights(nnz)
        integer :: cell,item,start,finish,self_count,neighbour_count,other

        locked_stencil_component_v65=.false.
        if(nnz<ncell.or.nnz>3*ncell) return
        if(offsets(1)/=1.or.offsets(ncell+1)/=nnz+1) return
        if(.not.all(ieee_is_finite(weights))) return
        do cell=1,ncell
            start=offsets(cell)
            finish=offsets(cell+1)-1
            if(start<1.or.finish<start.or.finish>nnz.or.finish-start+1>3) return
            self_count=0
            neighbour_count=0
            do item=start,finish
                if(indices(item)<1.or.indices(item)>ncell) return
                if(indices(item)==cell) then
                    self_count=self_count+1
                else
                    neighbour_count=neighbour_count+1
                    if(weights(item)/=0.25d0) return
                endif
                do other=start,item-1
                    if(indices(other)==indices(item)) return
                enddo
            enddo
            if(self_count/=1.or.neighbour_count>2) return
            do item=start,finish
                if(indices(item)==cell) then
                    if(weights(item)/=1.d0-0.25d0*real(neighbour_count,8)) return
                endif
            enddo
        enddo
        locked_stencil_component_v65=.true.
    end function locked_stencil_component_v65


    subroutine braids_scalar_flow_face_candidates_v65( &
            ncell,nface,nfrac,islope,upwind_bedload,lateral_average, &
            owner,neighbour,role,normal,cell_bed,total_direction,trace_bed, &
            nnzx,x_offsets,x_indices,x_weights, &
            nnzy,y_offsets,y_indices,y_weights, &
            downhill,total_current_ust2,gravity,solid_density,water_density, &
            depth,d50,mean_diameter,hiding, &
            alfabs,alfabn,ashld,bshld,cshld,dshld,alfpa,thcrpa, &
            raw_q,corrected_q,raw_donor,donor,donor_cell,status, &
            observed_threads)
        ! The CSR convention is one-based: offsets(1)=1 and the final offset
        ! equals nnz+1.  x rows are +/-icy candidates for su; y rows are
        ! +/-icx candidates for sv.  Only the locked 4-2-1 quarter stencil is
        ! admitted, never a generic convex smoother.
        integer, intent(in) :: ncell,nface,nfrac,islope,nnzx,nnzy
        logical, intent(in) :: upwind_bedload,lateral_average
        integer, intent(in) :: owner(nface),neighbour(nface),role(nface)
        real(8), intent(in) :: normal(2,nface)
        real(8), intent(in) :: cell_bed(2,nfrac,ncell)
        real(8), intent(in) :: total_direction(2,nfrac,ncell)
        real(8), intent(in) :: trace_bed(2,nfrac,nface)
        integer, intent(in) :: x_offsets(ncell+1),x_indices(nnzx)
        real(8), intent(in) :: x_weights(nnzx)
        integer, intent(in) :: y_offsets(ncell+1),y_indices(nnzy)
        real(8), intent(in) :: y_weights(nnzy)
        real(8), intent(in) :: downhill(2,nface),total_current_ust2(nface)
        real(8), intent(in) :: gravity,solid_density(nfrac),water_density(nface)
        real(8), intent(in) :: depth(nfrac,nface),d50(nfrac,nface)
        real(8), intent(in) :: mean_diameter(nfrac,nface)
        real(8), intent(in) :: hiding(nfrac,ncell)
        real(8), intent(in) :: alfabs,alfabn,ashld,bshld,cshld,dshld
        real(8), intent(in) :: alfpa,thcrpa
        real(8), intent(out) :: raw_q(2,nfrac,nface)
        real(8), intent(out) :: corrected_q(2,nfrac,nface)
        integer, intent(out) :: raw_donor(nfrac,nface),donor(nfrac,nface)
        integer, intent(out) :: donor_cell(nfrac,nface)
        integer, intent(out) :: status,observed_threads
        real(8), allocatable :: work_bed(:,:,:)
        integer, allocatable :: local_status(:,:)
        integer :: cell,face,fraction,item,start,finish,slope_status
        integer :: own,other,face_role
        real(8) :: q(2),corrected(2),left_sign,right_sign,normal_value
        real(8) :: relative_density,shields,hiding_left,hiding_right
        real(8) :: normal_length2

        raw_q=0.d0
        corrected_q=0.d0
        raw_donor=scalar_live_donor_none
        donor=scalar_live_donor_none
        donor_cell=0
        status=scalar_live_ok
        observed_threads=1
        if(ncell<=0.or.nface<=0.or.nfrac<=0.or.nnzx<=0.or.nnzy<=0) then
            status=scalar_live_invalid_shape
            return
        endif
        if(islope/=0.and.islope/=2.and.islope/=3.and.islope/=4) then
            status=scalar_live_invalid_input
            return
        endif
        if(.not.all(ieee_is_finite(normal)).or. &
           .not.all(ieee_is_finite(cell_bed)).or. &
           .not.all(ieee_is_finite(total_direction)).or. &
           .not.all(ieee_is_finite(trace_bed)).or. &
           .not.all(ieee_is_finite([alfabs,alfabn,ashld,bshld,cshld,dshld, &
                                     alfpa,thcrpa]))) then
            status=scalar_live_invalid_input
            return
        endif
        do face=1,nface
            normal_length2=sum(normal(:,face)*normal(:,face))
            if(abs(normal_length2-1.d0)>1024.d0*epsilon(1.d0)) then
                status=scalar_live_invalid_input
                return
            endif
            own=owner(face)
            other=neighbour(face)
            face_role=role(face)
            if(own<1.or.own>ncell.or.face_role<scalar_live_face_internal.or. &
               face_role>scalar_live_face_closed) then
                status=scalar_live_invalid_topology
                return
            endif
            if(face_role==scalar_live_face_internal) then
                if(other<1.or.other>ncell.or.other==own) then
                    status=scalar_live_invalid_topology
                    return
                endif
            else if(other/=0) then
                status=scalar_live_invalid_topology
                return
            endif
            if(face_role/=scalar_live_face_trace_inlet.and. &
               face_role/=scalar_live_face_trace_outlet) then
                if(any(trace_bed(:,:,face)/=0.d0)) then
                    status=scalar_live_invalid_input
                    return
                endif
            endif
        enddo
        if(lateral_average) then
            if(.not.locked_stencil_component_v65( &
                    ncell,nnzx,x_offsets,x_indices,x_weights).or. &
               .not.locked_stencil_component_v65( &
                    ncell,nnzy,y_offsets,y_indices,y_weights)) then
                status=scalar_live_invalid_stencil
                return
            endif
        endif
        if(islope/=0) then
            if(.not.ieee_is_finite(gravity).or.gravity<=0.d0.or. &
               .not.all(ieee_is_finite(downhill)).or. &
               .not.all(ieee_is_finite(total_current_ust2)).or. &
               .not.all(ieee_is_finite(solid_density)).or. &
               .not.all(ieee_is_finite(water_density)).or. &
               .not.all(ieee_is_finite(depth)).or. &
               .not.all(ieee_is_finite(d50)).or. &
               .not.all(ieee_is_finite(mean_diameter)).or. &
               .not.all(ieee_is_finite(hiding)).or. &
               any(total_current_ust2<0.d0).or.any(water_density<=0.d0).or. &
               any(depth<=0.d0).or.any(d50<=0.d0).or. &
               any(mean_diameter<=0.d0).or.any(hiding<0.d0)) then
                status=scalar_live_invalid_input
                return
            endif
            do face=1,nface
                do fraction=1,nfrac
                    if(solid_density(fraction)<=water_density(face)) then
                        status=scalar_live_invalid_input
                        return
                    endif
                enddo
            enddo
        endif

        allocate(work_bed(2,nfrac,ncell),local_status(nfrac,nface))
        work_bed=cell_bed
        local_status=scalar_live_ok
        ! One outer team owns all independent loops.  This avoids nested teams
        ! and does not hide an unmeasured serial/parallel workload threshold.
        !$omp parallel default(shared) private(cell,face,fraction,item,start, &
        !$omp& finish,own,other,face_role,q,corrected,left_sign,right_sign, &
        !$omp& normal_value,relative_density,shields,hiding_left,hiding_right, &
        !$omp& slope_status)
        !$omp single
        observed_threads=omp_get_num_threads()
        !$omp end single
        if(lateral_average) then
            !$omp do collapse(2) schedule(static,256)
            do cell=1,ncell
                do fraction=1,nfrac
                    work_bed(1,fraction,cell)=0.d0
                    start=x_offsets(cell)
                    finish=x_offsets(cell+1)-1
                    do item=start,finish
                        work_bed(1,fraction,cell)=work_bed(1,fraction,cell)+ &
                            x_weights(item)*cell_bed(1,fraction,x_indices(item))
                    enddo
                    work_bed(2,fraction,cell)=0.d0
                    start=y_offsets(cell)
                    finish=y_offsets(cell+1)-1
                    do item=start,finish
                        work_bed(2,fraction,cell)=work_bed(2,fraction,cell)+ &
                            y_weights(item)*cell_bed(2,fraction,y_indices(item))
                    enddo
                enddo
            enddo
            !$omp end do
        endif
        !$omp do collapse(2) schedule(static,256)
        do face=1,nface
            do fraction=1,nfrac
                own=owner(face)
                other=neighbour(face)
                face_role=role(face)
                q=0.d0
                select case(face_role)
                case(scalar_live_face_internal)
                    if(upwind_bedload) then
                        left_sign=dot_product( &
                            total_direction(:,fraction,own),normal(:,face))
                        right_sign=dot_product( &
                            total_direction(:,fraction,other),normal(:,face))
                        if(left_sign>0.d0.and.right_sign>0.d0) then
                            q=work_bed(:,fraction,own)
                            raw_donor(fraction,face)=scalar_live_donor_owner
                        else if(left_sign<0.d0.and.right_sign<0.d0) then
                            q=work_bed(:,fraction,other)
                            raw_donor(fraction,face)=scalar_live_donor_neighbour
                        else
                            q=0.5d0*(work_bed(:,fraction,own)+ &
                                     work_bed(:,fraction,other))
                            if(dot_product(q,normal(:,face))<0.d0) then
                                raw_donor(fraction,face)=scalar_live_donor_neighbour
                            else
                                raw_donor(fraction,face)=scalar_live_donor_owner
                            endif
                        endif
                    else
                        q=0.5d0*(work_bed(:,fraction,own)+ &
                                 work_bed(:,fraction,other))
                        if(dot_product(q,normal(:,face))<0.d0) then
                            raw_donor(fraction,face)=scalar_live_donor_neighbour
                        else
                            raw_donor(fraction,face)=scalar_live_donor_owner
                        endif
                    endif
                case(scalar_live_face_trace_inlet,scalar_live_face_trace_outlet)
                    q=trace_bed(:,fraction,face)
                    raw_donor(fraction,face)=scalar_live_donor_trace
                case(scalar_live_face_owner_outlet)
                    q=work_bed(:,fraction,own)
                    raw_donor(fraction,face)=scalar_live_donor_owner
                case(scalar_live_face_closed)
                    q=0.d0
                    raw_donor(fraction,face)=scalar_live_donor_none
                end select
                raw_q(:,fraction,face)=q
                corrected=q
                if(islope/=0.and.face_role/=scalar_live_face_closed) then
                    if(islope==2) then
                        ! Locked classic FLOW erosed initializes taurat to zero
                        ! before eqtran.  Positive TraFrm leaves its INTENT(OUT)
                        ! unassigned (a nonportable source defect); gfortran
                        ! 13.2 preserves the caller bytes at O0/O2/O3, while
                        ! locked ifx bytes are not externally observable.
                        ! BRAIDS binds the caller intent to exact zero.  Never
                        ! substitute RP_USTAR, RP_TAUB or a capacity diagnostic.
                        shields=0.d0
                    else
                        relative_density=(solid_density(fraction)- &
                                          water_density(face))/water_density(face)
                        shields=total_current_ust2(face)/( &
                            gravity*relative_density*d50(fraction,face))
                    endif
                    hiding_left=hiding(fraction,own)
                    if(face_role==scalar_live_face_internal) then
                        hiding_right=hiding(fraction,other)
                    else
                        hiding_right=hiding_left
                    endif
                    call braids_flow_bed_slope_direction( &
                        islope,q(1),q(2),downhill(1,face),downhill(2,face), &
                        shields,depth(fraction,face),d50(fraction,face), &
                        mean_diameter(fraction,face),hiding_left,hiding_right, &
                        alfabs,alfabn,ashld,bshld,cshld,dshld,alfpa,thcrpa, &
                        corrected(1),corrected(2),slope_status)
                    if(slope_status/=flow_slope_ok) then
                        local_status(fraction,face)=scalar_live_slope_failure
                        corrected=q
                    endif
                endif
                corrected_q(:,fraction,face)=corrected
                normal_value=dot_product(corrected,normal(:,face))
                select case(face_role)
                case(scalar_live_face_internal)
                    if(normal_value<0.d0) then
                        donor(fraction,face)=scalar_live_donor_neighbour
                        donor_cell(fraction,face)=other
                    else
                        donor(fraction,face)=scalar_live_donor_owner
                        donor_cell(fraction,face)=own
                    endif
                case(scalar_live_face_trace_inlet,scalar_live_face_trace_outlet)
                    donor(fraction,face)=scalar_live_donor_trace
                    donor_cell(fraction,face)=own
                case(scalar_live_face_owner_outlet)
                    donor(fraction,face)=scalar_live_donor_owner
                    donor_cell(fraction,face)=own
                case(scalar_live_face_closed)
                    donor(fraction,face)=scalar_live_donor_none
                    donor_cell(fraction,face)=0
                end select
            enddo
        enddo
        !$omp end do
        !$omp end parallel
        if(any(local_status/=scalar_live_ok)) status=scalar_live_slope_failure
        deallocate(work_bed,local_status)
    end subroutine braids_scalar_flow_face_candidates_v65


    subroutine braids_scalar_flow_project_exner_v65( &
            ncell,nface,nfrac,topology,corrected_q,donor_cell, &
            old_mass,bulk_density,cell_area,dt, &
            morphological_factor,active_fraction,inventory_fill,availability, &
            face_flux,cell_delta,candidate_mass,boundary_transfer, &
            boundary_absolute,conservation_residual,status,observed_threads, &
            active_application_count,inventory_application_count)
        integer, intent(in) :: ncell,nface,nfrac
        type(scalar_flow_projection_topology_v65), intent(in) :: topology
        real(8), intent(in) :: corrected_q(2,nfrac,nface)
        integer, intent(in) :: donor_cell(nfrac,nface)
        real(8), intent(in) :: old_mass(nfrac,ncell),bulk_density(nfrac)
        real(8), intent(in) :: cell_area(ncell),dt,morphological_factor
        real(8), intent(out) :: active_fraction(nfrac,ncell)
        real(8), intent(out) :: inventory_fill(nfrac,ncell)
        real(8), intent(out) :: availability(nfrac,ncell)
        real(8), intent(out) :: face_flux(nfrac,nface)
        real(8), intent(out) :: cell_delta(nfrac,ncell)
        real(8), intent(out) :: candidate_mass(nfrac,ncell)
        real(8), intent(out) :: boundary_transfer(nfrac)
        real(8), intent(out) :: boundary_absolute(nfrac)
        real(8), intent(out) :: conservation_residual(nfrac)
        integer, intent(out) :: status,observed_threads
        integer, intent(out) :: active_application_count
        integer, intent(out) :: inventory_application_count
        real(8), allocatable :: volume(:,:),total_volume(:),potential(:,:)
        real(8), allocatable :: demand_face(:,:),outgoing(:,:)
        real(8), allocatable :: residual_bound(:)
        logical, allocatable :: depleted(:)
        integer :: cell,face,fraction,own,other,face_role,source,item
        integer :: start,finish
        real(8) :: value,preliminary,scale,transfer,tolerance,residual_scale

        active_fraction=0.d0
        inventory_fill=0.d0
        availability=0.d0
        face_flux=0.d0
        cell_delta=0.d0
        candidate_mass=old_mass
        boundary_transfer=0.d0
        boundary_absolute=0.d0
        conservation_residual=0.d0
        status=scalar_live_ok
        observed_threads=1
        active_application_count=0
        inventory_application_count=0
        if(ncell<=0.or.nface<=0.or.nfrac<=0.or..not.topology%prepared.or. &
           topology%ncell/=ncell.or.topology%nface/=nface) then
            status=scalar_live_invalid_shape
            return
        endif
        if(.not.all(ieee_is_finite(corrected_q)).or. &
           .not.all(ieee_is_finite(old_mass)).or. &
           .not.all(ieee_is_finite(bulk_density)).or. &
           .not.all(ieee_is_finite(cell_area)).or. &
           .not.all(ieee_is_finite([dt,morphological_factor])).or. &
           any(old_mass<0.d0).or. &
           any(bulk_density<=0.d0).or.any(cell_area<=0.d0).or. &
           dt<=0.d0.or.morphological_factor<0.d0) then
            status=scalar_live_invalid_input
            return
        endif
        do face=1,nface
            face_role=topology%role(face)
            do fraction=1,nfrac
                source=donor_cell(fraction,face)
                if(face_role==scalar_live_face_closed) then
                    if(source/=0) then
                        status=scalar_live_invalid_topology
                        return
                    endif
                else if(source<1.or.source>ncell) then
                    status=scalar_live_invalid_topology
                    return
                endif
            enddo
        enddo
        associate(owner=>topology%owner,neighbour=>topology%neighbour, &
                  role=>topology%role,normal=>topology%normal, &
                  length_m=>topology%length_m, &
                  cell_face_offsets=>topology%cell_face_offsets, &
                  cell_faces=>topology%cell_faces, &
                  boundary_faces=>topology%boundary_faces, &
                  nboundary=>topology%nboundary)
        allocate(volume(nfrac,ncell),total_volume(ncell))
        allocate(potential(nfrac,nface),demand_face(nfrac,nface))
        allocate(outgoing(nfrac,ncell),residual_bound(nfrac),depleted(ncell))
        volume=0.d0
        total_volume=0.d0
        potential=0.d0
        demand_face=0.d0
        outgoing=0.d0
        residual_bound=0.d0
        depleted=.false.
        ! One team spans the complete projection.  Cell-local CSR reductions
        ! parallelize the limiter and Exner divergence without atomics or a
        ! serial all-face accumulation.
        !$omp parallel default(shared) private(cell,face,fraction,own,other, &
        !$omp& face_role,source,item,start,finish,value,preliminary,scale, &
        !$omp& transfer,tolerance,residual_scale)
        !$omp single
        observed_threads=omp_get_num_threads()
        !$omp end single
        !$omp do collapse(2) schedule(static,256)
        do cell=1,ncell
            do fraction=1,nfrac
                volume(fraction,cell)=old_mass(fraction,cell)/ &
                                      bulk_density(fraction)
            enddo
        enddo
        !$omp end do
        !$omp do schedule(static,256)
        do cell=1,ncell
            total_volume(cell)=sum(volume(:,cell))
            if(total_volume(cell)<=0.d0) then
                depleted(cell)=.true.
            else
                active_fraction(:,cell)=volume(:,cell)/total_volume(cell)
            endif
        enddo
        !$omp end do
        !$omp single
        if(any(depleted)) status=scalar_live_depleted_cell
        !$omp end single

        !$omp do collapse(2) schedule(static,256)
        do face=1,nface
            do fraction=1,nfrac
                face_role=role(face)
                value=dot_product(corrected_q(:,fraction,face),normal(:,face))
                select case(face_role)
                case(scalar_live_face_trace_inlet)
                    value=min(0.d0,value)
                case(scalar_live_face_trace_outlet,scalar_live_face_owner_outlet)
                    value=max(0.d0,value)
                case(scalar_live_face_closed)
                    value=0.d0
                end select
                potential(fraction,face)=value
                source=donor_cell(fraction,face)
                if(source>0) then
                    preliminary=value*active_fraction(fraction,source)
                    if((face_role==scalar_live_face_internal.and. &
                        preliminary/=0.d0).or. &
                       ((face_role==scalar_live_face_trace_outlet.or. &
                         face_role==scalar_live_face_owner_outlet).and. &
                        preliminary>0.d0)) then
                        demand_face(fraction,face)=abs(preliminary)* &
                            length_m(face)*dt*morphological_factor/ &
                            cell_area(source)
                    endif
                endif
            enddo
        enddo
        !$omp end do
        ! Each cell row is sorted by global face number.  The independent
        ! reductions therefore remain bitwise invariant to OpenMP team size.
        !$omp do collapse(2) schedule(static,256)
        do cell=1,ncell
            do fraction=1,nfrac
                start=cell_face_offsets(cell)
                finish=cell_face_offsets(cell+1)-1
                do item=start,finish
                    face=cell_faces(item)
                    if(donor_cell(fraction,face)==cell) then
                        outgoing(fraction,cell)=outgoing(fraction,cell)+ &
                                                demand_face(fraction,face)
                    endif
                enddo
            enddo
        enddo
        !$omp end do
        !$omp do collapse(2) schedule(static,256)
        do cell=1,ncell
            do fraction=1,nfrac
                if(old_mass(fraction,cell)==0.d0) then
                    inventory_fill(fraction,cell)=0.d0
                else if(outgoing(fraction,cell)>0.d0) then
                    inventory_fill(fraction,cell)=min(1.d0, &
                        old_mass(fraction,cell)/outgoing(fraction,cell))
                else
                    inventory_fill(fraction,cell)=1.d0
                endif
                availability(fraction,cell)=active_fraction(fraction,cell)* &
                                            inventory_fill(fraction,cell)
            enddo
        enddo
        !$omp end do
        !$omp single
        active_application_count=1
        inventory_application_count=1
        !$omp end single

        !$omp do collapse(2) schedule(static,256)
        do face=1,nface
            do fraction=1,nfrac
                source=donor_cell(fraction,face)
                if(source>0) face_flux(fraction,face)= &
                    potential(fraction,face)*availability(fraction,source)
            enddo
        enddo
        !$omp end do
        !$omp do collapse(2) schedule(static,256)
        do cell=1,ncell
            do fraction=1,nfrac
                start=cell_face_offsets(cell)
                finish=cell_face_offsets(cell+1)-1
                do item=start,finish
                    face=cell_faces(item)
                    transfer=dt*morphological_factor*length_m(face)* &
                             face_flux(fraction,face)
                    if(owner(face)==cell) then
                        cell_delta(fraction,cell)=cell_delta(fraction,cell)- &
                                                  transfer/cell_area(cell)
                    else
                        cell_delta(fraction,cell)=cell_delta(fraction,cell)+ &
                                                  transfer/cell_area(cell)
                    endif
                enddo
            enddo
        enddo
        !$omp end do
        ! The authenticated boundary catalog is O(perimeter), not O(area).
        ! Fractions are independent; each row retains canonical face order.
        !$omp do schedule(static)
        do fraction=1,nfrac
            do item=1,nboundary
                face=boundary_faces(item)
                transfer=dt*morphological_factor*length_m(face)* &
                         face_flux(fraction,face)
                boundary_transfer(fraction)=boundary_transfer(fraction)+ &
                                            transfer
                boundary_absolute(fraction)=boundary_absolute(fraction)+ &
                                            abs(transfer)
            enddo
        enddo
        !$omp end do
        !$omp do collapse(2) schedule(static,256)
        do cell=1,ncell
            do fraction=1,nfrac
                candidate_mass(fraction,cell)=old_mass(fraction,cell)+ &
                                              cell_delta(fraction,cell)
                tolerance=2048.d0*epsilon(1.d0)*max(1.d0, &
                                                   abs(old_mass(fraction,cell)))
                if(candidate_mass(fraction,cell)>=-tolerance) then
                    candidate_mass(fraction,cell)=max(0.d0, &
                                                      candidate_mass(fraction,cell))
                endif
            enddo
        enddo
        !$omp end do
        !$omp single
        if(any(candidate_mass<0.d0)) status=scalar_live_inventory_failure
        if(status==scalar_live_ok) then
            do cell=1,ncell
                if(sum(candidate_mass(:,cell)/bulk_density)<=0.d0) then
                    status=scalar_live_depleted_cell
                    exit
                endif
            enddo
        endif
        !$omp end single
        !$omp do schedule(static)
        do fraction=1,nfrac
            do cell=1,ncell
                conservation_residual(fraction)= &
                    conservation_residual(fraction)+ &
                    cell_delta(fraction,cell)*cell_area(cell)
            enddo
            conservation_residual(fraction)=conservation_residual(fraction)+ &
                                            boundary_transfer(fraction)
            residual_scale=max(1.d0,abs(boundary_transfer(fraction)))
            do cell=1,ncell
                residual_scale=residual_scale+ &
                    abs(cell_delta(fraction,cell)*cell_area(cell))
            enddo
            residual_bound(fraction)=4096.d0*epsilon(1.d0)*residual_scale
        enddo
        !$omp end do
        !$omp single
        if(any(abs(conservation_residual)>residual_bound)) &
            status=scalar_live_conservation_failure
        !$omp end single
        !$omp end parallel
        deallocate(volume,total_volume,potential,demand_face,outgoing, &
                   residual_bound,depleted)
        end associate
    end subroutine braids_scalar_flow_project_exner_v65

end module braids_scalar_flow_live_v65_module
