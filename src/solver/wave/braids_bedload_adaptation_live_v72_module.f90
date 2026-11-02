! Additive law-neutral native/OpenMP BRAIDS bed-load adaptation closure.
!
! This module owns no accepted state.  It consumes an already authenticated
! classwise equilibrium bed-capacity vector and returns the derived steady
! streamline-relaxed vector.  The caller must pass that vector to exactly one
! face/inventory/Exner owner.  It is a BRAIDS extension, not Delft3D coverage.
module braids_bedload_adaptation_live_v72_module
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use omp_lib, only: omp_get_num_threads
    implicit none
    private

    integer, parameter, public :: adaptation_v72_ok = 0
    integer, parameter, public :: adaptation_v72_invalid_shape = 720
    integer, parameter, public :: adaptation_v72_invalid_input = 721
    integer, parameter, public :: adaptation_v72_invalid_topology = 722
    integer, parameter, public :: adaptation_v72_cancelled = 723
    integer, parameter, public :: adaptation_v72_nonconverged = 724
    integer, parameter, public :: adaptation_v72_nonfinite = 725
    integer, parameter, public :: adaptation_v72_boundary_feed = 0
    integer, parameter, public :: adaptation_v72_boundary_equilibrium = 1
    integer, parameter, public :: adaptation_v72_max_accepted_steps = 4

    type, public :: braids_bedload_adaptation_topology_v72
        integer :: ncell = 0
        integer :: nface = 0
        integer :: nboundary = 0
        integer :: nadjacency = 0
        real(real64), allocatable :: xy(:,:)
        real(real64), allocatable :: mid(:,:)
        real(real64), allocatable :: normal(:,:)
        real(real64), allocatable :: face_length_m(:)
        real(real64), allocatable :: cell_area_m2(:)
        integer, allocatable :: owner(:)
        integer, allocatable :: neighbour(:)
        integer, allocatable :: boundary_index(:)
        integer, allocatable :: cell_face_offset(:)
        integer, allocatable :: cell_faces(:)
        integer, allocatable :: signs(:)
        integer, allocatable :: hydraulic_mode(:)
        integer, allocatable :: bed_boundary_mode(:)
        logical :: prepared = .false.
    end type braids_bedload_adaptation_topology_v72

    public :: braids_prepare_bedload_adaptation_topology_v72
    public :: braids_relax_bedload_capacity_v72

contains

    pure logical function inlet_mode_v72(mode)
        integer, intent(in) :: mode
        inlet_mode_v72 = mode == 1 .or. mode == 6 .or. mode == 7 .or. &
                          mode == 8 .or. mode == 9
    end function inlet_mode_v72

    subroutine braids_prepare_bedload_adaptation_topology_v72( &
            ncell, nface, nboundary, nadjacency, xy, mid, normal, &
            face_length_m, cell_area_m2, owner, neighbour, boundary_index, &
            cell_face_offset, cell_faces, signs, hydraulic_mode, &
            bed_boundary_mode, topology, status)
        integer, intent(in) :: ncell,nface,nboundary,nadjacency
        real(real64), intent(in) :: xy(2,ncell),mid(2,nface)
        real(real64), intent(in) :: normal(2,nface),face_length_m(nface)
        real(real64), intent(in) :: cell_area_m2(ncell)
        integer, intent(in) :: owner(nface),neighbour(nface)
        integer, intent(in) :: boundary_index(nface)
        integer, intent(in) :: cell_face_offset(ncell+1)
        integer, intent(in) :: cell_faces(nadjacency),signs(nadjacency)
        integer, intent(in) :: hydraulic_mode(nboundary)
        integer, intent(in) :: bed_boundary_mode(nboundary)
        type(braids_bedload_adaptation_topology_v72), intent(out) :: topology
        integer, intent(out) :: status
        integer :: cell,item,face,sign,local_cell

        status=adaptation_v72_invalid_shape
        if(ncell<=0.or.nface<=0.or.nboundary<0.or.nadjacency<=0) return
        status=adaptation_v72_invalid_input
        if(any(.not.ieee_is_finite(xy)).or.any(.not.ieee_is_finite(mid)).or. &
           any(.not.ieee_is_finite(normal)).or. &
           any(.not.ieee_is_finite(face_length_m)).or. &
           any(.not.ieee_is_finite(cell_area_m2)).or. &
           any(face_length_m<=0.0_real64).or.any(cell_area_m2<=0.0_real64)) return
        status=adaptation_v72_invalid_topology
        if(any(owner<1).or.any(owner>ncell).or.any(neighbour<0).or. &
           any(neighbour>ncell).or.any(boundary_index<0).or. &
           any(boundary_index>nboundary).or. &
           any((neighbour>0).and.(boundary_index>0)).or. &
           cell_face_offset(1)/=1.or. &
           cell_face_offset(ncell+1)/=nadjacency+1.or. &
           any(cell_face_offset(2:)-cell_face_offset(:ncell)<0).or. &
           any(cell_faces<1).or.any(cell_faces>nface).or. &
           any((signs/=-1).and.(signs/=1)).or. &
           any((bed_boundary_mode/=adaptation_v72_boundary_feed).and. &
               (bed_boundary_mode/=adaptation_v72_boundary_equilibrium))) return
        do cell=1,ncell
            do item=cell_face_offset(cell),cell_face_offset(cell+1)-1
                face=cell_faces(item);sign=signs(item)
                if(sign>0) then
                    local_cell=owner(face)
                else
                    local_cell=neighbour(face)
                endif
                if(local_cell/=cell) return
            enddo
        enddo
        topology%ncell=ncell;topology%nface=nface
        topology%nboundary=nboundary;topology%nadjacency=nadjacency
        allocate(topology%xy(2,ncell),topology%mid(2,nface), &
                 topology%normal(2,nface),topology%face_length_m(nface), &
                 topology%cell_area_m2(ncell),topology%owner(nface), &
                 topology%neighbour(nface),topology%boundary_index(nface), &
                 topology%cell_face_offset(ncell+1), &
                 topology%cell_faces(nadjacency),topology%signs(nadjacency), &
                 topology%hydraulic_mode(nboundary), &
                 topology%bed_boundary_mode(nboundary))
        topology%xy=xy;topology%mid=mid;topology%normal=normal
        topology%face_length_m=face_length_m
        topology%cell_area_m2=cell_area_m2;topology%owner=owner
        topology%neighbour=neighbour;topology%boundary_index=boundary_index
        topology%cell_face_offset=cell_face_offset
        topology%cell_faces=cell_faces;topology%signs=signs
        topology%hydraulic_mode=hydraulic_mode
        topology%bed_boundary_mode=bed_boundary_mode
        topology%prepared=.true.;status=adaptation_v72_ok
    end subroutine braids_prepare_bedload_adaptation_topology_v72

    subroutine braids_relax_bedload_capacity_v72( &
            topology,nfrac,q,d50_m,length_diameters,dry_tolerance,dx_m,dy_m, &
            boundary_feed_kg_m_s,equilibrium_capacity,adapted_capacity, &
            iterations,residual,tolerance,status,observed_threads,cancelled)
        type(braids_bedload_adaptation_topology_v72), intent(in) :: topology
        integer, intent(in) :: nfrac
        real(real64), intent(in) :: q(3,topology%ncell)
        real(real64), intent(in) :: d50_m(nfrac),length_diameters
        real(real64), intent(in) :: dry_tolerance,dx_m,dy_m
        real(real64), intent(in) :: &
            boundary_feed_kg_m_s(nfrac,topology%nboundary)
        real(real64), intent(in) :: &
            equilibrium_capacity(2,nfrac,topology%ncell)
        real(real64), intent(out) :: &
            adapted_capacity(2,nfrac,topology%ncell)
        integer, intent(out) :: iterations,status,observed_threads
        real(real64), intent(out) :: residual,tolerance
        logical, intent(in), optional :: cancelled
        real(real64), allocatable :: previous(:,:,:),following(:,:,:)
        real(real64), allocatable :: face_weight(:),total_weight(:)
        real(real64), allocatable :: distance(:),boundary_span(:)
        integer :: cell,grain,item,face,other,boundary,iteration
        integer :: sign
        real(real64) :: cell_velocity(2),other_velocity(2),face_velocity(2)
        real(real64) :: outward(2),upstream(2),source(2),delta(2)
        real(real64) :: speed,face_speed,inflow,weight,segment,dsum
        real(real64) :: attenuation,change,max_change,scale
        logical :: converged,abort_requested

        adapted_capacity=equilibrium_capacity;iterations=0;residual=0.0_real64
        tolerance=0.0_real64;observed_threads=1
        status=adaptation_v72_invalid_shape
        if(.not.topology%prepared.or.nfrac<=0.or.topology%ncell<=0.or. &
           topology%nface<=0) return
        status=adaptation_v72_invalid_input
        if(any(.not.ieee_is_finite(q)).or.any(.not.ieee_is_finite(d50_m)).or. &
           any(.not.ieee_is_finite(boundary_feed_kg_m_s)).or. &
           any(.not.ieee_is_finite(equilibrium_capacity)).or. &
           any(d50_m<=0.0_real64).or.any(boundary_feed_kg_m_s<0.0_real64).or. &
           .not.ieee_is_finite(length_diameters).or.length_diameters<=0.0_real64.or. &
           .not.ieee_is_finite(dry_tolerance).or.dry_tolerance<0.0_real64.or. &
           .not.ieee_is_finite(dx_m).or.dx_m<=0.0_real64.or. &
           .not.ieee_is_finite(dy_m).or.dy_m<=0.0_real64) return
        abort_requested=.false.
        if(present(cancelled)) abort_requested=cancelled
        if(abort_requested) then
            status=adaptation_v72_cancelled
            return
        endif
        allocate(previous(2,nfrac,topology%ncell), &
                 following(2,nfrac,topology%ncell), &
                 face_weight(topology%nadjacency), &
                 total_weight(topology%ncell),distance(topology%ncell), &
                 boundary_span(topology%nboundary))
        previous=equilibrium_capacity;following=equilibrium_capacity
        face_weight=0.0_real64;total_weight=0.0_real64
        distance=0.0_real64;boundary_span=0.0_real64
        do face=1,topology%nface
            boundary=topology%boundary_index(face)
            if(boundary>0) boundary_span(boundary)=boundary_span(boundary)+ &
                topology%face_length_m(face)
        enddo
        !$omp parallel do schedule(static,256) default(shared) &
        !$omp& private(cell,item,face,sign,other,boundary,cell_velocity, &
        !$omp& other_velocity,face_velocity,outward,delta,speed,face_speed, &
        !$omp& inflow,weight,segment,dsum)
        do cell=1,topology%ncell
            cell_velocity=0.0_real64
            if(q(1,cell)>dry_tolerance) cell_velocity=q(2:3,cell)/q(1,cell)
            speed=sqrt(sum(cell_velocity**2));dsum=0.0_real64
            do item=topology%cell_face_offset(cell), &
                    topology%cell_face_offset(cell+1)-1
                face=topology%cell_faces(item);sign=topology%signs(item)
                outward=real(sign,real64)*topology%normal(:,face)
                if(sign>0) then
                    other=topology%neighbour(face)
                else
                    other=topology%owner(face)
                endif
                weight=0.0_real64;segment=0.0_real64
                if(speed>0.0_real64.and.q(1,cell)>dry_tolerance) then
                    if(other>0) then
                        other_velocity=0.0_real64
                        if(q(1,other)>dry_tolerance) &
                            other_velocity=q(2:3,other)/q(1,other)
                        face_velocity=0.5_real64*(cell_velocity+other_velocity)
                        inflow=-dot_product(face_velocity,outward)
                        if(inflow>0.0_real64) then
                            weight=inflow*topology%face_length_m(face)
                            face_speed=sqrt(sum(face_velocity**2))
                            delta=topology%xy(:,cell)-topology%xy(:,other)
                            if(face_speed>0.0_real64) &
                                segment=abs(dot_product(delta,face_velocity/face_speed))
                            if(segment<=64.0_real64*epsilon(1.0_real64)*max(dx_m,dy_m)) &
                                segment=sqrt(sum(delta**2))
                        endif
                    else
                        boundary=topology%boundary_index(face)
                        if(boundary>0.and.inlet_mode_v72( &
                                topology%hydraulic_mode(boundary))) then
                            inflow=-dot_product(cell_velocity,outward)
                            if(inflow>0.0_real64) then
                                weight=inflow*topology%face_length_m(face)
                                delta=topology%xy(:,cell)-topology%mid(:,face)
                                segment=abs(dot_product(delta,cell_velocity/speed))
                                if(segment<=64.0_real64*epsilon(1.0_real64)*max(dx_m,dy_m)) &
                                    segment=sqrt(sum(delta**2))
                            endif
                        endif
                    endif
                endif
                face_weight(item)=weight
                total_weight(cell)=total_weight(cell)+weight
                dsum=dsum+weight*segment
            enddo
            if(total_weight(cell)>0.0_real64) then
                distance(cell)=dsum/total_weight(cell)
                if(distance(cell)<=64.0_real64*epsilon(1.0_real64)*max(dx_m,dy_m)) &
                    distance(cell)=topology%cell_area_m2(cell)*speed/total_weight(cell)
            endif
        enddo
        !$omp end parallel do

        scale=max(maxval(abs(equilibrium_capacity)),1.0e-20_real64)
        tolerance=1.0e-11_real64*scale;converged=.false.
        do iteration=1,4096
            max_change=0.0_real64
            !$omp parallel default(shared) private(cell,grain,item,face,sign, &
            !$omp& other,boundary,upstream,source,weight,attenuation,change)
            !$omp single
            observed_threads=max(observed_threads,omp_get_num_threads())
            !$omp end single
            !$omp do collapse(2) schedule(static,256) reduction(max:max_change)
            do cell=1,topology%ncell
                do grain=1,nfrac
                    following(:,grain,cell)=equilibrium_capacity(:,grain,cell)
                    if(total_weight(cell)>0.0_real64) then
                        upstream=0.0_real64
                        do item=topology%cell_face_offset(cell), &
                                topology%cell_face_offset(cell+1)-1
                            weight=face_weight(item)
                            if(weight<=0.0_real64) cycle
                            face=topology%cell_faces(item)
                            sign=topology%signs(item)
                            if(sign>0) then
                                other=topology%neighbour(face)
                            else
                                other=topology%owner(face)
                            endif
                            if(other>0) then
                                source=previous(:,grain,other)
                            else
                                boundary=topology%boundary_index(face)
                                source=0.0_real64
                                if(boundary>0) then
                                    if(topology%bed_boundary_mode(boundary)== &
                                            adaptation_v72_boundary_equilibrium) then
                                        source=equilibrium_capacity(:,grain,cell)
                                    else if(boundary_span(boundary)>0.0_real64) then
                                        source=-real(sign,real64)*topology%normal(:,face)* &
                                            boundary_feed_kg_m_s(grain,boundary)/ &
                                            boundary_span(boundary)
                                    endif
                                endif
                            endif
                            upstream=upstream+weight*source
                        enddo
                        attenuation=exp(-distance(cell)/( &
                            length_diameters*d50_m(grain)))
                        following(:,grain,cell)= &
                            (1.0_real64-attenuation)* &
                            equilibrium_capacity(:,grain,cell)+ &
                            attenuation*upstream/total_weight(cell)
                    endif
                    change=maxval(abs(following(:,grain,cell)- &
                                      previous(:,grain,cell)))
                    max_change=max(max_change,change)
                enddo
            enddo
            !$omp end do
            !$omp end parallel
            residual=max_change;iterations=iteration
            if(.not.ieee_is_finite(residual).or. &
               any(.not.ieee_is_finite(following))) then
                status=adaptation_v72_nonfinite
                deallocate(previous,following,face_weight,total_weight, &
                           distance,boundary_span)
                return
            endif
            if(residual<=tolerance) then
                converged=.true.;exit
            endif
            previous=following
        enddo
        if(.not.converged) then
            status=adaptation_v72_nonconverged
        else
            adapted_capacity=following;status=adaptation_v72_ok
        endif
        deallocate(previous,following,face_weight,total_weight,distance, &
                   boundary_span)
    end subroutine braids_relax_bedload_capacity_v72

end module braids_bedload_adaptation_live_v72_module
