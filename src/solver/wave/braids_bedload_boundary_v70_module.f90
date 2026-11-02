! Additive Q3/v70 noncohesive bed-vector boundary owner.
!
! Positive face transport is outward.  The caller owns constitutive capacity
! production and the complete internal-face demand.  This bounded module
! exercises the same prescribed/equilibrium/free/closed boundary transaction
! on the native CPU/OpenMP path; it does not implement Q6 bed-level modes.
module braids_bedload_boundary_v70_module
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use omp_lib, only: omp_get_num_threads
    implicit none
    private
    integer, parameter, public :: bed_boundary_v70_closed = 0
    integer, parameter, public :: bed_boundary_v70_equilibrium = 1
    integer, parameter, public :: bed_boundary_v70_prescribed = 2
    integer, parameter, public :: bed_boundary_v70_free_export = 3
    integer, parameter, public :: bed_boundary_v70_source_none = 0
    integer, parameter, public :: bed_boundary_v70_source_prescribed = 1
    integer, parameter, public :: bed_boundary_v70_source_equilibrium = 2
    integer, parameter, public :: bed_boundary_v70_source_owner_export = 3
    public :: braids_bedload_boundary_step_v70
contains
    subroutine braids_bedload_boundary_step_v70( &
        ncell,nface,nclass,nboundary,owner,boundary_index,mode,normal, &
        face_length,cell_area,hydraulic_outward,owner_vector,trace_vector, &
        prescribed_mass,old_mass,active_fraction,dt,morfac,potential, &
        requires_availability,source,provisional,demand,fill,availability, &
        face_flux,delta,new_mass,signed_transfer,absolute_transfer,residual, &
        worker_threads,status)
        integer, intent(in) :: ncell,nface,nclass,nboundary
        integer, intent(in) :: owner(nface),boundary_index(nface),mode(nface)
        real(real64), intent(in) :: normal(2,nface),face_length(nface)
        real(real64), intent(in) :: cell_area(ncell),hydraulic_outward(nface)
        real(real64), intent(in) :: owner_vector(2,nclass,ncell)
        real(real64), intent(in) :: trace_vector(2,nclass,nface)
        real(real64), intent(in) :: prescribed_mass(nclass,nface)
        real(real64), intent(in) :: old_mass(nclass,ncell)
        real(real64), intent(in) :: active_fraction(nclass,ncell),dt,morfac
        real(real64), intent(out) :: potential(nclass,nface)
        integer, intent(out) :: requires_availability(nclass,nface)
        integer, intent(out) :: source(nclass,nface)
        real(real64), intent(out) :: provisional(nclass,nface)
        real(real64), intent(out) :: demand(nclass,ncell),fill(nclass,ncell)
        real(real64), intent(out) :: availability(nclass,ncell)
        real(real64), intent(out) :: face_flux(nclass,nface)
        real(real64), intent(out) :: delta(nclass,ncell),new_mass(nclass,ncell)
        real(real64), intent(out) :: signed_transfer(nclass,nboundary)
        real(real64), intent(out) :: absolute_transfer(nclass,nboundary)
        real(real64), intent(out) :: residual(nclass)
        integer, intent(out) :: worker_threads,status
        integer :: grain,face,cell,boundary
        real(real64) :: projected,value,required,factor,integrated,amount
        real(real64) :: tolerance
        integer :: invalid_face(nclass,nface),invalid_cell(nclass,ncell)
        integer :: invalid_boundary(nclass,nboundary),invalid_residual(nclass)
        integer :: threads_seen

        potential=0.0_real64;requires_availability=0;source=0
        provisional=0.0_real64;demand=0.0_real64;fill=0.0_real64
        availability=0.0_real64;face_flux=0.0_real64
        delta=0.0_real64;new_mass=old_mass
        signed_transfer=0.0_real64;absolute_transfer=0.0_real64
        residual=0.0_real64;invalid_face=0;invalid_cell=0
        invalid_boundary=0;invalid_residual=0;threads_seen=1;status=0
        if (ncell<=0 .or. nface<=0 .or. nclass<=0 .or. nboundary<=0 .or. &
            .not.ieee_is_finite(dt) .or. dt<=0.0_real64 .or. &
            .not.ieee_is_finite(morfac) .or. morfac<0.0_real64) then
            status=1;worker_threads=1;return
        endif

!$omp parallel default(shared) private(grain,face,cell,boundary,projected,value,required,factor,integrated,amount,tolerance)
!$omp single
!$    threads_seen=omp_get_num_threads()
!$omp end single
!$omp do collapse(2)
        do face=1,nface
            do grain=1,nclass
                cell=owner(face)
                if (cell<1 .or. cell>ncell .or. boundary_index(face)<1 .or. &
                    boundary_index(face)>nboundary .or. mode(face)<0 .or. &
                    mode(face)>3 .or. .not.ieee_is_finite(hydraulic_outward(face)) .or. &
                    .not.all(ieee_is_finite(normal(:,face))) .or. &
                    .not.ieee_is_finite(face_length(face)) .or. face_length(face)<0.0_real64) then
                    invalid_face(grain,face)=1
                    cycle
                endif
                if (mode(face)==bed_boundary_v70_closed .or. hydraulic_outward(face)==0.0_real64) cycle
                if (hydraulic_outward(face)<0.0_real64) then
                    if (mode(face)==bed_boundary_v70_prescribed) then
                        if (.not.ieee_is_finite(prescribed_mass(grain,face)) .or. &
                            prescribed_mass(grain,face)<0.0_real64) then
                            invalid_face(grain,face)=1
                        else
                            potential(grain,face)=-prescribed_mass(grain,face)
                            source(grain,face)=bed_boundary_v70_source_prescribed
                        endif
                    elseif (mode(face)==bed_boundary_v70_equilibrium) then
                        projected=sum(trace_vector(:,grain,face)*normal(:,face))
                        if (.not.ieee_is_finite(projected)) then
                            invalid_face(grain,face)=1
                        else
                            potential(grain,face)=min(0.0_real64,projected)
                            requires_availability(grain,face)=1
                            source(grain,face)=bed_boundary_v70_source_equilibrium
                        endif
                    endif
                else
                    projected=sum(owner_vector(:,grain,cell)*normal(:,face))
                    if (.not.ieee_is_finite(projected)) then
                        invalid_face(grain,face)=1
                    else
                        potential(grain,face)=max(0.0_real64,projected)
                        requires_availability(grain,face)=1
                        source(grain,face)=bed_boundary_v70_source_owner_export
                    endif
                endif
            enddo
        enddo
!$omp end do
!$omp do collapse(2)
        do cell=1,ncell
            do grain=1,nclass
                if (.not.ieee_is_finite(cell_area(cell)) .or. cell_area(cell)<=0.0_real64 .or. &
                    .not.ieee_is_finite(old_mass(grain,cell)) .or. old_mass(grain,cell)<0.0_real64 .or. &
                    .not.ieee_is_finite(active_fraction(grain,cell)) .or. &
                    active_fraction(grain,cell)<0.0_real64 .or. active_fraction(grain,cell)>1.0_real64) then
                    invalid_cell(grain,cell)=1
                    cycle
                endif
                required=0.0_real64
                do face=1,nface
                    if (owner(face)/=cell) cycle
                    value=potential(grain,face)
                    if (requires_availability(grain,face)/=0) then
                        provisional(grain,face)=value*active_fraction(grain,cell)
                        if (value>0.0_real64) required=required+value*active_fraction(grain,cell)* &
                            face_length(face)*dt*morfac/cell_area(cell)
                    else
                        provisional(grain,face)=value
                    endif
                enddo
                demand(grain,cell)=required
                factor=merge(1.0_real64,0.0_real64,old_mass(grain,cell)>0.0_real64)
                if (required>0.0_real64) factor=min(factor,old_mass(grain,cell)/required)
                if (.not.ieee_is_finite(factor) .or. factor<0.0_real64 .or. factor>1.0_real64) then
                    invalid_cell(grain,cell)=1;factor=0.0_real64
                endif
                fill(grain,cell)=factor
                availability(grain,cell)=active_fraction(grain,cell)*factor
            enddo
        enddo
!$omp end do
!$omp do collapse(2)
        do face=1,nface
            do grain=1,nclass
                cell=owner(face)
                if (cell<1 .or. cell>ncell) cycle
                value=potential(grain,face)
                if (requires_availability(grain,face)/=0) value=value*availability(grain,cell)
                face_flux(grain,face)=value
            enddo
        enddo
!$omp end do
!$omp do collapse(2)
        do cell=1,ncell
            do grain=1,nclass
                integrated=0.0_real64
                do face=1,nface
                    if (owner(face)==cell) integrated=integrated+face_flux(grain,face)*face_length(face)
                enddo
                delta(grain,cell)=-dt*morfac*integrated/cell_area(cell)
                value=old_mass(grain,cell)+delta(grain,cell)
                tolerance=2048.0_real64*epsilon(1.0_real64)* &
                    max(1.0_real64,max(old_mass(grain,cell),abs(delta(grain,cell))))
                if (.not.ieee_is_finite(value) .or. value < -tolerance) then
                    invalid_cell(grain,cell)=1;delta(grain,cell)=0.0_real64
                    new_mass(grain,cell)=old_mass(grain,cell)
                else
                    new_mass(grain,cell)=max(0.0_real64,value)
                    delta(grain,cell)=new_mass(grain,cell)-old_mass(grain,cell)
                endif
            enddo
        enddo
!$omp end do
!$omp do collapse(2)
        do boundary=1,nboundary
            do grain=1,nclass
                do face=1,nface
                    if (boundary_index(face)/=boundary) cycle
                    amount=face_flux(grain,face)*face_length(face)*dt*morfac
                    if (.not.ieee_is_finite(amount)) then
                        invalid_boundary(grain,boundary)=1
                    else
                        signed_transfer(grain,boundary)=signed_transfer(grain,boundary)+amount
                        absolute_transfer(grain,boundary)=absolute_transfer(grain,boundary)+abs(amount)
                    endif
                enddo
            enddo
        enddo
!$omp end do
!$omp do
        do grain=1,nclass
            residual(grain)=sum(delta(grain,:)*cell_area)+sum(signed_transfer(grain,:))
            tolerance=4096.0_real64*epsilon(1.0_real64)*max(1.0_real64, &
                sum(abs(delta(grain,:)*cell_area))+sum(absolute_transfer(grain,:)))
            if (.not.ieee_is_finite(residual(grain)) .or. &
                abs(residual(grain))>tolerance) invalid_residual(grain)=1
        enddo
!$omp end do
!$omp end parallel
        worker_threads=threads_seen
        if (any(invalid_face/=0) .or. any(invalid_cell/=0) .or. &
            any(invalid_boundary/=0) .or. any(invalid_residual/=0)) status=1
    end subroutine braids_bedload_boundary_step_v70
end module braids_bedload_boundary_v70_module
