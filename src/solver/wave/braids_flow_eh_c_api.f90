! Paired classic-FLOW EH capacity/spiral bridge; no Delft source is compiled.
module braids_flow_eh_c_api
 use iso_c_binding
 use omp_lib
 use braids_bedload_current_module, only: braids_current_bedload
 use braids_flow2dh_secondary_operator_module, only: braids_flow2dh_scalar_bed_rotation
 use braids_flow_bed_slope_operator_module, only: braids_flow_bed_slope_direction
 implicit none
 contains
 subroutine braids_flow_eh_cells(n,threads,diameter,chezy,q,capacity_q,hi,out,status) bind(C)
  integer(c_int),value :: n,threads
  real(c_double),value :: diameter
  real(c_double),intent(in) :: chezy(n),q(n,3),capacity_q(n,3),hi(n)
  real(c_double),intent(out) :: out(n,6)
  integer(c_int),intent(out) :: status(n)
  integer :: i
  real(8) :: h,u,v
  logical :: valid,rotated
  !$omp parallel do num_threads(threads) schedule(static) private(h,u,v,valid,rotated)
  do i=1,n
   out(i,:)=0.d0;status(i)=0
   h=q(i,1)
   if(h<=1.d-3) then
    status(i)=1;cycle ! This bridge is explicitly fully wet.
   endif
   u=q(i,2)/h;v=q(i,3)/h
   call braids_current_bedload(8,capacity_q(i,1),capacity_q(i,2)/capacity_q(i,1), &
     capacity_q(i,3)/capacity_q(i,1),0.d0,1.d-5,diameter,2650.d0,1000.d0,9.81d0, &
     diameter,1.d0,1.d0,1.d0,out(i,1),out(i,2),out(i,3),valid,chezy(i))
   call braids_flow2dh_scalar_bed_rotation(.true.,u,v,9.81d0,chezy(i),1.d0,hi(i)/h, &
     out(i,1),out(i,2),out(i,4),out(i,5),out(i,6),rotated)
   if(.not.(valid.and.rotated)) status(i)=1
  enddo
  !$omp end parallel do
 end subroutine
 subroutine braids_flow_eh_slope(n,threads,x,out,status) bind(C)
  integer(c_int),value :: n,threads
  real(c_double),intent(in) :: x(n,10)
  real(c_double),intent(out) :: out(n,2)
  integer(c_int),intent(out) :: status(n)
  integer :: i
  !$omp parallel do num_threads(threads) schedule(static)
  do i=1,n
   call braids_flow_bed_slope_direction(3,x(i,1),x(i,2),x(i,3),x(i,4),x(i,5), &
    x(i,6),x(i,7),x(i,8),x(i,9),x(i,10),0.d0,0.d0,.85d0,.5d0,0.d0,0.d0, &
    1.d0,.05d0,out(i,1),out(i,2),status(i))
  enddo
  !$omp end parallel do
 end subroutine
end module
