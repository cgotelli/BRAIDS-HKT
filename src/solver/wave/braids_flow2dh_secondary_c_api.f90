! Batched C interface to the existing classic FLOW constitutive module.
! BRAIDS source only; no Delft3D source is compiled or modified here.
subroutine braids_flow2dh_batch(n,threads,x,p,y,flags) bind(C,name='braids_flow2dh_batch')
 use iso_c_binding
 use omp_lib
 use braids_flow2dh_secondary_operator_module
 implicit none
 integer(c_int),value :: n,threads
 real(c_double),intent(in) :: x(n,12),p(8)
 real(c_double),intent(out) :: y(n,15)
 integer(c_int),intent(out) :: flags(n,5)
 integer :: i
 logical :: capped,defined,enabled,clamped,valid
 call omp_set_dynamic(.false.)
 !$omp parallel do num_threads(threads) private(capped,defined,enabled,clamped,valid) schedule(static)
 do i=1,n
  call braids_flow2dh_secondary_state(.true.,x(i,1),x(i,2),x(i,3),x(i,4), &
   x(i,5),x(i,6),x(i,7),x(i,8),x(i,9),x(i,10),x(i,11),x(i,12), &
   p(1),p(2),p(3),p(4),p(5),p(6),p(7),p(8), &
   y(i,1),y(i,2),y(i,3),y(i,4),y(i,5),y(i,6),y(i,7),y(i,8), &
   y(i,9),y(i,10),y(i,11),y(i,12),y(i,13),y(i,14),y(i,15), &
   capped,defined,enabled,clamped,valid)
  flags(i,:)=[merge(1,0,capped),merge(1,0,defined),merge(1,0,enabled), &
              merge(1,0,clamped),merge(0,1,valid)]
 enddo
 !$omp end parallel do
end subroutine
