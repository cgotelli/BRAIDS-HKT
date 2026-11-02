! Pure additive coarse/fine viscosity contribution, unchanged v91 face law.
module braids_momentum_trace_v98_module
 use iso_c_binding
 use omp_lib
 use, intrinsic :: ieee_arithmetic
 implicit none
contains
 subroutine braids_momentum_trace_v98(nc,nf,nt,nadj,owner,lookup,q,vic,length,distance,area, &
  offset,faces,signs,wet,trace_q,trace_vic,trace_wet,threads,flux,rate,status) bind(C)
  integer(c_int),value :: nc,nf,nt,nadj,threads
  integer(c_int),intent(in) :: owner(nf),lookup(nf),offset(nc+1),faces(nadj),signs(nadj), &
   wet(nc),trace_wet(nt)
  real(c_double),intent(in) :: q(nc,3),vic(nc),length(nf),distance(nf),area(nc), &
   trace_q(nt,3),trace_vic(nt)
  real(c_double),intent(out) :: flux(nf,2),rate(nc,2)
  integer(c_int),intent(out) :: status(nf)
  integer :: f,c,t,k,slot
  real(c_double) :: coefficient,total
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) &
  !$omp& default(shared) private(f,c,t,k,coefficient)
  do f=1,nf
   flux(f,:)=0.d0;status(f)=0;t=lookup(f)+1
   if(t==0) cycle
   c=owner(f)+1
   if(wet(c)==0.or.trace_wet(t)==0) cycle
   coefficient=.5d0*(q(c,1)+trace_q(t,1))*.5d0*(vic(c)+trace_vic(t))*length(f)/distance(f)
   do k=1,2
    flux(f,k)=coefficient*(trace_q(t,k+1)/trace_q(t,1)-q(c,k+1)/q(c,1))
    if(.not.ieee_is_finite(flux(f,k))) status(f)=1
   enddo
  enddo
  !$omp end parallel do
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) &
  !$omp& default(shared) private(c,k,slot,f,total)
  do c=1,nc
   rate(c,:)=0.d0
   if(wet(c)==0) cycle
   do k=1,2
    total=0.d0
    do slot=offset(c)+1,offset(c+1)
     f=faces(slot)+1;total=total+signs(slot)*flux(f,k)
    enddo
    rate(c,k)=total/area(c)
   enddo
  enddo
  !$omp end parallel do
 end subroutine
end module
