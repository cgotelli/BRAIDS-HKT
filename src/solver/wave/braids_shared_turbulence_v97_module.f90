! Additive wet-active v91 momentum operator; same face law as v96.
module braids_shared_turbulence_v97_module
 use iso_c_binding
 use omp_lib
 use, intrinsic :: ieee_arithmetic
 implicit none
contains
 subroutine braids_v97_masked_momentum(nc,nf,nadj,q,vic,owner,neighbour,length,distance,area, &
  offset,faces,signs,wet,active,threads,flux,rate,status) bind(C)
  integer(c_int),value :: nc,nf,nadj,threads
  real(c_double),intent(in) :: q(nc,3),vic(nc),length(nf),distance(nf),area(nc)
  integer(c_int),intent(in) :: owner(nf),neighbour(nf),offset(nc+1),faces(nadj),signs(nadj),wet(nc),active(nf)
  real(c_double),intent(out) :: flux(nf,2),rate(nc,3)
  integer(c_int),intent(out) :: status
  integer :: f,c,l,r,k,slot,bad
  real(8) :: hl,hr,coefficient,total,value
  bad=0
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) &
  !$omp& private(f,l,r,hl,hr,k,coefficient,value) reduction(max:bad)
  do f=1,nf
   flux(f,:)=0.d0
   if(active(f)==0) cycle
   l=owner(f)+1; r=neighbour(f)+1
   if(r==0.or.wet(l)==0.or.wet(r)==0) then
    bad=1; cycle
   endif
   hl=q(l,1); hr=q(r,1)
   if(hl<=0.d0.or.hr<=0.d0) then
    bad=1; cycle
   endif
   coefficient=.5d0*(hl+hr)*.5d0*(vic(l)+vic(r))*length(f)/distance(f)
   do k=1,2
    value=coefficient*(q(r,k+1)/hr-q(l,k+1)/hl)
    if(.not.ieee_is_finite(value)) then
     bad=1
    else
     flux(f,k)=value
    endif
   enddo
  enddo
  !$omp end parallel do
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) &
  !$omp& private(c,k,slot,f,total,value) reduction(max:bad)
  do c=1,nc
   rate(c,:)=0.d0
   if(wet(c)==0) cycle
   do k=1,2
    total=0.d0
    do slot=offset(c)+1,offset(c+1)
     f=faces(slot)+1
     total=total+signs(slot)*flux(f,k)
    enddo
    value=total/area(c)
    if(.not.ieee_is_finite(value)) then
     bad=1
    else
     rate(c,k+1)=value
    endif
   enddo
  enddo
  !$omp end parallel do
  status=bad
 end subroutine
end module
