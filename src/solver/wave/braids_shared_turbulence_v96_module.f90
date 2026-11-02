! Native OpenMP implementation of the already admitted v91 auxiliary operator.
! No GeoClaw or Delft3D source is modified. Cell-owned gathers avoid atomics.
module braids_shared_turbulence_v96_module
 use iso_c_binding
 use, intrinsic :: ieee_arithmetic
 use omp_lib
 implicit none
contains
 subroutine braids_v96_cap(nc,width,raw_vic,raw_dic,dt,threads,vmax,dmax,vic,dic,flags,status) bind(C)
  integer(c_int),value :: nc,threads
  real(c_double),value :: dt
  real(c_double),intent(in) :: width(nc,2),raw_vic(nc),raw_dic(nc)
  real(c_double),intent(out) :: vmax(nc),dmax(nc),vic(nc),dic(nc)
  integer(c_int),intent(out) :: flags(nc,2),status
  integer :: c,bad
  real(8) :: dx,dy,metric,v,d
  status=0; flags=0; vmax=0.d0; dmax=0.d0; vic=0.d0; dic=0.d0
  if(nc<1.or.threads<1.or..not.ieee_is_finite(dt).or.dt<=0.d0) then
   status=1; return
  endif
  bad=0
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) &
  !$omp& private(c,dx,dy,metric,v,d) reduction(max:bad)
  do c=1,nc
   dx=width(c,1); dy=width(c,2); v=raw_vic(c); d=raw_dic(c)
   if(.not.ieee_is_finite(dx).or..not.ieee_is_finite(dy).or.dx<=0.d0.or.dy<=0.d0.or. &
      .not.ieee_is_finite(v).or..not.ieee_is_finite(d).or.v<0.d0.or.d<0.d0) then
    bad=1; cycle
   endif
   metric=1.d0/(dx*dx)+1.d0/(dy*dy)
   vmax(c)=.9d0/(2.d0*dt*metric); dmax(c)=.7d0*vmax(c)
   if(.not.ieee_is_finite(vmax(c)).or..not.ieee_is_finite(dmax(c))) then
    bad=1; cycle
   endif
   ! Preserve locked chkvic's strict > behavior, including exact equality.
   if(v>vmax(c)) then
    v=vmax(c); flags(c,1)=1
   endif
   if(d>dmax(c)) then
    d=dmax(c); flags(c,2)=1
   endif
   vic(c)=v; dic(c)=d
  enddo
  !$omp end parallel do
  status=bad
 end subroutine

 subroutine braids_v96_momentum(nc,nf,nadj,q,vic,owner,neighbour,length,distance,area, &
  offset,faces,signs,dry,threads,flux,rate,status) bind(C)
  integer(c_int),value :: nc,nf,nadj,threads
  real(c_double),value :: dry
  real(c_double),intent(in) :: q(nc,3),vic(nc),length(nf),distance(nf),area(nc)
  integer(c_int),intent(in) :: owner(nf),neighbour(nf),offset(nc+1),faces(nadj),signs(nadj)
  real(c_double),intent(out) :: flux(nf,2),rate(nc,3)
  integer(c_int),intent(out) :: status
  integer :: f,c,l,r,k,slot,bad
  real(8) :: hl,hr,coefficient,value,total
  flux=0.d0; rate=0.d0; status=0
  if(nc<1.or.nf<1.or.nadj<1.or.threads<1.or..not.ieee_is_finite(dry).or.dry<0.d0) then
   status=1; return
  endif
  bad=0
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) &
  !$omp& private(f,l,r,hl,hr,k,coefficient,value) reduction(max:bad)
  do f=1,nf
   l=owner(f)+1; r=neighbour(f)+1
   if(l<1.or.l>nc.or.r<0.or.r>nc) then
    bad=1; cycle
   endif
   if(r==0) cycle ! Zero normal viscous flux at physical boundary.
   hl=q(l,1); hr=q(r,1)
   if(.not.ieee_is_finite(hl).or..not.ieee_is_finite(hr).or.hl<=dry.or.hr<=dry.or. &
      .not.ieee_is_finite(vic(l)).or..not.ieee_is_finite(vic(r)).or.vic(l)<0.d0.or.vic(r)<0.d0.or. &
      .not.ieee_is_finite(length(f)).or..not.ieee_is_finite(distance(f)).or. &
      length(f)<=0.d0.or.distance(f)<=0.d0) then
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
  if(bad/=0) then
   status=bad; return
  endif
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) &
  !$omp& private(c,k,slot,f,total,value) reduction(max:bad)
  do c=1,nc
   if(.not.ieee_is_finite(q(c,1)).or.q(c,1)<=dry.or..not.ieee_is_finite(area(c)).or.area(c)<=0.d0.or. &
      offset(c)<0.or.offset(c+1)>nadj.or.offset(c)>offset(c+1)) then
    bad=1; cycle
   endif
   do k=1,2
    total=0.d0
    do slot=offset(c)+1,offset(c+1)
     f=faces(slot)+1
     if(f<1.or.f>nf.or.abs(signs(slot))/=1) then
      bad=1; cycle
     endif
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
