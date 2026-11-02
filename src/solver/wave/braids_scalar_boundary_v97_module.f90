! BRAIDS combined-lane boundary selection, memory and continuity bookkeeping.
! Pure OpenMP batch: no accepted publication, cache owner, or timestep clock.
module braids_scalar_boundary_v97_module
 use iso_c_binding
 use, intrinsic :: ieee_arithmetic
 use omp_lib
 implicit none
contains
 subroutine braids_scalar_boundary_v97(nc,nf,nl,nb,nadj,nbfaces,owner,neighbour,bid, &
  offset,faces,signs,boffset,bfaces,area,length,h0,h1,mass,prescribed,water, &
  return_time,last,clock,neumann,dt,dry,threads,boundary,amount,next_last, &
  next_clock,ledger,divergence,water_ledger,continuity,status) bind(C)
  integer(c_int),value :: nc,nf,nl,nb,nadj,nbfaces,threads
  integer(c_int),intent(in) :: owner(nf),neighbour(nf),bid(nf),offset(nc+1)
  integer(c_int),intent(in) :: faces(nadj),signs(nadj),boffset(nb+1),bfaces(nbfaces),neumann(nl)
  real(c_double),intent(in) :: area(nc),length(nf),h0(nc),h1(nc),mass(nc,nl),prescribed(nb,nl)
  real(c_double),intent(in) :: water(nf),return_time(nb,nl),last(nf,nl),clock(nf,nl)
  real(c_double),value :: dt,dry
  real(c_double),intent(out) :: boundary(nf,nl),amount(nf,nl),next_last(nf,nl),next_clock(nf,nl)
  real(c_double),intent(out) :: ledger(nb,nl),divergence(nc),water_ledger(nb),continuity(nc)
  integer(c_int),intent(out) :: status
  real(8) :: t,ci,p,qin,bval,next,weight,total,tolerance
  integer :: f,c,k,b,slot,bad
  status=0;bad=0
  if(nc<1.or.nf<1.or.nl<1.or.nb<0.or.nadj<1.or.nbfaces<0.or.threads<1.or. &
     .not.ieee_is_finite(dt).or.dt<=0.d0.or..not.ieee_is_finite(dry).or.dry<0.d0) then
   status=1;return
  endif
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) collapse(2) schedule(static) default(shared) private(k,f,c,b,t,ci,p,qin,bval,next,weight,tolerance) reduction(max:bad)
  do k=1,nl
   do f=1,nf
    boundary(f,k)=0.d0;amount(f,k)=0.d0
    next_last(f,k)=last(f,k);next_clock(f,k)=clock(f,k)
    c=owner(f)+1;b=bid(f)+1
    if(c<1.or.c>nc.or.neighbour(f)<-1.or.neighbour(f)>=nc.or.b<0.or.b>nb) then
     bad=1;cycle
    endif
    if(neighbour(f)>=0.or.b==0) cycle
    ci=0.d0
    if(h0(c)>dry) ci=mass(c,k)/h0(c)
    p=prescribed(b,k);t=return_time(b,k);qin=-water(f)*length(f);bval=p
    tolerance=256.d0*epsilon(1.d0)*max(1.d0,t)
    if(clock(f,k)<0.d0.or.last(f,k)<0.d0.or. &
       (t>0.d0.and.clock(f,k)>t+tolerance).or.(t<=0.d0.and.clock(f,k)/=0.d0)) then
     bad=1;cycle
    endif
    if(t<0.d0) then
     bval=ci;next_clock(f,k)=0.d0
    elseif(t>0.d0) then
     if(qin<0.d0) then
      next_last(f,k)=ci;next_clock(f,k)=t;bval=ci
     elseif(qin>0.d0) then
      next=max(clock(f,k)-dt,0.d0);next_clock(f,k)=next
      weight=.5d0*(cos(acos(-1.d0)*(next/t))+1.d0)
      bval=last(f,k)+weight*(p-last(f,k))
     else
      next_clock(f,k)=0.d0
     endif
    endif
    if(neumann(k)/=0) bval=max(0.d0,ci)
    boundary(f,k)=bval
    if(qin>0.d0) then
     amount(f,k)=qin*bval*dt
    else
     amount(f,k)=qin*ci*dt
    endif
    if(.not.ieee_is_finite(bval).or.bval<0.d0.or..not.ieee_is_finite(amount(f,k))) bad=1
   enddo
  enddo
  !$omp end parallel do
  if(bad/=0) then
   status=bad;return
  endif
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) private(c,slot,f,total) reduction(max:bad)
  do c=1,nc
   total=0.d0
   do slot=offset(c)+1,offset(c+1)
    f=faces(slot)+1
    total=total+signs(slot)*length(f)*water(f)/area(c)
   enddo
   divergence(c)=total;continuity(c)=h1(c)-h0(c)+dt*total
   if(.not.ieee_is_finite(total).or..not.ieee_is_finite(continuity(c))) bad=1
  enddo
  !$omp end parallel do
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) collapse(2) schedule(static) default(shared) private(k,b,slot,f,total) reduction(max:bad)
  do k=1,nl
   do b=1,nb
    total=0.d0
    do slot=boffset(b)+1,boffset(b+1)
     f=bfaces(slot)+1;total=total+amount(f,k)
    enddo
    ledger(b,k)=total
    if(.not.ieee_is_finite(total)) bad=1
   enddo
  enddo
  !$omp end parallel do
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) private(b,slot,f,total) reduction(max:bad)
  do b=1,nb
   total=0.d0
   do slot=boffset(b)+1,boffset(b+1)
    f=bfaces(slot)+1;total=total-water(f)*length(f)*dt
   enddo
   water_ledger(b)=total
   if(.not.ieee_is_finite(total)) bad=1
  enddo
  !$omp end parallel do
  status=bad
 end subroutine
end module
