! Pure native counterpart of BRAIDS v92 combined sediment/fluid diffusion.
! This is BRAIDS code; Delft3D reference sources are never compiled or modified.
module braids_scalar_diffusion_v96_module
 use iso_c_binding
 use, intrinsic :: ieee_arithmetic
 use omp_lib
 implicit none
contains
 subroutine braids_scalar_diffusion_v96(nc,nf,nl,nb,nadj,nbfaces,owner,neighbour,bid, &
  offset,faces,signs,boffset,bfaces,area,distance,dic,mass,depth,boundary,exchange, &
  active,dt,dry,threads,result,amount,ledger,effective,status) bind(C)
  integer(c_int),value :: nc,nf,nl,nb,nadj,nbfaces,threads
  integer(c_int),intent(in) :: owner(nf),neighbour(nf),bid(nf),offset(nc+1)
  integer(c_int),intent(in) :: faces(nadj),signs(nadj),boffset(nb+1),bfaces(nbfaces),active(nf)
  real(c_double),intent(in) :: area(nc),distance(nf),dic(nc),mass(nc,nl),depth(nc)
  real(c_double),intent(in) :: boundary(nf,nl),exchange(nf)
  real(c_double),value :: dt,dry
  real(c_double),intent(out) :: result(nc,nl),amount(nf,nl),ledger(nb,nl),effective(nf)
  integer(c_int),intent(out) :: status
  real(8),allocatable :: theta(:,:)
  real(8) :: hl,hr,left,right,nu,outgoing,inventory,value,total,tolerance
  integer :: f,c,k,l,r,b,slot,donor,bad
  status=0
  if(nc<1.or.nf<1.or.nl<1.or.nb<0.or.nadj<1.or.nbfaces<0.or.threads<1.or. &
     .not.ieee_is_finite(dt).or.dt<=0.d0.or..not.ieee_is_finite(dry).or.dry<0.d0) then
   status=1; return
  endif
  allocate(theta(nc,nl))
  bad=0
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) private(f,l,r,b,k,hl,hr,left,right,nu,value) reduction(max:bad)
  do f=1,nf
   amount(f,:)=0.d0; effective(f)=0.d0
   l=owner(f)+1; r=neighbour(f)+1; b=bid(f)+1
   if(l<1.or.l>nc.or.r<0.or.r>nc.or.b<0.or.b>nb.or. &
      (active(f)/=0.and.active(f)/=1)) then
    bad=1; cycle
   endif
   if(r==0.and.b==0) then
    if(active(f)/=0) bad=1
    cycle
   endif
   hl=depth(l); hr=hl
   if(r>0) hr=depth(r)
   if(.not.ieee_is_finite(hl).or..not.ieee_is_finite(hr).or.hl<0.d0.or.hr<0.d0.or. &
      .not.ieee_is_finite(dic(l)).or.dic(l)<0.d0.or. &
      .not.ieee_is_finite(exchange(f)).or.exchange(f)<0.d0.or. &
      .not.ieee_is_finite(distance(f)).or.distance(f)<=0.d0) then
    bad=1; cycle
   endif
   nu=dic(l)
   if(r>0) then
    if(.not.ieee_is_finite(dic(r)).or.dic(r)<0.d0) then
     bad=1; cycle
    endif
    nu=dic(r)
   endif
   nu=.5d0*(dic(l)+nu)/.7d0
   if(active(f)/=0) effective(f)=nu
   do k=1,nl
    left=0.d0; right=0.d0
    if(hl>0.d0) left=mass(l,k)/hl
    if(r>0) then
     if(hr>0.d0) right=mass(r,k)/hr
    else
     right=boundary(f,k)
    endif
    if(.not.ieee_is_finite(left).or..not.ieee_is_finite(right)) then
     bad=1; cycle
    endif
    if(active(f)/=0.and.min(hl,hr)>dry) then
     value=exchange(f)*nu*(left-right)/distance(f)*dt
     if(.not.ieee_is_finite(value)) then
      bad=1
     else
      amount(f,k)=value
     endif
    endif
   enddo
  enddo
  !$omp end parallel do
  if(bad/=0) then
   status=bad; return
  endif
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) collapse(2) schedule(static) default(shared) private(k,c,slot,f,outgoing,inventory) reduction(max:bad)
  do k=1,nl
   do c=1,nc
    theta(c,k)=1.d0
    if(.not.ieee_is_finite(mass(c,k)).or.mass(c,k)<0.d0.or. &
       .not.ieee_is_finite(area(c)).or.area(c)<=0.d0.or.offset(c)<0.or. &
       offset(c+1)>nadj.or.offset(c)>offset(c+1)) then
     bad=1; cycle
    endif
    outgoing=0.d0
    do slot=offset(c)+1,offset(c+1)
     f=faces(slot)+1
     if(f<1.or.f>nf.or.abs(signs(slot))/=1) then
      bad=1; cycle
     endif
     outgoing=outgoing+max(signs(slot)*amount(f,k),0.d0)
    enddo
    inventory=mass(c,k)*area(c)
    if(outgoing>0.d0.and.outgoing>=inventory) &
     theta(c,k)=min(1.d0,inventory/outgoing)*(1.d0-32.d0*epsilon(1.d0))
   enddo
  enddo
  !$omp end parallel do
  if(bad/=0) then
   status=bad; return
  endif
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) collapse(2) schedule(static) default(shared) private(k,f,donor,value)
  do k=1,nl
   do f=1,nf
    value=amount(f,k); donor=0
    if(value>0.d0) donor=owner(f)+1
    if(value<0.d0) donor=neighbour(f)+1
    if(donor>0) amount(f,k)=value*theta(donor,k)
   enddo
  enddo
  !$omp end parallel do
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) collapse(2) schedule(static) default(shared) private(k,c,slot,f,value,tolerance) reduction(max:bad)
  do k=1,nl
   do c=1,nc
    value=mass(c,k)
    do slot=offset(c)+1,offset(c+1)
     f=faces(slot)+1
     value=value-signs(slot)*amount(f,k)/area(c)
    enddo
    tolerance=8192.d0*epsilon(1.d0)*max(1.d0,abs(mass(c,k)))
    if(.not.ieee_is_finite(value).or.value < -tolerance) then
     bad=1
    endif
    result(c,k)=max(value,0.d0)
   enddo
  enddo
  !$omp end parallel do
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) collapse(2) schedule(static) default(shared) private(k,b,slot,f,total) reduction(max:bad)
  do k=1,nl
   do b=1,nb
    total=0.d0
    if(boffset(b)<0.or.boffset(b+1)>nbfaces.or.boffset(b)>boffset(b+1)) then
     bad=1; ledger(b,k)=0.d0; cycle
    endif
    do slot=boffset(b)+1,boffset(b+1)
     f=bfaces(slot)+1
     if(f<1.or.f>nf) then
      bad=1; cycle
     endif
     total=total-amount(f,k)
    enddo
    ledger(b,k)=total
   enddo
  enddo
  !$omp end parallel do
  status=bad
 end subroutine
end module
