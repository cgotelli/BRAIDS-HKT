! Additive v98 scalar FCT for a real AMR internal trace.
!
! The selected negative-neighbour faces are neither walls nor physical
! boundaries.  They consume an authenticated exterior scalar trace, enter the
! same low/high-order FCT calculation as ordinary faces, and are reported in a
! separate internal-interface ledger.  Named physical-boundary ledgers retain
! their historical meaning.
module braids_scalar_fct_interface_v98_module
 use iso_c_binding
 use, intrinsic :: ieee_arithmetic
 use omp_lib, only: omp_in_parallel
 use braids_scalar_fct_v96_module, only: smooth_bound_v96
 implicit none
contains

 subroutine braids_scalar_fct_interface_v98(nl,nc,nf,nb,nadj,nbf,ni,dt,dry,threads,mass, &
  depth,next_depth,water_flux,left_state,right_state,boundary,interface_state,hydro,owner, &
  neighbour,boundary_index,interface_active,interface_faces,face_length,area,offset,faces, &
  signs,width,normal,weno,boundary_offset,boundary_faces,result,flux,ledger,interface_ledger,status) bind(C)
  integer(c_int),value :: nl,nc,nf,nb,nadj,nbf,ni,threads
  real(c_double),value :: dt,dry
  real(c_double),intent(in) :: mass(nc,nl),depth(nc),next_depth(nc),water_flux(nf), &
   left_state(nf,nl),right_state(nf,nl),boundary(nf,nl),interface_state(nf,nl)
  integer(c_int),intent(in) :: hydro(nc),owner(nf),neighbour(nf),boundary_index(nf), &
   interface_active(nf),interface_faces(ni),offset(nc+1),faces(nadj),signs(nadj)
  real(c_double),intent(in) :: face_length(nf),area(nc),width(nc,2),normal(nf,2)
  integer(c_int),intent(in) :: weno(nf,2,5),boundary_offset(nb+1),boundary_faces(nbf)
  real(c_double),intent(out) :: result(nc,nl),flux(nf,nl),ledger(nb,nl),interface_ledger(ni,nl)
  integer(c_int),intent(out) :: status
  real(c_double),allocatable :: concentration(:,:),low_flux(:,:),anti(:,:),low_state(:,:), &
   rplus(:,:),rminus(:,:),scaled(:,:)
  integer :: allocstat,lane,cell,face,slot,own,nei,bid,adjacent,bad,index
  logical :: spawn_team
  real(c_double) :: water,low,lower,upper,pp,pm,zfac,correction,factor,tolerance,total
  real(c_double) :: smooth_lower,smooth_upper

  status=0; result=0.d0; flux=0.d0
  if(nb>0) ledger=0.d0
  if(ni>0) interface_ledger=0.d0
  if(nl<1 .or. nc<1 .or. nf<1 .or. nb<0 .or. nadj<1 .or. nbf<0 .or. ni<1 .or. &
     threads<1 .or. .not.ieee_is_finite(dt) .or. dt<=0.d0 .or. &
     .not.ieee_is_finite(dry) .or. dry<0.d0) then
   status=1; return
  endif
  spawn_team=.not.omp_in_parallel()
  allocate(concentration(nc,nl),low_flux(nf,nl),anti(nf,nl),low_state(nc,nl), &
   rplus(nc,nl),rminus(nc,nl),scaled(nf,nl),stat=allocstat)
  if(allocstat/=0) then
   status=2; return
  endif
  concentration=0.d0; low_flux=0.d0; anti=0.d0
  low_state=mass; rplus=1.d0; rminus=1.d0; scaled=0.d0

  bad=0
  !$omp parallel do collapse(2) schedule(static) if(spawn_team) num_threads(threads) &
  !$omp& default(shared) private(lane,cell) reduction(max:bad)
  do lane=1,nl
   do cell=1,nc
    if(.not.ieee_is_finite(mass(cell,lane)) .or. mass(cell,lane)<0.d0 .or. &
       .not.ieee_is_finite(depth(cell)) .or. depth(cell)<0.d0 .or. &
       .not.ieee_is_finite(next_depth(cell)) .or. next_depth(cell)<0.d0) then
     bad=3
    elseif(depth(cell)>dry) then
     concentration(cell,lane)=mass(cell,lane)/depth(cell)
    endif
   enddo
  enddo
  !$omp end parallel do
  if(bad/=0) then
   status=bad; deallocate(concentration,low_flux,anti,low_state,rplus,rminus,scaled); return
  endif

  ! Exact interface inventory: one distinct selected face per interface row.
  bad=0
  do index=1,ni
   face=interface_faces(index)+1
   if(face<1 .or. face>nf) then
    bad=11
   elseif(interface_active(face)/=1 .or. neighbour(face)/=-1 .or. boundary_index(face)/=-1) then
    bad=11
   endif
   do slot=1,index-1
    if(interface_faces(slot)==interface_faces(index)) bad=11
   enddo
  enddo
  do face=1,nf
   if(interface_active(face)/=0 .and. interface_active(face)/=1) bad=11
   if(interface_active(face)==1) then
    if(neighbour(face)/=-1 .or. boundary_index(face)/=-1) bad=11
    if(count(interface_faces==face-1)/=1) bad=11
   endif
  enddo
  if(bad/=0) then
   status=bad; deallocate(concentration,low_flux,anti,low_state,rplus,rminus,scaled); return
  endif

  bad=0
  !$omp parallel do schedule(static) if(spawn_team) num_threads(threads) default(shared) &
  !$omp& private(face,own,nei,bid,water,lane) reduction(max:bad)
  do face=1,nf
   own=owner(face)+1; nei=neighbour(face)+1; bid=boundary_index(face)+1
   water=water_flux(face)
   if(own<1 .or. own>nc .or. nei<0 .or. nei>nc .or. bid<0 .or. bid>nb .or. &
      .not.ieee_is_finite(water) .or. .not.ieee_is_finite(face_length(face)) .or. &
      face_length(face)<=0.d0) then
    bad=4; cycle
   endif
   if(nei==0) then
    if(bid==0 .and. interface_active(face)==0) then
     if(water/=0.d0) bad=max(bad,10)
     cycle
    elseif(bid>0 .and. interface_active(face)/=0) then
     bad=max(bad,11); cycle
    endif
    do lane=1,nl
     if(interface_active(face)==1) then
      if(.not.ieee_is_finite(interface_state(face,lane)) .or. interface_state(face,lane)<0.d0) then
       bad=max(bad,12); cycle
      endif
      if(water<0.d0) then
       low_flux(face,lane)=water*interface_state(face,lane)
       anti(face,lane)=water*right_state(face,lane)-low_flux(face,lane)
      else
       low_flux(face,lane)=water*concentration(own,lane)
       anti(face,lane)=water*left_state(face,lane)-low_flux(face,lane)
      endif
     elseif(water<0.d0) then
      low_flux(face,lane)=water*boundary(face,lane)
     else
      low_flux(face,lane)=water*concentration(own,lane)
     endif
    enddo
   elseif(water>=0.d0) then
    do lane=1,nl
     low_flux(face,lane)=water*concentration(own,lane)
     anti(face,lane)=water*left_state(face,lane)-low_flux(face,lane)
    enddo
   else
    do lane=1,nl
     low_flux(face,lane)=water*concentration(nei,lane)
     anti(face,lane)=water*right_state(face,lane)-low_flux(face,lane)
    enddo
   endif
  enddo
  !$omp end parallel do
  if(bad/=0) then
   status=bad; deallocate(concentration,low_flux,anti,low_state,rplus,rminus,scaled); return
  endif

  bad=0
  !$omp parallel do collapse(2) schedule(static) if(spawn_team) num_threads(threads) &
  !$omp& default(shared) private(lane,cell,low,lower,upper,pp,pm,slot,face,adjacent,zfac, &
  !$omp& correction,tolerance,smooth_lower,smooth_upper) reduction(max:bad)
  do lane=1,nl
   do cell=1,nc
    if(.not.ieee_is_finite(area(cell)) .or. area(cell)<=0.d0 .or. offset(cell)<0 .or. &
       offset(cell+1)>nadj .or. offset(cell)>offset(cell+1)) then
     bad=5; cycle
    endif
    low=mass(cell,lane); lower=concentration(cell,lane); upper=concentration(cell,lane)
    pp=0.d0; pm=0.d0
    do slot=offset(cell)+1,offset(cell+1)
     face=faces(slot)+1
     if(face<1 .or. face>nf .or. abs(signs(slot))/=1) then
      bad=5; cycle
     endif
     if(signs(slot)>0) then
      adjacent=neighbour(face)+1
     else
      adjacent=owner(face)+1
     endif
     if(adjacent>=1) then
      lower=min(lower,concentration(adjacent,lane)); upper=max(upper,concentration(adjacent,lane))
     elseif(interface_active(face)==1 .and. water_flux(face)<0.d0) then
      lower=min(lower,interface_state(face,lane)); upper=max(upper,interface_state(face,lane))
     elseif(boundary_index(face)>=0 .and. water_flux(face)<0.d0) then
      lower=min(lower,boundary(face,lane)); upper=max(upper,boundary(face,lane))
     endif
     zfac=-dt*real(signs(slot),c_double)*face_length(face)/area(cell)
     correction=zfac*anti(face,lane)
     low=low+zfac*low_flux(face,lane)
     pp=pp+max(correction,0.d0); pm=pm+min(correction,0.d0)
    enddo
    tolerance=4096.d0*epsilon(1.d0)*max(1.d0,abs(mass(cell,lane)))
    if(low < -tolerance .or. .not.ieee_is_finite(low)) then
     bad=max(bad,20); cycle
    endif
    low=max(0.d0,low)
    call smooth_bound_v96(nc,nf,nadj,lane,cell,concentration,hydro,owner,neighbour, &
     boundary_index,offset,faces,width,normal,weno,lower,upper,smooth_lower,smooth_upper)
    lower=smooth_lower; upper=smooth_upper; low_state(cell,lane)=low
    if(pp>0.d0) then
     rplus(cell,lane)=min(1.d0,max(0.d0,upper*next_depth(cell)-low-64.d0*tiny(1.d0))/pp)
     rplus(cell,lane)=rplus(cell,lane)*(1.d0-32.d0*epsilon(1.d0))
    endif
    if(pm<0.d0) then
     rminus(cell,lane)=min(1.d0,max(0.d0,low-lower*next_depth(cell)-64.d0*tiny(1.d0))/(-pm))
     rminus(cell,lane)=rminus(cell,lane)*(1.d0-32.d0*epsilon(1.d0))
    endif
   enddo
  enddo
  !$omp end parallel do
  if(bad/=0) then
   status=bad; deallocate(concentration,low_flux,anti,low_state,rplus,rminus,scaled); return
  endif

  !$omp parallel do collapse(2) schedule(static) if(spawn_team) num_threads(threads) &
  !$omp& default(shared) private(lane,face,own,nei,correction,factor)
  do lane=1,nl
   do face=1,nf
    own=owner(face)+1; nei=neighbour(face)+1; correction=anti(face,lane)
    if(correction>0.d0) then
     factor=rminus(own,lane); if(nei>=1) factor=min(factor,rplus(nei,lane))
    elseif(correction<0.d0) then
     factor=rplus(own,lane); if(nei>=1) factor=min(factor,rminus(nei,lane))
    else
     factor=1.d0
    endif
    scaled(face,lane)=factor*correction; flux(face,lane)=low_flux(face,lane)+scaled(face,lane)
   enddo
  enddo
  !$omp end parallel do

  bad=0
  !$omp parallel do collapse(2) schedule(static) if(spawn_team) num_threads(threads) &
  !$omp& default(shared) private(lane,cell,total,slot,face,tolerance) reduction(max:bad)
  do lane=1,nl
   do cell=1,nc
    total=low_state(cell,lane)
    do slot=offset(cell)+1,offset(cell+1)
     face=faces(slot)+1
     total=total-dt*real(signs(slot),c_double)*face_length(face)*scaled(face,lane)/area(cell)
    enddo
    tolerance=4096.d0*epsilon(1.d0)*max(1.d0,abs(mass(cell,lane)))
    if(total < -tolerance .or. .not.ieee_is_finite(total)) then
     bad=max(bad,30)
    else
     result(cell,lane)=max(total,0.d0)
    endif
   enddo
  enddo
  !$omp end parallel do
  if(bad/=0) then
   status=bad; deallocate(concentration,low_flux,anti,low_state,rplus,rminus,scaled); return
  endif

  if(nb>0) then
   bad=0
   if(boundary_offset(1)/=0 .or. boundary_offset(nb+1)/=nbf) bad=6
   do bid=1,nb
    if(boundary_offset(bid)<0 .or. boundary_offset(bid)>boundary_offset(bid+1) .or. &
       boundary_offset(bid+1)>nbf) then
     bad=6; cycle
    endif
    do slot=boundary_offset(bid)+1,boundary_offset(bid+1)
     face=boundary_faces(slot)+1
     if(face<1 .or. face>nf .or. boundary_index(face)/=bid-1) bad=6
    enddo
   enddo
   if(bad/=0) then
    status=bad; deallocate(concentration,low_flux,anti,low_state,rplus,rminus,scaled); return
   endif
   !$omp parallel do collapse(2) schedule(static) if(spawn_team) num_threads(threads) &
   !$omp& default(shared) private(lane,bid,slot,face,total)
   do lane=1,nl
    do bid=1,nb
     total=0.d0
     do slot=boundary_offset(bid)+1,boundary_offset(bid+1)
      face=boundary_faces(slot)+1; total=total-dt*face_length(face)*flux(face,lane)
     enddo
     ledger(bid,lane)=total
    enddo
   enddo
   !$omp end parallel do
  endif
  !$omp parallel do collapse(2) schedule(static) if(spawn_team) num_threads(threads) &
  !$omp& default(shared) private(lane,index,face)
  do lane=1,nl
   do index=1,ni
    face=interface_faces(index)+1
    interface_ledger(index,lane)=-dt*face_length(face)*flux(face,lane)
   enddo
  enddo
  !$omp end parallel do
  deallocate(concentration,low_flux,anti,low_state,rplus,rminus,scaled)
 end subroutine braids_scalar_fct_interface_v98
end module braids_scalar_fct_interface_v98_module
