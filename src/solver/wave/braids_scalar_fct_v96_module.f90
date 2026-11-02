! Pure native OpenMP transcription of the accepted v92 scalar FCT operator.
! Lanes are generic conserved scalars: sediment fractions and active T/S lanes
! are advanced together by one call.  Cell-owned gathers avoid atomics.
module braids_scalar_fct_v96_module
 use iso_c_binding
 use, intrinsic :: ieee_arithmetic
 use omp_lib, only: omp_in_parallel
 implicit none
contains

 subroutine smooth_bound_v96(nc,nf,nadj,lane,cell,concentration,hydro,owner, &
  neighbour,boundary_index,offset,faces,width,normal,weno,lower_in,upper_in, &
  lower_out,upper_out)
  integer(c_int),intent(in) :: nc,nf,nadj,lane,cell
  real(c_double),intent(in) :: concentration(nc,*),width(nc,2),normal(nf,2)
  integer(c_int),intent(in) :: hydro(nc),owner(nf),neighbour(nf), &
   boundary_index(nf),offset(nc+1),faces(nadj),weno(nf,2,5)
  real(c_double),intent(in) :: lower_in,upper_in
  real(c_double),intent(out) :: lower_out,upper_out
  integer :: slot,face,axis,side,i,adjacent,ids(5),kind,extremum_kind
  logical :: found,same_sign
  real(c_double) :: lo,hi,scale,tolerance,values(5),z(5),value_scale
  real(c_double) :: d20,d21,d22,amin,amax,slope,vertex,point,candidate

  lower_out=lower_in; upper_out=upper_in
  if(hydro(cell)/=0 .or. width(cell,1)<=0.d0 .or. width(cell,2)<=0.d0) return
  do slot=offset(cell)+1,offset(cell+1)
   face=faces(slot)+1
   if(boundary_index(face)>=0) return
  enddo
  lo=lower_in; hi=upper_in
  scale=max(abs(lower_in),abs(upper_in),tiny(1.d0))
  tolerance=4096.d0*epsilon(1.d0)*scale
  extremum_kind=1
  do axis=1,2
   found=.false.; ids=0
   do slot=offset(cell)+1,offset(cell+1)
    face=faces(slot)+1
    if(abs(normal(face,axis))/=1.d0) cycle
    if(cell-1==owner(face)) then
     side=1
    else
     side=2
    endif
    if(weno(face,side,1)>=0) then
     do i=1,5
      ids(i)=weno(face,side,i)+1
     enddo
     found=.true.
    endif
   enddo
   if(.not.found) then
    do slot=offset(cell)+1,offset(cell+1)
     face=faces(slot)+1
     if(abs(normal(face,axis))/=1.d0) cycle
     if(cell-1==owner(face)) then
      adjacent=neighbour(face)+1
     else
      adjacent=owner(face)+1
     endif
     if(adjacent<1) cycle
     if(hydro(adjacent)/=0 .or. &
        abs(concentration(adjacent,lane)-concentration(cell,lane))>tolerance) then
      lower_out=lower_in; upper_out=upper_in; return
     endif
    enddo
    cycle
   endif
   do i=1,5
    if(ids(i)<1 .or. ids(i)>nc .or. hydro(ids(i))/=0) then
     lower_out=lower_in; upper_out=upper_in; return
    endif
    values(i)=concentration(ids(i),lane)
   enddo
   value_scale=maxval(abs(values))
   if(value_scale==0.d0) cycle
   z=values/value_scale
   if(maxval(z)-minval(z)<=4096.d0*epsilon(1.d0)) cycle
   if((z(3)-z(2))*(z(4)-z(3))>0.d0) then
    lower_out=lower_in; upper_out=upper_in; return
   endif
   d20=z(1)-2.d0*z(2)+z(3)
   d21=z(2)-2.d0*z(3)+z(4)
   d22=z(3)-2.d0*z(4)+z(5)
   amin=min(abs(d20),abs(d21),abs(d22))
   amax=max(abs(d20),abs(d21),abs(d22))
   same_sign=(d20>0.d0 .and. d21>0.d0 .and. d22>0.d0) .or. &
             (d20<0.d0 .and. d21<0.d0 .and. d22<0.d0)
   if(amin<=4096.d0*epsilon(1.d0) .or. .not.same_sign .or. &
      amin<0.5d0*amax) then
    lower_out=lower_in; upper_out=upper_in; return
   endif
   slope=0.5d0*(z(4)-z(2))
   vertex=max(-0.5d0,min(0.5d0,-slope/d21))
   point=z(3)+slope*vertex+0.5d0*d21*vertex*vertex-d21/24.d0
   if(d21<0.d0) then
    kind=2
    candidate=value_scale*(z(3)+2.d0*max(point-z(3),0.d0))
   else
    kind=3
    candidate=max(0.d0,value_scale*(z(3)-2.d0*max(z(3)-point,0.d0)))
   endif
   if(extremum_kind/=1 .and. kind/=extremum_kind) then
    lower_out=lower_in; upper_out=upper_in; return
   endif
   extremum_kind=kind
   if(kind==2) then
    hi=max(hi,candidate)
   else
    lo=min(lo,candidate)
   endif
  enddo
  lower_out=max(0.d0,lo); upper_out=hi
 end subroutine smooth_bound_v96

 subroutine braids_scalar_fct_v96(nl,nc,nf,nb,nadj,nbf,dt,dry,threads,mass, &
  depth,next_depth,water_flux,left_state,right_state,boundary,hydro,owner, &
  neighbour,boundary_index,face_length,area,offset,faces,signs,width,normal, &
  weno,boundary_offset,boundary_faces,result,flux,ledger,status) bind(C)
  integer(c_int),value :: nl,nc,nf,nb,nadj,nbf,threads
  real(c_double),value :: dt,dry
  real(c_double),intent(in) :: mass(nc,nl),depth(nc),next_depth(nc), &
   water_flux(nf),left_state(nf,nl),right_state(nf,nl),boundary(nf,nl)
  integer(c_int),intent(in) :: hydro(nc),owner(nf),neighbour(nf), &
   boundary_index(nf),offset(nc+1),faces(nadj),signs(nadj)
  real(c_double),intent(in) :: face_length(nf),area(nc),width(nc,2),normal(nf,2)
  integer(c_int),intent(in) :: weno(nf,2,5)
  integer(c_int),intent(in) :: boundary_offset(nb+1),boundary_faces(nbf)
  real(c_double),intent(out) :: result(nc,nl),flux(nf,nl),ledger(nb,nl)
  integer(c_int),intent(out) :: status
  real(c_double),allocatable :: concentration(:,:),low_flux(:,:),anti(:,:), &
   low_state(:,:),rplus(:,:),rminus(:,:),scaled(:,:)
  integer :: allocstat,lane,cell,face,slot,own,nei,bid,adjacent,bad
  logical :: spawn_team
  real(c_double) :: water,low,lower,upper,pp,pm,zfac,correction,factor
  real(c_double) :: tolerance,total,smooth_lower,smooth_upper

  status=0; result=0.d0; flux=0.d0
  if(nb>0) ledger=0.d0
  if(nl<1 .or. nc<1 .or. nf<1 .or. nb<0 .or. nadj<1 .or. nbf<0 .or. &
     threads<1 .or. &
     .not.ieee_is_finite(dt) .or. dt<=0.d0 .or. &
     .not.ieee_is_finite(dry) .or. dry<0.d0) then
   status=1; return
  endif
  spawn_team=.not.omp_in_parallel()
  allocate(concentration(nc,nl),low_flux(nf,nl),anti(nf,nl), &
   low_state(nc,nl),rplus(nc,nl),rminus(nc,nl),scaled(nf,nl),stat=allocstat)
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
    if(bid==0) then
     if(water/=0.d0) bad=max(bad,10)
     cycle
    endif
    do lane=1,nl
     if(water<0.d0) then
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
  !$omp& default(shared) private(lane,cell,low,lower,upper,pp,pm,slot,face, &
  !$omp& adjacent,zfac,correction,tolerance,smooth_lower,smooth_upper) reduction(max:bad)
  do lane=1,nl
   do cell=1,nc
    if(.not.ieee_is_finite(area(cell)) .or. area(cell)<=0.d0 .or. &
       offset(cell)<0 .or. offset(cell+1)>nadj .or. offset(cell)>offset(cell+1)) then
     bad=5; cycle
    endif
    low=mass(cell,lane); lower=concentration(cell,lane)
    upper=concentration(cell,lane); pp=0.d0; pm=0.d0
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
      lower=min(lower,concentration(adjacent,lane))
      upper=max(upper,concentration(adjacent,lane))
     elseif(boundary_index(face)>=0 .and. water_flux(face)<0.d0) then
      lower=min(lower,boundary(face,lane))
      upper=max(upper,boundary(face,lane))
     endif
     zfac=-dt*real(signs(slot),c_double)*face_length(face)/area(cell)
     correction=zfac*anti(face,lane)
     low=low+zfac*low_flux(face,lane)
     pp=pp+max(correction,0.d0)
     pm=pm+min(correction,0.d0)
    enddo
    tolerance=4096.d0*epsilon(1.d0)*max(1.d0,abs(mass(cell,lane)))
    if(low < -tolerance .or. .not.ieee_is_finite(low)) then
     bad=max(bad,20); cycle
    endif
    low=max(0.d0,low)
    call smooth_bound_v96(nc,nf,nadj,lane,cell,concentration,hydro,owner, &
     neighbour,boundary_index,offset,faces,width,normal,weno,lower,upper, &
     smooth_lower,smooth_upper)
    lower=smooth_lower; upper=smooth_upper
    low_state(cell,lane)=low
    if(pp>0.d0) then
     rplus(cell,lane)=min(1.d0,max(0.d0,upper*next_depth(cell)-low- &
                         64.d0*tiny(1.d0))/pp)
     rplus(cell,lane)=rplus(cell,lane)*(1.d0-32.d0*epsilon(1.d0))
    endif
    if(pm<0.d0) then
     rminus(cell,lane)=min(1.d0,max(0.d0,low-lower*next_depth(cell)- &
                          64.d0*tiny(1.d0))/(-pm))
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
     factor=rminus(own,lane)
     if(nei>=1) factor=min(factor,rplus(nei,lane))
    elseif(correction<0.d0) then
     factor=rplus(own,lane)
     if(nei>=1) factor=min(factor,rminus(nei,lane))
    else
     factor=1.d0
    endif
    scaled(face,lane)=factor*correction
    flux(face,lane)=low_flux(face,lane)+scaled(face,lane)
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
     total=total-dt*real(signs(slot),c_double)*face_length(face)* &
           scaled(face,lane)/area(cell)
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
     if(face<1 .or. face>nf) then
      bad=6
     elseif(boundary_index(face)/=bid-1) then
      bad=6
     endif
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
      face=boundary_faces(slot)+1
      total=total-dt*face_length(face)*flux(face,lane)
     enddo
     ledger(bid,lane)=total
    enddo
   enddo
   !$omp end parallel do
  endif
  deallocate(concentration,low_flux,anti,low_state,rplus,rminus,scaled)
 end subroutine braids_scalar_fct_v96
end module braids_scalar_fct_v96_module
