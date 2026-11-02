! Native OpenMP conservative four-child refine/restrict batch for BRAIDS v98.
! This is a BRAIDS implementation.  It neither includes nor modifies Delft3D.
module braids_amr_zero_time_transfer_v98_module
 use iso_c_binding
 use omp_lib
 use, intrinsic :: ieee_arithmetic
 implicit none
 integer, parameter :: nchild=4
contains

 pure real(c_double) function lane_tolerance(values,target) result(tol)
  real(c_double),intent(in) :: values(nchild),target
  tol=2048.d0*epsilon(1.d0)*max(1.d0,abs(target),maxval(abs(values)))
 end function lane_tolerance

 subroutine close_lane(target,area,value,index,nonnegative,bad)
  real(c_double),intent(in) :: target,area(nchild)
  real(c_double),intent(inout) :: value(nchild)
  integer,intent(in) :: index
  logical,intent(in) :: nonnegative
  logical,intent(inout) :: bad
  real(c_double) :: current,residual,tol
  integer :: k
  current=0.d0
  do k=1,nchild
   current=current+value(k)*area(k)
  enddo
  residual=target-current
  value(index)=value(index)+residual/area(index)
  current=0.d0
  do k=1,nchild
   current=current+value(k)*area(k)
  enddo
  residual=target-current
  value(index)=value(index)+residual/area(index)
  tol=lane_tolerance(value,target)
  if(nonnegative) then
   if(any(value < -tol)) bad=.true.
   do k=1,nchild
    if(value(k)<0.d0) value(k)=0.d0
   enddo
  endif
  current=0.d0
  do k=1,nchild
   current=current+value(k)*area(k)
  enddo
  if(abs(current-target)>2048.d0*epsilon(1.d0)*max(1.d0,abs(target),abs(current))) bad=.true.
 end subroutine close_lane

 subroutine stable_threshold_order(bed,dry,order)
  real(c_double),intent(in) :: bed(nchild),dry
  integer,intent(out) :: order(nchild)
  integer :: i,j,key
  real(c_double) :: key_value
  do i=1,nchild
   order(i)=i
  enddo
  do i=2,nchild
   key=order(i);key_value=bed(key)+dry;j=i-1
   do while(j>=1)
    if(bed(order(j))+dry<=key_value) exit
    order(j+1)=order(j);j=j-1
   enddo
   order(j+1)=key
  enddo
 end subroutine stable_threshold_order

 subroutine stage_fill(volume,bed,area,dry,depth,bad)
  real(c_double),intent(in) :: volume,bed(nchild),area(nchild),dry
  real(c_double),intent(out) :: depth(nchild)
  logical,intent(inout) :: bad
  integer :: order(nchild),count,k,index,deepest
  real(c_double) :: eta,numerator,denominator,residual,current,tol
  logical :: lower_ok,upper_ok,found
  depth=0.d0
  if(volume==0.d0) return
  call stable_threshold_order(bed,dry,order)
  found=.false.;eta=0.d0;count=0
  do index=1,nchild
   numerator=volume;denominator=0.d0
   do k=1,index
    numerator=numerator+area(order(k))*bed(order(k))
    denominator=denominator+area(order(k))
   enddo
   eta=numerator/denominator
   lower_ok=.true.
   do k=1,index
    if(eta<=bed(order(k))+dry) lower_ok=.false.
   enddo
   upper_ok=(index==nchild)
   if(index<nchild) upper_ok=eta<=bed(order(index+1))+dry
   if(lower_ok.and.upper_ok) then
    found=.true.;count=index;exit
   endif
  enddo
  if(.not.found) then
   bad=.true.;return
  endif
  do k=1,count
   depth(order(k))=eta-bed(order(k))
  enddo
  ! Threshold sorting is also bed sorting because dry is uniform.  The first
  ! selected child is therefore the stable deepest-child correction owner.
  deepest=order(1)
  current=0.d0
  do k=1,nchild
   current=current+depth(k)*area(k)
  enddo
  residual=volume-current;depth(deepest)=depth(deepest)+residual/area(deepest)
  current=0.d0
  do k=1,nchild
   current=current+depth(k)*area(k)
  enddo
  residual=volume-current;depth(deepest)=depth(deepest)+residual/area(deepest)
  tol=2048.d0*epsilon(1.d0)*max(1.d0,abs(volume),maxval(abs(depth)))
  current=0.d0
  do k=1,nchild
   current=current+depth(k)*area(k)
   if(depth(k)<0.d0) bad=.true.
   if(depth(k)>0.d0.and.depth(k)<=dry) bad=.true.
  enddo
  if(abs(current-volume)>tol) bad=.true.
 end subroutine stage_fill

 subroutine braids_amr_zero_time_transfer_v98(np,ng,threads,dry,parent_area,child_area,child_bed_elevation, &
  parent_q,parent_suspended,parent_bed,parent_fluid,child_q,child_suspended,child_bed,child_fluid, &
  restricted_q,restricted_suspended,restricted_bed,restricted_fluid,parent_status) bind(C)
  integer(c_int),value :: np,ng,threads
  real(c_double),value :: dry
  real(c_double),intent(in) :: parent_area(np),child_area(nchild,np),child_bed_elevation(nchild,np)
  real(c_double),intent(in) :: parent_q(np,3),parent_suspended(np,ng),parent_bed(np,ng),parent_fluid(np,2)
  real(c_double),intent(out) :: child_q(nchild,np,3),child_suspended(nchild,np,ng), &
   child_bed(nchild,np,ng),child_fluid(nchild,np,2),restricted_q(np,3), &
   restricted_suspended(np,ng),restricted_bed(np,ng),restricted_fluid(np,2)
  integer(c_int),intent(out) :: parent_status(np)
  integer :: p,k,j,first_wet
  real(c_double) :: area(nchild),bedz(nchild),depth(nchild),lane(nchild),area0,h,target,current,tol
  logical :: bad
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) &
  !$omp& private(p,k,j,first_wet,area,bedz,depth,lane,area0,h,target,current,tol,bad)
  do p=1,np
   bad=.false.;parent_status(p)=0;depth=0.d0
   area=child_area(:,p);bedz=child_bed_elevation(:,p);area0=parent_area(p);h=parent_q(p,1)
   child_q(:,p,:)=0.d0;child_suspended(:,p,:)=0.d0;child_bed(:,p,:)=0.d0;child_fluid(:,p,:)=0.d0
   restricted_q(p,:)=0.d0;restricted_suspended(p,:)=0.d0;restricted_bed(p,:)=0.d0;restricted_fluid(p,:)=0.d0
   if(.not.ieee_is_finite(area0).or.area0<=0.d0.or.any(.not.ieee_is_finite(area)).or.any(area<=0.d0).or. &
      any(.not.ieee_is_finite(bedz)).or..not.ieee_is_finite(h).or.h<0.d0.or. &
      any(.not.ieee_is_finite(parent_q(p,:))).or.any(.not.ieee_is_finite(parent_suspended(p,:))).or. &
      any(.not.ieee_is_finite(parent_bed(p,:))).or.any(.not.ieee_is_finite(parent_fluid(p,:))).or. &
      any(parent_suspended(p,:)<0.d0).or.any(parent_bed(p,:)<0.d0).or.parent_fluid(p,2)<0.d0) bad=.true.
   current=sum(area);tol=2048.d0*epsilon(1.d0)*max(1.d0,abs(area0),maxval(abs(area)))
   if(abs(current-area0)>tol) bad=.true.
   if(h<=dry.and.(any(parent_q(p,2:3)/=0.d0).or.any(parent_suspended(p,:)/=0.d0).or. &
      any(parent_fluid(p,:)/=0.d0))) bad=.true.
   if(.not.bad) call stage_fill(h*area0,bedz,area,dry,depth,bad)
   child_q(:,p,1)=depth
   first_wet=1
   do k=1,nchild
    if(depth(k)>dry) then
     first_wet=k;exit
    endif
   enddo
   do j=1,2
    if(h>dry) then
     lane=(parent_q(p,j+1)/h)*depth
    else
     lane=0.d0
    endif
    target=parent_q(p,j+1)*area0
    call close_lane(target,area,lane,first_wet,.false.,bad)
    child_q(:,p,j+1)=lane
   enddo
   do j=1,ng
    if(h>dry) then
     lane=(parent_suspended(p,j)/h)*depth
    else
     lane=0.d0
    endif
    target=parent_suspended(p,j)*area0
    call close_lane(target,area,lane,first_wet,.true.,bad)
    child_suspended(:,p,j)=lane
    lane=parent_bed(p,j)
    target=parent_bed(p,j)*area0
    call close_lane(target,area,lane,nchild,.true.,bad)
    child_bed(:,p,j)=lane
   enddo
   do j=1,2
    if(h>dry) then
     lane=(parent_fluid(p,j)/h)*depth
    else
     lane=0.d0
    endif
    target=parent_fluid(p,j)*area0
    call close_lane(target,area,lane,first_wet,.false.,bad)
    child_fluid(:,p,j)=lane
   enddo
   ! Endpoint dry cleanup transfers each suspended class to its own bed class.
   do k=1,nchild
    if(depth(k)<=dry) then
     child_q(k,p,2:3)=0.d0;child_fluid(k,p,:)=0.d0
     do j=1,ng
      child_bed(k,p,j)=child_bed(k,p,j)+child_suspended(k,p,j)
      child_suspended(k,p,j)=0.d0
     enddo
    endif
   enddo
   do j=1,3
    current=0.d0
    do k=1,nchild
     current=current+child_q(k,p,j)*area(k)
    enddo
    restricted_q(p,j)=current/area0
   enddo
   do j=1,ng
    current=0.d0
    do k=1,nchild
     current=current+child_suspended(k,p,j)*area(k)
    enddo
    restricted_suspended(p,j)=current/area0
    current=0.d0
    do k=1,nchild
     current=current+child_bed(k,p,j)*area(k)
    enddo
    restricted_bed(p,j)=current/area0
   enddo
   do j=1,2
    current=0.d0
    do k=1,nchild
     current=current+child_fluid(k,p,j)*area(k)
    enddo
    restricted_fluid(p,j)=current/area0
   enddo
   tol=2048.d0*epsilon(1.d0)*max(1.d0,abs(h),abs(restricted_q(p,1)))
   if(abs(restricted_q(p,1)-h)>tol) bad=.true.
   do j=1,2
    tol=2048.d0*epsilon(1.d0)*max(1.d0,abs(parent_q(p,j+1)),abs(restricted_q(p,j+1)))
    if(abs(restricted_q(p,j+1)-parent_q(p,j+1))>tol) bad=.true.
    tol=2048.d0*epsilon(1.d0)*max(1.d0,abs(parent_fluid(p,j)),abs(restricted_fluid(p,j)))
    if(abs(restricted_fluid(p,j)-parent_fluid(p,j))>tol) bad=.true.
   enddo
   do j=1,ng
    tol=2048.d0*epsilon(1.d0)*max(1.d0,abs(parent_suspended(p,j)),abs(parent_bed(p,j)), &
     abs(restricted_suspended(p,j)),abs(restricted_bed(p,j)))
    if(abs(restricted_suspended(p,j)+restricted_bed(p,j)-parent_suspended(p,j)-parent_bed(p,j))>tol) bad=.true.
   enddo
   if(bad) parent_status(p)=1
  enddo
  !$omp end parallel do
 end subroutine braids_amr_zero_time_transfer_v98
end module braids_amr_zero_time_transfer_v98_module
