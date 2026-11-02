! Pure parallel physical-flux replacement, one donor limiter and pressure gather.
module braids_physical_flux_gather_v98_module
 use iso_c_binding
 use omp_lib
 use,intrinsic :: ieee_arithmetic
 implicit none
contains
 subroutine braids_physical_flux_gather_v98(nc,nf,nadj,threads,gravity,dry,dt,q,wet,owner,neighbour, &
   offset,faces,signs,length,area,normal,left0,right0,physical_flux,physical_mask, &
   raw_left,raw_right,limited_left,limited_right,theta,rate,water_flux,counts,status) bind(C)
  integer(c_int),value :: nc,nf,nadj,threads
  real(c_double),value :: gravity,dry,dt
  real(c_double),intent(in) :: q(nc,3),length(nf),area(nc),normal(nf,2),left0(nf,3),right0(nf,3),physical_flux(nf,3)
  integer(c_int),intent(in) :: wet(nc),owner(nf),neighbour(nf),offset(nc+1),faces(nadj),signs(nadj),physical_mask(nf)
  real(c_double),intent(out) :: raw_left(nf,3),raw_right(nf,3),limited_left(nf,3),limited_right(nf,3),theta(nc), &
   rate(nc,3),water_flux(nf)
  integer(c_int),intent(out) :: counts(2),status
  integer :: f,c,k,j,donor,limiting,physical_count,bad
  real(c_double) :: outgoing,available,factor,correction,u,v,h,pressure,w,s,contribution
  bad=0;limiting=0;physical_count=0
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) &
  !$omp& private(f,k) reduction(+:physical_count) reduction(max:bad)
  do f=1,nf
   do k=1,3
    raw_left(f,k)=left0(f,k);raw_right(f,k)=right0(f,k)
    if(physical_mask(f)/=0) then
     raw_left(f,k)=physical_flux(f,k);raw_right(f,k)=physical_flux(f,k)
    endif
   enddo
   if(physical_mask(f)/=0) physical_count=physical_count+1
   if(any(.not.ieee_is_finite(raw_left(f,:))).or.any(.not.ieee_is_finite(raw_right(f,:)))) bad=1
   if(raw_left(f,1)/=raw_right(f,1)) bad=1
  enddo
  !$omp end parallel do
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) &
  !$omp& private(c,j,f,outgoing,available) reduction(max:bad)
  do c=1,nc
   outgoing=0.d0
   do j=offset(c)+1,offset(c+1)
    f=faces(j)+1
    outgoing=outgoing+max(signs(j)*raw_left(f,1),0.d0)*length(f)
   enddo
   available=0.d0
   if(wet(c)/=0) available=q(c,1)*area(c)
   theta(c)=1.d0
   if(outgoing>0.d0.and.dt*outgoing>=available) then
    theta(c)=min(1.d0,available/(dt*outgoing))*(1.d0-32.d0*epsilon(1.d0))
   endif
   if(.not.ieee_is_finite(theta(c)).or.theta(c)<0.d0.or.theta(c)>1.d0) bad=1
  enddo
  !$omp end parallel do
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) &
  !$omp& private(f,k,donor,factor,correction,u,v) reduction(+:limiting)
  do f=1,nf
   limited_left(f,:)=raw_left(f,:);limited_right(f,:)=raw_right(f,:)
   donor=0
   if(raw_left(f,1)>0.d0) donor=owner(f)+1
   if(raw_left(f,1)<0.d0.and.neighbour(f)>=0) donor=neighbour(f)+1
   if(donor>0) then
    factor=theta(donor)
    if(factor<1.d0) then
     limiting=limiting+1;correction=(factor-1.d0)*raw_left(f,1)
     limited_left(f,1)=factor*raw_left(f,1);limited_right(f,1)=limited_left(f,1)
     u=0.d0;v=0.d0
     if(wet(donor)/=0.and.q(donor,1)>dry) then
      u=q(donor,2)/q(donor,1);v=q(donor,3)/q(donor,1)
     endif
     limited_left(f,2)=limited_left(f,2)+correction*u
     limited_left(f,3)=limited_left(f,3)+correction*v
     limited_right(f,2)=limited_right(f,2)+correction*u
     limited_right(f,3)=limited_right(f,3)+correction*v
    endif
   endif
   water_flux(f)=limited_left(f,1)
  enddo
  !$omp end parallel do
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) &
  !$omp& private(c,k,j,f,h,pressure,w,s,contribution) reduction(max:bad)
  do c=1,nc
   h=0.d0
   if(wet(c)/=0) h=q(c,1)
   pressure=.5d0*gravity*h*h
   rate(c,:)=0.d0
   do j=offset(c)+1,offset(c+1)
    f=faces(j)+1;s=real(signs(j),c_double);w=length(f)/area(c)
    do k=1,3
     contribution=limited_left(f,k)
     if(signs(j)<0) contribution=-limited_right(f,k)
     if(k>1) contribution=contribution-s*pressure*normal(f,k-1)
     rate(c,k)=rate(c,k)-contribution*w
    enddo
   enddo
   if(any(.not.ieee_is_finite(rate(c,:)))) bad=1
  enddo
  !$omp end parallel do
  counts(1)=physical_count;counts(2)=limiting;status=bad
 end subroutine
end module
