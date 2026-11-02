! Native Q3 face/inventory adapter using unchanged BRAIDS FLOW slope primitives.
! Closed internal wet/dry faces retain their real neighbour but own exact zero.
module braids_q3_face_inventory_v97_module
 use iso_c_binding
 use, intrinsic :: ieee_arithmetic
 use omp_lib
 use braids_flow_bed_slope_face_composer_module, only: braids_flow_bed_slope_face
 implicit none
contains
 subroutine braids_q3_face_inventory_v97(nc,nf,ng,nb,nadj,nbfaces,mode,threads,owner,neighbour,role,bid, &
  offset,faces,signs,boffset,bfaces,area,length,normal,down,vector,stress,depth,d50,mean,hiding,bed, &
  controls,dt,flux,delta,fill,named,absolute,depletion,residual,status) bind(C)
  integer(c_int),value :: nc,nf,ng,nb,nadj,nbfaces,mode,threads
  integer(c_int),intent(in) :: owner(nf),neighbour(nf),role(nf),bid(nf),offset(nc+1),faces(nadj),signs(nadj)
  integer(c_int),intent(in) :: boffset(nb+1),bfaces(nbfaces)
  real(c_double),intent(in) :: area(nc),length(nf),normal(nf,2),down(nf,2),vector(nc,ng,2)
  real(c_double),intent(in) :: stress(nf,ng),depth(nf,ng),d50(ng),mean(nf,ng),hiding(nc,ng),bed(nc,ng),controls(8)
  real(c_double),value :: dt
  real(c_double),intent(out) :: flux(nf,ng),delta(nc,ng),fill(nc,ng),named(nb,ng),absolute(nb,ng),depletion(ng),residual(ng)
  integer(c_int),intent(out) :: status
  real(8),allocatable :: corrections(:,:),partial_delta(:,:),partial_depletion(:,:)
  real(8) :: raw(2),corrected(2),left(2),right(2),zero(2),value,outgoing,candidate,tolerance,total,hidright
  integer :: k,f,c,l,r,slot,b,donor,rawdonor,cs,fs,bad,block,nblocks,start,last
  status=0;bad=0;zero=0.d0
  if(nc<1.or.nf<1.or.ng<1.or.nb<0.or.nadj<1.or.nbfaces<0.or.mode<2.or.mode>4.or.threads<1.or.dt<=0.d0) then
   status=1;return
  endif
  nblocks=(nc+1023)/1024
  allocate(corrections(nc,ng),partial_delta(nblocks,ng),partial_depletion(nblocks,ng))
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) collapse(2) schedule(static) default(shared) private(k,f,l,r,left,right,raw,corrected,value,donor,rawdonor,cs,fs,hidright) reduction(max:bad)
  do k=1,ng
   do f=1,nf
    flux(f,k)=0.d0
    if(role(f)==4) cycle
    l=owner(f)+1;r=neighbour(f)+1
    left=vector(l,k,:);right=left
    hidright=hiding(l,k)
    if(r>0) then
     right=vector(r,k,:);hidright=hiding(r,k)
    endif
    call braids_flow_bed_slope_face(mode,role(f),r>0,left,right,left,zero,normal(f,:),down(f,:), &
      stress(f,k),depth(f,k),d50(k),mean(f,k),hiding(l,k),hidright,1.d0,1.d0, &
      controls(1),controls(2),controls(3),controls(4),controls(5),controls(6),controls(7),controls(8), &
      raw,corrected,value,rawdonor,donor,cs,fs)
    if(cs/=0.or.fs/=0.or..not.ieee_is_finite(value)) then
     bad=1
    else
     flux(f,k)=value
    endif
   enddo
  enddo
  !$omp end parallel do
  if(bad/=0) then
   status=bad;return
  endif
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) collapse(2) schedule(static) default(shared) private(k,c,slot,f,outgoing,value)
  do k=1,ng
   do c=1,nc
    outgoing=0.d0;fill(c,k)=1.d0
    do slot=offset(c)+1,offset(c+1)
     f=faces(slot)+1
     if(role(f)/=0.and.role(f)/=3) cycle
     value=signs(slot)*flux(f,k)
     if(value>0.d0) outgoing=outgoing+value*length(f)*dt/area(c)
    enddo
    if(bed(c,k)==0.d0) fill(c,k)=0.d0
    if(outgoing>0.d0) fill(c,k)=min(1.d0,bed(c,k)/outgoing)
   enddo
  enddo
  !$omp end parallel do
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) collapse(2) schedule(static) default(shared) private(k,f,c)
  do k=1,ng
   do f=1,nf
    c=owner(f)+1
    if(flux(f,k)<0.d0.and.role(f)==0) c=neighbour(f)+1
    if(c>0) flux(f,k)=flux(f,k)*fill(c,k)
   enddo
  enddo
  !$omp end parallel do
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) collapse(2) schedule(static) default(shared) private(k,c,slot,f,value,candidate,tolerance) reduction(max:bad)
  do k=1,ng
   do c=1,nc
    value=0.d0
    do slot=offset(c)+1,offset(c+1)
     f=faces(slot)+1
     value=value-signs(slot)*flux(f,k)*length(f)*dt/area(c)
    enddo
    candidate=bed(c,k)+value;tolerance=4096.d0*epsilon(1.d0)*max(1.d0,abs(bed(c,k)))
    if(candidate < -tolerance.or..not.ieee_is_finite(candidate)) bad=1
    corrections(c,k)=0.d0
    if(candidate<0.d0) then
     corrections(c,k)=-bed(c,k)-value;value=-bed(c,k)
    endif
    delta(c,k)=value
   enddo
  enddo
  !$omp end parallel do
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) collapse(2) schedule(static) default(shared) private(k,b,slot,f,value,total)
  do k=1,ng
   do b=1,nb
    total=0.d0;value=0.d0
    do slot=boffset(b)+1,boffset(b+1)
     f=bfaces(slot)+1
     total=total+flux(f,k)*length(f)*dt
     value=value+abs(flux(f,k)*length(f)*dt)
    enddo
    named(b,k)=total;absolute(b,k)=value
   enddo
  enddo
  !$omp end parallel do
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) collapse(2) schedule(static) default(shared) private(k,block,c,start,last,value,total)
  do k=1,ng
   do block=1,nblocks
    start=(block-1)*1024+1;last=min(block*1024,nc);total=0.d0;value=0.d0
    do c=start,last
     total=total+delta(c,k)*area(c);value=value+corrections(c,k)*area(c)
    enddo
    partial_delta(block,k)=total;partial_depletion(block,k)=value
   enddo
  enddo
  !$omp end parallel do
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) private(k,block,b,value,total) reduction(max:bad)
  do k=1,ng
   total=0.d0;value=0.d0
   do block=1,nblocks
    total=total+partial_delta(block,k);value=value+partial_depletion(block,k)
   enddo
   depletion(k)=value
   do b=1,nb
    total=total+named(b,k)
   enddo
   residual(k)=total-value
   if(.not.ieee_is_finite(residual(k)).or..not.ieee_is_finite(value)) bad=1
  enddo
  !$omp end parallel do
  status=bad
 end subroutine
end module
