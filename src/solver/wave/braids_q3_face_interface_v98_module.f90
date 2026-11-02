! Additive v98 Q3 face/inventory operator for authenticated AMR traces.
module braids_q3_face_interface_v98_module
 use iso_c_binding
 use, intrinsic :: ieee_arithmetic
 use omp_lib
 use braids_flow_bed_slope_face_composer_module, only: braids_flow_bed_slope_face
 implicit none
contains
 subroutine braids_q3_face_interface_v98(nc,nf,ng,nb,nadj,nbfaces,ni,mode,threads, &
  owner,neighbour,role,bid,interface_active,interface_faces,offset,faces,signs,boffset,bfaces, &
  area,length,normal,down,vector,trace_vector,stress,depth,d50,mean,hiding,trace_hiding, &
  trace_availability,bed,controls,dt,flux,delta,fill,named,absolute,interface_transfer, &
  depletion,residual,status) bind(C)
  integer(c_int),value :: nc,nf,ng,nb,nadj,nbfaces,ni,mode,threads
  integer(c_int),intent(in) :: owner(nf),neighbour(nf),role(nf),bid(nf),interface_active(nf), &
   interface_faces(ni),offset(nc+1),faces(nadj),signs(nadj),boffset(nb+1),bfaces(nbfaces)
  real(c_double),intent(in) :: area(nc),length(nf),normal(nf,2),down(nf,2),vector(nc,ng,2), &
   trace_vector(ni,ng,2),stress(nf,ng),depth(nf,ng),d50(ng),mean(nf,ng),hiding(nc,ng), &
   trace_hiding(ni,ng),trace_availability(ni,ng),bed(nc,ng),controls(8)
  real(c_double),value :: dt
  real(c_double),intent(out) :: flux(nf,ng),delta(nc,ng),fill(nc,ng),named(nb,ng), &
   absolute(nb,ng),interface_transfer(ni,ng),depletion(ng),residual(ng)
  integer(c_int),intent(out) :: status
  real(8),allocatable :: corrections(:,:),partial_delta(:,:),partial_depletion(:,:)
  real(8) :: raw(2),corrected(2),left(2),right(2),zero(2),value,outgoing,candidate, &
   tolerance,total,hidright,external_availability
  integer :: k,f,c,l,r,slot,b,donor,rawdonor,cs,fs,bad,block,nblocks,start,last,index
  logical :: is_interface
  status=0;bad=0;zero=0.d0
  if(nc<1.or.nf<1.or.ng<1.or.nb<0.or.nadj<1.or.nbfaces<0.or.ni<1.or. &
     mode<2.or.mode>4.or.threads<1.or.dt<=0.d0.or..not.ieee_is_finite(dt)) then
   status=1;return
  endif
  do index=1,ni
   f=interface_faces(index)+1
   if(f<1.or.f>nf) bad=2
   if(f>=1.and.f<=nf) then
    if(interface_active(f)/=1.or.neighbour(f)/=-1.or.bid(f)/=-1.or.role(f)/=4) bad=2
   endif
   do slot=1,index-1
    if(interface_faces(slot)==interface_faces(index)) bad=2
   enddo
  enddo
  do f=1,nf
   if(interface_active(f)/=0.and.interface_active(f)/=1) bad=2
   if(interface_active(f)==1.and.count(interface_faces==f-1)/=1) bad=2
  enddo
  if(bad/=0) then;status=bad;return;endif
  nblocks=(nc+1023)/1024
  allocate(corrections(nc,ng),partial_delta(nblocks,ng),partial_depletion(nblocks,ng))
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) collapse(2) schedule(static) &
  !$omp& default(shared) private(k,f,l,r,left,right,raw,corrected,value,donor,rawdonor,cs,fs, &
  !$omp& hidright,index,slot,is_interface,external_availability) reduction(max:bad)
  do k=1,ng
   do f=1,nf
    flux(f,k)=0.d0;l=owner(f)+1;r=neighbour(f)+1
    is_interface=interface_active(f)==1
    if(role(f)==4.and..not.is_interface) cycle
    left=vector(l,k,:);right=left;hidright=hiding(l,k);external_availability=1.d0
    if(is_interface) then
     index=0
     do slot=1,ni
      if(interface_faces(slot)==f-1) index=slot
     enddo
     if(index<1.or..not.all(ieee_is_finite(trace_vector(index,k,:))).or. &
        .not.ieee_is_finite(trace_hiding(index,k)).or.trace_hiding(index,k)<0.d0.or. &
        .not.ieee_is_finite(trace_availability(index,k)).or. &
        trace_availability(index,k)<0.d0.or.trace_availability(index,k)>1.d0) then
      bad=3;cycle
     endif
     right=trace_vector(index,k,:);hidright=trace_hiding(index,k)
     external_availability=trace_availability(index,k)
     call braids_flow_bed_slope_face(mode,0,.true.,left,right,left,zero,normal(f,:),down(f,:), &
      stress(f,k),depth(f,k),d50(k),mean(f,k),hiding(l,k),hidright,1.d0, &
      external_availability,controls(1),controls(2),controls(3),controls(4),controls(5), &
      controls(6),controls(7),controls(8),raw,corrected,value,rawdonor,donor,cs,fs)
    else
     if(r>0) then;right=vector(r,k,:);hidright=hiding(r,k);endif
     call braids_flow_bed_slope_face(mode,role(f),r>0,left,right,left,zero,normal(f,:),down(f,:), &
      stress(f,k),depth(f,k),d50(k),mean(f,k),hiding(l,k),hidright,1.d0,1.d0, &
      controls(1),controls(2),controls(3),controls(4),controls(5),controls(6),controls(7), &
      controls(8),raw,corrected,value,rawdonor,donor,cs,fs)
    endif
    if(cs/=0.or.fs/=0.or..not.ieee_is_finite(value)) then;bad=4;else;flux(f,k)=value;endif
   enddo
  enddo
  !$omp end parallel do
  if(bad/=0) then;status=bad;deallocate(corrections,partial_delta,partial_depletion);return;endif
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) collapse(2) schedule(static) &
  !$omp& default(shared) private(k,c,slot,f,outgoing,value)
  do k=1,ng
   do c=1,nc
    outgoing=0.d0;fill(c,k)=1.d0
    do slot=offset(c)+1,offset(c+1)
     f=faces(slot)+1
     if(role(f)/=0.and.role(f)/=3.and.interface_active(f)/=1) cycle
     value=signs(slot)*flux(f,k)
     if(value>0.d0) outgoing=outgoing+value*length(f)*dt/area(c)
    enddo
    if(bed(c,k)==0.d0) fill(c,k)=0.d0
    if(outgoing>0.d0) fill(c,k)=min(1.d0,bed(c,k)/outgoing)
   enddo
  enddo
  !$omp end parallel do
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) collapse(2) schedule(static) &
  !$omp& default(shared) private(k,f,c)
  do k=1,ng
   do f=1,nf
    c=owner(f)+1
    if(flux(f,k)<0.d0.and.role(f)==0) c=neighbour(f)+1
    if(interface_active(f)==1) then
     if(flux(f,k)>0.d0) flux(f,k)=flux(f,k)*fill(owner(f)+1,k)
    elseif(c>0) then
     flux(f,k)=flux(f,k)*fill(c,k)
    endif
   enddo
  enddo
  !$omp end parallel do
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) collapse(2) schedule(static) &
  !$omp& default(shared) private(k,c,slot,f,value,candidate,tolerance) reduction(max:bad)
  do k=1,ng
   do c=1,nc
    value=0.d0
    do slot=offset(c)+1,offset(c+1)
     f=faces(slot)+1;value=value-signs(slot)*flux(f,k)*length(f)*dt/area(c)
    enddo
    candidate=bed(c,k)+value;tolerance=4096.d0*epsilon(1.d0)*max(1.d0,abs(bed(c,k)))
    if(candidate < -tolerance.or..not.ieee_is_finite(candidate)) bad=5
    corrections(c,k)=0.d0
    if(candidate<0.d0) then;corrections(c,k)=-bed(c,k)-value;value=-bed(c,k);endif
    delta(c,k)=value
   enddo
  enddo
  !$omp end parallel do
  if(bad/=0) then;status=bad;deallocate(corrections,partial_delta,partial_depletion);return;endif
  if(nb>0) then
   !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) collapse(2) schedule(static) &
   !$omp& default(shared) private(k,b,slot,f,value,total)
   do k=1,ng
    do b=1,nb
     total=0.d0;value=0.d0
     do slot=boffset(b)+1,boffset(b+1)
      f=bfaces(slot)+1;total=total+flux(f,k)*length(f)*dt;value=value+abs(flux(f,k)*length(f)*dt)
     enddo
     named(b,k)=total;absolute(b,k)=value
    enddo
   enddo
   !$omp end parallel do
  endif
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) collapse(2) schedule(static) &
  !$omp& default(shared) private(k,index,f)
  do k=1,ng
   do index=1,ni
    f=interface_faces(index)+1;interface_transfer(index,k)=flux(f,k)*length(f)*dt
   enddo
  enddo
  !$omp end parallel do
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) collapse(2) schedule(static) &
  !$omp& default(shared) private(k,block,c,start,last,value,total)
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
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) &
  !$omp& default(shared) private(k,block,b,index,value,total) reduction(max:bad)
  do k=1,ng
   total=0.d0;value=0.d0
   do block=1,nblocks
    total=total+partial_delta(block,k);value=value+partial_depletion(block,k)
   enddo
   do b=1,nb;total=total+named(b,k);enddo
   do index=1,ni;total=total+interface_transfer(index,k);enddo
   depletion(k)=value;residual(k)=total-value
   if(.not.ieee_is_finite(residual(k)).or..not.ieee_is_finite(value)) bad=6
  enddo
  !$omp end parallel do
  status=bad
  deallocate(corrections,partial_delta,partial_depletion)
 end subroutine braids_q3_face_interface_v98
end module braids_q3_face_interface_v98_module
