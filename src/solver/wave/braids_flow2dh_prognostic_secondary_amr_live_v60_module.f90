! Additive native-CPU composition of the accepted Q3-H3/v59 AMR primitives.
!
! This module owns only the first live slice: one synchronous, fully-wet,
! closed-boundary selected mesh.  The cut-cell controller remains the sole
! accepted-step/retry and topology-publication owner.  Physical faces are
! sorted by their immutable AMR lineage before entering the v59 kernels.
module braids_flow2dh_prognostic_secondary_amr_live_v60_module
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 use braids_flow2dh_prognostic_secondary_amr_v59_module, only: &
  braids_amr_restrict_signed_hi_v59,braids_amr_prolong_signed_hi_v59, &
  braids_amr_reconstruct_segment_traces_v59, &
  braids_amr_composite_hi_step_v59,braids_amr_betac_segments_v59, &
  braids_amr_coriolis_active_leaves_v59
 implicit none
 private
 public :: braids_amr_live_synchronous_step_v60
 public :: braids_amr_live_betac_increment_v60
 public :: braids_amr_live_prolong_signed_hi_v60
 public :: braids_amr_live_restrict_signed_hi_v60
 public :: braids_amr_live_fully_wet_v60

contains

 pure logical function braids_amr_live_fully_wet_v60(depth,dry_tolerance) &
  result(valid)
  real(8),intent(in) :: depth(:),dry_tolerance
  valid=size(depth)>0.and.ieee_is_finite(dry_tolerance).and. &
   dry_tolerance>0.d0.and.all(ieee_is_finite(depth)).and. &
   all(depth>dry_tolerance)
 end function braids_amr_live_fully_wet_v60

 subroutine sort_internal_faces_v60(nf,neighbour_all,source_id_all,face, &
  nsegment,valid)
  integer,intent(in) :: nf,neighbour_all(nf),source_id_all(nf)
  integer,intent(out) :: face(nf),nsegment
  logical,intent(out) :: valid
  integer,allocatable :: work(:)
  integer :: candidate,width,left,middle,right,i,j,k
  valid=.false.;nsegment=0;face=0
  do candidate=1,nf
   if(neighbour_all(candidate)<=0) cycle
   nsegment=nsegment+1;face(nsegment)=candidate
  enddo
  if(nsegment<1) return
  allocate(work(nsegment));width=1
  do while(width<nsegment)
   left=1
   do while(left<=nsegment)
    middle=min(left+width,nsegment+1);right=min(left+2*width,nsegment+1)
    i=left;j=middle;k=left
    do while(i<middle.and.j<right)
     if(source_id_all(face(i))<=source_id_all(face(j))) then
      work(k)=face(i);i=i+1
     else
      work(k)=face(j);j=j+1
     endif
     k=k+1
    enddo
    do while(i<middle)
     work(k)=face(i);i=i+1;k=k+1
    enddo
    do while(j<right)
     work(k)=face(j);j=j+1;k=k+1
    enddo
    left=left+2*width
   enddo
   face(:nsegment)=work;width=2*width
  enddo
  if(any(source_id_all(face(:nsegment))<=0)) then
   deallocate(work);return
  endif
  if(nsegment>1) then
   if(any(source_id_all(face(2:nsegment))<= &
          source_id_all(face(1:nsegment-1)))) then
    deallocate(work);return
   endif
  endif
  valid=.true.;deallocate(work)
 end subroutine sort_internal_faces_v60

 subroutine braids_amr_live_betac_increment_v60(nc,nf,owner_all, &
  neighbour_all,source_id_all,cell_level,area,depth_old,qx_old,qy_old, &
  hi_old,chezy_old,normal_all,length_all,betac,gravity,von_karman,chzmin, &
  dry_tolerance,dt,delta_qx,delta_qy,k_betac_all,valid)
  integer,intent(in) :: nc,nf,owner_all(nf),neighbour_all(nf)
  integer,intent(in) :: source_id_all(nf),cell_level(nc)
  real(8),intent(in) :: area(nc),depth_old(nc),qx_old(nc),qy_old(nc)
  real(8),intent(in) :: hi_old(nc),chezy_old(nc),normal_all(2,nf)
  real(8),intent(in) :: length_all(nf),betac,gravity,von_karman,chzmin
  real(8),intent(in) :: dry_tolerance,dt
  real(8),intent(out) :: delta_qx(nc),delta_qy(nc),k_betac_all(2,nf)
  logical,intent(out) :: valid
  integer,allocatable :: face(:),source_id(:),segment_level(:),owner(:),neighbour(:)
  real(8),allocatable :: segment_length(:),normal_x(:),normal_y(:),kx(:),ky(:)
  real(8),allocatable :: tensor_xx(:),tensor_xy(:),tensor_yy(:),beta(:),beta_star(:)
  logical :: catalog_valid,kernel_valid,cancelled
  integer :: nsegment,index,candidate,left,right
  delta_qx=0.d0;delta_qy=0.d0;k_betac_all=0.d0;valid=.false.
  if(nc<1.or.nf<1.or.dt<0.d0.or.dry_tolerance<=0.d0.or.gravity<=0.d0.or. &
     von_karman<=0.d0.or.chzmin<=0.d0) return
  if(.not.all(ieee_is_finite(area)).or.any(area<=0.d0).or. &
     .not.all(ieee_is_finite(depth_old)).or.any(depth_old<=dry_tolerance).or. &
     .not.all(ieee_is_finite(qx_old)).or..not.all(ieee_is_finite(qy_old)).or. &
     .not.all(ieee_is_finite(hi_old)).or..not.all(ieee_is_finite(chezy_old)).or. &
     any(chezy_old<=0.d0).or..not.all(ieee_is_finite(normal_all)).or. &
     .not.all(ieee_is_finite(length_all)).or.any(length_all<=0.d0).or. &
     .not.all(ieee_is_finite([betac,gravity,von_karman,chzmin,dry_tolerance,dt]))) return
  allocate(face(nf));call sort_internal_faces_v60(nf,neighbour_all, &
   source_id_all,face,nsegment,catalog_valid)
  if(.not.catalog_valid) then
   deallocate(face);return
  endif
  allocate(source_id(nsegment),segment_level(nsegment),owner(nsegment), &
   neighbour(nsegment),segment_length(nsegment),normal_x(nsegment), &
   normal_y(nsegment),kx(nsegment),ky(nsegment),tensor_xx(nc),tensor_xy(nc), &
   tensor_yy(nc),beta(nc),beta_star(nc))
  do index=1,nsegment
   candidate=face(index);left=owner_all(candidate);right=neighbour_all(candidate)
   if(left<1.or.left>nc.or.right<1.or.right>nc.or.left==right) then
    deallocate(face,source_id,segment_level,owner,neighbour,segment_length, &
     normal_x,normal_y,kx,ky,tensor_xx,tensor_xy,tensor_yy,beta,beta_star)
    return
   endif
   source_id(index)=source_id_all(candidate);owner(index)=left;neighbour(index)=right
   segment_level(index)=max(cell_level(left),cell_level(right))
   segment_length(index)=length_all(candidate)
   normal_x(index)=normal_all(1,candidate);normal_y(index)=normal_all(2,candidate)
  enddo
  if(maxval(abs(normal_x*normal_x+normal_y*normal_y-1.d0))> &
     128.d0*epsilon(1.d0)) then
   deallocate(face,source_id,segment_level,owner,neighbour,segment_length, &
    normal_x,normal_y,kx,ky,tensor_xx,tensor_xy,tensor_yy,beta,beta_star)
   return
  endif
  call braids_amr_betac_segments_v59(nc,nsegment,area,depth_old,qx_old,qy_old, &
   hi_old,chezy_old,owner,neighbour,source_id,cell_level,segment_level, &
   segment_length,normal_x,normal_y,betac,gravity,von_karman,chzmin, &
   dry_tolerance,dt,.false.,kx,ky,delta_qx,delta_qy,tensor_xx,tensor_xy, &
   tensor_yy,beta,beta_star,kernel_valid,cancelled)
  if(kernel_valid.and..not.cancelled) then
   do index=1,nsegment
    candidate=face(index);k_betac_all(1,candidate)=kx(index)
    k_betac_all(2,candidate)=ky(index)
   enddo
   valid=.true.
  else
   delta_qx=0.d0;delta_qy=0.d0;k_betac_all=0.d0
  endif
  deallocate(face,source_id,segment_level,owner,neighbour,segment_length, &
   normal_x,normal_y,kx,ky,tensor_xx,tensor_xy,tensor_yy,beta,beta_star)
 end subroutine braids_amr_live_betac_increment_v60

 subroutine braids_amr_live_synchronous_step_v60( &
  nc,nf,owner_all,neighbour_all,source_id_all,cell_offset,cell_faces, &
  cell_signs,cell_level,area,centre, &
  normal_all,length_all,midpoint_all,depth_old,hi_old, &
  depth_new,qx_hydraulic,qy_hydraulic,accepted_water_flux_all, &
  dry_tolerance,dt, &
  scalar_diffusivity,coriolis, &
  hi_new,qx_new,qy_new,j_adv_all,j_diff_all,k_betac_all, &
  coriolis_ledger,linear_residual,linear_iterations,valid)
  integer,intent(in) :: nc,nf,owner_all(nf),neighbour_all(nf)
  integer,intent(in) :: source_id_all(nf),cell_offset(nc+1)
  integer,intent(in) :: cell_faces(:),cell_signs(:),cell_level(nc)
  real(8),intent(in) :: area(nc),centre(2,nc),normal_all(2,nf)
  real(8),intent(in) :: length_all(nf),midpoint_all(2,nf)
  real(8),intent(in) :: depth_old(nc),hi_old(nc),depth_new(nc)
  real(8),intent(in) :: qx_hydraulic(nc),qy_hydraulic(nc)
  real(8),intent(in) :: accepted_water_flux_all(nf)
  real(8),intent(in) :: dry_tolerance,dt,scalar_diffusivity,coriolis
  real(8),intent(out) :: hi_new(nc),qx_new(nc),qy_new(nc)
  real(8),intent(out) :: j_adv_all(nf),j_diff_all(nf)
  real(8),intent(in) :: k_betac_all(2,nf)
  real(8),intent(out) :: coriolis_ledger(2),linear_residual
  integer,intent(out) :: linear_iterations
  logical,intent(out) :: valid
  integer,allocatable :: face(:),source_id(:),segment_level(:),owner(:),neighbour(:)
  real(8),allocatable :: segment_length(:),normal_x(:),normal_y(:)
  real(8),allocatable :: midpoint_x(:),midpoint_y(:),normal_distance(:)
  real(8),allocatable :: diffusivity(:),normal_discharge(:)
  real(8),allocatable :: primitive(:),gradient_x(:),gradient_y(:)
  real(8),allocatable :: primitive_minimum(:),primitive_maximum(:)
  real(8),allocatable :: owner_trace(:),neighbour_trace(:)
  real(8),allocatable :: delta_adv(:),delta_diff(:),j_adv(:),j_diff(:),conductance(:)
  logical,allocatable :: active(:)
  logical :: kernel_valid,cancelled
  integer :: nsegment,index,candidate,cell,left,right,other,incident
  real(8) :: dx,dy,dvalue,weight,a11,a12,a22,b1,b2,det,scale

  hi_new=hi_old;qx_new=qx_hydraulic;qy_new=qy_hydraulic
  j_adv_all=0.d0;j_diff_all=0.d0
  coriolis_ledger=0.d0;linear_residual=huge(1.d0)
  linear_iterations=0;valid=.false.
  if(nc<1.or.nf<1.or.dt<0.d0.or.dry_tolerance<=0.d0.or. &
     scalar_diffusivity<0.d0) return
  if(.not.all(ieee_is_finite(area)).or.any(area<=0.d0).or. &
     .not.all(ieee_is_finite(centre)).or. &
     .not.all(ieee_is_finite(normal_all)).or. &
     .not.all(ieee_is_finite(length_all)).or.any(length_all<=0.d0).or. &
     .not.all(ieee_is_finite(midpoint_all)).or. &
     .not.all(ieee_is_finite(depth_old)).or.any(depth_old<=dry_tolerance).or. &
     .not.all(ieee_is_finite(depth_new)).or.any(depth_new<=dry_tolerance).or. &
     .not.all(ieee_is_finite(hi_old)).or. &
     .not.all(ieee_is_finite(qx_hydraulic)).or. &
     .not.all(ieee_is_finite(qy_hydraulic)).or. &
     .not.all(ieee_is_finite(accepted_water_flux_all)).or. &
     .not.all(ieee_is_finite(k_betac_all)).or. &
     .not.all(ieee_is_finite([dry_tolerance,dt,scalar_diffusivity,coriolis]))) return
  if(size(cell_faces)/=size(cell_signs).or.cell_offset(1)/=1.or. &
     cell_offset(nc+1)-1/=size(cell_faces).or. &
     any(cell_offset(2:nc+1)<cell_offset(1:nc))) return
  do cell=1,nc
   do incident=cell_offset(cell),cell_offset(cell+1)-1
    candidate=cell_faces(incident)
    if(candidate<1.or.candidate>nf) return
    if(cell_signs(incident)==1) then
     if(owner_all(candidate)/=cell) return
    else if(cell_signs(incident)==-1) then
     if(neighbour_all(candidate)/=cell) return
    else
     return
    endif
   enddo
  enddo

  allocate(face(nf))
  call sort_internal_faces_v60(nf,neighbour_all,source_id_all,face,nsegment,kernel_valid)
  if(.not.kernel_valid) then
   deallocate(face);return
  endif
  allocate(source_id(nsegment),segment_level(nsegment), &
   owner(nsegment),neighbour(nsegment),segment_length(nsegment), &
   normal_x(nsegment),normal_y(nsegment),midpoint_x(nsegment), &
   midpoint_y(nsegment),normal_distance(nsegment),diffusivity(nsegment), &
   normal_discharge(nsegment),primitive(nc),gradient_x(nc),gradient_y(nc), &
   primitive_minimum(nc),primitive_maximum(nc),owner_trace(nsegment), &
   neighbour_trace(nsegment),delta_adv(nc),delta_diff(nc),j_adv(nsegment), &
   j_diff(nsegment),conductance(nsegment),active(nc))

  do index=1,nsegment
   candidate=face(index);left=owner_all(candidate);right=neighbour_all(candidate)
   if(left<1.or.left>nc.or.right<1.or.right>nc.or.left==right.or. &
      source_id_all(candidate)<=0) then
    deallocate(face,source_id,segment_level,owner,neighbour,segment_length, &
     normal_x,normal_y,midpoint_x,midpoint_y,normal_distance,diffusivity, &
     normal_discharge,primitive,gradient_x,gradient_y,primitive_minimum, &
     primitive_maximum,owner_trace,neighbour_trace,delta_adv,delta_diff, &
     j_adv,j_diff,conductance,active)
    return
   endif
   source_id(index)=source_id_all(candidate);owner(index)=left;neighbour(index)=right
   segment_level(index)=max(cell_level(left),cell_level(right))
   segment_length(index)=length_all(candidate)
   normal_x(index)=normal_all(1,candidate);normal_y(index)=normal_all(2,candidate)
   midpoint_x(index)=midpoint_all(1,candidate);midpoint_y(index)=midpoint_all(2,candidate)
   normal_distance(index)=dot_product(centre(:,right)-centre(:,left),normal_all(:,candidate))
   diffusivity(index)=scalar_diffusivity
  enddo
  if(nsegment>1) then
   if(any(source_id(2:nsegment)<=source_id(1:nsegment-1))) then
    deallocate(face,source_id,segment_level,owner,neighbour,segment_length, &
     normal_x,normal_y,midpoint_x,midpoint_y,normal_distance,diffusivity, &
     normal_discharge,primitive,gradient_x,gradient_y,primitive_minimum, &
     primitive_maximum,owner_trace,neighbour_trace,delta_adv,delta_diff, &
     j_adv,j_diff,conductance,active)
    return
   endif
  endif
  if(any(normal_distance<=0.d0).or. &
     maxval(abs(normal_x*normal_x+normal_y*normal_y-1.d0))> &
      128.d0*epsilon(1.d0)) then
   deallocate(face,source_id,segment_level,owner,neighbour,segment_length, &
    normal_x,normal_y,midpoint_x,midpoint_y,normal_distance,diffusivity, &
    normal_discharge,primitive,gradient_x,gradient_y,primitive_minimum, &
    primitive_maximum,owner_trace,neighbour_trace,delta_adv,delta_diff, &
    j_adv,j_diff,conductance,active)
   return
  endif

  primitive=hi_old/depth_old
  primitive_minimum=primitive;primitive_maximum=primitive
  gradient_x=0.d0;gradient_y=0.d0
  do index=1,nsegment
   left=owner(index);right=neighbour(index)
   primitive_minimum(left)=min(primitive_minimum(left),primitive(right))
   primitive_maximum(left)=max(primitive_maximum(left),primitive(right))
   primitive_minimum(right)=min(primitive_minimum(right),primitive(left))
   primitive_maximum(right)=max(primitive_maximum(right),primitive(left))
  enddo
  ! Cell-local weighted least squares over the native incidence map.  This is
  ! O(ninc), not O(nc*nf). Rank-deficient boundary rows retain the constant
  ! reconstruction; the v59 limiter still enforces stencil bounds.
  !$omp parallel do schedule(static,256) default(shared) &
  !$omp& private(cell,incident,candidate,other,dx,dy,dvalue,weight,a11,a12,a22,b1,b2,det,scale)
  do cell=1,nc
   a11=0.d0;a12=0.d0;a22=0.d0;b1=0.d0;b2=0.d0
   do incident=cell_offset(cell),cell_offset(cell+1)-1
    candidate=cell_faces(incident)
    if(neighbour_all(candidate)<=0) cycle
    if(cell_signs(incident)>0) then
     other=neighbour_all(candidate)
    else
     other=owner_all(candidate)
    endif
    dx=centre(1,other)-centre(1,cell);dy=centre(2,other)-centre(2,cell)
    dvalue=primitive(other)-primitive(cell)
    weight=1.d0/max(dx*dx+dy*dy,tiny(1.d0))
    a11=a11+weight*dx*dx;a12=a12+weight*dx*dy
    a22=a22+weight*dy*dy;b1=b1+weight*dx*dvalue;b2=b2+weight*dy*dvalue
   enddo
   det=a11*a22-a12*a12;scale=max(1.d0,a11*a22)
   if(det>128.d0*epsilon(1.d0)*scale) then
    gradient_x(cell)=(a22*b1-a12*b2)/det
    gradient_y(cell)=(a11*b2-a12*b1)/det
   endif
  enddo
  !$omp end parallel do
  call braids_amr_reconstruct_segment_traces_v59(nc,nsegment,centre(1,:), &
   centre(2,:),primitive,gradient_x,gradient_y,primitive_minimum, &
   primitive_maximum,source_id,cell_level,segment_level,owner,neighbour, &
   midpoint_x,midpoint_y,.false.,owner_trace,neighbour_trace,kernel_valid,cancelled)
  if(.not.kernel_valid.or.cancelled) then
   deallocate(face,source_id,segment_level,owner,neighbour,segment_length, &
    normal_x,normal_y,midpoint_x,midpoint_y,normal_distance,diffusivity, &
    normal_discharge,primitive,gradient_x,gradient_y,primitive_minimum, &
    primitive_maximum,owner_trace,neighbour_trace,delta_adv,delta_diff, &
    j_adv,j_diff,conductance,active)
   return
  endif

  active=.true.
  call braids_amr_coriolis_active_leaves_v59(nc,active,area,qx_hydraulic,qy_hydraulic, &
   coriolis,dt,.false.,qx_new,qy_new,coriolis_ledger(1),coriolis_ledger(2), &
   kernel_valid,cancelled)
  if(.not.kernel_valid.or.cancelled) then
   hi_new=hi_old;qx_new=qx_hydraulic;qy_new=qy_hydraulic
   deallocate(face,source_id,segment_level,owner,neighbour,segment_length, &
    normal_x,normal_y,midpoint_x,midpoint_y,normal_distance,diffusivity, &
    normal_discharge,primitive,gradient_x,gradient_y,primitive_minimum, &
    primitive_maximum,owner_trace,neighbour_trace,delta_adv,delta_diff, &
    j_adv,j_diff,conductance,active)
   return
  endif
  ! The transported hI state consumes the accepted hydraulic RK face flux.
  ! Reconstructing it from endpoint cell momenta would not be the same finite-
  ! volume transaction on a nonuniform or coarse/fine face.
  do index=1,nsegment
   normal_discharge(index)=accepted_water_flux_all(face(index))
  enddo
  call braids_amr_composite_hi_step_v59(nc,nsegment,area,depth_new,dry_tolerance, &
   source_id,cell_level,segment_level,owner,neighbour,segment_length,normal_distance, &
   diffusivity,normal_discharge,owner_trace,neighbour_trace,hi_old,dt,1.d-12, &
   1.d-14,1.d-12,512,.false.,.false.,hi_new,delta_adv,delta_diff,j_adv,j_diff, &
   conductance,linear_residual,linear_iterations,kernel_valid,cancelled)
  if(.not.kernel_valid.or.cancelled) then
   hi_new=hi_old;qx_new=qx_hydraulic;qy_new=qy_hydraulic
   coriolis_ledger=0.d0
   deallocate(face,source_id,segment_level,owner,neighbour,segment_length, &
    normal_x,normal_y,midpoint_x,midpoint_y,normal_distance,diffusivity, &
    normal_discharge,primitive,gradient_x,gradient_y,primitive_minimum, &
    primitive_maximum,owner_trace,neighbour_trace,delta_adv,delta_diff, &
    j_adv,j_diff,conductance,active)
   return
  endif
  do index=1,nsegment
   candidate=face(index);j_adv_all(candidate)=j_adv(index)
   j_diff_all(candidate)=j_diff(index)
  enddo
  valid=.true.
  deallocate(face,source_id,segment_level,owner,neighbour,segment_length, &
   normal_x,normal_y,midpoint_x,midpoint_y,normal_distance,diffusivity, &
   normal_discharge,primitive,gradient_x,gradient_y,primitive_minimum, &
   primitive_maximum,owner_trace,neighbour_trace,delta_adv,delta_diff, &
   j_adv,j_diff,conductance,active)
 end subroutine braids_amr_live_synchronous_step_v60

 subroutine braids_amr_live_prolong_signed_hi_v60(parent_area,parent_depth, &
  parent_hi,child_area,child_depth,primitive_seed,dry_tolerance, &
  accepted_child_hi,child_hi_candidate,valid)
  real(8),intent(in) :: parent_area,parent_depth,parent_hi,child_area(4)
  real(8),intent(in) :: child_depth(4),primitive_seed(4),dry_tolerance
  real(8),intent(in) :: accepted_child_hi(4)
  real(8),intent(out) :: child_hi_candidate(4)
  logical,intent(out) :: valid
  logical :: invalidated,cancelled
  real(8) :: parent_intensity,lower,upper
  child_hi_candidate=accepted_child_hi;valid=.false.
  if(parent_depth<=dry_tolerance) return
  parent_intensity=parent_hi/parent_depth
  lower=min(parent_intensity,minval(primitive_seed))
  upper=max(parent_intensity,maxval(primitive_seed))
  call braids_amr_prolong_signed_hi_v59(4,parent_area,parent_depth,parent_hi, &
   child_area,child_depth,primitive_seed,lower,upper,dry_tolerance, &
   accepted_child_hi,.false.,child_hi_candidate,invalidated,valid,cancelled)
  valid=valid.and.invalidated.and..not.cancelled
 end subroutine braids_amr_live_prolong_signed_hi_v60

 subroutine braids_amr_live_restrict_signed_hi_v60(parent_area,parent_hi_cached, &
  child_area,child_hi,parent_hi_candidate,valid)
  real(8),intent(in) :: parent_area,parent_hi_cached,child_area(4),child_hi(4)
  real(8),intent(out) :: parent_hi_candidate
  logical,intent(out) :: valid
  logical :: child_selected(4),invalidated
  real(8) :: active_inventory
  child_selected=.true.
  call braids_amr_restrict_signed_hi_v59(4,parent_area,parent_hi_cached, &
   child_area,child_hi,.false.,child_selected,.true.,.false., &
   parent_hi_candidate,active_inventory,invalidated,valid)
  valid=valid.and.invalidated
 end subroutine braids_amr_live_restrict_signed_hi_v60

end module braids_flow2dh_prognostic_secondary_amr_live_v60_module
