! Q4/v97 additive moving-wet/dry augmented-f-wave residual primitive.
!
! This module deliberately reuses the frozen GeoClaw augmented face routine
! from v95.  It adds only the method-identified active-set and donor-draining
! work needed by the implicit v97 route.  The immutable domain mask and the
! frozen wet mask are distinct inputs.  A frozen-dry cell is presented to the
! Riemann solver as the exact dry state during one inner solve; its residual
! row can nevertheless receive conservative inflow and become wet at the next
! outer active-set pass.
module braids_wetdry_fwave_v97_module
 use iso_c_binding
 use, intrinsic :: ieee_arithmetic
 use omp_lib
 use geoclaw_module, only: grav, dry_tolerance
 use braids_densin_residual_v95_module, only: augmented_face_v95
 implicit none
contains

 subroutine braids_wetdry_fwave_rate_v97( &
  nc,nf,nadj,q,bed,owner_zero,neighbour_zero,normal,face_length,face_distance, &
  cell_area,cell_face_offset_zero,cell_faces_zero,cell_face_sign,domain_active, &
  frozen_wet,accepted_density,reference_density,gravity,dry,dt,threads, &
  raw_left,raw_right,limited_left,limited_right,theta,base_rate, &
  face_water_flux,face_acceleration,wet_wet_face,riemann_calls, &
  limiter_application_count,status) bind(C,name='braids_wetdry_fwave_rate_v97')
  integer(c_int),value :: nc,nf,nadj,threads
  real(c_double),value :: reference_density,gravity,dry,dt
  real(c_double),intent(in) :: q(nc,3),bed(nc),normal(nf,2),face_length(nf)
  real(c_double),intent(in) :: face_distance(nf),cell_area(nc),accepted_density(nc)
  integer(c_int),intent(in) :: owner_zero(nf),neighbour_zero(nf)
  integer(c_int),intent(in) :: cell_face_offset_zero(nc+1),cell_faces_zero(nadj)
  integer(c_int),intent(in) :: cell_face_sign(nadj),domain_active(nc),frozen_wet(nc)
  real(c_double),intent(out) :: raw_left(nf,3),raw_right(nf,3)
  real(c_double),intent(out) :: limited_left(nf,3),limited_right(nf,3)
  real(c_double),intent(out) :: theta(nc),base_rate(nc,3),face_water_flux(nf)
  real(c_double),intent(out) :: face_acceleration(nf)
  integer(c_int),intent(out) :: wet_wet_face(nf)
  integer(c_int),intent(out) :: riemann_calls,limiter_application_count,status

  real(8) :: left(3),right(3),unit_normal(2),tangent(2),bed_right,speed
  real(8) :: normal_momentum,tangent_momentum,outgoing,factor,correction,velocity
  real(8) :: sum_flux(3),hydrostatic(2),hcell
  integer :: face,cell,slot,owner,neighbour,sign,donor,component,invalid
  integer :: active_riemann,limited_faces

  raw_left=0.d0; raw_right=0.d0; limited_left=0.d0; limited_right=0.d0
  theta=1.d0; base_rate=0.d0; face_water_flux=0.d0
  face_acceleration=0.d0; wet_wet_face=0
  riemann_calls=0; limiter_application_count=0; status=0
  if(nc<1.or.nf<1.or.nadj<1.or.threads<1.or.dt<=0.d0.or. &
     reference_density<=0.d0.or.gravity<=0.d0.or.dry<0.d0.or. &
     .not.ieee_is_finite(dt).or..not.ieee_is_finite(reference_density).or. &
     .not.ieee_is_finite(gravity).or..not.ieee_is_finite(dry)) then
   status=1; return
  endif
  if(.not.all(ieee_is_finite(q)).or..not.all(ieee_is_finite(bed)).or. &
     .not.all(ieee_is_finite(normal)).or..not.all(ieee_is_finite(face_length)).or. &
     .not.all(ieee_is_finite(face_distance)).or..not.all(ieee_is_finite(cell_area)).or. &
     .not.all(ieee_is_finite(accepted_density)).or.any(q(:,1)<0.d0).or. &
     any(face_length<=0.d0).or.any(face_distance<=0.d0).or.any(cell_area<=0.d0).or. &
     any(accepted_density<=0.d0)) then
   status=2; return
  endif
  if(any((domain_active/=0).and.(domain_active/=1)).or. &
     any((frozen_wet/=0).and.(frozen_wet/=1)).or. &
     any((frozen_wet==1).and.(domain_active==0)).or. &
     any((frozen_wet==1).and.(q(:,1)<=0.d0))) then
   status=3; return
  endif
  if(cell_face_offset_zero(1)/=0.or.cell_face_offset_zero(nc+1)/=nadj) then
   status=4; return
  endif
  do cell=1,nc
   if(cell_face_offset_zero(cell)>cell_face_offset_zero(cell+1)) then
    status=4; return
   endif
  enddo
  do face=1,nf
   if(owner_zero(face)<0.or.owner_zero(face)>=nc.or.neighbour_zero(face)<-1.or. &
      neighbour_zero(face)>=nc.or.neighbour_zero(face)==owner_zero(face).or. &
      abs(sqrt(sum(normal(face,:)**2))-1.d0)>2.d-13.or. &
      min(abs(normal(face,1)),abs(normal(face,2)))>2.d-13) then
    status=5; return
   endif
   owner=owner_zero(face)+1; neighbour=neighbour_zero(face)+1
   if(domain_active(owner)==0.or.(neighbour>0.and.domain_active(neighbour)==0)) then
    ! The bounded fixed-grid ABI admits computational faces only between
    ! immutable-domain cells.  Holes need the later cut-cell/AMR topology ABI.
    status=6; return
   endif
  enddo
  do slot=1,nadj
   if(cell_faces_zero(slot)<0.or.cell_faces_zero(slot)>=nf.or. &
      abs(cell_face_sign(slot))/=1) then
    status=7; return
   endif
  enddo

  grav=gravity; dry_tolerance=dry
  invalid=0; active_riemann=0
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) &
  !$omp& private(face,owner,neighbour,left,right,unit_normal,tangent,normal_momentum, &
  !$omp& tangent_momentum,bed_right,speed) reduction(max:invalid) reduction(+:active_riemann)
  do face=1,nf
   owner=owner_zero(face)+1; neighbour=neighbour_zero(face)+1
   unit_normal=normal(face,:)
   if(frozen_wet(owner)==1) then
    left=q(owner,:)
   else
    left=0.d0
   endif
   if(neighbour>0) then
    if(frozen_wet(neighbour)==1) then
     right=q(neighbour,:)
    else
     right=0.d0
    endif
    bed_right=bed(neighbour)
    if(frozen_wet(owner)==1.and.frozen_wet(neighbour)==1) wet_wet_face(face)=1
   else
    tangent=[-unit_normal(2),unit_normal(1)]
    normal_momentum=dot_product(left(2:3),unit_normal)
    tangent_momentum=dot_product(left(2:3),tangent)
    right(1)=left(1)
    right(2:3)=(-normal_momentum)*unit_normal+tangent_momentum*tangent
    bed_right=bed(owner)
   endif
   call augmented_face_v95(left,right,bed(owner),bed_right,unit_normal, &
    raw_left(face,:),raw_right(face,:),speed)
   raw_left(face,1)=.5d0*(raw_left(face,1)+raw_right(face,1))
   raw_right(face,1)=raw_left(face,1)
   limited_left(face,:)=raw_left(face,:)
   limited_right(face,:)=raw_right(face,:)
   if(wet_wet_face(face)==1) then
    face_acceleration(face)= &
     -gravity*(accepted_density(neighbour)-accepted_density(owner))* &
     (q(neighbour,1)+q(owner,1))/(4.d0*reference_density*face_distance(face))
   endif
   if(.not.all(ieee_is_finite(raw_left(face,:))).or. &
      .not.all(ieee_is_finite(raw_right(face,:))).or. &
      .not.ieee_is_finite(speed).or..not.ieee_is_finite(face_acceleration(face))) invalid=1
   active_riemann=active_riemann+1
  enddo
  !$omp end parallel do
  if(invalid/=0) then
   status=8; return
  endif

  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) &
  !$omp& private(cell,slot,face,sign,outgoing)
  do cell=1,nc
   outgoing=0.d0
   do slot=cell_face_offset_zero(cell)+1,cell_face_offset_zero(cell+1)
    face=cell_faces_zero(slot)+1; sign=cell_face_sign(slot)
    outgoing=outgoing+max(sign*raw_left(face,1),0.d0)*face_length(face)
   enddo
   if(outgoing>0.d0.and.dt*outgoing>=merge(q(cell,1),0.d0,frozen_wet(cell)==1)*cell_area(cell)) then
    theta(cell)=min(1.d0,merge(q(cell,1),0.d0,frozen_wet(cell)==1)*cell_area(cell)/(dt*outgoing))* &
     (1.d0-32.d0*epsilon(1.d0))
   endif
  enddo
  !$omp end parallel do

  limited_faces=0; invalid=0
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) &
  !$omp& private(face,owner,neighbour,donor,factor,correction,component,velocity) &
  !$omp& reduction(+:limited_faces) reduction(max:invalid)
  do face=1,nf
   owner=owner_zero(face)+1; neighbour=neighbour_zero(face)+1
   if(raw_left(face,1)>0.d0) then
    donor=owner
   elseif(raw_left(face,1)<0.d0.and.neighbour>0) then
    donor=neighbour
   else
    donor=0
   endif
   if(donor>0.and.theta(donor)<1.d0) then
    factor=theta(donor); correction=(factor-1.d0)*raw_left(face,1)
    limited_left(face,1)=factor*raw_left(face,1)
    limited_right(face,1)=limited_left(face,1)
    do component=2,3
     if(frozen_wet(donor)==1.and.q(donor,1)>dry) then
      velocity=q(donor,component)/q(donor,1)
     else
      velocity=0.d0
     endif
     limited_left(face,component)=raw_left(face,component)+correction*velocity
     limited_right(face,component)=raw_right(face,component)+correction*velocity
    enddo
    limited_faces=limited_faces+1
   endif
   face_water_flux(face)=limited_left(face,1)
   if(.not.all(ieee_is_finite(limited_left(face,:))).or. &
      .not.all(ieee_is_finite(limited_right(face,:))) ) invalid=1
  enddo
  !$omp end parallel do
  if(invalid/=0.or.any(theta<0.d0).or.any(theta>1.d0)) then
   status=9; return
  endif

  invalid=0
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) &
  !$omp& private(cell,slot,face,sign,sum_flux,hydrostatic,hcell) reduction(max:invalid)
  do cell=1,nc
   sum_flux=0.d0
   hcell=merge(q(cell,1),0.d0,frozen_wet(cell)==1)
   do slot=cell_face_offset_zero(cell)+1,cell_face_offset_zero(cell+1)
    face=cell_faces_zero(slot)+1; sign=cell_face_sign(slot)
    hydrostatic=.5d0*gravity*hcell*hcell*normal(face,:)
    if(sign>0) then
     sum_flux(1)=sum_flux(1)+(face_length(face)/cell_area(cell))*limited_left(face,1)
     sum_flux(2:3)=sum_flux(2:3)+(face_length(face)/cell_area(cell))* &
      (limited_left(face,2:3)-hydrostatic)
    else
     sum_flux(1)=sum_flux(1)-(face_length(face)/cell_area(cell))*limited_right(face,1)
     sum_flux(2:3)=sum_flux(2:3)-(face_length(face)/cell_area(cell))* &
      (limited_right(face,2:3)-hydrostatic)
    endif
   enddo
   base_rate(cell,:)=-sum_flux
   if(.not.all(ieee_is_finite(base_rate(cell,:)))) invalid=1
  enddo
  !$omp end parallel do
  if(invalid/=0) then
   status=10; return
  endif
  riemann_calls=active_riemann
  limiter_application_count=limited_faces
 end subroutine braids_wetdry_fwave_rate_v97
end module braids_wetdry_fwave_v97_module
