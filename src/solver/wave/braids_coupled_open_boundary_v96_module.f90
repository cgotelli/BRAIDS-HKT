! Additive v96 native open-boundary residual consumer.
!
! Open faces consume the physical trace and already-corrected physical flux
! emitted by the sole hydraulic boundary owner.  They are never passed to the
! augmented f-wave solver as a fabricated right state.  Interior and closed
! faces retain the frozen v95 augmented-face algebra.  The FLOW KMAX=1,
! ICREEP=0 exterior density term uses a parent-frozen boundary density and the
! current trace depth.  IBaroc gates only that exterior term.
module braids_coupled_open_boundary_v96_module
 use iso_c_binding
 use, intrinsic :: ieee_arithmetic
 use omp_lib, only: omp_in_parallel
 use geoclaw_module, only: grav, dry_tolerance
 use braids_densin_residual_v95_module, only: augmented_face_v95
 implicit none
contains

 subroutine braids_coupled_open_full_rate_v96( &
  nc,nf,nadj,q,bed,owner_zero,neighbour_zero,boundary_kind,exterior_density_active,normal,face_length,face_distance, &
  cell_area,cell_face_offset_zero,cell_faces_zero,cell_face_sign,accepted_density, &
  boundary_density,boundary_trace_q,boundary_physical_flux,reference_density,gravity,dry, &
  density_gradient_enabled,ibaroc,threads,base_rate,density_rate,full_rate, &
  face_acceleration,cell_acceleration,face_water_flux,riemann_calls,density_source_calls, &
  open_flux_calls,open_density_face_calls,status) &
  bind(C,name='braids_coupled_open_full_rate_v96')
  integer(c_int),value :: nc,nf,nadj,density_gradient_enabled,ibaroc,threads
  real(c_double),intent(in) :: q(nc,3),bed(nc),normal(nf,2),face_length(nf)
  real(c_double),intent(in) :: face_distance(nf),cell_area(nc),accepted_density(nc)
  real(c_double),intent(in) :: boundary_density(nf),boundary_trace_q(nf,3)
  real(c_double),intent(in) :: boundary_physical_flux(nf,3)
  real(c_double),value :: reference_density,gravity,dry
  integer(c_int),intent(in) :: owner_zero(nf),neighbour_zero(nf),boundary_kind(nf)
  integer(c_int),intent(in) :: exterior_density_active(nf)
  integer(c_int),intent(in) :: cell_face_offset_zero(nc+1),cell_faces_zero(nadj)
  integer(c_int),intent(in) :: cell_face_sign(nadj)
  real(c_double),intent(out) :: base_rate(nc,3),density_rate(nc,2),full_rate(nc,3)
  real(c_double),intent(out) :: face_acceleration(nf),cell_acceleration(nc,2)
  real(c_double),intent(out) :: face_water_flux(nf)
  integer(c_int),intent(out) :: riemann_calls,density_source_calls,open_flux_calls
  integer(c_int),intent(out) :: open_density_face_calls,status

  real(8) :: flux_left(nf,3),flux_right(nf,3),signal_speed(nf)
  real(8) :: left(3),right(3),unit_normal(2),tangent(2),normal_momentum,tangent_momentum,bed_right
  real(8) :: sum_flux(3),matrix11,matrix12,matrix22,rhs1,rhs2,determinant,scale
  real(8) :: outward1,outward2,directional,weight,hydrostatic(2)
  integer :: face,cell,slot,owner,neighbour,sign,invalid

  base_rate=0.d0;density_rate=0.d0;full_rate=0.d0
  face_acceleration=0.d0;cell_acceleration=0.d0;face_water_flux=0.d0
  flux_left=0.d0;flux_right=0.d0;signal_speed=0.d0
  riemann_calls=0;density_source_calls=0;open_flux_calls=0
  open_density_face_calls=0;status=0
  if(nc<1.or.nf<1.or.nadj<1.or.threads<1.or. &
     (density_gradient_enabled/=0.and.density_gradient_enabled/=1).or. &
     (ibaroc/=0.and.ibaroc/=1)) then
   status=1;return
  endif
  if(reference_density<=0.d0.or.gravity<=0.d0.or.dry<0.d0.or. &
     .not.ieee_is_finite(reference_density).or..not.ieee_is_finite(gravity).or. &
     .not.ieee_is_finite(dry)) then
   status=2;return
  endif
  if(.not.all(ieee_is_finite(q)).or..not.all(ieee_is_finite(bed)).or. &
     .not.all(ieee_is_finite(normal)).or..not.all(ieee_is_finite(face_length)).or. &
     .not.all(ieee_is_finite(face_distance)).or..not.all(ieee_is_finite(cell_area)).or. &
     .not.all(ieee_is_finite(accepted_density)).or. &
     .not.all(ieee_is_finite(boundary_density)).or. &
     .not.all(ieee_is_finite(boundary_trace_q)).or. &
     .not.all(ieee_is_finite(boundary_physical_flux)).or.any(q(:,1)<=dry).or. &
     any(face_length<=0.d0).or.any(face_distance<=0.d0).or.any(cell_area<=0.d0).or. &
     any(accepted_density<=0.d0)) then
   status=3;return
  endif
  if(cell_face_offset_zero(1)/=0.or.cell_face_offset_zero(nc+1)/=nadj) then
   status=4;return
  endif
  do cell=1,nc
   if(cell_face_offset_zero(cell)>cell_face_offset_zero(cell+1)) then
    status=4;return
   endif
  enddo
  do face=1,nf
   if(owner_zero(face)<0.or.owner_zero(face)>=nc.or.neighbour_zero(face)<-1.or. &
      neighbour_zero(face)>=nc.or.neighbour_zero(face)==owner_zero(face).or. &
      abs(sqrt(sum(normal(face,:)**2))-1.d0)>2.d-13.or. &
      min(abs(normal(face,1)),abs(normal(face,2)))>2.d-13.or. &
      (neighbour_zero(face)>=0.and.boundary_kind(face)/=1).or. &
      (neighbour_zero(face)<0.and.boundary_kind(face)/=2.and.boundary_kind(face)/=3).or. &
      (exterior_density_active(face)/=0.and.exterior_density_active(face)/=1).or. &
      (exterior_density_active(face)==1.and.boundary_kind(face)/=3)) then
    status=5;return
   endif
   if(boundary_kind(face)==3) then
    if(boundary_trace_q(face,1)<0.d0.or. &
       (exterior_density_active(face)==1.and. &
        (boundary_trace_q(face,1)<=dry.or.boundary_density(face)<=0.d0))) then
     status=5;return
    endif
   endif
  enddo
  do slot=1,nadj
   if(cell_faces_zero(slot)<0.or.cell_faces_zero(slot)>=nf.or. &
      abs(cell_face_sign(slot))/=1) then
    status=6;return
   endif
  enddo

  grav=gravity;dry_tolerance=dry
  invalid=0
  !$omp parallel do schedule(static) default(shared) if(.not.omp_in_parallel()) num_threads(threads) &
  !$omp& private(face,owner,neighbour,left,right,unit_normal,tangent,normal_momentum, &
  !$omp& tangent_momentum,bed_right) reduction(max:invalid) &
  !$omp& reduction(+:riemann_calls,open_flux_calls,open_density_face_calls)
  do face=1,nf
   owner=owner_zero(face)+1;neighbour=neighbour_zero(face)+1
   unit_normal=normal(face,:);left=q(owner,:)
   if(boundary_kind(face)==3) then
    flux_left(face,:)=boundary_physical_flux(face,:)
    flux_right(face,:)=boundary_physical_flux(face,:)
    face_water_flux(face)=boundary_physical_flux(face,1)
    if(boundary_trace_q(face,1)>dry) then
     signal_speed(face)=sqrt(gravity*boundary_trace_q(face,1))+ &
      abs(dot_product(boundary_trace_q(face,2:3),unit_normal))/boundary_trace_q(face,1)
    else
     signal_speed(face)=0.d0
    endif
    open_flux_calls=open_flux_calls+1
   else
    if(neighbour>0) then
     right=q(neighbour,:);bed_right=bed(neighbour)
    else
     tangent=[-unit_normal(2),unit_normal(1)]
     normal_momentum=dot_product(left(2:3),unit_normal)
     tangent_momentum=dot_product(left(2:3),tangent)
     right(1)=left(1)
     right(2:3)=(-normal_momentum)*unit_normal+tangent_momentum*tangent
     bed_right=bed(owner)
    endif
    call augmented_face_v95(left,right,bed(owner),bed_right,unit_normal, &
     flux_left(face,:),flux_right(face,:),signal_speed(face))
    face_water_flux(face)=flux_left(face,1)
    riemann_calls=riemann_calls+1
   endif
   if(density_gradient_enabled==1) then
    if(neighbour>0) then
     face_acceleration(face)= &
      -gravity*(accepted_density(neighbour)-accepted_density(owner))* &
      (q(neighbour,1)+q(owner,1))/(4.d0*reference_density*face_distance(face))
    else if(boundary_kind(face)==3.and.ibaroc==1.and. &
            exterior_density_active(face)==1) then
     face_acceleration(face)= &
      -gravity*(boundary_density(face)-accepted_density(owner))* &
      (boundary_trace_q(face,1)+q(owner,1))/ &
      (4.d0*reference_density*face_distance(face))
     open_density_face_calls=open_density_face_calls+1
    endif
   endif
   if(.not.all(ieee_is_finite(flux_left(face,:))).or. &
      .not.all(ieee_is_finite(flux_right(face,:))).or. &
      .not.ieee_is_finite(signal_speed(face)).or. &
      .not.ieee_is_finite(face_acceleration(face))) invalid=1
  enddo
  !$omp end parallel do
  if(invalid/=0) then
   status=7;return
  endif

  invalid=0
  !$omp parallel do schedule(static) default(shared) if(.not.omp_in_parallel()) num_threads(threads) &
  !$omp& private(cell,slot,face,sign,sum_flux,hydrostatic,matrix11,matrix12,matrix22, &
  !$omp& rhs1,rhs2,outward1,outward2,directional,weight,determinant,scale) reduction(max:invalid)
  do cell=1,nc
   sum_flux=0.d0;matrix11=0.d0;matrix12=0.d0;matrix22=0.d0
   rhs1=0.d0;rhs2=0.d0
   do slot=cell_face_offset_zero(cell)+1,cell_face_offset_zero(cell+1)
    face=cell_faces_zero(slot)+1;sign=cell_face_sign(slot)
    hydrostatic=.5d0*gravity*q(cell,1)**2*normal(face,:)
    if(sign>0) then
     sum_flux(1)=sum_flux(1)+(face_length(face)/cell_area(cell))*flux_left(face,1)
     sum_flux(2:3)=sum_flux(2:3)+(face_length(face)/cell_area(cell))* &
      (flux_left(face,2:3)-hydrostatic)
    else
     sum_flux(1)=sum_flux(1)-(face_length(face)/cell_area(cell))*flux_right(face,1)
     sum_flux(2:3)=sum_flux(2:3)-(face_length(face)/cell_area(cell))* &
      (flux_right(face,2:3)-hydrostatic)
    endif
    if(density_gradient_enabled==1) then
     outward1=sign*normal(face,1);outward2=sign*normal(face,2)
     directional=sign*face_acceleration(face);weight=face_length(face)
     matrix11=matrix11+weight*outward1*outward1
     matrix12=matrix12+weight*outward1*outward2
     matrix22=matrix22+weight*outward2*outward2
     rhs1=rhs1+weight*outward1*directional
     rhs2=rhs2+weight*outward2*directional
    endif
   enddo
   base_rate(cell,:)=-sum_flux
   if(density_gradient_enabled==1) then
    determinant=matrix11*matrix22-matrix12*matrix12
    scale=max(1.d0,abs(matrix11),abs(matrix12),abs(matrix22))
    if(determinant<=256.d0*epsilon(1.d0)*scale*scale) then
     invalid=1
    else
     cell_acceleration(cell,1)=(matrix22*rhs1-matrix12*rhs2)/determinant
     cell_acceleration(cell,2)=(matrix11*rhs2-matrix12*rhs1)/determinant
     density_rate(cell,:)=q(cell,1)*cell_acceleration(cell,:)
    endif
   endif
   full_rate(cell,1)=base_rate(cell,1)
   full_rate(cell,2:3)=base_rate(cell,2:3)+density_rate(cell,:)
   if(.not.all(ieee_is_finite(base_rate(cell,:))).or. &
      .not.all(ieee_is_finite(density_rate(cell,:))).or. &
      .not.all(ieee_is_finite(full_rate(cell,:)))) invalid=1
  enddo
  !$omp end parallel do
  if(invalid/=0) then
   status=8;return
  endif
  density_source_calls=density_gradient_enabled
 end subroutine braids_coupled_open_full_rate_v96
end module braids_coupled_open_boundary_v96_module
