! Q4/v95 validation-only CPU residual composer.
!
! This module does not replace or modify GeoClaw's augmented Riemann solver.
! It calls the frozen rpn2 routine on every Cartesian face, assembles a
! first-order finite-volume hydraulic rate, and (when selected) composes the
! KMAX=1 Delft3D DensIn sediment-density increment inside that same full-rate
! evaluation.  Face and cell work is OpenMP parallel.  The Python owner drives
! the bounded implicit-midpoint iteration and polls cancellation between calls.
module braids_densin_residual_v95_module
 use iso_c_binding
 use, intrinsic :: ieee_arithmetic
 use omp_lib
 use geoclaw_module, only: grav, dry_tolerance
 implicit none
contains

 subroutine augmented_face_v95(left,right,bed_left,bed_right,unit_normal,flux_left,flux_right,speed)
  real(8),intent(in) :: left(3),right(3),bed_left,bed_right,unit_normal(2)
  real(8),intent(out) :: flux_left(3),flux_right(3),speed
  real(8) :: ql(3,0:2),qr(3,0:2),al(1,0:2),ar(1,0:2)
  real(8) :: fwave(3,3,0:2),wave_speed(3,0:2),amdq(3,0:2),apdq(3,0:2)
  real(8) :: local_left(3),local_right(3),physical_left(3),physical_right(3)
  real(8) :: normal_flux_left(3),normal_flux_right(3),tangent(2)
  integer :: j
  external rpn2

  tangent=[-unit_normal(2),unit_normal(1)]
  local_left=[left(1),dot_product(left(2:3),unit_normal),dot_product(left(2:3),tangent)]
  local_right=[right(1),dot_product(right(2:3),unit_normal),dot_product(right(2:3),tangent)]
  do j=0,2
   ql(:,j)=local_right
   qr(:,j)=local_left
  enddo
  al=bed_right
  ar=bed_left
  call rpn2(1,1,3,3,1,1,0,ql,qr,al,ar,fwave,wave_speed,amdq,apdq)

  physical_left=0.d0
  physical_right=0.d0
  if(local_left(1)>dry_tolerance) physical_left=[ &
   local_left(2),local_left(2)**2/local_left(1)+.5d0*grav*local_left(1)**2, &
   local_left(2)*local_left(3)/local_left(1)]
  if(local_right(1)>dry_tolerance) physical_right=[ &
   local_right(2),local_right(2)**2/local_right(1)+.5d0*grav*local_right(1)**2, &
   local_right(2)*local_right(3)/local_right(1)]
  normal_flux_left=physical_left+amdq(:,1)
  normal_flux_right=physical_right-apdq(:,1)
  flux_left=[normal_flux_left(1), &
   normal_flux_left(2)*unit_normal(1)+normal_flux_left(3)*tangent(1), &
   normal_flux_left(2)*unit_normal(2)+normal_flux_left(3)*tangent(2)]
  flux_right=[normal_flux_right(1), &
   normal_flux_right(2)*unit_normal(1)+normal_flux_right(3)*tangent(1), &
   normal_flux_right(2)*unit_normal(2)+normal_flux_right(3)*tangent(2)]
  ! The augmented solver produces one shared conservative water flux.  Average
  ! the two roundoff-level representations exactly as the BRAIDS wrapper does.
  flux_left(1)=.5d0*(flux_left(1)+flux_right(1))
  flux_right(1)=flux_left(1)
  speed=maxval(abs(wave_speed(:,1)))
 end subroutine augmented_face_v95

 subroutine braids_densin_full_rate_v95( &
  nc,nf,nadj,q,bed,owner_zero,neighbour_zero,normal,face_length,face_distance, &
  cell_area,cell_face_offset_zero,cell_faces_zero,cell_face_sign,mixture_density, &
  reference_density,gravity,dry,densin_enabled,threads,base_rate,density_rate, &
  full_rate,face_acceleration,cell_acceleration,face_water_flux,riemann_calls, &
  density_source_calls,status) bind(C,name='braids_densin_full_rate_v95')
  integer(c_int),value :: nc,nf,nadj,densin_enabled,threads
  real(c_double),intent(in) :: q(nc,3),bed(nc),normal(nf,2),face_length(nf)
  real(c_double),intent(in) :: face_distance(nf),cell_area(nc),mixture_density(nc)
  real(c_double),value :: reference_density,gravity,dry
  integer(c_int),intent(in) :: owner_zero(nf),neighbour_zero(nf)
  integer(c_int),intent(in) :: cell_face_offset_zero(nc+1),cell_faces_zero(nadj)
  integer(c_int),intent(in) :: cell_face_sign(nadj)
  real(c_double),intent(out) :: base_rate(nc,3),density_rate(nc,2),full_rate(nc,3)
  real(c_double),intent(out) :: face_acceleration(nf),cell_acceleration(nc,2)
  real(c_double),intent(out) :: face_water_flux(nf)
  integer(c_int),intent(out) :: riemann_calls,density_source_calls,status

  real(8) :: flux_left(nf,3),flux_right(nf,3),signal_speed(nf)
  real(8) :: left(3),right(3),unit_normal(2),tangent(2),normal_momentum,tangent_momentum,bed_right
  real(8) :: sum_flux(3),matrix11,matrix12,matrix22,rhs1,rhs2,determinant,scale
  real(8) :: outward1,outward2,directional,weight,hydrostatic(2)
  integer :: face,cell,slot,owner,neighbour,sign,invalid

  base_rate=0.d0
  density_rate=0.d0
  full_rate=0.d0
  face_acceleration=0.d0
  cell_acceleration=0.d0
  face_water_flux=0.d0
  riemann_calls=0
  density_source_calls=0
  status=0
  if(nc<1.or.nf<1.or.nadj<1.or.threads<1.or. &
     (densin_enabled/=0.and.densin_enabled/=1)) then
   status=1
   return
  endif
  if(reference_density<=0.d0.or.gravity<=0.d0.or.dry<0.d0.or. &
     .not.ieee_is_finite(reference_density).or..not.ieee_is_finite(gravity).or. &
     .not.ieee_is_finite(dry)) then
   status=2
   return
  endif
  if(.not.all(ieee_is_finite(q)).or..not.all(ieee_is_finite(bed)).or. &
     .not.all(ieee_is_finite(normal)).or..not.all(ieee_is_finite(face_length)).or. &
     .not.all(ieee_is_finite(face_distance)).or..not.all(ieee_is_finite(cell_area)).or. &
     .not.all(ieee_is_finite(mixture_density)).or.any(q(:,1)<=dry).or. &
     any(face_length<=0.d0).or.any(face_distance<=0.d0).or.any(cell_area<=0.d0).or. &
     any(mixture_density<=0.d0)) then
   status=3
   return
  endif
  if(cell_face_offset_zero(1)/=0.or.cell_face_offset_zero(nc+1)/=nadj) then
   status=4
   return
  endif
  do cell=1,nc
   if(cell_face_offset_zero(cell)>cell_face_offset_zero(cell+1)) then
    status=4
    return
   endif
  enddo
  do face=1,nf
   if(owner_zero(face)<0.or.owner_zero(face)>=nc.or.neighbour_zero(face)<-1.or. &
      neighbour_zero(face)>=nc.or.neighbour_zero(face)==owner_zero(face).or. &
      abs(sqrt(sum(normal(face,:)**2))-1.d0)>2.d-13.or. &
      min(abs(normal(face,1)),abs(normal(face,2)))>2.d-13) then
    status=5
    return
   endif
  enddo
  do slot=1,nadj
   if(cell_faces_zero(slot)<0.or.cell_faces_zero(slot)>=nf.or. &
      abs(cell_face_sign(slot))/=1) then
    status=6
    return
   endif
  enddo

  grav=gravity
  dry_tolerance=dry
  call omp_set_dynamic(.false.)
  call omp_set_num_threads(threads)
  invalid=0
  !$omp parallel do schedule(static) default(shared) &
  !$omp& private(face,owner,neighbour,left,right,unit_normal,tangent,normal_momentum,tangent_momentum,bed_right) &
  !$omp& reduction(max:invalid)
  do face=1,nf
   owner=owner_zero(face)+1
   neighbour=neighbour_zero(face)+1
   unit_normal=normal(face,:)
   left=q(owner,:)
   if(neighbour>0) then
    right=q(neighbour,:)
    bed_right=bed(neighbour)
   else
    ! Closed reflective physical boundary.  The density boundary separately
    ! uses zero normal gradient, hence its DensIn acceleration is exactly zero.
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
   if(.not.all(ieee_is_finite(flux_left(face,:))).or. &
      .not.all(ieee_is_finite(flux_right(face,:))).or. &
      .not.ieee_is_finite(signal_speed(face))) invalid=1
   if(densin_enabled==1.and.neighbour>0) then
    face_acceleration(face)= &
     -gravity*(mixture_density(neighbour)-mixture_density(owner))* &
     (q(neighbour,1)+q(owner,1))/(4.d0*reference_density*face_distance(face))
   else
    face_acceleration(face)=0.d0
   endif
  enddo
  !$omp end parallel do
  if(invalid/=0.or..not.all(ieee_is_finite(face_acceleration))) then
   status=7
   return
  endif

  invalid=0
  !$omp parallel do schedule(static) default(shared) &
  !$omp& private(cell,slot,face,sign,sum_flux,hydrostatic,matrix11,matrix12,matrix22, &
  !$omp& rhs1,rhs2,outward1,outward2,directional,weight,determinant,scale) reduction(max:invalid)
  do cell=1,nc
   sum_flux=0.d0
   matrix11=0.d0
   matrix12=0.d0
   matrix22=0.d0
   rhs1=0.d0
   rhs2=0.d0
   do slot=cell_face_offset_zero(cell)+1,cell_face_offset_zero(cell+1)
    face=cell_faces_zero(slot)+1
    sign=cell_face_sign(slot)
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
    if(densin_enabled==1) then
     outward1=sign*normal(face,1)
     outward2=sign*normal(face,2)
     directional=sign*face_acceleration(face)
     weight=face_length(face)
     matrix11=matrix11+weight*outward1*outward1
     matrix12=matrix12+weight*outward1*outward2
     matrix22=matrix22+weight*outward2*outward2
     rhs1=rhs1+weight*outward1*directional
     rhs2=rhs2+weight*outward2*directional
    endif
   enddo
   base_rate(cell,:)=-sum_flux
   if(densin_enabled==1) then
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
   status=8
   return
  endif
  riemann_calls=nf
  if(densin_enabled==1) density_source_calls=1
 end subroutine braids_densin_full_rate_v95
end module braids_densin_residual_v95_module
