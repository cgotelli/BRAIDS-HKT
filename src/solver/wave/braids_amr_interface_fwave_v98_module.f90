! Q4/v98 additive 2:1 coarse/fine trace residual.
!
! The frozen v97 fixed-grid routine remains unchanged.  This wrapper first
! evaluates that exact residual, replaces only authenticated negative-neighbour
! interface faces by an external coarse trace using the same augmented f-wave
! face routine, then rebuilds the donor limiter and cell gather.  A trace face
! is internal AMR coupling; physical boundary ownership is outside this ABI.
module braids_amr_interface_fwave_v98_module
 use iso_c_binding
 use, intrinsic :: ieee_arithmetic
 use omp_lib
 use braids_densin_residual_v95_module, only: augmented_face_v95
 use braids_wetdry_fwave_v97_module, only: braids_wetdry_fwave_rate_v97
 implicit none
contains

 subroutine braids_amr_interface_fwave_rate_v98( &
  nc,nf,nadj,q,bed,owner_zero,neighbour_zero,normal,face_length,face_distance, &
  cell_area,cell_face_offset_zero,cell_faces_zero,cell_face_sign,domain_active, &
  frozen_wet,accepted_density,reference_density,gravity,dry,dt,threads, &
  trace_active,trace_q,trace_bed,trace_density,trace_wet, &
  raw_left,raw_right,limited_left,limited_right,theta,base_rate, &
  face_water_flux,face_acceleration,wet_wet_face,riemann_calls, &
  limiter_application_count,interface_riemann_calls,status) &
  bind(C,name='braids_amr_interface_fwave_rate_v98')
  integer(c_int),value :: nc,nf,nadj,threads
  real(c_double),value :: reference_density,gravity,dry,dt
  real(c_double),intent(in) :: q(nc,3),bed(nc),normal(nf,2),face_length(nf)
  real(c_double),intent(in) :: face_distance(nf),cell_area(nc),accepted_density(nc)
  integer(c_int),intent(in) :: owner_zero(nf),neighbour_zero(nf)
  integer(c_int),intent(in) :: cell_face_offset_zero(nc+1),cell_faces_zero(nadj)
  integer(c_int),intent(in) :: cell_face_sign(nadj),domain_active(nc),frozen_wet(nc)
  integer(c_int),intent(in) :: trace_active(nf),trace_wet(nf)
  real(c_double),intent(in) :: trace_q(nf,3),trace_bed(nf),trace_density(nf)
  real(c_double),intent(out) :: raw_left(nf,3),raw_right(nf,3)
  real(c_double),intent(out) :: limited_left(nf,3),limited_right(nf,3)
  real(c_double),intent(out) :: theta(nc),base_rate(nc,3),face_water_flux(nf)
  real(c_double),intent(out) :: face_acceleration(nf)
  integer(c_int),intent(out) :: wet_wet_face(nf)
  integer(c_int),intent(out) :: riemann_calls,limiter_application_count
  integer(c_int),intent(out) :: interface_riemann_calls,status

  real(8) :: left(3),right(3),unit_normal(2),speed
  real(8) :: outgoing,factor,correction,velocity
  real(8) :: sum_flux(3),hydrostatic(2),hcell
  integer :: face,cell,slot,owner,neighbour,sign,donor,component,invalid
  integer :: base_status,base_riemann,base_limiter,limited_faces,trace_calls

  interface_riemann_calls=0; status=0
  if(any((trace_active/=0).and.(trace_active/=1)).or. &
     any((trace_wet/=0).and.(trace_wet/=1)).or. &
     any((trace_active==0).and.(trace_wet/=0)).or. &
     .not.all(ieee_is_finite(trace_q)).or. &
     .not.all(ieee_is_finite(trace_bed)).or. &
     .not.all(ieee_is_finite(trace_density)).or. &
     any(trace_q(:,1)<0.d0).or.any(trace_density<=0.d0).or. &
     any((trace_wet==0).and.(abs(trace_q(:,1))+abs(trace_q(:,2))+ &
         abs(trace_q(:,3))>0.d0)).or. &
     any((trace_wet==1).and.(trace_q(:,1)<=dry))) then
   status=101; return
  endif
  do face=1,nf
   if(trace_active(face)==1.and.neighbour_zero(face)/=-1) then
    status=102; return
   endif
   if(trace_active(face)==0.and.(any(trace_q(face,:)/=0.d0).or. &
      trace_bed(face)/=0.d0.or.trace_wet(face)/=0)) then
    status=103; return
   endif
  enddo

  call braids_wetdry_fwave_rate_v97( &
   nc,nf,nadj,q,bed,owner_zero,neighbour_zero,normal,face_length,face_distance, &
   cell_area,cell_face_offset_zero,cell_faces_zero,cell_face_sign,domain_active, &
   frozen_wet,accepted_density,reference_density,gravity,dry,dt,threads, &
   raw_left,raw_right,limited_left,limited_right,theta,base_rate, &
   face_water_flux,face_acceleration,wet_wet_face,base_riemann, &
   base_limiter,base_status)
  if(base_status/=0) then
   status=200+base_status; return
  endif

  invalid=0; trace_calls=0
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) &
  !$omp& private(face,owner,left,right,unit_normal,speed) reduction(max:invalid) reduction(+:trace_calls)
  do face=1,nf
   if(trace_active(face)==1) then
    owner=owner_zero(face)+1; unit_normal=normal(face,:)
    ! This routine owns only the augmented f-wave replacement.  The additive
    ! v98 density-trace projection applies exterior DensIn separately to the
    ! gathered momentum rate.  Keeping this face out of wet_wet_face preserves
    ! the historical internal-only v91 mask; density and viscosity therefore
    ! cannot be accidentally coupled through one overloaded face mask.
    if(frozen_wet(owner)==1) then
     left=q(owner,:)
    else
     left=0.d0
    endif
    if(trace_wet(face)==1) then
     right=trace_q(face,:)
    else
     right=0.d0
    endif
    call augmented_face_v95(left,right,bed(owner),trace_bed(face), &
     unit_normal,raw_left(face,:),raw_right(face,:),speed)
    raw_left(face,1)=.5d0*(raw_left(face,1)+raw_right(face,1))
    raw_right(face,1)=raw_left(face,1)
    wet_wet_face(face)=0
    face_acceleration(face)=0.d0
    if(.not.all(ieee_is_finite(raw_left(face,:))).or. &
       .not.all(ieee_is_finite(raw_right(face,:))).or. &
       .not.ieee_is_finite(speed).or. &
       .not.ieee_is_finite(face_acceleration(face))) invalid=1
    trace_calls=trace_calls+1
   endif
  enddo
  !$omp end parallel do
  if(invalid/=0.or.trace_calls/=count(trace_active==1)) then
   status=104; return
  endif

  theta=1.d0
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) &
  !$omp& private(cell,slot,face,sign,outgoing)
  do cell=1,nc
   outgoing=0.d0
   do slot=cell_face_offset_zero(cell)+1,cell_face_offset_zero(cell+1)
    face=cell_faces_zero(slot)+1; sign=cell_face_sign(slot)
    outgoing=outgoing+max(sign*raw_left(face,1),0.d0)*face_length(face)
   enddo
   if(outgoing>0.d0.and.dt*outgoing>= &
      merge(q(cell,1),0.d0,frozen_wet(cell)==1)*cell_area(cell)) then
    theta(cell)=min(1.d0, &
     merge(q(cell,1),0.d0,frozen_wet(cell)==1)*cell_area(cell)/(dt*outgoing))* &
     (1.d0-32.d0*epsilon(1.d0))
   endif
  enddo
  !$omp end parallel do

  limited_left=raw_left; limited_right=raw_right
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
    ! Negative flux on an AMR trace is supplied by the separately owned
    ! coarse cell.  Coarse positivity is checked after the one reflux.
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
      .not.all(ieee_is_finite(limited_right(face,:)))) invalid=1
  enddo
  !$omp end parallel do
  if(invalid/=0.or.any(theta<0.d0).or.any(theta>1.d0)) then
   status=105; return
  endif

  invalid=0
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) &
  !$omp& private(cell,slot,face,sign,sum_flux,hydrostatic,hcell) reduction(max:invalid)
  do cell=1,nc
   sum_flux=0.d0; hcell=merge(q(cell,1),0.d0,frozen_wet(cell)==1)
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
   status=106; return
  endif
  riemann_calls=base_riemann
  limiter_application_count=limited_faces
  interface_riemann_calls=trace_calls
 end subroutine braids_amr_interface_fwave_rate_v98
end module braids_amr_interface_fwave_v98_module
