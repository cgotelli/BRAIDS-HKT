! SPDX-License-Identifier: AGPL-3.0-or-later
!
! Distributed noncohesive slope-bank supply.
!
! This is the local face form used by Parsapour-Moghaddam et al. (2023):
!
!   q_bank = alpha * abs(sum_g(q_toe,g) dot t_face)
!   when S_net = abs(z_bank-z_toe)/abs(x_bank-x_toe) > S_repose.
!
! q_toe,g is the reconstructed accepted bed-load mass-flux vector in the lower,
! channelward cell.  The production caller derives it from the same limited,
! SSPRK-weighted and (when active) refluxed face flux that advanced Exner.
! Total supply is partitioned using the active-layer mass composition of the
! higher erodible-bank donor, then moved to that lower cell. Transfers are
! positive in the native owner-to-neighbour convention and are ready for
! apply_surface_transfer.
!
! This module intentionally does not limit donor inventory. The production
! apply_surface_transfer routine owns that operation, scales every concurrent
! outgoing request from a donor once, and applies one conservative gather.
! Keeping the constitutive law separate avoids two order-dependent limiters.
module braids_distributed_bank_module
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 implicit none
 private

 integer,parameter,public :: BRAIDS_DISTRIBUTED_BANK_SUCCESS=0
 integer,parameter,public :: BRAIDS_DISTRIBUTED_BANK_BAD_DIMENSIONS=-1
 integer,parameter,public :: BRAIDS_DISTRIBUTED_BANK_BAD_DATA=-2
 integer,parameter,public :: BRAIDS_DISTRIBUTED_BANK_BAD_TOPOLOGY=-3
 integer,parameter,public :: BRAIDS_DISTRIBUTED_BANK_BAD_GEOMETRY=-4

 public :: braids_distributed_bank_face_transfers

contains

 pure subroutine braids_distributed_bank_face_transfers(nc,nf,nfrac,owner, &
      neighbour,cell_xy,bed,erodible_bank,face_normal,face_length, &
      toe_bedload,donor_active_mass,alpha,critical_slope, &
      transport_duration,face_transfer,eligible_face,donor_cell, &
      recipient_cell,local_slope,tangential_toe_transport,ierr,face_allowed)
  ! Form requested bank-to-channel sediment transfers on local slope faces.
  !
  ! Units:
  !   cell_xy, bed, face_length                 [m]
  !   toe_bedload                              [kg m-1 s-1]
  !   donor_active_mass                        [kg m-2]
  !   transport_duration                       [s]
  !   face_transfer                            [kg]
  !
  ! ``transport_duration`` is supplied by the time integrator. Any explicitly
  ! enabled morphological acceleration must appear exactly once across
  ! ``toe_bedload`` and this duration; the production accepted-flux caller has
  ! it already embedded in ``toe_bedload`` and therefore passes physical time.
  ! ``face_allowed`` is an optional physical-support/material-interface mask;
  ! it must not be a reach-specific calibration selector. No water-depth input
  ! is used: submerged and dry erodible-bank donors obey the same law.
  integer,intent(in) :: nc,nf,nfrac
  integer,intent(in) :: owner(nf),neighbour(nf)
  real(8),intent(in) :: cell_xy(2,nc),bed(nc)
  logical,intent(in) :: erodible_bank(nc)
  real(8),intent(in) :: face_normal(2,nf),face_length(nf)
  real(8),intent(in) :: toe_bedload(2,nfrac,nc)
  real(8),intent(in) :: donor_active_mass(nfrac,nc)
  real(8),intent(in) :: alpha,critical_slope,transport_duration
  real(8),intent(out) :: face_transfer(nfrac,nf)
  logical,intent(out) :: eligible_face(nf)
  integer,intent(out) :: donor_cell(nf),recipient_cell(nf)
  real(8),intent(out) :: local_slope(nf),tangential_toe_transport(nf)
  integer,intent(out) :: ierr
  logical,intent(in),optional :: face_allowed(nf)

  integer :: c,f,g,high,low,orientation
  real(8) :: active_total,amount,delta_z,distance,normal_size
  real(8) :: tangent(2),total_toe_bedload(2),unit_normal(2)

  call clear_outputs(face_transfer,eligible_face,donor_cell,recipient_cell, &
       local_slope,tangential_toe_transport)
  ierr=BRAIDS_DISTRIBUTED_BANK_SUCCESS

  if(nc<=0.or.nf<0.or.nfrac<=0) then
   ierr=BRAIDS_DISTRIBUTED_BANK_BAD_DIMENSIONS
   return
  endif
  if(.not.ieee_is_finite(alpha).or.alpha<0.d0.or.alpha>1.d0.or. &
     .not.ieee_is_finite(critical_slope).or.critical_slope<0.d0.or. &
     .not.ieee_is_finite(transport_duration).or.transport_duration<0.d0.or. &
     .not.all(ieee_is_finite(cell_xy)).or. &
     .not.all(ieee_is_finite(bed)).or. &
     .not.all(ieee_is_finite(toe_bedload)).or. &
     .not.all(ieee_is_finite(donor_active_mass)).or. &
     any(donor_active_mass<0.d0)) then
   ierr=BRAIDS_DISTRIBUTED_BANK_BAD_DATA
   return
  endif
  do c=1,nc
   active_total=sum(donor_active_mass(:,c))
   total_toe_bedload=sum(toe_bedload(:,:,c),dim=2)
   if(.not.ieee_is_finite(active_total).or. &
      .not.all(ieee_is_finite(total_toe_bedload))) then
    ierr=BRAIDS_DISTRIBUTED_BANK_BAD_DATA
    return
   endif
  enddo

  ! Validate the complete face geometry before emitting any partial request.
  do f=1,nf
   if(owner(f)<1.or.owner(f)>nc.or.neighbour(f)<0.or. &
      neighbour(f)>nc.or.neighbour(f)==owner(f)) then
    ierr=BRAIDS_DISTRIBUTED_BANK_BAD_TOPOLOGY
    call clear_outputs(face_transfer,eligible_face,donor_cell,recipient_cell, &
         local_slope,tangential_toe_transport)
    return
   endif
   if(.not.ieee_is_finite(face_length(f)).or.face_length(f)<=0.d0.or. &
      .not.all(ieee_is_finite(face_normal(:,f)))) then
    ierr=BRAIDS_DISTRIBUTED_BANK_BAD_GEOMETRY
    call clear_outputs(face_transfer,eligible_face,donor_cell,recipient_cell, &
         local_slope,tangential_toe_transport)
    return
   endif
   normal_size=sqrt(sum(face_normal(:,f)**2))
   if(.not.ieee_is_finite(normal_size).or.normal_size<=0.d0) then
    ierr=BRAIDS_DISTRIBUTED_BANK_BAD_GEOMETRY
    call clear_outputs(face_transfer,eligible_face,donor_cell,recipient_cell, &
         local_slope,tangential_toe_transport)
    return
   endif
   if(neighbour(f)>0) then
    distance=sqrt(sum((cell_xy(:,owner(f))-cell_xy(:,neighbour(f)))**2))
    if(.not.ieee_is_finite(distance).or.distance<=0.d0) then
     ierr=BRAIDS_DISTRIBUTED_BANK_BAD_GEOMETRY
     call clear_outputs(face_transfer,eligible_face,donor_cell,recipient_cell, &
          local_slope,tangential_toe_transport)
     return
    endif
   endif
  enddo

  do f=1,nf
   if(neighbour(f)<=0) cycle
   if(present(face_allowed)) then
    if(.not.face_allowed(f)) cycle
   endif

   if(bed(owner(f))>bed(neighbour(f))) then
    high=owner(f)
    low=neighbour(f)
    orientation=1
   else if(bed(neighbour(f))>bed(owner(f))) then
    high=neighbour(f)
    low=owner(f)
    orientation=-1
   else
    cycle
   endif
   delta_z=bed(high)-bed(low)
   distance=sqrt(sum((cell_xy(:,high)-cell_xy(:,low))**2))
   local_slope(f)=delta_z/distance
   if(.not.erodible_bank(high).or.local_slope(f)<=critical_slope) cycle

   eligible_face(f)=.true.
   donor_cell(f)=high
   recipient_cell(f)=low
   normal_size=sqrt(sum(face_normal(:,f)**2))
   unit_normal=face_normal(:,f)/normal_size
   tangent=[-unit_normal(2),unit_normal(1)]
   total_toe_bedload=sum(toe_bedload(:,:,low),dim=2)
   tangential_toe_transport(f)= &
        abs(dot_product(total_toe_bedload,tangent))
   active_total=sum(donor_active_mass(:,high))
   if(active_total<=0.d0) cycle
   amount=alpha*transport_duration*face_length(f)* &
        tangential_toe_transport(f)
   do g=1,nfrac
    face_transfer(g,f)=real(orientation,8)*amount* &
         donor_active_mass(g,high)/active_total
   enddo
  enddo

 contains

  pure subroutine clear_outputs(transfer,eligible,donor,recipient,slope, &
       tangential_transport)
   real(8),intent(out) :: transfer(:,:),slope(:),tangential_transport(:)
   logical,intent(out) :: eligible(:)
   integer,intent(out) :: donor(:),recipient(:)
   transfer=0.d0
   eligible=.false.
   donor=0
   recipient=0
   slope=0.d0
   tangential_transport=0.d0
  end subroutine clear_outputs

 end subroutine braids_distributed_bank_face_transfers

end module braids_distributed_bank_module
