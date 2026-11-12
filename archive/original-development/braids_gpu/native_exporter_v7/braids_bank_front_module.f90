! SPDX-License-Identifier: AGPL-3.0-or-later
!
! Conservative persistent subcell bank-front state primitives.
!
! The routines in this module do not choose a bank-erosion law.  They turn
! the actual per-fraction mass accepted by such a law into an unresolved
! vertical bank-prism retreat.  While a front is active, the effective bed
! closure keeps the top of that prism fixed.  Once its remaining plan area is
! exhausted, the caller may expose the cell without creating or deleting any
! sediment.  Geometry, hydraulics, AMR and checkpoint ownership stay with the
! caller so this module can be qualified independently of the full solver.
module braids_bank_front_module
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 implicit none
 private

 integer,parameter,public :: BANK_FRONT_INACTIVE=0
 integer,parameter,public :: BANK_FRONT_ACTIVE=1
 integer,parameter,public :: BANK_FRONT_PROMOTED=2

 integer,parameter,public :: BANK_FRONT_SUCCESS=0
 integer,parameter,public :: BANK_FRONT_BAD_DIMENSIONS=-1
 integer,parameter,public :: BANK_FRONT_BAD_DATA=-2
 integer,parameter,public :: BANK_FRONT_BAD_STATE=-3
 integer,parameter,public :: BANK_FRONT_BAD_GEOMETRY=-4
 integer,parameter,public :: BANK_FRONT_EXCEEDS_CAPACITY=-5

 public :: braids_bank_front_activate
 public :: braids_bank_front_effective_bed
 public :: braids_bank_front_capacity
 public :: braids_bank_front_limit_request
 public :: braids_bank_front_apply_actual_mass
 public :: braids_bank_front_apply_batch
 public :: braids_bank_front_exclude_incoming
 public :: braids_bank_front_validate
 public :: braids_bank_front_seed_plane

contains

 pure real(8) function scaled_roundoff(scale) result(tolerance)
  real(8),intent(in) :: scale
  tolerance=256.d0*epsilon(1.d0)*max(abs(scale),tiny(1.d0))
 end function scaled_roundoff


 pure subroutine braids_bank_front_activate(cell_area,raw_bed,base_elevation, &
      candidate_seed_face,status,seed_face,solid_area,reference_height, &
      cumulative_removed_mass,ierr)
  ! Activate one previously unresolved solid bank cell.  The reference
  ! height and base elevation are immutable for this traversal.  No material
  ! is moved here; cumulative mass starts at exactly zero.
  real(8),intent(in) :: cell_area,raw_bed,base_elevation
  integer,intent(in) :: candidate_seed_face
  integer,intent(inout) :: status,seed_face
  real(8),intent(inout) :: solid_area,reference_height
  real(8),intent(inout) :: cumulative_removed_mass(:)
  integer,intent(out) :: ierr
  real(8) :: height

  ierr=BANK_FRONT_SUCCESS
  if(size(cumulative_removed_mass)<=0) then
   ierr=BANK_FRONT_BAD_DIMENSIONS;return
  endif
  if(status/=BANK_FRONT_INACTIVE) then
   ierr=BANK_FRONT_BAD_STATE;return
  endif
  if(candidate_seed_face<=0) then
   ierr=BANK_FRONT_BAD_GEOMETRY;return
  endif
  if(.not.ieee_is_finite(cell_area).or.cell_area<=0.d0.or. &
     .not.ieee_is_finite(raw_bed).or. &
     .not.ieee_is_finite(base_elevation)) then
   ierr=BANK_FRONT_BAD_DATA;return
  endif
  height=raw_bed-base_elevation
  if(.not.ieee_is_finite(height).or.height<=0.d0) then
   ierr=BANK_FRONT_BAD_DATA;return
  endif

  status=BANK_FRONT_ACTIVE
  seed_face=candidate_seed_face
  solid_area=cell_area
  reference_height=height
  cumulative_removed_mass=0.d0
 end subroutine braids_bank_front_activate


 pure subroutine braids_bank_front_effective_bed(raw_bed,cell_area,status, &
      solid_area,base_elevation,effective_bed,ierr)
  ! The native inventory is stored per full cell area.  If a vertical prism
  ! of constant height H occupies only fraction phi of the plan area, its
  ! unresolved top is
  !
  !   z_eff = z_base + (z_raw-z_base)/phi .
  !
  ! Since actual removed volume lowers z_raw by dV/A while phi falls by
  ! dV/(A*H), z_eff remains exactly at the activation top.  Promoted and
  ! inactive cells use the ordinary inventory-derived bed.
  real(8),intent(in) :: raw_bed,cell_area,solid_area,base_elevation
  integer,intent(in) :: status
  real(8),intent(out) :: effective_bed
  integer,intent(out) :: ierr
  real(8) :: tolerance

  effective_bed=0.d0;ierr=BANK_FRONT_SUCCESS
  if(.not.ieee_is_finite(raw_bed).or. &
     .not.ieee_is_finite(cell_area).or.cell_area<=0.d0.or. &
     .not.ieee_is_finite(solid_area).or. &
     .not.ieee_is_finite(base_elevation)) then
   ierr=BANK_FRONT_BAD_DATA;return
  endif
  if(status/=BANK_FRONT_INACTIVE.and.status/=BANK_FRONT_ACTIVE.and. &
     status/=BANK_FRONT_PROMOTED) then
   ierr=BANK_FRONT_BAD_STATE;return
  endif
  if(status/=BANK_FRONT_ACTIVE) then
   effective_bed=raw_bed;return
  endif
  if(solid_area<=0.d0.or.solid_area>cell_area) then
   ierr=BANK_FRONT_BAD_STATE;return
  endif
  tolerance=scaled_roundoff(max(abs(raw_bed),abs(base_elevation)))
  if(raw_bed<base_elevation-tolerance) then
   ierr=BANK_FRONT_BAD_STATE;return
  endif
  effective_bed=base_elevation+cell_area*max(raw_bed-base_elevation,0.d0)/solid_area
  if(.not.ieee_is_finite(effective_bed)) then
   effective_bed=0.d0;ierr=BANK_FRONT_BAD_DATA
  endif
 end subroutine braids_bank_front_effective_bed


 pure subroutine braids_bank_front_capacity(status,solid_area, &
      reference_height,capacity,ierr)
  integer,intent(in) :: status
  real(8),intent(in) :: solid_area,reference_height
  real(8),intent(out) :: capacity
  integer,intent(out) :: ierr

  capacity=0.d0;ierr=BANK_FRONT_SUCCESS
  if(.not.ieee_is_finite(solid_area).or. &
     .not.ieee_is_finite(reference_height)) then
   ierr=BANK_FRONT_BAD_DATA;return
  endif
  select case(status)
  case(BANK_FRONT_ACTIVE)
   if(solid_area<=0.d0.or.reference_height<=0.d0) then
    ierr=BANK_FRONT_BAD_STATE;return
   endif
   capacity=solid_area*reference_height
   if(.not.ieee_is_finite(capacity)) then
    capacity=0.d0;ierr=BANK_FRONT_BAD_DATA
   endif
  case(BANK_FRONT_INACTIVE,BANK_FRONT_PROMOTED)
   if(solid_area<0.d0.or.reference_height<0.d0) ierr=BANK_FRONT_BAD_STATE
  case default
   ierr=BANK_FRONT_BAD_STATE
  end select
 end subroutine braids_bank_front_capacity


 pure subroutine braids_bank_front_limit_request(status,solid_area, &
      reference_height,requested_volume,accepted_volume,residual_volume,ierr)
  ! Capacity limiting is transactional.  The residual is explicit so a
  ! caller can carry it through a deterministic promotion/activation event;
  ! it must never be silently discarded.
  integer,intent(in) :: status
  real(8),intent(in) :: solid_area,reference_height,requested_volume
  real(8),intent(out) :: accepted_volume,residual_volume
  integer,intent(out) :: ierr
  real(8) :: capacity

  accepted_volume=0.d0;residual_volume=0.d0;ierr=BANK_FRONT_SUCCESS
  if(.not.ieee_is_finite(requested_volume).or.requested_volume<0.d0) then
   ierr=BANK_FRONT_BAD_DATA;return
  endif
  call braids_bank_front_capacity(status,solid_area,reference_height,capacity,ierr)
  if(ierr/=BANK_FRONT_SUCCESS) return
  if(status/=BANK_FRONT_ACTIVE) then
   if(requested_volume>0.d0) then
    ierr=BANK_FRONT_BAD_STATE
   else
    residual_volume=0.d0
   endif
   return
  endif
  accepted_volume=min(requested_volume,capacity)
  residual_volume=requested_volume-accepted_volume
 end subroutine braids_bank_front_limit_request


 pure subroutine braids_bank_front_apply_actual_mass(bulk_density,actual_mass, &
      cell_area,status,solid_area,reference_height,cumulative_removed_mass, &
      actual_volume,promoted,ierr)
  ! Advance geometry from ACTUAL inventory-limited mass only.  Mass is not
  ! created, routed or deposited here.  On failure no persistent state is
  ! changed.  A roundoff-sized final remainder is the only permitted snap to
  ! zero plan area.
  real(8),intent(in) :: bulk_density(:),actual_mass(:),cell_area
  integer,intent(inout) :: status
  real(8),intent(inout) :: solid_area,reference_height
  real(8),intent(inout) :: cumulative_removed_mass(:)
  real(8),intent(out) :: actual_volume
  logical,intent(out) :: promoted
  integer,intent(out) :: ierr
  real(8) :: capacity,new_area,volume_tolerance,area_tolerance
  real(8) :: initial_volume,removed_volume,remaining_volume,state_residual
  real(8) :: new_removed(size(cumulative_removed_mass))
  integer :: g,new_status

  actual_volume=0.d0;promoted=.false.;ierr=BANK_FRONT_SUCCESS
  if(size(bulk_density)<=0.or.size(actual_mass)/=size(bulk_density).or. &
     size(cumulative_removed_mass)/=size(bulk_density)) then
   ierr=BANK_FRONT_BAD_DIMENSIONS;return
  endif
  if(.not.all(ieee_is_finite(bulk_density)).or.any(bulk_density<=0.d0).or. &
     .not.all(ieee_is_finite(actual_mass)).or.any(actual_mass<0.d0).or. &
     .not.all(ieee_is_finite(cumulative_removed_mass)).or. &
     any(cumulative_removed_mass<0.d0).or. &
     .not.ieee_is_finite(cell_area).or.cell_area<=0.d0.or. &
     .not.ieee_is_finite(solid_area).or. &
     .not.ieee_is_finite(reference_height)) then
   ierr=BANK_FRONT_BAD_DATA;return
  endif
  if(status/=BANK_FRONT_ACTIVE.or.solid_area<=0.d0.or. &
     solid_area>cell_area.or.reference_height<=0.d0) then
   ierr=BANK_FRONT_BAD_STATE;return
  endif
  do g=1,size(bulk_density)
   actual_volume=actual_volume+actual_mass(g)/bulk_density(g)
  enddo
  if(.not.ieee_is_finite(actual_volume)) then
   actual_volume=0.d0;ierr=BANK_FRONT_BAD_DATA;return
  endif
  initial_volume=cell_area*reference_height
  removed_volume=0.d0
  do g=1,size(bulk_density)
   removed_volume=removed_volume+cumulative_removed_mass(g)/bulk_density(g)
  enddo
  capacity=initial_volume-removed_volume
  state_residual=capacity-solid_area*reference_height
  volume_tolerance=scaled_roundoff(max(initial_volume,max(removed_volume, &
       solid_area*reference_height)))
  if(capacity<0.d0.and.abs(capacity)>volume_tolerance) then
   actual_volume=0.d0;ierr=BANK_FRONT_BAD_STATE;return
  endif
  if(abs(state_residual)>volume_tolerance) then
   actual_volume=0.d0;ierr=BANK_FRONT_BAD_STATE;return
  endif
  capacity=max(capacity,0.d0)
  if(actual_volume>capacity+volume_tolerance) then
   actual_volume=0.d0;ierr=BANK_FRONT_EXCEEDS_CAPACITY;return
  endif
  new_removed=cumulative_removed_mass+actual_mass
  if(.not.all(ieee_is_finite(new_removed))) then
   actual_volume=0.d0;ierr=BANK_FRONT_BAD_DATA;return
  endif
  removed_volume=0.d0
  do g=1,size(bulk_density)
   removed_volume=removed_volume+new_removed(g)/bulk_density(g)
  enddo
  remaining_volume=initial_volume-removed_volume
  if(remaining_volume<0.d0.and.abs(remaining_volume)>volume_tolerance) then
   actual_volume=0.d0;ierr=BANK_FRONT_EXCEEDS_CAPACITY;return
  endif
  ! Cumulative removed mass is the authoritative front ledger.  Deriving the
  ! remaining plan area from that one ledger prevents independent roundoff in
  ! repeated ``solid_area -= dV/H`` updates from becoming a false invariant
  ! failure after many small accepted transfers.
  new_area=max(remaining_volume,0.d0)/reference_height
  area_tolerance=scaled_roundoff(cell_area)
  if(new_area>cell_area) then
   if(new_area-cell_area>area_tolerance) then
    actual_volume=0.d0;ierr=BANK_FRONT_BAD_STATE;return
   endif
   ! The multiply/subtract/divide round trip can land one ULP above the known
   ! cell area when the accepted removal is smaller than area resolution.
   new_area=cell_area
  endif
  new_status=status
  if(new_area<=area_tolerance) then
   new_area=0.d0;new_status=BANK_FRONT_PROMOTED;promoted=.true.
  endif

  solid_area=new_area
  status=new_status
  cumulative_removed_mass=new_removed
 end subroutine braids_bank_front_apply_actual_mass


 subroutine braids_bank_front_apply_batch(nfront,nfrac,bulk_density,cell_area, &
      status,solid_area,reference_height,cumulative_removed_mass,actual_mass, &
      actual_volume,promoted,ierr)
  ! Fronts are independent.  Static OpenMP scheduling changes only ownership,
  ! never arithmetic order inside a front, so one and many threads are
  ! bitwise identical.
  integer,intent(in) :: nfront,nfrac
  real(8),intent(in) :: bulk_density(nfrac),cell_area(nfront)
  integer,intent(inout) :: status(nfront)
  real(8),intent(inout) :: solid_area(nfront),reference_height(nfront)
  real(8),intent(inout) :: cumulative_removed_mass(nfrac,nfront)
  real(8),intent(in) :: actual_mass(nfrac,nfront)
  real(8),intent(out) :: actual_volume(nfront)
  logical,intent(out) :: promoted(nfront)
  integer,intent(out) :: ierr(nfront)
  integer :: c

  !$omp parallel do schedule(static) default(shared) private(c)
  do c=1,nfront
   call braids_bank_front_apply_actual_mass(bulk_density,actual_mass(:,c), &
        cell_area(c),status(c),solid_area(c),reference_height(c), &
        cumulative_removed_mass(:,c),actual_volume(c),promoted(c),ierr(c))
  enddo
  !$omp end parallel do
 end subroutine braids_bank_front_apply_batch


 pure subroutine braids_bank_front_exclude_incoming(ncell,nface,nfrac,status, &
      eligible,donor_cell,recipient_cell,face_mass,suppressed_faces,ierr)
  ! The current persistent front is erosion-only.  A cell which already has
  ! an active front, or which has accepted outgoing bank mass in this batch,
  ! must not simultaneously receive bank material: there is no reverse-front
  ! accretion state with which to update its geometry.  Zeroing the complete
  ! donor-to-recipient face transaction preserves mass exactly.
  integer,intent(in) :: ncell,nface,nfrac,status(ncell)
  logical,intent(inout) :: eligible(nface)
  integer,intent(in) :: donor_cell(nface),recipient_cell(nface)
  real(8),intent(inout) :: face_mass(nfrac,nface)
  integer,intent(out) :: suppressed_faces,ierr
  logical :: protected(ncell)
  integer :: c,f

  suppressed_faces=0;ierr=BANK_FRONT_SUCCESS
  if(ncell<=0.or.nface<0.or.nfrac<=0) then
   ierr=BANK_FRONT_BAD_DIMENSIONS;return
  endif
  if(.not.all(ieee_is_finite(face_mass))) then
   ierr=BANK_FRONT_BAD_DATA;return
  endif
  if(any(status/=BANK_FRONT_INACTIVE.and.status/=BANK_FRONT_ACTIVE.and. &
         status/=BANK_FRONT_PROMOTED)) then
   ierr=BANK_FRONT_BAD_STATE;return
  endif
  protected=status==BANK_FRONT_ACTIVE
  do f=1,nface
   if(.not.eligible(f).or.maxval(abs(face_mass(:,f)))<=0.d0) cycle
   c=donor_cell(f)
   if(c<1.or.c>ncell.or.recipient_cell(f)<1.or.recipient_cell(f)>ncell) then
    ierr=BANK_FRONT_BAD_DATA;return
   endif
   protected(c)=.true.
  enddo
  do f=1,nface
   if(.not.eligible(f).or.maxval(abs(face_mass(:,f)))<=0.d0) cycle
   c=recipient_cell(f)
   if(.not.protected(c)) cycle
   face_mass(:,f)=0.d0
   eligible(f)=.false.
   suppressed_faces=suppressed_faces+1
  enddo
 end subroutine braids_bank_front_exclude_incoming


 pure subroutine braids_bank_front_validate(bulk_density,cell_area,status, &
      seed_face,solid_area,reference_height,cumulative_removed_mass, &
      volume_residual,ierr)
  ! Validate the persistent mass/geometry invariant.  Promoted cells retain
  ! their history even though ordinary morphodynamics may subsequently alter
  ! their raw bed, so raw elevation is intentionally not part of this state
  ! check.
  real(8),intent(in) :: bulk_density(:),cell_area,solid_area,reference_height
  integer,intent(in) :: status,seed_face
  real(8),intent(in) :: cumulative_removed_mass(:)
  real(8),intent(out) :: volume_residual
  integer,intent(out) :: ierr
  real(8) :: initial_volume,removed_volume,remaining_volume,tolerance
  integer :: g

  volume_residual=0.d0;ierr=BANK_FRONT_SUCCESS
  if(size(bulk_density)<=0.or. &
     size(cumulative_removed_mass)/=size(bulk_density)) then
   ierr=BANK_FRONT_BAD_DIMENSIONS;return
  endif
  if(.not.all(ieee_is_finite(bulk_density)).or.any(bulk_density<=0.d0).or. &
     .not.all(ieee_is_finite(cumulative_removed_mass)).or. &
     any(cumulative_removed_mass<0.d0).or. &
     .not.ieee_is_finite(cell_area).or.cell_area<=0.d0.or. &
     .not.ieee_is_finite(solid_area).or. &
     .not.ieee_is_finite(reference_height)) then
   ierr=BANK_FRONT_BAD_DATA;return
  endif
  if(solid_area<0.d0.or.solid_area>cell_area.or.reference_height<0.d0) then
   ierr=BANK_FRONT_BAD_STATE;return
  endif
  removed_volume=0.d0
  do g=1,size(bulk_density)
   removed_volume=removed_volume+cumulative_removed_mass(g)/bulk_density(g)
  enddo
  select case(status)
  case(BANK_FRONT_INACTIVE)
   if(seed_face/=0.or.reference_height>0.d0.or. &
      abs(solid_area-cell_area)>scaled_roundoff(cell_area).or. &
      any(cumulative_removed_mass>0.d0)) then
    ierr=BANK_FRONT_BAD_STATE
   endif
  case(BANK_FRONT_ACTIVE)
   if(seed_face<=0.or.reference_height<=0.d0.or.solid_area<=0.d0) then
    ierr=BANK_FRONT_BAD_STATE;return
   endif
   initial_volume=cell_area*reference_height
   remaining_volume=solid_area*reference_height
   volume_residual=initial_volume-removed_volume-remaining_volume
   tolerance=scaled_roundoff(max(initial_volume,max(removed_volume,remaining_volume)))
   if(abs(volume_residual)>tolerance) ierr=BANK_FRONT_BAD_STATE
  case(BANK_FRONT_PROMOTED)
   if(seed_face<=0.or.reference_height<=0.d0.or. &
      solid_area>scaled_roundoff(cell_area)) then
    ierr=BANK_FRONT_BAD_STATE;return
   endif
   initial_volume=cell_area*reference_height
   volume_residual=initial_volume-removed_volume
   tolerance=scaled_roundoff(max(initial_volume,removed_volume))
   if(abs(volume_residual)>tolerance) ierr=BANK_FRONT_BAD_STATE
  case default
   ierr=BANK_FRONT_BAD_STATE
  end select
 end subroutine braids_bank_front_validate


 pure subroutine braids_bank_front_seed_plane(nface,face_midpoint, &
      inward_normal,face_length,origin,normal,ierr)
  ! A length-weighted seed plane gives a unique coordinate-covariant front
  ! for collinear fragments and ordinary convex corners.  Opposed faces with
  ! a vanishing resultant are rejected rather than resolved by an arbitrary
  ! face-order choice.
  integer,intent(in) :: nface
  real(8),intent(in) :: face_midpoint(2,nface),inward_normal(2,nface),face_length(nface)
  real(8),intent(out) :: origin(2),normal(2)
  integer,intent(out) :: ierr
  real(8) :: total_length,magnitude,resultant_tolerance,unit_normal(2)
  integer :: f

  origin=0.d0;normal=0.d0;ierr=BANK_FRONT_SUCCESS
  if(nface<=0) then
   ierr=BANK_FRONT_BAD_DIMENSIONS;return
  endif
  if(.not.all(ieee_is_finite(face_midpoint)).or. &
     .not.all(ieee_is_finite(inward_normal)).or. &
     .not.all(ieee_is_finite(face_length)).or.any(face_length<=0.d0)) then
   ierr=BANK_FRONT_BAD_DATA;return
  endif
  total_length=sum(face_length)
  if(.not.ieee_is_finite(total_length).or.total_length<=0.d0) then
   ierr=BANK_FRONT_BAD_DATA;return
  endif
  do f=1,nface
   magnitude=sqrt(sum(inward_normal(:,f)**2))
   if(.not.ieee_is_finite(magnitude).or.magnitude<=0.d0) then
    origin=0.d0;normal=0.d0;ierr=BANK_FRONT_BAD_GEOMETRY;return
   endif
   unit_normal=inward_normal(:,f)/magnitude
   origin=origin+face_length(f)*face_midpoint(:,f)
   normal=normal+face_length(f)*unit_normal
  enddo
  origin=origin/total_length
  magnitude=sqrt(sum(normal**2))
  resultant_tolerance=scaled_roundoff(total_length)
  if(.not.ieee_is_finite(magnitude).or.magnitude<=resultant_tolerance) then
   origin=0.d0;normal=0.d0;ierr=BANK_FRONT_BAD_GEOMETRY;return
  endif
  normal=normal/magnitude
 end subroutine braids_bank_front_seed_plane

end module braids_bank_front_module
