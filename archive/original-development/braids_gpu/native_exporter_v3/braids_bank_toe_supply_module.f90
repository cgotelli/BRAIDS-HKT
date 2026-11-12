! SPDX-License-Identifier: AGPL-3.0-or-later
!
! Conservative noncohesive bank-toe supply helpers.
!
! These routines deliberately know nothing about wet/dry state, curvature,
! reach labels, or a constitutive transport-capacity magnitude.  Their source
! is the resolved Exner divergence from the accepted production flux.  Thus a
! spatially uniform through-transport has no bank demand, while a toe cell
! with more outgoing than incoming bed mass has a positive demand.
module braids_bank_toe_supply_module
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 use braids_bank_toe_geometry_module, only: BANK_TOE_NOT_BANK, BANK_TOE_MAPPED
 implicit none
 private

 integer,parameter,public :: BANK_TOE_SUPPLY_SUCCESS=0
 integer,parameter,public :: BANK_TOE_SUPPLY_BAD_DIMENSIONS=-1
 integer,parameter,public :: BANK_TOE_SUPPLY_BAD_DATA=-2
 integer,parameter,public :: BANK_TOE_SUPPLY_BAD_TOPOLOGY=-3

 public :: braids_exner_demand_from_inventory
 public :: braids_grouped_exner_demand_from_inventory
 public :: braids_exner_demand_from_face_integral
 public :: braids_allocate_physical_toe_requests
 public :: braids_route_extracted_toe_mass

contains

 subroutine braids_exner_demand_from_inventory(nc,nfrac,cell_area, &
      bulk_density,before_inventory,after_inventory,is_toe_cell, &
      demand_mass,demand_volume,ierr)
  ! before_inventory/after_inventory are total bed mass per plan area for
  ! each class.  demand_mass is the accepted net erosive Exner loss [kg]
  ! over the caller's complete step, not a capacity proxy.
  integer,intent(in) :: nc,nfrac
  real(8),intent(in) :: cell_area(nc),bulk_density(nfrac)
  real(8),intent(in) :: before_inventory(nfrac,nc),after_inventory(nfrac,nc)
  logical,intent(in) :: is_toe_cell(nc)
  real(8),intent(out) :: demand_mass(nfrac,nc),demand_volume(nc)
  integer,intent(out) :: ierr
  integer :: c,g

  demand_mass=0.d0;demand_volume=0.d0;ierr=BANK_TOE_SUPPLY_SUCCESS
  if(nc<=0.or.nfrac<=0) then
   ierr=BANK_TOE_SUPPLY_BAD_DIMENSIONS;return
  endif
  if(.not.all(ieee_is_finite(cell_area)).or.any(cell_area<=0.d0).or. &
     .not.all(ieee_is_finite(bulk_density)).or.any(bulk_density<=0.d0).or. &
     .not.all(ieee_is_finite(before_inventory)).or. &
     .not.all(ieee_is_finite(after_inventory)).or. &
     any(before_inventory<0.d0).or.any(after_inventory<0.d0)) then
   ierr=BANK_TOE_SUPPLY_BAD_DATA;return
  endif
  do c=1,nc
   if(.not.is_toe_cell(c)) cycle
   do g=1,nfrac
    demand_mass(g,c)=max((before_inventory(g,c)-after_inventory(g,c))*cell_area(c),0.d0)
    demand_volume(c)=demand_volume(c)+demand_mass(g,c)/bulk_density(g)
   enddo
  enddo
 end subroutine braids_exner_demand_from_inventory


 pure subroutine braids_grouped_exner_demand_from_inventory(nc,nfrac, &
      cell_area,bulk_density,before_inventory,after_inventory, &
      erodible_bank,mapping_status,mapped_toe_donor,demand_mass, &
      demand_volume,ierr)
  ! Form one demand for each physical toe donor from all bank-material cells
  ! mapped to it.  The accepted inventory change remains signed while the
  ! group is accumulated: erosion contributes positive loss and deposition
  ! contributes negative loss.  max(net_loss,0) is applied only after the
  ! complete group has been summed, so equal cell-alternating erosion and
  ! deposition cannot create fictitious bank supply.
  !
  ! before_inventory/after_inventory are total bed mass per plan area for
  ! each sediment fraction.  Outputs are indexed by the physical bank donor;
  ! all non-donor entries remain zero.  Fractions are aggregated separately,
  ! preserving the composition of the resolved Exner change.
  integer,intent(in) :: nc,nfrac
  real(8),intent(in) :: cell_area(nc),bulk_density(nfrac)
  real(8),intent(in) :: before_inventory(nfrac,nc),after_inventory(nfrac,nc)
  logical,intent(in) :: erodible_bank(nc)
  integer,intent(in) :: mapping_status(nc),mapped_toe_donor(nc)
  real(8),intent(out) :: demand_mass(nfrac,nc),demand_volume(nc)
  integer,intent(out) :: ierr
  integer :: c,donor,g

  demand_mass=0.d0;demand_volume=0.d0
  ierr=BANK_TOE_SUPPLY_SUCCESS
  if(nc<=0.or.nfrac<=0) then
   ierr=BANK_TOE_SUPPLY_BAD_DIMENSIONS;return
  endif
  if(.not.all(ieee_is_finite(cell_area)).or.any(cell_area<=0.d0).or. &
     .not.all(ieee_is_finite(bulk_density)).or.any(bulk_density<=0.d0).or. &
     .not.all(ieee_is_finite(before_inventory)).or. &
     .not.all(ieee_is_finite(after_inventory)).or. &
     any(before_inventory<0.d0).or.any(after_inventory<0.d0)) then
   ierr=BANK_TOE_SUPPLY_BAD_DATA;return
  endif

  ! Validate the complete mapping before accumulating anything.  In
  ! particular, every referenced physical donor must be bank material, be a
  ! valid mapped cell itself, and map to itself.  This prevents an upslope
  ! cell, ambiguous mapping, or duplicated indirection from becoming a
  ! second accounting location for the same material.
  do c=1,nc
   if(erodible_bank(c)) then
    if(mapping_status(c)/=BANK_TOE_MAPPED) then
     ierr=BANK_TOE_SUPPLY_BAD_TOPOLOGY;return
    endif
    donor=mapped_toe_donor(c)
    if(donor<1.or.donor>nc) then
     ierr=BANK_TOE_SUPPLY_BAD_TOPOLOGY;return
    endif
    if(.not.erodible_bank(donor).or. &
       mapping_status(donor)/=BANK_TOE_MAPPED.or. &
       mapped_toe_donor(donor)/=donor) then
     ierr=BANK_TOE_SUPPLY_BAD_TOPOLOGY;return
    endif
   else
    if(mapping_status(c)/=BANK_TOE_NOT_BANK.or.mapped_toe_donor(c)/=0) then
     ierr=BANK_TOE_SUPPLY_BAD_TOPOLOGY;return
    endif
   endif
  enddo

  ! The ascending cell loop is intentionally serial and fixed-order.  It is
  ! pure and independent of mesh coordinates, hence deterministic for a
  ! given topology and invariant under rigid translation/rotation.
  do c=1,nc
   if(.not.erodible_bank(c)) cycle
   donor=mapped_toe_donor(c)
   do g=1,nfrac
    demand_mass(g,donor)=demand_mass(g,donor)+ &
         (before_inventory(g,c)-after_inventory(g,c))*cell_area(c)
   enddo
  enddo
  do donor=1,nc
   if(.not.erodible_bank(donor).or.mapped_toe_donor(donor)/=donor) cycle
   do g=1,nfrac
    demand_mass(g,donor)=max(demand_mass(g,donor),0.d0)
    demand_volume(donor)=demand_volume(donor)+ &
         demand_mass(g,donor)/bulk_density(g)
   enddo
  enddo
 end subroutine braids_grouped_exner_demand_from_inventory


 subroutine braids_exner_demand_from_face_integral(nc,nf,nfrac,owner, &
      neighbour,bed_mass_integral,is_toe_cell,bulk_density,demand_mass, &
      demand_volume,ierr)
  ! bed_mass_integral(:,1,f) is the signed owner-side accepted mass flux and
  ! bed_mass_integral(:,2,f) the signed neighbour-side flux, both positive in
  ! the stored owner-to-neighbour face direction and already integrated over
  ! face length and the complete accepted step.  Separate side values retain
  ! the exact AMR/reflux contract.
  integer,intent(in) :: nc,nf,nfrac
  integer,intent(in) :: owner(nf),neighbour(nf)
  real(8),intent(in) :: bed_mass_integral(nfrac,2,nf),bulk_density(nfrac)
  logical,intent(in) :: is_toe_cell(nc)
  real(8),intent(out) :: demand_mass(nfrac,nc),demand_volume(nc)
  integer,intent(out) :: ierr
  integer :: c,f,g,n

  demand_mass=0.d0;demand_volume=0.d0;ierr=BANK_TOE_SUPPLY_SUCCESS
  if(nc<=0.or.nf<0.or.nfrac<=0) then
   ierr=BANK_TOE_SUPPLY_BAD_DIMENSIONS;return
  endif
  if(.not.all(ieee_is_finite(bulk_density)).or.any(bulk_density<=0.d0).or. &
     .not.all(ieee_is_finite(bed_mass_integral))) then
   ierr=BANK_TOE_SUPPLY_BAD_DATA;return
  endif
  do f=1,nf
   c=owner(f);n=neighbour(f)
   if(c<1.or.c>nc.or.n<0.or.n>nc.or.n==c) then
    ierr=BANK_TOE_SUPPLY_BAD_TOPOLOGY;demand_mass=0.d0;demand_volume=0.d0;return
   endif
   if(is_toe_cell(c)) demand_mass(:,c)=demand_mass(:,c)+bed_mass_integral(:,1,f)
   if(n>0.and.is_toe_cell(n)) demand_mass(:,n)=demand_mass(:,n)-bed_mass_integral(:,2,f)
  enddo
  do c=1,nc
   if(.not.is_toe_cell(c)) then
    demand_mass(:,c)=0.d0;cycle
   endif
   do g=1,nfrac
    demand_mass(g,c)=max(demand_mass(g,c),0.d0)
    demand_volume(c)=demand_volume(c)+demand_mass(g,c)/bulk_density(g)
   enddo
  enddo
 end subroutine braids_exner_demand_from_face_integral


 subroutine braids_allocate_physical_toe_requests(nc,nf,capacity_fraction, &
      toe_demand_volume,is_toe_face,candidate_bank_cell,candidate_toe_cell, &
      face_length,face_volume_request,donor_volume_request,ierr)
  ! Allocate each unique toe cell's volume once over its physical material
  ! interfaces.  The final face receives the deterministic arithmetic
  ! residual, so duplicated upslope bank-to-toe mappings cannot multiply the
  ! demand and the requested volumes close in fixed face order.
  integer,intent(in) :: nc,nf
  real(8),intent(in) :: capacity_fraction,toe_demand_volume(nc),face_length(nf)
  logical,intent(in) :: is_toe_face(nf)
  integer,intent(in) :: candidate_bank_cell(nf),candidate_toe_cell(nf)
  real(8),intent(out) :: face_volume_request(nf),donor_volume_request(nc)
  integer,intent(out) :: ierr
  real(8) :: total_length(nc),remaining,request
  integer :: last_face(nc),c,f,bank,toe

  face_volume_request=0.d0;donor_volume_request=0.d0
  total_length=0.d0;last_face=0;ierr=BANK_TOE_SUPPLY_SUCCESS
  if(nc<=0.or.nf<0) then
   ierr=BANK_TOE_SUPPLY_BAD_DIMENSIONS;return
  endif
  if(.not.ieee_is_finite(capacity_fraction).or.capacity_fraction<0.d0.or. &
     capacity_fraction>1.d0.or..not.all(ieee_is_finite(toe_demand_volume)).or. &
     any(toe_demand_volume<0.d0).or..not.all(ieee_is_finite(face_length)).or. &
     any(face_length<=0.d0)) then
   ierr=BANK_TOE_SUPPLY_BAD_DATA;return
  endif
  do f=1,nf
   if(.not.is_toe_face(f)) cycle
   bank=candidate_bank_cell(f);toe=candidate_toe_cell(f)
   if(bank<1.or.bank>nc.or.toe<1.or.toe>nc.or.bank==toe) then
    ierr=BANK_TOE_SUPPLY_BAD_TOPOLOGY;return
   endif
   total_length(toe)=total_length(toe)+face_length(f);last_face(toe)=f
  enddo
  do c=1,nc
   if(toe_demand_volume(c)>0.d0.and.total_length(c)<=0.d0) then
    ierr=BANK_TOE_SUPPLY_BAD_TOPOLOGY;return
   endif
   remaining=capacity_fraction*toe_demand_volume(c)
   if(remaining<=0.d0) cycle
   do f=1,nf
    if(.not.is_toe_face(f).or.candidate_toe_cell(f)/=c) cycle
    if(f==last_face(c)) then
     request=remaining
    else
     request=capacity_fraction*toe_demand_volume(c)*face_length(f)/total_length(c)
     request=min(request,remaining)
    endif
    face_volume_request(f)=request
    remaining=max(0.d0,remaining-request)
    donor_volume_request(candidate_bank_cell(f))= &
     donor_volume_request(candidate_bank_cell(f))+request
   enddo
  enddo
 end subroutine braids_allocate_physical_toe_requests


 subroutine braids_route_extracted_toe_mass(nc,nf,nfrac,owner,neighbour, &
      is_toe_face,candidate_bank_cell,candidate_toe_cell,face_volume_request, &
      donor_volume_request,removed_mass,face_mass_transfer,ierr)
  ! Convert inventory-limited mass removed from each physical donor into
  ! signed face transfers. Positive is owner-to-neighbour, matching the native
  ! finite-volume face convention. The last face for each donor receives the
  ! exact residual of every fraction.
  integer,intent(in) :: nc,nf,nfrac
  integer,intent(in) :: owner(nf),neighbour(nf)
  logical,intent(in) :: is_toe_face(nf)
  integer,intent(in) :: candidate_bank_cell(nf),candidate_toe_cell(nf)
  real(8),intent(in) :: face_volume_request(nf),donor_volume_request(nc)
  real(8),intent(in) :: removed_mass(nfrac,nc)
  real(8),intent(out) :: face_mass_transfer(nfrac,nf)
  integer,intent(out) :: ierr
  integer :: last_face(nc),bank,toe,c,f,g
  real(8) :: remaining(nfrac),amount

  face_mass_transfer=0.d0;last_face=0;ierr=BANK_TOE_SUPPLY_SUCCESS
  if(nc<=0.or.nf<0.or.nfrac<=0) then
   ierr=BANK_TOE_SUPPLY_BAD_DIMENSIONS;return
  endif
  if(.not.all(ieee_is_finite(face_volume_request)).or.any(face_volume_request<0.d0).or. &
     .not.all(ieee_is_finite(donor_volume_request)).or.any(donor_volume_request<0.d0).or. &
     .not.all(ieee_is_finite(removed_mass)).or.any(removed_mass<0.d0)) then
   ierr=BANK_TOE_SUPPLY_BAD_DATA;return
  endif
  do f=1,nf
   if(.not.is_toe_face(f).or.face_volume_request(f)<=0.d0) cycle
   bank=candidate_bank_cell(f);toe=candidate_toe_cell(f)
   if(bank<1.or.bank>nc.or.toe<1.or.toe>nc.or.bank==toe.or. &
      .not.((owner(f)==bank.and.neighbour(f)==toe).or. &
            (owner(f)==toe.and.neighbour(f)==bank))) then
    ierr=BANK_TOE_SUPPLY_BAD_TOPOLOGY;face_mass_transfer=0.d0;return
   endif
   last_face(bank)=f
  enddo
  do c=1,nc
   if(donor_volume_request(c)<=0.d0) then
    if(any(removed_mass(:,c)>0.d0)) then
     ierr=BANK_TOE_SUPPLY_BAD_DATA;face_mass_transfer=0.d0;return
    endif
    cycle
   endif
   if(last_face(c)==0) then
    ierr=BANK_TOE_SUPPLY_BAD_TOPOLOGY;face_mass_transfer=0.d0;return
   endif
   remaining=removed_mass(:,c)
   do f=1,nf
    if(.not.is_toe_face(f).or.candidate_bank_cell(f)/=c.or. &
       face_volume_request(f)<=0.d0) cycle
    do g=1,nfrac
     if(f==last_face(c)) then
      amount=remaining(g)
     else
      amount=removed_mass(g,c)*face_volume_request(f)/donor_volume_request(c)
      amount=min(amount,remaining(g))
     endif
     remaining(g)=max(0.d0,remaining(g)-amount)
     if(owner(f)==c) then
      face_mass_transfer(g,f)=amount
     else
      face_mass_transfer(g,f)=-amount
     endif
    enddo
   enddo
  enddo
 end subroutine braids_route_extracted_toe_mass

end module braids_bank_toe_supply_module
