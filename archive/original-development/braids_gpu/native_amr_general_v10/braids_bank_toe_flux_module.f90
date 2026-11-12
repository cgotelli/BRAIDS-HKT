! Diagnostic accounting for accepted bed-load transport across physical bank-toe
! interfaces.  This module does not alter transport or bank-failure physics.
module braids_bank_toe_flux_module
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 implicit none
 private
 public :: braids_bank_toe_flux_from_rates,braids_bank_toe_flux_from_integrals
 public :: braids_bank_toe_add_face_rates,braids_bank_toe_add_face_integrals
 integer,parameter,public :: BANK_TOE_FLUX_SUCCESS=0
 integer,parameter,public :: BANK_TOE_FLUX_INVALID_DIMENSIONS=1
 integer,parameter,public :: BANK_TOE_FLUX_INVALID_FACE=2
 integer,parameter,public :: BANK_TOE_FLUX_NONFINITE=3
 integer,parameter,public :: BANK_TOE_FLUX_NONSHARED=4
contains

 pure subroutine braids_bank_toe_add_face_rates(nface,nfraction,ncomponent,bed_first,owner,neighbour, &
  is_toe_face,candidate_bank,candidate_toe,face_length,dt,weight,left_rate,right_rate, &
  signed_face_mass,error)
  integer,intent(in) :: nface,nfraction,ncomponent,bed_first,owner(nface),neighbour(nface)
  logical,intent(in) :: is_toe_face(nface)
  integer,intent(in) :: candidate_bank(nface),candidate_toe(nface)
  real(8),intent(in) :: face_length(nface),dt,weight,left_rate(ncomponent,nface),right_rate(ncomponent,nface)
  real(8),intent(inout) :: signed_face_mass(nfraction,nface)
  integer,intent(out) :: error
  real(8) :: left_mass,right_mass,tolerance
  integer :: face,fraction,component,orientation
  error=BANK_TOE_FLUX_SUCCESS
  if(nface<1.or.nfraction<1.or.ncomponent<1.or.bed_first<1.or. &
     bed_first+nfraction-1>ncomponent.or..not.ieee_is_finite(dt).or.dt<0.d0.or. &
     .not.ieee_is_finite(weight).or.weight<0.d0) then
   error=BANK_TOE_FLUX_INVALID_DIMENSIONS;return
  endif
  do face=1,nface
   if(.not.is_toe_face(face)) cycle
   call toe_face_orientation(owner(face),neighbour(face),candidate_bank(face),candidate_toe(face),orientation,error)
   if(error/=BANK_TOE_FLUX_SUCCESS) return
   if(.not.ieee_is_finite(face_length(face)).or.face_length(face)<=0.d0) then
    error=BANK_TOE_FLUX_INVALID_DIMENSIONS;return
   endif
   do fraction=1,nfraction
    component=bed_first+fraction-1
    if(.not.ieee_is_finite(left_rate(component,face)).or. &
       .not.ieee_is_finite(right_rate(component,face))) then
     error=BANK_TOE_FLUX_NONFINITE;return
    endif
    left_mass=dt*face_length(face)*left_rate(component,face)
    right_mass=dt*face_length(face)*right_rate(component,face)
    tolerance=256.d0*epsilon(1.d0)*max(1.d0,abs(left_mass),abs(right_mass))
    if(abs(left_mass-right_mass)>tolerance) then
     error=BANK_TOE_FLUX_NONSHARED;return
    endif
    signed_face_mass(fraction,face)=signed_face_mass(fraction,face)+ &
     weight*orientation*.5d0*(left_mass+right_mass)
   enddo
  enddo
 end subroutine

 pure subroutine braids_bank_toe_add_face_integrals(nface,nfraction,ncomponent,bed_first,owner,neighbour, &
  is_toe_face,candidate_bank,candidate_toe,weight,integral,signed_face_mass,error)
  integer,intent(in) :: nface,nfraction,ncomponent,bed_first,owner(nface),neighbour(nface)
  logical,intent(in) :: is_toe_face(nface)
  integer,intent(in) :: candidate_bank(nface),candidate_toe(nface)
  real(8),intent(in) :: weight,integral(ncomponent,2,nface)
  real(8),intent(inout) :: signed_face_mass(nfraction,nface)
  integer,intent(out) :: error
  real(8) :: left_mass,right_mass,tolerance
  integer :: face,fraction,component,orientation
  error=BANK_TOE_FLUX_SUCCESS
  if(nface<1.or.nfraction<1.or.ncomponent<1.or.bed_first<1.or. &
     bed_first+nfraction-1>ncomponent.or..not.ieee_is_finite(weight).or.weight<0.d0) then
   error=BANK_TOE_FLUX_INVALID_DIMENSIONS;return
  endif
  do face=1,nface
   if(.not.is_toe_face(face)) cycle
   call toe_face_orientation(owner(face),neighbour(face),candidate_bank(face),candidate_toe(face),orientation,error)
   if(error/=BANK_TOE_FLUX_SUCCESS) return
   do fraction=1,nfraction
    component=bed_first+fraction-1
    left_mass=integral(component,1,face);right_mass=integral(component,2,face)
    if(.not.ieee_is_finite(left_mass).or..not.ieee_is_finite(right_mass)) then
     error=BANK_TOE_FLUX_NONFINITE;return
    endif
    tolerance=256.d0*epsilon(1.d0)*max(1.d0,abs(left_mass),abs(right_mass))
    if(abs(left_mass-right_mass)>tolerance) then
     error=BANK_TOE_FLUX_NONSHARED;return
    endif
    signed_face_mass(fraction,face)=signed_face_mass(fraction,face)+ &
     weight*orientation*.5d0*(left_mass+right_mass)
   enddo
  enddo
 end subroutine

 pure subroutine braids_bank_toe_flux_from_rates(nface,nfraction,ncomponent,bed_first,owner,neighbour,is_toe_face, &
  candidate_bank,candidate_toe,face_length,dt,left_rate,right_rate,signed_mass, &
  bank_to_channel_mass,channel_to_bank_mass,error)
  integer,intent(in) :: nface,nfraction,ncomponent,bed_first,owner(nface),neighbour(nface)
  logical,intent(in) :: is_toe_face(nface)
  integer,intent(in) :: candidate_bank(nface),candidate_toe(nface)
  real(8),intent(in) :: face_length(nface),dt,left_rate(ncomponent,nface),right_rate(ncomponent,nface)
  real(8),intent(out) :: signed_mass(nfraction),bank_to_channel_mass(nfraction),channel_to_bank_mass(nfraction)
  integer,intent(out) :: error
  real(8) :: amount,left_mass,right_mass,tolerance
  integer :: face,fraction,component,orientation
  signed_mass=0.d0;bank_to_channel_mass=0.d0;channel_to_bank_mass=0.d0
  error=BANK_TOE_FLUX_SUCCESS
  if(nface<1.or.nfraction<1.or.ncomponent<1.or.bed_first<1.or. &
     bed_first+nfraction-1>ncomponent.or..not.ieee_is_finite(dt).or.dt<0.d0) then
   error=BANK_TOE_FLUX_INVALID_DIMENSIONS;return
  endif
  do face=1,nface
   if(.not.is_toe_face(face)) cycle
   call toe_face_orientation(owner(face),neighbour(face),candidate_bank(face),candidate_toe(face),orientation,error)
   if(error/=BANK_TOE_FLUX_SUCCESS) return
   if(.not.ieee_is_finite(face_length(face)).or.face_length(face)<=0.d0) then
    error=BANK_TOE_FLUX_INVALID_DIMENSIONS;return
   endif
   do fraction=1,nfraction
    component=bed_first+fraction-1
    if(.not.ieee_is_finite(left_rate(component,face)).or. &
       .not.ieee_is_finite(right_rate(component,face))) then
     error=BANK_TOE_FLUX_NONFINITE;return
    endif
    left_mass=dt*face_length(face)*left_rate(component,face)
    right_mass=dt*face_length(face)*right_rate(component,face)
    tolerance=256.d0*epsilon(1.d0)*max(1.d0,abs(left_mass),abs(right_mass))
    if(abs(left_mass-right_mass)>tolerance) then
     error=BANK_TOE_FLUX_NONSHARED;return
    endif
    amount=orientation*.5d0*(left_mass+right_mass)
    if(amount>=0.d0) then
     bank_to_channel_mass(fraction)=bank_to_channel_mass(fraction)+amount
    else
     channel_to_bank_mass(fraction)=channel_to_bank_mass(fraction)-amount
    endif
   enddo
  enddo
  signed_mass=bank_to_channel_mass-channel_to_bank_mass
 end subroutine

 pure subroutine braids_bank_toe_flux_from_integrals(nface,nfraction,ncomponent,bed_first,owner,neighbour,is_toe_face, &
  candidate_bank,candidate_toe,integral,signed_mass,bank_to_channel_mass, &
  channel_to_bank_mass,error)
  integer,intent(in) :: nface,nfraction,ncomponent,bed_first,owner(nface),neighbour(nface)
  logical,intent(in) :: is_toe_face(nface)
  integer,intent(in) :: candidate_bank(nface),candidate_toe(nface)
  real(8),intent(in) :: integral(ncomponent,2,nface)
  real(8),intent(out) :: signed_mass(nfraction),bank_to_channel_mass(nfraction),channel_to_bank_mass(nfraction)
  integer,intent(out) :: error
  real(8) :: amount,left_mass,right_mass,tolerance
  integer :: face,fraction,component,orientation
  signed_mass=0.d0;bank_to_channel_mass=0.d0;channel_to_bank_mass=0.d0
  error=BANK_TOE_FLUX_SUCCESS
  if(nface<1.or.nfraction<1.or.ncomponent<1.or.bed_first<1.or. &
     bed_first+nfraction-1>ncomponent) then
   error=BANK_TOE_FLUX_INVALID_DIMENSIONS;return
  endif
  do face=1,nface
   if(.not.is_toe_face(face)) cycle
   call toe_face_orientation(owner(face),neighbour(face),candidate_bank(face),candidate_toe(face),orientation,error)
   if(error/=BANK_TOE_FLUX_SUCCESS) return
   do fraction=1,nfraction
    component=bed_first+fraction-1
    left_mass=integral(component,1,face);right_mass=integral(component,2,face)
    if(.not.ieee_is_finite(left_mass).or..not.ieee_is_finite(right_mass)) then
     error=BANK_TOE_FLUX_NONFINITE;return
    endif
    tolerance=256.d0*epsilon(1.d0)*max(1.d0,abs(left_mass),abs(right_mass))
    if(abs(left_mass-right_mass)>tolerance) then
     error=BANK_TOE_FLUX_NONSHARED;return
    endif
    amount=orientation*.5d0*(left_mass+right_mass)
    if(amount>=0.d0) then
     bank_to_channel_mass(fraction)=bank_to_channel_mass(fraction)+amount
    else
     channel_to_bank_mass(fraction)=channel_to_bank_mass(fraction)-amount
    endif
   enddo
  enddo
  signed_mass=bank_to_channel_mass-channel_to_bank_mass
 end subroutine

 pure subroutine toe_face_orientation(owner,neighbour,bank,toe,orientation,error)
  integer,intent(in) :: owner,neighbour,bank,toe
  integer,intent(out) :: orientation
  integer,intent(out) :: error
  orientation=0;error=BANK_TOE_FLUX_SUCCESS
  if(owner<1.or.neighbour<1.or.bank<1.or.toe<1.or.bank==toe) then
   error=BANK_TOE_FLUX_INVALID_FACE;return
  endif
  if(bank==owner.and.toe==neighbour) then
   orientation=1
  else if(bank==neighbour.and.toe==owner) then
   orientation=-1
  else
   error=BANK_TOE_FLUX_INVALID_FACE
  endif
 end subroutine
end module
