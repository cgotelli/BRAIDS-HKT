! SPDX-License-Identifier: AGPL-3.0-or-later
!
! Reconstruct cell-centred bed-load vectors from an accepted conservative
! face ledger.  This is a diagnostic/constitutive reconstruction only: it
! does not alter the accepted finite-volume flux or any conserved state.
module braids_accepted_flux_reconstruction_module
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 implicit none
 private

 integer,parameter,public :: BRAIDS_ACCEPTED_FLUX_SUCCESS=0
 integer,parameter,public :: BRAIDS_ACCEPTED_FLUX_BAD_DIMENSIONS=-1
 integer,parameter,public :: BRAIDS_ACCEPTED_FLUX_BAD_DATA=-2
 integer,parameter,public :: BRAIDS_ACCEPTED_FLUX_BAD_TOPOLOGY=-3
 integer,parameter,public :: BRAIDS_ACCEPTED_FLUX_BAD_GEOMETRY=-4
 integer,parameter,public :: BRAIDS_ACCEPTED_FLUX_NOT_SHARED=-5

 public :: braids_reconstruct_accepted_bedload
 public :: braids_reconstruct_accepted_bedload_density

contains

 pure subroutine braids_reconstruct_accepted_bedload(nc,nf,nfrac,owner, &
      neighbour,face_normal,face_length,accepted_face_rate,cell_bedload, &
      rank_deficient,ierr)
  ! Length-weighted least-squares reconstruction of a vector q_c satisfying
  !
  !        q_c . n_f = accepted_face_rate(g,side,f)/face_length(f).
  !
  ! ``accepted_face_rate`` is the face-integrated mass rate [kg s-1].  The
  ! first side is the owner view and the second is the neighbour view.  Both
  ! use the stored owner-to-neighbour unit normal and therefore must be equal,
  ! rather than opposite.  Boundary faces retain the same two-view contract.
  ! The native accepted SSPRK face masses can be passed after division by the
  ! accepted step duration.  Positive rate removes mass from the owner and
  ! adds the same mass to an interior neighbour.
  !
  ! For every incident face, including an interior face seen from its
  ! neighbour, the normal equation receives the same scalar projection:
  !
  !        M_c += L_f n_f n_f^T,       b_c += n_f R_f.
  !
  ! This intentionally matches the validation oracle ``reconstruct_accepted_flux``.
  ! A cell whose two-dimensional system fails the oracle's rank criterion is
  ! flagged and its complete per-fraction vector is returned as zero.
  integer,intent(in) :: nc,nf,nfrac
  integer,intent(in) :: owner(:),neighbour(:)
  real(8),intent(in) :: face_normal(:,:),face_length(:)
  real(8),intent(in) :: accepted_face_rate(:,:,:)
  real(8),intent(out) :: cell_bedload(:,:,:)
  logical,intent(out) :: rank_deficient(:)
  integer,intent(out) :: ierr

  real(8),parameter :: unit_tolerance=1.d-12
  real(8),parameter :: shared_tolerance=1.d-12
  real(8),parameter :: rank_tolerance=1.d-14
  real(8) :: mxx(max(0,nc)),mxy(max(0,nc)),myy(max(0,nc))
  real(8) :: bx(max(0,nfrac),max(0,nc)),by(max(0,nfrac),max(0,nc))
  real(8) :: determinant,normal_size_squared,scale,shared_rate
  integer :: c,f,g,n

  cell_bedload=0.d0
  rank_deficient=.true.
  ierr=BRAIDS_ACCEPTED_FLUX_SUCCESS

  if(nc<=0.or.nf<0.or.nfrac<=0.or. &
     size(owner)/=nf.or.size(neighbour)/=nf.or. &
     size(face_normal,1)/=2.or.size(face_normal,2)/=nf.or. &
     size(face_length)/=nf.or. &
     size(accepted_face_rate,1)/=nfrac.or. &
     size(accepted_face_rate,2)/=2.or. &
     size(accepted_face_rate,3)/=nf.or. &
     size(cell_bedload,1)/=2.or. &
     size(cell_bedload,2)/=nfrac.or. &
     size(cell_bedload,3)/=nc.or.size(rank_deficient)/=nc) then
   ierr=BRAIDS_ACCEPTED_FLUX_BAD_DIMENSIONS
   return
  endif

  if(.not.all(ieee_is_finite(face_normal)).or. &
     .not.all(ieee_is_finite(face_length)).or. &
     .not.all(ieee_is_finite(accepted_face_rate))) then
   ierr=BRAIDS_ACCEPTED_FLUX_BAD_DATA
   return
  endif

  ! Validate every input before assembling any partial reconstruction.
  do f=1,nf
   if(owner(f)<1.or.owner(f)>nc.or.neighbour(f)<0.or. &
      neighbour(f)>nc.or.neighbour(f)==owner(f)) then
    ierr=BRAIDS_ACCEPTED_FLUX_BAD_TOPOLOGY
    return
   endif
   normal_size_squared=sum(face_normal(:,f)**2)
   if(.not.ieee_is_finite(normal_size_squared).or.face_length(f)<=0.d0.or. &
      abs(normal_size_squared-1.d0)>unit_tolerance) then
    ierr=BRAIDS_ACCEPTED_FLUX_BAD_GEOMETRY
    return
   endif
   do g=1,nfrac
    scale=max(1.d0,abs(accepted_face_rate(g,1,f)), &
         abs(accepted_face_rate(g,2,f)))
    if(abs(accepted_face_rate(g,1,f)-accepted_face_rate(g,2,f))> &
       shared_tolerance*scale) then
     ierr=BRAIDS_ACCEPTED_FLUX_NOT_SHARED
     return
    endif
   enddo
  enddo

  mxx=0.d0;mxy=0.d0;myy=0.d0;bx=0.d0;by=0.d0
  do f=1,nf
   c=owner(f)
   mxx(c)=mxx(c)+face_length(f)*face_normal(1,f)**2
   mxy(c)=mxy(c)+face_length(f)*face_normal(1,f)*face_normal(2,f)
   myy(c)=myy(c)+face_length(f)*face_normal(2,f)**2
   do g=1,nfrac
    shared_rate=.5d0*accepted_face_rate(g,1,f)+ &
         .5d0*accepted_face_rate(g,2,f)
    bx(g,c)=bx(g,c)+face_normal(1,f)*shared_rate
    by(g,c)=by(g,c)+face_normal(2,f)*shared_rate
   enddo

   n=neighbour(f)
   if(n<=0) cycle
   mxx(n)=mxx(n)+face_length(f)*face_normal(1,f)**2
   mxy(n)=mxy(n)+face_length(f)*face_normal(1,f)*face_normal(2,f)
   myy(n)=myy(n)+face_length(f)*face_normal(2,f)**2
   do g=1,nfrac
    shared_rate=.5d0*accepted_face_rate(g,1,f)+ &
         .5d0*accepted_face_rate(g,2,f)
    bx(g,n)=bx(g,n)+face_normal(1,f)*shared_rate
    by(g,n)=by(g,n)+face_normal(2,f)*shared_rate
   enddo
  enddo

  if(.not.all(ieee_is_finite(mxx)).or. &
     .not.all(ieee_is_finite(mxy)).or. &
     .not.all(ieee_is_finite(myy)).or. &
     .not.all(ieee_is_finite(bx)).or. &
     .not.all(ieee_is_finite(by))) then
   ierr=BRAIDS_ACCEPTED_FLUX_BAD_DATA
   return
  endif

  do c=1,nc
   determinant=mxx(c)*myy(c)-mxy(c)*mxy(c)
   scale=max(mxx(c)*myy(c),1.d0)
   if(.not.ieee_is_finite(determinant).or..not.ieee_is_finite(scale)) then
    cell_bedload=0.d0
    rank_deficient=.true.
    ierr=BRAIDS_ACCEPTED_FLUX_BAD_DATA
    return
   endif
   if(determinant<=rank_tolerance*scale) cycle
   rank_deficient(c)=.false.
   do g=1,nfrac
    cell_bedload(1,g,c)= &
         (myy(c)*bx(g,c)-mxy(c)*by(g,c))/determinant
    cell_bedload(2,g,c)= &
         (mxx(c)*by(g,c)-mxy(c)*bx(g,c))/determinant
   enddo
  enddo

  if(.not.all(ieee_is_finite(cell_bedload))) then
   cell_bedload=0.d0
   rank_deficient=.true.
   ierr=BRAIDS_ACCEPTED_FLUX_BAD_DATA
  endif
 end subroutine braids_reconstruct_accepted_bedload


 pure subroutine braids_reconstruct_accepted_bedload_density(nc,nf,nfrac, &
      owner,neighbour,face_normal,face_length,accepted_flux_density, &
      cell_bedload,rank_deficient,ierr)
  ! Convenience form for direct SSPRK-weighted normal flux densities
  ! [kg m-1 s-1].  Multiplying by the physical face length gives exactly the
  ! integrated rate consumed by ``braids_reconstruct_accepted_bedload``.
  integer,intent(in) :: nc,nf,nfrac
  integer,intent(in) :: owner(:),neighbour(:)
  real(8),intent(in) :: face_normal(:,:),face_length(:)
  real(8),intent(in) :: accepted_flux_density(:,:,:)
  real(8),intent(out) :: cell_bedload(:,:,:)
  logical,intent(out) :: rank_deficient(:)
  integer,intent(out) :: ierr

  real(8) :: accepted_face_rate(size(accepted_flux_density,1), &
       size(accepted_flux_density,2),size(accepted_flux_density,3))
  integer :: f

  accepted_face_rate=accepted_flux_density
  if(size(face_length)==nf.and.size(accepted_flux_density,3)==nf) then
   do f=1,nf
    accepted_face_rate(:,:,f)=accepted_flux_density(:,:,f)*face_length(f)
   enddo
  endif
  call braids_reconstruct_accepted_bedload(nc,nf,nfrac,owner,neighbour, &
       face_normal,face_length,accepted_face_rate,cell_bedload, &
       rank_deficient,ierr)
 end subroutine braids_reconstruct_accepted_bedload_density

end module braids_accepted_flux_reconstruction_module
