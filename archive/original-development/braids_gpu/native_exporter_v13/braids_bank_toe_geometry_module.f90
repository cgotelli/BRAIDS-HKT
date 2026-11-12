! SPDX-License-Identifier: AGPL-3.0-or-later
!
! Topological bank-material to physical-toe mapping.
!
! A physical toe cannot be inferred robustly from elevation and wet/dry state
! alone on a river mesh: a longitudinal bed slope is indistinguishable from a
! bank slope without material information.  The caller must therefore provide
! an erodible-bank *material* mask that includes submerged side-slope cells.
! This module deliberately has no water-depth argument.
!
! The toe is the interior material interface through which an erodible bank
! cell can descend to a non-bank cell.  All such downhill interfaces are
! reported.  Bank cells are mapped to them by a deterministic, monotone-
! downhill, multi-source shortest path on the supplied face adjacency.  The
! metric uses cell centroids and face midpoints, so it is invariant under rigid
! translation and rotation and works on non-orthogonal/cut/AMR faces.
!
! If a material mask contains an unintended end, hole, or equally near toe,
! geometry does not contain enough information to invent the intended bank.
! Such ties are returned as BANK_TOE_AMBIGUOUS.  A caller may pass the optional
! toe_face_allowed material-interface selector; this is a local material input,
! not a reach/inner/outer calibration mask.  Production erosion must not act on
! ambiguous or unreachable mappings.
module braids_bank_toe_geometry_module
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 implicit none
 private

 integer,parameter,public :: BANK_TOE_NOT_BANK=0
 integer,parameter,public :: BANK_TOE_MAPPED=1
 integer,parameter,public :: BANK_TOE_AMBIGUOUS=2
 integer,parameter,public :: BANK_TOE_UNREACHABLE=-1

 integer,parameter,public :: BANK_TOE_SUCCESS=0
 integer,parameter,public :: BANK_TOE_NO_CANDIDATE=1
 integer,parameter,public :: BANK_TOE_BAD_DIMENSIONS=-1
 integer,parameter,public :: BANK_TOE_BAD_GEOMETRY=-2
 integer,parameter,public :: BANK_TOE_BAD_TOPOLOGY=-3
 integer,parameter,public :: BANK_TOE_BAD_TOLERANCE=-4
 integer,parameter,public :: BANK_TOE_ALLOCATION_FAILURE=-5

 public :: braids_map_bank_toes

 type :: bank_toe_heap
  integer :: count=0
  integer,allocatable :: cell(:)
  real(8),allocatable :: key(:)
 end type bank_toe_heap

contains

 subroutine braids_map_bank_toes(nc,nf,cell_xy,cell_area,bed,erodible_bank, &
      owner,neighbour,face_mid,face_normal,face_length,elevation_tolerance, &
      distance_tolerance,toe_cell,toe_donor,toe_face,path_distance, &
      is_toe_cell,is_toe_donor,is_toe_face,candidate_bank_cell, &
      candidate_toe_cell,status,ierr,toe_face_allowed)
  integer,intent(in) :: nc,nf
  real(8),intent(in) :: cell_xy(2,nc),cell_area(nc),bed(nc)
  logical,intent(in) :: erodible_bank(nc)
  integer,intent(in) :: owner(nf),neighbour(nf)
  real(8),intent(in) :: face_mid(2,nf),face_normal(2,nf),face_length(nf)
  real(8),intent(in) :: elevation_tolerance,distance_tolerance
  integer,intent(out) :: toe_cell(nc),toe_donor(nc),toe_face(nc),status(nc)
  real(8),intent(out) :: path_distance(nc)
  logical,intent(out) :: is_toe_cell(nc),is_toe_donor(nc),is_toe_face(nf)
  integer,intent(out) :: candidate_bank_cell(nf),candidate_toe_cell(nf)
  integer,intent(out) :: ierr
  logical,intent(in),optional :: toe_face_allowed(nf)

  integer,allocatable :: degree(:),offset(:),cursor(:),adjacent(:),adjacent_face(:)
  real(8),allocatable :: edge_distance(:)
  logical,allocatable :: ambiguous(:)
  integer :: f,c,n,bank_cell,channel_cell,total_links,allocation_status
  integer :: candidate_count,heap_error
  real(8) :: normal_size,first_leg,second_leg,candidate_distance
  type(bank_toe_heap) :: heap

  toe_cell=0;toe_donor=0;toe_face=0;status=BANK_TOE_NOT_BANK
  path_distance=huge(1.d0)
  is_toe_cell=.false.;is_toe_donor=.false.;is_toe_face=.false.
  candidate_bank_cell=0;candidate_toe_cell=0
  ierr=BANK_TOE_SUCCESS

  if(nc<=0.or.nf<0) then
   ierr=BANK_TOE_BAD_DIMENSIONS;return
  endif
  if(elevation_tolerance<0.d0.or.distance_tolerance<0.d0.or. &
     .not.ieee_is_finite(elevation_tolerance).or. &
     .not.ieee_is_finite(distance_tolerance)) then
   ierr=BANK_TOE_BAD_TOLERANCE;return
  endif
  if(.not.all(ieee_is_finite(cell_xy)).or. &
     .not.all(ieee_is_finite(cell_area)).or. &
     .not.all(ieee_is_finite(bed)).or.any(cell_area<=0.d0)) then
   ierr=BANK_TOE_BAD_GEOMETRY;return
  endif
  if(nf>0) then
   if(.not.all(ieee_is_finite(face_mid)).or. &
      .not.all(ieee_is_finite(face_normal)).or. &
      .not.all(ieee_is_finite(face_length)).or.any(face_length<=0.d0)) then
    ierr=BANK_TOE_BAD_GEOMETRY;return
   endif
   do f=1,nf
    if(owner(f)<1.or.owner(f)>nc.or.neighbour(f)<0.or.neighbour(f)>nc.or. &
       neighbour(f)==owner(f)) then
     ierr=BANK_TOE_BAD_TOPOLOGY;return
    endif
    normal_size=sqrt(sum(face_normal(:,f)**2))
    if(.not.ieee_is_finite(normal_size).or.normal_size<=0.d0) then
     ierr=BANK_TOE_BAD_GEOMETRY;return
    endif
   enddo
  endif

  allocate(degree(nc),offset(nc+1),cursor(nc),edge_distance(nf), &
           ambiguous(nc),stat=allocation_status)
  if(allocation_status/=0) then
   ierr=BANK_TOE_ALLOCATION_FAILURE;return
  endif
  degree=0;edge_distance=0.d0;ambiguous=.false.
  do f=1,nf
   if(neighbour(f)<=0) cycle
   degree(owner(f))=degree(owner(f))+1
   degree(neighbour(f))=degree(neighbour(f))+1
   first_leg=sqrt(sum((face_mid(:,f)-cell_xy(:,owner(f)))**2))
   second_leg=sqrt(sum((face_mid(:,f)-cell_xy(:,neighbour(f)))**2))
   edge_distance(f)=first_leg+second_leg
   if(.not.ieee_is_finite(edge_distance(f)).or.edge_distance(f)<=0.d0) then
    ierr=BANK_TOE_BAD_GEOMETRY;return
   endif
  enddo
  offset(1)=1
  do c=1,nc
   offset(c+1)=offset(c)+degree(c)
  enddo
  total_links=offset(nc+1)-1
  allocate(adjacent(max(1,total_links)),adjacent_face(max(1,total_links)), &
           stat=allocation_status)
  if(allocation_status/=0) then
   ierr=BANK_TOE_ALLOCATION_FAILURE;return
  endif
  cursor=offset(1:nc)
  do f=1,nf
   if(neighbour(f)<=0) cycle
   c=owner(f);n=neighbour(f)
   adjacent(cursor(c))=n;adjacent_face(cursor(c))=f;cursor(c)=cursor(c)+1
   adjacent(cursor(n))=c;adjacent_face(cursor(n))=f;cursor(n)=cursor(n)+1
  enddo

  call heap_initialize(heap,max(16,nc),heap_error)
  if(heap_error/=0) then
   ierr=BANK_TOE_ALLOCATION_FAILURE;return
  endif

  ! An interface is a physical-toe candidate only when it exits bank material
  ! without climbing. Boundary faces have no acceptor cell and are excluded.
  candidate_count=0
  do f=1,nf
   if(neighbour(f)<=0) cycle
   if(present(toe_face_allowed)) then
    if(.not.toe_face_allowed(f)) cycle
   endif
   if(erodible_bank(owner(f)).eqv.erodible_bank(neighbour(f))) cycle
   if(erodible_bank(owner(f))) then
    bank_cell=owner(f);channel_cell=neighbour(f)
   else
    bank_cell=neighbour(f);channel_cell=owner(f)
   endif
   if(bed(channel_cell)>bed(bank_cell)+elevation_tolerance) cycle
   candidate_count=candidate_count+1
   is_toe_face(f)=.true.
   is_toe_donor(bank_cell)=.true.
   is_toe_cell(channel_cell)=.true.
   candidate_bank_cell(f)=bank_cell
   candidate_toe_cell(f)=channel_cell
   candidate_distance=edge_distance(f)
   call seed_candidate(bank_cell,channel_cell,f,candidate_distance, &
        bed,face_length,elevation_tolerance,distance_tolerance,toe_cell, &
        toe_donor,toe_face,path_distance,ambiguous,heap,heap_error)
   if(heap_error/=0) then
    ierr=BANK_TOE_ALLOCATION_FAILURE;return
   endif
  enddo

  if(candidate_count==0) then
   where(erodible_bank) status=BANK_TOE_UNREACHABLE
   ierr=BANK_TOE_NO_CANDIDATE
   return
  endif

  call propagate_mapping(nc,bed,erodible_bank,offset,adjacent,adjacent_face, &
       edge_distance,elevation_tolerance,distance_tolerance,face_length, &
       toe_cell,toe_donor,toe_face,path_distance,ambiguous,heap,heap_error)
  if(heap_error/=0) then
   ierr=BANK_TOE_ALLOCATION_FAILURE;return
  endif

  do c=1,nc
   if(.not.erodible_bank(c)) then
    status(c)=BANK_TOE_NOT_BANK
   else if(toe_cell(c)<=0) then
    status(c)=BANK_TOE_UNREACHABLE
   else if(ambiguous(c)) then
    status(c)=BANK_TOE_AMBIGUOUS
   else
    status(c)=BANK_TOE_MAPPED
   endif
  enddo
 end subroutine braids_map_bank_toes


 subroutine seed_candidate(cell,target,face,distance,bed,face_length, &
      elevation_tolerance,distance_tolerance,toe_cell,toe_donor,toe_face, &
      path_distance,ambiguous,heap,ierr)
  integer,intent(in) :: cell,target,face
  real(8),intent(in) :: distance,bed(:),face_length(:)
  real(8),intent(in) :: elevation_tolerance,distance_tolerance
  integer,intent(inout) :: toe_cell(:),toe_donor(:),toe_face(:)
  real(8),intent(inout) :: path_distance(:)
  logical,intent(inout) :: ambiguous(:)
  type(bank_toe_heap),intent(inout) :: heap
  integer,intent(out) :: ierr
  logical :: changed,same_target

  ierr=0;changed=.false.
  if(distance<path_distance(cell)-distance_tolerance) then
   toe_cell(cell)=target;toe_donor(cell)=cell;toe_face(cell)=face
   path_distance(cell)=distance;ambiguous(cell)=.false.;changed=.true.
  else if(abs(distance-path_distance(cell))<=distance_tolerance) then
   same_target=toe_cell(cell)==target.and.toe_donor(cell)==cell
   if(.not.same_target.and..not.ambiguous(cell)) then
    ambiguous(cell)=.true.;changed=.true.
   endif
   if(prefer_mapping(target,cell,face,toe_cell(cell),toe_donor(cell), &
        toe_face(cell),bed,face_length,elevation_tolerance)) then
    toe_cell(cell)=target;toe_donor(cell)=cell;toe_face(cell)=face
    path_distance(cell)=distance;changed=.true.
   endif
  endif
  if(changed) call heap_push(heap,cell,path_distance(cell),ierr)
 end subroutine seed_candidate


 subroutine propagate_mapping(nc,bed,erodible_bank,offset,adjacent, &
      adjacent_face,edge_distance,elevation_tolerance,distance_tolerance, &
      face_length,toe_cell,toe_donor,toe_face,path_distance,ambiguous,heap,ierr)
  integer,intent(in) :: nc,offset(nc+1),adjacent(:),adjacent_face(:)
  real(8),intent(in) :: bed(nc),edge_distance(:),elevation_tolerance
  real(8),intent(in) :: distance_tolerance,face_length(:)
  logical,intent(in) :: erodible_bank(nc)
  integer,intent(inout) :: toe_cell(nc),toe_donor(nc),toe_face(nc)
  real(8),intent(inout) :: path_distance(nc)
  logical,intent(inout) :: ambiguous(nc)
  type(bank_toe_heap),intent(inout) :: heap
  integer,intent(out) :: ierr
  integer :: cell,other,position,face,heap_error
  real(8) :: key,trial
  logical :: changed,same_mapping

  ierr=0
  do while(heap%count>0)
   call heap_pop(heap,cell,key)
   if(key>path_distance(cell)+distance_tolerance) cycle
   do position=offset(cell),offset(cell+1)-1
    other=adjacent(position);face=adjacent_face(position)
    if(.not.erodible_bank(other)) cycle
    ! Reverse propagation is permitted only uphill or across a flat. Thus the
    ! stored path from `other` to the toe is monotonically non-increasing.
    if(bed(other)+elevation_tolerance<bed(cell)) cycle
    trial=path_distance(cell)+edge_distance(face)
    changed=.false.
    if(trial<path_distance(other)-distance_tolerance) then
     toe_cell(other)=toe_cell(cell);toe_donor(other)=toe_donor(cell)
     toe_face(other)=toe_face(cell);path_distance(other)=trial
     ambiguous(other)=ambiguous(cell);changed=.true.
    else if(abs(trial-path_distance(other))<=distance_tolerance) then
     same_mapping=toe_cell(other)==toe_cell(cell).and. &
                  toe_donor(other)==toe_donor(cell)
     if((ambiguous(cell).or..not.same_mapping).and..not.ambiguous(other)) then
      ambiguous(other)=.true.;changed=.true.
     endif
     if(prefer_mapping(toe_cell(cell),toe_donor(cell),toe_face(cell), &
          toe_cell(other),toe_donor(other),toe_face(other),bed,face_length, &
          elevation_tolerance)) then
      toe_cell(other)=toe_cell(cell);toe_donor(other)=toe_donor(cell)
      toe_face(other)=toe_face(cell);path_distance(other)=trial
      changed=.true.
     endif
    endif
    if(changed) then
     call heap_push(heap,other,path_distance(other),heap_error)
     if(heap_error/=0) then
      ierr=heap_error;return
     endif
    endif
   enddo
  enddo
 end subroutine propagate_mapping


 logical function prefer_mapping(candidate_toe,candidate_donor,candidate_face, &
      current_toe,current_donor,current_face,bed,face_length,elevation_tolerance) &
      result(prefer)
  integer,intent(in) :: candidate_toe,candidate_donor,candidate_face
  integer,intent(in) :: current_toe,current_donor,current_face
  real(8),intent(in) :: bed(:),face_length(:),elevation_tolerance

  prefer=.false.
  if(current_toe<=0) then
   prefer=.true.;return
  endif
  if(bed(candidate_toe)<bed(current_toe)-elevation_tolerance) then
   prefer=.true.;return
  else if(bed(candidate_toe)>bed(current_toe)+elevation_tolerance) then
   return
  endif
  if(face_length(candidate_face)>face_length(current_face)) then
   prefer=.true.;return
  else if(face_length(candidate_face)<face_length(current_face)) then
   return
  endif
  if(candidate_toe<current_toe) then
   prefer=.true.;return
  else if(candidate_toe>current_toe) then
   return
  endif
  if(candidate_donor<current_donor) then
   prefer=.true.;return
  else if(candidate_donor>current_donor) then
   return
  endif
  if(candidate_face<current_face) prefer=.true.
 end function prefer_mapping


 subroutine heap_initialize(heap,capacity,ierr)
  type(bank_toe_heap),intent(inout) :: heap
  integer,intent(in) :: capacity
  integer,intent(out) :: ierr
  integer :: allocation_status
  allocate(heap%cell(capacity),heap%key(capacity),stat=allocation_status)
  heap%count=0
  ierr=merge(0,BANK_TOE_ALLOCATION_FAILURE,allocation_status==0)
 end subroutine heap_initialize


 subroutine heap_push(heap,cell,key,ierr)
  type(bank_toe_heap),intent(inout) :: heap
  integer,intent(in) :: cell
  real(8),intent(in) :: key
  integer,intent(out) :: ierr
  integer :: position,parent,temporary_cell,new_capacity,allocation_status
  integer,allocatable :: grown_cell(:)
  real(8) :: temporary_key
  real(8),allocatable :: grown_key(:)

  ierr=0
  if(heap%count==size(heap%cell)) then
   new_capacity=max(16,2*size(heap%cell))
   allocate(grown_cell(new_capacity),grown_key(new_capacity),stat=allocation_status)
   if(allocation_status/=0) then
    ierr=BANK_TOE_ALLOCATION_FAILURE;return
   endif
   grown_cell(1:heap%count)=heap%cell(1:heap%count)
   grown_key(1:heap%count)=heap%key(1:heap%count)
   call move_alloc(grown_cell,heap%cell)
   call move_alloc(grown_key,heap%key)
  endif
  heap%count=heap%count+1;position=heap%count
  heap%cell(position)=cell;heap%key(position)=key
  do while(position>1)
   parent=position/2
   if(.not.heap_less(heap%key(position),heap%cell(position), &
                     heap%key(parent),heap%cell(parent))) exit
   temporary_cell=heap%cell(parent);heap%cell(parent)=heap%cell(position)
   heap%cell(position)=temporary_cell
   temporary_key=heap%key(parent);heap%key(parent)=heap%key(position)
   heap%key(position)=temporary_key
   position=parent
  enddo
 end subroutine heap_push


 subroutine heap_pop(heap,cell,key)
  type(bank_toe_heap),intent(inout) :: heap
  integer,intent(out) :: cell
  real(8),intent(out) :: key
  integer :: position,left,right,smallest,temporary_cell
  real(8) :: temporary_key

  cell=heap%cell(1);key=heap%key(1)
  heap%cell(1)=heap%cell(heap%count);heap%key(1)=heap%key(heap%count)
  heap%count=heap%count-1;position=1
  do
   left=2*position;right=left+1;smallest=position
   if(left<=heap%count) then
    if(heap_less(heap%key(left),heap%cell(left), &
                 heap%key(smallest),heap%cell(smallest))) smallest=left
   endif
   if(right<=heap%count) then
    if(heap_less(heap%key(right),heap%cell(right), &
                 heap%key(smallest),heap%cell(smallest))) smallest=right
   endif
   if(smallest==position) exit
   temporary_cell=heap%cell(position);heap%cell(position)=heap%cell(smallest)
   heap%cell(smallest)=temporary_cell
   temporary_key=heap%key(position);heap%key(position)=heap%key(smallest)
   heap%key(smallest)=temporary_key
   position=smallest
  enddo
 end subroutine heap_pop


 pure logical function heap_less(first_key,first_cell,second_key,second_cell) result(less)
  real(8),intent(in) :: first_key,second_key
  integer,intent(in) :: first_cell,second_cell
  if(first_key<second_key) then
   less=.true.
  else if(first_key>second_key) then
   less=.false.
  else
   less=first_cell<second_cell
  endif
 end function heap_less

end module braids_bank_toe_geometry_module
