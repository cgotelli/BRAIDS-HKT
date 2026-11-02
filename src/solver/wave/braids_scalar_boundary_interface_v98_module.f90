! Additive AMR-interface admission for the frozen v97 scalar-boundary batch.
!
! Selected level-local faces have neighbour=-1 and boundary_index=-1, but
! are internal coarse/fine interfaces rather than physical boundaries.  They
! participate in the accepted hydraulic water-divergence gather while never
! consuming physical boundary concentration/return memory or contributing to
! a named physical-boundary ledger.  All numerical work remains in the exact
! v97 batch; this wrapper only authenticates the extra topology class.
module braids_scalar_boundary_interface_v98_module
 use iso_c_binding
 use, intrinsic :: ieee_arithmetic
 use braids_scalar_boundary_v97_module, only: braids_scalar_boundary_v97
 implicit none
contains
 subroutine braids_scalar_boundary_interface_v98(nc,nf,nl,nb,nadj,nbfaces,owner,neighbour,bid, &
  offset,faces,signs,boffset,bfaces,trace_mask,area,length,h0,h1,mass,prescribed,water, &
  return_time,last,clock,neumann,dt,dry,threads,boundary,amount,next_last, &
  next_clock,ledger,divergence,water_ledger,continuity,status) bind(C)
  integer(c_int),value :: nc,nf,nl,nb,nadj,nbfaces,threads
  integer(c_int),intent(in) :: owner(nf),neighbour(nf),bid(nf),offset(nc+1)
  integer(c_int),intent(in) :: faces(nadj),signs(nadj),boffset(nb+1),bfaces(nbfaces)
  integer(c_int),intent(in) :: trace_mask(nf),neumann(nl)
  real(c_double),intent(in) :: area(nc),length(nf),h0(nc),h1(nc),mass(nc,nl),prescribed(nb,nl)
  real(c_double),intent(in) :: water(nf),return_time(nb,nl),last(nf,nl),clock(nf,nl)
  real(c_double),value :: dt,dry
  real(c_double),intent(out) :: boundary(nf,nl),amount(nf,nl),next_last(nf,nl),next_clock(nf,nl)
  real(c_double),intent(out) :: ledger(nb,nl),divergence(nc),water_ledger(nb),continuity(nc)
  integer(c_int),intent(out) :: status
  integer :: f

  status=0
  do f=1,nf
   if(trace_mask(f)/=0.and.trace_mask(f)/=1) then
    status=11;return
   endif
   if(trace_mask(f)==1.and.(neighbour(f)/=-1.or.bid(f)/=-1)) then
    status=12;return
   endif
   if(neighbour(f)==-1.and.bid(f)==-1.and.water(f)/=0.d0.and.trace_mask(f)/=1) then
    status=13;return
   endif
   if((neighbour(f)>=0.or.bid(f)>=0).and.trace_mask(f)==1) then
    status=14;return
   endif
   if(.not.ieee_is_finite(water(f))) then
    status=15;return
   endif
  enddo

  call braids_scalar_boundary_v97(nc,nf,nl,nb,nadj,nbfaces,owner,neighbour,bid, &
   offset,faces,signs,boffset,bfaces,area,length,h0,h1,mass,prescribed,water, &
   return_time,last,clock,neumann,dt,dry,threads,boundary,amount,next_last, &
   next_clock,ledger,divergence,water_ledger,continuity,status)
 end subroutine
end module
