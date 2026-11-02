! Identify solid faces explicitly; zero normal discharge alone is not a wall.
module braids_turbulence_wall_mask_module
 use amr_module,only:mthbc,xlower,ylower
 use geoclaw_module,only:dry_tolerance
 use braids_boundary_module,only:braids_boundary_enabled,braids_boundary_side_enabled,braids_bc_face
 use hydraulic_bc_module,only:hydraulic_bc_defined
 implicit none
 private
 public::braids_turbulence_wall_masks
contains
 logical function exterior_wall(side,index) result(wall)
  integer,intent(in)::side,index
  integer::id
  real(8)::discharge
  wall=mthbc(side)==3
  if(mthbc(side)/=0)return
  if(.not.braids_boundary_enabled())return
  if(.not.braids_boundary_side_enabled(side).and.hydraulic_bc_defined(side))return
  call braids_bc_face(side,index,id,discharge)
  wall=id==0
 end function
 subroutine braids_turbulence_wall_masks(nvar,m,n,mbc,dx,dy,xlo,ylo,q,bed,link,wx,wy)
  integer,intent(in)::nvar,m,n,mbc
  real(8),intent(in)::dx,dy,xlo,ylo,q(nvar,m,n),bed(m,n)
  logical,intent(in)::link(4)
  integer,intent(out)::wx(m+1,n),wy(m,n+1)
  integer::i,j,il,ir,jb,jt,index,ia,iz,ja,jz
  ! Sign is the outward normal from the wet cell. Dry neighbors qualify
  ! only when their bed blocks the wet free surface, not at an open shoreline.
  il=mbc+1;ir=m-mbc;jb=mbc+1;jt=n-mbc;wx=0;wy=0
  ia=il;iz=ir;ja=jb;jz=jt
  if(link(1))ia=ia-1
  if(link(2))iz=iz+1
  if(link(3))ja=ja-1
  if(link(4))jz=jz+1
  do j=ja,jz;do i=ia,iz+1
   if(i==il.and..not.link(1))then
    index=nint((ylo+(j-.5d0)*dy-ylower)/dy-.5d0)
    if(exterior_wall(1,index))wx(i,j)=-1
   elseif(i==ir+1.and..not.link(2))then
    index=nint((ylo+(j-.5d0)*dy-ylower)/dy-.5d0)
    if(exterior_wall(2,index))wx(i,j)=1
   else
    if(q(1,i-1,j)>dry_tolerance.and.q(1,i,j)<=dry_tolerance)then
     if(bed(i,j)>=bed(i-1,j)+q(1,i-1,j))wx(i,j)=1
    elseif(q(1,i,j)>dry_tolerance.and.q(1,i-1,j)<=dry_tolerance)then
     if(bed(i-1,j)>=bed(i,j)+q(1,i,j))wx(i,j)=-1
    endif
   endif
  enddo;enddo
  do i=ia,iz;do j=ja,jz+1
   if(j==jb.and..not.link(3))then
    index=nint((xlo+(i-.5d0)*dx-xlower)/dx-.5d0)
    if(exterior_wall(3,index))wy(i,j)=-1
   elseif(j==jt+1.and..not.link(4))then
    index=nint((xlo+(i-.5d0)*dx-xlower)/dx-.5d0)
    if(exterior_wall(4,index))wy(i,j)=1
   else
    if(q(1,i,j-1)>dry_tolerance.and.q(1,i,j)<=dry_tolerance)then
     if(bed(i,j)>=bed(i,j-1)+q(1,i,j-1))wy(i,j)=1
    elseif(q(1,i,j)>dry_tolerance.and.q(1,i,j-1)<=dry_tolerance)then
     if(bed(i,j-1)>=bed(i,j)+q(1,i,j))wy(i,j)=-1
    endif
   endif
  enddo;enddo
 end subroutine
end module
