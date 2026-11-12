! Bank-only wet/dry reconstruction. The hydraulic state is never modified.
module braids_bank_reconstruction_module
 implicit none
 private
 public :: positive_triangle_mean,bank_cell_depth_intercept,bank_face_mobility
contains
 pure real(8) function positive_triangle_mean(values) result(value)
  ! Exact mean of the positive part of an affine field on a triangle.
  real(8),intent(in) :: values(3)
  real(8) :: lo,hi,md
  lo=minval(values);hi=maxval(values);md=sum(values)-lo-hi
  if(lo>=0) then
   value=sum(values)/3
  else if(hi<=0) then
   value=0
  else if(md<=0) then
   value=hi**3/(3*(hi-lo)*(hi-md))
  else
   value=sum(values)/3+(-lo)**3/(3*(md-lo)*(hi-lo))
  endif
 end function

 pure real(8) function bank_cell_depth_intercept(depth,down,center,cell_area,ids,signs,mid,normal,length) result(a)
  ! h(x)=max(a+down.(x-center),0). Fit a to the stored water volume.
  ! Signed centroid/edge triangles also integrate a nonconvex control volume;
  ! all geometry comes from its actual faces, including cut and AMR faces.
  real(8),intent(in) :: depth,down(2),center(2),cell_area,mid(:,:),normal(:,:),length(:)
  integer,intent(in) :: ids(:),signs(:)
  real(8) :: heights(3,size(ids)),weights(size(ids)),tangent(2),r(2),lo,hi,mean_depth,minimum,maximum
  integer :: k,f,iteration
  heights(1,:)=0
  do k=1,size(ids)
   f=ids(k);r=mid(:,f)-center;tangent=[-normal(2,f),normal(1,f)]
   heights(2,k)=dot_product(down,r-.5d0*length(f)*tangent)
   heights(3,k)=dot_product(down,r+.5d0*length(f)*tangent)
   weights(k)=.5d0*length(f)*real(signs(k),8)*dot_product(normal(:,f),r)/cell_area
  enddo
  minimum=minval(heights);maximum=maxval(heights)
  if(depth<=0) then
   a=-maximum;return
  endif
  a=depth
  if(a+minimum>=0) return ! Full cell: centroid property gives the exact mean.
  lo=-maximum;hi=depth-minimum
  do iteration=1,48
   a=.5d0*(lo+hi);mean_depth=0
   do k=1,size(ids)
    mean_depth=mean_depth+weights(k)*positive_triangle_mean(a+heights(:,k))
   enddo
   if(mean_depth>depth) then
    hi=a
   else
    lo=a
   endif
  enddo
  a=.5d0*(lo+hi)
 end function

 pure real(8) function bank_face_mobility(slope,dry_slope,wet_slope,transition,dry_tol,depth_ends) result(alpha)
  ! Exact line mean of max(1-Scrit(h)/|grad z|,0). Each side supplies an
  ! affine depth trace. Split at every crossing/clamp so the narrow wet
  ! opening is integrated, rather than classified by a cell average.
  real(8),intent(in) :: slope,dry_slope,wet_slope,transition,dry_tol,depth_ends(2,2)
  real(8) :: breaks(12),targets(3),a(2),b(2),root,tmp,t,h,weight,critical
  integer :: count,k,j,i,ntarget
  alpha=0
  if(slope<=0) return
  if(dry_slope==wet_slope) then
   alpha=max(1.d0-dry_slope/slope,0.d0);return
  endif
  a=depth_ends(1,:);b=depth_ends(2,:)-a
  breaks(1:2)=[0.d0,1.d0];count=2
  if(transition>0) then
   targets=[0.d0,transition,transition*(slope-dry_slope)/(wet_slope-dry_slope)];ntarget=3
  else
   targets=0;targets(1)=dry_tol;ntarget=1
  endif
  do i=1,2
   if(b(i)==0) cycle
   do j=1,ntarget
    root=(targets(j)-a(i))/b(i)
    if(root>0.and.root<1) then
     count=count+1;breaks(count)=root
    endif
   enddo
  enddo
  if(b(1)/=b(2)) then
   root=(a(2)-a(1))/(b(1)-b(2))
   if(root>0.and.root<1) then
    count=count+1;breaks(count)=root
   endif
  endif
  do i=2,count
   tmp=breaks(i);j=i-1
   do while(j>=1)
    if(breaks(j)<=tmp) exit
    breaks(j+1)=breaks(j);j=j-1
   enddo
   breaks(j+1)=tmp
  enddo
  do k=1,count-1
   t=.5d0*(breaks(k)+breaks(k+1));h=max(0.d0,maxval(a+b*t))
   if(transition>0) then
    weight=min(1.d0,h/transition)
   else
    weight=0;if(h>dry_tol) weight=1
   endif
   critical=dry_slope+weight*(wet_slope-dry_slope)
   alpha=alpha+(breaks(k+1)-breaks(k))*max(1.d0-critical/slope,0.d0)
  enddo
 end function
end module
