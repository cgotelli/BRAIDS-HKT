 logical function projected_ok188(x,y,count,ax,ay,b)
  integer,intent(in)::count
  real(8),intent(in)::x,y,ax(64),ay(64),b(64)
  integer::i
  projected_ok188=.false.
  do i=1,count
   if(ax(i)*x+ay(i)*y>b(i)+2.d-14)return
  enddo
  projected_ok188=.true.
 end function

 subroutine project_pair188(x,y,count,ax,ay,b)
  integer,intent(in)::count
  real(8),intent(in)::ax(64),ay(64),b(64)
  real(8),intent(inout)::x,y
  real(8)::rx,ry,bx,by,best,dp,cx,cy,d,det
  integer::i,j
  rx=x;ry=y
  if(projected_ok188(x,y,count,ax,ay,b))return
  bx=0.d0;by=0.d0;best=rx*rx+ry*ry
  do i=1,count
   dp=ax(i)*rx+ay(i)*ry-b(i);cx=rx-dp*ax(i);cy=ry-dp*ay(i)
   d=(cx-rx)*(cx-rx)+(cy-ry)*(cy-ry)
   if(d<best.and.projected_ok188(cx,cy,count,ax,ay,b))then
    bx=cx;by=cy;best=d
   endif
   do j=1,i-1
    det=ax(i)*ay(j)-ay(i)*ax(j)
    if(abs(det)<1.d-12)cycle
    cx=(b(i)*ay(j)-ay(i)*b(j))/det;cy=(ax(i)*b(j)-b(i)*ax(j))/det
    d=(cx-rx)*(cx-rx)+(cy-ry)*(cy-ry)
    if(d<best.and.projected_ok188(cx,cy,count,ax,ay,b))then
     bx=cx;by=cy;best=d
    endif
   enddo
  enddo
  x=bx;y=by
 end subroutine

 subroutine project_cell188(c,v,grad,vlo,vhi,theta)
  integer,intent(in)::c
  real(8),intent(in)::v(nprimitive,nc),vlo(nprimitive,nc),vhi(nprimitive,nc)
  real(8),intent(inout)::grad(nprimitive,2,nc),theta(nprimitive)
  integer::k,f,a,m,count
  real(8)::ax(64),ay(64),b(64),rx,ry,dd,h,rawbed(2)
  if(offset(c+1)-offset(c)>32)return
  do k=offset(c),offset(c+1)-1
   f=faces(k)
   if(neighbour(f)==0.and.bid(f)<=0)return
  enddo
  h=v(1,c)-v(bed_primitive_index(),c);rawbed=grad(bed_primitive_index(),:,c)
  do a=1,2
   m=1
   if(a==2)m=bed_primitive_index()
   count=0
   do k=offset(c),offset(c+1)-1
    f=faces(k);rx=mid(1,f)-xy(1,c);ry=mid(2,f)-xy(2,c);dd=sqrt(rx*rx+ry*ry)
    if(neighbour(f)==0.and.bid(f)>0)then
     if(mode(bid(f))==2.and.h>dry_tolerance)then
      if(abs(dot_product(v(2:3,c),normal(:,f)))<sqrt(grav*h))then
       if(abs(rawbed(1)*rx+rawbed(2)*ry)>64*epsilon(1.d0)*max(1.d0,abs(v(bed_primitive_index(),c))))cycle
      endif
     endif
    endif
    if(dd<=0.d0)cycle
    count=count+1;ax(count)=rx/dd;ay(count)=ry/dd;b(count)=max(0.d0,vhi(m,c)-v(m,c))/dd
    count=count+1;ax(count)=-rx/dd;ay(count)=-ry/dd;b(count)=max(0.d0,v(m,c)-vlo(m,c))/dd
   enddo
   call project_pair188(grad(m,1,c),grad(m,2,c),count,ax,ay,b)
   theta(m)=1.d0
  enddo
 end subroutine
