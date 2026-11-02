 logical function braids_projection_ok(x,y,count,ax,ay,b)
  integer,intent(in)::count
  real(8),intent(in)::x,y,ax(64),ay(64),b(64)
  integer::i
  braids_projection_ok=.false.
  do i=1,count
   if(ax(i)*x+ay(i)*y>b(i)+2.d-14)return
  enddo
  braids_projection_ok=.true.
 end function

 subroutine braids_project_pair(x,y,count,ax,ay,b)
  integer,intent(in)::count
  real(8),intent(in)::ax(64),ay(64),b(64)
  real(8),intent(inout)::x,y
  real(8)::rx,ry,bx,by,best,dp,cx,cy,d,det
  integer::i,j
  rx=x;ry=y
  if(braids_projection_ok(x,y,count,ax,ay,b))return
  bx=0.d0;by=0.d0;best=rx*rx+ry*ry
  do i=1,count
   dp=ax(i)*rx+ay(i)*ry-b(i);cx=rx-dp*ax(i);cy=ry-dp*ay(i)
   d=(cx-rx)*(cx-rx)+(cy-ry)*(cy-ry)
   if(d<best.and.braids_projection_ok(cx,cy,count,ax,ay,b))then
    bx=cx;by=cy;best=d
   endif
   do j=1,i-1
    det=ax(i)*ay(j)-ay(i)*ax(j)
    if(abs(det)<1.d-12)cycle
    cx=(b(i)*ay(j)-ay(i)*b(j))/det;cy=(ax(i)*b(j)-b(i)*ax(j))/det
    d=(cx-rx)*(cx-rx)+(cy-ry)*(cy-ry)
    if(d<best.and.braids_projection_ok(cx,cy,count,ax,ay,b))then
     bx=cx;by=cy;best=d
    endif
   enddo
  enddo
  x=bx;y=by
 end subroutine

 subroutine braids_project_cell(c,v,grad,vlo,vhi,theta)
  integer,intent(in)::c
  real(8),intent(in)::v(nprimitive,nc),vlo(nprimitive,nc),vhi(nprimitive,nc)
  real(8),intent(inout)::grad(nprimitive,2,nc),theta(nprimitive)
  integer::k,f,a,m,count
  real(8)::ax(64),ay(64),b(64),rx,ry,dd,h,rawbed(2)
  if(offset(c+1)-offset(c)>32)return
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
    if(dd<=0.d0.or.(m==bed_primitive_index().and.neighbour(f)==0.and.bid(f)<=0))cycle
    count=count+1;ax(count)=rx/dd;ay(count)=ry/dd;b(count)=max(0.d0,vhi(m,c)-v(m,c))/dd
    count=count+1;ax(count)=-rx/dd;ay(count)=-ry/dd;b(count)=max(0.d0,v(m,c)-vlo(m,c))/dd
   enddo
   call braids_project_pair(grad(m,1,c),grad(m,2,c),count,ax,ay,b)
   theta(m)=1.d0
  enddo
 end subroutine
