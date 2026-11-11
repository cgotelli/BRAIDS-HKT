! Performance implementation of the existing ell=2h validation operator.
! Auxiliary-physics MIC-PCG candidate; see performance_auxiliary_20260930_v1.
! The earlier v4 exact-order milestone did not select this preconditioner.
! New qualification uses physical-error/residual gates, not bitwise equality
! between different solvers. The ell=2h physical closure remains experimental.
! Same operator, true-residual check and cold initial guess. Historical strict
! tolerance is the API default; runtime builders may pass qualified targets.
! Row-sum-preserving modified IC(0) changes only the preconditioner.
module curvature_mic_backend
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 implicit none
 private
 public :: regularize_curvature,reset_curvature_workspace
 ! The native source half-step is serial (outside all OpenMP regions). This
 ! reusable workspace is intentionally not a concurrently callable API.
 integer,allocatable,save :: saved_owner(:),saved_neighbour(:),row(:),column(:),face_slot(:)
 integer,allocatable,save :: upper_row(:),upper_slot(:),upper_owner(:)
 real(8),allocatable,save :: saved_area(:),saved_xy(:,:),saved_normal(:,:),saved_length(:)
 real(8),allocatable,save :: projection(:),distance2(:),conductance(:),diagonal(:),factor(:),pivot(:)
 real(8),allocatable,save :: rhs(:),r(:),z(:),p(:),ap(:),icdiag(:)
contains
 subroutine reset_curvature_workspace()
  if(.not.allocated(saved_owner)) return
  deallocate(saved_owner,saved_neighbour,saved_area,saved_xy,saved_normal,saved_length, &
   row,column,face_slot,projection,distance2,conductance,diagonal,factor,pivot,rhs,r,z,p,ap, &
   upper_row,upper_slot,upper_owner,icdiag)
 end subroutine

 subroutine prepare_workspace(nc,nf,owner,neighbour,area,xy,normal,length,ok)
  integer,intent(in) :: nc,nf,owner(nf),neighbour(nf)
  real(8),intent(in) :: area(nc),xy(2,nc),normal(2,nf),length(nf)
  logical,intent(out) :: ok
  integer :: old_row(nc+1),cursor(nc),f,c,n,j,k,m,slot,start,finish,value
  real(8) :: delta(2)
  logical :: same
  ok=.false.;same=.false.
  if(allocated(saved_owner)) then
   if(size(saved_owner)==nf.and.size(saved_area)==nc) then
    same=all(saved_owner==owner).and.all(saved_neighbour==neighbour).and. &
     all(saved_area==area).and.all(saved_xy==xy).and.all(saved_normal==normal).and.all(saved_length==length)
   endif
  endif
  if(same) then
   ok=.true.;return
  endif
  call reset_curvature_workspace()
  allocate(saved_owner(nf),saved_neighbour(nf),saved_area(nc),saved_xy(2,nc), &
   saved_normal(2,nf),saved_length(nf),row(nc+1),column(nf),face_slot(nf), &
   projection(nf),distance2(nf),conductance(nf),diagonal(nc),factor(nf),pivot(nc), &
   rhs(nc),r(nc),z(nc),p(nc),ap(nc),upper_row(nc+1),upper_slot(nf),upper_owner(nf),icdiag(nc))
  saved_owner=owner;saved_neighbour=neighbour;saved_area=area
  saved_xy=xy;saved_normal=normal;saved_length=length
  projection=0;distance2=1;row=0;column=0;face_slot=0
  do f=1,nf
   c=owner(f);n=neighbour(f)
   if(n<=0) cycle
   delta=xy(:,n)-xy(:,c);distance2(f)=sum(delta*delta)
   if(distance2(f)<=0) then
    call reset_curvature_workspace();return
   endif
   projection(f)=max(0.d0,sum(delta*normal(:,f)))
   j=max(c,n);row(j+1)=row(j+1)+1
  enddo
  row(1)=1
  do c=1,nc
   row(c+1)=row(c+1)+row(c)
  enddo
  cursor=row(1:nc)
  do f=1,nf
   c=owner(f);n=neighbour(f)
   if(n<=0) cycle
   j=max(c,n);column(cursor(j))=min(c,n);cursor(j)=cursor(j)+1
  enddo
  ! Sort each lower-triangular row, and merge repeated cell-pair faces.
  old_row=row;slot=1
  do c=1,nc
   start=old_row(c);finish=old_row(c+1)-1
   do j=start+1,finish
    value=column(j);k=j-1
    do while(k>=start)
     if(column(k)<=value) exit
     column(k+1)=column(k);k=k-1
    enddo
    column(k+1)=value
   enddo
   row(c)=slot;m=0
   do j=start,finish
    value=column(j)
    if(value==m) cycle
    column(slot)=value;slot=slot+1;m=value
   enddo
  enddo
  row(nc+1)=slot
  upper_row=0
  do k=1,slot-1
   upper_row(column(k)+1)=upper_row(column(k)+1)+1
  enddo
  upper_row(1)=1
  do c=1,nc
   upper_row(c+1)=upper_row(c+1)+upper_row(c)
  enddo
  cursor=upper_row(1:nc)
  do c=1,nc
   do k=row(c),row(c+1)-1
    j=column(k);m=cursor(j)
    upper_slot(m)=k;upper_owner(m)=c;cursor(j)=m+1
   enddo
  enddo
  do f=1,nf
   c=owner(f);n=neighbour(f)
   if(n<=0) cycle
   j=max(c,n);m=min(c,n)
   do k=row(j),row(j+1)-1
    if(column(k)==m) then
     face_slot(f)=k;exit
    endif
   enddo
  enddo
  ok=.true.
 end subroutine

 subroutine regularize_curvature(nc,nf,owner,neighbour,area,xy,normal,length,h,dry, &
                                  depth_factor,raw,filtered,iterations,residual,ok,relative_tolerance,absolute_tolerance)
  integer,intent(in) :: nc,nf,owner(nf),neighbour(nf)
  real(8),intent(in) :: area(nc),xy(2,nc),normal(2,nf),length(nf),h(nc),dry,depth_factor,raw(nc)
  real(8),intent(out) :: filtered(nc),residual
  integer,intent(out) :: iterations
  logical,intent(out) :: ok
  ! Optional practical tolerances belong to the validation/runtime identity.
  ! Omitting them preserves the stringent historical kernel contract.
  real(8),intent(in),optional :: relative_tolerance,absolute_tolerance
  real(8) :: a,b,w,rho,rhonew,alpha,beta,denominator,scale,tolerance,entry
  real(8) :: relative_target,absolute_target
  integer :: c,n,f,k,j,left,right,slot,i
  logical :: use_ic
  filtered=raw;iterations=0;residual=huge(1.d0);ok=.false.
  relative_target=1.d-13;absolute_target=1.d-16
  if(present(relative_tolerance)) relative_target=relative_tolerance
  if(present(absolute_tolerance)) absolute_target=absolute_tolerance
  if(.not.ieee_is_finite(relative_target).or..not.ieee_is_finite(absolute_target)) return
  if(relative_target<=0.or.absolute_target<0) return
  if(nc<=0.or.nf<0) return
  if(.not.all(ieee_is_finite(raw)).or..not.all(ieee_is_finite(h))) return
  if(.not.all(ieee_is_finite(area)).or.any(area<=0).or.depth_factor<0) return
  if(.not.ieee_is_finite(depth_factor)) return
  if(depth_factor==0) then
   residual=0;ok=.true.;return
  endif
  if(any(owner<1).or.any(owner>nc).or.any(neighbour<0).or.any(neighbour>nc)) return
  if(any(owner==neighbour)) return
  if(.not.all(ieee_is_finite(xy)).or..not.all(ieee_is_finite(normal)).or. &
     .not.all(ieee_is_finite(length)).or.any(length<0)) return
  call prepare_workspace(nc,nf,owner,neighbour,area,xy,normal,length,ok)
  if(.not.ok) return
  ok=.false.;diagonal=area;conductance=0;factor=0
  do f=1,nf
   c=owner(f);n=neighbour(f)
   if(n<=0) cycle
   if(min(h(c),h(n))<=dry) cycle
   a=(depth_factor*h(c))**2;b=(depth_factor*h(n))**2
   ! Preserve the original face-coefficient arithmetic/order.
   w=2*a*b/(a+b)*length(f)*projection(f)/distance2(f)
   conductance(f)=w;diagonal(c)=diagonal(c)+w;diagonal(n)=diagonal(n)+w
   k=face_slot(f);factor(k)=factor(k)-w
  enddo
  ! Modified incomplete Cholesky with unchanged graph sparsity. Each dropped
  ! negative fill is moved to both affected diagonals, retaining its row sums.
  ! This handles the low-frequency error of a Laplacian much better than plain
  ! IC(0), without changing A or the PCG residual/convergence test. A nonpositive
  ! finite-precision pivot falls back to the original Jacobi preconditioner.
  use_ic=.true.;icdiag=diagonal
  do c=1,nc
   entry=icdiag(c)
   if(entry<=0.or..not.ieee_is_finite(entry)) then
    use_ic=.false.;exit
   endif
   pivot(c)=sqrt(entry)
   do left=upper_row(c),upper_row(c+1)-1
    k=upper_slot(left);factor(k)=factor(k)/pivot(c)
   enddo
   do left=upper_row(c),upper_row(c+1)-1
    i=upper_owner(left);k=upper_slot(left)
    icdiag(i)=icdiag(i)-factor(k)**2
    do right=left+1,upper_row(c+1)-1
     j=upper_owner(right);entry=factor(k)*factor(upper_slot(right));slot=0
     do n=row(j),row(j+1)-1
      if(column(n)==i) then
       slot=n;exit
      endif
     enddo
     if(slot>0) then
      factor(slot)=factor(slot)-entry
     else
      icdiag(i)=icdiag(i)-entry;icdiag(j)=icdiag(j)-entry
     endif
    enddo
   enddo
  enddo
  rhs=area*raw
  call multiply(filtered,ap)
  r=rhs-ap;scale=sqrt(sum(rhs*rhs))
  ! The algebraic target is explicit in the runtime identity. Changing it
  ! requires checking the coupled bed evolution, not just this residual.
  tolerance=relative_target*scale+absolute_target*sqrt(sum(area*area))
  residual=sqrt(sum(r*r))
  if(residual<=tolerance) then
   ok=.true.;return
  endif
  call precondition();p=z;rho=sum(r*z)
  do k=1,500
   call multiply(p,ap)
   denominator=sum(p*ap)
   if(denominator<=0.or..not.ieee_is_finite(denominator)) return
   alpha=rho/denominator
   filtered=filtered+alpha*p;r=r-alpha*ap
   iterations=k;residual=sqrt(sum(r*r))
   if(residual<=tolerance) then
    call multiply(filtered,ap)
    residual=sqrt(sum((rhs-ap)**2))
    ok=residual<=10*tolerance.and.all(ieee_is_finite(filtered))
    return
   endif
   call precondition();rhonew=sum(r*z);beta=rhonew/rho
   p=z+beta*p;rho=rhonew
  enddo
 contains
  subroutine multiply(x,y)
   real(8),intent(in) :: x(nc)
   real(8),intent(out) :: y(nc)
   real(8) :: flux
   integer :: f,c,n
   y=area*x
   do f=1,nf
    n=neighbour(f)
    if(n<=0.or.conductance(f)==0) cycle
    c=owner(f);flux=conductance(f)*(x(c)-x(n))
    y(c)=y(c)+flux;y(n)=y(n)-flux
   enddo
  end subroutine
  subroutine precondition()
   integer :: c,k
   if(.not.use_ic) then
    z=r/diagonal;return
   endif
   z=r
   do c=1,nc
    do k=row(c),row(c+1)-1
     z(c)=z(c)-factor(k)*z(column(k))
    enddo
    z(c)=z(c)/pivot(c)
   enddo
   do c=nc,1,-1
    z(c)=z(c)/pivot(c)
    do k=row(c),row(c+1)-1
     z(column(k))=z(column(k))-factor(k)*z(c)
    enddo
   enddo
  end subroutine
 end subroutine
end module

! Research preconditioner only: A = diag(area) + graph Helmholtz Laplacian.
! Same physical coefficients, cold guess and public residual contract as MIC.
! Piecewise-constant prolongation, transpose restriction, Galerkin coarse
! matrices, symmetric weighted-Jacobi V cycles and a small dense coarse solve.
! All state-dependent coefficients are rebuilt on every call. No warm history.
module curvature_multilevel_backend
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 use, intrinsic :: iso_fortran_env, only: int64
 use omp_lib, only: omp_get_max_threads
 implicit none
 private
 public :: regularize_curvature,reset_curvature_workspace
 integer,parameter :: maximum_levels=32,coarse_limit=32,tile_size=512
 type grid_level
  integer :: n=0,nf=0
  integer,allocatable :: owner(:),neighbour(:),row(:),edge(:),other(:),parent(:),child_row(:),child(:)
  integer,allocatable :: edge_child_row(:),edge_child(:)
  real(8),allocatable :: mass(:),xy(:,:),weight(:),diagonal(:),b(:),x(:),work(:),res(:)
 end type
 type(grid_level),save :: levels(maximum_levels)
 integer,save :: nlevels=0,team_threads=1
 integer,allocatable,save :: saved_owner(:),saved_neighbour(:)
 real(8),allocatable,save :: saved_area(:),saved_xy(:,:),saved_normal(:,:),saved_length(:)
 real(8),allocatable,save :: projection(:),distance2(:),rhs(:),r(:),z(:),p(:),ap(:),dots(:),coarse_factor(:,:)
contains
 subroutine reset_curvature_workspace()
  type(grid_level) :: empty
  integer :: l
  do l=1,nlevels
   levels(l)=empty
  enddo
  nlevels=0
  if(.not.allocated(saved_owner)) return
  deallocate(saved_owner,saved_neighbour,saved_area,saved_xy,saved_normal,saved_length, &
   projection,distance2,rhs,r,z,p,ap,dots,coarse_factor)
 end subroutine

 subroutine allocate_level(l,n,nf)
  integer,intent(in) :: l,n,nf
  levels(l)%n=n;levels(l)%nf=nf
  allocate(levels(l)%owner(nf),levels(l)%neighbour(nf),levels(l)%mass(n),levels(l)%xy(2,n), &
   levels(l)%weight(nf),levels(l)%diagonal(n),levels(l)%b(n),levels(l)%x(n),levels(l)%work(n),levels(l)%res(n))
 end subroutine

 subroutine adjacency(l)
  integer,intent(in) :: l
  integer :: c,n,f,k,cursor(levels(l)%n)
  allocate(levels(l)%row(levels(l)%n+1));levels(l)%row=0
  do f=1,levels(l)%nf
   c=levels(l)%owner(f);n=levels(l)%neighbour(f)
   if(n<=0) cycle
   levels(l)%row(c+1)=levels(l)%row(c+1)+1;levels(l)%row(n+1)=levels(l)%row(n+1)+1
  enddo
  levels(l)%row(1)=1
  do c=1,levels(l)%n
   levels(l)%row(c+1)=levels(l)%row(c+1)+levels(l)%row(c)
  enddo
  allocate(levels(l)%edge(levels(l)%row(levels(l)%n+1)-1),levels(l)%other(levels(l)%row(levels(l)%n+1)-1))
  cursor=levels(l)%row(:levels(l)%n)
  do f=1,levels(l)%nf
   c=levels(l)%owner(f);n=levels(l)%neighbour(f)
   if(n<=0) cycle
   k=cursor(c);levels(l)%edge(k)=f;levels(l)%other(k)=n;cursor(c)=k+1
   k=cursor(n);levels(l)%edge(k)=f;levels(l)%other(k)=c;cursor(n)=k+1
  enddo
 end subroutine

 ! Deterministic sparse pair grouping avoids an allocation spanning the bounding
 ! rectangle (which may contain large empty regions or projected coordinates).
 subroutine group_pairs(a,b,count,group,unique)
  integer(int64),intent(in) :: a(:),b(:)
  integer,intent(out) :: count,group(size(a)),unique(size(a))
  integer :: table(2*size(a)+1),i,k,j,nt
  nt=size(table);table=0;count=0;unique=0
  do i=1,size(a)
   k=1+int(modulo(modulo(a(i),int(nt,int64))*104729_int64+ &
    modulo(b(i),int(nt,int64))*13007_int64,int(nt,int64)))
   do
    j=table(k)
    if(j==0) then
     count=count+1;table(k)=i;group(i)=count;unique(count)=i;exit
    endif
    if(a(i)==a(j).and.b(i)==b(j)) then
     group(i)=group(j);exit
    endif
    k=1+mod(k,nt)
   enddo
  enddo
 end subroutine

 subroutine inverse_groups(parent,nparent,row,child)
  integer,intent(in) :: parent(:),nparent
  integer,allocatable,intent(out) :: row(:),child(:)
  integer :: j,k,cursor(nparent)
  allocate(row(nparent+1));row=0
  do j=1,size(parent)
   if(parent(j)>0) row(parent(j)+1)=row(parent(j)+1)+1
  enddo
  row(1)=1
  do j=1,nparent
   row(j+1)=row(j+1)+row(j)
  enddo
  allocate(child(row(nparent+1)-1));cursor=row(:nparent)
  do j=1,size(parent)
   if(parent(j)<=0) cycle
   k=cursor(parent(j));child(k)=j;cursor(parent(j))=k+1
  enddo
 end subroutine

 subroutine coarsen(l,width,origin,ok)
  integer,intent(in) :: l
  real(8),intent(inout) :: width
  real(8),intent(in) :: origin(2)
  logical,intent(out) :: ok
  integer(int64) :: bx(levels(l)%n),by(levels(l)%n),a(levels(l)%nf),b(levels(l)%nf)
  integer :: groups(levels(l)%n),unique(levels(l)%n),edge_map(levels(l)%nf),edge_unique(levels(l)%nf)
  integer :: c,n,f,k,j,ncoarse,nedge,counted,trial
  real(8) :: value(2)
  ok=.false.
  do trial=1,32
   do c=1,levels(l)%n
    value=(levels(l)%xy(:,c)-origin)/width
    if(any(abs(value)>1.d12)) return
    bx(c)=floor(value(1),kind=int64);by(c)=floor(value(2),kind=int64)
   enddo
   call group_pairs(bx,by,ncoarse,groups,unique)
   if(ncoarse<levels(l)%n) exit
   width=2*width
  enddo
  if(ncoarse>=levels(l)%n) return
  allocate(levels(l)%parent(levels(l)%n));levels(l)%parent=groups
  counted=0
  do f=1,levels(l)%nf
   c=groups(levels(l)%owner(f));n=0
   if(levels(l)%neighbour(f)>0) n=groups(levels(l)%neighbour(f))
   if(n<=0.or.n==c) cycle
   counted=counted+1;a(counted)=min(c,n);b(counted)=max(c,n)
  enddo
  call group_pairs(a(:counted),b(:counted),nedge,edge_map(:counted),edge_unique(:counted))
  call allocate_level(l+1,ncoarse,nedge);nlevels=l+1
  call inverse_groups(groups,ncoarse,levels(l)%child_row,levels(l)%child)
  do c=1,ncoarse
   levels(l+1)%mass(c)=0;levels(l+1)%xy(:,c)=0
   do k=levels(l)%child_row(c),levels(l)%child_row(c+1)-1
    j=levels(l)%child(k)
    levels(l+1)%mass(c)=levels(l+1)%mass(c)+levels(l)%mass(j)
    levels(l+1)%xy(:,c)=levels(l+1)%xy(:,c)+levels(l)%mass(j)*levels(l)%xy(:,j)
   enddo
   levels(l+1)%xy(:,c)=levels(l+1)%xy(:,c)/levels(l+1)%mass(c)
  enddo
  do f=1,nedge
   levels(l+1)%owner(f)=int(a(edge_unique(f)));levels(l+1)%neighbour(f)=int(b(edge_unique(f)))
  enddo
  ! Reuse a as the coarse-face mapping for all original faces (zero=internal).
  a=0;counted=0
  do f=1,levels(l)%nf
   c=groups(levels(l)%owner(f));n=0
   if(levels(l)%neighbour(f)>0) n=groups(levels(l)%neighbour(f))
   if(n<=0.or.n==c) cycle
   counted=counted+1;a(f)=edge_map(counted)
  enddo
  call inverse_groups(int(a),nedge,levels(l)%edge_child_row,levels(l)%edge_child)
  call adjacency(l+1);ok=.true.
 end subroutine

 subroutine prepare_workspace(nc,nf,owner,neighbour,area,xy,normal,length,ok)
  integer,intent(in) :: nc,nf,owner(nf),neighbour(nf)
  real(8),intent(in) :: area(nc),xy(2,nc),normal(2,nf),length(nf)
  logical,intent(out) :: ok
  integer :: f,c,n,l
  real(8) :: delta(2),origin(2),width
  logical :: same
  ok=.false.;same=.false.
  if(allocated(saved_owner)) then
   if(size(saved_owner)==nf.and.size(saved_area)==nc) then
    same=all(saved_owner==owner).and.all(saved_neighbour==neighbour).and.all(saved_area==area).and. &
     all(saved_xy==xy).and.all(saved_normal==normal).and.all(saved_length==length)
   endif
  endif
  if(same) then
   ok=.true.;return
  endif
  call reset_curvature_workspace()
  allocate(saved_owner(nf),saved_neighbour(nf),saved_area(nc),saved_xy(2,nc),saved_normal(2,nf),saved_length(nf), &
   projection(nf),distance2(nf),rhs(nc),r(nc),z(nc),p(nc),ap(nc),dots((nc+tile_size-1)/tile_size))
  ! Allocate the coarse factor even on failure so reset has one simple contract.
  allocate(coarse_factor(0,0))
  saved_owner=owner;saved_neighbour=neighbour;saved_area=area;saved_xy=xy;saved_normal=normal;saved_length=length
  call allocate_level(1,nc,nf);nlevels=1
  levels(1)%owner=owner;levels(1)%neighbour=neighbour;levels(1)%mass=area;levels(1)%xy=xy
  projection=0;distance2=1
  do f=1,nf
   c=owner(f);n=neighbour(f)
   if(n<=0) cycle
   delta=xy(:,n)-xy(:,c);distance2(f)=sum(delta*delta)
   if(distance2(f)<=0) then
    call reset_curvature_workspace();return
   endif
   projection(f)=max(0.d0,sum(delta*normal(:,f)))
  enddo
  call adjacency(1)
  ! Put regular Cartesian centroids inside bins, not on bin boundaries.
  ! This avoids floor() decisions on nearly integral floating-point ratios.
  origin=minval(xy,dim=2)-.5d0*sqrt(maxval(area));width=2*sqrt(maxval(area))
  do l=1,maximum_levels-1
   if(levels(l)%n<=coarse_limit) exit
   call coarsen(l,width,origin,ok)
   if(.not.ok) then
    call reset_curvature_workspace();return
   endif
   width=2*width
  enddo
  if(levels(nlevels)%n>coarse_limit) then
   call reset_curvature_workspace();return
  endif
  deallocate(coarse_factor);allocate(coarse_factor(levels(nlevels)%n,levels(nlevels)%n))
  ok=.true.
 end subroutine

 subroutine coefficients(h,dry,depth_factor,ok)
  real(8),intent(in) :: h(:),dry,depth_factor
  logical,intent(out) :: ok
  integer :: f,c,n,l,k,i,j
  real(8) :: a,b,w,value
  !$omp parallel do schedule(static) num_threads(team_threads) if(size(h)>=4096) default(shared) private(f,c,n,a,b,w)
  do f=1,levels(1)%nf
   c=levels(1)%owner(f);n=levels(1)%neighbour(f);levels(1)%weight(f)=0
   if(n<=0) cycle
   if(min(h(c),h(n))<=dry) cycle
   a=(depth_factor*h(c))**2;b=(depth_factor*h(n))**2
   w=2*a*b/(a+b)*saved_length(f)*projection(f)/distance2(f)
   levels(1)%weight(f)=w
  enddo
  !$omp end parallel do
  do l=1,nlevels
   if(l>1) then
    !$omp parallel do schedule(static) num_threads(team_threads) if(levels(l)%nf>=4096) default(shared) private(f,k,w)
    do f=1,levels(l)%nf
     w=0
     do k=levels(l-1)%edge_child_row(f),levels(l-1)%edge_child_row(f+1)-1
      w=w+levels(l-1)%weight(levels(l-1)%edge_child(k))
     enddo
     levels(l)%weight(f)=w
    enddo
    !$omp end parallel do
   endif
   !$omp parallel do schedule(static) num_threads(team_threads) if(levels(l)%n>=4096) default(shared) private(c,k,w)
   do c=1,levels(l)%n
    w=levels(l)%mass(c)
    do k=levels(l)%row(c),levels(l)%row(c+1)-1
     w=w+levels(l)%weight(levels(l)%edge(k))
    enddo
    levels(l)%diagonal(c)=w
   enddo
   !$omp end parallel do
  enddo
  l=nlevels;coarse_factor=0
  do c=1,levels(l)%n
   coarse_factor(c,c)=levels(l)%diagonal(c)
   do k=levels(l)%row(c),levels(l)%row(c+1)-1
    j=levels(l)%other(k);coarse_factor(c,j)=coarse_factor(c,j)-levels(l)%weight(levels(l)%edge(k))
   enddo
  enddo
  ok=.false.
  do j=1,levels(l)%n
   value=coarse_factor(j,j)-sum(coarse_factor(j,:j-1)**2)
   if(.not.ieee_is_finite(value).or.value<=0) return
   coarse_factor(j,j)=sqrt(value)
   do i=j+1,levels(l)%n
    coarse_factor(i,j)=(coarse_factor(i,j)-sum(coarse_factor(i,:j-1)*coarse_factor(j,:j-1)))/coarse_factor(j,j)
   enddo
  enddo
  ok=.true.
 end subroutine

 subroutine multiply(l,x,y)
  integer,intent(in) :: l
  real(8),intent(in) :: x(:)
  real(8),intent(out) :: y(:)
  integer :: c,k,f,n
  real(8) :: value,flux
  !$omp parallel do schedule(static) num_threads(team_threads) if(levels(l)%n>=4096) default(shared) private(c,k,f,n,value,flux)
  do c=1,levels(l)%n
   value=levels(l)%mass(c)*x(c)
   do k=levels(l)%row(c),levels(l)%row(c+1)-1
    f=levels(l)%edge(k);n=levels(l)%other(k)
    if(levels(l)%weight(f)==0) cycle
    if(levels(l)%owner(f)==c) then
     flux=levels(l)%weight(f)*(x(c)-x(n));value=value+flux
    else
     flux=levels(l)%weight(f)*(x(n)-x(c));value=value-flux
    endif
   enddo
   y(c)=value
  enddo
  !$omp end parallel do
 end subroutine

 subroutine smooth(l,zero_initial)
  integer,intent(in) :: l
  logical,intent(in) :: zero_initial
  integer :: sweep,c,k,f,n,first
  real(8) :: value
  real(8),allocatable :: swap(:)
  first=1
  if(zero_initial) then
   ! Every V-cycle level starts from zero: its first Jacobi sweep needs no
   ! neighbour gather. This is an exact algebraic special case, not lagging.
   !$omp parallel do schedule(static) num_threads(team_threads) if(levels(l)%n>=4096) default(shared) private(c)
   do c=1,levels(l)%n
    levels(l)%x(c)=(2.d0/3.d0)*levels(l)%b(c)/levels(l)%diagonal(c)
   enddo
   !$omp end parallel do
   first=2
  endif
  do sweep=first,2
   !$omp parallel do schedule(static) num_threads(team_threads) if(levels(l)%n>=4096) default(shared) private(c,k,f,n,value)
   do c=1,levels(l)%n
    value=levels(l)%b(c)
    do k=levels(l)%row(c),levels(l)%row(c+1)-1
     f=levels(l)%edge(k);n=levels(l)%other(k)
     value=value+levels(l)%weight(f)*levels(l)%x(n)
    enddo
    levels(l)%work(c)=levels(l)%x(c)/3.d0+(2.d0/3.d0)*value/levels(l)%diagonal(c)
   enddo
   !$omp end parallel do
   call move_alloc(levels(l)%x,swap);call move_alloc(levels(l)%work,levels(l)%x);call move_alloc(swap,levels(l)%work)
  enddo
 end subroutine

 recursive subroutine cycle(l)
  integer,intent(in) :: l
  integer :: c,k,i,n
  real(8) :: value
  if(l==nlevels) then
   n=levels(l)%n;levels(l)%x=levels(l)%b
   do c=1,n
    levels(l)%x(c)=(levels(l)%x(c)-sum(coarse_factor(c,:c-1)*levels(l)%x(:c-1)))/coarse_factor(c,c)
   enddo
   do c=n,1,-1
    levels(l)%x(c)=(levels(l)%x(c)-sum(coarse_factor(c+1:n,c)*levels(l)%x(c+1:n)))/coarse_factor(c,c)
   enddo
   return
  endif
  call smooth(l,.true.);call multiply(l,levels(l)%x,levels(l)%res)
  !$omp parallel do schedule(static) num_threads(team_threads) if(levels(l+1)%n>=2048) default(shared) private(c,k,i,value)
  do c=1,levels(l+1)%n
   value=0
   do k=levels(l)%child_row(c),levels(l)%child_row(c+1)-1
    i=levels(l)%child(k);value=value+levels(l)%b(i)-levels(l)%res(i)
   enddo
   levels(l+1)%b(c)=value;levels(l+1)%x(c)=0
  enddo
  !$omp end parallel do
  call cycle(l+1)
  !$omp parallel do schedule(static) num_threads(team_threads) if(levels(l)%n>=4096) default(shared) private(c)
  do c=1,levels(l)%n
   levels(l)%x(c)=levels(l)%x(c)+levels(l+1)%x(levels(l)%parent(c))
  enddo
  !$omp end parallel do
  call smooth(l,.false.)
 end subroutine

 real(8) function inner(x,y) result(value)
  real(8),intent(in) :: x(:),y(:)
  integer :: b,lo,hi
  !$omp parallel do schedule(static) num_threads(team_threads) if(size(x)>=4096) default(shared) private(b,lo,hi)
  do b=1,size(dots)
   lo=1+(b-1)*tile_size;hi=min(size(x),b*tile_size)
   dots(b)=sum(x(lo:hi)*y(lo:hi))
  enddo
  !$omp end parallel do
  value=sum(dots)
 end function

 subroutine precondition()
  integer :: c
  !$omp parallel do schedule(static) num_threads(team_threads) if(size(r)>=4096) default(shared) private(c)
  do c=1,size(r)
   levels(1)%b(c)=r(c);levels(1)%x(c)=0
  enddo
  !$omp end parallel do
  call cycle(1);z=levels(1)%x
 end subroutine

 subroutine regularize_curvature(nc,nf,owner,neighbour,area,xy,normal,length,h,dry, &
                                 depth_factor,raw,filtered,iterations,residual,ok,relative_tolerance,absolute_tolerance)
  integer,intent(in) :: nc,nf,owner(nf),neighbour(nf)
  real(8),intent(in) :: area(nc),xy(2,nc),normal(2,nf),length(nf),h(nc),dry,depth_factor,raw(nc)
  real(8),intent(out) :: filtered(nc),residual
  integer,intent(out) :: iterations
  logical,intent(out) :: ok
  real(8),intent(in),optional :: relative_tolerance,absolute_tolerance
  real(8) :: rt,at,tolerance,rho,rhonew,denominator,alpha,beta
  integer :: k,c
  filtered=raw;iterations=0;residual=huge(1.d0);ok=.false.;rt=1.d-13;at=1.d-16
  if(present(relative_tolerance)) rt=relative_tolerance
  if(present(absolute_tolerance)) at=absolute_tolerance
  if(.not.ieee_is_finite(rt).or..not.ieee_is_finite(at)) return
  if(rt<=0.or.at<0.or.nc<=0.or.nf<0) return
  if(.not.all(ieee_is_finite(raw)).or..not.all(ieee_is_finite(h))) return
  if(.not.all(ieee_is_finite(area)).or.any(area<=0)) return
  if(.not.ieee_is_finite(depth_factor).or.depth_factor<0) return
  if(depth_factor==0) then
   residual=0;ok=.true.;return
  endif
  if(any(owner<1).or.any(owner>nc).or.any(neighbour<0).or.any(neighbour>nc)) return
  if(any(owner==neighbour)) return
  if(.not.all(ieee_is_finite(xy)).or..not.all(ieee_is_finite(normal)).or. &
   .not.all(ieee_is_finite(length)).or.any(length<0)) return
  team_threads=min(omp_get_max_threads(),merge(4,merge(8,16,nc<524288),nc<131072))
  call prepare_workspace(nc,nf,owner,neighbour,area,xy,normal,length,ok)
  if(.not.ok) return
  call coefficients(h,dry,depth_factor,ok)
  if(.not.ok) return
  ok=.false.;rhs=area*raw
  tolerance=rt*sqrt(inner(rhs,rhs))+at*sqrt(inner(area,area))
  call multiply(1,filtered,ap);r=rhs-ap;residual=sqrt(inner(r,r))
  if(residual<=tolerance) then
   ok=.true.;return
  endif
  call precondition();p=z;rho=inner(r,z)
  do k=1,500
   call multiply(1,p,ap);denominator=inner(p,ap)
   if(denominator<=0.or..not.ieee_is_finite(denominator)) return
   alpha=rho/denominator
   !$omp parallel do schedule(static) num_threads(team_threads) if(nc>=4096) default(shared) private(c)
   do c=1,nc
    filtered(c)=filtered(c)+alpha*p(c);r(c)=r(c)-alpha*ap(c)
   enddo
   !$omp end parallel do
   iterations=k;residual=sqrt(inner(r,r))
   if(residual<=tolerance) then
    call multiply(1,filtered,ap);r=rhs-ap;residual=sqrt(inner(r,r))
    ok=residual<=10*tolerance.and.all(ieee_is_finite(filtered));return
   endif
   call precondition();rhonew=inner(r,z);beta=rhonew/rho
   !$omp parallel do schedule(static) num_threads(team_threads) if(nc>=4096) default(shared) private(c)
   do c=1,nc
    p(c)=z(c)+beta*p(c)
   enddo
   !$omp end parallel do
   rho=rhonew
  enddo
 end subroutine
end module

! Measured cell-count dispatch, independent of OpenMP team count.
! MIC remains faster on the 5k/20k controls. Multilevel starts at 65,536 cells.
! Fine-grid equations and tolerance do not change at this crossover.
module secondary_curvature_regularization
 use curvature_mic_backend,only:mic_solve=>regularize_curvature,mic_reset=>reset_curvature_workspace
 use curvature_multilevel_backend,only:mg_solve=>regularize_curvature,mg_reset=>reset_curvature_workspace
 implicit none
 private
 public :: regularize_curvature,reset_curvature_workspace
 integer,save :: selected=0
contains
 subroutine reset_curvature_workspace()
  call mic_reset();call mg_reset();selected=0
 end subroutine
 subroutine regularize_curvature(nc,nf,owner,neighbour,area,xy,normal,length,h,dry, &
                                 depth_factor,raw,filtered,iterations,residual,ok,relative_tolerance,absolute_tolerance)
  integer,intent(in) :: nc,nf,owner(nf),neighbour(nf)
  real(8),intent(in) :: area(nc),xy(2,nc),normal(2,nf),length(nf),h(nc),dry,depth_factor,raw(nc)
  real(8),intent(out) :: filtered(nc),residual
  integer,intent(out) :: iterations
  logical,intent(out) :: ok
  real(8),intent(in),optional :: relative_tolerance,absolute_tolerance
  integer :: wanted
  wanted=1
  if(nc>=65536) wanted=2
  if(selected/=wanted) then
   call reset_curvature_workspace();selected=wanted
  endif
  if(wanted==1) then
   call mic_solve(nc,nf,owner,neighbour,area,xy,normal,length,h,dry,depth_factor,raw, &
    filtered,iterations,residual,ok,relative_tolerance,absolute_tolerance)
  else
   call mg_solve(nc,nf,owner,neighbour,area,xy,normal,length,h,dry,depth_factor,raw, &
    filtered,iterations,residual,ok,relative_tolerance,absolute_tolerance)
  endif
 end subroutine
end module
