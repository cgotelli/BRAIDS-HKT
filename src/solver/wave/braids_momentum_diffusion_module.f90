! Conservative horizontal turbulence and momentum diffusion for Cartesian WAVE.
! Version 1: constant viscosity. Version 2: Erpicum et al. (2009)
! depth-integrated two-length-scale k-epsilon closure. q(6)=k', q(7)=epsilon'.
module braids_momentum_diffusion_module
 use,intrinsic::ieee_arithmetic,only:ieee_is_finite
 use geoclaw_module,only:coordinate_system,dry_tolerance,grav,use_fwave_positivity_limiter
 use amr_module,only:mxnest,xperdom,yperdom
 implicit none
 private
 public::configure_braids_momentum_diffusion,braids_momentum_diffusion,braids_turbulence_fluxes
 public::braids_wall_turbulence_enabled,braids_smooth_wall_state
 public::braids_diffusion_enabled,braids_diffusion_dt
 logical,save::wall_treatment=.false.
 logical,save::configured=.false.,enabled=.false.
 integer,save::closure_mode=0
 real(8),save::eddy_viscosity=0d0,water_viscosity=1d-6,manning_n=0d0,c_delta=.08d0
 real(8),save::maximum_mixing_length=huge(1d0)
 real(8),parameter::cmu=.09d0,c1e=1.44d0,c2e=1.92d0,c3e=.8d0,sk=1d0,se=1.3d0
 integer,save::sm=0,sn=0
 real(8),allocatable,save::u(:,:),v(:,:),ki(:,:),ei(:,:),ns(:,:),nt(:,:),hn(:,:), &
  qx(:,:),qy(:,:),rx(:,:),ry(:,:),ux(:,:),uy(:,:),vx(:,:),vy(:,:), &
  fxu(:,:),fxv(:,:),fyu(:,:),fyv(:,:),fxk(:,:),fyk(:,:),fxe(:,:),fye(:,:), &
  un(:,:),vn(:,:),kn(:,:),en(:,:)
!$omp threadprivate(sm,sn,u,v,ki,ei,ns,nt,hn,qx,qy,rx,ry,ux,uy,vx,vy,fxu,fxv,fyu,fyv, &
!$omp& fxk,fyk,fxe,fye,un,vn,kn,en)
contains
 subroutine fail(s)
  character(*),intent(in)::s
  write(*,'(a)')'BRAIDS momentum diffusion: '//trim(s); error stop 1
 end subroutine
 subroutine configure_braids_momentum_diffusion()
  integer::iu,ios,version; logical::exists
!$omp flush(configured)
  if(configured)return
!$omp critical(braids_momentum_diffusion_configuration)
!$omp flush(configured)
  if(.not.configured)then
   inquire(file='braids_momentum_diffusion.data',exist=exists)
   if(exists)then
    open(newunit=iu,file='braids_momentum_diffusion.data',status='old',action='read',iostat=ios)
    if(ios/=0)call fail('cannot open configuration')
    read(iu,*,iostat=ios)version
    if(ios/=0.or.(version<1.or.version>3))call fail('configuration version must be 1, 2 or 3')
    if(version==1)then
     read(iu,*,iostat=ios)eddy_viscosity
     if(ios/=0.or..not.ieee_is_finite(eddy_viscosity).or.eddy_viscosity<0d0) &
      call fail('eddy viscosity must be finite and non-negative')
     closure_mode=1; enabled=eddy_viscosity>0d0
    else
     read(iu,*,iostat=ios)water_viscosity
     if(ios==0)read(iu,*,iostat=ios)manning_n
     if(ios==0)read(iu,*,iostat=ios)c_delta
     if(ios==0)read(iu,*,iostat=ios)maximum_mixing_length
     if(ios/=0.or..not.all(ieee_is_finite([water_viscosity,manning_n,c_delta,maximum_mixing_length]))) &
      call fail('invalid two-length-scale configuration')
     if(water_viscosity<0d0.or.manning_n<=0d0.or.c_delta<0d0.or.maximum_mixing_length<=0d0) &
      call fail('invalid coefficient range')
     closure_mode=2; enabled=.true.; wall_treatment=version==3
     if(wall_treatment.and.water_viscosity<=0d0)call fail('smooth wall law requires positive water viscosity')
    endif
    close(iu)
   endif
   configured=.true.
!$omp flush(configured)
  endif
!$omp end critical(braids_momentum_diffusion_configuration)
 end subroutine
 logical function braids_diffusion_enabled() result(active)
  call configure_braids_momentum_diffusion()
  active=enabled
 end function
 real(8) function braids_diffusion_dt(nvar,m,n,mbc,dx,dy,q,link) result(limit)
  integer,intent(in)::nvar,m,n,mbc
  real(8),intent(in)::dx,dy,q(nvar,m,n)
  logical,intent(in)::link(4)
  real(8)::c
  call configure_braids_momentum_diffusion()
  limit=huge(1d0);if(.not.enabled)return
  if(closure_mode==2.and.nvar<7)call fail('two-length-scale mode requires seven states')
  if(coordinate_system/=1.or.mxnest/=1)call fail('first release requires Cartesian fixed level 1')
  call scratch(m,n);call fields(nvar,m,n,mbc,dx,dy,q,link)
  c=eddy_viscosity
  if(closure_mode==2)c=maxval(ns(mbc+1:m-mbc,mbc+1:n-mbc)+2d0*nt(mbc+1:m-mbc,mbc+1:n-mbc))
  if(c>0d0)limit=.45d0/(2d0*c*(1d0/dx**2+1d0/dy**2))
 end function
 logical function braids_wall_turbulence_enabled() result(active)
  call configure_braids_momentum_diffusion()
  active=enabled.and.wall_treatment
 end function
 pure subroutine braids_smooth_wall_state(h,velocity,distance,nu,ustar,k,e,yplus)
  ! Smooth-wall u+=ln(E*y+)/kappa, with its continuous viscous branch.
  ! Erpicum et al. (2009), equations 17-18 give the integrated k and e.
  ! E=9.8 and kappa=.41 are explicit modeling choices, not fitted to C-C.
  real(8),intent(in)::h,velocity,distance,nu
  real(8),intent(out)::ustar,k,e,yplus
  real(8),parameter::kappa=.41d0,elog=9.8d0,ylam=11.530107402304532d0
  real(8)::re,old,yp
  integer::it
  ustar=0d0;k=0d0;e=0d0;yplus=0d0
  if(min(h,distance,nu)<=0d0.or.velocity==0d0)return
  re=abs(velocity)*distance/nu
  yp=sqrt(re)
  if(yp>ylam)then
   do it=1,30
    old=yp;yp=(kappa*re+yp)/(1d0+log(elog*yp))
    if(abs(yp-old)<=1d-12*max(1d0,yp))exit
   enddo
  endif
  yplus=yp;ustar=yp*nu/distance
  if(yp>ylam)then
   k=h*ustar**2/sqrt(cmu);e=h**2*ustar**3/(kappa*distance)
  endif
  ! Below the log layer use molecular wall drag and zero turbulence targets.
 end subroutine
 subroutine wall_fluxes(nvar,m,n,mbc,dt,dx,dy,q,wx,wy,link)
  integer,intent(in)::nvar,m,n,mbc,wx(m+1,n),wy(m,n+1)
  logical,intent(in)::link(4)
  real(8),intent(in)::dt,dx,dy,q(nvar,m,n)
  integer::i,j,c,il,ir,jb,jt,sgn
  real(8)::h,ut,kw,ew,yp,drag,a,b
  il=mbc+1;ir=m-mbc;jb=mbc+1;jt=n-mbc
  if(link(1))il=il-1
  if(link(2))ir=ir+1
  if(link(3))jb=jb-1
  if(link(4))jt=jt+1
  do j=jb,jt;do i=il,ir+1
   sgn=wx(i,j);if(sgn==0)cycle
   c=i;if(sgn==1)c=i-1
   if(c<il.or.c>ir)cycle
   h=q(1,c,j);if(h<=dry_tolerance)cycle
   call braids_smooth_wall_state(h,v(c,j),.5d0*dx,water_viscosity,ut,kw,ew,yp)
   ! Limit the combined drag from two opposing walls to avoid reversal.
   drag=h*ut**2
   if(abs(v(c,j))>0d0)drag=drag/(1d0+2d0*dt*ut**2/(dx*abs(v(c,j))))
   fxu(i,j)=ki(c,j)
   fxv(i,j)=dble(sgn)*sign(drag,v(c,j))
   a=ns(c,j);b=nt(c,j)
   fxk(i,j)=dble(sgn)*(a+b/sk)*(ki(c,j)-kw)/(.5d0*dx)
   fxe(i,j)=dble(sgn)*(a+b/se)*(ei(c,j)-ew)/(.5d0*dx)
  enddo;enddo
  do i=il,ir;do j=jb,jt+1
   sgn=wy(i,j);if(sgn==0)cycle
   c=j;if(sgn==1)c=j-1
   if(c<jb.or.c>jt)cycle
   h=q(1,i,c);if(h<=dry_tolerance)cycle
   call braids_smooth_wall_state(h,u(i,c),.5d0*dy,water_viscosity,ut,kw,ew,yp)
   drag=h*ut**2
   if(abs(u(i,c))>0d0)drag=drag/(1d0+2d0*dt*ut**2/(dy*abs(u(i,c))))
   fyu(i,j)=dble(sgn)*sign(drag,u(i,c))
   fyv(i,j)=ki(i,c)
   a=ns(i,c);b=nt(i,c)
   fyk(i,j)=dble(sgn)*(a+b/sk)*(ki(i,c)-kw)/(.5d0*dy)
   fye(i,j)=dble(sgn)*(a+b/se)*(ei(i,c)-ew)/(.5d0*dy)
  enddo;enddo
 end subroutine
 pure real(8) function harm(a,b)result(c)
  real(8),intent(in)::a,b
  if(a>0d0.and.b>0d0)then;c=2d0*a*b/(a+b);else;c=0d0;endif
 end function
 subroutine scratch(m,n)
  integer,intent(in)::m,n
  if(allocated(u))then
   if(sm==m.and.sn==n)return
   deallocate(u,v,ki,ei,ns,nt,hn,qx,qy,rx,ry,ux,uy,vx,vy,fxu,fxv,fyu,fyv, &
    fxk,fyk,fxe,fye,un,vn,kn,en)
  endif
  allocate(u(m,n),v(m,n),ki(m,n),ei(m,n),ns(m,n),nt(m,n),hn(m,n), &
   qx(m,n),qy(m,n),rx(m,n),ry(m,n),ux(m,n),uy(m,n),vx(m,n),vy(m,n), &
   fxu(m+1,n),fxv(m+1,n),fyu(m,n+1),fyv(m,n+1),fxk(m+1,n),fyk(m,n+1), &
   fxe(m+1,n),fye(m,n+1),un(m,n),vn(m,n),kn(m,n),en(m,n))
  sm=m;sn=n
 end subroutine
 subroutine fields(nvar,m,n,mbc,dx,dy,q,link)
  integer,intent(in)::nvar,m,n,mbc; real(8),intent(in)::dx,dy,q(nvar,m,n);logical,intent(in)::link(4)
  integer::i,j,il,ir,jb,jt,ia,iz,ja,jz;real(8)::speed,ustar
  il=mbc+1;ir=m-mbc;jb=mbc+1;jt=n-mbc
  u=0d0;v=0d0;ki=0d0;ei=0d0;ns=0d0;nt=0d0;hn=0d0
  do j=1,n;do i=1,m
   if(q(1,i,j)>dry_tolerance)then
    u(i,j)=q(2,i,j)/q(1,i,j);v(i,j)=q(3,i,j)/q(1,i,j)
    if(closure_mode==1)then
     hn(i,j)=q(1,i,j)*eddy_viscosity
    else
     ki(i,j)=max(q(6,i,j),0d0);ei(i,j)=max(q(7,i,j),0d0)
     speed=hypot(u(i,j),v(i,j));ustar=sqrt(grav)*manning_n*speed/q(1,i,j)**(1d0/6d0)
     ns(i,j)=water_viscosity+c_delta*q(1,i,j)*ustar
     if(ki(i,j)>0d0.and.ei(i,j)>tiny(1d0))then
      nt(i,j)=cmu*ki(i,j)**2/ei(i,j)
      nt(i,j)=min(nt(i,j),cmu**.25d0*sqrt(ki(i,j)/q(1,i,j))*maximum_mixing_length)
     endif
    endif
   endif
  enddo;enddo
  if(.not.link(1))then
   u(il-1,:)=u(il,:);v(il-1,:)=v(il,:);ki(il-1,:)=ki(il,:);ei(il-1,:)=ei(il,:); &
    ns(il-1,:)=ns(il,:);nt(il-1,:)=nt(il,:);hn(il-1,:)=hn(il,:)
  endif
  if(.not.link(2))then
   u(ir+1,:)=u(ir,:);v(ir+1,:)=v(ir,:);ki(ir+1,:)=ki(ir,:);ei(ir+1,:)=ei(ir,:); &
    ns(ir+1,:)=ns(ir,:);nt(ir+1,:)=nt(ir,:);hn(ir+1,:)=hn(ir,:)
  endif
  if(.not.link(3))then
   u(:,jb-1)=u(:,jb);v(:,jb-1)=v(:,jb);ki(:,jb-1)=ki(:,jb);ei(:,jb-1)=ei(:,jb); &
    ns(:,jb-1)=ns(:,jb);nt(:,jb-1)=nt(:,jb);hn(:,jb-1)=hn(:,jb)
  endif
  if(.not.link(4))then
   u(:,jt+1)=u(:,jt);v(:,jt+1)=v(:,jt);ki(:,jt+1)=ki(:,jt);ei(:,jt+1)=ei(:,jt); &
    ns(:,jt+1)=ns(:,jt);nt(:,jt+1)=nt(:,jt);hn(:,jt+1)=hn(:,jt)
  endif
  if(closure_mode==2)then
   qx=0d0;qy=0d0;rx=0d0;ry=0d0;ux=0d0;uy=0d0;vx=0d0;vy=0d0
   if(mbc<2)call fail('turbulence stresses require two ghost layers')
   ia=il;iz=ir;ja=jb;jz=jt
   if(link(1))ia=ia-1
   if(link(2))iz=iz+1
   if(link(3))ja=ja-1
   if(link(4))jz=jz+1
   do j=ja,jz;do i=ia,iz
    qx(i,j)=(q(2,i+1,j)-q(2,i-1,j))/(2d0*dx);qy(i,j)=(q(2,i,j+1)-q(2,i,j-1))/(2d0*dy)
    rx(i,j)=(q(3,i+1,j)-q(3,i-1,j))/(2d0*dx);ry(i,j)=(q(3,i,j+1)-q(3,i,j-1))/(2d0*dy)
    ux(i,j)=(u(i+1,j)-u(i-1,j))/(2d0*dx);uy(i,j)=(u(i,j+1)-u(i,j-1))/(2d0*dy)
    vx(i,j)=(v(i+1,j)-v(i-1,j))/(2d0*dx);vy(i,j)=(v(i,j+1)-v(i,j-1))/(2d0*dy)
   enddo;enddo
  endif
 end subroutine
 subroutine constant_step(nvar,m,n,mbc,dt,dx,dy,q,link)
  integer,intent(in)::nvar,m,n,mbc;real(8),intent(in)::dt,dx,dy;real(8),intent(inout)::q(nvar,m,n)
  logical,intent(in)::link(4);integer::i,j,il,ir,jb,jt;real(8)::a
  il=mbc+1;ir=m-mbc;jb=mbc+1;jt=n-mbc;fxu=0d0;fxv=0d0;fyu=0d0;fyv=0d0
  do j=jb,jt;do i=il,ir+1
   if((i==il.and..not.link(1)).or.(i==ir+1.and..not.link(2)))cycle
   a=harm(hn(i-1,j),hn(i,j));fxu(i,j)=-a*(u(i,j)-u(i-1,j))/dx;fxv(i,j)=-a*(v(i,j)-v(i-1,j))/dx
  enddo;enddo
  do i=il,ir;do j=jb,jt+1
   if((j==jb.and..not.link(3)).or.(j==jt+1.and..not.link(4)))cycle
   a=harm(hn(i,j-1),hn(i,j));fyu(i,j)=-a*(u(i,j)-u(i,j-1))/dy;fyv(i,j)=-a*(v(i,j)-v(i,j-1))/dy
  enddo;enddo
  do j=jb,jt;do i=il,ir
   q(2,i,j)=q(2,i,j)-dt/dx*(fxu(i+1,j)-fxu(i,j))-dt/dy*(fyu(i,j+1)-fyu(i,j))
   q(3,i,j)=q(3,i,j)-dt/dx*(fxv(i+1,j)-fxv(i,j))-dt/dy*(fyv(i,j+1)-fyv(i,j))
  enddo;enddo
 end subroutine
 subroutine two_step(nvar,m,n,mbc,dt,dx,dy,q,link,wx,wy)
  integer,intent(in)::nvar,m,n,mbc;real(8),intent(in)::dt,dx,dy;real(8),intent(inout)::q(nvar,m,n)
  integer,intent(in),optional::wx(m+1,n),wy(m,n+1)
  logical,intent(in)::link(4);integer::i,j,il,ir,jb,jt,ia,iz,ja,jz;real(8)::a,b,hf,kf,g1,g2,speed,p,f,shape,div,r,loss,k0,e0,ktiny
  il=mbc+1;ir=m-mbc;jb=mbc+1;jt=n-mbc
  ia=il;iz=ir;ja=jb;jz=jt
  if(link(1))ia=ia-1
  if(link(2))iz=iz+1
  if(link(3))ja=ja-1
  if(link(4))jz=jz+1
  fxu=0d0;fxv=0d0;fyu=0d0;fyv=0d0;fxk=0d0;fyk=0d0;fxe=0d0;fye=0d0
  do j=ja,jz;do i=ia,iz+1
   if((i==il.and..not.link(1)).or.(i==ir+1.and..not.link(2)))cycle
   if(q(1,i-1,j)<=dry_tolerance.or.q(1,i,j)<=dry_tolerance)cycle
   a=harm(ns(i-1,j),ns(i,j));b=harm(nt(i-1,j),nt(i,j));hf=.5d0*(q(1,i-1,j)+q(1,i,j));kf=.5d0*(ki(i-1,j)+ki(i,j))
   g1=(q(2,i,j)-q(2,i-1,j))/dx;g2=(q(3,i,j)-q(3,i-1,j))/dx
   fxu(i,j)=-(a*g1+b*(g1-.5d0*(ry(i-1,j)+ry(i,j)))-kf)
   fxv(i,j)=-(a*g2+b*(g2+.5d0*(qy(i-1,j)+qy(i,j))))
   fxk(i,j)=-(a/hf*(ki(i,j)*q(1,i,j)-ki(i-1,j)*q(1,i-1,j))/dx+b/sk*(ki(i,j)-ki(i-1,j))/dx)
   fxe(i,j)=-(a/hf*(ei(i,j)*q(1,i,j)-ei(i-1,j)*q(1,i-1,j))/dx+b/se*(ei(i,j)-ei(i-1,j))/dx)
  enddo;enddo
  do i=ia,iz;do j=ja,jz+1
   if((j==jb.and..not.link(3)).or.(j==jt+1.and..not.link(4)))cycle
   if(q(1,i,j-1)<=dry_tolerance.or.q(1,i,j)<=dry_tolerance)cycle
   a=harm(ns(i,j-1),ns(i,j));b=harm(nt(i,j-1),nt(i,j));hf=.5d0*(q(1,i,j-1)+q(1,i,j));kf=.5d0*(ki(i,j-1)+ki(i,j))
   g1=(q(2,i,j)-q(2,i,j-1))/dy;g2=(q(3,i,j)-q(3,i,j-1))/dy
   fyu(i,j)=-(a*g1+b*(g1+.5d0*(rx(i,j-1)+rx(i,j))))
   fyv(i,j)=-(a*g2+b*(g2-.5d0*(qx(i,j-1)+qx(i,j)))-kf)
   fyk(i,j)=-(a/hf*(ki(i,j)*q(1,i,j)-ki(i,j-1)*q(1,i,j-1))/dy+b/sk*(ki(i,j)-ki(i,j-1))/dy)
   fye(i,j)=-(a/hf*(ei(i,j)*q(1,i,j)-ei(i,j-1)*q(1,i,j-1))/dy+b/se*(ei(i,j)-ei(i,j-1))/dy)
  enddo;enddo
  if(wall_treatment)then
   if(.not.present(wx).or..not.present(wy))call fail('wall mode requires explicit solid face masks')
   call wall_fluxes(nvar,m,n,mbc,dt,dx,dy,q,wx,wy,link)
  endif
  ! Limit only outgoing diffusive transfers. Each face retains one flux for
  ! both neighbors, so positivity does not sacrifice scalar conservation.
  un=0d0;vn=0d0;kn=1d0;en=1d0
  do j=ja,jz;do i=ia,iz+1
   if(fxk(i,j)>0d0.and.i>ia)un(i-1,j)=un(i-1,j)+fxk(i,j)/dx
   if(fxk(i,j)<0d0.and.i<=iz)un(i,j)=un(i,j)-fxk(i,j)/dx
   if(fxe(i,j)>0d0.and.i>ia)vn(i-1,j)=vn(i-1,j)+fxe(i,j)/dx
   if(fxe(i,j)<0d0.and.i<=iz)vn(i,j)=vn(i,j)-fxe(i,j)/dx
  enddo;enddo
  do i=ia,iz;do j=ja,jz+1
   if(fyk(i,j)>0d0.and.j>ja)un(i,j-1)=un(i,j-1)+fyk(i,j)/dy
   if(fyk(i,j)<0d0.and.j<=jz)un(i,j)=un(i,j)-fyk(i,j)/dy
   if(fye(i,j)>0d0.and.j>ja)vn(i,j-1)=vn(i,j-1)+fye(i,j)/dy
   if(fye(i,j)<0d0.and.j<=jz)vn(i,j)=vn(i,j)-fye(i,j)/dy
  enddo;enddo
  do j=ja,jz;do i=ia,iz
   if(un(i,j)>0d0)kn(i,j)=min(1d0,max(ki(i,j),0d0)/(dt*un(i,j)))
   if(vn(i,j)>0d0)en(i,j)=min(1d0,max(ei(i,j),0d0)/(dt*vn(i,j)))
  enddo;enddo
  do j=ja,jz;do i=ia,iz+1
   if(fxk(i,j)>0d0.and.i>ia)then;if(q(1,i-1,j)>dry_tolerance)fxk(i,j)=fxk(i,j)*kn(i-1,j)
   elseif(fxk(i,j)<0d0.and.i<=iz)then;if(q(1,i,j)>dry_tolerance)fxk(i,j)=fxk(i,j)*kn(i,j);endif
   if(fxe(i,j)>0d0.and.i>ia)then;if(q(1,i-1,j)>dry_tolerance)fxe(i,j)=fxe(i,j)*en(i-1,j)
   elseif(fxe(i,j)<0d0.and.i<=iz)then;if(q(1,i,j)>dry_tolerance)fxe(i,j)=fxe(i,j)*en(i,j);endif
  enddo;enddo
  do i=ia,iz;do j=ja,jz+1
   if(fyk(i,j)>0d0.and.j>ja)then;if(q(1,i,j-1)>dry_tolerance)fyk(i,j)=fyk(i,j)*kn(i,j-1)
   elseif(fyk(i,j)<0d0.and.j<=jz)then;if(q(1,i,j)>dry_tolerance)fyk(i,j)=fyk(i,j)*kn(i,j);endif
   if(fye(i,j)>0d0.and.j>ja)then;if(q(1,i,j-1)>dry_tolerance)fye(i,j)=fye(i,j)*en(i,j-1)
   elseif(fye(i,j)<0d0.and.j<=jz)then;if(q(1,i,j)>dry_tolerance)fye(i,j)=fye(i,j)*en(i,j);endif
  enddo;enddo
  un=0d0;vn=0d0;kn=0d0;en=0d0
  do j=jb,jt;do i=il,ir
   ! Solid faces exchange stress/turbulence with the wall, not its dry cell.
   if(q(1,i,j)<=dry_tolerance)cycle
   un(i,j)=q(2,i,j)-dt/dx*(fxu(i+1,j)-fxu(i,j))-dt/dy*(fyu(i,j+1)-fyu(i,j))
   vn(i,j)=q(3,i,j)-dt/dx*(fxv(i+1,j)-fxv(i,j))-dt/dy*(fyv(i,j+1)-fyv(i,j))
   k0=max(q(6,i,j),0d0);e0=max(q(7,i,j),0d0)
   kn(i,j)=k0-dt/dx*(fxk(i+1,j)-fxk(i,j))-dt/dy*(fyk(i,j+1)-fyk(i,j))
   en(i,j)=e0-dt/dx*(fxe(i+1,j)-fxe(i,j))-dt/dy*(fye(i,j+1)-fye(i,j))
   if(kn(i,j)<-1d-12*max(1d0,k0).or.en(i,j)<-1d-12*max(1d0,e0))call fail('limited turbulence diffusion violated positivity')
   kn(i,j)=max(kn(i,j),0d0);en(i,j)=max(en(i,j),0d0)
   if(q(1,i,j)>dry_tolerance)then
    speed=hypot(u(i,j),v(i,j));p=max(nt(i,j)*((qx(i,j)-ry(i,j))*(ux(i,j)-vy(i,j))+(qy(i,j)+rx(i,j))*(uy(i,j)+vx(i,j))),0d0)
    shape=0d0
    if(speed>tiny(1d0))shape=((qx(i,j)-ry(i,j))*(u(i,j)**2-v(i,j)**2)+2d0*(qy(i,j)+rx(i,j))*u(i,j)*v(i,j))/speed
    f=max(grav*manning_n**2/q(1,i,j)**(4d0/3d0)*(3d0*kn(i,j)*speed-nt(i,j)*shape),0d0)
    div=ux(i,j)+vy(i,j);ktiny=max(tiny(1d0),epsilon(1d0)*q(1,i,j)*max(speed**2,grav*q(1,i,j)))
    r=-kn(i,j)*div+p-f;loss=en(i,j)/q(1,i,j)
    if(r>=0d0)then;kn(i,j)=kn(i,j)+dt*r
    elseif(kn(i,j)>0d0)then;kn(i,j)=kn(i,j)/(1d0+dt*(-r)/max(kn(i,j),ktiny));endif
    if(kn(i,j)>0d0)kn(i,j)=kn(i,j)/(1d0+dt*loss/max(kn(i,j),ktiny))
    if(kn(i,j)>ktiny.and.en(i,j)>0d0)then
     r=c1e*en(i,j)/kn(i,j)*(p-(1d0-c3e)*f);loss=c2e*en(i,j)/(q(1,i,j)*kn(i,j))
     if(r>=0d0)then;en(i,j)=en(i,j)+dt*r;else;en(i,j)=en(i,j)/(1d0+dt*(-r)/max(en(i,j),tiny(1d0)));endif
     en(i,j)=en(i,j)/(1d0+dt*loss)
    else;en(i,j)=0d0;endif
   else;un(i,j)=0d0;vn(i,j)=0d0;kn(i,j)=0d0;en(i,j)=0d0;endif
   if(.not.all(ieee_is_finite([un(i,j),vn(i,j),kn(i,j),en(i,j)])))call fail('non-finite turbulence update')
  enddo;enddo
  do j=jb,jt;do i=il,ir;q(2,i,j)=un(i,j);q(3,i,j)=vn(i,j);q(6,i,j)=kn(i,j);q(7,i,j)=en(i,j);enddo;enddo
 end subroutine
 subroutine braids_momentum_diffusion(nvar,m,n,mbc,dt,dx,dy,q,connected,wx,wy)
  integer,intent(in)::nvar,m,n,mbc;real(8),intent(in)::dt,dx,dy;real(8),intent(inout)::q(nvar,m,n)
  integer,intent(in),optional::wx(m+1,n),wy(m,n+1)
  logical,intent(in),optional::connected(4);logical::link(4);integer::i,j,l,il,ir,jb,jt,nsub;real(8)::c,maxc,dts
  call configure_braids_momentum_diffusion();if(.not.enabled)return
  if(nvar<3.or.(closure_mode==2.and.nvar<7))call fail('two-length-scale mode requires seven states')
  if(coordinate_system/=1.or.mxnest/=1)call fail('first release requires Cartesian fixed level 1')
  if(.not.all(ieee_is_finite([dt,dx,dy])).or.min(dt,dx,dy)<=0d0)call fail('invalid step or cell dimensions')
  il=mbc+1;ir=m-mbc;jb=mbc+1;jt=n-mbc;link=.false.;if(present(connected))link=connected
  call scratch(m,n);call fields(nvar,m,n,mbc,dx,dy,q,link);maxc=0d0
  do j=jb,jt;do i=il,ir
   if(closure_mode==1)then;c=eddy_viscosity;else;c=ns(i,j)+2d0*nt(i,j);endif
   maxc=max(maxc,dt*2d0*c*(1d0/dx**2+1d0/dy**2))
  enddo;enddo
  nsub=max(1,ceiling(maxc/.45d0*(1d0-16d0*epsilon(1d0))))
  if(nsub>1.and.any(link)) call fail('multi-patch turbulence needs a smaller step')
  dts=dt/dble(nsub)
  do l=1,nsub
   call fields(nvar,m,n,mbc,dx,dy,q,link)
   if(closure_mode==1)then;call constant_step(nvar,m,n,mbc,dts,dx,dy,q,link)
   else;call two_step(nvar,m,n,mbc,dts,dx,dy,q,link,wx,wy);endif
  enddo
 end subroutine
 pure subroutine inlet_state(h,hu,hv,k,e)
  real(8),intent(in)::h,hu,hv;real(8),intent(out)::k,e;real(8)::speed
  k=0d0;e=0d0;if(h<=0d0)return;speed=hypot(hu,hv)/h;k=1d-4*h*speed**2;e=10d0*k**1.5d0/sqrt(h)
 end subroutine
 subroutine accepted_flux(update,physical,flux)
  real(8),intent(in)::update,physical;real(8),intent(out)::flux
  if(use_fwave_positivity_limiter)then;flux=update;else;flux=update+physical;endif
  if(.not.ieee_is_finite(flux))call fail('non-finite accepted water flux')
 end subroutine
 subroutine conservative_flux(lu,ru,lp,rp,flux)
  real(8),intent(in)::lu,ru,lp,rp;real(8),intent(out)::flux;real(8)::other,tol
  if(use_fwave_positivity_limiter)then;flux=lu;other=ru
  else;flux=lu+lp;other=ru+rp;endif
  tol=2048d0*epsilon(1d0)*max(1d0,abs(flux),abs(other))
  if(abs(flux-other)>tol)call fail('water flux mismatch')
  if(.not.ieee_is_finite(flux))call fail('non-finite accepted water flux')
 end subroutine
 subroutine braids_turbulence_fluxes(nvar,m,n,mbc,q,fm,fp,gm,gp,connected)
  integer,intent(in)::nvar,m,n,mbc;real(8),intent(in)::q(nvar,m,n)
  real(8),intent(inout)::fm(nvar,m,n),fp(nvar,m,n),gm(nvar,m,n),gp(nvar,m,n)
  logical,intent(in),optional::connected(4)
  logical::link(4)
  integer::i,j,il,ir,jb,jt,up;real(8)::wf,k,e,pk,pe,tol
  link=[xperdom,xperdom,yperdom,yperdom];if(present(connected))link=connected
  call configure_braids_momentum_diffusion();if(.not.enabled.or.closure_mode/=2)return
  if(nvar<7)call fail('turbulence transport requires seven states')
  il=mbc+1;ir=m-mbc;jb=mbc+1;jt=n-mbc
  do j=jb,jt;do i=il,ir+1
   if(i==il.and..not.link(1))then;call accepted_flux(fp(1,i,j),q(2,i,j),wf)
   elseif(i==ir+1.and..not.link(2))then;call accepted_flux(fm(1,i,j),q(2,i-1,j),wf)
   else;call conservative_flux(fm(1,i,j),fp(1,i,j),q(2,i-1,j),q(2,i,j),wf);endif
   up=i-1;if(wf<0d0)up=i;k=max(q(6,up,j),0d0);e=max(q(7,up,j),0d0)
   if((i==il.and..not.link(1).and.wf>0d0).or.(i==ir+1.and..not.link(2).and.wf<0d0)) &
    call inlet_state(q(1,up,j),q(2,up,j),q(3,up,j),k,e)
   pk=0d0;pe=0d0;if(q(1,up,j)>dry_tolerance)then;pk=k/q(1,up,j);pe=e/q(1,up,j);endif
   fm(6,i,j)=wf*pk;fp(6,i,j)=fm(6,i,j);fm(7,i,j)=wf*pe;fp(7,i,j)=fm(7,i,j)
  enddo;enddo
  do i=il,ir;do j=jb,jt+1
   if(j==jb.and..not.link(3))then;call accepted_flux(gp(1,i,j),q(3,i,j),wf)
   elseif(j==jt+1.and..not.link(4))then;call accepted_flux(gm(1,i,j),q(3,i,j-1),wf)
   else;call conservative_flux(gm(1,i,j),gp(1,i,j),q(3,i,j-1),q(3,i,j),wf);endif
   up=j-1;if(wf<0d0)up=j;k=max(q(6,i,up),0d0);e=max(q(7,i,up),0d0)
   if((j==jb.and..not.link(3).and.wf>0d0).or.(j==jt+1.and..not.link(4).and.wf<0d0)) &
    call inlet_state(q(1,i,up),q(2,i,up),q(3,i,up),k,e)
   pk=0d0;pe=0d0;if(q(1,i,up)>dry_tolerance)then;pk=k/q(1,i,up);pe=e/q(1,i,up);endif
   gm(6,i,j)=wf*pk;gp(6,i,j)=gm(6,i,j);gm(7,i,j)=wf*pe;gp(7,i,j)=gm(7,i,j)
  enddo;enddo
 end subroutine
end module
