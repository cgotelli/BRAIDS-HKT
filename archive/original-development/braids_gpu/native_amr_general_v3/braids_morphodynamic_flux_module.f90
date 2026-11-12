! Linearized coupled shallow-water/Exner f-wave splitting on one face.
module braids_morphodynamic_flux_module
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 implicit none
 private
 public :: coupled_exner_flux,mpm_mass_jacobian
contains
 pure subroutine mpm_mass_jacobian(h,u,v,n,d50,d90,relative_density,rhos,g,critical,hiding,cf, &
                                  roughness,hmin,hdry,derivative)
  real(8),intent(in) :: h,u,v,n,d50,d90,relative_density,rhos,g,critical,hiding,cf,hmin,hdry
  logical,intent(in) :: roughness
  real(8),intent(out) :: derivative(3)
  real(8) :: heff,velocity,chezy,grain,mu,alpha,dalpha,excess,k,root,capacity,weight,dweight,xi
  derivative=0
  if(h<=hdry) return
  heff=max(h,hmin);velocity=sqrt(u*u+v*v)
  if(velocity<=0) return
  if(roughness) then
   chezy=heff**(1.d0/6)/n;grain=18*log10(12*max(heff/d90,1.d0))
   mu=min((chezy/grain)**1.5d0,1.d0)
   alpha=mu/(chezy*chezy*relative_density*d50)
   if(mu>=1) then
    dalpha=-alpha/(3*heff)
   else
    dalpha=-alpha/(12*heff)
    if(heff>d90) dalpha=dalpha-1.5d0*alpha*18/(log(10.d0)*heff*grain)
   endif
  else
   alpha=cf/(g*relative_density*d50);dalpha=0
  endif
  if(h<hmin) dalpha=0
  excess=alpha*velocity**2-critical*hiding
  if(excess<=0) return
  root=sqrt(excess);k=rhos*8*sqrt(g*relative_density*d50**3)
  capacity=k*excess*root
  derivative(1)=1.5d0*k*root*velocity*u*(dalpha-2*alpha/h)
  derivative(2)=3*k*alpha*root*u*u/(h*velocity)+capacity*(1-u*u/velocity**2)/(h*velocity)
  derivative(3)=3*k*alpha*root*u*v/(h*velocity)-capacity*u*v/(h*velocity**3)
  if(h<hmin.and.hmin>hdry) then
   xi=(h-hdry)/(hmin-hdry);weight=xi*xi*(3-2*xi);dweight=6*xi*(1-xi)/(hmin-hdry)
   derivative=weight*derivative
   derivative(1)=derivative(1)+dweight*capacity*u/velocity
  endif
 end subroutine

 pure subroutine coupled_exner_flux(left,right,zleft,zright,qsleft,qsright,derivative,g, &
                                    fluxleft,fluxright,speed,valid)
  ! States are [h, h*u_normal, h*u_tangent]. Sediment flux is accelerated
  ! bed-volume transport. Derivatives use [h, h*u_normal, h*u_tangent].
  real(8),intent(in) :: left(3),right(3),zleft,zright,qsleft,qsright,derivative(3),g
  real(8),intent(out) :: fluxleft(4),fluxright(4),speed
  logical,intent(out) :: valid
  real(8) :: h,u,v,a(4,4),ident(4,4),proj(4,4),signa(4,4),pl(4),pr(4),jump(4),diss(4)
  real(8) :: b,c,d,p,q,theta,radius,roots(3),lambda,den,other(3),argument,scale
  integer :: i,k,index
  valid=.false.;fluxleft=0;fluxright=0;speed=0
  if(min(left(1),right(1),g)<=0) return
  h=.5d0*(left(1)+right(1))
  u=(left(2)/sqrt(left(1))+right(2)/sqrt(right(1)))/(sqrt(left(1))+sqrt(right(1)))
  v=(left(3)/sqrt(left(1))+right(3)/sqrt(right(1)))/(sqrt(left(1))+sqrt(right(1)))
  ! The fourth eigenvalue is the tangential contact speed u. Eliminating
  ! that contact leaves the cubic of the normal SWE-Exner subsystem.
  b=-2*u;c=u*u-g*h*(1+derivative(2));d=-g*h*(derivative(1)+v*derivative(3))
  p=c-b*b/3;q=2*b*b*b/27-b*c/3+d
  if(p>=0) return
  argument=3*q/(2*p)*sqrt(-3/p)
  if(abs(argument)>1+1.d-10) return
  theta=acos(max(-1.d0,min(1.d0,argument)))/3
  radius=2*sqrt(-p/3)
  do k=1,3
   roots(k)=radius*cos(theta-2*acos(-1.d0)*(k-1)/3)-b/3
  enddo
  speed=max(abs(u),maxval(abs(roots)))
  a=0;ident=0
  do i=1,4
   ident(i,i)=1
  enddo
  a(1,2)=1
  a(2,:)=[g*h-u*u,2*u,0.d0,g*h]
  a(3,:)=[-u*v,v,u,0.d0]
  a(4,1:3)=derivative
  ! The MPM Jacobian has one wave opposite to the mean normal flow.
  ! Only its projector is needed, avoiding divisions between coincident
  ! same-sign shear/bed eigenvalues.
  if(u>=0) then
   index=minloc(roots,dim=1);lambda=roots(index)
  else
   index=maxloc(roots,dim=1);lambda=roots(index)
  endif
  other(1:2)=pack(roots,[(k/=index,k=1,3)]);other(3)=u
  if((u>=0.and.minval(other)<-1.d-10*speed).or.(u<0.and.maxval(other)>1.d-10*speed)) return
  proj=ident;den=1
  do k=1,3
   proj=matmul(proj,a-other(k)*ident)
   den=den*(lambda-other(k))
  enddo
  scale=max(1.d-30,speed**3)
  if(abs(den)<1.d-12*scale) return
  proj=proj/den
  if(u>=0) then
   signa=ident-2*proj
  else
   signa=-ident+2*proj
  endif
  pl=[left(2),left(2)**2/left(1)+.5d0*g*left(1)**2,left(2)*left(3)/left(1),qsleft]
  pr=[right(2),right(2)**2/right(1)+.5d0*g*right(1)**2,right(2)*right(3)/right(1),qsright]
  jump=pr-pl;jump(2)=jump(2)+g*h*(zright-zleft)
  diss=matmul(signa,jump)
  fluxleft=pl+.5d0*(jump-diss)
  fluxright=pr-.5d0*(jump+diss)
  ! Water, tangential momentum and bed volume have one shared face flux.
  fluxleft(1)=.5d0*(fluxleft(1)+fluxright(1));fluxright(1)=fluxleft(1)
  fluxleft(3:4)=.5d0*(fluxleft(3:4)+fluxright(3:4));fluxright(3:4)=fluxleft(3:4)
  valid=all(ieee_is_finite(fluxleft)).and.all(ieee_is_finite(fluxright))
 end subroutine
end module
