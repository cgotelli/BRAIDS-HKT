! Linearized coupled shallow-water/Exner f-wave splitting on one face.
module braids_morphodynamic_flux_module
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 implicit none
 private
 public :: coupled_exner_flux,mpm_mass_jacobian,gaia_mass_jacobian, &
  gaia_fixed_inlet_face_terms,gaia_fixed_inlet_trace,gaia_fixed_inlet_pressure
contains
 pure subroutine gaia_fixed_inlet_face_terms(stage,fixed_bed,owner_state,normal,roughness, &
                                             hmin,hdry,depth,tangent,conveyance,valid)
  ! TELEMAC-GAIA LIEBOR=KENT fixes the boundary-node bed while LIQBOR=KSORT
  ! leaves the bed-load flux at local equilibrium.  BRAIDS has cell-centred
  ! mobile inventory, so represent that boundary condition only at the face:
  ! retain the live section water level, but measure the exterior water column
  ! from the immutable initial face bed.  The owner cell remains fully mobile.
  real(8),intent(in) :: stage,fixed_bed,owner_state(3),normal(2),roughness,hmin,hdry
  real(8),intent(out) :: depth,tangent(2),conveyance
  logical,intent(out) :: valid
  real(8) :: velocity(2),un

  depth=0.d0;tangent=0.d0;conveyance=0.d0;valid=.false.
  if(.not.all(ieee_is_finite([stage,fixed_bed,owner_state,normal,roughness,hmin,hdry]))) return
  if(roughness<=0.d0.or.hmin<0.d0.or.hdry<0.d0) return
  depth=max(stage-fixed_bed,0.d0)
  if(owner_state(1)>hdry) then
   velocity=owner_state(2:3)/owner_state(1)
   un=dot_product(velocity,normal)
   tangent=velocity-un*normal
  endif
  if(depth>hdry) conveyance=depth**1.5d0*max(depth,hmin)**(1.d0/6.d0)/roughness
  valid=ieee_is_finite(depth).and.all(ieee_is_finite(tangent)).and.ieee_is_finite(conveyance)
 end subroutine

 pure subroutine gaia_fixed_inlet_trace(depth,tangent,normal,conveyance,desired,total, &
                                        hdry,speed_limit,trace,valid)
  ! ``total`` is sum(conveyance*face_length) over the logical inlet.  Hence
  ! sum(dot(trace_momentum,n)*length)=-desired to roundoff: outward is positive
  ! in the finite-volume convention and an imposed inlet discharge is incoming.
  real(8),intent(in) :: depth,tangent(2),normal(2),conveyance,desired,total,hdry,speed_limit
  real(8),intent(out) :: trace(3)
  logical,intent(out) :: valid
  real(8) :: ub

  trace=0.d0;valid=.false.
  if(.not.all(ieee_is_finite([depth,tangent,normal,conveyance,desired,total,hdry,speed_limit]))) return
  if(min(depth,conveyance,desired,total,hdry)<0.d0.or.speed_limit<=0.d0) return
  if(desired>0.d0.and.total<=0.d0) return
  if(depth<=hdry) then
   valid=.true.;return
  endif
  ub=0.d0
  if(desired>0.d0) ub=desired*conveyance/(total*depth)
  if(sqrt(ub*ub+sum(tangent*tangent))>speed_limit) return
  trace=[depth,depth*(-ub*normal+tangent)]
  valid=all(ieee_is_finite(trace))
 end subroutine

 pure subroutine gaia_fixed_inlet_pressure(owner_depth,face_depth,owner_bed,fixed_bed,g,correction,valid)
  ! Complete the cell-centre-to-face hydrostatic source with the *current*
  ! geometric step.  A constant initial offset would be wrong once the owner
  ! inventory moves even though the imposed face bed stays fixed.
  real(8),intent(in) :: owner_depth,face_depth,owner_bed,fixed_bed,g
  real(8),intent(out) :: correction
  logical,intent(out) :: valid
  real(8) :: dz,hgeom
  correction=0.d0;valid=.false.
  if(.not.all(ieee_is_finite([owner_depth,face_depth,owner_bed,fixed_bed,g]))) return
  if(min(owner_depth,face_depth,g)<0.d0) return
  dz=fixed_bed-owner_bed
  hgeom=max(owner_depth-dz,0.d0)
  ! Integrate only the geometric bed step submerged by the owner water column.
  ! The boundary trace keeps its independently resolved live-section depth;
  ! at equal stage hgeom=face_depth and pressure plus correction recovers the
  ! owner-column pressure exactly.  A dry upward crest saturates at that full
  ! owner pressure instead of integrating through terrain above the shoreline.
  correction=.5d0*g*(owner_depth**2-hgeom**2)
  valid=ieee_is_finite(correction)
 end subroutine

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

 pure subroutine gaia_mass_jacobian(h,u,v,n,d,rhos,rhow,g,skin_ratio,hmin,hdry,derivative)
  ! Derivative of the TELEMAC-GAIA formula-3 base mass flux normal to a
  ! face, with respect to [h,h*u_normal,h*u_tangent].  Bed-slope and
  ! secondary-current stresses are deliberately excluded: they remain a
  ! physical residual on the coupled face, just as slope transport does in
  ! the established MPM split.
  real(8),intent(in) :: h,u,v,n,d,rhos,rhow,g,skin_ratio,hmin,hdry
  real(8),intent(out) :: derivative(3)
  real(8) :: heff,velocity,ksp,aa,cf,cfskin,tetap,theta,dlogtheta
  real(8) :: prefactor,capacity,capacity_h,capacity_speed,wet,dwet,xi
  real(8) :: total,total_h,total_speed,normal_fraction

  derivative=0.d0
  if(h<=hdry.or.n<=0.d0.or.d<=0.d0.or.rhos<=rhow.or.rhow<=0.d0.or. &
     g<=0.d0.or.skin_ratio<=0.d0) return
  velocity=sqrt(u*u+v*v)
  if(velocity<=0.d0) return
  heff=h
  if(h<hmin.and.hmin>hdry) heff=hmin*(1.d0+128.d0*epsilon(1.d0))
  ksp=skin_ratio*d
  if(heff<=ksp) return
  aa=2.5d0*log(11.036d0*heff/ksp)
  if(aa<=0.d0) return
  cf=2.d0*g*n*n/heff**(1.d0/3.d0)
  cfskin=2.d0/(aa*aa)
  tetap=.5d0*rhow*cfskin*velocity*velocity/((rhos-rhow)*g*d)+1.d-6
  if(tetap<=.06d0) return
  if(tetap<.384d0) then
   theta=sqrt(2.5d0*(tetap-.06d0));dlogtheta=.5d0/(tetap-.06d0)
  else if(tetap<1.080d0) then
   theta=1.066d0*tetap**.176d0;dlogtheta=.176d0/tetap
  else
   theta=tetap;dlogtheta=1.d0/tetap
  endif
  prefactor=.1d0*rhos*sqrt((rhos/rhow-1.d0)*g*d**3)
  capacity=prefactor*theta**2.5d0/max(cf,1.d-6)
  capacity_speed=capacity*2.5d0*dlogtheta*2.d0*(tetap-1.d-6)/velocity
  capacity_h=0.d0
  if(h>=hmin.or.hmin<=hdry) then
   capacity_h=capacity*2.5d0*dlogtheta*(-5.d0*(tetap-1.d-6)/(aa*heff))
   if(cf>1.d-6) capacity_h=capacity_h+capacity/(3.d0*heff)
  endif
  wet=1.d0;dwet=0.d0
  if(h<hmin.and.hmin>hdry) then
   xi=max(0.d0,min(1.d0,(h-hdry)/(hmin-hdry)))
   wet=xi*xi*(3.d0-2.d0*xi)
   dwet=6.d0*xi*(1.d0-xi)/(hmin-hdry)
  endif
  total=wet*capacity
  total_h=wet*capacity_h+dwet*capacity
  total_speed=wet*capacity_speed
  normal_fraction=u/velocity
  derivative(1)=normal_fraction*total_h-u*total_speed/h
  derivative(2)=(total_speed*normal_fraction**2+total*v*v/velocity**3)/h
  derivative(3)=u*v*(total_speed/velocity**2-total/velocity**3)/h
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
  integer :: i,k,index,upwind_sign
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
  upwind_sign=0
  if(u>=0.d0.and.lambda>=0.d0) upwind_sign=1
  if(u<0.d0.and.lambda<=0.d0) upwind_sign=-1
  if(upwind_sign/=0) then
   ! All four eigenvalues have the mean-flow sign.  The exact matrix sign is
   ! +/-I; constructing an "opposite" projector here would flip a physical
   ! downstream wave and can also become singular when same-sign roots meet.
   signa=upwind_sign*ident
  else
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
