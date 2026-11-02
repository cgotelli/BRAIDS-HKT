! Parallel corrected conserved images and existing FLOW EOS equations.
! BRAIDS implementation only: no Delft3D source is copied or compiled here.
module braids_amr_corrected_images_v98_module
 use iso_c_binding
 use omp_lib
 use, intrinsic :: ieee_arithmetic
 implicit none
contains
 subroutine braids_amr_corrected_images_v98(nc,ng,cell,threads,dry,tm,sm,eqt,eqs,rhow,eos, &
  q0,sus0,bed0,fluid0,dq,ds,db,df,z0,bedref,bulk,solid,q,sus,bed,fluid,conc,temperature, &
  salinity,z,eta,wet,dry_transfer,momentum_cleanup,fluid_cleanup,water_density,viscosity,mixture,status) bind(C)
  integer(c_int),value :: nc,ng,cell,threads,tm,sm,eos
  real(c_double),value :: dry,eqt,eqs,rhow
  real(c_double),intent(in) :: q0(nc,3),sus0(nc,ng),bed0(nc,ng),fluid0(nc,2),dq(3), &
   ds(ng),db(ng),df(2),z0(nc),bedref(nc,ng),bulk(ng),solid(ng)
  real(c_double),intent(out) :: q(nc,3),sus(nc,ng),bed(nc,ng),fluid(nc,2),conc(nc,ng), &
   temperature(nc),salinity(nc),z(nc),eta(nc),dry_transfer(nc,ng),momentum_cleanup(nc,2), &
   fluid_cleanup(nc,2),water_density(nc),viscosity(nc),mixture(nc)
  integer(c_int),intent(out) :: wet(nc),status(nc)
  integer :: c,k
  logical :: changed,active,bad
  real(c_double) :: h,s,b,t,tol,elevation,cp0,clam,clam0,cp1,t2,t3,t4,t5,sq,rhwa,asal,bsal,rho
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) &
  !$omp& private(c,k,changed,active,bad,h,s,b,t,tol,elevation,cp0,clam,clam0,cp1,t2,t3,t4,t5,sq,rhwa,asal,bsal,rho)
  do c=1,nc
   changed=(c==cell+1);bad=.false.;status(c)=0
   q(c,:)=q0(c,:);fluid(c,:)=fluid0(c,:)
   if(changed) then
    q(c,:)=q(c,:)+dq;fluid(c,:)=fluid(c,:)+df
   endif
   h=q(c,1);active=(h>dry);wet(c)=merge(1,0,active)
   if(any(.not.ieee_is_finite(q(c,:))).or.h<0.d0.or. &
      any(.not.ieee_is_finite(fluid(c,:))).or.fluid(c,2)<0.d0) bad=.true.
   momentum_cleanup(c,:)=0.d0;fluid_cleanup(c,:)=0.d0
   if(.not.active) then
    momentum_cleanup(c,:)=-q(c,2:3);q(c,2:3)=0.d0
    fluid_cleanup(c,:)=-fluid(c,:);fluid(c,:)=0.d0
   endif
   elevation=z0(c)
   do k=1,ng
    s=sus0(c,k);b=bed0(c,k)
    if(changed) then
     s=s+ds(k);b=b+db(k)
    endif
    tol=32768.d0*epsilon(1.d0)*max(1.d0,abs(s),abs(b))
    if(abs(s)<=tol) s=max(s,0.d0)
    if(abs(b)<=tol) b=max(b,0.d0)
    if(.not.ieee_is_finite(s).or..not.ieee_is_finite(b).or.s<0.d0.or.b<0.d0) bad=.true.
    dry_transfer(c,k)=0.d0
    if(.not.active) then
     dry_transfer(c,k)=s;b=b+s;s=0.d0
    endif
    sus(c,k)=s;bed(c,k)=b;conc(c,k)=0.d0
    if(active) conc(c,k)=s/h
    elevation=elevation+(b-bedref(c,k))/bulk(k)
    if(.not.ieee_is_finite(b).or..not.ieee_is_finite(conc(c,k))) bad=.true.
   enddo
   t=eqt;s=eqs
   if(active.and.tm/=0) t=fluid(c,1)/h
   if(active.and.sm/=0) s=fluid(c,2)/h
   tol=4096.d0*epsilon(1.d0)*max(1.d0,abs(s))
   if(.not.ieee_is_finite(t).or..not.ieee_is_finite(s).or.t<=-20.d0.or.s < -tol) bad=.true.
   s=max(s,0.d0);temperature(c)=t;salinity(c)=s;z(c)=elevation;eta(c)=h+elevation
   rho=rhow
   if(tm/=0.or.sm/=0) then
    if(eos==0) then
     cp0=5890.d0+38.d0*t-.3750d0*t*t;clam=1779.5d0+11.25d0*t-.0745d0*t*t
     clam0=3.8d0+.01d0*t;cp1=cp0+3.d0*s
     rho=1000.d0*cp1/(.698d0*cp1+clam-clam0*s)
    else if(eos==1) then
     t2=t*t;t3=t2*t;t4=t3*t;t5=t4*t;sq=sqrt(s)
     rhwa=999.842594d0+6.793952d-2*t-9.095290d-3*t2+1.001685d-4*t3-1.120083d-6*t4+6.536332d-9*t5
     asal=8.24493d-1-4.0899d-3*t+7.6438d-5*t2-8.2467d-7*t3+5.3875d-9*t4
     bsal=-5.72466d-3+1.0227d-4*t-1.6546d-6*t2
     rho=rhwa+(asal+bsal*sq+4.8314d-4*s)*s
    else
     t2=t*t;t3=t2*t
     rho=999.904d0+4.8292d-2*t-7.2312d-3*t2+2.9963d-5*t3+7.6427d-1*s-3.1490d-3*s*t+3.1273d-5*s*t2
    endif
   endif
   water_density(c)=rho;viscosity(c)=4.d-5/(20.d0+t);mixture(c)=rho
   do k=1,ng
    if(solid(k)<=rho) bad=.true.
    mixture(c)=mixture(c)+conc(c,k)*(1.d0-rho/solid(k))
   enddo
   if(.not.ieee_is_finite(rho).or.rho<=0.d0.or..not.ieee_is_finite(viscosity(c)).or. &
      viscosity(c)<=0.d0.or..not.ieee_is_finite(mixture(c)).or.mixture(c)<=0.d0.or. &
      .not.ieee_is_finite(elevation).or..not.ieee_is_finite(eta(c))) bad=.true.
   status(c)=merge(1,0,bad)
  enddo
  !$omp end parallel do
 end subroutine
end module
