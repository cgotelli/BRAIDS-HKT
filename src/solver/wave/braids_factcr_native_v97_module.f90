! Parallel producer adapter around existing, unchanged BRAIDS v66/v83 equations.
module braids_factcr_native_v97_module
 use iso_c_binding
 use, intrinsic :: ieee_arithmetic
 use omp_lib
 use braids_factcr_route_v83_module, only: braids_factcr_threshold_v83
 use braids_vr1993_route_v66_module, only: braids_vr1993_current_producer_v66
 implicit none
contains
 subroutine braids_factcr_native_v97(nc,ng,threads,frozen,factcr,dry,sed_threshold,hfloor, &
  q,zvel,rho,gravity,viscosity,tauadd,d50,d90,rhos,schmidt,aks,camax,hmin,bedfac,susfac, &
  ws,z0,fraction,frozen_fields,fields,rawbed,rawconc,bedcap,capacity,target,diag,counts,status) bind(C)
  integer(c_int),value :: nc,ng,threads,frozen
  real(c_double),value :: factcr,dry,sed_threshold,hfloor
  real(c_double),intent(in) :: q(nc,3),zvel(nc),rho(nc),gravity(nc),viscosity(nc),tauadd(nc)
  real(c_double),intent(in) :: d50(ng),d90(ng),rhos(ng),schmidt(ng),aks(ng),camax(ng),hmin(ng),bedfac(ng),susfac(ng)
  real(c_double),intent(in) :: ws(nc,ng),z0(nc,ng),fraction(nc,ng),frozen_fields(nc,ng,4)
  real(c_double),intent(out) :: fields(nc,ng,4),rawbed(nc,ng,2),rawconc(nc,ng),bedcap(nc,ng,2)
  real(c_double),intent(out) :: capacity(nc,ng),target(nc,ng),diag(nc,ng,6)
  integer(c_int),intent(out) :: counts(nc,ng),status
  real(8) :: h,ux,uy,speed,exchange_h,ustar,muc,ratio,caks,refheight,shear
  integer :: c,k,bad,local_status,count
  status=0;bad=0
  if(nc<1.or.ng<1.or.threads<1.or.(frozen/=0.and.frozen/=1)) then
   status=1;return
  endif
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) collapse(2) schedule(static) default(shared) private(k,c,h,ux,uy,speed,exchange_h,ustar,muc,ratio,caks,refheight,shear,local_status,count) reduction(max:bad)
  do k=1,ng
   do c=1,nc
    rawbed(c,k,:)=0.d0;rawconc(c,k)=0.d0;bedcap(c,k,:)=0.d0
    capacity(c,k)=0.d0;target(c,k)=0.d0;diag(c,k,:)=0.d0;counts(c,k)=1
    if(frozen/=0) then
     fields(c,k,:)=frozen_fields(c,k,:)
    else
     call braids_factcr_threshold_v83(-1,1,factcr,d50(k),rhos(k),rho(c),gravity(c),viscosity(c), &
       fields(c,k,1),fields(c,k,2),fields(c,k,3),fields(c,k,4),count,local_status)
     if(local_status/=0.or.count/=1) then
      bad=1;cycle
     endif
    endif
    h=q(c,1);ux=0.d0;uy=0.d0
    if(h>dry) then
     ux=q(c,2)/h;uy=q(c,3)/h
    endif
    call braids_vr1993_current_producer_v66(h,ux,uy,zvel(c),d50(k),d90(k),rhos(k),rho(c), &
      gravity(c),viscosity(c),schmidt(k),ws(c,k),z0(c,k),aks(k),camax(k),tauadd(c), &
      fields(c,k,1),fields(c,k,4),hmin(k),rawbed(c,k,1),rawbed(c,k,2),rawconc(c,k), &
      ustar,muc,ratio,caks,refheight,local_status)
    if(local_status/=0) then
     bad=1;cycle
    endif
    speed=sqrt(ux*ux+uy*uy)
    shear=0.d0
    if(h>hmin(k)) then
     ! Preserve the pre-square-root current stress diagnostic, not rho*ustar^2
     ! after the additional-shear square root. This owns no additional physics.
     shear=rho(c)*(speed*.41d0/log(1.d0+zvel(c)/z0(c,k)))**2
     if(tauadd(c)>0.d0) shear=sqrt(shear*shear+tauadd(c)*tauadd(c))
    endif
    diag(c,k,:)=[ustar,muc,ratio,caks,refheight,shear]
    bedcap(c,k,:)=fraction(c,k)*bedfac(k)*rawbed(c,k,:)
    if(h>dry.and.h>sed_threshold) then
     exchange_h=max(hfloor,h)
     capacity(c,k)=susfac(k)*rawconc(c,k)*(speed+1.d-6)*exchange_h
     target(c,k)=fraction(c,k)*susfac(k)*rawconc(c,k)
    endif
    if(.not.all(ieee_is_finite([fields(c,k,:),rawbed(c,k,:),rawconc(c,k),bedcap(c,k,:), &
       capacity(c,k),target(c,k),diag(c,k,:)]))) bad=1
   enddo
  enddo
  !$omp end parallel do
  status=bad
 end subroutine
end module
