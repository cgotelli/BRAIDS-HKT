! Pure parallel wet-active density projection. No time/state/flux owner.
module braids_density_active_projection_v97_module
 use iso_c_binding
 use omp_lib
 use, intrinsic :: ieee_arithmetic
 implicit none
contains
 subroutine braids_density_active_projection_v97(nc,nf,nadj,depth,accel,normal,length,offset, &
  faces,signs,wet,active,rank_tol,residual_tol,threads,cell_accel,rate,ranks,residual,status) bind(C)
  integer(c_int),value :: nc,nf,nadj,threads
  real(c_double),value :: rank_tol,residual_tol
  real(c_double),intent(in) :: depth(nc),accel(nf),normal(nf,2),length(nf)
  integer(c_int),intent(in) :: offset(nc+1),faces(nadj),signs(nadj),wet(nc),active(nf)
  real(c_double),intent(out) :: cell_accel(nc,2),rate(nc,2),residual(nc)
  integer(c_int),intent(out) :: ranks(nc),status(nc)
  integer :: c,slot,f,k,nactive
  real(8) :: a,b,d,bx,by,nx,ny,w,an,det,trace,gap,hi,lo,ratio,ax,ay,rx,ry,scale
  logical :: allzero
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) default(shared) &
  !$omp& private(c,slot,f,k,nactive,a,b,d,bx,by,nx,ny,w,an,det,trace,gap,hi,lo,ratio,ax,ay,rx,ry,scale,allzero)
  do c=1,nc
   cell_accel(c,:)=0.d0; rate(c,:)=0.d0; residual(c)=0.d0; ranks(c)=0; status(c)=0
   if(wet(c)==0) cycle
   a=0.d0; b=0.d0; d=0.d0; bx=0.d0; by=0.d0; nactive=0; allzero=.true.
   do slot=offset(c)+1,offset(c+1)
    f=faces(slot)+1
    if(active(f)==0) cycle
    nactive=nactive+1
    nx=signs(slot)*normal(f,1); ny=signs(slot)*normal(f,2)
    an=signs(slot)*accel(f); w=length(f)
    a=a+w*nx*nx; b=b+w*nx*ny; d=d+w*ny*ny
    bx=bx+w*nx*an; by=by+w*ny*an
    if(an/=0.d0) allzero=.false.
   enddo
   if(nactive==0) cycle
   if(allzero) then
    ranks(c)=-1; cycle ! Exact zero source: numerical rank is deliberately not evaluated.
   endif
   det=a*d-b*b; trace=a+d; gap=hypot(a-d,2.d0*b); hi=.5d0*(trace+gap)
   if(.not.ieee_is_finite(hi).or.hi<=0.d0.or..not.ieee_is_finite(det).or. &
      .not.ieee_is_finite(bx).or..not.ieee_is_finite(by)) then
    status(c)=1; cycle
   endif
   lo=det/hi; ratio=lo/hi
   if(ratio< -rank_tol.or.(ratio>.5d0*rank_tol.and.ratio<2.d0*rank_tol)) then
    status(c)=2; cycle ! Indefinite or ambiguous numerical rank: no regularisation.
   endif
   if(ratio>=2.d0*rank_tol) then
    ranks(c)=2
    ! Same arithmetic as the existing native/CUDA v85 full-rank projection.
    ax=(d*bx-b*by)/det; ay=(a*by-b*bx)/det
   else
    ranks(c)=1
    ! Spectral projector onto the identifiable principal-normal subspace.
    ! Equivalent to its Moore-Penrose inverse, with zero discarded tangent.
    ax=((a-lo)*bx+b*by)/((hi-lo)*hi)
    ay=(b*bx+(d-lo)*by)/((hi-lo)*hi)
   endif
   rx=a*ax+b*ay-bx; ry=b*ax+d*ay-by
   scale=max(tiny(1.d0),abs(bx)+abs(a*ax)+abs(b*ay),abs(by)+abs(b*ax)+abs(d*ay))
   residual(c)=max(abs(rx),abs(ry))/scale
   if(.not.ieee_is_finite(ax).or..not.ieee_is_finite(ay).or. &
      .not.ieee_is_finite(residual(c)).or.residual(c)>residual_tol) then
    status(c)=3; cycle
   endif
   cell_accel(c,1)=ax; cell_accel(c,2)=ay
   rate(c,1)=depth(c)*ax; rate(c,2)=depth(c)*ay
   if(.not.all(ieee_is_finite(rate(c,:)))) status(c)=4
  enddo
  !$omp end parallel do
 end subroutine
end module
