! Pure OpenMP coarse/fine density-face augmentation. Projection stays v97.
module braids_density_trace_projection_v98_module
 use iso_c_binding
 use omp_lib
 use, intrinsic :: ieee_arithmetic
 implicit none
contains
 subroutine braids_density_trace_faces_v98(nc,nf,nt,owner,lookup,depth,density,distance, &
  base_acceleration,base_mask,wet,trace_depth,trace_density,trace_wet,rho_ref,gravity, &
  threads,acceleration,mask,status) bind(C)
  integer(c_int),value :: nc,nf,nt,threads
  integer(c_int),intent(in) :: owner(nf),lookup(nf),base_mask(nf),wet(nc),trace_wet(nt)
  real(c_double),intent(in) :: depth(nc),density(nc),distance(nf),base_acceleration(nf), &
   trace_depth(nt),trace_density(nt)
  real(c_double),value :: rho_ref,gravity
  real(c_double),intent(out) :: acceleration(nf)
  integer(c_int),intent(out) :: mask(nf),status(nf)
  integer :: f,c,t
  !$omp parallel do if(.not.omp_in_parallel()) num_threads(threads) schedule(static) &
  !$omp& default(shared) private(f,c,t)
  do f=1,nf
   acceleration(f)=base_acceleration(f);mask(f)=base_mask(f);status(f)=0
   t=lookup(f)+1
   if(t==0) cycle
   c=owner(f)+1
   acceleration(f)=0.d0;mask(f)=0
   if(wet(c)==0.or.trace_wet(t)==0) cycle
   acceleration(f)=-gravity*(trace_density(t)-density(c))* &
    (trace_depth(t)+depth(c))/(4.d0*rho_ref*distance(f))
   mask(f)=1
   if(.not.ieee_is_finite(acceleration(f))) status(f)=1
  enddo
  !$omp end parallel do
 end subroutine
end module
