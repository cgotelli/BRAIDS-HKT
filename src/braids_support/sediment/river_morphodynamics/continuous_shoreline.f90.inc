 subroutine prepare_wet_dry_traces(state,v,grad,vlo,vhi)
  ! Research continuous reconstruction envelope, not a new Riemann solver.
  ! Both candidate and constant-surface depths must be positive for blending.
  ! Away from shore, 1-phi=O(dx), hence its face-value perturbation is O(dx^2).
  real(8),intent(in)::state(nstate,nc),v(nprimitive,nc),grad(nprimitive,2,nc),vlo(nprimitive,nc),vhi(nprimitive,nc)
  integer::c,k,f
  real(8)::h,eta,zb,phi
  if(allocated(shore_blend)) then
   if(size(shore_blend)/=nc) deallocate(shore_blend)
  endif
  if(.not.allocated(shore_blend)) allocate(shore_blend(nc))
  shore_blend=1.d0;hydro_constant=.false.
  if(scheme==1) return
  !$omp parallel do schedule(static) default(shared) private(c,k,f,h,eta,zb,phi)
  do c=1,nc
   h=state(1,c);phi=0.d0
   if(h>dry_tolerance) then
    phi=1.d0
    do k=offset(c),offset(c+1)-1
     f=faces(k);call face_elevations(c,f,v,grad,vlo,vhi,eta,zb)
     phi=min(phi,max(0.d0,min(eta-zb,v(1,c)-zb)/h))
     if(any(rectangle_width(:,c)<=0.d0).and.abs(zb-v(bed_primitive_index(),c))>=h) phi=0.d0
    enddo
   endif
   shore_blend(c)=phi;hydro_constant(c)=phi<=0.d0
  enddo
  !$omp end parallel do
 end subroutine
