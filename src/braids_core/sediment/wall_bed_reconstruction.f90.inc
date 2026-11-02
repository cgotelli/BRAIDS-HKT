! One-sided bed reconstruction for fully wet physical-wall cells.
! Do not treat the absent exterior neighbour as a terrain extremum. Limit
! against neighbour secants projected to the shared face. Comparing a face
! extrapolation to a full centroid-to-centroid prediction overlimits curved
! beds. Affine terrain remains affine; wet/dry/steep guards remain unchanged.
function braids_wall_bed_factor(c,v,grad,theta) result(factor)
 integer,intent(in) :: c
 real(8),intent(in) :: v(nprimitive,nc),grad(nprimitive,2,nc),theta(nprimitive)
 real(8) :: factor,candidate,pred,delta,eps,h,r(2),eta,zb,fraction,dn
 integer :: k,f,n,bi
 logical :: wall
 bi=bed_primitive_index();factor=theta(bi);h=v(1,c)-v(bi,c)
 if(amr_enabled.or.h<=20.d0*dry_tolerance) return
 wall=.false.;candidate=1.d0
 do k=offset(c),offset(c+1)-1
  f=faces(k)
  if(length(f)<=0.d0) cycle
  n=neighbour(f)
  if(signs(k)<0) n=owner(f)
  if(n==0) then
   if(bid(f)>0) return
   wall=.true.;cycle
  endif
  if(v(1,n)-v(bi,n)<=20.d0*dry_tolerance) return
  r=bed_ls_delta(:,k)
  dn=dot_product(r,normal(:,f))
  if(abs(dn)<=epsilon(1.d0)) return
  fraction=dot_product(mid(:,f)-xy(:,c),normal(:,f))/dn
  if(fraction<=0.d0.or.fraction>=1.d0) return
  pred=fraction*(grad(bi,1,c)*r(1)+grad(bi,2,c)*r(2))
  delta=v(bi,n)-v(bi,c)
  eps=64.d0*epsilon(1.d0)*max(1.d0,abs(v(bi,c)),abs(v(bi,n)))
  if(abs(pred)>eps) then
   if(abs(delta)<=eps) then
    candidate=0.d0
   else
    candidate=min(candidate,max(0.d0,(delta+sign(eps,pred))/pred))
   endif
  endif
 enddo
 if(.not.wall) return
 ! Preserve the existing shoreline/steep-bed envelope before extending the
 ! reconstruction. Surface and velocity limiters are unchanged.
 do k=offset(c),offset(c+1)-1
  f=faces(k);r=mid(:,f)-xy(:,c)
  eta=v(1,c)+theta(1)*(grad(1,1,c)*r(1)+grad(1,2,c)*r(2))
  zb=v(bi,c)+candidate*(grad(bi,1,c)*r(1)+grad(bi,2,c)*r(2))
  if(eta-zb<=20.d0*dry_tolerance.or.abs(zb-v(bi,c))>=h) return
 enddo
 factor=candidate
end function
