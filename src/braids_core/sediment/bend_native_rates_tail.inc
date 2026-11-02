  ! The reconstruction/traces/augmented flux prefix is copied byte-for-byte
  ! from the recorded native source, except for the subroutine name.
  ! Stop before explicit draining and CFL rejection. Fully wet qualification only.
  next=0.d0;accepted=0.d0
  !$omp parallel do schedule(static,256) default(shared) private(c,k,f,sumflux)
  do c=1,nc
   sumflux=0.d0
   do k=offset(c),offset(c+1)-1
    f=faces(k)
    if(signs(k)>0) then
     sumflux(1)=sumflux(1)+length(f)/area(c)*fl(1,f)
     sumflux(2:3)=sumflux(2:3)+length(f)/area(c)*(fl(2:3,f)-.5d0*grav*state(1,c)**2*normal(:,f))
    else
     sumflux(1)=sumflux(1)-length(f)/area(c)*fr(1,f)
     sumflux(2:3)=sumflux(2:3)-length(f)/area(c)*(fr(2:3,f)-.5d0*grav*state(1,c)**2*normal(:,f))
    endif
   enddo
   next(1:3,c)=-sumflux(1:3)
  enddo
  !$omp end parallel do
  if(present(instantaneous_face_flux)) instantaneous_face_flux=fl
  ok=all(ieee_is_finite(next)).and.all(ieee_is_finite(fl))
 end subroutine
