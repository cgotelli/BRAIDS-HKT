! Diagnostic ABI, inserted only into an isolated validation library.
subroutine bend_wall_traces(qin,zin,gout,tout,flags) bind(C)
 use iso_c_binding
 real(c_double),intent(in) :: qin(nc,3),zin(nc)
 real(c_double),intent(out) :: gout(nc,2,4),tout(nf,4,2)
 integer(c_int),intent(out) :: flags(nf,2,3)
 real(8) :: v(nprimitive,nc),bed(nc),grad(nprimitive,2,nc),vlo(nprimitive,nc),vhi(nprimitive,nc)
 real(8) :: bcc(nfrac,nb),value(nprimitive)
 integer :: i,f,c,side,m,indices(4),slot
 do i=1,nc
  q(1:3,i)=qin(i,:);q(4,i)=0.d0;q(5,i)=bulk*(zin(i)-rock(i))
 enddo
 active_cell=.true.;bcc=0.d0
 call primitive(q,v,bed,.false.)
 call gradients(v,bcc,grad,vlo,vhi)
 if(scheme==6) call prepare_wall_traces(q,v,vlo,vhi)
 call prepare_wet_dry_traces(q,v,grad,vlo,vhi)
 indices=[1,2,3,bed_primitive_index()]
 do i=1,nc
  do m=1,4
   gout(i,:,m)=grad(indices(m),:,i)
  enddo
 enddo
 tout=0.d0;flags=0
 do f=1,nf
  do side=1,2
   c=owner(f)
   if(side==2) c=neighbour(f)
   if(c==0) cycle
   value=face_value(c,f,v,grad,vlo,vhi)
   tout(f,:,side)=value(indices)
   if(weno_cells(1,side,f)>0) flags(f,side,1)=1
   if(hydro_constant(c)) flags(f,side,2)=1
   slot=wall_slot(side,f)
   if(slot>0) then
    if(wall_ready(slot,c)) flags(f,side,3)=1
   endif
  enddo
 enddo
end subroutine

! Sediment needs only fresh primitive face values, not diagnostic gradients and
! flags. The reconstruction calls and face_value function are exactly the same.
! Retain the diagnostic ABI above for independent comparisons and old callers.
subroutine bend_sediment_traces(qin,zin,nthreads,tout) bind(C)
 use iso_c_binding
 use omp_lib, only: omp_set_num_threads
 real(c_double),intent(in) :: qin(nc,3),zin(nc)
 integer(c_int),value :: nthreads
 real(c_double),intent(out) :: tout(nf,4,2)
 real(8) :: v(nprimitive,nc),bed(nc),grad(nprimitive,2,nc),vlo(nprimitive,nc),vhi(nprimitive,nc)
 real(8) :: bcc(nfrac,nb),value(nprimitive)
 integer :: i,f,c,side,indices(4)
 call omp_set_num_threads(max(1,nthreads))
 !$omp parallel do private(i) schedule(static)
 do i=1,nc
  q(1:3,i)=qin(i,:);q(4,i)=0.d0;q(5,i)=bulk*(zin(i)-rock(i))
 enddo
 !$omp end parallel do
 active_cell=.true.;bcc=0.d0
 call primitive(q,v,bed,.false.)
 call gradients(v,bcc,grad,vlo,vhi)
 if(scheme==6) call prepare_wall_traces(q,v,vlo,vhi)
 call prepare_wet_dry_traces(q,v,grad,vlo,vhi)
 indices=[1,2,3,bed_primitive_index()]
 !$omp parallel do private(f,side,c,value) schedule(static)
 do f=1,nf
  do side=1,2
   c=owner(f)
   if(side==2) c=neighbour(f)
   if(c==0) then
    tout(f,:,side)=0.d0
   else
    value=face_value(c,f,v,grad,vlo,vhi)
    tout(f,:,side)=value(indices)
   endif
  enddo
 enddo
 !$omp end parallel do
end subroutine
