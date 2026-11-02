! Expose the unchanged native Euler update INCLUDING its conservative water
! draining, augmented f-wave wet/dry traces and CFL/positivity rejection.
subroutine shoreline_native_euler(qin,zin,dt,at_time,nthreads,qout,water,code) bind(C)
 use iso_c_binding
 use omp_lib, only: omp_set_num_threads
 real(c_double),intent(in) :: qin(nc,3),zin(nc)
 real(c_double),value :: dt,at_time
 integer(c_int),value :: nthreads
 real(c_double),intent(out) :: qout(nc,3),water(nf)
 integer(c_int),intent(out) :: code
 real(8),allocatable,save :: result(:,:),accepted(:,:),faceflux(:,:)
 integer :: i
 logical :: ok
 call omp_set_num_threads(nthreads)
 if(.not.allocated(result)) allocate(result(nstate,nc),accepted(nledger,nb),faceflux(nstate,nf))
 !$omp parallel do schedule(static)
 do i=1,nc
  q(1:3,i)=qin(i,:);q(4,i)=0.d0;q(5,i)=bulk*(zin(i)-rock(i))
 enddo
 !$omp end parallel do
 section_cache_valid=.false.
 call euler(q,dt,at_time,1.d0,result,accepted,ok,instantaneous_face_flux=faceflux)
 code=1
 if(.not.ok) return
 !$omp parallel do schedule(static)
 do i=1,nc
  qout(i,:)=result(1:3,i)
 enddo
 !$omp end parallel do
 water=faceflux(1,:);code=0
end subroutine

! Scoped override of open-face hydraulic fluxes computed by the shared
! exterior-state kernel. Ordinary calls never inherit this mutable switch.
subroutine shoreline_native_boundary_rates(qin,zin,dt,at_time,nthreads,mask,flux,rate,water,code) bind(C)
 use iso_c_binding
 real(c_double),intent(in) :: qin(nc,3),zin(nc),flux(nf,3)
 real(c_double),value :: dt,at_time
 integer(c_int),value :: nthreads
 integer(c_int),intent(in) :: mask(nf)
 real(c_double),intent(out) :: rate(nc,3),water(nf)
 integer(c_int),intent(out) :: code
 integer :: f
 if(.not.allocated(river_exterior_mask)) allocate(river_exterior_mask(nf),river_exterior_flux(3,nf))
 river_exterior_mask=mask
 !$omp parallel do schedule(static)
 do f=1,nf
  river_exterior_flux(:,f)=flux(f,:)
 enddo
 !$omp end parallel do
 river_exterior_active=.true.
 call bend_native_rates(qin,zin,dt,at_time,nthreads,rate,water,code)
 river_exterior_active=.false.
end subroutine
