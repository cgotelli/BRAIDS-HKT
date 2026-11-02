! C ABI for an isolated copy of the current reconstructed native hydraulics.
! The builder inserts this INSIDE cutcell_solver, not into production sources.
subroutine bend_native_load(path,npath,ncell,nface,nthreads,code) bind(C)
 use iso_c_binding
 use omp_lib, only: omp_set_num_threads,omp_set_dynamic
 character(c_char),intent(in) :: path(*)
 integer(c_int),value :: npath,ncell,nface,nthreads
 integer(c_int),intent(out) :: code
 character(len=:),allocatable :: filename
 integer :: i
 code=1
 call omp_set_dynamic(.false.)
 call omp_set_num_threads(nthreads)
 allocate(character(len=npath)::filename)
 do i=1,npath
  filename(i:i)=path(i)
 enddo
 call load_case(filename)
 if(nc/=ncell.or.nf/=nface.or.nstate/=5.or.nfrac/=1.or.nbed_layers/=1) return
 if(bedlaw/=0.or.exchange_mode/=0.or.secondary_flow.or.horizontal_viscosity/=0) return
 code=0
end subroutine

subroutine bend_native_geometry(qout,zrock,centres,areas,owners,neighbours,bids,normals,lengths) bind(C)
 use iso_c_binding
 real(c_double),intent(out) :: qout(nc,5),zrock(nc),centres(nc,2),areas(nc),normals(nf,2),lengths(nf)
 integer(c_int),intent(out) :: owners(nf),neighbours(nf),bids(nf)
 integer :: i
 do i=1,nc
  qout(i,:)=q(:,i);centres(i,:)=xy(:,i)
 enddo
 do i=1,nf
  normals(i,:)=normal(:,i)
 enddo
 zrock=rock;areas=area;owners=owner-1;neighbours=neighbour-1;bids=bid-1;lengths=length
end subroutine

subroutine bend_native_rates(qin,zin,dt,at_time,nthreads,rate,water,code) bind(C)
 use iso_c_binding
 use omp_lib, only: omp_set_num_threads
 real(c_double),intent(in) :: qin(nc,3),zin(nc)
 real(c_double),value :: dt,at_time
 integer(c_int),value :: nthreads
 real(c_double),intent(out) :: rate(nc,3),water(nf)
 integer(c_int),intent(out) :: code
 real(8),allocatable,save :: next(:,:),accepted(:,:),faceflux(:,:)
 integer :: i
 logical :: ok
 call omp_set_num_threads(nthreads)
 if(.not.allocated(next)) allocate(next(nstate,nc),accepted(nledger,nb),faceflux(nstate,nf))
 !$omp parallel do schedule(static) private(i)
 do i=1,nc
  q(1:3,i)=qin(i,:);q(4,i)=0.d0;q(5,i)=bulk*(zin(i)-rock(i))
 enddo
 !$omp end parallel do
 ! Caches hold workspace only; no old-stage boundary latch may change residuals.
 section_cache_valid=.false.
 call bend_native_spatial_rates(q,dt,at_time,1.d0,next,accepted,ok,instantaneous_face_flux=faceflux)
 code=1
 if(.not.ok) return
 !$omp parallel do schedule(static) private(i)
 do i=1,nc
  rate(i,:)=next(1:3,i)
 enddo
 !$omp end parallel do
 water=faceflux(1,:)
 code=0
end subroutine
