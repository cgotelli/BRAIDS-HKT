! Fixed-level operator split: every patch reads the same pre-diffusion state.
! Ghost exchange is serial because GeoClaw bound() shares hierarchy scratch.
! The expensive closure and stress updates remain OpenMP patch-parallel.
subroutine braids_diffusion_level(level,nvar,naux)
 use amr_module
 use braids_momentum_diffusion_module
 use braids_turbulence_wall_mask_module
 implicit none
 integer,intent(in)::level,nvar,naux
 type patch_work
  real(8),allocatable::q(:,:,:)
  integer,allocatable::wx(:,:),wy(:,:)
  integer::mptr,m,n,loc,auxloc
  logical::link(4)
 end type
 type(patch_work),allocatable,save::work(:)
 integer::p,mp,m,n,loc,al,i,j,nx,ny
 real(8)::dx,dy,dt,left,ds,limit,time,xlo,ylo,patch_limit
 logical::walls
 if(.not.braids_diffusion_enabled())return
 if(mxnest/=1.or.level/=1)error stop 'BRAIDS turbulence requires fixed level 1'
 if(allocated(work))then
  if(size(work)/=numgrids(level))deallocate(work)
 endif
 if(.not.allocated(work))allocate(work(numgrids(level)))
 dx=hxposs(level);dy=hyposs(level);dt=possk(level)
 walls=braids_wall_turbulence_enabled()
 do p=1,size(work)
  mp=listOfGrids(listStart(level)+p-1)
  nx=node(ndihi,mp)-node(ndilo,mp)+1;ny=node(ndjhi,mp)-node(ndjlo,mp)+1
  m=nx+2*nghost;n=ny+2*nghost
  loc=node(store1,mp);al=node(storeaux,mp)
  if(allocated(work(p)%q))then
   if(any(shape(work(p)%q)/=[nvar,m,n]))deallocate(work(p)%q,work(p)%wx,work(p)%wy)
  endif
  if(.not.allocated(work(p)%q))allocate(work(p)%q(nvar,m,n),work(p)%wx(m+1,n),work(p)%wy(m,n+1))
  work(p)%mptr=mp;work(p)%m=m;work(p)%n=n;work(p)%loc=loc;work(p)%auxloc=al
  work(p)%link=[node(ndilo,mp)>0.or.xperdom,node(ndihi,mp)<iregsz(level)-1.or.xperdom, &
                node(ndjlo,mp)>0.or.yperdom,node(ndjhi,mp)<jregsz(level)-1.or.yperdom]
  time=rnode(timemult,mp)
  call b4step2(nghost,nx,ny,nvar,alloc(loc),rnode(cornxlo,mp),rnode(cornylo,mp), &
               dx,dy,time,dt,naux,alloc(al),.true.)
 enddo
 left=dt
 do while(left>0d0)
  limit=huge(1d0)
!$omp parallel do default(shared) private(p,m,n,loc,patch_limit) reduction(min:limit)
  do p=1,size(work)
   m=work(p)%m;n=work(p)%n;loc=work(p)%loc
   work(p)%q=reshape(alloc(loc:loc+nvar*m*n-1),[nvar,m,n])
   patch_limit=braids_diffusion_dt(nvar,m,n,nghost,dx,dy,work(p)%q,work(p)%link)
   limit=min(limit,patch_limit)
  enddo
!$omp end parallel do
  ! Equal substeps for the remaining interval, with stability re-evaluated
  ! after each synchronized update as the turbulent viscosity evolves.
  ds=left/dble(max(1,ceiling(left/limit)))
!$omp parallel do default(shared) private(p,mp,m,n,al,xlo,ylo)
  do p=1,size(work)
   mp=work(p)%mptr;m=work(p)%m;n=work(p)%n;al=work(p)%auxloc
   xlo=rnode(cornxlo,mp)-nghost*dx;ylo=rnode(cornylo,mp)-nghost*dy
   work(p)%wx=0;work(p)%wy=0
   if(walls)call braids_turbulence_wall_masks(nvar,m,n,nghost,dx,dy,xlo,ylo,work(p)%q, &
                         reshape(alloc(al:al+naux*m*n-1:naux),[m,n]),work(p)%link,work(p)%wx,work(p)%wy)
   call braids_momentum_diffusion(nvar,m,n,nghost,ds,dx,dy,work(p)%q,work(p)%link,work(p)%wx,work(p)%wy)
  enddo
!$omp end parallel do
  ! Commit interiors only, after all workers have finished reading halos.
!$omp parallel do default(shared) private(p,m,n,loc,i,j)
  do p=1,size(work)
   m=work(p)%m;n=work(p)%n;loc=work(p)%loc
   do j=nghost+1,n-nghost;do i=nghost+1,m-nghost
    alloc(loc+nvar*((j-1)*m+i-1):loc+nvar*((j-1)*m+i)-1)=work(p)%q(:,i,j)
   enddo;enddo
  enddo
!$omp end parallel do
  do p=1,size(work)
   mp=work(p)%mptr;m=work(p)%m;n=work(p)%n;loc=work(p)%loc;al=work(p)%auxloc
   time=rnode(timemult,mp)
   call bound(time,nvar,nghost,alloc(loc),m,n,mp,alloc(al),naux)
  enddo
  left=max(0d0,left-ds)
 enddo
end subroutine
