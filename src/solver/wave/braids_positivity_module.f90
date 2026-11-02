! G4 fixed-level conservative flux prepass/cache. No AMR or periodic domains.
! Each patch runs its actual b4step2/step2 once. Interior donor budgets are
! gathered on unique global cells, so neighboring patches use identical theta.
module braids_positivity_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use amr_module, only: alloc,node,rnode,numgrids,listStart,listOfGrids,nghost, &
        ndilo,ndihi,ndjlo,ndjhi,store1,storeaux,cornxlo,cornylo,iregsz,jregsz, &
        hxposs,hyposs,mxnest,mcapa,xperdom,yperdom
    use geoclaw_module, only: coordinate_system,use_fwave_positivity_limiter,grav
    use braids_boundary_module, only: braids_boundary_flux
    use braids_draining_module, only: braids_draining_corrections
    use braids_passive_module, only: braids_passive_fluxes
    use braids_bedload_module, only: braids_bedload_prepare,braids_bedload_ssprk
    use braids_rk_module, only: braids_rk_fluxes
    implicit none
    private
    public :: braids_positivity_prepare,braids_positivity_prepared,braids_positivity_take_flux
    public :: braids_positivity_finish,braids_positivity_statistics
    type :: patch_cache
        integer :: id,ilo,ihi,jlo,jhi,nx,ny
        logical :: consumed=.false.
        real(kind=8) :: cfl=0.d0
        real(kind=8), allocatable :: q(:,:,:),aux(:,:,:),fm(:,:,:),fp(:,:,:),gm(:,:,:),gp(:,:,:)
    end type
    type(patch_cache), allocatable, save :: cache(:)
    logical, save :: configured=.false.,enabled=.false.,active=.false.
    integer, save :: current_nvar=0,current_naux=0,limited_faces=0,limited_cells=0
    real(kind=8), save :: current_time=0.d0,current_dt=0.d0,current_dx=0.d0,current_dy=0.d0
    real(kind=8), save :: minimum_theta=1.d0,minimum_unlimited_depth=0.d0
contains
    subroutine fail(message)
        character(len=*), intent(in) :: message
        write(*,'(a)') 'BRAIDS positivity: '//trim(message)
        error stop 1
    end subroutine

    subroutine configure()
        logical :: exists
        integer :: unit,ios,flag
        character(len=256) :: line
        if (configured) return
        configured=.true.
        inquire(file='braids_positivity.data',exist=exists)
        if (.not.exists) return
        open(newunit=unit,file='braids_positivity.data',status='old',action='read',iostat=ios)
        if (ios/=0) call fail('cannot open braids_positivity.data')
        do
            read(unit,'(a)',iostat=ios) line
            if (ios/=0) call fail('braids_positivity.data requires one flag, 0 or 1')
            line=adjustl(line)
            if (len_trim(line)==0) cycle
            if (line(1:1)=='#') cycle
            read(line,*,iostat=ios) flag
            if (ios/=0) call fail('invalid braids_positivity.data flag')
            exit
        enddo
        close(unit)
        if (flag/=0.and.flag/=1) call fail('braids_positivity.data flag must be 0 or 1')
        enabled=flag==1
    end subroutine

    integer function slot(id) result(found)
        integer, intent(in) :: id
        integer :: k
        found=0
        do k=1,size(cache)
            if (cache(k)%id/=id) cycle
            found=k
            return
        enddo
        call fail('unregistered physical patch')
    end function

    logical function braids_positivity_prepared(id,time,dt) result(prepared)
        integer, intent(in) :: id
        real(kind=8), intent(in) :: time,dt
        integer :: k
        prepared=.false.
        if (.not.active) return
        if (time/=current_time.or.dt/=current_dt) call fail('physical patch has a stale accepted interval')
        k=slot(id)
        if (cache(k)%consumed) call fail('a physical patch was advanced twice')
        prepared=.true.
    end function

    subroutine braids_positivity_prepare(level,time,dt,nvar,naux)
        integer, intent(in) :: level,nvar,naux
        real(kind=8), intent(in) :: time,dt
        integer :: k,id,nx,ny,mitot,mjtot,lq,la,i,j,ig,jg,nxg,nyg,il,jb
        integer, allocatable :: cell_owner(:,:)
        real(kind=8), allocatable :: qglobal(:,:,:),fx(:,:),fy(:,:),theta(:,:),delta_x(:,:,:),delta_y(:,:,:)
        real(kind=8) :: value,reference,scale,xlow,ylow
        logical :: overdraw
        external b4step2
        call configure()
        if (active) call fail('preparation during an uncommitted physical step')
        limited_faces=0; limited_cells=0; minimum_theta=1.d0; minimum_unlimited_depth=0.d0
        if (.not.enabled) return
        if (level/=1.or.mxnest/=1.or.mcapa/=0.or.coordinate_system/=1.or.xperdom.or.yperdom) &
            call fail('requires one complete Cartesian fixed level without capacity or periodic boundaries')
        if (nvar<3.or.naux<1.or.use_fwave_positivity_limiter) &
            call fail('requires SWE water fluctuations in the first three equations')
        if (.not.all(ieee_is_finite([time,dt])).or.dt<=0.d0) call fail('invalid accepted time interval')
        current_time=time; current_dt=dt; current_nvar=nvar; current_naux=naux
        current_dx=hxposs(level); current_dy=hyposs(level)
        nxg=iregsz(level); nyg=jregsz(level)
        if (numgrids(level)<1.or.min(nxg,nyg)<1) call fail('empty fixed-level hierarchy')
        if (allocated(cache)) deallocate(cache)
        allocate(cache(numgrids(level)))
        allocate(qglobal(3,nxg,nyg),cell_owner(nxg,nyg),fx(nxg+1,nyg),fy(nxg,nyg+1))
        allocate(theta(nxg,nyg),delta_x(3,nxg+1,nyg),delta_y(3,nxg,nyg+1))
        cell_owner=0; fx=0.d0; fy=0.d0
        ! Commit the same b4step2 state previously prepared inside par_advanc.
        ! The matching later par_advanc call skips it, so no hook runs twice.
        do k=1,size(cache)
            id=listOfGrids(listStart(level)+k-1)
            cache(k)%id=id
            cache(k)%ilo=node(ndilo,id); cache(k)%ihi=node(ndihi,id)
            cache(k)%jlo=node(ndjlo,id); cache(k)%jhi=node(ndjhi,id)
            nx=cache(k)%ihi-cache(k)%ilo+1; ny=cache(k)%jhi-cache(k)%jlo+1
            if (min(nx,ny)<1.or.min(cache(k)%ilo,cache(k)%jlo)<0.or. &
                cache(k)%ihi>=nxg.or.cache(k)%jhi>=nyg) call fail('patch outside the fixed domain')
            cache(k)%nx=nx; cache(k)%ny=ny
            mitot=nx+2*nghost; mjtot=ny+2*nghost
            allocate(cache(k)%q(nvar,mitot,mjtot),cache(k)%aux(naux,mitot,mjtot))
            allocate(cache(k)%fm(nvar,mitot,mjtot),cache(k)%fp(nvar,mitot,mjtot), &
                     cache(k)%gm(nvar,mitot,mjtot),cache(k)%gp(nvar,mitot,mjtot))
            lq=node(store1,id); la=node(storeaux,id)
            xlow=rnode(cornxlo,id); ylow=rnode(cornylo,id)
            call b4step2(nghost,nx,ny,nvar,alloc(lq),xlow,ylow,current_dx,current_dy,time,dt, &
                         naux,alloc(la),.true.)
            cache(k)%q=reshape(alloc(lq:lq+nvar*mitot*mjtot-1),shape(cache(k)%q))
            cache(k)%aux=reshape(alloc(la:la+naux*mitot*mjtot-1),shape(cache(k)%aux))
            do j=1,ny
                jg=cache(k)%jlo+j
                do i=1,nx
                    ig=cache(k)%ilo+i
                    if (cell_owner(ig,jg)/=0) call fail('duplicate global donor cell')
                    cell_owner(ig,jg)=k
                    qglobal(:,ig,jg)=cache(k)%q(1:3,nghost+i,nghost+j)
                enddo
            enddo
        enddo
        if (any(cell_owner==0)) call fail('incomplete global donor coverage')
        if (braids_bedload_ssprk()) then
            call prepare_rk_fluxes(nxg,nyg)
            active=.true.
            return
        endif
        ! Same independent patch Riemann solves as advanc, once per step.
        !$omp parallel do private(k) schedule(static)
        do k=1,size(cache)
            call calculate_flux(k)
        enddo
        !$omp end parallel do
        do k=1,size(cache)
            il=nghost+1; jb=nghost+1
            do j=jb,jb+cache(k)%ny-1
                jg=cache(k)%jlo+j-nghost
                do i=il,il+cache(k)%nx-1
                    ig=cache(k)%ilo+i-nghost
                    fx(ig+1,jg)=cache(k)%q(2,i,j)+cache(k)%fm(1,i+1,j)
                    fy(ig,jg+1)=cache(k)%q(3,i,j)+cache(k)%gm(1,i,j+1)
                    if (ig==1) fx(1,jg)=cache(k)%q(2,i,j)+cache(k)%fp(1,i,j)
                    if (jg==1) fy(ig,1)=cache(k)%q(3,i,j)+cache(k)%gp(1,i,j)
                enddo
            enddo
        enddo
        ! Every interior interface has two fluctuation representations of
        ! one mass flux. Reject a material incompatibility rather than hiding it.
        do k=1,size(cache)
            do j=nghost+1,nghost+cache(k)%ny
                jg=cache(k)%jlo+j-nghost
                do i=nghost+1,nghost+cache(k)%nx
                    ig=cache(k)%ilo+i-nghost
                    value=cache(k)%q(2,i,j)+cache(k)%fp(1,i,j); reference=fx(ig,jg)
                    scale=max(tiny(1.d0),abs(value),abs(reference),abs(cache(k)%q(2,i,j)))
                    scale=max(scale,qglobal(1,ig,jg)*sqrt(grav*qglobal(1,ig,jg)))
                    if (ig>1) scale=max(scale,abs(qglobal(2,ig-1,jg)), &
                        qglobal(1,ig-1,jg)*sqrt(grav*qglobal(1,ig-1,jg)))
                    if (abs(value-reference)>2048.d0*epsilon(1.d0)*scale) then
                        write(*,*) 'G4_X_FLUX_MISMATCH time,dt,ig,jg,Fright,Fleft,scale: ', &
                            time,dt,ig,jg,value,reference,scale
                        write(*,*) 'G4_X_FLUX_STATES left,right: ',qglobal(:,max(1,ig-1),jg),qglobal(:,ig,jg)
                        call fail('incompatible shared x mass flux')
                    endif
                    value=cache(k)%q(3,i,j)+cache(k)%gp(1,i,j); reference=fy(ig,jg)
                    scale=max(tiny(1.d0),abs(value),abs(reference),abs(cache(k)%q(3,i,j)))
                    scale=max(scale,qglobal(1,ig,jg)*sqrt(grav*qglobal(1,ig,jg)))
                    if (jg>1) scale=max(scale,abs(qglobal(3,ig,jg-1)), &
                        qglobal(1,ig,jg-1)*sqrt(grav*qglobal(1,ig,jg-1)))
                    if (abs(value-reference)>2048.d0*epsilon(1.d0)*scale) then
                        write(*,*) 'G4_Y_FLUX_MISMATCH time,dt,ig,jg,Ftop,Fbottom,scale: ', &
                            time,dt,ig,jg,value,reference,scale
                        write(*,*) 'G4_Y_FLUX_STATES bottom,top: ',qglobal(:,ig,max(1,jg-1)),qglobal(:,ig,jg)
                        call fail('incompatible shared y mass flux')
                    endif
                enddo
            enddo
        enddo
        call braids_draining_corrections(nxg,nyg,qglobal,fx,fy,current_dx,current_dy,dt, &
            delta_x,delta_y,theta,limited_faces,limited_cells,minimum_unlimited_depth,overdraw,enforce_donor_cfl=nvar>3)
        if (overdraw) call fail('fixed exterior flux alone overdraws a cell; reduce the accepted timestep')
        minimum_theta=minval(theta)
        do k=1,size(cache)
            do j=nghost+1,nghost+cache(k)%ny
                jg=cache(k)%jlo+j-nghost
                do i=nghost+1,nghost+cache(k)%nx+1
                    ig=cache(k)%ilo+i-nghost
                    if (all(delta_x(:,ig,jg)==0.d0)) cycle
                    cache(k)%fm(1:3,i,j)=cache(k)%fm(1:3,i,j)+delta_x(:,ig,jg)
                    cache(k)%fp(1:3,i,j)=cache(k)%fp(1:3,i,j)+delta_x(:,ig,jg)
                enddo
            enddo
            do j=nghost+1,nghost+cache(k)%ny+1
                jg=cache(k)%jlo+j-nghost
                do i=nghost+1,nghost+cache(k)%nx
                    ig=cache(k)%ilo+i-nghost
                    if (all(delta_y(:,ig,jg)==0.d0)) cycle
                    cache(k)%gm(1:3,i,j)=cache(k)%gm(1:3,i,j)+delta_y(:,ig,jg)
                    cache(k)%gp(1:3,i,j)=cache(k)%gp(1:3,i,j)+delta_y(:,ig,jg)
                enddo
            enddo
        enddo
        ! Reconstruct and limit suspended fluxes over the complete fixed
        ! level. Artificial patch edges are not physical scalar boundaries.
        if (nvar>=4) call prepare_scalar_fluxes(nxg,nyg)
        active=.true.
    end subroutine

    subroutine prepare_scalar_fluxes(nx,ny)
        integer, intent(in) :: nx,ny
        integer :: k,i,j,ig,jg,mi,mj,il,ir,jb,jt,nv
        real(kind=8), allocatable :: q(:,:,:),fm(:,:,:),fp(:,:,:),gm(:,:,:),gp(:,:,:)
        if (size(cache)==1) then
            call braids_passive_fluxes(current_nvar,size(cache(1)%q,2),size(cache(1)%q,3), &
                nghost,current_dt,current_dx,current_dy,cache(1)%q, &
                cache(1)%fm,cache(1)%fp,cache(1)%gm,cache(1)%gp)
            call braids_bedload_prepare(current_nvar,size(cache(1)%q,2),size(cache(1)%q,3), &
                nghost,current_time,current_dt,current_dx,current_dy,cache(1)%q, &
                cache(1)%fm,cache(1)%fp,cache(1)%gm,cache(1)%gp)
            return
        endif
        nv=current_nvar; mi=nx+2*nghost; mj=ny+2*nghost
        allocate(q(nv,mi,mj),fm(nv,mi,mj),fp(nv,mi,mj),gm(nv,mi,mj),gp(nv,mi,mj))
        q=0.d0; fm=0.d0; fp=0.d0; gm=0.d0; gp=0.d0
        do k=1,size(cache)
            il=nghost+1; ir=nghost+cache(k)%nx
            jb=nghost+1; jt=nghost+cache(k)%ny
            ! Unique interior owners, plus the existing physical halo.
            do j=1,size(cache(k)%q,3)
                jg=cache(k)%jlo+j
                if ((j<jb.or.j>jt).and.jg>nghost.and.jg<=ny+nghost) cycle
                do i=1,size(cache(k)%q,2)
                    ig=cache(k)%ilo+i
                    if ((i<il.or.i>ir).and.ig>nghost.and.ig<=nx+nghost) cycle
                    q(:,ig,jg)=cache(k)%q(:,i,j)
                enddo
            enddo
            ! Each cell supplies its own one-sided hydraulic fluctuations.
            ! Use the accepted, donor-limited cache without a second solve.
            do j=jb,jt
                jg=cache(k)%jlo+j
                do i=il,ir
                    ig=cache(k)%ilo+i
                    fm(:,ig+1,jg)=cache(k)%fm(:,i+1,j)
                    fp(:,ig,jg)=cache(k)%fp(:,i,j)
                    gm(:,ig,jg+1)=cache(k)%gm(:,i,j+1)
                    gp(:,ig,jg)=cache(k)%gp(:,i,j)
                    if (ig==nghost+1) fm(:,ig,jg)=cache(k)%fm(:,i,j)
                    if (ig==nx+nghost) fp(:,ig+1,jg)=cache(k)%fp(:,i+1,j)
                    if (jg==nghost+1) gm(:,ig,jg)=cache(k)%gm(:,i,j)
                    if (jg==ny+nghost) gp(:,ig,jg+1)=cache(k)%gp(:,i,j+1)
                enddo
            enddo
        enddo
        call braids_passive_fluxes(nv,mi,mj,nghost,current_dt,current_dx,current_dy,q,fm,fp,gm,gp)
        call braids_bedload_prepare(nv,mi,mj,nghost,current_time,current_dt,current_dx,current_dy,q,fm,fp,gm,gp)
        ! Identical shared-face scalar fluxes are scattered to both patches.
        do k=1,size(cache)
            do j=nghost+1,nghost+cache(k)%ny
                jg=cache(k)%jlo+j
                do i=nghost+1,nghost+cache(k)%nx+1
                    ig=cache(k)%ilo+i
                    cache(k)%fm(4:min(5,nv),i,j)=fm(4:min(5,nv),ig,jg)
                    cache(k)%fp(4:min(5,nv),i,j)=fp(4:min(5,nv),ig,jg)
                enddo
            enddo
            do j=nghost+1,nghost+cache(k)%ny+1
                jg=cache(k)%jlo+j
                do i=nghost+1,nghost+cache(k)%nx
                    ig=cache(k)%ilo+i
                    cache(k)%gm(4:min(5,nv),i,j)=gm(4:min(5,nv),ig,jg)
                    cache(k)%gp(4:min(5,nv),i,j)=gp(4:min(5,nv),ig,jg)
                enddo
            enddo
        enddo
        deallocate(q,fm,fp,gm,gp)
    end subroutine prepare_scalar_fluxes

    subroutine prepare_rk_fluxes(nx,ny)
        integer, intent(in) :: nx,ny
        integer :: nv,na,mi,mj,k,i,j,ig,jg,il,ir,jb,jt
        real(8), allocatable :: q(:,:,:),aux(:,:,:),fm(:,:,:),fp(:,:,:),gm(:,:,:),gp(:,:,:)
        nv=current_nvar;na=current_naux;mi=nx+2*nghost;mj=ny+2*nghost
        allocate(q(nv,mi,mj),aux(na,mi,mj),fm(nv,mi,mj),fp(nv,mi,mj),gm(nv,mi,mj),gp(nv,mi,mj))
        q=0.d0;aux=0.d0
        do k=1,size(cache)
            il=nghost+1;ir=nghost+cache(k)%nx;jb=nghost+1;jt=nghost+cache(k)%ny
            do j=1,size(cache(k)%q,3)
                jg=cache(k)%jlo+j
                if((j<jb.or.j>jt).and.jg>nghost.and.jg<=ny+nghost) cycle
                do i=1,size(cache(k)%q,2)
                    ig=cache(k)%ilo+i
                    if((i<il.or.i>ir).and.ig>nghost.and.ig<=nx+nghost) cycle
                    q(:,ig,jg)=cache(k)%q(:,i,j);aux(:,ig,jg)=cache(k)%aux(:,i,j)
                enddo
            enddo
        enddo
        call braids_rk_fluxes(nv,na,mi,mj,current_time,current_dt,current_dx,current_dy, &
            q,aux,fm,fp,gm,gp,rk_hydro)
        do k=1,size(cache)
            do j=1,size(cache(k)%q,3)
                jg=cache(k)%jlo+j
                do i=1,size(cache(k)%q,2)
                    ig=cache(k)%ilo+i
                    cache(k)%q(:,i,j)=q(:,ig,jg);cache(k)%aux(:,i,j)=aux(:,ig,jg)
                    cache(k)%fm(:,i,j)=fm(:,ig,jg);cache(k)%fp(:,i,j)=fp(:,ig,jg)
                    cache(k)%gm(:,i,j)=gm(:,ig,jg);cache(k)%gp(:,i,j)=gp(:,ig,jg)
                enddo
            enddo
            cache(k)%cfl=0.d0
            do j=nghost+1,nghost+cache(k)%ny
                do i=nghost+1,nghost+cache(k)%nx
                    if(cache(k)%q(1,i,j)<=0.d0) cycle
                    cache(k)%cfl=max(cache(k)%cfl,current_dt*( &
                        (abs(cache(k)%q(2,i,j)/cache(k)%q(1,i,j))+sqrt(grav*cache(k)%q(1,i,j)))/current_dx &
                       +(abs(cache(k)%q(3,i,j)/cache(k)%q(1,i,j))+sqrt(grav*cache(k)%q(1,i,j)))/current_dy))
                enddo
            enddo
        enddo
    end subroutine

    subroutine rk_hydro(nv,na,mi,mj,q,aux,fm,fp,gm,gp)
        integer, intent(in) :: nv,na,mi,mj
        real(8), intent(in) :: q(nv,mi,mj),aux(na,mi,mj)
        real(8), intent(out) :: fm(nv,mi,mj),fp(nv,mi,mj),gm(nv,mi,mj),gp(nv,mi,mj)
        integer :: k,i,j,ig,jg,il,ir,jb,jt
        do k=1,size(cache)
            cache(k)%q=q(:,cache(k)%ilo+1:cache(k)%ihi+1+2*nghost, &
                           cache(k)%jlo+1:cache(k)%jhi+1+2*nghost)
            cache(k)%aux=aux(:,cache(k)%ilo+1:cache(k)%ihi+1+2*nghost, &
                              cache(k)%jlo+1:cache(k)%jhi+1+2*nghost)
        enddo
!$omp parallel do private(k) schedule(static)
        do k=1,size(cache)
            call calculate_flux(k,mol=.true.)
        enddo
!$omp end parallel do
        fm=0.d0;fp=0.d0;gm=0.d0;gp=0.d0
        do k=1,size(cache)
            il=nghost+1;ir=nghost+cache(k)%nx;jb=nghost+1;jt=nghost+cache(k)%ny
            do j=jb,jt
                jg=cache(k)%jlo+j
                do i=il,ir
                    ig=cache(k)%ilo+i
                    fm(:,ig+1,jg)=cache(k)%fm(:,i+1,j);fp(:,ig,jg)=cache(k)%fp(:,i,j)
                    gm(:,ig,jg+1)=cache(k)%gm(:,i,j+1);gp(:,ig,jg)=cache(k)%gp(:,i,j)
                    if(ig==nghost+1) fm(:,ig,jg)=cache(k)%fm(:,i,j)
                    if(ig==mi-nghost) fp(:,ig+1,jg)=cache(k)%fp(:,i+1,j)
                    if(jg==nghost+1) gm(:,ig,jg)=cache(k)%gm(:,i,j)
                    if(jg==mj-nghost) gp(:,ig,jg+1)=cache(k)%gp(:,i,j+1)
                enddo
            enddo
        enddo
    end subroutine

    subroutine calculate_flux(k,mol)
        logical, optional, intent(in) :: mol
        real(8) :: flux_dt
        integer, intent(in) :: k
        integer :: mitot,mjtot,id
        real(kind=8) :: xlow,ylow
        external step2,rpn2,rpt2
        id=cache(k)%id
        mitot=size(cache(k)%q,2); mjtot=size(cache(k)%q,3)
        xlow=rnode(cornxlo,id)-nghost*current_dx; ylow=rnode(cornylo,id)-nghost*current_dy
        flux_dt=current_dt
        if(present(mol)) then
            if(mol) flux_dt=0.d0
        endif
        call step2(max(cache(k)%nx,cache(k)%ny),current_nvar,current_naux,nghost,cache(k)%nx,cache(k)%ny, &
            cache(k)%q,cache(k)%aux,current_dx,current_dy,flux_dt,cache(k)%cfl, &
            cache(k)%fm,cache(k)%fp,cache(k)%gm,cache(k)%gp,rpn2,rpt2)
        call braids_boundary_flux(current_nvar,mitot,mjtot,nghost,current_dx,current_dy, &
            xlow,ylow,current_time,current_dt,cache(k)%q,cache(k)%aux,current_naux, &
            cache(k)%fm,cache(k)%fp,cache(k)%gm,cache(k)%gp)
    end subroutine

    subroutine braids_positivity_take_flux(id,nvar,naux,mitot,mjtot,mbc,time,dt,dx,dy,q,aux, &
                                         fm,fp,gm,gp,cfl,available)
        integer, intent(in) :: id,nvar,naux,mitot,mjtot,mbc
        real(kind=8), intent(in) :: time,dt,dx,dy,q(nvar,mitot,mjtot),aux(naux,mitot,mjtot)
        real(kind=8), intent(inout) :: fm(nvar,mitot,mjtot),fp(nvar,mitot,mjtot), &
            gm(nvar,mitot,mjtot),gp(nvar,mitot,mjtot),cfl
        logical, intent(out) :: available
        integer :: k
        available=braids_positivity_prepared(id,time,dt)
        if (.not.available) return
        k=slot(id)
        if (nvar/=current_nvar.or.naux/=current_naux.or.mbc/=nghost.or. &
            mitot/=size(cache(k)%q,2).or.mjtot/=size(cache(k)%q,3).or. &
            dx/=current_dx.or.dy/=current_dy) call fail('cached physical dimensions changed')
        if (any(q/=cache(k)%q).or.any(aux/=cache(k)%aux)) call fail('prepared state changed after flux prepass')
        fm=cache(k)%fm; fp=cache(k)%fp; gm=cache(k)%gm; gp=cache(k)%gp; cfl=cache(k)%cfl
        cache(k)%consumed=.true.
    end subroutine

    subroutine braids_positivity_finish()
        integer :: k
        if (.not.enabled) return
        if (.not.active) call fail('finish without accepted preparation')
        do k=1,size(cache)
            if (.not.cache(k)%consumed) call fail('accepted physical patch did not consume its flux')
        enddo
        active=.false.
        deallocate(cache)
    end subroutine

    subroutine braids_positivity_statistics(face_count,cell_count,min_theta,min_depth)
        integer, intent(out) :: face_count,cell_count
        real(kind=8), intent(out) :: min_theta,min_depth
        face_count=limited_faces; cell_count=limited_cells
        min_theta=minimum_theta; min_depth=minimum_unlimited_depth
    end subroutine
end module braids_positivity_module
