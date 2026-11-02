! One-fraction current-only bedload, fixed Cartesian level and explicit faces.
module braids_bedload_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use amr_module, only: alloc,node,store1,ndilo,ndihi,ndjlo,ndjhi,nghost, &
        listStart,listOfGrids,numgrids,hxposs,hyposs,mxnest,mcapa,xperdom,yperdom,iregsz,jregsz
    use geoclaw_module, only: coordinate_system,dry_tolerance
    use braids_settling_module, only: braids_bed_material,braids_sediment_dry_tolerance
    use braids_van_rijn_1993_module, only: braids_vr1993_grain_parameters
    use braids_bedload_vr1993_module, only: braids_vr1993_bedload_cell
    use braids_bedload_flux_module, only: braids_bedload_faces,braids_bedload_limit
    use braids_bedload_mpm_module, only: braids_mpm_bedload
    use braids_exner_mol_module, only: reconstruct
    implicit none
    private
    public :: braids_bedload_enabled,braids_bedload_dt,braids_bedload_prepare
    public :: braids_bedload_bed_update,braids_bedload_commit
    public :: braids_bedload_ssprk,braids_bedload_accept,braids_bedload_bulk
    integer, save :: law=1,reconstruction=0
    real(8), save :: mpm_cf=0.d0,mpm_critical=0.d0
    logical, save :: configured=.false.,enabled=.false.,pending=.false.,ledger_open=.false.
    integer, allocatable, save :: ids(:),sides(:),indices(:),modes(:)
    real(8), allocatable, save :: rates(:),inputs(:),exports(:)
    real(8), save :: rhos,bulk,rhow,g,nu,d50,d90,z0,dstar,taucr,minh,max_dt,bed_fraction
    real(8), save :: start_time,step_dt,before,after,min_theta,last_end=0.d0
    integer, save :: ledger
    character(*), parameter :: header='t_start,dt,boundary_id,input_kg,export_kg,'// &
        'bed_before_transport_kg,bed_after_transport_kg,minimum_theta'
contains
    subroutine fail(message)
        character(*), intent(in) :: message
        write(*,'(a)') 'BRAIDS bedload: '//message
        error stop 1
    end subroutine

    logical function braids_bedload_enabled() result(on)
        integer :: unit,ios,version,n,k
        real(8) :: reference(6)
        if (.not.configured) then
            inquire(file='braids_bedload.data',exist=enabled)
            if (enabled) then
                call braids_bed_material(rhos,bulk,reference)
                open(newunit=unit,file='braids_bedload.data',status='old',action='read',iostat=ios)
                if (ios/=0) call fail('cannot open configuration')
                read(unit,*,iostat=ios) version
                if (ios/=0.or.(version/=1.and.version/=2)) call fail('configuration version must be 1 or 2')
                read(unit,*,iostat=ios) rhow,g,nu,d50,d90,z0,minh,max_dt,bed_fraction
                if (ios/=0) call fail('missing material/depth/timestep parameters')
                if (.not.all(ieee_is_finite([rhow,g,nu,d50,d90,z0,minh,max_dt,bed_fraction]))) &
                    call fail('nonfinite parameters')
                if (min(rhow,g,nu,d50,d90,z0,minh,max_dt,bed_fraction)<=0.d0.or. &
                    rhos<=rhow.or.d90<d50.or.bed_fraction>0.1d0.or.minh<braids_sediment_dry_tolerance()) &
                    call fail('invalid material/depth/timestep parameters')
                if (version==2) then
                    read(unit,*,iostat=ios) law,reconstruction,mpm_cf,mpm_critical
                    if (ios/=0) call fail('missing v2 law/reconstruction/MPM settings')
                    if (law<0.or.law>2.or.(reconstruction/=2.and.reconstruction/=5)) &
                        call fail('v2 laws: 0 none, 1 Van Rijn, 2 MPM; reconstruction 2 MC or 5 WENO-Z')
                    if (.not.all(ieee_is_finite([mpm_cf,mpm_critical])).or.min(mpm_cf,mpm_critical)<0.d0) &
                        call fail('invalid MPM settings')
                endif
                if (law==1.and.any(reference>0.d0)) then
                    if (any(abs(reference-[rhow,g,nu,d50,d90,z0])>1.d-12*abs(reference))) &
                        call fail('bedload and suspended Van Rijn material/roughness must agree')
                endif
                call braids_vr1993_grain_parameters(d50,rhos,rhow,g,nu,dstar,taucr)
                read(unit,*,iostat=ios) n
                if (ios/=0.or.n<0) call fail('invalid boundary face count')
                if (law==0.and.n/=0) call fail('no-bedload law requires zero bedload boundary faces')
                allocate(ids(n),sides(n),indices(n),modes(n),rates(n),inputs(n),exports(n))
                do k=1,n
                    read(unit,*,iostat=ios) ids(k),sides(k),indices(k),modes(k),rates(k)
                    if (ios/=0) call fail('invalid face row')
                    if (ids(k)<1.or.sides(k)<1.or.sides(k)>4.or.indices(k)<0.or. &
                        modes(k)<1.or.modes(k)>3.or..not.ieee_is_finite(rates(k))) call fail('invalid face policy')
                    if (rates(k)<0.d0.or.(modes(k)/=1.and.rates(k)/=0.d0)) call fail('invalid prescribed rate')
                    if (any(sides(:k-1)==sides(k).and.indices(:k-1)==indices(k))) call fail('duplicate boundary face')
                    if (any(ids(:k-1)==ids(k).and.(modes(:k-1)/=modes(k).or.rates(:k-1)/=rates(k)))) &
                        call fail('faces sharing an ID must use the same policy and rate')
                enddo
                close(unit)
            endif
            configured=.true.
        endif
        on=enabled
    end function

    logical function braids_bedload_ssprk() result(on)
        on=braids_bedload_enabled()
        on=on.and.reconstruction>0
    end function

    real(8) function braids_bedload_bulk() result(value)
        value=bulk
    end function

    subroutine capacity(h,hu,hv,qx,qy)
        real(8), intent(in) :: h,hu,hv
        real(8), intent(out) :: qx,qy
        real(8) :: us,mu,tr
        logical :: valid
        ! Keep RK hydraulic/scalar transport and suspended Exner exchange active.
        ! No interior or external bedload flux is permitted for this control.
        if (law==0) then
            qx=0.d0; qy=0.d0
            return
        endif
        if (law==2) then
            qx=0.d0; qy=0.d0
            if (h<=minh) return
            call braids_mpm_bedload(hu/h,hv/h,mpm_cf,d50,rhos/rhow-1.d0,g,mpm_critical,qx,qy,valid)
            if (.not.valid) call fail('invalid MPM hydraulic state')
            qx=qx*rhos; qy=qy*rhos
            return
        endif
        call braids_vr1993_bedload_cell(h,hu,hv,d50,d90,rhos,rhow,z0,dstar,taucr,minh,qx,qy,us,mu,tr,valid)
        if (.not.valid) call fail('invalid hydraulic state for bedload log law')
    end subroutine

    real(8) function braids_bedload_dt(level,nvar) result(limit)
        integer, intent(in) :: level,nvar
        integer :: k,id,nx,ny,mi,mj,i,j,p
        real(8) :: qx,qy,h,rate,maxqx,maxqy,hmin,hydro_rate
        limit=huge(1.d0)
        if (.not.braids_bedload_enabled()) return
        if (level/=1.or.mxnest/=1.or.mcapa/=0.or.coordinate_system/=1.or.xperdom.or.yperdom.or.nvar/=5) &
            call fail('requires five states on one Cartesian fixed level without periodicity/capacity')
        limit=max_dt; hydro_rate=0.d0
        maxqx=0.d0; maxqy=0.d0; hmin=huge(1.d0)
        ! A conservative gross-flux bound on geometric bed change. MC faces
        ! never exceed twice a cell capacity. Include imposed boundary supply.
        do k=1,numgrids(level)
            id=listOfGrids(listStart(level)+k-1)
            nx=node(ndihi,id)-node(ndilo,id)+1; ny=node(ndjhi,id)-node(ndjlo,id)+1
            mi=nx+2*nghost; mj=ny+2*nghost
            do j=nghost+1,ny+nghost
                do i=nghost+1,nx+nghost
                    p=node(store1,id)+nvar*((j-1)*mi+i-1)
                    h=alloc(p)
                    ! Hydraulic CFL includes shallow wet cells, independently of SedThr.
                    if (h>dry_tolerance) hydro_rate=max(hydro_rate, &
                        (abs(alloc(p+1)/h)+sqrt(g*h))/hxposs(level) &
                        +(abs(alloc(p+2)/h)+sqrt(g*h))/hyposs(level))
                    call capacity(h,alloc(p+1),alloc(p+2),qx,qy)
                    maxqx=max(maxqx,abs(qx)); maxqy=max(maxqy,abs(qy))
                    hmin=min(hmin,max(h,minh))
                enddo
            enddo
        enddo
        if (reconstruction>0.and.hydro_rate>0.d0) limit=min(limit,.35d0/hydro_rate)
        rate=4.d0*(maxqx/hxposs(level)+maxqy/hyposs(level))
        if (size(rates)>0) rate=rate+2.d0*maxval(rates)/min(hxposs(level),hyposs(level))
        if (rate>0.d0) limit=min(limit,bed_fraction*bulk*hmin/rate)
    end function

    subroutine braids_bedload_prepare(nv,mi,mj,mbc,time,dt,dx,dy,q,fm,fp,gm,gp,record,theta_out)
        integer, intent(in) :: nv,mi,mj,mbc
        real(8), intent(in) :: time,dt,dx,dy,q(nv,mi,mj)
        real(8), intent(inout) :: fm(nv,mi,mj),fp(nv,mi,mj),gm(nv,mi,mj),gp(nv,mi,mj)
        logical, optional, intent(in) :: record
        real(8), optional, intent(out) :: theta_out
        real(8), allocatable :: qx(:,:),qy(:,:),mass(:,:),fx(:,:),fy(:,:),next_mass(:,:)
        logical, allocatable :: wet(:,:)
        integer :: nx,ny,i,j,k,side,index
        real(8) :: inward,water,length,value,tolerance
        if (.not.braids_bedload_enabled()) return
        if (pending) call fail('uncommitted transport step')
        nx=mi-2*mbc; ny=mj-2*mbc
        if (nv/=5.or.nx/=iregsz(1).or.ny/=jregsz(1)) call fail('bedload preparation requires complete fixed level')
        allocate(qx(nx,ny),qy(nx,ny),mass(nx,ny),wet(nx,ny),fx(nx+1,ny),fy(nx,ny+1),next_mass(nx,ny))
        mass=q(5,mbc+1:mi-mbc,mbc+1:mj-mbc)
        if (minval(mass)<0.d0.or..not.all(ieee_is_finite(mass))) call fail('invalid bed inventory')
!$omp parallel do default(none) shared(nx,ny,mbc,q,qx,qy,wet,minh) private(i,j)
        do j=1,ny
            do i=1,nx
                wet(i,j)=q(1,i+mbc,j+mbc)>minh
                call capacity(q(1,i+mbc,j+mbc),q(2,i+mbc,j+mbc),q(3,i+mbc,j+mbc),qx(i,j),qy(i,j))
            enddo
        enddo
!$omp end parallel do
        fx=0.d0; fy=0.d0
        ! Faces not explicitly listed remain walls for bedload. Modes:
        ! 1 prescribed inward mass flux/width; 2 equilibrium feed; 3 free export.
        do k=1,size(ids)
            side=sides(k); index=indices(k)+1
            if ((side<=2.and.index>ny).or.(side>2.and.index>nx)) call fail('face outside domain')
            i=1; j=1; length=dy
            select case(side)
            case(1)
                j=index; inward=qx(i,j)
                water=q(2,mbc+i,mbc+j)+fp(1,mbc+i,mbc+j)
            case(2)
                i=nx; j=index; inward=-qx(i,j)
                water=-(q(2,mbc+i,mbc+j)+fm(1,mbc+i+1,mbc+j))
            case(3)
                i=index; inward=qy(i,j); length=dx
                water=q(3,mbc+i,mbc+j)+gp(1,mbc+i,mbc+j)
            case(4)
                i=index; j=ny; inward=-qy(i,j); length=dx
                water=-(q(3,mbc+i,mbc+j)+gm(1,mbc+i,mbc+j+1))
            end select
            value=0.d0
            if (wet(i,j)) then
                if (modes(k)==1.and.water>0.d0) value=rates(k)
                if (modes(k)==2.and.water>0.d0) value=max(inward,0.d0)
                if (modes(k)==3.and.water<0.d0) value=min(inward,0.d0)
            endif
            select case(side)
            case(1); fx(1,j)=value
            case(2); fx(nx+1,j)=-value
            case(3); fy(i,1)=value
            case(4); fy(i,ny+1)=-value
            end select
        enddo
        if (reconstruction>0) then
            call reconstructed_faces(nv,mi,mj,mbc,q,fx,fy)
            call braids_bedload_limit(nx,ny,dt,dx,dy,mass,fx,fy,min_theta)
        else
            call braids_bedload_faces(nx,ny,dt,dx,dy,mass,wet,qx,qy,fx,fy,min_theta)
        endif
        if (present(theta_out)) theta_out=min_theta
        next_mass=mass-dt/dx*(fx(2:,:)-fx(:nx,:))-dt/dy*(fy(:,2:)-fy(:,:ny))
        tolerance=256.d0*epsilon(1.d0)*max(1.d0,maxval(mass))
        if (minval(next_mass)<-tolerance) call fail('donor limiter produced negative bed')
        ! Limit the actual net geometric change too, including neighbor capacities.
        if (any(abs(next_mass-mass)>bed_fraction*bulk*max(q(1,mbc+1:mi-mbc,mbc+1:mj-mbc),minh) &
            *(1.d0+1.d-10))) call fail('bed change exceeds explicit limit; reduce dt_max')
        fm(5,mbc+1:mi-mbc+1,mbc+1:mj-mbc)=fx
        fp(5,mbc+1:mi-mbc+1,mbc+1:mj-mbc)=fx
        gm(5,mbc+1:mi-mbc,mbc+1:mj-mbc+1)=fy
        gp(5,mbc+1:mi-mbc,mbc+1:mj-mbc+1)=fy
        if (present(record)) then
            if (.not.record) return
        endif
        call braids_bedload_accept(nv,mi,mj,mbc,time,dt,dx,dy,q,fm,gm,min_theta)
    end subroutine

    subroutine braids_bedload_accept(nv,mi,mj,mbc,time,dt,dx,dy,q,fm,gm,theta)
        integer, intent(in) :: nv,mi,mj,mbc
        real(8), intent(in) :: time,dt,dx,dy,q(nv,mi,mj),fm(nv,mi,mj),gm(nv,mi,mj),theta
        integer :: k,index,il,ir,jb,jt
        real(8) :: length,value
        il=mbc+1;ir=mi-mbc;jb=mbc+1;jt=mj-mbc
        if (pending) call fail('duplicate accepted bedload interval')
        inputs=0.d0; exports=0.d0
        do k=1,size(ids)
            index=indices(k)+mbc+1; length=dy
            select case(sides(k))
            case(1); value=fm(5,il,index)
            case(2); value=-fm(5,ir+1,index)
            case(3); value=gm(5,index,jb); length=dx
            case(4); value=-gm(5,index,jt+1); length=dx
            end select
            inputs(k)=max(value,0.d0)*length*dt
            exports(k)=max(-value,0.d0)*length*dt
        enddo
        before=sum(q(5,il:ir,jb:jt))*dx*dy
        after=before-dt*dy*sum(fm(5,ir+1,jb:jt)-fm(5,il,jb:jt)) &
                    -dt*dx*sum(gm(5,il:ir,jt+1)-gm(5,il:ir,jb))
        min_theta=theta;start_time=time;step_dt=dt;pending=.true.
    end subroutine

    subroutine reconstructed_faces(nv,mi,mj,ng,q,fx,fy)
        integer, intent(in) :: nv,mi,mj,ng
        real(8), intent(in) :: q(nv,mi,mj)
        real(8), intent(inout) :: fx(mi-2*ng+1,mj-2*ng),fy(mi-2*ng,mj-2*ng+1)
        real(8) :: left(3),right(3),vl(5),vr(5),qlx,qly,qrx,qry
        integer :: i,j,m,k,ii,jj,nx,ny
        nx=mi-2*ng;ny=mj-2*ng
!$omp parallel do default(shared) private(i,j,m,k,ii,vl,vr,left,right,qlx,qly,qrx,qry)
        do j=1,ny
            do i=2,nx
                fx(i,j)=0.d0
                if(min(q(1,i+ng-1,j+ng),q(1,i+ng,j+ng))<=minh) cycle
                do m=1,3
                    do k=1,5
                        ii=max(1,min(mi,i+ng+k-4));vl(k)=q(m,ii,j+ng)
                        ii=max(1,min(mi,i+ng+3-k));vr(k)=q(m,ii,j+ng)
                    enddo
                    left(m)=reconstruct(vl,reconstruction);right(m)=reconstruct(vr,reconstruction)
                enddo
                call capacity(left(1),left(2),left(3),qlx,qly)
                call capacity(right(1),right(2),right(3),qrx,qry)
                if(qlx*qrx<0.d0) cycle
                if(qlx+qrx>=0.d0) fx(i,j)=qlx
                if(qlx+qrx<0.d0) fx(i,j)=qrx
            enddo
        enddo
!$omp end parallel do
!$omp parallel do default(shared) private(i,j,m,k,jj,vl,vr,left,right,qlx,qly,qrx,qry)
        do j=2,ny
            do i=1,nx
                fy(i,j)=0.d0
                if(min(q(1,i+ng,j+ng-1),q(1,i+ng,j+ng))<=minh) cycle
                do m=1,3
                    do k=1,5
                        jj=max(1,min(mj,j+ng+k-4));vl(k)=q(m,i+ng,jj)
                        jj=max(1,min(mj,j+ng+3-k));vr(k)=q(m,i+ng,jj)
                    enddo
                    left(m)=reconstruct(vl,reconstruction);right(m)=reconstruct(vr,reconstruction)
                enddo
                call capacity(left(1),left(2),left(3),qlx,qly)
                call capacity(right(1),right(2),right(3),qrx,qry)
                if(qly*qry<0.d0) cycle
                if(qly+qry>=0.d0) fy(i,j)=qly
                if(qly+qry<0.d0) fy(i,j)=qry
            enddo
        enddo
!$omp end parallel do
    end subroutine

    subroutine braids_bedload_bed_update(nv,mi,mj,mbc,dt,dx,dy,maux,aux,fm,fp,gm,gp,cached)
        integer, intent(in) :: nv,mi,mj,mbc,maux
        real(8), intent(in) :: dt,dx,dy,fm(nv,mi,mj),fp(nv,mi,mj),gm(nv,mi,mj),gp(nv,mi,mj)
        real(8), intent(inout) :: aux(maux,mi,mj)
        logical, intent(in) :: cached
        integer :: i,j
        ! Setup is completed serially by the preflight before any patch worker.
        if (.not.enabled) return
        if (.not.cached.or..not.pending) call fail('bedload requires accepted positivity cache')
        do j=mbc+1,mj-mbc
            do i=mbc+1,mi-mbc
                aux(1,i,j)=aux(1,i,j)-(dt/dx*(fm(5,i+1,j)-fp(5,i,j)) &
                    +dt/dy*(gm(5,i,j+1)-gp(5,i,j)))/bulk
            enddo
        enddo
    end subroutine

    subroutine open_ledger()
        integer :: old,temp,ios,id,k,expected_id
        real(8) :: t,dt,ended
        character(1024) :: line
        logical :: exists
        inquire(file='braids_bedload_steps.csv',exist=exists)
        if (start_time>0.d0) then
            if (.not.exists) call fail('restart requires bedload ledger prefix')
            open(newunit=old,file='braids_bedload_steps.csv',status='old',action='read')
            open(newunit=temp,status='scratch',action='readwrite')
            read(old,'(a)',iostat=ios) line
            if (ios/=0.or.trim(line)/=header) call fail('invalid bedload ledger header')
            ended=0.d0
            do
                read(old,'(a)',iostat=ios) line
                if (ios<0) exit
                read(line,*,iostat=ios) t,dt,id
                if (ios/=0.or.id/=0.or.dt<=0.d0) call fail('invalid bedload ledger interval')
                if (t+dt>start_time+1.d-10) exit
                if (abs(t-ended)>1.d-10) call fail('bedload ledger gap/duplicate')
                write(temp,'(a)') trim(line)
                ! Per-face records follow the global row, in configured order.
                do k=1,size(ids)
                    read(old,'(a)',iostat=ios) line
                    if (ios/=0) call fail('incomplete bedload boundary ledger')
                    read(line,*,iostat=ios) t,dt,expected_id
                    if (ios/=0.or.expected_id/=ids(k).or.abs(t-ended)>1.d-10) call fail('changed restart boundary table')
                    write(temp,'(a)') trim(line)
                enddo
                ended=t+dt
            enddo
            close(old)
            if (abs(ended-start_time)>1.d-10) call fail('missing checkpoint-aligned bedload prefix')
            open(newunit=ledger,file='braids_bedload_steps.csv',status='replace')
            write(ledger,'(a)') header
            rewind(temp)
            do
                read(temp,'(a)',iostat=ios) line
                if (ios<0) exit
                write(ledger,'(a)') trim(line)
            enddo
            close(temp)
        else
            open(newunit=ledger,file='braids_bedload_steps.csv',status='replace')
            write(ledger,'(a)') header
        endif
        last_end=start_time; ledger_open=.true.
    end subroutine

    subroutine braids_bedload_commit()
        integer :: k
        if (.not.enabled) return
        if (.not.pending) call fail('missing accepted bedload step')
        if (.not.ledger_open) call open_ledger()
        if (abs(start_time-last_end)>1.d-10) call fail('noncontiguous bedload step')
        write(ledger,'(2(es25.17e3,","),i0,5(",",es25.17e3))') &
            start_time,step_dt,0,sum(inputs),sum(exports),before,after,min_theta
        do k=1,size(ids)
            write(ledger,'(2(es25.17e3,","),i0,5(",",es25.17e3))') &
                start_time,step_dt,ids(k),inputs(k),exports(k),0.d0,0.d0,min_theta
        enddo
        flush(ledger)
        last_end=start_time+step_dt; pending=.false.
    end subroutine
end module
