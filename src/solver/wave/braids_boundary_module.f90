! Opt-in, fixed-grid exterior section inlets.  See braids_boundary.data v1.
! Subcritical traces retain outgoing interior Riemann information and solve
! one section elevation for the prescribed, time-integrated total discharge.
! The conservative numerical boundary flux is installed after step2.
module braids_boundary_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private
    public :: braids_boundary_enabled, braids_boundary_side_enabled
    public :: braids_boundary_prepare, braids_boundary_next_event
    public :: braids_boundary_apply, braids_boundary_flux, braids_sediment_boundary_flux
    public :: braids_bc_face, braids_boundary_commit
    public :: braids_boundary_begin_step
    type boundary_t
        integer :: id, mode, allocation, endpoint, dry_policy, super_policy, backflow_policy
        real(kind=8) :: value, datum, manning, slope, request=0.d0, target=0.d0, stage=0.d0
        real(kind=8), allocatable :: station(:), bed(:), times(:), discharge(:)
        integer :: sediment_mode=0,sediment_endpoint=1
        real(kind=8) :: sediment_current=0.d0,sediment_inflow=0.d0
        real(kind=8), allocatable :: sediment_times(:),sediment_values(:)
    end type
    type face_t
        integer :: boundary, side, index, branch=0
        real(kind=8) :: bed, length, interior(3)=0.d0, rminus=0.d0, target_u=0.d0
        real(kind=8) :: tangent=0.d0, trace(3)=0.d0
        real(kind=8) :: interior_s=0.d0
    end type
    type(boundary_t), allocatable, save :: boundaries(:)
    type(face_t), allocatable, save :: faces(:)
    integer, allocatable, save :: lookup(:,:)
    integer, save :: table_version=1
    integer, save :: grid_nx, grid_ny
    real(kind=8), save :: grid_x, grid_y, grid_dx, grid_dy
    logical, save :: ready=.false., enabled=.false., prepared=.false., accepted_active=.false.
    real(kind=8), save :: prepared_time=0.d0, prepared_dt=0.d0
    integer, save :: diagnostic_unit=0
    real(kind=8), save :: diagnostic_end=0.d0
    character(len=*), parameter :: diagnostic_header= &
        'time_s,dt_s,end_time_s,boundary_id,requested_Q_m3_s,state_Q_m3_s,target_eta_m,compatible_eta_m,'// &
        'wet_eta_min_m,wet_eta_max_m,subcritical_faces,dry_target_faces,supercritical_target_faces,closed_faces,'// &
        'max_outgoing_invariant_error_m_s,state_Q_absolute_error_m3_s,state_Q_relative_error,local_backflow_faces'
contains
    subroutine fail(message)
        character(len=*), intent(in) :: message
        write(6,*) 'BRAIDS boundary error: ',message
        error stop 1
    end subroutine

    subroutine next_line(unit,line)
        integer, intent(in) :: unit
        character(len=*), intent(out) :: line
        integer :: ios
        do
            read(unit,'(a)',iostat=ios) line
            if (ios/=0) call fail('unexpected end of braids_boundary.data')
            line=adjustl(line)
            if (len_trim(line)==0) cycle
            if (line(1:1)=='#') cycle
            return
        enddo
    end subroutine

    subroutine setup()
        integer :: unit,version,nb,nf,b,f,i,j,np,nk,ios,id,side,index,k_legacy,legacy_mode
        real(kind=8) :: bed,length,left_slope,right_slope
        character(len=2048) :: line
        logical :: exists
        if (ready) return
        inquire(file='braids_boundary.data',exist=exists)
        if (.not.exists) then
            ready=.true.
            return
        endif
        open(newunit=unit,file='braids_boundary.data',status='old',action='read',iostat=ios)
        if (ios/=0) call fail('cannot open braids_boundary.data')
        call next_line(unit,line); read(line,*,iostat=ios) version
        if (ios/=0) call fail('invalid format version')
        if (version/=1.and.version/=2) call fail('unsupported braids_boundary.data version')
        table_version=version
        call next_line(unit,line)
        read(line,*,iostat=ios) grid_nx,grid_ny,grid_x,grid_y,grid_dx,grid_dy
        if (ios/=0) call fail('invalid grid row')
        if (grid_nx<1.or.grid_ny<1.or.grid_dx<=0.d0.or.grid_dy<=0.d0) call fail('invalid grid dimensions')
        if (.not.all(ieee_is_finite([grid_x,grid_y,grid_dx,grid_dy]))) call fail('non-finite grid')
        call next_line(unit,line); read(line,*,iostat=ios) nb,nf
        if (ios/=0) call fail('invalid table sizes')
        if (nb<1.or.nf<1) call fail('empty boundary table')
        allocate(boundaries(nb),faces(nf),lookup(4,0:max(grid_nx,grid_ny)-1))
        lookup=0
        do b=1,nb
            call next_line(unit,line)
            read(line,*,iostat=ios) boundaries(b)%id,boundaries(b)%mode,boundaries(b)%allocation, &
                boundaries(b)%endpoint,boundaries(b)%dry_policy,boundaries(b)%super_policy, &
                boundaries(b)%backflow_policy,np,nk, &
                boundaries(b)%value,boundaries(b)%datum,boundaries(b)%manning,boundaries(b)%slope
            if (ios/=0) call fail('invalid boundary header')
            if (boundaries(b)%id<1.or.np<2.or.nk<1) call fail('invalid ID/profile/hydrograph size')
            if (any(boundaries(:b-1)%id==boundaries(b)%id)) call fail('duplicate boundary ID')
            if (boundaries(b)%mode<1.or.boundaries(b)%mode>6) call fail('unsupported boundary mode')
            if (version==1.and.boundaries(b)%mode>4) call fail('outlet modes require format v2')
            if (boundaries(b)%allocation<1.or.boundaries(b)%allocation>2) call fail('unsupported allocation')
            if (boundaries(b)%endpoint<1.or.boundaries(b)%endpoint>3) call fail('unsupported endpoint policy')
            if (boundaries(b)%dry_policy<0.or.boundaries(b)%dry_policy>1.or. &
                boundaries(b)%super_policy<0.or.boundaries(b)%super_policy>1) call fail('unsupported regime policy')
            if (boundaries(b)%backflow_policy<0.or.boundaries(b)%backflow_policy>1) &
                call fail('unsupported local backflow policy')
            if (.not.all(ieee_is_finite([boundaries(b)%value,boundaries(b)%datum, &
                boundaries(b)%manning,boundaries(b)%slope]))) call fail('non-finite boundary parameters')
            if (boundaries(b)%manning<=0.d0) call fail('Manning n must be positive')
            if (boundaries(b)%mode==1.and.boundaries(b)%slope<=0.d0) call fail('normal depth needs positive slope')
            if (boundaries(b)%mode==3.and.boundaries(b)%value<0.d0) call fail('negative prescribed depth')
            allocate(boundaries(b)%station(np),boundaries(b)%bed(np), &
                boundaries(b)%times(nk),boundaries(b)%discharge(nk))
            do i=1,np
                call next_line(unit,line)
                read(line,*,iostat=ios) boundaries(b)%station(i),boundaries(b)%bed(i)
                if (ios/=0) call fail('invalid section profile row')
            enddo
            if (.not.all(ieee_is_finite(boundaries(b)%station)).or. &
                .not.all(ieee_is_finite(boundaries(b)%bed))) call fail('non-finite profile')
            if (any(boundaries(b)%station(2:)<=boundaries(b)%station(:np-1))) call fail('stations must increase')
            ! Initial native scope: convex, connected single-basin sections.
            do i=2,np-1
                left_slope=(boundaries(b)%bed(i)-boundaries(b)%bed(i-1))/ &
                    (boundaries(b)%station(i)-boundaries(b)%station(i-1))
                right_slope=(boundaries(b)%bed(i+1)-boundaries(b)%bed(i))/ &
                    (boundaries(b)%station(i+1)-boundaries(b)%station(i))
                if (right_slope<left_slope-1.d-10*max(1.d0,abs(left_slope),abs(right_slope))) &
                    call fail('native v1 requires a convex single-basin section')
            enddo
            do i=1,nk
                call next_line(unit,line)
                read(line,*,iostat=ios) boundaries(b)%times(i),boundaries(b)%discharge(i)
                if (ios/=0) call fail('invalid hydrograph row')
            enddo
            if (.not.all(ieee_is_finite(boundaries(b)%times)).or. &
                .not.all(ieee_is_finite(boundaries(b)%discharge))) call fail('non-finite hydrograph')
            if (any(boundaries(b)%discharge<0.d0)) call fail('negative discharge is not an inlet')
            if (boundaries(b)%mode>=5.and.any(boundaries(b)%discharge/=0.d0)) &
                call fail('outlet discharge is computed, not prescribed')
            if (any(boundaries(b)%times(2:)<=boundaries(b)%times(:nk-1))) call fail('times must increase')
        enddo
        do f=1,nf
            call next_line(unit,line); read(line,*,iostat=ios) id,side,index,bed,length
            if (ios/=0) call fail('invalid whole-face row')
            b=0
            do j=1,nb
                if (boundaries(j)%id==id) b=j
            enddo
            if (b==0.or.side<1.or.side>4) call fail('unknown boundary ID or side')
            if (index<0) call fail('negative global face index')
            if (side<=2) then
                if (index>=grid_ny) call fail('face outside grid')
                if (abs(length-grid_dy)>1.d-10*grid_dy) call fail('partial transverse face is unsupported')
            else
                if (index>=grid_nx) call fail('face outside grid')
                if (abs(length-grid_dx)>1.d-10*grid_dx) call fail('partial transverse face is unsupported')
            endif
            if (.not.all(ieee_is_finite([bed,length]))) call fail('non-finite face')
            if (lookup(side,index)/=0) call fail('overlapping boundary faces')
            faces(f)%boundary=b; faces(f)%side=side; faces(f)%index=index
            faces(f)%bed=bed; faces(f)%length=length; lookup(side,index)=f
        enddo
        do b=1,nb
            if (.not.any(faces%boundary==b)) call fail('boundary has no faces')
            side=0
            do f=1,nf
                if (faces(f)%boundary/=b) cycle
                if (boundaries(b)%mode<=4.and.side/=0.and.faces(f)%side/=side) &
                    call fail('one inlet section cannot span multiple exterior sides')
                side=faces(f)%side
            enddo
        enddo
        close(unit)
        if (table_version==2) then
            inquire(file='hydraulic_bc.data',exist=exists)
            if (exists) then
                open(newunit=unit,file='hydraulic_bc.data',status='old',action='read')
                do k_legacy=1,4
                    call next_line(unit,line);read(line,*,iostat=ios) side,legacy_mode
                    if (ios/=0.or.legacy_mode/=0) call fail('polygon v2 rejects legacy whole-side hydraulic overrides')
                enddo
                close(unit)
            endif
        endif
        call setup_sediment()
        enabled=.true.; ready=.true.
    end subroutine

    subroutine setup_sediment()
        integer :: unit,ios,version,nb,row,b,id,mode,endpoint,nk,i
        character(len=2048) :: line
        logical :: exists
        inquire(file='braids_sediment_boundary.data',exist=exists)
        if (.not.exists) return
        open(newunit=unit,file='braids_sediment_boundary.data',status='old',action='read',iostat=ios)
        if (ios/=0) call fail('cannot open braids_sediment_boundary.data')
        call next_line(unit,line); read(line,*,iostat=ios) version
        if (ios/=0.or.version/=1) call fail('unsupported sediment boundary format')
        call next_line(unit,line); read(line,*,iostat=ios) nb
        if (ios/=0.or.nb<0.or.nb>size(boundaries)) call fail('invalid sediment boundary count')
        do row=1,nb
            call next_line(unit,line); read(line,*,iostat=ios) id,mode,endpoint,nk
            if (ios/=0.or.mode<1.or.mode>2.or.endpoint<1.or.endpoint>3.or.nk<1) &
                call fail('invalid sediment boundary header')
            b=0
            do i=1,size(boundaries)
                if (boundaries(i)%id==id) b=i
            enddo
            if (b==0) call fail('sediment boundary ID has no hydraulic boundary')
            if (boundaries(b)%sediment_mode/=0) call fail('duplicate sediment boundary ID')
            boundaries(b)%sediment_mode=mode; boundaries(b)%sediment_endpoint=endpoint
            allocate(boundaries(b)%sediment_times(nk),boundaries(b)%sediment_values(nk))
            do i=1,nk
                call next_line(unit,line)
                read(line,*,iostat=ios) boundaries(b)%sediment_times(i),boundaries(b)%sediment_values(i)
                if (ios/=0) call fail('invalid sediment boundary series row')
            enddo
            if (.not.all(ieee_is_finite(boundaries(b)%sediment_times)).or. &
                .not.all(ieee_is_finite(boundaries(b)%sediment_values)).or. &
                any(boundaries(b)%sediment_values<0.d0)) call fail('invalid sediment boundary series')
            if (any(boundaries(b)%sediment_times(2:)<=boundaries(b)%sediment_times(:nk-1))) &
                call fail('sediment boundary times must increase')
        enddo
        close(unit)
    end subroutine setup_sediment

    logical function braids_boundary_enabled()
!$OMP CRITICAL (BraidsBoundarySetup)
        call setup()
!$OMP END CRITICAL (BraidsBoundarySetup)
        braids_boundary_enabled=enabled
    end function

    logical function braids_boundary_side_enabled(side)
        integer, intent(in) :: side
        braids_boundary_side_enabled=.false.
        if (.not.braids_boundary_enabled()) return
        if (side<1.or.side>4) return
        braids_boundary_side_enabled=any(lookup(side,:)>0)
    end function braids_boundary_side_enabled

    integer function first_at_or_after(values,value) result(index)
        real(kind=8), intent(in) :: values(:),value
        integer :: low,high,middle
        low=1; high=size(values)
        do while(low<high)
            middle=low+(high-low)/2
            if (value<=values(middle)) then
                high=middle
            else
                low=middle+1
            endif
        enddo
        index=low
    end function first_at_or_after

    integer function first_after(values,value) result(index)
        real(kind=8), intent(in) :: values(:),value
        integer :: low,high,middle,n
        n=size(values); low=1; high=n+1
        do while(low<high)
            middle=low+(high-low)/2
            if (middle>n) then
                high=middle
            else if (values(middle)<=value) then
                low=middle+1
            else
                high=middle
            endif
        enddo
        index=low
    end function first_after

    real(kind=8) function point_sediment(b,time) result(value)
        integer, intent(in) :: b
        real(kind=8), intent(in) :: time
        integer :: i,n
        real(kind=8) :: w,tol
        if (boundaries(b)%sediment_mode==0) then
            value=0.d0; return
        endif
        n=size(boundaries(b)%sediment_times); tol=1.d-12*max(1.d0,abs(time))
        if (time<boundaries(b)%sediment_times(1)-tol.or. &
            time>boundaries(b)%sediment_times(n)+tol) then
            select case(boundaries(b)%sediment_endpoint)
            case(1)
                if (time<boundaries(b)%sediment_times(1)) then
                    value=boundaries(b)%sediment_values(1)
                else
                    value=boundaries(b)%sediment_values(n)
                endif
            case(2)
                value=0.d0
            case default
                call fail('time outside sediment boundary series')
            end select
            return
        endif
        if (n==1.or.time<=boundaries(b)%sediment_times(1)) then
            value=boundaries(b)%sediment_values(1); return
        endif
        i=max(1,first_at_or_after(boundaries(b)%sediment_times,time)-1)
        w=(time-boundaries(b)%sediment_times(i))/ &
            (boundaries(b)%sediment_times(i+1)-boundaries(b)%sediment_times(i))
        value=(1.d0-w)*boundaries(b)%sediment_values(i)+ &
            w*boundaries(b)%sediment_values(i+1); return
        value=boundaries(b)%sediment_values(n)
    end function point_sediment

    real(kind=8) function mean_sediment(b,time,dt) result(value)
        integer, intent(in) :: b
        real(kind=8), intent(in) :: time,dt
        real(kind=8) :: a,z,finish,middle,total,va,vz
        integer :: i,n
        if (boundaries(b)%sediment_mode==0) then
            value=0.d0; return
        endif
        if (dt<=0.d0) then
            value=point_sediment(b,time); return
        endif
        finish=time+dt; a=time; total=0.d0; n=size(boundaries(b)%sediment_times)
        do while(a<finish)
            i=first_after(boundaries(b)%sediment_times,a); z=finish
            if (i<=n) z=min(z,boundaries(b)%sediment_times(i))
            middle=0.5d0*(a+z)
            if (boundaries(b)%sediment_endpoint==2.and. &
                (middle<boundaries(b)%sediment_times(1).or. &
                 middle>boundaries(b)%sediment_times(n))) then
                va=0.d0; vz=0.d0
            else
                va=point_sediment(b,a); vz=point_sediment(b,z)
            endif
            total=total+0.5d0*(va+vz)*(z-a); a=z
        enddo
        value=total/dt
    end function mean_sediment

    real(kind=8) function point_q(b,time) result(q)
        integer, intent(in) :: b
        real(kind=8), intent(in) :: time
        integer :: i,n
        real(kind=8) :: w,tol
        n=size(boundaries(b)%times); tol=1.d-12*max(1.d0,abs(time))
        if (time<boundaries(b)%times(1)-tol.or.time>boundaries(b)%times(n)+tol) then
            select case(boundaries(b)%endpoint)
            case(1)
                if (time<boundaries(b)%times(1)) then
                    q=boundaries(b)%discharge(1)
                else
                    q=boundaries(b)%discharge(n)
                endif
            case(2)
                q=0.d0
            case default
                call fail('time outside hydrograph with endpoint=error')
            end select
            return
        endif
        if (n==1.or.time<=boundaries(b)%times(1)) then
            q=boundaries(b)%discharge(1); return
        endif
        i=max(1,first_at_or_after(boundaries(b)%times,time)-1)
        w=(time-boundaries(b)%times(i))/(boundaries(b)%times(i+1)-boundaries(b)%times(i))
        q=(1.d0-w)*boundaries(b)%discharge(i)+w*boundaries(b)%discharge(i+1); return
        q=boundaries(b)%discharge(n)
    end function

    real(kind=8) function mean_q(b,time,dt) result(q)
        integer, intent(in) :: b
        real(kind=8), intent(in) :: time,dt
        real(kind=8) :: a,z,finish,middle,total,qa,qz
        integer :: i,n
        if (dt<=0.d0) then
            q=point_q(b,time); return
        endif
        finish=time+dt; a=time; total=0.d0; n=size(boundaries(b)%times)
        do while(a<finish)
            i=first_after(boundaries(b)%times,a); z=finish
            if (i<=n) z=min(z,boundaries(b)%times(i))
            middle=0.5d0*(a+z)
            ! The zero endpoint policy has jumps at the end knots: use the
            ! side of the interval, not the endpoint value, for its integral.
            if (boundaries(b)%endpoint==2.and. &
                (middle<boundaries(b)%times(1).or.middle>boundaries(b)%times(n))) then
                qa=0.d0; qz=0.d0
            else
                qa=point_q(b,a); qz=point_q(b,z)
            endif
            total=total+0.5d0*(qa+qz)*(z-a); a=z
        enddo
        q=total/dt
    end function

    real(kind=8) function braids_boundary_next_event(time) result(event)
        real(kind=8), intent(in) :: time
        real(kind=8) :: tol
        integer :: b,i
        event=huge(1.d0)
        if (.not.braids_boundary_enabled()) return
        tol=1.d-12*max(1.d0,abs(time))
        do b=1,size(boundaries)
            i=first_after(boundaries(b)%times,time+tol)
            if (i<=size(boundaries(b)%times)) event=min(event,boundaries(b)%times(i))
        enddo
    end function

    subroutine geometry(b,stage,area,perimeter,width)
        integer, intent(in) :: b
        real(kind=8), intent(in) :: stage
        real(kind=8), intent(out) :: area,perimeter,width
        real(kind=8) :: dx,dz,h1,h2,fraction
        integer :: i
        area=0.d0; perimeter=0.d0; width=0.d0
        do i=1,size(boundaries(b)%bed)-1
            dx=boundaries(b)%station(i+1)-boundaries(b)%station(i)
            dz=boundaries(b)%bed(i+1)-boundaries(b)%bed(i)
            h1=stage-boundaries(b)%bed(i); h2=stage-boundaries(b)%bed(i+1)
            if (max(h1,h2)<=0.d0) cycle
            if (min(h1,h2)>=0.d0) then
                fraction=1.d0; area=area+0.5d0*(h1+h2)*dx
            else
                fraction=max(h1,h2)/abs(dz)
                area=area+0.5d0*max(h1,h2)*fraction*dx
            endif
            perimeter=perimeter+fraction*sqrt(dx*dx+dz*dz); width=width+fraction*dx
        enddo
    end subroutine

    real(kind=8) function target_stage(b,q) result(stage)
        use geoclaw_module, only: grav
        integer, intent(in) :: b
        real(kind=8), intent(in) :: q
        real(kind=8) :: lo,hi,area,perimeter,width,capacity
        integer :: i,n
        n=size(boundaries(b)%bed)
        lo=minval(boundaries(b)%bed); hi=min(boundaries(b)%bed(1),boundaries(b)%bed(n))
        select case(boundaries(b)%mode)
        case(3)
            stage=boundaries(b)%datum+boundaries(b)%value
            if (stage>hi+1.d-10) call fail('prescribed depth exceeds surveyed section banks')
            return
        case(4)
            stage=boundaries(b)%value
            if (stage>hi+1.d-10) call fail('prescribed stage exceeds surveyed section banks')
            return
        end select
        if (q==0.d0) then
            stage=lo; return
        endif
        if (hi<=lo) call fail('normal/critical section requires enclosing banks above its minimum')
        call geometry(b,hi,area,perimeter,width)
        if (boundaries(b)%mode==1) then
            capacity=area*(area/perimeter)**(2.d0/3.d0)*sqrt(boundaries(b)%slope)/boundaries(b)%manning
        else
            capacity=sqrt(grav*area**3/width)
        endif
        if (q>capacity*(1.d0+1.d-12)) call fail('normal/critical target exceeds surveyed section banks')
        do i=1,90
            stage=0.5d0*(lo+hi); call geometry(b,stage,area,perimeter,width)
            if (area<=0.d0) then
                capacity=0.d0
            else if (boundaries(b)%mode==1) then
                capacity=area*(area/perimeter)**(2.d0/3.d0)*sqrt(boundaries(b)%slope)/boundaries(b)%manning
            else
                capacity=sqrt(grav*area**3/width)
            endif
            if (capacity<q) then
                lo=stage
            else
                hi=stage
            endif
        enddo
        stage=0.5d0*(lo+hi)
    end function

    integer function normal_component(side) result(component)
        integer, intent(in) :: side
        component=2
        if (side>2) component=3
    end function

    real(kind=8) function inward_sign(side) result(s)
        integer, intent(in) :: side
        s=1.d0
        if (side==2.or.side==4) s=-1.d0
    end function

    subroutine braids_boundary_prepare(level,time,dt,nvar,naux,rk_stage)
        use amr_module, only: mxnest,nghost,node,alloc,numgrids, &
            listStart,listOfGrids,ndilo,ndihi,ndjlo,ndjhi,store1,storeaux, &
            hxposs,hyposs,xlower,ylower,xupper,yupper,mthbc,xperdom,yperdom
        use geoclaw_module, only: coordinate_system,grav,dry_tolerance,use_fwave_positivity_limiter,speed_limit
        logical, optional, intent(in) :: rk_stage
        integer, intent(in) :: level,nvar,naux
        real(kind=8), intent(in) :: time,dt
        integer :: f,b,k,p,i,j,nx,ny,mitot,locq,loca,offset,owners,component,pass
        real(kind=8) :: h,u,c,area,weight,lo,hi,stage,qtest,desired,tol,bed,flow_scale,residual_tolerance
        real(kind=8) :: sediment_q
        logical :: inside,changed
        if (.not.braids_boundary_enabled()) return
        if (table_version==2.and.any(mthbc/=0)) call fail('polygon v2 requires user code 0 on all sides')
        if (accepted_active) then
            if (.not.present(rk_stage)) call fail('trace preparation during an accepted physical update')
            if (.not.rk_stage) call fail('trace preparation outside RK stage')
        endif
        if (level/=1.or.mxnest/=1.or.coordinate_system/=1.or.use_fwave_positivity_limiter) &
            call fail('v1 requires Cartesian fixed level 1 and original water fluctuation path')
        if ((nvar<3.or.nvar>7).or.naux<1) &
            call fail('v1 supports water, sediment and optional turbulence states')
        if (.not.all(ieee_is_finite([time,dt])).or.dt<0.d0) call fail('invalid preparation time step')
        tol=1.d-10*max(1.d0,abs(grid_x),abs(grid_y),grid_dx,grid_dy)
        if (maxval(abs([xlower-grid_x,ylower-grid_y,xupper-grid_x-grid_nx*grid_dx, &
            yupper-grid_y-grid_ny*grid_dy,hxposs(level)-grid_dx,hyposs(level)-grid_dy]))>tol) &
            call fail('boundary table does not match native Cartesian grid')
        ! Gather by global face key, independently of patch traversal order.
        do f=1,size(faces)
            if (mthbc(faces(f)%side)/=0) call fail('configured section side must use user boundary code 0')
            if ((faces(f)%side<=2.and.xperdom).or.(faces(f)%side>2.and.yperdom)) &
                call fail('periodic side cannot own an inlet')
            owners=0
            do k=1,numgrids(level)
                p=listOfGrids(listStart(level)+k-1)
                nx=node(ndihi,p)-node(ndilo,p)+1; ny=node(ndjhi,p)-node(ndjlo,p)+1
                inside=.false.
                select case(faces(f)%side)
                case(1,2)
                    inside=faces(f)%index>=node(ndjlo,p).and.faces(f)%index<=node(ndjhi,p)
                    if (faces(f)%side==1) then
                        inside=inside.and.node(ndilo,p)==0; i=nghost+1
                    else
                        inside=inside.and.node(ndihi,p)==grid_nx-1; i=nghost+nx
                    endif
                    j=nghost+1+faces(f)%index-node(ndjlo,p)
                case(3,4)
                    inside=faces(f)%index>=node(ndilo,p).and.faces(f)%index<=node(ndihi,p)
                    if (faces(f)%side==3) then
                        inside=inside.and.node(ndjlo,p)==0; j=nghost+1
                    else
                        inside=inside.and.node(ndjhi,p)==grid_ny-1; j=nghost+ny
                    endif
                    i=nghost+1+faces(f)%index-node(ndilo,p)
                end select
                if (.not.inside) cycle
                owners=owners+1; mitot=nx+2*nghost
                offset=(j-1)*mitot+i-1; locq=node(store1,p)+nvar*offset
                loca=node(storeaux,p)+naux*offset
                faces(f)%interior=alloc(locq:locq+2); bed=alloc(loca)
                faces(f)%interior_s=0.d0
                if (nvar>=4) faces(f)%interior_s=alloc(locq+3)
                if (nvar>=5) then
                    ! Exner updates the adjacent face bed between steps.
                    faces(f)%bed=bed
                else if (abs(bed-faces(f)%bed)>1.d-8*max(1.d0,abs(bed),abs(faces(f)%bed))) then
                    call fail('declared face bed differs from native adjacent-cell bed')
                endif
            enddo
            if (owners/=1) call fail('each configured face must have exactly one native patch owner')
            if (.not.all(ieee_is_finite(faces(f)%interior)).or.faces(f)%interior(1)<0.d0) &
                call fail('invalid interior conservative state')
        enddo
        do b=1,size(boundaries)
            if (boundaries(b)%mode>=5) then
                call prepare_outlet(b)
                cycle
            endif
            desired=mean_q(b,time,dt); boundaries(b)%request=desired
            boundaries(b)%target=target_stage(b,desired)
            area=0.d0
            do f=1,size(faces)
                if (faces(f)%boundary/=b) cycle
                h=max(boundaries(b)%target-faces(f)%bed,0.d0)
                weight=1.d0
                if (boundaries(b)%allocation==2) weight=h**(2.d0/3.d0)/boundaries(b)%manning
                faces(f)%target_u=weight; area=area+h*faces(f)%length*weight
            enddo
            if (desired>0.d0.and.area<=0.d0) call fail('positive discharge has no discrete wet face area')
            do f=1,size(faces)
                if (faces(f)%boundary/=b) cycle
                if (desired>0.d0) then
                    faces(f)%target_u=faces(f)%target_u*desired/area
                else
                    faces(f)%target_u=0.d0
                endif
                h=faces(f)%interior(1); component=normal_component(faces(f)%side)
                if (desired==0.d0) then
                    faces(f)%branch=0; faces(f)%trace=0.d0
                    if (h>dry_tolerance) then
                        u=inward_sign(faces(f)%side)*faces(f)%interior(component)/h
                        c=sqrt(grav*h)
                        faces(f)%trace(1)=max(c-0.5d0*u,0.d0)**2/grav
                        faces(f)%trace(5-component)=faces(f)%trace(1)*faces(f)%interior(5-component)/h
                    endif
                    cycle
                endif
                if (h<=dry_tolerance) then
                    faces(f)%branch=2; faces(f)%tangent=0.d0
                    if (boundaries(b)%dry_policy==0.and.boundaries(b)%target>faces(f)%bed) &
                        call fail('dry incoming face requires explicit target policy')
                    cycle
                endif
                u=inward_sign(faces(f)%side)*faces(f)%interior(component)/h; c=sqrt(grav*h)
                faces(f)%rminus=u-2.d0*c
                ! Tangential information enters with positive inflow and is
                ! prescribed as the section target (zero), not extrapolated.
                faces(f)%tangent=0.d0
                if (u<=-c) call fail('supercritical outward flow is not an admissible inlet')
                if (u>=c) then
                    if (boundaries(b)%super_policy==0) call fail('supercritical face requires explicit target policy')
                    faces(f)%branch=2
                else
                    faces(f)%branch=1
                endif
            enddo
            if (desired==0.d0) then
                boundaries(b)%stage=boundaries(b)%target; cycle
            endif
            ! Transition faces to the explicit incoming target branch only;
            ! never silently discard outgoing information on subcritical faces.
            do pass=1,size(faces)+1
                lo=huge(1.d0)
                do f=1,size(faces)
                    if (faces(f)%boundary==b) lo=min(lo,faces(f)%bed)
                enddo
                hi=min(boundaries(b)%bed(1),boundaries(b)%bed(size(boundaries(b)%bed)))
                if (hi<=lo.or.section_q(b,hi)<desired) &
                    call fail('no compatible discharge within surveyed section banks')
                do i=1,100
                    stage=0.5d0*(lo+hi); qtest=section_q(b,stage)
                    if (qtest<desired) then
                        lo=stage
                    else
                        hi=stage
                    endif
                enddo
                stage=0.5d0*(lo+hi); changed=.false.
                do f=1,size(faces)
                    if (faces(f)%boundary/=b) cycle
                    h=max(stage-faces(f)%bed,0.d0)
                    if (h<=dry_tolerance) cycle
                    if (faces(f)%branch==2.and.faces(f)%interior(1)<=dry_tolerance.and. &
                        boundaries(b)%dry_policy==0) call fail('compatible stage wets a face without dry target policy')
                    u=face_velocity(f,h)
                    if (faces(f)%branch==1.and.u>=sqrt(grav*h)*(1.d0-1.d-12)) then
                        if (boundaries(b)%super_policy==0) call fail('compatible trace crosses critical flow')
                        faces(f)%branch=2; changed=.true.
                    endif
                enddo
                if (.not.changed) exit
            enddo
            if (changed) call fail('regime selection did not settle')
            ! Choose the closest representable bracket endpoint. At terrain
            ! elevations of thousands of metres, stage subtraction can set the
            ! discharge residual floor even after bisection has converged.
            if (abs(section_q(b,lo)-desired)<abs(section_q(b,stage)-desired)) stage=lo
            if (abs(section_q(b,hi)-desired)<abs(section_q(b,stage)-desired)) stage=hi
            boundaries(b)%stage=stage
            flow_scale=0.d0
            do f=1,size(faces)
                if (faces(f)%boundary/=b) cycle
                h=max(stage-faces(f)%bed,0.d0); u=face_velocity(f,h)
                if (h>dry_tolerance.and.faces(f)%branch==2.and.faces(f)%interior(1)>dry_tolerance) then
                    if (u<sqrt(grav*h)*(1.d0-1.d-10)) &
                        call fail('wet target trace is subcritical; supercritical-to-subcritical transition is unsupported')
                endif
                if (h>0.d0.and.u<=-sqrt(grav*h)) call fail('supercritical outward trace is not an admissible inlet')
                if (u<0.d0) then
                    if (boundaries(b)%backflow_policy==0) call fail('local reverse discharge requires explicit allow policy')
                    if (faces(f)%interior(1)<=dry_tolerance) call fail('local backflow requires a wet interior state')
                    component=normal_component(faces(f)%side)
                    faces(f)%tangent=faces(f)%interior(5-component)/faces(f)%interior(1)
                else
                    faces(f)%tangent=0.d0
                endif
                if (sqrt(u*u+faces(f)%tangent**2)>speed_limit) &
                    call fail('boundary trace exceeds native speed limiter; CFL trace would change')
                faces(f)%trace(1)=h; component=normal_component(faces(f)%side)
                faces(f)%trace(component)=h*u*inward_sign(faces(f)%side)
                faces(f)%trace(5-component)=h*faces(f)%tangent
                flow_scale=flow_scale+h*faces(f)%length*(abs(faces(f)%rminus)+sqrt(grav*h)+abs(u))
            enddo
            ! Account for characteristic subtraction conditioning, while
            ! retaining a strict relative ceiling for small nonzero Q.
            residual_tolerance=min(1.d-8*desired,max(1.d-12*desired,32.d0*epsilon(1.d0)*flow_scale, &
                abs(section_q(b,nearest(stage,1.d0))-section_q(b,stage)), &
                abs(section_q(b,nearest(stage,-1.d0))-section_q(b,stage))))
            if (abs(section_q(b,stage)-desired)>residual_tolerance) &
                call fail('compatible section discharge residual exceeds solver tolerance')
        enddo
        do b=1,size(boundaries)
            boundaries(b)%sediment_current=mean_sediment(b,time,dt)
            boundaries(b)%sediment_inflow=0.d0
            do f=1,size(faces)
                if (faces(f)%boundary/=b) cycle
                component=normal_component(faces(f)%side)
                sediment_q=inward_sign(faces(f)%side)*faces(f)%trace(component)*faces(f)%length
                boundaries(b)%sediment_inflow=boundaries(b)%sediment_inflow+max(sediment_q,0.d0)
            enddo
            if (boundaries(b)%sediment_mode==2.and.boundaries(b)%sediment_current>0.d0.and. &
                boundaries(b)%sediment_inflow<=0.d0) &
                call fail('positive sediment mass load requires inward water flow')
        enddo
        prepared_time=time; prepared_dt=dt; prepared=.true.
    end subroutine

    subroutine prepare_outlet(b)
        use geoclaw_module, only: grav,dry_tolerance,speed_limit
        integer, intent(in) :: b
        integer :: f,component
        real(kind=8) :: h,u,c,hb,ub,tangent
        boundaries(b)%request=0.d0 ! No imposed Q; accepted face ledger measures export.
        boundaries(b)%target=boundaries(b)%value
        boundaries(b)%stage=boundaries(b)%value
        do f=1,size(faces)
            if (faces(f)%boundary/=b) cycle
            component=normal_component(faces(f)%side)
            h=faces(f)%interior(1);faces(f)%trace=0.d0;faces(f)%branch=0
            if (h<=dry_tolerance) then
                if (boundaries(b)%mode==5.and.boundaries(b)%value>faces(f)%bed+dry_tolerance) &
                    call fail('stage outlet wetting a dry cell requires a separate inundation policy')
                cycle
            endif
            u=inward_sign(faces(f)%side)*faces(f)%interior(component)/h
            c=sqrt(grav*h);faces(f)%rminus=u-2.d0*c
            if (u<=-c) then
                ! Both gravity characteristics leave: the interior controls the trace.
                faces(f)%trace=faces(f)%interior;faces(f)%branch=2
                cycle
            endif
            if (boundaries(b)%mode==6) call fail('supercritical outlet encountered subcritical flow')
            if (u>=c) call fail('supercritical incoming outlet requires an inlet state')
            hb=max(boundaries(b)%value-faces(f)%bed,0.d0)
            if (hb<=dry_tolerance) call fail('wet stage outlet cannot impose a dry trace; use a critical control')
            ub=faces(f)%rminus+2.d0*sqrt(grav*hb)
            if (abs(ub)>=sqrt(grav*hb)) call fail('stage outlet trace crosses critical flow')
            if (ub>0.d0.and.boundaries(b)%backflow_policy==0) &
                call fail('outlet reversal requires explicit allow and incoming sediment policies')
            tangent=faces(f)%interior(5-component)/h
            if (ub>0.d0) then
                tangent=0.d0
                if (boundaries(b)%sediment_mode/=1) &
                    call fail('outlet reversal requires explicit incoming concentration, including clear water')
            endif
            if (sqrt(ub*ub+tangent*tangent)>speed_limit) call fail('outlet trace exceeds speed limit')
            faces(f)%branch=1;faces(f)%trace(1)=hb
            faces(f)%trace(component)=hb*ub*inward_sign(faces(f)%side)
            faces(f)%trace(5-component)=hb*tangent
        enddo
    end subroutine prepare_outlet

    real(kind=8) function face_velocity(f,h) result(u)
        use geoclaw_module, only: grav
        integer, intent(in) :: f
        real(kind=8), intent(in) :: h
        u=faces(f)%target_u
        if (h<=0.d0) then
            u=0.d0
        else if (faces(f)%branch==1) then
            u=faces(f)%rminus+2.d0*sqrt(grav*h)
        endif
    end function

    real(kind=8) function section_q(b,stage) result(q)
        integer, intent(in) :: b
        real(kind=8), intent(in) :: stage
        integer :: f
        real(kind=8) :: h
        q=0.d0
        do f=1,size(faces)
            if (faces(f)%boundary/=b) cycle
            h=max(stage-faces(f)%bed,0.d0)
            q=q+h*face_velocity(f,h)*faces(f)%length
        enddo
    end function

    subroutine braids_bc_face(side,index,bc_id,requested_face_discharge,state)
        integer, intent(in) :: side,index
        integer, intent(out) :: bc_id
        real(kind=8), intent(out) :: requested_face_discharge
        real(kind=8), optional, intent(out) :: state(3)
        integer :: f,component
        bc_id=0; requested_face_discharge=0.d0
        if (present(state)) state=0.d0
        if (.not.braids_boundary_enabled()) return
        if (side<1.or.side>4.or.index<0.or.index>=size(lookup,2)) return
        f=lookup(side,index)
        if (f==0) return
        bc_id=boundaries(faces(f)%boundary)%id
        if (.not.prepared) return
        component=normal_component(side)
        requested_face_discharge=inward_sign(side)*faces(f)%trace(component)*faces(f)%length
        if (present(state)) state=faces(f)%trace
    end subroutine

    real(kind=8) function sediment_concentration(f) result(c)
        use geoclaw_module, only: dry_tolerance
        integer, intent(in) :: f
        integer :: b,component
        real(kind=8) :: inward_q,tolerance
        b=faces(f)%boundary; component=normal_component(faces(f)%side)
        inward_q=inward_sign(faces(f)%side)*faces(f)%trace(component)*faces(f)%length
        if (inward_q>=0.d0) then
            select case(boundaries(b)%sediment_mode)
            case(1)
                c=boundaries(b)%sediment_current
            case(2)
                if (boundaries(b)%sediment_inflow>0.d0) then
                    c=boundaries(b)%sediment_current/boundaries(b)%sediment_inflow
                else
                    c=0.d0
                endif
            case default
                c=0.d0
            end select
        else
            tolerance=256.d0*epsilon(1.d0)*max(1.d0,abs(faces(f)%interior_s))
            if (faces(f)%interior(1)<=dry_tolerance) then
                if (abs(faces(f)%interior_s)>tolerance) &
                    call fail('dry outward boundary cell contains suspended mass')
                c=0.d0
            else
                if (faces(f)%interior_s < -tolerance) &
                    call fail('negative suspended mass at outward boundary')
                c=max(faces(f)%interior_s,0.d0)/faces(f)%interior(1)
            endif
        endif
    end function sediment_concentration

    subroutine braids_boundary_apply(side,val,aux,nrow,ncol,meqn,naux,hx,hy,xlo,ylo,ng,time)
        integer, intent(in) :: side,nrow,ncol,meqn,naux,ng
        real(kind=8), intent(in) :: hx,hy,xlo,ylo,time
        real(kind=8), intent(inout) :: val(meqn,nrow,ncol),aux(naux,nrow,ncol)
        integer :: i,j,ir,jr,lo,hi,k,index,f,component
        real(kind=8) :: coordinate
        if (.not.braids_boundary_enabled()) call fail('internal apply called without opt-in table')
        component=normal_component(side)
        if (side<=2) then
            lo=1; hi=ng; ir=ng+1
            if (side==2) then
                ir=nrow-ng; lo=ir+1; hi=nrow
            endif
            do j=1,ncol
                coordinate=ylo+(j-0.5d0)*hy
                index=nint((coordinate-grid_y)/grid_dy-0.5d0); f=0
                if (index>=0.and.index<grid_ny) f=lookup(side,index)
                do i=lo,hi
                    aux(:,i,j)=aux(:,ir,j); val(:,i,j)=val(:,ir,j)
                    val(component,i,j)=-val(component,ir,j)
                    if (f==0.or..not.prepared) cycle
                    val(1:3,i,j)=faces(f)%trace
                    if (meqn>=4) val(4,i,j)=faces(f)%trace(1)*sediment_concentration(f)
                enddo
            enddo
        else
            lo=1; hi=ng; jr=ng+1
            if (side==4) then
                jr=ncol-ng; lo=jr+1; hi=ncol
            endif
            do i=1,nrow
                coordinate=xlo+(i-0.5d0)*hx
                index=nint((coordinate-grid_x)/grid_dx-0.5d0); f=0
                if (index>=0.and.index<grid_nx) f=lookup(side,index)
                do j=lo,hi
                    aux(:,i,j)=aux(:,i,jr); val(:,i,j)=val(:,i,jr)
                    val(component,i,j)=-val(component,i,jr)
                    if (f==0.or..not.prepared) cycle
                    val(1:3,i,j)=faces(f)%trace
                    if (meqn>=4) val(4,i,j)=faces(f)%trace(1)*sediment_concentration(f)
                enddo
            enddo
        endif
    end subroutine

    subroutine braids_sediment_boundary_flux(nvar,mitot,mjtot,mbc,dx,dy,xlow,ylow,q,fm,fp,gm,gp)
        integer, intent(in) :: nvar,mitot,mjtot,mbc
        real(kind=8), intent(in) :: dx,dy,xlow,ylow,q(nvar,mitot,mjtot)
        real(kind=8), intent(inout) :: fm(nvar,mitot,mjtot),fp(nvar,mitot,mjtot)
        real(kind=8), intent(inout) :: gm(nvar,mitot,mjtot),gp(nvar,mitot,mjtot)
        integer :: side,index,f,i,j,ilo,ihi,jlo,jhi,component
        real(kind=8) :: scalar_flux,c
        if (.not.enabled.or..not.accepted_active.or.nvar<4) return
        if (.not.prepared) call fail('sediment boundary flux used before preparation')
        call polygon_wall_flux(nvar,mitot,mjtot,mbc,dx,dy,xlow,ylow,q,fm,fp,gm,gp,.true.)
        ilo=nint((xlow+mbc*dx-grid_x)/grid_dx); ihi=ilo+mitot-2*mbc-1
        jlo=nint((ylow+mbc*dy-grid_y)/grid_dy); jhi=jlo+mjtot-2*mbc-1
        do side=1,4
            if (side==1.and.ilo/=0) cycle
            if (side==2.and.ihi/=grid_nx-1) cycle
            if (side==3.and.jlo/=0) cycle
            if (side==4.and.jhi/=grid_ny-1) cycle
            do f=1,size(faces)
                if (faces(f)%side/=side) cycle
                index=faces(f)%index
                if (side<=2) then
                    if (index<jlo.or.index>jhi) cycle
                    j=mbc+1+index-jlo; i=mbc+1
                    if (side==2) i=mitot-mbc
                else
                    if (index<ilo.or.index>ihi) cycle
                    i=mbc+1+index-ilo; j=mbc+1
                    if (side==4) j=mjtot-mbc
                endif
                component=normal_component(side); c=sediment_concentration(f)
                scalar_flux=faces(f)%trace(component)*c
                select case(side)
                case(1)
                    fm(4,i,j)=scalar_flux; fp(4,i,j)=scalar_flux
                case(2)
                    fm(4,i+1,j)=scalar_flux; fp(4,i+1,j)=scalar_flux
                case(3)
                    gm(4,i,j)=scalar_flux; gp(4,i,j)=scalar_flux
                case(4)
                    gm(4,i,j+1)=scalar_flux; gp(4,i,j+1)=scalar_flux
                end select
            enddo
        enddo
    end subroutine braids_sediment_boundary_flux

    subroutine polygon_wall_flux(nv,mi,mj,ng,dx,dy,xlow,ylow,q,fm,fp,gm,gp,scalar_only)
        integer, intent(in) :: nv,mi,mj,ng
        real(kind=8), intent(in) :: dx,dy,xlow,ylow,q(nv,mi,mj)
        real(kind=8), intent(inout) :: fm(nv,mi,mj),fp(nv,mi,mj),gm(nv,mi,mj),gp(nv,mi,mj)
        logical, intent(in) :: scalar_only
        integer :: side,index,i,j,ilo,ihi,jlo,jhi,n,component,tangent
        real(kind=8) :: inside(3)
        if (table_version/=2) return
        ilo=nint((xlow+ng*dx-grid_x)/grid_dx);ihi=ilo+mi-2*ng-1
        jlo=nint((ylow+ng*dy-grid_y)/grid_dy);jhi=jlo+mj-2*ng-1
        do side=1,4
            if (side==1.and.ilo/=0.or.side==2.and.ihi/=grid_nx-1.or. &
                side==3.and.jlo/=0.or.side==4.and.jhi/=grid_ny-1) cycle
            n=grid_nx
            if (side<=2) n=grid_ny
            do index=0,n-1
                if (lookup(side,index)/=0) cycle
                if (side<=2) then
                    if (index<jlo.or.index>jhi) cycle
                    j=ng+1+index-jlo;i=ng+1
                    if (side==2) i=mi-ng
                else
                    if (index<ilo.or.index>ihi) cycle
                    i=ng+1+index-ilo;j=ng+1
                    if (side==4) j=mj-ng
                endif
                component=normal_component(side);tangent=5-component
                call physical_flux(q(1:3,i,j),side,inside)
                ! Retain the Riemann solver's normal wall pressure. Explicitly
                ! enforce zero mass and tangential advective flux; reflected
                ! ghost cells alone do not enforce that after MOL correction.
                select case(side)
                case(1)
                    if (scalar_only) then
                        fm(4,i,j)=0.d0;fp(4,i,j)=0.d0
                    else
                        fp(1,i,j)=-inside(1);fp(tangent,i,j)=-inside(tangent)
                    endif
                case(2)
                    if (scalar_only) then
                        fm(4,i+1,j)=0.d0;fp(4,i+1,j)=0.d0
                    else
                        fm(1,i+1,j)=-inside(1);fm(tangent,i+1,j)=-inside(tangent)
                    endif
                case(3)
                    if (scalar_only) then
                        gm(4,i,j)=0.d0;gp(4,i,j)=0.d0
                    else
                        gp(1,i,j)=-inside(1);gp(tangent,i,j)=-inside(tangent)
                    endif
                case(4)
                    if (scalar_only) then
                        gm(4,i,j+1)=0.d0;gp(4,i,j+1)=0.d0
                    else
                        gm(1,i,j+1)=-inside(1);gm(tangent,i,j+1)=-inside(tangent)
                    endif
                end select
            enddo
        enddo
    end subroutine polygon_wall_flux

    subroutine physical_flux(q,side,flux)
        use geoclaw_module, only: grav
        real(kind=8), intent(in) :: q(3)
        integer, intent(in) :: side
        real(kind=8), intent(out) :: flux(3)
        integer :: component
        real(kind=8) :: u
        flux=0.d0
        if (q(1)<=0.d0) return
        component=normal_component(side); u=q(component)/q(1)
        flux(1)=q(component); flux(2:3)=q(2:3)*u
        flux(component)=flux(component)+0.5d0*grav*q(1)**2
    end subroutine

    subroutine braids_boundary_begin_step(time,dt)
        real(kind=8), intent(in) :: time,dt
        real(kind=8) :: tol
        if (.not.enabled) return
        if (accepted_active.or..not.prepared) call fail('invalid accepted boundary update lifecycle')
        tol=1.d-12*max(1.d0,abs(time),abs(dt))
        if (abs(time-prepared_time)>tol.or.abs(dt-prepared_dt)>tol) &
            call fail('accepted boundary update has stale prepared trace')
        accepted_active=.true.
    end subroutine

    subroutine braids_boundary_flux(nvar,mitot,mjtot,mbc,dx,dy,xlow,ylow,time,dt,q,aux,maux,fm,fp,gm,gp)
        integer, intent(in) :: nvar,mitot,mjtot,mbc,maux
        real(kind=8), intent(in) :: dx,dy,xlow,ylow,time,dt,q(nvar,mitot,mjtot),aux(maux,mitot,mjtot)
        real(kind=8), intent(inout) :: fm(nvar,mitot,mjtot),fp(nvar,mitot,mjtot)
        real(kind=8), intent(inout) :: gm(nvar,mitot,mjtot),gp(nvar,mitot,mjtot)
        integer :: side,index,f,i,j,ilo,ihi,jlo,jhi
        real(kind=8) :: desired(3),interior(3),tol
        if (.not.enabled.or..not.accepted_active) return
        if (.not.prepared) call fail('numerical flux used before global trace preparation')
        call polygon_wall_flux(nvar,mitot,mjtot,mbc,dx,dy,xlow,ylow,q,fm,fp,gm,gp,.false.)
        tol=1.d-12*max(1.d0,abs(time),abs(dt))
        if (abs(time-prepared_time)>tol.or.abs(dt-prepared_dt)>tol) call fail('stale boundary time-integrated trace')
        ilo=nint((xlow+mbc*dx-grid_x)/grid_dx); ihi=ilo+mitot-2*mbc-1
        jlo=nint((ylow+mbc*dy-grid_y)/grid_dy); jhi=jlo+mjtot-2*mbc-1
        do side=1,4
            if (side==1.and.ilo/=0) cycle
            if (side==2.and.ihi/=grid_nx-1) cycle
            if (side==3.and.jlo/=0) cycle
            if (side==4.and.jhi/=grid_ny-1) cycle
            do f=1,size(faces)
                if (faces(f)%side/=side) cycle
                index=faces(f)%index
                if (side<=2) then
                    if (index<jlo.or.index>jhi) cycle
                    j=mbc+1+index-jlo; i=mbc+1
                    if (side==2) i=mitot-mbc
                else
                    if (index<ilo.or.index>ihi) cycle
                    i=mbc+1+index-ilo; j=mbc+1
                    if (side==4) j=mjtot-mbc
                endif
                call physical_flux(faces(f)%trace,side,desired)
                call physical_flux(q(1:3,i,j),side,interior)
                select case(side)
                case(1)
                    fp(1:3,i,j)=desired-interior
                case(2)
                    fm(1:3,i+1,j)=desired-interior
                case(3)
                    gp(1:3,i,j)=desired-interior
                case(4)
                    gm(1:3,i,j+1)=desired-interior
                end select
            enddo
        enddo
    end subroutine

    subroutine open_diagnostics(restarting,time)
        logical, intent(in) :: restarting
        real(kind=8), intent(in) :: time
        integer :: source,temporary,ios,b,id,row_id
        real(kind=8) :: row_time,row_dt,row_end,group_time,group_end,tol
        logical :: exists,first
        logical, allocatable :: seen(:)
        character(len=4096) :: line
        character(len=*), parameter :: path='braids_boundary_diagnostics.csv'
        if (diagnostic_unit/=0) return
        if (.not.restarting) then
            open(newunit=diagnostic_unit,file=path,status='replace',action='write',iostat=ios)
            if (ios/=0) call fail('cannot create accepted boundary diagnostics')
            write(diagnostic_unit,'(a)') diagnostic_header
            diagnostic_end=time
            return
        endif
        inquire(file=path,exist=exists)
        if (.not.exists) call fail('restart requires its accepted boundary diagnostics ledger')
        open(newunit=source,file=path,status='old',action='read',iostat=ios)
        if (ios/=0) call fail('cannot read boundary restart ledger')
        open(newunit=temporary,file=path//'.restart.tmp',status='replace',action='write',iostat=ios)
        if (ios/=0) call fail('cannot create boundary restart ledger prefix')
        read(source,'(a)',iostat=ios) line
        if (ios/=0.or.trim(line)/=diagnostic_header) call fail('boundary restart ledger header mismatch')
        write(temporary,'(a)') diagnostic_header
        allocate(seen(size(boundaries))); seen=.false.; first=.true.
        tol=1.d-10*max(1.d0,abs(time)); group_time=time; group_end=time
        do
            read(source,'(a)',iostat=ios) line
            if (ios<0) exit
            if (ios/=0) call fail('cannot read boundary restart ledger row')
            read(line,*,iostat=ios) row_time,row_dt,row_end,row_id
            if (ios/=0) call fail('malformed boundary restart ledger row')
            if (.not.all(ieee_is_finite([row_time,row_dt,row_end])).or.row_dt<=0.d0) &
                call fail('invalid boundary restart ledger interval')
            if (abs(row_time+row_dt-row_end)>tol) call fail('boundary restart interval end is inconsistent')
            if (row_end>time+tol) cycle
            if (first) then
                group_time=row_time; group_end=row_end; first=.false.
            else if (abs(row_time-group_time)>tol) then
                if (.not.all(seen).or.abs(row_time-group_end)>tol) call fail('boundary restart prefix gap')
                seen=.false.; group_time=row_time; group_end=row_end
            endif
            if (abs(row_end-group_end)>tol) call fail('inconsistent boundary intervals in accepted step')
            b=0
            do id=1,size(boundaries)
                if (row_id==boundaries(id)%id) b=id
            enddo
            if (b==0) call fail('unknown boundary ID in restart ledger')
            if (seen(b)) call fail('duplicate boundary ID in restart ledger step')
            seen(b)=.true.; write(temporary,'(a)') trim(line)
        enddo
        if (first.or..not.all(seen).or.abs(group_end-time)>tol) &
            call fail('boundary restart ledger does not cover checkpoint')
        close(source); close(temporary); deallocate(seen)
        call rename(path//'.restart.tmp',path,ios)
        if (ios/=0) call fail('cannot replace accepted boundary ledger prefix')
        open(newunit=diagnostic_unit,file=path,status='old',position='append',action='write',iostat=ios)
        if (ios/=0) call fail('cannot append boundary restart diagnostics')
        diagnostic_end=time
    end subroutine

    subroutine braids_boundary_commit(time,dt,restarting)
        use geoclaw_module, only: grav,dry_tolerance
        real(kind=8), intent(in) :: time,dt
        logical, intent(in) :: restarting
        integer :: b,f,component,nsub,ndry,nsuper,nclosed,nbackflow
        real(kind=8) :: qsum,eta_min,eta_max,eta,common,error,residual,relative,h,u,tol
        if (.not.enabled) return
        if (.not.prepared.or..not.accepted_active) call fail('accepted boundary diagnostics before physical update')
        tol=1.d-10*max(1.d0,abs(time),abs(dt))
        if (abs(time-prepared_time)>tol.or.abs(dt-prepared_dt)>tol) &
            call fail('accepted boundary diagnostics have stale preparation')
        call open_diagnostics(restarting,time)
        if (abs(time-diagnostic_end)>tol) call fail('accepted boundary diagnostics are duplicated or discontinuous')
        do b=1,size(boundaries)
            qsum=0.d0; eta_min=huge(1.d0); eta_max=-huge(1.d0); error=0.d0
            nsub=0; ndry=0; nsuper=0; nclosed=0; nbackflow=0
            do f=1,size(faces)
                if (faces(f)%boundary/=b) cycle
                h=faces(f)%trace(1); component=normal_component(faces(f)%side)
                qsum=qsum+inward_sign(faces(f)%side)*faces(f)%trace(component)*faces(f)%length
                if (boundaries(b)%mode<=4) then
                    if (inward_sign(faces(f)%side)*faces(f)%trace(component)<0.d0) nbackflow=nbackflow+1
                else
                    if (inward_sign(faces(f)%side)*faces(f)%trace(component)>0.d0) nbackflow=nbackflow+1
                endif
                if (faces(f)%branch==0) nclosed=nclosed+1
                if (h<=dry_tolerance) cycle
                eta=h+faces(f)%bed; eta_min=min(eta_min,eta); eta_max=max(eta_max,eta)
                if (faces(f)%branch==1) then
                    nsub=nsub+1
                    u=inward_sign(faces(f)%side)*faces(f)%trace(component)/h
                    error=max(error,abs(u-2.d0*sqrt(grav*h)-faces(f)%rminus))
                else if (faces(f)%branch==2) then
                    if (faces(f)%interior(1)<=dry_tolerance) then
                        ndry=ndry+1
                    else
                        nsuper=nsuper+1
                    endif
                endif
            enddo
            common=boundaries(b)%stage
            if (eta_min==huge(1.d0)) then
                eta_min=common; eta_max=common
            endif
            ! Zero-flow closure has individual wall-compatible elevations;
            ! min/max retains that fact if the interior is nonuniform.
            if (boundaries(b)%request==0.d0) common=0.5d0*(eta_min+eta_max)
            residual=abs(qsum-boundaries(b)%request); relative=0.d0
            if (boundaries(b)%request>0.d0) relative=residual/boundaries(b)%request
            ! Outlet Q residual columns are unused (zero); no Q target is imposed.
            if (boundaries(b)%mode>=5) then
                residual=0.d0;relative=0.d0
            endif
            write(diagnostic_unit,'(*(g0,:,","))') time,dt,time+dt,boundaries(b)%id,boundaries(b)%request,qsum, &
                boundaries(b)%target,common,eta_min,eta_max,nsub,ndry,nsuper,nclosed,error,residual,relative,nbackflow
        enddo
        flush(diagnostic_unit)
        diagnostic_end=time+dt
        accepted_active=.false.
        prepared=.false.
    end subroutine
end module braids_boundary_module
