! One-fraction settling coupled conservatively to an Exner bed inventory.
module braids_settling_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use geoclaw_module, only: dry_tolerance
    use braids_van_rijn_1993_module, only: braids_vr1993_grain_parameters, &
                                              braids_vr1993_coefficients
    implicit none
    private
    public :: braids_settling_exchange,braids_sediment_dry_tolerance
    public :: braids_sediment_probe
    public :: braids_bed_evolves
    public :: braids_bed_material
    logical, save :: probe_configured=.false.,probe_enabled=.false.,probe_open=.false.
    real(kind=8), save :: probe_x=0.d0,probe_y=0.d0
    integer, save :: probe_unit=0
    logical, save :: configured=.false.,enabled=.false.
    logical, save :: update_bed_elevation=.false.
    real(kind=8), save :: settling_velocity=0.d0,grain_density=0.d0,porosity=0.d0
    real(kind=8), save :: sediment_dry_depth=-1.d0,erosion_rate=0.d0
    integer, save :: closure_mode=0
    ! Version 4: Delft3D 2DH SedThr guard (no exchange in shallow wet cells).
    ! Versions 1--3 retain their original behavior for reproducible regressions.
    real(kind=8), save :: minimum_exchange_depth=-1.d0
    real(kind=8), save :: water_density=0.d0,gravity=0.d0,viscosity=0.d0
    real(kind=8), save :: d50=0.d0,d90=0.d0,z0_current=0.d0,aks_factor=0.d0
    real(kind=8), save :: max_volume_concentration=0.d0,adaptation_factor=0.d0
    real(kind=8), save :: dstar=0.d0,tau_critical=0.d0
    logical, save :: use_delft_ws_floor=.false.
contains
    subroutine braids_bed_material(rhos,bulk,reference)
        real(8), intent(out) :: rhos,bulk
        real(8), optional, intent(out) :: reference(6)
        call configure()
        if (.not.enabled.or..not.update_bed_elevation) call fail('bedload requires coupled bed inventory')
        rhos=grain_density
        bulk=(1.d0-porosity)*grain_density
        if (present(reference)) then
            reference=0.d0
            if (closure_mode==1) reference=[water_density,gravity,viscosity,d50,d90,z0_current]
        endif
    end subroutine

    logical function braids_bed_evolves()
        ! Called during serial ghost preparation, including a fresh restart.
        call configure()
        braids_bed_evolves=enabled.and.update_bed_elevation
    end function braids_bed_evolves

    subroutine fail(message)
        character(len=*), intent(in) :: message
        write(*,'(a)') 'BRAIDS settling/Exner: '//trim(message)
        error stop 1
    end subroutine fail

    subroutine configure()
        integer :: unit,ios,version,bed_coupling,ws_floor
        logical :: exists,ready
!$OMP ATOMIC READ
        ready=configured
        if (ready) return
!$OMP CRITICAL (BraidsSettlingSetup)
        if (.not.configured) then
            inquire(file='braids_settling.data',exist=exists)
            if (exists) then
                open(newunit=unit,file='braids_settling.data',status='old',action='read',iostat=ios)
                if (ios/=0) call fail('cannot open braids_settling.data')
                read(unit,*,iostat=ios) version
                if (ios/=0.or.(version<1.or.version>4)) call fail('unsupported configuration version')
                if (version==1) then
                    read(unit,*,iostat=ios) settling_velocity,grain_density,porosity, &
                                            sediment_dry_depth,bed_coupling
                    erosion_rate=0.d0
                    closure_mode=0
                else if (version==2) then
                    read(unit,*,iostat=ios) settling_velocity,grain_density,porosity, &
                                            sediment_dry_depth,bed_coupling,erosion_rate
                    closure_mode=0
                else
                    read(unit,*,iostat=ios) settling_velocity,grain_density,porosity, &
                                            sediment_dry_depth,bed_coupling
                    if (ios==0) read(unit,*,iostat=ios) water_density,gravity,viscosity, &
                        d50,d90,z0_current,aks_factor,max_volume_concentration, &
                        adaptation_factor,ws_floor
                    if (ios==0.and.version==4) then
                        read(unit,*,iostat=ios) minimum_exchange_depth
                        if (ios/=0) call fail('missing sediment exchange depth')
                        if (.not.ieee_is_finite(minimum_exchange_depth)) &
                            call fail('nonfinite sediment exchange depth')
                        if (minimum_exchange_depth<sediment_dry_depth) &
                            call fail('sediment exchange depth must be at least the sediment dry depth')
                    endif
                    erosion_rate=0.d0
                    closure_mode=1
                endif
                close(unit)
                if (ios/=0.or..not.all(ieee_is_finite([settling_velocity,grain_density,porosity, &
                                                        sediment_dry_depth,erosion_rate]))) &
                    call fail('invalid settling/erosion parameters')
                if (settling_velocity<0.d0.or.erosion_rate<0.d0.or.grain_density<=0.d0.or. &
                    porosity<0.d0.or.porosity>=1.d0.or.sediment_dry_depth<dry_tolerance) &
                    call fail('settling, erosion, density, porosity, or dry depth is outside its valid range')
                if (bed_coupling<0.or.bed_coupling>1) call fail('bed coupling must be 0 or 1')
                if (erosion_rate>0.d0.and.bed_coupling/=1) &
                    call fail('erosion requires bed coupling')
                if (closure_mode==1) then
                    if (.not.all(ieee_is_finite([water_density,gravity,viscosity,d50,d90, &
                        z0_current,aks_factor,max_volume_concentration,adaptation_factor]))) &
                        call fail('invalid Van Rijn 1993 parameters')
                    if (bed_coupling/=1.or.water_density<=0.d0.or.grain_density<=water_density.or. &
                        gravity<=0.d0.or. &
                        viscosity<=0.d0.or.d50<=0.d0.or.d90<=0.d0.or.z0_current<=0.d0.or. &
                        aks_factor<=0.d0.or.max_volume_concentration<=0.d0.or. &
                        max_volume_concentration>1.d0.or.d90<d50.or. &
                        adaptation_factor<=0.d0.or.(ws_floor/=0.and.ws_floor/=1)) &
                        call fail('Van Rijn 1993 parameters are outside their valid range')
                    use_delft_ws_floor=ws_floor==1
                    call braids_vr1993_grain_parameters(d50,grain_density,water_density, &
                                                       gravity,viscosity,dstar,tau_critical)
                endif
                update_bed_elevation=bed_coupling==1
                enabled=.true.
            endif
!$OMP ATOMIC WRITE
            configured=.true.
        endif
!$OMP END CRITICAL (BraidsSettlingSetup)
    end subroutine configure

    real(kind=8) function braids_sediment_dry_tolerance() result(value)
        call configure()
        value=dry_tolerance
        if (enabled) value=sediment_dry_depth
    end function braids_sediment_dry_tolerance

    ! Opt-in read-only trace of one physical cell. Only its owning patch writes.
    ! A fresh process writes a new segment; this diagnostic is not a restart ledger.
    subroutine braids_sediment_probe(phase,time,dt,nvar,mi,mj,mbc,xlow,ylow,dx,dy,q,maux,aux)
        integer, intent(in) :: phase,nvar,mi,mj,mbc,maux
        real(kind=8), intent(in) :: time,dt,xlow,ylow,dx,dy,q(nvar,mi,mj),aux(maux,mi,mj)
        integer :: unit,ios,i,j
        logical :: ready
        real(kind=8) :: ceq,rate,ustar,tau_ratio,caks,aks,tsd,h,c,u
        if (nvar<5) return
!$OMP ATOMIC READ
        ready=probe_configured
        if (.not.ready) then
!$OMP CRITICAL (BraidsSedimentProbeSetup)
            if (.not.probe_configured) then
                inquire(file='braids_sediment_probe.data',exist=probe_enabled)
                if (probe_enabled) then
                    open(newunit=unit,file='braids_sediment_probe.data',status='old',action='read')
                    read(unit,*,iostat=ios) probe_x,probe_y
                    close(unit)
                    if (ios/=0) call fail('invalid sediment probe coordinates')
                    if (.not.all(ieee_is_finite([probe_x,probe_y]))) call fail('nonfinite probe coordinates')
                endif
!$OMP ATOMIC WRITE
                probe_configured=.true.
            endif
!$OMP END CRITICAL (BraidsSedimentProbeSetup)
        endif
        if (.not.probe_enabled) return
        if (probe_x<xlow.or.probe_x>=xlow+(mi-2*mbc)*dx.or. &
            probe_y<ylow.or.probe_y>=ylow+(mj-2*mbc)*dy) return
        i=mbc+1+floor((probe_x-xlow)/dx); j=mbc+1+floor((probe_y-ylow)/dy)
        call configure()
        ceq=0.d0;rate=0.d0;ustar=0.d0;tau_ratio=0.d0;caks=0.d0;aks=0.d0;tsd=0.d0
        h=q(1,i,j);c=0.d0;u=0.d0
        if (h>0.d0) then
            c=q(4,i,j)/h;u=sqrt(q(2,i,j)**2+q(3,i,j)**2)/h
        endif
        if (closure_mode==1.and.h>sediment_dry_depth) &
            call braids_vr1993_coefficients(h,q(2,i,j),q(3,i,j),settling_velocity,grain_density, &
                water_density,viscosity,d50,d90,z0_current,aks_factor,max_volume_concentration, &
                adaptation_factor,use_delft_ws_floor,dstar,tau_critical,ceq,rate,ustar,tau_ratio,caks,aks,tsd)
        if (.not.probe_open) then
            open(newunit=probe_unit,file='braids_sediment_probe.csv',status='replace',action='write')
            write(probe_unit,'(a)') 'phase,t_start,dt,h_m,hu_m2_s,hv_m2_s,S_kg_m2,Mbed_kg_m2,'// &
                'bed_m,C_kg_m3,speed_m_s,Ceq_kg_m3,rate_s_inv,ustar_m_s,tau_ratio,caks,aks_m,tsd'
            probe_open=.true.
        endif
        write(probe_unit,'(i0,17(",",es25.17e3))') phase,time,dt,h,q(2:5,i,j),aux(1,i,j), &
            c,u,ceq,rate,ustar,tau_ratio,caks,aks,tsd
        flush(probe_unit)
    end subroutine braids_sediment_probe

    subroutine braids_settling_exchange(nvar,mitot,mjtot,mbc,dt,q,maux,aux,exchange)
        integer, intent(in) :: nvar,mitot,mjtot,mbc,maux
        real(kind=8), intent(in) :: dt
        real(kind=8), intent(inout) :: q(nvar,mitot,mjtot),aux(maux,mitot,mjtot)
        ! Optional integrated [deposition, erosion, dry-film transfer], summed
        ! over interior cells in kg/m2; caller multiplies by Cartesian cell area.
        real(kind=8), optional, intent(out) :: exchange(3)
        real(kind=8) :: source_integral,deposition,erosion,hit_time,ratio,log_ratio
        integer :: i,j,il,ir,jb,jt
        real(kind=8) :: suspended,bed_before,bed_after,available,candidate
        real(kind=8) :: rate,equilibrium,bulk_density,tolerance
        real(kind=8) :: ustar,tau_ratio,caks,aks,tsd
        if (present(exchange)) exchange=0.d0
        call configure()
        if (.not.enabled) return
        if (nvar<5.or.maux<1) call fail('enabled exchange requires h,hu,hv,S,Mbed and bed auxiliary')
        if (.not.ieee_is_finite(dt).or.dt<=0.d0) call fail('invalid accepted time step')
        il=mbc+1; ir=mitot-mbc; jb=mbc+1; jt=mjtot-mbc
        bulk_density=(1.d0-porosity)*grain_density
        do j=jb,jt
            do i=il,ir
                suspended=q(4,i,j); bed_before=q(5,i,j)
                tolerance=4096.d0*epsilon(1.d0)*max(1.d0,abs(suspended),abs(q(5,i,j)))
                if (suspended<-tolerance.or.bed_before<-tolerance) &
                    call fail('negative suspended or bed inventory')
                suspended=max(suspended,0.d0); bed_before=max(bed_before,0.d0)
                source_integral=0.d0; rate=0.d0; equilibrium=0.d0
                available=suspended+bed_before
                if (q(1,i,j)<=sediment_dry_depth) then
                    candidate=0.d0
                else if (closure_mode==1.and.q(1,i,j)<=minimum_exchange_depth) then
                    ! Delft3D erosed: initialize source/sink to zero, then skip
                    ! kfsed=0 cells. Keep their suspension available to transport;
                    ! the conservative dry-film transfer above still applies.
                    candidate=suspended
                else if (closure_mode==1) then
                    call braids_vr1993_coefficients(q(1,i,j),q(2,i,j),q(3,i,j), &
                        settling_velocity,grain_density,water_density,viscosity, &
                        d50,d90,z0_current,aks_factor,max_volume_concentration, &
                        adaptation_factor,use_delft_ws_floor,dstar,tau_critical, &
                        equilibrium,rate,ustar,tau_ratio,caks,aks,tsd)
                    equilibrium=q(1,i,j)*equilibrium
                    if (rate>0.d0) then
                        source_integral=equilibrium*rate*dt
                        candidate=equilibrium+(suspended-equilibrium)*exp(-rate*dt)
                    else
                        candidate=suspended
                    endif
                else if (settling_velocity==0.d0) then
                    source_integral=erosion_rate*dt
                    candidate=suspended+erosion_rate*dt
                else
                    source_integral=erosion_rate*dt
                    rate=settling_velocity/q(1,i,j)
                    equilibrium=erosion_rate/rate
                    candidate=equilibrium+(suspended-equilibrium)*exp(-rate*dt)
                endif
                q(4,i,j)=min(max(candidate,0.d0),available)
                bed_after=available-q(4,i,j)
                q(5,i,j)=bed_after
                if (present(exchange)) then
                    if (q(1,i,j)<=sediment_dry_depth) then
                        exchange(3)=exchange(3)+suspended
                    else
                        ! Integrate the bounded, frozen-coefficient relaxation.
                        ! Before exhaustion E=rate*equilibrium; once Mbed=0,
                        ! deposition can only be re-entrained at D=rate*available.
                        ! The bounded final state is unchanged by this observer.
                        erosion=source_integral
                        if (candidate>available) then
                            if (rate>0.d0.and.equilibrium>available) then
                                ratio=bed_before/(equilibrium-available)
                                if (ratio<1.d-4) then
                                    log_ratio=ratio*(1.d0-ratio*(0.5d0-ratio/3.d0))
                                else
                                    log_ratio=log(1.d0+ratio)
                                endif
                                hit_time=min(dt,max(0.d0,log_ratio/rate))
                                erosion=rate*(equilibrium*hit_time+available*(dt-hit_time))
                            else
                                erosion=max(0.d0,q(4,i,j)-suspended)
                            endif
                        endif
                        deposition=max(0.d0,erosion+suspended-q(4,i,j))
                        erosion=max(0.d0,deposition+q(4,i,j)-suspended)
                        exchange(1)=exchange(1)+deposition
                        exchange(2)=exchange(2)+erosion
                    endif
                endif
                if (update_bed_elevation) &
                    aux(1,i,j)=aux(1,i,j)+(bed_after-bed_before)/bulk_density
            enddo
        enddo
    end subroutine braids_settling_exchange
end module braids_settling_module
