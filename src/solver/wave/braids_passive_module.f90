! Opt-in conservative passive suspended mass S=h*C.
!
! GeoClaw's three-wave water Riemann solver leaves q(4) and its fluctuations
! zero. After the accepted hydraulic face update is finalized, this module
! reconstructs the single conservative water flux represented by the two
! one-sided updates, verifies their equality, and assigns both q(4) face
! updates to F_h*C_face. Mode 1 is first-order upwind. Mode 2 uses an upwind
! conservative midpoint followed by bounded MC reconstruction. Mode 3 limits
! the MUSCL antidiffusive flux against the multidimensional low-order cell
! bounds. Mode 4 uses a fifth-order WENO-Z spatial candidate with the same FCT
! bounds. The standard stepgrid divergence then advances S with exactly the
! same accepted water flux.
module braids_passive_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use geoclaw_module, only: use_fwave_positivity_limiter
    use amr_module, only: xperdom,yperdom,mthbc
    use braids_boundary_module, only: braids_boundary_enabled
    use braids_settling_module, only: braids_sediment_dry_tolerance
    implicit none
    private
    public :: braids_passive_fluxes, braids_passive_diffusion
    public :: braids_passive_wetdry_fluxes

    logical, save :: configured=.false.
    integer, save :: transport_mode=0
    logical, save :: diffusion_configured=.false.
    integer, save :: diffusion_mode=0
    integer, save :: diffusion_nx=0,diffusion_ny=0
    real(kind=8), save :: diffusion_constant=0.d0
    real(kind=8), save :: diffusion_xlower=0.d0,diffusion_ylower=0.d0
    real(kind=8), save :: diffusion_dx=0.d0,diffusion_dy=0.d0
    real(kind=8), allocatable, save :: diffusion_field(:,:)

contains

    subroutine fail(message)
        character(len=*), intent(in) :: message
        write(*,'(a)') 'BRAIDS passive transport: '//trim(message)
        error stop 1
    end subroutine fail

    subroutine configure()
        integer :: unit,ios,mode
        character(len=512) :: line
        logical :: exists,ready
!$omp atomic read
        ready=configured
        if (ready) return
!$omp critical (BraidsPassiveSetup)
        if (.not.configured) then
        inquire(file='braids_passive.data',exist=exists)
        if (exists) then
        open(newunit=unit,file='braids_passive.data',status='old',action='read',iostat=ios)
        if (ios/=0) call fail('cannot open braids_passive.data')
        do
            read(unit,'(a)',iostat=ios) line
            if (ios/=0) call fail('braids_passive.data requires mode 0 through 4')
            line=adjustl(line)
            if (len_trim(line)==0 .or. line(1:1)=='#') cycle
            read(line,*,iostat=ios) mode
            if (ios/=0 .or. mode<0 .or. mode>4) &
                call fail('mode: 0 off, 1 upwind, 2 MUSCL, 3 FCT, 4 bounded WENO-Z')
            exit
        enddo
        close(unit)
        transport_mode=mode
        endif
!$omp atomic write
        configured=.true.
        endif
!$omp end critical (BraidsPassiveSetup)
    end subroutine configure

    subroutine configure_diffusion()
        integer :: unit,field_unit,ios,mode,i,j
        character(len=512) :: field_path
        logical :: exists
        if (diffusion_configured) return
        diffusion_configured=.true.
        inquire(file='braids_diffusion.data',exist=exists)
        if (.not.exists) return
        open(newunit=unit,file='braids_diffusion.data',status='old',action='read',iostat=ios)
        if (ios/=0) call fail('cannot open braids_diffusion.data')
        read(unit,*,iostat=ios) mode
        if (ios/=0.or.mode<0.or.mode>2) &
            call fail('diffusion mode: 0 off, 1 constant, 2 raster field')
        diffusion_mode=mode
        if (mode==1) then
            read(unit,*,iostat=ios) diffusion_constant
            if (ios/=0.or..not.ieee_is_finite(diffusion_constant).or. &
                diffusion_constant<0.d0) call fail('invalid constant diffusivity')
        else if (mode==2) then
            read(unit,*,iostat=ios) diffusion_nx,diffusion_ny
            if (ios/=0.or.diffusion_nx<1.or.diffusion_ny<1) &
                call fail('invalid diffusion raster dimensions')
            read(unit,*,iostat=ios) diffusion_xlower,diffusion_ylower
            if (ios/=0) call fail('invalid diffusion raster lower corner')
            read(unit,*,iostat=ios) diffusion_dx,diffusion_dy
            if (ios/=0.or.diffusion_dx<=0.d0.or.diffusion_dy<=0.d0) &
                call fail('invalid diffusion raster cell size')
            read(unit,*,iostat=ios) field_path
            if (ios/=0.or.len_trim(field_path)==0) &
                call fail('diffusion raster file is missing')
            allocate(diffusion_field(diffusion_nx,diffusion_ny))
            open(newunit=field_unit,file=trim(field_path),status='old', &
                 action='read',iostat=ios)
            if (ios/=0) call fail('cannot open diffusion raster file')
            do j=1,diffusion_ny
                read(field_unit,*,iostat=ios) (diffusion_field(i,j),i=1,diffusion_nx)
                if (ios/=0) call fail('cannot read diffusion raster row')
            enddo
            close(field_unit)
            if (.not.all(ieee_is_finite(diffusion_field)).or. &
                any(diffusion_field<0.d0)) call fail('invalid diffusion raster value')
        endif
        close(unit)
    end subroutine configure_diffusion

    real(kind=8) function diffusivity_at(x,y) result(value)
        real(kind=8), intent(in) :: x,y
        integer :: i,j
        if (diffusion_mode==1) then
            value=diffusion_constant
            return
        endif
        i=int(floor((x-diffusion_xlower)/diffusion_dx))+1
        j=int(floor((y-diffusion_ylower)/diffusion_dy))+1
        if (i<1.or.i>diffusion_nx.or.j<1.or.j>diffusion_ny) &
            call fail('diffusion raster does not cover a computational cell')
        value=diffusion_field(i,j)
    end function diffusivity_at

    real(kind=8) function harmonic(left,right) result(value)
        real(kind=8), intent(in) :: left,right
        if (left+right>0.d0) then
            value=2.d0*left*right/(left+right)
        else
            value=0.d0
        endif
    end function harmonic

    subroutine braids_passive_diffusion(nvar,mitot,mjtot,mbc,dt,dx,dy,xlow,ylow,q)
        integer, intent(in) :: nvar,mitot,mjtot,mbc
        real(kind=8), intent(in) :: dt,dx,dy,xlow,ylow
        real(kind=8), intent(inout) :: q(nvar,mitot,mjtot)
        integer :: i,j,il,ir,jb,jt
        real(kind=8) :: seam,tolerance,local_cfl,maximum_cfl
        real(kind=8), allocatable :: c(:,:),trans(:,:),fx(:,:),fy(:,:),snew(:,:)
        call configure_diffusion()
        if (diffusion_mode==0) return
        if (nvar<4) call fail('diffusion requires h, hu, hv, S')
        if (.not.all(ieee_is_finite([dt,dx,dy])).or.dt<=0.d0.or.dx<=0.d0.or.dy<=0.d0) &
            call fail('invalid diffusion step or cell dimensions')
        il=mbc+1; ir=mitot-mbc
        jb=mbc+1; jt=mjtot-mbc
        allocate(c(mitot,mjtot),trans(mitot,mjtot),fx(mitot+1,mjtot), &
                 fy(mitot,mjtot+1),snew(mitot,mjtot))
        c=0.d0; trans=0.d0; fx=0.d0; fy=0.d0; snew=0.d0
        do j=jb,jt
            do i=il,ir
                c(i,j)=concentration(q(1,i,j),q(4,i,j))
                if (q(1,i,j)>braids_sediment_dry_tolerance()) &
                    trans(i,j)=q(1,i,j)*diffusivity_at( &
                        xlow+(i-il+0.5d0)*dx,ylow+(j-jb+0.5d0)*dy)
            enddo
        enddo

        do j=jb,jt
            do i=il+1,ir
                fx(i,j)=-harmonic(trans(i-1,j),trans(i,j))*(c(i,j)-c(i-1,j))/dx
            enddo
            if (xperdom) then
                seam=-harmonic(trans(ir,j),trans(il,j))*(c(il,j)-c(ir,j))/dx
                fx(il,j)=seam; fx(ir+1,j)=seam
            endif
        enddo
        do i=il,ir
            do j=jb+1,jt
                fy(i,j)=-harmonic(trans(i,j-1),trans(i,j))*(c(i,j)-c(i,j-1))/dy
            enddo
            if (yperdom) then
                seam=-harmonic(trans(i,jt),trans(i,jb))*(c(i,jb)-c(i,jt))/dy
                fy(i,jb)=seam; fy(i,jt+1)=seam
            endif
        enddo

        maximum_cfl=0.d0
        do j=jb,jt
            do i=il,ir
                if (q(1,i,j)>braids_sediment_dry_tolerance()) then
                    local_cfl=dt/q(1,i,j)*( &
                        harmonic(trans(i-1,j),trans(i,j))/dx**2+ &
                        harmonic(trans(i,j),trans(i+1,j))/dx**2+ &
                        harmonic(trans(i,j-1),trans(i,j))/dy**2+ &
                        harmonic(trans(i,j),trans(i,j+1))/dy**2)
                    ! At physical edges the arrays outside the active range are
                    ! zero; periodic seam transmissivities replace those terms.
                    if (i==il) local_cfl=local_cfl- &
                        harmonic(trans(i-1,j),trans(i,j))*dt/q(1,i,j)/dx**2
                    if (i==ir) local_cfl=local_cfl- &
                        harmonic(trans(i,j),trans(i+1,j))*dt/q(1,i,j)/dx**2
                    if (j==jb) local_cfl=local_cfl- &
                        harmonic(trans(i,j-1),trans(i,j))*dt/q(1,i,j)/dy**2
                    if (j==jt) local_cfl=local_cfl- &
                        harmonic(trans(i,j),trans(i,j+1))*dt/q(1,i,j)/dy**2
                    if (i==il.and.xperdom) local_cfl=local_cfl+ &
                        harmonic(trans(ir,j),trans(il,j))*dt/q(1,i,j)/dx**2
                    if (i==ir.and.xperdom) local_cfl=local_cfl+ &
                        harmonic(trans(ir,j),trans(il,j))*dt/q(1,i,j)/dx**2
                    if (j==jb.and.yperdom) local_cfl=local_cfl+ &
                        harmonic(trans(i,jt),trans(i,jb))*dt/q(1,i,j)/dy**2
                    if (j==jt.and.yperdom) local_cfl=local_cfl+ &
                        harmonic(trans(i,jt),trans(i,jb))*dt/q(1,i,j)/dy**2
                    maximum_cfl=max(maximum_cfl,local_cfl)
                endif
            enddo
        enddo
        if (maximum_cfl>1.d0+1024.d0*epsilon(1.d0)) &
            call fail('explicit diffusion stability limit exceeded')

        do j=jb,jt
            do i=il,ir
                snew(i,j)=q(4,i,j)-dt/dx*(fx(i+1,j)-fx(i,j)) &
                                    -dt/dy*(fy(i,j+1)-fy(i,j))
                tolerance=4096.d0*epsilon(1.d0)*max(1.d0,abs(q(4,i,j)))
                if (snew(i,j)<-tolerance.or..not.ieee_is_finite(snew(i,j))) &
                    call fail('diffusion produced invalid suspended mass')
            enddo
        enddo
        do j=jb,jt
            do i=il,ir
                q(4,i,j)=max(snew(i,j),0.d0)
            enddo
        enddo
        deallocate(c,trans,fx,fy,snew)
    end subroutine braids_passive_diffusion

    subroutine braids_passive_wetdry_fluxes(nvar,mitot,mjtot,mbc,dt,dx,dy, &
                                            xlow,ylow,q,fm,fp,gm,gp)
        integer, intent(in) :: nvar,mitot,mjtot,mbc
        real(kind=8), intent(in) :: dt,dx,dy,xlow,ylow,q(nvar,mitot,mjtot)
        real(kind=8), intent(inout) :: fm(nvar,mitot,mjtot),fp(nvar,mitot,mjtot)
        real(kind=8), intent(inout) :: gm(nvar,mitot,mjtot),gp(nvar,mitot,mjtot)
        integer :: i,j,il,ir,jb,jt
        real(kind=8) :: water_flux,tolerance,cmax,ctest
        real(kind=8), allocatable :: fx(:,:),fy(:,:),fsx(:,:),fsy(:,:),hnew(:,:),snew(:,:)
        logical, allocatable :: dry(:,:)
        call configure()
        if (transport_mode==0) return
        if (nvar<4) call fail('wet/dry transport requires h, hu, hv, S')
        il=mbc+1; ir=mitot-mbc; jb=mbc+1; jt=mjtot-mbc
        allocate(fx(mitot+1,mjtot),fy(mitot,mjtot+1),fsx(mitot+1,mjtot), &
                 fsy(mitot,mjtot+1),hnew(mitot,mjtot),snew(mitot,mjtot), &
                 dry(mitot,mjtot))
        fx=0.d0; fy=0.d0; fsx=0.d0; fsy=0.d0; hnew=0.d0; snew=0.d0; dry=.false.
        do j=jb,jt
            do i=il,ir+1
                if (i==il.and..not.xperdom) then
                    call accepted_side_flux(fp(1,i,j),q(2,i,j),water_flux)
                else if (i==ir+1.and..not.xperdom) then
                    call accepted_side_flux(fm(1,i,j),q(2,i-1,j),water_flux)
                else
                    call conservative_flux(fm(1,i,j),fp(1,i,j),q(2,i-1,j),q(2,i,j),water_flux)
                endif
                fx(i,j)=water_flux
                tolerance=2048.d0*epsilon(1.d0)*max(1.d0,abs(fm(4,i,j)),abs(fp(4,i,j)))
                if (abs(fm(4,i,j)-fp(4,i,j))>tolerance) &
                    call fail('wet/dry x sediment face flux is not conservative')
                fsx(i,j)=0.5d0*(fm(4,i,j)+fp(4,i,j))
            enddo
        enddo
        do i=il,ir
            do j=jb,jt+1
                if (j==jb.and..not.yperdom) then
                    call accepted_side_flux(gp(1,i,j),q(3,i,j),water_flux)
                else if (j==jt+1.and..not.yperdom) then
                    call accepted_side_flux(gm(1,i,j),q(3,i,j-1),water_flux)
                else
                    call conservative_flux(gm(1,i,j),gp(1,i,j),q(3,i,j-1),q(3,i,j),water_flux)
                endif
                fy(i,j)=water_flux
                tolerance=2048.d0*epsilon(1.d0)*max(1.d0,abs(gm(4,i,j)),abs(gp(4,i,j)))
                if (abs(gm(4,i,j)-gp(4,i,j))>tolerance) &
                    call fail('wet/dry y sediment face flux is not conservative')
                fsy(i,j)=0.5d0*(gm(4,i,j)+gp(4,i,j))
            enddo
        enddo
        cmax=0.d0
        do j=jb,jt
            do i=il,ir
                cmax=max(cmax,concentration(q(1,i,j),q(4,i,j)))
            enddo
        enddo
        ! A prescribed sediment inlet may introduce a concentration that is
        ! absent from an initially clear domain.  Include finalized accepted
        ! face concentrations in the step bound without reading dry ghosts.
        do j=jb,jt
            do i=il,ir+1
                if (abs(fx(i,j))>tiny(1.d0)) &
                    cmax=max(cmax,abs(fsx(i,j)/fx(i,j)))
            enddo
        enddo
        do i=il,ir
            do j=jb,jt+1
                if (abs(fy(i,j))>tiny(1.d0)) &
                    cmax=max(cmax,abs(fsy(i,j)/fy(i,j)))
            enddo
        enddo
        do j=jb,jt
            do i=il,ir
                hnew(i,j)=q(1,i,j)-dt/dx*(fx(i+1,j)-fx(i,j)) &
                                    -dt/dy*(fy(i,j+1)-fy(i,j))
                snew(i,j)=q(4,i,j)-dt/dx*(fsx(i+1,j)-fsx(i,j)) &
                                    -dt/dy*(fsy(i,j+1)-fsy(i,j))
                tolerance=4096.d0*epsilon(1.d0)*max(1.d0,abs(q(4,i,j)))
                if (snew(i,j)<-tolerance) call fail('wet/dry predicted suspended mass is negative')
                snew(i,j)=max(snew(i,j),0.d0)
                dry(i,j)=hnew(i,j)<=braids_sediment_dry_tolerance()
            enddo
        enddo
        do j=jb,jt
            do i=il,ir
                if (.not.dry(i,j).and.hnew(i,j)>braids_sediment_dry_tolerance()) then
                    ctest=snew(i,j)/hnew(i,j)
                    if (.not.ieee_is_finite(ctest).or.ctest<0.d0) &
                        call fail('wet/dry prediction produced an invalid concentration')
                    if (snew(i,j)>cmax*hnew(i,j)+braids_sediment_dry_tolerance()*max(1.d0,cmax) &
                        +1.d-10*max(1.d0,cmax*hnew(i,j))) &
                        call fail('wet/dry prediction exceeds the incoming concentration bound')
                endif
            enddo
        enddo
        ! A sub-threshold positive water film retains its local scalar
        ! inventory.  GeoClaw already keeps that water depth while setting
        ! only momentum to zero.  Moving the scalar elsewhere would either
        ! violate the maximum principle or introduce nonlocal diffusion.
        deallocate(fx,fy,fsx,fsy,hnew,snew,dry)
    end subroutine braids_passive_wetdry_fluxes

    real(kind=8) function concentration(h,s) result(c)
        real(kind=8), intent(in) :: h,s
        real(kind=8) :: tolerance
        tolerance=256.d0*epsilon(1.d0)*max(1.d0,abs(s))
        if (h<=braids_sediment_dry_tolerance()) then
            ! A sub-threshold film has no active donor concentration.  Its
            ! local scalar inventory remains in q(4) until the cell rewets.
            c=0.d0
        else
            if (s < -tolerance) call fail('negative suspended mass in face state')
            c=max(s,0.d0)/h
        endif
    end function concentration

    subroutine conservative_flux(left_update,right_update,left_physical,right_physical,flux)
        real(kind=8), intent(in) :: left_update,right_update,left_physical,right_physical
        real(kind=8), intent(out) :: flux
        real(kind=8) :: other,tolerance
        if (use_fwave_positivity_limiter) then
            flux=left_update
            other=right_update
        else
            flux=left_update+left_physical
            other=right_update+right_physical
        endif
        tolerance=2048.d0*epsilon(1.d0)*max(1.d0,abs(flux),abs(other))
        if (abs(flux-other)>tolerance) &
            call fail('one-sided accepted water fluxes are not conservative')
        if (.not.ieee_is_finite(flux)) call fail('nonfinite accepted water flux')
    end subroutine conservative_flux

    subroutine accepted_side_flux(update,physical,flux)
        real(kind=8), intent(in) :: update,physical
        real(kind=8), intent(out) :: flux
        if (use_fwave_positivity_limiter) then
            flux=update
        else
            flux=update+physical
        endif
        if (.not.ieee_is_finite(flux)) call fail('nonfinite physical-boundary water flux')
    end subroutine accepted_side_flux

    real(kind=8) function minmod3(a,b,c) result(value)
        real(kind=8), intent(in) :: a,b,c
        if ((a>0.d0.and.b>0.d0.and.c>0.d0).or. &
            (a<0.d0.and.b<0.d0.and.c<0.d0)) then
            value=sign(min(abs(a),abs(b),abs(c)),a)
        else
            value=0.d0
        endif
    end function minmod3

    real(kind=8) function limited_slope_x(c,i,j) result(slope)
        real(kind=8), intent(in) :: c(:,:)
        integer, intent(in) :: i,j
        slope=minmod3(2.d0*(c(i,j)-c(i-1,j)), &
                      0.5d0*(c(i+1,j)-c(i-1,j)), &
                      2.d0*(c(i+1,j)-c(i,j)))
    end function limited_slope_x

    real(kind=8) function limited_slope_y(c,i,j) result(slope)
        real(kind=8), intent(in) :: c(:,:)
        integer, intent(in) :: i,j
        slope=minmod3(2.d0*(c(i,j)-c(i,j-1)), &
                      0.5d0*(c(i,j+1)-c(i,j-1)), &
                      2.d0*(c(i,j+1)-c(i,j)))
    end function limited_slope_y

    real(kind=8) function weno_z_left(vmm,vm,v0,vp,vpp) result(value)
        real(kind=8), intent(in) :: vmm,vm,v0,vp,vpp
        real(kind=8) :: p0,p1,p2,b0,b1,b2,tau5,eps
        real(kind=8) :: a0,a1,a2,total,scale
        p0=(2.d0*vmm-7.d0*vm+11.d0*v0)/6.d0
        p1=(-vm+5.d0*v0+2.d0*vp)/6.d0
        p2=(2.d0*v0+5.d0*vp-vpp)/6.d0
        b0=13.d0/12.d0*(vmm-2.d0*vm+v0)**2 &
           +0.25d0*(vmm-4.d0*vm+3.d0*v0)**2
        b1=13.d0/12.d0*(vm-2.d0*v0+vp)**2+0.25d0*(vm-vp)**2
        b2=13.d0/12.d0*(v0-2.d0*vp+vpp)**2 &
           +0.25d0*(3.d0*v0-4.d0*vp+vpp)**2
        tau5=abs(b0-b2)
        scale=max(vmm*vmm,vm*vm,v0*v0,vp*vp,vpp*vpp)
        eps=1.d-40+1.d-14*scale
        a0=0.1d0*(1.d0+(tau5/(b0+eps))**2)
        a1=0.6d0*(1.d0+(tau5/(b1+eps))**2)
        a2=0.3d0*(1.d0+(tau5/(b2+eps))**2)
        total=a0+a1+a2
        value=(a0*p0+a1*p1+a2*p2)/total
    end function weno_z_left

    real(kind=8) function cvalue(c,i,j,il,ir,jb,jt) result(value)
        real(kind=8), intent(in) :: c(:,:)
        integer, intent(in) :: i,j,il,ir,jb,jt
        integer :: ii,jj
        ii=i; jj=j
        if (ii<il.or.ii>ir) then
            if (xperdom) then
                ii=il+modulo(ii-il,ir-il+1)
            else
                ii=min(max(ii,il),ir)
            endif
        endif
        if (jj<jb.or.jj>jt) then
            if (yperdom) then
                jj=jb+modulo(jj-jb,jt-jb+1)
            else
                jj=min(max(jj,jb),jt)
            endif
        endif
        value=c(ii,jj)
    end function cvalue

    subroutine braids_passive_fluxes(nvar,mitot,mjtot,mbc,dt,dx,dy,q,fm,fp,gm,gp,mol)
        logical, optional, intent(in) :: mol
        real(8) :: predictor_dt
        integer, intent(in) :: nvar,mitot,mjtot,mbc
        real(kind=8), intent(in) :: dt,dx,dy
        real(kind=8), intent(in) :: q(nvar,mitot,mjtot)
        real(kind=8), intent(inout) :: fm(nvar,mitot,mjtot),fp(nvar,mitot,mjtot)
        real(kind=8), intent(inout) :: gm(nvar,mitot,mjtot),gp(nvar,mitot,mjtot)
        integer :: i,j,k,il,ir,jb,jt
        real(kind=8) :: water_flux,c,cl,cr,hhalf,shalf,tolerance
        real(kind=8) :: anti,alpha,dwest,deast,dsouth,dnorth,allowance
        real(kind=8), allocatable :: fx(:,:),fy(:,:),fsx(:,:),fsy(:,:)
        real(kind=8), allocatable :: highx(:,:),highy(:,:),chalf(:,:),cbefore(:,:)
        real(kind=8), allocatable :: hlow(:,:),slow(:,:),cmin(:,:),cmax(:,:)
        real(kind=8), allocatable :: pplus(:,:),pminus(:,:),rplus(:,:),rminus(:,:)
        call configure()
        if (transport_mode==0) return
        if (nvar<4) call fail('v1 requires h, hu, hv, S')
        if (mbc<1) call fail('at least one ghost-cell layer is required')
        if (transport_mode>=2.and.mbc<2) call fail('MUSCL/FCT requires two ghost-cell layers')
        if (.not.all(ieee_is_finite([dt,dx,dy])).or.dt<=0.d0.or.dx<=0.d0.or.dy<=0.d0) &
            call fail('invalid accepted step or cell dimensions')
        predictor_dt=dt
        if(present(mol)) then
            if(mol) predictor_dt=0.d0
        endif
        il=mbc+1; ir=mitot-mbc
        jb=mbc+1; jt=mjtot-mbc
        allocate(fx(mitot,mjtot),fy(mitot,mjtot),fsx(mitot,mjtot), &
                 fsy(mitot,mjtot),highx(mitot,mjtot), &
                 highy(mitot,mjtot),chalf(mitot,mjtot))
        fx=0.d0; fy=0.d0; fsx=0.d0; fsy=0.d0
        highx=0.d0; highy=0.d0; chalf=0.d0

        do j=jb,jt
            do i=il,ir+1
                if (i==il.and..not.xperdom) then
                    call accepted_side_flux(fp(1,i,j),q(2,i,j),water_flux)
                else if (i==ir+1.and..not.xperdom) then
                    call accepted_side_flux(fm(1,i,j),q(2,i-1,j),water_flux)
                else
                    call conservative_flux(fm(1,i,j),fp(1,i,j),q(2,i-1,j),q(2,i,j),water_flux)
                endif
                fx(i,j)=water_flux
                if (water_flux>=0.d0) then
                    c=concentration(q(1,i-1,j),q(4,i-1,j))
                else
                    c=concentration(q(1,i,j),q(4,i,j))
                endif
                fsx(i,j)=water_flux*c
            enddo
        enddo

        do i=il,ir
            do j=jb,jt+1
                if (j==jb.and..not.yperdom) then
                    call accepted_side_flux(gp(1,i,j),q(3,i,j),water_flux)
                else if (j==jt+1.and..not.yperdom) then
                    call accepted_side_flux(gm(1,i,j),q(3,i,j-1),water_flux)
                else
                    call conservative_flux(gm(1,i,j),gp(1,i,j),q(3,i,j-1),q(3,i,j),water_flux)
                endif
                fy(i,j)=water_flux
                if (water_flux>=0.d0) then
                    c=concentration(q(1,i,j-1),q(4,i,j-1))
                else
                    c=concentration(q(1,i,j),q(4,i,j))
                endif
                fsy(i,j)=water_flux*c
            enddo
        enddo

        if (transport_mode==1) then
            do j=jb,jt
                do i=il,ir+1
                    fm(4,i,j)=fsx(i,j); fp(4,i,j)=fsx(i,j)
                enddo
            enddo
            do i=il,ir
                do j=jb,jt+1
                    gm(4,i,j)=fsy(i,j); gp(4,i,j)=fsy(i,j)
                enddo
            enddo
            deallocate(fx,fy,fsx,fsy,highx,highy,chalf)
            return
        endif

        ! Conservative upwind midpoint. Constant concentration gives
        ! shalf=C*hhalf exactly because it uses the identical water flux.
        do j=jb,jt
            do i=il,ir
                hhalf=q(1,i,j)-0.5d0*predictor_dt/dx*(fx(i+1,j)-fx(i,j)) &
                                -0.5d0*predictor_dt/dy*(fy(i,j+1)-fy(i,j))
                shalf=q(4,i,j)-0.5d0*predictor_dt/dx*(fsx(i+1,j)-fsx(i,j)) &
                                -0.5d0*predictor_dt/dy*(fsy(i,j+1)-fsy(i,j))
                tolerance=2048.d0*epsilon(1.d0)*max(1.d0,abs(q(1,i,j)),abs(q(4,i,j)))
                if (shalf < -tolerance) &
                    call fail('MUSCL midpoint is negative; reduce the transport CFL')
                chalf(i,j)=concentration(max(hhalf,0.d0),max(shalf,0.d0))
            enddo
        enddo

        ! One ghost layer on each side is enough for the normal MC slopes.
        ! Periodic ghosts wrap; wall/extrapolation ghosts copy the edge cell.
        if (.not.xperdom.and.(mthbc(1)==0.or.mthbc(2)==0).and. &
            .not.braids_boundary_enabled()) &
            call fail('MUSCL/FCT v1 does not yet support user scalar x boundaries')
        if (.not.yperdom.and.(mthbc(3)==0.or.mthbc(4)==0).and. &
            .not.braids_boundary_enabled()) &
            call fail('MUSCL/FCT v1 does not yet support user scalar y boundaries')
        do k=1,mbc
            do j=jb,jt
                if (xperdom) then
                    chalf(il-k,j)=chalf(ir-k+1,j)
                    chalf(ir+k,j)=chalf(il+k-1,j)
                else
                    if (mthbc(1)==0.and.braids_boundary_enabled()) then
                        chalf(il-k,j)=concentration(q(1,il-k,j),q(4,il-k,j))
                    else
                        chalf(il-k,j)=chalf(il,j)
                    endif
                    if (mthbc(2)==0.and.braids_boundary_enabled()) then
                        chalf(ir+k,j)=concentration(q(1,ir+k,j),q(4,ir+k,j))
                    else
                        chalf(ir+k,j)=chalf(ir,j)
                    endif
                endif
            enddo
            do i=il-mbc,ir+mbc
                if (yperdom) then
                    chalf(i,jb-k)=chalf(i,jt-k+1)
                    chalf(i,jt+k)=chalf(i,jb+k-1)
                else
                    if (mthbc(3)==0.and.braids_boundary_enabled()) then
                        chalf(i,jb-k)=concentration(q(1,i,jb-k),q(4,i,jb-k))
                    else
                        chalf(i,jb-k)=chalf(i,jb)
                    endif
                    if (mthbc(4)==0.and.braids_boundary_enabled()) then
                        chalf(i,jt+k)=concentration(q(1,i,jt+k),q(4,i,jt+k))
                    else
                        chalf(i,jt+k)=chalf(i,jt)
                    endif
                endif
            enddo
        enddo

        if (transport_mode==4) then
!$omp parallel do default(shared) private(i,j,water_flux,c,cl,cr,hhalf,shalf,tolerance, &
!$omp& anti,alpha,dwest,deast,dsouth,dnorth,allowance) schedule(static)
            do j=jb,jt
                do i=il,ir+1
                    if (fx(i,j)>=0.d0) then
                        c=weno_z_left(cvalue(chalf,i-3,j,il,ir,jb,jt), &
                                    cvalue(chalf,i-2,j,il,ir,jb,jt), &
                                    cvalue(chalf,i-1,j,il,ir,jb,jt), &
                                    cvalue(chalf,i,j,il,ir,jb,jt), &
                                    cvalue(chalf,i+1,j,il,ir,jb,jt))
                    else
                        c=weno_z_left(cvalue(chalf,i+2,j,il,ir,jb,jt), &
                                    cvalue(chalf,i+1,j,il,ir,jb,jt), &
                                    cvalue(chalf,i,j,il,ir,jb,jt), &
                                    cvalue(chalf,i-1,j,il,ir,jb,jt), &
                                    cvalue(chalf,i-2,j,il,ir,jb,jt))
                    endif
                    highx(i,j)=fx(i,j)*c
                enddo
            enddo
!$omp end parallel do
!$omp parallel do default(shared) private(i,j,water_flux,c,cl,cr,hhalf,shalf,tolerance, &
!$omp& anti,alpha,dwest,deast,dsouth,dnorth,allowance) schedule(static)
            do i=il,ir
                do j=jb,jt+1
                    if (fy(i,j)>=0.d0) then
                        c=weno_z_left(cvalue(chalf,i,j-3,il,ir,jb,jt), &
                                    cvalue(chalf,i,j-2,il,ir,jb,jt), &
                                    cvalue(chalf,i,j-1,il,ir,jb,jt), &
                                    cvalue(chalf,i,j,il,ir,jb,jt), &
                                    cvalue(chalf,i,j+1,il,ir,jb,jt))
                    else
                        c=weno_z_left(cvalue(chalf,i,j+2,il,ir,jb,jt), &
                                    cvalue(chalf,i,j+1,il,ir,jb,jt), &
                                    cvalue(chalf,i,j,il,ir,jb,jt), &
                                    cvalue(chalf,i,j-1,il,ir,jb,jt), &
                                    cvalue(chalf,i,j-2,il,ir,jb,jt))
                    endif
                    highy(i,j)=fy(i,j)*c
                enddo
            enddo
!$omp end parallel do
        else
            do j=jb,jt
                do i=il,ir+1
                    cl=chalf(i-1,j); cr=chalf(i,j)
                    if (fx(i,j)>=0.d0) then
                        c=cl+0.5d0*limited_slope_x(chalf,i-1,j)
                    else
                        c=cr-0.5d0*limited_slope_x(chalf,i,j)
                    endif
                    c=min(max(c,min(cl,cr)),max(cl,cr))
                    highx(i,j)=fx(i,j)*c
                enddo
            enddo
            do i=il,ir
                do j=jb,jt+1
                    cl=chalf(i,j-1); cr=chalf(i,j)
                    if (fy(i,j)>=0.d0) then
                        c=cl+0.5d0*limited_slope_y(chalf,i,j-1)
                    else
                        c=cr-0.5d0*limited_slope_y(chalf,i,j)
                    endif
                    c=min(max(c,min(cl,cr)),max(cl,cr))
                    highy(i,j)=fy(i,j)*c
                enddo
            enddo
        endif

        if (transport_mode==2) then
            do j=jb,jt
                do i=il,ir+1
                    fm(4,i,j)=highx(i,j); fp(4,i,j)=highx(i,j)
                enddo
            enddo
            do i=il,ir
                do j=jb,jt+1
                    gm(4,i,j)=highy(i,j); gp(4,i,j)=highy(i,j)
                enddo
            enddo
            deallocate(fx,fy,fsx,fsy,highx,highy,chalf)
            return
        endif

        ! FCT starts from the bounded upwind result and admits as much of the
        ! MUSCL correction as each cell's positive/negative budget permits.
        allocate(cbefore(mitot,mjtot),hlow(mitot,mjtot),slow(mitot,mjtot), &
                 cmin(mitot,mjtot),cmax(mitot,mjtot),pplus(mitot,mjtot), &
                 pminus(mitot,mjtot),rplus(mitot,mjtot),rminus(mitot,mjtot))
        cbefore=0.d0; hlow=0.d0; slow=0.d0; cmin=0.d0; cmax=0.d0
        pplus=0.d0; pminus=0.d0; rplus=1.d0; rminus=1.d0
        do j=jb,jt
            do i=il,ir
                cbefore(i,j)=concentration(q(1,i,j),q(4,i,j))
            enddo
        enddo
        do k=1,mbc
            do j=jb,jt
                if (xperdom) then
                    cbefore(il-k,j)=cbefore(ir-k+1,j)
                    cbefore(ir+k,j)=cbefore(il+k-1,j)
                else
                    if (mthbc(1)==0.and.braids_boundary_enabled()) then
                        cbefore(il-k,j)=concentration(q(1,il-k,j),q(4,il-k,j))
                    else
                        cbefore(il-k,j)=cbefore(il,j)
                    endif
                    if (mthbc(2)==0.and.braids_boundary_enabled()) then
                        cbefore(ir+k,j)=concentration(q(1,ir+k,j),q(4,ir+k,j))
                    else
                        cbefore(ir+k,j)=cbefore(ir,j)
                    endif
                endif
            enddo
            do i=il-mbc,ir+mbc
                if (yperdom) then
                    cbefore(i,jb-k)=cbefore(i,jt-k+1)
                    cbefore(i,jt+k)=cbefore(i,jb+k-1)
                else
                    if (mthbc(3)==0.and.braids_boundary_enabled()) then
                        cbefore(i,jb-k)=concentration(q(1,i,jb-k),q(4,i,jb-k))
                    else
                        cbefore(i,jb-k)=cbefore(i,jb)
                    endif
                    if (mthbc(4)==0.and.braids_boundary_enabled()) then
                        cbefore(i,jt+k)=concentration(q(1,i,jt+k),q(4,i,jt+k))
                    else
                        cbefore(i,jt+k)=cbefore(i,jt)
                    endif
                endif
            enddo
        enddo

        do j=jb,jt
            do i=il,ir
                hlow(i,j)=q(1,i,j)-dt/dx*(fx(i+1,j)-fx(i,j)) &
                                  -dt/dy*(fy(i,j+1)-fy(i,j))
                slow(i,j)=q(4,i,j)-dt/dx*(fsx(i+1,j)-fsx(i,j)) &
                                  -dt/dy*(fsy(i,j+1)-fsy(i,j))
                cmin(i,j)=min(cbefore(i,j),cbefore(i-1,j),cbefore(i+1,j), &
                              cbefore(i,j-1),cbefore(i,j+1))
                cmax(i,j)=max(cbefore(i,j),cbefore(i-1,j),cbefore(i+1,j), &
                              cbefore(i,j-1),cbefore(i,j+1))
                if (abs(fx(i,j))>tiny(1.d0)) then
                    c=abs(fsx(i,j)/fx(i,j)); cmin(i,j)=min(cmin(i,j),c); cmax(i,j)=max(cmax(i,j),c)
                endif
                if (abs(fx(i+1,j))>tiny(1.d0)) then
                    c=abs(fsx(i+1,j)/fx(i+1,j)); cmin(i,j)=min(cmin(i,j),c); cmax(i,j)=max(cmax(i,j),c)
                endif
                if (abs(fy(i,j))>tiny(1.d0)) then
                    c=abs(fsy(i,j)/fy(i,j)); cmin(i,j)=min(cmin(i,j),c); cmax(i,j)=max(cmax(i,j),c)
                endif
                if (abs(fy(i,j+1))>tiny(1.d0)) then
                    c=abs(fsy(i,j+1)/fy(i,j+1)); cmin(i,j)=min(cmin(i,j),c); cmax(i,j)=max(cmax(i,j),c)
                endif
                tolerance=4096.d0*epsilon(1.d0)*max(1.d0,abs(q(4,i,j)))
                if (slow(i,j)<-tolerance) then
                    write(*,*) 'FCT cell i,j,dt,h,S,hlow,Slow,Cmin,Cmax:', &
                        i,j,dt,q(1,i,j),q(4,i,j),hlow(i,j),slow(i,j),cmin(i,j),cmax(i,j)
                    write(*,*) 'FCT outward water CFL:',dt/max(q(1,i,j),tiny(1.d0))*( &
                        (max(-fx(i,j),0.d0)+max(fx(i+1,j),0.d0))/dx+ &
                        (max(-fy(i,j),0.d0)+max(fy(i,j+1),0.d0))/dy)
                    call fail('FCT low-order state is negative; reduce the transport CFL')
                endif
                if (hlow(i,j)<=braids_sediment_dry_tolerance()) then
                    ! The cell's scalar is an inactive local inventory.  Use
                    ! the monotone flux at the moving shoreline and admit no
                    ! antidiffusive correction through this cell.
                    rplus(i,j)=0.d0
                    rminus(i,j)=0.d0
                    cycle
                endif
                if (slow(i,j)<cmin(i,j)*hlow(i,j)-tolerance &
                        -braids_sediment_dry_tolerance()*max(1.d0,cmax(i,j)).or. &
                    slow(i,j)>cmax(i,j)*hlow(i,j)+tolerance &
                        +braids_sediment_dry_tolerance()*max(1.d0,cmax(i,j))) then
                    write(*,*) 'FCT cell i,j,dt,h,S,hlow,Slow,Cmin,Cmax:', &
                        i,j,dt,q(1,i,j),q(4,i,j),hlow(i,j),slow(i,j),cmin(i,j),cmax(i,j)
                    write(*,*) 'FCT outward water CFL:',dt/max(q(1,i,j),tiny(1.d0))*( &
                        (max(-fx(i,j),0.d0)+max(fx(i+1,j),0.d0))/dx+ &
                        (max(-fy(i,j),0.d0)+max(fy(i,j+1),0.d0))/dy)
                    call fail('FCT low-order state violates local bounds')
                endif

                dwest= dt/dx*(highx(i,j)-fsx(i,j))
                deast=-dt/dx*(highx(i+1,j)-fsx(i+1,j))
                dsouth= dt/dy*(highy(i,j)-fsy(i,j))
                dnorth=-dt/dy*(highy(i,j+1)-fsy(i,j+1))
                pplus(i,j)=max(dwest,0.d0)+max(deast,0.d0) &
                          +max(dsouth,0.d0)+max(dnorth,0.d0)
                pminus(i,j)=max(-dwest,0.d0)+max(-deast,0.d0) &
                           +max(-dsouth,0.d0)+max(-dnorth,0.d0)
                allowance=max(cmax(i,j)*hlow(i,j)-slow(i,j),0.d0)
                if (pplus(i,j)>0.d0) &
                    rplus(i,j)=min(1.d0,allowance/pplus(i,j))
                allowance=max(slow(i,j)-cmin(i,j)*hlow(i,j),0.d0)
                if (pminus(i,j)>0.d0) &
                    rminus(i,j)=min(1.d0,allowance/pminus(i,j))
            enddo
        enddo

        do j=jb,jt
            do i=il,ir+1
                fm(4,i,j)=fsx(i,j); fp(4,i,j)=fsx(i,j)
            enddo
            do i=il+1,ir
                anti=highx(i,j)-fsx(i,j)
                if (anti>=0.d0) then
                    alpha=min(rminus(i-1,j),rplus(i,j))
                else
                    alpha=min(rplus(i-1,j),rminus(i,j))
                endif
                alpha=alpha*(1.d0-1.d-12)
                fm(4,i,j)=fsx(i,j)+alpha*anti
                fp(4,i,j)=fm(4,i,j)
            enddo
            if (xperdom) then
                tolerance=4096.d0*epsilon(1.d0)*max(1.d0,abs(highx(il,j)), &
                                                     abs(highx(ir+1,j)))
                if (abs(highx(il,j)-highx(ir+1,j))>tolerance.or. &
                    abs(fsx(il,j)-fsx(ir+1,j))>tolerance) &
                    call fail('periodic x sediment fluxes do not match')
                anti=0.5d0*((highx(il,j)-fsx(il,j)) &
                           +(highx(ir+1,j)-fsx(ir+1,j)))
                if (anti>=0.d0) then
                    alpha=min(rminus(ir,j),rplus(il,j))
                else
                    alpha=min(rplus(ir,j),rminus(il,j))
                endif
                alpha=alpha*(1.d0-1.d-12)
                c=0.5d0*(fsx(il,j)+fsx(ir+1,j))+alpha*anti
                fm(4,il,j)=c; fp(4,il,j)=c
                fm(4,ir+1,j)=c; fp(4,ir+1,j)=c
            endif
        enddo

        do i=il,ir
            do j=jb,jt+1
                gm(4,i,j)=fsy(i,j); gp(4,i,j)=fsy(i,j)
            enddo
            do j=jb+1,jt
                anti=highy(i,j)-fsy(i,j)
                if (anti>=0.d0) then
                    alpha=min(rminus(i,j-1),rplus(i,j))
                else
                    alpha=min(rplus(i,j-1),rminus(i,j))
                endif
                alpha=alpha*(1.d0-1.d-12)
                gm(4,i,j)=fsy(i,j)+alpha*anti
                gp(4,i,j)=gm(4,i,j)
            enddo
            if (yperdom) then
                tolerance=4096.d0*epsilon(1.d0)*max(1.d0,abs(highy(i,jb)), &
                                                     abs(highy(i,jt+1)))
                if (abs(highy(i,jb)-highy(i,jt+1))>tolerance.or. &
                    abs(fsy(i,jb)-fsy(i,jt+1))>tolerance) &
                    call fail('periodic y sediment fluxes do not match')
                anti=0.5d0*((highy(i,jb)-fsy(i,jb)) &
                           +(highy(i,jt+1)-fsy(i,jt+1)))
                if (anti>=0.d0) then
                    alpha=min(rminus(i,jt),rplus(i,jb))
                else
                    alpha=min(rplus(i,jt),rminus(i,jb))
                endif
                alpha=alpha*(1.d0-1.d-12)
                c=0.5d0*(fsy(i,jb)+fsy(i,jt+1))+alpha*anti
                gm(4,i,jb)=c; gp(4,i,jb)=c
                gm(4,i,jt+1)=c; gp(4,i,jt+1)=c
            endif
        enddo
        deallocate(fx,fy,fsx,fsy,highx,highy,chalf,cbefore,hlow,slow, &
                   cmin,cmax,pplus,pminus,rplus,rminus)
    end subroutine braids_passive_fluxes
end module braids_passive_module
