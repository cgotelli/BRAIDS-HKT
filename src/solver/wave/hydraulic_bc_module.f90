! Optional prescribed hydraulic boundaries for reproducible water benchmarks.
!
! hydraulic_bc.data has four rows: side, mode, stage, discharge.
! side: 1=west, 2=east, 3=south, 4=north
! mode: 0=unused, 1=stage, 2=unit discharge, 3=stage+unit discharge,
!       4=total discharge, 5=stage+total discharge.
module hydraulic_bc_module
    implicit none
    save

    integer :: hydraulic_mode(4)=0
    real(kind=8) :: hydraulic_stage(4)=0.d0
    real(kind=8) :: hydraulic_discharge(4)=0.d0
    logical :: hydraulic_bc_ready=.false.

contains

    subroutine setup_hydraulic_bc(path)
        character(len=*), optional, intent(in) :: path
        character(len=256) :: fname
        character(len=512) :: line
        integer :: unit,ios,k,side,mode_value
        real(kind=8) :: stage_value,discharge_value
        logical :: exists

        if (hydraulic_bc_ready) return
        fname='hydraulic_bc.data'
        if (present(path)) fname=path
        inquire(file=trim(fname),exist=exists)
        if (.not.exists) then
            hydraulic_bc_ready=.true.
            return
        endif
        unit=74
        open(unit=unit,file=trim(fname),status='old',action='read',iostat=ios)
        if (ios/=0) stop 'Unable to open hydraulic_bc.data.'
        k=0
        do while(k<4)
            read(unit,'(a)',iostat=ios) line
            if (ios/=0) stop 'Unable to read hydraulic_bc.data.'
            line=adjustl(line)
            if (len_trim(line)==0 .or. line(1:1)=='#') cycle
            read(line,*,iostat=ios) side,mode_value,stage_value,discharge_value
            if (ios/=0 .or. side<1 .or. side>4) stop 'Invalid hydraulic boundary row.'
            if (mode_value<0 .or. mode_value>5) stop 'Hydraulic mode must be 0..5.'
            hydraulic_mode(side)=mode_value
            hydraulic_stage(side)=stage_value
            hydraulic_discharge(side)=discharge_value
            k=k+1
        enddo
        close(unit)
        hydraulic_bc_ready=.true.
    end subroutine setup_hydraulic_bc


    logical function hydraulic_bc_defined(side)
        integer, intent(in) :: side
        call setup_hydraulic_bc()
        hydraulic_bc_defined=hydraulic_mode(side)/=0
    end function hydraulic_bc_defined

    subroutine apply_hydraulic_bc(side,val,aux,nrow,ncol,meqn,naux, &
                                  hx,hy,xlo_patch,ylo_patch,nghost_side)
        use amr_module, only: xlower,xupper,ylower,yupper
        integer, intent(in) :: side,nrow,ncol,meqn,naux,nghost_side
        real(kind=8), intent(in) :: hx,hy,xlo_patch,ylo_patch
        real(kind=8), intent(inout) :: val(meqn,nrow,ncol)
        real(kind=8), intent(inout) :: aux(naux,nrow,ncol)
        integer :: i,j,iref,jref,ilo,ihi,jlo,jhi,mode
        real(kind=8) :: h,h_ref,u_ref,v_ref,area,normal_velocity
        real(kind=8) :: x,y,stage,discharge
        logical :: prescribe_stage,prescribe_unit,prescribe_total

        call setup_hydraulic_bc()
        mode=hydraulic_mode(side)
        if (mode==0) stop 'User boundary has no hydraulic definition.'
        stage=hydraulic_stage(side)
        discharge=hydraulic_discharge(side)
        prescribe_stage=(mode==1 .or. mode==3 .or. mode==5)
        prescribe_unit=(mode==2 .or. mode==3)
        prescribe_total=(mode==4 .or. mode==5)

        ilo=1; ihi=nrow; jlo=1; jhi=ncol
        if (side==1) then
            iref=nghost_side+1; ihi=nghost_side
        else if (side==2) then
            iref=nrow-nghost_side; ilo=iref+1
        else if (side==3) then
            jref=nghost_side+1; jhi=nghost_side
        else
            jref=ncol-nghost_side; jlo=jref+1
        endif

        ! Start with zero-order extrapolation of the adjacent interior state.
        if (side<=2) then
            do j=1,ncol; do i=ilo,ihi
                aux(:,i,j)=aux(:,iref,j); val(:,i,j)=val(:,iref,j)
            enddo; enddo
        else
            do j=jlo,jhi; do i=1,nrow
                aux(:,i,j)=aux(:,i,jref); val(:,i,j)=val(:,i,jref)
            enddo; enddo
        endif

        if (prescribe_stage) then
            do j=jlo,jhi; do i=ilo,ihi
                h_ref=val(1,i,j)
                if (h_ref>0.d0) then
                    u_ref=val(2,i,j)/h_ref; v_ref=val(3,i,j)/h_ref
                else
                    u_ref=0.d0; v_ref=0.d0
                endif
                h=max(stage-aux(1,i,j),0.d0)
                val(1,i,j)=h; val(2,i,j)=h*u_ref; val(3,i,j)=h*v_ref
            enddo; enddo
        endif

        if (prescribe_unit) then
            do j=jlo,jhi; do i=ilo,ihi
                if (side<=2) then
                    val(2,i,j)=discharge; val(3,i,j)=0.d0
                else
                    val(2,i,j)=0.d0; val(3,i,j)=discharge
                endif
            enddo; enddo
        else if (prescribe_total) then
            area=0.d0
            if (side<=2) then
                do j=1,ncol
                    y=ylo_patch+(j-0.5d0)*hy
                    if (y>=ylower .and. y<=yupper) then
                        if (prescribe_stage) then
                            h=max(stage-aux(1,iref,j),0.d0)
                        else
                            h=max(val(1,iref,j),0.d0)
                        endif
                        area=area+h*hy
                    endif
                enddo
            else
                do i=1,nrow
                    x=xlo_patch+(i-0.5d0)*hx
                    if (x>=xlower .and. x<=xupper) then
                        if (prescribe_stage) then
                            h=max(stage-aux(1,i,jref),0.d0)
                        else
                            h=max(val(1,i,jref),0.d0)
                        endif
                        area=area+h*hx
                    endif
                enddo
            endif
            if (area<=0.d0 .and. .not.prescribe_stage .and. stage/=0.d0) then
                if (side<=2) then
                    do j=1,ncol
                        y=ylo_patch+(j-0.5d0)*hy
                        if (y>=ylower .and. y<=yupper) then
                            h=max(stage-aux(1,iref,j),0.d0); area=area+h*hy
                            do i=ilo,ihi
                                val(1,i,j)=h; val(2,i,j)=0.d0; val(3,i,j)=0.d0
                            enddo
                        endif
                    enddo
                else
                    do i=1,nrow
                        x=xlo_patch+(i-0.5d0)*hx
                        if (x>=xlower .and. x<=xupper) then
                            h=max(stage-aux(1,i,jref),0.d0); area=area+h*hx
                            do j=jlo,jhi
                                val(1,i,j)=h; val(2,i,j)=0.d0; val(3,i,j)=0.d0
                            enddo
                        endif
                    enddo
                endif
            endif
            if (area<=0.d0) stop 'Total-discharge boundary has zero wet area.'
            normal_velocity=discharge/area
            do j=jlo,jhi; do i=ilo,ihi
                h=val(1,i,j)
                if (side<=2) then
                    val(2,i,j)=h*normal_velocity; val(3,i,j)=0.d0
                else
                    val(2,i,j)=0.d0; val(3,i,j)=h*normal_velocity
                endif
            enddo; enddo
        endif
    end subroutine apply_hydraulic_bc
end module hydraulic_bc_module
