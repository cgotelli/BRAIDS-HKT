! Speed-only counterpart of Clawpack 5.14's rpn2_geoclaw.f for CFL preflight.
! The accepted step still uses the complete augmented Riemann solver.
subroutine rpn2_cfl(ixy,maxm,meqn,mwaves,maux,mbc,mx, &
                    ql,qr,auxl,auxr,fwave,s,amdq,apdq)
    use geoclaw_module, only: g => grav, drytol => dry_tolerance
    use geoclaw_module, only: earth_radius, deg2rad
    use amr_module, only: mcapa
    implicit none

    integer :: ixy,maxm,meqn,mwaves,maux,mbc,mx
    real(kind=8) :: ql(meqn,1-mbc:maxm+mbc),qr(meqn,1-mbc:maxm+mbc)
    real(kind=8) :: auxl(maux,1-mbc:maxm+mbc),auxr(maux,1-mbc:maxm+mbc)
    real(kind=8) :: fwave(meqn,mwaves,1-mbc:maxm+mbc)
    real(kind=8) :: s(mwaves,1-mbc:maxm+mbc)
    real(kind=8) :: amdq(meqn,1-mbc:maxm+mbc),apdq(meqn,1-mbc:maxm+mbc)
    integer :: i,mu,mw
    real(kind=8) :: hL,hR,huL,huR,uL,uR,bL,bR,wall(3),sw(3)
    real(kind=8) :: sL,sR,uhat,chat,sRoe1,sRoe2,sE1,sE2
    real(kind=8) :: hstar,hstartest,hm,s1m,s2m,dxdc
    logical :: rare1,rare2

    ! The callback signature matches rpn2, but these zero flux outputs must
    ! never be used to advance q. Only step2_cfl may call this routine.
    fwave = 0.d0
    s = 0.d0
    amdq = 0.d0
    apdq = 0.d0

    do i=2-mbc,mx+mbc
        ! Preserve the original diagnostic and slice-local input cleanup,
        ! including when the caller aliases ql and qr.
        if (qr(1,i-1)<0.d0 .or. ql(1,i)<0.d0) then
            write(*,*) 'Negative input: hl,hr,i=',qr(1,i-1),ql(1,i),i
        endif
        if (ixy==1) then
            mu=2
        else
            mu=3
        endif
        if (qr(1,i-1)<0.d0) then
            qr(1:3,i-1)=0.d0
        endif
        if (ql(1,i)<0.d0) then
            ql(1:3,i)=0.d0
        endif
        if (qr(1,i-1)<=drytol .and. ql(1,i)<=drytol) cycle

        hL=qr(1,i-1)
        hR=ql(1,i)
        huL=qr(mu,i-1)
        huR=ql(mu,i)
        bL=auxr(1,i-1)
        bR=auxl(1,i)
        if (hR>drytol) then
            uR=huR/hR
        else
            hR=0.d0
            huR=0.d0
            uR=0.d0
        endif
        if (hL>drytol) then
            uL=huL/hL
        else
            hL=0.d0
            huL=0.d0
            uL=0.d0
        endif

        ! These inundation/wall tests precede the wave-speed estimate in
        ! rpn2. A wall mirrors the wet state and masks the same wave speeds.
        wall=1.d0
        if (hR<=drytol) then
            call riemanntype(hL,hL,uL,-uL,hstar,s1m,s2m,rare1,rare2,1,drytol,g)
            hstartest=max(hL,hstar)
            if (hstartest+bL<bR) then
                wall(2)=0.d0
                wall(3)=0.d0
                hR=hL
                huR=-huL
                bR=bL
                uR=-uL
            elseif (hL+bL<bR) then
                bR=hL+bL
            endif
        elseif (hL<=drytol) then
            call riemanntype(hR,hR,-uR,uR,hstar,s1m,s2m,rare1,rare2,1,drytol,g)
            hstartest=max(hR,hstar)
            if (hstartest+bR<bL) then
                wall(1)=0.d0
                wall(2)=0.d0
                hL=hR
                huL=-huR
                bL=bR
                uL=-uR
            elseif (hR+bR<bL) then
                bL=hR+bR
            endif
        endif

        ! Keep the expression order identical to the full normal solver.
        sL=uL-sqrt(g*hL)
        sR=uR+sqrt(g*hR)
        uhat=(sqrt(g*hL)*uL + sqrt(g*hR)*uR)/(sqrt(g*hR)+sqrt(g*hL))
        chat=sqrt(g*0.5d0*(hR+hL))
        sRoe1=uhat-chat
        sRoe2=uhat+chat
        sE1=min(sL,sRoe1)
        sE2=max(sR,sRoe2)

        ! riemann_aug_JCP refines these bounds with the same one-iteration
        ! riemanntype call. Its rarecorrectortest is hard-coded false, so
        ! the middle speed is the arithmetic mean. The subsequent pressure
        ! balancing and wave decomposition do not change any wave speed.
        call riemanntype(hL,hR,uL,uR,hm,s1m,s2m,rare1,rare2,1,drytol,g)
        sw(1)=min(sE1,s2m)
        sw(3)=max(sE2,s1m)
        sw(2)=0.5d0*(sw(1)+sw(3))
        do mw=1,3
            sw(mw)=sw(mw)*wall(mw)
        enddo
        do mw=1,mwaves
            s(mw,i)=sw(mw)
        enddo
    enddo

    ! Match rpn2's existing mapped-coordinate scaling, not a new CFL rule.
    if (mcapa>0) then
        do i=2-mbc,mx+mbc
            if (ixy==1) then
                dxdc=(earth_radius*deg2rad)
            else
                dxdc=earth_radius*cos(auxl(3,i))*deg2rad
            endif
            do mw=1,mwaves
                s(mw,i)=dxdc*s(mw,i)
            enddo
        enddo
    endif
end subroutine rpn2_cfl
