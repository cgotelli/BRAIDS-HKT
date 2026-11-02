! GeoClaw pre-step hook with dry-state handling consistent with the Riemann solver.
subroutine b4step2(mbc,mx,my,meqn,q,xlower,ylower,dx,dy,t,dt, &
                   maux,aux,actualstep)

    use geoclaw_module, only: dry_tolerance, speed_limit
    use topo_module, only: aux_finalized
    use amr_module, only: NEEDS_TO_BE_SET
    use storm_module, only: set_storm_fields

    implicit none
    integer, intent(in) :: meqn
    integer, intent(inout) :: mbc,mx,my,maux
    real(kind=8), intent(inout) :: xlower,ylower,dx,dy,t,dt
    real(kind=8), intent(inout) :: q(meqn,1-mbc:mx+mbc,1-mbc:my+mbc)
    real(kind=8), intent(inout) :: aux(maux,1-mbc:mx+mbc,1-mbc:my+mbc)
    logical, intent(in) :: actualstep
    integer :: i,j
    real(kind=8) :: speed,scale

    call check4nans(meqn,mbc,mx,my,q,t,1)

    ! GeoClaw's Riemann solver classifies h <= dry_tolerance as dry.  Clear
    ! momentum with the same inclusive comparison before dividing by depth.
    forall(i=1-mbc:mx+mbc,j=1-mbc:my+mbc,q(1,i,j)<=dry_tolerance)
        q(1,i,j)=max(q(1,i,j),0.d0)
        q(2:3,i,j)=0.d0
    end forall

    do j=1-mbc,my+mbc
        do i=1-mbc,mx+mbc
            if (q(1,i,j)>dry_tolerance) then
                speed=sqrt(q(2,i,j)**2+q(3,i,j)**2)/q(1,i,j)
                if (speed>speed_limit) then
                    scale=speed_limit/speed
                    q(2,i,j)=q(2,i,j)*scale
                    q(3,i,j)=q(3,i,j)*scale
                endif
            endif
        enddo
    enddo

    if (aux_finalized<2 .and. actualstep) then
        aux(1,:,:)=NEEDS_TO_BE_SET
        call setaux(mbc,mx,my,xlower,ylower,dx,dy,maux,aux)
    endif
    if (actualstep) then
        call set_storm_fields(maux,mbc,mx,my,xlower,ylower,dx,dy,t,aux)
    endif
end subroutine b4step2
