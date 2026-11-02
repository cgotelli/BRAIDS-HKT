! Stage-consistent fixed-level SWE + suspended advection + bedload transport.
! GeoClaw's normal f-wave operator is evaluated at dt=0 (method of lines);
! its spatial limiter remains active, its Lax-Wendroff/transverse time terms do not.
! Bedload uses reconstructed hydraulic states, not a frozen scalar F(z).
module braids_rk_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use amr_module, only: nghost,xlower,ylower,alloc,node,listOfGrids,listStart,numgrids, &
        store1,storeaux,ndilo,ndihi,ndjlo,ndjhi
    use geoclaw_module, only: grav,dry_tolerance,speed_limit
    use braids_bedload_module, only: braids_bedload_prepare,braids_bedload_accept,braids_bedload_bulk
    use braids_boundary_module, only: braids_boundary_prepare,braids_sediment_boundary_flux
    use braids_passive_module, only: braids_passive_fluxes,braids_passive_wetdry_fluxes
    use braids_draining_module, only: braids_draining_corrections,braids_draining_roundoff
    implicit none
    private
    public :: braids_rk_fluxes
    abstract interface
        subroutine hydro_operator(nv,na,mi,mj,q,aux,fm,fp,gm,gp)
            integer, intent(in) :: nv,na,mi,mj
            real(8), intent(in) :: q(nv,mi,mj),aux(na,mi,mj)
            real(8), intent(out) :: fm(nv,mi,mj),fp(nv,mi,mj),gm(nv,mi,mj),gp(nv,mi,mj)
        end subroutine
    end interface
contains
    subroutine braids_rk_fluxes(nv,na,mi,mj,time,dt,dx,dy,q0,a0,fm,fp,gm,gp,hydro)
        integer, intent(in) :: nv,na,mi,mj
        real(8), intent(in) :: time,dt,dx,dy,q0(nv,mi,mj),a0(na,mi,mj)
        real(8), intent(out) :: fm(nv,mi,mj),fp(nv,mi,mj),gm(nv,mi,mj),gp(nv,mi,mj)
        procedure(hydro_operator) :: hydro
        real(8) :: q(nv,mi,mj),aux(na,mi,mj),next(nv,mi,mj),ffm(nv,mi,mj),ffp(nv,mi,mj)
        real(8) :: ggm(nv,mi,mj),ggp(nv,mi,mj),weight,theta,min_theta,bulk,cfl,speed
        real(8) :: operand_scale(mi,mj)
        real(8), parameter :: weights(3)=[1.d0/6,1.d0/6,2.d0/3]
        integer :: stage,i,j,il,ir,jb,jt
        external bc2amr
        il=nghost+1;ir=mi-nghost;jb=nghost+1;jt=mj-nghost
        bulk=braids_bedload_bulk();q=q0;fm=0.d0;fp=0.d0;gm=0.d0;gp=0.d0;min_theta=1.d0
        ! The f-wave bathymetric source subtracts absolute bed/stage
        ! elevations; their rounding scale persists even in a thin dry front.
        operand_scale=abs(a0(1,:,:))
        do stage=1,3
            aux=a0
            aux(1,il:ir,jb:jt)=a0(1,il:ir,jb:jt)+(q(5,il:ir,jb:jt)-q0(5,il:ir,jb:jt))/bulk
            if(.not.all(ieee_is_finite(q)).or.minval(q(1,il:ir,jb:jt))<0.d0.or. &
                minval(q(5,il:ir,jb:jt))<0.d0) error stop 'Invalid coupled RK stage'
            ! Match GeoClaw's wet/dry state preparation at EVERY RK stage.
            ! Dry cells can acquire momentum from the previous Euler stage;
            ! retaining it while the Riemann solver treats them as dry makes
            ! the reconstructed physical mass flux inconsistent with q.
            where(q(1,:,:)<=dry_tolerance)
                q(2,:,:)=0.d0
                q(3,:,:)=0.d0
            endwhere
            do j=1,mj
                do i=1,mi
                    if(q(1,i,j)<=dry_tolerance) cycle
                    speed=sqrt(q(2,i,j)**2+q(3,i,j)**2)/q(1,i,j)
                    if(speed>speed_limit) q(2:3,i,j)=q(2:3,i,j)*(speed_limit/speed)
                enddo
            enddo
            cfl=0.d0
            do j=jb,jt
                do i=il,ir
                    if(q(1,i,j)<=dry_tolerance) cycle
                    cfl=max(cfl,dt*((abs(q(2,i,j)/q(1,i,j))+sqrt(grav*q(1,i,j)))/dx &
                                  +(abs(q(3,i,j)/q(1,i,j))+sqrt(grav*q(1,i,j)))/dy))
                enddo
            enddo
            if(cfl>.45d0) error stop 'Coupled RK stage exceeds CFL 0.45'
            ! The boundary module queries AMR storage. Publish each stage there
            ! temporarily; restore the original state before physical advancement.
            call publish(nv,na,mi,mj,q,aux)
            call braids_boundary_prepare(1,time,dt,nv,na,rk_stage=.true.)
            call bc2amr(q,aux,mi,mj,nv,na,dx,dy,1,time,xlower-nghost*dx, &
                xlower+(mi-nghost)*dx,ylower-nghost*dy,ylower+(mj-nghost)*dy)
            call hydro(nv,na,mi,mj,q,aux,ffm,ffp,ggm,ggp)
            ! Preserve the operands BEFORE draining cancellation, including
            ! earlier stages, for a forward roundoff bound on the final sum.
            operand_scale(il:ir,jb:jt)=max(operand_scale(il:ir,jb:jt),abs(q(1,il:ir,jb:jt)) &
                +dt/dx*(abs(ffm(1,il+1:ir+1,jb:jt))+abs(ffp(1,il:ir,jb:jt))) &
                +dt/dy*(abs(ggm(1,il:ir,jb+1:jt+1))+abs(ggp(1,il:ir,jb:jt))))
            call drain(nv,mi,mj,q,dt,dx,dy,ffm,ffp,ggm,ggp)
            call braids_draining_roundoff(nv,mi,mj,nghost,q,dt,dx,dy,ffm,ffp,ggm,ggp,operand_scale)
            call braids_passive_fluxes(nv,mi,mj,nghost,dt,dx,dy,q,ffm,ffp,ggm,ggp,mol=.true.)
            call braids_sediment_boundary_flux(nv,mi,mj,nghost,dx,dy,xlower-nghost*dx, &
                ylower-nghost*dy,q,ffm,ffp,ggm,ggp)
            call braids_passive_wetdry_fluxes(nv,mi,mj,nghost,dt,dx,dy,xlower,ylower,q,ffm,ffp,ggm,ggp)
            call braids_bedload_prepare(nv,mi,mj,nghost,time,dt,dx,dy,q,ffm,ffp,ggm,ggp, &
                record=.false.,theta_out=theta)
            min_theta=min(min_theta,theta)
            next=q
            next(:,il:ir,jb:jt)=q(:,il:ir,jb:jt)-dt/dx*(ffm(:,il+1:ir+1,jb:jt)-ffp(:,il:ir,jb:jt)) &
                                                  -dt/dy*(ggm(:,il:ir,jb+1:jt+1)-ggp(:,il:ir,jb:jt))
            ! Rebase the water fluctuations onto the original cell momenta.
            ! This cancels in each cell divergence and makes the existing water
            ! ledger reconstruct the RK-weighted physical boundary flux exactly.
            do j=jb,jt
                do i=il,ir+1
                    ffm(1,i,j)=ffm(1,i,j)+q(2,i-1,j)-q0(2,i-1,j)
                    ffp(1,i,j)=ffp(1,i,j)+q(2,i,j)-q0(2,i,j)
                enddo
            enddo
            do j=jb,jt+1
                do i=il,ir
                    ggm(1,i,j)=ggm(1,i,j)+q(3,i,j-1)-q0(3,i,j-1)
                    ggp(1,i,j)=ggp(1,i,j)+q(3,i,j)-q0(3,i,j)
                enddo
            enddo
            weight=weights(stage)
            fm=fm+weight*ffm;fp=fp+weight*ffp;gm=gm+weight*ggm;gp=gp+weight*ggp
            if(stage==1) q=next
            if(stage==2) q=.75d0*q0+.25d0*next
        enddo
        ! Protect the final weighted fluctuation update from the same cancellation.
        call braids_draining_roundoff(nv,mi,mj,nghost,q0,dt,dx,dy,fm,fp,gm,gp,operand_scale)
        call publish(nv,na,mi,mj,q0,a0)
        call braids_boundary_prepare(1,time,dt,nv,na,rk_stage=.true.)
        call braids_bedload_accept(nv,mi,mj,nghost,time,dt,dx,dy,q0,fm,gm,min_theta)
    end subroutine

    subroutine publish(nv,na,mi,mj,q,aux)
        integer, intent(in) :: nv,na,mi,mj
        real(8), intent(in) :: q(nv,mi,mj),aux(na,mi,mj)
        integer :: k,id,nx,ny,i,j,ig,jg,lq,la
        do k=1,numgrids(1)
            id=listOfGrids(listStart(1)+k-1)
            nx=node(ndihi,id)-node(ndilo,id)+1+2*nghost
            ny=node(ndjhi,id)-node(ndjlo,id)+1+2*nghost
            lq=node(store1,id);la=node(storeaux,id)
            do j=1,ny
                jg=node(ndjlo,id)+j
                do i=1,nx
                    ig=node(ndilo,id)+i
                    alloc(lq+nv*((j-1)*nx+i-1):lq+nv*((j-1)*nx+i)-1)=q(:,ig,jg)
                    alloc(la+na*((j-1)*nx+i-1):la+na*((j-1)*nx+i)-1)=aux(:,ig,jg)
                enddo
            enddo
        enddo
    end subroutine

    subroutine drain(nv,mi,mj,q,dt,dx,dy,fm,fp,gm,gp)
        integer, intent(in) :: nv,mi,mj
        real(8), intent(in) :: q(nv,mi,mj),dt,dx,dy
        real(8), intent(inout) :: fm(nv,mi,mj),fp(nv,mi,mj),gm(nv,mi,mj),gp(nv,mi,mj)
        integer :: nx,ny,il,ir,jb,jt,i,j,nfaces,ncells
        real(8) :: minimum
        real(8) :: fx(mi-2*nghost+1,mj-2*nghost),fy(mi-2*nghost,mj-2*nghost+1)
        real(8) :: cx(3,mi-2*nghost+1,mj-2*nghost),cy(3,mi-2*nghost,mj-2*nghost+1)
        real(8) :: theta(mi-2*nghost,mj-2*nghost)
        logical :: overdraw
        il=nghost+1;ir=mi-nghost;jb=nghost+1;jt=mj-nghost;nx=ir-il+1;ny=jt-jb+1
        fx(1,:)=q(2,il,jb:jt)+fp(1,il,jb:jt)
        fx(2:,:)=q(2,il:ir,jb:jt)+fm(1,il+1:ir+1,jb:jt)
        fy(:,1)=q(3,il:ir,jb)+gp(1,il:ir,jb)
        fy(:,2:)=q(3,il:ir,jb:jt)+gm(1,il:ir,jb+1:jt+1)
        call braids_draining_corrections(nx,ny,q(1:3,il:ir,jb:jt),fx,fy,dx,dy,dt, &
            cx,cy,theta,nfaces,ncells,minimum,overdraw,enforce_donor_cfl=.true.)
        if(overdraw) error stop 'Coupled RK boundary overdraw'
        fm(1:3,il:ir+1,jb:jt)=fm(1:3,il:ir+1,jb:jt)+cx
        fp(1:3,il:ir+1,jb:jt)=fp(1:3,il:ir+1,jb:jt)+cx
        gm(1:3,il:ir,jb:jt+1)=gm(1:3,il:ir,jb:jt+1)+cy
        gp(1:3,il:ir,jb:jt+1)=gp(1:3,il:ir,jb:jt+1)+cy
    end subroutine
end module
