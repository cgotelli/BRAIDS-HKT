! G4 Cartesian draining-time correction. External physical fluxes are fixed.
! Only donor advective transfer is reduced; the original pressure/source
! fluctuation jump is preserved by adding the same correction on both sides.
module braids_draining_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private
    public :: braids_draining_corrections, braids_draining_roundoff
contains
    ! Repair only floating-point cancellation in the fluctuation update.
    ! Transfer water across ONE shared interior face, debiting its neighbour;
    ! never clip a cell or alter a prescribed exterior mass flux. Momentum uses
    ! the same donor velocity as the ordinary draining correction. Material
    ! undershoots still abort. This is used only by the opt-in RK path.
    subroutine braids_draining_roundoff(nv,mi,mj,mbc,q,dt,dx,dy,fm,fp,gm,gp,operand_scale)
        integer, intent(in) :: nv,mi,mj,mbc
        real(8), intent(in) :: q(nv,mi,mj),dt,dx,dy
        real(8), intent(inout) :: fm(nv,mi,mj),fp(nv,mi,mj),gm(nv,mi,mj),gp(nv,mi,mj)
        real(8), optional, intent(in) :: operand_scale(mi,mj)
        real(8) :: h(mi,mj),scale,tol,margin,amount,best,delta,velocity(3)
        integer :: i,j,k,ii,jj,di,dj,face,pass,il,ir,jb,jt
        integer, parameter :: oi(4)=[-1,1,0,0],oj(4)=[0,0,-1,1]
        il=mbc+1;ir=mi-mbc;jb=mbc+1;jt=mj-mbc
        do pass=1,3
            h=0.d0
            h(il:ir,jb:jt)=q(1,il:ir,jb:jt) &
                -dt/dx*(fm(1,il+1:ir+1,jb:jt)-fp(1,il:ir,jb:jt)) &
                -dt/dy*(gm(1,il:ir,jb+1:jt+1)-gp(1,il:ir,jb:jt))
            if (.not.all(ieee_is_finite(h))) error stop 'RK roundoff: nonfinite depth'
            if (minval(h(il:ir,jb:jt))>=0.d0) return
            do j=jb,jt
                do i=il,ir
                    if(h(i,j)>=0.d0) cycle
                    scale=max(tiny(1.d0),abs(q(1,i,j)), &
                        dt/dx*(abs(fm(1,i+1,j))+abs(fp(1,i,j))) &
                       +dt/dy*(abs(gm(1,i,j+1))+abs(gp(1,i,j))))
                    ! Each shared mass flux is reconstructed from a fluctuation
                    ! plus cell momentum. Include those cancelled operands.
                    scale=max(scale,dt/dx*maxval(abs(q(2,i-1:i+1,j))) &
                                   +dt/dy*maxval(abs(q(3,i,j-1:j+1))))
                    ! Draining can cancel an entire outgoing flux. Its
                    ! pre-cancellation scale also depends on adjacent wet
                    ! states in the Riemann stencil, even if this cell is dry.
                    scale=max(scale,maxval(abs(q(1,i-1:i+1,j-1:j+1))))
                    margin=32.d0*epsilon(1.d0)*scale
                    if(present(operand_scale)) scale=max(scale,operand_scale(i,j))
                    tol=256.d0*epsilon(1.d0)*scale
                    if(h(i,j)<-tol) then
                        write(*,*) 'RK negative depth: i,j,h,tolerance',i-mbc,j-mbc,h(i,j),tol
                        error stop 'RK roundoff: material negative depth'
                    endif
                    ! The acceptance bound includes terrain datum cancellation;
                    ! the actual transfer uses only local update roundoff.
                    amount=2.d0*abs(h(i,j))+margin
                    best=0.d0;di=0;dj=0
                    do k=1,4
                        ii=i+oi(k);jj=j+oj(k)
                        if(ii<il.or.ii>ir.or.jj<jb.or.jj>jt) cycle
                        if(h(ii,jj)<=best) cycle
                        best=h(ii,jj);di=oi(k);dj=oj(k)
                    enddo
                    ! A neighbour may have just wetted during this Euler stage.
                    ! Its provisional water is available for this roundoff-only
                    ! transfer, even when its pre-stage depth was zero.
                    amount=2.d0*abs(h(i,j))+min(margin,.125d0*best)
                    if(best<2.d0*amount.or.(di==0.and.dj==0)) &
                        error stop 'RK roundoff: no local donor'
                    velocity(1)=1.d0
                    velocity(2:3)=0.d0
                    if(q(1,i+di,j+dj)>0.d0) velocity(2:3)=q(2:3,i+di,j+dj)/q(1,i+di,j+dj)
                    if(di/=0) then
                        face=i+max(di,0);delta=-di*amount*dx/dt
                        fm(1:3,face,j)=fm(1:3,face,j)+delta*velocity
                        fp(1:3,face,j)=fp(1:3,face,j)+delta*velocity
                    else
                        face=j+max(dj,0);delta=-dj*amount*dy/dt
                        gm(1:3,i,face)=gm(1:3,i,face)+delta*velocity
                        gp(1:3,i,face)=gp(1:3,i,face)+delta*velocity
                    endif
                    h(i,j)=h(i,j)+amount
                    h(i+di,j+dj)=h(i+di,j+dj)-amount
                enddo
            enddo
        enddo
        error stop 'RK roundoff: correction failed to converge'
    end subroutine braids_draining_roundoff

    subroutine braids_draining_corrections(nx,ny,q,fx,fy,dx,dy,dt,delta_x,delta_y,theta, &
                                         limited_faces,limited_cells,min_unlimited_depth,boundary_overdraw,enforce_donor_cfl)
        integer, intent(in) :: nx,ny
        real(kind=8), intent(in) :: q(3,nx,ny),fx(nx+1,ny),fy(nx,ny+1),dx,dy,dt
        real(kind=8), intent(out) :: delta_x(3,nx+1,ny),delta_y(3,nx,ny+1),theta(nx,ny)
        integer, intent(out) :: limited_faces,limited_cells
        real(kind=8), intent(out) :: min_unlimited_depth
        logical, intent(out) :: boundary_overdraw
        logical, optional, intent(in) :: enforce_donor_cfl
        logical :: strict_donor
        integer :: i,j,donor
        real(kind=8) :: available,outgoing,fixed,scale,provisional,change,velocity(2),margin
        if (nx<1.or.ny<1.or.min(dx,dy,dt)<=0.d0) error stop 'BRAIDS draining: invalid cell/time dimensions'
        if (.not.all(ieee_is_finite(q)).or..not.all(ieee_is_finite(fx)).or. &
            .not.all(ieee_is_finite(fy)).or..not.all(ieee_is_finite([dx,dy,dt]))) &
            error stop 'BRAIDS draining: nonfinite state or flux'
        if (any(q(1,:,:)<0.d0)) error stop 'BRAIDS draining: negative prepared water depth'
        strict_donor=.false.
        if (present(enforce_donor_cfl)) strict_donor=enforce_donor_cfl
        delta_x=0.d0; delta_y=0.d0; theta=1.d0
        limited_faces=0; limited_cells=0; boundary_overdraw=.false.
        min_unlimited_depth=huge(1.d0)
        do j=1,ny
            do i=1,nx
                provisional=q(1,i,j)-dt/dx*(fx(i+1,j)-fx(i,j))-dt/dy*(fy(i,j+1)-fy(i,j))
                min_unlimited_depth=min(min_unlimited_depth,provisional)
                fixed=0.d0
                if (i==1) fixed=fixed+dt/dx*fx(i,j)
                if (i==nx) fixed=fixed-dt/dx*fx(i+1,j)
                if (j==1) fixed=fixed+dt/dy*fy(i,j)
                if (j==ny) fixed=fixed-dt/dy*fy(i,j+1)
                available=q(1,i,j)+fixed
                ! Passive upwinding spends only the initial donor inventory.
                ! Water entering during this step is not yet mixed into its
                ! donor concentration, even at a prescribed external inlet.
                if (strict_donor) available=q(1,i,j)+min(fixed,0.d0)
                scale=max(q(1,i,j),abs(fixed),tiny(1.d0))
                if (available < -64.d0*epsilon(1.d0)*scale) boundary_overdraw=.true.
                outgoing=0.d0
                if (i>1) outgoing=outgoing+dt/dx*max(-fx(i,j),0.d0)
                if (i<nx) outgoing=outgoing+dt/dx*max(fx(i+1,j),0.d0)
                if (j>1) outgoing=outgoing+dt/dy*max(-fy(i,j),0.d0)
                if (j<ny) outgoing=outgoing+dt/dy*max(fy(i,j+1),0.d0)
                ! Incoming interior transfers are nonnegative even if their
                ! donors are limited. They are deliberately not spent twice.
                if (outgoing>max(available,0.d0)) then
                    margin=1.d0-64.d0*epsilon(1.d0)
                    theta(i,j)=max(0.d0,min(1.d0,available/outgoing*margin))
                    limited_cells=limited_cells+1
                endif
            enddo
        enddo
        ! Keep the original high-order flux bit-for-bit whenever its complete
        ! update is already nonnegative; no limiter diffusion is then needed.
        if (min_unlimited_depth>=0.d0.and..not.strict_donor) then
            theta=1.d0; limited_cells=0; boundary_overdraw=.false.
            return
        endif
        if (boundary_overdraw) return
        do j=1,ny
            do i=2,nx
                if (fx(i,j)>=0.d0) then
                    donor=i-1
                else
                    donor=i
                endif
                change=(theta(donor,j)-1.d0)*fx(i,j)
                if (change==0.d0) cycle
                velocity=0.d0
                if (q(1,donor,j)>0.d0) velocity=q(2:3,donor,j)/q(1,donor,j)
                delta_x(1,i,j)=change
                delta_x(2:3,i,j)=change*velocity
                limited_faces=limited_faces+1
            enddo
        enddo
        do j=2,ny
            do i=1,nx
                if (fy(i,j)>=0.d0) then
                    donor=j-1
                else
                    donor=j
                endif
                change=(theta(i,donor)-1.d0)*fy(i,j)
                if (change==0.d0) cycle
                velocity=0.d0
                if (q(1,i,donor)>0.d0) velocity=q(2:3,i,donor)/q(1,i,donor)
                delta_y(1,i,j)=change
                delta_y(2:3,i,j)=change*velocity
                limited_faces=limited_faces+1
            enddo
        enddo
        if (.not.all(ieee_is_finite(delta_x)).or..not.all(ieee_is_finite(delta_y))) &
            error stop 'BRAIDS draining: nonfinite advective correction'
    end subroutine braids_draining_corrections
end module braids_draining_module
