! Native finite-volume scalar Exner candidate: WENO-Z/MC + SSPRK(3,3).
! Verification scope: local equilibrium flux F(z); NOT the full SWE-Exner system.
! One shared limited flux per face; physical flux, derivative bound and boundary
! states are supplied by the caller. No Python runs inside a time step.
module braids_exner_mol_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private
    public :: braids_exner_step,reconstruct
    abstract interface
        pure function scalar_flux(z,axis) result(f)
            real(8), intent(in) :: z
            integer, intent(in) :: axis
            real(8) :: f
        end function
    end interface
contains
    pure function mc(a,b) result(s)
        real(8), intent(in) :: a,b
        real(8) :: s
        s=0.d0
        if(a*b>0.d0) s=sign(min(2*abs(a),2*abs(b),.5d0*abs(a+b)),a)
    end function

    pure function reconstruct(v,scheme) result(s)
        real(8), intent(in) :: v(5)
        integer, intent(in) :: scheme
        real(8) :: s,p(3),b(3),w(3),tau,eps
        if(scheme==2) then
            s=v(3)+.5d0*mc(v(3)-v(2),v(4)-v(3)); return
        endif
        p=[2*v(1)-7*v(2)+11*v(3),-v(2)+5*v(3)+2*v(4),2*v(3)+5*v(4)-v(5)]/6.d0
        b(1)=13.d0/12*(v(1)-2*v(2)+v(3))**2+.25d0*(v(1)-4*v(2)+3*v(3))**2
        b(2)=13.d0/12*(v(2)-2*v(3)+v(4))**2+.25d0*(v(2)-v(4))**2
        b(3)=13.d0/12*(v(3)-2*v(4)+v(5))**2+.25d0*(3*v(3)-4*v(4)+v(5))**2
        tau=abs(b(1)-b(3)); eps=1.d-40+1.d-14*maxval(v*v)
        w=[.1d0,.6d0,.3d0]*(1.d0+(tau/(b+eps))**2)
        s=sum(w*p)/sum(w)
    end function

    subroutine euler(nx,ny,z,dt,dx,dy,flux,speed,lo,hi,bc,scheme,next,fx,fy,theta_min)
        integer, intent(in) :: nx,ny,scheme
        real(8), intent(in) :: z(nx,ny),dt,dx,dy,speed(2),lo,hi,bc(4)
        procedure(scalar_flux) :: flux
        real(8), intent(out) :: next(nx,ny),fx(nx+1,ny),fy(nx,ny+1),theta_min
        real(8) :: padded(-2:nx+3,-2:ny+3),lx(nx+1,ny),ly(nx,ny+1)
        real(8) :: ax(nx+1,ny),ay(nx,ny+1),rp(nx,ny),rm(nx,ny)
        real(8) :: l,r,pp,pm,a(4),theta
        integer :: i,j
        ! NaN boundary state means extrapolation; finite value means Dirichlet.
        do j=-2,ny+3
            do i=-2,nx+3
                padded(i,j)=z(max(1,min(nx,i)),max(1,min(ny,j)))
                if(i<1.and.ieee_is_finite(bc(1))) padded(i,j)=bc(1)
                if(i>nx.and.ieee_is_finite(bc(2))) padded(i,j)=bc(2)
                if(j<1.and.ieee_is_finite(bc(3))) padded(i,j)=bc(3)
                if(j>ny.and.ieee_is_finite(bc(4))) padded(i,j)=bc(4)
            enddo
        enddo
!$omp parallel do default(shared) private(i,j,l,r)
        do j=1,ny
            do i=1,nx+1
                l=padded(i-1,j); r=padded(i,j)
                lx(i,j)=.5d0*(flux(l,1)+flux(r,1)-speed(1)*(r-l))
                l=max(lo,min(hi,reconstruct(padded(i-3:i+1,j),scheme)))
                r=max(lo,min(hi,reconstruct(padded(i+2:i-2:-1,j),scheme)))
                fx(i,j)=.5d0*(flux(l,1)+flux(r,1)-speed(1)*(r-l))
            enddo
        enddo
!$omp end parallel do
!$omp parallel do default(shared) private(i,j,l,r)
        do j=1,ny+1
            do i=1,nx
                l=padded(i,j-1); r=padded(i,j)
                ly(i,j)=.5d0*(flux(l,2)+flux(r,2)-speed(2)*(r-l))
                l=max(lo,min(hi,reconstruct(padded(i,j-3:j+1),scheme)))
                r=max(lo,min(hi,reconstruct(padded(i,j+2:j-2:-1),scheme)))
                fy(i,j)=.5d0*(flux(l,2)+flux(r,2)-speed(2)*(r-l))
            enddo
        enddo
!$omp end parallel do
        next=z-dt/dx*(lx(2:,:)-lx(:nx,:))-dt/dy*(ly(:,2:)-ly(:,:ny))
        if(minval(next)<lo-1.d-12.or.maxval(next)>hi+1.d-12) error stop 'Exner low-order bounds/CFL'
        ax=dt/dx*(fx-lx); ay=dt/dy*(fy-ly)
        ! Conservative antidiffusive flux correction, against declared invariant
        ! scalar bounds. No clipping of cell averages and no loss of bed solids.
        do j=1,ny
            do i=1,nx
                a=[ax(i,j),-ax(i+1,j),ay(i,j),-ay(i,j+1)]
                pp=sum(max(a,0.d0)); pm=sum(max(-a,0.d0))
                rp(i,j)=1.d0; rm(i,j)=1.d0
                if(pp>0.d0) rp(i,j)=min(1.d0,max(0.d0,hi-next(i,j))/pp)
                if(pm>0.d0) rm(i,j)=min(1.d0,max(0.d0,next(i,j)-lo)/pm)
            enddo
        enddo
        theta_min=1.d0
        do j=1,ny
            do i=1,nx+1
                theta=1.d0
                if(ax(i,j)>0.d0) then
                    if(i>1) theta=min(theta,rm(i-1,j))
                    if(i<=nx) theta=min(theta,rp(i,j))
                else if(ax(i,j)<0.d0) then
                    if(i>1) theta=min(theta,rp(i-1,j))
                    if(i<=nx) theta=min(theta,rm(i,j))
                endif
                theta_min=min(theta_min,theta)
                fx(i,j)=lx(i,j)+theta*(fx(i,j)-lx(i,j))
            enddo
        enddo
        do j=1,ny+1
            do i=1,nx
                theta=1.d0
                if(ay(i,j)>0.d0) then
                    if(j>1) theta=min(theta,rm(i,j-1))
                    if(j<=ny) theta=min(theta,rp(i,j))
                else if(ay(i,j)<0.d0) then
                    if(j>1) theta=min(theta,rp(i,j-1))
                    if(j<=ny) theta=min(theta,rm(i,j))
                endif
                theta_min=min(theta_min,theta)
                fy(i,j)=ly(i,j)+theta*(fy(i,j)-ly(i,j))
            enddo
        enddo
        next=z-dt/dx*(fx(2:,:)-fx(:nx,:))-dt/dy*(fy(:,2:)-fy(:,:ny))
    end subroutine

    subroutine braids_exner_step(nx,ny,z,dt,dx,dy,flux,speed,lo,hi,bc,scheme,fx,fy,theta_min)
        integer, intent(in) :: nx,ny,scheme
        real(8), intent(inout) :: z(nx,ny)
        real(8), intent(in) :: dt,dx,dy,speed(2),lo,hi,bc(4)
        procedure(scalar_flux) :: flux
        real(8), intent(out) :: fx(nx+1,ny),fy(nx,ny+1),theta_min
        real(8) :: z1(nx,ny),z2(nx,ny),z3(nx,ny),gx(nx+1,ny),gy(nx,ny+1),theta
        if(nx<1.or.ny<1.or.(scheme/=2.and.scheme/=5)) error stop 'Exner dimensions/scheme'
        if(.not.all(ieee_is_finite([dt,dx,dy,speed,lo,hi]))) error stop 'Exner nonfinite controls'
        if(min(dt,dx,dy)<=0.d0.or.minval(speed)<0.d0.or.hi<=lo) error stop 'Exner controls'
        if(dt*(speed(1)/dx+speed(2)/dy)>.45d0) error stop 'Exner CFL exceeds 0.45'
        if(.not.all(ieee_is_finite(z))) error stop 'Exner nonfinite state'
        if(minval(z)<lo-1.d-12.or.maxval(z)>hi+1.d-12) error stop 'Exner state outside bounds'
        call euler(nx,ny,z,dt,dx,dy,flux,speed,lo,hi,bc,scheme,z1,fx,fy,theta_min)
        fx=fx/6.d0; fy=fy/6.d0
        call euler(nx,ny,z1,dt,dx,dy,flux,speed,lo,hi,bc,scheme,z2,gx,gy,theta)
        theta_min=min(theta_min,theta); fx=fx+gx/6.d0; fy=fy+gy/6.d0
        z2=.75d0*z+.25d0*z2
        call euler(nx,ny,z2,dt,dx,dy,flux,speed,lo,hi,bc,scheme,z3,gx,gy,theta)
        theta_min=min(theta_min,theta); fx=fx+2.d0/3*gx; fy=fy+2.d0/3*gy
        ! Reconstruct final state from accepted RK-weighted fluxes: the ledger
        ! and state use the same divergence, including physical boundary faces.
        z=z-dt/dx*(fx(2:,:)-fx(:nx,:))-dt/dy*(fy(:,2:)-fy(:,:ny))
    end subroutine
end module
