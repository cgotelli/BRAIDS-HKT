! Conservative shared-face bedload capacity reconstruction and donor limiting.
! Independent of AMR, files and closures; arrays contain interior cells only.
module braids_bedload_flux_module
    implicit none
    private
    public :: braids_bedload_faces,braids_bedload_limit
contains
    pure real(8) function mc(a,b) result(s)
        real(8), intent(in) :: a,b
        s=0.d0
        if (a*b>0.d0) s=sign(min(2.d0*abs(a),2.d0*abs(b),0.5d0*abs(a+b)),a)
    end function

    subroutine braids_bedload_faces(nx,ny,dt,dx,dy,mass,wet,qx,qy,fx,fy,minimum_theta)
        integer, intent(in) :: nx,ny
        real(8), intent(in) :: dt,dx,dy,mass(nx,ny),qx(nx,ny),qy(nx,ny)
        logical, intent(in) :: wet(nx,ny)
        ! Caller installs physical boundary fluxes first; these are also donor limited.
        real(8), intent(inout) :: fx(nx+1,ny),fy(nx,ny+1)
        real(8), intent(out) :: minimum_theta
        real(8) :: sx(nx,ny),sy(nx,ny),theta(nx,ny),left,right,outgoing,sign_flow
        integer :: i,j
        sx=0.d0; sy=0.d0
        if (nx>1) then
            where(wet(1,:).and.wet(2,:)) sx(1,:)=qx(2,:)-qx(1,:)
            where(wet(nx,:).and.wet(nx-1,:)) sx(nx,:)=qx(nx,:)-qx(nx-1,:)
        endif
        if (ny>1) then
            where(wet(:,1).and.wet(:,2)) sy(:,1)=qy(:,2)-qy(:,1)
            where(wet(:,ny).and.wet(:,ny-1)) sy(:,ny)=qy(:,ny)-qy(:,ny-1)
        endif
!$omp parallel do default(none) shared(nx,ny,qx,qy,sx,sy,wet) private(i,j)
        do j=1,ny
            do i=1,nx
                if (i>1.and.i<nx) then
                    if (wet(i-1,j).and.wet(i,j).and.wet(i+1,j)) &
                        sx(i,j)=mc(qx(i,j)-qx(i-1,j),qx(i+1,j)-qx(i,j))
                endif
                if (j>1.and.j<ny) then
                    if (wet(i,j-1).and.wet(i,j).and.wet(i,j+1)) &
                        sy(i,j)=mc(qy(i,j)-qy(i,j-1),qy(i,j+1)-qy(i,j))
                endif
            enddo
        enddo
!$omp end parallel do
!$omp parallel do default(none) shared(nx,ny,qx,qy,sx,sy,wet,fx,fy) private(i,j,left,right,sign_flow)
        do j=1,ny
            do i=2,nx
                fx(i,j)=0.d0
                if (.not.(wet(i-1,j).and.wet(i,j))) cycle
                left=qx(i-1,j)+0.5d0*sx(i-1,j); right=qx(i,j)-0.5d0*sx(i,j)
                ! Converging/diverging opposite transport directions are blocked
                ! in this flat-bed capacity path; a sign-consistent donor is required.
                if (left*right<0.d0) cycle
                sign_flow=left+right
                if (sign_flow>0.d0) fx(i,j)=left
                if (sign_flow<0.d0) fx(i,j)=right
            enddo
            if (j==1) cycle
            do i=1,nx
                fy(i,j)=0.d0
                if (.not.(wet(i,j-1).and.wet(i,j))) cycle
                left=qy(i,j-1)+0.5d0*sy(i,j-1); right=qy(i,j)-0.5d0*sy(i,j)
                if (left*right<0.d0) cycle
                sign_flow=left+right
                if (sign_flow>0.d0) fy(i,j)=left
                if (sign_flow<0.d0) fy(i,j)=right
            enddo
        enddo
!$omp end parallel do
        call braids_bedload_limit(nx,ny,dt,dx,dy,mass,fx,fy,minimum_theta)
    end subroutine

    subroutine braids_bedload_limit(nx,ny,dt,dx,dy,mass,fx,fy,minimum_theta)
        integer, intent(in) :: nx,ny
        real(8), intent(in) :: dt,dx,dy,mass(nx,ny)
        real(8), intent(inout) :: fx(nx+1,ny),fy(nx,ny+1)
        real(8), intent(out) :: minimum_theta
        real(8) :: theta(nx,ny),outgoing
        integer :: i,j
        ! Initial stock only: incoming bedload is not spent again within this step.
        do j=1,ny
            do i=1,nx
                outgoing=dt*((max(fx(i+1,j),0.d0)+max(-fx(i,j),0.d0))/dx &
                            +(max(fy(i,j+1),0.d0)+max(-fy(i,j),0.d0))/dy)
                theta(i,j)=1.d0
                if (outgoing>max(mass(i,j),0.d0)) &
                    theta(i,j)=max(mass(i,j),0.d0)/outgoing*(1.d0-8.d0*epsilon(1.d0))
            enddo
        enddo
        minimum_theta=minval(theta)
        do j=1,ny
            do i=1,nx+1
                if (fx(i,j)>0.d0.and.i>1) fx(i,j)=fx(i,j)*theta(i-1,j)
                if (fx(i,j)<0.d0.and.i<=nx) fx(i,j)=fx(i,j)*theta(i,j)
            enddo
        enddo
        do j=1,ny+1
            do i=1,nx
                if (fy(i,j)>0.d0.and.j>1) fy(i,j)=fy(i,j)*theta(i,j-1)
                if (fy(i,j)<0.d0.and.j<=ny) fy(i,j)=fy(i,j)*theta(i,j)
            enddo
        enddo
    end subroutine
end module
