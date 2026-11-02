module braids_maximum_waterdepth_v66_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private
    public :: braids_maximum_waterdepth_v66
contains
    subroutine braids_maximum_waterdepth_v66(ncell,nlink,cell_depth,transport_u,transport_v, &
                                             active,support_offset,support_depth,support_active, &
                                             enabled,direction_epsilon,observation_depth, &
                                             characteristic_u,characteristic_v,characteristic_speed, &
                                             observation_height,status)
        integer, intent(in) :: ncell,nlink,enabled
        real(8), intent(in) :: cell_depth(ncell),transport_u(ncell),transport_v(ncell)
        integer, intent(in) :: active(ncell),support_active(nlink)
        integer(8), intent(in) :: support_offset(ncell+1)
        real(8), intent(in) :: support_depth(nlink),direction_epsilon
        real(8), intent(out) :: observation_depth(ncell),characteristic_u(ncell)
        real(8), intent(out) :: characteristic_v(ncell),characteristic_speed(ncell)
        real(8), intent(out) :: observation_height(ncell)
        integer, intent(out) :: status(ncell)
        integer :: cell,item,first,last
        real(8) :: h,uu,vv

        !$omp parallel do default(none) schedule(static) &
        !$omp shared(ncell,nlink,cell_depth,transport_u,transport_v,active,support_offset, &
        !$omp        support_depth,support_active,enabled,direction_epsilon,observation_depth, &
        !$omp        characteristic_u,characteristic_v,characteristic_speed,observation_height,status) &
        !$omp private(cell,item,first,last,h,uu,vv)
        do cell=1,ncell
            observation_depth(cell)=0.d0
            characteristic_u(cell)=0.d0
            characteristic_v(cell)=0.d0
            characteristic_speed(cell)=0.d0
            observation_height(cell)=0.d0
            status(cell)=0
            if (active(cell)==0) cycle
            if (active(cell)/=1 .or. .not.ieee_is_finite(cell_depth(cell)) .or. &
                cell_depth(cell)<=0.d0 .or. .not.ieee_is_finite(transport_u(cell)) .or. &
                .not.ieee_is_finite(transport_v(cell)) .or. &
                .not.ieee_is_finite(direction_epsilon) .or. direction_epsilon<=0.d0) then
                status(cell)=1
                cycle
            endif
            first=int(support_offset(cell))+1
            last=int(support_offset(cell+1))
            if (first<1 .or. first>nlink+1 .or. last<first-1 .or. last>nlink) then
                status(cell)=1
                cycle
            endif
            h=cell_depth(cell)
            if (enabled/=0) then
                do item=first,last
                    if (support_active(item)/=0 .and. support_active(item)/=1) then
                        status(cell)=1
                        exit
                    endif
                    if (.not.ieee_is_finite(support_depth(item)) .or. support_depth(item)<0.d0) then
                        status(cell)=1
                        exit
                    endif
                    if (support_active(item)==1) h=max(h,support_depth(item))
                enddo
                if (status(cell)/=0) cycle
            endif
            uu=transport_u(cell)/h
            vv=transport_v(cell)/h
            if (abs(uu)<direction_epsilon) uu=0.d0
            if (abs(vv)<direction_epsilon) vv=0.d0
            observation_depth(cell)=h
            characteristic_u(cell)=uu
            characteristic_v(cell)=vv
            characteristic_speed(cell)=sqrt(uu*uu+vv*vv)
            observation_height(cell)=real(0.368,8)*h
        enddo
        !$omp end parallel do
    end subroutine braids_maximum_waterdepth_v66
end module braids_maximum_waterdepth_v66_module
