module internal_inflow_module
    implicit none
    save

    integer :: inflow_ntimes = 0
    integer :: inflow_ncells = 0
    integer :: inflow_version = 0
    real(kind=8), allocatable :: inflow_times(:)
    real(kind=8), allocatable :: inflow_x(:), inflow_y(:)
    real(kind=8), allocatable :: inflow_rates(:,:,:)
    ! Source locations are immutable after loading.  Their bounding box lets
    ! patches outside the shoreline skip both the time and source searches.
    real(kind=8), private :: inflow_xmin, inflow_xmax, inflow_ymin, inflow_ymax

contains

    subroutine read_internal_inflow(fname)
        character(len=*), intent(in) :: fname
        integer :: unit, version, it, icell, io

        unit = 73
        open(unit, file=trim(fname), status="old", action="read", iostat=io)
        if (io /= 0) then
            write(6,*) "*** Unable to open internal Wave inflow: ", trim(fname)
            stop
        end if
        read(unit,*,iostat=io) version
        if (io /= 0 .or. (version /= 1 .and. version /= 2)) then
            write(6,*) "*** Unsupported internal Wave inflow format"
            stop
        end if
        inflow_version = version
        read(unit,*,iostat=io) inflow_ntimes, inflow_ncells
        if (io /= 0 .or. inflow_ntimes < 2 .or. inflow_ncells < 0) then
            write(6,*) "*** Invalid internal Wave inflow dimensions"
            stop
        end if

        allocate(inflow_times(inflow_ntimes))
        allocate(inflow_x(inflow_ncells), inflow_y(inflow_ncells))
        allocate(inflow_rates(3,inflow_ntimes,inflow_ncells))
        inflow_rates = 0.d0
        read(unit,*,iostat=io) inflow_times
        if (io /= 0) then
            write(6,*) "*** Unable to read internal Wave inflow times"
            stop
        end if
        do it = 2, inflow_ntimes
            if (inflow_times(it) <= inflow_times(it-1)) then
                write(6,*) "*** Internal Wave inflow times are not increasing"
                stop
            end if
        end do
        do icell = 1, inflow_ncells
            read(unit,*,iostat=io) inflow_x(icell), inflow_y(icell)
            if (io /= 0) then
                write(6,*) "*** Unable to read internal Wave source cell"
                stop
            end if
            do it = 1, inflow_ntimes
                read(unit,*,iostat=io) inflow_rates(:,it,icell)
                if (io /= 0) then
                    write(6,*) "*** Unable to read internal Wave source rates"
                    stop
                end if
            end do
        end do
        close(unit)
        if (inflow_ncells > 0) then
            inflow_xmin = minval(inflow_x)
            inflow_xmax = maxval(inflow_x)
            inflow_ymin = minval(inflow_y)
            inflow_ymax = maxval(inflow_y)
        end if
        write(6,*) "Internal shoreline inflow cells: ", inflow_ncells
    end subroutine read_internal_inflow


    subroutine apply_internal_inflow(meqn, mbc, mx, my, xlower, ylower, dx, dy, q, t, dt)
        integer, intent(in) :: meqn, mbc, mx, my
        real(kind=8), intent(in) :: xlower, ylower, dx, dy, t, dt
        real(kind=8), intent(inout) :: q(meqn,1-mbc:mx+mbc,1-mbc:my+mbc)
        integer :: it, icell, i, j, lo, hi, midpoint
        real(kind=8) :: sample_time, weight, rate(3)

        if (inflow_ncells == 0 .or. dt <= 0.d0) return
        ! Match the existing half-open membership test below: lower edges
        ! belong to the patch, upper edges belong to its neighbor.
        if (inflow_xmax < xlower .or. inflow_xmin >= xlower + mx*dx) return
        if (inflow_ymax < ylower .or. inflow_ymin >= ylower + my*dy) return
        sample_time = t + 0.5d0*dt
        if (sample_time < inflow_times(1) .or. sample_time > inflow_times(inflow_ntimes)) return

        if (sample_time == inflow_times(inflow_ntimes)) then
            it = inflow_ntimes - 1
            weight = 1.d0
        else
            ! Stateless binary search also supports rejected/repeated times,
            ! AMR subcycling, and concurrent patch workers without a cache.
            lo = 1
            hi = inflow_ntimes
            do while (hi-lo > 1)
                midpoint = lo + (hi-lo)/2
                if (sample_time < inflow_times(midpoint)) then
                    hi = midpoint
                else
                    lo = midpoint
                end if
            end do
            it = lo
            weight = (sample_time - inflow_times(it)) / (inflow_times(it+1) - inflow_times(it))
        end if

        do icell = 1, inflow_ncells
            if (inflow_x(icell) < xlower .or. inflow_x(icell) >= xlower + mx*dx) cycle
            if (inflow_y(icell) < ylower .or. inflow_y(icell) >= ylower + my*dy) cycle
            i = int(floor((inflow_x(icell) - xlower) / dx)) + 1
            j = int(floor((inflow_y(icell) - ylower) / dy)) + 1
            if (i < 1 .or. i > mx .or. j < 1 .or. j > my) cycle
            rate = (1.d0-weight)*inflow_rates(:,it,icell) + weight*inflow_rates(:,it+1,icell)
            ! Version 1 stored dh/dt, dhu/dt and dhv/dt for the base-grid
            ! target cell.  That convention made the integrated source depend
            ! on AMR level because a refined cell has a smaller area.  Version
            ! 2 stores the conservative total rates Q, Q*u and Q*v for each
            ! shoreline source point.  Divide by the area of the patch cell
            ! that actually receives the source, preserving the same mass and
            ! momentum on every AMR level.  Version 1 remains readable so an
            ! already-prepared scenario does not fail abruptly.
            if (inflow_version == 2) rate = rate / (dx*dy)
            q(1:3,i,j) = q(1:3,i,j) + dt*rate
        end do
    end subroutine apply_internal_inflow

end module internal_inflow_module
