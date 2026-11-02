! Reusable private storage for WAVE's noncommitting CFL probes.
module wave_cfl_buffer_module
    use, intrinsic :: iso_fortran_env, only: int64
    implicit none
    private
    public :: get_wave_cfl_buffers

    ! Each worker owns its allocations. Flat storage permits an exact active
    ! shape on every call, even when AMR changes the patch dimensions. Passing
    ! a section of a larger 3-D allocation would have the wrong leading
    ! dimensions or require an expensive compiler-generated copy.
    real(kind=8), allocatable, target, save :: q_storage(:), aux_storage(:)
!$omp threadprivate(q_storage, aux_storage)

contains

    subroutine get_wave_cfl_buffers(nvar, naux, mitot, mjtot, qwork, auxwork)
        integer, intent(in) :: nvar, naux, mitot, mjtot
        real(kind=8), pointer, contiguous, intent(out) :: qwork(:,:,:), auxwork(:,:,:)
        integer(int64) :: q_count, aux_count

        ! Returned pointers are for one active probe on this worker only.
        ! Contents deliberately persist; the caller must overwrite all active
        ! q/aux entries from the accepted state before every trial, including
        ! retries with smaller timesteps or earlier patch times.
        if (nvar < 1 .or. naux < 0 .or. mitot < 1 .or. mjtot < 1) then
            error stop 'Invalid WAVE CFL scratch dimensions'
        endif
        q_count = int(nvar, int64) * int(mitot, int64) * int(mjtot, int64)
        aux_count = int(max(1, naux), int64) * int(mitot, int64) * int(mjtot, int64)
        if (allocated(q_storage)) then
            if (size(q_storage, kind=int64) < q_count) deallocate(q_storage)
        endif
        if (.not. allocated(q_storage)) allocate(q_storage(q_count))
        if (allocated(aux_storage)) then
            if (size(aux_storage, kind=int64) < aux_count) deallocate(aux_storage)
        endif
        if (.not. allocated(aux_storage)) allocate(aux_storage(aux_count))

        qwork(1:nvar, 1:mitot, 1:mjtot) => q_storage(1:q_count)
        auxwork(1:max(1,naux), 1:mitot, 1:mjtot) => aux_storage(1:aux_count)
    end subroutine get_wave_cfl_buffers
end module wave_cfl_buffer_module
