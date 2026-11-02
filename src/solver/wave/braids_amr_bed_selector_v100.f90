module braids_amr_bed_selector_v100_module
  use iso_c_binding
  implicit none
contains
  subroutine braids_amr_bed_selector_v100(nc, ng, nt, mode, combine, bed, water, diffused, transfer, cleanup, &
      exchange, divergence, selected, final_bed, status) bind(C)
    integer(c_int), value :: nc, ng, nt, mode
    integer(c_int), intent(in) :: combine(ng)
    real(c_double), intent(in) :: bed(nc,ng), water(nc,ng), diffused(nc,ng), transfer(nc,ng), cleanup(nc,ng)
    real(c_double), intent(out) :: exchange(nc,ng), divergence(nc,ng), selected(nc,ng), final_bed(nc,ng)
    integer(c_int), intent(out) :: status(nc,ng)
    integer :: i, k
    ! Each lane/cell is independent. No nested teams or new transport solve.
    !$omp parallel do collapse(2) schedule(static) num_threads(nt) private(i,k)
    do k=1,ng
      do i=1,nc
        exchange(i,k)=0.d0
        divergence(i,k)=0.d0
        if (combine(k) /= 0) then
          exchange(i,k)=-transfer(i,k)
          if (mode == 1) divergence(i,k)=diffused(i,k)-water(i,k)
        endif
        selected(i,k)=0.d0
        if (mode == 0) selected(i,k)=exchange(i,k)
        if (mode == 1) selected(i,k)=divergence(i,k)
        final_bed(i,k)=bed(i,k)+selected(i,k)+cleanup(i,k)
        status(i,k)=0
        if (final_bed(i,k) < 0.d0) status(i,k)=1
      enddo
    enddo
    !$omp end parallel do
  end subroutine
end module
