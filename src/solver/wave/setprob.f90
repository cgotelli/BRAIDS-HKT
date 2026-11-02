! BRAIDS water-only default; the original shoreline source is explicit opt-in.
subroutine setprob
    use internal_inflow_module, only: read_internal_inflow
    use hydraulic_bc_module, only: setup_hydraulic_bc
    use braids_momentum_diffusion_module, only: configure_braids_momentum_diffusion
    implicit none
    character(len=1024) :: inflow_file
    integer :: unit, ios
    logical :: exists

    ! A hydraulic run needs no setprob.data, CL directory or AVAC outputs.
    inquire(file="setprob.data", exist=exists)
    if (exists) then
        unit = 72
        call opendatafile(unit, "setprob.data")
        read(unit,*,iostat=ios) inflow_file
        close(unit)
        if (ios /= 0) then
            write(6,*) "ERROR: malformed optional setprob.data"
            stop 1
        endif
        if (trim(inflow_file) /= "none") then
            call read_internal_inflow(trim(inflow_file))
        endif
    endif
    ! Existing rectangular benchmark boundaries; not BRAIDS's section API.
    call setup_hydraulic_bc()
    ! Publish immutable viscosity configuration before OpenMP patch updates.
    call configure_braids_momentum_diffusion()
end subroutine setprob
