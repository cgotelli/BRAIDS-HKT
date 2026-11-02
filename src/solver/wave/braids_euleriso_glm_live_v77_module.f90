module braids_euleriso_glm_live_v77_module
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use omp_lib, only: omp_get_num_threads
    implicit none
    private

    integer, parameter, public :: euleriso_glm_v77_ok = 0
    integer, parameter, public :: euleriso_glm_v77_invalid = 10
    integer, parameter, public :: euleriso_glm_v77_unsupported_engine = 20
    integer, parameter, public :: euleriso_glm_v77_incompatible = 21
    integer, parameter, public :: euleriso_glm_v77_missing_forcing = 22
    integer, parameter, public :: euleriso_glm_v77_unsupported_wave = 23
    integer, parameter, public :: euleriso_glm_v77_orphaned_forcing = 24
    integer, parameter, public :: euleriso_glm_v77_update_limit = 30
    integer, parameter, public :: euleriso_glm_v77_cancelled = 31
    integer, parameter, public :: euleriso_glm_v77_invalid_state = 32
    integer, parameter, public :: euleriso_glm_v77_parent_mismatch = 33
    integer, parameter, public :: euleriso_glm_v77_invalid_trial = 34
    integer, parameter, public :: euleriso_glm_v77_engine_classic = 1
    integer, parameter, public :: euleriso_glm_v77_engine_fm = 2
    integer, parameter, public :: euleriso_glm_v77_max_updates = 4
    character(len=64), parameter, public :: euleriso_glm_v77_semantic = &
        '15bc080fbf510727f8e8ccbf537cac0a56ebce06df583dcb112764590d5e8923'

    type, public :: braids_euleriso_glm_state_v77
        private
        real(real64), allocatable :: concentration(:,:)
        integer :: accepted_updates = 0
        integer :: identity_checks = 0
        integer :: euleriso_active_updates = 0
        integer :: glmiso_logged_updates = 0
        integer :: engine = 0
        logical :: current_only = .false.
        logical :: wave_enabled = .false.
        logical :: roller_enabled = .false.
        logical :: forcing_present = .false.
        logical :: euleriso_requested = .false.
        logical :: glmiso_requested = .false.
        integer(int64) :: lineage = 0_int64
        integer(int64) :: generation = -1_int64
        integer(int64) :: integrity = 0_int64
        character(len=64) :: semantic = ''
        logical :: valid = .false.
    end type braids_euleriso_glm_state_v77

    type, public :: braids_euleriso_glm_trial_v77
        private
        type(braids_euleriso_glm_state_v77) :: candidate
        real(real64), allocatable :: input_concentration(:,:)
        real(real64), allocatable :: output_concentration(:,:)
        real(real64), allocatable :: control_delta(:,:)
        integer(int64) :: parent_integrity = 0_int64
        integer(int64) :: parent_lineage = 0_int64
        integer(int64) :: parent_generation = -1_int64
        integer(int64) :: integrity = 0_int64
        integer :: observed_parallel_workers = 0
        logical :: identity_exact = .false.
        logical :: consumed = .false.
        logical :: valid = .false.
    end type braids_euleriso_glm_trial_v77

    public :: braids_resolve_euleriso_glm_v77
    public :: braids_euler_velocity_probe_v77
    public :: braids_rolcor_flux_probe_v77
    public :: braids_initialize_euleriso_glm_state_v77
    public :: braids_begin_euleriso_glm_trial_v77
    public :: braids_commit_euleriso_glm_trial_v77
    public :: braids_rollback_euleriso_glm_trial_v77
    public :: braids_cancel_euleriso_glm_trial_v77
    public :: braids_checkpoint_euleriso_glm_v77
    public :: braids_restart_euleriso_glm_v77
    public :: braids_state_snapshot_v77
    public :: braids_trial_snapshot_v77

contains

    subroutine braids_resolve_euleriso_glm_v77(engine,current_only, &
            wave_enabled,roller_enabled,forcing_present,euleriso_glm, &
            glmiso_euler,euler_effective,glm_effective,status)
        integer, intent(in) :: engine
        logical, intent(in) :: current_only,wave_enabled,roller_enabled
        logical, intent(in) :: forcing_present,euleriso_glm,glmiso_euler
        logical, intent(out) :: euler_effective,glm_effective
        integer, intent(out) :: status

        euler_effective = .false.
        glm_effective = .false.
        if (engine /= euleriso_glm_v77_engine_classic) then
            status = euleriso_glm_v77_unsupported_engine
        elseif (.not.current_only) then
            status = euleriso_glm_v77_incompatible
        elseif (roller_enabled .and. .not.wave_enabled) then
            status = euleriso_glm_v77_incompatible
        elseif (forcing_present .and. .not.(wave_enabled .and. roller_enabled)) then
            status = euleriso_glm_v77_orphaned_forcing
        elseif (wave_enabled .and. roller_enabled .and. euleriso_glm .and. &
                .not.forcing_present) then
            status = euleriso_glm_v77_missing_forcing
        elseif (wave_enabled .or. roller_enabled) then
            status = euleriso_glm_v77_unsupported_wave
        else
            status = euleriso_glm_v77_ok
        endif
    end subroutine braids_resolve_euleriso_glm_v77

    subroutine braids_euler_velocity_probe_v77(nface,euleriso_glm, &
            wave_enabled,forcing_present,roller_mode,hu,hv,grmasu,grmasv, &
            grfacu,grfacv,kfu,kfv,du,dv,active_count,workers,status)
        integer, intent(in) :: nface,roller_mode
        logical, intent(in) :: euleriso_glm,wave_enabled,forcing_present
        real(real64), intent(in) :: hu(nface),hv(nface),grmasu(nface)
        real(real64), intent(in) :: grmasv(nface),grfacu(nface),grfacv(nface)
        integer, intent(in) :: kfu(nface),kfv(nface)
        real(real64), intent(out) :: du(nface),dv(nface)
        integer, intent(out) :: active_count,workers,status
        integer :: i
        logical :: route_ready

        du = 0.0_real64
        dv = 0.0_real64
        active_count = 0
        workers = 1
        route_ready = euleriso_glm .and. wave_enabled
        if (route_ready .and. .not.forcing_present) then
            status = euleriso_glm_v77_missing_forcing
            return
        endif
        if (nface < 1 .or. roller_mode < 0 .or. roller_mode > 2 .or. &
                any(.not.ieee_is_finite(hu)) .or. &
                any(.not.ieee_is_finite(hv)) .or. any(hu <= 0.0_real64) .or. &
                any(hv <= 0.0_real64) .or. &
                any(.not.ieee_is_finite(grmasu)) .or. &
                any(.not.ieee_is_finite(grmasv)) .or. &
                any(.not.ieee_is_finite(grfacu)) .or. &
                any(.not.ieee_is_finite(grfacv))) then
            status = euleriso_glm_v77_invalid
            return
        endif
        if (route_ready) then
            !$omp parallel shared(workers)
            !$omp single
            workers = omp_get_num_threads()
            !$omp end single
            !$omp do reduction(+:active_count)
            do i=1,nface
                if (kfu(i) == 1) then
                    if (roller_mode > 0) then
                        du(i) = -(grmasu(i)+grfacu(i))/hu(i)
                    else
                        du(i) = -grfacu(i)/hu(i)
                    endif
                endif
                if (kfv(i) == 1) then
                    if (roller_mode > 0) then
                        dv(i) = -(grmasv(i)+grfacv(i))/hv(i)
                    else
                        dv(i) = -grfacv(i)/hv(i)
                    endif
                endif
                if (kfu(i) == 1 .or. kfv(i) == 1) active_count=active_count+1
            enddo
            !$omp end do
            !$omp end parallel
        endif
        status = euleriso_glm_v77_ok
    end subroutine braids_euler_velocity_probe_v77

    subroutine braids_rolcor_flux_probe_v77(nface,euleriso_glm, &
            wave_enabled,roller_enabled,cyclic_solver,lstsci,forcing_present, &
            grmasu,grmasv,grfacu,grfacv,guu,gvv,tpu,tpv,kfu,kcu,kfv,kcv, &
            thick,dqx,dqy,active_count,workers,status)
        integer, intent(in) :: nface,lstsci
        logical, intent(in) :: euleriso_glm,wave_enabled,roller_enabled
        logical, intent(in) :: cyclic_solver,forcing_present
        real(real64), intent(in) :: grmasu(nface),grmasv(nface)
        real(real64), intent(in) :: grfacu(nface),grfacv(nface)
        real(real64), intent(in) :: guu(nface),gvv(nface),tpu(nface),tpv(nface)
        integer, intent(in) :: kfu(nface),kcu(nface),kfv(nface),kcv(nface)
        real(real64), intent(in) :: thick
        real(real64), intent(out) :: dqx(nface),dqy(nface)
        integer, intent(out) :: active_count,workers,status
        integer :: i
        logical :: route_ready,u_active,v_active

        dqx = 0.0_real64
        dqy = 0.0_real64
        active_count = 0
        workers = 1
        route_ready = euleriso_glm .and. wave_enabled .and. roller_enabled &
            .and. cyclic_solver .and. lstsci > 0
        if (route_ready .and. .not.forcing_present) then
            status = euleriso_glm_v77_missing_forcing
            return
        endif
        if (nface < 1 .or. .not.ieee_is_finite(thick) .or. &
                thick <= 0.0_real64 .or. any(.not.ieee_is_finite(grmasu)) .or. &
                any(.not.ieee_is_finite(grmasv)) .or. &
                any(.not.ieee_is_finite(grfacu)) .or. &
                any(.not.ieee_is_finite(grfacv)) .or. &
                any(.not.ieee_is_finite(guu)) .or. &
                any(.not.ieee_is_finite(gvv)) .or. &
                any(.not.ieee_is_finite(tpu)) .or. &
                any(.not.ieee_is_finite(tpv)) .or. any(guu <= 0.0_real64) .or. &
                any(gvv <= 0.0_real64) .or. any(tpu < 0.0_real64) .or. &
                any(tpv < 0.0_real64)) then
            status = euleriso_glm_v77_invalid
            return
        endif
        if (route_ready) then
            !$omp parallel shared(workers) private(u_active,v_active)
            !$omp single
            workers = omp_get_num_threads()
            !$omp end single
            !$omp do reduction(+:active_count)
            do i=1,nface
                u_active = kfu(i)==1 .and. kcu(i)<=2 .and. tpu(i)>0.5_real64
                v_active = kfv(i)==1 .and. kcv(i)<=2 .and. tpv(i)>0.5_real64
                if (u_active) dqx(i)=-(grmasu(i)+grfacu(i))*guu(i)*thick
                if (v_active) dqy(i)=-(grmasv(i)+grfacv(i))*gvv(i)*thick
                if (u_active .or. v_active) active_count=active_count+1
            enddo
            !$omp end do
            !$omp end parallel
        endif
        status = euleriso_glm_v77_ok
    end subroutine braids_rolcor_flux_probe_v77

    subroutine braids_initialize_euleriso_glm_state_v77(nfrac,ncell,values, &
            engine,current_only,wave_enabled,roller_enabled,forcing_present, &
            euleriso_glm,glmiso_euler,state,status)
        integer, intent(in) :: nfrac,ncell,engine
        real(real64), intent(in) :: values(nfrac,ncell)
        logical, intent(in) :: current_only,wave_enabled,roller_enabled
        logical, intent(in) :: forcing_present,euleriso_glm,glmiso_euler
        type(braids_euleriso_glm_state_v77), intent(out) :: state
        integer, intent(out) :: status
        logical :: euler_effective,glm_effective

        call braids_resolve_euleriso_glm_v77(engine,current_only,wave_enabled, &
            roller_enabled,forcing_present,euleriso_glm,glmiso_euler, &
            euler_effective,glm_effective,status)
        if (status /= euleriso_glm_v77_ok) return
        if (nfrac < 1 .or. ncell < 1 .or. &
                any(.not.ieee_is_finite(values)) .or. any(values<0.0_real64)) then
            status = euleriso_glm_v77_invalid
            return
        endif
        allocate(state%concentration(nfrac,ncell))
        state%concentration = values
        state%accepted_updates = 0
        state%identity_checks = 0
        state%euleriso_active_updates = 0
        state%glmiso_logged_updates = 0
        state%engine = engine
        state%current_only = current_only
        state%wave_enabled = wave_enabled
        state%roller_enabled = roller_enabled
        state%forcing_present = forcing_present
        state%euleriso_requested = euleriso_glm
        state%glmiso_requested = glmiso_euler
        state%semantic = euleriso_glm_v77_semantic
        state%generation = 0_int64
        state%lineage = ieor(matrix_bits_token(values), &
            execution_bits_token(state))
        state%valid = .true.
        state%integrity = state_bits_token(state)
        status = euleriso_glm_v77_ok
    end subroutine braids_initialize_euleriso_glm_state_v77

    subroutine braids_begin_euleriso_glm_trial_v77(parent,values,cancelled, &
            trial,status)
        type(braids_euleriso_glm_state_v77), intent(in) :: parent
        real(real64), intent(in) :: values(:,:)
        logical, intent(in) :: cancelled
        type(braids_euleriso_glm_trial_v77), intent(out) :: trial
        integer, intent(out) :: status
        integer :: i,j

        call validate_state(parent,status)
        if (status /= euleriso_glm_v77_ok) return
        if (cancelled) then
            status = euleriso_glm_v77_cancelled
            return
        endif
        if (parent%accepted_updates >= euleriso_glm_v77_max_updates) then
            status = euleriso_glm_v77_update_limit
            return
        endif
        if (any(shape(values) /= shape(parent%concentration)) .or. &
                any(.not.ieee_is_finite(values)) .or. any(values<0.0_real64)) then
            status = euleriso_glm_v77_parent_mismatch
            return
        endif
        if (.not.same_matrix_bits(values,parent%concentration)) then
            status = euleriso_glm_v77_parent_mismatch
            return
        endif
        allocate(trial%input_concentration(size(values,1),size(values,2)))
        allocate(trial%output_concentration(size(values,1),size(values,2)))
        allocate(trial%control_delta(size(values,1),size(values,2)))
        trial%input_concentration = values
        trial%control_delta = 0.0_real64
        trial%observed_parallel_workers = 1
        !$omp parallel shared(trial)
        !$omp single
        trial%observed_parallel_workers = omp_get_num_threads()
        !$omp end single
        !$omp do collapse(2)
        do j=1,size(values,2)
            do i=1,size(values,1)
                trial%output_concentration(i,j) = trial%input_concentration(i,j)
            enddo
        enddo
        !$omp end do
        !$omp end parallel
        if (.not.same_matrix_bits(trial%input_concentration, &
                trial%output_concentration)) then
            status = euleriso_glm_v77_invalid_trial
            return
        endif
        trial%candidate = parent
        trial%candidate%concentration = trial%output_concentration
        trial%candidate%accepted_updates = parent%accepted_updates + 1
        trial%candidate%identity_checks = parent%identity_checks + 1
        trial%candidate%euleriso_active_updates = parent%euleriso_active_updates
        trial%candidate%glmiso_logged_updates = parent%glmiso_logged_updates + 1
        trial%candidate%generation = parent%generation + 1_int64
        trial%candidate%valid = .true.
        trial%candidate%integrity = state_bits_token(trial%candidate)
        trial%parent_integrity = parent%integrity
        trial%parent_lineage = parent%lineage
        trial%parent_generation = parent%generation
        trial%identity_exact = .true.
        trial%consumed = .false.
        trial%valid = .true.
        trial%integrity = trial_bits_token(trial)
        status = euleriso_glm_v77_ok
    end subroutine braids_begin_euleriso_glm_trial_v77

    subroutine braids_commit_euleriso_glm_trial_v77(state,trial,status)
        type(braids_euleriso_glm_state_v77), intent(inout) :: state
        type(braids_euleriso_glm_trial_v77), intent(inout) :: trial
        integer, intent(out) :: status
        call validate_trial(state,trial,status)
        if (status /= euleriso_glm_v77_ok) return
        state = trial%candidate
        trial%consumed = .true.
        trial%valid = .false.
        status = euleriso_glm_v77_ok
    end subroutine braids_commit_euleriso_glm_trial_v77

    subroutine braids_rollback_euleriso_glm_trial_v77(state,trial,status)
        type(braids_euleriso_glm_state_v77), intent(inout) :: state
        type(braids_euleriso_glm_trial_v77), intent(inout) :: trial
        integer, intent(out) :: status
        call validate_trial(state,trial,status)
        if (status /= euleriso_glm_v77_ok) return
        trial%consumed = .true.
        trial%valid = .false.
        status = euleriso_glm_v77_ok
    end subroutine braids_rollback_euleriso_glm_trial_v77

    subroutine braids_cancel_euleriso_glm_trial_v77(state,trial,status)
        type(braids_euleriso_glm_state_v77), intent(inout) :: state
        type(braids_euleriso_glm_trial_v77), intent(inout) :: trial
        integer, intent(out) :: status
        call braids_rollback_euleriso_glm_trial_v77(state,trial,status)
    end subroutine braids_cancel_euleriso_glm_trial_v77

    subroutine braids_checkpoint_euleriso_glm_v77(state,checkpoint,status)
        type(braids_euleriso_glm_state_v77), intent(in) :: state
        type(braids_euleriso_glm_state_v77), intent(out) :: checkpoint
        integer, intent(out) :: status
        call validate_state(state,status)
        if (status /= euleriso_glm_v77_ok) return
        checkpoint = state
        status = euleriso_glm_v77_ok
    end subroutine braids_checkpoint_euleriso_glm_v77

    subroutine braids_restart_euleriso_glm_v77(checkpoint,state,status)
        type(braids_euleriso_glm_state_v77), intent(in) :: checkpoint
        type(braids_euleriso_glm_state_v77), intent(out) :: state
        integer, intent(out) :: status
        call validate_state(checkpoint,status)
        if (status /= euleriso_glm_v77_ok) return
        state = checkpoint
        call validate_state(state,status)
    end subroutine braids_restart_euleriso_glm_v77

    subroutine braids_state_snapshot_v77(state,concentration,accepted_updates, &
            identity_checks,euler_active,glm_logged,generation,status)
        type(braids_euleriso_glm_state_v77), intent(in) :: state
        real(real64), allocatable, intent(out) :: concentration(:,:)
        integer, intent(out) :: accepted_updates,identity_checks
        integer, intent(out) :: euler_active,glm_logged
        integer(int64), intent(out) :: generation
        integer, intent(out) :: status
        call validate_state(state,status)
        if (status /= euleriso_glm_v77_ok) return
        allocate(concentration(size(state%concentration,1), &
            size(state%concentration,2)))
        concentration = state%concentration
        accepted_updates = state%accepted_updates
        identity_checks = state%identity_checks
        euler_active = state%euleriso_active_updates
        glm_logged = state%glmiso_logged_updates
        generation = state%generation
    end subroutine braids_state_snapshot_v77

    subroutine braids_trial_snapshot_v77(parent,trial,control_delta,workers, &
            identity_exact,status)
        type(braids_euleriso_glm_state_v77), intent(in) :: parent
        type(braids_euleriso_glm_trial_v77), intent(in) :: trial
        real(real64), allocatable, intent(out) :: control_delta(:,:)
        integer, intent(out) :: workers
        logical, intent(out) :: identity_exact
        integer, intent(out) :: status
        call validate_trial(parent,trial,status)
        if (status /= euleriso_glm_v77_ok) return
        allocate(control_delta(size(trial%control_delta,1), &
            size(trial%control_delta,2)))
        control_delta = trial%control_delta
        workers = trial%observed_parallel_workers
        identity_exact = trial%identity_exact
    end subroutine braids_trial_snapshot_v77

    subroutine validate_state(state,status)
        type(braids_euleriso_glm_state_v77), intent(in) :: state
        integer, intent(out) :: status
        if (.not.state%valid .or. .not.allocated(state%concentration)) then
            status = euleriso_glm_v77_invalid_state
            return
        endif
        if (state%semantic /= euleriso_glm_v77_semantic .or. &
                state%engine /= euleriso_glm_v77_engine_classic .or. &
                .not.state%current_only .or. state%wave_enabled .or. &
                state%roller_enabled .or. state%forcing_present) then
            status = euleriso_glm_v77_invalid_state
            return
        endif
        if (any(.not.ieee_is_finite(state%concentration)) .or. &
                any(state%concentration<0.0_real64)) then
            status = euleriso_glm_v77_invalid_state
            return
        endif
        if (state%accepted_updates<0 .or. &
                state%accepted_updates>euleriso_glm_v77_max_updates .or. &
                state%identity_checks/=state%accepted_updates .or. &
                state%euleriso_active_updates/=0 .or. &
                state%glmiso_logged_updates/=state%accepted_updates .or. &
                state%generation/=int(state%accepted_updates,int64) .or. &
                state%integrity/=state_bits_token(state)) then
            status = euleriso_glm_v77_invalid_state
            return
        endif
        status = euleriso_glm_v77_ok
    end subroutine validate_state

    subroutine validate_trial(parent,trial,status)
        type(braids_euleriso_glm_state_v77), intent(in) :: parent
        type(braids_euleriso_glm_trial_v77), intent(in) :: trial
        integer, intent(out) :: status
        integer :: local_status
        integer(int64) :: plus_zero
        plus_zero = transfer(0.0_real64,plus_zero)
        call validate_state(parent,status)
        if (status /= euleriso_glm_v77_ok) return
        if (.not.trial%valid .or. trial%consumed .or. &
                .not.allocated(trial%input_concentration) .or. &
                .not.allocated(trial%output_concentration) .or. &
                .not.allocated(trial%control_delta)) then
            status = euleriso_glm_v77_invalid_trial
            return
        endif
        call validate_state(trial%candidate,local_status)
        if (local_status/=euleriso_glm_v77_ok .or. &
                trial%parent_integrity/=parent%integrity .or. &
                trial%parent_lineage/=parent%lineage .or. &
                trial%parent_generation/=parent%generation .or. &
                trial%candidate%lineage/=parent%lineage .or. &
                trial%candidate%generation/=parent%generation+1_int64 .or. &
                trial%candidate%accepted_updates/=parent%accepted_updates+1 .or. &
                trial%candidate%identity_checks/=parent%identity_checks+1 .or. &
                trial%candidate%euleriso_active_updates/= &
                    parent%euleriso_active_updates .or. &
                trial%candidate%glmiso_logged_updates/= &
                    parent%glmiso_logged_updates+1 .or. &
                .not.trial%identity_exact .or. &
                trial%observed_parallel_workers<1 .or. &
                .not.same_matrix_bits(trial%input_concentration, &
                    parent%concentration) .or. &
                .not.same_matrix_bits(trial%output_concentration, &
                    parent%concentration) .or. &
                .not.same_matrix_bits(trial%candidate%concentration, &
                    parent%concentration) .or. &
                any(transfer(trial%control_delta, [0_int64], &
                    size(trial%control_delta))/=plus_zero) .or. &
                trial%integrity/=trial_bits_token(trial)) then
            status = euleriso_glm_v77_invalid_trial
        else
            status = euleriso_glm_v77_ok
        endif
    end subroutine validate_trial

    pure logical function same_matrix_bits(left,right)
        real(real64), intent(in) :: left(:,:),right(:,:)
        integer :: i,j
        same_matrix_bits = .false.
        if (any(shape(left)/=shape(right))) return
        do j=1,size(left,2)
            do i=1,size(left,1)
                if (transfer(left(i,j),0_int64)/= &
                        transfer(right(i,j),0_int64)) return
            enddo
        enddo
        same_matrix_bits = .true.
    end function same_matrix_bits

    pure integer(int64) function matrix_bits_token(values) result(token)
        real(real64), intent(in) :: values(:,:)
        integer :: i,j
        integer(int64) :: bits,index_bits
        token = int(z'6A09E667F3BCC909',int64)
        do j=1,size(values,2)
            do i=1,size(values,1)
                bits = transfer(values(i,j),bits)
                index_bits = int(i+131*j,int64)
                token = ieor(ishftc(token,13),bits)
                token = ieor(token,ishftc(index_bits,7))
            enddo
        enddo
        token = ieor(token,int(size(values,1),int64))
        token = ieor(token,ishftc(int(size(values,2),int64),19))
    end function matrix_bits_token

    pure integer(int64) function execution_bits_token(state) result(token)
        type(braids_euleriso_glm_state_v77), intent(in) :: state
        token = int(z'BB67AE8584CAA73B',int64)
        token = ieor(token,int(state%engine,int64))
        if (state%current_only) token=ieor(token,int(z'1000000000000001',int64))
        if (state%wave_enabled) token=ieor(token,int(z'2000000000000002',int64))
        if (state%roller_enabled) token=ieor(token,int(z'3000000000000003',int64))
        if (state%forcing_present) token=ieor(token,int(z'4000000000000004',int64))
        if (state%euleriso_requested) token=ieor(token,int(z'5000000000000005',int64))
        if (state%glmiso_requested) token=ieor(token,int(z'6000000000000006',int64))
    end function execution_bits_token

    pure integer(int64) function state_bits_token(state) result(token)
        type(braids_euleriso_glm_state_v77), intent(in) :: state
        token = ieor(matrix_bits_token(state%concentration),state%lineage)
        token = ieor(token,execution_bits_token(state))
        token = ieor(token,ishftc(state%generation,11))
        token = ieor(token,ishftc(int(state%accepted_updates,int64),17))
        token = ieor(token,ishftc(int(state%identity_checks,int64),23))
        token = ieor(token,ishftc(int(state%euleriso_active_updates,int64),29))
        token = ieor(token,ishftc(int(state%glmiso_logged_updates,int64),37))
    end function state_bits_token

    pure integer(int64) function trial_bits_token(trial) result(token)
        type(braids_euleriso_glm_trial_v77), intent(in) :: trial
        token = ieor(trial%parent_integrity,trial%candidate%integrity)
        token = ieor(token,trial%parent_lineage)
        token = ieor(token,ishftc(trial%parent_generation,9))
        token = ieor(token,matrix_bits_token(trial%input_concentration))
        token = ieor(token,ishftc(matrix_bits_token( &
            trial%output_concentration),17))
        token = ieor(token,ishftc(matrix_bits_token(trial%control_delta),31))
        token = ieor(token,int(trial%observed_parallel_workers,int64))
        if (trial%identity_exact) token=ieor(token,int(z'3C6EF372FE94F82B',int64))
    end function trial_bits_token

end module braids_euleriso_glm_live_v77_module
