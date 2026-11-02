! Opt-in accepted-step boundary mass flux observations for Cartesian level 1.
! The numerical method and its states are never modified by this observer.
module braids_flux_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use amr_module, only: mxnest, mcapa, numgrids, listStart, listOfGrids, node, &
        ndilo, ndihi, ndjlo, ndjhi, iregsz, jregsz, xperdom, yperdom, spheredom
    use geoclaw_module, only: coordinate_system, use_fwave_positivity_limiter
    implicit none
    private
    public :: braids_flux_begin, braids_flux_capture, braids_flux_patch_volume, braids_flux_commit

    public :: braids_flux_patch_exchange

    type :: face_sample
        integer :: side = 0, global_index = -1
        real(kind=8) :: length = 0.d0, ghost_rate = 0.d0, interior_rate = 0.d0, accepted_rate = 0.d0
        real(kind=8) :: ghost_h=0.d0,interior_h=0.d0,bed=0.d0,sediment_rate=0.d0
    end type face_sample

    type :: patch_sample
        integer :: id = 0, ilo = 0, ihi = -1, jlo = 0, jhi = -1, phase = 0
        real(kind=8) :: exchange(3)=0.d0,sediment_in=0.d0,sediment_out=0.d0
        logical :: exchange_seen=.false.
        real(kind=8) :: before = 0.d0, hyperbolic = 0.d0, final = 0.d0
        type(face_sample), allocatable :: faces(:)
    end type patch_sample

    type :: history_sample
        real(kind=8) :: time=0.d0,dt=0.d0,volume=0.d0,face_volume=0.d0
        real(kind=8) :: absolute_face_volume=0.d0
        integer :: faces=0
    end type history_sample
    character(len=*), parameter :: face_header = &
        'step,t_start,dt,level,patch_id,side,global_index,face_length_m,'// &
        'ghost_state_inward_m3_s,interior_state_inward_m3_s,accepted_inward_m3_s,accepted_volume_m3,'// &
        'ghost_h_m,interior_h_m,bed_m'
    character(len=*), parameter :: step_header = &
        'step,t_start,dt,patch_count,volume_before_m3,volume_hyperbolic_m3,'// &
        'volume_final_m3,west_inward_m3_s,east_inward_m3_s,south_inward_m3_s,north_inward_m3_s,'// &
        'accepted_volume_m3,hyperbolic_balance_residual_m3,total_balance_residual_m3'
    character(len=*), parameter :: exchange_header = &
        'step,t_start,dt,sediment_input_kg,sediment_export_kg,deposition_kg,erosion_kg,dry_transfer_kg'
    logical, save :: face_sediment=.false.
    logical, save :: observe_exchange=.false.
    integer, save :: exchange_unit=0
    logical, save :: have_committed=.false.
    real(kind=8), save :: last_committed_time=0.d0
    logical, save :: configured = .false., enabled = .false., write_faces = .false.
    logical, save :: active = .false., files_open = .false.
    integer, save :: step_id = 0, current_level = 0, face_unit = 0, step_unit = 0
    integer, save :: side_count(4) = 0
    real(kind=8), save :: current_time = 0.d0, current_dt = 0.d0
    type(patch_sample), allocatable, save :: patches(:)
    integer, allocatable, save :: owner(:,:), owner_face(:,:)

contains
    function selected_face_header() result(header)
        character(len=:), allocatable :: header
        header=face_header
        if (face_sediment) header=header//',accepted_sediment_inward_kg_s,accepted_sediment_kg'
    end function selected_face_header


    subroutine fail(message)
        character(len=*), intent(in) :: message
        write(*,'(a)') 'BRAIDS flux observer: '//trim(message)
        error stop 1
    end subroutine fail

    subroutine configure()
        integer :: unit, ios, flag
        character(len=512) :: line
        logical :: exists
        if (configured) return
        configured = .true.
        inquire(file='braids_flux.data', exist=exists)
        if (.not.exists) return
        open(newunit=unit, file='braids_flux.data', status='old', action='read', iostat=ios)
        if (ios /= 0) call fail('cannot open braids_flux.data')
        do
            read(unit,'(a)',iostat=ios) line
            if (ios /= 0) call fail('braids_flux.data requires mode 0, 1, 2, or 3')
            line = adjustl(line)
            if (len_trim(line) == 0) cycle
            if (line(1:1) == '#') cycle
            read(line,*,iostat=ios) flag
            if (ios /= 0) call fail('braids_flux.data mode must be 0, 1, 2, or 3')
            exit
        enddo
        close(unit)
        if (flag < 0 .or. flag > 3) call fail('braids_flux.data mode must be 0, 1, 2, or 3')
        enabled = flag > 0
        write_faces = flag == 1.or.flag == 3
        face_sediment = flag == 3
        inquire(file='braids_exchange_observer.data',exist=observe_exchange)
        if (face_sediment.and..not.observe_exchange) call fail('mode 3 requires sediment exchange observation')
    end subroutine configure

    subroutine open_outputs(restarting,time)
        logical, intent(in) :: restarting
        real(kind=8), intent(in) :: time
        integer :: ios
        if (files_open) return
        if (restarting) then
            call restore_prefix(time)
            open(newunit=face_unit,file='braids_flux_faces.csv',status='old',position='append',action='write',iostat=ios)
            if (ios /= 0) call fail('cannot append restored face ledger')
            open(newunit=step_unit,file='braids_flux_steps.csv',status='old',position='append',action='write',iostat=ios)
            if (ios /= 0) call fail('cannot append restored step ledger')
        else
            open(newunit=face_unit, file='braids_flux_faces.csv', status='replace', action='write', iostat=ios)
            if (ios /= 0) call fail('cannot open braids_flux_faces.csv')
            open(newunit=step_unit, file='braids_flux_steps.csv', status='replace', action='write', iostat=ios)
            if (ios /= 0) call fail('cannot open braids_flux_steps.csv')
            write(face_unit,'(a)') selected_face_header()
            write(step_unit,'(a)') step_header
        endif
        if (observe_exchange) then
            call open_exchange_output(restarting,time)
        endif
        files_open = .true.
    end subroutine open_outputs

    subroutine open_exchange_output(restarting,time)
        logical, intent(in) :: restarting
        real(kind=8), intent(in) :: time
        integer :: input,output,ios,sid,count
        real(kind=8) :: t,dt,values(5),last_end
        character(len=2048) :: line
        if (restarting) then
            call recover_replacement('braids_exchange_steps.csv')
            open(newunit=input,file='braids_exchange_steps.csv',status='old',action='read',iostat=ios)
            if (ios/=0) call fail('restart requires the sediment exchange ledger')
            open(newunit=output,file='braids_exchange_steps.csv.restart.tmp',status='replace',action='write')
            read(input,'(a)',iostat=ios) line
            if (ios/=0.or.trim(line)/=exchange_header) call fail('exchange ledger schema mismatch')
            write(output,'(a)') exchange_header
            count=0; last_end=0.d0
            do
                read(input,'(a)',iostat=ios) line
                if (ios<0) exit
                if (ios/=0) call fail('cannot read exchange ledger')
                read(line,*,iostat=ios) sid,t,dt,values
                if (ios/=0) call fail('invalid exchange ledger row')
                if (.not.all(ieee_is_finite([t,dt,values])).or.dt<=0.d0) &
                    call fail('nonfinite or invalid exchange ledger values')
                if (t+dt>time.and..not.nearly_equal(t+dt,time)) exit
                if (sid/=count+1.or..not.nearly_equal(t,last_end)) call fail('exchange ledger time gap')
                count=count+1; last_end=t+dt
                write(output,'(a)') trim(line)
                if (nearly_equal(last_end,time)) exit
            enddo
            close(input); close(output)
            if (count/=step_id.or..not.nearly_equal(last_end,time)) &
                call fail('exchange ledger does not match restored water ledger')
            call replace_file('braids_exchange_steps.csv.restart.tmp','braids_exchange_steps.csv')
            open(newunit=exchange_unit,file='braids_exchange_steps.csv',status='old',position='append',action='write')
        else
            open(newunit=exchange_unit,file='braids_exchange_steps.csv',status='replace',action='write')
            write(exchange_unit,'(a)') exchange_header
        endif
    end subroutine open_exchange_output

    logical function nearly_equal(a,b)
        real(kind=8), intent(in) :: a,b
        nearly_equal = abs(a-b) <= 128.d0*epsilon(1.d0)*max(1.d0,abs(a),abs(b))
    end function nearly_equal

    subroutine delete_file(path,ios)
        character(len=*), intent(in) :: path
        integer, intent(out) :: ios
        integer :: unit
        logical :: exists
        inquire(file=path,exist=exists)
        ios=0
        if (.not.exists) return
        open(newunit=unit,file=path,status='old',action='readwrite',iostat=ios)
        if (ios==0) close(unit,status='delete',iostat=ios)
    end subroutine delete_file

    subroutine recover_replacement(destination)
        character(len=*), intent(in) :: destination
        character(len=:), allocatable :: backup
        integer :: ios
        logical :: destination_exists,backup_exists
        backup=trim(destination)//'.replace.backup'
        inquire(file=destination,exist=destination_exists)
        inquire(file=backup,exist=backup_exists)
        if (.not.backup_exists) return
        if (destination_exists) then
            call delete_file(backup,ios)
            if (ios/=0) call fail('cannot remove stale replacement backup for '//destination)
        else
            call rename(backup,destination,ios)
            if (ios/=0) call fail('cannot recover replacement backup for '//destination)
        endif
    end subroutine recover_replacement

    subroutine replace_file(temporary, destination)
        character(len=*), intent(in) :: temporary, destination
        character(len=:), allocatable :: backup
        integer :: ios,rollback_ios
        logical :: destination_exists,backup_exists
        ! The Fortran rename intrinsic does not replace an existing file on
        ! Windows. Use a same-directory transaction instead. The backup also
        ! makes an interruption recoverable: a later call restores it before
        ! attempting the replacement again. All ledger units are closed before
        ! this routine is entered.
        backup=trim(destination)//'.replace.backup'
        call recover_replacement(destination)
        inquire(file=destination,exist=destination_exists)
        if (destination_exists) then
            call rename(destination,backup,ios)
            if (ios/=0) call fail('cannot preserve previous '//destination)
        endif
        call rename(temporary,destination,ios)
        if (ios/=0) then
            inquire(file=backup,exist=backup_exists)
            if (backup_exists) call rename(backup,destination,rollback_ios)
            call fail('cannot replace '//destination)
        endif
        call delete_file(backup,ios)
        if (ios/=0) call fail('cannot remove replacement backup for '//destination)
    end subroutine replace_file

    subroutine restore_prefix(checkpoint_time)
        real(kind=8), intent(in) :: checkpoint_time
        type(history_sample), allocatable :: history(:), grown(:)
        logical, allocatable :: seen(:,:)
        integer :: input_steps,input_faces,temp_steps,temp_faces,ios,count,capacity,sid,patch_count
        integer :: previous_face_step,side,index,level,patch,expected_faces,k,retained_faces,meta_unit
        real(kind=8) :: time,dt,values(10),face_values(10),end_time,origin,last_end
        character(len=2048) :: line
        logical :: exists_steps,exists_faces,have_origin
        call recover_replacement('braids_flux_steps.csv')
        if (write_faces) call recover_replacement('braids_flux_faces.csv')
        inquire(file='braids_flux_steps.csv',exist=exists_steps)
        inquire(file='braids_flux_faces.csv',exist=exists_faces)
        if (.not.exists_steps .or. (write_faces.and..not.exists_faces)) &
            call fail('restart requires both accepted ledger files covering the checkpoint')
        open(newunit=input_steps,file='braids_flux_steps.csv',status='old',action='read',iostat=ios)
        if (ios /= 0) call fail('cannot read restart step ledger')
        if(write_faces) then
            open(newunit=input_faces,file='braids_flux_faces.csv',status='old',action='read',iostat=ios)
            if (ios /= 0) call fail('cannot read restart face ledger')
        endif
        open(newunit=temp_steps,file='braids_flux_steps.csv.restart.tmp',status='replace',action='write',iostat=ios)
        if (ios /= 0) call fail('cannot create temporary restart step ledger')
        open(newunit=temp_faces,file='braids_flux_faces.csv.restart.tmp',status='replace',action='write',iostat=ios)
        if (ios /= 0) call fail('cannot create temporary restart face ledger')
        read(input_steps,'(a)',iostat=ios) line
        if (ios /= 0 .or. trim(line) /= step_header) call fail('restart step ledger schema mismatch')
        write(temp_steps,'(a)') trim(line)
        if(write_faces) then
            read(input_faces,'(a)',iostat=ios) line
            if (ios /= 0 .or. trim(line) /= selected_face_header()) call fail('restart face ledger schema mismatch')
        endif
        write(temp_faces,'(a)') selected_face_header()
        capacity=64
        allocate(history(capacity))
        count=0
        have_origin=.false.
        origin=checkpoint_time
        last_end=checkpoint_time
        do
            read(input_steps,'(a)',iostat=ios) line
            if (ios < 0) exit
            if (ios /= 0) call fail('cannot read restart step row')
            read(line,*,iostat=ios) sid,time,dt,patch_count,values
            if (ios /= 0) call fail('invalid restart step row')
            if (.not.ieee_is_finite(time) .or. .not.ieee_is_finite(dt) .or. .not.all(ieee_is_finite(values))) &
                call fail('nonfinite restart step row')
            if (dt <= 0.d0 .or. patch_count < 1) call fail('invalid restart step interval or patch count')
            if (.not.write_faces) then
                if(abs(values(8)-dt*sum(values(4:7)))>128.d0*epsilon(1.d0)* &
                    max(1.d0,abs(values(8)),dt*sum(abs(values(4:7))))) &
                    call fail('restart aggregate boundary volumes do not match step total')
            endif
            if (.not.have_origin) then
                origin=time
                last_end=time
                have_origin=.true.
            endif
            if (sid /= count+1 .or. .not.nearly_equal(time,last_end)) &
                call fail('restart step prefix is duplicated, gapped or unordered')
            end_time=time+dt
            if (end_time <= checkpoint_time .or. nearly_equal(end_time,checkpoint_time)) then
                count=count+1
                if (count > capacity) then
                    allocate(grown(capacity*2))
                    grown(1:capacity)=history
                    call move_alloc(grown,history)
                    capacity=capacity*2
                endif
                history(count)%time=time
                history(count)%dt=dt
                history(count)%volume=values(8)
                last_end=end_time
                write(temp_steps,'(a)') trim(line)
                if (nearly_equal(last_end,checkpoint_time)) exit
            else
                if (time < checkpoint_time .and. .not.nearly_equal(time,checkpoint_time)) &
                    call fail('checkpoint cuts an accepted ledger interval')
                exit
            endif
        enddo
        if (.not.have_origin .or. .not.nearly_equal(last_end,checkpoint_time)) &
            call fail('restart ledger prefix does not reach the checkpoint')
        if (origin > checkpoint_time .and. .not.nearly_equal(origin,checkpoint_time)) &
            call fail('restart ledger starts after the checkpoint')
        retained_faces=0
        ! Compact mode has a header-only face ledger. Its accepted aggregate
        ! intervals are restored with the same step/order/checkpoint checks;
        ! detailed mode additionally validates every physical face below.
        if (write_faces) then
        expected_faces=sum(side_count)
        allocate(seen(4,0:size(owner,2)-1))
        seen=.false.
        previous_face_step=0
        retained_faces=0
        do
            read(input_faces,'(a)',iostat=ios) line
            if (ios < 0) exit
            if (ios /= 0) call fail('cannot read restart face row')
            read(line,*,iostat=ios) sid
            if (ios /= 0) call fail('invalid restart face step ID')
            if (sid > count) exit ! Future rows are discarded, never re-integrated.
            face_values=0.d0
            read(line,*,iostat=ios) sid,time,dt,level,patch,side,index,face_values(:merge(10,8,face_sediment))
            if (ios /= 0) call fail('invalid retained restart face row')
            if (sid < 1 .or. level /= 1 .or. patch < 1) call fail('invalid retained restart face identity')
            if (side < 1 .or. side > 4) call fail('invalid retained restart physical side')
            if (index < 0 .or. index >= side_count(side)) call fail('retained face does not match the current domain')
            if (sid /= previous_face_step) then
                if (sid /= previous_face_step+1) call fail('missing or unordered restart face step')
                previous_face_step=sid
                seen=.false.
            endif
            if (seen(side,index)) call fail('duplicate physical face in restart prefix')
            seen(side,index)=.true.
            if (.not.ieee_is_finite(time) .or. .not.ieee_is_finite(dt) .or. &
                .not.all(ieee_is_finite(face_values))) call fail('nonfinite retained restart face row')
            if (.not.nearly_equal(time,history(sid)%time) .or. .not.nearly_equal(dt,history(sid)%dt)) &
                call fail('restart face and step intervals disagree')
            if (face_values(1) <= 0.d0 .or. .not.nearly_equal(face_values(5),face_values(4)*dt)) &
                call fail('invalid retained face length or integrated volume')
            if (face_sediment.and..not.nearly_equal(face_values(10),face_values(9)*dt)) &
                call fail('invalid retained integrated sediment mass')
            history(sid)%faces=history(sid)%faces+1
            history(sid)%face_volume=history(sid)%face_volume+face_values(5)
            history(sid)%absolute_face_volume=history(sid)%absolute_face_volume+abs(face_values(5))
            retained_faces=retained_faces+1
            write(temp_faces,'(a)') trim(line)
        enddo
        do k=1,count
            if (history(k)%faces /= expected_faces) call fail('restart prefix is missing physical faces')
            if (abs(history(k)%face_volume-history(k)%volume) > 128.d0*epsilon(1.d0)* &
                max(1.d0,abs(history(k)%volume),history(k)%absolute_face_volume)) &
                call fail('restart face volumes do not match the accepted step total')
        enddo
        endif
        close(input_steps)
        if(write_faces) close(input_faces)
        close(temp_steps)
        close(temp_faces)
        ! If interrupted between these replacements, repeating this operation
        ! at the same checkpoint is idempotent: neither file is appended until
        ! both validated prefixes have replaced their originals.
        call replace_file('braids_flux_steps.csv.restart.tmp','braids_flux_steps.csv')
        call replace_file('braids_flux_faces.csv.restart.tmp','braids_flux_faces.csv')
        open(newunit=meta_unit,file='braids_flux_restart.csv.tmp',status='replace',action='write',iostat=ios)
        if (ios /= 0) call fail('cannot write restart ledger metadata')
        write(meta_unit,'(a)') 'checkpoint_time,retained_steps,retained_faces,next_step'
        write(meta_unit,'(es25.17e3,3(",",i0))') checkpoint_time,count,retained_faces,count+1
        close(meta_unit)
        call replace_file('braids_flux_restart.csv.tmp','braids_flux_restart.csv')
        step_id=count
        last_committed_time=checkpoint_time
        have_committed=.true.
    end subroutine restore_prefix


    subroutine braids_flux_begin(level, time, dt, restarting)
        integer, intent(in) :: level
        real(kind=8), intent(in) :: time, dt
        logical, intent(in) :: restarting
        integer :: k, j, id, previous, nx, ny, count, side, index, face, max_count
        integer(kind=8) :: cell_count
        integer, allocatable :: ids(:)
        call configure()
        if (.not.enabled) return
        if (active) call fail('begin called before the previous accepted step was committed')
        if (mxnest /= 1 .or. level /= 1) call fail('only fixed AMR level 1 is supported')
        if (coordinate_system /= 1 .or. mcapa /= 0 .or. spheredom) &
            call fail('only Cartesian metre coordinates without a capacity array are supported')
        if (.not.ieee_is_finite(time) .or. .not.ieee_is_finite(dt)) call fail('nonfinite accepted time or dt')
        if (dt <= 0.d0 .or. time + dt <= time) call fail('accepted dt must advance time')
        if (numgrids(level) < 1 .or. iregsz(level) < 1 .or. jregsz(level) < 1) &
            call fail('empty or invalid accepted grid hierarchy')
        if (allocated(patches)) deallocate(patches)
        if (allocated(owner)) deallocate(owner, owner_face)
        allocate(patches(numgrids(level)), ids(numgrids(level)))
        ids = listOfGrids(listStart(level):listStart(level)+numgrids(level)-1)
        ! Canonical patch order makes volume reductions independent of OpenMP scheduling.
        do k=2,size(ids)
            id = ids(k)
            j = k-1
            do while (j >= 1)
                if (ids(j) <= id) exit
                ids(j+1) = ids(j)
                j = j-1
            enddo
            ids(j+1) = id
        enddo
        side_count = [jregsz(level), jregsz(level), iregsz(level), iregsz(level)]
        if (xperdom) side_count(1:2) = 0
        if (yperdom) side_count(3:4) = 0
        max_count = max(1, maxval(side_count))
        allocate(owner(4,0:max_count-1), owner_face(4,0:max_count-1))
        owner = 0
        owner_face = 0
        previous = -1
        cell_count = 0
        do k=1,size(patches)
            id = ids(k)
            if (id <= 0 .or. id == previous) call fail('invalid or repeated patch ID in accepted hierarchy')
            previous = id
            patches(k)%id = id
            patches(k)%ilo = node(ndilo,id)
            patches(k)%ihi = node(ndihi,id)
            patches(k)%jlo = node(ndjlo,id)
            patches(k)%jhi = node(ndjhi,id)
            nx = patches(k)%ihi - patches(k)%ilo + 1
            ny = patches(k)%jhi - patches(k)%jlo + 1
            if (nx < 1 .or. ny < 1 .or. patches(k)%ilo < 0 .or. patches(k)%jlo < 0 .or. &
                patches(k)%ihi >= iregsz(level) .or. patches(k)%jhi >= jregsz(level)) &
                call fail('patch extent is outside the fixed physical domain')
            cell_count = cell_count + int(nx,kind=8)*int(ny,kind=8)
            count = 0
            if (.not.xperdom .and. patches(k)%ilo == 0) count = count + ny
            if (.not.xperdom .and. patches(k)%ihi == iregsz(level)-1) count = count + ny
            if (.not.yperdom .and. patches(k)%jlo == 0) count = count + nx
            if (.not.yperdom .and. patches(k)%jhi == jregsz(level)-1) count = count + nx
            allocate(patches(k)%faces(count))
            face = 0
            do side=1,4
                if (side_count(side) == 0) cycle
                if (side == 1 .and. patches(k)%ilo /= 0) cycle
                if (side == 2 .and. patches(k)%ihi /= iregsz(level)-1) cycle
                if (side == 3 .and. patches(k)%jlo /= 0) cycle
                if (side == 4 .and. patches(k)%jhi /= jregsz(level)-1) cycle
                if (side <= 2) then
                    do index=patches(k)%jlo,patches(k)%jhi
                        face = face + 1
                        call claim_face(k,face,side,index)
                    enddo
                else
                    do index=patches(k)%ilo,patches(k)%ihi
                        face = face + 1
                        call claim_face(k,face,side,index)
                    enddo
                endif
            enddo
        enddo
        if (cell_count /= int(iregsz(level),kind=8)*int(jregsz(level),kind=8)) &
            call fail('accepted patches do not cover the fixed domain exactly once')
        do side=1,4
            if (any(owner(side,0:side_count(side)-1) == 0)) call fail('unowned physical boundary face')
        enddo
        deallocate(ids)
        call open_outputs(restarting,time)
        if (have_committed) then
            if (.not.nearly_equal(time,last_committed_time)) call fail('accepted step leaves a ledger time gap or overlap')
        endif
        current_level = level
        current_time = time
        current_dt = dt
        step_id = step_id + 1
        active = .true.
    end subroutine braids_flux_begin

    subroutine claim_face(patch, face, side, index)
        integer, intent(in) :: patch, face, side, index
        if (owner(side,index) /= 0) call fail('a physical face has multiple patch owners')
        owner(side,index) = patch
        owner_face(side,index) = face
        patches(patch)%faces(face)%side = side
        patches(patch)%faces(face)%global_index = index
    end subroutine claim_face

    integer function patch_slot(id) result(slot)
        integer, intent(in) :: id
        integer :: k
        slot = 0
        do k=1,size(patches)
            if (patches(k)%id == id) then
                slot = k
                return
            endif
        enddo
        call fail('physical step references an unregistered patch')
    end function patch_slot

    subroutine braids_flux_capture(id, nvar, mitot, mjtot, mbc, q, fm, fp, gm, gp, dx, dy, time, dt, maux, aux)
        integer, intent(in) :: id, nvar, mitot, mjtot, mbc, maux
        real(kind=8), intent(in) :: q(nvar,mitot,mjtot), fm(nvar,mitot,mjtot), fp(nvar,mitot,mjtot)
        real(kind=8), intent(in) :: gm(nvar,mitot,mjtot), gp(nvar,mitot,mjtot), dx, dy, time, dt
        real(kind=8), intent(in) :: aux(maux,mitot,mjtot)
        integer :: slot, face, side, index, i, j, il, ir, jb, jt
        real(kind=8) :: raw, inside, ghost, sign, length, tolerance,sediment_rate
        if (.not.active) return
        if (observe_exchange.and.nvar<5) call fail('exchange observer requires sediment states')
        if (nvar < 3 .or. mbc < 1 .or. maux < 1) call fail('observer requires SWE depth, two momenta and ghost cells')
        if (.not.ieee_is_finite(dx) .or. .not.ieee_is_finite(dy) .or. dx <= 0.d0 .or. dy <= 0.d0) &
            call fail('invalid Cartesian cell dimensions')
        tolerance = 16.d0*epsilon(1.d0)*max(1.d0,abs(current_time),abs(current_dt))
        if (abs(time-current_time) > tolerance .or. abs(dt-current_dt) > tolerance) &
            call fail('patch time does not match the armed accepted step')
        slot = patch_slot(id)
        if (patches(slot)%phase /= 0) call fail('a patch flux was captured twice in the same accepted step')
        il = mbc+1
        ir = mitot-mbc
        jb = mbc+1
        jt = mjtot-mbc
        if (ir-il /= patches(slot)%ihi-patches(slot)%ilo .or. &
            jt-jb /= patches(slot)%jhi-patches(slot)%jlo) call fail('patch array dimensions do not match ownership')
        patches(slot)%before = sum(q(1,il:ir,jb:jt))*dx*dy
        if (.not.ieee_is_finite(patches(slot)%before)) call fail('nonfinite pre-update water volume')
        do face=1,size(patches(slot)%faces)
            side = patches(slot)%faces(face)%side
            index = patches(slot)%faces(face)%global_index
            select case(side)
            case(1)
                i = il
                j = jb + index - patches(slot)%jlo
                raw = fp(1,i,j)
                inside = q(2,i,j)
                ghost = q(2,i-1,j)
                patches(slot)%faces(face)%ghost_h=q(1,i-1,j)
                sign = 1.d0
                length = dy
            case(2)
                i = ir
                j = jb + index - patches(slot)%jlo
                raw = fm(1,i+1,j)
                inside = q(2,i,j)
                ghost = q(2,i+1,j)
                patches(slot)%faces(face)%ghost_h=q(1,i+1,j)
                sign = -1.d0
                length = dy
            case(3)
                i = il + index - patches(slot)%ilo
                j = jb
                raw = gp(1,i,j)
                inside = q(3,i,j)
                ghost = q(3,i,j-1)
                patches(slot)%faces(face)%ghost_h=q(1,i,j-1)
                sign = 1.d0
                length = dx
            case(4)
                i = il + index - patches(slot)%ilo
                j = jt
                raw = gm(1,i,j+1)
                inside = q(3,i,j)
                ghost = q(3,i,j+1)
                patches(slot)%faces(face)%ghost_h=q(1,i,j+1)
                sign = -1.d0
                length = dx
            end select
            if (observe_exchange) then
                select case(side)
                case(1)
                    sediment_rate=fp(4,i,j)*length
                case(2)
                    sediment_rate=-fm(4,i+1,j)*length
                case(3)
                    sediment_rate=gp(4,i,j)*length
                case(4)
                    sediment_rate=-gm(4,i,j+1)*length
                end select
                if (.not.ieee_is_finite(sediment_rate)) call fail('nonfinite sediment flux')
                patches(slot)%faces(face)%sediment_rate=sediment_rate
                patches(slot)%sediment_in=patches(slot)%sediment_in+max(sediment_rate,0.d0)*dt
                patches(slot)%sediment_out=patches(slot)%sediment_out+max(-sediment_rate,0.d0)*dt
            endif
            if (.not.use_fwave_positivity_limiter) raw = raw + inside
            patches(slot)%faces(face)%interior_h=q(1,i,j)
            patches(slot)%faces(face)%bed=aux(1,i,j)
            patches(slot)%faces(face)%length = length
            patches(slot)%faces(face)%ghost_rate = sign*ghost*length
            patches(slot)%faces(face)%interior_rate = sign*inside*length
            patches(slot)%faces(face)%accepted_rate = sign*raw*length
            if (.not.ieee_is_finite(raw) .or. .not.ieee_is_finite(ghost) .or. .not.ieee_is_finite(inside)) &
                call fail('nonfinite boundary mass flux')
            associate(sample => patches(slot)%faces(face))
                if (.not.all(ieee_is_finite([sample%accepted_rate,sample%ghost_rate,sample%interior_rate, &
                    sample%ghost_h,sample%interior_h,sample%bed]))) call fail('nonfinite boundary observation')
            end associate
        enddo
        patches(slot)%phase = 1
    end subroutine braids_flux_capture

    subroutine braids_flux_patch_exchange(id,exchange,dx,dy)
        integer, intent(in) :: id
        real(kind=8), intent(in) :: exchange(3),dx,dy
        integer :: slot
        if (.not.active.or..not.observe_exchange) return
        slot=patch_slot(id)
        if (patches(slot)%exchange_seen) call fail('duplicate exchange observation')
        if (.not.all(ieee_is_finite(exchange)).or.any(exchange<0.d0)) call fail('invalid exchange observation')
        patches(slot)%exchange=exchange*dx*dy
        patches(slot)%exchange_seen=.true.
    end subroutine braids_flux_patch_exchange

    subroutine braids_flux_patch_volume(id, phase, nvar, mitot, mjtot, mbc, q, dx, dy)
        integer, intent(in) :: id, phase, nvar, mitot, mjtot, mbc
        real(kind=8), intent(in) :: q(nvar,mitot,mjtot), dx, dy
        integer :: slot
        real(kind=8) :: volume
        if (.not.active) return
        slot = patch_slot(id)
        if (phase /= 1 .and. phase /= 2) call fail('invalid observation phase')
        if (patches(slot)%phase /= phase) call fail('missing or duplicate patch-volume observation')
        volume = sum(q(1,mbc+1:mitot-mbc,mbc+1:mjtot-mbc))*dx*dy
        if (.not.ieee_is_finite(volume)) call fail('nonfinite observed water volume')
        if (phase == 1) patches(slot)%hyperbolic = volume
        if (phase == 2) patches(slot)%final = volume
        patches(slot)%phase = phase+1
    end subroutine braids_flux_patch_volume

    subroutine braids_flux_commit()
        integer :: side, index, slot, face, k
        real(kind=8) :: before, hyperbolic, final, rates(4), volume, hyper_residual, final_residual,exchange(3)
        if (.not.enabled) return
        if (.not.active) call fail('commit called without an accepted-step context')
        before = 0.d0
        hyperbolic = 0.d0
        final = 0.d0
        rates = 0.d0
        exchange=0.d0
        do k=1,size(patches)
            if (patches(k)%phase /= 3) call fail('cannot commit an incomplete accepted patch step')
            exchange=exchange+patches(k)%exchange
            before = before + patches(k)%before
            hyperbolic = hyperbolic + patches(k)%hyperbolic
            final = final + patches(k)%final
        enddo
        ! Stable global face order, independent of patch partition and worker order.
        do side=1,4
            do index=0,side_count(side)-1
                slot = owner(side,index)
                face = owner_face(side,index)
                associate(sample => patches(slot)%faces(face))
                    rates(side) = rates(side) + sample%accepted_rate
                    if (write_faces) then
                        if (face_sediment) then
                            write(face_unit,'(i0,2(",",es25.17e3),4(",",i0),10(",",es25.17e3))') &
                                step_id,current_time,current_dt,current_level,patches(slot)%id,side,index, &
                                sample%length,sample%ghost_rate,sample%interior_rate,sample%accepted_rate, &
                                current_dt*sample%accepted_rate,sample%ghost_h,sample%interior_h,sample%bed, &
                                sample%sediment_rate,current_dt*sample%sediment_rate
                        else
                        write(face_unit,'(i0,2(",",es25.17e3),4(",",i0),8(",",es25.17e3))') &
                            step_id,current_time,current_dt,current_level,patches(slot)%id,side,index, &
                            sample%length,sample%ghost_rate,sample%interior_rate,sample%accepted_rate, &
                            current_dt*sample%accepted_rate,sample%ghost_h,sample%interior_h,sample%bed
                        endif
                    endif
                end associate
            enddo
        enddo
        volume = current_dt*sum(rates)
        hyper_residual = hyperbolic-before-volume
        final_residual = final-before-volume
        write(step_unit,'(i0,2(",",es25.17e3),",",i0,10(",",es25.17e3))') &
            step_id,current_time,current_dt,size(patches),before,hyperbolic,final,rates,volume,hyper_residual,final_residual
        if (observe_exchange) then
            if (.not.all(patches%exchange_seen)) call fail('missing exchange observation')
            write(exchange_unit,'(i0,7(",",es25.17e3))') step_id,current_time,current_dt, &
                sum(patches%sediment_in),sum(patches%sediment_out), &
                exchange
            flush(exchange_unit)
        endif
        if (write_faces) flush(face_unit)
        flush(step_unit)
        last_committed_time=current_time+current_dt
        have_committed=.true.
        active = .false.
    end subroutine braids_flux_commit
end module braids_flux_module
