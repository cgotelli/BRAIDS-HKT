! SPDX-License-Identifier: AGPL-3.0-or-later
!
! Additive Q3-H7/v71 ownership-changing wet/dry + AMR seam.
!
! The accepted v67 direct-vector state remains the only bedload/Exner owner.
! This module exports that committed state, transfers bed mass by exact overlap
! area, and restores it under a separately authenticated target v67 topology.
! Boundary ledgers and accepted clock are copied exactly.  Topology-local face
! registers are cleared rather than transferred.  No constitutive formula,
! hydraulic step, recursive AMR clock, suspended load or dry-bank erosion is
! implemented here.
module braids_direct_vector_wetdry_amr_live_v71_module
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use omp_lib, only: omp_get_num_threads
    use braids_direct_vector_family_native_v67_module, only: &
        braids_direct_family_topology_v67, braids_direct_family_state_v67, &
        braids_export_direct_family_state_v67, &
        braids_export_direct_family_state_identity_v68, &
        braids_restore_direct_family_state_v67, direct_family_v67_ok, &
        direct_family_v67_face_internal, direct_family_v67_face_closed
    implicit none
    private

    integer, parameter, public :: direct_amr_v71_ok = 0
    integer, parameter, public :: direct_amr_v71_invalid_shape = 710
    integer, parameter, public :: direct_amr_v71_invalid_input = 711
    integer, parameter, public :: direct_amr_v71_invalid_topology = 712
    integer, parameter, public :: direct_amr_v71_invalid_transfer = 713
    integer, parameter, public :: direct_amr_v71_not_conservative = 714
    integer, parameter, public :: direct_amr_v71_parent_mismatch = 715
    integer, parameter, public :: direct_amr_v71_trial_not_pending = 716
    integer, parameter, public :: direct_amr_v71_restore_failure = 717
    integer, parameter, public :: direct_amr_v71_cancelled = 718
    integer, parameter, public :: direct_amr_v71_max_accepted_steps = 4
    integer, parameter, public :: direct_amr_v71_transition_wetdry = 1
    integer, parameter, public :: direct_amr_v71_transition_refine = 2
    integer, parameter, public :: direct_amr_v71_transition_coarsen = 3
    integer, parameter, public :: direct_amr_v71_transition_identity = 4

    type, public :: braids_direct_amr_trial_v71
        private
        logical :: pending = .false.
        integer :: nfrac = 0
        integer :: nold = 0
        integer :: nnew = 0
        integer :: nnewface = 0
        integer :: parent_step = -1
        real(real64) :: parent_time_s = -1.0_real64
        integer :: old_mesh_epoch = -1
        integer :: new_mesh_epoch = -1
        integer :: observed_threads = 1
        integer :: transition_kind = 0
        integer :: wet_state_change_count = 0
        integer :: transfer_application_count = 0
        integer :: face_register_cleanup_count = 0
        integer :: availability_cleanup_count = 0
        integer :: exner_application_count = 0
        real(real64), allocatable :: parent_mass(:,:)
        real(real64), allocatable :: parent_density(:)
        real(real64), allocatable :: parent_signed(:)
        real(real64), allocatable :: parent_absolute(:)
        real(real64), allocatable :: new_mass(:,:)
        real(real64), allocatable :: conservation_residual_kg(:)
        real(real64), allocatable :: cleared_face_register(:,:)
        real(real64), allocatable :: cleared_cell_availability(:,:)
        character(len=64) :: parent_execution_fingerprint = ''
        character(len=64) :: parent_topology_fingerprint = ''
        character(len=64) :: target_execution_fingerprint = ''
        character(len=64) :: target_topology_fingerprint = ''
        logical :: maximum_waterdepth_is_explicit = .false.
        logical :: maximum_waterdepth_enabled = .false.
        type(braids_direct_family_state_v67) :: candidate
    end type braids_direct_amr_trial_v71

    public :: braids_validate_wetdry_topology_v71
    public :: braids_cleanup_wetdry_face_flux_v71
    public :: braids_begin_direct_amr_trial_v71
    public :: braids_commit_direct_amr_trial_v71
    public :: braids_rollback_direct_amr_trial_v71
    public :: braids_cancel_direct_amr_trial_v71
    public :: braids_export_direct_amr_trial_v71

contains

    pure logical function valid_sha256(value)
        character(len=*), intent(in) :: value
        integer :: item, code
        valid_sha256 = len_trim(value) == 64
        if (.not.valid_sha256) return
        do item=1,64
            code=iachar(value(item:item))
            if (.not.((code>=iachar('0').and.code<=iachar('9')).or. &
                      (code>=iachar('a').and.code<=iachar('f')))) then
                valid_sha256=.false.
                return
            endif
        enddo
    end function valid_sha256

    subroutine braids_validate_wetdry_topology_v71( &
            ncell,nface,owner,neighbour,role,wet,status)
        integer, intent(in) :: ncell,nface
        integer, intent(in) :: owner(nface),neighbour(nface),role(nface)
        integer, intent(in) :: wet(ncell)
        integer, intent(out) :: status
        integer, allocatable :: seen(:)
        integer :: face,own,other

        status=direct_amr_v71_invalid_shape
        if(ncell<1.or.nface<1) return
        status=direct_amr_v71_invalid_input
        if(any((wet/=0).and.(wet/=1))) return
        allocate(seen(ncell));seen=0
        status=direct_amr_v71_invalid_topology
        do face=1,nface
            own=owner(face);other=neighbour(face)
            if(own<1.or.own>ncell.or.role(face)<0.or.role(face)>4) then
                deallocate(seen);return
            endif
            seen(own)=seen(own)+1
            if(role(face)==direct_family_v67_face_internal) then
                if(other<1.or.other>ncell.or.other==own.or. &
                   wet(own)/=1.or.wet(other)/=1) then
                    deallocate(seen);return
                endif
                seen(other)=seen(other)+1
            else
                if(other/=0) then
                    deallocate(seen);return
                endif
                if(wet(own)/=1.and. &
                   role(face)/=direct_family_v67_face_closed) then
                    deallocate(seen);return
                endif
            endif
        enddo
        if(any(seen==0)) then
            deallocate(seen);return
        endif
        deallocate(seen)
        status=direct_amr_v71_ok
    end subroutine braids_validate_wetdry_topology_v71

    subroutine braids_cleanup_wetdry_face_flux_v71( &
            nfrac,ncell,nface,owner,neighbour,role,wet,candidate,cleaned, &
            observed_threads,status)
        integer, intent(in) :: nfrac,ncell,nface
        integer, intent(in) :: owner(nface),neighbour(nface),role(nface)
        integer, intent(in) :: wet(ncell)
        real(real64), intent(in) :: candidate(nfrac,nface)
        real(real64), intent(out) :: cleaned(nfrac,nface)
        integer, intent(out) :: observed_threads,status
        integer :: face,fraction,topology_status

        cleaned=0.0_real64;observed_threads=1
        call braids_validate_wetdry_topology_v71( &
            ncell,nface,owner,neighbour,role,wet,topology_status)
        if(topology_status/=direct_amr_v71_ok) then
            status=topology_status;return
        endif
        status=direct_amr_v71_invalid_input
        if(nfrac<1.or.any(.not.ieee_is_finite(candidate))) return
        !$omp parallel default(shared) private(face,fraction)
        !$omp single
        observed_threads=omp_get_num_threads()
        !$omp end single
        !$omp do collapse(2) schedule(static,256)
        do face=1,nface
            do fraction=1,nfrac
                if(role(face)==direct_family_v67_face_closed) then
                    cleaned(fraction,face)=0.0_real64
                else
                    cleaned(fraction,face)=candidate(fraction,face)
                endif
            enddo
        enddo
        !$omp end do
        !$omp end parallel
        status=direct_amr_v71_ok
    end subroutine braids_cleanup_wetdry_face_flux_v71

    subroutine braids_begin_direct_amr_trial_v71( &
            nfrac,old_state,old_area_m2,old_wet,old_cell_availability, &
            new_topology,new_area_m2,new_wet,nlink, &
            link_old,link_new,overlap_area_m2,target_execution_fingerprint, &
            target_topology_fingerprint,target_maximumwaterdepth_explicit, &
            target_maximumwaterdepth_enabled,old_mesh_epoch,new_nface, &
            cancelled,trial,status)
        integer, intent(in) :: nfrac,nlink,old_mesh_epoch,new_nface
        type(braids_direct_family_state_v67), intent(in) :: old_state
        type(braids_direct_family_topology_v67), intent(in) :: new_topology
        real(real64), intent(in) :: old_area_m2(:),new_area_m2(:)
        integer, intent(in) :: old_wet(:),new_wet(:)
        real(real64), intent(in) :: old_cell_availability(:,:)
        integer, intent(in) :: link_old(nlink),link_new(nlink)
        real(real64), intent(in) :: overlap_area_m2(nlink)
        character(len=*), intent(in) :: target_execution_fingerprint
        character(len=*), intent(in) :: target_topology_fingerprint
        logical, intent(in) :: target_maximumwaterdepth_explicit
        logical, intent(in) :: target_maximumwaterdepth_enabled
        logical, intent(in) :: cancelled
        type(braids_direct_amr_trial_v71), intent(out) :: trial
        integer, intent(out) :: status
        real(real64), allocatable :: old_mass(:,:),density(:),active(:,:)
        real(real64), allocatable :: signed(:),absolute(:)
        real(real64), allocatable :: old_cover(:),new_cover(:)
        real(real64) :: time_s,tolerance,scale,total
        integer :: accepted_step,export_status,identity_status,restore_status
        integer :: nold,nnew,link,target,fraction,previous_new,previous_old
        character(len=64) :: old_execution,old_topology
        logical :: old_explicit,old_enabled,identity_cover

        status=direct_amr_v71_invalid_shape
        nold=size(old_area_m2);nnew=size(new_area_m2)
        if(nfrac<1.or.nold<1.or.nnew<1.or.nlink<1.or.new_nface<1.or. &
           size(old_wet)/=nold.or.size(new_wet)/=nnew.or. &
           size(old_cell_availability,1)/=nfrac.or. &
           size(old_cell_availability,2)/=nold) return
        if(cancelled) then
            status=direct_amr_v71_cancelled;return
        endif
        status=direct_amr_v71_invalid_input
        if(old_mesh_epoch<0.or.any(.not.ieee_is_finite(old_area_m2)).or. &
           any(.not.ieee_is_finite(new_area_m2)).or. &
           any(.not.ieee_is_finite(overlap_area_m2)).or. &
           any(.not.ieee_is_finite(old_cell_availability)).or. &
           any(old_area_m2<=0.0_real64).or.any(new_area_m2<=0.0_real64).or. &
           any(overlap_area_m2<=0.0_real64).or. &
           any(old_cell_availability<0.0_real64).or. &
           any((old_wet/=0).and.(old_wet/=1)).or. &
           any((new_wet/=0).and.(new_wet/=1)).or. &
           .not.valid_sha256(target_execution_fingerprint).or. &
           .not.valid_sha256(target_topology_fingerprint).or. &
           .not.target_maximumwaterdepth_explicit) return

        call braids_export_direct_family_state_v67( &
            old_state,accepted_step,time_s,old_mass,density,active,signed, &
            absolute,export_status)
        call braids_export_direct_family_state_identity_v68( &
            old_state,old_execution,old_topology,old_explicit,old_enabled, &
            identity_status)
        if(export_status/=direct_family_v67_ok.or. &
           identity_status/=direct_family_v67_ok.or. &
           size(old_mass,1)/=nfrac.or.size(old_mass,2)/=nold.or. &
           accepted_step>direct_amr_v71_max_accepted_steps.or. &
           .not.old_explicit.or. &
           (old_enabled.neqv.target_maximumwaterdepth_enabled)) then
            status=direct_amr_v71_parent_mismatch
            return
        endif

        allocate(old_cover(nold),new_cover(nnew))
        old_cover=0.0_real64;new_cover=0.0_real64
        previous_new=0;previous_old=0
        status=direct_amr_v71_invalid_transfer
        do link=1,nlink
            if(link_old(link)<1.or.link_old(link)>nold.or. &
               link_new(link)<1.or.link_new(link)>nnew) then
                deallocate(old_cover,new_cover);return
            endif
            if(link_new(link)<previous_new.or. &
               (link_new(link)==previous_new.and.link_old(link)<=previous_old)) then
                deallocate(old_cover,new_cover);return
            endif
            previous_new=link_new(link);previous_old=link_old(link)
            old_cover(link_old(link))=old_cover(link_old(link))+ &
                overlap_area_m2(link)
            new_cover(link_new(link))=new_cover(link_new(link))+ &
                overlap_area_m2(link)
        enddo
        tolerance=8192.0_real64*epsilon(1.0_real64)
        if(any(abs(old_cover-old_area_m2)>tolerance*max(1.0_real64,old_area_m2)).or. &
           any(abs(new_cover-new_area_m2)>tolerance*max(1.0_real64,new_area_m2))) then
            deallocate(old_cover,new_cover);return
        endif
        deallocate(old_cover,new_cover)

        identity_cover=nold==nnew.and.nlink==nold
        if(identity_cover) then
            do link=1,nlink
                if(link_old(link)/=link_new(link).or. &
                   abs(overlap_area_m2(link)-old_area_m2(link_old(link)))> &
                       tolerance*max(1.0_real64,old_area_m2(link_old(link))).or. &
                   abs(overlap_area_m2(link)-new_area_m2(link_new(link)))> &
                       tolerance*max(1.0_real64,new_area_m2(link_new(link)))) then
                    identity_cover=.false.;exit
                endif
            enddo
        endif
        trial%wet_state_change_count=0
        if(identity_cover) trial%wet_state_change_count=count(old_wet/=new_wet)
        if(identity_cover.and.trial%wet_state_change_count>0) then
            trial%transition_kind=direct_amr_v71_transition_wetdry
        elseif(nnew>nold) then
            trial%transition_kind=direct_amr_v71_transition_refine
        elseif(nnew<nold) then
            trial%transition_kind=direct_amr_v71_transition_coarsen
        else
            trial%transition_kind=direct_amr_v71_transition_identity
        endif

        trial%nfrac=nfrac;trial%nold=nold;trial%nnew=nnew
        trial%nnewface=new_nface;trial%parent_step=accepted_step
        trial%parent_time_s=time_s;trial%old_mesh_epoch=old_mesh_epoch
        trial%new_mesh_epoch=old_mesh_epoch+1
        trial%parent_execution_fingerprint=old_execution
        trial%parent_topology_fingerprint=old_topology
        trial%target_execution_fingerprint=trim(target_execution_fingerprint)
        trial%target_topology_fingerprint=trim(target_topology_fingerprint)
        trial%maximum_waterdepth_is_explicit=target_maximumwaterdepth_explicit
        trial%maximum_waterdepth_enabled=target_maximumwaterdepth_enabled
        allocate(trial%parent_mass(nfrac,nold),trial%parent_density(nfrac), &
                 trial%parent_signed(nfrac),trial%parent_absolute(nfrac), &
                 trial%new_mass(nfrac,nnew), &
                 trial%conservation_residual_kg(nfrac), &
                 trial%cleared_face_register(nfrac,new_nface), &
                 trial%cleared_cell_availability(nfrac,nnew))
        trial%parent_mass=old_mass;trial%parent_density=density
        trial%parent_signed=signed;trial%parent_absolute=absolute
        trial%new_mass=0.0_real64
        trial%cleared_face_register=0.0_real64
        trial%cleared_cell_availability=0.0_real64
        trial%observed_threads=1
        !$omp parallel default(shared) private(target,fraction,link,total)
        !$omp single
        trial%observed_threads=omp_get_num_threads()
        !$omp end single
        !$omp do collapse(2) schedule(static,256)
        do target=1,nnew
            do fraction=1,nfrac
                total=0.0_real64
                do link=1,nlink
                    if(link_new(link)==target) total=total+ &
                        old_mass(fraction,link_old(link))*overlap_area_m2(link)
                enddo
                trial%new_mass(fraction,target)=total/new_area_m2(target)
            enddo
        enddo
        !$omp end do
        !$omp end parallel
        if(any(.not.ieee_is_finite(trial%new_mass)).or. &
           any(trial%new_mass<0.0_real64)) then
            status=direct_amr_v71_invalid_transfer;return
        endif
        do fraction=1,nfrac
            trial%conservation_residual_kg(fraction)= &
                sum(trial%new_mass(fraction,:)*new_area_m2)- &
                sum(old_mass(fraction,:)*old_area_m2)
            scale=max(1.0_real64, &
                sum(abs(trial%new_mass(fraction,:))*new_area_m2), &
                sum(abs(old_mass(fraction,:))*old_area_m2))
            if(abs(trial%conservation_residual_kg(fraction))> &
               16384.0_real64*epsilon(1.0_real64)*scale) then
                status=direct_amr_v71_not_conservative;return
            endif
        enddo

        call braids_restore_direct_family_state_v67( &
            nfrac,new_topology,accepted_step,time_s,trial%new_mass,density, &
            signed,absolute,target_execution_fingerprint, &
            target_topology_fingerprint,target_maximumwaterdepth_explicit, &
            target_maximumwaterdepth_enabled,trial%candidate,restore_status)
        if(restore_status/=direct_family_v67_ok) then
            status=direct_amr_v71_restore_failure;return
        endif
        trial%transfer_application_count=1
        trial%face_register_cleanup_count=1
        trial%availability_cleanup_count=1
        trial%exner_application_count=0
        trial%pending=.true.
        status=direct_amr_v71_ok
    end subroutine braids_begin_direct_amr_trial_v71

    subroutine validate_parent(state,trial,status)
        type(braids_direct_family_state_v67), intent(in) :: state
        type(braids_direct_amr_trial_v71), intent(in) :: trial
        integer, intent(out) :: status
        real(real64), allocatable :: mass(:,:),density(:),active(:,:)
        real(real64), allocatable :: signed(:),absolute(:)
        real(real64) :: time_s
        integer :: accepted_step,export_status,identity_status
        character(len=64) :: execution,topology
        logical :: explicit,enabled

        status=direct_amr_v71_trial_not_pending
        if(.not.trial%pending) return
        call braids_export_direct_family_state_v67( &
            state,accepted_step,time_s,mass,density,active,signed,absolute, &
            export_status)
        call braids_export_direct_family_state_identity_v68( &
            state,execution,topology,explicit,enabled,identity_status)
        status=direct_amr_v71_parent_mismatch
        if(export_status/=direct_family_v67_ok.or. &
           identity_status/=direct_family_v67_ok) return
        if(accepted_step/=trial%parent_step.or.time_s/=trial%parent_time_s.or. &
           trim(execution)/=trim(trial%parent_execution_fingerprint).or. &
           trim(topology)/=trim(trial%parent_topology_fingerprint).or. &
           size(mass,1)/=trial%nfrac.or.size(mass,2)/=trial%nold.or. &
           any(mass/=trial%parent_mass).or.any(density/=trial%parent_density).or. &
           any(signed/=trial%parent_signed).or. &
           any(absolute/=trial%parent_absolute)) return
        status=direct_amr_v71_ok
    end subroutine validate_parent

    subroutine braids_commit_direct_amr_trial_v71(state,trial,status)
        type(braids_direct_family_state_v67), intent(inout) :: state
        type(braids_direct_amr_trial_v71), intent(inout) :: trial
        integer, intent(out) :: status
        call validate_parent(state,trial,status)
        if(status/=direct_amr_v71_ok) return
        state=trial%candidate
        trial%pending=.false.
    end subroutine braids_commit_direct_amr_trial_v71

    subroutine braids_rollback_direct_amr_trial_v71(state,trial,status)
        type(braids_direct_family_state_v67), intent(in) :: state
        type(braids_direct_amr_trial_v71), intent(inout) :: trial
        integer, intent(out) :: status
        call validate_parent(state,trial,status)
        if(status==direct_amr_v71_ok) trial%pending=.false.
    end subroutine braids_rollback_direct_amr_trial_v71

    subroutine braids_cancel_direct_amr_trial_v71(state,trial,status)
        type(braids_direct_family_state_v67), intent(in) :: state
        type(braids_direct_amr_trial_v71), intent(inout) :: trial
        integer, intent(out) :: status
        call validate_parent(state,trial,status)
        if(status==direct_amr_v71_ok) trial%pending=.false.
    end subroutine braids_cancel_direct_amr_trial_v71

    subroutine braids_export_direct_amr_trial_v71( &
            trial,pending,old_mesh_epoch,new_mesh_epoch,observed_threads, &
            transition_kind,wet_state_change_count, &
            transfer_application_count,face_register_cleanup_count, &
            availability_cleanup_count,exner_application_count,new_mass, &
            conservation_residual_kg,cleared_face_register, &
            cleared_cell_availability,status)
        type(braids_direct_amr_trial_v71), intent(in) :: trial
        logical, intent(out) :: pending
        integer, intent(out) :: old_mesh_epoch,new_mesh_epoch,observed_threads
        integer, intent(out) :: transition_kind,wet_state_change_count
        integer, intent(out) :: transfer_application_count
        integer, intent(out) :: face_register_cleanup_count
        integer, intent(out) :: availability_cleanup_count
        integer, intent(out) :: exner_application_count,status
        real(real64), allocatable, intent(out) :: new_mass(:,:)
        real(real64), allocatable, intent(out) :: conservation_residual_kg(:)
        real(real64), allocatable, intent(out) :: cleared_face_register(:,:)
        real(real64), allocatable, intent(out) :: cleared_cell_availability(:,:)

        pending=trial%pending;old_mesh_epoch=trial%old_mesh_epoch
        new_mesh_epoch=trial%new_mesh_epoch
        observed_threads=trial%observed_threads
        transition_kind=trial%transition_kind
        wet_state_change_count=trial%wet_state_change_count
        transfer_application_count=trial%transfer_application_count
        face_register_cleanup_count=trial%face_register_cleanup_count
        availability_cleanup_count=trial%availability_cleanup_count
        exner_application_count=trial%exner_application_count
        status=direct_amr_v71_trial_not_pending
        if(.not.trial%pending) return
        allocate(new_mass(trial%nfrac,trial%nnew), &
                  conservation_residual_kg(trial%nfrac), &
                  cleared_face_register(trial%nfrac,trial%nnewface), &
                  cleared_cell_availability(trial%nfrac,trial%nnew))
        new_mass=trial%new_mass
        conservation_residual_kg=trial%conservation_residual_kg
        cleared_face_register=trial%cleared_face_register
        cleared_cell_availability=trial%cleared_cell_availability
        status=direct_amr_v71_ok
    end subroutine braids_export_direct_amr_trial_v71

end module braids_direct_vector_wetdry_amr_live_v71_module
