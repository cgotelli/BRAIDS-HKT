! SPDX-License-Identifier: AGPL-3.0-or-later
!
! Additive bounded Q3/v75 native/OpenMP transaction owner.  This module is
! deliberately outside every accepted child implementation.  It consumes a
! capacity/live-taurat pair sealed by the v73 contract, optionally derives a
! v72 cell channel, validates v71 wet/dry ownership, applies one v65 face +
! ISlope=2 + inventory + Exner transaction, and validates the supported v70
! boundary roles.  Prescribed external feed is rejected before a trial.
module braids_unified_direct_vector_v75_module
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use omp_lib, only: omp_get_num_threads
    use braids_bedload_adaptation_live_v72_module, only: &
        braids_bedload_adaptation_topology_v72, &
        braids_relax_bedload_capacity_v72, adaptation_v72_ok
    use braids_bedload_boundary_v70_module, only: &
        bed_boundary_v70_closed, bed_boundary_v70_equilibrium, &
        bed_boundary_v70_prescribed, bed_boundary_v70_free_export, &
        bed_boundary_v70_source_none, bed_boundary_v70_source_equilibrium, &
        bed_boundary_v70_source_owner_export
    use braids_direct_vector_wetdry_amr_live_v71_module, only: &
        braids_validate_wetdry_topology_v71, &
        braids_cleanup_wetdry_face_flux_v71, direct_amr_v71_ok, &
        direct_amr_v71_transition_wetdry, direct_amr_v71_transition_refine, &
        direct_amr_v71_transition_coarsen, direct_amr_v71_transition_identity
    use braids_flow_bed_slope_operator_module, only: &
        braids_flow_bed_slope_direction, flow_slope_ok
    use braids_scalar_flow_live_v65_module, only: &
        scalar_flow_projection_topology_v65, scalar_live_ok, &
        scalar_live_face_internal, scalar_live_face_trace_inlet, &
        scalar_live_face_trace_outlet, scalar_live_face_owner_outlet, &
        scalar_live_face_closed, scalar_live_donor_owner, &
        scalar_live_donor_neighbour, scalar_live_donor_trace, &
        scalar_live_donor_none, braids_prepare_scalar_flow_topology_v65, &
        braids_scalar_flow_face_candidates_v65, &
        braids_scalar_flow_project_exner_v65
    implicit none
    private

    integer, parameter, public :: unified_v75_ok = 0
    integer, parameter, public :: unified_v75_invalid_shape = 750
    integer, parameter, public :: unified_v75_invalid_input = 751
    integer, parameter, public :: unified_v75_invalid_topology = 752
    integer, parameter, public :: unified_v75_source_mismatch = 753
    integer, parameter, public :: unified_v75_prescribed_rejected = 754
    integer, parameter, public :: unified_v75_adaptation_failure = 755
    integer, parameter, public :: unified_v75_face_failure = 756
    integer, parameter, public :: unified_v75_exner_failure = 757
    integer, parameter, public :: unified_v75_parent_mismatch = 758
    integer, parameter, public :: unified_v75_trial_not_pending = 759
    integer, parameter, public :: unified_v75_step_limit = 760
    integer, parameter, public :: unified_v75_cancelled = 761
    integer, parameter, public :: unified_v75_transfer_failure = 762
    integer, parameter, public :: unified_v75_max_steps = 4
    character(len=64), parameter, public :: unified_v75_taurat_contract = &
        'cc0c86f58a8d38513c3793f0c1cf8f45bc25713f9c8f8e6fa4a20763648d6599'

    type, public :: braids_unified_topology_v75
        private
        logical :: prepared = .false.
        integer :: ncell = 0, nface = 0, nboundary = 0
        integer, allocatable :: owner(:),neighbour(:),role(:),wet(:)
        integer, allocatable :: boundary_faces(:),boundary_mode(:)
        real(real64), allocatable :: normal(:,:),length_m(:),cell_area(:)
        type(scalar_flow_projection_topology_v65) :: projection
    end type braids_unified_topology_v75

    type, public :: braids_unified_state_v75
        private
        logical :: initialized = .false.
        integer :: nfrac = 0, ncell = 0, nface = 0
        integer :: accepted_step = 0, mesh_epoch = 0
        real(real64) :: time_s = 0.0_real64
        real(real64), allocatable :: bed_mass(:,:),bulk_density(:)
        real(real64), allocatable :: active_fraction(:,:)
        real(real64), allocatable :: boundary_signed(:),boundary_absolute(:)
        real(real64), allocatable :: accepted_face_flux(:,:)
        real(real64), allocatable :: accepted_cell_availability(:,:)
    end type braids_unified_state_v75

    type, public :: braids_unified_trial_v75
        private
        logical :: pending = .false.
        integer :: nfrac = 0, ncell = 0, nface = 0, nboundary = 0
        integer :: parent_step = -1, parent_epoch = -1
        real(real64) :: parent_time = -1.0_real64
        integer :: producer_taurat_application_count = 0
        integer :: adaptation_application_count = 0
        integer :: wetdry_owner_application_count = 0
        integer :: face_reconstruction_application_count = 0
        integer :: bed_slope_application_count = 0
        integer :: boundary_owner_application_count = 0
        integer :: active_fraction_application_count = 0
        integer :: inventory_fill_application_count = 0
        integer :: exner_application_count = 0
        integer :: producer_threads = 1,adaptation_threads = 1
        integer :: face_threads = 1,slope_threads = 1,exner_threads = 1
        real(real64), allocatable :: parent_mass(:,:),parent_signed(:)
        real(real64), allocatable :: parent_absolute(:)
        real(real64), allocatable :: consumed_cell_capacity(:,:,:)
        real(real64), allocatable :: face_taurat(:,:),face_flux(:,:)
        real(real64), allocatable :: conservation_residual(:)
        real(real64), allocatable :: named_boundary_transfer(:,:)
        integer, allocatable :: boundary_source(:,:)
        type(braids_unified_state_v75) :: candidate
    end type braids_unified_trial_v75

    type, public :: braids_unified_regrid_trial_v75
        private
        logical :: pending = .false.
        integer :: nfrac=0,nold=0,nnew=0,parent_step=-1,parent_epoch=-1
        real(real64) :: parent_time=-1.0_real64
        integer :: transition_kind=0,wet_state_change_count=0
        integer :: transfer_application_count=0
        integer :: face_register_cleanup_count=0
        integer :: availability_cleanup_count=0
        integer :: exner_application_count=0
        integer :: observed_threads=1
        real(real64), allocatable :: parent_mass(:,:),residual(:)
        type(braids_unified_state_v75) :: candidate
    end type braids_unified_regrid_trial_v75

    public :: braids_prepare_unified_topology_v75
    public :: braids_initialize_unified_state_v75
    public :: braids_restore_unified_state_v75
    public :: braids_export_unified_state_v75
    public :: braids_begin_unified_trial_v75
    public :: braids_commit_unified_trial_v75
    public :: braids_rollback_unified_trial_v75
    public :: braids_cancel_unified_trial_v75
    public :: braids_export_unified_trial_v75
    public :: braids_begin_unified_regrid_v75
    public :: braids_commit_unified_regrid_v75
    public :: braids_rollback_unified_regrid_v75
    public :: braids_cancel_unified_regrid_v75
    public :: braids_export_unified_regrid_v75

contains

    subroutine derive_fraction_v75(state,status)
        type(braids_unified_state_v75), intent(inout) :: state
        integer, intent(out) :: status
        integer :: cell
        real(real64) :: total
        status=unified_v75_exner_failure
        if(.not.allocated(state%active_fraction)) &
            allocate(state%active_fraction(state%nfrac,state%ncell))
        state%active_fraction=0.0_real64
        do cell=1,state%ncell
            total=sum(state%bed_mass(:,cell)/state%bulk_density)
            if(.not.ieee_is_finite(total).or.total<=0.0_real64) return
            state%active_fraction(:,cell)= &
                (state%bed_mass(:,cell)/state%bulk_density)/total
        enddo
        status=unified_v75_ok
    end subroutine derive_fraction_v75

    subroutine braids_prepare_unified_topology_v75( &
            ncell,nface,owner,neighbour,role,wet,normal,length_m,cell_area, &
            ninc,cell_face_offsets,cell_faces,nboundary,boundary_faces, &
            boundary_mode,topology,status)
        integer, intent(in) :: ncell,nface,ninc,nboundary
        integer, intent(in) :: owner(nface),neighbour(nface),role(nface)
        integer, intent(in) :: wet(ncell),cell_face_offsets(ncell+1)
        integer, intent(in) :: cell_faces(ninc),boundary_faces(nboundary)
        integer, intent(in) :: boundary_mode(nboundary)
        real(real64), intent(in) :: normal(2,nface),length_m(nface)
        real(real64), intent(in) :: cell_area(ncell)
        type(braids_unified_topology_v75), intent(out) :: topology
        integer, intent(out) :: status
        integer :: nested,item,face,mode

        status=unified_v75_invalid_shape
        if(ncell<1.or.nface<1.or.ninc<1.or.nboundary<1) return
        status=unified_v75_invalid_input
        if(any(.not.ieee_is_finite(normal)).or. &
           any(.not.ieee_is_finite(length_m)).or. &
           any(.not.ieee_is_finite(cell_area)).or.any(length_m<=0.0_real64).or. &
           any(cell_area<=0.0_real64)) return
        if(any(boundary_mode==bed_boundary_v70_prescribed)) then
            status=unified_v75_prescribed_rejected;return
        endif
        if(any(boundary_mode<bed_boundary_v70_closed).or. &
           any(boundary_mode>bed_boundary_v70_free_export)) return
        call braids_validate_wetdry_topology_v71( &
            ncell,nface,owner,neighbour,role,wet,nested)
        if(nested/=direct_amr_v71_ok) then
            status=unified_v75_invalid_topology;return
        endif
        call braids_prepare_scalar_flow_topology_v65( &
            ncell,nface,owner,neighbour,role,normal,length_m,ninc, &
            cell_face_offsets,cell_faces,nboundary,boundary_faces, &
            topology%projection,nested)
        if(nested/=scalar_live_ok) then
            status=unified_v75_invalid_topology;return
        endif
        do item=1,nboundary
            face=boundary_faces(item);mode=boundary_mode(item)
            if(face<1.or.face>nface.or.role(face)==scalar_live_face_internal) then
                status=unified_v75_invalid_topology;return
            endif
            if(mode==bed_boundary_v70_closed.and. &
               role(face)/=scalar_live_face_closed) then
                status=unified_v75_invalid_topology;return
            endif
        enddo
        topology%ncell=ncell;topology%nface=nface;topology%nboundary=nboundary
        allocate(topology%owner(nface),topology%neighbour(nface), &
                 topology%role(nface),topology%wet(ncell), &
                 topology%normal(2,nface),topology%length_m(nface), &
                 topology%cell_area(ncell),topology%boundary_faces(nboundary), &
                 topology%boundary_mode(nboundary))
        topology%owner=owner;topology%neighbour=neighbour;topology%role=role
        topology%wet=wet;topology%normal=normal;topology%length_m=length_m
        topology%cell_area=cell_area;topology%boundary_faces=boundary_faces
        topology%boundary_mode=boundary_mode;topology%prepared=.true.
        status=unified_v75_ok
    end subroutine braids_prepare_unified_topology_v75

    subroutine braids_restore_unified_state_v75( &
            nfrac,topology,accepted_step,time_s,mesh_epoch,bed_mass, &
            bulk_density,boundary_signed,boundary_absolute,accepted_flux, &
            accepted_availability,state,status)
        integer, intent(in) :: nfrac,accepted_step,mesh_epoch
        type(braids_unified_topology_v75), intent(in) :: topology
        real(real64), intent(in) :: time_s
        real(real64), intent(in) :: bed_mass(nfrac,topology%ncell)
        real(real64), intent(in) :: bulk_density(nfrac)
        real(real64), intent(in) :: boundary_signed(nfrac)
        real(real64), intent(in) :: boundary_absolute(nfrac)
        real(real64), intent(in) :: accepted_flux(nfrac,topology%nface)
        real(real64), intent(in) :: &
            accepted_availability(nfrac,topology%ncell)
        type(braids_unified_state_v75), intent(out) :: state
        integer, intent(out) :: status

        status=unified_v75_invalid_input
        if(.not.topology%prepared.or.nfrac<1.or.accepted_step<0.or. &
           accepted_step>unified_v75_max_steps.or.mesh_epoch<0.or. &
           .not.ieee_is_finite(time_s).or.time_s<0.0_real64.or. &
           any(.not.ieee_is_finite(bed_mass)).or.any(bed_mass<0.0_real64).or. &
           any(.not.ieee_is_finite(bulk_density)).or. &
           any(bulk_density<=0.0_real64).or. &
           any(.not.ieee_is_finite(boundary_signed)).or. &
           any(.not.ieee_is_finite(boundary_absolute)).or. &
           any(boundary_absolute<abs(boundary_signed)).or. &
           any(.not.ieee_is_finite(accepted_flux)).or. &
           any(.not.ieee_is_finite(accepted_availability)).or. &
           any(accepted_availability<0.0_real64)) return
        state%nfrac=nfrac;state%ncell=topology%ncell;state%nface=topology%nface
        state%accepted_step=accepted_step;state%time_s=time_s
        state%mesh_epoch=mesh_epoch
        allocate(state%bed_mass(nfrac,topology%ncell), &
                 state%bulk_density(nfrac),state%boundary_signed(nfrac), &
                 state%boundary_absolute(nfrac), &
                 state%accepted_face_flux(nfrac,topology%nface), &
                 state%accepted_cell_availability(nfrac,topology%ncell))
        state%bed_mass=bed_mass;state%bulk_density=bulk_density
        state%boundary_signed=boundary_signed
        state%boundary_absolute=boundary_absolute
        state%accepted_face_flux=accepted_flux
        state%accepted_cell_availability=accepted_availability
        call derive_fraction_v75(state,status)
        if(status/=unified_v75_ok) return
        state%initialized=.true.
    end subroutine braids_restore_unified_state_v75

    subroutine braids_initialize_unified_state_v75( &
            nfrac,topology,bed_mass,bulk_density,state,status)
        integer, intent(in) :: nfrac
        type(braids_unified_topology_v75), intent(in) :: topology
        real(real64), intent(in) :: bed_mass(nfrac,topology%ncell)
        real(real64), intent(in) :: bulk_density(nfrac)
        type(braids_unified_state_v75), intent(out) :: state
        integer, intent(out) :: status
        real(real64), allocatable :: zero(:),zero_face(:,:),zero_cell(:,:)
        allocate(zero(nfrac),zero_face(nfrac,topology%nface), &
                 zero_cell(nfrac,topology%ncell))
        zero=0.0_real64;zero_face=0.0_real64;zero_cell=0.0_real64
        call braids_restore_unified_state_v75( &
            nfrac,topology,0,0.0_real64,0,bed_mass,bulk_density,zero,zero, &
            zero_face,zero_cell,state,status)
    end subroutine braids_initialize_unified_state_v75

    subroutine braids_export_unified_state_v75( &
            state,initialized,accepted_step,time_s,mesh_epoch,bed_mass, &
            active_fraction,boundary_signed,boundary_absolute,accepted_flux, &
            accepted_availability,status)
        type(braids_unified_state_v75), intent(in) :: state
        logical, intent(out) :: initialized
        integer, intent(out) :: accepted_step,mesh_epoch,status
        real(real64), intent(out) :: time_s
        real(real64), allocatable, intent(out) :: bed_mass(:,:),active_fraction(:,:)
        real(real64), allocatable, intent(out) :: boundary_signed(:)
        real(real64), allocatable, intent(out) :: boundary_absolute(:)
        real(real64), allocatable, intent(out) :: accepted_flux(:,:)
        real(real64), allocatable, intent(out) :: accepted_availability(:,:)
        initialized=state%initialized;accepted_step=state%accepted_step
        time_s=state%time_s;mesh_epoch=state%mesh_epoch
        status=unified_v75_invalid_input
        if(.not.state%initialized) return
        allocate(bed_mass(state%nfrac,state%ncell), &
                 active_fraction(state%nfrac,state%ncell), &
                 boundary_signed(state%nfrac),boundary_absolute(state%nfrac), &
                 accepted_flux(state%nfrac,state%nface), &
                 accepted_availability(state%nfrac,state%ncell))
        bed_mass=state%bed_mass;active_fraction=state%active_fraction
        boundary_signed=state%boundary_signed
        boundary_absolute=state%boundary_absolute
        accepted_flux=state%accepted_face_flux
        accepted_availability=state%accepted_cell_availability
        status=unified_v75_ok
    end subroutine braids_export_unified_state_v75

    subroutine braids_begin_unified_trial_v75( &
            nfrac,topology,adaptation_topology,adaptation_enabled, &
            live_taurat_fraction,equilibrium_capacity,trace_capacity, &
            cell_taurat,trace_taurat,taurat_contract,same_invocation, &
            q,d50_m,length_diameters,dry_tolerance,dx_m,dy_m,downhill, &
            depth,mean_diameter,hiding,alfabs,alfabn, &
            hydraulic_outward,state,dt_s,morphological_factor,cancelled, &
            trial,status)
        integer, intent(in) :: nfrac
        type(braids_unified_topology_v75), intent(in) :: topology
        type(braids_bedload_adaptation_topology_v72), intent(in) :: &
            adaptation_topology
        logical, intent(in) :: adaptation_enabled,same_invocation,cancelled
        integer, intent(in) :: live_taurat_fraction(nfrac)
        real(real64), intent(in) :: equilibrium_capacity(2,nfrac,topology%ncell)
        real(real64), intent(in) :: trace_capacity(2,nfrac,topology%nface)
        real(real64), intent(in) :: cell_taurat(nfrac,topology%ncell)
        real(real64), intent(in) :: trace_taurat(nfrac,topology%nface)
        character(len=*), intent(in) :: taurat_contract
        real(real64), intent(in) :: q(3,topology%ncell),d50_m(nfrac)
        real(real64), intent(in) :: length_diameters,dry_tolerance,dx_m,dy_m
        real(real64), intent(in) :: downhill(2,topology%nface)
        real(real64), intent(in) :: depth(nfrac,topology%nface)
        real(real64), intent(in) :: mean_diameter(nfrac,topology%nface)
        real(real64), intent(in) :: hiding(nfrac,topology%ncell)
        real(real64), intent(in) :: alfabs,alfabn
        real(real64), intent(in) :: hydraulic_outward(topology%nboundary)
        type(braids_unified_state_v75), intent(in) :: state
        real(real64), intent(in) :: dt_s,morphological_factor
        type(braids_unified_trial_v75), intent(out) :: trial
        integer, intent(out) :: status
        real(real64), allocatable :: boundary_feed(:,:),raw(:,:,:)
        real(real64), allocatable :: corrected(:,:,:),dummy_direction(:,:,:)
        real(real64), allocatable :: dummy_downhill(:,:),dummy_stress(:)
        real(real64), allocatable :: water_density(:),d50_face(:,:)
        real(real64), allocatable :: weights(:),cleaned(:,:)
        real(real64), allocatable :: cell_delta(:,:),inventory_fill(:,:)
        real(real64), allocatable :: availability(:,:),boundary_transfer(:)
        real(real64), allocatable :: boundary_absolute(:)
        integer, allocatable :: offsets(:),indices(:),raw_donor(:,:)
        integer, allocatable :: donor(:,:),donor_cell(:,:),local_status(:,:)
        integer :: nested,face,grain,own,other,item,source,mode
        integer :: raw_active,raw_inventory,cleanup_threads,derive_status
        integer :: iterations
        real(real64) :: residual,tolerance,corrected_value(2),normal_value
        real(real64) :: amount

        status=unified_v75_invalid_input
        if(cancelled) then
            status=unified_v75_cancelled;return
        endif
        if(.not.topology%prepared.or..not.state%initialized.or. &
           state%nfrac/=nfrac.or.state%ncell/=topology%ncell.or. &
           state%nface/=topology%nface.or.state%accepted_step>=unified_v75_max_steps) then
            if(state%accepted_step>=unified_v75_max_steps) &
                status=unified_v75_step_limit
            return
        endif
        if(trim(taurat_contract)/=unified_v75_taurat_contract.or. &
           .not.same_invocation) then
            status=unified_v75_source_mismatch;return
        endif
        if(any((live_taurat_fraction/=0).and.(live_taurat_fraction/=1)).or. &
           count(live_taurat_fraction==1)<1.or. &
           any(.not.ieee_is_finite(equilibrium_capacity)).or. &
           any(.not.ieee_is_finite(trace_capacity)).or. &
           any(.not.ieee_is_finite(cell_taurat)).or. &
           any(.not.ieee_is_finite(trace_taurat)).or. &
           any(cell_taurat<0.0_real64).or.any(trace_taurat<0.0_real64).or. &
           .not.ieee_is_finite(dt_s).or.dt_s<=0.0_real64.or. &
           .not.ieee_is_finite(morphological_factor).or. &
           morphological_factor<0.0_real64) return
        do grain=1,nfrac
            if(live_taurat_fraction(grain)==0.and. &
               (any(cell_taurat(grain,:)/=0.0_real64).or. &
                any(trace_taurat(grain,:)/=0.0_real64))) then
                status=unified_v75_source_mismatch;return
            endif
        enddo

        trial%nfrac=nfrac;trial%ncell=topology%ncell
        trial%nface=topology%nface;trial%nboundary=topology%nboundary
        trial%parent_step=state%accepted_step;trial%parent_time=state%time_s
        trial%parent_epoch=state%mesh_epoch
        allocate(trial%parent_mass(nfrac,topology%ncell), &
                 trial%parent_signed(nfrac),trial%parent_absolute(nfrac), &
                 trial%consumed_cell_capacity(2,nfrac,topology%ncell), &
                 trial%face_taurat(nfrac,topology%nface), &
                 trial%face_flux(nfrac,topology%nface), &
                 trial%conservation_residual(nfrac), &
                 trial%named_boundary_transfer(nfrac,topology%nboundary), &
                 trial%boundary_source(nfrac,topology%nboundary))
        trial%parent_mass=state%bed_mass;trial%parent_signed=state%boundary_signed
        trial%parent_absolute=state%boundary_absolute
        trial%consumed_cell_capacity=equilibrium_capacity
        trial%producer_taurat_application_count=1
        trial%producer_threads=1
        if(adaptation_enabled) then
            if(.not.adaptation_topology%prepared.or. &
               adaptation_topology%ncell/=topology%ncell.or. &
               adaptation_topology%nface/=topology%nface.or. &
               adaptation_topology%nboundary/=topology%nboundary.or. &
               any(adaptation_topology%owner/=topology%owner).or. &
               any(adaptation_topology%neighbour/=topology%neighbour)) then
                status=unified_v75_invalid_topology;return
            endif
            allocate(boundary_feed(nfrac,topology%nboundary));boundary_feed=0.0_real64
            call braids_relax_bedload_capacity_v72( &
                adaptation_topology,nfrac,q,d50_m,length_diameters, &
                dry_tolerance,dx_m,dy_m,boundary_feed,equilibrium_capacity, &
                trial%consumed_cell_capacity,iterations,residual,tolerance, &
                nested,trial%adaptation_threads,.false.)
            if(nested/=adaptation_v72_ok) then
                status=unified_v75_adaptation_failure;return
            endif
            trial%adaptation_application_count=1
        else
            trial%adaptation_application_count=0;trial%adaptation_threads=1
        endif

        allocate(dummy_direction(2,nfrac,topology%ncell), &
                 offsets(topology%ncell+1),indices(topology%ncell), &
                 weights(topology%ncell),dummy_downhill(2,topology%nface), &
                 dummy_stress(topology%nface),water_density(topology%nface), &
                 d50_face(nfrac,topology%nface), &
                 raw(2,nfrac,topology%nface), &
                 corrected(2,nfrac,topology%nface), &
                 raw_donor(nfrac,topology%nface), &
                 donor(nfrac,topology%nface), &
                 donor_cell(nfrac,topology%nface), &
                 local_status(nfrac,topology%nface))
        dummy_direction=trial%consumed_cell_capacity
        do item=1,topology%ncell+1;offsets(item)=item;enddo
        do item=1,topology%ncell;indices(item)=item;weights(item)=1.0_real64;enddo
        dummy_downhill=0.0_real64;dummy_stress=0.0_real64
        water_density=1000.0_real64
        do grain=1,nfrac;d50_face(grain,:)=d50_m(grain);enddo
        call braids_scalar_flow_face_candidates_v65( &
            topology%ncell,topology%nface,nfrac,0,.true.,.false., &
            topology%owner,topology%neighbour,topology%role,topology%normal, &
            trial%consumed_cell_capacity,dummy_direction,trace_capacity, &
            topology%ncell,offsets,indices,weights,topology%ncell,offsets, &
            indices,weights,dummy_downhill,dummy_stress,9.81_real64, &
            state%bulk_density,water_density,depth,d50_face,mean_diameter, &
            hiding,alfabs,alfabn,1.0_real64,1.0_real64,1.0_real64, &
            1.0_real64,1.0_real64,1.0_real64,raw,corrected,raw_donor, &
            donor,donor_cell,nested,trial%face_threads)
        if(nested/=scalar_live_ok) then
            status=unified_v75_face_failure;return
        endif
        trial%face_reconstruction_application_count=1
        local_status=0;trial%face_taurat=0.0_real64
        !$omp parallel default(shared) private(face,grain,own,other, &
        !$omp& corrected_value,normal_value,nested)
        !$omp single
        trial%slope_threads=omp_get_num_threads()
        !$omp end single
        !$omp do collapse(2) schedule(static,256)
        do face=1,topology%nface
            do grain=1,nfrac
                own=topology%owner(face);other=topology%neighbour(face)
                if(live_taurat_fraction(grain)==1) then
                    if(topology%role(face)==scalar_live_face_internal) then
                        trial%face_taurat(grain,face)=0.5_real64*( &
                            cell_taurat(grain,own)+cell_taurat(grain,other))
                    elseif(topology%role(face)/=scalar_live_face_closed) then
                        trial%face_taurat(grain,face)=cell_taurat(grain,own)
                    endif
                endif
                corrected_value=raw(:,grain,face)
                if(topology%role(face)/=scalar_live_face_closed) then
                    call braids_flow_bed_slope_direction( &
                        2,raw(1,grain,face),raw(2,grain,face), &
                        downhill(1,face),downhill(2,face), &
                        trial%face_taurat(grain,face),1.0_real64, &
                        d50_m(grain),mean_diameter(grain,face), &
                        hiding(grain,own),1.0_real64,alfabs,alfabn, &
                        1.0_real64,1.0_real64,1.0_real64,1.0_real64, &
                        1.0_real64,1.0_real64,corrected_value(1), &
                        corrected_value(2),nested)
                    if(nested/=flow_slope_ok) local_status(grain,face)=1
                endif
                corrected(:,grain,face)=corrected_value
                normal_value=dot_product(corrected_value,topology%normal(:,face))
                select case(topology%role(face))
                case(scalar_live_face_internal)
                    if(normal_value<0.0_real64) then
                        donor(grain,face)=scalar_live_donor_neighbour
                        donor_cell(grain,face)=other
                    else
                        donor(grain,face)=scalar_live_donor_owner
                        donor_cell(grain,face)=own
                    endif
                case(scalar_live_face_trace_inlet,scalar_live_face_trace_outlet)
                    donor(grain,face)=scalar_live_donor_trace
                    donor_cell(grain,face)=own
                case(scalar_live_face_owner_outlet)
                    donor(grain,face)=scalar_live_donor_owner
                    donor_cell(grain,face)=own
                case default
                    donor(grain,face)=scalar_live_donor_none
                    donor_cell(grain,face)=0
                end select
            enddo
        enddo
        !$omp end do
        !$omp end parallel
        if(any(local_status/=0)) then
            status=unified_v75_face_failure;return
        endif
        trial%bed_slope_application_count=1

        trial%boundary_source=bed_boundary_v70_source_none
        do item=1,topology%nboundary
            face=topology%boundary_faces(item);mode=topology%boundary_mode(item)
            source=bed_boundary_v70_source_none
            if(hydraulic_outward(item)<0.0_real64.and. &
               mode==bed_boundary_v70_equilibrium) then
                source=bed_boundary_v70_source_equilibrium
                if(topology%role(face)/=scalar_live_face_trace_inlet) then
                    status=unified_v75_invalid_topology;return
                endif
            elseif(hydraulic_outward(item)>0.0_real64) then
                source=bed_boundary_v70_source_owner_export
                if(topology%role(face)/=scalar_live_face_owner_outlet) then
                    status=unified_v75_invalid_topology;return
                endif
            else
                if(topology%role(face)/=scalar_live_face_closed.and. &
                   hydraulic_outward(item)==0.0_real64) then
                    status=unified_v75_invalid_topology;return
                endif
            endif
            trial%boundary_source(:,item)=source
        enddo
        trial%boundary_owner_application_count=1

        allocate(cell_delta(nfrac,topology%ncell), &
                 inventory_fill(nfrac,topology%ncell), &
                 availability(nfrac,topology%ncell), &
                 boundary_transfer(nfrac),boundary_absolute(nfrac))
        trial%candidate%nfrac=nfrac;trial%candidate%ncell=topology%ncell
        trial%candidate%nface=topology%nface
        allocate(trial%candidate%bed_mass(nfrac,topology%ncell), &
                 trial%candidate%bulk_density(nfrac), &
                 trial%candidate%active_fraction(nfrac,topology%ncell), &
                 trial%candidate%boundary_signed(nfrac), &
                 trial%candidate%boundary_absolute(nfrac), &
                 trial%candidate%accepted_face_flux(nfrac,topology%nface), &
                 trial%candidate%accepted_cell_availability( &
                    nfrac,topology%ncell))
        call braids_scalar_flow_project_exner_v65( &
            topology%ncell,topology%nface,nfrac,topology%projection, &
            corrected,donor_cell,state%bed_mass,state%bulk_density, &
            topology%cell_area,dt_s,morphological_factor, &
            trial%candidate%active_fraction,inventory_fill,availability, &
            trial%face_flux,cell_delta,trial%candidate%bed_mass, &
            boundary_transfer,boundary_absolute, &
            trial%conservation_residual,nested,trial%exner_threads, &
            raw_active,raw_inventory)
        if(nested/=scalar_live_ok) then
            status=unified_v75_exner_failure;return
        endif
        allocate(cleaned(nfrac,topology%nface))
        call braids_cleanup_wetdry_face_flux_v71( &
            nfrac,topology%ncell,topology%nface,topology%owner, &
            topology%neighbour,topology%role,topology%wet,trial%face_flux, &
            cleaned,cleanup_threads,nested)
        if(nested/=direct_amr_v71_ok.or.any(cleaned/=trial%face_flux)) then
            status=unified_v75_exner_failure;return
        endif
        do item=1,topology%ncell
            if(topology%wet(item)==0.and. &
               any(trial%candidate%bed_mass(:,item)/=state%bed_mass(:,item))) then
                status=unified_v75_exner_failure;return
            endif
        enddo
        trial%candidate%bulk_density=state%bulk_density
        trial%candidate%boundary_signed=state%boundary_signed+boundary_transfer
        trial%candidate%boundary_absolute=state%boundary_absolute+boundary_absolute
        trial%candidate%accepted_step=state%accepted_step+1
        trial%candidate%time_s=state%time_s+dt_s
        trial%candidate%mesh_epoch=state%mesh_epoch
        trial%candidate%accepted_face_flux=trial%face_flux
        trial%candidate%accepted_cell_availability=availability
        do item=1,topology%ncell
            if(topology%wet(item)==0) &
                trial%candidate%accepted_cell_availability(:,item)=0.0_real64
        enddo
        call derive_fraction_v75(trial%candidate,derive_status)
        if(derive_status/=unified_v75_ok) then
            status=unified_v75_exner_failure;return
        endif
        trial%candidate%initialized=.true.
        trial%named_boundary_transfer=0.0_real64
        do item=1,topology%nboundary
            face=topology%boundary_faces(item)
            do grain=1,nfrac
                amount=trial%face_flux(grain,face)*topology%length_m(face)* &
                       dt_s*morphological_factor
                trial%named_boundary_transfer(grain,item)=amount
            enddo
        enddo
        if(any(abs(sum(trial%named_boundary_transfer,dim=2)-boundary_transfer)> &
           64.0_real64*epsilon(1.0_real64))) then
            status=unified_v75_exner_failure;return
        endif
        trial%wetdry_owner_application_count=1
        trial%active_fraction_application_count=raw_active
        trial%inventory_fill_application_count=raw_inventory
        trial%exner_application_count=1
        if(raw_active/=1.or.raw_inventory/=1) then
            status=unified_v75_exner_failure;return
        endif
        trial%pending=.true.;status=unified_v75_ok
    end subroutine braids_begin_unified_trial_v75

    pure logical function parent_matches_v75(state,trial)
        type(braids_unified_state_v75), intent(in) :: state
        type(braids_unified_trial_v75), intent(in) :: trial
        parent_matches_v75=state%initialized.and.trial%pending.and. &
            state%nfrac==trial%nfrac.and.state%ncell==trial%ncell.and. &
            state%nface==trial%nface.and. &
            state%accepted_step==trial%parent_step.and. &
            state%time_s==trial%parent_time.and. &
            state%mesh_epoch==trial%parent_epoch
        if(.not.parent_matches_v75) return
        parent_matches_v75=all(state%bed_mass==trial%parent_mass).and. &
            all(state%boundary_signed==trial%parent_signed).and. &
            all(state%boundary_absolute==trial%parent_absolute)
    end function parent_matches_v75

    subroutine braids_commit_unified_trial_v75(state,trial,status)
        type(braids_unified_state_v75), intent(inout) :: state
        type(braids_unified_trial_v75), intent(inout) :: trial
        integer, intent(out) :: status
        if(.not.trial%pending) then
            status=unified_v75_trial_not_pending
        elseif(.not.parent_matches_v75(state,trial)) then
            status=unified_v75_parent_mismatch
        else
            state=trial%candidate;trial%pending=.false.;status=unified_v75_ok
        endif
    end subroutine braids_commit_unified_trial_v75

    subroutine braids_rollback_unified_trial_v75(state,trial,status)
        type(braids_unified_state_v75), intent(in) :: state
        type(braids_unified_trial_v75), intent(inout) :: trial
        integer, intent(out) :: status
        if(.not.trial%pending) then
            status=unified_v75_trial_not_pending
        elseif(.not.parent_matches_v75(state,trial)) then
            status=unified_v75_parent_mismatch
        else
            trial%pending=.false.;status=unified_v75_ok
        endif
    end subroutine braids_rollback_unified_trial_v75

    subroutine braids_cancel_unified_trial_v75(state,trial,status)
        type(braids_unified_state_v75), intent(in) :: state
        type(braids_unified_trial_v75), intent(inout) :: trial
        integer, intent(out) :: status
        call braids_rollback_unified_trial_v75(state,trial,status)
    end subroutine braids_cancel_unified_trial_v75

    subroutine braids_export_unified_trial_v75( &
            trial,pending,consumed_capacity,face_taurat,face_flux,residual, &
            named_boundary,boundary_source,threads,counters,status)
        type(braids_unified_trial_v75), intent(in) :: trial
        logical, intent(out) :: pending
        real(real64), allocatable, intent(out) :: consumed_capacity(:,:,:)
        real(real64), allocatable, intent(out) :: face_taurat(:,:),face_flux(:,:)
        real(real64), allocatable, intent(out) :: residual(:),named_boundary(:,:)
        integer, allocatable, intent(out) :: boundary_source(:,:)
        integer, intent(out) :: threads(5),counters(9),status
        pending=trial%pending
        threads=[trial%producer_threads,trial%adaptation_threads, &
                 trial%face_threads,trial%slope_threads,trial%exner_threads]
        counters=[trial%producer_taurat_application_count, &
                  trial%adaptation_application_count, &
                  trial%wetdry_owner_application_count, &
                  trial%face_reconstruction_application_count, &
                  trial%bed_slope_application_count, &
                  trial%boundary_owner_application_count, &
                  trial%active_fraction_application_count, &
                  trial%inventory_fill_application_count, &
                  trial%exner_application_count]
        status=unified_v75_trial_not_pending
        if(.not.trial%pending) return
        allocate(consumed_capacity(2,trial%nfrac,trial%ncell), &
                 face_taurat(trial%nfrac,trial%nface), &
                 face_flux(trial%nfrac,trial%nface),residual(trial%nfrac), &
                 named_boundary(trial%nfrac,trial%nboundary), &
                 boundary_source(trial%nfrac,trial%nboundary))
        consumed_capacity=trial%consumed_cell_capacity
        face_taurat=trial%face_taurat;face_flux=trial%face_flux
        residual=trial%conservation_residual
        named_boundary=trial%named_boundary_transfer
        boundary_source=trial%boundary_source;status=unified_v75_ok
    end subroutine braids_export_unified_trial_v75

    subroutine braids_begin_unified_regrid_v75( &
            nfrac,state,old_topology,new_topology,nlink,link_old,link_new, &
            overlap_area,cancelled,trial,status)
        integer, intent(in) :: nfrac,nlink
        type(braids_unified_state_v75), intent(in) :: state
        type(braids_unified_topology_v75), intent(in) :: old_topology
        type(braids_unified_topology_v75), intent(in) :: new_topology
        integer, intent(in) :: link_old(nlink),link_new(nlink)
        real(real64), intent(in) :: overlap_area(nlink)
        logical, intent(in) :: cancelled
        type(braids_unified_regrid_trial_v75), intent(out) :: trial
        integer, intent(out) :: status
        real(real64), allocatable :: old_cover(:),new_cover(:),zero_signed(:)
        real(real64), allocatable :: zero_face(:,:),zero_cell(:,:)
        real(real64), allocatable :: new_mass(:,:)
        integer :: link,target,grain,restore_status
        real(real64) :: total,scale
        logical :: identity

        status=unified_v75_invalid_shape
        if(cancelled) then;status=unified_v75_cancelled;return;endif
        if(.not.state%initialized.or..not.old_topology%prepared.or. &
           .not.new_topology%prepared.or.nfrac/=state%nfrac.or.nlink<1.or. &
           state%ncell/=old_topology%ncell) return
        allocate(old_cover(old_topology%ncell),new_cover(new_topology%ncell))
        old_cover=0.0_real64;new_cover=0.0_real64
        do link=1,nlink
            if(link_old(link)<1.or.link_old(link)>old_topology%ncell.or. &
               link_new(link)<1.or.link_new(link)>new_topology%ncell.or. &
               .not.ieee_is_finite(overlap_area(link)).or. &
               overlap_area(link)<=0.0_real64) then
                status=unified_v75_transfer_failure;return
            endif
            old_cover(link_old(link))=old_cover(link_old(link))+overlap_area(link)
            new_cover(link_new(link))=new_cover(link_new(link))+overlap_area(link)
        enddo
        if(any(abs(old_cover-old_topology%cell_area)> &
               8192.0_real64*epsilon(1.0_real64)* &
               max(1.0_real64,old_topology%cell_area)).or. &
           any(abs(new_cover-new_topology%cell_area)> &
               8192.0_real64*epsilon(1.0_real64)* &
               max(1.0_real64,new_topology%cell_area))) then
            status=unified_v75_transfer_failure;return
        endif
        trial%nfrac=nfrac;trial%nold=old_topology%ncell
        trial%nnew=new_topology%ncell;trial%parent_step=state%accepted_step
        trial%parent_time=state%time_s;trial%parent_epoch=state%mesh_epoch
        allocate(trial%parent_mass(nfrac,old_topology%ncell), &
                 trial%residual(nfrac),new_mass(nfrac,new_topology%ncell), &
                 zero_face(nfrac,new_topology%nface), &
                 zero_cell(nfrac,new_topology%ncell))
        trial%parent_mass=state%bed_mass;new_mass=0.0_real64
        trial%observed_threads=1
        !$omp parallel default(shared) private(target,grain,link,total)
        !$omp single
        trial%observed_threads=omp_get_num_threads()
        !$omp end single
        !$omp do collapse(2) schedule(static,256)
        do target=1,new_topology%ncell
            do grain=1,nfrac
                total=0.0_real64
                do link=1,nlink
                    if(link_new(link)==target) total=total+ &
                        state%bed_mass(grain,link_old(link))*overlap_area(link)
                enddo
                new_mass(grain,target)=total/new_topology%cell_area(target)
            enddo
        enddo
        !$omp end do
        !$omp end parallel
        do grain=1,nfrac
            trial%residual(grain)= &
                sum(new_mass(grain,:)*new_topology%cell_area)- &
                sum(state%bed_mass(grain,:)*old_topology%cell_area)
            scale=max(1.0_real64, &
                sum(abs(new_mass(grain,:))*new_topology%cell_area), &
                sum(abs(state%bed_mass(grain,:))*old_topology%cell_area))
            if(abs(trial%residual(grain))> &
               16384.0_real64*epsilon(1.0_real64)*scale) then
                status=unified_v75_transfer_failure;return
            endif
        enddo
        identity=old_topology%ncell==new_topology%ncell.and.nlink==old_topology%ncell
        if(identity) then
            do link=1,nlink
                if(link_old(link)/=link_new(link)) identity=.false.
            enddo
        endif
        trial%wet_state_change_count=0
        if(identity) trial%wet_state_change_count=count( &
            old_topology%wet/=new_topology%wet)
        if(identity.and.trial%wet_state_change_count>0) then
            trial%transition_kind=direct_amr_v71_transition_wetdry
        elseif(new_topology%ncell>old_topology%ncell) then
            trial%transition_kind=direct_amr_v71_transition_refine
        elseif(new_topology%ncell<old_topology%ncell) then
            trial%transition_kind=direct_amr_v71_transition_coarsen
        else
            trial%transition_kind=direct_amr_v71_transition_identity
        endif
        zero_face=0.0_real64;zero_cell=0.0_real64
        call braids_restore_unified_state_v75( &
            nfrac,new_topology,state%accepted_step,state%time_s, &
            state%mesh_epoch+1,new_mass,state%bulk_density, &
            state%boundary_signed,state%boundary_absolute,zero_face,zero_cell, &
            trial%candidate,restore_status)
        if(restore_status/=unified_v75_ok) then
            status=unified_v75_transfer_failure;return
        endif
        trial%transfer_application_count=1
        trial%face_register_cleanup_count=1
        trial%availability_cleanup_count=1
        trial%exner_application_count=0
        trial%pending=.true.;status=unified_v75_ok
    end subroutine braids_begin_unified_regrid_v75

    pure logical function regrid_parent_matches_v75(state,trial)
        type(braids_unified_state_v75), intent(in) :: state
        type(braids_unified_regrid_trial_v75), intent(in) :: trial
        regrid_parent_matches_v75=state%initialized.and.trial%pending.and. &
            state%nfrac==trial%nfrac.and.state%ncell==trial%nold.and. &
            state%accepted_step==trial%parent_step.and. &
            state%time_s==trial%parent_time.and. &
            state%mesh_epoch==trial%parent_epoch
        if(regrid_parent_matches_v75) &
            regrid_parent_matches_v75=all(state%bed_mass==trial%parent_mass)
    end function regrid_parent_matches_v75

    subroutine braids_commit_unified_regrid_v75(state,trial,status)
        type(braids_unified_state_v75), intent(inout) :: state
        type(braids_unified_regrid_trial_v75), intent(inout) :: trial
        integer, intent(out) :: status
        if(.not.trial%pending) then
            status=unified_v75_trial_not_pending
        elseif(.not.regrid_parent_matches_v75(state,trial)) then
            status=unified_v75_parent_mismatch
        else
            state=trial%candidate;trial%pending=.false.;status=unified_v75_ok
        endif
    end subroutine braids_commit_unified_regrid_v75

    subroutine braids_rollback_unified_regrid_v75(state,trial,status)
        type(braids_unified_state_v75), intent(in) :: state
        type(braids_unified_regrid_trial_v75), intent(inout) :: trial
        integer, intent(out) :: status
        if(.not.trial%pending) then
            status=unified_v75_trial_not_pending
        elseif(.not.regrid_parent_matches_v75(state,trial)) then
            status=unified_v75_parent_mismatch
        else
            trial%pending=.false.;status=unified_v75_ok
        endif
    end subroutine braids_rollback_unified_regrid_v75

    subroutine braids_cancel_unified_regrid_v75(state,trial,status)
        type(braids_unified_state_v75), intent(in) :: state
        type(braids_unified_regrid_trial_v75), intent(inout) :: trial
        integer, intent(out) :: status
        call braids_rollback_unified_regrid_v75(state,trial,status)
    end subroutine braids_cancel_unified_regrid_v75

    subroutine braids_export_unified_regrid_v75( &
            trial,pending,transition_kind,wet_changes,threads,counters, &
            residual,status)
        type(braids_unified_regrid_trial_v75), intent(in) :: trial
        logical, intent(out) :: pending
        integer, intent(out) :: transition_kind,wet_changes,threads
        integer, intent(out) :: counters(4),status
        real(real64), allocatable, intent(out) :: residual(:)
        pending=trial%pending;transition_kind=trial%transition_kind
        wet_changes=trial%wet_state_change_count;threads=trial%observed_threads
        counters=[trial%transfer_application_count, &
                  trial%face_register_cleanup_count, &
                  trial%availability_cleanup_count, &
                  trial%exner_application_count]
        status=unified_v75_trial_not_pending
        if(.not.trial%pending) return
        allocate(residual(trial%nfrac));residual=trial%residual
        status=unified_v75_ok
    end subroutine braids_export_unified_regrid_v75

end module braids_unified_direct_vector_v75_module
