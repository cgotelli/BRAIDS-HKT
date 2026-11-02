! Additive v92 Q3 producer seam with cell-wise transported-fluid density.
!
! This wrapper deliberately leaves the accepted v66/v67 constitutive
! implementations unchanged.  It only replaces v84's scalar profile water
! density with the parent v92 water-density field.  Viscosity and settling
! velocity retain the v84 array ABI.
module braids_q3_dynamic_fluid_producer_v92_module
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    use braids_direct_producer_adapter_v67_module, only: &
        braids_direct_producer_result_v67, braids_direct_route_controls_v67, &
        braids_bijker_input_v67, braids_1920_input_v67, &
        braids_run_bijker_adapter_v67, &
        braids_run_van_thiel_van_rijn_adapter_v67, &
        braids_run_xbeach_soulsby_van_rijn_adapter_v67
    implicit none
    private
    public :: braids_q3_dynamic_fluid_produce_v92

contains

    subroutine pack_result(result, raw, status)
        type(braids_direct_producer_result_v67), intent(in) :: result
        real(c_double), intent(out) :: raw(5)
        integer(c_int), intent(out) :: status

        raw(1:2) = result%raw_bed_mass
        raw(3:4) = result%raw_suspended_mass
        raw(5) = result%raw_equilibrium_mass_concentration
        if (result%producer_status == 0) then
            status = 0_c_int
        else
            status = int(result%producer_status, c_int)
        end if
    end subroutine pack_result

    subroutine braids_q3_dynamic_fluid_produce_v92(ncell, depth, velocity_x, &
            velocity_y, chezy, viscosity, settling_velocity, d50, d90, &
            solid_density, water_density, gravity, raw, status) &
            bind(C, name='braids_q3_dynamic_fluid_produce_v92')
        integer(c_int), value, intent(in) :: ncell
        real(c_double), intent(in) :: depth(ncell), velocity_x(ncell)
        real(c_double), intent(in) :: velocity_y(ncell), chezy(ncell)
        real(c_double), intent(in) :: viscosity(ncell)
        real(c_double), intent(in) :: settling_velocity(3, ncell)
        real(c_double), intent(in) :: d50(3), d90(3), solid_density(3)
        real(c_double), intent(in) :: water_density(ncell)
        real(c_double), value, intent(in) :: gravity
        real(c_double), intent(out) :: raw(5, 3, ncell)
        integer(c_int), intent(out) :: status(3, ncell)

        type(braids_direct_route_controls_v67) :: controls
        type(braids_bijker_input_v67) :: bijker
        type(braids_1920_input_v67) :: modern
        type(braids_direct_producer_result_v67) :: result
        integer :: cell

        raw = 0.0_c_double
        status = 1_c_int
        controls%route = 'tra_combine'
        controls%bed_factor = 1.0_c_double
        controls%suspended_factor = 1.0_c_double
        controls%active_fraction = 1.0_c_double
        controls%speed_epsilon = 1.0e-6_c_double
        controls%defer_surface_fraction_to_face = .true.

        do cell = 1, ncell
            bijker%depth = depth(cell)
            bijker%u = velocity_x(cell)
            bijker%v = velocity_y(cell)
            bijker%d50 = d50(1)
            bijker%d90 = d90(1)
            bijker%solid_density = solid_density(1)
            bijker%water_density = water_density(cell)
            bijker%gravity = gravity
            bijker%chezy = chezy(cell)
            bijker%settling_velocity = settling_velocity(1, cell)
            bijker%cal_bs = 5.0_c_double
            bijker%cal_bd = 4.0_c_double
            bijker%crit_cs = 0.4_c_double
            bijker%crit_cd = 0.05_c_double
            bijker%roughness_ks = -999.0_c_double
            bijker%formula_settling_velocity = -999.0_c_double
            bijker%porosity = 0.4_c_double
            bijker%formula_wave_period = 5.0_c_double
            bijker%fac_a = 0.0_c_double
            bijker%fac_u = 0.0_c_double
            bijker%eps_sl = 0.0_c_double
            call braids_run_bijker_adapter_v67(bijker, controls, result)
            call pack_result(result, raw(:, 1, cell), status(1, cell))

            modern%depth = depth(cell)
            modern%u = velocity_x(cell)
            modern%v = velocity_y(cell)
            modern%d15 = 0.0_c_double
            modern%d50 = d50(2)
            modern%d90 = d90(2)
            modern%solid_density = solid_density(2)
            modern%water_density = water_density(cell)
            modern%gravity = gravity
            modern%kinematic_viscosity = viscosity(cell)
            modern%chezy = chezy(cell)
            call braids_run_van_thiel_van_rijn_adapter_v67( &
                modern, controls, result)
            call pack_result(result, raw(:, 2, cell), status(2, cell))

            modern%d50 = d50(3)
            modern%d90 = d90(3)
            modern%solid_density = solid_density(3)
            call braids_run_xbeach_soulsby_van_rijn_adapter_v67( &
                modern, controls, result)
            call pack_result(result, raw(:, 3, cell), status(3, cell))
        end do
    end subroutine braids_q3_dynamic_fluid_produce_v92

end module braids_q3_dynamic_fluid_producer_v92_module
