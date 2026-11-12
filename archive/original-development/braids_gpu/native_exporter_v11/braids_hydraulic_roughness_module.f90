! Shared Delft3D-2DH hydraulic roughness conversion used by BRAIDS
! hydrodynamic friction and sediment forcing.
!
! The three public modes follow FLOW's ROUMET=C/M/W branches in TAUBOT.
! They convert the user parameter to one local Chezy coefficient before any
! caller forms shear or a semi-implicit momentum source.  Keeping that
! conversion in one pure operator prevents hydraulic friction, sediment shear
! and the GPU implementation from silently using different roughness physics.
module braids_hydraulic_roughness_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    integer, parameter, public :: braids_roughness_chezy = 1
    integer, parameter, public :: braids_roughness_manning = 2
    integer, parameter, public :: braids_roughness_white_colebrook = 3
    ! FLOW hydraulic roughness uses the global physical constant 0.41.  This
    ! must remain distinct from the 0.40 hard-coded by its Soulsby skin closure.
    ! INIPHY assigns the unsuffixed literal 0.41 to fp.  The locked Windows
    ! reference therefore promotes the binary32-rounded value to its working
    ! precision; retain that detail instead of silently substituting 0.41d0.
    real(8), parameter :: von_karman = real(0.41,8)
    real(8), parameter :: euler_number = 2.7182818284590452353602874713526625d0

    public :: braids_roughness_state
    public :: braids_roughness_shear
    public :: braids_roughness_friction_rate

contains

    pure subroutine braids_roughness_state(mode,depth,parameter,gravity,minimum_work_depth, &
                                           waqua_white_colebrook,chezy,z0,drag,valid)
        ! Convert one ROUMET parameter to the local depth-dependent state.
        !
        ! parameter units:
        !   Chezy             m**0.5 / s
        !   Manning           s / m**(1/3)
        !   White-Colebrook   Nikuradse ks [m]
        !
        ! z0 is the log-law-equivalent roughness height used by FLOW EROSED.
        ! drag is the dimensionless g/C**2 coefficient, so callers can form
        ! tau=rho*drag*|u|**2 without another mode-specific branch.
        integer, intent(in) :: mode
        real(8), intent(in) :: depth,parameter,gravity,minimum_work_depth
        logical, intent(in) :: waqua_white_colebrook
        real(8), intent(out) :: chezy,z0,drag
        logical, intent(out) :: valid
        real(8) :: work_depth,ratio,x,negative_exp

        chezy=0.d0;z0=0.d0;drag=0.d0
        valid=all(ieee_is_finite([depth,parameter,gravity,minimum_work_depth]))
        if(.not.valid) return
        valid=depth>=0.d0.and.parameter>0.d0.and.gravity>0.d0.and.minimum_work_depth>=0.d0
        if(.not.valid.or.depth==0.d0) return

        work_depth=max(depth,minimum_work_depth)
        select case(mode)
        case(braids_roughness_chezy)
            chezy=parameter
        case(braids_roughness_manning)
            chezy=work_depth**(1.d0/6.d0)/parameter
        case(braids_roughness_white_colebrook)
            if(waqua_white_colebrook) then
                ! FLOW ChzK2D branch: H/ks is floored at 0.5.
                ratio=max(.5d0,work_depth/parameter)
                chezy=18.d0*log10(12.d0*ratio)
            else
                ! Default FLOW branch, with ks/30 as the roughness length.
                chezy=sqrt(gravity)*log(1.d0+work_depth/(euler_number*parameter/30.d0))/von_karman
            endif
        case default
            valid=.false.;return
        end select
        if(.not.ieee_is_finite(chezy).or.chezy<=0.d0) then
            valid=.false.;chezy=0.d0;return
        endif

        x=von_karman*chezy/sqrt(gravity)
        ! Stable inverse of C=sqrt(g)/kappa*log(1+h/(e*z0)); this avoids
        ! overflowing EXP for a smooth/high-C state.
        negative_exp=exp(-x)
        z0=max(tiny(1.d0),work_depth*negative_exp/(euler_number*(1.d0-negative_exp)))
        drag=gravity/(chezy*chezy)
        valid=all(ieee_is_finite([chezy,z0,drag])).and.z0>0.d0.and.drag>0.d0
        if(.not.valid) then
            chezy=0.d0;z0=0.d0;drag=0.d0
        endif
    end subroutine braids_roughness_state

    pure subroutine braids_roughness_shear(mode,depth,speed,parameter,density,gravity, &
                                           minimum_work_depth,waqua_white_colebrook, &
                                           shear,chezy,z0,valid)
        integer, intent(in) :: mode
        real(8), intent(in) :: depth,speed,parameter,density,gravity,minimum_work_depth
        logical, intent(in) :: waqua_white_colebrook
        real(8), intent(out) :: shear,chezy,z0
        logical, intent(out) :: valid
        real(8) :: drag

        shear=0.d0
        call braids_roughness_state(mode,depth,parameter,gravity,minimum_work_depth, &
                                    waqua_white_colebrook,chezy,z0,drag,valid)
        if(.not.valid) return
        valid=ieee_is_finite(speed).and.ieee_is_finite(density).and.speed>=0.d0.and.density>0.d0
        if(.not.valid) then
            shear=0.d0;return
        endif
        shear=density*drag*speed*speed
        valid=ieee_is_finite(shear).and.shear>=0.d0
    end subroutine braids_roughness_shear

    pure subroutine braids_roughness_friction_rate(mode,depth,speed,parameter,gravity, &
                                                   minimum_work_depth,waqua_white_colebrook, &
                                                   rate,valid)
        ! Semi-implicit source coefficient q_new=q_old/(1+dt*rate).
        ! The physical depth stays in the momentum denominator while only the
        ! Chezy work depth is floored, matching BRAIDS' qualified shallow-flow
        ! Manning envelope and extending it consistently to C and W.
        integer, intent(in) :: mode
        real(8), intent(in) :: depth,speed,parameter,gravity,minimum_work_depth
        logical, intent(in) :: waqua_white_colebrook
        real(8), intent(out) :: rate
        logical, intent(out) :: valid
        real(8) :: chezy,z0,drag

        rate=0.d0
        call braids_roughness_state(mode,depth,parameter,gravity,minimum_work_depth, &
                                    waqua_white_colebrook,chezy,z0,drag,valid)
        if(.not.valid) return
        valid=ieee_is_finite(speed).and.speed>=0.d0
        if(.not.valid) return
        rate=drag*speed/depth
        valid=ieee_is_finite(rate).and.rate>=0.d0
    end subroutine braids_roughness_friction_rate

end module braids_hydraulic_roughness_module
