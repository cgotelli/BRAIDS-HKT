! Classic Delft3D4 FLOW 2DH prognostic spiral-intensity primitives.
!
! These routines implement the local parts of the Equili=N constituent
! equation without importing Delft3D source.  The locked, unmodified source
! anchors are secrhs.f90 (source/sink), difu.f90 (old/new-volume weighting and
! implicit sink), secbou.f90 (open-boundary equilibrium), and chkdry.f90
! (zero intensity in dry cells).
module braids_flow2dh_prognostic_secondary_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private

    public :: braids_flow2dh_prognostic_reaction
    public :: braids_flow2dh_prognostic_reaction_mass
    public :: braids_flow2dh_prognostic_boundary
    public :: braids_flow2dh_prognostic_dry_reset

contains

    pure subroutine braids_flow2dh_prognostic_reaction(active,depth_old,depth_new, &
                                                        intensity_old,source_coefficient, &
                                                        sink_coefficient,dt, &
                                                        conserved_new,intensity_new, &
                                                        source_increment,sink_increment,valid)
        ! One classic FLOW source/sink update with no transport:
        !
        !   h_new I_new = [h_old I_old + dt h_old S] / [1 + dt K]
        !
        ! SOUR is explicit at the old volume and SINK is implicit at the new
        ! volume in difu.f90.  I and h*I are signed; no positivity limiter is
        ! permitted.  A non-active/dry target follows chkdry and becomes zero.
        logical, intent(in) :: active
        real(8), intent(in) :: depth_old,depth_new,intensity_old
        real(8), intent(in) :: source_coefficient,sink_coefficient,dt
        real(8), intent(out) :: conserved_new,intensity_new
        real(8), intent(out) :: source_increment,sink_increment
        logical, intent(out) :: valid
        real(8) :: conserved_old

        conserved_old=depth_old*intensity_old
        call braids_flow2dh_prognostic_reaction_mass(active,depth_old,depth_new, &
            conserved_old,source_coefficient,sink_coefficient,dt,conserved_new, &
            intensity_new,source_increment,sink_increment,valid)
    end subroutine braids_flow2dh_prognostic_reaction

    pure subroutine braids_flow2dh_prognostic_reaction_mass(active,depth_old,depth_new, &
                                                             conserved_advected, &
                                                             source_coefficient, &
                                                             sink_coefficient,dt, &
                                                             conserved_new,intensity_new, &
                                                             source_increment,sink_increment, &
                                                             valid)
        ! Full transported FLOW update.  M_adv is the conservative h*I state
        ! after advection; it is not reconstructed from the accepted old
        ! depth.  The explicit source alone owns h_old, while the implicit
        ! sink coefficient was formed with h_new.
        logical, intent(in) :: active
        real(8), intent(in) :: depth_old,depth_new,conserved_advected
        real(8), intent(in) :: source_coefficient,sink_coefficient,dt
        real(8), intent(out) :: conserved_new,intensity_new
        real(8), intent(out) :: source_increment,sink_increment
        logical, intent(out) :: valid
        real(8) :: pre_sink,denominator

        conserved_new=0.d0;intensity_new=0.d0
        source_increment=0.d0;sink_increment=0.d0
        valid=all(ieee_is_finite([depth_old,depth_new,conserved_advected, &
                                  source_coefficient,sink_coefficient,dt]))
        if(.not.valid) return
        valid=depth_old>=0.d0.and.depth_new>=0.d0.and. &
              sink_coefficient>=0.d0.and.dt>=0.d0
        if(.not.valid) return
        if(.not.active.or.depth_new<=0.d0) return

        source_increment=dt*depth_old*source_coefficient
        pre_sink=conserved_advected+source_increment
        denominator=1.d0+dt*sink_coefficient
        valid=denominator>0.d0.and.all(ieee_is_finite([source_increment, &
              pre_sink,denominator]))
        if(.not.valid) then
            source_increment=0.d0;return
        endif

        conserved_new=pre_sink/denominator
        sink_increment=conserved_new-pre_sink
        intensity_new=conserved_new/depth_new
        valid=all(ieee_is_finite([conserved_new,intensity_new, &
                                  source_increment,sink_increment]))
        if(.not.valid) then
            conserved_new=0.d0;intensity_new=0.d0
            source_increment=0.d0;sink_increment=0.d0
        endif
    end subroutine braids_flow2dh_prognostic_reaction_mass

    pure subroutine braids_flow2dh_prognostic_boundary(active,depth_old,depth_new, &
                                                        dryflc,epsilon_value, &
                                                        source_coefficient, &
                                                        sink_coefficient, &
                                                        intensity_boundary, &
                                                        conserved_boundary, &
                                                        defined,valid)
        ! secbou.f90 applies this adjacent-interior equilibrium at both ends
        ! of every open row.  It does not branch on the instantaneous flux
        ! direction.  The caller owns the mapping from an unstructured open
        ! face to its adjacent interior cell.
        logical, intent(in) :: active
        real(8), intent(in) :: depth_old,depth_new,dryflc,epsilon_value
        real(8), intent(in) :: source_coefficient,sink_coefficient
        real(8), intent(out) :: intensity_boundary,conserved_boundary
        logical, intent(out) :: defined,valid
        real(8) :: denominator,epsh

        intensity_boundary=0.d0;conserved_boundary=0.d0;defined=.false.
        valid=all(ieee_is_finite([depth_old,depth_new,dryflc,epsilon_value, &
                                  source_coefficient,sink_coefficient]))
        if(.not.valid) return
        valid=depth_old>=0.d0.and.depth_new>=0.d0.and.dryflc>=0.d0.and. &
              epsilon_value>=0.d0.and.sink_coefficient>=0.d0
        if(.not.valid.or..not.active) return

        denominator=sink_coefficient*depth_new
        epsh=.5d0*epsilon_value*dryflc
        if(abs(denominator)<=epsh) return
        intensity_boundary=source_coefficient*depth_old/denominator
        conserved_boundary=depth_new*intensity_boundary
        valid=all(ieee_is_finite([intensity_boundary,conserved_boundary]))
        if(.not.valid) then
            intensity_boundary=0.d0;conserved_boundary=0.d0
            return
        endif
        defined=.true.
    end subroutine braids_flow2dh_prognostic_boundary

    pure subroutine braids_flow2dh_prognostic_dry_reset(wet,conserved_old, &
                                                         conserved_new,increment,valid)
        ! chkdry.f90 sets the secondary constituent to zero at a temporarily
        ! dry point.  Return the signed budget increment explicitly.
        logical, intent(in) :: wet
        real(8), intent(in) :: conserved_old
        real(8), intent(out) :: conserved_new,increment
        logical, intent(out) :: valid

        conserved_new=conserved_old;increment=0.d0
        valid=ieee_is_finite(conserved_old)
        if(.not.valid) then
            conserved_new=0.d0;return
        endif
        if(.not.wet) then
            conserved_new=0.d0
            increment=-conserved_old
        endif
    end subroutine braids_flow2dh_prognostic_dry_reset

end module braids_flow2dh_prognostic_secondary_module
