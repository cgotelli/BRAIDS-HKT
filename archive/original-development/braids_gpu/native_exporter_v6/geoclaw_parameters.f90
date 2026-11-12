! Minimal parameter bindings for the UNCHANGED native WAVE rpn2 source.
! Fixed Cartesian metric, no atmospheric pressure forcing in this executable.
! These are not substitutes for the numerical Riemann routines.
module geoclaw_module
 implicit none
 real(8) :: grav=9.81d0,dry_tolerance=1.d-8,rho=1000.d0
 ! Match the standard WAVE/GeoClaw stage preparation. The limiter acts only
 ! on momentum; it does not alter water or sediment inventories.
 real(8), parameter :: speed_limit=50.d0
 real(8), parameter :: earth_radius=6367500.d0,deg2rad=acos(-1.d0)/180.d0
end module
module amr_module
 implicit none
 integer, parameter :: mcapa=0
end module
module storm_module
 implicit none
 logical, parameter :: pressure_forcing=.false.
 integer, parameter :: pressure_index=1
end module
