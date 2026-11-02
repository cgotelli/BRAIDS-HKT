! Q4/v96 runtime-geometry ABI for the native augmented-f-wave evaluator.
!
! The frozen v95 kernel already accepts a cellwise bed and density field at
! every invocation; v95 intentionally froze those through its Python
! execution contract.  This additive ABI exposes those inputs to the v96
! atomic owner without changing either the GeoClaw Riemann routine or v95.
! ``density_gradient_enabled`` owns the complete accepted-density gradient:
! spatial water EOS is therefore retained when the sediment DensIn increment
! is disabled by the outer v96 method selection.
module braids_coupled_geometry_v96_module
 use iso_c_binding
 use braids_densin_residual_v95_module, only: braids_densin_full_rate_v95
 implicit none
contains

 subroutine braids_coupled_full_rate_v96( &
  nc,nf,nadj,q,bed,owner_zero,neighbour_zero,normal,face_length,face_distance, &
  cell_area,cell_face_offset_zero,cell_faces_zero,cell_face_sign,accepted_density, &
  reference_density,gravity,dry,density_gradient_enabled,threads,base_rate, &
  density_rate,full_rate,face_acceleration,cell_acceleration,face_water_flux, &
  riemann_calls,density_source_calls,status) &
  bind(C,name='braids_coupled_full_rate_v96')
  integer(c_int),value :: nc,nf,nadj,density_gradient_enabled,threads
  real(c_double),intent(in) :: q(nc,3),bed(nc),normal(nf,2),face_length(nf)
  real(c_double),intent(in) :: face_distance(nf),cell_area(nc),accepted_density(nc)
  real(c_double),value :: reference_density,gravity,dry
  integer(c_int),intent(in) :: owner_zero(nf),neighbour_zero(nf)
  integer(c_int),intent(in) :: cell_face_offset_zero(nc+1),cell_faces_zero(nadj)
  integer(c_int),intent(in) :: cell_face_sign(nadj)
  real(c_double),intent(out) :: base_rate(nc,3),density_rate(nc,2),full_rate(nc,3)
  real(c_double),intent(out) :: face_acceleration(nf),cell_acceleration(nc,2)
  real(c_double),intent(out) :: face_water_flux(nf)
  integer(c_int),intent(out) :: riemann_calls,density_source_calls,status

  call braids_densin_full_rate_v95( &
   nc,nf,nadj,q,bed,owner_zero,neighbour_zero,normal,face_length,face_distance, &
   cell_area,cell_face_offset_zero,cell_faces_zero,cell_face_sign,accepted_density, &
   reference_density,gravity,dry,density_gradient_enabled,threads,base_rate, &
   density_rate,full_rate,face_acceleration,cell_acceleration,face_water_flux, &
   riemann_calls,density_source_calls,status)
 end subroutine braids_coupled_full_rate_v96
end module braids_coupled_geometry_v96_module
