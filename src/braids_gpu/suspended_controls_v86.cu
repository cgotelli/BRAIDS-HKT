// Actual-CUDA operators for the additive Q4 suspended controls (v86).
//
// These kernels do not own suspended advection, diffusion, settling, or
// exchange.  They consume the authenticated v78/v81/v84 operator evidence
// once and select the Delft3D4 FLOW morphological bed-balance owner.

extern "C" __global__
void iopsus_diameter_v86(
    const int count,
    const int formula_code,
    const int iopsus,
    const double silt_diameter_m,
    const double* fixed_dss_m,
    const double* d50_m,
    const double* transport_stage,
    const double* d10_m,
    const double* mobility_psi,
    double* dss_m,
    int* status)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count) return;
    int error = 0;
    double result = fixed_dss_m[i];
    const double d50 = d50_m[i];
    if (!(isfinite(result) && result > 0.0 &&
          isfinite(d50) && d50 > 0.0)) error = 1;
    if (!(iopsus == 0 || iopsus == 1)) error = 2;
    if (!(formula_code == -1 || formula_code == -2 ||
          formula_code == -4)) error = 3;
    if (error == 0 && iopsus == 1 && formula_code == -1) {
        const double ta = transport_stage[i];
        if (!(isfinite(ta) && ta >= 0.0)) {
            error = 4;
        } else if (ta <= 1.0) {
            result = 0.64 * d50;
        } else if (ta >= 25.0) {
            result = d50;
        } else {
            result = d50 * (1.0 + 0.015 * (ta - 25.0));
        }
    }
    if (error == 0 && iopsus == 1 &&
        (formula_code == -2 || formula_code == -4)) {
        const double d10 = d10_m[i];
        const double psi = mobility_psi[i];
        if (!(isfinite(d10) && d10 > 0.0 && d10 <= d50 &&
              isfinite(psi) && psi >= 0.0 &&
              isfinite(silt_diameter_m) && silt_diameter_m > 0.0)) {
            error = 5;
        } else {
            result = psi < 550.0
                ? (1.0 + 0.0006 * (d50 / d10 - 1.0) *
                   (psi - 550.0)) * d50
                : d50;
            result = fmax(d10, result);
            if (d50 < silt_diameter_m) result = d50;
            result = fmax(result, 0.5 * silt_diameter_m);
        }
    }
    if (error == 0 && !(isfinite(result) && result > 0.0)) error = 6;
    dss_m[i] = error == 0 ? result : 0.0;
    status[i] = error;
}

extern "C" __global__
void suspended_bed_balance_v86(
    const int fraction_count,
    const int cell_count,
    const int face_count,
    const int neglect_entrainment,
    const int combine_route,
    const double dt_s,
    const int* routes,
    const int* cell_face_offset,
    const int* cell_faces,
    const int* cell_face_sign,
    const double* cell_area_m2,
    const double* face_length_m,
    const double* advective_face_flux_kg_m_s,
    const double* diffusive_face_mass_owner_out_kg,
    const double* transferred_to_suspension_kg_m2,
    double* exchange_bed_delta_kg_m2,
    double* flux_divergence_bed_delta_kg_m2,
    double* selected_bed_delta_kg_m2,
    int* status)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int count = fraction_count * cell_count;
    if (index >= count) return;
    const int grain = index / cell_count;
    const int cell = index - grain * cell_count;
    if (routes[grain] != combine_route) {
        exchange_bed_delta_kg_m2[index] = 0.0;
        flux_divergence_bed_delta_kg_m2[index] = 0.0;
        selected_bed_delta_kg_m2[index] = 0.0;
        status[index] = 0;
        return;
    }
    const double area = cell_area_m2[cell];
    double flux_delta = 0.0;
    int error = !(isfinite(area) && area > 0.0 &&
                  isfinite(dt_s) && dt_s > 0.0);
    for (int slot = cell_face_offset[cell];
         slot < cell_face_offset[cell + 1] && error == 0; ++slot) {
        const int face = cell_faces[slot];
        if (face < 0 || face >= face_count) {
            error = 2;
            break;
        }
        const double integral =
            advective_face_flux_kg_m_s[grain * face_count + face] *
            face_length_m[face] * dt_s +
            diffusive_face_mass_owner_out_kg[grain * face_count + face];
        if (!isfinite(integral)) {
            error = 3;
            break;
        }
        flux_delta -= ((double) cell_face_sign[slot]) * integral;
    }
    flux_delta /= area;
    const double exchange = -transferred_to_suspension_kg_m2[index];
    if (!(isfinite(exchange) && isfinite(flux_delta))) error = 4;
    exchange_bed_delta_kg_m2[index] = error == 0 ? exchange : 0.0;
    flux_divergence_bed_delta_kg_m2[index] = error == 0 ? flux_delta : 0.0;
    selected_bed_delta_kg_m2[index] = error == 0
        ? (neglect_entrainment ? flux_delta : exchange) : 0.0;
    status[index] = error;
}

extern "C" __global__
void suspended_boundary_ledger_v86(
    const int fraction_count,
    const int boundary_count,
    const int face_count,
    const double dt_s,
    const int* boundary_index,
    const double* face_length_m,
    const double* advective_face_flux_kg_m_s,
    const double* diffusive_face_mass_owner_out_kg,
    double* boundary_mass_in_kg,
    int* status)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int count = fraction_count * boundary_count;
    if (index >= count) return;
    const int grain = index / boundary_count;
    const int boundary = index - grain * boundary_count;
    double total = 0.0;
    int error = !(isfinite(dt_s) && dt_s > 0.0);
    for (int face = 0; face < face_count && error == 0; ++face) {
        if (boundary_index[face] == boundary) {
            const double integral =
                advective_face_flux_kg_m_s[grain * face_count + face] *
                face_length_m[face] * dt_s +
                diffusive_face_mass_owner_out_kg[grain * face_count + face];
            if (!isfinite(integral)) {
                error = 2;
                break;
            }
            total -= integral;
        }
    }
    boundary_mass_in_kg[index] = error == 0 ? total : 0.0;
    status[index] = error;
}
