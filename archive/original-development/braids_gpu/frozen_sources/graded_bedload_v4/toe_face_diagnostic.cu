// Each thread owns one grain/physical face.  The input is the *limited* Exner
// rate actually used by bed_gather, in kg/(m s), not raw bedload capacity.
extern "C" __global__ void toe_face_capture(
    const double* flux, const double* length, const int* owner,
    const int* neighbour, const int* is_toe_face, const int* candidate_bank,
    const int* candidate_toe, const long long nf, const long long count,
    const double dt, const double weight, double* step_net, int* failure) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const long long f = i % nf;
  if (!is_toe_face[f]) return;
  const int bank = candidate_bank[f], toe = candidate_toe[f];
  const int orientation = owner[f] == bank && neighbour[f] == toe ? 1 :
                          neighbour[f] == bank && owner[f] == toe ? -1 : 0;
  if (!orientation || !isfinite(flux[i]) || !isfinite(length[f]) ||
      length[f] <= 0.0 || !isfinite(dt) || dt < 0.0 ||
      !isfinite(weight) || weight < 0.0) {
    atomicExch(failure, 1);
    return;
  }
  const double next = step_net[i] + weight * orientation * dt * length[f] * flux[i];
  if (!isfinite(next)) atomicExch(failure, 1);
  else step_net[i] = next;
}

extern "C" __global__ void toe_face_validate(
    const double* step_net, const double* cumulative_signed,
    const double* cumulative_gross, const long long count, int* failure) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const double net = step_net[i];
  const double signed_next = cumulative_signed[i] + net;
  const double gross_next = cumulative_gross[i] + fabs(net);
  if (!isfinite(net) || !isfinite(signed_next) || !isfinite(gross_next) ||
      cumulative_gross[i] < 0.0 || gross_next < 0.0)
    atomicExch(failure, 1);
}

extern "C" __global__ void toe_face_commit(
    const double* step_net, double* cumulative_signed,
    double* cumulative_gross, const long long count, const int* global_status) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count || global_status[0] != 0) return;
  const double net = step_net[i];
  cumulative_signed[i] += net;
  cumulative_gross[i] += fabs(net);
}
