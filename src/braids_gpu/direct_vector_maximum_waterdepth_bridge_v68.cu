// SPDX-License-Identifier: GPL-3.0-or-later
// Fused resident-CUDA transcription of FLOW dwnvel followed by EROSED's
// effective-current reconstruction.  Physical depth remains an independent
// output and is never replaced by the maximum observation depth.

extern "C" __global__ void direct_vector_maximum_waterdepth_v68(
    const long long npoint,
    const long long nlink,
    const int enabled,
    const double* __restrict__ cell_depth,
    const double* __restrict__ transport_u,
    const double* __restrict__ transport_v,
    const double* __restrict__ effective_speed_input,
    const int* __restrict__ active,
    const long long* __restrict__ support_offset,
    const double* __restrict__ support_depth,
    const int* __restrict__ support_active,
    const double direction_epsilon,
    const double effective_direction_epsilon,
    double* __restrict__ observation_depth,
    double* __restrict__ characteristic_u,
    double* __restrict__ characteristic_v,
    double* __restrict__ characteristic_speed,
    double* __restrict__ observation_height,
    double* __restrict__ effective_u,
    double* __restrict__ effective_v,
    double* __restrict__ effective_speed_output,
    int* __restrict__ status)
{
    const long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= npoint) return;

    observation_depth[point] = 0.0;
    characteristic_u[point] = 0.0;
    characteristic_v[point] = 0.0;
    characteristic_speed[point] = 0.0;
    observation_height[point] = 0.0;
    effective_u[point] = 0.0;
    effective_v[point] = 0.0;
    effective_speed_output[point] = 0.0;
    status[point] = 0;

    const int flag = active[point];
    const long long first = support_offset[point];
    const long long last = support_offset[point + 1];
    if ((enabled != 0 && enabled != 1) ||
        (flag != 0 && flag != 1) || first < 0 || last < first ||
        first > nlink || last > nlink ||
        !isfinite(direction_epsilon) || direction_epsilon <= 0.0 ||
        !isfinite(effective_direction_epsilon) ||
        effective_direction_epsilon <= 0.0) {
        status[point] = 1;
        return;
    }
    if (flag == 0) return;

    const double physical_depth = cell_depth[point];
    const double numerator_u = transport_u[point];
    const double numerator_v = transport_v[point];
    const double target_speed = effective_speed_input[point];
    if (!isfinite(physical_depth) || physical_depth <= 0.0 ||
        !isfinite(numerator_u) || !isfinite(numerator_v) ||
        !isfinite(target_speed) || target_speed < 0.0) {
        status[point] = 1;
        return;
    }

    double work_depth = physical_depth;
    if (enabled == 1) {
        for (long long item = first; item < last; ++item) {
            const int link_flag = support_active[item];
            const double link_depth = support_depth[item];
            if ((link_flag != 0 && link_flag != 1) ||
                !isfinite(link_depth) || link_depth < 0.0) {
                status[point] = 1;
                return;
            }
            if (link_flag == 1 && link_depth > work_depth) {
                work_depth = link_depth;
            }
        }
    }

    double uu = numerator_u / work_depth;
    double vv = numerator_v / work_depth;
    if (fabs(uu) < direction_epsilon) uu = 0.0;
    if (fabs(vv) < direction_epsilon) vv = 0.0;
    const double speed = hypot(uu, vv);
    const double denominator = speed + effective_direction_epsilon;
    const double eu = target_speed * uu / denominator;
    const double ev = target_speed * vv / denominator;
    const double es = hypot(eu, ev);
    if (!isfinite(speed) || !isfinite(eu) || !isfinite(ev) || !isfinite(es)) {
        status[point] = 1;
        return;
    }

    observation_depth[point] = work_depth;
    characteristic_u[point] = uu;
    characteristic_v[point] = vv;
    characteristic_speed[point] = speed;
    // dwnvel's unsuffixed 0.368 is binary32 before promotion to fp.
    observation_height[point] = 0.3680000007152557373046875 * work_depth;
    effective_u[point] = eu;
    effective_v[point] = ev;
    effective_speed_output[point] = es;
}
