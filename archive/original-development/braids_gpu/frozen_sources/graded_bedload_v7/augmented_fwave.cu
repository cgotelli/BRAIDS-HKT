// Validation-only FP64 port of the locally frozen GeoClaw augmented solver.
// Copyright (c) 1994-2018, Clawpack Developers. BSD-3-Clause;
// see the accompanying Clawpack-LICENSE.txt. This C++/CUDA port is a derivative.
// Sources: solver/clawpack-v5.14.0/riemann/src/rpn2_geoclaw.f and
// geoclaw_riemann_utils.f, riemann_aug_JCP + riemanntype.
// The production Fortran and all Delft3D sources remain unmodified.
//
// Scope is exactly the BRAIDS Cartesian 3-equation/3-wave interface: mcapa=0,
// pressure_forcing=false, maxiter=1. This is NOT a complete hydraulic update:
// reconstruction, normal rotation, bed-pressure quadrature, boundary traces,
// positivity/draining, timestepping, AMR, and sediment are outside this kernel.
// Compile without --use_fast_math and with --fmad=false for equivalence tests.
//
// RawKernel entry point:
// augmented_fwave_local(states, n, grav, drytol, outputs, status)
// states: float64 C-contiguous shape (8,n), SoA rows
//   hL, huL, hvL, bL, hR, huR, hvR, bR (local normal/tangent momenta).
// outputs: float64 C-contiguous shape (18,n), SoA rows
//   0:3 amdq, 3:6 apdq, 6:9 speeds,
//   9+3*wave+component: fwave(component,wave), zero-based indices.
// status: int32 shape (n); 0 success, bit 1 negative input depth was clamped
// as in rpn2, bit 2 nonfinite output, -1 invalid/nonfinite input or parameters.
// n: signed int64; grav,drytol: float64. Launch enough 1-D CUDA threads for n.

// NVRTC supplies CUDA math intrinsics. Do not include host math headers: the
// isolated CuPy/CUDA-wheel runtime deliberately has no host compiler SDK.

struct RiemannType {
    double hm, s1m, s2m;
    bool rare1, rare2;
};

__device__ __forceinline__ RiemannType braids_riemanntype(
    const double hL, const double hR, const double uL, const double uR,
    const int maxiter, const double drytol, const double g)
{
    RiemannType out;
    const double h_min = fmin(hR, hL);
    const double h_max = fmax(hR, hL);
    const double delu = uR - uL;
    double h0, F0, slope, gL, gR, dfdh, u1m, u2m;

    if (h_min <= drytol) {
        out.hm = 0.0;
        out.s1m = uR + uL - 2.0 * sqrt(g * hR) + 2.0 * sqrt(g * hL);
        out.s2m = uR + uL - 2.0 * sqrt(g * hR) + 2.0 * sqrt(g * hL);
        out.rare2 = hL <= 0.0;
        out.rare1 = !out.rare2;
    } else {
        const double F_min = delu + 2.0 * (sqrt(g * h_min) - sqrt(g * h_max));
        const double F_max = delu + (h_max - h_min) *
            sqrt(0.5 * g * (h_max + h_min) / (h_max * h_min));
        if (F_min > 0.0) {
            // Two rarefactions. The source's unused um is not returned.
            const double root = fmax(0.0, -delu + 2.0 * (sqrt(g * hL) + sqrt(g * hR)));
            out.hm = (1.0 / (16.0 * g)) * (root * root);
            out.s1m = uL + 2.0 * sqrt(g * hL) - 3.0 * sqrt(g * out.hm);
            out.s2m = uR - 2.0 * sqrt(g * hR) + 3.0 * sqrt(g * out.hm);
            out.rare1 = true;
            out.rare2 = true;
        } else if (F_max <= 0.0) {
            // Two shocks; same Newton iteration on sqrt(h), including count.
            h0 = h_max;
            for (int iter = 0; iter < maxiter; ++iter) {
                gL = sqrt(0.5 * g * (1.0 / h0 + 1.0 / hL));
                gR = sqrt(0.5 * g * (1.0 / h0 + 1.0 / hR));
                F0 = delu + (h0 - hL) * gL + (h0 - hR) * gR;
                dfdh = gL - g * (h0 - hL) / (4.0 * (h0 * h0) * gL) +
                    gR - g * (h0 - hR) / (4.0 * (h0 * h0) * gR);
                slope = 2.0 * sqrt(h0) * dfdh;
                const double root = sqrt(h0) - F0 / slope;
                h0 = root * root;
            }
            out.hm = h0;
            u1m = uL - (out.hm - hL) * sqrt((0.5 * g) * (1.0 / out.hm + 1.0 / hL));
            u2m = uR + (out.hm - hR) * sqrt((0.5 * g) * (1.0 / out.hm + 1.0 / hR));
            out.s1m = u1m - sqrt(g * out.hm);
            out.s2m = u2m + sqrt(g * out.hm);
            out.rare1 = false;
            out.rare2 = false;
        } else {
            // One shock, one rarefaction; same secant iteration.
            h0 = h_min;
            for (int iter = 0; iter < maxiter; ++iter) {
                F0 = delu + 2.0 * (sqrt(g * h0) - sqrt(g * h_max)) +
                    (h0 - h_min) * sqrt(0.5 * g * (1.0 / h0 + 1.0 / h_min));
                slope = (F_max - F0) / (h_max - h_min);
                h0 = h0 - F0 / slope;
            }
            out.hm = h0;
            if (hL > hR) {
                out.s1m = uL + 2.0 * sqrt(g * hL) - 3.0 * sqrt(g * out.hm);
                out.s2m = uL + 2.0 * sqrt(g * hL) - sqrt(g * out.hm);
                out.rare1 = true;
                out.rare2 = false;
            } else {
                out.s2m = uR - 2.0 * sqrt(g * hR) + 3.0 * sqrt(g * out.hm);
                out.s1m = uR - 2.0 * sqrt(g * hR) + sqrt(g * out.hm);
                out.rare1 = false;
                out.rare2 = true;
            }
        }
    }
    return out;
}

__device__ __forceinline__ double braids_determinant(const double a[3][3])
{
    const double det1 = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]);
    const double det2 = a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]);
    const double det3 = a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    return det1 - det2 + det3;
}

__device__ __forceinline__ void braids_aug_JCP(
    const int maxiter, const double hL, const double hR,
    const double huL, const double huR, const double bL, const double bR,
    const double uL, const double uR, const double vL, const double vR,
    const double phiL, const double phiR, double sE1, double sE2,
    const double drytol, const double g, double sw[3], double fw[3][3])
{
    // fw is indexed [wave][component]; r and A remain [row][column].
    double A[3][3], r[3][3], lambda[3], delta[3], beta[3];
    const double delh = hR - hL;
    const double delhu = huR - huL;
    const double delphi = phiR - phiL;
    // Atmospheric pressures are both zero in the BRAIDS parameter module.
    const double delb = bR - bL;
    double delnorm = delh * delh + delphi * delphi;
    const RiemannType type = braids_riemanntype(hL, hR, uL, uR, 1, drytol, g);
    const double hm = type.hm, s1m = type.s1m, s2m = type.s2m;
    lambda[0] = fmin(sE1, s2m);
    lambda[2] = fmax(sE2, s1m);
    sE1 = lambda[0];
    sE2 = lambda[2];
    lambda[1] = 0.0;
    const double hstarHLL = fmax((huL - huR + sE2 * hR - sE1 * hL) / (sE2 - sE1), 0.0);

    // These exact switches are false in the frozen Fortran, not a GPU change.
    const bool rarecorrectortest = false;
    bool rarecorrector = false;
    if (rarecorrectortest) {
        const double sdelta = lambda[2] - lambda[0];
        double raremin = 0.5;
        const double raremax = 0.9;
        if (type.rare1 && sE1 * s1m < 0.0) raremin = 0.2;
        if (type.rare2 && sE2 * s2m < 0.0) raremin = 0.2;
        if (type.rare1 || type.rare2) {
            const double rare1st = 3.0 * (sqrt(g * hL) - sqrt(g * hm));
            const double rare2st = 3.0 * (sqrt(g * hR) - sqrt(g * hm));
            if (fmax(rare1st, rare2st) > raremin * sdelta &&
                fmax(rare1st, rare2st) < raremax * sdelta) {
                rarecorrector = true;
                if (rare1st > rare2st) lambda[1] = s1m;
                else if (rare2st > rare1st) lambda[1] = s2m;
                else lambda[1] = 0.5 * (s1m + s2m);
            }
        }
        if (hstarHLL < fmin(hL, hR) / 5.0) rarecorrector = false;
    }
    for (int wave = 0; wave < 3; ++wave) {
        r[0][wave] = 1.0;
        r[1][wave] = lambda[wave];
        r[2][wave] = lambda[wave] * lambda[wave];
    }
    if (!rarecorrector) {
        lambda[1] = 0.5 * (lambda[0] + lambda[2]);
        r[0][1] = 0.0; r[1][1] = 0.0; r[2][1] = 1.0;
    }

    const double criticaltol = fmax(drytol * g, 1.0e-6);
    const double criticaltol_2 = sqrt(criticaltol);
    double deldelh = -delb;
    double deldelphi = -0.5 * (hR + hL) * (g * delb);
    double hLstar = hL, hRstar = hR, uLstar = uL, uRstar = uR;
    double huLstar = uLstar * hLstar, huRstar = uRstar * hRstar;
    const double convergencetol = 1.0e-6;
    for (int iter = 0; iter < maxiter; ++iter) {
        if (fmin(hLstar, hRstar) < drytol && rarecorrector) {
            rarecorrector = false;
            hLstar = hL; hRstar = hR; uLstar = uL; uRstar = uR;
            huLstar = uLstar * hLstar; huRstar = uRstar * hRstar;
            lambda[1] = 0.5 * (lambda[0] + lambda[2]);
            r[0][1] = 0.0; r[1][1] = 0.0; r[2][1] = 1.0;
        }
        const double hbar = fmax(0.5 * (hLstar + hRstar), 0.0);
        const double usum = uLstar + uRstar;
        const double s1s2bar = 0.25 * (usum * usum) - g * hbar;
        const double s1s2tilde = fmax(0.0, uLstar * uRstar) - g * hbar;
        bool sonic = false;
        if (fabs(s1s2bar) <= criticaltol) sonic = true;
        else if (s1s2bar * s1s2tilde <= criticaltol * criticaltol) sonic = true;
        else if (s1s2bar * sE1 * sE2 <= criticaltol * criticaltol) sonic = true;
        else if (fmin(fabs(sE1), fabs(sE2)) < criticaltol_2) sonic = true;
        else if (sE1 < criticaltol_2 && s1m > -criticaltol_2) sonic = true;
        else if (sE2 > -criticaltol_2 && s2m < criticaltol_2) sonic = true;
        else if ((uL + sqrt(g * hL)) * (uR + sqrt(g * hR)) < 0.0) sonic = true;
        else if ((uL - sqrt(g * hL)) * (uR - sqrt(g * hR)) < 0.0) sonic = true;

        if (sonic) deldelh = -delb;
        else deldelh = delb * g * hbar / s1s2bar;
        if (sE1 < -criticaltol && sE2 > criticaltol) {
            deldelh = fmin(deldelh, hstarHLL * (sE2 - sE1) / sE2);
            deldelh = fmax(deldelh, hstarHLL * (sE2 - sE1) / sE1);
        } else if (sE1 >= criticaltol) {
            deldelh = fmin(deldelh, hstarHLL * (sE2 - sE1) / sE1);
            deldelh = fmax(deldelh, -hL);
        } else if (sE2 <= -criticaltol) {
            deldelh = fmin(deldelh, hR);
            deldelh = fmax(deldelh, hstarHLL * (sE2 - sE1) / sE2);
        }
        if (sonic) deldelphi = -g * hbar * delb;
        else deldelphi = -delb * g * hbar * s1s2tilde / s1s2bar;
        deldelphi = fmin(deldelphi, g * fmax(-hLstar * delb, -hRstar * delb));
        deldelphi = fmax(deldelphi, g * fmin(-hLstar * delb, -hRstar * delb));
        delta[0] = delh - deldelh;
        delta[1] = delhu;
        delta[2] = delphi - deldelphi;
        const double determinant = braids_determinant(r);
        for (int k = 0; k < 3; ++k) {
            for (int wave = 0; wave < 3; ++wave)
                for (int component = 0; component < 3; ++component)
                    A[component][wave] = r[component][wave];
            A[0][k] = delta[0]; A[1][k] = delta[1]; A[2][k] = delta[2];
            beta[k] = braids_determinant(A) / determinant;
        }
        if (fabs(delta[0] * delta[0] + delta[2] * delta[2] - delnorm) < convergencetol) break;
        delnorm = delta[0] * delta[0] + delta[2] * delta[2];
        hLstar = hL; hRstar = hR; uLstar = uL; uRstar = uR;
        huLstar = uLstar * hLstar; huRstar = uRstar * hRstar;
        for (int wave = 0; wave < 3; ++wave) {
            if (lambda[wave] < 0.0) {
                hLstar = hLstar + beta[wave] * r[0][wave];
                huLstar = huLstar + beta[wave] * r[1][wave];
            }
        }
        for (int wave = 2; wave >= 0; --wave) {
            if (lambda[wave] > 0.0) {
                hRstar = hRstar - beta[wave] * r[0][wave];
                huRstar = huRstar - beta[wave] * r[1][wave];
            }
        }
        if (hLstar > drytol) uLstar = huLstar / hLstar;
        else { hLstar = fmax(hLstar, 0.0); uLstar = 0.0; }
        if (hRstar > drytol) uRstar = huRstar / hRstar;
        else { hRstar = fmax(hRstar, 0.0); uRstar = 0.0; }
    }
    for (int wave = 0; wave < 3; ++wave) {
        sw[wave] = lambda[wave];
        fw[wave][0] = beta[wave] * r[1][wave];
        fw[wave][1] = beta[wave] * r[2][wave];
        fw[wave][2] = beta[wave] * r[1][wave];
    }
    fw[0][2] = fw[0][2] * vL;
    fw[2][2] = fw[2][2] * vR;
    fw[1][2] = 0.0;
    const double hustar_interface = huL + fw[0][0];
    if (hustar_interface <= 0.0)
        fw[0][2] = fw[0][2] + (hR * uR * vR - hL * uL * vL - fw[0][2] - fw[2][2]);
    else
        fw[2][2] = fw[2][2] + (hR * uR * vR - hL * uL * vL - fw[0][2] - fw[2][2]);
}

extern "C" __global__ void augmented_fwave_local(
    const double* states, const long long n, const double grav, const double drytol,
    double* outputs, int* status)
{
    const long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    for (int row = 0; row < 18; ++row) outputs[(long long)row * n + i] = 0.0;
    status[i] = 0;
    bool finite = isfinite(grav) && isfinite(drytol) && grav > 0.0 && drytol >= 0.0;
    for (int row = 0; row < 8; ++row) finite = finite && isfinite(states[(long long)row * n + i]);
    if (!finite) { status[i] = -1; return; }
    double hL = states[i], huL = states[n + i], hvL = states[2 * n + i], bL = states[3 * n + i];
    double hR = states[4 * n + i], huR = states[5 * n + i], hvR = states[6 * n + i], bR = states[7 * n + i];
    if (hL < 0.0) { hL = 0.0; huL = 0.0; hvL = 0.0; status[i] |= 1; }
    if (hR < 0.0) { hR = 0.0; huR = 0.0; hvR = 0.0; status[i] |= 1; }
    if (hL <= drytol && hR <= drytol) return;

    double uL, uR, vL, vR, phiL, phiR;
    if (hR > drytol) {
        uR = huR / hR; vR = hvR / hR;
        phiR = 0.5 * grav * (hR * hR) + (huR * huR) / hR;
    } else { hR = 0.0; huR = 0.0; hvR = 0.0; uR = 0.0; vR = 0.0; phiR = 0.0; }
    if (hL > drytol) {
        uL = huL / hL; vL = hvL / hL;
        phiL = 0.5 * grav * (hL * hL) + (huL * huL) / hL;
    } else { hL = 0.0; huL = 0.0; hvL = 0.0; uL = 0.0; vL = 0.0; phiL = 0.0; }
    double wall[3] = {1.0, 1.0, 1.0};
    if (hR <= drytol) {
        const RiemannType type = braids_riemanntype(hL, hL, uL, -uL, 1, drytol, grav);
        const double hstartest = fmax(hL, type.hm);
        if (hstartest + bL < bR) {
            wall[1] = 0.0; wall[2] = 0.0;
            hR = hL; huR = -huL; bR = bL; phiR = phiL; uR = -uL; vR = vL;
        } else if (hL + bL < bR) bR = hL + bL;
    } else if (hL <= drytol) {
        const RiemannType type = braids_riemanntype(hR, hR, -uR, uR, 1, drytol, grav);
        const double hstartest = fmax(hR, type.hm);
        if (hstartest + bR < bL) {
            wall[0] = 0.0; wall[1] = 0.0;
            hL = hR; huL = -huR; bL = bR; phiL = phiR; uL = -uR; vL = vR;
        } else if (hR + bR < bL) bL = hR + bR;
    }
    const double sL = uL - sqrt(grav * hL), sR = uR + sqrt(grav * hR);
    const double uhat = (sqrt(grav * hL) * uL + sqrt(grav * hR) * uR) /
        (sqrt(grav * hR) + sqrt(grav * hL));
    const double chat = sqrt(grav * 0.5 * (hR + hL));
    const double sE1 = fmin(sL, uhat - chat), sE2 = fmax(sR, uhat + chat);
    double sw[3], fw[3][3], am[3] = {0.0, 0.0, 0.0}, ap[3] = {0.0, 0.0, 0.0};
    braids_aug_JCP(1, hL, hR, huL, huR, bL, bR, uL, uR, vL, vR, phiL, phiR,
                   sE1, sE2, drytol, grav, sw, fw);
    for (int wave = 0; wave < 3; ++wave) {
        sw[wave] = sw[wave] * wall[wave];
        outputs[(long long)(6 + wave) * n + i] = sw[wave];
        for (int component = 0; component < 3; ++component) {
            fw[wave][component] = fw[wave][component] * wall[wave];
            outputs[(long long)(9 + 3 * wave + component) * n + i] = fw[wave][component];
            if (sw[wave] < -1.0e-14) am[component] = am[component] + fw[wave][component];
            else if (sw[wave] > 1.0e-14) ap[component] = ap[component] + fw[wave][component];
            else {
                am[component] = am[component] + 0.5 * fw[wave][component];
                ap[component] = ap[component] + 0.5 * fw[wave][component];
            }
        }
    }
    for (int component = 0; component < 3; ++component) {
        outputs[(long long)component * n + i] = am[component];
        outputs[(long long)(3 + component) * n + i] = ap[component];
    }
    for (int row = 0; row < 18; ++row)
        if (!isfinite(outputs[(long long)row * n + i])) status[i] |= 2;
}
