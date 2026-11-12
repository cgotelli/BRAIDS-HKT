"""Test a conservative face-visible transverse bed-slope correction.

This is an isolated CPU diagnostic.  It adds only the part of the physical
Jiang--Cao gravitational deflection that a centred cell gradient misses at an
internal face.  Affine bed fields receive exactly zero correction.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / "BRAIDS_JiangCao_Experiment228"
RUNTIME = PROJECT / "runtime"
OUT = ROOT / "validation/_transverse230/face_slope"
MANIFEST = "63b330057b70c20f8066ebeded4b113495fec82e97c734b5bf97a9a9973572f9"


def main() -> None:
    if OUT.exists():
        raise FileExistsError("Preserve the existing face-slope diagnostic")
    sys.path[:0] = [str(RUNTIME), str(ROOT)]
    os.environ.update(
        OMP_NUM_THREADS="8",
        OMP_DYNAMIC="FALSE",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        QT_QPA_PLATFORM="offscreen",
    )
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install

    verify()
    install(json.loads((RUNTIME / "PRECOMPILED.json").read_text()), RUNTIME)
    if hashlib.sha256((RUNTIME / "RUNTIME.json").read_bytes()).hexdigest() != MANIFEST:
        raise ValueError("Frozen runtime differs")

    from braids_core import river_worker
    from braids_core.sediment.jiang_cao_wetdry import JiangCaoShoreline

    original_rate = JiangCaoShoreline.rate

    def face_visible_rate(self, state, dt, time, inlet_concentration):
        rate, mixture_flux, solid_flux = original_rate(
            self, state, dt, time, inlet_concentration
        )
        xp = self.xp
        geometry = self.mesh.g
        owner = geometry["owner"]
        neighbour = geometry["neighbour"]
        internal = neighbour >= 0
        jj = xp.maximum(neighbour, 0)
        wet = state[0] > self.dry
        overlap = (
            xp.minimum(state[0, owner] + state[5, owner], state[0, jj] + state[5, jj])
            - xp.maximum(state[5, owner], state[5, jj])
        )
        active = internal & wet[owner] & wet[jj] & (overlap > self.dry)

        # Cell values reproduce the maintained constitutive law.  The pair
        # correction below only supplies the connector derivative hidden from
        # a centred gradient by a two-cell mode.
        _, capacity = self.target(state)
        gradient = self.mesh.capacity_gradient(state[5], state[:3])
        faces = xp.flatnonzero(active)
        o = owner[faces]
        j = neighbour[faces]
        dx = geometry["xy"][0, j] - geometry["xy"][0, o]
        dy = geometry["xy"][1, j] - geometry["xy"][1, o]
        distance2 = dx * dx + dy * dy
        gx = .5 * (gradient[0, o] + gradient[0, j])
        gy = .5 * (gradient[1, o] + gradient[1, j])
        missing = (state[5, j] - state[5, o] - gx * dx - gy * dy) / distance2
        corrected_gx = gx + missing * dx
        corrected_gy = gy + missing * dy

        mx = .5 * (state[1, o] + state[1, j])
        my = .5 * (state[2, o] + state[2, j])
        speed = xp.hypot(mx, my)
        tx = mx / xp.where(speed > 0., speed, 1.)
        ty = my / xp.where(speed > 0., speed, 1.)
        theta = .5 * (capacity.values[3, o] + capacity.values[3, j])
        critical = .5 * (capacity.values[4, o] + capacity.values[4, j])
        factor = -1.5 * xp.sqrt(critical / xp.where(theta > 0., theta, 1.))
        factor = xp.where((speed > 0.) & (theta > 0.), factor, 0.)

        def relative(normal_gradient):
            ratio = factor * normal_gradient
            norm = xp.hypot(1., ratio)
            return (
                (mx - ratio * my) / norm - mx,
                (my + ratio * mx) / norm - my,
            )

        base_left = -ty * gx + tx * gy
        corrected_left = -ty * corrected_gx + tx * corrected_gy
        base_x, base_y = relative(base_left)
        corrected_x, corrected_y = relative(corrected_left)
        nx = geometry["normal"][0, faces]
        ny = geometry["normal"][1, faces]
        missing_discharge = (
            (corrected_x - base_x) * nx + (corrected_y - base_y) * ny
        ) * geometry["length"][faces]
        concentration = state[3] / xp.where(state[0] > 0., state[0], 1.)
        donor = xp.where(missing_discharge >= 0., o, j)
        candidate = solid_flux.copy()
        candidate[faces] += missing_discharge * concentration[donor]

        incoming = (
            inlet_concentration(time)
            if callable(inlet_concentration)
            else inlet_concentration
        )
        incoming = xp.ascontiguousarray(
            xp.broadcast_to(xp.asarray(incoming, dtype=xp.float64), (self.nf,))
        )
        _, _, candidate, theta_face = self.front.flux(
            state[0],
            state[3],
            mixture_flux,
            candidate,
            dt,
            packing=self.B,
            inlet_concentration=incoming,
        )
        divergence = xp.empty(self.n)
        self.engine.launch(
            "jc_mobile_divergence",
            self.n,
            self.n,
            geometry["offset"],
            geometry["faces"],
            geometry["owner"],
            geometry["area"],
            candidate,
            divergence,
        )
        rate[3] = divergence
        self.phase_minimum_theta = min(
            self.phase_minimum_theta, float(xp.min(theta_face))
        )
        return rate, mixture_flux, candidate

    try:
        JiangCaoShoreline.rate = face_visible_rate
        result = river_worker.run(
            PROJECT / "prepared/experiment229",
            OUT / "run60",
            until=60.0,
            checkpoint_dir=OUT / "checkpoints",
        )
    finally:
        JiangCaoShoreline.rate = original_rate

    (OUT / "RESULT.json").write_text(
        json.dumps(
            {
                "scope": "isolated CPU diagnostic; not an admitted model or Desktop change",
                "frozen_runtime_sha256": MANIFEST,
                "method": "face-visible connector correction to gravitational deflection",
                "result": result,
            },
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
