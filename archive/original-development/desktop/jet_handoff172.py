"""Locate the remaining fixed-bed jet mismatch; do not infer causality from norms."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'desktop'), str(ROOT)]
from review_hydraulics162 import load
from stage_wall172 import sha, save

OUT = ROOT / 'validation/sediment/river_morphodynamics/reports/desktop_delivery172'


def rms(a):
    return float(np.sqrt(np.mean(a*a)))


def main():
    run = ROOT / 'validation/_delivery172/package/groin_mpm_cpu/full'
    native = ROOT / 'validation/_hydraulics162/mobile/delft_mpm'
    d, metadata = load(run, native)
    xy = d['xy']; times = d['store'].times
    j = int(np.flatnonzero(times == 40)[0])
    assert np.array_equal(d['bz'][:j+1], np.broadcast_to(d['bz'][0], d['bz'][:j+1].shape))
    assert np.array_equal(d['nz'][:j+1], np.broadcast_to(d['nz'][0], d['nz'][:j+1].shape))
    sections = []
    for x in (2.25, 4.25, 5.75, 6.25, 7.75, 8.25, 10.25, 12.25, 16.25):
        mask = xy[:, 0] == x
        order = np.argsort(xy[mask, 1]); y = xy[mask, 1][order]
        b, n = d['bu'][j][:, mask][:, order], d['nu'][j][:, mask][:, order]
        sections.append(dict(x_m=x, y_m=y.tolist(), braids_u=b[0].tolist(), flow_u=n[0].tolist(),
                             braids_v=b[1].tolist(), flow_v=n[1].tolist(),
                             braids_h=d['bh'][j, mask][order].tolist(), flow_h=d['nh'][j, mask][order].tolist(),
                             u_rms_m_s=rms(b[0]-n[0]), v_rms_m_s=rms(b[1]-n[1]),
                             h_rms_mm=1000*rms(d['bh'][j, mask]-d['nh'][j, mask]),
                             braids_peak_y_m=float(y[b[0].argmax()]), flow_peak_y_m=float(y[n[0].argmax()]),
                             braids_peak_u_m_s=float(b[0].max()), flow_peak_u_m_s=float(n[0].max())))
    tip = int(np.flatnonzero(np.all(xy == [8.25, 3.25], axis=1))[0])
    history = []
    for k, t in enumerate(times):
        if not 20 <= t <= 40:
            continue
        b, n = d['bu'][k, :, tip], d['nu'][k, :, tip]
        history.append(dict(time_s=float(t), braids_u=float(b[0]), braids_v=float(b[1]),
                            flow_u=float(n[0]), flow_v=float(n[1]),
                            braids_angle_deg=float(np.degrees(np.arctan2(b[1], b[0]))),
                            flow_angle_deg=float(np.degrees(np.arctan2(n[1], n[0]))),
                            braids_h=float(d['bh'][k, tip]), flow_h=float(d['nh'][k, tip])))
    region = (xy[:, 0] >= 6) & (xy[:, 0] <= 12) & (xy[:, 1] >= 3)
    k = int(np.flatnonzero(times == 30)[0])
    unsteady = dict(window_s=[30, 40], region='6<=x<=12, y>=3 m',
                    braids_velocity_change_rms_m_s=rms(d['bu'][j][:, region]-d['bu'][k][:, region]),
                    flow_velocity_change_rms_m_s=rms(d['nu'][j][:, region]-d['nu'][k][:, region]))
    source = ROOT / 'resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute'
    source_notes = {
        'mom_fls.f90': 'Lines 205-222 and 276-289: limited staggered transverse velocity, face-mask-dependent discharge averaging, then guarded velocity advection including -u times discharge divergence.',
        'uzd.f90': 'Staggered velocity solve and surface-pressure gradient; not the reflected cell-centred Riemann problem used by BRAIDS.'}
    result = dict(runtime_sha256=sha(ROOT / 'desktop/runtime/RUNTIME.json'), time_s=40.,
                  native_execution=metadata['native_execution'], initial_bed_difference_m=metadata['initial_bed_max_m'],
                  initial_depth_difference_m=metadata['initial_depth_max_m'], native_viscosity_m2_s=metadata['native_horizontal_viscosity_max_m2_s'],
                  sections=sections, tip_history=history, unsteadiness=unsteady, morphology_before_comparison=False,
                  source_notes=source_notes, native_sources={name:sha(source/name) for name in source_notes},
                  verified_native_inputs_sha256=sha(native/'INPUTS.json'), verified_native_execution_sha256=sha(native/'execution.json'),
                  braids_index_sha256=sha(run/'frames/INDEX.json'), script_sha256=sha(__file__),
                  caveats=['FLOW U1/V1 averaged to cell centres, not identical discrete velocity DOFs.',
                           'These are transient fixed-bed states, not equilibrium or experimental truth.',
                           'Spatial localization and code differences do not prove a single causal term.'],
                  closed_causal_gate=False)
    save(OUT/'JET_HANDOFF.json', result)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6), layout='constrained')
    for col, x in enumerate((4.25, 7.75, 8.25, 10.25)):
        s = next(r for r in sections if r['x_m'] == x)
        for row, field in enumerate(('u','v')):
            ax = axes[row, col]
            ax.plot(s['y_m'], s['braids_'+field], 'o-', ms=3, label='BRAIDS preview27')
            ax.plot(s['y_m'], s['flow_'+field], 's--', ms=3, label='FLOW 2DH')
            ax.axhline(0, color='gray', lw=.6); ax.grid(alpha=.2)
            ax.set(xlabel='Cross-channel y (m)', ylabel=field+' (m/s)', title=f'x = {x:g} m')
            if col == 0: ax.legend(fontsize=8)
    fig.suptitle('Remaining jet discrepancy at 40 s, before any bed change\nGroin occupies x = 8–10 m, y = 0–3 m; raw cell-centre observables')
    fig.savefig(OUT/'JET_SECTIONS.png', dpi=160); plt.close(fig)
    print(json.dumps(dict(tip=history[-1], unsteadiness=unsteady, sections=[{k:v for k,v in r.items() if k.endswith('rms_m_s') or k in ('x_m','braids_peak_y_m','flow_peak_y_m')} for r in sections]), indent=2))


if __name__ == '__main__':
    main()
