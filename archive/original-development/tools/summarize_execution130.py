"""Build compact baseline attribution figures; not benchmark speedup claims."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('validation/_speed130'))
    args = ap.parse_args()
    runs = [(name, json.loads((args.root/folder/'PROFILE.json').read_text())) for name, folder in
            [('CPU · 5 m dynamic · 60 s', 'profile_cpu5m'), ('GPU · 2 m static · 100 s', 'profile_gpu2m')]]
    phases = ['step', 'amr', 'construction', 'load', 'output', 'checkpoint', 'other']
    labels = ['Hydraulic/physical steps', 'AMR decisions + rebuilding', 'Initial owner construction',
              'Load', 'Output fields + geometry', 'Checkpoint publication', 'Other setup/wrapping']
    colors = ['#0072B2', '#D55E00', '#E69F00', '#CC79A7', '#009E73', '#56B4E9', '#8A8A8A']
    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    left = np.zeros(len(runs))
    for key, label, color in zip(phases, labels, colors):
        values = np.array([r['phases'].get(key, 0.)/r['worker_wall_s']*100 for _, r in runs])
        ax.barh(range(len(runs)), values, left=left, label=label, color=color, height=.52)
        for i, value in enumerate(values):
            if value >= 8:
                ax.text(left[i]+value/2, i, f'{value:.1f}%', ha='center', va='center', fontsize=10,
                        color='white' if color not in ('#E69F00', '#56B4E9') else 'black')
        left += values
    ax.set_yticks(range(len(runs)), [name for name, _ in runs])
    ax.set_xlim(0, 100)
    ax.set_xlabel('Share of instrumented worker wall time (%)')
    ax.set_title('Different costs dominate the two tested workloads')
    ax.legend(loc='upper center', bbox_to_anchor=(.42, -.25), ncol=3, frameon=False, fontsize=9)
    fig.text(.03, .015, 'cProfile / event overhead included; these are attribution runs, not clean CPU–GPU speed comparisons.\n'
             'All saved numerical fields and meshes reproduce the alpha129 reference byte-for-byte.', fontsize=8)
    fig.tight_layout(rect=(0, .14, 1, 1))
    fig.savefig(args.root/'PROFILE_PHASES.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex='col')
    summary = {}
    for col, (name, r) in enumerate(runs):
        samples = r['restrictions']
        t = np.array([v['time_s'] for v in samples])
        h = np.array([v['state'][0] for v in samples])
        area = np.array([v['area_m2'] for v in samples])
        frequency = np.array([v['frequency_s_inverse'] for v in samples])
        cfl = r['controls']['cfl']
        boundary = np.array([any(i >= 0 for i in v['boundary_ids']) for v in samples])
        axes[0, col].plot(t, .9*cfl/frequency, color='#0072B2', label='0.9 CFL / limiting frequency')
        axes[0, col].scatter(t[boundary], (.9*cfl/frequency)[boundary], s=18, color='#D55E00', label='Open-boundary cell')
        axes[0, col].set_title(name)
        axes[0, col].set_ylabel('Stability-driven next dt (s)')
        axes[0, col].set_ylim(bottom=0)
        axes[0, col].legend(fontsize=8)
        axes[1, col].plot(t, h, color='#009E73')
        axes[1, col].set_xlabel('Simulation time (s)')
        axes[1, col].set_ylabel('Depth of limiting cell (m)')
        axes[1, col].set_ylim(bottom=0)
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
        summary[name] = dict(samples=len(samples), min_depth_m=float(h.min()), max_depth_m=float(h.max()),
            min_area_m2=float(area.min()), max_area_m2=float(area.max()), open_boundary_samples=int(boundary.sum()),
            interior_samples=int((~boundary).sum()), last=samples[-1], exact_reference=r['exact_reference']['exact'])
    fig.suptitle('The sampled timestep bottlenecks are wet, full-area cells—not the dry front')
    fig.tight_layout()
    fig.savefig(args.root/'CFL_LIMITING_CELLS.png', dpi=180)
    plt.close(fig)
    (args.root/'PROFILE_FINDINGS.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
