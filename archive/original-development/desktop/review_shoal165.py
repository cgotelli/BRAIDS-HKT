"""Quantify public shoal qualification and draw raw-cell release figures."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from braids_desktop.results import ResultsStore
from braids_gpu.atomic_output import write_json_atomic
BASE = ROOT / 'validation/_shoal165'
OUT = ROOT / 'validation/sediment/river_morphodynamics/reports/desktop_shoal165'
LAWS = ('mpm', 'ashida_michiue_generalized', 'wang_fredsoe')
LABELS = ('MPM', 'Ashida–Michiue', 'Wang–Fredsøe')


def norm(x):
    return dict(maximum_m=float(np.max(np.abs(x))), rms_m=float(np.sqrt(np.mean(x*x))))


def load(path):
    store = ResultsStore(path / 'run')
    status = json.loads((path / 'run/PROGRESS.json').read_text())
    assert status['time_s'] == 28. and status['status'] == 'complete', status
    return store, [store.read(i) for i in range(len(store.times))]


def budgets(path):
    rows = json.loads((path / 'run/DIAGNOSTICS.json').read_text())
    r = dict(water_residual_m3=max(abs(r['water_residual_m3']) for r in rows),
             solid_residual_kg=max(abs(v) for r in rows for v in r['class_residual_kg']),
             wall_integral=max(abs(v) for r in rows for v in r['wall_integrals']),
             minimum_stock_kg_m2=min(r['minimum_stock'] for r in rows))
    assert r['water_residual_m3'] < 1e-8 and r['solid_residual_kg'] < 1e-8
    assert r['wall_integral'] == 0 and r['minimum_stock_kg_m2'] >= 0
    return r


def field_map(fig, ax, frame, values, title, *, cmap='RdBu_r', limit=None):
    polygons = [np.asarray(p.exterior.coords) for p in frame.polygons]
    c = PolyCollection(polygons, array=values, cmap=cmap, edgecolors='none')
    if limit is not None:
        c.set_clim(*limit)
    ax.add_collection(c); ax.autoscale_view(); ax.set_aspect('equal')
    ax.set(xlabel='x (m)', ylabel='y (m)', title=title)
    fig.colorbar(c, ax=ax, shrink=.75)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = dict(scope='Current Desktop explicit SSPRK22; three public single-grain bedload laws; no banks/secondary/suspension',
                 historical_EH_reproduction=False, Delft3D_modified=False, cases={})
    datasets = []
    for law in LAWS:
        cpu_path = BASE / f'packaged3/{law}_cpu'
        gpu_path = BASE / f'packaged/{law}_gpu'
        cpu, cf = load(cpu_path); gpu, gf = load(gpu_path)
        assert np.array_equal(cpu.times, gpu.times)
        for field in ('q', 'z', 'mass', 'rock', 'area'):
            np.testing.assert_array_equal(cf[0].arrays[field], gf[0].arrays[field])
        np.testing.assert_array_equal(cf[0].mesh['vertices'], gf[0].mesh['vertices'])
        dz = cf[-1].z - cf[0].z
        delta = np.array([a.z-b.z for a,b in zip(cf,gf)])
        xy = np.array([p.centroid.coords[0] for p in cf[0].polygons])
        analytic = .11*np.exp(-((xy[:,0]-3.)**4 + ((xy[:,1]-1.)/.7)**4))
        r = dict(cpu=budgets(cpu_path), gpu=budgets(gpu_path),
                 cpu_gpu_bed_all_frames=norm(delta), initial_paired_fields_exact=True,
                 initial_cell_average_minus_historical_centre=norm(cf[0].z-analytic),
                 initial_dry_cells=int(np.sum(cf[0].q[0]<1e-6)),
                 minimum_depth_m=float(min(f.q[0].min() for f in cf)),
                 maximum_froude=float(max(f.values['froude'].max() for f in cf)),
                 final_scour_m=float(-dz.min()), final_deposition_m=float(dz.max()),
                 scour_xy_m=xy[dz.argmin()].tolist(), deposition_xy_m=xy[dz.argmax()].tolist(),
                 cpu_steps=json.loads((cpu_path/'run/PROGRESS.json').read_text())['accepted_steps'],
                 gpu_steps=json.loads((gpu_path/'run/PROGRESS.json').read_text())['accepted_steps'])
        r['strict_0p0001mm_backend_screen'] = r['cpu_gpu_bed_all_frames']['maximum_m'] < 1e-7
        # The user accepts small millimetre-scale front differences. Retain
        # the stricter groin screen separately; use 0.1 mm for this new shoal.
        assert r['minimum_depth_m'] >= 0 and r['cpu_gpu_bed_all_frames']['maximum_m'] < 1e-4
        maximum_index = np.unravel_index(np.argmax(np.abs(delta)),delta.shape)
        r['backend_maximum_location'] = dict(time_s=float(cpu.times[maximum_index[0]]),
            xy_m=xy[maximum_index[1]].tolist(),depth_m=float(cf[maximum_index[0]].q[0,maximum_index[1]]))
        assert np.max(np.abs(np.array([f.z-cf[0].z for f in cf[:25]]))) == 0
        audit['cases'][law] = r
        datasets.append((cpu,cf,gf,xy))
    ref, rf = load(BASE/'continuous/mpm_cpu')
    half, hf = load(BASE/'half/mpm_cpu')
    practical, pf = load(BASE/'practical/mpm_cpu')
    audit['timestep'] = {}
    for label, frames, dt in [('half',hf,.00125), ('four_times_larger',pf,.01)]:
        bd = np.array([a.z-b.z for a,b in zip(frames,rf)])
        hd = np.array([a.q[0]-b.q[0] for a,b in zip(frames,rf)])
        audit['timestep'][label] = dict(dt_s=dt, reference_dt_s=.0025,
            bed_all_frames=norm(bd), depth_all_frames=norm(hd), final_bed=norm(bd[-1]))
        assert norm(bd)['maximum_m'] < 1e-4
    audit['restart'] = {}
    for backend, path in [('cpu', BASE/'continuous/mpm_cpu'), ('gpu', BASE/'base/mpm_gpu')]:
        original = ResultsStore(path/'run').read(-1)
        resumed = ResultsStore(BASE/f'restart/mpm_{backend}/run').read(-1)
        exact = {k: bool(np.array_equal(original.arrays[k],resumed.arrays[k]))
                 for k in ('q','z','mass','water','bed_boundary')}
        assert all(exact.values()), exact
        audit['restart'][backend] = dict(from_s=26., to_s=28., exact=exact)
    _, ah = load(BASE/'half/ashida_michiue_generalized_cpu')
    ac = datasets[1][1]
    ad = np.array([a.z-b.z for a,b in zip(ah,ac)])
    audit['ashida_timestep'] = dict(reference_dt_s=.0025, shorter_dt_s=.00125,
                                  bed_all_frames=norm(ad), final_bed=norm(ad[-1]))
    assert norm(ad)['maximum_m'] < 1e-4
    audit['groin_release_regression'] = {}
    for law in LAWS:
        for backend in ('cpu','gpu'):
            path = BASE/f'groin/{law}_{backend}'
            a=ResultsStore(path/'run').read(-1)
            b=ResultsStore(ROOT/f'validation/_wave164/maintained/{law}_{backend}/run').read(-1)
            exact={k:bool(np.array_equal(a.arrays[k],b.arrays[k])) for k in ('q','z','mass','water','bed_boundary')}
            assert a.time_s == b.time_s == 60. and all(exact.values()),exact
            audit['groin_release_regression'][f'{law}_{backend}']=dict(exact=exact,budgets=budgets(path))
    audit['packaged_matches_checkout'] = {}
    for backend, prefix, raw in [('cpu','packaged3','continuous'),('gpu','packaged','base')]:
        a=ResultsStore(BASE/f'{prefix}/mpm_{backend}/run').read(-1)
        b=ResultsStore(BASE/f'{raw}/mpm_{backend}/run').read(-1)
        exact={k:bool(np.array_equal(a.arrays[k],b.arrays[k])) for k in ('q','z','mass','water','bed_boundary')}
        assert all(exact.values()),exact
        audit['packaged_matches_checkout'][backend]=exact
    # Candidate3 differs from2 only by extra generated native source records.
    manifests=[json.loads((BASE/f'candidate{k}/RUNTIME.json').read_text()) for k in (2,3)]
    delta=[name for name,value in manifests[1]['files'].items() if manifests[0]['files'].get(name)!=value]
    assert delta and all(name.startswith('precompiled/') for name in delta), delta
    audit['candidate3_vs2_extra_generated_source']=delta
    audit['candidate3_runtime_sha256']=hashlib.sha256((BASE/'candidate3/RUNTIME.json').read_bytes()).hexdigest()
    write_json_atomic(OUT/'AUDIT.json',audit)

    fig,axes=plt.subplots(3,3,figsize=(16,8),layout='constrained')
    for row,(label,(_,cf,gf,xy)) in enumerate(zip(LABELS,datasets)):
        limit=max(np.max(np.abs(cf[-1].z-cf[0].z)),np.max(np.abs(gf[-1].z-gf[0].z)))*1000
        field_map(fig,axes[row,0],cf[-1],1000*(cf[-1].z-cf[0].z),label+' CPU: bed change (mm)',limit=(-limit,limit))
        field_map(fig,axes[row,1],gf[-1],1000*(gf[-1].z-gf[0].z),label+' GPU: bed change (mm)',limit=(-limit,limit))
        field_map(fig,axes[row,2],cf[-1],1e6*(gf[-1].z-cf[-1].z),'GPU − CPU (µm)')
    fig.suptitle('Public shallow shoal, MF10: 24 s hydraulic warm-up + 4 s mobile bed\nRaw finite-volume cells, no visualization smoothing')
    fig.savefig(OUT/'SHOAL_BACKENDS.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    xy=datasets[0][3]
    for col,point in enumerate(((3.375,.375),(3.625,1.625))):
        i=int(np.argmin(np.sum((xy-point)**2,axis=1)))
        for label,store,frames in [('dt 0.010',practical,pf),('dt 0.0025',ref,rf),('dt 0.00125',half,hf)]:
            axes[0,col].plot(store.times,[(f.z[i]-frames[0].z[i])*1000 for f in frames],label=label)
            axes[1,col].plot(store.times,[f.q[0,i]*1000 for f in frames],label=label)
        axes[0,col].set(title=f'Cell centre {tuple(xy[i])} m',ylabel='Bed change (mm)',xlim=(24,28))
        axes[1,col].set(xlabel='Physical time (s)',ylabel='Water depth (mm)',xlim=(24,28))
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('MPM timestep check at the two historical near-bank hotspots\nCurrent public worker and DEM-averaged initial bed; not an EH reproduction')
    fig.savefig(OUT/'SHALLOW_CELLS.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for label,frames in [('CPU dt 0.0025',ac),('GPU dt 0.0025',datasets[1][2]),('CPU dt 0.00125',ah)]:
        i=int(np.argmin(np.sum((xy-[3.625,1.125])**2,axis=1)))
        ax.plot(ref.times,[(f.z[i]-frames[0].z[i])*1000 for f in frames],label=label)
    ax.set(xlim=(24,28),xlabel='Physical time (s)',ylabel='Bed change (mm)',
           title='Ashida–Michiue: shallow-cell sensitivity at (3.625, 1.125) m')
    ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'ASHIDA_SHALLOW_CELL.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(11,8),layout='constrained')
    field_map(fig,axes[0],rf[0],1000*rf[0].z,'Prepared initial bed (mm)',cmap='terrain')
    field_map(fig,axes[1],rf[24],1000*rf[24].q[0],'Depth at sediment activation, t=24 s (mm)',cmap='Blues')
    field_map(fig,axes[2],rf[24],rf[24].values['froude'],'Froude number at t=24 s',cmap='viridis',limit=(0,2))
    fig.suptitle('Current Desktop test conditions — 8 × 2 m, 0.25 m cells, Q=0.04 m³/s')
    fig.savefig(OUT/'SHOAL_HYDRAULICS.png',dpi=160);plt.close(fig)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
