"""Stage-resolved scour attribution; raw cells, locked native reference."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'desktop/runtime'), str(ROOT), str(ROOT/'desktop')]
from braids_desktop.results import ResultsStore

BASE = ROOT/'validation/_scour166/v3'
OUT = ROOT/'validation/sediment/river_morphodynamics/reports/desktop_scour166'
LAWS = ('mpm', 'ashida_michiue_generalized', 'wang_fredsoe')
LABELS = ('MPM', 'Ashida–Michiue', 'Wang–Fredsøe')
CASES = ('BRAIDS', 'BRAIDS_FACE_DISCHARGE', 'NATIVE_DEPTH', 'NATIVE_PRIMARY',
         'NATIVE_SPEED', 'NATIVE_DIRECTION', 'NATIVE_CHARACTERISTIC', 'NATIVE_ALL_FORCING')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(values):
    return dict(rms=float(np.sqrt(np.mean(values**2))), maximum=float(abs(values).max()))


def bed_rate(flux, a):
    """Outward solid-volume flux per face length to bed rate, mm/s."""
    amount = flux*a['length']; total = np.zeros(len(a['area']))
    np.add.at(total, a['owner'], -amount)
    internal = a['neighbour'] >= 0
    np.add.at(total, a['neighbour'][internal], amount[internal])
    return total/a['area']/.6*1000.


def sign_upwind(capacity, a):
    left = np.sum(capacity[:, a['owner']]*a['normal'], axis=0)
    right = np.sum(capacity[:, np.maximum(a['neighbour'], 0)]*a['normal'], axis=0)
    flux = np.where((left > 0) & (right > 0), left,
                    np.where((left < 0) & (right < 0), right, .5*(left+right)))
    flux[a['neighbour'] < 0] = 0.  # Local interior comparison only.
    return flux


def native_flux(native, a):
    vertical = abs(a['normal'][0]) > .5
    mx = np.rint(a['mid'][0]/.5+np.where(vertical, 0., .5)).astype(int)
    my = np.rint(a['mid'][1]/.5+np.where(vertical, .5, 0.)).astype(int)
    return np.where(vertical, native['face_u'][mx, my]*a['normal'][0],
                    native['face_v'][mx, my]*a['normal'][1])


def verify_native(folder):
    inputs = json.loads((folder/'INPUTS.json').read_text())
    for name, checksum in inputs['inputs'].items():
        assert (folder/name).resolve().is_relative_to(folder.resolve())
        assert sha(folder/name) == checksum, name
    run = json.loads((folder/'execution.json').read_text())
    assert run['success']
    assert run['executable_sha256'] == '09f451b673fcf855ae8f07d819686e7a429ab9382ff6da372702aa3a46583623'
    for row in run['outputs']:
        assert sha(folder/row['name']) == row['sha256']
    return dict(inputs_sha256=sha(folder/'INPUTS.json'), execution_sha256=sha(folder/'execution.json'),
                executable_sha256=run['executable_sha256'])


def source_audit():
    names = ('mom_fls.f90', 'upwhu.f90', 'dwnvel.f90', 'erosed.f90', 'upwbed.f90',
             'adjust_bedload.f90', 'rdscour.f90', 'shearx.f90', 'tranb2.f90')
    result = {}
    with zipfile.ZipFile(ROOT/'resources/Delft3D-main.zip') as archive:
        members = archive.namelist()
        for name in names:
            matches = [p for p in members if p.endswith('/'+name)]
            assert len(matches) == 1, (name, matches)
            path = ROOT/'resources'/matches[0]
            assert path.read_bytes() == archive.read(matches[0]), str(path)
            result[str(path.relative_to(ROOT))] = sha(path)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}; raw = {}
    for law in LAWS:
        folder = BASE/f'{law}_cpu'
        n = np.load(folder/'NATIVE.npz')
        arrays = {k: np.load(folder/f'{k}.npz') for k in CASES}
        a = arrays['BRAIDS']; xy = a['xy']; tip = np.argmin(np.sum((xy-[8.25, 3.25])**2, axis=1))
        region = (xy[:, 0] >= 6) & (xy[:, 0] <= 16)
        local_faces = (a['mid'][0] >= 6) & (a['mid'][0] <= 16) & (a['neighbour'] >= 0)
        assert np.array_equal(a['area'], np.full(len(xy), .25))
        nf = native_flux(n, a); control = sign_upwind(n['capacity'], a)
        face_error = float(abs(nf[local_faces]-control[local_faces]).max())
        assert face_error < 2e-9, face_error
        rates = {k: bed_rate(v['class_flux'][0], v) for k, v in arrays.items()}
        rates['FLOW'] = bed_rate(nf, a)
        rates['NATIVE_CAPACITY_UPWIND'] = bed_rate(control, a)
        exact = arrays['NATIVE_ALL_FORCING']['bedcell'][3:5]/2650.
        relative = float(np.linalg.norm(exact-n['capacity'])/np.linalg.norm(n['capacity']))
        assert relative < 2e-7, relative
        stages = {}
        for name, b in arrays.items():
            g = np.load(BASE/f'{law}_gpu/{name}.npz')
            # Check each authoritative probe file against its receipt.
            for backend in ('cpu', 'gpu'):
                receipt = json.loads((BASE/f'{law}_{backend}/AUDIT.json').read_text())
                entry = next(r for r in receipt['rows'] if r['case'] == name)
                assert sha(BASE/f'{law}_{backend}'/entry['file']) == entry['sha256']
            parity = {k: norm(b[k]-g[k]) for k in ('q', 'bedcell', 'class_flux', 'frozen_transport_z')}
            assert parity['class_flux']['maximum'] < 1e-10
            assert parity['frozen_transport_z']['maximum'] < 1e-10
            stock = float(b['frozen_transport_stock'].min())
            budget = float(abs(b['frozen_transport_residual']).max())
            donor = float(b['frozen_transport_minimum_donor'])
            assert stock > 0 and budget < 1e-10 and donor == 1.
            wall = (b['neighbour'] < 0) & (b['bid'] < 0)
            assert np.max(abs(b['class_flux'][:, wall])) == 0.
            stages[name] = dict(tip_rate_mm_s=float(rates[name][tip]),
                tip_bed_change_0_1s_mm=float((b['frozen_transport_z'][tip]-b['z'][tip])*1000),
                region_rate_difference_mm_s=norm(rates[name][region]-rates['FLOW'][region]),
                region_gross_erosion_m3_s=float(np.sum(np.maximum(-rates[name][region], 0)*a['area'][region])*.6/1000),
                capacity_relative_l2=float(np.linalg.norm(b['bedcell'][3:5]-2650*n['capacity'])/np.linalg.norm(2650*n['capacity'])),
                cpu_gpu=parity, minimum_stock_kg_m2=stock, maximum_mass_residual_kg=budget,
                minimum_donor_factor=donor, wall_flux=0.)
        speed = np.linalg.norm(a['q'][1:]/a['q'][0], axis=0)
        shear = speed*.41/np.log1p(a['q'][0]*float(np.float32(.368))/a['capacity_z0'])
        samples = []
        for x in (7.75, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 12.25):
            i = int(np.argmin(np.sum((xy-[x, 3.25])**2, axis=1)))
            samples.append(dict(xy=xy[i].tolist(), BRAIDS_depth_m=float(a['q'][0, i]), FLOW_depth_m=float(n['h'][i]),
                BRAIDS_velocity_m_s=(a['q'][1:, i]/a['q'][0, i]).tolist(),
                FLOW_primary_velocity_m_s=n['primary'][:, i].tolist(),
                FLOW_sediment_velocity_m_s=n['characteristic'][:, i].tolist(),
                BRAIDS_shear_velocity_m_s=float(shear[i]), FLOW_shear_velocity_m_s=float(n['shear'][i]),
                BRAIDS_capacity_m2_s=(a['bedcell'][3:5, i]/2650).tolist(), FLOW_capacity_m2_s=n['capacity'][:, i].tolist(),
                rates_mm_s={k: float(v[i]) for k, v in rates.items()}))
        r = dict(cells=len(xy), native=verify_native(ROOT/f'validation/_hydraulics162/mobile/delft_{law}'),
            same_input_capacity_relative_l2=relative, native_upwind_replay_max_m2_s=face_error,
            tip_native_rate_mm_s=float(rates['FLOW'][tip]),
            tip_ordered_forcing_gap_closed_fraction=float((rates['NATIVE_ALL_FORCING'][tip]-rates['BRAIDS'][tip])/(rates['FLOW'][tip]-rates['BRAIDS'][tip])),
            region_hydraulics=dict(depth_m=norm(a['q'][0, region]-n['h'][region]),
                primary_velocity_m_s=norm(a['q'][1:, region]/a['q'][0, region]-n['primary'][:, region])),
            stages=stages, samples=samples)
        bmag = np.linalg.norm(a['bedcell'][3:5]/2650., axis=0)
        nmag = np.linalg.norm(n['capacity'], axis=0)
        r['capacity_jet_sections'] = []
        for x in (7.75, 8.25, 8.75, 9.25, 9.75, 10.25, 12.25):
            ids = np.flatnonzero((xy[:, 0] == x) & (xy[:, 1] >= 3.))
            r['capacity_jet_sections'].append(dict(x_m=x,
                BRAIDS_peak_y_m=float(xy[ids[np.argmax(bmag[ids])], 1]),
                FLOW_peak_y_m=float(xy[ids[np.argmax(nmag[ids])], 1]),
                BRAIDS_capacity_centroid_y_m=float(np.sum(xy[ids, 1]*bmag[ids])/np.sum(bmag[ids])),
                FLOW_capacity_centroid_y_m=float(np.sum(xy[ids, 1]*nmag[ids])/np.sum(nmag[ids]))))
        result[law] = r; raw[law] = (a, n, arrays, rates, shear)

    # Primary-flow versus sediment-characteristic quantities are kept separate.
    a, n, arrays, rates, shear = raw['mpm']; xy = a['xy']
    polygons = ResultsStore(ROOT/'validation/_shoal165/groin/mpm_cpu/run').read(0).polygons
    vertices = [np.asarray(p.exterior.coords) for p in polygons]
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.5), layout='constrained')
    for col, (bval, nval, title, unit) in enumerate((
        (np.linalg.norm(a['q'][1:]/a['q'][0], axis=0), np.linalg.norm(n['primary'], axis=0), 'Primary speed', 'm/s'),
        (shear, n['shear'], 'Sediment shear velocity', 'm/s'),
        (np.linalg.norm(a['bedcell'][3:5]/2650, axis=0)*1e3, np.linalg.norm(n['capacity'], axis=0)*1e3, 'MPM capacity', '10⁻³ m²/s'))):
        for row, (values, label) in enumerate(((bval, 'BRAIDS'), (nval, 'Delft3D'))):
            ax = axes[row, col]
            pc = PolyCollection(vertices, array=values, cmap='magma', clim=(0, max(bval.max(), nval.max())))
            ax.add_collection(pc); ax.set(xlim=(6, 16), ylim=(0, 8), aspect='equal', title=label+' | '+title)
            ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); fig.colorbar(pc, ax=ax, label=unit, shrink=.75)
    fig.suptitle('Before bed movement: t = 40 s, raw 0.5 m cells; no viscosity or secondary flow')
    fig.savefig(OUT/'HYDRAULIC_TO_CAPACITY.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained')
    labels = [('FLOW', 'FLOW actual', 'black'), ('BRAIDS', 'BRAIDS actual', '#2865a5'),
              ('NATIVE_ALL_FORCING', 'BRAIDS, all FLOW forcing', '#bd4e21'),
              ('BRAIDS_FACE_DISCHARGE', 'BRAIDS face-discharge control', '#27956d')]
    for law, label, ax in zip(LAWS, LABELS, axes):
        aa, nn, dd, rr, _ = raw[law]
        ids = np.flatnonzero((aa['xy'][:, 1] == 3.25) & (aa['xy'][:, 0] >= 6) & (aa['xy'][:, 0] <= 16))
        ids = ids[np.argsort(aa['xy'][ids, 0])]
        for key, legend, color in labels:
            ax.plot(aa['xy'][ids, 0], rr[key][ids], 'o-', ms=3, color=color, label=legend)
        ax.axhline(0, color='grey', lw=.7); ax.axvspan(8, 10, color='grey', alpha=.12)
        ax.set(title=label, xlabel='x [m], y = 3.25 m', ylabel='Bed rate [mm/s]'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Frozen-state attribution: negative = scour; shaded span is solid immediately below the row')
    fig.savefig(OUT/'SCOUR_ATTRIBUTION.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    ordered = ('BRAIDS', 'NATIVE_DEPTH', 'NATIVE_PRIMARY', 'NATIVE_CHARACTERISTIC', 'NATIVE_ALL_FORCING')
    for law, label, ax in zip(LAWS, LABELS, axes):
        vals = [result[law]['stages'][k]['tip_rate_mm_s'] for k in ordered]+[result[law]['tip_native_rate_mm_s']]
        ax.bar(range(6), -np.array(vals), color=['#2865a5']+['#bd4e21']*4+['#222222'])
        ax.set_xticks(range(6), ['BRAIDS', '+ FLOW\ndepth', '+ primary\nvelocity', '+ sediment\nvelocity', '+ roughness\n/all forcing', 'FLOW\nface rule'], rotation=35, ha='right')
        ax.set(title=label, ylabel='Tip erosion rate [mm/s]'); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Ordered interventions at (8.25, 3.25) m — not independent additive percentages')
    fig.savefig(OUT/'TIP_CHAIN.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), layout='constrained')
    for row, (law, label) in enumerate(zip(LAWS, LABELS)):
        store = ResultsStore(ROOT/f'validation/_shoal165/groin/{law}_cpu/run')
        zero = store.read(0); last = store.read(-1); xy = raw[law][0]['xy']
        ix = np.rint(xy[:, 0]/.5+.5).astype(int); iy = np.rint(xy[:, 1]/.5+.5).astype(int)
        with Dataset(ROOT/f'validation/_hydraulics162/mobile/delft_{law}/trim-sim_P057.nc') as ds:
            nd = -np.asarray(ds['DPS'][-1]-ds['DPS'][0])[ix, iy]*1000
            initial = dict(bed_max_m=float(abs(-np.asarray(ds['DPS'][0])[ix, iy]-zero.z).max()),
                depth_max_m=float(abs(np.asarray(ds['S1'][0]+ds['DPS'][0])[ix, iy]-zero.q[0]).max()),
                stock_max_kg_m2=float(abs(np.asarray(ds['MSED'][0, 0, 0])[ix, iy]-zero.arrays['mass'].sum(axis=(0, 1))).max()))
            assert initial['bed_max_m'] < 1e-12 and initial['depth_max_m'] < 3e-8
            assert initial['stock_max_kg_m2'] < 1e-9
            result[law]['initial_pair'] = initial
        bd = (last.z-zero.z)*1000; limit = max(abs(bd).max(), abs(nd).max())
        result[law]['retained_60s_tip_change_mm'] = dict(BRAIDS=float(bd[472]), FLOW=float(nd[472]))
        for col, (values, name) in enumerate(((bd, 'BRAIDS'), (nd, 'Delft3D'))):
            ax = axes[row, col]; pc = PolyCollection(vertices, array=values, cmap='RdBu_r', clim=(-limit, limit))
            ax.add_collection(pc); ax.set(xlim=(6, 16), ylim=(0, 8), aspect='equal', title=label+' | '+name)
            ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); fig.colorbar(pc, ax=ax, label='Bed change [mm]', shrink=.75)
    fig.suptitle('Retained, authenticated 60 s coupled runs (20 s mobile bed); no new long runs')
    fig.savefig(OUT/'RETAINED_BED_PATTERNS.png', dpi=170); plt.close(fig)

    sources = source_audit()
    audit = dict(scope='Frozen t40 current-release attribution; 48 probes and 480 transport-only updates of 0.01 s; no new coupled evolution.',
                 runtime_manifest_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
                 laws=result, native_sources_read_only=sources, installed_runtime_changed=False,
                 no_physical_coefficient_tuning=True, no_formula_changes=True,
                 capacity_reproduction_is_not_experimental_validation=True,
                 projection_control='Polygon face-discharge moment; not adopted: tip scour gets weaker for all three laws.',
                 scripts={f: sha(ROOT/'desktop'/f) for f in ('replay_scour166.py', 'review_scour166.py')})
    (OUT/'AUDIT.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({law: {k: v for k, v in r.items() if k not in ('samples', 'stages', 'native')} for law, r in result.items()}, indent=2))


if __name__ == '__main__':
    main()
