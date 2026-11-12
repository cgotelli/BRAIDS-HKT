"""Same-region acceleration accounting and cross-advection necessity checks.

Native momentum diagnostics are velocity equations. Their projection is a
kinematic budget for the same cell-averaged observable, not an invented native
finite-volume flux ledger. BRAIDS face-ledger balances are exact.
"""
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'desktop/runtime'), str(ROOT), str(ROOT/'desktop')]
from check_bedload161 import save, sha
from review_hydraulics162 import verify_native, load, norm
from review_corner167 import decompose
from review_groin163 import budgets

RAW = ROOT/'validation/_advection168'
OUT = ROOT/'validation/sediment/river_morphodynamics/reports/desktop_advection168'
PREV = ROOT/'validation/_corner167'
REGIONS = {'tip': (8, 8.5, 3, 3.5), 'corner': (7.5, 10.5, 3, 4.5),
           'first_row': (7.5, 10.5, 3, 3.5), 'upper_rows': (7.5, 10.5, 3.5, 4.5)}


def region_mask(xy, bounds):
    x0, x1, y0, y1 = bounds
    return (xy[:, 0]>x0)&(xy[:, 0]<x1)&(xy[:, 1]>y0)&(xy[:, 1]<y1)


def velocity_rate(q, rate):
    return (rate[1:]-(q[1:]/q[0])*rate[0])/q[0]


def suppression_counterfactual(a):
    """Offline one-sided velocity-term deletion, never an executed solver."""
    ids = np.flatnonzero(np.isclose(a['xy'][:,1],3.25)&(a['xy'][:,0]>8)&(a['xy'][:,0]<10))
    rows = []
    for c in ids:
        faces = np.flatnonzero(((a['owner']==c)|(a['neighbour']==c)) & (abs(a['normal'][1])>.99))
        side = a['owner'][faces]==c
        flux = np.where(side[None],a['fl'][:,faces],a['fr'][:,faces])
        rate_y = -(flux*np.where(side,1.,-1.)*a['length'][faces]).sum(1)/a['area'][c]
        u = a['q'][1,c]/a['q'][0,c]
        correction = -(rate_y[1]-u*rate_y[0])
        rows.append(dict(xy=a['xy'][c].tolist(),cross_velocity_acceleration_m_s2=float(-correction/a['q'][0,c]),
                         unpaired_momentum_source_m4_s2=float(correction*a['area'][c])))
    return dict(scope='Offline cancellation only in four wall-adjacent cells; no changes to neighbouring cells or flux ledger',
                cells=rows, net_unpaired_momentum_source_m4_s2=sum(r['unpaired_momentum_source_m4_s2'] for r in rows))


def finite_volume(a, mask):
    ids = np.flatnonzero(mask); area = a['area'][ids]
    volume = float(np.dot(area, a['q'][0, ids]))
    bulk = np.sum(a['q'][1:, ids]*area, axis=1)/volume
    pieces = np.sum([np.array(decompose(a, a['xy'][c])['pieces'])*a['area'][c] for c in ids], axis=0)
    # Each part has its own water-volume contribution, so this conversion is
    # additive. Pressure, bed and numerical wave work remain separate.
    acceleration = (pieces[:, 1:]-pieces[:, :1]*bulk)/volume
    own = mask[a['owner']]; nei = a['neighbour']; inside = nei>=0
    other = np.zeros(len(nei), bool); other[inside] = mask[nei[inside]]
    face_boundary = own != other
    signed = np.where(own[face_boundary], 1., -1.)
    flux = np.where(own[face_boundary][None], a['fl'][:, face_boundary], a['fr'][:, face_boundary])
    boundary_rate = -(flux*a['length'][face_boundary]*signed).sum(axis=1)
    internal = own & other
    internal_source = ((a['fr'][:, internal]-a['fl'][:, internal])*a['length'][internal]).sum(axis=1)
    actual = np.sum(a['rate'][:, ids]*area, axis=1)
    # The spatial residual excludes Strang-split hydraulic resistance. Include
    # its instantaneous generator for the full native acceleration comparison.
    uv = a['q'][1:, ids]/a['q'][0, ids]
    drag_rate = -9.81/45.**2 * np.sqrt((uv*uv).sum(0))*uv
    drag_acceleration = np.sum(drag_rate*area, axis=1)/volume
    closure = float(abs(actual-boundary_rate-internal_source).max())
    assert closure < 1e-11
    assert np.max(abs(pieces.sum(0)-actual)) < 1e-11
    return dict(cells=len(ids), area_m2=float(area.sum()), water_volume_m3=volume,
                bulk_velocity_m_s=bulk.tolist(), volume_and_momentum_rate=actual.tolist(),
                acceleration_parts_m_s2=acceleration.tolist(), bulk_acceleration_m_s2=acceleration.sum(0).tolist(),
                bed_resistance_acceleration_m_s2=drag_acceleration.tolist(),
                full_acceleration_m_s2=(acceleration.sum(0)+drag_acceleration).tolist(),
                external_face_rate=boundary_rate.tolist(), internal_bed_source=internal_source.tolist(),
                ledger_closure=closure)


def native_projection(folder, xy, parent):
    receipt = verify_native(folder)
    ix = np.rint(xy[:, 0]/.5+.5).astype(int); iy = np.rint(xy[:, 1]/.5+.5).astype(int)
    with Dataset(folder/'trim-sim_P057.nc') as d, Dataset(parent/'trim-sim_P057.nc') as ref:
        t = np.asarray(d['time'][:]); j = int(np.argmin(abs(t-40)))
        jr = int(np.argmin(abs(ref['time'][:]-40)))
        exact = {k: bool(np.array_equal(np.asarray(d[k][j]), np.asarray(ref[k][jr])))
                 for k in ('S1','U1','V1','DPS','SBCU','SBCV','MSED')}
        assert all(exact.values())
        assert np.allclose(t[j-2:j+1], [39.98, 39.99, 40.], atol=1e-10, rtol=0)
        h = np.asarray(d['S1'][:])+np.asarray(d['DPS'][:])
        u = np.asarray(d['U1'][:, 0]); v = np.asarray(d['V1'][:, 0])
        def map_u(arr): return .5*(arr[..., ix, iy]+arr[..., ix-1, iy])
        def map_v(arr): return .5*(arr[..., ix, iy]+arr[..., ix, iy-1])
        vel = np.stack([map_u(u), map_v(v)], axis=1); depth = h[:, ix, iy]
        # One-sided derivative only uses fixed-bed snapshots, before activation.
        derivative = lambda arr: (3*arr[j]-4*arr[j-1]+arr[j-2])/.02
        dh = derivative(depth); du = derivative(vel)
        terms = {}
        names = [('pressure','MOM_UPRESSURE','MOM_VPRESSURE'),
                 ('streamwise','MOM_UDUDX','MOM_VDVDY'),
                 ('cross','MOM_VDUDY','MOM_UDVDX'),
                 ('bed_resistance','MOM_UBEDSHEAR','MOM_VBEDSHEAR'),
                 ('reported','MOM_DUDT','MOM_DVDT')]
        for name, uk, vk in names:
            us = np.asarray(d[uk][j]).reshape(-1, 82, 18)[0]
            vs = np.asarray(d[vk][j]).reshape(-1, 82, 18)[0]
            terms[name] = np.stack([map_u(us), map_v(vs)])
        # Native true staggered terms, separate from projected observables.
        row_terms = {k: np.asarray(d[k][j]).reshape(-1, 82, 18)[0, 16:21, 7].tolist()
                     for k in ('MOM_DUDT','MOM_UPRESSURE','MOM_UDUDX','MOM_VDUDY','MOM_UBEDSHEAR')}
        fields = {k: np.asarray(d[k][j]).reshape(-1, 82, 18)[0]
                  for k in ('MOM_VDUDY','U1','V1','KFU','KFV')}
    rows = {}
    for name, bounds in REGIONS.items():
        m = region_mask(xy, bounds); H = depth[j, m]; U = vel[j][:, m]; A = .25
        vol = float(H.sum()*A); bulk = (H*U).sum(1)*A/vol
        parts = {k: (H*val[:, m]).sum(1)*A/vol for k, val in terms.items()}
        redistribution = ((U-bulk[:, None])*dh[m]).sum(1)*A/vol
        total = (H*du[:, m]).sum(1)*A/vol+redistribution
        diagnostic = parts['reported']+redistribution
        sums = sum(parts[k] for k in ('pressure','streamwise','cross','bed_resistance'))+redistribution
        rows[name] = dict(water_volume_m3=vol, bulk_velocity_m_s=bulk.tolist(),
            acceleration_parts_m_s2={k: v.tolist() for k,v in parts.items()},
            redistribution_m_s2=redistribution.tolist(), finite_difference_acceleration_m_s2=total.tolist(),
            reported_acceleration_m_s2=diagnostic.tolist(),
            split_diagnostic_residual_m_s2=(sums-diagnostic).tolist(),
            time_diagnostic_residual_m_s2=(diagnostic-total).tolist())
    return dict(runtime=receipt, unchanged_at40=exact, first_row_terms=row_terms, regions=rows), fields


def manufactured():
    rows = {}
    for name in ('uniform_out','uniform_in','shear_out','shear_in'):
        c = np.load(RAW/f'probes/cpu/{name}.npz'); g = np.load(RAW/f'probes/gpu/{name}.npz')
        assert np.max(abs(c['rate']-g['rate'])) < 1e-11
        selected = (np.isclose(g['xy'][:,1], 3.25)) & (g['xy'][:,0]>8.5)&(g['xy'][:,0]<9.5)
        b = -.3 if name.endswith('_in') else .3
        a = .2 if name.startswith('shear') else 0.
        exact = -a*b*.25
        acc = velocity_rate(g['q'], g['rate'])[0, selected]
        # Manufactured h=.5, u=1+a*y, v=b*y, y measured from the wall.
        # At the wall v=0; at the open north face y=.5, hv*u is NOT zero.
        top_mass = .5*b*.5
        top_momentum = top_mass*(1+a*.5)
        exact_momentum_rate = -top_momentum/.5
        exact_depth_rate = -top_mass/.5
        mean_u = 1+a*.25
        identity = (exact_momentum_rate-mean_u*exact_depth_rate)/.5
        assert abs(identity-exact) < 1e-14
        # Deliberately wrong, offline counterfactual: delete momentum flux but
        # retain continuity. It is NOT installed or advanced in the solver.
        deleted_flux_acceleration = -mean_u*exact_depth_rate/.5
        wall = (g['neighbour']<0)&(g['bid']<0)
        assert np.max(abs(g['water'][wall])) == 0
        if a == 0: assert np.max(abs(acc)) < 1e-12
        rows[name] = dict(analytic_acceleration_m_s2=exact,
            braids_acceleration_m_s2=acc.tolist(),
            suppressed_velocity_term_m_s2=0.,
            deleted_momentum_flux_acceleration_m_s2=deleted_flux_acceleration,
            exact_open_face_mass_flux_m2_s=top_mass,
            exact_open_face_momentum_flux_m3_s2=top_momentum,
            cpu_gpu_rate_max=float(np.max(abs(c['rate']-g['rate']))),
            wall_water_flux_max=float(np.max(abs(g['water'][wall]))))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    states = {k: np.load(PREV/f'baseline/gpu/{k}.npz') for k in ('BRAIDS','FLOW_PRIMARY')}
    xy = states['BRAIDS']['xy']
    fv = {k: {name: finite_volume(a, region_mask(xy, bounds)) for name,bounds in REGIONS.items()}
          for k,a in states.items()}
    flow, ff = native_projection(PREV/'native_momentum_fine', xy, ROOT/'validation/_hydraulics162/mobile/delft_mpm')
    cyclic, cf = native_projection(RAW/'native_cyclic', xy, ROOT/'validation/_groin163/delft_mpm_cyclic')
    smooth = manufactured()
    sources = json.loads((ROOT/'validation/sediment/river_morphodynamics/reports/desktop_corner167/AUDIT.json').read_text())['native_sources_read_only']
    # Every audited source remains byte-identical to the retained upstream zip.
    with zipfile.ZipFile(ROOT/'resources/Delft3D-main.zip') as z:
        for path, digest in sources.items():
            assert sha(ROOT/path) == digest
            assert z.read(next(n for n in z.namelist() if n.endswith('/'+Path(path).name))) == (ROOT/path).read_bytes()
    installed = json.loads((ROOT/'desktop/runtime/RUNTIME.json').read_text())
    for name, checksum in installed['files'].items():
        assert sha(ROOT/'desktop/runtime'/name) == checksum
    project = json.loads((ROOT/'validation/_shoal165/groin/mpm_cpu/project.json').read_text())
    assert project['roughness_parameters']['chezy_m_half_s'] == 45.
    assert project['model']['physics']['hydrodynamics']['roughness'] == 'roughness.chezy'
    old_tangent = np.load(PREV/'final_tangent/gpu/SHEAR.npz')
    select = np.isclose(xy[:,1],3.25)&(xy[:,0]>8.5)&(xy[:,0]<9.5)
    tangent_out = velocity_rate(old_tangent['q'],old_tangent['rate'])[0,select]
    assert np.max(abs(tangent_out+.015)) < 1e-12
    # Preserve the existing authenticated installed manifest without new builds.
    coupled = {}
    for backend in ('cpu','gpu'):
        coupled[backend] = budgets(ROOT/f'validation/_shoal165/groin/mpm_{backend}/run')
    a, _ = load(ROOT/'validation/_shoal165/groin/mpm_cpu/run', ROOT/'validation/_groin163/delft_mpm_cyclic')
    _, flood_ref = load(ROOT/'validation/_shoal165/groin/mpm_cpu/run', ROOT/'validation/_hydraulics162/mobile/delft_mpm')
    target = np.isclose(xy[:,0],8.25)&(xy[:,1]>=3)
    cycle_stats = dict(tip_depth40_m=float(a['nh'][8,472]), tip_u40_m_s=float(a['nu'][8,0,472]),
                       tip_scour60_mm=float((a['nz'][0,472]-a['nz'][-1,472])*1000),
                       peak_y40_m=float(xy[np.flatnonzero(target)[np.argmax(a['nu'][8,0,target])],1]))
    result = dict(scope='Necessity of wall-adjacent cross-advection suppression; conservative momentum audit',
        decision='Keep conservative cross-momentum transport; do not copy FLOW boundary suppression',
        suppression_necessary_for_BRAIDS=False, improved_groin_jet_qualified=False,
        production_changed=False, source_modified=False, runtime_modified=False,
        native_sources_read_only=sources, runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
        manifest_installed=installed, conservative_regions=fv, FLOW=flow, Cyclic=cyclic,
        Cyclic_coupled=cycle_stats, manufactured=smooth, retained_coupled= coupled,
        suppression_counterfactual=suppression_counterfactual(states['BRAIDS']),
        historical_tangent_out_acceleration_m_s2=tangent_out.tolist(),
        manual=dict(url='https://content.oss.deltares.nl/delft3d4/Delft3D-FLOW_User_Manual.pdf',
                    sha256=sha(RAW/'FLOW_User_Manual.pdf'), printed_pages=[274,275], pdf_pages=[297,298],
                    section='10.5.1', equations=['10.29','10.30']),
        scripts={p.name:sha(p) for p in (ROOT/'desktop').glob('*168.py')},
        evidence_inputs={str(p.relative_to(ROOT)):sha(p) for p in [
            RAW/'native_cyclic/INPUTS.json', RAW/'native_cyclic/execution.json',
            RAW/'probes/cpu/AUDIT.json', RAW/'probes/gpu/AUDIT.json',
            PREV/'baseline/gpu/BRAIDS.npz', PREV/'baseline/gpu/FLOW_PRIMARY.npz']})
    save(OUT/'AUDIT.json', result)
    # Physical cross-momentum versus velocity acceleration: exact simple field.
    fig, axs = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    labels = ['Uniform u\noutflow','Uniform u\ninflow','Sheared u\noutflow','Sheared u\ninflow']
    xx = np.arange(4)
    axs[0].bar(xx-.25,[v['analytic_acceleration_m_s2'] for v in smooth.values()],.25,label='Analytic cell mean')
    axs[0].bar(xx,[v['braids_acceleration_m_s2'][0] for v in smooth.values()],.25,label='BRAIDS CPU/GPU')
    axs[0].plot(xx+.25,[0]*4,'x',color='#bc3030',ms=8,label='Suppressed acceleration')
    axs[0].set(xticks=xx,xticklabels=labels,ylabel='Streamwise acceleration (m/s²)',title='No physical rule requires zero beside a wall')
    axs[0].axhline(0,color='.3',lw=.7);axs[0].legend(fontsize=8)
    axs[1].bar(xx-.15,[v['analytic_acceleration_m_s2'] for v in smooth.values()],.3,label='Correct momentum + continuity')
    axs[1].bar(xx+.15,[v['deleted_momentum_flux_acceleration_m_s2'] for v in smooth.values()],.3,label='Wrong: delete only momentum flux')
    axs[1].set(xticks=xx,xticklabels=labels,ylabel='Streamwise acceleration (m/s²)',title='Deleting an open-face flux breaks consistency')
    axs[1].axhline(0,color='.3',lw=.7);axs[1].legend(fontsize=8)
    fig.savefig(OUT/'CROSS_ADVECTION_CHECK.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for ax, comp, title in zip(axs[:2],range(2),('x momentum','y momentum')):
        pieces=np.array(fv['BRAIDS']['corner']['acceleration_parts_m_s2'])[:,comp]
        ax.bar(['Physical\nadvection','Pressure','Interface\ncorrection','Bed\nsource'],pieces)
        ax.axhline(0,color='.3',lw=.7);ax.set(ylabel='Contribution to bulk acceleration (m/s²)',title=f'BRAIDS {title}, matched corner region')
    for label, row in [('FLOW Flood',flow),('FLOW Cyclic',cyclic)]:
        term=row['first_row_terms']['MOM_VDUDY']
        axs[2].plot([8,8.5,9,9.5,10],term,'o-',label=label)
    axs[2].axhline(0,color='.3',lw=.7);axs[2].legend()
    axs[2].set(xlabel='Native U-point x (m), y = 3.25 m',ylabel='Cross-advection acceleration (m/s²)',title='Stock schemes use different boundary stencils')
    fig.savefig(OUT/'CORNER_MOMENTUM.png',dpi=160);plt.close(fig)
    b,_=load(ROOT/'validation/_shoal165/groin/mpm_cpu/run',ROOT/'validation/_hydraulics162/mobile/delft_mpm')
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for k, x in enumerate((8.25,8.75,10.25)):
        ids=np.flatnonzero(np.isclose(xy[:,0],x)&(xy[:,1]>=3));y=xy[ids,1]
        for label,values in [('BRAIDS',b['bu'][8,0,ids]),('FLOW Flood',b['nu'][8,0,ids]),('FLOW Cyclic',a['nu'][8,0,ids])]:
            axs[k].plot(y,values,'o-',label=label,ms=3)
        axs[k].set(xlabel='Cross-channel y (m)',ylabel='Primary streamwise velocity (m/s)',ylim=(0,2.3),title=f'x = {x:g} m at 40 s')
        axs[k].grid(alpha=.2)
    axs[2].legend(fontsize=9);fig.savefig(OUT/'NATIVE_SCHEME_COMPARISON.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(cyclic=cycle_stats,production_changed=False,
          full_BRAIDS_acceleration=fv['BRAIDS']['corner']['full_acceleration_m_s2'],
          suppression_counterfactual=result['suppression_counterfactual']),indent=2))


if __name__ == '__main__':
    main()
