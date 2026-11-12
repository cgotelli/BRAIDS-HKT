"""Frozen-state attribution on installed preview26; no solver/reference edits.

Each intervention is a diagnostic only. It is not a new physical combination,
accepted trajectory, or alternative public method. Output folders are exclusive.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'desktop/runtime'), str(ROOT), str(ROOT/'desktop')]
from check_bedload161 import save, sha
from braids_core.sediment.bed_columns_v101 import host
from braids_core.sediment.portable_river_runtime import install
from braids_core import river_worker
from braids_desktop.results import ResultsStore

LAWS = ('mpm', 'ashida_michiue_generalized', 'wang_fredsoe')
BASE = ROOT/'validation/_scour166'
NATIVE = ROOT/'validation/_hydraulics162/mobile'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['cpu', 'gpu'], required=True)
    ap.add_argument('--tag', default='v3')
    args = ap.parse_args()
    runtime = ROOT/'desktop/runtime'
    assert sha(runtime/'RUNTIME.json') == '373b51e24ca405ac5bc7bdb28632d847caa2092a1c6504670eba41591fc21f60'
    install(json.loads((runtime/'PRECOMPILED.json').read_text()), runtime)
    for law in LAWS:
        out = BASE/args.tag/f'{law}_{args.backend}'
        out.mkdir(parents=True, exist_ok=False)
        source = ROOT/f'validation/_shoal165/groin/{law}_{args.backend}'
        store = ResultsStore(source/'run')
        frame = store.read(8)
        assert frame.time_s == 40.
        np.testing.assert_array_equal(frame.z, store.read(0).z)
        xy = np.array([p.centroid.coords[0] for p in frame.polygons])
        ix = np.rint(xy[:, 0]/.5+.5).astype(int)
        iy = np.rint(xy[:, 1]/.5+.5).astype(int)
        native_path = NATIVE/f'delft_{law}/trim-sim_P057.nc'
        with Dataset(native_path) as ds:
            get = lambda key: np.asarray(ds[key][8]).squeeze()[..., ix, iy]
            h = get('S1')+get('DPS')
            u = np.asarray(ds['U1'][8, 0]); v = np.asarray(ds['V1'][8, 0])
            primary = np.array([.5*(u+np.roll(u, 1, axis=0)),
                                .5*(v+np.roll(v, 1, axis=1))])[:, ix, iy]
            char = np.array([get('UUU'), get('VVV')])
            z0 = get('ZUMOD')/np.expm1(.41*get('UMOD')/np.maximum(get('USTAR'), 1e-30))
            assert np.isfinite(z0).all() and np.min(z0) > 0
            chezy = np.sqrt(9.81)/.41*np.log1p(np.maximum(.01, h)/(np.e*z0))
            factor = .41*chezy/(np.sqrt(9.81)*np.log1p(h*float(np.float32(.368))/z0))
            nq = np.ascontiguousarray(np.vstack([h, h*char]))
            cq = nq.copy(); cq[1:] *= factor
            native = dict(xy=xy, h=h, primary=primary, characteristic=char, z0=z0,
                          chezy=chezy, effective_q=cq, capacity=np.array([get('SBCU'), get('SBCV')]),
                          shear=get('USTAR'), zumod=get('ZUMOD'),
                          face_u=np.asarray(ds['SBUU'][8, 0]), face_v=np.asarray(ds['SBVV'][8, 0]))
        np.savez(out/'NATIVE.npz', **native)
        original = river_worker.core.factory
        rows = []

        def factory(*a, **kw):
            make = original(*a, **kw)

            def build(*a, **kw):
                owner = make(*a, **kw)
                p = owner.physics; xp = owner.xp; c = owner.midpoint
                cap = owner.capacity[0]
                evaluate = cap.evaluate
                # Ordered interventions, plus independent speed/direction controls.
                bv = frame.q[1:]/frame.q[0]
                bmag = np.linalg.norm(bv, axis=0); nmag = np.linalg.norm(char, axis=0)
                speed_only = bv*nmag/np.maximum(bmag, 1e-30)
                direction_only = char*bmag/np.maximum(nmag, 1e-30)
                c.interval_base = xp.asarray(frame.q).copy(); c.minimum_theta = 1.; c.frequency_peak = 0.
                _, wf = c.rate(c.interval_base, .0001, 40.)
                # Polygon moment reconstruction of discharge. On squares this
                # is the native half-sum of opposite face discharges / depth.
                # This is a control only, not a production change.
                geo = owner.data; oo = geo['owner']; jj = geo['neighbour']; mask = jj >= 0
                moment = np.zeros((p.n, 2)); ww = host(wf)*geo['length']
                np.add.at(moment, oo, (geo['mid'].T-xy[oo])*ww[:, None])
                np.add.at(moment, jj[mask], (geo['mid'][:, mask].T-xy[jj[mask]])*(-ww[mask, None]))
                face_q = frame.q.copy(); face_q[1:] = moment.T/geo['area']
                cases = [('BRAIDS', frame.q),
                         ('BRAIDS_FACE_DISCHARGE', face_q),
                         ('NATIVE_DEPTH', np.vstack([h, h*bv])),
                         ('NATIVE_PRIMARY', np.vstack([h, h*primary])),
                         ('NATIVE_SPEED', np.vstack([frame.q[0], frame.q[0]*speed_only])),
                         ('NATIVE_DIRECTION', np.vstack([frame.q[0], frame.q[0]*direction_only])),
                         ('NATIVE_CHARACTERISTIC', nq),
                         ('NATIVE_ALL_FORCING', nq)]
                try:
                    for label, q in cases:
                        q = np.ascontiguousarray(q)
                        raw = xp.asarray(q).copy(); z = xp.asarray(frame.z)
                        c.interval_base = raw.copy(); c.minimum_theta = 1.; c.frequency_peak = 0.
                        rate, water = c.rate(raw, .0001, 40.)
                        captured = {}

                        def probe(qin, hi, *, capacity_q=None, chezy=None):
                            old = cap.z0, cap.d90
                            try:
                                if label == 'NATIVE_ALL_FORCING':
                                    capacity_q = xp.asarray(cq); chezy = xp.asarray(native['chezy'])
                                    cap.z0 = xp.asarray(z0)
                                    cap.d90 = xp.full(p.n, float(np.float32(.001)))
                                captured.update(capacity_q=host(capacity_q).copy(),
                                    capacity_input_q=host(qin).copy(), capacity_chezy=host(chezy).copy(),
                                    capacity_z0=host(cap.z0).copy(), capacity_d90=host(cap.d90).copy())
                                result = evaluate(qin, hi, capacity_q=capacity_q, chezy=chezy)
                                captured['capacity_output'] = host(result).copy()
                                return result
                            finally:
                                cap.z0, cap.d90 = old

                        cap.evaluate = probe
                        owner.bed_faces(raw, z, xp.zeros(p.n), water)
                        cap.evaluate = evaluate
                        data = dict(q=q, z=frame.z, xy=xy, rate=host(rate).copy(),
                                    water=host(water).copy(), **captured)
                        data.update({name: host(getattr(p, name)).copy() for name in (
                            'effective_q', 'effective_fields', 'effective_face', 'bedcell',
                            'bedface', 'bedcorrected', 'capacity_gradient', 'capacity_limiter',
                            'aux', 'class_flux', 'wave_diagnostic', 'wave_physical_normal')})
                        data.update({name: owner.data[name] for name in (
                            'owner', 'neighbour', 'normal', 'length', 'mid', 'bid', 'area')})
                        # Ten transport-only updates through the actual native
                        # class donor/divergence kernels, with fixed hydraulics.
                        # Not a coupled flow prediction. Columns remain private
                        # copies; no source state or accepted owner is advanced.
                        available = owner.columns.available.copy()
                        bed = z.copy(); residuals = []; minimum = 1.
                        cap.evaluate = probe
                        for step in range(10):
                            owner.bed_faces(raw, bed, xp.zeros(p.n), water)
                            minimum = min(minimum, owner._class_update.apply(available, .01))
                            delta = p.class_delta.copy(); available += delta
                            bed += delta[0]/1590.
                            outside = xp.asarray((geo['neighbour'] < 0) & (geo['bid'] >= 0))
                            residuals.append(float(xp.sum(delta*p.g['area'])+
                                .01*2650.*xp.sum(p.class_flux[:, outside]*p.g['length'][outside])))
                        cap.evaluate = evaluate
                        data.update(frozen_transport_z=host(bed).copy(),
                            frozen_transport_stock=host(available).copy(),
                            frozen_transport_residual=np.asarray(residuals),
                            frozen_transport_minimum_donor=np.array(minimum))
                        file = out/f'{label}.npz'
                        np.savez(file, **data)
                        rows.append(dict(case=label, file=file.name, sha256=sha(file)))
                finally:
                    cap.evaluate = evaluate
                return owner
            build.__dict__.update(make.__dict__)
            return build

        river_worker.core.factory = factory
        try:
            # Reprepare rather than interpreting any prior runtime's identities.
            prepared = river_worker.prepare(source/'project.json', out/'prepared')
            status = river_worker.run(prepared, out/'construction_run', until=.001)
            save(out/'AUDIT.json', dict(rows=rows, status=status, backend=args.backend,
                runtime_sha256=sha(runtime/'RUNTIME.json'), script_sha256=sha(__file__),
                native_output_sha256=sha(native_path), source_project_sha256=sha(source/'project.json'),
                note='Frozen t=40 probes; construction run is not an evolved diagnostic trajectory.'))
            print(law, args.backend, 'finished', flush=True)
        finally:
            river_worker.core.factory = original


if __name__ == '__main__':
    main()
