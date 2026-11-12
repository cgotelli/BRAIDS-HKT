"""Read-only state replay through the authenticated Desktop public operators.

This is a diagnostic, not a coupled simulation or alternate release. All face
arrays are copied before the next call can reuse their scratch storage.
"""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'desktop/runtime'), str(ROOT), str(ROOT/'desktop')]
from check_bedload161 import sha, save
from braids_core.sediment.bed_columns_v101 import host
from braids_core.sediment.portable_river_runtime import install
from braids_core import river_worker
from braids_desktop.results import ResultsStore

BASE = ROOT/'validation/_hydraulics162/mobile'
OUT = ROOT/'validation/_groin163'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['cpu', 'gpu'], required=True)
    ap.add_argument('--reflected', action='store_true')
    ap.add_argument('--affine', action='store_true')
    ap.add_argument('--tag')
    args = ap.parse_args()
    out = OUT/(args.tag or ('reflected' if args.reflected else 'affine' if args.affine else '.'))/args.backend
    out.mkdir(parents=True, exist_ok=False)
    install(json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text()), ROOT/'desktop/runtime')
    if args.reflected:
        from wall_bounds163 import install as wall_test
        wall_test(args.backend)
    store = ResultsStore(BASE/'braids_mpm_cpu/run')
    xy = np.array([p.centroid.coords[0] for p in store.read(0).polygons])
    ix = np.rint(xy[:, 0]/.5+.5).astype(int)
    iy = np.rint(xy[:, 1]/.5+.5).astype(int)
    original = river_worker.core.factory
    rows = []

    def factory(*a, **kw):
        make = original(*a, **kw)

        def build(*a, **kw):
            owner = make(*a, **kw)
            p = owner.physics
            xp = owner.xp
            c = owner.midpoint
            if args.reflected:
                owner.identity['groin163_research']='reflected_wall_velocity_limiter_bounds'
            with Dataset(BASE/'delft_mpm/trim-sim_P057.nc') as ds:
                for seconds in (10, 30, 40):
                    index = seconds//5
                    frame = store.read(index)
                    h = np.asarray(ds['S1'][index]+ds['DPS'][index])[ix, iy]
                    u = np.asarray(ds['U1'][index, 0]); v = np.asarray(ds['V1'][index, 0])
                    primary = np.array([.5*(u+np.roll(u, 1, axis=0)),
                                        .5*(v+np.roll(v, 1, axis=1))])[:, ix, iy]
                    characteristic = np.array([ds['UUU'][index], ds['VVV'][index]])[:, ix, iy]
                    native = {name: np.asarray(ds[name][index])[..., ix, iy]
                              for name in ('S1', 'DPS', 'UUU', 'VVV', 'UMOD', 'USTAR', 'ZUMOD',
                                           'SBUU', 'SBVV') if name in ds.variables}
                    np.savez(out/f'NATIVE_{seconds:04d}.npz', primary=primary,
                             characteristic=characteristic, xy=xy, **native)
                    cases=[('BRAIDS', frame.q), ('FLOW_PRIMARY', np.vstack([h, h*primary])),
                           ('FLOW_CHARACTERISTIC', np.vstack([h, h*characteristic]))]
                    if args.affine:
                        cases=[(label,np.array([np.ones(p.n),np.ones(p.n),slope*(xy[:,1]-3.)]))
                               for label,slope in [('AFFINE',.2),('INWARD',-.2),('FAST',2.),('TANGENT',0.)]]
                    for label, q in cases:
                        q = np.ascontiguousarray(q)
                        raw = xp.asarray(q).copy(); z = xp.asarray(frame.z)
                        c.interval_base = raw.copy(); c.minimum_theta = 1.; c.frequency_peak = 0.
                        rate, water = c.rate(raw, .0001, seconds)
                        data = dict(q=q, z=frame.z, xy=xy, rate=host(rate).copy(), water=host(water).copy())
                        if args.backend == 'gpu':
                            data.update({name: host(owner.hydraulics.flux.buffers[name]).copy()
                                         for name in ('fl', 'fr', 'local', 'riemann', 'closed')})
                            data['trace'] = host(owner.hydraulics.flux.inputs['trace']).copy()
                        else:
                            trace = np.zeros((2, 4, p.nf)); grad = np.zeros((4, 2, p.n))
                            flags = np.zeros((3, 2, p.nf), np.int32)
                            fun = owner.hydraulics.lib.bend_wall_traces
                            fun.argtypes = [ctypes.c_void_p]*5; fun.restype = None
                            fun(*(x.ctypes.data for x in (q, frame.z, grad, trace, flags)))
                            data.update(trace=trace, hydraulic_gradient=grad, hydraulic_flags=flags)
                        # Public, class-resolved path, not the standalone law.
                        owner.bed_faces(raw, z, xp.zeros(p.n), xp.asarray(data['water']))
                        data.update({name: host(getattr(p, name)).copy() for name in (
                            'effective_q', 'effective_fields', 'effective_face', 'bedcell',
                            'bedface', 'bedcorrected', 'capacity_gradient', 'capacity_limiter',
                            'aux', 'class_flux', 'wave_diagnostic', 'wave_physical_normal')})
                        data.update({name: owner.data[name] for name in (
                            'owner', 'neighbour', 'normal', 'length', 'mid', 'bid', 'area')})
                        file = out/f'{label}_{seconds:04d}.npz'
                        np.savez(file, **data)
                        rows.append(dict(source=label, time_s=seconds, file=file.name, sha256=sha(file)))
                    if args.affine:break
            return owner
        build.__dict__.update(make.__dict__)
        return build

    river_worker.core.factory = factory
    try:
        status = river_worker.run(BASE/f'braids_mpm_{args.backend}/prepared', out/'construction_run', until=.001)
        save(out/'AUDIT.json', dict(rows=rows, status=status, backend=args.backend,
                                   runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'), script_sha256=sha(__file__)))
    finally:
        river_worker.core.factory = original


if __name__ == '__main__':
    main()
