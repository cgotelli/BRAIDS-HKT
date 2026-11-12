"""Isolated wall-normal least-squares control on the current wall-pressure base.

Reuse the recorded geometrically weighted reflective observation from 167,
not a strong zero-normal-velocity assignment at the reconstructed face. Its
previous coupled test predates the exact reflecting-wall pressure correction.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'desktop/runtime'), str(ROOT), str(ROOT/'desktop')]
from check_jet173 import setup, document
from stage_wall172 import save, sha
from corner_candidate167 import install
BASE = ROOT/'validation/_quadrature176/wall_fit'


def run(backend, end=None):
    from braids_core import river_worker
    from braids_core.sediment.bed_columns_v101 import host
    setup(); install(backend, 'normal')
    out = BASE/('probe' if end is None else 'trajectory')/backend
    out.mkdir(parents=True, exist_ok=False)
    if end is None:
        inputs = {'potential': ROOT/'validation/_corner174/potential/gpu/RESIDUAL.npz',
                  'affine': ROOT/'validation/_corner174/probes_v2/gpu/affine_+0.5.npz',
                  'base40': ROOT/'validation/_zero175/probes_v2/baseline/gpu/base40.npz'}
        cases = {name: dict(np.load(p)) for name,p in inputs.items()}
        original = river_worker.core.factory
        def factory(*args, **kw):
            make = original(*args, **kw)
            def build(*args, **kw):
                owner = make(*args, **kw); c = owner.midpoint; xp = owner.xp
                oldz = owner.z.copy(); save(out/'OWNER.json', owner.identity)
                for name,a in cases.items():
                    q = np.ascontiguousarray(a['q']); owner.z[:] = xp.asarray(a['z'])
                    c.interval_base = xp.asarray(q).copy(); c.minimum_theta = 1.; c.frequency_peak = 0.
                    rate,water = c.rate(c.interval_base, 1e-6, 40.)
                    arrays = dict(q=q,z=a['z'],xy=a['xy'],rate=host(rate).copy(),water=host(water).copy())
                    arrays.update({k:owner.data[k] for k in ('owner','neighbour','normal','mid','length','area','bid')})
                    if backend == 'gpu':
                        arrays.update({k:host(v).copy() for k,v in owner.hydraulics.flux.buffers.items()})
                        arrays.update({'rec_'+k:host(v).copy() for k,v in owner.hydraulics.reconstruction.buffers.items()})
                    np.savez(out/(name+'.npz'), **arrays)
                owner.z[:] = oldz
                return owner
            build.__dict__.update(make.__dict__); return build
        river_worker.core.factory = factory
    save(out/'project.json', document(backend, end or 120.))
    if end is None:
        prepared = river_worker.prepare(out/'project.json',out/'prepared')
        status = river_worker.run(prepared,out/'construction',until=.001)
    else:
        from braids_desktop.preparation import prepare
        prepare(out/'project.json',out/'prepared',asset_cache=BASE/'assets')
        status = river_worker.run(out/'prepared',out/'run')
    save(out/'CHECK.json',dict(status=status,runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
        script_sha256=sha(__file__),implementation_sha256=sha(ROOT/'desktop/corner_candidate167.py'),
        changes=dict(wall_gradient='geometrically weighted reflected normal observation',
                     viscosity=0.,sediment_active=False),production_modified=False,Delft3D_modified=False))


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=('cpu','gpu'),default='cpu')
    ap.add_argument('--end',type=float);a=ap.parse_args();run(a.backend,a.end)
