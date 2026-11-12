"""Read-only face integration audit on frozen zero-viscosity states.

The CPU calls the loaded Fortran f-wave functions; CUDA calls the loaded
resident face kernels. No source/runtime is replaced, no trajectory advanced
with the diagnostic quadrature, and no physical viscosity is introduced.
"""
import argparse
import ctypes as ct
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'desktop/runtime'), str(ROOT), str(ROOT/'desktop')]
from check_jet173 import setup, document
from stage_wall172 import save, sha
from corner_exact174 import velocity, face_flux, mean_rate

BASE = ROOT/'validation/_quadrature176'


def gather(a, left, right):
    """One shared mass flux; distinct momentum bed-source work on each side."""
    rate = np.zeros_like(a['q'])
    o, n, length = a['owner'], a['neighbour'], a['length']
    for k in range(3):
        np.add.at(rate[k], o, -left[k]*length)
        use = n >= 0
        np.add.at(rate[k], n[use], right[k, use]*length[use])
    return rate/a['area']


def values_to_trace(a, values):
    t = a['rec_trace'].copy()
    for side, cells in enumerate((a['owner'], a['neighbour'])):
        use = cells >= 0
        h = values[0, side, use]-values[3, side, use]
        if np.any(h <= .001):
            raise ValueError('This diagnostic requires fully wet quadrature traces')
        t[side*4, use] = h
        t[side*4+1:side*4+3, use] = h*values[1:3, side][:, use]
        t[side*4+3, use] = values[3, side, use]
        t[8+side, use] = h
    wall = (a['neighbour'] < 0) & (a['bid'] < 0)
    t[4:8, wall] = t[:4, wall]; t[9, wall] = t[8, wall]
    opened = (a['neighbour'] < 0) & (a['bid'] >= 0)
    t[:, opened] = a['rec_trace'][:, opened]
    return np.ascontiguousarray(t)


def transverse_trace(a, position):
    """Two Gauss nodes of the existing limited linear transverse field.

    Centre values retain the actual WENO/LS normal reconstruction. This does
    not claim high-order recovery of a singularity from cell averages.
    """
    values = a['rec_face_value'].copy()
    tangent = np.array([-a['normal'][1], a['normal'][0]])
    for side, cells in enumerate((a['owner'], a['neighbour'])):
        use = cells >= 0; ids = cells[use]
        slope = np.sum(a['rec_grad'][:, :, ids]*tangent[:, use][None], axis=1)
        values[:, side, use] += slope*(position*.5*a['length'][use])
    return values_to_trace(a, values)


class Flux:
    def __init__(self, owner):
        self.owner = owner
        if owner.backend == 'cpu':
            lib = owner.hydraulics.lib
            self.wave = getattr(lib, '__cutcell_solver_MOD_hr_step_flux')
            self.wave.argtypes = [ct.c_void_p]*8
            self.wave.restype = None
            self.wall = lib.braids_wall_pressure
            self.wall.argtypes = [ct.c_double]*4
            self.wall.restype = ct.c_double

    def __call__(self, a, trace):
        o = self.owner
        if o.backend == 'gpu':
            import cupy as cp
            op = o.hydraulics.flux; g = op.inputs; b = op.buffers
            g['q'][...] = cp.asarray(a['q']); g['bed'][...] = cp.asarray(a['z'])
            g['trace'][...] = cp.asarray(trace)
            nc, nf = np.int64(a['q'].shape[1]), np.int64(len(a['length']))
            grav, dry = np.float64(9.81), np.float64(o.physics.parameters.dry)
            def launch(name, args): op.kernels[name](((int(nf)+127)//128,), (128,), args)
            launch('resident_prepare_faces', (g['trace'], g['q'], g['bed'], g['normal'], g['distance'],
                g['owner'], g['neighbour'], g['kind'], nc, nf, grav, dry, b['local'], b['closed']))
            launch('augmented_fwave_local', (b['local'], nf, grav, dry, b['riemann'], b['augmented_status']))
            launch('resident_assemble_flux', (g['trace'], b['local'], b['riemann'], g['q'], g['bed'], g['normal'],
                g['owner'], g['neighbour'], g['kind'], g['boundary_mode'], g['sampled'], g['bedoffset'],
                g['hydro_constant'], b['closed'], nc, nf, grav, dry, b['fl'], b['fr'], b['speed'], b['face_status']))
            assert not bool(cp.any(b['augmented_status'])) and not bool(cp.any(b['face_status']))
            fl, fr = cp.asnumpy(b['fl']), cp.asnumpy(b['fr'])
        else:
            nf = len(a['length']); fl = np.zeros((3, nf)); fr = fl.copy()
            for f in range(nf):
                c, n = int(a['owner'][f]), int(a['neighbour'][f])
                if n < 0 and a['bid'][f] >= 0: continue
                normal = np.ascontiguousarray(a['normal'][:, f])
                l, r = np.ascontiguousarray(trace[:3, f]), np.ascontiguousarray(trace[4:7, f])
                ll, rr = np.zeros(3), np.zeros(3); speed = np.zeros(1)
                if n < 0:
                    ll[1:] = self.wall(l[0], float(np.dot(l[1:], normal)), 9.81, o.physics.parameters.dry)*normal
                    rr[:] = ll
                else:
                    bl, br = np.array(trace[3, f]), np.array(trace[7, f])
                    self.wave(*(ct.c_void_p(v.ctypes.data) for v in (l, r, bl, br, normal, ll, rr, speed)))
                for cell, side, result in ((c, 0, ll), (n, 1, rr)):
                    if cell < 0: continue
                    assert not a['rec_hydro_constant'][cell], 'Fully wet source quadrature only'
                    result[1:] += .5*9.81*(a['q'][0, cell]+trace[8+side, f])*(trace[3+4*side, f]-a['z'][cell])*normal
                if n < 0: rr[:] = ll
                fl[:, f], fr[:, f] = ll, rr
        opened = (a['neighbour'] < 0) & (a['bid'] >= 0)
        # Exact, previously evaluated section-boundary flux remains untouched.
        fl[:, opened], fr[:, opened] = a['fl'][:, opened], a['fr'][:, opened]
        return fl, fr


def exact_face_table(a, nodes=None):
    """Analytic face data only for the local corner; not a numerical fix."""
    fl = a['fl'].copy()
    local = a['mid'].T-[8., 3.]
    use = np.max(abs(local), axis=1) <= 1.5
    for f in np.flatnonzero(use):
        p = local[f]; axis = int(np.argmax(abs(a['normal'][:, f])))
        sign = a['normal'][axis, f]; length = a['length'][f]
        if nodes is None:
            value = face_flux(*p, axis, length)
        else:
            node, weight = np.polynomial.legendre.leggauss(nodes)
            value = np.zeros(3)
            for k, w in zip(node, weight/2):
                xy = p.copy(); xy[1-axis] += k*length/2
                u = velocity(*xy)
                v = .5*u[axis]*np.r_[1., u]; v[1+axis] += .5*9.81*.5**2
                value += w*v
        fl[:, f] = sign*value
    return fl


def run(backend):
    from braids_core import river_worker
    from braids_core.sediment.bed_columns_v101 import host
    setup(); out = BASE/backend; out.mkdir(parents=True, exist_ok=False)
    inputs = {'potential': ROOT/'validation/_corner174/potential/gpu/RESIDUAL.npz'}
    inputs.update({f'base{t}': ROOT/f'validation/_zero175/probes_v2/baseline/gpu/base{t}.npz' for t in (40,120,300)})
    cases = {key: dict(np.load(path)) for key, path in inputs.items()}
    original = river_worker.core.factory
    def factory(*args, **kw):
        make = original(*args, **kw)
        def build(*args, **kw):
            owner = make(*args, **kw); c = owner.midpoint; xp = owner.xp
            oldz = owner.z.copy(); flux = Flux(owner); rows = {}
            for name, a in cases.items():
                owner.z[:] = xp.asarray(a['z']); q = xp.asarray(np.ascontiguousarray(a['q']))
                c.interval_base = q.copy(); c.minimum_theta = 1.; c.frequency_peak = 0.
                rate, water = c.rate(q, 1e-6, 40. if name == 'potential' else float(name[4:]))
                # Replaying exterior at 40 rather than 120/300 is immaterial:
                # this deck has constant data, but use exact recorded time.
                ref = host(rate).copy()
                assert np.max(abs(ref-a['rate'])) < 1e-12
                fl, fr = flux(a, a['rec_trace'])
                closed = (a['neighbour'] >= 0) | (a['bid'] < 0)
                parity = max(np.max(abs(fl[:, closed]-a['fl'][:, closed])), np.max(abs(fr[:, closed]-a['fr'][:, closed])))
                assert parity < 1e-12, (name, parity)
                left = np.zeros_like(fl); right = np.zeros_like(fr)
                for x in (-1/np.sqrt(3), 1/np.sqrt(3)):
                    l, r = flux(a, transverse_trace(a, x)); left += .5*l; right += .5*r
                result = gather(a, left, right)
                # Base assembly closure includes cancellation of uniform
                # pressure in its gather. No other momentum source is active.
                assert np.max(abs(gather(a, fl, fr)-ref)) < 1e-12
                np.savez(out/(name+'.npz'), baseline=ref, quadrature=result, fl=left, fr=right, q=a['q'], xy=a['xy'])
                rows[name] = dict(base_flux_replay_max=parity, maximum_rate_change=float(abs(result-ref).max()),
                    shared_mass_max=float(abs(left[0]-right[0]).max()),
                    volume_rate_change=float(np.sum((result[0]-ref[0])*a['area'])))
            owner.z[:] = oldz
            save(out/'AUDIT.json', rows)
            return owner
        build.__dict__.update(make.__dict__); return build
    river_worker.core.factory = factory
    save(out/'project.json', document(backend))
    prepared = river_worker.prepare(out/'project.json', out/'prepared')
    status = river_worker.run(prepared, out/'construction', until=.001)
    save(out/'CHECK.json', dict(status=status, sources={str(p): sha(p) for p in inputs.values()},
        script_sha256=sha(__file__), runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
        quadrature_trajectory_run=False, production_changed=False, Delft3D_changed=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--backend', choices=('cpu','gpu'), default='cpu')
    run(ap.parse_args().backend)
