"""Transient, capability-gated shortcuts for the current river worker.

No accepted arrays or physical formulae are cached. Each call installs the
wrappers only for the duration of one public run, after which they are removed.
"""
from contextlib import contextmanager
import inspect
import threading
import textwrap
import warnings
import numpy as np

_lock = threading.RLock()


@contextmanager
def optimized_river_run(*, prepared=None, restart=None):
    from .shoreline_solver import ShorelinePhysics, ShorelineSolver
    from .flow2dh_fixed_bed import Flow2DHFixedBed
    from .river_exterior_sediment import ExteriorSediment
    from .runtime_snapshot import make_copier
    from .runtime_shallow_fusion import make_installer
    from validation.sediment.river_morphodynamics import shallow_face_order
    from . import adaptive_river_worker
    from .bend_methods_v1 import OPTIONS
    refresh_original = ShorelinePhysics.refresh_secondary_momentum
    finish_original = Flow2DHFixedBed.finish
    step_original = ShorelineSolver.step
    snapshot_original = ShorelineSolver._trial_snapshot
    hierarchy_original = adaptive_river_worker.hierarchy
    exterior_original = ExteriorSediment.prepare
    shallow_original = shallow_face_order.install
    fused_kernel = [None]
    fusion_available = [True]

    def fused_manning(raw, h, rough):
        import cupy as cp
        if fused_kernel[0] is None:
            fused_kernel[0] = cp.ElementwiseKernel(
                'float64 raw, float64 h, float64 rough',
                'float64 c, float64 z0, float64 cs, float64 factor', r'''
                if (isnan(raw)||isnan(h)||isnan(rough)) {c=z0=cs=factor=nan("");}
                else {
                  c=pow(raw,0.16666666666666666)/rough;
                  z0=raw/(2.718281828459045*expm1(.41*c/3.132091952673165));
                  cs=fmax(1.e-6,3.132091952673165/.41*log1p(fmax(.01,h)/fmax(1.e-8,2.718281828459045*z0)));
                  factor=.41*cs/(3.132091952673165*log1p(h*0.36800000071525574/z0));
                }''', 'braids_fused_exterior_manning_v1',
                options=('--std=c++17','--fmad=false','--prec-div=true','--prec-sqrt=true'))
        return fused_kernel[0](raw, h, rough)

    def make_exterior_shortcut():
        # Rewrite only this exact, recognizable calculation block. Every
        # unrecognized source and non-Manning case retains the canonical path.
        try:
            if exterior_original.__code__.co_name != 'prepare':
                return exterior_original
            code = textwrap.dedent(inspect.getsource(exterior_original))
            first = '    if self.exterior.bedforms is not None:'
            last = '    cq=safe.copy();cq[1:]*=factor'
            if code.count(first) != 1 or code.count(last) != 1:
                return exterior_original
            a, b = code.index(first), code.index(last)
            if a >= b: return exterior_original
            previous = code[a:b]
            replacement = ('    if xp is not np and rm==2 and self.exterior.bedforms is None:\n'
                           '        c,z0,cs,factor=fused_manning(raw,h,rv)\n'
                           '    else:\n' + ''.join('    '+line for line in previous.splitlines(keepends=True)))
            namespace = dict(exterior_original.__globals__)
            namespace['fused_manning'] = fused_manning
            exec(compile(code[:a]+replacement+code[b:], '<braids-exterior-fusion>', 'exec'), namespace)
            optimized = namespace['prepare']
            def prepare(self, q, z, hi):
                if not fusion_available[0]:
                    return exterior_original(self, q, z, hi)
                try:
                    return optimized(self, q, z, hi)
                except Exception as exc:
                    # No accepted state has changed. A missing GPU compiler may
                    # safely use the same original boundary evaluation.
                    if not type(exc).__module__.startswith('cupy.cuda.compiler'): raise
                    fusion_available[0] = False
                    warnings.warn('Boundary fusion unavailable; using the existing sediment operator: '+str(exc), RuntimeWarning)
                    return exterior_original(self, q, z, hi)
            return prepare
        except (OSError, TypeError, SyntaxError, ValueError):
            return exterior_original

    exterior_shortcut = make_exterior_shortcut()

    def hierarchy(profile, stop=None):
        from .runtime_topology_cache import cached_restart_tree
        tree=cached_restart_tree(prepared,restart,profile,stop)
        return hierarchy_original(profile,stop) if tree is None else tree

    def refresh(p, q, hi):
        selected = p.identity.get('resolved_river_model', {}).get('model', {}).get('physics', {})
        hydro = selected.get('hydrodynamics', {})
        safe = (hydro.get('secondary_flow') == 'secondary.off' and
                hydro.get('turbulence') == 'off' and
                not p.identity.get('river_bedforms', {}).get('enabled', False) and
                not getattr(getattr(p, '_secondary_clock', None), 'active', False))
        # A restart/import can carry a nonzero state even if secondary flow is
        # now disabled. Check once after state restoration, per mesh owner.
        if safe and not hasattr(p, '_zero_secondary_shortcut'):
            p._zero_secondary_shortcut = not bool(p.xp.any(p.W != 0.))
        if safe and getattr(p, '_zero_secondary_shortcut', False):
            return None
        return refresh_original(p, q, hi)

    def finish(p, qold, qnew, water, *, morphology=False):
        if getattr(p, '_zero_secondary_shortcut', False):
            if morphology or not p._open:
                raise ValueError('Inactive secondary shortcut needs an open fixed-bed stage')
            p.Wtrial[:] = p.W
            p.ztrial[:] = p.z0
            p.Mtrial[:] = p.M
            p.face_integral.fill(0.)
            p.scalarflux_integral.fill(0.)
            p.last = dict(morphology=False, solid_volume_residual_m3=0., solid_boundary_m3=0.,
                          secondary_residual=0., secondary_source=0., secondary_projection=0.,
                          secondary_boundary=0.)
            return p.last
        return finish_original(p, qold, qnew, water, morphology=morphology)

    def step(owner, *args, **kwargs):
        p = owner.physics
        if owner.backend == 'cpu' and owner.morphology and not hasattr(p, '_wet_face_shortcut'):
            p._wet_face_shortcut = False
            wave = getattr(p, 'wave_engine', None)
            fn = None if wave is None else wave.fn.get('matrix_bed_faces')
            state = p.identity.get('sediment_wave_research', {}).get('wave_state', '')
            if (wave is not None and 'matrix_bed_faces_active' in wave.fn and
                    getattr(p, 'sediment_depth', 0.) >= .001 and
                    state.startswith('hydraulic depth') and getattr(fn, '__name__', '') == 'call' and
                    'original' in fn.__code__.co_freevars):
                cell = fn.__closure__[fn.__code__.co_freevars.index('original')]
                if getattr(cell.cell_contents, '__name__', None) == 'matrix_bed_faces_cached':
                    cell.cell_contents = wave.fn['matrix_bed_faces_active']
                    p._wet_face_shortcut = True
        if owner.backend == 'gpu' and owner.morphology and not hasattr(p, '_wet_face_shortcut'):
            p._wet_face_shortcut = False
            wave = getattr(p, 'wave_engine', None)
            fn = None if wave is None else wave.fn.get('matrix_bed_faces')
            code = None if wave is None else getattr(getattr(wave, 'module', None), 'code', None)
            marker = 'diagnostic[k*nf+i]=0.;if(j<0)return;'
            state = p.identity.get('sediment_wave_research', {}).get('wave_state', '')
            if (isinstance(code, str) and code.count(marker) == 1 and
                    getattr(p, 'sediment_depth', 0.) >= .001 and
                    state.startswith('hydraulic depth') and getattr(fn, '__name__', '') == 'call' and
                    'original' in fn.__code__.co_freevars):
                cell = fn.__closure__[fn.__code__.co_freevars.index('original')]
                raw = cell.cell_contents
                if getattr(raw, 'name', None) == 'matrix_bed_faces_cached':
                    try:
                        import cupy as cp
                        from .river_kernel_cache import cuda_module
                        # q is the existing physical wave state. A face with a
                        # <=1 mm endpoint is already closed by the canonical
                        # final sediment mask. Gradients and boundary work are
                        # unchanged; zero diagnostics before skipping the wave.
                        fast = code.replace(marker, marker[:-len('if(j<0)return;')] +
                            'if(j<0||q[o]<=.001||q[j]<=.001)return;')
                        # Gradients of inactive cells cannot be consumed by
                        # the remaining interior wet-wet wave faces. Wet
                        # cells still read every neighbour, including dry
                        # neighbours, to preserve the original stencil.
                        gradient = ' LOOP(n){int na=off[n];\n  for(int c=0;c<4;++c)'
                        names = ('matrix_bed_faces_cached',)
                        if fast.count(gradient) == 1:
                            fast = fast.replace(gradient, ' LOOP(n){int na=off[n];\n'
                                '  if(q[i]<=.001){for(int c=0;c<8;++c)grad[c*n+i]=0.;continue;}\n'
                                '  for(int c=0;c<4;++c)')
                            names = ('wave_state_gradients', 'matrix_bed_faces_cached')
                        module, entries = cuda_module(cp, fast, OPTIONS, names)
                        if 'wave_state_gradients' in entries:
                            wave.fn['wave_state_gradients'] = entries['wave_state_gradients']
                        cell.cell_contents = entries['matrix_bed_faces_cached']
                        p._wet_face_module = module
                        p._wet_face_shortcut = True
                    except Exception as exc:
                        warnings.warn('Wet-face optimization unavailable; using the existing sediment operator: '+str(exc), RuntimeWarning)
        return step_original(owner, *args, **kwargs)

    with _lock:
        ShorelinePhysics.refresh_secondary_momentum = refresh
        Flow2DHFixedBed.finish = finish
        ShorelineSolver.step = step
        ShorelineSolver._trial_snapshot = make_copier(snapshot_original)
        ExteriorSediment.prepare = exterior_shortcut
        shallow_face_order.install = make_installer(shallow_original)
        adaptive_river_worker.hierarchy = hierarchy
        try:
            yield
        finally:
            ShorelinePhysics.refresh_secondary_momentum = refresh_original
            Flow2DHFixedBed.finish = finish_original
            ShorelineSolver.step = step_original
            ShorelineSolver._trial_snapshot = snapshot_original
            ExteriorSediment.prepare = exterior_original
            shallow_face_order.install = shallow_original
            adaptive_river_worker.hierarchy = hierarchy_original
