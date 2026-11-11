"""Optional GPU fusion of an existing shallow-face projection and mask."""
import inspect
import textwrap
import warnings
import numpy as np


def make_installer(original):
    kernel = [None]
    enabled = [True]
    def fused(owner, active):
        if not enabled[0] or owner.xp is np or not hasattr(owner, 'wave_diagnostic'):
            return False
        d = owner.wave_diagnostic
        if d.shape != (3, owner.nf):
            return False
        import cupy as cp
        if (owner.xp is not cp or owner.bedface.dtype != cp.float64 or
                d.dtype != cp.float64 or active.dtype != cp.bool_):
            return False
        try:
            if kernel[0] is None:
                kernel[0] = cp.ElementwiseKernel(
                    'float64 nx, float64 ny, bool a, float64 x, float64 y, float64 d0, float64 d1, float64 d2',
                    'float64 outx, float64 outy, float64 e0, float64 e1, float64 e2', r'''
                    double normal=x*nx+y*ny;double factor=a?0.:normal;
                    outx=x-nx*factor;outy=y-ny*factor;e0=a?d0:0.;e1=a?d1:0.;e2=a?d2:0.;
                    ''','braids_shallow_face_projection_v1',
                    options=('--std=c++17','--fmad=false','--prec-div=true','--prec-sqrt=true'))
            g=owner.g; v=owner.bedface
            kernel[0](g['normal'][0],g['normal'][1],active,v[0],v[1],d[0],d[1],d[2],v[0],v[1],d[0],d[1],d[2])
            return True
        except Exception as exc:
            if not type(exc).__module__.startswith('cupy.cuda.compiler'):
                raise
            enabled[0] = False
            warnings.warn('Shallow-face fusion unavailable; using the existing operator: '+str(exc), RuntimeWarning)
            return False
    try:
        source=textwrap.dedent(inspect.getsource(original))
        first='        normal = xp.sum(owner.bedface[:2] * g[\'normal\'], axis=0)'
        last='\n    def bed_faces(owner, q, z, hi):'
        if source.count(first)!=1 or source.count(last)!=1:
            return original
        start=source.index(first);end=source.index(last,start)
        previous=source[start:end]
        if "owner.wave_diagnostic[:, ~active] = 0." not in previous:
            return original
        replacement='        if not fused(owner, active):\n'+''.join('    '+line for line in previous.splitlines(keepends=True))
        namespace=dict(original.__globals__);namespace['fused']=fused
        exec(compile(source[:start]+replacement+source[end:],'<braids-shallow-face-fusion>','exec'),namespace)
        return namespace['install']
    except (OSError,TypeError,ValueError,SyntaxError):
        return original
