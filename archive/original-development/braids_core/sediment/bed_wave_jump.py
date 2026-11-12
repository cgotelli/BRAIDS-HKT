"""Canonical true-face bed jump for split bed-diagonal numerical viscosity.

Fresh limited LS bed values; physical capacity, hydraulic states, mean waves,
boundary policies and conservative assembly are unchanged. All arithmetic is
shared native CPU/CUDA. Private scratch is local to an owner and is not saved.
"""
from pathlib import Path
import ctypes
from .bend_methods_v1 import digest
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine

METHOD='split_bed_diagonal_independent_true_face_jump_v4'
NAMES=('bed_jump_limits','bed_jump_faces')


def install(owner):
    p=owner.physics;xp=owner.xp;g=p.g
    if owner.time or owner.steps or p._open or hasattr(p,'bed_jump_gradient'):
        raise ValueError('Install canonical bed jump once on a fresh owner')
    engine=WaveEngine(owner.backend,p.flow.threads,source=Path(__file__).with_suffix('.h'),names=NAMES)
    p.bed_jump_gradient=xp.empty((2,p.n));p.bed_jump_value=xp.empty(p.nf)
    wave=p.wave_engine
    # Cache dictionaries can be shared by runtime loaders. This owner alone
    # owns the dispatch wrapper; no other owner or frozen engine is patched.
    wave.fn=dict(wave.fn);original=wave.fn['matrix_bed_faces'];cached=hasattr(wave,'_class_shared')
    def call(*args,**kwargs):
        batch=getattr(p,'_river_hydraulic_batch',None)
        if batch is None or not batch.get('bed_jump_limits',False):
            engine.launch('bed_jump_limits',p.n,p.n,p.nf,p.z0,p.grad[4:6],
                g['offset'],g['faces'],g['owner'],g['neighbour'],g['dxdy'],p.bed_jump_gradient)
            engine.launch('bed_jump_faces',p.nf,p.nf,p.n,p.z0,p.bed_jump_gradient,
                g['owner'],g['neighbour'],g['displacement'],p.bed_jump_value)
            if batch is not None:batch['bed_jump_limits']=True
        # The kernel ABI appends the separate jump before optional class-cache
        # arguments. Hydraulic traces and their opening coefficient stay raw.
        if owner.backend=='cpu':
            values=list(args);index=len(values)-2 if cached else len(values)
            values.insert(index,ctypes.c_void_p(p.bed_jump_value.ctypes.data))
            return original(*values,**kwargs)
        values=list(args[2]);index=len(values)-2 if cached else len(values)
        values.insert(index,p.bed_jump_value)
        return original(args[0],args[1],tuple(values),*args[3:],**kwargs)
    wave.fn['matrix_bed_faces']=call
    p.identity['bed_wave_jump']=dict(method=METHOD,wrapper_sha256=digest(__file__),runtime=engine.identity,
        reconstruction='neighbour-centroid-extrema limited LS bed at actual common face; preserve skewed affine beds, retain extrema jumps',
        geometry='true owner/neighbour displacements and interior adjacency; closed exterior unchanged',
        hydraulic_state='unchanged eta/u/v/z, HR opening and mean waves; separate numerical bed-jump argument',
        cache='derived owner scratch; reuse only inside one same-state class batch',
        physical_capacity_changed=False)
    owner.identity['bed_wave_jump']=p.identity['bed_wave_jump']
    return owner
