"""Isolated volume-consistent affine geometry and conservative edge weighting.

Area fractions are not face-length fractions. No sediment-only weighting can
qualify a change of hydraulic geometry by itself. This candidate remains
research-only and preserves the existing external boundary contracts.
"""
from pathlib import Path
from types import MethodType
from .sediment_wave import WaveEngine
from braids_core.sediment.bend_methods_v1 import digest

NAMES=('exposure_gradient','exposure_cells','exposure_faces','exposure_scale')


def engine(backend,threads=4):
    return WaveEngine(backend,threads,source=Path(__file__).with_suffix('.h'),names=NAMES)


def install(a):
    p=a.physics;xp=a.xp;g=p.g
    if p.time or p.steps or p._open or hasattr(p,'exposure_engine'):raise ValueError('Install once before execution')
    p.exposure_engine=engine(a.backend,p.flow.threads)
    p.exposure_gradient=xp.zeros((2,p.n));p.exposure_cell=xp.zeros((4,p.n));p.exposure_face=xp.ones((2,p.nf))
    p.exposure_unweighted=xp.zeros(p.nf)
    dx=float(a.data['dx'])
    if not bool(xp.all(xp.abs(g['area']-dx*dx)<1e-12)):raise ValueError('Square full-cell research only')
    def geometry(q,z):
        e=p.exposure_engine
        e.launch('exposure_gradient',p.n,p.n,z,g['offset'],g['faces'],g['owner'],g['neighbour'],g['dxdy'],g['inverse'],p.exposure_gradient)
        e.launch('exposure_cells',p.n,p.n,dx,dx,.001,q[0],z,p.exposure_gradient,p.exposure_cell)
        e.launch('exposure_faces',p.nf,p.nf,p.n,g['owner'],g['neighbour'],g['normal'],g['length'],g['displacement'],z,p.exposure_gradient,p.exposure_cell,p.exposure_face)
    p.exposure_geometry=geometry;previous=p.bed_faces
    def faces(owner,q,z,hi):
        previous(q,z,hi);geometry(q,z);owner.exposure_unweighted[:]=owner.aux[2]
        owner.exposure_engine.launch('exposure_scale',owner.nf,owner.nf,owner.exposure_face[0],owner.aux[2])
    p.bed_faces=MethodType(faces,p)
    p.identity['wet_exposure_research']=dict(runtime=p.exposure_engine.identity,wrapper_sha256=digest(__file__),
        method='affine_cell_volume_inversion_and_exact_shared_edge_wet_length',
        hydraulic_operator_changed=False,bank_transfer_changed=False,physical_laws_changed=False,public_support=False,
        limitation='sediment-only geometry probe, not a consistent subcell hydraulic model; no automatic promotion')
    return a
