"""Isolated FLOW-order shallow-face experiment, not a supported release.

SedThr removes normal capacity BEFORE slope rotation. Hydraulic dryness closes
the FINAL face. This distinction is not bank erosion: truly dry faces remain
closed and the existing shared-face stock limiter still owns Exner transfers.
"""
from types import MethodType
from pathlib import Path
from braids_core.sediment.bend_methods_v1 import digest
from braids_core.sediment.flow2dh_current_bed import Flow2DHCurrentBed


def install(p, *, face_tangent=False):
    if p.time or p.steps or p._open:
        raise ValueError('Install shallow-face experiment before execution')
    if hasattr(p, 'shallow_input_q'):
        raise ValueError('Shallow-face experiment already installed')
    p.shallow_input_q = p.xp.zeros((3, p.n))
    p.shallow_sediment_depth=getattr(p,'sediment_depth',.001)
    previous = p.reconstruct_capacity
    if face_tangent:
        from .sediment_wave import WaveEngine
        p.shallow_engine=WaveEngine(p.engine.backend,p.flow.threads,
            source=Path(__file__).with_suffix('.h'),names=('normal_cell_vector','transverse_face_vector',
                'normal_cell_geometry','normal_cell_vector_cached'))
        g=p.g;geometry=p.xp.zeros((4,p.n))
        p.shallow_engine.launch('normal_cell_geometry',p.n,p.n,p.nf,
            g['offset'],g['faces'],g['normal'],g['length'],geometry)
        # The closure owns immutable mesh data, outside mutable trial snapshots.
        # Keep the original launch ABI for composed activity and audit adapters.
        original_launch=p.shallow_engine.launch
        def launch(name,count,*args):
            if name=='normal_cell_vector':
                return original_launch('normal_cell_vector_cached',count,*args[:-2],geometry,*args[-2:])
            return original_launch(name,count,*args)
        p.shallow_engine.launch=launch
        p.shallow_vector=p.xp.zeros((2,p.n))
        if hasattr(p,'wave_base_reconstruction'):
            p.wave_physical_normal=p.xp.zeros(p.nf)

    def reconstruct(owner):
        previous()
        g = owner.g; xp = owner.xp; o = g['owner']; j = g['neighbour']
        h = owner.shallow_input_q[0]
        batch=getattr(owner,'_river_hydraulic_batch',None)
        active=None if batch is None else batch.get('sediment_active')
        if active is None:
            active = (h[o] > owner.shallow_sediment_depth) & xp.where(
                j >= 0, h[xp.maximum(j, 0)] > owner.shallow_sediment_depth, True)
            if batch is not None:batch['sediment_active']=active
        if face_tangent:
            # The hydraulic wave correction must NOT become transverse physical
            # capacity on another face. Gather the original MUSCL normals.
            raw=(owner.wave_physical_normal.copy() if hasattr(owner,'wave_physical_normal')
                 else xp.sum(owner.bedface[:2]*g['normal'],axis=0))
            raw*=active
            raw[(j<0)&(g['bid']<0)]=0.
            owner.shallow_engine.launch('normal_cell_vector',owner.n,owner.n,owner.nf,
                g['offset'],g['faces'],g['normal'],g['length'],raw,owner.shallow_vector)
            owner.shallow_engine.launch('transverse_face_vector',owner.nf,owner.nf,owner.n,
                o,j,g['normal'],owner.shallow_vector,owner.bedface)
        normal = xp.sum(owner.bedface[:2] * g['normal'], axis=0)
        # Retain the reconstructed transverse capacity. Only its physical
        # slope rotation can create a normal flux on a sediment-inactive face.
        owner.bedface[:2] -= g['normal'] * xp.where(active, 0., normal)
        if hasattr(owner, 'wave_diagnostic'):
            owner.wave_diagnostic[:, ~active] = 0.

    def bed_faces(owner, q, z, hi):
        batch=getattr(owner,'_river_hydraulic_batch',None)
        prepared=None if batch is None else batch.get('shallow_state')
        if prepared is None:
            owner.shallow_input_q[:] = q
            owner.z0[:] = z
            if hasattr(owner, 'wave_q'):owner.wave_q[:] = q
            if hasattr(owner,'safe'):
                safe, active = owner.safe(q, owner.shallow_sediment_depth)
            else:
                # Fully wet historical callers still reject a shoreline.
                if bool(owner.xp.any(q[0]<=owner.shallow_sediment_depth)):
                    raise ValueError('Fully wet regression cannot admit a shoreline')
                safe=q;active=owner.xp.ones(owner.n,dtype=bool)
            prepared=(safe,owner.xp.where(active,hi,0.))
            if batch is not None:batch['shallow_state']=prepared
        safe,safe_hi=prepared
        owner._effective_physical_q=q
        try:Flow2DHCurrentBed.bed_faces(owner, safe, z, safe_hi)
        finally:del owner._effective_physical_q
        g = owner.g; o = g['owner']; j = g['neighbour']; xp = owner.xp
        both=None if batch is None else batch.get('hydraulic_wet')
        if both is None:
            wet = q[0] > owner.parameters.dry
            both = wet[o] & xp.where(j >= 0, wet[xp.maximum(j, 0)], True)
            if batch is not None:batch['hydraulic_wet']=both
        owner.aux[2] *= both

    p.reconstruct_capacity = MethodType(reconstruct, p)
    p.bed_faces = MethodType(bed_faces, p)
    p.identity['shallow_face_order_research'] = dict(
        source_sha256=digest(__file__), public_support=False,
        before_slope='zero normal capacity unless both cells exceed SedThr',
        after_slope='zero final flux unless both cells are hydraulically wet',
        tangent='normal-face vector gather (Cartesian four-face mean)' if face_tangent else 'existing second-order BRAIDS cell capacity',
        runtime=p.shallow_engine.identity if face_tangent else None,
        stock='unchanged conservative donor limiter', dry_bank_transfer=False)
    return p
