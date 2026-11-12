"""Nested refinement with prepared subcell rock shape and conserved parent means.

This is a mesh-transfer rule, not a replacement hydraulic reconstruction.
Coarse rock must be the area restriction of the fine reference. Only rounding
differences are reconciled; transported columns remain layer-index aligned.
"""
from pathlib import Path
import hashlib
import numpy as np
from .bed_columns_v101 import host,array_sha
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine


class TerrainEngine:
    def __init__(self,backend,threads=4):
        if backend not in ('cpu','gpu'):raise ValueError('Explicit CPU/GPU required')
        self.xp=np
        if backend=='gpu':
            import cupy as cp
            self.xp=cp
        self.engine=WaveEngine(backend,threads,source=Path(__file__).with_suffix('.h'),names=('terrain_prolong',))
        self.identity=self.engine.identity

    def prolong(self,mapping,rock,reference_rock,thickness,q,carried,*,dry):
        xp=self.xp;ns=len(mapping.source_area);nt=len(mapping.target_area)
        if not np.all(np.diff(mapping.offsets)==1) or not np.allclose(mapping.weights,1,atol=1e-12,rtol=0):
            raise ValueError('Terrain prolongation requires nested parent/child cells')
        parent=mapping.indices;ids=np.argsort(parent,kind='stable').astype(np.int32)
        offsets=np.r_[0,np.cumsum(np.bincount(parent,minlength=ns))].astype(np.int32)
        arrays=[xp.ascontiguousarray(v,dtype=xp.float64) for v in (rock,reference_rock,thickness,q,carried)]
        if any(not bool(xp.all(xp.isfinite(v))) for v in arrays) or not np.isfinite(dry) or dry<=0:raise ValueError('Invalid terrain transfer fields')
        if [v.shape for v in arrays]!=[(ns,),(nt,),(nt,),(3,ns),(carried.shape[0],ns)]:raise ValueError('Terrain transfer shape differs')
        if bool(xp.any(q[0]<0)):raise ValueError('Negative source water depth')
        # Accepted hydraulics can retain conservative, sub-dry water traces.
        # Splitting such a parent copies its depth to its children; it must
        # neither manufacture a wetting film nor delete already present water.
        mask=q[0]<=dry
        if bool(xp.any(q[1:,mask])) or bool(xp.any(carried[:,mask])):raise ValueError('Dry parent carries momentum or transported inventory')
        output_rock=xp.empty(nt);output_q=xp.empty((3,nt));output=xp.empty((carried.shape[0],nt));shift=xp.empty(ns)
        self.engine.launch('terrain_prolong',ns,ns,nt,carried.shape[0],float(dry),xp.asarray(offsets),xp.asarray(ids),
            xp.asarray(mapping.target_area),xp.asarray(mapping.source_area),*arrays,output_rock,output_q,output,shift)
        disagreement=float(xp.max(xp.abs(shift)))
        if disagreement>1e-10:raise ValueError('Inconsistent terrain hierarchy: coarse rock must be the area restriction of prepared fine rock')
        if any(not bool(xp.all(xp.isfinite(v))) for v in (output_rock,output_q,output)):raise ValueError('Terrain prolongation failed')
        return output_rock,output_q,output,dict(runtime=self.identity,
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reference_rock_sha256=array_sha(host(reference_rock)),
            max_terrain_mean_disagreement_m=disagreement,
            method='prepared subcell rock relief, conservative parent mean; volume stage solve; depth-weighted transported inventories; sub-dry zero-inventory parent depth copied to children')
