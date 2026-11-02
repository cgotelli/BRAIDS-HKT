"""Resident GPU active-layer exchange for ordered substrates.

This component implements the native fixed-target burial/exhumation order for
two or more substrate layers. ResidentSedimentTransport invokes it for those
configurations; the existing qualified one-substrate kernels remain unchanged.
The coupled solver must still accept multilayer input before this is an
end-to-end multilayer simulation.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import re
import numpy as np

HERE=Path(__file__).resolve().parent
OPTIONS=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true','--ftz=false')
SOURCE_POLICY_MARKER='#define BRAIDS_STRATIGRAPHY_EXHUMATION_POLICY 3'
EXHUMATION_POLICY='symmetric_64eps_exchange_before_repack_v2'
PREVIOUS_EXHUMATION_POLICY='actual_positive_transfer_before_repack_v1'
PREVIOUS_SOURCE_SHA256=frozenset(('60aa8be6dc2d686b8ae6cb4eb606c3fc5164ee27f35220b40dbf2e101e35ad05',))
LEGACY_EXHUMATION_POLICY='unconditional_exhumation_branch_repack_v1'
LEGACY_SOURCE_SHA256=frozenset(('49ffa7f2fced3603476754f8b2d874a708bc961bae1df55681ed9154203b42bf',))


def exhumation_source_policy(source):
    """Identify paired current/explicit frozen behavior without rewriting it."""
    payload=source if isinstance(source,bytes) else source.encode('utf-8')
    source=payload.decode('utf-8')
    directive,name,version=SOURCE_POLICY_MARKER.split()
    prefix=r'\s*#\s*'+re.escape(directive[1:])+r'\s+'+re.escape(name)
    declarations=re.findall(r'^'+prefix+r'\b[^\n]*',source,flags=re.MULTILINE)
    if not declarations:
        if hashlib.sha256(payload).hexdigest() not in LEGACY_SOURCE_SHA256:
            raise ValueError('Unmarked stratigraphy source is not an authenticated legacy source')
        return LEGACY_EXHUMATION_POLICY
    if len(declarations)!=1:
        raise ValueError('Unknown or malformed stratigraphy exhumation policy')
    if re.fullmatch(prefix+r'\s+'+re.escape(version)+r'\s*',declarations[0]):
        return EXHUMATION_POLICY
    if (re.fullmatch(prefix+r'\s+2\s*',declarations[0]) and
            hashlib.sha256(payload).hexdigest() in PREVIOUS_SOURCE_SHA256):
        return PREVIOUS_EXHUMATION_POLICY
    raise ValueError('Unknown or malformed stratigraphy exhumation policy')


class ResidentMultilayerStratigraphy:
    """Layers are contiguous FP64 (1+nsubstrate, ngrain, ncell), active first."""

    def __init__(self,ncell,bulk,active_target,substrate_thickness,nsubstrate_layers,*,source_path=None):
        if not isinstance(ncell,(int,np.integer)) or ncell<1:
            raise ValueError('Positive cell count required')
        if not isinstance(nsubstrate_layers,(int,np.integer)) or not 2<=nsubstrate_layers<=32:
            raise ValueError('Ordered multilayer component requires 2-32 substrate layers')
        bulk=np.asarray(bulk,dtype=np.float64)
        if bulk.ndim!=1 or not 1<=len(bulk)<=32 or not np.isfinite(bulk).all() or np.any(bulk<=0):
            raise ValueError('Expected 1-32 positive finite bulk densities')
        if not np.isfinite([active_target,substrate_thickness]).all() or min(active_target,substrate_thickness)<=0:
            raise ValueError('Positive finite active/substrate targets required')
        self.source_path=Path(source_path or HERE/'sediment_stratigraphy.cu')
        payload=self.source_path.read_bytes()
        self.exhumation_policy=exhumation_source_policy(payload)
        if source_path is None and self.exhumation_policy!=EXHUMATION_POLICY:
            raise ValueError('Current stratigraphy source lacks its exhumation policy')
        self.source_receipt=dict(schema=1,source_sha256=hashlib.sha256(payload).hexdigest(),
            exhumation_policy=self.exhumation_policy,
            legacy_frozen_source=self.exhumation_policy!=EXHUMATION_POLICY)
        source=payload.decode('utf-8')
        import cupy as cp
        self.cp=cp;self.nc=int(ncell);self.ng=len(bulk);self.ns=int(nsubstrate_layers)
        self.bulk=cp.asarray(np.ascontiguousarray(bulk))
        self.target=np.float64(active_target);self.substrate_thickness=np.float64(substrate_thickness)
        names=('strat_rebalance','strat_bed')
        self.module=cp.RawModule(code=source,options=OPTIONS,name_expressions=names)
        self.k={name:self.module.get_function(name) for name in names}
        self.scratch=cp.empty((1+self.ns,self.ng,self.nc),cp.float64)
        self.status=cp.empty(self.nc,cp.int32)
        self.bed_status=cp.zeros(self.nc,cp.int32)
        self.bed=cp.empty(self.nc,cp.float64)

    def _array(self,value,shape):
        if not isinstance(value,self.cp.ndarray) or value.shape!=shape or value.dtype!=self.cp.float64 or not value.flags.c_contiguous:
            raise ValueError('Expected contiguous FP64 device array '+str(shape))

    def rebalance(self,layers):
        """Mutate the trial layers; caller must reject the trial on nonzero status."""
        self._array(layers,(1+self.ns,self.ng,self.nc))
        self.k['strat_rebalance'](((self.nc+127)//128,),(128,),
            (np.int32(self.nc),np.int32(self.ng),np.int32(self.ns),layers,self.bulk,
             self.target,self.substrate_thickness,self.scratch,self.status))
        return self.status

    def bed_elevation(self,layers,rock,out=None):
        self._array(layers,(1+self.ns,self.ng,self.nc));self._array(rock,(self.nc,))
        out=self.bed if out is None else out;self._array(out,(self.nc,))
        self.k['strat_bed'](((self.nc+127)//128,),(128,),
            (np.int32(self.nc),np.int32(self.ng),np.int32(self.ns),layers,rock,self.bulk,out,self.bed_status))
        return {'bed':out,'status':self.bed_status}
