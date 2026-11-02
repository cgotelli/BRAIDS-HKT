"""Augmentation-only internal DensIn seam: its hydraulic caller projects once.

This additive API avoids running a discarded diagnostic projection inside each
real residual. The qualified complete projection helper remains unchanged.
"""
from pathlib import Path

import numpy as np

from .density_active_projection_v97 import _array,_digest,_hash,_sha
from .density_trace_projection_v98 import (
    SOURCE,BASE_SOURCE,FROZEN_BASE_SHA,build_native_density_trace_projection_v98,
    validate_density_trace_inputs_v98,
)

ABI="braids-v98-density-trace-face-augmentation-only-v1"


def make_density_trace_faces_result_v98(geometry,inputs,provenance,acceleration,mask,*,
                                      runtime_sha,backend,threads):
    if backend not in ("native_cpu","actual_cuda"):
        raise ValueError("v98 density face-only backend differs")
    h,rho,distance,base,wet,base_mask,faces,th,tr,tw=inputs
    accel=_array(acceleration,(geometry.face_count,),"density trace acceleration")
    active=_array(mask,(geometry.face_count,),"density trace mask",integer=True)
    expected=np.array(base,copy=True);expected_mask=np.array(base_mask,copy=True)
    chosen=(wet[geometry.owner[faces]]!=0)&(tw!=0)
    f,owner=faces[chosen],geometry.owner[faces[chosen]]
    expected[faces]=0.
    expected[f]=-provenance["gravity"]*(tr[chosen]-rho[owner])*(th[chosen]+h[owner])/(
        4*provenance["reference_density"]*distance[f])
    expected_mask[faces]=chosen.astype(np.int32)
    if (not np.array_equal(active,expected_mask) or
        np.any(np.abs(accel-expected)>32*np.finfo(float).eps*np.maximum(np.abs(expected),np.finfo(float).tiny))):
        raise ValueError("v98 density face-only combined equation differs")
    metadata=dict(abi=ABI,backend=backend,geometry_sha256=geometry.fingerprint,
        inputs_sha256=_digest(*inputs),outputs_sha256=_digest(accel,active),
        runtime_sha256=runtime_sha,native_threads=threads,
        trace_face_evaluation_count=int(faces.size),density_projection_evaluation_count=0,
        physical_boundary_surrogate_count=0,v91_mask_modification_count=0,
        numerical_python_cell_face_loops=0,state_or_clock_publication_count=0,
        automatic_cpu_fallback=False,production_admission=False,**provenance)
    return dict(face_acceleration_m_s2=accel,density_face_mask=active,
        metadata=metadata,receipt_sha256=_hash(metadata))


def validate_density_trace_faces_result_v98(result,geometry,midpoint_depth,accepted_density_cell,*,runtime_sha,**kwargs):
    if type(result) is not dict or set(result)!={"face_acceleration_m_s2","density_face_mask","metadata","receipt_sha256"}:
        raise ValueError("v98 density face-only result type differs")
    inputs,_,provenance,_=validate_density_trace_inputs_v98(geometry,midpoint_depth,accepted_density_cell,**kwargs)
    rebuilt=make_density_trace_faces_result_v98(geometry,inputs,provenance,
        result["face_acceleration_m_s2"],result["density_face_mask"],runtime_sha=runtime_sha,
        backend=result["metadata"].get("backend"),
        threads=128 if result["metadata"].get("backend")=="actual_cuda" else kwargs.get("native_threads",4))
    if result["metadata"]!=rebuilt["metadata"] or result["receipt_sha256"]!=rebuilt["receipt_sha256"]:
        raise ValueError("v98 density face-only runtime/input/result identity differs")
    return result


class NativeDensityTraceFacesRuntimeV98:
    def __init__(self,base_runtime):
        self.base_runtime=base_runtime
        self.receipt=dict(abi=ABI,base_runtime_sha256=base_runtime.receipt["receipt_sha256"],
            wrapper_sha256=_sha(Path(__file__)),density_projection_evaluation_count=0,
            production_admission=False)
        self.receipt["receipt_sha256"]=_hash(self.receipt)

    def evaluate(self,geometry,midpoint_depth,accepted_density_cell,**kwargs):
        inputs,lookup,provenance,cancel=validate_density_trace_inputs_v98(
            geometry,midpoint_depth,accepted_density_cell,**kwargs)
        base=self.base_runtime
        if (self.receipt["receipt_sha256"]!=_hash({k:v for k,v in self.receipt.items() if k!="receipt_sha256"}) or
            _sha(Path(__file__))!=self.receipt["wrapper_sha256"] or
            base.receipt["receipt_sha256"]!=self.receipt["base_runtime_sha256"] or
            base.receipt["receipt_sha256"]!=_hash({k:v for k,v in base.receipt.items() if k!="receipt_sha256"}) or
            _sha(SOURCE)!=base.receipt["source_sha256"] or _sha(BASE_SOURCE)!=FROZEN_BASE_SHA):
            raise ValueError("v98 density face-only native runtime/source identity differs")
        # Keep the actual base wrapper and loaded DLL identities authenticated.
        from . import density_trace_projection_v98 as module
        if (_sha(Path(module.__file__))!=base.receipt["wrapper_sha256"] or
            _sha(base.library_path)!=base.receipt["library_sha256"]):
            raise ValueError("v98 density face-only native wrapper/library identity differs")
        h,rho,distance,accel,wet,active,faces,th,tr,tw=inputs
        out=np.empty(geometry.face_count);mask=np.empty(geometry.face_count,np.int32)
        status=np.empty(geometry.face_count,np.int32);threads=kwargs.get("native_threads",4)
        base.function(geometry.cell_count,geometry.face_count,len(faces),geometry.owner,lookup,
            h,rho,distance,accel,active,wet,th,tr,tw,provenance["reference_density"],
            provenance["gravity"],threads,out,mask,status)
        if np.any(status):raise ValueError("v98 density face-only native construction failed")
        if cancel():raise InterruptedError("v98 density face-only cancelled after launch")
        return make_density_trace_faces_result_v98(geometry,inputs,provenance,out,mask,
            runtime_sha=self.receipt["receipt_sha256"],backend="native_cpu",threads=threads)


def build_native_density_trace_faces_v98(build_directory,*,compiler=None):
    return NativeDensityTraceFacesRuntimeV98(build_native_density_trace_projection_v98(build_directory,compiler=compiler))
