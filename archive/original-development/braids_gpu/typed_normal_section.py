"""Resident non-Manning mode-9 normal-depth root for the GPU path.

The historical mode-9 component is intentionally left unchanged because its
``group_manning`` arithmetic is part of frozen validation evidence.  This
additive component consumes the authenticated raw Chezy C or Nikuradse ks
bytes from :mod:`braids_gpu.forcing_profile` and evaluates exactly the same
section conveyance used by the native CPU solver::

    K_group = A_group * C(R_group, parameter) * sqrt(R_group)

where ``R_group=A_group/P_group`` and C is the selected Delft3D C/W
roughness conversion.  No host root, CPU fallback, or equivalent-Manning
conversion is used.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType

import numpy as np

from validation.gpu_fwave import live_normal_section as legacy_root
from validation.gpu_fwave import live_section_metrics as legacy_geometry
from validation.gpu_fwave.time_integration import OPTIONS

from .forcing_profile import BINDING_FIELD, forcing_identity


NAMES = ("tns_inputs", "tns_metrics", "tns_solve")
STATUS_BITS = dict(
    invalid_live_input=1,
    inadmissible_current_profile=2,
    no_normal_root_found=8,
    invalid_arithmetic_or_roughness=16,
)
_MODES = {"chezy": 1, "white_colebrook": 3}
require = legacy_geometry.require
freeze = legacy_geometry.freeze
identity = legacy_geometry.identity


def _validate_geometry(geometry):
    """Validate topology while naming the raw field as a typed parameter."""
    flag = np.asarray(geometry.get("production_section_geometry", False))
    require(flag.shape == () and flag.dtype.kind == "b" and bool(flag),
            "production_section_geometry=True is required for typed mode 9")
    keys = (
        "owner", "faces", "offsets", "length", "left_offset",
        "right_offset", "mid_offset", "roughness_parameter",
    )
    integer_keys = {"owner", "faces", "offsets"}
    data = {}
    for key in keys:
        value = np.asarray(geometry[key])
        dtype = np.int32 if key in integer_keys else np.float64
        require(value.ndim == 1 and value.dtype == dtype and
                value.flags.c_contiguous,
                "Canonical typed section vector required: " + key)
        if key not in integer_keys:
            require(np.isfinite(value).all(),
                    "Nonfinite typed section geometry: " + key)
        data[key] = freeze(value)
    owner, faces, offsets, parameter = (
        data[key] for key in
        ("owner", "faces", "offsets", "roughness_parameter"))
    nc, nf, nbf, nb = len(parameter), len(owner), len(faces), len(offsets)-1
    legacy_geometry.validate_counts(nc, nf, nbf, nb)
    require(offsets[0] == 0 and offsets[-1] == nbf and
            np.all(np.diff(offsets.astype(np.int64)) > 0),
            "Typed section CSR must contain nonempty ordered boundaries")
    require(np.all((owner >= 0) & (owner < nc)) and
            np.all((faces >= 0) & (faces < nf)) and
            len(np.unique(faces)) == nbf,
            "Typed section owner/face topology differs")
    require(np.all(parameter > 0.0),
            "Typed section roughness parameter must be strictly positive")
    for key in ("length", "left_offset", "right_offset", "mid_offset"):
        require(data[key].shape == (nf,),
                "Typed section per-face shape differs: " + key)
    require(np.all(data["length"] > 0.0),
            "Typed section face lengths must be strictly positive")
    return MappingProxyType(data)


def _topology(host):
    offsets, faces, owner = (host[key] for key in
                             ("offsets", "faces", "owner"))
    nb = len(offsets)-1
    parameter = host["roughness_parameter"][owner[faces]]
    starts = np.zeros(len(faces), dtype=bool)
    starts[offsets[:-1]] = True
    starts[1:] |= parameter[1:] != parameter[:-1]
    group_offsets = np.r_[np.flatnonzero(starts), len(faces)].astype(np.int32)
    boundary_group_offsets = np.r_[
        np.int32(0), np.cumsum(starts, dtype=np.int32)][offsets]
    face_boundary = np.repeat(np.arange(nb, dtype=np.int32), np.diff(offsets))
    return MappingProxyType({key: freeze(value) for key, value in dict(
        node_offsets=offsets + np.arange(nb+1, dtype=np.int32),
        face_boundary=face_boundary,
        group_offsets=group_offsets,
        boundary_group_offsets=boundary_group_offsets,
        group_boundary=face_boundary[group_offsets[:-1]],
        face_group=np.repeat(np.arange(len(group_offsets)-1, dtype=np.int32),
                             np.diff(group_offsets)),
        group_parameter=parameter[group_offsets[:-1]],
    ).items()})


class ResidentTypedNormalSection:
    """Device-resident typed C/W conveyance root with fixed storage."""

    def __init__(self, geometry, controls, bed_device, discharge_device, *,
                 binding, source_path, roughness_source_path,
                 metric_stage_device=None,
                 maximum_workspace_bytes=512*1024*1024):
        self.binding = dict(binding)
        mode = self.binding.get("hydraulic_roughness_mode")
        require(mode in _MODES,
                "Typed normal section requires Chezy or White-Colebrook")
        self.roughness_mode = np.int32(_MODES[mode])
        self.gravity = np.float64(controls.pop("gravity"))
        self.minimum_work_depth = np.float64(
            self.binding["effective_minimum_work_depth_m"])
        self.waqua = np.int32(bool(
            self.binding["white_colebrook_chzk2d"]))
        require(np.isfinite(self.gravity) and self.gravity > 0.0 and
                np.isfinite(self.minimum_work_depth) and
                self.minimum_work_depth > 0.0,
                "Typed normal section controls are invalid")
        self.host = _validate_geometry(geometry)
        # The frozen control validator has no roughness arithmetic; reuse its
        # exact mode/slope/seed checks with our canonical topology.
        control_geometry = {
            "offsets": self.host["offsets"],
            "faces": self.host["faces"],
        }
        self.controls = legacy_root.validate_controls(control_geometry, controls)
        self.topology = _topology(self.host)
        self.layout = legacy_root.workspace_layout(
            control_geometry, maximum_workspace_bytes=maximum_workspace_bytes)
        self.nc = len(self.host["roughness_parameter"])
        self.nf = len(self.host["owner"])
        self.nbf = len(self.host["faces"])
        self.nb = len(self.host["offsets"])-1
        self.ng = len(self.topology["group_parameter"])

        import cupy as cp
        self.cp = cp
        self.device_id = cp.cuda.Device().id
        self.bed = bed_device
        self.discharge = discharge_device
        self.metric_stage = metric_stage_device
        inputs = [("bed", self.bed, (self.nc,)),
                  ("discharge", self.discharge, (self.nb,))]
        if self.metric_stage is not None:
            inputs.append(("metric_stage", self.metric_stage, (self.nb,)))
        for name, value, shape in inputs:
            require(isinstance(value, cp.ndarray) and value.shape == shape and
                    value.dtype == cp.float64 and value.flags.c_contiguous and
                    value.device.id == self.device_id,
                    "Resident typed normal input differs: " + name)

        host = dict(self.host, **self.controls, **self.topology)
        host["level_capacity"] = self.layout["level_capacity"]
        host["level_offsets"] = self.layout["level_offsets"]
        self._host_arrays = MappingProxyType(host)
        self._arrays = {key: cp.asarray(value) for key, value in host.items()}
        self.outputs = self._metric_outputs()
        self.outputs.update(
            stage=cp.zeros(self.nb), residual=cp.empty(self.nb),
            cache_hit=cp.zeros(self.nb, dtype=cp.int32),
            selected_branch=cp.full(self.nb, -1, dtype=cp.int32),
            selected_kind=cp.full(self.nb, -1, dtype=cp.int32),
            bisections=cp.zeros(self.nb, dtype=cp.int32),
            expansions=cp.zeros(self.nb, dtype=cp.int32),
            metric_evaluations=cp.zeros(self.nb, dtype=cp.int32))
        self.metric_outputs = self._metric_outputs()
        self.cache = dict(
            valid=cp.zeros(self.nb, dtype=cp.int32),
            bed=cp.empty(self.nbf), discharge=cp.empty(self.nb),
            stage=cp.empty(self.nb), area=cp.empty(self.nb),
            top_width=cp.empty(self.nb), conveyance=cp.empty(self.nb),
            residual=cp.empty(self.nb),
            selected_branch=cp.empty(self.nb, dtype=cp.int32),
            selected_kind=cp.empty(self.nb, dtype=cp.int32))
        self.work = dict(
            levels=cp.empty(int(self.layout["level_offsets"][-1])),
            global_status=cp.zeros(1, dtype=cp.int32))
        self._source = Path(source_path).resolve()
        self._roughness_source = Path(roughness_source_path).resolve()
        require(self._source.is_file() and not self._source.is_symlink() and
                self._roughness_source.is_file() and
                not self._roughness_source.is_symlink(),
                "Archived typed normal CUDA sources differ")
        self._source_sha = hashlib.sha256(self._source.read_bytes()).hexdigest()
        self._roughness_sha = hashlib.sha256(
            self._roughness_source.read_bytes()).hexdigest()
        self._python_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        code = (self._roughness_source.read_text(encoding="utf-8") + "\n" +
                self._source.read_text(encoding="utf-8"))
        self.module = cp.RawModule(code=code, options=OPTIONS,
                                   name_expressions=NAMES)
        self.kernels = {key: self.module.get_function(key) for key in NAMES}
        self._array_identity = {key: identity(value)
                                for key, value in host.items()}
        self._control_identity = (
            int(self.roughness_mode), float(self.gravity),
            float(self.minimum_work_depth), int(self.waqua),
            self.binding["fingerprint"])
        self._storage = self._storage_identity()

    def _metric_outputs(self):
        cp = self.cp
        return dict(
            nodes=cp.empty(self.nbf+self.nb),
            node_offsets=self._arrays["node_offsets"],
            area=cp.empty(self.nb), top_width=cp.empty(self.nb),
            conveyance=cp.empty(self.nb),
            status=cp.zeros(self.nb, dtype=cp.int32))

    def _storage_identity(self):
        values = [("bed", self.bed), ("discharge", self.discharge)]
        if self.metric_stage is not None:
            values.append(("metric_stage", self.metric_stage))
        values += [(prefix+key, value) for prefix, mapping in (
            ("geometry:", self._arrays), ("root:", self.outputs),
            ("metric:", self.metric_outputs), ("cache:", self.cache),
            ("work:", self.work)) for key, value in mapping.items()]
        return tuple((key, id(value), value.data.ptr, value.shape,
                      value.dtype.str, bool(value.flags.c_contiguous),
                      value.device.id) for key, value in values)

    def check_storage(self):
        require(self.cp.cuda.Device().id == self.device_id,
                "Current CUDA device differs from typed normal section")
        require(self._storage_identity() == self._storage,
                "Typed normal-section storage changed")
        require((int(self.roughness_mode), float(self.gravity),
                 float(self.minimum_work_depth), int(self.waqua),
                 self.binding["fingerprint"]) == self._control_identity,
                "Typed normal-section controls changed")
        return True

    def invalidate_cache(self):
        self.check_storage()
        self.cache["valid"].fill(0)

    def _validate_live(self, value, outputs, nonnegative):
        outputs["status"].fill(0)
        self.work["global_status"].fill(0)
        self.kernels["tns_inputs"](
            ((max(self.nc, self.nb)+127)//128,), (128,),
            (self.bed, value, np.int32(self.nc), np.int32(self.nb),
             np.int32(nonnegative), self.work["global_status"],
             outputs["status"]))

    def _roughness_arguments(self):
        return (self.roughness_mode, self.gravity,
                self.minimum_work_depth, self.waqua)

    def metrics(self):
        self.check_storage()
        require(self.metric_stage is not None,
                "Bind typed normal metric stage at construction")
        a, o = self._arrays, self.metric_outputs
        self._validate_live(self.metric_stage, o, False)
        self.kernels["tns_metrics"]((self.nb,), (128,), (
            a["owner"], a["faces"], a["offsets"], a["face_group"],
            a["group_parameter"], a["length"], a["left_offset"],
            a["right_offset"], self.bed, self.metric_stage,
            self.work["global_status"], np.int32(self.nb),
            *self._roughness_arguments(), o["nodes"], o["area"],
            o["top_width"], o["conveyance"], o["status"]))
        return o

    def solve(self):
        self.check_storage()
        a, o, c = self._arrays, self.outputs, self.cache
        self._validate_live(self.discharge, o, True)
        self.kernels["tns_solve"]((self.nb,), (128,), (
            a["owner"], a["faces"], a["offsets"], a["face_group"],
            a["group_parameter"], a["length"], a["left_offset"],
            a["right_offset"], a["energy_slope"], self.bed,
            self.discharge, self.work["global_status"], a["level_offsets"],
            a["level_capacity"], np.int32(self.nb),
            *self._roughness_arguments(), o["nodes"], self.work["levels"],
            c["valid"], c["bed"], c["discharge"], c["stage"], c["area"],
            c["top_width"], c["conveyance"], c["residual"],
            c["selected_branch"], c["selected_kind"], o["stage"],
            o["area"], o["top_width"], o["conveyance"], o["residual"],
            o["status"], o["cache_hit"], o["selected_branch"],
            o["selected_kind"], o["bisections"], o["expansions"],
            o["metric_evaluations"]))
        return o

    def verify_geometry(self):
        self.check_storage()
        require(all(identity(self.cp.asnumpy(value)) ==
                    self._array_identity[key]
                    for key, value in self._arrays.items()),
                "Typed normal geometry/control bytes changed")
        require(hashlib.sha256(self._source.read_bytes()).hexdigest() ==
                self._source_sha and
                hashlib.sha256(self._roughness_source.read_bytes()).hexdigest()
                == self._roughness_sha and
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest() ==
                self._python_sha,
                "Executed typed normal source identity changed")
        return True

    def receipt(self):
        return dict(
            backend="gpu", automatic_CPU_fallback=False, precision="FP64",
            device_id=self.device_id, source=str(self._source),
            source_sha256=self._source_sha,
            hydraulic_roughness_source_sha256=self._roughness_sha,
            python_sha256=self._python_sha,
            requested_cuda_options=list(OPTIONS),
            immutable_inputs=self._array_identity,
            normal_only=True, mode9_trace_implemented=False,
            production_guard_changed=False,
            roughness_method_id=self.binding["roughness_method_id"],
            roughness_parameter_name=self.binding["roughness_parameter_name"],
            roughness_parameter_units=self.binding["roughness_parameter_units"],
            roughness_parameter_array=dict(
                self.binding["roughness_parameter_array"]),
            forcing_fingerprint=self.binding["fingerprint"],
            equivalent_manning_conversion=False,
            conveyance="sum(A_group*C(A_group/P_group)*sqrt(A_group/P_group))",
            root_control="device cooperative block; no host iteration",
            maximum_faces_per_section=int(np.diff(self.host["offsets"]).max()),
            estimated_dynamic_bytes=self.layout["estimated_dynamic_bytes"],
            maximum_workspace_bytes=self.layout["maximum_workspace_bytes"],
            exact_cache_keys=["discharge equality",
                              "every section face owner bed equality"],
            current_device_guard=True,
            raw_graph_replay_requires_check_storage=True,
            status_bits=STATUS_BITS, physical_validation=False)


def make_resident_normal_section(data, geometry, controls, bed_device,
                                 discharge_device, *, legacy_type,
                                 legacy_source, typed_source,
                                 roughness_source):
    """Select by authenticated physical method, never mesh size or fallback."""
    binding = data.get(BINDING_FIELD)
    if not isinstance(binding, dict) or \
            binding.get("hydraulic_roughness_mode") == "manning":
        return legacy_type(
            geometry, controls, bed_device, discharge_device,
            source_path=legacy_source)
    authenticated = forcing_identity(data)
    require(authenticated["hydraulic_roughness_method_id"] ==
            binding["roughness_method_id"],
            "Typed normal forcing identity differs")
    parameter = np.asarray(data.get("roughness_parameter"))
    require(np.array_equal(parameter, np.asarray(geometry["manning"])),
            "Typed normal geometry does not contain exact raw parameter bytes")
    typed_geometry = dict(geometry)
    typed_geometry.pop("manning")
    typed_geometry["roughness_parameter"] = parameter
    typed_controls = dict(controls, gravity=float(data["grav"]))
    return ResidentTypedNormalSection(
        typed_geometry, typed_controls, bed_device, discharge_device,
        binding=binding, source_path=typed_source,
        roughness_source_path=roughness_source)


__all__ = [
    "ResidentTypedNormalSection", "make_resident_normal_section",
    "STATUS_BITS",
]
