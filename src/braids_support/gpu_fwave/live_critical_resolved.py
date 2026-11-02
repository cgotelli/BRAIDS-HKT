"""Isolated resolved-criticality numerical candidate; NOT legacy parity claim.

Keeps the frozen metric geometry, cache and graph interfaces. Changes only the
critical stationary-point calculation and positive-Q root/knot convergence.
Original native CPU and v2 GPU references remain immutable comparisons.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from . import live_critical_section as base

BASE_PY_SHA = '03454e984f9ae31c537df8ddaee55e00f01d90339b630f742842cbd017444f33'
BASE_CUDA_SHA = '88f163eb2ab730c6924476f301c3bf7ef96431b3ab9a3d4602a472d187867c96'
POLICY = dict(relative_discharge_target=1e-10, arithmetic_floor_epsilon_multiplier=32,
    tolerance_formula='1e-10*Q + 32*epsilon(FP64)*max(Q,abs(Qcritical))',
    maximum_bisections=128, adjacent_stage_froude_squared_error=1e-6,
    fixed_stage_width_success=False, native_branch_exclusion_retained=True,
    stationary_derivative='Exact partial-segment sum(length/abs(rise))',
    stationary_admission='Strict one-sided D=3*T*T-A*Tprime sign bracket inside one terrain interval',
    unresolved_any_admissible_interval_fails_boundary=True,
    exact_knots_require_relative_discharge_target=True)
STATUS_BITS = dict(base.STATUS_BITS, unresolved_critical_root=64)
QUALITY = {'0': 'Relative-Q target met', '1': 'Adjacent representable-stage limit; Fr squared gate met',
           '2': 'Zero-discharge dry convention; not Fr1', '-1': 'No usable critical root'}


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def workspace_layout(geometry, *, maximum_workspace_bytes=512*1024*1024):
    layout = base.workspace_layout(geometry, maximum_workspace_bytes=maximum_workspace_bytes)
    extra = 4*(len(geometry['offsets'])-1)+20*int(layout['level_offsets'][-1])+12*int(layout['root_offsets'][-1])
    layout = dict(layout, estimated_dynamic_bytes=layout['estimated_dynamic_bytes']+extra)
    base.require(layout['estimated_dynamic_bytes'] <= maximum_workspace_bytes,
                 'Declared resolved GPU workspace byte capacity exceeded; no CPU fallback')
    return layout


class ResidentResolvedCriticalSection(base.ResidentCriticalSection):
    """Same fixed-pointer API, with quality and per-candidate quality outputs.

    The old native absolute-Q stop is not claimed equivalent. A positive-Q
    solution never succeeds by an absolute stage-width condition alone. If
    adjacent stages cannot satisfy the declared physical Fr tolerance, status64
    makes the boundary unusable; it does not silently choose another branch.
    """
    def __init__(self, geometry, controls, bed_device, discharge_device, *, gravity,
                 metric_stage_device=None, source_path=None, metric_source_path=None,
                 maximum_workspace_bytes=512*1024*1024):
        base.require(_sha(base.__file__) == BASE_PY_SHA
            and _sha(Path(base.__file__).with_suffix('.cu')) == BASE_CUDA_SHA,
            'Frozen original critical component changed')
        host = base.geometry_module.validate_geometry(geometry)
        layout = workspace_layout(host, maximum_workspace_bytes=maximum_workspace_bytes)
        source = Path(source_path) if source_path else Path(__file__).with_suffix('.cu')
        super().__init__(geometry, controls, bed_device, discharge_device, gravity=gravity,
            metric_stage_device=metric_stage_device, source_path=source, metric_source_path=metric_source_path,
            maximum_workspace_bytes=maximum_workspace_bytes)
        self.layout = layout
        self.outputs['quality'] = self.cp.full(self.nb, -1, dtype=self.cp.int32)
        self.outputs['interval_quality'] = self.cp.full(int(layout['level_offsets'][-1]), -1, dtype=self.cp.int32)
        self.outputs['candidate_quality'] = self.cp.full(int(layout['root_offsets'][-1]), -1, dtype=self.cp.int32)
        self.outputs['interval_rate_lo'] = self.cp.full(int(layout['level_offsets'][-1]), self.cp.nan)
        self.outputs['interval_rate_hi'] = self.cp.full(int(layout['level_offsets'][-1]), self.cp.nan)
        self.outputs['candidate_residual'] = self.cp.full(int(layout['root_offsets'][-1]), self.cp.nan)
        self._resolved_python_sha = _sha(__file__)
        self._storage = self._storage_identity()

    def solve(self):
        self.check_storage()
        a, o, c = self._arrays, self.outputs, self.cache
        self._validate_live(self.discharge, o, True)
        self.kernels['lc_solve']((self.nb,), (128,),
            (a['owner'], a['faces'], a['offsets'], a['face_group'], a['group_manning'], a['length'],
             a['left_offset'], a['right_offset'], self.bed, self.discharge, self.work['global_status'], np.float64(self.gravity),
             a['level_offsets'], a['level_capacity'], a['root_offsets'], a['root_capacity'], np.int32(self.nb),
             o['nodes'], c['valid'], c['bed'], c['discharge'],
             o['stage'], o['area'], o['top_width'], o['conveyance'], o['residual'],
             o['status'], o['cache_hit'], o['selected_branch'], o['selected_kind'], o['selected_index'],
             o['nlevels'], o['nterrain'], o['nroots'], o['bisections'], o['metric_evaluations'],
             o['levels'], o['interval_roots'], o['interval_lo'], o['interval_hi'], o['roots'], o['energies'],
             o['candidate_valid'], o['knot_valid'], o['interval_reason'], o['interval_iterations'], o['root_kind'], o['root_branch'],
             o['quality'], o['interval_quality'], o['candidate_quality'],
             o['interval_rate_lo'], o['interval_rate_hi'], o['candidate_residual']))
        return o

    def verify_geometry(self):
        super().verify_geometry()
        base.require(_sha(__file__) == self._resolved_python_sha
            and _sha(base.__file__) == BASE_PY_SHA
            and _sha(Path(base.__file__).with_suffix('.cu')) == BASE_CUDA_SHA,
            'Resolved-critical wrapper or frozen base identity changed')
        return True

    def receipt(self):
        result = super().receipt()
        result.update(base_python_sha256=result['python_sha256'], python_sha256=self._resolved_python_sha,
            base_critical_cuda_sha256=BASE_CUDA_SHA,
            numerical_policy_changed=True, original_absolute_Q_stop_parity_claimed=False,
            resolved_policy=dict(POLICY), quality=dict(QUALITY), status_bits=STATUS_BITS,
            interval_reason=dict(unused=0, excluded_width=1, descending=2, unbracketed=3,
                                 target_converged=4, representability_limited=5, unresolved=6),
            new_diagnostics='quality(nb); interval_quality/rate_lo/rate_hi(level-packed); candidate_quality/residual(root-packed)',
            adjacent_rate_diagnostics='Actual Qcritical at final lo/hi evaluated only on adjacent-stage termination; otherwise NaN',
            mathematical_method_scope='Critical-inlet stationary splitting/convergence only; no f-wave, sediment or normal/outlet changes')
        return result
