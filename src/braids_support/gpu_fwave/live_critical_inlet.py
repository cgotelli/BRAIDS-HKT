"""Validation-only composition of the resolved critical root and mode9 trace.

No numerical kernel is added: the frozen root runs before the unchanged
all-pool discharge trace on every evaluation. A critical CONTINUOUS section
does not imply critical midpoint-strip or individual face flow. In particular,
a wet continuous section can have no wet face midpoint and fail the trace.
This is not a coupled boundary, an accepted Riemann flux, or a production API.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType

from . import live_critical_resolved as root_module
from . import live_normal_trace as trace_module

require = root_module.base.require
FROZEN = MappingProxyType({
    'root_python': 'fd3ab504719e340bdc3f067c1b638b0a16df798f9d4befda00dfe93dac30f2e3',
    'root_cuda': 'cebf9f74aa06fec4eea8e581c5ddb422a7a610d585274e5c616e3ac1e98ffc2a',
    'trace_python': '3639789161b491565edebe80b1d87f3624ca08c2f7aa9bb91464a16ed1101d95',
    'trace_cuda': '4207923ca4bb67611406f4fd5b4ea61a54b76f81212805d941950a7c1e11d474',
    'metric_cuda': root_module.base.NORMAL_CUDA_SHA,
})


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_sources(*, root_source_path=None, metric_source_path=None, trace_source_path=None):
    """Archive relocation is legal; different numerical bytes are not."""
    paths = dict(root_python=Path(root_module.__file__), trace_python=Path(trace_module.__file__),
        root_cuda=Path(root_source_path) if root_source_path else Path(root_module.__file__).with_suffix('.cu'),
        trace_cuda=Path(trace_source_path) if trace_source_path else Path(trace_module.__file__).with_suffix('.cu'),
        metric_cuda=Path(metric_source_path) if metric_source_path else Path(root_module.base.normal.__file__).with_suffix('.cu'))
    for key, path in paths.items():
        require(path.is_file() and _sha(path) == FROZEN[key], 'Frozen critical-inlet dependency changed: '+key)
    return MappingProxyType({key: path.resolve() for key, path in paths.items()})


class ResidentCriticalInlet:
    """Same-bed/Q, fixed-pointer, device-resident critical root then trace.

    Only bed, hydraulic q and prescribed Q values may change in-place. Root and
    trace output arrays are private by contract: callers must not modify them.
    Static geometry is detached/immutable, with explicit verify_geometry() for
    byte auditing outside capture. Before externally captured graph replay the
    caller MUST invoke check_storage(); raw graph.launch bypasses Python guards.
    Both status namespaces remain exposed: successful root status alone never
    authorizes the trace. Trace status0 implies this evaluation's root succeeded.
    """
    def __init__(self, geometry, controls, bed_device, q_device, discharge_device, *,
                 gravity, dry, speed_limit, root_source_path=None, metric_source_path=None,
                 trace_source_path=None, maximum_workspace_bytes=512*1024*1024):
        self._sources = validate_sources(root_source_path=root_source_path,
            metric_source_path=metric_source_path, trace_source_path=trace_source_path)
        # Validate the shared host contract before any component allocation.
        shared = trace_module.validate_geometry(geometry)
        canonical = dict(shared, production_section_geometry=True)
        root_module.base.validate_controls(shared, controls)
        root_module.base.validate_gravity(gravity)
        trace_module.validate_controls(dry, speed_limit)
        self.bed, self.q, self.discharge = bed_device, q_device, discharge_device
        self.root = root_module.ResidentResolvedCriticalSection(canonical, controls,
            self.bed, self.discharge, gravity=gravity,
            source_path=self._sources['root_cuda'], metric_source_path=self._sources['metric_cuda'],
            maximum_workspace_bytes=maximum_workspace_bytes)
        self.trace_component = trace_module.ResidentNormalTrace(canonical, self.bed, self.q,
            self.root.outputs['stage'], self.root.outputs['status'], self.discharge,
            dry=dry, speed_limit=speed_limit, source_path=self._sources['trace_cuda'])
        self.cp, self.device_id = self.root.cp, self.root.device_id
        self.root_outputs, self.trace_outputs = self.root.outputs, self.trace_component.outputs
        self.outputs = MappingProxyType(dict(root=self.root_outputs, trace=self.trace_outputs))
        require(all(root_module.base.identity(value) == trace_module.identity(self.trace_component.host[key])
                    for key, value in self.root.host.items()), 'Root/trace geometry differs')
        self._bindings = self._binding_identity()
        self._python_sha = _sha(__file__)
        self.check_storage()

    def _binding_identity(self):
        return tuple(id(value) for value in (self.root, self.trace_component,
            self.bed, self.q, self.discharge, self.root_outputs, self.trace_outputs, self.outputs))

    def check_storage(self):
        require(self.cp.cuda.Device().id == self.device_id, 'Current CUDA device differs from critical inlet')
        require(self._binding_identity() == self._bindings, 'Captured critical-inlet component bindings changed')
        r, t = self.root, self.trace_component
        require(r.device_id == t.device_id == self.device_id, 'Root/trace CUDA devices differ')
        require(r.bed is t.bed is self.bed and r.discharge is t.discharge is self.discharge
            and t.q is self.q and t.stage is r.outputs['stage'] and t.root_status is r.outputs['status'],
            'Critical root and trace must share the same live bed/Q/stage/status bindings')
        require(self.root_outputs is r.outputs and self.trace_outputs is t.outputs
            and self.outputs['root'] is r.outputs and self.outputs['trace'] is t.outputs,
            'Captured critical-inlet output bindings changed')
        r.check_storage(); t.check_storage()
        return True

    def evaluate(self):
        self.check_storage()
        self.root.solve()
        self.trace_component.evaluate()
        return self.outputs

    def invalidate_cache(self):
        self.check_storage()
        self.root.invalidate_cache()

    def verify_geometry(self):
        self.check_storage()
        self.root.verify_geometry(); self.trace_component.verify_geometry()
        require(_sha(__file__) == self._python_sha and
            all(_sha(path) == FROZEN[key] for key, path in self._sources.items()),
            'Critical-inlet wrapper or frozen numerical dependency changed')
        return True

    def receipt(self):
        return dict(backend='gpu', automatic_CPU_fallback=False, precision='FP64',
            source=str(Path(__file__).resolve()), python_sha256=self._python_sha,
            dependencies={key: dict(path=str(path), sha256=FROZEN[key]) for key,path in self._sources.items()},
            root=self.root.receipt(), trace=self.trace_component.receipt(),
            control=dict(mode=9, section_control=2), same_bed_Q_geometry_device_enforced=True,
            sequence='Every evaluate: storage guards, critical root.solve(), unchanged all-pool trace.evaluate()',
            exact_root_cache_retained=True, root_status_namespace=root_module.STATUS_BITS,
            trace_status_namespace=trace_module.STATUS_BITS,
            usable_trace='Require trace status0; root success alone is insufficient',
            continuous_criticality='g*A_continuous^3=Q^2*T_continuous; Q0 is a separate dry convention',
            trace_closure='Uniform inward velocity Q/sum(h_midpoint*length) over h_midpoint>dry; zero tangent',
            discrete_strip_or_per_face_Froude_one_claimed=False,
            zero_discrete_area_policy='Positive Q fails explicitly even if continuous root is wet/critical',
            current_device_guard=True, raw_graph_replay_requires_check_storage=True,
            new_CUDA_or_numerical_method=False,
            root_matrix_qualification='Resolved v1 retains failed nonwinning candidate checks; composition is development only',
            accepted_Riemann_flux_qualified=False, coupled_implemented=False,
            critical_outlet_implemented=False, production_guard_changed=False,
            performance_qualified=False, physical_validation=False)
