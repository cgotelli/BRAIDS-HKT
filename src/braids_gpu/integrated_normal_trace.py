"""Resident root-consistent normal-boundary face integration, version 1.

This replaces only the surveyed normal inlet of the general live-bank route.
Frozen midpoint closures remain available to reproduce historical results.
The root's geometry is authoritative; there is no stored subcell water state.
"""
from pathlib import Path
import hashlib

import numpy as np

from braids_support.gpu_fwave.live_normal_trace import ResidentNormalTrace, require


RULE = 'root_linear_face_moments_owner_hydrostatic_path_v1'
NATIVE_MODE9_FACE_POLICY = 'native_mode9_midpoint_depth_uniform_speed_sampled_pressure_v1'
FACE_POLICIES = (RULE, NATIVE_MODE9_FACE_POLICY)


class ResidentIntegratedNormalTrace(ResidentNormalTrace):
    def __init__(self, geometry, bed_device, q_device, stage_device,
                 root_status_device, discharge_device, *, nodes_device,
                 dry, speed_limit, source_path=None, face_policy=RULE):
        import cupy as cp
        expected = len(geometry['faces']) + len(geometry['offsets']) - 1
        require(isinstance(nodes_device, cp.ndarray) and
                nodes_device.shape == (expected,) and nodes_device.dtype == cp.float64
                and nodes_device.flags.c_contiguous and
                nodes_device.device.id == cp.cuda.Device().id,
                'Root-consistent trace needs canonical resident root nodes')
        self.nodes = nodes_device
        require(face_policy in FACE_POLICIES,
                'Unknown surveyed normal face-distribution policy')
        self.face_policy = face_policy
        self._face_policy_code = np.int32(
            FACE_POLICIES.index(face_policy))
        source_path = (Path(source_path) if source_path else
            Path(__file__).resolve().parents[1] /
            'braids_support/gpu_fwave/integrated_normal_trace.cu')
        super().__init__(geometry, bed_device, q_device, stage_device,
            root_status_device, discharge_device, dry=dry,
            speed_limit=speed_limit, source_path=source_path)
        for name in ('depth_mean', 'depth_moment2', 'depth_max',
                     'owner_depth_moment2', 'pressure_correction'):
            self.outputs[name] = cp.zeros(self.nf, cp.float64)
        self.kernels['lnt_flux'] = self.module.get_function('lnt_flux')
        self._storage = self._storage_identity()
        self._extension_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self._flux_storage = None

    def _storage_identity(self):
        a = self.nodes
        return super()._storage_identity() + (('root_nodes', id(a), a.data.ptr,
            a.shape, a.dtype.str, bool(a.flags.c_contiguous), a.device.id),)

    def evaluate(self):
        self.check_storage()
        a, o = self._arrays, self.outputs
        o['trace'].fill(0); o['face_status'].fill(0)
        self.work['invalid'].fill(0)
        self.kernels['lnt_inputs'](((self.nc+127)//128,), (128,),
            (self.bed, self.q, np.int32(self.nc), self.work['invalid']))
        self.kernels['lnt_apply']((self.nb,), (128,), (
            a['owner'], a['faces'], a['offsets'], a['length'], a['normal'],
            a['mid_offset'], self.nodes, self.bed, self.q, self.stage, self.root_status,
            self.discharge, self.work['invalid'], np.int32(self.nc),
            np.int32(self.nf), np.int32(self.nb), self._face_policy_code,
            self.dry, self.speed_limit,
            o['trace'], o['status'], o['face_status'], o['discrete_wet_area'],
            o['speed'], o['inward_discharge'], o['depth_mean'],
            o['depth_moment2'], o['depth_max'], o['owner_depth_moment2'],
            o['pressure_correction']))
        return o

    def apply_flux(self, buffers, gravity):
        """Overwrite only normal faces before stress and conservative draining."""
        self.check_storage()
        require(np.isscalar(gravity) and np.isfinite(gravity) and gravity > 0,
                'Positive finite boundary gravity required')
        cp = self.cp
        storage = []
        for name, shape, dtype in (('fl', (3, self.nf), cp.float64),
            ('fr', (3, self.nf), cp.float64), ('speed', (self.nf,), cp.float64),
            ('face_status', (self.nf,), cp.int32)):
            value = buffers[name]
            require(isinstance(value, cp.ndarray) and value.shape == shape
                    and value.dtype == dtype and value.flags.c_contiguous
                    and value.device.id == self.device_id,
                    'Canonical integrated boundary flux buffer required: '+name)
            storage.append((name, id(value), value.data.ptr, value.nbytes))
        storage = tuple(storage)
        if self._flux_storage is None:
            def span(value):
                return value.data.ptr, value.data.ptr + value.nbytes
            targets = [buffers[name] for name, *_ in storage]
            inputs = [self.nodes, self.bed, self.q, self.stage, self.root_status,
                self.discharge, *self._arrays.values(), *self.outputs.values(),
                *self.work.values()]
            for index, target in enumerate(targets):
                lo, hi = span(target)
                require(all(not (lo < span(other)[1] and span(other)[0] < hi)
                        for other in inputs + targets[:index]),
                        'Integrated normal flux destination storage overlaps')
            self._flux_storage = storage
        require(storage == self._flux_storage,
                'Integrated normal flux destination storage changed')
        a, o = self._arrays, self.outputs
        self.kernels['lnt_flux']((self.nb,), (128,), (
            a['faces'], a['offsets'], a['normal'], o['trace'], o['status'],
            o['depth_moment2'], o['depth_max'], o['pressure_correction'],
            o['speed'], np.int32(self.nb), np.int32(self.nf), np.float64(gravity),
            self.dry, buffers['fl'], buffers['fr'], buffers['speed'],
            buffers['face_status']))

    def receipt(self):
        result = super().receipt()
        native = self.face_policy == NATIVE_MODE9_FACE_POLICY
        result.update(rule=RULE, face_distribution_policy=self.face_policy,
            coupled_implemented=True,
            production_section_geometry=True,
            trace_geometry=('native mode9 midpoint sampled depth' if native else
                            'exact positive-depth mean over the current root nodes'),
            pressure_geometry=('native mode9 midpoint hydrostatic pressure' if native else
                               'exact second depth moment over the same root nodes'),
            bed_source=('g/2*(h_owner+h_trace)*boundary_bed_offset; zero for dry owner'
                        if native else
                        'g/2*(h_owner^2-mean(max(eta_owner-z(s),0)^2)); zero for dry owner'),
            dry_policy=(
                'native mode9 zeroes sampled trace, pressure moment, and wave speed at h<=dry'
                if native else
                'transport only where face mean depth exceeds unchanged dry threshold'),
            cfl_depth=('native mode9 sampled midpoint depth' if native else
                       'maximum segment depth, not its mean'),
            native_equivalence=native, performance_qualified=False,
            extension_sha256=self._extension_sha, physical_validation=False)
        return result

    def verify_geometry(self):
        super().verify_geometry()
        require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == self._extension_sha,
                'Integrated normal trace Python source changed')
        return True


def integrated_boundary(Base, *, face_policy=RULE):
    """Reuse normal root, schedule, status latches, outlet and pointer guards."""
    require(face_policy in FACE_POLICIES,
            'Unknown surveyed normal face-distribution policy')
    class IntegratedNormalBoundary(Base):
        def _normal_finish_binding(self):
            super()._normal_finish_binding()
            if self._normal_ready_storage is None:
                return
            self.normal_trace = ResidentIntegratedNormalTrace(
                self._normal_geometry, self.inputs['bed'], self._normal_q,
                self.normal_root.outputs['stage'], self.normal_root.outputs['status'],
                self._normal_Q, nodes_device=self.normal_root.outputs['nodes'],
                dry=float(self.dry), speed_limit=float(self.speed),
                source_path=self._normal_bridge_source.parent/'integrated_normal_trace.cu',
                face_policy=face_policy)

        def integrated_flux(self, buffers):
            self.normal_trace.apply_flux(buffers, self.grav)

        def receipt(self):
            result = super().receipt()
            result['normal'].update(sampled_pressure_flux_body_unchanged=False,
                section_flux_rule=RULE,
                face_distribution_policy=face_policy,
                section_flux_hook='after hydraulic assembly before secondary stress and draining')
            return result

    return IntegratedNormalBoundary
