"""Native surveyed legacy mode-1 characteristic discharge, resident on CUDA.

This is not the conveyance inlet (mode6) or the surveyed control (mode9).
Only the production whole-section geometry contract is admitted. The unchanged
augmented f-wave kernel supplies the native incoming-bed-step admissibility test.
"""
from pathlib import Path
import numpy as np

RULE = 'sampled_legacy_characteristic_augmented_fan_v1'


def validate_geometry(data):
    def require(ok, message):
        if not ok:
            raise ValueError('Sampled discharge: ' + message)
    modes = np.asarray(data['mode'])
    owner, neighbour = np.asarray(data['owner']), np.asarray(data['neighbour'])
    nf, nb, nc = len(owner), len(modes), np.shape(data['q'])[1]
    flags, delta = np.asarray(data['sampled']), np.asarray(data['bedoffset'])
    offsets, faces, bid = (np.asarray(data[k]) for k in ('boundary_offset', 'boundary_faces', 'bid'))
    require(flags.shape == (nf,) and flags.dtype == np.int32 and np.isin(flags, (0, 1)).all(),
            'invalid physical-face flags')
    require(delta.shape == (nf,) and delta.dtype == np.float64 and np.isfinite(delta).all()
            and not np.any(delta[flags == 0]), 'invalid sampled bed offsets')
    require(owner.shape == neighbour.shape == bid.shape == (nf,) and
            all(v.dtype == np.int32 for v in (owner, neighbour, bid, modes, offsets, faces)),
            'invalid integer topology')
    require(np.all((owner >= 0) & (owner < nc)) and
            np.all((bid >= -1) & (bid < nb)), 'invalid face owner or boundary ID')
    require(offsets.shape == (nb+1,) and offsets[0] == 0 and offsets[-1] == len(faces)
            and np.all(np.diff(offsets) >= 0) and faces.ndim == 1 and
            np.all((faces >= 0) & (faces < nf)) and len(np.unique(faces)) == len(faces),
            'invalid boundary CSR')
    require(np.array_equal(np.sort(faces), np.flatnonzero(bid >= 0)), 'incomplete boundary CSR')
    sampled_faces = np.flatnonzero(flags)
    require(len(sampled_faces) > 0 and np.all(bid[sampled_faces] >= 0) and
            np.all(modes[bid[sampled_faces]] == 1) and np.all(neighbour[sampled_faces] == -1),
            'only exterior legacy mode1 faces may be sampled')
    production = np.asarray(data.get('production_section_geometry'))
    require(production.shape == () and production.dtype.kind in 'iu' and int(production) == 1,
            'whole production section required; historical connected-seed selection is unsupported')
    controls = np.asarray(data['section_control'])
    require(controls.shape == modes.shape and controls.dtype == np.int32 and not controls.any(),
            'legacy mode1 has no mode9 section control')
    selected = np.zeros(nb, np.int32)
    selected[np.unique(bid[sampled_faces])] = 1
    for key in ('section_seed_node', 'boundary_seed_position'):
        value = np.asarray(data[key])
        require(value.shape == (nb,) and value.dtype == np.int32, 'invalid ' + key)
        for b in range(nb):
            count = offsets[b+1]-offsets[b]
            limit = count if key == 'section_seed_node' else count-1
            require((-1 <= value[b] <= limit) if selected[b] else value[b] == -1,
                    'invalid or unrelated ' + key)
    for b in range(nb):
        fs = faces[offsets[b]:offsets[b+1]]
        require(np.all(bid[fs] == b) and np.all(neighbour[fs] == -1), 'CSR boundary differs')
        if selected[b]:
            require(len(fs) > 0 and np.all(flags[fs] == 1), 'partly sampled legacy section')
    length, normal = np.asarray(data['length']), np.asarray(data['normal'])
    require(length.shape == (nf,) and length.dtype == np.float64 and np.isfinite(length).all()
            and np.all(length > 0), 'invalid face lengths')
    require(normal.shape == (2, nf) and normal.dtype == np.float64 and np.isfinite(normal).all()
            and np.allclose(np.sum(normal*normal, axis=0), 1., rtol=0., atol=1.e-12),
            'invalid unit normals')
    forcing = np.asarray(data['forcing'])
    require(forcing.ndim == 2 and forcing.shape[0] >= 2 and forcing.shape[1] == nb and
            np.isfinite(forcing[:2]).all() and np.all(forcing[0, selected == 1] >= 0),
            'invalid Q/reference stage')
    return selected


class SampledDischarge:
    def __init__(self, data, inputs, arrays, folder):
        import cupy as cp
        from braids_support.gpu_fwave.resident_update import OPTIONS
        selected = validate_geometry(data)
        self.cp, self.inputs, self.arrays = cp, inputs, arrays
        self.nc, self.nf, self.nb = np.shape(data['q'])[1], len(data['owner']), len(selected)
        faces = np.flatnonzero(np.asarray(data['sampled'])).astype(np.int32)
        slot = np.full(self.nf, -1, np.int32); slot[faces] = np.arange(len(faces), dtype=np.int32)
        self.nfan = len(faces)
        self.geometry = dict(selected=cp.asarray(selected), slot=cp.asarray(slot),
                             delta=cp.asarray(data['bedoffset']))
        self.buffers = dict(branch=cp.zeros(self.nf, cp.int32), invariant=cp.zeros(self.nf),
            bounds=cp.zeros((4, self.nb)), fan_state=cp.zeros((8, self.nfan)),
            fan_result=cp.zeros((18, self.nfan)), fan_status=cp.zeros(self.nfan, cp.int32),
            fan_needed=cp.zeros(self.nfan, cp.int32), stage=cp.zeros(self.nb))
        self.grav, self.dry, self.limit = map(np.float64,
            (data['grav'], data['dry'], data.get('speed_limit', 50.)))
        from braids_core.windows_paths import filesystem_path
        folder = filesystem_path(folder)
        self.module = cp.RawModule(code=(folder/'sample_discharge_boundary.cu').read_text(),
            options=OPTIONS, name_expressions=('sampled_mode1_prepare', 'sampled_mode1_solve'))
        self.kernels = {name:self.module.get_function(name) for name in
                        ('sampled_mode1_prepare', 'sampled_mode1_solve')}
        self.fwave_module = cp.RawModule(code=(folder/'augmented_fwave.cu').read_text(),
            options=OPTIONS, name_expressions=('augmented_fwave_local',))
        self.fwave = self.fwave_module.get_function('augmented_fwave_local')

    def apply(self, q, trace, status, face_status):
        a, b, g, v = self.inputs, self.arrays, self.geometry, self.buffers
        shape = (np.int64(self.nc), np.int64(self.nf), np.int32(self.nb), np.int64(self.nfan))
        common = (q, a['bed'], a['normal'], a['length'], a['owner'],
            b['boundary_offset'], b['boundary_faces'], b['forcing'], g['delta'],
            g['selected'], g['slot'], *shape, self.grav, self.dry)
        self.kernels['sampled_mode1_prepare']((self.nb,), (128,), (*common,
            v['branch'], v['invariant'], v['bounds'], v['fan_state'], v['fan_needed'], status))
        self.fwave(((self.nfan+127)//128,), (128,),
            (v['fan_state'], np.int64(self.nfan), self.grav, self.dry,
             v['fan_result'], v['fan_status']))
        self.kernels['sampled_mode1_solve']((self.nb,), (128,), (*common, self.limit,
            v['branch'], v['invariant'], v['bounds'], v['fan_result'], v['fan_needed'],
            v['fan_status'], v['stage'], trace, status, face_status))


def wrap_boundary(Parent):
    class SampledLegacyBoundary(Parent):
        def __init__(self, data, inputs, source=None, **kwargs):
            super().__init__(data, inputs, source=source, **kwargs)
            self.sampled_discharge = None
            if np.any(np.asarray(data['sampled'])) and np.any(np.asarray(data['mode']) == 1):
                self.sampled_discharge = SampledDischarge(data, inputs, self.arrays, Path(source).parent)
                selected = validate_geometry(data)
                # Dispatch mask only. Actual mode/forcing/geometry are unchanged.
                self.unsampled_modes = self.cp.asarray(np.where(selected, 0, data['mode']).astype(np.int32))
                self.has_unsampled_legacy = bool(np.any((np.asarray(data['mode']) == 1) & (selected == 0)))
                self.mode1 = False

        def apply(self, q, trace):
            super().apply(q, trace)
            if self.sampled_discharge is None:
                return
            if self.has_unsampled_legacy:
                a, b = self.inputs, self.arrays
                self.kernels['gpu_mode1_traces']((self.nb,), (128,),
                    (q,a['bed'],a['normal'],a['length'],a['owner'],b['boundary_offset'],
                     b['boundary_faces'],self.unsampled_modes,b['forcing'],np.int64(self.nc),
                     np.int64(self.nf),np.int32(self.nb),self.grav,self.dry,self.speed,
                     trace,self.status,self.face_status))
            self.sampled_discharge.apply(q, trace, self.status, self.face_status)
    return SampledLegacyBoundary
