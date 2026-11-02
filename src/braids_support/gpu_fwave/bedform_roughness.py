"""Resident fixed-grid Fredsoe/Van Rijn 1984 equilibrium bedform roughness.

This is an isolated I2 component, not an enabled coupled-solver option.  The
native closure has no history: evaluate it from the *current* RK stage state
for bed capacity and again from the pre-friction state in each source step.
"""
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OPTIONS = ('--std=c++11', '--fmad=false', '--prec-div=true', '--prec-sqrt=true', '--ftz=false')


def validate_scope(data, *, composed_bank_avalanche=False,
                   composed_physical_toe_avalanche=False):
    """Retain native bedform incompatibilities unless composition is explicit.

    ``composed_bank_avalanche`` is reserved for the recursive GPU-AMR owner
    that evaluates bedforms on level-local SSPRK stages, then applies one
    accepted-flux distributed-bank transaction and one parent avalanche after
    root reflux.  It does not make the standalone/fixed-grid bedform selector
    acquire bank physics implicitly.

    ``composed_physical_toe_avalanche`` is the corresponding explicit mode-2
    owner.  It retains the same bedform timing, but consumes the completed root
    Exner register through the pre-surveyed toe/donor/face geometry's supply,
    undercut and collapse transaction before its one parent-owned avalanche.
    Live surveyed section boundaries are not admitted.  The two composition
    selectors are intentionally exclusive so a caller cannot blur their
    different bank histories and face-ledger contracts.
    """
    if (type(composed_bank_avalanche) is not bool or
            type(composed_physical_toe_avalanche) is not bool or
            (composed_bank_avalanche and composed_physical_toe_avalanche)):
        raise ValueError('Bedform bank/avalanche composition switches are invalid')
    composed = composed_bank_avalanche or composed_physical_toe_avalanche
    if not bool(data.get('bedform_roughness', False)):
        raise ValueError('Bedform roughness must be explicitly enabled')
    if int(data.get('bedlaw', -1)) != 2 or int(data.get('mpm_friction_mode', -1)) != 1:
        raise ValueError('Bedform roughness requires roughness-corrected MPM bed load')
    if float(data.get('exchange_mode', 0)) != 0 or bool(data.get('suspended_transport_active', False)):
        raise ValueError('Bedform roughness is not qualified with suspended transport/exchange')
    forbidden = ('secondary_flow', 'bedload_adaptation', 'morphological_acceleration',
                 'bank_avalanche', 'bank_front_tracking', 'amr_enabled')
    if not composed:
        forbidden += ('bank_failure', 'general_avalanche')
    if any(bool(data.get(key, False)) for key in forbidden) or int(data.get('secondary_flow_model', 0)) != 0:
        raise ValueError('Bedform roughness is not qualified with another experimental process or AMR')
    if composed_bank_avalanche and not (
            bool(data.get('bank_failure', False)) and
            int(data.get('bank_toe_erosion_mode', 0)) == 3 and
            bool(data.get('general_avalanche', False))):
        raise ValueError('Composed bedform roughness requires distributed bank mode 3 and parent avalanche')
    if composed_physical_toe_avalanche and not (
            bool(data.get('bank_failure', False)) and
            int(data.get('bank_toe_erosion_mode', 0)) == 2 and
            bool(data.get('general_avalanche', False))):
        raise ValueError('Composed bedform roughness requires physical-toe bank mode 2 and parent avalanche')
    if float(data.get('bank_erosion_fraction', 0)) > 0 or float(data.get('morfac', data.get('morphological_factor', 1))) != 1:
        raise ValueError('Bedform roughness requires zero bank erosion and MorFac 1')
    if np.any(np.asarray(data.get('sediment_type_fraction', 0)) != 0):
        raise ValueError('Bedform roughness requires noncohesive sediment fractions')
    if bool(data.get('production_section_geometry', False)) and np.any(
            np.asarray(data.get('section_control', 0)) == 1):
        raise ValueError('Bedform roughness is not qualified with normal-depth inlet control')
    arrays = {}
    for key in ('manning_cell', 'd50_fraction', 'rhos_fraction', 'bulk_fraction'):
        if key not in data:
            raise ValueError('Missing bedform control ' + key)
        arrays[key] = np.asarray(data[key], dtype=np.float64)
        if arrays[key].ndim != 1 or not np.isfinite(arrays[key]).all() or np.any(arrays[key] <= 0):
            raise ValueError('Invalid positive bedform control ' + key)
    ng = arrays['d50_fraction'].size
    if not 1 <= ng <= 32 or arrays['rhos_fraction'].shape != (ng,) or arrays['bulk_fraction'].shape != (ng,):
        raise ValueError('Invalid bedform grain arrays')
    rho = float(data.get('rho', data.get('water_density', np.nan)))
    dry = float(data.get('dry_tolerance', data.get('dry', np.nan)))
    critical = float(data.get('mpm_critical', np.nan))
    if not np.isfinite([rho, dry, critical]).all() or rho <= 0 or dry <= 0 or critical < 0:
        raise ValueError('Invalid bedform fluid/depth/Shields controls')
    if np.any(arrays['rhos_fraction'] <= rho):
        raise ValueError('Bedform grains must be denser than water')
    return arrays, rho, dry, critical


class ResidentBedformRoughness:
    """One CUDA cell thread per closure; outputs are overwritten on each call.

    ``properties`` rows are effective Manning n, dune height, dune length,
    grain ks, dune ks and total ks. A nonzero device ``status`` is fatal to
    the stage; the caller must check it before consuming these buffers.
    """

    def __init__(self, data, *, source_path=None,
                 composed_bank_avalanche=False,
                 composed_physical_toe_avalanche=False):
        import cupy as cp

        arrays, self.rho, self.dry, self.critical = validate_scope(data,
            composed_bank_avalanche=composed_bank_avalanche,
            composed_physical_toe_avalanche=
                composed_physical_toe_avalanche)
        self.cp = cp
        self.nc = arrays['manning_cell'].size
        self.ng = arrays['d50_fraction'].size
        self.manning = cp.asarray(np.ascontiguousarray(arrays['manning_cell']))
        self.d50 = cp.asarray(np.ascontiguousarray(arrays['d50_fraction']))
        self.rhos = cp.asarray(np.ascontiguousarray(arrays['rhos_fraction']))
        self.bulk = cp.asarray(np.ascontiguousarray(arrays['bulk_fraction']))
        self.properties = cp.empty((6, self.nc), dtype=cp.float64)
        self.iterations = cp.empty(self.nc, dtype=cp.int32)
        self.status = cp.empty(self.nc, dtype=cp.int32)
        source = Path(source_path or HERE / 'bedform_roughness.cu').read_text()
        module = cp.RawModule(code=source, options=OPTIONS, name_expressions=('bf_properties',))
        self.kernel = module.get_function('bf_properties')

    def evaluate(self, q, layers):
        cp = self.cp
        if (not isinstance(q, cp.ndarray) or q.shape != (3, self.nc) or q.dtype != cp.float64
                or not q.flags.c_contiguous):
            raise ValueError('Expected contiguous resident q(3,nc)')
        if (not isinstance(layers, cp.ndarray) or layers.shape[1:] != (self.ng, self.nc)
                or layers.shape[0] < 1 or layers.dtype != cp.float64 or not layers.flags.c_contiguous):
            raise ValueError('Expected contiguous resident layers(nlayer,ng,nc)')
        self.kernel(((self.nc + 127) // 128,), (128,),
                    (np.int32(self.nc), np.int32(self.ng), q, layers, self.manning,
                     self.d50, self.rhos, self.bulk, np.float64(self.rho),
                     np.float64(self.dry), np.float64(self.critical), self.properties,
                     self.iterations, self.status))
        return {'properties': self.properties, 'iterations': self.iterations, 'status': self.status}
