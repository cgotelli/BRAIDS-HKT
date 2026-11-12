"""Write explicit native bedload settings; no Python timestep calculations."""
from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class BedloadFace:
    boundary_id: int
    side: int  # 1 west, 2 east, 3 south, 4 north
    index: int  # zero-based face index on that domain side
    mode: str  # prescribed, equilibrium, export
    inward_mass_flux_kg_m_s: float = 0.0


def write_native_bedload(path: Path, faces: list[BedloadFace], *, nx: int, ny: int,
                         d50_m: float, d90_m: float, z0_m: float,
                         minimum_depth_m: float = 0.5, maximum_timestep_s: float = 0.1,
                         maximum_bed_change_fraction: float = 0.02,
                         water_density_kg_m3: float = 1000.0, gravity_m_s2: float = 9.81,
                         viscosity_m2_s: float = 1e-6, scheme: str = 'legacy_mc',
                         transport_law: str = 'van_rijn_1993', mpm_cf: float = 0.01,
                         mpm_critical_shields: float = 0.047) -> None:
    """All unlisted faces are bedload walls; rates are per width, not total kg/s.

    Bed density/porosity are read from the existing settling/Exner configuration.
    Policies are constant in time in v1; accepted inward/outward water flux and
    the sediment depth gate determine whether a boundary can supply/export.
    V2 law ``none`` keeps the RK transport integrator and suspended Exner
    exchange, but sets all bedload fluxes to zero and forbids bedload faces.
    """
    values = [water_density_kg_m3, gravity_m_s2, viscosity_m2_s, d50_m, d90_m,
              z0_m, minimum_depth_m, maximum_timestep_s, maximum_bed_change_fraction]
    if nx < 1 or ny < 1 or any(not isfinite(v) or v <= 0 for v in values):
        raise ValueError('Bedload dimensions and parameters must be positive and finite')
    if d90_m < d50_m or maximum_bed_change_fraction > 0.1:
        raise ValueError('Require d90 >= d50 and bed-change fraction <= 0.1')
    schemes = {'legacy_mc': 0, 'mc_ssprk3': 2, 'weno_z_ssprk3': 5}
    laws = {'none': 0, 'van_rijn_1993': 1, 'mpm': 2}
    if scheme not in schemes or transport_law not in laws:
        raise ValueError('Unknown bedload scheme or transport law')
    if scheme == 'legacy_mc' and transport_law != 'van_rijn_1993':
        raise ValueError('This transport law requires the stage-consistent v2 path')
    if transport_law == 'none' and faces:
        raise ValueError('No-bedload transport requires zero bedload boundary faces')
    if any(not isfinite(v) or v < 0 for v in (mpm_cf, mpm_critical_shields)):
        raise ValueError('MPM friction/critical Shields must be finite and nonnegative')
    modes = {'prescribed': 1, 'equilibrium': 2, 'export': 3}
    seen = set()
    policies = {}
    lines = ['1' if scheme == 'legacy_mc' else '2', ' '.join(format(v, '.17g') for v in values)]
    if scheme != 'legacy_mc':
        lines.append(f'{laws[transport_law]} {schemes[scheme]} {mpm_cf:.17g} {mpm_critical_shields:.17g}')
    lines.append(str(len(faces)))
    for face in faces:
        key = (face.side, face.index)
        if (face.boundary_id < 1 or face.side not in (1, 2, 3, 4) or
                face.index < 0 or face.index >= (ny if face.side <= 2 else nx) or
                face.mode not in modes or key in seen):
            raise ValueError('Invalid or duplicate bedload boundary face')
        rate = face.inward_mass_flux_kg_m_s
        if not isfinite(rate) or rate < 0 or (face.mode != 'prescribed' and rate != 0):
            raise ValueError('Only prescribed supply accepts a nonnegative mass flux')
        policy = (face.mode, rate)
        if face.boundary_id in policies and policies[face.boundary_id] != policy:
            raise ValueError('Faces sharing an ID must use the same policy and rate')
        policies[face.boundary_id] = policy
        seen.add(key)
        lines.append(f'{face.boundary_id} {face.side} {face.index} {modes[face.mode]} {rate:.17g}')
    Path(path).write_text('\n'.join(lines) + '\n')
