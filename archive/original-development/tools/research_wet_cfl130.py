"""Isolated, explicitly labelled hydraulic CFL/dry-threshold research.

No maintained source, project schema, installed plugin or Delft3D file changes.
Each experiment must use a NEW private preparation/output folder. This module
alters in-process construction only; its source digest and numerical overrides
are attached to every solver identity. Do not use it to resume public runs.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@contextmanager
def candidate(cfl_cap, dry_tolerance=1e-6):
    """Install matching CPU/CUDA research controls for one private process.

    The public deck retains its supported CFL. The explicitly different actual
    limit is recorded in the controller identity; comparison must read this
    identity rather than mistake the public-deck value for the executed cap.
    Only hydrodynamics is admitted: sediment threshold policies are unchanged.
    """
    from braids_core.sediment.river_explicit import RiverExplicit
    from braids_core.sediment.bend_methods_v1 import Parameters
    from braids_core.boundaries import cut_cell_io
    import numpy as np

    if not np.isfinite([cfl_cap, dry_tolerance]).all() or not 0 < cfl_cap <= 1.5 or not 1e-6 <= dry_tolerance <= 1e-3:
        raise ValueError('Research envelope: face-sum cap (0,1.5], dry threshold [1e-6,1e-3] m')
    identity = dict(schema='isolated_wet_cfl130.v1',
        override_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        actual_face_sum_cfl=float(cfl_cap), dry_tolerance_m=float(dry_tolerance),
        cfl_definition='dt*max_cell(sum_face(length*max_abs_wave_speed)/actual_area)',
        spatial_solver='unchanged augmented f-wave', public_admitted=False,
        source_files_modified=False, sediment_admitted=False)
    original_explicit = RiverExplicit.__init__
    original_parameters = Parameters.__init__
    original_writer = cut_cell_io.write_cut_case
    owners = []

    def parameters(self, *args, **kwargs):
        # No callers in this isolated path supply positional dry values.
        if len(args) > 1:
            raise ValueError('Research threshold requires unambiguous Parameters construction')
        if 'dry' in kwargs and kwargs['dry'] != dry_tolerance:
            raise ValueError('Conflicting explicit physical dry threshold')
        kwargs['dry'] = float(dry_tolerance)
        return original_parameters(self, *args, **kwargs)

    def writer(*args, **kwargs):
        kwargs['dry_tolerance_m'] = float(dry_tolerance)
        return original_writer(*args, **kwargs)

    def explicit(self, solver, cfl=.225):
        original_explicit(self, solver, cfl)
        if solver.physics.parameters.dry != dry_tolerance or solver.data['dry'] != dry_tolerance:
            raise ValueError('CPU/CUDA native and physical hydraulic thresholds differ')
        self.cfl = float(cfl_cap)
        self.identity['cfl'] = self.cfl
        self.identity['isolated_research'] = dict(identity, deck_face_sum_cfl=float(cfl))
        owners.append(solver)

    Parameters.__init__ = parameters
    cut_cell_io.write_cut_case = writer
    RiverExplicit.__init__ = explicit
    try:
        yield dict(identity=identity, owners=owners)
    finally:
        RiverExplicit.__init__ = original_explicit
        cut_cell_io.write_cut_case = original_writer
        Parameters.__init__ = original_parameters


def main():
    import argparse
    from tools.run_canonical_inlet129 import run_case, comparison, read
    from braids_gpu.atomic_output import write_json_atomic
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--analytic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cfl', type=float, required=True)
    parser.add_argument('--dry', type=float, default=1e-6)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    with candidate(args.cfl, args.dry) as context:
        result = run_case(args.source, args.output, read(args.analytic))
        # Old analyzer correctly used the old public cap. Preserve its original
        # check and record a separate comparison with the ACTUALLY executed cap.
        research = dict(context['identity'], owner_count=len(context['owners']),
            owner_thresholds=[dict(native=a.data['dry'], physical=a.physics.parameters.dry)
                              for a in context['owners']],
            accepted_actual_cap=result['hydraulic_dt']['accepted_cfl_max'] <= args.cfl*(1+1e-10),
            checks_except_original_public_cap={k:v for k,v in result['checks'].items()
                                               if k != 'accepted_cfl_within_bound'})
        if args.reference:
            research['comparison'] = comparison(args.reference, args.output)
            research['depth_within5mm'] = research['comparison']['depth_max_m'] <= .005
        write_json_atomic(args.output/'RESEARCH.json', research)
    print(json.dumps(research, indent=2), flush=True)
    if not research['accepted_actual_cap'] or not all(research['checks_except_original_public_cap'].values()):
        raise SystemExit('Research run checks require review')


if __name__ == '__main__':
    main()
