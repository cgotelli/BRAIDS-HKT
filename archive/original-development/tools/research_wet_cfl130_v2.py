"""Extend the isolated threshold experiment past its explicit admission gate.

The stock validator pins dry=1e-6 as a qualification restriction. This research
wrapper changes ONLY that admission check, keeping real native and Python/CUDA
physical thresholds at the requested value. It does not weaken other checks.
The first v1 construction failures remain preserved as non-simulation evidence.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import research_wet_cfl130 as original
_original_candidate=original.candidate


@contextmanager
def candidate(cfl_cap,dry_tolerance=1e-6):
    from braids_core.sediment import current_river_solver as current
    from braids_core.sediment.river_explicit import RiverExplicit
    from braids_core.sediment.bend_methods_v1 import digest
    qualified=current.validate_current_river_input
    audit=dict(schema='isolated_dry_admission130.v2',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stock_validator_sha256=digest(current.__file__),
        replaced_admission='dry==1e-6 qualification restriction only',
        required_actual_dry_m=float(dry_tolerance), all_other_material_checks_retained=True)

    def validate(data,controls):
        if float(data['dry'])!=float(dry_tolerance):
            raise ValueError('Native data differ from explicit research threshold')
        return qualified(dict(data,dry=1e-6),controls)

    with _original_candidate(cfl_cap,dry_tolerance) as context:
        explicit=RiverExplicit.__init__
        def construct(self,solver,cfl=.225):
            explicit(self,solver,cfl)
            self.identity['isolated_research']['dry_admission_override']=dict(audit)
        RiverExplicit.__init__=construct
        current.validate_current_river_input=validate
        context['identity']['dry_admission_override']=dict(audit)
        try:yield context
        finally:
            current.validate_current_river_input=qualified
            RiverExplicit.__init__=explicit


if __name__=='__main__':
    original.candidate=candidate
    try:original.main()
    finally:original.candidate=_original_candidate
