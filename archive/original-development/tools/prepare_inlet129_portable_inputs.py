"""Explicit portable-runtime coverage replacement for an adverse normal inlet.

This is a validation input change, not a migration or a normal-slope fix. The
original deck and failed preparation are preserved and hashed. The resulting
critical-inlet run only qualifies deployment on that geometry.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.prepare_inlet129_release import differences
from braids_core.river_project import validate_project
from braids_core.river_open_boundary import resolve


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--boundary',required=True)
    parser.add_argument('--rejected-slope',type=float,required=True)
    parser.add_argument('--failed-evidence',type=Path,required=True)
    args=parser.parse_args();source=args.source.resolve();out=args.output.resolve()
    if not source.is_relative_to(ROOT/'validation') or not out.is_relative_to(ROOT/'validation') or source==out:
        raise ValueError('Separate repository validation input required')
    if args.rejected_slope>=0:raise ValueError('Only recorded non-positive normal-slope rejection covered')
    raw=source.read_bytes();before=json.loads(raw);after=deepcopy(before);settings=after['settings']
    control=settings['hydraulics'][args.boundary]
    if control['method']!='inlet.normal_discharge' or control['energy_slope']!='mesh':
        raise ValueError('Input must be the rejected automatic-normal inlet')
    settings['hydraulics'][args.boundary]=dict(method='inlet.critical_discharge',discharge_m3_s=control['discharge_m3_s'])
    kinds={a['bc_id']:a['kind'] for a in settings['assignments'] if a['kind']!='wall'}
    after['boundary_resolution']=resolve(settings['hydraulics'],after['boundary_states'],kinds,
        sediment=after['model']['physics']['sediment'],ng=len(settings['columns']['fraction_ids']),end_time=settings['end_time_s'])
    validate_project(after)
    payload=json.dumps(after,indent=2).encode('utf-8')
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and out.read_bytes()!=payload:raise FileExistsError(out)
    out.write_bytes(payload)
    evidence=dict(source=str(source),source_sha256=hashlib.sha256(raw).hexdigest(),
        output=str(out),output_sha256=hashlib.sha256(payload).hexdigest(),
        failed_preparation=str(args.failed_evidence.resolve()),boundary=args.boundary,
        rejection='Normal-flow preparation correctly rejects a non-positive signed inward bed fall',
        recorded_signed_inward_slope_m_m=args.rejected_slope,
        reason='Explicitly approved critical-inlet substitution solely for portable-runtime geometry coverage',
        not_normal_flow_validation=True,changes=differences(before,after))
    out.with_suffix('.provenance.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    if source.read_bytes()!=raw:raise AssertionError('Source changed')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
