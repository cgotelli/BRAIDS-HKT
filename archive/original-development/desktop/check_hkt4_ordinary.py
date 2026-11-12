"""Matched ordinary hydrodynamics control against the retained preview51 engine.

Old-format input is constructed only for that old engine as a reference fixture;
the new application never migrates or executes old projects.
"""
import argparse,json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--reference',action='store_true');args=parser.parse_args()
runtime=ROOT/('desktop/runtime' if args.reference else 'desktop/qualification/hkt4/candidate');sys.path.insert(0,str(runtime))
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
from braids_core.river_project import SCHEMA,validate_project
from braids_core import river_worker

if __name__=='__main__':
    verify();install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime);shutil.which=lambda name:None
    root=ROOT/'desktop/qualification/hkt4'/('ordinary_reference' if args.reference else 'ordinary_release');root.mkdir(exist_ok=False)
    fixture=json.loads((ROOT/'desktop/qualification/hkt4/public_release_cpu/input.json').read_text());fixture['schema']=SCHEMA
    fixture['settings']['end_time_s']=.5;fixture['settings']['checkpoint_interval_s']=.5
    if args.reference:fixture.pop('turbines')
    else:fixture['turbines']=None
    fixture=validate_project(fixture);source=root/'input.json';source.write_text(json.dumps(fixture,indent=2)+'\n')
    river_worker.prepare(source,root/'prepared');result=river_worker.run(root/'prepared',root/'run');assert result['status']=='complete';print(result['status'])
