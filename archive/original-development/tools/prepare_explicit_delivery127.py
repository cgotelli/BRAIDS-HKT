"""Create new release qualification decks from frozen laboratory inputs.

Validation only: not a project migration route and never edits user inputs.
Geometry, forcing and physical methods are retained; each changed field is recorded.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from braids_core.river_model import INTEGRATOR,resolve_model
from braids_core.river_project import SCHEMA,validate_project
from braids_gpu.atomic_output import write_json_atomic


def main():
    out=ROOT/'validation/_delivery127/projects';out.mkdir(parents=True,exist_ok=True)
    sources=list((ROOT/'validation/_accuracy122/release_inputs').glob('cpu_*.project.json'))
    sources+=[ROOT/'validation/_open_boundary123/projects'/f'check_{mode}_{backend}_{physics}.json'
        for mode in ('static','dynamic') for backend in ('cpu','gpu') for physics in ('hydro','bedload')]
    sources+=[ROOT/'validation/_explicit_fwave124'/f'input_{size}'/'PROJECT.json' for size in ('1m','5m')]
    records=[]
    for source in sorted(sources):
        original=json.loads(source.read_text(encoding='utf-8'));d=deepcopy(original)
        d['schema']=SCHEMA;request=d['model']['request']
        request['execution']['integrator']=INTEGRATOR
        d['model']=resolve_model(request).document();d['settings']['controls']['cfl']=.225
        if source.name=='PROJECT.json':
            d['settings'].update(end_time_s=4.,dt_max_s=.5,output_interval_s=.5)
            names=[(source.parent.name+'_cpu.json','cpu'),(source.parent.name+'_gpu.json','gpu')]
        else:names=[(source.name,request['execution']['backend'])]
        for name,backend in names:
            d['model']['request']['execution']['backend']=backend
            d['model']=resolve_model(d['model']['request']).document()
            validate_project(d)
            if (out/name).exists():
                if json.loads((out/name).read_text())!=d:raise ValueError('Existing validation input differs: '+name)
            else:write_json_atomic(out/name,d)
            records.append(dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                target=str(out/name),changes=dict(schema=SCHEMA,integrator=INTEGRATOR,cfl=.225,backend=backend,
                    clock_override=d['settings'] if source.name=='PROJECT.json' else None)))
    write_json_atomic(out.parent/'INPUT_PROVENANCE.json',dict(purpose='new explicit release qualification; no runtime migration',inputs=records))
    print(len(records),'new validation inputs')


if __name__=='__main__':main()
