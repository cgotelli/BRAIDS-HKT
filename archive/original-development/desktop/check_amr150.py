"""Fresh dynamic-AMR qualification using the exact reported project settings."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_amr150'
runtime=OUT/'candidate';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(runtime))
from braids_core.river_launcher import verify
verify()
from braids_core.sediment.portable_river_runtime import install
install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)
from braids_desktop import services,preparation
from braids_core import river_worker
from braids_core.river_model import resolve_model
from braids_core.sediment.adaptive_river_worker import read_live_frame
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['cpu','gpu'],required=True)
parser.add_argument('--until',type=float,default=20.);parser.add_argument('--name',default='fixed')
parser.add_argument('--restart',action='store_true');parser.add_argument('--breakline',action='store_true')
parser.add_argument('--bedload',action='store_true')
parser.add_argument('--control',action='store_true',help='Retained small bedload control, not the natural-river hydro setup')
args=parser.parse_args()
source=Path('C:/Users/STREEM/Desktop/0101ttt/River_data/prepared/prepare_20261024_163840_37d6b0b0/PROJECT.json')
if args.control:source=ROOT/'validation/_breaklines147/final_bedload_cpu/braids_project.json'
raw=source.read_bytes();doc=json.loads(raw)
doc['model']['request']['execution']['backend']=args.backend
doc['model']=resolve_model(doc['model']['request']).document()
if args.breakline:
    doc['settings']['breaklines']=json.loads((ROOT/'validation/_merge149/user_v2_cpu/braids_project.json').read_text())['settings']['breaklines']
if args.bedload:
    from braids_core.river_model import new_bedload_selection
    from braids_core.river_open_boundary import new_state,resolve
    selection=new_bedload_selection()
    doc['settings']['columns']['fraction_ids']=[g['id'] for g in selection['grains']]
    selection['bed']=dict(composition='bed.layered',transport_layer='transport_layer.constant',coarse_layer=False,sorting='sorting.off')
    doc['model']['request']['sediment']=dict(enabled=True,selection=selection)
    doc['model']=resolve_model(doc['model']['request']).document()
    for name,control in doc['settings']['hydraulics'].items():
        doc['boundary_states'][name]=new_state(control['method'],sediment=True)
    doc['boundary_resolution']=resolve(doc['settings']['hydraulics'],doc['boundary_states'],
        {a['bc_id']:a['kind'] for a in doc['settings']['assignments'] if a['kind']!='wall'},
        sediment=doc['model']['physics']['sediment'],ng=1,end_time=doc['settings']['end_time_s'])
case=OUT/(args.name+'_'+args.backend)
if not args.restart:
    if case.exists():raise FileExistsError(case)
    services.save_project_folder(case,doc,source_base=source.parent if args.control else source.parents[2])
    prepared=preparation.prepare(case/'braids_project.json',case/'prepared/current',asset_cache=case/'model_layers/preparation_cache')
else:prepared=case/'prepared/current'
name='continued' if args.restart else 'run';stamp=time.perf_counter()
result=river_worker.run(prepared,case/name,until=args.until,restart=case/'run/checkpoint.npz' if args.restart else None)
elapsed=time.perf_counter()-stamp
rows=json.loads((case/name/'DIAGNOSTICS.json').read_text());events=json.loads((case/name/'ADAPTATION.json').read_text())
row,mesh,values=read_live_frame(case/name)
report=dict(source=str(source),source_sha256=hashlib.sha256(raw).hexdigest(),backend=args.backend,
    run_seconds=elapsed,run=result,adaptations=[dict(time_s=e['time_s'],changed=e['changed'],cells=e['to_cells']) for e in events],
    minimum_depth_m=float(values['q'][0].min()),maximum_water_residual_m3=max(abs(r['water_residual_m3']) for r in rows),
    maximum_class_residual_kg=max(max(abs(x) for x in r['class_residual_kg']) for r in rows),
    maximum_wall_integral=max(max(r['wall_integrals']) for r in rows))
(case/(name+'_CHECK.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));assert source.read_bytes()==raw
assert report['maximum_water_residual_m3']<1e-8 and report['maximum_wall_integral']==0
assert report['minimum_depth_m']>=0
