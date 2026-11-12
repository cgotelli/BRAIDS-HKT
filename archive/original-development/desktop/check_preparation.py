"""Bounded cold/repeated preparation benchmark; no hydraulic timestepping."""
import json
from pathlib import Path
import statistics
import time
import launch
import numpy as np
from braids_core import river_worker
from braids_core.sediment.portable_river_runtime import install
from braids_desktop.demo import create_demo
from braids_desktop import preparation,services

base=Path(__file__).resolve().parent
install(json.loads((base/'runtime/PRECOMPILED.json').read_text()),base/'runtime')
folder=base.parent/'validation/_desktop_ui142'/('preparation_'+time.strftime('%Y%m%d_%H%M%S'))
root=folder/'case';doc=create_demo(root,end_time_s=5.)
from braids_core.boundaries.domain import PolygonDomain
doc['settings']['grid']=services.geometry_grid(PolygonDomain(**doc['settings']['domain']),.25,3)
doc['settings']['levels']=3
services.save_project_folder(root,doc)
project=root/'braids_project.json';asset=root/'model_layers/preparation_cache'
records=[]

def measure(label,fn):
    stamp=time.perf_counter();target=fn();elapsed=time.perf_counter()-stamp
    project,receipt=river_worker.load_prepared(target)
    meta=target/'DESKTOP_PREPARATION.json'
    row=dict(kind=label,wall_s=elapsed,prepared=str(target),project_id=receipt['project_id'])
    if meta.is_file():row.update(json.loads(meta.read_text()))
    records.append(row);print(json.dumps(row),flush=True)
    return target

stock=[];cold=[];warm=[]
for i in range(2):
    stock.append(measure('stock_cold',lambda i=i:river_worker.prepare(project,root/f'prepared/stock{i}')))
    cold.append(measure('desktop_cold',lambda i=i:preparation.prepare(project,root/f'prepared/cold{i}',asset_cache=asset/str(i))))
doc['settings']['end_time_s']=6.;doc['settings']['dt_max_s']=.08
services.save_project_folder(root,doc)
for i in range(3):
    warm.append(measure('desktop_repeated_changed_controls',lambda i=i:preparation.prepare(project,root/f'prepared/warm{i}',asset_cache=asset/'0')))
reference=measure('stock_changed_reference',lambda:river_worker.prepare(project,root/'prepared/changed_reference'))
with np.load(reference/'core/prepared/initial.npz',allow_pickle=False) as expected:
    for target in warm:
        with np.load(target/'core/prepared/initial.npz',allow_pickle=False) as got:
            assert set(got.files)==set(expected.files)
            for name in got.files:np.testing.assert_array_equal(got[name],expected[name])
        assert json.loads((target/'NORMAL_SLOPES.json').read_text())==json.loads((reference/'NORMAL_SLOPES.json').read_text())
        assert json.loads((target/'core/prepared/profile.json').read_text())==json.loads((reference/'core/prepared/profile.json').read_text())
preview=[]
for label,target in (('stock',reference),('desktop',warm[-1])):
    stamp=time.perf_counter();polygons,faces=services.prepared_mesh(target)
    preview.append(dict(kind=label,wall_s=time.perf_counter()-stamp,cells=len(polygons),faces=faces))
median={label:statistics.median(r['wall_s'] for r in records if r['kind']==label)
        for label in ('stock_cold','desktop_cold','desktop_repeated_changed_controls')}
summary=dict(records=records,median_wall_s=median,preview=preview,initial_fields_exact=True,normal_slopes_exact=True,
    current_profile_exact=True,repeated_vs_stock_speedup=median['stock_cold']/median['desktop_repeated_changed_controls'],
    runtime_manifest_sha256=preparation.sha(base/'runtime/RUNTIME.json'))
destination=base/'evidence/ui142';destination.mkdir(parents=True,exist_ok=True)
(destination/'PREPARATION.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(dict(median_wall_s=median,preview=preview,speedup=summary['repeated_vs_stock_speedup']),indent=2))
