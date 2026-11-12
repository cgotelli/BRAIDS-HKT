"""Current capacities against authenticated output of unmodified FLOW 2DH."""
import importlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'validation/_wall188/desktop_candidate'
sys.path[:0]=[str(RUNTIME),str(ROOT),str(ROOT/'desktop')]
os.environ['OMP_NUM_THREADS']='4'
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
verify();install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
for name in ('validation','validation.sediment','validation.sediment.river_morphodynamics'):
    importlib.import_module(name).__path__.append(str(ROOT/Path(*name.split('.'))))
from validation.sediment.river_morphodynamics.unified_capacity_reference import replay
from validation.sediment.river_morphodynamics.unified_additional_capacity_reference import replay as additional
sys.path[:0]=[str(RUNTIME),str(ROOT),str(ROOT/'desktop')]
from stage_secondary185 import save,sha


if __name__=='__main__':
    out=ROOT/'validation/_bend189/flow_replay';out.mkdir(parents=True,exist_ok=True)
    results=[]
    for backend in ('cpu','gpu'):
        for name,law in (('mpm',2),('wang_fredsoe',13),('ashida_michiue_generalized',14)):
            folder=out/(name+'_'+backend)
            if law==2:result=replay('mpm',backend,folder)
            else:
                ref=ROOT/f'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/scalar_families_flow_v4/flow{law}'
                result=additional(ref,law,backend,folder)
            for row in result['rows']:
                assert row['capacity_relative_l2']<5e-7 and row['capacity_max_m2_s']<2e-11,row
            result.pop('samples',None)
            results.append(dict(method=name,backend=backend,**result))
            print('PASS',name,backend,flush=True)
    save(out/'CHECK.json',dict(runtime=sha(RUNTIME/'RUNTIME.json'),results=results,complete=True))
