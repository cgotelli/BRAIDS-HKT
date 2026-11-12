"""Installed runtime rejects changed coefficient/checkpoint identities."""
import json,sys
from stage_smart183 import ROOT,BASE,OUT,sha,save
sys.path[:0]=[str(BASE),str(ROOT),str(ROOT/'desktop')]
from braids_desktop import services
from braids_desktop.preparation import prepare
from braids_core.river_model import resolve_model
from braids_core import river_worker
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
from analyze_smart183 import compare

def main():
    verify();install(json.loads((BASE/'PRECOMPILED.json').read_text()),BASE)
    evidence={};fields=('q','z','mass','water','bed_boundary','bedload_vector_m2_s')
    for b in ('cpu','gpu'):
        label='smart_jaggi_'+b
        diff=compare(OUT/'release'/label/'full',OUT/'installed'/label/'full',fields)
        assert max(diff.values())==0.
        doc=services.load_project_folder(OUT/'installed'/label/'project')
        doc['model']['request']['sediment']['selection']['capacity']['parameters']['ACal']=3.9
        doc['model']=resolve_model(doc['model']['request']).document()
        out=OUT/'identity'/b;out.mkdir(parents=True,exist_ok=False)
        save(out/'project.json',doc);prep=out/'prepared';prepare(out/'project.json',prep,asset_cache=OUT/'assets')
        try:river_worker.run(prep,out/'rejected',restart=OUT/'installed'/label/'cancelled/checkpoint.npz')
        except (ValueError,RuntimeError) as e:message=str(e)
        else:raise AssertionError('A changed coefficient reused a checkpoint')
        evidence[b]=dict(installed_trajectory_difference=diff,changed_coefficient_restart_rejected=message)
    save(OUT/'INSTALLED_CHECK.json',dict(runtime=sha(BASE/'RUNTIME.json'),backends=evidence))
    print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()
