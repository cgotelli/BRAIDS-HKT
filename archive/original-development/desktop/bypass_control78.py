"""Matched transparent-rotor control: retain mesh/refinement and viscosity."""
from pathlib import Path
from particle_bypass78 import ROOT,TARGET,REPORT,read,save,sha,environment
environment()
from braids_desktop import services,preparation
from braids_desktop.results import ResultsStore
from braids_core import river_worker
import numpy as np

USER=Path.home()/'Desktop/BRAIDS_Bern_Turbines77/Bern_data/runs/run_20261101_211340_4f57d1e6'
HOME=TARGET.parent/'transparent_control'
def main():
    HOME.mkdir(exist_ok=True);d=read(USER/'PROJECT.json')
    source=sha(USER/'PROJECT.json')
    d['terrain']['dem_path']=str((USER.parents[1]/d['terrain']['dem_path']).resolve())
    d.pop('editor_state',None)
    for r in d['turbines']['turbines']:r['control']=dict(mode='constant_ct',thrust_coefficient=0.,power_coefficient=0.)
    d=services.validate_document(d);save(HOME/'input.json',d)
    prepared=preparation.prepare(HOME/'input.json',HOME/'prepared',asset_cache=TARGET.parent/'assets')
    result=river_worker.run(prepared,HOME/'run',until=14.)
    assert result['status']=='segment_complete',result
    a=ResultsStore(USER);b=ResultsStore(HOME/'run');a0=a.read(0);b0=b.read(0)
    np.testing.assert_array_equal(a0.q,b0.q);np.testing.assert_array_equal(a0.z,b0.z)
    assert len(a0.polygons)==len(b0.polygons)
    assert all(p.equals_exact(q,0) for p,q in zip(a0.polygons,b0.polygons))
    samples=read(HOME/'run/TURBINE_PERFORMANCE.json');assert all(s['force_per_density_m4_s2']==0. for s in samples)
    assert sha(USER/'PROJECT.json')==source
    save(REPORT/'CONTROL.json',dict(passed=True,user_run=str(USER),source_project_sha256=source,control=str(HOME/'run'),result=result,
        identical_initial_q_bed_polygons=True,changes='Only rotor control replaced by transparent Ct=Cp=0; original duration, roughness, viscosity, initial state, mesh and BC retained. No user file modified.'))
    print('Matched transparent control complete',flush=True)
if __name__=='__main__':main()
