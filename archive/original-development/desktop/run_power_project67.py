"""Run a user-supplied current project with explicit turbine speed surfaces.

Qualification CLI, separate from the installed Desktop UI. No benchmark names,
curves, spacings, geometry or turbulence constants are selected here.
"""
import argparse,csv,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'desktop/qualification/energy67/candidate_v3'
sys.path[:0]=[str(RUNTIME),str(ROOT)];sys.dont_write_bytecode=True
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('project',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--initial-checkpoint',type=Path);p.add_argument('--map-changed-mesh',action='store_true');p.add_argument('--until',type=float);a=p.parse_args()
    from braids_core.sediment import portable_river_runtime
    from braids_core import river_worker,river_state_snapshot as snap,river_state_initialization as init
    from braids_core.river_turbine_surface import integrate_energy
    from braids_desktop import services,preparation
    portable_river_runtime.install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
    doc=services.load_project_folder(a.project) if a.project.is_dir() else json.loads(a.project.read_text(encoding='utf-8'))
    if doc['settings']['adaptive']:
        raise ValueError('The speed-surface power runner is qualified on a fixed mesh. Disable dynamic adaptation; static local rotor/wake refinement remains available.')
    a.output.mkdir(parents=True,exist_ok=False)
    doc=services.save_project_folder(a.output/'project',doc,source_base=a.project if a.project.is_dir() else a.project.parent)
    path=a.output/'input.json';path.write_text(json.dumps(doc,indent=2),encoding='utf-8')
    prepared=preparation.prepare(path,a.output/'prepared',asset_cache=a.output/'assets');state=None
    if a.initial_checkpoint:
        src=a.output/'initial_source.npz';src.write_bytes(snap.source_from_checkpoint(a.initial_checkpoint));state=a.output/'initialized.npz'
        decision=init.prepare(prepared,src,state)
        (a.output/'INITIALIZATION_DECISION.json').write_text(json.dumps(decision,indent=2))
        if decision['status']=='mapping_required':
            if not a.map_changed_mesh:raise ValueError('Mesh differs. Domain is checked; rerun in a fresh output directory with --map-changed-mesh only if conservative mapping is desired.')
            decision=init.prepare(prepared,src,state,allow_mapping=True)
        if decision['status']!='ready':raise ValueError(str(decision))
    result=river_worker.run(prepared,a.output/'run',initial_state=state,until=a.until)
    records=json.loads((a.output/'run/TURBINE_PERFORMANCE.json').read_text());energy=integrate_energy(records)
    with (a.output/'TURBINE_ENERGY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(energy[0]));w.writeheader();w.writerows(energy)
    (a.output/'RESULT.json').write_text(json.dumps(result,indent=2));print(a.output/'TURBINE_ENERGY.csv')
if __name__=='__main__':main()
