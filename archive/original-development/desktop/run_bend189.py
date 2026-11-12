"""Bounded matched advanced-flow contrasts using the immutable preview30 runtime."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'validation/_bend189'
RUNTIME=ROOT/'validation/_wall188/desktop_candidate'
sys.path[:0]=[str(RUNTIME),str(ROOT),str(ROOT/'desktop')]
os.environ['QT_QPA_PLATFORM']='offscreen';os.environ['OMP_NUM_THREADS']='4'
import numpy as np
from stage_secondary185 import save,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['cpu','gpu'],required=True)
    ap.add_argument('--mode',choices=['off','secondary','viscosity'],required=True)
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args();out=BASE/'runs'/(args.mode+'_'+args.backend)
    if not args.check_only:out.mkdir(parents=True,exist_ok=False)
    from PySide6.QtWidgets import QApplication
    from braids_desktop.forms import SettingsPanel
    from braids_desktop import services
    from braids_desktop.preparation import prepare
    from braids_desktop.results import ResultsStore
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    from braids_core import river_worker
    verify();install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
    source=ROOT/f'validation/_wall188/release/wet_prescribed_{args.backend}/project'
    if args.check_only:
        status=json.loads((out/'run/STATUS.json').read_text())
        check(out,source,status,status['wall_s']);return
    doc=services.load_project_folder(source)
    app=QApplication.instance() or QApplication([])
    panel=SettingsPanel();panel.load_document(doc)
    panel.secondary.setChecked(args.mode=='secondary');panel.turbulence.setChecked(args.mode=='viscosity')
    # Do not retain an inactive editor value as active physics.
    if args.mode=='secondary':panel.betac.setValue(1.);panel.secondary_diffusion.setValue(.01)
    if args.mode=='viscosity':panel.viscosity.setValue(.01)
    changed=panel.document();panel.close();changed.pop('editor_state',None)
    for k in ('adaptation','grid','levels','domain','hydraulics','sediment_boundaries','sediment_start_time_s','solids','breaklines'):
        assert changed['settings'][k]==doc['settings'][k],k
    assert changed['initial']==doc['initial']
    assert changed['model']['request']['sediment']==doc['model']['request']['sediment']
    services.save_project_folder(out/'project',changed)
    save(out/'INPUT.json',dict(runtime=sha(RUNTIME/'RUNTIME.json'),source=str(source),
        backend=args.backend,mode=args.mode,until=220.,warmup=200.,morph_factor=5.,
        flow=changed['model']['request']['hydrodynamics'],slope='off',performance_comparison=False))
    start=time.perf_counter();prepare(out/'project/braids_project.json',out/'prepared',asset_cache=out/'assets')
    print('PREPARED',args.mode,args.backend,flush=True)
    def progress(path,row):
        if row['time_s']%20==0 or row['time_s']>=200:
            print('t',row['time_s'],'wall',round(time.perf_counter()-start,1),flush=True)
    status=river_worker.run(out/'prepared',out/'run',until=220.,checkpoint_dir=out/'checkpoints',on_frame=progress)
    check(out,source,status,time.perf_counter()-start)


def check(out,source,status,elapsed):
    from braids_desktop.results import ResultsStore
    assert status['status']=='segment_complete' and status['time_s']==220. and status['error'] is None
    store=ResultsStore(out/'run');ref=ResultsStore(source.parent/'run')
    f,g=store.read(0),ref.read(0)
    for k in ('q','z','W','mass','area'):assert np.array_equal(f.arrays[k],g.arrays[k]),k
    for k in ('vertices','offsets','levels'):assert np.array_equal(f.mesh[k],g.mesh[k]),k
    diag=json.loads((out/'run/DIAGNOSTICS.json').read_text());last=store.read(-1)
    assert all(abs(d['water_residual_m3'])<=d['water_roundoff_limit_m3'] and
        max(abs(v) for v in d['class_residual_kg'])<=d['class_roundoff_limit_kg'] and not any(d['wall_integrals']) for d in diag)
    assert np.isfinite(last.q).all() and last.q[0].min()>=0 and last.arrays['mass'].min()>=0
    save(out/'CHECK.json',dict(status=status,initial_mesh_exact=True,runtime=sha(RUNTIME/'RUNTIME.json'),
        water_residual=max(abs(d['water_residual_m3']) for d in diag),
        sediment_residual=max(max(abs(v) for v in d['class_residual_kg']) for d in diag),elapsed=elapsed))
    print('PASS',out.name,flush=True)


if __name__=='__main__':main()
