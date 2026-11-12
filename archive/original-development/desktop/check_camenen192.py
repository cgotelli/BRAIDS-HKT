"""Fresh old-law trajectories must retain the installed preview32 fields."""
import argparse,json,os,sys
from pathlib import Path
from stage_camenen192 import ROOT,OUT,TARGET,sha
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=('cpu','gpu'),required=True)
    ap.add_argument('--runtime',type=Path,default=TARGET);ap.add_argument('--tag',default='controls')
    ap.add_argument('--exclude-law',default='camenen_larson_2005')
    ap.add_argument('--reference-root',type=Path,default=ROOT/'validation/_grains191/installed')
    ap.add_argument('--baseline',action='store_true')
    args=ap.parse_args();runtime=args.runtime.resolve()
    os.environ.update(QT_QPA_PLATFORM='offscreen',OMP_NUM_THREADS='4',BRAIDS_AUDIT_RUNTIME=str(runtime))
    sys.path[:0]=[str(runtime),str(ROOT),str(ROOT/'desktop')]
    import numpy as np
    from PySide6.QtWidgets import QApplication
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    from braids_core import river_worker
    from braids_desktop import services
    from braids_desktop.forms import SettingsPanel
    from braids_desktop.river_physics import set_choice
    from braids_desktop.preparation import prepare
    from braids_desktop.results import ResultsStore
    from check_boundaries159 import document
    from braids_desktop.bedload_catalog import CATALOG
    verify();install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)
    import subprocess
    original=subprocess.Popen
    def guard(command,*a,**kw):
        executable=str(command[0] if isinstance(command,(list,tuple)) else command).lower()
        assert not any(s in executable for s in ('gfortran','g++','gcc','clang')),executable
        return original(command,*a,**kw)
    subprocess.Popen=guard
    app=QApplication.instance() or QApplication([])
    root=OUT/args.tag/args.backend;root.mkdir(parents=True,exist_ok=False);checks={}
    for method in CATALOG:
        name=method.removeprefix('capacity.')
        if name==args.exclude_law:continue
        folder=root/name;doc=document(folder/'project',args.backend,'stage')
        doc['terrain']['dem_path']=str((folder/'project'/doc['terrain']['dem_path']).resolve())
        f=SettingsPanel();f.load_document(doc);assert not f.sediment_enabled.isChecked()
        f.sediment_enabled.setChecked(True);p=f.physics;p.bedload_on.setChecked(True)
        set_choice(p.capacity,method);p.grains.item(0,1).setText('5')
        f.end_time.setValue(2.);f.output_interval.setValue(.5);f.dt_max.setValue(.05)
        f.sediment_start.setValue(.1);f.morfac.setValue(1.)
        f.checkpoint_duration.setChecked(False);f.checkpoint_interval.setValue(1.)
        doc=f.document();f.close();services.save_project_folder(folder/'project',doc)
        prep=folder/'prepared';prepare(folder/'project/braids_project.json',prep,asset_cache=OUT/'assets')
        status=river_worker.run(prep,folder/'full');assert status['status']=='complete'
        frame=ResultsStore(folder/'full').read(-1)
        if args.baseline:
            checks[name]=dict(status=status,compiler_free=True)
            (root/'CHECK.json').write_text(json.dumps(dict(runtime=sha(runtime/'RUNTIME.json'),checks=checks),indent=2))
            print(args.tag,args.backend,name,'reference complete',flush=True)
            continue
        reference=args.reference_root/args.backend/name/'full'
        old=ResultsStore(reference).read(-1)
        exact={k:bool(np.array_equal(frame.arrays[k],old.arrays[k])) for k in ('q','z','mass','water','bed_boundary')}
        assert all(exact.values()),(method,exact)
        try:river_worker.read_checkpoint(reference/'checkpoint.npz',
            river_worker.checkpoint_identity(*river_worker.load_prepared(prep)))
        except ValueError:pass
        else:raise AssertionError('Previous-runtime checkpoint admitted')
        checks[name]=dict(previous_release_fields_exact=exact,old_restart_rejected=True)
        (root/'CHECK.json').write_text(json.dumps(dict(runtime=sha(runtime/'RUNTIME.json'),checks=checks),indent=2))
        print(args.tag,args.backend,name,'unchanged',flush=True)
if __name__=='__main__':main()
