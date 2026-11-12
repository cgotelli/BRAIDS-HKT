"""Fresh six-law CPU/GPU groin checks, schema guard, restart and UI evidence."""
import argparse,json,os,sys,time
from pathlib import Path
from stage_grains191 import ROOT,BASE,OUT,TARGET,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runtime',type=Path,default=TARGET)
    ap.add_argument('--tag',required=True);ap.add_argument('--backend',choices=('cpu','gpu'),required=True)
    args=ap.parse_args();runtime=args.runtime.resolve()
    os.environ['QT_QPA_PLATFORM']='offscreen';os.environ['OMP_NUM_THREADS']='4'
    os.environ['BRAIDS_AUDIT_RUNTIME']=str(runtime)
    sys.path[:0]=[str(runtime),str(ROOT),str(ROOT/'desktop')]
    import numpy as np
    from PySide6.QtWidgets import QApplication
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    from braids_core import river_worker
    from braids_desktop import services
    from braids_desktop.forms import SettingsPanel
    from braids_desktop.river_physics import set_choice
    from braids_desktop.bedload_catalog import CATALOG
    from braids_desktop.preparation import prepare
    from braids_desktop.results import ResultsStore
    from check_boundaries159 import document
    verify();install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)
    import subprocess
    original=subprocess.Popen
    def guard(command,*a,**kw):
        executable=str(command[0] if isinstance(command,(list,tuple)) else command).lower()
        assert not any(s in executable for s in ('gfortran','g++','gcc','clang')),executable
        return original(command,*a,**kw)
    subprocess.Popen=guard
    app=QApplication.instance() or QApplication([])
    root=OUT/args.tag/args.backend;root.mkdir(parents=True,exist_ok=False)
    checks={}
    for method in CATALOG:
        name=method.removeprefix('capacity.');folder=root/name
        doc=document(folder/'project',args.backend,'stage')
        doc['terrain']['dem_path']=str((folder/'project'/doc['terrain']['dem_path']).resolve())
        f=SettingsPanel();f.load_document(doc);assert not f.sediment_enabled.isChecked()
        f.sediment_enabled.setChecked(True);p=f.physics;p.bedload_on.setChecked(True)
        set_choice(p.capacity,method);p.grains.item(0,1).setText('5')
        f.end_time.setValue(2.);f.output_interval.setValue(.5);f.dt_max.setValue(.05)
        f.sediment_start.setValue(.1);f.morfac.setValue(1.)
        f.checkpoint_duration.setChecked(False);f.checkpoint_interval.setValue(1.)
        doc=f.document()
        services.save_project_folder(folder/'project',doc)
        loaded=services.load_project_folder(folder/'project')
        g=SettingsPanel();g.load_document(loaded)
        assert g.physics.selection()==p.selection();g.close()
        if runtime!=BASE:
            # Reject wrong/missing metadata before creating any preparation output.
            for suffix,bad in (('wrong','d10'),('missing',None)):
                broken=json.loads(json.dumps(doc));grain=broken['model']['request']['sediment']['selection']['grains'][0]
                if bad is None:del grain['diameter_definition']
                else:grain['diameter_definition']=bad
                path=folder/(suffix+'.json');path.write_text(json.dumps(broken))
                try:prepare(path,folder/suffix,asset_cache=OUT/'assets')
                except ValueError as exc:
                    assert 'diameter' in str(exc),str(exc)
                    assert not (folder/suffix).exists()
                else:raise AssertionError('Invalid grain reached mesh preparation')
            old=json.loads(json.dumps(doc));old['schema']='braids.river-project.v25'
            try:services.validate_document(old)
            except ValueError as exc:assert 'Old projects' in str(exc)
            else:raise AssertionError('Old project admitted')
            if args.backend=='cpu':
                p.material_box.setParent(None);p.material_box.resize(680,450);p.material_box.show()
                for _ in range(4):app.processEvents()
                p.material_box.grab().save(str(folder/'GRAIN_TABLE.png'));p.material_box.close()
        f.close()
        prep=folder/'prepared';prepare(folder/'project/braids_project.json',prep,asset_cache=OUT/'assets')
        began=time.perf_counter();status=river_worker.run(prep,folder/'full')
        assert status['status']=='complete',status
        frame=ResultsStore(folder/'full').read(-1)
        assert np.isfinite(frame.q).all() and frame.q[0].min()>=0 and frame.arrays['mass'].min()>=0
        diag=json.loads((folder/'full/DIAGNOSTICS.json').read_text())
        water=max(abs(d['water_residual_m3']) for d in diag)
        sediment=max(np.max(abs(np.asarray(d['class_residual_kg']))) for d in diag)
        assert water<1e-8 and sediment<1e-7
        assert all(not any(d['wall_integrals']) for d in diag)
        exact={}
        if runtime!=BASE:
            first=river_worker.run(prep,folder/'prefix',until=1.)
            assert first['time_s']==1.
            resumed=river_worker.run(prep,folder/'resumed',restart=folder/'prefix/checkpoint.npz')
            assert resumed['status']=='complete'
            r=ResultsStore(folder/'resumed').read(-1)
            for k in ('q','z','mass','water','bed_boundary'):
                exact[k]=bool(np.array_equal(frame.arrays[k],r.arrays[k]))
            assert all(exact.values()),exact
            # A change in diameter semantics cannot reuse a checkpoint identity.
            ref=OUT/'reference_v2'/args.backend/name
            try:river_worker.read_checkpoint(ref/'full/checkpoint.npz',
                river_worker.checkpoint_identity(*river_worker.load_prepared(prep)))
            except ValueError:pass
            else:raise AssertionError('Old checkpoint admitted')
            old_frame=ResultsStore(ref/'full').read(-1)
            equal={k:bool(np.array_equal(frame.arrays[k],old_frame.arrays[k])) for k in exact}
            assert all(equal.values()),equal
        else:equal={}
        checks[name]=dict(water_residual_m3=water,sediment_residual_kg=float(sediment),
            runtime_s=time.perf_counter()-began,restart_exact=exact,previous_release_fields_exact=equal,
            grain=doc['model']['physics']['sediment']['grains'][0])
        (root/'CHECK.json').write_text(json.dumps(dict(runtime=sha(runtime/'RUNTIME.json'),checks=checks),indent=2))
        print(args.tag,args.backend,name,'PASS',flush=True)
if __name__=='__main__':main()
