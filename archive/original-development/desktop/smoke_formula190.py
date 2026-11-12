"""Authenticated delivered runtime, fresh worker and original checkpoints."""
import argparse,os,sys,json,subprocess
from pathlib import Path
from stage_formula190 import ROOT,OUT
from stage_secondary185 import save,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runtime',type=Path,required=True)
    ap.add_argument('--tag',required=True);ap.add_argument('--backend',choices=('cpu','gpu'))
    args=ap.parse_args();runtime=args.runtime.resolve()
    os.environ['OMP_NUM_THREADS']='4';os.environ['QT_QPA_PLATFORM']='offscreen'
    if args.backend is None:
        for backend in ('cpu','gpu'):
            subprocess.run([sys.executable,__file__,'--runtime',str(runtime),'--tag',args.tag,
                            '--backend',backend],check=True,cwd=ROOT)
        return
    sys.path[:0]=[str(runtime),str(ROOT)]
    import numpy as np
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    from braids_core import river_worker
    from braids_desktop.preparation import prepare
    from braids_desktop.results import ResultsStore
    import braids_desktop
    verify();assert braids_desktop.__version__=='0.1.0-preview31'
    install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)
    original=subprocess.Popen
    def guard(command,*a,**kw):
        executable=str(command[0] if isinstance(command,(list,tuple)) else command).lower()
        assert not any(s in executable for s in ('gfortran','g++','gcc','clang')),executable
        return original(command,*a,**kw)
    subprocess.Popen=guard
    from PySide6.QtWidgets import QApplication
    from braids_desktop.forms import SettingsPanel
    from braids_desktop.river_physics import set_choice
    app=QApplication([]);f=SettingsPanel();assert not f.sediment_enabled.isChecked()
    f.sediment_enabled.setChecked(True);f.physics.bedload_on.setChecked(True)
    checks={}
    for law in ('ancey_recking_2023','recking_2010'):
        set_choice(f.physics.capacity,'capacity.'+law)
        assert f.physics.selection()['capacity']['method']=='capacity.'+law
        source=OUT/'runs/wet'/(law+'_'+args.backend)
        folder=OUT/args.tag/(law+'_'+args.backend)
        prepared=folder/'prepared'
        prepare(source/'project/braids_project.json',prepared,asset_cache=OUT/'assets')
        status=river_worker.run(prepared,folder/'prefix',until=6.)
        assert status['status']=='segment_complete' and status['time_s']==6.,status
        # No old-runtime checkpoint is reinterpreted after release metadata changes.
        try:river_worker.read_checkpoint(source/'cancelled/checkpoint.npz',
                    river_worker.checkpoint_identity(*river_worker.load_prepared(prepared)))
        except ValueError:pass
        else:raise AssertionError('Old-runtime checkpoint was accepted')
        output=folder/'resumed'
        status=river_worker.run(prepared,output,restart=folder/'prefix/checkpoint.npz',until=8.)
        assert status['status']=='segment_complete' and status['time_s']==8.,status
        original_store=ResultsStore(source/'full')
        a=original_store.read(list(original_store.times).index(8.));b=ResultsStore(output).read(-1)
        exact={k:bool(np.array_equal(a.arrays[k],b.arrays[k])) for k in ('q','z','mass','water','bed_boundary')}
        assert all(exact.values()),exact
        checks[law]=exact
    f.close()
    save(OUT/args.tag/(args.backend+'_CHECK.json'),dict(runtime=sha(runtime/'RUNTIME.json'),
        version=braids_desktop.__version__,compiler_free=True,sediment_default_off=True,
        old_checkpoint_rejected=True,fresh_prepare_and_6_to_8s_restart_exact=checks))
    print(args.tag,args.backend,'PASS')
if __name__=='__main__':main()
