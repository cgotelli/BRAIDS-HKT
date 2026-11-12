"""Fresh compiler-free package runs, without solver hooks or physics substitution."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',required=True,choices=('cpu','gpu'))
    ap.add_argument('--suite',choices=('walls','groin'),required=True)
    ap.add_argument('--law',default='mpm',choices=('mpm','ashida_michiue_generalized','wang_fredsoe'))
    ap.add_argument('--isolated',action='store_true');ap.add_argument('--runtime',type=Path,default=ROOT/'validation/_delivery172/candidate')
    ap.add_argument('--tag',default='package');a=ap.parse_args()
    # The reference must remain preview26 even after installing the release.
    runtime=(ROOT/'validation/_delivery172/preview26_runtime') if a.isolated else a.runtime.resolve()
    sys.path[:0]=[str(runtime),str(ROOT),str(ROOT/'desktop')]
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    from braids_core import river_worker
    from braids_desktop.results import ResultsStore
    from braids_desktop.preparation import prepare
    import numpy as np
    from stage_wall172 import sha,save
    verify();install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)
    if a.isolated:
        import wall_pressure170 as prototype
        if a.backend=='cpu':
            # Relocate only the frozen prototype's build lookup. Its sealed
            # original script predates installation and hardcodes desktop/runtime.
            import hashlib
            table=json.loads((runtime/'PRECOMPILED.json').read_text())
            row=next(v for k,v in table.items() if 'shoreline_hydraulic_probe.build' in k
                     and v['native']['file']=='precompiled/d440a06059c179ec44ed/hydraulics.dll')
            source=runtime/Path(row['native']['file']).parent/'shoreline_probe.f90'
            digest=hashlib.sha256(prototype.native_source(source.read_text()).encode()).hexdigest()
            build=ROOT/'validation/_interface170/build'/digest[:20]
            identity=json.loads((build/'IDENTITY.json').read_text())
            assert identity['source_sha256']==digest and sha(build/'hydraulics.dll')==identity['library_sha256']
            prototype.build=lambda:(build/'hydraulics.dll',identity,row['identity']['command'][0])
        prototype.install(a.backend)
    else:
        original_popen=subprocess.Popen
        def no_compiler(args,*p,**kw):
            name=str(args[0] if isinstance(args,(list,tuple)) else args).lower()
            if any(x in name for x in ('gfortran','g++','gcc','clang')):raise AssertionError('Release requested external compiler: '+name)
            return original_popen(args,*p,**kw)
        subprocess.Popen=no_compiler
    if a.suite=='walls':
        specs=[('v1','reflection_17'),('v1','wetting_37'),('amr_v1','wetting_37'),('lake_v1','wetting_37'),('vacuum_v1','reflection_37')]
    else:specs=[('groin',a.law)]
    for tag,name in specs:
        out=ROOT/f'validation/_delivery172/{a.tag}/{tag}_{name}_{a.backend}'
        out.mkdir(parents=True,exist_ok=False)
        if tag=='groin':
            source=ROOT/f'validation/_shoal165/groin/{name}_{a.backend}/project.json'
            reference=ROOT/f'validation/_interface170/runs/pressure_v2/{a.backend}/run' if name=='mpm' else None
        else:
            old=ROOT/f'validation/_wall171/{tag}/{name}_{a.backend}'
            source=old/'inputs/project/braids_project.json';reference=old/'full'
        doc=json.loads(source.read_text());dem=Path(doc['terrain']['dem_path'])
        doc['terrain']['dem_path']=str(dem if dem.is_absolute() else (source.parent/dem).resolve())
        save(out/'project.json',doc)
        prepared=out/'prepared';prepare(out/'project.json',prepared,asset_cache=out.parent/'asset_cache')
        owners=[];original=river_worker.core.factory
        def factory(*args,**kw):
            make=original(*args,**kw)
            def build(*args,**kw):
                owner=make(*args,**kw);owners.append(owner)
                if not a.isolated:assert owner.identity['reflecting_wall']['method']=='exact_reflecting_wall_riemann_v1'
                save(out/'OWNER.json',owner.identity)
                return owner
            build.__dict__.update(make.__dict__);return build
        river_worker.core.factory=factory
        status=river_worker.run(prepared,out/'full',checkpoint_dir=out/'checkpoints')
        assert status['status']=='complete'
        restart={};comparison={}
        if not a.isolated:
            stop=out/'STOP';split=45. if tag=='groin' else 2.
            def cancel(folder,row):
                if row['time_s']>=split:stop.touch()
            part=river_worker.run(prepared,out/'cancelled',stop_file=stop,on_frame=cancel)
            assert part['status']=='cancelled' and part['time_s']==split,part
            resume=river_worker.run(prepared,out/'resumed',restart=out/'cancelled/checkpoint.npz')
            x=ResultsStore(out/'full').read(-1);y=ResultsStore(out/'resumed').read(-1)
            exact={k:bool(np.array_equal(x.arrays[k],y.arrays[k])) for k in ('q','z','mass','water','bed_boundary')}
            assert resume['status']=='complete' and all(exact.values()),exact
            restart=dict(cancel=part,resume=resume,exact=exact)
            if reference is None:reference=ROOT/f'validation/_delivery172/isolated/{tag}_{name}_{a.backend}/full'
            old=ResultsStore(reference);new=ResultsStore(out/'full');assert np.array_equal(old.times,new.times)
            diffs={k:0. for k in ('q','z','mass','water','bed_boundary')}
            for i in range(len(old.times)):
                x=old.read(i);y=new.read(i)
                for k in diffs:diffs[k]=max(diffs[k],float(abs(x.arrays[k]-y.arrays[k]).max(initial=0)))
            assert max(diffs.values())==0,diffs
            comparison=dict(reference=str(reference.relative_to(ROOT)),maximum_difference=diffs,exact=True)
        river_worker.core.factory=original
        save(out/'CHECK.json',dict(status=status,restart=restart,comparison=comparison,
            isolated=a.isolated,runtime_sha256=sha(runtime/'RUNTIME.json'),script_sha256=sha(__file__)))
        print(tag,name,a.backend,'complete',comparison.get('exact'),flush=True)


if __name__=='__main__':main()
