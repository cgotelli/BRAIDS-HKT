"""Short public-route experiments, separate identities and immutable inputs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from check_bedload161 import save,sha
from braids_core.sediment.portable_river_runtime import install
from braids_core.sediment.bed_columns_v101 import host
from braids_core import river_worker
from braids_desktop.results import ResultsStore
from wall_bounds163 import install as wall_install
from wave164 import install as wave_install


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--backend',required=True,choices=['cpu','gpu'])
    ap.add_argument('--mode',default='matrix',choices=['matrix','hll','rusanov','bed_diagonal','morph_llf','capacity_diagonal'])
    ap.add_argument('--probe',action='store_true')
    ap.add_argument('--formula',default='mpm',choices=['mpm','ashida_michiue_generalized','wang_fredsoe'])
    ap.add_argument('--until',type=float,default=60.)
    ap.add_argument('--restart')
    ap.add_argument('--half-step',action='store_true')
    ap.add_argument('--tag',default='coupled')
    args=ap.parse_args()
    out=ROOT/f'validation/_wave164/{args.tag}/{args.mode}_{args.formula}_{args.backend}'
    out.mkdir(parents=True,exist_ok=False)
    parent=ROOT/'validation/_hydraulics162/mobile'
    install(json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text()),ROOT/'desktop/runtime')
    wall_install(args.backend);wave_install(args.mode,args.probe)
    original=river_worker.core.factory
    def factory(*a,**kw):
        make=original(*a,**kw)
        def build(*a,**kw):
            owner=make(*a,**kw)
            owner.identity['wave164_research']=dict(mode=args.mode,wall163=True,source_sha256=sha(ROOT/'desktop/wave164.py'))
            if args.probe:
                p=owner.physics;xp=owner.xp
                p.wave_diagnostic=xp.zeros((23,p.nf))
                for label in ('BRAIDS','FLOW_PRIMARY','FLOW_CHARACTERISTIC'):
                    old=np.load(ROOT/f'validation/_groin163/cpu/{label}_0040.npz')
                    q=xp.asarray(old['q']);z=xp.asarray(old['z']);water=xp.asarray(old['water'])
                    owner.bed_faces(q,z,xp.zeros(p.n),water)
                    fields={k:host(getattr(p,k)).copy() for k in ('wave_diagnostic','hydraulic_wave_trace','bedcell','capacity_gradient','class_flux','wave_physical_normal')}
                    fields.update({k:old[k] for k in ('q','z','xy','owner','neighbour','normal','mid','area','length','bid')})
                    np.savez(out/f'{label}.npz',**fields)
                p.wave_diagnostic=xp.zeros((3,p.nf)) if not args.probe else xp.zeros((23,p.nf))
            return owner
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory
    prepared=parent/f'braids_{args.formula}_{args.backend}/prepared'
    if args.half_step:
        from braids_desktop import services
        from braids_desktop.preparation import prepare
        original_project=parent/f'braids_{args.formula}_{args.backend}/project'
        doc=services.load_project_folder(original_project)
        dem=Path(doc['terrain']['dem_path'])
        if not dem.is_absolute():doc['terrain']['dem_path']=str((original_project/dem).resolve())
        doc['settings']['dt_max_s']*=.5
        doc['settings']['controls']['cfl']*=.5
        services.save_project_folder(out/'project',doc)
        prepared=out/'prepared'
        prepare(out/'project/braids_project.json',prepared,asset_cache=parent/'assets')
    status=river_worker.run(prepared,out/'run',
                            until=.001 if args.probe else args.until,restart=args.restart)
    save(out/'STATUS.json',dict(status=status,settings=vars(args),script_sha256=sha(__file__),
        wave_sha256=sha(ROOT/'desktop/wave164.py'),installed_runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json')))
    print(status)


if __name__=='__main__':main()
