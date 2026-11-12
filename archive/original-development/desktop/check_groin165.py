"""Fresh release-candidate replay of the retained three-law groin regression."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['cpu','gpu'],required=True)
    ap.add_argument('--runtime',type=Path,default=ROOT/'validation/_shoal165/candidate3')
    a=ap.parse_args();runtime=a.runtime.resolve()
    sys.path[:0]=[str(runtime),str(ROOT)]
    from braids_core.sediment.portable_river_runtime import install
    from braids_core import river_worker
    from braids_gpu.atomic_output import write_json_atomic
    install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)
    for law in ('mpm','ashida_michiue_generalized','wang_fredsoe'):
        source=ROOT/f'validation/_hydraulics162/mobile/braids_{law}_{a.backend}/project'
        doc=json.loads((source/'braids_project.json').read_text())
        doc['terrain']['dem_path']=str((source/doc['terrain']['dem_path']).resolve())
        out=ROOT/f'validation/_shoal165/groin/{law}_{a.backend}'
        out.mkdir(parents=True,exist_ok=False)
        write_json_atomic(out/'project.json',doc)
        prepared=river_worker.prepare(out/'project.json',out/'prepared')
        status=river_worker.run(prepared,out/'run')
        write_json_atomic(out/'STATUS.json',status)
        print(law,a.backend,status['status'],flush=True)


if __name__=='__main__':main()
