"""Matched current-law temporal qualification with isolated cell-local drag."""
from pathlib import Path
import sys,hashlib,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from validation.hydraulics.resistance126 import cell_drag
from tools import profile_explicit_fwave124

if __name__=='__main__':
    out=Path(sys.argv[sys.argv.index('--output')+1])
    cell_drag.install_factory()
    try:profile_explicit_fwave124.main()
    finally:
        if (out/'EXPERIMENT.json').exists():
            from braids_gpu.atomic_output import write_json_atomic
            p=out/'EXPERIMENT.json';d=json.loads(p.read_text());d['scope']='resistance126_research'
            d['hydraulic_resistance']=cell_drag.METHOD
            for f in (Path(__file__),Path(cell_drag.__file__),Path(cell_drag.__file__).with_suffix('.h')):
                d['sources'][str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
            write_json_atomic(p,d)
