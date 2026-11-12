"""Isolated packaged entry; verify all source/binary bytes before computation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True

def verify():
    path=ROOT/'RUNTIME.json';doc=json.loads(path.read_text())
    if doc.get('schema')!=1 or doc.get('route')!='combined_adaptive_river_v1':raise ValueError('Unknown runtime contract')
    for name,digest in doc['files'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Runtime artifact differs: '+name)
    return doc

def main():
    verify()
    from braids_core.sediment.portable_river_runtime import install
    install(json.loads((ROOT/'PRECOMPILED.json').read_text()),ROOT)
    if len(sys.argv)>1 and sys.argv[1]=='prepare-gis':
        p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('--recipe',required=True);p.add_argument('--output',required=True);p.add_argument('--stop-file')
        a=p.parse_args()
        from braids_core.sediment.adaptive_river_gis import prepare_recipe
        print(prepare_recipe(json.loads(Path(a.recipe).read_text()),a.output,a.stop_file));return
    from braids_core.sediment.adaptive_river_worker import main as execute
    try:execute()
    finally:
        if len(sys.argv)>1 and sys.argv[1]=='run' and '--input' in sys.argv and '--output' in sys.argv:
            source=Path(sys.argv[sys.argv.index('--input')+1]);out=Path(sys.argv[sys.argv.index('--output')+1])
            if out.is_dir() and (source/'profile.json').is_file():
                from braids_gpu.atomic_output import write_json_atomic
                p=json.loads((source/'profile.json').read_text())
                write_json_atomic(out/'RUN_CONTEXT.json',dict(crs=p['domain']['spatial_reference'],input=str(source.resolve()),
                    runtime_sha256=hashlib.sha256((ROOT/'RUNTIME.json').read_bytes()).hexdigest(),route='combined_adaptive_river_v1'))

if __name__=='__main__':main()
