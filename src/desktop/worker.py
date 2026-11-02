"""Desktop child bootstrap; configure GDAL policy before the verified worker."""
import runpy
import sys
import json
from pathlib import Path
from osgeo import osr
osr.UseExceptions()
launcher=sys.argv[1]
sys.argv=sys.argv[1:]
if len(sys.argv)>1 and sys.argv[1]=='prepare':
    namespace=runpy.run_path(launcher,run_name='_desktop_verified_launcher');namespace['verify']()
    runtime=Path(launcher).resolve().parents[1]
    from braids_core.sediment.portable_river_runtime import install
    install(json.loads((runtime/'PRECOMPILED.json').read_text(encoding='utf-8')),runtime)
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action');parser.add_argument('--project',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--stop-file');parser.add_argument('--asset-cache',required=True)
    args=parser.parse_args()
    from braids_desktop.preparation import prepare
    print(prepare(args.project,args.output,stop_file=args.stop_file,asset_cache=args.asset_cache))
else:runpy.run_path(launcher,run_name='__main__')
