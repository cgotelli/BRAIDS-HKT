"""Stage only the reviewed initial mesh handoff over Desktop preview2."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import stage_desktop_geometry137 as staging

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    staging.CHANGED=('braids_core/sediment/river_leaf_controller.py',
                     'braids_core/sediment/adaptive_river_worker.py')
    staging.ADDED=()
    staging.stage(ROOT/'validation/_geometry137/candidate3',args.output)
