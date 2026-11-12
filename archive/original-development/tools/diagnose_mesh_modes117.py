"""Read-only installed runtime instrumentation; never patches the package."""
import sys,json,traceback
from pathlib import Path
runtime=Path('C:/Users/STREEM/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/braids_qgis/river_runtime')
sys.path.insert(0,str(runtime))
from braids_core.river_launcher import verify
verify()
from braids_core.sediment.portable_river_runtime import install
install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)
from braids_core.sediment.shoreline_solver import ShorelineSolver
from braids_core.sediment.bend_methods_v1 import BendTrialRejected
from braids_core import river_worker
original=ShorelineSolver.step
def trace(self,*a,**kw):
    try:return original(self,*a,**kw)
    except BendTrialRejected:raise
    except BaseException:
        traceback.print_exc();raise
ShorelineSolver.step=trace
river_worker.run(sys.argv[1],sys.argv[2])
