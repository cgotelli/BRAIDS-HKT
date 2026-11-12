"""Replay historical tests with only the original alpha136 geometry functions.

No files/runtime/source identities are patched. This isolates whether failures
are introduced by the geometry changes, while retaining the same checkout and
historical fixture paths as the candidate run.
"""
from pathlib import Path
import importlib.util
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def load(name):
    path=ROOT/'dist/unified-alpha136/braids_qgis/river_runtime/braids_core/sediment'/f'{name}.py'
    spec=importlib.util.spec_from_file_location('braids_core.sediment._baseline137_'+name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod


if __name__=='__main__':
    from braids_core.sediment import river_polygon_banks as banks,river_amr_transfer as transfer
    old=load('river_polygon_banks')
    banks.topology=old.topology
    banks.PolygonProcesses.__init__=old.PolygonProcesses.__init__
    transfer.CellMap.polygons=classmethod(load('river_amr_transfer').CellMap.polygons.__func__)
    import pytest
    raise SystemExit(pytest.main(['tests/test_river_polygon_banks.py','tests/test_river_amr_transfer.py',
        'tests/test_river_dynamic_leaf.py','tests/test_river_automatic_leaf.py','tests/test_river_mixed_leaf.py',
        '-q','--tb=short','--junitxml=validation/_geometry137/HISTORICAL_BASELINE.xml']))
