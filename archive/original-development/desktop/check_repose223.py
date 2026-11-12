"""Run the actual compiler-free package with preserved, named receipts."""
import os,sys,json,importlib
from pathlib import Path
from stage_repose223 import ROOT,OUT,TARGET,BASE,save,sha
runtime=BASE if '--baseline' in sys.argv else TARGET
if '--baseline' in sys.argv:sys.argv.remove('--baseline')
sys.path[:0]=[str(runtime),str(ROOT),str(ROOT/'tests'),str(ROOT/'desktop')]
os.environ['QT_QPA_PLATFORM']='offscreen';os.environ['OMP_NUM_THREADS']='4'
os.environ['BRAIDS_DESKTOP_TEST_RUNTIME']=str(runtime)
os.environ['BRAIDS_MF_RUNTIME']=sha(runtime/'RUNTIME.json')
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
verify();install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)
# Operator-only tests precede the production hydraulic constructor, which
# normally loads these verified bundled dependencies. No system compiler/DLL.
import ctypes
omp=next((runtime/'precompiled').rglob('libgomp-1.dll'))
dll_context=os.add_dll_directory(str(omp.parent));omp_library=ctypes.CDLL(str(omp))
for name in ('validation','validation.sediment','validation.sediment.river_morphodynamics'):
    module=importlib.import_module(name);module.__path__.append(str(ROOT/Path(*name.split('.'))))
name=os.environ.get('BRAIDS_COLLAPSE_CHECK','baseline' if runtime==BASE else 'qualification')
folder=OUT/name
if folder.exists():raise FileExistsError('Preserve evidence: '+str(folder))
os.environ['BRAIDS_MF_OUTPUT']=str(folder)
import pytest
code=pytest.main(['-q',*(sys.argv[1:] or ['tests/test_repose223.py']), '--tb=short','--maxfail=5',
    '--basetemp='+str(folder),'--junitxml='+str(folder.with_suffix('.xml'))])
raise SystemExit(code)

