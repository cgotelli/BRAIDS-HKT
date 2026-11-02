import os, sys
from pathlib import Path
base=Path(sys.prefix)
_dll_handles=[]
for folder in [base,base/'Library/bin']:
    if folder.is_dir():
        _dll_handles.append(os.add_dll_directory(str(folder)))
os.environ['PATH']=str(base/'Library/bin')+os.pathsep+str(base)+os.pathsep+os.environ.get('SystemRoot','C:/Windows')+'/System32'
os.environ['PYTHONNOUSERSITE']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUPY_CACHE_DIR']=str(Path(os.environ['LOCALAPPDATA'])/'BRAIDS-HKT/cache/cupy')
sys.dont_write_bytecode=True
