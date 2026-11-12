"""Package the unchanged prescribed/equilibrium bed boundary CPU kernel."""
import hashlib,inspect,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];runtime=ROOT/'validation/_bedload157/candidate'
sys.path.insert(0,str(runtime))
from validation.sediment.river_morphodynamics import sediment_wave
from braids_core.sediment.portable_river_runtime import _bindings,_key

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=runtime/'braids_core/river_boundary.h'
engine=sediment_wave.WaveEngine('cpu',4,source=source,names=('river_prescribed_boundary',))
bound=inspect.signature(sediment_wave.WaveEngine.__init__).bind(engine,'cpu',4,source=source,names=('river_prescribed_boundary',))
bound.apply_defaults();key=_key(sediment_wave,'WaveEngine',bound.arguments)
row=_bindings(engine);manifest=json.loads((runtime/'RUNTIME.json').read_text())
for name,filename in row['libraries'].items():
    src=Path(filename);folder=runtime/'precompiled'/sha(src)[:20];folder.mkdir(parents=True,exist_ok=True)
    files=[src,*[Path(shutil.which('g++')).parent/n for n in ('libgomp-1.dll','libwinpthread-1.dll')]]
    for file in files:
        target=folder/file.name
        if target.exists():assert sha(target)==sha(file)
        else:shutil.copy2(file,target)
        manifest['files'][target.relative_to(runtime).as_posix()]=sha(target)
    row['libraries'][name]=dict(file=(folder/src.name).relative_to(runtime).as_posix(),sha256=sha(src))
path=runtime/'PRECOMPILED.json';table=json.loads(path.read_text());table[key]=row
path.write_text(json.dumps(table,indent=2)+'\n',encoding='utf-8');manifest['files']['PRECOMPILED.json']=sha(path)
(runtime/'RUNTIME.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(key,sha(runtime/'RUNTIME.json'))
