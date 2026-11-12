"""Stage authenticated preview35; never copy simulations or Delft3D sources."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'desktop/runtime';OUT=ROOT/'validation/_bundle207';TARGET=OUT/'candidate_v3'
BASE_HASH='c3ee9ec7c98539e62bb19e273f99a93580c26b35437fc2e8cff67ba895d96dee'
FILES=('braids_core/river_model.py','braids_core/river_project.py','braids_core/river_worker.py',
       'braids_core/river_jiang_cao.py','braids_core/sediment/capacity_parameters.py',
       'braids_core/sediment/grain_diameter_contract.py','braids_desktop/bedload_catalog.py',
       'braids_desktop/forms.py','braids_desktop/results.py','braids_desktop/river_physics.py',
       'braids_desktop/sediment_workflow.py','braids_desktop/__init__.py')
FILES+=tuple(p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'braids_core/sediment').glob('jiang_cao*')) if p.is_file())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def verify(root):
    root=Path(root).resolve();m=json.loads((root/'RUNTIME.json').read_text())
    for n,h in m['files'].items():assert (root/n).resolve().is_relative_to(root) and sha(root/n)==h,n
    return m
def main():
    assert sha(BASE/'RUNTIME.json')==BASE_HASH
    m=verify(BASE);assert not TARGET.exists(),'Use a new stage directory, retaining tested candidates'
    for n in m['files']:
        dst=TARGET/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dst)
    for n in FILES:
        dst=TARGET/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst);m['files'][n]=sha(dst)
    table=json.loads((BASE/'PRECOMPILED.json').read_text())
    # Preserve every existing kernel binding. Add only captured new kernels.
    records=json.loads((ROOT/'validation/_desktop207_probe/front_cpu/NATIVE_RECORDS.json').read_text())
    dependencies={p.name:p for p in (BASE/'precompiled').rglob('*.dll') if p.name in ('libgomp-1.dll','libgfortran-5.dll','libquadmath-0.dll','libgcc_s_seh-1.dll','libwinpthread-1.dll','libstdc++-6.dll')}
    def library(original,native=False):
        source=Path(original);dest=TARGET/'precompiled'/sha(source)[:20];dest.mkdir(parents=True,exist_ok=True)
        payload=[source,*dependencies.values()]
        if native:payload += [p for p in source.parent.iterdir() if p.is_file() and p.suffix in ('.f90','.json')]
        for p in payload:
            dst=dest/p.name
            if dst.exists():assert sha(dst)==sha(p)
            else:shutil.copy2(p,dst)
            m['files'][dst.relative_to(TARGET).as_posix()]=sha(dst)
        return dict(file=(dest/source.name).relative_to(TARGET).as_posix(),sha256=sha(source))
    added=[]
    for key,row in records.items():
        if key in table:continue
        if 'libraries' in row:row['libraries']={k:library(v) for k,v in row['libraries'].items()}
        elif 'native' in row:row['native']=library(row['native'],True)
        else:raise ValueError('Unreviewed native binding: '+key)
        table[key]=row;added.append(key)
    save(TARGET/'PRECOMPILED.json',table);m['files']['PRECOMPILED.json']=sha(TARGET/'PRECOMPILED.json')
    save(TARGET/'RUNTIME.json',m);verify(TARGET)
    save(OUT/'STAGE.json',dict(previous=BASE_HASH,runtime=sha(TARGET/'RUNTIME.json'),files=FILES,new_native=added))
    print('Staged',sha(TARGET/'RUNTIME.json'),len(m['files']),'payloads',flush=True)
if __name__=='__main__':main()
