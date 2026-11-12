"""Freeze a separate inlet candidate, never overwrite v6 or installed Desktop."""
import json,shutil,sys
from pathlib import Path
from stage_secondary185 import ROOT,save,sha
BASE=ROOT/'validation/_secondary185/candidate'
TARGET=ROOT/'validation/_inlet187/prescribed_candidate'
FILES=('braids_core/sediment/river_secondary_clock.py','braids_core/sediment/river_secondary_clock.h',
       'braids_core/river_project.py','braids_core/river_open_boundary.py','braids_core/river_boundary_draft.py',
       'braids_desktop/river_boundaries.py','braids_desktop/forms.py')

def main():
    assert 'self.secondary_intensity=' in (ROOT/'braids_desktop/river_boundaries.py').read_text(encoding='utf-8'), (
        'The pending editor is frozen in prescribed_candidate, not live beside preview29. '
        'Apply its reviewed UI delta only when delivering the matching runtime.')
    assert sha(BASE/'RUNTIME.json')=='1b078961ad95f0038ba07da0a4aca9c3dd15a7f3cbffff2ec413986d27b4723b'
    manifest=json.loads((BASE/'RUNTIME.json').read_text())
    for n,h in manifest['files'].items():assert sha(BASE/n)==h,n
    assert not TARGET.exists(),'Candidate is immutable: use a new folder for a revised build'
    shutil.copytree(BASE,TARGET,ignore=shutil.ignore_patterns('__pycache__'))
    # The installed runtime does not contain the Desktop package. Freeze the
    # standalone editor separately here so its new contract is tested in isolation.
    shutil.copytree(ROOT/'braids_desktop',TARGET/'braids_desktop',ignore=shutil.ignore_patterns('__pycache__'))
    for p in (TARGET/'braids_desktop').rglob('*'):
        if p.is_file():manifest['files'][p.relative_to(TARGET).as_posix()]=sha(p)
    for n in FILES:
        shutil.copy2(ROOT/n,TARGET/n);manifest['files'][n]=sha(TARGET/n)
    sys.path[:0]=[str(TARGET),str(ROOT)]
    from braids_core.sediment.river_secondary_clock import NAMES
    from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
    from braids_core.sediment.portable_river_runtime import _bindings
    source=TARGET/FILES[1];op=WaveEngine('cpu',4,source=source,names=NAMES);row=_bindings(op)
    table=json.loads((BASE/'PRECOMPILED.json').read_text())
    existing=next(r for k,r in table.items() if k.startswith('validation.sediment.river_morphodynamics.sediment_wave.WaveEngine:'))
    librel=Path(next(iter(existing['libraries'].values()))['file'])
    deps=[p for p in (BASE/librel.parent).glob('*.dll') if p.name!=librel.name]
    for attr,original in list(row['libraries'].items()):
        lib=Path(original);package=TARGET/'precompiled'/sha(lib)[:20];package.mkdir(exist_ok=True)
        for p in [lib,*deps]:
            dst=package/p.name
            if not dst.exists():shutil.copy2(p,dst)
            assert sha(p)==sha(dst)
            manifest['files'][dst.relative_to(TARGET).as_posix()]=sha(dst)
        row['libraries'][attr]=dict(file=(package/lib.name).relative_to(TARGET).as_posix(),sha256=sha(lib))
    table['validation.sediment.river_morphodynamics.sediment_wave.WaveEngine:'+sha(source)+':'+','.join(NAMES)]=row
    save(TARGET/'PRECOMPILED.json',table);manifest['files']['PRECOMPILED.json']=sha(TARGET/'PRECOMPILED.json')
    save(TARGET/'RUNTIME.json',manifest)
    save(TARGET.parent/'PRESCRIBED_STAGE.json',dict(baseline=sha(BASE/'RUNTIME.json'),candidate=sha(TARGET/'RUNTIME.json'),files=FILES))
    print(sha(TARGET/'RUNTIME.json'),flush=True)

if __name__=='__main__':main()
