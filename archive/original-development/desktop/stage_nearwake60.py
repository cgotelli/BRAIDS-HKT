"""Stage only the cumulative arithmetic check; preserve active/frozen runtimes."""
from nearwake60 import *
import shutil
TARGET=ROOT/'desktop/qualification/nearwake60/candidate'


def stage():
    old=read(RUNTIME/'RUNTIME.json');TARGET.mkdir(parents=True,exist_ok=False)
    for n,h in old['files'].items():
        assert sha(RUNTIME/n)==h,n
        dest=TARGET/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(RUNTIME/n,dest)
    name='braids_core/sediment/adaptive_river_worker.py'
    before=(TARGET/name).read_text(encoding='utf-8');after=(ROOT/name).read_text(encoding='utf-8')
    # No unrelated checkout edits are allowed into this staged worker.
    a=before.index('    water_tol=max(1e-9,eps*max(abs(water_initial)')
    b=before.index("    if getattr(a,'_physical_toe_failure'",a)
    c=after.index('    from .river_budget_roundoff import cumulative_water_roundoff')
    d=after.index("    if getattr(a,'_physical_toe_failure'",c)
    assert before[:a]==after[:c] and before[b:]==after[d:]
    shutil.copy2(ROOT/name,TARGET/name)
    newname='braids_core/sediment/river_budget_roundoff.py';shutil.copy2(ROOT/newname,TARGET/newname)
    version=TARGET/'braids_desktop/__init__.py';s=version.read_text();assert 'preview58' in s
    version.write_text(s.replace('preview58','preview59'),encoding='utf-8')
    new=dict(old,files={n:sha(TARGET/n) for n in list(old['files'])+[newname]});save(TARGET/'RUNTIME.json',new)
    changed=[n for n in new['files'] if new['files'][n]!=old['files'].get(n)]
    assert set(changed)=={name,newname,'braids_desktop/__init__.py'}
    save(REPORT/'ARITHMETIC_STAGE.json',dict(parent_runtime=sha(RUNTIME/'RUNTIME.json'),runtime_sha256=sha(TARGET/'RUNTIME.json'),
        changed=changed,physical_fields_unchanged=True,per_step_checks_unchanged=True,
        incident='zero8_gpu cumulative check at 250s, 50 accepted outer updates'))
    print(TARGET)


if __name__=='__main__':stage()
