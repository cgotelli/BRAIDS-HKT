"""Retain reproducible source and metadata without copying simulation outputs."""
from package_workflow61 import *

for name in ['GATES.json','INSTALL.json','INSTALLED_CHECK.json','VISUAL_CHECK.json','UI_CHECK.json']:
    assert read(REPORT/name)['status']=='passed',name
assert sha(BASE/'RUNTIME.json')==read(REPORT/'INSTALL.json')['runtime_sha256']
snapshot=REPORT/'source_snapshot';snapshot.mkdir(exist_ok=False)
for name in read(REPORT/'STAGE.json')['changed']:
    dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(TARGET/name,dest)
shutil.copy2(ROOT/'desktop/PROVENANCE.json',REPORT/'DELIVERY_PROVENANCE.json')
repros=['qualify_workflow61.py','test_workflow61.py','check_workflow61.py','workflow61_ui.py','inspect_workflow61.py','figures_workflow61.py','package_workflow61.py','verify_workflow61_installed.py','seal_workflow61.py']
files=[p for p in REPORT.rglob('*') if p.is_file() and not {'ui_settings','inspection_settings'}&set(p.parts) and p.name!='FINAL_MANIFEST.json']
files.extend(ROOT/'desktop'/p for p in repros);files.append(ROOT/'tests/test_desktop_workflow61.py')
save(REPORT/'FINAL_MANIFEST.json',dict(schema=1,version='0.1.0-preview60',runtime_sha256=sha(BASE/'RUNTIME.json'),files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}))
print('Workflow61 sources, evidence and identities sealed',flush=True)
