"""Install the qualified JC outer-CFL predictor as Desktop preview50."""
from __future__ import annotations

import hashlib,json,shutil,subprocess
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'desktop/runtime';CANDIDATE=ROOT/'validation/_experiment229/candidate_v2'
OUT=ROOT/'validation/_experiment229';PROJECT=ROOT.parent/'BRAIDS_JiangCao_Experiment228'
OLD='f8f896a590b38f0a0563b2d04fb6db823948a42ad688829849a4dfab370da4c5'
NEW='63b330057b70c20f8066ebeded4b113495fec82e97c734b5bf97a9a9973572f9'
SOURCE='d55826eca99c82867a79744458157b24606f784a1e5e9102cdcfc33e320ed005'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
def verify(root):
    m=json.loads((root/'RUNTIME.json').read_text())
    for n,h in m['files'].items():
        if sha(root/n)!=h:raise ValueError('Runtime payload differs: '+str(root/n))
    return m
def processes():
    command=("Get-CimInstance Win32_Process | Where-Object {($_.Name -match '^python|^BRAIDS') -and "
        "($_.CommandLine -match 'desktop[\\\\/]launch.py|desktop[\\\\/]worker.py|braids_desktop.app')} | "
        "Select-Object -ExpandProperty ProcessId")
    return subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True).strip()


def publish(target,backup,changed):
    if backup.exists():raise FileExistsError(backup)
    for name in sorted(changed|{'RUNTIME.json'}):
        dst=backup/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target/name,dst)
    for name in sorted(changed):shutil.copy2(CANDIDATE/name,target/name)
    shutil.copy2(CANDIDATE/'RUNTIME.json',target/'RUNTIME.json');verify(target)
    if sha(target/'RUNTIME.json')!=NEW:raise ValueError('Published runtime identity differs')


def main():
    if processes():raise RuntimeError('Close BRAIDS Desktop before installation')
    if sha(BASE/'RUNTIME.json')!=OLD or sha(CANDIDATE/'RUNTIME.json')!=NEW:raise ValueError('Runtime base differs')
    old=verify(BASE);new=verify(CANDIDATE)
    changed={n for n,h in old['files'].items() if new['files'].get(n)!=h}
    if changed!={'braids_core/sediment/jiang_cao_desktop.py','braids_core/sediment/jiang_cao_coupled.py',
                 'braids_core/sediment/jiang_cao_wetdry.py'} or set(new['files'])!=set(old['files']):
        raise ValueError(sorted(changed))
    publish(BASE,OUT/'preview49_backup',changed)
    frozen=PROJECT/'runtime'
    if sha(frozen/'RUNTIME.json')!=OLD:raise ValueError('Project frozen runtime is not preview49')
    verify(frozen);publish(frozen,OUT/'project_preview49_backup',changed)
    provenance=json.loads((ROOT/'desktop/PROVENANCE.json').read_text())
    report='validation/sediment/river_morphodynamics/reports/desktop_experiment225/timestep229/REPORT.md'
    provenance.update(version='0.1.0-preview50',ui_qualification=report)
    provenance['solver'].update(source='validation/_experiment229/candidate_v2',runtime_manifest_sha256=NEW,
        qualification=report,modification_scope=(
            'Retain preview49 Jiang-Cao face-compatible capacity correction. Add a stage-wave outer-CFL predictor '
            'for the coupled Jiang-Cao clock, saving the accepted next-step hint in checkpoint diagnostics. '
            'CFL remains user-selected at 0.5; outer maximum 0.03 s. Physical formulas, QGIS and Delft3D unchanged.'))
    save(ROOT/'desktop/PROVENANCE.json',provenance)
    receipt=dict(version='0.1.0-preview50',runtime=NEW,source_identity=SOURCE,previous=OLD,
        changed=sorted(changed),desktop_backup=str(OUT/'preview49_backup'),
        project_backup=str(OUT/'project_preview49_backup'),physical_formulas_changed=False,
        project_frozen_runtime_updated=True,QGIS_changed=False,Delft3D_modified=False)
    save(OUT/'INSTALL.json',receipt);print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
