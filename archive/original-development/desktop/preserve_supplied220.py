"""Preserve this milestone's evidence after an external validation-folder loss."""
from pathlib import Path
import shutil,json,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'validation/_supplied220';target=ROOT/'desktop/qualification/supplied220'
target.mkdir(parents=True,exist_ok=True)
for name in ('REGRESSION.xml','STAGE.json','EDITOR.json','CANDIDATE_LAUNCH.json',
             'QUALIFICATION_METRICS.json','FIGURE_METRICS.json','EXPORT.json','EXAMPLE.json','SOURCES.json',
             'focus1.xml','focus2.xml','column3.xml','column4.xml'):
    p=source/name
    if p.is_file():shutil.copy2(p,target/name)
suite=ET.parse(source/'REGRESSION.xml').getroot().find('testsuite')
status=dict(status='not_installed_missing_validation_sources',regression=suite.attrib,
    candidate_runtime=json.loads((source/'STAGE.json').read_text())['runtime'],
    installed_version=json.loads((ROOT/'desktop/PROVENANCE.json').read_text())['version'],
    missing='validation/sediment/river_morphodynamics',
    source_changes_preserved=True,user_projects_modified=False,
    next_step='Recover missing validation source files, then run the interrupted and unexecuted regression cases. Do not relax the delivery gate.')
(target/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
print(json.dumps(status,indent=2))
