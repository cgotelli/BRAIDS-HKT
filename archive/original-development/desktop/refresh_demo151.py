"""Re-publish the bundled demo template only; never migrate user workspaces."""
from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'validation/_solids151/candidate'))
from braids_core import river_folder as folder
from braids_core.river_project import validate_project,identity
from braids_desktop import project_file
root=ROOT/'desktop/examples/river_demo'
project=validate_project(json.loads((root/'braids_project.json').read_text(encoding='utf-8')))
assert project['schema']=='braids.river-project.v22' and project['settings']['solids']==[]
manifest=json.loads((root/folder.MANIFEST_FILE).read_text(encoding='utf-8'))
timestamp=folder._now()
revision=dict(schema=folder.REVISION_SCHEMA,timestamp=timestamp,kind='saved',parent=manifest['revision'],
    project=project,project_id=identity(project),assets=manifest['assets'],session=manifest['session'],geometry_review=manifest['geometry_review'])
name=folder._record_path(Path('history/revisions'),revision).as_posix()
folder._atomic_json(root/name,revision)
folder._atomic_json(root/'braids_project.json',project)
manifest.update(saved_at=timestamp,project_id=identity(project),project_sha256=hashlib.sha256((root/'braids_project.json').read_bytes()).hexdigest(),revision=name)
folder._atomic_json(root/folder.MANIFEST_FILE,manifest)
project_file.write(root/'river_demo.braids.json',root,project)
folder.load_folder(root);print('Bundled v22 template and folder identities verified')
