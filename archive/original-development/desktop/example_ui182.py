"""Build a new bundled example; never migrate a user's saved project."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'validation/_ui182/runtime'),str(ROOT)]
from braids_desktop import services,project_file

def main():
    source=ROOT/'desktop/examples/river_demo158'
    destination=ROOT/'desktop/examples/river_demo182'
    if destination.exists():raise FileExistsError(destination)
    doc=json.loads((source/'braids_project.json').read_text(encoding='utf-8'))
    doc.pop('editor_state',None)
    doc['terrain']['dem_path']=str((source/doc['terrain']['dem_path']).resolve())
    doc['settings']['adaptation']['domain_border_enabled']=True
    doc['name']='Natural river demonstration'
    saved=services.save_project_folder(destination,doc)
    project_file.write(project_file.default_file(destination),destination,saved)
    assert services.load_project_folder(destination)['settings']['adaptation']['domain_border_enabled']
    print('New example ready:',destination)
if __name__=='__main__':main()
