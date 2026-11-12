"""Add standalone launch assets and a fresh current demo before qualification."""
import json,shutil,sys
from stage_wall188 import ROOT,OUT,TARGET as SOURCE,sha,save
TARGET=OUT/'desktop_candidate'


def main():
    assert not TARGET.exists()
    m=json.loads((SOURCE/'RUNTIME.json').read_text())
    for n,h in m['files'].items():assert sha(SOURCE/n)==h,n
    shutil.copytree(SOURCE,TARGET,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('desktop/worker.py','BRAIDS_logo.png'):
        p=TARGET/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
        m['files'][name]=sha(p)
    sys.path[:0]=[str(TARGET),str(ROOT)]
    from braids_desktop import services,project_file
    from braids_core.river_project import SCHEMA
    from braids_core.river_open_boundary import resolve
    old=ROOT/'desktop/examples/river_demo182'
    doc=json.loads((old/'braids_project.json').read_text(encoding='utf-8'))
    doc.pop('editor_state',None);doc['schema']=SCHEMA
    doc['terrain']['dem_path']=str((old/doc['terrain']['dem_path']).resolve())
    for state in doc['boundary_states'].values():state['secondary_intensity_m_s']=0.
    roles={n:('inlet' if h['method'].startswith('inlet.') else 'outlet') for n,h in doc['settings']['hydraulics'].items()}
    doc['boundary_resolution']=resolve(doc['settings']['hydraulics'],doc['boundary_states'],roles,
        sediment=doc['model']['physics']['sediment'],ng=1,end_time=doc['settings']['end_time_s'])
    demo=TARGET/'desktop/examples/river_demo182'
    saved=services.save_project_folder(demo,doc)
    project_file.write(project_file.default_file(demo),demo,saved)
    for p in demo.rglob('*'):
        if p.is_file():m['files'][p.relative_to(TARGET).as_posix()]=sha(p)
    save(TARGET/'RUNTIME.json',m)
    save(OUT/'DELIVERY_STAGE.json',dict(baseline=sha(SOURCE/'RUNTIME.json'),candidate=sha(TARGET/'RUNTIME.json'),
        purpose='Self-contained Desktop UI, bootstrap, logo and new schema-v25 demo; solver files unchanged',
        new_user_project_migration=False))
    print('DELIVERY STAGED',sha(TARGET/'RUNTIME.json'))


if __name__=='__main__':main()
