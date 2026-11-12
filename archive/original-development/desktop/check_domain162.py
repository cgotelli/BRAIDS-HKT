"""Matched straight/extended groin controls using unmodified solver runtimes."""
import argparse
import shutil
import sys
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'desktop'))
import check_bedload161 as old
import numpy as np

BASE=ROOT/'validation/_hydraulics162/domain_v2'


def delft(kind, end):
    import runtime
    parent=ROOT/'validation/_bedload161_matched/delft_mpm_v8'
    out=BASE/kind/'delft'
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((parent/'INPUTS.json').read_text())
    for name, checksum in manifest['inputs'].items():
        assert old.sha(parent/name)==checksum
        shutil.copy2(parent/name,out/name)
    mdf=(out/'sim_P057.mdf').read_text()
    for key, value in dict(Tstop=str(end),Flmap=f'0 5 {end:g}',Flhis=f'0 5 {end:g}').items():
        mdf=old.replace(mdf,key,value)
    if kind=='straight':mdf=old.replace(mdf,'FilDry','# #')
    if kind=='cyclic':mdf=old.replace(mdf,'Momsol','#Cyclic#')
    if kind=='long':
        mdf=old.replace(mdf,'MNKmax','82 18 1')
        xx,yy=np.meshgrid(np.arange(81)*.5,np.arange(17)*.5)
        grid='Coordinate System = Cartesian\n81 17\n0 0 0\n'
        for array in (xx,yy):
            for j,row in enumerate(array):
                for start in range(0,len(row),5):
                    grid+=(f' ETA= {j+1:4d} ' if start==0 else '           ')+' '.join(f'{v:.17e}' for v in row[start:start+5])+'\n'
        (out/'flume.grd').write_text(grid)
        (out/'flume.enc').write_text('1 1\n82 1\n82 18\n1 18\n1 1\n')
        # DP-mode depths are at water-level centres, positive down.
        depths=np.broadcast_to(-.1+.001*(np.arange(82)-.5)*.5,(18,82))
        np.savetxt(out/'bed.dep',depths,fmt='%.17g')
        # Native initial fields have N rows / M columns; all velocities zero.
        with (out/'initial.ini').open('w') as f:
            for value in (.6,0.,0.):np.savetxt(f,np.full((18,82),value),fmt='%.17g')
        bnd=(out/'flume.bnd').read_text().replace('    42','    82')
        (out/'flume.bnd').write_text(bnd)
        (out/'sections.crs').write_text((out/'sections.crs').read_text().replace('41','81'))
    (out/'sim_P057.mdf').write_text(mdf)
    # No morphology in these hydraulic controls, even at t=40 exactly.
    mor=(out/'mor.mor').read_text()
    mor=old.replace(mor,'MorStt',str(end+1))
    (out/'mor.mor').write_text(mor)
    old.save(out/'INPUTS.json',dict(parent=str(parent),parent_manifest_sha256=old.sha(parent/'INPUTS.json'),
        changes=dict(control=kind,end_s=end,morphology='disabled by MorStt > end',
                     long_outlet_stage='unchanged 0.58 m; identical BRAIDS/FLOW, not an original-domain equivalence'),
        Delft3D_source_modified=False,Delft3D_runtime_modified=False,
        inputs={p.name:old.sha(p) for p in out.iterdir() if p.is_file()}))
    print(runtime.run_case(out,configuration='config_flow2d3d.xml',timeout_s=180),flush=True)


def braids(kind,backend,end):
    from PySide6.QtWidgets import QApplication
    from check_boundaries159 import document,services
    from braids_core.river_open_boundary import resolve
    from braids_core.boundaries.domain import PolygonDomain
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    from braids_desktop.preparation import prepare
    from braids_core.river_worker import run
    from braids_desktop.forms import SettingsPanel
    verify();install(json.loads((old.RUNTIME/'PRECOMPILED.json').read_text()),old.RUNTIME)
    app=QApplication.instance() or QApplication([])
    out=BASE/kind/('braids_'+backend)
    out.mkdir(parents=True,exist_ok=False)
    doc=document(out/'project',backend,'weir_recovered' if kind=='long' else 'stage')
    if kind=='straight':
        doc=services.replace_domain(doc,[(0,0),(20,0),(20,8),(0,8)],'EPSG:2056',for_editing=True)
    length=40 if kind=='long' else 20
    for x,name,role,control in ((0,'Upstream','inlet',dict(method='inlet.stage_discharge',discharge_m3_s=4.,stage_m=.6)),
                               (length,'Downstream','outlet',dict(method='outlet.stage',stage_m=.58))):
        edges=PolygonDomain(**doc['settings']['domain']).edges
        ids=[e.edge_id for e in edges if e.start_xy[0]==e.end_xy[0]==x]
        doc=services.assign_boundary(doc,ids,name,role,control,for_editing=True)
    f=SettingsPanel();f.load_document(doc);f.end_time.setValue(end);f.output_interval.setValue(5.)
    doc=f.document();f.close()
    doc['initial'].update(stage_m=.6,velocity_m_s=[0.,0.])
    doc['boundary_states']['Downstream']['stage_policy']='stage.subcritical_backflow'
    doc.pop('editor_state',None)
    doc['boundary_resolution']=resolve(doc['settings']['hydraulics'],doc['boundary_states'],
        dict(Upstream='inlet',Downstream='outlet'),sediment=None,ng=1,end_time=end)
    services.save_project_folder(out/'project',doc)
    prepare(out/'project/braids_project.json',out/'prepared',asset_cache=BASE/'assets')
    status=run(out/'prepared',out/'run',until=end)
    old.save(out/'STATUS.json',dict(status=status,script_sha256=old.sha(__file__),
        runtime_manifest_sha256=old.sha(old.RUNTIME/'RUNTIME.json')))
    print(status,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('engine',choices=['braids','delft'])
    p.add_argument('kind',choices=['straight','long','cyclic']);p.add_argument('--backend',choices=['cpu','gpu'],default='cpu')
    p.add_argument('--end',type=float,default=40.)
    a=p.parse_args()
    if a.engine=='delft':delft(a.kind,a.end)
    else:
        assert a.kind!='cyclic'
        braids(a.kind,a.backend,a.end)
