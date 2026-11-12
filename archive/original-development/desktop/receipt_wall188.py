"""Seal qualification only from complete public runs and immutable payloads."""
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_wall188';TARGET=OUT/'desktop_candidate'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_wall188'
sys.path[:0]=[str(TARGET),str(ROOT)]
from stage_secondary185 import save,sha
from braids_desktop.results import ResultsStore
from qualify_inlet187 import checkpoint_arrays


def main():
    manifest=json.loads((TARGET/'RUNTIME.json').read_text())
    for n,h in manifest['files'].items():assert sha(TARGET/n)==h,n
    release=json.loads((REPORT/'RELEASE.json').read_text());assert release['runtime']==sha(TARGET/'RUNTIME.json')
    for start,row in release['cold'].items():
        for kind in ('bed','depth'):assert row['maximum_over_frames'][kind]<1.,(start,kind)
        for backend in ('cpu','gpu'):
            assert row['checks'][backend]['status']['time_s']==220.
            assert all(row['restarts'][backend]['frame_exact'].values())
            assert all(row['restarts'][backend]['checkpoint_arrays_exact'].values())
            steps=json.loads((OUT/f'release/{start}_prescribed_{backend}/STEPS.json').read_text())
            assert max(s['secondary_cfl_max'] for s in steps)<=.8*(1+1e-10)
    suite=ET.parse(OUT/'TESTS_FINAL.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==130 and int(suite.attrib['errors'])==int(suite.attrib['failures'])==0
    math=json.loads((OUT/'PACKAGED_UNIT.json').read_text());assert math['max_cpu_gpu']==0 and math['max_constraint_excess']<2e-14
    boundaries=json.loads((OUT/'BOUNDARY_JOBS.json').read_text());assert boundaries['passed']==24
    identity=json.loads((OUT/'IDENTITY.json').read_text());assert all(x['rejected'] and not x['output_created'] for x in identity['checks'].values())
    assert json.loads((OUT/'JOBS.json').read_text())['complete']
    hydraulics={}
    for p in sorted((OUT/'hydraulics').glob('*/CHECK.json')):
        row=json.loads(p.read_text());assert row['runtime']==sha(TARGET/'RUNTIME.json')
        assert row['status']['status']=='complete' and row['wall_water_zero'] and row['minimum_depth']>=0.
        if p.parent.name.startswith('lake'):assert row['max_q_change']<1e-10
        hydraulics[p.parent.name]=row
    assert len(hydraulics)==8
    for tag,row in release['smoke'].items():
        for kind in ('bed','depth'):assert row[kind]['max_mm']<1.
        if tag.endswith('smart_jaggi'):
            for backend in ('cpu','gpu'):assert all(row['checks'][backend]['restart']['exact'].values())
    epochs={};amr_restart={}
    for backend in ('cpu','gpu'):
        root=OUT/f'smoke/amr/smart_jaggi_{backend}'
        s=ResultsStore(root/'full')
        epochs[backend]=len({sha(s.read(i).mesh_path) for i in range(len(s.times))}) if hasattr(s.read(0),'mesh_path') else len({s.read(i).mesh['vertices'].tobytes() for i in range(len(s.times))})
        assert epochs[backend]>1
        left=checkpoint_arrays(root/'full/checkpoint.npz');right=checkpoint_arrays(root/'resumed/checkpoint.npz')
        exact={k:bool(np.array_equal(left[k],right[k])) for k in left};assert all(exact.values()),(backend,exact)
        amr_restart[backend]=exact
    records=[]
    for p in (OUT/'release').rglob('records.npz'):
        with np.load(p) as f:
            count=np.diff(f['offset']);records.append(dict(path=str(p.relative_to(OUT)),maximum_faces=int(count.max()),fallback_cells=int(np.count_nonzero(count>32))))
    on=json.loads((ROOT/'desktop/examples/two_meanders188/advanced_on/braids_project.json').read_text())
    off=json.loads((ROOT/'desktop/examples/two_meanders188/advanced_off/braids_project.json').read_text())
    for key in ('adaptation','adaptive','assignments','bank','breaklines','columns','domain','grid','hydraulics','levels',
                'sediment_boundaries','sediment_start_time_s','solids','suspension'):
        assert on['settings'][key]==off['settings'][key],key
    for key in ('execution','sediment'):assert on['model']['request'][key]==off['model']['request'][key],key
    assert on['initial']==off['initial']
    a=np.load(OUT/'packaged_replay/dt02/FIELDS.npz');b=np.load(OUT/'packaged_replay/dt01/FIELDS.npz');old=np.load(OUT/'replay/walls_cpu/FIELDS.npz')
    method=dict(packaged_matches_isolated=all(np.array_equal(a[k],old[k]) for k in ('final_q','final_z')),
        half_step_bed_mm=float(abs(a['final_z']-b['final_z']).max()*1000),
        half_step_depth_mm=float(abs(a['final_q'][0]-b['final_q'][0]).max()*1000),
        half_step_bed_rms_mm=float(np.sqrt(np.average((a['final_z']-b['final_z'])**2,weights=a['geometry_area']))*1000))
    assert method['packaged_matches_isolated'] and method['half_step_bed_mm']<1. and method['half_step_depth_mm']<1.
    preserved={}
    for name,folder,expected in [('preview29',ROOT/'desktop/runtime','fddbf4c03a89f4d19aa72e89dc3fb02967470289d0621c1c47d7ab9a73f71ae3'),
        ('inlet187',ROOT/'validation/_inlet187/prescribed_candidate','8b27294c161aec91332a74c89b9de99c9c91cd78a94630369bc0557ae8081015')]:
        assert sha(folder/'RUNTIME.json')==expected
        m=json.loads((folder/'RUNTIME.json').read_text())
        for n,h in m['files'].items():assert sha(folder/n)==h,(name,n)
        preserved[name]=expected
    save(REPORT/'QUALIFICATION.json',dict(runtime=sha(TARGET/'RUNTIME.json'),qualified=True,installed=False,
        payloads_verified=len(manifest['files']),tests=130,projection=math,boundary_controls=boundaries,
        hydraulic_controls=hydraulics,amr_mesh_epochs=epochs,amr_checkpoint_arrays_exact=amr_restart,geometry_guard=records,temporal=method,
        checkpoint_guards=identity,preserved=preserved,physical_laws_changed=False,sediment_reconstruction_changed=False,
        hydraulic_method='explicit augmented-fwave SSPRK22',Delft3D_modified=False))
    print('QUALIFIED',sha(TARGET/'RUNTIME.json'),method)


if __name__=='__main__':main()
