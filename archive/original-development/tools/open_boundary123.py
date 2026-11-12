"""Explicit release inputs and evidence for inlet/outlet-only alignment/pinning."""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import argparse,hashlib,json,shutil,sys,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'validation/_open_boundary123'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/open_boundary123'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2),encoding='utf-8')

def inputs():
    from braids_core.river_project import SCHEMA,validate_project
    from braids_core.boundaries.axis_aligned import derive_axis_aligned
    from braids_core.boundaries.domain import PolygonDomain,BoundaryAssignment
    from braids_core.boundaries.inlet_io import NativeGrid
    out=BASE/'projects';out.mkdir(exist_ok=False);changes=[]
    original=ROOT/'validation/_axis_boundary120/ORIGINAL.json'
    for source in sorted((ROOT/'validation/_accuracy122/projects').glob('*.json')):
        if not source.name.startswith(('check_','River1m_')):continue
        doc=json.loads(source.read_text());doc['schema']=SCHEMA;s=doc['settings']
        if source.name.startswith('River1m_'):
            old=json.loads(original.read_text())
            for key in ('domain','assignments','grid'):s[key]=deepcopy(old['settings'][key])
        g=NativeGrid(**s['grid']);r=2**(s['levels']-1)
        d,a,audit=derive_axis_aligned(PolygonDomain(**s['domain']),[BoundaryAssignment(**v) for v in s['assignments']],
            NativeGrid(g.nx*r,g.ny*r,g.xlower_m,g.ylower_m,g.dx_m/r,g.dy_m/r))
        s['domain']=dict(domain_id=d.domain_id,vertices_xy=d.vertices_xy,spatial_reference=d.crs_wkt)
        s['assignments']=[asdict(v) for v in a]
        doc['name']='Inlets/outlets aligned; polygonal walls'
        validate_project(doc);write(out/source.name,doc);write(out/(source.stem+'.geometry-review.json'),audit)
        changes.append(dict(source=str(source),source_sha256=sha(source),target=str(out/source.name),
            geometry_origin=str(original) if source.name.startswith('River1m_') else str(source),
            schema=SCHEMA,wall_refinement='indicators and grading only; explicit survey corridor retained',
            physics_and_controls_changed=False))
    write(BASE/'INPUT_CHANGES.json',changes);print(len(changes),'explicit new projects',flush=True)

def inspect():
    from braids_core import river_worker
    from braids_core.sediment.adaptive_river_worker import load_prepared,hierarchy,initial_leaves
    prepared=BASE/'real_prepared_final'
    if not prepared.exists():river_worker.prepare(BASE/'projects/River1m_cpu.json',prepared)
    river_worker.load_prepared(prepared)
    p,a,t=load_prepared(prepared/'core/prepared');leaves=initial_leaves(p,a,t);m=t.select(leaves)
    wall={f.owner for f in m.faces if f.neighbour<0 and f.kind=='wall'}
    opened={f.owner for f in m.faces if f.neighbour<0 and f.kind in ('inlet','outlet')}
    assert all(m.cells[i].level==t.finest_level for i in opened)
    assert any(m.cells[i].level<t.finest_level for i in wall)
    assert any(m.cells[i].area_m2<.99*m.cells[i].nominal_area_m2 for i in wall)
    metrics=dict(cells=len(m.cells),finest_cells=len(t.finest.cells),
        levels={str(k):sum(c.level==k for c in m.cells) for k in range(t.levels)},
        wall_cells=len(wall),wall_cells_below_finest=sum(m.cells[i].level<t.finest_level for i in wall),
        cropped_wall_cells=sum(m.cells[i].area_m2<.99*m.cells[i].nominal_area_m2 for i in wall),
        open_cells=len(opened),open_cells_all_finest=True,mesh=m.metrics(),runtime_id=river_worker.source_identity())
    write(BASE/'REAL_MESH.json',metrics);print(json.dumps(metrics),flush=True)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    REPORT.mkdir(parents=True,exist_ok=True)
    original=json.loads((ROOT/'validation/_axis_boundary120/ORIGINAL.json').read_text())
    doc=json.loads((BASE/'projects/River1m_cpu.json').read_text());audit=json.loads((BASE/'projects/River1m_cpu.geometry-review.json').read_text())
    offset=np.array([p['grid']['xlower_m'],p['grid']['ylower_m']])
    fig,axes=plt.subplots(1,3,figsize=(14,8),layout='constrained')
    old=np.array(original['settings']['domain']['vertices_xy'])-offset;new=np.array(doc['settings']['domain']['vertices_xy'])-offset
    axes[0].plot(*old.T,'--',color='0.6',label='Original polygon');axes[0].plot(*new.T,color='black',lw=1,label='New polygon')
    for name,row in audit['sections'].items():
        xy=np.array(row['aligned_coordinates'])-offset;axes[0].plot(*xy.T,lw=3,label=row['kind']+' '+name)
    axes[0].legend(fontsize=8);axes[0].set_title('Only open sections aligned')
    for ax in axes[1:]:
        pc=PolyCollection([np.array(c.vertices_xy)-offset for c in m.cells],array=np.array([c.level for c in m.cells]),
            cmap='viridis',clim=(0,t.finest_level),edgecolors='0.25',linewidths=.15)
        ax.add_collection(pc);ax.autoscale_view()
    axes[1].set_title(f'Static terrain mesh: {len(m.cells):,} cells')
    # A coarse cropped wall, away from inlet/outlet endpoints.
    endpoints=np.array([v for s in audit['sections'].values() for v in (s['aligned_coordinates'][0],s['aligned_coordinates'][-1])])-offset
    candidates=np.array([m.cells[i].centroid_xy for i in sorted(wall)
        if m.cells[i].level==0 and m.cells[i].area_m2<.99*m.cells[i].nominal_area_m2])-offset
    distance=np.min(np.linalg.norm(candidates[:,None]-endpoints[None],axis=2),axis=1)
    centre=candidates[np.argmax(distance)];axes[2].set_xlim(centre[0]-20,centre[0]+20);axes[2].set_ylim(centre[1]-20,centre[1]+20)
    axes[2].set_title('Wall detail: cropped, not forced finest')
    for ax in axes:ax.set_aspect('equal');ax.set_xlabel('Local easting (m)');ax.set_ylabel('Local northing (m)')
    fig.colorbar(pc,ax=axes[1:],ticks=list(range(t.levels)),label='Refinement level (0 = 4 m, 2 = 1 m)',shrink=.5)
    fig.savefig(REPORT/'MESH.png',dpi=170);plt.close(fig)

def smoke():
    from braids_core import river_worker
    from braids_core.sediment.adaptive_river_worker import read_live_frame
    result={};states={}
    for backend in ('cpu','gpu'):
        prepared=BASE/('real_prepared_final' if backend=='cpu' else 'real_prepared_gpu')
        if not prepared.exists():river_worker.prepare(BASE/f'projects/River1m_{backend}.json',prepared)
        out=BASE/('real_run_'+backend)
        if out.exists():
            assert json.loads((out/'RUN_CONTEXT.json').read_text())['runtime_id']==river_worker.source_identity()
            status=json.loads((out/'STATUS.json').read_text())
        else:status=river_worker.run(prepared,out,until=.1)
        assert status['status']=='segment_complete' and status['time_s']==.1
        _,geometry,state=read_live_frame(out);states[backend]=state
        diagnostics=json.loads((out/'DIAGNOSTICS.json').read_text())
        assert all(abs(r['water_residual_m3'])<=r['water_roundoff_limit_m3'] for r in diagnostics)
        assert all(not any(r['wall_integrals']) for r in diagnostics)
        assert np.all(np.isfinite(state['q'])) and state['q'][0].min()>=0
        result[backend]=dict(status=status,maximum_water_residual_m3=max(abs(r['water_residual_m3']) for r in diagnostics),
            minimum_depth_m=float(state['q'][0].min()),maximum_depth_m=float(state['q'][0].max()),cells=len(geometry['area']))
        print(backend,'real 1 m smoke completed',flush=True)
    result['maximum_depth_backend_difference_m']=float(np.max(abs(states['cpu']['q'][0]-states['gpu']['q'][0])))
    write(BASE/'REAL_SMOKE.json',result)

def bindings():
    record=BASE/'release_record_final';record.mkdir(exist_ok=False)
    for name in ('RECORD.json','BINDINGS.json'):shutil.copy2(ROOT/'validation/_accuracy122/release_record'/name,record/name)
    paths=['braids_core/boundaries/axis_aligned.py','braids_core/sediment/river_leaf_controller.py','braids_core/river_project.py']
    with zipfile.ZipFile(BASE/'BEFORE.zip') as z:
        updates=[dict(path=p,before=hashlib.sha256(z.read(p)).hexdigest(),after=sha(ROOT/p)) for p in paths]
    updates.append(dict(path='validation/sediment/river_morphodynamics/automatic_leaf.py',
        before=sha(BASE/'automatic_leaf_before.py'),after=sha(ROOT/'validation/sediment/river_morphodynamics/automatic_leaf.py')))
    write(BASE/'PYTHON_UPDATES.json',dict(updates=updates))

def deliver():
    from braids_core.river_project import load_project,SCHEMA
    original=Path('C:/Users/STREEM/Documents/BRAIDS/1m.json');expected=sha(ROOT/'validation/_axis_boundary120/ORIGINAL.json')
    assert sha(original)==expected
    assert json.loads((BASE/'INSTALL.json').read_text())['status']=='installed'
    assert all(json.loads((BASE/'qgis_installed_final/CHECKS.json').read_text())['checks'].values())
    target=original.parent/'alpha123';target.mkdir(exist_ok=False);files=[]
    for backend in ('cpu','gpu'):
        source=BASE/f'projects/River1m_{backend}.json';load_project(source)
        dest=target/f'1m_open_aligned_{backend}.json';shutil.copy2(source,dest);files.append(dict(path=str(dest),sha256=sha(dest)))
    shutil.copy2(BASE/'projects/River1m_cpu.geometry-review.json',target/'GEOMETRY_CHANGES.json')
    write(target/'READ_ME.json',dict(version='0.1.0-alpha.123',schema=SCHEMA,sediment_enabled=False,
        scope='New project from original polygon; inlet/outlet alignment only, no wall rasterization',
        mesh='Static terrain-only 4/2/1 m; only inlet/outlet and explicit survey corridors forced finest',
        original_project=str(original),original_sha256=expected,files=files,
        instructions=['Load either backend project and prepare again. Original and alpha122 projects are untouched.',
            'The 60 s run duration is retained; this release has short smoke tests, not a new full-run accuracy claim.',
            'Wall-only vertices are unchanged. Wall ends shared with inlet/outlet sections reconnect to the moved endpoints.',
            'Cropped wall cells remain possible, including next to open-section corners.'],report=str(REPORT/'REPORT.md')))
    write(BASE/'DELIVERY.json',dict(directory=str(target),files=files,original_sha256=sha(original)))

def seal():
    import xml.etree.ElementTree as ET
    from tools.build_unified_qgis import verify
    installed=json.loads((BASE/'INSTALL.json').read_text());manifest=verify(Path(installed['destination']))
    assert manifest['version']=='0.1.0-alpha.123'
    assert sha(Path(installed['archive']))==installed['archive_sha256']
    tests={}
    for name in ('TESTS.xml','PARTIAL_TESTS.xml','AMR_TESTS_FINAL.xml'):
        suite=ET.parse(BASE/name).getroot().find('testsuite')
        assert suite.get('failures')=='0' and suite.get('errors')=='0';tests[name]=int(suite.get('tests'))
    checks={}
    for name in ('portable_staged_final','qgis_staged_pass','qgis_installed_final'):
        values=json.loads((BASE/name/'CHECKS.json').read_text())['checks'];assert all(values.values());checks[name]=len(values)
        if name.startswith('qgis'):assert json.loads((BASE/name/'LAUNCHER.json').read_text())['returncode']==0
    paths=['README.md','validation/RIVER_MORPHODYNAMICS_STATUS.md',
        'validation/sediment/river_morphodynamics/UNIFIED_QGIS_PLAN.md',
        'braids_core/boundaries/axis_aligned.py','braids_core/sediment/river_leaf_controller.py','braids_core/river_project.py',
        'validation/sediment/river_morphodynamics/automatic_leaf.py',
        'braids_qgis/gui/river_dock.py','braids_qgis/metadata.txt','braids_qgis/FIRST_RELEASE.md','braids_qgis/MODEL_SETUP.md',
        'tests/test_axis_aligned_boundaries.py','tests/test_open_boundary_refinement123.py','tests/test_river_project.py',
        'tests/test_river_automatic_leaf.py','tests/support/qgis_open_boundary123.py',
        'tools/open_boundary123.py','tools/run_qgis_open_boundary123.py','tools/build_unified_qgis.py','tools/install_unified_qgis.py']
    with zipfile.ZipFile(REPORT/'SOURCES.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in paths:z.write(ROOT/path,path)
    evidence=['BEFORE.zip','automatic_leaf_before.py','INPUT_CHANGES.json','PYTHON_UPDATES.json',
        'REAL_MESH.json','REAL_SMOKE.json','release_record_final/RECORD.json','release_record_final/BINDINGS.json',
        'release_record_final/PACKAGE.json','INSTALL.json','DELIVERY.json',*tests.keys()]
    for name in checks:
        evidence.append(name+'/CHECKS.json')
        if name.startswith('qgis'):evidence.append(name+'/LAUNCHER.json')
    for mode in ('STATIC','DYNAMIC'):
        shutil.copy2(BASE/'qgis_installed_final'/f'MESH_{mode}.png',REPORT/f'QGIS_{mode}.png')
    write(REPORT/'MANIFEST.json',dict(schema='braids.open-boundary123-delivery.v1',
        tests=tests,checks=checks,release=installed,source_files={p:sha(ROOT/p) for p in paths},
        evidence={str(BASE/p):sha(BASE/p) for p in evidence},
        report_files={p.name:sha(p) for p in REPORT.iterdir() if p.is_file() and p.name!='MANIFEST.json'}))
    print('Verified and sealed',installed['version'],tests,checks,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('inputs','inspect','bindings','deliver','smoke','seal'));a=p.parse_args()
    globals()[a.action]()
