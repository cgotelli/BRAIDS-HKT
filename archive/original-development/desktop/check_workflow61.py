"""Short real CPU/GPU saved-state, model-change, mapping and restart checks."""
import argparse,json,os,sys,time
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];TARGET=ROOT/'desktop/qualification/workflow61/candidate';OUT=TARGET.parent
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/desktop_workflow61'
sys.path.insert(0,str(TARGET));sys.path.insert(1,str(ROOT));os.environ['QT_QPA_PLATFORM']='offscreen'
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
from braids_core.sediment.portable_river_runtime import install
from braids_core import river_worker,river_state_snapshot as snapshot,river_state_initialization as initialize
from braids_desktop import services,preparation
from braids_desktop.results import ResultsStore


def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(root,backend,mode,dem,*,end=.12,dx=.5,cropped=False,adaptive=False):
    from braids_core.boundaries.domain import PolygonDomain
    from PySide6.QtWidgets import QApplication
    from braids_desktop.forms import SettingsPanel
    shift=np.array([2600000.,1200000.]) if cropped else np.zeros(2)
    vertices=np.array([[0.,0.],[8.,0.],[8.,4.],[0.,3.5 if cropped else 4.]])+shift
    doc=services.default_document(dem,vertices.tolist(),'EPSG:2056',backend=backend,minimum_size_m=dx,levels=2,mesh_mode='static_terrain',end_time_s=end)
    d=PolygonDomain(**doc['settings']['domain'])
    for x,name,kind,bc in [(shift[0],'Upstream','inlet',dict(method='inlet.critical_discharge',discharge_m3_s=.8)),(shift[0]+8.,'Downstream','outlet',dict(method='outlet.stage',stage_m=1.))]:
        ids=[e.edge_id for e in d.edges if e.start_xy[0]==e.end_xy[0]==x]
        doc=services.assign_boundary(doc,ids,name,kind,bc)
    if cropped:doc['settings']['solids']=[dict(id='solid',points_xy=(np.array([[3.,1.],[4.,1.25],[3.2,2.]])+shift).tolist())]
    doc['initial'].update(stage_m=1.,velocity_m_s=[.2,0.])
    doc['settings']['adaptation'].update(terrain_enabled=False,domain_border_enabled=False,transition_layers=2,interval=.06)
    doc['settings'].update(dt_max_s=.03,output_interval_s=.06,checkpoint_interval_s=.06)
    doc['settings']['adaptive']=adaptive
    if mode=='jiang':
        from braids_core.river_model import new_model,new_bedload_selection,resolve_model
        from braids_core.river_jiang_cao import METHOD,ROUGHNESS,DEFAULTS,COLLAPSE
        from braids_core.sediment.bed_columns_v101 import ColumnMethodsV101
        from braids_core.sediment.river_column_methods import method_ids
        from braids_core.river_boundary_draft import new_state
        m=new_model(backend=backend);m['hydrodynamics'].update(secondary_flow='secondary.prognostic',roughness=ROUGHNESS)
        s=new_bedload_selection();s['capacity']=dict(method=METHOD,parameters=dict(DEFAULTS),grain_sizes_m=dict(d50=.001,d90=.001))
        s['grains'][0]['diameter_m']=.001;s['avalanche']=COLLAPSE;m['sediment']=dict(enabled=True,selection=s)
        doc['model']=resolve_model(m).document()
        columns=replace(ColumnMethodsV101(**doc['settings']['columns']),fraction_ids=('sand',),diameter_m=(.001,),bed='mixed',base_update=1)
        doc['settings']['columns']=asdict(columns);doc['column_methods']=method_ids(columns)
        doc['settings']['controls'].update(diameter=.001,morfac=1.,betac=0.,diffusivity=0.,viscosity=0.)
        doc['initial'].update(layer_thickness_m=[.2],layer_fractions=[[1.]])
        for name in ('Upstream','Downstream'):doc['boundary_states'][name]=new_state(doc['settings']['hydraulics'][name]['method'],sediment=True)
        doc['boundary_states']['Upstream']['composition']=dict(method='composition.reference_volume_fraction',fractions=[1.])
        doc['settings']['sediment_boundaries']['Upstream']=dict(kind='inlet_solid_water_ratio',solid_water_ratio=[0.],concentration_kg_m3=[0.])
        doc=services._resolve_boundary_state(doc)
    app=QApplication.instance() or QApplication([]);panel=SettingsPanel();panel.load_document(doc)
    if mode=='bedload':
        panel.physics.enabled.setChecked(True);panel.physics.bedload_on.setChecked(True)
        panel.physics.capacity.setCurrentIndex(panel.physics.capacity.findData('capacity.mpm'))
        for r in range(panel.physics.grain_sizes.rowCount()):panel.physics.grain_sizes.item(r,1).setText('1')
    doc=panel.document(doc);panel.close()
    doc['settings'].update(checkpoint_interval_s=.06,output_interval_s=.06)
    doc['editor_state']['checkpoint_clock']=dict(mode='custom',custom_interval_s=.06)
    doc=services.validate_document(doc);root.mkdir(exist_ok=False);source=root/'input.json';save(source,doc)
    return preparation.prepare(source,root/'prepared',asset_cache=OUT/'assets')


def last(output):
    store=ResultsStore(output);return store.read(len(store.times)-1)


def check(backend,scenario):
    from osgeo import gdal,osr
    install(read(TARGET/'PRECOMPILED.json'),TARGET)
    root=OUT/f'{backend}_{scenario}_{time.time_ns()}';root.mkdir();dem=root/'dem.tif';cropped=scenario=='mapped'
    shift=np.array([2600000.,1200000.]) if cropped else np.zeros(2)
    x=-1.+(np.arange(80)+.5)*.125;y=5.-(np.arange(48)+.5)*.125
    z=.004*np.sin(x[None,:])*np.cos(y[:,None]);ds=gdal.GetDriverByName('GTiff').Create(str(dem),80,48,1,gdal.GDT_Float64)
    ds.SetGeoTransform((shift[0]-1.,.125,0.,shift[1]+5.,0.,-.125));crs=osr.SpatialReference();crs.ImportFromEPSG(2056)
    ds.SetProjection(crs.ExportToWkt());ds.GetRasterBand(1).WriteArray(z);ds=None
    jiang=scenario.startswith('jiang');dynamic=scenario in ('jiang_dynamic','ordinary_dynamic')
    source_backend=('gpu' if backend=='cpu' else 'cpu') if not jiang else backend
    mode='jiang' if jiang else 'hydro'
    source=create(root/'source',source_backend,mode,dem,cropped=cropped,adaptive=dynamic)
    result=river_worker.run(source,root/'source_run');assert result['status']=='complete'
    old=last(root/'source_run');raw=snapshot.source_from_checkpoint(root/'source_run/checkpoint.npz')
    state=root/'source_state.npz';state.write_bytes(raw)
    target=create(root/'target',backend,'jiang' if jiang else 'bedload',dem,end=.24,dx=.4 if cropped else .5,cropped=cropped,adaptive=dynamic)
    initialized=root/'initialized.npz';decision=initialize.prepare(target,state,initialized)
    if cropped:
        assert decision['status']=='mapping_required' and not initialized.exists()
        decision=initialize.prepare(target,state,initialized,allow_mapping=True)
    else:assert decision['status']=='ready' and not decision['mesh_changed']
    result=river_worker.run(target,root/'run',initial_state=initialized,checkpoint_dir=root/'checkpoints')
    assert result['status']=='complete';store=ResultsStore(root/'run');first=store.read(0)
    assert first.time_s==.12 and store.times[-1]==.24
    if not cropped:
        np.testing.assert_array_equal(first.q,old.q);np.testing.assert_allclose(first.z,old.z,rtol=0,atol=1e-12)
    checkpoint=next((root/'checkpoints').glob('t_0.18_step_*.npz'))
    state.rename(root/'source_state.saved');initialized.rename(root/'initialized.saved')
    result=river_worker.run(target,root/'resume',restart=checkpoint);assert result['status']=='complete'
    a,b=last(root/'run'),last(root/'resume')
    exact={k:bool(np.array_equal(a.arrays[k],b.arrays[k])) for k in ('q','z','mass')}
    assert all(exact.values()),exact
    # New checkpoints retain the imported baseline: no dependency on the
    # original source or mapped-state file is needed for exact continuation.
    if dynamic:
        status=read(root/'run/ADAPTIVE.json') if (root/'run/ADAPTIVE.json').is_file() else None
        assert status is None or len(status)>0
    check=dict(status='passed',backend=backend,source_backend=source_backend,scenario=scenario,root=str(root),
        runtime_sha256=snapshot.sha_bytes((TARGET/'RUNTIME.json').read_bytes()),
        saved_time_s=.12,end_time_s=.24,decision=decision,exact_restart_fields=exact,
        source_cells=len(old.polygons),target_cells=len(first.polygons),native_compiler_required=False)
    save(root/'CHECK.json',check);save(REPORT/f'CHECK_{backend}_{scenario}.json',check);print(json.dumps(check),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=['cpu','gpu']);p.add_argument('scenario',choices=['same','mapped','jiang','jiang_dynamic','ordinary_dynamic']);a=p.parse_args();check(a.backend,a.scenario)
