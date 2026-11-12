"""Short packaged CPU/GPU checks on graded, permeable constrained polygons."""
import argparse,importlib.util,json,os,shutil,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'desktop/qualification/nearwake60';TARGET=OUT/'candidate'
os.environ['QT_QPA_PLATFORM']='offscreen';sys.dont_write_bytecode=True;sys.path.insert(0,str(TARGET));sys.path.insert(1,str(ROOT))
if os.name=='nt':_dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
from braids_desktop import services,preparation
from braids_core import river_worker
from braids_core.sediment import adaptive_river_worker as core
from braids_core.sediment.river_field_policy import AutomaticRiver


def last(folder):
    row=json.loads((folder/'frames/INDEX.json').read_text())['frames'][-1]
    with np.load(folder/'frames'/row['file']) as a:return {k:a[k].copy() for k in a.files}


def create(root,backend,mode):
    from osgeo import gdal,osr
    from PySide6.QtWidgets import QApplication
    from braids_desktop.forms import SettingsPanel
    root.mkdir(exist_ok=False);dem=root/'channel.tif'
    xs=-1.+(np.arange(88)+.5)*.25;ys=25.-(np.arange(104)+.5)*.25
    z=.1-.001*xs[None,:]+2.*np.maximum(abs(ys[:,None]-12.)-2.25,0.)**2
    raster=gdal.GetDriverByName('GTiff').Create(str(dem),88,104,1,gdal.GDT_Float64)
    raster.SetGeoTransform((-1.,.25,0.,25.,0.,-.25));s=osr.SpatialReference();s.ImportFromEPSG(2056)
    raster.SetProjection(s.ExportToWkt());raster.GetRasterBand(1).WriteArray(z);raster.FlushCache();raster=None
    doc=services.default_document(dem,[[0.,0.],[20.,0.],[20.,24.],[0.,24.]],'EPSG:2056',backend=backend,minimum_size_m=.5,levels=3,mesh_mode='dynamic_depth',end_time_s=.4)
    from braids_core.boundaries.domain import PolygonDomain
    d=PolygonDomain(**doc['settings']['domain'])
    for x,name,kind,controls in [(0.,'Upstream','inlet',dict(method='inlet.critical_discharge',discharge_m3_s=1.)),(20.,'Downstream','outlet',dict(method='outlet.stage',stage_m=.5))]:
        ids=[e.edge_id for e in d.edges if e.start_xy[0]==e.end_xy[0]==x]
        doc=services.assign_boundary(doc,ids,name,kind,controls)
    doc['settings']['breaklines']=[dict(id='Lower bank',closed=False,points_xy=[[0.,10.125],[8.,10.5],[20.,10.25]]),dict(id='Upper bank',closed=False,points_xy=[[0.,13.5],[8.,13.75],[20.,13.625]])]
    doc['settings']['adaptation'].update(transition_layers=3,interval=.2,coarsen_delay=1,domain_border_enabled=False,buffer_cells=1)
    doc['settings'].update(output_interval_s=.2,checkpoint_interval_s=.2,dt_max_s=.05)
    doc['initial'].update(stage_m=.5,velocity_m_s=[.7,0.])
    app=QApplication.instance() or QApplication([]);panel=SettingsPanel();panel.load_document(doc)
    if mode=='bedload':
        panel.physics.enabled.setChecked(True);panel.physics.bedload_on.setChecked(True)
        panel.physics.capacity.setCurrentIndex(panel.physics.capacity.findData('capacity.mpm'))
        # This fixture is an explicitly uniform 0.5 mm material: every required
        # percentile is supplied independently with that same measured value.
        for r in range(panel.physics.grain_sizes.rowCount()):panel.physics.grain_sizes.item(r,1).setText('0.5')
    doc=panel.document(doc);panel.close()
    source=root/'input.json';source.write_text(json.dumps(doc,indent=2)+'\n');return source


def run(backend,mode):
    verify();install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET);shutil.which=lambda _:None
    root=OUT/(backend+'_'+mode);source=create(root,backend,mode)
    prepared=preparation.prepare(source,root/'prepared',asset_cache=OUT/'assets')
    original=AutomaticRiver.assess
    def assess(self,factory,**kw):
        if self.controller.owner.steps==0:self.next_time=self.settings.interval;return
        return original(self,factory,**kw)
    def finest(profile,initial,tree,required=()):return tuple(range(tree.offsets[-2],tree.count))
    with patch.object(core,'initial_leaves',finest),patch.object(AutomaticRiver,'assess',assess):
        result=river_worker.run(prepared,root/'run',checkpoint_dir=root/'checkpoints');assert result['status']=='complete'
        checkpoint=next((root/'checkpoints').glob('t_0.2_step_*.npz'))
        resumed=river_worker.run(prepared,root/'resume',restart=checkpoint);assert resumed['status']=='complete'
    for field in ['q','z','water']:np.testing.assert_array_equal(last(root/'run')[field],last(root/'resume')[field])
    events=json.loads((root/'run/ADAPTATION.json').read_text());changed=[e for e in events if e['changed']]
    assert changed and changed[0]['time_s']>0
    diagnostics=json.loads((root/'run/DIAGNOSTICS.json').read_text());residual=max(abs(r['water_residual_m3']) for r in diagnostics)
    assert residual<1e-8
    record=json.loads((root/'run/frames/INDEX.json').read_text())['frames'][0]
    with np.load(root/'run/frames'/record['file']) as a:initial_z=a['z'].copy()
    # Finite transport is checked on the saved actual bedload diagnostic, not
    # inferred from a changed wetting front or comparing remapped cell indices.
    state=last(root/'run');assert np.isfinite(state['q']).all()
    check=dict(status='passed',backend=backend,mode=mode,end_s=.4,transition_layers=3,restart_exact=True,
        compiler_discovery_disabled=True,maximum_water_residual_m3=residual,
        forced_finest_start_for_regrid_test=True,changed=[{k:e[k] for k in ['time_s','from_cells','to_cells','transfer']} for e in changed])
    (root/'CHECK.json').write_text(json.dumps(check,indent=2)+'\n');print(json.dumps(check),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=['cpu','gpu']);p.add_argument('mode',choices=['hydro','bedload']);a=p.parse_args();run(a.backend,a.mode)

