"""Short constant-BC wiring matrix, not long-term hydraulic validation."""
import sys,json,os,argparse
from pathlib import Path
from stage_bundle207 import ROOT,OUT,TARGET,save,sha
p=argparse.ArgumentParser();p.add_argument('--backend',choices=['cpu','gpu'],required=True);args=p.parse_args()
sys.path[:0]=[str(TARGET),str(ROOT),str(ROOT/'tests')]
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
verify();install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET)
from test_jiang_cao207 import project
from braids_core.river_open_boundary import new_state,resolve
from braids_core.river_project import validate_project
from braids_desktop.preparation import prepare
from braids_core.river_worker import run
from braids_desktop.results import ResultsStore
from osgeo import gdal
import numpy as np
rows=[]
for im in ('total_discharge','stage_discharge','critical_discharge','normal_discharge'):
    for om in ('stage','critical','free_weir'):
        root=OUT/'boundaries_v3'/args.backend/(im+'_'+om)
        root.mkdir(parents=True,exist_ok=False);d=project(root/'input',args.backend,opened=True)
        raster=gdal.Open(d['terrain']['dem_path'],gdal.GA_Update)
        x=-1+(np.arange(48)+.5)*.1;raster.GetRasterBand(1).WriteArray(np.broadcast_to(.01*(2-x),(32,48)).copy());raster=None
        inlet=dict(method='inlet.'+im,discharge_m3_s=.024)
        if im=='stage_discharge':inlet['stage_m']=.15
        if im=='normal_discharge':inlet['energy_slope']='mesh'
        outlet=dict(method='outlet.'+om)
        if om=='stage':outlet['stage_m']=.15
        if om=='free_weir':outlet.update(crest_m=.08,coefficient_m_half_s=1.7,effective_width_m=.8)
        d['settings']['hydraulics']=dict(inlet=inlet,outlet=outlet)
        d['boundary_states']={name:new_state(h['method'],sediment=True) for name,h in d['settings']['hydraulics'].items()}
        d['settings'].update(end_time_s=.004,output_interval_s=.002,checkpoint_interval_s=.004)
        d['boundary_resolution']=resolve(d['settings']['hydraulics'],d['boundary_states'],dict(inlet='inlet',outlet='outlet'),sediment=d['model']['physics']['sediment'],ng=1,end_time=.004)
        d=validate_project(d);save(root/'project.json',d);prep=prepare(root/'project.json',root/'prepared',asset_cache=OUT/'bc_assets')
        status=run(prep,root/'run');store=ResultsStore(root/'run')
        water=store.boundary_flow('inlet',magnitude=False).values
        solid=store.boundary_flow('inlet',quantity='bedload',magnitude=False).values/2650.
        np.testing.assert_allclose(water+solid,.024,rtol=2e-10,atol=1e-13)
        np.testing.assert_allclose(solid/water,.001,rtol=2e-10,atol=1e-13)
        outsolid=store.boundary_flow('outlet',quantity='bedload',magnitude=False).values
        assert np.max(outsolid)<=1e-12
        row=dict(inlet=im,outlet=om,status=status,water=water.tolist(),solid=solid.tolist(),outlet_solid=outsolid.tolist())
        rows.append(row);save(OUT/'boundaries_v3'/args.backend/'CHECK.json',dict(runtime=sha(TARGET/'RUNTIME.json'),rows=rows))
        print(args.backend,im,om,'passed',flush=True)
