"""Matched synthetic two-bend Smart-Jaeggi cases using the installed Desktop runtime.

No solver patch, calibration to an expected bar, or edited initial bar is used.
Create once; each run segment goes to a new folder and is authenticated normally.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'desktop/runtime'
OUT = ROOT / 'desktop/examples/two_meanders184'
REPORT = ROOT / 'validation/sediment/river_morphodynamics/reports/desktop_meanders184'
sys.path[:0] = [str(RUNTIME), str(ROOT), str(ROOT / 'desktop')]
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def geometry():
    """Two opposite 60-degree circular bends, R/B=5, axis-aligned open ends."""
    r, a, lead, gap = 20., np.pi/3., 10., 4./np.sqrt(3.)
    arc = r*a
    xend = 2*r*np.sin(a)+gap*np.cos(a)+lead
    tail = 56.-xend
    breaks = np.array([0., lead, lead+arc, lead+arc+gap, lead+2*arc+gap, lead+2*arc+gap+tail])
    s = np.unique(np.r_[np.linspace(0, breaks[-1], int(breaks[-1]/.1)+1), breaks])
    xy = np.zeros((len(s), 2)); angle = np.zeros(len(s)); curvature = np.zeros(len(s))
    for i, v in enumerate(s):
        if v <= breaks[1]: xy[i] = v, 0
        elif v <= breaks[2]:
            theta = (v-lead)/r; angle[i] = theta; curvature[i] = 1/r
            xy[i] = lead+r*np.sin(theta), r*(1-np.cos(theta))
        elif v <= breaks[3]:
            d = v-breaks[2]; angle[i] = a
            xy[i] = lead+r*np.sin(a)+d*np.cos(a), r*(1-np.cos(a))+d*np.sin(a)
        elif v <= breaks[4]:
            theta = a-(v-breaks[3])/r; angle[i] = theta; curvature[i] = -1/r
            xy[i] = lead+2*r*np.sin(a)+gap*np.cos(a)-r*np.sin(theta), r*(1-2*np.cos(a)+np.cos(theta))+gap*np.sin(a)
        else: xy[i] = xend+v-breaks[4], 22.
    xy[0] = (0., 0.); xy[-1] = (56., 22.)
    n = np.column_stack((-np.sin(angle), np.cos(angle)))
    # Analytic tangent offsets, then coarsen only the wall polyline, not the DEM.
    ids = np.unique(np.r_[np.arange(0, len(s), 5), len(s)-1,
                         [np.argmin(abs(s-v)) for v in breaks]])
    left, right = xy[ids]+2*n[ids], xy[ids]-2*n[ids]
    vertices = np.vstack((right, left[::-1]))
    vertices[0]=(0.,-2.);vertices[len(ids)-1]=(56.,20.)
    vertices[len(ids)]=(56.,24.);vertices[-1]=(0.,2.)
    # Eliminate roundoff-only (e.g. 24+7e-15) kinks at circular/straight joins.
    # Keep exact physical sections, and remove redundant collinear wall vertices.
    from shapely.geometry import Polygon
    vertices=np.asarray(Polygon(np.round(vertices,10)).simplify(1e-9,preserve_topology=True).exterior.coords)[:-1]
    return dict(s=s.tolist(), xy=xy.tolist(), angle=angle.tolist(), curvature=curvature.tolist(),
                breaks=breaks.tolist(), domain=vertices.tolist(), radius_m=r, width_m=4.,
                slope=.005, datum_m=1., normal_depth_m=.6, chezy=30.,
                discharge_m3_s=float(4*.6*30*np.sqrt(.6*.005)), grain_mm=10.,
                warmup_s=200., end_s=400., morph_factor=5., dem_pixel_m=.25, mesh_m=.5,
                viscosity_m2_s=.01, secondary_diffusivity_m2_s=.01,
                backend='cpu', threads=4)


def create():
    from osgeo import gdal, osr
    from PySide6.QtWidgets import QApplication
    from shapely.geometry import LineString, Polygon
    import shapely
    from braids_desktop import services, project_file
    from braids_desktop.forms import SettingsPanel
    from braids_desktop.river_physics import set_choice
    from braids_core.boundaries.domain import PolygonDomain
    from braids_core.river_open_boundary import resolve
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True); cfg = geometry()
    save(OUT/'CASE.json', cfg)
    line=LineString(cfg['xy']); domain=Polygon(cfg['domain'])
    assert domain.is_valid and domain.area > 250
    ds=.25; xmin,ymin,xmax,ymax=-6.,-8.,62.,30.
    xs=np.arange(xmin+ds/2,xmax,ds);ys=np.arange(ymax-ds/2,ymin,-ds)
    xx,yy=np.meshgrid(xs,ys);p=shapely.points(xx.ravel(),yy.ravel())
    station=shapely.line_locate_point(line,p); dist=shapely.distance(p,line)
    z=(1.-.005*station+np.clip(dist-2.,0.,1.5)).reshape(xx.shape)
    dem=OUT/'two_meanders_dem.tif'
    gdal.UseExceptions();raster=gdal.GetDriverByName('GTiff').Create(str(dem),len(xs),len(ys),1,gdal.GDT_Float64,
        options=['COMPRESS=LZW','TILED=YES'])
    raster.SetGeoTransform((xmin,ds,0.,ymax,0.,-ds));srs=osr.SpatialReference();srs.ImportFromEPSG(2056)
    raster.SetProjection(srs.ExportToWkt());raster.GetRasterBand(1).WriteArray(z);raster=None
    # Synthetic Cartesian coordinates: EPSG is only a metre-unit label, not a real surveyed location.
    app=QApplication.instance() or QApplication([])
    docs=[]
    for enabled in (False,True):
        key='advanced_on' if enabled else 'advanced_off'; folder=OUT/key
        doc=services.default_document(dem,cfg['domain'],'EPSG:2056',name='Two meanders - '+key,
            backend='cpu',minimum_size_m=.5,levels=1,end_time_s=400.)
        domain_edges=PolygonDomain(**doc['settings']['domain']).edges
        for x,name,role,control in ((0.,'Upstream','inlet',dict(method='inlet.normal_discharge',discharge_m3_s=cfg['discharge_m3_s'],energy_slope='mesh')),
            (56.,'Downstream','outlet',dict(method='outlet.stage',stage_m=1.-.005*cfg['s'][-1]+.6))):
            ids=[e.edge_id for e in domain_edges if e.start_xy[0]==e.end_xy[0]==x]
            doc=services.assign_boundary(doc,ids,name,role,control,for_editing=True)
        f=SettingsPanel();f.load_document(doc)
        set_choice(f.roughness,'roughness.chezy');f.chezy.setValue(30.)
        f.sediment_enabled.setChecked(True);f.physics.bedload_on.setChecked(True)
        set_choice(f.physics.capacity,'capacity.smart_jaggi')
        f.physics.grains.item(0,1).setText('10');f.physics.layers.item(0,0).setText('1')
        f.secondary.setChecked(enabled);f.betac.setValue(1.)
        f.secondary_diffusion.setValue(.01);f.turbulence.setChecked(enabled);f.viscosity.setValue(.01)
        f.sediment_start.setValue(200.);f.morfac.setValue(5.)
        f.end_time.setValue(400.);f.dt_max.setValue(.2);f.cfl.setValue(.8);f.output_interval.setValue(10.)
        f.threads.setValue(4);doc=f.document();f.close();doc.pop('editor_state',None)
        doc['initial'].update(stage_m=1.6,velocity_m_s=[0.,0.])
        doc['settings']['sediment_boundaries']['Upstream']=dict(kind='flow_free_equilibrium',concentration_kg_m3=[0.])
        doc['settings']['checkpoint_interval_s']=200.
        doc['boundary_resolution']=resolve(doc['settings']['hydraulics'],doc['boundary_states'],
            dict(Upstream='inlet',Downstream='outlet'),sediment=doc['model']['physics']['sediment'],ng=1,end_time=400.)
        saved=services.save_project_folder(folder,doc)
        project_file.write(project_file.default_file(folder),folder,saved);docs.append(doc)
    def differences(a,b,path=''):
        if isinstance(a,dict) and isinstance(b,dict):
            return sum((differences(a.get(k),b.get(k),path+'/'+k) for k in a.keys()|b.keys()),[])
        return [] if a==b else [dict(path=path,off=a,on=b)]
    save(OUT/'INPUT_DIFFERENCES.json',differences(*docs))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    REPORT.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(12,6))
    im=ax.imshow(z,extent=[xmin,xmax,ymin,ymax],origin='upper',cmap='terrain')
    v=np.asarray(cfg['domain']);ax.plot(*np.vstack((v,v[:1])).T,'k',lw=.7)
    xy=np.asarray(cfg['xy']);ax.plot(*xy.T,'w--',lw=.8)
    ax.annotate('Inlet: Q = 3.944 m³/s',xy=(0,0),xytext=(-3,-6),arrowprops={'arrowstyle':'->'})
    ax.annotate('Outlet: fixed stage',xy=(56,22),xytext=(42,28),arrowprops={'arrowstyle':'->'})
    for j,s in enumerate(((cfg['breaks'][1]+cfg['breaks'][2])/2,(cfg['breaks'][3]+cfg['breaks'][4])/2)):
        k=np.argmin(abs(np.asarray(cfg['s'])-s));n=np.array([-np.sin(cfg['angle'][k]),np.cos(cfg['angle'][k])])
        ax.plot(*np.array([xy[k]-3*n,xy[k]+3*n]).T,'k--');ax.text(*xy[k],f'  Bend {j+1}')
    ax.set(xlabel='Synthetic x (m)',ylabel='Synthetic y (m)',title='Two open 60° bends | R = 20 m, width = 4 m | initial transverse bed flat')
    fig.colorbar(im,ax=ax,label='DEM elevation (m)');fig.tight_layout();fig.savefig(REPORT/'INITIAL_DEM.png',dpi=170);plt.close(fig)
    print('CREATED',OUT,flush=True)


def run(key,until,restart):
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    from braids_core import river_worker,river_folder
    from braids_desktop.preparation import prepare
    verify();install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
    folder=OUT/key;prep=folder/'prepared/base';output=folder/f'runs/to_{until:g}'
    if output.exists():raise FileExistsError(output)
    if not prep.exists():
        t=time.perf_counter();prepare(folder/'braids_project.json',prep,asset_cache=OUT/'cache')
        save(folder/'PREPARATION_TIME.json',dict(elapsed_s=time.perf_counter()-t))
    checkpoint=None if restart is None else folder/f'runs/to_{restart:g}/checkpoint.npz'
    river_folder.record_event(folder,'synthetic_meander_run',status='started',prepared='prepared/base',
        output=output.relative_to(folder).as_posix(),restart=checkpoint.relative_to(folder).as_posix() if checkpoint else None)
    begin=time.perf_counter()
    def progress(path,row):
        print(key,'time',row['time_s'],'wall',round(time.perf_counter()-begin,2),flush=True)
    status=river_worker.run(prep,output,until=until,restart=checkpoint,on_frame=progress,checkpoint_dir=folder/'checkpoints')
    save(output/'EXECUTION.json',dict(status=status,elapsed_s=time.perf_counter()-begin,
         runtime_sha256=hashlib.sha256((RUNTIME/'RUNTIME.json').read_bytes()).hexdigest()))
    river_folder.record_event(folder,'synthetic_meander_run',status=status['status'],prepared='prepared/base',output=output.relative_to(folder).as_posix())
    river_folder.update_session(folder,dict(prepared='prepared/base',output=output.relative_to(folder).as_posix()))
    assert status['status'] in ('complete','segment_complete') and status['time_s']==until,status
    print('COMPLETE',key,until,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('create','run'))
    ap.add_argument('--case',choices=('advanced_off','advanced_on'));ap.add_argument('--until',type=float,default=200.)
    ap.add_argument('--restart',type=float)
    args=ap.parse_args()
    if args.action=='create':create()
    else:run(args.case,args.until,args.restart)
