"""Synthetic metre-scale trapezoidal channel and a ready 1 m rotor project."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
from osgeo import gdal,osr

ROOT=Path(__file__).resolve().parents[1]


def create(root,backend='cpu',*,duration=60.):
    from braids_desktop import services,project_file
    from braids_core.boundaries.domain import PolygonDomain
    from braids_core import river_turbines as turbine
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    c=dict(length_m=30.,bottom_width_m=4.,domain_width_m=12.,bank_height_m=2.5,side_slope_H_per_V=1.5,
        bed_slope=.001,outlet_invert_m=500.,reference_depth_m=2.,manning_n=.025,pixel_m=.1,
        origin_xy=[2600000.,1200000.],rotor_diameter_m=1.,hub_height_m=.85,minimum_bed_clearance_m=.25,minimum_surface_clearance_m=.25)
    ox,oy=c['origin_xy'];nx,ny=320,140;pixel=c['pixel_m'];x=-1+(np.arange(nx)+.5)*pixel;y=13-(np.arange(ny)+.5)*pixel
    bank=np.minimum(c['bank_height_m'],np.maximum(abs(y-6)-c['bottom_width_m']/2,0)/c['side_slope_H_per_V'])
    z=c['outlet_invert_m']+c['bed_slope']*(c['length_m']-x[None,:])+bank[:,None]
    dem=root/'trapezoidal_channel_0p10m.tif';r=gdal.GetDriverByName('GTiff').Create(str(dem),nx,ny,1,gdal.GDT_Float64,options=['COMPRESS=DEFLATE','PREDICTOR=3'])
    r.SetGeoTransform((ox-1,pixel,0,oy+13,0,-pixel));srs=osr.SpatialReference();srs.ImportFromEPSG(2056);r.SetProjection(srs.ExportToWkt());r.GetRasterBand(1).WriteArray(z);r.GetRasterBand(1).SetUnitType('m');r.SetMetadataItem('SYNTHETIC','YES');r.SetMetadataItem('DESCRIPTION','Straight trapezoidal test channel, not measured terrain');r.FlushCache();r=None
    depth=c['reference_depth_m'];area=(c['bottom_width_m']+c['side_slope_H_per_V']*depth)*depth;perimeter=c['bottom_width_m']+2*depth*math.sqrt(1+c['side_slope_H_per_V']**2)
    Q=area*(area/perimeter)**(2/3)*math.sqrt(c['bed_slope'])/c['manning_n'];c.update(reference_area_m2=area,reference_discharge_m3_s=Q,reference_mean_velocity_m_s=Q/area)
    coords=[[ox,oy],[ox+30,oy],[ox+30,oy+12],[ox,oy+12]]
    doc=services.default_document(dem,coords,'EPSG:2056',name='Trapezoidal channel — 1 m turbine',backend=backend,minimum_size_m=.25,levels=2,mesh_mode='static_terrain',end_time_s=duration)
    doc['roughness_parameters']['manning_s_m_third']=c['manning_n'];doc['settings']['adaptation']['domain_border_enabled']=False
    doc['settings']['controls']['threads']=4;doc['settings'].update(dt_max_s=.5,output_interval_s=min(.5,duration),checkpoint_interval_s=duration)
    doc['initial'].update(stage_m=500.+depth+.001*15,velocity_m_s=[Q/area,0.])
    domain=PolygonDomain(**doc['settings']['domain']);left=[e.edge_id for e in domain.edges if e.start_xy[0]==e.end_xy[0]==ox];right=[e.edge_id for e in domain.edges if e.start_xy[0]==e.end_xy[0]==ox+30]
    doc=services.assign_boundary(doc,left,'Upstream','inlet',dict(method='inlet.normal_discharge',discharge_m3_s=Q,energy_slope='mesh'))
    doc=services.assign_boundary(doc,right,'Downstream','outlet',dict(method='outlet.stage',stage_m=500.+depth))
    doc['turbines']=turbine.default();doc['turbines']['turbines']=[turbine.definition([ox+10,oy+6],[1.,0.],name='1 m rotor',diameter=1.,hub_height=.85,minimum_bed_clearance=.25,minimum_surface_clearance=.25)]
    doc=services.validate_document(doc)
    (root/'input.json').write_text(json.dumps(doc,indent=2)+'\n');saved=services.save_project_folder(root/'project_data',doc)
    project_file.write(root/'Channel.braids.json',root/'project_data',saved)
    (root/'CHANNEL_SPEC.json').write_text(json.dumps(c,indent=2)+'\n')
    gj=dict(type='FeatureCollection',crs=dict(type='name',properties=dict(name='EPSG:2056')),features=[dict(type='Feature',properties=dict(name='Synthetic channel domain'),geometry=dict(type='Polygon',coordinates=[coords+[coords[0]]]))])
    (root/'domain.geojson').write_text(json.dumps(gj,indent=2)+'\n')
    return doc,c


def figure(root,c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    y=np.linspace(-6,6,601);bed=np.minimum(2.5,np.maximum(abs(y)-2,0)/1.5);h=c['reference_depth_m'];fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    ax=axes[0];ax.plot(y,bed,color='sienna',label='Bed');ax.fill_between(y,bed,h,where=bed<h,color='skyblue',alpha=.4);ax.axhline(h,color='tab:blue',label='Reference water level')
    ax.add_patch(Circle((0,.85),.5,fill=False,color='teal',lw=2,label='1 m rotor'))
    for lo,hi in [(0.,.35),(1.35,2.)]:
        ax.plot([.8,.8],[lo,hi],color='black',lw=.9)
        for level in [lo,hi]:ax.plot([.73,.87],[level,level],color='black',lw=.9)
    ax.text(.95,.13,'0.35 m actual bed gap',fontsize=8);ax.text(.95,1.62,'0.65 m reference cover',fontsize=8)
    ax.set(xlabel='Across channel (m)',ylabel='Height above invert (m)',title='Trapezoidal section; required gaps = 0.25 m each',aspect='equal',ylim=(-.1,2.8));ax.legend(fontsize=7,loc='lower left')
    axes[1].plot([0,30],[.03,0],color='sienna',label='Invert');axes[1].plot([0,30],[2.03,2],color='tab:blue',label='Uniform-depth reference');axes[1].plot([0,30],[2.53,2.5],color='.65',ls=':',label='Banks')
    axes[1].plot([10,10],[.37,1.37],color='teal',lw=4,label='Rotor envelope')
    axes[1].set(xlabel='Downstream distance (m)',ylabel='Height above outlet invert (m)',title='30 m channel; longitudinal slope 0.1%');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(Path(root)/'CHANNEL_AND_TURBINE.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--backend',choices=['cpu','gpu'],default='cpu');p.add_argument('--duration',type=float,default=60.);p.add_argument('--runtime',type=Path);a=p.parse_args()
    if a.runtime:sys.path.insert(0,str(a.runtime.resolve()))
    d,c=create(a.root,a.backend,duration=a.duration);figure(a.root,c);print(json.dumps(c,indent=2))
