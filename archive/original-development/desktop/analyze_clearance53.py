"""Audit the synthetic DEM and display the two independent shutdown checks."""
import json,shutil
from pathlib import Path
import numpy as np
from osgeo import gdal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'desktop/qualification/turbine_clearance53';REPORT=ROOT/'validation/hydrokinetic_turbines/reports/clearance_hkt4b'


def frame(folder):
    r=folder/'frames';index=json.loads((r/'INDEX.json').read_text());row=index['frames'][-1]
    with np.load(r/row['file']) as d:q={k:d[k] for k in d.files}
    with np.load(r/row['epoch']) as d:m={k:d[k] for k in d.files}
    return q,m


def main():
    REPORT.mkdir(parents=True,exist_ok=True);template=OUT/'channel_template';c=json.loads((template/'CHANNEL_SPEC.json').read_text());p=template/'trapezoidal_channel_0p10m.tif'
    ds=gdal.Open(str(p));z=ds.ReadAsArray();gt=ds.GetGeoTransform();ny,nx=z.shape;x=(np.arange(nx)+.5)*gt[1]-1;y=13+(np.arange(ny)+.5)*gt[5]
    exact=500.+.001*(30.-x[None,:])+np.minimum(2.5,np.maximum(abs(y[:,None]-6.)-2.,0.)/1.5)
    metrics=dict(dem=dict(shape=[ny,nx],pixel_m=[gt[1],-gt[5]],all_finite=bool(np.isfinite(z).all()),maximum_elevation_error_m=float(np.max(abs(z-exact))),spec=c),cases={})
    assert metrics['dem']['all_finite'] and metrics['dem']['maximum_elevation_error_m']<1e-10
    ds=None
    for case in ['normal','surface_off','bed_off']:
        a,ma=frame(OUT/'coupled_cpu'/case/'run');b,mb=frame(OUT/'coupled_gpu'/case/'run')
        for k in ['area','vertices','offsets','leaves']:np.testing.assert_array_equal(ma[k],mb[k])
        data=dict(maximum_backend_depth_m=float(np.max(abs(a['q'][0]-b['q'][0]))),maximum_backend_momentum=float(np.max(abs(a['q'][1:]-b['q'][1:]))),minimum_area_m2=float(np.min(ma['area'])),cells=len(ma['leaves']))
        peak=int(np.argmax(abs(a['q'][0]-b['q'][0])));p=Polygon(ma['vertices'][ma['offsets'][peak]:ma['offsets'][peak+1]]).centroid
        data['maximum_difference_cell_relative_xy_m']=[p.x-2600000.,p.y-1200000.]
        for backend in ['cpu','gpu']:
            root=OUT/('coupled_'+backend)/case/'run';rows=json.loads((root/'TURBINE_PERFORMANCE.json').read_text());diag=json.loads((root/'DIAGNOSTICS.json').read_text())
            data[backend]=dict(last=rows[-1],maximum_water_residual_m3=max(abs(r['water_residual_m3']) for r in diag))
        metrics['cases'][case]=data
    power_max=max(r['useful_power_W'] for case in ['normal','surface_off','bed_off'] for b in ['cpu','gpu'] for r in json.loads((OUT/('coupled_'+b)/case/'run/TURBINE_PERFORMANCE.json').read_text()))
    fig,axes=plt.subplots(2,3,figsize=(13,7.5))
    for i,case in enumerate(['normal','surface_off','bed_off']):
        d=metrics['cases'][case]['gpu']['last'];values=[d['minimum_bed_clearance_m'],d['required_bed_clearance_m'],d['minimum_coverage_margin_m'],d['required_surface_clearance_m']]
        ax=axes[0,i];ax.bar(range(4),values,color=['sienna','tan','tab:blue','skyblue']);ax.set(xticks=range(4),xticklabels=['Bed gap','Bed limit','Top cover','Top limit'],ylabel='Distance (m)',ylim=(0,1.1),title=f'{case.replace("_"," ")}, t={d["time_s"]:g} s\n'+('Rotor ON' if d['active'] else 'Rotor OFF'))
        ax.tick_params(axis='x',labelrotation=25);ax.grid(axis='y',alpha=.2)
        for b,color in [('cpu','tab:blue'),('gpu','tab:orange')]:
            rows=json.loads((OUT/('coupled_'+b)/case/'run/TURBINE_PERFORMANCE.json').read_text());axes[1,i].plot([r['time_s'] for r in rows],[r['useful_power_W'] for r in rows],color=color,ls='-' if b=='cpu' else '--',label=b.upper())
        axes[1,i].set(xlabel='Hydraulic time (s)',ylabel='Useful power (W)',ylim=(0,1.1*power_max));axes[1,i].grid(alpha=.2);axes[1,i].legend(fontsize=8)
    fig.suptitle('Independent clearance checks; no tuning or new experimental comparison');fig.tight_layout();fig.savefig(REPORT/'CLEARANCE_CHECKS.png',dpi=170);plt.close(fig)
    shutil.copy2(template/'CHANNEL_AND_TURBINE.png',REPORT/'CHANNEL_AND_TURBINE.png');(REPORT/'METRICS.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps({case:{k:v for k,v in data.items() if k not in ['cpu','gpu']} for case,data in metrics['cases'].items()},indent=2))


if __name__=='__main__':main()
