"""Small coupled release comparison figures and machine-readable checks."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from braids_core.sediment.adaptive_river_worker import read_live_frame
from braids_gpu.atomic_output import write_json_atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--tests-root',type=Path,required=True);a=p.parse_args()
    out=ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/explicit_delivery127'
    out.mkdir(parents=True,exist_ok=True);metrics={}
    base=next(d for d in a.tests_root.glob('unified-river*') if (d/'bedload_amr_gpu_0.225').is_dir())
    for case in ('hydro_fixed','hydro_amr','bedload_fixed','bedload_amr','lake_amr'):
        c=read_live_frame(base/f'{case}_cpu_0.225/run')[2]
        _,mesh,g=read_live_frame(base/f'{case}_gpu_0.225/run')
        f=read_live_frame(base/f'{case}_gpu_0.05625/run')[2]
        metrics[case]=dict(cpu_gpu_depth_mm=float(abs(c['q'][0]-g['q'][0]).max()*1000),
            cpu_gpu_bed_mm=float(abs(c['z']-g['z']).max()*1000),
            finer_time_depth_mm=float(abs(f['q'][0]-g['q'][0]).max()*1000),
            finer_time_bed_mm=float(abs(f['z']-g['z']).max()*1000),
            bed_change_range_mm=[float(((g['z']-g['rock']-.22)*1000).min()),float(((g['z']-g['rock']-.22)*1000).max())])
        if case=='bedload_amr':
            polygons=[mesh['vertices'][mesh['offsets'][i]:mesh['offsets'][i+1]] for i in range(len(mesh['leaves']))]
            fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
            values=[1000*(c['z']-c['rock']-.22),1000*(g['z']-g['rock']-.22),1000*(g['z']-f['z'])]
            limit=max(abs(values[0]).max(),abs(values[1]).max(),1e-10)
            for i,(ax,v,title) in enumerate(zip(axes,values,('CPU bed change','GPU bed change','GPU minus CFL / 4'))):
                h=PolyCollection(polygons,array=v,cmap='RdBu_r',edgecolors='0.7',linewidths=.2)
                bound=limit if i<2 else max(abs(v).max(),.001);h.set_clim(-bound,bound)
                ax.add_collection(h);ax.autoscale();ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('East (m)')
                fig.colorbar(h,ax=ax,label='mm')
            axes[0].set_ylabel('North (m)');fig.savefig(out/'BEDLOAD_AMR.png',dpi=170);plt.close(fig)
    write_json_atomic(out/'COUPLED_METRICS.json',metrics)
    audit=ROOT/'validation/_delivery127/amr_budget_audit/AMR_AUDIT.json'
    rows=json.loads(audit.read_text());fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    ax.plot([r['after']['time'] for r in rows],[1e9*(r['after']['ledger']-r['after']['water']) for r in rows],
        'o-',label='Before: remapped boundary ledger')
    for backend in ('cpu','gpu'):
        rows=json.loads((ROOT/f'validation/_delivery127/amr_history_fixed/{backend}_run/DIAGNOSTICS.json').read_text())
        ax.plot([r['time_s'] for r in rows],[-1e9*r['water_residual_m3'] for r in rows],'.-',label='Corrected '+backend.upper())
    ax.set(xlabel='Simulation time (s)',ylabel='Ledger minus water (microlitres)',title='Exact open-face history during wall remeshing')
    ax.legend();fig.savefig(out/'AMR_BOUNDARY_HISTORY.png',dpi=170);plt.close(fig)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
