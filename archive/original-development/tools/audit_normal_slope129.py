"""Read-only slope audit of supplied prepared geometry/initial fields."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    from braids_core.sediment.adaptive_river_worker import hierarchy
    from braids_core.river_normal_slope import inward_bed_slope
    from braids_gpu.atomic_output import write_json_atomic
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,len(a.prepared),figsize=(6*len(a.prepared),8),squeeze=False)
    rows=[]
    for index,path in enumerate(a.prepared):
        raw=(path/'profile.json').read_bytes();profile=json.loads(raw);t=hierarchy(profile,None)
        with np.load(path/'initial.npz') as f:bed=f['rock']+f['thickness'].sum(0)
        name=next(n for n,v in profile['hydraulics'].items() if v['mode']=='total_discharge')
        row=inward_bed_slope(t.finest,bed,name);xy=np.array([c.centroid_xy for c in t.finest.cells])
        rows.append(dict(prepared=str(path.resolve()),profile_sha256=hashlib.sha256(raw).hexdigest(),normal_slope=row))
        inward=np.array(row['normal_inward']);tangent=np.array([-inward[1],inward[0]])
        distance=xy@inward;station=xy@tangent;first=min(distance[row['sample_cell_ids']])
        stations=[];slopes=[];widths=[]
        for strip in row['strips']:
            ids=np.array(strip['cells']);order=np.argsort(distance[ids]);ids=ids[order]
            axes[1,index].plot(distance[ids]-first,bed[ids]-bed[ids[0]],alpha=.6,lw=1)
            stations.append(float(np.mean(station[ids])));slopes.append(strip['inward_bed_fall_m_m']);widths.append(strip['boundary_width_m'])
        x=np.array(stations)-min(stations);axes[0,index].bar(x,slopes,width=.8*np.array(widths),color='#4f81bd')
        axes[0,index].axhline(row['energy_slope'],color='#ca5825',lw=2,label=f"Selected signed median {row['energy_slope']:.6g}")
        axes[0,index].axhline(row['area_weighted_mean_fit_slope_m_m'],color='#8652a2',ls='--',label='Unrobust section mean')
        axes[0,index].axhline(0.,color='k',lw=.7);axes[0,index].set_title(f"{row['finest_spacing_m']:g} m finest mesh — inlet strips")
        axes[0,index].set_xlabel('Transverse strip position (m)');axes[0,index].set_ylabel('Signed inward bed fall (m/m)');axes[0,index].legend(fontsize=8)
        axes[1,index].set_xlabel('Distance inward from first sample (m)');axes[1,index].set_ylabel('Bed change from strip first sample (m)')
        axes[1,index].axhline(0.,color='k',lw=.7);axes[1,index].set_title('Raw mesh bed profiles; rising levees are not channel slope')
        for ax in axes[:,index]:ax.grid(alpha=.2)
        print(json.dumps({k:v for k,v in row.items() if k not in ('strips','sample_cell_ids')},indent=2),flush=True)
    fig.tight_layout();fig.savefig(out/'NORMAL_SLOPE_MESH.png',dpi=160);plt.close(fig)
    write_json_atomic(out/'NORMAL_SLOPE_AUDIT.json',rows)


if __name__=='__main__':main()
