"""Audit accepted fluxes and plot model states without treating either as truth."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from . import run_case as r
from . import experiment as e


def load(folder):
    p=json.loads((folder/'input_provenance.json').read_text())
    execution=json.loads((folder/'execution.json').read_text())
    if execution['return_code'] != 0:raise ValueError('Failed engine execution')
    for name,digest in p['files'].items():
        if r.sha256(folder/name)!=digest:raise ValueError(f'Changed input: {name}')
    g=np.load(folder/'analysis_geometry.npz')
    if p['engine']=='braids':
        frames=r.shared.read_frames(r.shared.native_frames_path(folder))
        t=np.array([f['time_s'] for f in frames]);q=np.stack([f['q'] for f in frames])
        h=q[:,:,0];z=g['bed'][None,:]-r.RESERVOIR+q[:,:,5:9].sum(axis=2)/r.BULK
        speed=np.linalg.norm(q[:,:,1:3],axis=2)/np.maximum(h,1e-12)
        xy,area,s,n=(g[k] for k in ('xy','area','s','n'))
        ledger=np.stack([f['ledger'] for f in frames])
        win=ledger[:,0,0];wout=-ledger[:,-1,0]
        sin=ledger[:,0,3:].sum(axis=1);sout=-ledger[:,-1,3:].sum(axis=1)
        mesh,_=r.native_mesh(p['dx']);polygons=[c.vertices_xy for c in mesh.cells]
        nk=None
    else:
        with Dataset(folder/'trim-basin.nc') as d:
            t=np.asarray(d['time'][:]);h=np.asarray(d['S1'][:,1:-1,1:-1]+d['DPS'][:,1:-1,1:-1]).reshape(len(t),-1)
            z=-np.asarray(d['DPS'][:,1:-1,1:-1]).reshape(len(t),-1)
            xy=np.column_stack([np.asarray(d[k][1:-1,1:-1]).ravel() for k in ('XZ','YZ')])
            area=np.asarray(d['GSQS'][1:-1,1:-1]).ravel()
            u=np.asarray(d['U1'][:,0]);v=np.asarray(d['V1'][:,0])
            uc=(u[:,1:-1,1:-1]+u[:,:-2,1:-1])/2
            vc=(v[:,1:-1,1:-1]+v[:,1:-1,:-2])/2
            speed=np.hypot(uc,vc).reshape(len(t),-1)
            nk=d['U1'].shape[1]
        s,n=e.sn(*xy.T)
        with Dataset(folder/'trih-basin.nc') as d:
            np.testing.assert_allclose(d['time'][:],t)
            win,wout=np.asarray(d['FLTR'][:]).T
            sin,sout=(np.asarray(d['SBTRC'][:,0,:])*r.RHOS*p['morfac']).T
        points=np.stack([g['gx'].T,g['gy'].T],axis=-1)
        polygons=np.stack([points[:-1,:-1],points[1:,:-1],points[1:,1:],points[:-1,1:]],axis=2).reshape(-1,4,2)
    if not np.isclose(t[-1],p['end']):raise ValueError('Incomplete trajectory')
    if not np.isfinite(h).all() or not np.isfinite(z).all():raise ValueError('Nonfinite trajectory')
    return dict(p=p,folder=folder,execution=execution,t=t,th=t*p['morfac'],h=h,z=z,speed=speed,
        xy=xy,s=s,n=n,area=area,win=win,wout=wout,sin=sin,sout=sout,polygons=polygons,nk=nk)


def sample(d,values,s,n):
    points=np.column_stack([d['s'],d['n']]);query=np.column_stack([np.ravel(s),np.ravel(n)])
    z=LinearNDInterpolator(points,values)(query);outside=~np.isfinite(z)
    if outside.any():z[outside]=NearestNDInterpolator(points,values)(query[outside])
    return z,outside


def audit(d):
    p=d['p'];last=d['t'][-1]-d['t'][-2]
    win,wout,sin,sout=[float(d[k][-1]-d[k][0]) for k in ('win','wout','sin','sout')]
    water_change=float(np.sum((d['h'][-1]-d['h'][0])*d['area']))
    dz=d['z'][-1]-d['z'][0]
    bed_change=float(np.sum(dz*d['area'])*r.BULK)
    feed=0. if p['hydraulic'] else float(r.FEED*d['th'][-1])
    qin=float((d['win'][-1]-d['win'][-2])/last)
    qout=float((d['wout'][-1]-d['wout'][-2])/last)
    bend=(d['s']>=e.APPROACH)&(d['s']<=e.BEND_END)
    inlet=d['s']<p['dx']*1.1;outlet=d['s']>e.LENGTH-2*p['dx']
    # A local-plane residual diagnoses short-scale variation, not a proof of
    # checkerboarding: real bank gradients also contribute. No smoothing used.
    distances,neighbors=cKDTree(d['xy']).query(d['xy'],k=9)
    residual=[]
    for i,ids in enumerate(neighbors):
        A=np.column_stack([np.ones(len(ids)),d['xy'][ids]-d['xy'][i]])
        fitted=np.linalg.lstsq(A,dz[ids],rcond=None)[0][0]
        residual.append(dz[i]-fitted)
    stage=d['h']+d['z']
    return dict(engine=p['engine'],hydraulic=p['hydraulic'],end_hydraulic_s=float(d['t'][-1]),
        end_morphological_s=float(d['th'][-1]),vertical_layers=d['nk'],
        qin_last_interval_m3_s=qin,qout_last_interval_m3_s=qout,
        inlet_discharge_relative_error=abs(qin/e.Q-1),outlet_discharge_relative_error=abs(qout/e.Q-1),
        min_depth_m=float(d['h'].min()),minimum_inlet_depth_m=float(d['h'][:,inlet].min()),
        final_stage_change_over_output_interval_m=float(np.max(abs(stage[-1]-stage[-2]))),
        final_bed_rms_change_over_output_interval_m=float(np.sqrt(np.mean((d['z'][-1,bend]-d['z'][-2,bend])**2))),
        water_inventory_change_m3=water_change,water_in_m3=win,water_out_m3=wout,
        water_budget_residual_m3=water_change-win+wout,
        prescribed_feed_kg=feed,accepted_feed_kg=sin,exported_sediment_kg=sout,
        prescribed_feed_fraction=(sin/feed if feed>0 else None),
        bed_inventory_change_kg=bed_change,sediment_budget_residual_kg=bed_change-sin+sout,
        sediment_budget_residual_over_throughput=abs(bed_change-sin+sout)/max(abs(sin),abs(sout),1e-9),
        max_erosion_m=float(-dz.min()),max_deposition_m=float(dz.max()),
        outlet_max_deposition_m=float(dz[outlet].max()),
        bend_outer_mean_change_m=float(np.mean(dz[bend&(d['n']<e.WIDTH*.25)])),
        bend_inner_mean_change_m=float(np.mean(dz[bend&(d['n']>e.WIDTH*.75)])),
        local_plane_residual_rms_m=float(np.sqrt(np.mean(np.asarray(residual)[bend]**2))),
        local_plane_residual_note='Unfiltered cell-scale diagnostic, not proof of checkerboard instability',
        equilibrium_established=False)


def analyze(tag,folders):
    report=r.HERE/'reports'/tag;report.mkdir(parents=True,exist_ok=True)
    data=[load(Path(f)) for f in folders]
    metrics=[audit(d) for d in data]
    r.write_json(report/'metrics.json',dict(cases=metrics,reference='digitized experiment',
        exact_paper_replication=False,accuracy_ranking_supported=False))
    fig,axes=plt.subplots(2,len(data),figsize=(6*len(data),9.5),squeeze=False,layout='constrained')
    lim=max(.001,max(float(np.max(abs(d['z'][-1]-d['z'][0]))) for d in data))
    for j,d in enumerate(data):
        for row,field,cmap,clim,title in [(0,d['h'][-1],'Blues',(0,.25),'Water depth (m)'),
            (1,d['z'][-1]-d['z'][0],'RdBu_r',(-lim,lim),'Bed change (m): blue erosion, red deposition')]:
            ax=axes[row,j];im=PolyCollection(d['polygons'],array=field,cmap=cmap,clim=clim,edgecolors='none')
            ax.add_collection(im);ax.autoscale();ax.set_aspect('equal');ax.set(xlabel='x (m)',ylabel='y (m)')
            ax.set_title(f"{d['p']['engine'].upper()} 2D, dx={d['p']['dx']:g} m\n{title}",fontsize=11)
            fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('Actual-cell fields; no display smoothing\n'+', '.join(
        f"{d['p']['engine']}: {d['th'][-1]:g} morphological s" for d in data))
    fig.savefig(report/'fields.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),sharey=True,layout='constrained')
    comparisons=[]
    for ax,profile in zip(axes,e.observations()['profiles']):
        pts=profile['points'];coord=np.array([p['relative_from_outer'] for p in pts])
        obs=np.array([p['bed_change_m'] for p in pts]);primary=np.array([p['primary_interior'] for p in pts])
        ss=np.full(len(coord),e.APPROACH+e.RADIUS*math.radians(profile['angle_degrees']))
        ax.errorbar(coord,obs,yerr=.005,fmt='ko',ms=3,capsize=2,label='Published equilibrium markers')
        ax.axhline(0,color='.5',lw=.8)
        for d in data:
            pred,outside=sample(d,d['z'][-1]-d['z'][0],ss,coord*e.WIDTH)
            ax.plot(coord,pred,'o-',ms=2,color={'braids':'#bf432f','delft':'#2469ad'}[d['p']['engine']],
                    label=f"{d['p']['engine']} {d['th'][-1]:g} s (transient)")
            err=pred-obs;interior=primary&~outside
            comparisons.append(dict(engine=d['p']['engine'],folder=str(d['folder']),
                angle_degrees=profile['angle_degrees'],predicted_bed_change_m=pred.tolist(),
                interior_rmse_m=float(np.sqrt(np.mean(err[interior]**2))),
                demeaned_interior_rmse_m=float(np.sqrt(np.mean((err[interior]-err[interior].mean())**2))),
                nearest_fallback_count=int(outside.sum()),primary_used=int(interior.sum()),
                metrics=e.profile_metrics(coord,pred)))
        ax.set(title=f"{profile['angle_degrees']} degree section",xlabel='0 outer wall -> 1 inner wall');ax.grid(alpha=.2)
    axes[0].set_ylabel('Bed change (m)');axes[-1].legend(fontsize=8)
    fig.suptitle('First model evolution versus experimental EQUILIBRIUM shape\nTransient errors are not an accuracy ranking; bars show graphical tolerance only',fontsize=11)
    fig.savefig(report/'profiles.png',dpi=180);plt.close(fig)
    r.write_json(report/'profiles.json',comparisons)
    print(json.dumps(metrics,indent=2),flush=True)
    return data,metrics


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tag');p.add_argument('folders',nargs='+')
    a=p.parse_args();analyze(a.tag,a.folders)
