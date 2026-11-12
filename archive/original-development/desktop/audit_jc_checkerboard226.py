"""Read-only native-frame audit of the Jiang-Cao experiment's odd-even mode."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CASE=ROOT.parent/'BRAIDS_JiangCao_Experiment225'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_experiment225/checkerboard_audit226'

def main():
    frames=CASE/'runs/experiment225/frames'
    index=json.loads((frames/'INDEX.json').read_text(encoding='utf8'))
    row=min(index['frames'],key=lambda r:abs(r['time_s']-240.))
    if row['time_s']!=240.:raise ValueError('The frozen 240 s frame is required')
    with np.load(frames/row['epoch']) as mesh:
        vertices,offsets=mesh['vertices'],mesh['offsets']
        cx=np.array([(vertices[offsets[i]:offsets[i+1],0].min()+vertices[offsets[i]:offsets[i+1],0].max())/2 for i in range(len(offsets)-1)])
        cy=np.array([(vertices[offsets[i]:offsets[i+1],1].min()+vertices[offsets[i]:offsets[i+1],1].max())/2 for i in range(len(offsets)-1)])
        area=mesh['area'].copy()
    ix=np.rint((cx-.025)/.05).astype(int);iy=np.rint((cy-.025)/.05).astype(int)
    if len(set(zip(ix,iy)))!=15000:raise ValueError('Expected regular 500 x 30 diagnostic grid')
    def grid(value):
        result=np.empty((30,500));result[iy,ix]=value;return result
    with np.load(frames/row['file']) as frame:
        depth=grid(frame['q'][0]);bed=grid(frame['z']-frame['initial_bed_m'])
        qx=grid(frame['bedload_vector_m2_s'][0]);qy=grid(frame['bedload_vector_m2_s'][1])
        capacity=grid(frame['bedload_capacity_m2_s'])
    magnitude=np.hypot(qx,qy);x=.025+.05*np.arange(500);y=.025+.05*np.arange(30)
    mean_bed=bed.mean(0);mean_capacity=capacity.mean(0);mean_transport=magnitude.mean(0)
    face_slope=np.diff(grid(frame['z']) if False else bed,axis=1)/.05
    centered=(bed[:,2:]-bed[:,:-2])/.1
    feed=(.00286/2650.)/1.5
    metrics=dict(time_s=240.,bed_change_mm=dict(min=float(bed.min()*1000),max=float(bed.max()*1000)),
        first_30_width_mean_bed_change_mm=(mean_bed[:30]*1000).tolist(),
        adjacent_bed_change_slope_x_first_2m=dict(max_abs=float(np.max(abs(face_slope[:,:39]))),median_abs=float(np.median(abs(face_slope[:,:39])))),
        centered_gradient_proxy_x_first_2m=dict(max_abs=float(np.max(abs(centered[:,:38]))),median_abs=float(np.median(abs(centered[:,:38])))),
        inlet_feed_solid_per_width_m2_s=feed,
        net_bulk_bed_change_first_3m_m3=float(np.sum(bed[:,x<3])*.0025),
        net_solid_bed_change_first_3m_m3=float(np.sum(bed[:,x<3])*.0025*.6),
        supplied_solid_to_240s_m3=float(.00286/2650.*240.),
        interpretation='Growing one-cell longitudinal odd-even mode; not physical bars or transverse sorting.')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'METRICS.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    fig,axes=plt.subplots(3,1,figsize=(12,9),layout='constrained')
    im=axes[0].pcolormesh(np.r_[x-.025,25.],np.r_[y-.025,1.5],bed*1000,cmap='RdBu_r',vmin=-5,vmax=5,shading='flat')
    axes[0].set(title='Raw bed change at 240 s (no smoothing)',ylabel='y (m)',xlim=(0,5));fig.colorbar(im,ax=axes[0],label='bed change (mm)')
    axes[1].plot(x,mean_bed*1000,color='#222',lw=1,label='cross-width mean')
    axes[1].fill_between(x,(mean_bed-bed.std(0))*1000,(mean_bed+bed.std(0))*1000,color='#5da5da',alpha=.25,label='± transverse std.')
    axes[1].set(title='Longitudinal odd-even bed mode',ylabel='bed change (mm)',xlim=(0,5));axes[1].axhline(0,color='.6',lw=.7);axes[1].legend()
    axes[2].semilogy(x,np.maximum(mean_transport,1e-25),label='instantaneous transported bedload magnitude')
    axes[2].semilogy(x,np.maximum(mean_capacity,1e-25),label='instantaneous capacity')
    axes[2].axhline(feed,color='k',ls='--',label='prescribed inlet solid feed / width')
    axes[2].set(title='Cross-width mean transport fields',xlabel='x (m)',ylabel='solid discharge (m²/s)',xlim=(0,5),ylim=(1e-10,2e-6));axes[2].legend()
    fig.savefig(OUT/'CHECKERBOARD_AND_TRANSPORT.png',dpi=180);plt.close(fig)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
