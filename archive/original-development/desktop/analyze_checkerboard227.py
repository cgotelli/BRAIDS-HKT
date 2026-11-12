"""Raw-field comparison for the Jiang--Cao odd/even correction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_experiment225/checkerboard_fix227'
CASES={
    'Original centered coupling':ROOT.parent/'BRAIDS_JiangCao_Experiment225/runs/experiment225/frames',
    'Slope-only candidate':ROOT/'validation/_checkerboard227/runs/cpu_60s/run/frames',
    'Face-compatible candidate':ROOT/'validation/_checkerboard227/runs/cpu_v3_120s_b/run/frames',
}
CORRECTED_240=ROOT/'validation/_checkerboard227/runs/cpu_v3_restart240/run/frames'


def load(folder,time_s):
    index=json.loads((folder/'INDEX.json').read_text(encoding='utf8'))
    row=min(index['frames'],key=lambda item:abs(item['time_s']-time_s))
    if abs(row['time_s']-time_s)>1e-9:raise ValueError(f'{folder} lacks {time_s:g} s frame')
    with np.load(folder/row['epoch']) as mesh:
        vertices,offsets=mesh['vertices'],mesh['offsets']
        cx=np.array([(vertices[offsets[i]:offsets[i+1],0].min()+vertices[offsets[i]:offsets[i+1],0].max())/2
                     for i in range(len(offsets)-1)])
        cy=np.array([(vertices[offsets[i]:offsets[i+1],1].min()+vertices[offsets[i]:offsets[i+1],1].max())/2
                     for i in range(len(offsets)-1)])
    ix=np.rint((cx-.025)/.05).astype(int);iy=np.rint((cy-.025)/.05).astype(int)
    def grid(values):
        out=np.empty((30,500));out[iy,ix]=values;return out
    with np.load(folder/row['file']) as frame:
        data=dict(time_s=time_s,bed=grid(frame['z']-frame['initial_bed_m']),
            depth=grid(frame['q'][0]),mobile=grid(frame['mobile_bedload_solid_m']),
            capacity=grid(frame['bedload_capacity_m2_s']),
            transport=grid(np.hypot(*frame['bedload_vector_m2_s'])))
    return data


def metrics(data):
    bed=data['bed'];mean=bed.mean(0);window=mean[2:40]
    high=window[1:-1]-.5*(window[:-2]+window[2:])
    cross=data['transport'][:,9]
    return dict(time_s=data['time_s'],bed_min_mm=float(bed.min()*1000),bed_max_mm=float(bed.max()*1000),
        width_mean_first_12_mm=(mean[:12]*1000).tolist(),
        adjacent_pair_mean_0p1_to_2m_mm=float(np.mean(np.abs(np.diff(window)))*1000),
        three_point_high_frequency_rms_0p1_to_2m_mm=float(np.sqrt(np.mean(high*high))*1000),
        negative_bed_volume_first_3m_m3=float(np.sum(np.minimum(bed[:,:60],0))*.0025),
        positive_bed_volume_first_3m_m3=float(np.sum(np.maximum(bed[:,:60],0))*.0025),
        transverse_bed_std_x1m_mm=float(np.std(bed[:,20])*1000),
        transverse_transport_mean_x0p475_m2_s=float(np.mean(cross)),
        transverse_transport_std_x0p475_m2_s=float(np.std(cross)),
        transverse_transport_cv_x0p475=float(np.std(cross)/max(np.mean(cross),1e-30)))


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    data60={name:load(path,60.) for name,path in CASES.items()}
    original240=load(CASES['Original centered coupling'],240.)
    corrected240=load(CORRECTED_240,240.)
    result={'at_60_s':{name:metrics(value) for name,value in data60.items()},
            'at_240_s':{'Original centered coupling':metrics(original240),
                        'Face-compatible candidate':metrics(corrected240)}}
    cpu=load(ROOT/'validation/_checkerboard227/runs/cpu_v3_120s_b/run/frames',20.)
    gpu=load(ROOT/'validation/_checkerboard227/runs/gpu_v3_20s/run/frames',20.)
    result['backend_20_s']={key+'_max_abs':float(np.max(np.abs(cpu[key]-gpu[key])))
                            for key in ('bed','depth','mobile','capacity','transport')}
    continuous=load(ROOT/'validation/_checkerboard227/runs/cpu_v3_120s_b/run/frames',120.)
    restarted=load(ROOT/'validation/_checkerboard227/runs/cpu_v3_restart240/run/frames',120.)
    result['restart_load_120_s']={key+'_max_abs':float(np.max(np.abs(continuous[key]-restarted[key])))
                            for key in ('bed','depth','mobile','capacity','transport')}
    result['runtime_sha256']='f8f896a590b38f0a0563b2d04fb6db823948a42ad688829849a4dfab370da4c5'
    (REPORT/'METRICS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=.025+.05*np.arange(500);y=.025+.05*np.arange(30)
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    for row,(name,data) in enumerate(data60.items()):
        image=axes[row,0].pcolormesh(np.r_[x-.025,25],np.r_[y-.025,1.5],data['bed']*1000,
            cmap='RdBu_r',vmin=-1.1,vmax=1.1,shading='flat')
        axes[row,0].set(title=name+' - raw bed change at 60 s',xlim=(0,3),ylim=(0,1.5),ylabel='y (m)')
        axes[row,1].plot(x,data['bed'].mean(0)*1000,color='#16324f',lw=1.2)
        axes[row,1].axhline(0,color='.6',lw=.7);axes[row,1].set(title='Cross-width mean',xlim=(0,3),ylabel='bed change (mm)')
    axes[-1,0].set_xlabel('x (m)');axes[-1,1].set_xlabel('x (m)')
    fig.colorbar(image,ax=axes[:,0],label='bed change (mm)',shrink=.75)
    fig.savefig(REPORT/'COUPLING_COMPARISON_60S.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(2,1,figsize=(13,6.5),layout='constrained')
    for ax,data,title in ((axes[0],original240,'Original centered coupling'),
                          (axes[1],corrected240,'Face-compatible candidate')):
        image=ax.pcolormesh(np.r_[x-.025,25],np.r_[y-.025,1.5],data['bed']*1000,
            cmap='RdBu_r',vmin=-5,vmax=5,shading='flat')
        ax.set(title=title+' - raw bed change at 240 s',xlim=(0,5),ylim=(0,1.5),ylabel='y (m)')
    axes[1].set_xlabel('x (m)');fig.colorbar(image,ax=axes,label='bed change (mm)',shrink=.8)
    fig.savefig(REPORT/'RAW_BED_240S.png',dpi=180);plt.close(fig)

    # Preserve the physical transverse signal: plot every raw row at 240 s,
    # plus the width mean.  No smoothing or interpolated display is used.
    fig,ax=plt.subplots(figsize=(12,4.5),layout='constrained')
    for row in corrected240['bed']:
        ax.plot(x,row*1000,color='#7aa6c2',alpha=.16,lw=.55)
    ax.plot(x,corrected240['bed'].mean(0)*1000,color='#142b3b',lw=1.4,label='width mean')
    ax.set(title='Corrected 240 s bed change: all 30 raw transverse rows',xlim=(0,5),xlabel='x (m)',ylabel='bed change (mm)')
    ax.axhline(0,color='.6',lw=.7);ax.legend();fig.savefig(REPORT/'TRANSVERSE_SIGNAL_240S.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(16,4.4),layout='constrained')
    image=axes[0].pcolormesh(np.r_[x-.025,25],np.r_[y-.025,1.5],corrected240['transport']*1e6,
        cmap='viridis',shading='flat')
    axes[0].set(title='Raw mobile bedload magnitude at 240 s',xlim=(0,.75),xlabel='x (m)',ylabel='y (m)')
    fig.colorbar(image,ax=axes[0],label='transport (mm2/s)')
    for column in (0,4,9,10):
        axes[1].plot(corrected240['transport'][:,column]*1e6,y,label=f'x={x[column]:.3f} m')
    axes[1].set(title='Unsmoothed transverse profiles',xlabel='transport (mm2/s)',ylabel='y (m)');axes[1].legend()
    mean=corrected240['transport'].mean(0)*1e6
    lo=np.percentile(corrected240['transport'],10,axis=0)*1e6
    hi=np.percentile(corrected240['transport'],90,axis=0)*1e6
    axes[2].fill_between(x,lo,hi,color='#8ebbd1',alpha=.5,label='10-90% across width')
    axes[2].plot(x,mean,color='#17384d',label='width mean')
    axes[2].set(title='Longitudinal transport envelope',xlim=(0,.75),xlabel='x (m)',ylabel='transport (mm2/s)');axes[2].legend()
    fig.savefig(REPORT/'TRANSPORT_VARIABILITY_240S.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
