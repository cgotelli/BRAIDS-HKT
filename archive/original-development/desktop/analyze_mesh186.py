"""Matched mesh/backend differences and raw inlet stripes at a common time."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
from stripes_mesh186 import ROOT,REPORT,frame,map_field
from braids_desktop.results import ResultsStore
from stage_secondary185 import save,sha
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ROOT/'validation/_mesh186'


def coordinates(f):
    return np.array([p.centroid.coords[0] for p in f.polygons]).T


def norms(d,area,xy):
    k=int(np.argmax(abs(d)))
    return dict(max_abs_mm=float(abs(d[k])*1000),rms_mm=float(np.sqrt(np.mean(d*d))*1000),
                area_rms_mm=float(np.sqrt(np.average(d*d,weights=area))*1000),peak_xy=xy[:,k].tolist())


def analyze():
    REPORT.mkdir(parents=True,exist_ok=True)
    summary=dict(physics='Smart & Jaggi; full advanced bundle; gravitational bed-slope OFF',
                 initial='stage 1.6 m, zero velocity; independent 200 s hydraulic warm-ups',
                 scope='Fixed meshes; no dynamic remesh; 20 s sediment at MF5',pairs={},runs={})
    all_frames={}
    docs=[]
    for mesh in ('coarse','refined'):
        for backend in ('cpu','gpu'):
            tag=mesh+'_'+backend;folder=OUT/tag
            d=json.loads((folder/'project/braids_project.json').read_text())
            d['terrain']['dem_path']='identical_dem'
            d['model']['request']['execution']['backend']='cpu'
            d['model']={k:d['model'][k] for k in ('physics','request')}
            d['settings'].pop('grid');d['settings'].pop('levels')
            docs.append(d)
            store=ResultsStore(folder/'run');initial=store.read(0);warm=frame(folder/'run',200)
            frames=[frame(folder/'run',t) for t in range(200,221)]
            all_frames[tag]=(initial,warm,frames)
            xy=coordinates(initial);x,y=xy;area=initial.arrays['area'];inlet=(x<5)
            assert np.array_equal(initial.z,warm.z)
            assert all(f.epoch==initial.epoch for f in frames)
            row=dict(check=json.loads((folder/'CHECK.json').read_text()),inlet=[])
            for f in frames:
                dz=f.z-initial.z
                row['inlet'].append(dict(time=f.time_s,bed_min_mm=float(dz[inlet].min()*1000),
                     bed_max_mm=float(dz[inlet].max()*1000),
                     bed_area_rms_mm=float(np.sqrt(np.average(dz[inlet]**2,weights=area[inlet]))*1000),
                     transverse_speed_max_m_s=float(abs(f.q[2,inlet]/f.q[0,inlet]).max()),
                     secondary_intensity_max_m_s=float(abs(f.arrays['W'][inlet]/f.q[0,inlet]).max())))
            row['initial_straight_bed_departure_from_plane_mm']=float(abs(initial.z[x<9]-(1-.005*x[x<9])).max()*1000)
            summary['runs'][tag]=row
        a0,aw,af=all_frames[mesh+'_cpu'];b0,bw,bf=all_frames[mesh+'_gpu']
        for k in ('vertices','offsets','levels'):
            np.testing.assert_array_equal(a0.mesh[k],b0.mesh[k])
        for k in ('q','z','mass','W','rock'):
            np.testing.assert_array_equal(a0.arrays[k],b0.arrays[k])
        xy=coordinates(a0);area=a0.arrays['area'];inlet=xy[0]<5
        rows=[]
        for a,b in zip(af,bf):
            rows.append(dict(time_s=a.time_s,bed=norms(a.z-b.z,area,xy),
                 depth=norms(a.q[0]-b.q[0],area,xy),
                 inlet_bed=norms((a.z-b.z)[inlet],area[inlet],xy[:,inlet]),
                 inlet_depth=norms((a.q[0]-b.q[0])[inlet],area[inlet],xy[:,inlet])))
        summary['pairs'][mesh]=rows
    # Besides these explicitly removed mesh/backend/path keys the inputs match.
    assert all(d==docs[0] for d in docs[1:]),'Unexpected configuration difference'
    summary['matched_input_check']=True
    summary['runtime_sha256']=sha(ROOT/'validation/_secondary185/candidate/RUNTIME.json')
    save(REPORT/'MATCHED.json',summary)
    fig,axs=plt.subplots(2,2,figsize=(14,7),layout='constrained')
    for r,mesh in enumerate(('coarse','refined')):
        for c,backend in enumerate(('cpu','gpu')):
            initial,warm,fs=all_frames[mesh+'_'+backend];f=fs[-1]
            p=map_field(axs[r,c],f,1000*(f.z-initial.z),-120,120,f'{mesh.title()} {backend.upper()} | bed change at 220 s (mm)')
            fig.colorbar(p,ax=axs[r,c],shrink=.7,extend='both')
    fig.suptitle('Matched wet starts and unchanged physics: raw inlet cells, same colour scale')
    fig.savefig(REPORT/'MATCHED_INLET.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(17,7),layout='constrained')
    for r,mesh in enumerate(('coarse','refined')):
        initial,warm,fs=all_frames[mesh+'_cpu']
        for c,t in enumerate((201,205,220)):
            f=fs[t-200]
            p=map_field(axs[r,c],f,1000*(f.z-initial.z),-120,120,f'{mesh.title()} CPU | t={t} s')
    fig.colorbar(p,ax=axs.ravel().tolist(),shrink=.7,label='Bed change (mm)',extend='both')
    fig.suptitle('Onset and growth of the inlet stripes, without bed-slope physics changes')
    fig.savefig(REPORT/'STRIPE_ONSET.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for mesh,col in (('coarse','#0072B2'),('refined','#D55E00')):
        rs=summary['pairs'][mesh];ts=[r['time_s'] for r in rs]
        axs[0].semilogy(ts,[max(1e-10,r['bed']['max_abs_mm']) for r in rs],label=mesh,color=col)
        axs[1].semilogy(ts,[max(1e-10,r['depth']['max_abs_mm']) for r in rs],label=mesh,color=col)
        rs=summary['runs'][mesh+'_cpu']['inlet']
        axs[2].plot(ts,[r['bed_area_rms_mm'] for r in rs],label=mesh,color=col)
    for ax,title in zip(axs,('CPU/GPU maximum bed difference','CPU/GPU maximum depth difference','CPU inlet bed change, area RMS')):
        ax.set(title=title,xlabel='Physical time (s)',ylabel='mm');ax.grid(alpha=.2);ax.legend()
    fig.savefig(REPORT/'GROWTH.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(17,7),layout='constrained')
    for r,mesh in enumerate(('coarse','refined')):
        initial,warm,fs=all_frames[mesh+'_cpu'];xy=coordinates(initial)
        fields=[1000*(initial.z-(1-.005*xy[0])),warm.q[2]/warm.q[0],warm.arrays['W']/warm.q[0]]
        for c,(a,lo,hi,title) in enumerate(zip(fields,(-40,-.03,-.02),(40,.03,.02),
              ('Initial bed minus analytic plane (mm)','Cross-stream velocity before sediment (m/s)','Secondary intensity before sediment (m/s)'))):
            p=map_field(axs[r,c],warm,a,lo,hi,mesh.title()+' | '+title)
            fig.colorbar(p,ax=axs[r,c],shrink=.65,extend='both')
    fig.savefig(REPORT/'PRE_SEDIMENT.png',dpi=170);plt.close(fig)
    print(json.dumps({m:summary['pairs'][m][-1] for m in summary['pairs']},indent=2))


if __name__=='__main__':analyze()
