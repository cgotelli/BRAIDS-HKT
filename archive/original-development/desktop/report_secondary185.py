"""Compact qualification receipts and raw-cell scientific figures."""
import json,sys
from stage_secondary185 import ROOT,TARGET,OUT,save,sha
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from braids_desktop.results import ResultsStore
from compare_secondary185 import compare

REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_secondary185'
def read(path):return json.loads(path.read_text())
def last(path):return ResultsStore(path).read(-1)
def panel(ax,frame,values,*,title,cmap='viridis',limit=None):
    p=PolyCollection([np.asarray(p.exterior.coords) for p in frame.polygons],array=values,
                     cmap=cmap,edgecolors='none',rasterized=True)
    if limit:p.set_clim(*limit)
    ax.add_collection(p);ax.autoscale_view();ax.set_aspect('equal');ax.set_title(title)
    ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');plt.colorbar(p,ax=ax,shrink=.8)

def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    rows=read(OUT/'COMPARE_v6.json')
    rows['runtime']=sha(TARGET/'RUNTIME.json')
    rows['off_regression']={b:compare(ROOT/f'validation/_smart183/installed/smart_jaggi_{b}/full',
                                   OUT/f'smoke/release_v6/smart_jaggi_{b}/full') for b in ('cpu','gpu')}
    for r in rows['off_regression'].values():
        assert all(v['max_abs']==0 for k,v in r['fields'].items() if k!='depth')
    rows['amr']={}
    for b in ('cpu','gpu'):
        folder=OUT/f'smoke/amr_secondary_v6/smart_jaggi_{b}'
        check=read(folder/'CHECK.json');index=read(folder/'full/frames/INDEX.json')
        rows['amr'][b]=dict(status=check['status']['status'],bed_min_m=check['bed_min_m'],bed_max_m=check['bed_max_m'],
            water_residual_m3=check['water_residual_m3'],sediment_residual_kg=check['sediment_residual_kg'],
            exact_restart=check['restart']['exact'],mesh_epochs=len({r['epoch'] for r in index['frames']}))
        assert rows['amr'][b]['mesh_epochs']>1
    rows['amr_backend']=compare(OUT/'smoke/amr_secondary_v6/smart_jaggi_cpu/full',OUT/'smoke/amr_secondary_v6/smart_jaggi_gpu/full')
    rows['continuation']={}
    for b in ('cpu','gpu'):
        check=read(OUT/f'qualified_{b}_v6/CHECK.json')
        rows['continuation'][b]={k:check[k] for k in ('diagnostics','extrema','restart_exact','runtime')}
        rows['continuation'][b]['time_s']=check['status']['time_s']
        rows['continuation'][b]['status']=check['status']['status']
        assert check['runtime']==rows['runtime']
    rows['coupled_backend']=compare(OUT/'qualified_cpu_v6/to_220',OUT/'qualified_gpu_v6/to_220')
    save(REPORT/'CHECKS.json',rows)
    baseline=read(OUT/'original/TRACE.json');trace=read(OUT/'signed_cpu_v6/TRACE.json')
    f,ax=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for name,t in [('Original (includes rejected trials)',baseline),('Corrected CPU',trace)]:
        ax[0,0].semilogy([r['time_s']+r['dt_s'] for r in t],[max(1e-15,r['W_trial_max']) for r in t],label=name)
    ax[0,0].set(xlabel='Time (s)',ylabel='Maximum |h I| (m²/s)',title='Original instability and corrected trajectory')
    ax[0,0].legend();ax[0,0].grid(alpha=.2)
    x=np.linspace(-.2,.2,1001);native=np.where(abs(x)>.1,.1,x)
    ax[0,1].plot(x,native,label='Literal FLOW cap');ax[0,1].plot(x,np.clip(x,-.1,.1),label='BRAIDS sign-preserving cap')
    ax[0,1].set(xlabel='Signed I / speed',ylabel='beta / beta_st',title='Momentum limiter at negative saturation')
    ax[0,1].legend();ax[0,1].grid(alpha=.2)
    a=last(OUT/'signed_cpu_v6/run');b=last(OUT/'signed_half_cpu_v6/run')
    panel(ax[1,0],a,1000*abs(a.q[0]-b.q[0]),title='Halved outer timestep: depth difference at 20 s (mm)',cmap='magma')
    ax[1,1].plot([r['time_s']+r['dt_s'] for r in trace],[r['accepted_secondary_CFL'] for r in trace])
    ax[1,1].axhline(.8,color='red',linestyle='--',label='Selected bound 0.8')
    ax[1,1].set(xlabel='Time (s)',ylabel='Maximum accepted scalar CFL',title='Advection and diffusion substep bound',ylim=(0,.85))
    ax[1,1].legend();f.savefig(REPORT/'SECONDARY_STABILITY.png',dpi=170);plt.close(f)
    cpu=last(OUT/'qualified_cpu_v6/to_220');gpu=last(OUT/'qualified_gpu_v6/to_220')
    first=ResultsStore(OUT/'signed_cpu_v6/run').read(0)
    fields=[frame.z-first.z for frame in (cpu,gpu)];m=max(abs(v).max() for v in fields)*1000
    wi=max(abs(frame.arrays['W']).max() for frame in (cpu,gpu))
    f,ax=plt.subplots(2,3,figsize=(16,7),layout='constrained')
    for i,(label,frame) in enumerate((('CPU',cpu),('GPU',gpu))):
        panel(ax[i,0],frame,frame.q[0],title=label+' depth at 220 s (m)',cmap='Blues',limit=(0,max(cpu.q[0].max(),gpu.q[0].max())))
        panel(ax[i,1],frame,frame.arrays['W'],title=label+' signed hI at 220 s (m²/s)',cmap='coolwarm',limit=(-wi,wi))
        panel(ax[i,2],frame,1000*fields[i],title=label+' bed change after 20 s of bedload, MF5 (mm)',cmap='RdBu_r',limit=(-m,m))
    f.savefig(REPORT/'REFINED_CASE.png',dpi=170);plt.close(f)
    f,ax=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    panel(ax[0,0],cpu,1000*(gpu.q[0]-cpu.q[0]),title='GPU minus CPU depth at 220 s (mm)',cmap='coolwarm',limit=(-11,11))
    panel(ax[0,1],cpu,1000*(gpu.z-cpu.z),title='GPU minus CPU bed at 220 s (mm)',cmap='coolwarm',limit=(-11,11))
    times=[];depth=[];bed=[];rms=[]
    cs=ResultsStore(OUT/'qualified_cpu_v6/to_220');gs=ResultsStore(OUT/'qualified_gpu_v6/to_220')
    for i in range(len(cs.times)):
        a,b=cs.read(i),gs.read(i);assert a.time_s==b.time_s
        times.append(a.time_s);depth.append(1000*abs(a.q[0]-b.q[0]).max())
        bed.append(1000*abs(a.z-b.z).max());rms.append(1000*np.sqrt(np.mean((a.z-b.z)**2)))
    ax[1,0].plot(times,depth,label='Depth maximum');ax[1,0].plot(times,bed,label='Bed maximum')
    ax[1,0].plot(times,rms,label='Bed RMS');ax[1,0].axvline(200,color='gray',ls='--',label='Sediment activation')
    ax[1,0].set(xlabel='Time (s)',ylabel='CPU/GPU difference (mm)',title='Warm-up and first sediment interval')
    ax[1,0].legend();ax[1,0].grid(alpha=.2)
    panel(ax[1,1],cpu,1000*(gpu.z-cpu.z),title='Inlet-bank detail, same limits (mm)',cmap='coolwarm',limit=(-11,11))
    ax[1,1].set_xlim(0,5);ax[1,1].set_ylim(-2.1,2.1)
    f.savefig(REPORT/'BACKEND_DIFFERENCE.png',dpi=170);plt.close(f)
    xy=np.array([p.centroid.coords[0] for p in cpu.polygons]);d=abs(cpu.z-gpu.z);areas=np.array([p.area for p in cpu.polygons])
    rows['coupled_spatial_audit']=dict(bed_peak_xy=xy[int(np.argmax(d))].tolist(),
        cells_above_1mm=int((d>.001).sum()),total_cells=len(d),bed_area_weighted_rms_m=float(np.sqrt(np.sum(d*d*areas)/sum(areas))),
        bent_reach_max_m=float(d[(xy[:,0]>=10)&(xy[:,0]<=45)].max()))
    rows['coupled_evolution']=dict(time_s=times,depth_max_mm=depth,bed_max_mm=bed,bed_rms_mm=rms)
    save(REPORT/'CHECKS.json',rows)
    print(json.dumps({k:rows[k] for k in ('backend','half_step','coupled_backend','amr_backend')},indent=2))
if __name__=='__main__':main()
