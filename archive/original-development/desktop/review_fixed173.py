"""Fixed-bed time-history check separating upstream transients from corner transport."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from review_hydraulics162 import load
from review_jet173 import shape,rms,OUT
from stage_wall172 import save,sha
from braids_desktop.results import ResultsStore


def main():
    base=ROOT/'validation/_jet173/fixed'
    d,meta=load(base/'cpu/run',base/'delft');g=ResultsStore(base/'gpu/run')
    assert np.array_equal(g.times,d['store'].times)
    assert abs(d['bz']-d['bz'][0]).max()==0 and abs(d['nz']-d['nz'][0]).max()==0
    qdiff=max(float(abs(f.q-g.read(j).q).max()) for j,f in enumerate(d['frames']))
    depth_difference=max(float(abs(f.q[0]-g.read(j).q[0]).max()) for j,f in enumerate(d['frames']))
    velocity_difference=max(float(abs(f.q[1:]/f.q[0]-g.read(j).q[1:]/g.read(j).q[0]).max()) for j,f in enumerate(d['frames']))
    # Retain the failed mixed-units 1e-6 screen explicitly. The user's physical
    # millimetre allowance, not bitwise coupled agreement, governs this control.
    assert depth_difference<.001,depth_difference
    xy=d['xy'];m=xy[:,0]==4.25;y=xy[m,1];tip=int(np.flatnonzero(np.all(xy==[8.25,3.25],axis=1))[0])
    details=[];series=[]
    for j,t in enumerate(d['store'].times):
        b,n=d['bu'][j,1,m],d['nu'][j,1,m]
        s=dict(time_s=float(t),braids_peak=float(b.max()),flow_peak=float(n.max()),
               v_rms_difference=rms(b-n),tip_braids_u=float(d['bu'][j,0,tip]),tip_flow_u=float(d['nu'][j,0,tip]))
        series.append(s)
        if t in (40,60,80,100,120):
            details.append(dict(**s,BRAIDS=shape(b,y),FLOW=shape(n,y),braids_v=b.tolist(),flow_v=n.tolist()))
    late=np.asarray(d['store'].times)>=80
    bmean=d['bu'][late,1][:,m].mean(0);nmean=d['nu'][late,1][:,m].mean(0)
    noise=dict(window_s=[80,120],braids_fluctuation_rms=rms(d['bu'][late,1][:,m]-bmean),
               flow_fluctuation_rms=rms(d['nu'][late,1][:,m]-nmean),mean_profile_rms_difference=rms(bmean-nmean))
    before,_=load(ROOT/'validation/_delivery172/package/groin_mpm_cpu/full',ROOT/'validation/_hydraulics162/mobile/delft_mpm')
    histories=dict(braids_q_change_at40=float(abs(d['frames'][40].q-before['frames'][8].q).max()),
                   braids_depth_change_at40_m=float(abs(d['frames'][40].q[0]-before['frames'][8].q[0]).max()),
                   native_q_change_at40=float(abs(d['nu'][40]-before['nu'][8]).max()))
    histories['strict_q_1e_9_pass']=histories['braids_q_change_at40']<1e-9
    assert histories['braids_depth_change_at40_m']<1e-5 and histories['native_q_change_at40']==0
    budget={}
    for backend in ('cpu','gpu'):
        rows=json.loads((base/backend/'run/DIAGNOSTICS.json').read_text())
        budget[backend]=dict(water=max(abs(r['water_residual_m3']) for r in rows),
                             bed=max(abs(v) for r in rows for v in r['class_residual_kg']),
                             wall=max(abs(v) for r in rows for v in r['wall_integrals']))
        assert max(budget[backend].values())<1e-8
    # Peak location stays an explicitly separate gate.
    peak=[]
    for t in (40,80,120):
        for x in (8.25,10.25):
            mask=xy[:,0]==x
            peak.append(dict(time_s=t,x=x,braids_y=float(xy[mask,1][np.argmax(d['bu'][t,0,mask])]),
                             flow_y=float(xy[mask,1][np.argmax(d['nu'][t,0,mask])]),
                             braids_peak=float(d['bu'][t,0,mask].max()),flow_peak=float(d['nu'][t,0,mask].max())))
    sources={}
    folder=ROOT/'resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute'
    for name in ('mom_fls.f90','uzd.f90','upwhu.f90'):sources[str((folder/name).relative_to(ROOT))]=sha(folder/name)
    save(OUT/'FIXED_BED.json',dict(initial=meta,CPU_GPU_q_max=qdiff,CPU_GPU_depth_max_m=depth_difference,
        CPU_GPU_velocity_max_m_s=velocity_difference,strict_mixed_q_1e_6_pass=qdiff<1e-6,
        physical_depth_1mm_pass=depth_difference<.001,budget=budget,details=details,series=series,
        late_window=noise,retained40s=histories,jet_peaks=peak,bed_change_BRAIDS=0.,bed_change_FLOW=0.,
        native_sources_read_only=sources,runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in base.rglob('*') if p.is_file() and
                (p.suffix in ('.json','.nc') or p.parent.name=='frames' and p.suffix=='.npz')},
        upstream40s_symmetry_imposed=False,upstream_transient_identified=True,corner_jet_fix_qualified=False))
    fig,axs=plt.subplots(2,2,figsize=(12,7.8),layout='constrained')
    times=d['store'].times
    axs[0,0].plot(times,[r['braids_peak'] for r in series],label='BRAIDS CPU/GPU')
    axs[0,0].plot(times,[r['flow_peak'] for r in series],label='FLOW 2DH')
    axs[0,0].set(title='Section maximum transverse velocity',xlabel='Time (s)',ylabel='v (m/s)');axs[0,0].legend()
    for t,style in ((40,'--'),(120,'-')):
        axs[0,1].plot(y,d['bu'][t,1,m],style,color='#1f77b4',label=f'BRAIDS {t} s')
        axs[0,1].plot(y,d['nu'][t,1,m],style,color='#ff7f0e',label=f'FLOW {t} s')
    axs[0,1].set(title='x = 4.25 m profiles',xlabel='y (m)',ylabel='v (m/s)');axs[0,1].legend(fontsize=8)
    axs[1,0].plot(times,[r['v_rms_difference'] for r in series]);axs[1,0].set(title='Upstream profile difference',xlabel='Time (s)',ylabel='v RMS difference (m/s)')
    axs[1,1].plot(times,d['bu'][:,0,tip],label='BRAIDS')
    axs[1,1].plot(times,d['nu'][:,0,tip],label='FLOW');axs[1,1].set(title='Groin wall-side cell remains different',xlabel='Time (s)',ylabel='u at (8.25, 3.25) m (m/s)');axs[1,1].legend()
    for ax in axs.flat:ax.axhline(0,color='gray',lw=.6);ax.grid(alpha=.2)
    fig.suptitle('120 s fixed-bed control: no sediment or bed evolution before comparison')
    fig.savefig(OUT/'FIXED_BED_HISTORY.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(parity=qdiff,budget=budget,late=noise,details=[{k:r[k] for k in ('time_s','braids_peak','flow_peak','v_rms_difference')} for r in details],peaks=peak),indent=2))


if __name__=='__main__':main()
