"""Close the wave attribution and one bounded reflecting-wall experiment."""
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from check_bedload161 import sha,save
from check_interface170 import cpu_solver,physical
from wall_pressure170 import analytic_pressure
from review_hydraulics162 import load,norm
from review_groin163 import budgets
from braids_desktop.results import ResultsStore
RAW=ROOT/'validation/_interface170'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_interface170'
NATIVE=ROOT/'validation/_hydraulics162/mobile/delft_mpm'


def probe_checks():
    result={};evidence={}
    for name in ('BRAIDS','FLOW_PRIMARY','REST'):
        paths=[ROOT/f'validation/_corner167/baseline/gpu/{name}.npz',
               RAW/f'probes/cpu/{name}.npz',RAW/f'probes/gpu/{name}.npz']
        old,cpu,gpu=[np.load(p) for p in paths]
        evidence.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
        wall=np.flatnonzero((old['neighbour']<0)&(old['bid']<0));expected=np.zeros_like(old['q'])
        changes=[]
        for f in wall:
            c=int(old['owner'][f]);n=old['normal'][:,f];h,m=old['local'][:2,f]
            before=float(np.dot(old['fl'][1:,f],n))
            # Bed quadrature is identical and excluded from the wall-Riemann comparison.
            trace=old['rec_trace'];q=old['q']
            corr=(.5*9.81*(q[0,c]**2-trace[0,f]**2) if old['rec_hydro_constant'][c] else
                  .5*9.81*(q[0,c]+trace[8,f])*(trace[3,f]-old['z'][c]))
            exact=analytic_pressure(h,m);after=exact+corr
            expected[1:,c]-=old['length'][f]/old['area'][c]*(after-before)*n
            np.testing.assert_allclose(gpu['fl'][1:,f],after*n,atol=2e-12,rtol=2e-12)
            assert gpu['fl'][0,f]==0 and gpu['water'][f]==0
            changes.append(dict(face=int(f),cell=c,xy=old['mid'][:,f].tolist(),h=float(h),
                normal_velocity=float(m/h) if h else 0.,old_pressure=before-corr,exact_pressure=exact))
        interior=old['neighbour']>=0
        np.testing.assert_array_equal(gpu['rec_trace'],old['rec_trace'])
        np.testing.assert_array_equal(gpu['local'],old['local'])
        np.testing.assert_array_equal(gpu['riemann'],old['riemann'])
        np.testing.assert_array_equal(gpu['fl'][:,interior],old['fl'][:,interior])
        np.testing.assert_allclose(gpu['rate']-old['rate'],expected,atol=4e-12,rtol=3e-12)
        np.testing.assert_allclose(cpu['rate'],gpu['rate'],atol=4e-12,rtol=3e-12)
        assert np.all(gpu['limiter_theta']==1) and np.all(cpu['limiter_theta']==1)
        result[name]=dict(cpu_gpu_rate_max=float(abs(cpu['rate']-gpu['rate']).max()),
            wall_only_update_max_error=float(abs(gpu['rate']-old['rate']-expected).max()),
            water_rate_change_max=float(abs(gpu['rate'][0]-old['rate'][0]).max()),
            internal_flux_exact=True,reconstruction_exact=True,augmented_waves_exact=True,
            walls=changes,rate_max=float(abs(gpu['rate']).max()))
        if name=='REST':
            inner=(old['xy'][:,0]>2)&(old['xy'][:,0]<38)
            result[name]['interior_rest_max']=float(abs(gpu['rate'][:,inner]).max())
            assert result[name]['interior_rest_max']<1e-12
    return result,evidence


def coupled():
    result={};data={};evidence={}
    for name,path in [('baseline',ROOT/'validation/_shoal165/groin/mpm_cpu/run'),
                      ('exact_wall',RAW/'runs/pressure_v2/cpu/run')]:
        a,receipt=load(path,NATIVE);data[name]=a;xy=a['xy'];region=(xy[:,0]>6)&(xy[:,0]<16)
        dz=a['bz'][-1]-a['bz'][0];ndz=a['nz'][-1]-a['nz'][0]
        result[name]=dict(depth40_m=float(a['bh'][8,472]),uv40_m_s=a['bu'][8,:,472].tolist(),
            velocity40_region=norm((a['bu'][8]-a['nu'][8])[:,region]),
            depth40_region=norm((a['bh'][8]-a['nh'][8])[region]),
            tip_scour60_mm=float(-dz[472]*1000),maximum_scour_mm=float(-dz.min()*1000),
            maximum_scour_xy=xy[dz.argmin()].tolist(),maximum_deposition_mm=float(dz.max()*1000),
            bed60_region=norm((dz-ndz)[region]),bed60_all=norm(dz-ndz),cpu=budgets(path),
            initial_bed_error_m=receipt['initial_bed_max_m'],initial_depth_error_m=receipt['initial_depth_max_m'],
            tip_series_mm=((a['bz'][0,472]-a['bz'][:,472])*1000).tolist(),times=a['store'].times.tolist(),
            jet_peak_y_at_tip=float(xy[np.flatnonzero(np.isclose(xy[:,0],8.25))[np.argmax(a['bu'][8,0,np.isclose(xy[:,0],8.25)])],1]))
    a=data['exact_wall'];b=data['baseline'];g=ResultsStore(RAW/'runs/pressure_v2/gpu/run')
    result['exact_wall']['gpu']=budgets(RAW/'runs/pressure_v2/gpu/run')
    result['cpu_gpu']={k:max(float(abs(f.arrays[k]-g.read(i).arrays[k]).max())
                            for i,f in enumerate(a['frames'])) for k in ('q','z','mass','water','bed_boundary')}
    assert result['cpu_gpu']['q']<1e-7 and result['cpu_gpu']['z']<1e-9
    result['candidate_change']={k:norm(a[k]-b[k]) for k in ('bh','bu','bz')}
    result['native']=dict(depth40_m=float(a['nh'][8,472]),uv40_m_s=a['nu'][8,:,472].tolist(),
        tip_scour60_mm=float((a['nz'][0,472]-a['nz'][-1,472])*1000),
        maximum_scour_mm=float((a['nz'][0]-a['nz'][-1]).max()*1000),
        maximum_deposition_mm=float((a['nz'][-1]-a['nz'][0]).max()*1000),
        executable_sha256=receipt['native_execution']['executable_sha256'])
    for backend in ('cpu','gpu'):
        folder=RAW/f'runs/pressure_v2/{backend}'
        for p in folder.rglob('*'):
            if p.is_file() and (p.suffix in ('.npz','.json','.py') or p.name=='case.data'):
                evidence[str(p.relative_to(ROOT))]=sha(p)
        status=json.loads((folder/'STATUS.json').read_text())['status']
        assert status['status']=='complete' and status['time_s']==60 and not status['rejected_trials']
        result['exact_wall'][backend+'_hydraulic_status']=status['hydraulic']
        receipt=json.loads((folder/'CANDIDATE.json').read_text())
        assert receipt['script_sha256']==sha(folder/'wall_pressure170.py')
    return result,data,evidence


def figures(waves,probes,result,data):
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    labels=['Central\nadvection','Central\npressure','Acoustic\nleft','Augmented\nmiddle','Acoustic\nright','Contact','Bed\nquadrature']
    ids=[0,1,2,3,4,5,7];x=np.arange(len(ids))
    for shift,key,color,label in [(-.18,'BRAIDS','#316da5','BRAIDS state'),(.18,'FLOW_PRIMARY','#c36420','Mapped FLOW state')]:
        d=waves['cases'][key];region=np.array(d['region_acceleration']);tip=next(r for r in d['cells'] if r['cell']==472)
        axs[0].bar(x+shift,region[ids,0],.36,color=color,label=label)
        axs[1].bar(x+shift,np.array(tip['acceleration_pieces'])[ids,0],.36,color=color,label=label)
    for ax in axs:
        ax.set_xticks(x,labels,rotation=30,ha='right',fontsize=8);ax.axhline(0,color='.3',lw=.7)
        ax.set_ylabel('Streamwise velocity acceleration (m/s²)');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    axs[0].set_title('18-cell corner region');axs[1].set_title('Leading tip cell (8.25, 3.25) m')
    fig.suptitle('Exact frozen-state wave accounting; no force terms removed')
    fig.savefig(OUT/'WAVE_PARTS.png',dpi=160);plt.close(fig)

    h=.5;fr=np.linspace(-2.5,2.,181);m=h*fr*np.sqrt(9.81*h)
    states=np.ascontiguousarray(np.vstack([np.full(len(fr),h),m,np.zeros(len(fr)),np.zeros(len(fr)),
                                          np.full(len(fr),h),-m,np.zeros(len(fr)),np.zeros(len(fr))]))
    output=cpu_solver(states);old=np.array([(physical(states[:4,i])+output[:3,i])[1] for i in range(len(fr))])
    exact=np.array([analytic_pressure(h,v) for v in m]);scale=.5*9.81*h*h
    fig,axs=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    axs[0].plot(fr,old/scale,label='Current augmented wall flux',color='#316da5')
    axs[0].plot(fr,exact/scale,label='Exact wall Riemann pressure',color='#c36420')
    axs[0].axhline(0,color='.3',lw=.7);axs[0].axvline(0,color='.5',lw=.7)
    axs[0].set(xlabel='Normal Froude number (positive toward wall)',ylabel='Wall flux / hydrostatic pressure',title='Flat-bed reflection, no fitting')
    axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
    w=[r for r in probes['BRAIDS']['walls'] if 7.5<r['xy'][0]<10.5 and r['xy'][1]==3.]
    for key,label,color in [('old_pressure','Current','#316da5'),('exact_pressure','Exact wall','#c36420')]:
        axs[1].plot([r['xy'][0] for r in w],[r[key] for r in w],'o-',label=label,color=color)
    axs[1].set(xlabel='x along groin top wall (m)',ylabel='Wall momentum flux (m³/s²)',title='Same BRAIDS state at 40 s')
    axs[1].set_ylim(bottom=0);axs[1].legend();axs[1].grid(alpha=.2)
    fig.savefig(OUT/'WALL_PRESSURE.png',dpi=160);plt.close(fig)

    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained');a=data['exact_wall'];b=data['baseline'];xy=a['xy']
    polygons=[np.array(p.exterior.coords) for p in a['frames'][0].polygons]
    maps=[(b['bz'][-1]-b['bz'][0])*1000,(a['bz'][-1]-a['bz'][0])*1000,(a['nz'][-1]-a['nz'][0])*1000]
    for ax,values,title in zip(axs[0],maps,['BRAIDS current','BRAIDS exact wall pressure','Unmodified FLOW Flood 2DH']):
        p=PolyCollection(polygons,array=values,cmap='RdBu',clim=(-50,50),edgecolors='none');ax.add_collection(p)
        ax.set(xlim=(6,16),ylim=(0,6),aspect='equal',xlabel='x (m)',ylabel='y (m)',title=title+' at 60 s')
        ax.plot(8.25,3.25,'kx',ms=6)
    fig.colorbar(p,ax=axs[0],label='Bed change (mm), negative = erosion',shrink=.8)
    mask=np.isclose(xy[:,0],8.25);ys=xy[mask,1]
    for values,label,color,style in [(b,'Current','#316da5','-'),(a,'Exact wall','#c36420','-')]:
        axs[1,0].plot(values['bu'][8,0,mask],ys,'o'+style,label=label,color=color,ms=3)
        axs[1,1].plot(values['bh'][8,mask],ys,'o'+style,label=label,color=color,ms=3)
        axs[1,2].plot(values['store'].times,(values['bz'][0,472]-values['bz'][:,472])*1000,label=label,color=color)
    axs[1,0].plot(a['nu'][8,0,mask],ys,'k.--',label='FLOW')
    axs[1,1].plot(a['nh'][8,mask],ys,'k.--',label='FLOW')
    axs[1,2].plot(a['store'].times,(a['nz'][0,472]-a['nz'][:,472])*1000,'k--',label='FLOW')
    axs[1,0].set(xlabel='Streamwise velocity (m/s)',ylabel='y (m)',title='x = 8.25 m, hydraulics at 40 s',xlim=(0,3),ylim=(3,8))
    axs[1,1].set(xlabel='Water depth (m)',ylabel='y (m)',title='Same hydraulic section',xlim=(0,.65),ylim=(3,8))
    axs[1,2].set(xlabel='Simulation time (s)',ylabel='Tip scour (mm)',title='20 s of bed evolution, MF1',xlim=(40,60),ylim=(0,50))
    for ax in axs[1]:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('One wall-pressure correction; identical inputs, MPM, mesh and boundaries')
    fig.savefig(OUT/'GROIN_COMPARISON.png',dpi=160);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    waves=json.loads((RAW/'v2/AUDIT.json').read_text());pressure=json.loads((RAW/'pressure_v3/AUDIT.json').read_text())
    probes,evidence=probe_checks();result,data,more=coupled();evidence.update(more)
    native={}
    with zipfile.ZipFile(ROOT/'resources/Delft3D-main.zip') as z:
        for name in ('uzd.f90','mom_fls.f90','mom_cyclic.f90','upwhu.f90'):
            members=[p for p in z.namelist() if p.endswith('/'+name)];assert len(members)==1
            p=ROOT/'resources'/members[0];assert p.read_bytes()==z.read(members[0])
            native[str(p.relative_to(ROOT))]=sha(p)
    for folder in ('v2','pressure_v3'):
        for p in (RAW/folder).iterdir():
            if p.is_file():evidence[str(p.relative_to(ROOT))]=sha(p)
    evidence.update(waves['inputs'])
    runtime=ROOT/'desktop/runtime';identity=json.loads((runtime/'RUNTIME.json').read_text())
    for name,digest in identity['files'].items():assert sha(runtime/name)==digest,name
    figures(waves,probes,result,data)
    audit=dict(scope='Wave attribution, exact wall-pressure prototype and one short coupled comparison',
        waves=waves,pressure_controls=pressure,probes=probes,coupled=result,inputs=evidence,
        native_sources_read_only=native,runtime_manifest_sha256=sha(runtime/'RUNTIME.json'),
        production_changed=False,Delft3D_modified=False,groin_improvement_gate=False,
        interior_fwave_changed=False,cross_advection_suppressed=False,sediment_changed=False,
        scripts={p.name:sha(p) for p in (ROOT/'desktop').glob('*170.py')})
    save(OUT/'AUDIT.json',audit)
    print(json.dumps(dict(probes={k:{q:v for q,v in r.items() if q!='walls'} for k,r in probes.items()},coupled=result),indent=2))


if __name__=='__main__':main()
