"""Measured bank qualification and raw polygon comparison figures."""
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon
from stage_banks224 import ROOT,OUT,TARGET,REPORT,save,sha


def qualification():
    REPORT.mkdir(parents=True,exist_ok=True);root=OUT/'release_checks'
    runs={}
    for p in root.rglob('IDENTITY.json'):
        d=json.loads(p.read_text());runs[d['bed'],d['sus'],d['backend']]=p.parent
    assert len(runs)==142
    paired=[]
    for bed,sus,backend in runs:
        if backend!='cpu':continue
        a,b=(np.load(runs[bed,sus,k]/'FIELDS.npz') for k in ('cpu','gpu'))
        dz=(a['z']-b['z'])*1000
        paired.append(dict(bed=bed,suspension=sus,max_bed_mm=float(abs(dz).max()),
            rms_bed_mm=float(np.sqrt(np.mean(dz**2))),max_bank_kg=float(abs(a['bank']-b['bank']).max())))
    # Exclude the failed pre-fix retained-block AMR screen from final metrics.
    qualified_roots=[root/f'part_{i:02d}' for i in range(1,16)]+[OUT/'packaged_checks']
    temporal=[json.loads(p.read_text()) for r in qualified_roots for p in r.rglob('TEMPORAL.json')]
    checks=[json.loads(p.read_text()) for r in qualified_roots for p in r.rglob('METRICS.json')]
    adaptive={};adaptive_pairs=[]
    for r in (root/'part_15',OUT/'packaged_checks/part_16'):
        for p in r.glob('test_adaptive*/**/project.json'):
            if not (p.parent/'FIELDS.npz').exists():continue
            d=json.loads(p.read_text());model=d['model']
            adaptive[model['physics']['sediment']['banks'],p.parent.name,model['request']['execution']['backend']]=p.parent
    for method,case,backend in adaptive:
        if backend!='cpu':continue
        with np.load(adaptive[method,case,'cpu']/'FIELDS.npz') as a,np.load(adaptive[method,case,'gpu']/'FIELDS.npz') as b:
            maximum=float(np.max(np.abs(a['z']-b['z']))*1000)
            assert maximum<1.
            if 'bank_failure_count' in a:np.testing.assert_array_equal(a['bank_failure_count'],b['bank_failure_count'])
            adaptive_pairs.append(dict(method=method,case=case,max_bed_mm=maximum,
                retained_event_counts_exact='bank_failure_count' in a))
    assert len(adaptive_pairs)==6
    source=ROOT/'resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute_sediment/bott3d.f90'
    save(OUT/'METRICS.json',dict(runtime=sha(TARGET/'RUNTIME.json'),paired=paired,temporal=temporal,adaptive_pairs=adaptive_pairs,
        max_water_residual_m3=max(r['water'] for r in checks),max_sediment_residual_kg=max(r['sediment'] for r in checks),
        max_local_transfer_residual_kg=max(r['local'] for r in checks),
        exact_restart_cases=sum(r.get('restart_exact',False) for r in checks),
        active_block_cases=sum(r['block_count']>0 for r in checks),
        Delft3D_sources_read_only={str(source.relative_to(ROOT)):sha(source)},
        execution_provenance='QUALIFICATION_PARTS.json records final-package reruns and unchanged-operator carryover after the additive CPU bank_counts packaging fix.'))
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    ax[0].plot(np.arange(1,72),[max(1e-12,p['max_bed_mm']) for p in paired],'.')
    ax[0].set(yscale='log',xlabel='Compatible transport configuration',ylabel='Maximum bed difference (mm)',title='CPU versus GPU: 71 transport configurations')
    ax[0].text(.02,.92,'Plot floor: 1e-12 mm',transform=ax[0].transAxes,fontsize=8)
    ax[1].bar(['CPU','GPU'],[p['max_bed_mm'] for p in temporal]);ax[1].set(ylabel='Maximum bed difference (mm)',title='Timestep halved: retained blocks + repose')
    fig.savefig(REPORT/'QUALIFICATION.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained');count=0
    for path in root.rglob('test_first*/project.json'):
        d=json.loads(path.read_text())
        if d['model']['physics']['sediment']['banks']!='bank.braids_retained_undercut_block_collapse':continue
        backend=d['model']['request']['execution']['backend']
        index=json.loads((path.parent/'run/frames/INDEX.json').read_text());time=[];events=[];mass=[]
        for row in index['frames']:
            with np.load(path.parent/'run/frames'/row['file']) as f:
                time.append(float(f['time']));events.append(int(f['bank_failure_count'].sum()))
                mass.append(float(np.maximum(f['bank_failure_class_cumulative'],0).sum()))
        assert events[-1]>0
        for ax,value in zip(axs,(events,mass)):ax.plot(time,value,'o-' if backend=='cpu' else 'x--',label=backend.upper())
        count+=1
    assert count==2
    axs[0].set(ylabel='Retained block events',xlabel='Hydraulic time (s)')
    axs[1].set(ylabel='Transferred block mass (kg)',xlabel='Hydraulic time (s)')
    for ax in axs:ax.legend()
    fig.suptitle('Forced-trigger numerical control: threshold 1e-8 m, MF1; not a calibrated bank law')
    fig.savefig(REPORT/'RETAINED_BLOCK_TEST.png',dpi=170);plt.close(fig)

def examples():
    REPORT.mkdir(parents=True,exist_ok=True)
    data={}
    for case in sorted((ROOT/'desktop/examples/banks224').iterdir()):
        if not case.is_dir():continue
        ix=json.loads((case/'runs/example/frames/INDEX.json').read_text());frames=[]
        for row in ix['frames']:
            with np.load(case/'runs/example/frames'/row['file']) as f:frames.append({k:f[k].copy() for k in f.files})
        mesh=np.load(case/'runs/example/frames'/ix['frames'][-1]['epoch']);v=mesh['vertices'];o=mesh['offsets']
        polys=[v[o[i]:o[i+1]] for i in range(len(o)-1)]
        xy=np.array([[Polygon(p).centroid.x,Polygon(p).centroid.y] for p in polys])
        data[case.name]=(frames,polys,xy)
    reference=data['off']
    for name,(f,p,x) in data.items():
        for k in ('q','z','mass','suspended','rock','area'):np.testing.assert_array_equal(f[0][k],reference[0][0][k],err_msg=name+': '+k)
        assert len(p)==len(reference[1])
        for a,b in zip(p,reference[1]):np.testing.assert_array_equal(a,b)
    for filename,names in [('BANK_METHODS',['off','sharing','toe','block']),
                           ('COLLAPSE_COMPOSITION',['block','block_equalization','block_repose','gpu_block_repose'])]:
        limit=max(abs(data[name][0][-1]['z']-data[name][0][0]['z']).max() for name in names)*1000
        fig,axs=plt.subplots(2,2,figsize=(12,6.5),layout='constrained')
        for ax,name in zip(axs.flat,names):
            f,p,xy=data[name];val=(f[-1]['z']-f[0]['z'])*1000
            pc=PolyCollection(p,array=val,cmap='RdBu_r',clim=(-limit,limit),edgecolors='0.25',linewidths=.3)
            ax.add_collection(pc);ax.autoscale_view();ax.set_aspect('equal');ax.set(title=name.replace('_',' '),xlabel='x (m)',ylabel='y (m)')
            fig.colorbar(pc,ax=ax,label='Bed change (mm)')
        fig.suptitle('Matched initial fields and cropped mesh; 0.4 s hydraulic time, MF25')
        fig.savefig(REPORT/(filename+'.png'),dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,3.4),layout='constrained')
    names=['sharing','toe','block']
    limit=max(abs(data[name][0][-1]['z']-reference[0][-1]['z']).max() for name in names)*1000
    for ax,name in zip(axs,names):
        f,p,xy=data[name];val=(f[-1]['z']-reference[0][-1]['z'])*1000
        pc=PolyCollection(p,array=val,cmap='RdBu_r',clim=(-limit,limit),edgecolors='0.25',linewidths=.3)
        ax.add_collection(pc);ax.autoscale_view();ax.set_aspect('equal');ax.set(title=name,xlabel='x (m)',ylabel='y (m)')
        fig.colorbar(pc,ax=ax,label='Bed difference from bank OFF (mm)')
    fig.suptitle('Bank-process contribution; same initial condition, transport and MF25')
    fig.savefig(REPORT/'BANK_EFFECT.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for name,(f,p,xy) in data.items():
        t=[float(r['time']) for r in f]
        axs[0,0].plot(t,[float(r['bank_gross'].sum()) for r in f],'.-',label=name)
        axs[0,1].plot(t,[float(r['collapse_gross'].sum()) for r in f],'.-',label=name)
        take=np.abs(xy[:,0]-1.7)<.09;order=np.where(take)[0][np.argsort(xy[take,1])]
        axs[1,0].plot(xy[order,1],f[-1]['z'][order],'.-',label=name)
        axs[1,1].plot(t,[int(r.get('bank_failure_count',np.zeros(1)).sum()) for r in f],'.-',label=name)
    axs[1,0].plot(xy[order,1],reference[0][0]['z'][order],'k--',label='Initial bed')
    for ax,title,ylabel in zip(axs.flat,['Gross bank supply','Gross collapse redistribution','Raw bank section near x = 1.7 m','Retained block events'],['kg','kg','Elevation (m)','Count']):
        ax.set(title=title,ylabel=ylabel);ax.legend(fontsize=6,ncol=2)
    for ax in (axs[0,0],axs[0,1],axs[1,1]):ax.set_xlabel('Hydraulic time (s)')
    axs[0,0].set_ylim(bottom=0.);axs[0,1].set_ylim(bottom=0.)
    maximum_events=max(int(f[-1].get('bank_failure_count',np.zeros(1)).sum()) for f,p,x in data.values())
    axs[1,1].set_ylim(0,max(1,1.1*maximum_events))
    from matplotlib.ticker import MaxNLocator
    axs[1,1].yaxis.set_major_locator(MaxNLocator(integer=True))
    if maximum_events==0:
        axs[1,1].legend().remove()
        axs[1,1].text(.05,.8,'No block release in this 0.4 s window\nThreshold = 0.0001 m',transform=axs[1,1].transAxes)
    axs[1,0].set_xlabel('y (m)')
    fig.savefig(REPORT/'BANK_HISTORY.png',dpi=170);plt.close(fig)
    a=data['block_repose'][0][-1];b=data['gpu_block_repose'][0][-1]
    save(OUT/'EXAMPLE_COMPARISON.json',dict(initial_fields_and_mesh_exact=True,
        cpu_gpu_max_bed_mm=float(abs(a['z']-b['z']).max()*1000),
        cases={name:dict(min_change_mm=float((f[-1]['z']-f[0]['z']).min()*1000),
            max_change_mm=float((f[-1]['z']-f[0]['z']).max()*1000),
            bank_kg=float(f[-1]['bank_gross'].sum()),collapse_kg=float(f[-1]['collapse_gross'].sum()),
            block_count=int(f[-1].get('bank_failure_count',np.zeros(1)).sum())) for name,(f,p,x) in data.items()},
        scope='Short synthetic closed bank, not a physical river benchmark or new FLOW trajectory.'))
    print('Bank figures and measured qualification metrics written',flush=True)

if __name__=='__main__':
    if '--figures-only' not in sys.argv:qualification()
    examples()
