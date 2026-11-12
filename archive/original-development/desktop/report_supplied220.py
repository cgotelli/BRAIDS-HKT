"""Numerical figures and release evidence for supplied suspension."""
import os,sys,json
from pathlib import Path
from stage_supplied220 import ROOT,OUT,TARGET,REPORT,sha,save
sys.path[:0]=[str(TARGET),str(ROOT),str(ROOT/'tests')]
os.environ['QT_QPA_PLATFORM']='offscreen';os.environ['OMP_NUM_THREADS']='4'
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
verify();install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from test_supplied220 import pulse
from test_suspended_core215 import settling_speed

def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    results=OUT/'packaged_final';rows=[];temporal=[];analytical=[];pulses=[]
    for source in results.rglob('project.json'):
        if not source.parent.name.startswith('test_') and source.parent.parent==results:continue
        # Select the new suite, not retained S2 cases using the same method.
        if source.parents[0]==results:continue
        top=source.relative_to(results).parts[0]
        if not top.startswith(('test_worker','test_other_bedload','test_temporal','test_advanced','test_closed_settling')):continue
        doc=json.loads(source.read_text());s=doc.get('model',{}).get('physics',{}).get('sediment') or {}
        if s.get('suspension')!='suspension.settling_only':continue
        path=source.parent/'run/DIAGNOSTICS.json'
        if not path.exists():continue
        diag=json.loads(path.read_text())
        rows.append(dict(case=str(source.parent.relative_to(results)),
            water=max(abs(r['water_residual_m3']) for r in diag),
            sediment=max(abs(x) for r in diag for x in r['class_residual_kg']),
            minimum_stock=min(r['minimum_stock'] for r in diag),
            minimum_suspended=min(r['minimum_suspension'] for r in diag),
            wall_mass=max(abs(r['wall_integrals'][0]) for r in diag),final_time=diag[-1]['time_s']))
    for p in results.glob('*/TEMPORAL.json'):
        if not p.parent.name.startswith('test_temporal_and_zero_erodibl'):continue
        doc=json.loads((p.parent/'0/project.json').read_text())
        if (doc['model']['physics']['sediment'] or {}).get('suspension')=='suspension.settling_only':
            temporal.append(dict(case=p.parent.name,errors=json.loads(p.read_text())))
    for p in results.glob('*/ANALYTICAL.json'):analytical.append(json.loads(p.read_text()))
    for p in results.glob('*/PULSE.json'):pulses.append(json.loads(p.read_text()))
    assert len(rows)==48,len(rows)
    assert len(analytical)==6 and len(pulses)==2 and len(temporal)==2
    summary=dict(cases=len(rows),water=max(r['water'] for r in rows),sediment=max(r['sediment'] for r in rows),
        minimum_stock=min(r['minimum_stock'] for r in rows),minimum_suspended=min(r['minimum_suspended'] for r in rows),
        wall_mass=max(r['wall_mass'] for r in rows))
    save(OUT/'QUALIFICATION_METRICS.json',dict(runtime=sha(TARGET/'RUNTIME.json'),summary=summary,
        temporal=temporal,analytical=analytical,pulses=pulses,cases=rows))
    a=pulse('cpu',240);b=pulse('gpu',240)
    parity={k:float(np.max(abs(a[k]-b[k]))) for k in ('mass','deposited')}
    for k in parity:assert parity[k]<1e-12
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for t,m,d in a['rows']:ax[0,0].plot(a['x'],m,label=f'{t:g} s')
    ax[0,0].set(title='Supplied pulse evolves and settles',xlabel='Distance (m)',ylabel='Suspended mass (kg/m²)',xlim=(0,4))
    ax[0,0].legend(ncol=2,fontsize=8)
    ax[0,1].plot(a['x'],a['exact'],'k--',label='Analytical characteristics')
    ax[0,1].plot(a['x'],a['mass'],label='BRAIDS CPU')
    ax[0,1].plot(b['x'][::6],b['mass'][::6],'.',label='BRAIDS GPU')
    ax[0,1].set(title='At 6 s: first-order advection broadens the pulse',xlabel='Distance (m)',ylabel='Suspended mass (kg/m²)',xlim=(0,4));ax[0,1].legend(fontsize=8)
    ax[1,0].plot(a['x'],a['deposited'],label='CPU deposited')
    ax[1,0].plot(b['x'][::6],b['deposited'][::6],'.',label='GPU deposited')
    ax[1,0].set(title='Initially non-erodible bed receives the settling pulse',xlabel='Distance (m)',ylabel='Deposited mass (kg/m²)',xlim=(0,4));ax[1,0].legend(fontsize=8)
    inventories=[a[k] for k in ('initial_inventory','final_inventory','bed_inventory','exported')]
    ax[1,1].bar(['Initial','Suspended','Deposited','Exported'],inventories,color=['gray','steelblue','sienna','seagreen'])
    ax[1,1].set(title=f'Suspended + deposited + exported = initial\nResidual {a["residual"]:.2e} kg',ylabel='Mass (kg)')
    for axis in ax.flat:axis.grid(alpha=.2);axis.set_ylim(bottom=0)
    fig.suptitle('Supplied suspension control • Ds 0.1 mm • h 0.5 m • u 0.2 m/s • MF1\nFrozen-hydraulics operator check, not an experimental river',fontsize=12)
    fig.savefig(REPORT/'SUPPLIED_PULSE.png',dpi=160);plt.close(fig)
    # Actual coupled worker outputs, not only the isolated exponential kernel.
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    t=np.linspace(0,2,201);ws=settling_speed(.0001,1.65,20.);mass=.03*np.exp(-ws*t/.15)
    ax[0].plot(t,mass,'k--',label='Analytical suspended');ax[1].plot(t,.03-mass,'k--',label='Analytical deposited')
    for backend in ('cpu','gpu'):
        folder=next(results.glob('test_closed_settling_column_exa*'+backend+'*'),None)
        if folder is None:
            folder=next(p.parent for p in results.glob('*/ANALYTICAL.json') if json.loads(p.read_text())['backend']==backend)
        index=json.loads((folder/'run/frames/INDEX.json').read_text());tt=[];mm=[];bb=[]
        for row in index['frames']:
            with np.load(folder/'run/frames'/row['file']) as f:
                tt.append(float(f['time'] if 'time' in f else row.get('time_s',row.get('time'))))
                mm.append(float(np.mean(f['suspended'])));bb.append(float(np.mean(f['mass'].sum(axis=0))))
        ax[0].plot(tt,mm,'.',label=backend.upper());ax[1].plot(tt,bb,'.',label=backend.upper())
    for axis,title in zip(ax,('Actual Desktop worker: suspended inventory','Actual Desktop worker: deposited inventory')):
        axis.set(title=title,xlabel='Time (s)',ylabel='Mass per unit area (kg/m²)');axis.legend();axis.grid(alpha=.2);axis.set_ylim(bottom=0)
    fig.savefig(REPORT/'SETTLING_COLUMN.png',dpi=160);plt.close(fig)
    save(OUT/'FIGURE_METRICS.json',dict(runtime=sha(TARGET/'RUNTIME.json'),pulse_backend_max=parity,
        pulse_rms=a['rms'],pulse_budget=a['residual'],column_max=max(r['maximum_error'] for r in analytical)))
    source=ROOT/'resources/Delft3D-main/src/utils_gpl/morphology/packages/morphology_kernel/src/eqsettle.f90'
    save(OUT/'SOURCES.json',dict(runtime=sha(TARGET/'RUNTIME.json'),reference_source=str(source.relative_to(ROOT)),
        sha256=sha(source),lines='152-184',read_only=True,scope='SettleFrm2 clear-water law only; not FLOW Galappatti exchange',
        kernel_sha256=sha(TARGET/'braids_core/sediment/river_settling_only.h')))
    print(json.dumps(dict(summary=summary,analytical=analytical,pulses=pulses,parity=parity),indent=2))

if __name__=='__main__':main()
