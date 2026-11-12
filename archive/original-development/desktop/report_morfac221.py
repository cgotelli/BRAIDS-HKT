"""Recompute all comparisons from saved worker fields; no fabricated curves."""
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stage_morfac221 import ROOT,OUT,REPORT,TARGET,sha,save

def fields(path):
    with np.load(path) as f:return {k:f[k].copy() for k in f.files}

def records(folder,pattern):
    result={}
    for p in folder.glob(pattern+'/IDENTITY.json'):
        d=json.loads(p.read_text());key=(d['method'],d['backend'],d.get('case','open'))
        result[key]=(fields(p.parent/'FIELDS.npz'),p.parent)
    return result

def main():
    name=sys.argv[1] if len(sys.argv)>1 else 'qualification5'
    current=OUT/name;REPORT.mkdir(parents=True,exist_ok=True)
    baseline=records(OUT/'baseline2','test_mf1*');mf1=records(current,'test_mf1*')
    assert len(baseline)==len(mf1)==14
    for key,(a,_) in baseline.items():
        b=mf1[key][0]
        for field in a:
            assert a[field].dtype==b[field].dtype and a[field].shape==b[field].shape
            assert a[field].tobytes()==b[field].tobytes(),(key,field)
    run=records(current,'test_coupled*');assert len(run)==56
    pairs=[];max_budget=0.;max_water=0.;rejections=[]
    for (method,backend,case),(a,folder) in run.items():
        diag=json.loads((folder/'run/DIAGNOSTICS.json').read_text())
        max_budget=max(max_budget,max(abs(x) for d in diag for x in d['class_residual_kg']))
        max_water=max(max_water,max(abs(d['water_residual_m3']) for d in diag))
        status=json.loads((folder/'run/status.json').read_text())
        if status['rejected_trials']:rejections.append(dict(method=method,backend=backend,case=case,count=len(status['rejected_trials'])))
        if backend!='cpu':continue
        b=run[method,'gpu',case][0]
        errors={k:float(np.max(abs(a[k]-b[k]))) for k in ('q','z','mass','suspended','acceleration')}
        assert errors['z']<.0001 and errors['q']<.0001, (method,case,errors)
        pairs.append(dict(method=method,case=case,max_abs=errors))
    temporal=[json.loads(p.read_text()) for p in current.glob('test_timestep*/TIMESTEP.json')]
    assert len(temporal)==14
    for path in current.glob('test_timestep*/changed/PROJECT.json'):
        a=json.loads((path.parents[1]/'0.001/prepared/PROJECT.json').read_text())
        b=json.loads(path.read_text())
        assert a['settings']['controls']['morfac']==25 and b['settings']['controls']['morfac']==10
        b['settings']['controls']['morfac']=25
        assert a==b # The restart rejection is not due to another changed input.
    source_paths=[
      'src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute_sediment/'+n
      for n in ('bott3d.f90','erosed.f90','red_soursin.f90')]
    sources={p:sha(ROOT/'resources/Delft3D-main'/p) for p in source_paths}
    metrics=dict(runtime=sha(TARGET/'RUNTIME.json'),mf1_byte_equal_cases=14,paired_mf25=pairs,
        time_step_halving=temporal,max_budget_kg=max_budget,max_water_residual_m3=max_water,
        rejected_trials=rejections,Delft3D_sources_read_only=sources,
        scope='Numerical MF qualification, not experimental validation or a coupled FLOW run.')
    save(OUT/'METRICS.json',metrics)
    names=list(dict.fromkeys(r['method'] for r in pairs))
    fig,ax=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for i,method in enumerate(names):
        ax[0].plot(i,max(r['max_abs']['z'] for r in pairs if r['method']==method)*1000,'o')
        ax[1].plot(i,max(r['errors']['z'] for r in temporal if r['method']==method)*1000,'o')
    for a in ax:
        a.set_xticks(range(len(names)),[n.split('.')[-1].replace('_',' ') for n in names],rotation=60,ha='right')
        a.set_ylabel('Maximum bed difference (mm)');a.set_ylim(bottom=0);a.grid(alpha=.25)
    ax[0].set_title('MF25: CPU versus GPU\nwet, cropped, remeshing and finite-stock cases')
    ax[1].set_title('MF25: timestep 0.002 versus 0.001 s\nsuspension-only open-channel controls')
    fig.savefig(REPORT/'QUALIFICATION.png',dpi=170);plt.close(fig)
    example=ROOT/'desktop/examples/morfac221'
    rows=[];fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for backend,mf in [('cpu',1),('cpu',5),('cpu',25),('gpu',25)]:
        case=example/f'{backend}_MF{mf}';index=json.loads((case/'runs/example/frames/INDEX.json').read_text())
        t=[];bed=[];mobile=[];extra=[]
        for row in index['frames']:
            f=fields(case/'runs/example/frames'/row['file'])
            t.append(float(f['time']));bed.append(float((f['mass'].sum(0)*f['area']).sum()))
            mobile.append(float((f['suspended']*f['area']).sum()));extra.append(float(f['acceleration'].sum()))
        style='--' if backend=='gpu' else '-';label=f'MF {mf} {backend.upper()}'
        ax[0].plot(t,bed,style,label=label);ax[1].plot(t,mobile,style,label=label);ax[2].plot(t,extra,style,label=label)
        rows.append(dict(backend=backend,mf=mf,bed_kg=bed[-1],suspended_kg=mobile[-1],acceleration_kg=extra[-1]))
    for a,title in zip(ax,('Bed sediment','Suspended sediment','Explicit MF acceleration ledger')):
        a.set_title(title);a.set_xlabel('Hydraulic time (s)');a.set_ylabel('Mass (kg)');a.grid(alpha=.25);a.set_ylim(bottom=0)
    ax[0].legend();fig.suptitle('MPM + Garcia-Parker supplied-sand demonstration; initially non-erodible bed')
    fig.savefig(REPORT/'MF_EVOLUTION.png',dpi=170);plt.close(fig)
    save(OUT/'EXAMPLE_COMPARISON.json',rows)
    print(json.dumps(dict(max_budget_kg=max_budget,max_water_residual_m3=max_water,
        max_backend_bed_mm=max(r['max_abs']['z'] for r in pairs)*1000,
        max_temporal_bed_mm=max(r['errors']['z'] for r in temporal)*1000,example=rows),indent=2))
if __name__=='__main__':main()
