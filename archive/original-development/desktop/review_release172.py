"""Gate the compiler-free Desktop delivery and preserve the open jet question."""
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'validation/_delivery172'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_delivery172'
sys.path[:0]=[str(BASE/'candidate'),str(ROOT),str(ROOT/'desktop')]
from stage_wall172 import sha,save
from braids_desktop.results import ResultsStore
from braids_core import river_worker

CASES=('total','stage','critical','normal','dry_stage','dry_critical','dry_normal','out_critical','weir_valid',
       'closed_weir','backflow','closed_stage','zero','schedule','bedload_weir_valid','invalid_dry_total')
LAWS=('mpm','ashida_michiue_generalized','wang_fredsoe')


def budget(run):
    rows=json.loads((run/'DIAGNOSTICS.json').read_text())
    b=dict(water=max(abs(x['water_residual_m3']) for x in rows),
        sediment=max(abs(v) for x in rows for v in x['class_residual_kg']),
        wall=max(abs(v) for x in rows for v in x['wall_integrals']),stock=min(x['minimum_stock'] for x in rows))
    assert b['water']<1e-8 and b['sediment']<1e-8 and b['wall']==0 and b['stock']>=0,b
    return b


def main():
    OUT.mkdir(parents=True,exist_ok=True);stage=json.loads((BASE/'STAGE.json').read_text())
    runtime=BASE/'candidate'
    assert sha(runtime/'RUNTIME.json')==stage['runtime_sha256']
    for name,digest in json.loads((runtime/'RUNTIME.json').read_text())['files'].items():assert sha(runtime/name)==digest,name
    runs={};bounds={};laws={};inputs={}
    for p in sorted((BASE/'package').iterdir()):
        if not (p/'CHECK.json').is_file():continue
        c=json.loads((p/'CHECK.json').read_text());assert c['comparison']['exact'] and all(c['restart']['exact'].values())
        assert c['status']['status']=='complete' and c['runtime_sha256']==stage['runtime_sha256']
        owner=json.loads((p/'OWNER.json').read_text());assert owner['reflecting_wall']['method']=='exact_reflecting_wall_riemann_v1'
        runs[p.name]=dict(check=c,budget=budget(p/'full'))
        for q in p.rglob('*'):
            if q.is_file() and (q.name in ('CHECK.json','OWNER.json','project.json','STATUS.json','DIAGNOSTICS.json','INDEX.json','checkpoint.npz') or q.parent.name=='frames' and q.suffix=='.npz'):
                inputs[str(q.relative_to(ROOT))]=sha(q)
    assert len(runs)==16,len(runs)
    for case in CASES:
        for backend in ('cpu','gpu'):
            p=BASE/f'boundaries/{case}_{backend}';c=json.loads((p/'CHECK.json').read_text())
            assert c['status']==('expected_rejection' if case.startswith('invalid') else 'complete'),c
            item=dict(check=c)
            if c['status']=='complete':
                item['budget']=budget(p/'project/runs/audit')
                traces=json.loads((p/'TRACES.json').read_text());item['traces']=traces
                if 'weir_valid' in case:
                    item['maximum_upstream_boundary_Fr']=max(x['boundaries']['Downstream']['trace_Froude_min'] for x in traces)
                    assert item['maximum_upstream_boundary_Fr']<1
            bounds[f'{case}_{backend}']=item
            for name in ('CHECK.json','TRACES.json','REJECTION.json'):
                q=p/name
                if q.exists():inputs[str(q.relative_to(ROOT))]=sha(q)
    for law in LAWS:
        cpu=ResultsStore(BASE/f'package/groin_{law}_cpu/full');gpu=ResultsStore(BASE/f'package/groin_{law}_gpu/full')
        assert np.array_equal(cpu.times,gpu.times)
        parity={k:0. for k in ('q','z','mass')}
        for i in range(len(cpu.times)):
            x=cpu.read(i);y=gpu.read(i)
            for k in parity:parity[k]=max(parity[k],float(abs(x.arrays[k]-y.arrays[k]).max()))
        assert parity['q']<1e-5 and parity['z']<1e-7 and parity['mass']<1e-4,parity
        z=cpu.read(-1).z-cpu.read(0).z
        laws[law]=dict(parity=parity,scour_mm=-1000*float(z.min()),deposition_mm=1000*float(z.max()))
    # Fresh runtime identities must reject old preparation and checkpoints.
    identity_checks={}
    for backend in ('cpu','gpu'):
        old=ROOT/f'validation/_wall171/v1/reflection_17_{backend}'
        new=BASE/f'package/v1_reflection_17_{backend}'
        try:river_worker.load_prepared(old/'prepared')
        except ValueError as exc:message=str(exc)
        else:raise AssertionError('Old preparation was accepted')
        assert 'Runtime changed' in message,message
        project,receipt=river_worker.load_prepared(new/'prepared');expected=river_worker.checkpoint_identity(project,receipt)
        try:river_worker.read_checkpoint(old/'cancelled/checkpoint.npz',expected)
        except ValueError as exc:checkpoint=str(exc)
        else:raise AssertionError('Old checkpoint accepted')
        identity_checks[backend]=dict(preparation=message,checkpoint=checkpoint)
    oldaudit=json.loads((ROOT/'validation/sediment/river_morphodynamics/reports/desktop_interface170/AUDIT.json').read_text())
    for name,digest in oldaudit['native_sources_read_only'].items():assert sha(ROOT/name)==digest
    figures(laws,bounds)
    save(OUT/'AUDIT.json',dict(stage=stage,runs=runs,boundaries=bounds,laws=laws,identity_checks=identity_checks,
        inputs=inputs,source_reference=oldaudit['native_sources_read_only'],
        scripts={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'desktop').glob('*172.py')},
        Delft3D_modified=False,sediment_laws_changed=False,groin_jet_discrepancy_closed=False))
    print(json.dumps(dict(runs=len(runs),boundaries=len(bounds),laws=laws,identity_checks=identity_checks),indent=2))


def figures(laws,bounds):
    fig,axs=plt.subplots(1,3,figsize=(12,4.3),layout='constrained')
    for ax,case in zip(axs,('critical','normal','weir_valid')):
        for backend,style in [('cpu','-'),('gpu','--')]:
            rows=bounds[case+'_'+backend]['traces'];t=[r['time_s'] for r in rows]
            ax.plot(t,[-r['boundaries']['Upstream']['Q_out_m3_s'] for r in rows],style,label=backend.upper()+' inlet')
            ax.plot(t,[r['boundaries']['Downstream']['Q_out_m3_s'] for r in rows],style,label=backend.upper()+' outlet')
        ax.set(title=case.replace('_',' '),xlabel='Time (s)',ylabel='Discharge (m³/s)',ylim=(0,None));ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Packaged runtime: prescribed inlet discharge and distinct outlet controls')
    fig.savefig(OUT/'BOUNDARIES.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    names=['MPM','Ashida–Michiue','Wang–Fredsøe'];x=np.arange(3)
    axs[0].bar(x-.16,[laws[k]['scour_mm'] for k in LAWS],width=.32,label='Maximum scour')
    axs[0].bar(x+.16,[laws[k]['deposition_mm'] for k in LAWS],width=.32,label='Maximum deposition')
    axs[0].set_xticks(x,names);axs[0].set(ylabel='Bed change magnitude (mm)',title='Same groin inputs, 20 s morphology');axs[0].legend()
    axs[1].bar(x,[laws[k]['parity']['z']*1000 for k in LAWS]);axs[1].set_xticks(x,names)
    axs[1].set(ylabel='Maximum CPU/GPU bed difference (mm)',title='All saved frames, not just endpoint')
    for ax in axs:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Three supported laws; package exactly matches qualified wall arithmetic on each backend')
    fig.savefig(OUT/'BEDLOAD_PACKAGE.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
