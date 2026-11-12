"""Accepted-ledger budgets and raw-field comparisons for the groin matrix."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT)]
from braids_desktop.results import ResultsStore

def audit(root):
    check=json.loads((root/'CHECK.json').read_text())
    if check['status']=='expected_rejection':return check,None
    store=ResultsStore(root/'project/runs/audit');frames=[store.read(i) for i in range(len(store.times))]
    traces=json.loads((root/'TRACES.json').read_text());doc=store.project
    water={name:np.array([f.arrays['water'][store._boundary_mask(f,name)].sum() for f in frames]) for name in store.boundary_ids}
    volumes=np.array([np.dot(f.q[0],f.mesh['area']) for f in frames]);times=store.times
    schedule=doc['boundary_states']['Upstream']['hydraulic_schedule']
    if schedule['method']=='forcing.constant':expected=times*doc['settings']['hydraulics']['Upstream']['discharge_m3_s']
    else:
        expected=[]
        for t in times:
            points=np.unique(np.r_[0.,np.array(schedule['time_s'])[np.array(schedule['time_s'])<t],t])
            expected.append(np.trapezoid(np.interp(points,schedule['time_s'],schedule['discharge_m3_s']),points))
        expected=np.array(expected)
    diagnostics=json.loads((store.run/'DIAGNOSTICS.json').read_text()) if hasattr(store,'run') else json.loads((root/'project/runs/audit/DIAGNOSTICS.json').read_text())
    out=check|dict(cells=len(frames[-1].z),time_s=float(times[-1]),
        inlet_volume_error_m3=float(np.max(abs(water['Upstream']-expected))),
        water_balance_error_m3=float(np.max(abs(volumes-volumes[0]-water['Upstream']-water['Downstream']))),
        diagnostic_residual_m3=max(abs(r['water_residual_m3']) for r in diagnostics),
        wall_flux_max=max(abs(v) for r in diagnostics for v in r['wall_integrals']),
        minimum_depth_m=min(float(f.q[0].min()) for f in frames),
        maximum_depth_m=max(float(f.q[0].max()) for f in frames),
        outlet_Q_last_m3_s=traces[-1]['boundaries']['Downstream']['Q_out_m3_s'],
        outlet_Q_min_m3_s=min(r['boundaries']['Downstream']['Q_out_m3_s'] for r in traces),
        outlet_Q_max_m3_s=max(r['boundaries']['Downstream']['Q_out_m3_s'] for r in traces))
    assert out['inlet_volume_error_m3']<2e-8,out
    assert out['water_balance_error_m3']<2e-8 and out['wall_flux_max']==0 and out['minimum_depth_m']>=0,out
    states=[r['boundaries']['Upstream'] for r in traces]
    crit=[v['reference_section_Fr2'] for v in states if 'reference_section_Fr2' in v]
    if crit:
        out['inlet_reference_Fr2_error']=max(abs(x-1) for x in crit);assert out['inlet_reference_Fr2_error']<1e-10
    normal=[v['normal_metrics'] for v in states if 'normal_metrics' in v]
    if normal:
        slopes=json.loads((root/'project/prepared/base/NORMAL_SLOPES.json').read_text())
        out['normal_slope_receipt']=slopes
        out['normal_Q_error_m3_s']=max(abs(x[3]-4.) for x in normal);assert out['normal_Q_error_m3_s']<1e-9
        # Independent rectangle force balance, not the kernel's own residual.
        reference=[]
        for x in normal:
            A,P,D,Q=x;reference.append(abs(A*45.*np.sqrt((A/P)*.001)-4.))
        out['normal_independent_Q_error_m3_s']=max(reference);assert max(reference)<.00001
    crit=[r['boundaries']['Downstream'].get('controlled_section_Fr2') for r in traces]
    crit=[v for v in crit if v is not None]
    if crit:
        out['outlet_controlled_Fr2_error']=max(abs(x-1) for x in crit);assert out['outlet_controlled_Fr2_error']<1e-10
    weir=[]
    for r in traces:
        v=r['boundaries']['Downstream']
        if 'weir_stage_m' in v:
            control=doc['settings']['hydraulics']['Downstream']
            target=control['coefficient_m_half_s']*control['effective_width_m']*max(v['weir_stage_m']-control['crest_m'],0)**1.5
            if not any(v['weir_branch']):target=0.
            weir.append(abs(v['Q_out_m3_s']-target))
    if weir:out['weir_law_error_m3_s']=max(weir);assert max(weir)<1e-10
    if check['case']=='backflow':assert out['outlet_Q_min_m3_s']<0
    else:assert out['outlet_Q_min_m3_s']>=-1e-12
    if check['case'] in ('closed_stage','closed_weir'):assert out['outlet_Q_max_m3_s']==0.
    if check['case'].startswith('bedload_'):
        out['sediment_residual_kg']=max(abs(v) for r in diagnostics for v in r['class_residual_kg'])
        out['deposition_max_mm']=float(1000*(frames[-1].z-frames[-1].arrays['initial_bed_m']).max())
        out['sediment_stock_min']=float(frames[-1].arrays['mass'].min())
        out['outlet_sediment_ledger_max_m3']=max(float(f.arrays['bed_boundary'][...,store._boundary_mask(f,'Downstream')].max()) for f in frames)
        assert out['sediment_residual_kg']<1e-8 and out['sediment_stock_min']>=0 and out['outlet_sediment_ledger_max_m3']<=0
    return out,(store,frames,traces,water,volumes)

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);summary={};data={};failures={}
    for group in args.inputs:
        for root in sorted(group.iterdir()):
            if (root/'CHECK.json').is_file():summary[root.name],data[root.name]=audit(root)
            elif (root/'project/runs/audit/STATUS.json').is_file():
                status=json.loads((root/'project/runs/audit/STATUS.json').read_text())
                if status['status']=='failed':
                    failures[group.name+'/'+root.name]={k:status[k] for k in ('status','time_s','error')}
                    if (root/'REJECTION.json').is_file():failures[group.name+'/'+root.name]['last_rejection']=json.loads((root/'REJECTION.json').read_text())
    parity={}
    for name,value in data.items():
        if name.endswith('_cpu') and value is not None:
            other=data[name[:-3]+'gpu'];c=value[1][-1];g=other[1][-1]
            assert np.array_equal(c.mesh['leaves'],g.mesh['leaves'])
            parity[name[:-4]]={k:float(np.max(abs(c.arrays[k]-g.arrays[k]))) for k in ('q','z','water','mass','bed_boundary')}
            # Judge physical fields and integrated section totals, not a
            # unit-mixing bitwise test on every interior cumulative face.
            parity[name[:-4]]['depth_m']=float(np.max(abs(c.q[0]-g.q[0])))
            for bc in ('Upstream','Downstream'):
                cm=value[0]._boundary_mask(c,bc);gm=other[0]._boundary_mask(g,bc)
                difference=abs(float(c.arrays['water'][cm].sum()-g.arrays['water'][gm].sum()))
                parity[name[:-4]][bc+'_volume_m3']=difference
                scale=max(1.,abs(float(c.arrays['water'][cm].sum())))
                assert difference<1e-6*scale,(name,bc,difference)
            wet=(c.q[0]>1e-5)&(g.q[0]>1e-5)
            parity[name[:-4]]['velocity_m_s']=float(np.max(abs(c.q[1:,wet]/c.q[0,wet]-g.q[1:,wet]/g.q[0,wet]))) if wet.any() else 0.
            # User's millimetre-scale hydraulic/bed tolerance; velocities use
            # their own units. Conserved mass budgets are checked independently.
            assert max(parity[name[:-4]][k] for k in ('depth_m','z'))<.001,(name,parity[name[:-4]])
            assert parity[name[:-4]]['velocity_m_s']<.005,(name,parity[name[:-4]])
    duplicates={}
    for label,a,b in [('depth_equals_elevation','stage','depth'),('supercritical_equals_free','supercritical_valid','free_supercritical')]:
        if a+'_cpu' in data and b+'_cpu' in data:
            duplicates[label]={}
            for backend in ('cpu','gpu'):
                x=data[a+'_'+backend][1][-1];y=data[b+'_'+backend][1][-1]
                duplicates[label][backend]={k:float(np.max(abs(x.arrays[k]-y.arrays[k]))) for k in ('q','z','water')}
                assert max(duplicates[label][backend].values())==0.
    (args.output/'COMPARISON.json').write_text(json.dumps(dict(runs=summary,backend_max_abs=parity,duplicates=duplicates,failed_runs=failures),indent=2))
    groups=[('INLETS', ['total','stage','depth','critical','normal','schedule']),
            ('OUTLETS',['stage','out_critical','weir_valid','free_supercritical','backflow','closed_stage'])]
    for title,cases in groups:
        fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
        for ax,case in zip(axes.flat,cases):
            if case+'_cpu' not in data:continue
            s,fs,tr,water,v=data[case+'_cpu'];mid=(s.times[1:]+s.times[:-1])/2
            ax.plot(mid,np.diff(water['Upstream'])/np.diff(s.times),label='CPU inlet')
            ax.plot(mid,-np.diff(water['Downstream'])/np.diff(s.times),label='CPU outlet')
            gs,gfs,gtr,gw,gv=data[case+'_gpu'];gm=(gs.times[1:]+gs.times[:-1])/2
            ax.plot(gm,-np.diff(gw['Downstream'])/np.diff(gs.times),'--',label='GPU outlet')
            ax.axhline(0,color='gray',linewidth=.6);ax.set(title=case.replace('_',' '),xlabel='Physical time (s)',ylabel='Discharge (m3/s)');ax.grid(alpha=.2)
        axes.flat[0].legend();fig.suptitle('Groin hydraulic boundary audit: accepted flux / output interval\nNegative outlet flow means water entering through the outlet')
        fig.savefig(args.output/(title+'.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,9),layout='constrained')
    for ax,case in zip(axes.flat,['stage','normal','out_critical','weir_valid','dry_critical','backflow']):
        if case+'_cpu' not in data:continue
        s,fs,tr,water,v=data[case+'_cpu'];f=fs[-1]
        art=PolyCollection([np.asarray(p.exterior.coords) for p in f.polygons],array=f.q[0],cmap='Blues',edgecolors='none',clim=(0,1.))
        ax.add_collection(art);ax.fill([8,8,10,10],[0,3,3,0],color='#555555')
        ax.set(xlim=(0,20),ylim=(0,8),aspect='equal',title=case.replace('_',' ')+f' ({f.time_s:g} s)',xlabel='x (m)',ylabel='y (m)')
    fig.colorbar(art,ax=list(axes.flat),label='Water depth (m)',shrink=.7);fig.suptitle('Raw mesh water depths around the impermeable groin (CPU; GPU agrees)')
    fig.savefig(args.output/'DEPTH_MAPS.png',dpi=175);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for group in args.inputs:
        for case in ('out_weir','weir_recovered'):
            root=group/(case+'_cpu')
            if not (root/'REJECTION.json').is_file():continue
            s=ResultsStore(root/'project/runs/audit');tr=json.loads((root/'TRACES.json').read_text())
            velocities=[]
            for i in range(len(s.times)):
                f=s.read(i);xy=np.array([p.centroid.coords[0] for p in f.polygons]);edge=xy[:,0]>xy[:,0].max()-.01
                velocities.append(float(np.min(f.q[1,edge]/f.q[0,edge])))
            length=40 if case=='weir_recovered' else 20
            axes[0].plot(s.times,velocities,label=f'{length} m reach')
            axes[1].plot([r['time_s'] for r in tr],[r['boundaries']['Downstream']['Q_out_m3_s'] for r in tr],label=f'{length} m reach')
    for ax in axes:ax.axhline(0,color='gray',linewidth=.7);ax.set(xlabel='Time (s)');ax.grid(alpha=.2);ax.legend()
    axes[0].set(ylabel='Minimum outlet-adjacent cell velocity (m/s)')
    axes[1].set(ylabel='Last sampled outward weir discharge (m3/s)')
    fig.suptitle('Open issue: weir stops when a bank-side interior velocity reverses\nExtending the downstream reach delays the restriction; it does not remove it')
    fig.savefig(args.output/'WEIR_LIMITATION.png',dpi=175);plt.close(fig)
    print(json.dumps(dict(runs=len(summary),backend_pairs=len(parity),
        max_backend_depth_mm=1000*max(v['depth_m'] for v in parity.values()),
        max_backend_velocity_m_s=max(v['velocity_m_s'] for v in parity.values())),indent=2))
if __name__=='__main__':main()
