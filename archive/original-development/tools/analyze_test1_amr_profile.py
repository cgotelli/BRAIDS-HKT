"""Summarize profiles and illustrate a proposed depth-only mask offline."""
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'validation/_test1_amr_profile'
REPORT = ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/amr_profile111'
RUNTIME = Path('C:/Users/STREEM/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/braids_qgis/river_runtime')
sys.path.insert(0, str(RUNTIME)); sys.dont_write_bytecode = True
from braids_core.sediment.adaptive_river_worker import load_prepared
from braids_core.sediment.river_leaf_controller import RiverLeafController


def read(path): return json.loads(path.read_text())


def frame(folder, row):
    with np.load(folder/row['epoch']) as f: mesh = {k:f[k] for k in f.files}
    with np.load(folder/row['file']) as f: fields = {k:f[k] for k in f.files}
    return mesh,fields


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    summaries = {b:read(EVIDENCE/(b+'_clean')/'PROFILE_SUMMARY.json') for b in ('cpu','gpu')}
    metrics = {}
    for b,s in summaries.items():
        failed = [v for v in s['singles'] if v['error']]
        accepted = [v for v in s['singles'] if not v['error']]
        events = read(EVIDENCE/(b+'_clean')/'run/ADAPTATION.json')
        d = read(EVIDENCE/(b+'_clean')/'run/DIAGNOSTICS.json')
        metrics[b] = dict(wall_s=s['worker_call_s'],preparation_s=s['preparation_s'],phases=s['phases'],
            failed_single_s=sum(v['wall_s'] for v in failed),failed_singles=len(failed),
            accepted_single_s=sum(v['wall_s'] for v in accepted),accepted_singles=len(accepted),
            failed_evaluations=sum(v['evaluations'] for v in failed),
            total_evaluations=sum(v['evaluations'] for v in s['singles']),
            amr_assessments=len(events),amr_changes=sum(v['changed'] for v in events),
            indicators_s=s['inclusive_calls']['indicators'][1],
            maximum_water_budget_residual_m3=max(abs(v['water_residual_m3']) for v in d),
            maximum_wall_flux=max(max(v['wall_integrals']) for v in d),
            initial_cells=d[0]['cells'],final_cells=d[-1]['cells'],runtime_identity=s['runtime_identity'],
            failure_reasons=dict(Counter(v['error'].split(': norm=')[0] for v in failed)))
    folders={b:EVIDENCE/(b+'_clean')/'run/frames' for b in summaries}
    indices={b:read(f/'INDEX.json') for b,f in folders.items()}
    paired=[]
    for rc,rg in zip(indices['cpu']['frames'],indices['gpu']['frames'],strict=True):
        mc,c=frame(folders['cpu'],rc);mg,g=frame(folders['gpu'],rg)
        assert rc['time_s']==rg['time_s'] and np.array_equal(mc['leaves'],mg['leaves'])
        paired.append(dict(time_s=rc['time_s'],depth_max_m=float(abs(c['q'][0]-g['q'][0]).max()),
                           discharge_components_max_m2_s=float(abs(c['q'][1:]-g['q'][1:]).max())))
    metrics['paired']=paired
    labels=['Failed implicit attempts','Successful implicit substeps','Other step work',
            'AMR','Frames and checkpoints','Loading and other']
    colors=['#bf4e30','#287271','#86bbbd','#e9c46a','#6870a8','#c7c7c7']
    fig,ax=plt.subplots(figsize=(10,4.3),layout='constrained')
    left=np.zeros(2)
    values=[]
    for b in ('cpu','gpu'):
        m=metrics[b];p=m['phases']
        values.append([m['failed_single_s'],m['accepted_single_s'],
            p['stepping']-m['failed_single_s']-m['accepted_single_s'],p['amr'],
            p['output']+p['checkpoint'],p['load']+p['initialization']+p['other']])
    values=np.array(values)
    for k,(label,color) in enumerate(zip(labels,colors)):
        ax.barh(['CPU 8 threads','GPU RTX 5080'],values[:,k],left=left,label=label,color=color)
        for i,v in enumerate(values[:,k]):
            if v>2:ax.text(left[i]+v/2,i,f'{v:.1f}',ha='center',va='center',fontsize=10)
        left+=values[:,k]
    ax.set_xlabel('Wall seconds for 0–8 simulated seconds, excluding preparation')
    ax.set_title('Test1 20/10/5 m mesh — unchanged alpha111 solver')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,frameon=False,fontsize=9)
    fig.savefig(REPORT/'RUNTIME_BREAKDOWN.png',dpi=170);plt.close(fig)

    p,initial,tree=load_prepared(EVIDENCE/'gpu_clean/prepared/core/prepared')
    finebase=tree.offsets[-2];nf=len(tree.finest.cells)
    edge=np.array([(f.owner,f.neighbour) for f in tree.finest.faces if f.neighbour>=0],dtype=int)
    pins=np.unique([f.owner for f in tree.finest.faces if f.neighbour<0])
    raw=np.zeros(tree.count,dtype=int)
    minimum=RiverLeafController.pin_boundaries(tree,tree.balanced_leaves(raw))
    snapshots=[]
    for row in indices['gpu']['frames']:
        mesh,fields=frame(folders['gpu'],row)
        leaves=mesh['leaves'];cover=tree.leaf_cover(leaves)[finebase:]
        local={int(v):i for i,v in enumerate(leaves)}
        depth=fields['q'][0,np.array([local[int(v)] for v in cover])]
        wet=depth>1e-6;halo=wet.copy()
        for _ in range(2):
            enlarged=halo.copy()
            enlarged[edge[:,0]] |= halo[edge[:,1]]
            enlarged[edge[:,1]] |= halo[edge[:,0]]
            # Repeated destination indices require an explicit OR reduction.
            np.logical_or.at(enlarged,edge[:,0],halo[edge[:,1]])
            np.logical_or.at(enlarged,edge[:,1],halo[edge[:,0]])
            halo=enlarged
        raw=np.zeros(tree.count,dtype=int)
        raw[finebase+np.flatnonzero(halo)]=tree.finest_level
        target=RiverLeafController.pin_boundaries(tree,tree.balanced_leaves(raw))
        snapshots.append(dict(time_s=row['time_s'],current_cells=len(leaves),
            proposed_static_cells=len(target),wet_fine_cells=int(wet.sum()),
            same_leaf_ids=np.array_equal(np.asarray(target),leaves),
            wet_plus_two_rings_fine_cells=int(halo.sum()),boundary_finest_cells=len(pins)))
        if row is indices['gpu']['frames'][-1]:
            final=(mesh,fields,wet,halo,target)
    metrics['depth_only_static_analysis']=dict(
        meaning='Offline target counts only, not a numerical run or speed measurement. Two fine-face adjacency rings; no coarsening delay or dynamic front-safety schedule.',
        dry_threshold_m=1e-6,boundary_only_minimum_cells=len(minimum),frames=snapshots)
    mesh,fields,wet,halo,target=final
    origin=np.min(np.concatenate([c.vertices_xy for c in tree.finest.cells]),axis=0)
    polygons=[np.array(c.vertices_xy)-origin for c in tree.finest.cells]
    cover=tree.leaf_cover(mesh['leaves'])[finebase:]
    currentlevels=tree.node_level[cover]
    proposedlevels=tree.node_level[tree.leaf_cover(target)[finebase:]]
    category=np.zeros(nf);category[halo]=1;category[wet]=2
    drypin=np.zeros(nf,dtype=bool);drypin[pins]=True;category[drypin&~halo]=3
    fig,axs=plt.subplots(1,3,figsize=(10,7),layout='constrained',sharex=True,sharey=True)
    cm=ListedColormap(['#eeeeee','#e9c46a','#287271'])
    for ax,v,title in zip(axs[:2],[currentlevels,proposedlevels],
                          ['Current mesh at 8 s','Proposed depth-only static mask']):
        pc=PolyCollection(polygons,array=v,cmap=cm,norm=BoundaryNorm([-.5,.5,1.5,2.5],3),linewidth=0)
        ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');ax.set_title(title,fontsize=11)
    fig.colorbar(pc,ax=axs[:2],ticks=[0,1,2],fraction=.035,pad=.025,label='Level 0/1/2 = 20/10/5 m')
    pc2=PolyCollection(polygons,array=category,cmap=ListedColormap(['#eeeeee','#e9c46a','#287271','#8073ac']),
        norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4),linewidth=0)
    axs[2].add_collection(pc2);axs[2].autoscale();axs[2].set_aspect('equal')
    axs[2].set_title('Wet cells, buffer and fixed boundaries',fontsize=10)
    from matplotlib.patches import Patch
    axs[2].legend(handles=[Patch(color=c,label=l) for c,l in zip(
        ['#eeeeee','#e9c46a','#287271','#8073ac'],['Dry interior','Two-cell halo','Wet','Dry fixed boundary'])],
        loc='upper right',fontsize=8)
    for ax in axs:ax.set_xlabel('Relative easting (m)')
    axs[0].set_ylabel('Relative northing (m)')
    fig.savefig(REPORT/'DEPTH_ONLY_PROPOSAL.png',dpi=170);plt.close(fig)
    (REPORT/'METRICS.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
