"""Evidence and raw-cell figures for exact reflecting-wall qualification."""
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
from braids_desktop.results import ResultsStore
from check_bedload161 import save,sha
from review_groin163 import budgets
RAW=ROOT/'validation/_wall171'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_reflection171'


def assess(tag,case,angle):
    base=[RAW/tag/f'{case}_{angle}_{backend}' for backend in ('cpu','gpu')]
    stores=[ResultsStore(p/'full') for p in base]
    assert np.array_equal(stores[0].times,stores[1].times)
    histories=[];maximum={k:0. for k in ('q','z','mass','water','bed_boundary')}
    frames=[]
    for i,time in enumerate(stores[0].times):
        x,y=[s.read(i) for s in stores];frames.append(x)
        assert x.q.shape==y.q.shape
        for k in maximum:maximum[k]=max(maximum[k],float(abs(x.arrays[k]-y.arrays[k]).max(initial=0)))
        histories.append(dict(time_s=time,cells=x.q.shape[1],depth_min=float(x.q[0].min()),depth_max=float(x.q[0].max()),
            wet_cells=int((x.q[0]>1e-6).sum()),wet_area_m2=float(sum(p.area for p,h in zip(x.polygons,x.q[0]) if h>1e-6)),
            depth_parity_m=float(abs(x.q[0]-y.q[0]).max()),
            volume_m3=float(sum(p.area*h for p,h in zip(x.polygons,x.q[0])))))
    # Wetting controls agree at roundoff. The nonlinear mobile groin amplifies
    # roundoff to 0.004 mm depth: below the user's millimetre-scale allowance,
    # but retain/report its failed preliminary 1e-6 mixed-q screen honestly.
    assert maximum['q']<(1e-5 if case=='groin' else 1e-11) and maximum['z']<1e-8 and maximum['mass']<1e-5,maximum
    assert min(r['depth_min'] for r in histories)>=0
    result=dict(tag=tag,case=case,angle=angle,maximum_backend_difference=maximum,history=histories,backends={})
    for backend,p in zip(('cpu','gpu'),base):
        audit=json.loads((p/'AUDIT.json').read_text());b=budgets(p/'full');adapt=json.loads((p/'full/ADAPTATION.json').read_text())
        result['backends'][backend]=dict(budgets=b,status=audit['status'],restart=audit['restart'],
            changed_mesh_count=sum(bool(r['changed']) for r in adapt),
            minimum_area_m2=min(v.area for f in frames for v in f.polygons))
        assert b['wall_integral']==0 and audit['status']['hydraulic']['accepted_cfl_max']<=1+1e-12
        if audit['restart']:
            assert all(audit['restart']['exact_fields'][k] for k in maximum)
            assert audit['restart']['cancel']['status']=='cancelled'
        if tag=='amr_v1':assert result['backends'][backend]['changed_mesh_count']>0
    return result,stores,frames


def figures(results,data):
    fig,axs=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for row,name in enumerate(('reflection_37','wetting_37')):
        stores,frames=data[name]
        for col,i in enumerate((0,4,-1)):
            f=frames[i];ax=axs[row,col]
            polygons=[np.asarray(p.exterior.coords) for p in f.polygons]
            vals=np.ma.masked_where(f.q[0]<=1e-6,f.q[0])
            pc=PolyCollection(polygons,array=vals,cmap='Blues',clim=(0,.8),edgecolors='#777777',linewidths=.45)
            ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal')
            ax.set(title=f'{name.split("_")[0].capitalize()}, t = {stores[0].times[i]:g} s',xlabel='x (m)',ylabel='y (m)')
            centres=np.array([[p.centroid.x,p.centroid.y] for p in f.polygons]);wet=f.q[0]>.01
            uv=f.q[1:,wet]/f.q[0,wet]
            ax.quiver(centres[wet,0],centres[wet,1],uv[0],uv[1],color='#b44d24',scale=9,width=.004)
    fig.colorbar(pc,ax=axs,label='Water depth (m); dry cells unfilled',shrink=.65)
    fig.suptitle('Exact wall pressure: raw cropped/merged cells at 37 degrees\nCPU shown; GPU trajectory differences reported separately')
    fig.savefig(OUT/'WALL_TRAJECTORIES.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    for name,r in results.items():
        if r['case']=='groin':continue
        times=[x['time_s'] for x in r['history']]
        axs[0,0].plot(times,[x['depth_parity_m']*1000 for x in r['history']],label=name)
        if r['case']=='wetting':
            axs[0,1].plot(times,[x['wet_area_m2'] for x in r['history']],label=name)
            axs[1,0].plot(times,[x['volume_m3']-r['history'][0]['volume_m3'] for x in r['history']],label=name)
    axs[0,0].set(title='CPU/GPU agreement on complete trajectories',xlabel='Time (s)',ylabel='Maximum depth difference (mm)')
    axs[0,1].set(title='Wetting on the sloping bed',xlabel='Time (s)',ylabel='Wet area (m²)')
    axs[1,0].set(title='Closed basin: no water source',xlabel='Time (s)',ylabel='Water-volume change (m³)')
    r=results['amr_wetting_37'];axs[1,1].step([x['time_s'] for x in r['history']],[x['cells'] for x in r['history']],where='post',label='AMR cell count')
    axs[1,1].axvline(2,color='#b44d24',ls='--',label='Cancel / exact restart')
    axs[1,1].set(title='Changing cropped-wall ownership',xlabel='Time (s)',ylabel='Active cells')
    for ax in axs.flat:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Numerical qualification, not experimental validation')
    fig.savefig(OUT/'CONSERVATION_AND_PARITY.png',dpi=150);plt.close(fig)
    stores,frames=data['groin_0'];f=frames[-1];initial=frames[0];resumed=ResultsStore(RAW/'v1/groin_0_cpu/resumed').read(-1)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,values,title,clim in ((axs[0],(f.z-initial.z)*1000,'MPM bed change at 44 s (4 s mobile bed)',(-8,8)),
                               (axs[1],(f.z-resumed.z)*1000,'Uninterrupted minus cancel/restart',(-1,1))):
        pc=PolyCollection([np.asarray(p.exterior.coords) for p in f.polygons],array=values,cmap='RdBu',clim=clim,edgecolors='.75',linewidths=.12)
        ax.add_collection(pc);ax.set(xlim=(6,16),ylim=(0,6),aspect='equal',xlabel='x (m)',ylabel='y (m)',title=title)
        fig.colorbar(pc,ax=ax,label='Bed difference (mm)',shrink=.8)
    fig.suptitle('Bedload restart stays exact; this short check does not resolve the FLOW jet discrepancy')
    fig.savefig(OUT/'BEDLOAD_RESTART.png',dpi=150);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True);results={};data={};probe=[]
    for case,angles in [('reflection',(0,17,37,63,90)),('wetting',(0,17,37)),('groin',(0,))]:
        for angle in angles:
            r,s,f=assess('v1',case,angle);name=f'{case}_{angle}';results[name]=r;data[name]=(s,f)
            if case=='reflection':
                a=RAW/'v1'/f'{case}_{angle}_cpu/probes';b=RAW/'v1'/f'{case}_{angle}_gpu/probes'
                cpu=json.loads((a/'PROBES.json').read_text());gpu=json.loads((b/'PROBES.json').read_text())
                for x,y in zip(cpu,gpu):
                    aa=np.load(a/(x['name']+'.npz'));bb=np.load(b/(x['name']+'.npz'))
                    diff=float(abs(aa['rate']-bb['rate']).max());water=float(abs(aa['water']-bb['water']).max())
                    assert diff<1e-11 and water<1e-12,(angle,x,diff,water)
                    probe.append(dict(angle=angle,cpu=x,gpu=y,rate_parity=diff,water_parity=water))
    r,s,f=assess('amr_v1','wetting',37);results['amr_wetting_37']=r;data['amr_wetting_37']=(s,f)
    for tag,case in [('lake_v1','wetting'),('vacuum_v1','reflection')]:
        r,s,f=assess(tag,case,37);name=tag.split('_')[0]+'_37';results[name]=r;data[name]=(s,f)
        if tag=='lake_v1':
            r['lake_drift_max']=float(max(abs(x.q-f[0].q).max() for x in f))
            assert r['lake_drift_max']<1e-10,r
    figures(results,data)
    old=json.loads((ROOT/'validation/sediment/river_morphodynamics/reports/desktop_interface170/AUDIT.json').read_text())
    native=old['native_sources_read_only']
    with zipfile.ZipFile(ROOT/'resources/Delft3D-main.zip') as z:
        for name,digest in native.items():
            assert sha(ROOT/name)==digest
            assert (ROOT/name).read_bytes()==z.read(Path(name).as_posix().removeprefix('resources/'))
    runtime=ROOT/'desktop/runtime';manifest=json.loads((runtime/'RUNTIME.json').read_text())
    for name,digest in manifest['files'].items():assert sha(runtime/name)==digest,name
    files=[p for p in RAW.rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.py','.tif')]
    audit=dict(cases=results,probes=probe,production_changed=False,Delft3D_modified=False,
        source_reference=native,runtime_manifest_sha256=sha(runtime/'RUNTIME.json'),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in files},
        scripts={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'desktop').glob('*171.py')})
    save(OUT/'AUDIT.json',audit)
    print(json.dumps({k:dict(parity=v['maximum_backend_difference'],cpu_budget=v['backends']['cpu']['budgets'],
        wet_area=[v['history'][0]['wet_area_m2'],max(x['wet_area_m2'] for x in v['history']),v['history'][-1]['wet_area_m2']],
        amr_changes=v['backends']['cpu']['changed_mesh_count']) for k,v in results.items()},indent=2))


if __name__=='__main__':main()
