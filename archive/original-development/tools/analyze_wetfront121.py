"""Measured nonlinear research summaries, raw-cell plots and immutable evidence."""
import json
import hashlib
import sys
import zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'validation/_wetfront121'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/wetfront121'


def main():
    from braids_core.sediment.adaptive_river_worker import read_live_frame
    REPORT.mkdir(parents=True,exist_ok=True)
    folders={p.name:p for p in BASE.iterdir() if (p/'PROFILE_SUMMARY.json').exists()}
    for name in ('aligned_gpu_main','aligned_gpu_half','aligned_cpu_main'):
        folders[name]=ROOT/'validation/_axis_boundary120'/name
    results={};states={};geom=None
    for name,path in folders.items():
        profile=json.loads((path/'PROFILE_SUMMARY.json').read_text())
        status=profile['result'];_,g,s=read_live_frame(path/'run')
        if status['time_s']!=3.75:continue
        q=s['q'];wet=q[0]>.001
        results[name]=dict(path=str(path),profile=profile,volume=float(q[0]@g['area']),
            minimum_depth=float(q[0].min()),maximum_depth=float(q[0].max()),
            wet_area=float(g['area'][wet].sum()),
            maximum_wet_speed=float((np.linalg.norm(q[1:,wet],axis=0)/q[0,wet]).max()),
            diagnostics=json.loads((path/'run/DIAGNOSTICS.json').read_text()))
        states[name]=s;geom=g
    comparisons={}
    for a in states:
        comparisons[a]={}
        for b in states:
            qa,qb=states[a]['q'],states[b]['q'];dh=qa[0]-qb[0];wet=(qa[0]>.001)|(qb[0]>.001)
            both=(qa[0]>.001)&(qb[0]>.001)
            comparisons[a][b]=dict(max_depth_mm=float(abs(dh).max()*1000),
                wet_rms_depth_mm=float(np.sqrt(np.mean(dh[wet]**2))*1000),max_cell=int(abs(dh).argmax()),
                wet_velocity_rms_m_s=float(np.sqrt(np.mean(np.sum((qa[1:,both]/qa[0,both]-qb[1:,both]/qb[0,both])**2,axis=0)))))
    (REPORT/'SUMMARY.json').write_text(json.dumps(dict(runs=results,comparisons=comparisons),indent=2))
    table=[]
    for n,r in results.items():
        h=r['profile']['result']['hydraulic']
        table.append(dict(run=n,step_s=r['profile']['phases']['stepping'],substeps=h['accepted_substeps'],
            fwave=h['fwave_evaluations'],min_dt=h['minimum_substep_s'],max_h=r['maximum_depth']))
    print(json.dumps(table,indent=2))
    names=[n for n in ('aligned_gpu_main','jfnk_gpu_cap03125','jfnk_gpu_drag03125','jfnk_gpu_recover',
        'final_gpu','jfnk_gpu_tol7_eighth') if n in states]
    labels={'aligned_gpu_main':'Baseline acoustic iteration','jfnk_gpu_cap03125':'Full-residual direction only',
        'jfnk_gpu_drag03125':'Hydraulic-step friction','jfnk_gpu_recover':'+ Earlier step recovery',
        'final_gpu':'Selected short-case candidate','jfnk_gpu_tol7_eighth':'Finer temporal control'}
    fig,ax=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    for i,n in enumerate(names):
        prof=results[n]['profile'];h=prof['result']['hydraulic']
        ax[0].barh(i,prof['phases']['stepping']);ax[1].barh(i,h['accepted_substeps'])
    for a in ax:a.set_yticks(range(len(names)),[labels[n] for n in names]);a.grid(axis='x',alpha=.2)
    ax[0].set_xlabel('Hydraulic stepping wall time (s)');ax[1].set_xlabel('Accepted hydraulic substeps')
    fig.suptitle('Short cold start to 3.75 s — research variants, not released')
    fig.savefig(REPORT/'WORK_COMPARISON.png',dpi=150);plt.close(fig)
    selected=[n for n in ('aligned_gpu_main','final_gpu','jfnk_gpu_tol7_eighth') if n in states]
    fig,axs=plt.subplots(1,len(selected),figsize=(4*len(selected),6),layout='constrained')
    polys=[geom['vertices'][geom['offsets'][i]:geom['offsets'][i+1]]-np.array([2571550,1106690]) for i in range(len(geom['area']))]
    centers=np.array([p.mean(axis=0) for p in polys]);sel=(centers[:,1]<30)&(centers[:,1]>0)&(centers[:,0]>0)&(centers[:,0]<40)
    for ax,n in zip(np.atleast_1d(axs),selected):
        pc=PolyCollection([p for p,ok in zip(polys,sel) if ok],array=states[n]['q'][0,sel],cmap='Blues',clim=(0,.45),linewidth=.12,edgecolor='gray')
        ax.add_collection(pc);ax.set(xlim=(0,40),ylim=(0,30),aspect='equal',title=labels[n],xlabel='Easting minus 2571550 (m)')
    axs[0].set_ylabel('Northing minus 1106690 (m)');fig.colorbar(pc,ax=axs,label='Depth (m)',shrink=.6)
    fig.savefig(REPORT/'RAW_INLET_DEPTHS.png',dpi=150);plt.close(fig)
    if 'final_gpu' in states and 'jfnk_gpu_tol7_eighth' in states:
        final=states['final_gpu'];fine=states['jfnk_gpu_tol7_eighth']
        fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
        delta=(final['q'][0]-fine['q'][0])*1000
        pc=PolyCollection([p for p,ok in zip(polys,sel) if ok],array=delta[sel],cmap='RdBu_r',clim=(-.5,.5),linewidth=.2,edgecolor='gray')
        ax.add_collection(pc);ax.set(xlim=(0,40),ylim=(0,30),aspect='equal',xlabel='Easting minus 2571550 (m)',
            ylabel='Northing minus 1106690 (m)',title='Depth sensitivity to halving the hydraulic step cap')
        fig.colorbar(pc,ax=ax,label='Candidate minus finer control (mm)')
        fig.savefig(REPORT/'TEMPORAL_DIFFERENCE.png',dpi=150);plt.close(fig)
    if '--seal' in sys.argv:
        archive=REPORT/'SOURCES.zip'
        if archive.exists():raise FileExistsError('Do not overwrite sealed research sources')
        sources=[ROOT/p for p in ('tools/wetfront121.py','tools/analyze_wetfront121.py',
            'tools/profile_test1_amr.py','tests/test_wetfront121_research.py','tests/test_axis_aligned_boundaries.py',
            'braids_core/sediment/river_midpoint.py','braids_core/sediment/river_midpoint.h',
            'braids_core/sediment/shoreline_solver.py','braids_core/sediment/river_roughness.py',
            'validation/RIVER_MORPHODYNAMICS_STATUS.md','validation/sediment/river_morphodynamics/UNIFIED_QGIS_PLAN.md')]
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for path in sources:z.write(path,path.relative_to(ROOT))
        files=[*sources,*REPORT.glob('*.png'),REPORT/'REPORT.md',REPORT/'SUMMARY.json',archive]
        files += [ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/axis_boundary120/SOURCES.zip']
        for folder in BASE.iterdir():
            if not folder.is_dir():continue
            for name in ('PROJECT.json','RESEARCH.json','PROFILE_SUMMARY.json','EXECUTED_RESEARCH.py','TESTS.xml',
                'run/DIAGNOSTICS.json','run/STATUS.json','run/frames/INDEX.json'):
                if (folder/name).is_file():files.append(folder/name)
        manifest={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        (REPORT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
