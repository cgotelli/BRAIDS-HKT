"""Read-only installed-solver diagnosis of the refined two-bend failure.

The diagnostic wrapper records values without changing arithmetic. Control
inputs and outputs are separate from the user's current project and history.
"""
import argparse,json,os,sys,time
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'desktop/runtime'
BASE=ROOT/'desktop/examples/two_meanders184/advanced_on'
FAILED=BASE/'runs/run_20261027_084649_a62dd04f'
PREP=BASE/'prepared/prepare_20261027_084626_ba724164'
OUT=ROOT/'validation/_secondary185'
sys.path[:0]=[str(RUNTIME),str(ROOT)]
os.environ['QT_QPA_PLATFORM']='offscreen'
import numpy as np


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dt',type=float,default=.2)
    parser.add_argument('--until',type=float,default=5.);parser.add_argument('--tag',required=True)
    parser.add_argument('--runtime',type=Path,default=RUNTIME)
    parser.add_argument('--backend',choices=('cpu','gpu'),default='cpu')
    parser.add_argument('--wet',action='store_true')
    parser.add_argument('--sediment-start',type=float)
    args=parser.parse_args();target=OUT/args.tag
    if target.exists():raise FileExistsError(target)
    target.mkdir(parents=True)
    runtime=args.runtime.resolve();sys.path.insert(0,str(runtime))
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    verify();install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)
    from braids_core import river_worker
    from braids_desktop import services
    from braids_desktop.preparation import prepare
    from braids_core.sediment.flow2dh_fixed_bed import Flow2DHFixedBed
    from braids_core.sediment.bed_columns_v101 import host
    prepared=PREP
    if args.dt!=.2 or runtime!=RUNTIME or args.backend!='cpu' or args.wet:
        doc=json.loads((FAILED/'PROJECT.json').read_text());doc.pop('editor_state',None)
        dem=Path(doc['terrain']['dem_path'])
        doc['terrain']['dem_path']=str(dem if dem.is_absolute() else BASE/dem)
        doc['settings']['dt_max_s']=args.dt
        from braids_core.river_model import resolve_model
        request=doc['model']['request'];request['execution']['backend']=args.backend
        doc['model']=resolve_model(request).document()
        if args.wet:doc['initial']['stage_m']=1.6
        if args.sediment_start is not None:doc['settings']['sediment_start_time_s']=args.sediment_start
        services.save_project_folder(target/'project',doc)
        prepared=target/'prepared'
        prepare(target/'project/braids_project.json',prepared,asset_cache=OUT/'cache')
    rows=[];original=Flow2DHFixedBed.finish
    def recorded(p,qold,qnew,water,**kwargs):
        error=None
        try:return original(p,qold,qnew,water,**kwargs)
        except Exception as exc:error=str(exc);raise
        finally:
            g={k:host(v) for k,v in p.g.items()};h=host(qold)[0]
            w=host(p.Wtrial);old=host(p.W);flux=host(water);dt=p._dt
            i=int(np.argmax(abs(w)));fids=g['faces'];o=g['owner'];nb=g['neighbour'];L=g['length']
            outgoing=np.bincount(o,weights=L*np.maximum(flux,0),minlength=p.n)
            mask=nb>=0
            outgoing+=np.bincount(nb[mask],weights=L[mask]*np.maximum(-flux[mask],0),minlength=p.n)
            cfl=np.divide(dt*outgoing,g['area']*h,out=np.zeros_like(h),where=h>1e-4)
            outside=(nb<0)&(g['bid']>=0)
            boundary=-np.sum(host(p.scalarflux_integral)*L*outside)
            source=np.sum(host(p.source_increment)*g['area'])
            residual=np.sum((w-old)*g['area'])-boundary-source
            row=dict(time_s=float(p.time),dt_s=dt,rejected=error,
                W_before_max=float(abs(old).max()),W_trial_max=float(abs(w).max()),
                W_peak_xy=g['xy'].reshape(2,-1)[:,i].tolist(),W_peak_cell=i,
                W_peak_area=float(g['area'][i]),W_peak_old_depth=float(h[i]),
                scalar_advective_CFL_max=float(cfl.max()),scalar_advective_CFL_at_peak=float(cfl[i]),
                secondary_residual=float(residual))
            if hasattr(p,'secondary_clock_stats'):
                row['accepted_secondary_substeps']=int(p.secondary_clock_stats[1])
                row['accepted_secondary_CFL']=float(p.secondary_clock_stats[2])
            rows.append(row)
            if len(rows)<12 or error or len(rows)%100==0:
                print(json.dumps(row),flush=True)
    Flow2DHFixedBed.finish=recorded
    start=time.perf_counter();error=None
    try:status=river_worker.run(prepared,target/'run',until=args.until)
    except Exception as exc:
        error=str(exc);status=json.loads((target/'run/STATUS.json').read_text())
    finally:
        Flow2DHFixedBed.finish=original
        save(target/'TRACE.json',rows)
    from braids_desktop.results import ResultsStore
    last=ResultsStore(target/'run').read(-1)
    save(target/'CHECK.json',dict(dt_max_s=args.dt,status=status['status'],time_s=status['time_s'],
        elapsed_s=time.perf_counter()-start,error=error,maximum_W=float(abs(last.arrays['W']).max()),
        maximum_speed=float(last.values['velocity_m_s'].max()),
        secondary_calls=len(rows),
        hypothetical_outer_scalar_advective_CFL_max=max(r['scalar_advective_CFL_max'] for r in rows),
        accepted_secondary_CFL_max=max((r['accepted_secondary_CFL'] for r in rows if 'accepted_secondary_CFL' in r),default=None)))
    print('FINISHED',args.tag,status['status'],status['time_s'],flush=True)


if __name__=='__main__':main()
