"""Independent experiment benchmark, launch supervision and local progress UI."""
from pathlib import Path
import argparse,csv,datetime as dt,html,json,os,statistics,subprocess,sys,time
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
PROJECT=HERE if (HERE/'runtime').is_dir() and (HERE/'INPUT_AUDIT.json').is_file() else HERE.parent.parent/'BRAIDS_JiangCao_Experiment225'
ROOT=HERE.parent if HERE.name=='desktop' else Path(json.loads((HERE/'CONTROL.json').read_text())['repository'])
PYTHON=ROOT/'desktop/.venv/Scripts/python.exe'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_experiment225'
ZONE=ZoneInfo('Europe/Zurich')

def read(path,default=None):
    try:return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):return default

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2) if not isinstance(value,str) else value
    tmp=path.with_name('.'+path.name+'.tmp');tmp.write_text(text,encoding='utf8')
    for attempt in range(6):
        try:tmp.replace(path);return
        except PermissionError:
            if attempt==5:raise
            time.sleep(.05*2**attempt)

def environment(threads):
    return dict(os.environ,OMP_NUM_THREADS=str(threads),OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUTF8='1')

def invoke(args,log,threads):
    with log.open('x',encoding='utf8') as stream:
        result=subprocess.run([str(PYTHON),'-X','utf8',*map(str,args)],cwd=ROOT,env=environment(threads),
            stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Worker exited {result.returncode}; see {log}')

def benchmark():
    rows=[]
    for n in (1,2,4,8,16,32):
        label=f'cpu_{n:02d}_a'
        invoke([ROOT/'desktop/experiment225.py','trial','--threads',n,'--label',label,'--end','.25'],
            PROJECT/'benchmarks'/(label+'.log'),n)
        row=read(PROJECT/'benchmarks'/label/'TIMING.json');rows.append(row)
        print(n,'threads:',round(row['step_s'],3),'s stepping; median',round(row['warm_step_median_s'],6),flush=True)
    first=sorted(rows,key=lambda r:r['step_s'])[:2]
    for row in reversed(first):
        n=row['threads'];label=f'cpu_{n:02d}_b'
        invoke([ROOT/'desktop/experiment225.py','trial','--threads',n,'--label',label,'--end','.25'],
            PROJECT/'benchmarks'/(label+'.log'),n)
        rows.append(read(PROJECT/'benchmarks'/label/'TIMING.json'))
    # Paired repeats decide the winner; preserve every timing and actual field.
    finalists={r['threads'] for r in first}
    scores={n:statistics.median(r['step_s'] for r in rows if r['threads']==n) for n in finalists}
    winner=min(scores,key=scores.get)
    import numpy as np
    def final(row):
        folder=Path(row['output'])/'frames';index=read(folder/'INDEX.json')
        return np.load(folder/index['frames'][-1]['file'])
    differences=[]
    with final(rows[0]) as ref:
        for row in rows[1:]:
            with final(row) as other:
                bed=float(np.max(np.abs(ref['z']-other['z']))*1000)
                depth=float(np.max(np.abs(ref['q'][0]-other['q'][0]))*1000)
                assert max(bed,depth)<.1,(row['label'],bed,depth)
                differences.append(dict(label=row['label'],bed_mm=bed,depth_mm=depth))
    record=dict(selected_threads=winner,rows=rows,scores_s=scores,field_differences=differences,
        method='Same full flume, same cold start and 50 nominal steps; CPU execution only. Best two repeated in reverse order. Construction/output excluded from step timings.',
        environment=environment(winner))
    # Save only the relevant environment; exclude unrelated host secrets.
    record['environment']={k:environment(winner)[k] for k in ('OMP_NUM_THREADS','OMP_DYNAMIC','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONUTF8')}
    write(PROJECT/'CORE_BENCHMARK.json',record);write(REPORT/'CORE_BENCHMARK.json',record)
    invoke([ROOT/'desktop/experiment225.py','select','--threads',winner],PROJECT/'logs/select_cores.log',winner)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for r in rows:ax.plot(r['threads'],r['step_s'],'o',color='#1464a0')
    ax.set(xlabel='CPU threads',ylabel='Time in solver steps (s)',title=f'Jiang-Cao flume: CPU core-count benchmark; selected {winner} threads')
    ax.set_xticks([1,2,4,8,16,32]);ax.set_ylim(bottom=0);fig.savefig(PROJECT/'CORE_BENCHMARK.png',dpi=160);fig.savefig(REPORT/'CORE_BENCHMARK.png',dpi=160);plt.close(fig)
    print('Selected',winner,'CPU threads',flush=True)

def prepare():
    n=read(PROJECT/'CORE_BENCHMARK.json')['selected_threads']
    invoke([PROJECT/'worker.py',PROJECT/'runtime/braids_core/river_launcher.py','prepare',
        '--project',PROJECT/'braids_project.json','--output',PROJECT/'prepared/experiment225',
        '--asset-cache',PROJECT/'cache'],PROJECT/'logs/prepare_experiment225.log',n)
    print('Full experiment prepared',flush=True)

def gpu_benchmark():
    n=read(PROJECT/'CORE_BENCHMARK.json')['selected_threads'];rows=[]
    for label in ('gpu_a','gpu_b'):
        invoke([ROOT/'desktop/experiment225.py','trial','--backend','gpu','--threads',n,'--label',label,'--end','.25'],
            PROJECT/'benchmarks'/(label+'.log'),n)
        r=read(PROJECT/'benchmarks'/label/'TIMING.json');rows.append(r)
        print(label,round(r['step_s'],3),'s stepping',flush=True)
    cpu=read(PROJECT/'CORE_BENCHMARK.json');cpu_time=statistics.median(r['step_s'] for r in cpu['rows'] if r['threads']==n)
    import numpy as np
    ref=Path(cpu['rows'][0]['output'])/'frames';ix=read(ref/'INDEX.json');differences=[]
    with np.load(ref/ix['frames'][-1]['file']) as a:
        for r in rows:
            path=Path(r['output'])/'frames';i=read(path/'INDEX.json')
            with np.load(path/i['frames'][-1]['file']) as b:
                bed=float(abs(a['z']-b['z']).max()*1000);depth=float(abs(a['q'][0]-b['q'][0]).max()*1000)
                assert max(bed,depth)<.1,(bed,depth)
                differences.append(dict(label=r['label'],bed_mm=bed,depth_mm=depth))
    gpu_time=statistics.median(r['step_s'] for r in rows)
    record=dict(cpu_threads=n,cpu_step_s=cpu_time,gpu_step_s=gpu_time,cpu_to_gpu_speedup=cpu_time/gpu_time,
        rows=rows,field_differences=differences,scope='Short actual-flume startup comparison; no automatic backend switch')
    write(PROJECT/'BACKEND_BENCHMARK.json',record);write(REPORT/'BACKEND_BENCHMARK.json',record)
    print(json.dumps(record),flush=True)

def render(output):
    index=read(output/'frames/INDEX.json')
    if not index or not index['frames']:return
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    row=index['frames'][-1];folder=output/'frames'
    with np.load(folder/row['file']) as f:
        at=float(f['time']);h=f['q'][0].copy();dz=(f['z']-f['initial_bed_m']).copy()
    with np.load(folder/row['epoch']) as mesh:
        v=mesh['vertices'];o=mesh['offsets'];polys=[v[o[i]:o[i+1]].copy() for i in range(len(o)-1)]
    fig,axs=plt.subplots(2,1,figsize=(13,4.5),layout='constrained')
    for ax,value,limit,cmap,label in ((axs[0],np.ma.masked_where(h<.001,h),(0,.03),'Blues','Water depth (m)'),
            (axs[1],dz,(-.03,.03),'RdBu_r','Bed change (m)')):
        p=PolyCollection(polys,array=value,cmap=cmap,clim=limit,edgecolors='none')
        ax.add_collection(p);ax.set(xlim=(0,25),ylim=(0,1.5),xlabel='x (m)',ylabel='y (m)',aspect='equal')
        fig.colorbar(p,ax=ax,label=label,shrink=.8)
    fig.suptitle(f'Jiang-Cao Case 1 reconstruction · MF1 · t = {at:.2f} s ({at/60:.2f} min)')
    temp=PROJECT/'.latest.png';fig.savefig(temp,dpi=160);plt.close(fig);temp.replace(PROJECT/'latest.png')
    return at

def dashboard(state):
    def esc(x):return html.escape(str(x))
    eta=state.get('estimated_finish_local') or 'Estimating from accepted progress…'
    cards=[('Status',state['status']),('Simulation',f"{state.get('time_s',0):.2f} / 18000 s"),
        ('Progress',f"{state.get('percent',0):.3f}%"),('Backend',state.get('backend','cpu').upper()+f" · CPU threads {state['threads']}"),
        ('Time controls',f"CFL {state.get('cfl','?')} · outer max {state.get('dt_max_s','?')} s"),
        ('Elapsed',f"{state.get('wall_s',0)/3600:.2f} hours"),('Estimated finish',eta)]
    image='<img src="latest.png" alt="Latest published water depth and bed change">' if (PROJECT/'latest.png').exists() else '<p>The first result frame is being prepared.</p>'
    body=''.join(f'<div class="card"><small>{esc(k)}</small><strong>{esc(v)}</strong></div>' for k,v in cards)
    text=f'''<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="10"><title>BRAIDS experiment progress</title>
<style>body{{font:16px Segoe UI,Arial;background:#f4f7fa;color:#19324a;margin:28px}}h1{{font-size:25px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.card{{background:white;border:1px solid #d4dce4;padding:16px}}small,strong{{display:block}}strong{{margin-top:8px}}img{{width:100%;background:white;margin-top:20px}}.note{{color:#526579}}progress{{width:100%;height:22px;margin:18px 0}}</style></head>
<body><h1>Jiang–Cao experiment · MF1</h1><p class="note">Updated {esc(state['updated_local'])} · Europe/Zurich · refreshes every 10 seconds</p><div class="grid">{body}</div>
<progress max="100" value="{state.get('percent',0)}"></progress><p>{esc(state.get('phase',''))}</p>{image}
<p class="note">The image updates when the solver publishes a frame (every 60 simulated seconds). ETA changes as the channel wets and morphology develops.</p>
<p>{esc(state.get('error') or '')}</p><p>All inputs, results, logs, checkpoints and the frozen runtime are in this project folder.</p></body></html>'''
    write(PROJECT/'progress.html',text)

def supervise():
    config=read(PROJECT/'CORE_BENCHMARK.json');n=config['selected_threads']
    project=read(PROJECT/'braids_project.json');backend=project['model']['request']['execution']['backend']
    cfl=project['settings']['controls']['cfl'];dt_max=project['settings']['dt_max_s']
    output=PROJECT/'runs/experiment225';assert not output.exists(),'Preserve existing run'
    guard=PROJECT/'RUN_LOCK.json'
    with guard.open('x',encoding='utf8') as f:json.dump(dict(supervisor_pid=os.getpid(),started_utc=dt.datetime.now(dt.timezone.utc).isoformat()),f)
    command=[str(PYTHON),'-X','utf8',str(PROJECT/'worker.py'),str(PROJECT/'runtime/braids_core/river_launcher.py'),
        'run','--input',str(PROJECT/'prepared/experiment225'),'--output',str(output),
        '--checkpoint-dir',str(PROJECT/'checkpoints/experiment225'),'--stop-file',str(PROJECT/'STOP.request')]
    start=time.monotonic();samples=[];last_image=None
    with (PROJECT/'logs/experiment225.log').open('x',encoding='utf8') as log:
        child=subprocess.Popen(command,cwd=PROJECT,env=environment(n),stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        write(PROJECT/'RUN_PROCESS.json',dict(supervisor_pid=os.getpid(),worker_pid=child.pid,command=command,
            started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),threads=n,backend=backend,cfl=cfl,dt_max_s=dt_max))
        with (PROJECT/'progress_history.csv').open('x',encoding='utf8',newline='') as stream:
            history=csv.writer(stream);history.writerow(['UTC','wall_s','simulation_s','status'])
            while True:
                code=child.poll();p=read(output/'PROGRESS.json',{}) or {};now=dt.datetime.now(ZONE);wall=time.monotonic()-start
                at=float(p.get('time_s',0.));samples.append((wall,at));samples=[x for x in samples if wall-x[0]<=300]
                speed=(at-samples[0][1])/(wall-samples[0][0]) if len(samples)>1 and wall>samples[0][0] else 0.
                eta=(now+dt.timedelta(seconds=(18000-at)/speed)).strftime('%Y-%m-%d %H:%M:%S %Z') if speed>0 else None
                status='running' if code is None else ('complete' if code==0 and read(output/'STATUS.json',{}).get('status')=='complete' else read(output/'STATUS.json',{}).get('status','failed'))
                state=dict(status=status,phase=p.get('phase','Loading frozen solver'),threads=n,backend=backend,cfl=cfl,dt_max_s=dt_max,time_s=at,
                    percent=100*at/18000.,wall_s=wall,estimated_finish_local=eta if code is None else None,
                    updated_local=now.strftime('%Y-%m-%d %H:%M:%S %Z'),supervisor_pid=os.getpid(),worker_pid=child.pid,
                    error=read(output/'STATUS.json',{}).get('error') if code is not None else None)
                index=read(output/'frames/INDEX.json');new=index['frames'][-1]['file'] if index and index['frames'] else None
                if new and new!=last_image:
                    try:state['latest_frame_s']=render(output);last_image=new
                    except Exception as exc:state['preview_error']=str(exc)
                write(PROJECT/'RUN_STATE.json',state);dashboard(state)
                history.writerow([now.astimezone(dt.timezone.utc).isoformat(),wall,at,status]);stream.flush()
                if code is not None:
                    write(PROJECT/'FINISHED.txt',f"{status.upper()}\n{state['updated_local']}\nSimulation time {at:g} s\nWorker exit code {code}\n{state.get('error') or ''}\n")
                    write(PROJECT/'COMPLETION.json',dict(state,exit_code=code,diagnostics=read(output/'DIAGNOSTICS.json')))
                    break
                time.sleep(10)

def monitor():
    while True:
        state=read(PROJECT/'RUN_STATE.json',{}) or {}
        if os.name=='nt':os.system('cls')
        print('BRAIDS Jiang-Cao experiment — MF1\n',PROJECT,'\n',flush=True)
        if state:
            for k in ('status','phase','backend','threads','cfl','dt_max_s','time_s','percent','wall_s','estimated_finish_local','updated_local','error'):
                if state.get(k) is not None:print(f'{k}: {state[k]}')
            if state.get('status')!='running':
                if os.name=='nt':
                    import winsound
                    winsound.MessageBeep()
                input('\nFinished. Press Enter to close.');return
        else:print('Waiting for the run to start…')
        time.sleep(10)

def install_controls():
    import shutil
    shutil.copy2(ROOT/'desktop/experiment_control225.py',PROJECT/'experiment_control.py')
    write(PROJECT/'CONTROL.json',dict(repository=str(ROOT),python=str(PYTHON)))
    (PROJECT/'Open progress.cmd').write_text('@echo off\nstart "" "%~dp0progress.html"\n',encoding='utf8')
    (PROJECT/'Check progress.cmd').write_text('@echo off\n"'+str(PYTHON)+'" -X utf8 "%~dp0experiment_control.py" monitor\n',encoding='utf8')
    (PROJECT/'Stop safely.cmd').write_text('@echo off\ntype nul > "%~dp0STOP.request"\necho Stop requested. The solver will save a checkpoint before exiting.\npause\n',encoding='utf8')
    project=read(PROJECT/'braids_project.json');backend=project['model']['request']['execution']['backend']
    dashboard(dict(status='ready',phase='Prepared; waiting for launch',backend=backend,
        cfl=project['settings']['controls']['cfl'],dt_max_s=project['settings']['dt_max_s'],
        threads=read(PROJECT/'CORE_BENCHMARK.json')['selected_threads'],updated_local=dt.datetime.now(ZONE).strftime('%Y-%m-%d %H:%M:%S %Z')))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['benchmark','gpu_benchmark','prepare','supervise','monitor','controls'])
    args=parser.parse_args();{'benchmark':benchmark,'gpu_benchmark':gpu_benchmark,'prepare':prepare,
        'supervise':supervise,'monitor':monitor,'controls':install_controls}[args.action]()
