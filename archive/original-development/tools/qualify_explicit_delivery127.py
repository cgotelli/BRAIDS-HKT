"""Compiler-free natural-river replay against the frozen qualified explicit run."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def frame(folder,index=-1):
    rows=json.loads((folder/'frames/INDEX.json').read_text())['frames']
    with np.load(folder/'frames'/rows[index]['file'],allow_pickle=False) as f:
        return {k:f[k] for k in ('q','z','mass')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--plugin',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    runtime=a.plugin.resolve()/'river_runtime';out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    executable=Path(sys.executable).resolve()
    env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(runtime),
        PATH=os.pathsep.join([str(executable.parent/'Library/bin'),str(executable.parent),
            os.environ.get('SystemRoot','C:/Windows')+'/System32']))
    for k in ('PYTHONHOME','QGIS_PREFIX_PATH','QT_PLUGIN_PATH','BRAIDS_TEST_TOOLCHAIN'):env.pop(k,None)
    launcher=runtime/'braids_core/river_launcher.py';results={}
    for size in ('1m','5m'):
        for backend in ('cpu','gpu'):
            name=f'{size}_{backend}';root=out/name;root.mkdir()
            project=ROOT/'validation/_delivery127/projects'/f'input_{size}_{backend}.json'
            times={}
            for action,args in [('prepare',['--project',project,'--output',root/'prepared']),
                                ('run',['--input',root/'prepared','--output',root/'run'])]:
                started=time.perf_counter()
                r=subprocess.run([str(executable),str(launcher),action,*map(str,args)],cwd=out,env=env,
                    capture_output=True,text=True,timeout=300)
                (root/(action+'.log')).write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
                if r.returncode:raise RuntimeError(name+' '+action+' '+r.stdout+r.stderr)
                times[action]=time.perf_counter()-started
            status=json.loads((root/'run/STATUS.json').read_text())
            assert status['status']=='complete' and status['integrator']=='explicit_fwave_ssprk22_river_v1'
            old=ROOT/'validation/_resistance126'/f'{size}_explicit_{backend}_cfl0.225'/'run'
            rows=json.loads((root/'run/frames/INDEX.json').read_text())['frames']
            before=json.loads((old/'frames/INDEX.json').read_text())['frames'];assert len(rows)==len(before)
            with np.load(root/'run/frames'/rows[0]['epoch']) as current_mesh,np.load(old/'frames'/before[0]['epoch']) as reference_mesh:
                for key in ('vertices','offsets','area','leaves'):
                    np.testing.assert_array_equal(current_mesh[key],reference_mesh[key])
            errors=[]
            for i in range(len(rows)):
                new=frame(root/'run',i);ref=frame(old,i)
                error={k:float(np.max(abs(new[k]-ref[k]))) for k in new};errors.append(error)
                for k in new:np.testing.assert_allclose(new[k],ref[k],atol=1e-10,rtol=0)
            diag=json.loads((root/'run/DIAGNOSTICS.json').read_text())
            assert max(abs(r['water_residual_m3']) for r in diag)<1e-10
            assert all(not any(r['wall_integrals']) for r in diag)
            results[name]=dict(times=times,timing_scope='deployment check, not an isolated performance benchmark',
                status=status,frame_errors=errors,reference=str(old),run=str(root/'run'))
            (out/'CHECKS.json').write_text(json.dumps(results,indent=2));print(name,'passed',times,flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,size in zip(axes,('1m','5m')):
        for backend in ('cpu','gpu'):
            row=results[f'{size}_{backend}'];q=frame(Path(row['run']))['q'][0]
            ref=frame(Path(row['reference']))['q'][0]
            wet=(ref>0)|(q>0)
            ax.plot(ref[wet],q[wet],'.',ms=2,label=backend.upper())
        upper=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([0,upper],[0,upper],'k--',lw=1,label='Equal depth')
        ax.set(title=f'{size} natural river at 4 s',xlabel='Qualified reference depth (m)',ylabel='Packaged depth (m)');ax.legend()
    fig.savefig(out/'PACKAGED_DEPTH_COMPARISON.png',dpi=170);plt.close(fig)
    from matplotlib.collections import PolyCollection
    run=Path(results['1m_gpu']['run']);rows=json.loads((run/'frames/INDEX.json').read_text())['frames']
    with np.load(run/'frames'/rows[-1]['epoch'],allow_pickle=False) as g:
        polygons=[g['vertices'][g['offsets'][i]:g['offsets'][i+1]].copy() for i in range(len(g['area']))]
    q=frame(run)['q'][0];ref=frame(Path(results['1m_gpu']['reference']))['q'][0]
    wet=(q>.001)|(ref>.001);origin=np.min(np.concatenate(polygons),axis=0)
    visible=[p-origin for p,selected in zip(polygons,wet) if selected]
    fig,axes=plt.subplots(1,3,figsize=(11,5),constrained_layout=True)
    for ax,values,title in zip(axes,(ref[wet],q[wet],1000*(q-ref)[wet]),
            ('Qualified reference','Packaged GPU','Difference (mm)')):
        c=PolyCollection(visible,array=values,cmap='Blues' if title!='Difference (mm)' else 'RdBu_r',edgecolors='none')
        c.set_clim((0,float(max(ref.max(),q.max()))) if title!='Difference (mm)' else (-.001,.001))
        ax.add_collection(c);ax.autoscale();ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('Local east (m)')
        fig.colorbar(c,ax=ax,label='Depth (m)' if title!='Difference (mm)' else 'Difference (mm)')
    axes[0].set_ylabel('Local north (m)');fig.savefig(out/'NATURAL_RIVER_DEPTH.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
