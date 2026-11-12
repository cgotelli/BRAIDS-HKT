"""Compare unsmoothed CV velocities; keep near/far metrics separate."""
from nearwake60 import *
import csv,numpy as np
from scipy.spatial import Delaunay
import matplotlib;matplotlib.use('Agg')
from matplotlib import pyplot as plt
from braids_desktop.results import ResultsStore

def model(name,start=1500.,end=2000.):
    receipt=read(REPORT/name/('RECEIPT.json' if (REPORT/name/'RECEIPT.json').exists() else 'STARTED.json'))
    store=ResultsStore(Path(receipt['output']));ids=np.flatnonzero((store.times>=start-1e-8)&(store.times<=end+1e-8))
    if len(ids)<2:raise ValueError(f'{name}: only {len(ids)} samples in requested window; last={store.times[-1]}')
    states=[]
    for idx in ids:
        f=store.read(int(idx));states.append(np.array([f.q[0],f.q[1]/f.q[0],f.q[2]/f.q[0]]))
    xy=np.array([p.centroid.coords[0] for p in f.polygons]);tri=Delaunay(xy)
    state=np.mean(states,axis=0);vel=np.mean([np.hypot(s[1],s[2]) for s in states],axis=0)
    return dict(name=name,xy=xy,tri=tri,state=state,speed=vel,times=store.times[ids],states=states,output=receipt['output'],area=f.mesh['area'])

def cached(name):
    p=np.load(PARENT/name/'TIME_AVERAGE.npz');xy=p['centers_xy'];q=np.array([p['h'],p['u'],p['v']])
    return dict(name=name,xy=xy,tri=Delaunay(xy),state=q,speed=np.hypot(q[1],q[2]),times=p['times_s'],area=p['area_m2'])

def sample(case,x,y,field='speed'):
    x,y=np.broadcast_arrays(x,y);p=np.column_stack((500+10*x.ravel(),295+10*y.ravel()));tri=case['tri'];idx=tri.find_simplex(p)
    assert (idx>=0).all()
    b=np.einsum('nij,nj->ni',tri.transform[idx,:2],p-tri.transform[idx,2]);w=np.column_stack((b,1-b.sum(1)))
    return np.sum(case[field][tri.simplices[idx]]*w,axis=1).reshape(x.shape)

def deficit(case,control,x,y):return 1-sample(case,x,y)/sample(control,x,y)
def metric(p,t):
    e=100*(p-t);return dict(RMS_pp=float(np.sqrt(np.mean(e*e))),max_pp=float(np.max(abs(e))),bias_pp=float(np.mean(e)))

def analyze(names,start=1500.,end=2000.,label='COMPARISON'):
    cases=[cached('nash_gpu')]+[model(n,start,end) for n in names];control=cached('zero_gpu')
    references=list(csv.DictReader((ROOT/'validation/hydrokinetic_turbines/measurements/nash_2015_figure6b.csv').open()))
    refs={k:np.array([[float(r['x_over_d']),float(r['velocity_deficit'])] for r in references if r['series']==k]) for k in ['published_2d_model','experiment_depth_average']}
    result={};fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    xx=np.linspace(2.5,20,176)
    for c in cases:
        axes[0].plot(xx,100*deficit(c,control,xx,0.),label=c['name'])
        result[c['name']]=dict(samples=c['times'].tolist(),cells=len(c['xy']),comparisons={})
        for k,r in refs.items():
            p=deficit(c,control,r[:,0],0.);result[c['name']]['comparisons'][k]={}
            for name,mask in [('near_2p5_to4D',r[:,0]<=4),('far_ge4D',r[:,0]>=4),('all',r[:,0]>=0)]:
                result[c['name']]['comparisons'][k][name]=metric(p[mask],r[mask,1])
        for ax,x in zip(axes[1:],[4.,10.]):
            yy=np.linspace(-3,3,121);ax.plot(100*deficit(c,control,x,yy),yy,label=c['name'])
    for k,r in refs.items():axes[0].plot(r[:,0],100*r[:,1],'--',label=k,color='black' if k.startswith('published') else '#37994b')
    lateral=list(csv.DictReader((ROOT/'validation/hydrokinetic_turbines/measurements/nash_2015_figure7.csv').open()))
    for ax,x in zip(axes[1:],[4.,10.]):
        rows=[r for r in lateral if float(r['x_over_d'])==x]
        ax.plot([100*float(r['velocity_deficit']) for r in rows],[float(r['y_over_d']) for r in rows],'k--',label='Nash 2D')
        ax.set(title=f'{x:g}D lateral (published model)',xlabel='Velocity deficit (%)',ylabel='y/D',ylim=(-3,3))
    axes[0].set(title='Authors’ depth-average experiment + 2D model',xlabel='x/D',ylabel='Velocity deficit (%)',ylim=(0,27))
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle(f'Unchanged thrust law and viscosity; {start:g}–{end:g} s averages (old control interpolation for diagnostic only)')
    fig.savefig(REPORT/(label+'.png'),dpi=170);plt.close(fig)
    save(REPORT/(label+'.json'),result);print(json.dumps(result,indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('names',nargs='+');a.add_argument('--start',type=float,default=1500);a.add_argument('--end',type=float,default=2000);a.add_argument('--label',default='COMPARISON');v=a.parse_args();analyze(v.names,v.start,v.end,v.label)
