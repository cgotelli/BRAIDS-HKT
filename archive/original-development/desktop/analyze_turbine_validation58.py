"""Analyze authenticated public Desktop output against published, frozen curves."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'desktop/runtime';HOME=Path.home()/'Desktop/BRAIDS_Turbine_Validation58'
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/desktop_validation58'
sys.path.insert(0,str(RUNTIME));sys.path.insert(1,str(ROOT))
from braids_desktop.results import ResultsStore
from validation.hydrokinetic_turbines.digitize_hkt3j import otf,operating,performance


def measured_low_ti():
    path=HOME/'02_Mycek_08_CPU/references/mycek_2014_figure_A12_velocity.csv'
    rows=[r for r in csv.DictReader(path.open(encoding='utf-8')) if float(r['ambient_turbulence_percent'])==3.]
    result={}
    for x in sorted({float(r['x_over_d']) for r in rows}):
        data=sorted([r for r in rows if float(r['x_over_d'])==x],key=lambda r:float(r['y_over_d']))
        result[x]=dict(y=np.array([float(r['y_over_d']) for r in data]),u=np.array([float(r['mean_u_over_u_infinity']) for r in data]),std=np.array([float(r['std_u_over_u_infinity']) for r in data]))
    return result


def axisymmetric_depth_average(profile,y_values):
    radius=np.round(abs(profile['y']),6);radii=np.unique(radius)
    mean=np.array([profile['u'][radius==r].mean() for r in radii]);sigma=np.array([profile['std'][radius==r].mean() for r in radii])
    if radii[0]>1e-8:radii=np.r_[0.,radii];mean=np.r_[mean[0],mean];sigma=np.r_[sigma[0],sigma]
    # Explicitly assumed radial symmetry through the full 2 m water column.
    # This is the retained surrogate, never labelled as a measured depth mean.
    half=2./(2*.7);z=np.linspace(-half,half,2001)
    def integrate(values):return np.array([np.trapezoid(np.interp(np.sqrt(y*y+z*z),radii,values,left=values[0],right=1.),z)/(2*half) for y in np.asarray(y_values)])
    return integrate(mean),integrate(mean-sigma),integrate(mean+sigma)


def save(path,data):Path(path).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
def norms(error):
    e=np.asarray(error);return dict(RMSE_pp=float(100*np.sqrt(np.mean(e*e))),maximum_absolute_pp=float(100*np.max(abs(e))))
def export(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


class Case:
    def __init__(self,name):
        self.name=name;self.receipt=json.loads((REPORT/name/'RECEIPT.json').read_text());self.output=Path(self.receipt['output'])
        self.c=json.loads((HOME/'PROJECTS.json').read_text())[name]['configuration'];self.store=ResultsStore(self.output)
        assert len(self.store.mesh_history)==1
        self.indices=np.flatnonzero(self.store.times>=self.c['average']-1e-8)
        assert len(self.indices)>=4
        frame=self.store.read(int(self.indices[0]));self.polygons=frame.polygons
        self.xy=np.array([p.centroid.coords[0] for p in frame.polygons]);self.tri=Delaunay(self.xy)
        self.area=frame.mesh['area'];self.mean=np.zeros_like(frame.q);self.history=[];self.sample_cache={}
        for i in self.indices:
            frame=self.store.read(int(i));q=frame.q;state=np.array([q[0],q[1]/q[0],q[2]/q[0]])
            self.mean+=state;self.history.append((frame.time_s,state))
        self.mean/=len(self.indices)
        raw=(self.output/'TURBINE_PERFORMANCE_INDEX.json').read_bytes();index=json.loads(raw)
        for n,h in index['files'].items():assert hashlib.sha256((self.output/n).read_bytes()).hexdigest()==h
        self.rows=[r for r in json.loads((self.output/'TURBINE_PERFORMANCE.json').read_text()) if r['time_s']>=self.c['average']-1e-8]
        assert self.rows and all(r['active'] for r in self.rows)
        np.savez_compressed(REPORT/name/'TIME_AVERAGE.npz',h=self.mean[0],u=self.mean[1],v=self.mean[2],centers_xy=self.xy,area_m2=self.area,times_s=self.store.times[self.indices])

    def sample(self,x_over_d,y_over_d,state=None):
        y=np.asarray(y_over_d);points=np.column_stack((np.full(y.size,self.c['X']+x_over_d*self.c['D']),self.c['Y']+y*self.c['D']))
        key=points.tobytes()
        if key not in self.sample_cache:
            ids=self.tri.find_simplex(points);assert np.all(ids>=0)
            b=np.einsum('nij,nj->ni',self.tri.transform[ids,:2,:],points-self.tri.transform[ids,2,:])
            self.sample_cache[key]=(self.tri.simplices[ids],np.column_stack((b,1-b.sum(1))))
        ids,w=self.sample_cache[key];v=self.mean[1] if state is None else state[1]
        return np.sum(v[ids]*w,axis=1)

    def stationarity(self,stations,y):
        mid=(self.c['average']+self.c['end'])/2
        first=np.mean([s for t,s in self.history if t<=mid],axis=0);second=np.mean([s for t,s in self.history if t>mid],axis=0)
        return max(float(100*np.max(abs(self.sample(x,y,first)-self.sample(x,y,second)))/self.c['U']) for x in stations)

    def loads(self):
        c=self.c;A=np.pi*c['D']**2/4;F=float(np.mean([r['thrust_N'] for r in self.rows]));P=float(np.mean([r['useful_power_W'] for r in self.rows]))
        y=np.linspace(-.5,.5,81);up=float(np.trapezoid(self.sample(-1.,y),y))
        ct=F/(.5*1000*A*c['U']**2);cp=P/(.5*1000*A*c['U']**3)
        ref=operating(c['U']) if self.name!='otf_gpu' else dict(Ct=.85,Cp=.85*.5*(1+np.sqrt(1-.85)))
        if self.name=='zero_cpu':ref=dict(Ct=0.,Cp=0.)
        diagnostic=json.loads((self.output/'DIAGNOSTICS.json').read_text())
        return dict(thrust_N=F,mechanical_power_W=P,input_Ct=float(self.rows[-1]['thrust_coefficient']),input_Cp=float(self.rows[-1]['power_coefficient']),
            integrated_nominal_Ct=ct,integrated_nominal_Cp=cp,reference_Ct=ref['Ct'],reference_Cp=ref['Cp'],
            relative_Ct_error=None if ref['Ct']==0 else ct/ref['Ct']-1,relative_Cp_error=None if ref['Cp']==0 else cp/ref['Cp']-1,
            independent_1D_strip_velocity_m_s=up,integrated_1D_strip_Ct=F/(.5*1000*A*up**2),integrated_1D_strip_Cp=P/(.5*1000*A*up**3),
            maximum_water_residual_m3=max(abs(r['water_residual_m3']) for r in diagnostic),
            relative_water_throughput=max(abs(r['water_residual_m3']) for r in diagnostic)/(c['U']*c['W']*c['H']*c['end']))


def analyze(selected=None):
    keys=selected or list(json.loads((HOME/'PROJECTS.json').read_text()))
    cases={n:Case(n) for n in keys if (REPORT/n/'RECEIPT.json').is_file()};metrics={};csvrows=[]
    if 'otf_gpu' in cases:
        a=cases['otf_gpu'];fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');errors=[]
        for ax,x in zip(axes,[6.,12.]):
            ys=np.linspace(-2,2,161);ax.plot((1-a.sample(x,ys)/a.c['U'])*100,ys,label='Installed BRAIDS Desktop',color='#236da2')
            ref=sorted([r for r in otf() if r['model']=='OpenTidalFarm' and r['viscosity_m2_s']==.6 and r['x_over_d']==x],key=lambda r:r['y_over_d'])
            y=np.array([r['y_over_d'] for r in ref]);target=np.array([r['deficit'] for r in ref]);pred=1-a.sample(x,y)/a.c['U'];errors.extend(pred-target)
            ax.plot(target*100,y,'k--',label='Published OpenTidalFarm 2D');ax.set(xlabel='Axial velocity deficit (%)',ylabel='y/D',title=f'{x:g}D downstream');ax.grid(alpha=.2);ax.legend(fontsize=8)
            for yy,p,t in zip(y,pred,target):csvrows.append(dict(case='OTF',station_D=x,y_D=yy,braids_deficit=p,reference_deficit=t,reference_kind='Published OTF 2D'))
            metrics[f'otf_{x:g}D']=norms(pred-target)
        fig.suptitle('SWE model comparison: U=2.9 m/s, nu=0.6 m²/s, Ct=0.85; no fitted loading');fig.savefig(REPORT/'PUBLISHED_2D_WAKE.png',dpi=170);plt.close(fig)
        metrics['otf_pooled']=norms(errors);metrics['otf_stationarity_pp']=a.stationarity([6.,12.],np.linspace(-2,2,81))
    if 'mycek_cpu' in cases:
        a=cases['mycek_cpu'];fig,axes=plt.subplots(1,4,figsize=(14,4),layout='constrained');allerr=[];far=[];raw=[]
        for x,p in measured_low_ti().items():
            y=p['y'];pred=a.sample(x,y)/a.c['U'];target,low,high=axisymmetric_depth_average(p,y)
            allerr.extend(pred-target);raw.extend(pred-p['u'])
            if x>=3:far.extend(pred-target)
            for yy,b,t,hub in zip(y,pred,target,p['u']):csvrows.append(dict(case='Mycek',station_D=x,y_D=yy,braids_deficit=1-b,reference_deficit=1-t,reference_kind='Assumed depth projection of measured hub profile'))
            if x in [1.2,3.,6.,10.]:
                ax=axes[[1.2,3.,6.,10.].index(x)];ax.plot((1-pred)*100,y,label='BRAIDS depth mean',color='#236da2')
                ax.plot((1-p['u'])*100,y,':',color='.5',label='Measured hub plane (raw)')
                ax.plot((1-target)*100,y,'--',color='#cb7629',label='Assumed depth projection')
                ax.fill_betweenx(y,(1-high)*100,(1-low)*100,color='#cb7629',alpha=.15,label='Projected plotted SD')
                ax.set(title=f'{x:g}D downstream',xlabel='Axial velocity deficit (%)',ylabel='y/D',ylim=(-1.75,1.75));ax.grid(alpha=.2)
        axes[0].legend(fontsize=7);fig.suptitle('Mycek flume: raw measurements retained; the projected curve is NOT a measured depth mean')
        fig.savefig(REPORT/'EXPERIMENTAL_WAKE.png',dpi=170);plt.close(fig)
        metrics.update(mycek_projected_all=norms(allerr),mycek_projected_ge_3D=norms(far),mycek_raw_hub_not_like_for_like=norms(raw),
            mycek_stationarity_pp=a.stationarity([3.,6.,10.],np.linspace(-1.5,1.5,81)))
        if 'mycek_gpu' in cases:
            b=cases['mycek_gpu'];np.testing.assert_allclose(a.xy,b.xy,atol=0,rtol=0)
            metrics['backend']=dict(maximum_mean_depth_m=float(np.max(abs(a.mean[0]-b.mean[0]))),maximum_mean_velocity_m_s=float(np.max(abs(a.mean[1:]-b.mean[1:]))))
        if 'zero_cpu' in cases:
            b=cases['zero_cpu'];assert b.loads()['thrust_N']==b.loads()['mechanical_power_W']==0.
            metrics['zero_control_maximum_speed_departure_percent']=float(100*np.max(abs(b.mean[1]-b.c['U']))/b.c['U'])
    for name,a in cases.items():metrics[name+'_loads']=a.loads()
    if 'mycek_cpu' in cases:
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,key in zip(axes,['Ct','Cp']):
            for U,col in [(.8,'#5d9969'),(1.,'#d4934c')]:
                r=[r for r in performance() if r['field']==key and r['inflow_m_s']==U]
                ax.plot([r['TSR'] for r in r],[r['coefficient'] for r in r],'-o',ms=3,color=col,label=f'Measured U={U:g} m/s')
            for n,marker in [('mycek_cpu','o'),('mycek_holdout_cpu','X')]:
                if n in cases:
                    ax.scatter([3.67],[metrics[n+'_loads']['integrated_nominal_'+key]],s=85,marker=marker,color='#236da2',label=f'BRAIDS U={cases[n].c["U"]:g} m/s')
            ax.set(xlabel='TSR',ylabel=key,ylim=(0,1 if key=='Ct' else .6));ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle('Integrated loading/power: 0.8 m/s table supplied; 1.0 m/s measured curve held out')
        fig.savefig(REPORT/'CT_CP.png',dpi=170);plt.close(fig)
    if csvrows:export(REPORT/'WAKE_COMPARISON.csv',csvrows)
    for name,a in cases.items():
        fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
        axes[0].plot([r['time_s'] for r in a.rows],[r['thrust_N'] for r in a.rows]);axes[1].plot([r['time_s'] for r in a.rows],[r['useful_power_W'] for r in a.rows])
        axes[0].set(ylabel='Thrust (N)');axes[1].set(ylabel='Mechanical power (W)')
        for ax in axes:ax.set_xlabel('Time (s)');ax.set_ylim(bottom=0);ax.grid(alpha=.2)
        fig.suptitle(name+' — final averaging window');fig.savefig(REPORT/name/'LOAD_HISTORY.png',dpi=160);plt.close(fig)
    showing=[n for n in ['otf_gpu','mycek_cpu','zero_cpu'] if n in cases]
    if showing:
        fig,axes=plt.subplots(len(showing),1,figsize=(11,3*len(showing)),squeeze=False,layout='constrained')
        for ax,name in zip(axes[:,0],showing):
            a=cases[name];c=a.c;lower=c['X']-2*c['D'];upper=c['X']+13*c['D'];half=min(2*c['D'],.45*c['W'])
            mask=(a.xy[:,0]>=lower-c['dx'])&(a.xy[:,0]<=upper+c['dx'])&(abs(a.xy[:,1]-c['Y'])<=half+c['dy'])
            ids=np.flatnonzero(mask)
            polygons=[(np.asarray(a.polygons[i].exterior.coords)-[c['X'],c['Y']])/c['D'] for i in ids]
            artist=PolyCollection(polygons,array=a.mean[1,ids]/c['U'],cmap='viridis',clim=(.65,1.03),edgecolors='none')
            ax.add_collection(artist);ax.plot([0,0],[-.5,.5],color='#de4747',lw=2,label='Rotor plane')
            ax.set(xlim=(-2,13),ylim=(-half/c['D'],half/c['D']),xlabel='x/D from rotor',ylabel='y/D',title=name,aspect='equal')
            fig.colorbar(artist,ax=ax,label='Mean axial velocity / nominal inflow');ax.legend(loc='upper right',fontsize=7)
        fig.suptitle('Actual cell means: extraction wake and matched zero-thrust control')
        fig.savefig(REPORT/'WAKE_MAPS.png',dpi=170);plt.close(fig)
    save(REPORT/'METRICS.json',metrics);print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':analyze(sys.argv[1:] or None)
