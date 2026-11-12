"""Auditable corner momentum decomposition and bounded candidate qualification."""
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from check_bedload161 import sha,save
from review_hydraulics162 import load,norm,verify_native
from review_groin163 import budgets
from braids_desktop.results import ResultsStore
from corner_candidate167 import VARIANTS
RAW=ROOT/'validation/_corner167'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_corner167'
NATIVE=ROOT/'validation/_hydraulics162/mobile/delft_mpm'
TITLES={'baseline':'Installed baseline','tangent':'Tangential extension',
        'normal':'Normal fit + tangential extension','lswall':'Linear near walls',
        'normal_ls':'Normal fit + linear near walls'}


def decompose(a,point):
    """Exact interior/wall residual, physical central flux plus f-wave remainder.

    The remainder is a numerical interface correction, NOT a pure positive
    viscosity. In a nonuniform state it can accelerate one momentum component.
    No open face or drained flux is admitted to this diagnostic stencil.
    """
    c=int(np.argmin(((a['xy']-point)**2).sum(1)))
    faces=np.flatnonzero((a['owner']==c)|(a['neighbour']==c))
    rows=[];rates=np.zeros((4,3));maximum=0.
    for f in faces:
        s=1 if a['owner'][f]==c else -1;side=0 if s==1 else 1;n=a['normal'][:,f]
        assert a['neighbour'][f]>=0 or a['bid'][f]<0
        assert not a['rec_hydro_constant'][c]
        adv=[];pressure=[]
        for k in (0,1):
            h,mn,mt,_=a['local'][4*k:4*k+4,f];assert h>0
            uv=np.array([mn*n[0]-mt*n[1],mn*n[1]+mt*n[0]])/h
            adv.append(np.r_[mn,mn*uv]);pressure.append(np.r_[0.,.5*9.81*h*h*n])
        adv=np.mean(adv,axis=0);pressure=np.mean(pressure,axis=0)
        t=a['rec_trace'];corr=.5*9.81*(a['q'][0,c]+t[8+side,f])*(t[3+4*side,f]-a['z'][c])
        bed=np.r_[0.,corr*n];actual=a['fl' if side==0 else 'fr'][:,f]
        remainder=actual-adv-pressure-bed
        pieces=(-s*a['length'][f]/a['area'][c])*np.array([adv,pressure,remainder,bed])
        rates+=pieces
        # Independently rebuild from the raw augmented solver fluctuations.
        h,mn,mt,_=a['local'][4*side:4*side+4,f]
        raw=np.array([mn,mn*mn/h+.5*9.81*h*h,mn*mt/h])+s*a['riemann'][3*side:3*side+3,f]
        rebuilt=np.array([raw[0],raw[1]*n[0]-raw[2]*n[1],raw[1]*n[1]+raw[2]*n[0]])
        if a['neighbour'][f]<0:
            rebuilt[0]=0.;rebuilt[1:]=np.dot(rebuilt[1:],n)*n
        rebuilt+=bed;maximum=max(maximum,float(abs(rebuilt-actual).max()))
        rows.append(dict(face=int(f),mid=a['mid'][:,f].tolist(),normal=n.tolist(),
                         side=side,rate_pieces=pieces.tolist(),flux=actual.tolist()))
    residual=float(abs(rates.sum(0)-a['rate'][:,c]).max())
    assert maximum<1e-12 and residual<1e-12
    return dict(cell=c,xy=a['xy'][c].tolist(),pieces=rates.tolist(),actual_rate=a['rate'][:,c].tolist(),
                flux_reconstruction_max=maximum,residual_max=residual,faces=rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    decomposition={label:[decompose(np.load(RAW/f'baseline/gpu/{label}.npz'),xy)
        for xy in ([7.75,3.25],[8.25,3.25],[8.75,3.25],[8.25,3.75])]
        for label in ('BRAIDS','FLOW_PRIMARY')}
    operators={};manufactured={}
    for variant in ('baseline',)+VARIANTS:
        tag='baseline' if variant=='baseline' else 'final_'+variant
        operators[variant]={}
        for label in ('AFFINE','SHEAR','QUADRATIC','REST','BRAIDS','FLOW_PRIMARY'):
            a=np.load(RAW/f'{tag}/cpu/{label}.npz');b=np.load(RAW/f'{tag}/gpu/{label}.npz')
            wall=(a['neighbour']<0)&(a['bid']<0)
            interior=(a['xy'][:,0]>2)&(a['xy'][:,0]<38)
            row=dict(rate_parity=norm(a['rate']-b['rate']),water_parity=norm(a['water']-b['water']),
                     wall_flux=float(abs(a['water'][wall]).max()),
                     water_residual=float(abs(np.dot(a['rate'][0],a['area'])+
                         np.dot(a['water'][a['neighbour']<0],a['length'][a['neighbour']<0]))))
            assert row['rate_parity']['maximum']<1e-11 and row['wall_flux']==0 and row['water_residual']<1e-11
            if label=='REST':
                row['interior_rest_residual']=float(abs(a['rate'][:,interior]).max())
                assert row['interior_rest_residual']<1e-11
            operators[variant][label]=row
            if label=='SHEAR':
                mask=wall&np.isclose(a['mid'][1],3)&(a['mid'][0]>8)&(a['mid'][0]<10)
                manufactured[variant]=dict(wall_x=a['mid'][0,mask].tolist(),
                    wall_tangential=a['trace'][0,1,mask].tolist(),
                    normal_error=float(abs(a['trace'][0,2,mask]).max()),
                    tangent_error=float(abs(a['trace'][0,1,mask]-1.).max()),
                    first_row_gradient=a['grad'][1,:,472].tolist())
    data={};cases={}
    for variant in ('baseline',)+VARIANTS:
        path=ROOT/'validation/_shoal165/groin/mpm_cpu/run' if variant=='baseline' else RAW/f'runs/final_{variant}/cpu/run'
        a,r=load(path,NATIVE);data[variant]=a
        xy=a['xy'];region=(xy[:,0]>6)&(xy[:,0]<16);tip=472
        z=a['bz'][-1]-a['bz'][0];nz=a['nz'][-1]-a['nz'][0]
        gp=ROOT/'validation/_shoal165/groin/mpm_gpu/run' if variant=='baseline' else RAW/f'runs/final_{variant}/gpu/run'
        g=ResultsStore(gp)
        parity={k:0. for k in ('q','z','mass')};depth_parity=0.
        for i,f in enumerate(a['frames']):
            gf=g.read(i)
            for k in parity:parity[k]=max(parity[k],float(abs(f.arrays[k]-gf.arrays[k]).max()))
            depth_parity=max(depth_parity,float(abs(f.q[0]-gf.q[0]).max()))
        series=[]
        for j,t in enumerate(a['store'].times):
            dz=a['bz'][j]-a['bz'][0];ndz=a['nz'][j]-a['nz'][0]
            series.append(dict(t=float(t),tip_scour_mm=float(-dz[tip]*1000),native_tip_scour_mm=float(-ndz[tip]*1000),
                max_scour_mm=float(-dz.min()*1000),max_deposition_mm=float(dz.max()*1000)))
        peak=[]
        for x in (8.25,8.75,9.25,10.25):
            ids=np.flatnonzero(np.isclose(xy[:,0],x));ids=ids[xy[ids,1]>=3]
            peak.append(dict(x=x,braids_y=float(xy[ids[np.argmax(a['bu'][8,0,ids])],1]),
                             native_y=float(xy[ids[np.argmax(a['nu'][8,0,ids])],1])))
        cases[variant]=dict(initial_bed_max_m=r['initial_bed_max_m'],initial_depth_max_m=r['initial_depth_max_m'],
            h40_m=float(a['bh'][8,tip]),uv40_m_s=a['bu'][8,:,tip].tolist(),
            depth40_region=norm((a['bh'][8]-a['nh'][8])[region]),
            uv40_region=norm((a['bu'][8]-a['nu'][8])[:,region]),bed60=norm(z-nz),
            bed60_region=norm((z-nz)[region]),bed60_correlation=float(np.corrcoef(z,nz)[0,1]),
            tip_scour_mm=float(-z[tip]*1000),max_scour_mm=float(-z.min()*1000),
            max_scour_xy=xy[z.argmin()].tolist(),max_deposition_mm=float(z.max()*1000),
            max_deposition_xy=xy[z.argmax()].tolist(),series=series,jet_peaks=peak,
            cpu_gpu_max=parity,cpu_gpu_depth_max_m=depth_parity,
            strict_coupled_parity_screen=bool(parity['q']<1e-4 and parity['z']<1e-6),
            cpu=budgets(path),gpu=budgets(gp),
            cpu_status_sha256=sha(path.parent/'STATUS.json'),gpu_status_sha256=sha(gp.parent/'STATUS.json'))
    a=data['baseline'];native=dict(h40_m=float(a['nh'][8,472]),uv40_m_s=a['nu'][8,:,472].tolist(),
         tip_scour60_mm=float((a['nz'][0,472]-a['nz'][-1,472])*1000))
    with Dataset(NATIVE/'trim-sim_P057.nc') as d:
        masks={str(j):[int(d['KFV'][8,16,j]),int(d['KFV'][8,17,j])] for j in (6,7)}
    assert masks=={'6':[1,0],'7':[1,1]}
    momentum={}
    for label in ('native_momentum','native_momentum_fine'):
        receipt=verify_native(RAW/label)
        with Dataset(RAW/label/'trim-sim_P057.nc') as d,Dataset(NATIVE/'trim-sim_P057.nc') as ref:
            j=int(np.argmin(abs(d['time'][:]-40)))
            exact={k:bool(np.array_equal(np.asarray(d[k][j]),np.asarray(ref[k][8])))
                   for k in ('S1','U1','V1','DPS','SBCU','SBCV','MSED')}
            assert all(exact.values())
            exact_all={k:bool(np.array_equal(np.asarray(d[k][:]),np.asarray(ref[k][:])))
                       for k in exact} if label=='native_momentum' else None
            if exact_all is not None:assert all(exact_all.values())
            terms={k:np.asarray(d[k][j]).reshape(-1,82,18)[0,16:21,7].tolist()
                   for k in ('MOM_DUDT','MOM_UPRESSURE','MOM_UDUDX','MOM_VDUDY','MOM_UBEDSHEAR')}
            assert np.max(np.abs(terms['MOM_VDUDY']))==0
            momentum[label]=dict(runtime=receipt,unchanged_at40=exact,unchanged_all_frames=exact_all,
                 first_row_x_m=[8,8.5,9,9.5,10],
                 first_row_y_m=3.25,terms_m_s2=terms,inputs_sha256=sha(RAW/label/'INPUTS.json'))
    sources={}
    previous=json.loads((ROOT/'validation/sediment/river_morphodynamics/reports/desktop_scour166/AUDIT.json').read_text())
    with zipfile.ZipFile(ROOT/'resources/Delft3D-main.zip') as archive:
        for name,digest in previous['native_sources_read_only'].items():
            path=ROOT/name;assert sha(path)==digest
            member=next(v for v in archive.namelist() if v.endswith('/'+path.name))
            assert archive.read(member)==path.read_bytes();sources[name]=digest
        for filename in ('rdtimo.f90','wrtmap.f90','init_mom_output.f90','trisol.f90'):
            members=[v for v in archive.namelist() if v.endswith('/'+filename)];assert len(members)==1
            path=ROOT/'resources'/members[0];assert archive.read(members[0])==path.read_bytes()
            sources[str(path.relative_to(ROOT))]=sha(path)
    audit=dict(scope='Bounded corner qualification; no candidate promoted',production_changed=False,
        improvement_gate_met=False,no_formula_or_coefficient_changes=True,Delft3D_modified=False,
        runtime_manifest_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),native_sources_read_only=sources,
        native=native,native_KFV_corner_masks=masks,native_momentum=momentum,operators=operators,manufactured=manufactured,
        decomposition=decomposition,cases=cases,
        rejection_reason='Tangent/normal extensions worsen the coupled comparison; linear near-wall reconstruction gives only marginal improvement without correcting the displaced jet.',
        scripts={p.name:sha(p) for p in (ROOT/'desktop').glob('*167.py')})
    save(OUT/'AUDIT.json',audit)
    # Figures deliberately use raw cells/connected samples, not smoothing.
    colors=dict(baseline='#333333',tangent='#ce7d00',normal='#8e44ad',lswall='#1976d2',normal_ls='#158864')
    fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for col,x in enumerate((8.25,8.75,10.25)):
        ids=np.flatnonzero(np.isclose(a['xy'][:,0],x)&(a['xy'][:,1]>=3));y=a['xy'][ids,1]
        for row,key in enumerate(('bh','bu')):
            for variant,d in data.items():
                v=d[key][8,ids] if row==0 else d[key][8,0,ids]
                axs[row,col].plot(y,v,label=TITLES[variant],color=colors[variant],lw=1.5)
            v=a['nh'][8,ids] if row==0 else a['nu'][8,0,ids]
            axs[row,col].plot(y,v,'r.--',label='FLOW 2DH',lw=1.5)
            axs[row,col].set(xlabel='Cross-channel y (m)',ylabel='Water depth (m)' if row==0 else 'Streamwise velocity (m/s)',title=f'x = {x:g} m, t = 40 s')
            axs[row,col].set_ylim(bottom=0)
            axs[row,col].grid(alpha=.2)
    axs[1,2].legend(fontsize=8);fig.savefig(OUT/'HYDRAULIC_SECTIONS.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    polys=[np.asarray(p.exterior.coords) for p in a['frames'][0].polygons]
    for ax,variant in zip(axs.flat,('FLOW',)+tuple(data)):
        z=(a['nz'][-1]-a['nz'][0]) if variant=='FLOW' else data[variant]['bz'][-1]-data[variant]['bz'][0]
        pc=PolyCollection(polys,array=z*1000,cmap='RdBu',clim=(-45,45),edgecolors='none');ax.add_collection(pc)
        ax.set(xlim=(5,17),ylim=(0,8),aspect='equal',title='FLOW 2DH' if variant=='FLOW' else TITLES[variant],xlabel='x (m)',ylabel='y (m)')
        ax.add_patch(plt.Rectangle((8,0),2,3,color='.7'));ax.plot(8.25,3.25,'ko',ms=3)
    fig.colorbar(pc,ax=axs.ravel().tolist(),label='Bed change at 60 s (mm)');fig.savefig(OUT/'BED_PATTERNS.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for variant,d in cases.items():
        axs[0].plot([r['t'] for r in d['series']],[r['tip_scour_mm'] for r in d['series']],label=TITLES[variant],color=colors[variant])
    axs[0].plot(a['store'].times,[r['native_tip_scour_mm'] for r in cases['baseline']['series']],'r.--',label='FLOW 2DH')
    axs[0].set(xlim=(40,60),ylim=(0,50),xlabel='Time (s)',ylabel='Groin-tip scour (mm)',title='Morphology starts at 40 s');axs[0].legend(fontsize=8)
    for variant,row in manufactured.items():axs[1].plot(row['wall_x'],np.asarray(row['wall_tangential'])-1.,'o-',color=colors[variant],label=TITLES[variant])
    axs[1].axhline(0,color='r',ls='--',label='Exact slip-wall trace');axs[1].set(xlabel='Wall coordinate x (m)',ylabel='Tangential trace error (m/s)',title='Manufactured shear: exact wall u = 1 m/s')
    for ax in axs:ax.grid(alpha=.2)
    fig.savefig(OUT/'SCOUR_AND_SMOOTH_WALL.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    names=['Central advection','Central pressure','f-wave remainder','Bed source']
    for col,label in enumerate(('BRAIDS','FLOW_PRIMARY')):
        for row,cell in enumerate((0,1)):
            r=decomposition[label][cell];pieces=np.array(r['pieces']);ax=axs[row,col];xx=np.arange(4)
            ax.bar(xx-.18,pieces[:,1],.36,label='x momentum');ax.bar(xx+.18,pieces[:,2],.36,label='y momentum')
            ax.axhline(0,color='k',lw=.5);ax.set_xticks(xx,names,rotation=12);ax.set(ylabel='Momentum rate (m²/s²)',title=f"{label}, cell {r['xy']}")
            ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'MOMENTUM_BALANCE.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    row=momentum['native_momentum_fine'];terms=row['terms_m_s2']
    for key,title in [('MOM_UPRESSURE','Pressure'),('MOM_UDUDX','Streamwise convection'),
                      ('MOM_VDUDY','Cross-advection'),('MOM_UBEDSHEAR','Bed resistance'),('MOM_DUDT','Acceleration')]:
        axs[0].plot(row['first_row_x_m'],terms[key],'o-',label=title)
    axs[0].set(xlabel='Native U-point x (m)',ylabel='Acceleration term (m/s²)',title='FLOW first row above groin at t = 40 s')
    axs[0].legend(fontsize=8);axs[0].axhline(0,color='.5',lw=.5);axs[0].grid(alpha=.2)
    with Dataset(RAW/'native_momentum_fine/trim-sim_P057.nc') as d:
        j=int(np.argmin(abs(d['time'][:]-40)));v=np.asarray(d['MOM_VDUDY'][j,0])
        im=axs[1].pcolormesh(np.arange(83)*.5-.25,np.arange(19)*.5-.5,v.T,
                             cmap='RdBu_r',vmin=-3,vmax=3,shading='flat')
    axs[1].add_patch(plt.Rectangle((8,0),2,3,color='.7'));axs[1].plot(row['first_row_x_m'],[3.25]*5,'ko',ms=3)
    axs[1].set(xlim=(5,13),ylim=(0,6),aspect='equal',xlabel='x (m)',ylabel='y (m)',title='Native cross-advection at staggered U points')
    fig.colorbar(im,ax=axs[1],label='Cross-advection acceleration (m/s²)')
    fig.savefig(OUT/'NATIVE_MOMENTUM.png',dpi=160);plt.close(fig)
    print(json.dumps({k:{x:v[x] for x in ('tip_scour_mm','uv40_region','cpu_gpu_max')} for k,v in cases.items()},indent=2))


if __name__=='__main__':main()
