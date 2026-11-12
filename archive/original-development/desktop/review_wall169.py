"""Read-only FLOW scour chain and rotated Cartesian-wall qualification."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from review_scour166 import bed_rate,sign_upwind,native_flux,verify_native

RAW=ROOT/'validation/_wall169'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_wall169'
PRIOR=ROOT/'validation/sediment/river_morphodynamics/reports'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')


def wall_checks():
    results={};receipts={};files={};bounds={}
    for variant in ('baseline','tangent'):
        folder=RAW/'v3'/variant
        receipt=json.loads((folder/'AUDIT.json').read_text())
        assert receipt['script_sha256']==sha(ROOT/'desktop/check_wall169.py')
        assert receipt['cuda_source_sha256']==sha(folder/'reconstruction.cu')
        receipts[variant]=receipt;rows=[]
        for row in receipt['rows']:
            p=folder/f"angle{row['angle']}"/(row['case']+'.npz')
            files[str(p.relative_to(ROOT))]=sha(p)
            with np.load(p) as a:
                err=[];wallerr=[]
                for c in a['cells']:
                    ids=np.flatnonzero(((a['owner']==c)|(a['neighbour']==c))&(a['neighbour']>=0))
                    for f in ids:
                        side=0 if a['owner'][f]==c else 1
                        err.append(a['face'][side,1:3,f]-a['expected'][f])
                opposite=float(abs(np.asarray(err)).max())
                # Local Cartesian component gradients of an exactly linear field.
                b={'outward':.3,'inward':-.3,'tangent':0.,'uniform':0.}[row['case']]
                slope=0. if row['case']=='uniform' else .2
                exact=np.outer(slope*a['t']+b*a['n'],a['n'])
                ge=float(abs(a['grad'][1:3][:,:,a['cells']]-exact[:,:,None]).max())
                rows.append(dict(row,opposite_fluid_trace_max_m_s=opposite,gradient_max_error_s_inv=ge))
                if row['angle']==0 and row['case']=='outward':
                    c=int(a['cells'][len(a['cells'])//2]);ids=np.flatnonzero((a['owner']==c)|(a['neighbour']==c))
                    bounds[variant]=dict(cell_xy=a['xy'][c].tolist(),cell_velocity=(a['q'][1:,c]/a['q'][0,c]).tolist(),
                        lower=a['vlo'][1:3,c].tolist(),upper=a['vhi'][1:3,c].tolist(),
                        limited_gradient=a['grad'][1:3,:,c].tolist(),
                        faces=[dict(xy=a['mid'][f].tolist(),wall=bool(a['neighbour'][f]<0),
                                    value=a['face'][0 if a['owner'][f]==c else 1,1:3,f].tolist(),
                                    exact=a['expected'][f].tolist()) for f in ids])
        rests={str(d):json.loads((folder/f'angle{d}/REST.json').read_text()) for d in (0,17,37,63,90)}
        results[variant]=dict(rows=rows,rest=rests)
        for p in folder.rglob('*'):
            if p.is_file():files[str(p.relative_to(ROOT))]=sha(p)
    return dict(results=results,receipts=receipts,bounds=bounds,files=files)


def native_chain():
    folder=ROOT/'validation/_scour166/v3/mpm_cpu'
    b=np.load(folder/'BRAIDS.npz');a=np.load(folder/'NATIVE_ALL_FORCING.npz');n=np.load(folder/'NATIVE.npz')
    tip=int(np.argmin(np.linalg.norm(a['xy']-[8.25,3.25],axis=1)))
    region=(a['xy'][:,0]>=6)&(a['xy'][:,0]<=16)
    walls=(a['neighbour']<0)&(a['bid']<0)
    # Raw MUSCL vectors are pre-boundary values. Apply the same closed wall
    # before comparing with final fluxes; otherwise a false wall leak appears.
    muscl=a['wave_physical_normal']/2650.;muscl=muscl.copy();muscl[walls]=0.
    flux=dict(BRAIDS=b['class_flux'][0],all_FLOW_forcing=a['class_flux'][0],
              same_capacity_MUSCL=muscl,FLOW=native_flux(n,a),
              same_capacity_upwind=sign_upwind(n['capacity'],a))
    ids=np.flatnonzero((a['owner']==tip)|(a['neighbour']==tip))
    rows=[]
    for f in ids:
        sign=1 if a['owner'][f]==tip else -1
        factor=-sign*a['length'][f]/a['area'][tip]/.6*1000.
        rows.append(dict(face=int(f),xy=a['mid'][:,f].tolist(),normal=a['normal'][:,f].tolist(),
            owner=int(a['owner'][f]),neighbour=int(a['neighbour'][f]),
            flux_m2_s={k:float(v[f]) for k,v in flux.items()},
            bed_rate_mm_s={k:float(factor*v[f]) for k,v in flux.items()}))
    stages={k:float(bed_rate(v,a)[tip]) for k,v in flux.items()}
    capacity=a['bedcell'][3:5]/2650.
    speed=np.linalg.norm(a['capacity_q'][1:]/a['capacity_q'][0],axis=0)
    chezy=a['capacity_chezy'];diameter=.001;d90=a['capacity_d90']
    grain_chezy=18*np.log10(12*np.maximum(a['capacity_input_q'][0]/d90,1))
    mu=np.minimum((chezy/grain_chezy)**1.5,1)
    theta=(speed/chezy)**2/1.65/diameter
    excess=np.maximum(mu*theta-.047,0)
    calculated=8*np.sqrt(9.81*1.65*diameter)*diameter*excess**1.5
    scalar_error=float(np.linalg.norm(calculated-np.linalg.norm(n['capacity'],axis=0))/np.linalg.norm(n['capacity']))
    fine=ROOT/'validation/_corner167/native_momentum_fine'
    receipt=verify_native(fine)
    with Dataset(fine/'trim-sim_P057.nc') as ds:
        times=np.asarray(ds['time'][:]);morph=np.asarray(ds['MFTAVG'][:])*86400.
        assert np.max(abs(morph-np.maximum(times-40.,0)))<1e-11
        # wrsedmavg writes CUMULATIVE averages since MORFT0, not a last-map
        # interval average. Multiply by elapsed morphology time, THEN difference.
        u=np.asarray(ds['SBUUA'][:,0])*morph[:,None,None]
        v=np.asarray(ds['SBVVA'][:,0])*morph[:,None,None]
        integrated=((np.roll(u,1,axis=1)-u)+(np.roll(v,1,axis=2)-v))/.5/.6
        xy=a['xy'];ix=np.rint(xy[:,0]/.5+.5).astype(int);iy=np.rint(xy[:,1]/.5+.5).astype(int)
        z=-np.asarray(ds['DPS'][:])[:,ix,iy];zero=int(np.argmin(abs(times-40)))
        actual=z-z[zero];pred=integrated[:,ix,iy]
        stock=np.asarray(ds['MSED'][:,0,0])[:,ix,iy]
        instantaneous=[]
        for j in range(len(times)):
            nv=dict(face_u=np.asarray(ds['SBUU'][j,0]),face_v=np.asarray(ds['SBVV'][j,0]))
            instantaneous.append(bed_rate(native_flux(nv,a),a))
        instantaneous=np.array(instantaneous)
    moving=times>40
    update_error=float(abs((actual-pred)[moving][:,region]).max())
    mass_error=float(abs(((stock-stock[zero])/1590-actual)[moving][:,region]).max())
    selected=np.flatnonzero(moving)
    increments=[]
    for j in selected:
        actual_step=actual[j]-actual[j-1];pred_step=pred[j]-pred[j-1]
        increments.append(dict(time_s=float(times[j]),tip_actual_mm=float(actual_step[tip]*1000),
            tip_flux_integral_mm=float(pred_step[tip]*1000),
            maximum_region_closure_m=float(abs(actual_step[region]-pred_step[region]).max()),
            endpoint_flux_dt_difference_mm=float((instantaneous[j,tip]*(times[j]-times[j-1])-actual_step[tip]*1000))))
    series=dict(time_s=times[moving].tolist(),actual_tip_mm=(actual[moving,tip]*1000).tolist(),
                predicted_tip_mm=(pred[moving,tip]*1000).tolist())
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [folder/'BRAIDS.npz',folder/'NATIVE_ALL_FORCING.npz',folder/'NATIVE.npz',
        fine/'trim-sim_P057.nc',fine/'INPUTS.json',fine/'execution.json']}
    result=dict(tip_cell=tip,tip_xy=a['xy'][tip].tolist(),face_ledger=rows,tip_rates_mm_s=stages,
        independent_MPM_relative_l2=scalar_error,
        capacity_relative_l2=float(np.linalg.norm(capacity-n['capacity'])/np.linalg.norm(n['capacity'])),
        wave_max_tip_flux_m2_s=float(abs(flux['all_FLOW_forcing'][ids]-muscl[ids]).max()),
        upwind_replay_tip_flux_max_m2_s=float(abs(flux['FLOW'][ids]-flux['same_capacity_upwind'][ids]).max()),
        native_cumulative_bed_closure_m=update_error,native_stock_bed_closure_m=mass_error,
        native_updates=increments,native_series=series,native_receipt=receipt,inputs=inputs,
        wall_flux=float(abs(a['class_flux'][0,walls]).max()),
        minimum_donor=float(a['frozen_transport_minimum_donor']))
    return result


def native_sources():
    names=('trisol.f90','mom_fls.f90','mom_cyclic.f90','upwhu.f90','caldpu.f90','dwnvel.f90','erosed.f90',
           'eqtran.f90','tranb2.f90','upwbed.f90','adjust_bedload.f90','getfixfac.f90','bott3d.f90',
           'rdscour.f90','shearx.f90','wrsedmavg.f90','bedcomposition_module.f90','ck_dpopt.f90')
    result={}
    with zipfile.ZipFile(ROOT/'resources/Delft3D-main.zip') as archive:
        for name in names:
            members=[p for p in archive.namelist() if p.endswith('/'+name)]
            assert len(members)==1,(name,members)
            p=ROOT/'resources'/members[0]
            assert p.read_bytes()==archive.read(members[0]),name
            result[str(p.relative_to(ROOT))]=sha(p)
    return result


def figures(wall,chain,prior):
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for variant,color in [('baseline','#316da5'),('tangent','#c36420')]:
        for case,style in [('outward','-'),('inward','--')]:
            rows=[r for r in wall['results'][variant]['rows'] if r['case']==case]
            x=[r['angle'] for r in rows]
            for ax,key in zip(axes,('tangent_error','normal_error','opposite_fluid_trace_max_m_s')):
                ax.plot(x,[1000*r[key] for r in rows],style,marker='o',color=color,label=f'{variant}, {case}')
    for ax,title in zip(axes,('Wall tangential velocity','Wall normal trace','Adjacent internal face trace')):
        ax.set(xlabel='Wall angle to Cartesian x (degrees)',ylabel='Maximum trace error (mm/s)',title=title)
        ax.set_ylim(bottom=-.6);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Linear slip-wall shear on clipped Cartesian cells; CPU and GPU traces match exactly')
    fig.savefig(OUT/'ROTATED_WALLS.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
    a=axes[0];names=['West inflow','East outflow','North outflow','South wall'];x=np.arange(4)
    for shift,key,label,color in [(-.18,'FLOW','FLOW cell-upwind','#333333'),(.18,'all_FLOW_forcing','BRAIDS reconstructed','#316da5')]:
        a.bar(x+shift,[r['bed_rate_mm_s'][key] for r in chain['face_ledger']],.36,label=label,color=color)
    a.set_xticks(x,names,rotation=25,ha='right');a.axhline(0,color='.5',lw=.6)
    a.set(ylabel='Contribution to bed rate (mm/s)',title='Same capacity inputs, four tip faces');a.legend(fontsize=8)
    labels=['Own\nforcing','FLOW\nforcing','FLOW forcing\nno wave term','FLOW\nface rule']
    vals=[chain['tip_rates_mm_s'][k] for k in ('BRAIDS','all_FLOW_forcing','same_capacity_MUSCL','FLOW')]
    axes[1].bar(range(4),-np.array(vals),color=['#316da5','#c36420','#b58a36','#333333'])
    axes[1].set_xticks(range(4),labels,fontsize=8);axes[1].set(ylabel='Tip erosion rate (mm/s)',title='At 40 s, before any bed change')
    s=chain['native_series'];axes[2].plot(s['time_s'],s['actual_tip_mm'],'o-',label='FLOW bed output')
    axes[2].plot(s['time_s'],s['predicted_tip_mm'],'x--',label='Integrated face flux / (1-p)')
    axes[2].set(xlabel='Hydraulic time (s)',ylabel='Cumulative tip bed change (mm)',title='First ten native mobile-bed steps')
    axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Scour chain: capacity, face mapping, conservative bed update; no formula tuning')
    fig.savefig(OUT/'SCOUR_CHAIN.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for variant,color in [('baseline','#316da5'),('tangent','#c36420')]:
        d=prior['cases'][variant]
        axes[0].plot([r['t'] for r in d['series']],[r['tip_scour_mm'] for r in d['series']],label=variant,color=color)
    s=prior['cases']['baseline']['series']
    axes[0].plot([r['t'] for r in s],[r['native_tip_scour_mm'] for r in s],'k--',label='FLOW Flood')
    axes[0].set(xlim=(40,60),ylim=(0,50),xlabel='Time (s)',ylabel='Tip scour (mm)',title='Retained matched coupled runs')
    axes[0].legend()
    for variant,color in [('baseline','#316da5'),('tangent','#c36420')]:
        d=wall['bounds'][variant];faces=[r for r in d['faces'] if abs(r['xy'][0]-d['cell_xy'][0])<1e-10]
        faces=sorted(faces,key=lambda r:r['xy'][1])
        axes[1].plot([r['xy'][1]-5.17 for r in faces],[r['value'][0] for r in faces],'o-',color=color,label=variant)
    xs=np.linspace(0,.33,30);axes[1].plot(xs,1+.2*xs,'k--',label='Exact linear shear')
    axes[1].set(xlabel='Distance from slip wall (m)',ylabel='Tangential velocity (m/s)',title='One cell: wall and opposite face')
    axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('A correct smooth-wall reconstruction does not by itself fix the groin jet')
    fig.savefig(OUT/'WALL_AND_COUPLED.png',dpi=160);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    wall=wall_checks();chain=native_chain()
    old=PRIOR/'desktop_corner167/AUDIT.json';prior=json.loads(old.read_text())
    # Verify the already completed paired coupled controls rather than rerun them.
    manifest=json.loads((old.parent/'MANIFEST.json').read_text())
    assert sha(old)==manifest['artifacts']['AUDIT.json']
    for backend in ('cpu','gpu'):
        p=ROOT/f'validation/_corner167/runs/final_tangent/{backend}/STATUS.json'
        assert sha(p)==prior['cases']['tangent'][backend+'_status_sha256']
    runtime=ROOT/'desktop/runtime';identity=json.loads((runtime/'RUNTIME.json').read_text())
    for name,digest in identity['files'].items():assert sha(runtime/name)==digest,name
    figures(wall,chain,prior)
    audit=dict(scope='Rotated-wall component qualification and source-to-scour accounting; not an improved groin release',
        wall=wall,chain=chain,native_sources_read_only=native_sources(),
        retained_coupled={k:prior['cases'][k] for k in ('baseline','tangent')},
        prior_audit_sha256=sha(old),runtime_manifest_sha256=sha(runtime/'RUNTIME.json'),
        production_changed=False,Delft3D_modified=False,groin_improvement_gate=False,
        cross_advection_suppressed=False,physical_laws_changed=False,
        scripts={n:sha(ROOT/'desktop'/n) for n in ('check_wall169.py','review_wall169.py')})
    save(OUT/'AUDIT.json',audit)
    print(json.dumps(dict(wall_max={k:max(r['tangent_error'] for r in v['rows']) for k,v in wall['results'].items()},
        opposite_max={k:max(r['opposite_fluid_trace_max_m_s'] for r in v['rows']) for k,v in wall['results'].items()},chain=chain),indent=2))


if __name__=='__main__':main()
