"""Raw-cell figures and receipts for the short saved-state sensitivity audit."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'validation/_wall188';RUNS=BASE/'replay'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_wall188'
sys.path[:0]=[str(ROOT/'validation/_inlet187/prescribed_candidate'),str(ROOT)]
from braids_desktop.results import ResultsStore


def read(tag):return np.load(RUNS/tag/'FIELDS.npz')
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    initial=read('common_cpu');area=initial['geometry_area'];xy=initial['geometry_xy']
    store=ResultsStore(ROOT/'validation/_inlet187/wet_prescribed_cpu/run')
    mesh=store.read(0).mesh;vertices=mesh['vertices'];offset=mesh['offsets']
    polygons=[vertices[offset[i]:offset[i+1]] for i in range(len(area))]
    def comparison(a,b):
        x=read(a);y=read(b);result={}
        for field,key in (('bed','final_z'),('depth','final_q')):
            delta=x[key]-y[key]
            if field=='depth':delta=delta[0]
            c=int(np.argmax(abs(delta)))
            result[field]=dict(max_mm=float(abs(delta[c])*1000),rms_mm=float(np.sqrt(np.sum(area*delta**2)/sum(area))*1000),
                cell=c,xy_m=xy[:,c].tolist(),area_m2=float(area[c]))
        return result
    pairs={'common_start':('common_cpu','common_gpu'),
        'nanometre_perturbation':('common_cpu','perturb_cpu'),
        'hydraulics_only':('hydro_cpu','hydro_gpu'),
        'half_backend':('half_cpu','half_gpu'),'quarter_backend':('quarter_cpu','quarter_gpu'),
        'dt_02_01':('common_cpu','half_cpu'),'dt_01_005':('half_cpu','quarter_cpu'),
        'dt_005_0025':('quarter_cpu','eighth_cpu'),
        'no_momentum_or_viscosity_perturbation':('nofeedback_cpu','nofeedback_perturb'),
        'no_bed_steering_perturbation':('nodirection_cpu','nodirection_perturb'),
        'steered_wave_backend':('steered_cpu','steered_gpu'),
        'projected_interior_backend':('projected_cpu4','projected_common'),
        'projected_all_walls_backend':('walls_cpu','walls_common'),
        'projected_all_walls_own_warm':('walls_cpu','walls_wet_gpu'),
        'projected_all_walls_perturbation':('walls_common','walls_perturb'),
        'projected_all_walls_method_change':('common_cpu','walls_cpu'),
        'projected_secant_backend':('secant_cpu2','secant_gpu'),
        'projected_secant_method_change':('common_cpu','secant_cpu2'),
        'projected_secant_own_warm':('secant_cpu2','secant_own_gpu'),
        'projected_secant_perturbation':('secant_gpu','secant_perturb'),
        'projected_secant_dry_backend':('secant_dry_cpu','secant_dry_gpu')}
    metrics={k:comparison(*v) for k,v in pairs.items() if all((RUNS/t/'FIELDS.npz').exists() for t in v)}
    runs={}
    for tag in set(t for pair in pairs.values() for t in pair):
        file=RUNS/tag/'STEPS.json'
        if not file.exists():continue
        rows=json.loads(file.read_text());f=read(tag);dz=f['final_z']-f['initial_z']
        runs[tag]=dict(steps=len(rows),time=rows[-1]['time'],rejected=sum(len(s['rejected']) for s in rows),
            bed_min_mm=float(min(dz)*1000),bed_max_mm=float(max(dz)*1000),
            erosion_bulk_m3=float(sum(area*np.maximum(-dz,0))),deposition_bulk_m3=float(sum(area*np.maximum(dz,0))),
            scour_cell=int(np.argmin(dz)),deposition_cell=int(np.argmax(dz)),
            maximum_water_residual_m3=max(abs(s['water_residual_m3']) for s in rows),
            maximum_solid_residual_m3=max(abs(s['solid_volume_residual_m3']) for s in rows),
            input=json.loads((RUNS/tag/'INPUT.json').read_text()),fields_sha256=sha(RUNS/tag/'FIELDS.npz'))
    same={};f=np.load(RUNS/'hydro_same_cpu/HYDRAULIC.npz');g=np.load(RUNS/'hydro_same_gpu/HYDRAULIC.npz')
    for k in (0,20,26,31,43,71):
        same[str(k)]={field:float(np.max(abs(f[f'{k}_{field}']-g[f'{k}_{field}']))) for field in ('rate','water','exterior_flux','rec_face_value')}
    sens={};f=np.load(RUNS/'onset_sensitivity/SENSITIVITY.npz');g=np.load(RUNS/'projected_sensitivity3/SENSITIVITY.npz')
    for label,data in (('original',f),('projected',g)):
        sens[label]={field:float(np.max(abs(data['cpu_0_'+field]-data['gpu_0_'+field]))) for field in ('q','z','rate','rec_grad','rec_face_value')}
    save('COMPARISON.json',dict(comparisons=metrics,runs=runs,same_input_hydraulics=same,first_stage_sensitivity=sens,
        interpretation='Saved physical fields and fresh histories, NOT authenticated cross-backend restarts. No timing claims.'))
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    selections=[('common_cpu','common_gpu','Original common start'),('common_cpu','perturb_cpu','Original +1 nm depth'),
        ('projected_cpu4','projected_common','Directional interior only'),('walls_cpu','walls_common','Directional global wall bounds'),
        ('secant_cpu2','secant_gpu','Directional existing wall bounds')]
    for a,b,label in selections:
        if not all((RUNS/t/'FIELDS.npz').exists() for t in (a,b)):continue
        x=read(a);y=read(b);history=abs(x['history']-y['history'])*1000
        times=[r['time']-200 for r in json.loads((RUNS/a/'STEPS.json').read_text())]
        for ax,row in zip(axes,(3,0)):ax.semilogy(times,np.maximum(history[:,row].max(axis=1),1e-10),label=label)
    for ax,field in zip(axes,('Bed elevation','Water depth')):
        ax.set(xlabel='Hydraulic seconds after sediment activation',ylabel=f'Maximum {field.lower()} difference (mm)',title=field)
        ax.axhline(1,color='k',ls=':',lw=1);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Fixed multilevel two-meander mesh: sensitivity after sediment activation')
    fig.savefig(OUT/'SENSITIVITY_HISTORY.png',dpi=180);plt.close(fig)
    selected='walls_common';cpu='walls_cpu'
    candidate=read(selected);reference=read(cpu)
    fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
    arrays=[(read('common_cpu')['final_z']-initial['initial_z'])*1000,
        (candidate['final_z']-initial['initial_z'])*1000,
        (read('common_gpu')['final_z']-read('common_cpu')['final_z'])*1000,
        (candidate['final_z']-reference['final_z'])*1000]
    titles=['Original bed evolution, 200 to 220 s','Directional limiter with global wall bounds',
        'Original GPU minus CPU bed elevation','Experimental GPU minus CPU bed elevation']
    for i,(ax,z,title) in enumerate(zip(axes.ravel(),arrays,titles)):
        bound=200 if i<2 else 2
        pc=PolyCollection(polygons,array=z,cmap='RdBu_r',clim=(-bound,bound),edgecolors='none')
        ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');ax.set(title=title,xlabel='x (m)',ylabel='y (m)')
        fig.colorbar(pc,ax=ax,label='mm')
    fig.suptitle('Raw finite-volume cells, no spatial smoothing; bed-slope correction OFF')
    fig.savefig(OUT/'RAW_BED_COMPARISON.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    difference=(candidate['final_z']-read('common_gpu')['final_z'])*1000
    for ax,limits,title in zip(axes,((0,56,-1,26),(20,26,4,11)),('Whole domain','First bend detail')):
        pc=PolyCollection(polygons,array=difference,cmap='RdBu_r',clim=(-40,40),edgecolors='#777',linewidths=.1)
        ax.add_collection(pc);ax.set(xlim=limits[:2],ylim=limits[2:],aspect='equal',xlabel='x (m)',ylabel='y (m)',title=title)
        fig.colorbar(pc,ax=ax,label='New minus original bed (mm)')
    fig.suptitle('Numerical method change is NOT the CPU/GPU discrepancy')
    fig.savefig(OUT/'METHOD_CHANGE.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    c=3829;labels=['surface x slope','bed x slope'];xx=np.arange(2)
    for i,(key,label) in enumerate((('original','Original'),('projected','Directional'))):
        d=f if i==0 else g
        values=[abs(d['cpu_0_rec_grad'][m,0,c]-d['gpu_0_rec_grad'][m,0,c]) for m in (0,3)]
        axes[0].bar(xx+(i-.5)*.32,values,width=.32,label=label)
    axes[0].set(xticks=xx,xticklabels=labels,yscale='log',ylabel='Absolute gradient discrepancy',title='Same saved near-identical states at outlet cell 3829')
    axes[0].legend();axes[0].grid(axis='y',alpha=.2)
    x=read('common_cpu');y=read('common_gpu');z=(y['history'][1,0]-x['history'][1,0])*1e6
    pc=PolyCollection(polygons,array=z,cmap='RdBu_r',clim=(-6,6),edgecolors='#999',linewidths=.15)
    axes[1].add_collection(pc);axes[1].set(xlim=(54.9,56.1),ylim=(20,24.1),aspect='equal',xlabel='x (m)',ylabel='y (m)',title='First separation at 200.4 s')
    fig.colorbar(pc,ax=axes[1],label='GPU minus CPU depth (micrometres)')
    fig.savefig(OUT/'OUTLET_ONSET.png',dpi=180);plt.close(fig)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
