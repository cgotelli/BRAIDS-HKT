"""Native-cell, matched-input bend comparison and public formula audit figures."""
import json,sys
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'validation/_bend189'
RUNTIME=ROOT/'validation/_wall188/desktop_candidate'
sys.path[:0]=[str(RUNTIME),str(ROOT),str(ROOT/'desktop')]
import numpy as np
import shapely
from shapely.geometry import LineString
from shapely.strtree import STRtree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from braids_desktop.results import ResultsStore
from stage_secondary185 import save,sha
from bedload_oracle189 import formula
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_bend189'
SOURCES={'on_cpu':ROOT/'validation/_wall188/release/wet_prescribed_cpu',
         'on_gpu':ROOT/'validation/_wall188/release/wet_prescribed_gpu',
         **{k:BASE/'runs'/k for k in ('off_cpu','off_gpu','secondary_cpu','viscosity_cpu')}}
LABELS={'off':'Both OFF','on':'Advanced ON','secondary':'Secondary only','viscosity':'Viscosity only'}
COLORS={'off':'#3366aa','on':'#d34b38','secondary':'#9a5eb0','viscosity':'#37956c'}


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'desktop/examples/two_meanders184/CASE.json').read_text())
    line=LineString(cfg['xy']);ss=np.array(cfg['s']);angles=np.array(cfg['angle']);br=cfg['breaks']
    data={};metrics={'runtime_sha256':sha(RUNTIME/'RUNTIME.json'),'cases':{}}
    input_ref=json.loads((SOURCES['on_cpu']/'project/braids_project.json').read_text())
    input_rows={}
    for key,source in SOURCES.items():
        doc=json.loads((source/'project/braids_project.json').read_text())
        sa,sb=deepcopy(input_ref['settings']),deepcopy(doc['settings'])
        changed_controls={k:sb['controls'].pop(k) for k in ('betac','diffusivity','viscosity')}
        for k in changed_controls:sa['controls'].pop(k)
        assert sa==sb,key
        for k in ('initial','material_zones','roughness_parameters','slope_parameters',
                  'sediment_schedules','column_methods','bedform_methods','hiding_parameters'):
            assert doc[k]==input_ref[k],(key,k)
        assert doc['model']['request']['sediment']==input_ref['model']['request']['sediment']
        input_rows[key]=dict(project_sha256=sha(source/'project/braids_project.json'),
            unchanged_controls_and_settings=True,hydrodynamics=doc['model']['request']['hydrodynamics'],
            changed_controls=changed_controls)
        store=ResultsStore(source/'run')
        def at(t):return store.read(int(np.flatnonzero(store.times==t)[0]))
        first,late,warm,end=map(at,(0,180,200,220));area=end.arrays['area']
        assert np.array_equal(first.z,warm.z),key
        if not data:
            polygons=[np.asarray(p.exterior.coords) for p in first.polygons]
            xy=np.array([p.centroid.coords[0] for p in first.polygons])
            station=shapely.line_locate_point(line,shapely.points(xy))
            foot=shapely.get_coordinates(shapely.line_interpolate_point(line,station))
            angle=np.interp(station,ss,angles);tangent=np.array([np.cos(angle),np.sin(angle)])
            normal=np.array([-np.sin(angle),np.cos(angle)])
            cross=np.sum((xy-foot)*normal.T,axis=1)
            masks=[]
            for start,stop,sign in ((br[1],br[2],1),(br[3],br[4],-1)):
                inside=(station>start+2)&(station<stop-2)
                masks.append(dict(inner=inside&(sign*cross>.4),outer=inside&(sign*cross<-.4)))
            tree=STRtree(first.polygons)
        else:
            ref=data['on_cpu']['first']
            for k in ('q','z','W','mass','area'):np.testing.assert_array_equal(first.arrays[k],ref.arrays[k])
            for k in ('vertices','offsets','levels'):np.testing.assert_array_equal(first.mesh[k],ref.mesh[k])
        dz=end.z-first.z;dh=warm.q[0]-late.q[0]
        speed=lambda f:np.hypot(*f.q[1:])/f.q[0]
        du=speed(warm)-speed(late)
        diags=json.loads((source/'run/DIAGNOSTICS.json').read_text())
        assert all(not any(d['wall_integrals']) for d in diags)
        def extrema(index):return dict(cell=int(index),xy_m=xy[index].tolist(),station_m=float(station[index]),cross_left_m=float(cross[index]))
        m=dict(cells=len(area),min_area_m2=float(area.min()),end_time_s=220.,morphological_duration_s=100.,
            bed_min_mm=float(dz.min()*1000),bed_max_mm=float(dz.max()*1000),
            erosion_bulk_m3=float(np.dot(area,np.maximum(-dz,0))),deposition_bulk_m3=float(np.dot(area,np.maximum(dz,0))),
            scour_location=extrema(np.argmin(dz)),bar_location=extrema(np.argmax(dz)),
            water_residual_m3=max(abs(d['water_residual_m3']) for d in diags),
            sediment_residual_kg=max(max(abs(x) for x in d['class_residual_kg']) for d in diags),
            warmup_180_200_depth_max_mm=float(abs(dh).max()*1000),
            warmup_180_200_depth_rms_mm=float(np.sqrt(np.average(dh**2,weights=area))*1000),
            warmup_180_200_speed_rms_m_s=float(np.sqrt(np.average(du**2,weights=area))),bends=[])
        for regions in masks:
            m['bends'].append({k:dict(mean_mm=float(np.average(dz[ix]*1000,weights=area[ix])),
                min_mm=float(dz[ix].min()*1000),max_mm=float(dz[ix].max()*1000),
                erosion_m3=float(np.dot(area[ix],np.maximum(-dz[ix],0))),
                deposition_m3=float(np.dot(area[ix],np.maximum(dz[ix],0)))) for k,ix in regions.items()})
        evolution=[]
        for time in range(200,221):
            frame=at(time);change=frame.z-first.z
            evolution.append(dict(time_s=time,morph_s=(time-200)*5,
                erosion_m3=float(np.dot(area,np.maximum(-change,0))),
                deposition_m3=float(np.dot(area,np.maximum(change,0))),
                min_mm=float(change.min()*1000),max_mm=float(change.max()*1000)))
        m['evolution']=evolution;metrics['cases'][key]=m
        data[key]=dict(first=first,warm=warm,end=end,dz=dz,area=area)
    def diff(a,b,field='dz',scale=1000):
        values=data[a][field]-data[b][field]
        return dict(max_abs=float(abs(values).max()*scale),rms=float(np.sqrt(np.average(values**2,weights=area))*scale))
    metrics['bed_differences_mm']={f'{a}_minus_{b}':diff(a,b) for a,b in (
        ('on_cpu','off_cpu'),('on_cpu','on_gpu'),('off_cpu','off_gpu'),
        ('viscosity_cpu','off_cpu'),('secondary_cpu','off_cpu'),('on_cpu','secondary_cpu'))}
    metrics['depth_backend_mm']={}
    for mode in ('on','off'):
        delta=data[mode+'_cpu']['end'].q[0]-data[mode+'_gpu']['end'].q[0]
        metrics['depth_backend_mm'][mode]=dict(max_abs=float(abs(delta).max()*1000),rms=float(np.sqrt(np.average(delta**2,weights=area))*1000))
    metrics['initial_arrays_exact']=True
    metrics['input_checks']=input_rows
    def draw(ax,values,title,limit,cmap='RdBu_r'):
        pc=PolyCollection(polygons,array=values,cmap=cmap,edgecolors='none',rasterized=True)
        pc.set_clim(-limit,limit);ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal')
        ax.set(xlabel='x (m)',ylabel='y (m)',title=title)
        return pc
    modes=('off','viscosity','secondary','on');limit=max(abs(data[k+'_cpu']['dz']).max() for k in modes)*1000
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for ax,k in zip(axes.flat,modes):
        pc=draw(ax,data[k+'_cpu']['dz']*1000,LABELS[k],limit)
        for i,(start,stop,sign) in enumerate(((br[1],br[2],1),(br[3],br[4],-1))):
            s=(start+stop)/2;p=np.array(line.interpolate(s).coords[0]);theta=np.interp(s,ss,angles)
            n=sign*np.array([-np.sin(theta),np.cos(theta)])
            ax.annotate(f'Inner bend {i+1}',p+1.4*n,xytext=p+6*n,fontsize=9,
                        arrowprops={'arrowstyle':'->','color':'#333333'})
    fig.colorbar(pc,ax=axes.ravel().tolist(),label='Bed change (mm): blue = erosion; red = deposition',shrink=.8)
    fig.suptitle('Smart & Jäggi: matched mesh and initial fields\n200 s hydraulic warm-up + 100 morphological seconds (MF5); raw cells')
    fig.savefig(REPORT/'BED_COMPARISON.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,10),layout='constrained')
    for ax,(a,b,title) in zip(axes,(('on_cpu','off_cpu','Physical contrast: advanced ON − OFF'),
                                  ('on_cpu','on_gpu','Backend verification: ON CPU − GPU'))):
        delta=(data[a]['dz']-data[b]['dz'])*1000
        pc=draw(ax,delta,title,abs(delta).max());fig.colorbar(pc,ax=ax,label='Bed difference (mm)',shrink=.85)
    fig.suptitle('Different scales: physics changes are much larger than backend roundoff sensitivity')
    fig.savefig(REPORT/'DIFFERENCES.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True,layout='constrained')
    for row,(start,stop,sign) in enumerate(((br[1],br[2],1),(br[3],br[4],-1))):
        for col,fraction in enumerate((.25,.5,.75)):
            s=start+(stop-start)*fraction;p=np.array(line.interpolate(s).coords[0]);theta=np.interp(s,ss,angles)
            n=sign*np.array([-np.sin(theta),np.cos(theta)]);pos=np.linspace(-1.95,1.95,161)
            hits=tree.query(shapely.points(p+pos[:,None]*n),predicate='intersects');idx=np.full(len(pos),-1)
            idx[hits[0]]=hits[1];valid=idx>=0;ax=axes[row,col]
            for k in ('off','on'):
                ax.plot(pos[valid],data[k+'_cpu']['dz'][idx[valid]]*1000,color=COLORS[k],label=LABELS[k])
            ax.axhline(0,color='k',lw=.5);ax.grid(alpha=.2)
            ax.set(title=f'Bend {row+1}: {fraction*60:g}°',xlabel='Cross-channel (m): outer − / inner +')
        axes[row,0].set_ylabel('Bed change (mm)')
    axes[0,0].legend();fig.suptitle('Bed sections at 100 morphological seconds — no display smoothing')
    fig.savefig(REPORT/'CROSS_SECTIONS.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for k in modes:
        e=metrics['cases'][k+'_cpu']['evolution'];times=[r['morph_s'] for r in e]
        for ax,field,title in zip(axes.flat,('deposition_m3','erosion_m3','max_mm','min_mm'),
            ('Bulk deposition (m³)','Bulk erosion (m³)','Maximum deposition (mm)','Deepest bed change (mm)')):
            ax.plot(times,[r[field] for r in e],color=COLORS[k],label=LABELS[k]);ax.set(xlabel='Morphological time (s)',ylabel=title);ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Transient evolution, not an equilibrium or experimental comparison')
    fig.savefig(REPORT/'EVOLUTION.png',dpi=170);plt.close(fig)
    # Actual inputs to the first mobile-bed capacity evaluation after warm-up.
    operators={k:dict(np.load(BASE/'operators_v2'/k/'INPUTS.npz')) for k in modes}
    opmetrics={};edges=np.arange(0,66,1.);mid=.5*(edges[1:]+edges[:-1]);bins=np.digitize(station,edges)-1
    def average(values,mask=None):
        if mask is None:mask=np.ones(len(area),bool)
        return np.array([np.average(values[(bins==i)&mask],weights=area[(bins==i)&mask])
            if np.any((bins==i)&mask) else np.nan for i in range(len(mid))])
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for k in modes:
        o=operators[k];q=o['q'];h=q[0];vel=q[1:]/h;speed=np.hypot(*vel)
        flux=o['value'][3:5]/2650.;raw=o['value'][:2]/2650.;mag=np.hypot(*flux)
        turn=np.rad2deg(np.arctan2(vel[0]*flux[1]-vel[1]*flux[0],vel[0]*flux[0]+vel[1]*flux[1]))
        j=speed**2/(o['hydraulic_chezy']**2*h)
        per_bend=[]
        for region in masks:
            ix=region['inner']|region['outer']
            per_bend.append(dict(mean_secondary_I_m_s=float(np.average(o['hi'][ix]/h[ix],weights=area[ix])),
                mean_turn_left_deg=float(np.average(turn[ix],weights=area[ix])),
                mean_capacity_m2_s=float(np.average(mag[ix],weights=area[ix])),
                mean_speed_m_s=float(np.average(speed[ix],weights=area[ix]))))
        opmetrics[k]=dict(energy_slope_min=float(j.min()),energy_slope_max=float(j.max()),energy_slope_median=float(np.median(j)),
            area_below_J005=float(area[j<.005].sum()/area.sum()),area_above_J02=float(area[j>.2].sum()/area.sum()),
            magnitude_change_by_steering_max_m2_s=float(abs(np.hypot(*raw)-mag).max()),
            secondary_I_max_m_s=float(abs(o['hi']/h).max()),bends=per_bend,
            predictor_q_difference=json.loads((BASE/'operators_v2'/k/'CHECK.json').read_text())['warm_q_max_difference'])
        fields=[np.sum(vel*tangent,axis=0),o['hi']/h,turn,mag*1000]
        for ax,values in zip(axes.flat,fields):ax.plot(mid,average(values),color=COLORS[k],label=LABELS[k])
    for ax,label in zip(axes.flat,('Streamwise velocity (m/s)','Secondary intensity I (m/s)',
                                 'Bedload deflection from flow (degrees, left +)','Transport capacity (10⁻³ m²/s)')):
        ax.set(xlabel='Distance along channel (m)',ylabel=label);ax.grid(alpha=.2)
        for a,b in ((br[1],br[2]),(br[3],br[4])):ax.axvspan(a,b,alpha=.07,color='black')
        if 'capacity' not in label.lower():ax.axhline(0,color='k',lw=.5)
    axes[0,0].legend();fig.suptitle('Actual capacity inputs immediately after 200 s warm-up\nArea-weighted 1 m station bins; shading marks the two bends; not reconstructed display values')
    fig.savefig(REPORT/'FLOW_AND_TRANSPORT.png',dpi=170);plt.close(fig)
    for k in ('on','secondary','viscosity'):
        a,b=operators[k],operators['off'];u=a['q'][1:]/a['q'][0];v=b['q'][1:]/b['q'][0]
        delta=np.hypot(*u)-np.hypot(*v);cap=np.hypot(*a['value'][3:5])-np.hypot(*b['value'][3:5])
        opmetrics[k]['vs_off']=dict(speed_rms_m_s=float(np.sqrt(np.average(delta**2,weights=area))),
            depth_rms_mm=float(np.sqrt(np.average((a['q'][0]-b['q'][0])**2,weights=area))*1000),
            capacity_relative_l2=float(np.sqrt(np.dot(area,cap**2)/np.dot(area,np.hypot(*b['value'][3:5])**2))))
    metrics['operators']=opmetrics
    save(REPORT/'COMPARISON.json',metrics)
    formula_figures()
    print(json.dumps({k:{n:v for n,v in m.items() if n in ('bed_min_mm','bed_max_mm','erosion_bulk_m3','deposition_bulk_m3','bends')} for k,m in metrics['cases'].items()},indent=2))


def formula_figures():
    # Plot physical forcing, not falsely interchangeable characteristic speeds.
    speed=np.linspace(0,4,401);h=np.full_like(speed,.6);c=np.full_like(speed,30.)
    z0=h/(np.e*np.expm1(.41*c/np.sqrt(9.81)));shear=speed*.41/np.log1p(h*float(np.float32(.368))/z0)
    effective=shear*c/np.sqrt(9.81);d=.01;d90=np.full_like(h,1.5*d);hide=np.ones_like(h)
    defaults={'capacity.mpm':{'ACal':1.},'capacity.ashida_michiue_generalized':{'ACal':17.,'ThetaC':.05,'PowerM':1.5,'PowerP':1.,'PowerQ':1.},
        'capacity.wang_fredsoe':{'ACal':1.,'VicMol':1e-6},'capacity.smart_jaggi':{'ACal':4.,'ThetaC':.05}}
    labels=('MPM','Ashida–Michiue (FLOW)','Wang–Fredsøe bedload','Smart & Jäggi')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained');curves={}
    for (method,p),label in zip(defaults.items(),labels):
        qb=formula(method,p,speed if method.endswith('smart_jaggi') else effective,h,c,d,d90,hide,shear)
        axes[0].plot(speed,qb*1000,label=label);axes[1].semilogy(speed,np.where(qb>0,qb,np.nan),label=label)
        curves[method]=dict(physical_speed_m_s=speed.tolist(),solid_capacity_m2_s=qb.tolist())
    for ax in axes:ax.grid(alpha=.2);ax.set_xlabel('Physical depth-averaged speed (m/s)')
    axes[0].set_ylabel('Solid-volume bedload capacity (10⁻³ m²/s)');axes[1].set_ylabel('Capacity (m²/s, logarithmic)')
    axes[0].legend();axes[1].set_ylim(1e-9,.1)
    # Smart's application range is a caveat, not a code clipping condition.
    low=np.sqrt(.005*30**2*.6)
    for ax in axes:ax.axvspan(0,low,color='grey',alpha=.09);ax.axvline(low,color='grey',ls=':',lw=1)
    fig.suptitle('Four different laws on identical physical forcing: h = 0.6 m, C = 30, d = 10 mm\nGrey region: Smart & Jäggi energy slope below the cited 0.005 range; not a ranking of accuracy')
    fig.savefig(REPORT/'FORMULA_CURVES.png',dpi=170);plt.close(fig);save(REPORT/'FORMULA_CURVES.json',curves)
    audit=json.loads((BASE/'formulas/AUDIT.json').read_text());live={}
    for path in sorted((BASE/'live/groin').glob('*/INPUT_AUDIT.json')):
        v=json.loads(path.read_text());check=json.loads((path.parent/'CHECK.json').read_text())
        live[path.parent.name]=dict(sampled_calls=v['sampled_calls'],sampled_cells=v['sampled_cells'],
            max_field_error=max(r['max_field_error'] for r in v['rows']),check=check)
    flow=json.loads((BASE/'flow_replay/CHECK.json').read_text())
    result=dict(equation_input_cases=sum(r['points'] for r in audit['rows']),batches=len(audit['rows']),
        scalar_max_error_m2_s=max(r['scalar_max_m2_s'] for r in audit['rows']),
        field_max_error=max(r['field_max'] for r in audit['rows']),live=live,
        locked_flow={r['method']+'_'+r['backend']:dict(max_error_m2_s=max(x['capacity_max_m2_s'] for x in r['rows']),
            max_relative_l2=max(x['capacity_relative_l2'] for x in r['rows'])) for r in flow['results']})
    pairs={}
    for family in ('live','dry'):
        if family=='dry' and not (BASE/'dry/JOBS.json').exists():continue
        for law in ('mpm','ashida_michiue_generalized','wang_fredsoe','smart_jaggi'):
            folders=[BASE/family/'groin'/(law+'_'+b) for b in ('cpu','gpu')]
            stores=[ResultsStore(p/'full') for p in folders];a,b=[s.read(-1) for s in stores]
            # Compare true polygons, not just equal cell counts after AMR.
            for k in ('vertices','offsets','levels'):np.testing.assert_array_equal(a.mesh[k],b.mesh[k])
            area=a.arrays['area'];dz=a.z-b.z;dh=a.q[0]-b.q[0]
            checks=[json.loads((p/'CHECK.json').read_text()) for p in folders]
            pairs[family+'_'+law]=dict(final_mesh_exact=True,cells=len(area),
                mesh_epochs=[len(s.index['epochs']) for s in stores],
                bed_max_mm=float(abs(dz).max()*1000),bed_rms_mm=float(np.sqrt(np.average(dz**2,weights=area))*1000),
                depth_max_mm=float(abs(dh).max()*1000),depth_rms_mm=float(np.sqrt(np.average(dh**2,weights=area))*1000),
                water_residual_m3=max(x['water_residual_m3'] for x in checks),
                sediment_residual_kg=max(x['sediment_residual_kg'] for x in checks))
    result['coupled_pairs']=pairs
    save(REPORT/'FORMULA_AUDIT.json',result)


if __name__=='__main__':main()
