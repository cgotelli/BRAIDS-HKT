"""Current public Desktop projects for published SWE and laboratory benchmarks."""
import argparse,csv,hashlib,json,math,shutil,sys
from copy import deepcopy
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'desktop/runtime'
HOME=Path.home()/'Desktop/BRAIDS_Turbine_Validation58'
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/desktop_validation58'
REFERENCE=ROOT/'validation/hydrokinetic_turbines'
sys.path.insert(0,str(RUNTIME));sys.path.insert(1,str(ROOT))
IDENTITY='1dcf63c3b753ebbbe012313f7ee05932461aeb38d9526f22d41af6c478aa097a'
CASES={
    'otf_gpu':dict(folder='01_OpenTidalFarm_2D_GPU',backend='gpu',D=18.75,H=33.,L=1500.,W=775.,X=375.,Y=387.5,U=2.9,
        cb=.0025,nu=.6,nx=640,ny=331,end=250.,average=200.,output=5.,reference='Abolghasemi et al., arXiv:1810.09827, Figure 3, OTF nu=0.6, U=2.9; published 2D model, not experiment'),
    'mycek_cpu':dict(folder='02_Mycek_08_CPU',backend='cpu',U=.8),
    'mycek_gpu':dict(folder='03_Mycek_08_GPU',backend='gpu',U=.8),
    'mycek_holdout_cpu':dict(folder='04_Mycek_10_Holdout_CPU',backend='cpu',U=1.),
    'zero_cpu':dict(folder='05_Mycek_Zero_Thrust_CPU',backend='cpu',U=.8),
}
for name,c in CASES.items():
    if name!='otf_gpu':
        c.update(D=.7,H=2.,L=18.,W=4.,X=5.03,Y=2.,cb=9.81/1000**2,nu=.585/(2.9*18.75)*c['U']*.7,
            nx=144,ny=32,end=40.,average=30.,output=1.,reference='Mycek et al. 2014, doi:10.1016/j.renene.2013.12.036, Figures A.12, 6a, 7a; TI=3%, TSR=3.67')
    c.update(dx=c['L']/c['nx'],dy=c['L']/c['nx'])
    c['ny']=math.ceil(c['W']/c['dy'])


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):Path(path).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')


def controls():
    from validation.hydrokinetic_turbines.digitize_hkt3j import performance
    rows=performance();curves={k:sorted([r for r in rows if r['field']==k and r['inflow_m_s']==.8],key=lambda r:r['TSR']) for k in ['Ct','Cp']}
    assert len(curves['Ct'])==len(curves['Cp'])==13
    # Original vector axes differ slightly; interpolation onto the Ct knots is
    # an explicit CSV/table operation, not a fitted performance model.
    return dict(mode='prescribed_tsr_table',tip_speed_ratio=3.67,performance=[dict(tip_speed_ratio=r['TSR'],thrust_coefficient=r['coefficient'],
        power_coefficient=float(np.interp(r['TSR'],[q['TSR'] for q in curves['Cp']],[q['coefficient'] for q in curves['Cp']]))) for r in curves['Ct'] if r['TSR']>0])


def measurements(c):
    stations=[6.,12.] if c['D']>1 else [1.2,3.,6.,10.]
    profiles=[dict(id='profile_'+str(i+1),name=f'{s:g}D transverse velocity',coords=[[c['X']+s*c['D'],c['Y']-min(2*c['D'],.45*c['W'])],[c['X']+s*c['D'],c['Y']+min(2*c['D'],.45*c['W'])]]) for i,s in enumerate(stations)]
    profiles.append(dict(id='profile_'+str(len(profiles)+1),name='Wake centreline',coords=[[c['X']-2*c['D'],c['Y']],[c['X']+12*c['D'],c['Y']]]))
    probes=[dict(id='probe_'+str(i+1),name=name,coords=[[c['X']+x*c['D'],c['Y']]]) for i,(x,name) in enumerate([(-2.,'Upstream reference'),(6.,'Wake 6D'),(10.,'Wake 10D')])]
    return dict(schema=2,profiles=profiles,probes=probes,selected_profile='profile_1',selected_probe='probe_1')


def create():
    from osgeo import gdal,osr
    from braids_desktop import services,project_file
    from braids_core.river_model import resolve_model
    from braids_core.river_turbines import default,definition
    from braids_core.boundaries.domain import PolygonDomain
    assert sha(RUNTIME/'RUNTIME.json')==IDENTITY
    HOME.mkdir(exist_ok=False);REPORT.mkdir(parents=True,exist_ok=True)
    table=controls();records={}
    for key,c in CASES.items():
        folder=HOME/c['folder'];folder.mkdir();dem=folder/'flat_flume.tif';dx=c['dx'];dy=c['dy']
        raster=gdal.GetDriverByName('GTiff').Create(str(dem),c['nx']+2,c['ny']+2,1,gdal.GDT_Float64,options=['COMPRESS=LZW'])
        raster.SetGeoTransform((-dx,dx,0,c['W']+dy,0,-dy));s=osr.SpatialReference();s.ImportFromEPSG(2056)
        raster.SetProjection(s.ExportToWkt());raster.GetRasterBand(1).Fill(0.);raster.FlushCache();raster=None
        doc=services.default_document(dem,[[0.,0.],[c['L'],0.],[c['L'],c['W']],[0.,c['W']]],'EPSG:2056',name=key,
            backend=c['backend'],minimum_size_m=dx,levels=1,mesh_mode='static_terrain',end_time_s=c['end'])
        req=doc['model']['request'];req['hydrodynamics']['roughness']='roughness.chezy';doc['model']=resolve_model(req).document()
        C=math.sqrt(9.81/c['cb']);doc['roughness_parameters']['chezy_m_half_s']=C;doc['settings']['controls'].update(chezy=C,cfl=.5,threads=8)
        doc['settings']['grid']=dict(nx=c['nx'],ny=c['ny'],xlower_m=0.,ylower_m=0.,dx_m=dx,dy_m=dy)
        domain=PolygonDomain(**doc['settings']['domain'])
        for x,name,kind,control in [(0.,'Upstream','inlet',dict(method='inlet.total_discharge',discharge_m3_s=c['U']*c['H']*c['W'])),
                                   (c['L'],'Downstream','outlet',dict(method='outlet.stage',stage_m=c['H']))]:
            doc=services.assign_boundary(doc,[e.edge_id for e in domain.edges if e.start_xy[0]==e.end_xy[0]==x],name,kind,control)
        doc['initial'].update(stage_m=c['H'],velocity_m_s=[c['U'],0.])
        doc['settings'].update(dt_max_s=2.,output_interval_s=c['output'],checkpoint_interval_s=c['end'])
        doc['settings']['adaptation'].update(domain_border_enabled=False,transition_layers=1)
        turbine=definition([c['X'],c['Y']],[1.,0.],name=key,diameter=c['D'],hub_height=c['H']/2,minimum_bed_clearance=0.,minimum_surface_clearance=0.)
        if key=='otf_gpu':turbine['control']=dict(mode='constant_ct',thrust_coefficient=.85,power_coefficient=.85*.5*(1+math.sqrt(1-.85)))
        elif key=='zero_cpu':turbine['control']=dict(mode='constant_ct',thrust_coefficient=0.,power_coefficient=0.)
        else:turbine['control']=deepcopy(table)
        doc['turbines']=default();doc['turbines'].update(turbines=[turbine],ambient_viscosity_m2_s=c['nu'])
        doc=services.validate_document(doc)
        session=dict(desktop=dict(model_step='Hydrokinetic turbines',result_step='Turbines',live_results=True,measurements=measurements(c),advanced_hydrodynamics=False))
        saved=services.save_project_folder(folder/'Project_data',doc,session=session)
        entry=folder/'Validation.braids.json';project_file.write(entry,folder/'Project_data',saved)
        refs=folder/'references';refs.mkdir()
        files=['hkt3j_digitization.json','abolghasemi_2019_figure3_U29.csv'] if key=='otf_gpu' else ['hkt3j_digitization.json','mycek_2014_figures6a_7a.csv','mycek_2014_figure_A12_velocity.csv','mycek_2014_figure_A12_digitization.json']
        for name in files:shutil.copy2(REFERENCE/'measurements'/name,refs/name)
        with (refs/'INPUT_TSR_Ct_Cp.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['TSR','Ct','Cp']);w.writeheader()
            w.writerows(dict(TSR=r['tip_speed_ratio'],Ct=r['thrust_coefficient'],Cp=r['power_coefficient']) for r in table['performance'])
        spec=dict(case=key,configuration=c,project=str(entry),runtime_sha256=IDENTITY,reference_files={p.name:sha(p) for p in refs.iterdir()},
            control=turbine['control'],support='No additional support drag; Mycek measured assembly Ct is used as an equivalent loading input.',
            grid='One fixed level; square Cartesian interior cells, true cropped/merged wall cells. OTF uses D/8 square spacing, rather than the research reference slightly rectangular grid. Exact 1500 x 775 m domain retained. No grid sensitivity campaign.',
            initial='Uniform published depth and velocity. The research reference used a small balancing surface gradient; here the public project spins up from its representable uniform state.',
            mixing='OTF published nu=0.6. Mycek retains the frozen OTF nu/(U D) scaling, no fit to experiment; no rotor-generated turbulence.',
            limits=['Mycek raw wake is hub-plane, not depth mean; assumed axisymmetric projection reported separately.',
                'Supplied Ct/Cp table replay is not independent blade-performance prediction.',
                'The published zero/parked-TSR point remains in the raw reference CSV, but is excluded from the strictly positive prescribed-TSR input table.',
                'U=1.0 uses only U=0.8 coefficients; measured U=1.0 curve held out.',
                'Zero-thrust control keeps the same viscosity and turbine mesh with Ct=Cp=0.',
                'Published OTF curves are digitized; its author engine is not executed here.',
                'Local metre coordinates use a metric CRS tag, not an actual geographic survey.'],
            engineering_screens=dict(otf_RMSE_pp=3.,otf_max_abs_pp=6.,mycek_projected_farwake_RMSE_pp=5.,holdout_coeff_relative=.10,profile_window_change_pp=.5))
        save(folder/'VALIDATION_SPEC.json',spec)
        (folder/'README.txt').write_text('Open Validation.braids.json in BRAIDS Desktop. All model inputs, results, checkpoints and history are in Project_data.\n'
            'Select Results > Turbines for power/Ct/Cp and CSV; Profiles contains the comparison sections.\n'
            'Read VALIDATION_SPEC.json for the reference, supplied coefficients and limitations. This is hydrodynamics only.\n',encoding='utf-8')
        records[key]=dict(project=str(entry),folder=str(folder),input_project_sha256=sha(entry),configuration=c)
    save(HOME/'PROJECTS.json',records);save(REPORT/'LOCKED_SPEC.json',dict(runtime_sha256=IDENTITY,projects=records,
        fixed_before_runs=True,source_pdf_hashes={n:sha(REFERENCE/'vendor'/n) for n in ['abolghasemi_2019_depth_average.pdf','mycek_2014.pdf']}))
    print(json.dumps(records,indent=2),flush=True)


def status():
    for key,row in json.loads((HOME/'PROJECTS.json').read_text()).items():
        root=Path(row['folder'])/'Project_data';runs=sorted((root/'runs').glob('*'))
        if not runs:print(key+': not started');continue
        output=runs[-1];progress=output/'PROGRESS.json'
        try:d=json.loads(progress.read_text());print(key+': '+str(d.get('status'))+'; '+str(d.get('time_s'))+' s; '+str(d.get('phase','')))
        except (ValueError,OSError):print(key+': starting / publishing')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['create','status']);globals()[p.parse_args().action]()
