"""Frozen current-Desktop comparison with Nash's published depth-mean data."""
import argparse,hashlib,json,math,shutil,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'desktop/runtime'
HOME=Path.home()/'Desktop/BRAIDS_Turbine_Validation59'
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/desktop_validation59'
REF=ROOT/'validation/hydrokinetic_turbines'
sys.path.insert(0,str(RUNTIME));sys.path.insert(1,str(ROOT/'desktop'));sys.path.insert(2,str(ROOT))
IDENTITY='1dcf63c3b753ebbbe012313f7ee05932461aeb38d9526f22d41af6c478aa097a'
D=10.;H=20.;U=.2;L=1500.;W=600.;X=500.;Y=295.;DX=5.
CHEZY=18*math.log10(12*H/.05);AREA=math.pi*D*D/4
EQUIVALENT_CT=.8+.9*9/AREA
NU=.30*math.sqrt(9.81)*U/CHEZY*H
CASES={k:dict(folder=f,backend=b,ct=ct,D=D,H=H,U=U,L=L,W=W,X=X,Y=Y,dx=DX,dy=DX,
             end=2000.,average=1500.,output=50.,nu=NU) for k,f,b,ct in [
    ('nash_cpu','01_Nash_Equivalent_Load_CPU','cpu',EQUIVALENT_CT),
    ('nash_gpu','02_Nash_Equivalent_Load_GPU','gpu',EQUIVALENT_CT),
    ('zero_cpu','03_Matched_Zero_Thrust_CPU','cpu',0.),
    ('zero_gpu','04_Matched_Zero_Thrust_GPU','gpu',0.)]}


def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def create(single=None):
    from osgeo import gdal,osr
    from braids_desktop import services,project_file
    from braids_core.river_model import resolve_model
    from braids_core.river_turbines import default,definition
    from braids_core.boundaries.domain import PolygonDomain
    assert sha(RUNTIME/'RUNTIME.json')==IDENTITY
    if single is None:
        assert not (REPORT/'LOCKED_SPEC.json').exists()
        HOME.mkdir(exist_ok=False);REPORT.mkdir(parents=True,exist_ok=True);records={};runtime_id=IDENTITY
    else:
        records=json.loads((HOME/'PROJECTS.json').read_text());assert single not in records
        runtime_id=json.loads((REPORT/'LOCKED_SPEC.json').read_text())['runtime_sha256']
    for name,c in CASES.items():
        if single is not None and name!=single:continue
        if single is None and name=='zero_gpu':continue
        folder=HOME/c['folder'];folder.mkdir();dem=folder/'flat_channel.tif'
        ds=gdal.GetDriverByName('GTiff').Create(str(dem),302,122,1,gdal.GDT_Float64,options=['COMPRESS=LZW'])
        ds.SetGeoTransform((-DX,DX,0,W+DX,0,-DX));s=osr.SpatialReference();s.ImportFromEPSG(2056)
        ds.SetProjection(s.ExportToWkt());ds.GetRasterBand(1).Fill(0.);ds.FlushCache();ds=None
        doc=services.default_document(dem,[[0,0],[L,0],[L,W],[0,W]],'EPSG:2056',name=name,
            backend=c['backend'],minimum_size_m=DX,levels=1,mesh_mode='static_terrain',end_time_s=c['end'])
        r=doc['model']['request'];r['hydrodynamics']['roughness']='roughness.chezy';doc['model']=resolve_model(r).document()
        doc['roughness_parameters']['chezy_m_half_s']=CHEZY;doc['settings']['controls'].update(chezy=CHEZY,cfl=.5,threads=8)
        doc['settings']['grid']=dict(nx=300,ny=120,xlower_m=0.,ylower_m=0.,dx_m=DX,dy_m=DX)
        domain=PolygonDomain(**doc['settings']['domain'])
        for x,bc,kind,control in [(0.,'Upstream','inlet',dict(method='inlet.total_discharge',discharge_m3_s=U*H*W)),
            (L,'Downstream','outlet',dict(method='outlet.stage',stage_m=H))]:
            doc=services.assign_boundary(doc,[e.edge_id for e in domain.edges if e.start_xy[0]==e.end_xy[0]==x],bc,kind,control)
        doc['initial'].update(stage_m=H,velocity_m_s=[U,0.])
        doc['settings'].update(dt_max_s=5.,output_interval_s=c['output'],checkpoint_interval_s=c['end'])
        doc['settings']['adaptation'].update(domain_border_enabled=False,transition_layers=1)
        t=definition([X,Y],[1,0],name=name,diameter=D,hub_height=9.,minimum_bed_clearance=0.,minimum_surface_clearance=0.)
        t['control']=dict(mode='constant_ct',thrust_coefficient=c['ct'],power_coefficient=0.)
        doc['turbines']=default();doc['turbines'].update(turbines=[t],ambient_viscosity_m2_s=NU)
        doc=services.validate_document(doc)
        profiles=[dict(id='profile_'+str(i+1),name=f'{x:g}D transverse',coords=[[X+x*D,Y-5*D],[X+x*D,Y+5*D]]) for i,x in enumerate([2.,4.,10.,20.,40.])]
        profiles.append(dict(id='profile_6',name='Wake centreline',coords=[[X-2*D,Y],[X+40*D,Y]]))
        measurements=dict(schema=2,profiles=profiles,probes=[dict(id='probe_1',name='10D wake',coords=[[X+10*D,Y]])],selected_profile='profile_6',selected_probe='probe_1')
        session=dict(desktop=dict(model_step='Hydrokinetic turbines',result_step='Turbines',live_results=True,measurements=measurements,advanced_hydrodynamics=False))
        saved=services.save_project_folder(folder/'Project_data',doc,session=session)
        entry=folder/'Validation.braids.json';project_file.write(entry,folder/'Project_data',saved)
        refs=folder/'references';refs.mkdir()
        for file in ['nash_2015_figure6b.csv','nash_2015_figure6b.json','nash_2015_figure7.csv','nash_2015_figure7.json']:
            shutil.copy2(REF/'measurements'/file,refs/file)
        spec=dict(case=name,configuration=c,article='https://doi.org/10.3390/en81212380',reference_pdf_sha256=sha(REF/'vendor/nash_2015_2d.pdf'),
            loading=dict(rotor_Ct=.8,rotor_area_m2=AREA,support_area_m2=9.,support_Cd=.9,equivalent_total_Ct=EQUIVALENT_CT,
                identity='Ct_equivalent = Ct_rotor + Cd_support * A_support/A_rotor',
                limitation='Equivalent zero-yaw integrated loading: rotor and support share the public corrected approach velocity/footprint, not independently resolved support drag.'),
            mixing=dict(nu_m2_s=NU,coefficient=.30,identity='Frozen inlet-equivalent nu = 0.30 u_star H; u_star=sqrt(g) U/Chezy',
                limitation='Nash states PML but does not report the coefficient. 0.30 is retained from the historical audit, not fitted here. Public nu is constant, not dynamic PML.'),
            other_differences=['Frozen steady flood U=0.2 replaces author 4-cycle tide; not an exact author input replay.',
                'Uniform initial stage/velocity; no-slip sidewall stress from author is not an available public option; walls are impermeable.',
                'Public smooth bump/Kramer-Piggott actual-depth source differs from author staggered drag insertion.',
                'Public minimum rotor resolution is two cells: 5m square cells, versus author 10m. The one-cell input was rejected before preparation; guard retained.',
                'Exact rotor plane x=500 is grid-face aligned; author reference comparison is turbine-relative, no claim of identical source stencil.',
                'Published Figure6b already averages full water-column measurements. No new depth projection is applied.',
                'Cp=0 is explicit: this porous-disc/support benchmark has no power-validation target.',
                'Local metre coordinates with a metric CRS tag, not a geographic survey.'],
            frozen_before_execution=True,runtime_sha256=runtime_id,
            screens=dict(farwake_from_D=4,centreline_RMS_pp=3.,centreline_maximum_pp=6.,relative_L2_deficit=.25,
                profile_halfwindow_change_pp=.5,maximum_water_residual_m3=1e-6,
                note='Engineering comparison screens, not experimental uncertainty or permission to tune. Failure is retained.'))
        save(folder/'VALIDATION_SPEC.json',spec);records[name]=dict(project=str(entry),folder=str(folder),configuration=c,input_sha256=sha(entry))
    save(HOME/'PROJECTS.json',records)
    if single is None:
        save(REPORT/'LOCKED_SPEC.json',dict(runtime_sha256=runtime_id,projects=records,
            fixed_before_runs=True,source_pdf_sha256=sha(REF/'vendor/nash_2015_2d.pdf'),
            controls='Compare 1 - speed_with_turbine / matched_zero_thrust_speed, following Nash Figure6b normalization.'))
    else:
        d=json.loads((REPORT/'LOCKED_SPEC.json').read_text());d['projects']=records
        d['bounded_cpu_verification']='CPU companions are accepted-state stopped after >=100s for matched-field verification; full wake averaging uses two GPU cases. No physical settings changed.'
        save(REPORT/'LOCKED_SPEC.json',d)
    print('Created',HOME,flush=True)


def run(case):
    import run_turbine_validation59_ui as ui
    ui.run(case)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['create','create-zero-gpu','run']);p.add_argument('case',nargs='?');a=p.parse_args()
    if a.action=='run':run(a.case)
    else:create('zero_gpu' if a.action=='create-zero-gpu' else None)
