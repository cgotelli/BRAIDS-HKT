"""Short Bern reach; supplied DEM unchanged except lossless spatial crop."""
import copy,time
from pathlib import Path
from hub_bern77 import ROOT,TARGET,REPORT,read,save,sha,environment
environment()
import numpy as np
from osgeo import gdal
from braids_desktop import services,preparation,project_file
from braids_core import river_turbines,river_worker
from braids_core.boundaries.domain import PolygonDomain

HOME=Path.home()/'Desktop/BRAIDS_Bern_Turbines77'
def main():
    HOME.mkdir(exist_ok=True);data=HOME/'Bern_data';inputs=HOME/'source';inputs.mkdir(exist_ok=True)
    if (data/'runs/startup_check/PROGRESS.json').exists():
        result=read(data/'runs/startup_check/PROGRESS.json')
        assert result['status']=='segment_complete' and result['time_s']==2.
        elapsed=read(data/'prepared/initial/DESKTOP_PREPARATION.json')['timings']['total_s']
        finish(result,elapsed);return
    source=Path.home()/'Desktop/BERN/BERN.tif';dem=inputs/'Bern_reach.tif'
    ds=gdal.Open(str(source));projection=ds.GetProjection()
    gdal.Translate(str(dem),ds,projWin=[2600360,1198832,2600544,1198528],creationOptions=['COMPRESS=LZW','TILED=YES'])
    vertices=[[2600392,1198560],[2600512,1198560],[2600512,1198800],[2600392,1198800]]
    d=services.default_document(dem,vertices,projection,name='Bern - short northbound reach (demonstration)',backend='cpu',minimum_size_m=8.,levels=1,end_time_s=300.)
    d['initial'].update(stage_m=502.3,velocity_m_s=[.288,1.8])
    d['roughness_parameters']['manning_s_m_third']=.03
    d['settings']['adaptation'].update(terrain_enabled=True,domain_border_enabled=False,transition_layers=1)
    d['settings']['controls'].update(cfl=.5,threads=4)
    d['settings'].update(dt_max_s=.2,output_interval_s=2.,checkpoint_interval_s=300.)
    domain=PolygonDomain(**d['settings']['domain'])
    inlet=next(e.edge_id for e in domain.edges if e.start_xy[1]==e.end_xy[1]==1198560)
    outlet=next(e.edge_id for e in domain.edges if e.start_xy[1]==e.end_xy[1]==1198800)
    d=services.assign_boundary(d,[inlet],'South inlet','inlet',dict(method='inlet.critical_discharge',discharge_m3_s=150.))
    d=services.assign_boundary(d,[outlet],'North outlet','outlet',dict(method='outlet.stage',stage_m=502.3))
    n=np.array([.16,1.]);n/=np.linalg.norm(n);t=np.array([-n[1],n[0]]);center=np.array([2600452.,1198656.]);D=.7
    reference=read(ROOT/'desktop/qualification/coupled_wake76/staggered_gpu/input.json')['turbines']['turbines'][0]['control']['wake']
    rows=[]
    for label,xy in [('West upstream',center+.7*t),('East upstream',center-.7*t),('Downstream',center+7*D*n)]:
        row=river_turbines.definition(xy,n,name=label,diameter=D,hub_height=.5,minimum_bed_clearance=.1,minimum_surface_clearance=.1)
        row['control']=dict(mode='fixed_rpm_surface',model_name='DEMONSTRATION constant Ct 0.8 / Cp 0.4; not manufacturer data',rotational_speed_rpm=90.,outside_surface='error',
            performance=[dict(inflow_m_s=u,curve=[dict(tip_speed_ratio=tsr,thrust_coefficient=.8,power_coefficient=.4) for tsr in [.001,100.]]) for u in [.001,10.]],
            wake=dict(reference,provenance='Illustrative transferred laboratory high-TI wake shape from milestone76; NOT calibrated for the Aare or a commercial turbine. Constant Ct/Cp preserve the supplied Bern demonstration assumptions.'))
        row['wake_refinement']=dict(enabled=True,length_diameters=10.)
        rows.append(row)
    d['turbines']=dict(river_turbines.default(),turbines=rows,ambient_viscosity_m2_s=.006)
    d=services.save_project_folder(data,d,source_base=HOME)
    project=HOME/'Bern.braids.json';project_file.write(project,data,d)
    prepared=data/'prepared/initial';start=time.monotonic()
    preparation.prepare(data/'project.json' if (data/'project.json').exists() else write_input(d),prepared,asset_cache=TARGET.parent/'assets')
    elapsed=time.monotonic()-start
    # Bounded startup smoke only. The editable project remains configured for 300 s.
    result=river_worker.run(prepared,data/'runs/startup_check',until=2.)
    assert result['status'] in ('complete','segment_complete'),result
    finish(result,elapsed)

def finish(result,elapsed):
    data=HOME/'Bern_data';project=HOME/'Bern.braids.json';source=Path.home()/'Desktop/BERN/BERN.tif';dem=HOME/'source/Bern_reach.tif'
    session=dict(prepared='prepared/initial',output='runs/startup_check',desktop=dict(model_step='Hydrokinetic turbines',result_step='Turbines'))
    services.update_project_session(data,**session)
    launch=HOME/'Open Bern project.cmd';launch.write_text('@echo off\ncall "'+str(ROOT/'Launch BRAIDS Desktop.cmd')+'" "'+str(project)+'" %*\nexit /b\n')
    save(REPORT/'BERN.json',dict(project=str(project),shortcut=str(launch),source_dem=str(source),source_sha256=sha(source),cropped_dem=str(dem),cropped_sha256=sha(dem),prepare_seconds=elapsed,result=result,
        assumptions=dict(discharge_m3_s=150.,manning=.03,stage_m=502.3,initial_velocity_m_s=[.288,1.8],rotor_diameter_m=.7,ordinary_cell_m=8.,rotor_cell_cap_m=.35,wake_length_D=10.,bathymetry_verified=False,device_map='explicit constant Ct/Cp demonstration surface',wake_calibration='transferred laboratory example, not site calibration')))
    print('Bern project ready:',project,flush=True)

def write_input(d):
    p=TARGET.parent/'bern_input.json';save(p,d);return p
if __name__=='__main__':main()
