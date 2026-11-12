"""Exercise all ten target input combinations without a native solver step."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np

from braids_core.boundaries.cut_cell_io import CutSedimentFraction,write_cut_case
from braids_core.boundaries.cut_cells import map_cut_cells
from braids_core.boundaries.domain import PolygonDomain
from braids_core.boundaries.inlet_io import NativeGrid
from braids_gpu.delft_components_exporter import verify_source_exporter
from braids_gpu.graded_native_exporter import export_environment
from validation.gpu_fwave.coupled_reference import read_records


def run():
    identity=verify_source_exporter(ROOT)
    destination=Path(__file__).resolve().parent/'smoke_v2'
    destination.mkdir(exist_ok=False)
    domain=PolygonDomain('closed',[(0.,0.),(4.,0.),(4.,2.),(0.,2.)],'EPSG:2056')
    mesh=map_cut_cells(domain,[],NativeGrid(4,2,0.,0.,1.,1.))
    nc=len(mesh.cells);initial=np.zeros((nc,9));initial[:,0]=.1;initial[:,1]=.04
    bulk=2650.*.625
    initial[:,5:7]=.5*.02*bulk;initial[:,7:9]=.5*.08*bulk
    fractions=tuple(CutSedimentFraction(fraction_id=f'sand_{i}',d50_m=.001,d90_m=.0015,
        grain_density_kg_m3=2650.,porosity=.375,settling_velocity_m_s=0.,
        minimum_bedload_depth_m=.001,initial_volume_fraction=.5,
        initial_active_volume_fraction=.5) for i in range(2))
    build=json.loads((ROOT/'solver/cut_cells/build/build.json').read_text())
    environment=export_environment(identity['runtime'])
    environment['PATH']=str(Path(identity['build']['command'][0]).parent)+os.pathsep+environment['PATH']
    # The initializer also exits when its historical trigger is disabled.
    environment['BRAIDS_GPU_INITIAL_EXPORT']='0'
    rows=[]
    configurations=[(law,slope,'off') for law in (2,8) for slope in (0,2,3)]
    configurations.extend((law,3 if law==8 else 2,secondary)
                          for law in (2,8) for secondary in ('equilibrium','prognostic'))
    for law,delft_slope,secondary in configurations:
        native_slope=5 if law==8 and delft_slope==2 else delft_slope
        folder=destination/f'law{law}_slope{delft_slope}_{secondary}'
        folder.mkdir()
        kf=native_slope==3
        write_cut_case(folder/'case.data',mesh,initial,np.full(nc,-.1),{},
            end_time_s=1.,dt_max_s=.005,output_interval_s=.5,cfl=.45,scheme=6,
            bedload=law,exchange=0,sediment_fractions=fractions,
            active_layer_thickness_m=.02,substrate_layer_count=1,
            substrate_layer_thickness_m=.08,manning=.025,
            engelund_hansen_chezy=39. if law==8 else None,
            mpm_friction_mode=1 if law==2 else 0,
            mpm_critical_shields=.047,
            bed_slope=native_slope,bed_slope_alfabs=1. if native_slope!=0 else 0.,
            bed_slope_alfabn=.85 if kf else 1.5,
            bed_friction_angle_deg=.5 if kf else 30.,
            secondary_flow=secondary!='off',
            secondary_flow_model='prognostic' if secondary=='prognostic' else 'equilibrium',
            secondary_flow_momentum_beta=.5 if secondary=='prognostic' else 0.,
            horizontal_viscosity_m2_s=1.e-6,
            horizontal_viscosity_formulation='velocity_laplacian',
            compact_sidecars=True,source_sha256=build['source_sha256'])
        result=subprocess.run([identity['executable'],'case.data'],cwd=folder,env=environment,
            capture_output=True,text=True,timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        (folder/'initial.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        result.check_returncode()
        assert 'no solver timestep performed' in result.stderr+result.stdout
        data=read_records(folder/'coupled_input.bin')
        expected_model={'off':0,'equilibrium':1,'prognostic':2}[secondary]
        assert data['bedlaw']==law and data['bed_slope']==native_slope
        assert data['secondary_flow_model']==expected_model and data['time']==0.
        assert data['engelund_hansen_chezy']==(39. if law==8 else 0.)
        np.testing.assert_array_equal(data['state'][:9],initial.T)
        if expected_model==2:np.testing.assert_array_equal(data['state'][9],np.zeros(nc))
        np.testing.assert_array_equal(data['manning_cell'],np.full(nc,.025))
        normal=read_records(folder/'normal_metadata.bin')
        diffusion=read_records(folder/'coupled_diffusion.bin')
        assert normal['nb']==data['nb'] and diffusion['rate'].shape==(nc,)
        artifacts={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in folder.iterdir() if path.suffix=='.bin'}
        rows.append(dict(name=folder.name,law=law,Delft3D_ISlope=delft_slope,
            native_slope=native_slope,secondary_model=expected_model,
            initial_state_unchanged=True,native_time_s=data['time'],
            solver_steps=0,diffusion_coefficients_exported=True,artifacts=artifacts))
    receipt=dict(schema=1,scope='Native initialization/metadata proof only; no CPU or GPU physical trajectory',
        exporter_executable_sha256=identity['executable_sha256'],
        old_export_environment_trigger_disabled=True,unconditional_zero_step_verified=True,
        rows=rows,count=len(rows),Delft3D_source_modified=False)
    (destination/'RESULT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(count=len(rows),result=str(destination/'RESULT.json'))))
    return receipt


if __name__=='__main__':run()
