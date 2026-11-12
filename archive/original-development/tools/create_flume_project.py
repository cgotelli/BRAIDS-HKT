#!/usr/bin/env python3
"""Create a current standalone BRAIDS project for a rectangular laboratory flume.

Run with BRAIDS' installed Python, or use the accompanying CMD launcher.
Use --help for parameters and examples. The output is a NEW project folder;
existing files are never replaced. Lengths are metres and flow is m3/s.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

# Geometric/flow defaults are from Gotelli, Musa, Guala & Escauriaza (2019),
# https://doi.org/10.3390/en12163188. The paper's 0.42 m/s is an upstream
# velocity, not a reported cross-section mean: Q=U*h*B is a stated estimate.
DEFAULTS = dict(length=14.6, width=.9, slope=.001, depth=.3, velocity=.42,
                d50=.0018, mesh=.05, dem_pixel=.025, dem_margin=.2,
                duration=60., dt_max=.2, output_interval=5., cfl=.9,
                manning=.03, erodible_depth=.1, rotor_diameter=.092,
                hub_height=.1, ct=.8, cp=.4, wake_diameters=10.)


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=('Examples (Windows launcher):\n'
                '  Create flume.cmd --output "C:\\Flumes\\Sediment" --sediment\n'
                '  Create flume.cmd --output "C:\\Flumes\\Two turbines" --turbine 6.448 0.45 --turbine 7 0.45\n'
                '  Create flume.cmd --output "C:\\Flumes\\Custom" --length 8 --width 1.2 --discharge 0.08 --mesh .08\n'))
    p.add_argument('--output',type=Path,default=Path.home()/'Desktop'/'BRAIDS_Minnesota_Flume',help='New project directory; default: Desktop/BRAIDS_Minnesota_Flume')
    p.add_argument('--runtime',type=Path,default=Path(os.environ.get('LOCALAPPDATA',str(Path.home()))) / 'Programs/BRAIDS-HKT/runtime',help='Installed BRAIDS runtime directory')
    p.add_argument('--name',default='Minnesota flume',help='Project name')
    for flag,key,help_text in (
        ('--length','length','Flume length (m)'),('--width','width','Internal wall-to-wall width (m)'),
        ('--slope','slope','Downstream bed slope (m/m)'),('--depth','depth','Target downstream water depth (m)'),
        ('--velocity','velocity','Initial velocity and assumed cross-section mean when Q is omitted (m/s)'),
        ('--mesh','mesh','Minimum ordinary square mesh size (m)'),('--dem-pixel','dem_pixel','DEM pixel size (m)'),
        ('--dem-margin','dem_margin','DEM extends beyond each wall by at least this distance (m)'),
        ('--duration','duration','Simulation duration (s)'),('--dt-max','dt_max','Maximum outer interval (s)'),
        ('--cfl','cfl','Maximum explicit CFL, >0 and <=1'),
        ('--manning','manning','Manning n (s/m^(1/3)); assumed unless measured'),
        ('--d50','d50','Single bulk sediment D50 (m)'),('--erodible-depth','erodible_depth','Initial available erodible bed thickness (m)'),
        ('--rotor-diameter','rotor_diameter','Turbine rotor diameter (m)'),('--hub-height','hub_height','Turbine hub height above local bed (m)'),
        ('--ct','ct','Illustrative constant thrust coefficient; replace with device data'),
        ('--cp','cp','Illustrative constant power coefficient; replace with device data'),
        ('--wake-diameters','wake_diameters','Optional wake-refinement length in rotor diameters')):
        p.add_argument(flag,type=float,default=DEFAULTS[key],help=f'{help_text}; default {DEFAULTS[key]:g}')
    p.add_argument('--discharge',type=float,default=None,help='Inlet water discharge (m3/s); otherwise velocity * depth * width is an approximation')
    p.add_argument('--output-interval',type=float,default=None,help='Results output interval (s); default: smaller of 5 s and total duration')
    p.add_argument('--backend',choices=('cpu','gpu'),default='cpu')
    p.add_argument('--solver',choices=('hllc','fwave'),default='hllc')
    p.add_argument('--sediment',action='store_true',help='Enable one-material BRAIDS standard MPM bedload')
    p.add_argument('--sediment-feed',choices=('zero','capacity'),default='zero',help='Inlet bedload condition when sediment is enabled')
    p.add_argument('--turbines',action='store_true',help='Add one illustrative turbine at x=7 m (or the flume midpoint for a shorter flume)')
    p.add_argument('--turbine',nargs=2,type=float,metavar=('X','Y'),action='append',default=[],help='Add a turbine at local x,y (m); repeat for an array')
    p.add_argument('--turbine-control',type=Path,help='JSON file containing a BRAIDS turbine control, for measured Ct/Cp data')
    p.add_argument('--no-wake-refinement',action='store_true',help='Disable automatic downstream wake refinement')
    return p


def validate(a,p):
    if a.output_interval is None:a.output_interval=min(DEFAULTS['output_interval'],a.duration)
    finite=('length','width','depth','mesh','dem_pixel','dem_margin','duration','dt_max',
            'output_interval','cfl','manning','d50','rotor_diameter','hub_height','wake_diameters')
    for key in finite:
        v=getattr(a,key)
        if not math.isfinite(v) or v<=0:p.error('--'+key.replace('_','-')+' must be positive and finite')
    if not math.isfinite(a.slope) or a.slope<0:p.error('--slope must be finite and nonnegative')
    if not math.isfinite(a.velocity) or a.velocity<0:p.error('--velocity must be finite and nonnegative')
    if not math.isfinite(a.erodible_depth) or a.erodible_depth<0:p.error('--erodible-depth must be finite and nonnegative')
    if a.cfl>1:p.error('--cfl must not exceed 1 for an explicit solver')
    if a.discharge is not None and (not math.isfinite(a.discharge) or a.discharge<0):p.error('--discharge must be finite and nonnegative')
    if a.sediment and (a.turbines or a.turbine):
        p.error('The installed BRAIDS solver does not yet support coupled sediment and turbines. Generate separate sediment and turbine projects; neither feature will be silently disabled.')
    if not a.sediment and a.sediment_feed!='zero':p.error('--sediment-feed requires --sediment')
    if a.turbine_control and not (a.turbines or a.turbine):p.error('--turbine-control requires --turbines or --turbine')
    if a.output_interval>a.duration:p.error('--output-interval cannot exceed --duration')
    if a.mesh>min(a.length,a.width):p.error('--mesh must not exceed the flume length or width')
    if a.rotor_diameter>=a.width and (a.turbines or a.turbine):p.error('The rotor diameter must be smaller than the flume width')
    if not 0<=a.ct<1 or not 0<=a.cp<=16/27:p.error('Require 0 <= Ct < 1 and 0 <= Cp <= 16/27')
    output=a.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):p.error('Choose a new or empty output directory; existing files are not replaced')


def create(a):
    # Import the installed application after argument validation, so --help
    # works without its scientific Python environment.
    runtime=a.runtime.expanduser().resolve()
    if not (runtime/'RUNTIME.json').is_file():raise RuntimeError('BRAIDS runtime not found: '+str(runtime))
    sys.path.insert(0,str(runtime))
    import numpy as np
    from osgeo import gdal,osr
    from braids_core.river_launcher import verify
    verify()
    from braids_desktop import services,project_file
    from braids_core.boundaries.domain import PolygonDomain
    from braids_core.river_model import resolve_model,new_bedload_selection,HLLC_INTEGRATOR,INTEGRATOR
    from braids_core.sediment.river_column_methods import method_ids
    from braids_core.sediment.bed_columns_v101 import ColumnMethodsV101
    from braids_core.river_turbines import definition,default

    Q=a.discharge if a.discharge is not None else a.velocity*a.width*a.depth
    positions=a.turbine or ([[7. if a.length>7.5 else a.length/2.,a.width/2.]] if a.turbines else [])
    with tempfile.TemporaryDirectory(prefix='braids-flume-') as temp:
        dem=Path(temp)/'flume_bed.tif';p=a.dem_pixel;margin=a.dem_margin
        nx=math.ceil((a.length+2*margin)/p);ny=math.ceil((a.width+2*margin)/p)
        if nx*ny>100_000_000:raise ValueError('The DEM would exceed 100 million pixels; increase --dem-pixel')
        ds=gdal.GetDriverByName('GTiff').Create(str(dem),nx,ny,1,gdal.GDT_Float64)
        if ds is None:raise RuntimeError('Cannot create flume DEM')
        ds.SetGeoTransform((-margin,p,0.,a.width+margin,0.,-p))
        srs=osr.SpatialReference();srs.SetLocalCS('BRAIDS laboratory flume');srs.SetLinearUnits('metre',1.)
        wkt=srs.ExportToWkt();ds.SetProjection(wkt)
        x=-margin+(np.arange(nx)+.5)*p
        ds.GetRasterBand(1).WriteArray(np.broadcast_to(-a.slope*x,(ny,nx)).copy())
        ds.FlushCache();ds=None
        # Domain boundaries, not terrain banks, provide vertical side walls.
        points=[[0.,0.],[a.length,0.],[a.length,a.width],[0.,a.width]]
        doc=services.default_document(dem,points,wkt,name=a.name,backend=a.backend,
            minimum_size_m=a.mesh,levels=1,mesh_mode='static_manual',end_time_s=a.duration)
        request=doc['model']['request'];request['execution']['integrator']=HLLC_INTEGRATOR if a.solver=='hllc' else INTEGRATOR
        if a.sediment:
            selection=new_bedload_selection();selection['grains'][0]['diameter_m']=a.d50
            selection['capacity']['grain_sizes_m']={'d50':a.d50}
            request['sediment']=dict(enabled=True,selection=selection)
        doc['model']=resolve_model(request).document()
        s=doc['settings'];s['controls'].update(diameter=a.d50,cfl=a.cfl,morfac=1.)
        s.update(dt_max_s=a.dt_max,output_interval_s=a.output_interval,checkpoint_interval_s=a.duration)
        s['adaptation'].update(open_boundary_enabled=False,domain_border_enabled=False,terrain_enabled=False)
        doc['roughness_parameters']['manning_s_m_third']=a.manning
        # A uniform water level over a sloping bed starts close to, but is not,
        # uniform depth; the user can change it later in BC & IC.
        outlet_stage=a.depth-a.slope*a.length
        doc['initial'].update(stage_m=outlet_stage,velocity_m_s=[a.velocity,0.],
                              layer_thickness_m=[a.erodible_depth],layer_fractions=[[1.]])
        columns=s['columns'];columns.update(bed='mixed',diameter_m=[a.d50]);doc['column_methods']=method_ids(ColumnMethodsV101(**columns))
        domain=PolygonDomain(**s['domain'])
        for x,name,kind,control in ((0.,'Inlet','inlet',dict(method='inlet.total_discharge',discharge_m3_s=Q)),
                                     (a.length,'Outlet','outlet',dict(method='outlet.stage',stage_m=outlet_stage))):
            edges=[e.edge_id for e in domain.edges if e.start_xy[0]==e.end_xy[0]==x]
            doc=services.assign_boundary(doc,edges,name,kind,control)
        if a.sediment:
            doc['settings']['sediment_boundaries']['Inlet']=(
                dict(kind='flow_free_equilibrium',concentration_kg_m3=[0.]) if a.sediment_feed=='capacity' else
                dict(kind='inlet_total_solid_discharge',solid_discharge_m3_s=[0.],concentration_kg_m3=[0.]))
        if positions:
            rows=[];control=None
            if a.turbine_control is not None:control=json.loads(a.turbine_control.read_text(encoding='utf8'))
            for i,(x,y) in enumerate(positions,1):
                row=definition([x,y],[1.,0.],name=f'Turbine {i}',diameter=a.rotor_diameter,hub_height=a.hub_height,
                               minimum_bed_clearance=0.,minimum_surface_clearance=0.)
                row['control']=control if control is not None else dict(mode='constant_ct',thrust_coefficient=a.ct,power_coefficient=a.cp)
                row['wake_refinement']=dict(enabled=not a.no_wake_refinement,length_diameters=a.wake_diameters)
                rows.append(row)
            doc['turbines']=default();doc['turbines'].update(turbines=rows,ambient_viscosity_m2_s=0.)
        doc=services.validate_document(services._resolve_boundary_state(doc))
        output=a.output.expanduser().resolve();output.mkdir(parents=True,exist_ok=True)
        if any(output.iterdir()):raise FileExistsError('Output folder is no longer empty; no files were replaced')
        data=output/'Flume_data';saved=services.save_project_folder(data,doc)
        project=output/'Flume.braids.json';project_file.write(project,data,saved)
        metadata=dict(source='https://doi.org/10.3390/en12163188',flume_length_m=a.length,flume_width_m=a.width,
            slope_m_per_m=a.slope,target_downstream_depth_m=a.depth,discharge_m3_s=Q,
            discharge_basis='user supplied' if a.discharge is not None else 'estimated as velocity × width × depth; reported velocity is not established as cross-section mean',
            roughness_manning=a.manning,roughness_basis='user-selected assumption; not established by source',
            grain_d50_m=a.d50,sediment_enabled=a.sediment,turbine_count=len(positions),
            turbine_performance_basis=('user-provided control' if a.turbine_control else 'illustrative assumed Ct/Cp, not experimental device data') if positions else 'not active',
            hub_height_basis='user-selected assumption; 2019 table states 0.010 m, inconsistent with its 0.092 m rotor diameter',
            initial_surface='uniform absolute stage based on downstream bed and target depth; sloping bed means nonuniform initial depth',
            channel_walls='vertical no-through-flow domain boundaries; DEM is flat across channel and extends beyond walls',
            coordinate_system='arbitrary local laboratory metres, not a geographic location',solver=a.solver,backend=a.backend)
        (output/'FLUME_PARAMETERS.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('Project:',project)
    print('Domain:',f'{a.length:g} m x {a.width:g} m; straight walls are domain boundaries.')
    print('Inlet Q:',f'{Q:g} m3/s', '(estimated from U * depth * width)' if a.discharge is None else '(specified)')
    print('Physics:', 'bedload (MPM)' if a.sediment else f'{len(positions)} turbine(s)' if positions else 'hydrodynamics only')
    print('Assumptions and reference:',output/'FLUME_PARAMETERS.json')
    return project


def main(argv=None):
    p=parser();a=p.parse_args(argv);validate(a,p)
    try:create(a)
    except (ValueError,RuntimeError,OSError) as exc:p.exit(2,f'Cannot create project: {exc}\n')


if __name__=='__main__':main()
