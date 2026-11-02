"""Execute stock one-layer FLOW and native BRAIDS; input changes only."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
from netCDF4 import Dataset
from . import experiment as e
from .build_package import native_mesh, delft_geometry, write_json
from .digitize import sha256
from braids_support.sediment.synthetic_expansion import run_comparison as shared
from braids_core.boundaries.cut_cell_io import CutBoundary, CutSedimentFraction, write_cut_case

HERE = e.HERE
ROOT = HERE.parents[2]
RHOS, POROSITY = 2650., .4
BULK = RHOS * (1 - POROSITY)
D50, D90 = e.P['d50_m'], 1.5 * e.P['d50_m']
RESERVOIR, ACTIVE = .5, .02
VISCOSITY, BETA, ALFABS, ALFABN = .001, 1., 1., 1.5


def mpm_capacity(h, u):
    """Solid volume per width, FLOW tranb2 law (not the paper's smoothing)."""
    h, u = np.asarray(h), np.asarray(u)
    chezy = h ** (1 / 6) / e.MANNING
    grain = 18 * np.log10(12 * np.maximum(h / D90, 1))
    ripple = np.minimum((chezy / grain) ** 1.5, 1)
    excess = np.maximum(ripple * (u / chezy) ** 2 / (1.65 * D50) - .047, 0)
    return 8 * np.sqrt(9.81 * 1.65 * D50 ** 3) * excess ** 1.5


FEED = float(mpm_capacity(e.DEPTH, e.Q / (e.WIDTH * e.DEPTH)) * RHOS * e.WIDTH)


def write(path, text):
    path.write_text(text.rstrip() + '\n', encoding='utf-8')


def provenance(folder, engine, options):
    record = dict(
        engine=engine, Q=e.Q, manning=e.MANNING, feed_kg_s=FEED,
        porosity=POROSITY, d50_m=D50, d90_m=D90, mobile_thickness_m=RESERVOIR,
        viscosity_m2_s=VISCOSITY, secondary_momentum_beta=BETA,
        longitudinal_slope_factor=ALFABS, transverse_slope_factor=ALFABN,
        engine_source_modified=False,
        reconstruction='Fixed feed from MPM capacity at h=.20m, U=.50m/s; not a reported measured feed',
        grain_note='One nominal grain class; assumed D90=1.5 D50 for common default grain roughness',
        files={p.name: sha256(p) for p in folder.iterdir() if p.is_file()},
        runner_sha256=sha256(Path(__file__)), specification_sha256=sha256(HERE/'specification.json'))
    record.update(options)
    write_json(folder / 'input_provenance.json', record)


def prepare_native(folder, *, dx, end, output, dt, morfac, hydraulic, seed=None,
                   cfl=.4, secondary_momentum_beta=BETA, scheme=6):
    folder.mkdir(parents=True, exist_ok=False)
    _, build = shared.verified_braids_build()
    mesh, _ = native_mesh(dx)
    xy = np.array([c.centroid_xy for c in mesh.cells])
    s, n = e.sn(*xy.T)
    z = e.initial_bed(s)
    q = np.zeros((len(s), 9)); q[:, 0] = e.DEPTH
    theta = np.clip((s-e.APPROACH)/e.RADIUS, 0, e.ANGLE)
    q[:, 1] = e.Q/e.WIDTH*np.cos(theta)
    q[:, 2] = -e.Q/e.WIDTH*np.sin(theta)  # clockwise bend
    q[:, 5:7] = BULK*ACTIVE/2; q[:, 7:9] = BULK*(RESERVOIR-ACTIVE)/2
    if seed:
        old = np.load(seed/'analysis_geometry.npz')
        np.testing.assert_allclose(old['xy'], xy, atol=1e-12)
        q = shared.read_frames(shared.native_frames_path(seed))[-1]['q'][:, :9].copy()
    material = tuple(CutSedimentFraction(fraction_id=f'sand_{k}', d50_m=D50, d90_m=D90,
        grain_density_kg_m3=RHOS, porosity=POROSITY, settling_velocity_m_s=0.,
        z0_m=D50/30, minimum_bedload_depth_m=.001, roughness_mode='manning',
        initial_volume_fraction=.5, initial_active_volume_fraction=.5) for k in range(2))
    boundaries = dict(inlet=CutBoundary('total_discharge', discharge_m3_s=e.Q,
        fraction_bed_input_kg_s={f'sand_{k}': 0 if hydraulic else FEED/2 for k in range(2)}),
        outlet=CutBoundary('stage', stage_m=e.DEPTH))
    meta = write_cut_case(folder/'case.data', mesh, q, z-RESERVOIR, boundaries,
        end_time_s=end, dt_max_s=dt, output_interval_s=output, cfl=cfl, scheme=scheme,
        dry_tolerance_m=.0001, bedload=0 if hydraulic else 2, exchange=0, manning=e.MANNING,
        sediment_fractions=material, active_layer_thickness_m=ACTIVE,
        substrate_layer_count=1, substrate_layer_thickness_m=RESERVOIR-ACTIVE,
        mpm_friction_mode=0 if hydraulic else 1, mpm_critical_shields=.047,
        horizontal_viscosity_m2_s=VISCOSITY, horizontal_viscosity_formulation='velocity_laplacian',
        bed_slope=0 if hydraulic else 2, bed_slope_alfabs=ALFABS, bed_slope_alfabn=ALFABN,
        secondary_flow=True, secondary_flow_model='prognostic',
        secondary_flow_transport_direction=not hydraulic, secondary_flow_momentum_beta=secondary_momentum_beta,
        morphological_factor=morfac, source_sha256=build['source_sha256'], compact_sidecars=True)
    mesh.write_geometry(folder/'geometry.npz')
    np.savez_compressed(folder/'analysis_geometry.npz', xy=xy, s=s, n=n, bed=z,
                        area=np.array([c.area_m2 for c in mesh.cells]))
    write_json(folder/'preparation.json', meta)
    provenance(folder, 'braids', dict(dx=dx,end=end,output=output,dt=dt,morfac=morfac,
        hydraulic=hydraulic,seed=str(seed),cells=len(s),secondary_seed='reinitialized by public writer',
        cfl=cfl,scheme=scheme,secondary_momentum_beta=secondary_momentum_beta,
        physical_material='two identical 50/50 labels; one physical grain, no sorting'))


def prepare_delft(folder, *, dx, end, output, dt, morfac, hydraulic, seed=None):
    folder.mkdir(parents=True, exist_ok=False)
    w = lambda name, text: write(folder/name, text)
    grid = delft_geometry(dx)
    se, ne, gx, gy = (grid[k] for k in ('s','n','x','y'))
    nx, ny, nk = len(se)-1, len(ne)-1, 1
    w('basin.grd', '\n'.join(('Coordinate System = Cartesian', f'{nx+1:8d}{ny+1:8d}',
        '       0       0       0', *shared.common._grid_rows(gx), *shared.common._grid_rows(gy))))
    w('basin.enc', '\n'.join(f'{i:6d}{j:6d}' for i,j in ((1,1),(nx+2,1),(nx+2,ny+2),(1,ny+2),(1,1))))
    s,n = np.meshgrid((se[1:]+se[:-1])/2, (ne[1:]+ne[:-1])/2, indexing='ij')
    z=e.initial_bed(s); pad=lambda a: np.pad(a,1,mode='edge')
    initial=[pad(z+e.DEPTH), pad(np.full_like(z,e.Q/e.WIDTH/e.DEPTH)),pad(np.zeros_like(z)),
             pad(np.zeros_like(z)),pad(np.zeros_like(z))]
    if seed:
        with Dataset(seed/'trim-basin.nc') as d:
            z=-np.asarray(d['DPS'][-1,1:-1,1:-1])
            # Public initial-condition deck, not binary engine restart. Reset
            # secondary intensity in BOTH engines; qualify this short transient.
            initial=[np.asarray(d['S1'][-1]),np.asarray(d['U1'][-1,0]),np.asarray(d['V1'][-1,0]),
                     pad(np.zeros_like(z)),pad(np.zeros_like(z))]
    np.savetxt(folder/'basin.dep', -pad(z).T, fmt='%.17e')
    np.savetxt(folder/'initial.ini', np.concatenate([a.T for a in initial]), fmt='%.17e')
    w('basin.bnd',f'inlet                T T {1:5d} {2:5d} {1:5d} {ny+1:5d} 0.0 Uniform\n'
      f'outlet               Z T {nx+2:5d} {2:5d} {nx+2:5d} {ny+1:5d} 0.0 Uniform')
    w('basin.bct',shared.common._boundary_table(1,'inlet','total discharge (t)','m3/s',e.Q,end/60)+'\n'+
      shared.common._boundary_table(2,'outlet','water elevation (z)','m',e.DEPTH,end/60))
    w('basin.bcc','\n'.join(shared.common._sediment_boundary_table(i,name,0,end/60)
                          for i,name in enumerate(('inlet','outlet'),1)))
    w('basin.sed',f'''[SedimentFileInformation]
 FileVersion = 02.00
[SedimentOverall]
 Cref = 1e6
 IopSus = 0
[Sediment]
 Name = #Sediment 1#
 SedTyp = gravel
 RhoSol = {RHOS}
 SedDia = {D50}
 CDryB = {BULK}
 IniSedThick = {RESERVOIR}
 FacDSS = 1
 TraFrm = #formula.tra#''')
    w('formula.tra','[TransportFormulaFileInformation]\n FileVersion = 01.00\n[TransportFormula]\n Number = 2\n ACal = 1')
    rate=0 if hydraulic else FEED/BULK/e.WIDTH
    w('basin.bcm','\n'.join(("table-name 'Reconstructed fixed MPM feed'","contents 'Uniform'","location 'inlet'",
      "time-function 'non-equidistant'","reference-time 20010601","time-unit 'minutes'","interpolation 'linear'",
      "parameter 'time' unit '[min]'","parameter 'transport incl pores Sediment 1' unit '[m2/s]'",
      'records-in-table 2',f'0 {rate:.17g}',f'{end/60:.17g} {rate:.17g}')))
    w('basin.mor',f'''[MorphologyFileInformation]
 FileVersion = 02.00
[Morphology]
 MorFac = {morfac}
 MorStt = 0
 Thresh = 0
 MorUpd = {'false' if hydraulic else 'true'}
 EqmBc = false
 DensIn = false
 ISlope = 2
 AlfaBs = {ALFABS}
 AlfaBn = {ALFABN}
 Espir = 1
 IopKCW = 1
 Sus = 0
 Bed = {0 if hydraulic else 1}
 SusW = 0
 BedW = 0
 SedThr = .001
 ThetSD = 0
 Hmaxth = .001
 WetSlope = 1000000
 DuneAvalan = false
 EpsPar = false
 BcFil = #basin.bcm#
[Boundary]
 Name = #inlet#
 IBedCond = 4
[Boundary]
 Name = #outlet#
 IBedCond = 0
[Output]
 Default = true
 TranspType = 2
 BedTranspDueToCurrentsAtZeta = true
 BedTranspDueToCurrentsAtFlux = true
 Taub = true''')
    w('basin.crs',f'inlet                1 2 1 {ny+1}\noutlet               {nx+1} 2 {nx+1} {ny+1}')
    spec=dict(geometry=dict(delft3d_nx=nx,delft3d_ny=ny,narrow_bottom_width_m=e.WIDTH),
              hydraulics=dict(discharge_m3_s=e.Q,manning_n_s_m13=e.MANNING),time=dict(delft3d_dt_s=dt))
    mdf=shared.delft_mdf(spec,end,output)
    for key,val in dict(MNKmax=f'{nx+2} {ny+2} 1',Thick='100',Sub1='#  I #',Betac=str(BETA),Equili='#N#',
        Vicouv=str(VISCOSITY),Dryflc='.0001',Tempw='18',Runtxt='#LFM 2D fixed walls MPM#').items():
        mdf=re.sub(r'(?m)^'+key+r'\s*=.*$',f'{key} = {val}',mdf)
    mdf=re.sub(r'(?m)^(FilbcQ|FmtbcQ).*\n','',mdf)
    w('basin.mdf',mdf)
    w('config_d_hydro.xml','''<?xml version="1.0"?>
<deltaresHydro xmlns="http://schemas.deltares.nl/deltaresHydro">
 <documentation><fileVersion>1.00</fileVersion><createdBy>BRAIDS validation</createdBy></documentation>
 <control><sequence><start>bend</start></sequence></control>
 <flow2D3D name="bend"><library>flow2d3d</library><mdfFile>basin.mdf</mdfFile></flow2D3D>
</deltaresHydro>''')
    np.savez_compressed(folder/'analysis_geometry.npz',s=s,n=n,bed=e.initial_bed(s),gx=gx,gy=gy)
    provenance(folder,'delft',dict(dx=dx,end=end,output=output,dt=dt,morfac=morfac,
        hydraulic=hydraulic,seed=str(seed),nx=nx,ny=ny,nk=nk,secondary_seed='zero'))


def execute_native(folder,threads,timeout_s=7200):
    exe,build=shared.verified_braids_build(); start=time.perf_counter()
    with (folder/'native.log').open('w') as log:
        p=subprocess.run([str(exe),'case.data'],cwd=folder,stdout=log,stderr=subprocess.STDOUT,timeout=timeout_s,
            env={**os.environ,'OMP_NUM_THREADS':str(threads),'OMP_DYNAMIC':'FALSE','OMP_WAIT_POLICY':'PASSIVE',
                 'BRAIDS_PROFILE':'1','BRAIDS_PROGRESS_STEPS':'2000'})
    result=dict(return_code=p.returncode,wall_s=time.perf_counter()-start,threads=threads,
        executable_sha256=sha256(exe),source_sha256=build['source_sha256'])
    write_json(folder/'execution.json',result)
    if p.returncode: raise RuntimeError(f'BRAIDS failed: {folder}')
    return result


def run(tag,engines,**options):
    report=HERE/'reports'/tag;report.mkdir(parents=True,exist_ok=False)
    threads=options.pop('threads'); seed_tag=options.pop('seed_tag')
    write_json(report/'plan.json',dict(engines=engines,threads=threads,seed_tag=seed_tag,**options,
        stage='first transient comparison, not equilibrium validation',feed_kg_s=FEED,
        engine_source_modified=False))
    def one(engine):
        folder=HERE/'runs'/f'{tag}_{engine}'
        seed=HERE/'runs'/f'{seed_tag}_{engine}' if seed_tag else None
        fn=prepare_native if engine=='braids' else prepare_delft
        write_json(report/f'{engine}_status.json',dict(status='preparing',folder=str(folder)))
        fn(folder,seed=seed,**options)
        write_json(report/f'{engine}_status.json',dict(status='running',folder=str(folder)))
        print(engine, 'running',folder,flush=True)
        try:
            result=execute_native(folder,threads) if engine=='braids' else shared.delft_runtime.run_case(folder,timeout_s=7200)
            write_json(report/f'{engine}_status.json',dict(status='complete',execution=result))
        except Exception as exc:
            write_json(report/f'{engine}_status.json',dict(status='failed',error=str(exc)))
            raise
        print(engine,'completed',result.get('wall_s',result.get('elapsed_wall_s')),flush=True)
    with ThreadPoolExecutor(max_workers=len(engines)) as pool:
        list(pool.map(one,engines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('tag');p.add_argument('--engines',nargs='+',choices=['braids','delft'],default=['braids','delft'])
    p.add_argument('--dx',type=float,default=.2125);p.add_argument('--end',type=float,default=300)
    p.add_argument('--output',type=float,default=30);p.add_argument('--dt',type=float,default=.025)
    p.add_argument('--morfac',type=float,default=1);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--hydraulic',action='store_true');p.add_argument('--seed-tag')
    a=vars(p.parse_args());run(**a)
