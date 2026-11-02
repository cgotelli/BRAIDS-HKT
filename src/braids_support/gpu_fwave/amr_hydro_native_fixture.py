"""Small unchanged-native two-level scheme-6 wet/dry AMR reference.

This is a preparation/evolution reference for the first GPU AMR integration
experiment. It does not claim a GPU trajectory or physical braids_support.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from braids_core.boundaries.amr import AMRSettings, build_hierarchy
from braids_core.boundaries.cut_cell_io import CutSedimentFraction, write_cut_case
from braids_core.boundaries.domain import PolygonDomain
from braids_core.boundaries.inlet_io import NativeGrid
from braids_qgis.core.results import read_frames
from . import coupled_reference as exporter
from . import cpu_filter_cold20h as frozen_cpu


HERE=Path(__file__).resolve().parent
TAG='amr_hydro_twolevel_native_dev_20261004_v4'
ROOT=HERE/'runs'/TAG
BULK=2650.*.6
END=.08


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def geometry():
    domain=PolygonDomain('GPU two-level wet/dry prototype',
                         [(0.,0.),(16.,0.),(16.,4.),(0.,4.)],'EPSG:2056')
    settings=AMRSettings(enabled=True,levels=2,regrid_interval_s=.02,
        buffer_cells=2,refine_wet_dry=True,refine_walls=False,
        refine_bathymetry=False,refine_hydraulics=False,refine_sediment=False,
        subcycling=True)
    tree=build_hierarchy(domain,[],NativeGrid(8,2,0.,0.,2.,2.),levels=2)
    xy=np.asarray([cell.centroid_xy for cell in tree.finest.cells])
    state=np.zeros((len(xy),9),dtype=np.float64)
    state[:,0]=np.where(xy[:,0]<4.,.5,0.)
    state[:,5:7]=.025*BULK;state[:,7:9]=.025*BULK
    rock=np.full(len(xy),-.1)
    return tree,settings,state,rock


def fraction():
    return tuple(CutSedimentFraction(fraction_id=f'inert_sand_{g}',d50_m=diameter,
        d90_m=1.5*diameter,grain_density_kg_m3=2650.,porosity=.4,
        settling_velocity_m_s=0.,z0_m=diameter/30.,
        minimum_bedload_depth_m=.001,minimum_exchange_depth_m=.001,
        roughness_mode='manning',initial_volume_fraction=.5,
        initial_active_volume_fraction=.5)
        for g,diameter in enumerate((.0002,.0003)))


def prepare(output=ROOT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('AMR native fixture is write-once')
    _,sources=frozen_cpu.runtime_identity()
    if sha(frozen_cpu.TIGHT_EXE)!=frozen_cpu.TIGHT_SHA:
        raise ValueError('Frozen CPU executable changed')
    tree,settings,state,rock=geometry()
    folder=output/'input';folder.mkdir(parents=True)
    deck=folder/'case.data'
    write_cut_case(deck,tree.finest,state,rock,{},
        end_time_s=END,dt_max_s=.002,output_interval_s=.02,cfl=.4,
        scheme=6,bedload=0,exchange=0,manning=.03,dry_tolerance_m=.0001,
        sediment_fractions=fraction(),active_layer_thickness_m=.05,
        substrate_layer_count=1,substrate_layer_thickness_m=.05,
        mpm_friction_mode=0,secondary_flow=False,morphological_factor=1.,
        amr_hierarchy=tree,amr_settings=settings,compact_sidecars=True,
        source_sha256={'fixture':sha(__file__)})
    tree.finest.write_geometry(folder/'geometry.npz')
    np.savez_compressed(folder/'initial_analysis.npz',state=state,rock=rock,
                        parent=tree.parent,node_level=tree.node_level,
                        offsets=np.asarray(tree.offsets),xy=np.asarray([c.centroid_xy for c in tree.cells]),
                        area=np.asarray([c.area_m2 for c in tree.cells]))
    plan=dict(schema=1,tag=TAG,deck=str(deck),
        source_sha256={**sources,str(Path(__file__).resolve()):sha(__file__)},
        input_sha256={str(p):sha(p) for p in folder.iterdir() if p.is_file()},
        native_executable=str(frozen_cpu.TIGHT_EXE),
        native_executable_sha256=frozen_cpu.TIGHT_SHA,
        settings=dict(levels=2,ratio=2,subcycling=True,
                      regrid_interval_s=.02,scheme=6,bedlaw=0),
        Delft3D_source_modified=False,GPU_steps_performed=0)
    save(output/'plan.json',plan)
    return plan


def run_native(output=ROOT):
    output=Path(output).resolve()
    plan=json.loads((output/'plan.json').read_text())
    for name,digest in {**plan['source_sha256'],**plan['input_sha256'],
                        plan['native_executable']:plan['native_executable_sha256']}.items():
        if sha(name)!=digest:raise ValueError('Frozen AMR native input/runtime differs: '+name)
    deck=Path(plan['deck']);folder=output/'cpu'
    origin=exporter.copy_inputs(deck,folder)
    env={key:value for key,value in os.environ.items() if not key.upper().startswith(('BRAIDS_','OMP_'))}
    env.update(OMP_NUM_THREADS='8',OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',
               BRAIDS_TIME_INTEGRATOR='auto')
    with (folder/'native.log').open('x',encoding='utf-8') as log:
        process=subprocess.run([plan['native_executable'],deck.name],cwd=folder,
            env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    record=dict(status='completed' if process.returncode==0 else 'failed',
                return_code=process.returncode,plan_sha256=sha(output/'plan.json'),
                origin=origin,executable_sha256=sha(plan['native_executable']),
                output_sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file()},
                Delft3D_source_modified=False)
    save(folder/'execution.json',record)
    if process.returncode:
        raise RuntimeError('Frozen native AMR run failed; inspect native.log')
    frames=read_frames(folder/'frames.txt')
    tree,_,_,_=geometry()
    counts=[dict(time_s=float(f['time_s']),mesh_id=int(f['mesh_id']),
                 nodes=len(f['node_ids']),
                 coarse=int(np.count_nonzero(tree.node_level[f['node_ids']]==0)),
                 fine=int(np.count_nonzero(tree.node_level[f['node_ids']]==1)))
            for f in frames]
    return dict(frames=counts,output=str(folder),plan_sha256=record['plan_sha256'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run_native'))
    parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args()
    print(json.dumps(globals()[args.action](args.output),indent=2))


if __name__=='__main__':main()
