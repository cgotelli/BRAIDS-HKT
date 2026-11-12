"""Rotated slip walls on the fixed Cartesian mesh; no production edits."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from check_bedload161 import sha,save
from braids_core.boundaries.domain import PolygonDomain
from braids_core.boundaries.inlet_io import NativeGrid
from braids_core.boundaries.cut_cells import map_cut_cells
from braids_core.boundaries.cut_cell_io import write_cut_case,CutSediment
from braids_core.sediment.portable_river_runtime import install
from braids_core.sediment import river_mesh_records,bend_native_hydraulics as native,wall_bed_hydraulics as wall
from corner_candidate167 import build_native,cuda_source


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['baseline','tangent'],required=True)
    ap.add_argument('--tag',default='v3')
    args=ap.parse_args();out=ROOT/f'validation/_wall169/{args.tag}/{args.variant}';out.mkdir(parents=True,exist_ok=False)
    install(json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text()),ROOT/'desktop/runtime')
    table=json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text())
    row=next(v for k,v in table.items() if 'shoreline_hydraulic_probe.build' in k and
        'call braids_wall_velocity_bounds' in (ROOT/'desktop/runtime'/Path(v['native']['file']).parent/'shoreline_probe.f90').read_text())
    selected=(ROOT/'desktop/runtime'/row['native']['file'],row['identity'],row['identity']['command'][0])
    if args.variant=='tangent':selected=build_native('tangent')
    native.build=lambda:selected
    from validation.gpu_fwave.reconstruction import ResidentReconstruction,OPTIONS
    import cupy as cp
    base=ROOT/'braids_gpu/frozen_sources/graded_bedload_v8/reconstruction.cu'
    code=wall.cuda_source(base.read_text())
    if args.variant=='tangent':code=cuda_source(code,'tangent')
    source=out/'reconstruction.cu';source.write_text(code)
    rows=[]
    for degrees in (0,17,37,63,90):
        theta=np.deg2rad(degrees);t=np.array([np.cos(theta),np.sin(theta)]);n=np.array([-t[1],t[0]])
        origin=np.array([6.13,5.17]);corners=np.array([[-3,0],[3,0],[3,3],[-3,3]])
        vertices=origin+corners[:,0,None]*t+corners[:,1,None]*n
        folder=out/f'angle{degrees}';folder.mkdir()
        domain=PolygonDomain('wall',vertices,'EPSG:32632')
        mesh=map_cut_cells(domain,[],NativeGrid(24,24,0,0,.5,.5))
        initial=np.zeros((len(mesh.cells),5));initial[:,0]=.5
        write_cut_case(folder/'case.data',mesh,initial,np.zeros(len(mesh.cells)),{},
            end_time_s=1.,dt_max_s=.01,output_interval_s=1.,cfl=.4,scheme=6,
            dry_tolerance_m=1e-6,bedload=0,exchange=0,manning=.03,
            sediment=CutSediment(d50_m=.001,d90_m=.0015,minimum_bedload_depth_m=.001,roughness_mode='manning'),
            compact_sidecars=True,source_sha256={'validation':sha(__file__)})
        d,identity=river_mesh_records.initialize(folder/'case.data',threads=4)
        save(folder/'RECORDS.json',identity)
        cpu=native.NativeHydraulics(folder/'case.data',d,4)
        gpu=ResidentReconstruction(d,source_path=source)
        xy=d['xy'].T;mid=d['mid'].T;r=(xy-origin)@n
        wall_faces=(d['neighbour']<0)&(np.linalg.norm(d['normal'].T+n,axis=1)<1e-9)&(abs((mid-origin)@t)<1.5)
        assert wall_faces.sum()>0
        cells=np.unique(d['owner'][wall_faces])
        local_faces=np.isin(d['owner'],cells)|np.isin(d['neighbour'],cells)
        for name,a,b in [('outward',.2,.3),('inward',.2,-.3),('tangent',.2,0.),('uniform',0.,0.)]:
            uv=(1+a*r)[:,None]*t+(b*r)[:,None]*n
            q=np.ascontiguousarray(np.vstack([np.full(len(xy),.5),.5*uv.T]));z=np.zeros(len(xy))
            grad=np.zeros((4,2,len(xy)));face=np.zeros((2,4,d['nf']));flags=np.zeros((3,2,d['nf']),np.int32)
            f=cpu.lib.bend_wall_traces;f.argtypes=[ctypes.c_void_p]*5;f.restype=None
            f(*(v.ctypes.data for v in (q,z,grad,face,flags)))
            got=gpu.reconstruct(cp.asarray(q));gg=cp.asnumpy(got['grad']);gf=cp.asnumpy(got['face_value']).transpose(1,0,2)
            exact=(1+a*((mid-origin)@n))[:,None]*t+(b*((mid-origin)@n))[:,None]*n
            wuv=face[0,1:3][:,wall_faces].T
            error=wuv-exact[wall_faces]
            # The raw LS field and final scalar/component limiter bounds are
            # retained to distinguish geometric fitting from limiter changes.
            np.savez(folder/(name+'.npz'),q=q,z=z,xy=xy,mid=mid,normal=d['normal'].T,
                     owner=d['owner'],neighbour=d['neighbour'],area=d['area'],length=d['length'],
                     grad=grad,gpu_grad=gg,face=face,gpu_face=gf,wall_faces=wall_faces,cells=cells,
                     vlo=cp.asnumpy(got['vlo']),vhi=cp.asnumpy(got['vhi']),
                     expected=exact,t=t,n=n,origin=origin)
            rows.append(dict(angle=degrees,case=name,cells=len(xy),wall_faces=int(wall_faces.sum()),
                minimum_area_m2=float(d['area'].min()),
                tangent_error=float(abs(error@t).max()),normal_error=float(abs(error@n).max()),
                face_parity=float(abs(face-gf).max()),gradient_parity=float(abs(grad-gg).max()),
                selected_face_error=float(abs(face[0,1:3][:,wall_faces]-gf[0,1:3][:,wall_faces]).max())))
        rest=np.vstack([np.full(len(xy),.5),np.zeros((2,len(xy)))])
        rate,water=cpu.rates(rest,np.zeros(len(xy)),1e-8,0.)
        save(folder/'REST.json',dict(rate_max=float(abs(rate).max()),wall_water_max=float(abs(water).max())))
    save(out/'AUDIT.json',dict(variant=args.variant,rows=rows,script_sha256=sha(__file__),
         native_library_sha256=sha(selected[0]),cuda_source_sha256=sha(source),
         production_changed=False,Delft3D_modified=False))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
