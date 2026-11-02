"""Independent fixed-bed CFL checks; not a universal CFL-default qualification."""
from __future__ import annotations
import json
import os
import subprocess
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss

from braids_core.boundaries.domain import PolygonDomain,BoundaryAssignment
from braids_core.boundaries.inlet_io import NativeGrid
from braids_core.boundaries.cut_cells import map_cut_cells
from braids_core.boundaries.cut_cell_io import write_cut_case,CutBoundary,CutHydraulicBoundarySeries
from . import hydraulic_performance as h


def fixture(kind):
    if kind=='steep_wave':
        from braids_support.sediment.reservoir_polygon.geometry_resolution import slope_wave as wave
        domain=PolygonDomain(kind,[(-800,0),(1200,0),(1200,1),(-800,1)],'EPSG:2056')
        assignments=[BoundaryAssignment(domain.edges[3].edge_id,'inlet','inlet'),
                     BoundaryAssignment(domain.edges[1].edge_id,'outlet','outlet')]
        mesh=map_cut_cells(domain,assignments,NativeGrid(200,1,-800,0,10,1))
        x=np.array([c.centroid_xy[0] for c in mesh.cells])
        q=np.zeros((len(x),5));q[:,0],q[:,1]=wave.means(x,10,0)
        bc=dict(inlet=CutBoundary('prescribed_depth_trace',hydraulic_series=CutHydraulicBoundarySeries(
            (0.,wave.END),(wave.U,wave.U+wave.G*wave.S*wave.END),(wave.H,wave.H))),
            outlet=CutBoundary('supercritical'))
        ref,_=wave.means(x,10,wave.END);core=(x>=150)&(x<=650)
        return mesh,q,-wave.S*x,bc,wave.END,ref,core,float(np.sum(abs(ref[core]-wave.H)))
    if kind=='dry_dam_break':
        domain=PolygonDomain(kind,[(-10,0),(10,0),(10,.2),(-10,.2)],'EPSG:2056')
        mesh=map_cut_cells(domain,[],NativeGrid(400,2,-10,0,.05,.1))
        x=np.array([c.centroid_xy[0] for c in mesh.cells])
        q=np.zeros((len(x),5));q[:,0]=np.where(x<0,1.,0.)
        nodes,w=leggauss(32);xx=x[:,None]+.025*nodes;c=np.sqrt(9.81)
        ref=np.where(xx<-c,1.,np.where(xx<2*c,(2*c-xx)**2/(9*9.81),0.))@w/2
        return mesh,q,np.zeros(len(x)),{},1.,ref,np.ones(len(x),bool),float(ref.sum())
    if kind=='cutcell_lake':
        domain=PolygonDomain(kind,[(.15,.3),(8.7,.1),(8.4,5.8),(.2,5.3)],'EPSG:2056')
        mesh=map_cut_cells(domain,[],NativeGrid(18,12,0,0,.5,.5))
        xy=np.array([c.centroid_xy for c in mesh.cells]);bed=.12*xy[:,0]+.03*xy[:,1]
        q=np.zeros((len(bed),5));q[:,0]=np.maximum(.55-bed,0.)
        return mesh,q,bed,{},2.,q[:,0].copy(),np.ones(len(bed),bool),float(q[:,0].sum())
    raise ValueError(kind)


def run(tag=h.TAG):
    root=h.r.HERE/'runs'/tag;report=h.r.HERE/'reports'/tag
    exe,identity=h.runtime(root,'equivalent')
    rows=[]
    for kind in ('steep_wave','dry_dam_break','cutcell_lake'):
        mesh,q,bed,bc,end,ref,core,scale=fixture(kind)
        area=np.array([c.area_m2 for c in mesh.cells]);reference=None
        for cfl in (.2,.4,.8):
            folder=root/f'qualify_{kind}_cfl{cfl:g}';folder.mkdir(exist_ok=False)
            write_cut_case(folder/'case.data',mesh,q,bed,bc,end_time_s=end,dt_max_s=1.,
                output_interval_s=end,cfl=cfl,scheme=6,bedload=0,exchange=0,
                settling_velocity_m_s=0,dry_tolerance_m=.0001,manning=0,
                source_sha256=identity['source_sha256'])
            result=subprocess.run([str(exe),'case.data'],cwd=folder,capture_output=True,text=True,
                env=dict(os.environ,OMP_NUM_THREADS='4',OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',BRAIDS_PROFILE='1'),timeout=120)
            (folder/'native.log').write_text(result.stdout+result.stderr);result.check_returncode()
            final=h.r.shared.read_frames(folder/'frames.txt')[-1]
            state=final['q'];np.savez(folder/'reference.npz',initial=q,reference_depth=ref,core=core)
            error=float(np.sum(abs(state[core,0]-ref[core]))/max(scale,1e-30))
            if reference is None:reference=state.copy()
            difference=float(np.sum(abs(state[core,0]-reference[core,0]))/max(scale,1e-30))
            budget=float(np.dot(area,state[:,0]-q[:,0])-final['ledger'][:,0].sum())
            row=dict(case=kind,cfl=cfl,depth_relative_L1=error,vs_small_step_relative_L1=difference,
                     minimum_depth_m=float(state[:,0].min()),water_residual_m3=budget,
                     maximum_lake_momentum=float(abs(state[:,1:3]).max()) if kind=='cutcell_lake' else None,
                     profile=h.parse_profile(result.stdout),executable_sha256=identity['executable_sha256'],
                     case_sha256=h.r.sha256(folder/'case.data'),
                     passed=bool(error<.02 and difference<.01 and abs(budget)<1e-8 and
                                 np.isfinite(state).all() and state[:,0].min()>=0))
            if kind=='cutcell_lake':
                row['passed'] &= bool(abs(state-q).max()<1e-10)
            rows.append(row);h.r.write_json(report/'cfl_physics_checks.json',rows)
            print(json.dumps(row),flush=True)
    assert all(x['passed'] for x in rows)
    # Explicit baseline control for the existing outlet regression, not an
    # exemption inferred from a previous report or a weakened test assertion.
    sys.path.insert(0,str(h.r.ROOT/'tests'))
    from test_stage_outlet_wetting import run_stage
    a=run_stage(root/'known_outfall_baseline',stage=.02,executable=h.runtime(root,'baseline')[0])[-1]
    b=run_stage(root/'known_outfall_optimized',stage=.02,executable=exe)[-1]
    h.r.write_json(report/'known_outfall_regression.json',dict(
        state_bitwise_equal=bool(np.array_equal(a['q'],b['q'])),
        ledger_bitwise_equal=bool(np.array_equal(a['ledger'],b['ledger'])),
        flow_m3_s=float(-b['ledger'][0,0]/b['time_s'])))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default=h.TAG)
    run(parser.parse_args().tag)
