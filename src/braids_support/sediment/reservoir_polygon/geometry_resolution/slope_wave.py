"""Exact smooth simple wave on a sloping bed, before characteristic crossing."""
from pathlib import Path
import json,subprocess,os,sys,hashlib
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT));HERE=Path(__file__).resolve().parent
import numpy as np
from numpy.polynomial.legendre import leggauss
from braids_core.boundaries.domain import PolygonDomain,BoundaryAssignment
from braids_core.boundaries.inlet_io import NativeGrid
from braids_core.boundaries.cut_cells import map_cut_cells
from braids_core.boundaries.cut_cell_io import write_cut_case,CutBoundary,CutHydraulicBoundarySeries
from braids_qgis.core.results import read_frames
G=9.81;S=.3;H=1.;U=5.;EPS=.05;CENTER=300.;SIGMA=40.;END=5.

def exact(points,t):
    # x = xi + [U + 3 sqrt(g h_i(xi)) - 2 sqrt(g H)] t + g S t²/2.
    xi=points-(U+np.sqrt(G*H))*t-.5*G*S*t*t
    for _ in range(20):
        e=EPS*np.exp(-.5*((xi-CENTER)/SIGMA)**2);h=H+e;c=np.sqrt(G*h)
        f=xi+(U+3*c-2*np.sqrt(G*H))*t+.5*G*S*t*t-points
        derivative=1+1.5*G/c*(-(xi-CENTER)/SIGMA**2*e)*t
        assert derivative.min()>.9
        xi-=f/derivative
    h=H+EPS*np.exp(-.5*((xi-CENTER)/SIGMA)**2)
    u=U+G*S*t+2*(np.sqrt(G*h)-np.sqrt(G*H))
    return h,h*u

def means(centers,dx,t):
    nodes,w=leggauss(32);points=centers[:,None]+dx/2*nodes;h,hu=exact(points,t)
    return h@w/2,hu@w/2

def main():
    p=HERE/'steep_wet_balance';build=json.loads((ROOT/'solver/cut_cells/build/build.json').read_text());results={}
    full=(p/'before/cutcell_solver.f90').read_text();program='\nprogram run_cutcell'+full.split('\nprogram run_cutcell')[1]
    for label in ['current','fully_wet']:
        b=p/label;source=(b/'module.f90').read_text()+program;cached=(b/'solver.f90').exists() and (b/'solver.f90').read_text()==source and (b/'xwave').exists();(b/'solver.f90').write_text(source);cmd=build['command'][:];cmd[cmd.index(str(ROOT/'solver/cut_cells/cutcell_solver.f90'))]=str(b/'solver.f90');cmd[-1]=str(b/'xwave');
        if not cached:
            r=subprocess.run(cmd,cwd=b,capture_output=True,text=True,timeout=60);(b/'wave_build.log').write_text(r.stdout+r.stderr);r.check_returncode()
        rows=[]
        for dx in [20.,10.,5.]:
            work=b/f'wave_extended_dx{dx:g}';work.mkdir(exist_ok=True);nx=round(2000/dx);domain=PolygonDomain('plane',[(-800,0),(1200,0),(1200,1),(-800,1)],'EPSG:2056');assign=[BoundaryAssignment(domain.edges[3].edge_id,'inlet','inlet'),BoundaryAssignment(domain.edges[1].edge_id,'outlet','outlet')];mesh=map_cut_cells(domain,assign,NativeGrid(nx,1,-800,0,dx,1.));x=np.array([c.centroid_xy[0] for c in mesh.cells]);bed=-S*x;q=np.zeros((len(x),5));q[:,0],q[:,1]=means(x,dx,0)
            bc={'inlet':CutBoundary('prescribed_depth_trace',hydraulic_series=CutHydraulicBoundarySeries((0.,END),(U,U+G*S*END),(H,H))),'outlet':CutBoundary('supercritical')}
            write_cut_case(work/'case.data',mesh,q,bed,bc,end_time_s=END,dt_max_s=.005,output_interval_s=END,scheme=6,bedload=0,exchange=0,settling_velocity_m_s=0,manning=0,source_sha256={'solver':hashlib.sha256(source.encode()).hexdigest()})
            r=subprocess.run([str(b/'xwave'),'case.data'],cwd=work,capture_output=True,text=True,timeout=30,env={**os.environ,'OMP_NUM_THREADS':'1','OMP_WAIT_POLICY':'PASSIVE'});(work/'native.log').write_text(r.stdout+r.stderr);r.check_returncode();f=read_frames(work/'frames.txt')[-1];ref,_=means(x,dx,END);core=(x>=150)&(x<=650);delta=f['q'][:,0]-ref
            row={'dx_m':dx,'depth_pulse_relative_L1':float(np.sum(abs(delta[core]))/np.sum(abs(ref[core]-H))),'peak_loss_relative':float(1-(f['q'][core,0].max()-H)/(ref[core].max()-H)),'minimum_depth':float(f['q'][:,0].min()),'water_budget':float(dx*np.sum(f['q'][:,0]-q[:,0])-f['ledger'][:,0].sum())};rows.append(row);np.savez(work/'reference.npz',x=x,depth=ref,initial=q[:,0]);print(label,row,flush=True)
        results[label]=rows
    (p/'wave_extended_verification.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
