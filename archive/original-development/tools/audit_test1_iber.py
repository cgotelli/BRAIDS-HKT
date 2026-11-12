"""Read-only Test1/Iber mesh and timestep audit; outputs are private evidence.

Never changes either solver, either user project, or historical runs.
"""
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from braids_core.sediment.adaptive_river_worker import hierarchy
from braids_core.boundaries.agglomeration import agglomerate_small_cells
from braids_core.boundaries.amr import build_hierarchy
from braids_core.boundaries.domain import PolygonDomain, BoundaryAssignment
from braids_core.boundaries.inlet_io import NativeGrid
from braids_core.sediment.river_leaf_controller import RiverLeafController
from braids_gpu.atomic_output import write_json_atomic

IBER = Path('C:/Users/STREEM/Desktop/DEM-project-CIVIL410/test.gid')
BASE = ROOT/'validation/_test1_amr_interval/cpu_installed_1s'
OUT = ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/iber_inlet_audit114'

def read(path): return json.loads(path.read_text())

def statistics(polys, ids=None):
    selected = polys if ids is None else [polys[i] for i in ids]
    areas = np.array([p.area for p in selected])
    lengths = np.array([p.area/p.length for p in selected])
    return dict(cells=len(selected), area_m2=float(areas.sum()),
        minimum_area_m2=float(areas.min()), median_area_m2=float(np.median(areas)),
        minimum_sqrt_area_m=float(np.sqrt(areas.min())),
        minimum_area_over_perimeter_m=float(lengths.min()),
        under_1_m2=int((areas<1).sum()), under_12p5_m2=int((areas<12.5).sum()),
        under_25_m2=int((areas<25-1e-8).sum()))

def iber():
    lines=(IBER/'Iber2D.dat').read_text().splitlines()
    n=int(lines[3]);cells=[s.split() for s in lines[4:4+n]]
    assert lines[4+n]=='VERTEX'
    nv=int(lines[5+n]);vertices={int(s.split()[3]):list(map(float,s.split()[:3])) for s in lines[6+n:6+n+nv]}
    origin=np.array(vertices[1][:2])
    polygons=[Polygon(np.array([vertices[int(i)][:2] for i in c[:3]])-origin) for c in cells]
    boundary=lines[lines.index('CC: CONDICIONS CONTORN')+1:]
    inlet=sorted({int(s.split()[0])-1 for s in boundary if 'CriticoLento' in s})
    series=[]
    for line in (IBER/'proceso.rep').read_text().splitlines():
        m=re.match(r'\s*([\d.]+)\s+([\d.]+)\s+(\d+):(\d+):(\d+):(\d+)\s+([\d.]+)\s+([\d.]+)',line)
        if m:
            t,dt,h,mi,s,cs,qi,qo=map(float,m.groups())
            series.append(dict(time_s=t,dt_s=dt,clock_s=h*3600+mi*60+s+cs*.01,qin=qi,qout=qo))
    return polygons,origin,inlet,series,dict(mesh=statistics(polygons),inlet=statistics(polygons,inlet),
        nodes=nv,manning=sorted({float(c[4]) for c in cells}),
        generated_header=lines[1],
        source_sha256={name:hashlib.sha256((IBER/name).read_bytes()).hexdigest() for name in
            ('Iber2D.dat','Iber_Problemdata.dat','Iber_CPUSolver.dat','test.prb','proceso.rep')})

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ip,origin,inlet,iseries,im=iber()
    p=read(BASE/'prepared/core/prepared/profile.json');tree=hierarchy(p)
    mesh=tree.finest;bp=[Polygon(np.array(c.vertices_xy)-origin) for c in mesh.cells]
    bif=[f for f in mesh.faces if f.kind=='inlet'];bi=sorted({f.owner for f in bif})
    inletline=unary_union([LineString(np.array([f.start_xy,f.end_xy])-origin) for f in bif])
    merged,_,ma=agglomerate_small_cells(mesh,.5,preserve_parent_boundaries=True)
    mp=[Polygon(np.array(c.vertices_xy)-origin) for c in merged.cells]
    mi=sorted({f.owner for f in merged.faces if f.kind=='inlet'})
    mfree,_,fa=agglomerate_small_cells(mesh,.5,preserve_parent_boundaries=False)
    fp=[Polygon(np.array(c.vertices_xy)-origin) for c in mfree.cells]
    def exterior_signature(m):
        return [(f.start_xy,f.end_xy,f.outward_normal,f.length_m,f.edge_id,f.bc_id,f.kind)
                for f in m.faces if f.neighbour<0]
    assert exterior_signature(mesh)==exterior_signature(merged)==exterior_signature(mfree)
    assert unary_union(bp).symmetric_difference(unary_union(mp)).area<1e-7
    assert unary_union(bp).symmetric_difference(unary_union(fp)).area<1e-7
    tree_result={}
    try:
        mt=build_hierarchy(PolygonDomain(**p['domain']),[BoundaryAssignment(**a) for a in p['assignments']],
            NativeGrid(**p['grid']),levels=p['levels'],minimum_area_fraction=.5)
        leaves=RiverLeafController.pin_boundaries(mt,mt.roots)
        tree_result=dict(status='passed',initial_leaf_count=len(leaves))
    except Exception as exc:tree_result=dict(status='failed',error=str(exc))
    events=read(BASE/'run/ADAPTATION.json');summary=read(BASE/'PROFILE_SUMMARY.json');failures=[]
    for row in summary['singles']:
        if not row['error']:continue
        match=re.search(r'cell=(\d+)',row['error']);r=dict(row)
        if match:
            event=max((e for e in events if e['time_s']<=row['time']+1e-9),key=lambda e:e['time_s'])
            node=event['leaves'][int(match[1])];c=tree.cells[node];poly=Polygon(np.array(c.vertices_xy)-origin)
            fine_id=node-tree.offsets[-2] if tree.node_level[node]==tree.finest_level else None
            r.update(node=node,area_m2=c.area_m2,centroid=list(c.centroid_xy),
                distance_to_inlet_m=poly.distance(inletline),is_inlet=fine_id in bi)
        failures.append(r)
    bpoly=unary_union(bp);ipoly=unary_union(ip)
    data=dict(iber=im,braids=dict(mesh=statistics(bp),inlet=statistics(bp,bi),
        current_runtime=summary['runtime_identity'],worker_s=summary['worker_call_s'],
        phases=summary['phases'],hydraulic=summary['result']['hydraulic']),
        domains=dict(iber_area_m2=ipoly.area,braids_area_m2=bpoly.area,
            symmetric_difference_m2=ipoly.symmetric_difference(bpoly).area),
        agglomeration_nested=dict(mesh=statistics(mp),inlet=statistics(mp,mi),audit=ma,public_hierarchy=tree_result),
        agglomeration_unconstrained=dict(mesh=statistics(fp),audit=fa),
        failures=failures,iber_log=iseries,
        clock_note='Last timestamp field interpreted as hundredths; final report independently states 22 seconds.')
    template=Path('C:/Program Files/Iber/Iber 3.4/problemtypes/IBER.gid/IBER.bas')
    im['exporter_sha256']=hashlib.sha256(template.read_bytes()).hexdigest()
    header=im['generated_header'].split()
    im['generated_settings']=dict(dt_max_s=float(header[0]),tvd=int(header[3]),cfl=float(header[4]),
        dry_depth_m=float(header[5]),scheme='second order (tvd=1)',threads=1,
        inlet='Total discharge Critical/Subcritical (-41, depth not supplied)',
        outlet='Supercritical/Critical (-40)',wall_friction_manning=.015,initial_volume_reported_m3=.084710)
    def frames(folder):
        result=[]
        for row in read(folder/'INDEX.json')['frames']:
            with np.load(folder/row['epoch']) as f:leaves=f['leaves']
            with np.load(folder/row['file']) as f:q=f['q']
            cover=tree.leaf_cover(leaves)[tree.offsets[-2]:];ids=np.searchsorted(leaves,cover)
            assert np.array_equal(leaves[ids],cover)
            result.append((row['time_s'],q[:,ids]))
        return result
    reference=frames(BASE/'run/frames');area=np.array([c.area_m2 for c in mesh.cells])
    control=ROOT/'validation/_test1_iber_audit/cpu_dt04'
    if (control/'PROFILE_SUMMARY.json').exists():
        cs=read(control/'PROFILE_SUMMARY.json');diff=[]
        for (t,q),(u,r) in zip(reference,frames(control/'run/frames'),strict=True):
            assert abs(t-u)<1e-10
            wet=np.maximum(q[0],r[0])>1e-6;delta=r[0]-q[0]
            diff.append(dict(time_s=t,max_depth_difference_mm=float(abs(delta).max()*1000),
                wet_area_rms_mm=float(np.sqrt(np.average(delta[wet]**2,weights=area[wet]))*1000) if wet.any() else 0.))
        data['dt04_control']=dict(runtime=cs['runtime_identity'],worker_s=cs['worker_call_s'],
            preparation_s=cs['preparation_s'],phases=cs['phases'],hydraulic=cs['result']['hydraulic'],
            frame_differences=diff,water_residual_max_m3=max(abs(r['water_residual_m3']) for r in read(control/'run/DIAGNOSTICS.json')))
    smallest=int(np.argmin(area));data['braids']['smallest_cell']=dict(finest_id=smallest,
        centroid=list(mesh.cells[smallest].centroid_xy),
        maximum_saved_depth_first8s_m=max(float(q[0,smallest]) for _,q in reference))
    write_json_atomic(OUT/'AUDIT.json',data)
    center=np.mean([np.array(c.centroid_xy)-origin for c in mesh.cells if c.cell_id in bi],axis=0)
    fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
    for ax,polys,title in zip(axes,(ip,bp,mp),('Iber: boundary-fitted triangles','BRAIDS: current cropped squares','Existing merge: parent-constrained')):
        coll=PolyCollection([np.asarray(g.exterior.coords) for g in polys],array=np.array([g.area for g in polys]),
            cmap='viridis',norm=LogNorm(.01,50),edgecolors='#555555',linewidths=.35)
        ax.add_collection(coll);ax.set_xlim(center[0]-35,center[0]+35);ax.set_ylim(center[1]-25,center[1]+45)
        ax.set_aspect('equal');ax.set_title(title,fontsize=10);ax.set_xlabel('Local east (m)');ax.set_ylabel('Local north (m)')
        for f in bif:
            xy=np.array([f.start_xy,f.end_xy])-origin;ax.plot(*xy.T,color='red',lw=2)
    for r in failures:
        if 'centroid' in r:axes[1].plot(*(np.array(r['centroid'])-origin),marker='x',color='red',ms=7)
    fig.colorbar(coll,ax=axes,label='Cell area (m²)',shrink=.8)
    fig.suptitle('Inlet geometry: red line = BRAIDS inflow; crosses = largest reported nonlinear residuals')
    fig.savefig(OUT/'INLET_MESH.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    acc=[s for s in summary['singles'] if s['error'] is None]
    axes[0].plot([s['time'] for s in acc],[s['dt'] for s in acc],'.',ms=3,label='BRAIDS accepted internal steps')
    early=[s for s in iseries if 0<s['time_s']<=8.2]
    axes[0].plot([s['time_s'] for s in early],[s['dt_s'] for s in early],'o-',label='Iber reported step')
    axes[0].set_yscale('log');axes[0].set_xlabel('Simulated time (s)');axes[0].set_ylabel('Timestep (s)');axes[0].legend(fontsize=8)
    axes[1].bar(['Accepted trials','Rejected trials'],[sum(s['evaluations'] for s in summary['singles'] if (s['error'] is None)==v) for v in (True,False)])
    axes[1].set_ylabel('BRAIDS f-wave residual evaluations');axes[1].set_title('Work over the first 8 simulated seconds')
    fig.savefig(OUT/'TIMESTEP_COST.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(iber=im['mesh'],braids=data['braids']['mesh'],
        nested_merge=tree_result,dt04=data.get('dt04_control'),
        located_failures=sum('centroid' in r for r in failures),
        inlet_failures=sum(r.get('is_inlet',False) for r in failures)),indent=2))

if __name__=='__main__':main()
