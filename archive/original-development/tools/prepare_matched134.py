"""Construct a fixed-mesh hydraulic cost control from the executed Iber deck.

The user's Iber files and Delft3D checkouts are read only. Outputs belong to
validation/_performance134/matched. This is input construction, not timing.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import gzip
from pathlib import Path
import sys

import numpy as np
from osgeo import gdal
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from braids_core.boundaries.domain import PolygonDomain, BoundaryAssignment
from braids_core.boundaries.inlet_io import NativeGrid
from braids_core.boundaries.axis_aligned import derive_axis_aligned
from braids_core.river_model import resolve_model
from braids_core.river_project import validate_project

IBER = Path('C:/Users/STREEM/Desktop/DEM-project-CIVIL410/test.gid')
OUT = ROOT/'validation/_performance134/matched'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def mesh():
    lines = (IBER/'Iber2D.dat').read_text().splitlines()
    nc = int(lines[3])
    cells = np.array([list(map(int, row.split()[:3])) for row in lines[4:4+nc]])-1
    nv = int(lines[5+nc])
    nodes = np.array([list(map(float, row.split()[:3])) for row in lines[6+nc:6+nc+nv]])
    bounds = {'inlet': set(), 'outlet': set()}
    for row in lines[lines.index('CC: CONDICIONS CONTORN')+1:]:
        kind = 'inlet' if 'CriticoLento' in row else 'outlet' if 'Critico/Rapido' in row else None
        if kind:
            c, s = [int(x)-1 for x in row.split()[:2]]
            bounds[kind].add(tuple(sorted((cells[c,s], cells[c,(s+1)%3]))))
    return nodes, cells, bounds


def inspect_dem(nodes):
    rows = []
    for name in ('DEM_2019_domain.tif','DEM_2024_domain.tif','flattened2019.tif',
                 'flattened2024.tif','flattened2024_corrected.tif'):
        p = IBER.parent/name
        ds = gdal.Open(str(p)); a = ds.ReadAsArray()
        gt = ds.GetGeoTransform(); inv = gdal.InvGeoTransform(gt)
        x = inv[0]+nodes[:,0]*inv[1]+nodes[:,1]*inv[2]-.5
        y = inv[3]+nodes[:,0]*inv[4]+nodes[:,1]*inv[5]-.5
        z = map_coordinates(a, [y,x], order=1, mode='nearest')
        valid = (x >= -.5)&(x <= ds.RasterXSize-.5)&(y >= -.5)&(y <= ds.RasterYSize-.5)
        no = ds.GetRasterBand(1).GetNoDataValue()
        if no is not None:
            valid &= np.abs(z-no) > .01
        valid &= np.isfinite(z)&(np.abs(z-nodes[:,2])<100)
        d = z[valid]-nodes[valid,2]
        near = map_coordinates(a, [y,x], order=0, mode='nearest')-nodes[:,2]
        rows.append(dict(path=str(p), sha256=sha(p), valid=int(valid.sum()),
                         nodes=len(nodes), rms_m=float(np.sqrt(np.mean(d*d))),
                         maximum_abs_m=float(np.max(np.abs(d))),
                         median_m=float(np.median(d)), geotransform=list(gt),
                         nearest_rms_m=float(np.sqrt(np.mean(near[valid]**2))),
                         nearest_maximum_abs_m=float(np.max(np.abs(near[valid])))))
    return rows


def domain_and_grid(nodes, cells, bounds, spatial):
    original = unary_union([Polygon(nodes[c,:2]) for c in cells])
    # Iber writes coordinates rounded to 1e-5 m, introducing micron-scale
    # zigzags on straight design edges. Preserve the six design corners and
    # explicitly report the (millimetre-square) simplification discrepancy.
    simple = original.simplify(1e-4, preserve_topology=True)
    vertices = list(simple.exterior.coords)[:-1]
    sections = {}
    for kind, edges in bounds.items():
        degree = {}
        for edge in edges:
            for i in edge:
                degree[i] = degree.get(i,0)+1
        endpoints = nodes[[i for i,n in degree.items() if n==1],:2]
        assert len(endpoints)==2
        sections[kind] = endpoints
    added = []
    ring = []
    for a,b in zip(vertices,vertices[1:]+vertices[:1]):
        line = LineString([a,b]); candidates = [(0.,a)]
        for kind, endpoints in sections.items():
            for endpoint in endpoints:
                if line.distance(Point(endpoint)) < 1e-4:
                    t = line.project(Point(endpoint)); projected = line.interpolate(t)
                    candidates.append((t,(projected.x,projected.y)))
                    added.append(dict(kind=kind,original=endpoint.tolist(),
                                      projected=[projected.x,projected.y],
                                      shift_m=float(np.linalg.norm(endpoint-[projected.x,projected.y]))))
        ring += [p for _,p in sorted(candidates)]
    domain = PolygonDomain('iber134_executed_domain',ring,spatial)
    assignments = []
    for kind,endpoints in sections.items():
        target = LineString(endpoints)
        matches = [e for e in domain.edges
                   if target.distance(Point(np.mean([e.start_xy,e.end_xy],axis=0))) < 1e-4
                   and abs(e.length_m-target.length)<1e-4]
        assert len(matches)==1, (kind,len(matches))
        assignments.append(BoundaryAssignment(matches[0].edge_id,{'inlet':'i','outlet':'o'}[kind],kind))
    spacing = 3.25
    low = np.floor(np.array(original.bounds[:2])/spacing)*spacing-spacing
    high = np.ceil(np.array(original.bounds[2:])/spacing)*spacing+spacing
    grid = NativeGrid(int(round((high[0]-low[0])/spacing)),int(round((high[1]-low[1])/spacing)),
                      *low,spacing,spacing)
    aligned, assignments, audit = derive_axis_aligned(domain,assignments,grid)
    geom = Polygon(aligned.vertices_xy)
    polygons = []
    for j in range(grid.ny):
        for i in range(grid.nx):
            cell = box(low[0]+i*spacing,low[1]+j*spacing,
                       low[0]+(i+1)*spacing,low[1]+(j+1)*spacing).intersection(geom)
            if cell.area > 1e-10:
                assert cell.geom_type=='Polygon'
                polygons.append(cell)
    areas = np.array([p.area for p in polygons])
    metrics = dict(cells=len(polygons),square_spacing_m=spacing,
                   area_m2=float(areas.sum()),median_cell_area_m2=float(np.median(areas)),
                   minimum_cell_area_m2=float(areas.min()),
                   median_2A_over_perimeter_m=float(np.median([2*p.area/p.length for p in polygons])),
                   original_iber_area_m2=original.area,
                   simplification_symmetric_difference_m2=original.symmetric_difference(simple).area,
                   original_to_aligned_symmetric_difference_m2=original.symmetric_difference(geom).area,
                   original_to_aligned_removed_m2=original.difference(geom).area,
                   original_to_aligned_added_m2=geom.difference(original).area,
                   open_endpoint_projection=added,
                   note='Static level-one Cartesian geometry; exact active-cell receipts come from public preparation.')
    return original,aligned,assignments,grid,audit,metrics,polygons


def rasterize_executed_bed(nodes, cells, grid, spatial):
    resolution = .25
    xlo = np.floor((grid.xlower_m-1)/resolution)*resolution
    yhi = np.ceil((grid.ylower_m+grid.ny*grid.dy_m+1)/resolution)*resolution
    nx = int(np.ceil((grid.xlower_m+grid.nx*grid.dx_m+1-xlo)/resolution))
    ny = int(np.ceil((yhi-grid.ylower_m+1)/resolution))
    x = xlo+(np.arange(nx)+.5)*resolution
    y = yhi-(np.arange(ny)+.5)*resolution
    # Translation avoids loss of precision in the triangulation search.
    origin = nodes[:,:2].min(0)
    xy = nodes[:,:2]-origin
    triangles = mtri.Triangulation(xy[:,0],xy[:,1],cells)
    interpolate = mtri.LinearTriInterpolator(triangles,nodes[:,2])
    xx,yy = np.meshgrid(x-origin[0],y-origin[1])
    z = interpolate(xx,yy)
    missing = np.ma.getmaskarray(z)
    # Only exterior pixels need extension. Project onto the actual exterior
    # edge and linearly interpolate its two saved node elevations; do not use
    # unrelated terrain or a tuned elevation offset in axis-added strips.
    edge_counts = {}
    for c in cells:
        for k in range(3):
            edge=tuple(sorted((c[k],c[(k+1)%3])))
            edge_counts[edge]=edge_counts.get(edge,0)+1
    boundary=np.array([e for e,n in edge_counts.items() if n==1])
    a=xy[boundary[:,0]];b=xy[boundary[:,1]];ab=b-a
    tree=cKDTree((a+b)/2)
    points=np.column_stack((xx[missing],yy[missing]))
    _,near=tree.query(points,k=8)
    delta=points[:,None,:]-a[near]
    t=np.clip(np.sum(delta*ab[near],axis=2)/np.sum(ab[near]**2,axis=2),0,1)
    distances=np.sum((delta-t[:,:,None]*ab[near])**2,axis=2)
    pick=np.argmin(distances,axis=1);row=np.arange(len(points))
    edge=near[row,pick];f=t[row,pick]
    extension=(1-f)*nodes[boundary[edge,0],2]+f*nodes[boundary[edge,1],2]
    values=np.asarray(z.filled(0),dtype=np.float64);values[missing]=extension
    path=OUT/'iber_executed_bed_025m.tif'
    ds=gdal.GetDriverByName('GTiff').Create(str(path),nx,ny,1,gdal.GDT_Float64,
              options=['TILED=YES','COMPRESS=LZW','PREDICTOR=3'])
    ds.SetGeoTransform((xlo,resolution,0,yhi,0,-resolution));ds.SetProjection(spatial)
    ds.GetRasterBand(1).WriteArray(values);ds.FlushCache();ds=None
    centroid=nodes[cells,:2].mean(1)
    ix=(centroid[:,0]-xlo)/resolution-.5;iy=(yhi-centroid[:,1])/resolution-.5
    sampled=map_coordinates(values,[iy,ix],order=1,mode='nearest')
    error=sampled-nodes[cells,2].mean(1)
    audit=dict(path=str(path),sha256=sha(path),pixel_size_m=resolution,
               shape=[ny,nx],dtype='float64',method='Executed Iber vertex elevations; linear interpolation on the actual Iber triangles',
               extension='Nearest original exterior segment, linear interpolation of its two vertex elevations',
               extrapolated_envelope_pixels=int(missing.sum()),
               centroid_rasterization_rms_m=float(np.sqrt(np.mean(error*error))),
               centroid_rasterization_maximum_abs_m=float(np.max(np.abs(error))),
               note='The original executed bed is reproduced before raster resampling. BRAIDS subsequently computes cell-average bed elevations; that discretization is not identical to Iber.')
    audit.update(raster_sampling_audit(path,nodes,cells))
    return path,audit


def raster_sampling_audit(path,nodes,cells):
    """Include triangle-edge/kink effects, not only the smooth centroids."""
    rng=np.random.default_rng(134)
    nc=len(cells); ids=np.repeat(np.arange(nc),12)
    w=rng.dirichlet(np.ones(3),size=len(ids))
    xyz=np.einsum('ij,ijk->ik',w,nodes[cells[ids]])
    ds=gdal.Open(str(path));gt=ds.GetGeoTransform();values=ds.ReadAsArray()
    ix=(xyz[:,0]-gt[0])/gt[1]-.5;iy=(xyz[:,1]-gt[3])/gt[5]-.5
    error=map_coordinates(values,[iy,ix],order=1,mode='nearest')-xyz[:,2]
    return dict(random_barycentric_samples=len(ids),random_barycentric_seed=134,
                sampled_rasterization_rms_m=float(np.sqrt(np.mean(error*error))),
                sampled_rasterization_p99_abs_m=float(np.quantile(abs(error),.99)),
                sampled_rasterization_maximum_abs_m=float(np.max(abs(error))))


def inspect_result_header(nodes,cells):
    """Read only the first small decompressed block, not the whole campaign."""
    path=IBER/'test.post.res'
    with gzip.open(path,'rb') as f:
        raw=f.read(43000)
    assert b'GiDPostEx1.2\x00' in raw[:32]
    marker=b'Values -1 Indexed\x00';offset=raw.index(marker)+len(marker)
    values=np.frombuffer(raw,dtype=[('id','<i4'),('z','<f4')],count=len(cells),offset=offset)
    assert np.array_equal(values['id'],np.arange(1,len(cells)+1))
    delta=values['z']-nodes[cells,2].mean(1)
    result=dict(path=str(path),size_bytes=path.stat().st_size,sha256=sha(path),
                container='gzip',format='GiDPostEx1.2 binary',location='OnGaussPoints GP_ELEMENT_1',
                indexing='sparse cell IDs, dry-element results omitted by saved Iber configuration',
                first_decompressed_bytes_inspected=len(raw),
                first_elevation_record='5189 little-endian int32 IDs and float32 elevations',
                saved_cell_bed_vs_vertex_mean_rms_m=float(np.sqrt(np.mean(delta*delta))),
                saved_cell_bed_vs_vertex_mean_maximum_abs_m=float(np.max(abs(delta))),
                variables_enabled=['Depth','Velocity','Specific_Discharge','Water_Elevation','Froude_Number',
                                   'Maximum_Depth','Maximum_Velocity','Maximum_Spec_Discharge','Maximum_Water_Elev'],
                output_interval_s=1.,note='No full field extraction performed. This confirms saved cell bed and output format, not equivalence of hydraulic trajectories.')
    (OUT/'IBER_RESULT_FORMAT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def build_inputs(base, domain, assignments, grid, dem):
    cases={}
    settings=[('cpu1_output1', 'cpu',1,1.,2.),
              ('cpu8_output1','cpu',8,1.,2.),('gpu_output1','gpu',8,1.,2.),
              ('cpu8_output10','cpu',8,10.,2.),('gpu_output10','gpu',8,10.,2.),
              ('cpu8_output60','cpu',8,60.,2.),('gpu_output60','gpu',8,60.,2.),
              ('cpu8_dt05','cpu',8,60.,.5),('cpu8_dt5','cpu',8,60.,5.),
              ('cpu8_clock05_output10','cpu',8,10.,.5),('cpu8_clock5_output10','cpu',8,10.,5.)]
    settings += [(('cpu8' if backend=='cpu' else 'gpu')+'_clock1_output'+str(interval),backend,8,float(interval),1.)
                 for backend in ('cpu','gpu') for interval in (1,10,60)]
    for name,backend,threads,interval,dtmax in settings:
        p=copy.deepcopy(base);p.pop('editor_state',None)
        p['name']='Iber executed bed: fixed 3.25 m '+name
        request=copy.deepcopy(p['model']['request']);request['execution']['backend']=backend
        p['model']=resolve_model(request).document()
        s=p['settings'];s['domain']=dict(domain_id=domain.domain_id,vertices_xy=domain.vertices_xy,spatial_reference=domain.crs_wkt)
        s['assignments']=[asdict(a) for a in assignments];s['grid']=asdict(grid);s['levels']=1
        s['adaptive']=False;s['adaptation']['terrain_enabled']=True
        s['end_time_s']=60.;s['dt_max_s']=dtmax;s['output_interval_s']=interval
        s['controls']['cfl']=1.;s['controls']['threads']=threads
        p['terrain']['dem_path']=str(dem)
        p=validate_project(p)
        target=OUT/(name+'.json');target.write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8')
        cases[name]=dict(path=str(target),sha256=sha(target),backend=backend,threads=threads,
                         dt_max_s=dtmax,output_interval_s=interval,end_time_s=60.,cfl=1.)
    return cases


def add_clock_variants():
    """Append I/O controls without touching already supplied decks or terrain."""
    audit_path=OUT/'MATCHED_INPUTS.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    for backend in ('cpu','gpu'):
        stem='cpu8' if backend=='cpu' else 'gpu'
        source=json.loads((OUT/(stem+'_output1.json')).read_text(encoding='utf-8'))
        for interval in (1,10,60):
            p=copy.deepcopy(source);name=stem+'_clock1_output'+str(interval)
            p['name']='Iber executed bed: fixed 3.25 m '+name
            p['settings']['dt_max_s']=1.;p['settings']['output_interval_s']=float(interval)
            p=validate_project(p);target=OUT/(name+'.json')
            text=json.dumps(p,indent=2)+'\n'
            if target.exists():
                assert target.read_text(encoding='utf-8')==text
            else:target.write_text(text,encoding='utf-8')
            audit['inputs'][name]=dict(path=str(target),sha256=sha(target),backend=backend,threads=8,
                         dt_max_s=1.,output_interval_s=float(interval),end_time_s=60.,cfl=1.)
    source=json.loads((OUT/'cpu8_output10.json').read_text(encoding='utf-8'))
    for name,dt in [('cpu8_clock05_output10',.5),('cpu8_clock5_output10',5.)]:
        p=copy.deepcopy(source);p['name']='Iber executed bed: fixed 3.25 m '+name
        p['settings']['dt_max_s']=dt;p=validate_project(p);target=OUT/(name+'.json')
        text=json.dumps(p,indent=2)+'\n'
        if target.exists():assert target.read_text(encoding='utf-8')==text
        else:target.write_text(text,encoding='utf-8')
        audit['inputs'][name]=dict(path=str(target),sha256=sha(target),backend='cpu',threads=8,
                         dt_max_s=dt,output_interval_s=10.,end_time_s=60.,cfl=1.)
    audit['builder_sha256']=sha(__file__)
    audit_path.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    nodes, cells, bounds = mesh()
    rows = inspect_dem(nodes)
    print(json.dumps(rows, indent=2))
    (OUT/'DEM_SOURCE_COMPARISON.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    if args.inspect_only:
        return
    template=ROOT/'validation/_amr132/clean_final/cpu5m_PROJECT.json'
    base=json.loads(template.read_text(encoding='utf-8'))
    spatial=base['settings']['domain']['spatial_reference']
    original,domain,assignments,grid,alignment,metrics,polygons=domain_and_grid(nodes,cells,bounds,spatial)
    dem,terrain=rasterize_executed_bed(nodes,cells,grid,spatial)
    cases=build_inputs(base,domain,assignments,grid,dem)
    result_header=inspect_result_header(nodes,cells)
    audit=dict(schema='braids.validation.matched-iber134.v1',builder_sha256=sha(__file__),sources={str(IBER/'Iber2D.dat'):sha(IBER/'Iber2D.dat'),
            str(IBER/'Iber_Problemdata.dat'):sha(IBER/'Iber_Problemdata.dat'),
            str(IBER/'proceso.rep'):sha(IBER/'proceso.rep'),str(template):sha(template)},
            mesh=metrics,axis_alignment=alignment,terrain=terrain,inputs=cases,result_header=result_header,
            limitations=[
                'Iber triangles and BRAIDS Cartesian cut cells are not identical. 3.25 m square area 10.5625 m2 is within 1.17% of Iber median area 10.68724 m2; side lengths and face-sum geometry differ.',
                'Only assigned partial inlet/outlet sections are rotated/snapped to the BRAIDS lattice. Adjacent walls reconnect; other walls retain their original design geometry to source coordinate precision.',
                'Iber initial volume is 0.08471 m3 over 56850 m2 (about 1.49 micrometres mean). BRAIDS explicitly starts dry with zero momentum. No hidden water film is introduced.',
                'Bed Manning n=0.03 matches. Iber saved wall Manning=0.015 has no identical additional BRAIDS wall-drag setting; BRAIDS uses impermeable reflective walls with its current bed resistance.',
                'Iber dry threshold is 0.001 m. BRAIDS retains its qualified dry/wet and small-cell treatment, not a substituted Iber wetting implementation.',
                'Both use second-order explicit methods, but different flux, reconstruction, CFL definition and boundary realizations. Equal CFL numbers would not imply equal timesteps.',
                'Iber executed vertex bed is primary provenance; raw corrected DEM was not exactly identical. Raster representation and cell averaging introduce an explicitly measured resampling difference.',
                'Existing Iber timings include its own I/O and startup conventions. No Iber simulation or user input is modified by this helper.'],
            outputs_unchanged_source_check={str(IBER/'Iber2D.dat'):sha(IBER/'Iber2D.dat')})
    (OUT/'MATCHED_INPUTS.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,8),layout='constrained')
    origin=nodes[:,:2].min(0)
    for ax,title in zip(axes,['Original Iber executed mesh','BRAIDS fixed 3.25 m geometry']):
        ax.set(title=title,aspect='equal',xlabel='Relative easting (m)',ylabel='Relative northing (m)')
    from matplotlib.collections import PolyCollection
    axes[0].add_collection(PolyCollection([nodes[c,:2]-origin for c in cells],facecolors='none',edgecolors='#74828b',linewidths=.2))
    axes[1].add_collection(PolyCollection([np.array(p.exterior.coords)-origin for p in polygons],facecolors='none',edgecolors='#3281a3',linewidths=.2))
    for kind,edges in bounds.items():
        for edge in edges:axes[0].plot(*(nodes[list(edge),:2]-origin).T,color='red' if kind=='inlet' else 'blue',linewidth=2)
    for seg in domain.boundary_segments(assignments):
        if seg.kind!='wall':axes[1].plot(*(np.array([seg.start_xy,seg.end_xy])-origin).T,color='red' if seg.kind=='inlet' else 'blue',linewidth=2)
    for ax in axes:ax.autoscale_view();ax.grid(alpha=.15)
    fig.suptitle('Matched physical reach; open-section axis adjustment is explicitly recorded\nRed = inlet Q11 m³/s, blue = critical outlet; walls remain cropped')
    fig.savefig(OUT/'MATCHED_GEOMETRY.png',dpi=160);plt.close(fig)
    print(json.dumps({'mesh':metrics,'axis_sections':alignment['sections'],'terrain':terrain,'cases':list(cases)},indent=2))


if __name__ == '__main__':
    main()
