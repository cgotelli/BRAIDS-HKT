"""Conservative, static union of small polygon fragments and face neighbours.

Only preparation geometry changes: no time-dependent averaging, relaxed CFL,
discarded area or moved physical boundary. Each resulting polygon is one native
finite volume with the usual shared-face flux and accepted-boundary ledger.
"""
import heapq
import math
import numpy as np
from osgeo import ogr
from .cut_cells import CutCell, CutFace, CutCellMesh, cut_cell_polygon


def _renumber_cell(cell, cell_id):
    """Copy a cell without the reflection overhead of dataclasses.replace."""
    return CutCell(
        cell_id, cell.i, cell.j, cell.component, cell.area_m2,
        cell.centroid_xy, cell.vertices_xy, cell.member_indices, cell.holes_xy,
        cell.level, cell.hierarchy_node_id, cell.nominal_area_m2,
    )


def _renumber_face(face, face_id, owner, neighbour):
    """Copy a face in the hot remapping loop using the fixed schema directly."""
    return CutFace(
        face_id, owner, neighbour, face.start_xy, face.end_xy,
        face.outward_normal, face.length_m, face.edge_id, face.segment_id,
        face.bc_id, face.kind, face.edge_start_m, face.edge_end_m,
        face.bed_elevation_m, face.section_bed_start_m,
        face.section_bed_end_m, face.structure_id, face.boundary_role, face.breakline_ids,
    )


def agglomerate_small_cells(mesh, minimum_area_fraction=.5, *, preserve_parent_boundaries=False,
                            consume_input=False):
    """Join small cells locally, returning mesh, old-to-new IDs and an audit.

    Clusters stay within a 2 by 2 block of background cells. Only a positive
    shared interior face permits a union, so walls and disconnected fragments
    cannot be crossed. An isolated/unresolvable fragment is retained and
    reported; the native CFL check is always applied to the actual geometry.
    """
    fraction=float(minimum_area_fraction)
    if not isinstance(consume_input,bool):
        raise ValueError('consume_input must be boolean')
    if not math.isfinite(fraction) or not 0<=fraction<=.5:
        raise ValueError('Minimum boundary-cell area fraction must lie in [0, 0.5]')
    if any(c.member_indices for c in mesh.cells):
        raise ValueError('Apply agglomeration once to the original clipped mesh')
    count=len(mesh.cells);area=np.array([c.area_m2 for c in mesh.cells])
    nominal=mesh.grid.dx_m*mesh.grid.dy_m;target=fraction*nominal
    parent=np.arange(count);groups={};pending=[(a,i) for i,a in enumerate(area) if a<target]
    if not pending:
        return mesh,parent,{'minimum_area_fraction':fraction,'merged_cell_count':0,
            'removed_interior_faces':0,'unresolved_small_cells':[], 'clusters':[]}
    offset,face_ids,_=mesh.incidence();heapq.heapify(pending)

    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=int(parent[i])
        return i

    def members(i):
        return groups.get(i,[i])

    while pending:
        old_area,c=heapq.heappop(pending)
        if root(c)!=c or area[c]!=old_area or area[c]>=target:continue
        adjacent={}
        for original in members(c):
            for at in range(offset[original]-1,offset[original+1]-1):
                face=mesh.faces[face_ids[at]-1]
                n=face.neighbour if face.owner==original else face.owner
                if n<0:continue
                n=root(n)
                if n!=c:adjacent[n]=adjacent.get(n,0.)+face.length_m
        admissible=[]
        for n,length in adjacent.items():
            ids=members(c)+members(n)
            ii=[mesh.cells[i].i for i in ids];jj=[mesh.cells[i].j for i in ids]
            if preserve_parent_boundaries and (len({i//2 for i in ii})>1 or len({j//2 for j in jj})>1):
                continue
            if max(ii)-min(ii)<=1 and max(jj)-min(jj)<=1:
                # The finite-volume reconstruction needs a centre inside the
                # kernel of the resulting polygon, on the inward side of
                # every retained face. Do not create a folded control volume.
                origin=np.array(mesh.cells[ids[0]].centroid_xy)
                center=origin+sum((mesh.cells[i].area_m2*(np.array(mesh.cells[i].centroid_xy)-origin) for i in ids),np.zeros(2))/math.fsum(mesh.cells[i].area_m2 for i in ids)
                contained=set(ids);valid=True
                for original in ids:
                    for at in range(offset[original]-1,offset[original+1]-1):
                        face=mesh.faces[face_ids[at]-1]
                        other=face.neighbour if face.owner==original else face.owner
                        if other in contained:continue
                        sign=1 if face.owner==original else -1
                        mid=.5*(np.array(face.start_xy)+face.end_xy)
                        if sign*np.dot(center-mid,face.outward_normal)>1.e-9*max(mesh.grid.dx_m,mesh.grid.dy_m):
                            valid=False;break
                    if not valid:break
                if valid:admissible.append((length,area[n],-n,n))
        if not admissible:continue
        n=max(admissible)[-1]
        groups[n]=members(n)+members(c);groups.pop(c,None)
        parent[c]=n;area[n]=math.fsum(mesh.cells[i].area_m2 for i in groups[n])
        if area[n]<target:heapq.heappush(pending,(area[n],n))
    roots={root(i) for i in range(count)}
    anchors={r:max(members(r),key=lambda i:(mesh.cells[i].area_m2,-i)) for r in roots}
    ordered=sorted(roots,key=lambda r:(mesh.cells[anchors[r]].j,mesh.cells[anchors[r]].i,mesh.cells[anchors[r]].component))
    new_id={r:i for i,r in enumerate(ordered)}
    mapping=np.array([new_id[root(i)] for i in range(count)])
    origin=np.array([mesh.grid.xlower_m,mesh.grid.ylower_m]);cells=[];clusters=[]
    for r in ordered:
        ids=members(r);anchor=mesh.cells[anchors[r]];number=new_id[r]
        if len(ids)==1:
            if consume_input:
                object.__setattr__(anchor,'cell_id',number)
                cells.append(anchor)
            else:
                cells.append(_renumber_cell(anchor, number))
            continue
        geometry=None
        for i in ids:
            poly=cut_cell_polygon(mesh.cells[i],origin)
            geometry=poly if geometry is None else geometry.Union(poly)
        if ogr.GT_Flatten(geometry.GetGeometryType())!=ogr.wkbPolygon or not geometry.IsValid():
            raise ValueError(f'Cannot form one polygon from clipped cells {ids}')
        total=math.fsum(mesh.cells[i].area_m2 for i in ids)
        if abs(geometry.GetArea()-total)>1.e-8*max(nominal,total):
            raise ValueError('Cell union did not preserve clipped area')
        centroid=geometry.Centroid()
        if not centroid.Within(geometry):centroid=geometry.PointOnSurface()
        center=origin+np.array([centroid.GetX(),centroid.GetY()])
        vertices=tuple(tuple(np.array(p[:2])+origin) for p in geometry.GetGeometryRef(0).GetPoints())
        holes=tuple(tuple(tuple(np.array(p[:2])+origin) for p in geometry.GetGeometryRef(ring).GetPoints())
                    for ring in range(1,geometry.GetGeometryCount()))
        indices=tuple(sorted((mesh.cells[i].i,mesh.cells[i].j,mesh.cells[i].component) for i in ids))
        cells.append(CutCell(
            number, anchor.i, anchor.j, anchor.component, total, tuple(center),
            vertices, indices, holes, anchor.level, anchor.hierarchy_node_id,
            anchor.nominal_area_m2,
        ))
        clusters.append({'cell_id':number,'original_cell_ids':sorted(ids),'area_m2':total})
    faces=[]
    for face in mesh.faces:
        c=int(mapping[face.owner]);n=-1 if face.neighbour<0 else int(mapping[face.neighbour])
        if n==c:continue
        if consume_input:
            object.__setattr__(face,'face_id',len(faces))
            object.__setattr__(face,'owner',c)
            object.__setattr__(face,'neighbour',n)
            faces.append(face)
        else:
            faces.append(_renumber_face(face, len(faces), c, n))
    merged=CutCellMesh(mesh.domain_id, mesh.crs_wkt, mesh.grid, tuple(cells),
                       tuple(faces), mesh.obstacle_ids, mesh.solid_area_m2)
    audit={'minimum_area_fraction':fraction,'maximum_cluster_span_background_cells':[2,2],
        'merged_cell_count':count-len(cells),'removed_interior_faces':len(mesh.faces)-len(faces),
        'original_minimum_area_fraction':float(min(c.area_m2 for c in mesh.cells)/nominal),
        'final_minimum_area_fraction':float(min(c.area_m2 for c in cells)/nominal),
        'area_difference_m2':math.fsum(c.area_m2 for c in cells)-math.fsum(c.area_m2 for c in mesh.cells),
        'unresolved_small_cells':[c.cell_id for c in cells if c.area_m2<target],
        'boundary_geometry_and_IDs_unchanged':True,'clusters':clusters}
    return merged,mapping,audit
