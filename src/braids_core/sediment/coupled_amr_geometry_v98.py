"""Authenticated Cartesian geometry for a joined 2:1 leaf interface.

Topology alone has no global coordinates. This additive certificate binds real
v76 topology to supplied global rectangle bounds and IDs. It derives the normal
cell-centre distance for a piecewise-constant coarse donor. A physical-boundary
half-cell distance must not be reused for that donor. No solver state, clock or
constitutive operator is created here. Covered rectangles are topology evidence,
not accepted cells or proof that a regrid transaction has been composed.
"""
from dataclasses import dataclass

import numpy as np

from .density_active_projection_v97 import _array,_digest,_hash
from .density_trace_projection_v98 import _hex
from .suspended_lifecycle_live_v76 import SuspendedLifecycleExecutionV76,execution_document_v76


def _frozen(a):
    result=np.array(a,copy=True,order="C");result.setflags(write=False);return result


def _ids(value,count,name):
    a=np.asarray(value)
    if (a.shape!=(count,) or a.dtype.kind not in "iu" or np.any(a<0) or len(np.unique(a))!=count):
        raise ValueError("v98 "+name+" must be distinct nonnegative integer IDs")
    return _frozen(a.astype(np.int64))


def _close(a,b,name):
    tol=512*np.finfo(float).eps*max(1.,float(np.max(np.abs(a))),float(np.max(np.abs(b))))
    if np.any(np.abs(np.asarray(a)-np.asarray(b))>tol):
        raise ValueError("v98 geometric "+name+" differs")


@dataclass(frozen=True)
class CartesianAMRPatchGeometryV98:
    execution_fingerprint: str
    cell_bounds_m: np.ndarray
    face_endpoints_m: np.ndarray
    cell_global_ids: np.ndarray
    face_global_ids: np.ndarray
    receipt_sha256: str


def _patch_payload(p):
    return dict(kind="cartesian_AMR_patch_geometry_v98",execution=p.execution_fingerprint,
        geometry_sha256=_digest(p.cell_bounds_m,p.face_endpoints_m,p.cell_global_ids,p.face_global_ids),
        physics_evaluation_count=0,state_or_clock_publication_count=0)


def resolve_cartesian_amr_patch_geometry_v98(grid,*,cell_bounds_m,cell_global_ids,face_global_ids):
    if type(grid) is not SuspendedLifecycleExecutionV76:
        raise ValueError("v98 geometry requires actual v76 topology")
    execution_document_v76(grid)
    nc,nf=grid.cell_count,grid.face_count
    bounds=_array(cell_bounds_m,(nc,4),"global Cartesian cell bounds")
    widths=bounds[:,2:]-bounds[:,:2]
    if np.any(widths<=0):raise ValueError("v98 global Cartesian bounds must have positive widths")
    _close(widths.T,grid.rectangle_width_m,"cell widths")
    _close(np.prod(widths,axis=1),grid.cell_area_m2,"cell areas")
    ids=_ids(cell_global_ids,nc,"global cell IDs");fids=_ids(face_global_ids,nf,"global face IDs")
    owner=np.asarray(grid.owner);nei=np.asarray(grid.neighbour)
    normal=np.asarray(grid.normal).T
    cardinal=(np.abs(normal[:,0])==1)&(normal[:,1]==0) | (normal[:,0]==0)&(np.abs(normal[:,1])==1)
    if not np.all(cardinal):raise ValueError("v98 Cartesian faces require cardinal unit normals")
    ob=bounds[owner];vertical=normal[:,0]!=0
    pos=normal[:,0]+normal[:,1]>0
    endpoints=np.empty((nf,2,2))
    x=np.where(pos,ob[:,2],ob[:,0]);y=np.where(pos,ob[:,3],ob[:,1])
    endpoints[:,0,0]=np.where(vertical,x,ob[:,0]);endpoints[:,1,0]=np.where(vertical,x,ob[:,2])
    endpoints[:,0,1]=np.where(vertical,ob[:,1],y);endpoints[:,1,1]=np.where(vertical,ob[:,3],y)
    _close(np.linalg.norm(endpoints[:,1]-endpoints[:,0],axis=1),grid.face_length_m,"face lengths")
    interior=np.flatnonzero(nei>=0)
    if interior.size:
        nb=bounds[nei[interior]];ov=ob[interior];vv=vertical[interior];pp=pos[interior]
        opposing=np.where(vv,np.where(pp,nb[:,0],nb[:,2]),np.where(pp,nb[:,1],nb[:,3]))
        plane=np.where(vv,x[interior],y[interior]);_close(opposing,plane,"internal face planes")
        _close(np.where(vv,nb[:,1],nb[:,0]),np.where(vv,ov[:,1],ov[:,0]),"internal lower endpoints")
        _close(np.where(vv,nb[:,3],nb[:,2]),np.where(vv,ov[:,3],ov[:,2]),"internal upper endpoints")
        centres=(bounds[:,:2]+bounds[:,2:])*.5
        distance=np.sum((centres[nei[interior]]-centres[owner[interior]])*normal[interior],axis=1)
        _close(distance,grid.centre_distance_m[interior],"ordinary internal centre distances")
    value=CartesianAMRPatchGeometryV98(grid.fingerprint,_frozen(bounds),_frozen(endpoints),ids,fids,"")
    return CartesianAMRPatchGeometryV98(**{**value.__dict__,"receipt_sha256":_hash(_patch_payload(value))})


def validate_cartesian_amr_patch_geometry_v98(patch,grid):
    if (type(patch) is not CartesianAMRPatchGeometryV98 or patch.execution_fingerprint!=grid.fingerprint or
        patch.receipt_sha256!=_hash(_patch_payload(patch)) or any(a.flags.writeable for a in (
            patch.cell_bounds_m,patch.face_endpoints_m,patch.cell_global_ids,patch.face_global_ids))):
        raise ValueError("v98 global patch geometry identity differs")
    actual=resolve_cartesian_amr_patch_geometry_v98(grid,cell_bounds_m=patch.cell_bounds_m,
        cell_global_ids=patch.cell_global_ids,face_global_ids=patch.face_global_ids)
    if actual.receipt_sha256!=patch.receipt_sha256:raise ValueError("v98 rebuilt patch geometry differs")


@dataclass(frozen=True)
class CoupledAMRJoinedGeometryV98:
    coarse_patch_sha256: str
    fine_patch_sha256: str
    interface_fingerprint: str
    coarse_face: int
    fine_faces: tuple
    fine_owner_cells: np.ndarray
    coarse_owner_cell: int
    normal_centre_distance_m: np.ndarray
    tangential_centre_offset_m: np.ndarray
    covered_parent_global_id: int
    covered_parent_bounds_m: np.ndarray
    covered_child_cells: np.ndarray
    covered_child_global_ids: np.ndarray
    covered_parent_area_m2: float
    receipt_sha256: str


def _joined_payload(p):
    return dict(kind="joined_2to1_cell_centre_geometry_v98",coarse=p.coarse_patch_sha256,fine=p.fine_patch_sha256,
        interface=p.interface_fingerprint,coarse_face=p.coarse_face,fine_faces=list(p.fine_faces),
        coarse_owner_cell=p.coarse_owner_cell,covered_parent_global_id=p.covered_parent_global_id,
        geometry_sha256=_digest(p.fine_owner_cells,p.normal_centre_distance_m,p.tangential_centre_offset_m,
            p.covered_parent_bounds_m,p.covered_child_cells,p.covered_child_global_ids),
        covered_parent_area_m2=p.covered_parent_area_m2,donor_reconstruction="cell_centre_piecewise_constant",
        covered_parent_accepted_owner_count=0,state_or_clock_publication_count=0,regrid_admission=False)


def resolve_joined_amr_geometry_v98(coarse_grid,fine_grid,coarse_patch,fine_patch,*,interface_fingerprint,
    coarse_face,fine_faces,covered_parent_global_id,covered_parent_bounds_m,covered_child_cells):
    validate_cartesian_amr_patch_geometry_v98(coarse_patch,coarse_grid)
    validate_cartesian_amr_patch_geometry_v98(fine_patch,fine_grid)
    identity=_hex(interface_fingerprint,"interface fingerprint");faces=tuple(fine_faces)
    if (type(coarse_face) is not int or not 0<=coarse_face<coarse_grid.face_count or len(faces)!=2 or
        len(set(faces))!=2 or any(type(f) is not int or not 0<=f<fine_grid.face_count for f in faces)):
        raise ValueError("v98 joined face selection differs")
    c=coarse_grid;f=fine_grid
    if (c.neighbour[coarse_face]!=-1 or c.boundary_index[coarse_face]!=-1 or
        np.any(f.neighbour[list(faces)]!=-1) or np.any(f.boundary_index[list(faces)]!=-1)):
        raise ValueError("v98 joined faces must be actual nonphysical internal traces")
    if np.intersect1d(coarse_patch.cell_global_ids,fine_patch.cell_global_ids).size or np.intersect1d(
        coarse_patch.face_global_ids,fine_patch.face_global_ids).size:
        raise ValueError("v98 joined active global IDs overlap")
    cn=c.normal[:,coarse_face];_close(f.normal[:,list(faces)],np.repeat(-cn[:,None],2,axis=1),"interface normals")
    ce=coarse_patch.face_endpoints_m[coarse_face];fe=fine_patch.face_endpoints_m[list(faces)]
    vertical=abs(cn[0])==1;axis=1 if vertical else 0;plane=1-axis
    _close(fe[:,:,plane],np.full((2,2),ce[0,plane]),"coarse/fine face positions")
    order=np.argsort(fe[:,0,axis],kind="stable");ordered=fe[order]
    _close(np.array([ordered[0,0,axis],ordered[0,1,axis],ordered[1,1,axis]]),
           np.array([ce[0,axis],ordered[1,0,axis],ce[1,axis]]),"two-face partition endpoints")
    _close(c.face_length_m[coarse_face],sum(f.face_length_m[list(faces)]),"interface partition length")
    cc=int(c.owner[coarse_face]);fc=np.asarray(f.owner[list(faces)],np.int64)
    cb=coarse_patch.cell_bounds_m[cc];fb=fine_patch.cell_bounds_m[fc]
    separation=(fb[:,:2]+fb[:,2:])*.5-(cb[:2]+cb[2:])*.5
    distance=separation@cn;tangent=separation@np.array([-cn[1],cn[0]])
    if np.any(distance<=0):raise ValueError("v98 donor-centre distance must be positive")
    _close((cb[2:]-cb[:2]),2*(fb[:,2:]-fb[:,:2]),"2to1 cell dimensions")
    pbound=_array(covered_parent_bounds_m,(4,),"covered parent bounds")
    children=_ids(covered_child_cells,4,"covered child indices")
    if np.any(children>=f.cell_count):raise ValueError("v98 covered children outside fine patch")
    if (type(covered_parent_global_id) is not int or covered_parent_global_id<0 or
        covered_parent_global_id in coarse_patch.cell_global_ids or covered_parent_global_id in fine_patch.cell_global_ids):
        raise ValueError("v98 covered parent cannot also be an accepted leaf")
    pwidth=pbound[2:]-pbound[:2]
    if np.any(pwidth<=0):raise ValueError("v98 covered parent dimensions invalid")
    children_bounds=fine_patch.cell_bounds_m[children];cw=children_bounds[:,2:]-children_bounds[:,:2]
    _close(cw,np.repeat((pwidth*.5)[None,:],4,axis=0),"covered child sizes")
    half=pwidth*.5
    expected=np.array([pbound[:2]+np.array([i*half[0],j*half[1]]) for j in (0,1) for i in (0,1)])
    actual=children_bounds[:,:2];sort=lambda a:a[np.lexsort((a[:,0],a[:,1]))]
    _close(sort(actual),sort(expected),"covered child tiling")
    area=float(np.prod(pwidth));_close(f.cell_area_m2[children].sum(),area,"covered parent area")
    p=CoupledAMRJoinedGeometryV98(coarse_patch.receipt_sha256,fine_patch.receipt_sha256,identity,coarse_face,faces,
        _frozen(fc),cc,_frozen(distance),_frozen(tangent),covered_parent_global_id,_frozen(pbound),
        children,_frozen(fine_patch.cell_global_ids[children]),area,"")
    return CoupledAMRJoinedGeometryV98(**{**p.__dict__,"receipt_sha256":_hash(_joined_payload(p))})


def validate_joined_amr_geometry_v98(p,coarse_grid,fine_grid,coarse_patch,fine_patch):
    if (type(p) is not CoupledAMRJoinedGeometryV98 or p.receipt_sha256!=_hash(_joined_payload(p)) or
        any(a.flags.writeable for a in (p.fine_owner_cells,p.normal_centre_distance_m,p.tangential_centre_offset_m,
            p.covered_parent_bounds_m,p.covered_child_cells,p.covered_child_global_ids))):
        raise ValueError("v98 joined geometry identity differs")
    actual=resolve_joined_amr_geometry_v98(coarse_grid,fine_grid,coarse_patch,fine_patch,
        interface_fingerprint=p.interface_fingerprint,coarse_face=p.coarse_face,fine_faces=p.fine_faces,
        covered_parent_global_id=p.covered_parent_global_id,covered_parent_bounds_m=p.covered_parent_bounds_m,
        covered_child_cells=p.covered_child_cells)
    if actual.receipt_sha256!=p.receipt_sha256:raise ValueError("v98 rebuilt joined geometry differs")
