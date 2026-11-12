"""Exact polygon-to-exterior-face mapping and grouped native inlet/outlet export.

Current native scope is a rectangle matching the fixed Cartesian grid, with
any number of collinear polygon edges. Unsupported geometry is rejected, never
silently rasterized, rotated or replaced with its bounding box.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import tempfile
import numpy as np
from .domain import PolygonDomain, BoundaryAssignment
from .inlet import Face, InletError, InletSpec
from .inlet_io import NativeGrid, NativeFace, SIDE_NORMALS, write_native_inlets
from .section import Section


@dataclass(frozen=True)
class MappedFace:
    edge_id: str
    segment_id: str
    bc_id: str
    kind: str
    side: int
    index: int
    midpoint_xy: tuple[float, float]
    inward_normal: tuple[float, float]
    length_m: float
    bed_m: float


@dataclass(frozen=True)
class DomainFaceMap:
    domain_id: str
    grid: NativeGrid
    crs_wkt: str
    faces: tuple[MappedFace, ...]

    def inlet_faces(self, bc_id: str, *, station_origin_m: float = 0.) -> tuple[NativeFace, ...]:
        """Map one connected straight inlet, preserving its full section width.

        The caller's Section uses the same increasing x/y station convention,
        regardless of the polygon's CCW direction. Disconnected/noncollinear
        pieces need an explicit allocation/control design and are rejected.
        """
        rows=sorted((f for f in self.faces if f.bc_id==bc_id),key=lambda f:(f.side,f.index))
        if not rows or any(f.kind!='inlet' for f in rows):raise InletError('Unknown inlet ID')
        if len({f.side for f in rows})!=1 or any(b.index!=a.index+1 for a,b in zip(rows,rows[1:])):
            raise InletError('Disconnected or noncollinear inlet requires explicit allocation/control')
        if not math.isfinite(station_origin_m):raise InletError('Section station origin must be finite')
        return tuple(NativeFace(Face(f'{f.side}:{f.index}',station_origin_m+k*f.length_m,
            station_origin_m+(k+1)*f.length_m,f.bed_m,f.inward_normal),f.side,f.index) for k,f in enumerate(rows))

    def metadata(self):
        return dict(domain_id=self.domain_id,grid=asdict(self.grid),crs_wkt=self.crs_wkt,
            scope='Exact rectangular perimeter; whole exterior Cartesian faces',
            faces=[asdict(f) for f in self.faces])


def map_domain_faces(domain: PolygonDomain, assignments: list[BoundaryAssignment],
                     grid: NativeGrid, bed_m) -> DomainFaceMap:
    """Map every perimeter face exactly once, including all default walls.

    ``bed_m[i,j]`` is the represented native cell bed, shape (nx,ny), in the
    domain/grid projected metre CRS. It is not resampled by this function.
    """
    if not isinstance(domain,PolygonDomain) or not isinstance(grid,NativeGrid):
        raise InletError('An explicit PolygonDomain and NativeGrid are required')
    bed=np.asarray(bed_m,dtype=float)
    if bed.shape!=(grid.nx,grid.ny) or not np.isfinite(bed).all():
        raise InletError('Native bed must be finite with shape (nx,ny)')
    x0,y0=grid.xlower_m,grid.ylower_m;x1=x0+grid.nx*grid.dx_m;y1=y0+grid.ny*grid.dy_m
    tol=1e-9*min(grid.dx_m,grid.dy_m)
    near=lambda a,b:abs(a-b)<=tol
    rows={}
    for segment in domain.boundary_segments(assignments):
        a,b=segment.start_xy,segment.end_xy
        if near(a[0],x0) and near(b[0],x0):side=1
        elif near(a[0],x1) and near(b[0],x1):side=2
        elif near(a[1],y0) and near(b[1],y0):side=3
        elif near(a[1],y1) and near(b[1],y1):side=4
        else:raise InletError('Polygon needs native oblique/interior cut faces; exterior mapping cannot represent it')
        if not np.allclose(segment.inward_normal,SIDE_NORMALS[side],rtol=0,atol=1e-12):
            raise InletError('Polygon and native inward normals disagree')
        axis=1 if side<=2 else 0;spacing=grid.dy_m if side<=2 else grid.dx_m
        origin=y0 if side<=2 else x0;count=grid.ny if side<=2 else grid.nx
        bounds=sorted([(a[axis]-origin)/spacing,(b[axis]-origin)/spacing]);indices=[round(v) for v in bounds]
        if any(abs(v-i)>1e-9 for v,i in zip(bounds,indices)):
            raise InletError('Polygon edge/assignment cuts a native face; partial faces require native support')
        if not 0<=indices[0]<indices[1]<=count:raise InletError('Boundary lies outside native perimeter')
        for index in range(*indices):
            key=(side,index)
            if key in rows:raise InletError('Duplicate physical face ownership')
            i=(0 if side==1 else grid.nx-1) if side<=2 else index
            j=index if side<=2 else (0 if side==3 else grid.ny-1)
            midpoint=(x0 if side==1 else x1,y0+(index+.5)*spacing) if side<=2 else (x0+(index+.5)*spacing,y0 if side==3 else y1)
            rows[key]=MappedFace(segment.edge_id,segment.segment_id,segment.bc_id,segment.kind,
                side,index,midpoint,SIDE_NORMALS[side],spacing,float(bed[i,j]))
    expected={(s,i) for s in range(1,5) for i in range(grid.ny if s<=2 else grid.nx)}
    if set(rows)!=expected or not math.isclose(domain.area_m2,grid.nx*grid.ny*grid.dx_m*grid.dy_m,rel_tol=1e-12):
        raise InletError('Polygon must exactly cover the Cartesian grid rectangle')
    return DomainFaceMap(domain.domain_id,grid,domain.crs_wkt,tuple(rows[k] for k in sorted(rows)))


@dataclass(frozen=True)
class OutletSpec:
    bc_id: str
    mode: str = 'stage'  # stage or supercritical; no imposed discharge
    stage_m: float | None = None
    backflow_policy: str = 'error'

    def __post_init__(self):
        if not isinstance(self.bc_id,str) or not self.bc_id or any(c.isspace() for c in self.bc_id):
            raise InletError('Outlet requires a nonempty stable ID without whitespace')
        if self.mode not in ('stage','supercritical'):raise InletError('Unknown outlet mode')
        if self.backflow_policy not in ('error','allow'):raise InletError('Explicit outlet backflow policy required')
        if self.mode=='stage' and (self.stage_m is None or isinstance(self.stage_m,bool) or not math.isfinite(self.stage_m)):
            raise InletError('Stage outlet requires finite absolute elevation')
        if self.mode=='supercritical' and (self.stage_m is not None or self.backflow_policy!='error'):
            raise InletError('Supercritical outlet has no prescribed stage or incoming-flow policy')


def write_domain_boundaries(path: Path, mapping: DomainFaceMap, inlets: list[InletSpec],
                            sections: dict[str,Section], outlets: list[OutletSpec], *,
                            station_origins_m=None, dry_policy='error', supercritical_policy='error',
                            inlet_backflow_policy='error') -> dict:
    """Export v2 IDs shared by water, suspended and bedload configuration.

    Inlet hydraulic sections retain the existing convex-basin validation.
    An outlet ID may span several sides: one common stage, local outgoing
    characteristics, and no imposed/duplicated Q. Wall faces stay unlisted in
    the native open-face table but remain explicit in the geometry sidecar.
    All native exterior boundary codes must be user code 0, without the legacy
    whole-side hydraulic_bc override; the validator below enforces that deck.
    """
    if not isinstance(mapping, DomainFaceMap):
        raise InletError('Native v2 requires DomainFaceMap; cut-cell geometry is not yet connected to WAVE stepping')
    active={f.bc_id:f.kind for f in mapping.faces if f.kind!='wall'}
    specs={s.bc_id:s for s in [*inlets,*outlets]}
    if len(specs)!=len(inlets)+len(outlets) or set(specs)!=set(active):
        raise InletError('Each selected inlet/outlet ID needs exactly one matching specification')
    if any(active[s.bc_id]!='inlet' for s in inlets) or any(active[s.bc_id]!='outlet' for s in outlets):
        raise InletError('Boundary kind differs from polygon assignment')
    ids={name:k+1 for k,name in enumerate(sorted(active))}
    origins=station_origins_m or {};headers={}
    # Reuse the established section exporter as the inlet-validation authority.
    if inlets:
        with tempfile.TemporaryDirectory(prefix='braids_domain_') as temporary:
            target=Path(temporary)/'braids_boundary.data'
            meta=write_native_inlets(target,mapping.grid,inlets,sections,
                {s.bc_id:mapping.inlet_faces(s.bc_id,station_origin_m=origins.get(s.bc_id,0.)) for s in inlets},
                dry_policy=dry_policy,supercritical_policy=supercritical_policy,backflow_policy=inlet_backflow_policy)
            lines=target.read_text().splitlines();cursor=3
            for old_id in sorted(meta['numeric_to_string_id'],key=int):
                name=meta['numeric_to_string_id'][old_id];head=lines[cursor].split();size=1+int(head[7])+int(head[8])
                head[0]=str(ids[name]);headers[name]=[' '.join(head),*lines[cursor+1:cursor+size]];cursor+=size
    for spec in outlets:
        # Placeholder profile/Q rows keep the v1 table layout. They are never
        # used to compute outlet geometry or impose discharge in mode 5/6.
        headers[spec.bc_id]=[f'{ids[spec.bc_id]} {5 if spec.mode=="stage" else 6} 1 1 0 0 {int(spec.backflow_policy=="allow")} 2 1 {spec.stage_m if spec.stage_m is not None else 0:.17g} 0 .03 1',
            '0 0','1 0','0 0']
    opened=[f for f in mapping.faces if f.kind!='wall'];g=mapping.grid
    if not opened:raise InletError('At least one explicit open boundary is required by native v2')
    lines=['2',f'{g.nx} {g.ny} {g.xlower_m:.17g} {g.ylower_m:.17g} {g.dx_m:.17g} {g.dy_m:.17g}',f'{len(ids)} {len(opened)}']
    for name in sorted(ids):lines.extend(headers[name])
    lines.extend(f'{ids[f.bc_id]} {f.side} {f.index} {f.bed_m:.17g} {f.length_m:.17g}' for f in opened)
    payload='\n'.join(lines)+'\n'
    metadata=mapping.metadata();metadata.update(format_version=2,string_to_numeric_id=ids,
        numeric_to_string_id={str(v):k for k,v in ids.items()},outlets=[asdict(s) for s in outlets],
        data_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        outlet_Q_diagnostics='No Q prescribed; legacy requested/error columns unused for outlets. Use accepted face ledger.',
        inlets=[asdict(s) for s in inlets],station_origins_m=origins)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(payload);path.with_name(path.name+'.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def validate_domain_native_deck(work: Path) -> None:
    """Prevent a legacy side override from opening unassigned polygon walls."""
    lines=(Path(work)/'claw.data').read_text().splitlines()
    for key in ('bc_lower','bc_upper'):
        matches=[line.split('=:')[0].split() for line in lines if '=: '+key in line]
        if len(matches)!=1 or matches[0]!=['0','0']:
            raise InletError('Polygon deck requires user boundary code 0 on all exterior sides')
    legacy=Path(work)/'hydraulic_bc.data'
    if legacy.exists():
        for line in legacy.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#') and int(line.split()[1])!=0:
                raise InletError('Polygon deck cannot use a legacy whole-side hydraulic override')
