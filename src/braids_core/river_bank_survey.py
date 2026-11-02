"""Geographic bank survey, resolved once on the finest physical mesh.

Geometry is preparation work, independent of the numerical backend. Whole
eligible columns retain their existing material; a polygon does not invent a
second stock. Survey corridors remain finest while the rest of the mesh adapts.
The public project and preparation receipt carry this same explicit definition.
"""
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point, Polygon

SCHEMA = 'braids.geographic-bank-survey.v1'
POLICIES = dict(mapping='finest_cell_centroid_and_toe_corridor',
    grouping='nearest_toe_path_coordinate_ties', primary_toe='longest_face_coordinate_ties',
    refinement='survey_columns_and_toe_recipients_finest')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def _exact(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(name+': explicit fields required: '+', '.join(sorted(keys)))


def _points(value, minimum, name):
    if (not isinstance(value, list) or len(value) < minimum or
        any(not isinstance(p, list) or len(p) != 2 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in p) for p in value)):
        raise ValueError(name+': finite explicit XY coordinate lists required')
    return value


def validate(value, spatial_reference):
    """Strict metric coordinates, no CRS guessing or migration."""
    _exact(value, {'schema','spatial_reference','policies','fraction','critical_undercut_m','zones'}, 'survey')
    if value['schema'] != SCHEMA or value['policies'] != POLICIES:
        raise ValueError('Current survey schema and explicit mapping policies required')
    from osgeo import osr
    def crs(text):
        if not isinstance(text, str) or not text:
            raise ValueError('Explicit survey and domain CRS required')
        s = osr.SpatialReference()
        if s.SetFromUserInput(text) != 0 or not s.IsProjected() or abs(s.GetLinearUnits()-1.) > 1e-12:
            raise ValueError('Survey and domain must use a projected CRS in metres')
        return s
    if not crs(value['spatial_reference']).IsSame(crs(spatial_reference)):
        raise ValueError('Survey CRS differs from the domain; transform explicitly before import')
    fraction, critical = value['fraction'], value['critical_undercut_m']
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Toe supply fraction must be in (0,1]')
    if critical is not None and (type(critical) not in (int, float) or not math.isfinite(critical) or critical <= 0):
        raise ValueError('Block release requires an explicit positive undercut threshold, otherwise null')
    zones = value['zones']
    if not isinstance(zones, list) or not zones:
        raise ValueError('At least one surveyed bank zone required')
    names = set(); polygons = []
    for z in zones:
        _exact(z, {'id','shell_xy','holes_xy','toe_xy','tolerance_m'}, 'zone')
        if not isinstance(z['id'], str) or not z['id'].strip() or z['id'] in names:
            raise ValueError('Unique nonempty bank zone identifiers required')
        names.add(z['id'])
        shell = _points(z['shell_xy'], 3, 'bank polygon')
        if not isinstance(z['holes_xy'], list):
            raise ValueError('Explicit polygon hole list required')
        holes = [_points(r, 3, 'bank hole') for r in z['holes_xy']]
        poly = Polygon(shell, holes); line = LineString(_points(z['toe_xy'], 2, 'toe line'))
        if poly.is_empty or not poly.is_valid or poly.area <= 0:
            raise ValueError('Invalid bank polygon: '+z['id'])
        if line.is_empty or not line.is_simple or line.is_ring or line.length <= 0:
            raise ValueError('One simple open toe polyline required: '+z['id'])
        if any(poly.intersection(other).area > 0 for other in polygons):
            raise ValueError('Surveyed bank polygons overlap')
        polygons.append(poly)
        tol = z['tolerance_m']
        if type(tol) not in (int, float) or not math.isfinite(tol) or tol <= 0:
            raise ValueError('Explicit positive toe mapping tolerance required')
    return json.loads(canonical(value))


def _face_key(face, leaves):
    return (tuple(sorted((int(leaves[face.owner]), int(leaves[face.neighbour])))),
            tuple(sorted((tuple(face.start_xy), tuple(face.end_xy)))))


@dataclass(frozen=True)
class BankSurveyMap:
    """Immutable canonical JSON; callers receive copies, never live maps."""
    _definition: str
    _mapping: str

    @property
    def definition(self): return json.loads(self._definition)

    @property
    def mapping(self): return json.loads(self._mapping)

    @property
    def identity(self):
        return dict(schema=SCHEMA, definition_sha256=digest(self.definition),
                    mapping_sha256=digest(self.mapping), policies=dict(POLICIES),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    @property
    def required_leaves(self): return tuple(self.mapping['required_leaves'])

    def document(self):
        return dict(definition=self.definition,mapping=self.mapping,identity=self.identity)

    @classmethod
    def from_document(cls,doc,definition):
        _exact(doc,{'definition','mapping','identity'},'prepared survey')
        if doc['definition']!=definition:raise ValueError('Prepared survey differs from the project')
        validate(definition,definition['spatial_reference'])
        result=cls(canonical(definition),canonical(doc['mapping']))
        if result.identity!=doc['identity']:raise ValueError('Prepared survey map or mapper identity changed')
        return result

    def local(self, leaves, mesh):
        """Reindex fixed physical faces/groups, never resample the survey."""
        leaves = tuple(map(int, leaves)); mapping = self.mapping
        if len(leaves) != len(mesh.cells) or len(set(leaves)) != len(leaves):
            raise ValueError('Unique hierarchy leaves in actual mesh order required')
        local = {node:i for i,node in enumerate(leaves)}
        if not set(self.required_leaves).issubset(local):
            raise ValueError('Survey corridor must remain at finest resolution')
        # Compare full polygon geometry as well as stable IDs. Cropped cells
        # and holes cannot be replaced with their nominal Cartesian boxes.
        for row in mapping['cells']:
            c = mesh.cells[local[row['node']]]
            geometry = dict(vertices=c.vertices_xy, holes=c.holes_xy, area=c.area_m2)
            if digest(geometry) != row['geometry_sha256']:
                raise ValueError('Survey corridor physical cell geometry changed')
        faces = {}
        for f in mesh.faces:
            if f.neighbour >= 0:
                key = _face_key(f, leaves)
                if key in faces: raise ValueError('Duplicate physical internal face')
                faces[key] = f.face_id
        ids=[]; banks=[]; primary={}; lookup={}
        for row in mapping['faces']:
            key=(tuple(row['nodes']), tuple(tuple(p) for p in row['endpoints']))
            if key not in faces:
                raise ValueError('Surveyed toe face changed or became an exterior wall')
            face=faces[key]; ids.append(face); banks.append(local[row['bank']]); lookup[row['finest_face']]=face
        for bank, face in mapping['primary_faces'].items(): primary[local[int(bank)]]=lookup[face]
        groups={local[int(bank)]:[local[c] for c in members] for bank,members in mapping['groups'].items()}
        return dict(face_ids=ids, bank_cells=banks, groups=groups, primary_faces=primary)

    def install(self, owner, leaves, mesh):
        """Bind the qualified common operators without selecting a backend."""
        from .sediment.river_physical_toe import install as toe
        from .sediment.river_bank_failure import install as failure
        values = self.local(leaves, mesh); definition=self.definition
        # The exporter must retain the exact actual face indexing.
        d=owner.data
        for f in mesh.faces:
            if int(d['owner'][f.face_id]) != f.owner or int(d['neighbour'][f.face_id]) != f.neighbour:
                raise ValueError('Exported face order differs from geographic survey mesh')
        toe(owner, values['face_ids'], values['bank_cells'], fraction=definition['fraction'])
        if definition['critical_undercut_m'] is not None:
            failure(owner, values['groups'], values['primary_faces'],
                    critical_undercut=definition['critical_undercut_m'])
        owner._survey_required_leaves=self.required_leaves
        owner.identity['geographic_bank_survey']=self.identity
        return owner


def compile_survey(definition, hierarchy, *, cancelled=lambda:False):
    """Build the whole-column map once; no physical field affects selection."""
    definition=validate(definition, hierarchy.finest.crs_wkt)
    fine=tuple(range(hierarchy.offsets[-2],hierarchy.count)); mesh=hierarchy.select(fine)
    xy=np.array([c.centroid_xy for c in mesh.cells]); points=[Point(p) for p in xy]
    def check(i):
        if i%4096==0 and cancelled(): raise InterruptedError('Bank survey preparation cancelled')
    graph=[[] for _ in mesh.cells]
    for i,f in enumerate(mesh.faces):
        check(i)
        if f.neighbour>=0:
            a,b=f.owner,f.neighbour; distance=float(np.linalg.norm(xy[a]-xy[b]))
            if distance<=0: raise ValueError('Distinct adjacent bank centroids required')
            graph[a].append((b,distance)); graph[b].append((a,distance))
    eligibility={}; records=[]
    for zone in sorted(definition['zones'], key=lambda z:z['id']):
        if cancelled(): raise InterruptedError('Bank survey preparation cancelled')
        poly=Polygon(zone['shell_xy'],zone['holes_xy']); line=LineString(zone['toe_xy'])
        members=set()
        for i,p in enumerate(points):
            check(i)
            if poly.covers(p): members.add(i)
        if not members: raise ValueError('Bank polygon contains no finest cell centroid: '+zone['id'])
        if set(eligibility).intersection(members): raise ValueError('Shared boundary centroid belongs to two bank zones')
        eligibility.update({i:zone['id'] for i in members}); records.append((zone,line,members))
    all_faces=[]; groups={}; primary={}; audit=[]; required=set()
    for zone,line,members in records:
        selected=[]; distances={}
        for i,f in enumerate(mesh.faces):
            check(i)
            if f.neighbour<0 or (f.owner in members)==(f.neighbour in members): continue
            bank=f.owner if f.owner in members else f.neighbour
            recipient=f.neighbour if bank==f.owner else f.owner
            dist=line.distance(Point(np.mean([f.start_xy,f.end_xy],axis=0)))
            if dist<=zone['tolerance_m']:
                if recipient in eligibility: raise ValueError('Toe recipient is another surveyed bank column')
                selected.append((f,bank,recipient)); distances[f.face_id]=dist
        if not selected: raise ValueError('Toe guide selects no internal bank-toe face: '+zone['id'])
        # One guide must select one continuous face chain. A wide buffer cannot
        # silently select opposite sides or two disconnected fragments.
        endpoint_faces={}
        for f,_,_ in selected:
            for p in (f.start_xy,f.end_xy): endpoint_faces.setdefault(tuple(p),[]).append(f.face_id)
        if any(len(v)>2 for v in endpoint_faces.values()): raise ValueError('Toe corridor branches; narrow its tolerance')
        if sum(len(v)==1 for v in endpoint_faces.values())!=2:
            raise ValueError('Toe corridor must resolve to one open face chain')
        adjacency={f.face_id:set() for f,_,_ in selected}
        for ids in endpoint_faces.values():
            for f in ids: adjacency[f].update(set(ids)-{f})
        reached=set(); pending=[selected[0][0].face_id]
        while pending:
            f=pending.pop()
            if f not in reached: reached.add(f); pending.extend(adjacency[f]-reached)
        if len(reached)!=len(selected): raise ValueError('Toe corridor selects disconnected faces; revise the guide or tolerance')
        # Multisource shortest paths through actual internal bank apertures.
        # Coordinate-based tie breaking does not depend on face orientation.
        roots={b for _,b,_ in selected}; rank={b:tuple(xy[b]) for b in roots}
        best={b:(0.,rank[b],b) for b in roots}; queue=[(0.,rank[b],b,b) for b in roots];heapq.heapify(queue)
        count=0
        while queue:
            check(count);count+=1
            distance,key,root,c=heapq.heappop(queue)
            if best[c]!=(distance,key,root): continue
            for j,w in graph[c]:
                proposal=(distance+w,key,root)
                if j in members and (j not in best or proposal<best[j]):
                    best[j]=proposal;heapq.heappush(queue,(*proposal,j))
        if set(best)!=members: raise ValueError('Bank region contains cells disconnected from its toe across a wall')
        for root in roots:
            groups[str(fine[root])]=[fine[i] for i in sorted(members) if best[i][2]==root]
            ownfaces=[f for f,b,_ in selected if b==root]
            chosen=min(ownfaces,key=lambda f:(-f.length_m,tuple(np.mean([f.start_xy,f.end_xy],axis=0)),
                                                tuple(sorted((f.start_xy,f.end_xy)))))
            primary[str(fine[root])]=chosen.face_id
        required.update(fine[c] for c in members)
        for f,bank,recipient in selected:
            required.add(fine[recipient]);nodes,ends=_face_key(f,fine)
            all_faces.append(dict(finest_face=f.face_id,nodes=nodes,endpoints=ends,bank=fine[bank],zone=zone['id']))
        audit.append(dict(id=zone['id'],bank_cells=len(members),toe_faces=len(selected),
            mapped_length_m=math.fsum(f.length_m for f,_,_ in selected),
            maximum_guide_distance_m=max(distances.values()),polygon_area_m2=Polygon(zone['shell_xy'],zone['holes_xy']).area,
            eligible_cell_area_m2=math.fsum(mesh.cells[i].area_m2 for i in members)))
    if cancelled(): raise InterruptedError('Bank survey preparation cancelled')
    cells=[dict(node=node,vertices_xy=mesh.cells[node-fine[0]].vertices_xy,
        holes_xy=mesh.cells[node-fine[0]].holes_xy,area_m2=mesh.cells[node-fine[0]].area_m2,
        geometry_sha256=digest(dict(vertices=mesh.cells[node-fine[0]].vertices_xy,
        holes=mesh.cells[node-fine[0]].holes_xy,area=mesh.cells[node-fine[0]].area_m2))) for node in sorted(required)]
    mapping=dict(required_leaves=sorted(required),cells=cells,faces=sorted(all_faces,key=lambda r:r['finest_face']),
                 groups=groups,primary_faces=primary,zones=audit)
    return BankSurveyMap(canonical(definition),canonical(mapping))
