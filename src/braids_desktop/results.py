"""Authenticated native-mesh results and a QGIS-free desktop viewer.

Probes sample the stored cell-constant fields. Transects integrate stored
momentum across actual clipped polygons, not a display raster. Left-normal
flow is signed; the UI offers its magnitude explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from io import BytesIO
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
from typing import Callable

import numpy as np
from .presentation import display_number
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree


VARIABLES = {
    'depth_m': ('Water depth', 'm', 'Blues'),
    'velocity_m_s': ('Flow speed', 'm/s', 'Reds'),
    'stage_m': ('Water surface', 'm', 'viridis'),
    'bed_m': ('Bed elevation', 'm', 'BrBG'),
    'bed_change_m': ('Bed change', 'm', 'RdBu_r'),
    'initial_bed_m': ('Initial bed elevation', 'm', 'BrBG'),
    'bedrock_m': ('Non-erodible base', 'm', 'BrBG'),
    'velocity_x_m_s': ('Eastward velocity', 'm/s', 'coolwarm'),
    'velocity_y_m_s': ('Northward velocity', 'm/s', 'coolwarm'),
    'unit_discharge_m2_s': ('Unit discharge magnitude', 'm²/s', 'Purples'),
    'froude': ('Froude number', '—', 'YlOrBr'),
    'bedload_m2_s': ('Bedload transport magnitude','m²/s solid','YlOrBr'),
    'mobile_bedload_concentration': ('Mobile bedload solid fraction','m³/m³','YlOrBr'),
    'secondary_intensity_m_s': ('Secondary-flow intensity','m/s','coolwarm'),
    'bedload_capacity_m2_s': ('Equilibrium bedload capacity','m²/s solid','YlOrBr'),
    'suspended_mass_kg_m2': ('Suspended mass','kg/m²','YlGn'),
    'secondary_hI_m2_s': ('Secondary-flow state','m²/s','coolwarm'),
    'dispersion_flux_m2_s': ('Dispersion-flux magnitude','m²/s','PuBuGn'),
    'bank_erosion_kg_m2': ('Cumulative bank erosion','kg/m²','YlOrBr'),
    'collapse_kg_m2': ('Cumulative collapse','kg/m²','YlOrBr'),
    'bank_failure_count': ('Bank-failure count','—','YlOrBr'),
    'bedform_height_m': ('Bedform height','m','YlOrBr'),
    'erodible_depth_m': ('Erodible bed depth','m','copper'),
    'sediment_stock_kg_m2': ('Bed sediment stock','kg/m²','copper'),
    'suspended_kg_m3': ('Suspended concentration','kg/m³','YlGn'),
}
def variable_info(key):
    from .hub_velocity import PREFIX,info
    if key.startswith(PREFIX):return info(key)
    if key not in VARIABLES:raise ValueError('Unknown result variable')
    return VARIABLES[key]

SEDIMENT_VARIABLES={'bedload_m2_s','erodible_depth_m','sediment_stock_kg_m2','suspended_kg_m3','mobile_bedload_concentration','bedload_capacity_m2_s','suspended_mass_kg_m2'}

# Only these scalar quantities are offered by the map-field selector.  The
# x/y velocity components remain available inside the conserved q array for
# diagnostics and vector operations, but are deliberately not user-facing
# report choices unless a later vector-display tool explicitly asks for them.
CORE_MAP_VARIABLES=frozenset(('depth_m','velocity_m_s','stage_m','bed_m',
                              'unit_discharge_m2_s','froude'))
OUTPUT_MAP_VARIABLES={
    'initial_bed': frozenset(('initial_bed_m','bed_change_m')),
    'bedrock': frozenset(('bedrock_m','erodible_depth_m')),
    'bedload': frozenset(('bedload_m2_s','bedload_capacity_m2_s')),
    'suspended': frozenset(('suspended_mass_kg_m2','suspended_kg_m3')),
    'advanced_hydraulics': frozenset(('secondary_hI_m2_s','secondary_intensity_m_s','dispersion_flux_m2_s')),
    'bank_and_collapse': frozenset(('bank_erosion_kg_m2','collapse_kg_m2','bank_failure_count')),
    'bedforms': frozenset(('bedform_height_m',)),
}


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('Duplicate result metadata field: '+key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _archive(root, name, expected):
    if (not isinstance(name, str) or not name or Path(name).is_absolute() or PureWindowsPath(name).root
            or PureWindowsPath(name).drive or '..' in name.replace('\\', '/').split('/')):
        raise ValueError('Unsafe result artifact path')
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Result artifact escapes the frame folder')
    payload = path.read_bytes()
    if _sha(payload) != expected:
        raise ValueError('Result checksum differs: '+name)
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError('Duplicate result array name')
        result = {key: archive[key] for key in archive.files}
    for array in result.values():
        array.flags.writeable = False
    return result


@dataclass(frozen=True)
class ResultFrame:
    index: int
    time_s: float
    epoch: str
    mesh: dict
    arrays: dict
    polygons: tuple

    @property
    def q(self): return self.arrays['q']

    @property
    def z(self): return self.arrays['z']

    @property
    def water_vector(self):
        # JC evolves mixture discharge; the saved water vector subtracts mobile
        # solid advection AND relative drift. Ordinary routes retain q exactly.
        return self.arrays.get('water_vector_m2_s',self.q[1:])

    @property
    def values(self):
        def cell_total(array):
            """Sum class/component axes while preserving the cell axis."""
            value=np.asarray(array)
            if value.ndim==1:return value
            return value.sum(axis=tuple(range(value.ndim-1)))

        h = self.q[0]
        u = np.divide(self.q[1], h, out=np.zeros_like(h), where=h > 1e-6)
        v = np.divide(self.q[2], h, out=np.zeros_like(h), where=h > 1e-6)
        speed = np.hypot(u, v)
        values=dict(depth_m=h, velocity_m_s=speed, stage_m=h+self.z, bed_m=self.z,
            velocity_x_m_s=u, velocity_y_m_s=v, unit_discharge_m2_s=np.hypot(*self.q[1:]),
            froude=np.divide(speed, np.sqrt(9.81*np.maximum(h, 0.)), out=np.zeros_like(h), where=h > 1e-6))
        if 'initial_bed_m' in self.arrays:values['bed_change_m']=self.z-self.arrays['initial_bed_m']
        if 'bedload_vector_m2_s' in self.arrays:values['bedload_m2_s']=np.linalg.norm(self.arrays['bedload_vector_m2_s'],axis=0)
        for name in ('mobile_bedload_concentration','secondary_intensity_m_s','secondary_hI_m2_s','bedload_capacity_m2_s'):
            if name in self.arrays:values[name]=self.arrays[name]
        # The ordinary secondary-current route stores its conservative state
        # hI as ``W``.  Present both the state and physical intensity without
        # writing a duplicate array solely for visualisation.
        if 'W' in self.arrays:
            values['secondary_hI_m2_s']=self.arrays['W']
            values['secondary_intensity_m_s']=np.divide(self.arrays['W'],h,out=np.zeros_like(h),where=h>1e-6)
        if 'rock' in self.arrays:values['erodible_depth_m']=np.maximum(0.,self.z-self.arrays['rock'])
        if 'initial_bed_m' in self.arrays:values['initial_bed_m']=self.arrays['initial_bed_m']
        if 'rock' in self.arrays:values['bedrock_m']=self.arrays['rock']
        if 'mass' in self.arrays:values['sediment_stock_kg_m2']=self.arrays['mass'].sum(axis=(0,1))
        if 'suspended' in self.arrays:
            suspended=cell_total(self.arrays['suspended'])
            values['suspended_mass_kg_m2']=suspended
            values['suspended_kg_m3']=np.divide(suspended,h,out=np.zeros_like(h),where=h>1e-6)
        if 'bank' in self.arrays:values['bank_erosion_kg_m2']=cell_total(self.arrays['bank'])
        if 'collapse' in self.arrays:values['collapse_kg_m2']=cell_total(self.arrays['collapse'])
        if 'bank_failure_count' in self.arrays:values['bank_failure_count']=cell_total(self.arrays['bank_failure_count'])
        if 'bedforms' in self.arrays:values['bedform_height_m']=self.arrays['bedforms'][0]
        if 'dispersion_flux_m2_s' in self.arrays:values['dispersion_flux_m2_s']=np.linalg.norm(self.arrays['dispersion_flux_m2_s'],axis=0)
        return values


@dataclass(frozen=True)
class Series:
    time_s: np.ndarray
    values: np.ndarray
    label: str
    unit: str
    method: str

    def export_csv(self, path):
        path = Path(path)
        with path.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time_s', f'{self.label} [{self.unit}]'])
            writer.writerows(zip(self.time_s, self.values))
        return path


@dataclass(frozen=True)
class SpatialProfile:
    distance_m: np.ndarray
    values: np.ndarray
    xy: np.ndarray
    time_s: float
    variable: str
    label: str
    unit: str
    sampling: str='raw_cell_intersections'

    def export_csv(self,path):
        with Path(path).open('x',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(['distance_m','easting_m','northing_m',self.variable,'time_s','sampling'])
            for s,xy,value in zip(self.distance_m,self.xy,self.values):writer.writerow([s,*xy,value,self.time_s,self.sampling])


class ResultsStore:
    """Immutable published artifacts; mutable live INDEX is refreshed explicitly."""
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.root = self.output/'frames'
        self.context = _read_json(self.output/'RUN_CONTEXT.json')
        if self.context.get('schema') != 'braids.river-results.v1':
            raise ValueError('Choose results from the current BRAIDS river solver')
        self.project = None
        if 'project_file' in self.context:
            if self.context['project_file'] != 'PROJECT.json':
                raise ValueError('Unexpected saved project path')
            raw = (self.output/'PROJECT.json').read_bytes()
            if _sha(raw) != self.context.get('project_sha256'):
                raise ValueError('Saved run parameters are missing or changed')
            self.project = json.loads(raw)
            canonical = json.dumps(self.project, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            if self.context.get('project_id') and _sha(canonical) != self.context['project_id']:
                raise ValueError('Saved run parameter identity differs')
            if 'model' in self.context and self.project.get('model') != self.context['model']:
                raise ValueError('Saved model differs from results')
        self.crs = self.context.get('crs', '')
        self._polygons = {}; self._trees = {}; self._weights = {}; self._initial_bed = {}; self._boundary_masks = {}
        self.index = {'frames': [], 'epochs': {}}
        self.refresh()

    @property
    def times(self): return np.array([row['time_s'] for row in self.index['frames']], dtype=float)

    @property
    def mesh_history(self):
        return tuple(dict(epoch=name, first_time_s=next(r['time_s'] for r in self.index['frames'] if r['epoch'] == name))
            for name in dict.fromkeys(r['epoch'] for r in self.index['frames']))

    @property
    def boundary_ids(self):
        """Only IDs actually assigned in the saved run, never editor history."""
        settings = self.project.get('settings', {}) if self.project else {}
        return tuple(sorted({a['bc_id'] for a in settings.get('assignments', []) if a.get('kind') in ('inlet', 'outlet')}))

    def _boundary_mask(self, frame, bc_id):
        key = (frame.epoch, bc_id)
        if key in self._boundary_masks: return self._boundary_masks[key]
        roots = [self.output/'cache']
        prepared = self.context.get('input')
        if prepared:
            prepared = Path(prepared)
            if not prepared.is_absolute():
                if self.context.get('input_path_base') != 'run_directory':
                    raise ValueError('Saved prepared-input path has no explicit base')
                prepared = self.output/prepared
            roots.extend((prepared/'cache', prepared/'core'/'prepared'/'cache'))
        for root in roots:
            for path in sorted(root.glob('*/CACHE.json')):
                receipt = _read_json(path); identity = receipt.get('identity', {})
                if identity.get('leaves') != frame.mesh['leaves'].tolist(): continue
                vertices = identity.get('polygons', [])
                if len(vertices) != len(frame.polygons): continue
                offsets = frame.mesh['offsets']
                if any(not np.array_equal(np.asarray(poly), frame.mesh['vertices'][offsets[i]:offsets[i+1]]) for i, poly in enumerate(vertices)):
                    continue
                # Bind the face-ID order to the very same retained mesh, then
                # authenticate both records and their numeric-ID mapping.
                files = receipt.get('files', {})
                if not {'records.npz', 'case.data.json'} <= files.keys(): continue
                records = _archive(path.parent, 'records.npz', files['records.npz'])
                raw = (path.parent/'case.data.json').read_bytes()
                if _sha(raw) != files['case.data.json']: raise ValueError('Saved boundary-ID mapping checksum differs')
                case = json.loads(raw); mapping = case.get('string_to_numeric_id', {})
                if bc_id not in mapping: continue
                bid, neighbour = records.get('bid'), records.get('neighbour')
                if bid is None or neighbour is None or bid.ndim != 1 or neighbour.shape != bid.shape:
                    raise ValueError('Saved boundary-face records differ')
                mask = (bid == int(mapping[bc_id])-1) & (neighbour < 0)
                if not mask.any(): raise ValueError('No retained faces for saved boundary '+bc_id)
                mask.flags.writeable = False; self._boundary_masks[key] = mask
                return mask
        raise ValueError('Boundary flow needs the retained prepared-input cache matching this mesh. Open the complete project folder, not a copied run alone.')

    def boundary_flow(self, bc_id, *, quantity='water', magnitude=True, progress=None, cancelled=None):
        """Accepted native ledger increments / output interval, not a probe."""
        if bc_id not in self.boundary_ids: raise ValueError('Boundary is not assigned in the loaded run')
        fields = {'water': ('water', 'Water flow', 'm³/s'),
                  'bedload': ('bed_boundary', 'Bedload flow', 'kg/s'),
                  'suspended': ('sus_boundary', 'Suspended flow', 'kg/s')}
        if quantity not in fields: raise ValueError('Unknown boundary-flow quantity')
        field, label, unit = fields[quantity]; cumulative = []
        # A progress callback can process UI events and refresh the live index.
        # Keep one immutable time axis for the whole measurement.
        times = self.times
        for i in range(len(times)):
            if cancelled and cancelled(): raise InterruptedError('Boundary flow cancelled')
            frame = self.read(i); mask = self._boundary_mask(frame, bc_id)
            ledger = frame.arrays.get(field)
            if ledger is None or ledger.shape[-1:] != mask.shape or not np.isfinite(ledger).all():
                raise ValueError('Accepted boundary ledger is unavailable or does not match the saved faces')
            cumulative.append(float(np.sum(ledger[..., mask])))
            if progress: progress((i+1)/len(times))
        rate = np.diff(cumulative)/np.diff(times)
        if magnitude: rate = np.abs(rate)
        return Series((times[1:]+times[:-1])/2, rate, f'{label}: {bc_id}', unit,
            'Accepted numerical boundary-ledger increment / output interval; plotted at interval midpoint. '+
            ('Nonnegative net magnitude; boundary role identifies inlet/outlet.' if magnitude else 'Positive into the domain.'))

    def refresh(self):
        doc = _read_json(self.root/'INDEX.json')
        if not isinstance(doc, dict) or not isinstance(doc.get('frames'), list) or not isinstance(doc.get('epochs'), dict):
            raise ValueError('Invalid result publication index')
        previous = -math.inf
        for row in doc['frames']:
            if not isinstance(row, dict) or not {'file', 'epoch', 'sha256', 'time_s'} <= row.keys():
                raise ValueError('Invalid result frame metadata')
            t = row['time_s']
            if not isinstance(t, (float, int)) or not math.isfinite(t) or t <= previous:
                raise ValueError('Result times must be finite and strictly increasing')
            if row['epoch'] not in doc['epochs']:
                raise ValueError('Result references an unknown mesh')
            previous = t
        old = self.index['frames']
        if doc['frames'][:len(old)] != old:
            raise ValueError('Previously published result history changed')
        for name, digest in self.index['epochs'].items():
            if doc['epochs'].get(name) != digest:
                raise ValueError('Previously published mesh history changed')
        changed = len(doc['frames']) != len(old)
        self.index = doc
        return changed

    def _geometry(self, name, mesh):
        if name in self._polygons:
            return self._polygons[name]
        vertices, offsets, leaves = (mesh[k] for k in ('vertices', 'offsets', 'leaves'))
        n = len(leaves)
        if (leaves.ndim != 1 or leaves.dtype.kind not in 'iu' or len(set(leaves)) != n
                or vertices.ndim != 2 or vertices.shape[1] != 2 or not np.isfinite(vertices).all()
                or offsets.dtype.kind not in 'iu' or offsets.shape != (n+1,)
                or offsets[0] != 0 or offsets[-1] != len(vertices) or np.any(np.diff(offsets) < 3)):
            raise ValueError('Invalid saved computational mesh')
        if 'area' in mesh and (mesh['area'].shape != (n,) or not np.isfinite(mesh['area']).all() or np.any(mesh['area'] <= 0)):
            raise ValueError('Invalid saved cell areas')
        domain = None
        if self.project:
            data = self.project.get('settings', {}).get('domain', {})
            if data.get('holes_xy'):
                domain = Polygon(data['vertices_xy'], data['holes_xy'])
        result = []
        for i in range(n):
            poly = Polygon(vertices[offsets[i]:offsets[i+1]])
            if domain is not None: poly = poly.intersection(domain)
            if poly.is_empty or not poly.is_valid or poly.geom_type not in ('Polygon', 'MultiPolygon'):
                raise ValueError('Invalid retained cell polygon')
            if 'area' in mesh and not math.isclose(poly.area, float(mesh['area'][i]), rel_tol=1e-8, abs_tol=1e-7):
                raise ValueError('Saved polygon does not match cell area; missing cut-cell/hole geometry')
            result.append(poly)
        self._polygons[name] = tuple(result)
        self._trees[name] = STRtree(result)
        return self._polygons[name]

    def read(self, index=-1):
        rows = self.index['frames']
        if not rows: raise ValueError('No published result frames yet')
        index = int(index)
        if index < 0: index += len(rows)
        if not 0 <= index < len(rows): raise IndexError('Result frame is outside the published history')
        row = rows[index]
        arrays = _archive(self.root, row['file'], row['sha256'])
        mesh = _archive(self.root, row['epoch'], self.index['epochs'][row['epoch']])
        if not {'vertices', 'offsets', 'leaves'} <= mesh.keys() or not {'q', 'z', 'mesh_epoch'} <= arrays.keys():
            raise ValueError('Result is missing required arrays')
        n = len(mesh['leaves'])
        if (arrays['q'].shape != (3, n) or arrays['z'].shape != (n,)
                or arrays['mesh_epoch'].shape != () or str(arrays['mesh_epoch']) != row['epoch']
                or arrays['q'].dtype.kind != 'f' or arrays['z'].dtype.kind != 'f'
                or not np.isfinite(arrays['q']).all() or not np.isfinite(arrays['z']).all()
                or np.any(arrays['q'][0] < 0)):
            raise ValueError('Frame does not match its mesh or has invalid hydraulic values')
        polygons = self._geometry(row['epoch'], mesh)
        return ResultFrame(index, float(row['time_s']), row['epoch'], mesh, arrays, polygons)

    @property
    def hub_variables(self):
        from .hub_velocity import variables
        return variables(self.project)

    def hub_field(self,frame,key):
        from .hub_velocity import reconstruct
        cache=getattr(self,'_hub_fields',{});token=(frame.index,frame.epoch,key)
        if token not in cache:
            if len(cache)>=6:cache.clear()
            cache[token]=reconstruct(self,frame,key);self._hub_fields=cache
        return cache[token]

    def values(self, frame, variable):
        if variable.startswith('hub_velocity_m_s:'):return self.hub_field(frame,variable)['speed_m_s']
        if variable != 'bed_change_m':
            variable_info(variable)
            if variable not in frame.values:raise ValueError('This result does not contain '+variable_info(variable)[0]+'. Run with the current output format; missing fields are not reconstructed as zeros.')
            return frame.values[variable]
        if 'initial_bed_m' in frame.arrays:
            baseline=frame.arrays['initial_bed_m']
            change=frame.z-baseline
            # Bed and reference follow different floating-point accumulation
            # paths. Do not magnify cancellation noise into a coloured map.
            tolerance=8*np.finfo(float).eps*np.maximum(1.,np.maximum(np.abs(frame.z),np.abs(baseline)))
            return np.where(np.abs(change)<=tolerance,0.,change)
        first = self.read(0)
        key = frame.epoch
        if key not in self._initial_bed:
            if key == first.epoch:
                baseline = first.z.copy()
            else:
                tree = self._trees[first.epoch]; baseline = np.empty(len(frame.polygons))
                for i, poly in enumerate(frame.polygons):
                    area = 0.; integral = 0.
                    for j in tree.query(poly):
                        overlap = poly.intersection(first.polygons[int(j)]).area
                        area += overlap; integral += overlap*first.z[int(j)]
                    if not math.isclose(area, poly.area, rel_tol=1e-7, abs_tol=1e-7):
                        raise ValueError('Initial mesh does not cover the current result cell')
                    baseline[i] = integral/area
            self._initial_bed[key] = baseline
        baseline=self._initial_bed[key];change=frame.z-baseline
        tolerance=8*np.finfo(float).eps*np.maximum(1.,np.maximum(np.abs(frame.z),np.abs(baseline)))
        return np.where(np.abs(change)<=tolerance,0.,change)

    def _owner(self, frame, point):
        candidates = [int(i) for i in self._trees[frame.epoch].query(point) if frame.polygons[int(i)].covers(point)]
        if not candidates: return None
        # An exactly shared edge has no unique piecewise-constant trace.
        return min(candidates, key=lambda i: int(frame.mesh['leaves'][i]))

    def probe(self, point, variable='depth_m', *, progress=None, cancelled=None):
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all(): raise ValueError('A finite probe coordinate pair is required')
        variable_info(variable)
        point = Point(*point); values = []; times = self.times
        for i in range(len(times)):
            if cancelled and cancelled(): raise InterruptedError('Probe cancelled')
            frame = self.read(i); cell = self._owner(frame, point)
            values.append(np.nan if cell is None else self.values(frame, variable)[cell])
            if progress: progress((i+1)/len(times))
        label, unit, _ = variable_info(variable)
        return Series(times, np.asarray(values), label, unit,
            'Stored cell-constant value at probe; outside domain is NaN; shared-edge tie uses lowest leaf ID')

    def spatial_profile(self,points,variable='depth_m',index=-1,*,interpolate=False):
        """Cell intersections, optionally joined through interval-centre values.

        Linear display interpolation cannot overshoot cell extrema and never
        bridges an exterior/solid gap. Raw steps remain available for auditing.
        """
        points=np.asarray(points,float)
        if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all():raise ValueError('A finite polyline with at least two points is required')
        if np.linalg.norm(np.diff(points,axis=0),axis=1).sum()<=0:raise ValueError('Profile must have positive length')
        frame=self.read(index);values=self.values(frame,variable)
        if not hasattr(self,'_profile_intersections'):self._profile_intersections={}
        key=(frame.epoch,tuple(map(tuple,points)))
        if key not in self._profile_intersections:
            distance=[];coordinates=[];cells=[];offset=0.
            for a,b in zip(points,points[1:]):
                direction=b-a;length=float(np.linalg.norm(direction))
                if length==0:continue
                line=LineString([a,b]);breaks=[0.,1.]
                for i in self._trees[frame.epoch].query(line):
                    intersection=line.intersection(frame.polygons[int(i)])
                    for part in intersection.geoms if hasattr(intersection,'geoms') else (intersection,):
                        if part.geom_type!='LineString' or part.length==0:continue
                        for xy in (part.coords[0],part.coords[-1]):breaks.append(float(np.clip(np.dot(np.asarray(xy)-a,direction)/(length*length),0.,1.)))
                breaks=np.unique(breaks)
                for lo,hi in zip(breaks,breaks[1:]):
                    if hi-lo<=1e-13:continue
                    owner=self._owner(frame,Point(*(a+.5*(lo+hi)*direction)))
                    distance.extend((offset+lo*length,offset+hi*length));coordinates.extend((a+lo*direction,a+hi*direction));cells.extend((-1 if owner is None else owner,)*2)
                offset+=length
            self._profile_intersections[key]=(np.asarray(distance),np.asarray(coordinates),np.asarray(cells,int))
        distance,xy,cells=self._profile_intersections[key];result=np.full(len(cells),np.nan);inside=cells>=0;result[inside]=values[cells[inside]]
        label,unit,_=variable_info(variable)
        if interpolate and len(result):
            from .profile_interpolation import interpolate_intervals
            distance,result,xy=interpolate_intervals(distance,result,xy)
        return SpatialProfile(distance.copy(),result,xy.copy(),frame.time_s,variable,label,unit,
            'linear_cell_interval_centres' if interpolate else 'raw_cell_intersections')

    def _transect_weights(self, frame, points):
        key = (frame.epoch, tuple(map(tuple, points)))
        if key in self._weights: return self._weights[key]
        weights = np.zeros((2, len(frame.polygons))); tree = self._trees[frame.epoch]
        for a, b in zip(points, points[1:]):
            direction = b-a; length = float(np.linalg.norm(direction))
            if length == 0: continue
            line = LineString([a, b]); breaks = [0., 1.]
            for i in tree.query(line):
                intersection = line.intersection(frame.polygons[int(i)])
                pieces = list(intersection.geoms) if hasattr(intersection, 'geoms') else [intersection]
                for part in pieces:
                    if part.geom_type != 'LineString' or part.length == 0: continue
                    for xy in (part.coords[0], part.coords[-1]):
                        breaks.append(float(np.clip(np.dot(np.asarray(xy)-a, direction)/(length*length), 0., 1.)))
            breaks = np.unique(breaks)
            normal = np.array([-direction[1], direction[0]])/length
            for lo, hi in zip(breaks, breaks[1:]):
                if hi-lo <= 1e-13: continue
                owner = self._owner(frame, Point(*(a+.5*(lo+hi)*direction)))
                if owner is not None: weights[:, owner] += normal*(hi-lo)*length
        self._weights[key] = weights
        return weights

    def measurement_table(self,points,*,profile=False,magnitude=True,progress=None,cancelled=None):
        """One read per frame; all displayed variables plus profile discharge."""
        points=np.asarray(points,float)
        if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():raise ValueError('Finite measurement coordinates required')
        if (not profile and len(points)!=1) or (profile and (len(points)<2 or np.sum(np.linalg.norm(np.diff(points,axis=0),axis=1))<=0)):
            raise ValueError('One probe point or a positive-length profile is required')
        first=self.read(0);available=first.values
        variables=[key for key in VARIABLES if key=='bed_change_m' or key in available]+self.hub_variables
        bedload=profile and 'bedload_vector_m2_s' in first.arrays
        physics=(self.project or {}).get('model',{}).get('physics',{}).get('sediment') or {}
        suspension=profile and physics.get('suspension','off')!='off' and 'suspended' in first.arrays
        columns=['time_s',*variables]
        if profile:columns.append('flow_m3_s')
        if bedload:columns.append('bedload_solid_m3_s')
        if suspension:columns.append('suspended_advective_kg_s')
        rows=[];times=self.times
        for i,time in enumerate(times):
            if cancelled and cancelled():raise InterruptedError('CSV export cancelled')
            frame=self.read(i);row=[time]
            if profile:
                weights=self._transect_weights(frame,points)
                lengths=sum((np.linalg.norm(self._transect_weights(frame,np.array([a,b])),axis=0) for a,b in zip(points,points[1:])),np.zeros(len(frame.polygons)))
                for variable in variables:
                    values=self.values(frame,variable);valid=np.isfinite(values);total=float(lengths[valid].sum())
                    row.append(float(np.dot(lengths[valid],values[valid])/total) if total>0 else np.nan)
                flow=float(np.sum(weights*frame.water_vector));row.append(abs(flow) if magnitude else flow)
                if bedload:
                    flow=float(np.sum(weights*frame.arrays['bedload_vector_m2_s']));row.append(abs(flow) if magnitude else flow)
                if suspension:
                    flow=float(np.sum(weights*frame.q[1:]*frame.values['suspended_kg_m3']));row.append(abs(flow) if magnitude else flow)
            else:
                owner=self._owner(frame,Point(*points[0]))
                row.extend(np.nan if owner is None else self.values(frame,v)[owner] for v in variables)
            rows.append(row)
            if progress:progress((i+1)/len(times))
        return columns,rows

    def transect(self, points, *, magnitude=False, progress=None, cancelled=None,quantity='water'):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError('A finite polyline with at least two points is required')
        if np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) <= 0:
            raise ValueError('Transect has zero length')
        result = []; times = self.times
        for i in range(len(times)):
            if cancelled and cancelled(): raise InterruptedError('Transect cancelled')
            frame = self.read(i);weights=self._transect_weights(frame,points)
            if quantity=='water':vector=frame.water_vector
            elif quantity in ('bedload','total_solid'):
                if 'bedload_vector_m2_s' not in frame.arrays:raise ValueError('Bedload vector was not stored in this result')
                vector=frame.arrays['bedload_vector_m2_s'].copy()
                if quantity=='total_solid':
                    rho=np.asarray(self.project['settings']['columns']['solid_density_kg_m3'])
                    volume=(frame.arrays['suspended']/rho[:,None]).sum(0)
                    vector+=np.divide(frame.q[1:],frame.q[:1],out=np.zeros_like(vector),where=frame.q[:1]>1e-6)*volume
            elif quantity=='suspended':vector=frame.q[1:]*frame.values['suspended_kg_m3']
            else:
                values=self.values(frame,quantity)
                lengths=sum((np.linalg.norm(self._transect_weights(frame,np.array([a,b])),axis=0) for a,b in zip(points,points[1:])),np.zeros(len(frame.polygons)))
                flow=float(np.dot(lengths,values)/lengths.sum()) if lengths.sum()>0 else np.nan
                result.append(flow)
                if progress:progress((i+1)/len(times))
                continue
            flow = float(np.sum(weights*vector))
            result.append(abs(flow) if magnitude else flow)
            if progress: progress((i+1)/len(times))
        names={'water':('Water flow','m³/s'),'bedload':('Bedload flow','m³/s solid'),
               'suspended':('Suspended advective flow','kg/s'),'total_solid':('Total sediment advective + bedload flow','m³/s solid')}
        label,unit=names[quantity] if quantity in names else variable_info(quantity)[:2]
        method=('Line integral of stored cell vectors; bedload is reconstructed from the last accepted limited face flux, without the morphological multiplier. Not an exact boundary ledger; suspended diffusion is not included.' if quantity in names else 'Length-weighted mean of stored cell values along the profile.')
        if quantity in names and 'mobile_bedload_solid_m' in self.read(0).arrays:
            method='Line integral of instantaneous cell phase vectors: mobile-solid advection plus relative drift, water = mixture minus solid. Not an exact accepted boundary ledger.'
        return Series(times,np.asarray(result),label,unit,method)

    def payload(self, index=-1, variable='depth_m', *, mesh_mode='none', limits=None,palette=None,
                zero_transparent=True):
        frame = self.read(index); values = self.values(frame, variable)
        display=self.display_values(frame,variable,values=values,zero_transparent=zero_transparent)
        rgba, limits = colour_values(display, variable, limits=limits,palette=palette,
                                     zero_transparent=zero_transparent)
        if mesh_mode not in ('none', 'initial', 'adaptive'): raise ValueError('Unknown mesh display mode')
        if mesh_mode=='initial':
            mesh_polygons=frame.polygons if self.index['frames'][0]['epoch']==frame.epoch else self.read(0).polygons
        else:mesh_polygons=frame.polygons if mesh_mode=='adaptive' else ()
        label, unit, default_palette = variable_info(variable);palette=palette or default_palette
        if variable.startswith('hub_velocity_m_s:'):label+=f"; z={self.hub_field(frame,variable)['elevation_m']:.3f} m; near wake masked"
        return dict(frame=frame, variable=variable, values=values, rgba=rgba, limits=limits, palette=palette,
            label=f'{label} [{unit}]', time_s=frame.time_s, polygons=frame.polygons, mesh_polygons=mesh_polygons)

    def display_values(self,frame,variable,*,values=None,zero_transparent=True):
        """Exclude dry terrain from hydraulic colors AND their legend range."""
        values=(self.values(frame,variable) if values is None else values).copy()
        if zero_transparent and variable not in ('bed_m','bed_change_m','initial_bed_m','bedrock_m','erodible_depth_m','sediment_stock_kg_m2'):values[frame.q[0]<=1e-6]=np.nan
        return values


def colour_values(values, variable, *, limits=None,palette=None,zero_transparent=True):
    from matplotlib import colormaps
    values = np.asarray(values, dtype=float); valid = np.isfinite(values)
    variable_info(variable)
    if limits is None:
        lo = .001 if variable == 'depth_m' else 0. if variable in ('velocity_m_s', 'unit_discharge_m2_s', 'froude') else float(np.min(values[valid], initial=np.inf))
        hi = float(np.max(values[valid], initial=-np.inf))
        if not math.isfinite(lo): lo = 0.
        if not math.isfinite(hi): hi = lo
        if variable == 'bed_change_m': lo, hi = -max(abs(lo), abs(hi), 1e-9), max(abs(lo), abs(hi), 1e-9)
        limits = (lo, max(hi, lo+1e-9))
    lo, hi = map(float, limits)
    if not math.isfinite(lo+hi) or hi <= lo: raise ValueError('Color limits must be finite and increasing')
    rgba = colormaps[palette or variable_info(variable)[2]](np.clip(np.nan_to_num((values-lo)/(hi-lo)), 0, 1), bytes=True)
    rgba[~valid, 3] = 0
    if zero_transparent: rgba[values == 0., 3] = 0
    return rgba, (lo, hi)


def export_png_sequence(store, directory, *, variable='depth_m', indices=None, mesh_mode='none', render=None,
                        progress=None, cancelled=None):
    """Export numbered PNGs; optional render(payload)->QImage preserves the map.

    Limits are frozen for the sequence. Existing target files are never replaced.
    A cancelled export leaves completed PNGs and an explicit incomplete manifest.
    """
    indices = list(range(len(store.times))) if indices is None else list(indices)
    if not indices: raise ValueError('No frames selected')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    targets = [directory/f'frame_{i:06d}.png' for i in range(len(indices))]
    manifest_path = directory/'ANIMATION.json'
    if manifest_path.exists() or any(path.exists() for path in targets): raise FileExistsError('Animation export already exists')
    low, high = math.inf, -math.inf
    for index in indices:
        if cancelled and cancelled(): raise InterruptedError('Animation export cancelled')
        frame = store.read(index); values = store.values(frame, variable); finite = values[np.isfinite(values)]
        if len(finite): low = min(low, float(finite.min())); high = max(high, float(finite.max()))
    _, limits = colour_values(np.array([low, high]), variable)
    manifest = dict(variable=variable, source_run=str(store.output), limits=limits, complete=False, frames=[])
    try:
        for number, (index, target) in enumerate(zip(indices, targets)):
            if cancelled and cancelled(): raise InterruptedError('Animation export cancelled')
            payload = store.payload(index, variable, mesh_mode=mesh_mode, limits=limits)
            if render is not None:
                image = render(payload)
                if not image.save(str(target), 'PNG'): raise OSError('Could not save PNG '+str(target))
            else:
                _render_png(payload, target)
            manifest['frames'].append(dict(file=target.name, time_s=payload['time_s']))
            if progress: progress((number+1)/len(indices))
        manifest['complete'] = True
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return targets


def _render_png(payload, target):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.path import Path as PlotPath
    from matplotlib.patches import PathPatch
    from shapely.geometry.polygon import orient
    fig = Figure(figsize=(9, 7), constrained_layout=True); FigureCanvasAgg(fig); ax = fig.subplots()
    def draw(poly, color, edge='none', width=0):
        pieces = poly.geoms if poly.geom_type == 'MultiPolygon' else (poly,)
        for piece in pieces:
            piece = orient(piece, 1.); vertices = []; codes = []
            for ring in (piece.exterior, *piece.interiors):
                xy = np.asarray(ring.coords); vertices.extend(xy)
                codes.extend([PlotPath.MOVETO]+[PlotPath.LINETO]*(len(xy)-2)+[PlotPath.CLOSEPOLY])
            ax.add_patch(PathPatch(PlotPath(vertices, codes), facecolor=color, edgecolor=edge, linewidth=width))
    for poly, color in zip(payload['polygons'], payload['rgba']/255.): draw(poly, color)
    for poly in payload['mesh_polygons']: draw(poly, 'none', '#555555', .25)
    ax.autoscale_view(); ax.set_aspect('equal'); ax.set_title(f"{payload['label']} · t = {payload['time_s']:g} s")
    ax.set_xlabel('Easting [m]'); ax.set_ylabel('Northing [m]')
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    fig.colorbar(ScalarMappable(Normalize(*payload['limits']), cmap=payload['palette']), ax=ax, label=payload['label'])
    fig.savefig(target, dpi=150)


try:
    from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
    from PySide6.QtGui import QColor, QPainterPath, QPen, QPainter
    from PySide6.QtWidgets import (QWidget, QGraphicsItem, QVBoxLayout, QHBoxLayout, QPushButton,
        QLabel, QSlider, QComboBox, QCheckBox, QGroupBox, QFileDialog, QDialog, QProgressDialog, QApplication,
        QPlainTextEdit, QDialogButtonBox, QMessageBox, QLayout)
except ImportError:
    QWidget = None


if QWidget is not None:
    class ResultPolygonItem(QGraphicsItem):
        """One Qt item, native polygons, no interpolation/display grid."""
        def __init__(self, to_scene=None, *, outline=False):
            super().__init__(); self.to_scene = to_scene or (lambda x, y: QPointF(x, -y))
            self.outline = outline; self.paths = []; self.colors = []; self._bounds = QRectF(); self._key = None; self._polygons = None; self._outline_path=QPainterPath()
            self.setZValue(21 if outline else 20)

        def set_data(self, polygons, rgba=None):
            # Keep a strong reference and include the map origin so reloading
            # terrain cannot reuse paths belonging to the old scene transform.
            origin = self.to_scene(0., 0.)
            key = (id(polygons), origin.x(), origin.y())
            if key != self._key:
                self.prepareGeometryChange(); self.paths = []; bounds = QRectF();self._outline_path=QPainterPath()
                for poly in polygons:
                    path = QPainterPath(); path.setFillRule(Qt.FillRule.OddEvenFill)
                    for piece in (poly.geoms if poly.geom_type == 'MultiPolygon' else (poly,)):
                        for ring in (piece.exterior, *piece.interiors):
                            points = list(ring.coords); path.moveTo(self.to_scene(*points[0]))
                            for point in points[1:]: path.lineTo(self.to_scene(*point))
                            path.closeSubpath()
                    self.paths.append(path); bounds = bounds.united(path.boundingRect())
                    if self.outline:self._outline_path.addPath(path)
                self._bounds = bounds; self._key = key; self._polygons = polygons
            self.colors = [] if rgba is None else [QColor(*map(int, row)) for row in rgba]
            self.setVisible(bool(self.paths)); self.update()

        def boundingRect(self): return self._bounds

        def paint(self, painter, option, widget=None):
            if self.outline:
                from ._avac.preferences import mesh_color
                pen = QPen(QColor(mesh_color()), .65); pen.setCosmetic(True); painter.setPen(pen); painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(self._outline_path)
            else:
                painter.save(); painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
                if getattr(self,'domain_clip',None) is not None:painter.setClipPath(self.domain_clip,Qt.IntersectClip)
                painter.setPen(Qt.PenStyle.NoPen)
                for path, color in zip(self.paths, self.colors):
                    if color.alpha(): painter.setBrush(color); painter.drawPath(path)
                painter.restore()


    class ResultsPanel(QWidget):
        frame_changed = Signal(object)
        probe_requested = Signal()
        transect_requested = Signal()
        measurements_changed = Signal(object)
        geometry_edit_started = Signal()
        error = Signal(str)

        def __init__(self, map_view=None, parent=None):
            super().__init__(parent); self.map_view = map_view; self.store = None; self.probe_point = None; self.transect_points = None
            self._turbine_snapshot=None;self._live_output=None
            self._field_item = None; self._mesh_item = None; self._terrain_item = None; self._last_series = None; self._dialogs = [];self._smooth_item=None;self._smooth_cache=OrderedDict()
            from ._avac.preferences import settings
            from ._avac.preferences import smooth_level
            self.smooth=settings().value('map/smooth_results',True,type=bool);self.smooth_level=smooth_level();self.terrain_aware_wetting=settings().value('map/terrain_aware_wetting',False,type=bool);self._results_active=True;self._particles_enabled=False;self.particles=None
            if map_view is not None and hasattr(map_view,'terrain') and callable(getattr(map_view,'to_scene',None)):
                from .visual_fields import ParticleTracks
                self.particles=ParticleTracks(map_view)
            from PySide6.QtWidgets import QStackedWidget
            root=QVBoxLayout(self);self.page_stack=QStackedWidget();root.addWidget(self.page_stack);self.pages={};self.page_layouts={}
            self.probes=[];self.profiles=[];self._selected_probe=None;self._selected_profile=None;self._active_kind='gauge'
            self.measurements_visible={'gauge':True,'profile':True}
            self.extrema_enabled={'minimum':False,'maximum':False};self._extrema_items=[];self._last_payload=None
            self.project_name='project';self._capture_new=True
            for name in ('Probes','Profiles','Boundary flow','Turbines'):
                page=QWidget();self.pages[name]=page;self.page_layouts[name]=QVBoxLayout(page);self.page_stack.addWidget(page)
            self.map_controls=QWidget();controls=QHBoxLayout(self.map_controls);controls.setContentsMargins(0,0,0,0)
            self.variable = QComboBox()
            self.variable.setMinimumContentsLength(8);self.variable.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            self.set_sediment_available(False)
            self.variable.setMaximumWidth(180);controls.addWidget(self.variable)
            self.previous=QPushButton('◀');self.previous.setToolTip('Previous frame');self.previous.clicked.connect(lambda:self.step_frame(-1))
            self.play = QPushButton('▶');self.play.setToolTip('Play');self.play.clicked.connect(self.toggle_animation)
            self.next=QPushButton('▶');self.next.setToolTip('Next frame');self.next.clicked.connect(lambda:self.step_frame(1))
            for button in (self.previous,self.play,self.next):button.setMaximumWidth(55);controls.addWidget(button)
            self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setRange(0, 0);self.slider.setMinimumWidth(70);controls.addWidget(self.slider,1)
            self.caption = QLabel('No results loaded');self.caption.hide()
            self.mesh = QCheckBox(self);self.mesh.setChecked(True);self.mesh.hide()
            self.animate_mesh = QCheckBox(self);self.animate_mesh.hide()
            from ._avac.preferences import adaptive_mesh
            self.animate_mesh.setChecked(adaptive_mesh())
            self.export = QPushButton('Export');self.export.setToolTip('Export a PNG frame or MP4 video');controls.addWidget(self.export);self.export.clicked.connect(self.export_animation)
            self.timer = QTimer(self); self.timer.setInterval(250); self.timer.timeout.connect(self._next_frame)
            self.probe_list=QComboBox();self.profile_list=QComboBox()
            self.profile_quantity=QComboBox();self.profile_quantity.addItem('Water flow [m³/s]','water')
            self.profile_mode=QComboBox();self.profile_mode.addItem('Spatial profile of displayed map','spatial');self.profile_mode.addItem('Flow / mean time series','time')
            self.profile_animate=QCheckBox('Follow map frame / animation');self.profile_animate.setChecked(True)
            for title,listing,show,edit in (('Probe',self.probe_list,self.plot_probe,self.edit_probe),('Profile',self.profile_list,self.plot_transect,self.edit_transect)):
                kind='gauge' if title=='Probe' else 'profile';form=self.page_layouts['Probes' if kind=='gauge' else 'Profiles']
                form.addWidget(listing);listing.currentIndexChanged.connect(lambda _,k=kind:self._select_measurement(k))
                if kind=='profile':
                    form.addWidget(self.profile_mode);form.addWidget(self.profile_quantity);form.addWidget(self.profile_animate)
                row=QHBoxLayout();button=QPushButton('New '+title.lower());button.clicked.connect(lambda _,k=kind:self.new_measurement(k));row.addWidget(button)
                edit_button=QPushButton('Edit '+title.lower());edit_button.clicked.connect(edit);row.addWidget(edit_button);form.addLayout(row)
                delete=QPushButton('Delete '+title.lower());delete.clicked.connect(lambda _,k=kind:self.delete_measurement(k))
                plot=QPushButton('Plot '+title.lower());plot.clicked.connect(show);form.addWidget(plot)
                csv_button=QPushButton('Export all variables to CSV');csv_button.clicked.connect(lambda _,k=kind:self.export_measurement(k));form.addWidget(csv_button)
                if kind=='profile':self.profile_plot=plot;self.profile_csv=csv_button
                form.addWidget(delete)
                note=QLabel('Select or draw a line. Spatial profiles plot the current map field against distance, with gaps outside the mesh and through solids. Flow / mean mode plots the history across the line.' if kind=='profile' else 'Place a probe to plot the selected map field over time.');note.setWordWrap(True);form.addWidget(note)
            self.profile_mode.currentIndexChanged.connect(self._profile_mode_changed);self._profile_mode_changed()
            boundary = QGroupBox('Boundary flow'); bform = QVBoxLayout(boundary); self.page_layouts['Boundary flow'].addWidget(boundary)
            self.boundary = QComboBox(); bform.addWidget(self.boundary)
            self.boundary_quantity = QComboBox()
            for key, name in (('water', 'Water'), ('bedload', 'Bedload'), ('suspended', 'Suspended sediment')): self.boundary_quantity.addItem(name, key)
            bform.addWidget(self.boundary_quantity)
            boundary_plot = QPushButton('Plot accepted boundary flow'); boundary_plot.clicked.connect(self.plot_boundary); bform.addWidget(boundary_plot)
            bnote = QLabel('Positive magnitudes from accepted numerical fluxes, averaged between saved frames. Requires the retained prepared cache.'); bnote.setWordWrap(True); bform.addWidget(bnote)
            tform=self.page_layouts['Turbines'];self.power_estimate_button=QPushButton('Calculate / refresh power estimates');self.power_estimate_button.clicked.connect(self.calculate_power_plan);self.power_estimate_button.hide();tform.addWidget(self.power_estimate_button)
            self.turbine_select=QComboBox();tform.addWidget(self.turbine_select);self.turbine_quantity=QComboBox()
            for key,label in [('thrust_coefficient','Applied Ct'),('power_coefficient','Applied Cp'),('thrust_N','Thrust (N)'),('useful_power_W','Rotor mechanical power (W)'),('electrical_power_W','Estimated electrical power (W)'),('shaft_energy_Wh','Accumulated shaft energy (Wh)'),('electrical_energy_Wh','Accumulated electrical energy (Wh)'),('upstream_velocity_m_s','Incoming reference velocity (m/s)'),('local_velocity_m_s','Local normal velocity (m/s)'),('tip_speed_ratio','Tip-speed ratio'),('rotational_speed_rpm','Rotor speed (RPM)'),('hydraulic_work_W','Hydraulic work removed (W)'),('minimum_coverage_margin_m','Water cover above rotor (m)'),('minimum_bed_clearance_m','Rotor–bed clearance (m)'),('surface_clearance_reserve_m','Surface clearance above required limit (m)'),('bed_clearance_reserve_m','Bed clearance above required limit (m)'),('active','Rotor active')]:self.turbine_quantity.addItem(label,key)
            tform.addWidget(self.turbine_quantity);tplot=QPushButton('Plot turbine history');tplot.clicked.connect(self.plot_turbine);tform.addWidget(tplot)
            texport=QPushButton('Export all turbines to CSV');texport.clicked.connect(self.export_turbine);tform.addWidget(texport)
            selected_export=QPushButton('Export selected turbine to CSV');selected_export.clicked.connect(lambda:self.export_turbine(selected=True));tform.addWidget(selected_export)
            energy_export=QPushButton('Export all turbine power and energy to CSV');energy_export.clicked.connect(self.export_turbine_energy);tform.addWidget(energy_export)
            self.jordan_compare_button=QPushButton('Jordan analytical / hydraulic comparison…');self.jordan_compare_button.clicked.connect(self.compare_jordan);tform.addWidget(self.jordan_compare_button)
            self.jordan_compare_button.setToolTip('Compare this turbine-on run with a separate turbine-free ambient run and an explicit analytical calibration. Keeps both powers and the hydraulic velocity field.')
            self.turbine_buttons=(tplot,texport,selected_export,energy_export)
            for key,label in [('uncorrected_power_W','Uncorrected Thetis power (W)'),('power_correction_factor','Analytical wake power factor'),('rotor_power_velocity_m_s','Rotor energy-equivalent speed (m/s)'),('active_upstream_wakes','Contributing upstream wakes'),('ambient_reference_velocity_m_s','First-row incoming reference (m/s)'),('inferred_ambient_discrepancy_percent','Local ambient consistency difference (%)')]:self.turbine_quantity.addItem(label,key)
            self.turbine_status=QLabel('Waiting for the first saved turbine sample.');self.turbine_status.setWordWrap(True);tform.addWidget(self.turbine_status)
            for button in self.turbine_buttons:button.setEnabled(False)
            tnote=QLabel('Choose a turbine to plot its individual history. CSV includes its ID, model, Ct/Cp, velocities, coverage, thrust, power and status at each accepted result time. Each uncovered rotor switches off independently.');tnote.setWordWrap(True);tform.addWidget(tnote)
            for layout in self.page_layouts.values():layout.addStretch();layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
            for button in self.findChildren(QPushButton): button.setMinimumHeight(28)
            self.slider.valueChanged.connect(self.show_frame); self.variable.currentIndexChanged.connect(self.show_frame)
            self.mesh.toggled.connect(self.show_frame);self.animate_mesh.toggled.connect(self.show_frame)
            if map_view is not None:
                if hasattr(map_view, 'gauge_placed'): map_view.gauge_placed.connect(self.set_probe)
                if hasattr(map_view, 'profile_drawn'): map_view.profile_drawn.connect(self.set_transect)
                if hasattr(map_view,'measurement_selected'):map_view.measurement_selected.connect(self.select_measurement)
                if hasattr(map_view,'measurement_edited'):map_view.measurement_edited.connect(self._edited_on_map)
                if hasattr(map_view,'inspect_result'):map_view.inspect_result.connect(self.identify)

        def set_page(self,name):self.page_stack.setCurrentWidget(self.pages[name])

        def _profile_mode_changed(self,*_):
            spatial=self.profile_mode.currentData()=='spatial';self.profile_quantity.setVisible(not spatial);self.profile_animate.setVisible(spatial)
            self.profile_plot.setText('Plot spatial profile' if spatial else 'Plot flow / mean time series')
            self.profile_csv.setText('Export displayed profile CSV' if spatial else 'Export all variables to CSV')

        def set_extremum(self,kind,enabled):
            self.extrema_enabled[kind]=bool(enabled)
            if self._last_payload is not None:self._draw_extrema(self._last_payload)

        def _draw_extrema(self,payload):
            from .result_annotations import draw_extrema
            draw_extrema(self,payload)

        def set_extrema_visible(self,visible):
            from shiboken6 import isValid
            for item in self._extrema_items:
                if isValid(item):item.setVisible(visible and self.result_style().get('opacity',1.)>0)

        def set_sediment_available(self,enabled):
            selected=self.variable.currentData();self.variable.blockSignals(True);self.variable.clear()
            choices=dict(VARIABLES)
            if self.store:choices.update({key:variable_info(key) for key in self.store.hub_variables})
            configured=None if self.store is None else (self.store.project or {}).get('settings',{}).get('output_fields')
            permitted=set(CORE_MAP_VARIABLES)
            # Runs saved before selectable output fields retain their historic
            # full inventory.  New runs use precisely their saved selection.
            groups=OUTPUT_MAP_VARIABLES if configured is not None else {key:frozenset((key,)) for key in ()}
            if configured is None:
                permitted.update(key for key in choices if key not in {'velocity_x_m_s','velocity_y_m_s'})
            else:
                for group in configured:permitted.update(groups.get(group,()))
            for key,(label,unit,_) in choices.items():
                available=(key.startswith('hub_velocity_m_s:') or self.store is None or key in self.store.read(0).values)
                sediment=(self.store.project or {}).get('model',{}).get('physics',{}).get('sediment') if self.store else None
                if key=='suspended_kg_m3' and (not sediment or sediment['suspension']=='off'):available=False
                if available and key in permitted:self.variable.addItem(f'{label} [{unit}]',key)
            if self.variable.findData(selected)>=0:self.variable.setCurrentIndex(self.variable.findData(selected))
            self.variable.blockSignals(False)
            if hasattr(self,'profile_quantity'):
                current=self.profile_quantity.currentData();self.profile_quantity.clear();self.profile_quantity.addItem('Water flow [m³/s]','water')
                if enabled and self.store is not None:
                    if 'bedload_vector_m2_s' in self.store.read(0).arrays:self.profile_quantity.addItem('Bedload flow [m³/s solid]','bedload')
                    if sediment and sediment['suspension']!='off':
                        self.profile_quantity.addItem('Suspended advective flow [kg/s]','suspended')
                        if 'bedload_vector_m2_s' in self.store.read(0).arrays:self.profile_quantity.addItem('Total sediment flow [m³/s solid]','total_solid')
                for i in range(self.variable.count()):self.profile_quantity.addItem('Mean '+self.variable.itemText(i),self.variable.itemData(i))
                index=self.profile_quantity.findData(current)
                if index>=0:self.profile_quantity.setCurrentIndex(index)

        def identify(self,xy):
            if self.store is None:self._fail('Load results first');return
            try:
                frame=self.store.read(self.slider.value());cell=self.store._owner(frame,Point(*xy));variable=self.variable.currentData()
                if cell is None:text='Outside the result mesh'
                else:
                    label,unit,_=variable_info(variable);value=self.store.values(frame,variable)[cell]
                    text=f'{label}: {display_number(value)} {unit}\nt = {display_number(frame.time_s)} s · cell {int(frame.mesh["leaves"][cell])}'
                from PySide6.QtWidgets import QToolTip
                from PySide6.QtGui import QCursor
                QToolTip.showText(QCursor.pos(),text,self.map_view);self.caption.setText(text)
            except Exception as exc:self._fail(exc)

        def _fail(self, exc): self.caption.setText(str(exc)); self.error.emit(str(exc))

        def choose_output(self):
            path = QFileDialog.getExistingDirectory(self, 'BRAIDS run results')
            if path:
                try: self.load_output(path)
                except Exception as exc: self._fail(exc)

        def load_output(self, output, *, latest=True):
            candidate = ResultsStore(output)
            # Never reinterpret world coordinates under an unrelated terrain
            # CRS, and retain the current result if the requested load fails.
            # Generic graphics-view clients without a terrain contract remain
            # supported for headless exports and isolated viewer tests.
            if self.map_view is not None and hasattr(self.map_view, 'terrain'):
                terrain = self.map_view.terrain
                if terrain is None:
                    raise ValueError('Open the containing project folder first; its terrain must be loaded before displaying results.')
                domain = (candidate.project or {}).get('settings', {}).get('domain', {})
                target = getattr(terrain, 'wkt', '') or getattr(getattr(terrain, 'raster', None), 'crs_authid', '')
                source = domain.get('spatial_reference', '')
                try:
                    from ._avac.geo import spatial_reference
                    source_srs, target_srs = spatial_reference(source), spatial_reference(target)
                    matching = source_srs is not None and target_srs is not None and bool(source_srs.IsSame(target_srs))
                except Exception as exc:
                    raise ValueError('Open the containing project folder first; the result/terrain CRS cannot be verified.') from exc
                if not matching:
                    raise ValueError('Open the containing project folder first; the result CRS differs from the loaded terrain or is missing.')
            self.timer.stop(); self.play.setText('▶');self.play.setToolTip('Play'); self.store = candidate;self._turbine_snapshot=None;self._smooth_cache.clear()
            current_turbine=self.turbine_select.currentText();self.turbine_select.clear()
            self.turbine_select.addItems([r['id'] for r in ((candidate.project or {}).get('power_plan',{}).get('layout') or candidate.context.get('turbines') or {}).get('turbines',[])])
            self.power_estimate_button.setVisible(bool((candidate.project or {}).get('power_plan')))
            if self.turbine_select.findText(current_turbine)>=0:self.turbine_select.setCurrentText(current_turbine)
            self.set_sediment_available(bool((candidate.project or {}).get('model',candidate.context.get('model',{})).get('physics',{}).get('sediment')))
            self.boundary.clear(); self.boundary.addItems(self.store.boundary_ids)
            self.slider.blockSignals(True); self.slider.setRange(0, max(0, len(self.store.times)-1)); self.slider.setValue(self.slider.maximum() if latest else 0); self.slider.blockSignals(False)
            self._draw_measurements()
            self._refresh_turbine_status()
            return self.show_frame()

        def clear(self):
            """Detach results when the main window opens a different project."""
            from shiboken6 import isValid
            self.timer.stop(); self.play.setText('▶');self.play.setToolTip('Play'); self.store = None
            self._turbine_snapshot=None;self.turbine_select.clear();self._refresh_turbine_status()
            for dialog in list(self._dialogs):dialog.close()
            for item in self._extrema_items:
                if isValid(item) and item.scene() is not None:item.scene().removeItem(item)
            self._extrema_items=[];self._last_payload=None
            if self.particles:self.particles.clear()
            self.probe_point = self.transect_points = self._last_series = None
            self.probes=[];self.profiles=[];self._selected_probe=self._selected_profile=None
            self._sync_lists()
            self.slider.blockSignals(True); self.slider.setRange(0, 0); self.slider.blockSignals(False)
            for item in (self._field_item, self._mesh_item,self._terrain_item,self._smooth_item):
                if item is not None and isValid(item) and item.scene() is not None: item.scene().removeItem(item)
            self._field_item = self._mesh_item = self._terrain_item = None; self.caption.setText('No results loaded')
            self._smooth_item=None;self._smooth_cache.clear()
            self.boundary.clear()
            self._draw_measurements()
            if self.map_view is not None:
                if hasattr(self.map_view, 'result_metadata'): self.map_view.result_metadata = None
                if hasattr(self.map_view, 'time_text'): self.map_view.time_text = ''
                self.map_view.viewport().update()

        def refresh(self, *, latest=True):
            if self.store is None: return False
            try:
                changed = self.store.refresh()
                if changed:
                    self.slider.blockSignals(True); self.slider.setMaximum(max(0, len(self.store.times)-1))
                    if latest: self.slider.setValue(self.slider.maximum())
                    self.slider.blockSignals(False); self.show_frame()
                self._refresh_turbine_status()
                return changed
            except Exception as exc: self._fail(exc); return False

        def _mesh_toggled(self, other, checked):
            if checked: other.setChecked(False)
            self.show_frame()

        def _mode(self): return 'none' if not self.mesh.isChecked() else 'adaptive' if self.animate_mesh.isChecked() else 'initial'

        def result_style(self,variable=None):
            return getattr(self.map_view,'styles',{}).get('result:'+(variable or self.variable.currentData()),{})

        def _saved_domain_geometry(self):
            """Exact saved domain used to clip display-only result rasters."""
            project=(self.store.project if self.store is not None else None) or {}
            data=project.get('settings',{}).get('domain',{})
            vertices=data.get('vertices_xy')
            if not vertices:return None
            try:
                from shapely.geometry import Polygon
                geometry=Polygon(vertices,data.get('holes_xy',()))
                return geometry if geometry.is_valid and not geometry.is_empty else None
            except Exception:return None

        def _render(self, payload):
            if self.map_view is None: return
            from shiboken6 import isValid
            if self._field_item is None or not isValid(self._field_item):
                mapping = getattr(self.map_view, 'to_scene', None)
                self._field_item = ResultPolygonItem(mapping); self._mesh_item = ResultPolygonItem(mapping, outline=True)
                self.map_view.scene().addItem(self._field_item); self.map_view.scene().addItem(self._mesh_item)
            self._field_item.set_data(payload['polygons'], payload['rgba'])
            from .interpolated_field import painter_path
            domain_geometry=self._saved_domain_geometry()
            self._field_item.domain_clip=(painter_path(domain_geometry,self._field_item.to_scene)
                                          if domain_geometry is not None else None)
            if payload['mesh_polygons']:self._mesh_item.set_data(payload['mesh_polygons'])
            else:self._mesh_item.hide()
            opacity=self.result_style(payload['variable']).get('opacity',1.)
            self._field_item.setOpacity(opacity)
            self._smooth_ready=False
            if self._smooth_item is not None and not isValid(self._smooth_item):self._smooth_item=None
            render_smooth=self.smooth and self.smooth_level>0
            # This is deliberately restricted to depth-averaged hydraulic
            # layers.  Bedload, suspended load and bed change remain shown on
            # their actual finite-volume cells; a display-only wet wedge must
            # not imply that those processes were computed sub-cell.
            terrain_wetting_variables=frozenset(('depth_m','velocity_m_s','stage_m','unit_discharge_m2_s','froude','velocity_x_m_s','velocity_y_m_s'))
            render_terrain_wetting=self.terrain_aware_wetting and payload['variable'] in terrain_wetting_variables
            frame_for_display=payload['frame']
            # A published live frame must never make the event loop wait for
            # a large sub-cell drawing pass.  Keep the responsive native-cell
            # view until the run completes, then restore this option.
            if render_terrain_wetting:
                # Keep the native finite-volume map as the base layer and
                # replace only its shallow shoreline cells with exact,
                # volume-conserving terrain wedges. This remains responsive
                # during live AMR because thousands of unchanged interiors do
                # not need new QPainter paths each saved frame.
                from .visual_fields import terrain_wet_edges
                rows,display=terrain_wet_edges(frame_for_display)
                rgba=np.array(payload['rgba'],copy=True)
                if self.result_style(payload['variable']).get('zero_transparent',True):
                    rgba[rows,3]=0
                else:
                    zero_color=colour_values(np.array([0.]),payload['variable'],limits=payload['limits'],palette=payload['palette'],zero_transparent=False)[0][0]
                    rgba[rows]=zero_color
                self._field_item.set_data(payload['polygons'],rgba)
                if self._terrain_item is None or not isValid(self._terrain_item):
                    self._terrain_item=ResultPolygonItem(self.map_view.to_scene);self._terrain_item.setZValue(20.1);self.map_view.scene().addItem(self._terrain_item)
                self._terrain_item.set_data(display,payload['rgba'][rows])
                self._terrain_item.domain_clip=self._field_item.domain_clip
                self._terrain_item.setOpacity(opacity);self._terrain_item.setVisible(bool(len(rows)))
                self._field_item.show()
                if self._smooth_item is not None:self._smooth_item.hide()
                self._smooth_ready=False
            else:
                # Clear, rather than only hide, the prior frame's clipped
                # wedges.  A later layer-visibility refresh must never revive
                # a terrain overlay after this option is switched off.
                if self._terrain_item is not None and isValid(self._terrain_item):
                    self._terrain_item.set_data((),())
                    self._terrain_item.hide()
            if render_smooth and not payload['variable'].startswith('hub_velocity_m_s:') and hasattr(self.map_view,'terrain') and callable(getattr(self.map_view,'to_scene',None)):
                from .interpolated_field import InterpolatedFieldItem
                frame=frame_for_display
                origin=self.map_view.to_scene(0.,0.);east=self.map_view.to_scene(1.,0.);north=self.map_view.to_scene(0.,1.)
                row=self.store.index['frames'][frame.index]
                zero_transparent=self.result_style(payload['variable']).get(
                    'zero_transparent', self.map_view.default_zero_transparent('result:'+payload['variable']))
                key=(str(self.store.output),row['sha256'],row['epoch'],payload['variable'],tuple(payload['limits']),payload['palette'],self.smooth_level,zero_transparent,render_terrain_wetting,origin.x(),origin.y(),east.x(),east.y(),north.x(),north.y())
                if self._smooth_item is None:
                    self._smooth_item=InterpolatedFieldItem(self.map_view.to_scene);self.map_view.scene().addItem(self._smooth_item)
                if getattr(self._smooth_item,'field_key',None)!=key:
                    self._smooth_item.set_field(frame,payload['values'],payload['variable'],payload['limits'],payload['palette'],domain=self._saved_domain_geometry(),terrain=render_terrain_wetting,zero_transparent=zero_transparent)
                    self._smooth_item.field_key=key
                self._field_item.show()
                self._smooth_item.setOpacity(opacity*min(1.,self.smooth_level/100.))
                self._smooth_ready=True
            elif self._smooth_item is not None:self._smooth_item.hide()
            if hasattr(self.map_view, 'time_text'): self.map_view.time_text = f"t = {payload['time_s']:g} s"
            if hasattr(self.map_view, 'set_result_metadata'): self.map_view.set_result_metadata(payload)
            self._last_payload=payload;self._draw_extrema(payload)
            self.map_view.viewport().update()

        def show_frame(self, *_):
            if self.store is None: return False
            try:
                style=self.result_style()
                variable=self.variable.currentData()
                default_zero=getattr(self.map_view,'default_zero_transparent',lambda key:True)
                zero_transparent=style.get('zero_transparent',default_zero('result:'+variable))
                payload = self.store.payload(self.slider.value(), variable, mesh_mode=self._mode(),limits=style.get('limits'),palette=style.get('palette'),zero_transparent=zero_transparent)
                self._render(payload); self.frame_changed.emit(payload)
                self._update_particles(payload['frame'])
                self.caption.setText(f"t = {display_number(payload['time_s'])} s · {len(payload['values']):,} cells · frame {self.slider.value()+1}/{len(self.store.times)}\n{payload['label']}: {display_number(payload['limits'][0])} to {display_number(payload['limits'][1])}")
                return True
            except Exception as exc: self._fail(exc); self.timer.stop(); self.play.setText('▶'); return False

        def toggle_animation(self):
            if self.timer.isActive(): self.timer.stop(); self.play.setText('▶');self.play.setToolTip('Play')
            elif self.store is not None and len(self.store.times): self.timer.start(); self.play.setText('⏸');self.play.setToolTip('Pause')
            self._update_particles()

        def set_particles(self,enabled):self._particles_enabled=enabled;self._update_particles()

        def _update_particles(self,frame=None):
            if self.particles is None:return
            # Particle advection uses stored depth-averaged flow, not this
            # diagnostic plane. Do not imply a 3-D trajectory on its map.
            allowed=self._particles_enabled and self._results_active and not self.timer.isActive() and self.store is not None and not str(self.variable.currentData()).startswith('hub_velocity_m_s:')
            if not allowed:
                self.particles.timer.stop()
                if self.particles.item is not None:
                    from shiboken6 import isValid
                    if isValid(self.particles.item):self.particles.item.hide()
                return
            from ._avac.preferences import settings
            density=settings().value('map/particle_density',1000,type=int)
            frame=frame or self.store.read(self.slider.value())
            if self.particles.frame is not frame:self.particles.set_frame(frame,density)
            if self.particles.item is not None:self.particles.item.show();self.particles.timer.start()

        def _next_frame(self): self.slider.setValue((self.slider.value()+1) % (self.slider.maximum()+1))

        def step_frame(self,direction):
            self.timer.stop();self.play.setText('▶');self.play.setToolTip('Play');self.slider.setValue(max(0,min(self.slider.maximum(),self.slider.value()+direction)));self._update_particles()

        def set_probe(self, x, y=None):
            point = np.asarray(x if y is None else (x, y), dtype=float)
            if point.shape != (2,) or not np.isfinite(point).all(): self._fail('A finite probe coordinate pair is required'); return
            name='Probe '+str(max([int(p['id'].split('_')[-1]) for p in self.probes]+[0])+1)
            identifier='probe_'+name.split()[-1]
            self.geometry_edit_started.emit()
            self.probes.append(dict(id=identifier,name=name,coords=[point.tolist()]));self._selected_probe=identifier;self._active_kind='gauge';self._sync_lists()
            if self.map_view is not None and hasattr(self.map_view, 'set_tool'): self.map_view.set_tool('pan')
            self.caption.setText(f'Probe: {self.probe_point[0]:.3f}, {self.probe_point[1]:.3f}')
            self._draw_measurements(); self.measurements_changed.emit(self.measurement_state())

        def set_transect(self, points):
            points = np.asarray(points, float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all(): self._fail('Draw at least two finite transect points'); return
            if np.sum(np.linalg.norm(np.diff(points,axis=0),axis=1))<=0:self._fail('Profile must have positive length');return
            name='Profile '+str(max([int(p['id'].split('_')[-1]) for p in self.profiles]+[0])+1)
            identifier='profile_'+name.split()[-1]
            self.geometry_edit_started.emit()
            self.profiles.append(dict(id=identifier,name=name,coords=points.tolist()));self._selected_profile=identifier;self._active_kind='profile';self._sync_lists()
            if self.map_view is not None and hasattr(self.map_view, 'set_tool'): self.map_view.set_tool('pan')
            self.caption.setText(f'Flow transect: {len(points)} vertices')
            self._draw_measurements(); self.measurements_changed.emit(self.measurement_state())

        def measurement_state(self):
            from copy import deepcopy
            return dict(schema=2,probes=deepcopy(self.probes),profiles=deepcopy(self.profiles),selected_probe=self._selected_probe,selected_profile=self._selected_profile)

        def set_measurement_state(self, state):
            from copy import deepcopy
            if not isinstance(state,dict) or set(state)!={'schema','probes','profiles','selected_probe','selected_profile'} or state['schema']!=2:raise ValueError('Current multiple-measurement state required')
            for key,minimum in (('probes',1),('profiles',2)):
                identifiers=set()
                for row in state[key]:
                    points=np.asarray(row['coords'],float)
                    prefix='probe_' if key=='probes' else 'profile_'
                    if (set(row)!={'id','name','coords'} or not row['id'].startswith(prefix) or not row['id'][len(prefix):].isdigit()
                            or row['id'] in identifiers or not row['name'].strip() or points.ndim!=2 or points.shape[1]!=2
                            or len(points)<minimum or (minimum==1 and len(points)!=1) or not np.isfinite(points).all()):raise ValueError('Invalid measurement geometry or identity')
                    if minimum==2 and np.sum(np.linalg.norm(np.diff(points,axis=0),axis=1))<=0:raise ValueError('Profile must have positive length')
                    identifiers.add(row['id'])
            self.probes=deepcopy(state['probes']);self.profiles=deepcopy(state['profiles'])
            self._selected_probe=state['selected_probe'];self._selected_profile=state['selected_profile'];self._sync_lists();self._draw_measurements()

        def _sync_lists(self):
            for rows,listing,selected in ((self.probes,self.probe_list,self._selected_probe),(self.profiles,self.profile_list,self._selected_profile)):
                listing.blockSignals(True);listing.clear()
                for row in rows:listing.addItem(row['name'],row['id'])
                listing.setCurrentIndex(listing.findData(selected));listing.blockSignals(False)
            probe=next((p for p in self.probes if p['id']==self._selected_probe),None)
            profile=next((p for p in self.profiles if p['id']==self._selected_profile),None)
            self.probe_point=None if probe is None else tuple(probe['coords'][0])
            self.transect_points=None if profile is None else np.array(profile['coords'])

        def _select_measurement(self,kind):
            listing=self.probe_list if kind=='gauge' else self.profile_list
            self.select_measurement(kind,listing.currentData())

        def select_measurement(self,kind,identifier):
            if kind=='gauge':self._selected_probe=identifier
            else:self._selected_profile=identifier
            self._active_kind=kind
            if self.map_view is not None:self.map_view.set_tool('pan')
            self._sync_lists();self._draw_measurements();self.measurements_changed.emit(self.measurement_state())

        def new_measurement(self,kind):
            if self.map_view is not None:self.map_view.set_tool('gauge' if kind=='gauge' else 'profile')
            (self.probe_requested if kind=='gauge' else self.transect_requested).emit()

        def delete_measurement(self,kind):
            identifier=self._selected_probe if kind=='gauge' else self._selected_profile
            if identifier is None:return
            self.geometry_edit_started.emit()
            if self.map_view is not None:self.map_view.set_tool('pan')
            if kind=='gauge':self.probes=[p for p in self.probes if p['id']!=identifier];self._selected_probe=None
            else:self.profiles=[p for p in self.profiles if p['id']!=identifier];self._selected_profile=None
            self._sync_lists();self._draw_measurements();self.measurements_changed.emit(self.measurement_state())

        def _edited_on_map(self,kind,identifier,points):
            rows=self.probes if kind=='gauge' else self.profiles
            row=next((p for p in rows if p['id']==identifier),None)
            if row is None:return
            if kind=='profile' and np.sum(np.linalg.norm(np.diff(points,axis=0),axis=1))<=0:
                self._fail('Profile must retain positive length');self.map_view.set_tool('pan');self._draw_measurements();return
            self.geometry_edit_started.emit()
            row['coords']=np.asarray(points,float).tolist();self._sync_lists();self._draw_measurements();self.measurements_changed.emit(self.measurement_state())

        def _draw_measurements(self):
            if self.map_view is None or not hasattr(self.map_view, 'set_analysis_features'): return
            suffix = ' — '+self.store.output.name if self.store else ''
            features = []
            for kind,rows,selected in (('gauge',self.probes,self._selected_probe),('profile',self.profiles,self._selected_profile)):
                if not self.measurements_visible.get(kind,True):continue
                features.extend(dict(kind=kind,id=row['id'],name=row['name']+suffix,coords=row['coords'],selected=row['id']==selected and self._active_kind==kind) for row in rows)
            self.map_view.set_analysis_features(features)
            # Native AVAC analysis overlays use Z6; hydraulic cells use Z20.
            # Keep probe/flow annotations visible without changing map tools.
            for item in getattr(self.map_view, 'analysis_items', ()):item.setZValue(32)

        def _edit_coordinates(self, title, points, minimum):
            dialog = QDialog(self); dialog.setWindowTitle(title); dialog.resize(440, 300); layout = QVBoxLayout(dialog)
            layout.addWidget(QLabel('One easting; northing pair per line [map CRS units].'))
            initial_lines=[f'{display_number(x)}; {display_number(y)}' for x,y in points]
            editor = QPlainTextEdit('\n'.join(initial_lines)); layout.addWidget(editor)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel); layout.addWidget(buttons)
            result = []
            def accept():
                try:
                    from ._avac.i18n import parse_number,numeric_locale
                    rows=[]
                    for i,line in enumerate(editor.toPlainText().splitlines()):
                        if not line.strip():continue
                        if i<len(initial_lines) and line==initial_lines[i]:rows.append(list(points[i]));continue
                        parts=line.split(';') if ';' in line else (line.replace(',',' ').split() if numeric_locale().decimalPoint()=='.' else line.split())
                        rows.append([parse_number(value) for value in parts])
                    values = np.asarray(rows, dtype=float)
                    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all() or len(values) < minimum or (minimum == 1 and len(values) != 1):
                        raise ValueError('Enter exactly one coordinate pair' if minimum == 1 else 'Enter at least two coordinate pairs')
                    if minimum > 1 and np.sum(np.linalg.norm(np.diff(values, axis=0), axis=1)) <= 0: raise ValueError('Transect has zero length')
                except ValueError as exc: QMessageBox.warning(dialog, title, str(exc)); return
                result.append(values); dialog.accept()
            buttons.accepted.connect(accept); buttons.rejected.connect(dialog.reject)
            return result[0] if dialog.exec() == QDialog.DialogCode.Accepted else None

        def edit_probe(self):
            if self.probe_point is None: self._fail('Place a probe first'); return
            if self.map_view is not None:self.map_view.begin_measurement_edit('gauge',self._selected_probe,[self.probe_point])

        def edit_transect(self):
            if self.transect_points is None: self._fail('Draw a transect first'); return
            if self.map_view is not None:self.map_view.begin_measurement_edit('profile',self._selected_profile,self.transect_points)

        def _measure(self, fn):
            progress = QProgressDialog('Reading native result history…', 'Cancel', 0, 100, self); progress.setWindowModality(Qt.WindowModality.WindowModal); progress.show()
            def tick(value): progress.setValue(round(value*100)); QApplication.processEvents()
            try: return fn(tick, progress.wasCanceled)
            finally: progress.close()

        def plot_probe(self):
            if self.store is None or self.probe_point is None: self._fail('Load results and place a probe first'); return
            try: self._plot(self._measure(lambda p, c: self.store.probe(self.probe_point, self.variable.currentData(), progress=p, cancelled=c)),
                next((row['name'] for row in self.probes if row['id']==self._selected_probe),'Probe'))
            except Exception as exc: self._fail(exc)

        def plot_transect(self):
            if self.store is None or self.transect_points is None: self._fail('Load results and draw a transect first'); return
            if self.profile_mode.currentData()=='spatial':
                try:
                    from .result_annotations import SpatialProfileDialog
                    name=next((row['name'] for row in self.profiles if row['id']==self._selected_profile),'Profile')
                    dialog=SpatialProfileDialog(self,self.transect_points.copy(),name,follow=self.profile_animate.isChecked())
                    self._dialogs.append(dialog);dialog.finished.connect(lambda _:self._dialogs.remove(dialog) if dialog in self._dialogs else None);dialog.show()
                except Exception as exc:self._fail(exc)
                return
            try: self._plot(self._measure(lambda p, c: self.store.transect(self.transect_points, magnitude=True, progress=p, cancelled=c,quantity=self.profile_quantity.currentData())),
                next((row['name'] for row in self.profiles if row['id']==self._selected_profile),'Profile'))
            except Exception as exc: self._fail(exc)

        def plot_boundary(self):
            if self.store is None or not self.boundary.currentText(): self._fail('Load results with assigned inlet/outlet boundaries first'); return
            try: self._plot(self._measure(lambda p, c: self.store.boundary_flow(self.boundary.currentText(), quantity=self.boundary_quantity.currentData(), progress=p, cancelled=c)))
            except Exception as exc: self._fail(exc)

        def set_live_output(self,path):
            self._live_output=Path(path).resolve() if path is not None else None

        def _refresh_turbine_status(self):
            if self.store is None:
                self.turbine_status.setText('Waiting for the first saved turbine sample.')
                for button in self.turbine_buttons:button.setEnabled(False)
                return
            try:self._turbine_rows()
            except (OSError,ValueError,KeyError,TypeError) as exc:
                pending=self._live_output==Path(self.store.output).resolve()
                self.turbine_status.setText('Waiting for the first complete turbine sample.' if pending else str(exc))
                for button in self.turbine_buttons:button.setEnabled(False)

        def _turbine_rows(self):
            if self.store is None:raise ValueError('Load results first.')
            if (self.store.project or {}).get('power_plan'):
                from .power_planning import read
                rows,data=read(self.store);self._turbine_snapshot=(Path(self.store.output).resolve(),rows,data)
                self.turbine_status.setText(f'POWER ESTIMATE — finite-depth Gaussian + linear overlap. Last calculated: {rows[-1]["time_s"]:g} s. Hydraulic map is turbine-free. Refresh to include newer frames. Uniform vertical ambient profile; quasi-steady wakes; low-TI cases remain uncertain.')
                for button in self.turbine_buttons:button.setEnabled(True)
                return rows
            root=Path(self.store.output).resolve();pending=False;error=None
            for attempt in range(2):
                try:
                    raw=(root/'TURBINE_PERFORMANCE_INDEX.json').read_bytes();index=json.loads(raw)
                    if index.get('schema')!='braids.turbine-performance.v1' or set(index.get('files',{}))!={'TURBINE_PERFORMANCE.json','TURBINE_PERFORMANCE.csv'}:raise ValueError('Current turbine performance index required.')
                    blobs={name:(root/name).read_bytes() for name in index['files']}
                    if any(_sha(blobs[name])!=h for name,h in index['files'].items()) or (root/'TURBINE_PERFORMANCE_INDEX.json').read_bytes()!=raw:raise ValueError('Turbine performance is updating or its checksum differs. Refresh after publication.')
                    rows=json.loads(blobs['TURBINE_PERFORMANCE.json'])
                    if len(rows)!=index['rows'] or not rows:raise ValueError('No complete turbine samples are available yet.')
                    self._turbine_snapshot=(root,rows,blobs['TURBINE_PERFORMANCE.csv']);break
                except (OSError,ValueError,KeyError,TypeError) as exc:error=exc
            else:
                if self._live_output==root and self._turbine_snapshot is not None and self._turbine_snapshot[0]==root:
                    rows=self._turbine_snapshot[1];pending=True
                else:raise error
            self.turbine_status.setText(('Data updating — last verified sample' if pending else 'Latest saved turbine sample')+f": {display_number(rows[-1]['time_s'])} s. Plot/export uses this complete snapshot.")
            for button in self.turbine_buttons:button.setEnabled(True)
            names=list(dict.fromkeys(r['turbine_id'] for r in rows))
            if names!=[self.turbine_select.itemText(i) for i in range(self.turbine_select.count())]:
                current=self.turbine_select.currentText();self.turbine_select.clear();self.turbine_select.addItems(names)
                if current in names:self.turbine_select.setCurrentText(current)
            return rows

        def calculate_power_plan(self):
            if self.store is None:return
            try:
                from .power_planning import calculate
                self.power_estimate_button.setEnabled(False);self.store.refresh()
                self._measure(lambda p,c:calculate(self.store,progress=lambda v,label:p(v/100),cancelled=c));self._refresh_turbine_status()
            except Exception as exc:self._fail(exc)
            finally:self.power_estimate_button.setEnabled(True)

        def compare_jordan(self):
            if self.store is None:self._fail('Load the turbine-on results first.');return
            ambient=QFileDialog.getExistingDirectory(self,'Choose turbine-free ambient run (contains RUN_CONTEXT.json)')
            if not ambient:return
            config,_=QFileDialog.getOpenFileName(self,'Choose Jordan wake calibration and device curves','','JSON (*.json)')
            if not config:return
            try:
                from .jordan_comparison import compare
                self.jordan_compare_button.setEnabled(False)
                dest,rows=self._measure(lambda p,c:compare(ResultsStore(ambient),self.store,_read_json(config),progress=p,cancelled=c))
                QMessageBox.information(self,'Jordan comparison exported',f'Analytical and hydraulic power are saved separately in:\n{dest / "POWER.csv"}\n\nThe hydraulic velocity field is unchanged. Calibration against the hydraulic model is not independent experimental validation.')
            except Exception as exc:self._fail(exc)
            finally:self.jordan_compare_button.setEnabled(True)

        def plot_turbine(self):
            try:
                rows=[r for r in self._turbine_rows() if r['turbine_id']==self.turbine_select.currentText()];key=self.turbine_quantity.currentData();unit='N' if key=='thrust_N' else 'W' if key.endswith('_W') else 'm/s' if key.endswith('_m_s') else 'RPM' if key.endswith('_rpm') else 'm' if key.endswith('_m') else '—'
                if key.endswith('_energy_Wh'):
                    from braids_core.river_turbine_surface import integrate_energy
                    rows=integrate_energy(rows);unit='Wh'
                if any(r.get(key) is None for r in rows):raise ValueError('This quantity is unavailable: electrical power requires supplied efficiency, and a planning estimate does not remove hydraulic work from the ambient flow.')
                description=('Analytical power planning from turbine-free hydraulics; finite-depth Gaussian + linear overlap, quasi-steady, uniform vertical ambient profile. No hydraulic feedback.' if (self.store.project or {}).get('power_plan') else 'Accepted-state turbine diagnostics; declared performance model/control, fixed orientation.')
                series=Series(np.array([r['time_s'] for r in rows]),np.array([r[key] for r in rows],float),self.turbine_quantity.currentText(),unit,description+' Energy integrates saved samples in this run segment. Shaft power excludes generator losses; inactive rotor power is zero.')
                self._plot(series,rows[0]['turbine_id'])
            except Exception as e:self._fail(e)

        def export_turbine_energy(self):
            try:
                import csv
                from braids_core.river_turbine_surface import integrate_energy
                rows=integrate_energy(self._turbine_rows())
                name,_=QFileDialog.getSaveFileName(self,'Export shaft and electrical energy',self.project_name+'_turbine_power_energy.csv','CSV (*.csv)')
                if name:
                    with Path(name).open('x',newline='',encoding='utf-8') as f:
                        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            except Exception as e:self._fail(e)

        def export_turbine(self,selected=False):
            try:
                rows=self._turbine_rows();csv_bytes=self._turbine_snapshot[2];suffix='turbine_performance'
                if selected:
                    import re
                    turbine=self.turbine_select.currentText();rows=[r for r in rows if r['turbine_id']==turbine];suffix=re.sub(r'[^\w-]+','_',turbine)+'_performance'
                name,_=QFileDialog.getSaveFileName(self,'Export turbine performance',self.project_name+'_'+suffix+'.csv','CSV (*.csv)')
                if name:
                    if selected:
                        import csv
                        with Path(name).open('x',newline='',encoding='utf-8') as stream:
                            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                    else:
                        with Path(name).open('xb') as stream:stream.write(csv_bytes)
            except Exception as e:self._fail(e)

        def _plot(self, series,measurement_name=None):
            os.environ['QT_API'] = 'pyside6'
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self._last_series = series; dialog = QDialog(self); dialog.setWindowTitle(series.label); dialog.resize(800, 500)
            layout = QVBoxLayout(dialog); fig = Figure(constrained_layout=True); canvas = FigureCanvasQTAgg(fig); layout.addWidget(canvas)
            ax = fig.subplots(); ax.plot(series.time_s, series.values,color='red'); ax.set(xlabel='Time [s]', ylabel=f'{series.label} [{series.unit}]')
            from matplotlib.ticker import FuncFormatter
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:display_number(value)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:display_number(value)))
            finite = series.values[np.isfinite(series.values)];hi=float(np.max(finite,initial=0.));lo=float(np.min(finite,initial=0.))
            # Nonnegative series follow exactly the requested 0..1.1*maximum.
            # Signed elevations/velocity components retain negative values.
            ax.set_ylim(bottom=0. if lo>=0 else 1.1*lo,top=1.1*hi if hi>0 else 1e-9)
            ax.grid(alpha=.25); note = QLabel(series.method); note.setWordWrap(True); layout.addWidget(note)
            png=QPushButton('Save plot as PNG');png.setToolTip('Export this plot as PNG')
            from PySide6.QtWidgets import QStyle
            png.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton));layout.addWidget(png)
            import re
            stem='_'.join(filter(None,(measurement_name,self.project_name,series.label)))
            stem=re.sub(r'\s+','_',re.sub(r'[<>:"/\\|?*\x00-\x1f]','_',stem)).strip(' ._') or 'plot'
            def save_plot():
                filename,_=QFileDialog.getSaveFileName(dialog,'Export plot PNG',stem+'.png','PNG (*.png)')
                if filename:
                    try:
                        if Path(filename).exists():raise FileExistsError('Choose a new filename; existing plots are not overwritten.')
                        fig.savefig(filename,dpi=180)
                    except Exception as exc:self._fail(exc)
            png.clicked.connect(save_plot)
            export = QPushButton('Export CSV…'); export.clicked.connect(lambda: self.export_series(series,default_name=stem+'.csv')); layout.addWidget(export)
            self._dialogs.append(dialog); dialog.finished.connect(lambda _: self._dialogs.remove(dialog) if dialog in self._dialogs else None); dialog.show()

        def export_series(self, series=None,*,default_name=None):
            series = series if isinstance(series, Series) else self._last_series
            if series is None: self._fail('Plot a probe or transect first'); return
            path, _ = QFileDialog.getSaveFileName(self, 'Export time series', default_name or 'series.csv', 'CSV (*.csv)')
            if path:
                try: series.export_csv(path)
                except Exception as exc: self._fail(exc)

        def export_measurement(self,kind):
            rows=self.probes if kind=='gauge' else self.profiles;identifier=self._selected_probe if kind=='gauge' else self._selected_profile
            selected=next((row for row in rows if row['id']==identifier),None)
            if self.store is None or selected is None:self._fail('Select a probe/profile and available results first');return
            if kind=='profile' and self.profile_mode.currentData()=='spatial':
                try:
                    from .result_annotations import export_profile
                    series=self.store.spatial_profile(selected['coords'],self.variable.currentData(),self.slider.value(),interpolate=True)
                    export_profile(self,series,selected['name']+'_'+self.project_name)
                except Exception as exc:self._fail(exc)
                return
            import re
            name=re.sub(r'[<>:"/\\|?*\x00-\x1f]','_',selected['name']+'_'+self.project_name+'_all_variables')+'.csv'
            filename,_=QFileDialog.getSaveFileName(self,'Export all variables',name,'CSV (*.csv)')
            if not filename:return
            try:
                columns,values=self._measure(lambda p,c:self.store.measurement_table(selected['coords'],profile=kind=='profile',magnitude=True,progress=p,cancelled=c))
                with Path(filename).open('x',newline='',encoding='utf-8') as stream:
                    writer=csv.writer(stream);writer.writerow(columns);writer.writerows(values)
            except Exception as exc:self._fail(exc)

        def export_animation(self):
            if self.store is None or not len(self.store.times): self._fail('No saved result frames are available'); return
            from .media_export import ExportDialog,export_media
            size=self.map_view.viewport().size() if self.map_view is not None else self.size()
            dialog=ExportDialog(self.store.times,self.slider.value(),(size.width(),size.height()),self)
            if dialog.exec()!=QDialog.Accepted:return
            try:selection=dialog.selection()
            except Exception as exc:self._fail(exc);return
            kind=selection['kind'];name=self.project_name+'_'+self.variable.currentData()
            if kind=='png':name+=f"_frame_{selection['indices'][0]+1:04d}"
            filename,_=QFileDialog.getSaveFileName(self,'Export results',name+'.'+kind,kind.upper()+f' (*.{kind})')
            if not filename:return
            self.timer.stop(); self.play.setText('▶');self.play.setToolTip('Play')
            if self.particles:
                self.particles.timer.stop()
                # A chosen PNG may not be the currently paused frame. Never
                # export stale frozen-field trails over a different time frame.
                if self.particles.item is not None:self.particles.item.hide()
            def render(payload):
                self._render(payload);self.frame_changed.emit(payload)
                return self.map_view.export_image() if hasattr(self.map_view, 'export_image') else self.map_view.viewport().grab().toImage()
            try:
                style=self.result_style()
                self._measure(lambda p,c:export_media(self.store,filename,**selection,variable=self.variable.currentData(),mesh_mode=self._mode(),limits=style.get('limits'),palette=style.get('palette'),render=render,progress=p,cancelled=c))
                self.caption.setText('Exported '+filename)
            except Exception as exc: self._fail(exc)
            finally: self.show_frame()

        def export_current_bank_lines(self):
            if self.store is None or not len(self.store.times):
                self._fail('Load saved results before exporting river-bank breaklines');return
            frame=self.store.read(self.slider.value())
            filename,_=QFileDialog.getSaveFileName(self,'Export river-bank breaklines',
                f'{self.project_name}_river_banks_t{frame.time_s:g}s.geojson','GeoJSON (*.geojson)')
            if not filename:return
            try:
                from .river_bank_lines import export_bank_lines
                path,audit_path,audit=export_bank_lines(frame,self.store.project,filename,crs_wkt=self.store.crs)
                self.caption.setText(f'Exported {audit["lines"]} river-bank breaklines at t={frame.time_s:g} s')
                QMessageBox.information(self,'River-bank breaklines exported',
                    f'River-bank breaklines saved to:\n{path}\n\n'
                    'Import them with Mesh > Import breaklines, then generate the mesh. '
                    'The lines follow the selected frame shoreline and allow water across them as discharge changes.')
            except Exception as exc:self._fail(exc)

        def _legacy_export_current_domain(self):
            """Export the displayed wet footprint as an importable domain polygon."""
            if self.store is None or not len(self.store.times):
                self._fail('Load saved results before exporting a flooded domain'); return
            frame = self.store.read(self.slider.value())
            default = f'{self.project_name}_flooded_domain_t{frame.time_s:g}s.geojson'
            filename, _ = QFileDialog.getSaveFileName(
                self, 'Export result as domain', default, 'GeoJSON (*.geojson *.json)')
            if not filename:return
            try:
                from .result_domain import export_result_domain
                path, audit_path, audit = export_result_domain(frame, filename, crs_wkt=self.store.crs)
                note = (f'Flooded domain exported: {path.name} · {audit["vertices"]} vertices · '
                        f'{audit["wet_depth_threshold_m"]:g} m wet-depth threshold')
                if audit['disconnected_components_discarded']:
                    note += f' · discarded {audit["disconnected_components_discarded"]} disconnected wet component(s)'
                self.caption.setText(note)
                QMessageBox.information(self, 'Flooded domain exported',
                    f'An importable, smoothed domain was written to:\n{path}\n\n'
                    f'Export audit: {audit_path}\n\n'
                    'It uses the currently displayed result time and a 5 mm wet-depth threshold. '
                    'Import it from Geometry → Import domain, then review and assign its boundaries.')
            except Exception as exc:self._fail(exc)
else:
    class ResultsPanel:
        def __init__(self, *args, **kwargs): raise RuntimeError('BRAIDS Desktop requires PySide6')
