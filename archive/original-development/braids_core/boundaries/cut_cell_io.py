"""Input for the opt-in xbraids_cutcell executable (not the AMR WAVE deck).

Boundary support here is deliberately explicit: four total-discharge inlet
controls with an outgoing-characteristic trace, stage/critical/weir outlet
controls, or strict supercritical release.
Suspended concentration and prescribed total bedload may vary piecewise-linearly
per inlet ID and, for mixtures, independently for every sediment fraction.
Bedload may instead use capacity from the evolving adjacent flow.
Section-derived inlet targets are resolved for preparation diagnostics. The
production deck also carries the sampled endpoint terrain, control type and
energy slope so normal/critical levels follow the evolving boundary bed.
"""
from dataclasses import asdict,dataclass,replace
from bisect import bisect_left
from collections.abc import Mapping,Sequence
import hashlib,json,math,re,time
from pathlib import Path
from braids_core.windows_paths import filesystem_path
import numpy as np
from .cut_cells import CutCellMesh
from .section import Section
from braids_core.sediment.settling import (
    DEFAULT_HINDERED_REFERENCE_CONCENTRATION_KG_M3,
    HINDERED_SETTLING_EXPONENT,
)
from braids_core.sediment.capacity_parameters import (
    CapacityParameterError,
    formula_contract,
    resolve_capacity_parameters,
)
from braids_core.sediment.capacity_profile import (
    CapacityProfileError,
    capacity_diagnostic_contract_document,
    capacity_diagnostic_contract_id,
    capacity_profile_document,
    capacity_profile_semantic_id,
    load_capacity_profile,
    resolve_capacity_profile_from_code,
)
from braids_core.sediment.capacity_execution_identity import (
    EH_DELFT_METHOD_ID,
    EH_LEGACY_METHOD_ID,
    MPM_DELFT_METHOD_ID,
    MPM_LEGACY_METHOD_ID,
    capacity_execution_semantic_id,
    validate_typed_roughness_binding,
)
from braids_core.sediment.capacity_foundation_execution import (
    SEMANTIC_ID as CAPACITY_FOUNDATION_EXECUTION_SEMANTIC_ID,
    load_capacity_foundation_execution,
)
from braids_core.sediment.hiding_foundation_execution import (
    EXECUTION_KEY as HIDING_FOUNDATION_EXECUTION_KEY,
    ORDERED_ROUTE_KEY as HIDING_FOUNDATION_ORDERED_ROUTE_KEY,
    validate_hiding_foundation_execution_document,
)
from braids_core.sediment.secondary_foundation_execution import (
    EXECUTION_KEY as SECONDARY_FOUNDATION_EXECUTION_KEY,
    ORDERED_ROUTE_KEY as SECONDARY_FOUNDATION_ORDERED_ROUTE_KEY,
    validate_secondary_foundation_execution_document,
)
from braids_core.sediment.prognostic_secondary_execution import (
    EXECUTION_KEY as PROGNOSTIC_SECONDARY_EXECUTION_KEY,
    ORDERED_ROUTE_KEY as PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY,
    validate_prognostic_secondary_execution_document,
)
from braids_core.sediment.channel_composition import (
    CHANNEL_COMPOSITION_NATIVE_HEX_CHUNK,
    CHANNEL_COMPOSITION_NATIVE_WIRE_SCHEMA,
    CHANNEL_COMPOSITION_SCHEMA,
    CHANNEL_COMPOSITION_SCOPE_ID,
    TRA_BEDLOAD,
    TRA_COMBINE,
    ChannelCompositionError,
    ChannelCompositionProfile,
    channel_composition_native_wire_document,
    channel_composition_profile_document,
    channel_composition_semantic_id,
    load_channel_composition_profile,
    resolve_channel_composition_profile,
)
from braids_core.sediment.execution_profile_v2 import (
    EXECUTION_PROFILE_SCHEMA,
    Flow2DHExecutionProfile,
    ExecutionProfileError,
    execution_profile_document,
    execution_profile_payload_json,
    execution_profile_semantic_id,
    load_execution_profile,
)
from braids_core.sediment.flow2dh_exchange import (
    FLOW2DH_EXCHANGE_ID,
    FLOW2DH_EXCHANGE_SEMANTIC_ID,
)
from braids_core.sediment.flow2dh_boundary_diffusion import (
    FLOW2DH_HORIZONTAL_DIFFUSION_ID,
    FLOW2DH_HORIZONTAL_DIFFUSION_SEMANTIC_ID,
    FLOW2DH_HORIZONTAL_SCHMIDT,
    FLOW2DH_OPEN_BOUNDARY_ID,
    FLOW2DH_OPEN_BOUNDARY_SEMANTIC_ID,
)
from braids_core.sediment.flow2dh_settling import (
    FLOW2DH_SETTLING_FORM,
    FLOW2DH_SETTLING_ID,
    FLOW2DH_SETTLING_SEMANTIC_ID,
    settlefrm2_noncohesive,
)
from braids_core.sediment.suspension_profile import (
    resolve_suspension_profile,
    suspension_profile_document,
)
from braids_core.sediment.suspended_amr_contract import (
    SUSPENDED_AMR_SEMANTIC_ID,
    resolve_suspended_amr_contract,
    suspended_amr_contract_document,
)

NATIVE_DRY_TOLERANCE_M = 1.0e-3

# These identifiers are part of the native v46/v47/v48 wire protocol, not
# aliases for the latest backend-neutral catalogues.  The latter legitimately
# evolve as additional formulas and execution contracts are added, whereas an
# established native envelope must remain readable by frozen executables and
# restarts.  Keep the historical wire IDs explicit and carry newer canonical
# profile identities in their own outer envelopes/sidecars.
NATIVE_V46_CAPACITY_PARAMETER_SCHEMA_ID = (
    "6a9237f904d32258afacfb4c0949316509e29403b1154cc712beb3e8717edcc4"
)
NATIVE_V46_CAPACITY_PROFILE_SCHEMA_ID = (
    "c60c0ec96b946c5b9ef33f117ce0733deccd7d02137ffeb4f3fba11fa8f29bba"
)
NATIVE_V47_SUSPENSION_PROFILE_SCHEMA_ID = (
    "e833adfb99f0831b59a57b2ac5ea2b41af9db79e627215fe71cb97e65297b012"
)


def _canonical_section_endpoints(face):
    """Absolute section elevations in the native positive-tangent order."""
    tangent_x, tangent_y = -face.outward_normal[1], face.outward_normal[0]
    if tangent_x < 0 or (tangent_x == 0 and tangent_y < 0):
        tangent_x, tangent_y = -tangent_x, -tangent_y
    forward = ((face.end_xy[0]-face.start_xy[0])*tangent_x+
               (face.end_xy[1]-face.start_xy[1])*tangent_y) >= 0
    return ((face.section_bed_start_m, face.section_bed_end_m) if forward else
            (face.section_bed_end_m, face.section_bed_start_m))


def _amr_sampled_section_geometry(hierarchy, finest):
    """Fill missing AMR face survey data from the authoritative fine section.

    Main-mesh preparation already supplies physical section endpoints. Reuse
    that piecewise-linear survey at every hierarchy level; callers need not
    independently decorate the tree. Existing samples and all coordinates,
    normals, owner IDs and conserved state are left unchanged.
    """
    groups = {}
    for face in finest.faces:
        if face.section_bed_start_m is not None:
            groups.setdefault(face.bc_id, []).append(face)
    if not groups:
        return hierarchy
    surveys = {}
    for boundary, source in groups.items():
        normal = np.asarray(source[0].outward_normal)
        tangent = np.asarray([-normal[1], normal[0]])
        if tangent[0] < 0 or (tangent[0] == 0 and tangent[1] < 0):
            tangent = -tangent
        intervals, elevations, midpoints = [], [], []
        for face in source:
            if not np.allclose(face.outward_normal, normal, rtol=0, atol=1e-12):
                raise ValueError('AMR sampled section changes physical inlet normal')
            lo, hi = sorted((float(np.dot(face.start_xy, tangent)),
                             float(np.dot(face.end_xy, tangent))))
            intervals.append((lo, hi))
            elevations.append(_canonical_section_endpoints(face))
            midpoints.append(face.bed_elevation_m)
        bounds = np.asarray(intervals)
        order = np.argsort(bounds[:, 0], kind='stable')
        bounds = bounds[order]
        section_scale = max(1., float(np.max(np.abs(bounds))))
        tolerance = 64*np.finfo(float).eps*section_scale
        if np.any(bounds[:, 1] <= bounds[:, 0]) or np.any(bounds[1:, 0] < bounds[:-1, 1]-tolerance):
            raise ValueError('AMR sampled section has overlapping or empty physical segments')
        surveys[boundary] = (normal, tangent, bounds,
                             np.asarray(elevations)[order], np.asarray(midpoints)[order], section_scale)
    meshes = []
    for mesh in hierarchy.meshes:
        faces = []
        for face in mesh.faces:
            if face.bc_id in surveys and face.section_bed_start_m is None:
                normal, tangent, bounds, height, middle, section_scale = surveys[face.bc_id]
                if not np.allclose(face.outward_normal, normal, rtol=0, atol=1e-12):
                    raise ValueError('AMR sampled section changes physical inlet normal')
                start, end = float(np.dot(face.start_xy, tangent)), float(np.dot(face.end_xy, tangent))
                scale = max(section_scale, abs(start), abs(end))
                tolerance = 64*np.finfo(float).eps*scale

                def at(position):
                    index = max(0, int(np.searchsorted(bounds[:, 0], position, side='right'))-1)
                    lo, hi = bounds[index]
                    if position < lo-tolerance or position > hi+tolerance:
                        raise ValueError('AMR section face extends beyond the physical fine survey')
                    fraction = min(1., max(0., (position-lo)/(hi-lo)))
                    value = height[index, 0]+fraction*(height[index, 1]-height[index, 0])
                    # Same arithmetic mean used by native shared section nodes,
                    # with O(log N) lookup instead of scanning N fine faces per
                    # coarse endpoint on large river boundary sections.
                    if index and abs(position-lo) <= tolerance and abs(bounds[index-1, 1]-lo) <= tolerance:
                        value = .5*(value+height[index-1, 1])
                    elif (index+1 < len(bounds) and abs(position-hi) <= tolerance and
                          abs(bounds[index+1, 0]-hi) <= tolerance):
                        value = .5*(value+height[index+1, 0])
                    return float(value)

                lo, hi = sorted((start, end))
                index = max(0, int(np.searchsorted(bounds[:, 0], lo+tolerance, side='right'))-1)
                same = abs(bounds[index, 0]-lo) <= tolerance and abs(bounds[index, 1]-hi) <= tolerance
                midpoint = float(middle[index]) if same else at(.5*(start+end))
                face = replace(face, bed_elevation_m=midpoint,
                    section_bed_start_m=at(start), section_bed_end_m=at(end))
            faces.append(face)
        meshes.append(replace(mesh, faces=tuple(faces)))
    return replace(hierarchy, meshes=tuple(meshes))


def _replace_with_windows_retry(source: Path, target: Path) -> None:
    """Promote one complete sidecar despite short antivirus/indexer locks."""
    for attempt in range(8):
        try:
            source.replace(target)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32) or attempt == 7:
                raise
            time.sleep(0.02 * (2 ** attempt))


@dataclass(frozen=True)
class CutSedimentBoundarySeries:
    """Piecewise-linear inlet sediment forcing, held outside its time range.

    Bedload is a total dry-sediment mass rate for the complete logical ID,
    regardless of how many polygon edges or native face fragments it spans.
    """
    times_s: tuple[float,...]
    concentration_kg_m3: tuple[float,...]
    bed_input_kg_s: tuple[float,...]

    def __post_init__(self):
        for name in ('times_s','concentration_kg_m3','bed_input_kg_s'):
            values=tuple(getattr(self,name))
            object.__setattr__(self,name,values)
            if any(isinstance(v,bool) or not math.isfinite(v) for v in values):
                raise ValueError('Boundary time-series data must be finite')
        if len(self.times_s)<2 or len(self.concentration_kg_m3)!=len(self.times_s) or len(self.bed_input_kg_s)!=len(self.times_s):
            raise ValueError('Boundary time series require at least two equally sized arrays')
        if any(b<=a for a,b in zip(self.times_s,self.times_s[1:])):
            raise ValueError('Boundary time-series times must be strictly increasing')
        if self.times_s[0]<0 or min(self.concentration_kg_m3)<0 or min(self.bed_input_kg_s)<0:
            raise ValueError('Boundary time, concentration and bed input must be nonnegative')


@dataclass(frozen=True)
class CutHydraulicBoundarySeries:
    """Piecewise-linear hydraulic forcing, held outside its time range.

    ``control_m`` is absolute level for stage modes, depth above the fixed
    section datum for a depth inlet, and absent for normal/critical inlets.
    Their target levels are resolved from discharge at every declared knot.
    """
    times_s: tuple[float,...]
    discharge_m3_s: tuple[float,...]
    control_m: tuple[float,...]|None=None

    def __post_init__(self):
        times=tuple(self.times_s);discharge=tuple(self.discharge_m3_s)
        control=None if self.control_m is None else tuple(self.control_m)
        object.__setattr__(self,'times_s',times);object.__setattr__(self,'discharge_m3_s',discharge)
        object.__setattr__(self,'control_m',control)
        arrays=(times,discharge) if control is None else (times,discharge,control)
        if any(isinstance(value,bool) or not math.isfinite(value) for values in arrays for value in values):
            raise ValueError('Hydraulic time-series data must be finite')
        if len(times)<2 or len(discharge)!=len(times) or (control is not None and len(control)!=len(times)):
            raise ValueError('Hydraulic time series require at least two equally sized arrays')
        if any(right<=left for left,right in zip(times,times[1:])):
            raise ValueError('Hydraulic time-series times must be strictly increasing')
        if times[0]<0 or min(discharge)<0:
            raise ValueError('Hydraulic time and discharge must be nonnegative')


@dataclass(frozen=True)
class CutMorphologicalFactorSeries:
    """Delft3D-compatible block MorFac forcing in hydraulic seconds."""
    times_s: tuple[float,...]
    factors: tuple[float,...]

    def __post_init__(self):
        times=tuple(self.times_s);factors=tuple(self.factors)
        object.__setattr__(self,'times_s',times);object.__setattr__(self,'factors',factors)
        if any(isinstance(value,bool) or not math.isfinite(value) for values in (times,factors) for value in values):
            raise ValueError('Morphological-factor time-series data must be finite')
        if len(times)<2 or len(factors)!=len(times):
            raise ValueError('Morphological-factor time series require at least two equally sized arrays')
        if any(right<=left for left,right in zip(times,times[1:])):
            raise ValueError('Morphological-factor times must be strictly increasing')
        if times[0]<0 or min(factors)<1 or max(factors)>20:
            raise ValueError('Morphological-factor times must be nonnegative and factors must be within 1 to 20')


@dataclass(frozen=True)
class CutBoundary:
    mode: str
    discharge_m3_s: float=0.
    stage_m: float=0.
    concentration_kg_m3: float=0.
    bed_input_kg_m_s: float=0.
    bed_input_kg_s: float|None=None
    bedload_boundary_mode: str='prescribed'
    sediment_series: CutSedimentBoundarySeries|None=None
    reversal_policy: str='reject'  # v48 may explicitly opt into FLOW return-time reversal
    depth_m: float=0.
    energy_slope: float=.001
    boundary_manning_n: float|None=None
    weir_coefficient: float=1.7
    weir_width_m: float|None=None
    hydraulic_series: CutHydraulicBoundarySeries|None=None
    fraction_concentration_kg_m3: dict[str,float]|None=None
    fraction_bed_input_kg_s: dict[str,float]|None=None
    fraction_sediment_series: dict[str,CutSedimentBoundarySeries]|None=None
    total_discharge_profile: str='stock_conveyance'
    suspended_return_time_s: float=0.

    def __post_init__(self):
        if self.mode not in ('discharge','total_discharge','prescribed_trace','prescribed_depth_trace','inlet_stage','inlet_depth','inlet_normal','inlet_critical',
                             'stage','supercritical','critical','weir'):
            raise ValueError('Unsupported cut-cell boundary mode')
        if self.total_discharge_profile not in ('stock_conveyance','viscous_manning_equilibrium'):
            raise ValueError('Total-discharge profile must be stock_conveyance or viscous_manning_equilibrium')
        if self.total_discharge_profile!='stock_conveyance' and self.mode!='total_discharge':
            raise ValueError('A viscosity-equilibrated profile requires a total_discharge boundary')
        if isinstance(self.sediment_series,dict):
            object.__setattr__(self,'sediment_series',CutSedimentBoundarySeries(**self.sediment_series))
        if isinstance(self.hydraulic_series,dict):
            object.__setattr__(self,'hydraulic_series',CutHydraulicBoundarySeries(**self.hydraulic_series))
        for field_name in ('fraction_concentration_kg_m3','fraction_bed_input_kg_s'):
            supplied=getattr(self,field_name)
            values={} if supplied is None else dict(supplied)
            if any(not isinstance(key,str) or not key.strip() for key in values):
                raise ValueError('Sediment fraction IDs must be nonempty text')
            if any(isinstance(value,bool) or not math.isfinite(value) or value<0 for value in values.values()):
                raise ValueError('Per-fraction boundary sediment values must be finite and nonnegative')
            object.__setattr__(self,field_name,values)
        supplied={} if self.fraction_sediment_series is None else dict(self.fraction_sediment_series)
        fraction_series={}
        for key,value in supplied.items():
            if not isinstance(key,str) or not key.strip():
                raise ValueError('Sediment fraction IDs must be nonempty text')
            if isinstance(value,dict):value=CutSedimentBoundarySeries(**value)
            if not isinstance(value,CutSedimentBoundarySeries):
                raise ValueError('Each fraction sediment series must be CutSedimentBoundarySeries')
            fraction_series[key]=value
        object.__setattr__(self,'fraction_sediment_series',fraction_series)
        if self.sediment_series is not None and not isinstance(self.sediment_series,CutSedimentBoundarySeries):
            raise ValueError('sediment_series must be CutSedimentBoundarySeries')
        if self.hydraulic_series is not None and not isinstance(self.hydraulic_series,CutHydraulicBoundarySeries):
            raise ValueError('hydraulic_series must be CutHydraulicBoundarySeries')
        if self.reversal_policy not in ('reject','flow2dh_return'):
            raise ValueError(
                "Boundary reversal policy must be 'reject' or the v48-only "
                "'flow2dh_return' policy")
        if self.bedload_boundary_mode not in (
                'prescribed','equilibrium_capacity','prescribed_fixed_face_bed'):
            raise ValueError(
                "Bed-load boundary mode must be 'prescribed', 'equilibrium_capacity', "
                "or 'prescribed_fixed_face_bed'")
        values=[self.discharge_m3_s,self.stage_m,self.concentration_kg_m3,self.bed_input_kg_m_s,
                self.depth_m,self.energy_slope,self.weir_coefficient,
                self.suspended_return_time_s]
        if self.bed_input_kg_s is not None:
            values.append(self.bed_input_kg_s)
        if self.boundary_manning_n is not None:values.append(self.boundary_manning_n)
        if self.weir_width_m is not None:values.append(self.weir_width_m)
        if any(isinstance(v,bool) or not math.isfinite(v) for v in values):
            raise ValueError('Boundary data must be finite')
        if min(self.discharge_m3_s,self.concentration_kg_m3,self.bed_input_kg_m_s,self.depth_m)<0:
            raise ValueError('Q, depth, incoming concentration and bed input must be nonnegative')
        if self.mode=='inlet_normal' and (self.energy_slope<=0 or
                (self.boundary_manning_n is not None and self.boundary_manning_n<=0)):
            raise ValueError('Normal-depth inlet needs positive energy slope and roughness parameter')
        if self.mode=='weir' and (self.weir_coefficient<=0 or
                (self.weir_width_m is not None and self.weir_width_m<=0)):
            raise ValueError('Weir outlet needs positive coefficient and optional width')
        if self.bed_input_kg_s is not None and self.bed_input_kg_m_s:
            raise ValueError('Specify either total or per-width bed input, not both')
        if self.bedload_boundary_mode=='equilibrium_capacity':
            if not self.is_inlet:
                raise ValueError('Equilibrium-capacity bed load is supported on inlet IDs only')
            if (self.bed_input_kg_m_s or self.bed_input_kg_s is not None or any(self.fraction_bed_input_kg_s.values()) or
                    any(any(series.bed_input_kg_s) for series in self.fraction_sediment_series.values())):
                raise ValueError('Equilibrium-capacity bed load cannot also prescribe a bed input')
            if self.sediment_series is not None:
                raise ValueError('Equilibrium-capacity bed load cannot use a prescribed sediment time series')
        if self.bedload_boundary_mode=='prescribed_fixed_face_bed':
            if self.mode!='total_discharge':
                raise ValueError(
                    'A prescribed fixed-face bed-load boundary requires a total_discharge inlet')
            positive_feed=(
                self.bed_input_kg_m_s>0 or
                (self.bed_input_kg_s is not None and self.bed_input_kg_s>0) or
                any(value>0 for value in self.fraction_bed_input_kg_s.values()) or
                (self.sediment_series is not None and
                 any(value>0 for value in self.sediment_series.bed_input_kg_s)) or
                any(any(value>0 for value in series.bed_input_kg_s)
                    for series in self.fraction_sediment_series.values()))
            if not positive_feed:
                raise ValueError(
                    'A prescribed fixed-face bed-load boundary requires an explicit positive bed input')
        if not self.is_inlet and (self.discharge_m3_s or self.bed_input_kg_m_s or self.bed_input_kg_s or
                                  any(self.fraction_concentration_kg_m3.values()) or any(self.fraction_bed_input_kg_s.values()) or
                                  self.fraction_sediment_series):
            raise ValueError('An outlet cannot prescribe discharge or incoming bedload')
        if self.sediment_series is not None:
            if not self.is_inlet:
                raise ValueError('Sediment time series are currently supported on inlet IDs only')
            if self.concentration_kg_m3 or self.bed_input_kg_m_s or self.bed_input_kg_s is not None:
                raise ValueError('A sediment time series replaces constant concentration and bed input')
        if self.fraction_sediment_series:
            if not self.is_inlet:
                raise ValueError('Fraction sediment time series are supported on inlet IDs only')
            if self.sediment_series is not None:
                raise ValueError('Use either single-material or per-fraction sediment time series')
            constant_fractions={key for key,value in self.fraction_concentration_kg_m3.items() if value}
            constant_fractions|={key for key,value in self.fraction_bed_input_kg_s.items() if value}
            overlap=set(self.fraction_sediment_series)&constant_fractions
            if overlap:
                raise ValueError('A fraction sediment time series replaces that fraction constant supply')
        if self.hydraulic_series is not None:
            series=self.hydraulic_series
            if self.mode not in ('discharge','total_discharge','prescribed_trace','prescribed_depth_trace','inlet_stage','inlet_depth','inlet_normal','inlet_critical','stage'):
                raise ValueError('Hydraulic time series are supported on inlets and prescribed-stage outlets')
            needs_control=self.mode in ('discharge','prescribed_trace','prescribed_depth_trace','inlet_stage','inlet_depth','stage')
            if needs_control!=(series.control_m is not None):
                raise ValueError('Hydraulic control values must match the selected boundary mode')
            if self.mode=='stage' and any(series.discharge_m3_s):
                raise ValueError('A stage-outlet series cannot prescribe discharge')
            if self.mode in ('inlet_depth','prescribed_depth_trace') and min(series.control_m)<0:
                raise ValueError('Hydraulic depth-series values must be nonnegative')
            constant_control=(self.stage_m if self.mode in ('discharge','prescribed_trace','inlet_stage','stage') else
                              self.depth_m if self.mode in ('inlet_depth','prescribed_depth_trace') else 0.)
            if self.discharge_m3_s or constant_control:
                raise ValueError('A hydraulic time series replaces constant discharge and level/depth controls')
        if self.mode=='prescribed_depth_trace':
            discharge=(self.hydraulic_series.discharge_m3_s if self.hydraulic_series else (self.discharge_m3_s,))
            depths=(self.hydraulic_series.control_m if self.hydraulic_series else (self.depth_m,))
            if any(q>0 and h<=NATIVE_DRY_TOLERANCE_M for q,h in zip(discharge,depths)):
                raise ValueError('A prescribed depth trace cannot carry positive discharge at a dry depth')

    @property
    def is_inlet(self):
        return self.mode in ('discharge','total_discharge','prescribed_trace','prescribed_depth_trace','inlet_stage','inlet_depth','inlet_normal','inlet_critical')

    @property
    def needs_section(self):
        return self.mode in ('inlet_stage','inlet_depth','inlet_normal','inlet_critical')


def resolve_boundary_target(spec: CutBoundary, section: Section|None=None, *,
                            default_manning_n: float=.03,
                            hydraulic_roughness_mode: str='manning',
                            white_colebrook_chzk2d: bool=False,
                            roughness_minimum_work_depth_m: float=0.) -> dict:
    """Resolve one user control to the target consumed by the native trace."""
    if not isinstance(spec,CutBoundary):raise ValueError('CutBoundary is required')
    solution=None
    if spec.mode=='total_discharge':
        return {'requested_mode':spec.mode,'native_mode':6,'target_stage_m':None,
                'depth_m':None,'section_geometry':None,'hydraulic_solution':None}
    if spec.mode=='prescribed_trace':
        stage=spec.stage_m
        geometry=None if section is None else section.geometry(stage)
        if geometry is not None and spec.discharge_m3_s>0 and geometry.area_m2<=0:
            raise ValueError('Positive inlet discharge requires a wet target section')
        return {'requested_mode':spec.mode,'native_mode':7,'target_stage_m':stage,
                'depth_m':None if section is None else stage-section.datum_m,
                'section_geometry':None if geometry is None else asdict(geometry),'hydraulic_solution':None}
    if spec.mode=='prescribed_depth_trace':
        return {'requested_mode':spec.mode,'native_mode':8,'target_stage_m':spec.depth_m,
                'depth_m':spec.depth_m,'section_geometry':None,'hydraulic_solution':None}
    if spec.mode in ('discharge','inlet_stage'):
        stage=spec.stage_m
    elif spec.mode=='inlet_depth':
        if not isinstance(section,Section):raise ValueError('Depth inlet requires its sampled Section')
        stage=section.datum_m+spec.depth_m;section.geometry(stage)
    elif spec.mode=='inlet_normal':
        if not isinstance(section,Section):raise ValueError('Normal-depth inlet requires its sampled Section')
        coefficient=default_manning_n if spec.boundary_manning_n is None else spec.boundary_manning_n
        section_n=None if section.segment_manning_n is not None else coefficient
        solution=section.normal_stage(
            spec.discharge_m3_s,section_n,spec.energy_slope,
            roughness_mode=hydraulic_roughness_mode,
            white_colebrook_chzk2d=white_colebrook_chzk2d,
            minimum_work_depth_m=roughness_minimum_work_depth_m);stage=solution.stage_m
    elif spec.mode=='inlet_critical':
        if not isinstance(section,Section):raise ValueError('Critical-depth inlet requires its sampled Section')
        solution=section.critical_stage(spec.discharge_m3_s);stage=solution.stage_m
    else:
        return {'requested_mode':spec.mode,'native_mode':dict(stage=2,supercritical=3,critical=4,weir=5)[spec.mode],
                'target_stage_m':spec.stage_m if spec.mode in ('stage','weir') else None}
    geometry=None
    if section is not None:
        geometry=section.geometry(stage)
        if spec.discharge_m3_s>0 and geometry.area_m2<=0:
            raise ValueError('Positive inlet discharge requires a wet target section')
    # The user-facing section controls prescribe both total discharge and the
    # horizontal water line (directly, or from normal/critical depth). Keep
    # that resolved section state throughout the run. Mode 1 remains the
    # legacy characteristic discharge boundary, whose compatible stage may
    # depart from the requested section level as the interior evolves.
    native_mode=9 if spec.mode in ('inlet_stage','inlet_depth','inlet_normal','inlet_critical') else 1
    return {'requested_mode':spec.mode,'native_mode':native_mode,'target_stage_m':stage,
            'depth_m':None if section is None else stage-section.datum_m,
            'section_geometry':None if geometry is None else asdict(geometry),
            'hydraulic_solution':None if solution is None else asdict(solution)}


def _deferred_live_section_target(spec: CutBoundary) -> dict:
    """Describe the production solver's live section law, not a fixed stage.

    Production endpoints meet reflective domain edges. In particular, the
    native normal-flow law can extend vertical sidewalls above those endpoints;
    the standalone Section calculator deliberately cannot. Constant and
    scheduled production controls must therefore use the same live resolver.
    """
    return {'requested_mode':spec.mode,'native_mode':9,'target_stage_m':None,
            'depth_m':None,'section_geometry':None,'hydraulic_solution':None,
            'runtime_resolution':'live section control at every SSPRK stage'}


def resolve_hydraulic_series(spec: CutBoundary, section: Section|None=None, *,
                             default_manning_n: float=.03,
                             hydraulic_roughness_mode: str='manning',
                             white_colebrook_chzk2d: bool=False,
                             roughness_minimum_work_depth_m: float=0.,
                             defer_live_section_control: bool=False) -> dict|None:
    """Resolve user schedule knots to native total-Q and absolute-stage rows."""
    series=spec.hydraulic_series
    if series is None:return None
    targets=[]
    for index,(time_s,discharge) in enumerate(zip(series.times_s,series.discharge_m3_s)):
        if defer_live_section_control and spec.mode in ('inlet_normal','inlet_critical'):
            # A v21 production section recomputes its control level from the
            # live boundary bed at every SSPRK stage. Resolving thousands of
            # preparation knots here is both redundant and very expensive for
            # a densely sampled survey section. The serialized placeholder is
            # never used by the native control; the runtime section trace is
            # the authoritative level and is written to boundary_sections.txt.
            targets.append({'time_s':time_s,'discharge_m3_s':discharge,
                'requested_control_m':None,**_deferred_live_section_target(spec)})
            continue
        options={'hydraulic_series':None,'discharge_m3_s':discharge}
        requested_control=None if series.control_m is None else series.control_m[index]
        if spec.mode in ('discharge','prescribed_trace','inlet_stage','stage'):options['stage_m']=requested_control
        elif spec.mode in ('inlet_depth','prescribed_depth_trace'):options['depth_m']=requested_control
        knot=replace(spec,**options)
        target=resolve_boundary_target(
            knot,section,default_manning_n=default_manning_n,
            hydraulic_roughness_mode=hydraulic_roughness_mode,
            white_colebrook_chzk2d=white_colebrook_chzk2d,
            roughness_minimum_work_depth_m=roughness_minimum_work_depth_m)
        targets.append({'time_s':time_s,'discharge_m3_s':discharge,
                        'requested_control_m':requested_control,**target})
    return {'interpolation':'piecewise linear','outside_range':'hold endpoint',
            'control_kind':{'discharge':'water_level','total_discharge':'free_water_level','prescribed_trace':'water_level',
                'inlet_stage':'water_level',
                'prescribed_depth_trace':'water_depth','inlet_depth':'water_depth','inlet_normal':'normal_depth_from_discharge',
                'inlet_critical':'critical_depth_from_discharge','stage':'water_level'}[spec.mode],
            'knots':targets}


def _held_linear(times,values,time_s):
    if time_s<=times[0]:return values[0]
    if time_s>=times[-1]:return values[-1]
    # ``times`` is normally a tuple. np.searchsorted converted that entire
    # tuple to an array on every lookup, making cross-validation of long,
    # multi-fraction schedules quadratic in memory traffic. Binary-search the
    # already validated Python sequence directly.
    index=bisect_left(times,time_s)
    weight=(time_s-times[index-1])/(times[index]-times[index-1])
    return (1-weight)*values[index-1]+weight*values[index]


@dataclass(frozen=True)
class CutSediment:
    grain_density_kg_m3: float=2650.
    water_density_kg_m3: float=1000.
    porosity: float=.4
    d50_m: float=.0002
    d90_m: float=.0003
    z0_m: float=.001
    minimum_exchange_depth_m: float=.05
    minimum_bedload_depth_m: float=.05
    reference_height_factor: float=1.
    roughness_mode: str='fixed'

    def __post_init__(self):
        if self.roughness_mode not in ('fixed','manning'):
            raise ValueError('Sediment roughness_mode must be fixed or manning')
        numeric={name:getattr(self,name) for name in CutSediment.__dataclass_fields__ if name!='roughness_mode'}
        if any(isinstance(v,bool) or not math.isfinite(v) for v in numeric.values()):
            raise ValueError('Sediment properties must be finite numbers')
        if not 0<=self.porosity<1 or self.grain_density_kg_m3<=self.water_density_kg_m3:
            raise ValueError('Invalid porosity/density')
        if min(self.water_density_kg_m3,self.d50_m,self.d90_m,self.z0_m)<=0 or self.d90_m<self.d50_m or min(self.minimum_exchange_depth_m,self.minimum_bedload_depth_m)<0:
            raise ValueError('Invalid grain/roughness/depth properties')
        if self.reference_height_factor<=0:
            raise ValueError('Reference-height factor must be positive')

    @property
    def bulk_kg_m3(self):
        return self.grain_density_kg_m3*(1-self.porosity)


@dataclass(frozen=True)
class CutSedimentFraction(CutSediment):
    """One independently conserved suspended/bed class in a shared flow."""
    fraction_id: str='sand'
    settling_velocity_m_s: float=.05
    initial_volume_fraction: float=1.
    initial_active_volume_fraction: float|None=None
    sediment_type: str='sand'
    critical_erosion_shear_pa: float=.2
    critical_deposition_shear_pa: float=.1
    erosion_parameter_kg_m2_s: float=1.e-4
    erosion_exponent: float=1.

    def __post_init__(self):
        super().__post_init__()
        if self.sediment_type not in ('sand','mud'):
            raise ValueError("Fraction sediment type must be 'sand' or 'mud'")
        if not isinstance(self.fraction_id,str) or re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,31}',self.fraction_id) is None:
            raise ValueError('Sediment fraction ID must start with a letter and contain at most 32 letters, digits or underscores')
        if (isinstance(self.settling_velocity_m_s,bool) or not math.isfinite(self.settling_velocity_m_s) or
                self.settling_velocity_m_s<0):
            raise ValueError('Fraction settling velocity must be finite and nonnegative')
        if (isinstance(self.initial_volume_fraction,bool) or not math.isfinite(self.initial_volume_fraction) or
                self.initial_volume_fraction<0):
            raise ValueError('Initial fraction volume must be finite and nonnegative')
        if self.initial_active_volume_fraction is not None and (
                isinstance(self.initial_active_volume_fraction,bool) or
                not math.isfinite(self.initial_active_volume_fraction) or
                self.initial_active_volume_fraction<0):
            raise ValueError('Initial active-layer fraction volume must be finite and nonnegative')
        cohesive=(self.critical_erosion_shear_pa,self.critical_deposition_shear_pa,
                  self.erosion_parameter_kg_m2_s,self.erosion_exponent)
        if any(isinstance(value,bool) or not math.isfinite(value) for value in cohesive):
            raise ValueError('Cohesive parameters must be finite numbers')
        if self.critical_erosion_shear_pa<=0 or self.critical_deposition_shear_pa<0 or \
                self.erosion_parameter_kg_m2_s<0 or self.erosion_exponent<=0:
            raise ValueError('Invalid cohesive erosion/deposition parameters')


_NATIVE_BINARY_MAGIC = b"BRAIDS_NATIVE_V1"


def _write_binary_array(stream, values, dtype):
    if isinstance(values, (np.ndarray, list, tuple)):
        array = np.asarray(values, dtype=dtype)
    else:
        array = np.fromiter(values, dtype=dtype)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    array.tofile(stream)


def _write_native_binary(path, mesh, q, rock, local_manning, default_manning,
                         bank_mask, ids, version, production_section, sediment,
                         fractions, bed_layers, hierarchy, minimum_levels):
    """Write bulk mesh/tree arrays for one-read Fortran stream input.

    Controls and forcing remain in the inspectable text deck.  The arrays that
    grow with cell/face count use explicit little-endian 32-bit integers and
    64-bit floats, so neither Python formatting nor Fortran list-directed
    parsing sits on the large-case preparation path.
    """
    path = filesystem_path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    cell_count, face_count = len(mesh.cells), len(mesh.faces)
    state_count = q.shape[1]
    with temporary.open("wb") as stream:
        stream.write(_NATIVE_BINARY_MAGIC)
        _write_binary_array(stream, [1, cell_count, face_count, state_count,
            int(local_manning is not None), int(np.any(bank_mask)), version,
            int(production_section), int(hierarchy is not None)], "<i4")

        _write_binary_array(stream, (cell.i + 1 for cell in mesh.cells), "<i4")
        _write_binary_array(stream, (cell.j + 1 for cell in mesh.cells), "<i4")
        _write_binary_array(stream, (cell.area_m2 for cell in mesh.cells), "<f8")
        _write_binary_array(stream,
            (coordinate for cell in mesh.cells for coordinate in cell.centroid_xy), "<f8")
        _write_binary_array(stream, rock, "<f8")
        _write_binary_array(stream, q, "<f8")
        _write_binary_array(stream,
            (np.full(cell_count, default_manning) if local_manning is None else local_manning), "<f8")
        _write_binary_array(stream, np.asarray(bank_mask, dtype=np.int32), "<i4")

        _write_binary_array(stream, (face.owner + 1 for face in mesh.faces), "<i4")
        _write_binary_array(stream, (face.neighbour + 1 for face in mesh.faces), "<i4")
        _write_binary_array(stream, (ids.get(face.bc_id, 0) for face in mesh.faces), "<i4")
        _write_binary_array(stream,
            (component for face in mesh.faces for component in face.outward_normal), "<f8")
        _write_binary_array(stream, (face.length_m for face in mesh.faces), "<f8")
        _write_binary_array(stream,
            (coordinate for face in mesh.faces for coordinate in
             ((face.start_xy[0] + face.end_xy[0]) * .5,
              (face.start_xy[1] + face.end_xy[1]) * .5)), "<f8")
        _write_binary_array(stream,
            (int(face.bed_elevation_m is not None) for face in mesh.faces), "<i4")
        if version >= 11:
            if fractions is None:
                owner_bed = rock + q[:, 4] / sediment.bulk_kg_m3
            else:
                owner_bed = rock.copy()
                for index, fraction in enumerate(fractions):
                    owner_bed += sum(
                        q[:, 3 + len(fractions) + layer * len(fractions) + index]
                        for layer in range(bed_layers)
                    ) / fraction.bulk_kg_m3
        else:
            owner_bed = rock
        _write_binary_array(stream,
            (0. if face.bed_elevation_m is None else
             face.bed_elevation_m - owner_bed[face.owner] for face in mesh.faces), "<f8")

        def endpoint_offsets(face):
            if not production_section or face.section_bed_start_m is None:
                return 0., 0.
            left, right = _canonical_section_endpoints(face)
            return left - owner_bed[face.owner], right - owner_bed[face.owner]

        if production_section:
            endpoints = np.fromiter(
                (value for face in mesh.faces for value in endpoint_offsets(face)),
                dtype="<f8", count=2 * face_count).reshape(face_count, 2)
            _write_binary_array(stream, endpoints[:, 0], "<f8")
            _write_binary_array(stream, endpoints[:, 1], "<f8")
        else:
            zeros = np.zeros(face_count, dtype="<f8")
            _write_binary_array(stream, zeros, "<f8")
            _write_binary_array(stream, zeros, "<f8")

        if hierarchy is not None:
            minimum = (np.zeros(hierarchy.count, dtype=np.int32) if minimum_levels is None
                       else np.asarray(minimum_levels, dtype=np.int32))
            _write_binary_array(stream, [1], "<i4")
            _write_binary_array(stream, hierarchy.parent + 1, "<i4")
            _write_binary_array(stream, hierarchy.node_level, "<i4")
            for attribute in ("i", "j"):
                _write_binary_array(stream, (
                    getattr(cell, attribute) * hierarchy.refinement_ratio **
                    (hierarchy.finest_level - level) + 1
                    for level, level_mesh in enumerate(hierarchy.meshes)
                    for cell in level_mesh.cells), "<i4")
            _write_binary_array(stream, (
                hierarchy.refinement_ratio ** (hierarchy.finest_level - level)
                for level, level_mesh in enumerate(hierarchy.meshes)
                for _ in level_mesh.cells), "<i4")
            _write_binary_array(stream,
                (cell.area_m2 for level_mesh in hierarchy.meshes for cell in level_mesh.cells), "<f8")
            _write_binary_array(stream,
                (coordinate for level_mesh in hierarchy.meshes for cell in level_mesh.cells
                 for coordinate in cell.centroid_xy), "<f8")
            _write_binary_array(stream, minimum, "<i4")
            _write_binary_array(stream, hierarchy.maximum_level, "<i4")

            _write_binary_array(stream, (
                hierarchy.offsets[level] + face.owner + 1
                for level, level_mesh in enumerate(hierarchy.meshes)
                for face in level_mesh.faces), "<i4")
            _write_binary_array(stream, (
                0 if face.neighbour < 0 else hierarchy.offsets[level] + face.neighbour + 1
                for level, level_mesh in enumerate(hierarchy.meshes)
                for face in level_mesh.faces), "<i4")
            _write_binary_array(stream, (
                ids.get(face.bc_id, 0) for level_mesh in hierarchy.meshes
                for face in level_mesh.faces), "<i4")
            _write_binary_array(stream, (
                level for level, level_mesh in enumerate(hierarchy.meshes)
                for _ in level_mesh.faces), "<i4")
            _write_binary_array(stream, (
                face.face_id + 1 if level == hierarchy.finest_level else 0
                for level, level_mesh in enumerate(hierarchy.meshes)
                for face in level_mesh.faces), "<i4")
            _write_binary_array(stream, (
                int(face.boundary_role == "solid_obstacle")
                for level_mesh in hierarchy.meshes for face in level_mesh.faces), "<i4")
            _write_binary_array(stream, (
                component for level_mesh in hierarchy.meshes for face in level_mesh.faces
                for component in face.outward_normal), "<f8")
            _write_binary_array(stream, (
                face.length_m for level_mesh in hierarchy.meshes for face in level_mesh.faces), "<f8")
            _write_binary_array(stream, (
                coordinate for level_mesh in hierarchy.meshes for face in level_mesh.faces
                for coordinate in ((face.start_xy[0] + face.end_xy[0]) * .5,
                                   (face.start_xy[1] + face.end_xy[1]) * .5)), "<f8")
            _write_binary_array(stream, (
                int(face.bed_elevation_m is not None)
                for level_mesh in hierarchy.meshes for face in level_mesh.faces), "<i4")
            for name in ("bed_elevation_m", "section_bed_start_m", "section_bed_end_m"):
                _write_binary_array(stream, (
                    (0. if getattr(face, name) is None else
                     getattr(face, name) if name == "bed_elevation_m" else
                     _canonical_section_endpoints(face)[0 if name == "section_bed_start_m" else 1])
                    for level_mesh in hierarchy.meshes for face in level_mesh.faces), "<f8")
    _replace_with_windows_retry(temporary, path)
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_cut_case(path,mesh,initial,bedrock,boundaries,*,end_time_s,dt_max_s=.01,output_interval_s=.5,cfl=.45,
                   scheme=5,bedload=1,exchange=1,settling_velocity_m_s=None,manning=0.,source_sha256,
                   manning_by_cell=None,sections=None,dry_tolerance_m=NATIVE_DRY_TOLERANCE_M,
                   hydraulic_roughness_mode='manning',white_colebrook_chzk2d=False,
                   sediment_shear_mode='total',skin_kinematic_viscosity_m2_s=1.e-6,
                   skin_sand_roughness_m=.003,skin_silt_roughness_m=.0003,
                   skin_critical_mud_fraction_low=.2,skin_critical_mud_fraction_high=.8,
                   skin_fluff_fraction=0.,
                   horizontal_viscosity_m2_s=0.,horizontal_viscosity_formulation='depth_conservative',
                   manning_friction_chezy_minimum_depth_m=0.,
                   sediment=None,sediment_fractions=None,mpm_cf=.01,mpm_critical_shields=0.,mpm_friction_mode=0,bed_slope=0,
                   engelund_hansen_chezy=None,
                   gaia_skin_roughness_ratio=3.,gaia_slope_beta=1.3,
                   gaia_deviation_beta2=.85,gaia_secondary_alpha=1.,
                   gaia_diagnostic_v44=False,
                   gaia_interior_slope_talmon_enabled=None,
                   gaia_interior_secondary_stress_enabled=None,
                   gaia_open_boundary_secondary_stress_enabled=None,
                   bed_slope_alfabs=1.,bed_slope_alfabn=1.5,bed_friction_angle_deg=30.,
                   active_layer_thickness_m=0.,active_layer_mode='constant',
                   active_layer_depth_fraction=0.,active_layer_minimum_thickness_m=None,hiding_exposure=0,
                   hiding_parker_alpha=.6,hiding_wu_exponent=-.6,
                   substrate_layer_count=1,substrate_layer_thickness_m=None,
                   critical_mud_fraction=.5,transport_calibration_factor=1.,
                   settling_mode=0,
                   hindered_settling_reference_concentration_kg_m3=DEFAULT_HINDERED_REFERENCE_CONCENTRATION_KG_M3,
                   settling_formula=None,settling_fac_dss=1.,
                   settling_temperature_mode='equilibrium_constant',settling_temperature_c=20.,
                   settling_reference_concentration_kg_m3=None,
                   suspended_calibration=1.,
                   bank_erodible_by_cell=None,bank_erosion_fraction=0.,bank_avalanche=False,
                   bank_wet_slope=10.,bank_dry_slope=1.,bank_avalanche_time_s=86400.,
                   bank_wet_depth_switch_m=.1,bank_relaxation_length_m=1.,
                   general_avalanche=False,general_avalanche_wet_slope=10.,
                   general_avalanche_dry_slope=1.,general_avalanche_time_s=86400.,
                   general_avalanche_wet_depth_switch_m=.1,
                   general_avalanche_relaxation_length_m=1.,
                   secondary_flow=False,secondary_flow_espir=1.,
                   secondary_flow_model='equilibrium',
                   secondary_flow_transport_direction=True,
                   secondary_flow_momentum_beta=0.,
                   bank_failure=False,bank_toe_erodibility_m_pa_s=0.,
                   bank_toe_critical_shear_pa=0.,bank_toe_exponent=1.,
                   bank_critical_undercut_m=0.,
                   bank_toe_erosion_mode='cohesive_excess_shear',
                   bank_toe_capacity_fraction=1.,
                   bank_distributed_transport_fraction=.1,
                   bank_distributed_critical_slope=.35,
                   bank_front_tracking=False,
                    bedload_adaptation=False,bedload_adaptation_length_diameters=100.,
                    morphological_factor=1.,morphological_factor_series=None,
                   bedform_roughness=False,
                   amr_hierarchy=None, amr_settings=None,
                   amr_minimum_levels=None,compact_sidecars=False,
                   sediment_catalog_id=None,
                   capacity_parameters_by_fraction=None,
                   capacity_output_contract='bed_only',
                   constitutive_kinematic_viscosity_m2_s=1.e-6,
                   capacity_fraction_basis=None,
                   suspended_adaptation_factor=1.,
                   frozen_pre_m2c_suspension=False,
                    suspended_neubc_sand=None,
                    suspended_horizontal_diffusivity_m2_s=None,
                    suspended_diffusion_dry_threshold_m=None,
                    capacity_method_id=None,
                    capacity_execution_semantic_id_value=None,
                    capacity_execution_fingerprint=None,
                    capacity_runtime_profile_semantic_id=None,
                    capacity_profile_fingerprint=None,
                     capacity_profile_acal=None,
                      channel_composition_profile=None,
                      channel_composition_wire_version=51,
                      execution_profile_v2=None,
                      execution_runtime_backend='cpu',
                      capacity_runtime_profile=None,
                      capacity_foundation_execution_profile=None,
                      capacity_forcing_profile=None,
                      hiding_foundation_execution_profile=None,
                      secondary_foundation_execution_profile=None,
                      prognostic_secondary_identity=None):
    """Write an explicit cut-cell run with common IDs for all conserved fluxes.

    A legacy single material uses columns h,hu,hv,S,Mbed. A multi-fraction
    run uses h,hu,hv,S_1..S_n,Mbed_1..Mbed_n. An active-layer run stores
    Mactive_1..Mactive_n followed by one or more ordered substrate layers. The physical
    bed elevation is bedrock plus the sum of all bed inventories divided by
    their bulk densities.
    Initial values are control-volume averages and densities per retained area.

    ``bed_slope`` 2 keeps the legacy parameter meanings ``AlfaBs``,
    ``AlfaBn``, and friction angle (degrees).  Mode 3 is the published
    Van Bendegom/Koch--Flokstra specialization with ``CShld=DShld=0``; the
    same three stored fields mean ``AlfaBs``, ``AShld``, and ``BShld``.
    Nonzero CShld/DShld are intentionally not represented by this deck. Mode
    4 is the distinct TELEMAC-GAIA v8p5r1 DEVIA=2/SLOPEFF=1 contract and is
    admitted only with law 9 and secondary model ``gaia``.

    ``gaia_diagnostic_v44`` is an explicitly opt-in validation envelope.  It
    can remove the interior Talmon/slope term and interior or open-boundary
    secondary-current stress independently.  The three switches are absent
    from ordinary decks so every established v43 byte remains unchanged.

    Delft3D current-capacity laws 11--14 are introduced only through the v46
    outer envelope.  Every fraction carries the complete, source-ordered
    parameter row and ``capacity_output_contract`` must be
    ``'retain_bed_and_suspended'``.  The CPU M1 path retains both formula
    outputs and writes them to ``capacity_split.txt``.  Boundary trace and
    reversal evaluations are retained separately in
    ``capacity_split_boundary.txt``; only the bed channel is connected to
    Exner until the conservative suspended-capacity route is completed in M2.
    A bed-only request therefore fails closed instead of silently dropping or
    adding the suspended channel.  Version 47 is an opt-in typed constitutive
    envelope.  It binds molecular viscosity, FLOW ``IFractions`` basis,
    diagnostic applicability and the generic FLOW 2DH exchange identity.
    Exchange mode 4 is initially admitted only for a one-fraction law-11
    slice with nonzero ``SusACal``; modes 2 and 4 remain distinct closures.
    Supplying ``suspended_neubc_sand`` selects the v48 FLOW-2DH open-boundary
    and horizontal-diffusion route.  ``None`` preserves the frozen v47 route;
    an explicit false value is therefore the prescribed-concentration choice,
    not an implicit legacy fallback.

    New exchange-mode-4 cases are wrapped by v49, which binds conservative
    wet/dry ownership and, when present, AMR transfer/reflux semantics.  The
    ``frozen_pre_m2c_suspension`` switch exists only to reproduce archived
    v47/v48 fixed-grid evidence; it is rejected with AMR and must never be
    selected implicitly by a production preparation path.

    Version 50 is an additive execution-identity envelope for the two
    historically ambiguous integer selectors (laws 2 and 8).  Its canonical
    byte order is closure-execution identity first and capacity-profile
    identity second.  It does not reinterpret an omitted legacy envelope.
    Corrected MPM is admitted only as law 2 / mode 1 with both authenticated
    fingerprints; corrected Delft3D Engelund--Hansen remains fail-closed until
    its selected-hydraulic-Chezy kernel is connected.

    Version 51 is an opt-in, authenticated law-11 channel-composition
    envelope.  Its complete canonical profile must be supplied explicitly;
    ordinary preparation does not infer or silently activate the route.  The
    profile binds the ordered FLOW transport classes, model-global ``Bed`` and
    ``Sus`` factors and denominator epsilon.  The prior law-11 v46 or v49
    chain remains nested byte-for-byte.  Version 50 remains exclusively the
    law-2/law-8 capacity-execution identity and is neither required nor
    reinterpreted by version 51.

    Version 52 carries the exact canonical ASCII preimages for the channel
    and schema-v2 execution profiles, but component documents remain
    hash-only bindings. Version 53 is the public execution envelope: it wraps
    v52 and carries the exact sorted canonical component-document preimages so
    native code can recompute their hashes and cross-check their contents
    against the operational deck. Existing v51/v52 byte streams are unchanged.
    Version 54 is an additive Q3-C envelope around v53. It carries the exact
    typed-forcing and capacity-foundation execution preimages so native code
    can authenticate the raw per-cell FP64 C/n/ks bytes, all friction/shear
    controls, and their ordered execution-to-capacity-profile dependency.
    Historical v53 Q3-S/P byte streams are not reinterpreted.

    Version 55 is an additive Q3-H1 envelope around one complete v54 deck. It
    authenticates the generic hiding/exposure operator, its two ordered
    fractions, active-surface ownership and the exact nested Q3-C execution
    and capacity-profile fingerprints. Historical v54 decks remain unchanged.

    Version 56 is an additive Q3-H2 envelope around one complete v54 or v55
    deck.  It authenticates classic Delft3D4 FLOW 2DH equilibrium secondary
    flow without reinterpreting the nested foundation's ``secondary.off``
    declaration.  The first live native route is fixed-grid, zero-Coriolis,
    zero-Betac and rotates only the raw bed channel before composition.

    Version 57 is the sibling Q3-H3 envelope around one complete v54 or v55
    deck.  It reuses the authenticated classic-FLOW curvature, target and
    typed-roughness primitives, but never executes v56's equilibrium bed
    rotation.  One signed transported spiral-intensity state owns the sole
    raw-bed direction rotation.  Historical v37 prognostic decks retain their
    original wire and semantics.

    The schema-v2 physical profile is backend-neutral. The separate
    ``execution_runtime_backend`` receipt field records the user-selected CPU
    or implicit-fwave GPU runtime without changing that physical identity.

    The unchanged v51 envelope remains nested as the operational compatibility
    declaration; callers that intentionally need the internal host-attested
    format must request ``channel_composition_wire_version=51`` explicitly.
    """
    resolved_capacity_runtime_profile=None
    resolved_capacity_runtime_document=None
    resolved_capacity_foundation_execution_document=None
    resolved_capacity_forcing_document=None
    resolved_capacity_foundation_execution_preimage=None
    resolved_capacity_forcing_preimage=None
    resolved_hiding_foundation_execution_document=None
    resolved_hiding_foundation_execution_preimage=None
    resolved_secondary_foundation_execution_document=None
    resolved_secondary_foundation_execution_preimage=None
    resolved_prognostic_secondary_execution_document=None
    resolved_prognostic_secondary_execution_preimage=None
    if capacity_runtime_profile is not None:
        try:
            if isinstance(capacity_runtime_profile,Mapping):
                resolved_capacity_runtime_profile=load_capacity_profile(
                    capacity_runtime_profile)
            else:
                raise CapacityProfileError(
                    'capacity runtime profile must be a canonical mapping',
                    category='invalid_configuration')
            resolved_capacity_runtime_document=capacity_profile_document(
                resolved_capacity_runtime_profile)
        except CapacityProfileError as error:
            raise ValueError(f'Invalid native capacity runtime profile: {error}') from error
        if resolved_capacity_runtime_profile.backend is not None:
            raise ValueError(
                'Native capacity runtime profile must remain backend neutral')
    if ((capacity_foundation_execution_profile is None) !=
            (capacity_forcing_profile is None)):
        raise ValueError(
            'Q3-C native identity requires both forcing and execution documents')
    if capacity_foundation_execution_profile is not None:
        try:
            resolved_capacity_foundation_execution_document=(
                load_capacity_foundation_execution(
                    capacity_foundation_execution_profile))
            validate_typed_roughness_binding(capacity_forcing_profile)
        except (TypeError,ValueError) as error:
            raise ValueError(
                f'Invalid Q3-C forcing/execution document: {error}') from error
        resolved_capacity_forcing_document=dict(capacity_forcing_profile)
        resolved_capacity_foundation_execution_preimage=json.dumps(
            resolved_capacity_foundation_execution_document,sort_keys=True,
            separators=(',',':'),ensure_ascii=True,allow_nan=False)
        resolved_capacity_forcing_preimage=json.dumps(
            resolved_capacity_forcing_document,sort_keys=True,
            separators=(',',':'),ensure_ascii=True,allow_nan=False)
    if hiding_foundation_execution_profile is not None:
        try:
            resolved_hiding_foundation_execution_document=(
                validate_hiding_foundation_execution_document(
                    hiding_foundation_execution_profile))
        except (TypeError,ValueError) as error:
            raise ValueError(
                f'Invalid Q3-H1 hiding execution document: {error}') from error
        resolved_hiding_foundation_execution_preimage=json.dumps(
            resolved_hiding_foundation_execution_document,sort_keys=True,
            separators=(',',':'),ensure_ascii=True,allow_nan=False)
    if secondary_foundation_execution_profile is not None:
        try:
            resolved_secondary_foundation_execution_document=(
                validate_secondary_foundation_execution_document(
                    secondary_foundation_execution_profile))
        except (TypeError,ValueError) as error:
            raise ValueError(
                f'Invalid Q3-H2 secondary execution document: {error}') from error
        resolved_secondary_foundation_execution_preimage=json.dumps(
            resolved_secondary_foundation_execution_document,sort_keys=True,
            separators=(',',':'),ensure_ascii=True,allow_nan=False)
    if prognostic_secondary_identity is not None:
        try:
            if (not isinstance(prognostic_secondary_identity,Mapping) or
                    set(prognostic_secondary_identity)!={
                        PROGNOSTIC_SECONDARY_EXECUTION_KEY}):
                raise ValueError(
                    'Q3-H3 native identity must contain exactly its canonical execution document')
            resolved_prognostic_secondary_execution_document=(
                validate_prognostic_secondary_execution_document(
                    prognostic_secondary_identity[
                        PROGNOSTIC_SECONDARY_EXECUTION_KEY]))
        except (TypeError,ValueError) as error:
            raise ValueError(
                f'Invalid Q3-H3 prognostic-secondary execution document: {error}'
            ) from error
        resolved_prognostic_secondary_execution_preimage=json.dumps(
            resolved_prognostic_secondary_execution_document,sort_keys=True,
            separators=(',',':'),ensure_ascii=True,allow_nan=False)
    if (resolved_secondary_foundation_execution_document is not None and
            resolved_prognostic_secondary_execution_document is not None):
        raise ValueError(
            'Q3-H2 equilibrium and Q3-H3 prognostic outer identities are mutually exclusive')
    if execution_runtime_backend not in ('cpu','implicit_fwave_midpoint_gpu'):
        raise ValueError(
            "Execution runtime backend must be 'cpu' or "
            "'implicit_fwave_midpoint_gpu'")
    if (execution_runtime_backend!='cpu' and execution_profile_v2 is None):
        raise ValueError(
            'A non-CPU execution runtime backend requires an authenticated '
            'schema-v2 execution profile')
    capacity_identity_text_values=(
        capacity_method_id,capacity_execution_semantic_id_value,
        capacity_execution_fingerprint,capacity_runtime_profile_semantic_id,
        capacity_profile_fingerprint)
    capacity_identity_declared=(
        any(value is not None for value in capacity_identity_text_values) or
        capacity_profile_acal is not None)
    if capacity_identity_declared and (
            not all(isinstance(value,str) and value
                    for value in capacity_identity_text_values) or
            capacity_profile_acal is None):
        raise ValueError(
            'A native capacity identity requires method ID, both semantic IDs, '
            'and both instance fingerprints')
    if capacity_identity_declared:
        foundation_execution_identity=(
            capacity_execution_semantic_id_value==
            CAPACITY_FOUNDATION_EXECUTION_SEMANTIC_ID)
        if capacity_execution_semantic_id_value not in (
                capacity_execution_semantic_id(),
                CAPACITY_FOUNDATION_EXECUTION_SEMANTIC_ID):
            raise ValueError('Capacity execution semantic identity differs')
        if capacity_runtime_profile_semantic_id!=capacity_profile_semantic_id():
            raise ValueError('Capacity profile semantic identity differs')
        for name,value in (
                ('Capacity execution fingerprint',capacity_execution_fingerprint),
                ('Capacity profile fingerprint',capacity_profile_fingerprint)):
            if re.fullmatch(r'[0-9a-f]{64}',value) is None:
                raise ValueError(f'{name} must be a lowercase SHA-256 digest')
        if (isinstance(capacity_profile_acal,bool) or
                not isinstance(capacity_profile_acal,(int,float)) or
                not math.isfinite(capacity_profile_acal) or capacity_profile_acal<0):
            raise ValueError('Capacity profile ACal must be finite and nonnegative')
        expected={
            EH_LEGACY_METHOD_ID:(8,0),
            EH_DELFT_METHOD_ID:(8,0),
            MPM_LEGACY_METHOD_ID:(2,0),
            MPM_DELFT_METHOD_ID:(2,1),
            'capacity.general':(11,0),
        }
        if capacity_method_id not in expected:
            raise ValueError('Unsupported native capacity execution method identity')
        expected_law,expected_mpm_mode=expected[capacity_method_id]
        if bedload!=expected_law or mpm_friction_mode!=expected_mpm_mode:
            raise ValueError(
                'Capacity method identity disagrees with bed-load law or MPM friction mode')
        if (capacity_method_id==MPM_DELFT_METHOD_ID and
                not math.isclose(mpm_critical_shields,.047,rel_tol=0.,abs_tol=1.e-14)):
            raise ValueError('Corrected Delft MPM requires critical Shields 0.047')
        corrected_method=capacity_method_id in (
            MPM_DELFT_METHOD_ID,EH_DELFT_METHOD_ID,'capacity.general')
        if corrected_method and resolved_capacity_runtime_profile is not None:
            profile_methods={
                value.method_id for value in
                resolved_capacity_runtime_profile.fractions}
            if profile_methods!={capacity_method_id}:
                raise ValueError(
                    'Capacity execution identity differs from the canonical '
                    'capacity runtime profile')
            profile_acal=[]
            for fraction in resolved_capacity_runtime_profile.fractions:
                coefficients={value.keyword:value.value
                              for value in fraction.coefficients}
                profile_acal.append(coefficients.get('ACal'))
            if (any(value is None or isinstance(value,bool) or
                    not isinstance(value,(int,float)) or
                    not math.isfinite(float(value)) or float(value)<0.
                    for value in profile_acal) or
                    float(profile_acal[0])!=float(capacity_profile_acal)):
                raise ValueError(
                    'Capacity-profile v50 ACal sentinel differs from the first '
                    'canonical runtime fraction')
        elif capacity_method_id==EH_DELFT_METHOD_ID:
            raise ValueError(
                'Corrected Delft Engelund-Hansen requires the authenticated '
                'schema-v2 capacity runtime profile')
        elif capacity_method_id in (MPM_LEGACY_METHOD_ID,MPM_DELFT_METHOD_ID):
            if capacity_profile_acal!=1.:
                raise ValueError(
                    'The historical native MPM route requires capacity-profile ACal=1')
        elif capacity_method_id=='capacity.general':
            raise ValueError(
                'General Q3-C capacity requires the authenticated schema-v2 '
                'capacity runtime profile')
        elif capacity_profile_acal!=transport_calibration_factor:
            raise ValueError(
                'Engelund-Hansen capacity-profile ACal differs from native calibration')
        if foundation_execution_identity and resolved_capacity_runtime_profile is None:
            raise ValueError(
                'Q3-C forcing execution identity requires a schema-v2 capacity profile')
        if (capacity_method_id==MPM_DELFT_METHOD_ID and hiding_exposure!=0 and
                resolved_hiding_foundation_execution_document is None):
            raise ValueError(
                'Corrected Delft MPM generic hiding remains unavailable in the Q3-0 route')
    if (isinstance(constitutive_kinematic_viscosity_m2_s,bool) or
            not math.isfinite(constitutive_kinematic_viscosity_m2_s) or
            constitutive_kinematic_viscosity_m2_s<=0):
        raise ValueError('Constitutive kinematic viscosity must be positive and finite (m2/s)')
    if capacity_fraction_basis not in (None,'dry_mass','solid_volume'):
        raise ValueError("Capacity fraction basis must be 'dry_mass', 'solid_volume', or None")
    if (isinstance(suspended_adaptation_factor,bool) or
            not math.isfinite(suspended_adaptation_factor) or
            suspended_adaptation_factor<=0):
        raise ValueError('Suspended adaptation factor FacTsd must be positive and finite')
    if not isinstance(frozen_pre_m2c_suspension,bool):
        raise ValueError('frozen_pre_m2c_suspension must be Boolean')
    if frozen_pre_m2c_suspension and amr_hierarchy is not None:
        raise ValueError(
            'The frozen pre-M2-C suspension route is fixed-grid only')
    flow2dh_boundary_diffusion_v48=(
        (exchange==4 and not frozen_pre_m2c_suspension) or
        suspended_neubc_sand is not None or
        suspended_horizontal_diffusivity_m2_s is not None or
        suspended_diffusion_dry_threshold_m is not None or
        any(getattr(boundary,'suspended_return_time_s',0.) != 0.
            for boundary in boundaries.values()))
    if flow2dh_boundary_diffusion_v48 and suspended_neubc_sand is None:
        raise ValueError(
            'FLOW 2DH suspended boundary/diffusion controls require an explicit '
            'NeuBcSand Boolean; true is the locked FLOW sand default and false '
            'selects prescribed concentration')
    if suspended_neubc_sand is not None and not isinstance(
            suspended_neubc_sand,bool):
        raise ValueError('NeuBcSand must be an explicit Boolean or None')
    resolved_neubc_sand=(False if suspended_neubc_sand is None else
                         suspended_neubc_sand)
    for name,boundary in boundaries.items():
        if boundary.reversal_policy=='flow2dh_return':
            if not flow2dh_boundary_diffusion_v48:
                raise ValueError(
                    f"Boundary {name!r} FLOW return-time reversal requires "
                    "the explicit v48 boundary/diffusion route")
            if boundary.suspended_return_time_s<=0.:
                raise ValueError(
                    f"Boundary {name!r} FLOW return-time reversal requires "
                    "a positive suspended return time")
    resolved_suspended_diffusivity=(
        0. if suspended_horizontal_diffusivity_m2_s is None else
        suspended_horizontal_diffusivity_m2_s)
    resolved_suspended_diffusion_dry_threshold=(
        dry_tolerance_m if suspended_diffusion_dry_threshold_m is None else
        suspended_diffusion_dry_threshold_m)
    if flow2dh_boundary_diffusion_v48:
        for name,value in (
                ('Suspended horizontal diffusivity',resolved_suspended_diffusivity),
                ('Suspended diffusion DryFlc',
                 resolved_suspended_diffusion_dry_threshold)):
            if (isinstance(value,bool) or not isinstance(value,(int,float)) or
                    not math.isfinite(value) or value<0):
                raise ValueError(f'{name} must be finite and nonnegative')
    if (isinstance(horizontal_viscosity_m2_s,bool) or not math.isfinite(horizontal_viscosity_m2_s)
            or horizontal_viscosity_m2_s<0):
        raise ValueError('Horizontal viscosity must be finite and nonnegative (m2/s)')
    if (sediment_catalog_id is not None and
            (not isinstance(sediment_catalog_id,str) or
             re.fullmatch(r'[0-9a-f]{64}',sediment_catalog_id) is None)):
        raise ValueError('Sediment method catalogue identity must be a SHA-256 digest')
    roughness_codes={'chezy':1,'manning':2,'white_colebrook':3}
    if hydraulic_roughness_mode not in roughness_codes:
        raise ValueError('Hydraulic roughness mode must be chezy, manning, or white_colebrook')
    if not isinstance(white_colebrook_chzk2d,bool):
        raise ValueError('White-Colebrook ChzK2D switch must be true or false')
    if white_colebrook_chzk2d and hydraulic_roughness_mode!='white_colebrook':
        raise ValueError('ChzK2D is valid only with White-Colebrook roughness')
    if sediment_shear_mode not in ('total','soulsby_clarke_skin'):
        raise ValueError('Sediment shear mode must be total or soulsby_clarke_skin')
    skin_values=(skin_kinematic_viscosity_m2_s,skin_sand_roughness_m,
                 skin_silt_roughness_m,skin_critical_mud_fraction_low,
                 skin_critical_mud_fraction_high,skin_fluff_fraction)
    if any(isinstance(value,bool) or not math.isfinite(value) for value in skin_values):
        raise ValueError('Sediment skin-shear parameters must be finite numbers')
    if (min(skin_kinematic_viscosity_m2_s,skin_sand_roughness_m,
            skin_silt_roughness_m)<=0 or
            not 0<=skin_critical_mud_fraction_low<skin_critical_mud_fraction_high<=1 or
            not 0<=skin_fluff_fraction<=1):
        raise ValueError('Invalid Soulsby/Clarke skin-shear parameters')
    if horizontal_viscosity_formulation not in ('depth_conservative','velocity_laplacian'):
        raise ValueError('Horizontal viscosity formulation must be depth_conservative or velocity_laplacian')
    if horizontal_viscosity_formulation=='velocity_laplacian' and horizontal_viscosity_m2_s<=0:
        raise ValueError('The velocity_laplacian formulation requires positive horizontal viscosity')
    equilibrated_inlets=tuple(name for name,spec in boundaries.items()
                              if spec.total_discharge_profile=='viscous_manning_equilibrium')
    if equilibrated_inlets and (horizontal_viscosity_formulation!='velocity_laplacian' or
                                horizontal_viscosity_m2_s<=0):
        raise ValueError('A viscosity-equilibrated total-discharge profile requires positive velocity_laplacian viscosity')
    if (isinstance(manning_friction_chezy_minimum_depth_m,bool) or
            not math.isfinite(manning_friction_chezy_minimum_depth_m) or
            manning_friction_chezy_minimum_depth_m<0):
        raise ValueError('Manning friction Chezy minimum depth must be finite and nonnegative (m)')
    if isinstance(dry_tolerance_m,bool) or not math.isfinite(dry_tolerance_m) or dry_tolerance_m<=0:
        raise ValueError('Hydraulic dry tolerance must be finite and positive')
    for spec in boundaries.values():
        if spec.mode=='prescribed_depth_trace':
            discharge=spec.hydraulic_series.discharge_m3_s if spec.hydraulic_series else (spec.discharge_m3_s,)
            depths=spec.hydraulic_series.control_m if spec.hydraulic_series else (spec.depth_m,)
            if any(q>0 and h<=dry_tolerance_m for q,h in zip(discharge,depths)):
                raise ValueError('A prescribed depth trace cannot carry positive discharge at the configured dry depth')
    fractions=None if sediment_fractions is None else tuple(sediment_fractions)
    # Native envelopes >=15 carry one boundary-series count per physical
    # fraction.  A Q3-C single-reservoir MPM deck can legitimately reach that
    # envelope through its roughness-corrected identity even though it uses
    # the compact one-material state layout, so keep an explicit canonical
    # identifier for that one physical fraction as well.
    fraction_ids=['single'] if fractions is None else None
    if fractions is not None:
        if len(fractions)<2 or len(fractions)>32 or not all(isinstance(value,CutSedimentFraction) for value in fractions):
            raise ValueError('Multi-fraction runs require 2 to 32 CutSedimentFraction values')
        fraction_ids=[value.fraction_id for value in fractions]
        if len(set(fraction_ids))!=len(fraction_ids):raise ValueError('Sediment fraction IDs must be unique')
        if not math.isclose(sum(value.initial_volume_fraction for value in fractions),1.,rel_tol=0,abs_tol=1e-12):
            raise ValueError('Initial sediment volume fractions must sum to one')
        active_composition=[value.initial_volume_fraction if value.initial_active_volume_fraction is None
                            else value.initial_active_volume_fraction for value in fractions]
        if not math.isclose(sum(active_composition),1.,rel_tol=0,abs_tol=1e-12):
            raise ValueError('Initial active-layer volume fractions must sum to one')
        if any(value.water_density_kg_m3!=fractions[0].water_density_kg_m3 for value in fractions[1:]):
            raise ValueError('All sediment fractions must use the same water density')
        if sediment is not None:raise ValueError('sediment_fractions replaces the single sediment material')
        sediment=CutSediment(**{name:getattr(fractions[0],name) for name in CutSediment.__dataclass_fields__})
    else:
        sediment=CutSediment() if sediment is None else sediment
    if not isinstance(sediment,CutSediment):raise ValueError('CutSediment is required')
    if settling_reference_concentration_kg_m3 is not None:
        if (hindered_settling_reference_concentration_kg_m3 !=
                DEFAULT_HINDERED_REFERENCE_CONCENTRATION_KG_M3 and
                not math.isclose(
                    settling_reference_concentration_kg_m3,
                    hindered_settling_reference_concentration_kg_m3,
                    rel_tol=0.,abs_tol=0.)):
            raise ValueError(
                'The legacy and typed settling reference concentrations disagree')
        hindered_settling_reference_concentration_kg_m3 = \
            settling_reference_concentration_kg_m3
    if (isinstance(hindered_settling_reference_concentration_kg_m3,bool) or
            not math.isfinite(hindered_settling_reference_concentration_kg_m3) or
            hindered_settling_reference_concentration_kg_m3<=0):
        raise ValueError('Settling reference concentration must be positive and finite')
    if (isinstance(settling_fac_dss,bool) or not math.isfinite(settling_fac_dss) or
            settling_fac_dss<=0):
        raise ValueError('Settling FacDSS must be positive and finite')
    if (isinstance(settling_temperature_c,bool) or
            not math.isfinite(settling_temperature_c) or settling_temperature_c<=-20.):
        raise ValueError('Settling temperature must be finite and exceed -20 C')
    if (isinstance(suspended_calibration,bool) or
            not math.isfinite(suspended_calibration) or suspended_calibration<=0):
        raise ValueError('Suspended calibration Sus must be positive and finite')
    settling_form_code=0
    settling_temperature_mode_code=0
    sediment_activity_contract_code=1
    wet_mask_provider_code=1
    resolved_settling_temperature_c=0.
    if settling_formula is None:
        if settling_velocity_m_s is None:
            settling_velocity_m_s=.05
        if (settling_fac_dss!=1. or
                settling_temperature_mode!='equilibrium_constant' or
                settling_temperature_c!=20. or suspended_calibration!=1. or
                settling_reference_concentration_kg_m3 is not None):
            raise ValueError(
                'Typed settling controls require '
                "settling_formula='delft_function_of_diameter'")
        settling_clear_water_velocity_m_s=float(settling_velocity_m_s)
    elif settling_formula=='delft_function_of_diameter':
        if fractions is not None:
            raise ValueError(
                'The initial typed SettleFrm=2 CPU slice requires one sediment fraction')
        if settling_velocity_m_s is not None:
            raise ValueError(
                'SettleFrm=2 derives clear-water settling velocity; do not supply '
                'settling_velocity_m_s')
        if settling_temperature_mode!='equilibrium_constant':
            raise ValueError(
                'The initial typed SettleFrm=2 CPU slice requires '
                "settling_temperature_mode='equilibrium_constant'")
        settling_form_code=FLOW2DH_SETTLING_FORM
        settling_temperature_mode_code=1
        resolved_settling_temperature_c=float(settling_temperature_c)
        try:
            settling_result=settlefrm2_noncohesive(
                np.zeros((1,1)),
                np.array([sediment.d50_m]),
                np.array([settling_fac_dss]),
                np.array([sediment.grain_density_kg_m3]),
                sediment.water_density_kg_m3,
                resolved_settling_temperature_c,
                settling_reference_concentration_kg_m3=
                    hindered_settling_reference_concentration_kg_m3,
                gravity_m_s2=9.81,
                wet_mask=np.array([True]),
            )
        except ValueError as error:
            raise ValueError(f'Invalid SettleFrm=2 configuration: {error}') from error
        settling_clear_water_velocity_m_s=float(
            settling_result.clear_water_settling_velocity_m_s[0,0])
        settling_velocity_m_s=settling_clear_water_velocity_m_s
    else:
        raise ValueError(
            "Settling formula must be None or 'delft_function_of_diameter'")
    if (isinstance(settling_velocity_m_s,bool) or
            not math.isfinite(settling_velocity_m_s) or settling_velocity_m_s<0):
        raise ValueError('Settling velocity must be finite and nonnegative')
    split_capacity_laws=(11,12,13,14)
    resolved_capacity_parameter_rows=None
    capacity_parameter_schema_id=None
    capacity_profile_schema_id=None
    resolved_foundation_parameter_rows=None
    native_fraction_ids=(('single',) if fractions is None else
                         tuple(fraction_ids))
    capacity_foundation_profile=(resolved_capacity_runtime_profile is not None)
    if capacity_foundation_profile:
        profile_fraction_ids=tuple(
            value.fraction_id for value in
            resolved_capacity_runtime_profile.fractions)
        profile_formula_codes={
            value.formula_code for value in
            resolved_capacity_runtime_profile.fractions}
        profile_method_ids={
            value.method_id for value in
            resolved_capacity_runtime_profile.fractions}
        if profile_fraction_ids!=native_fraction_ids:
            raise ValueError(
                'Capacity runtime profile fraction order differs from the native deck')
        if (profile_formula_codes!={bedload} or bedload not in (2,8,11) or
                len(profile_method_ids)!=1):
            raise ValueError(
                'The initial common scalar-capacity runtime requires one '
                'corrected law 2, corrected law 8, or general law 11 identity')
        expected_method={2:'capacity.mpm',8:'capacity.engelund_hansen',
                         11:'capacity.general'}[bedload]
        if profile_method_ids!={expected_method}:
            raise ValueError(
                'Capacity runtime formula identity differs from the selected native law')
        if (resolved_capacity_foundation_execution_document is None or
                resolved_capacity_forcing_document is None):
            raise ValueError(
                'The Q3-C capacity runtime requires its exact typed-forcing '
                'and foundation-execution documents')
        if (resolved_capacity_foundation_execution_document['method_id']!=
                expected_method or
                resolved_capacity_foundation_execution_document['formula_code']!=
                bedload or
                resolved_capacity_foundation_execution_document[
                    'capacity_profile_fingerprint']!=
                resolved_capacity_runtime_profile.fingerprint or
                resolved_capacity_foundation_execution_document[
                    'forcing_fingerprint']!=
                resolved_capacity_forcing_document['fingerprint'] or
                resolved_capacity_foundation_execution_document[
                    'roughness_method_id']!=
                resolved_capacity_forcing_document['roughness_method_id']):
            raise ValueError(
                'Q3-C forcing/execution identity differs from the capacity '
                'profile or selected roughness method')
        if (bedload in (2,8) and
                (not capacity_identity_declared or
                 capacity_method_id!=expected_method)):
            raise ValueError(
                'Corrected law 2/8 requires its ordered v50 execution identity '
                'inside the authenticated v53 profile')
        if (resolved_capacity_runtime_profile.sediment_shear.method_id!=
                'shear.total_roughness' or
                resolved_capacity_runtime_profile.hiding.method_id!='hiding.none' or
                resolved_capacity_runtime_profile.bed_slope.method_id!='bed_slope.off' or
                resolved_capacity_runtime_profile.secondary_flow.method_id!=
                'secondary.off'):
            raise ValueError(
                'The initial common scalar-capacity runtime requires total shear, '
                'hiding off, bed slope off, and secondary flow off')
        if (sediment_shear_mode!='total' or
                (hiding_exposure!=0 and
                 resolved_hiding_foundation_execution_document is None) or
                bed_slope!=0 or
                secondary_flow or exchange!=0):
            raise ValueError(
                'Native controls differ from the authenticated scalar-capacity '
                'foundation profile')
        resolved_foundation_parameter_rows=tuple({
            coefficient.keyword:coefficient.value
            for coefficient in fraction.coefficients}
            for fraction in resolved_capacity_runtime_profile.fractions)
    elif (resolved_capacity_foundation_execution_document is not None or
          resolved_capacity_forcing_document is not None):
        raise ValueError(
            'Q3-C forcing/execution documents require the canonical capacity '
            'runtime profile')
    if bedload in split_capacity_laws:
        if capacity_output_contract!='retain_bed_and_suspended':
            raise ValueError(
                'Delft3D split-capacity laws 11-14 require '
                'capacity_output_contract="retain_bed_and_suspended"; '
                'bed-only use is not permitted')
        if exchange not in (0,4):
            raise ValueError(
                'The M1 split-capacity route does not yet couple suspended capacity '
                'through exchange modes 1-3; use diagnostic-only mode 0 or the '
                'distinct generic FLOW 2DH exchange mode 4 for law 11. Legacy '
                'Van Rijn mode 2 is not a generic capacity adapter')
        if capacity_parameters_by_fraction is None:
            raise ValueError('Split-capacity laws require complete per-fraction parameters')
        fraction_keys=(('single',) if fractions is None else
                       tuple(value.fraction_id for value in fractions))
        if isinstance(capacity_parameters_by_fraction,Mapping):
            # A direct source-keyword mapping is convenient and unambiguous
            # only for the one-material layout.  Multi-fraction mappings must
            # name every fraction so a reordered project cannot swap rows.
            if fractions is None and set(capacity_parameters_by_fraction).issubset(
                    {row.keyword for row in formula_contract(bedload).parameters}):
                supplied_rows=(capacity_parameters_by_fraction,)
            else:
                supplied=set(capacity_parameters_by_fraction)
                if supplied!=set(fraction_keys):
                    missing=sorted(set(fraction_keys)-supplied)
                    extra=sorted(supplied-set(fraction_keys))
                    raise ValueError(
                        'Capacity parameter fraction IDs must match exactly; '
                        f'missing={missing}, extra={extra}')
                supplied_rows=tuple(capacity_parameters_by_fraction[key]
                                    for key in fraction_keys)
        elif (isinstance(capacity_parameters_by_fraction,Sequence) and
              not isinstance(capacity_parameters_by_fraction,(str,bytes))):
            supplied_rows=tuple(capacity_parameters_by_fraction)
            if len(supplied_rows)!=len(fraction_keys):
                raise ValueError('Capacity parameter rows must contain exactly one row per fraction')
        else:
            raise ValueError('Capacity parameters must be a mapping or one mapping per fraction')
        try:
            resolved_capacity_parameter_rows=tuple(
                resolve_capacity_parameters(bedload,row) for row in supplied_rows)
        except CapacityParameterError as error:
            raise ValueError(str(error)) from error
        capacity_parameter_schema_id=NATIVE_V46_CAPACITY_PARAMETER_SCHEMA_ID
        capacity_profile_schema_id=NATIVE_V46_CAPACITY_PROFILE_SCHEMA_ID
        if capacity_foundation_profile:
            if tuple(dict(value) for value in resolved_capacity_parameter_rows)!=\
                    tuple(dict(value) for value in
                          resolved_foundation_parameter_rows):
                raise ValueError(
                    'Law-11 native coefficients differ from the authenticated '
                    'capacity runtime profile')
        if exchange==4:
            if bedload!=11:
                raise ValueError(
                    'Generic FLOW 2DH suspended exchange mode 4 is currently '
                    'implemented only for Delft3D general capacity law 11')
            if fractions is not None:
                raise ValueError(
                    'Generic FLOW 2DH suspended exchange mode 4 is initially '
                    'limited to one noncohesive sediment fraction')
            if capacity_fraction_basis is None:
                raise ValueError(
                    'Generic FLOW 2DH suspended exchange requires an explicit '
                    "capacity_fraction_basis ('dry_mass' or 'solid_volume')")
            if any(float(row['SusACal'])<=0. for row in resolved_capacity_parameter_rows):
                raise ValueError(
                    'Generic FLOW 2DH suspended exchange requires positive SusACal '
                    'for every active fraction')
            if settling_formula!='delft_function_of_diameter':
                raise ValueError(
                    'Generic FLOW 2DH suspended exchange requires '
                    "settling_formula='delft_function_of_diameter' (SettleFrm=2)")
            if settling_mode!=1:
                raise ValueError(
                    'Generic FLOW 2DH suspended exchange requires settling_mode=1 '
                    'hindering on the source-derived clear-water velocity')
    else:
        if capacity_parameters_by_fraction is not None:
            raise ValueError('Capacity parameter rows are valid only for laws 11-14')
        if capacity_output_contract!='bed_only':
            raise ValueError(
                'retain_bed_and_suspended is valid only for Delft3D split-capacity laws 11-14')
    if flow2dh_boundary_diffusion_v48 and exchange!=4:
        raise ValueError(
            'FLOW 2DH suspended boundary/diffusion controls require generic '
            'exchange mode 4; they cannot alter legacy exchange methods')
    if settling_form_code==FLOW2DH_SETTLING_FORM and (
            bedload!=11 or sediment_shear_mode!='total' or hiding_exposure!=0 or
            bed_slope!=0 or secondary_flow):
        raise ValueError(
            'The initial typed suspension profile requires law 11 with total '
            'hydraulic shear and no hiding, bed-slope, or secondary-flow coupling')
    if (isinstance(active_layer_thickness_m,bool) or not math.isfinite(active_layer_thickness_m) or
            active_layer_thickness_m<0):raise ValueError('Active-layer thickness must be finite and nonnegative')
    layered=active_layer_thickness_m>0
    if layered and fractions is None:raise ValueError('Active layers require multiple sediment fractions')
    if active_layer_mode not in ('constant','water_depth'):
        raise ValueError('Active-layer mode must be constant or water_depth')
    if active_layer_minimum_thickness_m is None:
        active_layer_minimum_thickness_m=active_layer_thickness_m
    dynamic_active=active_layer_mode=='water_depth'
    active_values=(active_layer_depth_fraction,active_layer_minimum_thickness_m)
    if any(isinstance(value,bool) or not math.isfinite(value) for value in active_values) or \
            active_layer_depth_fraction<0 or active_layer_minimum_thickness_m<0:
        raise ValueError('Active-layer scaling values must be finite and nonnegative')
    if dynamic_active and (not layered or active_layer_depth_fraction<=0 or active_layer_minimum_thickness_m<=0):
        raise ValueError('Water-depth active layers require positive layer, depth fraction, and minimum thickness')
    if isinstance(substrate_layer_count,bool) or not isinstance(substrate_layer_count,int) or not 1<=substrate_layer_count<=32:
        raise ValueError('Substrate layer count must be an integer from 1 to 32')
    if bedload==9 and substrate_layer_count!=1:
        raise ValueError('TELEMAC-GAIA law 9 requires the exact fixed-grid one-substrate-layer scope')
    if substrate_layer_thickness_m is None:
        substrate_layer_thickness_m=active_layer_thickness_m
    if (isinstance(substrate_layer_thickness_m,bool) or not math.isfinite(substrate_layer_thickness_m) or
            substrate_layer_thickness_m<0 or (layered and substrate_layer_count>1 and substrate_layer_thickness_m<=0)):
        raise ValueError('Preserved substrate layer thickness must be positive when multiple layers are enabled')
    if (bedload in (3,4,5,6,7,9) or exchange==3) and (fractions is None or not layered):
        raise ValueError('Extended current-driven or cohesive sediment laws require multiple fractions and a positive active-layer thickness')
    if isinstance(hiding_exposure,bool) or not isinstance(hiding_exposure,int) or hiding_exposure not in (0,2,3,4,5):
        raise ValueError('Hiding/exposure must be disabled (0) or a Delft3D mode from 2 to 5')
    if (isinstance(hiding_parker_alpha,bool) or not math.isfinite(hiding_parker_alpha) or
            isinstance(hiding_wu_exponent,bool) or not math.isfinite(hiding_wu_exponent)):
        raise ValueError('Hiding/exposure exponents must be finite')
    if hiding_exposure and not layered:
        raise ValueError('Hiding/exposure requires multiple fractions and a positive active-layer thickness')
    if hiding_exposure and bedload not in (2,11):
        raise ValueError(
            'Hiding/exposure currently requires MPM bed load or the v46 '
            'Delft3D general-capacity law (BRAIDS law 11)')
    hiding_foundation_profile=(
        resolved_hiding_foundation_execution_document is not None)
    if hiding_foundation_profile:
        hiding_document=resolved_hiding_foundation_execution_document
        if (not capacity_foundation_profile or
                resolved_capacity_foundation_execution_document is None or
                resolved_capacity_runtime_profile is None):
            raise ValueError(
                'Q3-H1 hiding requires one complete authenticated Q3-C capacity foundation')
        if (hiding_exposure==0 or hiding_document['mode']!=hiding_exposure or
                hiding_document['capacity_law']!=bedload or
                tuple(hiding_document['fraction_ids'])!=tuple(fraction_ids) or
                hiding_document['parker_alpha']!=float(hiding_parker_alpha) or
                hiding_document['wu_exponent']!=float(hiding_wu_exponent) or
                hiding_document['capacity_execution_fingerprint']!=
                    resolved_capacity_foundation_execution_document['fingerprint'] or
                hiding_document['capacity_profile_fingerprint']!=
                    resolved_capacity_runtime_profile.fingerprint):
            raise ValueError(
                'Q3-H1 hiding identity differs from its native controls, ordered '
                'fractions, or nested Q3-C fingerprints')
        if (fractions is None or len(fractions)!=2 or not layered or
                any(value.sediment_type!='sand' for value in fractions) or
                sediment_shear_mode!='total' or exchange!=0 or bed_slope!=0 or
                secondary_flow or amr_hierarchy is not None):
            raise ValueError(
                'Q3-H1 hiding requires exactly two noncohesive active-surface '
                'fractions, total shear, fixed grid, and exchange/slope/secondary off')
    elif capacity_foundation_profile and hiding_exposure!=0:
        raise ValueError(
            'Authenticated Q3-C capacity cannot consume generic hiding without '
            'the additive Q3-H1 execution identity')
    secondary_foundation_profile=(
        resolved_secondary_foundation_execution_document is not None)
    if secondary_foundation_profile:
        secondary_document=resolved_secondary_foundation_execution_document
        if (not capacity_foundation_profile or
                resolved_capacity_foundation_execution_document is None or
                resolved_capacity_runtime_profile is None or
                resolved_capacity_forcing_document is None):
            raise ValueError(
                'Q3-H2 secondary flow requires one complete authenticated '
                'Q3-C capacity foundation')
        expected_roughness_method={
            'chezy':'roughness.chezy',
            'manning':'roughness.manning',
            'white_colebrook':(
                'roughness.white_colebrook_waqua'
                if white_colebrook_chzk2d else
                'roughness.white_colebrook'),
        }[hydraulic_roughness_mode]
        expected_hiding_fingerprint=(
            None if not hiding_foundation_profile else
            resolved_hiding_foundation_execution_document['fingerprint'])
        expected_base_envelope=55 if hiding_foundation_profile else 54
        if (secondary_document['method_id']!='secondary.equilibrium' or
                secondary_document['model']!=1 or
                secondary_document['capacity_law']!=bedload or
                secondary_document['base_envelope_version']!=
                    expected_base_envelope or
                secondary_document['capacity_execution_fingerprint']!=
                    resolved_capacity_foundation_execution_document['fingerprint'] or
                secondary_document['capacity_profile_fingerprint']!=
                    resolved_capacity_runtime_profile.fingerprint or
                secondary_document['hiding_execution_fingerprint']!=
                    expected_hiding_fingerprint or
                secondary_document['roughness_method_id']!=
                    expected_roughness_method or
                secondary_document['roughness_fingerprint']!=
                    resolved_capacity_forcing_document['fingerprint']):
            raise ValueError(
                'Q3-H2 secondary identity differs from its nested capacity, '
                'hiding, or typed-roughness foundation')
        if (secondary_document['coriolis_disposition']!='zero' or
                secondary_document['coriolis_constant_s_inverse']!=0. or
                secondary_document['coriolis_source']!=
                    'constant_zero_planar' or
                secondary_document['betac']!=0.):
            raise ValueError(
                'The first Q3-H2 live route requires zero Coriolis and zero Betac')
        if (secondary_flow or exchange!=0 or bed_slope!=0 or
                amr_hierarchy is not None or sediment_shear_mode!='total' or
                bedload not in (2,8,11) or
                bedload_adaptation or bedform_roughness):
            raise ValueError(
                'Q3-H2 live secondary flow requires fixed-grid Q3-C/Q3-H1, '
                'total shear, exchange/slope/legacy-secondary/adaptation/'
                'bedform off, and law 2, 8, or 11')
    prognostic_secondary_profile=(
        resolved_prognostic_secondary_execution_document is not None)
    if prognostic_secondary_profile:
        prognostic_document=resolved_prognostic_secondary_execution_document
        if (not capacity_foundation_profile or
                resolved_capacity_foundation_execution_document is None or
                resolved_capacity_runtime_profile is None or
                resolved_capacity_forcing_document is None):
            raise ValueError(
                'Q3-H3 prognostic secondary flow requires one complete '
                'authenticated Q3-C capacity foundation')
        expected_roughness_method={
            'chezy':'roughness.chezy',
            'manning':'roughness.manning',
            'white_colebrook':(
                'roughness.white_colebrook_waqua'
                if white_colebrook_chzk2d else
                'roughness.white_colebrook'),
        }[hydraulic_roughness_mode]
        expected_hiding_fingerprint=(
            None if not hiding_foundation_profile else
            resolved_hiding_foundation_execution_document['fingerprint'])
        expected_base_envelope=55 if hiding_foundation_profile else 54
        expected_fraction_ids=(('single',) if fractions is None else
                               tuple(fraction_ids))
        if (prognostic_document['method_id']!='secondary.prognostic' or
                prognostic_document['model']!=2 or
                prognostic_document['capacity_law']!=bedload or
                prognostic_document['base_envelope_version']!=
                    expected_base_envelope or
                prognostic_document['capacity_execution_fingerprint']!=
                    resolved_capacity_foundation_execution_document['fingerprint'] or
                prognostic_document['capacity_profile_fingerprint']!=
                    resolved_capacity_runtime_profile.fingerprint or
                prognostic_document['hiding_execution_fingerprint']!=
                    expected_hiding_fingerprint or
                tuple(prognostic_document['ordered_fraction_ids'])!=
                    expected_fraction_ids or
                prognostic_document['roughness_method_id']!=
                    expected_roughness_method or
                prognostic_document['roughness_fingerprint']!=
                    resolved_capacity_forcing_document['fingerprint']):
            raise ValueError(
                'Q3-H3 prognostic identity differs from its nested capacity, '
                'hiding, fraction, or typed-roughness foundation')
        if (prognostic_document['coriolis_disposition']!='zero' or
                prognostic_document['coriolis_constant_s_inverse']!=0. or
                prognostic_document['coriolis_source']!=
                    'constant_zero_planar' or
                prognostic_document['betac']!=0. or
                prognostic_document['scalar_diffusivity_m2_s']!=0. or
                prognostic_document['transport_direction'] is not True or
                prognostic_document['inflow_mode']!=1):
            raise ValueError(
                'The first Q3-H3 live route requires zero Coriolis/Betac/'
                'diffusivity, direction coupling, and inflow mode 1')
        if (secondary_flow or exchange!=0 or bed_slope!=0 or
                horizontal_viscosity_m2_s!=0. or
                amr_hierarchy is not None or sediment_shear_mode!='total' or
                bedload not in (2,8,11) or
                bedload_adaptation or bedform_roughness):
            raise ValueError(
                'Q3-H3 live prognostic secondary flow requires fixed-grid '
                'Q3-C/Q3-H1, total shear, and exchange/slope/legacy-secondary/'
                'viscosity/adaptation/bedform off with law 2, 8, or 11')
    extended_values=(critical_mud_fraction,transport_calibration_factor)
    if any(isinstance(value,bool) or not math.isfinite(value) for value in extended_values) or \
            not 0<=critical_mud_fraction<=1 or transport_calibration_factor<0:
        raise ValueError('Invalid mixed-bed or transport-calibration setting')
    if bedload in (8,9) and transport_calibration_factor!=1.:
        raise ValueError('Engelund-Hansen validation scopes require transport calibration 1')
    if bedload==8:
        if capacity_foundation_profile:
            if engelund_hansen_chezy is not None:
                raise ValueError(
                    'Corrected Engelund-Hansen consumes the selected typed '
                    'hydraulic roughness; do not supply an independent constant Chezy')
        elif (isinstance(engelund_hansen_chezy,bool) or
              engelund_hansen_chezy is None or
              not math.isfinite(engelund_hansen_chezy) or
              engelund_hansen_chezy<=0):
            raise ValueError(
                'Legacy Engelund-Hansen bed load requires a positive finite '
                'constant Chezy coefficient')
        if exchange!=0:
            raise ValueError('Engelund-Hansen law 8 is qualified only as bedload-only transport (exchange 0)')
    elif engelund_hansen_chezy is not None:
        raise ValueError('The constant Engelund-Hansen Chezy coefficient is valid only for bed-load law 8')
    resolved_channel_composition_profile=None
    resolved_channel_composition_document=None
    resolved_channel_composition_native_wire=None
    resolved_execution_profile=None
    resolved_execution_profile_document=None
    resolved_execution_profile_preimage=None
    resolved_component_document_receipts=None
    if capacity_foundation_profile and channel_composition_profile is None:
        raise ValueError(
            'The common scalar-capacity runtime requires its authenticated '
            'channel-composition and schema-v2 execution profile')
    if (type(channel_composition_wire_version) is not int or
            channel_composition_wire_version not in (51,52,53)):
        raise ValueError(
            'Channel-composition wire version must be the exact integer 51, 52, or 53')
    if capacity_foundation_profile and channel_composition_wire_version!=53:
        raise ValueError(
            'Q3-C native forcing authentication requires the v53 component '
            'envelope nested inside its additive v54 identity')
    if capacity_foundation_profile and (
            not capacity_identity_declared or
            not foundation_execution_identity or
            capacity_execution_fingerprint!=
                resolved_capacity_foundation_execution_document['fingerprint'] or
            capacity_profile_fingerprint!=
                resolved_capacity_runtime_profile.fingerprint):
        raise ValueError(
            'Q3-C ordered execution/profile identity differs from its exact '
            'v54 preimages')
    if channel_composition_profile is not None:
        # Re-resolve both the supplied profile
        # and the profile implied by the already validated native controls so
        # no stale document, reordered fraction, changed factor, or altered
        # class can enter the live route under a valid-looking fingerprint.
        try:
            if isinstance(channel_composition_profile,ChannelCompositionProfile):
                supplied_channel_profile=channel_composition_profile
                channel_composition_profile_document(supplied_channel_profile)
            elif isinstance(channel_composition_profile,Mapping):
                supplied_channel_profile=load_channel_composition_profile(
                    channel_composition_profile)
            else:
                raise ChannelCompositionError(
                    'channel composition profile must be a canonical mapping or profile')
        except ChannelCompositionError as error:
            raise ValueError(f'Invalid channel-composition profile: {error}') from error
        if bedload not in (2,8,11):
            raise ValueError(
                'Native scalar-capacity channel composition supports only '
                'corrected laws 2/8 and general law 11')
        if bedload in (2,8) and not capacity_foundation_profile:
            raise ValueError(
                'Corrected law-2/law-8 channel composition requires the '
                'canonical capacity runtime profile')
        if exchange==0:
            transport_class=TRA_BEDLOAD
        elif exchange==4:
            transport_class=TRA_COMBINE
        else:
            raise ValueError(
                'Native channel composition v51 requires exchange 0 (class 1) '
                'or exchange 4 (class 3)')
        channel_fraction_ids=(('single',) if fractions is None else
                              tuple(fraction_ids))
        try:
            expected_channel_profile=resolve_channel_composition_profile(
                channel_fraction_ids,
                (transport_class,)*len(channel_fraction_ids),
                bed_factor=(supplied_channel_profile.bed_factor
                            if capacity_foundation_profile else
                            transport_calibration_factor),
                suspended_factor=(supplied_channel_profile.suspended_factor
                                  if capacity_foundation_profile else
                                  suspended_calibration),
                epsilon=1.e-6)
        except ChannelCompositionError as error:
            raise ValueError(
                f'Invalid native channel-composition controls: {error}') from error
        if supplied_channel_profile!=expected_channel_profile:
            raise ValueError(
                'Channel-composition profile is stale, forged, or disagrees '
                'with ordered fractions, exchange class, Bed/Sus factors, or epsilon')
        resolved_channel_composition_profile=expected_channel_profile
        resolved_channel_composition_document=channel_composition_profile_document(
            expected_channel_profile)
        if channel_composition_wire_version in (52,53):
            if channel_composition_wire_version==53 and transport_class!=TRA_BEDLOAD:
                raise ValueError(
                    'Native v53 public admission currently supports only the '
                    'scalar-capacity algebraic total-load route (transport class 1)')
            try:
                resolved_channel_composition_native_wire=(
                    channel_composition_native_wire_document(
                        expected_channel_profile))
            except ChannelCompositionError as error:
                raise ValueError(
                    f'Invalid native channel-composition v52 identity: {error}'
                ) from error
            if execution_profile_v2 is None:
                raise ValueError(
                    'Native channel-composition v52 requires a complete '
                    'backend-neutral execution_profile_v2')
            try:
                if isinstance(execution_profile_v2,Flow2DHExecutionProfile):
                    supplied_execution_profile=execution_profile_v2
                    execution_document=execution_profile_document(
                        supplied_execution_profile)
                elif isinstance(execution_profile_v2,Mapping):
                    supplied_execution_profile=load_execution_profile(
                        execution_profile_v2)
                    execution_document=execution_profile_document(
                        supplied_execution_profile)
                else:
                    raise ExecutionProfileError(
                        'execution_profile_v2 must be a canonical mapping or profile',
                        category='invalid_configuration')
            except ExecutionProfileError as error:
                raise ValueError(
                    f'Invalid native execution profile v2: {error}') from error
            if supplied_execution_profile.backend is not None:
                raise ValueError(
                    'Native cut-cell v52 currently requires the neutral '
                    'execution-profile backend until the Q3-S route is admitted; '
                    'the deck separately binds the actual CPU runtime')
            execution_fraction_ids=tuple(
                value.fraction_id
                for value in supplied_execution_profile.fraction_layout)
            execution_classes=tuple(
                value.transport_class
                for value in supplied_execution_profile.fraction_layout)
            expected_classes=tuple(
                value.transport_class
                for value in expected_channel_profile.fractions)
            if (execution_fraction_ids!=channel_fraction_ids or
                    execution_classes!=expected_classes):
                raise ValueError(
                    'Execution-profile fraction layout differs from the ordered '
                    'channel-composition fractions/classes')
            channel_components=[
                value for value in supplied_execution_profile.component_profiles
                if value.component_id=='channel_composition'
            ]
            channel_document_sha256=hashlib.sha256(json.dumps(
                resolved_channel_composition_document,sort_keys=True,
                separators=(',',':'),ensure_ascii=True,allow_nan=False,
            ).encode('ascii')).hexdigest()
            if (len(channel_components)!=1 or
                    channel_components[0].schema!=CHANNEL_COMPOSITION_SCHEMA or
                    channel_components[0].scope_id!=CHANNEL_COMPOSITION_SCOPE_ID or
                    channel_components[0].semantic_id!=channel_composition_semantic_id() or
                    channel_components[0].fingerprint!=expected_channel_profile.fingerprint or
                    channel_components[0].document_sha256!=channel_document_sha256):
                raise ValueError(
                    'Execution profile does not bind the exact channel_composition '
                    'component document')
            # Q3-S admits one fully enumerated baseline combination.  The
            # profile is neutral only because the ledger route is not public
            # yet; the actual executable path remains explicitly CPU and the
            # nested deck is independently checked below.  No selector may be
            # silently omitted or inferred from a component fingerprint.
            if (sediment_shear_mode!='total' or
                    (hiding_exposure!=0 and not hiding_foundation_profile) or
                    bed_slope!=0 or secondary_flow):
                raise ValueError(
                    'Native v52 currently admits total shear, hiding off, '
                    'bed slope off, and secondary flow off only')
            scoped_model=lambda entry: (entry,(('model_id','model'),))
            expected_methods={
                scoped_model('shear.total_roughness'),
                scoped_model('hiding.none'),
                scoped_model('bed_slope.off'),
                scoped_model('secondary.off'),
            }
            capacity_method_selection={
                2:'capacity.mpm',8:'capacity.engelund_hansen',
                11:'capacity.general'}[bedload]
            if capacity_foundation_profile:
                roughness_method={
                    'chezy':'roughness.chezy',
                    'manning':'roughness.manning',
                    'white_colebrook':(
                        'roughness.white_colebrook_waqua'
                        if white_colebrook_chzk2d else
                        'roughness.white_colebrook'),
                }[hydraulic_roughness_mode]
                expected_methods.add(scoped_model(roughness_method))
            for fraction_id in channel_fraction_ids:
                scoped_fraction=lambda entry,fraction_id=fraction_id: (
                    entry,(('fraction_id',fraction_id),))
                expected_methods.update((
                    scoped_fraction(capacity_method_selection),
                    scoped_fraction(
                        'transport_route.total' if exchange==0 else
                        'transport_route.combine'),
                    scoped_fraction(
                        'suspension.off' if exchange==0 else
                        'suspension.van_rijn_exchange'),
                ))
                if exchange==4:
                    expected_methods.update((
                        scoped_fraction('settling.grain_size'),
                    ))
            actual_methods={
                (value.entry_id,value.scope)
                for value in supplied_execution_profile.method_selections
            }
            if actual_methods!=expected_methods:
                raise ValueError(
                    'Execution-profile method selections differ from the '
                    'resolved native v52 operator chain')
            parameter_prefix={2:'parameter.capacity.formula_2.',
                              8:'parameter.capacity.formula_1.',
                              11:'parameter.capacity.formula_4.'}[bedload]
            profile_parameter_rows=(
                resolved_foundation_parameter_rows
                if capacity_foundation_profile else
                resolved_capacity_parameter_rows)
            expected_parameter_values={
                (f'{parameter_prefix}{key.lower()}',
                 (('fraction_id',fraction_id),)):float(value)
                for fraction_id,row in zip(
                    channel_fraction_ids,profile_parameter_rows,strict=True)
                for key,value in row.items()
            }
            actual_parameters={}
            for value in supplied_execution_profile.parameter_bindings:
                if value.representation!='inline':
                    raise ValueError(
                        'Execution-profile scalar-capacity parameter binding scope differs')
                actual_parameters[(value.entry_id,value.scope)]=value.value
            if (set(actual_parameters)!=set(expected_parameter_values) or any(
                    type(actual_parameters[key]) is bool or
                    float(actual_parameters[key])!=expected
                    for key,expected in expected_parameter_values.items())):
                raise ValueError(
                    'Execution-profile scalar-capacity parameter bindings differ from deck')
            expected_controls={
                'control.flux_limiter':'none',
                'control.laterally_averaged_bedload':False,
                'control.maximum_waterdepth':False,
                'control.maximum_waterdepth_fraction':1.0,
                'control.upwind_bedload':True,
            }
            actual_controls={}
            for value in supplied_execution_profile.control_bindings:
                if (value.representation!='inline' or
                        value.scope!=(('model_id','model'),)):
                    raise ValueError(
                        'Execution-profile comparison-control scope differs')
                actual_controls[value.entry_id]=value.value
            if actual_controls!=expected_controls:
                raise ValueError(
                    'Execution-profile comparison controls differ from the '
                    'native v52 baseline')
            if capacity_foundation_profile:
                capacity_instance_profile=resolved_capacity_runtime_profile
                expected_capacity_document=resolved_capacity_runtime_document
            else:
                capacity_instance_profile=resolve_capacity_profile_from_code(
                    11,channel_fraction_ids,
                    (dict(resolved_capacity_parameter_rows[0])
                     if len(channel_fraction_ids)==1 else
                     {fraction_id:dict(row) for fraction_id,row in zip(
                         channel_fraction_ids,resolved_capacity_parameter_rows,
                         strict=True)}),
                    sediment_shear='shear.total_roughness',hiding='hiding.none',
                    bed_slope='bed_slope.off',secondary_flow='secondary.off')
                expected_capacity_document=capacity_profile_document(
                    capacity_instance_profile)
            expected_capacity_component=hashlib.sha256(json.dumps(
                expected_capacity_document,sort_keys=True,separators=(',',':'),
                ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
            capacity_components=[
                value for value in supplied_execution_profile.component_profiles
                if value.component_id=='capacity'
            ]
            expected_component_ids={
                'capacity','channel_composition',
                *(() if exchange==0 else ('suspension',)),
            }
            if ({value.component_id for value in
                 supplied_execution_profile.component_profiles}!=expected_component_ids or
                    len(capacity_components)!=1 or
                    capacity_components[0].fingerprint!=capacity_instance_profile.fingerprint or
                    capacity_components[0].document_sha256!=expected_capacity_component):
                raise ValueError(
                    'Execution profile does not bind the exact capacity/component set')
            resolved_execution_profile=supplied_execution_profile
            resolved_execution_profile_document=execution_document
            resolved_execution_profile_preimage=execution_profile_payload_json(
                supplied_execution_profile)
            if (not resolved_execution_profile_preimage.isascii() or
                    len(resolved_execution_profile_preimage)>262144):
                raise ValueError(
                    'Execution-profile v2 canonical preimage exceeds the native bound')
    elif execution_profile_v2 is not None:
        raise ValueError(
            'execution_profile_v2 is admitted only with channel-composition v52')
    gaia_values=(gaia_skin_roughness_ratio,gaia_slope_beta,
                  gaia_deviation_beta2,gaia_secondary_alpha)
    if any(isinstance(value,bool) or not math.isfinite(value) for value in gaia_values) or \
            gaia_skin_roughness_ratio<=0 or gaia_slope_beta<0 or \
            gaia_deviation_beta2<=0 or gaia_secondary_alpha<=0:
        raise ValueError('Invalid TELEMAC-GAIA law-9 parameters')
    if not isinstance(gaia_diagnostic_v44,bool):
        raise ValueError('gaia_diagnostic_v44 must be true or false')
    gaia_diagnostic_values=(gaia_interior_slope_talmon_enabled,
                            gaia_interior_secondary_stress_enabled,
                            gaia_open_boundary_secondary_stress_enabled)
    if gaia_diagnostic_v44:
        gaia_diagnostic_values=tuple(True if value is None else value
                                     for value in gaia_diagnostic_values)
        if not all(isinstance(value,bool) for value in gaia_diagnostic_values):
            raise ValueError('TELEMAC-GAIA v44 diagnostic switches must be boolean')
    else:
        if any(value is not None for value in gaia_diagnostic_values):
            raise ValueError('TELEMAC-GAIA diagnostic switches require gaia_diagnostic_v44=True')
        gaia_diagnostic_values=(True,True,True)
    if isinstance(settling_mode,bool) or not isinstance(settling_mode,int) or settling_mode not in (0,1):
        raise ValueError('Settling treatment must be constant (0) or concentration-dependent hindered settling (1)')
    if (isinstance(hindered_settling_reference_concentration_kg_m3,bool) or
            not math.isfinite(hindered_settling_reference_concentration_kg_m3) or
            hindered_settling_reference_concentration_kg_m3<=0):
        raise ValueError('Hindered-settling reference concentration must be positive and finite')
    if any(isinstance(v,bool) or not math.isfinite(v) or v<0 for v in [mpm_cf,mpm_critical_shields]):
        raise ValueError('Invalid MPM friction/threshold')
    if isinstance(mpm_friction_mode,bool) or not isinstance(mpm_friction_mode,int) or mpm_friction_mode not in (0,1):
        raise ValueError('MPM friction mode must be specified coefficient (0) or roughness-corrected (1)')
        if mpm_friction_mode and (
                bedload!=2 or
                ((fractions is None or not layered) and
                 not capacity_foundation_profile)):
            raise ValueError(
                'Roughness-corrected MPM requires MPM bed load and, outside '
                'the authenticated Q3-C single-reservoir route, an active '
                'multi-fraction bed')
    if mpm_friction_mode and not math.isclose(mpm_critical_shields,.047,rel_tol=0,abs_tol=1e-14):
        raise ValueError('Roughness-corrected MPM requires the source critical Shields value 0.047')
    slope_values=[bed_slope_alfabs,bed_slope_alfabn,bed_friction_angle_deg]
    if bed_slope not in (0,2,3,4,5) or any(isinstance(v,bool) or not math.isfinite(v) for v in slope_values):
        raise ValueError('Slope correction must be disabled (0), legacy Bagnold/Ikeda (2), Koch-Flokstra (3), TELEMAC-GAIA (4), or Delft3D FLOW Bagnold/Ikeda (5), with finite parameters')
    if min(slope_values)<0 or (bed_slope in (2,5) and not 0<bed_friction_angle_deg<90):
        raise ValueError('Invalid slope-correction parameters')
    if bed_slope and bedload==0:
        raise ValueError('Slope correction requires an active bed-load formulation')
    if bed_slope in (3,5) and bedload not in (2,8):
        raise ValueError('Delft3D FLOW Koch-Flokstra/mode-5 requires Engelund-Hansen or MPM (law 8/2)')
    if bed_slope==4 and bedload!=9:
        raise ValueError('TELEMAC-GAIA slope mode 4 requires bed-load law 9')
    if bedload in split_capacity_laws and bed_slope!=0:
        raise ValueError(
            'The M1 split-capacity live slice currently requires bed-slope correction off; '
            'slope composition is admitted only after its shared diagnostic ABI is connected')
    if not isinstance(mesh,CutCellMesh):raise ValueError('CutCellMesh is required')
    if not isinstance(compact_sidecars,bool):raise ValueError('compact_sidecars must be boolean')
    q=np.asarray(initial,dtype=float);rock=np.asarray(bedrock,dtype=float)
    bed_layers=1+(substrate_layer_count if layered else 0)
    if prognostic_secondary_profile:
        expected_fraction_count=1 if fractions is None else len(fractions)
        physical_state_count=3+(1+bed_layers)*expected_fraction_count
        physical_primitive_count=4+expected_fraction_count
        physical_ledger_count=1+2*expected_fraction_count
        expected_layout={
            'index_base':0,
            'fraction_count':expected_fraction_count,
            'bed_layer_count':bed_layers,
            'physical_state_count':physical_state_count,
            'runtime_state_count':physical_state_count+1,
            'secondary_state_index':physical_state_count,
            'physical_primitive_count':physical_primitive_count,
            'runtime_primitive_count':physical_primitive_count+1,
            'secondary_primitive_index':3+expected_fraction_count,
            'physical_ledger_count':physical_ledger_count,
            'runtime_ledger_count':physical_ledger_count+1,
            'secondary_ledger_index':physical_ledger_count,
        }
        if prognostic_document['runtime_layout']!=expected_layout:
            raise ValueError(
                'Q3-H3 prognostic runtime layout differs from the native '
                'fraction/layer state, primitive, or ledger layout')
    expected_columns=5 if fractions is None else 3+(1+bed_layers)*len(fractions)
    if q.shape!=(len(mesh.cells),expected_columns) or rock.shape!=(len(mesh.cells),):raise ValueError('Wrong initial-state shape')
    if not np.isfinite(q).all() or not np.isfinite(rock).all() or (q[:,[0,*range(3,expected_columns)]]<0).any():
        raise ValueError('Initial water and sediment inventories must be finite/nonnegative')
    if layered:
        active=q[:,3+len(fractions):3+2*len(fractions)]
        substrate=q[:,3+2*len(fractions):].reshape(len(q),substrate_layer_count,len(fractions))
        bulk_values=np.array([value.bulk_kg_m3 for value in fractions])
        active_depth=(active/bulk_values).sum(axis=1)
        substrate_depth=(substrate/bulk_values).sum(axis=2)
        total_depth=active_depth+substrate_depth.sum(axis=1)
        configured_target=(np.maximum(active_layer_depth_fraction*np.maximum(q[:,0],0.),
                                      active_layer_minimum_thickness_m)
                           if dynamic_active else np.full(len(q),active_layer_thickness_m))
        target=np.minimum(configured_target,total_depth)
        tolerance=1e-12+64*np.finfo(float).eps*np.maximum(1.,total_depth)
        if not np.all(np.abs(active_depth-target)<=tolerance):
            raise ValueError('Initial active inventory must fill the configured layer or the available bed thickness')
        if substrate_layer_count>1 and np.any(substrate_depth[:,:-1]>substrate_layer_thickness_m+tolerance[:,None]):
            raise ValueError('An initial preserved substrate layer exceeds its configured thickness')
        if substrate_layer_count>1:
            remaining=substrate_depth.sum(axis=1)
            for layer in range(substrate_layer_count-1):
                expected=np.minimum(remaining,substrate_layer_thickness_m)
                if np.any(np.abs(substrate_depth[:,layer]-expected)>tolerance):
                    raise ValueError('Initial preserved substrate layers must be packed from the surface downward')
                remaining=np.maximum(0.,remaining-expected)
    if any(not math.isfinite(v) or v<=0 for v in [end_time_s,dt_max_s,output_interval_s]):
        raise ValueError('Positive finite time settings are required')
    if isinstance(cfl,bool) or not math.isfinite(cfl) or not 0<cfl<=.9:
        raise ValueError('CFL limit must be finite, positive, and no greater than 0.9')
    if any(not math.isfinite(v) or v<0 for v in [settling_velocity_m_s,manning]):
        raise ValueError('Invalid settling/roughness')
    if hydraulic_roughness_mode!='manning' and manning<=0:
        raise ValueError('Chezy and White-Colebrook roughness parameters must be positive')
    local_manning=None
    if manning_by_cell is not None:
        local_manning=np.asarray(manning_by_cell,dtype=float)
        if local_manning.shape!=(len(mesh.cells),) or not np.isfinite(local_manning).all() or (local_manning<0).any():
            label=('Manning values' if hydraulic_roughness_mode=='manning'
                   else 'roughness values')
            raise ValueError(f'Per-cell {label} must be one nonnegative finite value per retained cell')
        if hydraulic_roughness_mode!='manning' and (local_manning<=0).any():
            raise ValueError('Per-cell Chezy and White-Colebrook parameters must be positive')
    if capacity_foundation_profile:
        raw_roughness=np.ascontiguousarray(
            (np.full(len(mesh.cells),manning,dtype=np.float64)
             if local_manning is None else local_manning),dtype='<f8')
        raw_identity=resolved_capacity_forcing_document[
            'roughness_parameter_array']
        if (raw_identity!={
                'shape':[len(mesh.cells)],'dtype':raw_roughness.dtype.str,
                'sha256':hashlib.sha256(
                    raw_roughness.tobytes(order='C')).hexdigest()} or
                resolved_capacity_forcing_document[
                    'roughness_default_parameter']!=float(manning) or
                resolved_capacity_forcing_document[
                    'roughness_parameter_minimum']!=float(np.min(raw_roughness)) or
                resolved_capacity_forcing_document[
                    'roughness_parameter_maximum']!=float(np.max(raw_roughness))):
            raise ValueError(
                'Q3-C forcing document differs from the exact native FP64 '
                'roughness bytes or base parameter')
    if sediment_shear_mode=='soulsby_clarke_skin':
        if bedload==9:
            raise ValueError('Delft3D BsSkin cannot replace the distinct TELEMAC-GAIA skin correction')
        if bedload not in (5,6,7) and exchange!=3:
            raise ValueError('Soulsby/Clarke skin shear requires an RP_TAUB consumer: law 5/6/7 or mixed/cohesive exchange')
    if not isinstance(secondary_flow,bool):
        raise ValueError('secondary_flow must be true or false')
    if (not isinstance(secondary_flow_model,str) or
            secondary_flow_model not in ('equilibrium','prognostic','gaia')):
        raise ValueError("Secondary-flow model must be 'equilibrium', 'prognostic', or 'gaia'")
    if not isinstance(secondary_flow_transport_direction,bool):
        raise ValueError('Secondary-flow transport-direction switch must be true or false')
    if isinstance(secondary_flow_espir,bool) or not math.isfinite(secondary_flow_espir) or secondary_flow_espir<0:
        raise ValueError('Secondary-flow Espir must be a finite nonnegative number')
    if (isinstance(secondary_flow_momentum_beta,bool) or
            not math.isfinite(secondary_flow_momentum_beta) or
            secondary_flow_momentum_beta<0):
        raise ValueError('Secondary-flow momentum Beta must be a finite nonnegative number')
    prognostic_secondary_flow=secondary_flow and secondary_flow_model=='prognostic'
    gaia_secondary_flow=secondary_flow and secondary_flow_model=='gaia'
    if secondary_flow and secondary_flow_model=='equilibrium':
        if not secondary_flow_transport_direction or secondary_flow_momentum_beta!=0:
            raise ValueError(
                'Legacy equilibrium secondary flow has fixed direction-only coupling and zero momentum Beta')
        if bedload not in (2,3,5,6,7,8,11,12,14):
            raise ValueError('Secondary flow requires a magnitude-based bed-load formulation')
    if prognostic_secondary_flow and secondary_flow_transport_direction and bedload not in (1,2,3,5,6,7,8,11,12,14):
        raise ValueError(
            'Prognostic secondary-flow transport direction requires a supported bed-load formulation')
    if gaia_secondary_flow and (bedload!=9 or bed_slope!=4 or not secondary_flow_transport_direction or
                                 secondary_flow_momentum_beta!=0):
        raise ValueError('TELEMAC-GAIA secondary flow requires law 9, slope 4, direction coupling, and zero momentum Beta')
    if gaia_diagnostic_v44 and not gaia_secondary_flow:
        raise ValueError('TELEMAC-GAIA v44 diagnostic envelope requires the exact law-9 GAIA scope')
    if secondary_flow and (manning<=0 if local_manning is None else bool((local_manning<=0).any())):
        raise ValueError('Secondary flow requires positive Manning roughness in every cell')
    if scheme not in (1,5,6) or bedload not in (*range(10),11,12,13,14) or exchange not in (0,1,2,3,4):raise ValueError('Unsupported numerical modes')
    if exchange==4 and amr_hierarchy is not None and not flow2dh_boundary_diffusion_v48:
        raise ValueError(
            'Generic FLOW 2DH suspended exchange with AMR requires the explicit '
            'v48 boundary/diffusion controls and the v49 wet/dry-AMR ownership contract')
    if exchange==4 and not frozen_pre_m2c_suspension and prognostic_secondary_flow:
        raise ValueError(
            'The v49 suspended wet/dry-AMR ownership route is not yet qualified '
            'with prognostic secondary flow')
    if (exchange==4 and not frozen_pre_m2c_suspension and
            np.any((q[:,0] <= dry_tolerance_m) & (q[:,3] > 0.))):
        raise ValueError(
            'The v49 ownership contract requires zero initial suspended '
            'inventory in hydraulically dry cells; transfer it to the active bed')
    if (exchange==4 and not frozen_pre_m2c_suspension and
            np.any((q[:,0] <= dry_tolerance_m) &
                   np.any(q[:,1:3] != 0., axis=1))):
        raise ValueError(
            'The v49 ownership contract requires zero initial momentum in '
            'hydraulically dry cells')
    if capacity_fraction_basis is not None and bedload not in split_capacity_laws:
        raise ValueError('The explicit capacity fraction basis currently applies only to laws 11-14')
    if suspended_adaptation_factor!=1. and exchange!=4:
        raise ValueError('FacTsd is active only for generic FLOW 2DH suspended exchange mode 4')
    if bedload in (8,9,11,14) and (manning<=0 if local_manning is None else bool((local_manning<=0).any())):
        raise ValueError('The selected capacity law requires positive hydraulic roughness in every cell')
    bank_values=(bank_erosion_fraction,bank_wet_slope,bank_dry_slope,
                 bank_avalanche_time_s,bank_wet_depth_switch_m,bank_relaxation_length_m)
    if any(isinstance(value,bool) or not math.isfinite(value) for value in bank_values):
        raise ValueError('Bank-morphology settings must be finite numbers')
    if (not 0<=bank_erosion_fraction<=1 or
            min(bank_wet_slope,bank_dry_slope,bank_avalanche_time_s,bank_relaxation_length_m)<=0 or
            bank_wet_depth_switch_m<0):
        raise ValueError('Invalid bank-morphology settings')
    if not isinstance(bank_avalanche,bool):raise ValueError('bank_avalanche must be true or false')
    bank_enabled=bank_erosion_fraction>0 or bank_avalanche
    bank_mask=np.zeros(len(mesh.cells),dtype=bool) if bank_erodible_by_cell is None else np.asarray(bank_erodible_by_cell)
    if bank_mask.shape!=(len(mesh.cells),):raise ValueError('Bank mask must contain one value per retained cell')
    if bank_mask.dtype.kind not in 'biu' or not np.isin(bank_mask,[0,1]).all():
        raise ValueError('Bank mask values must be boolean or zero/one')
    bank_mask=bank_mask.astype(bool,copy=False)
    if bank_enabled and (fractions is None or not layered):
        raise ValueError('Bank morphology requires multiple fractions and a positive active-layer thickness')
    if bank_enabled and not bank_mask.any():
        raise ValueError('Enabled bank morphology requires at least one erodible-bank cell')
    general_values=(general_avalanche_wet_slope,general_avalanche_dry_slope,
                    general_avalanche_time_s,general_avalanche_wet_depth_switch_m,
                    general_avalanche_relaxation_length_m)
    if not isinstance(general_avalanche,bool):
        raise ValueError('general_avalanche must be true or false')
    if any(isinstance(value,bool) or not math.isfinite(value) for value in general_values) or \
            min(general_avalanche_wet_slope,general_avalanche_dry_slope,
                general_avalanche_time_s,general_avalanche_relaxation_length_m)<=0 or \
            general_avalanche_wet_depth_switch_m<0:
        raise ValueError('Invalid general-avalanche settings')
    if general_avalanche and (fractions is None or not layered):
        raise ValueError('General avalanching requires multiple fractions and a positive active-layer thickness')
    if general_avalanche and bank_avalanche:
        raise ValueError('General and bank-only avalanching cannot both be enabled')
    if not isinstance(bank_failure,bool):
        raise ValueError('Process-separated bank failure must be true or false')
    if bank_toe_erosion_mode not in (
            'cohesive_excess_shear','noncohesive_transport_capacity',
            'distributed_slope_transport'):
        raise ValueError(
            'Bank-toe erosion mode must be cohesive_excess_shear, '
            'noncohesive_transport_capacity, or distributed_slope_transport')
    if (isinstance(bank_toe_capacity_fraction,bool) or
            not math.isfinite(bank_toe_capacity_fraction) or
            not 0<=bank_toe_capacity_fraction<=1):
        raise ValueError('Bank-toe capacity fraction must be finite and within zero to one')
    if (isinstance(bank_distributed_transport_fraction,bool) or
            not math.isfinite(bank_distributed_transport_fraction) or
            not 0<=bank_distributed_transport_fraction<=1):
        raise ValueError(
            'Distributed bank transport fraction must be finite and within zero to one')
    if (isinstance(bank_distributed_critical_slope,bool) or
            not math.isfinite(bank_distributed_critical_slope) or
            bank_distributed_critical_slope<=0):
        raise ValueError('Distributed bank critical slope must be finite and positive')
    noncohesive_bank_failure=(bank_failure and
                              bank_toe_erosion_mode in (
                                  'noncohesive_transport_capacity',
                                  'distributed_slope_transport'))
    physical_toe_bank_failure=(bank_failure and
                               bank_toe_erosion_mode=='noncohesive_transport_capacity')
    distributed_bank_failure=(bank_failure and
                              bank_toe_erosion_mode=='distributed_slope_transport')
    if not isinstance(bank_front_tracking,bool):
        raise ValueError('bank_front_tracking must be true or false')
    if bank_front_tracking:
        raise ValueError('Persistent bank-front tracking is retired; use bed-elevation bank evolution')
    if bank_front_tracking and not distributed_bank_failure:
        raise ValueError(
            'Persistent bank-front tracking requires enabled '
            'distributed_slope_transport bank failure')
    bank_failure_values=(bank_toe_erodibility_m_pa_s,bank_toe_critical_shear_pa,
                         bank_toe_exponent,bank_critical_undercut_m)
    if any(isinstance(value,bool) or not math.isfinite(value) for value in bank_failure_values) or \
            bank_toe_erodibility_m_pa_s<0 or bank_toe_critical_shear_pa<0 or \
            bank_toe_exponent<=0 or bank_critical_undercut_m<0:
        raise ValueError('Invalid process-separated bank-failure settings')
    if bank_failure and (fractions is None or not layered):
        raise ValueError('Process-separated bank failure requires multiple fractions and a positive active-layer thickness')
    if bank_failure and not bank_mask.any():
        raise ValueError('Process-separated bank failure requires at least one erodible-bank cell')
    if bank_failure and bank_critical_undercut_m<=0:
        raise ValueError('Enabled bank failure requires a positive critical undercut')
    if bank_failure and not noncohesive_bank_failure and bank_toe_erodibility_m_pa_s<=0:
        raise ValueError('Cohesive bank failure requires positive toe erodibility')
    if noncohesive_bank_failure and bedload==0:
        raise ValueError('Noncohesive bank failure requires an active bed-load formulation')
    if noncohesive_bank_failure and any(value.sediment_type!='sand' for value in fractions):
        raise ValueError('Noncohesive bank failure requires noncohesive sediment fractions')
    if noncohesive_bank_failure and exchange!=0:
        raise ValueError('Noncohesive bank failure requires bed load without suspended exchange')
    if bank_failure and bank_enabled:
        raise ValueError('Legacy bank transfer/avalanche and process-separated bank failure cannot both be enabled')
    if bank_failure and not noncohesive_bank_failure and (manning<=0 if local_manning is None else bool((local_manning<=0).any())):
        raise ValueError('Process-separated bank failure requires positive Manning roughness in every cell')
    if not isinstance(bedload_adaptation,bool):
        raise ValueError('Bed-load adaptation switch must be true or false')
    if (isinstance(bedload_adaptation_length_diameters,bool) or
            not math.isfinite(bedload_adaptation_length_diameters) or
            bedload_adaptation_length_diameters<=0):
        raise ValueError('Bed-load adaptation length must be a positive number of grain diameters')
    if bedload_adaptation and bedload==0:
        raise ValueError('Bed-load adaptation requires an active bed-load formulation')
    if isinstance(morphological_factor_series,dict):
        morphological_factor_series=CutMorphologicalFactorSeries(**morphological_factor_series)
    if morphological_factor_series is not None and not isinstance(
            morphological_factor_series,CutMorphologicalFactorSeries):
        raise ValueError('morphological_factor_series must be CutMorphologicalFactorSeries')
    if (isinstance(morphological_factor,bool) or not math.isfinite(morphological_factor) or
            not 1<=morphological_factor<=20):
        raise ValueError('Morphological factor must be within the qualified 1 to 20 range')
    morphological_acceleration=(morphological_factor_series is not None or morphological_factor!=1.)
    if morphological_acceleration and bedload==0:
        raise ValueError('Morphological acceleration requires an active bed-load formulation')
    if morphological_acceleration and exchange!=0:
        raise ValueError('Bed-load morphological acceleration is not qualified with suspended exchange')
    if morphological_acceleration and (bank_enabled or bank_failure):
        raise ValueError('Morphological acceleration is not qualified with bank erosion or bank failure')
    if morphological_acceleration and any(
            getattr(boundary,'hydraulic_series',None) is not None for boundary in boundaries.values()):
        raise ValueError('Morphological acceleration is not qualified with time-varying hydraulic boundaries')
    if not isinstance(bedform_roughness,bool):
        raise ValueError('Bedform-roughness switch must be true or false')
    if bedform_roughness:
        if hydraulic_roughness_mode!='manning':
            raise ValueError('The current bedform-feedback operator requires Manning hydraulic roughness')
        if bedload!=2 or mpm_friction_mode!=1:
            raise ValueError('Bedform roughness requires roughness-corrected MPM bed load')
        if exchange!=0:
            raise ValueError('Bedform roughness is not qualified with suspended exchange')
        composed_boundary = next(iter(boundaries.values())) if len(boundaries) == 1 else None
        recursive_bank_bedform = (
            amr_settings is not None and
            getattr(amr_settings, 'enabled', False) is True and
            getattr(amr_settings, 'subcycling', False) is True and
            bank_failure and
            bank_toe_erosion_mode in (
                'noncohesive_transport_capacity',
                'distributed_slope_transport') and
            not bank_front_tracking and general_avalanche and
            not secondary_flow and not bedload_adaptation and
            not morphological_acceleration and not bank_enabled and
            bed_slope == 0 and hiding_exposure == 0 and
            horizontal_viscosity_m2_s == 0. and
            active_layer_mode == 'constant' and substrate_layer_count == 1 and
            composed_boundary is not None and composed_boundary.mode == 'stage' and
            composed_boundary.hydraulic_series is None and
            composed_boundary.sediment_series is None and
            not composed_boundary.fraction_sediment_series and
            not composed_boundary.fraction_concentration_kg_m3 and
            composed_boundary.concentration_kg_m3 == 0. and
            composed_boundary.bedload_boundary_mode != 'equilibrium_capacity' and
            composed_boundary.bed_input_kg_m_s == 0. and
            composed_boundary.bed_input_kg_s in (None, 0.) and
            not composed_boundary.fraction_bed_input_kg_s)
        if (any((secondary_flow,bedload_adaptation,morphological_acceleration,bank_enabled,
                 bank_failure,general_avalanche)) and not recursive_bank_bedform):
            raise ValueError('Bedform roughness is not qualified with another experimental morphology process')
        if fractions is not None and any(value.sediment_type!='sand' for value in fractions):
            raise ValueError('Bedform roughness requires noncohesive sediment fractions')
        if manning<=0 if local_manning is None else bool((local_manning<=0).any()):
            raise ValueError('Bedform roughness requires positive Manning roughness in every cell')
        if any(boundary.mode=='inlet_normal' for boundary in boundaries.values()):
            raise ValueError('Bedform roughness is not qualified with normal-depth inlet control')
    if fractions is None and exchange==3:
        raise ValueError('Cohesive or mixed exchange requires at least two sediment fractions')
    if fractions is not None and any(value.sediment_type=='mud' for value in fractions) and exchange!=3:
        raise ValueError('Mud fractions require cohesive or mixed suspended exchange')
    if bedload==9:
        unsupported_gaia=(scheme!=6 or exchange!=0 or bed_slope!=4 or
            not gaia_secondary_flow or not secondary_flow_transport_direction or
            secondary_flow_momentum_beta!=0 or mpm_friction_mode!=0 or hiding_exposure!=0 or
            bedload_adaptation or bedform_roughness or morphological_acceleration or
            bank_enabled or bank_failure or general_avalanche or horizontal_viscosity_m2_s!=0 or
            dynamic_active or substrate_layer_count!=1 or
            (amr_settings is not None and amr_settings.enabled) or
            any(value.sediment_type!='sand' for value in fractions) or
            any(boundary.bedload_boundary_mode in (
                    'equilibrium_capacity','prescribed_fixed_face_bed') and
                boundary.mode!='total_discharge' for boundary in boundaries.values()))
        if unsupported_gaia:
            raise ValueError('TELEMAC-GAIA law 9 requires the exact fixed-grid scheme6 bedload-only slope4/model-gaia scope')
    elif any(boundary.bedload_boundary_mode=='prescribed_fixed_face_bed'
             for boundary in boundaries.values()):
        raise ValueError(
            'A prescribed fixed-face bed-load boundary is supported only by TELEMAC-GAIA law 9')
    elif secondary_flow_model=='gaia':
        raise ValueError('TELEMAC-GAIA secondary model is reserved for bed-load law 9')
    active={f.bc_id:f.kind for f in mesh.faces if f.kind in ('inlet','outlet')}
    if set(active)!=set(boundaries) or not all(isinstance(b,CutBoundary) for b in boundaries.values()):
        raise ValueError('Exactly one CutBoundary per selected logical ID is required')
    if fractions is not None:
        known=set(fraction_ids)
        for name,spec in boundaries.items():
            supplied=(set(spec.fraction_concentration_kg_m3)|set(spec.fraction_bed_input_kg_s)|
                      set(spec.fraction_sediment_series))
            if supplied-known:raise ValueError(f'Boundary {name} names an unknown sediment fraction')
            if (spec.concentration_kg_m3 or spec.bed_input_kg_m_s or spec.bed_input_kg_s is not None or
                    spec.sediment_series is not None):
                raise ValueError('Multi-fraction boundaries require per-fraction sediment fields')
    elif any(spec.fraction_sediment_series for spec in boundaries.values()):
        raise ValueError('Per-fraction sediment time series require multiple sediment fractions')
    sections={} if sections is None else dict(sections)
    for name,section in sections.items():
        if name in boundaries and boundaries[name].mode in (
                'inlet_stage','inlet_depth','inlet_normal','inlet_critical'):
            if section.wetting_policy!='all_submerged':
                raise ValueError('Production inlet sections must include all submerged portions')
    for name,spec in boundaries.items():
        if spec.is_inlet!=(active[name]=='inlet'):raise ValueError('Boundary kind/mode mismatch')
        if not spec.is_inlet:continue
        rows=[f for f in mesh.faces if f.bc_id==name]
        normal=np.array(rows[0].outward_normal);tangent=np.array([-normal[1],normal[0]])
        origin=np.array(rows[0].start_xy)
        intervals=[]
        for f in rows:
            points=np.array([f.start_xy,f.end_xy])-origin
            if not np.allclose(f.outward_normal,normal,atol=1e-12,rtol=0) or abs(points@normal).max()>1e-8:
                raise ValueError('Noncollinear inlet needs an explicit allocation/control rule')
            intervals.append(sorted(points@tangent))
        intervals.sort()
        if any(abs(b[0]-a[1])>1e-8 for a,b in zip(intervals,intervals[1:])):
            raise ValueError('Disconnected or overlapping inlet pieces require explicit allocation')
    ids={name:k+1 for k,name in enumerate(sorted(active))}
    resolved_bed_input={}
    resolved_fraction_input={}
    for name in ids:
        spec=boundaries[name]
        total_length=sum(f.length_m for f in mesh.faces if f.bc_id==name)
        if spec.mode=='weir' and spec.weir_width_m is not None and spec.weir_width_m>total_length*(1+1e-12):
            raise ValueError('Weir width cannot exceed the selected boundary-face length')
        per_width=spec.bed_input_kg_m_s
        if spec.bed_input_kg_s is not None:
            if not spec.is_inlet or total_length<=0:
                raise ValueError('Total bedload input requires a nonempty inlet ID')
            per_width=spec.bed_input_kg_s/total_length
        resolved_bed_input[name]=dict(per_width_kg_m_s=per_width,
            total_kg_s=per_width*total_length,boundary_length_m=total_length)
        if fractions is not None:
            resolved_fraction_input[name]={fraction_id:{
                'concentration_kg_m3':spec.fraction_concentration_kg_m3.get(fraction_id,0.),
                'bed_input_kg_s':spec.fraction_bed_input_kg_s.get(fraction_id,0.),
                'time_series':(None if fraction_id not in spec.fraction_sediment_series else
                               asdict(spec.fraction_sediment_series[fraction_id]))}
                for fraction_id in fraction_ids}
    equilibrium_bedload=any(b.bedload_boundary_mode=='equilibrium_capacity' for b in boundaries.values())
    series_bedload=any(b.sediment_series is not None and max(b.sediment_series.bed_input_kg_s)>0
                       for b in boundaries.values())
    fraction_bedload=(fractions is not None and any(row['bed_input_kg_s']>0
        for values in resolved_fraction_input.values() for row in values.values()))
    fraction_series_bedload=(fractions is not None and any(
        row['time_series'] is not None and max(row['time_series']['bed_input_kg_s'])>0
        for values in resolved_fraction_input.values() for row in values.values()))
    if bedload==0 and (equilibrium_bedload or any(v['total_kg_s']>0 for v in resolved_bed_input.values()) or
                       series_bedload or fraction_bedload or fraction_series_bedload):
        raise ValueError('Bedload supply requires an enabled bedload law')
    has_sediment_series=any(b.sediment_series is not None for b in boundaries.values())
    has_fraction_sediment_series=any(b.fraction_sediment_series for b in boundaries.values())
    has_hydraulic_series=any(b.hydraulic_series is not None for b in boundaries.values())
    source_total_discharge=any(spec.mode=='total_discharge' for spec in boundaries.values())
    prescribed_trace=any(spec.mode in ('prescribed_trace','prescribed_depth_trace') for spec in boundaries.values())
    extended_hydraulics=any(spec.mode not in ('discharge','stage','supercritical') for spec in boundaries.values())
    version=(9 if source_total_discharge or prescribed_trace else (8 if equilibrium_bedload else (7 if has_hydraulic_series else (6 if extended_hydraulics else (5 if local_manning is not None else
             (4 if has_sediment_series else (3 if bed_slope or any(b.bed_input_kg_s is not None for b in boundaries.values()) else
              (2 if bedload==2 else 1))))))))
    if sediment.reference_height_factor!=1. or sediment.roughness_mode!='fixed':
        version=10
    if any(f.bed_elevation_m is not None for f in mesh.faces):
        version=11
        for f in mesh.faces:
            if f.bed_elevation_m is None:continue
            if not math.isfinite(f.bed_elevation_m) or f.kind!='inlet' or boundaries[f.bc_id].mode not in (
                    'discharge','inlet_stage','inlet_depth','inlet_normal','inlet_critical'):
                raise ValueError('Sampled boundary terrain requires a finite mode-1 inlet elevation')
    production_section=any(f.section_bed_start_m is not None or
                           f.section_bed_end_m is not None for f in mesh.faces)
    if production_section:
        for f in mesh.faces:
            supplied=(f.section_bed_start_m is not None,f.section_bed_end_m is not None)
            if any(supplied)!=all(supplied):
                raise ValueError('Production section faces require both terrain endpoint elevations')
            if not any(supplied):
                continue
            if (f.bed_elevation_m is None or f.kind!='inlet' or
                    not all(math.isfinite(value) for value in
                            (f.section_bed_start_m,f.section_bed_end_m))):
                raise ValueError('Production section geometry requires finite sampled inlet terrain')
    if fractions is not None:
        # Laws 11--14 are authorized by their v46 outer envelope; they must
        # not force the nested multi-fraction deck into v16's cohesive +
        # active-layer layout.  The nested version remains exactly the one
        # required by the pre-v46 physical features.
        extended=(bedload in (3,4,5,6,7,9) or exchange==3 or
                  any(value.sediment_type=='mud' for value in fractions))
        version=16 if extended else (15 if has_fraction_sediment_series else (14 if hiding_exposure else (13 if layered else 12)))
    if bank_enabled or bank_failure:version=17
    # The legacy v18 nested layout couples the MPM closure flag to the
    # multi-fraction/active-layer record shape.  Q3-C authenticates the
    # corrected MPM closure in its outer component/forcing profile and may
    # own one compact physical reservoir, so do not corrupt that compact
    # layout merely to repeat the legacy flag.  The native v54 reader restores
    # mode 1 from the authenticated route before validating the law.
    if mpm_friction_mode and not (
            capacity_foundation_profile and fractions is None):
        version=18
    if layered and substrate_layer_count>1:version=19
    if has_fraction_sediment_series and not layered:
        raise ValueError('Per-fraction time series require an active-layer thickness')
    roughness_materials=(sediment,) if fractions is None else fractions
    if any(value.roughness_mode=='manning' for value in roughness_materials):
        coefficients=local_manning if local_manning is not None else [manning]
        if min(coefficients)<=0:
            raise ValueError('Manning-derived sediment roughness requires positive Manning in every cell')
    if version>=3:
        for name,spec in boundaries.items():
            maximum_bed=resolved_bed_input[name]['total_kg_s']
            if fractions is not None:
                maximum_bed=max(max(row['bed_input_kg_s'],
                    max(row['time_series']['bed_input_kg_s']) if row['time_series'] is not None else 0.)
                    for row in resolved_fraction_input[name].values())
            if spec.sediment_series is not None:
                maximum_bed=max(spec.sediment_series.bed_input_kg_s)
            maximum_q=(max(spec.hydraulic_series.discharge_m3_s)
                       if spec.hydraulic_series is not None else spec.discharge_m3_s)
            if spec.is_inlet and maximum_bed>0 and maximum_q<=0:
                raise ValueError('A positive total bedload supply requires positive water discharge')
            if spec.is_inlet and spec.hydraulic_series is not None and maximum_bed>0:
                times=set(spec.hydraulic_series.times_s)
                if spec.sediment_series is not None:times.update(spec.sediment_series.times_s)
                for series in spec.fraction_sediment_series.values():times.update(series.times_s)
                for time_s in times:
                    water=_held_linear(spec.hydraulic_series.times_s,spec.hydraulic_series.discharge_m3_s,time_s)
                    if spec.sediment_series is not None:
                        bed=_held_linear(spec.sediment_series.times_s,spec.sediment_series.bed_input_kg_s,time_s)
                    elif spec.fraction_sediment_series:
                        bed=sum(_held_linear(series.times_s,series.bed_input_kg_s,time_s)
                                for series in spec.fraction_sediment_series.values())
                        bed+=sum(spec.fraction_bed_input_kg_s.values())
                    else:bed=maximum_bed
                    if bed>0 and water<=0:raise ValueError('Positive bedload forcing requires positive hydraulic-series discharge')
    # Keep the typed roughness work depth distinct from the legacy Manning
    # friction/Chezy floor.  A legacy Manning deck serialises only envelope
    # 36, so its native typed-roughness value is exactly zero; the effective
    # depth used by boundary hydraulics is nevertheless the maximum of both
    # independently owned controls.  Conflating them made the JSON receipt
    # disagree with the native parser for otherwise valid v36/v49 decks.
    roughness_work_depth=(max(.01,manning_friction_chezy_minimum_depth_m)
                          if (hydraulic_roughness_mode!='manning' or
                              white_colebrook_chzk2d) else 0.)
    boundary_roughness_work_depth=max(
        roughness_work_depth,manning_friction_chezy_minimum_depth_m)
    resolved_hydraulic_series={name:resolve_hydraulic_series(
                                   boundaries[name],sections.get(name),default_manning_n=manning,
                                   hydraulic_roughness_mode=hydraulic_roughness_mode,
                                   white_colebrook_chzk2d=white_colebrook_chzk2d,
                                   roughness_minimum_work_depth_m=boundary_roughness_work_depth,
                                   defer_live_section_control=production_section)
                               for name in ids}
    resolved_targets={}
    for name in ids:
        resolved=resolved_hydraulic_series[name]
        if resolved is None:
            spec=boundaries[name]
            if production_section and spec.mode in ('inlet_normal','inlet_critical'):
                resolved_targets[name]=_deferred_live_section_target(spec)
            else:
                resolved_targets[name]=resolve_boundary_target(
                    spec,sections.get(name),default_manning_n=manning,
                    hydraulic_roughness_mode=hydraulic_roughness_mode,
                    white_colebrook_chzk2d=white_colebrook_chzk2d,
                    roughness_minimum_work_depth_m=boundary_roughness_work_depth)
        else:
            first=resolved['knots'][0]
            resolved_targets[name]={key:value for key,value in first.items()
                                    if key not in ('time_s','discharge_m3_s','requested_control_m')}
            resolved_targets[name]['from_hydraulic_series_first_knot_s']=first['time_s']
    for name,target in resolved_targets.items():
        spec=boundaries[name]
        if spec.mode=='weir':
            target['weir_coefficient']=spec.weir_coefficient
            target['weir_width_m']=resolved_bed_input[name]['boundary_length_m'] if spec.weir_width_m is None else spec.weir_width_m
    # Version 21 is an envelope around the established state layouts. It adds
    # physical section endpoints and live normal/critical controls without
    # forcing a one-material case into the multi-fraction record layout.
    serialized_version=21 if production_section else (20 if settling_mode else version)
    path=filesystem_path(path)
    lines=[str(serialized_version)]
    if serialized_version in (20,21):lines.append(str(version))
    lines.extend([f'{len(mesh.cells)} {len(mesh.faces)} {len(ids)} {mesh.grid.nx} {mesh.grid.ny}',
           ' '.join(format(v,'.17g') for v in [9.81,dry_tolerance_m,sediment.bulk_kg_m3,manning,settling_velocity_m_s,sediment.d50_m,sediment.d90_m,sediment.grain_density_kg_m3,sediment.water_density_kg_m3,sediment.z0_m,sediment.minimum_exchange_depth_m,mesh.grid.dx_m,mesh.grid.dy_m]),
           ])
    if fractions is not None:
        lines.append(str(len(fractions)))
        lines.extend(' '.join(format(value,'.17g') for value in (
            [fraction.bulk_kg_m3,fraction.settling_velocity_m_s,fraction.d50_m,fraction.d90_m,
             fraction.grain_density_kg_m3,fraction.z0_m,fraction.minimum_exchange_depth_m,
             fraction.minimum_bedload_depth_m,fraction.reference_height_factor,
             int(fraction.roughness_mode=='manning'),
             int(fraction.sediment_type=='mud'),fraction.critical_erosion_shear_pa,
             fraction.critical_deposition_shear_pa,fraction.erosion_parameter_kg_m2_s,
             fraction.erosion_exponent] if version>=16 else
            [fraction.bulk_kg_m3,fraction.settling_velocity_m_s,fraction.d50_m,fraction.d90_m,
             fraction.grain_density_kg_m3,fraction.z0_m,fraction.minimum_exchange_depth_m,
             fraction.minimum_bedload_depth_m,fraction.reference_height_factor,
             int(fraction.roughness_mode=='manning')])) for fraction in fractions)
        if layered:lines.append(format(active_layer_thickness_m,'.17g'))
        if version>=19:lines.append(f'{substrate_layer_count} {substrate_layer_thickness_m:.17g}')
        if version>=14:lines.append(str(hiding_exposure))
        if version>=16:lines.append(f'{critical_mud_fraction:.17g} {transport_calibration_factor:.17g}')
        if version>=17:
            lines.append(' '.join(format(value,'.17g') for value in
                (bank_erosion_fraction,bank_wet_slope,bank_dry_slope,bank_avalanche_time_s,
                 bank_wet_depth_switch_m,int(bank_avalanche))))
        if version>=18:lines.append(str(mpm_friction_mode))
    time_line_index=len(lines)
    lines.extend([f'{end_time_s:.17g} {dt_max_s:.17g} {output_interval_s:.17g}',f'{scheme} {bedload} {exchange}'])
    if version==2:lines.append(f'{mpm_cf:.17g} {mpm_critical_shields:.17g}')
    if version>=3:
        lines.append(f'{mpm_cf:.17g} {mpm_critical_shields:.17g}')
        lines.append(f'{bed_slope} {bed_slope_alfabs:.17g} {bed_slope_alfabn:.17g} {bed_friction_angle_deg:.17g} {sediment.minimum_bedload_depth_m:.17g}')
    def iter_mesh_lines():
        for c in mesh.cells:
            row=[c.i+1,c.j+1,c.area_m2,*c.centroid_xy,rock[c.cell_id],*q[c.cell_id]]
            if version>=5:row.append(manning if local_manning is None else local_manning[c.cell_id])
            if version>=17:row.append(int(bank_mask[c.cell_id]))
            yield ' '.join(format(v,'.17g') for v in row)
        for f in mesh.faces:
            row=[f.owner+1,f.neighbour+1,ids.get(f.bc_id,0),*f.outward_normal,f.length_m,*((np.array(f.start_xy)+f.end_xy)/2)]
            if version>=11:
                if fractions is None:
                    owner_bed=rock[f.owner]+q[f.owner,4]/sediment.bulk_kg_m3
                else:
                    owner_bed=rock[f.owner]+sum(
                        sum(q[f.owner,3+len(fractions)+layer*len(fractions)+index]
                            for layer in range(bed_layers))/fraction.bulk_kg_m3
                        for index,fraction in enumerate(fractions))
                row.extend((int(f.bed_elevation_m is not None),0. if f.bed_elevation_m is None else
                    f.bed_elevation_m-owner_bed))
                if production_section:
                    if f.section_bed_start_m is None:
                        row.extend((0.,0.))
                    else:
                        tangent=np.array([-f.outward_normal[1],f.outward_normal[0]])
                        if tangent[0]<0 or (tangent[0]==0 and tangent[1]<0):tangent=-tangent
                        forward=float((np.array(f.end_xy)-np.array(f.start_xy))@tangent)>=0
                        left=f.section_bed_start_m if forward else f.section_bed_end_m
                        right=f.section_bed_end_m if forward else f.section_bed_start_m
                        row.extend((left-owner_bed,right-owner_bed))
            yield ' '.join(format(v,'.17g') for v in row)
    suffix=[]
    for name in ids:
        spec=boundaries[name]
        native_bed=resolved_bed_input[name]['total_kg_s'] if version>=3 else resolved_bed_input[name]['per_width_kg_m_s']
        target=resolved_targets[name];hydraulic=resolved_hydraulic_series[name]
        base_q=spec.discharge_m3_s if hydraulic is None else hydraulic['knots'][0]['discharge_m3_s']
        width=(resolved_bed_input[name]['boundary_length_m'] if spec.weir_width_m is None else spec.weir_width_m)
        row=[target['native_mode'],base_q,
             target['target_stage_m'] if target['target_stage_m'] is not None else 0.,
             0. if fractions is not None else spec.concentration_kg_m3,
             0. if fractions is not None else native_bed]
        if version>=6:row.extend((spec.weir_coefficient,width if spec.mode=='weir' else 0.))
        suffix.append(' '.join(format(v,'.17g') for v in row))
    if fractions is not None:
        for name in ids:
            values=resolved_fraction_input[name]
            suffix.append(' '.join(format(values[fraction_id]['concentration_kg_m3'],'.17g')
                                  for fraction_id in fraction_ids))
            suffix.append(' '.join(format(values[fraction_id]['bed_input_kg_s'],'.17g')
                                  for fraction_id in fraction_ids))
    if version>=15 and ids:
        suffix.append(' '.join(str(len(boundaries[name].fraction_sediment_series[fraction_id].times_s)
                                  if fraction_id in boundaries[name].fraction_sediment_series else 0)
                              for name in ids for fraction_id in fraction_ids))
        for name in ids:
            for fraction_id in fraction_ids:
                series=boundaries[name].fraction_sediment_series.get(fraction_id)
                if series is None:continue
                suffix.extend(' '.join(format(v,'.17g') for v in row) for row in
                             zip(series.times_s,series.concentration_kg_m3,series.bed_input_kg_s))
    if version>=4 and ids:
        # One policy/count entry per sorted logical ID, followed by each ID's
        # contiguous rows. Code 1 preserves the established no-reversal
        # contract. Code 0 is admitted only by the explicit v48
        # ``flow2dh_return`` policy so Thatcher--Harleman can observe a real
        # outflow-to-inflow transition; it is never a legacy substitution.
        suffix.append(' '.join(
            '0' if boundaries[name].reversal_policy=='flow2dh_return' else '1'
            for name in ids))
        suffix.append(' '.join(str(len(boundaries[name].sediment_series.times_s)
                                  if boundaries[name].sediment_series is not None else 0) for name in ids))
        for name in ids:
            series=boundaries[name].sediment_series
            if series is None:continue
            suffix.extend(' '.join(format(v,'.17g') for v in row) for row in
                         zip(series.times_s,series.concentration_kg_m3,series.bed_input_kg_s))
    if version>=7 and ids:
        suffix.append(' '.join(str(len(boundaries[name].hydraulic_series.times_s)
                                  if boundaries[name].hydraulic_series is not None else 0) for name in ids))
        for name in ids:
            resolved=resolved_hydraulic_series[name]
            if resolved is None:continue
            suffix.extend(' '.join(format(v,'.17g') for v in
                (row['time_s'],row['discharge_m3_s'],
                 0. if row['target_stage_m'] is None else row['target_stage_m']))
                 for row in resolved['knots'])
    if version>=8 and ids:
        # Code 0 preserves the prescribed total/per-width contract. Code 1
        # evaluates the selected native bed-load law on the hydraulic trace.
        boundary_mode_code={
            'prescribed':0,
            'equilibrium_capacity':1,
            'prescribed_fixed_face_bed':2,
        }
        suffix.append(' '.join(str(boundary_mode_code[boundaries[name].bedload_boundary_mode])
                              for name in ids))
    if version>=10:
        suffix.append(f'{sediment.reference_height_factor:.17g} {int(sediment.roughness_mode=="manning")}')
    if settling_mode or production_section:
        # Optional runtime-control footer. Keeping this independent of the
        # cumulative deck version makes the closure available to both the
        # one-material and multi-fraction state layouts.
        suffix.append(f'{cfl:.17g} {settling_mode} {hindered_settling_reference_concentration_kg_m3:.17g}')
    elif cfl!=.45:
        suffix.append(f'{cfl:.17g}')
    if production_section:
        for name in ids:
            control={'inlet_normal':1,'inlet_critical':2}.get(boundaries[name].mode,0)
            suffix.append(f'{control} {boundaries[name].energy_slope:.17g}')
    hierarchy_rows=None
    hierarchy_prefix=()
    if amr_hierarchy is not None:
        from .amr import AMRSettings, iter_native_hierarchy_lines
        amr_settings = AMRSettings(enabled=True) if amr_settings is None else amr_settings
        fine = amr_hierarchy.finest
        if (fine.grid != mesh.grid or len(fine.cells) != len(mesh.cells) or
                len(fine.faces) != len(mesh.faces) or
                any(a.vertices_xy != b.vertices_xy for a, b in zip(fine.cells, mesh.cells)) or
                any((a.owner,a.neighbour,a.start_xy,a.end_xy)!=(b.owner,b.neighbour,b.start_xy,b.end_xy)
                    for a,b in zip(fine.faces,mesh.faces))):
            raise ValueError('AMR case initial fields must use the exact finest hierarchy mesh')
        if production_section:
            amr_hierarchy = _amr_sampled_section_geometry(amr_hierarchy, mesh)
        if not (settling_mode or production_section):
            if cfl != .45:
                suffix[-1] = f'{cfl:.17g} 0 {hindered_settling_reference_concentration_kg_m3:.17g}'
            else:
                suffix.append(f'{cfl:.17g} 0 {hindered_settling_reference_concentration_kg_m3:.17g}')
        hierarchy_rows=lambda:iter_native_hierarchy_lines(
            amr_hierarchy,amr_settings,ids,amr_minimum_levels)
        lines.insert(0, '22')
        time_line_index += 1
        serialized_version = 22
    elif amr_settings is not None and amr_settings.enabled:
        raise ValueError('Enabled AMR requires a hierarchy')
    if bank_enabled:
        # Version 23 wraps any established fixed/adaptive deck. It adds the
        # physical length L used by D=L²/(2T) without changing state layouts.
        lines.insert(0,'23')
        lines.insert(1,format(bank_relaxation_length_m,'.17g'))
        time_line_index += 2
        serialized_version = 23
    native_binary_path=None
    native_binary_sha256=None
    if compact_sidecars:
        if hierarchy_rows is not None:
            prefix_iterator=iter(hierarchy_rows())
            hierarchy_prefix=(next(prefix_iterator),next(prefix_iterator))
            # Binary tree records use v6: the same controls/header semantics
            # as v5, with bulk node/face arrays read from the native stream.
            values=hierarchy_prefix[0].split()
            values[0]='6'
            hierarchy_prefix=(' '.join(values),hierarchy_prefix[1])
        native_binary_path=path.with_suffix(path.suffix+'.bin')
        native_binary_sha256=_write_native_binary(
            native_binary_path,mesh,q,rock,local_manning,manning,bank_mask,
            ids,version,production_section,sediment,fractions,bed_layers,
            amr_hierarchy,amr_minimum_levels)
        lines.insert(0,'24')
        lines.insert(1,f'{native_binary_path.name} {native_binary_sha256}')
        time_line_index+=2
        serialized_version=24
    if hiding_exposure in (4,5):
        # Version 25 wraps any established fixed/adaptive/binary deck with the
        # two calibration exponents required by Delft3D modes 4 and 5.
        lines.insert(0,'25')
        lines.insert(1,f'{hiding_parker_alpha:.17g} {hiding_wu_exponent:.17g}')
        time_line_index+=2
        serialized_version=25
    if dynamic_active:
        # Version 26 is a pure configuration envelope around every established
        # state layout and optional binary/AMR/hiding envelope.
        lines.insert(0,'26')
        lines.insert(1,f'1 {active_layer_depth_fraction:.17g} {active_layer_minimum_thickness_m:.17g}')
        time_line_index+=2
        serialized_version=26
    if general_avalanche:
        # Version 27 is an experimental configuration envelope. Unlike the
        # v17 bank mode, every retained bed cell may donate across an internal
        # face; domain and obstacle boundaries remain closed by construction.
        lines.insert(0,'27')
        lines.insert(1,' '.join(format(value,'.17g') for value in
            (int(general_avalanche),general_avalanche_wet_slope,
             general_avalanche_dry_slope,general_avalanche_time_s,
             general_avalanche_wet_depth_switch_m,
             general_avalanche_relaxation_length_m)))
        time_line_index+=2
        serialized_version=27
    if secondary_flow and secondary_flow_model=='equilibrium':
        # Version 28 is an outer experimental envelope and therefore composes
        # with the established state, AMR, active-layer, and avalanche layouts.
        lines.insert(0,'28')
        lines.insert(1,f'{int(secondary_flow)} {secondary_flow_espir:.17g}')
        time_line_index+=2
        serialized_version=28
    if bank_failure and not noncohesive_bank_failure:
        # Version 29 is an outer experimental envelope.  The four values are
        # the excess-shear toe law and the resolved critical undercut.  It
        # composes with AMR and the qualified domain-wide repose operator.
        lines.insert(0,'29')
        lines.insert(1,' '.join(format(value,'.17g') for value in
            (int(bank_failure),bank_toe_erodibility_m_pa_s,
             bank_toe_critical_shear_pa,bank_toe_exponent,
             bank_critical_undercut_m)))
        time_line_index+=2
        serialized_version=29
    if bedload_adaptation:
        # Version 30 adds a steady streamline advection-relaxation closure.
        # Each class uses L_g = factor * D50_g and the disabled path remains
        # byte-for-byte identical to the qualified local-equilibrium solver.
        lines.insert(0,'30')
        lines.insert(1,f'{int(bedload_adaptation)} {bedload_adaptation_length_diameters:.17g}')
        time_line_index+=2
        serialized_version=30
    if morphological_acceleration:
        # Version 31 follows Delft3D's dtmor=dts*MorFac semantics.  A zero
        # knot count selects the constant value; otherwise endpoint-held,
        # block-interpolated rows follow before the composable base deck.
        series=morphological_factor_series
        prefix=['31',f'{0 if series is None else len(series.times_s)} {morphological_factor:.17g}']
        if series is not None:
            prefix.extend(f'{time_s:.17g} {factor:.17g}' for time_s,factor in zip(series.times_s,series.factors))
        lines[0:0]=prefix
        time_line_index+=len(prefix)
        serialized_version=31
    if bedform_roughness:
        # Version 32 enables the qualified local-equilibrium Fredsoe/Van Rijn
        # feedback without adding prognostic state, so AMR and restart remain
        # exact and the disabled v31-and-earlier byte stream is unchanged.
        lines.insert(0,'32')
        lines.insert(1,'1')
        time_line_index+=2
        serialized_version=32
    if horizontal_viscosity_m2_s>0:
        # An optional envelope keeps all disabled legacy decks byte-identical.
        lines[0:0]=['33',f'{horizontal_viscosity_m2_s:.17g}']
        time_line_index+=2
        serialized_version=33
    if horizontal_viscosity_formulation=='velocity_laplacian':
        # Version 34 is deliberately emitted only for the nondefault mode and
        # wraps v33. Version 35 then selects the consistent nonorthogonal
        # normal-gradient flux. Archived v34 mode-1 decks remain readable and
        # retain their original two-point arithmetic bit for bit.
        lines[0:0]=['34','1']
        time_line_index+=2
        serialized_version=34
        lines[0:0]=['35','1']
        time_line_index+=2
        serialized_version=35
    if manning_friction_chezy_minimum_depth_m>0:
        # Version 36 reproduces stock FLOW's shallow momentum-friction work
        # depth while retaining physical h in the coefficient.  Zero emits no
        # envelope, so every established deck remains byte-for-byte identical.
        lines[0:0]=['36',f'{manning_friction_chezy_minimum_depth_m:.17g}']
        time_line_index+=2
        serialized_version=36
    if prognostic_secondary_flow:
        # Version 37 is the sole secondary-flow declaration for the prognostic
        # model.  It is always the outer envelope and deliberately does not
        # contain the frozen v28 equilibrium declaration.  The public initial
        # q and compact binary layouts remain the established physical layout.
        lines[0:0]=['37',' '.join((
            '2',str(int(secondary_flow_transport_direction)),
            format(secondary_flow_espir,'.17g'),
            format(secondary_flow_momentum_beta,'.17g'),'1'))]
        time_line_index+=2
        serialized_version=37
    if physical_toe_bank_failure:
        # Version 38 is deliberately separate from the established v29
        # cohesive excess-shear declaration. It supplies bank material only
        # from accepted net erosive Exner divergence at mapped physical toes;
        # raw transport capacity is never interpreted as an erosion source.
        lines[0:0]=['38',' '.join((
            '1',format(bank_toe_capacity_fraction,'.17g'),
            format(bank_critical_undercut_m,'.17g')))]
        time_line_index+=2
        serialized_version=38
    if equilibrated_inlets:
        # Version 39 marks only the selected logical mode-6 IDs.  The legacy
        # stock-conveyance allocation remains the default and therefore keeps
        # every established deck byte-identical.  One physical longitudinal
        # acceleration then balances the selected velocity Laplacian and the
        # raw typed Chezy/Manning/White-Colebrook drag while the total-Q
        # constraint fixes its magnitude.
        selected=tuple(ids[name] for name in equilibrated_inlets)
        lines[0:0]=['39',str(len(selected)),' '.join(str(value) for value in selected)]
        time_line_index+=3
        serialized_version=39
    if distributed_bank_failure:
        # Version 40 implements the published local distributed-slope bank
        # supply rule on explicitly erodible material.  It wraps every prior
        # envelope so the disabled path and all archived decks remain exact.
        lines[0:0]=['40',' '.join((
            '1',format(bank_distributed_transport_fraction,'.17g'),
            format(bank_distributed_critical_slope,'.17g'),
            format(bank_critical_undercut_m,'.17g')))]
        time_line_index+=2
        serialized_version=40
    if bank_front_tracking:
        # Version 41 is a pure opt-in state-semantics envelope around v40.
        # The flag reserves persistent subcell bank-front state without
        # changing the established distributed-slope configuration record.
        # False emits no envelope, preserving every existing v40 deck byte for
        # byte and preventing an older runtime from silently accepting the new
        # persistence contract.
        lines[0:0]=['41',str(int(bank_front_tracking))]
        time_line_index+=2
        serialized_version=41
    if bedload==8 and not capacity_foundation_profile:
        # Version 42 makes the transport-law roughness contract explicit.
        # Engelund--Hansen in the T2 reference uses a constant Chezy value;
        # deriving it from local depth and Manning silently changes capacity.
        lines[0:0]=['42',format(engelund_hansen_chezy,'.17g')]
        time_line_index+=2
        serialized_version=42
    if bedload==9:
        # Version 43 is a distinct, fail-closed TELEMAC-GAIA correspondence:
        # Chollet-Cunge EH, ICR=1 skin correction, Talmon deviation and the
        # free-surface-gradient secondary-current stress addition.  It never
        # aliases the frozen law-8 / slope-3 / model-1 contract.
        lines[0:0]=['43',' '.join(format(value,'.17g') for value in gaia_values)]
        time_line_index+=2
        serialized_version=43
    if gaia_diagnostic_v44:
        # Version 44 is a validation-only outer envelope around an otherwise
        # complete v43 deck.  Keeping v43 nested verbatim makes the single
        # process removals auditable and leaves ordinary law-9 serialization
        # byte-for-byte unchanged.
        lines[0:0]=['44',' '.join(str(int(value)) for value in gaia_diagnostic_values)]
        time_line_index+=2
        serialized_version=44
    typed_roughness=(hydraulic_roughness_mode!='manning' or
                     white_colebrook_chzk2d or
                     sediment_shear_mode!='total')
    if typed_roughness:
        # Version 45 is a typed roughness/shear envelope around every existing
        # deck, including the exact v42/v43/v44 constitutive envelopes.  The
        # disabled legacy Manning/total-shear path emits no bytes.
        lines[0:0]=['45',' '.join(format(value,'.17g') for value in (
            roughness_codes[hydraulic_roughness_mode],
            int(white_colebrook_chzk2d),
            int(sediment_shear_mode=='soulsby_clarke_skin'),
            roughness_work_depth,skin_kinematic_viscosity_m2_s,
            skin_sand_roughness_m,skin_silt_roughness_m,
            skin_critical_mud_fraction_low,skin_critical_mud_fraction_high,
            skin_fluff_fraction))]
        time_line_index+=2
        serialized_version=45
    if resolved_capacity_parameter_rows is not None:
        # Version 46 is a profile- and restart-identity-bound outer envelope.
        # It never changes
        # v45 or any older byte stream.  Source-keyword order is pinned by the
        # schema digest; every fraction has a complete resolved row, including
        # optional defaults and identity-relevant inert fields.
        parameter_count=len(formula_contract(bedload).parameters)
        prefix=['46',f'{bedload} {len(resolved_capacity_parameter_rows)} {parameter_count} 1',
                capacity_parameter_schema_id,capacity_profile_schema_id]
        prefix.extend(' '.join(format(value,'.17g') for value in row.values())
                      for row in resolved_capacity_parameter_rows)
        lines[0:0]=prefix
        time_line_index+=len(prefix)
        serialized_version=46
    resolved_capacity_fraction_basis=(capacity_fraction_basis or 'dry_mass')
    capacity_fraction_basis_code={'dry_mass':1,'solid_volume':2}[
        resolved_capacity_fraction_basis]
    typed_constitutive_v47=(
        exchange==4 or capacity_fraction_basis is not None or
        constitutive_kinematic_viscosity_m2_s!=1.e-6 or
        suspended_adaptation_factor!=1. or settling_form_code!=0 or
        suspended_calibration!=1.)
    capacity_diagnostic_schema_id=capacity_diagnostic_contract_id()
    suspension_profile_schema_id=NATIVE_V47_SUSPENSION_PROFILE_SCHEMA_ID
    resolved_suspension_profile=None
    resolved_suspension_profile_document=None
    suspension_profile_instance_fingerprint='0'*64
    if settling_form_code==FLOW2DH_SETTLING_FORM:
        try:
            capacity_instance_profile=resolve_capacity_profile_from_code(
                bedload,('single',),dict(resolved_capacity_parameter_rows[0]))
            resolved_suspension_profile=resolve_suspension_profile(
                capacity_instance_profile,
                fraction_basis=resolved_capacity_fraction_basis,
                settling_fac_dss=[settling_fac_dss],
                sediment_d50_m=[sediment.d50_m],
                sediment_density_kg_m3=[sediment.grain_density_kg_m3],
                fac_tsd=suspended_adaptation_factor,
                sediment_threshold_m=sediment.minimum_exchange_depth_m,
                settling_reference_concentration_kg_m3=
                    hindered_settling_reference_concentration_kg_m3,
                water_density_kg_m3=sediment.water_density_kg_m3,
                temperature_mode=settling_temperature_mode,
                temperature_value_c=resolved_settling_temperature_c,
                suspended_calibration_factor=suspended_calibration,
                gravity_m_s2=9.81,
                neubc_sand=resolved_neubc_sand,
                neubc_mud=False,
                horizontal_diffusivity_mode='uniform_constant',
                horizontal_diffusivity_m2_s=
                    resolved_suspended_diffusivity,
                diffusion_dry_threshold_m=
                    resolved_suspended_diffusion_dry_threshold,
            )
            resolved_suspension_profile_document=suspension_profile_document(
                resolved_suspension_profile)
        except ValueError as error:
            raise ValueError(f'Invalid canonical suspension profile: {error}') from error
        suspension_profile_instance_fingerprint=resolved_suspension_profile.fingerprint
    if resolved_execution_profile is not None and exchange==4:
        suspension_components=[
            value for value in resolved_execution_profile.component_profiles
            if value.component_id=='suspension'
        ]
        suspension_document_sha256=hashlib.sha256(json.dumps(
            resolved_suspension_profile_document,sort_keys=True,separators=(',',':'),
            ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
        if (len(suspension_components)!=1 or
                suspension_components[0].fingerprint!=
                    resolved_suspension_profile.fingerprint or
                suspension_components[0].document_sha256!=
                    suspension_document_sha256):
            raise ValueError(
                'Execution profile does not bind the exact suspension component')
    if typed_constitutive_v47:
        # Version 47 is a strict outer envelope: the nested v46/base deck is
        # unchanged, while every newly typed constitutive choice and both
        # semantic contracts are native-validated and restart-bound.
        prefix=['47',' '.join(format(value,'.17g') for value in (
                    constitutive_kinematic_viscosity_m2_s,
                    capacity_fraction_basis_code,
                    suspended_adaptation_factor)),
                ' '.join(format(value,'.17g') for value in (
                    settling_form_code,
                    settling_fac_dss,
                    settling_temperature_mode_code,
                    resolved_settling_temperature_c,
                    hindered_settling_reference_concentration_kg_m3,
                    suspended_calibration,
                    sediment_activity_contract_code,
                    wet_mask_provider_code)),
                capacity_diagnostic_schema_id,
                suspension_profile_schema_id,
                FLOW2DH_EXCHANGE_SEMANTIC_ID,
                FLOW2DH_SETTLING_SEMANTIC_ID,
                suspension_profile_instance_fingerprint]
        lines[0:0]=prefix
        time_line_index+=len(prefix)
        serialized_version=47
    if flow2dh_boundary_diffusion_v48:
        # Version 48 composes the source-audited FLOW 2DH open-boundary and
        # horizontal-diffusion operators around the complete v47 route.  The
        # global sand Neumann switch follows FLOW; return times remain one
        # authenticated value per logical boundary and are expanded to every
        # native face only after the nested mesh has been read.
        prefix=['48',' '.join(format(value,'.17g') for value in (
                    int(resolved_neubc_sand),1,
                    resolved_suspended_diffusivity,
                    resolved_suspended_diffusion_dry_threshold,
                    FLOW2DH_HORIZONTAL_SCHMIDT,len(ids))),
                FLOW2DH_OPEN_BOUNDARY_SEMANTIC_ID,
                FLOW2DH_HORIZONTAL_DIFFUSION_SEMANTIC_ID,
                suspension_profile_schema_id,
                suspension_profile_instance_fingerprint]
        prefix.extend(
            f'{ids[name]} {boundaries[name].suspended_return_time_s:.17g}'
            for name in ids)
        lines[0:0]=prefix
        time_line_index+=len(prefix)
        serialized_version=48
    resolved_suspended_amr_contract=None
    if exchange==4 and not frozen_pre_m2c_suspension:
        # Version 49 adds only ownership/integration semantics.  The nested
        # v48/v47 physical FLOW profile remains byte-for-byte identifiable.
        # Every instance field is serialized as well as fingerprinted so the
        # native runtime can reject a mismatched hierarchy or dry threshold.
        resolved_suspended_amr_contract=resolve_suspended_amr_contract(
            suspension_profile_instance_fingerprint,
            fraction_count=1,
            hydraulic_dry_tolerance_m=dry_tolerance_m,
            diffusion_dry_threshold_m=
                resolved_suspended_diffusion_dry_threshold,
            amr_enabled=amr_hierarchy is not None,
            amr_levels=(amr_settings.levels
                        if amr_hierarchy is not None else 1),
            amr_refinement_ratio=(amr_settings.refinement_ratio
                                  if amr_hierarchy is not None else 2),
            amr_wet_depth_m=(amr_settings.wet_depth_m
                             if amr_hierarchy is not None else 0.),
            amr_regrid_interval_s=(amr_settings.regrid_interval_s
                                   if amr_hierarchy is not None else 0.),
        )
        prefix=[
            '49',
            SUSPENDED_AMR_SEMANTIC_ID,
            resolved_suspended_amr_contract.fingerprint,
            suspension_profile_instance_fingerprint,
            ' '.join(format(value,'.17g') for value in (
                1,dry_tolerance_m,
                resolved_suspended_diffusion_dry_threshold,
                int(amr_hierarchy is not None),
                (amr_settings.levels if amr_hierarchy is not None else 1),
                (amr_settings.refinement_ratio
                 if amr_hierarchy is not None else 2),
                (amr_settings.wet_depth_m
                 if amr_hierarchy is not None else 0.),
                (amr_settings.regrid_interval_s
                 if amr_hierarchy is not None else 0.))),
        ]
        lines[0:0]=prefix
        time_line_index+=len(prefix)
        serialized_version=49
    if capacity_identity_declared:
        # V50 is deliberately the outermost envelope.  The execution
        # fingerprint precedes the profile fingerprint everywhere (deck,
        # restart signature, sidecar and result receipt); changing that order
        # is an identity change rather than a presentation change.
        prefix=[
            '50',
            capacity_method_id,
            '1',
            format(capacity_profile_acal,'.17g'),
            capacity_execution_semantic_id_value,
            capacity_runtime_profile_semantic_id,
            capacity_execution_fingerprint,
            capacity_profile_fingerprint,
        ]
        lines[0:0]=prefix
        time_line_index+=len(prefix)
        serialized_version=50
    if resolved_channel_composition_profile is not None:
        # V51 is the outermost law-11 envelope.  The order is part of the
        # restart identity: numeric instance fields, static scope, semantic
        # identity, instance fingerprint, then the ordered class vector.
        # The nested v46/v49 rows are not rewritten or reinterpreted.
        prefix=[
            '51',
            ' '.join((
                str(CHANNEL_COMPOSITION_SCHEMA),
                str(len(resolved_channel_composition_profile.fractions)),
                format(resolved_channel_composition_profile.bed_factor,'.17g'),
                format(resolved_channel_composition_profile.suspended_factor,'.17g'),
                format(resolved_channel_composition_profile.epsilon,'.17g'),
            )),
            CHANNEL_COMPOSITION_SCOPE_ID,
            channel_composition_semantic_id(),
            resolved_channel_composition_profile.fingerprint,
            ' '.join(str(value.transport_class)
                     for value in resolved_channel_composition_profile.fractions),
        ]
        lines[0:0]=prefix
        time_line_index+=len(prefix)
        serialized_version=51
        if channel_composition_wire_version in (52,53):
            # V52 wraps the byte-stable v51 envelope.  It carries the exact
            # canonical ASCII JSON preimage plus independently serialized
            # identifiers/classes/factors so native code can reconstruct and
            # SHA-256 verify the profile before the nested v51 controls become
            # live.  Hex rows keep the line grammar ASCII-only and bounded.
            native_wire=resolved_channel_composition_native_wire
            assert native_wire is not None
            preimage_hex=native_wire['preimage_ascii'].encode('ascii').hex()
            chunks=[
                preimage_hex[index:index+CHANNEL_COMPOSITION_NATIVE_HEX_CHUNK]
                for index in range(0,len(preimage_hex),
                                   CHANNEL_COMPOSITION_NATIVE_HEX_CHUNK)
            ]
            identifier_rows=[
                f'{len(token)} {token.encode("ascii").hex()}'
                for token in native_wire['ordered_fraction_identifier_tokens']
            ]
            numeric=native_wire['numeric_tokens']
            prefix=[
                '52',
                ' '.join((
                    str(CHANNEL_COMPOSITION_NATIVE_WIRE_SCHEMA),
                    str(len(resolved_channel_composition_profile.fractions)),
                    str(len(native_wire['preimage_ascii'])),
                    str(len(chunks)),
                )),
                CHANNEL_COMPOSITION_SCOPE_ID,
                channel_composition_semantic_id(),
                native_wire['preimage_sha256'],
                ' '.join((numeric['bed_factor'],numeric['suspended_factor'],
                          numeric['epsilon'])),
                *identifier_rows,
                ' '.join(str(value) for value in
                         native_wire['ordered_transport_classes']),
                *chunks,
            ]
            execution_preimage_hex=resolved_execution_profile_preimage.encode(
                'ascii').hex()
            execution_chunks=[
                execution_preimage_hex[index:index+CHANNEL_COMPOSITION_NATIVE_HEX_CHUNK]
                for index in range(0,len(execution_preimage_hex),
                                   CHANNEL_COMPOSITION_NATIVE_HEX_CHUNK)
            ]
            prefix.extend([
                str(EXECUTION_PROFILE_SCHEMA),
                execution_profile_semantic_id(),
                resolved_execution_profile.fingerprint,
                str(len(resolved_execution_profile_preimage)),
                *execution_chunks,
            ])
            lines[0:0]=prefix
            time_line_index+=len(prefix)
            serialized_version=52
            if channel_composition_wire_version==53:
                # V53 makes every component binding native-verifiable.  V52
                # authenticates the complete execution-profile preimage, but
                # that preimage contains only hashes of component documents.
                # Carry the exact canonical component documents in a generic,
                # sorted inventory so the native runtime can recompute their
                # document hashes and cross-check their physical contents
                # against the nested operational deck.  V51/V52 remain byte
                # stable and readable; public schema-v2 preparation uses V53.
                component_documents={
                    'capacity':expected_capacity_document,
                    'channel_composition':resolved_channel_composition_document,
                }
                if exchange==4:
                    component_documents['suspension']=resolved_suspension_profile_document
                component_ids=sorted(component_documents)
                if component_ids!=sorted(
                        value.component_id
                        for value in resolved_execution_profile.component_profiles):
                    raise ValueError(
                        'Native v53 component-document inventory differs from '
                        'the execution-profile bindings')
                component_rows=[]
                resolved_component_document_receipts={}
                for component_id in component_ids:
                    document_ascii=json.dumps(
                        component_documents[component_id],sort_keys=True,
                        separators=(',',':'),ensure_ascii=True,allow_nan=False)
                    if not document_ascii.isascii() or len(document_ascii)>262144:
                        raise ValueError(
                            f'Native v53 {component_id} component document exceeds '
                            'the canonical ASCII bound')
                    binding=next(
                        value for value in resolved_execution_profile.component_profiles
                        if value.component_id==component_id)
                    if hashlib.sha256(document_ascii.encode('ascii')).hexdigest()!=\
                            binding.document_sha256:
                        raise ValueError(
                            f'Native v53 {component_id} component document hash differs')
                    resolved_component_document_receipts[component_id]=dict(
                        canonical_preimage_ascii=document_ascii,
                        document_sha256=binding.document_sha256,
                        fingerprint=binding.fingerprint,
                        schema=binding.schema,
                        scope_id=binding.scope_id,
                        semantic_id=binding.semantic_id,
                    )
                    preimage_hex=document_ascii.encode('ascii').hex()
                    chunks=[
                        preimage_hex[index:index+CHANNEL_COMPOSITION_NATIVE_HEX_CHUNK]
                        for index in range(
                            0,len(preimage_hex),CHANNEL_COMPOSITION_NATIVE_HEX_CHUNK)
                    ]
                    component_rows.append(' '.join((
                        str(len(component_id)),component_id.encode('ascii').hex(),
                        str(len(document_ascii)),str(len(chunks)),
                    )))
                    component_rows.extend(chunks)
                prefix=['53',str(len(component_ids)),*component_rows]
                lines[0:0]=prefix
                time_line_index+=len(prefix)
                serialized_version=53
    if capacity_foundation_profile:
        # V54 is Q3-C's native forcing-authority envelope.  V53 authenticates
        # the physical component graph, while this outer layer carries the
        # complete forcing and ordered execution documents whose fingerprints
        # depend on the exact raw FP64 C/n/ks bytes.  Keep those documents as
        # canonical ASCII hex so no list-directed conversion can change a
        # byte before the native SHA-256 verification.
        if serialized_version!=53:
            raise ValueError(
                'Q3-C v54 must wrap one complete native v53 component envelope')
        forcing_hex=resolved_capacity_forcing_preimage.encode('ascii').hex()
        execution_hex=(resolved_capacity_foundation_execution_preimage
                       .encode('ascii').hex())
        forcing_chunks=[
            forcing_hex[index:index+CHANNEL_COMPOSITION_NATIVE_HEX_CHUNK]
            for index in range(0,len(forcing_hex),
                               CHANNEL_COMPOSITION_NATIVE_HEX_CHUNK)
        ]
        foundation_execution_chunks=[
            execution_hex[index:index+CHANNEL_COMPOSITION_NATIVE_HEX_CHUNK]
            for index in range(0,len(execution_hex),
                               CHANNEL_COMPOSITION_NATIVE_HEX_CHUNK)
        ]
        prefix=[
            '54',
            ' '.join((
                '1',str(len(resolved_capacity_forcing_preimage)),
                str(len(forcing_chunks)),
                str(len(resolved_capacity_foundation_execution_preimage)),
                str(len(foundation_execution_chunks)),
            )),
            *forcing_chunks,*foundation_execution_chunks,
        ]
        lines[0:0]=prefix
        time_line_index+=len(prefix)
        serialized_version=54
    if hiding_foundation_profile:
        # V55 is the additive Q3-H1 identity.  Its canonical document binds
        # the selected generic-hiding mode and controls to the two ordered
        # active-surface fractions and to the exact nested Q3-C fingerprints.
        # The nested v54 document remains byte-for-byte meaningful as the
        # capacity foundation with hiding disabled; v55 owns the composition.
        if serialized_version!=54:
            raise ValueError(
                'Q3-H1 v55 must wrap one complete native v54 Q3-C envelope')
        hiding_hex=resolved_hiding_foundation_execution_preimage.encode(
            'ascii').hex()
        hiding_chunks=[
            hiding_hex[index:index+CHANNEL_COMPOSITION_NATIVE_HEX_CHUNK]
            for index in range(0,len(hiding_hex),
                               CHANNEL_COMPOSITION_NATIVE_HEX_CHUNK)
        ]
        prefix=[
            '55',
            ' '.join((
                '1',str(len(resolved_hiding_foundation_execution_preimage)),
                str(len(hiding_chunks)),
            )),
            *hiding_chunks,
        ]
        lines[0:0]=prefix
        time_line_index+=len(prefix)
        serialized_version=55
    if secondary_foundation_profile:
        # V56 owns classic FLOW equilibrium-secondary composition.  The
        # nested v54/v55 document remains an opaque secondary.off foundation;
        # this outer preimage is the sole authority for the new live route.
        expected_parent=55 if hiding_foundation_profile else 54
        if serialized_version!=expected_parent:
            raise ValueError(
                'Q3-H2 v56 must wrap one complete native v54 or v55 foundation')
        secondary_hex=resolved_secondary_foundation_execution_preimage.encode(
            'ascii').hex()
        secondary_chunks=[
            secondary_hex[index:index+CHANNEL_COMPOSITION_NATIVE_HEX_CHUNK]
            for index in range(0,len(secondary_hex),
                               CHANNEL_COMPOSITION_NATIVE_HEX_CHUNK)
        ]
        prefix=[
            '56',
            ' '.join((
                '1',str(len(resolved_secondary_foundation_execution_preimage)),
                str(len(secondary_chunks)),
            )),
            *secondary_chunks,
        ]
        lines[0:0]=prefix
        time_line_index+=len(prefix)
        serialized_version=56
    if prognostic_secondary_profile:
        # V57 is a sibling of v56.  It wraps the same intact v54/v55
        # secondary.off foundation directly and replaces (rather than stacks)
        # equilibrium rotation with one transported-state rotation.
        expected_parent=55 if hiding_foundation_profile else 54
        if serialized_version!=expected_parent:
            raise ValueError(
                'Q3-H3 v57 must wrap one complete native v54 or v55 foundation')
        prognostic_hex=(
            resolved_prognostic_secondary_execution_preimage.encode(
                'ascii').hex())
        prognostic_chunks=[
            prognostic_hex[index:index+CHANNEL_COMPOSITION_NATIVE_HEX_CHUNK]
            for index in range(0,len(prognostic_hex),
                               CHANNEL_COMPOSITION_NATIVE_HEX_CHUNK)
        ]
        prefix=[
            '57',
            ' '.join((
                '1',str(len(resolved_prognostic_secondary_execution_preimage)),
                str(len(prognostic_chunks)),
            )),
            *prognostic_chunks,
        ]
        lines[0:0]=prefix
        time_line_index+=len(prefix)
        serialized_version=57
    identity=hashlib.sha256()
    for index,row in enumerate(lines):
        if index:identity.update(b'\n')
        if index==time_line_index:row=f'END {dt_max_s:.17g} {output_interval_s:.17g}'
        identity.update(row.encode('utf-8'))
    case_temporary=path.with_suffix(path.suffix+'.partial')
    case_temporary.parent.mkdir(parents=True,exist_ok=True)
    try:
        with case_temporary.open('w+b') as stream:
            # The fingerprint is always the second physical line.  Its fixed
            # 64-byte placeholder lets the exact native deck be written once,
            # while the normalized identity is accumulated from the same rows.
            stream.write(lines[0].encode('utf-8'));stream.write(b'\n')
            fingerprint_offset=stream.tell();stream.write(b'0'*64)
            for row in lines[1:]:
                stream.write(b'\n');stream.write(row.encode('utf-8'))

            def write_batched(rows):
                first=True;batch=[]
                def flush():
                    nonlocal first
                    if not batch:return
                    payload=b'\n'.join(batch)
                    if not first:payload=b'\n'+payload
                    stream.write(payload);identity.update(payload)
                    batch.clear();first=False
                for row in rows:
                    batch.append(row.encode('utf-8'))
                    if len(batch)>=16384:flush()
                flush()

            stream.write(b'\n');identity.update(b'\n')
            if native_binary_path is None:
                write_batched(iter_mesh_lines())
            for row in suffix:
                encoded=row.encode('utf-8')
                stream.write(b'\n'+encoded);identity.update(b'\n'+encoded)
            if hierarchy_rows is not None:
                stream.write(b'\n');identity.update(b'\n')
                write_batched(hierarchy_prefix if native_binary_path is not None else hierarchy_rows())
            stream.write(b'\n')
            identity.update(json.dumps(source_sha256,sort_keys=True).encode())
            if sediment_catalog_id is not None:
                identity.update(b'\0sediment_method_catalog\0')
                identity.update(sediment_catalog_id.encode('ascii'))
            fingerprint=identity.hexdigest()
            stream.seek(fingerprint_offset);stream.write(fingerprint.encode('ascii'))
        _replace_with_windows_retry(case_temporary, path)
    except BaseException:
        case_temporary.unlink(missing_ok=True)
        raise
    metadata=dict(format=f'xbraids_cutcell-v{serialized_version}',state_layout_format=f'xbraids_cutcell-v{version}',
                  amr=None if amr_hierarchy is None else asdict(amr_settings),
                  sidecars=dict(geometry=('geometry.npz' if compact_sidecars else 'geometry.json'),
                    amr_hierarchy=(None if amr_hierarchy is None else
                                   'amr_hierarchy.npz' if compact_sidecars else 'amr_hierarchy.json'),
                    native_binary=(None if native_binary_path is None else native_binary_path.name),
                    bank_front_history=('bank_fronts.txt' if bank_front_tracking else None),
                    suspended_exchange_ledger=(
                        'suspended_exchange_ledger.txt' if exchange==4 else None),
                    suspended_boundary_ledger=(
                        'suspended_boundary_ledger.txt'
                        if flow2dh_boundary_diffusion_v48 else None),
                    suspended_boundary_memory=(
                        'suspended_boundary_memory.txt'
                        if flow2dh_boundary_diffusion_v48 else None)),
                  native_binary_sha256=native_binary_sha256,
                  fingerprint=fingerprint,string_to_numeric_id=ids,
                  capacity_route_identity=(None if not capacity_identity_declared else dict(
                    envelope_version=(54 if foundation_execution_identity else 50),
                    identity_schema=1,
                    ordered_parts=['capacity_execution_fingerprint',
                                   'capacity_profile_fingerprint'],
                    method_id=capacity_method_id,
                    capacity_profile_acal=capacity_profile_acal,
                    capacity_execution_semantic_id=
                        capacity_execution_semantic_id_value,
                    capacity_profile_semantic_id=
                        capacity_runtime_profile_semantic_id,
                    capacity_execution_fingerprint=
                        capacity_execution_fingerprint,
                    capacity_profile_fingerprint=
                        capacity_profile_fingerprint,
                    corrected_mpm_acal=(
                        capacity_profile_acal
                        if capacity_method_id==MPM_DELFT_METHOD_ID else None),
                    corrected_mpm_critical_shields=(
                        .047 if capacity_method_id==MPM_DELFT_METHOD_ID else None))),
                  hiding_foundation_identity=(
                    None if not hiding_foundation_profile else dict(
                      envelope_version=55,
                      canonical_preimage_ascii=
                        resolved_hiding_foundation_execution_preimage,
                      canonical_preimage_sha256=hashlib.sha256(
                        resolved_hiding_foundation_execution_preimage.encode(
                            'ascii')).hexdigest(),
                      **{
                        HIDING_FOUNDATION_EXECUTION_KEY:
                          resolved_hiding_foundation_execution_document,
                        HIDING_FOUNDATION_ORDERED_ROUTE_KEY:dict(
                          ordered_parts=[
                            'capacity_execution_fingerprint',
                            'capacity_profile_fingerprint',
                            'hiding_execution_fingerprint',
                          ],
                          capacity_execution_fingerprint=
                            resolved_hiding_foundation_execution_document[
                              'capacity_execution_fingerprint'],
                          capacity_profile_fingerprint=
                            resolved_hiding_foundation_execution_document[
                              'capacity_profile_fingerprint'],
                          hiding_execution_fingerprint=
                            resolved_hiding_foundation_execution_document[
                              'fingerprint'],
                        ),
                      })),
                  secondary_foundation_identity=(
                    None if not secondary_foundation_profile else dict(
                      envelope_version=56,
                      canonical_preimage_ascii=
                        resolved_secondary_foundation_execution_preimage,
                      canonical_preimage_sha256=hashlib.sha256(
                        resolved_secondary_foundation_execution_preimage.encode(
                            'ascii')).hexdigest(),
                      **{
                        SECONDARY_FOUNDATION_EXECUTION_KEY:
                          resolved_secondary_foundation_execution_document,
                        SECONDARY_FOUNDATION_ORDERED_ROUTE_KEY:dict(
                          ordered_parts=[
                            'capacity_execution_fingerprint',
                            'capacity_profile_fingerprint',
                            'hiding_execution_fingerprint',
                            'secondary_execution_fingerprint',
                          ],
                          capacity_execution_fingerprint=
                            resolved_secondary_foundation_execution_document[
                              'capacity_execution_fingerprint'],
                          capacity_profile_fingerprint=
                            resolved_secondary_foundation_execution_document[
                              'capacity_profile_fingerprint'],
                          hiding_execution_fingerprint=
                            resolved_secondary_foundation_execution_document[
                              'hiding_execution_fingerprint'],
                          secondary_execution_fingerprint=
                            resolved_secondary_foundation_execution_document[
                              'fingerprint'],
                        ),
                      })),
                  prognostic_secondary_identity=(
                    None if not prognostic_secondary_profile else dict(
                      envelope_version=57,
                      canonical_preimage_ascii=
                        resolved_prognostic_secondary_execution_preimage,
                      canonical_preimage_sha256=hashlib.sha256(
                        resolved_prognostic_secondary_execution_preimage.encode(
                            'ascii')).hexdigest(),
                      prognostic_secondary_execution_fingerprint=
                        resolved_prognostic_secondary_execution_document[
                          'fingerprint'],
                      prognostic_secondary_bed_boundary_rotation_policy=
                        resolved_prognostic_secondary_execution_document[
                          'bed_boundary_rotation_policy'],
                      prognostic_secondary_state_units='m2_s',
                      prognostic_secondary_integrated_ledger_units=
                        resolved_prognostic_secondary_execution_document[
                          'integrated_ledger_units'],
                      **{
                        PROGNOSTIC_SECONDARY_EXECUTION_KEY:
                          resolved_prognostic_secondary_execution_document,
                        PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY:dict(
                          capacity_execution_fingerprint=
                            resolved_prognostic_secondary_execution_document[
                              'capacity_execution_fingerprint'],
                          capacity_profile_fingerprint=
                            resolved_prognostic_secondary_execution_document[
                              'capacity_profile_fingerprint'],
                          hiding_execution_fingerprint=
                            resolved_prognostic_secondary_execution_document[
                              'hiding_execution_fingerprint'],
                          prognostic_secondary_execution_fingerprint=
                            resolved_prognostic_secondary_execution_document[
                              'fingerprint'],
                        ),
                      })),
                  channel_composition_identity=(
                    None if resolved_channel_composition_document is None else dict(
                      envelope_version=channel_composition_wire_version,
                      wire_schema=(
                        CHANNEL_COMPOSITION_SCHEMA
                        if channel_composition_wire_version==51 else
                        CHANNEL_COMPOSITION_NATIVE_WIRE_SCHEMA),
                      canonical_wire_order=(
                        ['schema_fraction_count_Bed_Sus_epsilon',
                         'scope_id','semantic_id','profile_fingerprint',
                         'ordered_transport_classes']
                        if channel_composition_wire_version==51 else
                         ['wire_schema_fraction_count_preimage_length_chunk_count',
                         'scope_id','semantic_id','profile_fingerprint',
                         'canonical_numeric_tokens',
                         'ordered_fraction_identifier_json_tokens',
                         'ordered_transport_classes',
                          'canonical_preimage_ascii_hex_chunks',
                          ('component_document_inventory_then_nested_v52'
                           if channel_composition_wire_version==53 else
                           'nested_v51')]),
                      class_resolution=(
                        'exchange_0_to_TRA_BEDLOAD_1' if exchange==0 else
                        'exchange_4_to_TRA_COMBINE_3'),
                      profile_fingerprint_authority=(
                        'managed_host_attested_opaque_v51'
                        if channel_composition_wire_version==51 else
                        ('native_sha256_verified_component_preimages_v53'
                         if channel_composition_wire_version==53 else
                         'native_sha256_verified_canonical_preimage_v52')),
                      native_profile_fingerprint_verification=(
                        channel_composition_wire_version in (52,53)),
                      native_profile_preimage=(
                        'ordered_fraction_ids_not_serialized_in_v51'
                        if channel_composition_wire_version==51 else
                        resolved_channel_composition_native_wire['preimage_ascii']),
                      native_profile_preimage_sha256=(
                        None if channel_composition_wire_version==51 else
                        resolved_channel_composition_native_wire['preimage_sha256']),
                      ordered_fraction_identifiers=(
                        None if channel_composition_wire_version==51 else
                        [value.fraction_id for value in
                         resolved_channel_composition_profile.fractions]),
                      native_verification_followup=(
                        'Q3-S_schema_v2_complete_preimage_and_in_process_verification'
                        if channel_composition_wire_version==51 else None),
                      factor_domain=(
                        'first live law-11 slice inherits nonnegative Bed and '
                        'positive Sus from established native controls; the '
                        'backend-neutral composer itself retains signed factors'),
                      global_factor_owner='channel_composer_exactly_once',
                      profile=resolved_channel_composition_document,
                      execution_profile=(
                        None if channel_composition_wire_version==51 else
                        dict(
                          schema=EXECUTION_PROFILE_SCHEMA,
                          semantic_id=execution_profile_semantic_id(),
                          fingerprint=resolved_execution_profile.fingerprint,
                          canonical_preimage_ascii=
                            resolved_execution_profile_preimage,
                          profile=resolved_execution_profile_document)))),
                  execution_profile_v2=(
                    None if resolved_execution_profile is None else dict(
                      envelope_version=channel_composition_wire_version,
                      runtime_backend=execution_runtime_backend,
                      profile_backend=(
                        ('backend_neutral_physical_profile'
                         if channel_composition_wire_version==53 else
                         'neutral_pending_q3s_route_admission')
                        if resolved_execution_profile.backend is None else
                        resolved_execution_profile.backend),
                      schema=EXECUTION_PROFILE_SCHEMA,
                      semantic_id=execution_profile_semantic_id(),
                      fingerprint=resolved_execution_profile.fingerprint,
                      canonical_preimage_sha256=hashlib.sha256(
                        resolved_execution_profile_preimage.encode('ascii')
                      ).hexdigest(),
                      canonical_preimage_ascii=resolved_execution_profile_preimage,
                      profile=resolved_execution_profile_document,
                      **({} if resolved_component_document_receipts is None else
                         {'component_documents':resolved_component_document_receipts}))),
                  suspended_amr_ownership=(
                    None if resolved_suspended_amr_contract is None else
                    suspended_amr_contract_document(
                        resolved_suspended_amr_contract)),
                  frozen_pre_m2c_suspension=frozen_pre_m2c_suspension,
                  boundary_terrain=dict(sampled_inlet_faces=sum(f.bed_elevation_m is not None for f in mesh.faces),
                    representation=('fixed initial absolute face elevation at law-9 equilibrium inlet; owner-cell bed remains mobile'
                                    if bedload==9 and any(boundary.bedload_boundary_mode=='equilibrium_capacity'
                                                          for boundary in boundaries.values()) else
                                    'piecewise-linear DEM endpoints plus owner bed increment' if production_section else
                                    'fixed initial DEM face offset plus owner bed increment' if version>=11 else 'owner-cell bed'),
                    wetting='all submerged portions of each assigned inlet' if production_section or any(
                        spec.mode in ('inlet_stage','inlet_depth','inlet_normal','inlet_critical')
                        for spec in boundaries.values()) else
                            'connected to selected section minimum' if version>=11 else 'all positive-depth faces',
                    live_controls=production_section,
                    section_control_update=('normal/critical level recomputed from current boundary bed at every RK stage'
                                            if production_section else None),
                    section_endpoint_policy=('adjacent reflective polygon edges act as vertical sidewalls if the live normal-depth stage rises above an evolving terrain endpoint'
                                             if production_section else None),
                    roughness=(f'per-owner-cell {hydraulic_roughness_mode} parameter, grouped into contiguous conveyance subsections'
                               if production_section else None)),
                  source_sha256=source_sha256,boundaries={k:asdict(v) for k,v in boundaries.items()},
                  resolved_boundary_targets=resolved_targets,
                  resolved_hydraulic_series=resolved_hydraulic_series,
                  resolved_bed_input=resolved_bed_input,
                  resolved_fraction_input=resolved_fraction_input,
                  total_discharge_profiles={name:boundaries[name].total_discharge_profile for name in ids},
                   bedload_boundary_modes={name:boundaries[name].bedload_boundary_mode for name in ids},
                   bedload_flux_semantics={name:(
                     'equilibrium_capacity_inflow' if boundaries[name].bedload_boundary_mode=='equilibrium_capacity' else
                     'prescribed_fixed_face_bed_inflow' if boundaries[name].bedload_boundary_mode=='prescribed_fixed_face_bed' else
                     'prescribed_inflow' if boundaries[name].is_inlet else
                    'free_capacity_export_on_hydraulic_outflow; incoming reversal rejected')
                    for name in ids},
                   bedload_boundary_mode_scope=(
                     'code 2 composes prescribed incoming class feed with the law-9 fixed-face hydraulic trace; every hydraulic outlet exports local capacity on outflow'
                     if any(boundaries[name].bedload_boundary_mode=='prescribed_fixed_face_bed'
                            for name in ids) else
                     'controls incoming bed-load prescription only; every hydraulic outlet exports local capacity on outflow'),
                   gaia_fixed_inlet=(dict(
                     enabled=True,
                     boundary_ids=[name for name in ids
                                   if boundaries[name].bedload_boundary_mode in (
                                       'equilibrium_capacity','prescribed_fixed_face_bed')],
                     trigger=(
                       'law9 + prescribed_fixed_face_bed + mode6 total_discharge'
                       if any(boundaries[name].bedload_boundary_mode=='prescribed_fixed_face_bed'
                              for name in ids) else
                       'law9 + equilibrium_capacity + mode6 total_discharge'),
                    face_bed='immutable absolute initial face elevation',
                    owner_cells_mobile=True,
                    section_stage='live length-weighted adjacent free-surface elevation',
                    face_depth='max(live section stage - fixed initial face bed, 0)',
                    hydrostatic_correction='dynamic fixed face bed minus current owner-cell bed',
                    outlet='mobile owner bed with free-capacity export')
                    if bedload==9 and any(boundary.bedload_boundary_mode in (
                                          'equilibrium_capacity','prescribed_fixed_face_bed')
                                           for boundary in boundaries.values()) else None),
                  native_bed_boundary_units=('prescribed total kg/s per logical ID; equilibrium capacity is native kg/(m s) per face'
                                             if equilibrium_bedload else
                                             ('total kg/s per logical ID' if version>=3 else 'kg/(m s) per incoming face')),
                  sediment_time_series=dict(interpolation='piecewise linear',outside_range='hold endpoint',
                    timestep_policy='stop at every forcing knot',concentration_units='kg/m3',
                    bed_input_units='total kg/s per logical ID',
                    per_fraction=bool(has_fraction_sediment_series)),
                  hydraulic_time_series=dict(interpolation='piecewise linear',outside_range='hold endpoint',
                    timestep_policy='stop at every forcing knot',discharge_units='m3/s per logical ID',
                    native_control_units=('water depth m for prescribed_depth_trace; otherwise absolute water level m'
                                          if prescribed_trace else 'absolute water level m')),
                  fraction_ids=[] if fractions is None else fraction_ids,
                  initial_columns=(['h_m','hu_m2_s','hv_m2_s','S_kg_m2','Mbed_kg_m2'] if fractions is None else
                    ['h_m','hu_m2_s','hv_m2_s']+[f'S_{name}_kg_m2' for name in fraction_ids]+
                    ([f'Mactive_{name}_kg_m2' for name in fraction_ids]+
                     ([f'Msubstrate_{name}_kg_m2' for name in fraction_ids]
                      if substrate_layer_count==1 else
                      [f'Msubstrate_layer_{layer}_{name}_kg_m2'
                       for layer in range(1,substrate_layer_count+1)
                       for name in fraction_ids]) if layered else
                     [f'Mbed_{name}_kg_m2' for name in fraction_ids])),
                  ledger_columns=(['water_volume_m3','suspended_mass_kg','bedload_mass_kg'] if fractions is None else
                    ['water_volume_m3']+[f'suspended_{name}_kg' for name in fraction_ids]+
                    [f'bedload_{name}_kg' for name in fraction_ids]),
                  manning=(dict(default_manning_n=manning,
                    per_cell=version>=5,
                    minimum_manning_n=float(local_manning.min()) if local_manning is not None else manning,
                    maximum_manning_n=float(local_manning.max()) if local_manning is not None else manning,
                    cell_column='manning_n' if version>=5 else None)
                    if hydraulic_roughness_mode=='manning' else dict(
                    storage_only=True,physical_type=False,
                    storage_field='manning_cell' if version>=5 else 'legacy_scalar',
                    note='native compatibility storage contains the selected raw C/ks parameter; not Manning n')),
                  hydraulic_roughness=dict(
                    mode=hydraulic_roughness_mode,
                    parameter_name=({'chezy':'chezy_C','manning':'manning_n',
                                      'white_colebrook':'nikuradse_ks'}[hydraulic_roughness_mode]),
                    parameter_units=({'chezy':'m^0.5/s','manning':'s/m^(1/3)',
                                      'white_colebrook':'m'}[hydraulic_roughness_mode]),
                    parameter_label=({'chezy':'Chezy C','manning':'Manning n',
                                      'white_colebrook':'Nikuradse ks'}[hydraulic_roughness_mode]),
                    default_parameter=manning,
                    per_cell=version>=5,
                    minimum_parameter=float(local_manning.min()) if local_manning is not None else manning,
                    maximum_parameter=float(local_manning.max()) if local_manning is not None else manning,
                    white_colebrook_chzk2d=white_colebrook_chzk2d,
                    minimum_work_depth_m=roughness_work_depth,
                    storage_field='manning_cell' if version>=5 else 'legacy_scalar',
                    storage_field_is_physical_manning=(hydraulic_roughness_mode=='manning'),
                    deck_envelope_version=45 if typed_roughness else None,
                    legacy_manning_deck=not typed_roughness),
                  sediment_shear=dict(
                    mode=sediment_shear_mode,
                    affects='sediment RP_TAUB consumers only',
                    hydraulic_friction_unchanged=True,
                    kinematic_viscosity_m2_s=(skin_kinematic_viscosity_m2_s
                                              if sediment_shear_mode!='total' else None),
                    sand_roughness_m=(skin_sand_roughness_m
                                      if sediment_shear_mode!='total' else None),
                    silt_roughness_m=(skin_silt_roughness_m
                                      if sediment_shear_mode!='total' else None),
                    critical_mud_fraction_low=(skin_critical_mud_fraction_low
                                               if sediment_shear_mode!='total' else None),
                    critical_mud_fraction_high=(skin_critical_mud_fraction_high
                                                if sediment_shear_mode!='total' else None),
                    fixed_fluff_fraction=(skin_fluff_fraction
                                          if sediment_shear_mode!='total' else None)),
                  engelund_hansen_chezy=(engelund_hansen_chezy if bedload==8 else None),
                  gaia_engelund_chollet_cunge=(dict(
                    source_contract='TELEMAC-GAIA v8p5r1 formula 3; model correspondence, not solver identity',
                    skin_roughness_ratio=gaia_skin_roughness_ratio,
                    slope_magnitude_beta=gaia_slope_beta,
                    talmon_deviation_beta2=gaia_deviation_beta2,
                    secondary_alpha=gaia_secondary_alpha,
                    hydraulic_friction='BRAIDS Manning Cf=2g/(h^(1/6)/n)^2; depth-dependent mapping from TELEMAC law-5 Nikuradse ks',
                    skin_friction='KSP=skin_roughness_ratio*d; ICR=1',
                    **(dict(diagnostic_v44=dict(
                        validation_only=True,
                        interior_slope_talmon_enabled=gaia_diagnostic_values[0],
                        interior_secondary_stress_enabled=gaia_diagnostic_values[1],
                        open_boundary_secondary_stress_enabled=gaia_diagnostic_values[2],
                    )) if gaia_diagnostic_v44 else {})) if bedload==9 else None),
                  materials=({**asdict(sediment),"bulk_kg_m3":sediment.bulk_kg_m3} if fractions is None else
                    {'water_density_kg_m3':fractions[0].water_density_kg_m3,
                     'fractions':[{**asdict(value),'bulk_kg_m3':value.bulk_kg_m3} for value in fractions]}),
                  bed_layers=dict(enabled=layered,active_layer_thickness_m=active_layer_thickness_m,
                    active_layer_mode=active_layer_mode,
                    active_layer_depth_fraction=active_layer_depth_fraction if dynamic_active else None,
                    active_layer_minimum_thickness_m=active_layer_minimum_thickness_m if dynamic_active else None,
                    substrate_layer_count=substrate_layer_count if layered else 0,
                    substrate_layer_thickness_m=substrate_layer_thickness_m if layered else None,
                    substrate_order='surface downward; final layer is the variable-thickness base' if layered else None,
                    active_composition_basis='solid bulk volume',
                    transport_availability='upwind active volume fraction times active-thickness factor' if layered else 'total fraction inventory'),
                  bank_morphology=dict(enabled=bank_enabled,
                    erodible_cell_count=int(bank_mask.sum()) if bank_enabled else 0,
                    erosion_supply_fraction=bank_erosion_fraction if bank_enabled else 0.,
                    avalanche_enabled=bank_avalanche if bank_enabled else False,
                    wet_maximum_slope=bank_wet_slope if bank_enabled else None,
                    dry_maximum_slope=bank_dry_slope if bank_enabled else None,
                    avalanche_time_s=bank_avalanche_time_s if bank_enabled else None,
                    relaxation_length_m=bank_relaxation_length_m if bank_enabled else None,
                    relaxation_diffusivity_m2_s=(bank_relaxation_length_m**2/(2*bank_avalanche_time_s)
                                                 if bank_enabled and bank_avalanche else None),
                    wet_depth_switch_m=bank_wet_depth_switch_m if bank_enabled else None,
                    cell_rule='whole retained cell on positive-area polygon intersection',
                    transfer_basis='conservative per-fraction active-layer mass across internal faces',
                    domain_boundary='fixed; faces without a neighbouring retained cell never participate',
                    newly_eroded_cells='hydraulic wetting remains governed by the native wet/dry flux'),
                   bank_failure=dict(enabled=bank_failure,
                    qualified=(not noncohesive_bank_failure),
                    status=(
                        'experimental distributed slope-bank supply; global bend qualification pending'
                        if distributed_bank_failure else
                        'experimental physical-toe supply from accepted erosive Exner divergence; production bend qualification pending'
                        if physical_toe_bank_failure else
                        'qualified for excess-shear toe erosion, threshold block failure, conservation, restart, AMR, and thread determinism'
                        if bank_failure else 'disabled'),
                    qualification_evidence=(
                        'validation/sediment/jamieson2013_bend/reports/crossed_bank_forcing_oracle_v1/RESULTS.md'
                        if distributed_bank_failure else
                        'validation/sediment/bank_toe_geometry/SUPPLY_RESULTS_V1.md'
                        if physical_toe_bank_failure else
                        'validation/sediment/delft3d_current/bank_failure/reports/metrics.json'
                        if bank_failure else None),
                    toe_erosion_mode=(bank_toe_erosion_mode if bank_failure else None),
                    toe_law=(
                        'alpha times the adjacent downhill-cell bedload component tangent to each supercritical bank-slope face'
                        if distributed_bank_failure else
                        'capacity_fraction times accepted positive net Exner divergence at each unique physical toe'
                        if physical_toe_bank_failure else
                        'excess shear: retreat_rate = kd * max(tau - tau_c, 0)^a'
                        if bank_failure else None),
                    toe_capacity_fraction=(bank_toe_capacity_fraction
                                           if physical_toe_bank_failure else None),
                    distributed_transport_fraction=(bank_distributed_transport_fraction
                                                    if distributed_bank_failure else None),
                    distributed_critical_slope=(bank_distributed_critical_slope
                                                if distributed_bank_failure else None),
                    toe_erodibility_m_pa_s=(bank_toe_erodibility_m_pa_s
                                            if bank_failure and not noncohesive_bank_failure else None),
                    toe_critical_shear_pa=(bank_toe_critical_shear_pa
                                           if bank_failure and not noncohesive_bank_failure else None),
                    toe_exponent=(bank_toe_exponent
                                  if bank_failure and not noncohesive_bank_failure else None),
                    critical_undercut_m=bank_critical_undercut_m if bank_failure else None,
                    erodible_cell_count=int(bank_mask.sum()) if bank_failure else 0,
                    collapse_rule=('resolved undercut threshold with the remaining overhang transferred as a discrete block'
                                   if bank_failure else None),
                    transfer_basis=(
                        'conservative active-layer donor composition across each eligible downhill internal face'
                        if distributed_bank_failure else
                        'conservative per-fraction layered mass from mapped donor to floor-side physical toe; one demand per unique toe'
                        if physical_toe_bank_failure else
                        'conservative per-fraction layered mass; toe material and failed block deposit in adjacent wet cells'
                        if bank_failure else None),
                    bank_material_mask=(
                        'persistent user material polygon; wet and dry donors are both eligible and exterior/nonmaterial cells are excluded'
                        if distributed_bank_failure else
                        'persistent user material polygon including submerged side-slope cells through the physical toe'
                        if physical_toe_bank_failure else None),
                    repose_coupling=('qualified domain-wide avalanche follows each accepted step'
                                     if bank_failure and general_avalanche else
                                     'disabled; enable general_avalanche for post-collapse repose redistribution'
                                     if bank_failure else None),
                     domain_boundary=(
                        'fixed; only eligible internal faces of explicitly marked bank material participate'
                        if distributed_bank_failure else
                         'fixed; only mapped internal bank-material interfaces participate'
                         if physical_toe_bank_failure else
                         'fixed; only internal wet/dry faces participate' if bank_failure else None),
                     bank_front_tracking_enabled=bank_front_tracking),
                   bank_front_tracking=dict(
                     enabled=bank_front_tracking,
                     qualified=False,
                     status=(
                         'experimental persistent subcell bank-front contract; native evolution, restart, AMR, and global morphology qualification pending'
                         if bank_front_tracking else 'disabled'),
                     default=False,
                     deck_envelope_version=41 if bank_front_tracking else None,
                     required_bank_mode=(
                         'distributed_slope_transport' if bank_front_tracking else None),
                     state_semantics=(
                         'persistent erodible-bank plan-area occupancy/retreat state, distinct from Eulerian bed elevation'
                         if bank_front_tracking else None),
                     forcing_semantics=(
                         'advance only from conservative accepted distributed-slope bank transfer after the accepted Exner step'
                         if bank_front_tracking else None),
                     geometry_semantics=(
                         'immutable mesh topology with a prognostic subcell bank-material front'
                         if bank_front_tracking else None),
                     persistence_contract=(
                         'checkpoint/restart and any AMR transfer must preserve front state conservatively and deterministically'
                         if bank_front_tracking else None),
                     output=(
                         'bank_fronts.txt: per-output-cell plan-area occupancy, retreat fraction, effective/raw bed, and removed volume'
                         if bank_front_tracking else None),
                     qualification_evidence=None),
                   bedload_adaptation=dict(enabled=bedload_adaptation,qualified=True,
                    status=('qualified against the analytic relaxation solution and Delft3D equilibrium limit'
                            if bedload_adaptation else 'disabled'),
                    qualification_evidence=('validation/sediment/delft3d_current/adaptation_length/reports/metrics.json'
                                            if bedload_adaptation else None),
                    model=('steady first-order streamline advection-relaxation'
                           if bedload_adaptation else None),
                    length_diameters=(bedload_adaptation_length_diameters
                                      if bedload_adaptation else None),
                    per_fraction_length_m=([bedload_adaptation_length_diameters*value.d50_m
                                            for value in fractions]
                                           if bedload_adaptation and fractions is not None else
                                           [bedload_adaptation_length_diameters*sediment.d50_m]
                                           if bedload_adaptation else None),
                     equilibrium_limit='exact disabled-path identity'),
                   morphological_acceleration=dict(enabled=morphological_acceleration,
                     qualified=True,
                     status=('qualified through MorFac 20 only for quasi-steady subcritical bed-load/Exner flow; MorFac 1 required for wet/dry hydraulic transients'
                             if morphological_acceleration else 'disabled'),
                     qualification_evidence='validation/sediment/delft3d_current/morphological_acceleration/reports/metrics.json',
                     published_regime_limits={'quasi_steady_subcritical_bedform':20,
                                              'wet_dry_hydraulic_transient':1},
                     constant_factor=morphological_factor,
                     series=(None if morphological_factor_series is None else
                             asdict(morphological_factor_series)),
                     interpolation=('block (left-held between knots; endpoint-held outside range)'
                                    if morphological_factor_series is not None else 'constant'),
                     hydraulic_clock='seconds',morphological_clock='integral of MorFac over hydraulic seconds',
                     timestep_policy=('stop at every factor knot' if morphological_factor_series is not None else None),
                     native_diagnostic=('morphological_acceleration.txt' if morphological_acceleration else None),
                     source_semantics='Delft3D dtmor = dts * MorFac; capacity is not multiplied',
                     supported_processes='conservative bed-load/Exner only',
                     unsupported_processes=['suspended exchange','bank erosion','bank failure']),
                   bedform_roughness=dict(enabled=bedform_roughness,qualified=True,
                     status=('qualified instantaneous equilibrium Fredsoe-MPM/Van Rijn 1984 feedback'
                             if bedform_roughness else 'disabled'),
                     qualification_evidence='validation/sediment/delft3d_current/bedform_roughness_assessment/reports/metrics.json',
                     model=('Fredsoe MPM equilibrium dune height; Van Rijn 1984 dune length and roughness'
                            if bedform_roughness else None),
                     grain_roughness='3 D90 combined quadratically with dune roughness',
                     equilibrium='deterministic local fixed point; no hidden prognostic state',
                     native_diagnostic=('bedform_roughness.txt' if bedform_roughness else None),
                     default='disabled',
                     unsupported_processes=['suspended exchange','normal-depth inlet control',
                       'secondary flow','adaptation length','morphological acceleration','bank processes']),
                  general_avalanche=dict(enabled=general_avalanche,qualified=True,
                    status=('qualified by paired wet/dry current-source D-Flow FM validation'
                            if general_avalanche else 'disabled'),
                    qualification_evidence=('validation/sediment/delft3d_current/general_avalanche/reports/metrics.json'
                                            if general_avalanche else None),
                    wet_maximum_slope=general_avalanche_wet_slope if general_avalanche else None,
                    dry_maximum_slope=general_avalanche_dry_slope if general_avalanche else None,
                    avalanche_time_s=general_avalanche_time_s if general_avalanche else None,
                    relaxation_length_m=general_avalanche_relaxation_length_m if general_avalanche else None,
                    relaxation_diffusivity_m2_s=(general_avalanche_relaxation_length_m**2/
                        (2*general_avalanche_time_s) if general_avalanche else None),
                    wet_depth_switch_m=general_avalanche_wet_depth_switch_m if general_avalanche else None,
                    donor_rule='every retained bed cell; availability limited by exposed active inventory',
                    transfer_basis='conservative per-fraction active-layer mass across internal faces',
                    domain_boundary='fixed; faces without a neighbouring retained cell never participate'),
                   secondary_flow=dict(
                     enabled=bool(secondary_flow or prognostic_secondary_profile),
                     qualified=bool(secondary_flow and not prognostic_secondary_flow and not gaia_secondary_flow),
                     experimental=bool(prognostic_secondary_flow or gaia_secondary_flow or
                                       prognostic_secondary_profile),
                     status=(
                       'authenticated classic FLOW prognostic-secondary v57; qualification pending'
                       if prognostic_secondary_profile else
                       'source-equation correspondence for TELEMAC-GAIA v8p5r1; matched-case qualification pending'
                       if gaia_secondary_flow else
                       'experimental prognostic configuration; numerical qualification pending'
                       if prognostic_secondary_flow else
                       'qualified by paired mild-bend and symmetric-bifurcation current-source D-Flow FM validation'
                       if secondary_flow else 'disabled'),
                     qualification_evidence=(
                       'validation/sediment/delft3d_current/secondary_flow/reports/metrics.json'
                       if secondary_flow and not prognostic_secondary_flow and not gaia_secondary_flow else None),
                     model=('prognostic' if prognostic_secondary_profile else
                            secondary_flow_model if secondary_flow else None),
                     method_id=(resolved_prognostic_secondary_execution_document[
                         'method_id'] if prognostic_secondary_profile else None),
                     transport_direction=(
                         resolved_prognostic_secondary_execution_document[
                             'transport_direction'] if prognostic_secondary_profile else
                         secondary_flow_transport_direction if secondary_flow else None),
                     momentum_beta=(secondary_flow_momentum_beta if secondary_flow else None),
                     classic_betac=(
                         resolved_prognostic_secondary_execution_document['betac']
                         if prognostic_secondary_profile else None),
                     scalar_diffusivity_m2_s=(
                         resolved_prognostic_secondary_execution_document[
                             'scalar_diffusivity_m2_s']
                         if prognostic_secondary_profile else None),
                     inflow_mode=(
                         resolved_prognostic_secondary_execution_document[
                             'inflow_mode'] if prognostic_secondary_profile else
                         1 if prognostic_secondary_flow else None),
                     espir=(resolved_prognostic_secondary_execution_document['espir']
                            if prognostic_secondary_profile else
                            secondary_flow_espir if secondary_flow else None),
                     compatible_bedload_laws=(
                         [2,8,11] if prognostic_secondary_profile else
                         [9] if gaia_secondary_flow else [2,3,5,6,7,8]),
                      transport_effect=(
                        'transported signed spiral intensity rotates raw bed load once'
                        if prognostic_secondary_profile else
                        'GAIA stress addition changes both direction and magnitude'
                        if gaia_secondary_flow else
                       'direction only; scalar bed-load magnitude is preserved exactly'
                       if secondary_flow and secondary_flow_transport_direction else
                       'disabled'),
                     momentum_effect=(
                       'classic Betac stress divergence disabled in first v57 scope'
                       if prognostic_secondary_profile else
                       f'experimental BetaSpiral feedback with Beta={secondary_flow_momentum_beta:.17g}'
                       if prognostic_secondary_flow and secondary_flow_momentum_beta>0 else
                        'disabled; no BetaSpiral force is applied'),
                     diagnostic_output=(dict(
                       schema='BRAIDS_GAIA_SECONDARY_STRESS_V1',
                       file='gaia_secondary_stress.txt',
                       clock='accepted output frame',
                       eta_gradient='grad(z+h)',
                       bed_gradient='grad(z), positive uphill',
                       stress_decomposition='Talmon/SLOPEFF stress + secondary delta = total stress',
                       angle='signed counter-clockwise radians in [-pi,pi]',
                       capacity='cell-centred equilibrium capacity; not accepted face flux',
                       legacy_file='secondary_flow.txt',
                       legacy_not_applicable_columns=['curvature_1_m','spiral_intensity_m_s','alpha'])
                       if gaia_secondary_flow else None),
                     state_layout=('runtime prognostic intensity is not part of the public physical initial q layout'
                                    if (prognostic_secondary_flow or
                                        prognostic_secondary_profile) else None),
                     prognostic_state_units=(
                       'm2_s' if prognostic_secondary_profile else None),
                     prognostic_integrated_ledger_units=(
                       resolved_prognostic_secondary_execution_document[
                         'integrated_ledger_units']
                       if prognostic_secondary_profile else None),
                     bed_boundary_rotation_policy=(
                       resolved_prognostic_secondary_execution_document[
                         'bed_boundary_rotation_policy']
                       if prognostic_secondary_profile else None),
                     coriolis_effect='disabled in Cartesian planar hydraulics'),
                  solid_obstacles=dict(enabled=bool(mesh.obstacle_ids),ids=list(mesh.obstacle_ids),
                    excluded_area_m2=mesh.solid_area_m2,
                    wall_fragment_count=mesh.metrics()['internal_wall_fragment_count'],
                    wall_length_m=mesh.metrics()['internal_wall_length_m'],
                    geometry='exact polygon cut faces on the fixed Cartesian background grid',
                    hydraulic_behavior='impermeable reflective shallow-water Riemann wall',
                    sediment_behavior='zero suspended and bed-load flux'),
                  hiding_exposure=dict(mode=hiding_exposure,
                    name={0:'none',2:'Egiazaroff',3:'Ashida-Michiue',
                          4:'Parker-Klingeman-McLean',5:'Wu-Wang-Jia'}[hiding_exposure],
                    mean_diameter=('active-layer volume-weighted arithmetic D50'
                                   if hiding_exposure in (2,3,4) else None),
                    parker_alpha=hiding_parker_alpha if hiding_exposure==4 else None,
                    wu_exponent=hiding_wu_exponent if hiding_exposure==5 else None,
                    application=(
                        ('fraction critical Shields threshold in MPM'
                         if bedload==2 else
                         'fraction critical Shields threshold in Delft3D general capacity')
                        if hiding_exposure else None)),
                  reconstruction_scheme=scheme,
                  cfl=cfl,dry_tolerance_m=dry_tolerance_m,
                  horizontal_viscosity_m2_s=horizontal_viscosity_m2_s,
                  horizontal_viscosity_formulation=horizontal_viscosity_formulation,
                  horizontal_viscosity_discretization=(
                    'boundary_quadratic_lsq_face_gradient_v35'
                    if horizontal_viscosity_formulation=='velocity_laplacian' else
                    'two_point_harmonic_depth_v33' if horizontal_viscosity_m2_s>0 else None),
                  bedload_law=bedload,mpm_cf=mpm_cf,mpm_critical_shields=mpm_critical_shields,
                  mpm_friction_mode=dict(mode=mpm_friction_mode,
                    name='Manning and fraction D90 roughness correction' if mpm_friction_mode else 'specified friction coefficient',
                    source='FLOW tranb2 Chezy/ripple correction' if mpm_friction_mode else 'Zhu et al. benchmark coefficient'),
                  bedload_law_name={0:'off',1:'Van Rijn 1993',2:'Meyer-Peter-Muller',
                    3:'Ashida-Michiue current-only',4:'Soulsby current-only',5:'Wilcock-Crowe',
                    6:'modified Wilcock-Crowe',7:'Gaeuman et al.',
                    8:'Engelund-Hansen current-only total load',
                    9:'TELEMAC-GAIA Engelund-Hansen / Chollet-Cunge bed load',
                    11:'Delft3D general current capacity (TraFrm 4)',
                    12:'modified Van Rijn 1984 current capacity (TraFrm 7)',
                    13:'Soulsby/Van Rijn current capacity (TraFrm 11)',
                    14:'Wang/Fredsoe current capacity (TraFrm 13)'}[bedload],
                  capacity_formula=(None if resolved_capacity_parameter_rows is None else dict(
                    envelope_version=47 if typed_constitutive_v47 else 46,
                    parameter_schema_id=capacity_parameter_schema_id,
                    profile_semantic_id=capacity_profile_schema_id,
                    diagnostic_schema_id=(capacity_diagnostic_schema_id
                                          if typed_constitutive_v47 else None),
                    diagnostic_contract=(capacity_diagnostic_contract_document()
                                         if typed_constitutive_v47 else None),
                    method_id=formula_contract(bedload).method_id,
                    output_shape=formula_contract(bedload).output_shape,
                    output_contract='retain_bed_and_suspended',
                    bed_channel='connected to conservative bed-load/Exner route',
                    suspended_channel=(
                        ('connected to conservative depth-integrated suspension '
                         'through generic FLOW 2DH Galappatti exchange mode 4'
                         if exchange==4 else
                         'retained as capacity_split.txt cell and '
                         'capacity_split_boundary.txt boundary diagnostics; '
                         'conservative routing is not selected')),
                    direction_semantics=(
                        'secondary flow steers only scalar bed capacity; '
                        'suspended capacity remains signed and follows the '
                        'local depth-averaged flow or boundary trace'),
                    boundary_roles={
                        0:'reversal or no capacity transfer',
                        1:'equilibrium inlet evaluated on hydraulic trace',
                        2:'stage outlet evaluated on hydraulic trace',
                        3:'outlet using owner capacity'},
                    diagnostics=['absolute_shields','critical_shields','stress_ratio','transport_stage'],
                    diagnostic_note=(
                        'v47 rows carry an applicability mask; numeric zero is '
                        'retained for not-applicable fields and never aliases '
                        'the common stress_ratio field' if typed_constitutive_v47 else
                        'transport_stage is formula-specific and never aliases '
                        'the common stress_ratio field'),
                    public_bed_only_supported=False,
                    fraction_ids=list(('single',) if fractions is None else fraction_ids),
                    per_fraction_parameters=[dict(row) for row in resolved_capacity_parameter_rows])),
                  constitutive_contract=(dict(
                    envelope_version=47,
                    kinematic_viscosity_m2_s=constitutive_kinematic_viscosity_m2_s,
                    capacity_fraction_basis=resolved_capacity_fraction_basis,
                    ifractions_code=capacity_fraction_basis_code,
                    diagnostic_schema_id=capacity_diagnostic_schema_id,
                    suspension_profile_semantic_id=suspension_profile_schema_id,
                    suspension_profile_fingerprint=
                        suspension_profile_instance_fingerprint,
                    flow2dh_exchange_method_id=FLOW2DH_EXCHANGE_ID,
                    flow2dh_exchange_semantic_id=FLOW2DH_EXCHANGE_SEMANTIC_ID,
                    flow2dh_settling_method_id=FLOW2DH_SETTLING_ID,
                    flow2dh_settling_semantic_id=FLOW2DH_SETTLING_SEMANTIC_ID,
                    factsd=suspended_adaptation_factor,
                    settlefrm_code=settling_form_code,
                    settling_formula=settling_formula,
                    settling_fac_dss=settling_fac_dss,
                    settling_temperature_mode=settling_temperature_mode,
                    settling_temperature_mode_code=settling_temperature_mode_code,
                    settling_temperature_c=resolved_settling_temperature_c,
                    settling_reference_concentration_kg_m3=
                        hindered_settling_reference_concentration_kg_m3,
                    suspended_calibration_factor=suspended_calibration,
                    settling_clear_water_velocity_m_s=[settling_clear_water_velocity_m_s],
                    sediment_d50_m=[sediment.d50_m],
                    sediment_density_kg_m3=[sediment.grain_density_kg_m3],
                    water_density_kg_m3=sediment.water_density_kg_m3,
                    gravity_m_s2=9.81,
                    sediment_activity_contract=
                        'h1_times_boolean_wet_mask_gt_sedthr',
                    wet_mask_provider=
                        'braids_h1_gt_dry_tolerance_maps_to_flow_kfs_v1',
                    sedthr_m=[sediment.minimum_exchange_depth_m] if fractions is None else
                             [value.minimum_exchange_depth_m for value in fractions])
                    if typed_constitutive_v47 else None),
                  suspended_exchange_name={0:'off',1:'settling only',2:'Van Rijn 1993 equilibrium',
                    3:'cohesive or mixed sand/mud',
                    4:'generic FLOW 2DH capacity/Galappatti'}[exchange],
                  suspended_exchange_route=(dict(
                    method_id=FLOW2DH_EXCHANGE_ID,
                    semantic_id=FLOW2DH_EXCHANGE_SEMANTIC_ID,
                    capacity_law=bedload,
                    source_order='explicit equilibrium source plus implicit sink, mass form',
                    finite_inventory_limiter='BRAIDS nonnegative available-mass cap; not FLOW red_soursin equivalence',
                    boundary_ledger_sign='positive into domain',
                    exchange_ledger_sign='positive bed to suspended',
                    factsd=suspended_adaptation_factor,
                    suspended_calibration_factor=suspended_calibration,
                    depth_contract='h0=accepted pre-step depth; h1=post-advection endpoint depth',
                    activity_contract='wet_mask(h1) and h1 > SedThr',
                    sedthr_m=[sediment.minimum_exchange_depth_m] if fractions is None else
                             [value.minimum_exchange_depth_m for value in fractions])
                    if exchange==4 else None),
                  suspension_profile=resolved_suspension_profile_document,
                  flow2dh_open_boundary=(dict(
                    method_id=FLOW2DH_OPEN_BOUNDARY_ID,
                    semantic_id=FLOW2DH_OPEN_BOUNDARY_SEMANTIC_ID,
                    neubc_sand=resolved_neubc_sand,
                    neubc_mud=False,
                    prescribed_series_provider=
                        'authenticated_case_logical_boundary_fraction_series_uniform_face_expansion_v1',
                    prescribed_series_restart_identity=
                        'resolved_static_and_timed_values_lengths_times_and_boundary_order',
                    return_time_s={name:boundaries[name].suspended_return_time_s
                                   for name in ids},
                    hydraulic_reversal_policy={
                        name:boundaries[name].reversal_policy for name in ids},
                    restart_memory=
                        'per_face_per_fraction_last_outflow_concentration_and_return_clock',
                    restart_memory_value=
                        'adjacent_pretransport_macrostep_r0_concentration_on_accepted_outflow',
                    clock_time_mapping=
                        'once_per_accepted_ssprk33_macro_step_using_exact_sign_of_integrated_water_transport',
                    accepted_advective_ledger_sign='positive_into_domain')
                    if flow2dh_boundary_diffusion_v48 else None),
                  flow2dh_horizontal_diffusion=(dict(
                    method_id=FLOW2DH_HORIZONTAL_DIFFUSION_ID,
                    semantic_id=FLOW2DH_HORIZONTAL_DIFFUSION_SEMANTIC_ID,
                    dicuv_m2_s=resolved_suspended_diffusivity,
                    cell_to_face='arithmetic_mean',
                    schmidt=FLOW2DH_HORIZONTAL_SCHMIDT,
                    dryflc_m=resolved_suspended_diffusion_dry_threshold,
                    face_exchange_area=
                        'BRAIDS accepted reconstructed hydraulic face-trace arithmetic depth times face width; porosity resolved in face geometry',
                    centre_distance=
                        'absolute_face_normal_owner_to_neighbour_centres_or_owner_centre_to_boundary_face_once',
                    face_rate_sign='positive_owner_to_neighbour_or_exterior',
                    activity='strict_min_endpoint_h1_gt_DryFlc',
                    numerical_mapping=
                        'explicit_post_FCT_conservative_donor_limited_v1',
                    temporal_treatment=
                        'each_SSPRK33_Euler_stage_after_accepted_water_FCT_advection',
                    positivity=
                        'deterministic_cell_outgoing_inventory_ratio_then_shared_face_donor_scale',
                    ledger_sign='positive_into_domain')
                    if flow2dh_boundary_diffusion_v48 else None),
                  settling_treatment=dict(mode=settling_mode,
                    name=('concentration-dependent hindered settling' if settling_mode else 'constant settling velocity'),
                    total_concentration='sum of all suspended dry-sediment fractions per water volume',
                    reference_concentration_kg_m3=(hindered_settling_reference_concentration_kg_m3
                                                   if settling_mode else None),
                    exponent=HINDERED_SETTLING_EXPONENT if settling_mode else None,
                    limiter='max(0, 1 - C_total / C_reference)^5' if settling_mode else None,
                    **(dict(
                        settling_formula=settling_formula,
                        settlefrm_code=settling_form_code,
                        settling_fac_dss=settling_fac_dss,
                        temperature_mode=settling_temperature_mode,
                        temperature_c=resolved_settling_temperature_c,
                        clear_water_velocity_m_s=[settling_clear_water_velocity_m_s])
                       if typed_constitutive_v47 else {})),
                  critical_mud_fraction=critical_mud_fraction,
                  transport_calibration_factor=transport_calibration_factor,
                  bed_slope=(dict(islope=bed_slope,alfabs=bed_slope_alfabs,
                                  ashld=bed_slope_alfabn,bshld=bed_friction_angle_deg,
                                  cshld=0.,dshld=0.,
                                  source='Van Bendegom and Koch-Flokstra; CShld=DShld=0')
                             if bed_slope==3 else
                             dict(islope=bed_slope,slope_effect_formula=1,
                                  slope_magnitude_beta=gaia_slope_beta,
                                  deviation_formula=2,
                                  talmon_deviation_beta2=gaia_deviation_beta2,
                                  open_boundary_order=(
                                      'reset bed-slope magnitude/direction to flow, then add GAIA secondary stress'),
                                  source='TELEMAC-GAIA v8p5r1 BEDLOAD_EFFPNT SLOPEFF=1 and DEVIA=2 Talmon')
                             if bed_slope==4 else
                             dict(islope=5,delft3d_islope=2,alfabs=bed_slope_alfabs,
                                  alfabn=bed_slope_alfabn,
                                  friction_angle_deg=bed_friction_angle_deg,
                                  taurat_contract='zero_for_positive_EH_and_MPM_formulas',
                                  source='Locked Delft3D FLOW ISlope=2, distinct from legacy BRAIDS mode 2')
                             if bed_slope==5 else
                             dict(islope=bed_slope,alfabs=bed_slope_alfabs,
                                  alfabn=bed_slope_alfabn,
                                  friction_angle_deg=bed_friction_angle_deg,
                                  source='Bagnold longitudinal and Ikeda/Van Rijn transverse slope correction')),
                  inlet_policy=('Total Q once per ID; prescribed hydraulic trace (stage or depth) and section '
                                'conveyance allocation; diagnostic strong control; local reversal rejected'
                                if prescribed_trace else
                                ('Total Q once per ID; free water level; zero-normal-gradient adjacent stage; '
                                + ('viscosity-and-{}-equilibrated section allocation on selected IDs; '.format(
                                    {'manning':'Manning','chezy':'Chezy',
                                     'white_colebrook':'White-Colebrook'}[hydraulic_roughness_mode])
                                   if equilibrated_inlets else 'stock section-conveyance allocation; ')
                                + 'local reversal rejected'
                                if source_total_discharge else
                                ('Total Q once per ID; resolved horizontal section stage and uniform normal velocity '
                                 'are imposed on all submerged portions; dry faces receive no prescribed inflow'
                                 if any(spec.mode in ('inlet_stage','inlet_depth','inlet_normal','inlet_critical')
                                        for spec in boundaries.values()) else
                                 'Total Q once per ID; section-derived target; common stage from outgoing characteristics; '
                                 'target on dry/supercritical incoming faces; local reversal rejected'))),
                  outlet_policy='Explicit selected faces; stage, section-critical, free-weir rating, or supercritical release; stage/critical dry or reversing traces close locally; unsupported incoming supercritical flow is rejected')
    if not typed_constitutive_v47:
        # V47 is a true outer envelope.  Do not change the established JSON
        # shape of legacy/v46 receipts merely to carry absent typed fields.
        metadata.pop('constitutive_contract',None)
        metadata.pop('suspended_exchange_route',None)
        metadata.pop('suspension_profile',None)
    if not flow2dh_boundary_diffusion_v48:
        metadata.pop('flow2dh_open_boundary',None)
        metadata.pop('flow2dh_horizontal_diffusion',None)
    if not capacity_identity_declared:
        # Keep every pre-v50 JSON receipt unchanged as well as its deck bytes.
        metadata.pop('capacity_route_identity',None)
    if resolved_channel_composition_document is None:
        # Likewise, an omitted v51 profile changes neither a legacy deck nor
        # its JSON receipt.  No ordinary route is upgraded implicitly.
        metadata.pop('channel_composition_identity',None)
    elif channel_composition_wire_version==51:
        # Preserve the established host-attested v51 receipt byte meaning;
        # every new native/preimage field belongs only to the additive v52.
        for key in ('native_profile_preimage_sha256',
                    'ordered_fraction_identifiers','execution_profile'):
            metadata['channel_composition_identity'].pop(key,None)
    if resolved_execution_profile is None:
        metadata.pop('execution_profile_v2',None)
    if manning_friction_chezy_minimum_depth_m>0:
        metadata['manning']['friction_chezy_minimum_depth_m']=manning_friction_chezy_minimum_depth_m
        metadata['manning']['friction_formulation']='physical_h_times_floored_chezy_work_depth_one_third'
    if sediment_catalog_id is not None:
        metadata['semantic_catalog_id']=sediment_catalog_id
    path.with_suffix(path.suffix+'.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if amr_hierarchy is not None:
        amr_hierarchy.write_geometry(path.parent/metadata['sidecars']['amr_hierarchy'])
    return metadata
