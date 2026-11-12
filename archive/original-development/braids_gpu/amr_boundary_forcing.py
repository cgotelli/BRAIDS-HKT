"""Resident recursive boundary clocks; no persistent host forcing cursor.

Each local RK step samples its own t, t+dt and t+dt/2. The outer controller
clips at the union of hydraulic/class-feed knots. Tables are uploaded once;
numerical interpolation runs on CUDA. Surveyed roots are an explicit capability:
the authenticated ordinary live-bank components are reused, never substituted.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
import numpy as np

from validation.gpu_fwave.boundary_events import BoundaryEvents, tolerance
from validation.gpu_fwave.boundary_schedule import ResidentBoundarySchedule


def require(value, message):
    if not value:raise ValueError(message)


SURVEYED_RULE='level_rk33_current_section_integrated_normal_v1'


def validate_surveyed_scope(data):
    """The ordinary live-section scope intersected with the recursive port.

    Adaptation uses the existing whole-leaf steady closure with an explicit
    recursive transaction. The section control itself is preserved (0, 1 or 2).
    """
    from .general_live_bank import _validate_prepared
    # The fixed-grid adapter also admits equilibrium/OFF steering and further
    # hiding laws. Those new combinations do not expand recursive AMR merely
    # because it shares the boundary reader and CUDA components.
    require(int(data['bank_toe_erosion_mode'])==3 and
            int(data['secondary_flow_model'])==2 and
            int(data['hiding_exposure']) in (0,4),
            'Recursive surveyed sections require mode3, prognostic secondary and hiding 0/4')
    return _validate_prepared(data)


@dataclass(frozen=True)
class SurveyedComponents:
    BoundaryControls: type
    ResidentSedimentTransport: type
    ResidentBoundarySchedule: type
    source_folder: Path
    source_receipt: dict
    inlet_control: int


@contextmanager
def surveyed_components(folder,data):
    """Load only the authenticated spatial adapters, without a solver clock.

    The returned classes keep their module globals after this context closes.
    No shared CoupledSolver globals are patched. All CUDA arguments supplied by
    the caller must use ``source_folder``; mixing a different sediment kernel
    with this section bridge is not this capability.
    """
    from . import general_live_bank as bank
    from .g1_live_inlet_boundary import make_boundary_controls
    from .integrated_normal_trace import integrated_boundary
    from validation.gpu_fwave.normal_outlet_boundary import make_boundary_controls as normal_controls
    validate_surveyed_scope(data)
    folder=Path(folder).resolve()
    receipt=bank.verify_sources(folder)
    require(receipt['bank_mode']==3 and receipt['nfrac']==int(data['nfrac']) and
            receipt['nbed_layers']==int(data['nbed_layers']),
            'Recursive surveyed source material schema differs')
    names=('boundary_schedule.py','boundaries.py','sediment_transport.py',
           'outlet_boundary.py','live_normal_boundary.py')
    keys={name:'validation.gpu_fwave._recursive_live_'+name[:-3] for name in names}
    require(not any(key in sys.modules for key in keys.values()),
            'Recursive surveyed component load is already active')
    modules={};loaded=[]
    try:
        for name in names:
            path=folder/(bank.PRIVATE[name]+'.py')
            spec=importlib.util.spec_from_file_location(keys[name],path)
            require(spec is not None and spec.loader is not None,
                    'Recursive surveyed component is missing: '+name)
            module=importlib.util.module_from_spec(spec)
            sys.modules[keys[name]]=module;loaded.append(keys[name])
            spec.loader.exec_module(module);modules[name]=module
        control=int(data['section_control'][0])
        parent=modules['boundaries.py'].BoundaryControls
        outlet=modules['outlet_boundary.py']
        boundary=(integrated_boundary(normal_controls(parent,
            modules['live_normal_boundary.py'],outlet)) if control==1 else
            make_boundary_controls(parent,outlet))
        identity=dict(rule=SURVEYED_RULE,
            archive_sha256=hashlib.sha256((folder/'general_live_bank_sources.json').read_bytes()).hexdigest(),
            factory_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            normal_trace_rule=receipt['normal_trace_rule'],
            normal_root_rule=receipt['normal_root_rule'],
            generated={name:dict(receipt['generated'][name]) for name in names})
        yield SurveyedComponents(boundary,
            modules['sediment_transport.py'].ResidentSedimentTransport,
            modules['boundary_schedule.py'].ResidentBoundarySchedule,
            folder,identity,control)
    finally:
        for key in reversed(loaded):sys.modules.pop(key,None)


def _initialize_live_workspace(boundary):
    """Initialize private scratch, excluding every static/input array alias.

    Some existing section components allocate ``empty`` scratch. A complete
    recursive rollback must snapshot deterministic bytes before the first
    trial. This initializes scratch only, not any section geometry or state.
    """
    from collections.abc import Mapping
    cp=boundary.cp;seen=set();protected=set();mutable={}
    def arrays(value):
        if isinstance(value,cp.ndarray):yield value
        elif isinstance(value,Mapping):
            for child in value.values():yield from arrays(child)
        elif isinstance(value,(list,tuple)):
            for child in value:yield from arrays(child)
    def visit(obj):
        if id(obj) in seen:return
        seen.add(id(obj))
        if not type(obj).__module__.startswith(('braids_gpu','validation.gpu_fwave')):return
        for key,value in vars(obj).items():
            if key in ('_arrays','arrays','inputs'):
                protected.update(int(a.data.ptr) for a in arrays(value))
            if key in ('outputs','metric_outputs','work','cache','root_outputs','trace_outputs',
                       'inlet_records','normal_records','outlet_records','stage_records'):
                for array in arrays(value):mutable[int(array.data.ptr)]=array
            if not callable(value) and not isinstance(value,cp.ndarray):visit(value)
    visit(boundary)
    for pointer,array in mutable.items():
        if pointer not in protected:array.fill(0)
    if hasattr(boundary,'normal_root'):boundary.normal_root.invalidate_cache()


class RecursiveBoundaryForcing:
    SLOTS=(0,2,1)
    FRACTIONS=(0.,1.,.5)

    def __init__(self,solver,levels,*,source_path=None,live_components=None):
        require(solver.external_boundary_schedule and
                solver.distributed_flux_ownership=='external_all_face',
                'Recursive forcing requires explicit external bank ownership')
        self.solver=solver;self.cp=solver.cp
        self.levels=np.asarray(levels).copy()
        data=solver.data;owner=np.asarray(data['owner']);bid=np.asarray(data['bid'])
        require(self.levels.shape==(int(data['nc']),) and self.levels.dtype.kind in 'iu' and
                np.all((self.levels>=0)&(self.levels<=30)),
                'Recursive boundary levels are invalid')
        require(np.all(self.levels[owner[bid>=0]]==self.levels.max()),
                'Recursive forced boundary faces must remain on the finest level')
        self.finest=int(self.levels.max());self.surveyed=live_components is not None
        self.components=live_components;self._q=None;self.recorded_stage=0
        if self.surveyed:
            require(isinstance(live_components,SurveyedComponents) and
                    bool(getattr(solver,'external_live_sections',False)),
                    'Surveyed recursive boundaries require explicit live capability')
            validate_surveyed_scope(data)
            require(isinstance(solver.boundary,live_components.BoundaryControls) and
                    int(data['section_control'][0])==live_components.inlet_control and
                    Path(solver.folder).resolve()==live_components.source_folder,
                    'Recursive surveyed solver/components differ')
            source=live_components.source_folder/'boundary_schedule.cu'
            require(source_path is None or Path(source_path).resolve()==source,
                    'Recursive surveyed schedule source differs')
            schedule_type=live_components.ResidentBoundarySchedule
        else:
            require(not bool(getattr(solver,'external_live_sections',False)) and
                    np.isin(data['mode'],(2,6)).all(),
                    'Surveyed sections require authenticated live components')
            source=source_path;schedule_type=ResidentBoundarySchedule
        self.events=BoundaryEvents.from_tables(data)
        self.schedule=schedule_type(data,source_path=source)
        for array in self.schedule.outputs.values():array.fill(0.)
        self.clock=None;self.stage=0

    def bind_stage_state(self,q):
        """Bind the stable PRIVATE Euler state before transaction inventory."""
        require(self.surveyed and self._q is None and self.clock is None,
                'Recursive surveyed state must be bound exactly once')
        cp=self.cp;s=self.solver
        require(isinstance(q,cp.ndarray) and q.shape==s.state['q'].shape and
                q.dtype==cp.float64 and q.flags.c_contiguous and
                q.device.id==s.state['q'].device.id,
                'Recursive surveyed state shape/storage differs')
        lo,hi=int(q.data.ptr),int(q.data.ptr)+q.nbytes
        accepted=s.state['q'];start=int(accepted.data.ptr)
        require(hi<=start or start+accepted.nbytes<=lo,
                'Recursive surveyed Euler state aliases accepted hydraulics')
        s.flux.inputs['q']=q
        s.boundary.bind_sediment_hydraulic(s.sediment.geometry['bc_hydraulic'])
        s.boundary.bind_stage_state(q)
        _initialize_live_workspace(s.boundary)
        s.boundary.check_storage();self._q=q

    def begin_rk(self,level,current,dt):
        require(self.clock is None,'Recursive boundary RK clock is already open')
        require(type(level) in (int,np.int32,np.int64) and 0<=int(level)<=self.finest,
                'Recursive boundary RK level is invalid')
        if self.surveyed:
            require(self._q is not None,'Recursive surveyed Euler state is not bound')
            self.solver.boundary.check_storage()
        self.schedule.evaluate(current,dt)
        if self.surveyed:self.solver.boundary.begin_trial()
        self.clock=(int(level),float(current),float(dt));self.stage=0;self.recorded_stage=0

    def apply(self,current,dt,level):
        require(self.clock is not None and self.stage<3,
                'Recursive boundary stage has no open RK clock')
        require(not self.surveyed or self.recorded_stage==self.stage,
                'Recursive surveyed boundary omitted the previous stage record')
        expected_level,start,span=self.clock
        expected=start+self.FRACTIONS[self.stage]*span
        require(level==expected_level and float(dt)==span and
                abs(float(current)-expected)<=tolerance(expected),
                'Recursive boundary stage time/order differs')
        slot=self.SLOTS[self.stage];values=self.schedule.outputs;cp=self.cp;s=self.solver
        require(not bool(cp.any(values['status'][slot]).item()),
                'Recursive boundary forcing is invalid at this stage')
        cp.copyto(s.boundary.arrays['forcing'],values['hydraulic'][slot])
        cp.copyto(s.sediment.geometry['bc_hydraulic'],values['hydraulic'][slot])
        cp.copyto(s.sediment.geometry['feed'],values['feed'][slot])
        self.stage+=1

    def record_stage(self):
        """Call immediately after boundary.apply on the current private q/bed."""
        if not self.surveyed:return
        require(self.clock is not None and self.stage==self.recorded_stage+1,
                'Recursive surveyed record has no unrecorded Euler stage')
        self.solver.boundary.record_stage(self.stage-1)
        self.recorded_stage=self.stage

    def integrated_flux(self,buffers,level):
        """Insert the qualified normal closure before stress and draining.

        All assigned boundaries are finest-level faces; overwriting a coarser
        level's excluded flux would violate the level-local incidence mask.
        """
        if not self.surveyed:return
        require(self.clock is not None and int(level)==self.clock[0] and
                1<=self.stage<=3 and self.recorded_stage==self.stage,
                'Recursive surveyed flux is outside its evaluated stage')
        if int(level)==self.finest and self.components.inlet_control==1:
            self.solver.boundary.integrated_flux(buffers)

    def finish_rk(self):
        require(self.clock is not None and self.stage==3 and
                (not self.surveyed or self.recorded_stage==3),
                'Recursive boundary RK omitted a stage')
        self.clear()

    def clear(self):
        self.clock=None;self.stage=0;self.recorded_stage=0
        if self.surveyed and hasattr(self.solver.boundary,'_g1_next_stage'):
            self.solver.boundary._g1_next_stage=None

    def require_macro_clock(self,current,dt):
        knot=self.events.next_after(current)
        require(current+dt<=knot+tolerance(knot),
                'Recursive macro step crosses a forcing knot; use next_timestep()')
