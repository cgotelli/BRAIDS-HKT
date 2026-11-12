"""Research foundation: recursive GPU mode-3 bank AMR on a fixed leaf graph.

Arbitrary admitted grain/layer counts and 2:1 levels; native SSPRK33 clocks,
coarse dense output, all-face fine replacement, then one outer bank/avalanche
transaction. Ordinary AMR dispatch is provided by amr_recursive_route.

The legacy Euler bridge evaluates the whole graph before masking cells and
rejects donor limiting. An explicit level-local research bridge instead limits
incident face fluxes before advancing only the current level. An explicit
recursive wet/dry research mode prepares private RK operands in place and
records hydraulic momentum cleanup. Suspension remains unsupported.
Surveyed live sections and scheduled stage/total-Q boundaries use local RK clocks;
no mode silently substitutes a synchronous/CPU route.
Whole-graph reconstruction remains in the new bridge.
"""
from __future__ import annotations

from pathlib import Path
from weakref import WeakKeyDictionary
import hashlib
import json
import copy

import numpy as np

from .a1_full_array_transaction import _unique_arrays, _require_disjoint_storage
from .atomic_output import save_arrays_atomic, write_json_atomic
from .amr_recursive_engine import RecursiveLevelScheduler
from validation.gpu_fwave.a2_segmented_array_copy import SegmentedCopyPlan, signature
from validation.gpu_fwave.coupled_solver import CoupledSolver, normalized
from validation.gpu_fwave.distributed_bank import FLOAT_HISTORY


HERE = Path(__file__).resolve().parent
_PLANS = WeakKeyDictionary()
METHOD = 'research_recursive_mode3_all_face_register_v1'
LEVEL_LOCAL_METHOD = 'research_recursive_mode3_level_local_euler_v2'
REPAIRED_METHOD = 'research_recursive_mode3_conservative_reflux_v3'
WET_METHOD = 'research_recursive_mode3_wet_preparation_v4'
SCHEDULED_METHOD = 'research_recursive_mode3_scheduled_forcing_v5'
SECONDARY_METHOD = 'research_recursive_mode3_secondary_v6'
BEDFORM_BANK_ORDER = 'level_ssprk33_stage_capacity_and_pre_friction_v1'
BEDFORM_BANK_METHOD = 'recursive_bedform_mode3_parent_avalanche_v1'
BEDFORM_PHYSICAL_TOE_METHOD = \
    'recursive_bedform_mode2_physical_toe_parent_avalanche_v1'
BEDFORM_BANK_METHOD_CONTRACT = dict(
    policy=BEDFORM_BANK_METHOD,
    bedform_order=BEDFORM_BANK_ORDER,
    stage_capacity_evaluations_per_level_step=3,
    pre_friction_evaluations_per_level_step=1,
    accepted_diagnostics='lazy_from_accepted_state',
    checkpointed_diagnostics=False,
    bank_transaction='distributed_mode3_post_reflux_once_per_macro_v1',
    avalanche_transaction='parent_post_bank_once_per_macro_v1',
    calls_per_macro=dict(bank=1,avalanche=1))
BEDFORM_PHYSICAL_TOE_METHOD_CONTRACT = dict(
    policy=BEDFORM_PHYSICAL_TOE_METHOD,
    bedform_order=BEDFORM_BANK_ORDER,
    stage_capacity_evaluations_per_level_step=3,
    pre_friction_evaluations_per_level_step=1,
    accepted_diagnostics='lazy_from_accepted_state',
    checkpointed_diagnostics=False,
    accepted_flux_owner='completed_root_exner_register_v1',
    bank_transaction='physical_toe_supply_undercut_collapse_post_reflux_once_per_macro_v1',
    avalanche_transaction='parent_post_bank_once_per_macro_v1',
    mapping_transaction='reconcile_then_evolving_toe_rebuild_before_commit_v1',
    calls_per_macro=dict(bank=1,avalanche=1))


class RecursiveStepRejected(RuntimeError):
    """Numerical trial rejection; all accepted state has been restored."""
    def __init__(self,message,*,cfl=float('nan')):
        super().__init__(message)
        self.cfl=float(cfl)


def require(value, message):
    if not value:
        raise ValueError(message)


def _digest_arrays(values):
    digest=hashlib.sha256()
    for key,value in sorted(values.items()):
        array=np.asarray(value)
        require(array.dtype.kind in 'fiub','Non-numerical recursive input '+key)
        header=json.dumps([key,array.dtype.str,array.shape],separators=(',',':')).encode()
        digest.update(len(header).to_bytes(8,'little'));digest.update(header)
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _source_files(folder,register_source,level_source=None,repair_source=None,wet_source=None,forcing=False,secondary_source=None):
    # Include the complete operator-Python directory to fail closed if a new
    # lazily imported operator is introduced. Paths are logical/relocatable.
    paths={}
    import validation.gpu_fwave.coupled_solver as coupled_module
    origin=Path(coupled_module.__file__).resolve().parent
    for path in sorted(origin.glob('*.py')):paths['operators/'+path.name]=path
    for name in ('amr_bank_subcycle.py','amr_recursive_engine.py',
                 'a1_full_array_transaction.py','atomic_output.py'):
        paths['driver/'+name]=HERE/name
    for path in sorted(Path(folder).rglob('*.cu')):
        paths['cuda/'+path.relative_to(folder).as_posix()]=path
    for path in sorted(Path(folder).rglob('*.py')):
        paths['generated/'+path.relative_to(folder).as_posix()]=path
    for path in sorted(Path(folder).rglob('general_live_bank_sources.json')):
        paths['receipts/'+path.relative_to(folder).as_posix()]=path
        receipt=json.loads(path.read_text())
        for name in receipt['sources']:
            relative=Path(name)
            require(not relative.is_absolute() and '..' not in relative.parts,
                    'Recursive live source path escapes package')
            if relative.suffix=='.py':paths['live/'+relative.as_posix()]=HERE.parent/relative
    paths['cuda/recursive_register.cu']=Path(register_source)
    if level_source is not None:
        paths['driver/amr_level_euler.py']=HERE/'amr_level_euler.py'
        paths['cuda/level_euler.cu']=Path(level_source)
    if repair_source is not None:
        paths['driver/amr_bank_repair.py']=HERE/'amr_bank_repair.py'
        paths['cuda/reflux_repair.cu']=Path(repair_source)
    if wet_source is not None:
        paths['driver/amr_bank_wet.py']=HERE/'amr_bank_wet.py'
        paths['cuda/wet_preparation.cu']=Path(wet_source)
    if forcing:paths['driver/amr_boundary_forcing.py']=HERE/'amr_boundary_forcing.py'
    if secondary_source is not None:
        paths['driver/amr_level_secondary.py']=HERE/'amr_level_secondary.py'
        paths['cuda/level_secondary.cu']=Path(secondary_source)
    return {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}


def level_topology(owner, neighbour, levels):
    """Host-only immutable schedule/CSR construction, no physical state."""
    owner, neighbour, levels = map(np.asarray, (owner, neighbour, levels))
    require(levels.ndim==1 and len(levels)>0 and levels.dtype.kind in 'iu' and
            np.all((levels>=0)&(levels<=30)) and owner.ndim==1 and owner.dtype.kind in 'iu'
            and neighbour.shape==owner.shape and neighbour.dtype.kind in 'iu' and
            np.all((owner>=0)&(owner<len(levels))) and
            np.all((neighbour>=-1)&(neighbour<len(levels))) and
            np.all(owner!=neighbour), 'Invalid recursive bank AMR graph/levels')
    internal=neighbour>=0
    require(np.all(np.abs(levels[owner[internal]]-levels[neighbour[internal]])<=1),
            'Recursive bank AMR requires a balanced 2:1 leaf graph')
    maximum=int(np.max(levels));records=[]
    for level in range(maximum+1):
        cells=np.flatnonzero(levels==level).astype(np.int32)
        touches=(levels[owner]==level)|np.where(internal,levels[np.maximum(neighbour,0)]==level,False)
        cross=internal & (((levels[owner]==level)&(levels[np.maximum(neighbour,0)]==level+1)) |
                          ((levels[owner]==level+1)&(levels[np.maximum(neighbour,0)]==level)))
        own_faces=np.flatnonzero(touches & ~cross).astype(np.int32)
        by_cell={}
        for face in np.flatnonzero(cross):
            side=0 if levels[owner[face]]==level else 1
            cell=int(owner[face] if side==0 else neighbour[face])
            by_cell.setdefault(cell,[]).append((int(face),side))
        groups=sorted(by_cell);offset=[0];faces=[];sides=[]
        for cell in groups:
            entries=sorted(by_cell[cell])
            faces.extend(f for f,_ in entries);sides.extend(s for _,s in entries)
            offset.append(len(faces))
        records.append(dict(cells=cells,own_faces=own_faces,
            coarse_cells=np.asarray(groups,np.int32),offset=np.asarray(offset,np.int32),
            faces=np.asarray(faces,np.int32),sides=np.asarray(sides,np.int32)))
    return records


class SubcycledDistributedBankStepper:
    """Private recursive bank trial with an accepted all-face Exner register.

    The historical class name is retained for checkpoint/source compatibility.
    The recursive clock and register are shared by physical-toe mode 2 and
    distributed-bank mode 3; only their one-per-macro bank transaction differs.
    """

    def __init__(self, data, levels, *, source_folder, register_source=None, wet_depth=1.e-6,
                 euler_mode='legacy_unlimited',reflux_mode='reject_negative',wet_front_mode='reject',
                 boundary_mode='constant_stage',secondary_mode='off',
                 evolving_toe_variant=False):
        require(type(euler_mode) is str and euler_mode in ('legacy_unlimited','level_local'),
                'Unknown recursive Euler mode')
        self.euler_mode=euler_mode
        require(type(reflux_mode) is str and reflux_mode in ('reject_negative','conservative') and
                (reflux_mode!='conservative' or euler_mode=='level_local'),
                'Conservative reflux requires explicit level-local Euler')
        self.reflux_mode=reflux_mode
        require(type(wet_front_mode) is str and wet_front_mode in ('reject','recursive') and
                (wet_front_mode!='recursive' or
                 (euler_mode=='level_local' and reflux_mode=='conservative')),
                'Recursive wet fronts require explicit level-local Euler and conservative reflux')
        self.wet_front_mode=wet_front_mode
        self.method=LEVEL_LOCAL_METHOD if euler_mode=='level_local' else METHOD
        if reflux_mode=='conservative':self.method=REPAIRED_METHOD
        if wet_front_mode=='recursive':self.method=WET_METHOD
        require(type(boundary_mode) is str and boundary_mode in ('constant_stage','scheduled','surveyed') and
                (boundary_mode=='constant_stage' or
                 (euler_mode=='level_local' and reflux_mode=='conservative' and wet_front_mode=='recursive')),
                'Scheduled recursive boundaries require level-local, conservative, wet-front stepping')
        self.boundary_mode=boundary_mode
        scheduled=boundary_mode!='constant_stage'
        surveyed=boundary_mode=='surveyed'
        if scheduled:self.method=SCHEDULED_METHOD
        require(type(secondary_mode) is str and secondary_mode in ('off','native') and
                (secondary_mode!='native' or
                 (euler_mode=='level_local' and reflux_mode=='conservative' and wet_front_mode=='recursive')),
                'Recursive secondary requires explicit level-local conservative wet-front stepping')
        secondary=secondary_mode=='native'
        require(not surveyed or secondary,'Surveyed recursive boundaries require native secondary flow')
        self.secondary_mode=secondary_mode
        if secondary:self.method=SECONDARY_METHOD+('_surveyed' if surveyed else '_scheduled' if scheduled else '_constant')
        data=normalized(data)
        self.bank_mode=int(data['bank_toe_erosion_mode'])
        require(self.bank_mode in (2,3) and
                not (self.bank_mode==3 and evolving_toe_variant),
                'Recursive bank AMR requires physical-toe mode 2 or distributed mode 3')
        self.evolving_toe_variant=bool(evolving_toe_variant)
        if self.bank_mode==2:
            self.method=self.method.replace('mode3','mode2_physical_toe')
        require(not (self.bank_mode==2 and surveyed),
                'Recursive physical-toe mode 2 surveyed sections are not yet admitted')
        self.bedform_bank_composition=bool(int(data['bedform_roughness']))
        self.bedform_physical_toe_composition=bool(
            self.bedform_bank_composition and self.bank_mode==2)
        self.bedform_distributed_bank_composition=bool(
            self.bedform_bank_composition and self.bank_mode==3)
        if self.bedform_bank_composition:
            require(boundary_mode=='constant_stage' and
                    not scheduled and not secondary and
                    euler_mode=='level_local' and reflux_mode=='conservative' and
                    wet_front_mode=='recursive' and
                    bool(int(data['general_avalanche'])),
                    'Recursive bedform composition requires constant-stage '
                    'mode-2/3 bank erosion, level-local conservative wet-front '
                    'stepping, parent avalanche and secondary OFF')
            self.method += '_'+(BEDFORM_PHYSICAL_TOE_METHOD
                if self.bedform_physical_toe_composition else
                BEDFORM_BANK_METHOD)
        adaptation=bool(data['bedload_adaptation'])
        require(not adaptation or (self.bank_mode==3 and surveyed and secondary and
                int(data['secondary_flow_model'])==2),
                'Recursive adaptation requires surveyed mode3 prognostic secondary scope')
        self.adaptation_policy='full_leaf_steady_v1' if adaptation else None
        if adaptation:self.method+='_adaptation_full_leaf_v1'
        require(int(data['bank_toe_erosion_mode']) in (2,3) and
                int(data['secondary_flow_model']) in ((1,2) if secondary else (0,))
                and int(data['exchange_mode'])==0 and int(data['amr_enabled'])==0
                and not np.any(data['suspended_mass']) and
                (scheduled or (np.all(np.asarray(data['mode'])==2) and
                 not np.any(data['fraction_bed_input']) and
                 not np.any(data.get('nhydro_series',0)) and
                 not np.any(data.get('nfraction_series',0)))) and
                not np.any(data['fraction_concentration']) and
                np.isfinite(wet_depth) and wet_depth>0,
                'Recursive bank foundation requires unsuspended bank transport and admitted boundaries')
        self.levels=np.array(levels,copy=True)
        self.layout=level_topology(data['owner'],data['neighbour'],self.levels)
        require(len(self.levels)==int(data['nc']), 'Cell levels differ from accepted geometry')
        numerical_folder=Path(source_folder)/'live_bank' if surveyed else Path(source_folder)
        bank_flags=(dict(
                        recursive_bedform_bank_avalanche_mode2_variant=True,
                        evolving_toe_variant=self.evolving_toe_variant,
                        external_physical_toe=True)
                    if self.bedform_physical_toe_composition else
                    dict(
                        recursive_bedform_bank_avalanche_mode3_variant=True)
                    if self.bedform_distributed_bank_composition else
                    dict(distributed_bank_avalanche_mode3_variant=True)
                    if self.bank_mode==3 else
                    dict(bank_avalanche_mode2_variant=True,
                         g3_bank_secondary_variant=secondary,
                         evolving_toe_variant=self.evolving_toe_variant,
                         external_physical_toe=True))
        self.solver=CoupledSolver(data,source_folder=numerical_folder,integrator='ssp33',
            distributed_flux_ownership='external_all_face',
            external_boundary_schedule=scheduled,external_level_secondary=secondary,
            external_live_sections=surveyed,**bank_flags)
        self.forcing=None
        if scheduled:
            from .amr_boundary_forcing import RecursiveBoundaryForcing
            self.forcing=RecursiveBoundaryForcing(self.solver,self.levels,
                source_path=numerical_folder/'boundary_schedule.cu',
                live_components=self.solver.live_components)
            self.solver.recursive_boundary_forcing=self.forcing
        self.level_euler=None
        if euler_mode=='level_local':
            from .amr_level_euler import ResidentLevelEuler
            self.level_euler=ResidentLevelEuler(self.solver,self.levels)
        self.cp=self.solver.cp;cp=self.cp
        self.data=self.solver.data;self.state=self.solver.state
        self.nc,self.nf,self.ng=self.solver.nc,self.solver.nf,self.solver.ng
        self.nl=int(data['nbed_layers']);self.ns=int(data['nstate']);self.nlevels=len(self.layout)
        self.secondary_index=int(data['secondary_index'])
        require((int(data['secondary_flow_model'])==2 and self.secondary_index==self.ns-1) or
                (int(data['secondary_flow_model'])!=2 and self.secondary_index==-1),
                'Recursive secondary state schema differs')
        self.wet_depth=float(wet_depth)
        self.area=cp.asarray(data['area']);self.length=cp.asarray(data['length'])
        self.owner=cp.asarray(data['owner']);self.neighbour=cp.asarray(data['neighbour'])
        self.bed_indices=cp.asarray(data['bed_indices']);self.active_indices=self.bed_indices[0].copy()
        self.device_layout=[{key:cp.asarray(value) for key,value in row.items()} for row in self.layout]
        for device,native in ((name,'bank_'+name) for name in FLOAT_HISTORY):
            cp.copyto(self.solver.bank_history[device],cp.asarray(data.get(native,np.zeros(self.nc))))
        cp.copyto(self.solver.bank_history['collapse_count'],
                  cp.asarray(data.get('bank_collapse_count',np.zeros(self.nc,np.int32))))
        self.work=self._new_state();self.euler_state=self._new_state()
        if surveyed:self.forcing.bind_stage_state(self.euler_state['q'])
        self.source_q=cp.zeros_like(self.state['q'])
        self.secondary_source_trial=cp.zeros(())
        self.level_secondary=None if self.level_euler is None else self.level_euler.secondary
        self.secondary_source=None if self.level_secondary is None else self.level_secondary.source_path
        if self.level_secondary is not None:
            self.level_secondary.reserve_filter(self.state['q'],self.state['secondary'])
        self.rk=[[self._new_state() for _ in range(4)] for _ in self.layout]
        self.dense_old=self._new_state();self.dense_new=self._new_state()
        self.dense_times=np.zeros((self.nlevels,2))
        self.stage_integral=[[cp.zeros((self.nf,2,self.ns)) for _ in range(3)] for _ in self.layout]
        self.level_integral=[cp.zeros((self.nf,2,self.ns)) for _ in self.layout]
        self.subtree_integral=[cp.zeros((self.nf,2,self.ns)) for _ in self.layout]
        self.level_scheduler=RecursiveLevelScheduler(
            level_has_work=[len(row['cells']) != 0 for row in self.layout],
            own_faces=[row['own_faces'] for row in self.device_layout],
            level_integrals=self.level_integral,
            subtree_integrals=self.subtree_integral)
        self.accepted_integral=cp.zeros((self.nf,2,self.ns))
        self.integrated_bed=cp.zeros((self.ng,2,self.nf))
        self.flat_work=cp.zeros((self.nc,self.ns))
        self.repair=None;self.repair_source=None
        if reflux_mode=='conservative':
            from .amr_bank_repair import ResidentBankRefluxRepair
            self.repair=ResidentBankRefluxRepair(data['area'],self.levels,data['offset'],
                data['faces'],data['signs'],data['owner'],data['neighbour'],
                active_bed_indices=np.asarray(data['bed_indices'])[0],
                suspended_indices=np.asarray(data['suspended_indices']),secondary_index=self.secondary_index)
            self.repair.reserve(self.ns)
            self.repair_source=self.repair.source_path
            self.repair_totals=cp.zeros((self.ng+1,4))
        self.wet_preparation=None;self.wet_source=None
        if wet_front_mode=='recursive':
            from .amr_bank_wet import ResidentWetPreparation
            self.wet_preparation=ResidentWetPreparation(data['area'],self.levels,
                data['dry'],float(self.solver.stepper.hydro.speed_limit))
            self.wet_source=self.wet_preparation.source_path
        self.checkpoint_schema=3 if self.wet_preparation is not None else 2 if self.repair is not None else 1
        if scheduled:self.checkpoint_schema=4
        if secondary:self.checkpoint_schema=5
        if self.bank_mode==2:self.checkpoint_schema=7
        self.global_flag=cp.zeros((),dtype=cp.int32)
        self.trial_ledgers={key:cp.zeros_like(value) for key,value in self.solver.ledger.items()}
        self.solver.flux.inputs['q']=self.euler_state['q']
        self.accepted_steps=0;self.time_s=0.;self.maximum_cfl=0.
        self.epoch_origin=dict(accepted_steps=0,time_s=0.,euler_calls=0)
        self.macro_cfl=0.
        self.level_steps=np.zeros(self.nlevels,np.int64)
        self.bedform_stage_evaluations=np.zeros(self.nlevels,np.int64)
        self.bedform_friction_evaluations=np.zeros(self.nlevels,np.int64)
        self.bank_updates=0;self.avalanche_updates=0
        source=Path(register_source or HERE/'amr_bank_subcycle.cu')
        code=source.read_text(encoding='utf-8')
        require('#define BRAIDS_BANK_SUBCYCLE_REGISTER_ABI 1' in code,'Subcycle register source ABI differs')
        self.register_source=source.resolve()
        self.source_folder=Path(source_folder).resolve()
        self.level_source=None if self.level_euler is None else self.level_euler.source_path
        self.source_files=self._current_sources()
        self.source_identity=hashlib.sha256(json.dumps(self.source_files,sort_keys=True,
            separators=(',',':')).encode()).hexdigest()
        self.topology_identity=_digest_arrays(dict(data,recursive_cell_levels=self.levels))
        self.method_identity=hashlib.sha256(json.dumps(dict(method=self.method,
            source_identity=self.source_identity,wet_depth=self.wet_depth),sort_keys=True,
            separators=(',',':')).encode()).hexdigest()
        module=cp.RawModule(code=code,options=('--std=c++11','--fmad=false','--prec-div=true',
                            '--prec-sqrt=true','--ftz=false'),name_expressions=('bs_reflux',))
        self.reflux_kernel=module.get_function('bs_reflux')

    def _current_sources(self):
        return _source_files(self.source_folder,self.register_source,self.level_source,self.repair_source,
            self.wet_source,self.forcing is not None,self.secondary_source)

    def persistent(self):
        """Compact accepted state only, never trial buffers or an open clock."""
        require(self.solver._distributed_external_clock is None and
                self.solver._distributed_external_phase is None and
                (self.forcing is None or self.forcing.clock is None),
                'Cannot persist a pending external macro transaction')
        require(self._current_sources()==self.source_files,
                'Recursive source changed after initialization')
        cp=self.cp
        fields={name:np.asarray(value,dtype=np.int64) for name,value in dict(
            schema=self.checkpoint_schema,accepted_steps=self.accepted_steps,euler_calls=self.solver.euler_calls,
            bank_updates=self.bank_updates,avalanche_updates=self.avalanche_updates,
            collapse_waves=(self.solver.distributed_bank.collapse_waves
                            if self.bank_mode==3 else 0)).items()}
        fields['distributed_stage_index' if self.bank_mode==3 else 'toe_stage_index']=np.asarray(
            self.solver.distributed_stage_index if self.bank_mode==3 else self.solver.toe_stage_index,
            dtype=np.int64)
        fields.update({name:np.asarray(value,dtype=np.float64) for name,value in dict(
            time_s=self.time_s,maximum_cfl=self.maximum_cfl).items()})
        fields.update({name:np.asarray(getattr(self,name)) for name in
            ('method_identity','source_identity','topology_identity')})
        fields['level_steps']=self.level_steps.copy()
        if self.checkpoint_schema in (6,8):
            fields.update(epoch_start_steps=np.int64(self.epoch_origin['accepted_steps']),
                epoch_start_time_s=np.float64(self.epoch_origin['time_s']),
                epoch_start_euler_calls=np.int64(self.epoch_origin['euler_calls']))
        fields['accepted_integral']=cp.asnumpy(self.accepted_integral)
        if self.repair is not None:fields['repair_totals']=cp.asnumpy(self.repair_totals)
        if self.wet_preparation is not None:
            fields['hydraulic_preparation_impulse']=cp.asnumpy(self.wet_preparation.impulse)
        for prefix,values in (('state',self.state),('history',self.solver.bank_history),
                              ('ledger',self.solver.ledger)):
            fields.update({prefix+'_'+key:cp.asnumpy(value) for key,value in values.items()})
        if self.bank_mode==2:
            fields['toe_signed']=cp.asnumpy(self.solver.toe_face_signed)
            fields['toe_gross']=cp.asnumpy(self.solver.toe_face_gross)
            if self.solver.evolving_toe is not None:
                fields.update({'toe_map_'+key:cp.asnumpy(value) for key,value in
                               self.solver.evolving_toe.persistent_arrays().items()})
        self._validate_persistent(fields)
        return fields

    def _validate_persistent(self,fields):
        shapes={name:array.shape for prefix,values in (('state',self.state),
            ('history',self.solver.bank_history),('ledger',self.solver.ledger))
            for key,array in values.items() for name in (prefix+'_'+key,)}
        shapes.update(time_s=(),maximum_cfl=(),accepted_integral=(self.nf,2,self.ns))
        if self.repair is not None:shapes['repair_totals']=(self.ng+1,4)
        if self.wet_preparation is not None:shapes['hydraulic_preparation_impulse']=(2,2)
        integers={name:() for name in ('schema','accepted_steps','euler_calls','bank_updates',
                    'avalanche_updates','collapse_waves')}
        integers['distributed_stage_index' if self.bank_mode==3 else 'toe_stage_index']=()
        integers['level_steps']=(self.nlevels,)
        schema=int(np.asarray(fields.get('schema',-1)))
        epoch_schema=schema in (6,8)
        expected_epoch_schema=8 if self.bank_mode==2 else 6
        if epoch_schema:
            shapes['epoch_start_time_s']=()
            integers.update(epoch_start_steps=(),epoch_start_euler_calls=())
        names=('method_identity','source_identity','topology_identity')
        if self.bank_mode==2:
            shapes.update(toe_signed=(self.ng,self.nf),toe_gross=(self.ng,self.nf))
            if self.solver.evolving_toe is not None:
                shapes.update({'toe_map_'+key:value.shape for key,value in
                               self.solver.evolving_toe.persistent_arrays().items()})
        require(set(fields)==set(shapes)|set(integers)|set(names),'Recursive checkpoint fields differ')
        for name,shape in shapes.items():
            value=np.asarray(fields[name])
            if name=='history_collapse_count':
                dtype=np.dtype(np.int32)
            elif name.startswith('toe_map_') and self.solver.evolving_toe is not None:
                dtype=np.dtype(self.solver.evolving_toe.persistent_arrays()[name.removeprefix('toe_map_')].dtype)
            else:
                dtype=np.dtype(np.float64)
            require(value.dtype==dtype and value.shape==shape and np.isfinite(value).all(),
                    'Invalid finite recursive checkpoint field '+name)
        for name,shape in integers.items():
            value=np.asarray(fields[name])
            require(value.dtype==np.int64 and value.shape==shape and np.all(value>=0),
                    'Invalid recursive checkpoint counter '+name)
        for name in names:
            value=np.asarray(fields[name])
            require(value.shape==() and value.dtype.kind=='U' and str(value)==getattr(self,name),
                    'Recursive checkpoint '+name+' differs')
        step=int(fields['accepted_steps']);current=float(fields['time_s'])
        origin_step=int(fields['epoch_start_steps']) if epoch_schema else 0
        origin_time=float(fields['epoch_start_time_s']) if epoch_schema else 0.
        origin_euler=int(fields['epoch_start_euler_calls']) if epoch_schema else 0
        expected=np.array([(step-origin_step)*(1<<level) if len(row['cells']) else 0
                           for level,row in enumerate(self.layout)],np.int64)
        require(((schema==expected_epoch_schema) if epoch_schema else
                 schema==self.checkpoint_schema) and
                0<=origin_step<=step and 0<=origin_time<=current and
                (origin_step==0)==(origin_time==0.) and (origin_step==0)==(origin_euler==0) and
                (origin_euler%3==0) and
                np.array_equal(fields['level_steps'],expected) and
                int(fields['euler_calls'])==origin_euler+3*int(expected.sum()) and
                int(fields['bank_updates'])==step and
                int(fields['avalanche_updates'])==(step if self.solver.avalanche is not None else 0) and
                int(fields['distributed_stage_index' if self.bank_mode==3 else 'toe_stage_index'])==0 and
                int(fields['collapse_waves'])<=(self.nc if self.bank_mode==3 else 0) and
                0<=current<=min(float(self.data['endtime']),origin_time+(step-origin_step)*float(self.data['dtmax']))+1.e-12 and
                (current==0)==(step==0) and
                0<=float(fields['maximum_cfl'])<=float(self.data['cfl_target'])+1.e-12,
                'Recursive checkpoint clock/counters differ')
        if self.bank_mode==2:
            tolerance=1.e-12*np.maximum(1.,np.maximum(np.abs(fields['toe_signed']),fields['toe_gross']))
            require(np.all(fields['toe_gross']>=0) and
                    np.all(fields['toe_gross']+tolerance>=np.abs(fields['toe_signed'])),
                    'Recursive physical-toe checkpoint diagnostics differ')
        require(np.all(fields['state_q'][0]>=0) and np.all(fields['state_layers']>=0) and
                (self.secondary_index>=0 or np.all(fields['state_secondary']==0)) and
                all(np.all(fields['history_'+name]>=0) for name in self.solver.bank_history),
                'Recursive checkpoint state/history is negative or unsupported')
        if self.repair is not None:
            totals=fields['repair_totals']
            require(np.all(totals[:,:2]>=0) and np.all(totals[:,0]==np.floor(totals[:,0])) and
                    np.all(totals[1:,2:]==0) and (step>0 or np.all(totals==0)),
                    'Invalid recursive repair diagnostics')
        internal=np.asarray(self.data['neighbour'])>=0;depth=fields['state_q'][0]
        if self.wet_preparation is None:
            require(not np.any((depth[np.asarray(self.data['owner'])[internal]]>self.wet_depth)!=
                               (depth[np.asarray(self.data['neighbour'])[internal]]>self.wet_depth)),
                    'Recursive checkpoint has an unqualified wet front')
        else:
            require(np.all(fields['state_q'][1:,depth<=self.data['dry']]==0),
                    'Recursive wet checkpoint contains dry momentum')
            require(step>0 or not np.any(fields['hydraulic_preparation_impulse']),
                    'Fresh recursive checkpoint contains a preparation impulse')
        indices=np.asarray(self.data['bed_indices'])[0]
        integral=fields['accepted_integral']
        used=np.r_[np.arange(3),indices,([self.secondary_index] if self.secondary_index>=0 else [])]
        unused=np.setdiff1d(np.arange(self.ns),used)
        require(np.array_equal(integral[:,0,indices],integral[:,1,indices]) and
                np.all(integral[:,:,unused]==0) and (step>origin_step or np.all(integral==0)),
                'Recursive checkpoint accepted register is invalid')
        if self.secondary_index>=0:
            require(np.array_equal(integral[:,0,self.secondary_index],integral[:,1,self.secondary_index]),
                    'Recursive secondary face register is not conservative')

    def restore(self,fields):
        """Restore a fully validated accepted macro boundary, never a trial."""
        require(self.accepted_steps==0 and self.time_s==0 and
                self.solver._distributed_external_clock is None and
                self.solver._distributed_external_phase is None,
                'Recursive restore requires a fresh, idle stepper')
        require(self._current_sources()==self.source_files,
                'Recursive source changed after initialization')
        self._validate_persistent(fields)
        saved=self._snapshot()
        try:
            for prefix,values in (('state',self.state),('history',self.solver.bank_history),
                                  ('ledger',self.solver.ledger)):
                for key,value in values.items():self.cp.copyto(value,self.cp.asarray(fields[prefix+'_'+key]))
            self.cp.copyto(self.accepted_integral,self.cp.asarray(fields['accepted_integral']))
            if self.repair is not None:self.cp.copyto(self.repair_totals,self.cp.asarray(fields['repair_totals']))
            if self.wet_preparation is not None:
                self.cp.copyto(self.wet_preparation.impulse,self.cp.asarray(fields['hydraulic_preparation_impulse']))
            for name in ('accepted_steps','bank_updates','avalanche_updates'):
                setattr(self,name,int(fields[name]))
            self.time_s=float(fields['time_s']);self.maximum_cfl=float(fields['maximum_cfl'])
            self.level_steps=np.array(fields['level_steps'],copy=True)
            self.solver.euler_calls=int(fields['euler_calls'])
            if self.bank_mode==3:
                self.solver.distributed_stage_index=int(fields['distributed_stage_index'])
                self.solver.distributed_bank.collapse_waves=int(fields['collapse_waves'])
            else:
                self.solver.toe_stage_index=int(fields['toe_stage_index'])
                self.cp.copyto(self.solver.toe_face_signed,self.cp.asarray(fields['toe_signed']))
                self.cp.copyto(self.solver.toe_face_gross,self.cp.asarray(fields['toe_gross']))
                if self.solver.evolving_toe is not None:
                    toe={key:fields['toe_map_'+key] for key in
                         self.solver.evolving_toe.persistent_arrays()}
                    self.solver.evolving_toe.restore(toe,self.solver.bed(),
                        self.state['q'][0],self.state['layers'])
            if int(fields['schema']) in (6,8):
                self.epoch_origin=dict(accepted_steps=int(fields['epoch_start_steps']),
                    time_s=float(fields['epoch_start_time_s']),euler_calls=int(fields['epoch_start_euler_calls']))
                self.checkpoint_schema=8 if self.bank_mode==2 else 6
        except BaseException:
            self._rollback(saved)
            raise

    def write_checkpoint(self,folder):
        folder=Path(folder)
        require(not folder.exists(),'Preserve existing recursive checkpoint')
        fields=self.persistent();folder.mkdir(parents=True)
        save_arrays_atomic(folder/'accepted.npz',**fields)
        write_json_atomic(folder/'checkpoint.json',dict(schema=int(fields['schema']),kind=self.method,
            payload_sha256=hashlib.sha256((folder/'accepted.npz').read_bytes()).hexdigest(),
            method_identity=self.method_identity,topology_identity=self.topology_identity,
            source_identity=self.source_identity,source_files=self.source_files))

    def restore_checkpoint(self,folder):
        folder=Path(folder);manifest=json.loads((folder/'checkpoint.json').read_text())
        require(set(manifest)=={'schema','kind','payload_sha256','method_identity','topology_identity',
            'source_identity','source_files'} and manifest['schema']==self.checkpoint_schema and manifest['kind']==self.method and
            all(manifest[name]==getattr(self,name) for name in
                ('method_identity','topology_identity','source_identity','source_files')) and
            manifest['payload_sha256']==hashlib.sha256((folder/'accepted.npz').read_bytes()).hexdigest(),
            'Recursive checkpoint manifest/payload authentication failed')
        with np.load(folder/'accepted.npz',allow_pickle=False) as archive:
            fields={key:archive[key] for key in archive.files}
        self.restore(fields)

    def _new_state(self):
        return {key:self.cp.zeros_like(value) for key,value in self.state.items()}

    def _require_wet_scope(self,state):
        if self.wet_preparation is not None:
            require(bool(self.cp.all(self.cp.isfinite(state['q'])).item()) and
                    bool(self.cp.all(state['q'][0]>=0).item()),
                    'Recursive wet state is nonfinite or has negative water')
            return
        internal=self.neighbour>=0;depth=state['q'][0]
        require(not bool(self.cp.any((depth[self.owner[internal]]>self.wet_depth)!=
                    (depth[self.neighbour[internal]]>self.wet_depth)).item()),
                'Subcycled bank wet-front evolution is not qualified')

    def _copy(self, target, source, indices=None):
        if indices is None:
            for key in target:self.cp.copyto(target[key],source[key])
        else:
            for key in target:target[key][...,indices]=source[key][...,indices]

    def full(self, state=None):
        state=self.state if state is None else state
        result=self.cp.zeros((self.nc,self.ns))
        result[:,:3]=state['q'].T
        for layer in range(self.nl):result[:,self.bed_indices[layer]]=state['layers'][layer].T
        if self.secondary_index>=0:result[:,self.secondary_index]=state['secondary']
        return result

    def _install_rows(self, rows, state):
        self.cp.copyto(state['q'],self.cp.ascontiguousarray(rows[:,:3].T))
        for layer in range(self.nl):
            self.cp.copyto(state['layers'][layer],self.cp.ascontiguousarray(rows[:,self.bed_indices[layer]].T))
        if self.secondary_index>=0:self.cp.copyto(state['secondary'],self.cp.ascontiguousarray(rows[:,self.secondary_index]))
        else:state['secondary'].fill(0.)

    def _ghost(self,state,current,level):
        for coarse in range(level):
            indices=self.device_layout[coarse]['cells']
            if not len(indices):continue
            start,end=self.dense_times[coarse]
            alpha=0. if end<=start else min(1.,max(0.,(current-start)/(end-start)))
            for key in state:
                state[key][...,indices]=((1.-alpha)*self.dense_old[key][...,indices]+
                                        alpha*self.dense_new[key][...,indices])
        self._require_wet_scope(state)

    def _rebalance(self,state,level=None):
        status=(self.level_euler.rebalance(state['layers'],level)
                if self.level_euler is not None and level is not None
                else self.solver.sediment.rebalance(state['layers']))
        require(not bool(self.cp.any(status).item()),
                'Subcycle active-layer rebalance failed')

    def _euler(self,state,dt,current,level,stage,output,*,prepared=False):
        cp=self.cp
        if self.wet_preparation is not None and not prepared:
            self.wet_preparation.prepare(state['q'],level,(1.,1./6.,2./3.)[stage])
        self._copy(self.euler_state,state)
        hydro=self.solver.stepper.hydro
        if self.wet_preparation is None:
            hydro.status.fill(0);hydro._prepare(self.euler_state['q'],None)
            require(not bool(hydro.status.item()),'Subcycle hydraulic preparation failed')
        self._require_wet_scope(self.euler_state)
        try:
            result=(self.solver.euler(self.euler_state,dt,current) if self.level_euler is None
                    else self.level_euler.evaluate(self.euler_state,dt,current,level))
        except RuntimeError as error:
            if self.adaptation_policy is None or str(error) not in (
                    'GPU bed-load adaptation solve did not converge','Invalid GPU adapted bed-load capacity'):
                raise
            raise RecursiveStepRejected(str(error)) from error
        if bool(cp.any(result['status']).item()):
            raise RecursiveStepRejected('Subcycle Euler status failed')
        if self.solver.bedform is not None:
            self.bedform_stage_evaluations[level]+=1
        if self.level_euler is None:
            require(bool(cp.all(self.solver.flux.buffers['theta']==1.).item()) and
                    bool(cp.all(self.solver.exner.buffers['theta']==1.).item()),
                    'Subcycle level-local draining/bed donor limiting is not yet qualified')
        indices=self.device_layout[level]['cells']
        cfl=float(cp.max(result['cfl'][indices]).item())
        if not np.isfinite(cfl) or cfl>float(self.data['cfl_target'])+1.e-12:
            raise RecursiveStepRejected('Subcycle active-level CFL rejected',cfl=cfl)
        self.maximum_cfl=max(self.maximum_cfl,cfl)
        self.macro_cfl=max(self.macro_cfl,cfl)
        self._copy(output,state);self._copy(output,result['state'],indices)
        self._require_wet_scope(output)
        integral=self.stage_integral[level][stage];integral.fill(0.)
        factor=dt*self.length[:,None]
        integral[:,0,:3]=factor*self.solver.flux.buffers['fl'].T
        integral[:,1,:3]=factor*self.solver.flux.buffers['fr'].T
        bed=factor*self.solver.exner.buffers['flux'].T
        integral[:,0,self.active_indices]=bed
        integral[:,1,self.active_indices]=bed
        if self.secondary_index>=0:
            flux=dt*self.length*self.solver.secondary.buffers['flux']
            integral[:,0,self.secondary_index]=flux
            integral[:,1,self.secondary_index]=flux

    def _relax_secondary(self,state,dt,level):
        result=self.level_secondary.relax(state['q'],state['secondary'],dt,level)
        require(not bool(result['status'].item()),'Recursive secondary half-source failed')
        self.cp.copyto(state['secondary'],result['next'])
        self.secondary_source_trial+=result['source_integral']

    def _rk(self,level,dt,current):
        cp=self.cp;indices=self.device_layout[level]['cells']
        if self.forcing is not None:self.forcing.begin_rk(level,current,dt)
        initial,a,b,final=self.rk[level]
        self._copy(initial,self.work);self._ghost(initial,current,level)
        if self.secondary_index>=0:
            self.wet_preparation.prepare(initial['q'],level)
            self._relax_secondary(initial,.5*dt,level)
            self._euler(initial,dt,current,level,0,a,prepared=True)
        else:self._euler(initial,dt,current,level,0,a)
        self._ghost(a,current+dt,level)
        self._euler(a,dt,current+dt,level,1,b)
        for key in b:b[key][...]=.75*initial[key]+.25*b[key]
        self._rebalance(b,level);self._ghost(b,current+.5*dt,level)
        self._euler(b,dt,current+.5*dt,level,2,final)
        if self.forcing is not None:self.forcing.finish_rk()
        for key in final:final[key][...]=initial[key]/3.+2.*final[key]/3.
        self._rebalance(final,level)
        hydro=self.solver.stepper.hydro
        if self.wet_preparation is not None:
            self.wet_preparation.prepare(final['q'],level)
        else:
            hydro.status.fill(0);hydro._prepare(final['q'],None)
            require(not bool(hydro.status.item()),'Subcycle final hydraulic preparation failed')
        self._require_wet_scope(final)
        # Only advancing cells receive friction. Off-level ghost velocities
        # are still inputs to the second-half full-domain curvature solve.
        cp.copyto(self.source_q,final['q'])
        roughness=hydro.manning
        if self.solver.bedform is not None:
            roughness,status=self.solver.friction_roughness(final)
            require(not bool(cp.any(status).item()),
                    'Subcycle pre-friction bedform feedback failed')
            self.bedform_friction_evaluations[level]+=1
        hydro._launch('ti_manning',self.nc,(self.source_q,roughness,np.int64(self.nc),
            np.float64(dt),hydro.grav,hydro.dry,hydro.chezy_min,hydro.cell_status))
        require(not bool(cp.any(hydro.cell_status[indices]).item()),'Subcycle friction source failed')
        final['q'][:,indices]=self.source_q[:,indices]
        if self.secondary_index>=0:self._relax_secondary(final,.5*dt,level)
        self._copy(self.dense_old,self.work,indices);self._copy(self.dense_new,final,indices)
        self.dense_times[level]=(current,current+dt)
        self._copy(self.work,final,indices)
        self._require_wet_scope(self.work)
        fa,fb,fc=self.stage_integral[level]
        self.level_integral[level][...]=(fa+fb)/6.+2.*fc/3.
        self.level_steps[level]+=1

    def _reflux(self,level,coarse,fine):
        cp=self.cp;layout=self.device_layout[level];groups=len(layout['coarse_cells'])
        if not groups:return
        cp.copyto(self.flat_work,self.full(self.work))
        self.reflux_kernel(((groups*self.ns+127)//128,),(128,),(
            self.flat_work,coarse,fine,layout['coarse_cells'],layout['offset'],layout['faces'],
            layout['sides'],self.area,np.int32(groups),np.int32(self.ns)))
        if self.repair is not None:
            self._repair_rows(level)
        require(bool(cp.all(cp.isfinite(self.flat_work)).item()) and
                bool(cp.all(self.flat_work[:,0]>=0).item()) and
                bool(cp.all(self.flat_work[:,3:self.secondary_index if self.secondary_index>=0 else self.ns]>=0).item()),
                'Subcycle reflux requires unqualified native positivity/phase repair')
        self._install_rows(self.flat_work,self.work);self._rebalance(self.work,level)
        self._require_wet_scope(self.work)

    def _repair_rows(self,level,*,root_water=False):
        proposal=self.repair.propose(self.flat_work,level,root_water=root_water)
        require(not bool(proposal['status'].item()),'Conservative subcycle reflux repair rejected')
        self.cp.copyto(self.flat_work,proposal['state'])
        self.repair_totals+=proposal['diagnostics']

    def _subcycle(self,level,dt,current):
        require(level==0,'Recursive scheduler must start at its root level')
        return self.level_scheduler.advance(dt,current,self._rk,self._reflux)

    def _snapshot(self,*,independent=False):
        arrays=_unique_arrays(self);current=signature(arrays)
        plan=None if independent else _PLANS.get(self)
        if plan is None or plan.signature!=current:
            _require_disjoint_storage(arrays);plan=SegmentedCopyPlan(self.cp,arrays)
            if not independent:_PLANS[self]=plan
        plan.snapshot()
        result=dict(plan=plan,accepted_steps=self.accepted_steps,time_s=self.time_s,
            maximum_cfl=self.maximum_cfl,macro_cfl=self.macro_cfl,
            level_steps=self.level_steps.copy(),dense_times=self.dense_times.copy(),
            bedform_stage_evaluations=self.bedform_stage_evaluations.copy(),
            bedform_friction_evaluations=self.bedform_friction_evaluations.copy(),
            euler_calls=self.solver.euler_calls,bank_updates=self.bank_updates,avalanche_updates=self.avalanche_updates,
            bank_trial=getattr(self.solver,'bank_trial',None),
            stage=(self.solver.distributed_stage_index if self.bank_mode==3
                   else self.solver.toe_stage_index),
            flux_dt=self.solver.flux.controls['dt'],
            filter_receipt=copy.deepcopy(self.solver.secondary.last_filter_receipt),
            adaptation=(None if self.adaptation_policy is None else (
                self.solver.sediment.adaptation.iterations,self.solver.sediment.adaptation.residual,
                copy.deepcopy(self.solver.sediment.adaptation_result))),
            avalanche_trial_substeps=self.solver.avalanche_trial_substeps.copy())
        if self.bank_mode==3:
            result['waves']=self.solver.distributed_bank.collapse_waves
        return result

    def _rollback(self,saved):
        saved['plan'].restore()
        for key in ('accepted_steps','time_s','maximum_cfl','macro_cfl','level_steps','dense_times',
                    'bedform_stage_evaluations','bedform_friction_evaluations',
                    'bank_updates','avalanche_updates'):
            setattr(self,key,saved[key])
        self.solver.euler_calls=saved['euler_calls'];self.solver.bank_trial=saved['bank_trial']
        if self.bank_mode==3:
            self.solver.distributed_bank.collapse_waves=saved['waves']
            self.solver.distributed_stage_index=saved['stage']
        else:
            self.solver.toe_stage_index=saved['stage']
        self.solver.flux.controls['dt']=saved['flux_dt']
        self.solver.avalanche_trial_substeps=saved['avalanche_trial_substeps']
        self.solver.secondary.last_filter_receipt=saved['filter_receipt']
        if saved['adaptation'] is not None:
            (self.solver.sediment.adaptation.iterations,self.solver.sediment.adaptation.residual,
             self.solver.sediment.adaptation_result)=saved['adaptation']

    def _totals(self,state):
        cp=self.cp
        parts=[cp.sum(state['q'][0]*self.area)[None],
            cp.sum(cp.sum(state['layers'],axis=0)*self.area[None,:],axis=1)]
        if self.secondary_index>=0:parts.append(cp.sum(state['secondary']*self.area)[None])
        return cp.concatenate(parts)

    def begin_topology_epoch(self,previous):
        """Adopt cumulative macro diagnostics after conservative private remap.

        Level counts restart at zero for this graph, whereas physical ledgers,
        source/sink totals and global macro/Euler counts remain cumulative.
        The old all-face integral was consumed and is not a pending new flux.
        """
        require(isinstance(previous,SubcycledDistributedBankStepper) and previous is not self and
                self.accepted_steps==0 and self.time_s==0 and
                self.method_identity==previous.method_identity and
                self.bank_mode==previous.bank_mode and
                self.ng==previous.ng and self.ns==previous.ns and
                all(model.solver._distributed_external_clock is None and
                    model.solver._distributed_external_phase is None and
                    (model.forcing is None or model.forcing.clock is None)
                    for model in (self,previous)),
                'Invalid recursive topology epoch handoff')
        require(not bool(self.cp.any(self.accepted_integral).item()),
                'New topology already has an accepted face integral')
        self.epoch_origin=dict(accepted_steps=previous.accepted_steps,time_s=previous.time_s,
                              euler_calls=previous.solver.euler_calls)
        for name in ('accepted_steps','time_s','maximum_cfl','bank_updates','avalanche_updates'):
            setattr(self,name,getattr(previous,name))
        self.solver.euler_calls=previous.solver.euler_calls
        self.level_steps.fill(0)
        self.bedform_stage_evaluations.fill(0)
        self.bedform_friction_evaluations.fill(0)
        self.checkpoint_schema=8 if self.bank_mode==2 else 6
        if self.repair is not None:self.cp.copyto(self.repair_totals,previous.repair_totals)
        if self.wet_preparation is not None:
            self.cp.copyto(self.wet_preparation.impulse,previous.wet_preparation.impulse)
        # collapse_waves is a last-trial iteration count, not cumulative history;
        # do not carry a value bounded by the old cell count onto a smaller mesh.
        if self.bank_mode==3:self.solver.distributed_bank.collapse_waves=0

    def next_timestep(self,hint,*,output=None):
        """Clip a requested macro step at forcing/output/end; no hidden retry."""
        from validation.gpu_fwave.boundary_events import BoundaryEvents
        events=self.forcing.events if self.forcing is not None else BoundaryEvents([])
        end=float(self.data['endtime'])
        return events.trial(self.time_s,min(float(hint),float(self.data['dtmax'])),
            output=end if output is None else float(output),end=end).dt

    def step(self,dt,current=None,*,reject_after=None,snapshot=None):
        current=self.time_s if current is None else float(current)
        require(np.isfinite([dt,current]).all() and 0<dt<=float(self.data['dtmax']) and
                current==self.time_s and current+dt<=float(self.data['endtime'])+1.e-12 and
                reject_after in (None,'reflux','bank','avalanche'), 'Invalid subcycle clock/injection')
        require(self.solver._distributed_external_clock is None,
                'Subcycled bank macro transaction is already open')
        if self.forcing is not None:self.forcing.require_macro_clock(current,dt)
        cp=self.cp;self._require_wet_scope(self.state)
        saved=self._snapshot() if snapshot is None else snapshot
        before=self._totals(self.state)
        try:
            self.macro_cfl=0.
            self.bedform_stage_evaluations.fill(0)
            self.bedform_friction_evaluations.fill(0)
            self.solver.begin_external_bank_trial(dt,current)
            self.secondary_source_trial.fill(0.)
            self._copy(self.work,self.state);self._copy(self.dense_old,self.state);self._copy(self.dense_new,self.state)
            self.dense_times[:]=current
            root=self._subcycle(0,float(dt),current)
            if self.repair is not None:
                cp.copyto(self.flat_work,self.full(self.work))
                self._repair_rows(-1,root_water=True)
                self._install_rows(self.flat_work,self.work)
            self._require_wet_scope(self.work)
            if reject_after=='reflux':raise RuntimeError('injected post-reflux bank-subcycle rejection')
            self.integrated_bed[...]=root[:,:,self.active_indices].transpose(2,1,0)
            self.solver.install_external_bank_integral(self.integrated_bed,dt=dt,current=current)
            if self.bank_mode==2:
                flag=self.solver.bank_avalanche_pre_commit(self.work,dt)
                require(not bool(flag.item()),'Subcycled physical-toe bank transaction failed')
                self.bank_updates+=1
                if self.solver.avalanche is not None:self.avalanche_updates+=1
            else:
                flag=self.solver.distributed_bank_pre_commit(self.work,dt)
                require(not bool(flag.item()),'Subcycled distributed bank transaction failed')
                self.bank_updates+=1
            if reject_after=='bank':raise RuntimeError('injected post-bank bank-subcycle rejection')
            if self.bank_mode==3:
                if self.solver.avalanche is not None:
                    result=self.solver.avalanche.step(self.work['layers'],self.work['q'][0],dt)
                    require(not bool(result['status'].item()),'Subcycled bank avalanche failed')
                    cp.copyto(self.work['layers'],result['layers'])
                    self.solver.avalanche_trial_substeps.append(int(result['substeps']))
                    self.avalanche_updates+=1
                else:self.solver.avalanche_trial_substeps.append(0)
            self._rebalance(self.work)
            if self.wet_preparation is not None:
                self.wet_preparation.prepare(self.work['q'],-1,dry_only=True)
            if self.secondary_index>=0:
                # Root reflux or bank/avalanche work can leave a newly dry cell
                # after the last level half-source. Apply only the native dry
                # source rule here, once, and account for its signed inventory.
                # This accepted-boundary ordering extension is not a remap fix.
                cleaned=self.level_secondary.cleanup_dry(self.work['q'],self.work['secondary'])
                require(not bool(cleaned['status'].item()),'Accepted dry secondary cleanup failed')
                cp.copyto(self.work['secondary'],cleaned['next'])
                self.secondary_source_trial+=cleaned['source_integral']
            self._require_wet_scope(self.work)
            if reject_after=='avalanche':raise RuntimeError('injected post-avalanche bank-subcycle rejection')
            for value in self.trial_ledgers.values():value.fill(0.)
            for boundary in range(self.solver.nb):
                faces=cp.asarray(np.flatnonzero(np.asarray(self.data['bid'])==boundary))
                self.trial_ledgers['water'][boundary]=-cp.sum(root[faces,0,0])
                self.trial_ledgers['bed'][:,boundary]=-cp.sum(root[faces,0][:,self.active_indices],axis=0)
                if self.secondary_index>=0:
                    self.trial_ledgers['secondary'][boundary]=-cp.sum(root[faces,0,self.secondary_index])
            if self.secondary_index>=0:self.cp.copyto(self.trial_ledgers['source'],self.secondary_source_trial)
            boundary=cp.concatenate((cp.sum(self.trial_ledgers['water'])[None],
                                    cp.sum(self.trial_ledgers['bed'],axis=1)))
            if self.secondary_index>=0:
                boundary=cp.concatenate((boundary,(cp.sum(self.trial_ledgers['secondary'])+
                    self.trial_ledgers['source'])[None]))
            residual=self._totals(self.work)-before-boundary
            require(bool(cp.all(cp.isfinite(residual)).item()) and
                    abs(float(residual[0].item()))<=1.e-8 and
                    float(cp.max(cp.abs(residual[1:])).item())<=1.e-7 and
                    all(bool(cp.all(cp.isfinite(value)).item()) for value in self.work.values()) and
                    bool(cp.all(self.work['q'][0]>=0).item()) and bool(cp.all(self.work['layers']>=0).item()),
                    'Subcycled bank conservation/positivity failed')
            self._copy(self.state,self.work);cp.copyto(self.accepted_integral,root)
            self.global_flag.fill(0)
            if self.bank_mode==2:self.solver.physical_toe_commit(self.global_flag)
            else:self.solver.distributed_bank_commit(self.global_flag)
            for key in self.solver.ledger:self.solver.ledger[key]+=self.trial_ledgers[key]
            self.accepted_steps+=1;self.time_s=current+dt
            bedform_stage=self.bedform_stage_evaluations.tolist()
            bedform_friction=self.bedform_friction_evaluations.tolist()
            bedform_calls=int(self.bedform_stage_evaluations.sum()+
                              self.bedform_friction_evaluations.sum())
            level_delta=np.asarray(self.level_steps)-np.asarray(saved['level_steps'])
            require((self.solver.bedform is None and bedform_calls==0) or
                    (self.solver.bedform is not None and
                     self.bedform_stage_evaluations.sum()==
                        3*self.bedform_friction_evaluations.sum() and
                     np.array_equal(self.bedform_friction_evaluations,
                                    level_delta)),
                    'Recursive bank bedform evaluation schedule differs')
            avalanche_substeps=(int(self.solver.avalanche_trial_substeps[0])
                if self.solver.avalanche_trial_substeps else 0)
            require((self.solver.avalanche is None and
                     self.solver.avalanche_trial_substeps in ([],[0])) or
                    (self.solver.avalanche is not None and
                     len(self.solver.avalanche_trial_substeps)==1 and
                     avalanche_substeps>=0),
                    'Recursive bank parent avalanche schedule differs')
            return dict(time_s=self.time_s,steps=self.accepted_steps,level_steps=self.level_steps.tolist(),
                euler_calls=self.solver.euler_calls,bank_updates=self.bank_updates,
                avalanche_updates=self.avalanche_updates,maximum_cfl=self.maximum_cfl,cfl=self.macro_cfl,
                bedform_order=(BEDFORM_BANK_ORDER if self.solver.bedform is not None else 'disabled'),
                bedform_stage_evaluations_per_level=bedform_stage,
                bedform_friction_evaluations_per_level=bedform_friction,
                bedform_evaluations_per_macro=bedform_calls,
                avalanche_substeps=avalanche_substeps,
                budget_residual=cp.asnumpy(residual).tolist(),
                route=((BEDFORM_PHYSICAL_TOE_METHOD
                        if self.bedform_physical_toe_composition else
                        BEDFORM_BANK_METHOD) if self.bedform_bank_composition else
                       ('research_recursive_mode2_physical_toe' if self.bank_mode==2 else
                        'research_recursive_mode3_secondary') if self.level_secondary is not None else
                       'research_recursive_mode3_scheduled_forcing' if self.forcing is not None else
                       'research_recursive_mode3_wet_preparation' if self.wet_preparation is not None else
                       'research_recursive_mode3_conservative_reflux' if self.repair is not None else
                       'research_recursive_mode3_all_face_register' if self.level_euler is None
                       else 'research_recursive_mode3_level_local_euler'))
        except BaseException:
            self._rollback(saved)
            raise
        finally:
            self.solver.end_external_bank_trial()
            if self.forcing is not None:self.forcing.clear()
