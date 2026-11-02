"""Isolated native-semantics level-local unsuspended GPU Euler adapter.

This preserves the existing augmented f-wave and sediment reconstructions.
Derived reconstruction/capacity is refreshed on the whole graph initially;
only incident faces participate in donor sums and only current-level cells
are advanced. Ghost cells still supply their real water/grain inventories to
the limiter. No claim of sparse reconstruction performance is made.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np


HERE=Path(__file__).resolve().parent
OPTIONS=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true','--ftz=false')


def require(ok,message):
    if not ok:raise ValueError(message)


def level_masks(data,levels):
    """Immutable schedules; no host physical computation at an Euler stage."""
    levels=np.asarray(levels);owner=np.asarray(data['owner']);other=np.asarray(data['neighbour'])
    nc=len(data['area']);nf=len(owner)
    require(levels.shape==(nc,) and levels.dtype.kind in 'iu' and
        np.all((levels>=0)&(levels<=30)),'Invalid level-local cell levels')
    require(owner.shape==(nf,) and other.shape==(nf,) and owner.dtype.kind in 'iu' and
        other.dtype.kind in 'iu' and np.all((owner>=0)&(owner<nc)) and
        np.all((other>=-1)&(other<nc)) and np.all(owner!=other),'Invalid level-local topology')
    internal=other>=0
    require(np.all(np.abs(levels[owner[internal]]-levels[other[internal]])<=1),
            'Level-local Euler requires a balanced 2:1 graph')
    offset,faces,signs=(np.asarray(data[k]) for k in ('offset','faces','signs'))
    weno=np.asarray(data['weno_cells'])
    result=[]
    for level in range(int(levels.max())+1):
        cells=levels==level
        incident=cells[owner]|(internal&cells[np.maximum(other,0)])
        endpoints=np.zeros(nc,bool);endpoints[owner[incident]]=True
        endpoints[other[incident&internal]]=True
        halo=endpoints.copy()
        ids=weno[...,incident];halo[ids[ids>=0]]=True
        for cell in np.flatnonzero(halo.copy()):
            ff=faces[offset[cell]:offset[cell+1]];ss=signs[offset[cell]:offset[cell+1]]
            adjacent=np.where(ss>0,other[ff],owner[ff]);halo[adjacent[adjacent>=0]]=True
        result.append(dict(cells=cells.astype(np.int32),faces=incident.astype(np.int32),
                           endpoints=endpoints.astype(np.int32),halo=halo.astype(np.int32)))
    return result


class ResidentLevelEuler:
    """One strict external-bank macro owner, arbitrary admitted grain/layer counts."""
    def __init__(self,solver,levels,*,source_path=None):
        data=solver.data
        external_bank=(solver.distributed_bank_mode3_variant or
            (solver.physical_toe_mode2_variant and
             getattr(solver,'external_physical_toe',False)))
        require(solver.distributed_flux_ownership=='external_all_face' and
            external_bank and
            (int(data['secondary_flow_model'])==0 or solver.external_level_secondary) and
            int(data['exchange_mode'])==0 and solver.inlet_schedule is None and
            not solver.has_suspended and int(data['bedlaw'])==2 and
            (not int(data['bedload_adaptation']) or
             (solver.external_bedload_adaptation and solver.external_live_sections and
              int(data['secondary_flow_model'])==2)) and solver.diffusion is None and
            (solver.bedform is None or
             getattr(solver,
                 'recursive_bedform_bank_avalanche_mode2_variant',False) or
             getattr(solver,
                 'recursive_bedform_bank_avalanche_mode3_variant',False)) and
            (np.all(np.asarray(data['mode'])==2) or solver.external_boundary_schedule),
            'Level-local Euler requires explicit recursive unsuspended bank stage scope')
        self.solver=solver;self.cp=cp=solver.cp
        self.adaptation=bool(data['bedload_adaptation'])
        if self.adaptation:
            require(solver.sediment.adaptation is not None,'Recursive adaptation operator is missing')
            solver.sediment.capacity_validator=self._validate_adaptation_capacity
        self.secondary=None
        if solver.external_level_secondary:
            from .amr_level_secondary import ResidentLevelSecondary
            self.secondary=ResidentLevelSecondary(solver.secondary,data,levels)
        self.nc,self.nf,self.ng=solver.nc,solver.nf,solver.ng
        self.layout=level_masks(data,levels)
        self.masks=[{name:cp.asarray(value) for name,value in row.items()} for row in self.layout]
        self.all_cells=cp.ones(self.nc,dtype=cp.int32)
        self.kind=cp.zeros(self.nf,dtype=cp.int32)
        self.coupled_kind=cp.zeros(self.nf,dtype=cp.int32)
        self.bed_active=cp.zeros(self.nc,dtype=cp.int32)
        self.raw_bed=cp.zeros((self.ng,self.nf))
        self.rebalance_work=cp.zeros_like(solver.state['layers'])
        self.rebalance_status=cp.zeros(self.nc,dtype=cp.int32)
        self.scoped={name:cp.zeros(size,dtype=cp.int32) for name,size in
                     (('cell',self.nc),('face',self.nf),('bedface',self.ng*self.nf),
                      ('graincell',self.ng*self.nc))}
        self.zero_secondary_ledger=cp.zeros(solver.nb)
        source=Path(source_path or HERE/'amr_level_euler.cu')
        code=source.read_text(encoding='utf-8')
        require('#define BRAIDS_AMR_LEVEL_EULER_ABI 2' in code,'Level-local Euler CUDA ABI differs')
        self.source_path=source.resolve();self.source_sha256=hashlib.sha256(code.encode()).hexdigest()
        names=('ale_kinds','ale_coupled_kinds','ale_active','ale_mask_flux','ale_scoped_status','ale_copy_level')
        self.module=cp.RawModule(code=code,options=OPTIONS,name_expressions=names)
        self.k={name:self.module.get_function(name) for name in names}

    def _launch(self,kernel,n,args):
        if n:kernel(((n+127)//128,),(128,),args)

    def _level(self,level):
        require(type(level) is int and 0<=level<len(self.masks) and
                np.any(self.layout[level]['cells']),'Euler requires a populated integer level')
        return self.masks[level]

    def _validate_adaptation_capacity(self,q,cell_status,rotation_status):
        """The steady closure depends on all leaves, not a finite RK halo."""
        from .amr_bank_subcycle import RecursiveStepRejected
        cp=self.cp;s=self.solver
        s._merge((cell_status,rotation_status))
        valid=cp.all(cp.isfinite(q)) & cp.all(q[0]>=0)
        valid &= cp.all(cp.isfinite(s.sediment.buffers['capacity']))
        valid &= cp.all(cp.isfinite(s.sediment.geometry['feed']))
        if bool((s.euler_flag!=0).item()) or not bool(valid.item()):
            raise RecursiveStepRejected('Invalid global recursive adaptation contributor')

    def _merge_scoped(self,status,mask,kind='face'):
        if status is None:return
        if isinstance(status,(list,tuple)):
            for item in status:self._merge_scoped(item,mask,kind)
            return
        scratch=self.scoped[kind];n=mask.size;components=status.size//n
        require(status.size==scratch.size and n*components==status.size,'Invalid scoped status size')
        self._launch(self.k['ale_scoped_status'],status.size,(mask,status,np.int32(n),
            np.int32(components),scratch))
        self.solver._merge(scratch)

    def _merge_rotation(self,status,mask):
        if status is not None:
            self._merge_scoped(status,mask,'cell' if status.size==self.nc else 'graincell')

    def rebalance(self,layers,level):
        """Reuse the unchanged independent-column law; publish selected only.

        The legacy kernel has no cell mask. It runs on PRIVATE scratch, then
        only current-level columns are copied back. Off-level physical arrays
        are never mutated, even transiently. Sparse work is a later optimization.
        """
        cp=self.cp;m=self._level(level)
        require(isinstance(layers,cp.ndarray) and layers.shape==self.rebalance_work.shape and
            layers.dtype==cp.float64 and layers.flags.c_contiguous and
            not (layers.data.ptr < self.rebalance_work.data.ptr+self.rebalance_work.nbytes and
                 self.rebalance_work.data.ptr < layers.data.ptr+layers.nbytes),
            'Invalid or overlapping level-rebalance input')
        cp.copyto(self.rebalance_work,layers)
        status=self.solver.sediment.rebalance(self.rebalance_work)
        self._launch(self.k['ale_scoped_status'],self.nc,(m['cells'],status,np.int32(self.nc),
            np.int32(1),self.rebalance_status))
        self._launch(self.k['ale_copy_level'],layers.size,(m['cells'],self.rebalance_work,
            np.int64(layers.size),np.int32(self.nc),layers))
        return self.rebalance_status

    def _limit_and_gather(self,state,dt,level,raw_bed):
        """Post-face adapter seam, also used by analytic donor-limit tests."""
        s=self.solver;cp=self.cp;m=self._level(level)
        a,b=s.flux.inputs,s.flux.buffers;ea,eb=s.exner.geometry,s.exner.buffers
        nc,nf,ng=np.int64(self.nc),np.int64(self.nf),np.int32(self.ng)
        dt=np.float64(dt);g=np.float64(s.data['grav']);dry=np.float64(s.data['dry'])
        self._launch(self.k['ale_mask_flux'],self.nf,(m['faces'],raw_bed,np.int32(self.nf),ng,
            b['fl'],b['fr'],b['speed'],self.raw_bed))
        self._launch(self.k['ale_active'],self.nc,(m['cells'],a['active'],np.int32(self.nc),self.bed_active))
        cp.copyto(b['raw_fl'],b['fl']);cp.copyto(b['raw_fr'],b['fr'])
        self._launch(s.flux.kernels['resident_draining'],self.nc,(state['q'],a['area'],a['length'],
            b['fl'],a['offset'],a['faces'],a['signs'],self.all_cells,nc,nf,dt,b['theta']))
        self._launch(s.flux.kernels['resident_limit_flux'],self.nf,(state['q'],b['theta'],a['owner'],
            a['neighbour'],self.kind,nc,nf,dry,b['fl'],b['fr'],b['face_status']))
        cp.max(state['q'][0],out=b['maxdepth'])
        self._launch(s.flux.kernels['resident_gather_update'],self.nc,(state['q'],a['area'],a['length'],
            a['normal'],b['fl'],b['fr'],b['speed'],b['maxdepth'],a['offset'],a['faces'],a['signs'],
            m['cells'],nc,nf,dt,g,np.float64(s.data['cfl_target']),b['next'],b['cfl_ratio'],b['cell_status']))
        self._launch(s.exner.k['bed_scale'],self.nf*self.ng,(self.raw_bed,ea['owner'],ea['neighbour'],
            ea['bid'],a['active'],nf,ng,np.float64(s.data['morfac']),eb['flux'],eb['face_status']))
        self._launch(s.exner.k['bed_donors'],self.nc*self.ng,(state['layers'][0],ea['area'],ea['length'],
            ea['offset'],ea['faces'],ea['signs'],self.all_cells,eb['flux'],nc,nf,ng,dt,eb['theta']))
        self._launch(s.exner.k['bed_limit'],self.nf*self.ng,(ea['owner'],ea['neighbour'],eb['theta'],nc,nf,ng,eb['flux']))
        self._launch(s.exner.k['bed_gather'],self.nc,(state['layers'][0],ea['area'],ea['length'],ea['offset'],
            ea['faces'],ea['signs'],self.bed_active,eb['flux'],ea['rock'],ea['bulk'],eb['face_status'],nc,nf,ng,dt,
            eb['next'],eb['bed'],eb['status']))
        if s.nb:
            s.exner.k['bed_ledger']((s.nb,self.ng),(128,),(eb['flux'],ea['length'],ea['boundary_offset'],
                ea['boundary_faces'],nf,ng,np.int32(s.nb),dt,eb['ledger']))
        cp.copyto(s.euler_layers,state['layers']);cp.copyto(s.euler_layers[0],eb['next'])
        rebalanced=self.rebalance(s.euler_layers,level)
        s._merge((b['cell_status'],eb['status'],eb['face_status'],rebalanced))
        self._merge_scoped(b['face_status'],m['faces'])
        water=s.boundary.integrate_ledger(b['fl'],dt)
        secondary=(self.secondary.advect(state['q'],b['next'],state['secondary'],b['fl'],float(dt),level)
                   if self.secondary is not None else None)
        if secondary is not None:s._merge(secondary['status'])
        return dict(state=dict(q=b['next'],layers=s.euler_layers,
                              secondary=state['secondary'] if secondary is None else secondary['next']),
            status=s.euler_flag,cfl=b['cfl_ratio'],
            ledger=dict(water=water,bed=eb['ledger'],
                        secondary=self.zero_secondary_ledger if secondary is None else secondary['ledger']))

    def evaluate(self,state,dt,current,level):
        s=self.solver;cp=self.cp;m=self._level(level)
        clock=s._distributed_external_clock
        tolerance=64*np.finfo(float).eps*max(1.,abs(float(current)))
        require(s._distributed_external_phase=='open' and clock is not None and
            np.isfinite([dt,current]).all() and 0<dt<=clock[1] and
            clock[0]-tolerance<=current<=clock[0]+clock[1]+tolerance,
            'Level-local Euler is outside its open macro clock')
        require(set(state)==set(s.state) and all(isinstance(value,cp.ndarray) and
            value.shape==s.state[key].shape and value.dtype==cp.float64 and value.flags.c_contiguous
            for key,value in state.items()),'Invalid resident level-local Euler state')
        s.euler_flag.fill(0)
        if s.external_boundary_schedule:
            require(s.recursive_boundary_forcing is not None,
                    'Missing recursive boundary stage controller')
            s.recursive_boundary_forcing.apply(current,dt,level)
        bed=s.bed(state['layers']);q=state['q']
        bedform_status=None
        if s.bedform is not None:
            bedform_result=s.bedform.evaluate(q,state['layers'])
            bedform_status=bedform_result['status']
            cp.copyto(s.sediment.geometry['manning_cell'],
                      bedform_result['properties'][0])
        r=s.reconstruction.reconstruct(q)
        if self.adaptation:
            # The adaptation operator can propagate an upstream contribution
            # beyond this level's reconstruction halo in the same steady solve.
            s._merge((r['primitive_status'],s.sediment.buffers['bed_status']))
        # Equilibrium steering must be computed BEFORE sediment.prepare rotates
        # capacity, otherwise that branch uses a previous stage's curvature.
        if self.secondary is not None:
            secondary=self.secondary.reconstruct(q,bed,state['secondary'],level)
            s._merge(secondary['status'])
        coupled=bool(s.sediment.coupled_mpm)
        capacity_rotation=s.sediment.prepare(q,bed,state['layers'],state['secondary'],r['active']) if coupled else None
        s.boundary.apply(q,r['trace'])
        if s.external_boundary_schedule:s.recursive_boundary_forcing.record_stage()
        stage=s.sediment.stage_characteristic(q,bed,r['face_value'],r['trace']) if coupled else None
        self._launch(self.k['ale_kinds'],self.nf,(m['faces'],r['kind'],np.int32(self.nf),self.kind))
        a,b=s.flux.inputs,s.flux.buffers
        s.flux.inputs['q']=q;s.flux.controls['dt']=np.float64(dt)
        nc,nf=np.int64(self.nc),np.int64(self.nf)
        g,dry=(np.float64(s.data[k]) for k in ('grav','dry'))
        self._launch(s.flux.kernels['resident_prepare_faces'],self.nf,(r['trace'],q,bed,a['normal'],a['distance'],
            a['owner'],a['neighbour'],self.kind,nc,nf,g,dry,b['local'],b['closed']))
        self._launch(s.flux.kernels['augmented_fwave_local'],self.nf,(b['local'],nf,g,dry,b['riemann'],b['augmented_status']))
        self._launch(s.flux.kernels['resident_assemble_flux'],self.nf,(r['trace'],b['local'],b['riemann'],q,bed,
            a['normal'],a['owner'],a['neighbour'],self.kind,a['boundary_mode'],a['sampled'],a['bedoffset'],
            r['hydro_constant'],b['closed'],nc,nf,g,dry,b['fl'],b['fr'],b['speed'],b['face_status']))
        sediment=s.sediment.fluxes(q,bed,state['layers'],state['secondary'],
            dict(trace=r['trace'],raw_fl=b['fl'],closed=b['closed'],active=r['active']),prepared=coupled)
        if coupled:
            self._launch(self.k['ale_coupled_kinds'],self.nf,(self.kind,a['owner'],a['neighbour'],
                s.sediment.geometry['open_boundary_distance'],b['local'],np.int32(self.nf),g,self.coupled_kind))
        coupled_status=s.sediment.coupled_faces(dict(local=b['local'],riemann=b['riemann'],
            kind=self.coupled_kind,closed=b['closed']),b) if coupled else None
        if s.external_boundary_schedule:s.recursive_boundary_forcing.integrated_flux(b,level)
        if self.secondary is not None:
            stress=s.secondary.add_stress(r['trace'],self.kind,b['closed'],b['fl'],b['fr'],b['speed'])
            s._merge(stress['status'])
        self._merge_scoped(r['primitive_status'],m['halo'],'cell')
        self._merge_scoped(bedform_status,m['halo'],'cell')
        self._merge_scoped(r['reconstruction_status'],m['faces'])
        self._merge_scoped(b['augmented_status'],m['faces'])
        self._merge_scoped(s.boundary.face_status,m['faces'])
        # The recursive controller owns stage times and current-bed section roots.
        s._merge(s.boundary.status)
        self._merge_scoped(stage,m['faces']);self._merge_scoped(coupled_status,m['faces'])
        self._merge_scoped(s.sediment.buffers['cell_status'],m['halo'],'cell')
        self._merge_scoped(s.sediment.buffers['face_status'],m['faces'])
        # fluxes()[2] is the rotation status; the prepared model-0 branch
        # returns an unrotated placeholder, so keep its earlier result as well.
        self._merge_rotation(sediment['status'][2],m['halo'])
        self._merge_rotation(capacity_rotation,m['halo'])
        if self.adaptation:s._merge(s.sediment.buffers['adaptation_status'])
        if s.sediment.stratigraphy is not None:s._merge(s.sediment.buffers['bed_status'])
        result=self._limit_and_gather(state,dt,level,sediment['raw'])
        s.euler_calls+=1
        return result
