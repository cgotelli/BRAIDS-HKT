"""Isolated conservative GPU repair after recursive water/bed reflux.

This is not a wet-front integrator or an exchange/suspension implementation.
Inputs are never modified. All numerical state, allocations and diagnostics
remain resident. Independent repair requests run concurrently; requests sharing
a donor follow ascending coarse-cell order. The root global-water seam has a
whole-domain dependency and deliberately retains native sequential GPU order.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ABI = 2
LEGACY_SOURCE_SHA256 = frozenset({
    'd1669971affca240c02b4b7e21473c844b8c5ed1e0426333d5a06395e4cd412a',
})
OPTIONS = ('--std=c++11', '--fmad=false', '--prec-div=true',
           '--prec-sqrt=true', '--ftz=false')
DIAGNOSTIC_COLUMNS = ('repaired_cells', 'reclaimed_inventory',
                      'transferred_x_momentum', 'transferred_y_momentum')
STATUS = dict(nonfinite=1, suspended=2, protected_negative=4,
              wrong_level_negative=8, insufficient_water=16,
              insufficient_bed=32, arithmetic=64, allocation_residual=128,
              budget=256)


def _require(value, message):
    if not value:
        raise ValueError(message)


def repair_source(source_path=None):
    """Authenticate the CUDA call signature before any device allocation."""
    path=Path(source_path or HERE/'amr_bank_repair.cu').resolve()
    raw=path.read_bytes();source=raw.decode('utf-8');digest=hashlib.sha256(raw).hexdigest()
    markers=[line.strip() for line in source.splitlines()
             if line.strip().startswith('#define BRAIDS_BANK_REFLUX_REPAIR_ABI')]
    if markers==['#define BRAIDS_BANK_REFLUX_REPAIR_ABI 2']:
        abi=2;behavior='conservative_water_signed_secondary_and_active_bed'
    elif (source_path is not None and
          markers==['#define BRAIDS_BANK_REFLUX_REPAIR_ABI 1'] and
          digest in LEGACY_SOURCE_SHA256):
        abi=1;behavior='authenticated_legacy_without_signed_secondary'
    else:
        raise ValueError('Bank reflux repair CUDA ABI or legacy source identity differs')
    return source,dict(path=str(path),sha256=digest,abi=abi,behavior=behavior)


def _indices(value, name):
    array=np.asarray(value)
    _require(array.ndim==1 and (array.size==0 or array.dtype.kind in 'iu') and
             np.all(array>=np.iinfo(np.int32).min) and
             np.all(array<=np.iinfo(np.int32).max),
             name+' must be a one-dimensional int32-range integer array')
    return np.array(array,dtype=np.int32,copy=True,order='C')


def repair_topology(area, levels, offset, faces, signs, owner, neighbour):
    """Host immutable graph planning, never numerical-state inspection.

    Repeated *neighbors* from distinct faces are valid and deduplicated. Repeated
    occurrences of the same directed face are corrupt and rejected. Water uses
    ascending donor IDs; bed uses first occurrence in original directed CSR.
    """
    area=np.ascontiguousarray(area,dtype=np.float64)
    levels=_indices(levels,'levels');offset=_indices(offset,'offset')
    faces=_indices(faces,'faces');signs=_indices(signs,'signs')
    owner=_indices(owner,'owner');neighbour=_indices(neighbour,'neighbour')
    nc=len(levels);nf=len(owner)
    _require(nc>0 and area.shape==(nc,) and np.isfinite(area).all() and
             np.all(area>0),'Finite positive cell areas required')
    _require(np.all((levels>=0)&(levels<=30)),'Invalid AMR levels')
    _require(neighbour.shape==(nf,) and np.all((owner>=0)&(owner<nc)) and
             np.all((neighbour>=-1)&(neighbour<nc)) and
             np.all(owner!=neighbour),'Invalid owner/neighbour graph')
    internal=neighbour>=0
    _require(np.all(np.abs(levels[owner[internal]]-levels[neighbour[internal]])<=1),
             'Repair requires a balanced 2:1 leaf graph')
    _require(offset.shape==(nc+1,) and offset[0]==0 and offset[-1]==len(faces)
             and np.all(np.diff(offset)>=0) and signs.shape==faces.shape and
             np.all((faces>=0)&(faces<nf)) and np.all(np.abs(signs)==1),
             'Invalid directed face CSR')
    seen=np.zeros((nf,2),np.int32);all_neighbors=[]
    for c in range(nc):
        row=[];seen_neighbor=set()
        for k in range(int(offset[c]),int(offset[c+1])):
            f=int(faces[k]);side=0 if signs[k]>0 else 1
            _require((owner[f] if side==0 else neighbour[f])==c,
                     'Directed face does not belong to its CSR cell')
            seen[f,side]+=1
            _require(seen[f,side]==1,'Duplicate directed face in repair topology')
            n=int(neighbour[f] if side==0 else owner[f])
            if n>=0 and n not in seen_neighbor:
                row.append(n);seen_neighbor.add(n)
        all_neighbors.append(row)
    _require(np.all(seen[:,0]==1) and np.array_equal(seen[:,1],internal.astype(np.int32)),
             'Incomplete directed face CSR')

    def csr(rows):
        return (np.asarray([0]+list(np.cumsum([len(row) for row in rows])),np.int32),
                np.asarray([n for row in rows for n in row],np.int32))

    bed_neighbors=[[n for n in row if levels[n]==levels[c]+1]
                   for c,row in enumerate(all_neighbors)]
    water_neighbors=[sorted(row) for row in bed_neighbors]
    waves={}
    for level in sorted(set(levels.tolist())):
        # Dependency depth is an ordering, not graph coloring: a later cell
        # cannot leapfrog an earlier request on a shared donor.
        last={};groups=[]
        for c in np.flatnonzero(levels==level).tolist():
            touched=[c]+bed_neighbors[c]
            wave=1+max((last.get(n,-1) for n in touched),default=-1)
            while len(groups)<=wave:groups.append([])
            groups[wave].append(c)
            for n in touched:last[n]=wave
        waves[level]=tuple(np.asarray(row,np.int32) for row in groups)
    return dict(area=area,levels=levels,water=csr(water_neighbors),
                bed=csr(bed_neighbors),root=csr([sorted(row) for row in all_neighbors]),
                waves=waves)


class ResidentBankRefluxRepair:
    """Private proposal on row-major ``state[cell,state]`` native inventories.

    ``suspended_indices`` is mandatory, even when empty: the caller explicitly
    identifies transported suspended rows, which must all be exactly zero.
    All rows except h/hu/hv, the explicitly declared signed secondary row and
    active-bed rows are protected nonnegative stock. Water repair redistributes
    the signed secondary integral with the repaired cell's momentum. Bed repair
    never changes it. The admitted native schema puts secondary last, after all
    substrate columns. Dry-cell secondary cleanup belongs to the caller.

    ``propose(state, level)`` repairs water then bed only at that level.
    ``propose(state, -1, root_water=True)`` does water only, using adjacent
    positive stock first and all-domain stock only if that is insufficient.
    A nonzero device status invalidates the entire private proposal. There is
    no clipping, active-layer rebalance, momentum dry cleanup or auto-commit.
    Returned arrays belong to this object and are overwritten by its next call.
    """
    def __init__(self, area, levels, offset, faces, signs, owner, neighbour, *,
                 active_bed_indices, suspended_indices, secondary_index=-1,
                 source_path=None):
        topology=repair_topology(area,levels,offset,faces,signs,owner,neighbour)
        active=_indices(active_bed_indices,'active_bed_indices')
        suspended=_indices(suspended_indices,'suspended_indices')
        _require(len(active)>0 and np.all(active>=3) and np.all(suspended>=3)
                 and len(set(active.tolist()))==len(active)
                 and len(set(suspended.tolist()))==len(suspended)
                 and not set(active.tolist()).intersection(suspended.tolist()),
                 'Distinct active-bed and suspended state columns required')
        _require(isinstance(secondary_index,(int,np.integer)) and
                 not isinstance(secondary_index,(bool,np.bool_)) and
                 (secondary_index==-1 or 3<=secondary_index<=np.iinfo(np.int32).max),
                 'Secondary index must be -1 or an integer native state column')
        secondary_index=int(secondary_index)
        _require(secondary_index not in active and secondary_index not in suspended,
                 'Secondary column overlaps an active-bed or suspended column')
        source,self.source_receipt=repair_source(source_path)
        self.source_abi=self.source_receipt['abi']
        _require(self.source_abi>=2 or secondary_index==-1,
                 'Legacy repair CUDA cannot transport signed secondary state')
        self.secondary_index=secondary_index
        self.source_path=Path(self.source_receipt['path'])
        import cupy as cp
        self.cp=cp;self.device=int(cp.cuda.Device().id)
        self.nc=len(topology['area']);self.ng=len(active)
        self.active_host=active;self.suspended_host=suspended
        self.area=cp.asarray(topology['area']);self.levels=cp.asarray(topology['levels'])
        self.active=cp.asarray(active);self.suspended=cp.asarray(suspended)
        self.donor={name:tuple(cp.asarray(a) for a in topology[name])
                    for name in ('water','bed','root')}
        self.waves={level:tuple(cp.asarray(row) for row in rows)
                    for level,rows in topology['waves'].items()}
        self.wave_sizes={level:tuple(len(row) for row in rows)
                         for level,rows in topology['waves'].items()}
        self.status=cp.zeros((),cp.int32);self.gate=cp.zeros((),cp.int32)
        self.cell_status=cp.zeros(self.nc,cp.int32)
        self.cell_diagnostics=cp.zeros((self.nc,self.ng+1,4),cp.float64)
        self.failure_details=cp.zeros((self.nc,self.ng+1,3),cp.float64)
        self.diagnostics=cp.zeros((self.ng+1,4),cp.float64)
        self.trial=None;self.states=None;self.conservation_residual=None
        self.conservation_scale=None;self.dry_momentum_cells=cp.zeros((),cp.int64)
        names=('br_validate_input','br_water_wave','br_bed_wave','br_root_water',
               'br_validate_result')
        module=cp.RawModule(code=source,options=OPTIONS,name_expressions=names)
        self.kernels={name:module.get_function(name) for name in names}

    def reserve(self,nstate):
        """Eagerly create every persistent output before an outer snapshot.

        This method never changes an existing schema or replaces allocated
        buffers. A repeated call for the same schema is a no-op, including its
        contents, so an outer rollback inventory remains stable.
        """
        _require(isinstance(nstate,(int,np.integer)) and
                 not isinstance(nstate,(bool,np.bool_)) and
                 4<=nstate<=np.iinfo(np.int32).max,'Integer native state width required')
        nstate=int(nstate)
        _require(np.all(self.active_host<nstate) and np.all(self.suspended_host<nstate)
                 and self.secondary_index<nstate,
                 'Repair column outside state')
        _require(self.secondary_index==-1 or self.secondary_index==nstate-1,
                 'Secondary column must be the final native state column')
        _require(self.states is None or self.states==nstate,'Repair state schema changed')
        if self.trial is not None:return
        cp=self.cp
        self.trial=cp.zeros((self.nc,nstate),cp.float64)
        self.conservation_residual=cp.zeros(nstate,cp.float64)
        self.conservation_scale=cp.zeros(nstate,cp.float64)
        self.states=nstate

    def _arrays(self):
        values=[self.area,self.levels,self.active,self.suspended,self.status,self.gate,
                self.cell_status,self.cell_diagnostics,self.failure_details,self.diagnostics,
                self.dry_momentum_cells,self.trial,self.conservation_residual,self.conservation_scale]
        values.extend(a for pair in self.donor.values() for a in pair)
        values.extend(a for rows in self.waves.values() for a in rows)
        return [a for a in values if a is not None]

    def _guard(self,state):
        cp=self.cp
        _require(isinstance(state,cp.ndarray) and state.dtype==cp.float64 and
                 state.ndim==2 and state.shape[0]==self.nc and state.shape[1]>=4 and
                 state.flags.c_contiguous and int(state.device.id)==self.device,
                 'Resident contiguous FP64 cell/state input on the repair device required')
        ns=state.shape[1]
        _require(np.all(self.active_host<ns) and np.all(self.suspended_host<ns)
                 and self.secondary_index<ns,
                 'Repair column outside state')
        _require(self.secondary_index==-1 or self.secondary_index==ns-1,
                 'Secondary column must be the final native state column')
        _require(self.states is None or self.states==ns,'Repair state schema changed')
        start=int(state.data.ptr);end=start+state.nbytes
        for array in self._arrays():
            a=int(array.data.ptr);b=a+array.nbytes
            _require(not (start<b and a<end),'Input aliases private repair storage')

    def _launch(self,name,size,args):
        if size:self.kernels[name](((size+127)//128,),(128,),args)

    def propose(self,state,level,*,root_water=False):
        self._guard(state)
        _require(isinstance(level,(int,np.integer)) and not isinstance(level,(bool,np.bool_)),
                 'Integer synchronized repair level required')
        _require(isinstance(root_water,(bool,np.bool_)) and
                 ((root_water and level==-1) or (not root_water and level in self.waves)),
                 'Root water repair must be explicit; otherwise use an existing AMR level')
        cp=self.cp;ns=state.shape[1]
        self.reserve(ns)
        cp.copyto(self.trial,state)
        for array in (self.status,self.gate,self.cell_status,self.cell_diagnostics,
                      self.failure_details,self.diagnostics,self.dry_momentum_cells):array.fill(0)
        ints=(np.int32(self.nc),np.int32(ns),np.int32(self.ng))
        secondary=(np.int32(self.secondary_index),) if self.source_abi>=2 else ()
        self._launch('br_validate_input',self.nc,(state,self.area,self.levels,self.active,
            self.suspended,*ints,np.int32(len(self.suspended_host)),np.int32(level),
            np.int32(root_water),*secondary,self.cell_status,self.status))
        if root_water:
            cp.copyto(self.gate,self.status)
            offsets,donors=self.donor['root']
            self.kernels['br_root_water']((1,),(1,),(self.trial,self.area,offsets,donors,
                *ints,*secondary,self.gate,self.cell_status,self.status,self.cell_diagnostics,self.failure_details))
        else:
            for name,kind in (('br_water_wave','water'),('br_bed_wave','bed')):
                offsets,donors=self.donor[kind]
                for cells in self.waves[int(level)]:
                    cp.copyto(self.gate,self.status)
                    args=(self.trial,self.area,offsets,donors,cells,self.active,
                          *ints,np.int32(len(cells)),*(secondary if kind=='water' else ()),
                          self.gate,self.cell_status,self.status,
                          self.cell_diagnostics,self.failure_details)
                    self._launch(name,len(cells)*(self.ng if kind=='bed' else 1),args)
        self._launch('br_validate_result',self.nc,(self.trial,self.suspended,*ints,
            np.int32(len(self.suspended_host)),*secondary,self.cell_status,self.status))
        # These sums have no donor-order dependency. Use parallel device
        # reductions, not one serial full-domain loop per physical row.
        cp.copyto(self.conservation_residual,cp.sum((self.trial-state)*self.area[:,None],axis=0))
        cp.copyto(self.conservation_scale,cp.maximum(1.,cp.maximum(
            cp.sum(cp.abs(state*self.area[:,None]),axis=0),
            cp.sum(cp.abs(self.trial*self.area[:,None]),axis=0))))
        invalid=(~cp.isfinite(self.conservation_residual)|~cp.isfinite(self.conservation_scale)|
                 (cp.abs(self.conservation_residual)>512*np.finfo(np.float64).eps*self.conservation_scale))
        self.status[...]|=cp.any(invalid).astype(cp.int32)*np.int32(STATUS['budget'])
        cp.copyto(self.diagnostics,cp.sum(self.cell_diagnostics,axis=0))
        cp.copyto(self.dry_momentum_cells,cp.count_nonzero((self.trial[:,0]==0)&
            ((self.trial[:,1]!=0)|(self.trial[:,2]!=0))))
        return dict(state=self.trial,status=self.status,cell_status=self.cell_status,
                    diagnostics=self.diagnostics,failure_details=self.failure_details,
                    conservation_residual=self.conservation_residual,
                    conservation_scale=self.conservation_scale,
                    dry_momentum_cells=self.dry_momentum_cells)
