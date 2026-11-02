"""Resident physical-toe undercut history and discrete noncohesive collapse.

ABI3 intentionally repairs the old native mode-2 geometric-history semantics.
It consumes ACTUAL toe mass, pre-step constitutive geometry, and CURRENT trial
depth/bed for exposed-bank lifetime and bounded simultaneous collapse. Counts
are actual positive collapse transactions, not requested threshold quanta.
It does not implement persistent subcell-front/event-time tracking.

All changes are private trial arrays. The enclosing solver must combine this
with the toe-supply transaction, active-layer rebalance, global rejection and
checkpointing before it is a supported coupled bank model.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .surface_transfer import validate_geometry

HERE=Path(__file__).resolve().parent
OPTIONS=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true','--ftz=false')
FLOAT_HISTORY=('undercut','reference_height','toe_height','reference_length',
               'toe_cumulative','collapse_cumulative')


class ResidentPhysicalUndercut:
    def __init__(self,area,owner,neighbour,offset,faces,signs,length,bulk,
                 is_toe_face,candidate_bank,candidate_toe,is_toe_donor,
                 physical_toe_cell,mapped_toe_donor,toe_distance,bank_erodible,
                 nlayer,*,dry_tolerance,critical_undercut,source_path=None):
        import cupy as cp
        topology,area=validate_geometry(area,owner,neighbour,offset,faces,signs)
        owner,neighbour,_,_,_=topology
        nc=len(area);nf=len(owner)
        length=np.asarray(length,dtype=np.float64);bulk=np.asarray(bulk,dtype=np.float64)
        if (length.shape!=(nf,) or not np.isfinite(length).all() or np.any(length<=0) or
            bulk.ndim!=1 or not 1<=len(bulk)<=32 or not np.isfinite(bulk).all() or np.any(bulk<=0) or
            not isinstance(nlayer,(int,np.integer)) or not 1<=nlayer<=33):
            raise ValueError('Invalid undercut geometry, grain density or layer count')
        if not np.isfinite([dry_tolerance,critical_undercut]).all() or \
                dry_tolerance<0 or critical_undercut<=0:
            raise ValueError('Invalid dry/critical undercut controls')
        toe_face=np.asarray(is_toe_face);bank=np.asarray(candidate_bank);toe=np.asarray(candidate_toe)
        donor=np.asarray(is_toe_donor);mapped_cell=np.asarray(physical_toe_cell)
        mapped_donor=np.asarray(mapped_toe_donor);distance=np.asarray(toe_distance,dtype=np.float64)
        erodible=np.asarray(bank_erodible)
        if (toe_face.shape!=(nf,) or bank.shape!=(nf,) or toe.shape!=(nf,) or
            donor.shape!=(nc,) or mapped_cell.shape!=(nc,) or mapped_donor.shape!=(nc,) or
            distance.shape!=(nc,) or erodible.shape!=(nc,) or
            any(v.dtype.kind not in 'biu' for v in (toe_face,donor,erodible)) or
            any(v.dtype.kind not in 'iu' for v in (bank,toe,mapped_cell,mapped_donor)) or
            np.any((toe_face!=0)&(toe_face!=1)) or np.any((donor!=0)&(donor!=1)) or
            np.any((erodible!=0)&(erodible!=1)) or not np.isfinite(distance).all() or
            np.any(distance<0)):
            raise ValueError('Invalid physical toe mapping')
        bank=bank.astype(np.int64);toe=toe.astype(np.int64)
        mapped_cell=mapped_cell.astype(np.int64);mapped_donor=mapped_donor.astype(np.int64)
        donor_face_groups=[[] for _ in range(nc)]
        for f in range(nf):
            if not toe_face[f]:
                if bank[f]!=-1 or toe[f]!=-1:
                    raise ValueError('Non-toe face has physical mapping')
                continue
            b=int(bank[f]);t=int(toe[f])
            if (neighbour[f]<0 or b<0 or b>=nc or t<0 or t>=nc or b==t or
                not donor[b] or not erodible[b] or
                not ((owner[f]==b and neighbour[f]==t) or
                     (owner[f]==t and neighbour[f]==b))):
                raise ValueError('Invalid physical toe donor/recipient face')
            donor_face_groups[b].append(f)
        candidate_groups=[[] for _ in range(nc)];toe_donor_groups=[[] for _ in range(nc)]
        for c in range(nc):
            if donor[c]:
                t=int(mapped_cell[c])
                if t<0 or t>=nc or t==c or not donor_face_groups[c]:
                    raise ValueError('Physical toe donor lacks mapped toe/face')
                toe_donor_groups[t].append(c)
            if erodible[c]:
                d=int(mapped_donor[c])
                if d<0 or d>=nc or not donor[d]:
                    raise ValueError('Erodible bank cell lacks physical donor')
                candidate_groups[d].append(c)
        # Native collapse processes donors in cell order and deposits each block
        # immediately. If a toe recipient is itself an erodible candidate, a
        # later donor may excavate that newly deposited block. The parallel
        # trial is equivalent only for dependency-disjoint donor/recipient sets.
        # A chained map needs an ordered GPU wavefront, not a silent shortcut.
        if any(erodible[int(mapped_cell[c])] for c in range(nc) if donor[c]):
            raise ValueError('Chained physical toe/collapse mapping requires ordered wavefront')
        for group in candidate_groups:group.sort(key=lambda c:(distance[c],c))
        def flatten(groups):
            counts=np.array([len(v) for v in groups],dtype=np.int32)
            return np.r_[0,np.cumsum(counts)].astype(np.int32),np.array(
                [item for row in groups for item in row],dtype=np.int32)
        donor_face_offset,donor_faces=flatten(donor_face_groups)
        candidate_offset,candidates=flatten(candidate_groups)
        toe_donor_offset,toe_donors=flatten(toe_donor_groups)
        source=Path(source_path or HERE/'bank_undercut.cu').read_text()
        if '#define BRAIDS_UNDERCUT_INTERFACE_ABI 3' not in source:
            raise ValueError('Undercut Python/CUDA ABI mismatch: current-geometry ABI3 required')
        self.cp=cp;self.nc=nc;self.nf=nf;self.ng=len(bulk);self.nl=int(nlayer)
        self.dry=np.float64(dry_tolerance);self.critical=np.float64(critical_undercut)
        self.area=cp.asarray(area);self.bulk=cp.asarray(np.ascontiguousarray(bulk))
        self.length=cp.asarray(np.ascontiguousarray(length))
        self.candidate_toe=cp.asarray(np.ascontiguousarray(toe,dtype=np.int32))
        self.is_donor=cp.asarray(np.ascontiguousarray(donor,dtype=np.int32))
        self.toe_cell=cp.asarray(np.ascontiguousarray(mapped_cell,dtype=np.int32))
        self.donor_face_offset=cp.asarray(donor_face_offset);self.donor_faces=cp.asarray(donor_faces)
        self.candidate_offset=cp.asarray(candidate_offset);self.candidates=cp.asarray(candidates)
        self.toe_donor_offset=cp.asarray(toe_donor_offset);self.toe_donors=cp.asarray(toe_donors)
        names=('bu_current_bed','bu_update_history','bu_propose','bu_limit_shared_toe',
               'bu_collapse','bu_deposit','bu_budget','bu_reconcile','bu_validate')
        module=cp.RawModule(code=source,
                            options=OPTIONS,name_expressions=names)
        self.k={name:module.get_function(name) for name in names}
        self.trial_layers=cp.empty((self.nl,self.ng,nc),cp.float64)
        self.trial_history={name:cp.empty(nc,cp.float64) for name in FLOAT_HISTORY}
        self.trial_history['collapse_count']=cp.empty(nc,cp.int32)
        self.removed_by_layer=cp.empty_like(self.trial_layers)
        self.removed_mass=cp.empty((self.ng,nc),cp.float64)
        self.collapse_cell=cp.empty(nc,cp.int32);self.collapse_target=cp.empty(nc,cp.int32)
        self.exposed_cell=cp.empty(nc,cp.int32)
        self.post_toe_bed=cp.empty(nc,cp.float64);self.post_collapse_bed=cp.empty(nc,cp.float64)
        self.proposal=cp.empty(nc,cp.float64);self.above_height=cp.empty(nc,cp.float64)
        self.toe_scale=cp.empty(nc,cp.float64);self.actual_volume=cp.empty(nc,cp.float64)
        self.recipient_gain=cp.empty((self.ng,nc),cp.float64)
        self.grain_budget=cp.empty((self.ng,2),cp.float64)
        self.failure=cp.zeros((),cp.int32);self._tmp=cp.zeros_like(self.failure)
        self.status={name:cp.empty(size,cp.int32) for name,size in
                     [('input_validate',nc),('post_toe_bed',nc),('history',nc),
                      ('proposal',nc),('shared_toe',nc),('collapse',nc),('deposit',nc),
                      ('budget',self.ng),('post_collapse_bed',nc),('reconcile',nc),
                      ('validate',nc)]}

    def _array(self,value,shape,dtype):
        if (not isinstance(value,self.cp.ndarray) or value.shape!=shape or
            value.dtype!=dtype or not value.flags.c_contiguous):
            raise ValueError('Expected contiguous device '+str(dtype)+' array '+str(shape))

    @staticmethod
    def _span(value):
        start=int(value.data.ptr)
        return start,start+int(value.nbytes)

    @staticmethod
    def _overlap(a,b):
        return a[0]<b[1] and b[0]<a[1]

    def _mutable_storage(self):
        arrays={name:getattr(self,name) for name in (
            'trial_layers','removed_by_layer','removed_mass','collapse_cell',
            'collapse_target','exposed_cell','post_toe_bed','post_collapse_bed',
            'proposal','above_height','toe_scale','actual_volume','recipient_gain',
            'grain_budget','failure','_tmp')}
        arrays.update(('history.'+k,v) for k,v in self.trial_history.items())
        arrays.update(('status.'+k,v) for k,v in self.status.items())
        if any(not isinstance(v,self.cp.ndarray) or not v.flags.c_contiguous
               for v in arrays.values()):
            raise ValueError('Private undercut storage must be contiguous device arrays')
        spans=sorted((self._span(v),name) for name,v in arrays.items() if v.nbytes)
        for (left,lname),(right,rname) in zip(spans,spans[1:]):
            if self._overlap(left,right):
                raise ValueError('Private undercut arrays overlap: '+lname+' / '+rname)
        return dict((name,span) for span,name in spans)

    def _guard_inputs(self,inputs,history=None,*,allow_private_layers=False):
        """Check byte ranges BEFORE any private copy, fill or launch.

        Pointer equality misses shifted contiguous views and cross-history
        aliases. All mutable output/scratch storage is protected, not just the
        primary state. Read-only physical inputs may alias each other.
        """
        mutable=self._mutable_storage()
        for name,value in inputs.items():
            span=self._span(value)
            for target,other in mutable.items():
                if (allow_private_layers and name=='layers' and
                    target=='trial_layers' and span==other):
                    continue
                if self._overlap(span,other):
                    raise ValueError('Undercut input overlaps private storage: '+name+' / '+target)
        if history is not None:
            seen=[]
            for name,value in history.items():
                span=self._span(value)
                for other_name,other in seen:
                    if self._overlap(span,other):
                        raise ValueError('Accepted histories overlap: '+name+' / '+other_name)
                for other_name,other in inputs.items():
                    if self._overlap(span,self._span(other)):
                        raise ValueError('Accepted history overlaps physical input: '+name+' / '+other_name)
                for other_name,other in mutable.items():
                    if self._overlap(span,other):
                        raise ValueError('Accepted history overlaps private storage: '+name+' / '+other_name)
                seen.append((name,span))

    def _launch(self,name,n,args):
        if n:self.k[name](((n+127)//128,),(128,),args)

    def _fold(self,names):
        cp=self.cp
        for name in names:
            cp.max(cp.where(self.status[name]!=0,cp.int32(1),cp.int32(0)),out=self._tmp)
            cp.maximum(self.failure,self._tmp,out=self.failure)

    def reconcile(self,layers,current_depth,reference_layers,reference_bed,history):
        """Retire vanished geometric history on PRIVATE trial state only.

        This is also called after avalanche. It never resets the status latch,
        modifies material, or clears cumulative ledgers/transaction counts.
        """
        cp=self.cp;nc=np.int32(self.nc);ng=np.int32(self.ng);nl=np.int32(self.nl)
        for value,shape in ((layers,(self.nl,self.ng,self.nc)),
                            (reference_layers,(self.nl,self.ng,self.nc)),
                            (current_depth,(self.nc,)),(reference_bed,(self.nc,))):
            self._array(value,shape,cp.float64)
        if set(history)!=set(self.trial_history):
            raise ValueError('Reconcile requires this operator private trial history')
        for name,trial in self.trial_history.items():
            self._array(history[name],(self.nc,),trial.dtype)
            if self._span(history[name])!=self._span(trial):
                raise ValueError('Reconcile requires this operator private trial history')
        self._guard_inputs(dict(layers=layers,current_depth=current_depth,
            reference_layers=reference_layers,reference_bed=reference_bed),
            allow_private_layers=True)
        self._launch('bu_current_bed',self.nc,(nc,ng,nl,self.bulk,reference_layers,
            layers,reference_bed,current_depth,self.post_collapse_bed,
            self.status['post_collapse_bed']))
        self._launch('bu_reconcile',self.nc,(nc,ng,nl,layers,self.post_collapse_bed,
            current_depth,self.toe_cell,self.is_donor,self.candidate_offset,
            self.candidates,self.dry,history['undercut'],history['reference_height'],
            history['toe_height'],history['reference_length'],self.status['reconcile']))
        self._fold(('post_collapse_bed','reconcile'))
        return self.failure

    def step(self,reference_layers,after_toe_layers,reference_depth,
             reference_bed,toe_removed_mass,accepted_history,*,current_depth):
        """Propose history/collapse on private state; caller owns global commit."""
        cp=self.cp;nc=np.int32(self.nc);ng=np.int32(self.ng);nl=np.int32(self.nl)
        for value,shape in ((reference_layers,(self.nl,self.ng,self.nc)),
                            (after_toe_layers,(self.nl,self.ng,self.nc)),
                            (reference_depth,(self.nc,)),(reference_bed,(self.nc,)),
                            (current_depth,(self.nc,)),
                            (toe_removed_mass,(self.ng,self.nc))):
            self._array(value,shape,cp.float64)
        if set(accepted_history)!=set(self.trial_history):
            raise ValueError('Complete physical undercut history required')
        for name,trial in self.trial_history.items():
            value=accepted_history[name];self._array(value,(self.nc,),trial.dtype)
        self._guard_inputs(dict(reference_layers=reference_layers,
            after_toe_layers=after_toe_layers,reference_depth=reference_depth,
            reference_bed=reference_bed,current_depth=current_depth,
            toe_removed_mass=toe_removed_mass),accepted_history)
        for name,trial in self.trial_history.items():cp.copyto(trial,accepted_history[name])
        cp.copyto(self.trial_layers,after_toe_layers)
        self.removed_by_layer.fill(0.);self.failure.fill(0)
        for item in self.status.values():item.fill(0)
        h=self.trial_history
        # Validate before retirement can clear an invalid accepted history.
        self._launch('bu_validate',self.nc,(nc,ng,nl,reference_layers,
            reference_depth,reference_bed,toe_removed_mass,after_toe_layers,
            h['undercut'],h['reference_height'],h['toe_height'],h['reference_length'],
            h['toe_cumulative'],h['collapse_cumulative'],h['collapse_count'],
            self.status['input_validate']))
        self._launch('bu_current_bed',self.nc,(nc,ng,nl,self.bulk,reference_layers,
            after_toe_layers,reference_bed,current_depth,self.post_toe_bed,
            self.status['post_toe_bed']))
        self._launch('bu_update_history',self.nc,(nc,ng,nl,self.area,self.bulk,
            reference_depth,reference_bed,toe_removed_mass,self.is_donor,
            self.donor_face_offset,self.donor_faces,self.candidate_toe,self.length,
            after_toe_layers,self.post_toe_bed,current_depth,self.toe_cell,
            self.candidate_offset,self.candidates,self.dry,self.exposed_cell,
            h['undercut'],h['reference_height'],h['toe_height'],h['reference_length'],
            h['toe_cumulative'],self.status['history']))
        self._launch('bu_propose',self.nc,(nc,ng,nl,self.area,self.bulk,current_depth,
            self.post_toe_bed,after_toe_layers,self.exposed_cell,self.toe_cell,
            self.is_donor,self.critical,h['undercut'],h['reference_length'],
            self.proposal,self.above_height,self.collapse_cell,self.collapse_target,
            self.status['proposal']))
        self._launch('bu_limit_shared_toe',self.nc,(nc,self.area,self.proposal,
            self.above_height,self.collapse_cell,self.collapse_target,
            self.toe_donor_offset,self.toe_donors,self.toe_scale,self.status['shared_toe']))
        self._launch('bu_collapse',self.nc,(nc,ng,nl,self.area,self.bulk,self.proposal,
            self.above_height,self.toe_scale,self.collapse_cell,self.collapse_target,
            self.trial_layers,h['undercut'],h['reference_length'],h['collapse_cumulative'],
            h['collapse_count'],self.removed_by_layer,self.removed_mass,
            self.actual_volume,self.status['collapse']))
        self._launch('bu_deposit',self.nc,(nc,ng,self.area,self.removed_mass,
            self.collapse_target,self.toe_donor_offset,self.toe_donors,
            self.trial_layers,self.recipient_gain,self.status['deposit']))
        self._launch('bu_budget',self.ng,(nc,ng,self.removed_mass,self.recipient_gain,
            self.grain_budget,self.status['budget']))
        self._fold(('input_validate','post_toe_bed','history','proposal','shared_toe',
                    'collapse','deposit','budget'))
        self.reconcile(self.trial_layers,current_depth,reference_layers,reference_bed,h)
        self._launch('bu_validate',self.nc,(nc,ng,nl,reference_layers,
            reference_depth,reference_bed,toe_removed_mass,self.trial_layers,h['undercut'],
            h['reference_height'],h['toe_height'],h['reference_length'],
            h['toe_cumulative'],h['collapse_cumulative'],h['collapse_count'],
            self.status['validate']))
        self._fold(('validate',))
        return {'trial_layers':self.trial_layers,'trial_history':self.trial_history,
                'collapse_cell':self.collapse_cell,'collapse_target':self.collapse_target,
                'removed_by_layer':self.removed_by_layer,'removed_mass':self.removed_mass,
                'actual_volume':self.actual_volume,'proposal_volume':self.proposal,
                'recipient_scale':self.toe_scale,
                'recipient_gain':self.recipient_gain,'grain_budget':self.grain_budget,
                'status':self.failure,'detail_status':tuple(self.status.values())}
