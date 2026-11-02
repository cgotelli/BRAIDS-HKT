"""Private evolving-bank AMR handoff on CUDA, with host geometry preparation.

Transfers ordinary-cell bank support and interface-width memory. This is not
a stored cavity-volume model, a fractional front, or a change to f-wave fluxes.
"""
from pathlib import Path
from bisect import bisect_left
import hashlib
import numpy as np

from .amr_bank_wetfront import _history_matrix, _install_history, COUNT, HISTORY_NAMES
from .amr_face_correspondence import plan_face_correspondence, apply_face_ledgers
from braids_support.gpu_fwave.evolving_toe_map import OPTIONS


class BankRegridVeto(RuntimeError):
    """Admissible accepted physics, but this candidate cannot retain its memory.

    Only explicitly identified geometric/constitutive incompatibilities use
    this exception. Invalid arrays, conservation failures and CUDA errors do not.
    """
    def __init__(self,code,message):
        super().__init__(message)
        self.code=code


def _private_target(old,new):
    """Reject shared device byte ranges before a candidate can be written."""
    from .a1_full_array_transaction import _unique_arrays
    if old is new or old.solver is new.solver:
        raise ValueError('Evolving AMR target must be private, not the accepted model')
    prior=sorted((int(v.data.ptr),int(v.data.ptr)+v.nbytes)
                 for v in _unique_arrays(old.solver).values() if v.nbytes)
    starts=[a for a,b in prior];ends=[];furthest=0
    for a,b in prior:furthest=max(furthest,b);ends.append(furthest)
    for value in _unique_arrays(new.solver).values():
        if not value.nbytes:continue
        start=int(value.data.ptr);index=bisect_left(starts,start+value.nbytes)-1
        if index>=0 and ends[index]>start:
            raise ValueError('Evolving AMR private target storage aliases accepted device state')


def cell_overlaps(plan):
    """One-level complete groups, in destination CSR order; geometry only."""
    old,new=plan.old,plan.new
    oi={int(n):i for i,n in enumerate(old.node_ids)}
    ni={int(n):i for i,n in enumerate(new.node_ids)}
    rows=[[] for _ in new.node_ids]
    for n in plan.retained_ids:rows[ni[n]].append((oi[n],float(old.data['area'][oi[n]])))
    for p,children in zip(plan.coarsen_parents,plan.coarsen_children):
        rows[ni[p]].extend((oi[c],float(old.data['area'][oi[c]])) for c in children)
    for p,children in zip(plan.refine_parents,plan.refine_children):
        for c in children:rows[ni[c]].append((oi[p],float(new.data['area'][ni[c]])))
    if any(not row for row in rows):raise ValueError('Incomplete AMR cell overlap')
    return (np.r_[0,np.cumsum([len(r) for r in rows])].astype(np.int32),
            np.array([i for row in rows for i,a in row],np.int32),
            np.array([a for row in rows for i,a in row],np.float64))


def _accepted_support_history(old):
    """Validate inputs before a geometric veto can bypass transfer braids_support.

    This is read-only. A missing accepted support is corruption, not proof
    that a proposed coarsening swallowed a previously valid interface.
    """
    from braids_support.gpu_fwave.evolving_toe_map import checkpoint_specs
    cp=old.cp;solver=old.solver;mapping=solver.evolving_toe
    nc,nf=mapping.nc,mapping.nf;m=mapping.accepted
    specs=checkpoint_specs(nc,nf)
    if set(m)!=set(specs):raise ValueError('Invalid accepted AMR support inventory')
    for key,(shape,dtype) in specs.items():
        value=m[key]
        if (not isinstance(value,cp.ndarray) or value.shape!=shape or value.dtype!=dtype or
            not value.flags.c_contiguous or value.device.id!=mapping.material.device.id):
            raise ValueError('Invalid accepted AMR support array: '+key)
    for key in ('support_donor','support_toe','support_face_bank','support_face_toe'):
        if bool(cp.any((m[key]<-1)|(m[key]>=nc))):
            raise ValueError('Invalid accepted AMR support endpoint/root IDs')
    if (int(m['initialized'].item())!=1 or
        bool(cp.any((m['support_face']<-1)|(m['support_face']>=nf))) or
        bool(cp.any(~cp.isfinite(m['support_distance'])|(m['support_distance']<0))) or
        bool(cp.any((m['support_root']<0)|(m['support_root']>1)))):
        raise ValueError('Invalid accepted AMR support identity')
    bank,toe=m['support_face_bank'],m['support_face_toe'];present=bank>=0
    owner=cp.asarray(old.topology.data['owner']);other=cp.asarray(old.topology.data['neighbour'])
    endpoints=((bank==owner)&(toe==other))|((bank==other)&(toe==owner))
    if (bool(cp.any(present!=(toe>=0))) or
        bool(cp.any(present&((other<0)|~endpoints)))):
        raise ValueError('Accepted AMR support does not match authenticated face endpoints')
    roots=cp.bincount(bank[present],minlength=nc)>0
    if bool(cp.any(roots!=(m['support_root']!=0))):
        raise ValueError('Accepted AMR support root/face coverage differs')
    history=solver.bank_history
    for name in (*HISTORY_NAMES,COUNT):
        value=history[name];dtype=cp.int32 if name==COUNT else cp.float64
        if (not isinstance(value,cp.ndarray) or value.shape!=(nc,) or value.dtype!=dtype or
            not value.flags.c_contiguous or value.device.id!=mapping.material.device.id or
            bool(cp.any(value<0)) or (name!=COUNT and bool(cp.any(~cp.isfinite(value))))):
            raise ValueError('Invalid accepted AMR bank history: '+name)
    rows=_history_matrix(solver)
    memory=cp.any(rows[:,:4]>0,axis=1)
    if (bool(cp.any(memory&~roots)) or
        bool(cp.any((rows[:,0]>0)&(rows[:,3]<=0)))):
        raise ValueError('Accepted AMR bank history has lost its support')
    return rows


def _swallowed_dormant_support(plan,old,face_plan,history,source_folder):
    """Decline only a proven internalized, valid dormant physical interface.

    General lost/duplicated overlap (history status 8) remains fatal. No
    reference memory is discarded or clipped. The rare veto candidate gets a
    private canonical map rebuild before it can be classified as harmless.
    """
    if not plan.coarsen_parents:return
    cp=old.cp;mapping=old.solver.evolving_toe;m=mapping.accepted
    ids=np.asarray(old.ids);index={int(node):i for i,node in enumerate(ids)}
    merged=np.full(len(ids),-1,np.int64)
    for parent,children in zip(plan.coarsen_parents,plan.coarsen_children):
        for child in children:merged[index[child]]=parent
    data=plan.old.data;owner=np.asarray(data['owner']);other=np.asarray(data['neighbour'])
    internal=(other>=0)&(merged[owner]>=0)&(merged[owner]==merged[np.maximum(other,0)])
    disappeared=np.zeros(len(owner),bool)
    disappeared[face_plan.unmapped_old]=face_plan.unmapped_fraction==1.
    faces=cp.asarray(np.flatnonzero(internal&disappeared),dtype=cp.int64)
    if not len(faces):return
    roots=m['support_face_bank'][faces]
    eligible=(roots>=0)&(m['is_toe_face'][faces]==0)
    roots=roots[eligible]
    if not len(roots) or not bool(cp.any(history[roots,:4]>0)):return
    # Authenticate all accepted map fields, not just the face that prompted
    # the veto. An unrelated malformed/duplicated CSR must not be hidden.
    from braids_support.gpu_fwave.evolving_toe_map import ResidentEvolvingToeMap
    verifier=ResidentEvolvingToeMap(*(data[key] for key in
        ('area','owner','neighbour','offset','faces','signs','length','xy','mid')),
        data['bank_material_mask'],data['bulk_fraction'],old.state['layers'].shape[0],
        dry_tolerance=float(data['dry_tolerance']),
        source_path=Path(source_folder)/'evolving_toe_map.cu')
    verifier.restore(m,old.solver.bed(),old.state['q'][0],old.state['layers'])
    raise BankRegridVeto('swallowed_dormant_support',
        'AMR coarsening would swallow a valid dormant bank interface; retain finer support')


def _kernel(source):
    import cupy as cp
    text=Path(source).read_text()
    if '#define BRAIDS_AMR_EVOLVING_SUPPORT_ABI 1' not in text:
        raise ValueError('Evolving AMR support ABI mismatch')
    names=('ae_seed','ae_lengths','ae_edges','ae_exposure','ae_merge_guard')
    module=cp.RawModule(code=text,options=OPTIONS,name_expressions=names)
    return {name:module.get_function(name) for name in names}


def _launch(k,name,n,args):
    if n:k[name](((n+127)//128,),(128,),args)


def _exposure(cp,k,solver,mapping):
    nc=solver.nc
    ids=cp.arange(nc,dtype=cp.int32)
    roots=mapping['support_donor']
    keys=cp.where(roots>=0,roots,nc).astype(cp.int32)
    order=cp.lexsort(cp.stack((ids,mapping['support_distance'],keys)))
    counts=cp.bincount(keys,minlength=nc+1)[:nc]
    offsets=cp.empty(nc+1,cp.int32);offsets[0]=0;offsets[1:]=cp.cumsum(counts,dtype=cp.int32)
    result=cp.zeros(nc,cp.float64)
    _launch(k,'ae_exposure',nc,(np.int32(nc),np.int32(solver.ng),np.int32(solver.state['layers'].shape[0]),
        offsets,ids[order],mapping['support_toe'],solver.state['layers'],solver.bed(),
        solver.state['q'][0],np.float64(solver.data['dry_tolerance']),result))
    active=cp.zeros(nc,cp.float64)
    _launch(k,'ae_exposure',nc,(np.int32(nc),np.int32(solver.ng),np.int32(solver.state['layers'].shape[0]),
        mapping['candidate_offset'],mapping['candidates'],mapping['mapped_toe'],
        solver.state['layers'],solver.bed(),solver.state['q'][0],
        np.float64(solver.data['dry_tolerance']),active))
    return cp.where(mapping['is_toe_donor']!=0,active,result)


def transfer_evolving_bank(prepared,plan,old,new,*,source_folder):
    """Populate only the private target model; old accepted arrays stay intact.

    Target physical state must already be conservatively installed. Commit of
    topology/model belongs to the caller after this returns successfully.
    """
    from braids_support.gpu_fwave.amr_evolving_history import propose_history
    from .amr_catalog_mixed_transition import plan_mixed_transition
    _private_target(old,new)
    expected=plan_mixed_transition(prepared,plan.old,plan.new)
    if (old.topology is not plan.old or new.topology is not plan.new or
        any(getattr(plan,key)!=getattr(expected,key) for key in
            ('retained_ids','coarsen_parents','coarsen_children','refine_parents','refine_children'))):
        raise ValueError('Evolving AMR transition differs from authenticated complete groups')
    cp=old.cp;src=Path(source_folder)
    old_solver,new_solver=old.solver,new.solver
    om,nm=old_solver.evolving_toe,new_solver.evolving_toe
    if om is None or nm is None:raise ValueError('Evolving AMR requires both resident mappings')
    face_plan=plan_face_correspondence(prepared,plan.old,plan.new)
    history=_accepted_support_history(old)
    _swallowed_dormant_support(plan,old,face_plan,history,src)
    k=_kernel(src/'amr_evolving_support.cu')
    old_face=cp.asarray(face_plan.old_face,dtype=cp.int64)
    new_face=cp.asarray(face_plan.new_face,dtype=cp.int64)
    orientation=cp.asarray(face_plan.orientation,dtype=cp.int32)
    offsets=cp.asarray(face_plan.new_offsets,dtype=cp.int64)
    seed_bank=cp.full(nm.nf,-1,cp.int32);seed_toe=cp.full(nm.nf,-1,cp.int32)
    status=cp.zeros(nm.nf,cp.int32)
    _launch(k,'ae_seed',nm.nf,(np.int32(nm.nf),offsets,old_face,orientation,om.owner,
        om.accepted['support_face_bank'],om.accepted['support_face_toe'],nm.owner,nm.neighbour,
        seed_bank,seed_toe,status))
    if bool(cp.any(status)):
        raise BankRegridVeto('opposed_dormant_support','Incompatible/opposed dormant AMR supports')
    rebuilt=nm.build_remapped_support(new_solver.bed(),new.state['q'][0],
        new.state['layers'],seed_bank,seed_toe)
    if int(rebuilt['status'].item()):raise RuntimeError('AMR target bank support rebuild failed')
    lengths=cp.zeros(om.nc,cp.float64)
    _launch(k,'ae_lengths',om.nc,(np.int32(om.nc),om.offset,om.faces,
        om.accepted['support_face_bank'],om.length,lengths))
    ne=len(face_plan.old_face)
    roots_old=cp.full(ne,-1,cp.int32);roots_new=cp.full(ne,-1,cp.int32)
    weights=cp.zeros(ne,cp.float64)
    _launch(k,'ae_edges',ne,(np.int32(ne),old_face,new_face,
        cp.asarray(face_plan.old_fraction),om.accepted['support_face_bank'],
        nm.trial['support_face_bank'],seed_bank,om.length,lengths,roots_old,roots_new,weights))
    valid=roots_new>=0
    ro=roots_old[valid];rn=roots_new[valid];ww=weights[valid];ff=old_face[valid]
    # Stable destination/root/physical-face ordering, all mapping work resident.
    order=cp.lexsort(cp.stack((ff,ro,rn)))
    ro=cp.ascontiguousarray(ro[order]);rn=rn[order]
    ww=cp.ascontiguousarray(ww[order]);ff=cp.ascontiguousarray(ff[order])
    counts=cp.bincount(rn,minlength=nm.nc)
    so=cp.empty(nm.nc+1,cp.int32);so[0]=0;so[1:]=cp.cumsum(counts,dtype=cp.int32)
    # Stored L is the last submerged interface length, not a cavity volume.
    # If it covers only part of an old support, splitting it proportionally
    # would make the next wet update restore L on a child while a dormant
    # child retains its share. Defer that topology change, without affecting
    # unrelated retained roots or an entirely dormant one-face interface.
    valid_edge=roots_old>=0
    er=roots_old[valid_edge];en=roots_new[valid_edge]
    ef=old_face[valid_edge];eg=new_face[valid_edge]
    changed=((cp.asarray(old.ids)[er]!=cp.asarray(new.ids)[en])|
             (cp.asarray(face_plan.old_fraction)[valid_edge]!=1.)|
             (om.length[ef]!=nm.length[eg]))
    changes=cp.bincount(er[changed],minlength=om.nc)
    mismatch=cp.abs(history[:,3]-lengths)>1024*np.finfo(float).eps*cp.maximum(lengths,history[:,3])
    if bool(cp.any((changes>0)&(history[:,3]>0)&mismatch)):
        raise BankRegridVeto('partial_reference_length',
            'AMR changing support has partial wet/dormant reference length; retain finer support')
    merge_status=cp.zeros(nm.nc,cp.int32)
    normal=np.asarray(plan.old.data['normal'],dtype=np.float64)
    _launch(k,'ae_merge_guard',nm.nc,(np.int32(nm.nc),np.int32(om.nf),so,ro,ff,
        om.accepted['support_face_bank'],om.owner,cp.asarray(np.ascontiguousarray(normal)),
        history,merge_status))
    if bool(cp.any(merge_status)):
        raise BankRegridVeto('opposed_bank_interfaces','AMR cannot merge opposed bank interfaces')
    cell_offsets,cell_old,cell_area=map(cp.asarray,cell_overlaps(plan))
    result=propose_history(history,old_solver.bank_history[COUNT],om.area,nm.area,
        cell_offsets,cell_old,cell_area,so,ro,ww,lengths,
        critical_width=float(old_solver.bank_undercut.critical),
        old_collapse_exposure=_exposure(cp,k,old_solver,om.accepted),
        new_collapse_exposure=_exposure(cp,k,new_solver,nm.trial),
        source_path=src/'amr_evolving_history.cu')
    if int(result.status.item())==32:
        raise BankRegridVeto('mixed_collapse_trigger',
            'AMR cannot merge bank roots on opposite sides of the collapse threshold')
    if int(result.status.item()):
        raise RuntimeError('Evolving bank AMR history transfer rejected: '+str(result.receipt))
    ledger=apply_face_ledgers(cp,face_plan,old_solver.toe_face_signed,
                             old_solver.toe_face_gross,nm.nf)
    _install_history(new_solver,result.history,result.count)
    cp.copyto(new_solver.toe_face_signed,ledger['signed'])
    cp.copyto(new_solver.toe_face_gross,ledger['gross'])
    nm.commit(nm.failure)
    archived=dict(operator_identity=plan.old.source_identity,
                  node_ids=plan.old.node_ids.copy(),
                  old_face=ledger['archived_old_face'],
                  fraction=ledger['archived_fraction'],
                  signed=ledger['archived_signed'],gross=ledger['archived_gross'])
    receipt=dict(result.receipt,bank_history_rule='length_weighted_interface_memory_v1',
        evolving_mapping_rebuilt=True,face_orientation='owner_to_neighbour',
        partial_interface_length_change_guard=True,
        collapse_exposure_rule='active_candidates_else_dormant_support',
        mapped_face_overlaps=ne,archived_face_segments=len(face_plan.unmapped_old),
        support_source_sha256=hashlib.sha256((src/'amr_evolving_support.cu').read_bytes()).hexdigest())
    return receipt,archived
