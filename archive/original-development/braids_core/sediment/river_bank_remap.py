"""Zero-time surveyed bank history transfer on actual oriented face overlaps.

No physical source, reclassification or inferred survey. Factories must provide
an equivalent explicit map and candidate footprint on both mesh epochs.
"""
from pathlib import Path
import numpy as np
from .bed_columns_v101 import host
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine


def bank_contract(a):
    toe=getattr(a,'_physical_toe',None);failure=getattr(a,'_physical_toe_failure',None)
    if toe is None:
        if failure is not None:raise ValueError('Bank failure requires toe owner during remeshing')
        return None
    return dict(toe={k:v for k,v in toe.identity.items() if k not in ('map','geometry_sha256')},
        failure=None if failure is None else {k:v for k,v in failure.identity.items()
            if k not in ('groups','primary_faces','centroid_sha256','toe_supply')})


def face_overlap(source,target,*,primary=False):
    """Return overlap / source length, with physical bank-to-toe orientation."""
    old,new=source.data,target.data;left=source._physical_toe.identity['map'];right=target._physical_toe.identity['map']
    src=left['face_ids'];dst=right['face_ids']
    if primary:
        src=sorted(source._physical_toe_failure.identity['primary_faces'].values())
        dst=sorted(target._physical_toe_failure.identity['primary_faces'].values())
    sb=dict(zip(left['face_ids'],left['bank_cells']));tb=dict(zip(right['face_ids'],right['bank_cells']))
    os=np.zeros(old['nf']);ts=np.zeros(new['nf']);coverage=np.zeros(old['nf']);covered=np.zeros(new['nf'])
    for f in src:os[f]=1. if old['owner'][f]==sb[f] else -1.
    for f in dst:ts[f]=1. if new['owner'][f]==tb[f] else -1.
    rows={}
    for j in dst:
        normal=new['normal'][:,j]*ts[j];values=[]
        for i in src:
            if np.dot(normal,old['normal'][:,i]*os[i])<1-1e-10:continue
            shift=new['mid'][:,j]-old['mid'][:,i]
            tol=64*np.spacing(max(1.,float(np.max(np.abs(old['mid'][:,i]))),float(np.max(np.abs(new['mid'][:,j])))))
            if abs(np.dot(shift,normal))>max(tol,1e-11*max(old['length'][i],new['length'][j])):continue
            tangent=np.array([-normal[1],normal[0]]);center=float(np.dot(shift,tangent))
            overlap=max(0.,min(.5*old['length'][i],center+.5*new['length'][j])-max(-.5*old['length'][i],center-.5*new['length'][j]))
            if overlap>0:
                w=overlap/old['length'][i];values.append((i,w));coverage[i]+=w;covered[j]+=overlap/new['length'][j]
        rows[j]=values
    if not np.allclose(coverage[src],1.,atol=1e-9,rtol=0) or not np.allclose(covered[dst],1.,atol=1e-9,rtol=0):
        raise ValueError('Surveyed '+('primary ' if primary else '')+'toe partition, orientation or physical location changed during remeshing')
    off=[0];ids=[];weights=[]
    for j in range(new['nf']):
        for i,w in rows.get(j,[]):ids.append(i);weights.append(w)
        off.append(len(ids))
    return np.asarray(off,np.int32),np.asarray(ids,np.int32),np.asarray(weights,float),os,ts


def anchor_overlap(source,target,face,mapping):
    """Bank history follows surveyed line measure, not donor-cell area."""
    off,ids,weights,_,_=face;old,new=source.data,target.data
    left=source._physical_toe.identity['map'];right=target._physical_toe.identity['map']
    sb=dict(zip(left['face_ids'],left['bank_cells']));tb=dict(zip(right['face_ids'],right['bank_cells']))
    lengths={b:sum(old['length'][f] for f in left['face_ids'] if sb[f]==b) for b in set(sb.values())}
    rows={}
    for j,b in tb.items():
        row=rows.setdefault(b,{})
        for k in range(off[j],off[j+1]):
            i=int(ids[k]);s=sb[i];row[s]=row.get(s,0.)+weights[k]*old['length'][i]/lengths[s]
    # Explicit candidate regions must match by actual cell overlaps. A target
    # group may split/merge roots only along the same overlapping toe survey.
    def members(op,n):
        result=np.full(n,-1,np.int32)
        for b,cells in op.identity['groups'].items():result[cells]=int(b)
        return result
    sg=members(source._physical_toe_failure,source.physics.n);tg=members(target._physical_toe_failure,target.physics.n)
    for c in range(target.physics.n):
        for k in range(mapping.offsets[c],mapping.offsets[c+1]):
            oldgroup=sg[mapping.indices[k]];newgroup=tg[c]
            if ((oldgroup<0)!=(newgroup<0) or (newgroup>=0 and oldgroup not in rows[newgroup])):
                raise ValueError('Surveyed candidate footprint or group ownership changed during remeshing')
    offsets=[0];indices=[];w=[]
    for c in range(target.physics.n):
        for s,value in sorted(rows.get(c,{}).items()):indices.append(s);w.append(value)
        offsets.append(len(indices))
    idx=np.asarray(indices,np.int32);w=np.asarray(w,float)
    total=np.bincount(idx,weights=w,minlength=source.physics.n)
    if not np.allclose(total[list(lengths)],1.,atol=1e-9,rtol=0):raise ValueError('Incomplete bank history partition')
    # Each old integer transaction marker is retained once, at the greatest
    # overlap (centroid-coordinate tie break). Never round/split an event count.
    xy=host(target.polygon.g['xy']);choice={}
    for c,row in rows.items():
        for s,value in row.items():
            key=(-value,float(xy[c,0]),float(xy[c,1]))
            if s not in choice or key<choice[s][0]:choice[s]=(key,c)
    counts={}
    for s,(_,c) in choice.items():counts.setdefault(c,[]).append(s)
    coff=[0];ci=[]
    for c in range(target.physics.n):ci.extend(sorted(counts.get(c,[])));coff.append(len(ci))
    return np.asarray(offsets,np.int32),idx,w,np.asarray(coff,np.int32),np.asarray(ci,np.int32)


def stage_bank_history(source,target,engine,mapping,columns,q):
    if bank_contract(source)!=bank_contract(target):raise ValueError('Surveyed bank physics changed during remeshing')
    if getattr(source,'_physical_toe',None) is None:return {},None
    p,t,xp=source.physics,target.physics,source.xp;face=face_overlap(source,target)
    off,ids,w,os,ts=face
    if bool(xp.any(p.physical_toe_face_cumulative[:,xp.asarray(os==0)])):
        raise ValueError('Toe ledger outside the surveyed faces')
    signed=p.physical_toe_face_cumulative*xp.asarray(os)
    carried=engine.apply(signed,off,ids,w,t.nf)
    if not bool(xp.allclose(carried.sum(1),signed.sum(1),rtol=1e-12,atol=1e-12)):
        raise ValueError('Toe face mass history changed during remeshing')
    state=dict(physical_toe_face_cumulative=carried*xp.asarray(ts),physical_toe_face_trial=xp.zeros_like(carried))
    receipt=dict(policy='actual oriented toe-face overlap; no physical transfer or clock advance',
        toe_class_residual_kg=host(carried.sum(1)-signed.sum(1)).tolist(),bank_failure=False)
    if getattr(source,'_physical_toe_failure',None) is None:return state,receipt
    face_overlap(source,target,primary=True)
    ao,ai,aw,co,ci=anchor_overlap(source,target,face,mapping)
    old=p.bank_failure_history;sa=xp.asarray(mapping.source_area);ta=xp.asarray(mapping.target_area)
    source._physical_toe_failure._validate(source.columns.mass,source.columns.sv,source.columns.rock,source.q[0],old,p.bank_failure_count)
    mapped=lambda value:engine.apply(xp.ascontiguousarray(value),ao,ai,aw,t.n)
    length=mapped(old[3]);height=mapped(old[1]*old[3]);water=mapped(old[2]*old[3])
    cross=(old[1]-old[2])*old[3];potential=mapped(old[0]*cross)
    h=xp.zeros((6,t.n));h[3]=length
    h[1]=xp.where(length>0,height/xp.maximum(length,1e-300),0.)
    h[2]=xp.where(length>0,water/xp.maximum(length,1e-300),0.)
    h[0]=xp.where(height>water,potential/xp.maximum(height-water,1e-300),0.)
    if bool(xp.any((potential>0)&(height<=water))):raise ValueError('Unsafe retained undercut geometry during remeshing')
    h[4:]=mapped(old[4:]*sa)/ta
    # An unchanged surveyed segment is an exact history copy, not a multiply
    # then divide round trip. Normal refinement changes only the volume/area
    # diagnostics; it need not round the retained width or geometric measures.
    rows=np.flatnonzero(np.diff(ao)==1);parents=ai[ao[rows]]
    use=aw[ao[rows]]==1.;rows=rows[use];parents=parents[use]
    h[:4,xp.asarray(rows)]=old[:4,xp.asarray(parents)]
    exact=mapping.target_area[rows]==mapping.source_area[parents]
    h[4:,xp.asarray(rows[exact])]=old[4:,xp.asarray(parents[exact])]
    actual_potential=h[0]*(h[1]-h[2])*h[3]
    oldvol=(old[4:]*sa).sum(1);newvol=(h[4:]*ta).sum(1)
    if (not bool(xp.allclose(newvol,oldvol,rtol=1e-12,atol=1e-14))
            or not bool(xp.allclose(actual_potential.sum(),(old[0]*cross).sum(),rtol=1e-12,atol=1e-14))):
        raise ValueError('Bank retained-volume history changed during remeshing')
    counts=xp.zeros(t.n,xp.int64);status=xp.zeros(t.n,xp.int32)
    native=WaveEngine(source.backend,p.flow.threads,source=Path(__file__).with_suffix('.h'),names=('bank_counts',))
    native.launch('bank_counts',t.n,t.n,xp.asarray(co),xp.asarray(ci),p.bank_failure_count,counts,status)
    # Python integers only verify the diagnostic reduction; state arithmetic
    # and publication stay on the selected backend, including CUDA int64.
    before=sum(map(int,host(p.bank_failure_count)));after=sum(map(int,host(counts)))
    if bool(xp.any(status)) or before!=after:raise ValueError('Bank transaction counter overflow or loss during remeshing')
    checked=target._physical_toe_failure.reconcile(columns.mass,columns.sv,columns.rock,q[0],h,counts)
    if bool(xp.any((h[0]>0)&(checked[0]==0))):
        raise ValueError('Unsafe bank-history coarsening removes exposed undercut; retain refinement')
    state.update(bank_failure_history=checked,bank_failure_history_trial=checked.copy(),
        bank_failure_count=counts,bank_failure_count_trial=counts.copy(),
        bank_failure_class_cumulative=engine.cells(p.bank_failure_class_cumulative,mapping,extensive=True),
        bank_failure_class_trial=xp.zeros_like(t.bank_failure_class_trial))
    receipt.update(bank_failure=True,cumulative_bulk_residual_m3=host(newvol-oldvol).tolist(),
        retained_potential_residual_m3=float(actual_potential.sum()-(old[0]*cross).sum()),
        actual_transactions_before=before,actual_transactions_after=after,
        counters='one integer marker per old transaction; greatest overlap, centroid tie break; diagnostic only',
        native=native.identity)
    return state,receipt
