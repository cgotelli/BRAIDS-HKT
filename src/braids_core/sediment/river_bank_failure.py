"""Retained noncohesive undercut and block release, one CPU/GPU column owner.

Actual toe volume drives history; block material comes from explicit connected
bank groups. This BRAIDS extension is not FLOW ThetSD or cohesive excess shear.
Public groups come from the authenticated geographic survey, not raw cell IDs.
"""
from pathlib import Path
from types import MethodType
import heapq
import numpy as np
from .bed_columns_v101 import host,digest as object_digest
from .bend_methods_v1 import BendTrialRejected,digest
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

METHOD='bank.braids_retained_undercut_block_collapse'
HISTORY=('undercut','reference_height','toe_height','reference_length','toe_cumulative','collapse_cumulative')
KERNELS=('failure_bed','failure_history','failure_propose','failure_shared_limit',
         'failure_extract','failure_deposit','failure_reconcile')


class PhysicalBankFailure:
    def __init__(self,supply,data,cell_xy,groups,primary_faces,*,critical_undercut,dry):
        if (type(critical_undercut) not in (float,int) or not np.isfinite(critical_undercut)
                or critical_undercut<=0 or type(dry) not in (float,int) or not np.isfinite(dry) or dry<0):
            raise ValueError('Explicit positive undercut threshold and nonnegative hydraulic dry tolerance required')
        self.supply=supply;self.xp=supply.xp;self.n=n=supply.n;self.critical=float(critical_undercut);self.dry=float(dry)
        xy=np.asarray(host(cell_xy),float);own=np.asarray(host(data['owner']));nb=np.asarray(host(data['neighbour']))
        if xy.shape!=(n,2) or not np.isfinite(xy).all():raise ValueError('Actual cell centroids required for surveyed bank groups')
        mapping=supply.identity['map'];banks=set(mapping['bank_cells']);recipients=set(mapping['toe_cells'])
        if (not isinstance(groups,dict) or set(groups)!=banks or not isinstance(primary_faces,dict)
                or set(primary_faces)!=banks):raise ValueError('Each toe donor needs one explicit connected bank group and primary toe face')
        facebank=dict(zip(mapping['face_ids'],mapping['bank_cells']));faceto=dict(zip(mapping['face_ids'],mapping['toe_cells']))
        graph=[[] for _ in range(n)]
        for a,b in zip(own,nb):
            if b<0:continue
            distance=float(np.linalg.norm(xy[a]-xy[b]))
            if distance<=0:raise ValueError('Distinct centroids required across an internal bank face')
            graph[a].append((int(b),distance));graph[b].append((int(a),distance))
        seen=set();ordered={};toe=np.full(n,-1,np.int32)
        for bank in sorted(banks):
            face=primary_faces[bank]
            if type(face) is not int or facebank.get(face)!=bank:raise ValueError('Primary toe must be an actual mapped interface of its donor')
            toe[bank]=faceto[face]
            values=groups[bank]
            if (not isinstance(values,(list,tuple)) or not values or any(type(v) is not int or v<0 or v>=n for v in values)
                    or len(set(values))!=len(values) or bank not in values):raise ValueError('Unique explicit candidate cells including their toe donor required')
            members=set(values)
            if seen.intersection(members) or recipients.intersection(members):raise ValueError('Bank groups and toe recipients must be dependency-disjoint')
            seen.update(members);dist={bank:0.};queue=[(0.,bank)]
            while queue:
                length,c=heapq.heappop(queue)
                if length!=dist[c]:continue
                for j,w in graph[c]:
                    if j in members and length+w<dist.get(j,np.inf):
                        dist[j]=length+w;heapq.heappush(queue,(length+w,j))
            if set(dist)!=members:raise ValueError('Surveyed bank group crosses a wall or disconnected component')
            ordered[bank]=sorted(members,key=lambda c:(dist[c],c))
        def csr(rows):
            values=[rows.get(c,[]) for c in range(n)]
            return np.r_[0,np.cumsum([len(v) for v in values])].astype(np.int32),np.asarray([i for v in values for i in v],np.int32)
        off,candidates=csr(ordered);recipient_groups={}
        for b in sorted(banks):recipient_groups.setdefault(int(toe[b]),[]).append(b)
        toff,donors=csr(recipient_groups);xp=self.xp
        self.g={k:xp.asarray(v) for k,v in dict(offset=off,candidates=candidates,toe=toe,toff=toff,donors=donors).items()}
        self.engine=WaveEngine(supply.engine.backend,supply.engine.threads,source=Path(__file__).with_suffix('.h'),names=KERNELS)
        self.identity=dict(method=METHOD,source_sha256=digest(__file__),runtime=self.engine.identity,
            critical_undercut_m=self.critical,dry_tolerance_m=self.dry,
            groups={str(b):v for b,v in ordered.items()},primary_faces={str(b):primary_faces[b] for b in sorted(banks)},
            centroid_sha256=object_digest(xy.tolist()),toe_supply=supply.identity,
            selection='nearest connected currently dry stocked bank column by centroid-path distance',
            trigger='critical width with 64-ulp threshold guard; count actual positive release transactions',
            geometry='current bed, depth and actual packing; shared-recipient simultaneous-volume bound',
            history='actual supplied BULK volume / submerged material-interface area; no second clock multiplier',
            provenance='BRAIDS ABI3 physical undercut adapted to common layered packing; not FLOW or cohesive excess shear',public_support=False)

    def new_history(self):return self.xp.zeros((6,self.n)),self.xp.zeros(self.n,self.xp.int64)

    def _validate(self,mass,sv,rock,depth,history,count):
        xp=self.xp;n=self.n;ng=len(self.supply.density);nl=mass.shape[0]
        fields=((mass,(nl,ng,n)),(sv,(nl,n)),(rock,(n,)),(depth,(n,)),(history,(6,n)))
        if nl<1 or any(not isinstance(a,xp.ndarray) or a.shape!=shape or a.dtype!=xp.float64 or not a.flags.c_contiguous for a,shape in fields):
            raise ValueError('Resident contiguous float64 bank state required')
        if (not isinstance(count,xp.ndarray) or count.shape!=(n,) or count.dtype!=xp.int64 or not count.flags.c_contiguous
                or bool(xp.any(count<0)) or any(not bool(xp.all(xp.isfinite(a))) for a,_ in fields)
                or bool(xp.any(mass<0)) or bool(xp.any(sv<=0)) or bool(xp.any(sv>1))
                or bool(xp.any(depth<0)) or bool(xp.any(history<0))):raise ValueError('Invalid accepted bank inventory or retained history')

    def bed_fields(self,mass,sv,rock):
        xp=self.xp;n=self.n;bed=xp.empty(n);stock=xp.empty(n);maxsv=xp.empty(n)
        self.engine.launch('failure_bed',n,n,len(self.supply.density),mass.shape[0],mass,sv,self.supply.density,rock,
                           self.supply.g['area'],bed,stock,maxsv)
        if not bool(xp.all(xp.isfinite(bed))) or not bool(xp.all(xp.isfinite(stock))):
            raise BendTrialRejected('Nonfinite physical bank geometry')
        return bed,stock,maxsv

    def reconcile(self,mass,sv,rock,depth,history,count):
        self._validate(mass,sv,rock,depth,history,count)
        result=history.copy();bed,stock,_=self.bed_fields(mass,sv,rock);g=self.g
        self.engine.launch('failure_reconcile',self.n,self.n,self.dry,g['offset'],g['candidates'],g['toe'],bed,depth,stock,result)
        return result

    def propose(self,mass,sv,rock,depth,actual_toe_volume,interface_area,history,count):
        self._validate(mass,sv,rock,depth,history,count)
        xp=self.xp;n=self.n;nf=self.supply.nf;ng=mass.shape[1];g=self.g;s=self.supply.g
        for a,shape in ((actual_toe_volume,(n,)),(interface_area,(nf,))):
            if (not isinstance(a,xp.ndarray) or a.shape!=shape or a.dtype!=xp.float64 or not a.flags.c_contiguous
                    or not bool(xp.all(xp.isfinite(a))) or bool(xp.any(a<0))):raise ValueError('Actual nonnegative toe volume and submerged interface area required')
        if bool(xp.any(actual_toe_volume[g['toe']<0]!=0)) or bool(xp.any(interface_area[s['bank']<0]!=0)):
            raise ValueError('Toe history input outside the explicit surveyed map')
        h=history.copy();events=count.copy();work=mass.copy();moved=xp.zeros((ng,n));exposed=xp.empty(n,xp.int32)
        status=xp.zeros((3,n),xp.int32);proposal=xp.empty(n);above=xp.empty(n);scale=xp.empty(n);removed=xp.empty((ng,n));actual=xp.empty(n)
        bed,stock,maxsv=self.bed_fields(mass,sv,rock);run=self.engine.launch
        run('failure_history',n,n,self.dry,g['offset'],g['candidates'],g['toe'],s['boff'],s['bfaces'],
            interface_area,s['length'],actual_toe_volume,s['area'],bed,depth,stock,h,exposed,status[0])
        run('failure_propose',n,n,self.critical,g['toe'],exposed,s['area'],bed,depth,stock,h,proposal,above)
        run('failure_shared_limit',n,n,g['toff'],g['donors'],exposed,s['area'],proposal,above,sv,maxsv,scale,status[1])
        run('failure_extract',n,n,ng,mass.shape[0],exposed,g['toe'],s['area'],self.supply.density,sv,proposal,
            above,scale,work,h,events,removed,actual,moved,status[2])
        run('failure_deposit',n,n,ng,g['toff'],g['donors'],removed,s['area'],work,moved)
        if bool(xp.any(status)):raise BendTrialRejected('Physical block geometry or history update failed')
        local=(work-mass).sum(0)*s['area']-moved;tolerance=2e-12*max(1.,float(xp.max(mass*s['area'])))
        if (not bool(xp.all(xp.isfinite(work))) or bool(xp.any(work<0))
                or float(xp.max(xp.abs(local)))>tolerance or float(xp.max(xp.abs(moved.sum(1))))>tolerance):
            raise BendTrialRejected('Physical block stock or per-class closure failed')
        h=self.reconcile(work,sv,rock,depth,h,events)
        return dict(mass=work,history=h,count=events,moved=moved,removed=removed,volume=actual,
                    proposal=proposal,recipient_scale=scale,selected=exposed)


def install(owner,groups,primary_faces,*,critical_undercut):
    p=owner.physics;xp=owner.xp;c=owner.columns
    if (getattr(owner,'_physical_toe',None) is None or owner.time or owner.steps or p._open
            or getattr(owner,'_physical_toe_failure',None) is not None):
        raise ValueError('Install retained failure once after toe supply and before execution')
    if owner._wet_avalanche is not None:raise ValueError('Physical block collapse requires explicit parent-owned repose, not a second wet-slope flux')
    op=PhysicalBankFailure(owner._physical_toe,owner.data,owner.polygon.g['xy'],groups,primary_faces,
        critical_undercut=critical_undercut,dry=float(p.parameters.dry))
    owner._physical_toe_failure=op
    p.bank_failure_history,p.bank_failure_count=op.new_history()
    p.bank_failure_history_trial=p.bank_failure_history.copy();p.bank_failure_count_trial=p.bank_failure_count.copy()
    p.bank_failure_class_trial=xp.zeros_like(p.class_bank_trial)
    p.bank_failure_class_cumulative=xp.zeros_like(p.class_bank_trial)
    commit=p.commit
    def publish(physics):
        commit();physics.bank_failure_history[:]=physics.bank_failure_history_trial
        physics.bank_failure_count[:]=physics.bank_failure_count_trial
        physics.bank_failure_class_cumulative+=physics.bank_failure_class_trial
    p.commit=MethodType(publish,p)
    finish=p.finish
    def trial(physics,*args,**kw):
        physics.bank_failure_history_trial[:]=physics.bank_failure_history
        physics.bank_failure_count_trial[:]=physics.bank_failure_count
        physics.bank_failure_class_trial.fill(0)
        return finish(*args,**kw)
    p.finish=MethodType(trial,p)
    previous=owner.post_redistribution_columns
    def reconcile(this,qnew):
        previous(qnew)
        p.bank_failure_history_trial[:]=op.reconcile(*c.pending[:2],c.rock,qnew[0],
            p.bank_failure_history_trial,p.bank_failure_count_trial)
    owner.post_redistribution_columns=MethodType(reconcile,owner)
    p.identity['physical_bank_failure']=op.identity
    return owner
