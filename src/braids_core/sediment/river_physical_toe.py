"""Noncohesive accepted-Exner supply on the common river owner.

This is NOT the cohesive excess-shear / block-collapse bundle. It supplies
existing toe scour from surveyed adjacent material, with no separate sediment
stock. Public execution requires the authenticated geographic survey binding;
arbitrary private maps are not public project inputs. Legacy sources/checkpoints
are not execution inputs here.
"""
from pathlib import Path
from types import MethodType
import numpy as np
from .bed_columns_v101 import host, digest as object_digest
from .bend_methods_v1 import BendTrialRejected, digest
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

METHOD = 'bank.braids_accepted_exner_toe_supply'
KERNELS = ('toe_state','toe_request','toe_stock_limit','toe_totals',
           'toe_geometry_scales','toe_limit_requests','toe_extract','toe_route','toe_deposit')


class PhysicalToeSupply:
    """One immutable actual-face map; no index guesses or exterior walls."""
    def __init__(self, backend, data, face_ids, bank_cells, *, fraction, threads=1):
        if backend not in ('cpu','gpu'):
            raise ValueError('Explicit CPU or GPU backend required')
        if type(fraction) not in (int,float) or not np.isfinite(fraction) or not 0<fraction<=1:
            raise ValueError('Explicit toe supply fraction in (0,1] required')
        area=np.asarray(host(data['area']),float);length=np.asarray(host(data['length']),float)
        owner=np.asarray(host(data['owner']));neighbour=np.asarray(host(data['neighbour']))
        n=len(area);nf=len(length);ids=np.asarray(face_ids);banks=np.asarray(bank_cells)
        if (area.shape!=(n,) or n<1 or not np.isfinite(area).all() or np.any(area<=0)
                or owner.shape!=(nf,) or neighbour.shape!=(nf,)
                or owner.dtype.kind not in 'iu' or neighbour.dtype.kind not in 'iu'
                or np.any(owner<0) or np.any(owner>=n) or np.any(neighbour>=n)
                or np.any(neighbour<-1) or np.any(neighbour==owner)
                or not np.isfinite(length).all() or np.any(length<=0)):
            raise ValueError('Invalid actual toe mesh')
        if (ids.ndim!=1 or not len(ids) or banks.shape!=ids.shape
                or ids.dtype.kind not in 'iu' or banks.dtype.kind not in 'iu'
                or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=nf)
                or np.any(banks<0) or np.any(banks>=n)):
            raise ValueError('Unique explicit toe face IDs and bank cells required')
        if np.any(neighbour[ids]<0) or np.any((owner[ids]!=banks)&(neighbour[ids]!=banks)):
            raise ValueError('Toe transfer requires an actual internal bank-toe face, never a wall')
        toes=np.where(owner[ids]==banks,neighbour[ids],owner[ids])
        if np.intersect1d(banks,toes).size:
            raise ValueError('A toe recipient cannot also own bank extraction in the same map')
        order=np.argsort(ids);ids=ids[order];banks=banks[order];toes=toes[order]
        def csr(cells):
            groups=[[] for _ in range(n)]
            for f,c in zip(ids,cells):groups[int(c)].append(int(f))
            return (np.r_[0,np.cumsum([len(v) for v in groups])].astype(np.int32),
                    np.asarray([f for v in groups for f in v],np.int32))
        boff,bfaces=csr(banks);toff,tfaces=csr(toes)
        bank=np.full(nf,-1,np.int32);toe=bank.copy();bank[ids]=banks;toe[ids]=toes
        if backend=='cpu':self.xp=np
        else:
            import cupy
            self.xp=cupy
        xp=self.xp;self.n=n;self.nf=nf;self.fraction=float(fraction)
        self.g={k:xp.asarray(v) for k,v in dict(area=area,length=length,
            owner=owner.astype(np.int32),bank=bank,toe=toe,boff=boff,bfaces=bfaces,toff=toff,tfaces=tfaces).items()}
        self.engine=WaveEngine(backend,threads,source=Path(__file__).with_suffix('.h'),names=KERNELS)
        self.identity=dict(method=METHOD,source_sha256=digest(__file__),runtime=self.engine.identity,
            map=dict(face_ids=ids.tolist(),bank_cells=banks.tolist(),toe_cells=toes.tolist()),
            geometry_sha256=object_digest(dict(area=area.tolist(),length=length.tolist(),owner=owner.tolist(),neighbour=neighbour.tolist())),
            fraction=self.fraction,demand='positive NET signed reference-volume transport loss; no capacity proxy',
            extraction='surface down; actual layer packing and finite stock',
            deposition='adjacent toe top layer, one authoritative column publisher',
            provenance='BRAIDS physical-toe supply; not FLOW dry-bank sharing or cohesive excess shear',
            public_support=False)

    def apply(self,mass,sv,rock,delta,surface_sv,depth,*,dry):
        """Return a detached trial. Delta is accepted transport kg/m2, not capacity.

        No timestep or MorFac argument: both are already in delta. Packing is
        held during transfer; the parent may then perform its selected rebalance.
        """
        xp=self.xp;g=self.g;n=self.n;nf=self.nf
        if not hasattr(self,'density'):
            raise ValueError('Bind explicit column reference densities before toe execution')
        ng=len(self.density);nl=mass.shape[0]
        shapes=((mass,(nl,ng,n)),(sv,(nl,n)),(rock,(n,)),(delta,(ng,n)),(surface_sv,(n,)),(depth,(n,)))
        if nl<1 or any(not isinstance(a,xp.ndarray) or a.dtype!=xp.float64 or a.shape!=s
                       or not a.flags.c_contiguous for a,s in shapes):
            raise ValueError('Resident contiguous float64 toe material fields required')
        if (any(not bool(xp.all(xp.isfinite(a))) for a,_ in shapes)
                or bool(xp.any(mass<0)) or bool(xp.any(sv<=0)) or bool(xp.any(sv>1))
                or bool(xp.any(surface_sv<=0)) or bool(xp.any(surface_sv>1)) or bool(xp.any(depth<0))
                or not np.isfinite(dry) or dry<0):
            raise ValueError('Invalid finite toe inventory, packing or depth')
        bed=xp.empty(n);stock=xp.empty(n);maxsv=xp.empty(n);demand=xp.empty(n)
        request=xp.zeros(nf);surface=xp.zeros(nf);outgoing=xp.empty(n);incoming=xp.empty(n)
        donor_scale=xp.empty(n);toe_scale=xp.empty(n)
        work=xp.empty_like(mass);removed=xp.empty((ng,n));volume=xp.empty(n);total=xp.empty(n)
        face=xp.zeros((ng,nf));moved=xp.empty((ng,n));run=self.engine.launch
        run('toe_state',n,n,ng,nl,mass,sv,self.density,rock,g['area'],bed,stock,maxsv)
        run('toe_request',n,n,ng,self.fraction,float(dry),g['toff'],g['tfaces'],g['bank'],g['length'],
            g['area'],delta,self.density,surface_sv,bed,depth,demand,request,surface)
        run('toe_stock_limit',n,n,g['boff'],g['bfaces'],stock,request)
        run('toe_totals',n,n,g['boff'],g['bfaces'],g['toff'],g['tfaces'],g['bank'],request,maxsv,sv,outgoing,incoming)
        run('toe_geometry_scales',n,n,g['bank'],g['toe'],g['boff'],g['bfaces'],g['toff'],g['tfaces'],g['area'],g['length'],
            surface,bed,outgoing,incoming,donor_scale,toe_scale)
        run('toe_limit_requests',nf,nf,g['bank'],g['toe'],donor_scale,toe_scale,request)
        run('toe_extract',n,n,ng,nl,g['boff'],g['bfaces'],request,g['area'],self.density,sv,mass,work,removed,volume,total)
        run('toe_route',n,n,nf,ng,g['boff'],g['bfaces'],g['owner'],request,total,removed,face)
        run('toe_deposit',n,n,nf,ng,g['toff'],g['tfaces'],g['area'],face,removed,work,moved)
        local=(work-mass).sum(0)*g['area']-moved
        scale=max(1.,float(xp.max(mass*g['area'])))
        if (not bool(xp.all(xp.isfinite(work))) or bool(xp.any(work<0))
                or float(xp.max(xp.abs(moved.sum(1))))>2e-12*scale
                or float(xp.max(xp.abs(local)))>2e-12*scale):
            raise BendTrialRejected('Physical toe stock or class transfer closure failed')
        return dict(mass=work,moved=moved,face_mass=face,removed=removed,volume=volume,
                    demand=demand,request=request,surface=surface)

    def bind_material(self,density):
        d=np.asarray(host(density),float)
        if d.ndim!=1 or not 1<=len(d)<=32 or not np.isfinite(d).all() or np.any(d<=0):
            raise ValueError('Positive ordered column reference densities required')
        if hasattr(self,'density'):raise ValueError('Toe material is immutable after binding')
        self.density=self.xp.asarray(d.copy());self.identity['column_density_kg_m3']=d.tolist()
        return self


def install(owner,face_ids,bank_cells,*,fraction):
    """Bind one explicit actual-face map; public admission belongs to the factory."""
    p=owner.physics;c=owner.columns;xp=owner.xp
    if owner.time or owner.steps or p._open or getattr(owner,'_physical_toe',None) is not None:
        raise ValueError('Install physical toe once before execution')
    if (owner.bank_theta!=0 or any(getattr(cap,'total_load',False) for cap in owner.capacity)
            or owner.methods.base_update!=1):
        raise ValueError('Physical toe requires finite stock, separate transport/exchange and FLOW dry-bank sharing OFF')
    if p.morfac!=1 or p.identity.get('river_suspension',{}).get('enabled',False):
        from dataclasses import asdict
        from .bank_setup import validate_inventory
        validate_inventory(asdict(owner.methods))
    op=PhysicalToeSupply(owner.backend,owner.data,face_ids,bank_cells,fraction=fraction,threads=p.flow.threads)
    op.bind_material(owner.material.column);owner._physical_toe=op
    p.physical_toe_face_trial=xp.zeros((len(owner.capacity),p.nf))
    p.physical_toe_face_cumulative=xp.zeros_like(p.physical_toe_face_trial)
    commit=p.commit
    def publish(physics):
        commit();physics.physical_toe_face_cumulative+=physics.physical_toe_face_trial
    p.commit=MethodType(publish,p)
    previous=owner.post_transport_columns
    def post(this,qnew):
        previous(qnew)
        mass,sv,ext,basal,dt,residual=c.pending
        result=op.apply(mass,sv,c.rock,p.class_delta,c.sv[0],qnew[0],dry=p.parameters.dry)
        p.physical_toe_face_trial[:]=result['face_mass']
        p.class_bank_trial[:]+=result['moved']
        work=result['mass'];changed=bool(xp.any(result['moved']!=0))
        failure=getattr(this,'_physical_toe_failure',None)
        if failure is not None:
            block=failure.propose(work,sv,c.rock,qnew[0],result['volume'],result['surface'],
                p.bank_failure_history,p.bank_failure_count)
            work=block['mass'];p.bank_failure_class_trial[:]=block['moved']
            p.class_collapse_trial[:]+=block['moved']
            p.bank_failure_history_trial[:]=block['history'];p.bank_failure_count_trial[:]=block['count']
            changed=changed or bool(xp.any(block['moved']!=0))
            p.last.update(physical_block_release_m3=float(block['volume'].sum()),
                physical_block_transactions=int((block['count']-p.bank_failure_count).sum()))
        if changed:
            work,nextsv,base=c.runtime.evaluate(work,sv,xp.zeros_like(p.class_delta),c.runtime.targets(qnew[0]),p._dt)
            if bool(xp.any(base!=0)):raise BendTrialRejected('Finite toe bed cannot import basal material')
            residual=host((work*p.g['area']).sum(axis=(0,2)))-host((c.mass*p.g['area']).sum(axis=(0,2)))-ext-basal
            if abs(residual).max()>1e-9:raise BendTrialRejected('Physical toe column budget failed')
            c.pending=(work,nextsv,ext,basal,dt,residual);c.seal_pending()
        p.last.update(physical_toe_supply_kg=float(result['removed'].sum()),physical_toe_supply_m3=float(result['volume'].sum()))
    owner.post_transport_columns=MethodType(post,owner)
    finish=p.finish
    def finish_trial(physics,*args,**kwargs):
        physics.physical_toe_face_trial.fill(0)
        return finish(*args,**kwargs)
    p.finish=MethodType(finish_trial,p)
    p.identity['physical_toe_supply']=op.identity
    p.identity['dry_bank']=dict(method=METHOD,fraction=fraction,geometry='surveyed_actual_internal_faces')
    return owner
