"""Resident ordinary-cell channel advance and deterministic bank ownership.

Immutable material is not a wet/dry bank mask. Wet material becomes channel
only along a wet path from original non-bank cells, after its bed has levelled
with a channel neighbour to native geometric roundoff. All numerical mapping
work stays on GPU; the host checks one convergence scalar per parallel sweep.
No persistent subcell front, fractional solid area or moving exterior exists.
"""
from pathlib import Path
import numpy as np
from .surface_transfer import validate_geometry

HERE=Path(__file__).resolve().parent
OPTIONS=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true','--ftz=false')
CORE=('channel','bank','is_toe_cell','is_toe_donor','is_toe_face',
      'candidate_bank','candidate_toe','mapped_toe','mapped_donor','mapped_face','distance','ambiguous')
GEOMETRIC_HISTORY=('undercut','reference_height','toe_height','reference_length')
SUPPORT=('support_donor','support_toe','support_face','support_distance',
         'support_ambiguous','support_root','support_live',
         'support_face_bank','support_face_toe')


def checkpoint_specs(nc,nf):
    """Host-only exact persistent ABI2 layout (unprefixed checkpoint names)."""
    if type(nc) is not int or type(nf) is not int or nc<1 or nf<0:
        raise ValueError('Invalid evolving toe checkpoint dimensions')
    shapes={key:(nf if key in ('is_toe_face','candidate_bank','candidate_toe') else nc)
            for key in CORE}
    shapes.update({key:nc+1 for key in ('toe_offset','donor_offset','candidate_offset','toe_donor_offset')})
    shapes.update(toe_faces=nf,donor_faces=nf,candidates=nc,toe_donors=nc)
    shapes.update({key:(nf if key in ('support_face_bank','support_face_toe') else nc)
                   for key in SUPPORT})
    result={key:((size,),np.dtype(np.float64 if key in ('distance','support_distance') else np.int32))
            for key,size in shapes.items()}
    result['initialized']=((),np.dtype(np.int32))
    return result


class ResidentEvolvingToeMap:
    def __init__(self,area,owner,neighbour,offset,faces,signs,length,xy,mid,
                 material,bulk,nlayer,*,dry_tolerance,source_path=None):
        import cupy as cp
        topology,area=validate_geometry(area,owner,neighbour,offset,faces,signs)
        owner,neighbour,offset,faces,signs=topology
        nc,nf=len(area),len(owner)
        xy=np.asarray(xy,dtype=np.float64);mid=np.asarray(mid,dtype=np.float64)
        length=np.asarray(length,dtype=np.float64);material=np.asarray(material)
        bulk=np.asarray(bulk,dtype=np.float64)
        if (xy.shape!=(2,nc) or mid.shape!=(2,nf) or length.shape!=(nf,) or
            not all(np.isfinite(v).all() for v in (xy,mid,length,bulk)) or
            np.any(length<=0) or bulk.ndim!=1 or not 1<=len(bulk)<=32 or np.any(bulk<=0) or
            not isinstance(nlayer,(int,np.integer)) or not 1<=nlayer<=33 or
            material.shape!=(nc,) or material.dtype.kind not in 'biu' or
            not np.isin(material,(0,1)).all() or
            not np.isfinite(dry_tolerance) or dry_tolerance<0):
            raise ValueError('Invalid evolving toe geometry/material controls')
        edge=np.zeros(nf)
        use=neighbour>=0
        edge[use]=(np.linalg.norm(mid[:,use]-xy[:,owner[use]],axis=0)+
                   np.linalg.norm(mid[:,use]-xy[:,neighbour[use]],axis=0))
        if not np.isfinite(edge).all() or np.any(edge[use]<=0):
            raise ValueError('Invalid evolving toe adjacency metric')
        source=Path(source_path or HERE/'evolving_toe_map.cu').read_text()
        if '#define BRAIDS_EVOLVING_TOE_ABI 2' not in source:
            raise ValueError('Evolving toe Python/CUDA ABI mismatch')
        self.cp=cp;self.nc=nc;self.nf=nf;self.ng=len(bulk);self.nl=int(nlayer)
        self.dry=np.float64(dry_tolerance)
        self.owner,self.neighbour,self.offset,self.faces=(cp.asarray(v) for v in topology[:4])
        self.length=cp.asarray(np.ascontiguousarray(length));self.edge=cp.asarray(edge)
        self.area=cp.asarray(area);self.bulk=cp.asarray(np.ascontiguousarray(bulk))
        self.material=cp.asarray(material,dtype=cp.int32)
        self.ids=cp.arange(nc,dtype=cp.int32);self.face_ids=cp.arange(nf,dtype=cp.int32)
        names=('tm_initial','tm_reach','tm_bank','tm_faces','tm_seed','tm_relax',
               'tm_support_faces','tm_support_seed','tm_support_live',
               'tm_changed_support','tm_retire')
        module=cp.RawModule(code=source,options=OPTIONS,name_expressions=names)
        self.k={n:module.get_function(n) for n in names}
        self.accepted={key:cp.zeros(shape,dtype=dtype)
                       for key,(shape,dtype) in checkpoint_specs(nc,nf).items()}
        self.trial={key:cp.zeros_like(v) for key,v in self.accepted.items()}
        self.work={key:cp.empty(nc,cp.float64 if key=='distance' else cp.int32)
                   for key in ('channel','mapped_donor','mapped_toe','mapped_face','distance','ambiguous')}
        self.inventory=cp.empty(nc,cp.int32)
        self.status_cell=cp.zeros(nc,cp.int32);self.status_face=cp.zeros(nf,cp.int32)
        self.failure=cp.zeros((),cp.int32);self.changed=cp.zeros((),cp.int32)
        self.tolerance=cp.zeros((),cp.float64)
        self.built=cp.zeros((),cp.int32)
        self.changed_roots=cp.zeros(nc,cp.int32)
        self.support_recipient=cp.zeros(nc,cp.int32)
        self.sweeps=cp.zeros(3,cp.int32)
        self._private_history=None;self._accepted_history=None

    def _launch(self,name,n,args):
        if n:self.k[name](((n+127)//128,),(128,),args)

    def _array(self,value,shape):
        if (not isinstance(value,self.cp.ndarray) or value.dtype!=self.cp.float64 or
            value.shape!=shape or not value.flags.c_contiguous or
            value.device.id!=self.material.device.id):
            raise ValueError('Expected resident contiguous FP64 evolving toe state')

    @staticmethod
    def _span(array):
        return int(array.data.ptr),int(array.data.ptr)+int(array.nbytes)

    @staticmethod
    def _overlap(a,b):return a[0]<b[1] and b[0]<a[1]

    def _storage(self):
        arrays={group+'.'+k:v for group in ('accepted','trial','work')
                for k,v in getattr(self,group).items()}
        arrays.update({k:getattr(self,k) for k in ('inventory','status_cell','status_face',
            'failure','changed','tolerance','built','changed_roots','support_recipient','sweeps')})
        for value in arrays.values():
            if not isinstance(value,self.cp.ndarray) or not value.flags.c_contiguous:
                raise ValueError('Evolving toe storage must be contiguous device arrays')
        ordered=sorted((self._span(v),k) for k,v in arrays.items() if v.nbytes)
        for (a,aname),(b,bname) in zip(ordered,ordered[1:]):
            if self._overlap(a,b):raise ValueError('Evolving toe private buffers overlap: '+aname+'/'+bname)
        return arrays

    def _guard(self,arrays):
        storage=self._storage()
        for name,value in arrays.items():
            span=self._span(value)
            for target,other in storage.items():
                if self._overlap(span,self._span(other)):
                    raise ValueError('Evolving toe input overlaps map storage: '+name+'/'+target)

    def _csr(self,keys,ids,offset,items,secondary=None):
        cp=self.cp
        key=cp.where(keys>=0,keys,self.nc).astype(cp.int32)
        # CuPy's API requires a resident 2-D key array, not NumPy's tuple form.
        order=cp.lexsort(cp.stack((ids,key) if secondary is None else (ids,secondary,key)))
        cp.copyto(items,ids[order])
        counts=cp.bincount(key,minlength=self.nc+1)[:self.nc]
        offset[0]=0;offset[1:]=cp.cumsum(counts,dtype=cp.int32)

    def build(self,bed,depth,layers):
        """Compute map_(n+1) on private fixed-size storage; no accepted mutation."""
        return self._build(bed,depth,layers,self.accepted)

    def build_remapped_support(self,bed,depth,layers,face_bank,face_toe):
        """Rebuild an AMR candidate from explicitly remapped dormant interfaces.

        Local ownership/CSR indices are derived afresh, never interpolated.
        This does not retire or install history: the AMR transaction separately
        transfers the old interface-width memory after checking face coverage.
        """
        cp=self.cp
        for value in (face_bank,face_toe):
            if (not isinstance(value,cp.ndarray) or value.shape!=(self.nf,) or
                value.dtype!=cp.int32 or not value.flags.c_contiguous or
                value.device.id!=self.material.device.id or
                bool(cp.any((value < -1)|(value >= self.nc)))):
                raise ValueError('Invalid remapped support endpoint IDs')
        self._guard(dict(face_bank=face_bank,face_toe=face_toe))
        active=face_bank>=0
        valid=((self.owner==face_bank)&(self.neighbour==face_toe))|\
              ((self.owner==face_toe)&(self.neighbour==face_bank))
        if (bool(cp.any(active!=(face_toe>=0))) or
            bool(cp.any(active&((self.neighbour<0)|~valid)))):
            raise ValueError('Remapped support does not match target face geometry')
        seed={key:cp.full(self.nc,-1,cp.int32) for key in
              ('support_donor','support_toe','support_face')}
        seed.update(initialized=cp.ones((),cp.int32),
                    support_face_bank=face_bank,support_face_toe=face_toe)
        return self._build(bed,depth,layers,seed)

    def _relax(self,m,bank,fields,root_key,bed,slot):
        cp=self.cp;nc=np.int32(self.nc)
        for iteration in range(self.nc):
            self.changed.fill(0)
            self._launch('tm_relax',self.nc,(nc,bank,m[root_key],self.offset,self.faces,
                self.owner,self.neighbour,bed,self.length,self.edge,self.tolerance,
                *(m[k] for k in fields),*(self.work[k] for k in
                    ('mapped_donor','mapped_toe','mapped_face','distance','ambiguous')),
                self.changed,self.status_cell))
            cp.maximum(self.failure,cp.max(self.status_cell),out=self.failure)
            for key,work in zip(fields,('mapped_donor','mapped_toe','mapped_face','distance','ambiguous')):
                cp.copyto(m[key],self.work[work])
            if int(self.changed.item())==0:break
        else:self.failure.fill(1)
        self.sweeps[slot]=iteration+1

    def _build(self,bed,depth,layers,support):
        cp=self.cp;nc=np.int32(self.nc);nf=np.int32(self.nf)
        self._array(bed,(self.nc,));self._array(depth,(self.nc,))
        self._array(layers,(self.nl,self.ng,self.nc))
        self._guard(dict(bed=bed,depth=depth,layers=layers))
        m=self.trial;w=self.work
        self.failure.fill(0);self.built.fill(0);self.changed_roots.fill(0)
        self.tolerance[...] = 512.*np.finfo(float).eps*cp.maximum(1.,cp.max(cp.abs(bed)))
        self._launch('tm_initial',self.nc,(nc,np.int32(self.ng),np.int32(self.nl),
            self.material,bed,depth,layers,self.dry,m['channel'],self.inventory,self.status_cell))
        cp.maximum(self.failure,cp.max(self.status_cell),out=self.failure)
        for iteration in range(self.nc):
            self.changed.fill(0)
            self._launch('tm_reach',self.nc,(nc,self.material,self.offset,self.faces,
                self.owner,self.neighbour,bed,depth,self.dry,self.tolerance,
                m['channel'],w['channel'],self.changed,self.status_cell))
            cp.maximum(self.failure,cp.max(self.status_cell),out=self.failure)
            cp.copyto(m['channel'],w['channel'])
            if int(self.changed.item())==0:break
        else:self.failure.fill(1)
        self.sweeps[0]=iteration+1
        self._launch('tm_bank',self.nc,(nc,self.material,m['channel'],self.inventory,m['bank']))
        self._launch('tm_faces',self.nf,(nf,self.owner,self.neighbour,m['bank'],m['channel'],
            bed,m['is_toe_face'],m['candidate_bank'],m['candidate_toe'],self.status_face))
        if self.nf:cp.maximum(self.failure,cp.max(self.status_face),out=self.failure)
        self._launch('tm_seed',self.nc,(nc,self.offset,self.faces,m['candidate_bank'],m['candidate_toe'],
            self.edge,bed,self.length,m['mapped_donor'],m['mapped_toe'],m['mapped_face'],m['distance'],
            m['is_toe_donor'],m['is_toe_cell'],m['ambiguous']))
        fields=('mapped_donor','mapped_toe','mapped_face','distance','ambiguous')
        self._relax(m,m['bank'],fields,'is_toe_donor',bed,1)
        self._csr(m['candidate_toe'],self.face_ids,m['toe_offset'],m['toe_faces'])
        self._csr(m['candidate_bank'],self.face_ids,m['donor_offset'],m['donor_faces'])
        self._csr(m['mapped_donor'],self.ids,m['candidate_offset'],m['candidates'],m['distance'])
        roots=cp.where(m['is_toe_donor']!=0,m['mapped_toe'],-1)
        self._csr(roots,self.ids,m['toe_donor_offset'],m['toe_donors'])
        # Support is a separate geometric identity. Old still-present material
        # interfaces may be dormant when disconnected/dry, but NEVER enter the
        # active face masks used for erosion or sediment diagnostics.
        self._launch('tm_support_faces',self.nf,(nf,self.owner,self.neighbour,m['bank'],
            bed,m['candidate_bank'],m['candidate_toe'],m['mapped_donor'],support['initialized'],
            support['support_face_bank'],support['support_face_toe'],
            m['support_face_bank'],m['support_face_toe'],self.status_face))
        if self.nf:cp.maximum(self.failure,cp.max(self.status_face),out=self.failure)
        support_fields=('support_donor','support_toe','support_face','support_distance','support_ambiguous')
        self._launch('tm_seed',self.nc,(nc,self.offset,self.faces,m['support_face_bank'],
            m['support_face_toe'],self.edge,bed,self.length,
            *(m[k] for k in support_fields[:4]),m['support_root'],self.support_recipient,
            m['support_ambiguous']))
        self._launch('tm_support_seed',self.nc,(nc,m['is_toe_donor'],
            *(m[k] for k in fields),*(m[k] for k in support_fields),m['support_root']))
        self._relax(m,m['bank'],support_fields,'support_root',bed,2)
        m['support_live'].fill(0)
        self._launch('tm_support_live',self.nc,(nc,m['support_donor'],m['support_toe'],
            self.inventory,bed,depth,self.dry,m['support_live'],self.status_cell))
        cp.maximum(self.failure,cp.max(self.status_cell),out=self.failure)
        self._launch('tm_changed_support',max(self.nc,self.nf),(nc,nf,support['initialized'],
            *(support[k] for k in ('support_donor','support_toe','support_face')),
            *(m[k] for k in ('support_donor','support_toe','support_face')),
            support['support_face_bank'],support['support_face_toe'],
            m['support_face_bank'],m['support_face_toe'],m['support_live'],self.changed_roots))
        m['initialized'].fill(1);self.built.fill(1)
        return dict(mapping=m,status=self.failure)

    def retire_changed_history(self,history):
        """Retire unsupported geometric state only; cumulative ledgers remain."""
        if self._private_history is None:raise ValueError('Register private history before map retirement')
        for key in GEOMETRIC_HISTORY:
            self._array(history[key],(self.nc,))
            if self._span(history[key])!=self._span(self._private_history[key]):
                raise ValueError('Retirement requires registered private history')
        self._validate_history_storage(self._private_history,self._accepted_history)
        if int(self.built.item())!=1:raise ValueError('Build a trial map before history retirement')
        self._launch('tm_retire',self.nc,(np.int32(self.nc),self.changed_roots,
            *(history[k] for k in GEOMETRIC_HISTORY)))

    def commit(self,global_status):
        cp=self.cp
        if not isinstance(global_status,cp.ndarray) or global_status.shape!=() or global_status.dtype!=cp.int32:
            raise ValueError('Expected resident map commit status')
        success=(global_status==0)&(self.failure==0)&(self.built==1)
        for key in self.accepted:cp.copyto(self.accepted[key],cp.where(success,self.trial[key],self.accepted[key]))

    def bind(self,toe,bank,diagnostic,accepted_history=None):
        """Bind stable accepted-map buffers once; never recreate GPU operators."""
        if int(self.accepted['initialized'].item())!=1:
            raise ValueError('Cannot bind an uninitialized evolving toe map')
        for operator in (toe,bank):
            if any(getattr(operator,key,None)!=getattr(self,key) for key in ('nc','nf','ng','nl')):
                raise ValueError('Evolving toe operator dimensions differ')
        if getattr(diagnostic,'nf',None)!=self.nf or getattr(diagnostic,'ng',None)!=self.ng:
            raise ValueError('Evolving toe diagnostic dimensions differ')
        for operator in (toe,bank,diagnostic):
            if getattr(operator,'cp',None) is not self.cp:raise ValueError('Evolving toe device backend differs')
        # Equal dimensions alone do not establish compatible topology/material.
        for operator,keys in ((toe,('area','bulk','owner','neighbour','offset','faces','length')),
                              (bank,('area','bulk','length')),
                              (diagnostic,('owner','neighbour','length'))):
            for key in keys:
                value=getattr(operator,key,None);expected=getattr(self,key)
                if (not isinstance(value,self.cp.ndarray) or value.dtype!=expected.dtype or
                    value.shape!=expected.shape or not value.flags.c_contiguous or
                    value.device.id!=expected.device.id or
                    not bool(self.cp.array_equal(value,expected).item())):
                    raise ValueError('Evolving toe bound geometry differs: '+key)
        if getattr(bank,'dry',None)!=self.dry:raise ValueError('Evolving toe dry tolerance differs')
        self._storage()
        if accepted_history is not None:self.register_private_history(bank.trial_history,accepted_history)
        m=self.accepted
        for name in ('toe_offset','toe_faces','donor_offset','donor_faces','candidate_bank','candidate_toe'):
            setattr(toe,name,m[name])
        toe.toe_cell=m['is_toe_cell']
        for name,key in (('candidate_toe','candidate_toe'),('is_donor','is_toe_donor'),
            ('toe_cell','mapped_toe'),('donor_face_offset','donor_offset'),('donor_faces','donor_faces'),
            ('candidate_offset','candidate_offset'),('candidates','candidates'),
            ('toe_donor_offset','toe_donor_offset'),('toe_donors','toe_donors')):
            setattr(bank,name,m[key])
        for name in ('is_toe_face','candidate_bank','candidate_toe'):setattr(diagnostic,name,m[name])

    def _validate_history_storage(self,private,accepted):
        arrays={}
        for group,history in (('private',private),('accepted',accepted)):
            if not isinstance(history,dict) or not all(k in history for k in GEOMETRIC_HISTORY):
                raise ValueError('Complete geometric history registration required')
            for key,value in history.items():
                dtype=self.cp.int32 if key=='collapse_count' else self.cp.float64
                if (not isinstance(value,self.cp.ndarray) or value.shape!=(self.nc,) or
                    value.dtype!=dtype or not value.flags.c_contiguous):
                    raise ValueError('Invalid registered history storage')
                arrays[group+'.'+key]=value
        self._guard(arrays)
        order=sorted((self._span(v),k) for k,v in arrays.items())
        for (a,aname),(b,bname) in zip(order,order[1:]):
            if self._overlap(a,b):raise ValueError('Registered histories overlap: '+aname+'/'+bname)

    def register_private_history(self,private_history,accepted_history):
        self._validate_history_storage(private_history,accepted_history)
        self._private_history=dict(private_history);self._accepted_history=dict(accepted_history)

    def persistent_arrays(self):
        """Stable accepted device fields; includes dormant support and readiness."""
        return dict(self.accepted)

    def restore(self,fields,bed,depth,layers):
        """Validate checkpoint map+support by canonical rebuild, then commit.

        Supplied checkpoint data is never installed as accepted state before
        braids_support. Dormant support is explicit checkpoint state, not inferred
        from a freshly wet-only map that would lose dry/rewet continuity.
        """
        cp=self.cp;specs=checkpoint_specs(self.nc,self.nf)
        if set(fields)!=set(specs):raise ValueError('Evolving toe checkpoint field inventory differs')
        temporary={}
        for key,(shape,dtype) in specs.items():
            value=fields[key]
            if not isinstance(value,(np.ndarray,cp.ndarray)) or value.shape!=shape or value.dtype!=dtype:
                raise ValueError('Evolving toe checkpoint shape/dtype differs: '+key)
            temporary[key]=cp.array(value,copy=True,order='C')
        if int(temporary['initialized'].item())!=1:
            raise ValueError('Evolving toe checkpoint is not initialized')
        # Check IDs BEFORE kernels use checkpoint support indices.
        for key in ('support_donor','support_toe','support_face_bank','support_face_toe'):
            if bool(cp.any((temporary[key]<-1)|(temporary[key]>=self.nc)).item()):
                raise ValueError('Invalid checkpoint support cell IDs')
        if bool(cp.any((temporary['support_face']<-1)|(temporary['support_face']>=self.nf)).item()):
            raise ValueError('Invalid checkpoint support face IDs')
        result=self._build(bed,depth,layers,temporary)
        if int(result['status'].item())!=0:
            raise ValueError('Checkpoint physical state cannot rebuild evolving toe map')
        different=[key for key in specs if not bool(cp.array_equal(temporary[key],self.trial[key]).item())]
        if different:raise ValueError('Evolving toe checkpoint/state mismatch: '+','.join(different))
        self.commit(self.failure)
        return self.failure
