"""Resident GPU numerical tags for the general AMR selector.

Only integer topology/delay metadata and compact diagnostics are downloaded.
The immutable geometry and final balanced leaf planning may live on the host;
accepted and cached solution arrays never do. No CPU numerical fallback exists.
"""
from pathlib import Path
import hashlib
import numpy as np

OPTIONS=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true','--ftz=false')


class ResidentGeneralIndicators:
    def __init__(self,selector,*,source_path=None,record_diagnostics=False):
        import cupy as cp
        self.cp,self.s=cp,selector
        if type(record_diagnostics) is not bool:
            raise ValueError('Indicator diagnostic opt-in must be boolean')
        self.record_diagnostics=record_diagnostics
        source=Path(source_path or Path(__file__).with_suffix('.cu')).read_bytes()
        if b'#define BRAIDS_AMR_GENERAL_INDICATORS_ABI 2' not in source:
            raise ValueError('AMR numerical indicator kernel ABI differs')
        self.source_sha256=hashlib.sha256(source).hexdigest()
        self.module=cp.RawModule(code=source.decode(),options=OPTIONS)
        self.kernel={name:self.module.get_function(name) for name in
                     ('gi_restrict','gi_faces','gi_emerge','gi_reach','gi_veto',
                      'gi_tree_bed','gi_sediment_primitives','gi_cells',
                      'gi_ancestor_detail','gi_ancestor_causes')}
        s=selector
        self.children=np.full((s.nn,4),-1,np.int32)
        for node,group in enumerate(s.children):
            if group:self.children[node]=group
        self.dchildren=cp.asarray(self.children)
        self.area=cp.asarray(s.area)
        self.rock=cp.asarray(s.rock)
        self.level=cp.asarray(s.level,dtype=cp.int32)
        self.minimum=cp.asarray(s.minimum,dtype=cp.int32)
        self.maximum=cp.asarray(s.maximum,dtype=cp.int32)
        self.parent=cp.asarray(s.parent,dtype=cp.int32)
        self.xy=cp.ascontiguousarray(cp.asarray(s.xy,dtype=cp.float64))
        self.level_nodes=[np.flatnonzero(s.level==level).astype(np.int32)
                          for level in range(s.levels)]
        self.edges=[(cp.asarray(nodes),cp.asarray(s.parent[nodes],dtype=cp.int32))
                    for nodes in self.level_nodes[1:]]
        self.geometry={}
        self.last_diagnostics=None

    def _launch(self,name,n,args):
        if n:self.kernel[name](((n+127)//128,),(128,),args)

    def _array(self,a,shape,name):
        cp=self.cp
        if (not isinstance(a,cp.ndarray) or a.shape!=shape or a.dtype!=cp.float64 or
            not a.flags.c_contiguous or a.device.id!=cp.cuda.Device().id):
            raise ValueError('AMR GPU selector requires resident contiguous FP64 '+name)
        if not bool(cp.all(cp.isfinite(a)).item()):raise ValueError('Nonfinite AMR '+name)

    def _geometry(self,top):
        cp,s=self.cp,self.s
        key=id(top)
        if key in self.geometry:return self.geometry[key]
        data=top.data;ids=np.asarray(top.node_ids,np.int64);n=len(ids)
        if getattr(s,'extended_indicators',False):
            from braids_gpu.amr_indicator_reference import directed_geometry
            directed_geometry(data,n)  # Immutable host metadata only.
            bi=np.asarray(data['bed_indices']);bulk=np.asarray(data['bulk_fraction'])
            si=np.asarray(data['suspended_indices']);ng=int(data['nfrac']);ns=int(data['nstate'])
            if (bi.ndim!=2 or not bi.size or bi.shape[1]!=ng or bi.dtype.kind not in 'iu' or
                np.any((bi<0)|(bi>=ns)) or si.shape!=(ng,) or si.dtype.kind not in 'iu' or
                np.any((si<0)|(si>=ns)) or bulk.shape!=(ng,) or bulk.dtype!=np.float64 or
                not np.isfinite(bulk).all() or np.any(bulk<=0) or
                s.xy.shape!=(s.nn,2) or not np.isfinite(s.xy).all() or
                s.area.shape!=(s.nn,) or not np.isfinite(s.area).all() or np.any(s.area<=0)):
                raise ValueError('Invalid AMR extended indicator geometry')
        adj=[[] for _ in ids]
        for a,b in zip(data['owner'],data['neighbour']):
            if b>=0:
                distance=float(np.linalg.norm(s.xy[ids[a]]-s.xy[ids[b]]))
                adj[a].append((int(b),distance));adj[b].append((int(a),distance))
        local={int(node):i for i,node in enumerate(ids)}
        parents=[p for p,group in enumerate(s.children) if len(group)==4 and
                 all(child in local for child in group)]
        cover=s._cover(ids)
        restrict=[cp.asarray(nodes[(cover[nodes]<0)&(self.children[nodes,0]>=0)])
                  for nodes in self.level_nodes[:-1]]
        bed_index=np.asarray(data['bed_indices'],np.int32)
        result=dict(ids=cp.asarray(ids,dtype=cp.int32),cover=cover,restrict=restrict,
            owner=cp.asarray(data['owner'],dtype=cp.int32),
            neighbour=cp.asarray(data['neighbour'],dtype=cp.int32),
            bid=cp.asarray(data['bid'],dtype=cp.int32),
            material=cp.asarray(data.get('bank_material_mask',
                np.zeros(n, dtype=np.int32)),dtype=cp.int32),
            manning=cp.asarray(data['manning_cell'],dtype=cp.float64),
            offset=cp.asarray(np.r_[0,np.cumsum([len(row) for row in adj])],dtype=cp.int32),
            adjacent=cp.asarray([j for row in adj for j,_ in row],dtype=cp.int32),
            distance=cp.asarray([d for row in adj for _,d in row],dtype=cp.float64),
            parents=parents,groups=cp.asarray([[local[c] for c in s.children[p]]
                for p in parents],dtype=cp.int32).reshape((-1,4)),
            bed_index=cp.asarray(bed_index.ravel()),
            bulk=cp.asarray(np.tile(np.asarray(data['bulk_fraction']),len(bed_index))),
            sediment=cp.asarray(np.r_[np.asarray(data['suspended_indices']).ravel(),
                bed_index.ravel()],dtype=cp.int32))
        if getattr(s,'extended_indicators',False):
            result.update(cell_offset=cp.asarray(data['offset'],dtype=cp.int32),
                cell_faces=cp.asarray(data['faces'],dtype=cp.int32),
                cell_signs=cp.asarray(data['signs'],dtype=cp.int32),
                suspended=cp.asarray(data['suspended_indices'],dtype=cp.int32),
                active=cp.asarray(bed_index[0],dtype=cp.int32),
                class_bulk=cp.asarray(data['bulk_fraction'],dtype=cp.float64))
            if s.settings['refine_bathymetry']:
                # Immutable topology only. Each ancestor gathers its current
                # descendants in original leaf order; never atomically sum FP.
                descendants=[[] for _ in range(s.nn)]
                for i,node in enumerate(ids):
                    ancestor=int(s.parent[node])
                    while ancestor>=0:
                        descendants[ancestor].append(i)
                        ancestor=int(s.parent[ancestor])
                result.update(ancestor_offset=cp.asarray(
                    np.r_[0,np.cumsum([len(row) for row in descendants])],dtype=cp.int32),
                    descendants=cp.asarray([i for row in descendants for i in row],dtype=cp.int32))
        self.geometry[key]=result
        return result

    def _extended_tags(self,top,accepted,rows,bed,wet,raw,cause,moving_source,
                       *,selector_last_bed,bed_gradient):
        """Private auxiliary indicators, with no accepted-state writes/copies."""
        cp,s=self.cp,self.s;data=top.data;g=self._geometry(top)
        n,ns,ng=len(top.node_ids),int(data['nstate']),int(data['nfrac'])
        sediment=bool(s.settings['refine_sediment'])
        bathy=bool(s.settings['refine_bathymetry'])
        status=cp.zeros((),cp.int32);near=cp.zeros(n,cp.int32)
        raised=cp.zeros(s.nn,cp.int32)
        # Unused pointer arguments get private device storage, not host data.
        dummy=cp.zeros(1,cp.float64)
        concentration=fraction=volume=dummy
        if sediment:
            concentration=cp.empty((n,ng),cp.float64)
            fraction=cp.empty_like(concentration);volume=cp.empty(n,cp.float64)
            self._launch('gi_sediment_primitives',n,(np.int32(n),np.int32(ns),np.int32(ng),
                g['suspended'],g['active'],g['class_bulk'],accepted,np.float64(data['dry_tolerance']),
                concentration,fraction,volume,status))
        tree_bed=dummy
        if bathy:
            tree_bed=cp.empty(s.nn,cp.float64)
            self._launch('gi_tree_bed',s.nn,(np.int32(s.nn),np.int32(ns),
                np.int32(len(g['bed_index'])),g['bed_index'],g['bulk'],self.rock,rows,tree_bed,status))
        self._launch('gi_cells',n,(np.int32(n),np.int32(ng),g['ids'],g['cell_offset'],
            g['cell_faces'],g['cell_signs'],g['owner'],g['neighbour'],g['bid'],g['material'],
            self.dchildren,self.xy,bed,tree_bed,wet,concentration,fraction,volume,
            selector_last_bed if sediment else dummy,bed_gradient if bathy else dummy,
            np.int32(s.settings['refine_walls']),np.int32(sediment),np.int32(bathy),
            np.int32(s.levels-1),np.float64(s.c_tolerance if sediment else 0.),
            np.float64(s.mix_tolerance if sediment else 0.),np.float64(s.dz_tolerance if sediment else 0.),
            np.float64(s.bed_tolerance if bathy else 0.),raw,cause,moving_source,near,status))
        if bathy:
            self._launch('gi_ancestor_detail',s.nn,(np.int32(s.nn),g['ancestor_offset'],
                g['descendants'],g['ids'],self.area,self.xy,bed,tree_bed,near,self.level,
                np.float64(s.bed_tolerance),raw,raised,status))
            self._launch('gi_ancestor_causes',n,(np.int32(n),g['ids'],self.parent,raised,cause))
        if int(status.item()):raise ValueError('Nonfinite or overflowing AMR extended indicator')
        return dict(near_water=near,ancestor_raised=raised)

    def propose(self,top,accepted,cached,delay_state,*,time_s,initial,
                selector_last_bed=None,bed_gradient=None):
        cp,s=self.cp,self.s;data=top.data;n=len(top.node_ids);ns=int(data['nstate'])
        self._array(accepted,(n,ns),'accepted state')
        self._array(cached,(s.nn,ns),'cached tree state')
        extended=getattr(s,'extended_indicators',False)
        if extended and s.settings['refine_sediment']:
            self._array(selector_last_bed,(s.nn,),'last bed reference')
        if extended and s.settings['refine_bathymetry']:
            self._array(bed_gradient,(n,2),'accepted limited bed gradient')
        if not np.isfinite(time_s) or time_s<0 or abs(time_s/s.interval-round(time_s/s.interval))>1e-9:
            raise ValueError('AMR accepted clock/delay differs')
        if not isinstance(delay_state,(np.ndarray,cp.ndarray)) or delay_state.shape!=(s.nn,) or delay_state.dtype.kind not in 'iu':
            raise ValueError('Invalid AMR selector delay counters')
        delay=cp.asarray(delay_state)
        if bool(cp.any(delay<0).item()) or bool(cp.any(delay>np.iinfo(np.int64).max).item()):
            raise ValueError('Invalid AMR selector delay counters')
        delay=delay.astype(cp.int64,copy=True)
        g=self._geometry(top);ids=g['ids'];finest=np.int32(s.levels-1)
        if bool(cp.any(accepted[:,0]<-1e-12).item()) or bool(cp.any(accepted[:,g['sediment']]<-1e-12).item()):
            raise ValueError('Invalid AMR accepted water/sediment')
        rows=cached.copy();rows[ids]=accepted
        for nodes in reversed(g['restrict']):
            self._launch('gi_restrict',len(nodes)*ns,(np.int32(len(nodes)),np.int32(ns),
                nodes,self.dchildren,self.area,rows))
        bed=self.rock[ids]+cp.sum(accepted[:,g['bed_index']]/g['bulk'][None,:],axis=1)
        depth=cp.ascontiguousarray(accepted[:,0]);eta=bed+depth;wet=(depth>s.wet_depth).astype(cp.int32)
        moving=depth>float(data['dry_tolerance'])
        safe=cp.where(moving,depth,1.)
        u=cp.where(moving,accepted[:,1]/safe,0.)
        v=cp.where(moving,accepted[:,2]/safe,0.)
        if not bool(cp.all(cp.isfinite(bed)&cp.isfinite(eta)&cp.isfinite(u)&cp.isfinite(v)).item()):
            raise ValueError('Nonfinite AMR hydraulic indicator')
        raw=self.minimum.copy()
        raw[ids]=cp.maximum(raw[ids],cp.where(wet,np.int32(s.settings['wet_level']),0))
        cause=wet*(1<<1);front=cp.zeros(n,cp.int32);moving_source=cp.zeros(n,cp.int32)
        self._launch('gi_faces',len(g['owner']),(np.int32(len(g['owner'])),g['owner'],
            g['neighbour'],ids,wet,eta,u,v,g['manning'],g['material'],g['bid'],finest,
            np.float64(s.eta_tolerance),np.float64(s.velocity_tolerance),raw,cause,front,moving_source))
        self._launch('gi_emerge',n,(np.int32(n),np.int32(ns),np.int32(len(g['bed_index'])),
            ids,self.dchildren,g['bed_index'],g['bulk'],self.rock,rows,wet,eta,
            np.float64(s.wet_depth),finest,raw,cause,moving_source))
        if extended:
            diagnostics=self._extended_tags(top,accepted,rows,bed,wet,raw,cause,moving_source,
                selector_last_bed=selector_last_bed,bed_gradient=bed_gradient)
            if self.record_diagnostics:
                self.last_diagnostics=dict(raw=raw.copy(),cause=cause.copy(),
                    moving_source=moving_source.copy(),**diagnostics)
        protected=s._protected(top)
        raw[cp.asarray(sorted(protected),dtype=cp.int32)]=finest
        moving_source[cp.asarray(s._inlet_owners(top),dtype=cp.int32)]=1
        speed=cp.where((moving_source!=0)&(wet!=0),cp.sqrt(u*u+v*v)+
            cp.where(front!=0,2.,1.)*cp.sqrt(float(data['grav'])*cp.maximum(depth,0.)),0.)
        if not bool(cp.all(cp.isfinite(speed)).item()):raise ValueError('AMR forecast speed overflow')
        buffer=s.settings['buffer_cells']*max(float(data['dx']),float(data['dy']))
        reach=cp.where(raw[ids]==finest,buffer+speed*s.interval,-cp.inf)
        following=cp.empty_like(reach);changed=cp.zeros((),cp.int32)
        for _ in range(n):
            changed.fill(0)
            self._launch('gi_reach',n,(np.int32(n),g['offset'],g['adjacent'],g['distance'],reach,following,changed))
            reach,following=following,reach
            if int(changed.item())==0:break
        else:raise RuntimeError('AMR forecast buffering did not converge')
        buffered=reach>=0;raw[ids[buffered]]=finest;cause[buffered]|=1<<14
        npg=len(g['parents']);veto=cp.zeros(npg,cp.int32)
        mixed=cp.zeros_like(veto);material=cp.zeros_like(veto);span=cp.zeros(npg,cp.float64)
        self._launch('gi_veto',npg,(np.int32(npg),g['groups'],ids,self.level,depth,eta,
            g['material'],np.float64(data['dry_tolerance']),np.float64(s.stage_span),finest,
            np.int32(s.evolving_banks),raw,veto,mixed,material,span))
        ix=cp.flatnonzero(veto)
        hi=cp.asnumpy(ix);hm=cp.asnumpy(mixed[ix]);ht=cp.asnumpy(material[ix]);hs=cp.asnumpy(span[ix])
        vetoed=[]
        for i,m,t,h in zip(hi,hm,ht,hs):
            entry=dict(parent_id=g['parents'][int(i)],mixed_solver_wet_dry=bool(m),wet_child_stage_span_m=float(h))
            if s.evolving_banks:entry['mixed_material']=bool(t)
            vetoed.append(entry)
        if not initial:
            reset=raw[ids]>=self.level[ids]
            old_delay=delay[ids]
            if bool(cp.any((~reset)&(old_delay==np.iinfo(np.int64).max)).item()):
                raise ValueError('AMR coarsen delay overflow')
            updated=cp.where(reset,0,old_delay+1);delay[ids]=updated
            raw[ids]=cp.where((~reset)&(updated<s.settings['coarsen_delay']),self.level[ids],raw[ids])
        wanted=raw.copy()
        for child,parent in self.edges:wanted[child]=cp.maximum(wanted[child],wanted[parent])
        for child,parent in reversed(self.edges):cp.maximum.at(wanted,parent,wanted[child])
        wanted=cp.minimum(wanted,self.maximum)
        stats={name:int(cp.count_nonzero(cause&bit).item()) for name,bit in (
            ('wet_dry_trigger_cells',1<<2),('hydraulic_trigger_cells',1<<3),
            ('wet_child_bed_trigger_cells',1<<12),('open_boundary_trigger_cells',1<<8))}
        stats['forecast_buffer_cells']=int(cp.count_nonzero(buffered).item())
        if extended:
            for name,bit in s.INDICATOR_STATS.items():
                stats[name]=int(cp.count_nonzero(cause&(1<<bit)).item())
            if self.record_diagnostics:
                self.last_diagnostics.update(final_raw=raw.copy(),final_cause=cause.copy())
        return dict(wanted=cp.asnumpy(wanted).astype(np.int64),delay=cp.asnumpy(delay),
            protected=protected,vetoed=vetoed,stats=stats)
