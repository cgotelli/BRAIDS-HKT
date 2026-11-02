"""Resident CUDA curvature filter: unchanged operator, multilevel PCG sandbox.

Geometry hierarchy is prepared once on the CPU. All depth-dependent
coefficients, V cycles and Krylov vectors remain on-device. Only convergence
scalars are read each iteration. This is not integrated in a coupled solver.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OPTIONS=("--std=c++11","--fmad=false","--prec-div=true","--prec-sqrt=true","--ftz=false")


def grouped(values):
    table={};groups=[];unique=[]
    for i,value in enumerate(values):
        key=tuple(value)
        if key not in table:table[key]=len(table);unique.append(i)
        groups.append(table[key])
    return np.array(groups,dtype=np.int32),np.array(unique,dtype=np.int32)


def inverse_groups(groups,count):
    groups=np.asarray(groups);valid=groups>=0
    row=np.r_[0,np.cumsum(np.bincount(groups[valid],minlength=count))].astype(np.int32)
    child=np.flatnonzero(valid)[np.argsort(groups[valid],kind='stable')].astype(np.int32)
    return row,child


def adjacency(owner,neighbour,nc):
    valid=np.flatnonzero(neighbour>=0);cells=np.r_[owner[valid],neighbour[valid]]
    faces=np.tile(valid,2);others=np.r_[neighbour[valid],owner[valid]]
    # Sorting by face first preserves the frozen Fortran per-cell face order.
    order=np.lexsort((faces,cells));row=np.r_[0,np.cumsum(np.bincount(cells,minlength=nc))]
    return row.astype(np.int32),faces[order].astype(np.int32),others[order].astype(np.int32)


def validate_geometry(data):
    nc=len(data['area']);nf=len(data['owner'])
    for key,shape in [('owner',(nf,)),('neighbour',(nf,)),('area',(nc,)),('xy',(2,nc)),('normal',(2,nf)),('length',(nf,))]:
        value=np.asarray(data[key])
        if value.shape!=shape or not np.isfinite(value).all():raise ValueError(f'Invalid {key}')
    if nc<1 or (np.asarray(data['area'])<=0).any() or (np.asarray(data['length'])<0).any():raise ValueError('Invalid mass/length')
    owner=np.asarray(data['owner']);other=np.asarray(data['neighbour'])
    if ((owner<0)|(owner>=nc)|(other< -1)|(other>=nc)|(owner==other)).any():raise ValueError('Invalid connectivity')
    if not np.equal(owner,np.floor(owner)).all() or not np.equal(other,np.floor(other)).all():raise ValueError('Non-integer connectivity')
    valid=other>=0;delta=data['xy'][:,other[valid]]-data['xy'][:,owner[valid]]
    if (np.sum(delta*delta,axis=0)<=0).any():raise ValueError('Coincident neighbour centroids')


def hierarchy(data):
    """Frozen Fortran grouping/order, including coarse mass-weighted centroids."""
    validate_geometry(data)
    first={k:np.array(data[k],copy=True) for k in ('owner','neighbour','xy')}
    first['mass']=np.array(data['area'],copy=True);levels=[first]
    origin=np.min(first['xy'],axis=1)-.5*np.sqrt(np.max(first['mass']))
    width=2*np.sqrt(np.max(first['mass']))
    while len(levels[-1]['mass'])>32:
        fine=levels[-1];nc=len(fine['mass'])
        for attempt in range(32):
            bins=np.floor((fine['xy'].T-origin)/width).astype(np.int64)
            parent,unique=grouped(bins);count=len(unique)
            if count<nc:break
            width*=2
        else:raise ValueError('Cannot coarsen disconnected geometry')
        fine['parent']=parent;fine['child_row'],fine['child']=inverse_groups(parent,count)
        coarse={'mass':np.zeros(count),'xy':np.zeros((2,count))}
        for c in range(count):
            for k in range(fine['child_row'][c],fine['child_row'][c+1]):
                j=fine['child'][k];coarse['mass'][c]+=fine['mass'][j]
                coarse['xy'][:,c]+=fine['mass'][j]*fine['xy'][:,j]
            coarse['xy'][:,c]/=coarse['mass'][c]
        own=parent[fine['owner']];other=np.full_like(own,-1);valid=fine['neighbour']>=0
        other[valid]=parent[fine['neighbour'][valid]];valid&=other!=own
        pairs=np.stack([np.minimum(own[valid],other[valid]),np.maximum(own[valid],other[valid])],axis=1)
        groups,unique=grouped(pairs);coarse['owner']=pairs[unique,0].astype(np.int32);coarse['neighbour']=pairs[unique,1].astype(np.int32)
        edge_map=np.full(len(own),-1,dtype=np.int32);edge_map[valid]=groups
        fine['edge_child_row'],fine['edge_child']=inverse_groups(edge_map,len(unique))
        levels.append(coarse);width*=2
        if len(levels)>32:raise ValueError('Too many curvature levels')
    for level in levels:
        level['row'],level['edge'],level['other']=adjacency(level['owner'],level['neighbour'],len(level['mass']))
        level['owner']=level['owner'].astype(np.int32);level['neighbour']=level['neighbour'].astype(np.int32)
    return levels


class ResidentCurvature:
    """One object per fixed mesh, no CPU input traces or filtered results."""
    def __init__(self,data,source_path=None,*,parallel_coarse=False):
        import cupy as cp
        self.cp=cp;host=hierarchy(data);self.levels=[];self.parallel_coarse=bool(parallel_coarse)
        names=('coeff','reduce_edges','diagonal','multiply','smooth','restrict','prolong','factor','coarse_solve','pcg_update','pcg_direction','convergence','factor_parallel','coarse_solve_parallel')
        self.module=cp.RawModule(code=Path(source_path or HERE/'secondary_curvature.cu').read_text(),options=OPTIONS,
                                 name_expressions=tuple('sc_'+n for n in names))
        self.kernels={n:self.module.get_function('sc_'+n) for n in names}
        for item in host:
            level={key:cp.asarray(value) for key,value in item.items() if key!='xy'}
            level['nc']=len(item['mass']);level['nf']=len(item['owner'])
            for key in ('diagonal','b','x','work','res'):level[key]=cp.empty(level['nc'],dtype=cp.float64)
            level['w']=cp.empty(level['nf'],dtype=cp.float64);self.levels.append(level)
        fine=host[0];nf=len(fine['owner']);valid=fine['neighbour']>=0
        delta=data['xy'][:,fine['neighbour'][valid]]-data['xy'][:,fine['owner'][valid]]
        distance=np.ones(nf);projection=np.zeros(nf)
        distance[valid]=np.sum(delta*delta,axis=0);projection[valid]=np.maximum(0,np.sum(delta*data['normal'][:,valid],axis=0))
        self.length=cp.asarray(data['length']);self.distance=cp.asarray(distance);self.projection=cp.asarray(projection)
        self.nc=len(fine['mass']);self.factor=cp.empty((host[-1]['mass'].size,)*2,dtype=cp.float64)
        self.status=cp.zeros(1,dtype=cp.int32)
        self.done=cp.zeros(1,dtype=cp.int32);self.count=cp.zeros(1,dtype=cp.int32)
        self.buffers={key:cp.empty(self.nc,dtype=cp.float64) for key in ('rhs','r','p','ap','filtered','true_residual','product')}
        self.scalar={key:cp.empty((),dtype=cp.float64) for key in ('rho','rho_next','denominator','residual2','tolerance')}
        self.graph=None;self.graph_stream=cp.cuda.Stream(non_blocking=True);self.graph_pool=cp.cuda.MemoryPool()

    def launch(self,name,count,args):
        self.kernels[name](((max(count,1)+127)//128,),(128,),tuple(args))

    def array(self,value):
        if not isinstance(value,self.cp.ndarray) or value.dtype!=self.cp.float64 or value.shape!=(self.nc,) or not value.flags.c_contiguous:
            raise ValueError('Expected contiguous resident FP64 cell array')

    def coefficients(self,h,dry=1.e-4,depth_factor=2.):
        self.array(h)
        if not np.isfinite([dry,depth_factor]).all() or dry<0 or depth_factor<0:raise ValueError('Invalid filter controls')
        fine=self.levels[0]
        self.launch('coeff',fine['nf'],(np.int32(fine['nf']),fine['owner'],fine['neighbour'],h,self.length,self.projection,
            self.distance,np.float64(dry),np.float64(depth_factor),fine['w']))
        for i,level in enumerate(self.levels):
            if i:
                previous=self.levels[i-1]
                self.launch('reduce_edges',level['nf'],(np.int32(level['nf']),previous['edge_child_row'],previous['edge_child'],previous['w'],level['w']))
            self.launch('diagonal',level['nc'],(np.int32(level['nc']),level['row'],level['edge'],level['mass'],level['w'],level['diagonal']))
        last=self.levels[-1]
        self.kernels['factor_parallel' if self.parallel_coarse else 'factor']((1,),(32 if self.parallel_coarse else 1,),
            (np.int32(last['nc']),last['row'],last['edge'],last['other'],last['w'],last['diagonal'],self.factor,self.status))

    def multiply(self,x,out,level=0):
        g=self.levels[level]
        self.launch('multiply',g['nc'],(np.int32(g['nc']),g['row'],g['edge'],g['other'],g['owner'],g['mass'],g['w'],x,out))

    def smooth(self,i,zero):
        g=self.levels[i]
        for sweep in range(2):
            self.launch('smooth',g['nc'],(np.int32(g['nc']),g['row'],g['edge'],g['other'],g['w'],g['diagonal'],g['b'],g['x'],g['work'],np.int32(zero and sweep==0)))
            g['x'],g['work']=g['work'],g['x']

    def cycle(self,i=0):
        g=self.levels[i]
        if i==len(self.levels)-1:
            self.kernels['coarse_solve_parallel' if self.parallel_coarse else 'coarse_solve']((1,),(32 if self.parallel_coarse else 1,),
                (np.int32(g['nc']),self.factor,g['b'],g['x']));return
        self.smooth(i,True);self.multiply(g['x'],g['res'],i);coarse=self.levels[i+1]
        self.launch('restrict',coarse['nc'],(np.int32(coarse['nc']),g['child_row'],g['child'],g['b'],g['res'],coarse['b']))
        self.cycle(i+1)
        self.launch('prolong',g['nc'],(np.int32(g['nc']),g['parent'],coarse['x'],g['x']))
        self.smooth(i,False)

    def precondition(self,r):
        self.levels[0]['b'][...]=r;self.cycle();return self.levels[0]['x']

    def graph_iteration(self):
        cp=self.cp;b=self.buffers;s=self.scalar
        self.multiply(b['p'],b['ap']);cp.multiply(b['p'],b['ap'],out=b['product']);cp.sum(b['product'],out=s['denominator'])
        self.launch('pcg_update',self.nc,(np.int32(self.nc),b['filtered'],b['r'],b['p'],b['ap'],s['rho'],s['denominator'],self.status,self.done))
        cp.multiply(b['r'],b['r'],out=b['product']);cp.sum(b['product'],out=s['residual2'])
        self.kernels['convergence']((1,),(1,),(s['residual2'],s['tolerance'],self.done,self.count,self.status))
        z=self.precondition(b['r']);cp.multiply(b['r'],z,out=b['product']);cp.sum(b['product'],out=s['rho_next'])
        self.launch('pcg_direction',self.nc,(np.int32(self.nc),b['p'],z,s['rho'],s['rho_next'],self.status,self.done))
        cp.copyto(s['rho'],s['rho_next'])

    def solve(self,h,raw,*,dry=1.e-4,depth_factor=2.,rtol=1.e-4,atol=1.e-10,maxiter=500,graph_batch=0):
        if not isinstance(graph_batch,int) or graph_batch<0:raise ValueError('Nonnegative graph batch required')
        if graph_batch:
            self.cp.cuda.get_current_stream().synchronize()
            with self.graph_stream,self.cp.cuda.using_allocator(self.graph_pool.malloc):
                return self._solve(h,raw,dry=dry,depth_factor=depth_factor,rtol=rtol,atol=atol,maxiter=maxiter,graph_batch=graph_batch)
        return self._solve(h,raw,dry=dry,depth_factor=depth_factor,rtol=rtol,atol=atol,maxiter=maxiter,graph_batch=0)

    def _solve(self,h,raw,*,dry,depth_factor,rtol,atol,maxiter,graph_batch):
        """Return resident filtered field and small host receipt; reject on failure.

        Identical physical matrix and relative/area-scaled residual contract.
        For fewer than 65,536 cells CPU uses MIC rather than this multilevel
        preconditioner; that algorithm difference is explicitly experimental.
        """
        self.array(raw);self.array(h)
        if not np.isfinite([rtol,atol]).all() or rtol<=0 or atol<0 or maxiter<1:raise ValueError('Invalid tolerance')
        cp=self.cp;self.done.fill(0);self.count.fill(0);self.coefficients(h,dry,depth_factor)
        if int(self.status.get()[0]):raise RuntimeError('Coarse matrix factorization failed')
        b=self.buffers;s=self.scalar;mass=self.levels[0]['mass'];b['filtered'][...]=raw
        cp.multiply(mass,raw,out=b['rhs']);self.multiply(b['filtered'],b['ap']);cp.subtract(b['rhs'],b['ap'],out=b['r'])
        tolerance=float((rtol*cp.sqrt(cp.sum(b['rhs']**2))+atol*cp.sqrt(cp.sum(mass**2))).get())
        s['tolerance'].fill(tolerance)
        residual=float(cp.sqrt(cp.sum(b['r']**2)).get());iterations=0
        if residual>tolerance:
            z=self.precondition(b['r']);b['p'][...]=z;cp.sum(b['r']*z,out=s['rho'])
            if graph_batch:
                # First real iteration warms every kernel/reduction before capture.
                self.graph_iteration();launched=1
                while int(self.done.get()[0])==0 and launched<maxiter:
                    if self.graph is None:
                        self.graph_stream.synchronize()
                        try:
                            self.graph_stream.begin_capture();self.graph_iteration();self.graph=self.graph_stream.end_capture()
                        except Exception:
                            if self.graph_stream.is_capturing():self.graph_stream.end_capture()
                            raise
                        self.graph.upload(stream=self.graph_stream)
                    batch=min(graph_batch,maxiter-launched)
                    for _ in range(batch):self.graph.launch(stream=self.graph_stream)
                    launched+=batch
                iterations=int(self.count.get()[0]);residual=float(cp.sqrt(cp.sum(b['r']**2)).get())
            for iterations_plain in range(1,0 if graph_batch else maxiter+1):
                iterations=iterations_plain
                self.multiply(b['p'],b['ap']);cp.sum(b['p']*b['ap'],out=s['denominator'])
                self.launch('pcg_update',self.nc,(np.int32(self.nc),b['filtered'],b['r'],b['p'],b['ap'],s['rho'],s['denominator'],self.status,self.done))
                residual=float(cp.sqrt(cp.sum(b['r']**2)).get())
                if residual<=tolerance:break
                z=self.precondition(b['r']);cp.sum(b['r']*z,out=s['rho_next'])
                self.launch('pcg_direction',self.nc,(np.int32(self.nc),b['p'],z,s['rho'],s['rho_next'],self.status,self.done))
                cp.copyto(s['rho'],s['rho_next'])
        self.multiply(b['filtered'],b['ap']);cp.subtract(b['rhs'],b['ap'],out=b['true_residual'])
        true=float(cp.sqrt(cp.sum(b['true_residual']**2)).get());finite=bool(cp.all(cp.isfinite(b['filtered'])).get())
        ok=finite and int(self.status.get()[0])==0 and residual<=tolerance and true<=10*tolerance
        receipt=dict(iterations=iterations,recurrence_residual=residual,true_residual=true,tolerance=tolerance,passed=ok,
                     levels=[g['nc'] for g in self.levels],graph_batch=graph_batch,parallel_coarse=self.parallel_coarse,
                     method='unchanged operator; symmetric multilevel PCG; device convergence with batched graph' if graph_batch else 'unchanged operator; symmetric multilevel PCG; host convergence scalars')
        if not ok:raise RuntimeError(f'GPU curvature convergence failed: {receipt}')
        return b['filtered'],receipt
