"""FLOW dry-bank mass transfer on a complete Cartesian patch.

This operator has no hydraulic clock. In particular, using its CUDA kernels
does not imply qualification of an implicit wet/dry integrator.
"""
from pathlib import Path
from types import MethodType
import numpy as np
from .planform_v105 import _native_library_v105, HEADER, SOURCE
from .bend_methods_v1 import digest, BendTrialRejected

CUDA = Path(__file__).resolve().parents[2] / 'braids_gpu/planform_v105.cu'


class DryBankTransfer:
    def __init__(self, backend, *, threads=4):
        if backend not in ('cpu', 'gpu'):
            raise ValueError('Explicit CPU/GPU backend required')
        if not isinstance(threads, int) or not 1 <= threads <= 256:
            raise ValueError('Explicit CPU threads in [1,256] required')
        self.backend, self.threads = backend, threads
        self.identity = dict(operator='FLOW2DH_net_mass_dry_bank',
            source=digest(SOURCE), header=digest(HEADER), cuda=digest(CUDA),
            adapter=digest(__file__), backend=backend, threads=threads,
            source_anchor='bott3d.f90:884-989', integration='owned_by_caller')
        if backend == 'cpu':
            self.xp = np
            self.library = _native_library_v105()
            self.dry = self.library[5]
            self.identity['native'] = self.library[-1]
        else:
            import cupy as cp
            self.xp = cp
            options=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true','--ftz=false')
            self.module=cp.RawModule(code='#define BRAIDS_HD __device__ inline\n'+HEADER.read_text()+CUDA.read_text(), options=options)
            self.kernels=[self.module.get_function('braids_q8a_dry_'+n+'_v105') for n in ('proposal','scale','gather')]
            self.identity['cuda_options']=list(options)

    def apply(self, bed, depth, area, bulk, wet, erodible, erosion, fraction, fixfac, initial,
              *, theta, hmaxth, sedthr):
        """Return post-transfer areal stock and signed transferred mass per cell.

        bed/depth are the PRE-update geometry. erosion is the signed accepted
        transport increment in kg/m2 (including deposition). initial is the
        POST-transport available inventory. Arrays remain on the chosen backend.
        """
        xp=self.xp
        if bed.ndim!=2 or min(bed.shape)<1:
            raise ValueError('Complete Cartesian patch required')
        ny,nx=bed.shape; nc=nx*ny; ng=len(bulk)
        if not (0<=theta<=1 and hmaxth>=0 and sedthr>=0):
            raise ValueError('Invalid dry-bank controls')
        def real(a,shape):
            a=xp.ascontiguousarray(a,dtype=xp.float64)
            if a.shape!=shape or not bool(xp.all(xp.isfinite(a))):
                raise ValueError('Invalid dry-bank shape or value')
            return a
        bed,depth,area=[real(a,(ny,nx)) for a in (bed,depth,area)]
        bulk=real(bulk,(ng,))
        erosion,fraction,fixfac,initial=[real(a,(ng,ny,nx)) for a in (erosion,fraction,fixfac,initial)]
        if bool(xp.any(area<=0)) or bool(xp.any(bulk<=0)) or any(bool(xp.any(a<0)) for a in (depth,fraction,fixfac,initial)):
            raise ValueError('Invalid dry-bank inventory/material')
        flags=[]
        for a in (wet,erodible):
            a=xp.ascontiguousarray(a,dtype=xp.int32)
            if a.shape!=(ny,nx) or bool(xp.any((a!=0)&(a!=1))):raise ValueError('Invalid bank mask')
            flags.append(a)
        wet,erodible=flags
        result=xp.empty_like(initial);moved=xp.zeros_like(initial);status=xp.zeros(nc,dtype=xp.int32)
        if self.backend=='cpu':
            self.dry(nx,ny,ng,self.threads,float(theta),float(hmaxth),float(sedthr),
                area,bulk,bed,depth,wet,erodible,erosion,fraction,fixfac,initial,result,moved,status)
        else:
            proposal=xp.empty(nc*4*ng);scale=xp.empty_like(initial)
            ij=tuple(np.int32(a) for a in (nx,ny,ng)); re=tuple(np.float64(a) for a in (theta,hmaxth,sedthr))
            self.kernels[0](((nc+127)//128,),(128,),(*ij,*re,area,bulk,bed,depth,wet,erodible,erosion,fraction,fixfac,initial,proposal))
            grid=((nc*ng+127)//128,)
            self.kernels[1](grid,(128,),(*ij,area,wet,initial,proposal,scale))
            self.kernels[2](grid,(128,),(*ij,area,wet,initial,proposal,scale,result,moved,status))
        if bool(xp.any(status)) or abs(float(xp.sum(moved)))>1e-9:
            raise BendTrialRejected('Dry-bank stock or internal-transfer budget failed')
        return result,moved


def install_shoreline_bank(solver, xy, *, theta=.5, hmaxth=0., sedthr=.001):
    """Compose one dry-bank owner with the explicit research shoreline solver.

    Internal transfers do not enter the external boundary ledger. All persistent
    arrays live on physics, so its transaction/restart owns them atomically.
    Only full, uniform Cartesian grids without internal walls are admitted.
    """
    p=solver.physics;xp=p.xp
    if p.time or p.steps or p._open or hasattr(p,'bank_engine'):
        raise ValueError('Install one bank owner before execution')
    xy=np.asarray(xy);x=np.unique(xy[:,0]);y=np.unique(xy[:,1]);nx=len(x);ny=len(y)
    if nx*ny!=p.n or min(nx,ny)<2:raise ValueError('Full Cartesian grid required')
    dx=np.diff(x);dy=np.diff(y)
    if not (np.allclose(dx,dx[0]) and np.allclose(dy,dy[0])):raise ValueError('Uniform grid required')
    order=np.lexsort((xy[:,0],xy[:,1]));back=np.argsort(order)
    target=np.array(np.meshgrid(x,y)).reshape(2,-1).T
    if not np.allclose(xy[order],target):raise ValueError('Duplicate/missing grid cells')
    g=p.g;o=solver.host(g['owner']);j=solver.host(g['neighbour'])
    if not np.allclose(solver.host(g['area']),dx[0]*dy[0]):raise ValueError('Cut cells not admitted')
    # Every Cartesian interior edge must exist, and no extra internal wall.
    rank=back;edges=set(tuple(sorted((int(rank[a]),int(rank[b])))) for a,b in zip(o,j) if b>=0)
    expected=set()
    for c in range(p.n):
        if c%nx+1<nx:expected.add((c,c+1))
        if c//nx+1<ny:expected.add((c,c+nx))
    if edges!=expected:raise ValueError('Internal wall/hole not admitted')
    p.bank_engine=DryBankTransfer(solver.backend,threads=solver.controls.threads)
    p.bank_order=xp.asarray(order);p.bank_back=xp.asarray(back)
    p.bank_cumulative_mass=xp.zeros(p.n);p.bank_trial_mass=xp.zeros(p.n)
    base_finish=p.finish;base_commit=p.commit
    def finish(owner,qold,qnew,water,*,morphology=True):
        result=base_finish(qold,qnew,water,morphology=morphology)
        owner.bank_trial_mass.fill(0.)
        if not morphology:return result
        order=owner.bank_order;shape=(ny,nx)
        def field(a):return a[order].reshape(shape)
        mass=(1590.*(owner.ztrial-owner.z0))[None,:]
        stock=(1590.*(owner.ztrial-owner.rock))[None,:]
        _,moved=owner.bank_engine.apply(field(owner.z0),field(qnew[0]),field(g['area']),
            xp.array([1590.]),field(qnew[0]>sedthr),xp.ones(shape,dtype=xp.int32),
            mass[:,order].reshape(1,ny,nx),xp.ones((1,ny,nx)),xp.ones((1,ny,nx)),
            stock[:,order].reshape(1,ny,nx),theta=theta,hmaxth=hmaxth,sedthr=sedthr)
        owner.bank_trial_mass[:]=moved.ravel()[owner.bank_back]
        owner.ztrial+=owner.bank_trial_mass/(1590.*g['area'])
        owner.last.update(bank_transfer_mass_kg=float(xp.sum(xp.maximum(owner.bank_trial_mass,0.))),
                          bank_internal_residual_kg=float(xp.sum(owner.bank_trial_mass)))
        return owner.last
    def commit(owner):
        base_commit();owner.bank_cumulative_mass+=owner.bank_trial_mass
    p.finish=MethodType(finish,p);p.commit=MethodType(commit,p)
    p.identity['dry_bank']=dict(theta=theta,hmaxth=hmaxth,sedthr=sedthr,
        runtime=p.bank_engine.identity,collapse=False,preupdate_geometry=True,
        post_transport_inventory=True,shape=[ny,nx])
    return solver
