"""Bounded zero-time JC wet/dry regrid; not an adaptive Desktop controller.

Use actual polygon overlaps and the existing paired terrain prolongation.
Old internal-face history stays attached to its mesh epoch, never interpolated
onto a different graph. Post-remesh checkpoint admission is a separate gate.
"""
from dataclasses import asdict
import copy,hashlib,io,json
import numpy as np
from .jiang_cao_checkpoint import JiangCaoRestart,canonical,array_hash
from .bend_methods_v1 import digest
from .river_amr_transfer import CellMap,TransferEngine,boundary_map,refinement_subset
from .river_terrain_transfer import TerrainEngine


class JiangCaoRemesh(JiangCaoRestart):
    def __init__(self,*args,polygons,**kwargs):
        super().__init__(*args,**kwargs)
        from .river_polygon_banks import topology
        from shapely.geometry import Polygon,LineString
        data=self.solver.a.data;checked=topology(data,polygons,transfers=False)
        self.polygons=tuple(np.frombuffer(np.asarray(v,dtype=float).tobytes(),dtype=float).reshape(-1,2) for v in polygons)
        scale=max(1.,max(float(np.max(abs(v))) for v in self.polygons));tol=max(1e-10,32*np.spacing(scale))
        if not np.allclose(checked['xy'].T,self.a.host(self.mesh.g['xy']),atol=tol,rtol=0):raise ValueError('Polygon cell order/centroid differs')
        shapes=[Polygon(v) for v in self.polygons];perimeter=np.zeros(self.n)
        for f,(i,j) in enumerate(zip(data['owner'],data['neighbour'])):
            normal=data['normal'][:,f];tangent=np.array([-normal[1],normal[0]])
            line=LineString([data['mid'][:,f]+sign*.5*data['length'][f]*tangent for sign in (-1,1)])
            for c in (i,j) if j>=0 else (i,):
                if not shapes[c].boundary.buffer(tol).covers(line):raise ValueError('Polygon face does not match hydraulic geometry')
                perimeter[c]+=data['length'][f]
        if not np.allclose(perimeter,[p.length for p in shapes],rtol=1e-10,atol=10*tol):raise ValueError('Incomplete polygon face coverage')
        self._epochs=();self._receipts=();self._epoch_gross_start=0.
        self.identity=dict(self.identity,method='lifecycle.jc2026_remesh203_v1',
            remesh_sha256=digest(__file__),polygons_sha256=hashlib.sha256(b''.join(v.tobytes() for v in self.polygons)).hexdigest(),
            scope='zero-time fixed-domain nested mesh transfers; no automatic AMR or post-remesh restart admission')

    @property
    def epochs(self):return self._epochs # immutable serialized epoch records

    @property
    def receipts(self):return copy.deepcopy(self._receipts)

    def checkpoint(self,*args,**kwargs):
        if self._receipts:raise ValueError('Post-remesh checkpoints not admitted by the fixed-mesh schema')
        return super().checkpoint(*args,**kwargs)

    def restart(self,*args,**kwargs):
        if self._receipts:raise ValueError('Post-remesh restart requires an epoch-aware schema')
        return super().restart(*args,**kwargs)


def contract(a):
    s=a.solver
    clean=lambda v:{k:x for k,x in v.items() if k not in ('geometry','geometry_sha256')}
    return dict(backend=s.backend,material=asdict(s.exchange.material),roughness=asdict(s.roughness.parameters),
        shallow=asdict(s.shallow),controls=asdict(s.a.controls),hydraulic=s.a.midpoint.identity,
        physical={k:clean(v) if isinstance(v,dict) else v for k,v in s.identity.items()
            if k not in ('hydraulic','geometry','initial_state_sha256')},
        collapse=[s.collapse.target_slope,s.collapse.tolerance,s.collapse.max_iterations])


def validate(a):
    a._assert_configuration();a._assert_carrier();s=a.solver;xp=s.xp;g=s.mesh.g
    s._validate(s.state)
    if bool(xp.any(s.state[1:4,s.state[0]<=s.dry])):raise ValueError('Unclean dry accepted state')
    if s.state.shape[0]==7 and bool(xp.any(s.state[6,s.state[0]<=s.dry])):
        raise ValueError('Dry state carries secondary intensity')
    if float(xp.max(abs(s.state[5]-s.state[4]/s.B-xp.asarray(a._bedrock))))>1e-10:
        raise ValueError('Accepted bedrock datum changed')
    forbidden=(g['neighbour']>=0)|(g['kind']==1)
    if any(bool(xp.any(getattr(s,k)[forbidden])) for k in ('solid_integral','water_integral')):
        raise ValueError('Nonphysical boundary ledger')
    if bool(xp.any(s.collapse_integral[g['neighbour']<0])):raise ValueError('Exterior collapse ledger')
    solid,water=s.totals(s.state)
    if max(abs(solid+float(xp.sum(s.solid_integral))-s.initial_solid),
           abs(water+float(xp.sum(s.water_integral))-s.initial_water))>1e-10*max(1.,s.initial_solid,s.initial_water):
        raise ValueError('Accepted phase budget failed')


def epoch_record(a):
    s=a.solver;host=s.a.host
    header=dict(identity=a.identity,time_s=s.time,steps=s.steps,
                gross_start_m3=a._epoch_gross_start,gross_end_m3=s.collapse_gross,
                last=copy.deepcopy(s.last))
    arrays={k:host(v).copy() for k,v in s.mesh.g.items()}
    arrays['collapse_integral']=host(s.collapse_integral).copy()
    for k in ('secondary_fields','dispersion_flux'):
        arrays[k]=host(getattr(s,k)).copy() if hasattr(s,k) else np.zeros(a._shapes()[k])
    header['arrays']={k:array_hash(v) for k,v in arrays.items()}
    stream=io.BytesIO();np.savez(stream,header=np.frombuffer(canonical(header),dtype=np.uint8),**arrays)
    return stream.getvalue()


def transfer(source,target,*,check=lambda:None):
    """Stage into a fresh private target. Caller publishes only after success.

    No hydraulic/transport/collapse step or physical time advance occurs here.
    Unsafe coarsening is rejected, so an adaptive controller must retain it.
    """
    if type(source) is not JiangCaoRemesh or type(target) is not JiangCaoRemesh or source is target:
        raise ValueError('Distinct JC remesh owners required')
    with source._exclusive(),target._exclusive():
        check();validate(source);validate(target)
        s,t=source.solver,target.solver;xp=s.xp;host=s.a.host
        if t.time or t.steps or target._receipts:raise ValueError('Fresh unpublished target required')
        if canonical(contract(source))!=canonical(contract(target)):raise ValueError('Remesh physical method/backend changed')
        for name in ('forcing','mode','reversal_policy'):
            if not np.array_equal(s.a.data[name],t.a.data[name]):raise ValueError('Hydraulic boundary forcing changed')
        # The carrier authenticates these polygons against its actual faces and
        # areas on construction; do not accept a caller-supplied arbitrary map.
        mapping=CellMap.polygons(source.polygons,target.polygons,
                                host(s.mesh.g['area']),host(t.mesh.g['area']))
        subset=refinement_subset(mapping)
        engine=TransferEngine(s.backend,s.a.physics.flow.threads)
        face=boundary_map(s.a.data,t.a.data);off,ids,w=face
        rows=np.repeat(np.arange(t.nf),np.diff(off))
        oldkind=host(s.mesh.g['kind']);newkind=host(t.mesh.g['kind'])
        if not np.array_equal(oldkind[ids],newkind[rows]):raise ValueError('Boundary kind changed')
        intensive=w*s.a.data['length'][ids]/t.a.data['length'][rows]
        mappedfeed=engine.apply(source._feed,off,ids,intensive,t.nf)
        outside=t.mesh.g['neighbour']<0
        if not bool(xp.allclose(mappedfeed[outside],target._feed[outside],rtol=1e-13,atol=1e-15)):
            raise ValueError('Sediment inlet forcing changed')
        # Public AMR pins open boundaries. Enforce exact faces/history here,
        # while allowing ordinary walls to change their partition.
        opened=np.flatnonzero(newkind>=2)
        for f in opened:
            k=off[f]
            if off[f+1]-k!=1 or w[k]!=1. or s.a.data['length'][ids[k]]!=t.a.data['length'][f]:
                raise ValueError('Open boundary faces must remain unchanged')
        st=engine.cells(s.state,mapping)
        rock=engine.cells(s.state[5]-s.state[4]/s.B,mapping)
        # Coarse cells represent the average rock; finer cells recover ONLY
        # the prepared subcell rock relief, not a stale pre-erosion bed surface.
        terrain=None
        if subset is not None:
            si,ti,sub=subset;si=xp.asarray(si);ti=xp.asarray(ti)
            te=TerrainEngine(s.backend,s.a.physics.flow.threads)
            carried=s.state[[3,6]][:,si] if s.adjustment is not None else s.state[3:4,si]
            r,q,m,terrain=te.prolong(sub,(s.state[5]-s.state[4]/s.B)[si],
                xp.asarray(target._bedrock)[ti],st[4,ti]/s.B,s.state[:3,si],carried,dry=s.dry)
            rock[ti]=r;st[:3,ti]=q;st[3,ti]=m[0];st[5,ti]=r+st[4,ti]/s.B
            if s.adjustment is not None:st[6,ti]=m[1]
        if float(xp.max(abs(rock-xp.asarray(target._bedrock))))>1e-10:
            raise ValueError('Prepared rock hierarchy changed; no remesh terrain reset')
        # Do not erase a resolved shoreline by averaging wet and dry children.
        wet=(s.state[0]>s.dry).astype(xp.float64)
        fraction=engine.cells(wet,mapping);coarsened=xp.asarray(np.diff(mapping.offsets)>1)
        if bool(xp.any(coarsened&(fraction>1e-12)&(fraction<1-1e-12))):
            raise ValueError('Unsafe shoreline coarsening; retain refinement')
        t._validate(st)
        if bool(xp.any(st[1:4,st[0]<=s.dry])):raise ValueError('Unsafe dry inventory coarsening')
        if s.adjustment is not None and bool(xp.any(st[6,st[0]<=s.dry])):
            raise ValueError('Unsafe dry secondary inventory coarsening')
        boundary={k:engine.apply(getattr(s,k),*face,t.nf) for k in ('solid_integral','water_integral')}
        before=host((s.state[:5]*s.mesh.g['area']).sum(1))
        after=host((st[:5]*t.mesh.g['area']).sum(1));residual=after-before
        if np.max(abs(residual))>1e-11*max(1.,float(np.max(abs(before)))):raise ValueError('Conservative inventory transfer failed')
        secondary_residual=None
        if s.adjustment is not None:
            secondary_residual=float(xp.sum(st[6]*t.mesh.g['area'])-xp.sum(s.state[6]*s.mesh.g['area']))
            if abs(secondary_residual)>1e-11:raise ValueError('Secondary inventory transfer failed')
        phase=t.totals(st)
        budgets=[phase[i]+float(xp.sum(boundary[k]))-v for i,(k,v) in enumerate(
            (('solid_integral',s.initial_solid),('water_integral',s.initial_water)))]
        if max(abs(v) for v in budgets)>1e-10*max(1.,s.initial_solid,s.initial_water):raise ValueError('Remapped phase budget failed')
        for k in boundary:
            np.testing.assert_array_equal(host(boundary[k])[opened],host(getattr(s,k))[ids[off[opened]]])
        archived=epoch_record(source)
        newarrays=dict(state=st,**boundary,collapse_integral=xp.zeros(t.nf),
                       secondary_fields=xp.zeros((9,t.n)),dispersion_flux=xp.zeros((2,t.nf)))
        receipt=dict(method='remesh.jc2026_terrain_volume203_v1',source_cells=s.n,target_cells=t.n,
            time_s=s.time,steps=s.steps,mapping_sha256=mapping.identity,terrain=terrain,
            transfer_runtime=engine.identity,inventory_residual=residual.tolist(),phase_budget_residual_m3=budgets,
            archived_epoch_sha256=hashlib.sha256(archived).hexdigest(),gross_collapse_m3=s.collapse_gross,
            zero_time=True,open_faces_exact=len(opened),new_diagnostics='invalidated; rebuilt by the next physical stage')
        if secondary_residual is not None:receipt['secondary_inventory_residual_m4_s']=secondary_residual
        if s.backend=='gpu':xp.cuda.get_current_stream().synchronize()
        check() # neither owner has changed yet, including all rejection paths
        oldq=t.a.q.copy();oldz=t.a.z.copy();oldz0=t.a.physics.z0.copy()
        try:
            t.a.q[...]=st[:3];t.a.z[...]=st[5];t.a.physics.z0[...]=st[5]
            if s.backend=='gpu':xp.cuda.get_current_stream().synchronize()
        except BaseException:
            t.a.q[...]=oldq;t.a.z[...]=oldz;t.a.physics.z0[...]=oldz0;raise
        for k,v in newarrays.items():setattr(t,k,v)
        t.time=s.time;t.steps=s.steps;t.last=copy.deepcopy(s.last)
        t.initial_solid=s.initial_solid;t.initial_water=s.initial_water;t.collapse_gross=s.collapse_gross
        t.a.midpoint.evaluations=s.a.midpoint.evaluations;t.a.midpoint.matvecs=s.a.midpoint.matvecs
        target._initial_totals=source._initial_totals
        target._epochs=source._epochs+(archived,);target._receipts=source._receipts+(receipt,)
        target._epoch_gross_start=s.collapse_gross;target._clear_trial()
        return copy.deepcopy(receipt)
