"""Classwise EH and finite active/substrate stocks on the selected river route.

Common midpoint hydraulics, pure-class second-order bed-row fluxes, then FLOW's
donor surface fraction, conservative active-stock limiter and ONE layer update.
This is not a full coupled Hirano eigensystem or an implicit/AMR/QGIS admission.
Banks, collapse and suspension require their own class-resolved composition.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import MethodType
import numpy as np
from .bend_methods_v1 import BendTrialRejected, digest
from .bed_columns_v101 import NativeColumnRuntimeV101, initialize_columns_v101, host
from .river_columns import RiverColumnOwner
from .flow2dh_eh_operator import ClassicFlowEH
from .shoreline_solver import ShorelinePhysics
from .flow2dh_mobile_bed import Flow2DHMobileBed
from braids_gpu.atomic_output import arrays_to_bytes,write_bytes_atomic
from .river_material import RiverMaterial


class RiverMixtureSolver:
    def __init__(self, solver, methods, thickness, fractions, class_feed):
        a=solver;p=a.physics;xp=a.xp;g=p.g
        if a.time or a.steps or p._open: raise ValueError('Install mixtures before execution')
        identity=a.identity['selected_river_bundle']
        if p.identity['dry_bank']['theta'] != 0 or p.identity['slope_relaxation']['method'] != 'off':
            raise ValueError('Mixtures require banks and collapse OFF until class-resolved composition')
        if methods.bed not in ('mixed','layered') or methods.transport_layer not in ('constant','depth'):
            raise ValueError('Explicit mixed/layered material contract required')
        mass,sv=initialize_columns_v101(methods,thickness,fractions)
        self.material=RiverMaterial.from_columns(methods).resident(xp)
        # The frozen pure-class Exner row uses bulk/solid conversion for this
        # class. Do not use the hydraulic carrier's quartz packing for all rows.
        self._class_porosity=tuple(1.-c/s for c,s in zip(
            self.material.definition.column_density,self.material.definition.solid_density))
        nc=p.n;nl=mass.shape[0];ng=len(methods.fraction_ids)
        feed=np.asarray(class_feed,float)
        if (feed.shape!=(ng,p.nf) or not np.isfinite(feed).all() or np.any(feed<0)
                or np.any(feed[:,host(p.bedbc)!=1])):
            raise ValueError('Explicit classwise solid-volume feed required only at feed boundaries')
        if a.backend=='cpu':rt=NativeColumnRuntimeV101(methods,nc,nl,threads=p.flow.threads)
        else:
            from braids_gpu.bed_columns_v101 import ResidentColumnRuntimeV101
            rt=ResidentColumnRuntimeV101(methods,nc,nl)
        rock=host(a.z)-host(self.material.bed(xp.asarray(mass),xp.asarray(sv),xp.zeros(p.n)))
        self.columns=RiverColumnOwner(rt,mass,sv,rock,host(g['area']))
        self.solver=a;self.methods=methods
        p.rock[:]=xp.asarray(rock)
        if a.backend=='gpu':a.hydraulics.rock[:]=p.rock
        self.capacity=[ClassicFlowEH(a.backend,diameter=d,chezy=p.chezy,threads=p.flow.threads)
                       for d in methods.diameter_m]
        self.feed=xp.asarray(feed)
        p.class_flux=xp.zeros((ng,p.nf));p.class_delta=xp.zeros((ng,nc))
        # Instantaneous, un-limited constitutive capacity at each accepted
        # end state.  It is output-only: Exner continues to use class_flux
        # after composition, available-stock and boundary treatment.
        p.class_capacity=xp.zeros((ng,p.nf))
        p.class_boundary_ledger=xp.zeros((ng,p.nf));p.class_face_integral=xp.zeros((ng,p.nf))
        from .class_bed_update import ClassBedUpdate
        self._class_update=ClassBedUpdate(p,ng,self.material.solid)
        self._single_faces=p.bed_faces
        self._wet_avalanche=None
        p.bed_faces=MethodType(lambda owner,q,z,hi:self.bed_faces(q,z,hi),p)
        p.finish=MethodType(lambda owner,qold,qnew,water,**kw:self.finish(qold,qnew,water,**kw),p)
        def commit(owner):
            if self.columns.pending is not None:self.columns.commit()
            Flow2DHMobileBed.commit(owner)
            self._class_update.commit_ledger()
            owner.transport_cumulative_mass+=owner.transport_trial_mass
        p.commit=MethodType(commit,p)
        identity['mixtures_integrated']=True
        identity['scope']='classwise_midpoint_finite_mixed_or_layered_stock'
        p.identity['river_mixtures']=dict(wrapper_sha256=digest(__file__),
            owner_sha256=digest(Path(__file__).with_name('river_columns.py')),methods=methods.__dict__,
            column_runtime=rt.identity,capacity=[c.identity for c in self.capacity],
            material=self.material.identity,
            composition=self.columns.composition.identity,
            class_feed_sha256=__import__('hashlib').sha256(feed.tobytes()).hexdigest(),
            hiding='none: EH has no critical-Shields/hiding term',
            fraction='upwind by corrected class flux, once after slope; prescribed feed already class-resolved',
            inventory='one column owner; active-stock donor limiter; explicit basal-policy source ledger',
            representation=methods.bed,transport_layer=methods.transport_layer,
            targets='backend-resident; current accepted-end depth; mixed bed has no layer redistribution',
            class_update=self._class_update.identity,
            wave='existing frozen pure-class bed-row splitting; not a coupled mixture eigensystem',
            public_support=False)
        from .river_column_methods import method_ids
        p.identity['river_mixtures']['column_methods'] = method_ids(methods)
        p.identity['river_mixtures']=json.loads(json.dumps(p.identity['river_mixtures']))

    def __getattr__(self,name):return getattr(self.solver,name)

    @property
    def morphology(self):return self.solver.morphology

    @morphology.setter
    def morphology(self,value):
        if type(value) is not bool:raise ValueError('Explicit morphology switch required')
        self.solver.morphology=value

    @property
    def cancelled(self):return self.solver.cancelled

    @cancelled.setter
    def cancelled(self,value):self.solver.cancelled=value

    def bed_faces(self,q,z,hi,water=None):
        p=self.physics;xp=self.xp;g=p.g;o=g['owner'];j=g['neighbour']
        exterior=getattr(p,'exterior_sediment',None)
        if exterior is not None:exterior.prepare(q,z,hi)
        frac=self.columns.surface_fraction
        old=p.eh,p.parameters,p.feed
        # Only capacity, diameter and feed vary inside this loop. Reuse the
        # exact same hydraulic geometry/roughness and face traces for all
        # classes. Never retain a cache across a call, stage, step or regrid.
        if hasattr(p,'_river_hydraulic_batch'):
            raise RuntimeError('Nested class hydraulic batch')
        batch=p._river_hydraulic_batch=dict(effective=False,traces=False)
        effective=p.effective_inputs
        def shared_effective(owner,water,bed):
            if not batch['effective']:
                effective(water,bed);batch['effective']=True
        p.effective_inputs=MethodType(shared_effective,p)
        try:
            for k,(d,capacity) in enumerate(zip(self.methods.diameter_m,self.capacity)):
                if capacity.solid_density != self.material.definition.solid_density[k]:
                    raise ValueError('Capacity density differs from the explicit class inventory')
                p.eh=capacity;p.parameters=replace(old[1],diameter=d,porosity=self._class_porosity[k]);p.feed=self.feed[k]
                self._single_faces(q,z,hi)
                # Capture the formula result before donor composition,
                # finite-stock limiting and prescribed boundary supply alter
                # the physical transported flux.
                p.class_capacity[k]=p.aux[2]
                if self._wet_avalanche is not None:
                    if k==0:avalanche_flux=self._wet_avalanche.evaluate(q)
                    p.aux[2]+=avalanche_flux
                donor=xp.where((p.aux[2]>=0)|(j<0),o,xp.maximum(j,0))
                # Prescribed flux already identifies a class; never multiply it twice.
                factor=xp.where(p.bedbc==1,1.,frac[k,donor])
                if exterior is not None:factor=exterior.donor_fraction(k,factor)
                p.class_flux[k]=p.aux[2]*factor
                if hasattr(self,'_transport_boundary'):self._transport_boundary(k,water)
        finally:
            p.eh,p.parameters,p.feed=old
            p.effective_inputs=effective
            del p._river_hydraulic_batch
        p.aux[2]=p.class_flux.sum(axis=0)

    def begin_columns(self,delta,qnew,dt):
        return self.columns.begin(delta,qnew[0],dt)

    def finish(self,qold,qnew,water,*,morphology=True):
        p=self.physics;xp=self.xp;g=p.g;c=self.columns
        ShorelinePhysics.finish(p,qold,qnew,water,morphology=False)
        p.class_face_integral.fill(0);p.transport_trial_mass.fill(0)
        if not morphology:return p.last
        dt=p._dt*p.morfac;self.bed_faces(qnew,p.z0,p.Wtrial,water)
        minimum=self._class_update.apply(c.available,dt)
        # A composed suspended owner contributes before the single layer update
        # and before bank sharing sees net TOTAL erosion. Bedload diagnostics
        # remain separate from the local water/bed exchange.
        bedload_delta=p.class_delta.copy()
        exchange=self.process_bed_delta(qold,qnew,water)
        column_delta=self.column_delta(qnew)
        self.begin_columns(column_delta,qnew,p._dt)
        out,sv,ext,basal,_,residual=c.pending
        if self.methods.base_update==1 and np.any(basal):
            raise BendTrialRejected('Independent finite base must not import basal material')
        p.ztrial[:]=self.material.bed(out,sv,c.rock)
        p.face_integral[:]=p.class_face_integral.sum(axis=0)
        outside=(g['neighbour']<0)&(g['bid']>=0)
        boundary=self.material.boundary_mass(p.class_face_integral,g['length'],outside)
        closure=float(xp.max(xp.abs(xp.sum(bedload_delta*g['area'],axis=1)-boundary)))
        # Keep the transport-only increment guard when a composed owner adds
        # conservative bank transfers before the single column evaluation.
        # The transport-only path guards the actual selected-policy bed response.
        transport_dz=(p.ztrial-p.z0 if column_delta is p.class_delta
                      else self.material.bulk_increment(p.class_delta,c.sv[0]))
        change=float(xp.max(xp.abs(transport_dz)/xp.maximum(qnew[0],p.sediment_depth)))
        if closure>1e-10 or change>.05 or not bool(xp.all(xp.isfinite(p.ztrial))):
            error=BendTrialRejected('Class transport budget or bed increment failed')
            if closure<=1e-10 and change>.05 and bool(xp.all(xp.isfinite(p.ztrial))):
                error.relative_bed_increment=change
            raise error
        # Count solid material directly, not porosity times a bed difference.
        # Class-dependent packing makes the latter invalid even at fixed rock.
        solid_change=self.material.solid_volume(out)-self.material.solid_volume(c.mass)
        solid_external=(boundary+xp.sum(exchange*g['area'],axis=1)+xp.asarray(basal))/self.material.solid
        solid_res=float(xp.sum(solid_change*g['area'])-xp.sum(solid_external))
        if abs(solid_res)>1e-10:raise BendTrialRejected('Layer/bed-volume closure failed')
        p.transport_trial_mass[:]=bedload_delta.sum(axis=0)*g['area']
        p.last.update(morphology=True,class_flux_budget_max_kg=closure,
            class_column_budget_max_kg=float(abs(residual).max()),solid_volume_residual_m3=solid_res,
            basal_source_max_kg=float(abs(basal).max()),
            solid_boundary_m3=float(xp.sum(boundary/self.material.solid)),donor_factor_min=minimum,
            maximum_relative_bed_increment=change,morphfac=p.morfac)
        return p.last

    def column_delta(self,qnew):
        """Compose optional internal transfers BEFORE the one column trial."""
        return self.physics.class_delta

    def process_bed_delta(self,qold,qnew,water):
        """Additional local class exchange, kg/m2; default bedload-only path."""
        return self.xp.zeros_like(self.physics.class_delta)

    def step(self,*args,**kwargs):
        c=self.columns
        if c.pending is not None:raise ValueError('Previous column trial remains open')
        # Accepted arrays are immutable on CPU and never changed in-place on CUDA.
        before=dict(c.__dict__)
        for k in ('external','basal'):before[k]=before[k].copy()
        try:
            result=self.solver.step(*args,**kwargs)
            if not kwargs.get('commit',True):c.rollback()
            return result
        except BaseException:
            c.__dict__.clear();c.__dict__.update(before);c.rollback();raise

    def checkpoint_bytes(self):
        payload=self.solver.checkpoint_bytes()
        return arrays_to_bytes(river=np.frombuffer(payload,dtype=np.uint8),
            columns=np.array(json.dumps(self.columns.checkpoint(),sort_keys=True)))

    def checkpoint(self,path):
        write_bytes_atomic(path,self.checkpoint_bytes())

    def restart(self,path):
        # Validate both halves before accepting either; roll back on any failure.
        before=self.columns.checkpoint()
        with np.load(path,allow_pickle=False) as f, tempfile.TemporaryDirectory(prefix='braids-mixture-') as tmp:
            if set(f.files)!={'river','columns'}:raise ValueError('Mixture checkpoint schema differs')
            p=Path(tmp)/'river.npz';p.write_bytes(f['river'].tobytes())
            try:
                self.columns.restore(json.loads(str(f['columns'])))
                with np.load(p,allow_pickle=False) as native:
                    if not np.allclose(native['saved_z'],host(self.columns.bed),rtol=0,atol=1e-12):
                        raise ValueError('Restored bed differs from authoritative columns')
                self.solver.restart(p)
            except BaseException:self.columns.restore(before);raise
