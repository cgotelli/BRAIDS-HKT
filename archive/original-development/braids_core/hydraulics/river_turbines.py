"""Existing actuator and symmetric stress on the hydrostatic HLLC owner.

The f-wave-specific source/pressure correction is not an additional physical
force. HLLC uses its own hydrostatic face pressure; no f-wave trace is consumed.
"""
from pathlib import Path
from types import MethodType
from braids_core.sediment.bend_methods_v1 import digest
from braids_core.sediment.river_turbine_flow import Engine,RotorArray
from braids_core.sediment.river_hydrokinetic_stress import FullStressEngine


def install(a,cells,rows,root):
    p=a.physics;xp=a.xp;g=p.g;Path(root).mkdir(parents=True,exist_ok=True)
    if a.time or a.steps or p._open:raise ValueError('Fresh accepted-state owner required')
    if not rows or len({r['nu'] for r in rows})!=1:raise ValueError('One shared ambient viscosity required')
    from braids_core.sediment.river_turbine_thetis import ThetisArray
    cls=ThetisArray
    e=Engine(a.backend,p.flow.threads);stress=FullStressEngine(a.backend,p.flow.threads)
    source=cls(cells,rows,a.backend);a.turbine_source=source
    grad=xp.zeros((10,p.n));face=xp.zeros((2,p.nf));work=xp.zeros((3,p.n));ones=xp.ones(p.n)
    nu=xp.full(p.n,rows[0]['nu']);previous=p.add_momentum
    def gradients(q,z):
        e.launch('hkt_wet_gradients',p.n,p.n,float(p.parameters.dry),q,z,g['offset'],g['faces'],g['owner'],g['neighbour'],g['dxdy'],g['inverse'],grad)
    def add(self,q,water,rate):
        previous(q,water,rate);gradients(q,self.z0)
        if rows[0]['nu']>0.:
            stress.launch('hkt_full_strain_faces',p.nf,p.nf,p.n,float(p.parameters.dry),q,self.z0,grad,nu,g['owner'],g['neighbour'],g['normal'],g['distance'],g['xy'],face)
            work.fill(0.)
            p.flow.launch('flow_momentum_gather',p.n,p.n,p.nf,ones,face,g['offset'],g['faces'],g['signs'],g['length'],g['area'],work)
            rate[1:]+=work[1:]
        self.turbine_diagnostics=source.apply(q,self.z0,grad,rate)
    p.add_momentum=MethodType(add,p)
    def diagnose():gradients(a.q,a.z);return source.sample(a.q,a.z,grad)
    a.hkt_diagnose=diagnose;a.hkt_gradient=grad;a.hkt_status=xp.zeros(p.nf,dtype=xp.int32)
    a.hkt_low_speed_turbines=lambda:source.low_speed(a.q,a.z,grad,lambda:gradients(a.q,a.z))
    from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
    pressure=WaveEngine(a.backend,p.flow.threads,source=Path(__file__).with_name('hllc_turbines.h'),names=('hllc_turbine_pressure',))
    def correct(op):
        k=op.g
        pressure.launch('hllc_turbine_pressure',op.nf,op.n,op.nf,op.gravity,op.dry,op.half,op.grad,k['xy'],k['mid'],k['normal'],k['owner'],k['neighbour'],op.fl,op.fr)
    getattr(a,'solver',a).hkt_hllc_pressure=correct
    identity=dict(method='hkt.hllc_hancock_actuator_symmetric_stress_v1',source=source.identity,
        engine=e.identity,stress=stress.identity,nu_m2_s=rows[0]['nu'],wrapper=digest(__file__),
        scope=('fixed-axis normal-loading and stock flow-directed rotors; fixed bed and explicit power controls; hydrodynamics only' if any(r.get('orientation',{}).get('mode')=='fixed_axis' for r in rows) else 'same zero-yaw array, fixed-bed, coverage and power controls; hydrodynamics only'),
        pressure='hydrostatic HLLC with fully wet small-jump low-Froude acoustic correction; no f-wave trace access',pressure_engine=pressure.identity,
        temporal='actuator and symmetric stress predicted/evaluated at Hancock half time',
        desktop_admission='hkt_hllc_hydrodynamics_only')
    if a.morphology:
        identity['scope']='ordinary sediment owner retained; fixed hub elevation; evolving bed and clearance'
        identity['desktop_admission']='hkt_ordinary_sediment_composition_v1'
    a.identity['hkt_terrain']=identity;p.identity['hkt_terrain']=identity
    return a
