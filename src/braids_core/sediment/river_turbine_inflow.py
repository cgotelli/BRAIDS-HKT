"""Explicit, device-configured rotor-area inflow for the speed-surface control.

Supports separated rotors and a single upstream wake per exposed rotor. Wake
turning, overlapping wakes and yaw require further qualification, not defaults.
"""
import math
import numpy as np
from scipy.spatial import Delaunay
from braids_core.river_turbine_surface import resolve,resolve_moments

def _quadrature(radius):
    x,w=np.polynomial.legendre.leggauss(16);r=radius*np.sqrt((x+1)/2);a=2*np.pi*(np.arange(64)+.5)/64
    return (r[:,None]*np.cos(a)).ravel(),(r[:,None]*np.sin(a)).ravel(),np.broadcast_to(w[:,None]/128,(16,64)).copy().ravel()

def _mean(depth,hub,sigma):
    s=math.sqrt(2)*sigma
    return sigma*math.sqrt(math.pi/2)/depth*(math.erf((depth-hub)/s)+math.erf(hub/s))

def make_array_class(base):
    class InflowArray(base):
        def __init__(self,cells,rows,backend):
            super().__init__(cells,rows,backend)
            modes=[r['control']['mode']=='fixed_rpm_surface' for r in rows];self.enabled=any(modes)
            if not self.enabled:return
            if not all(modes):raise ValueError('Use explicit speed surfaces for every rotor in this array; mixed wake conventions are not admitted.')
            from shapely.geometry import Polygon,LineString
            from shapely.ops import unary_union
            # Closing only coordinate-roundoff seams avoids classifying an
            # internal shared edge as a real bank after a rigid rotation.
            geometry_tol=64*np.finfo(float).eps*max(1.,float(np.max(np.abs(self.rotors[0].xy))))
            fluid=unary_union([Polygon(c.vertices_xy) for c in cells]).buffer(geometry_tol)
            from braids_core.river_turbine_surface import exposure_parents
            self.parents=exposure_parents([[r.c['X'],r.c['Y']] for r in self.rotors],[r.normal for r in self.rotors],[r.c['D'] for r in self.rotors],[r.c['control'] for r in self.rotors]);self.sample_geometry={};self._node_gather={}
            for j,down in enumerate(self.rotors):
                for i,xD,lateral,sigma in self.parents[j]:
                    up=self.rotors[i]
                    if not fluid.covers(LineString([(up.c['X'],up.c['Y']),(down.c['X'],down.c['Y'])])):
                        raise ValueError('Straight wake path crosses a bank or solid; a streamline-based wake model is required for this placement.')
                y,z,w=_quadrature(down.c['D']/2);t=np.array([-down.normal[1],down.normal[0]])
                points=np.array([down.c['X'],down.c['Y']])+y[:,None]*t
                tri=Delaunay(down.xy);si=tri.find_simplex(points)
                if np.any(si<0):raise ValueError('Rotor interpolation lies outside mesh centroids; refine/move the rotor.')
                bary=np.einsum('nij,nj->ni',tri.transform[si,:2],points-tri.transform[si,2]);xp=down.xp
                self.sample_geometry[j]=(xp.asarray(tri.simplices[si]),xp.asarray(np.column_stack((bary,1-bary.sum(axis=1)))),xp.asarray(y),xp.asarray(z),xp.asarray(w))
                if backend=='cpu':
                    nodes=self.sample_geometry[j][0];unique,inverse=np.unique(nodes,return_inverse=True);self._node_gather[j]=(unique,inverse.reshape(nodes.shape))
            # Parent-first traversal, independent of user insertion order.
            self.order=[];pending=set(range(len(rows)))
            while pending:
                ready=sorted(j for j in pending if all(p[0] in self.order for p in self.parents[j]))
                if not ready:raise ValueError('Cyclic wake dependence is unsupported.')
                self.order.extend(ready);pending.difference_update(ready)
            self.identity.update(method='rotor_inflow.source_projection_linear.v1',wake_parameters=[r['control']['wake'] for r in rows],
              scope='straight zero-yaw wakes; linear zero-column-mean vertical correction; source-consistent rotor moments',
              support='lateral Gaussian tails outside four sigma plus receiver radius omitted; no streamwise mesh-length cutoff')
            if self._batch is not None:self._batch.prepare_moments(self.sample_geometry,self.parents,self.order)

        def _evaluate(self,q,z,grad):
            if not self.enabled:return [r.sample(q,z,grad) for r in self.rotors]
            # Native samples retain exact coverage/bed-clearance guards. Use a
            # neutral nonzero Ct carrier so its uniform surface solve cannot
            # reject a state that the subsequent nonuniform solve supports.
            samples=[];moments={};factors=np.ones(len(self.rotors))
            if self._batch is not None:samples=self._batch.sample(q,z,grad,neutral=True)
            else:
                for r in self.rotors:
                    original=r.c;r.c=dict(original,ct=.5,cp=0.,control=dict(mode='constant_ct'))
                    try:samples.append(r.sample(q,z,grad))
                    finally:r.c=original
            for j in self.order:
                r=self.rotors[j];xp=r.xp;s,k,u,v=samples[j];control=r.c['control']
                s.update(performance_model=control['model_name'],control_mode=control['mode'],commanded_rotational_speed_rpm=control['rotational_speed_rpm'],
                  requested_Ct=r.c['ct'],requested_Cp=r.c['cp'],
                  reconstruction_active=False,rotor_mean_velocity_m_s=0.,rotor_power_velocity_m_s=0.,rotor_thrust_velocity_m_s=0.,own_induction_residual_m_s=0.)
                op=control.get('operation')
                s.update(operating_speed_basis='resolved local normal velocity before own-induction correction',
                  electrical_efficiency=None if op is None else op['electrical_efficiency'],electrical_power_W=None if op is None else 0.)
                if not s['active']:continue
                if op is not None and not op['cut_in_m_s']<=s['local_velocity_m_s']<op['cut_out_m_s']:
                    reason='below_local_speed_cut_in' if s['local_velocity_m_s']<op['cut_in_m_s'] else 'at_or_above_local_speed_cut_out'
                    s.update(active=False,inactive_reason=reason,force_per_density_m4_s2=0.,useful_power_per_density_m5_s3=0.,hydraulic_power_removed_per_density_m5_s3=0.,thrust_coefficient=0.,power_coefficient=0.,rotational_speed_rpm=0.)
                    factors[j]=0.;samples[j]=(s,xp.zeros_like(k) if self._batch is None else k,u,v);continue
                if self._batch is not None:
                    if j not in moments:
                        layer=next(layer for layer in self._batch.layers if j in layer)
                        def enabled(i):
                            state=samples[i][0];window=self.rotors[i].c['control'].get('operation')
                            return state['active'] and (window is None or window['cut_in_m_s']<=state['local_velocity_m_s']<window['cut_out_m_s'])
                        selected=[i for i in layer if enabled(i)];data,counts=self._batch.moment_rows(q,samples,selected)
                        moments.update({i:(data[i].copy(),counts[i]) for i in selected})
                    row,count=moments[j];s['resolved_disk_minus_source_m_s']=float(row[8]);s['active_upstream_wakes']=count;s['reconstruction_active']=bool(count)
                    point=resolve_moments(control,r.c['D'],s['mean_depth_m'],s['local_velocity_m_s'],*map(float,row[:4]))
                else:
                    point=self._point(j,q,z,s,samples)
                if not point['active']:
                    s.update(active=False,inactive_reason=point['reason'],force_per_density_m4_s2=0.,useful_power_per_density_m5_s3=0.,hydraulic_power_removed_per_density_m5_s3=0.,thrust_coefficient=0.,power_coefficient=0.,rotational_speed_rpm=0.)
                    factors[j]=0.;samples[j]=(s,xp.zeros_like(k) if self._batch is None else k,u,v);continue
                factor=point['F']/s['force_per_density_m4_s2'];factors[j]=factor
                if self._batch is None:k=k*factor
                work=s['hydraulic_power_removed_per_density_m5_s3']*factor
                if point['P']>work*(1+1e-10):raise ValueError('Estimated shaft power exceeds applied hydraulic work; verify performance reference and rotor inflow.')
                s.update(upstream_velocity_m_s=point['U0'],rotor_mean_velocity_m_s=point['mean'],rotor_thrust_velocity_m_s=math.sqrt(point['second']),rotor_power_velocity_m_s=point['third']**(1/3),
                  tip_speed_ratio=point['TSR'],rotational_speed_rpm=control['rotational_speed_rpm'],thrust_coefficient=point['Ct'],power_coefficient=point['Cp'],
                  force_per_density_m4_s2=point['F'],useful_power_per_density_m5_s3=point['P'],hydraulic_power_removed_per_density_m5_s3=work,own_induction_residual_m_s=point['residual'])
                if op is not None:s['electrical_power_W']=point['P']*r.c['rho']*op['electrical_efficiency']
                samples[j]=(s,k,u,v)
            if self._batch is not None:self._batch.scale_sources(factors)
            return samples

        def _point(self,j,q,z,s,samples):
                r=self.rotors[j];xp=r.xp;control=r.c['control']
                nodes,bary,y,zrel,w=self.sample_geometry[j];h=q[0]
                unique,inverse=self._node_gather[j]
                if bool(xp.any(h[unique]<=r.dry)):raise ValueError('Rotor quadrature spans dry cells; refine or relocate the turbine.')
                un=(q[1,unique]*r.normal[0]+q[2,unique]*r.normal[1])/xp.maximum(h[unique],r.dry)
                sampled=xp.sum(bary*un[inverse],axis=1);offsets=sampled-s['local_velocity_m_s']
                s['resolved_disk_minus_source_m_s']=float(xp.sum(w*offsets))
                terms=[]
                for i,xD,lateral,sigma in self.parents[j]:
                    up=self.rotors[i];us=samples[i][0]
                    if us['active']:
                        H=s['mean_depth_m'];bed=float(xp.sum(r.normalized[r.ids]*z[r.ids]));hub=up.c['hub_elevation_m']-bed
                        if not 0<hub<H:raise ValueError('Wake axis outside receiving water column')
                        rad=1-us['thrust_coefficient']/(8*(sigma/up.c['D'])**2)
                        if rad<=0:raise ValueError('Gaussian wake outside real momentum domain')
                        amp=us['upstream_velocity_m_s']*(1-math.sqrt(rad))*xp.exp(-.5*((y+lateral)/sigma)**2)
                        terms.append((amp,hub,sigma))
                if terms:
                    offsets+=overlap_anomaly(terms,zrel+r.c['hub_elevation_m']-bed,H,control['wake']['method'],xp)
                    s['reconstruction_active']=True
                s['active_upstream_wakes']=len(terms)
                return resolve(control,r.c['D'],s['mean_depth_m'],s['local_velocity_m_s'],offsets,w,xp)

        def evaluate(self,q,z,grad):
            samples=self._evaluate(q,z,grad)
            if self._batch is None:return samples
            # Public samples must not alias reusable private kernel storage.
            return [(s,k.copy(),u.copy(),v.copy()) for s,k,u,v in samples]
        def apply(self,q,z,grad,rate):
            samples=self._evaluate(q,z,grad)
            if self._batch is not None:
                self._batch.apply(samples,rate);return [s[0] for s in samples]
            for rotor,(s,k,u,v) in zip(self.rotors,samples):
                speed=rotor.xp.hypot(u,v);rate[1,rotor.ids]-=k*speed*u;rate[2,rotor.ids]-=k*speed*v
            return [s[0] for s in samples]
        def sample(self,q,z,grad):return [a[0] for a in self._evaluate(q,z,grad)]
    return InflowArray

def overlap_anomaly(terms,z,H,method,xp=np):
    """Only unresolved vertical structure: never subtract the SWE deficit twice."""
    if method not in ('gaussian_zero_mean_single_wake','gaussian_zero_mean_linear_overlap'):
        raise ValueError('Use the supported coupled linear wake correction.')
    result=xp.zeros_like(z)
    for amp,hub,sigma in terms:
        result+=amp*(_mean(H,hub,sigma)-xp.exp(-.5*((z-hub)/sigma)**2))
    return result
