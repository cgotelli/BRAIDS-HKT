"""Fixed-axis, rotor-normal Thetis actuator extension.

Normal loading follows Zhang et al. (2025), JMSE 13, 1210, with the
Kramer--Piggott (2016) discrete-depth induction used by Thetis. This is an
explicit extension: stock Thetis supplies an isotropic drag, not yaw.
No fitted lateral steering force or analytical wake is added. Device Cp is
shaft power, distinct from momentum-source work. Both curves use incoming
normal speed at the specified device/controller conditions.
"""
import hashlib
from pathlib import Path
import numpy as np
from .river_turbine_flow import CoveredRotor
from .river_turbine_thetis import cell_terms, validate_control, curve_rows, stop_below_cut_in, REFERENCE

METHOD='actuator.thetis_rotor_normal.v1'


class FixedAxisRotor(CoveredRotor):
    def __init__(self,cells,c,backend,**kwargs):
        self.original=dict(c);self.control=validate_control(c['control'])
        if self.control['mode']=='thetis_speed_table':
            raise ValueError('Fixed-axis loading needs incoming normal-speed device curves or constants.')
        if c.get('power_estimation',{'method':'thetis'})!={'method':'thetis'}:
            raise ValueError('Non-yawed Gaussian power is not qualified for fixed-axis loading.')
        n=np.asarray(c['normal_xy'],float)
        if n.shape!=(2,) or not np.isfinite(n).all() or abs(np.linalg.norm(n)-1)>1e-10:
            raise ValueError('Fixed rotor normal must be a finite unit vector.')
        neutral=dict(c,ct=0.,cp=0.,control={'mode':'constant_ct'})
        super().__init__(cells,neutral,backend,**kwargs)
        self.density_integrals=self.weights/((c['D']/2)**2*1.45661)
        self.identity.update(method=METHOD,reference_commit=REFERENCE,physical=self.original,
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            force='-density A Ct/(2 alpha(H)^2) |u dot n| (u dot n) n',
            power='integral .5 rho A Cp(Un) Un^3 density; Un=(u dot n)/alpha',
            support='rotor-oriented D-by-D Thetis exponential bump; same 1.45661 normalization',
            direction='fixed normal; front-side operation only; local yaw reported',
            wake='resolved SWE interactions; no analytical deficit or empirical lateral force')

    def sample(self,q,z,grad):
        from .turbine_constant_batch import eligible,ConstantRotorBatch
        if eligible(self):
            if not hasattr(self,'_constant_batch'):self._constant_batch=ConstantRotorBatch((self,))
            return self._constant_batch.evaluate(q,z,grad)[0]
        if self.xp is not np and len(self.ids)<=256 and len(self.plane)<=512:
            from .turbine_sample_pack import sample
            return self._sample_packed(q,z,grad,sample(self,q,z,grad))
        return self._sample_packed(q,z,grad,None)

    def _sample_packed(self,q,z,grad,packed,power_batch=None):
        xp=self.xp;ids=self.ids;nx,ny=self.normal
        if packed is not None:
            h,w,un,ut,stats=packed
        else:
            h=q[0,ids];w=self.normalized[ids]
            u=q[1,ids]/xp.maximum(h,self.dry);v=q[2,ids]/xp.maximum(h,self.dry)
            un=u*nx+v*ny;ut=-u*ny+v*nx
            row=self.plane;eta=q[0,row]+z[row]
            surface=eta+grad[4,row]*self.delta[0]+grad[5,row]*self.delta[1]
            bed=z[row]+grad[6,row]*self.delta[0]+grad[7,row]*self.delta[1]
            stats=xp.stack((xp.min(xp.minimum(surface,eta)),xp.max(xp.maximum(bed,z[row])),
                            xp.any(h<=self.dry),xp.sum(w*un),xp.sum(w*h),xp.sum(w*ut)))
            if xp is not np:stats=stats.get()
        d,_,_=self._operating_state(tuple(map(float,stats[:6])))
        k=power_batch[0] if power_batch is not None and d['active'] else xp.zeros_like(h);fx=fy=0.
        if d['active']:
            # A rotor operating in mixed forward/reverse flow needs blade/control
            # data outside the admitted normal-speed approximation.
            if (stats[6]<-1e-10 if packed is not None else bool(xp.any(un < -1e-10))):
                raise ValueError(f'Turbine "{self.c["name"]}": mixed forward/reverse flow across fixed rotor. Move the rotor out of recirculation; rear-side operation is unsupported.')
            if self.control['mode']!='constant_ct':
                rows=curve_rows(self.control,'thrust_coefficient');first=rows[0];last=rows[-1]
                if power_batch is not None:
                    above=bool(power_batch[5][0])
                else:
                    limit=last['speed_m_s']*.5*(1+xp.sqrt(1-np.pi*self.c['D']*last['thrust_coefficient']/(4*h)))
                    above=bool(xp.any(un>=limit))
                if above:
                    raise ValueError(f'Turbine "{self.c["name"]}": local normal velocity exceeds the induction-corrected upper range of its incoming-speed Ct curve (incoming range {first["speed_m_s"]:g} to {last["speed_m_s"]:g} m/s, upper bound exclusive). Supply supported device data at higher speeds. Speeds below the lower bound stop the rotor with zero power; speeds above the upper bound cannot be extrapolated.')
        stop_below_cut_in(d,self.control,self.c['D'],h,un,xp,below_cut_in=bool(power_batch[5][1]) if power_batch is not None else None)
        if not d['active'] and power_batch is not None:k=xp.zeros_like(h)
        if d['active']:
            if power_batch is not None:
                if self.control['mode']!='constant_ct' and power_batch[5][2]:
                    raise ValueError('Incident-speed device curves require a fully covered rotor (local depth at least D).')
                k,values,invalid=power_batch[:3]
                if invalid:
                    raise ValueError('Thetis induction correction is outside its real domain: Ct*A exceeds rotor diameter times local water depth. Check turbine diameter, depth and Ct data.')
            else:
                K,power,incident,ct,cp=cell_terms(self.control,self.c['D'],h,un,xp.zeros_like(un),xp)
                den=self.density_integrals[ids];k=K*den/self.area[ids]
                values=xp.stack((xp.sum(K*den*un**2),xp.sum(K*den*abs(un)**3),xp.sum(power*den),
                                xp.sum(w*incident),xp.sum(w*ct),xp.sum(w*cp)))
                if xp is not np:values=values.get()
            force,work,P,incoming,ct_mean,cp_mean=map(float,values)
            if P>work+1e-10*max(1.,work):
                raise ValueError(f'Turbine "{self.c["name"]}": supplied Cp extracts more power than normal loading removes. Check Ct/Cp reference speed and device/controller data.')
            fx=-force*nx;fy=-force*ny
            d.update(force_per_density_m4_s2=force,hydraulic_power_removed_per_density_m5_s3=work,
                     useful_power_per_density_m5_s3=P,upstream_velocity_m_s=incoming,
                     thrust_coefficient=ct_mean,power_coefficient=cp_mean)
        d.update(method=METHOD,control_mode=self.control['mode'],
            performance_model=self.control.get('model_name','Constant normal-speed coefficients'),
            requested_Ct=self.control.get('thrust_coefficient'),requested_Cp=self.control.get('power_coefficient'),
            power_basis='normal incident-speed rotor mechanical power; generator losses excluded',
            coefficient_basis='Ct and Cp referenced to incoming normal speed; controller/RPM must match the device curves',
            local_normal_velocity_m_s=float(stats[3]),local_tangent_velocity_m_s=float(stats[5]),
            local_yaw_deg=float(np.degrees(np.arctan2(-stats[5],stats[3]))),
            rotor_normal_x=float(nx),rotor_normal_y=float(ny),
            force_x_per_density_m4_s2=float(fx),force_y_per_density_m4_s2=float(fy),
            reconstruction_active=False,active_upstream_wakes=None,
            electrical_efficiency=None,electrical_power_W=None)
        # CoveredRotor.apply then computes exactly -k |un| un n, never -k |u|u.
        if power_batch is not None:return d,k,power_batch[3],power_batch[4]
        return d,k,un*nx,un*ny
