"""Fixed-axis, rotor-normal Thetis actuator extension.

Normal loading follows Zhang et al. (2022), IMEJ 5, 273-280, with the
Kramer--Piggott (2016) discrete-depth induction used by Thetis. This is an
explicit extension: stock Thetis supplies an isotropic drag, not yaw.
No fitted lateral steering force or analytical wake is added. Device Cp is
shaft power, distinct from momentum-source work. Zero-yaw curves use incoming
SPEED MAGNITUDE at matching controller/RPM; P=P0(U,RPM)*cos(yaw)^3.
This fixed-pitch approximation is tested by Gambuzza et al. (2025), section4.2.
The source-normal induction correction leaves the tangential component
unchanged. Its use in nonuniform wakes is an explicit mean-flow assumption.
"""
import hashlib
from pathlib import Path
import numpy as np
from .river_turbine_flow import CoveredRotor
from .river_turbine_thetis import cell_terms, validate_control, curve_rows, curve_arrays, REFERENCE

METHOD='actuator.zhang_normal_cosine_power.v2'
from . import river_turbine_device_yaw as device_yaw
_YAW_KERNEL=None


def bounds(control,diameter,depth,un,ut,xp):
    """Normal local bounds for zero-yaw curves referenced to incident magnitude."""
    rows=curve_rows(control,'thrust_coefficient');a,b=rows[0],rows[-1]
    c=np.pi*diameter/(4*depth)
    def normal(row):
        return .5*(1+xp.sqrt(1-c*row['thrust_coefficient']))*xp.sqrt(xp.maximum(row['speed_m_s']**2-ut**2,0.))
    return (un<normal(a)), ((abs(ut)>=b['speed_m_s']) | (un>=normal(b)))


def yaw_terms(control,diameter,depth,un,ut,xp=np):
    """K,P/rho,U,Ct,Cp. Force=-K |un|un n; power uses incoming normal cubed.

    Solve un=alpha(Ct(U),H)*sqrt(U^2-ut^2). The published small-yaw
    approximation has no direct tangential loading. Monotonicity follows
    from the existing U*alpha(U) curve guard, because U/sqrt(U^2-ut^2)>=1.
    Bounds are checked by the owner BEFORE this operator; no extrapolation.
    """
    global _YAW_KERNEL
    if control['mode']=='constant_ct':
        K,P,normal,ct,cp=cell_terms(control,diameter,depth,un,xp.zeros_like(un),xp)
        return K,P,xp.hypot(normal,ut),ct,cp
    xs,cts=curve_arrays(control,'thrust_coefficient',xp);px,cps=curve_arrays(control,'power_coefficient',xp)
    if xp is not np:
        if _YAW_KERNEL is None:
            _YAW_KERNEL=xp.ElementwiseKernel(
                'float64 D,float64 H,float64 un,float64 ut,raw float64 xs,raw float64 cts,int32 count,raw float64 px,raw float64 cps,int32 pcount',
                'float64 K,float64 P,float64 U,float64 Ct,float64 Cp',r'''
                double c=3.14159265358979323846*D/(4*H),lo=fmax(xs[0],fabs(ut)),hi=xs[count-1];
                for(int it=0;it<44;it++){
                    double mid=.5*(lo+hi);int a=0,b=count-1;
                    while(b-a>1){int m=(a+b)/2;if(xs[m]<=mid)a=m;else b=m;}
                    double ct=cts[a]+(cts[b]-cts[a])*(mid-xs[a])/(xs[b]-xs[a]);
                    double local=.5*(1+sqrt(1-c*ct))*sqrt(fmax(mid*mid-ut*ut,0.));
                    if(local<un)lo=mid;else hi=mid;
                }
                U=.5*(lo+hi);int a=0,b=count-1;
                while(b-a>1){int m=(a+b)/2;if(xs[m]<=U)a=m;else b=m;}
                Ct=cts[a]+(cts[b]-cts[a])*(U-xs[a])/(xs[b]-xs[a]);
                a=0;b=pcount-1;while(b-a>1){int m=(a+b)/2;if(px[m]<=U)a=m;else b=m;}
                Cp=cps[a]+(cps[b]-cps[a])*(U-px[a])/(px[b]-px[a]);
                double alpha=.5*(1+sqrt(1-c*Ct)),A=3.14159265358979323846*D*D/4,n=un/alpha;
                K=A*Ct/(2*alpha*alpha);P=.5*A*Cp*n*n*n;
                ''','braids_zhang_cosine_device_v2')
        return _YAW_KERNEL(diameter,depth,un,ut,xs,cts,len(xs),px,cps,len(px))
    c=np.pi*diameter/(4*depth);lo=xp.maximum(xs[0],abs(ut));hi=xp.full_like(un,xs[-1])
    for _ in range(44):
        mid=.5*(lo+hi);ct=xp.interp(mid,xs,cts)
        local=.5*(1+xp.sqrt(1-c*ct))*xp.sqrt(xp.maximum(mid**2-ut**2,0.))
        below=local<un;lo=xp.where(below,mid,lo);hi=xp.where(below,hi,mid)
    U=.5*(lo+hi);ct=xp.interp(U,xs,cts);cp=xp.interp(U,px,cps)
    alpha=.5*(1+xp.sqrt(1-c*ct));A=np.pi*diameter**2/4
    return A*ct/(2*alpha**2),.5*A*cp*(un/alpha)**3,U,ct,cp


class FixedAxisRotor(CoveredRotor):
    def __init__(self,cells,c,backend,**kwargs):
        self.original=dict(c);self.control=validate_control(c['control'])
        if self.control['mode']=='thetis_speed_table':
            raise ValueError('Fixed-axis loading needs zero-yaw incoming-speed device curves at matching controller/RPM, or constants.')
        if c.get('power_estimation',{'method':'thetis'}).get('method') not in ('thetis',device_yaw.METHOD,'power.gaussian_device_yaw.v1','power.dou2019_device_yaw.v1'):
            raise ValueError('Non-yawed Gaussian power is not qualified for fixed-axis loading.')
        setting=c.get('power_estimation',{})
        self.device_setting=setting.get('device') if setting.get('method') in ('power.gaussian_device_yaw.v1','power.dou2019_device_yaw.v1') else setting
        self.device_yaw=self.device_setting.get('method')==device_yaw.METHOD
        if self.device_yaw:
            device_yaw.validate(self.device_setting)
            if abs(c['D']-self.device_setting['diameter_m'])>1e-12:raise ValueError('Yaw device diameter mismatch.')
        n=np.asarray(c['normal_xy'],float)
        if n.shape!=(2,) or not np.isfinite(n).all() or abs(np.linalg.norm(n)-1)>1e-10:
            raise ValueError('Fixed rotor normal must be a finite unit vector.')
        neutral=dict(c,ct=0.,cp=0.,control={'mode':'constant_ct'})
        super().__init__(cells,neutral,backend,**kwargs)
        self.density_integrals=self.weights/((c['D']/2)**2*1.45661)
        self.identity.update(method=METHOD,reference_commit=REFERENCE,physical=self.original,
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            force='-density A Ct/(2 alpha(H)^2) |u dot n| (u dot n) n',
            power='integral .5 rho A Cp(U) (U cos(yaw))^3 density; U=hypot(un/alpha,ut); Ct(U)',
            support='rotor-oriented D-by-D Thetis exponential bump; same 1.45661 normalization',
            direction='fixed normal; front-side operation only; local yaw reported',
            wake='resolved SWE interactions; no analytical deficit or empirical lateral force')

        if self.device_yaw:
            self.identity.update(method='actuator.normal_device_bem_power.v1',power='Cp0(U,controller)*U^3*device BEM yaw/zero ratio; source loading unchanged; no additional wake deficit')

    def sample(self,q,z,grad):
        if self.xp is not np:
            from .turbine_yaw_batch import YawRotorBatch
            if not hasattr(self,'_yaw_batch'):self._yaw_batch=YawRotorBatch((self,))
            return self._yaw_batch.evaluate(q,z,grad)[0]
        return self._sample_packed(q,z,grad,None)

    def _sample_packed(self,q,z,grad,packed):
        xp=self.xp;ids=self.ids;nx,ny=self.normal
        if packed is not None:
            h,w,un,ut,stats,batch_k,batch_values,batch_bounds=packed
        else:
            xp=self.xp;ids=self.ids;h=q[0,ids];w=self.normalized[ids]
            u=q[1,ids]/xp.maximum(h,self.dry);v=q[2,ids]/xp.maximum(h,self.dry)
            nx,ny=self.normal;un=u*nx+v*ny;ut=-u*ny+v*nx
            row=self.plane;eta=q[0,row]+z[row]
            surface=eta+grad[4,row]*self.delta[0]+grad[5,row]*self.delta[1]
            bed=z[row]+grad[6,row]*self.delta[0]+grad[7,row]*self.delta[1]
            stats=xp.stack((xp.min(xp.minimum(surface,eta)),xp.max(xp.maximum(bed,z[row])),
                            xp.any(h<=self.dry),xp.sum(w*un),xp.sum(w*h),xp.sum(w*ut)))
            if xp is not np:stats=stats.get()
        d,_,_=self._operating_state(tuple(map(float,stats)))
        k=xp.zeros_like(h);fx=fy=0.
        if d['active']:
            # A rotor operating in mixed forward/reverse flow needs blade/control
            # data outside the admitted normal-speed approximation.
            if (stats[6]<-1e-10 if packed is not None else bool(xp.any(un < -1e-10))):
                raise ValueError(f'Turbine "{self.c["name"]}": mixed forward/reverse flow across fixed rotor. Move the rotor out of recirculation; rear-side operation is unsupported.')
            if self.control['mode']!='constant_ct':
                rows=curve_rows(self.control,'thrust_coefficient');first=rows[0];last=rows[-1]
                low,high=bounds(self.control,self.c['D'],h,un,ut,xp) if packed is None else batch_bounds
                if (bool(high) if packed is not None else bool(xp.any(high))):
                    raise ValueError(f'Turbine "{self.c["name"]}": incident speed magnitude exceeds the induction-corrected upper range of its zero-yaw incoming-speed Ct curve (incoming range {first["speed_m_s"]:g} to {last["speed_m_s"]:g} m/s, upper bound exclusive). Supply supported device data at higher speeds. Speeds below the lower bound stop the rotor with zero power; speeds above the upper bound cannot be extrapolated.')
                if (bool(low) if packed is not None else bool(xp.any(low))):
                    d.update(active=False,inactive_reason='below_cut_in_speed',rotational_speed_rpm=0.,
                        thrust_coefficient=0.,power_coefficient=0.,force_per_density_m4_s2=0.,
                        useful_power_per_density_m5_s3=0.,hydraulic_power_removed_per_density_m5_s3=0.)
        if d['active'] and self.device_yaw and packed is None:
            _,_,incident0,_,_=yaw_terms(self.control,self.c['D'],h,un,ut,xp)
            if bool(xp.any(incident0<self.device_setting['speed_m_s'][0])):
                d.update(active=False,inactive_reason='below_cut_in_speed',rotational_speed_rpm=0.,thrust_coefficient=0.,power_coefficient=0.,force_per_density_m4_s2=0.,useful_power_per_density_m5_s3=0.,hydraulic_power_removed_per_density_m5_s3=0.)
        if d.get('inactive_reason')=='no_incoming_flow':d['rotational_speed_rpm']=0.
        if d['active']:
            if packed is not None:
                k=batch_k;values=batch_values
            else:
                K,power,incident,ct,cp=yaw_terms(self.control,self.c['D'],h,un,ut,xp)
                if self.device_yaw:power,cp=device_yaw.correct(self.device_setting,power,incident,ut,cp,xp)
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
            performance_model=self.control.get('model_name','Constant zero-yaw coefficients with cosine yaw reduction'),
            requested_Ct=self.control.get('thrust_coefficient'),requested_Cp=self.control.get('power_coefficient'),
            power_basis='P0(incoming magnitude,controller) cos(yaw)^3; rotor mechanical power; generator losses excluded',
            coefficient_basis='Zero-yaw Ct/Cp at incident speed magnitude and matching controller/RPM; normal induction, tangential velocity unchanged',
            local_normal_velocity_m_s=float(stats[3]),local_tangent_velocity_m_s=float(stats[5]),
            local_yaw_deg=float(np.degrees(np.arctan2(-stats[5],stats[3]))),
            rotor_normal_x=float(nx),rotor_normal_y=float(ny),
            force_x_per_density_m4_s2=float(fx),force_y_per_density_m4_s2=float(fy),
            reconstruction_active=False,active_upstream_wakes=None,
            electrical_efficiency=None,electrical_power_W=None)
        if self.device_yaw:
            d.update(method='actuator.normal_device_bem_power.v1',power_basis='Cp0(U,controller) U^3 times explicit BEM yaw/zero ratio; rotor shaft power; no extra cosine or wake deficit',coefficient_basis='Ct0 uses incident normal squared loading; reported Cp includes device yaw ratio and references incident magnitude cubed',performance_model=self.device_setting['model_name'])
            if d['active']:d['rotational_speed_rpm']=self.device_setting['rpm']
        # CoveredRotor.apply then computes exactly -k |un| un n, never -k |u|u.
        return d,k,un*nx,un*ny
