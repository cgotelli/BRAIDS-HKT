"""Shared scalar rotor performance: shaft-power curves at a prescribed RPM.

The incoming-velocity correction depends on Ct, while Ct depends on TSR.
Resolve that dependence together rather than lagging yesterday's coefficient.
This is a curve-driven actuator model, not a blade-geometry/BEM prediction.
"""
import math
import numpy as np


def validate(control):
    fields={'mode','model_name','rotational_speed_rpm','performance','outside_curve'}
    if not isinstance(control,dict) or set(control)!=fields or control['mode']!='fixed_rpm_table':
        raise ValueError('RPM turbine model requires its name, RPM, TSR/Ct/Cp table and explicit outside_curve policy.')
    if not isinstance(control['model_name'],str) or not control['model_name'].strip():
        raise ValueError('Give the turbine performance model a name identifying its curve source.')
    rpm=control['rotational_speed_rpm']
    if type(rpm) not in (int,float) or not math.isfinite(rpm) or rpm<0:
        raise ValueError('Turbine RPM must be finite and nonnegative; zero parks the rotor.')
    if control['outside_curve']!='off':raise ValueError('Outside the supplied TSR range the turbine must switch off; no curve extrapolation is qualified.')
    rows=control['performance']
    if not isinstance(rows,list) or len(rows)<2:raise ValueError('Supply at least two TSR/Ct/Cp rows for the turbine model.')
    table=[]
    for row in rows:
        if not isinstance(row,dict) or set(row)!={'tip_speed_ratio','thrust_coefficient','power_coefficient'}:
            raise ValueError('Each turbine curve row requires TSR, Ct and mechanical Cp.')
        x,ct,cp=(row[k] for k in ['tip_speed_ratio','thrust_coefficient','power_coefficient'])
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (x,ct,cp)):
            raise ValueError('Turbine curve entries must be finite numbers.')
        if x<0 or not 0<=ct<1 or not 0<=cp<=16/27:raise ValueError('Turbine curves require TSR ≥ 0, 0 ≤ Ct < 1 and 0 ≤ Cp ≤ 16/27.')
        if cp>ct*.5*(1+math.sqrt(1-ct))+1e-12:
            raise ValueError('Mechanical Cp exceeds the ideal actuator power available at this Ct. Correct the curve/reference convention.')
        table.append(dict(tip_speed_ratio=float(x),thrust_coefficient=float(ct),power_coefficient=float(cp)))
    table.sort(key=lambda r:r['tip_speed_ratio'])
    if any(a['tip_speed_ratio']==b['tip_speed_ratio'] for a,b in zip(table,table[1:])):raise ValueError('Turbine curve TSR values must be unique.')
    return dict(control,model_name=control['model_name'].strip(),rotational_speed_rpm=float(rpm),performance=table)


def operating_point(control,diameter,depth,local_normal_velocity):
    """Resolve all admissible piecewise-linear branches; reject ambiguity.

    x=alpha_num=U_local/U_in, k=omega*R/U_local, lambda=k*x.
    Ct=m*lambda+b on each interval. Squaring the momentum relation gives
    4*x*x + (K*m*k-4)*x + K*b = 0, K=A/(D*H).
    Candidates must also satisfy the original unsquared equation and bounds.
    """
    rpm=control['rotational_speed_rpm'];u=float(local_normal_velocity)
    if rpm==0:return dict(active=False,reason='parked',rotational_speed_rpm=rpm)
    if u<=1e-10:return dict(active=False,reason='reverse_flow' if u<0 else 'no_incoming_flow',rotational_speed_rpm=rpm)
    K=math.pi*diameter/(4*depth);k=(rpm*math.pi/30)*(diameter/2)/u;roots=[]
    rows=control['performance']
    for left,right in zip(rows,rows[1:]):
        lo=left['tip_speed_ratio'];hi=right['tip_speed_ratio'];m=(right['thrust_coefficient']-left['thrust_coefficient'])/(hi-lo);b=left['thrust_coefficient']-m*lo
        B=K*m*k-4;C=K*b;disc=B*B-16*C
        if disc < -64*np.finfo(float).eps*max(1.,B*B,abs(16*C)):continue
        q=-.5*(B+math.copysign(math.sqrt(max(0.,disc)),B))
        for x in ([q/4,C/q] if q else [0.]):
            tsr=k*x
            if not .5-1e-12<=x<=1+1e-12 or not lo-1e-10<=tsr<=hi+1e-10:continue
            tsr=min(hi,max(lo,tsr));ct=left['thrust_coefficient']+m*(tsr-lo)
            if not 0<=K*ct<1:continue
            alpha=.5*(1+math.sqrt(1-K*ct))
            if abs(tsr-k*alpha)>1e-9*max(1.,tsr):continue
            if any(abs(tsr-r['tip_speed_ratio'])<1e-9*max(1.,tsr) for r in roots):continue
            cp=left['power_coefficient']+(right['power_coefficient']-left['power_coefficient'])*(tsr-lo)/(hi-lo)
            roots.append(dict(active=True,reason=None,tip_speed_ratio=tsr,ct=ct,cp=cp,upstream_velocity_m_s=u/alpha,rotational_speed_rpm=rpm))
    if len(roots)>1:raise ValueError('The turbine curve gives multiple self-consistent operating points. Correct its Ct curve or use a different RPM; BRAIDS will not choose a branch silently.')
    return roots[0] if roots else dict(active=False,reason='outside_performance_curve',rotational_speed_rpm=rpm)
