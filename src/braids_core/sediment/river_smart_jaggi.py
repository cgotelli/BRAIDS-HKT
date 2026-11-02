"""Single-grain Smart-Jaeggi energy-slope forcing for the shared CPU/CUDA law.

Reference: BASEHPC 4.0.2 Reference Manual, section 1.2.2.3.4, Eqs 1.29-1.31.
This is not a FLOW formula/selector. No BASEMENT software is used or copied.
qb = A/Delta * (d90/d30)^0.2 * J^0.6 * h*|U| * max(J-Jcr,0),
J = |U|^2/(C_hyd^2*h), Jcr = ThetaC*Delta*dm/h. Dm, D30 and D90 are measured inputs.
The field API exports mass flux; scalar law exports solid-volume flux. Only
the shared owner converts density, surface fraction, packing and morph factor.
"""


def capacity_entry(source, replace):
    cell = source[source.index("ENTRY void split_capacity"):source.index("ENTRY void sus_faces")]
    cell = replace(cell, "ENTRY void split_capacity", "ENTRY void smart_capacity")
    cell = replace(cell, "const double*z0,double*out", "const double*z0,const double*hydraulic_c,double*out")
    start = cell.index("  double speed=hypot(ue,ve)")
    end = cell.index("  double mag=bed_factor*law_params[13]*bed;", start)
    cell = cell[:start]+"""  // Consume the real hydraulic drag, not erosed's depth-floor/log-law mapping.
  double speed=hypot(u,v),bed,sus,derivative;
  if(!(hydraulic_c[i]>0.)||!isfinite(hydraulic_c[i])){status[i]=2;continue;}
  double theta=speed*speed/(hydraulic_c[i]*hydraulic_c[i]*law_params[14]*d);
  river_scalar_law(law_params,speed,hydraulic_c[i],h,d,d,1.,bed,sus,derivative);
  if(!isfinite(bed)||!isfinite(derivative)){status[i]=2;continue;}
  ue=u;ve=v;
"""+cell[end:]
    return cell


def wave_adapter(source, replace, *, depth_power=.6):
    source = replace(source, "const double*law_params,const double*packing,",
                     "const double*hydraulic_c,const double*law_params,const double*packing,")
    source = replace(source,
        "river_scalar_law(law_params,ve,cwork,fmax(.01,h),diameter,dm90,.5*(hide[o]+hide[j]),mag,sus,derivative,shear);",
        "double physical_c=1./sqrt(.5*(1./(hydraulic_c[o]*hydraulic_c[o])+1./(hydraulic_c[j]*hydraulic_c[j])));\n"
        "  river_scalar_law(law_params,sqrt(sp2),physical_c,h,diameter,diameter,1.,mag,sus,derivative);\n"
        "  velocity_factor=1.;")
    # Derivative with respect to conserved (h,hu,hv), not primitive velocity.
    if depth_power==0.:return source
    if depth_power!=.6:raise ValueError('Unqualified depth derivative')
    return replace(source, "der[0]=-radial*u;",
                   "der[0]=-radial*u-.6*acceleration*mag*u/(h*sqrt(sp2));")
