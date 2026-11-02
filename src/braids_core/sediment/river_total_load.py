"""Explicit FLOW algebraic-total assembly on the shared river owner."""
from pathlib import Path

HERE=Path(__file__).parent
ZERO_SUSPENDED=(2,12,14,16,17,18)


def body():
    return (HERE/"river_total_load.h").read_text()


def capacity_source(source, replace_once):
    # Clone the current bed-only cell entry, then ADD the equilibrium suspended
    # vector after scalar bed steering. No change to the established entries.
    cell=source[source.index("ENTRY void split_capacity"):source.index("ENTRY void sus_faces")]
    cell=replace_once(cell,"ENTRY void split_capacity","ENTRY void total_capacity")
    cell=replace_once(cell,"double*taurat){",
        "double*taurat,const double*fraction,const double*dss){")
    needle="out[4*n+i]=bs>0.?mag*by/bs:0.;out[5*n+i]=alpha;"
    cell=replace_once(cell,needle,needle+"""
  double sm,sx,sy,nextdss;
  if(river_total_suspended(law_params,h,u,v,cq[n+i]/cq[i],cq[2*n+i]/cq[i],
     c[i],d,d90[i],hide[i],z0[i],fraction[i],dss[i],sm,sx,sy,nextdss)){
   status[i]=4;continue;
  }
  out[i]+=sx;out[n+i]+=sy;out[3*n+i]+=sx;out[4*n+i]+=sy;
""")
    return source+"\n"+body()+"\n"+cell+"""
ENTRY void total_suspended_probe(int n,double d,const double*q,const double*cq,
 const double*c,const double*d90,const double*hide,const double*z0,const double*fraction,
 const double*dss,const double*law,double*out,int*status){
 LOOP(n){double h=q[i],u=q[n+i]/h,v=q[2*n+i]/h,sm,sx,sy,nd;
  status[i]=river_total_suspended(law,h,u,v,cq[n+i]/cq[i],cq[2*n+i]/cq[i],c[i],
    d,d90[i],hide[i],z0[i],fraction[i],dss[i],sm,sx,sy,nd);
  out[i]=sx;out[n+i]=sy;out[2*n+i]=nd;out[3*n+i]=sm;
 }
}
ENTRY void total_radial_probe(int n,double d,const double*q,const double*factor,
 const double*c,const double*d90,const double*hide,const double*z0,const double*fraction,
 const double*dss,const double*law,double*out,int*status){
 LOOP(n){double h=q[i],sm,der;
  status[i]=river_total_suspended_radial(law,h,q[n+i]/h,q[2*n+i]/h,factor[i],c[i],
    d,d90[i],hide[i],z0[i],fraction[i],dss[i],sm,der);
  out[i]=sm;out[n+i]=der;
 }
}
"""


def wave_source(source,replace_once,*,vector,vr1993):
    source=replace_once(source,"ENTRY void wave_state_gradients",body()+"\nENTRY void wave_state_gradients")
    source=replace_once(source,"const double*law_params,const double*packing,",
        "const double*law_params,const double*packing,const double*total_frac,const double*total_dss,")
    needle="  if(!(h>0.)"
    pos=source.index(needle)
    source=source[:pos]+"""
  double sm,sd;
  // Use GLOBAL components for formula direction, including Soulsby angular terms.
  double gu=u*nx-v*ny,gv=u*ny+v*nx;
  // VR bedload uses physical speed, but eqtran folds concentration with EFUMN.
  double vf=.41*cmean/(sqrt(9.81)*log1p(h*double(.368f)/zrough));
  if(river_total_suspended_radial(law_params,h,gu,gv,vf,cwork,diameter,dm90,
      .5*(hide[o]+hide[j]),zrough,.5*(total_frac[o]+total_frac[j]),
      .5*(total_dss[o]+total_dss[j]),sm,sd)){diagnostic[2*nf+i]=8.;continue;}
  sm/=law_params[13];sd/=law_params[13];
  mag+=sm;
""" + ("""
  // Positive radial principal part uses the same summed physical flux.
  // Actual secondary steering remains in the reconstructed physical vector.
  derivative=derivative*velocity_factor+sd;velocity_factor=1.;
""" if not vector or vr1993 else """
  // For current-only Soulsby--van Rijn, both magnitudes have the same speed
  // dependence. Convert the physical radial derivative back to EF components.
  double ps=sqrt(sp2),eps=double(1.e-6f);
  double dv=vf*ps*(ps+2.*eps)/((ps+eps)*(ps+eps));
  if(ps>0.&&dv>0.){gx+=sd/dv*gu/ps;gy+=sd/dv*gv/ps;}
""")+source[pos:]
    return source
