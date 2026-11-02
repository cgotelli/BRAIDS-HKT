"""Exact shallow-water pressure at an impermeable reflecting wall.

Only the physical wall momentum flux is replaced. Keep the augmented f-wave
interior solver, its wall speed estimate, bed-source quadrature, reconstruction
and conservative zero wall mass/tangential flux. Outward momentum is positive.
Arithmetic is paired on CPU/CUDA and qualified in Desktop170--171.
"""
METHOD='exact_reflecting_wall_riemann_v1'

CUDA_HELPER=r'''
__device__ double braids_wall_pressure(double h,double momentum,double g,double dry) {
    if(h<=dry) return 0.;
    const double c=sqrt(g*h), v=momentum/h/c;
    double ratio;
    if(v<=0.) {const double a=fmax(0.,1.+.5*v);ratio=a*a;}
    else {
        double lo=1.,hi=1.+sqrt(2.)*v;
        double r=1.+v;
        for(int k=0;k<32;++k) {
            const double s=sqrt(.5*(1.+1./r));
            const double f=(r-1.)*s-v;
            if(fabs(f)<=2.e-15*fmax(1.,v)) break;
            if(f>0.) hi=r;else lo=r;
            const double df=s-(r-1.)/(4.*s*r*r);
            const double trial=r-f/df;
            r=(trial>lo && trial<hi) ? trial : .5*(lo+hi);
        }
        ratio=r;
    }
    const double star=h*ratio;return .5*g*star*star;
}
'''
FORTRAN_HELPER=r'''
 real(c_double) function braids_wall_pressure(h,momentum,g,dry) bind(C)
  use iso_c_binding, only:c_double
  real(c_double),value::h,momentum,g,dry
  real(c_double)::c,v,ratio,a,lo,hi,r,s,ff,df,trial,star
  integer::k
  braids_wall_pressure=0.d0
  if(h<=dry)return
  c=sqrt(g*h);v=momentum/h/c
  if(v<=0.d0)then
   a=max(0.d0,1.d0+.5d0*v);ratio=a*a
  else
   lo=1.d0;hi=1.d0+sqrt(2.d0)*v;r=1.d0+v
   do k=1,32
    s=sqrt(.5d0*(1.d0+1.d0/r));ff=(r-1.d0)*s-v
    if(abs(ff)<=2.d-15*max(1.d0,v))exit
    if(ff>0.d0)then
     hi=r
    else
     lo=r
    endif
    df=s-(r-1.d0)/(4.d0*s*r*r);trial=r-ff/df
    if(trial>lo.and.trial<hi)then
     r=trial
    else
     r=.5d0*(lo+hi)
    endif
   enddo
   ratio=r
  endif
  star=h*ratio;braids_wall_pressure=.5d0*g*star*star
 end function
'''


def native_source(code):
    old='fl(1,f)=0;fl(2:3,f)=dot_product(fl(2:3,f),normal(:,f))*normal(:,f);fr(:,f)=fl(:,f)'
    if code.count(old)!=2:raise ValueError('Reflecting-wall native assembly layout changed')
    new='fl(1,f)=0;fl(2:3,f)=braids_wall_pressure(l(1),dot_product(l(2:3),normal(:,f)),grav,dry_tolerance)*normal(:,f);fr(:,f)=fl(:,f)'
    code=code.replace(old,new);end=code.rindex('\nend module')
    return code[:end]+FORTRAN_HELPER+code[end:]


def cuda_source(code):
    old='const double pressure=resident_dot(l[1],l[2],nx,ny);'
    if code.count(old)!=1:raise ValueError('Reflecting-wall CUDA assembly layout changed')
    return CUDA_HELPER+'\n'+code.replace(old,'const double pressure=braids_wall_pressure(hL,local[nf+f],grav,dry);')
