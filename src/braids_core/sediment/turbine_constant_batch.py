"""Small fixed-axis constant/incident-curve rotors: one launch and host transfer.

Only immutable geometry is packed. Every evaluation reads the current fields;
operating decisions and error messages remain owned by the original rotor.
"""
import math
import numpy as np
_KERNEL=None

def eligible(r):
    from .river_turbine_yaw_legacy import FixedAxisRotor
    return type(r) is FixedAxisRotor and r.xp is not np and r.control['mode'] in ('constant_ct','incident_speed_table','incident_speed_curves') and len(r.ids)<=256 and len(r.plane)<=512

class ConstantRotorBatch:
    def __init__(self,rotors):
        self.rotors=tuple(rotors);self.xp=rotors[0].xp;xp=self.xp
        self.so=np.cumsum([0]+[len(r.ids) for r in rotors]);self.po=np.cumsum([0]+[len(r.plane) for r in rotors])
        self.ids=xp.concatenate([r.ids for r in rotors]).astype(xp.int64)
        self.plane=xp.concatenate([r.plane for r in rotors]).astype(xp.int64)
        self.delta=xp.ascontiguousarray(xp.concatenate([r.delta for r in rotors],axis=1))
        self.geometry=xp.ascontiguousarray(xp.stack([xp.concatenate([getattr(r,name)[r.ids] for r in rotors]) for name in ('normalized','density_integrals','area')]))
        self.offset=xp.asarray(np.stack([self.so,self.po]).astype(np.int32))
        self.parameters=xp.empty((len(rotors),6));self._parameter_key=None;self._curve_key=None

    def evaluate(self,q,z,grad):
        global _KERNEL
        xp=self.xp;nr=len(self.rotors);ns=int(self.so[-1]);np_=int(self.po[-1])
        from .river_turbine_thetis import curve_rows
        params=[];metadata=[];tx=[];tv=[];px=[];pv=[]
        for r in self.rotors:
            c=r.control;constant=c['mode']=='constant_ct'
            ct=c['thrust_coefficient'] if constant else 0.
            cp=(c['power_coefficient'] or .5*ct*(1+math.sqrt(1-ct))) if constant else 0.
            params.append((r.dry,*r.normal,r.c['D'],ct,cp))
            tr=[] if constant else curve_rows(c,'thrust_coefficient')
            pr=[] if constant else curve_rows(c,'power_coefficient')
            metadata.append((0 if constant else (1 if c['mode']=='incident_speed_table' else 2),len(tx),len(tr),len(px),len(pr)))
            tx.extend(v['speed_m_s'] for v in tr);tv.extend(v['thrust_coefficient'] for v in tr)
            px.extend(v['speed_m_s'] for v in pr);pv.extend(v['power_coefficient'] for v in pr)
        params=tuple(params)
        if params!=self._parameter_key:self.parameters.set(np.asarray(params,dtype=np.float64));self._parameter_key=params
        key=(tuple(metadata),tuple(tx),tuple(tv),tuple(px),tuple(pv))
        if key!=self._curve_key:
            self.curves=(xp.asarray(metadata,dtype=xp.int32),*[xp.asarray(v or [0.],dtype=xp.float64) for v in (tx,tv,px,pv)])
            self._curve_key=key
        if _KERNEL is None:
            _KERNEL=xp.RawKernel(r'''
            __device__ double mn(double a,double b){return isnan(a)?a:(isnan(b)?b:fmin(a,b));}
            __device__ double mx(double a,double b){return isnan(a)?a:(isnan(b)?b:fmax(a,b));}
            extern "C" __global__ void batch(int n,int nr,int ns,int np,const int* off,const long long* ids,const long long* plane,const double* delta,const double* geom,const double* par,const int* curves,const double* tx,const double* tv,const double* px,const double* pv,const double* q,const double* z,const double* grad,double* out,double* stats){
              int r=blockIdx.x,tid=threadIdx.x;if(r>=nr)return;
              __shared__ double part[15][256];
              double dry=par[6*r],nx=par[6*r+1],ny=par[6*r+2],D=par[6*r+3],ct=par[6*r+4];
              double A=3.14159265358979323846*(D*D)/4.;double lo=1./0.,hi=-1./0.,is_dry=0.,su=0.,sh=0.,st=0.,umin=1./0.;
              double force=0.,work=0.,power=0.,incoming=0.,ctmean=0.,cpmean=0.,invalid=0.,upper=0.,lower=0.,shallow=0.;
              int mode=curves[5*r],t0=curves[5*r+1],nt=curves[5*r+2],p0=curves[5*r+3],npower=curves[5*r+4];
              for(int j=off[r]+tid;j<off[r+1];j+=blockDim.x){
                int c=ids[j];double h=q[c],den=mx(h,dry),u=q[n+c]/den,v=q[2*n+c]/den;
                double un=u*nx+v*ny,ut=-u*ny+v*nx,w=geom[j],density=geom[ns+j];
                out[j]=h;out[ns+j]=w;out[2*ns+j]=un;out[3*ns+j]=ut;out[5*ns+j]=un*nx;out[6*ns+j]=un*ny;
                int k=j-off[r];part[0][k]=h<=dry?1.:0.;part[1][k]=w*un;part[2][k]=w*h;part[3][k]=w*ut;part[4][k]=un;
                invalid=upper=lower=shallow=0.;
                double K,U,Ct=ct,Cp=par[6*r+5],P;
                if(!mode){
                  double rad=1.-A*ct/(D*h);invalid=fmax(invalid,rad<0.?1.:0.);
                  double alpha=.5*(1.+sqrt(rad));U=hypot(un,0.)/alpha;K=A*ct/(2.*(alpha*alpha));P=(.5*A*Cp)*pow(U,3.);
                }else{
                  double c=3.14159265358979323846*D/(4*h),speed=hypot(un,0.);
                  double l=tx[t0],hi=tx[t0+nt-1];
                  double smin=l*.5*(1+sqrt(1-c*tv[t0])),smax=hi*.5*(1+sqrt(1-c*tv[t0+nt-1]));
                  // Guard arithmetic follows the original array-expression order.
                  double guardlo=l*.5*(1+sqrt(1-((3.14159265358979323846*D)*tv[t0])/(4*h)));
                  double guardhi=hi*.5*(1+sqrt(1-((3.14159265358979323846*D)*tv[t0+nt-1])/(4*h)));
                  lower=fmax(lower,un<guardlo?1.:0.);upper=fmax(upper,un>=guardhi?1.:0.);
                  shallow=fmax(shallow,h<D*(1-1e-10)?1.:0.);
                  bool active=speed>=smin && speed<smax;
                  for(int it=0;it<44;it++){
                    double mid=.5*(l+hi);int a=t0,b=t0+nt-1;
                    while(b-a>1){int m=(a+b)/2;if(tx[m]<=mid)a=m;else b=m;}
                    double v=tv[a]+(tv[b]-tv[a])*(mid-tx[a])/(tx[b]-tx[a]);
                    double local=mid*.5*(1+sqrt(1-c*v));if(local<speed)l=mid;else hi=mid;
                  }
                  U=active?.5*(l+hi):speed;Ct=0.;Cp=0.;
                  if(active){
                    int a=t0,b=t0+nt-1;while(b-a>1){int m=(a+b)/2;if(tx[m]<=U)a=m;else b=m;}
                    double t=(U-tx[a])/(tx[b]-tx[a]);Ct=tv[a]+(tv[b]-tv[a])*t;
                    if(mode==1){int pa=p0+(a-t0),pb=p0+(b-t0);Cp=pv[pa]+(pv[pb]-pv[pa])*t;}
                    else {int pa=p0,pb=p0+npower-1;while(pb-pa>1){int m=(pa+pb)/2;if(px[m]<=U)pa=m;else pb=m;}Cp=pv[pa]+(pv[pb]-pv[pa])*(U-px[pa])/(px[pb]-px[pa]);}
                  }
                  double alpha=.5*(1+sqrt(1-c*Ct)),area=(3.14159265358979323846*D)*D/4;
                  K=(area*Ct)/((2*alpha)*alpha);P=(((.5*area)*Cp)*U)*U*U;
                }
                out[4*ns+j]=K*density/geom[2*ns+j];
                part[5][k]=(K*density)*(un*un);part[6][k]=(K*density)*pow(fabs(un),3.);part[7][k]=P*density;
                part[8][k]=w*U;part[9][k]=w*Ct;part[10][k]=w*Cp;part[11][k]=invalid;part[12][k]=upper;part[13][k]=lower;part[14][k]=shallow;
              }
              __syncthreads();if(tid)return;
              invalid=upper=lower=shallow=0.;
              for(int k=0;k<off[r+1]-off[r];k++){
                is_dry=fmax(is_dry,part[0][k]);su+=part[1][k];sh+=part[2][k];st+=part[3][k];umin=mn(umin,part[4][k]);
                force+=part[5][k];work+=part[6][k];power+=part[7][k];incoming+=part[8][k];ctmean+=part[9][k];cpmean+=part[10][k];
                invalid=fmax(invalid,part[11][k]);upper=fmax(upper,part[12][k]);lower=fmax(lower,part[13][k]);shallow=fmax(shallow,part[14][k]);
              }
              for(int j=off[nr+1+r];j<off[nr+1+r+1];j++){
                int c=plane[j];double eta=q[c]+z[c];
                double surf=(eta+grad[4*n+c]*delta[j])+grad[5*n+c]*delta[np+j];
                double bed=(z[c]+grad[6*n+c]*delta[j])+grad[7*n+c]*delta[np+j];
                lo=mn(lo,mn(surf,eta));hi=mx(hi,mx(bed,z[c]));
              }
              double* s=stats+17*r;s[0]=lo;s[1]=hi;s[2]=is_dry;s[3]=su;s[4]=sh;s[5]=st;s[6]=umin;
              s[7]=force;s[8]=work;s[9]=power;s[10]=incoming;s[11]=ctmean;s[12]=cpmean;s[13]=invalid;s[14]=upper;s[15]=lower;s[16]=shallow;
            }''','batch',options=('--fmad=false',))
        # Detached outputs: later samples must not overwrite previously returned forces.
        values=xp.empty((7,ns));stats=xp.empty((nr,17))
        _KERNEL((nr,),(128,),(np.int32(q.shape[1]),np.int32(nr),np.int32(ns),np.int32(np_),self.offset,self.ids,self.plane,self.delta,self.geometry,self.parameters,*self.curves,q,z,grad,values,stats))
        host=stats.get();result=[]
        for i,r in enumerate(self.rotors):
            a,b=self.so[i:i+2];result.append(r._sample_packed(q,z,grad,(*values[:4,a:b],host[i,:7]),(values[4,a:b],host[i,7:13],host[i,13],values[5,a:b],values[6,a:b],host[i,14:17])))
        return result
