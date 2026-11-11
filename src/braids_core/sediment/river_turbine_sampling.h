// GPU array sampling. One deterministic block per rotor, never atomic sums.
#define INFINITY (1./0.)
extern "C" __global__ void rotor_stats(int n,int nr,const double*q,const double*z,const double*g,
 const int*off,const int*ids,const double*w,const double*normal,const double*dry,
 const int*poff,const int*plane,const double*delta,double*uv,double*out){
 int r=blockIdx.x,t=threadIdx.x;if(r>=nr)return;
 __shared__ double s[6][128];double v[6]={INFINITY,-INFINITY,0.,0.,0.,0.};int ns=off[nr],np=poff[nr];
 for(int j=off[r]+t;j<off[r+1];j+=128){int i=ids[j];double h=q[i],u=h>dry[r]?q[n+i]/fmax(h,dry[r]):0.,a=h>dry[r]?q[2*n+i]/fmax(h,dry[r]):0.;
  uv[j]=u;uv[ns+j]=a;uv[2*ns+j]=hypot(u,a);v[2]=fmax(v[2],h<=dry[r]?1.:0.);
  v[3]+=w[j]*(u*normal[2*r]+a*normal[2*r+1]);v[4]+=w[j]*h;v[5]+=w[j]*z[i];}
 for(int j=poff[r]+t;j<poff[r+1];j+=128){int i=plane[j];double eta=q[i]+z[i];
  double surface=eta+g[4*n+i]*delta[j]+g[5*n+i]*delta[np+j],bed=z[i]+g[6*n+i]*delta[j]+g[7*n+i]*delta[np+j];
  v[0]=fmin(v[0],fmin(surface,eta));v[1]=fmax(v[1],fmax(bed,z[i]));}
 for(int k=0;k<6;++k)s[k][t]=v[k];__syncthreads();
 for(int d=64;d;d>>=1){if(t<d){s[0][t]=fmin(s[0][t],s[0][t+d]);s[1][t]=fmax(s[1][t],s[1][t+d]);s[2][t]=fmax(s[2][t],s[2][t+d]);for(int k=3;k<6;++k)s[k][t]+=s[k][t+d];}__syncthreads();}
 if(t==0)for(int k=0;k<6;++k)out[6*r+k]=s[k][0];
}

extern "C" __global__ void rotor_force(int nr,const int*off,const int*ids,const double*w,
 const double*area,const double*normal,const double*scale,const double*uv,double*k,double*out){
 int r=blockIdx.x,t=threadIdx.x;if(r>=nr)return;int ns=off[nr];__shared__ double f[128],p[128];double F=0.,P=0.;
 for(int j=off[r]+t;j<off[r+1];j+=128){double c=scale[r]*w[j]/area[ids[j]];k[j]=c;
  if(scale[r]!=0.){double u=uv[j],v=uv[ns+j],speed=uv[2*ns+j];F+=c*area[ids[j]]*speed*(u*normal[2*r]+v*normal[2*r+1]);P+=c*area[ids[j]]*speed*(u*u+v*v);}}
 f[t]=F;p[t]=P;__syncthreads();for(int d=64;d;d>>=1){if(t<d){f[t]+=f[t+d];p[t]+=p[t+d];}__syncthreads();}
 if(t==0){out[2*r]=f[0];out[2*r+1]=p[0];}
}

extern "C" __global__ void rotor_apply(int nc,int ns,int n,const int*cells,const int*off,
 const int*support,const double*k,const double*uv,double*rate){
 for(int j=blockIdx.x*blockDim.x+threadIdx.x;j<nc;j+=blockDim.x*gridDim.x){int i=cells[j];double x=rate[n+i],y=rate[2*n+i];
  for(int a=off[j];a<off[j+1];++a){int s=support[a];x-=k[s]*uv[2*ns+s]*uv[s];y-=k[s]*uv[2*ns+s]*uv[ns+s];}rate[n+i]=x;rate[2*n+i]=y;}
}

extern "C" __global__ void rotor_scale(int nr,const int*off,const double*factor,double*k){
 int r=blockIdx.x;if(r>=nr)return;for(int j=off[r]+threadIdx.x;j<off[r+1];j+=128)k[j]*=factor[r];
}

extern "C" __global__ void rotor_moments(int n,int nr,int count,int nq,const int*selected,
 const double*q,const int*nodes,const double*bary,const double*w,const double*normal,
 const double*dry,const double*local,const int*poff,const double*amp,const double*mean,
 const double*ey,const double*ez,double*offsets,double*out){
 int s=blockIdx.x,t=threadIdx.x;if(s>=count)return;int r=selected[s];__shared__ double sum[8][128];
 double a[8]={0.,0.,INFINITY,0.,0.,0.,0.,0.};
 for(int j=t;j<nq;j+=128){int k=r*nq+j;double u=0.;
  for(int c=0;c<3;++c){int i=nodes[3*k+c];double h=q[i];if(h<=dry[r])a[5]=1.;double un=(q[n+i]*normal[2*r]+q[2*n+i]*normal[2*r+1])/fmax(h,dry[r]);u+=bary[3*k+c]*un;}
  double raw=u-local[6*r+3],anomaly=0.;for(int p=poff[r];p<poff[r+1];++p)anomaly+=amp[p]*ey[p*nq+j]*(mean[p]-ez[p*nq+j]);
  double v=raw+anomaly;offsets[k]=v;a[0]+=w[k]*v;a[1]+=w[k];a[2]=fmin(a[2],v);a[3]=fmax(a[3],!isfinite(v)?1.:0.);a[4]=fmax(a[4],w[k]<0.?1.:0.);a[6]+=w[k]*raw;}
 for(int k=0;k<8;++k)sum[k][t]=a[k];__syncthreads();
 for(int d=64;d;d>>=1){if(t<d){sum[0][t]+=sum[0][t+d];sum[1][t]+=sum[1][t+d];sum[2][t]=fmin(sum[2][t],sum[2][t+d]);for(int k=3;k<=5;++k)sum[k][t]=fmax(sum[k][t],sum[k][t+d]);sum[6][t]+=sum[6][t+d];}__syncthreads();}
 double mu=sum[0][0],v2=0.,v3=0.;__syncthreads();
 for(int j=t;j<nq;j+=128){int k=r*nq+j;double c=offsets[k]-mu;v2+=w[k]*c*c;v3+=w[k]*c*c*c;}
 sum[0][t]=v2;sum[7][t]=v3;__syncthreads();for(int d=64;d;d>>=1){if(t<d){sum[0][t]+=sum[0][t+d];sum[7][t]+=sum[7][t+d];}__syncthreads();}
 if(t==0){double*v=out+9*r;v[0]=mu;v[1]=sum[0][0];v[2]=sum[7][0];v[3]=sum[2][0];v[4]=sum[3][0];v[5]=sum[4][0];v[6]=sum[1][0];v[7]=sum[5][0];v[8]=sum[6][0];}
}
