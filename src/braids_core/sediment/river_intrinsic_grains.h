// Noncohesive constant-per-class branch of FLOW compdiam/compsandfrac.
// Caller resolves mixed/layered composition weights. No within-class spread
// enters DgSD in this branch. The <2mm sand test has no lower size cutoff.
ENTRY void intrinsic_grains(int n,int ng,const double*weight,const double*diam,
 double*out,int*status){
 LOOP(n){
  double total=0.,logmean=0.,sand=0.;bool valid=true;
  out[i]=out[n+i]=out[2*n+i]=0.;
  for(int k=0;k<ng;++k){
   double w=weight[k*n+i],d=diam[k];
   if(!isfinite(w)||w<0.||!isfinite(d)||d<=0.){valid=false;break;}
   total+=w;logmean+=w*log(d);if(d<.002)sand+=w;
  }
  status[i]=valid&&isfinite(total)&&isfinite(logmean)?0:1;
  if(status[i]||total==0.)continue;
  logmean/=total;
  double variance=0.;
  for(int k=0;k<ng;++k){double v=log(diam[k])-logmean;variance+=weight[k*n+i]/total*v*v;}
  out[i]=exp(logmean);out[n+i]=exp(sqrt(variance));out[2*n+i]=sand/total;
  if(!isfinite(out[i])||!isfinite(out[n+i])||!isfinite(out[2*n+i]))status[i]=1;
 }
}
