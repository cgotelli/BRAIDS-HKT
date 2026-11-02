// Isolated fixed-target active layer and ordered-substrate exchange.
#define BRAIDS_STRATIGRAPHY_EXHUMATION_POLICY 3
// Policy 3: symmetric_64eps_exchange_before_repack_v2. Both unresolved burial
// and no-exchange exhumation preserve the ordered substrate. Reuse the
// existing absolute 64-epsilon thickness cutoff; resolved arithmetic is unchanged.
// Layout: layers[(layer * ng + grain) * nc + cell], active layer first.
// The cell-owned scratch has the same layout: inserted surface material first,
// followed by every pre-existing substrate layer from top to base.
__device__ void strat_repack(int c,int nc,int ng,int ns,double thickness_target,
                            const double* bulk,double* layers,double* segments){
 const double tolerance=64.*2.2204460492503131e-16;
 int source=0;
 for(int layer=1;layer<=ns;++layer)
  for(int g=0;g<ng;++g)layers[(layer*ng+g)*nc+c]=0.;
 for(int layer=1;layer<=ns;++layer){
  double capacity=layer<ns?thickness_target:1.e300;
  while(source<=ns&&capacity>0.){
   double thickness=0.;
   for(int g=0;g<ng;++g)thickness+=segments[(source*ng+g)*nc+c]/bulk[g];
   if(thickness<=tolerance){++source;continue;}
   double take=layer<ns?fmin(thickness,capacity):thickness;
   double factor=fmin(1.,take/thickness);
   for(int g=0;g<ng;++g){int si=(source*ng+g)*nc+c,di=(layer*ng+g)*nc+c;
    double amount=segments[si]*factor;
    layers[di]+=amount;segments[si]=segments[si]*(1.-factor);}
   if(layer<ns)capacity=fmax(0.,capacity-take);
   if(factor>=1.-tolerance)++source;
  }
 }
}

extern "C" __global__ void strat_rebalance(int nc,int ng,int ns,double* layers,
 const double* bulk,double active_target,double substrate_thickness,double* segments,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 status[c]=0;double active=0.,substrate=0.;
 // Preserve the native summation order: grain outer, layer inner.
 for(int g=0;g<ng;++g){double mass=layers[g*nc+c];
  if(!isfinite(mass)||mass<0.){status[c]=1;return;}
  active+=mass/bulk[g];
  for(int layer=1;layer<=ns;++layer){mass=layers[(layer*ng+g)*nc+c];
   if(!isfinite(mass)||mass<0.){status[c]=1;return;}
   substrate+=mass/bulk[g];}
 }
 if(!isfinite(active)||!isfinite(substrate)){status[c]=1;return;}
 double target=fmin(active_target,active+substrate);
 if(active>target&&active>0.){
  double move=active-target,factor=fmin(1.,move/active);
  if(move<=64.*2.2204460492503131e-16)return;
  for(int g=0;g<ng;++g){int ai=g*nc+c;double amount=layers[ai]*factor;
   layers[ai]-=amount;segments[g*nc+c]=amount;}
  for(int layer=1;layer<=ns;++layer)
   for(int g=0;g<ng;++g)segments[(layer*ng+g)*nc+c]=layers[(layer*ng+g)*nc+c];
  strat_repack(c,nc,ng,ns,substrate_thickness,bulk,layers,segments);
 }else if(active<target&&substrate>0.){
  double move=target-active;bool exchanged=false;
  for(int layer=1;layer<=ns;++layer){
   if(move<=64.*2.2204460492503131e-16)break;
   double thickness=0.;
   for(int g=0;g<ng;++g)thickness+=layers[(layer*ng+g)*nc+c]/bulk[g];
   if(thickness<=0.)continue;
   double take=fmin(move,thickness),factor=fmin(1.,take/thickness);
   for(int g=0;g<ng;++g){int li=(layer*ng+g)*nc+c,ai=g*nc+c;
    double amount=layers[li]*factor;
    if(amount>0.)exchanged=true;
    layers[li]-=amount;layers[ai]+=amount;}
   move=fmax(0.,move-take);
  }
  if(exchanged){
   for(int g=0;g<ng;++g)segments[g*nc+c]=0.;
   for(int layer=1;layer<=ns;++layer)
    for(int g=0;g<ng;++g)segments[(layer*ng+g)*nc+c]=layers[(layer*ng+g)*nc+c];
   strat_repack(c,nc,ng,ns,substrate_thickness,bulk,layers,segments);
  }
 }
 for(int layer=0;layer<=ns;++layer)
  for(int g=0;g<ng;++g){double mass=layers[(layer*ng+g)*nc+c];
   if(!isfinite(mass)||mass<0.)status[c]=1;}
}

extern "C" __global__ void strat_bed(int nc,int ng,int ns,const double* layers,
 const double* rock,const double* bulk,double* bed,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 status[c]=0;double z=rock[c];if(!isfinite(z))status[c]=1;
 for(int g=0;g<ng;++g){double mass=layers[g*nc+c];
  if(!isfinite(mass)||mass<0.)status[c]=1;
  z+=mass/bulk[g];
  for(int layer=1;layer<=ns;++layer){mass=layers[(layer*ng+g)*nc+c];
   if(!isfinite(mass)||mass<0.)status[c]=1;
   z+=mass/bulk[g];}
 }
 bed[c]=z;if(!isfinite(z))status[c]=1;
}
