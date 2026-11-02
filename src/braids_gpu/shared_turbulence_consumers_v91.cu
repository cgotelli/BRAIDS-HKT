// Q4/v91 actual-CUDA used-total cap and variable-viscosity momentum consumer.
// Fully wet uniform Cartesian fixed-grid scope only; physical boundary flux=0.
extern "C" __global__ void v91_chkvic_used_total(
 const long long nc,const double dt,const double* width,
 const double* background_vic,const double* background_dic,
 const double* lagged_vic,const double* lagged_dic,
 double* vicmax,double* dicmax,double* capped_vic,double* capped_dic,
 int* capped_vic_flag,int* capped_dic_flag,int* status){
 long long c=blockIdx.x*(long long)blockDim.x+threadIdx.x;if(c>=nc)return;
 status[c]=0;capped_vic_flag[c]=0;capped_dic_flag[c]=0;
 double dx=width[c],dy=width[nc+c];
 double metric=1./(dx*dx)+1./(dy*dy);
 double vmax=.9/(2.*dt*metric),dmax=.7*vmax;
 double v=background_vic[c]+lagged_vic[c];
 double d=background_dic[c]+lagged_dic[c];
 if(!isfinite(dx)||!isfinite(dy)||dx<=0.||dy<=0.||!isfinite(vmax)||
    !isfinite(dmax)||!isfinite(v)||!isfinite(d)||v<0.||d<0.){
   status[c]=1;vicmax[c]=dicmax[c]=capped_vic[c]=capped_dic[c]=0.;return;}
 vicmax[c]=vmax;dicmax[c]=dmax;
 // Preserve equality exactly: locked dynamic chkvic uses strict >.
 if(v>vmax){v=vmax;capped_vic_flag[c]=1;}
 if(d>dmax){d=dmax;capped_dic_flag[c]=1;}
 capped_vic[c]=v;capped_dic[c]=d;
}

extern "C" __global__ void v91_momentum_face_flux(
 const long long nc,const long long nf,const double dry,const double* q,
 const double* vic,const int* owner,const int* neighbour,
 const double* length,const double* distance,double* flux,int* status){
 long long f=blockIdx.x*(long long)blockDim.x+threadIdx.x;if(f>=nf)return;
 status[f]=0;flux[f]=0.;flux[nf+f]=0.;int l=owner[f],r=neighbour[f];
 if(l<0||l>=nc||r>=nc||r< -1){status[f]=1;return;}
 if(r<0)return;
 double hl=q[l],hr=q[r],vl=vic[l],vr=vic[r],dn=distance[f];
 if(!isfinite(hl)||!isfinite(hr)||hl<=dry||hr<=dry||
    !isfinite(vl)||!isfinite(vr)||vl<0.||vr<0.||
    !isfinite(length[f])||length[f]<=0.||!isfinite(dn)||dn<=0.){
   status[f]=1;return;}
 // One antisymmetric momentum flux: h_f * nu_f * grad(u) * face_length.
 double coefficient=.5*(hl+hr)*.5*(vl+vr)*length[f]/dn;
 for(int component=0;component<2;++component){
   double ul=q[(component+1)*nc+l]/hl;
   double ur=q[(component+1)*nc+r]/hr;
   double value=coefficient*(ur-ul);
   if(!isfinite(value)){status[f]=1;value=0.;}
   flux[component*nf+f]=value;
 }
}

extern "C" __global__ void v91_momentum_gather_add(
 const long long nc,const long long nf,const double dry,const double* q,
 const double* area,const int* offset,const int* faces,const int* signs,
 const double* flux,const int* face_status,double* closure_rate,
 double* full_rate,int* status){
 long long c=blockIdx.x*(long long)blockDim.x+threadIdx.x;if(c>=nc)return;
 status[c]=0;closure_rate[c]=0.;closure_rate[nc+c]=0.;
 closure_rate[2*nc+c]=0.;double h=q[c];
 if(!isfinite(h)||h<=dry||!isfinite(area[c])||area[c]<=0.){
   status[c]=1;return;}
 for(int component=0;component<2;++component){double sum=0.;
   for(int slot=offset[c];slot<offset[c+1];++slot){int f=faces[slot];
     if(f<0||f>=nf||face_status[f]){status[c]=1;continue;}
     sum+=signs[slot]*flux[component*nf+f];}
   double value=sum/area[c];
   if(!isfinite(value)){status[c]=1;value=0.;}
   closure_rate[(component+1)*nc+c]=value;
   full_rate[(component+1)*nc+c]+=value;
 }
}
