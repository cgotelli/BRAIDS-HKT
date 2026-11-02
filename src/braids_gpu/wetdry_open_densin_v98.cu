// Sole physical-open face equation, before the unchanged one active projection.
// Physical flux ownership is independent of whether exterior DensIn is active.
extern "C" __global__ void wetdry_physical_densin_v98(
 int nf,double gravity,double reference_density,double dry,int ibaroc,
 const double* depth,const double* selected_density,const int* wet,
 const double* trace,const double* parent_boundary_density,const int* owner,
 const double* distance,const int* geometry_open,const int* physical_owned,
 const int* requested_density,const double* base_acceleration,const int* base_mask,
 double* acceleration,int* mask,int* exterior_active,int* status){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
 bool bad=false;status[f]=0;exterior_active[f]=0;
 acceleration[f]=base_acceleration[f];mask[f]=base_mask[f];
 if(!isfinite(base_acceleration[f])||(base_mask[f]!=0&&base_mask[f]!=1))bad=true;
 if((physical_owned[f]!=0&&physical_owned[f]!=1)||
    (requested_density[f]!=0&&requested_density[f]!=1)||
    physical_owned[f]!=geometry_open[f]||requested_density[f]>physical_owned[f])bad=true;
 if(!physical_owned[f]){status[f]=bad?1:0;return;}
 int c=owner[f];
 if((wet[c]!=0&&wet[c]!=1)||!isfinite(depth[c])||depth[c]<0.||
    !isfinite(trace[f])||trace[f]<0.||base_mask[f]!=0)bad=true;
 bool active=requested_density[f]&&wet[c]&&depth[c]>dry&&trace[f]>dry;
 acceleration[f]=0.;mask[f]=active?1:0;exterior_active[f]=active?1:0;
 if(active){
  double rho=selected_density[c],other=parent_boundary_density[f];
  if(!isfinite(rho)||!isfinite(other)||rho<=0.||other<=0.)bad=true;
  if(ibaroc)acceleration[f]=-gravity*(other-rho)*(trace[f]+depth[c])/
    (4.*reference_density*distance[f]);
 }
 if(!isfinite(acceleration[f]))bad=true;
 status[f]=bad?1:0;
}
