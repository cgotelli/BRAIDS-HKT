// Private-stage preparation. No water clipping or serial cell sweep.
#define BRAIDS_BANK_WET_PREPARATION_ABI 1
extern "C" __global__ void bw_prepare(double* q,const double* area,const int* levels,
 int nc,int level,double dry,double speed_limit,int dry_only,double* impulse,int* status){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 status[c]=0;for(int k=0;k<4;++k)impulse[4*c+k]=0.;
 double h=q[c],x=q[nc+c],y=q[2*nc+c];
 if(!isfinite(h)||!isfinite(x)||!isfinite(y)||h<0.){status[c]=1;return;}
 double nx=x,ny=y;int category=-1;
 if(h<=dry){nx=0.;ny=0.;category=0;}
 else if(!dry_only){
  double speed=sqrt(x*x+y*y)/h;
  if(!isfinite(speed)){status[c]=1;return;}
  if(speed>speed_limit){nx=x*(speed_limit/speed);ny=y*(speed_limit/speed);category=1;}
 }
 q[nc+c]=nx;q[2*nc+c]=ny;
 if(category>=0&&(level<0||levels[c]==level)){
  impulse[4*c+2*category]=(nx-x)*area[c];
  impulse[4*c+2*category+1]=(ny-y)*area[c];
 }
}
