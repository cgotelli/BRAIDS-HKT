// M2-C accepted wet/dry phase ownership for the implicit law-11 route.
// One thread owns one fraction/cell.  The proposal arrays are disjoint from
// accepted state; the host commits only after all status and budget checks.

extern "C" __global__ void m2c_wetdry_phase_owner(
    const double* depth,const double* suspended,const double* active_bed,
    const double* area,int cells,int fractions,double dry,
    double* next_suspended,double* next_active_bed,double* phase_change,
    int* status) {
  int i=blockDim.x*blockIdx.x+threadIdx.x;
  int count=cells*fractions;
  if(i>=count)return;
  int c=i%cells;
  double h=depth[c],s=suspended[i],b=active_bed[i],a=area[c];
  int error=0;
  if(!isfinite(h)||!isfinite(s)||!isfinite(b)||!isfinite(a)||
     h<0.0||s<0.0||b<0.0||!(a>0.0))error=1;
  double moved=(!error && h<=dry)?s:0.0;
  double ns=s-moved,nb=b+moved;
  if(!isfinite(ns)||!isfinite(nb)||ns<0.0||nb<0.0)error=2;
  next_suspended[i]=error?s:ns;
  next_active_bed[i]=error?b:nb;
  // Whole-cell signed suspended-phase change in kg; active-bed gain is -this.
  phase_change[i]=error?0.0:-moved*a;
  status[i]=error;
}

