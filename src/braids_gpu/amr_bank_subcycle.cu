// Generic all-face accepted register and deterministic coarse-cell reflux.
// No bank source is evaluated here. All arrays are private macro-step work.
#define BRAIDS_BANK_SUBCYCLE_REGISTER_ABI 1
extern "C" __global__ void bs_reflux(
    double* state, const double* coarse, const double* fine,
    const int* cell, const int* offset, const int* faces, const int* side,
    const double* area, int groups, int ns) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=groups*ns)return;
  int p=i/ns,j=i%ns,c=cell[p]; double value=state[c*ns+j];
  for(int k=offset[p];k<offset[p+1];++k){
    int f=faces[k],s=side[k]; long long row=((long long)f*2+s)*ns+j;
    double delta=coarse[row]-fine[row];
    value+=(s==0?delta:-delta)/area[c];
  }
  state[c*ns+j]=value;
}
