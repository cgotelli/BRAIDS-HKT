// Bounded repair-free native AMR reflux: two accepted fine substep integrals
// replace the coarse integral on both traces of each 2:1 interface face.
// One thread per coarse state row applies faces in native ascending order.
// The whole proposal remains separate from accepted state and flux register.

extern "C" __global__ void amr_fine_register(
    const double* fine, int faces, int states, double* trial_register) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int length=faces*2*states;
    if(i>=length)return;
    trial_register[i]=fine[i]+fine[length+i];
}

extern "C" __global__ void amr_reflux_coarse(
    const double* coarse_flux,const double* fine_register,
    const int* coarse_cell,const int* offset,const int* face,const int* side,
    const double* area,int coarse_count,int states,double* trial) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=coarse_count*states)return;
    int p=i/states,j=i%states,c=coarse_cell[p];
    double updated=trial[c*states+j];
    for(int k=offset[p];k<offset[p+1];++k){
        int f=face[k],s=side[k],row=(f*2+s)*states+j;
        double delta=coarse_flux[row]-fine_register[row];
        updated+=(s==0?delta:-delta)/area[c];
    }
    trial[c*states+j]=updated;
}

extern "C" __global__ void amr_validate_reflux(
    const double* state,const double* register_flux,const double* coarse_flux,
    const double* fine_flux,int cells,int faces,
    int states,int grains,int physical_states,int prognostic_secondary,
    int* status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=cells && c>=faces)return;
    int error=0;
    if(c<cells)
        for(int j=0;j<states;++j){
            double value=state[c*states+j];
            if(!isfinite(value)){error=1;break;}
            if((j==0 || (j>=3 && j<physical_states)) && value<0.0){error=2;break;}
            if(j>=3 && j<3+grains && state[c*states]==0.0 && value!=0.0){error=3;break;}
            if(prognostic_secondary && j==physical_states &&
               state[c*states]==0.0 && value!=0.0){error=5;break;}
        }
    if(c<faces){
        int length=faces*2*states;
        for(int j=0;j<2*states;++j){
            int row=c*2*states+j;
            if(!isfinite(register_flux[row]) || !isfinite(coarse_flux[row]) ||
               !isfinite(fine_flux[row]) || !isfinite(fine_flux[length+row]))error=4;
        }
    }
    status[c]=error;
}
