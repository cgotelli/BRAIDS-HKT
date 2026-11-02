// Shared Q6b bed/composition boundary and point-load equations.
#ifndef BRAIDS_HD
#define BRAIDS_HD
#endif

BRAIDS_HD inline void braids_q6b_bed_cell_v103(int c,int nc,int mode,int inward,
    double dt,const double* computed,const double* interior,const double* depth,
    const double* prescribed,double* result,int* status) {
    int selected=inward?mode:0;
    if(selected==6 || selected==7) {result[c]=computed[c];status[c]=selected;return;}
    if(selected<0 || selected>8) {result[c]=computed[c];status[c]=1;return;}
    if(selected==1) result[c]=0.;
    else if(selected==2) result[c]=depth[c]-prescribed[c];
    else if(selected==3) result[c]=-prescribed[c]*dt;
    else result[c]=interior[c]; // 0,4,5,8: FLOW free boundary bed update
    status[c]=isfinite(result[c])?0:1;
}

BRAIDS_HD inline void braids_q6b_composition_cell_v103(int c,int nc,int ng,int nl,
    int mode,int coarse,const double* rho,const double* interior,const double* values,
    double* boundary,int* status) {
    if(mode<0 || mode>3) {status[c]=1;return;}
    if(mode==0 || (mode==1 && coarse)) {
        for(int k=0;k<nl;k++) for(int g=0;g<ng;g++)
            boundary[(k*ng+g)*nc+c]=interior[(k*ng+g)*nc+c];
        status[c]=0;return;
    }
    if(mode==1) {status[c]=0;return;}
    double total=0.,den=0.;
    for(int g=0;g<ng;g++) {
        double v=values[g*nc+c];
        if(!(v>=0.) || !(rho[g]>0.)) {status[c]=1;return;}
        total+=boundary[g*nc+c]; den+=mode==2?v:v*rho[g];
    }
    if(!(den>0.) || !(total>=0.)) {status[c]=1;return;}
    for(int g=0;g<ng;g++) {
        double v=values[g*nc+c];
        boundary[g*nc+c]=total*(mode==2?v:v*rho[g])/den;
    }
    status[c]=0;
}

BRAIDS_HD inline void braids_q6b_load_cell_v103(int c,int nc,int ng,double dt,
    const double* discharge,const double* concentration,const double* initial_volume,
    const double* current_volume,const double* area,const int* wet,double* mass,
    double* delta,int* status) {
    double q=discharge[c];
    if(!isfinite(q) || !(dt>=0.) || !(area[c]>0.) || !(initial_volume[c]>=0.) ||
       !(current_volume[c]>=0.)) {status[c]=1;return;}
    if(q>0. && !(initial_volume[c]>0.)) {status[c]=2;return;}
    for(int g=0;g<ng;g++) {
        int i=g*nc+c; double old=mass[i],change=0.;
        if(!(old>=0.) || !(concentration[i]>=0.) || !isfinite(concentration[i])) {
            status[c]=1;return;
        }
        if(q>0.) change=q*concentration[i]*dt/area[c];
        else if(q<0. && wet[c] && current_volume[c]>0.)
            change=-fmin(old,(-q)*old*area[c]/current_volume[c]*dt/area[c]);
        mass[i]=old+change;delta[i]=change;
    }
    status[c]=0;
}
