// Native mode-3 persistent front preflight; never debits inventory.
// Supplied face mass is already donor-limited; (grain,face) signed total kg.
#define PF_EPS 2.2204460492503131e-16

__device__ __forceinline__ double pf_roundoff(double value) {
    return 256.*PF_EPS*fmax(fabs(value),2.2250738585072014e-308);
}

extern "C" __global__ void pf_effective(int nc,int nf,int ng,int nl,
 const double* area,const double* rock,const double* bulk,const double* layers,
 const int* front_status,const int* seed,const double* solid,
 const double* height,const double* base,const double* removed,
 double* raw,double* effective,int* cell_status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    double bed=rock[c],removed_volume=0.;int bad=0;
    // Native raw_bed_elevation loops grain outer, then active and substrates.
    for(int g=0;g<ng;++g){
        for(int l=0;l<nl;++l){double mass=layers[(l*ng+g)*nc+c];
            if(!isfinite(mass)||mass<0.)bad=2;
            bed+=mass/bulk[g];}
        double historical=removed[g*nc+c];if(!isfinite(historical)||historical<0.)bad=2;
        removed_volume+=historical/bulk[g];
    }
    double a=area[c],s=solid[c],h=height[c],b=base[c];int st=front_status[c],id=seed[c];
    if(!isfinite(bed)||!isfinite(s)||!isfinite(h)||!isfinite(b)||
       s<0.||s>a||h<0.||st<0||st>2||!isfinite(removed_volume))bad=2;
    if(st==0){if(id!=-1||h>0.||fabs(s-a)>pf_roundoff(a)||removed_volume>0.)bad=2;}
    else if(st==1){
        if(id<0||id>=nf||h<=0.||s<=0.)bad=2;
        double initial=a*h,remaining=s*h;
        if(fabs(initial-removed_volume-remaining)>
           pf_roundoff(fmax(initial,fmax(removed_volume,remaining))))bad=2;
    }else if(st==2){
        if(id<0||id>=nf||h<=0.||s>pf_roundoff(a))bad=2;
        double initial=a*h;
        if(fabs(initial-removed_volume)>pf_roundoff(fmax(initial,removed_volume)))bad=2;
    }
    raw[c]=bed;effective[c]=bed;
    if(st==1&&bad==0){
        if(bed<b-pf_roundoff(fmax(fabs(bed),fabs(b))))bad=2;
        else effective[c]=b+a*fmax(bed-b,0.)/s;
    }
    if(!isfinite(effective[c]))bad=2;
    cell_status[c]=bad;
}

extern "C" __global__ void pf_protected(int nc,int nf,int ng,
 const int* offset,const int* faces,const int* eligible,const int* donor,
 const double* mass,const int* front_status,int* protected_cell) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    int protected_=front_status[c]==1;
    for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];
        if(!eligible[f]||donor[f]!=c)continue;
        for(int g=0;g<ng;++g)if(fabs(mass[g*nf+f])>0.)protected_=1;
    }
    protected_cell[c]=protected_;
}

extern "C" __global__ void pf_exclude(int nc,int nf,int ng,
 const int* owner,const int* neighbour,const int* protected_cell,
 const int* donor,const int* recipient,double* mass,int* eligible,
 int* suppressed,int* face_status) {
    int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
    suppressed[f]=0;face_status[f]=0;
    bool nonzero=false;for(int g=0;g<ng;++g){double x=mass[g*nf+f];
        if(!isfinite(x)){face_status[f]=2;return;}if(fabs(x)>0.)nonzero=true;}
    if(!eligible[f]||!nonzero)return;
    int d=donor[f],r=recipient[f];
    if(d<0||d>=nc||r<0||r>=nc||d==r||
       !((owner[f]==d&&neighbour[f]==r)||
         (owner[f]==r&&neighbour[f]==d))){face_status[f]=2;return;}
    int sign=owner[f]==d?1:-1;
    for(int g=0;g<ng;++g)if(sign*mass[g*nf+f]<0.){
        face_status[f]=2;return;}
    if(!protected_cell[r])return;
    for(int g=0;g<ng;++g)mass[g*nf+f]=0.;
    eligible[f]=0;suppressed[f]=1;
}

extern "C" __global__ void pf_request(int nc,int nf,int ng,
 const int* offset,const int* faces,const double* bulk,
 const int* eligible,const int* donor,const double* mass,
 double* request,int* seed) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    double total=0.,largest=-1.;int source=-1;
    for(int k=offset[c];k<offset[c+1];++k){int f=faces[k];
        if(!eligible[f]||donor[f]!=c)continue;
        double face=0.;for(int g=0;g<ng;++g)face+=fabs(mass[g*nf+f])/bulk[g];
        if(face<=0.)continue;total+=face;
        if(face>largest){largest=face;source=f;}
    }
    request[c]=total;seed[c]=source;
}

extern "C" __global__ void pf_preflight(int nc,const double* area,const double* rock,
 const double* bed,const int* recipient,double* request,const int* seed,
 const int* front_status,const double* solid,const double* front_height,
 const double* front_base,double* candidate_base,double* candidate_height,
 double* scale,double* retry,int* zero_donor,int* cell_status) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    double volume=request[c];scale[c]=1.;retry[c]=1.;zero_donor[c]=0;
    candidate_base[c]=front_base[c];candidate_height[c]=front_height[c];
    if(cell_status[c]||volume<=0.)return;
    int st=front_status[c];double capacity=0.;
    if(st==2){cell_status[c]=2;return;}
    if(st==0){int f=seed[c];if(f<0){cell_status[c]=2;return;}
        int r=recipient[f];if(r<0||r>=nc){cell_status[c]=2;return;}
        double base=fmax(bed[r],rock[c]),height=bed[c]-base;
        candidate_base[c]=base;candidate_height[c]=height;
        double tol=512.*PF_EPS*fmax(fmax(fabs(bed[c]),fabs(base)),1.);
        if(height<=tol){request[c]=0.;zero_donor[c]=1;return;}
        capacity=area[c]*height;
    }else if(st==1){capacity=solid[c]*front_height[c];
        if(!isfinite(capacity)||capacity<=0.){cell_status[c]=2;return;}
    }else{cell_status[c]=2;return;}
    double tol=1024.*PF_EPS*fmax(fmax(capacity,volume),1.e-30);
    if(volume>capacity+tol){
        retry[c]=fmax(1.e-8,capacity/volume);cell_status[c]=1;
    }else if(volume>capacity){scale[c]=capacity/volume;}
}

extern "C" __global__ void pf_merge(int nc,int nf,const int* cell_status,
 const int* face_status,const double* retry,int* global_status,double* retry_scale) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<nc){if(cell_status[i])atomicMax(global_status,cell_status[i]);
        atomicMin((unsigned long long*)retry_scale,
                  (unsigned long long)__double_as_longlong(retry[i]));}
    if(i<nf&&face_status[i])atomicMax(global_status,face_status[i]);
}

extern "C" __global__ void pf_finish_faces(int nc,int nf,int ng,
 const int* donor,const int* zero_donor,const double* scale,
 const int* global_status,double* mass) {
    int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf||*global_status==2)return;
    int c=donor[f];if(c<0||c>=nc)return;
    if(zero_donor[c])for(int g=0;g<ng;++g)mass[g*nf+f]=0.;
    else if(*global_status==0&&scale[c]!=1.)
        for(int g=0;g<ng;++g)mass[g*nf+f]*=scale[c];
}

extern "C" __global__ void pf_activate(int nc,int ng,const double* area,
 const double* request,const int* seed_candidate,const double* candidate_base,
 const double* candidate_height,const int* global_status,int* front_status,
 int* seed,double* solid,double* height,double* base,double* removed) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc||*global_status)return;
    if(request[c]<=0.||front_status[c]!=0)return;
    front_status[c]=1;seed[c]=seed_candidate[c];solid[c]=area[c];
    height[c]=candidate_height[c];base[c]=candidate_base[c];
    for(int g=0;g<ng;++g)removed[g*nc+c]=0.;
}
