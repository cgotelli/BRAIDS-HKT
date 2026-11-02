extern "C" __global__ void braids_q8a_slopes_v105(int nx,int ny,double dx,double dy,
    const double* bed,double* sx,double* sy,double* magnitude,int* status) {
    const int c=blockDim.x*blockIdx.x+threadIdx.x,nc=nx*ny;if(c>=nc)return;
    const int i=c%nx,j=c/nx;status[c]=0;
    if(!isfinite(bed[c])){status[c]=1;sx[c]=sy[c]=magnitude[c]=0.;return;}
    sx[c]=i+1<nx?q8a_depth_slope_v105(bed[c],bed[c+1],dx):0.;
    sy[c]=j+1<ny?q8a_depth_slope_v105(bed[c],bed[c+nx],dy):0.;
    magnitude[c]=hypot(sx[c],sy[c]);if(!isfinite(magnitude[c]))status[c]=1;
}

extern "C" __global__ void braids_q8a_shear_map_v105(int nc,int nm,const double* shear,
    const int* reference,const int* application,const double* factor,double* mapped,int* status) {
    const int c=blockDim.x*blockIdx.x+threadIdx.x;if(c>=nc)return;
    mapped[c]=shear[c];status[c]=isfinite(shear[c])?0:1;
    for(int k=0;k<nm;k++)if(application[k]==c){const int r=reference[k];
        if(r<0||r>=nc||!isfinite(factor[k])){status[c]=2;continue;}
        mapped[c]=shear[r]*factor[k];}
    if(!isfinite(mapped[c]))status[c]=2;
}

extern "C" __global__ void braids_q8a_classic_flux_v105(int nx,int ny,double dx,double dy,
    double wet_slope,double avalanche_time,const double* area,const double* bed,
    const double* sx,const double* sy,const double* magnitude,double* qx,double* qy,int* status) {
    const int c=blockDim.x*blockIdx.x+threadIdx.x,nc=nx*ny;if(c>=nc)return;
    const int i=c%nx,j=c/nx;status[c]=0;qx[c]=qy[c]=0.;
    if(area[c]<=0.||!isfinite(area[c]+bed[c]+sx[c]+sy[c]+magnitude[c])||wet_slope<=0.||avalanche_time<=0.){status[c]=1;return;}
    if(i+1<nx)qx[c]=q8a_wet_flux_v105(area[c],area[c+1],bed[c],bed[c+1],dx,sx[c],magnitude[c],wet_slope,avalanche_time);
    if(j+1<ny)qy[c]=q8a_wet_flux_v105(area[c],area[c+nx],bed[c],bed[c+nx],dy,sy[c],magnitude[c],wet_slope,avalanche_time);
    if(!isfinite(qx[c]+qy[c]))status[c]=2;
}

__device__ inline void q8a_scour_pair_gpu(int a,int b,int nc,int ng,double distance,double slope,
    const double* area,const double* bulk,const double* base,double* layers) {
    double za=base[a],zb=base[b];for(int g=0;g<ng;g++){za+=layers[g*nc+a]/bulk[g];zb+=layers[g*nc+b]/bulk[g];}
    const double ddep=za-zb;if(fabs(ddep/distance)<=slope)return;
    double volume=.25*fabs(ddep)*(area[a]+area[b]);const int donor=ddep>0.?a:b,recipient=ddep>0.?b:a;
    double active=0.;for(int g=0;g<ng;g++)active+=layers[g*nc+donor]/bulk[g]*area[donor];
    volume=fmin(volume,active*(1.-64.*2.2204460492503131e-16));if(volume<=0.)return;
    for(int g=0;g<ng;g++){const double fraction=(layers[g*nc+donor]/bulk[g]*area[donor])/active;
        const double mass=volume*fraction*bulk[g];layers[g*nc+donor]-=mass/area[donor];layers[g*nc+recipient]+=mass/area[recipient];}
}

extern "C" __global__ void braids_q8a_scour_rows_v105(int nx,int ny,int ng,double dx,double slope,
    const double* area,const double* bulk,const double* base,double* layers) {
    const int j=blockDim.x*blockIdx.x+threadIdx.x;if(j>=ny)return;
    for(int i=0;i<nx-1;i++)q8a_scour_pair_gpu(j*nx+i,j*nx+i+1,nx*ny,ng,dx,slope,area,bulk,base,layers);
}
extern "C" __global__ void braids_q8a_scour_columns_v105(int nx,int ny,int ng,double dy,double slope,
    const double* area,const double* bulk,const double* base,double* layers) {
    const int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=nx)return;
    for(int j=0;j<ny-1;j++)q8a_scour_pair_gpu(j*nx+i,(j+1)*nx+i,nx*ny,ng,dy,slope,area,bulk,base,layers);
}

__device__ inline int q8a_neighbour_gpu(int c,int d,int nx,int ny){const int i=c%nx,j=c/nx;
    if(d==0)return i>0?c-1:-1;if(d==1)return i+1<nx?c+1:-1;if(d==2)return j>0?c-nx:-1;return j+1<ny?c+nx:-1;}

extern "C" __global__ void braids_q8a_dry_proposal_v105(int nx,int ny,int ng,double thetsd,
    double hmaxth,double sedthr,const double* area,const double* bulk,const double* bed,
    const double* depth,const int* wet,const int* erodible,const double* erosion,
    const double* fraction,const double* fixfac,const double* initial,double* proposal) {
    const int c=blockDim.x*blockIdx.x+threadIdx.x,nc=nx*ny;if(c>=nc)return;
    for(int d=0;d<4;d++)for(int g=0;g<ng;g++)proposal[((long long)c*4+d)*ng+g]=0.;
    // Match FLOW's net total mass erosion, including depositing classes.
    if(!wet[c])return;double erosion_mass=0.;for(int g=0;g<ng;g++)erosion_mass-=erosion[g*nc+c];
    if(erosion_mass<=0.)return;double weight=0.,minimum_area=area[c];
    for(int d=0;d<4;d++){const int n=q8a_neighbour_gpu(c,d,nx,ny);if(n<0||wet[n]||!erodible[n]||bed[n]<=bed[c])continue;
        double stock=0.,local=0.;for(int g=0;g<ng;g++){stock+=initial[g*nc+n]/bulk[g];local+=fixfac[g*nc+n]*fraction[g*nc+n];}
        if(stock>0.){minimum_area=fmin(minimum_area,area[n]);weight+=local;}}
    if(weight<=1.e-7)return;const double theta=q8a_dry_bank_theta_v105(depth[c],thetsd,hmaxth,sedthr);
    const double mass=minimum_area*erosion_mass*theta;
    for(int d=0;d<4;d++){const int n=q8a_neighbour_gpu(c,d,nx,ny);if(n<0||wet[n]||!erodible[n]||bed[n]<=bed[c])continue;
        double stock=0.;for(int g=0;g<ng;g++)stock+=initial[g*nc+n]/bulk[g];if(stock<=0.)continue;
        for(int g=0;g<ng;g++)proposal[((long long)c*4+d)*ng+g]=mass*fixfac[g*nc+n]*fraction[g*nc+n]/weight;}
}

extern "C" __global__ void braids_q8a_dry_scale_v105(int nx,int ny,int ng,const double* area,
    const int* wet,const double* initial,const double* proposal,double* scale) {
    const int k=blockDim.x*blockIdx.x+threadIdx.x,nc=nx*ny;if(k>=nc*ng)return;
    const int c=k%nc,g=k/nc;if(wet[c]){scale[k]=1.;return;}double request=0.;
    for(int d=0;d<4;d++){const int w=q8a_neighbour_gpu(c,d,nx,ny);if(w>=0)request+=proposal[((long long)w*4+(d^1))*ng+g];}
    scale[k]=q8a_transfer_scale_v105(initial[k]*area[c],request);
}

extern "C" __global__ void braids_q8a_dry_gather_v105(int nx,int ny,int ng,const double* area,
    const int* wet,const double* initial,const double* proposal,const double* scale,
    double* result,double* moved,int* status) {
    const int k=blockDim.x*blockIdx.x+threadIdx.x,nc=nx*ny;if(k>=nc*ng)return;
    const int c=k%nc,g=k/nc;double change=0.;
    if(wet[c])for(int d=0;d<4;d++){const int n=q8a_neighbour_gpu(c,d,nx,ny);if(n>=0)change+=proposal[((long long)c*4+d)*ng+g]*scale[g*nc+n];}
    else for(int d=0;d<4;d++){const int w=q8a_neighbour_gpu(c,d,nx,ny);if(w>=0)change-=proposal[((long long)w*4+(d^1))*ng+g]*scale[k];}
    result[k]=initial[k]+change/area[c];moved[k]=change;
    if(!isfinite(result[k])||result[k]<-1.e-10)atomicExch(status+c,2);
}

extern "C" __global__ void braids_q8a_bed_v105(int nc,int ng,const double* bulk,const double* rock,
    const double* layers,double* bed){const int c=blockDim.x*blockIdx.x+threadIdx.x;if(c>=nc)return;
    bed[c]=rock[c];for(int g=0;g<ng;g++)bed[c]+=layers[g*nc+c]/bulk[g];}

extern "C" __global__ void braids_q8a_repose_proposal_v105(int nx,int ny,int ng,double dx,double dy,
    double ds,double diffusivity,double wet_slope,double dry_slope,double switch_depth,
    const double* bulk,const double* bed,const double* depth,const double* layers,double* qx,double* qy) {
    const int c=blockDim.x*blockIdx.x+threadIdx.x,nc=nx*ny;if(c>=nc)return;const int i=c%nx,j=c/nx;
    for(int g=0;g<ng;g++){qx[g*nc+c]=0.;qy[g*nc+c]=0.;}
    for(int axis=0;axis<2;axis++){if((axis==0&&i+1>=nx)||(axis==1&&j+1>=ny))continue;const int n=axis==0?c+1:c+nx;
        const double distance=axis==0?dx:dy,face=axis==0?dy:dx,slope=(bed[c]-bed[n])/distance;
        const double critical=q8a_repose_v105(depth[c],depth[n],switch_depth,wet_slope,dry_slope);if(fabs(slope)<=critical)continue;
        const int donor=slope>0.?c:n;double stock=0.;for(int g=0;g<ng;g++)stock+=layers[g*nc+donor]/bulk[g];if(stock<=0.)continue;
        const double volume=ds*diffusivity*face*(fabs(slope)-critical);
        for(int g=0;g<ng;g++){const double mass=volume*(layers[g*nc+donor]/bulk[g])/stock*bulk[g];
            (axis==0?qx:qy)[g*nc+c]=(donor==c?mass:-mass);}}
}

extern "C" __global__ void braids_q8a_repose_scale_v105(int nx,int ny,int ng,const double* area,
    const double* layers,const double* qx,const double* qy,double* scale){
    const int k=blockDim.x*blockIdx.x+threadIdx.x,nc=nx*ny;if(k>=nc*ng)return;const int c=k%nc,g=k/nc,i=c%nx,j=c/nx;
    double request=0.;if(i+1<nx)request+=fmax(qx[k],0.);if(i>0)request+=fmax(-qx[g*nc+c-1],0.);
    if(j+1<ny)request+=fmax(qy[k],0.);if(j>0)request+=fmax(-qy[g*nc+c-nx],0.);
    scale[k]=q8a_transfer_scale_v105(layers[k]*area[c],request);
}

extern "C" __global__ void braids_q8a_repose_gather_v105(int nx,int ny,int ng,const double* area,
    const double* initial,const double* qx,const double* qy,const double* scale,double* result,int* status){
    const int k=blockDim.x*blockIdx.x+threadIdx.x,nc=nx*ny;if(k>=nc*ng)return;const int c=k%nc,g=k/nc,i=c%nx,j=c/nx;double change=0.;
    if(i+1<nx){double q=qx[k];q*=scale[g*nc+(q>0?c:c+1)];change-=q;}if(i>0){double q=qx[g*nc+c-1];q*=scale[g*nc+(q>0?c-1:c)];change+=q;}
    if(j+1<ny){double q=qy[k];q*=scale[g*nc+(q>0?c:c+nx)];change-=q;}if(j>0){double q=qy[g*nc+c-nx];q*=scale[g*nc+(q>0?c-nx:c)];change+=q;}
    result[k]=initial[k]+change/area[c];if(result[k]<0.&&result[k]>-1.e-10)result[k]=0.;
    if(!isfinite(result[k])||result[k]<0.)atomicExch(status+c,2);
}
