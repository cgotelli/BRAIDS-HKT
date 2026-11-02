#include <algorithm>
#include <cmath>
#include <vector>
#include "braids_planform_v105.h"

using std::isfinite;

static inline int q8a_cell(int i,int j,int nx) { return j*nx+i; }

extern "C" void braids_q8a_slopes_v105(int nx,int ny,int nt,double dx,double dy,
    const double* bed,double* sx,double* sy,double* magnitude,int* status) {
    const int nc=nx*ny;
    #pragma omp parallel for schedule(static) num_threads(nt)
    for(int c=0;c<nc;c++) {
        const int i=c%nx,j=c/nx;
        status[c]=0;
        if(!isfinite(bed[c])) {status[c]=1;sx[c]=sy[c]=magnitude[c]=0.;continue;}
        sx[c]=i+1<nx?q8a_depth_slope_v105(bed[c],bed[c+1],dx):0.;
        sy[c]=j+1<ny?q8a_depth_slope_v105(bed[c],bed[c+nx],dy):0.;
        magnitude[c]=std::hypot(sx[c],sy[c]);
        if(!isfinite(magnitude[c])) status[c]=1;
    }
}

extern "C" void braids_q8a_shear_map_v105(int nc,int nm,int nt,const double* shear,
    const int* reference,const int* application,const double* factor,double* mapped,int* status) {
    #pragma omp parallel for schedule(static) num_threads(nt)
    for(int c=0;c<nc;c++) {mapped[c]=shear[c];status[c]=isfinite(shear[c])?0:1;}
    #pragma omp parallel for schedule(static) num_threads(nt)
    for(int c=0;c<nc;c++) {
        for(int k=0;k<nm;k++) if(application[k]==c) {
            const int r=reference[k];
            if(r<0||r>=nc||!isfinite(factor[k])) {status[c]=2;continue;}
            mapped[c]=shear[r]*factor[k]; /* FLOW: the last matching row wins. */
        }
        if(!isfinite(mapped[c])) status[c]=2;
    }
}

extern "C" void braids_q8a_classic_flux_v105(int nx,int ny,int nt,double dx,double dy,
    double wet_slope,double avalanche_time,const double* area,const double* bed,
    const double* sx,const double* sy,const double* magnitude,double* qx,double* qy,
    int* status) {
    const int nc=nx*ny;
    #pragma omp parallel for schedule(static) num_threads(nt)
    for(int c=0;c<nc;c++) {
        const int i=c%nx,j=c/nx;status[c]=0;qx[c]=qy[c]=0.;
        if(area[c]<=0.||!isfinite(area[c]+bed[c]+sx[c]+sy[c]+magnitude[c])||
           wet_slope<=0.||avalanche_time<=0.) {status[c]=1;continue;}
        if(i+1<nx) qx[c]=q8a_wet_flux_v105(area[c],area[c+1],bed[c],bed[c+1],
            dx,sx[c],magnitude[c],wet_slope,avalanche_time);
        if(j+1<ny) qy[c]=q8a_wet_flux_v105(area[c],area[c+nx],bed[c],bed[c+nx],
            dy,sy[c],magnitude[c],wet_slope,avalanche_time);
        if(!isfinite(qx[c]+qy[c])) status[c]=2;
    }
}

static void q8a_scour_pair(int a,int b,int nc,int ng,double distance,double slope,
    const double* area,const double* bulk,const double* base,double* layers) {
    double za=base[a],zb=base[b];
    for(int g=0;g<ng;g++) {za+=layers[g*nc+a]/bulk[g];zb+=layers[g*nc+b]/bulk[g];}
    const double ddep=za-zb; /* dps_b-dps_a */
    if(std::abs(ddep/distance)<=slope) return;
    double volume=.25*std::abs(ddep)*(area[a]+area[b]);
    const int donor=ddep>0.?a:b, recipient=ddep>0.?b:a;
    double active=0.;for(int g=0;g<ng;g++) active+=layers[g*nc+donor]/bulk[g]*area[donor];
    volume=std::min(volume,active*(1.-64.*2.2204460492503131e-16));
    if(volume<=0.) return;
    for(int g=0;g<ng;g++) {
        const double fraction=(layers[g*nc+donor]/bulk[g]*area[donor])/active;
        const double mass=volume*fraction*bulk[g];
        layers[g*nc+donor]-=mass/area[donor];
        layers[g*nc+recipient]+=mass/area[recipient];
    }
}

extern "C" void braids_q8a_scour_v105(int nx,int ny,int ng,int nt,double dx,double dy,
    double slope,const double* area,const double* bulk,const double* base,const double* initial,double* result,
    int* status) {
    const int nc=nx*ny;
    std::copy(initial,initial+ng*nc,result);
    std::fill(status,status+nc,0);
    if(slope<=0.||dx<=0.||dy<=0.) {std::fill(status,status+nc,1);return;}
    /* Exact FLOW traversal direction, parallel only across independent rows/columns. */
    #pragma omp parallel for schedule(static) num_threads(nt)
    for(int j=0;j<ny;j++) for(int i=0;i<nx-1;i++)
        q8a_scour_pair(q8a_cell(i,j,nx),q8a_cell(i+1,j,nx),nc,ng,dx,slope,area,bulk,base,result);
    #pragma omp parallel for schedule(static) num_threads(nt)
    for(int i=0;i<nx;i++) for(int j=0;j<ny-1;j++)
        q8a_scour_pair(q8a_cell(i,j,nx),q8a_cell(i,j+1,nx),nc,ng,dy,slope,area,bulk,base,result);
    #pragma omp parallel for schedule(static) num_threads(nt)
    for(int c=0;c<nc;c++) for(int g=0;g<ng;g++)
        if(!isfinite(result[g*nc+c])||result[g*nc+c]<-1.e-10) status[c]=2;
}

static inline int q8a_neighbour(int c,int direction,int nx,int ny) {
    const int i=c%nx,j=c/nx;
    if(direction==0) return i>0?c-1:-1;
    if(direction==1) return i+1<nx?c+1:-1;
    if(direction==2) return j>0?c-nx:-1;
    return j+1<ny?c+nx:-1;
}
static inline int q8a_opposite(int d) {return d^1;}

extern "C" void braids_q8a_dry_bank_v105(int nx,int ny,int ng,int nt,double thetsd,
    double hmaxth,double sedthr,const double* area,const double* bulk,const double* bed,
    const double* depth,const int* wet,const int* erodible,const double* erosion,
    const double* fraction,const double* fixfac,const double* initial,double* result,
    double* moved,int* status) {
    const int nc=nx*ny,nlink=4*nc;
    std::vector<double> proposal((size_t)nlink*ng,0.),scale((size_t)nc*ng,1.);
    std::copy(initial,initial+ng*nc,result);std::fill(moved,moved+ng*nc,0.);
    std::fill(status,status+nc,0);
    #pragma omp parallel for schedule(static) num_threads(nt)
    for(int c=0;c<nc;c++) {
        if(!wet[c]) continue;
        // FLOW bott3d: redistribution is driven by NET total mass erosion,
        // not the sum of erosive classes or an equivalent eroded volume.
        double erosion_mass=0.;for(int g=0;g<ng;g++) erosion_mass-=erosion[g*nc+c];
        if(erosion_mass<=0.) continue;
        double weight=0.,minimum_area=area[c];
        for(int d=0;d<4;d++) {const int n=q8a_neighbour(c,d,nx,ny);
            if(n<0||wet[n]||!erodible[n]||bed[n]<=bed[c]) continue;
            double stock=0.,local_weight=0.;for(int g=0;g<ng;g++) {
                stock+=initial[g*nc+n]/bulk[g];local_weight+=fixfac[g*nc+n]*fraction[g*nc+n];}
            if(stock>0.) {minimum_area=std::min(minimum_area,area[n]);weight+=local_weight;}
        }
        if(weight<=1.e-7) continue;
        const double theta=q8a_dry_bank_theta_v105(depth[c],thetsd,hmaxth,sedthr);
        const double mass=minimum_area*erosion_mass*theta;
        for(int d=0;d<4;d++) {const int n=q8a_neighbour(c,d,nx,ny);if(n<0||wet[n]||!erodible[n]||bed[n]<=bed[c])continue;
            double stock=0.;for(int g=0;g<ng;g++)stock+=initial[g*nc+n]/bulk[g];if(stock<=0.)continue;
            for(int g=0;g<ng;g++) proposal[((size_t)c*4+d)*ng+g]=mass*fixfac[g*nc+n]*fraction[g*nc+n]/weight;
        }
    }
    #pragma omp parallel for schedule(static) num_threads(nt)
    for(int c=0;c<nc;c++) if(!wet[c]) for(int g=0;g<ng;g++) {
        double request=0.;for(int d=0;d<4;d++) {const int w=q8a_neighbour(c,d,nx,ny);if(w>=0)
            request+=proposal[((size_t)w*4+q8a_opposite(d))*ng+g];}
        scale[g*nc+c]=q8a_transfer_scale_v105(initial[g*nc+c]*area[c],request);
    }
    #pragma omp parallel for schedule(static) num_threads(nt)
    for(int c=0;c<nc;c++) for(int g=0;g<ng;g++) {
        double change=0.;
        if(wet[c]) for(int d=0;d<4;d++) {const int n=q8a_neighbour(c,d,nx,ny);if(n>=0)
            change+=proposal[((size_t)c*4+d)*ng+g]*scale[g*nc+n];}
        else for(int d=0;d<4;d++) {const int w=q8a_neighbour(c,d,nx,ny);if(w>=0)
            change-=proposal[((size_t)w*4+q8a_opposite(d))*ng+g]*scale[g*nc+c];}
        result[g*nc+c]+=change/area[c];moved[g*nc+c]=change;
        if(!isfinite(result[g*nc+c])||result[g*nc+c]<-1.e-10) status[c]=2;
    }
}

extern "C" void braids_q8a_repose_v105(int nx,int ny,int ng,int nt,double dx,double dy,
    double dt,double diffusivity,double wet_slope,double dry_slope,double switch_depth,
    const double* area,const double* bulk,const double* rock,const double* depth,
    const double* initial,double* result,int* status,int* substeps_out) {
    const int nc=nx*ny;std::copy(initial,initial+ng*nc,result);std::fill(status,status+nc,0);
    std::vector<double> bed(nc),qx((size_t)ng*nc),qy((size_t)ng*nc),scale((size_t)ng*nc);
    double remaining=dt;int substeps=0;
    while(remaining>64.*2.2204460492503131e-16*std::max(dt,1.)) {
        if(++substeps>100000) {std::fill(status,status+nc,3);break;}
        #pragma omp parallel for schedule(static) num_threads(nt)
        for(int c=0;c<nc;c++) {bed[c]=rock[c];for(int g=0;g<ng;g++)bed[c]+=result[g*nc+c]/bulk[g];}
        double maximum=0.;
        #pragma omp parallel for reduction(max:maximum) schedule(static) num_threads(nt)
        for(int c=0;c<nc;c++) {const int i=c%nx,j=c/nx;double conductance=0.;
            if(i+1<nx) conductance+=diffusivity*dy/dx;
            if(i>0) conductance+=diffusivity*dy/dx;
            if(j+1<ny) conductance+=diffusivity*dx/dy;
            if(j>0) conductance+=diffusivity*dx/dy;
            maximum=std::max(maximum,conductance/area[c]);}
        if(maximum<=0.) break;
        const double ds=std::min(remaining,.45/maximum);
        std::fill(qx.begin(),qx.end(),0.);std::fill(qy.begin(),qy.end(),0.);
        #pragma omp parallel for schedule(static) num_threads(nt)
        for(int c=0;c<nc;c++) {const int i=c%nx,j=c/nx;
            for(int axis=0;axis<2;axis++) {if((axis==0&&i+1>=nx)||(axis==1&&j+1>=ny))continue;
                const int n=axis==0?c+1:c+nx;const double distance=axis==0?dx:dy;
                const double face=axis==0?dy:dx;const double slope=(bed[c]-bed[n])/distance;
                const double critical=q8a_repose_v105(depth[c],depth[n],switch_depth,wet_slope,dry_slope);
                if(std::abs(slope)<=critical)continue;
                const int donor=slope>0.?c:n;
                double stock=0.;for(int g=0;g<ng;g++)stock+=result[g*nc+donor]/bulk[g];if(stock<=0.)continue;
                const double volume=ds*diffusivity*face*(std::abs(slope)-critical);
                for(int g=0;g<ng;g++) {const double mass=volume*(result[g*nc+donor]/bulk[g])/stock*bulk[g];
                    (axis==0?qx:qy)[g*nc+c]=(donor==c?mass:-mass);}
            }}
        #pragma omp parallel for schedule(static) num_threads(nt)
        for(int c=0;c<nc;c++) for(int g=0;g<ng;g++) {double request=0.;const int i=c%nx,j=c/nx;
            if(i+1<nx)request+=std::max(qx[g*nc+c],0.);
            if(i>0)request+=std::max(-qx[g*nc+c-1],0.);
            if(j+1<ny)request+=std::max(qy[g*nc+c],0.);
            if(j>0)request+=std::max(-qy[g*nc+c-nx],0.);
            scale[g*nc+c]=q8a_transfer_scale_v105(result[g*nc+c]*area[c],request);}
        #pragma omp parallel for schedule(static) num_threads(nt)
        for(int c=0;c<nc;c++) for(int g=0;g<ng;g++) {const int i=c%nx,j=c/nx;double change=0.;
            if(i+1<nx){double q=qx[g*nc+c];q*=scale[g*nc+(q>0?c:c+1)];change-=q;}
            if(i>0){double q=qx[g*nc+c-1];q*=scale[g*nc+(q>0?c-1:c)];change+=q;}
            if(j+1<ny){double q=qy[g*nc+c];q*=scale[g*nc+(q>0?c:c+nx)];change-=q;}
            if(j>0){double q=qy[g*nc+c-nx];q*=scale[g*nc+(q>0?c-nx:c)];change+=q;}
            result[g*nc+c]+=change/area[c];if(result[g*nc+c]<0.&&result[g*nc+c]>-1.e-10)result[g*nc+c]=0.;
            if(!isfinite(result[g*nc+c])||result[g*nc+c]<0.)status[c]=2;}
        if(std::any_of(status,status+nc,[](int x){return x!=0;}))break;
        remaining=std::max(0.,remaining-ds);
    }
    *substeps_out=substeps;
}
