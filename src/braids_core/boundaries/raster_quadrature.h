// Exact bilinear DEM quadrature over polygon rings, in local pixel coordinates.
// Preparation only: independent jobs use CPU OpenMP on either solver backend.
#include <algorithm>
#include <cmath>
#include <vector>
#include <omp.h>
#ifdef _WIN32
#define ENTRY extern "C" __declspec(dllexport)
#else
#define ENTRY extern "C"
#endif
ENTRY void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
struct RasterPoint {double x,y;};

static void raster_clip(std::vector<RasterPoint>& a,
                        std::vector<RasterPoint>& b,int axis,double edge,bool lower){
    b.clear();if(a.empty())return;
    RasterPoint p=a.back();double pv=axis?p.y:p.x;
    bool pin=lower?pv>=edge:pv<=edge;
    for(const RasterPoint& q:a){
        double qv=axis?q.y:q.x;bool qin=lower?qv>=edge:qv<=edge;
        if(pin!=qin){
            double t=(edge-pv)/(qv-pv);
            RasterPoint r={p.x+t*(q.x-p.x),p.y+t*(q.y-p.y)};
            if(axis)r.y=edge;else r.x=edge;
            b.push_back(r);
        }
        if(qin)b.push_back(q);
        p=q;pv=qv;pin=qin;
    }
    a.swap(b);
}

ENTRY void raster_integrals(int n,int width,int height,double pixel_area,
 const int* cell_rings,const int* ring_points,const double* xy,
 const int* jobs,const double* z,const unsigned char* valid,
 double* output,int* error){
    #pragma omp parallel for schedule(dynamic,1)
    for(int job=0;job<n;++job){
        const int* spec=jobs+5*job;int cell=spec[0];
        double integral=0.,area=0.;int bad=0;
        std::vector<RasterPoint> a,b;
        for(int row=spec[3];row<spec[4]&&!bad;++row){
            double yl=std::max(0.,row+.5),yr=std::min(double(height),row+1.5);
            for(int col=spec[1];col<spec[2];++col){
                double xl=std::max(0.,col+.5),xr=std::min(double(width),col+1.5);
                double m[4]={0.,0.,0.,0.};
                for(int ring=cell_rings[cell];ring<cell_rings[cell+1];++ring){
                    a.clear();
                    for(int k=ring_points[ring];k<ring_points[ring+1];++k)
                        a.push_back({(xy[2*k]-xl)/(xr-xl),(xy[2*k+1]-yl)/(yr-yl)});
                    raster_clip(a,b,0,0.,true);raster_clip(a,b,0,1.,false);
                    raster_clip(a,b,1,0.,true);raster_clip(a,b,1,1.,false);
                    if(a.size()<3)continue;
                    double s[4]={0.,0.,0.,0.};
                    for(size_t k=0;k<a.size();++k){
                        const RasterPoint p=a[k],q=a[(k+1)%a.size()];
                        double cross=p.x*q.y-q.x*p.y;
                        s[0]+=cross;s[1]+=(p.x+q.x)*cross;s[2]+=(p.y+q.y)*cross;
                        s[3]+=(2*p.x*p.y+p.x*q.y+q.x*p.y+2*q.x*q.y)*cross;
                    }
                    if(s[0]==0.)continue;
                    double sign=(ring==cell_rings[cell]?1.:-1.)*std::copysign(1.,s[0]);
                    m[0]+=sign*s[0]/2.;m[1]+=sign*s[1]/6.;
                    m[2]+=sign*s[2]/6.;m[3]+=sign*s[3]/24.;
                }
                if(m[0]<=0.)continue;
                int r0=std::max(0,row),r1=std::min(height-1,row+1);
                int c0=std::max(0,col),c1=std::min(width-1,col+1);
                size_t p00=size_t(r0)*width+c0,p01=size_t(r0)*width+c1;
                size_t p10=size_t(r1)*width+c0,p11=size_t(r1)*width+c1;
                if(!valid[p00]||!valid[p01]||!valid[p10]||!valid[p11]){bad=1;break;}
                double jac=(xr-xl)*(yr-yl)*pixel_area;
                integral+=(z[p00]*m[0]+(z[p01]-z[p00])*m[1]+(z[p10]-z[p00])*m[2]
                    +(z[p11]-z[p01]-z[p10]+z[p00])*m[3])*jac;
                area+=m[0]*jac;
            }
        }
        output[2*job]=integral;output[2*job+1]=area;error[job]=bad;
    }
}
