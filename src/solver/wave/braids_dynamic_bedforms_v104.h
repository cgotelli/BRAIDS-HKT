// Q7b dynamic Delft3D4-FLOW bedform equations shared by OpenMP and CUDA.
// Delft3D is a read-only reference.  BRAIDS keeps its constitutive choices,
// but advances the scalar dune-height field with a conservative TVD operator.
#ifndef BRAIDS_HD
#define BRAIDS_HD
#endif

BRAIDS_HD inline double b104_minmod(double a, double b) {
    return a*b <= 0.0 ? 0.0 : (fabs(a) < fabs(b) ? a : b);
}
BRAIDS_HD inline double b104_clip(double x, double a, double b) {
    return fmin(b, fmax(a, x));
}

// mode: relaxation, advection, morphological clock, CFL control, roughness.
// p: g,T_H,L_H,LHc,Hmax,LHp,Pmax,aC,bC,CHp,CHn,Gmin,aK,bK,dry,
//    ripple/mega-ripple/dune roughness relaxation times.
BRAIDS_HD inline void b104_diagnostic_cell(
    int c, int nc, const int* mode, const double* p,
    const double* depth, const double* u, const double* v,
    const double* bedload_x, const double* bedload_y,
    const double* height, const double* equilibrium, const double* length,
    const double* ripple, const double* mega, const double* static_dune,
    double* out, int* status) {
    status[c] = 0;
    const double h = depth[c], ux = u[c], uy = v[c];
    const double speed = hypot(ux, uy);
    if (!isfinite(h+ux+uy+bedload_x[c]+bedload_y[c]+height[c]+equilibrium[c]+
                  length[c]+ripple[c]+mega[c]+static_dune[c]) || h < 0.0 || height[c] < 0.0 ||
        equilibrium[c] < 0.0 || length[c] <= 0.0 || ripple[c] < 0.0 || mega[c] < 0.0 ||
        static_dune[c] < 0.0) {
        status[c] = 1; return;
    }
    double cx = 0.0, cy = 0.0, trelax = 0.0;
    const int rlx = mode[0];
    if (rlx > 0 && h > p[14]) {
        if (rlx <= 3) {
            const double scale = p[7] * pow(fmax(speed, 1.e-30), p[8]-1.0) / h;
            cx = scale*ux; cy = scale*uy;
        } else {
            const double gamma = b104_clip(pow(h/p[4], p[9]), p[11], 1.0);
            const double fr2 = speed*speed/(p[0]*h);
            const double scale = gamma*p[10]/(h*fmax(1.0-fr2, 0.1));
            cx = scale*bedload_x[c]; cy = scale*bedload_y[c];
        }
        if (rlx == 1) trelax = p[1];
        else {
            double adaptation_length = p[2];
            if (rlx == 3) adaptation_length = p[3]*length[c];
            if (rlx == 4) {
                const double phi = fmax(fmin(pow(p[4]/h, p[5]), p[6]), 1.0);
                adaptation_length = phi*length[c];
            }
            const double celerity = hypot(cx, cy);
            trelax = celerity > 1.e-20 ? adaptation_length/celerity : 0.0;
        }
    }
    double dune_roughness = static_dune[c];
    if (mode[4] == 1)
        dune_roughness = 1.1*height[c]*(1.0-exp(-25.0*height[c]/fmax(length[c],1.e-6)));
    else if (mode[4] == 2)
        dune_roughness = p[12]*pow(fabs(height[c]),p[13]);
    const double total = sqrt(ripple[c]*ripple[c] + mega[c]*mega[c] +
                              dune_roughness*dune_roughness);
    out[c] = cx; out[nc+c] = cy; out[2*nc+c] = trelax;
    out[3*nc+c] = dune_roughness; out[4*nc+c] = total;
    if (!isfinite(cx+cy+trelax+dune_roughness+total) || trelax < 0.0) status[c] = 2;
}

BRAIDS_HD inline void b104_relax_roughness_cell(int c,int nc,const double* p,double dt,
    const double* ripple,const double* mega,const double* dune,const double* target_ripple,
    const double* target_mega,const double* target_dune,double* out,int* status) {
    status[c]=0;
    const double current[3]={ripple[c],mega[c],dune[c]};
    const double target[3]={target_ripple[c],target_mega[c],target_dune[c]};
    for(int k=0;k<3;k++) {
        if(!isfinite(current[k]+target[k]) || current[k]<0. || target[k]<0.) {status[c]=1;return;}
        const double tau=p[15+k];
        const double factor=tau>0.?exp(-dt/tau):0.;
        out[k*nc+c]=factor*current[k]+(1.-factor)*target[k];
    }
}

BRAIDS_HD inline int b104_wrap(int q, int n) {
    q %= n; return q < 0 ? q+n : q;
}

// boundary order: west,east,south,north. 0 closed/zero, 1 open/zero-gradient,
// 2 periodic.  The wet mask blocks flux through moving wet/dry interfaces.
BRAIDS_HD inline double b104_sample(const double* a, int nx, int ny, int i, int j,
                                    const int* boundary, int closed_zero) {
    if (i < 0) {
        if (boundary[0] == 2) i = b104_wrap(i,nx);
        else if (boundary[0] == 1 || !closed_zero) i = 0;
        else return 0.0;
    } else if (i >= nx) {
        if (boundary[1] == 2) i = b104_wrap(i,nx);
        else if (boundary[1] == 1 || !closed_zero) i = nx-1;
        else return 0.0;
    }
    if (j < 0) {
        if (boundary[2] == 2) j = b104_wrap(j,ny);
        else if (boundary[2] == 1 || !closed_zero) j = 0;
        else return 0.0;
    } else if (j >= ny) {
        if (boundary[3] == 2) j = b104_wrap(j,ny);
        else if (boundary[3] == 1 || !closed_zero) j = ny-1;
        else return 0.0;
    }
    return a[j*nx+i];
}

BRAIDS_HD inline int b104_wet_sample(const int* wet, int nx, int ny, int i, int j,
                                     const int* boundary) {
    if (i < 0) {
        if (boundary[0] == 2) i=b104_wrap(i,nx); else if (boundary[0] == 1) i=0; else return 1;
    } else if (i >= nx) {
        if (boundary[1] == 2) i=b104_wrap(i,nx); else if (boundary[1] == 1) i=nx-1; else return 1;
    }
    if (j < 0) {
        if (boundary[2] == 2) j=b104_wrap(j,ny); else if (boundary[2] == 1) j=0; else return 1;
    } else if (j >= ny) {
        if (boundary[3] == 2) j=b104_wrap(j,ny); else if (boundary[3] == 1) j=ny-1; else return 1;
    }
    return wet[j*nx+i];
}

BRAIDS_HD inline double b104_slope_x(const double* h,int nx,int ny,int i,int j,const int* b) {
    const double hm=b104_sample(h,nx,ny,i-1,j,b,1), hc=b104_sample(h,nx,ny,i,j,b,1);
    const double hp=b104_sample(h,nx,ny,i+1,j,b,1);
    return b104_minmod(hc-hm,hp-hc);
}
BRAIDS_HD inline double b104_slope_y(const double* h,int nx,int ny,int i,int j,const int* b) {
    const double hm=b104_sample(h,nx,ny,i,j-1,b,1), hc=b104_sample(h,nx,ny,i,j,b,1);
    const double hp=b104_sample(h,nx,ny,i,j+1,b,1);
    return b104_minmod(hc-hm,hp-hc);
}

BRAIDS_HD inline double b104_flux_x(const double* h,const double* cx,const int* wet,
    int nx,int ny,int il,int j,const int* b) {
    const int ir=il+1;
    if (!b104_wet_sample(wet,nx,ny,il,j,b) || !b104_wet_sample(wet,nx,ny,ir,j,b)) return 0.0;
    const double hl=fmax(0.0,b104_sample(h,nx,ny,il,j,b,1)+.5*b104_slope_x(h,nx,ny,il,j,b));
    const double hr=fmax(0.0,b104_sample(h,nx,ny,ir,j,b,1)-.5*b104_slope_x(h,nx,ny,ir,j,b));
    const double cf=.5*(b104_sample(cx,nx,ny,il,j,b,0)+b104_sample(cx,nx,ny,ir,j,b,0));
    return cf*(cf>=0.0?hl:hr);
}
BRAIDS_HD inline double b104_flux_y(const double* h,const double* cy,const int* wet,
    int nx,int ny,int i,int jl,const int* b) {
    const int jr=jl+1;
    if (!b104_wet_sample(wet,nx,ny,i,jl,b) || !b104_wet_sample(wet,nx,ny,i,jr,b)) return 0.0;
    const double hl=fmax(0.0,b104_sample(h,nx,ny,i,jl,b,1)+.5*b104_slope_y(h,nx,ny,i,jl,b));
    const double hr=fmax(0.0,b104_sample(h,nx,ny,i,jr,b,1)-.5*b104_slope_y(h,nx,ny,i,jr,b));
    const double cf=.5*(b104_sample(cy,nx,ny,i,jl,b,0)+b104_sample(cy,nx,ny,i,jr,b,0));
    return cf*(cf>=0.0?hl:hr);
}

BRAIDS_HD inline void b104_update_cell(int c,int nx,int ny,const int* mode,const int* boundary,
    double dt,double dx,double dy,const double* oldh,const double* equilibrium,
    const double* cx,const double* cy,const double* trelax,const int* wet,
    double* next,int* status) {
    status[c]=0;
    if (!wet[c]) { next[c]=oldh[c]; return; }
    if (mode[0]==0) { next[c]=equilibrium[c]; return; }
    const int i=c%nx,j=c/nx;
    double value=oldh[c];
    if (mode[1]) {
        const double div=(b104_flux_x(oldh,cx,wet,nx,ny,i,j,boundary)-
                          b104_flux_x(oldh,cx,wet,nx,ny,i-1,j,boundary))/dx+
                         (b104_flux_y(oldh,cy,wet,nx,ny,i,j,boundary)-
                          b104_flux_y(oldh,cy,wet,nx,ny,i,j-1,boundary))/dy;
        value -= dt*div;
    }
    if (trelax[c] > 0.0) {
        const double r=dt/trelax[c];
        value=(value+r*equilibrium[c])/(1.0+r); // FLOW: explicit source, implicit sink.
    }
    if (!isfinite(value) || value < -1.e-12) { status[c]=3; return; }
    next[c]=fmax(0.0,value);
}
