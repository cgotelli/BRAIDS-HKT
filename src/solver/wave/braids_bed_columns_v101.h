// BRAIDS column implementation. Equations inspected in locked FLOW bedcomposition_module.
// Independent columns run in OpenMP/CUDA; vertical transfers are ordered, not parallel races.
#ifndef BRAIDS_HD
#define BRAIDS_HD
#endif
BRAIDS_HD inline void bed_targets_v101(int c,int nc,int nl,int method,int nlag,
    double active,double alpha,double minimum,double eulerian,double exchange,
    double lagrangian,const double* depth,double* target) {
    int start=1+int(exchange>0.);
    for(int k=0;k<nl;k++) {
        double value=eulerian;
        if(k==0) value=method==2?fmax(alpha*depth[c],minimum):active;
        else if(exchange>0. && k==1) value=exchange;
        else if(k>=start && k<start+nlag) value=lagrangian;
        target[k*nc+c]=value;
    }
}
BRAIDS_HD inline double bed_porosity_v101(int ng, int nc, int c, int mode,
    const double* mass, const double* phi, const double* sig,
    double sv0, double svsand, double svmud, const int* mud) {
    double total=0., mean=0., variance=0.;
    for(int g=0;g<ng;g++) total+=mass[g*nc+c];
    if(total<=0.) return 0.;
    for(int g=0;g<ng;g++) mean+=mass[g*nc+c]*phi[g*nc+c]/total;
    for(int g=0;g<ng;g++) {
        double p=phi[g*nc+c]-mean, s=sig[g*nc+c];
        variance+=mass[g*nc+c]*(p*p+s*s)/total;
    }
    double sigma=sqrt(variance);
    if(mode==1) return fmax(0.,.36-.06*sigma);
    if(mode==2) { // analytical monodisperse limit, avoiding infinity / infinity
        if(sigma==0.) return .45;
        double x=3.7632*pow(sigma,-.7552); return .45*x/(1.+x);
    }
    if(mode==3) return 1.-sv0;
    if(mode==4) {
        double p=0.;
        for(int g=0;g<ng;g++) p+=(1.-(mud[g]?svmud:svsand))*mass[g*nc+c]/total;
        return p;
    }
    return 0.; // 0 embeds packing in density; locked FLOW option 5 falls through.
}
BRAIDS_HD inline double bed_thickness_v101(int ng,int nc,int c,int k,
    const double* mass,const double* sv,const double* rho) {
    double v=0.; for(int g=0;g<ng;g++) v+=mass[(k*ng+g)*nc+c]/rho[g];
    return v/sv[k*nc+c];
}
BRAIDS_HD inline void bed_limits_v101(int c,int nc,int ng,double mf,
    const double* active,const double* water,const double* depth,
    const double* bedrate,const double* waterrate,const double* rho,
    double maxbed,double maxconc,double entrainment_fraction,double* limits) {
    const double infinity=1.0/0.0;
    double stock=infinity, concentration=infinity, entrainment=infinity, dz=0.;
    for(int g=0;g<ng;g++) {
        int i=g*nc+c; double r=bedrate[i]*mf, w=waterrate[i];
        if(r<0.) stock=fmin(stock,active[i]/(-r));
        if(w>0. && maxconc>0.) concentration=fmin(concentration,fmax(0.,maxconc*depth[c]-water[i])/w);
        if(w>0. && entrainment_fraction>0.) entrainment=fmin(entrainment,entrainment_fraction*active[i]/w);
        dz+=r/rho[g];
    }
    limits[c]=stock;
    limits[nc+c]=concentration;
    limits[2*nc+c]=entrainment;
    limits[3*nc+c]=(maxbed>0. && dz!=0.)?maxbed/fabs(dz):infinity;
}
BRAIDS_HD inline double bed_cdf_v101(int c,int nc,int g,int np,double x,
    const double* nodes,const double* cdf) {
    int start=g*np*nc+c;
    if(x<nodes[start]) return 0.;
    for(int j=1;j<np;j++) {
        double b=nodes[start+j*nc], a=nodes[start+(j-1)*nc];
        if(x<=b && b>a) return cdf[start+(j-1)*nc]+(x-a)/(b-a)*(cdf[start+j*nc]-cdf[start+(j-1)*nc]);
    }
    return 1.;
}
BRAIDS_HD inline void bed_characteristics_v101(int c,int nc,int ng,int np,int nq,
    const double* fractions,const double* medians,const double* means,
    const double* nodes,const double* cdf,const double* quantiles,
    double* result) {
    double dg=0.,dm=0.,variance=0.,lo=1e300,hi=-1e300,total=0.;
    for(int g=0;g<ng;g++) total+=fractions[g*nc+c];
    if(total<=0.) {for(int j=0;j<3+nq;j++) result[j*nc+c]=0.;return;}
    for(int g=0;g<ng;g++) {
        double f=fractions[g*nc+c]/total;
        dg+=f*log(medians[g*nc+c]); dm+=f*means[g*nc+c];
        if(f>0.) {lo=fmin(lo,nodes[g*np*nc+c]);hi=fmax(hi,nodes[(g*np+np-1)*nc+c]);}
    }
    for(int g=0;g<ng;g++) {
        double d=log(medians[g*nc+c])-dg; variance+=fractions[g*nc+c]/total*d*d;
    }
    result[c]=exp(dg);result[nc+c]=dm;result[2*nc+c]=exp(sqrt(variance));
    for(int q=0;q<nq;q++) {
        double a=lo,b=hi;
        for(int it=0;it<56;it++) {
            double x=.5*(a+b), probability=0.;
            for(int g=0;g<ng;g++) probability+=fractions[g*nc+c]/total*bed_cdf_v101(c,nc,g,np,x,nodes,cdf);
            if(probability<quantiles[q]) a=x; else b=x;
        }
        result[(3+q)*nc+c]=exp(.5*(a+b));
    }
}
BRAIDS_HD inline void bed_fixfac_v101(int c,int nc,int ng,int nl,int mode,double threshold,
    const double* mass,const double* sv,const double* rho,const double* target,double* result) {
    double top=bed_thickness_v101(ng,nc,c,0,mass,sv,rho),all=0.;
    for(int k=0;k<nl;k++) all+=bed_thickness_v101(ng,nc,c,k,mass,sv,rho);
    for(int g=0;g<ng;g++) {
        double h=all;
        if(nl==1 && ng>1) h=mass[g*nc+c]/rho[g];
        double value=mode==2?top/fmax(1e-10,target[c]):h/fmax(1e-10,threshold);
        result[g*nc+c]=fmin(1.,fmax(0.,value));
    }
}
BRAIDS_HD inline void bed_tracer_v101(int c,int nc,int ng,const double* mass,
    const double* rho,const double* diameter,const double* calibration,
    const double* fractions,const double* fixfac,double* outfrac,double* outfix) {
    for(int g=0;g<ng;g++) {
        int i=g*nc+c;
        outfrac[i]=fractions[i]; outfix[i]=fixfac[i];
        if(calibration[g]>0.) {
            // erosed FLOW: grkg/grm2 = 12/(physical density * diameter).
            // Tracer fractions deliberately need not sum to one.
            double grain_layers=12.*mass[i]/(rho[g]*diameter[i]);
            outfrac[i]=fmin(1.,fmax(0.,grain_layers))*calibration[g]; outfix[i]=1.;
        }
    }
}
BRAIDS_HD inline void bed_remap_v101(int c,int nc,int oldnc,int ng,int nl,
    const int* offsets,const int* indices,const double* weights,
    const double* mass,const double* sv,const double* rock,const double* rho,
    double* out,double* outsv,double* outrock) {
    outrock[c]=0.;
    for(int p=offsets[c];p<offsets[c+1];p++) outrock[c]+=weights[p]*rock[indices[p]];
    for(int k=0;k<nl;k++) {
        double thickness=0.,solid=0.;
        for(int g=0;g<ng;g++) {
            double m=0.;
            for(int p=offsets[c];p<offsets[c+1];p++) m+=weights[p]*mass[(k*ng+g)*oldnc+indices[p]];
            out[(k*ng+g)*nc+c]=m; solid+=m/rho[g];
        }
        for(int p=offsets[c];p<offsets[c+1];p++) {
            int j=indices[p]; double s=0.;
            for(int g=0;g<ng;g++) s+=mass[(k*ng+g)*oldnc+j]/rho[g];
            thickness+=weights[p]*s/sv[k*oldnc+j];
        }
        // Nonnegative overlap weights imply a volume-weighted harmonic
        // solid fraction bounded by contributing values. Preserve constant
        // porosity exactly (not 1+epsilon on cropped restriction), on CPU/CUDA.
        double svmin=1.,svmax=0.;
        for(int p=offsets[c];p<offsets[c+1];++p) if(weights[p]>0.) {
            double value=sv[k*oldnc+indices[p]];
            svmin=fmin(svmin,value);svmax=fmax(svmax,value);
        }
        outsv[k*nc+c]=thickness>0.?fmin(svmax,fmax(svmin,solid/thickness)):1.;
    }
}
BRAIDS_HD inline void bed_packet_v101(int ng,int nc,int c,int from,int to,
    double fraction,double* mass,double* sv,const double* rho) {
    double a=bed_thickness_v101(ng,nc,c,from,mass,sv,rho)*fraction;
    double b=bed_thickness_v101(ng,nc,c,to,mass,sv,rho);
    double solid=a*sv[from*nc+c]+b*sv[to*nc+c];
    for(int g=0;g<ng;g++) {
        int i=(from*ng+g)*nc+c,j=(to*ng+g)*nc+c;
        double x=mass[i]*fraction; mass[i]-=x; mass[j]+=x;
    }
    if(a+b>0.) sv[to*nc+c]=solid/(a+b);
}
BRAIDS_HD inline void bed_eulerian_bury_v101(int ng,int nc,int c,int nl,int from,int start,
    double amount,const double* target,double* mass,double* sv,const double* rho) {
    if(amount<=0.) return;
    if(start==nl-1) {
        double h=bed_thickness_v101(ng,nc,c,from,mass,sv,rho);
        bed_packet_v101(ng,nc,c,from,start,fmin(1.,amount/h),mass,sv,rho); return;
    }
    int k=start;
    while(k<nl-1 && bed_thickness_v101(ng,nc,c,k,mass,sv,rho)==0.) k++;
    if(k==nl-1 && bed_thickness_v101(ng,nc,c,k,mass,sv,rho)>0.) k--;
    while(k>=start && amount>0.) {
        double h=bed_thickness_v101(ng,nc,c,k,mass,sv,rho), cap=target[k*nc+c];
        if(h<cap) {
            // FLOW allows all new sediment into an empty Eulerian layer;
            // only a partially filled layer is capped at ThEuLyr.
            double a=h>0.?fmin(amount,cap-h):amount;
            double src=bed_thickness_v101(ng,nc,c,from,mass,sv,rho);
            bed_packet_v101(ng,nc,c,from,k,fmin(1.,a/src),mass,sv,rho); amount-=a;
        }
        k--;
    }
    if(amount>0.) {
        bed_packet_v101(ng,nc,c,nl-2,nl-1,1.,mass,sv,rho);
        for(int j=nl-2;j>start;j--) {
            sv[j*nc+c]=sv[(j-1)*nc+c];
            for(int g=0;g<ng;g++) mass[(j*ng+g)*nc+c]=mass[((j-1)*ng+g)*nc+c];
        }
        sv[start*nc+c]=1.; for(int g=0;g<ng;g++) mass[(start*ng+g)*nc+c]=0.;
        double src=bed_thickness_v101(ng,nc,c,from,mass,sv,rho);
        bed_packet_v101(ng,nc,c,from,start,fmin(1.,amount/src),mass,sv,rho);
    }
}
BRAIDS_HD inline void bed_bury_v101(int ng,int nc,int c,int nl,int from,int start,int nlag,
    double amount,const double* target,double* mass,double* sv,const double* rho,double* work) {
    int end=start+nlag;
    if(nlag==0) {bed_eulerian_bury_v101(ng,nc,c,nl,from,start,amount,target,mass,sv,rho);return;}
    double* old=work+5*nl*nc; double* oldsv=old+nl*ng*nc;
    for(int k=start;k<end;k++) {
        oldsv[k*nc+c]=sv[k*nc+c]; sv[k*nc+c]=1.;
        for(int g=0;g<ng;g++) {int i=(k*ng+g)*nc+c;old[i]=mass[i];mass[i]=0.;}
    }
    int first=start;
    for(int k=start;k<end;k++) {
        double cap=target[k*nc+c];
        if(amount>0.) {
            double a=fmin(amount,cap), src=bed_thickness_v101(ng,nc,c,from,mass,sv,rho);
            bed_packet_v101(ng,nc,c,from,k,fmin(1.,a/src),mass,sv,rho); amount-=a;
        }
        double need=cap-bed_thickness_v101(ng,nc,c,k,mass,sv,rho);
        while(first<end && need>0.) {
            double h=bed_thickness_v101(ng,nc,c,first,old,oldsv,rho);
            if(h<=0.) {first++;continue;}
            double a=fmin(h,need), b=bed_thickness_v101(ng,nc,c,k,mass,sv,rho);
            double solid=a*oldsv[first*nc+c]+b*sv[k*nc+c];
            for(int g=0;g<ng;g++) {
                int i=(first*ng+g)*nc+c,j=(k*ng+g)*nc+c;
                double x=old[i]*(a/h); old[i]-=x; mass[j]+=x;
            }
            sv[k*nc+c]=solid/(a+b); need-=a;
            if(a==h) first++;
        }
    }
    // Old bottom packets precede the new surplus in FLOW's Eulerian bookkeeping.
    for(int k=end-1;k>=start;k--) {
        double h=bed_thickness_v101(ng,nc,c,k,old,oldsv,rho);
        if(h<=0.) continue;
        // Temporarily use the now-empty work layer as a packet donor in a
        // separate scratch array, then copy the changed Eulerian rows back.
        for(int j=end;j<nl;j++) {
            oldsv[j*nc+c]=sv[j*nc+c];
            for(int g=0;g<ng;g++) old[(j*ng+g)*nc+c]=mass[(j*ng+g)*nc+c];
        }
        bed_eulerian_bury_v101(ng,nc,c,nl,k,end,h,target,old,oldsv,rho);
        for(int j=end;j<nl;j++) {
            sv[j*nc+c]=oldsv[j*nc+c];
            for(int g=0;g<ng;g++) mass[(j*ng+g)*nc+c]=old[(j*ng+g)*nc+c];
        }
    }
    if(amount>0.) bed_eulerian_bury_v101(ng,nc,c,nl,from,end,amount,target,mass,sv,rho);
}
BRAIDS_HD inline void bed_column_v101(int c,int nc,int ng,int nl,
    int topmode,int basemode,int pormode,int nlag,int exchange,double dt,
    const double* initial,const double* initial_sv,const double* delta,
    const double* target,const double* rho,const double* phi,const double* sig,
    const int* mud,double sv0,double svsand,double svmud,
    int nd,const double* zd,const double* kd,
    double* mass,double* sv,double* source,double* work,int* status,
    const double* coarse_mobility=0) {
    status[c]=0;
    for(int k=0;k<nl;k++) {
        sv[k*nc+c]=initial_sv[k*nc+c];
        for(int g=0;g<ng;g++) mass[(k*ng+g)*nc+c]=initial[(k*ng+g)*nc+c];
    }
    double oldh=bed_thickness_v101(ng,nc,c,0,mass,sv,rho), eroded=0.,dep=0.;
    for(int g=0;g<ng;g++) {
        int i=g*nc+c; double d=delta[i];
        if(mass[i]+d<0.) {status[c]=1;return;} // reject, never manufacture stock
        eroded+=fmax(-d,0.)/rho[g]; dep+=fmax(d,0.)/rho[g];
        mass[i]+=d; source[i]=0.;
        work[i]=fmax(d,0.);
    }
    double solid=0.; for(int g=0;g<ng;g++) solid+=mass[g*nc+c]/rho[g];
    // With packing included in rho, the separate solid-fraction state is
    // exactly one. Recomputing it as solid/h under topmode 2 can create
    // 1+epsilon and reject a perfectly conservative finite-stock update.
    if(pormode==0 || pormode==5) sv[c]=1.;
    else if(topmode==1) sv[c]=1.-bed_porosity_v101(ng,nc,c,pormode,mass,phi,sig,sv0,svsand,svmud,mud);
    else {
        double fresh=1.-bed_porosity_v101(ng,nc,c,pormode,work,phi,sig,sv0,svsand,svmud,mud);
        double h=oldh-eroded/initial_sv[c]+dep/fresh;
        sv[c]=(solid>0. && h>0.)?solid/h:1.;
    }
    // Transport (+ explicitly selected exchange) are Lagrangian surface layers.
    // Eulerian underlayers are not refilled during erosion, preserving buried beds.
    // FLOW's coarse layer is NOT an ordinary exchange layer. Only its
    // mobile inventory may replenish the transport layer; deeper material
    // refills the coarse layer afterwards, never bypassing it in this update.
    int start=1+exchange+int(coarse_mobility!=0);
    // FLOW chooses a copied basal composition inside lyrerosion, BEFORE
    // returning the eroded packet to its surface receiver. Preserve that
    // receiver's composition rather than copying its newly mixed contents.
    double* receiver=work+(6*nl+nl*ng)*nc;
    double* receiversv=receiver+nl*ng*nc;
    bool eroding=false;
    for(int k=0;k<start && k<nl-1;k++) {
        if(basemode==3) {
            receiversv[k*nc+c]=sv[k*nc+c];
            for(int g=0;g<ng;g++) receiver[(k*ng+g)*nc+c]=mass[(k*ng+g)*nc+c];
        }
        double h=bed_thickness_v101(ng,nc,c,k,mass,sv,rho), t=target[k*nc+c];
        double tolerance=1.4210854715202004e-14*fmax(1.,fmax(h,t));
        if(h-t>tolerance) {
            if(k+1<start) bed_packet_v101(ng,nc,c,k,k+1,(h-t)/h,mass,sv,rho);
            else bed_bury_v101(ng,nc,c,nl,k,start,nlag,h-t,target,mass,sv,rho,work);
        }
        else if(t-h>tolerance) {
            eroding=true;
            double need=t-h;
            if(k==0 && coarse_mobility) {
                double mobile_solid=0.;
                for(int g=0;g<ng;g++)
                    mobile_solid+=coarse_mobility[g*nc+c]*mass[(ng+g)*nc+c]/rho[g];
                double avail=mobile_solid/sv[nc+c],amount=fmin(avail,need);
                if(avail>0.) {
                    double fraction=amount/avail;
                    double solid=h*sv[c]+amount*sv[nc+c];
                    for(int g=0;g<ng;g++) {
                        int a=g*nc+c,b=(ng+g)*nc+c;
                        double moved=fraction*coarse_mobility[a]*mass[b];
                        mass[b]-=moved;mass[a]+=moved;
                    }
                    sv[c]=solid/(h+amount);
                }
            } else {
                for(int j=k+1;j<nl && need>0.;j++) {
                    double avail=bed_thickness_v101(ng,nc,c,j,mass,sv,rho);
                    if(avail<=0.) continue;
                    double amount=fmin(avail,need);
                    bed_packet_v101(ng,nc,c,j,k,amount/avail,mass,sv,rho); need-=amount;
                }
            }
            if(k==0 && topmode==1) sv[c]=1.-bed_porosity_v101(ng,nc,c,pormode,mass,phi,sig,sv0,svsand,svmud,mud);
        }
    }
    if(eroding) for(int k=start;k<start+nlag;k++) {
        double need=target[k*nc+c]-bed_thickness_v101(ng,nc,c,k,mass,sv,rho);
        for(int j=k+1;j<nl && need>0.;j++) {
            double h=bed_thickness_v101(ng,nc,c,j,mass,sv,rho);
            if(h<=0.) continue;
            double a=fmin(need,h); bed_packet_v101(ng,nc,c,j,k,a/h,mass,sv,rho);need-=a;
        }
    }
    bool basal_changed=false;
    for(int g=0;g<ng;g++) if(mass[((nl-1)*ng+g)*nc+c]!=initial[((nl-1)*ng+g)*nc+c]) basal_changed=true;
    if(nl>1 && basemode!=1 && (basal_changed || (basemode==3 && eroding))) {
        int k=nl-1, ref=k;
        if(basemode==3) for(int j=k-1;j>=0;j--) {
            if(bed_thickness_v101(ng,nc,c,j,mass,sv,rho)>0.) {ref=j;break;}
        }
        double h=bed_thickness_v101(ng,nc,c,k,mass,sv,rho);
        double h0=bed_thickness_v101(ng,nc,c,k,initial,initial_sv,rho);
        double href=basemode==3?bed_thickness_v101(ng,nc,c,ref,mass,sv,rho):h0;
        const double* copied=mass;
        double copied_sv=sv[ref*nc+c];
        if(basemode==3 && eroding && ref<start) {
            copied=receiver;copied_sv=receiversv[ref*nc+c];
            href=bed_thickness_v101(ng,nc,c,ref,receiver,receiversv,rho);
        }
        for(int g=0;g<ng;g++) {
            int i=(k*ng+g)*nc+c; double before=mass[i], after=before;
            if(basemode==2) after=h0>0.?initial[i]*h/h0:0.;
            if(basemode==3 && href>0.) after=copied[(ref*ng+g)*nc+c]*h/href;
            if(basemode==4) after=initial[i];
            if(basemode==5) after=h>0.?before*h0/h:initial[i];
            mass[i]=after; source[g*nc+c]=after-before;
        }
        if(basemode==2 || basemode==4) sv[k*nc+c]=initial_sv[k*nc+c];
        if(basemode==3 && href>0.) sv[k*nc+c]=copied_sv;
    }
    // General vertical bed mixing: conservative backward Euler tridiagonal solve.
    // No water-column Schmidt coefficient and no FM ALDiff are consumed.
    if(nd>0 && dt>0. && nl>1) {
        // compact nonempty layers first, leaving zero inventory at the bottom
        int count=0;
        for(int k=0;k<nl;k++) if(bed_thickness_v101(ng,nc,c,k,mass,sv,rho)>0.) {
            if(k!=count) {
                sv[count*nc+c]=sv[k*nc+c];
                for(int g=0;g<ng;g++) mass[(count*ng+g)*nc+c]=mass[(k*ng+g)*nc+c];
            }
            count++;
        }
        for(int k=count;k<nl;k++) {
            sv[k*nc+c]=1.; for(int g=0;g<ng;g++) mass[(k*ng+g)*nc+c]=0.;
        }
        // work = layer arrays [h, lower, diag, upper, solid-volume RHS]
        double z=0.;
        for(int k=0;k<count;k++) {
            double h=bed_thickness_v101(ng,nc,c,k,mass,sv,rho);
            work[(0*nl+k)*nc+c]=h; work[(1*nl+k)*nc+c]=0.;
            work[(2*nl+k)*nc+c]=1.; work[(3*nl+k)*nc+c]=0.;
            work[(4*nl+k)*nc+c]=h*sv[k*nc+c];
            if(k>0) {
                int d=0; while(d<nd-1 && zd[d]<z) d++;
                double diffusion=kd[d*nc+c];
                if(d>0 && z<zd[nd-1]) diffusion=kd[(d-1)*nc+c]+(z-zd[d-1])/(zd[d]-zd[d-1])*(diffusion-kd[(d-1)*nc+c]);
                double prev=work[(k-1)*nc+c];
                double f=.5*dt*diffusion/(prev+h);
                work[(2*nl+k-1)*nc+c]+=f/prev;
                work[(3*nl+k-1)*nc+c]=-f/h;
                work[(1*nl+k)*nc+c]=-f/prev;
                work[(2*nl+k)*nc+c]+=f/h;
            }
            z+=h;
        }
        for(int k=0;k<count;k++) {
            if(k>0) {
                double a=work[(1*nl+k)*nc+c];
                work[(2*nl+k)*nc+c]-=a*work[(3*nl+k-1)*nc+c];
                work[(4*nl+k)*nc+c]-=a*work[(4*nl+k-1)*nc+c];
                for(int g=0;g<ng;g++) mass[(k*ng+g)*nc+c]-=a*mass[((k-1)*ng+g)*nc+c];
            }
            double b=work[(2*nl+k)*nc+c];
            work[(3*nl+k)*nc+c]/=b; work[(4*nl+k)*nc+c]/=b;
            for(int g=0;g<ng;g++) mass[(k*ng+g)*nc+c]/=b;
        }
        for(int k=count-2;k>=0;k--) {
            double a=work[(3*nl+k)*nc+c];
            work[(4*nl+k)*nc+c]-=a*work[(4*nl+k+1)*nc+c];
            for(int g=0;g<ng;g++) mass[(k*ng+g)*nc+c]-=a*mass[((k+1)*ng+g)*nc+c];
        }
        for(int k=0;k<count;k++) {
            double s=work[(4*nl+k)*nc+c]/work[k*nc+c];
            // Roundoff in a fully dense column is not physical compaction.
            if(s>1. && s<=1.+1.7763568394002505e-15) s=1.;
            sv[k*nc+c]=s;
        }
    }
    for(int k=0;k<nl;k++) for(int g=0;g<ng;g++) {
        double x=mass[(k*ng+g)*nc+c];
        if(!(x>=0.) || !isfinite(x) || !(sv[k*nc+c]>0. && sv[k*nc+c]<=1.)) status[c]=2;
    }
}
