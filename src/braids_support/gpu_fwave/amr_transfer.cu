// Isolated native-semantic, four-child 2:1 AMR state transfer.
// One GPU thread owns one parent transaction. No topology or physics is
// silently synthesized: suspended and signed secondary inventories are
// conservatively prolonged. No front state, selection or subcycling.
__device__ inline double maximum(double a, double b) { return a > b ? a : b; }

__device__ void positive_transfer(double value[4], const double weight[4], double target) {
    if (target <= 0.0) {
        for (int k=0;k<4;++k) value[k]=0.0;
        return;
    }
    double total=0.0, area=0.0;
    for (int k=0;k<4;++k) {
        value[k]=maximum(value[k],0.0);
        total+=value[k]*weight[k];
        area+=weight[k];
    }
    if (total>0.0) {
        double factor=target/total;
        for (int k=0;k<4;++k) value[k]*=factor;
    } else {
        for (int k=0;k<4;++k) value[k]=target/area;
    }
    int largest=0;
    for (int k=1;k<4;++k)
        if (value[k]*weight[k]>value[largest]*weight[largest]) largest=k;
    double sum=0.0;
    for (int k=0;k<4;++k) sum+=value[k]*weight[k];
    double residual=target-sum;
    value[largest]=maximum(0.0,value[largest]+residual/weight[largest]);
}

__device__ void fill_water(double volume, const double bed[4], const double weight[4], double depth[4]) {
    if (volume<=0.0) {
        for (int k=0;k<4;++k) depth[k]=0.0;
        return;
    }
    int order[4]={0,1,2,3};
    for (int k=1;k<4;++k) {
        int key=order[k],j=k-1;
        while (j>=0 && bed[order[j]]>bed[key]) {
            order[j+1]=order[j];--j;
        }
        order[j+1]=key;
    }
    double wet_area=0.0,moment=0.0,stage=0.0;
    for (int k=0;k<4;++k) {
        int j=order[k];
        wet_area+=weight[j];
        moment+=weight[j]*bed[j];
        stage=(volume+moment)/wet_area;
        if (k==3 || stage<=bed[order[k+1]]) break;
    }
    for (int k=0;k<4;++k) depth[k]=maximum(stage-bed[k],0.0);
    int largest=0;
    for (int k=1;k<4;++k)
        if (depth[k]*weight[k]>depth[largest]*weight[largest]) largest=k;
    double total=0.0;
    for (int k=0;k<4;++k) total+=depth[k]*weight[k];
    depth[largest]=maximum(0.0,depth[largest]+(volume-total)/weight[largest]);
}

__device__ int signed_water_transfer(double parent_mass,double parent_depth,
    const double depth[4],const double weight[4],const double raw[4],double dry,
    double output[4]) {
    const double eps=2.2204460492503131e-16;
    if(parent_depth<=dry) {
        double tolerance=512.0*eps*maximum(1.0,fabs(parent_mass));
        if(fabs(parent_mass)>tolerance)return 1;
        for(int k=0;k<4;++k)output[k]=0.0;
        return 0;
    }
    double parent_intensity=parent_mass/parent_depth;
    double lower=parent_intensity,upper=parent_intensity;
    for(int k=0;k<4;++k) {
        if(depth[k]<=dry)continue;
        lower=fmin(lower,raw[k]);upper=fmax(upper,raw[k]);
    }
    double intensity[4],capacity[4],wet_integral=0.0,got=0.0,absolute=0.0;
    for(int k=0;k<4;++k) {
        wet_integral+=depth[k]*weight[k];
        if(depth[k]<=dry) { intensity[k]=0.0;output[k]=0.0; }
        else {
            intensity[k]=fmax(lower,fmin(upper,raw[k]));
            output[k]=depth[k]*intensity[k];
        }
        got+=output[k]*weight[k];absolute+=fabs(output[k]*weight[k]);
    }
    double target=parent_intensity*wet_integral;
    double residual=target-got;
    double tolerance=1024.0*eps*maximum(1.0,maximum(fabs(target),absolute));
    if(fabs(residual)<=tolerance) {
        int largest=0;
        for(int k=1;k<4;++k)
            if(depth[k]*weight[k]>depth[largest]*weight[largest])largest=k;
        if(depth[largest]>dry)output[largest]+=residual/weight[largest];
        return 0;
    }
    double total_capacity=0.0;
    for(int k=0;k<4;++k) {
        capacity[k]=weight[k]*depth[k]*maximum(
            residual>0.0 ? upper-intensity[k] : intensity[k]-lower,0.0);
        total_capacity+=capacity[k];
    }
    if(total_capacity+tolerance<fabs(residual))return 2;
    if(total_capacity<=0.0)return 3;
    for(int k=0;k<4;++k) {
        if(capacity[k]<=0.0)continue;
        double share=residual*capacity[k]/total_capacity;
        output[k]+=share/weight[k];
    }
    got=0.0;for(int k=0;k<4;++k)got+=output[k]*weight[k];
    residual=target-got;
    int largest=0;
    for(int k=1;k<4;++k)if(capacity[k]>capacity[largest])largest=k;
    output[largest]+=residual/weight[largest];
    got=0.0;for(int k=0;k<4;++k)got+=output[k]*weight[k];
    return fabs(target-got)>4.0*tolerance ? 4 : 0;
}

extern "C" __global__ void amr_restrict_native_rows(
    const double* child, const double* weight, int parents, int states,
    double* parent, int* status) {
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    if (p>=parents) return;
    double w[4],total=0.0;
    for (int k=0;k<4;++k) {
        w[k]=weight[p*4+k];
        if (!(w[k]>0.0) || !isfinite(w[k])) { status[p]=1;return; }
        total+=w[k];
    }
    for (int j=0;j<states;++j) {
        double sum=0.0;
        for (int k=0;k<4;++k) {
            double value=child[(p*4+k)*states+j];
            if (!isfinite(value)) { status[p]=2;return; }
            sum+=w[k]*value;
        }
        parent[p*states+j]=sum/total;
    }
    status[p]=0;
}

extern "C" __global__ void amr_prolong_native_dry_bed(
    const double* parent, const double* cached, const double* weight,
    const double* parent_area, const double* rock, const double* position,
    const double* gradient, const double* bulk, int parents, int grains,
    int layers, int states, int primitives, int secondary, double dry,
    double* trial, int* status) {
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    if (p>=parents) return;
    int first=p*4*states, pb=p*states;
    int physical_states=states-secondary;
    if(states!=3+grains*(1+layers)+secondary ||
       primitives!=4+grains+secondary) {status[p]=8;return;}
    double w[4],bed[4],depth[4],value[4];
    double total_area=0.0;
    for (int k=0;k<4;++k) {
        w[k]=weight[p*4+k];
        if (!(w[k]>0.0) || !isfinite(w[k])) { status[p]=1;return; }
        total_area+=w[k];
    }
    double area=parent_area[p];
    if (!(area>0.0) || !isfinite(area) || fabs(area-total_area)>1e-12*area ||
        !(parent[pb]>=0.0) || !isfinite(parent[pb])) { status[p]=2;return; }
    // Native suspended rows are classwise nonnegative water-column mass.
    for (int g=0;g<grains;++g) {
        double mass=parent[pb+3+g];
        if (!(mass>=0.0) || !isfinite(mass) ||
            (parent[pb]<=dry && mass>0.0)) { status[p]=3;return; }
        for (int k=0;k<4;++k)
            if (!(cached[first+k*states+3+g]>=0.0) ||
                !isfinite(cached[first+k*states+3+g])) { status[p]=3;return; }
    }
    if (parent[pb]==0.0 && (parent[pb+1]!=0.0 || parent[pb+2]!=0.0)) {
        status[p]=4;return;
    }
    for (int j=3+grains;j<physical_states;++j) {
        double mean=0.0;
        for (int k=0;k<4;++k) {
            double prior=cached[first+k*states+j];
            if (!(prior>=0.0) || !isfinite(prior)) { status[p]=5;return; }
            mean+=prior*w[k];
        }
        mean/=total_area;
        double target=parent[pb+j];
        if (!(target>=0.0) || !isfinite(target)) { status[p]=5;return; }
        for (int k=0;k<4;++k) value[k]=cached[first+k*states+j]+target-mean;
        positive_transfer(value,w,area*target);
        for (int k=0;k<4;++k) trial[first+k*states+j]=value[k];
    }
    // Native amr_node_bed on a front-free cell, followed by the stage-gradient
    // correction used by amr_prolong_node before the water-fill solve.
    for (int k=0;k<4;++k) {
        bed[k]=rock[p*4+k];
        for (int l=0;l<layers;++l)
            for (int g=0;g<grains;++g)
                bed[k]+=trial[first+k*states+3+grains+l*grains+g]/bulk[g];
        double x=position[(p*4+k)*2],y=position[(p*4+k)*2+1];
        bed[k]-=gradient[(p*primitives)*2]*x+gradient[(p*primitives)*2+1]*y;
    }
    double water=area*parent[pb];
    fill_water(water,bed,w,depth);
    for (int k=0;k<4;++k) trial[first+k*states]=depth[k];
    for (int j=1;j<=2;++j) {
        double velocity=parent[pb]>dry ? parent[pb+j]/parent[pb] : 0.0;
        double sum=0.0;
        for (int k=0;k<4;++k) {
            double x=position[(p*4+k)*2],y=position[(p*4+k)*2+1];
            value[k]=depth[k]*(velocity+gradient[(p*primitives+j)*2]*x+
                                       gradient[(p*primitives+j)*2+1]*y);
            sum+=value[k]*w[k];
        }
        if (water>0.0) {
            double correction=(area*parent[pb+j]-sum)/water;
            for (int k=0;k<4;++k) value[k]+=depth[k]*correction;
        }
        for (int k=0;k<4;++k) trial[first+k*states+j]=value[k];
    }
    for (int g=0;g<grains;++g) {
        int si=3+g;
        double concentration=parent[pb]>dry ? parent[pb+si]/parent[pb] : 0.0;
        for (int k=0;k<4;++k) {
            double x=position[(p*4+k)*2],y=position[(p*4+k)*2+1];
            double raw=concentration+gradient[(p*primitives+si)*2]*x+
                                     gradient[(p*primitives+si)*2+1]*y;
            value[k]=depth[k]*maximum(0.0,raw);
        }
        positive_transfer(value,w,area*parent[pb+si]);
        for (int k=0;k<4;++k) trial[first+k*states+si]=value[k];
    }
    for (int k=0;k<4;++k)
        for (int j=0;j<states;++j)
            if (!isfinite(trial[first+k*states+j])) { status[p]=6;return; }
    status[p]=0;
}

extern "C" __global__ void amr_prolong_signed_secondary(
    const double* parent,const double* weight,const double* position,
    const double* gradient,int parents,int states,int primitives,int grains,
    double dry,double* trial,int* status) {
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    if(p>=parents || status[p]!=0)return;
    int first=p*4*states,pb=p*states,physical_states=states-1;
    double parent_mass=parent[pb+physical_states];
    if(!isfinite(parent_mass)) {status[p]=7;return;}
    double depth[4],w[4],raw[4],signed_mass[4];
    double intensity=parent[pb]>dry ? parent_mass/parent[pb] : 0.0;
    int secondary_primitive=3+grains;
    for(int k=0;k<4;++k) {
        depth[k]=trial[first+k*states];w[k]=weight[p*4+k];
        double x=position[(p*4+k)*2],y=position[(p*4+k)*2+1];
        raw[k]=intensity+gradient[(p*primitives+secondary_primitive)*2]*x+
                         gradient[(p*primitives+secondary_primitive)*2+1]*y;
        if(!isfinite(raw[k])) {status[p]=7;return;}
    }
    int result=signed_water_transfer(parent_mass,parent[pb],depth,w,raw,dry,signed_mass);
    if(result) {status[p]=7;return;}
    for(int k=0;k<4;++k) {
        if(!isfinite(signed_mass[k])) {status[p]=7;return;}
        trial[first+k*states+physical_states]=signed_mass[k];
    }
}
