#ifndef BRAIDS_PLANFORM_V105_H
#define BRAIDS_PLANFORM_V105_H

#ifndef BRAIDS_HD
#define BRAIDS_HD inline
#endif

/*
 * Q8a equations shared by the OpenMP and CUDA implementations.
 *
 * Sign convention: BRAIDS stores bed elevation positive upward, whereas
 * Delft3D4 FLOW's dps is positive downward.  The helpers below make the sign
 * conversion explicit.  No Delft3D source is compiled or copied into BRAIDS.
 */

BRAIDS_HD double q8a_bed_v105(int c,int nc,int ng,int nl,const double* layers,
                              const double* bulk,const double* rock) {
    double z=rock[c];
    for(int l=0;l<nl;l++) for(int g=0;g<ng;g++)
        z += layers[(l*ng+g)*nc+c]/bulk[g];
    return z;
}

BRAIDS_HD double q8a_depth_slope_v105(double bed0,double bed1,double distance) {
    /* (dps_1-dps_0)/distance with dps=-bed. */
    return (bed0-bed1)/distance;
}

BRAIDS_HD double q8a_wet_flux_v105(double area0,double area1,double bed0,
    double bed1,double distance,double slope_component,double slope_magnitude,
    double wet_slope,double avalanche_time) {
    if(!(slope_magnitude>wet_slope) || slope_magnitude<=0.) return 0.;
    const double dps_difference=bed0-bed1;
    return area0*(dps_difference+wet_slope*(slope_component/slope_magnitude)*distance)/
           ((1.+area0/area1)*avalanche_time);
}

BRAIDS_HD double q8a_dry_bank_theta_v105(double water_depth,double thetsd,
    double hmaxth,double sedthr) {
    double value=thetsd;
    if(hmaxth>sedthr) value=(water_depth-sedthr)/(hmaxth-sedthr)*thetsd;
    if(value>thetsd) value=thetsd;
    if(value<0.) value=0.;
    return value;
}

BRAIDS_HD double q8a_repose_v105(double depth0,double depth1,double switch_depth,
                                 double wet_slope,double dry_slope) {
    return (depth0>switch_depth && depth1>switch_depth)?wet_slope:dry_slope;
}

BRAIDS_HD double q8a_transfer_scale_v105(double available,double requested) {
    if(requested<=0.) return 1.;
    if(available<=0.) return 0.;
    const double scale=available/requested;
    return scale<1.?scale*(1.-64.*2.2204460492503131e-16):1.;
}

#endif
