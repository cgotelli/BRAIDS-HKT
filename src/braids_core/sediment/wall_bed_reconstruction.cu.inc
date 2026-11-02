            // Same face-distance secant limiter as braids_wall_bed_factor.
            // Only fully wet, closed-wall cells; open ends retain their BCs.
            {
                bool eligible=h>20.*dry,wall=false;double candidate=1.;
                for(int k=offset[c];k<offset[c+1] && eligible;++k) {
                    const int f=faces[k],n=signs[k]<0?owner[f]:neighbour[f];
                    if(n<0) {if(bid[f]>=0)eligible=false;else wall=true;continue;}
                    if(v[n]-v[3*nc+n]<=20.*dry) {eligible=false;break;}
                    const double dn=ls_delta[k]*normal[f]+ls_delta[nadj+k]*normal[nf+f];
                    if(fabs(dn)<=2.2204460492503131e-16){eligible=false;break;}
                    const double fraction=((mid[f]-xy[c])*normal[f]+(mid[nf+f]-xy[nc+c])*normal[nf+f])/dn;
                    if(fraction<=0. || fraction>=1.){eligible=false;break;}
                    const double pred=fraction*(gx[3]*ls_delta[k]+gy[3]*ls_delta[nadj+k]);
                    const double delta=v[3*nc+n]-centre[3];
                    const double eps=64.*2.2204460492503131e-16*fmax(1.,fmax(fabs(centre[3]),fabs(v[3*nc+n])));
                    if(fabs(pred)>eps)candidate=fabs(delta)<=eps?0.:fmin(candidate,fmax(0.,(delta+copysign(eps,pred))/pred));
                }
                if(eligible && wall) {
                    for(int k=offset[c];k<offset[c+1];++k) {
                        const int f=faces[k];const double rx=mid[f]-xy[c],ry=mid[nf+f]-xy[nc+c];
                        const double eta=centre[0]+theta[0]*(gx[0]*rx+gy[0]*ry);
                        const double zb=centre[3]+candidate*(gx[3]*rx+gy[3]*ry);
                        if(eta-zb<=20.*dry || fabs(zb-centre[3])>=h) {eligible=false;break;}
                    }
                    if(eligible)theta[3]=candidate;
                }
            }
