__device__ inline bool braids_projection_ok(double x,double y,int count,const double* ax,const double* ay,const double* b) {
 for(int i=0;i<count;++i) if(ax[i]*x+ay[i]*y>b[i]+2.e-14)return false;
 return true;
}
__device__ inline void braids_project_pair(double &x,double &y,int count,const double* ax,const double* ay,const double* b) {
 const double rx=x,ry=y;
 if(braids_projection_ok(x,y,count,ax,ay,b))return;
 double bx=0.,by=0.,best=rx*rx+ry*ry;
 for(int i=0;i<count;++i) {
  double dot=ax[i]*rx+ay[i]*ry-b[i],cx=rx-dot*ax[i],cy=ry-dot*ay[i];
  double d=(cx-rx)*(cx-rx)+(cy-ry)*(cy-ry);
  if(d<best && braids_projection_ok(cx,cy,count,ax,ay,b)){bx=cx;by=cy;best=d;}
  for(int j=0;j<i;++j) {
   const double det=ax[i]*ay[j]-ay[i]*ax[j];if(fabs(det)<1.e-12)continue;
   cx=(b[i]*ay[j]-ay[i]*b[j])/det;cy=(ax[i]*b[j]-b[i]*ax[j])/det;
   d=(cx-rx)*(cx-rx)+(cy-ry)*(cy-ry);
   if(d<best && braids_projection_ok(cx,cy,count,ax,ay,b)){bx=cx;by=cy;best=d;}
  }
 }
 x=bx;y=by;
}
// CELL_INSERT
            double px[2]={gx[0],gx[3]},py[2]={gy[0],gy[3]};
            const bool project_candidate=offset[c+1]-offset[c]<=32;
            if(project_candidate) for(int a=0;a<2;++a) {
              const int m=a?3:0;double ax[64],ay[64],b[64];int count=0;
              for(int k=offset[c];k<offset[c+1];++k) {
                const int f=faces[k];const double rx=mid[f]-xy[c],ry=mid[nf+f]-xy[nc+c],length=hypot(rx,ry);
                if(neighbour[f]<0 && bid[f]>=0 && boundary_mode[f]==2 && h>dry &&
                   fabs(centre[1]*normal[f]+centre[2]*normal[nf+f])<sqrt(grav*h) &&
                   fabs(gx[3]*rx+gy[3]*ry)>64.0*2.2204460492503131e-16*fmax(1.0,fabs(centre[3])))continue;
                if(length<=0. || (m==3 && neighbour[f]<0 && bid[f]<0))continue;
                ax[count]=rx/length;ay[count]=ry/length;b[count++]=fmax(0.,hi[m]-centre[m])/length;
                ax[count]=-rx/length;ay[count]=-ry/length;b[count++]=fmax(0.,centre[m]-lo[m])/length;
              }
              braids_project_pair(px[a],py[a],count,ax,ay,b);
            }
