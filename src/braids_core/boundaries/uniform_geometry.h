// Preparation-only recognition of exact axis-aligned rectangle rings.
// No clipping, coordinate movement, area recomputation or topology decisions.
#include <cmath>
#include <omp.h>
extern "C" __declspec(dllexport) void mobile_threads(int n){omp_set_dynamic(0);omp_set_num_threads(n);}
extern "C" __declspec(dllexport) void mesh_rectangles(int n,const int*offset,
 const double*xy,const int*holes,int*out){
 #pragma omp parallel for schedule(static)
 for(int i=0;i<n;++i){
  out[i]=0;int k=offset[i];if(holes[i] || offset[i+1]-k!=5)continue;
  const double *p=xy+2*k;
  bool finite=true;for(int j=0;j<10;++j)finite=finite && std::isfinite(p[j]);
  if(!finite)continue;
  bool closed=p[0]==p[8] && p[1]==p[9];
  bool horizontal=p[1]==p[3] && p[2]==p[4] && p[5]==p[7] && p[6]==p[0];
  bool vertical=p[0]==p[2] && p[3]==p[5] && p[4]==p[6] && p[7]==p[1];
  if(!closed || (!horizontal && !vertical) || p[0]==p[4] || p[1]==p[5])continue;
  // Return orientation without multiplying potentially large/small edges.
  bool positive=horizontal ? ((p[2]>p[0])==(p[5]>p[3]))
                           : ((p[4]>p[2])!=(p[3]>p[1]));
  out[i]=positive?1:-1;
 }
}
