"""Retain resolved hydraulic gradients in rank-deficient neighbour geometry.

The existing full-rank inverse and downstream limiting are unchanged. A
rank-one positive-semidefinite normal matrix uses A / trace(A)^2; an empty
stencil retains zero slope. No unresolved transverse derivative is invented.
"""
METHOD='rank_aware_hydraulic_least_squares_v1'


def native_source(code):
    old='''   if(det<=1.d-12*(a11+a22)**2) cycle
   grad(:,1,c)=(a22*b(:,1)-a12*b(:,2))/det
   grad(:,2,c)=(a11*b(:,2)-a12*b(:,1))/det'''
    new='''   if(a11+a22<=0.d0) cycle
   if(det>1.d-12*(a11+a22)**2) then
    grad(:,1,c)=(a22*b(:,1)-a12*b(:,2))/det
    grad(:,2,c)=(a11*b(:,2)-a12*b(:,1))/det
   else
    ! Minimum-norm rank-one LS: retain the resolved spatial direction.
    grad(:,1,c)=(a11*b(:,1)+a12*b(:,2))/(a11+a22)**2
    grad(:,2,c)=(a12*b(:,1)+a22*b(:,2))/(a11+a22)**2
   endif'''
    count=code.count(old)
    if count!=3:raise ValueError(f'Expected three native hydraulic LS sites, found {count}')
    return code.replace(old,new)


def cuda_source(code):
    old='''        if(det>1.e-12*((a11+a22)*(a11+a22))) {
            for(int m=0;m<4;++m) {gx[m]=(a22*bx[m]-a12*by[m])/det;gy[m]=(a11*by[m]-a12*bx[m])/det;}'''
    new='''        if(a11+a22>0.) {
            for(int m=0;m<4;++m) {
                if(det>1.e-12*((a11+a22)*(a11+a22))) {
                    gx[m]=(a22*bx[m]-a12*by[m])/det;gy[m]=(a11*by[m]-a12*bx[m])/det;
                } else {
                    gx[m]=(a11*bx[m]+a12*by[m])/((a11+a22)*(a11+a22));
                    gy[m]=(a12*bx[m]+a22*by[m])/((a11+a22)*(a11+a22));
                }
            }'''
    if code.count(old)!=1:raise ValueError('Expected one CUDA hydraulic LS site')
    return code.replace(old,new)
