"""Slip-wall ghost bounds for the hydraulic velocity reconstruction.

Fluid-centre extrema alone exclude the zero-normal-velocity wall trace of an
affine field. Include its reflected cell state, as in the reflective Riemann
boundary, without relaxing surface, terrain, concentration or interior bounds.
This does not impose no-slip or replace the wall pressure flux.
"""


def native_source(code):
    marker = '   grad(:,2,c)=(a11*b(:,2)-a12*b(:,1))/det'
    if code.count(marker) != 3:
        raise ValueError('Expected all three native hydraulic reconstruction routes')
    code = code.replace(marker, marker+'\n   call braids_wall_velocity_bounds(c,v,vlo,vhi)')
    helper = '''
 subroutine braids_wall_velocity_bounds(c,v,vlo,vhi)
  integer,intent(in)::c
  real(8),intent(in)::v(nprimitive,nc)
  real(8),intent(inout)::vlo(nprimitive,nc),vhi(nprimitive,nc)
  integer::k,f
  real(8)::ghost(2),un
  if(v(1,c)-v(bed_primitive_index(),c)<=dry_tolerance)return
  do k=offset(c),offset(c+1)-1
   f=faces(k)
   if(neighbour(f)/=0.or.bid(f)>0)cycle
   un=dot_product(v(2:3,c),normal(:,f))
   ghost=v(2:3,c)-2.d0*un*normal(:,f)
   vlo(2:3,c)=min(vlo(2:3,c),ghost)
   vhi(2:3,c)=max(vhi(2:3,c),ghost)
  enddo
 end subroutine
'''
    end = code.rindex('\nend module')
    return code[:end]+helper+code[end:]


def cuda_source(code):
    marker = '        if(det>1.e-12*((a11+a22)*(a11+a22))) {'
    if code.count(marker) != 1:
        raise ValueError('Expected one CUDA hydraulic reconstruction loop')
    extra = '''        if(q[c]>dry)for(int k=offset[c];k<offset[c+1];++k) {
            const int f=faces[k];if(neighbour[f]>=0 || bid[f]>=0)continue;
            const double nx=normal[f],ny=normal[nf+f],un=centre[1]*nx+centre[2]*ny;
            const double gx=centre[1]-2.*un*nx,gy=centre[2]-2.*un*ny;
            lo[1]=fmin(lo[1],gx);hi[1]=fmax(hi[1],gx);
            lo[2]=fmin(lo[2],gy);hi[2]=fmax(hi[2],gy);
        }
'''
    return code.replace(marker, extra+marker)
