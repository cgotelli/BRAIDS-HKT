"""Isolated paired wall reconstruction experiments; not a public option.

The tangential extension is linear, not no-slip/even tangential reflection.
The optional normal fit adds an impermeability observation to the existing
fluid least-squares fit, with the mean fluid-face length as its geometric
weight. A rank-one inverse update handles multiple wall normals consistently.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('tangent','normal','lswall','normal_ls')


def native_source(code,variant):
    if variant not in VARIANTS:raise ValueError(variant)
    if variant.endswith('ls') or variant=='lswall':
        if variant=='normal_ls':code=native_source(code,'normal')
        helper='''
 subroutine corner167_linear_walls() bind(C)
  integer::c,k,f,side
  logical::wall
  do c=1,nc
   wall=.false.
   do k=offset(c),offset(c+1)-1
    f=faces(k)
    if(neighbour(f)==0.and.bid(f)==0)wall=.true.
   enddo
   if(.not.wall)cycle
   wall_stencil_ids(:,:,c)=0
   do k=offset(c),offset(c+1)-1
    f=faces(k);side=1
    if(owner(f)/=c)side=2
    weno_cells(:,side,f)=0
   enddo
  enddo
 end subroutine
'''
        end=code.rindex('\nend module');return code[:end]+helper+code[end:]
    marker='   call braids_wall_velocity_bounds(c,v,vlo,vhi)'
    assert code.count(marker)==3
    code=code.replace(marker,marker+'\n   call corner167_fit(c,v,grad,vlo,vhi)')
    normal='''
  inv=0.d0
  inv(1,1)=a22/det;inv(2,2)=a11/det;inv(1,2)=-a12/det;inv(2,1)=-a12/det
  inv(3:4,3:4)=inv(1:2,1:2)
  gg=[grad(2,1,c),grad(2,2,c),grad(3,1,c),grad(3,2,c)]
  count=0
  do k=offset(c),offset(c+1)-1
   f=faces(k)
   if(neighbour(f)>0)count=count+1
  enddo
  do k=offset(c),offset(c+1)-1
   f=faces(k)
   if(neighbour(f)/=0.or.bid(f)>0)cycle
   n=normal(:,f);r=2.d0*dot_product(mid(:,f)-xy(:,c),n)*n
   if(sum(r*r)<=1.d-24)cycle
   w=(a11+a22)/(max(1,count)*sum(r*r))
   vec=[n(1)*r(1),n(1)*r(2),n(2)*r(1),n(2)*r(2)]
   vv=matmul(inv,vec);den=1.d0+w*dot_product(vec,vv)
   res=-2.d0*dot_product(v(2:3,c),n)-dot_product(vec,gg)
   gg=gg+(w*res/den)*vv
   do i=1,4
    do j=1,4
     inv(i,j)=inv(i,j)-(w/den)*vv(i)*vv(j)
    enddo
   enddo
  enddo
  grad(2,:,c)=gg(1:2);grad(3,:,c)=gg(3:4)
''' if variant=='normal' else ''
    helper='''
 subroutine corner167_fit(c,v,grad,vlo,vhi)
  integer,intent(in)::c
  real(8),intent(in)::v(nprimitive,nc)
  real(8),intent(inout)::grad(nprimitive,2,nc),vlo(nprimitive,nc),vhi(nprimitive,nc)
  integer::k,f,i,j,count
  real(8)::n(2),r(2),dv(2),ghost(2),a11,a12,a22,det,inv(4,4),gg(4),vec(4),vv(4),w,den,res
  if(v(1,c)-v(bed_primitive_index(),c)<=dry_tolerance)return
  a11=hydraulic_ls_matrix(1,c);a12=hydraulic_ls_matrix(2,c)
  a22=hydraulic_ls_matrix(3,c);det=hydraulic_ls_matrix(4,c)
'''+normal+'''
  do k=offset(c),offset(c+1)-1
   f=faces(k)
   if(neighbour(f)/=0.or.bid(f)>0)cycle
   n=normal(:,f);r=2.d0*dot_product(mid(:,f)-xy(:,c),n)*n
   dv=matmul(grad(2:3,:,c),r);dv=dv-dot_product(dv,n)*n
   ghost=v(2:3,c)+dv-2.d0*dot_product(v(2:3,c),n)*n
   vlo(2:3,c)=min(vlo(2:3,c),ghost);vhi(2:3,c)=max(vhi(2:3,c),ghost)
  enddo
 end subroutine
'''
    end=code.rindex('\nend module')
    return code[:end]+helper+code[end:]


def cuda_source(code,variant):
    if variant not in VARIANTS:raise ValueError(variant)
    if variant=='lswall':return code
    if variant=='normal_ls':return cuda_source(code,'normal')
    marker='            const double h=centre[0]-centre[3];'
    assert code.count(marker)==1
    normal='''
                double inv[4][4]={{a22/det,-a12/det,0.,0.},{-a12/det,a11/det,0.,0.},
                                  {0.,0.,a22/det,-a12/det},{0.,0.,-a12/det,a11/det}};
                double gg[4]={gx[1],gy[1],gx[2],gy[2]};int count=0;
                for(int k=offset[c];k<offset[c+1];++k)if(neighbour[faces[k]]>=0)++count;
                for(int k=offset[c];k<offset[c+1];++k) {
                    const int f=faces[k];if(neighbour[f]>=0 || bid[f]>=0)continue;
                    const double nx=normal[f],ny=normal[nf+f];
                    const double d=2.*((mid[f]-xy[c])*nx+(mid[nf+f]-xy[nc+c])*ny),rx=d*nx,ry=d*ny;
                    if(rx*rx+ry*ry<=1.e-24)continue;
                    const double w=(a11+a22)/(fmax(1.,(double)count)*(rx*rx+ry*ry));
                    double vec[4]={nx*rx,nx*ry,ny*rx,ny*ry},vv[4]={0.,0.,0.,0.};
                    for(int i=0;i<4;++i)for(int j=0;j<4;++j)vv[i]+=inv[i][j]*vec[j];
                    double den=1.,res=-2.*(centre[1]*nx+centre[2]*ny);
                    for(int i=0;i<4;++i){den+=w*vec[i]*vv[i];res-=vec[i]*gg[i];}
                    for(int i=0;i<4;++i)gg[i]+=(w*res/den)*vv[i];
                    for(int i=0;i<4;++i)for(int j=0;j<4;++j)inv[i][j]-=(w/den)*vv[i]*vv[j];
                }
                gx[1]=gg[0];gy[1]=gg[1];gx[2]=gg[2];gy[2]=gg[3];
''' if variant=='normal' else ''
    extra='''
            if(q[c]>dry) {
'''+normal+'''
                for(int k=offset[c];k<offset[c+1];++k) {
                    const int f=faces[k];if(neighbour[f]>=0 || bid[f]>=0)continue;
                    const double nx=normal[f],ny=normal[nf+f];
                    const double d=2.*((mid[f]-xy[c])*nx+(mid[nf+f]-xy[nc+c])*ny),rx=d*nx,ry=d*ny;
                    double vx=gx[1]*rx+gy[1]*ry,vy=gx[2]*rx+gy[2]*ry;
                    const double vn=vx*nx+vy*ny,un=centre[1]*nx+centre[2]*ny;
                    vx=centre[1]+(vx-vn*nx)-2.*un*nx;vy=centre[2]+(vy-vn*ny)-2.*un*ny;
                    lo[1]=fmin(lo[1],vx);hi[1]=fmax(hi[1],vx);
                    lo[2]=fmin(lo[2],vy);hi[2]=fmax(hi[2],vy);
                }
            }
'''
    return code.replace(marker,extra+marker)


def build_native(variant):
    from check_bedload161 import sha,save
    table=json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text())
    rows=[v for k,v in table.items() if 'shoreline_hydraulic_probe.build' in k
          and 'call braids_wall_velocity_bounds' in
          (ROOT/'desktop/runtime'/Path(v['native']['file']).parent/'shoreline_probe.f90').read_text()]
    assert len(rows)==1
    row=rows[0]
    original=ROOT/'desktop/runtime'/Path(row['native']['file']).parent/'shoreline_probe.f90'
    assert sha(original)==row['identity']['source_sha256']
    code=native_source(original.read_text(),variant);digest=hashlib.sha256(code.encode()).hexdigest()
    out=ROOT/'validation/_corner167/build'/digest[:20];target=out/'hydraulics.dll'
    if not target.exists():
        out.mkdir(parents=True,exist_ok=True);source=out/'corner.f90';source.write_text(code)
        cmd=list(row['identity']['command']);old=next(v for v in cmd if v.endswith('shoreline_probe.f90'))
        cmd[cmd.index(old)]=str(source);cmd[cmd.index('-o')+1]=str(target)
        result=subprocess.run(cmd,cwd=out,capture_output=True,text=True,timeout=180)
        (out/'BUILD.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(result.stderr)
        save(out/'IDENTITY.json',dict(parent=row,command=cmd,source_sha256=digest,
             library_sha256=sha(target),Delft3D_modified=False,production_modified=False,method=variant))
    identity=json.loads((out/'IDENTITY.json').read_text());assert sha(target)==identity['library_sha256']
    return target,identity,row['identity']['command'][0]


def install(backend,variant):
    from braids_core.sediment import wall_bed_hydraulics as wall
    from validation.sediment.river_morphodynamics import shoreline_hydraulic_probe as probe
    if backend=='cpu':
        built=build_native(variant);probe.build=lambda mode:built
    else:
        previous=wall.cuda_source;wall.cuda_source=lambda code:cuda_source(previous(code),variant)


def configure(owner,variant):
    """Only research constructors call this, before the first hydraulic step."""
    if variant not in ('lswall','normal_ls'):return
    if owner.backend=='cpu':
        f=owner.hydraulics.lib.corner167_linear_walls;f.argtypes=[];f.restype=None;f()
    else:
        import numpy as np
        from braids_core.sediment.bed_columns_v101 import host
        g=owner.hydraulics.reconstruction.geometry;xp=owner.xp
        cells=np.unique(owner.data['owner'][(owner.data['neighbour']<0)&(owner.data['bid']<0)])
        ids=host(g['wall_stencil_ids']).copy();ids[:,:,cells]=-1;g['wall_stencil_ids'][:]=xp.asarray(ids)
        weno=host(g['weno_cells']).copy()
        for side,key in enumerate(('owner','neighbour')):
            faces=np.isin(owner.data[key],cells);weno[:,side,faces]=-1
        g['weno_cells'][:]=xp.asarray(weno)
