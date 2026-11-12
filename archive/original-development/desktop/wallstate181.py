"""Isolated paired bounded quadratic incident state at eligible full walls."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
METHOD='bounded_quadratic_wall_incident181_research'

FORTRAN=r'''
 function face_value(c,f,v,grad,vlo,vhi) result(value)
  integer,intent(in)::c,f
  real(8),intent(in)::v(nprimitive,nc),grad(nprimitive,2,nc),vlo(nprimitive,nc),vhi(nprimitive,nc)
  real(8)::value(nprimitive),other(nprimitive),h,h1,p,p1,aa,bb,cross,theta,delta,centre,lo,hi,hf,s,poly,disp(2),width
  integer::j,t,r,k,ff
  value=base_face_value181(c,f,v,grad,vlo,vhi)
  if(scheme/=6.or.neighbour(f)/=0.or.bid(f)/=0.or.hydro_constant(c))return
  if(.not.allocated(poly181_ids))return
  r=poly181_ids(c);if(r<=0)return
  j=poly181_axis(c);t=3-j
  h=v(1,c)-v(bed_primitive_index(),c);h1=v(1,r)-v(bed_primitive_index(),r)
  if(min(h,h1)<=dry_tolerance)return
  p=h*v(j+1,c);p1=h1*v(j+1,r);width=sqrt(area(c))
  aa=3.5d0*p-.5d0*p1;bb=.75d0*p1-2.25d0*p
  cross=2*width*(h*grad(j+1,t,c)+v(j+1,c)*(grad(1,t,c)-grad(bed_primitive_index(),t,c)))
  theta=1.d0
  do k=offset(c),offset(c+1)-1
   ff=faces(k)
   if(neighbour(ff)==0.and.bid(ff)>0)return
   other=base_face_value181(c,ff,v,grad,vlo,vhi);hf=other(1)-other(bed_primitive_index())
   if(hf<=dry_tolerance)return
   disp=(mid(:,ff)-xy(:,c))/width;s=.5d0-poly181_sign(c)*disp(j)
   poly=aa*s+bb*s*s+cross*s*disp(t);delta=(poly-p)/hf;centre=p/hf
   lo=min(vlo(j+1,c),centre);hi=max(vhi(j+1,c),centre)
   if(abs(delta)<=64*epsilon(1.d0)*max(1.d0,abs(centre),abs(lo),abs(hi)))cycle
   if(delta>0)theta=min(theta,(hi-centre)/delta)
   if(delta<0)theta=min(theta,(lo-centre)/delta)
  enddo
  hf=value(1)-value(bed_primitive_index());value(j+1)=p*(1-max(0.d0,theta))/hf
 end function
 subroutine poly181_set(ids,axes,sgn) bind(C)
  use iso_c_binding
  integer(c_int),intent(in)::ids(nc),axes(nc)
  real(c_double),intent(in)::sgn(nc)
  if(allocated(poly181_ids))deallocate(poly181_ids,poly181_axis,poly181_sign)
  allocate(poly181_ids(nc),poly181_axis(nc),poly181_sign(nc))
  poly181_ids=ids+1;poly181_axis=axes+1;poly181_sign=sgn
 end subroutine
'''

CUDA=r'''
extern "C" __global__ void wallstate181(
 const double* q,const double* grad,const double* lo0,const double* hi0,const int* flat,
 const double* xy,const double* mid,const double* area,const int* owner,const int* neighbour,const int* bid,
 const int* offset,const int* faces,const int* ids,const int* axes,const double* sgn,
 long long nc,long long nf,double dry,double* fv,double* trace) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 int r=ids[c];if(r<0 || flat[c] || q[c]<=dry || q[r]<=dry)return;
 int j=axes[c],t=1-j,wf=-1;double h=q[c],p=q[(1+j)*nc+c],p1=q[(1+j)*nc+r],width=sqrt(area[c]);
 double aa=3.5*p-.5*p1,bb=.75*p1-2.25*p;
 double cross=2*width*(h*grad[((j+1)*2+t)*nc+c]+p/h*(grad[t*nc+c]-grad[(6+t)*nc+c]));
 double theta=1.;
 for(int k=offset[c];k<offset[c+1];++k) {
  int f=faces[k],side=owner[f]==c?0:1;
  if(neighbour[f]<0){if(bid[f]>=0)return;wf=f;}
  double hf=fv[side*nf+f]-fv[(6+side)*nf+f];if(hf<=dry)return;
  double ds=(mid[j*nf+f]-xy[j*nc+c])/width,dt=(mid[t*nf+f]-xy[t*nc+c])/width;
  double s=.5-sgn[c]*ds,poly=aa*s+bb*s*s+cross*s*dt,delta=(poly-p)/hf,centre=p/hf;
  double lo=fmin(lo0[(1+j)*nc+c],centre),hi=fmax(hi0[(1+j)*nc+c],centre);
  if(fabs(delta)<=64*2.2204460492503131e-16*fmax(1.,fmax(fabs(centre),fmax(fabs(lo),fabs(hi)))))continue;
  if(delta>0)theta=fmin(theta,(hi-centre)/delta);
  if(delta<0)theta=fmin(theta,(lo-centre)/delta);
 }
 if(wf<0)return;
 double hf=fv[wf]-fv[6*nf+wf],momentum=p*(1-fmax(0.,theta));
 fv[(2+2*j)*nf+wf]=momentum/hf;
 trace[(1+j)*nf+wf]=momentum;trace[(5+j)*nf+wf]=momentum;
}
'''

def native_source(code):
    marker=' logical, allocatable :: wall_triangle(:),wall_ready(:,:)';assert code.count(marker)==1
    code=code.replace(marker,marker+'\n integer,allocatable :: poly181_ids(:),poly181_axis(:)\n real(8),allocatable :: poly181_sign(:)')
    assert code.count(' function face_value(')==1
    code=code.replace(' function face_value(',' function base_face_value181(')
    at=code.rindex('\nend module');return code[:at]+FORTRAN+code[at:]

def build():
    from stage_wall172 import sha,save
    table=json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text())
    rows=[v for k,v in table.items() if 'shoreline_hydraulic_probe.build' in k and
        'call braids_wall_velocity_bounds' in (ROOT/'desktop/runtime'/Path(v['native']['file']).parent/'shoreline_probe.f90').read_text()]
    assert len(rows)==1;row=rows[0];p=ROOT/'desktop/runtime'/Path(row['native']['file']).parent/'shoreline_probe.f90'
    assert sha(p)==row['identity']['source_sha256'];code=native_source(p.read_text());digest=hashlib.sha256(code.encode()).hexdigest()
    folder=ROOT/'validation/_incident181/build'/digest[:20];target=folder/'hydraulics.dll'
    if not target.exists():
        folder.mkdir(parents=True,exist_ok=True);source=folder/'wallstate.f90';source.write_text(code)
        cmd=list(row['identity']['command']);old=next(s for s in cmd if s.endswith('shoreline_probe.f90'))
        cmd[cmd.index(old)]=str(source);cmd[cmd.index('-o')+1]=str(target)
        r=subprocess.run(cmd,cwd=folder,capture_output=True,text=True,timeout=180)
        (folder/'BUILD.log').write_text(r.stdout+r.stderr);r.check_returncode()
        save(folder/'IDENTITY.json',dict(parent=row,command=cmd,source_sha256=digest,library_sha256=sha(target),method=METHOD))
    ident=json.loads((folder/'IDENTITY.json').read_text());assert sha(target)==ident['library_sha256']
    return target,ident,row['identity']['command'][0]

def install(backend):
    if backend=='cpu':
        from validation.sediment.river_morphodynamics import shoreline_hydraulic_probe as probe
        built=build();probe.build=lambda mode:built

def configure(o):
    from wallpoly181 import geometry
    from stage_wall172 import sha
    a=dict(o.data);a['xy']=np.asarray(a['xy']).T;ids,axes,sgn=geometry(a)
    if o.backend=='cpu':
        f=o.hydraulics.lib.poly181_set;f.argtypes=[ct.c_void_p]*3;f.restype=None
        f(*(ct.c_void_p(v.ctypes.data) for v in (ids,axes,sgn)))
    else:
        import cupy as cp
        from validation.gpu_fwave.reconstruction import OPTIONS
        module=cp.RawModule(code=CUDA,options=OPTIONS);kernel=module.get_function('wallstate181')
        rec=o.hydraulics.reconstruction;original=rec.reconstruct;g=rec.geometry
        resident=[cp.asarray(v) for v in (ids,axes,sgn,o.data['area'])]
        def reconstruct(q,*args,**kw):
            b=original(q,*args,**kw)
            kernel(((rec.nc+127)//128,),(128,),(q,b['grad'],b['vlo'],b['vhi'],b['hydro_constant'],g['xy'],g['mid'],resident[3],
                g['owner'],g['neighbour'],g['bid'],g['offset'],g['faces'],*resident[:3],np.int64(rec.nc),np.int64(rec.nf),
                np.float64(o.physics.parameters.dry),b['face_value'],b['trace']))
            return b
        rec.reconstruct=reconstruct
    o.identity['wallstate181']=dict(method=METHOD,source_sha256=sha(__file__),eligible_cells=int(np.sum(ids>=0)),public_support=False)
