"""Isolated bounded, least-change hydraulic gradient experiment (not released)."""

HELPER=r'''
__device__ inline bool projection_ok(double x,double y,int count,const double* ax,const double* ay,const double* b) {
 for(int i=0;i<count;++i) if(ax[i]*x+ay[i]*y>b[i]+2.e-14)return false;
 return true;
}
__device__ inline void project_gradient(double &x,double &y,int count,const double* ax,const double* ay,const double* b) {
 const double rx=x,ry=y;
 if(projection_ok(x,y,count,ax,ay,b))return;
 double bx=0.,by=0.,best=rx*rx+ry*ry;
 for(int i=0;i<count;++i) {
  double dot=ax[i]*rx+ay[i]*ry-b[i],cx=rx-dot*ax[i],cy=ry-dot*ay[i];
  double d=(cx-rx)*(cx-rx)+(cy-ry)*(cy-ry);
  if(d<best && projection_ok(cx,cy,count,ax,ay,b)){bx=cx;by=cy;best=d;}
  for(int j=0;j<i;++j) {
   const double det=ax[i]*ay[j]-ay[i]*ax[j];if(fabs(det)<1.e-12)continue;
   cx=(b[i]*ay[j]-ay[i]*b[j])/det;cy=(ax[i]*b[j]-b[i]*ax[j])/det;
   d=(cx-rx)*(cx-rx)+(cy-ry)*(cy-ry);
   if(d<best && projection_ok(cx,cy,count,ax,ay,b)){bx=cx;by=cy;best=d;}
  }
 }
 x=bx;y=by;
}
'''

INSERT=r'''
            double px[2]={gx[0],gx[3]},py[2]={gy[0],gy[3]};
            bool candidate=offset[c+1]-offset[c]<=32;
            // Preserve the accepted wall-secant reconstruction in this diagnostic.
            for(int k=offset[c];k<offset[c+1];++k) {
              const int f=faces[k];if(neighbour[f]<0 && bid[f]<0)candidate=false;
            }
            if(candidate) for(int a=0;a<2;++a) {
              const int m=a?3:0;double ax[64],ay[64],b[64];int count=0;
              for(int k=offset[c];k<offset[c+1];++k) {
                const int f=faces[k];const double rx=mid[f]-xy[c],ry=mid[nf+f]-xy[nc+c],length=hypot(rx,ry);
                if(neighbour[f]<0 && bid[f]>=0 && boundary_mode[f]==2 && h>dry &&
                   fabs(centre[1]*normal[f]+centre[2]*normal[nf+f])<sqrt(grav*h) &&
                   fabs(gx[3]*rx+gy[3]*ry)>64.0*2.2204460492503131e-16*fmax(1.0,fabs(centre[3])))continue;
                if(length<=0.)continue;
                ax[count]=rx/length;ay[count]=ry/length;b[count++]=fmax(0.,hi[m]-centre[m])/length;
                ax[count]=-rx/length;ay[count]=-ry/length;b[count++]=fmax(0.,centre[m]-lo[m])/length;
              }
              project_gradient(px[a],py[a],count,ax,ay,b);
            }
'''


def install(a,folder,walls=False,secant=False):
    if a.backend=='cpu':return install_cpu(a,folder,walls=walls,secant=secant)
    import cupy as cp
    from validation.gpu_fwave.reconstruction import OPTIONS
    r=a.hydraulics.reconstruction
    code=r.module.code
    marker='            for(int m=0;m<4;++m) {gx[m]*=fmax(0.,theta[m]);gy[m]*=fmax(0.,theta[m]);}'
    assert code.count(marker)==1
    code=HELPER+code.replace(marker,INSERT+marker+'\n            if(candidate){gx[0]=px[0];gy[0]=py[0];gx[3]=px[1];gy[3]=py[1];}')
    if walls:
        code=code.replace('if(neighbour[f]<0 && bid[f]<0)candidate=false;','')
        code=code.replace('if(length<=0.)continue;','if(length<=0. || (m==3 && neighbour[f]<0 && bid[f]<0))continue;')
    if secant:
        code=code.replace('            if(candidate) for(int a=0;a<2;++a) {',r'''
            bool atwall=false,secantok=h>20.*dry;
            for(int k=offset[c];k<offset[c+1];++k) {
              const int f=faces[k],n=signs[k]<0?owner[f]:neighbour[f];
              if(n<0){if(bid[f]>=0)secantok=false;else atwall=true;continue;}
              const double dn=ls_delta[k]*normal[f]+ls_delta[nadj+k]*normal[nf+f];
              const double frac=((mid[f]-xy[c])*normal[f]+(mid[nf+f]-xy[nc+c])*normal[nf+f])/dn;
              if(v[n]-v[3*nc+n]<=20.*dry || fabs(dn)<=2.2204460492503131e-16 || frac<=0. || frac>=1.)secantok=false;
            }
            if(candidate) for(int a=0;a<2;++a) {''')
        code=code.replace('              project_gradient(px[a],py[a],count,ax,ay,b);',r'''
              if(m==3 && atwall) {
                if(!secantok){px[a]*=theta[3];py[a]*=theta[3];continue;}
                count=0;
                for(int k=offset[c];k<offset[c+1];++k){
                  const int f=faces[k],n=signs[k]<0?owner[f]:neighbour[f];if(n<0)continue;
                  const double dn=ls_delta[k]*normal[f]+ls_delta[nadj+k]*normal[nf+f];
                  const double frac=((mid[f]-xy[c])*normal[f]+(mid[nf+f]-xy[nc+c])*normal[nf+f])/dn;
                  const double rx=frac*ls_delta[k],ry=frac*ls_delta[nadj+k],len=hypot(rx,ry),delta=v[3*nc+n]-centre[3];
                  const double eps=64.*2.2204460492503131e-16*fmax(1.,fmax(fabs(centre[3]),fabs(v[3*nc+n])));
                  ax[count]=rx/len;ay[count]=ry/len;b[count++]=(fmax(0.,delta)+eps)/len;
                  ax[count]=-rx/len;ay[count]=-ry/len;b[count++]=(-fmin(0.,delta)+eps)/len;
                }
              }
              project_gradient(px[a],py[a],count,ax,ay,b);''')
    names=tuple(r.kernels)
    r.module=cp.RawModule(code=code,options=OPTIONS,name_expressions=names)
    for name in names:
        fn=r.module.get_function(name)
        r.kernels[name]=(lambda grid,block,args,fn=fn:fn(grid,block,(*args,r.shore_blend))) if name!='reconstruction_primitive' else fn
    (folder/'PROJECTED_RECONSTRUCTION.cu').write_text(code)


def install_cpu(a,folder,walls=False,secant=False):
    import hashlib,json,subprocess
    from pathlib import Path
    from braids_core.sediment import shoreline_solver as shoreline,bend_native_hydraulics as native
    from braids_core.sediment.bend_methods_v1 import digest
    root=Path(__file__).resolve().parents[1]
    target=root/'validation/_inlet187/prescribed_candidate'
    table=json.loads((target/'PRECOMPILED.json').read_text())
    base=Path(a.hydraulics.lib._name).parent.parent/'shoreline_probe.f90'
    assert base.is_file(),base
    code=base.read_text()
    marker='   do m=1,nprimitive\n    grad(m,:,c)=grad(m,:,c)*max(0.d0,theta(m))'
    assert code.count(marker)==3
    code=code.replace(marker,'   call project_cell188(c,v,grad,vlo,vhi,theta)\n'+marker)
    end=code.rindex('\nend module')
    code=code[:end]+'\n'+Path(__file__).with_suffix('.f90.inc').read_text()+code[end:]
    if walls:
        code=code.replace('   if(neighbour(f)==0.and.bid(f)<=0)return','')
        code=code.replace('    if(dd<=0.d0)cycle','    if(dd<=0.d0.or.(m==bed_primitive_index().and.neighbour(f)==0.and.bid(f)<=0))cycle')
    if secant:
        start=code.index(' subroutine project_cell188(')
        prefix,code=code[:start],code[start:]
        code=code.replace('  integer::k,f,a,m,count','  integer::k,f,a,m,count,n\n  logical::atwall,secantok\n  real(8)::dn,frac,delta,eps')
        code=code.replace('  do a=1,2','''  atwall=.false.;secantok=h>20.d0*dry_tolerance
  do k=offset(c),offset(c+1)-1
   f=faces(k);n=neighbour(f)
   if(signs(k)<0)n=owner(f)
   if(n==0)then
    if(bid(f)>0)then
     secantok=.false.
    else
     atwall=.true.
    endif
    cycle
   endif
   dn=dot_product(bed_ls_delta(:,k),normal(:,f))
   frac=dot_product(mid(:,f)-xy(:,c),normal(:,f))/dn
   if(v(1,n)-v(bed_primitive_index(),n)<=20.d0*dry_tolerance.or.abs(dn)<=epsilon(1.d0).or.frac<=0.d0.or.frac>=1.d0)secantok=.false.
  enddo
  do a=1,2''')
        marker='   call project_pair188(grad(m,1,c),grad(m,2,c),count,ax,ay,b)'
        code=code.replace(marker,'''   if(m==bed_primitive_index().and.atwall)then
    if(.not.secantok)cycle
    count=0
    do k=offset(c),offset(c+1)-1
     f=faces(k);n=neighbour(f)
     if(signs(k)<0)n=owner(f)
     if(n==0)cycle
     dn=dot_product(bed_ls_delta(:,k),normal(:,f))
     frac=dot_product(mid(:,f)-xy(:,c),normal(:,f))/dn
     rx=frac*bed_ls_delta(1,k);ry=frac*bed_ls_delta(2,k);dd=sqrt(rx*rx+ry*ry)
     delta=v(m,n)-v(m,c);eps=64*epsilon(1.d0)*max(1.d0,abs(v(m,c)),abs(v(m,n)))
     count=count+1;ax(count)=rx/dd;ay(count)=ry/dd;b(count)=(max(0.d0,delta)+eps)/dd
     count=count+1;ax(count)=-rx/dd;ay(count)=-ry/dd;b(count)=(-min(0.d0,delta)+eps)/dd
    enddo
   endif
'''+marker)
        code=prefix+code
    key=hashlib.sha256(code.encode()).hexdigest()[:20]
    dest=root/'validation/_wall188/build'/key;dest.mkdir(parents=True,exist_ok=True)
    generated=dest/'projected.f90';binary=dest/'hydraulics.dll'
    command=table['braids_core.sediment.wall_bed_hydraulics.build']['identity']['command']
    dependencies=[s for s in command if s.endswith(('.f','.f90'))][:-1]
    command=[command[0],'-O3','-fopenmp','-shared','-ffree-line-length-none','-ffixed-line-length-none',
        *dependencies,str(generated),'-o',str(binary),'-Wl,--export-all-symbols']
    if not binary.exists():
        generated.write_text(code)
        result=subprocess.run(command,cwd=dest,capture_output=True,text=True)
        (dest/'build.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(str(dest/'build.log'))
    identity=dict(method='isolated_projected_gradient188',source_sha256=digest(generated),
        base_sha256=digest(base),library_sha256=digest(binary),command=command,production_modified=False)
    (folder/'PROJECTED.json').write_text(json.dumps(identity,indent=2))
    with native._lock:
        previous=shoreline.build_native
        try:
            shoreline.build_native=lambda:(binary,identity,command[0])
            old=a.hydraulics.lib
            a.hydraulics=shoreline.NativeShoreline(a.hydraulics.deck,a.data,a.controls.threads)
            import ctypes
            for name,fn in vars(old).items():
                if isinstance(fn,ctypes._CFuncPtr):
                    replacement=getattr(a.hydraulics.lib,name)
                    replacement.argtypes=fn.argtypes;replacement.restype=fn.restype
            a.midpoint.solver.hydraulics=a.hydraulics
            # Fresh sediment-wave traces must use the SAME reconstructed
            # hydraulic state, not a closure retaining the previous library.
            import inspect
            seen=set();rebound=[]
            def rebind(function):
                if not inspect.isfunction(function) or id(function) in seen:return
                seen.add(id(function))
                for cell in function.__closure__ or ():
                    value=cell.cell_contents
                    if isinstance(value,ctypes._CFuncPtr) and value.__name__=='bend_sediment_traces':
                        cell.cell_contents=a.hydraulics.lib.bend_sediment_traces;rebound.append(value.__name__)
                    elif inspect.isfunction(value):rebind(value)
            rebind(a.physics.wave_engine.launch)
            assert len(rebound)==1,rebound
            identity['fresh_sediment_trace_binding']=rebound
            (folder/'PROJECTED.json').write_text(json.dumps(identity,indent=2))
        finally:shoreline.build_native=previous
