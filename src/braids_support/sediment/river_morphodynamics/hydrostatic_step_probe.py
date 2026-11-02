"""Isolated Audusse-type interface experiment; NOT a maintained solver change.

Keep the augmented f-wave Riemann routine, reconstruction and within-cell bed
quadrature. Solve at the common upper face bed, with hydrostatic side-pressure
corrections. This does NOT claim the HR positivity/entropy theorem for this
particular solver. Dynamic run-up and steep-slope limitations must be screened.
"""
from pathlib import Path
import hashlib
from . import shoreline_hydraulic_probe as probe
from braids_core.sediment.bend_methods_v1 import digest

WRAPPER = '''
 subroutine hr_step_flux(left,right,bl,br,n,fl,fr,speed)
  real(8),intent(in)::left(3),right(3),bl,br,n(2)
  real(8),intent(out)::fl(3),fr(3),speed
  real(8)::ll(3),rr(3),top,hl,hr
  top=max(bl,br);hl=max(0.d0,left(1)+bl-top);hr=max(0.d0,right(1)+br-top)
  ll=0.d0;rr=0.d0;ll(1)=hl;rr(1)=hr
  if(left(1)>dry_tolerance) ll(2:3)=left(2:3)*(hl/left(1))
  if(right(1)>dry_tolerance) rr(2:3)=right(2:3)*(hr/right(1))
  call wave_flux(ll,rr,top,top,n,fl,fr,speed)
  fl(2:3)=fl(2:3)+.5d0*grav*(left(1)**2-hl**2)*n
  fr(2:3)=fr(2:3)+.5d0*grav*(right(1)**2-hr**2)*n
 end subroutine
'''


def native_transform(code):
    needle='call wave_flux(l,r,left(bed_primitive_index()),right(bed_primitive_index()), &'
    # NativeShoreline includes both the base Euler and its exported bridge.
    assert code.count(needle)==2
    return code.replace(needle,needle.replace('wave_flux','hr_step_flux')).replace(
        ' subroutine wave_flux(',WRAPPER+'\n subroutine wave_flux(',1)


def cuda_transform(code):
    needle='    if (kind[f] != 1 || fmin(q[c],q[n])>dry) return;'
    assert code.count(needle)==1
    code=code.replace(needle,'''    if(kind[f]==1) {
        const double top=fmax(bL,bR), hl=fmax(0.,hL+bL-top), hr=fmax(0.,hR+bR-top);
        local[f]=hl;local[3*nf+f]=top;local[4*nf+f]=hr;local[7*nf+f]=top;
        local[nf+f]=hL>dry?unL*(hl/hL):0.;local[2*nf+f]=hL>dry?utL*(hl/hL):0.;
        local[5*nf+f]=hR>dry?unR*(hr/hR):0.;local[6*nf+f]=hR>dry?utR*(hr/hR):0.;
    }
'''+needle)
    start=code.index('        double pl[3]=')
    end=code.index('        if (kind[f]==2)',start)
    fragment=code[start:end]
    fragment=fragment.replace('hL','starL').replace('hR','starR')
    fragment='        const double starL=local[f], starR=local[4*nf+f];\n'+fragment
    fragment+='''        if(kind[f]==1) {
            const double cl=.5*grav*(hL*hL-starL*starL), cr=.5*grav*(hR*hR-starR*starR);
            l[1]+=cl*nx;l[2]+=cl*ny;r[1]+=cr*nx;r[2]+=cr*ny;
        }
'''
    return code[:start]+fragment+code[end:]


def install(a):
    if a.backend=='cpu':
        original=probe.native_code
        try:
            probe.native_code=lambda code,mode:native_transform(original(code,mode))
            probe.install(a,'positive_bounded')
        finally:probe.native_code=original
    else:
        import cupy as cp
        from braids_support.gpu_fwave.resident_update import OPTIONS
        folder=Path(a._shoreline_source_folder)
        code=(folder/'augmented_fwave.cu').read_text()+'\n'+cuda_transform((folder/'resident_update.cu').read_text())
        op=a.hydraulics.flux
        names=tuple(op.kernels)
        op.module=cp.RawModule(code=code,options=OPTIONS,name_expressions=names)
        op.kernels={name:op.module.get_function(name) for name in names}
        a.identity['hr_generated_cuda_sha256']=hashlib.sha256(code.encode()).hexdigest()
    a.identity['hydrostatic_step_probe']=dict(driver_sha256=digest(__file__),production=False,
        augmented_fwave_unchanged=True,sediment_changed=False,
        reference='Audusse et al. 2004 DOI 10.1137/S1064827503431090, side corrections Eq. 2.16')
    return a
