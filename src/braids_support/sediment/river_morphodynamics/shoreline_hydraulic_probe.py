"""Isolated BRAIDS hydraulic contrasts; never changes reference/production code."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import ctypes
from braids_core.sediment import shoreline_solver as shoreline
from braids_core.sediment import bend_native_hydraulics as native
from braids_core.sediment import wall_bed_hydraulics as wall
from braids_core.sediment.bend_methods_v1 import digest

HERE=Path(__file__).parent


def native_code(code,mode):
    if mode in ('continuous','bounded_bed','positive_bounded'):
        declaration=' logical, allocatable :: hydro_constant(:)'
        assert code.count(declaration)==1
        code=code.replace(declaration,declaration+'\n real(8),allocatable :: shore_blend(:)')
        begin=code.index(' subroutine prepare_wet_dry_traces(');end=code.index('\n end subroutine',begin)+len('\n end subroutine')
        envelope=(HERE/'continuous_shoreline.f90.inc').read_text().rstrip()
        if mode=='bounded_bed':
            envelope=envelope.replace('real(8)::h,eta,zb,phi','real(8)::h,eta,zb,phi\n  logical::flatten')
            envelope=envelope.replace('h=state(1,c);phi=0.d0','h=state(1,c);phi=0.d0;flatten=h<=dry_tolerance')
            envelope=envelope.replace('phi=min(phi,max(0.d0,min(eta-zb,v(1,c)-zb)/h))',
                'phi=min(phi,h/max(abs(zb-v(bed_primitive_index(),c)),tiny(1.d0)))\n     if(eta-zb<=dry_tolerance) flatten=.true.')
            envelope=envelope.replace('phi=0.d0\n    enddo','flatten=.true.\n    enddo')
            envelope=envelope.replace('hydro_constant(c)=phi<=0.d0','hydro_constant(c)=flatten')
            envelope=envelope.replace('phi)','phi,flatten)')
        if mode=='positive_bounded':
            envelope=envelope.replace('phi=min(phi,max(0.d0,min(eta-zb,v(1,c)-zb)/h))',
                'phi=min(phi,h/max(abs(zb-v(bed_primitive_index(),c)),tiny(1.d0)))\n     if(eta-zb<dry_tolerance) phi=min(phi,(h-dry_tolerance)/max(h-(eta-zb),tiny(1.d0)))')
            envelope=envelope.replace('hydro_constant(c)=phi<=0.d0','hydro_constant(c)=h<=dry_tolerance')
            # The h/abs(dz) bound above ALREADY limits cropped-cell bed
            # excursions. The legacy boolean override made phi jump 0 -> 1
            # at abs(dz)==h, destroying continuity of the implicit residual.
            envelope=envelope.replace('     if(any(rectangle_width(:,c)<=0.d0).and.abs(zb-v(bed_primitive_index(),c))>=h) phi=0.d0\n','')
        code=code[:begin]+envelope+code[end:]
        begin=code.index(' function face_value(');end=code.index('\n end function',begin)
        routine=code[begin:end]
        # Each early-return branch and the ordinary exit apply the same scalar
        # blend. Existing constant-surface source quadrature remains exact at
        # phi=0; phi>0 keeps all face depths positive and uses wet quadrature.
        blend='value(1:3)=v(1:3,c)+shore_blend(c)*(value(1:3)-v(1:3,c))'
        if mode in ('bounded_bed','positive_bounded'):
            blend='value(1)=v(1,c)+shore_blend(c)*(value(1)-v(1,c))\n  value(bed_primitive_index())=v(bed_primitive_index(),c)+shore_blend(c)*(value(bed_primitive_index())-v(bed_primitive_index(),c))'
        routine=routine.replace('     return','     '+blend+'\n     return')
        # Exact indentation avoids applying the blend twice at wall returns.
        routine=routine.replace('\n   return','\n   '+blend+'\n   return')
        routine+='\n  '+blend
        code=code[:begin]+routine+code[end:]
        for name in ('prepare_hydraulic_traces_fused','prepare_adaptive_traces_fused'):
            needle=f'   call {name}(state,v,bc_concentration,grad,vlo,vhi)'
            code=code.replace(needle,needle+'\n   call prepare_wet_dry_traces(state,v,grad,vlo,vhi)')
        return code
    if mode=='constant':
        needle='   call prepare_wet_dry_traces(state,v,grad,vlo,vhi)'
        assert code.count(needle)==2
        return code.replace(needle,needle+'\n   hydro_constant=.true.')
    if mode=='no_aperture':
        needle='    if(min(state(1,c),state(1,n))<=dry_tolerance) then'
        assert code.count(needle)==2
        return code.replace(needle,'    if(.false.) then')
    raise ValueError(mode)


def build(mode):
    library,parent,compiler=shoreline.build_native()
    source=library.parent/'shoreline.f90'
    code=native_code(source.read_text(),mode)
    key=hashlib.sha256((digest(library)+code).encode()).hexdigest()[:20]
    dst=Path('build/shoreline_hydraulic_probe').resolve()/key;dst.mkdir(parents=True,exist_ok=True)
    output=dst/'hydraulics.dll'
    if not output.exists():
        from tools.build_cutcell import SOURCES
        original=wall.build()[0].parent/'source'
        generated=dst/'shoreline_probe.f90';generated.write_text(code)
        command=[compiler,'-O3','-fopenmp','-shared','-ffree-line-length-none','-ffixed-line-length-none',
            *[str(original/Path(p).name) for p in SOURCES[:-1]],str(generated),'-o',str(output),'-Wl,--export-all-symbols']
        result=subprocess.run(command,cwd=dst,capture_output=True,text=True,timeout=180)
        (dst/'build.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(str(dst/'build.log'))
        (dst/'identity.json').write_text(json.dumps(dict(mode=mode,parent=parent,command=command,
            source_sha256=digest(generated),library_sha256=digest(output),wrapper_sha256=digest(__file__),
            production_modified=False,Delft3D_modified=False),indent=2))
    identity=json.loads((dst/'identity.json').read_text())
    assert digest(output)==identity['library_sha256']
    return output,identity,compiler


def install(a,mode):
    if a.backend=='gpu':
        if mode not in ('continuous','bounded_bed','positive_bounded'):raise ValueError('Diagnostic contrast not implemented on CUDA')
        import cupy as cp
        from braids_support.gpu_fwave.reconstruction import OPTIONS
        # Caller retains the authenticated physical input/source folder.
        from braids_core.windows_paths import filesystem_path
        source=filesystem_path(a._shoreline_source_folder)/'reconstruction.cu'
        code=wall.cuda_source(source.read_text())
        code=code.replace('int* constant)','int* constant,double* blend)')
        code=code.replace('int* kind,int* status)','int* kind,int* status,const double* blend)')
        code=code.replace('    bool flatten=q[c]<=dry;','    double phi=q[c]>dry?1.:0.;\n    bool flatten=q[c]<=dry;')
        predicate='if(eta-zb<=dry || ((rectangle[c]<=0. || rectangle[nc+c]<=0.) && fabs(zb-centre[3])>=q[c])) {flatten=true;break;}'
        assert code.count(predicate)==1
        if mode=='continuous':
            replacement='phi=fmin(phi,fmax(0.,fmin(eta-zb,centre[0]-zb)/q[c]));\n        if((rectangle[c]<=0. || rectangle[nc+c]<=0.) && fabs(zb-centre[3])>=q[c])phi=0.;'
        elif mode=='bounded_bed':
            replacement='phi=fmin(phi,q[c]/fmax(fabs(zb-centre[3]),2.2250738585072014e-308));\n        if(eta-zb<=dry || ((rectangle[c]<=0. || rectangle[nc+c]<=0.) && fabs(zb-centre[3])>=q[c]))flatten=true;'
        else:
            replacement='phi=fmin(phi,q[c]/fmax(fabs(zb-centre[3]),2.2250738585072014e-308));\n        if(eta-zb<dry)phi=fmin(phi,(q[c]-dry)/fmax(q[c]-(eta-zb),2.2250738585072014e-308));'
        code=code.replace(predicate,replacement)
        code=code.replace('    constant[c]=flatten?1:0;',('    constant[c]=phi<=0.?1:0;' if mode=='continuous' else '    constant[c]=flatten?1:0;')+'blend[c]=phi;')
        needle='        for(int m=0;m<4;++m) {face_value[(m*2+side)*nf+f]=value[m];'
        assert code.count(needle)==1
        blend_code='        for(int m=0;m<3;++m)value[m]=v[m*nc+cell]+blend[cell]*(value[m]-v[m*nc+cell]);\n'
        if mode in ('bounded_bed','positive_bounded'):blend_code='        value[0]=v[cell]+blend[cell]*(value[0]-v[cell]);value[3]=v[3*nc+cell]+blend[cell]*(value[3]-v[3*nc+cell]);\n'
        code=code.replace(needle,blend_code+needle)
        names=('reconstruction_primitive','reconstruction_cells','reconstruction_faces')
        rec=a.hydraulics.reconstruction;rec.module=cp.RawModule(code=code,options=OPTIONS,name_expressions=names)
        blend=cp.ones(rec.nc);functions={name:rec.module.get_function(name) for name in names}
        for name,fn in functions.items():
            rec.kernels[name]=(lambda grid,block,args,fn=fn:fn(grid,block,(*args,blend))) if name!='reconstruction_primitive' else fn
        rec.shore_blend=blend
        a.identity['hydraulic_probe']=dict(mode=mode,backend='gpu',source_sha256=digest(source),
            generated_sha256=hashlib.sha256(code.encode()).hexdigest(),wrapper_sha256=digest(__file__))
        return a
    # NativeShoreline supplies its own builder; explicitly replace only during
    # construction under the shared library lock, then restore it.
    result=build(mode)
    with native._lock:
        previous=shoreline.build_native
        try:
            shoreline.build_native=lambda:result
            a.hydraulics=shoreline.NativeShoreline(a.hydraulics.deck,a.data,a.controls.threads)
        finally:shoreline.build_native=previous
    a.identity['hydraulic_probe']=a.hydraulics.identity
    return a
