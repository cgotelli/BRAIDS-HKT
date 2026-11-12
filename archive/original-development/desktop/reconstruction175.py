"""Research-only second-order hydraulic reconstruction contrast.

Changes only the hydraulic five-cell interpolation, not the sediment
reconstruction, Riemann solver, wall pressure, boundary flux, or physical nu.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def native_source(code,variant='minmod'):
    if variant=='surface_constant':
        old='value(1)=v(1,c)+shore_blend(c)*(value(1)-v(1,c))'
        assert code.count(old)==3
        return code.replace(old,'value(1)=v(1,c) ! isolated water-surface-order diagnostic175')
    # Every two-dimensional stencil call below is hydraulic. The separate
    # sediment capacity routine uses reconstruct(stencil,5) and is untouched.
    old='reconstruct(stencil(:,m),5)'
    assert code.count(old)==4
    code=code.replace(old,'zero175_minmod(stencil(:,m))')
    assert code.count('reconstruct(tangential,5)')==3
    code=code.replace('reconstruct(tangential,5)','zero175_minmod(tangential)')
    for m in (1,2):
        old=f'reconstruct(stencil(:,{m}),5)'
        assert code.count(old)==1
        code=code.replace(old,f'zero175_minmod(stencil(:,{m}))')
    helper='''
 pure real(8) function zero175_minmod(v) result(out)
  real(8),intent(in)::v(5)
  real(8)::a,b
  a=v(3)-v(2);b=v(4)-v(3);out=v(3)
  if(a*b>0.d0) out=out+.5d0*sign(min(abs(a),abs(b)),a)
 end function
'''
    at=code.rindex('\nend module')
    return code[:at]+helper+code[at:]


def cuda_source(code,variant='minmod'):
    if variant=='surface_constant':
        old='        for(int m=0;m<4;++m) {face_value[(m*2+side)*nf+f]=value[m];'
        assert code.count(old)==1
        return code.replace(old,'        value[0]=v[cell]; // isolated surface-order diagnostic175\n'+old)
    begin=code.index('__device__ __forceinline__ double recon_weno(')
    end=code.index('\nextern "C"',begin)
    replacement='''__device__ __forceinline__ double recon_weno(const double* v) {
    const double a=v[2]-v[1],b=v[3]-v[2];
    return v[2]+(a*b>0. ? .5*copysign(fmin(fabs(a),fabs(b)),a) : 0.);
}
'''
    return code[:begin]+replacement+code[end:]


def build_native(variant='minmod'):
    from stage_wall172 import save,sha
    table=json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text())
    rows=[v for k,v in table.items() if 'shoreline_hydraulic_probe.build' in k
        and 'call braids_wall_velocity_bounds' in
        (ROOT/'desktop/runtime'/Path(v['native']['file']).parent/'shoreline_probe.f90').read_text()]
    assert len(rows)==1
    row=rows[0];original=ROOT/'desktop/runtime'/Path(row['native']['file']).parent/'shoreline_probe.f90'
    assert sha(original)==row['identity']['source_sha256']
    code=native_source(original.read_text(),variant);digest=hashlib.sha256(code.encode()).hexdigest()
    out=ROOT/'validation/_zero175/build'/digest[:20];target=out/'hydraulics.dll'
    if not target.exists():
        out.mkdir(parents=True,exist_ok=True);source=out/'hydraulics.f90';source.write_text(code)
        cmd=list(row['identity']['command']);old=next(v for v in cmd if v.endswith('shoreline_probe.f90'))
        cmd[cmd.index(old)]=str(source);cmd[cmd.index('-o')+1]=str(target)
        result=subprocess.run(cmd,cwd=out,capture_output=True,text=True,timeout=180)
        (out/'BUILD.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(result.stderr)
        save(out/'IDENTITY.json',dict(parent=row,command=cmd,source_sha256=digest,
            library_sha256=sha(target),production_modified=False,Delft3D_modified=False,
            method='hydraulic_'+variant+'_contrast175'))
    identity=json.loads((out/'IDENTITY.json').read_text());assert sha(target)==identity['library_sha256']
    return target,identity,row['identity']['command'][0]


def install(backend,variant='minmod'):
    from braids_core.sediment import wall_bed_hydraulics as wall
    from validation.sediment.river_morphodynamics import shoreline_hydraulic_probe as probe
    if backend=='cpu':
        built=build_native(variant);probe.build=lambda mode:built
    else:
        previous=wall.cuda_source;wall.cuda_source=lambda code:cuda_source(previous(code),variant)
