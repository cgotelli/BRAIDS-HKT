"""Isolated slip-wall limiter-bound test; not an installed numerical option."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def native_source(code):
    marker = '   grad(:,2,c)=(a11*b(:,2)-a12*b(:,1))/det'
    if code.count(marker) != 3:
        raise ValueError('Expected all three native reconstruction routes')
    code = code.replace(marker, marker+'\n   call groin163_wall_bounds(c,v,vlo,vhi)')
    helper = '''
 subroutine groin163_wall_bounds(c,v,vlo,vhi)
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
    return code[:code.rindex('\nend module')]+helper+code[code.rindex('\nend module'):]


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


def build_native():
    from check_bedload161 import sha, save
    table=json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text())
    row=next(v for k,v in table.items() if 'shoreline_hydraulic_probe.build' in k)
    original=ROOT/'desktop/runtime'/Path(row['native']['file']).parent/'shoreline_probe.f90'
    assert sha(original)==row['identity']['source_sha256']
    code=native_source(original.read_text())
    digest=hashlib.sha256(code.encode()).hexdigest()
    out=ROOT/'validation/_groin163/build'/digest[:20]
    target=out/'hydraulics.dll'
    if not target.exists():
        out.mkdir(parents=True,exist_ok=True)
        source=out/'wall_bounds.f90';source.write_text(code)
        cmd=list(row['identity']['command'])
        old_source=next(v for v in cmd if v.endswith('shoreline_probe.f90'))
        cmd[cmd.index(old_source)]=str(source)
        cmd[cmd.index('-o')+1]=str(target)
        result=subprocess.run(cmd,cwd=out,capture_output=True,text=True,timeout=180)
        (out/'BUILD.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(result.stderr)
        save(out/'IDENTITY.json',dict(parent=row,command=cmd,source_sha256=digest,
             library_sha256=sha(target),Delft3D_modified=False,production_modified=False,
             method='reflected_wall_velocity_limiter_bounds_research163'))
    identity=json.loads((out/'IDENTITY.json').read_text())
    assert sha(target)==identity['library_sha256']
    return target,identity,row['identity']['command'][0]


def install(backend):
    """Call after portable loader install, before public owner construction."""
    from braids_core.sediment import wall_bed_hydraulics as wall
    from validation.sediment.river_morphodynamics import shoreline_hydraulic_probe as probe
    if backend=='cpu':
        built=build_native()
        probe.build=lambda mode:built
    else:
        previous=wall.cuda_source
        wall.cuda_source=lambda code:cuda_source(previous(code))
