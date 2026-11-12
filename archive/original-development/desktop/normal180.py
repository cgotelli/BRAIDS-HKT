"""Research: mirrored normal-velocity reconstruction, unchanged tangential states.

Only full Cartesian columns with complete reflected five-point paths qualify.
Cropped/oblique geometry keeps its current numerical reconstruction. The wall
Riemann solution, exterior BCs, bed/depth and sediment traces are unchanged.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def native_source(code):
    marker=' logical, allocatable :: wall_triangle(:),wall_ready(:,:)'
    assert code.count(marker)==1
    code=code.replace(marker,marker+'\n logical, allocatable :: normal_only180(:,:)')
    # The cached normal value is computed by the existing reflected WENO.
    # Do not let its unrelated depth/bed/tangential cache override the base.
    for name in ('face_value','face_elevations'):
        start=code.index(' function face_value(') if name=='face_value' else code.index(' subroutine face_elevations(')
        end=code.index('\n end function',start) if name=='face_value' else code.index('\n end subroutine',start)
        block=code[start:end]
        marker='    if(wall_ready(slot,c)) then'
        assert block.count(marker)==1
        block=block.replace(marker,'    if(wall_ready(slot,c).and..not.normal180_slot(c,slot)) then')
        if name=='face_value':
            block=block.replace('     return','     call normal180_value(c,f,value)\n     return')
            block=block.replace('   return','   call normal180_value(c,f,value)\n   return')
            block+='\n  call normal180_value(c,f,value)'
        code=code[:start]+block+code[end:]
    helper='''
 logical function normal180_slot(c,slot) result(yes)
  integer,intent(in)::c,slot
  yes=.false.
  if(allocated(normal_only180))yes=normal_only180((slot+1)/2,c)
 end function
 subroutine normal180_value(c,f,value)
  integer,intent(in)::c,f
  real(8),intent(inout)::value(nprimitive)
  integer::side,slot,a
  if(scheme/=6.or.hydro_constant(c))return
  if(neighbour(f)==0.and.bid(f)>0)return
  side=1;if(c/=owner(f))side=2
  slot=wall_slot(side,f);if(slot<=0)return
  if(.not.normal180_slot(c,slot))return
  if(.not.wall_ready(slot,c))return
  a=(slot+1)/2
  value(a+1)=wall_values(a+1,slot,c)
 end subroutine
 subroutine normal180_set(ids,kinds,normals) bind(C)
  use iso_c_binding
  integer(c_int),intent(in)::ids(nc,2,5),kinds(nc,2,5)
  real(c_double),intent(in)::normals(nc,2,5,2)
  integer::c,a,k
  if(allocated(normal_only180))deallocate(normal_only180)
  allocate(normal_only180(2,nc));normal_only180=.false.
  !$omp parallel do schedule(static) private(a,k)
  do c=1,nc
   do a=1,2
    normal_only180(a,c)=any(kinds(c,a,:)==4)
    do k=1,5
     wall_stencil_ids(k,a,c)=ids(c,a,k)+1
     wall_stencil_kind(k,a,c)=kinds(c,a,k)
     if(kinds(c,a,k)==4)wall_stencil_kind(k,a,c)=2
     wall_stencil_normal(:,k,a,c)=normals(c,a,k,:)
    enddo
   enddo
  enddo
  !$omp end parallel do
 end subroutine
'''
    pos=code.rindex('\nend module');return code[:pos]+helper+code[pos:]


def cuda_source(code):
    marker='                const int slot=2*a+side;double stencil[4][5],tangential[5];'
    assert code.count(marker)==1
    code=code.replace(marker,marker+'bool normal_only=false;')
    marker='                    if(kind>0) {'
    assert code.count(marker)==1
    code=code.replace(marker,'                    if(kind==4)normal_only=true;\n'+marker)
    code=code.replace('(double)kind*un*nx','(double)(kind==4?2:kind)*un*nx').replace('(double)kind*un*ny','(double)(kind==4?2:kind)*un*ny')
    assert code.count('                ready[slot]=1;')==1
    code=code.replace('                ready[slot]=1;','                ready[slot]=normal_only?2:1;')
    code=code.replace('if(slot>=0 && ready[slot]) {eta=','if(slot>=0 && ready[slot]==1) {eta=')
    code=code.replace('if(slot>=0 && wall_ready[slot*nc+cell])for','if(slot>=0 && wall_ready[slot*nc+cell]==1)for')
    marker='        if(constant[cell]) {for(int m=0;m<3;++m)value[m]=v[m*nc+cell];'
    assert code.count(marker)==1
    code=code.replace(marker,'''        if(type!=3 && slot>=0 && wall_ready[slot*nc+cell]==2) {
            const int m=1+slot/2;value[m]=wall_values[(m*4+slot)*nc+cell];
        }
'''+marker)
    return code


def build():
    from stage_wall172 import sha,save
    table=json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text())
    rows=[v for k,v in table.items() if 'shoreline_hydraulic_probe.build' in k and
        'call braids_wall_velocity_bounds' in (ROOT/'desktop/runtime'/Path(v['native']['file']).parent/'shoreline_probe.f90').read_text()]
    assert len(rows)==1;row=rows[0];p=ROOT/'desktop/runtime'/Path(row['native']['file']).parent/'shoreline_probe.f90'
    assert sha(p)==row['identity']['source_sha256'];code=native_source(p.read_text());digest=hashlib.sha256(code.encode()).hexdigest()
    dst=ROOT/'validation/_pair180/build'/digest[:20];target=dst/'hydraulics.dll'
    if not target.exists():
        dst.mkdir(parents=True,exist_ok=True);source=dst/'normal.f90';source.write_text(code)
        cmd=list(row['identity']['command']);old=next(s for s in cmd if s.endswith('shoreline_probe.f90'))
        cmd[cmd.index(old)]=str(source);cmd[cmd.index('-o')+1]=str(target)
        r=subprocess.run(cmd,cwd=dst,capture_output=True,text=True,timeout=180)
        (dst/'BUILD.log').write_text(r.stdout+r.stderr);r.check_returncode()
        save(dst/'IDENTITY.json',dict(parent=row,command=cmd,source_sha256=digest,library_sha256=sha(target),
            method='normal_reflected180_research',production_changed=False,Delft3D_changed=False))
    identity=json.loads((dst/'IDENTITY.json').read_text());assert sha(target)==identity['library_sha256']
    return target,identity,row['identity']['command'][0]


def install(backend):
    if backend=='cpu':
        from validation.sediment.river_morphodynamics import shoreline_hydraulic_probe as probe
        built=build();probe.build=lambda mode:built
    else:
        from braids_core.sediment import wall_bed_hydraulics as wall
        previous=wall.cuda_source;wall.cuda_source=lambda code:cuda_source(previous(code))


def configure(owner):
    from axis_stencil176 import geometry
    from stage_wall172 import sha
    fields,changed=geometry(owner.data)
    for c,axis in changed:
        k=fields['wall_stencil_kind'][:,axis,c];k[k==2]=4
    if owner.backend=='cpu':
        f=owner.hydraulics.lib.normal180_set;f.argtypes=[ct.c_void_p]*3;f.restype=None
        arrays=[np.ascontiguousarray(fields[k]) for k in ('wall_stencil_ids','wall_stencil_kind','wall_stencil_normal')]
        f(*(ct.c_void_p(a.ctypes.data) for a in arrays))
    else:
        for k,v in fields.items():owner.hydraulics.reconstruction.geometry[k][...]=owner.xp.asarray(v)
    owner.identity['normal180']=dict(source_sha256=sha(__file__),changed_cell_axes=len(changed),
        geometry_sha256={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in fields.items()},public_support=False)

