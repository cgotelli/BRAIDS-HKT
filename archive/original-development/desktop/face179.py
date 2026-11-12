"""Isolated wall-cell fallback velocity limiter; no physical or f-wave changes.

Keep the unlimited LS velocity slope only in wall cells, then bound each
fallback face separately, just as the existing WENO branch bounds its faces.
Depth, bed, sediment reconstruction, reflection and open-boundary maps stay fixed.
"""
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]


def native_source(code):
    marker='   theta(bed_primitive_index())=braids_wall_bed_factor(c,v,grad,theta)'
    assert code.count(marker)==3
    code=code.replace(marker,marker+'\n   if(face179_wall(c)) theta(2:3)=1.d0')
    start=code.index(' function face_value(');end=code.index('\n end function',start)
    block=code[start:end]
    marker='  if(weno_cells(1,side,f)==0) then'
    assert block.count(marker)==1
    block=block.replace(marker,marker+'\n   value(2:3)=max(vlo(2:3,c),min(vhi(2:3,c),value(2:3)))')
    code=code[:start]+block+code[end:]
    helper='''
 logical function face179_wall(c) result(found)
  integer,intent(in)::c
  integer::k,f
  found=.false.
  do k=offset(c),offset(c+1)-1
   f=faces(k)
   if(neighbour(f)==0.and.bid(f)==0)then
    found=.true.;return
   endif
  enddo
 end function
'''
    pos=code.rindex('\nend module');return code[:pos]+helper+code[pos:]


def cuda_source(code):
    marker='            for(int m=0;m<4;++m) {gx[m]*=fmax(0.,theta[m]);gy[m]*=fmax(0.,theta[m]);}'
    assert code.count(marker)==1
    code=code.replace(marker,'''            for(int k=offset[c];k<offset[c+1];++k) {
                const int f=faces[k];if(neighbour[f]<0 && bid[f]<0){theta[1]=1.;theta[2]=1.;break;}
            }
'''+marker)
    marker='            if(weno[side*nf+f]>=0)for(int m=0;m<4;++m) {'
    assert code.count(marker)==1
    return code.replace(marker,'''            if(weno[side*nf+f]<0)for(int m=1;m<3;++m)
                value[m]=fmax(vlo[m*nc+cell],fmin(vhi[m*nc+cell],value[m]));
'''+marker)


def build():
    from stage_wall172 import sha,save
    table=json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text())
    rows=[v for k,v in table.items() if 'shoreline_hydraulic_probe.build' in k and
        'call braids_wall_velocity_bounds' in (ROOT/'desktop/runtime'/Path(v['native']['file']).parent/'shoreline_probe.f90').read_text()]
    assert len(rows)==1;row=rows[0]
    source=ROOT/'desktop/runtime'/Path(row['native']['file']).parent/'shoreline_probe.f90'
    assert sha(source)==row['identity']['source_sha256']
    code=native_source(source.read_text());digest=hashlib.sha256(code.encode()).hexdigest()
    folder=ROOT/'validation/_trace179/build'/digest[:20];target=folder/'hydraulics.dll'
    if not target.exists():
        folder.mkdir(parents=True,exist_ok=True);p=folder/'hydraulics.f90';p.write_text(code)
        cmd=list(row['identity']['command']);old=next(s for s in cmd if s.endswith('shoreline_probe.f90'))
        cmd[cmd.index(old)]=str(p);cmd[cmd.index('-o')+1]=str(target)
        r=subprocess.run(cmd,cwd=folder,capture_output=True,text=True,timeout=180)
        (folder/'BUILD.log').write_text(r.stdout+r.stderr);r.check_returncode()
        save(folder/'IDENTITY.json',dict(parent=row,command=cmd,source_sha256=digest,library_sha256=sha(target),
            method='wall_fallback_facewise_velocity179_research',production_modified=False,Delft3D_modified=False))
    identity=json.loads((folder/'IDENTITY.json').read_text());assert sha(target)==identity['library_sha256']
    return target,identity,row['identity']['command'][0]


def install(backend):
    from braids_core.sediment import wall_bed_hydraulics as wall
    from validation.sediment.river_morphodynamics import shoreline_hydraulic_probe as probe
    if backend=='cpu':
        built=build();probe.build=lambda mode:built
    else:
        previous=wall.cuda_source;wall.cuda_source=lambda code:cuda_source(previous(code))
