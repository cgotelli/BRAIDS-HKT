"""Isolated extension of existing reflected WENO geometry to axis walls.

Walk full shared faces, not grid coordinates: never read through an obstacle.
Only complete, unambiguous five-point paths are admitted. Reconstruction,
limiting, f-wave, wall pressure and all physical formulas remain unchanged.
The same host geometry is loaded once into each explicitly chosen backend.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def geometry(data):
    nc=len(data['area']);xy=np.asarray(data['xy']);mid=data['mid'];norm=data['normal']
    width=np.asarray(data['rectangle_width']);dx=float(data['dx']);tol=1e-10*dx
    full=np.all(abs(width-dx)<tol,axis=0)
    ids=np.array(data['wall_stencil_ids'],copy=True);kind=np.array(data['wall_stencil_kind'],copy=True)
    normals=np.array(data['wall_stencil_normal'],copy=True)
    o=data['owner'];n=data['neighbour'];bid=data['bid']
    # Neighbour graph: -2 unsupported, -1 exact full wall, >=0 full neighbour.
    graph=np.full((2,2,nc),-2,dtype=np.int32)
    for f in range(len(o)):
        axis=int(np.argmax(abs(norm[:,f])))
        if abs(abs(norm[axis,f])-1)>1e-12 or abs(norm[1-axis,f])>1e-12:continue
        if abs(data['length'][f]-dx)>tol:continue
        for c,other,sign in ((int(o[f]),int(n[f]),1),(int(n[f]),int(o[f]),-1)):
            if c<0 or not full[c]:continue
            side=1 if sign*norm[axis,f]>0 else 0
            expected=xy[:,c].copy();expected[axis]+=(side-.5)*dx
            if np.max(abs(expected-mid[:,f]))>tol:continue
            if other>=0:
                if not full[other]:continue
                target=xy[:,c].copy();target[axis]+=(2*side-1)*dx
                if np.max(abs(target-xy[:,other]))>tol:continue
                graph[axis,side,c]=other
            elif bid[f]<0:graph[axis,side,c]=-1
    changed=[]
    for c in np.flatnonzero(full):
        for axis in (0,1):
            actual={0:int(c)};hits={};valid=True
            for direction,side in ((-1,0),(1,1)):
                cur=int(c)
                for step in (1,2):
                    nxt=int(graph[axis,side,cur])
                    if nxt==-1:hits[direction]=step;break
                    if nxt<0:valid=False;break
                    actual[direction*step]=nxt;cur=nxt
            if not valid or not hits:continue
            path=np.full(5,-1,dtype=np.int32);kinds=np.zeros(5,dtype=np.int32);ns=np.zeros((2,5))
            for offset in range(-2,3):
                if offset in actual:path[offset+2]=actual[offset];continue
                direction=1 if offset>0 else -1;hit=hits.get(direction)
                if hit is None:valid=False;break
                mirror=direction*(2*hit-1-abs(offset))
                if mirror not in actual:valid=False;break
                path[offset+2]=actual[mirror];kinds[offset+2]=2;ns[axis,offset+2]=direction
            if not valid:continue
            ids[:,axis,c]=path;kind[:,axis,c]=kinds;normals[:,:,axis,c]=ns;changed.append((int(c),axis))
    return dict(wall_stencil_ids=ids,wall_stencil_kind=kind,wall_stencil_normal=normals),changed


def build():
    from stage_wall172 import sha,save
    table=json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text())
    rows=[v for k,v in table.items() if 'shoreline_hydraulic_probe.build' in k and
        'call braids_wall_velocity_bounds' in (ROOT/'desktop/runtime'/Path(v['native']['file']).parent/'shoreline_probe.f90').read_text()]
    assert len(rows)==1;row=rows[0]
    p=ROOT/'desktop/runtime'/Path(row['native']['file']).parent/'shoreline_probe.f90'
    assert sha(p)==row['identity']['source_sha256'];code=p.read_text()
    helper='''
subroutine axis176_set_stencil(ids,kinds,normals) bind(C)
 use iso_c_binding
 integer(c_int),intent(in)::ids(nc,2,5),kinds(nc,2,5)
 real(c_double),intent(in)::normals(nc,2,5,2)
 integer::c,a,k
 !$omp parallel do schedule(static) private(a,k)
 do c=1,nc
  do a=1,2
   do k=1,5
    wall_stencil_ids(k,a,c)=ids(c,a,k)+1
    wall_stencil_kind(k,a,c)=kinds(c,a,k)
    wall_stencil_normal(:,k,a,c)=normals(c,a,k,:)
   enddo
  enddo
 enddo
 !$omp end parallel do
end subroutine
'''
    pos=code.rindex('\nend module');code=code[:pos]+helper+code[pos:]
    digest=hashlib.sha256(code.encode()).hexdigest();dst=ROOT/'validation/_quadrature176/build'/digest[:20]
    target=dst/'hydraulics.dll'
    if not target.exists():
        dst.mkdir(parents=True,exist_ok=True);source=dst/'axis_stencil.f90';source.write_text(code)
        cmd=list(row['identity']['command']);old=next(s for s in cmd if s.endswith('shoreline_probe.f90'))
        cmd[cmd.index(old)]=str(source);cmd[cmd.index('-o')+1]=str(target)
        r=subprocess.run(cmd,cwd=dst,capture_output=True,text=True,timeout=180)
        (dst/'BUILD.log').write_text(r.stdout+r.stderr);r.check_returncode()
        save(dst/'IDENTITY.json',dict(parent=row,command=cmd,source_sha256=digest,library_sha256=sha(target),
            method='axis_reflected_stencil176_research',production_changed=False,Delft3D_changed=False))
    identity=json.loads((dst/'IDENTITY.json').read_text());assert sha(target)==identity['library_sha256']
    return target,identity,row['identity']['command'][0]


def install(backend):
    if backend=='cpu':
        from validation.sediment.river_morphodynamics import shoreline_hydraulic_probe as probe
        built=build();probe.build=lambda mode:built


def configure(owner):
    from stage_wall172 import sha
    from braids_core.sediment.bed_columns_v101 import host
    # The CPU input includes the same exported geometry as the GPU input.
    fields,changed=geometry(owner.data)
    if owner.backend=='cpu':
        f=owner.hydraulics.lib.axis176_set_stencil;f.argtypes=[ct.c_void_p]*3;f.restype=None
        arrays=[np.ascontiguousarray(fields[k]) for k in ('wall_stencil_ids','wall_stencil_kind','wall_stencil_normal')]
        f(*(ct.c_void_p(a.ctypes.data) for a in arrays))
    else:
        g=owner.hydraulics.reconstruction.geometry
        for k,v in fields.items():g[k][...]=owner.xp.asarray(v)
    owner.identity['axis_stencil176']=dict(source_sha256=sha(__file__),changed_cell_axes=len(changed),
        geometry_sha256={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in fields.items()},
        public_support=False)
    return changed
