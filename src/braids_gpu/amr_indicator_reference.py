"""NumPy AMR indicator oracle and cold-preparation bed gradient.

These routines never serve as a fallback for resident GPU accepted state.
Directed CSR traversal deliberately preserves the native near-water ordering.
"""
from __future__ import annotations

import numpy as np


def directed_geometry(data, n):
    """Validate complete incidence without reordering authenticated cell faces."""
    names=('owner','neighbour','bid','offset','faces','signs')
    values=[]
    for name in names:
        value=np.asarray(data[name])
        if value.ndim!=1 or value.dtype.kind not in 'iu':
            raise ValueError('Invalid AMR directed CSR '+name)
        values.append(value.astype(np.int64,copy=False))
    owner,other,bid,offset,faces,signs=values
    nf=len(owner)
    if (other.shape!=(nf,) or bid.shape!=(nf,) or offset.shape!=(n+1,) or
            signs.shape!=faces.shape or offset[0]!=0 or offset[-1]!=len(faces) or
            np.any(np.diff(offset)<0) or np.any((owner<0)|(owner>=n)) or
            np.any((other<-1)|(other>=n)) or np.any(other==owner) or
            np.any((faces<0)|(faces>=nf)) or not np.isin(signs,(-1,1)).all()):
        raise ValueError('Invalid AMR directed CSR shape or endpoint')
    cells=np.repeat(np.arange(n),np.diff(offset))
    if (not np.array_equal(np.where(signs>0,owner[faces],other[faces]),cells) or
            len(np.unique(2*faces+(signs<0)))!=len(faces) or
            len(faces)!=nf+np.count_nonzero(other>=0)):
        raise ValueError('Invalid AMR directed CSR incidence')
    return owner,other,bid,offset,faces,signs


def _bed_inputs(data, accepted):
    if (not isinstance(accepted,np.ndarray) or accepted.ndim!=2 or
            accepted.dtype!=np.float64 or not np.isfinite(accepted).all()):
        raise ValueError('AMR indicator reference requires finite FP64 NumPy state')
    index=np.asarray(data['bed_indices'])
    bulk=np.asarray(data['bulk_fraction'])
    rock=np.asarray(data['rock'])
    if (index.ndim!=2 or index.dtype.kind not in 'iu' or not index.size or
            np.any((index<0)|(index>=accepted.shape[1])) or
            bulk.shape!=(index.shape[1],) or bulk.dtype!=np.float64 or
            not np.isfinite(bulk).all() or np.any(bulk<=0) or
            rock.shape!=(len(accepted),) or rock.dtype!=np.float64 or
            not np.isfinite(rock).all()):
        raise ValueError('Invalid AMR bed/composition geometry')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        bed=rock+np.sum(accepted[:,index]/bulk[None,None,:],axis=(1,2))
    if not np.isfinite(bed).all():raise ValueError('Nonfinite AMR bed')
    return index.astype(np.int64,copy=False),bulk,bed


def cold_bed_gradient(data, accepted):
    """Native limited bed LS for CPU cold preparation, not runtime fallback.

    Mirrors the bed branch of reconstruction.cu, including the solver-dry active
    stencil and subcritical stage-outlet bound extension/limiter skip.
    """
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            return _cold_bed_gradient(data,accepted)
    except FloatingPointError as exc:
        raise ValueError('Nonfinite AMR cold bed-gradient arithmetic') from exc


def _cold_bed_gradient(data, accepted):
    _,_,bed=_bed_inputs(data,accepted)
    n=len(accepted)
    owner,other,bid,offset,faces,signs=directed_geometry(data,n)
    nf=len(owner);na=len(faces)
    arrays={}
    for name,shape in (('xy',(2,n)),('mid',(2,nf)),('normal',(2,nf)),
                       ('hydraulic_ls_matrix',(4,n)),('hydraulic_ls_weight',(na,)),
                       ('bed_ls_delta',(2,na))):
        value=np.asarray(data[name])
        if value.shape!=shape or value.dtype!=np.float64 or not np.isfinite(value).all():
            raise ValueError('Invalid AMR cold gradient '+name)
        arrays[name]=value
    xy,mid,normal=(arrays[k] for k in ('xy','mid','normal'))
    matrix,weights,delta=(arrays[k] for k in ('hydraulic_ls_matrix','hydraulic_ls_weight','bed_ls_delta'))
    mode=np.asarray(data['boundary_mode'])
    if mode.shape!=(nf,) or mode.dtype.kind not in 'iu':
        raise ValueError('Invalid AMR cold gradient boundary modes')
    dry=float(data['dry_tolerance']);grav=float(data['grav'])
    extension=int(data.get('outlet_extension',0))
    if not np.isfinite([dry,grav]).all() or dry<=0 or grav<=0 or extension not in (0,1):
        raise ValueError('Invalid AMR cold gradient controls')
    if np.any(accepted[:,0]<0):raise ValueError('Invalid AMR cold gradient water')
    out=np.zeros((n,2),np.float64)
    for c in range(n):
        wet=accepted[c,0]>dry
        entries=range(offset[c],offset[c+1])
        active=wet or any(bid[faces[k]]>=0 or
            ((j:=int(other[faces[k]] if signs[k]>0 else owner[faces[k]]))>=0 and accepted[j,0]>dry)
            for k in entries)
        if not active:continue
        bx=by=0.;lo=hi=bed[c]
        for k in entries:
            f=faces[k];j=other[f] if signs[k]>0 else owner[f]
            if j<0:continue
            diff=bed[j]-bed[c]
            bx+=weights[k]*delta[0,k]*diff
            by+=weights[k]*delta[1,k]*diff
            lo=min(lo,bed[j]);hi=max(hi,bed[j])
        a11,a12,a22,det=matrix[:,c]
        if det<=1e-12*(a11+a22)**2:continue
        gx=(a22*bx-a12*by)/det;gy=(a11*by-a12*bx)/det
        u=accepted[c,1]/accepted[c,0] if wet else 0.
        v=accepted[c,2]/accepted[c,0] if wet else 0.
        # Native h is reconstructed primitive eta minus bed, not a fresh q copy.
        h=(accepted[c,0]+bed[c])-bed[c]
        def stage_face(f,increment):
            return (other[f]<0 and bid[f]>=0 and mode[f]==2 and h>dry and
                abs(u*normal[0,f]+v*normal[1,f])<np.sqrt(grav*h) and
                abs(increment)>64*np.finfo(float).eps*max(1.,abs(bed[c])))
        if extension and h>dry:
            for k in entries:
                f=faces[k];r=mid[:,f]-xy[:,c];change=gx*r[0]+gy*r[1]
                if stage_face(f,change):
                    exterior=bed[c]+2*change;lo=min(lo,exterior);hi=max(hi,exterior)
        theta=1.
        eps=64*np.finfo(float).eps*max(1.,abs(bed[c]),abs(lo),abs(hi))
        for k in entries:
            f=faces[k];r=mid[:,f]-xy[:,c];change=gx*r[0]+gy*r[1]
            if stage_face(f,change) or abs(change)<=eps:continue
            if change>0:theta=min(theta,(hi-bed[c])/change)
            else:theta=min(theta,(lo-bed[c])/change)
        out[c]=(gx*max(0.,theta),gy*max(0.,theta))
    if not np.isfinite(out).all():raise ValueError('Nonfinite AMR cold gradient')
    return out


def extended_tags(s,data,ids,accepted,rows,bed,wet,raw,cause,moving,*,
                  selector_last_bed,bed_gradient):
    """Apply the missing native tags to PRIVATE host scratch, never state.

    Returns component diagnostics without attaching mutable epoch state to the
    selector. The ordinary public proposal exposes compact counters only.
    """
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            return _extended_tags(s,data,ids,accepted,rows,bed,wet,raw,cause,moving,
                selector_last_bed=selector_last_bed,bed_gradient=bed_gradient)
    except FloatingPointError as exc:
        raise ValueError('Nonfinite extended AMR indicator arithmetic') from exc


def _extended_tags(s,data,ids,accepted,rows,bed,wet,raw,cause,moving,*,
                   selector_last_bed,bed_gradient):
    n=len(ids);owner,other,bid,offset,faces,signs=directed_geometry(data,n)
    use_sed=s.settings['refine_sediment'];use_bed=s.settings['refine_bathymetry']
    near=wet.copy();finest=s.levels-1
    material=np.asarray(data['bank_material_mask'])
    if material.shape!=(n,) or not np.isin(material,(0,1)).all():
        raise ValueError('Invalid AMR bank material')
    if use_sed:
        indices,bulk,_=_bed_inputs(data,accepted)
        suspended=np.asarray(data['suspended_indices'])
        if (suspended.shape!=(len(bulk),) or suspended.dtype.kind not in 'iu' or
                np.any((suspended<0)|(suspended>=accepted.shape[1]))):
            raise ValueError('Invalid AMR concentration indices')
        concentration=np.zeros((n,len(bulk)),np.float64)
        positive=accepted[:,0]>float(data['dry_tolerance'])
        concentration[positive]=accepted[positive][:,suspended]/accepted[positive,0,None]
        active=accepted[:,indices[0]]/bulk[None,:]
        thickness=np.sum(active,axis=1)
        fractions=np.zeros_like(active)
        occupied=thickness>0
        fractions[occupied]=accepted[occupied][:,indices[0]]/(thickness[occupied,None]*bulk[None,:])
    tree_bed=s._bed(rows,np.arange(s.nn),data) if use_bed else None
    if use_bed and (not np.isfinite(tree_bed).all() or not np.isfinite(s.xy).all() or
                   not np.isfinite(s.area).all() or np.any(s.area<=0)):
        raise ValueError('Invalid AMR ancestor bed geometry')
    for c,node in enumerate(ids):
        for k in range(offset[c],offset[c+1]):
            f=faces[k];j=other[f] if signs[k]>0 else owner[f]
            if j>=0:
                near[c]=near[c] or wet[j]
                if use_sed and near[c]:
                    if np.any(np.abs(concentration[c]-concentration[j])>s.c_tolerance):
                        raw[node]=finest;cause[c]|=1<<4
                    if material[c] and abs(bed[c]-bed[j])>s.dz_tolerance:
                        raw[node]=finest;cause[c]|=1<<5
                    if thickness[c]>0 and thickness[j]>0 and np.any(
                            np.abs(fractions[c]-fractions[j])>s.mix_tolerance):
                        raw[node]=finest;cause[c]|=1<<6
            elif s.settings['refine_walls'] and wet[c] and bid[f]==-1:
                raw[node]=finest;cause[c]|=1<<9
        if use_bed and near[c]:
            change=0.
            for child in s.children[node]:
                residual=tree_bed[child]-bed[c]-np.dot(bed_gradient[c],s.xy[child]-s.xy[node])
                change=max(change,abs(residual))
            for k in range(offset[c],offset[c+1]):
                f=faces[k];j=other[f] if signs[k]>0 else owner[f]
                if j<0:continue
                residual=bed[j]-bed[c]-np.dot(bed_gradient[c],s.xy[ids[j]]-s.xy[node])
                change=max(change,abs(residual))
            if change>s.bed_tolerance:raw[node]=finest;cause[c]|=1<<10
        if use_sed and abs(bed[c]-selector_last_bed[node])>s.dz_tolerance:
            raw[node]=finest;cause[c]|=1<<11
    ancestor_raised=np.zeros(s.nn,bool)
    if use_bed:
        moments=np.zeros((s.nn,5),np.float64);slope=np.zeros((s.nn,2),np.float64)
        wet_parent=np.zeros(s.nn,bool)
        for c,node in enumerate(ids):
            parent=s.parent[node]
            while parent>=0:
                x,y=s.xy[node]-s.xy[parent];dz=bed[c]-tree_bed[parent]
                moments[parent]+=s.area[node]*np.asarray([x*x,x*y,y*y,x*dz,y*dz])
                wet_parent[parent]|=near[c];parent=s.parent[parent]
        for node in np.flatnonzero(wet_parent):
            xx,xy,yy,xz,yz=moments[node];det=xx*yy-xy*xy
            if det>1e-12*xx*yy:
                slope[node]=((yy*xz-xy*yz)/det,(xx*yz-xy*xz)/det)
            elif xx+yy>0:slope[node]=(xz/(xx+yy),yz/(xx+yy))
        before=raw.copy()
        for c,node in enumerate(ids):
            parent=s.parent[node]
            while parent>=0:
                if wet_parent[parent] and abs(bed[c]-tree_bed[parent]-np.dot(
                        slope[parent],s.xy[node]-s.xy[parent]))>s.bed_tolerance:
                    raw[parent]=max(raw[parent],s.level[parent]+1)
                parent=s.parent[parent]
        ancestor_raised=raw>before
        for c,node in enumerate(ids):
            parent=node
            while parent>=0:
                if ancestor_raised[parent]:cause[c]|=1<<13
                parent=s.parent[parent]
    moving|=(cause&((1<<4)|(1<<5)|(1<<6)|(1<<11)))!=0
    return dict(near_water=near,ancestor_raised=ancestor_raised)
