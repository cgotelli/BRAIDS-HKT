"""Split a named opening into physically connected boundary-face chains.

An opening can be assigned to multiple disjoint channels. A shared coordinate
alone is insufficient to join them: their owner cells must also be the same or
share an interior face. This module contains no hydraulic boundary policy.
"""
import numpy as np
from scipy.spatial import cKDTree


def connected_opening_chains(data, boundary):
    """Return deterministically ordered face chains for one boundary ID."""
    owner=np.asarray(data['owner']); neighbour=np.asarray(data['neighbour'])
    bid=np.asarray(data['bid']); mid=np.asarray(data['mid'],float)
    normals=np.asarray(data['normal'],float); lengths=np.asarray(data['length'],float)
    ids=np.flatnonzero((neighbour<0)&(bid==boundary))
    if not len(ids):raise ValueError('Boundary has no physical open faces')
    if (mid.shape!=(2,len(owner)) or normals.shape!=mid.shape or lengths.shape!=owner.shape
            or np.any(~np.isfinite(mid[:,ids])) or np.any(~np.isfinite(normals[:,ids]))
            or np.any(~np.isfinite(lengths[ids])) or np.any(lengths[ids]<=0)):
        raise ValueError('Invalid boundary face geometry')
    if not np.allclose(np.linalg.norm(normals[:,ids],axis=0),1.,rtol=0,atol=1e-10):
        raise ValueError('Boundary face normals must be unit vectors')
    tangent=np.stack((-normals[1,ids],normals[0,ids]),axis=1)
    centre=mid[:,ids].T;half=.5*lengths[ids,None]*tangent
    endpoints=np.empty((2*len(ids),2));endpoints[0::2]=centre-half;endpoints[1::2]=centre+half
    scale=max(1.,float(np.max(np.abs(endpoints))));tol=max(1e-10,128*np.spacing(scale))
    neighbours=set()
    for a,b in zip(owner[neighbour>=0],neighbour[neighbour>=0]):
        neighbours.add((min(int(a),int(b)),max(int(a),int(b))))
    adjacency=[set() for _ in ids]
    for a,b in sorted(cKDTree(endpoints).query_pairs(tol)):
        i,j=a//2,b//2
        if i==j:raise ValueError('Degenerate opening face')
        oi,oj=int(owner[ids[i]]),int(owner[ids[j]])
        if oi==oj or (min(oi,oj),max(oi,oj)) in neighbours:
            adjacency[i].add(j);adjacency[j].add(i)
    if any(len(a)>2 for a in adjacency):raise ValueError('Branched opening geometry needs explicit boundary sections')
    remaining=set(range(len(ids)));result=[]
    while remaining:
        seed=min(remaining,key=lambda i:int(ids[i]));stack=[seed];component=set()
        while stack:
            i=stack.pop()
            if i in component:continue
            component.add(i);stack.extend(adjacency[i]-component)
        ends=[i for i in component if len(adjacency[i])<=1]
        if not ends:raise ValueError('Closed opening boundary is not a section')
        at=min(ends,key=lambda i:int(ids[i]));ordered=[];previous=-1
        while True:
            ordered.append(int(ids[at]));remaining.remove(at)
            following=adjacency[at]-{previous}
            if not following:break
            previous,at=at,next(iter(following))
        if len(ordered)!=len(component):raise ValueError('Could not order the physical boundary chain')
        result.append(ordered)
    return result
