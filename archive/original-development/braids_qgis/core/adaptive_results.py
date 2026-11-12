"""Authenticated mesh-epoch results, usable without the numerical worker."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import numpy as np
from braids_core.windows_paths import filesystem_path

def read_index(output):
    """Read publication metadata without decoding a solution frame.

    This is not an artifact authenticity check: ``read_frame`` still verifies
    every byte of its selected frame and mesh on every call. A live writer
    publishes the index atomically after those immutable artifacts exist.
    """
    root=filesystem_path(output)/'frames'
    doc=json.loads((root/'INDEX.json').read_text(encoding='utf-8'))
    if (not isinstance(doc,dict) or not isinstance(doc.get('epochs'),dict)
            or not isinstance(doc.get('frames'),list) or not doc['frames']):
        raise ValueError('Result index has no published mesh frames')
    return doc


def read_frame(output,index=-1):
    root=(filesystem_path(output)/'frames').resolve();doc=read_index(output)
    row=doc['frames'][index]
    if (not isinstance(row,dict) or not {'file','epoch','sha256'}.issubset(row)
            or not isinstance(row['epoch'],str) or row['epoch'] not in doc['epochs']):
        raise ValueError('Result frame index differs')
    def load(name,expected,selected=None):
        if (not isinstance(name,str) or not name or Path(name).is_absolute()
                or Path(name).drive or '..' in Path(name).parts):
            raise ValueError('Result checksum/path differs')
        p=(root/name).resolve()
        if not p.is_relative_to(root):raise ValueError('Result checksum/path differs')
        # Hash and decode the same immutable bytes, not two independent file
        # reads. Full-file authentication still covers unused NPZ members.
        payload=p.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError('Result checksum/path differs')
        with np.load(BytesIO(payload),allow_pickle=False) as archive:
            if len(archive.files)!=len(set(archive.files)):
                raise ValueError('Result archive has duplicate array names')
            names=archive.files if selected is None else selected
            if any(name not in archive.files for name in names):
                raise ValueError('Result archive is missing display arrays')
            return {name:archive[name] for name in names}
    data=load(row['file'],row['sha256'],('mesh_epoch','q','z'))
    mesh=load(row['epoch'],doc['epochs'][row['epoch']])
    if not {'leaves','vertices','offsets'}.issubset(mesh):
        raise ValueError('Result mesh is missing geometry arrays')
    leaves,vertices,offsets=mesh['leaves'],mesh['vertices'],mesh['offsets']
    if (leaves.ndim!=1 or leaves.dtype.kind not in 'iu'
            or vertices.ndim!=2 or vertices.shape[1]!=2 or vertices.dtype.kind not in 'fiu'
            or not np.isfinite(vertices).all() or offsets.dtype.kind not in 'iu'
            or offsets.shape!=(len(leaves)+1,) or offsets[0]!=0
            or offsets[-1]!=len(vertices) or np.any(offsets[1:]<=offsets[:-1])):
        raise ValueError('Result mesh geometry shape or dtype differs')
    if (data['mesh_epoch'].shape!=() or data['mesh_epoch'].dtype.kind not in 'US'
            or str(data['mesh_epoch'])!=row['epoch']
            or data['q'].shape!=(3,len(leaves)) or data['z'].shape!=(len(leaves),)):
        raise ValueError('Frame and mesh do not match')
    if (data['q'].dtype.kind!='f' or data['z'].dtype.kind!='f'
            or not np.isfinite(data['q']).all() or not np.isfinite(data['z']).all()):
        raise ValueError('Result hydraulic array dtype or finite values differ')
    h=data['q'][0];speed=np.divide(np.hypot(*data['q'][1:]),h,out=np.zeros_like(h),where=h>1e-6)
    return doc,row,mesh,dict(depth_m=h,velocity_m_s=speed,stage_m=h+data['z'],bed_m=data['z'])
