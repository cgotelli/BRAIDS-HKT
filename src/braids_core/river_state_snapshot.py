"""Portable accepted fields for explicit same-domain initialization.

This reads data, never restores an old solver or its transactions. Exact restart
continues to authenticate the complete numerical identity separately.
"""
from pathlib import Path
import hashlib,io,json,zipfile,math
import numpy as np
from shapely.geometry import Polygon
from .river_project import canonical,identity,read_json

SCHEMA='braids.accepted-fields.v1'
MAX_BYTES=1024**3


def sha_bytes(value):return hashlib.sha256(value).hexdigest()


def archive(raw):
    if len(raw)>MAX_BYTES:raise ValueError('Saved-state archive exceeds 1 GiB')
    arrays={}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        rows=z.infolist();names=[r.filename for r in rows]
        if len(rows)>512 or len(set(names))!=len(names) or sum(r.file_size for r in rows)>MAX_BYTES:
            raise ValueError('Duplicate or oversized saved-state archive')
        for row in rows:
            if not row.filename.endswith('.npy') or '/' in row.filename or '\\' in row.filename:raise ValueError('Unexpected saved-state member')
            data=z.read(row.filename);stream=io.BytesIO(data);version=np.lib.format.read_magic(stream)
            reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}.get(version)
            if reader is None:raise ValueError('Unsupported saved-state array encoding')
            shape,fortran,dtype=reader(stream,max_header_size=10000);count=math.prod(shape)
            if dtype.hasobject or count*dtype.itemsize!=len(data)-stream.tell():raise ValueError('Invalid or truncated saved-state array; object arrays are forbidden')
            arrays[row.filename[:-4]]=np.frombuffer(data,dtype=dtype,count=count,offset=stream.tell()).reshape(shape,order='F' if fortran else 'C').copy()
    return arrays


def array_hash(a):
    a=np.ascontiguousarray(a)
    return sha_bytes(canonical(dict(shape=list(a.shape),dtype=a.dtype.str)).encode()+a.tobytes())


def encode(meta,arrays):
    from braids_gpu.atomic_output import arrays_to_bytes
    meta=dict(meta,hashes={k:array_hash(v) for k,v in arrays.items()})
    return arrays_to_bytes(metadata=np.array(canonical(meta)),**arrays)


def decode(raw):
    arrays=archive(raw);meta=read_json(str(arrays.pop('metadata')))
    if meta.get('schema')!=SCHEMA or set(meta.get('hashes',{}))!=set(arrays):raise ValueError('Current accepted-field archive required')
    if any(array_hash(v)!=meta['hashes'][k] for k,v in arrays.items()):raise ValueError('Saved-state field checksum differs')
    n=len(arrays['area']);off=arrays['offsets'];xy=arrays['vertices']
    if (arrays['q'].shape!=(3,n) or arrays['z'].shape!=(n,) or off.shape!=(n+1,)
        or any(arrays[k].dtype!=np.float64 for k in ('q','z','vertices','area'))
        or off.dtype.kind not in 'iu' or off[0]!=0 or off[-1]!=len(xy) or np.any(np.diff(off)<3)
        or xy.ndim!=2 or xy.shape[1]!=2 or np.any(arrays['area']<=0)
        or any(not np.isfinite(v).all() for v in arrays.values()) or np.any(arrays['q'][0]<0)):
        raise ValueError('Invalid saved-state arrays or polygons')
    if type(meta.get('time_s')) not in (int,float) or not np.isfinite(meta['time_s']) or meta['time_s']<0:raise ValueError('Invalid saved-state time')
    for key in ('rock','W'):
        if key in arrays and arrays[key].shape!=(n,):raise ValueError('Invalid saved-state '+key+' shape')
    if 'mass' in arrays:
        mass=arrays['mass'];sv=arrays.get('sv')
        if mass.ndim!=3 or mass.shape[-1]!=n or np.any(mass<0) or sv is None or sv.shape!=(mass.shape[0],n) or np.any(sv<=0) or np.any(sv>1):raise ValueError('Invalid saved-state bed inventory')
    if 'suspended' in arrays and (arrays['suspended'].ndim!=2 or arrays['suspended'].shape[-1]!=n or np.any(arrays['suspended']<0)):raise ValueError('Invalid saved-state suspended mass')
    if 'jc_state' in arrays and (arrays['jc_state'].shape not in ((6,n),(7,n)) or np.any(arrays['jc_state'][3:5]<0)):raise ValueError('Invalid saved-state Jiang-Cao inventory')
    return meta,arrays


def polygons(arrays):return [arrays['vertices'][a:b] for a,b in zip(arrays['offsets'][:-1],arrays['offsets'][1:])]


def fluid_domain(project):
    s=project['settings'];p=Polygon(s['domain']['vertices_xy'])
    for row in s['solids']:p=p.difference(Polygon(row['points_xy']))
    if not p.is_valid or p.is_empty:raise ValueError('Invalid fluid domain')
    return p


def same_domain(source,target):
    if source['settings']['domain']['spatial_reference']!=target['settings']['domain']['spatial_reference']:
        from osgeo import osr
        a=osr.SpatialReference();b=osr.SpatialReference()
        if a.SetFromUserInput(source['settings']['domain']['spatial_reference']) or b.SetFromUserInput(target['settings']['domain']['spatial_reference']) or not a.IsSame(b):
            raise ValueError('Saved state and target must use the same domain coordinate reference system')
    if not fluid_domain(source).equals(fluid_domain(target)):
        raise ValueError('Saved state requires an unchanged physical domain, including internal solids')


def pack(project,mesh,arrays,time_s,*,origin=None):
    shapes=[np.asarray(c.vertices_xy,float) for c in mesh.cells]
    values={k:np.asarray(v).copy() for k,v in arrays.items()}
    values.update(vertices=np.concatenate(shapes),offsets=np.r_[0,np.cumsum([len(p) for p in shapes])].astype(np.int64),
        area=np.array([c.area_m2 for c in mesh.cells]))
    raw=encode(dict(schema=SCHEMA,project=project,time_s=float(time_s),origin=origin or {}),values)
    decode(raw);return raw


def capture(project,mesh,owner,*,jiang_cao=False):
    if jiang_cao:
        s=owner.solver;st=owner.a.host(s.state).copy()
        arrays=dict(q=st[:3],z=st[5],jc_state=st,rock=st[5]-st[4]/s.B)
    else:
        host=owner.host;c=owner.columns;p=owner.physics
        arrays=dict(q=host(owner.q),z=host(owner.z),mass=host(c.mass),sv=host(c.sv),rock=host(c.rock),
                    suspended=host(p.suspended_mass),W=host(p.W))
    return pack(project,mesh,arrays,owner.time)


def _location(path,metadata):
    side=path.with_suffix('.json')
    if side.is_file():
        d=read_json(side.read_text(encoding='utf-8'))
        if d.get('schema')!='braids.checkpoint-location.v1' or d.get('sha256')!=sha_bytes(path.read_bytes()):raise ValueError('Checkpoint sidecar checksum differs')
        prepared=(path.parent/d['prepared']).resolve()
    elif (path.parent/'RUN_CONTEXT.json').is_file():
        d=read_json((path.parent/'RUN_CONTEXT.json').read_text(encoding='utf-8'));prepared=(path.parent/d['input']).resolve()
    else:raise ValueError('This checkpoint needs its original .json sidecar/prepared folder. Select it in the original project; newer checkpoints are self-contained.')
    receipt=read_json((prepared/'PREPARED.json').read_text(encoding='utf-8'))
    project=read_json((prepared/'PROJECT.json').read_text(encoding='utf-8'))
    if identity(receipt)!=metadata['prepared_id'] or identity(project)!=metadata['project_id'] or receipt['runtime_id']!=metadata['runtime_id']:
        raise ValueError('Checkpoint source preparation identity differs')
    for name,h in receipt['files'].items():
        member=(prepared/name).resolve()
        if not member.is_relative_to(prepared) or sha_bytes(member.read_bytes())!=h:raise ValueError('Source preparation checksum differs')
    return prepared,project


def _tree(prepared):
    """Read a verified geometry asset; a geometry-only rebuild is a fallback."""
    from braids_desktop.preparation import decode_tree
    profile=read_json((prepared/'core/prepared/profile.json').read_text(encoding='utf-8'))
    runtime=read_json((prepared/'PREPARED.json').read_text(encoding='utf-8'))['runtime_id']
    for root in prepared.parents:
        folder=root/'model_layers/preparation_cache/geometry'
        if not folder.is_dir():continue
        for path in folder.glob('*/TREE.json'):
            asset=read_json(path.with_name('ASSET.json').read_text(encoding='utf-8'));c=asset['contract']
            if c.get('runtime')==runtime and all(c.get(k)==profile[k] for k in ('domain','assignments','grid','levels','breaklines','solids')):
                if sha_bytes(path.read_bytes())!=asset['sha256']:raise ValueError('Source geometry cache checksum differs')
                return decode_tree(read_json(path.read_text(encoding='utf-8')))
    from .sediment.adaptive_river_worker import hierarchy
    return hierarchy(profile)


def source_from_checkpoint(path):
    """V4 data extraction is explicit initialization, never legacy execution."""
    path=Path(path).resolve();raw=path.read_bytes();saved=archive(raw)
    if 'metadata' in saved:
        # A previously imported, self-contained accepted-fields source.
        decode(raw);return raw
    if set(saved) not in ({'identity','payload'},{'identity','payload','state','initialization'}):raise ValueError('A public BRAIDS checkpoint is required')
    m=read_json(str(saved['identity']));payload=saved['payload'].tobytes()
    if m.get('payload_sha256')!=sha_bytes(payload):raise ValueError('Checkpoint payload checksum differs')
    if 'state' in saved:
        state=saved['state'].tobytes();init=saved['initialization'].tobytes()
        if m.get('schema')!='braids.river-checkpoint.v5':raise ValueError('Current transferable checkpoint format required')
        if m.get('state_sha256')!=sha_bytes(state) or m.get('initialization_sha256')!=sha_bytes(init):raise ValueError('Checkpoint transferable-state checksum differs')
        state_meta,_=decode(state)
        if identity(state_meta['project'])!=m['project_id']:raise ValueError('Checkpoint transferable project differs')
        return state
    if m.get('schema')!='braids.river-checkpoint.v4':raise ValueError('Unsupported checkpoint data format')
    prepared,project=_location(path,m);tree=_tree(prepared)
    from .river_jiang_cao import active
    if active(project['model']):
        from .sediment.jiang_cao_adaptive import archive_read,metadata
        from .sediment.jiang_cao_checkpoint import array_hash as jc_hash
        members=archive_read(payload);header=metadata(members['metadata.json'])
        content=header.pop('content_sha256')
        if identity(header)!=content or any(sha_bytes(members[k])!=h for k,h in header['members'].items()):raise ValueError('Jiang-Cao archive checksum differs')
        fields=archive(members['accepted.npz']);h=read_json(fields.pop('header').tobytes().decode('utf-8'))
        checksum=h.pop('content_sha256')
        if identity(h)!=checksum or any(jc_hash(v)!=h['arrays'][k] for k,v in fields.items()):raise ValueError('Jiang-Cao field checksum differs')
        from .sediment.river_leaf_controller import _geometry_snapshot
        if sha_bytes(repr(_geometry_snapshot(tree,tree.finest)).encode())!=header['identity']['geometry_sha256']:raise ValueError('Source hierarchy geometry differs')
        st=fields['state'];arrays=dict(q=st[:3],z=st[5],jc_state=st);at=h['time_s'];leaves=header['leaves']
    else:
        policy=archive(payload);controller=archive(policy['payload'].tobytes());c=read_json(str(controller['metadata']))
        if sha_bytes(controller['owner'].tobytes())!=c['payload_sha256']:raise ValueError('Saved owner checksum differs')
        if sha_bytes(b''.join(np.asarray(cell.vertices_xy,float).tobytes() for cell in tree.cells))!=c['geometry_sha256']:raise ValueError('Source hierarchy geometry differs')
        owner=archive(controller['owner'].tobytes());columns=read_json(str(owner['columns']));body=archive(owner['river'].tobytes());h=read_json(str(body['metadata']))
        if identity({k:v for k,v in columns.items() if k!='checkpoint_sha256'})!=columns['checkpoint_sha256']:raise ValueError('Saved columns checksum differs')
        if any(sha_bytes(v.tobytes())!=h['hashes'][k] for k,v in body.items() if k!='metadata'):raise ValueError('Saved hydraulic field checksum differs')
        arrays=dict(q=body['saved_q'],z=body['saved_z'],rock=np.asarray(columns['rock']),mass=np.asarray(columns['mass']),sv=np.asarray(columns['sv']))
        for name in ('suspended_mass','W'):
            if name in body:arrays['suspended' if name=='suspended_mass' else name]=body[name]
        at=h['time'];leaves=c['leaves']
    return pack(project,tree.select(tuple(leaves)),arrays,at,origin=dict(checkpoint_sha256=sha_bytes(raw),checkpoint_name=path.name,source_schema=m['schema']))
