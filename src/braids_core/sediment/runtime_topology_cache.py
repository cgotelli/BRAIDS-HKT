"""Use complete, preparation-pinned topology on fixed-mesh restart only."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np


def cached_restart_tree(prepared, restart, profile, stop=None):
    if prepared is None or restart is None or profile.get('adaptive',False):
        return None
    root=Path(prepared)
    receipt=json.loads((root/'PREPARED.json').read_text(encoding='utf-8-sig'))
    expected=receipt.get('files',{}).get('TOPOLOGY.json')
    if expected is None:
        return None  # Old preparations have no authenticated full topology.
    from braids_desktop.preparation import decode_tree
    from .adaptive_river_worker import check_stop
    check_stop(stop)
    path=root/'TOPOLOGY.json'
    if path.is_symlink() or not path.is_file():
        raise ValueError('Prepared topology is missing or linked')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected:
        raise ValueError('Prepared topology checksum differs')
    tree=decode_tree(json.loads(data))
    check_stop(stop)
    with np.load(restart,allow_pickle=False) as f:
        meta=json.loads(str(f['identity']));raw=f['payload'].tobytes()
    if hashlib.sha256(raw).hexdigest()!=meta['payload_sha256']:
        raise ValueError('Checkpoint payload differs during topology reuse')
    with np.load(io.BytesIO(raw),allow_pickle=False) as f:
        meta=json.loads(str(f['metadata']));raw=f['payload'].tobytes()
    if hashlib.sha256(raw).hexdigest()!=meta['payload_sha256']:
        raise ValueError('Field checkpoint payload differs during topology reuse')
    with np.load(io.BytesIO(raw),allow_pickle=False) as f:
        meta=json.loads(str(f['metadata']));owner=f['owner'].tobytes()
    if hashlib.sha256(owner).hexdigest()!=meta['payload_sha256']:
        raise ValueError('Leaf checkpoint payload differs during topology reuse')
    vertices=hashlib.sha256(b''.join(np.asarray(c.vertices_xy,float).tobytes() for c in tree.cells)).hexdigest()
    if vertices!=meta['geometry_sha256']:
        raise ValueError('Prepared topology differs from checkpoint hierarchy')
    return tree
