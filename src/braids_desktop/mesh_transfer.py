"""Transfer a verified BRAIDS geometry cache between matching projects."""
import json,zipfile,tempfile,shutil
from pathlib import Path
from braids_core.river_project import identity,worker_profile
from braids_core import river_worker
from braids_core.sediment.river_leaf_io import sha,publish_directory
from . import preparation


def contract(document):
    profile=worker_profile(document)
    return dict(schema=1,runtime=river_worker.source_identity(),helper=sha(preparation.__file__),
        **{k:profile[k] for k in ('domain','assignments','grid','levels','breaklines','solids','local_refinement')})


def export_mesh(document,cache,destination):
    c=contract(document);folder=Path(cache)/'geometry'/identity(c)
    receipt=json.loads((folder/'ASSET.json').read_text(encoding='utf8'));tree=folder/'TREE.json'
    if receipt!={'contract':c,'sha256':sha(tree)}:raise ValueError('Prepared mesh cache checksum or settings differ. Prepare first.')
    destination=Path(destination)
    with tempfile.NamedTemporaryFile(dir=destination.parent,suffix='.zip',delete=False) as f:temp=Path(f.name)
    try:
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as archive:
            archive.write(folder/'ASSET.json','ASSET.json');archive.write(tree,'TREE.json')
        temp.replace(destination)
    finally:temp.unlink(missing_ok=True)


def import_mesh(document,cache,source):
    c=contract(document);root=Path(cache)/'geometry';root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.mesh-import-',dir=root) as temporary:
        folder=Path(temporary)/'ready';folder.mkdir()
        with zipfile.ZipFile(source) as archive:
            if sorted(archive.namelist())!=['ASSET.json','TREE.json']:raise ValueError('Not a BRAIDS mesh archive.')
            for name in ['ASSET.json','TREE.json']:
                info=archive.getinfo(name)
                if info.file_size>(4_000_000 if name=='ASSET.json' else 4_000_000_000):raise ValueError('Mesh archive member is too large.')
                with archive.open(info) as src,(folder/name).open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
        receipt=json.loads((folder/'ASSET.json').read_text(encoding='utf8'))
        if receipt!={'contract':c,'sha256':sha(folder/'TREE.json')}:
            raise ValueError('Mesh does not match this project geometry, refinement or runtime. Use the original settings.')
        preparation.decode_tree(json.loads((folder/'TREE.json').read_text(encoding='utf8')))
        target=root/identity(c)
        if target.exists():
            existing=json.loads((target/'ASSET.json').read_text(encoding='utf8'))
            if existing!=receipt or sha(target/'TREE.json')!=receipt['sha256']:raise ValueError('Existing geometry cache differs; do not overwrite it.')
        else:publish_directory(folder,target,lambda:None)
    return target
