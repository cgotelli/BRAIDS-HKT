"""Portable editor drafts and imported assets; only validated projects can run."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from braids_core import river_folder
from braids_core.river_project import identity

STATE='model_layers/desktop/EDITOR.json'
ASSETS='model_layers/desktop/ASSETS.json'
SCHEMA='braids.desktop-editor.v1'


def member(root,name):return river_folder._member(Path(root).resolve(),name)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_asset(root,source,*,kind='file'):
    """Copy a data source and its sidecars together, retaining relative references."""
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True);source=Path(source).resolve()
    if not source.is_file():raise FileNotFoundError(source)
    if source.is_relative_to(root):return member(root,source.relative_to(root).as_posix())
    if kind=='raster':files=river_folder._raster_files(source)
    elif kind=='vector':
        from osgeo import gdal
        ds=gdal.OpenEx(str(source),gdal.OF_VECTOR|gdal.OF_READONLY)
        if ds is None:raise ValueError('Cannot read vector source: '+source.name)
        files={Path(p).resolve() for p in (ds.GetFileList() or [])};ds=None;files.add(source)
        if source.suffix.lower()=='.shp':
            files.update(p.resolve() for p in source.parent.glob(source.stem+'.*') if p.suffix.lower() in ('.shp','.shx','.dbf','.prj','.cpg','.qix','.sbn','.sbx'))
        files=sorted(files)
    else:files=[source]
    if any(p.parent!=source.parent or not p.is_file() for p in files):raise ValueError('Export a self-contained dataset before importing')
    rows=[dict(name=p.name,sha256=digest(p)) for p in files]
    if len({p.name.casefold() for p in files})!=len(files):raise ValueError('Input filenames collide')
    key=identity(rows);relative='model_layers/imports/'+key;destination=member(root,relative)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        with tempfile.TemporaryDirectory(prefix='.braids-import-',dir=destination.parent) as temporary:
            ready=Path(temporary)/'bundle';ready.mkdir()
            for path,row in zip(files,rows):
                shutil.copy2(path,ready/path.name)
                if digest(ready/path.name)!=row['sha256'] or digest(path)!=row['sha256']:raise ValueError('Input changed while importing')
            os.rename(ready,destination)
    for row in rows:
        if digest(destination/row['name'])!=row['sha256']:raise ValueError('Imported asset checksum differs')
    inventory=member(root,ASSETS);saved=json.loads(inventory.read_text()) if inventory.exists() else {}
    saved.update({relative+'/'+r['name']:r['sha256'] for r in rows});river_folder._atomic_json(inventory,saved)
    return destination/source.name


def save_editor(root,document,ui,session,*,has_domain,has_terrain,issue=None):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True);document=deepcopy(document)
    if has_terrain:
        local=import_asset(root,document['terrain']['dem_path'],kind='raster')
        document['terrain']['dem_path']=local.relative_to(root).as_posix()
    else:document['terrain']['dem_path']=''
    value=dict(schema=SCHEMA,document=document,ui=deepcopy(ui),session=deepcopy(session),has_domain=has_domain,
               has_terrain=has_terrain,issue=issue,run_project_id=identity(document) if issue is None and has_domain and has_terrain else None)
    payload=dict(value=value,sha256=identity(value))
    revision=member(root,'model_layers/desktop/revisions/'+payload['sha256']+'.json')
    if not revision.exists():river_folder._atomic_json(revision,payload)
    river_folder._atomic_json(member(root,STATE),payload)
    return value


def read_editor(root,*,verify_assets=True):
    root=Path(root).resolve();path=member(root,STATE)
    if not path.is_file():return None
    payload=json.loads(path.read_text(encoding='utf-8'));value=payload['value']
    if value.get('schema')!=SCHEMA or payload['sha256']!=identity(value):raise ValueError('Current desktop draft required or draft checksum differs')
    document=deepcopy(value['document'])
    from braids_core.river_project import SCHEMA as RIVER_SCHEMA
    if document.get('schema')!=RIVER_SCHEMA:raise ValueError('Current project format required')
    if verify_assets:
        inventory=member(root,ASSETS)
        for name,expected in (json.loads(inventory.read_text()) if inventory.exists() else {}).items():
            if digest(member(root,name))!=expected:raise ValueError('Imported input changed: '+name)
    if value['has_terrain']:
        dem=member(root,document['terrain']['dem_path'])
        if not dem.is_file():raise FileNotFoundError(dem)
        document['terrain']['dem_path']=str(dem)
    value=deepcopy(value);value['document']=document;return value


def check_runnable(root,document):
    value=read_editor(root,verify_assets=False)
    if value is not None and (value['issue'] is not None or value['run_project_id']!=identity(document)):
        raise ValueError('The saved editor draft needs correction and a successful Save before Prepare or Run: '+str(value['issue'] or 'parameters changed'))
