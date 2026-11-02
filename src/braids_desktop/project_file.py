"""Editable project entry beside its self-contained data directory."""
from copy import deepcopy
import json
from pathlib import Path,PureWindowsPath
from braids_core import river_folder
from braids_core.river_project import identity
from . import workspace

SCHEMA='braids.desktop-project.v1'
SUFFIX='.braids.json'


def default_file(root):
    root=Path(root).resolve()
    return root/(root.name+SUFFIX)


def data_directory(filename):
    filename=Path(filename).resolve()
    name=filename.name[:-len(SUFFIX)] if filename.name.endswith(SUFFIX) else filename.stem
    return filename.parent/(name+'_data')


def write(filename,root,document):
    filename=Path(filename).resolve();root=Path(root).resolve()
    if not root.is_relative_to(filename.parent):raise ValueError('Keep the data folder beside or inside the project file folder.')
    relative=root.relative_to(filename.parent).as_posix();document=deepcopy(document)
    if document['terrain']['dem_path']:
        path=Path(document['terrain']['dem_path'])
        if path.is_absolute():document['terrain']['dem_path']=path.resolve().relative_to(root).as_posix()
    value=dict(schema=SCHEMA,data_folder=relative,project=document)
    river_folder._atomic_json(filename,value)
    return workspace.digest(filename)


def read(filename):
    filename=Path(filename).resolve();value=river_folder.read_json(filename.read_text(encoding='utf-8'))
    if not isinstance(value,dict) or set(value)!={'schema','data_folder','project'} or value['schema']!=SCHEMA:
        raise ValueError('Choose a current BRAIDS .braids.json project file.')
    relative=value['data_folder']
    if (not isinstance(relative,str) or not relative or PureWindowsPath(relative).drive or '\\' in relative
            or Path(relative).is_absolute() or '..' in relative.split('/')):raise ValueError('The project data folder must be a contained relative path.')
    root=(filename.parent/relative).resolve()
    if not root.is_relative_to(filename.parent):raise ValueError('The project data folder escapes the project directory.')
    if not root.is_dir():raise ValueError('The project data folder is missing. Move the project file and its data folder together.')
    document=deepcopy(value['project'])
    from braids_core.river_project import SCHEMA as RIVER_SCHEMA
    if document.get('schema')!=RIVER_SCHEMA:raise ValueError('Only the current river project format is supported.')
    draft=workspace.read_editor(root)
    baseline=None if draft is None else deepcopy(draft['document'])
    if baseline and draft['has_terrain']:baseline['terrain']['dem_path']=Path(baseline['terrain']['dem_path']).relative_to(root).as_posix()
    changed=baseline is None or identity(baseline)!=identity(document)
    if changed:
        from braids_core.river_model import resolve_model
        from .services import _resolve_boundary_state,validate_document
        document['model']=resolve_model(document['model']['request']).document()
        clock=document.get('editor_state',{}).get('checkpoint_clock')
        if clock:
            control=document['settings'];old=(baseline or document)['settings']
            if clock['mode']=='end_time' and (control['checkpoint_interval_s']==old['checkpoint_interval_s'] or control['checkpoint_interval_s']==control['end_time_s']):
                control['checkpoint_interval_s']=control['end_time_s']
            else:clock.update(mode='custom',custom_interval_s=control['checkpoint_interval_s'])
        document=validate_document(_resolve_boundary_state(document))
    if document['terrain']['dem_path']:document['terrain']['dem_path']=str(workspace.member(root,document['terrain']['dem_path']))
    # Drafts are used only when the public configuration has not changed.
    # Hand edits to authoritative fields must not be overwritten by old widgets.
    return root,document,None if changed else draft,changed
