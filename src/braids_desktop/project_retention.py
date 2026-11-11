"""Bound project-generated storage while retaining live/restart dependencies."""
from pathlib import Path
import ctypes
import json
import os
import re
import shutil
from braids_core import river_folder
from braids_core.river_project import identity

TERMINAL={'complete','segment_complete','failed','cancelled'}


def read(path):
    try:return json.loads(path.read_text(encoding='utf8'))
    except (OSError,ValueError):return {}


def _pid_alive(pid):
    if not isinstance(pid,int) or pid<=0:return False
    if os.name=='nt':
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
        kernel.OpenProcess.restype=ctypes.c_void_p
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:return ctypes.get_last_error()==5
        code=ctypes.c_uint32();kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32)]
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        okay=kernel.GetExitCodeProcess(handle,ctypes.byref(code));kernel.CloseHandle(handle)
        return not okay or code.value==259
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
    except PermissionError:return True


def active_marker(root,name):return river_folder._member(Path(root).resolve(),'model_layers/desktop/active_workers/'+name+'.json')


def _member(root,path):
    p=Path(path)
    if not p.is_absolute():p=root/p
    try:return river_folder._member(root,Path(os.path.abspath(p)).relative_to(root).as_posix())
    except (ValueError,OSError):return None


def _children(root,name,prefix):
    parent=_member(root,name)
    if parent is None or not parent.is_dir():return []
    result=[]
    for p in parent.iterdir():
        if p.name.startswith(prefix) and p.is_dir() and _member(root,p)==p:
            result.append(p)
    return sorted(result,key=lambda p:p.name)


def plan_cleanup(project_root,*,protected=(),include_incomplete=False):
    root=Path(project_root).resolve()
    manifest=read(root/'BRAIDS_FOLDER.json')
    if manifest.get('schema')!='braids.river-folder.v1' or not (root/'braids_project.json').is_file():
        raise ValueError('Cleanup requires a BRAIDS project data folder.')
    marker_root=_member(root,'model_layers/desktop/active_workers');stale_markers=[]
    if marker_root and marker_root.exists():
        for marker in marker_root.glob('*.json'):
            if _pid_alive(read(marker).get('pid')):
                return dict(root=str(root),remove=[],keep=[],skipped='A project worker is active.')
            stale_markers.append(marker)
    keep=set();remove=set();sessions=[manifest.get('session',{})]
    editor=read(root/'model_layers/desktop/EDITOR.json').get('value',{})
    sessions.append(editor.get('session',{}))
    selected=list(protected)
    for s in sessions:
        if s.get('restart'):selected.append(s['restart'])
    selected.append(editor.get('ui',{}).get('controls',{}).get('restart',{}).get('value',''))
    selected={p for name in selected if name and (p:=_member(root,name)) and p.exists()}
    keep.update(selected)
    runs=_children(root,'runs','run_');preps=_children(root,'prepared','prepare_')
    if runs:keep.add(runs[-1])
    good=[p for p in preps if (p/'PREPARED.json').is_file()]
    if good:keep.add(good[-1])
    elif preps:keep.add(preps[-1])
    # Preserve the current prepared selection, even after an unsuccessful new
    # preparation; it is still the runnable input exposed by the desktop.
    for s in sessions:
        if s.get('prepared'):
            p=_member(root,s['prepared'])
            if p and p.is_dir() and p.parent==root/'prepared':keep.add(p)
    for p in selected:
        for group in (*runs,*preps):
            if p==group or p.is_relative_to(group):keep.add(group)

    checkpoint_groups=_children(root,'checkpoints','run_')
    checkpoint_sets={}
    for folder in checkpoint_groups:
        entries=[]
        for path in folder.glob('*.npz'):
            location=read(path.with_suffix('.json'))
            if location.get('schema')=='braids.checkpoint-location.v1':entries.append((location.get('time_s',-1),path))
        checkpoint_sets[folder]=entries
    if checkpoint_groups:
        usable=[p for p in checkpoint_groups if checkpoint_sets[p]]
        if usable:keep.add(max(checkpoint_sets[usable[-1]],key=lambda r:r[0])[1])
    # Reference closure: kept checkpoints and result ledgers need their exact
    # prepared input. Never remove it just because another mesh is newer.
    changed=True
    while changed:
        before=len(keep)
        for path in list(keep):
            if path in runs:
                context=read(path/'RUN_CONTEXT.json');name=context.get('input')
                if name:
                    p=_member(root,Path(name) if Path(name).is_absolute() else path/name)
                    if p and p.is_dir():keep.add(p)
            if path.suffix=='.npz':
                location=read(path.with_suffix('.json'))
                if location.get('schema')=='braids.checkpoint-location.v1':
                    for key in ('prepared','run'):
                        if location.get(key):
                            p=_member(root,path.parent/location[key])
                            if p and p.is_dir():keep.add(p)
        changed=len(keep)>before
    for p in runs:
        # Missing/unknown status may indicate an older external worker. It is
        # not safe to assume it has stopped from a timestamp alone.
        if p not in keep and (read(p/'STATUS.json').get('status') in TERMINAL or include_incomplete or any(m.stem==p.name for m in stale_markers)):remove.add(p)
    for p in preps:
        finished=(p/'PREPARED.json').is_file() or read(p/'STATUS.json').get('status') in TERMINAL
        if p not in keep and (p/'PROJECT.json').is_file() and (finished or include_incomplete or any(m.stem==p.name for m in stale_markers)):remove.add(p)
    remove.update(stale_markers)
    for folder,entries in checkpoint_sets.items():
        retained=[p for _,p in entries if p in keep]
        if not retained:
            if entries:remove.add(folder)
        else:
            for _,p in entries:
                if p not in keep:
                    remove.add(p);sidecar=p.with_suffix('.json')
                    if sidecar.exists():remove.add(sidecar)
            keep.add(folder)

    # Keep only assets used by retained preparations. Older app versions did
    # not record usage, so reconstruct their exact geometry key from receipts.
    asset_keep=set();asset_root=root/'model_layers/preparation_cache'
    for prep in [p for p in keep if p in preps]:
        usage=read(prep/'DESKTOP_ASSETS.json')
        for name in usage.get('assets',{}).values():
            p=_member(root,asset_root/name)
            if p:asset_keep.add(p)
        receipt=read(prep/'PREPARED.json');desktop=read(prep/'DESKTOP_PREPARATION.json')
        profile=read(prep/'core/prepared/profile.json')
        keys=('domain','assignments','grid','levels','breaklines','solids','local_refinement')
        if all(k in profile for k in keys) and desktop.get('helper_sha256'):
            contract=dict(schema=1,runtime=receipt.get('runtime_id'),helper=desktop['helper_sha256'],**{k:profile[k] for k in keys})
            geometry=identity(contract);asset_keep.add(asset_root/'geometry'/geometry)
            terrain_root=_member(root,asset_root/'terrain')
            if terrain_root and terrain_root.is_dir():
                for asset in terrain_root.iterdir():
                    c=read(asset/'ASSET.json').get('contract',{})
                    if c.get('geometry')==geometry and receipt.get('terrain_sha256') in c.get('dem',{}).values():asset_keep.add(asset)
    for category in ('geometry','terrain'):
        parent=_member(root,asset_root/category)
        if parent and parent.is_dir():
            for p in parent.iterdir():
                if p not in asset_keep and p.is_dir() and (p/'ASSET.json').is_file():remove.add(p)
    keep.update(asset_keep)
    # Imported sources remain project inputs. Only generated initialized states
    # and standalone editor snapshots have a bounded retention policy.
    initial=_member(root,'initial_states')
    if initial and initial.is_dir():
        for prefix in ('source_','initialized_'):
            paths=sorted(initial.glob(prefix+'*.npz'),key=lambda p:p.stat().st_mtime_ns)
            if paths:keep.add(paths[-1])
            remove.update(p for p in paths if p not in keep)
    revision_root=_member(root,'model_layers/desktop/revisions')
    current_editor=read(root/'model_layers/desktop/EDITOR.json').get('sha256')
    if revision_root and revision_root.is_dir() and current_editor:
        for p in revision_root.glob('*.json'):
            if re.fullmatch('[0-9a-f]{64}',p.stem) and p.stem!=current_editor:remove.add(p)
    kept_logs={p.name+'.log' for p in keep if p in runs or p in preps}
    kept_logs.update(Path(s['log']).name for s in sessions if s.get('log'))
    history=_member(root,'history')
    if history and history.is_dir():
        for p in history.glob('*.log'):
            if re.fullmatch(r'(run|prepare)_\d{8}_\d{6}_[0-9a-f]{8}\.log',p.name) and p.name not in kept_logs:remove.add(p)
    for parent in ('prepared','runs'):
        p=_member(root,parent)
        if p and p.is_dir():
            remove.update(x for x in p.glob('.braids_stop_*') if re.fullmatch(r'\.braids_stop_[0-9a-f]{32}',x.name))
    # Every target must remain an explicit contained project descendant.
    targets=[]
    for p in sorted(remove,key=str):
        safe=_member(root,p)
        if safe is None or safe==root or any(p==k or k.is_relative_to(p) for k in keep):continue
        if any(p.is_relative_to(parent) for parent in remove if parent!=p):continue
        targets.append(p.relative_to(root).as_posix())
    return dict(root=str(root),remove=targets,keep=sorted(str(p.relative_to(root)) for p in keep if p.is_relative_to(root)),skipped=None)


def cleanup_project(project_root,*,protected=(),dry_run=False,include_incomplete=False):
    root=Path(project_root).resolve()
    with river_folder._guard(root):
        plan=plan_cleanup(root,protected=protected,include_incomplete=include_incomplete)
        if dry_run:return plan
        removed=[];errors=[];size=0
        for name in plan['remove']:
            try:
                path=river_folder._member(root,name)
                if path.is_dir():
                    contents=list(path.rglob('*'))
                    if any(p.is_symlink() or getattr(p,'is_junction',lambda:False)() for p in contents):
                        raise ValueError('Linked descendants are not cleanup targets.')
                    count=sum(p.stat().st_size for p in contents if p.is_file())
                    shutil.rmtree(path)
                else:count=path.stat().st_size;path.unlink()
                size+=count;removed.append(name)
            except (OSError,ValueError) as exc:errors.append(dict(path=name,error=str(exc)))
        return dict(**plan,removed=removed,removed_bytes=size,errors=errors)


def prune_live_checkpoints(project_root,run_name,*,protected=()):
    """Drop superseded snapshots only after the newer pair is published."""
    root=Path(project_root).resolve();folder=_member(root,'checkpoints/'+run_name)
    if folder is None or not folder.is_dir():return 0
    with river_folder._guard(root):
        markers=_member(root,'model_layers/desktop/active_workers')
        if markers and markers.exists():
            for marker in markers.glob('*.json'):
                if marker.stem!=run_name and _pid_alive(read(marker).get('pid')):return 0
        pairs=[]
        for path in folder.glob('t_*_step_*.npz'):
            location=read(path.with_suffix('.json'))
            if location.get('schema')=='braids.checkpoint-location.v1' and isinstance(location.get('time_s'),(int,float)):
                pairs.append((location['time_s'],path))
        if len(pairs)<2:return 0
        latest=max(pairs,key=lambda r:r[0])[1]
        pins={p for x in protected if x and (p:=_member(root,x))}
        removed=0
        for _,path in pairs:
            if path==latest or path in pins:continue
            river_folder._member(root,path.relative_to(root).as_posix()).unlink()
            path.with_suffix('.json').unlink();removed+=1
        return removed
