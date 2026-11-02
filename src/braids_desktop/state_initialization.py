"""Nonblocking import and explicit mapping of saved fields in Desktop."""
from pathlib import Path
import numpy as np
from braids_core.river_project import identity,read_json,load_project
from braids_core import river_state_snapshot as snapshot
from braids_core import river_state_initialization as initialize
from braids_gpu.atomic_output import write_bytes_atomic


def is_source(path):
    try:
        with np.load(path,allow_pickle=False) as f:
            return 'metadata' in f.files and read_json(str(f['metadata'])).get('schema')==snapshot.SCHEMA
    except (ValueError,OSError,KeyError):return False


def import_checkpoint(window,path,then=None):
    path=Path(path).resolve()
    if window.project_root is None:
        window.forms.restart.clear()
        window.save(then=lambda:import_checkpoint(window,path,then));return
    root=window.project_root;target=window._collect(validate=False)
    def work():
        raw=snapshot.source_from_checkpoint(path);meta,_=snapshot.decode(raw);snapshot.same_domain(meta['project'],target)
        local=root/'initial_states'/('source_'+snapshot.sha_bytes(raw)+'.npz');local.parent.mkdir(parents=True,exist_ok=True)
        if local.exists():
            if local.read_bytes()!=raw:raise ValueError('Saved-state import changed')
        else:write_bytes_atomic(local,raw)
        return local,meta
    def done(result):
        local,meta=result;window.forms.restart.setText(str(local));window.forms.map_state.setEnabled(False)
        window.forms.state_summary.setText(f"Saved state imported at t = {meta['time_s']:g} s. Prepare the intended setup, then check its mesh. The original checkpoint and runtime are preserved.")
        window.status.setText('Saved state imported. Prepare the intended settings; mapping is requested only if its mesh differs.')
        if then:then()
    window._background(work,done,'Reading accepted fields and source geometry…')


def check(window,allow_mapping=False,then_run=False):
    path=window.forms.restart.text().strip()
    if not path:window.error('Select a saved checkpoint/state first.');return
    if not is_source(path):
        import_checkpoint(window,path,then=lambda:check(window,allow_mapping,then_run));return
    prepared=window.prepared;root=window.project_root
    if window.dirty:window.error('Save and Prepare the intended settings before checking or mapping the saved state.');return
    if prepared is None or not (prepared/'PREPARED.json').is_file():window.error('Prepare the intended setup before checking or mapping the saved state.');return
    current=load_project(root/'braids_project.json');receipt=read_json((prepared/'PREPARED.json').read_text(encoding='utf-8'))
    if receipt['project_id']!=identity(current):window.error('The setup changed. Prepare it again before mapping the saved state.');return
    window._chain_run=False
    def work():
        key=identity(dict(prepared=identity(receipt),source=snapshot.sha_bytes(Path(path).read_bytes())))
        destination=root/'initial_states'/('initialized_'+key+'.npz')
        return initialize.prepare(prepared,path,destination,allow_mapping=allow_mapping)
    def done(result):
        changed=result['mesh_changed'];window.forms.map_state.setEnabled(result['status']=='mapping_required')
        text=f"Continue at t = {result['time_s']:g} s. Cells: {result['source_cells']:,} → {result['target_cells']:,}. "
        if result['status']=='mapping_required':
            window.forms.state_summary.setText(text+result['message']);window.status.setText(result['message'])
            if then_run:window.error(result['message'])
            return
        text+=('Conservative mesh mapping completed. ' if changed else 'Same cells: copied without interpolation. ')
        text+='Loaded: '+', '.join(result['loaded'])+'. '
        if result['initialized']:text+='Initialized: '+'; '.join(result['initialized'])+'. '
        if result['not_carried']:text+='Not loaded: '+', '.join(result['not_carried'])+'. '
        if result['transition_assumption']!='Same hydraulic state definition.':text+=result['transition_assumption']+' '
        text+=result['histories']
        window.forms.state_summary.setText(text);window.status.setText('Saved state is ready. Press Run to continue from its saved time.')
        if then_run:window._launch_run(initial_state=result['path'])
    window._background(work,done,'Mapping saved state conservatively…' if allow_mapping else 'Comparing saved and prepared mesh…')
