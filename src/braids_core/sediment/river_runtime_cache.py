"""One authenticated inert hydraulic source archive for many leaf meshes.

Only mesh-independent code is shared. Native records, boundary arrays, initial
fields and every live owner remain mesh-specific. Full source verification is
retained; a cache hit is never permission to skip integrity checks.
"""
import json
from pathlib import Path
import re
import tempfile

from braids_gpu.atomic_output import write_json_atomic
from braids_core.windows_paths import filesystem_path
from .river_leaf_io import sha, fingerprint, publish_directory


def prepare_shared(root, mesh_folder, data, cancelled=lambda:None):
    from braids_gpu import general_live_transport as transport
    # This cache belongs only to the backend-neutral inert carrier. It cannot
    # absorb a different sediment/forcing bundle under an old archive identity.
    transport._validate_prepared(data, integrator='ssp33')
    for name in ('bedlaw','exchange_mode','secondary_flow_model','general_avalanche'):
        if int(data[name]) != 0:raise ValueError('Shared river runtime requires inert hydraulic carrier')
    if any(key in data for key in ('delft_forcing_binding',transport.PUBLIC_ADMISSION_KEY)):
        raise ValueError('Shared river runtime cannot cache a typed physical bundle')
    contract={key:int(data[key]) for key in ('cpu_time_integrator','nfrac','nbed_layers')}
    contract.update(schema=1,helper_sha256=sha(__file__),transport_sha256=sha(transport.__file__),
        source_io_sha256=sha(Path(__file__).with_name('river_source_io.py')),integrator='ssp33')
    # Short directory spelling avoids needlessly consuming Windows MAX_PATH.
    # The complete key and contract remain checked; a prefix collision fails.
    key=fingerprint(contract);parent=filesystem_path(root)/'_runtime';parent.mkdir(parents=True,exist_ok=True)
    target=parent/key[:20];cancelled()
    if not target.exists():
        pending=filesystem_path(tempfile.mkdtemp(prefix=key[:20]+'.partial-',dir=parent))
        transport.prepare_sources(pending/'archive',data,integrator='ssp33',cancelled=cancelled);cancelled()
        write_json_atomic(pending/'CONTRACT.json',contract)
        # Preparation is synchronous in the worker. Never overwrite a published
        # archive; an interrupted private directory does not occupy this key.
        publish_directory(pending,target,cancelled)
    elif json.loads((target/'CONTRACT.json').read_text()) != contract:
        raise ValueError('Shared river runtime contract differs')
    write_json_atomic(filesystem_path(mesh_folder)/'RUNTIME.json',dict(schema=1,key=key,
        contract_sha256=sha(target/'CONTRACT.json'),
        sources_sha256=sha(target/'archive/general_live_transport_sources.json')))
    return target/'archive'


def source_folder(mesh_folder):
    """Resolve an authenticated cache-local reference, never arbitrary paths."""
    mesh_folder=filesystem_path(mesh_folder)
    row=json.loads((mesh_folder/'RUNTIME.json').read_text())
    if set(row)!={'schema','key','contract_sha256','sources_sha256'} or row['schema']!=1:
        raise ValueError('Shared river runtime reference differs')
    if not isinstance(row['key'],str) or re.fullmatch('[0-9a-f]{64}',row['key']) is None:
        raise ValueError('Shared river runtime key differs')
    root=mesh_folder.parent/'_runtime';target=root/row['key'][:20]
    if (target.is_symlink() or (target/'archive').is_symlink() or
        not target.resolve().is_relative_to(mesh_folder.parent.resolve())):
        raise ValueError('Shared river runtime escapes prepared cache')
    if (sha(target/'CONTRACT.json')!=row['contract_sha256'] or
        sha(target/'archive/general_live_transport_sources.json')!=row['sources_sha256']):
        raise ValueError('Shared river runtime receipt changed')
    from braids_gpu import general_live_transport as transport
    contract=json.loads((target/'CONTRACT.json').read_text())
    if (fingerprint(contract)!=row['key'] or contract.get('helper_sha256')!=sha(__file__) or
        contract.get('source_io_sha256')!=sha(Path(__file__).with_name('river_source_io.py')) or
        contract.get('transport_sha256')!=sha(transport.__file__)):
        raise ValueError('Shared river runtime implementation changed')
    return target/'archive'
