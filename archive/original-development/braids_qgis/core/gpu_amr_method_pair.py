"""Host-only admission for explicit historical and current AMR source pairs."""
from __future__ import annotations
import hashlib
from pathlib import Path

LEGACY_REVISION = 'inventory_bounded_stage_guard_stratigraphy_v5'
PREVIOUS_REVISION = 'inventory_bounded_symmetric_stratigraphy_v6'
REVISION = 'inventory_bounded_secondary_outlet_pair_v7'
CUDA_ABIS = {
    'toe_supply.cu': '#define BRAIDS_TOE_INTERFACE_ABI 4',
    'bank_undercut.cu': '#define BRAIDS_UNDERCUT_INTERFACE_ABI 3',
    'toe_face_diagnostic.cu': '#define BRAIDS_TOE_DIAGNOSTIC_ABI 2',
    'live_normal_section.cu': '#define BRAIDS_NORMAL_ENDPOINT_ROOT_ABI 2',
    'sediment_transport.cu': '#define BRAIDS_SEDIMENT_STAGE_GUARD_ABI 1',
    'sediment_stratigraphy.cu': '#define BRAIDS_STRATIGRAPHY_EXHUMATION_POLICY 3',
    'secondary_transport.cu': '#define BRAIDS_SECONDARY_OUTLET_ABI 1',
}
PYTHON_FILES = {'validation/gpu_fwave/'+name for name in (
    'toe_supply.py','bank_undercut.py','toe_face_diagnostic.py','coupled_solver.py',
    'live_normal_section.py','live_normal_boundary.py','live_critical_section.py',
    'normal_outlet_boundary.py','sediment_transport.py','sediment_stratigraphy.py',
    'secondary_transport.py')}


def verify_method_pair(root, source, manifest, *, revision=REVISION):
    """Verify authenticated metadata without importing solver or CUDA modules."""
    if revision not in (REVISION,PREVIOUS_REVISION,LEGACY_REVISION):
        raise ValueError('Unknown GPU AMR shared sediment source pairing revision')
    abis=dict(CUDA_ABIS)
    python_files=set(PYTHON_FILES)
    if revision!=REVISION:
        # Only explicitly selected historical revisions have the smaller
        # inventory. A malformed current pair may never fall back to them.
        abis.pop('secondary_transport.cu')
        python_files.remove('validation/gpu_fwave/secondary_transport.py')
    if revision==LEGACY_REVISION:
        abis['sediment_stratigraphy.cu']='#define BRAIDS_STRATIGRAPHY_EXHUMATION_POLICY 2'
    pair = manifest.get('interface_source_pair', {})
    if (not isinstance(pair, dict) or set(pair) != {'schema','revision','required_cuda_abi',
            'python_sha256','cuda_sha256'} or pair.get('schema') != 1 or
            pair.get('revision') != revision or pair.get('required_cuda_abi') != abis or
            not isinstance(pair.get('python_sha256'),dict) or
            not isinstance(pair.get('cuda_sha256'),dict) or
            set(pair['python_sha256']) != python_files or set(pair['cuda_sha256']) != set(abis) or
            manifest.get('intentional_cuda_revisions') != sorted(abis)):
        raise ValueError('GPU AMR worker shared sediment source pairing differs')
    root, source = Path(root), Path(source)
    for name, marker in abis.items():
        path = source/name
        if (not path.is_file() or path.is_symlink() or
                hashlib.sha256(path.read_bytes()).hexdigest() != pair['cuda_sha256'][name] or
                sum(line.strip() == marker for line in path.read_text(encoding='utf-8').splitlines()) != 1):
            raise ValueError('GPU AMR shared sediment kernel pairing differs: '+name)
    for name in python_files:
        path = root/name
        if (not path.is_file() or path.is_symlink() or
                hashlib.sha256(path.read_bytes()).hexdigest() != pair['python_sha256'][name]):
            raise ValueError('GPU AMR shared sediment Python pairing differs: '+name)
