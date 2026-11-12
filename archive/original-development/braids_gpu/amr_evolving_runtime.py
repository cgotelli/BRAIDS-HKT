"""Host-only source pairing for evolving banks and resident AMR indicators."""
from pathlib import Path

from .amr_portable_runtime import sha,require

REVISION='evolving_bank_on_demand_indicators_v2'
KERNEL_ABIS={
    'evolving_toe_map.cu':'#define BRAIDS_EVOLVING_TOE_ABI 2',
    'amr_evolving_history.cu':'#define BRAIDS_AMR_EVOLVING_HISTORY_ABI 1',
    'amr_evolving_support.cu':'#define BRAIDS_AMR_EVOLVING_SUPPORT_ABI 1',
    'amr_general_indicators.cu':'#define BRAIDS_AMR_GENERAL_INDICATORS_ABI 2',
}
PYTHON_FILES=(
    'validation/gpu_fwave/evolving_toe_map.py',
    'validation/gpu_fwave/amr_evolving_history.py',
    'validation/gpu_fwave/amr_general_indicators.py',
    'braids_gpu/amr_evolving_bank.py','braids_gpu/amr_face_correspondence.py',
    'braids_gpu/amr_general_selector.py','braids_gpu/amr_on_demand.py',
    'braids_gpu/amr_indicator_reference.py',
    'braids_gpu/amr_catalog_bank.py','braids_gpu/amr_catalog_bank_checkpoint.py',
)


def source_pair(root,kernels):
    root=Path(root)
    require(set(kernels)==set(KERNEL_ABIS),'Evolving AMR kernel inventory differs')
    for name,marker in KERNEL_ABIS.items():
        path=Path(kernels[name])
        require(path.is_file() and not path.is_symlink() and
                sum(line.strip()==marker for line in path.read_text().splitlines())==1,
                'Evolving AMR kernel ABI differs: '+name)
    require(all((root/name).is_file() and not (root/name).is_symlink() for name in PYTHON_FILES),
            'Evolving AMR Python inventory differs')
    return dict(schema=1,revision=REVISION,required_cuda_abi=dict(KERNEL_ABIS),
                cuda_sha256={name:sha(path) for name,path in kernels.items()},
                python_sha256={name:sha(root/name) for name in PYTHON_FILES})


def current_sources(root):
    root=Path(root)
    kernels={name:root/'validation/gpu_fwave'/name for name in KERNEL_ABIS}
    return kernels,source_pair(root,kernels)


def verify(root,kernels,receipt):
    require(source_pair(root,kernels)==receipt,'Evolving AMR source pair differs')
