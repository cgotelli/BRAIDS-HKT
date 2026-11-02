"""Resident CUDA bed selection; no host numerical upload/fallback."""
from pathlib import Path
import hashlib
import numpy as np
from braids_core.sediment.amr_suspended_bed_selection_v100 import mode_v100
from braids_core.sediment.suspended_exner_owner_v93 import ROUTE_COMBINE


class ResidentAMRBedSelectorV100:
    def __init__(self, execution, cp):
        self.execution, self.cp = execution, cp
        source = Path(__file__).with_suffix('.cu').read_text(encoding='utf-8')
        self.kernel = cp.RawKernel(source, 'braids_amr_bed_selector_v100',
                                   options=('--std=c++11', '--fmad=false'))
        self.combine = cp.asarray(np.asarray(execution.routes) == ROUTE_COMBINE, dtype=cp.int32)
        self.runtime_sha256 = hashlib.sha256((source+Path(__file__).read_text(encoding='utf-8')+
            execution.fingerprint).encode('utf-8')).hexdigest()

    def launch(self, bed, water, diffused, transfer, cleanup):
        cp = self.cp
        ng, nc = self.execution.fraction_count, self.execution.cell_count
        if any(type(a) is not cp.ndarray or a.dtype != cp.float64 or
               a.shape != (ng, nc) or not a.flags.c_contiguous
               for a in (bed, water, diffused, transfer, cleanup)):
            raise ValueError('v100 CUDA bed selector requires resident contiguous numerical arrays')
        outputs = [cp.empty((ng, nc), dtype=cp.float64) for _ in range(4)]
        status = cp.empty((ng, nc), dtype=cp.int32)
        self.kernel(((nc*ng+127)//128,), (128,), (
            np.int32(nc), np.int32(ng), np.int32(mode_v100(self.execution)),
            self.combine, bed, water, diffused, transfer, cleanup, *outputs, status))
        if bool(cp.any(status).item()):
            raise ValueError('v100 selected suspended-bed method overdraws finite stock')
        return outputs
