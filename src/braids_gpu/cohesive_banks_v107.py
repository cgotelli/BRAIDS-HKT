"""Actual CUDA backend for the Q8b cohesive-bank transaction."""
from __future__ import annotations

import cupy as cp

from braids_core.sediment.cohesive_banks_v107 import (
    CohesiveBankMethodsV107, CohesiveBankParametersV107,
    CohesiveBankRuntimeV107,
)
from braids_core.sediment.fines_v106 import FineMethodsV106, FineParametersV106
from braids_core.sediment.planform_v105 import PlanformMethodsV105, PlanformParametersV105
from .fines_v106 import ResidentFineRuntimeV106
from .planform_v105 import ResidentPlanformRuntimeV105


class ResidentCohesiveBankRuntimeV107(CohesiveBankRuntimeV107):
    """CuPy-resident composition with no CPU numerical fallback."""

    def __init__(self, methods=CohesiveBankMethodsV107(),
                 parameters=CohesiveBankParametersV107(), *,
                 fine_methods=FineMethodsV106(),
                 fine_parameters=FineParametersV106(),
                 planform_parameters=PlanformParametersV105()):
        fine=ResidentFineRuntimeV106(fine_methods,fine_parameters)
        planform=None
        if methods.avalanche:
            planform=ResidentPlanformRuntimeV105(
                PlanformMethodsV105("braids_conservative_wet_dry_avalanche"),
                planform_parameters)
        super().__init__(methods,parameters,fine_runtime=fine,planform_runtime=planform)
        self.device_id=int(cp.cuda.Device().id)
