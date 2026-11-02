"""Resident CUDA Q9I orchestration; backend choice is explicit and has no fallback."""
from braids_core.sediment.cross_physics_v108 import (
    CrossPhysicsOwnerV108, TimestepArbiterV108,
)


class ResidentTimestepArbiterV108(TimestepArbiterV108):
    def __init__(self):
        import cupy as cp
        super().__init__(cp, backend="implicit_fwave_midpoint_gpu")


class ResidentCrossPhysicsOwnerV108(CrossPhysicsOwnerV108):
    def __init__(self, profile, **kwargs):
        if profile.backend != "implicit_fwave_midpoint_gpu":
            raise ValueError("Resident Q9I owner requires the explicit GPU profile")
        kwargs["arbiter"] = ResidentTimestepArbiterV108()
        super().__init__(profile, **kwargs)


__all__ = ["ResidentCrossPhysicsOwnerV108", "ResidentTimestepArbiterV108"]
