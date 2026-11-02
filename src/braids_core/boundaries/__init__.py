"""Boundary geometry and hydraulic section calculations, independent of QGIS."""
from .section import (HydraulicSolution, NoHydraulicSolution, Section, SectionError,
                      SectionGeometry, WetInterval, register_sections)
from .inlet import (Face, FaceState, Hydrograph, InletError, InletSpec, InletTarget,
                    build_inlet_target, build_inlet_targets)
from .inlet_io import NativeFace, NativeGrid, write_native_inlets

__all__ = ["Section", "SectionError", "NoHydraulicSolution", "SectionGeometry",
           "HydraulicSolution", "WetInterval", "register_sections", "Face", "FaceState",
           "Hydrograph", "InletError", "InletSpec", "InletTarget", "build_inlet_target",
           "build_inlet_targets", "NativeFace", "NativeGrid", "write_native_inlets"]
__all__ += ["SolidObstacle"]


def __getattr__(name):
    # Geometry archive readers and hydraulic sections do not require GDAL.
    # Keep the public obstacle API, loading its GIS dependency only on demand.
    if name == "SolidObstacle":
        from .cut_cells import SolidObstacle
        globals()[name] = SolidObstacle
        return SolidObstacle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
