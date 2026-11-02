"""Resident FP64 hydraulic WENO/limiter port for isolated fixed-grid research.

Geometry is uploaded once. Each reconstruct() call reads the supplied DEVICE
state and performs primitive conversion, least squares, limiting, virtual-wall
WENO, wet/dry flattening and face reconstruction on the GPU. No expected/oracle
array is uploaded. This implements scheme 6 for four hydraulic primitives only;
sediment, secondary flow, AMR and boundary characteristic calculations are not
implemented here.

If boundary_trace_device is None, kind==3 traces are the raw owner hydraulic
reconstruction and MUST be passed through a boundary solver before flux work.
Supplying boundary_trace_device explicitly applies externally computed boundary
states; only its first three rows on kind==3 faces are read.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OPTIONS = ("--std=c++11", "--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false")
GEOMETRY_FIELDS = (
    "bed", "xy", "mid", "normal", "owner", "neighbour", "bid", "boundary_mode",
    "offset", "faces", "signs", "active", "hydraulic_ls_matrix", "hydraulic_ls_weight",
    "bed_ls_delta", "rectangle_width", "weno_cells", "face_displacement", "wall_triangle",
    "wall_stencil_ids", "wall_stencil_kind", "wall_stencil_normal", "wall_slot",
    "wall_normal", "wall_mean_distance", "wall_box_center",
)
INTEGER_FIELDS = frozenset(("owner", "neighbour", "bid", "boundary_mode", "offset", "faces",
    "signs", "active", "weno_cells", "wall_triangle", "wall_stencil_ids", "wall_stencil_kind", "wall_slot"))


def validate_geometry(data):
    """Reject incomplete/invalid host geometry before allowing CUDA indexing."""
    nc, nf, nadj = len(data["bed"]), len(data["owner"]), len(data["faces"])
    if min(nc, nf, nadj) <= 0:
        raise ValueError("Reconstruction requires nonempty cell/face/CSR geometry")
    shapes = {"bed":(nc,), "xy":(2,nc), "mid":(2,nf), "normal":(2,nf),
        "owner":(nf,), "neighbour":(nf,), "bid":(nf,), "boundary_mode":(nf,),
        "offset":(nc+1,), "faces":(nadj,), "signs":(nadj,), "active":(nc,),
        "hydraulic_ls_matrix":(4,nc), "hydraulic_ls_weight":(nadj,), "bed_ls_delta":(2,nadj),
        "rectangle_width":(2,nc), "weno_cells":(5,2,nf), "face_displacement":(2,2,nf),
        "wall_triangle":(nc,), "wall_stencil_ids":(5,2,nc), "wall_stencil_kind":(5,2,nc),
        "wall_stencil_normal":(2,5,2,nc), "wall_slot":(2,nf), "wall_normal":(2,nc),
        "wall_mean_distance":(nc,), "wall_box_center":(2,nc)}
    result = {}
    for name, shape in shapes.items():
        values = np.asarray(data[name])
        if values.shape != shape:
            raise ValueError(f"{name}: expected {shape}, got {values.shape}")
        if name in INTEGER_FIELDS:
            if values.dtype.kind not in "biu":
                raise ValueError(f"{name}: integer indices required")
            if np.any(values < np.iinfo(np.int32).min) or np.any(values > np.iinfo(np.int32).max):
                raise ValueError(f"{name}: int32 index range exceeded")
            result[name] = np.ascontiguousarray(values, dtype=np.int32)
        else:
            if not np.isfinite(values).all():
                raise ValueError(f"{name}: nonfinite geometry")
            result[name] = np.ascontiguousarray(values, dtype=np.float64)
    for name, upper, minimum in (("owner",nc,0),("neighbour",nc,-1),("faces",nf,0),
                                 ("weno_cells",nc,-1),("wall_stencil_ids",nc,-1),("wall_slot",4,-1)):
        if np.any(result[name] < minimum) or np.any(result[name] >= upper):
            raise ValueError(f"{name}: index out of bounds")
    if result["offset"][0] != 0 or result["offset"][-1] != nadj or np.any(np.diff(result["offset"]) < 0):
        raise ValueError("Invalid CSR offsets")
    if not np.isin(result["signs"],(-1,1)).all() or not np.isin(result["active"],(0,1)).all():
        raise ValueError("Invalid CSR sign/active marker")
    if not np.isin(result["wall_triangle"],(0,1)).all() or not np.isin(result["wall_stencil_kind"],(0,1,2)).all():
        raise ValueError("Invalid wall classification")
    if np.any(result["bid"] < -1):
        raise ValueError("Invalid boundary index")
    # WENO queries require a complete five-point stencil. Partial wall stencils
    # are intentionally allowed: the original CPU code safely skips them.
    valid_weno = result["weno_cells"][0] >= 0
    if np.any(np.any(result["weno_cells"] < 0, axis=0) & valid_weno):
        raise ValueError("Incomplete enabled WENO stencil")
    for cell in range(nc):
        ids = result["faces"][result["offset"][cell]:result["offset"][cell+1]]
        signs = result["signs"][result["offset"][cell]:result["offset"][cell+1]]
        if np.any(np.where(signs>0,result["owner"][ids],result["neighbour"][ids]) != cell):
            raise ValueError("CSR incidence does not match face ownership")
    if np.any((result["wall_triangle"] != 0) & (result["wall_mean_distance"] == 0)):
        raise ValueError("A triangular wall cannot have zero mean distance")
    for name in ("dry", "grav", "dx"):
        if not np.isfinite(data[name]) or data[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    return result


class ResidentReconstruction:
    """Persistent scheme-6 hydraulic reconstruction buffers; no host stage work."""

    def __init__(self, data, source_path=None, *, source=None):
        import cupy as cp
        geometry = validate_geometry(data)
        self.cp = cp
        self.nc, self.nf, self.nadj = len(geometry["bed"]), len(geometry["owner"]), len(geometry["faces"])
        self.geometry = {key:cp.asarray(value) for key,value in geometry.items()}
        self.inputs = self.geometry
        self.controls = {key:np.float64(data[key]) for key in ("dry", "grav", "dx")}
        self.outlet_extension = int(data.get("outlet_extension", 0))
        if self.outlet_extension not in (0,1):
            raise ValueError("outlet_extension must be 0 or 1")
        if source_path is not None and source is not None:
            raise ValueError("Choose source_path or source, not both")
        from braids_core.windows_paths import filesystem_path
        code = filesystem_path(source_path or source or HERE/"reconstruction.cu").read_text(encoding="utf-8")
        names = ("reconstruction_primitive", "reconstruction_cells", "reconstruction_faces")
        self.module = cp.RawModule(code=code, options=OPTIONS, name_expressions=names)
        self.kernels = {name:self.module.get_function(name) for name in names}
        nc,nf=self.nc,self.nf
        self.buffers = {"primitive":cp.empty((4,nc),dtype=cp.float64),
            "grad":cp.empty((4,2,nc),dtype=cp.float64), "vlo":cp.empty((4,nc),dtype=cp.float64),
            "vhi":cp.empty((4,nc),dtype=cp.float64), "wall_values":cp.empty((4,4,nc),dtype=cp.float64),
            "wall_ready":cp.empty((4,nc),dtype=cp.int32), "hydro_constant":cp.empty(nc,dtype=cp.int32),
            "face_value":cp.empty((4,2,nf),dtype=cp.float64), "trace":cp.empty((10,nf),dtype=cp.float64),
            "kind":cp.empty(nf,dtype=cp.int32), "primitive_status":cp.empty(nc,dtype=cp.int32),
            "reconstruction_status":cp.empty(nf,dtype=cp.int32)}
        self.buffers["active"] = self.geometry["active"]
        self._unused_boundary = cp.empty((3,nf),dtype=cp.float64)

    def reconstruct(self, q_device, boundary_trace_device=None, outlet_extension=None):
        """Reconstruct the current device state, with dynamic wet/dry face kinds.

        No synchronization, device-to-host transfer or allocation occurs. Caller
        must reject nonzero status arrays and resolve kind==3 boundaries before
        flux evaluation if boundary_trace_device was not provided. Returned
        arrays are overwritten in place on the next call.
        """
        cp,a,b=self.cp,self.geometry,self.buffers
        if not isinstance(q_device,cp.ndarray) or q_device.shape!=(3,self.nc) or q_device.dtype!=cp.float64 or not q_device.flags.c_contiguous:
            raise ValueError("q_device must be a C-contiguous FP64 GPU (3,nc) array")
        supplied=boundary_trace_device is not None
        boundary=self._unused_boundary if not supplied else boundary_trace_device
        if supplied and (not isinstance(boundary,cp.ndarray) or boundary.ndim!=2 or boundary.shape[0]<3 or boundary.shape[1]!=self.nf or boundary.dtype!=cp.float64 or not boundary.flags.c_contiguous):
            raise ValueError("boundary_trace_device must be C-contiguous FP64 GPU (>=3,nf)")
        extension=self.outlet_extension if outlet_extension is None else int(outlet_extension)
        if extension not in (0,1):raise ValueError("outlet_extension must be 0 or 1")
        nc,nf,nadj=np.int64(self.nc),np.int64(self.nf),np.int64(self.nadj)
        dry,grav,dx=(self.controls[k] for k in ("dry","grav","dx"))
        def launch(name,size,args):self.kernels[name](((size+127)//128,),(128,),args)
        launch("reconstruction_primitive",self.nc,(q_device,a["bed"],a["owner"],a["neighbour"],a["bid"],
            a["offset"],a["faces"],a["signs"],nc,dry,b["primitive"],a["active"],b["primitive_status"]))
        launch("reconstruction_cells",self.nc,(q_device,b["primitive"],a["xy"],a["mid"],a["normal"],
            a["owner"],a["neighbour"],a["bid"],a["boundary_mode"],a["offset"],a["faces"],a["signs"],a["active"],
            a["hydraulic_ls_matrix"],a["hydraulic_ls_weight"],a["bed_ls_delta"],a["rectangle_width"],
            a["weno_cells"],a["face_displacement"],a["wall_triangle"],a["wall_stencil_ids"],a["wall_stencil_kind"],
            a["wall_stencil_normal"],a["wall_slot"],a["wall_normal"],a["wall_mean_distance"],a["wall_box_center"],
            nc,nf,nadj,dry,grav,dx,np.int32(extension),b["grad"],b["vlo"],b["vhi"],b["wall_values"],b["wall_ready"],b["hydro_constant"]))
        launch("reconstruction_faces",self.nf,(q_device,b["primitive"],b["grad"],b["vlo"],b["vhi"],
            b["wall_values"],b["wall_ready"],b["hydro_constant"],a["owner"],a["neighbour"],a["bid"],
            a["wall_slot"],a["weno_cells"],a["face_displacement"],boundary,np.int32(supplied),nc,nf,dry,
            b["face_value"],b["trace"],b["kind"],b["reconstruction_status"]))
        return b


def validate_reconstruction(data, source_path=None):
    """Compare one GPU stage to the frozen CPU oracle; not a speed benchmark.

    CPU-computed boundary states are supplied explicitly here. The GPU receives
    no expected primitive, gradient, limiter, wall, face or update arrays. Full
    stage/trajectory qualification must separately exercise boundary handling.
    """
    reconstruction = ResidentReconstruction(data, source_path=source_path)
    cp = reconstruction.cp
    buffers = reconstruction.reconstruct(cp.asarray(data["q"]), cp.asarray(data["boundary_trace"]))
    cp.cuda.Stream.null.synchronize()
    errors = {}
    statuses = {}
    for name, array in buffers.items():
        actual = cp.asnumpy(array)
        if name.endswith("_status"):
            statuses[name] = dict(values=np.unique(actual).tolist(), zero=bool(np.all(actual == 0)))
            continue
        expected = data["face_values" if name == "face_value" else name]
        difference = np.abs(actual-expected)
        errors[name] = dict(max_absolute=float(difference.max()),
            max_scaled=float(np.max(difference/np.maximum(1.,np.abs(expected)))),
            bitwise_equal=bool(np.array_equal(actual,expected)), finite=bool(np.isfinite(actual).all()))
    passed = all(row["finite"] and row["max_scaled"] <= 2.e-11 for row in errors.values())
    passed &= all(row["zero"] for row in statuses.values())
    return dict(cells=reconstruction.nc, faces=reconstruction.nf, errors=errors, statuses=statuses,
        all_bitwise_equal=all(row["bitwise_equal"] for row in errors.values()), passed=bool(passed),
        outlet_extension=reconstruction.outlet_extension, cpu_boundary_states_supplied=True,
        oracle_arrays_uploaded=False, full_solver=False, precision="float64", timing_claim=None,
        cuda_options=list(OPTIONS), wall_triangles=int(np.count_nonzero(data["wall_triangle"])),
        weno_face_sides=int(np.count_nonzero(data["weno_cells"][0]>=0)),
        ready_wall_slots=int(np.count_nonzero(data["wall_ready"])),
        flattened_cells=int(np.count_nonzero(data["hydro_constant"])))
