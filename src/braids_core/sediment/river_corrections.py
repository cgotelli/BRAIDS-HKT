"""Selected FLOW slope/hiding operators on the shared river owner.

One generated OpenMP/CUDA body reuses the audited resident constitutive
operators. Only launch/index syntax is adapted; FLOW source is never changed.
Evolving surface composition is refreshed once per bed-face evaluation.
"""
from copy import deepcopy
from pathlib import Path
from types import MethodType
import numpy as np

from .bend_methods_v1 import digest
from .river_capacity import replace_once
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

ROOT = Path(__file__).resolve().parents[2]
SLOPE_MODES = dict(zip(("bed_slope.off", "bed_slope.delft_bagnold_ikeda",
    "bed_slope.delft_koch_flokstra", "bed_slope.delft_parker_andrews"), (0, 2, 3, 4)))
HIDING_MODES = dict(zip(("hiding.none", "hiding.egiazaroff", "hiding.ashida_michiue",
    "hiding.parker_klm", "hiding.wu_wang_jia"), (0, 2, 3, 4, 5)))
SLOPE_KEYS = ("alfabs", "alfabn", "ashld", "bshld", "cshld", "dshld", "alfpa", "thcrpa")
NAMES = ("flow_bed_slope_direction", "hiding_exposure_factors", "river_slope_inputs", "river_taurat_faces")


def source():
    prefix = (ROOT/"braids_core/sediment/river_mpm.h").read_text().split("ENTRY void mixture")[0]
    slope = (ROOT/"braids_gpu/flow_bed_slope_runtime.cu").read_text()
    slope = replace_once(slope, 'extern "C" __global__ void', 'ENTRY void')
    slope = replace_once(slope, "const long long count,", "const int count,")
    slope = replace_once(slope,
        "    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;\n    if(i>=count) return;",
        "    LOOP(count) {")
    slope = slope.replace("return;", "continue;") + "\n}\n"
    hiding = (ROOT/"braids_gpu/hiding_exposure_runtime.cu").read_text()
    hiding = replace_once(hiding, "__device__ __forceinline__", "RIVER_LOCAL")
    hiding = replace_once(hiding, 'extern "C" __global__ void', 'ENTRY void')
    hiding = replace_once(hiding, "    const long long nstate,", "    const int nstate,")
    hiding = replace_once(hiding,
        "    const long long state=(long long)blockIdx.x*blockDim.x+threadIdx.x;\n    if(state>=nstate) return;",
        "    LOOP(nstate) {\n    const int state=i;")
    # A failed class must leave this state, not merely continue the grain loop.
    # Put the existing per-state body in a local function instead of rewriting
    # return semantics inside its nested class loops.
    hiding = hiding.replace("ENTRY void hiding_exposure_factors(", "RIVER_LOCAL void hiding_state(int state,")
    hiding = hiding.replace("    LOOP(nstate) {\n    const int state=i;\n", "")
    hiding += """
ENTRY void hiding_exposure_factors(int nstate,int nclass,int mode,const double*active,
 const double*diam,double parker,double wu,double*factors,double*mean,int*status){
 LOOP(nstate){hiding_state(i,nstate,nclass,mode,active,diam,parker,wu,factors,mean,status);}
}
"""
    return prefix + """
#ifdef __CUDACC__
#define RIVER_LOCAL __device__
#else
#define RIVER_LOCAL inline
using std::isfinite;
#endif
""" + slope + hiding + """
ENTRY void river_slope_inputs(int nf,int mode,const int*owner,const int*neighbour,
 const double*mean,const double*hide,double*fields){
 LOOP(nf){int o=owner[i],j=neighbour[i];if(j<0)j=o;
  fields[7*nf+i]=sqrt(mean[o]*mean[j]);
  fields[8*nf+i]=hide[o];fields[9*nf+i]=hide[j];
  // erosed initializes TAURAT to zero. Positive FLOW selectors 1/2/4/7/13/14 do
  // not write it in eqtran; inventing theta/theta_cr here is NOT equivalent.
  // Negative-selector Van Rijn bundles must supply their own TAURAT.
  if(mode==2)fields[4*nf+i]=0.;
 }
}
ENTRY void river_taurat_faces(int nf,const int*owner,const int*neighbour,
 const double*distance,const double*taurat,double*fields){
 LOOP(nf){int o=owner[i],j=neighbour[i];double t=taurat[o];
  if(j>=0){double d=distance[i]+distance[nf+i];
   t=(distance[nf+i]*t+distance[i]*taurat[j])/d;}
  fields[4*nf+i]=t;
 }
}
"""


class RiverCorrections:
    def __init__(self, backend, folder, *, threads=4):
        if backend == "cpu":
            self.xp = np
        elif backend == "gpu":
            import cupy
            self.xp = cupy
        else:
            raise ValueError("Explicit CPU/GPU required")
        from braids_core.windows_paths import filesystem_path
        path = filesystem_path(folder)/"resolved_corrections.h"
        code = source()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != code:
            raise ValueError("Correction source differs in existing owner folder")
        if not path.exists():
            path.write_text(code)
        self.engine = WaveEngine(backend, threads, source=path, names=NAMES)
        self.identity = dict(runtime=self.engine.identity, binding_sha256=digest(__file__),
            slope_source=digest(ROOT/"braids_gpu/flow_bed_slope_runtime.cu"),
            hiding_source=digest(ROOT/"braids_gpu/hiding_exposure_runtime.cu"))

    def slope(self, fields, method, parameters):
        xp = self.xp
        if fields.ndim != 2 or fields.shape[0] != 10 or fields.dtype != xp.float64 or not fields.flags.c_contiguous:
            raise ValueError("Ten contiguous resident FP64 slope-input rows required")
        mode = SLOPE_MODES[method]
        if mode == 0:
            return fields[:2].copy()
        n = fields.shape[1]
        out, status = xp.zeros((2, n)), xp.empty(n, xp.int32)
        self.engine.launch("flow_bed_slope_direction", n, n, mode, *list(fields),
            *(float(parameters[key]) for key in SLOPE_KEYS), *list(out), status)
        if bool(xp.any(status)):
            raise ValueError("Selected FLOW slope operator rejected face input")
        return out

    def hiding(self, volume, diameters, method, parameters, factors, mean):
        xp = self.xp
        ng, n = volume.shape
        status = xp.empty(n, xp.int32)
        self.engine.launch("hiding_exposure_factors", n, n, ng, HIDING_MODES[method],
            volume, diameters, float(parameters["parker_alpha"]), float(parameters["wu_exponent"]),
            factors, mean, status)
        if bool(xp.any((status != 0) & (status != 2))):
            raise ValueError("Selected hiding operator rejected surface composition")
        # No artificial material: depleted columns have zero donor stock.
        # Only constitutive scratch needs a finite geometric-mean diameter.
        mean[status == 2] = diameters[0]


def install(a, folder, selection, parameters):
    p, xp = a.physics, a.xp
    op = RiverCorrections(a.backend, folder, threads=p.flow.threads)
    n, ng = p.n, len(a.methods.diameter_m)
    diameters = xp.asarray(a.methods.diameter_m)
    slope_hide = xp.ones((ng, n))
    slope_method = selection["bed_slope"]
    hiding = parameters["hiding"]
    intrinsic = a.capacity[0].needs_intrinsic_mixture
    if intrinsic:
        p.intrinsic_grains = xp.empty((3, n))
        grain_status = xp.empty(n, dtype=xp.int32)
    original = a.bed_faces
    for k, cap in enumerate(a.capacity):
        def slope(capacity, fields, k=k):
            if slope_method != "bed_slope.off":
                op.engine.launch("river_slope_inputs", p.nf, p.nf, SLOPE_MODES[slope_method],
                    p.g["owner"], p.g["neighbour"], p.mpm_mean, slope_hide[k], fields)
                if capacity.is_vr1993 and SLOPE_MODES[slope_method]==2:
                    op.engine.launch("river_taurat_faces",p.nf,p.nf,p.g["owner"],p.g["neighbour"],
                                     p.g["distance"],capacity.taurat,fields)
            return op.slope(fields, slope_method, parameters["slope"])
        cap.slope = MethodType(slope, cap)
        cap.identity.update(slope=slope_method, slope_parameters=deepcopy(parameters["slope"]),
            capacity_hiding=selection["capacity_hiding"], slope_hiding=selection["slope_hiding"],
            hiding_parameters=deepcopy(hiding), corrections=op.identity)

    def faces(owner, q, z, hi, water=None):
        explicit_d90 = selection['capacity'].get('grain_sizes_m', {}).get('d90')
        if explicit_d90 is not None:
            p.mpm_d90.fill(explicit_d90)
        elif selection["capacity"]["method"] in ("capacity.mpm", "capacity.van_rijn_1984_modified",
                                               "capacity.wang_fredsoe", "capacity.van_rijn_1993"):
            # Percentile geometry is independent of either hiding selection.
            owner.capacity[0].engine.launch("mixture", n, n, ng, 0,
                owner.columns.composition.weights(owner.columns.available),
                diameters, p.mpm_hiding, p.mpm_d90, p.mpm_mean)
        if intrinsic:
            owner.capacity[0].engine.launch("intrinsic_grains", n, n, ng,
                owner.columns.composition.weights(owner.columns.available), diameters,
                p.intrinsic_grains, grain_status)
            if bool(xp.any(grain_status)):
                raise ValueError("Invalid intrinsic mixture surface composition")
        for cap in owner.capacity:
            if cap.needs_characteristic_shear:
                cap.z0 = p.effective_fields[2]
            if cap.uses_physical_drag:
                # All physical-drag laws consume the selected current CELL
                # resistance. A second formula-name list omitted Wong--Parker
                # and silently supplied the face-averaged surrogate instead.
                if hasattr(p,'prescribed_chezy'):
                    cap.hydraulic_chezy=p.prescribed_chezy
                else:
                    # The constant-Chezy fast path has no dynamic cell
                    # resistance array. Its exact physical value is the
                    # selected constant, not a face-averaged surrogate.
                    if p.identity['river_roughness']['method']!='roughness.chezy':
                        raise ValueError('Selected physical-drag law requires current cell hydraulic resistance')
                    # Keep immutable derived storage off conserved restart
                    # fields; do not alter the schema after the first step.
                    if not hasattr(cap,'_constant_physical_chezy'):
                        cap._constant_physical_chezy=xp.full(n,p.chezy)
                    cap.hydraulic_chezy=cap._constant_physical_chezy
            if intrinsic:
                cap.grain_statistics = p.intrinsic_grains
                # This array is refreshed in-place by effective_inputs before
                # each capacity evaluation, using current physical face depths.
                cap.hydraulic_chezy = p.bedform_chezy
        for k,cap in enumerate(owner.capacity):
            if cap.algebraic_suspension:
                # Columns may publish a new surface-fraction array at commit.
                cap.surface_fraction=owner.columns.surface_fraction[k]
        hiding_stock = owner.columns.available
        if (selection["bed"]["coarse_layer"] and
                parameters["dependent_layers"]["hiding_scope"] == "active_coarse"):
            hiding_stock = hiding_stock + owner.columns.mass[1]
        volume = xp.ascontiguousarray(owner.columns.composition.weights(hiding_stock))
        op.hiding(volume, diameters, "hiding.none" if intrinsic else selection["capacity_hiding"], hiding["capacity"],
                  p.mpm_hiding, p.mpm_mean)
        if selection["slope_hiding"] != "hiding.none":
            op.hiding(volume, diameters, selection["slope_hiding"], hiding["slope"],
                      slope_hide, p.mpm_mean)
        return original(q, z, hi, water)
    a.bed_faces = MethodType(faces, a)
    p.identity["river_corrections"] = dict(runtime=op.identity, parameters=deepcopy(parameters),
        slope=slope_method, capacity_hiding=selection["capacity_hiding"], slope_hiding=selection["slope_hiding"],
        composition=a.columns.composition.identity,
        geometry="actual Cartesian/cropped face; geometric mixture mean; owner at exterior",
        bagnold_stress="Positive scalar FLOW selectors leave erosed TAURAT=0; not Shields/critical",
        intrinsic_surface=("active composition Dg/DgSD/sand; independent of optional slope hiding"
                           if intrinsic else "not consumed"))
    # Expose diagnostics without making scratch part of the conserved restart.
    p.river_slope_hiding = slope_hide
    return a
