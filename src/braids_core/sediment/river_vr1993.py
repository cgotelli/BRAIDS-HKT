"""Shared current-only van Rijn 1993 producer; coupled admission is still guarded.

No backend fallback, inferred class properties, hidden fraction multiplication,
or automatic total-load folding. All forcing fields have explicit identities.
"""
from pathlib import Path
import numpy as np

from .bend_methods_v1 import digest
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

FIELDS = ("depth", "u", "v", "observation_height", "current_z0", "apparent_z0",
          "d50", "mixture_d90", "solid_density", "water_density", "gravity",
          "viscosity", "settling_velocity", "dstar", "class_critical_stress",
          "bed_critical_stress", "surface_fraction", "mud_fraction",
          "suspended_diameter", "additional_stress", "von_karman", "epsilon",
          "maximum_volume_concentration", "molecular_schmidt")
PARAMETERS = ("IopSus", "AksFac", "RWave", "RDC", "RDW", "IopKCW",
              "EpsPar", "BetaM", "iTauCr")
OUTPUTS = ("raw_bed_x_kg_m_s", "raw_bed_y_kg_m_s", "equilibrium_kg_m3",
           "taurat", "fraction_reference_volume_concentration", "reference_height_m",
           "suspended_diameter_m", "shear_velocity_m_s", "ripple_efficiency",
           "current_stress_pa", "bed_speed_derivative_kg_m2", "raw_bed_magnitude_kg_m_s")
REFERENCE_PARAMETERS = dict(critical_stress_factor=1., molecular_schmidt=1.,
    maximum_volume_concentration=.65, bed_critical_stress_pa=.2, suspended_diameter_m=.0005)
DESKTOP_PARAMETERS = dict(IopSus=0, AksFac=1., RWave=2., RDC=.01,
    RDW=.02, IopKCW=1, EpsPar=0, BetaM=3., iTauCr=1)


def embedded_source():
    """Identical constitutive body, with explicit current river forcing."""
    return ("#define RIVER_VR1993_EMBEDDED\n#define LOCAL RIVER_LOCAL\n"+
        Path(__file__).with_suffix(".h").read_text()+"""
RIVER_LOCAL int river_vr_current(const double*p,double h,double u,double v,double d,
 double d90,double z0,double fraction,double dss,double ws,double*out,double sigmol_override=0.){
 double ds=d*pow(p[14]*9.81/(p[15]*p[15]),.3333);
 // Bulk D90 is measured, not the percentile of a one-class carrier.
 if(p[46]>0.)d90=p[46];
 double theta=ds<=4.?.24/ds:ds<=10.?.14/pow(ds,.64):
  ds<=20.?.04/pow(ds,.1):ds<=150.?.013*pow(ds,.29):.055;
 double rho=p[13]/(1.+p[14]);
 double forcing[24]={fmax(.01,h),u,v,h*double(.368f),z0,z0,
  d,d90,p[13],rho,9.81,p[15],ws,ds,p[38]*rho*p[14]*9.81*d*theta,p[41],
  fraction,0.,dss,0.,double(.41f),double(1.e-6f),p[40],sigmol_override>0.?sigmol_override:p[39]};
 return river_vr1993_point(forcing,1,0,p+29,out);
}
""")


class RiverVR1993:
    def __init__(self, backend, parameters, *, threads=4):
        if backend not in ("cpu", "gpu"):
            raise ValueError("Select cpu or gpu explicitly")
        if set(parameters) != set(PARAMETERS):
            raise ValueError("Every named van Rijn 1993 parameter is required")
        for key, allowed in (("IopSus", (0, 1)), ("IopKCW", (0, 1)),
                             ("EpsPar", (0, 1)), ("iTauCr", (1, 2))):
            value = parameters[key]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value not in allowed:
                raise ValueError("Invalid integer selection: "+key)
        wire = np.asarray([parameters[k] for k in PARAMETERS], dtype=np.float64)
        # Keep the native reference-height clamps and finite BetaM exponent.
        # Wave-only values and RDC when IopKCW=1 are identities, not forcing.
        # Do not invent stricter physics bounds for inactive coefficients.
        if (any(isinstance(v, (bool, np.bool_)) for v in parameters.values())
                or not np.isfinite(wire).all()
                or (parameters["IopKCW"] == 0 and parameters["RDC"] <= 0)):
            raise ValueError("Invalid van Rijn 1993 coefficient")
        self.xp = np if backend == "cpu" else __import__("cupy")
        self.wire = self.xp.asarray(wire)
        source = Path(__file__).with_suffix(".h")
        self.engine = WaveEngine(backend, threads, source=source, names=("vr1993_producer",))
        self.identity = dict(method="capacity.van_rijn_1993", parameters=dict(parameters),
            wrapper_sha256=digest(__file__), runtime=self.engine.identity, public_support=False,
            scope="current-only FLOW 2DH algebraic 20-layer concentration producer",
            capacity_units="mass bed vector without surface fraction; concentration includes fraction",
            direction="physical characteristic velocity; no external secondary rotation",
            fraction_order="reference concentration multiplied before 1e-6 volume cutoff",
            wave_only_parameters=["RWave", "RDW", "EpsPar"],
            state_requirement="IopSus=1 DSS output must become next accepted settling input")

    def evaluate(self, forcing):
        xp = self.xp
        if not isinstance(forcing, xp.ndarray) or forcing.dtype != xp.float64:
            raise ValueError("Resident float64 forcing is required")
        if forcing.ndim != 2 or forcing.shape[0] != len(FIELDS) or not forcing.flags.c_contiguous:
            raise ValueError("Expected contiguous (24, point) forcing")
        n = forcing.shape[1]
        if n == 0:
            raise ValueError("At least one forcing point is required")
        out, status = xp.empty((len(OUTPUTS), n)), xp.empty(n, dtype=xp.int32)
        self.engine.launch("vr1993_producer", n, n, forcing, self.wire, out, status)
        if bool(xp.any(status)):
            raise ValueError("Invalid van Rijn 1993 forcing; no clipped or fallback result")
        return out
