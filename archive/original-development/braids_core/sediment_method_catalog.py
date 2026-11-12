"""Versioned sediment-method catalogue shared by CPU, GPU, QGIS and tests.

The catalogue is deliberately pure standard-library code.  It records method
meaning and compatibility separately from backend admission and validation
evidence.  Numerical workers must still validate arrays, geometry and runtime
state; a catalogue entry is never, by itself, permission to execute a route.

The reference selectors and source anchors describe the locked, unmodified
Delft3D4 FLOW 2DH source identified by ``DELFT_SOURCE_COMMIT``.  BRAIDS legacy
integer controls remain explicit so old projects can be migrated without
reinterpreting them as Delft3D selector numbers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Iterable, Mapping


CATALOG_SCHEMA = 1
CATALOG_SCOPE_ID = "delft3d4-flow-2dh-noncohesive-river-v1"
DELFT_SOURCE_COMMIT = "c05ed672f0a7fe81c5b9867ef354cfdd53c25379"
DELFT_SOURCE_LOCK = "validation/sediment/delft3d_current/source.lock.json"
GPU_INTEGRATOR_ID = "implicit_fwave_midpoint"

SCOPE_REQUIRED = "required"
SCOPE_CONDITIONAL = "conditional"
SCOPE_EXCLUDED = "excluded"
SCOPE_REFERENCE_DISABLED = "reference_disabled"
SCOPE_BRAIDS_LEGACY = "braids_legacy"
SCOPE_BRAIDS_EXTENSION = "braids_extension"

ROUTE_PUBLIC = "public"
ROUTE_PUBLIC_LIMITED = "public_limited"
ROUTE_PROTOTYPE = "prototype"
ROUTE_MISSING = "missing"
ROUTE_NOT_APPLICABLE = "not_applicable"
ADMITTED_ROUTE_STATUSES = frozenset((ROUTE_PUBLIC, ROUTE_PUBLIC_LIMITED))

QUALIFIED = "operator_qualified"
FUNCTIONALITY_ONLY = "functionality_only"
HISTORICAL_COMPONENT = "historical_component"
UNQUALIFIED = "unqualified"
NOT_APPLICABLE = "not_applicable"

PHYSICAL_RULE = "physical"
BUNDLE_RULE = "inseparable_bundle"
IMPLEMENTATION_RULE = "implementation"
SCOPE_RULE = "scope"


@dataclass(frozen=True)
class MethodSpec:
    """One stable physical/control identity.

    ``legacy_code`` is a BRAIDS persistence value, never a Delft3D selector.
    ``reference_selector`` is text on purpose: Delft3D uses integers, negative
    formula numbers, strings and interacting boolean flags in different files.
    """

    method_id: str
    component: str
    label: str
    reference_selector: str | None
    legacy_code: int | str | None
    disposition: str
    cpu_route: str
    implicit_gpu_route: str
    qualification: str
    source_id: str
    output_contract: str
    source_anchors: tuple[str, ...]
    requires: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompatibilityRule:
    """Declarative compatibility edge.

    A rule is active when all ``when_all`` identities are selected.  Enforced
    rules require at least one member of every ``requires_any`` group and
    reject every selected ``forbids_any`` member.  Descriptive bundle rules
    use ``enforced=False`` to record a mandatory resolution behavior that a
    later full-profile resolver must expose rather than incorrectly rejecting.
    """

    rule_id: str
    category: str
    when_all: tuple[str, ...]
    requires_any: tuple[tuple[str, ...], ...] = ()
    forbids_any: tuple[str, ...] = ()
    resolution: str = ""
    source_anchors: tuple[str, ...] = ()
    enforced: bool = True


@dataclass(frozen=True)
class ResolvedProfile:
    schema: int
    scope_id: str
    semantic_catalog_id: str
    selected_method_ids: tuple[str, ...]
    backend: str | None
    reasons: tuple[str, ...]
    profile_identity: str

    def as_dict(self) -> dict:
        result = asdict(self)
        result["selected_method_ids"] = list(self.selected_method_ids)
        result["reasons"] = list(self.reasons)
        return result


class SedimentCatalogError(ValueError):
    """Fail-closed catalogue or compatibility error with a stable category."""

    def __init__(self, message: str, *, category: str):
        super().__init__(message)
        self.category = category


EQTRAN = (
    "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
    "morphology_kernel/src/eqtran.f90:326-859"
)
RDTRAFRM = (
    "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
    "morphology_io/src/rdtrafrm.f90:1205-1501"
)
RDMOR = (
    "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
    "morphology_io/src/rdmor.f90"
)
RDMORLYR = (
    "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
    "morphology_io/src/rdmorlyr.f90"
)
RDSed = (
    "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
    "morphology_io/src/rdsed.f90:1520-1528,2143-2187"
)
ADJUST = (
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_kernel/src/compute_sediment/adjust_bedload.f90:303-435"
)
HIDING = (
    "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
    "morphology_kernel/src/comphidexp.f90:76-139"
)
SECONDARY = (
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_io/src/input/rdproc.f90:1088-1138"
)
SETTLING = (
    "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
    "morphology_kernel/src/eqsettle.f90:120-193"
)
BEDFORM = (
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_io/src/input/rdbedformpar.f90:253-590"
)
ROUGHNESS = (
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_io/src/input/rdhyb.f90:78-166"
)
ROUGHNESS_NUMERICS = (
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_io/src/input/rdnum.f90:352-354"
)
ROUGHNESS_SHEAR = (
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_kernel/src/compute/taubot.f90:373-424"
)
ROUGHNESS_CONSTANTS = (
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_kernel/src/inichk/iniphy.f90"
)
SKIN_SHEAR = (
    "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
    "morphology_kernel/src/compbsskin.f90:157-307"
)
TRANB4 = (
    "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
    "morphology_kernel/src/tranb4.f90"
)
TRANB7 = (
    "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
    "morphology_kernel/src/tranb7.f90"
)
TRAB11 = (
    "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
    "morphology_kernel/src/trab11.f90"
)
TRAN9T = (
    "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
    "morphology_kernel/src/tran9t.f90"
)
BED_UPDATE = (
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_kernel/src/compute_sediment/bott3d.f90:938-989"
)


def _m(method_id: str, component: str, label: str, selector: str | None,
       code: int | str | None, disposition: str, cpu: str, gpu: str,
       qualification: str, source_id: str, output: str, *anchors: str,
       requires: tuple[str, ...] = (), notes: tuple[str, ...] = ()) -> MethodSpec:
    return MethodSpec(method_id, component, label, selector, code, disposition,
                      cpu, gpu, qualification, source_id, output, tuple(anchors),
                      requires, notes)


_METHODS = (
    # Hydraulic roughness feeding sediment shear.
    _m("roughness.chezy", "hydraulic_roughness", "Chezy", "Roumet=C", "C",
       SCOPE_REQUIRED, ROUTE_PUBLIC_LIMITED, ROUTE_PROTOTYPE, FUNCTIONALITY_ONLY,
       "d3d4_flow_chezy", "hydraulic_roughness", ROUGHNESS,
       ROUGHNESS_NUMERICS, ROUGHNESS_SHEAR, ROUGHNESS_CONSTANTS),
    _m("roughness.manning", "hydraulic_roughness", "Manning", "Roumet=M", "M",
       SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_PUBLIC, HISTORICAL_COMPONENT,
       "d3d4_flow_manning", "hydraulic_roughness", ROUGHNESS,
       ROUGHNESS_NUMERICS, ROUGHNESS_SHEAR, ROUGHNESS_CONSTANTS),
    _m("roughness.white_colebrook", "hydraulic_roughness",
       "White-Colebrook / Nikuradse", "Roumet=W", "W", SCOPE_REQUIRED,
       ROUTE_PUBLIC_LIMITED, ROUTE_PROTOTYPE, FUNCTIONALITY_ONLY, "d3d4_flow_white_colebrook",
       "hydraulic_roughness", ROUGHNESS, ROUGHNESS_NUMERICS, ROUGHNESS_SHEAR,
       ROUGHNESS_CONSTANTS,
       notes=("Normal FLOW branch with ChzK2D=false.",)),
    _m("roughness.white_colebrook_waqua", "hydraulic_roughness",
       "White-Colebrook / Nikuradse (legacy ChzK2D)",
       "Roumet=W; ChzK2D=true", "W_chzk2d", SCOPE_REQUIRED,
       ROUTE_PUBLIC_LIMITED, ROUTE_PROTOTYPE, FUNCTIONALITY_ONLY,
       "d3d4_flow_white_colebrook_chzk2d", "hydraulic_roughness", ROUGHNESS,
       ROUGHNESS_NUMERICS, ROUGHNESS_SHEAR, ROUGHNESS_CONSTANTS,
       notes=("Legacy WAQUA-compatible 18 log10(12 h/ks) branch.",)),
    _m("roughness.length_2dh_invalid", "hydraulic_roughness", "Roughness length",
       "Roumet=Z", "Z", SCOPE_EXCLUDED, ROUTE_NOT_APPLICABLE,
       ROUTE_NOT_APPLICABLE, NOT_APPLICABLE, "d3d4_flow_roughness_length",
       "rejected_in_2dh", ROUGHNESS, ROUGHNESS_NUMERICS,
       notes=("Locked FLOW rejects roughness length for KMAX=1.",)),

    # Sediment-only stress selection. BsSkin does not replace hydraulic
    # friction, Chezy, z0 or effective velocity; only consumers of RP_TAUB see
    # the skin value.
    _m("shear.total_roughness", "sediment_shear", "Total hydraulic bed shear",
       "BsSkin=false", "total", SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_PUBLIC,
       HISTORICAL_COMPONENT, "d3d4_total_bed_shear", "sediment_bed_shear",
       SKIN_SHEAR, ROUGHNESS_SHEAR),
    _m("shear.soulsby_clarke_skin", "sediment_shear",
       "Soulsby/Clarke skin-friction shear", "BsSkin=true", "skin",
       SCOPE_REQUIRED, ROUTE_PUBLIC_LIMITED, ROUTE_PROTOTYPE, FUNCTIONALITY_ONLY,
       "d3d4_soulsby_clarke_skin_shear", "sediment_bed_shear", SKIN_SHEAR,
       notes=("Uses its own 0.40 coefficient; hydraulic roughness uses 0.41.",
              "It may replace RP_TAUB only for an actual consumer; it never "
              "replaces hydraulic friction or TELEMAC-GAIA law-9 shear.")),

    # Algebraic total-load versus bed+suspended route.
    _m("transport_route.off", "transport_route", "Sediment transport off", None,
       0, SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_PUBLIC, QUALIFIED, "off", "none",
       RDSed),
    _m("transport_route.combine", "transport_route",
       "Bed capacity plus advected suspended fraction", "TotalLoad=false", None,
       SCOPE_REQUIRED, ROUTE_PUBLIC_LIMITED, ROUTE_MISSING, HISTORICAL_COMPONENT,
       "d3d4_tra_combine", "bedload_and_suspended_split", RDSed, EQTRAN),
    _m("transport_route.total", "transport_route", "Algebraic total load",
       "TotalLoad=true", None, SCOPE_REQUIRED, ROUTE_PUBLIC_LIMITED,
       ROUTE_PUBLIC_LIMITED, FUNCTIONALITY_ONLY, "d3d4_tra_bedload",
       "total_load_to_bed_update", RDSed, EQTRAN),

    # Capacity formula inventory.  Existing BRAIDS laws retain codes 0--9.
    _m('capacity.wong_parker_2006','capacity','Wong & Parker (2006)',None,None,
       SCOPE_BRAIDS_EXTENSION,ROUTE_PUBLIC_LIMITED,ROUTE_NOT_APPLICABLE,QUALIFIED,
       'wong_parker_2006_397_single_grain','scalar_bedload_capacity',
       'https://doi.org/10.1061/(ASCE)0733-9429(2006)132:11(1159)',
       notes=('Authors preprint Eq24: coefficient 3.97, critical Shields .0495, exponent 1.5.',
              'One uniform grain, arithmetic mean Dm and local hydraulic bed shear; no MPM form drag.',
              'Desktop193: paired CPU/CUDA equations, wet/dry groin AMR, conservation and exact restart.',
              'Separate from unchanged MPM. No mixture, suspension, implicit or experimental-validation claim.')),
    _m('capacity.camenen_larson_2005','capacity','Camenen & Larson (2005)',None,None,
       SCOPE_BRAIDS_EXTENSION,ROUTE_PUBLIC_LIMITED,ROUTE_NOT_APPLICABLE,QUALIFIED,
       'camenen_larson_2005_current_single_grain','scalar_bedload_capacity',
       'https://doi.org/10.1016/j.ecss.2004.10.019',
       notes=('Steady-current bedload only; D50, total hydraulic shear and Soulsby-Whitehouse critical Shields.',
              'Authors CR-07-1 Eqs49,81. Fixed coefficients 12/4.5, explicit multiplier and molecular viscosity.',
              'Desktop192: shared CPU/CUDA equations, wet/dry groin AMR, conservation and exact restart.',
              'No mixture, waves, suspension, implicit-route or experimental-validation claim.')),
    _m('capacity.recking_2010','capacity','Recking (2010)',None,None,
       SCOPE_BRAIDS_EXTENSION,ROUTE_PUBLIC_LIMITED,ROUTE_NOT_APPLICABLE,QUALIFIED,
       'recking_2010_eq12_single_grain','scalar_bedload_capacity',
       'https://doi.org/10.1029/2009WR008007',
       notes=('Equations 4,12-14; one-grain D84=D50 limit of the main surface-based formula.',
              'Explicit representative reach slope; local physical shear, not a replacement hydraulic resistance model.',
              'Desktop190: explicit CPU/CUDA equations, wet/dry AMR and restart; no implicit-route or experimental claim.')),
    _m('capacity.ancey_recking_2023','capacity','Ancey & Recking (2023)',None,None,
       SCOPE_BRAIDS_EXTENSION,ROUTE_PUBLIC_LIMITED,ROUTE_NOT_APPLICABLE,QUALIFIED,
       'ancey_recking_2023_eq42_single_grain','scalar_bedload_capacity',
       'https://doi.org/10.1016/j.earscirev.2023.104571',
       notes=('Equation 42, physical hydraulic stress and surface d50; one grain, bedload only.',
              'No hard threshold. Laboratory scaling is not proof of natural-river prediction.',
              'Desktop190: explicit CPU/CUDA equations, wet/dry AMR and restart; no implicit-route or experimental claim.')),
    _m("capacity.smart_jaggi", "capacity", "Smart & Jäggi (energy slope)", None, None,
       SCOPE_BRAIDS_EXTENSION, ROUTE_PUBLIC_LIMITED, ROUTE_NOT_APPLICABLE, QUALIFIED,
       "smart_jaggi_energy_slope_single_grain", "scalar_bedload_capacity",
       "https://people.ee.ethz.ch/~basement/baseweb/download/documentation/BMdoc_Reference_Manual_BASEHPC_v4-0-2.pdf#page=27",
       notes=("BASEHPC Eqs 1.29-1.31; physical velocity/depth and total hydraulic drag; not a FLOW selector.",
              "One grain, bedload only, grading factor=1. Explicit CPU/CUDA river route; no implicit-route claim.",
              "Operator and short coupled checks: desktop_smart183; no experimental calibration claim.")),
    _m("capacity.off", "capacity", "Off", None, 0, SCOPE_REQUIRED,
       ROUTE_PUBLIC, ROUTE_PUBLIC, QUALIFIED, "off", "none", EQTRAN),
    _m("capacity.van_rijn_1993", "capacity", "Van Rijn 1993", "TraFrm=-1", 1,
       SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_PROTOTYPE, QUALIFIED,
       "d3d4_van_rijn_1993", "split_bed_suspended_vectors", EQTRAN, RDTRAFRM),
    _m("capacity.van_rijn_2007", "capacity", "Van Rijn 2007 / TRANSPOR2004",
       "TraFrm=-2", 10, SCOPE_REQUIRED, ROUTE_PROTOTYPE, ROUTE_PROTOTYPE,
       FUNCTIONALITY_ONLY, "d3d4_van_rijn_2007", "split_bed_suspended_vectors",
       EQTRAN, RDTRAFRM, requires=("bedform.enabled",)),
    _m("capacity.engelund_hansen", "capacity", "Engelund-Hansen (Delft3D)",
       "TraFrm=1", None, SCOPE_REQUIRED, ROUTE_MISSING, ROUTE_MISSING,
       UNQUALIFIED, "d3d4_engelund_hansen_resolved_hydraulic_chezy",
       "scalar_total_capacity", EQTRAN, RDTRAFRM,
       notes=("The locked FLOW route consumes RP_CHEZY resolved from the "
              "selected hydraulic roughness. Historical independent-Chezy "
              "BRAIDS runs use a separate execution identity.",)),
    _m("capacity.engelund_hansen_constant_chezy_legacy", "capacity",
       "Engelund-Hansen (BRAIDS legacy independent Chezy)", None, 8,
       SCOPE_BRAIDS_LEGACY, ROUTE_PUBLIC, ROUTE_PUBLIC, HISTORICAL_COMPONENT,
       "braids_engelund_hansen_independent_constant_chezy",
       "scalar_total_capacity", EQTRAN,
       notes=("Preserves every integer-only historical law-8 project. It is "
              "not Delft3D FLOW TraFrm=1 forcing semantics.",)),
    _m("capacity.mpm", "capacity", "Meyer-Peter-Mueller (Delft3D)",
       "TraFrm=2", None,
       SCOPE_REQUIRED, ROUTE_PROTOTYPE, ROUTE_PROTOTYPE, FUNCTIONALITY_ONLY,
       "d3d4_mpm_tranb2_resolved_hydraulic_roughness",
       "scalar_total_capacity", EQTRAN, RDTRAFRM,
       notes=("Consumes Chezy resolved from selected hydraulic C/M/W "
              "roughness plus generic hiding where selected; it never "
              "consumes RP_TAUB/BsSkin.",
              "Historical specified-Cf BRAIDS/Zhu runs use a separate "
              "execution identity.",
              "The low-level CPU mode-1 route has a v50 combined capacity-"
              "profile and closure-execution identity with ACal=1, critical "
              "Shields 0.047, and hiding disabled. Product admission remains "
              "withheld until live ACal/hiding integration is completed. The "
              "GPU route also remains a prototype until the same identity is "
              "carried through prepared input and checkpoint readers.")),
    _m("capacity.mpm_specified_cf_legacy", "capacity",
       "Meyer-Peter-Mueller (BRAIDS/Zhu specified Cf)", None, 2,
       SCOPE_BRAIDS_LEGACY, ROUTE_PUBLIC, ROUTE_PUBLIC, HISTORICAL_COMPONENT,
       "braids_zhu_mpm_specified_friction", "scalar_total_capacity", EQTRAN,
       notes=("Preserves integer-only historical law-2 mode-0 projects. It is "
              "not the locked FLOW tranb2 roughness closure.",)),
    _m("capacity.general", "capacity", "Parameterized general bed load",
       "TraFrm=4", 11, SCOPE_REQUIRED, ROUTE_PROTOTYPE, ROUTE_PROTOTYPE,
       FUNCTIONALITY_ONLY, "d3d4_general_bedload", "split_bed_suspended_vectors",
       EQTRAN, RDTRAFRM, TRANB4),
    _m("capacity.van_rijn_1984_modified", "capacity", "Modified Van Rijn 1984",
       "TraFrm=7", 12, SCOPE_REQUIRED, ROUTE_PROTOTYPE, ROUTE_PROTOTYPE,
       FUNCTIONALITY_ONLY, "d3d4_modified_van_rijn_1984", "split_bed_suspended_vectors",
       EQTRAN, RDTRAFRM, TRANB7),
    _m("capacity.soulsby_van_rijn", "capacity", "Soulsby / Van Rijn",
       "TraFrm=11", 13, SCOPE_REQUIRED, ROUTE_PROTOTYPE, ROUTE_PROTOTYPE,
       FUNCTIONALITY_ONLY, "d3d4_soulsby_van_rijn", "split_bed_suspended_vectors",
       EQTRAN, RDTRAFRM, TRAB11),
    _m("capacity.soulsby", "capacity", "Soulsby", "TraFrm=12", 4,
       SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_PROTOTYPE, QUALIFIED,
       "d3d4_soulsby_current", "split_bed_suspended_vectors", EQTRAN, RDTRAFRM),
    _m("capacity.wang_fredsoe", "capacity", "Wang / Fredsoe legacy test",
       "TraFrm=13", 14, SCOPE_REQUIRED, ROUTE_PROTOTYPE, ROUTE_PROTOTYPE,
       FUNCTIONALITY_ONLY, "d3d4_wang_fredsoe_legacy", "split_bed_suspended_vectors",
       EQTRAN, RDTRAFRM, TRAN9T,
       notes=("Enabled in the locked source but never an automatic default.",)),
    _m("capacity.ashida_michiue_generalized", "capacity",
       "Generalized Ashida-Michiue", "TraFrm=14", 3, SCOPE_REQUIRED,
       ROUTE_PUBLIC, ROUTE_PROTOTYPE, QUALIFIED, "d3d4_ashida_michiue_current",
       "scalar_total_capacity", EQTRAN, RDTRAFRM),
    _m("capacity.wilcock_crowe", "capacity", "Wilcock-Crowe", "TraFrm=16", 5,
       SCOPE_REQUIRED, ROUTE_PUBLIC_LIMITED, ROUTE_PROTOTYPE, QUALIFIED,
       "d3d4_wilcock_crowe", "intrinsic_mixture_scalar_capacity", EQTRAN, RDTRAFRM,
       notes=("The current CPU route maps the generic transport calibration "
              "to ACal; a source-authenticated per-formula coefficient "
              "profile is still required for full public admission.",)),
    _m("capacity.wilcock_crowe_modified", "capacity", "Modified Wilcock-Crowe",
       "TraFrm=17", 6, SCOPE_REQUIRED, ROUTE_PUBLIC_LIMITED, ROUTE_PROTOTYPE, QUALIFIED,
       "d3d4_modified_wilcock_crowe", "intrinsic_mixture_scalar_capacity",
       EQTRAN, RDTRAFRM,
       notes=("The current CPU route fixes ThetaC0=0.021 and Alpha0=0.33; "
              "a source-authenticated per-formula coefficient profile is "
              "still required for full public admission.",)),
    _m("capacity.gaeuman", "capacity", "Gaeuman / Trinity River", "TraFrm=18", 7,
       SCOPE_REQUIRED, ROUTE_PUBLIC_LIMITED, ROUTE_PROTOTYPE, QUALIFIED,
       "d3d4_gaeuman", "intrinsic_mixture_scalar_capacity", EQTRAN, RDTRAFRM,
       notes=("The current CPU route fixes ThetaC0=0.03 and Alpha0=0.30; "
              "a source-authenticated per-formula coefficient profile is "
              "still required for full public admission.",)),
    _m("capacity.bijker", "capacity", "Bijker wave-current", "TraFrm=5", None,
       SCOPE_EXCLUDED, ROUTE_NOT_APPLICABLE, ROUTE_NOT_APPLICABLE,
       NOT_APPLICABLE, "d3d4_bijker", "coastal_wave_current", EQTRAN, RDTRAFRM),
    _m("capacity.santoss", "capacity", "SANTOSS current-only 2DH", "TraFrm=-4", None,
       SCOPE_REQUIRED, ROUTE_PROTOTYPE, ROUTE_PROTOTYPE,
       FUNCTIONALITY_ONLY, "d3d4_santoss_current_only", "current_bedload_vector",
       EQTRAN, RDTRAFRM,
       notes=("Q7a implements and tests the no-wave branch. Wave-current SANTOSS "
              "remains outside the noncohesive river route and is rejected, not disabled.",)),
    _m("capacity.van_thiel_van_rijn", "capacity", "Van Thiel / Van Rijn 2008",
       "TraFrm=19", None, SCOPE_EXCLUDED, ROUTE_NOT_APPLICABLE,
       ROUTE_NOT_APPLICABLE, NOT_APPLICABLE, "d3d4_van_thiel_van_rijn_2008",
       "coastal_wave_current", EQTRAN, RDTRAFRM),
    _m("capacity.xbeach_soulsby_van_rijn", "capacity",
       "XBeach Soulsby / Van Rijn", "TraFrm=20", None, SCOPE_EXCLUDED,
       ROUTE_NOT_APPLICABLE, ROUTE_NOT_APPLICABLE, NOT_APPLICABLE,
       "d3d4_xbeach_soulsby_van_rijn", "coastal_wave_current", EQTRAN, RDTRAFRM),
    _m("capacity.asmita", "capacity", "ASMITA", "TraFrm=22", None,
       SCOPE_EXCLUDED, ROUTE_NOT_APPLICABLE, ROUTE_NOT_APPLICABLE,
       NOT_APPLICABLE, "d3d4_asmita", "estuarine_equilibrium_concentration",
       EQTRAN, RDTRAFRM),
    _m("capacity.user_dll", "capacity", "User-defined DLL", "TraFrm=15", None,
       SCOPE_EXCLUDED, ROUTE_NOT_APPLICABLE, ROUTE_NOT_APPLICABLE,
       NOT_APPLICABLE, "d3d4_user_dll", "user_defined", EQTRAN, RDTRAFRM),
    _m("capacity.ackers_white_disabled", "capacity", "Ackers-White",
       "TraFrm=3", None, SCOPE_REFERENCE_DISABLED, ROUTE_NOT_APPLICABLE,
       ROUTE_NOT_APPLICABLE, NOT_APPLICABLE, "d3d4_disabled_ackers_white",
       "hard_error", EQTRAN),
    _m("capacity.bailard_disabled", "capacity", "Bailard", "TraFrm=6", None,
       SCOPE_REFERENCE_DISABLED, ROUTE_NOT_APPLICABLE, ROUTE_NOT_APPLICABLE,
       NOT_APPLICABLE, "d3d4_disabled_bailard", "hard_error", EQTRAN),
    _m("capacity.van_rijn_ribberink_disabled", "capacity",
       "Van Rijn / Ribberink 1994", "TraFrm=8", None, SCOPE_REFERENCE_DISABLED,
       ROUTE_NOT_APPLICABLE, ROUTE_NOT_APPLICABLE, NOT_APPLICABLE,
       "d3d4_disabled_van_rijn_ribberink", "hard_error", EQTRAN),
    _m("capacity.old_silt_disabled", "capacity", "Original MOR silt",
       "TraFrm=9", None, SCOPE_REFERENCE_DISABLED, ROUTE_NOT_APPLICABLE,
       ROUTE_NOT_APPLICABLE, NOT_APPLICABLE, "d3d4_disabled_old_silt",
       "hard_error", EQTRAN),
    _m("capacity.ashida_michiue_original_disabled", "capacity",
       "Original Ashida-Michiue", "TraFrm=10", None, SCOPE_REFERENCE_DISABLED,
       ROUTE_NOT_APPLICABLE, ROUTE_NOT_APPLICABLE, NOT_APPLICABLE,
       "d3d4_disabled_ashida_michiue", "hard_error", EQTRAN),
    _m("capacity.cohesive_family", "capacity", "Cohesive erosion family",
       "TraFrm=-3/-5/21", None, SCOPE_CONDITIONAL, ROUTE_PUBLIC_LIMITED,
       ROUTE_PUBLIC_LIMITED, FUNCTIONALITY_ONLY, "d3d4_cohesive_family",
       "cohesive_exchange", RDTRAFRM,
       notes=("Built-in TraFrm=-3/-5 are connected by Q9 v106; the external "
              "TraFrm=21 DLL interface remains separately fail-closed.",)),
    _m("capacity.gaia_research", "capacity", "TELEMAC-GAIA research law",
       None, 9, SCOPE_BRAIDS_EXTENSION, ROUTE_PUBLIC_LIMITED, ROUTE_MISSING,
       HISTORICAL_COMPONENT, "telemac_gaia_formula_3", "vector_capacity", EQTRAN),

    # Bed-slope direction.  Delft off is ISlope=1; BRAIDS persists it as 0.
    _m("bed_slope.off", "bed_slope", "Off", "ISlope=1", 0, SCOPE_REQUIRED,
       ROUTE_PUBLIC, ROUTE_PUBLIC, FUNCTIONALITY_ONLY, "d3d4_islope1_off",
       "unchanged_transport", ADJUST, RDMOR),
    _m("bed_slope.delft_bagnold_ikeda", "bed_slope",
       "Bagnold longitudinal + Ikeda/Van Rijn transverse", "ISlope=2", 5,
       SCOPE_REQUIRED, ROUTE_PUBLIC_LIMITED, ROUTE_PUBLIC_LIMITED,
       FUNCTIONALITY_ONLY, "d3d4_flow_bagnold_ikeda", "corrected_transport_vector",
       ADJUST, RDMOR),
    _m("bed_slope.delft_koch_flokstra", "bed_slope",
       "Van Bendegom / Koch-Flokstra", "ISlope=3", 3, SCOPE_REQUIRED,
       ROUTE_PUBLIC_LIMITED, ROUTE_PUBLIC_LIMITED, FUNCTIONALITY_ONLY,
       "d3d4_koch_flokstra_cshld_dshld_zero", "corrected_transport_vector",
       ADJUST, RDMOR),
    _m("bed_slope.delft_parker_andrews", "bed_slope", "Parker-Andrews",
       "ISlope=4", 6, SCOPE_REQUIRED, ROUTE_PROTOTYPE, ROUTE_PROTOTYPE,
       FUNCTIONALITY_ONLY, "d3d4_parker_andrews", "corrected_transport_vector",
       ADJUST, RDMOR),
    _m("bed_slope.braids_legacy_bagnold", "bed_slope",
       "BRAIDS legacy capacity/stress-ratio slope", None, 2,
       SCOPE_BRAIDS_LEGACY, ROUTE_PUBLIC, ROUTE_PUBLIC, HISTORICAL_COMPONENT,
       "braids_bagnold_ikeda_capacity_ratio", "corrected_transport_vector", ADJUST),
    _m("bed_slope.gaia_talmon", "bed_slope", "TELEMAC-GAIA Talmon research",
       None, 4, SCOPE_BRAIDS_EXTENSION, ROUTE_PUBLIC_LIMITED, ROUTE_MISSING,
       HISTORICAL_COMPONENT, "telemac_gaia_talmon", "corrected_transport_vector",
       ADJUST),

    # Secondary-flow state and steering.
    _m("secondary.off", "secondary_flow", "Off", "Sub1 excludes I", 0,
       SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_PUBLIC, FUNCTIONALITY_ONLY,
       "d3d4_secondary_off", "none", SECONDARY, EQTRAN),
    _m("secondary.prognostic", "secondary_flow", "Transported spiral intensity",
       "Sub1=I, Equili=N", 2, SCOPE_REQUIRED, ROUTE_PUBLIC_LIMITED,
       ROUTE_PUBLIC_LIMITED, FUNCTIONALITY_ONLY, "d3d4_secondary_prognostic",
       "spiral_state_direction_and_optional_momentum", SECONDARY, EQTRAN),
    _m("secondary.equilibrium", "secondary_flow", "Equilibrium spiral intensity",
       "Sub1=I, Equili=Y", 1, SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_PUBLIC,
       FUNCTIONALITY_ONLY, "d3d4_secondary_equilibrium", "transport_direction",
       SECONDARY, EQTRAN),
    _m("secondary.gaia_research", "secondary_flow", "TELEMAC-GAIA stress addition",
       None, 3, SCOPE_BRAIDS_EXTENSION, ROUTE_PUBLIC_LIMITED, ROUTE_MISSING,
       HISTORICAL_COMPONENT, "telemac_gaia_secondary_stress", "stress_vector", SECONDARY),

    # Hiding/exposure.  Delft IHidExp=1 is persisted as BRAIDS 0.
    _m("hiding.none", "hiding", "None", "IHidExp=1", 0, SCOPE_REQUIRED,
       ROUTE_PUBLIC, ROUTE_PUBLIC, FUNCTIONALITY_ONLY, "d3d4_hiding_none",
       "factor_one", HIDING, RDMOR),
    _m("hiding.egiazaroff", "hiding", "Egiazaroff", "IHidExp=2", 2,
       SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_PROTOTYPE, HISTORICAL_COMPONENT,
       "d3d4_hiding_egiazaroff", "classwise_critical_stress_factor", HIDING, RDMOR),
    _m("hiding.ashida_michiue", "hiding", "Modified Egiazaroff / Ashida-Michiue",
       "IHidExp=3", 3, SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_PROTOTYPE,
       HISTORICAL_COMPONENT, "d3d4_hiding_ashida_michiue",
       "classwise_critical_stress_factor", HIDING, RDMOR),
    _m("hiding.parker_klm", "hiding", "Soehngen-Kellermann-Loy / Parker-KLM",
       "IHidExp=4", 4, SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_PROTOTYPE,
       HISTORICAL_COMPONENT, "d3d4_hiding_parker_klm",
       "classwise_critical_stress_factor", HIDING, RDMOR),
    _m("hiding.wu_wang_jia", "hiding", "Wu-Wang-Jia", "IHidExp=5", 5,
       SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_PROTOTYPE, HISTORICAL_COMPONENT,
       "d3d4_hiding_wu_wang_jia", "classwise_critical_stress_factor", HIDING, RDMOR),

    # Bed composition, layers and sorting.
    _m("bed.fully_mixed", "bed_composition", "Fully mixed bed", "IUnderLyr=1", 1,
       SCOPE_REQUIRED, ROUTE_PUBLIC_LIMITED, ROUTE_MISSING, HISTORICAL_COMPONENT,
       "d3d4_fully_mixed_bed", "bulk_composition", RDMORLYR),
    _m("bed.layered", "bed_composition", "Transport layer and ordered substrate",
       "IUnderLyr=2", 2, SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_PROTOTYPE,
       HISTORICAL_COMPONENT, "d3d4_layered_bed", "classwise_layer_inventory", RDMORLYR),
    _m("fraction_basis.mass", "fraction_basis", "Mass fractions", "IFractions=mass",
       "mass", SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_PROTOTYPE,
       HISTORICAL_COMPONENT, "d3d4_mass_fractions", "classwise_mass", RDMORLYR),
    _m("fraction_basis.solid_volume", "fraction_basis", "Solid-volume fractions",
       "IFractions=volume", "volume", SCOPE_REQUIRED, ROUTE_MISSING,
       ROUTE_MISSING, UNQUALIFIED, "d3d4_solid_volume_fractions",
       "classwise_solid_volume", RDMORLYR),
    _m("transport_layer.constant", "transport_layer", "Constant/spatial thickness",
       "TTLForm=1", "constant", SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_PROTOTYPE,
       HISTORICAL_COMPONENT, "d3d4_ttl_constant", "active_layer_thickness", RDMORLYR,
       requires=("bed.layered",)),
    _m("transport_layer.water_depth", "transport_layer", "Water-depth-scaled thickness",
       "TTLForm=2", "water_depth", SCOPE_REQUIRED, ROUTE_PUBLIC_LIMITED,
       ROUTE_MISSING, HISTORICAL_COMPONENT, "d3d4_ttl_water_depth",
       "active_layer_thickness", RDMORLYR, requires=("bed.layered",)),
    _m("transport_layer.dune_height", "transport_layer", "Dune-height-scaled thickness",
       "TTLForm=3", "dune_height", SCOPE_REQUIRED, ROUTE_MISSING,
       ROUTE_MISSING, UNQUALIFIED, "d3d4_ttl_dune_height", "active_layer_thickness",
       RDMORLYR, requires=("bed.layered", "bedform.enabled")),
    _m("exchange_layer.off", "exchange_layer", "No exchange/coarse layer",
       "exchange layer off", 0, SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_PROTOTYPE,
       HISTORICAL_COMPONENT, "d3d4_exchange_layer_off", "none", RDMORLYR),
    _m("exchange_layer.enabled", "exchange_layer", "Exchange/coarse layer",
       "exchange layer on", 1, SCOPE_REQUIRED, ROUTE_MISSING, ROUTE_MISSING,
       UNQUALIFIED, "d3d4_exchange_coarse_layer", "classwise_layer_inventory",
       RDMORLYR, requires=("bed.layered",)),
    _m("sorting.off", "vertical_sorting", "Vertical sorting off", "sorting=off", 0,
       SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_PROTOTYPE, HISTORICAL_COMPONENT,
       "d3d4_sorting_off", "none", RDMORLYR),
    _m("sorting.discrete_shields", "vertical_sorting", "Discrete Shields mobility",
       "sorting=discrete_shields", 1, SCOPE_REQUIRED, ROUTE_MISSING,
       ROUTE_MISSING, UNQUALIFIED, "d3d4_sorting_discrete_shields",
       "vertical_class_exchange", RDMORLYR, requires=("bed.layered",)),
    _m("sorting.continuous_shields", "vertical_sorting", "Continuous Shields mobility",
       "sorting=continuous_shields", 2, SCOPE_REQUIRED, ROUTE_MISSING,
       ROUTE_MISSING, UNQUALIFIED, "d3d4_sorting_continuous_shields",
       "vertical_class_exchange", RDMORLYR, requires=("bed.layered",)),
    _m("sorting.wilcock_mcardell", "vertical_sorting", "Wilcock-McArdell 1997",
       "sorting=wilcock_mcardell", 3, SCOPE_REQUIRED, ROUTE_MISSING,
       ROUTE_MISSING, UNQUALIFIED, "d3d4_sorting_wilcock_mcardell",
       "vertical_class_exchange", RDMORLYR, requires=("bed.layered",)),
    _m("sorting.transport_hiding", "vertical_sorting", "Transport/hiding-based mobility",
       "sorting=transport_hiding", 4, SCOPE_REQUIRED, ROUTE_MISSING,
       ROUTE_MISSING, UNQUALIFIED, "d3d4_sorting_transport_hiding",
       "vertical_class_exchange", RDMORLYR, requires=("bed.layered",)),
    _m("sorting.active_layer_diffusion", "vertical_sorting", "Active-layer diffusion",
       "Diffusion=constant/spatial", "diffusion", SCOPE_CONDITIONAL,
       ROUTE_MISSING, ROUTE_MISSING, UNQUALIFIED, "d3d4_active_layer_diffusion",
       "vertical_class_diffusion", RDMORLYR, requires=("bed.layered",),
       notes=("Retain as an advanced graded-bed option, not a default.",)),

    # Suspension and settling for the noncohesive river scope.
    _m("suspension.off", "suspension", "Suspended transport off", None, 0,
       SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_PUBLIC, FUNCTIONALITY_ONLY,
       "off", "none", RDSed, EQTRAN),
    _m("suspension.settling_only", "suspension", "Settling/deposition only", None, 1,
       SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_MISSING, HISTORICAL_COMPONENT,
       "braids_settling_only", "suspended_mass_exchange", SETTLING),
    _m("suspension.van_rijn_exchange", "suspension",
       "FLOW 2DH noncohesive equilibrium/adaptation exchange",
       "TRA_COMBINE", 2, SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_MISSING,
       HISTORICAL_COMPONENT, "d3d4_flow2dh_galappatti",
       "equilibrium_suspended_mass_exchange", EQTRAN, SETTLING,
       notes=("The stable method ID predates the source audit. FLOW applies this "
              "generic Galappatti route to every noncohesive TRA_COMBINE formula; "
              "Van Rijn 1993 is only one equilibrium-capacity producer.",)),
    _m("suspension.cohesive_mixed", "suspension", "Cohesive or mixed sand/mud exchange",
       "TraFrm=-3/-5", 3, SCOPE_CONDITIONAL, ROUTE_PUBLIC_LIMITED,
       ROUTE_PUBLIC_LIMITED, FUNCTIONALITY_ONLY, "d3d4_cohesive_mixed_exchange",
       "cohesive_suspended_mass_exchange", RDTRAFRM),
    _m("settling.grain_size", "settling",
       "Grain-size settling with Richardson-Zaki hindering", "SettleFrm=2", 0,
       SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_PROTOTYPE, FUNCTIONALITY_ONLY,
       "d3d4_settlefrm2_dss_richardson_zaki", "hindered_settling_velocity",
       SETTLING,
       notes=("Ordinary FLOW SettleFrm=2 already multiplies the grain-size "
              "fall velocity by max(0,1-Ctot/Cref)^5.",)),
    _m("settling.hindered", "settling",
       "Formula-coupled Van Rijn 2004 DSS settling/flocculation",
       "Van Rijn 2004 DSS_2004", 1, SCOPE_REQUIRED, ROUTE_PROTOTYPE,
       ROUTE_PROTOTYPE, FUNCTIONALITY_ONLY, "d3d4_formula_coupled_dss_2004",
       "formula_coupled_settling_velocity", SETTLING,
       requires=("capacity.van_rijn_2007",),
       notes=("The stable ID predates the source audit. This is not an "
              "independent hindering switch: FLOW selects DSS_2004 with "
              "compatible transport formulas such as TraFrm=-2.",)),
    _m("settling.flocculation_family", "settling", "Fine-sediment/floc settling family",
       "SettleFrm=1/3/4", None, SCOPE_CONDITIONAL, ROUTE_PUBLIC_LIMITED,
       ROUTE_PUBLIC_LIMITED, FUNCTIONALITY_ONLY, "d3d4_flocculation_settling",
       "settling_velocity", SETTLING),

    # Boundary selectors.  Codes are Delft IBedCond/ICmpCond values because
    # BRAIDS has no conflicting public integer namespace for these records.
    *tuple(_m(f"bed_boundary.mode_{code}", "bed_boundary", label,
              f"IBedCond={code}", code, SCOPE_REQUIRED,
              ROUTE_PUBLIC_LIMITED if code in (0, 4, 5, 8) else ROUTE_MISSING,
              ROUTE_MISSING, HISTORICAL_COMPONENT if code in (0, 4, 5, 8) else UNQUALIFIED,
              f"d3d4_bed_boundary_{code}", "boundary_bed_or_transport", RDMOR)
           for code, label in enumerate((
               "Free", "Fixed depth", "Prescribed depth", "Prescribed depth change",
               "Transport including pores", "Transport excluding pores", "Bed level",
               "Bed-level change", "Transport mass"))),
    *tuple(_m(f"composition_boundary.mode_{code}", "composition_boundary", label,
              f"ICmpCond={code}", code, SCOPE_REQUIRED,
              ROUTE_PUBLIC_LIMITED if code == 0 else ROUTE_MISSING,
              ROUTE_MISSING, HISTORICAL_COMPONENT if code == 0 else UNQUALIFIED,
              f"d3d4_composition_boundary_{code}", "boundary_composition", RDMORLYR)
           for code, label in enumerate(("Free", "Fixed", "Prescribed mass fractions",
                                         "Prescribed volume fractions"))),

    # Morphology clock and limit controls are independent controls, not
    # constitutive alternatives.  They remain separate stable identities.
    _m("morph_clock.morfac", "morph_control", "Constant/time-varying MorFac",
       "MorFac", "morfac", SCOPE_REQUIRED, ROUTE_PUBLIC_LIMITED, ROUTE_MISSING,
       HISTORICAL_COMPONENT, "d3d4_morfac", "morphological_clock", RDMOR),
    _m("morph_clock.start_times", "morph_control", "Sediment/bed/composition start times",
       "SedTransStt/BedUpdStt/CmpUpdStt", "start_times", SCOPE_REQUIRED,
       ROUTE_MISSING, ROUTE_MISSING, UNQUALIFIED, "d3d4_morph_start_times",
       "morphological_clock", RDMOR),
    _m("morph_control.update_switches", "morph_control", "Bed/composition update switches",
       "BedUpd/CmpUpd", "update_switches", SCOPE_REQUIRED, ROUTE_MISSING,
       ROUTE_MISSING, UNQUALIFIED, "d3d4_update_switches", "update_policy", RDMOR),
    _m("morph_control.limits", "morph_control",
       "Erosion, entrainment, concentration and bed-change limits",
       "Thresh/NeglectEntrainment/SedThr/DzMax/CaMax/UpdInf", "limits",
       SCOPE_REQUIRED, ROUTE_PUBLIC_LIMITED, ROUTE_MISSING, HISTORICAL_COMPONENT,
       "d3d4_morph_limits", "bounded_update", RDMOR),

    # Bedforms.  One off/enabled identity gates the detailed selector families.
    _m("bedform.off", "bedform_state", "Bedforms off", "Bdf=false", 0,
       SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_PUBLIC, FUNCTIONALITY_ONLY,
       "d3d4_bedforms_off", "none", BEDFORM),
    _m("bedform.enabled", "bedform_state", "Bedforms enabled", "Bdf=true", 1,
       SCOPE_REQUIRED, ROUTE_PUBLIC_LIMITED, ROUTE_PROTOTYPE, FUNCTIONALITY_ONLY,
       "d3d4_bedforms_enabled", "dune_state_and_roughness", BEDFORM),
    *tuple(_m(f"bedform_height.{key}", "bedform_height", label, selector, key,
              SCOPE_REQUIRED, ROUTE_PROTOTYPE, ROUTE_PROTOTYPE, FUNCTIONALITY_ONLY,
              f"d3d4_bedform_height_{key}", "equilibrium_dune_height", BEDFORM,
              requires=("bedform.enabled",))
           for key, label, selector in (
               ("van_rijn_1984", "Van Rijn 1984", "BedFormHeight=VanRijn84"),
               ("fredsoe_mpm", "Fredsoe-MPM", "BedFormHeight=FredsoeMPM"),
               ("fredsoe_eh", "Fredsoe-EH", "BedFormHeight=FredsoeEH"),
               ("power", "User power relation", "BedFormHeight=PowerRelation"))),
    *tuple(_m(f"bedform_relaxation.{key}", "bedform_relaxation", label, selector, key,
              SCOPE_REQUIRED, ROUTE_MISSING, ROUTE_MISSING, UNQUALIFIED,
              f"d3d4_bedform_relaxation_{key}", "dune_relaxation", BEDFORM,
              requires=("bedform.enabled",))
           for key, label, selector in (
               ("equilibrium", "Instantaneous equilibrium", "BedFormRel=equilibrium"),
               ("constant_time", "Constant relaxation time", "BedFormRel=constant_time"),
               ("length_celerity_1", "Length/celerity relation 1", "BedFormRel=Lc1"),
               ("length_celerity_2", "Length/celerity relation 2", "BedFormRel=Lc2"),
               ("length_celerity_3", "Length/celerity relation 3", "BedFormRel=Lc3"))),
    *tuple(_m(f"bedform_length.{key}", "bedform_length", label, selector, key,
              SCOPE_REQUIRED, ROUTE_PROTOTYPE, ROUTE_PROTOTYPE, FUNCTIONALITY_ONLY,
              f"d3d4_bedform_length_{key}", "dune_length", BEDFORM,
              requires=("bedform.enabled",))
           for key, label, selector in (
               ("van_rijn_1984", "Van Rijn 1984", "BedFormLength=VanRijn84"),
               ("power", "User power relation", "BedFormLength=PowerRelation"))),
    *tuple(_m(f"bedform_roughness.{key}", "bedform_roughness", label, selector, key,
              SCOPE_REQUIRED, ROUTE_PROTOTYPE, ROUTE_PROTOTYPE, FUNCTIONALITY_ONLY,
              f"d3d4_bedform_roughness_{key}", "hydraulic_roughness_feedback", BEDFORM,
              requires=("bedform.enabled",))
           for key, label, selector in (
               ("van_rijn_2007", "Van Rijn 2007", "BedFormRoughness=VanRijn07"),
               ("van_rijn_1984", "Van Rijn 1984", "BedFormRoughness=VanRijn84"),
               ("power", "User power relation", "BedFormRoughness=PowerRelation"))),
    _m("bedform.advection", "bedform_advection", "Bedform advection and CFL control",
       "BedFormAdvection=true", 1, SCOPE_REQUIRED, ROUTE_MISSING, ROUTE_MISSING,
       UNQUALIFIED, "d3d4_bedform_advection", "dune_state_advection", BEDFORM,
       requires=("bedform.enabled",)),

    # Redistribution and BRAIDS planform extensions.
    _m("redistribution.off", "redistribution", "Redistribution off", None, 0,
       SCOPE_REQUIRED, ROUTE_PUBLIC, ROUTE_PUBLIC, FUNCTIONALITY_ONLY,
       "off", "none", ADJUST, BED_UPDATE),
    _m("redistribution.avalanche", "redistribution", "Classic FLOW wet-slope flux avalanche",
       "WetSlope/AvalTime", 1, SCOPE_REQUIRED,
       ROUTE_PUBLIC, ROUTE_PUBLIC, QUALIFIED,
       "d3d4_classic_wet_slope_flux", "avalanche_bedload_face_flux", ADJUST, RDMOR),
    _m("redistribution.dry_bank", "redistribution", "Dry-bank erosion transfer",
       "ThetSD/HMaxTH/SedThr", 2, SCOPE_REQUIRED, ROUTE_PUBLIC,
       ROUTE_PUBLIC, QUALIFIED, "d3d4_dry_bank_transfer",
       "wet_to_dry_sediment_transfer", BED_UPDATE, RDMOR),
    _m("redistribution.braids_toe_collapse", "redistribution",
       "BRAIDS toe erosion and bank collapse", None, 3, SCOPE_BRAIDS_EXTENSION,
       ROUTE_PUBLIC, ROUTE_PUBLIC, QUALIFIED,
       "braids_toe_collapse", "planform_bank_transfer", BED_UPDATE),
    _m("redistribution.braids_conservative_wet_dry_avalanche", "redistribution",
       "BRAIDS conservative wet/dry repose avalanche", None, 4, SCOPE_BRAIDS_EXTENSION,
       ROUTE_PUBLIC, ROUTE_PUBLIC, QUALIFIED,
       "braids_conservative_wet_dry_avalanche", "classwise_surface_mass_transfer", BED_UPDATE),
    _m("redistribution.classic_scour_bundle", "redistribution",
       "Classic FLOW mapped-shear and legacy Scour bundle", "Scour=file", 5, SCOPE_REQUIRED,
       ROUTE_PUBLIC, ROUTE_PUBLIC, QUALIFIED,
       "d3d4_classic_scour_bundle", "mapped_shear_and_post_bed_slump", BED_UPDATE),
)


_RULES = (
    CompatibilityRule(
        "capacity_requires_transport_route", PHYSICAL_RULE,
        tuple(), (("transport_route.off", "transport_route.combine", "transport_route.total"),),
        resolution="Every profile explicitly records whether sediment is off, split, or total load.",
        source_anchors=(RDSed,), enforced=False),
    CompatibilityRule(
        "cohesive_cannot_use_total_load", PHYSICAL_RULE,
        ("capacity.cohesive_family",), forbids_any=("transport_route.total",),
        resolution="Silt/clay fractions cannot use Delft3D TotalLoad.",
        source_anchors=(RDSed,)),
    CompatibilityRule(
        "van_rijn_2007_forces_bedforms", BUNDLE_RULE,
        ("capacity.van_rijn_2007",), (("bedform.enabled",),),
        resolution="Van Rijn 2007 uses its inseparable bedform-roughness bundle.",
        source_anchors=(BEDFORM,)),
    CompatibilityRule(
        "van_rijn_2007_forces_dss2004_settling", BUNDLE_RULE,
        ("capacity.van_rijn_2007",), (("settling.hindered",),),
        resolution=("FLOW automatically couples TraFrm=-2 to the Van Rijn "
                    "2004 DSS settling/flocculation branch; it is not a "
                    "freely selectable hindering switch."),
        source_anchors=(RDSed, SETTLING)),
    CompatibilityRule(
        "transport_layer_dune_requires_bedforms", PHYSICAL_RULE,
        ("transport_layer.dune_height",), (("bedform.enabled",),),
        resolution="Dune-scaled transport-layer thickness requires an active bedform state.",
        source_anchors=(RDMORLYR,)),
    CompatibilityRule(
        "layer_options_require_layered_bed", PHYSICAL_RULE,
        tuple(), resolution="Exchange, sorting and transport-layer methods require IUnderLyr=2.",
        source_anchors=(RDMORLYR,), enforced=False),
    CompatibilityRule(
        "generic_hiding_capacity_application", BUNDLE_RULE,
        tuple(), resolution=(
            "Generic hiding modifies capacities 2, 4 and 14. Wilcock-Crowe/Gaeuman "
            "retain intrinsic mixture dependence; ISlope=4 consumes the selected hiding factor "
            "without applying it twice to capacity."),
        source_anchors=(HIDING, EQTRAN, ADJUST), enforced=False),
    CompatibilityRule(
        "skin_shear_requires_live_rp_taub_consumer", PHYSICAL_RULE,
        ("shear.soulsby_clarke_skin",),
        (("capacity.wilcock_crowe", "capacity.wilcock_crowe_modified",
          "capacity.gaeuman"),),
        resolution=(
            "BsSkin is selectable only when a live formula consumes RP_TAUB. "
            "A future cohesive route must prove an actually cohesive fraction "
            "in its fraction profile rather than passing this catalogue-only gate."),
        source_anchors=(SKIN_SHEAR, EQTRAN)),
    CompatibilityRule(
        "skin_shear_never_replaces_gaia_shear", PHYSICAL_RULE,
        ("shear.soulsby_clarke_skin",),
        forbids_any=("capacity.gaia_research",),
        resolution=(
            "Delft3D BsSkin changes RP_TAUB consumers only and must never replace "
            "the separate TELEMAC-GAIA law-9 shear correction."),
        source_anchors=(SKIN_SHEAR,)),
    CompatibilityRule(
        "secondary_external_steering", BUNDLE_RULE,
        tuple(), resolution=(
            "Secondary-flow steering is applied only to formulas returning a steerable scalar "
            "capacity; native vector formulas retain their own direction."),
        source_anchors=(EQTRAN, SECONDARY), enforced=False),
    CompatibilityRule(
        "total_load_no_duplicate_suspension", BUNDLE_RULE,
        ("transport_route.total",), forbids_any=(
            "suspension.settling_only", "suspension.van_rijn_exchange",
            "suspension.cohesive_mixed"),
        resolution="An algebraic total-load route cannot receive an independent suspended load.",
        source_anchors=(RDSed, EQTRAN)),
    CompatibilityRule(
        "avalanche_dry_bank_distinct", BUNDLE_RULE,
        tuple(), resolution=(
            "Avalanching and dry-bank transfer may coexist only as separately ledgered mechanisms; "
            "neither may consume the other's sediment transfer."),
        source_anchors=(ADJUST, BED_UPDATE), enforced=False),
)


METHODS = tuple(_METHODS)
COMPATIBILITY_RULES = tuple(_RULES)
METHOD_BY_ID = MappingProxyType({method.method_id: method for method in METHODS})

_SINGLE_CHOICE_COMPONENTS = frozenset((
    "hydraulic_roughness", "sediment_shear", "transport_route", "capacity", "bed_slope",
    "secondary_flow", "hiding", "bed_composition", "fraction_basis",
    "transport_layer", "exchange_layer", "settling", "suspension",
    "bedform_state", "bedform_height", "bedform_relaxation", "bedform_length",
    "bedform_roughness",
))


def methods_for_component(component: str, *, include_excluded: bool = True) -> tuple[MethodSpec, ...]:
    rows = tuple(method for method in METHODS if method.component == component)
    if not include_excluded:
        rows = tuple(method for method in rows if method.disposition in (
            SCOPE_REQUIRED, SCOPE_CONDITIONAL, SCOPE_BRAIDS_LEGACY,
            SCOPE_BRAIDS_EXTENSION))
    return rows


def method(method_id: str) -> MethodSpec:
    try:
        return METHOD_BY_ID[method_id]
    except (KeyError, TypeError) as error:
        raise SedimentCatalogError(
            f"Unknown sediment method identity: {method_id!r}",
            category="unknown_method",
        ) from error


def legacy_code_map(component: str, *, admitted_only: bool = False,
                    backend: str | None = None) -> Mapping[int | str, MethodSpec]:
    result: dict[int | str, MethodSpec] = {}
    for row in methods_for_component(component):
        if row.legacy_code is None:
            continue
        if admitted_only:
            status = row.cpu_route if backend == "cpu" else row.implicit_gpu_route
            if status not in ADMITTED_ROUTE_STATUSES:
                continue
        if row.legacy_code in result:
            raise SedimentCatalogError(
                f"Duplicate {component} legacy code: {row.legacy_code!r}",
                category="catalogue_integrity",
            )
        result[row.legacy_code] = row
    return MappingProxyType(result)


def admitted_legacy_codes(component: str, backend: str) -> tuple[int | str, ...]:
    if backend not in ("cpu", "implicit_gpu"):
        raise SedimentCatalogError(
            "Backend must be 'cpu' or 'implicit_gpu'",
            category="invalid_configuration",
        )
    values = tuple(legacy_code_map(component, admitted_only=True, backend=backend))
    return tuple(sorted(values, key=lambda value: (isinstance(value, str), str(value))))


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _semantic_method(row: MethodSpec) -> dict:
    return {
        "method_id": row.method_id,
        "component": row.component,
        "reference_selector": row.reference_selector,
        "legacy_code": row.legacy_code,
        "disposition": row.disposition,
        "source_id": row.source_id,
        "output_contract": row.output_contract,
        "requires": list(row.requires),
    }


def semantic_document() -> dict:
    """Return restart-relevant semantics without labels/evidence bookkeeping."""
    return {
        "schema": CATALOG_SCHEMA,
        "scope_id": CATALOG_SCOPE_ID,
        "single_choice_components": sorted(_SINGLE_CHOICE_COMPONENTS),
        "methods": [_semantic_method(row) for row in sorted(METHODS,
                    key=lambda value: value.method_id)],
        "rules": [
            {
                "rule_id": rule.rule_id,
                "category": rule.category,
                "when_all": list(rule.when_all),
                "requires_any": [list(group) for group in rule.requires_any],
                "forbids_any": list(rule.forbids_any),
                "resolution": rule.resolution,
                "enforced": rule.enforced,
            }
            for rule in sorted(COMPATIBILITY_RULES, key=lambda value: value.rule_id)
        ],
    }


def semantic_catalog_id() -> str:
    return hashlib.sha256(_canonical_json(semantic_document()).encode("ascii")).hexdigest()


def catalogue_document() -> dict:
    """Return the complete JSON-compatible audit document."""
    document = {
        "schema": CATALOG_SCHEMA,
        "scope_id": CATALOG_SCOPE_ID,
        "delft_source_commit": DELFT_SOURCE_COMMIT,
        "delft_source_lock": DELFT_SOURCE_LOCK,
        "gpu_integrator": GPU_INTEGRATOR_ID,
        "semantic_catalog_id": semantic_catalog_id(),
        "single_choice_components": sorted(_SINGLE_CHOICE_COMPONENTS),
        "methods": [asdict(row) for row in sorted(METHODS,
                    key=lambda value: value.method_id)],
        "compatibility_rules": [asdict(rule) for rule in sorted(
            COMPATIBILITY_RULES, key=lambda value: value.rule_id)],
    }
    # ``dataclasses.asdict`` deliberately preserves tuple containers.  Return
    # the canonical JSON shape so callers never depend on tuple/list details.
    return json.loads(_canonical_json(document))


def catalogue_json(*, indent: int | None = 2) -> str:
    return json.dumps(catalogue_document(), sort_keys=True, indent=indent,
                      ensure_ascii=True, allow_nan=False) + ("\n" if indent is not None else "")


def _route_status(row: MethodSpec, backend: str) -> str:
    if backend == "cpu":
        return row.cpu_route
    if backend == "implicit_gpu":
        return row.implicit_gpu_route
    raise SedimentCatalogError(
        "Backend must be 'cpu' or 'implicit_gpu'",
        category="invalid_configuration",
    )


def resolve_profile(method_ids: Iterable[str], *, backend: str | None = None) -> ResolvedProfile:
    """Validate and canonically identify an explicit method profile.

    This first catalogue resolver never chooses CPU/GPU and never rewrites an
    explicit choice.  Automatic auxiliary selection is layered on top of this
    deterministic validator in later catalogue milestones.
    """
    if isinstance(method_ids, (str, bytes)):
        raise SedimentCatalogError(
            "Method profile must be an iterable of identities",
            category="invalid_configuration",
        )
    selected = tuple(sorted(set(method_ids)))
    rows = tuple(method(value) for value in selected)
    by_component: dict[str, list[str]] = {}
    for row in rows:
        by_component.setdefault(row.component, []).append(row.method_id)
        if row.disposition in (SCOPE_EXCLUDED, SCOPE_REFERENCE_DISABLED):
            raise SedimentCatalogError(
                f"Method is outside the active river scope: {row.method_id}",
                category="scope_restriction",
            )
        if backend is not None and _route_status(row, backend) not in ADMITTED_ROUTE_STATUSES:
            raise SedimentCatalogError(
                f"{row.method_id} is not connected to the {backend} route",
                category="implementation_restriction",
            )
        missing = tuple(value for value in row.requires if value not in selected)
        if missing:
            raise SedimentCatalogError(
                f"{row.method_id} requires {', '.join(missing)}",
                category="physical_incompatibility",
            )
    for component, values in by_component.items():
        if component in _SINGLE_CHOICE_COMPONENTS and len(values) > 1:
            raise SedimentCatalogError(
                f"Multiple methods selected for {component}: {', '.join(values)}",
                category="physical_incompatibility",
            )

    reasons: list[str] = []
    chosen = set(selected)
    for rule in COMPATIBILITY_RULES:
        active = not rule.when_all or set(rule.when_all) <= chosen
        if not active:
            continue
        if rule.enforced:
            for group in rule.requires_any:
                if not chosen.intersection(group):
                    raise SedimentCatalogError(
                        f"{rule.rule_id}: {rule.resolution}",
                        category="physical_incompatibility",
                    )
            forbidden = chosen.intersection(rule.forbids_any)
            if forbidden:
                raise SedimentCatalogError(
                    f"{rule.rule_id}: {rule.resolution}",
                    category="physical_incompatibility",
                )
        if rule.when_all and rule.resolution:
            reasons.append(rule.resolution)

    identity_payload = {
        "schema": CATALOG_SCHEMA,
        "scope_id": CATALOG_SCOPE_ID,
        "semantic_catalog_id": semantic_catalog_id(),
        "selected_method_ids": list(selected),
        "backend": backend,
    }
    identity = hashlib.sha256(_canonical_json(identity_payload).encode("ascii")).hexdigest()
    return ResolvedProfile(CATALOG_SCHEMA, CATALOG_SCOPE_ID,
                           semantic_catalog_id(), selected, backend,
                           tuple(reasons), identity)


def capacity_source_by_legacy_code() -> Mapping[int, str]:
    """Backward-compatible public law-code provenance for the old facade."""
    result = {}
    for code, row in legacy_code_map("capacity").items():
        if isinstance(code, int) and 0 <= code <= 8:
            result[code] = row.source_id
    return MappingProxyType(result)


def slope_source_by_legacy_code() -> Mapping[int, str]:
    result = {}
    for code, row in legacy_code_map("bed_slope").items():
        if isinstance(code, int) and code in (0, 2, 3, 5):
            result[code] = row.source_id
    return MappingProxyType(result)


if __name__ == "__main__":  # pragma: no cover - convenience audit export
    print(catalogue_json(), end="")
