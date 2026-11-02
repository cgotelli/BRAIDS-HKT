"""Backend-independent physical selection contract for the unified river UI.

This is a method resolver, not execution admission or a complete run input.
Geometry, boundary/initial fields and process-specific numerical parameters are
validated by preparation. No old project, integer selector or missing-field
inference is accepted here. Importing this module needs neither QGIS nor CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re

from .sediment.capacity_parameters import (
    FORMULA_CONTRACTS, CapacityParameterError, resolve_capacity_parameters,
)

SCHEMA = "braids.river-model.v3"
RESOLVED_SCHEMA = "braids.resolved-river-model.v3"
INTEGRATOR = "explicit_fwave_ssprk22_river_v1"
CAPACITIES = tuple(k for k in FORMULA_CONTRACTS if not k.endswith("_legacy"))
HIDING = ("hiding.none", "hiding.egiazaroff", "hiding.ashida_michiue",
          "hiding.parker_klm", "hiding.wu_wang_jia")
SLOPES = ("bed_slope.off", "bed_slope.delft_bagnold_ikeda",
          "bed_slope.delft_koch_flokstra", "bed_slope.delft_parker_andrews")
INTRINSIC = frozenset(("capacity.wilcock_crowe", "capacity.wilcock_crowe_modified",
                       "capacity.gaeuman"))
SURVEY_BANKS = ('bank.braids_accepted_exner_toe_supply',
                'bank.braids_retained_undercut_block_collapse')


class ModelError(ValueError):
    """Stable category and field location for both UI and headless diagnostics."""

    def __init__(self, category: str, path: str, message: str):
        self.category, self.path = category, path
        super().__init__(f"{path}: {message}")


def _fail(path, message, category="invalid_input"):
    raise ModelError(category, path, message)


def _object(value, keys, path):
    if not isinstance(value, dict) or set(value) != set(keys):
        _fail(path, "Expected exactly these fields: " + ", ".join(sorted(keys)))


def _choice(value, choices, path):
    if type(value) is not str or value not in choices:
        _fail(path, "Choose an explicit method ID from: " + ", ".join(choices))


def _boolean(value, path):
    if type(value) is not bool:
        _fail(path, "An explicit boolean is required")


def _number(value, path, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        _fail(path, "A finite number is required")
    if value < 0 or (positive and value == 0):
        _fail(path, "A positive value is required" if positive else "A nonnegative value is required")


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ModelError("invalid_input", "model", "Finite JSON data required") from exc


def _hash(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail(key, "Duplicate JSON field")
        result[key] = value
    return result


def _read(value):
    if isinstance(value, str):
        try:
            value = json.loads(value, object_pairs_hook=_pairs,
                               parse_constant=lambda s: _fail("model", f"Nonfinite JSON value {s}"))
        except json.JSONDecodeError as exc:
            raise ModelError("invalid_input", "model", "Invalid JSON") from exc
    # Check the version BEFORE interpreting any older fields.
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        _fail("schema", f"Only {SCHEMA} is accepted. Old BRAIDS formats cannot be run "
              "or migrated; create a new project.", "unsupported_format")
    return value


def new_model(*, backend="cpu"):
    """Fresh editor selection: hydrodynamics only, all optional processes off."""
    _choice(backend, ("cpu", "gpu"), "execution.backend")
    return {
        "schema": SCHEMA,
        "execution": {"backend": backend, "integrator": INTEGRATOR},
        "hydrodynamics": {
            "roughness": "roughness.chezy",
            "secondary_flow": "secondary.off",
            "turbulence": "off",
        },
        "sediment": {"enabled": False, "selection": None},
    }


def new_bedload_selection():
    """Explicit editable MPM proposal; does not enable sediment in a model."""
    return {
        "capacity": {"method": "capacity.mpm", "parameters": {"ACal": 1.0}, "grain_sizes_m": {}},
        "transport": "bedload",
        "grains": [{"id": "sand", "diameter_m": 0.0005, "diameter_definition": "d50",
                    "solid_density_kg_m3": 2650.0, "fraction": 1.0}],
        "fraction_basis": "fraction_basis.solid_volume",
        "capacity_hiding": "auto",
        "slope_hiding": "hiding.none",
        "bed_slope": "bed_slope.off",
        "shear": "shear.total_roughness",
        "secondary_direction": "auto",
        "suspension": "off",
        "settling": "auto",
        "bed": {"composition": "bed.fully_mixed", "transport_layer": "off",
                "coarse_layer": False, "sorting": "sorting.off"},
        "bedforms": "off",
        "banks": "off",
        "avalanche": "off",
    }


def _grains(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 32:
        _fail("sediment.grains", "Supply 1..32 noncohesive fractions")
    ids = set()
    for i, row in enumerate(rows):
        path = f"sediment.grains[{i}]"
        if isinstance(row, dict) and 'diameter_definition' not in row:
            _fail(path + '.diameter_definition', 'Explicit grain diameter definition required. '
                  'Create a new project and enter the formula-required diameter in Grain fractions; '
                  'old unlabeled grain sizes are not inferred or migrated.')
        _object(row, ("id", "diameter_m", "diameter_definition", "solid_density_kg_m3", "fraction"), path)
        if type(row["id"]) is not str or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", row["id"]):
            _fail(path + ".id", "Use a stable ASCII identifier starting with a letter")
        if row["id"].casefold() in ids:
            _fail(path + ".id", "Fraction IDs must be unique, ignoring case")
        ids.add(row["id"].casefold())
        for name in ("diameter_m", "solid_density_kg_m3"):
            _number(row[name], path + "." + name, positive=True)
        _number(row["fraction"], path + ".fraction")
        if row["fraction"] > 1:
            _fail(path + ".fraction", "Fractions must be between zero and one")
    if abs(math.fsum(row["fraction"] for row in rows) - 1.0) > 1.e-10:
        _fail("sediment.grains", "Fractions must sum to one; values are never silently normalized")


def _resolve_sediment(selection, hydro, decisions):
    path = "sediment.selection"
    _object(selection, new_bedload_selection(), path)
    from .river_jiang_cao import METHOD as JC_METHOD, resolve_selection
    if selection.get('capacity', {}).get('method') == JC_METHOD:
        return resolve_selection(selection, hydro, decisions)
    s = json.loads(_canonical(selection))
    from .sediment.collapse_setup import resolve as resolve_collapse, EXNER
    collapse = resolve_collapse(s['avalanche'], s['capacity']['method'])
    if collapse != s['avalanche']:
        decisions.append(dict(path='sediment.avalanche', requested=s['avalanche'], resolved=collapse,
            reason='The geometric repose criterion uses the ordinary finite Exner bed inventory; no Jiang-Cao coupled phase physics is added.'))
        s['avalanche'] = collapse
    capacity = s["capacity"]
    _object(capacity, ("method", "parameters", "grain_sizes_m"), path + ".capacity")
    method = capacity["method"]
    _choice(method, CAPACITIES, path + ".capacity.method")
    contract = FORMULA_CONTRACTS[method]
    try:
        values = dict(resolve_capacity_parameters(method, capacity["parameters"]))
    except CapacityParameterError as exc:
        raise ModelError("invalid_input", path + ".capacity.parameters", str(exc)) from exc
    if "ACal" in values and values["ACal"] < 0:
        _fail(path + ".capacity.parameters.ACal", "Transport calibration cannot be negative")
    if method == "capacity.engelund_hansen" and not 0 <= values["SusFac"] <= 1:
        _fail(path + ".capacity.parameters.SusFac",
              "The suspended share must be between zero and one; FLOW partitions "
              "total capacity as (1-SusFac) bedload and SusFac suspension")
    if method == "capacity.ashida_michiue_generalized" and values["ThetaC"] < 0:
        _fail(path + ".capacity.parameters.ThetaC", "Critical Shields stress must be nonnegative")
    if method == "capacity.van_rijn_1984_modified":
        decisions.append({"path": path + ".capacity.parameters", "requested": method,
                          "resolved": values, "reason": "Stock tranb7 applies ACal to suspension only; "
                          "BetaM has no effect with the current noncohesive fractions (mud fraction zero)"})
    if method == "capacity.wang_fredsoe":
        decisions.append({"path": path + ".capacity", "requested": method, "resolved": method,
                          "reason": "Uses characteristic shear, not floored Chezy shear; negative "
                          "native suspended equilibrium is rejected when used as a concentration source"})
    if method in INTRINSIC:
        for key in ("ThetaC0", "Alpha0"):
            if key in values and values[key] < 0.:
                _fail(path + ".capacity.parameters." + key, "Intrinsic mixture coefficients must be nonnegative")
        decisions.append({"path": path + ".capacity", "requested": method, "resolved": method,
                          "reason": ("Uses evolving surface Dg/DgSD/sand and selected skin bed stress; "
                                     if s["shear"]=="shear.soulsby_clarke_skin" else
                                     "Uses evolving surface Dg/DgSD/sand and hydraulic bed stress; ") +
                          "effective sediment velocity remains separate. Intrinsic hiding is applied "
                          "once; suspended equilibrium is zero"})
    if method in ("capacity.soulsby_van_rijn", "capacity.soulsby"):
        if values["ACal"] < 0.:
            _fail(path + ".capacity.parameters.ACal", "Transport calibration must be nonnegative")
        if method == "capacity.soulsby" and values["RatioD50Z0"] <= 0.:
            _fail(path + ".capacity.parameters.RatioD50Z0", "Roughness ratio must be positive")
        decisions.append({"path": path + ".capacity", "requested": method, "resolved": method,
                          "reason": "Current-only vector capacity owns its direction: no external "
                          "spiral rotation. Secondary-flow momentum remains a separate selected "
                          "hydraulic process. Slope forcing uses characteristic shear, not the "
                          "capacity's internal drag. Wave-only parameters are saved but inactive"})
    for name, value in values.items():
        if name not in capacity["parameters"]:
            decisions.append({"path": f"{path}.capacity.parameters.{name}",
                              "requested": "unspecified", "resolved": value,
                              "reason": "Locked FLOW formula default"})
    capacity["parameters"] = values
    _grains(s["grains"])
    from .sediment.grain_diameter_contract import REQUIRED, validate_grain_sizes
    if method in REQUIRED and len(s['grains']) != 1:
        _fail(path+'.grains', 'This bulk formula transports one material. Enter its characteristic diameters in grain_sizes_m, not as transported classes.')
    try:
        validate_grain_sizes(capacity, primary=s['grains'][0]['diameter_m'])
    except ValueError as exc:
        _fail(path+'.capacity.grain_sizes_m', str(exc), 'incompatible_grain_diameter')
    independent_suspension=s['transport']=='split' and s['suspension'] in ('suspension.settling_only','suspension.garcia_parker_1991','suspension.de_leeuw_2020','suspension.van_rijn_1984','suspension.zyserman_fredsoe_1994','suspension.camenen_larson_2008')
    if method == 'capacity.wong_parker_2006':
        if len(s['grains'])!=1 or (s['transport']!='bedload' and not independent_suspension):
            _fail(path+'.capacity','Wong & Parker currently requires one grain and bedload only', 'integration_pending')
        decisions.append(dict(path=path+'.capacity',requested=method,resolved=method,
            reason='Wong & Parker (2006), 3.97*max(theta-0.0495,0)^1.5 (authors preprint Eq24). '
            'Arithmetic mean Dm, local hydraulic bed shear; no MPM form-drag reduction or external hiding. '
            'The 2D bed resistance is bed traction, not the cross-section total bed-plus-wall stress '
            'used before the paper flume sidewall correction. No second sidewall correction is applied. '
            'Single uniform grain, lower-regime plane-bed scope. Fixed coefficients; ACal multiplier standard 1.'))
    if method == 'capacity.camenen_larson_2005':
        if len(s['grains'])!=1 or (s['transport']!='bedload' and not independent_suspension):
            _fail(path+'.capacity','Camenen & Larson currently requires one grain and bedload only', 'integration_pending')
        decisions.append(dict(path=path+'.capacity',requested=method,resolved=method,
            reason='Camenen & Larson (2005), steady-current bedload: 12*theta^1.5*exp(-4.5*theta_cr/theta). '
            'Physical hydraulic total drag and median D50. Critical Shields is computed using '
            'Soulsby & Whitehouse (1997), as in the authors CR-07-1 Eq49, from D50, densities and '
            'explicit kinematic viscosity VicMol (m2/s). No hard threshold, MPM ripple reduction, '
            'waves or suspended transport. ACal is a multiplier, standard 1.'))
    if method == 'capacity.recking_2010':
        if len(s['grains'])!=1 or (s['transport']!='bedload' and not independent_suspension):
            _fail(path+'.capacity','Recking (2010) currently requires one grain and bedload only', 'integration_pending')
        decisions.append(dict(path=path+'.capacity',requested=method,resolved=method,
            reason='Recking (2010) Eqs.4 and 12-14, independently measured D84 and D50. Explicit reach slope '
            'S='+str(values['ReachSlope'])+' m/m, not local friction slope. Local hydraulic shear '
            'replaces the reach discharge-to-radius inversion, without changing hydraulic roughness. '
            'Exact Eq12 intersection avoids rounded-breakpoint discontinuity; no hard motion threshold.'))
    if method == 'capacity.ancey_recking_2023':
        if len(s['grains'])!=1 or (s['transport']!='bedload' and not independent_suspension):
            _fail(path+'.capacity','Ancey & Recking (2023) currently requires one grain and bedload only', 'integration_pending')
        decisions.append(dict(path=path+'.capacity',requested=method,resolved=method,
            reason='Ancey & Recking (2023) Eq. 42: physical hydraulic stress and surface d50; '
            'fixed Shields transitions 0.078 and 0.40, no hard motion threshold. '
            'Single-grain local capacity adaptation, not a river-wide validation or FLOW formula.'))
    if method == "capacity.smart_jaggi":
        if len(s["grains"]) != 1 or (s["transport"] != "bedload" and not independent_suspension):
            _fail(path + ".capacity", "Smart & Jäggi currently supports one grain size and bedload only; "
                  "select one fraction and bedload transport", "integration_pending")
        decisions.append(dict(path=path+".capacity", requested=method, resolved=method,
            reason="Smart-Jaeggi energy-slope formulation (BASEHPC 4.0.2 Eqs 1.29-1.31): "
            "J=|U|^2/(C_hyd^2 h), physical depth/velocity and hydraulic resistance. "
            "Measured Dm, D30 and D90; grading factor=(D90/D30)^0.2; ACal=4 and ThetaC=0.05 are the "
            "standard coefficients. No MPM ripple reduction or implicit repose correction. "
            "Published application range J=0.005..0.2; values outside it are extrapolation, not clamped."))
    from .sediment.grain_diameter_contract import require_diameter_definition
    for i, grain in enumerate(s['grains']):
        try:
            diameter = require_diameter_definition(method, len(s['grains']), grain['diameter_definition'])
        except ValueError as exc:
            _fail(f'sediment.grains[{i}].diameter_definition', str(exc), 'incompatible_grain_diameter')
    decisions.append(dict(path='sediment.grains.diameter_definition', requested=diameter.definition,
                          resolved=diameter.definition, reason=diameter.note))
    _choice(s["fraction_basis"], ("fraction_basis.mass", "fraction_basis.solid_volume"),
            path + ".fraction_basis")
    _choice(s["transport"], ("bedload", "split", "total_load"), path + ".transport")
    _choice(s["capacity_hiding"], ("auto", "hiding.intrinsic") + HIDING, path + ".capacity_hiding")
    _choice(s["slope_hiding"], HIDING, path + ".slope_hiding")
    _choice(s["bed_slope"], SLOPES, path + ".bed_slope")
    _choice(s["shear"], ("shear.total_roughness", "shear.soulsby_clarke_skin"), path + ".shear")
    _choice(s["secondary_direction"], ("auto", "off", "external", "capacity_owned"),
            path + ".secondary_direction")
    _choice(s["suspension"], ("off", "suspension.settling_only", "suspension.van_rijn_exchange", "suspension.garcia_parker_1991", "suspension.de_leeuw_2020", "suspension.van_rijn_1984", "suspension.zyserman_fredsoe_1994", "suspension.camenen_larson_2008"),
            path + ".suspension")
    _choice(s["settling"], ("auto", "off", "settling.grain_size"), path + ".settling")
    _choice(s["bedforms"], ("off", "bedform.enabled"), path + ".bedforms")
    _choice(s["banks"], ("off", "redistribution.dry_bank", *SURVEY_BANKS), path + ".banks")
    _choice(s["avalanche"], ("off", "redistribution.avalanche",
                            "redistribution.braids_conservative_wet_dry_avalanche", EXNER), path + ".avalanche")

    def incompatible(field, message):
        _fail(f"{path}.{field}", message, "incompatible_methods")

    def automatic(field, value, reason):
        if s[field] == "auto":
            s[field] = value
            decisions.append({"path": f"{path}.{field}", "requested": "auto",
                              "resolved": value, "reason": reason})

    if s['banks'] in SURVEY_BANKS:
        decisions.append(dict(path=path+'.banks',requested=s['banks'],resolved=s['banks'],
            reason='Explicit geographic bank polygons and toe guides; finite existing columns, '
            'accepted net Exner toe supply, and finest survey-corridor protection. '
            'Block mode additionally retains undercut and releases finite stock. '
            'This BRAIDS extension is distinct from FLOW dry-bank sharing; neither runs twice.'))

    intrinsic = method in INTRINSIC
    automatic("capacity_hiding", "hiding.intrinsic" if intrinsic else "hiding.none",
              "Mixture dependence is intrinsic to this capacity" if intrinsic else
              "No optional hiding correction was requested")
    if intrinsic and s["capacity_hiding"] != "hiding.intrinsic":
        incompatible("capacity_hiding", "This capacity owns its mixture/hiding dependence; "
                     "it cannot be disabled or multiplied by another hiding law")
    if not intrinsic and s["capacity_hiding"] == "hiding.intrinsic":
        incompatible("capacity_hiding", "This capacity has no intrinsic mixture hiding")
    if not contract.generic_hiding and not intrinsic and s["capacity_hiding"] != "hiding.none":
        incompatible("capacity_hiding", "This capacity formula does not consume generic capacity hiding")
    if s["slope_hiding"] != "hiding.none" and s["bed_slope"] != "bed_slope.delft_parker_andrews":
        incompatible("slope_hiding", "This separate hiding factor is consumed by Parker–Andrews only")
    if s["shear"] == "shear.soulsby_clarke_skin" and not intrinsic:
        incompatible("shear", "This formula does not consume the replaceable RP_TAUB stress. "
                     "Skin shear cannot replace its Chezy/ripple closure")
    if s["shear"] == "shear.soulsby_clarke_skin":
        decisions.append(dict(path=path+".shear", requested=s["shear"], resolved=s["shear"],
            reason="FLOW current-only noncohesive compbsskin replaces intrinsic RP_TAUB only. "
                   "Explicit sand roughness/viscosity; hydraulic drag, capacity prefactor and "
                   "bed-slope characteristic shear remain separate. Reynolds-dependent wave derivative."))

    secondary_on = hydro["secondary_flow"] != "secondary.off"
    direction = ("external" if contract.external_secondary_direction else "capacity_owned") if secondary_on else "off"
    automatic("secondary_direction", direction,
              "Scalar capacities use classic FLOW steering; vector capacities retain direction ownership; "
              "steering is off when secondary flow is off")
    if s["secondary_direction"] != direction:
        incompatible("secondary_direction", f"The selected coherent FLOW combination requires {direction}")

    split = s["transport"] == "split"
    if split and s["suspension"] == "off":
        incompatible("suspension", "Split transport requires an explicit suspended-mass process")
    if not split and s["suspension"] != "off":
        incompatible("suspension", "Bedload/total-load routing cannot also transport independent suspended mass")
    automatic("settling", "settling.grain_size" if split else "off",
              "Settling is required for suspended mass, and inactive without it")
    if (s["settling"] != "off") != split:
        incompatible("settling", "Settling must be active exactly when suspended mass is active")
    if s["suspension"]=="suspension.settling_only":
        decisions.append({"path":path+".suspension","requested":s["suspension"],"resolved":s["suspension"],
            "reason":"BRAIDS deposition-only extension, not a FLOW Galappatti selector. "
            "Supplied mass advects and settles exponentially on explicit measured Ds with clear-water SettleFrm2; no entrainment, "
            "equilibrium capacity or evolving suspended diameter. The entrainment factor must be zero."})
    if s['suspension']=='suspension.garcia_parker_1991':
        decisions.append(dict(path=path+'.suspension',requested=s['suspension'],resolved=s['suspension'],
            reason='Independent GP1991 reference entrainment, bounded EH skin partition, clear-water settling, '
            'explicit Rouse depth-mean extension. Native algebraic suspended capacity is NOT added. '
            'Hydraulic resistance and selected bedload closure remain unchanged.'))
    if s['suspension']=='suspension.de_leeuw_2020':
        decisions.append(dict(path=path+'.suspension',requested=s['suspension'],resolved=s['suspension'],
            reason='Independent de Leeuw2020 Eq26a, measured bed D84 and suspended Ds, Ferguson-Church settling, '
            'preferred concentration-profile Rouse fit and 1.66Ca below reference. Local friction-slope/shoreline extensions are explicit. '
            'No native algebraic suspended capacity is added; selected bedload and hydraulic resistance remain unchanged.'))
    from .sediment.river_classic_suspension import METHODS as CLASSIC_SUSPENSION
    if s['suspension'] in CLASSIC_SUSPENSION:
        decisions.append(dict(path=path+'.suspension',requested=s['suspension'],resolved=s['suspension'],
            reason='Named S6 reference concentration, measured bed descriptors and Ds; physical local shear and declared vertical profile. '
            'Conservative independent suspended mass exchange. No native algebraic suspended source; selected bedload and hydraulic resistance unchanged.'))
    # A zero equilibrium suspended capacity is not physical incompatibility.
    # FLOW can still transport and relax externally supplied suspended mass.
    zero_suspended = method in INTRINSIC or method in (
        "capacity.mpm", "capacity.ashida_michiue_generalized", "capacity.soulsby", "capacity.smart_jaggi",
        "capacity.ancey_recking_2023", "capacity.recking_2010", "capacity.camenen_larson_2005", "capacity.wong_parker_2006")
    zero_suspended = zero_suspended or (
        method == "capacity.general" and values["SusACal"] == 0)
    zero_suspended = zero_suspended or (
        method == "capacity.engelund_hansen" and values["SusFac"] == 0)
    if zero_suspended and s["suspension"] == "suspension.van_rijn_exchange":
        decisions.append({"path": path + ".suspension", "requested": s["suspension"],
                          "resolved": s["suspension"], "reason":
                          "This capacity supplies zero equilibrium suspended load; "
                          "externally supplied suspension still transports and relaxes"})
    if s["transport"] == "bedload" and not zero_suspended:
        decisions.append({"path": path + ".transport", "requested": "bedload", "resolved": "bedload",
                          "reason": "Only the bedload channel is selected; suspended capacity "
                          "is not added to the bed or transported"})
    if s["transport"] == "total_load":
        decisions.append({"path":path+".transport","requested":"total_load","resolved":"total_load",
                          "reason":"Bedload plus algebraic equilibrium suspended flux, with unit suspended "
                          "factor; no resident suspended inventory or exchange. The bed and suspended "
                          "vectors retain their distinct FLOW directions."})
        if method=="capacity.van_rijn_1993":
            decisions.append({"path":path+".settling","requested":s["settling"],"resolved":"off",
                "reason":"FLOW non-suspfrac algebraic profile uses zero fall speed, molecular Schmidt "
                "number 1 and nominal diameter. IopSus state evolution and split-route reference "
                "diameter/Schmidt inputs are inactive; use split transport for physical settling."})

    bed = s["bed"]
    _object(bed, ("composition", "transport_layer", "coarse_layer", "sorting"), path + ".bed")
    _choice(bed["composition"], ("bed.fully_mixed", "bed.layered"), path + ".bed.composition")
    _choice(bed["transport_layer"], ("off", "transport_layer.constant",
                                    "transport_layer.water_depth", "transport_layer.dune_height"),
            path + ".bed.transport_layer")
    _boolean(bed["coarse_layer"], path + ".bed.coarse_layer")
    _choice(bed["sorting"], ("sorting.off", "sorting.discrete_shields", "sorting.continuous_shields",
                             "sorting.wilcock_mcardell", "sorting.transport_hiding"), path + ".bed.sorting")
    if bed["composition"] == "bed.fully_mixed":
        if bed["transport_layer"] != "off" or bed["coarse_layer"] or bed["sorting"] != "sorting.off":
            incompatible("bed", "Active/coarse layers and sorting require a layered bed")
    elif bed["transport_layer"] == "off":
        incompatible("bed.transport_layer", "A layered bed requires an explicit active-layer thickness method")
    if bed["sorting"] != "sorting.off" and not bed["coarse_layer"]:
        incompatible("bed.sorting", "Sorting requires an explicitly enabled coarse layer")
    if bed["transport_layer"] == "transport_layer.dune_height" and s["bedforms"] == "off":
        incompatible("bed.transport_layer", "Dune-scaled thickness requires bedform state")
    from .sediment.river_material import composition_method
    s["composition_fraction"] = composition_method("mixed" if bed["composition"] == "bed.fully_mixed" else "layered")
    decisions.append(dict(path=path+".composition_fraction", requested="automatic",
        resolved=s["composition_fraction"], reason="FLOW uses mass weighting for fully mixed beds and "
        "reference-column volume weighting for layered beds. Input fraction units do not select this physics."))
    if s["bedforms"] != "off" and len({g["solid_density_kg_m3"] for g in s["grains"]}) != 1:
        incompatible("bedforms", "The selected FLOW bedform relation assumes equal solid densities across fractions")
    return s


@dataclass(frozen=True)
class ResolvedModel:
    # Store canonical bytes as strings, not mutable dictionaries hidden in a
    # frozen dataclass. Every returned document is a detached copy.
    _request: str
    _physics: str
    _decisions: str

    @property
    def physics_id(self):
        """Active method-selection identity, not complete run/checkpoint authentication."""
        return _hash(json.loads(self._physics))

    @property
    def configuration_id(self):
        return _hash(json.loads(self._request))

    def document(self):
        return {"schema": RESOLVED_SCHEMA, "request": json.loads(self._request),
                "physics": json.loads(self._physics), "decisions": json.loads(self._decisions),
                "physics_id": self.physics_id, "configuration_id": self.configuration_id,
                "qualification": "selection_only"}


def resolve_model(value) -> ResolvedModel:
    """Resolve a current method selection without changing the caller's data.

    A successful result proves selection consistency, NOT runtime coverage.
    The shared worker must separately admit the exact resolved configuration.
    """
    request = _read(value)
    _object(request, ("schema", "execution", "hydrodynamics", "sediment"), "model")
    execution = request["execution"]
    _object(execution, ("backend", "integrator"), "execution")
    _choice(execution["backend"], ("cpu", "gpu"), "execution.backend")
    _choice(execution["integrator"], (INTEGRATOR,), "execution.integrator")
    hydro = request["hydrodynamics"]
    _object(hydro, ("roughness", "secondary_flow", "turbulence"), "hydrodynamics")
    from .river_jiang_cao import ROUGHNESS as JC_ROUGHNESS, METHOD as JC_METHOD
    _choice(hydro["roughness"], ("roughness.chezy", "roughness.manning",
                                 "roughness.white_colebrook", JC_ROUGHNESS), "hydrodynamics.roughness")
    _choice(hydro["secondary_flow"], ("secondary.off", "secondary.prognostic"),
            "hydrodynamics.secondary_flow")
    _choice(hydro["turbulence"], ("off", "constant_eddy_viscosity"), "hydrodynamics.turbulence")
    sediment = request["sediment"]
    _object(sediment, ("enabled", "selection"), "sediment")
    _boolean(sediment["enabled"], "sediment.enabled")
    if hydro['roughness'] == JC_ROUGHNESS and not (sediment['enabled'] and
            isinstance(sediment['selection'], dict) and sediment['selection'].get('capacity', {}).get('method') == JC_METHOD):
        _fail('hydrodynamics.roughness', 'Automatic Sun/van Rijn resistance requires the active Jiang–Cao bundle. Choose Manning, Chézy or White–Colebrook for this configuration.')
    decisions = []
    if sediment["enabled"]:
        selected = _resolve_sediment(sediment["selection"], hydro, decisions)
    else:
        # Retain an inactive valid selection without enabling its processes or
        # hashing them into active physics. Resolve it independently of the
        # currently selected hydraulic secondary state.
        if sediment["selection"] is not None:
            inactive = sediment["selection"]
            inactive_hydro = dict(hydro)
            direction = inactive.get("secondary_direction") if isinstance(inactive, dict) else None
            inactive_hydro["secondary_flow"] = ("secondary.prognostic" if direction in
                                               ("external", "capacity_owned") else "secondary.off")
            from .river_jiang_cao import METHOD as JC_METHOD
            if inactive.get('capacity', {}).get('method') == JC_METHOD:
                inactive_hydro.update(secondary_flow='secondary.prognostic', turbulence='off', roughness=JC_ROUGHNESS)
            _resolve_sediment(inactive, inactive_hydro, [])
        selected = None
    physics = {"hydrodynamics": hydro, "sediment": selected}
    if selected is not None and 'bundle' in selected:
        bundle = selected['bundle']
        physics['hydrodynamics'] = dict(hydro, roughness=bundle['roughness'], secondary_flow=bundle['secondary'])
        decisions.append(dict(path='hydrodynamics', requested=hydro, resolved=physics['hydrodynamics'],
            reason='The explicitly selected research bundle owns resistance and finite-adjustment secondary flow; '
                   'the named automatic roughness is explicitly selected in Domain.'))
    if selected is not None and selected["avalanche"] == "redistribution.avalanche":
        decisions.append(dict(path="sediment.avalanche", requested=selected["avalanche"],
            resolved=selected["avalanche"], reason="Wet-slope flux is added before donor-fraction selection "
            "and shares the finite-stock limiter. It replaces, rather than adds to, post-update collapse."))
    if selected is not None and selected["bedforms"] != "off":
        decisions.append(dict(path="hydrodynamics.total_resistance", requested=hydro["roughness"],
            resolved="bedform.total_resistance", reason="Enabled bedform feedback owns total resistance. "
            "The prescribed roughness law initializes its predictor and supplies isolated-cell fallback; "
            "it is not added as a second drag term."))
    return ResolvedModel(_canonical(request), _canonical(physics), _canonical(decisions))


def verify_resolved(document) -> ResolvedModel:
    """Re-resolve saved choices; reject altered or stale method identities."""
    if not isinstance(document, dict) or document.get("schema") != RESOLVED_SCHEMA:
        _fail("schema", "Current resolved model identity required", "unsupported_format")
    if "request" not in document:
        _fail("request", "Explicit requested model is missing")
    resolved = resolve_model(document["request"])
    if _canonical(document) != _canonical(resolved.document()):
        _fail("model", "Saved physical identity or resolved methods differ; explicitly prepare again",
              "identity_mismatch")
    return resolved
