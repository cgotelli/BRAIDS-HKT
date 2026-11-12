"""Strict runtime reader for the additive Q0b FLOW-2DH scope overlay.

Q0/Q0a remain frozen in :mod:`live_catalogue_v1`.  This module authenticates
that installed parent, validates a separately hashed Q0b delta, and composes an
effective document on deep copies.  Catalogue presence is not execution
admission: excluded interfaces and out-of-scope requests fail before inherited
resolution, allocation, library loading, CUDA initialisation, or stepping.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from . import live_catalogue_v1 as _v1


SCOPE_OVERLAY_FILE = Path(__file__).with_name("flow2dh_scope_overlay_v1.json")
PARENT_LEDGER_FILE = Path(__file__).with_name("flow2dh_selector_ledger_v2.json")
LIVE_CATALOGUE_FILE = SCOPE_OVERLAY_FILE
LIVE_CATALOGUE_SCHEMA = 2
LIVE_CATALOGUE_REVISION = "flow2dh-live-catalogue-v2-effective"
SCOPE_OVERLAY_SCHEMA = 1
SCOPE_OVERLAY_REVISION = "flow2dh-q0b-scope-overlay-v1"
BACKENDS = _v1.BACKENDS
ADMITTED = _v1.ADMITTED
LiveCatalogueError = _v1.LiveCatalogueError

_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOP_LEVEL_KEYS = frozenset((
    "artifact_schema", "artifact_schema_version", "revision_id", "status",
    "base_live_catalogue", "scope_contract", "corrections", "added_entries",
    "compatibility_rules", "request_policy", "stage_manifests", "counts",
    "claims", "identity",
))
_BASE_KEYS = frozenset((
    "path", "sha256", "revision_id", "live_physical_semantic_id",
    "catalogue_erratum_registry_id", "live_catalogue_manifest_id",
))
_CORRECTION_KEYS = frozenset((
    "correction_id", "entry_id", "parent_row_sha256", "patches",
    "lifecycle_patches", "source_anchors", "owner", "applicability",
    "fail_closed",
))
_PATCH_KEYS = frozenset(("path", "expected", "effective"))
_OWNER_KEYS = frozenset(("milestone", "responsibility"))
_FAIL_CLOSED_KEYS = frozenset(("enabled", "category", "phase"))
_ADDED_REQUIRED_KEYS = frozenset((
    "entry_id", "namespace", "kind", "family", "label", "counts_as_method",
    "selector", "source_status", "owner", "applicability", "fail_closed",
    "lifecycle", "value_schema", "source_anchors",
))
_OUTPUT_WIRING_KEYS = frozenset((
    "result_field_id", "checkpoint_field_id", "wiring_state",
))
_ADDED_METHOD_KEYS = frozenset((
    "backend_status", "compatibility", "consumer_order", "requiredness",
    "runtime_method_id", "schema_v1_source_id", "scope", "subkind",
    "unresolved_semantics",
))
_ADDED_OPTIONAL_KEYS = _OUTPUT_WIRING_KEYS | _ADDED_METHOD_KEYS
_LIFECYCLE_KEYS = frozenset((
    "accepted_state_connection", "entry_id", "exact_profile_refs", "family",
    "implementation", "kind", "legacy_surface", "namespace",
    "non_target_backend_history", "owner_edges", "product_profile_refs",
))
_IDENTITY_KEYS = frozenset((
    "base_binding_id", "correction_registry_id", "added_entry_registry_id",
    "compatibility_registry_id", "stage_manifest_registry_id",
    "effective_scope_id", "overlay_manifest_id",
))
_REQUEST_POLICY_LISTS = frozenset((
    "host_interface_entry_ids", "host_route_tags", "host_activation_keywords",
    "fm_only_keywords", "managed_operation_keywords",
    "strict_2dh_metadata_only_entry_ids", "not_yet_admitted_keywords",
    "output_keywords", "known_parameter_keywords",
))
_REQUEST_POLICY_MAPS = frozenset(("failure_categories", "failure_phases"))
_REQUEST_POLICY_KEYS = _REQUEST_POLICY_LISTS | _REQUEST_POLICY_MAPS
_POLICY_DISPOSITIONS = frozenset((
    "host_interface", "fm_only", "managed_operation",
    "strict_2dh_numerical_activation", "not_yet_admitted",
))
_COUNT_KEYS = frozenset((
    "corrections", "added_entries", "output_selectors",
    "non_output_additions", "effective_lifecycle_rows",
    "effective_physical_methods", "stage_manifests",
))
_STAGE_SUMMARY_KEYS = frozenset((
    "stage_id", "path", "manifest_id", "sha256",
))
_STAGE_ORDER = (
    "Q0b", "Q4", "Q5a", "Q6a", "Q7a", "Q5b", "Q6b", "Q7b",
    "Q8a", "Q9", "Q8b", "Q9I", "Q10",
)
_STAGE_IDS = frozenset(_STAGE_ORDER)
_EXPECTED_STAGE_SUMMARY_IDS = {
    "Q0b": (
        "2cf6b373334ee6918c646f4dab4856ad85d9899d79789dd5921b6936fd058c04",
        "9531885153fd00ad3044a8cd53e148c21d0c3ff46bff68e310bf0264882005bb"),
    "Q4": (
        "99c32758983fa97defb07d705d2df47a71d4c12e3fad3f6249096012eafe4afe",
        "976131da5dc081d53a02f6b0e514b4d961b4b6bedcc2bdceb83a5b7fd7ff0689"),
    "Q5a": (
        "cfff693c4a91452e9e1dfe87cdd0e06a2e759a2aaa969103f026bb9df8124e35",
        "69527e4b44d1128353240f6eff15f2bf74da14990cb02f3387737ad038763a87"),
    "Q6a": (
        "859d1c51fada804ab7ce2d878dfc7d9e5772e49199ec1b0812dfb3fdc96b3aa2",
        "07a17e5f2989452e84250011c15d5a7066dd3e63ef1aec4006db0fdcde3cef41"),
    "Q7a": (
        "2a16e373cf37fe1ee9210c999c0f5fc1d67c57192762b3e60e86b65bd67c26ee",
        "f390b7ef010ac61a94d53c6437d39fc16b53bf006e5e5f547ba148fdd29d9258"),
    "Q5b": (
        "35f3705175b23fcd148fb494f6c360932564f43c99e3fa32951188b716dc5e0b",
        "677d393ef10fb95ffda2979f3ef35659dc16e30cc9db9172d559a00bc8a2cd60"),
    "Q6b": (
        "bd880c49493e73802c9cc3c99dfd539482da3d96ae19632847ef4fb45dae7432",
        "a978a0eddbdb5850f6a52b231b4eb31942d8b38ae08b0629f4799220393cffc7"),
    "Q7b": (
        "8a972890c6d7cf95ef766d7cca877a8aad36f6d51e483a9c41b755b2314b209d",
        "2745b3a16b6a8edb95176768a2d5c1072164e8c3fecdfd26d1fc55bf5423dfd9"),
    "Q8a": (
        "26567d35766e6655216912c0e970eb466bdb271961083cc9f58800986ee23e19",
        "aba7a21a0479345beee74759613bbaba677dae2d9eca3845296d812cc3aebe6b"),
    "Q9": (
        "f8c36a38fee8b4675a9a14f1ebf16882da1c46c924c5cd256c2f8c3606ab79cf",
        "7d6f142efa679c6950594512d046b610411afbfe778908d62bb9445f86968ce2"),
    "Q8b": (
        "f4f46507d9b37c6ebd18c66be34a4ffe3e819198ef4b90d6a820d2451026748d",
        "43a17043fb2d68a3f132486223c20d324c57f41a24b28479ab100b59a95a3fb4"),
    "Q9I": (
        "b29cf4cf678eaf6404e910acd9a3790e5a90789edbc33633be58a57971c581b8",
        "ed906cb58d460cefe16d7e5b5191044543d721879b77a7207a446fa5951f923c"),
    "Q10": (
        "8a08414b176b4b5c77b7320ec337ee0f58d46701f1557f91814bebcb431a041d",
        "edda1451b9ce7d4f653dd8b3c46dc8058e009488f73773803b2386f59ff41c75"),
}
_CORRECTION_ENTRY_IDS = frozenset((
    "parameter.redistribution.bermslope",
    "parameter.redistribution.bermslopebed",
    "parameter.redistribution.bermslopedepth",
    "parameter.redistribution.bermslopefac",
    "parameter.redistribution.bermslopegamma",
    "parameter.redistribution.bermslopesus",
    "parameter.redistribution.bermslopetransport",
    "capacity.user_dll", "capacity.external_subroutine_21",
    "parameter.suspension.mudtps",
    "parameter.suspension.schmidtnumber",
    "parameter.suspension.schmidtnumbermud",
    "parameter.suspension.schmidtnumbersand",
    "parameter.fine_sediment.bsskin",
    "redistribution.avalanche",
    "parameter.redistribution.duneavalan",
    "parameter.redistribution.dryslope",
    "parameter.redistribution.hswitch",
    "parameter.redistribution.dzmaxdune",
))
_NON_OUTPUT_ADDITION_IDS = frozenset((
    "payload.bed_composition.inicomp",
    "payload.morphology_boundary.bcfil",
    "payload.point_discharge.timeseries",
    "payload.point_discharge.geometry",
    "internal_load.suspended_source",
    "internal_load.suspended_sink",
    "payload.point_discharge.fraction_mapping",
    "payload.managed.fildad",
    "managed.dredging", "managed.dumping", "managed.sand_mining",
    "managed.nourishment",
    "redistribution.dflowfm_dune_avalanche",
    "redistribution.braids_conservative_wet_dry_avalanche",
    "payload.scour.file",
    "redistribution.classic_scour_bundle",
))
_ADDED_KIND_METHOD_IDS = frozenset((
    "redistribution.dflowfm_dune_avalanche",
    "redistribution.braids_conservative_wet_dry_avalanche",
    "redistribution.classic_scour_bundle",
))
_ADDED_COUNTED_METHOD_IDS = frozenset((
    "redistribution.classic_scour_bundle",
))
_FM_DUNE_AVALANCHE_ID = "redistribution.dflowfm_dune_avalanche"
_BRAIDS_CONSERVATIVE_AVALANCHE_ID = (
    "redistribution.braids_conservative_wet_dry_avalanche"
)
_FM_DUNE_PARAMETER_DEFAULTS = {
    "DrySlope": 1.0,
    "Hswitch": 0.1,
    "DzMaxDune": 100.0,
}
_FM_DUNE_PARAMETER_IDS = frozenset((
    "parameter.redistribution.duneavalan",
    "parameter.redistribution.dryslope",
    "parameter.redistribution.hswitch",
    "parameter.redistribution.dzmaxdune",
))
_CLASSIC_WET_SLOPE_ID = "redistribution.avalanche"
_WET_SLOPE_OTHER_CONSUMERS = frozenset((
    "capacity.partheniades_krone_difference",
    "capacity.partheniades_krone_ratio",
))
_SCOUR_PAYLOAD_ID = "payload.scour.file"
_CLASSIC_SCOUR_ID = "redistribution.classic_scour_bundle"
_DRY_BANK_ID = "redistribution.dry_bank"
_HOST_INTERFACE_IDS = frozenset((
    "capacity.user_dll", "capacity.external_subroutine_21",
))
_HOST_ROUTE_TAGS = frozenset((15, 21))
_HOST_ALIASES = frozenset((
    "DLL", "TranspLib", "function", "TranspFunction", "InputFile",
    "TranspInput", "SettleLib", "SettleFunction", "SettleInput",
))
_TRANSPORT_SELECTOR_ALIASES = (
    "TraFrm", "TranspFrm", "iform", "transport_formula",
)
# Raw Delft3D selector values are not execution identities.  Until the
# catalogue exposes and binds their typed rows, accepting these keys would let
# the inherited Q0 resolver treat them as unrelated metadata and silently run
# different physics.  Keep this inventory explicit (rather than matching name
# patterns) so a source selector cannot become admitted accidentally.
_OTHER_RAW_TYPED_SELECTOR_KEYWORDS = (
    "SettleFrm", "IALDiff", "IDiffusion", "IErosion", "IConsolidate",
    "IPorosity", "BurType", "FlocModel", "Type", "UpdBaseLyr",
    "UpdTopLyr", "IThresh", "BdfMor", "CrsLyr", "NeuBcSand", "SedTyp",
    "NeuBcMud", "BsSkin",
)
_RAW_TYPED_SELECTOR_REASON = (
    "Use a canonical typed catalogue entry ID for this method selector."
)
_EXPECTED_BSSKIN_VALUE_SCHEMA = {
    "default": {
        "anchors": [
            "src/utils_gpl/morphology/packages/morphology_io/src/rdsed.f90:579"
        ],
        "status": "source_proven",
        "value": False,
    },
    "range": {
        "anchors": [
            "src/utils_gpl/morphology/packages/morphology_io/src/rdsed.f90:579-581"
        ],
        "status": "source_proven",
        "value": [False, True],
    },
    "type": "boolean",
    "units": {
        "anchors": [],
        "reason": "A logical method selector has no physical unit.",
        "status": "not_applicable",
    },
}
_EXPECTED_VALUE_TRIGGER_OWNER_GUARDS = [
    {
        "keywords": ["PmCrit"],
        "required_method_id": "mixed_sediment.sand_mud_pmcrit",
        "failure_category": "scope_incompatibility",
        "failure_phase": "before_solver_allocation",
        "reason": (
            "PmCrit has no consumer unless "
            "mixed_sediment.sand_mud_pmcrit is selected."
        ),
        "value_trigger": {
            "inactive": ["null", "finite_number_lt_0"],
            "active": [
                "finite_number_gte_0", "nonblank_string_spatial_field_path"],
            "invalid": [
                "boolean", "nonfinite_number", "blank_string",
                "all_other_types"],
            "invalid_failure_category": "invalid_parameter",
            "invalid_failure_phase": "before_solver_allocation",
        },
    },
    {
        "keywords": ["ReturnTime", "return_time"],
        "required_method_id": "suspended_boundary.thatcher_harleman",
        "failure_category": "scope_incompatibility",
        "failure_phase": "before_solver_allocation",
        "reason": (
            "ReturnTime has no consumer unless "
            "suspended_boundary.thatcher_harleman is selected."
        ),
        "value_trigger": {
            "inactive": ["null", "blank_string", "finite_number_eq_0"],
            "active": ["finite_number_ne_0"],
            "invalid": [
                "boolean", "nonfinite_number", "nonblank_string",
                "all_other_types"],
            "invalid_failure_category": "invalid_parameter",
            "invalid_failure_phase": "before_solver_allocation",
        },
    },
]
_OPTIONAL_FILE_PARAMETER_KEYWORDS = frozenset((
    "IniComp", "BcFil", "Filsrc", "Fildis", "Fildad", "Scour",
))
_VALUE_SENSITIVE_NOT_ADMITTED_KEYWORDS = frozenset(("DuneAvalan",))
_SCOPE_EXCLUDED_OUTPUT_KEYWORDS = frozenset((
    "MorStatsOutputInterval", "BedTranspDueToWavesAtZeta",
    "BedTranspDueToWavesAtFlux", "SuspTranspDueToWavesAtZeta",
    "SuspTranspDueToWavesAtFlux", "OrbitalVel",
))
_SCOPE_EXCLUDED_OUTPUT_IDS = frozenset((
    "output.mor_stats_output_interval",
    "output.bed_transp_due_to_waves_at_zeta",
    "output.bed_transp_due_to_waves_at_flux",
    "output.susp_transp_due_to_waves_at_zeta",
    "output.susp_transp_due_to_waves_at_flux",
    "output.orbital_vel",
))
_BERM_IDS = frozenset(
    entry_id for entry_id in _CORRECTION_ENTRY_IDS
    if entry_id.startswith("parameter.redistribution.bermslope")
)
_BERM_KEYWORDS = frozenset(
    entry_id.rsplit(".", 1)[-1] for entry_id in _BERM_IDS
)
_MANAGED_IDS = frozenset((
    "payload.managed.fildad", "managed.dredging", "managed.dumping",
    "managed.sand_mining", "managed.nourishment",
))
_MANAGED_KEYWORDS = frozenset((
    "Fildad", "dredging", "dumping", "sand_mining", "nourishment",
))
_STRICT_2DH_METADATA_IDS = frozenset((
    "parameter.suspension.mudtps",
    "parameter.suspension.schmidtnumber",
    "parameter.suspension.schmidtnumbermud",
    "parameter.suspension.schmidtnumbersand",
))
_UNRESOLVED_BED_BOUNDARY_IDS = frozenset((
    "bed_boundary.mode_6", "bed_boundary.mode_7",
))
_EXPECTED_SCOPE_CONTRACT = {
    "dimensionality": "strict_2DH_KMAX_1",
    "effective_scope_id": "delft3d4-flow-2dh-natural-unmanaged-river-q0b-v1",
    "engine": "Delft3D4 FLOW",
    "excludes": [
        "D-Flow FM-only operators", "waves and rollers", "3D-only numerics",
        "host DLL/subroutine physics", "managed Fildad interventions",
        "ASMITA and external MORmerge",
    ],
    "forcing": "current_only",
    "includes": [
        "locked classic FLOW 2DH built-in sediment physics",
        "BRAIDS CPU and user-selected implicit_fwave_midpoint GPU paths",
    ],
    "process_scope": "natural_unmanaged_river_morphodynamics",
}
_EXPECTED_CLAIMS = {
    "delft_source_modifications": 0,
    "exact_profile_admissions_added": 0,
    "inventory_presence_is_not_execution_admission": True,
    "output_classic_activation_requires_consumer_or_writer_trace": True,
    "output_defaults_source_proven": 69,
    "output_parser_presence_does_not_imply_classic_activation": True,
    "parent_catalogue_modifications": 0,
    "product_profile_admissions_added": 0,
    "simulation_steps": 0,
    "stage_manifest_identity_construction": (
        "stage core -> semantic manifest_id -> file bytes/SHA -> overlay "
        "registry; stage files do not embed overlay_manifest_id"
    ),
    "unknown_requests_fail_closed": True,
}
_EXPECTED_IDENTITY = {
    "added_entry_registry_id":
        "75c76c5b601b9df7a61708155595b7e26625e4b87739bf5ef5990b09604119a6",
    "base_binding_id":
        "e438d55f4654d27ac94896b7acd78d02f547e6f5c5fb97066fb1102d8539786a",
    "compatibility_registry_id":
        "0a894de4baf6642b5cf667cf292196913db683fbc2f012adabf03bf6cd92e2e6",
    "correction_registry_id":
        "d41d89c1b388961bfd0503d6e7cdc20f5c9162b39b3d73ac9fc93bf3a35e031d",
    "effective_scope_id": "delft3d4-flow-2dh-natural-unmanaged-river-q0b-v1",
    "overlay_manifest_id":
        "35b0b348eeb41358892525a6d87993a67b64bc3e2de33ba91333861c82ab3713",
    "stage_manifest_registry_id":
        "b3e5ccb5b5afdfef3bd9859208dedaccbdb13599b54c68a7b06113f76006ce97",
}
_EXPECTED_FILSRC_SCHEMA = {
    "type": "source_geometry_file",
    "source_name": {"type": "string", "maximum_characters": 20},
    "source_type": {
        "type": "integer_enum",
        "allowed": {
            "0": "regular", "1": "walking", "2": "power_station",
            "3": "culvert", "4": "culvert", "5": "culvert",
            "6": "q_type_power_station", "7": "user_dll_culvert",
            "8": "culvert",
        },
        "raw_code_case": "case_insensitive",
        "raw_code_to_canonical": {
            "N": 0, "W": 1, "P": 2, "C": 3, "E": 4,
            "D": 5, "Q": 6, "U": 7, "F": 8,
        },
    },
    "coordinates": {
        "inlet": "M,N,K integer triplet",
        "outfall": "M,N,K integer triplet",
        "outfall_required_for_raw_types": ["P", "Q", "C", "E", "D", "U", "F"],
    },
    "interpolation_flag": {
        "allowed": {"N": "block", "Y": "linear"},
        "applies_to": "linked Fildis time-series payload",
        "field": "Disint",
    },
    "layer_distribution": {
        "K=0": "depth_distributed_by_layer_thickness",
        "K>0": "single_selected_layer",
    },
    "time_series_requirement": {
        "join_key": "source_name",
        "momentum_columns": {
            "location": "Fildis time-series payload",
            "required_when": "linked discharge content/source profile is momentum",
        },
        "ordering": "one linked time-series record set per Filsrc source in source order",
        "payload_entry_id": "payload.point_discharge.timeseries",
        "selector_keyword": "Fildis",
    },
}
_EXPECTED_FILDIS_SCHEMA = {
    "type": "time_series_table",
    "columns": [
        {"identity": "time", "reference": "ITDATE", "units": "min"},
        {"identity": "discharge", "signed": True, "units": "m3/s"},
        {
            "identity": "one_concentration_per_suspended_fraction",
            "cardinality": "lsed", "units": "kg/m3",
        },
    ],
    "constituent_ordering_identity": {
        "array_alignment": "dimsedconst name/type reordering defines the shared indices",
        "suspended_fraction_prefix": (
            "first lsed constituents match suspended-sediment fractions"),
        "table_cardinality": "lstsc",
    },
    "interpolation_mode": {
        "type": "enum", "allowed": ["linear", "block"],
        "selector": "Disint",
        "owner_payload_entry_id": "payload.point_discharge.geometry",
    },
    "source_binding": {
        "cardinality": "one time-series record set per Filsrc source",
        "geometry_entry_id": "payload.point_discharge.geometry",
        "join_key": "source_name",
        "ordering": "preserve Filsrc per-source order",
    },
    "time": {
        "advancement": "incdis/upddis advance the external-table records",
        "cursor": "itdis(1:5,isrc)",
        "cursor_state": "mutable_runtime_table_record_indices",
        "initialization": "inidis initializes or reconstructs from Fildis and model time",
        "reference_date": "ITDATE from table metadata",
        "restart_requirement": (
            "Q6b must checkpoint the cursor or reconstruct it deterministically"),
        "unit": "min",
    },
    "optional_momentum_columns": [
        {"identity": "flow_magnitude", "units": "m/s"},
        {"identity": "flow_direction", "units": "degrees"},
    ],
}
_OUTPUT_ALIASES = {
    "OutDefault": "Default", "SedParOut": "SedPar",
    "td": "depositTime", "preload": "BedLayerPreload",
}
_OUTPUT_DESTINATIONS = {
    "Default": "all_boolean_output_flags",
    "OutDefault": "all_boolean_output_flags",
    "VelocAtZeta": "uuuvvv", "VelocMagAtZeta": "umod",
    "VelocZAtZeta": "zumod", "ShearVeloc": "ustar",
    "TranspType": "transptype", "BedTranspAtFlux": "sbuuvv",
    "SuspTranspAtFlux": "ssuuvv",
    "BedTranspDueToCurrentsAtZeta": "sbcuv",
    "BedTranspDueToCurrentsAtFlux": "sbcuuvv",
    "BedTranspDueToWavesAtZeta": "sbwuv",
    "BedTranspDueToWavesAtFlux": "sbwuuvv",
    "SuspTranspDueToWavesAtZeta": "sswuv",
    "SuspTranspDueToCurrentsAtZeta": "sscuv",
    "SuspTranspDueToWavesAtFlux": "sswuuvv",
    "TotalTransport": "sxytot", "NearBedRefConcentration": "rca",
    "EquilibriumConcentration": "rsedeq", "Concentration": "sedconc",
    "NearBedTranspCorrAtFlux": "suvcor", "SourceSinkTerms": "sourcesink",
    "ReferenceHeight": "aks", "SettlingVelocity": "ws",
    "RawTransportsAtZeta": "rawtransports", "OrbitalVel": "orbvel",
    "Seddif": "seddif", "SedParOut": "sedpar", "SedPar": "sedpar",
    "Bedslope": "dzduuvv", "EroPar": "eropar", "TcrEro": "tcrero",
    "Taub": "taub", "Taurat": "taurat", "Dm": "dm", "Dg": "dg",
    "Dgsd": "dgsd", "Frac": "frac", "MudFrac": "mudfrac",
    "SandFrac": "sandfrac", "FixFac": "fixfac", "HidExp": "hidexp",
    "CumNetSedimentationFlux": "dmsedcum",
    "BedLayerSedimentMass": "msed/bodsed",
    "BedLayerVolumeFractions": "lyrfrac",
    "BedLayerThickness": "dpsed/thlyr", "BedLayerDepth": "dpbedlyr",
    "BedLayerPorosity": "poros", "td": "depos_time",
    "depositTime": "depos_time", "preload": "preload",
    "BedLayerPreload": "preload", "FluffDepositionFlux": "depflxf",
    "FluffErosionFlux": "eroflxf", "FluffBurialFlux": "burflxf",
    "TimeAveragedTransport": "sxyavg", "AverageAtEachOutputTime": "cumavg",
    "Morfac": "morfac", "MainChannelAveragedBedLevel": "blave",
    "MainChannelCellArea": "bamor", "MainChannelWidthAtFlux": "wumor",
    "ALDiff": "aldiff", "MorStatsOutputInterval": "avgintv/morstats",
    "MorstatsWeightFactor": "weightflg", "StatWaterDepth": "statflg(:,1)",
    "StatVelocity": "statflg(:,2)", "StatBedLoad": "statflg(:,3)",
    "StatSuspLoad": "statflg(:,4)", "Percentiles": "percentiles",
}
_OUTPUT_TRUE_DEFAULTS = frozenset((
    "BedTranspAtFlux", "SuspTranspAtFlux", "TotalTransport",
    "NearBedRefConcentration", "EquilibriumConcentration", "Concentration",
    "SettlingVelocity", "BedLayerSedimentMass", "BedLayerVolumeFractions",
    "BedLayerThickness", "BedLayerDepth", "BedLayerPorosity",
    "TimeAveragedTransport", "Morfac",
))
_CLASSIC_ACTIVE_OUTPUTS = frozenset((
    "StatWaterDepth", "StatVelocity", "StatBedLoad", "StatSuspLoad",
))
_WAVE_ONLY_OUTPUTS = _SCOPE_EXCLUDED_OUTPUT_KEYWORDS - {
    "MorStatsOutputInterval"}
_OUTPUT_DEFAULT_SOURCE_PATH = (
    "src/utils_gpl/morphology/packages/morphology_data/src/"
    "morphology_data_module.f90"
)
_OUTPUT_DEFAULT_SOURCE_HASH = (
    "dd9ca54571498a067a118a19626d1779cd62c5cb558c391e9b8dd627c4e381ac"
)
_EXPECTED_DETAILED_PAYLOAD_SCHEMA_IDS = {
    "payload.bed_composition.inicomp":
        "0fc16c0b17c0e80958b239a6ed0fc30d81e4e100fd21f3df03c96b4fae8b1948",
    "payload.morphology_boundary.bcfil":
        "23eb3912aa049df27eef45942637a47fbae7686e3c1c261ef1baa43cd8c7a61a",
    "payload.managed.fildad":
        "16fd90cd1c24cd933a76b1b4a844d6e67548b59830ad7e3e1c43cc87b0534249",
}
_EXPECTED_AVALANCHE_SCOUR_SCHEMA_IDS = {
    "redistribution.avalanche":
        "6c16fc2734a612434f486e3e1eb1d62c9f03119de2b248aaab7b4b747798139c",
    "redistribution.dflowfm_dune_avalanche":
        "53157424f0f77e28be1ca0ebd4bafdf57888948f68dd5f8de6021ef8971befd0",
    "redistribution.braids_conservative_wet_dry_avalanche":
        "b4c627358de28c54cc82dff66982bc3d5970231273192a26513709534391bdf1",
    "payload.scour.file":
        "7fbee8d0e415caeda89fb259098d68cc3b56f5440e7f32745d9cf289a989db61",
    "redistribution.classic_scour_bundle":
        "fc2b35938820b40238dc4648b321adc3445d3cb9cdf3d4959945e67913d26f51",
}
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DELFT_SOURCE_ROOT = _REPOSITORY_ROOT / "resources" / "Delft3D-main"
# Immutable source-lock projection for every Delft3D anchor accepted by Q0b.
# The plugin can validate the registry without shipping the source checkout;
# a development checkout, when present, is authenticated byte-for-byte too.
_SOURCE_LOCKS = {
    "src/engines_gpl/flow2d3d/packages/flow2d3d_io/src/input/dimdis.f90":
        "58738d56555d57975f169843cea0b5c3f75448a50cbe38d9b51768b406e54604",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_io/src/input/dimpro.f90":
        "51fc9a27faed0ed5fbbf57f14b540f9af55dfbd30bfd363900342ae73149cbaf",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_io/src/input/"
    "dimsedconst.f90":
        "0173f1a270cbb37f397d754aa329ab32e75d0459b3c82a9787da4186d9eec74f",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_io/src/input/"
    "rddredge_d3d4.f90":
        "b7b71b1ddcc593ac34b79acac8d0c14b2e9e56719edc87cbeaaa4ec376b339af",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_io/src/input/rdspec.f90":
        "4942bfddc7ab03741d711a080d45bd93e79552b7e30b73bab1a6fcadc7bb746c",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_io/src/input/rdscour.f90":
        "9be20af5f3bef3e2ad0009b4729614ef0dc7547e4f24cb4fe939c6ea3506f9a0",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_io/src/input/rdbedformpar.f90":
        "061bf0be2d93f8edbd540f4ccd30b743eeff15996c70d4475f75318697328ec0",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_io/src/input/srcfil.f90":
        "4e108f939804f51cabdaf842ffab6466e7a1f8432782d6a78b368636b9d52343",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_io/src/preprocessor/rddis.f90":
        "d62a4232e7fd1cb7c86e954cfc99425a9c88212c602a5270d6cab9b2a94d9d12",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute/difu.f90":
        "e8e11d438a6e3aa8e912ce1133efdec093b5bad60605aefa6c4b037ea6aa5c17",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute/difuvl.f90":
        "333bda22e3e568c4a5bd478d966544d922a9e29f8ed54edcecc118b3d49791f8",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute/discha.f90":
        "46277dd8d90390455e177b6c6e55512aea2d50f804b65817d6338731b0204849",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute/culver.f90":
        "0c97da609637e1f9ec19fd18e390023c782d54c12e37c5acf3fe99451f33e13b",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute/thahbc.f90":
        "e2c53f54fa0bf18e68d21b59e2d83658443ab1994bd833ea5d35181c8ce502a8",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute_sediment/"
    "bott3d.f90":
        "5abfee8230f66ba42b3eb87ff89948f65a0bfccc915a015c25f8c1412f4e47b1",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute_sediment/"
    "dredge_d3d4.f90":
        "6d688932ec97587c99eb8143797c2676b691b49d3e61bb85b53dbb6e61652442",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute_sediment/"
    "erosed.f90":
        "4571fd7912233db03f597161af1a18702ee974483676004cda8186ccf24308c0",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute_sediment/"
    "adjust_bedload.f90":
        "a3dae8226082977ec1ee78748e0d439f0f7356d267a9f33cba58636a61afdcba",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute_sediment/"
    "shearx.f90":
        "99834d93f88d1ffd35df9cfa279ea4f91afeafa43a506ce59a02797878ac016c",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute_sediment/"
    "avalan.f90":
        "33a928eaf236bfd4db9047b911b5cdd50064659c92071d35ca682c3f70e88728",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/inichk/tkecof.f90":
        "3191875709af047d43ed2354e17a17a16071f7c79ec6b40881902a1cf7898cf9",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/inichk/inidis.f90":
        "0b045f9d983113ca8ad075d007fd1c3a229cae5f7b7c38924aef01f5d2c143af",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/timedep/upddis.f90":
        "fd5ad886bdf6d95668ea5f32f3bee5e38ec7882acf6ce2ae8e9b7b8da1544827",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/timedep/incdis.f90":
        "9001c7b67dd0c62a51a8d8354e29bbb9bb10b04a0cd96cc5b9dd28265d4af339",
    "src/utils_gpl/morphology/packages/morphology_data/src/"
    "dredge_data_module.f90":
        "881c466c51dac872b6dd083d38059a4ddc896883a3b531e98552ecacc99c0050",
    "src/utils_gpl/morphology/packages/morphology_data/src/"
    "morphology_data_module.f90":
        "dd9ca54571498a067a118a19626d1779cd62c5cb558c391e9b8dd627c4e381ac",
    "src/utils_gpl/morphology/packages/morphology_io/src/rdmor.f90":
        "ab4e3a1bdf1290e790412375ea5a91cddf9039670948307169c38194314e0d49",
    "src/utils_gpl/morphology/packages/morphology_io/src/rdmorlyr.f90":
        "3da0bf1de13ef8cb288e78ab8bd6134acfd880f5308b1ee063c209e223dcd2ad",
    "src/utils_gpl/morphology/packages/morphology_io/src/rdsed.f90":
        "b3de2a67842935bee704eecee329e0e0a71fdd0bb95704a15f9d2ef567514fd2",
    "src/utils_gpl/morphology/packages/morphology_io/src/rddredge.f90":
        "8ddab86acefd070c051641ab16ffacd20eca2f7b67210104a6dbc59fff510f04",
    "src/utils_gpl/morphology/packages/morphology_io/src/rdstm.f90":
        "e2e54cf21ceef4e2b1b2f9e3cc26d8c3eb4df425c324b0b9b47828e7af7858b0",
    "src/utils_gpl/morphology/packages/morphology_io/src/rdtrafrm.f90":
        "0a2542980bc9e9e03e46b2abb8b06eda2fea3e28a67c78c4e61b619572154ea7",
    "src/utils_gpl/morphology/packages/morphology_kernel/src/eqtran.f90":
        "d2e8601cff68884f51ee3dcb0cd2fb52d9f0d676b1baaa3283c68913d4778388",
    "src/utils_gpl/morphology/packages/morphology_kernel/src/erosilt.f90":
        "78568b5a19251ef788be18406169a64f83cc3828b4510849bc9cfe5bdf7ec726",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_io/src/output/wrmorst.f90":
        "cc702179a4e7520246b584fc19955486572b32e338475b514bc5b02c64e4e9eb",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute_sediment/"
    "morstatistics.f90":
        "6414b84e7e2fd1f8eef73b57de841baa2229a9cd1f1b8bc8499b55ec8ec7325d",
    "src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute_sediment/"
    "bndmorlyr.f90":
        "8ea71243f39b9166d40e26b0e479d5aa29298e7ac8176f56de2d733452d38bea",
    "src/engines_gpl/dflowfm/packages/dflowfm_kernel/src/dflowfm_kernel/"
    "compute_sediment/duneaval.f90":
        "960838578eecd4c9bde49b4284d20093f31046bee18392616f25d2758e8bc1e2",
    "src/engines_gpl/dflowfm/packages/dflowfm_kernel/src/dflowfm_kernel/"
    "compute_sediment/m_fm_bott3d.f90":
        "8e406f7646ae2f3e77b74c89d24d929d14a2d8c9a5106f858b8279798a05ddda",
    "src/engines_gpl/dflowfm/packages/dflowfm_kernel/src/dflowfm_kernel/"
    "compute/setdt.f90":
        "9fa8b2da18ff1e7c9e7c3a24098059bec68500513822f3aeff18976dc02b6aee",
    "src/engines_gpl/dflowfm/packages/dflowfm_kernel/src/dflowfm_kernel/"
    "compute/setdtmaxavalan.f90":
        "58264951c7103b331cc2152b2a57f9191f733ded0cb7aa749f33bd83ff21bb7f",
}
# BRAIDS provenance is kept separate from the immutable Delft3D source lock:
# these anchors document an existing local implementation/history and never
# imply that the Delft3D reference was modified.  Generation authenticates
# these mutable local files; the packaged reader pins only the embedded
# path/hash so later BRAIDS development is not a runtime dependency.
_BRAIDS_SOURCE_LOCKS = {
    "solver/cut_cells/cutcell_solver.f90":
        "7f14c1531473ec5f246e4b6d79c1e11bc23d43407879732091804a6ae6c58f2c",
    "braids_core/boundaries/cut_cell_io.py":
        "e81249f4e87e77912cc9c117b19dd237880a3893951d3f727b757c86a13b2c0b",
}
_IMPLEMENTED_STATES = frozenset(("live_route", "legacy_live_route"))
_CONNECTED_STATE = "connected"


def _fail(message: str, *, category: str = "catalogue_integrity") -> None:
    raise LiveCatalogueError(message, category=category)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"Duplicate JSON key: {key!r}", category="invalid_serialization")
        result[key] = value
    return result


def _read_json(payload: bytes | str | Mapping[str, object]) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return copy.deepcopy(dict(payload))
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {item}")),
        )
    except LiveCatalogueError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LiveCatalogueError(
            "Q0b scope overlay is not strict JSON",
            category="invalid_serialization",
        ) from exc
    if not isinstance(value, dict):
        _fail("Q0b scope overlay must be one JSON object",
              category="invalid_serialization")
    return value


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise LiveCatalogueError(
            "Q0b scope overlay is not canonicalizable",
            category="invalid_serialization",
        ) from exc


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _same(left: object, right: object) -> bool:
    return _canonical(left) == _canonical(right)


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{label} is not a SHA-256 digest")
    return value


def _require_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} path differs")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        _fail(f"{label} path is not repository-relative")
    return value.replace("\\", "/")


def _require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} differs")
    return value


def _validate_owner(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _OWNER_KEYS:
        _fail(f"{label} owner differs")
    return {
        "milestone": _require_nonempty_string(value.get("milestone"), label),
        "responsibility": _require_nonempty_string(
            value.get("responsibility"), label),
    }


def _validate_fail_closed(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _FAIL_CLOSED_KEYS:
        _fail(f"{label} fail-closed disposition differs")
    if type(value.get("enabled")) is not bool:
        _fail(f"{label} fail-closed switch differs")
    category = value.get("category")
    phase = value.get("phase")
    if value["enabled"]:
        _require_nonempty_string(category, f"{label} failure category")
        _require_nonempty_string(phase, f"{label} failure phase")
    elif category is not None or phase is not None:
        _fail(f"{label} disabled fail-closed row carries a failure")
    return copy.deepcopy(dict(value))


def _validate_source_anchors(value: object, label: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        _fail(f"{label} source-anchor inventory differs")
    result: list[dict[str, object]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            _fail(f"{label} source anchor {index} differs")
        required = {"path", "lines", "sha256"}
        optional = {"symbol", "role", "symbols"}
        if not required.issubset(raw) or not set(raw).issubset(required | optional):
            _fail(f"{label} source anchor {index} fields differ")
        source_path = _require_relative_path(
            raw.get("path"), f"{label} source anchor")
        _require_nonempty_string(raw.get("lines"), f"{label} source lines")
        source_hash = _require_digest(raw.get("sha256"), f"{label} source hash")
        if source_path in _SOURCE_LOCKS:
            if _SOURCE_LOCKS[source_path] != source_hash:
                _fail(f"{label} source anchor is not in the immutable Delft lock")
            installed_source = _DELFT_SOURCE_ROOT / Path(source_path)
            source_kind = "locked Delft"
        elif source_path in _BRAIDS_SOURCE_LOCKS:
            if _BRAIDS_SOURCE_LOCKS[source_path] != source_hash:
                _fail(f"{label} source anchor is not in the immutable BRAIDS lock")
            # This is historical provenance of an existing BRAIDS component,
            # not a frozen runtime dependency.  Generation/check authenticates
            # the live bytes; later solver development must not invalidate an
            # already installed catalogue merely because this source changes.
            installed_source = None
            source_kind = "locked BRAIDS"
        else:
            _fail(f"{label} source anchor is not in an immutable source lock")
        if installed_source is not None and installed_source.is_file():
            try:
                installed_hash = hashlib.sha256(installed_source.read_bytes()).hexdigest()
            except OSError as exc:
                raise LiveCatalogueError(
                    f"{label} {source_kind} source cannot be authenticated",
                    category="catalogue_integrity",
                ) from exc
            if installed_hash != source_hash:
                _fail(f"{label} {source_kind} source bytes differ")
        if "symbol" in raw:
            _require_nonempty_string(raw.get("symbol"), f"{label} source symbol")
        if "role" in raw:
            _require_nonempty_string(raw.get("role"), f"{label} source role")
        if "symbols" in raw and (not isinstance(raw["symbols"], list) or
                                  not raw["symbols"] or
                                  not all(isinstance(item, str) and item
                                          for item in raw["symbols"])):
            _fail(f"{label} source symbols differ")
        result.append(copy.deepcopy(dict(raw)))
    return result


def _pointer_parts(pointer: object, label: str) -> tuple[str, ...]:
    if not isinstance(pointer, str) or not pointer.startswith("/") or pointer == "/":
        _fail(f"{label} JSON pointer differs")
    parts: list[str] = []
    for encoded in pointer[1:].split("/"):
        if re.search(r"~(?![01])", encoded):
            _fail(f"{label} JSON pointer escape differs")
        decoded = encoded.replace("~1", "/").replace("~0", "~")
        if not decoded or decoded == "-":
            _fail(f"{label} JSON pointer component differs")
        parts.append(decoded)
    return tuple(parts)


def _validate_patches(
    value: object, label: str, *, forbidden_roots: Iterable[str] = (),
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        _fail(f"{label} patches differ")
    paths: list[str] = []
    result: list[dict[str, object]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != _PATCH_KEYS:
            _fail(f"{label} patch {index} fields differ")
        pointer = raw.get("path")
        parts = _pointer_parts(pointer, f"{label} patch {index}")
        if parts[0] in set(forbidden_roots):
            _fail(f"{label} patch {index} targets immutable identity/admission data")
        assert isinstance(pointer, str)
        paths.append(pointer)
        # Canonicalization also rejects non-finite values in injected mappings.
        _canonical(raw.get("expected"))
        _canonical(raw.get("effective"))
        result.append(copy.deepcopy(dict(raw)))
    if len(paths) != len(set(paths)) or paths != sorted(paths):
        _fail(f"{label} patch paths are duplicate or non-canonical")
    return result


def _resolve_pointer(document: object, parts: Sequence[str], label: str) -> tuple[object, str]:
    target = document
    for part in parts[:-1]:
        if isinstance(target, Mapping):
            if part not in target:
                _fail(f"{label} does not match its parent")
            target = target[part]
        elif isinstance(target, list):
            if not part.isdigit() or int(part) >= len(target):
                _fail(f"{label} does not match its parent")
            target = target[int(part)]
        else:
            _fail(f"{label} traverses a scalar")
    return target, parts[-1]


def _apply_patches(document: object, patches: Sequence[Mapping[str, object]],
                   label: str) -> object:
    result = copy.deepcopy(document)
    for index, patch in enumerate(patches):
        parts = _pointer_parts(patch.get("path"), f"{label} patch {index}")
        target, leaf = _resolve_pointer(result, parts, f"{label} patch {index}")
        if isinstance(target, dict):
            if leaf not in target or not _same(target[leaf], patch.get("expected")):
                _fail(f"{label} patch {index} expected value differs")
            target[leaf] = copy.deepcopy(patch.get("effective"))
        elif isinstance(target, list):
            if not leaf.isdigit() or int(leaf) >= len(target):
                _fail(f"{label} patch {index} list index differs")
            offset = int(leaf)
            if not _same(target[offset], patch.get("expected")):
                _fail(f"{label} patch {index} expected value differs")
            target[offset] = copy.deepcopy(patch.get("effective"))
        else:
            _fail(f"{label} patch {index} target differs")
    return result


def _installed_parent_bytes() -> bytes:
    path = _v1.LIVE_CATALOGUE_FILE
    if not path.is_file():
        _fail("Installed Q0/Q0a live catalogue is missing", category="missing_catalogue")
    return path.read_bytes()


def _installed_parent() -> dict[str, Any]:
    return _v1.load_live_catalogue()


def _installed_selector_ledger(parent: Mapping[str, object]) -> dict[str, Any]:
    if not PARENT_LEDGER_FILE.is_file():
        _fail("Installed selector-ledger parent is missing", category="missing_catalogue")
    payload = PARENT_LEDGER_FILE.read_bytes()
    binding = parent.get("parent_ledger")
    if not isinstance(binding, Mapping) or hashlib.sha256(payload).hexdigest() != \
            binding.get("sha256"):
        _fail("Installed selector-ledger parent hash differs")
    document = _read_json(payload)
    if document.get("semantic_id") != binding.get("semantic_id"):
        _fail("Installed selector-ledger parent identity differs")
    return document


def scope_identity_payloads(overlay: Mapping[str, object]) -> dict[str, str]:
    """Return the independent Q0b identity projections."""
    base = copy.deepcopy(overlay.get("base_live_catalogue"))
    base_id = _hash({"schema": 1, "base_live_catalogue": base})
    correction_id = _hash({
        "schema": 1, "base_binding_id": base_id,
        "corrections": copy.deepcopy(overlay.get("corrections")),
    })
    addition_id = _hash({
        "schema": 1, "base_binding_id": base_id,
        "added_entries": copy.deepcopy(overlay.get("added_entries")),
    })
    compatibility_id = _hash({
        "schema": 1, "base_binding_id": base_id,
        "compatibility_rules": copy.deepcopy(overlay.get("compatibility_rules")),
        "request_policy": copy.deepcopy(overlay.get("request_policy")),
    })
    stage_id = _hash({
        "schema": 1, "base_binding_id": base_id,
        "stage_manifests": copy.deepcopy(overlay.get("stage_manifests")),
    })
    scope = overlay.get("scope_contract")
    effective_scope_id = scope.get("effective_scope_id") \
        if isinstance(scope, Mapping) else None
    manifest_id = _hash({
        "schema": 1,
        "revision_id": overlay.get("revision_id"),
        "base_binding_id": base_id,
        "correction_registry_id": correction_id,
        "added_entry_registry_id": addition_id,
        "compatibility_registry_id": compatibility_id,
        "stage_manifest_registry_id": stage_id,
        "effective_scope_id": effective_scope_id,
        "counts": copy.deepcopy(overlay.get("counts")),
        "claims": copy.deepcopy(overlay.get("claims")),
    })
    return {
        "base_binding_id": base_id,
        "correction_registry_id": correction_id,
        "added_entry_registry_id": addition_id,
        "compatibility_registry_id": compatibility_id,
        "stage_manifest_registry_id": stage_id,
        "effective_scope_id": effective_scope_id,
        "overlay_manifest_id": manifest_id,
    }


def _validate_lifecycle(value: object, entry_id: str, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _LIFECYCLE_KEYS:
        _fail(f"{label} lifecycle fields differ")
    if value.get("entry_id") != entry_id:
        _fail(f"{label} lifecycle identity differs")
    if value.get("namespace") != "delft3d4_flow_2dh_q0b":
        _fail(f"{label} lifecycle namespace differs")
    for backend in BACKENDS:
        for field in ("implementation", "accepted_state_connection", "exact_profile_refs"):
            branch = value.get(field)
            if not isinstance(branch, Mapping) or set(branch) != set(BACKENDS):
                _fail(f"{label} {field} backend inventory differs")
            if backend not in branch:
                _fail(f"{label} {field} backend inventory differs")
        products = value.get("product_profile_refs")
        if not isinstance(products, Mapping) or set(products) != {
                "qgis_prepare_run", "windows_package"}:
            _fail(f"{label} product surfaces differ")
        for surface in products.values():
            if not isinstance(surface, Mapping) or set(surface) != set(BACKENDS):
                _fail(f"{label} product backend inventory differs")
    owners = value.get("owner_edges")
    if not isinstance(owners, list) or len(owners) != 1:
        _fail(f"{label} must have one primary lifecycle owner")
    _validate_owner(owners[0], label)
    for backend in BACKENDS:
        implementation = value["implementation"][backend]
        accepted = value["accepted_state_connection"][backend]
        exact_refs = value["exact_profile_refs"][backend]
        allowed_implementation_states = {"missing", "not_applicable"}
        # This one additive row preserves already-existing BRAIDS CPU
        # component history under its own identity.  It is deliberately not
        # an execution admission: accepted-state and every profile/product
        # reference remain empty below.  No Delft-derived added row receives
        # this exception.
        if entry_id == _BRAIDS_CONSERVATIVE_AVALANCHE_ID and backend == "cpu":
            allowed_implementation_states.add("component_present")
        if not isinstance(implementation, Mapping) or \
                implementation.get("state") not in allowed_implementation_states:
            _fail(f"{label} manufactures an implementation admission")
        if implementation.get("state") == "component_present" and (
                not isinstance(implementation.get("evidence"), list) or
                not implementation["evidence"]):
            _fail(f"{label} component history lacks evidence")
        if not isinstance(accepted, Mapping) or accepted.get("state") not in {
                "not_connected", "not_applicable"}:
            _fail(f"{label} manufactures an accepted-state connection")
        if exact_refs != []:
            _fail(f"{label} manufactures an exact-profile admission")
        for surface in value["product_profile_refs"].values():
            if surface[backend] != []:
                _fail(f"{label} manufactures a product-profile admission")
    if value.get("legacy_surface") != {
            "exposed": False, "marker": None,
            "is_product_admission": False}:
        _fail(f"{label} manufactures a legacy/product admission")
    return copy.deepcopy(dict(value))


def _validate_correction_lifecycle_downgrade(
    parent: Mapping[str, object], effective: Mapping[str, object], label: str,
) -> None:
    promoted_states = {"component_present", "legacy_live_route", "live_route"}
    try:
        for backend in BACKENDS:
            parent_impl = parent["implementation"][backend]["state"]
            effective_impl = effective["implementation"][backend]["state"]
            if effective_impl in promoted_states and effective_impl != parent_impl:
                _fail(f"{label} promotes an implementation state")
            parent_connection = parent[
                "accepted_state_connection"][backend]["state"]
            effective_connection = effective[
                "accepted_state_connection"][backend]["state"]
            if effective_connection == "connected" and \
                    parent_connection != "connected":
                _fail(f"{label} promotes an accepted-state connection")
    except (KeyError, TypeError):
        _fail(f"{label} lifecycle state shape differs")
    parent_legacy = parent.get("legacy_surface")
    effective_legacy = effective.get("legacy_surface")
    if isinstance(parent_legacy, Mapping) and isinstance(effective_legacy, Mapping):
        if (not parent_legacy.get("exposed") and effective_legacy.get("exposed")) or \
                (not parent_legacy.get("is_product_admission") and
                 effective_legacy.get("is_product_admission")):
            _fail(f"{label} promotes legacy/product admission")


def _unique_casefold(values: Iterable[object]) -> list[str]:
    result: dict[str, str] = {}
    for value in values:
        if not isinstance(value, str):
            continue
        rendered = value.strip()
        if rendered:
            result.setdefault(rendered.casefold(), rendered)
    return sorted(result.values(), key=lambda item: (item.casefold(), item))


def _validate_request_policy(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _REQUEST_POLICY_KEYS:
        _fail("Q0b request-policy fields differ")
    for name in _REQUEST_POLICY_LISTS:
        items = value.get(name)
        if not isinstance(items, list):
            _fail(f"Q0b request-policy {name} differs")
        if name == "host_route_tags":
            if not all(isinstance(item, int) and not isinstance(item, bool)
                       for item in items):
                _fail(f"Q0b request-policy {name} differs")
        elif not all(isinstance(item, str) and item for item in items):
            _fail(f"Q0b request-policy {name} differs")
        if len({_canonical(item) for item in items}) != len(items):
            _fail(f"Q0b request-policy {name} has duplicate values")
    for name in _REQUEST_POLICY_MAPS:
        mapping = value.get(name)
        if not isinstance(mapping, Mapping) or set(mapping) != _POLICY_DISPOSITIONS:
            _fail(f"Q0b request-policy {name} differs")
        for key, item in mapping.items():
            _require_nonempty_string(key, f"Q0b request-policy {name} key")
            _require_nonempty_string(item, f"Q0b request-policy {name} value")
    return copy.deepcopy(dict(value))


def _output_slug(keyword: str) -> str:
    if keyword == "ALDiff":
        return "aldiff"
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", keyword)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first).lower()


def _output_anchor(
    path: str, lines: str, role: str, symbols: Sequence[str],
) -> dict[str, object]:
    return {
        "path": path, "lines": lines, "sha256": _SOURCE_LOCKS[path],
        "role": role, "symbols": list(symbols),
    }


def _expected_output_value_schema(keyword: str) -> dict[str, object]:
    destination = _OUTPUT_DESTINATIONS[keyword]
    if keyword in {"Default", "OutDefault"}:
        schema: dict[str, object] = {
            "type": "boolean",
            "meaning": "optional bulk output-default override",
        }
        default: object = None
        effect: object | None = {
            "state": "selector_absent",
            "effect": "preserve_initmoroutput_per_field_defaults",
        }
    elif keyword == "TranspType":
        schema = {
            "type": "integer_enum", "allowed": [0, 1, 2],
            "meaning": "mass or volume transport output units",
        }
        default = 2
        effect = {
            "unit_basis": "volume_excluding_pores", "sediment_amount": "m3"}
    elif keyword == "MorStatsOutputInterval":
        schema = {"type": "number_vector", "length": 3}
        default = [-999.0, 0.0, 0.0]
        effect = {
            "raw_initialization": [-999.0, -999.0, -999.0],
            "absent_path_normalization": (
                "negative second element sets elements 2 and 3 to zero"),
            "morstats": False,
        }
    elif keyword == "MorstatsWeightFactor":
        schema = {"type": "string_selector"}
        default = 1
        effect = {"selector_string": "", "destination": "weightflg"}
    elif keyword in _CLASSIC_ACTIVE_OUTPUTS:
        schema = {"type": "string_selector"}
        default = ""
        effect = {"statflg": [0, 0, 0, 0, 0, 0]}
    elif keyword == "Percentiles":
        schema = {"type": "boolean_or_percentile_list"}
        default = ""
        effect = {"percentiles": False, "selector_string": ""}
    else:
        schema = {"type": "boolean", "meaning": "request named result field"}
        canonical_keyword = _OUTPUT_ALIASES.get(keyword, keyword)
        default = canonical_keyword in _OUTPUT_TRUE_DEFAULTS
        effect = None
    schema.update({
        "destination": destination,
        "source_default": default,
        "source_default_provenance": _output_anchor(
            _OUTPUT_DEFAULT_SOURCE_PATH, "1778-1867",
            "initmoroutput exact source default", [destination]),
    })
    if effect is not None:
        schema["source_default_effect"] = effect
    if keyword in {"Default", "OutDefault"}:
        schema["source_default_effect_provenance"] = _output_anchor(
            "src/utils_gpl/morphology/packages/morphology_io/src/rdmor.f90",
            "869-873", "optional aggregate override absence semantics",
            [keyword, "initmoroutput"],
        )
    elif keyword == "MorStatsOutputInterval":
        schema["source_default_effect_provenance"] = _output_anchor(
            "src/utils_gpl/morphology/packages/morphology_io/src/rdmor.f90",
            "965-973", "absent interval normalization and FM stream activation",
            ["avgintv", "morstats"],
        )
    elif keyword == "MorstatsWeightFactor":
        schema["source_default_effect_provenance"] = _output_anchor(
            "src/utils_gpl/morphology/packages/morphology_io/src/rdmor.f90",
            "974-982", "blank selector and destination mapping", ["weightflg"],
        )
    elif keyword in _CLASSIC_ACTIVE_OUTPUTS:
        schema["source_default_effect_provenance"] = _output_anchor(
            "src/utils_gpl/morphology/packages/morphology_io/src/rdmor.f90",
            "986-1032", "blank selector and zero statistic flags", [destination],
        )
    elif keyword == "Percentiles":
        schema["source_default_effect_provenance"] = _output_anchor(
            "src/utils_gpl/morphology/packages/morphology_io/src/rdmor.f90",
            "1037-1065", "blank percentile selector default", ["percentiles"],
        )
    return schema


def _validate_output_entry(raw: Mapping[str, object], label: str) -> None:
    selector = raw.get("selector")
    keyword = selector.get("keyword") if isinstance(selector, Mapping) else None
    if keyword not in _OUTPUT_DESTINATIONS or selector != {
            "section": "Output", "keyword": keyword, "value": "requested"}:
        _fail(f"{label} Output selector identity differs")
    assert isinstance(keyword, str)
    expected_entry_id = f"output.{_output_slug(keyword)}"
    canonical_keyword = _OUTPUT_ALIASES.get(keyword, keyword)
    canonical_slug = _output_slug(canonical_keyword)
    if (raw.get("entry_id") != expected_entry_id or
            raw.get("result_field_id") != f"result.morphology.{canonical_slug}" or
            raw.get("checkpoint_field_id") !=
            f"checkpoint.morphology_output.{canonical_slug}" or
            not _same(raw.get("value_schema"),
                      _expected_output_value_schema(keyword))):
        _fail(f"{label} Output default/provenance/wiring identity differs")

    if keyword == "MorStatsOutputInterval":
        status = "dflow_fm_only"
        applicability: object = {
            "scope": "strict_classic_flow_2dh",
            "disposition": "not_applicable_fm_only",
        }
        category = "scope_incompatibility"
        excluded = True
    elif keyword in _WAVE_ONLY_OUTPUTS:
        status = "classic_flow_wave_only"
        applicability = {
            "scope": "current_only",
            "disposition": (
                "not_applicable_to_current_only_scope_requires_"
                "wave_or_orbital_forcing"),
        }
        category = "scope_incompatibility"
        excluded = True
    elif keyword in _CLASSIC_ACTIVE_OUTPUTS:
        status = "classic_flow_active"
        applicability = {
            "scope": "strict_classic_flow_2dh_current_only",
            "disposition": "catalogued_pending_q9i_wiring",
        }
        category = "implementation_restriction"
        excluded = False
    else:
        status = "classic_flow_parser_only_pending_consumer"
        applicability = {
            "scope": "strict_classic_flow_2dh_current_only",
            "disposition": (
                "catalogued_pending_classic_consumer_trace_and_owner_wiring"),
        }
        if keyword in _OUTPUT_ALIASES:
            applicability["alias_of"] = f"output.{canonical_slug}"
        category = "implementation_restriction"
        excluded = False
    if (raw.get("source_status") != status or
            not _same(raw.get("applicability"), applicability) or
            raw.get("fail_closed") != {
                "enabled": True, "category": category,
                "phase": "before_solver_allocation",
            }):
        _fail(f"{label} Output source disposition differs")
    lifecycle = raw.get("lifecycle")
    if not isinstance(lifecycle, Mapping):
        _fail(f"{label} Output lifecycle differs")
    expected_impl = "not_applicable" if excluded else "missing"
    expected_connection = "not_applicable" if excluded else "not_connected"
    for backend in BACKENDS:
        if (lifecycle["implementation"][backend].get("state") != expected_impl or
                lifecycle["accepted_state_connection"][backend].get("state") !=
                expected_connection):
            _fail(f"{label} Output lifecycle scope disposition differs")
    roles = {
        anchor.get("role") for anchor in raw.get("source_anchors", ())
        if isinstance(anchor, Mapping)
    }
    if status == "classic_flow_active" and not {
            "classic FLOW statistics consumer",
            "classic FLOW statistics writer"}.issubset(roles):
        _fail(f"{label} active Output lacks classic consumer/writer evidence")


def _expected_host_interface_schema(route: int) -> dict[str, object]:
    return {
        "type": "per_fraction_host_interface_payload",
        "cardinality": "one_effective_payload_per_fraction",
        "route": {
            "type": "integer_enum", "allowed": [route], "required": True},
        "fraction_type": {
            "type": "sediment_fraction_type_identity", "required": True,
            "route_semantics": (
                "mud_or_cohesive_fraction" if route == 21 else
                "noncohesive_or_generic_transport_fraction"),
        },
        "library": {
            "type": "path", "required": True, "canonical_keyword": "DLL",
            "accepted_aliases": ["TranspLib"],
        },
        "function": {
            "type": "exported_symbol", "required": True,
            "canonical_keyword": "function",
            "accepted_aliases": ["TranspFunction"],
        },
        "input_file": {
            "type": "optional_path", "required": False,
            "canonical_keyword": "InputFile",
            "accepted_aliases": ["TranspInput"],
        },
        "fraction_block_origin": {
            "type": "enum", "required": True,
            "allowed": ["inherited_default", "explicit_fraction_block"],
        },
        "inheritance": (
            "if a fraction has no explicit block, copy the complete ifrac=0 "
            "default payload without losing its inherited origin"),
        "execution_disposition": "catalogued_identity_only_fail_closed",
    }


def _validate_overlay(overlay: Mapping[str, object]) -> None:
    if set(overlay) != _TOP_LEVEL_KEYS:
        _fail("Q0b scope-overlay top-level inventory differs")
    if (overlay.get("artifact_schema") != "braids.flow2dh_scope_overlay" or
            overlay.get("artifact_schema_version") != SCOPE_OVERLAY_SCHEMA or
            overlay.get("revision_id") != SCOPE_OVERLAY_REVISION or
            overlay.get("status") != "applied_zero_step"):
        _fail("Q0b scope-overlay identity differs")

    base = overlay.get("base_live_catalogue")
    if not isinstance(base, Mapping) or set(base) != _BASE_KEYS:
        _fail("Q0b base-live-catalogue binding differs")
    if _require_relative_path(base.get("path"), "Q0b base") != \
            "braids_core/sediment/flow2dh_live_catalogue_v1.json":
        _fail("Q0b base-live-catalogue path differs")
    expected_parent_hash = _require_digest(base.get("sha256"), "Q0b base hash")
    parent_bytes = _installed_parent_bytes()
    if hashlib.sha256(parent_bytes).hexdigest() != expected_parent_hash:
        _fail("Installed Q0/Q0a parent bytes differ")
    parent = _installed_parent()
    if (base.get("revision_id") != parent.get("revision_id") or
            base.get("live_physical_semantic_id") != parent["identity"].get(
                "live_physical_semantic_id") or
            base.get("catalogue_erratum_registry_id") != parent["identity"].get(
                "catalogue_erratum_registry_id") or
            base.get("live_catalogue_manifest_id") != parent["identity"].get(
                "live_catalogue_manifest_id")):
        _fail("Q0b base-live-catalogue identities differ")

    scope = overlay.get("scope_contract")
    if not isinstance(scope, Mapping) or not _same(scope, _EXPECTED_SCOPE_CONTRACT):
        _fail("Q0b scope contract differs")

    selector_ledger = _installed_selector_ledger(parent)
    physical = {row.get("entry_id"): row for row in selector_ledger.get("entries", [])
                if isinstance(row, Mapping)}
    lifecycle = {row.get("entry_id"): row for row in parent.get("entry_lifecycle", [])
                 if isinstance(row, Mapping)}
    corrections = overlay.get("corrections")
    if not isinstance(corrections, list) or not corrections:
        _fail("Q0b correction registry differs")
    correction_ids: list[str] = []
    corrected_entries: list[str] = []
    physical_method_delta = 0
    for index, raw in enumerate(corrections):
        label = f"Q0b correction {index}"
        if not isinstance(raw, Mapping) or set(raw) != _CORRECTION_KEYS:
            _fail(f"{label} fields differ")
        correction_id = _require_nonempty_string(
            raw.get("correction_id"), label)
        correction_ids.append(correction_id)
        entry_id = _require_nonempty_string(raw.get("entry_id"), label)
        corrected_entries.append(entry_id)
        if entry_id not in physical or entry_id not in lifecycle:
            _fail(f"{label} references an unknown parent entry")
        if _hash(physical[entry_id]) != _require_digest(
                raw.get("parent_row_sha256"), f"{label} parent-row hash"):
            _fail(f"{label} parent-row hash differs")
        physical_forbidden_roots = {
            "entry_id", "selector", "exact_profile_refs",
            "product_profile_refs", "implementation",
            "accepted_state_connection", "runtime_method_id",
            "schema_v1_source_id",
        }
        classic_avalanche_selector_repair = (
            correction_id ==
            "q0b.redistribution.avalanche_classic_flux_semantics" and
            entry_id == "redistribution.avalanche"
        )
        if classic_avalanche_selector_repair:
            physical_forbidden_roots.remove("selector")
        patches = _validate_patches(
            raw.get("patches"), label,
            forbidden_roots=physical_forbidden_roots,
        )
        selector_patches = [
            patch for patch in patches
            if _pointer_parts(patch["path"], label)[0] == "selector"
        ]
        if classic_avalanche_selector_repair:
            if selector_patches != [{
                    "path": "/selector",
                    "expected": {
                        "section": "redistribution",
                        "keyword": "WetSlope/DrySlope/DuneAvalan/AvalTime",
                        "value": "WetSlope/DrySlope/DuneAvalan/AvalTime",
                    },
                    "effective": {
                        "section": "Morphology",
                        "keyword": "WetSlope/AvalTime",
                        "value": "current_bedload_flux_avalanche",
                    },
            }]:
                _fail(f"{label} classic avalanche selector repair differs")
        elif selector_patches:
            _fail(f"{label} cannot patch selector identity")
        lifecycle_patches = _validate_patches(
            raw.get("lifecycle_patches"), label,
            forbidden_roots={
                "entry_id", "exact_profile_refs", "product_profile_refs",
            },
        )
        if not patches and not lifecycle_patches:
            _fail(f"{label} has no correction")
        effective_physical = _apply_patches(physical[entry_id], patches, label)
        if not isinstance(effective_physical, Mapping):
            _fail(f"{label} physical projection differs")
        if entry_id in _HOST_INTERFACE_IDS:
            route = 21 if entry_id == "capacity.external_subroutine_21" else 15
            if not _same(effective_physical.get("value_schema"),
                         _expected_host_interface_schema(route)):
                _fail(f"{label} per-fraction host identity schema differs")
        if entry_id == "parameter.fine_sediment.bsskin" and (
                effective_physical.get("subkind") != "physical_selector_alias" or
                effective_physical.get("unresolved_semantics") != [] or
                not _same(effective_physical.get("value_schema"),
                          _EXPECTED_BSSKIN_VALUE_SCHEMA)):
            _fail(f"{label} BsSkin logical-selector correction differs")
        if entry_id == "redistribution.avalanche" and _hash(
                effective_physical.get("value_schema")) != \
                _EXPECTED_AVALANCHE_SCOUR_SCHEMA_IDS[entry_id]:
            _fail(f"{label} classic avalanche physical schema differs")
        physical_method_delta += int(
            effective_physical.get("counts_as_method") is True) - int(
                physical[entry_id].get("counts_as_method") is True)
        effective_lifecycle = _apply_patches(
            lifecycle[entry_id], lifecycle_patches, f"{label} lifecycle")
        owner = _validate_owner(raw.get("owner"), label)
        if not isinstance(effective_lifecycle, Mapping) or \
                effective_lifecycle.get("owner_edges") != [owner]:
            _fail(f"{label} owner does not match effective lifecycle")
        if (effective_lifecycle.get("entry_id") != entry_id or
                effective_lifecycle.get("kind") != effective_physical.get("kind") or
                effective_lifecycle.get("family") !=
                effective_physical.get("family")):
            _fail(f"{label} physical/lifecycle projections differ")
        _validate_correction_lifecycle_downgrade(
            lifecycle[entry_id], effective_lifecycle, label)
        if not isinstance(raw.get("applicability"), (str, Mapping)):
            _fail(f"{label} applicability differs")
        if not _validate_fail_closed(raw.get("fail_closed"), label)["enabled"]:
            _fail(f"{label} may not disable its fail-closed disposition")
        _validate_source_anchors(raw.get("source_anchors"), label)
    if (len(correction_ids) != len(set(correction_ids)) or
            len(corrected_entries) != len(set(corrected_entries)) or
            correction_ids != sorted(correction_ids)):
        _fail("Q0b correction identities are duplicate or non-canonical")
    if set(corrected_entries) != _CORRECTION_ENTRY_IDS:
        _fail("Q0b correction target inventory differs")

    additions = overlay.get("added_entries")
    if not isinstance(additions, list) or not additions:
        _fail("Q0b added-entry registry differs")
    known_ids = set(lifecycle)
    added_ids: list[str] = []
    for index, raw in enumerate(additions):
        label = f"Q0b added entry {index}"
        if not isinstance(raw, Mapping) or not _ADDED_REQUIRED_KEYS.issubset(raw) or \
                not set(raw).issubset(_ADDED_REQUIRED_KEYS | _ADDED_OPTIONAL_KEYS):
            _fail(f"{label} fields differ")
        entry_id = _require_nonempty_string(raw.get("entry_id"), label)
        if entry_id in known_ids:
            _fail(f"{label} shadows an accepted parent entry")
        known_ids.add(entry_id)
        added_ids.append(entry_id)
        is_added_method = entry_id in _ADDED_KIND_METHOD_IDS
        counts_as_method = entry_id in _ADDED_COUNTED_METHOD_IDS
        if (raw.get("namespace") != "delft3d4_flow_2dh_q0b" or
                raw.get("kind") not in {
                    "method", "payload", "interface", "output_selector"} or
                type(raw.get("counts_as_method")) is not bool or
                (raw.get("kind") == "method") != is_added_method or
                raw.get("counts_as_method") is not counts_as_method):
            _fail(f"{label} classification differs")
        optional_fields = set(raw) - _ADDED_REQUIRED_KEYS
        if is_added_method:
            if optional_fields != _ADDED_METHOD_KEYS or (
                    raw.get("runtime_method_id") != entry_id or
                    not isinstance(raw.get("backend_status"), Mapping) or
                    set(raw["backend_status"]) != set(BACKENDS) or
                    not isinstance(raw.get("compatibility"), Mapping) or
                    not isinstance(raw.get("consumer_order"), list) or
                    not isinstance(raw.get("scope"), Mapping) or
                    not isinstance(raw.get("unresolved_semantics"), list)):
                _fail(f"{label} method projection differs")
            for field in ("requiredness", "schema_v1_source_id", "subkind"):
                _require_nonempty_string(raw.get(field), f"{label} {field}")
        elif raw.get("kind") == "output_selector":
            if optional_fields != _OUTPUT_WIRING_KEYS:
                _fail(f"{label} output projection differs")
        elif optional_fields:
            _fail(f"{label} non-method projection differs")
        for field in ("family", "label", "source_status"):
            _require_nonempty_string(raw.get(field), f"{label} {field}")
        if not isinstance(raw.get("selector"), Mapping) or \
                not isinstance(raw.get("value_schema"), Mapping):
            _fail(f"{label} selector/value schema differs")
        _require_nonempty_string(
            raw["selector"].get("keyword"), f"{label} selector keyword")
        owner = _validate_owner(raw.get("owner"), label)
        if not _validate_fail_closed(raw.get("fail_closed"), label)["enabled"]:
            _fail(f"{label} may not disable its fail-closed disposition")
        _validate_source_anchors(raw.get("source_anchors"), label)
        lifecycle_row = _validate_lifecycle(raw.get("lifecycle"), entry_id, label)
        if lifecycle_row["owner_edges"] != [owner] or \
                lifecycle_row["kind"] != raw.get("kind") or \
                lifecycle_row["family"] != raw.get("family"):
            _fail(f"{label} physical/lifecycle projections differ")
        if not isinstance(raw.get("applicability"), (str, Mapping)):
            _fail(f"{label} applicability differs")
        present_wiring = set(raw) & _OUTPUT_WIRING_KEYS
        if raw.get("kind") != "output_selector" and present_wiring:
            _fail(f"{label} output wiring is attached to a non-output row")
        if raw.get("kind") == "output_selector":
            if present_wiring != _OUTPUT_WIRING_KEYS:
                _fail(f"{label} output lacks a complete wiring disposition")
            for field in ("result_field_id", "checkpoint_field_id"):
                if raw[field] is not None and (
                        not isinstance(raw[field], str) or not raw[field].strip()):
                    _fail(f"{label} {field} differs")
            if raw["wiring_state"] != "pending_owner":
                _fail(f"{label} output wiring state differs")
            _validate_output_entry(raw, label)
    if len(added_ids) != len(set(added_ids)):
        _fail("Q0b added-entry identities are duplicate")
    non_output_ids = {
        row["entry_id"] for row in additions
        if row.get("kind") != "output_selector"
    }
    if non_output_ids != _NON_OUTPUT_ADDITION_IDS:
        _fail("Q0b non-output addition inventory differs")
    added_by_id = {row["entry_id"]: row for row in additions}
    geometry = added_by_id["payload.point_discharge.geometry"]
    timeseries = added_by_id["payload.point_discharge.timeseries"]
    if (geometry["selector"] != {
            "section": "discharge", "keyword": "Filsrc",
            "value": "source_geometry_path"} or
            not _same(geometry["value_schema"], _EXPECTED_FILSRC_SCHEMA)):
        _fail("Q0b Filsrc geometry/source-identity schema differs")
    if (timeseries["selector"] != {
            "section": "discharge", "keyword": "Fildis",
            "value": "time_series_path"} or
            not _same(timeseries["value_schema"], _EXPECTED_FILDIS_SCHEMA)):
        _fail("Q0b Fildis interpolation/cursor schema differs")
    for entry_id, schema_id in _EXPECTED_DETAILED_PAYLOAD_SCHEMA_IDS.items():
        if _hash(added_by_id[entry_id].get("value_schema")) != schema_id:
            _fail(f"Q0b detailed payload schema differs: {entry_id}")
    for entry_id, schema_id in _EXPECTED_AVALANCHE_SCOUR_SCHEMA_IDS.items():
        if entry_id == "redistribution.avalanche":
            continue
        if _hash(added_by_id[entry_id].get("value_schema")) != schema_id:
            _fail(f"Q0b avalanche/scour schema differs: {entry_id}")

    rules = overlay.get("compatibility_rules")
    if not isinstance(rules, list):
        _fail("Q0b compatibility-rule registry differs")
    rule_ids = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            _fail(f"Q0b compatibility rule {index} differs")
        rule_ids.append(_require_nonempty_string(
            rule.get("rule_id"), f"Q0b compatibility rule {index}"))
        if "source_anchors" in rule:
            _validate_source_anchors(
                rule["source_anchors"], f"Q0b compatibility rule {index}")
    if len(rule_ids) != len(set(rule_ids)) or rule_ids != sorted(rule_ids):
        _fail("Q0b compatibility-rule identities are duplicate or non-canonical")
    q6b_rule = next((
        rule for rule in rules
        if rule.get("rule_id") ==
        "q0b.bed_boundary_modes_6_7.consumer_unresolved"
    ), None)
    if not isinstance(q6b_rule, Mapping) or q6b_rule.get(
            "raw_parameter_guard") != {
                "keyword": "IBedCond",
                "blocked_integer_values": [6, 7],
                "failure_category": "implementation_restriction",
                "failure_phase": "before_solver_allocation",
            }:
        _fail("Q0b IBedCond 6/7 compatibility guard differs")
    declared_rule = next((
        rule for rule in rules
        if rule.get("rule_id") == "q0b.declared_scope_exclusions.fail_closed"
    ), None)
    expected_route_22_guards = [
        {
            "keyword": keyword, "blocked_integer_values": [22],
            "failure_category": "scope_incompatibility",
            "failure_phase": "before_solver_allocation",
        }
        for keyword in _TRANSPORT_SELECTOR_ALIASES
    ]
    if (not isinstance(declared_rule, Mapping) or
            declared_rule.get("entry_ids") != ["capacity.asmita"] or
            declared_rule.get("raw_parameter_guards") !=
            expected_route_22_guards + [
                {
                    "keyword": "Multi", "blocked_values": [True],
                    "failure_category": "scope_incompatibility",
                    "failure_phase": "before_solver_allocation",
                },
            ]):
        _fail("Q0b declared-scope exclusion guard differs")
    typed_rule = next((
        rule for rule in rules
        if rule.get("rule_id") ==
        "q0b.raw_selector_values.require_typed_selection"
    ), None)
    expected_typed_guards = [
        {
            "keyword": "IBedCond",
            "blocked_values": "all",
            "failure_category": "invalid_selection",
            "failure_phase": "before_solver_allocation",
            "reason": "Use a canonical typed bed_boundary.mode_<n> entry ID.",
        },
        {
            "keyword": "ICmpCond", "blocked_values": "all",
            "failure_category": "invalid_selection",
            "failure_phase": "before_solver_allocation",
            "reason": (
                "Use a canonical typed composition_boundary.mode_<n> entry ID."),
        },
    ] + [
        {
            "keyword": keyword, "blocked_values": "all",
            "failure_category": "invalid_selection",
            "failure_phase": "before_solver_allocation",
            "reason": "Use a canonical typed capacity entry ID.",
        }
        for keyword in _TRANSPORT_SELECTOR_ALIASES
    ] + [
        {
            "keyword": keyword, "blocked_values": "all",
            "failure_category": "invalid_selection",
            "failure_phase": "before_solver_allocation",
            "reason": _RAW_TYPED_SELECTOR_REASON,
        }
        for keyword in _OTHER_RAW_TYPED_SELECTOR_KEYWORDS
    ]
    if not isinstance(typed_rule, Mapping) or typed_rule.get(
            "raw_parameter_guards") != expected_typed_guards:
        _fail("Q0b typed raw-selector guard differs")
    trigger_rule = next((
        rule for rule in rules
        if rule.get("rule_id") ==
        "q0b.value_bearing_method_triggers.require_typed_owner"
    ), None)
    if (not isinstance(trigger_rule, Mapping) or
            trigger_rule.get("owner") != {
                "milestone": "Q0b",
                "responsibility": "value-bearing method-trigger ownership",
            } or
            trigger_rule.get("action") !=
            "validate_value_then_reject_active_orphan_parameter" or
            trigger_rule.get("behavior") !=
            "inactive_values_pass; invalid_values_fail; "
            "active_values_require_required_method_id" or
            trigger_rule.get("raw_parameter_owner_guards") !=
            _EXPECTED_VALUE_TRIGGER_OWNER_GUARDS):
        _fail("Q0b value-bearing method-trigger owner guard differs")
    current_only_rule = next((
        rule for rule in rules
        if rule.get("rule_id") ==
        "q0b.current_only_output_and_forcing_scope.fail_closed"
    ), None)
    if (not isinstance(current_only_rule, Mapping) or
            set(current_only_rule.get("entry_ids", ())) !=
            _SCOPE_EXCLUDED_OUTPUT_IDS or
            set(current_only_rule.get("selector_keywords", ())) !=
            _SCOPE_EXCLUDED_OUTPUT_KEYWORDS or
            current_only_rule.get("failure_category") !=
            "scope_incompatibility" or
            current_only_rule.get("failure_phase") !=
            "before_solver_allocation" or
            current_only_rule.get("api_argument_guard") != {
                "argument": "wave_or_orbital_forcing",
                "blocked_values": [True],
                "failure_category": "scope_incompatibility",
                "failure_phase": "before_solver_allocation",
            }):
        _fail("Q0b current-only output/forcing guard differs")
    policy = _validate_request_policy(overlay.get("request_policy"))
    for item in policy["host_interface_entry_ids"] + \
            policy["strict_2dh_metadata_only_entry_ids"]:
        if item not in known_ids:
            _fail("Q0b request policy references an unknown entry")
    output_rows = [
        row for row in additions if row.get("kind") == "output_selector"]
    output_policy = set(policy["output_keywords"])
    output_ids = {row["entry_id"] for row in output_rows}
    output_keywords = {row["selector"].get("keyword") for row in output_rows}
    if output_ids | output_keywords != output_policy:
        _fail("Q0b output request-policy inventory differs")
    non_output_rows = [
        row for row in additions if row.get("kind") != "output_selector"]
    non_output_keywords = {
        row["selector"].get("keyword") for row in non_output_rows}
    if _NON_OUTPUT_ADDITION_IDS | non_output_keywords != set(
            policy["not_yet_admitted_keywords"]):
        _fail("Q0b non-output request-policy inventory differs")
    if (set(policy["host_interface_entry_ids"]) != _HOST_INTERFACE_IDS or
            set(policy["host_route_tags"]) != _HOST_ROUTE_TAGS or
            set(policy["host_activation_keywords"]) != _HOST_ALIASES or
            set(policy["fm_only_keywords"]) !=
            _BERM_IDS | _BERM_KEYWORDS | _SCOPE_EXCLUDED_OUTPUT_IDS |
            _SCOPE_EXCLUDED_OUTPUT_KEYWORDS or
            set(policy["managed_operation_keywords"]) !=
            _MANAGED_IDS | _MANAGED_KEYWORDS or
            set(policy["strict_2dh_metadata_only_entry_ids"]) !=
            _STRICT_2DH_METADATA_IDS):
        _fail("Q0b mandatory request-policy inventory differs")
    expected_known: list[object] = []
    for row in selector_ledger.get("entries", []):
        if not isinstance(row, Mapping) or row.get("kind") == "method":
            continue
        selector = row.get("selector")
        if isinstance(selector, Mapping):
            expected_known.append(selector.get("keyword"))
    for collection in (
        parent["physical_revision"]["extension_namespace"]["entries"],
        parent["catalogue_erratum"]["host_extension_dispositions"],
        additions,
    ):
        for row in collection:
            selector = row.get("selector") if isinstance(row, Mapping) else None
            if isinstance(selector, Mapping):
                expected_known.append(selector.get("keyword"))
    expected_known.extend((
        "TraFrm", "TranspFrm", "iform", "transport_formula", "DLL",
        "TranspLib", "function", "TranspFunction", "InputFile",
        "TranspInput", "IniComp", "BcFil", "Filsrc", "Fildis", "Fildad",
        "IBedCond", "ICmpCond", "SettleLib", "SettleFunction",
        "SettleInput", "ReturnTime",
        "parameter.bedload_adaptation.length_diameters",
    ))
    for name in _REQUEST_POLICY_LISTS - {
            "host_route_tags", "known_parameter_keywords"}:
        expected_known.extend(policy[name])
    expected_known.extend(_OTHER_RAW_TYPED_SELECTOR_KEYWORDS)
    if policy["known_parameter_keywords"] != _unique_casefold(expected_known):
        _fail("Q0b known-parameter inventory differs")

    stages = overlay.get("stage_manifests")
    if not isinstance(stages, list) or not stages:
        _fail("Q0b stage-manifest registry differs")
    stage_ids: list[str] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, Mapping) or set(stage) != _STAGE_SUMMARY_KEYS:
            _fail(f"Q0b stage manifest {index} differs")
        stage_ids.append(_require_nonempty_string(
            stage.get("stage_id"), f"Q0b stage manifest {index}"))
        for key in ("manifest_id", "sha256"):
            if key in stage:
                _require_digest(stage[key], f"Q0b stage manifest {index} {key}")
        if "path" in stage:
            _require_relative_path(stage["path"], f"Q0b stage manifest {index}")
        expected_stage_path = (
            "validation/sediment/delft3d_current/q0b_catalogue_scope_repair/"
            f"stage_manifests/{stage_ids[-1]}.json"
        )
        if stage.get("path") != expected_stage_path:
            _fail(f"Q0b stage manifest {index} path differs")
        expected_stage_identity = _EXPECTED_STAGE_SUMMARY_IDS.get(stage_ids[-1])
        if expected_stage_identity is None or (
                stage.get("manifest_id"), stage.get("sha256")) != \
                expected_stage_identity:
            _fail(f"Q0b stage manifest {index} installed identity differs")
    if len(stage_ids) != len(set(stage_ids)):
        _fail("Q0b stage-manifest identities are duplicate")
    if tuple(stage_ids) != _STAGE_ORDER:
        _fail("Q0b stage-manifest scope differs")

    counts = overlay.get("counts")
    if not isinstance(counts, Mapping) or set(counts) != _COUNT_KEYS or \
            not isinstance(overlay.get("claims"), Mapping):
        _fail("Q0b counts/claims differ")
    expected_counts = {
        "corrections": len(corrections),
        "added_entries": len(additions),
        "output_selectors": sum(
            row.get("kind") == "output_selector" for row in additions),
        "non_output_additions": sum(
            row.get("kind") != "output_selector" for row in additions),
        "effective_lifecycle_rows": len(lifecycle) + len(additions),
        "effective_physical_methods":
            parent["counts"]["effective_strict_delft_methods"] +
            physical_method_delta + sum(
                row.get("counts_as_method") is True for row in additions),
        "stage_manifests": len(stages),
    }
    if dict(counts) != expected_counts or expected_counts != {
            "corrections": 19,
            "added_entries": 85,
            "output_selectors": 69,
            "non_output_additions": 16,
            "effective_lifecycle_rows": 631,
            "effective_physical_methods": 149,
            "stage_manifests": 13,
    }:
        _fail("Q0b count projection differs")
    claims = overlay["claims"]
    if not _same(claims, _EXPECTED_CLAIMS):
        _fail("Q0b zero-step/read-only claim differs")

    identity = overlay.get("identity")
    if not isinstance(identity, Mapping) or set(identity) != _IDENTITY_KEYS:
        _fail("Q0b identity inventory differs")
    expected = scope_identity_payloads(overlay)
    if dict(identity) != expected or dict(identity) != _EXPECTED_IDENTITY:
        _fail("Q0b identity projection differs")
    for key, value in identity.items():
        if key != "effective_scope_id":
            _require_digest(value, f"Q0b {key}")


def load_scope_overlay(
    path: str | Path = SCOPE_OVERLAY_FILE,
) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise LiveCatalogueError(
            f"Installed Q0b scope overlay is missing: {target}",
            category="missing_catalogue",
        )
    document = _read_json(target.read_bytes())
    _validate_overlay(document)
    return document


def compose_effective_catalogue(
    parent: Mapping[str, object], overlay: Mapping[str, object],
) -> dict[str, Any]:
    """Purely compose an authenticated Q0b view over an accepted Q0/Q0a parent."""
    if not isinstance(parent, Mapping) or not isinstance(overlay, Mapping):
        _fail("Q0b composition inputs must be objects")
    _validate_overlay(overlay)
    installed_parent = _installed_parent()
    if not _same(parent, installed_parent):
        _fail("Q0b composition parent differs from the installed accepted parent")
    parent_copy = copy.deepcopy(dict(parent))
    selector = _installed_selector_ledger(parent_copy)

    physical_rows: dict[str, dict[str, Any]] = {
        row["entry_id"]: copy.deepcopy(row)
        for row in selector["entries"]
    }
    # The accepted Q0 extension and Q0a host-interface descriptions are part of
    # the effective physical namespace even though they are not selector-ledger
    # parent rows.
    for row in parent_copy["physical_revision"]["extension_namespace"]["entries"]:
        physical_rows[row["entry_id"]] = copy.deepcopy(row)
    for row in parent_copy["catalogue_erratum"]["host_extension_dispositions"]:
        physical_rows[row["entry_id"]] = copy.deepcopy(row)
    lifecycle_rows = {
        row["entry_id"]: copy.deepcopy(row)
        for row in parent_copy["entry_lifecycle"]
    }

    for correction in overlay["corrections"]:
        entry_id = correction["entry_id"]
        physical_rows[entry_id] = _apply_patches(
            physical_rows[entry_id], correction["patches"],
            f"Q0b correction {correction['correction_id']}",
        )
        lifecycle_rows[entry_id] = _apply_patches(
            lifecycle_rows[entry_id], correction["lifecycle_patches"],
            f"Q0b lifecycle correction {correction['correction_id']}",
        )

    for added in overlay["added_entries"]:
        entry_id = added["entry_id"]
        physical_rows[entry_id] = {
            key: copy.deepcopy(value)
            for key, value in added.items()
            if key != "lifecycle"
        }
        lifecycle_rows[entry_id] = copy.deepcopy(added["lifecycle"])

    result = parent_copy
    result["artifact_schema"] = "braids.flow2dh_live_catalogue_effective"
    result["artifact_schema_version"] = LIVE_CATALOGUE_SCHEMA
    result["revision_id"] = LIVE_CATALOGUE_REVISION
    result["parent_live_catalogue_identity"] = copy.deepcopy(parent["identity"])
    result["catalogue_scope_repair"] = copy.deepcopy(dict(overlay))
    result["scope_contract"] = copy.deepcopy(overlay["scope_contract"])
    result["effective_entries"] = [physical_rows[key] for key in sorted(physical_rows)]
    result["entry_lifecycle"] = [lifecycle_rows[key] for key in sorted(lifecycle_rows)]
    result["parent_counts"] = copy.deepcopy(parent["counts"])
    result["counts"] = copy.deepcopy(overlay["counts"])
    result["identity"] = {
        "parent_live_catalogue": copy.deepcopy(parent["identity"]),
        **copy.deepcopy(overlay["identity"]),
    }
    # Q0b is inventory/scope repair only. Admission cannot change here.
    if (not _same(result["exact_profile_admissions"],
                  parent["exact_profile_admissions"]) or
            not _same(result["product_profile_admissions"],
                      parent["product_profile_admissions"])):
        _fail("Q0b composition changed inherited admission")
    return result


def load_live_catalogue(
    path: str | Path = SCOPE_OVERLAY_FILE,
) -> dict[str, Any]:
    return compose_effective_catalogue(_installed_parent(), load_scope_overlay(path))


def _catalogue(document: Mapping[str, object] | None) -> dict[str, Any]:
    if document is None:
        return load_live_catalogue()
    if not isinstance(document, Mapping):
        _fail("Injected Q0b catalogue must be an object")
    overlay = document.get("catalogue_scope_repair")
    if not isinstance(overlay, Mapping):
        _fail("Injected Q0b catalogue lacks its authenticated overlay")
    expected = compose_effective_catalogue(_installed_parent(), overlay)
    if not _same(document, expected):
        _fail("Injected Q0b effective catalogue differs from deterministic composition")
    return copy.deepcopy(expected)


def lifecycle_entry(
    entry_id: str, *, document: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    catalogue = _catalogue(document)
    for row in catalogue["entry_lifecycle"]:
        if row.get("entry_id") == entry_id:
            return copy.deepcopy(row)
    raise LiveCatalogueError(
        f"Unknown live-catalogue entry: {entry_id}", category="unknown_entry")


def effective_entry(
    entry_id: str, *, document: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    catalogue = _catalogue(document)
    for row in catalogue["effective_entries"]:
        if row.get("entry_id") == entry_id:
            return copy.deepcopy(row)
    raise LiveCatalogueError(
        f"Unknown live-catalogue entry: {entry_id}", category="unknown_entry")


def effective_requiredness(
    entry_id: str, *, document: Mapping[str, object] | None = None,
    parent_requiredness: str | None = None,
) -> str | None:
    row = effective_entry(entry_id, document=document)
    value = row.get("requiredness")
    return str(value) if value is not None else parent_requiredness


def effective_compatibility_rules(
    *, document: Mapping[str, object] | None = None,
) -> tuple[dict[str, Any], ...]:
    catalogue = _catalogue(document)
    parent_rules = list(_v1.effective_compatibility_rules())
    additions = copy.deepcopy(
        catalogue["catalogue_scope_repair"]["compatibility_rules"])
    rows = parent_rules + additions
    identifiers = [row.get("rule_id") for row in rows]
    if len(identifiers) != len(set(identifiers)):
        _fail("Effective Q0b compatibility-rule IDs are not unique")
    return tuple(rows)


def _iterable_strings(value: Iterable[str], label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise LiveCatalogueError(
            f"{label} must be an iterable, not a string",
            category="invalid_selection")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise LiveCatalogueError(
            f"{label} must be iterable", category="invalid_selection") from exc
    if not all(isinstance(item, str) and item for item in result):
        raise LiveCatalogueError(
            f"{label} must contain nonempty strings", category="invalid_selection")
    if len(result) != len(set(result)):
        raise LiveCatalogueError(
            f"{label} must be unique", category="invalid_selection")
    return result


def _active(value: object) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value)) and float(value) != 0.0
    return True


def _policy_failure(policy: Mapping[str, object], name: str,
                    details: Iterable[object]) -> None:
    categories = policy["failure_categories"]
    phases = policy["failure_phases"]
    category = categories.get(name, "scope_incompatibility")
    phase = phases.get(name, "before_solver_allocation")
    rendered = ", ".join(sorted(str(item) for item in details))
    raise LiveCatalogueError(
        f"Q0b request is rejected at {phase}: {rendered}", category=category)


def _compatibility_failure(
    overlay: Mapping[str, object], rule_id: str, details: Iterable[object],
) -> None:
    rule = next((
        item for item in overlay["compatibility_rules"]
        if item.get("rule_id") == rule_id
    ), None)
    if not isinstance(rule, Mapping):
        _fail(f"Required Q0b compatibility rule is missing: {rule_id}")
    category = _require_nonempty_string(
        rule.get("failure_category"), f"{rule_id} failure category")
    phase = _require_nonempty_string(
        rule.get("failure_phase"), f"{rule_id} failure phase")
    rendered = ", ".join(sorted(str(item) for item in details))
    raise LiveCatalogueError(
        f"Q0b request is rejected at {phase}: {rendered}", category=category)


def _integer_selector(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    return None


def _numeric_selector(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _boolean_selector(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and \
            math.isfinite(float(value)):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
    return None


def _value_trigger_state(keyword: str, value: object) -> str:
    """Classify the two source value-bearing method triggers."""
    folded = keyword.casefold()
    if value is None:
        return "inactive"
    if isinstance(value, bool):
        return "invalid"
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return "invalid"
        if folded == "pmcrit":
            return "active" if float(value) >= 0.0 else "inactive"
        return "active" if float(value) != 0.0 else "inactive"
    if isinstance(value, str):
        if folded == "pmcrit":
            return "active" if value.strip() else "invalid"
        return "invalid" if value.strip() else "inactive"
    return "invalid"


def _invalid_raw_parameter(key: object, value: object, reason: str) -> None:
    raise LiveCatalogueError(
        "Q0b request is rejected at before_solver_allocation: "
        f"{key}={value}. {reason}",
        category="invalid_parameter",
    )


def _finite_raw_number(key: object, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
            not math.isfinite(float(value)):
        _invalid_raw_parameter(key, value, "A finite number is required.")
    return float(value)


def _raw_guard_failure(
    overlay: Mapping[str, object], rule_id: str, keyword: str,
    details: Iterable[object],
) -> None:
    rule = next((
        item for item in overlay["compatibility_rules"]
        if item.get("rule_id") == rule_id
    ), None)
    if not isinstance(rule, Mapping):
        _fail(f"Required Q0b raw-selector rule is missing: {rule_id}")
    guards = rule.get("raw_parameter_guards")
    if not isinstance(guards, list):
        _fail(f"Required Q0b raw-selector guards are missing: {rule_id}")
    guard = next((item for item in guards
                  if item.get("keyword") == keyword), None)
    if not isinstance(guard, Mapping):
        _fail(f"Required Q0b raw-selector guard is missing: {keyword}")
    category = _require_nonempty_string(
        guard.get("failure_category"), f"{rule_id}/{keyword} category")
    phase = _require_nonempty_string(
        guard.get("failure_phase"), f"{rule_id}/{keyword} phase")
    reason = guard.get("reason")
    rendered = ", ".join(sorted(str(item) for item in details))
    suffix = f" {reason}" if isinstance(reason, str) and reason else ""
    raise LiveCatalogueError(
        f"Q0b request is rejected at {phase}: {rendered}.{suffix}",
        category=category,
    )


def _preflight_scope_requests(
    selected_ids: Iterable[str], parameter_values: Mapping[str, object],
    overlay: Mapping[str, object], *, wave_or_orbital_forcing: bool = False,
) -> None:
    policy = overlay["request_policy"]
    selected = set(selected_ids)
    output_ids = {
        row["entry_id"] for row in overlay["added_entries"]
        if row.get("kind") == "output_selector"
    }
    raw = dict(parameter_values)
    lower = {str(key).casefold(): (key, value) for key, value in raw.items()}
    if len(lower) != len(raw):
        raise LiveCatalogueError(
            "Parameter names must be unique under case-insensitive matching",
            category="invalid_parameter",
        )

    if type(wave_or_orbital_forcing) is not bool:
        raise LiveCatalogueError(
            "wave_or_orbital_forcing must be boolean",
            category="invalid_parameter",
        )
    if wave_or_orbital_forcing:
        _compatibility_failure(
            overlay, "q0b.current_only_output_and_forcing_scope.fail_closed",
            ["wave_or_orbital_forcing=True"],
        )

    host_ids = selected & set(policy["host_interface_entry_ids"])
    host_keywords = []
    for keyword in policy["host_activation_keywords"]:
        pair = lower.get(str(keyword).casefold())
        if pair is None:
            continue
        value = pair[1]
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if not isinstance(value, str):
            raise LiveCatalogueError(
                f"Q0b host path/function parameter {pair[0]} must be a string",
                category="invalid_parameter",
            )
        host_keywords.append(pair[0])
    route_tags = {int(value) for value in policy["host_route_tags"]
                  if isinstance(value, int) and not isinstance(value, bool)}
    route_requests: list[object] = []
    transport_selectors: list[tuple[str, object, int | None]] = []
    for key in _TRANSPORT_SELECTOR_ALIASES:
        pair = lower.get(key.casefold())
        if pair is None:
            continue
        route = _integer_selector(pair[1])
        transport_selectors.append((str(pair[0]), pair[1], route))
        if route in route_tags:
            route_requests.append(f"{pair[0]}={route}")
    if host_ids or host_keywords or route_requests:
        _policy_failure(policy, "host_interface",
                        list(host_ids) + host_keywords + route_requests)

    multi = lower.get("multi")
    multi_state = _boolean_selector(multi[1]) if multi is not None else None
    if multi is not None and multi_state is None:
        raise LiveCatalogueError(
            "Q0b Multi must be an explicit boolean before_solver_allocation",
            category="invalid_parameter",
        )
    declared_scope_requests: list[object] = list(
        selected & {"capacity.asmita"})
    excluded_transport = [
        (key, value) for key, value, route in transport_selectors
        if route == 22
    ]
    declared_scope_requests.extend(
        f"{key}=22" for key, _value in excluded_transport)
    if multi is not None and multi_state is True:
        declared_scope_requests.append(f"{multi[0]}={multi[1]}")
    if declared_scope_requests:
        # Both raw guards have the same pinned category and phase.
        keyword = "TraFrm" if excluded_transport else "Multi"
        if selected & {"capacity.asmita"} and not (
                excluded_transport or multi_state is True):
            raise LiveCatalogueError(
                "Q0b request is rejected at before_solver_allocation: "
                "capacity.asmita is outside natural unmanaged river scope",
                category="scope_incompatibility",
            )
        _raw_guard_failure(
            overlay, "q0b.declared_scope_exclusions.fail_closed",
            keyword, declared_scope_requests)

    # D-Flow FM's dune-slumping controls share the morphology parser but do
    # not have a classic FLOW consumer.  Exact source defaults are inert in
    # the current-only scope; activation is scope-incompatible.
    fm_dune_requests: list[object] = list(
        selected & ({_FM_DUNE_AVALANCHE_ID} | _FM_DUNE_PARAMETER_IDS))
    dune = lower.get("duneavalan")
    if dune is not None:
        value = dune[1]
        if value is None or value is False:
            pass
        elif value is True:
            fm_dune_requests.append(f"{dune[0]}=True")
        else:
            _invalid_raw_parameter(
                dune[0], value, "DuneAvalan must be boolean or null.")
    for raw_keyword, default in _FM_DUNE_PARAMETER_DEFAULTS.items():
        pair = lower.get(raw_keyword.casefold())
        if pair is None or pair[1] is None:
            continue
        parsed = _finite_raw_number(pair[0], pair[1])
        if parsed != default:
            fm_dune_requests.append(f"{pair[0]}={pair[1]}")
    if fm_dune_requests:
        raise LiveCatalogueError(
            "Q0b request is rejected at before_solver_allocation: " +
            ", ".join(sorted(str(item) for item in fm_dune_requests)) +
            ". DuneAvalan controls are D-Flow FM-only.",
            category="scope_incompatibility",
        )

    # Classic FLOW's wet-slope flux correction is distinct from FM dune
    # slumping.  Its non-default parameters must name a real consuming method.
    classic_parameter_specs = (
        ("AvalTime", 86400.0, {_CLASSIC_WET_SLOPE_ID}),
        ("WetSlope", 10.0,
         {_CLASSIC_WET_SLOPE_ID} | set(_WET_SLOPE_OTHER_CONSUMERS)),
    )
    for raw_keyword, default, owners in classic_parameter_specs:
        pair = lower.get(raw_keyword.casefold())
        if pair is None or pair[1] is None or (
                isinstance(pair[1], str) and not pair[1].strip()):
            continue
        parsed = _finite_raw_number(pair[0], pair[1])
        if parsed != default and not selected.intersection(owners):
            raise LiveCatalogueError(
                "Q0b request is rejected at before_solver_allocation: "
                f"{pair[0]}={pair[1]} has no selected classic FLOW "
                "consumer; select one of " + ", ".join(sorted(owners)),
                category="scope_incompatibility",
            )

    scour = lower.get("scour")
    if scour is not None:
        value = scour[1]
        if value is None or (isinstance(value, str) and not value.strip()):
            pass
        elif isinstance(value, str):
            raise LiveCatalogueError(
                "Q0b request is rejected at "
                "before_file_read_and_solver_allocation: "
                f"{scour[0]}={value}. The classic Scour payload/bundle is "
                "catalogued but not yet admitted.",
                category="implementation_restriction",
            )
        else:
            _invalid_raw_parameter(
                scour[0], value, "Scour must be a file path string or null.")

    # Classic dry-bank transfer is active only when ThetSD is positive (or a
    # nonblank spatial field).  HMaxTH modifies that owner but cannot activate
    # it; SedThr has other consumers and deliberately is not owner-bound here.
    thetsd = lower.get("thetsd")
    thetsd_active = False
    if thetsd is not None:
        value = thetsd[1]
        if value is None or (isinstance(value, str) and not value.strip()):
            pass
        elif isinstance(value, str):
            thetsd_active = True
        else:
            parsed = _finite_raw_number(thetsd[0], value)
            if parsed < 0.0:
                _invalid_raw_parameter(
                    thetsd[0], value, "ThetSD cannot be negative.")
            thetsd_active = parsed > 0.0
    hmaxth = lower.get("hmaxth")
    hmaxth_active = False
    if hmaxth is not None:
        value = hmaxth[1]
        if value is None or (isinstance(value, str) and not value.strip()):
            pass
        else:
            parsed = _finite_raw_number(hmaxth[0], value)
            if parsed < 0.0:
                _invalid_raw_parameter(
                    hmaxth[0], value, "HMaxTH cannot be negative.")
            hmaxth_active = parsed != 1.0
    if (thetsd_active or hmaxth_active) and _DRY_BANK_ID not in selected:
        names = []
        if thetsd_active and thetsd is not None:
            names.append(f"{thetsd[0]}={thetsd[1]}")
        if hmaxth_active and hmaxth is not None:
            names.append(f"{hmaxth[0]}={hmaxth[1]}")
        raise LiveCatalogueError(
            "Q0b request is rejected at before_solver_allocation: " +
            ", ".join(names) +
            f" requires canonical method {_DRY_BANK_ID}.",
            category="scope_incompatibility",
        )
    if _DRY_BANK_ID in selected and not thetsd_active:
        raise LiveCatalogueError(
            "Q0b request is rejected at before_solver_allocation: "
            "redistribution.dry_bank has no effect unless ThetSD is positive "
            "or a nonblank spatial field.",
            category="implementation_restriction",
        )

    current_only_outputs: list[object] = list(
        selected & _SCOPE_EXCLUDED_OUTPUT_IDS)
    for keyword in _SCOPE_EXCLUDED_OUTPUT_KEYWORDS:
        pair = lower.get(keyword.casefold())
        if pair is not None:
            current_only_outputs.append(pair[0])
    if current_only_outputs:
        _compatibility_failure(
            overlay, "q0b.current_only_output_and_forcing_scope.fail_closed",
            current_only_outputs,
        )

    fm_requests = list(selected & set(policy["fm_only_keywords"]))
    for keyword in policy["fm_only_keywords"]:
        pair = lower.get(str(keyword).casefold())
        if pair is not None:
            fm_requests.append(pair[0])
    if fm_requests:
        _policy_failure(policy, "fm_only", fm_requests)

    managed_requests = selected & set(policy["managed_operation_keywords"])
    for keyword in policy["managed_operation_keywords"]:
        pair = lower.get(str(keyword).casefold())
        if pair is None:
            continue
        if str(keyword).casefold() == "fildad":
            value = pair[1]
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            if not isinstance(value, str):
                raise LiveCatalogueError(
                    f"Q0b file parameter {pair[0]} must be a string",
                    category="invalid_parameter",
                )
            managed_requests.add(str(pair[0]))
        elif _active(pair[1]):
            managed_requests.add(str(pair[0]))
    if managed_requests:
        _policy_failure(policy, "managed_operation", managed_requests)

    metadata_activation = selected & set(
        policy["strict_2dh_metadata_only_entry_ids"])
    if metadata_activation:
        _policy_failure(
            policy, "strict_2dh_numerical_activation", metadata_activation)

    bed_boundary_requests: list[object] = list(
        selected & _UNRESOLVED_BED_BOUNDARY_IDS)
    ibedcond = lower.get("ibedcond")
    if ibedcond is not None and _integer_selector(ibedcond[1]) in {6, 7}:
        bed_boundary_requests.append(
            f"{ibedcond[0]}={_integer_selector(ibedcond[1])}")
    if bed_boundary_requests:
        _compatibility_failure(
            overlay, "q0b.bed_boundary_modes_6_7.consumer_unresolved",
            bed_boundary_requests)

    typed_rule_id = "q0b.raw_selector_values.require_typed_selection"
    if ibedcond is not None:
        _raw_guard_failure(
            overlay, typed_rule_id, "IBedCond",
            [f"{ibedcond[0]}={ibedcond[1]}"])
    icmpcond = lower.get("icmpcond")
    if icmpcond is not None:
        _raw_guard_failure(
            overlay, typed_rule_id, "ICmpCond",
            [f"{icmpcond[0]}={icmpcond[1]}"])
    if transport_selectors:
        _raw_guard_failure(
            overlay, typed_rule_id, "TraFrm",
            [f"{key}={value}" for key, value, _route in transport_selectors])
    for keyword in _OTHER_RAW_TYPED_SELECTOR_KEYWORDS:
        pair = lower.get(keyword.casefold())
        if pair is not None:
            _raw_guard_failure(
                overlay, typed_rule_id, keyword,
                [f"{pair[0]}={pair[1]}"])

    trigger_rule = next(
        rule for rule in overlay["compatibility_rules"]
        if rule.get("rule_id") ==
        "q0b.value_bearing_method_triggers.require_typed_owner"
    )
    for guard in trigger_rule["raw_parameter_owner_guards"]:
        supplied = []
        for keyword in guard["keywords"]:
            pair = lower.get(str(keyword).casefold())
            if pair is not None and pair not in supplied:
                supplied.append(pair)
        if len(supplied) > 1:
            raise LiveCatalogueError(
                "Q0b request is rejected at before_solver_allocation: "
                "semantic aliases may not be supplied together (" +
                ", ".join(str(pair[0]) for pair in supplied) + ")",
                category="invalid_parameter",
            )
        active = []
        for pair in supplied:
            state = _value_trigger_state(str(pair[0]), pair[1])
            if state == "invalid":
                trigger = guard["value_trigger"]
                raise LiveCatalogueError(
                    f"Q0b request is rejected at "
                    f"{trigger['invalid_failure_phase']}: "
                    f"{pair[0]}={pair[1]} is not a valid value-bearing "
                    "method trigger",
                    category=str(trigger["invalid_failure_category"]),
                )
            if state == "active":
                active.append(pair)
        if active and guard["required_method_id"] not in selected:
            rendered = ", ".join(
                f"{key}={value}" for key, value in active)
            raise LiveCatalogueError(
                f"Q0b request is rejected at {guard['failure_phase']}: "
                f"{rendered}. {guard['reason']}",
                category=str(guard["failure_category"]),
            )

    not_admitted = list(selected & set(policy["not_yet_admitted_keywords"]))
    # Output inventory is not output admission.  Until the owning stage wires
    # a canonical field, an explicit output request fails at the same early
    # implementation gate as every other catalogued-only entry.
    not_admitted.extend(selected & output_ids)
    for keyword in policy["not_yet_admitted_keywords"]:
        pair = lower.get(str(keyword).casefold())
        if pair is None:
            continue
        # The FM method row contributes DuneAvalan to this generated
        # inventory, but its source default False/null is explicitly inert.
        # Its dedicated typed guard above has already rejected True, invalid,
        # and conflicting nondefault bundle controls.
        if str(keyword) in _VALUE_SENSITIVE_NOT_ADMITTED_KEYWORDS:
            continue
        if str(keyword) in _OPTIONAL_FILE_PARAMETER_KEYWORDS:
            value = pair[1]
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            if not isinstance(value, str):
                raise LiveCatalogueError(
                    f"Q0b file parameter {pair[0]} must be a string",
                    category="invalid_parameter",
                )
        not_admitted.append(pair[0])
    for keyword in policy["output_keywords"]:
        pair = lower.get(str(keyword).casefold())
        if pair is not None:
            not_admitted.append(pair[0])
    if not_admitted:
        _policy_failure(policy, "not_yet_admitted", not_admitted)

    # The inherited Q0/Q0a resolver deliberately ignored unrelated mapping
    # keys.  Q0b is the installation preflight, so accepting a misspelling or
    # an uncatalogued input here would silently change physics.  The generated
    # policy therefore carries the complete case-insensitive key inventory:
    # inherited selector parameters plus Q0b metadata/payload/output aliases.
    known = {
        str(keyword).casefold()
        for keyword in policy["known_parameter_keywords"]
    }
    unknown = [
        key for key in raw
        if not isinstance(key, str) or not key or key.casefold() not in known
    ]
    if unknown:
        raise LiveCatalogueError(
            "Unknown Q0b parameter key(s) rejected before solver allocation: " +
            ", ".join(sorted(repr(key) for key in unknown)),
            category="invalid_parameter",
        )


def _metadata_identity_reasons(
    parameter_values: Mapping[str, object], overlay: Mapping[str, object],
) -> tuple[str, ...]:
    strict_ids = overlay["request_policy"][
        "strict_2dh_metadata_only_entry_ids"]
    aliases = {
        alias.casefold()
        for entry_id in strict_ids
        for alias in (str(entry_id), str(entry_id).rsplit(".", 1)[-1])
    }
    supplied = sorted(
        str(key) for key in parameter_values
        if isinstance(key, str) and key.casefold() in aliases
    )
    if not supplied:
        return ()
    return (
        "Q0b retained " + ", ".join(supplied) +
        " as strict-KMAX=1 input identity metadata; it is numerically inert "
        "because classic horizontal diffusion uses fixed dicuv/0.7.",
    )


def resolve_additive_compatibility(
    selected_method_ids: Iterable[str], *,
    parameter_values: Mapping[str, object] | None = None,
    wave_or_orbital_forcing: bool = False,
    document: Mapping[str, object] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Apply Q0b exclusion preflight, then the unchanged inherited resolver."""
    catalogue = _catalogue(document)
    selected = _iterable_strings(selected_method_ids, "Selected method IDs")
    if parameter_values is not None and not isinstance(parameter_values, Mapping):
        raise LiveCatalogueError(
            "Parameter values must be an object", category="invalid_parameter")
    parameters = dict(parameter_values or {})
    _preflight_scope_requests(
        selected, parameters, catalogue["catalogue_scope_repair"],
        wave_or_orbital_forcing=wave_or_orbital_forcing,
    )
    # Q0b does not change automatic physical selection in this zero-step gate.
    resolved, inherited_reasons = _v1.resolve_additive_compatibility(
        selected,
        parameter_values=parameters,
        wave_or_orbital_forcing=wave_or_orbital_forcing,
    )
    return resolved, inherited_reasons + _metadata_identity_reasons(
        parameters, catalogue["catalogue_scope_repair"])


def _backend_admitted(row: Mapping[str, object], backend: str) -> bool:
    implementation = row.get("implementation")
    accepted = row.get("accepted_state_connection")
    return (isinstance(implementation, Mapping) and
            isinstance(implementation.get(backend), Mapping) and
            implementation[backend].get("state") in _IMPLEMENTED_STATES and
            isinstance(accepted, Mapping) and
            isinstance(accepted.get(backend), Mapping) and
            accepted[backend].get("state") == _CONNECTED_STATE)


def resolve_catalogue_profile(
    selected_method_ids: Iterable[str], *,
    parameter_values: Mapping[str, object] | None = None,
    payload_ids: Iterable[str] = (), output_selector_ids: Iterable[str] = (),
    backend: str | None = None, wave_or_orbital_forcing: bool = False,
    document: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Resolve the typed Q0b method/payload/output surface without fallback."""
    catalogue = _catalogue(document)
    methods = _iterable_strings(selected_method_ids, "Selected method IDs")
    payloads = _iterable_strings(payload_ids, "Payload IDs")
    outputs = _iterable_strings(output_selector_ids, "Output selector IDs")
    if parameter_values is not None and not isinstance(parameter_values, Mapping):
        raise LiveCatalogueError(
            "Parameter values must be an object", category="invalid_parameter")
    parameters = dict(parameter_values or {})
    _preflight_scope_requests(
        methods + payloads + outputs, parameters,
        catalogue["catalogue_scope_repair"],
        wave_or_orbital_forcing=wave_or_orbital_forcing,
    )
    selected_methods, reasons = _v1.resolve_additive_compatibility(
        methods,
        parameter_values=parameters,
        wave_or_orbital_forcing=wave_or_orbital_forcing,
    )
    reasons = reasons + _metadata_identity_reasons(
        parameters, catalogue["catalogue_scope_repair"])
    physical = {row["entry_id"]: row for row in catalogue["effective_entries"]}
    lifecycle = {row["entry_id"]: row for row in catalogue["entry_lifecycle"]}
    for entry_id in payloads:
        row = physical.get(entry_id)
        if row is None or row.get("kind") not in {"payload", "interface"}:
            raise LiveCatalogueError(
                f"Unknown or non-payload entry: {entry_id}",
                category="unknown_entry")
    for entry_id in outputs:
        row = physical.get(entry_id)
        if row is None or row.get("kind") != "output_selector":
            raise LiveCatalogueError(
                f"Unknown or non-output entry: {entry_id}",
                category="unknown_entry")
    if backend is not None:
        if backend not in BACKENDS:
            raise LiveCatalogueError(
                f"Unknown backend: {backend}", category="invalid_backend")
        unavailable = [
            entry_id
            for entry_id in selected_methods + payloads + outputs
                       if not _backend_admitted(lifecycle[entry_id], backend)]
        if unavailable:
            raise LiveCatalogueError(
                f"Q0b entries are not admitted on {backend}: " +
                ", ".join(sorted(unavailable)),
                category="implementation_restriction")
    return {
        "method_ids": selected_methods,
        "payload_ids": tuple(sorted(payloads)),
        "output_selector_ids": tuple(sorted(outputs)),
        "automatic_selection_reasons": reasons,
        "backend": backend,
        "parent_live_catalogue_identity": copy.deepcopy(
            catalogue["parent_live_catalogue_identity"]),
        "q0b_identity": copy.deepcopy(
            catalogue["catalogue_scope_repair"]["identity"]),
    }


def exact_profile_admission(
    admission_id: str, backend: str, *,
    document: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    if backend not in BACKENDS:
        raise LiveCatalogueError(
            f"Unknown backend: {backend}", category="invalid_backend")
    catalogue = _catalogue(document)
    for row in catalogue["exact_profile_admissions"]:
        if row.get("admission_id") == admission_id:
            status = row.get("backend_status")
            if not isinstance(status, Mapping) or status.get(backend) not in ADMITTED:
                raise LiveCatalogueError(
                    f"Exact profile {admission_id} is not admitted on {backend}",
                    category="implementation_restriction")
            return copy.deepcopy(row)
    raise LiveCatalogueError(
        f"Unknown exact profile admission: {admission_id}",
        category="unknown_profile")


def product_profile_admission(
    product_profile_id: str, backend: str, *, require_windows_package: bool = False,
    document: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    if backend not in BACKENDS:
        raise LiveCatalogueError(
            f"Unknown backend: {backend}", category="invalid_backend")
    catalogue = _catalogue(document)
    for row in catalogue["product_profile_admissions"]:
        if row.get("product_profile_id") != product_profile_id:
            continue
        status = row.get("backend_status")
        surfaces = row.get("surfaces")
        if not isinstance(status, Mapping) or status.get(backend) != "qgis_admitted":
            raise LiveCatalogueError(
                f"Product profile {product_profile_id} is not admitted on {backend}",
                category="product_restriction")
        if require_windows_package and (not isinstance(surfaces, Mapping) or
                                        surfaces.get("windows_package") != "admitted"):
            raise LiveCatalogueError(
                f"Product profile {product_profile_id} is not admitted in a Windows package",
                category="product_restriction")
        return copy.deepcopy(row)
    raise LiveCatalogueError(
        f"Unknown product profile admission: {product_profile_id}",
        category="unknown_product_profile")


def stage_manifest(
    stage_id: str, *, document: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    catalogue = _catalogue(document)
    for row in catalogue["catalogue_scope_repair"]["stage_manifests"]:
        if row.get("stage_id") == stage_id:
            return copy.deepcopy(row)
    raise LiveCatalogueError(
        f"Unknown Q0b stage manifest: {stage_id}", category="unknown_entry")


__all__ = [
    "ADMITTED", "BACKENDS", "LIVE_CATALOGUE_FILE", "LIVE_CATALOGUE_REVISION",
    "LIVE_CATALOGUE_SCHEMA", "SCOPE_OVERLAY_FILE", "SCOPE_OVERLAY_REVISION",
    "SCOPE_OVERLAY_SCHEMA", "LiveCatalogueError", "compose_effective_catalogue",
    "effective_compatibility_rules", "effective_entry", "effective_requiredness",
    "exact_profile_admission", "lifecycle_entry", "load_live_catalogue",
    "load_scope_overlay", "product_profile_admission",
    "resolve_additive_compatibility", "resolve_catalogue_profile",
    "scope_identity_payloads", "stage_manifest",
]
