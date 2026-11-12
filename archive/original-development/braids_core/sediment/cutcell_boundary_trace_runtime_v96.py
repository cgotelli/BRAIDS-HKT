"""Isolated exact cut-cell boundary-owner adapter for Q4/v96 research.

This module does not replace or patch :mod:`solver.cut_cells.cutcell_solver`.
At build time it verifies the frozen parent SHA, extracts the existing native
mode-2/6/7/8/9 trace and physical-flux statements byte-for-byte, and places them
in a caller-owned core.  Two small ABI wrappers (``explicit`` and ``implicit``)
call that one core.  Standard and viscosity-equilibrated mode 6, diagnostic
stage/depth modes 7/8, surveyed mode 9, and the mode-6 GAIA fixed-face pressure
branch are admitted. Mode 2 is admitted only with the exact-type current-
iterate production ``scheme==1`` reconstruction owner; bedload law 2 also
requires its accepted hiding/D90/availability/MorFac receipt. Higher-order
reconstruction is still fail-closed and is never replaced by a cell-centred
substitute.

The resulting provider implements the already-frozen
``HydraulicBoundaryTraceProviderV96`` handoff.  It owns neither a clock nor an
accepted state and it never calls the non-reentrant production ``euler``
routine or its ``SAVE`` workspace.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Mapping

import numpy as np

from .hydraulic_boundary_trace_cpu_v96 import (
    OWNER_CUTCELL,
    HydraulicBoundaryTraceExecutionV96,
    HydraulicBoundaryTraceProviderV96,
    execution_document_hydraulic_boundary_trace_v96,
)


_ROOT = Path(__file__).resolve().parents[2]
_PARENT = _ROOT / "solver/cut_cells/cutcell_solver.f90"
_WRAPPER = Path(__file__).resolve()
_DEPENDENCIES = (
    "solver/wave/braids_hydraulic_roughness_module.f90",
    "solver/wave/braids_morphodynamic_flux_module.f90",
    "solver/cut_cells/braids_viscous_inlet_module.f90",
)
_FROZEN_DEPENDENCY_SHA256 = {
    "solver/wave/braids_hydraulic_roughness_module.f90":
        "255e523e3650e5b1d28fdbdea7c7d6302e4d1d954adda540b7ad019af6b04fbf",
    "solver/wave/braids_morphodynamic_flux_module.f90":
        "9b252be10db35f102fe80fbde8dd92bb7a950b94922c26d5f4a7e4328156ae6a",
    "solver/cut_cells/braids_viscous_inlet_module.f90":
        "7c91d088a7224a6bedced1e51e45d5509397dcfead139445c368bf56f23e8f58",
}

# Frozen at authorization of the isolated extraction on 2026-10-10.  A later
# production refactor must use a separately labelled compatibility replay; it
# must not silently remint this reference.
FROZEN_PARENT_SHA256 = \
    "7f14c1531473ec5f246e4b6d79c1e11bc23d43407879732091804a6ae6c58f2c"
ABI = "braids-cutcell-boundary-trace-v96-extracted-modes26789-v4"
SCOPE = "q4-v96-isolated-exact-cutcell-modes26789-boundary-owner"
_SHA = re.compile(r"[0-9a-f]{64}")


class CutcellBoundaryTraceRuntimeV96Error(ValueError):
    """Fail-closed error for the isolated native boundary owner."""


def _file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _digest(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _hex(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise CutcellBoundaryTraceRuntimeV96Error(
            f"v96 cutcell {name} SHA-256 differs")
    return value


def _float_array(value: object, shape: tuple[int, ...], name: str,
                 *, nonnegative: bool = False,
                 positive: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "fiub" or
            not np.isfinite(raw).all()):
        raise CutcellBoundaryTraceRuntimeV96Error(
            f"v96 cutcell {name} shape/type/finite identity differs")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    if ((nonnegative and np.any(result < 0.0)) or
            (positive and np.any(result <= 0.0))):
        raise CutcellBoundaryTraceRuntimeV96Error(
            f"v96 cutcell {name} sign differs")
    result.setflags(write=False)
    return result


def _integer_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "biuf" or
            not np.isfinite(raw).all() or np.any(raw != np.floor(raw)) or
            np.any(np.abs(raw) > np.iinfo(np.int32).max)):
        raise CutcellBoundaryTraceRuntimeV96Error(
            f"v96 cutcell {name} integer identity differs")
    result = np.array(raw, dtype=np.int32, order="C", copy=True)
    result.setflags(write=False)
    return result


def _line_end(source: bytes, start: int) -> int:
    end = source.index(b"\n", start) + 1
    return end


def _fragment(source: bytes, name: str, start: bytes, end: bytes,
              *, include_end_line: bool = False,
              search_start: int = 0,
              search_end: int | None = None) -> dict[str, object]:
    stop = len(source) if search_end is None else search_end
    if source.count(start, search_start, stop) != 1:
        raise CutcellBoundaryTraceRuntimeV96Error(
            "non-unique frozen cutcell fragment anchor: " + name)
    first = source.index(start, search_start, stop)
    if end == b"\n":
        last = source.index(end, first, stop) + 1
    else:
        last = source.index(end, first, stop)
    if include_end_line and end != b"\n":
        last = _line_end(source, last)
    data = source[first:last]
    return {
        "name": name,
        "start_byte": first,
        "end_byte": last,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": data,
    }


def _extract_parent_fragments(source: bytes) -> dict[str, dict[str, object]]:
    """Return exact production bytes used by the isolated core."""
    pieces: dict[str, dict[str, object]] = {}
    trace_begin = source.index(b" subroutine traces(")
    trace_end = source.index(b"\n end subroutine", trace_begin)
    region = source[trace_begin:trace_end]

    def trace_piece(name: str, start: bytes, end: bytes,
                    *, line: bool = False) -> None:
        if region.count(start) != 1:
            raise CutcellBoundaryTraceRuntimeV96Error(
                "non-unique frozen trace anchor: " + name)
        a = trace_begin + region.index(start)
        if end == b"\n":
            b = source.index(end, a) + 1
        else:
            b = source.index(end, a)
        if line and end != b"\n":
            b = _line_end(source, b)
        value = source[a:b]
        pieces[name] = {
            "name": name, "start_byte": a, "end_byte": b,
            "sha256": hashlib.sha256(value).hexdigest(), "bytes": value,
        }

    sampled_start = source.index(b" subroutine sampled_section_span(")
    sampled_end = source.index(b"\n end subroutine", sampled_start)
    sampled_end = _line_end(source, sampled_end+1)
    sampled = source[sampled_start:sampled_end]
    pieces["sampled_section_span"] = {
        "name": "sampled_section_span", "start_byte": sampled_start,
        "end_byte": sampled_end,
        "sha256": hashlib.sha256(sampled).hexdigest(), "bytes": sampled,
    }
    trace_piece(
        "trace_initialization",
        b"  ok=.false.;trace=0;branch=0;invariant=0;target_u=0;sampled_boundary_open=.true.",
        b"  if(present(pv)) then")
    trace_piece("trace_boundary_loop", b"  do b=1,nb",
                b"   if(mode(b)==4) then")
    trace_piece("trace_desired", b"   desired=hydraulic(1,b);targetarea=0",
                b"\n", line=True)
    trace_piece("trace_mode6", b"   if(mode(b)==6) then",
                b"   if(mode(b)==7) then")
    trace_piece("trace_mode7", b"   if(mode(b)==7) then",
                b"   if(mode(b)==8) then")
    trace_piece("trace_mode8", b"   if(mode(b)==8) then",
                b"   if(mode(b)==9) then")
    trace_piece("trace_mode9", b"   if(mode(b)==9) then",
                b"   call sampled_section_span(b,hydraulic(2,b),bed,first_wet,last_wet)")
    trace_piece(
        "trace_mode2_characteristic",
        b"   call sampled_section_span(b,hydraulic(2,b),bed,first_wet,last_wet)",
        b"   if(mode(b)/=1) cycle", line=True)
    success = (b"  enddo\n  ok=.true." if
               b"  enddo\n  ok=.true." in region else
               b"  enddo\r\n  ok=.true.")
    trace_piece("trace_success", success, b" contains")

    euler = source.index(b" subroutine euler(")
    euler_end = source.index(b"\n end subroutine", euler)
    pieces["physical_flux"] = _fragment(
        source, "physical_flux",
        b"    l=trace(:,f);depth=l(1);fl(1,f)=dot_product(l(2:3),normal(:,f))",
        b"    if(suspended_active) then", search_start=euler,
        search_end=euler_end)
    pieces["coupled_stage_call"] = _fragment(
        source, "coupled_stage_call",
        b"    if(mode(b)==2.and.bedlaw==2.and..not.capacity_foundation_runtime_declaration.and.",
        b"    l=trace(:,f);depth=l(1);fl(1,f)=dot_product(l(2:3),normal(:,f))",
        search_start=euler, search_end=euler_end)
    pieces["exchange_speed"] = _fragment(
        source, "exchange_speed",
        b"    hydraulic_face_exchange_area(f)=length(f)*max(0.d0,l(1))",
        b"    if(depth>dry_tolerance) speed(f)=speed(f)+abs(fl(1,f))/depth",
        include_end_line=True, search_start=euler, search_end=euler_end)
    pieces["sampled_pressure"] = _fragment(
        source, "sampled_pressure",
        b"   else if(sampled_boundary_bed(f)==1) then",
        b"   endif", include_end_line=True, search_start=euler,
        search_end=euler_end)
    pieces["fixed_pressure"] = _fragment(
        source, "fixed_pressure",
        b"   else if(fixed_bed_face) then",
        b"   else if(sampled_boundary_bed(f)==1) then",
        search_start=euler, search_end=euler_end)
    pieces["mode2_pressure"] = _fragment(
        source, "mode2_pressure",
        b"   else if(mode(b)==2) then",
        b"   else if(fixed_bed_face) then",
        search_start=euler, search_end=euler_end)
    pieces["water_ledger"] = _fragment(
        source, "water_ledger",
        b"   accepted(1,b)=accepted(1,b)-dt*length(f)*fl(1,f)",
        b"\n", include_end_line=True, search_start=euler,
        search_end=euler_end)
    face_value_start = source.index(b" function face_value(")
    scheme1_start = source.index(b"  value=v(:,c)", face_value_start)
    scheme1_end = _line_end(
        source, source.index(b"  if(scheme==1) return", scheme1_start))
    scheme1 = source[scheme1_start:scheme1_end]
    pieces["face_value_scheme1"] = {
        "name": "face_value_scheme1", "start_byte": scheme1_start,
        "end_byte": scheme1_end,
        "sha256": hashlib.sha256(scheme1).hexdigest(),
        "bytes": scheme1,
    }
    roughness_at_start = source.index(b" pure subroutine hydraulic_roughness_at(")
    roughness_at_end = source.index(b" end subroutine", roughness_at_start)
    roughness_at_end = _line_end(source, roughness_at_end + 1)
    roughness_at = source[roughness_at_start:roughness_at_end]
    pieces["hydraulic_roughness_at"] = {
        "name": "hydraulic_roughness_at", "start_byte": roughness_at_start,
        "end_byte": roughness_at_end,
        "sha256": hashlib.sha256(roughness_at).hexdigest(),
        "bytes": roughness_at,
    }
    roughness_weight_start = source.index(b" pure function roughness_face_weight(")
    roughness_weight_end = source.index(b" end function", roughness_weight_start)
    roughness_weight_end = _line_end(source, roughness_weight_end + 1)
    roughness_weight = source[roughness_weight_start:roughness_weight_end]
    pieces["roughness_face_weight"] = {
        "name": "roughness_face_weight", "start_byte": roughness_weight_start,
        "end_byte": roughness_weight_end,
        "sha256": hashlib.sha256(roughness_weight).hexdigest(),
        "bytes": roughness_weight,
    }
    equivalent_start = source.index(b" pure function hydraulic_equivalent_manning(")
    equivalent_end = source.index(b" end function", equivalent_start)
    equivalent_end = _line_end(source, equivalent_end + 1)
    equivalent = source[equivalent_start:equivalent_end]
    pieces["hydraulic_equivalent_manning"] = {
        "name": "hydraulic_equivalent_manning",
        "start_byte": equivalent_start, "end_byte": equivalent_end,
        "sha256": hashlib.sha256(equivalent).hexdigest(),
        "bytes": equivalent,
    }
    viscous_start = source.index(
        b" subroutine viscosity_equilibrated_mode6_profile(")
    viscous_end = source.index(b" end subroutine", viscous_start)
    viscous_end = _line_end(source, viscous_end + 1)
    viscous = source[viscous_start:viscous_end]
    pieces["viscosity_equilibrated_mode6_profile"] = {
        "name": "viscosity_equilibrated_mode6_profile",
        "start_byte": viscous_start, "end_byte": viscous_end,
        "sha256": hashlib.sha256(viscous).hexdigest(), "bytes": viscous,
    }
    coupled_start = source.index(b" subroutine coupled_stage_characteristic(")
    coupled_end = source.index(b" end subroutine", coupled_start)
    coupled_end = _line_end(source, coupled_end + 1)
    coupled = source[coupled_start:coupled_end]
    pieces["coupled_stage_characteristic"] = {
        "name": "coupled_stage_characteristic",
        "start_byte": coupled_start, "end_byte": coupled_end,
        "sha256": hashlib.sha256(coupled).hexdigest(), "bytes": coupled,
    }
    return pieces


_PREFIX = r"""module braids_cutcell_boundary_trace_extracted_v96
 use iso_c_binding
 use, intrinsic :: ieee_arithmetic
 use braids_hydraulic_roughness_module, only: braids_roughness_state,braids_roughness_manning
 use braids_morphodynamic_flux_module, only: gaia_fixed_inlet_face_terms,gaia_fixed_inlet_trace, &
  gaia_fixed_inlet_pressure,mpm_mass_jacobian
 use braids_viscous_inlet_module, only: braids_viscous_roughness_profile,BRAIDS_VISCOUS_INLET_SUCCESS
 implicit none
contains
 subroutine cutcell_boundary_core_v96(nc,nf,nb,nbf,nfrac,state,bed,hydraulic,reconstructed_face_input,hydro_constant_input, &
  manning_input,mid_input,fixed_bed_input, &
  owner_zero,normal,length,bid_zero,boundary_offset_zero,boundary_faces_zero,mode_input,fixed_mode_input, &
  viscosity_input,reversal_policy_input,boundary_seed_zero,sampled_input,bed_offset_input,roughness_mode_input,chzk2d_input, &
  bedlaw_input,production_flag,roughness_work_depth_input,horizontal_viscosity_input,dry_input, &
  manning_friction_work_depth_input,speed_limit_input,gravity_input,dt, &
  hiding_input,mixture_d90_input,availability_input,d50_input,rhos_input,bulk_input,min_depth_input, &
  rho_input,mpm_critical_input,mpm_cf_input,mpm_friction_mode_input,step_morfac_input, &
  trace,fl,speed,exchange_area, &
  physical_owned,exterior_density_active, &
  accepted_water,diagnostics,failed_values,status,route_kind)
  integer(c_int),intent(in) :: nc,nf,nb,nbf,nfrac,roughness_mode_input,chzk2d_input,bedlaw_input,production_flag,route_kind
  real(c_double),intent(in) :: state(3,nc),bed(nc),hydraulic(2,nb),manning_input(nc),mid_input(2,nf), &
   fixed_bed_input(nf),normal(2,nf),length(nf),reconstructed_face_input(4,nf)
  real(c_double),intent(in) :: bed_offset_input(nf),roughness_work_depth_input,horizontal_viscosity_input, &
   dry_input,manning_friction_work_depth_input,speed_limit_input,gravity_input,dt
  real(c_double),intent(in) :: hiding_input(nfrac,nc),mixture_d90_input(nc),availability_input(nfrac,nc), &
   d50_input(nfrac),rhos_input(nfrac),bulk_input(nfrac),min_depth_input(nfrac),rho_input,mpm_critical_input, &
   mpm_cf_input,step_morfac_input
  integer(c_int),intent(in) :: mpm_friction_mode_input
  integer(c_int),intent(in) :: owner_zero(nf),bid_zero(nf),boundary_offset_zero(nb+1),boundary_faces_zero(nbf)
  integer(c_int),intent(in) :: mode_input(nb),fixed_mode_input(nb),viscosity_input(nb), &
   reversal_policy_input(nb),boundary_seed_zero(nb),sampled_input(nf),hydro_constant_input(nc)
  real(c_double),intent(out) :: trace(3,nf),fl(3,nf),speed(nf),exchange_area(nf),accepted_water(nb),failed_values(6)
  integer(c_int),intent(out) :: physical_owned(nf),exterior_density_active(nf),diagnostics(6),status
  integer :: owner(nf),bid(nf),boundary_offset(nb+1),boundary_faces(nbf),mode(nb),boundary_seed_position(nb)
  integer :: sampled_boundary_bed(nf),fixed_boundary_bed_mode(nb)
  integer :: reversal_policy(nb)
  logical :: sampled_boundary_open(nf),production_section_geometry,ok,viscosity_equilibrated_inlet(nb),hydro_constant(nc)
  logical :: white_colebrook_chzk2d,fixed_bed_face,valid
  logical :: capacity_foundation_runtime_declaration,bedload_adaptation,bedform_roughness
  integer :: bedlaw,f,b,c,n,failure_reason,failed_speed_boundary,failed_trace_boundary,failed_trace_face
  integer :: failed_trace_condition,hydraulic_roughness_mode
  integer :: mpm_friction_mode
  real(8) :: boundary_bed_offset(nf),grav,dry_tolerance,speed_limit,failed_trace_speed,failed_trace_values(6)
  real(8) :: fr(3,nf),hydraulic_face_exchange_area(nf),accepted(1,nb),l(3),left(4),depth,boundary_stage_bed
  real(8) :: reconstructed_face_primitive(4,nf),pv_dummy(4,nc),pg_dummy(4,2,nc),plo_dummy(4,nc),phi_dummy(4,nc)
  real(8) :: manning_cell(nc),mid(2,nf),fixed_boundary_bed(nf)
  real(8) :: bc_hydraulic(2,nb)
  real(8) :: roughness_minimum_work_depth,horizontal_viscosity,manning_friction_chezy_minimum_depth
  real(8) :: hiding(nfrac,nc),mixture_d90(nc),resolved_bed_availability(nfrac,nc)
  real(8) :: d50_fraction(nfrac),rhos_fraction(nfrac),bulk_fraction(nfrac),bedload_min_depth_fraction(nfrac)
  real(8) :: rho,mpm_critical,mpm_cf,step_morfac
  real(8),parameter :: manning_chezy_minimum_depth=1.d-2
  integer,parameter :: nstate=3,nprimitive=4
  owner=owner_zero+1;bid=bid_zero+1;boundary_offset=boundary_offset_zero+1
  boundary_faces=boundary_faces_zero+1;mode=mode_input;boundary_seed_position=boundary_seed_zero+1
  sampled_boundary_bed=sampled_input;boundary_bed_offset=bed_offset_input
  fixed_boundary_bed_mode=fixed_mode_input;viscosity_equilibrated_inlet=viscosity_input/=0
  reversal_policy=reversal_policy_input;hydro_constant=hydro_constant_input/=0
  reconstructed_face_primitive=reconstructed_face_input
  manning_cell=manning_input;mid=mid_input;fixed_boundary_bed=fixed_bed_input
  bc_hydraulic=hydraulic
  hydraulic_roughness_mode=roughness_mode_input;white_colebrook_chzk2d=chzk2d_input==1
  roughness_minimum_work_depth=roughness_work_depth_input;horizontal_viscosity=horizontal_viscosity_input
  manning_friction_chezy_minimum_depth=manning_friction_work_depth_input
  production_section_geometry=production_flag==1;bedlaw=bedlaw_input
  hiding=hiding_input;mixture_d90=mixture_d90_input;resolved_bed_availability=availability_input
  d50_fraction=d50_input;rhos_fraction=rhos_input;bulk_fraction=bulk_input
  bedload_min_depth_fraction=min_depth_input;rho=rho_input;mpm_critical=mpm_critical_input
  mpm_cf=mpm_cf_input;mpm_friction_mode=mpm_friction_mode_input;step_morfac=step_morfac_input
  capacity_foundation_runtime_declaration=.false.;bedload_adaptation=.false.;bedform_roughness=.false.
  grav=gravity_input;dry_tolerance=dry_input;speed_limit=speed_limit_input
  failure_reason=0;failed_speed_boundary=0;failed_trace_boundary=0;failed_trace_face=0
  failed_trace_condition=0;failed_trace_speed=0.d0;failed_trace_values=0.d0;failed_values=0.d0
  trace=0.d0;fl=0.d0;fr=0.d0;speed=0.d0;hydraulic_face_exchange_area=0.d0
  accepted=0.d0;physical_owned=0;exterior_density_active=0;diagnostics=0;status=0
  if(nc<1.or.nf<1.or.nb<1.or.nbf<1.or.nfrac<1.or.production_flag/=1.or.dt<0.d0.or. &
     dry_tolerance<0.d0.or.speed_limit<=0.d0.or.grav<=0.d0.or. &
     roughness_minimum_work_depth<0.d0.or.horizontal_viscosity<0.d0.or. &
     manning_friction_chezy_minimum_depth<0.d0.or. &
     rho<=0.d0.or.mpm_critical<0.d0.or.mpm_cf<0.d0.or.step_morfac<0.d0.or. &
     all(hydraulic_roughness_mode/=[1,2,3])) then
   status=1;return
  endif
  if(.not.all(ieee_is_finite(state)).or..not.all(ieee_is_finite(bed)).or. &
     .not.all(ieee_is_finite(hydraulic)).or..not.all(ieee_is_finite(normal)).or. &
     .not.all(ieee_is_finite(length)).or..not.all(ieee_is_finite(boundary_bed_offset)).or. &
     .not.all(ieee_is_finite(manning_cell)).or..not.all(ieee_is_finite(mid)).or. &
     .not.all(ieee_is_finite(fixed_boundary_bed)).or..not.all(ieee_is_finite(hiding)).or. &
     .not.all(ieee_is_finite(mixture_d90)).or..not.all(ieee_is_finite(resolved_bed_availability)).or. &
     .not.all(ieee_is_finite(d50_fraction)).or..not.all(ieee_is_finite(rhos_fraction)).or. &
     .not.all(ieee_is_finite(bulk_fraction)).or..not.all(ieee_is_finite(bedload_min_depth_fraction))) then
   status=2;return
  endif
  pv_dummy=0.d0;pg_dummy=0.d0;plo_dummy=0.d0;phi_dummy=0.d0
  call native_boundary_trace_v96(state,bed,hydraulic,trace,ok,pv_dummy,pg_dummy,plo_dummy,phi_dummy)
  if(.not.ok) then
   status=10+max(0,failed_trace_condition)
   diagnostics=[failure_reason,failed_trace_boundary,failed_trace_face,failed_trace_condition,failed_speed_boundary,route_kind]
   failed_values=failed_trace_values
   return
  endif
  call native_open_flux_v96()
  do f=1,nf
   if(bid(f)>0) then
    physical_owned(f)=1
    if(sampled_boundary_open(f).and.trace(1,f)>dry_tolerance) exterior_density_active(f)=1
   endif
  enddo
  exchange_area=hydraulic_face_exchange_area;accepted_water=accepted(1,:);failed_values=failed_trace_values
  diagnostics=[failure_reason,failed_trace_boundary,failed_trace_face,failed_trace_condition,failed_speed_boundary,route_kind]
 contains
"""


_TRACE_DECLARATION = r"""
 subroutine native_boundary_trace_v96(state,bed,hydraulic,trace,ok,pv,pg,plo,phi)
  real(8),intent(in) :: state(nstate,nc),bed(nc),hydraulic(2,nb)
  real(8),intent(out) :: trace(3,nf)
  logical,intent(out) :: ok
  real(8),optional,intent(in) :: pv(nprimitive,nc),pg(nprimitive,2,nc),plo(nprimitive,nc),phi(nprimitive,nc)
  logical :: owner_incoming
  real(8) :: invariant(nf),target_u(nf),h,u,c,lo,hi,stage,total,targetarea,desired,ub,tangent(2),zavg,wet_length,weight
  real(8) :: incoming_left(3),incoming_right(3),incoming_speed,rightmost,tolerance
  real(8) :: edge(nprimitive),edge_state(3),edge_bed
  real(8) :: profile_driving,profile_discharge_residual,profile_acceleration_residual
  integer :: branch(nf),b,f,o,iter,k,first_wet,last_wet,profile_iterations
  logical :: profile_ok,gaia_fixed,face_valid
"""


_COUPLED_OWNER_HELPERS = r"""
 pure function bed_availability(state,fraction,cell) result(value)
  real(8),intent(in) :: state(nstate,nc)
  integer,intent(in) :: fraction,cell
  real(8) :: value
  value=resolved_bed_availability(fraction,cell)
 end function bed_availability

 pure logical function secondary_bedload_direction() result(enabled)
  enabled=.false.
 end function secondary_bedload_direction
"""


_FLUX_PREFIX = r"""
 function face_value(c,f,v,grad,vlo,vhi) result(value)
  integer,intent(in) :: c,f
  real(8),intent(in) :: v(nprimitive,nc),grad(nprimitive,2,nc),vlo(nprimitive,nc),vhi(nprimitive,nc)
  real(8) :: value(nprimitive)
  value=reconstructed_face_primitive(:,f)
 end function face_value

 pure integer function bed_primitive_index() result(index)
  index=4
 end function bed_primitive_index

 subroutine native_open_flux_v96()
  fl=0.d0;fr=0.d0;speed=0.d0;hydraulic_face_exchange_area=0.d0;accepted=0.d0
  do f=1,nf
   b=bid(f)
   if(b==0) cycle
   c=owner(f);n=-1
   left=reconstructed_face_primitive(:,f)
"""


_SUFFIX = r"""
  enddo
 end subroutine native_open_flux_v96
 end subroutine cutcell_boundary_core_v96

 subroutine braids_cutcell_boundary_explicit_v96(nc,nf,nb,nbf,nfrac,state,bed,hydraulic,reconstructed,hydroconstant,manning,mid,fixedbed,owner,normal, &
  length,bid,offset,faces,mode,fixedmode,viscosity,reversal,seed,sampled,bedoffset,roughmode,chzk2d,bedlaw,production,roughwork, &
  horizontalviscosity,dry,manningfrictionwork,speed_limit,gravity,dt,hiding,d90,availability,d50,rhos,bulk,mindepth, &
  waterdensity,critical,cf,frictionmode,morfac,trace,flux,wavespeed,exchange,physical,densin,ledger,diagnostics, &
  failed_values,status) bind(C,name='braids_cutcell_boundary_explicit_v96')
  integer(c_int),value :: nc,nf,nb,nbf,nfrac,roughmode,chzk2d,bedlaw,production,frictionmode
  real(c_double),intent(in) :: state(3,nc),bed(nc),hydraulic(2,nb),reconstructed(4,nf),manning(nc),mid(2,nf),fixedbed(nf),normal(2,nf), &
   length(nf),bedoffset(nf)
  real(c_double),value :: roughwork,horizontalviscosity,dry,manningfrictionwork,speed_limit,gravity,dt
  real(c_double),intent(in) :: hiding(nfrac,nc),d90(nc),availability(nfrac,nc),d50(nfrac),rhos(nfrac),bulk(nfrac), &
   mindepth(nfrac)
  real(c_double),value :: waterdensity,critical,cf,morfac
  integer(c_int),intent(in) :: hydroconstant(nc),owner(nf),bid(nf),offset(nb+1),faces(nbf),mode(nb),fixedmode(nb), &
   viscosity(nb),reversal(nb),seed(nb),sampled(nf)
  real(c_double),intent(out) :: trace(3,nf),flux(3,nf),wavespeed(nf),exchange(nf),ledger(nb),failed_values(6)
  integer(c_int),intent(out) :: physical(nf),densin(nf),diagnostics(6),status(1)
  call cutcell_boundary_core_v96(nc,nf,nb,nbf,nfrac,state,bed,hydraulic,reconstructed,hydroconstant,manning,mid,fixedbed, &
   owner,normal,length,bid,offset,faces,mode,fixedmode,viscosity,reversal,seed,sampled,bedoffset,roughmode,chzk2d,bedlaw,production,roughwork, &
   horizontalviscosity,dry,manningfrictionwork,speed_limit,gravity,dt,hiding,d90,availability,d50,rhos,bulk,mindepth, &
   waterdensity,critical,cf,frictionmode,morfac,trace,flux,wavespeed,exchange,physical,densin,ledger,diagnostics, &
   failed_values,status(1),1)
 end subroutine

 subroutine braids_cutcell_boundary_implicit_v96(nc,nf,nb,nbf,nfrac,state,bed,hydraulic,reconstructed,hydroconstant,manning,mid,fixedbed,owner,normal, &
  length,bid,offset,faces,mode,fixedmode,viscosity,reversal,seed,sampled,bedoffset,roughmode,chzk2d,bedlaw,production,roughwork, &
  horizontalviscosity,dry,manningfrictionwork,speed_limit,gravity,dt,hiding,d90,availability,d50,rhos,bulk,mindepth, &
  waterdensity,critical,cf,frictionmode,morfac,trace,flux,wavespeed,exchange,physical,densin,ledger,diagnostics, &
  failed_values,status) bind(C,name='braids_cutcell_boundary_implicit_v96')
  integer(c_int),value :: nc,nf,nb,nbf,nfrac,roughmode,chzk2d,bedlaw,production,frictionmode
  real(c_double),intent(in) :: state(3,nc),bed(nc),hydraulic(2,nb),reconstructed(4,nf),manning(nc),mid(2,nf),fixedbed(nf),normal(2,nf), &
   length(nf),bedoffset(nf)
  real(c_double),value :: roughwork,horizontalviscosity,dry,manningfrictionwork,speed_limit,gravity,dt
  real(c_double),intent(in) :: hiding(nfrac,nc),d90(nc),availability(nfrac,nc),d50(nfrac),rhos(nfrac),bulk(nfrac), &
   mindepth(nfrac)
  real(c_double),value :: waterdensity,critical,cf,morfac
  integer(c_int),intent(in) :: hydroconstant(nc),owner(nf),bid(nf),offset(nb+1),faces(nbf),mode(nb),fixedmode(nb), &
   viscosity(nb),reversal(nb),seed(nb),sampled(nf)
  real(c_double),intent(out) :: trace(3,nf),flux(3,nf),wavespeed(nf),exchange(nf),ledger(nb),failed_values(6)
  integer(c_int),intent(out) :: physical(nf),densin(nf),diagnostics(6),status(1)
  call cutcell_boundary_core_v96(nc,nf,nb,nbf,nfrac,state,bed,hydraulic,reconstructed,hydroconstant,manning,mid,fixedbed, &
   owner,normal,length,bid,offset,faces,mode,fixedmode,viscosity,reversal,seed,sampled,bedoffset,roughmode,chzk2d,bedlaw,production,roughwork, &
   horizontalviscosity,dry,manningfrictionwork,speed_limit,gravity,dt,hiding,d90,availability,d50,rhos,bulk,mindepth, &
   waterdensity,critical,cf,frictionmode,morfac,trace,flux,wavespeed,exchange,physical,densin,ledger,diagnostics, &
   failed_values,status(1),2)
 end subroutine
end module braids_cutcell_boundary_trace_extracted_v96
"""


def _assembled_source(source: bytes) -> tuple[bytes, dict[str, dict[str, object]]]:
    pieces = _extract_parent_fragments(source)
    text = _PREFIX.encode()
    text += pieces["sampled_section_span"]["bytes"]
    text += pieces["hydraulic_roughness_at"]["bytes"]
    text += pieces["hydraulic_equivalent_manning"]["bytes"]
    text += pieces["roughness_face_weight"]["bytes"]
    text += pieces["viscosity_equilibrated_mode6_profile"]["bytes"]
    text += _COUPLED_OWNER_HELPERS.encode()
    text += pieces["coupled_stage_characteristic"]["bytes"]
    text += _TRACE_DECLARATION.encode()
    for name in ("trace_initialization", "trace_boundary_loop",
                 "trace_desired", "trace_mode6", "trace_mode7",
                 "trace_mode8", "trace_mode9", "trace_mode2_characteristic",
                 "trace_success"):
        text += pieces[name]["bytes"]
    text += b"\n end subroutine native_boundary_trace_v96\n"
    text += _FLUX_PREFIX.encode()
    text += pieces["coupled_stage_call"]["bytes"]
    text += pieces["physical_flux"]["bytes"]
    text += pieces["exchange_speed"]["bytes"]
    text += b"    fixed_bed_face=.false.\n"
    text += b"    if(bedlaw==9) fixed_bed_face=fixed_boundary_bed_mode(b)==1\n"
    # No interior/neighbour branch exists in this boundary-only owner.  The
    # deliberately false arm preserves the exact production ELSE-IF bodies.
    text += b"    if(.false.) then\n"
    text += pieces["mode2_pressure"]["bytes"]
    text += pieces["fixed_pressure"]["bytes"]
    text += pieces["sampled_pressure"]["bytes"]
    text += pieces["water_ledger"]["bytes"]
    text += _SUFFIX.encode()
    return text, pieces


@dataclass(frozen=True)
class NativeCutcellBoundaryBuildReceiptV96:
    abi: str
    scope: str
    parent_source_sha256: str
    wrapper_source_sha256: str
    generated_source_sha256: str
    library_sha256: str
    compiler_sha256: str
    compiler_identity: str
    compile_flags: tuple[str, ...]
    fragment_identity: tuple[tuple[str, int, int, str], ...]
    dependency_source_identity: tuple[tuple[str, str], ...]
    active_production_source_modified: bool
    recursive_euler_call_count: int
    save_workspace_count: int
    receipt_sha256: str


def _build_payload(value: NativeCutcellBoundaryBuildReceiptV96) -> dict:
    return {key: item for key, item in value.__dict__.items()
            if key != "receipt_sha256"}


def _validate_build_receipt(value: NativeCutcellBoundaryBuildReceiptV96,
                            library_path: Path) -> None:
    if type(value) is not NativeCutcellBoundaryBuildReceiptV96:
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell native build receipt type differs")
    if (value.abi != ABI or value.scope != SCOPE or
            value.parent_source_sha256 != FROZEN_PARENT_SHA256 or
            value.active_production_source_modified or
            value.recursive_euler_call_count != 0 or
            value.save_workspace_count != 0 or
            value.receipt_sha256 != _fingerprint(_build_payload(value)) or
            _file_sha(_PARENT) != value.parent_source_sha256 or
            _file_sha(_WRAPPER) != value.wrapper_source_sha256 or
            any(not (_ROOT / relative).is_file() or
                _file_sha(_ROOT / relative) != digest or
                _FROZEN_DEPENDENCY_SHA256.get(relative) != digest
                for relative, digest in value.dependency_source_identity) or
            not library_path.is_file() or
            _file_sha(library_path) != value.library_sha256):
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell native build receipt differs")


@dataclass(frozen=True)
class CutcellBoundaryTraceExecutionV96:
    boundary_execution: HydraulicBoundaryTraceExecutionV96
    boundary_offset: np.ndarray
    boundary_faces: np.ndarray
    boundary_mode: np.ndarray
    fixed_boundary_bed_mode: np.ndarray
    boundary_seed_position: np.ndarray
    sampled_boundary_bed: np.ndarray
    boundary_bed_offset_m: np.ndarray
    manning_cell: np.ndarray
    face_midpoint_m: np.ndarray
    fixed_boundary_bed_m: np.ndarray
    viscosity_equilibrated_inlet: np.ndarray
    reversal_policy: np.ndarray
    normal: np.ndarray
    face_length_m: np.ndarray
    hydraulic_roughness_mode: int
    white_colebrook_chzk2d: bool
    roughness_minimum_work_depth_m: float
    manning_friction_chezy_minimum_depth_m: float
    horizontal_viscosity_m2_s: float
    gravity_m_s2: float
    dry_tolerance_m: float
    speed_limit_m_s: float
    bedload_law: int
    runtime_receipt_sha256: str
    fingerprint: str

    @property
    def cell_count(self) -> int:
        return self.boundary_execution.cell_count

    @property
    def face_count(self) -> int:
        return self.boundary_execution.face_count

    @property
    def boundary_count(self) -> int:
        return int(self.boundary_mode.size)


@dataclass(frozen=True)
class CutcellFaceReconstructionReceiptV96:
    """Exact transaction-local production ``scheme==1`` face image."""

    execution_fingerprint: str
    midpoint_q_sha256: str
    bed_elevation_sha256: str
    forcing_receipt_sha256: str
    evaluation_index: int
    reconstruction_scheme: int
    face_conserved_q: np.ndarray
    face_primitive: np.ndarray
    hydro_constant: np.ndarray
    parent_fragment_sha256: str
    resolver_runtime_sha256: str
    resolver_call_count: int
    receipt_sha256: str


def _reconstruction_token(value: CutcellFaceReconstructionReceiptV96) -> str:
    return _fingerprint({
        "scope": SCOPE + ":scheme1-face-reconstruction",
        "execution_fingerprint": value.execution_fingerprint,
        "midpoint_q_sha256": value.midpoint_q_sha256,
        "bed_elevation_sha256": value.bed_elevation_sha256,
        "forcing_receipt_sha256": value.forcing_receipt_sha256,
        "evaluation_index": value.evaluation_index,
        "reconstruction_scheme": value.reconstruction_scheme,
        "arrays_sha256": _digest(
            value.face_conserved_q, value.face_primitive,
            value.hydro_constant),
        "parent_fragment_sha256": value.parent_fragment_sha256,
        "resolver_runtime_sha256": value.resolver_runtime_sha256,
        "resolver_call_count": value.resolver_call_count,
        "owner": "same_cutcell_scheme1_reconstruction_owner",
    })


class CutcellFaceReconstructionResolverV96:
    """Exact-type, stateless owner for the existing first-order face path."""

    __slots__ = ("execution", "runtime_receipt", "runtime_sha256")

    def __init__(
        self, execution: CutcellBoundaryTraceExecutionV96,
        runtime_receipt: NativeCutcellBoundaryBuildReceiptV96,
    ) -> None:
        if (type(execution) is not CutcellBoundaryTraceExecutionV96 or
                execution.fingerprint !=
                    _fingerprint(_execution_payload(execution)) or
                type(runtime_receipt) is not
                    NativeCutcellBoundaryBuildReceiptV96 or
                runtime_receipt.receipt_sha256 !=
                    execution.runtime_receipt_sha256):
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell reconstruction resolver binding differs")
        fragments = dict(
            (name, digest) for name, _start, _end, digest in
            runtime_receipt.fragment_identity)
        if "face_value_scheme1" not in fragments:
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell scheme-1 source fragment is absent")
        self.execution = execution
        self.runtime_receipt = runtime_receipt
        self.runtime_sha256 = _fingerprint({
            "scope": SCOPE + ":scheme1-reconstruction-runtime",
            "execution": execution.fingerprint,
            "build": runtime_receipt.receipt_sha256,
            "fragment": fragments["face_value_scheme1"],
            "active_production_source_modified": False,
        })

    def resolve(
        self, midpoint_q: object, bed_elevation_m: object,
        forcing_receipt: object, *, evaluation_index: int = 1,
        cancelled: bool | object = False,
    ) -> CutcellFaceReconstructionReceiptV96:
        cancel = cancelled if callable(cancelled) else lambda: cancelled
        if not callable(cancelled) and type(cancelled) is not bool:
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell reconstruction cancellation differs")
        if cancel():
            raise InterruptedError(
                "v96 cutcell reconstruction cancelled before evaluation")
        execution = self.execution
        q = _float_array(
            midpoint_q, (3, execution.cell_count),
            "reconstruction midpoint state")
        bed = _float_array(
            bed_elevation_m, (execution.cell_count,),
            "reconstruction accepted bed")
        if np.any(q[0] < 0.0):
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell reconstruction depth differs")
        forcing_sha = getattr(forcing_receipt, "receipt_sha256", None)
        if (not isinstance(forcing_sha, str) or
                _SHA.fullmatch(forcing_sha) is None or
                type(evaluation_index) is not int or evaluation_index < 1):
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell reconstruction forcing identity differs")
        face = np.zeros((4, execution.face_count), dtype=np.float64)
        owner = execution.boundary_execution.owner
        face_q = np.ascontiguousarray(q[:, owner])
        face[0] = q[0, owner]+bed[owner]
        wet = q[0, owner] > execution.dry_tolerance_m
        face[1, wet] = q[1, owner[wet]]/q[0, owner[wet]]
        face[2, wet] = q[2, owner[wet]]/q[0, owner[wet]]
        face[3] = bed[owner]
        hydro = np.zeros(execution.cell_count, dtype=np.int32)
        face.setflags(write=False)
        face_q.setflags(write=False)
        hydro.setflags(write=False)
        fragment = dict(
            (name, digest) for name, _start, _end, digest in
            self.runtime_receipt.fragment_identity)["face_value_scheme1"]
        placeholder = CutcellFaceReconstructionReceiptV96(
            execution.fingerprint, _digest(q), _digest(bed), forcing_sha,
            evaluation_index, 1, face_q, face, hydro, fragment,
            self.runtime_sha256, 1, "")
        result = CutcellFaceReconstructionReceiptV96(
            **{**placeholder.__dict__,
               "receipt_sha256": _reconstruction_token(placeholder)})
        if cancel():
            raise InterruptedError(
                "v96 cutcell reconstruction cancelled after evaluation")
        return result


def _execution_payload(value: CutcellBoundaryTraceExecutionV96) -> dict:
    return {
        "scope": SCOPE,
        "boundary_execution": value.boundary_execution.fingerprint,
        "arrays_sha256": _digest(
            value.boundary_offset, value.boundary_faces,
            value.boundary_mode, value.fixed_boundary_bed_mode,
            value.boundary_seed_position,
            value.sampled_boundary_bed, value.boundary_bed_offset_m,
            value.manning_cell, value.face_midpoint_m,
            value.fixed_boundary_bed_m,
            value.viscosity_equilibrated_inlet,
            value.reversal_policy,
            value.normal, value.face_length_m),
        "hydraulic_roughness_mode": value.hydraulic_roughness_mode,
        "white_colebrook_chzk2d": value.white_colebrook_chzk2d,
        "roughness_minimum_work_depth_m":
            value.roughness_minimum_work_depth_m,
        "manning_friction_chezy_minimum_depth_m":
            value.manning_friction_chezy_minimum_depth_m,
        "horizontal_viscosity_m2_s": value.horizontal_viscosity_m2_s,
        "gravity_m_s2": value.gravity_m_s2,
        "dry_tolerance_m": value.dry_tolerance_m,
        "speed_limit_m_s": value.speed_limit_m_s,
        "bedload_law": value.bedload_law,
        "runtime_receipt_sha256": value.runtime_receipt_sha256,
        "production_section_geometry": True,
        "admitted_boundary_modes": sorted(
            int(item) for item in np.unique(value.boundary_mode)),
        "gaia_fixed_face_branch": bool(
            value.bedload_law == 9 and
            np.any(value.fixed_boundary_bed_mode == 1)),
    }


def resolve_cutcell_boundary_trace_execution_v96(
    boundary_execution: HydraulicBoundaryTraceExecutionV96, *,
    boundary_offset: object, boundary_faces: object,
    boundary_mode: object, boundary_seed_position: object,
    fixed_boundary_bed_mode: object, bedload_law: int,
    sampled_boundary_bed: object, boundary_bed_offset_m: object,
    manning_cell: object | None = None,
    face_midpoint_m: object | None = None,
    fixed_boundary_bed_m: object | None = None,
    viscosity_equilibrated_inlet: object | None = None,
    reversal_policy: object | None = None,
    normal: object, face_length_m: object,
    hydraulic_roughness_mode: int = 2,
    white_colebrook_chzk2d: bool = False,
    roughness_minimum_work_depth_m: float = 0.0,
    manning_friction_chezy_minimum_depth_m: float = 0.0,
    horizontal_viscosity_m2_s: float = 0.0,
    gravity_m_s2: float, dry_tolerance_m: float, speed_limit_m_s: float,
    runtime_receipt: NativeCutcellBoundaryBuildReceiptV96,
) -> CutcellBoundaryTraceExecutionV96:
    document = execution_document_hydraulic_boundary_trace_v96(
        boundary_execution)
    if (boundary_execution.hydraulic_owner_kind != OWNER_CUTCELL or
            document["physical_boundary_qualification"] is not True):
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell execution requires the physical cutcell owner kind")
    nf = boundary_execution.face_count
    raw_mode = np.asarray(boundary_mode)
    if raw_mode.ndim != 1 or raw_mode.size < 1:
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell boundary mode dimensions differ")
    nb = int(raw_mode.size)
    offset = _integer_array(boundary_offset, (nb+1,), "boundary offset")
    raw_faces = np.asarray(boundary_faces)
    if raw_faces.ndim != 1:
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell boundary faces dimensions differ")
    faces = _integer_array(raw_faces, (raw_faces.size,), "boundary faces")
    mode = _integer_array(raw_mode, (nb,), "boundary mode")
    fixed = _integer_array(
        fixed_boundary_bed_mode, (nb,), "fixed boundary bed mode")
    seed = _integer_array(
        boundary_seed_position, (nb,), "boundary seed position")
    sampled = _integer_array(
        sampled_boundary_bed, (nf,), "sampled boundary bed")
    bed_offset = _float_array(
        boundary_bed_offset_m, (nf,), "boundary bed offset")
    supplied_manning = manning_cell is not None
    supplied_midpoint = face_midpoint_m is not None
    supplied_fixed_bed = fixed_boundary_bed_m is not None
    manning = _float_array(
        np.full(boundary_execution.cell_count, 0.03)
        if manning_cell is None else manning_cell,
        (boundary_execution.cell_count,), "Manning/roughness parameter",
        positive=True)
    midpoint = _float_array(
        np.zeros((2, nf)) if face_midpoint_m is None else face_midpoint_m,
        (2, nf), "face midpoint")
    fixed_bed = _float_array(
        np.zeros(nf) if fixed_boundary_bed_m is None
        else fixed_boundary_bed_m,
        (nf,), "fixed boundary bed")
    viscosity = _integer_array(
        np.zeros(nb, dtype=np.int32)
        if viscosity_equilibrated_inlet is None
        else viscosity_equilibrated_inlet,
        (nb,), "viscosity-equilibrated inlet")
    reversal = _integer_array(
        np.ones(nb, dtype=np.int32) if reversal_policy is None
        else reversal_policy, (nb,), "outlet reversal policy")
    unit_normal = _float_array(normal, (2, nf), "face normal")
    length = _float_array(
        face_length_m, (nf,), "face length", positive=True)
    gravity = float(gravity_m_s2)
    dry = float(dry_tolerance_m)
    speed = float(speed_limit_m_s)
    rough_work = float(roughness_minimum_work_depth_m)
    friction_work = float(manning_friction_chezy_minimum_depth_m)
    horizontal_viscosity = float(horizontal_viscosity_m2_s)
    if (type(bedload_law) is not int or
            type(hydraulic_roughness_mode) is not int or
            type(white_colebrook_chzk2d) is not bool):
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell bedload law identity differs")
    if (not all(math.isfinite(value) for value in (
            gravity, dry, speed, rough_work, friction_work,
            horizontal_viscosity)) or
            gravity <= 0.0 or dry < 0.0 or speed <= 0.0 or
            rough_work < 0.0 or friction_work < 0.0 or
            horizontal_viscosity < 0.0 or
            hydraulic_roughness_mode not in (1, 2, 3) or
            runtime_receipt.scope != SCOPE or
            _SHA.fullmatch(runtime_receipt.receipt_sha256) is None):
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell physical/runtime controls differ")
    if (offset[0] != 0 or offset[-1] != faces.size or
            np.any(np.diff(offset) < 1) or
            np.any(faces < 0) or np.any(faces >= nf) or
            np.unique(faces).size != faces.size or
            not np.array_equal(np.sort(faces),
                               np.flatnonzero(boundary_execution.open_mask)) or
            np.any(~np.isin(mode, (2, 6, 7, 8, 9))) or
            np.any(seed < -1) or
            np.any((fixed != 0) & (fixed != 1)) or
            np.any((viscosity != 0) & (viscosity != 1)) or
            np.any((reversal != 0) & (reversal != 1)) or
            np.any((sampled != 0) & (sampled != 1)) or
            np.any(np.abs(np.linalg.norm(unit_normal, axis=0)-1.0) > 2e-13)):
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell modes-2/6/7/8/9 topology/geometry scope differs")
    if ((np.any(np.isin(mode, (6, 7, 8))) and not supplied_manning) or
            (np.any(viscosity == 1) and
             (not supplied_midpoint or horizontal_viscosity <= 0.0)) or
            np.any((viscosity == 1) & (mode != 6)) or
            (np.any(fixed == 1) and
             (bedload_law != 9 or not supplied_fixed_bed)) or
            np.any((fixed == 1) & (mode != 6))):
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell branch dependency scope differs")
    for boundary in range(nb):
        selected = faces[offset[boundary]:offset[boundary+1]]
        if (np.any(boundary_execution.boundary_index[selected] != boundary) or
                np.any(boundary_execution.boundary_mode[selected] !=
                       mode[boundary])):
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell face/boundary mode ownership differs")
    placeholder = CutcellBoundaryTraceExecutionV96(
        boundary_execution, offset, faces, mode, fixed, seed, sampled,
        bed_offset, manning, midpoint, fixed_bed, viscosity, reversal,
        unit_normal, length, hydraulic_roughness_mode,
        white_colebrook_chzk2d, rough_work, friction_work,
        horizontal_viscosity, gravity, dry, speed, bedload_law,
        runtime_receipt.receipt_sha256, "")
    return CutcellBoundaryTraceExecutionV96(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


@dataclass(frozen=True)
class CutcellBoundaryTrialForcingReceiptV96:
    execution_fingerprint: str
    hydraulic_request_sha256: str
    parent_state_sha256: str
    bed_elevation_sha256: str
    evaluation_time_s: float
    dt_s: float
    resolved_hydraulic: np.ndarray
    section_cache_before_sha256: str
    section_cache_after_sha256: str
    accepted_state_publication_count: int
    receipt_sha256: str


def _forcing_token(value: CutcellBoundaryTrialForcingReceiptV96) -> str:
    return _fingerprint({
        "scope": SCOPE + ":trial-forcing",
        "execution_fingerprint": value.execution_fingerprint,
        "hydraulic_request_sha256": value.hydraulic_request_sha256,
        "parent_state_sha256": value.parent_state_sha256,
        "bed_elevation_sha256": value.bed_elevation_sha256,
        "evaluation_time_s": value.evaluation_time_s,
        "dt_s": value.dt_s,
        "resolved_hydraulic_sha256": _digest(value.resolved_hydraulic),
        "section_cache_before_sha256": value.section_cache_before_sha256,
        "section_cache_after_sha256": value.section_cache_after_sha256,
        "accepted_state_publication_count":
            value.accepted_state_publication_count,
    })


def make_cutcell_boundary_trial_forcing_receipt_v96(
    execution: CutcellBoundaryTraceExecutionV96, *,
    hydraulic_request_sha256: str, parent_state_sha256: str,
    bed_elevation_m: object, evaluation_time_s: float, dt_s: float,
    resolved_hydraulic: object,
    section_cache_before_sha256: str,
    section_cache_after_sha256: str,
) -> CutcellBoundaryTrialForcingReceiptV96:
    if (type(execution) is not CutcellBoundaryTraceExecutionV96 or
            execution.fingerprint !=
                _fingerprint(_execution_payload(execution))):
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell execution fingerprint differs")
    bed = _float_array(
        bed_elevation_m, (execution.cell_count,), "forcing bed")
    hydraulic = _float_array(
        resolved_hydraulic, (2, execution.boundary_count),
        "resolved hydraulic")
    if (np.any(hydraulic[0] < 0.0) or
            np.any(hydraulic[1, execution.boundary_mode == 8] < 0.0)):
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell resolved hydraulic sign differs")
    time = float(evaluation_time_s)
    dt = float(dt_s)
    if (not math.isfinite(time) or time < 0.0 or
            not math.isfinite(dt) or dt < 0.0):
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell forcing clock differs")
    placeholder = CutcellBoundaryTrialForcingReceiptV96(
        execution.fingerprint,
        _hex(hydraulic_request_sha256, "hydraulic request"),
        _hex(parent_state_sha256, "parent state"), _digest(bed), time, dt,
        hydraulic,
        _hex(section_cache_before_sha256, "section cache before"),
        _hex(section_cache_after_sha256, "section cache after"), 0, "")
    return CutcellBoundaryTrialForcingReceiptV96(
        **{**placeholder.__dict__, "receipt_sha256":
           _forcing_token(placeholder)})


@dataclass(frozen=True)
class CutcellBoundaryTrialForcingResolverV96:
    """Stateless accepted-schedule bridge for an actual outer trial.

    The hydraulic schedule and transaction-local section-cache identities are
    resolved by the cut-cell owner before the nonlinear solve. This object
    binds that immutable image, then mints the forcing receipt from the actual
    outer hydraulic request, accepted parent, bed and residual time. It owns
    no clock and publishes no cache state.
    """

    execution_fingerprint: str
    runtime_receipt_sha256: str
    dt_s: float
    resolved_hydraulic: np.ndarray
    section_cache_before_sha256: str
    section_cache_after_sha256: str
    accepted_state_publication_count: int
    resolver_sha256: str

    def resolve(
        self, execution: CutcellBoundaryTraceExecutionV96, *,
        hydraulic_request_sha256: str, parent_state_sha256: str,
        bed_elevation_m: object, evaluation_time_s: float,
        evaluation_dt_s: float,
    ) -> CutcellBoundaryTrialForcingReceiptV96:
        evaluation_dt = float(evaluation_dt_s)
        if (type(execution) is not CutcellBoundaryTraceExecutionV96 or
                execution.fingerprint != self.execution_fingerprint or
                execution.runtime_receipt_sha256 !=
                    self.runtime_receipt_sha256 or
                self.resolver_sha256 != _forcing_resolver_token(self) or
                self.accepted_state_publication_count != 0 or
                not math.isfinite(evaluation_dt) or
                evaluation_dt != self.dt_s):
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell forcing resolver binding differs")
        result = make_cutcell_boundary_trial_forcing_receipt_v96(
            execution,
            hydraulic_request_sha256=hydraulic_request_sha256,
            parent_state_sha256=parent_state_sha256,
            bed_elevation_m=bed_elevation_m,
            evaluation_time_s=evaluation_time_s,
            dt_s=self.dt_s,
            resolved_hydraulic=self.resolved_hydraulic,
            section_cache_before_sha256=
                self.section_cache_before_sha256,
            section_cache_after_sha256=
                self.section_cache_after_sha256)
        if (result.hydraulic_request_sha256 !=
                _hex(hydraulic_request_sha256, "hydraulic request") or
                result.parent_state_sha256 !=
                _hex(parent_state_sha256, "parent state")):
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell forcing resolver request/parent differs")
        return result


def _forcing_resolver_token(
    value: CutcellBoundaryTrialForcingResolverV96,
) -> str:
    return _fingerprint({
        "scope": SCOPE + ":stateless-trial-forcing-resolver",
        "execution_fingerprint": value.execution_fingerprint,
        "runtime_receipt_sha256": value.runtime_receipt_sha256,
        "dt_s": value.dt_s,
        "resolved_hydraulic_sha256": _digest(value.resolved_hydraulic),
        "section_cache_before_sha256":
            value.section_cache_before_sha256,
        "section_cache_after_sha256": value.section_cache_after_sha256,
        "accepted_state_publication_count":
            value.accepted_state_publication_count,
        "forcing_owner": "cutcell_boundary_hydraulic_at_resolved_image",
    })


def resolve_cutcell_boundary_trial_forcing_resolver_v96(
    execution: CutcellBoundaryTraceExecutionV96,
    runtime_receipt: NativeCutcellBoundaryBuildReceiptV96, *,
    dt_s: float, resolved_hydraulic: object,
    section_cache_before_sha256: str,
    section_cache_after_sha256: str,
) -> CutcellBoundaryTrialForcingResolverV96:
    if (type(execution) is not CutcellBoundaryTraceExecutionV96 or
            execution.fingerprint != _fingerprint(
                _execution_payload(execution)) or
            type(runtime_receipt) is not
                NativeCutcellBoundaryBuildReceiptV96 or
            runtime_receipt.receipt_sha256 !=
                execution.runtime_receipt_sha256 or
            runtime_receipt.receipt_sha256 !=
                _fingerprint(_build_payload(runtime_receipt))):
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell forcing resolver execution/runtime differs")
    dt = float(dt_s)
    hydraulic = _float_array(
        resolved_hydraulic, (2, execution.boundary_count),
        "forcing resolver hydraulic")
    if (not math.isfinite(dt) or dt <= 0.0 or
            np.any(hydraulic[0] < 0.0) or
            np.any(hydraulic[1, execution.boundary_mode == 8] < 0.0)):
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell forcing resolver clock/hydraulic differs")
    placeholder = CutcellBoundaryTrialForcingResolverV96(
        execution.fingerprint, runtime_receipt.receipt_sha256, dt,
        hydraulic,
        _hex(section_cache_before_sha256, "section cache before"),
        _hex(section_cache_after_sha256, "section cache after"),
        0, "")
    return CutcellBoundaryTrialForcingResolverV96(
        **{**placeholder.__dict__,
           "resolver_sha256": _forcing_resolver_token(placeholder)})


@dataclass(frozen=True)
class CutcellMode2MPMReceiptV96:
    execution_fingerprint: str
    forcing_receipt_sha256: str
    runtime_receipt_sha256: str
    accepted_sediment_state_sha256: str
    hiding_owner_receipt_sha256: str
    availability_owner_receipt_sha256: str
    hiding: np.ndarray
    mixture_d90_m: np.ndarray
    bed_availability: np.ndarray
    d50_m: np.ndarray
    sediment_density_kg_m3: np.ndarray
    bulk_fraction: np.ndarray
    minimum_transport_depth_m: np.ndarray
    water_density_kg_m3: float
    critical_shields: float
    friction_coefficient: float
    roughness_based_friction: bool
    morphology_factor: float
    capacity_foundation_runtime_declaration: bool
    bedload_adaptation: bool
    secondary_bedload_direction: bool
    bedform_roughness: bool
    receipt_sha256: str


def _mode2_mpm_token(value: CutcellMode2MPMReceiptV96) -> str:
    return _fingerprint({
        "scope": SCOPE + ":mode2-coupled-mpm-parent-receipt",
        "execution_fingerprint": value.execution_fingerprint,
        "forcing_receipt_sha256": value.forcing_receipt_sha256,
        "runtime_receipt_sha256": value.runtime_receipt_sha256,
        "accepted_sediment_state_sha256":
            value.accepted_sediment_state_sha256,
        "hiding_owner_receipt_sha256": value.hiding_owner_receipt_sha256,
        "availability_owner_receipt_sha256":
            value.availability_owner_receipt_sha256,
        "arrays_sha256": _digest(
            value.hiding, value.mixture_d90_m, value.bed_availability,
            value.d50_m, value.sediment_density_kg_m3,
            value.bulk_fraction, value.minimum_transport_depth_m),
        "water_density_kg_m3": value.water_density_kg_m3,
        "critical_shields": value.critical_shields,
        "friction_coefficient": value.friction_coefficient,
        "roughness_based_friction": value.roughness_based_friction,
        "morphology_factor": value.morphology_factor,
        "excluded_production_branches": [
            value.capacity_foundation_runtime_declaration,
            value.bedload_adaptation,
            value.secondary_bedload_direction,
            value.bedform_roughness,
        ],
        "operator": "exact_extracted_coupled_stage_characteristic",
    })


def make_cutcell_mode2_mpm_receipt_v96(
    execution: CutcellBoundaryTraceExecutionV96,
    forcing: CutcellBoundaryTrialForcingReceiptV96, *,
    hiding: object, mixture_d90_m: object, bed_availability: object,
    d50_m: object, sediment_density_kg_m3: object,
    bulk_fraction: object, minimum_transport_depth_m: object,
    water_density_kg_m3: float, critical_shields: float,
    friction_coefficient: float, roughness_based_friction: bool,
    morphology_factor: float,
    accepted_sediment_state_sha256: str,
    hiding_owner_receipt_sha256: str,
    availability_owner_receipt_sha256: str,
    capacity_foundation_runtime_declaration: bool = False,
    bedload_adaptation: bool = False,
    secondary_bedload_direction: bool = False,
    bedform_roughness: bool = False,
) -> CutcellMode2MPMReceiptV96:
    if (type(execution) is not CutcellBoundaryTraceExecutionV96 or
            execution.fingerprint !=
                _fingerprint(_execution_payload(execution)) or
            type(forcing) is not CutcellBoundaryTrialForcingReceiptV96 or
            forcing.receipt_sha256 != _forcing_token(forcing) or
            forcing.execution_fingerprint != execution.fingerprint or
            not np.any(execution.boundary_mode == 2) or
            execution.bedload_law != 2):
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell MPM receipt binding differs")
    raw_d50 = np.asarray(d50_m)
    if raw_d50.ndim != 1 or raw_d50.size < 1:
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell MPM fraction dimensions differ")
    ng = int(raw_d50.size)
    nc = execution.cell_count
    hide = _float_array(hiding, (ng, nc), "MPM hiding", nonnegative=True)
    d90 = _float_array(
        mixture_d90_m, (nc,), "MPM mixture D90", positive=True)
    availability = _float_array(
        bed_availability, (ng, nc), "MPM bed availability",
        nonnegative=True)
    d50 = _float_array(raw_d50, (ng,), "MPM D50", positive=True)
    rhos = _float_array(
        sediment_density_kg_m3, (ng,), "MPM sediment density",
        positive=True)
    bulk = _float_array(
        bulk_fraction, (ng,), "MPM bulk fraction", positive=True)
    hmin = _float_array(
        minimum_transport_depth_m, (ng,), "MPM minimum depth",
        nonnegative=True)
    rho = float(water_density_kg_m3)
    critical = float(critical_shields)
    coefficient = float(friction_coefficient)
    morfac = float(morphology_factor)
    flags = (capacity_foundation_runtime_declaration,
             bedload_adaptation, secondary_bedload_direction,
             bedform_roughness)
    if (type(roughness_based_friction) is not bool or
            any(type(item) is not bool for item in flags) or any(flags) or
            not all(math.isfinite(item) for item in
                    (rho, critical, coefficient, morfac)) or
            rho <= 0.0 or critical < 0.0 or coefficient < 0.0 or
            morfac <= 0.0 or np.any(rhos <= rho) or
            np.any(availability > 1.0+64*np.finfo(float).eps)):
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell MPM physical/profile controls differ")
    placeholder = CutcellMode2MPMReceiptV96(
        execution.fingerprint, forcing.receipt_sha256,
        execution.runtime_receipt_sha256,
        _hex(accepted_sediment_state_sha256, "accepted sediment state"),
        _hex(hiding_owner_receipt_sha256, "hiding owner receipt"),
        _hex(availability_owner_receipt_sha256,
             "availability owner receipt"),
        hide, d90, availability,
        d50, rhos, bulk, hmin, rho, critical, coefficient,
        roughness_based_friction, morfac, *flags, "")
    return CutcellMode2MPMReceiptV96(
        **{**placeholder.__dict__,
           "receipt_sha256": _mode2_mpm_token(placeholder)})


@dataclass(frozen=True)
class CutcellMode2MPMResolverV96:
    """Accepted-parent binder for the mode-2 MPM boundary coupling."""

    execution_fingerprint: str
    runtime_receipt_sha256: str
    accepted_parent_state_sha256: str
    accepted_sediment_state_sha256: str
    hiding_owner_receipt_sha256: str
    availability_owner_receipt_sha256: str
    hiding: np.ndarray
    mixture_d90_m: np.ndarray
    bed_availability: np.ndarray
    d50_m: np.ndarray
    sediment_density_kg_m3: np.ndarray
    bulk_fraction: np.ndarray
    minimum_transport_depth_m: np.ndarray
    water_density_kg_m3: float
    critical_shields: float
    friction_coefficient: float
    roughness_based_friction: bool
    morphology_factor: float
    capacity_foundation_runtime_declaration: bool
    bedload_adaptation: bool
    secondary_bedload_direction: bool
    bedform_roughness: bool
    resolver_sha256: str

    def resolve(
        self, execution: CutcellBoundaryTraceExecutionV96,
        forcing: CutcellBoundaryTrialForcingReceiptV96, *,
        parent_state_sha256: str,
    ) -> CutcellMode2MPMReceiptV96:
        if (type(execution) is not CutcellBoundaryTraceExecutionV96 or
                execution.fingerprint != self.execution_fingerprint or
                execution.runtime_receipt_sha256 !=
                    self.runtime_receipt_sha256 or
                _hex(parent_state_sha256, "MPM accepted parent") !=
                    self.accepted_parent_state_sha256 or
                self.resolver_sha256 != _mode2_mpm_resolver_token(self)):
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell MPM resolver binding differs")
        return make_cutcell_mode2_mpm_receipt_v96(
            execution, forcing,
            hiding=self.hiding,
            mixture_d90_m=self.mixture_d90_m,
            bed_availability=self.bed_availability,
            d50_m=self.d50_m,
            sediment_density_kg_m3=self.sediment_density_kg_m3,
            bulk_fraction=self.bulk_fraction,
            minimum_transport_depth_m=self.minimum_transport_depth_m,
            water_density_kg_m3=self.water_density_kg_m3,
            critical_shields=self.critical_shields,
            friction_coefficient=self.friction_coefficient,
            roughness_based_friction=self.roughness_based_friction,
            morphology_factor=self.morphology_factor,
            accepted_sediment_state_sha256=
                self.accepted_sediment_state_sha256,
            hiding_owner_receipt_sha256=
                self.hiding_owner_receipt_sha256,
            availability_owner_receipt_sha256=
                self.availability_owner_receipt_sha256,
            capacity_foundation_runtime_declaration=
                self.capacity_foundation_runtime_declaration,
            bedload_adaptation=self.bedload_adaptation,
            secondary_bedload_direction=
                self.secondary_bedload_direction,
            bedform_roughness=self.bedform_roughness)


def _mode2_mpm_resolver_token(value: CutcellMode2MPMResolverV96) -> str:
    return _fingerprint({
        "scope": SCOPE + ":mode2-mpm-parent-resolver",
        "execution_fingerprint": value.execution_fingerprint,
        "runtime_receipt_sha256": value.runtime_receipt_sha256,
        "accepted_parent_state_sha256":
            value.accepted_parent_state_sha256,
        "accepted_sediment_state_sha256":
            value.accepted_sediment_state_sha256,
        "hiding_owner_receipt_sha256":
            value.hiding_owner_receipt_sha256,
        "availability_owner_receipt_sha256":
            value.availability_owner_receipt_sha256,
        "arrays_sha256": _digest(
            value.hiding, value.mixture_d90_m, value.bed_availability,
            value.d50_m, value.sediment_density_kg_m3,
            value.bulk_fraction, value.minimum_transport_depth_m),
        "water_density_kg_m3": value.water_density_kg_m3,
        "critical_shields": value.critical_shields,
        "friction_coefficient": value.friction_coefficient,
        "roughness_based_friction": value.roughness_based_friction,
        "morphology_factor": value.morphology_factor,
        "excluded_production_branches": [
            value.capacity_foundation_runtime_declaration,
            value.bedload_adaptation,
            value.secondary_bedload_direction,
            value.bedform_roughness,
        ],
        "operator": "exact_extracted_coupled_stage_characteristic",
    })


def resolve_cutcell_mode2_mpm_resolver_v96(
    execution: CutcellBoundaryTraceExecutionV96,
    runtime_receipt: NativeCutcellBoundaryBuildReceiptV96, *,
    accepted_parent_state_sha256: str,
    accepted_sediment_state_sha256: str,
    hiding_owner_receipt_sha256: str,
    availability_owner_receipt_sha256: str,
    hiding: object, mixture_d90_m: object, bed_availability: object,
    d50_m: object, sediment_density_kg_m3: object,
    bulk_fraction: object, minimum_transport_depth_m: object,
    water_density_kg_m3: float, critical_shields: float,
    friction_coefficient: float, roughness_based_friction: bool,
    morphology_factor: float,
    capacity_foundation_runtime_declaration: bool = False,
    bedload_adaptation: bool = False,
    secondary_bedload_direction: bool = False,
    bedform_roughness: bool = False,
) -> CutcellMode2MPMResolverV96:
    if (type(execution) is not CutcellBoundaryTraceExecutionV96 or
            execution.fingerprint != _fingerprint(
                _execution_payload(execution)) or
            type(runtime_receipt) is not
                NativeCutcellBoundaryBuildReceiptV96 or
            runtime_receipt.receipt_sha256 !=
                execution.runtime_receipt_sha256 or
            runtime_receipt.receipt_sha256 !=
                _fingerprint(_build_payload(runtime_receipt)) or
            not np.any(execution.boundary_mode == 2) or
            execution.bedload_law != 2):
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell MPM resolver execution/runtime differs")
    raw_d50 = np.asarray(d50_m)
    if raw_d50.ndim != 1 or raw_d50.size < 1:
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell MPM resolver fraction dimensions differ")
    ng, nc = int(raw_d50.size), execution.cell_count
    hide = _float_array(hiding, (ng, nc), "MPM resolver hiding",
                        nonnegative=True)
    d90 = _float_array(mixture_d90_m, (nc,), "MPM resolver D90",
                       positive=True)
    availability = _float_array(
        bed_availability, (ng, nc), "MPM resolver availability",
        nonnegative=True)
    d50 = _float_array(raw_d50, (ng,), "MPM resolver D50", positive=True)
    rhos = _float_array(
        sediment_density_kg_m3, (ng,), "MPM resolver sediment density",
        positive=True)
    bulk = _float_array(
        bulk_fraction, (ng,), "MPM resolver bulk fraction", positive=True)
    hmin = _float_array(
        minimum_transport_depth_m, (ng,), "MPM resolver minimum depth",
        nonnegative=True)
    rho = float(water_density_kg_m3)
    critical = float(critical_shields)
    coefficient = float(friction_coefficient)
    morfac = float(morphology_factor)
    flags = (capacity_foundation_runtime_declaration,
             bedload_adaptation, secondary_bedload_direction,
             bedform_roughness)
    if (type(roughness_based_friction) is not bool or
            any(type(item) is not bool for item in flags) or any(flags) or
            not all(math.isfinite(item) for item in
                    (rho, critical, coefficient, morfac)) or
            rho <= 0.0 or critical < 0.0 or coefficient < 0.0 or
            morfac <= 0.0 or np.any(rhos <= rho) or
            np.any(availability > 1.0+64*np.finfo(float).eps)):
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell MPM resolver physical/profile controls differ")
    placeholder = CutcellMode2MPMResolverV96(
        execution.fingerprint, runtime_receipt.receipt_sha256,
        _hex(accepted_parent_state_sha256, "MPM accepted parent"),
        _hex(accepted_sediment_state_sha256,
             "accepted sediment state"),
        _hex(hiding_owner_receipt_sha256, "hiding owner receipt"),
        _hex(availability_owner_receipt_sha256,
             "availability owner receipt"),
        hide, d90, availability, d50, rhos, bulk, hmin,
        rho, critical, coefficient, roughness_based_friction, morfac,
        *flags, "")
    return CutcellMode2MPMResolverV96(
        **{**placeholder.__dict__,
           "resolver_sha256": _mode2_mpm_resolver_token(placeholder)})


@dataclass(frozen=True)
class CutcellBoundaryTraceEvaluationV96:
    trace_q: np.ndarray
    physical_open_face_flux: np.ndarray
    physical_boundary_flux_owned_mask: np.ndarray
    exterior_densin_active_mask: np.ndarray
    face_signal_speed_m_s: np.ndarray
    hydraulic_face_exchange_area_m2: np.ndarray
    boundary_water_ledger_m3: np.ndarray
    diagnostics: np.ndarray
    failed_values: np.ndarray
    route: str
    reconstruction_receipt_sha256: str
    result_sha256: str


class NativeCutcellBoundaryTraceRuntimeV96:
    """Loaded allocation-free extraction with explicit and implicit entries."""

    def __init__(self, receipt: NativeCutcellBoundaryBuildReceiptV96,
                 library: ctypes.CDLL, dependency: object,
                 library_path: Path, generated_source_path: Path) -> None:
        self.receipt = receipt
        self._library = library
        self._dependency = dependency
        self.library_path = Path(library_path)
        self.generated_source_path = Path(generated_source_path)
        self._functions = {
            "explicit": library.braids_cutcell_boundary_explicit_v96,
            "implicit": library.braids_cutcell_boundary_implicit_v96,
        }
        f = np.ctypeslib.ndpointer(dtype=np.float64,
                                  flags="F_CONTIGUOUS")
        fa = np.ctypeslib.ndpointer(dtype=np.float64,
                                   flags="C_CONTIGUOUS")
        ia = np.ctypeslib.ndpointer(dtype=np.int32,
                                   flags="C_CONTIGUOUS")
        argtypes = [ctypes.c_int]*5 + [
            f, fa, f, f, ia, fa, f, fa, ia, f, fa, ia, ia, ia,
            ia, ia, ia, ia, ia, ia, fa,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double,
            f, fa, f, fa, fa, fa, fa,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_int, ctypes.c_double,
            f, f, fa, fa, ia, ia, fa, ia, fa, ia]
        for function in self._functions.values():
            function.argtypes = argtypes
            function.restype = None

    def _validate(self) -> None:
        _validate_build_receipt(self.receipt, self.library_path)
        if (_file_sha(self.generated_source_path) !=
                self.receipt.generated_source_sha256):
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell generated extraction identity differs")

    def evaluate(
        self, execution: CutcellBoundaryTraceExecutionV96,
        midpoint_q: object, bed_elevation_m: object,
        forcing: CutcellBoundaryTrialForcingReceiptV96, *,
        route: str,
        reconstruction_resolver: CutcellFaceReconstructionResolverV96 | None = None,
        mode2_mpm_receipt: CutcellMode2MPMReceiptV96 | None = None,
        evaluation_index: int = 1,
        cancelled: bool | object = False,
    ) -> CutcellBoundaryTraceEvaluationV96:
        cancel = cancelled if callable(cancelled) else lambda: cancelled
        if not callable(cancelled) and type(cancelled) is not bool:
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell cancellation must be bool/callable")
        if cancel():
            raise InterruptedError(
                "v96 cutcell boundary resolution cancelled before launch")
        if type(evaluation_index) is not int or evaluation_index < 1:
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell residual evaluation index differs")
        self._validate()
        if route not in self._functions:
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell route must be explicit or implicit")
        if (type(execution) is not CutcellBoundaryTraceExecutionV96 or
                execution.fingerprint !=
                    _fingerprint(_execution_payload(execution)) or
                execution.runtime_receipt_sha256 !=
                    self.receipt.receipt_sha256):
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell execution/runtime identity differs")
        if (type(forcing) is not CutcellBoundaryTrialForcingReceiptV96 or
                forcing.receipt_sha256 != _forcing_token(forcing) or
                forcing.execution_fingerprint != execution.fingerprint or
                forcing.accepted_state_publication_count != 0):
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell trial forcing receipt differs")
        q = _float_array(midpoint_q, (3, execution.cell_count),
                         "midpoint state")
        bed = _float_array(bed_elevation_m, (execution.cell_count,),
                           "accepted bed")
        if (forcing.bed_elevation_sha256 != _digest(bed) or
                np.any(q[0] < 0.0)):
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell forcing bed/depth identity differs")
        nc, nf, nb = (execution.cell_count, execution.face_count,
                      execution.boundary_count)
        qf = np.asfortranarray(q)
        normal = np.asfortranarray(execution.normal)
        hydraulic = np.asfortranarray(forcing.resolved_hydraulic)
        trace = np.empty((3, nf), dtype=np.float64, order="F")
        flux = np.empty((3, nf), dtype=np.float64, order="F")
        signal = np.empty(nf)
        exchange = np.empty(nf)
        physical = np.empty(nf, dtype=np.int32)
        densin = np.empty(nf, dtype=np.int32)
        ledger = np.empty(nb)
        diagnostics = np.empty(6, dtype=np.int32)
        failed = np.empty(6)
        status = np.empty(1, dtype=np.int32)
        has_mode2 = bool(np.any(execution.boundary_mode == 2))
        if has_mode2:
            if (type(reconstruction_resolver) is not
                    CutcellFaceReconstructionResolverV96 or
                    reconstruction_resolver.execution.fingerprint !=
                    execution.fingerprint or
                    reconstruction_resolver.runtime_receipt.receipt_sha256 !=
                    self.receipt.receipt_sha256):
                raise CutcellBoundaryTraceRuntimeV96Error(
                    "v96 cutcell mode-2 requires its exact reconstruction owner")
            reconstruction = reconstruction_resolver.resolve(
                q, bed, forcing, evaluation_index=evaluation_index,
                cancelled=cancelled)
            if (type(reconstruction) is not
                    CutcellFaceReconstructionReceiptV96 or
                    reconstruction.receipt_sha256 !=
                        _reconstruction_token(reconstruction) or
                    reconstruction.execution_fingerprint !=
                        execution.fingerprint or
                    reconstruction.midpoint_q_sha256 != _digest(q) or
                    reconstruction.bed_elevation_sha256 != _digest(bed) or
                    reconstruction.forcing_receipt_sha256 !=
                        forcing.receipt_sha256 or
                    reconstruction.evaluation_index != evaluation_index or
                    reconstruction.reconstruction_scheme != 1 or
                    reconstruction.resolver_call_count != 1):
                raise CutcellBoundaryTraceRuntimeV96Error(
                    "v96 cutcell reconstruction receipt differs")
            expected_q = np.ascontiguousarray(
                q[:, execution.boundary_execution.owner])
            expected_face = np.zeros((4, nf), dtype=np.float64)
            expected_face[0] = expected_q[0] + \
                bed[execution.boundary_execution.owner]
            expected_face[3] = bed[execution.boundary_execution.owner]
            expected_wet = expected_q[0] > execution.dry_tolerance_m
            expected_face[1, expected_wet] = (
                expected_q[1, expected_wet]/expected_q[0, expected_wet])
            expected_face[2, expected_wet] = (
                expected_q[2, expected_wet]/expected_q[0, expected_wet])
            if (not np.array_equal(
                    reconstruction.face_conserved_q, expected_q) or
                    not np.array_equal(
                        reconstruction.face_primitive, expected_face) or
                    not np.array_equal(
                        reconstruction.hydro_constant,
                        np.zeros(nc, dtype=np.int32))):
                raise CutcellBoundaryTraceRuntimeV96Error(
                    "v96 cutcell reconstruction numerical image differs")
            reconstructed_face = np.asfortranarray(
                reconstruction.face_primitive)
            hydro_constant = np.ascontiguousarray(
                reconstruction.hydro_constant)
            reconstruction_sha = reconstruction.receipt_sha256
        else:
            if reconstruction_resolver is not None:
                raise CutcellBoundaryTraceRuntimeV96Error(
                    "v96 cutcell unused reconstruction owner differs")
            reconstructed_face = np.zeros(
                (4, nf), dtype=np.float64, order="F")
            hydro_constant = np.zeros(nc, dtype=np.int32)
            reconstruction_sha = _fingerprint({
                "scope": SCOPE + ":no-reconstruction-required",
                "modes": execution.boundary_mode.tolist(),
            })
        has_coupled_mpm = has_mode2 and execution.bedload_law == 2
        if has_coupled_mpm:
            if (type(mode2_mpm_receipt) is not CutcellMode2MPMReceiptV96 or
                    mode2_mpm_receipt.receipt_sha256 !=
                        _mode2_mpm_token(mode2_mpm_receipt) or
                    mode2_mpm_receipt.execution_fingerprint !=
                        execution.fingerprint or
                    mode2_mpm_receipt.forcing_receipt_sha256 !=
                        forcing.receipt_sha256 or
                    mode2_mpm_receipt.runtime_receipt_sha256 !=
                        self.receipt.receipt_sha256 or
                    any((mode2_mpm_receipt.
                         capacity_foundation_runtime_declaration,
                         mode2_mpm_receipt.bedload_adaptation,
                         mode2_mpm_receipt.secondary_bedload_direction,
                         mode2_mpm_receipt.bedform_roughness))):
                raise CutcellBoundaryTraceRuntimeV96Error(
                    "v96 cutcell coupled mode-2 MPM receipt differs")
            ng = int(mode2_mpm_receipt.d50_m.size)
            hiding = np.asfortranarray(mode2_mpm_receipt.hiding)
            d90 = np.ascontiguousarray(mode2_mpm_receipt.mixture_d90_m)
            availability = np.asfortranarray(
                mode2_mpm_receipt.bed_availability)
            d50 = np.ascontiguousarray(mode2_mpm_receipt.d50_m)
            rhos = np.ascontiguousarray(
                mode2_mpm_receipt.sediment_density_kg_m3)
            bulk = np.ascontiguousarray(mode2_mpm_receipt.bulk_fraction)
            min_depth = np.ascontiguousarray(
                mode2_mpm_receipt.minimum_transport_depth_m)
            rho = mode2_mpm_receipt.water_density_kg_m3
            critical = mode2_mpm_receipt.critical_shields
            cf = mode2_mpm_receipt.friction_coefficient
            friction_mode = int(
                mode2_mpm_receipt.roughness_based_friction)
            morfac = mode2_mpm_receipt.morphology_factor
            mpm_sha = mode2_mpm_receipt.receipt_sha256
        else:
            if mode2_mpm_receipt is not None:
                raise CutcellBoundaryTraceRuntimeV96Error(
                    "v96 cutcell unused mode-2 MPM receipt differs")
            ng = 1
            hiding = np.ones((1, nc), dtype=np.float64, order="F")
            d90 = np.full(nc, 1e-3)
            availability = np.zeros((1, nc), dtype=np.float64, order="F")
            d50 = np.asarray([1e-3])
            rhos = np.asarray([2650.0])
            bulk = np.asarray([0.6])
            min_depth = np.asarray([0.0])
            rho, critical, cf, friction_mode, morfac = (
                1000.0, 0.047, 0.0, 0, 1.0)
            mpm_sha = _fingerprint({
                "scope": SCOPE + ":coupled-mpm-not-applicable",
                "bedload_law": execution.bedload_law,
                "has_mode2": has_mode2,
            })
        self._functions[route](
            nc, nf, nb, execution.boundary_faces.size, ng,
            qf, np.ascontiguousarray(bed), hydraulic,
            reconstructed_face, hydro_constant,
            np.ascontiguousarray(execution.manning_cell),
            np.asfortranarray(execution.face_midpoint_m),
            np.ascontiguousarray(execution.fixed_boundary_bed_m),
            np.ascontiguousarray(execution.boundary_execution.owner), normal,
            np.ascontiguousarray(execution.face_length_m),
            np.ascontiguousarray(
                execution.boundary_execution.boundary_index),
            np.ascontiguousarray(execution.boundary_offset),
            np.ascontiguousarray(execution.boundary_faces),
            np.ascontiguousarray(execution.boundary_mode),
            np.ascontiguousarray(execution.fixed_boundary_bed_mode),
            np.ascontiguousarray(execution.viscosity_equilibrated_inlet),
            np.ascontiguousarray(execution.reversal_policy),
            np.ascontiguousarray(execution.boundary_seed_position),
            np.ascontiguousarray(execution.sampled_boundary_bed),
            np.ascontiguousarray(execution.boundary_bed_offset_m),
            execution.hydraulic_roughness_mode,
            int(execution.white_colebrook_chzk2d),
            execution.bedload_law, 1,
            execution.roughness_minimum_work_depth_m,
            execution.horizontal_viscosity_m2_s,
            execution.dry_tolerance_m,
            execution.manning_friction_chezy_minimum_depth_m,
            execution.speed_limit_m_s,
            execution.gravity_m_s2, forcing.dt_s,
            hiding, d90, availability, d50, rhos, bulk, min_depth,
            rho, critical, cf, friction_mode, morfac,
            trace, flux, signal, exchange, physical, densin, ledger,
            diagnostics, failed, status)
        if cancel():
            raise InterruptedError(
                "v96 cutcell boundary resolution cancelled after launch")
        if status[0] != 0:
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell exact boundary branch rejected input: " +
                json.dumps({"status": int(status[0]),
                            "diagnostics": diagnostics.tolist(),
                            "failed_values": failed.tolist()}))
        values = tuple(np.array(item, copy=True, order="C") for item in
                       (trace, flux, physical, densin, signal, exchange,
                        ledger, diagnostics, failed))
        if (not all(np.isfinite(item).all() for item in values) or
                not np.array_equal(values[2].astype(bool),
                                   execution.boundary_execution.open_mask) or
                np.any(values[3].astype(bool) & ~values[2].astype(bool)) or
                np.any(values[0][0, values[2].astype(bool)] < 0.0)):
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell native output contract differs")
        for item in values:
            item.setflags(write=False)
        result_sha = _fingerprint({
            "scope": SCOPE + ":evaluation",
            "execution": execution.fingerprint,
            "forcing": forcing.receipt_sha256,
            "route": route,
            "evaluation_index": evaluation_index,
            "runtime": self.receipt.receipt_sha256,
            "reconstruction": reconstruction_sha,
            "mode2_mpm": mpm_sha,
            "input_sha256": _digest(q, bed),
            "output_sha256": _digest(*values),
        })
        return CutcellBoundaryTraceEvaluationV96(
            values[0], values[1], values[2], values[3], values[4],
            values[5], values[6], values[7], values[8], route,
            reconstruction_sha, result_sha)


def build_cutcell_boundary_trace_runtime_v96(
    repository_root: str | Path, build_directory: str | Path, *,
    compiler: str | Path | None = None,
) -> NativeCutcellBoundaryTraceRuntimeV96:
    root = Path(repository_root).resolve()
    parent = root / "solver/cut_cells/cutcell_solver.f90"
    if parent.resolve() != _PARENT or _file_sha(parent) != FROZEN_PARENT_SHA256:
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell frozen parent source identity differs")
    original = parent.read_bytes()
    generated, pieces = _assembled_source(original)
    build_root = Path(build_directory).resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    chosen = Path(compiler).resolve() if compiler is not None else None
    if chosen is None:
        found = shutil.which("gfortran")
        if found is None and os.name == "nt":
            fallback = Path(r"C:\Strawberry\c\bin\gfortran.exe")
            found = str(fallback) if fallback.is_file() else None
        if found is None:
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell extraction requires gfortran")
        chosen = Path(found).resolve()
    flags = ("-O3", "-fopenmp", "-shared", "-ffree-line-length-none",
             "-static-libgfortran", "-static-libgcc")
    folder = Path(tempfile.mkdtemp(prefix="cutcell_trace_v96_",
                                   dir=build_root))
    source_path = folder / "cutcell_boundary_trace_extracted_v96.f90"
    source_path.write_bytes(generated)
    name = ("cutcell_boundary_trace_v96.dll" if os.name == "nt" else
            "libcutcell_boundary_trace_v96.so")
    partial, library_path = folder/(name+".partial"), folder/name
    dependency_sources = [str(root / relative) for relative in _DEPENDENCIES]
    command = [str(chosen), *flags, *dependency_sources,
               str(source_path), "-o", str(partial)]
    completed = subprocess.run(command, cwd=folder, capture_output=True,
                               text=True, timeout=180, check=False)
    (folder/"build.log").write_text(
        completed.stdout+completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not partial.is_file():
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell extraction build failed:\n"+
            completed.stdout+completed.stderr)
    if (_file_sha(parent) != FROZEN_PARENT_SHA256 or
            parent.read_bytes() != original or
            _assembled_source(parent.read_bytes())[0] != generated):
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell parent changed during extraction build")
    os.replace(partial, library_path)
    version = subprocess.run(
        [str(chosen), "--version"], capture_output=True, text=True,
        check=True, timeout=30).stdout.splitlines()[0]
    fragment_identity = tuple(
        (name, int(row["start_byte"]), int(row["end_byte"]),
         str(row["sha256"])) for name, row in sorted(pieces.items()))
    dependency_identity = tuple(
        (relative, _file_sha(root / relative)) for relative in _DEPENDENCIES)
    if dict(dependency_identity) != _FROZEN_DEPENDENCY_SHA256:
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell frozen helper source identity differs")
    placeholder = NativeCutcellBoundaryBuildReceiptV96(
        ABI, SCOPE, FROZEN_PARENT_SHA256, _file_sha(_WRAPPER),
        _file_sha(source_path), _file_sha(library_path), _file_sha(chosen),
        version, flags, fragment_identity, dependency_identity,
        False, 0, 0, "")
    receipt = NativeCutcellBoundaryBuildReceiptV96(
        **{**placeholder.__dict__,
           "receipt_sha256": _fingerprint(_build_payload(placeholder))})
    dependency = (os.add_dll_directory(str(chosen.parent))
                  if os.name == "nt" and hasattr(os, "add_dll_directory")
                  else None)
    runtime = NativeCutcellBoundaryTraceRuntimeV96(
        receipt, ctypes.CDLL(str(library_path)), dependency,
        library_path, source_path)
    runtime._validate()
    return runtime


def make_cutcell_boundary_trace_provider_v96(
    runtime: NativeCutcellBoundaryTraceRuntimeV96,
    execution: CutcellBoundaryTraceExecutionV96, *,
    trial_forcing_receipt: CutcellBoundaryTrialForcingReceiptV96 | None = None,
    trial_forcing_resolver: CutcellBoundaryTrialForcingResolverV96 | None = None,
    reconstruction_resolver: CutcellFaceReconstructionResolverV96 | None = None,
    mode2_mpm_receipt: CutcellMode2MPMReceiptV96 | None = None,
    mode2_mpm_resolver: CutcellMode2MPMResolverV96 | None = None,
) -> HydraulicBoundaryTraceProviderV96:
    """Adapt the exact implicit route to the frozen generic typed handoff."""
    if type(runtime) is not NativeCutcellBoundaryTraceRuntimeV96:
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell runtime type differs")
    runtime._validate()
    static_forcing = type(trial_forcing_receipt) is \
        CutcellBoundaryTrialForcingReceiptV96
    dynamic_forcing = type(trial_forcing_resolver) is \
        CutcellBoundaryTrialForcingResolverV96
    if (type(execution) is not CutcellBoundaryTraceExecutionV96 or
            execution.fingerprint !=
                _fingerprint(_execution_payload(execution)) or
            static_forcing == dynamic_forcing or
            (static_forcing and
             trial_forcing_receipt.receipt_sha256 !=
                _forcing_token(trial_forcing_receipt)) or
            (dynamic_forcing and
             (trial_forcing_resolver.execution_fingerprint !=
                  execution.fingerprint or
              trial_forcing_resolver.runtime_receipt_sha256 !=
                  runtime.receipt.receipt_sha256 or
              trial_forcing_resolver.resolver_sha256 !=
                  _forcing_resolver_token(trial_forcing_resolver)))):
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell provider binding differs")
    has_mode2 = bool(np.any(execution.boundary_mode == 2))
    if ((has_mode2 and
         type(reconstruction_resolver) is not
         CutcellFaceReconstructionResolverV96) or
            (not has_mode2 and reconstruction_resolver is not None)):
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell provider reconstruction binding differs")
    requires_mpm = has_mode2 and execution.bedload_law == 2
    static_mpm = type(mode2_mpm_receipt) is CutcellMode2MPMReceiptV96
    dynamic_mpm = type(mode2_mpm_resolver) is CutcellMode2MPMResolverV96
    if ((requires_mpm and static_mpm == dynamic_mpm) or
            (not requires_mpm and
             (mode2_mpm_receipt is not None or
              mode2_mpm_resolver is not None)) or
            (static_mpm and dynamic_forcing) or
            (dynamic_mpm and not dynamic_forcing) or
            (dynamic_mpm and
             (mode2_mpm_resolver.execution_fingerprint !=
                  execution.fingerprint or
              mode2_mpm_resolver.runtime_receipt_sha256 !=
                  runtime.receipt.receipt_sha256 or
              mode2_mpm_resolver.resolver_sha256 !=
                  _mode2_mpm_resolver_token(mode2_mpm_resolver)))):
        raise CutcellBoundaryTraceRuntimeV96Error(
            "v96 cutcell provider MPM binding differs")
    stable_state = {
        "scope": SCOPE + ":transaction-local-owner",
        "runtime_receipt_sha256": runtime.receipt.receipt_sha256,
        "execution_fingerprint": execution.fingerprint,
        "trial_forcing_receipt_sha256": (
            trial_forcing_receipt.receipt_sha256
            if static_forcing else None),
        "trial_forcing_resolver_sha256": (
            trial_forcing_resolver.resolver_sha256
            if dynamic_forcing else None),
        "reconstruction_runtime_sha256": (
            reconstruction_resolver.runtime_sha256
            if reconstruction_resolver is not None else None),
        "mode2_mpm_receipt_sha256": (
            mode2_mpm_receipt.receipt_sha256
            if mode2_mpm_receipt is not None else None),
        "mode2_mpm_resolver_sha256": (
            mode2_mpm_resolver.resolver_sha256
            if mode2_mpm_resolver is not None else None),
        "accepted_state_publication_count": 0,
    }

    def resolver(*, midpoint_q, bed_elevation_m, evaluation_index,
                 evaluation_time_s, hydraulic_request_sha256,
                 parent_state_sha256, evaluation_dt_s,
                 execution: HydraulicBoundaryTraceExecutionV96) -> Mapping[str, object]:
        if execution.fingerprint != \
                execution_outer.boundary_execution.fingerprint:
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell provider request/clock differs")
        forcing = trial_forcing_receipt
        if dynamic_forcing:
            forcing = trial_forcing_resolver.resolve(
                execution_outer,
                hydraulic_request_sha256=hydraulic_request_sha256,
                parent_state_sha256=parent_state_sha256,
                bed_elevation_m=bed_elevation_m,
                evaluation_time_s=evaluation_time_s,
                evaluation_dt_s=evaluation_dt_s)
        elif (hydraulic_request_sha256 !=
                  trial_forcing_receipt.hydraulic_request_sha256 or
              parent_state_sha256 !=
                  trial_forcing_receipt.parent_state_sha256 or
              float(evaluation_time_s) !=
                  trial_forcing_receipt.evaluation_time_s or
              evaluation_dt_s is None or
              float(evaluation_dt_s) != trial_forcing_receipt.dt_s):
            raise CutcellBoundaryTraceRuntimeV96Error(
                "v96 cutcell provider request/parent/clock differs")
        mpm = mode2_mpm_receipt
        if dynamic_mpm:
            mpm = mode2_mpm_resolver.resolve(
                execution_outer, forcing,
                parent_state_sha256=parent_state_sha256)
        result = runtime.evaluate(
            execution_outer, midpoint_q, bed_elevation_m,
            forcing, route="implicit",
            reconstruction_resolver=reconstruction_resolver,
            mode2_mpm_receipt=mpm,
            evaluation_index=evaluation_index)
        reconstruction = result.reconstruction_receipt_sha256
        branch = _fingerprint({
            "scope": SCOPE + ":exact-modes6789-branch",
            "fragment_identity": runtime.receipt.fragment_identity,
            "evaluation_index": evaluation_index,
            "result_sha256": result.result_sha256,
        })
        owner_evaluation = _fingerprint({
            "scope": SCOPE + ":owner-evaluation",
            "runtime": runtime.receipt.receipt_sha256,
            "execution": execution_outer.fingerprint,
            "forcing": forcing.receipt_sha256,
            "evaluation_index": evaluation_index,
            "result": result.result_sha256,
        })
        return {
            "trace_q": result.trace_q,
            "physical_open_face_flux": result.physical_open_face_flux,
            "physical_boundary_flux_owned_mask":
                result.physical_boundary_flux_owned_mask,
            "exterior_densin_active_mask":
                result.exterior_densin_active_mask,
            "reconstruction_receipt_sha256": reconstruction,
            "boundary_branch_receipt_sha256": branch,
            "owner_evaluation_sha256": owner_evaluation,
        }

    execution_outer = execution
    return HydraulicBoundaryTraceProviderV96(
        execution.boundary_execution, resolver=resolver,
        state_probe=lambda: stable_state)


__all__ = [
    "ABI", "FROZEN_PARENT_SHA256", "SCOPE",
    "CutcellFaceReconstructionReceiptV96",
    "CutcellFaceReconstructionResolverV96",
    "CutcellMode2MPMReceiptV96",
    "CutcellMode2MPMResolverV96",
    "CutcellBoundaryTrialForcingResolverV96",
    "CutcellBoundaryTraceExecutionV96",
    "CutcellBoundaryTraceEvaluationV96",
    "CutcellBoundaryTrialForcingReceiptV96",
    "CutcellBoundaryTraceRuntimeV96Error",
    "NativeCutcellBoundaryBuildReceiptV96",
    "NativeCutcellBoundaryTraceRuntimeV96",
    "build_cutcell_boundary_trace_runtime_v96",
    "make_cutcell_boundary_trace_provider_v96",
    "make_cutcell_boundary_trial_forcing_receipt_v96",
    "make_cutcell_mode2_mpm_receipt_v96",
    "resolve_cutcell_mode2_mpm_resolver_v96",
    "resolve_cutcell_boundary_trial_forcing_resolver_v96",
    "resolve_cutcell_boundary_trace_execution_v96",
]
