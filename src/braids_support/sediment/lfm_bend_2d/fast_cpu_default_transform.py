"""Assemble guarded CPU defaults from the qualified, frozen integrators.

This module does not edit production or a runtime. ``transform`` returns one
self-contained Fortran source for the caller to freeze, build and qualify.
Legacy mode retains the old SSP33 routines and every original arithmetic path;
the automatic selector uses SSPMS6 only for its hydraulic-only scope, SSP43
for qualified fixed-mesh MPM coupling, and SSP33 for everything else.
"""
from . import larger_timesteps as p
from . import explicit_multistep_transform as ms

HERE = p.r.HERE

DECLARATIONS = """ ! Guarded CPU time integration; input CFL is never overridden here.
 integer :: cpu_time_integrator=33,cpu_outlet_extension=0
 character(len=64) :: cpu_time_integrator_requested='auto'
 logical :: cpu_multistep_eligible=.false.,cpu_ssp43_eligible=.false.
""" + ms.DECLARATIONS.replace(
    " integer :: explicit_history_method=0,implicit_outlet_extension=1",
    " integer :: explicit_history_method=0",
)


def _rename(text):
    replacements = {
        "research_explicit_advance": "advance_ssp43",
        "research_history_advance": "advance_multistep6",
        "research_history_step": "cpu_history_step",
        "research_ssp43": "cpu_ssp43",
        "research_prepare_stage": "cpu_prepare_stage",
        "research_momentum": "cpu_momentum",
        "implicit_outlet_bounds": "cpu_outlet_bounds",
        "implicit_outlet_extension": "cpu_outlet_extension",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _ssp43(original, name):
    """Change only the SSP stages and their accepted quadrature weights."""
    value = p.methods.routine(original, name)
    value = p.replace(value, "subroutine " + name + "(",
                      "subroutine " + name + "_ssp43(")
    if name == "advance":
        value = p.replace(value, "call advance_prognostic_secondary(dt,ok)",
                          "call advance_prognostic_secondary_ssp43(dt,ok)")
        value = p.replace(value, "  integer :: target",
                          "  real(8) :: cpu_momentum(2,2)\n  integer :: target")
        initial, momentum = "q", "cpu_momentum"
    else:
        initial, momentum = "a", "momentum_a"
    start = value.index("  step_morfac=morphological_factor_at(time)")
    end = value.index("  call rebalance_active_layer(c)", start)
    value = value[:start] + f"""  step_morfac=morphological_factor_at(time)
  call cpu_ssp43({initial},dt,step_morfac,c,fa,{momentum},ok)
  if(.not.ok) return
""" + value[end:]
    value = p.replace(value, "ledger=ledger+(fa+fb)/6.d0+2.d0*fc/3.d0",
                      "ledger=ledger+fa")
    if name == "advance_prognostic_secondary":
        value = p.replace(value,
            "secondary_momentum_boundary_trial=(momentum_a+momentum_b)/6.d0+2.d0*momentum_c/3.d0",
            "secondary_momentum_boundary_trial=momentum_a")
    return value


def _multistep_advance(ssp43):
    """Adapt the already-qualified transactional hydraulic history wrapper."""
    value = p.replace(ssp43, "subroutine advance_ssp43(",
                      "subroutine advance_multistep6(")
    value = p.replace(value, "  integer :: target",
                      "  integer :: target\n  logical :: used_history")
    value = p.replace(value, "  call prepare_hydraulic_stage(q)",
        "  !$omp parallel do schedule(static,256) default(shared) private(target)\n"
        "  do target=1,nc\n"
        "   a(:,target)=q(:,target)\n"
        "  enddo\n"
        "  !$omp end parallel do\n"
        "  call prepare_hydraulic_stage(a)")
    value = p.replace(value,
        "  call cpu_ssp43(q,dt,step_morfac,c,fa,cpu_momentum,ok)",
        "  call cpu_history_step(a,dt,step_morfac,c,fa,cpu_momentum,used_history,ok)")
    value = p.replace(value, "  call source(c,dt)",
                      "  if(.not.used_history) call source(c,dt)")
    value = p.replace(value,
        "  if(profile_enabled) then\n   profile_source_s=profile_source_s+omp_get_wtime()-profile_start;profile_source_calls=profile_source_calls+1",
        "  if(profile_enabled.and..not.used_history) then\n   profile_source_s=profile_source_s+omp_get_wtime()-profile_start;profile_source_calls=profile_source_calls+1")
    value = p.replace(value,
        "  !$omp parallel do schedule(static,256) if(nc*nstate>=131072) default(shared) private(target)",
        "  call explicit_history_commit(a,ledger,time,dt,used_history)\n"
        "  !$omp parallel do schedule(static,256) default(shared) private(target)")
    return value


def transform(original):
    """Return guarded auto/legacy/ssp43/multistep6 integration source.

    No MF guard, deck setting, source closure, diffusion routine, augmented
    Riemann wrapper or sediment-reconstruction formula is changed here.
    The caller owns checkpoint-method metadata and production installation.
    """
    if "cpu_time_integrator_requested" in original:
        raise ValueError("CPU integration transform is already present")
    text = p.replace(original, " implicit none\n integer :: nc,nf",
                      " implicit none\n" + DECLARATIONS + " integer :: nc,nf")
    routines = []
    for name in ("advance", "advance_prognostic_secondary"):
        old = p.methods.routine(text, name)
        legacy = p.replace(old, "subroutine " + name + "(",
                           "subroutine " + name + "_ssp33(")
        if name == "advance":
            legacy = p.replace(legacy, "call advance_prognostic_secondary(dt,ok)",
                               "call advance_prognostic_secondary_ssp33(dt,ok)")
        candidate = _ssp43(original, name)
        text = p.replace(text, old, legacy)
        routines.append(candidate)
        # Keep the exact legacy arithmetic, including original stage preparation.
        restored = legacy.replace("subroutine " + name + "_ssp33(",
                                  "subroutine " + name + "(", 1)
        if name == "advance":
            restored = restored.replace("call advance_prognostic_secondary_ssp33(dt,ok)",
                                        "call advance_prognostic_secondary(dt,ok)", 1)
        assert restored == old
    routines.append(_multistep_advance(routines[0]))
    helpers = _rename((HERE / "native/ssp43_step.inc").read_text())
    helpers = helpers.replace("! Research-only SSPRK(4,3). Not part of the production build.",
                              "! Qualified fixed-mesh SSPRK(4,3) transport update.")
    history = _rename((HERE / "native/explicit_multistep.inc").read_text())
    # Its research dispatcher is replaced by the scoped production dispatcher.
    history = p.replace(history, p.methods.routine(history, "advance"), "")
    history = history.replace("! Validation-only variable-step SSP linear multistep method, order two.",
                              "! Qualified hydraulic-only variable-step SSP multistep method, order two.")
    outlet = _rename((HERE / "native/implicit_outlet_bounds.inc").read_text())
    outlet = outlet.replace("! Research-only boundary reconstruction repair.",
                            "! Qualified wet, subcritical stage-outlet bound extension.")
    helpers += "\n" + history + "\n" + outlet
    helpers += "\n" + (HERE / "native/fast_cpu_integrators.inc").read_text()
    text = p.replace(text, " subroutine profile_report()",
                      "\n\n".join(routines) + "\n" + helpers + "\n subroutine profile_report()")
    for name in ("gradients", "prepare_hydraulic_traces_fused"):
        before = p.methods.routine(text, name)
        after = p.replace(before, "   theta=1",
                           "   call cpu_outlet_bounds(c,v,grad,vlo,vhi)\n   theta=1")
        text = p.replace(text, before, after)
    marker = " ! Qualified same-equation fixed-grid hydraulic workflow. Resolve after restart"
    text = p.replace(text, marker,
        " ! Resolve the guarded integrator only after input/restart state loading.\n"
        " call resolve_cpu_time_integrator(len_trim(restart)>0)\n" + marker)
    before = p.methods.routine(text, "profile_report")
    report = "\n  write(*,'(a,i0)') 'time integrator: active method=',cpu_time_integrator\n"
    report += ms.REPORT
    text = p.replace(text, before, p.replace(before, "  if(.not.profile_enabled) return",
                                             "  if(.not.profile_enabled) return" + report))
    import re
    changed = {"advance", "advance_prognostic_secondary", "gradients",
               "prepare_hydraulic_traces_fused", "profile_report"}
    for name in re.findall(r"(?im)^ *(?:subroutine|function) (\w+)\(", original):
        if name not in changed:
            assert p.methods.routine(text, name) == p.methods.routine(original, name), name
    return text
