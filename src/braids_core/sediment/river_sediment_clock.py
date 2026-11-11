"""Accepted-state outer-step estimate; never replaces an acceptance guard.

The Exner transaction still enforces its 5% relative bed increment and exact
budgets. This scalar controller avoids proposing the same oversized step again.
It is independent of the hydraulic CFL/substep controller and of the backend.
"""
import math

METHOD = 'accepted_sediment_failure_cap_v2'
LEGACY_METHOD = 'accepted_sediment_step_hint_v1'
LEGACY_SHA256 = '78c8d04c9e0e19e49826e74bbdd1af4628b0b81f0d71e98464ef3396c6b969ab'


class SedimentHint(float):
    """A numeric timestep hint with an optional recent-failure ceiling."""
    def __new__(cls, value, failure_cap=None):
        obj = super().__new__(cls, value)
        obj.failure_cap = failure_cap
        return obj

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        # CuPy does not otherwise accept float subclasses as scalar operands.
        return getattr(ufunc, method)(*(float(x) if isinstance(x, SedimentHint) else x
                                        for x in args), **kwargs)


def restore_hint(value, failure_cap=None):
    if value is None:
        if failure_cap is not None: raise ValueError('Failure cap needs a timestep hint')
        return None
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError('Positive finite sediment step hint required')
    if failure_cap is not None and (isinstance(failure_cap, bool) or
            not math.isfinite(failure_cap) or failure_cap <= 0 or value > failure_cap):
        raise ValueError('Invalid sediment failure cap')
    return SedimentHint(value, failure_cap)


def next_hint(previous, accepted_dt, maximum_dt, relative_change, rejected):
    """Aim below the existing 0.05 guard, recovering by at most 25% per step.

    Output/forcing/AMR clocks may truncate a proposal. Do not treat that shorter
    interval as evidence of a smaller stability limit. A rejection, in contrast,
    resets the growth base to the step that actually passed all guards.
    """
    if not all(math.isfinite(v) and v > 0 for v in (accepted_dt, maximum_dt)):
        raise ValueError('Positive finite sediment clock intervals required')
    if not math.isfinite(relative_change) or relative_change < 0:
        raise ValueError('Finite nonnegative accepted bed increment required')
    if previous is not None and (not math.isfinite(previous) or previous <= 0):
        raise ValueError('Positive finite sediment step hint required')
    base = accepted_dt if rejected else max(accepted_dt, previous or maximum_dt)
    estimate = accepted_dt * .04 / relative_change if relative_change > 0 else maximum_dt
    cap = getattr(previous, 'failure_cap', None)
    if rejected:
        cap = min(cap if cap is not None else maximum_dt, 1.7 * accepted_dt)
    elif cap is not None:
        cap = min(maximum_dt, 1.015 * cap)
        if cap >= maximum_dt:
            cap = None
    result = min(maximum_dt, 1.25 * base, estimate,
                 cap if cap is not None else maximum_dt)
    return SedimentHint(result, cap)
