"""Accepted-state outer-step estimate; never replaces an acceptance guard.

The Exner transaction still enforces its 5% relative bed increment and exact
budgets. This scalar controller avoids proposing the same oversized step again.
It is independent of the hydraulic CFL/substep controller and of the backend.
"""
import math

METHOD = 'accepted_sediment_step_hint_v1'


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
    return min(maximum_dt, 1.25 * base, estimate)
