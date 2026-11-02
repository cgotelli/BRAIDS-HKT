"""Floating-point allowance for a cumulative, separately step-checked budget."""
import math
import numbers
import sys


def cumulative_water_roundoff(initial_volume, current_volume, absolute_boundary_volume, accepted_steps):
    """Accumulate the public 64-epsilon per-increment arithmetic allowance.

    This does not replace the local conservative trial check. That check rejects
    each update independently; the worker must also allow rounding accumulated
    by previously accepted updates. A single-reduction allowance is not a bound
    for an arbitrarily long sequence of state/ledger additions.

    Inventory/throughput scale is used rather than the small net inlet-outlet
    difference. One extra operation budget covers the final reductions. The
    factor is the standard gamma_n = n*eps/(1-n*eps), with 64 operations per
    accepted outer update, consistent with the existing local check.
    """
    values=(float(initial_volume),float(current_volume),float(absolute_boundary_volume))
    if not all(math.isfinite(v) and v>=0 for v in values):
        raise ValueError('Finite nonnegative water inventory and throughput required')
    if isinstance(accepted_steps,bool) or not isinstance(accepted_steps,numbers.Integral) or accepted_steps<0:
        raise ValueError('Nonnegative integer accepted step count required')
    product=64*(int(accepted_steps)+1)*sys.float_info.epsilon
    if product>=1:raise ValueError('Water arithmetic operation count exceeds finite error bound')
    return max(1e-9,product/(1-product)*max(values))
