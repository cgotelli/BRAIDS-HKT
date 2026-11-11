"""Use a failed bed-increment measurement only to propose the next trial.

No acceptance criterion is changed. The new proposal is evaluated afresh,
including hydraulics, stock limiting, bed movement and all budgets.
"""
import math

def retry_scale(error):
    change=getattr(error,'relative_bed_increment',None)
    if change is None or not math.isfinite(change) or change<=.05:
        return .5
    # Skip only clearly oversized bisection candidates. Retaining the binary
    # timestep ladder avoids a small arbitrary proposal depressing the
    # accepted-state failure cap for many subsequent steps. This estimate is
    # deliberately rounded UP; its fresh trial may still need another retry.
    if change>=.4:return .125 # also avoid overflow in change/.05
    exponent=max(1,math.floor(math.log2(change/.05)))
    return .5**exponent
