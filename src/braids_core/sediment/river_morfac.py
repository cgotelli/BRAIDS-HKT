"""Named bed-clock acceleration for ordinary Exner + suspended transport.

This is not a rescaling of the hydraulic or suspended-material clock.  If a
wet cell exchanges physical mass dm with suspension, its bed receives -MF*dm.
The existing acceleration ledger stores the difference from physical mass
exchange. Dry stranding is a one-off transfer and is deliberately unscaled.
Jiang--Cao's coupled phase equations are outside this contract.
"""
METHOD = 'morphology.bed_clock_morfac_v1'
DESCRIPTION = (
    'MF scales bedload divergence and wet bed/suspension exchange; water, '
    'suspended advection and concentration relaxation use hydraulic time. '
    'Erosion is limited by available bed mass / MF; dry stranding is unscaled. '
    'The extra bed mass is recorded in the acceleration ledger, not hidden '
    'as physical sediment supply. MF=1 gives ordinary physical exchange.'
)
LIMITATION = (
    'MF above 1 approximates slow morphology under repeated or near-steady '
    'forcing. It is not exact time compression of an unsteady flood or finite '
    'sediment pulse. Check MF sensitivity for the intended case; larger MF '
    'may require smaller timesteps. Jiang-Cao remains MF1 only.'
)


def physical_stock(xp, available, bedload_delta, mf):
    """Reserve already committed trial bedload before physical entrainment."""
    stock = xp.maximum(available + bedload_delta, 0.)
    if mf == 1.:
        return stock
    physical = stock / mf
    # The source kernels reserve 16 relative ULPs. That reserve rounds to
    # zero in the subnormal range, where division by MF can round UP and the
    # later MF multiplication can overdraw an exhausted bed. Round the
    # subnormal quotient toward zero by one representable number instead.
    # No physical stock threshold, mass clipping or inventory tolerance.
    return xp.where(physical < 2.2250738585072014e-308,
                    xp.nextafter(physical, xp.zeros_like(physical)), physical)


def accelerate_wet_exchange(xp, exchange, depth, dry, mf):
    """In-place adapter for MF1 source kernels, on the selected backend."""
    if mf != 1.:
        exchange *= xp.where(depth > dry, mf, 1.)
