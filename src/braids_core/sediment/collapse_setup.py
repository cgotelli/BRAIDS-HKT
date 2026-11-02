"""One compatibility contract for the two Desktop bed-redistribution owners.

Collapse transfers existing bed inventory, never a second entrainment source.
The ordinary threshold/equalization rule is NOT the JC repose projection.
"""
from braids_core.river_model import ModelError

ORDINARY = 'redistribution.braids_conservative_wet_dry_avalanche'
JC = 'redistribution.jiang_cao_repose200_v1'
EXNER = 'redistribution.jiang_cao_repose_exner223_v1'
JC_CAPACITY = 'capacity.mpm_jc2026_p15_reconstruction_v1'


def resolve(method, capacity):
    return EXNER if method == JC and capacity != JC_CAPACITY else method


def validate(method, capacity, suspension='off', *, mf=1., banks='off'):
    def fail(message):
        raise ModelError('incompatible_physics', 'sediment.avalanche', message)
    if method == 'off':
        return
    if method not in (ORDINARY, JC, EXNER):
        # The legacy FLOW wet-slope flux has its own admission; it is not one
        # of the two Desktop collapse options and is not admitted to suspension.
        if suspension != 'off':
            fail('Select BRAIDS conservative wet/dry collapse for independent suspension. The FLOW wet-slope flux is not qualified with suspension.')
        return
    if capacity == JC_CAPACITY:
        if method != JC or suspension != 'off' or mf != 1.:
            fail('The Jiang-Cao bundle requires its own repose projection, MF=1 and no independent suspension. Choose that projection or turn collapse off.')
    if capacity == JC_CAPACITY and banks != 'off':
        fail('The coupled Jiang-Cao bundle cannot use ordinary Exner bank supply. Turn bank erosion off.')


def summary(method, *, slope=None, angle=None):
    if method == 'off':
        return 'Collapse: OFF.'
    if method == JC:
        return ('Collapse: Jiang-Cao repose projection (bundle only, MF=1). '
                'Projects internal bed slopes to the bundle repose criterion; '
                'moves bed solid and pore water. No extra suspended source.')
    if method == EXNER:
        return ('Collapse: Jiang-Cao repose criterion with ordinary Exner inventory.' +
                ('' if angle is None else f' Repose angle: {angle:g} degrees.') +
                ' Iterative downhill shared-face redistribution of finite bed stock; no transfer through exterior boundaries. '
                'Preserves water volume; no extra suspended source and no MF multiplier. '
                'Immobile substrate may retain a supercritical slope. No inferred angle from grain diameter; '
                'not a kinetic or geotechnical block-failure model. Bank/toe supply precedes this redistribution, using the same remaining stock.')
    if method == ORDINARY:
        trigger = '' if slope is None else f' Trigger slope: {slope:g} m/m.'
        return ('Collapse: BRAIDS conservative wet/dry redistribution.' + trigger +
                ' Over-steep neighbour pairs redistribute finite bed stock toward equal elevation. '
                'This is a threshold/equalization rule, not a projection to the repose angle. '
                'One pass follows bedload and suspended exchange each accepted step; '
                'MF does not multiply collapse. No transfer through exterior walls or open boundaries; '
                'no direct injection into suspension. Water volume is unchanged. '
                'No geotechnical block failure or undercutting; compare timesteps when collapse is active.')
    return 'Redistribution: ' + method
