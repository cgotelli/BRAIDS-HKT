"""Shared admission for accepted-demand bank supply (Desktop milestone224).

Neither sharing nor surveyed supply creates an entrainment capacity. Both use
the net accepted bed change, after suspended exchange, exactly once.
"""
from braids_core.river_model import ModelError, SURVEY_BANKS
from .collapse_setup import JC_CAPACITY

SHARING = 'redistribution.dry_bank'
METHODS = (SHARING, *SURVEY_BANKS)


def validate_inventory(columns):
    if (len(columns['fraction_ids']) != 1 or columns['bed'] != 'mixed'
            or columns['porosity'] != 0 or columns['base_update'] != 1):
        raise ModelError('integration_pending', 'settings.columns',
            'Desktop bank supply currently requires one bulk material, a fully mixed finite bed '
            'and dry-bulk-density packing. Mixture/layer bank coupling is not qualified here.')


def validate(method, capacity, suspension='off', *, bedload=True):
    if method == 'off':
        return
    def fail(message):
        raise ModelError('incompatible_physics', 'sediment.banks', message)
    if method not in METHODS:
        fail('Choose local dry-bank sharing or a surveyed toe-supply method.')
    if capacity == JC_CAPACITY:
        raise ModelError('integration_pending', 'sediment.banks',
            'The coupled Jiang–Cao mobile-bed/water inventory needs a separately qualified bank-supply coupling. '
            'Turn bank erosion off; its own repose collapse remains available.')
    if not bedload and suspension in (None, 'off'):
        fail('Bank supply requires accepted bed erosion. Enable bedload or suspended transport, or turn bank erosion off. Collapse can operate independently.')


def summary(method):
    if method == 'off':
        return 'Bank erosion: OFF.'
    common = (' Uses accepted NET bed erosion after bedload and suspended exchange; '
              'deposition alone requests no bank supply. Finite bed stock, internal faces only. '
              'MF is already included in bed demand and is not applied again. '
              'Transferred material enters the bed, not suspension. Either collapse option may follow. ')
    if method == SHARING:
        return ('Bank erosion: FLOW-style local dry-bank sharing.' + common +
                'Higher dry neighbours supply the wet eroding cell. No retained undercut or survey. '
                'BRAIDS polygon wet/dry activation differs from FLOW masks.')
    return ('Bank erosion: surveyed accepted-demand toe supply.' + common +
            'Surveyed bank and toe cells stay at finest resolution. The mapped toe corridor does not automatically migrate. ' +
            ('Retained undercut releases finite blocks at the specified threshold; this is a geometric rule, not a cohesive geotechnical model.'
             if method == SURVEY_BANKS[1] else 'No retained undercut or block release.'))
