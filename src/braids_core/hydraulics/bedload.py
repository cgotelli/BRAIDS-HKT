"""Ordinary sediment admission for the conservative HLLC outer bed clock.

The existing finite-stock Exner owner consumes the accepted hydraulic endpoint
and time-integrated physical water flux. It owns bed evolution and rollback;
Hancock does not execute an SSP sediment stage or an extra bed update.
"""
from braids_core.river_model import ModelError

METHOD = 'hllc_endpoint_mpm_finite_stock_mf1_v1'
METHODS = {'capacity.general': 'hllc_endpoint_suspension_no_bedload_v1',
           'capacity.mpm': METHOD,
           'capacity.mpm_braids_5_055': 'hllc_endpoint_braids_mpm5_055_finite_stock_mf1_v1',
           'capacity.wong_parker_2006': 'hllc_endpoint_wong_parker_2006_finite_stock_mf1_v1',
           'capacity.ashida_michiue_generalized': 'hllc_endpoint_ashida_michiue_flow14_finite_stock_mf1_v1',
           'capacity.smart_jaggi': 'hllc_endpoint_smart_jaggi_graded_4_005_finite_stock_mf1_v1',
           'capacity.ancey_recking_2023': 'hllc_endpoint_ancey_recking_2023_eq42_finite_stock_mf1_v1',
           'capacity.recking_2010': 'hllc_endpoint_recking_2010_eq12_finite_stock_mf1_v1',
           'capacity.camenen_larson_2005': 'hllc_endpoint_camenen_larson_2005_sw_total_stress_finite_stock_mf1_v1',
           'capacity.wang_fredsoe': 'hllc_endpoint_wang_fredsoe_flow13_finite_stock_mf1_v1',
           'capacity.van_rijn_1993': 'hllc_endpoint_van_rijn_1993_flow_current_finite_stock_mf1_v1'}


def pending(path, message):
    raise ModelError('integration_pending', path, message +
                     ' Select GeoClaw augmented f-wave for the other sediment configurations.')


def validate_selection(s, hydro):
    if s is None:
        return
    split_suspension=s['transport']=='split' and s['suspension'] in (
        'suspension.settling_only','suspension.garcia_parker_1991',
        'suspension.de_leeuw_2020','suspension.van_rijn_1984',
        'suspension.zyserman_fredsoe_1994','suspension.camenen_larson_2008',
        'suspension.van_rijn_exchange')
    if s['capacity']['method'] not in METHODS or not (s['transport']=='bedload' and s['capacity']['method']!='capacity.general' or split_suspension):
        pending('sediment.capacity', 'Fast HLLC supports the ordinary Desktop bedload and suspended formulas. The separate Jiang–Cao mixture solver is not implemented for HLLC.')
    # Formula-specific consistency and available physics remain owned by the
    # normal river resolver. HLLC consumes accepted hydraulic fluxes through
    # the same sediment transaction as f-wave.


def validate_execution(profile):
    if profile['controls']['morfac'] != 1.:
        raise ModelError('integration_pending','settings.controls.morfac','BRAIDS standard MPM currently requires morphological factor 1 with either hydraulic method.')
    if profile['adaptive']:
        raise ModelError('integration_pending','settings.adaptive','BRAIDS standard MPM currently requires a static mesh with either hydraulic method.')
    if profile['columns']['base_update'] != 1:
        raise ModelError('integration_pending','settings.columns.base_update','BRAIDS standard MPM currently requires finite erodible stock above a fixed base with either hydraulic method.')


def identity(solver):
    """Record the actual split order, without claiming coupled second order."""
    from braids_core.sediment.bend_methods_v1 import digest
    p = solver.physics
    capacity = p.identity['resolved_river_model']['model']['physics']['sediment']['capacity']
    method=METHODS[capacity['method']]
    if capacity['method']=='capacity.smart_jaggi':
        params=capacity['parameters']
        if params['ACal']==4. and params['ThetaC']==.047:
            method='hllc_endpoint_smart_jaggi_graded_4_0047_finite_stock_mf1_v1'
        elif params['ACal']!=4. or params['ThetaC']!=.05:
            method='hllc_endpoint_smart_jaggi_user_parameters_v1'
    if p.morfac!=1.:method=method.replace('_mf1_v1','_morfac_v1')
    return dict(method=method, wrapper_sha256=digest(__file__),morphological_factor=float(p.morfac),
                capacity=capacity, suspension=p.identity['resolved_river_model']['model']['physics']['sediment']['suspension'],
                evaluation='accepted hydraulic endpoint at the beginning-of-interval bed',
                water='accepted time-integrated HLLC physical face flux divided by outer interval',
                bed='one conservative shared-face sediment-column transaction per outer interval',
                suspended='accepted water-flux advection and selected conservative exchange on the hydraulic clock',
                temporal_order='first-order hydraulic/bed splitting; second-order spatial reconstruction retained',
                transaction='parent begin/finish/commit/rollback; no Hancock sediment predictor')
