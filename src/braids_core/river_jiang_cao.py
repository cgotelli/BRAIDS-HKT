"""Explicit Desktop contract for the bounded Jiang–Cao research composition.

This is NOT MPM with a different coefficient. It owns non-equilibrium mobile
bedload, phase exchange, density feedback, slope direction and resistance.
The eight established capacity/Exner routes do not use this module.
"""
from copy import deepcopy
import math

METHOD = 'capacity.mpm_jc2026_p15_reconstruction_v1'
BUNDLE = 'bundle.jiang_cao_p15_bulk_diameters213_v1'
COLLAPSE = 'redistribution.jiang_cao_repose200_v1'
SECONDARY = 'secondary.kb86_jc_profiles_adjustment206_v1'
ROUGHNESS = 'roughness.sun2015_van_rijn1984_reconstruction_v1'
SHALLOW = 'shallow.submerged_grain_bedload_separate_profile209_v2'
DEFAULTS = dict(VicMol=1.e-6, ReposeAngle_deg=30., ProtectInitialDryBank=0)


def active(model):
    s = model['physics']['sediment']
    return s is not None and s['capacity']['method'] == METHOD


def fail(path, message):
    from .river_model import ModelError
    raise ModelError('integration_pending', 'jiang_cao.'+path, message)


def resolve_selection(selection, hydro, decisions):
    from .river_model import _grains, new_bedload_selection
    from .sediment.capacity_parameters import resolve_capacity_parameters
    from .sediment.grain_diameter_contract import require_diameter_definition, validate_grain_sizes
    s = deepcopy(selection)
    _grains(s['grains'])
    if len(s['grains']) != 1:
        fail('grains', 'Use one bulk material with separately entered D50 and D90; transported classes are not admitted.')
    require_diameter_definition(METHOD, 1, s['grains'][0]['diameter_definition'])
    if set(s['capacity']) != {'method', 'parameters', 'grain_sizes_m'}:
        fail('grain_sizes_m', 'Enter explicit D50 and D90 in Grain sizes; old single-diameter projects are not accepted.')
    validate_grain_sizes(s['capacity'], primary=s['grains'][0]['diameter_m'])
    if hydro['roughness'] != ROUGHNESS:
        fail('roughness', 'Select automatic Sun/van Rijn (Jiang–Cao) resistance in Domain. This formula owns its roughness; prescribed Manning/Chézy is incompatible.')
    p = dict(resolve_capacity_parameters(METHOD, s['capacity']['parameters']))
    if not 0 < p['ReposeAngle_deg'] < 90:
        fail('ReposeAngle_deg', 'Enter a repose angle strictly between 0 and 90 degrees.')
    baseline = new_bedload_selection()
    for key in ('transport', 'slope_hiding', 'bed_slope', 'shear', 'suspension',
                'bed', 'bedforms', 'banks'):
        if s[key] != baseline[key]:
            fail(key, 'The research bundle owns slope, exchange and roughness. '
                 'Disable separate corrections, banks, suspension, layers and bedform switches.')
    if s['fraction_basis'] not in ('fraction_basis.solid_volume', 'fraction_basis.mass'):
        fail('fraction_basis', 'Use explicit mass or solid-volume fractions.')
    for key, allowed in (('capacity_hiding', ('auto', 'hiding.none')),
                         ('secondary_direction', ('auto', 'capacity_owned')),
                         ('settling', ('auto', 'off'))):
        if s[key] not in allowed:
            fail(key, 'Remove the external correction; this bundle owns the corresponding operator.')
    if s['avalanche'] not in ('off', COLLAPSE):
        fail('collapse', 'Select Jiang–Cao repose projection or Off. The BRAIDS wet/dry '
             'redistribution belongs to the capacity/Exner formulas; never run both.')
    if p['ProtectInitialDryBank'] and s['avalanche'] != COLLAPSE:
        fail('ProtectInitialDryBank', 'Enable Jiang–Cao repose projection, or set dry-bank protection to 0.')
    if hydro['turbulence'] != 'off':
        fail('turbulence', 'Turn off eddy viscosity for this bounded research integration.')
    if hydro['secondary_flow'] != 'secondary.prognostic':
        fail('secondary', 'Enable secondary flow in Advanced hydrodynamics. This bundle resolves '
             'it to the named finite-adjustment extension, not classic FLOW spiral intensity.')
    s['capacity']['parameters'] = p
    s.update(capacity_hiding='hiding.none', secondary_direction='capacity_owned', settling='off',
             composition_fraction='composition.single_bulk_material',
             bundle=dict(method=BUNDLE, secondary=SECONDARY, roughness=ROUGHNESS,
                         shallow=SHALLOW, grain_sizes_m=deepcopy(s['capacity']['grain_sizes_m']),
                         bedforms='equilibrium van Rijn 1984 inside bundle resistance',
                         state_fields=['h','hu','hv','mobile_solid','bed_solid','bed_z','hI'],
                         collapse=s['avalanche'], exponent=1.5))
    decisions.append(dict(path='sediment.bundle', requested=METHOD, resolved=s['bundle'],
        reason='Research composition: non-equilibrium bedload, water/pore-water and mixture '
               'momentum coupling; Sun/van Rijn roughness; submerged-grain bedload with separate '
               'secondary-profile gating; finite-stock exchange without target clipping; '
               'finite-adjustment secondary flow. Prescribed roughness and FLOW Betac are inactive. '
               'MF must be 1. Not a reproduction of the authors’ transient implementation.'))
    return s


def material(project):
    from .sediment.jiang_cao_capacity import JiangCaoMaterial
    from .sediment.jiang_cao_exchange import ExchangeMaterial
    s=project['model']['physics']['sediment']; g=s['grains'][0]; p=s['capacity']['parameters']
    dry=project['settings']['columns']['dry_density_kg_m3'][0]
    return ExchangeMaterial(JiangCaoMaterial(g['diameter_m'],g['solid_density_kg_m3'],1000.,
        math.radians(p['ReposeAngle_deg'])),1-dry/g['solid_density_kg_m3'],p['VicMol'])


def validate_project(project):
    """Reject ignored/incompatible settings before allocating geometry."""
    from .river_boundary import RATIO
    from .sediment.jiang_cao_roughness import RoughnessParameters
    s=project['settings']; p=s['controls']; selection=project['model']['physics']['sediment']
    mat=material(project)
    RoughnessParameters(mat,selection['capacity']['grain_sizes_m']['d90'])
    if p['morfac'] != 1:
        fail('morfac', 'Set morphological factor to 1; accelerated phase exchange is not qualified.')
    if any(p[k] for k in ('betac','diffusivity','viscosity')):
        fail('hydrodynamics', 'Set FLOW Betac, secondary diffusion and eddy viscosity to 0. '
             'The selected finite-adjustment bundle supplies its own secondary feedback.')
    if s['sediment_start_time_s']:
        fail('start', 'Set sediment start time to 0. Delayed activation is not integrated for this bundle.')
    if any(project['initial']['concentration_kg_m3']):
        fail('initial', 'Initial mobile bedload must be zero in this integration. Suspended concentration '
             'is not reinterpreted as mobile bedload; supply mobile sediment through qs/q inlets.')
    if project['material_zones']['roughness']:
        fail('roughness', 'Remove roughness polygons: the bundle computes resistance from explicit D50, D90 and bedforms.')
    if s['adaptation']['domain_border_enabled']:
        fail('mesh', 'Disable domain-border refinement. Use depth/buffer AMR and optional initial dry-bank protection.')
    if selection['capacity']['parameters']['ProtectInitialDryBank'] and not s['adaptive']:
        fail('ProtectInitialDryBank', 'Use dynamic depth AMR, or set initial dry-bank protection to 0.')
    if s['adaptive'] and s['adaptation']['interval'] > 0:
        from .sediment.jiang_cao_adaptive import MAX_EVENTS
        if s['end_time_s']/s['adaptation']['interval'] > MAX_EVENTS:
            fail('mesh', f'This research checkpoint admits at most {MAX_EVENTS} AMR assessments. '
                 'Increase the AMR interval or shorten the run; long braided-river runs are not qualified yet.')
    for name,row in s['sediment_boundaries'].items():
        if row['kind']=='outflow':continue
        if row['kind']!=RATIO:
            fail('inlet.'+name, 'Choose Sediment / water ratio (qs/q). The mobile-load bundle currently '
                 'requires a constant incoming solid/water volume ratio, not equilibrium capacity or total feed.')
        r=row['solid_water_ratio'][0]; c=r/(1+r)
        if c >= 1-mat.porosity:
            fail('inlet.'+name, 'Reduce qs/q: the incoming mobile solid fraction must be below bed packing.')
    if any(row['method']!='forcing.constant' for row in project['sediment_schedules'].values()):
        fail('inlet', 'Use constant sediment supply; time-varying mobile feed is not yet integrated.')
    for name,row in project['boundary_states'].items():
        if row['bed'] != dict(method='bed.free',initial=dict(method='initial.prepared_bed')):
            fail('boundary.'+name, 'Use the free exterior bed: it follows the current adjacent bed.')
        if row['secondary_intensity_m_s'] != 0:
            fail('boundary.'+name,'Incoming secondary intensity must be zero for the finite-adjustment bundle.')
        if any(row[k]['method']!='forcing.constant' for k in ('hydraulic_schedule','bed_schedule','composition_schedule')):
            fail('boundary.'+name,'Use constant boundary forcing in this first research integration.')


def validate_binding(profile, resolved):
    """The seven-field worker, not the equilibrium Exner installer, owns this route."""
    s=resolved['physics']['sediment']; cols=profile['columns']
    g=s['grains'][0]
    if cols['bed']!='mixed' or len(cols['fraction_ids'])!=1 or cols['base_update']!=1:
        fail('bed', 'Use one fully mixed finite bed, with independent finite base.')
    if (cols['diameter_m'][0]!=g['diameter_m'] or cols['solid_density_kg_m3'][0]!=g['solid_density_kg_m3']
            or profile['controls']['diameter']!=g['diameter_m']):
        fail('grains','Carrier and selected material must agree exactly.')
    if profile['bank']['bank_theta'] or profile['bank']['collapse']:
        fail('collapse','The legacy bank/collapse carrier must be off; Jiang–Cao owns its selected projection.')
    return resolved
