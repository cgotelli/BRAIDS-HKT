"""Shared sediment admission: named S3–S6 bundles, never implicit fallback."""
from __future__ import annotations

import math
from ..river_model import ModelError
from ..river_jiang_cao import METHOD as JIANG_CAO
from .grain_diameter_contract import REQUIRED, LABELS
from .river_classic_suspension import EDITOR as CLASSIC_EDITOR, METHODS as CLASSIC_METHODS
from .river_supplied_suspension import EDITOR as SUPPLIED
from .river_morfac import DESCRIPTION as MF_DESCRIPTION, LIMITATION as MF_LIMITATION

SCHEMA = 'braids.sediment-setup.morfac.v1'
GP = 'suspended.garcia_parker_1991'
DL = 'suspended.de_leeuw_2020'
VR = 'suspended.van_rijn_1993_flow2dh'
VR_BED = 'capacity.van_rijn_1993'
DIAMETER_LABELS = dict(LABELS, ds='Ds (suspended representative; not a percentile)')
SUSPENSION = {
    'suspended.settling_only': dict(
        label='Supplied suspension / settling only', sizes=('ds',), stage='supplied220',
        reference='BRAIDS conservative supplied-suspension control',
        closure='No entrainment; supplied material advects and settles.',
        auxiliaries='Measured Ds; clear-water FLOW SettleFrm2 settling; well-mixed depth-mean concentration; exact frozen-depth exponential deposition, no entrainment or equilibrium source. No hindering.'),
    'suspended.van_rijn_1993_flow2dh': dict(
        label='Van Rijn (1993) — FLOW 2DH bundle', sizes=('d50', 'd90', 'ds'), stage='S3',
        reference='Locked, unmodified Delft3D4 FLOW 2DH reference-column implementation',
        closure='Source-specific reference concentration, vertical profile and suspended-diameter treatment.',
        auxiliaries='FLOW reference-column shear, settling and exchange must be qualified together.'),
    'suspended.garcia_parker_1991': dict(
        label='García & Parker (1991)', sizes=('ds',), stage='S4',
        reference='https://doi.org/10.1061/(ASCE)0733-9429(1991)117:4(414)',
        closure='Noncohesive bed-material entrainment; not a total-load formula.',
        auxiliaries='Bounded Engelund–Hansen skin partition; a/h=0.05; clear-water SettleFrm2 on measured Ds; Rouse depth mean with total shear and constant concentration below a.'),
    'suspended.de_leeuw_2020': dict(
        label='de Leeuw et al. (2020)', sizes=('d84', 'ds'), stage='S5',
        reference='https://doi.org/10.5194/esurf-8-485-2020',
        closure='Sand/gravel entrainment with skin shear and Froude number; reference level 0.1 × depth.',
        auxiliaries='Measured D84 sets ks=3D84 in the Manning-Strickler Hsk solve; measured Ds sets Ferguson-Church settling. Eq26a threshold/saturation; P=(ws/u*total)^0.45, a/h=0.1, 1.66Ca below a.'),
    'suspended.van_rijn_1984': dict(
        label='Van Rijn (1984) — original suspension', sizes=('d50', 'd90', 'ds'), stage='S6',
        reference='https://www.leovanrijn-sediment.com/papers/P2-1984b.pdf',
        closure='Original suspended-load reference formulation, NOT FLOW modified tranb7.',
        auxiliaries='Measured Ds/D50/D90; grain Chezy skin shear bounded by total; a=max(30z0,0.01h); original three-branch settling; parabolic-constant profile with beta and damping; integral above a.'),
    'suspended.zyserman_fredsoe_1994': dict(
        label='Zyserman & Fredsøe (1994)', sizes=('d50', 'ds'), stage='S6',
        reference='https://doi.org/10.1061/(ASCE)0733-9429(1994)120:9(1021)',
        closure='Near-bed reference concentration from noncohesive sediment mobility.',
        auxiliaries='A=0.331, n=1.75, theta_c=0.045, saturation=0.46 at a=2D50; Eq9-11 boundary-layer skin shear with kN=2.5D50 bounded by total; Rubey1933 settling on measured Ds; Rouse profile integral above a.'),
    'suspended.camenen_larson_2008': dict(
        label='Camenen & Larson (2008) — current only', sizes=('d50', 'ds'), stage='S6',
        reference='https://lup.lub.lu.se/search/publication/c41e33fd-1d85-47bc-9aca-b11f50a9c5a6',
        closure='Steady-current suspended-load component; no wave contribution.',
        auxiliaries='Total hydraulic shear; AcR=0.0015 exp(-0.2Dstar), exp(-4.5 theta_c/theta); Soulsby-Whitehouse threshold; Van Rijn1984 settling on measured Ds; constant diffusivity and exponential depth mean.'),
}


def required_diameters(bedload, suspended):
    """Union by physical meaning, not by count or numerical similarity."""
    keys = list(REQUIRED.get(bedload, ()))
    if suspended in SUSPENSION:
        keys.extend(SUSPENSION[suspended]['sizes'])
    return tuple(dict.fromkeys(keys))


def resolve_setup(*, bedload=None, suspended=None, diameters_m=None):
    """Validate a proposal and report its admission independently of its physics.

    None means off. Inactive editor values must be filtered by the caller.
    There are no aliases, inferred percentiles or auto-selected bedload laws.
    """
    if bedload is not None and bedload not in REQUIRED:
        raise ModelError('invalid_input', 'sediment.setup.bedload', 'Choose a listed bulk bedload formula.')
    if suspended is not None and suspended not in SUSPENSION:
        raise ModelError('invalid_input', 'sediment.setup.suspended', 'Choose a named suspended formulation; legacy exchange IDs are not capacity laws.')
    if suspended and bedload == JIANG_CAO:
        raise ModelError('incompatible_physics', 'sediment.setup.combination',
            'Jiang–Cao owns coupled mobile bedload, phase exchange and density feedback. '
            'Independent suspension is not part of that bundle. Disable suspension or choose an ordinary bedload formula; no formula was changed.')
    if suspended == VR and bedload not in (None, VR_BED):
        raise ModelError('integration_pending', 'sediment.setup.combination',
            'The Van Rijn 1993 reference bundle requires its matching bedload producer, '
            'not a different bedload law. Select Van Rijn (1993) in Bedload, or turn bedload off '
            'for suspended-only transport. The selected bedload law will not be replaced.')
    sizes = {} if diameters_m is None else diameters_m
    keys = required_diameters(bedload, suspended)
    if not isinstance(sizes, dict) or set(sizes) != set(keys):
        raise ModelError('incompatible_grain_diameter', 'sediment.setup.diameters_m',
            'Enter exactly the active measured diameters: ' + ', '.join(DIAMETER_LABELS[k] for k in keys) +
            '. No diameter is inferred; inactive measurements are retained only in the editor.')
    for key in keys:
        x = sizes[key]
        if type(x) not in (int, float) or not math.isfinite(x) or x <= 0:
            raise ModelError('incompatible_grain_diameter', 'sediment.setup.diameters_m.' + key,
                'Enter a positive, finite measured ' + DIAMETER_LABELS[key] + ' in General / Grain sizes.')
    quantiles = [k for k in ('d30', 'd50', 'd84', 'd90') if k in sizes]
    for lo, hi in zip(quantiles, quantiles[1:]):
        if sizes[lo] > sizes[hi]:
            raise ModelError('incompatible_grain_diameter', 'sediment.setup.diameters_m.' + lo,
                f'{lo.upper()} cannot exceed {hi.upper()}. Correct the measurements; sizes are not sorted automatically.')
    # Dm and Ds are deliberately NOT ordered as bed percentiles.
    assumptions = ['One bulk noncohesive material; no transported grain classes, sorting or armoring.',
        'Required bed percentiles, arithmetic mean Dm and suspended representative Ds are independent measured inputs.',
        'Bedload and suspended inventories must exchange with the same finite bed without double-counting sediment.']
    limits = []
    if bedload == 'capacity.wong_parker_2006':
        limits.append('Wong–Parker retains its uniform, lower-regime plane-bed applicability; adding suspension does not extend that range.')
    if bedload == 'capacity.smart_jaggi':
        limits.append('Smart–Jäggi energy slope is computed from hydraulic resistance; it is not an instantaneous DEM slope.')
    if bedload == 'capacity.recking_2010':
        limits.append('Recking bed mobility uses D84 and the prescribed reach slope; D84 is NOT silently reused as the settling diameter.')
    if bedload == 'capacity.wang_fredsoe':
        limits.append('Only Wang–Fredsøe bedload is used; its native algebraic suspended residual is not an additional source.')
    if suspended:
        assumptions.append(MF_DESCRIPTION)
        limits.append(MF_LIMITATION)
        limits += [
            'Near-bed reference concentration is not the depth-mean concentration and neither is a transport-weighted concentration.',
            'Ds describes the same bulk material, not a separate supply of fines over immobile gravel; selective availability is deferred.',
            'No waves or cohesive sediment. Conservative wet/dry collapse is available; accepted-demand bank/toe supply is available.']
        if suspended == VR:
            limits += ['Current-only FLOW 2DH bundle: IopSus=0 retains measured Ds; no automatic grain-size derivation.',
                'Positive MF, sediment start=0, one finite mixed bed; sorting/bedform combinations are not admitted.',
                'FLOW returns a transport-weighted equilibrium concentration to its Galappatti 2DH exchange. This is not the reference-level concentration.',
                'First-order donor-cell suspended advection, once per accepted outer transaction; second-order bedload reconstruction is unchanged.',
                'Sand-bed application, not a validated universal steep-gravel entrainment law. No experimental validation claim.']
        elif suspended == GP:
            limits += ['GP1991 uniform-material law: A=1.3e-7, saturation 0.3, Z=(u*skin/ws) Rep^0.6. No mixture correction.',
                'Entrainment uses bounded EH1967 skin shear; vertical mixing uses total hydraulic shear. Hydraulic resistance is not replaced.',
                'Rouse concentration above a=0.05h, constant below a; whole-depth mass average. Advection uses water depth-mean velocity (no profile velocity correction).',
                'Small-slope, dilute noncohesive suspension. Calibration: Rep 3.5–37, u*skin/ws 0.70–7.50, h/Ds 240–2400. Outside these ranges is extrapolation, not mountain-river braids_support.',
                'Explicit shallow extension: h<20Ds suppresses entrainment and uses well-mixed settling; dry material strands conservatively.',
                'Positive MF, start=0, one finite fully mixed bed; no sorting/bedforms. No hindered settling or density feedback.',
                'First-order conservative suspended advection. Bedload reconstruction is unchanged. Native bed-formula suspended capacity is NOT added.',
                'This independently assembled 2DH bundle is component-qualified, not a reproduction of an experimental river or a complete published solver.']
        elif suspended == SUPPLIED:
            limits += ['No resuspension: deposited material cannot return to suspension. Active bedload may still redistribute it.',
                'Measured Ds sets clear-water FLOW SettleFrm2; no nominal bed-size substitution, hindering, shear activation or minimum settling-speed floor.',
                'Well-mixed vertical concentration and first-order conservative advection using accepted water flux. This is a supplied-sediment control, not a turbulent entrainment or equilibrium-profile model.',
                'All wet depths settle at ws/h; dry mass deposits once. Wet bed deposition is multiplied by MF; suspension loses only physical settled mass. One finite mixed bulk bed and start 0.',
                'Ds >=0.0625 mm; no cohesive or dense-flow coupling. Banks/bedforms/sorting/slope/hiding extras are deferred. No native bed-formula suspended capacity is added.']
        elif suspended == DL:
            limits += ['de Leeuw 2020 Eq26a as printed: coefficient 4.74e-4, threshold 0.015, exponents 1.5 and 1.18, denominator factor 3 (limit 1/3, despite the paper prose 0.3).',
                'Preferred full-concentration profile (section 3.3): P=(ws/u*total)^0.45, NOT the separate best-fit two-parameter P equation. a=0.1h; below a, C=1.66Ca (section 4.2).',
                'Manning-Strickler Eq7 solves Hsk using local physical friction slope Sf=u*total^2/(gh), ks=3D84; skin shear is bounded by total. These local-flow bounds are named BRAIDS adaptations; hydraulic resistance is not replaced.',
                'Ferguson-Church natural-grain settling C1=18 C2=1; measured Ds >=0.0625 mm. D84 is an independent bed descriptor, not a transported fraction.',
                'Explicit shallow extension: h<max(10Ds,3D84) suppresses entrainment and uses well-mixed settling; dry material strands conservatively.',
                'Calibration is mainly sand, bed medians 44-517 micrometres and depths 0.06-32 m. Gravel is extrapolation supported by saltation theory, not measured gravel-suspension braids_support.',
                'One bulk material (bed fraction 1); positive MF, start=0, finite mixed bed; no sorting/bedforms/slope/hiding extras, hindered settling or density feedback.',
                'Depth-mean mass uses the whole profile; advection uses mean water velocity, not a velocity-profile correction. Below-reference material is assigned to the suspension inventory; this is not an exact bedload/suspension partition.',
                'First-order conservative suspended advection; unchanged second-order bedload reconstruction. Native bed-formula algebraic suspended capacity is NOT added.',
                'Component-qualified 2DH composition, not a full reproduction of the paper calibration dataset or an experimental river.']
        elif suspended in CLASSIC_EDITOR:
            limits += [
                'Current-only single-bulk equilibrium closure in BRAIDS 2DH; measured Ds, bed D50 (and D90 for VR1984). No inferred percentiles.',
                'Physical local total shear comes from hydraulic resistance. VR1984 uses grain Chezy with 3D90; ZF1994 solves its Eq9-11 boundary layer with kN=2.5D50. Both skin values are bounded by total shear; CL2008 uses total shear.',
                'Clear-water settling on measured Ds: original Van Rijn1984 branches for VR; original Rubey1933 for ZF; Van Rijn1984 as the explicitly selected CL auxiliary.',
                'VR/ZF suspension inventory integrates only above a (no below-reference mass added); CL integrates its exponential from the bed. Transport uses depth-mean water velocity, not the original VR velocity-weighted transport integral.',
                'Named shallow extension: h<20max(Ds,D50), or a>=h/2, suppresses entrainment and uses well-mixed settling; dry material strands conservatively. VR/CL reference concentration cannot exceed packing 0.65.',
                'Positive MF, start=0, one finite fully mixed bed; no bedforms/sorting/slope/hiding extras, waves, hindered settling or density feedback.',
                'First-order conservative suspended advection; unchanged second-order bedload. Native bed-formula algebraic suspension is not added. Numerical qualification is not experimental mountain-river braids_support.']
            limits.append({
                CLASSIC_EDITOR[0]: 'VR1984: Eq38 concentration, a=max(30z0,0.01h), Eq19 parabolic-constant profile with Zprime=Z+psi (Eq33-34). Beta=1+2r^2 uses r=min(ws/ut,1), limit 3, not the retained FLOW producer limit 1.5.',
                CLASSIC_EDITOR[1]: 'ZF1994: A=0.331, exponent=1.75, threshold=0.045, saturation=0.46 at a=2D50; Rouse profile with total shear and kappa=0.4.',
                CLASSIC_EDITOR[2]: 'CL2008 current: AcR=0.0015 exp(-0.2Dstar), exp(-4.5 theta_cr/theta), total shear and Soulsby-Whitehouse threshold. Exponential profile and Schmidt coefficients 0.4/3.5, not the 2007 coefficients. Threshold ratio is corrected to theta_cr/theta as in the 2007 author report and Camenen et al.2018; supplied 2008 Eq16 prints the inverted ratio.',
            }[suspended])
        else:limits += ['Suspended equations, auxiliary coefficients and coupled CPU/GPU execution are NOT qualified for this selection.']
    return dict(schema=SCHEMA, bedload=bedload, suspended=suspended,
        diameters_m=dict(sizes), status='qualified_supplied' if suspended == SUPPLIED else 'qualified_s6' if suspended in CLASSIC_EDITOR else 'qualified_s5' if suspended == DL else 'qualified_s4' if suspended == GP else 'qualified_s3' if suspended == VR else 'qualification_pending' if suspended else 'existing_bedload_path' if bedload else 'hydrodynamics_only',
        next_gate=SUSPENSION[suspended]['stage'] if suspended and suspended not in (VR,GP,DL,SUPPLIED,*CLASSIC_EDITOR) else None,
        auxiliary_status='resolved_clearwater_supplied_bundle' if suspended == SUPPLIED else 'resolved_named_s6_2dh_bundle' if suspended in CLASSIC_EDITOR else 'resolved_de_leeuw2020_2dh_bundle' if suspended == DL else 'resolved_gp1991_2dh_bundle' if suspended == GP else 'resolved_flow2dh_bundle' if suspended == VR else 'specification_only' if suspended else 'existing_model_resolver',
        auxiliary_specification=SUSPENSION[suspended]['auxiliaries'] if suspended else None,
        source=SUSPENSION[suspended]['reference'] if suspended else None,
        assumptions=assumptions, limitations=limits)


def require_runnable(setup):
    if setup['status'] == 'qualification_pending':
        raise ModelError('integration_pending', 'sediment.setup.suspended',
            SUSPENSION[setup['suspended']]['label'] + ' is a saved configuration draft, not a runnable suspended solver. '
            'Qualification is scheduled at ' + setup['next_gate'] + '. Disable suspended transport to run the current bedload solver. '
            'No suspended physics will be silently omitted.')


def summary(setup):
    if setup['status'] == 'hydrodynamics_only':
        return 'Hydrodynamics only. Opening the sediment extension adds no physics.'
    lines = ['Configuration status: ' + setup['status'].replace('_', ' '),
             'Bedload: ' + (setup['bedload'] or 'OFF'),
             'Suspended: ' + (setup['suspended'] or 'OFF'),
             'Measured diameters: ' + ', '.join(f'{DIAMETER_LABELS[k]} = {v*1000:g} mm' for k, v in setup['diameters_m'].items())]
    if setup['suspended']:
        row = SUSPENSION[setup['suspended']]
        lines += ['Closure: ' + row['closure'],
                  'Resolved auxiliaries: FLOW current reference column, SettleFrm2, measured Ds, Galappatti exchange; no algebraic suspended flux added.' if setup['suspended']==VR else
                  'Resolved auxiliaries: '+row['auxiliaries'] if setup['suspended'] in (GP,DL,SUPPLIED,*CLASSIC_EDITOR) else
                  'Auxiliary specification (not active physics): ' + row['auxiliaries'],
                  'Source: ' + row['reference']]
    return '\n\n'.join(lines + ['Assumptions:\n' + '\n'.join(setup['assumptions']),
        'Limitations:\n' + '\n'.join(setup['limitations'])])


def validate_execution(project):
    """Recompute admission; a saved summary is not an execution permission."""
    from .river_vr1993 import DESKTOP_PARAMETERS, REFERENCE_PARAMETERS
    draft=project['editor_state']['sediment'];w=draft['widgets']
    bed=w.get('capacity') if w.get('bedload_on') else None
    sus=w.get('suspended_formula')
    sizes={}
    for key in required_diameters(bed,sus):
        try:sizes[key]=float(draft['grain_sizes'].get(key,''))/1000.
        except (ValueError,TypeError):sizes[key]=None
    setup=resolve_setup(bedload=bed,suspended=sus,diameters_m=sizes)
    require_runnable(setup)
    s=project['model']['physics']['sediment'];p=project['settings']
    def fail(message):raise ModelError('integration_pending','sediment.setup',message)
    if sus==SUPPLIED:
        validate_supplied_execution(project)
        from .river_supplied_suspension import parameters
        if project['supplied_suspension_parameters']!=parameters(sizes['ds']):
            fail('Measured Ds or supplied-suspension auxiliaries differ between editor and model. Prepare again.')
        if s['capacity']['method']!=(bed or 'capacity.general') or s['capacity']['grain_sizes_m']!={k:sizes[k] for k in REQUIRED.get(bed,())}:
            fail('The supplied-suspension bedload selection or descriptors differ from the editor. Prepare again.')
        if p['suspension']['bed_factor']!=float(bed is not None):fail('The bedload switch differs from the prepared supplied-suspension model.')
        return
    if sus in CLASSIC_EDITOR:
        validate_classic_execution(project)
        from .river_classic_suspension import parameters
        if project['classic_suspension_parameters']!=parameters(sus.replace('suspended.','suspension.'),sizes):
            fail('Measured S6 diameters or auxiliaries differ between editor and model. Re-enter the measurements and prepare again.')
        if s['capacity']['method']!=(bed or 'capacity.general') or s['capacity']['grain_sizes_m']!={k:sizes[k] for k in REQUIRED.get(bed,())}:
            fail('S6 bedload selection or measured diameters differ from the editor. Prepare again.')
        if p['suspension']['bed_factor']!=float(bed is not None):fail('The bedload switch differs from the prepared S6 model.')
        return
    if sus==GP:
        validate_gp_execution(project)
        if project['garcia_parker_parameters']['suspended_diameter_m']!=sizes['ds']:
            fail('Measured Ds differs between the editor and GP1991 model. Re-enter Ds and prepare again.')
        if s['capacity']['method']!=(bed or 'capacity.general') or s['capacity']['grain_sizes_m']!={k:sizes[k] for k in REQUIRED.get(bed,())}:
            fail('GP1991 bedload selection or measured grain descriptors differ from the editor. Prepare again.')
        if p['suspension']['bed_factor']!=float(bed is not None):fail('The bedload switch differs from the prepared GP1991 model.')
        return
    if sus==DL:
        validate_dl_execution(project)
        params=project['de_leeuw_parameters']
        if params['suspended_diameter_m']!=sizes['ds'] or params['bed_d84_m']!=sizes['d84']:
            fail('Measured Ds/D84 differ between the editor and de Leeuw model. Re-enter the measurements and prepare again.')
        if s['capacity']['method']!=(bed or 'capacity.general') or s['capacity']['grain_sizes_m']!={k:sizes[k] for k in REQUIRED.get(bed,())}:
            fail('de Leeuw bedload selection or measured grain descriptors differ from the editor. Prepare again.')
        if p['suspension']['bed_factor']!=float(bed is not None):fail('The bedload switch differs from the prepared de Leeuw model.')
        return
    if sus!=VR:fail('Choose the qualified Van Rijn 1993 FLOW 2DH suspension bundle.')
    if s is None or s['capacity']['method']!=VR_BED or s['transport']!='split' or s['suspension']!='suspension.van_rijn_exchange':
        fail('The active model must match the selected Van Rijn 1993 split suspension setup; prepare the project again.')
    if s['capacity']['parameters']!=DESKTOP_PARAMETERS:
        fail('S3 uses the displayed fixed FLOW parameters, including IopSus=0. Restore the standard Van Rijn bundle; evolving Ds is not admitted in Desktop.')
    for key,value in REFERENCE_PARAMETERS.items():
        if key!='suspended_diameter_m' and project['van_rijn_reference_parameters'][key]!=value:
            fail('S3 requires '+key+'='+str(value)+'. Restore this documented FLOW reference setting; no silent correction is applied.')
    if s['capacity']['grain_sizes_m']!={k:sizes[k] for k in ('d50','d90')} or project['van_rijn_reference_parameters']['suspended_diameter_m']!=sizes['ds']:
        fail('Measured D50, D90 and Ds differ between the editor and run model. Re-enter the grain sizes and prepare again.')
    if p['sediment_start_time_s']!=0.:
        fail('Suspension requires sediment start time 0. Set this in General before preparation.')
    if p['suspension']['bed_factor']!=float(bed is not None) or p['suspension']['sus_factor']!=1.:
        fail('S3 requires the explicit bedload switch and active Van Rijn entrainment (factor 1); prepare again.')
    if any(s[k]!='off' for k in ('bedforms',)) or s['bed']['composition']!='bed.fully_mixed':
        fail('Turn off bedforms and use one finite mixed bed. These combinations are deferred until suspension qualification is finished.')


def validate_gp_execution(project):
    """Core admission also applies to documents without a Desktop editor."""
    from .river_garcia_parker import validate, ZERO_CAPACITY, METHOD
    def fail(message):raise ModelError('incompatible_physics','sediment.garcia_parker',message)
    try:validate(project.get('garcia_parker_parameters'))
    except ValueError as exc:fail(str(exc))
    s=project['model']['physics']['sediment'];p=project['settings']
    if s is None or s['suspension']!=METHOD or s['transport']!='split':fail('Select GP1991 with independently transported suspended mass.')
    method=s['capacity']['method'];bed=p['suspension']['bed_factor']
    if bed:
        if method not in REQUIRED or method==JIANG_CAO:fail('Choose an ordinary single-material bedload formula; Jiang–Cao cannot add independent suspension.')
    elif method!='capacity.general' or s['capacity']['parameters']!=ZERO_CAPACITY:
        fail('Suspended-only GP1991 requires its explicit zero-capacity carrier, not a hidden bedload formula. Prepare again.')
    if len(s['grains'])!=1 or p['sediment_start_time_s']!=0 or p['suspension']['sus_factor']!=1:
        fail('GP1991 requires one bulk material, start time 0 and active entrainment. Correct General settings.')
    if not bed and s['grains'][0]['diameter_m']!=project['garcia_parker_parameters']['suspended_diameter_m']:
        fail('The suspended-only bulk carrier must use the measured Ds, without inferring another diameter.')
    if any(s[k]!='off' for k in ('bedforms',)) or s['bed']['composition']!='bed.fully_mixed' or s['bed_slope']!='bed_slope.off' or s['capacity_hiding']!='hiding.none' or p['columns']['base_update']!=1:
        fail('Use one finite mixed bed, without bedforms/slope/hiding extras. Their suspension combinations are deferred.')



def validate_supplied_execution(project):
    """No hidden nominal-size or hindered-settling execution route."""
    from .river_supplied_suspension import validate,METHOD
    from .river_garcia_parker import ZERO_CAPACITY
    def fail(message):raise ModelError('incompatible_physics','sediment.supplied_suspension',message)
    try:params=validate(project.get('supplied_suspension_parameters'))
    except ValueError as exc:fail(str(exc))
    s=project['model']['physics']['sediment'];p=project['settings']
    if s is None or s['suspension']!=METHOD or s['transport']!='split':
        fail('Select supplied suspension with independent suspended mass transport.')
    method=s['capacity']['method'];bed=p['suspension']['bed_factor']
    if bed:
        if method not in REQUIRED or method==JIANG_CAO:fail('Choose an ordinary bedload formula or turn bedload off; Jiang-Cao cannot add independent suspension.')
    elif method!='capacity.general' or s['capacity']['parameters']!=ZERO_CAPACITY:
        fail('Supplied-only suspension requires its explicit zero-bedload carrier. Prepare again.')
    if len(s['grains'])!=1 or p['sediment_start_time_s']!=0 or p['suspension']['sus_factor']!=0:
        fail('Supplied suspension requires one bulk material, start 0 and zero entrainment. Correct General settings.')
    if not bed and s['grains'][0]['diameter_m']!=params['suspended_diameter_m']:
        fail('Supplied-only carrier must use measured Ds. Prepare again.')
    if any(s[k]!='off' for k in ('bedforms',)) or s['bed']['composition']!='bed.fully_mixed' or s['bed_slope']!='bed_slope.off' or s['capacity_hiding']!='hiding.none' or p['columns']['base_update']!=1:
        fail('Use one finite mixed bed without bedform/sorting/slope/hiding extras. Their suspension combinations are deferred.')
    descriptors=dict(s['capacity']['grain_sizes_m']);descriptors['ds']=params['suspended_diameter_m']
    resolve_setup(bedload=method if bed else None,suspended=SUPPLIED,diameters_m=descriptors)


def validate_classic_execution(project):
    """Strict shared admission, also for headless projects without editor state."""
    from .river_classic_suspension import validate,FIELDS
    from .river_garcia_parker import ZERO_CAPACITY
    def fail(message):raise ModelError('incompatible_physics','sediment.classic_suspension',message)
    try:params=validate(project.get('classic_suspension_parameters'))
    except ValueError as exc:fail(str(exc))
    s=project['model']['physics']['sediment'];p=project['settings']
    if s is None or s['suspension']!=params['method'] or s['transport']!='split':
        fail('Select the same named S6 formula with independently transported suspended mass.')
    method=s['capacity']['method'];bed=p['suspension']['bed_factor']
    if bed:
        if method not in REQUIRED or method==JIANG_CAO:fail('Choose an ordinary bulk bedload formula; Jiang-Cao cannot add independent suspension.')
    elif method!='capacity.general' or s['capacity']['parameters']!=ZERO_CAPACITY:
        fail('Suspended-only S6 requires its explicit zero-capacity carrier. Prepare again.')
    if len(s['grains'])!=1 or p['sediment_start_time_s']!=0 or p['suspension']['sus_factor']!=1:
        fail('S6 requires one bulk material, start time 0 and active entrainment. Correct General settings.')
    if not bed and s['grains'][0]['diameter_m']!=params['suspended_diameter_m']:
        fail('Suspended-only bulk carrier must use measured Ds.')
    if any(s[k]!='off' for k in ('bedforms',)) or s['bed']['composition']!='bed.fully_mixed' or s['bed_slope']!='bed_slope.off' or s['capacity_hiding']!='hiding.none' or p['columns']['base_update']!=1:
        fail('Use one finite mixed bed, without bedforms/slope/hiding extras. These combinations remain deferred.')
    descriptors=dict(s['capacity']['grain_sizes_m'])
    for key,field in FIELDS.items():
        if field not in params:continue
        if key in descriptors and descriptors[key]!=params[field]:fail('Bedload and suspension must use the same measured '+key.upper()+'. Re-enter the value and prepare again.')
        descriptors[key]=params[field]
    resolve_setup(bedload=method if bed else None,suspended=params['method'].replace('suspension.','suspended.'),diameters_m=descriptors)


def validate_dl_execution(project):
    """Core admission also applies to documents without a Desktop editor."""
    from .river_de_leeuw import validate, METHOD
    from .river_garcia_parker import ZERO_CAPACITY
    def fail(message):raise ModelError('incompatible_physics','sediment.de_leeuw',message)
    try:validate(project.get('de_leeuw_parameters'))
    except ValueError as exc:fail(str(exc))
    s=project['model']['physics']['sediment'];p=project['settings']
    if s is None or s['suspension']!=METHOD or s['transport']!='split':fail('Select de Leeuw 2020 with independently transported suspended mass.')
    method=s['capacity']['method'];bed=p['suspension']['bed_factor']
    if bed:
        if method not in REQUIRED or method==JIANG_CAO:fail('Choose an ordinary single-material bedload formula; Jiang–Cao cannot add independent suspension.')
    elif method!='capacity.general' or s['capacity']['parameters']!=ZERO_CAPACITY:
        fail('Suspended-only de Leeuw 2020 requires its explicit zero-capacity carrier, not a hidden bedload formula. Prepare again.')
    if len(s['grains'])!=1 or p['sediment_start_time_s']!=0 or p['suspension']['sus_factor']!=1:
        fail('de Leeuw 2020 requires one bulk material, start time 0 and active entrainment. Correct General settings.')
    if not bed and s['grains'][0]['diameter_m']!=project['de_leeuw_parameters']['suspended_diameter_m']:
        fail('The suspended-only bulk carrier must use the measured Ds, without inferring another diameter.')
    if any(s[k]!='off' for k in ('bedforms',)) or s['bed']['composition']!='bed.fully_mixed' or s['bed_slope']!='bed_slope.off' or s['capacity_hiding']!='hiding.none' or p['columns']['base_update']!=1:
        fail('Use one finite mixed bed, without bedforms/slope/hiding extras. Their suspension combinations are deferred.')
    descriptors=dict(s['capacity']['grain_sizes_m'])
    d84=project['de_leeuw_parameters']['bed_d84_m']
    if 'd84' in descriptors and descriptors['d84']!=d84:
        fail('Bedload and suspension must use the same measured bed D84. Re-enter D84 and prepare again.')
    descriptors.update(d84=d84,ds=project['de_leeuw_parameters']['suspended_diameter_m'])
    resolve_setup(bedload=method if bed else None,suspended=DL,diameters_m=descriptors)
