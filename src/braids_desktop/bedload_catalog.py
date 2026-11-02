"""Small, explicit single-grain bedload menu; no hidden suspended channel."""
CATALOG={
    # Public order keeps MPM as the default; Smart-Jaeggi is appended below.
    'capacity.mpm':dict(label='MPM',parameters=dict(ACal=1.),
        note='Meyer–Peter–Müller, FLOW implementation with its grain/form-drag correction. Standard multiplier 1.'),
    'capacity.wong_parker_2006':dict(label='Wong & Parker (2006)',parameters=dict(ACal=1.),
        note='Single-grain, plane-bed bedload: 3.97 × max(Shields − 0.0495, 0)^1.5. '
        'Arithmetic mean diameter Dm, NOT a percentile. Fixed coefficient, threshold and exponent; '
        'transport multiplier 1 reproduces the published relation. Uses local hydraulic bed shear '
        'without the MPM form-drag reduction. Intended for uniform sediment and lower-regime '
        'plane beds, not a universal graded-river predictor. No suspension or external hiding.'),
    'capacity.ashida_michiue_generalized':dict(label='Ashida–Michiue (FLOW)',
        parameters=dict(ACal=17.,ThetaC=.05,PowerM=1.5,PowerP=1.,PowerQ=1.),
        note='FLOW generalized Ashida–Michiue: A=17, critical Shields=0.05, powers 1.5/1/1. One grain; no extra hiding. These are explicit selected coefficients, not inferred FLOW defaults.'),
    'capacity.wang_fredsoe':dict(label='Wang–Fredsøe (bedload)',parameters=dict(ACal=1.,VicMol=1.e-6),
        note='FLOW Wang–Fredsøe bedload component (Engelund–Fredsøe bedload law), with characteristic shear and grain-size-dependent threshold. Molecular viscosity 1e-6 m²/s. Its algebraic suspended residual is NOT transported or added to bedload.'),
    'capacity.smart_jaggi':dict(label='Smart & Jäggi',parameters=dict(ACal=4.,ThetaC=.05),
        note='Smart & Jäggi, energy-slope formulation (BASEHPC equations 1.29–1.31; not a FLOW formula). '
        'Slope J is computed from physical velocity, water depth and hydraulic resistance, not directly from the DEM. '
        'Measured Dm, D30 and D90; grading factor = (D90/D30)^0.2. Coefficient A = 4 (not a multiplier of 4); '
        'critical Shields = 0.05. Intended for steep gravel channels, J = 0.005–0.2; outside this range is extrapolation. '
        'No additional repose-angle correction or MPM form-drag reduction is applied.'),
    'capacity.ancey_recking_2023':dict(label='Ancey & Recking (2023)',parameters=dict(ACal=1.),
        note='Ancey & Recking (2023), Eq. 42, Bagnold master curve. Physical hydraulic shear and surface d50. '
        'Fixed dimensionless transitions 0.078 and 0.40; no hard critical threshold. '
        'One grain; bedload only. Multiplier 1 reproduces the published curve. '
        'Laboratory scaling; substantial natural-river discrepancies are reported by the authors. '
        'No MPM form-drag reduction or external hiding is applied.'),
    'capacity.recking_2010':dict(label='Recking (2010)',parameters=dict(ACal=1.,ReachSlope=.01),
        note='Recking (2010), main surface-based Eq. 12 with threshold Eq. 4. '
        'Measured surface D84 and D50, with their ratio in the published law; no sorting or armor model. '
        'Set representative reach slope S in m/m (0.01 = 1%); it is NOT calculated from instantaneous cell friction. '
        'Uses local hydraulic shear without changing the chosen hydraulic resistance. '
        'Field range S = 0.0002–0.08 and D84 = 0.9–558 mm; outside is extrapolation. '
        'No hard motion threshold. Exact branch intersection retains continuity.'),
    'capacity.camenen_larson_2005':dict(label='Camenen & Larson (2005)',parameters=dict(ACal=1.,VicMol=1.e-6),
        note='Steady-current, single-grain bedload only. Median D50 and physical hydraulic total shear. '
        'Published coefficients 12 and 4.5 are fixed; transport multiplier 1 reproduces the law. '
        'Critical Shields is calculated from D50, densities and kinematic water viscosity using '
        'Soulsby & Whitehouse (authors CR-07-1, Eq. 49). Smooth exponential tail, not an MPM hard cutoff. '
        'No waves, mixed-grain correction or suspended transport. No extra form-drag reduction.'),
}

from braids_core.river_jiang_cao import METHOD as JC_METHOD, DEFAULTS as JC_DEFAULTS
from braids_core.sediment.river_vr1993 import DESKTOP_PARAMETERS
CATALOG['capacity.van_rijn_1993']=dict(label='Van Rijn (1993) — FLOW 2DH',parameters=DESKTOP_PARAMETERS,
    note='Current-only FLOW 2DH reference-column law. Independent measured D50 and D90; '
    'with suspension enabled, also supply measured Ds. Matching bedload + suspended bundle or '
    'bedload alone; no waves, sorting or inferred percentiles. Standard coefficients are fixed in this release.')
CATALOG[JC_METHOD]=dict(label='Jiang–Cao 3/2 (research bundle)',parameters=JC_DEFAULTS,
    note='Non-equilibrium mobile bedload, NOT ordinary MPM/Exner. One bulk material: enter D50 and D90 independently. '
    'Automatic Sun/van Rijn resistance and bed-slope direction; finite-adjustment secondary flow '
    'and the explicitly named shallow-front treatment. MF = 1, sediment starts at 0. '
    'Enable secondary flow, set classic FLOW Betac/diffusion and eddy viscosity to zero. '
    'Use constant qs/q inlet supply. Repose projection is optional in Morpho evolution. '
    'This research composition has short CPU/GPU checks, not experimental braided-river braids_support.')
