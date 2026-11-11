"""Named BRAIDS default: calibrated MPM-like variant, never a FLOW alias.

Coefficient5/critical.055/exponent1.5 follows the Saiedi ST_2 reference in
BASEMENT4.2's Testcases manual. Independent physical evidence remains separate
from that calibrated case. One measured D50, solid-volume bedload only.
"""
METHOD='capacity.mpm_braids_5_055'
COEFFICIENTS=dict(ACal=5.,PowerB=0.,PowerC=1.5,RipFac=1.,ThetaC=.055,
    SusACal=0.,SusPowerB=3.,SusPowerC=0.,SusRipFac=1.,SusThetaC=0.)
LABEL='MPM — BRAIDS standard'
DESCRIPTION=('Fixed MPM-like relation: 5 × max(Shields − 0.055, 0)^1.5. '
    'Measured D50; physical velocity and cell-depth hydraulic resistance, without a FLOW '
    'effective-velocity or grain/form-drag correction. The selected Manning, Chezy or '
    'White–Colebrook resistance is retained. No wall-stress partition is inferred. '
    'Coefficients originate from the calibrated Saiedi/BASEMENT reference; this is a '
    'BRAIDS standard variant, not original MPM or universal experimental validation. '
    'Currently one bulk material, bedload only, finite mixed bed, MF1 and static mesh; '
    'secondary flow, turbulence, slope corrections, banks, collapse and bedforms are not yet qualified with this variant.')
