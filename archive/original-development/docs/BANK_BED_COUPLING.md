# Bed-coupled bank evolution (development runtime)

The experimental persistent bank-front overlay is retired. The audit in
[current-model difference](../validation/sediment/jamieson2013_bend/reports/current_model_difference_audit/RESULTS.md)
found that it treated submerged slope redistribution as a moving vertical
front, suppressed ordinary Exner transport, and kept the hydraulic bed at its
activation elevation until whole-cell promotion.

## Supported representation

Banks and the channel bed use the same conservative sediment inventories.
For each cell, the bed is bedrock plus the active/substrate inventory divided
by each fraction's bulk density. A bank transfer lowers the donating bed and
raises the receiving bed immediately. The next hydraulic evaluation uses
these elevations. It does not artificially create water or momentum.

The distributed slope-bank option retains its existing constitutive law:
the accepted tangential bed-load transport in the receiving cell drives an
alpha-scaled transfer from a higher erodible neighbour above the configured
slope threshold. It is a slope-supply model, not a resolved overhanging wall.

Fully submerged slope movement contributes to the transferred-volume
diagnostic but no longer accumulates a wall-undercut trigger. Exposed-bank
transfers can retain the existing threshold-based release model. Each release
is inventory limited and cannot move its donor below its receiving bed: both
the donor's lowering and the recipient's rise enter this geometric limit.
This safeguard and volume conservation are numerical checks, not laboratory
validation of the collapse law.

The retired front's solid-area fraction is not used to claim a moving bankline.
Bank movement must be assessed from actual bed contours or cross-sections.
A genuine subcell moving-boundary model would additionally need consistent
water storage, hydraulic face openings and sediment/flow coupling.

## Saved projects and runs

- The experimental front checkbox is removed from the development QGIS UI.
- Loading a saved project with `front_tracking_enabled=true` disables that
  option in memory and records a compatibility note. The original project
  file is not overwritten. New preparation records the note in resolved inputs.
- Other bank parameters remain unchanged. New distributed-bank decks use v40.
- Creating a new project/deck with the retired option raises a clear error.
- The rebuilt native executable rejects old v41 decks before writing results.
  Prepare a new run; do not reuse a v41 checkpoint with the new physics.
- Archived results and their frozen executables are preserved. This change
  does not update the installed alpha-80 QGIS package by itself.

## Verification

`tests/test_bank_front_solver_integration.py` compiles a driver against the
production solver module. It checks conservative donor/receiver changes,
equality of hydraulic and inventory bed, a nonzero response in the next flow
step, absence of submerged-wall collapse, and the exposed-release cap.
The distributed-bank tests cover restart identity, thread identity, accepted
SSPRK/AMR flux forcing, and fraction conservation.

The fresh four-run bend comparison is controlled by
`validation/sediment/jamieson2013_bend/run_bed_coupling_milestone.py`.
Its frozen inputs and runtime are in `reports/bed_coupling_milestone_v1/`.
It compares stock Delft3D and BRAIDS bank-on/off trajectories at equal times;
it must not inherit a physical PASS from earlier runs. Delft3D source and its
locked runtime are never modified.
