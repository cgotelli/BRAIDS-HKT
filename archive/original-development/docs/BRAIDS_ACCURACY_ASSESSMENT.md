# How to judge BRAIDS and Delft3D predictions

**Current resource-DEM status (11 September 2026):** physical inlet geometry is fixed. A consistent bilinear DEM surface and water-driven bedload closure at GeoClaw's dry-step walls now correct two additional demonstrated defects. [Proof, source comparison and measured scope](../validation/sediment/reservoir_polygon/geometry_resolution/reports/CLOSED_DRY_FACE_BEDLOAD.md). The fresh `closed_face_reservoir` study retains the same forcing, cells, footprint and 5% inventory / 10% deposit-map limits. Its full-event qualification remains pending. The earlier 5 m timestep control is complete and changes the four quantities by only 0.048 / 0.338 / 0.380 / 0.622%, much less than the retained failed spatial comparison. Historical inputs/builds are kept separate. [Current execution status](../validation/sediment/reservoir_polygon/geometry_resolution/PROGRESS.md).

We cannot yet identify the more accurate prediction for the current reservoir.
The **13.16% difference between positive lake deposits is an intermodel
difference, not a measured error**. Halving the grid spacing changes those
deposits by 31.8% in BRAIDS and 34.7% in FLOW. That sensitivity prevents a
defensible ranking from the present pair of grids. See the [matched combined
report](../validation/sediment/delft3d_intercomparison/reservoir_scale/matched_combined/reports/MATCHED_COMBINED.md).

## What evidence would distinguish them?

| Question | Independent evidence | Decision |
|---|---|---|
| Is the numerical implementation solving the intended equations? | Exact/manufactured solutions, independently evaluated transport laws, and known balances | Compare each solver with the reference under its stated assumptions. Smaller error supports that implementation for that case. |
| Is transport sufficiently low in numerical diffusion? | A pulse or bedform with known evolution and zero physical dispersion | Measure peak loss, spreading, front position and profile error together. Also check overshoots, nonnegative inventories and conservation; a sharper curve alone is insufficient. |
| Is the reservoir prediction adequately resolved? | Systematically refined space and time, with matched physical inputs and accepted boundary supplies | Estimate numerical uncertainty separately for each solver and each output quantity. |
| Does the sediment model represent the physical reservoir? | Measured bed profiles, trapped/exported mass and concentration histories, with measurement/input uncertainty | Compare independently validated predictions against observations. This is the experimental stage, after the plugin is fully functional. |

Refinement can estimate numerical error even when an exact reservoir solution
does not exist. Three systematically related grids allow an observed order of
convergence to be estimated and checked. Richardson extrapolation or a Grid
Convergence Index is useful only where its assumptions are supported; we must
not force an error bar onto nonconvergent results or a discontinuous local
profile. Time-step error must also be checked. This follows the [NASA/NPARC
grid-convergence guidance](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html).

Physical validation requires observations and their uncertainty, beyond
numerical convergence. A model can converge accurately to equations that
incompletely describe the experiment. See the [NASA/NPARC validation
guidance](https://www.grc.nasa.gov/www/wind/valid/tutorial/valassess.html).

## Apply this to the existing evidence

- BRAIDS's water and sediment conservation passes the current native checks.
  Conservation is necessary, but it does not establish the right spatial
  distribution of deposits.
- The [common-state flux audit](../validation/sediment/delft3d_intercomparison/reservoir_scale/matched_bedload/common_state_flux/reports/COMMON_STATE_BED_FLUX.md)
  found lower BRAIDS flux errors on an independent smooth spatial reference
  than the tested upstream-cell transport evaluation. This supports the
  reconstructed transport on that test; it does not rank the full reservoir
  predictions. Its original phase-sensitive refinement failure remains.
- The Zhu and pulse checks assess particular assumptions, regimes and grids.
  They do not guarantee accuracy at every reservoir transition or wetting
  front. Existing limitations remain in the roadmap.
- FLOW and BRAIDS use different water/bed displacement conventions in the
  inspected wet-cell update. Their bedload supplies also respond to their own
  evolving hydraulic states. Those differences must be accounted for before
  attributing the entire discrepancy to sediment diffusion.
- Positive local deposition and net lake bed change answer different
  questions. The latest case has local deposits and net lake erosion in both
  models. Neither quantity should be substituted for the other.

## Practical decision rule

Choose target quantities before running: net bed-volume change, positive
deposition, bed-profile error and maximum thickness, deposit/front position,
trapping efficiency, exported sediment, and water/solids balance. Retain
absolute errors as well as relative errors when the retained mass is small.

For verification cases, prefer the solver that meets the reference tolerances
with lower numerical error at a comparable computational cost. For a physical
application, prefer predictions consistent with independent measurements
within numerical, input and measurement uncertainty. If uncertainty prevents
distinguishing the predictions, report that limitation rather than declaring
a winner.

The original [evolving pulse/Exner assessment](../validation/sediment/delft3d_intercomparison/evolving_pulse/reports/EVOLVING_PULSE_EXNER.md) provides a retained baseline for that independent comparison. At 0.5 m the baseline BRAIDS concentration error is 2.0956% and peak loss is 4.1756%; FLOW has 0.60958% concentration error and 0.31819% peak loss. BRAIDS preserves nonnegative inventories and mass to roundoff, while this FLOW configuration has negative concentrations and a −0.02668% solid residual. Neither original configuration passes the complete frozen criteria. At smaller actual timesteps, FLOW’s concentration error is 0.33444% and BRAIDS’s is 2.08920%; BRAIDS’s peak damping persists. This is evidence that the tested FLOW configuration transported this particular smooth pulse more accurately than the original BRAIDS build, together with evidence of its separate bounds/balance shortcomings. It is not a ranking of reservoir predictions.

The reference is exact for prescribed uniform flow; the coupled runs use a dilute pulse, with hydraulic departures measured below 1.7e−6 relative. It tests suspended advection, settling and Exner deposition under those assumptions, not bedload, wet/dry fronts or finite-amplitude reservoir morphology. Negative concentrations also make a small signed variance error an unreliable standalone measure of diffusion. Use profile errors, peak loss, bounds and balances together.

The subsequent [native limiter correction](../validation/sediment/delft3d_intercomparison/evolving_pulse/peak_limiter/reports/PEAK_LIMITER_CORRECTION.md) closes BRAIDS’s pulse-specific failure: concentration error falls to **0.13367%**, peak loss to **0.04422%**, and deposited-inventory error to **0.02674%** on the same 0.5 m grid. The conservative limiter had flattened resolved extrema; guarded finite-volume bounds preserve them. All 42 final criteria pass, including nonnegative inventories, accepted budgets and genuine timestep sensitivity. Smaller actual timesteps give 0.12941% concentration error. Corrected BRAIDS is therefore more accurate than the tested FLOW settings **for this smooth dilute fixture**, while also passing the recorded bounds/balance checks. This does not rank the reservoir predictions or close the distinct bedload, irregular-cell and sharp-transition limitations.

The [corrected-solver resource application](../validation/sediment/reservoir_polygon/peak_limiter_application/reports/RESOURCE_RESERVOIR_APPLICATION.md) now completes 600 s and passes all 35 application checks. Suspended sediment is present in the lake by the 450 s snapshot; at 600 s there are 20.831 t of positive lake bed changes and 12.370 t of net bed gain. Water/solids conserve, and all five saved states and boundary ledgers are exactly identical to an immediate pre-correction control with matched physical inputs. Thus the pulse improvement preserves this coarse reservoir result; it does not newly qualify reservoir diffusion or deposition accuracy.

The [20/10 m resource comparison](../validation/sediment/reservoir_polygon/grid_comparison/matched_stage/reports/GRID_COMPARISON.md) finds a separate inlet/terrain limitation. At the original level the fine opening implies 73.24 m/s, while cells are capped at 50 m/s; that preflight remains failed. The common-stage control uses a new, identical level on both grids and passes 21/21 execution/conservation checks. Positive lake bed changes differ by 22.23% between grids. This is spatial sensitivity of that control, not a passed original-case refinement or a measured error. Changing the stage alone also changes coarse positive deposits by 25.60%, so the control cannot be substituted for the original case.

The [native inlet-speed correction and QGIS diagnostics](../validation/hydraulics/cut_cells/inlet_speed/reports/INLET_SPEED_DIAGNOSTICS.md) now reject the original fine setup at preparation and direct native execution, with no accepted water or sediment flux. Valid compact states/ledgers remain unchanged and thread-identical. This closes the speed-check inconsistency; it does not resolve the represented-section or reservoir accuracy limitations.

The source-DEM diagnosis now identifies a concrete geometry defect: the original level lies below the physical inlet edge, while downhill owner-cell averages created a false opening. Edge sampling with connected wetting corrects that inconsistency without changing cells. A separately declared critical-depth-derived level supplies the new study; the original input is retained and rejected. Native traces confirm the actual Q and horizontal level match across grids.

The follow-up identified [two further hydraulic defects](../validation/sediment/reservoir_polygon/geometry_resolution/reports/HYDRAULIC_SOURCE_CORRECTIONS.md): dry hillside elevations were reconstructed as higher water surfaces, and the intermediate bed-source quadrature generated a spurious force on curved terrain. Dry-land surface extension restores the stationary lake; actual-column source quadrature corrects the curved-bed reference and is applied consistently at sampled inlets. The final 10 m simple-wave profile error is **0.4415%**, versus **26.21%** with the original wet-cell flattening. The intermediate 0.2235% result remains recorded but does not describe the final source quadrature. The complete matched reservoir space/time study is now rerunning with these fixes and the original frozen limits. Conservation and analytical verification alone do not establish reservoir convergence. Follow the [current evidence](../validation/sediment/reservoir_polygon/geometry_resolution/PROGRESS.md); experimental validation remains after functional plugin delivery.
