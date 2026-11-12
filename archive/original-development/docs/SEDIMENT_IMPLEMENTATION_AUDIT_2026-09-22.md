# Sediment implementation audit — 2026-09-22

## Decision

BRAIDS already has a real production bed-load and Exner implementation. It is
not merely a user-interface placeholder: the QGIS run deck reaches the native
cut-cell solver, bed-load capacity is evaluated per fraction, transport is
reconstructed to shared faces, available sediment limits the donor flux, and
the conservative face-flux divergence updates bed inventory inside the coupled
SSPRK3 stages.

The implementation is nevertheless **not yet qualified as equivalent to the
current Delft3D source**. The code has grown faster than its validation evidence,
the production and legacy solver paths are insufficiently distinguished, and
the retained Delft3D comparisons do not all belong to the source snapshot now
available on this machine. Bed-load and Exner should therefore be treated as
implemented but under qualification, not as production-validated morphology.

## Audited reference

The untracked local archive `resources/Delft3D-main.zip` has SHA-256
`49485f22cc7e3a9ed25ecec44d52d34445e634550f59305c88fa620523fd0ddc`.
The extracted directory has no Git metadata, but audited sediment files match
Deltares/Delft3D commit
`c05ed672f0a7fe81c5b9867ef354cfdd53c25379` byte for byte. Its FLOW source
reports version 6.04.03. The exact critical-file hashes are recorded in
`validation/sediment/delft3d_current/source.lock.json`.

Neither the archive nor the extracted source is tracked by BRAIDS. This is a
hard repository policy, not merely a packaging exclusion.

## Production data path

1. The QGIS sediment panel writes fractions, exchange law, bed-load law,
   roughness, slope correction, active/substrate layers, boundary supply and
   bank parameters into the native run deck.
2. `solver/cut_cells/cutcell_solver.f90` computes water state and mixture
   statistics, hiding/exposure and formula-specific capacity per cell.
3. Capacity is reconstructed to faces. Certified Cartesian stencils use WENO;
   irregular cut cells use bounded least squares. Direction determines the
   donor; an opposing-direction case falls back to a bounded central choice.
4. Optional Delft3D-style Bagnold/Ikeda slope correction changes the face
   transport vector.
5. A shared face flux and donor limiter prohibit removal of more active-layer
   mass than is available. Boundary faces accept prescribed total supply or an
   equilibrium-capacity condition, per fraction.
6. Conservative divergence changes solid mass. Bed elevation is rock elevation
   plus stored solid volume divided by bulk density, with porosity represented
   through `rho_s * (1 - porosity)`.
7. The same update is evaluated within all SSPRK3 stages; active and substrate
   layers are rebalanced after stages. AMR includes sediment transfer and
   coarse/fine reflux paths.

The generic `braids_exner_mol_module` supplies reconstruction utilities to this
path, but its standalone scalar `braids_exner_step` is not the production bed
update. Likewise, `solver/wave/braids_bedload_module.f90` belongs to the retained
WAVE solver path, not the QGIS cut-cell production path. Documentation and tests
must make that distinction explicit.

## Capability comparison

| Capability | BRAIDS state | Delft3D analogue | Audit verdict |
|---|---|---|---|
| Van Rijn 1993 bed load | Implemented per fraction | `bedbc1993`, `bedtr1993` through `eqtran` | Rebaseline formula oracle and full case against the current lock |
| Other bed-load laws | MPM, Ashida-Michiue, Soulsby, Wilcock-Crowe, modified Wilcock-Crowe and Gaeuman implemented | `tranb2`, `trab12`, `trab14`, `trabwc`, `trabwc2`, `trabg` | Historical oracle evidence exists; current-source qualification required |
| Hydrodynamic forcing and roughness | Fixed or Manning-derived sediment roughness | Formula-specific inputs assembled by `eqtran` | Must isolate shear/roughness before judging transport-law agreement |
| Conservative Exner update | Face flux changes finite solid inventory within SSPRK3 | Morphological bed update in FLOW/FM | Strong design, but needs current-source budgets, convergence and bedform tests |
| Finite availability and non-erodible base | Active layer, substrate layers, rock floor, donor limiter | Bed-composition and availability machinery | Unit evidence exists; cross-model exhaustion/exposure case missing |
| Bed slope effect | Delft3D ISlope=2-style Bagnold/Ikeda option | Slope adjustments around `eqtran`/FM adjustment | Needs rotated-slope component tests and a full oblique case |
| Mixtures and hiding | Grain statistics plus all four Delft3D hiding relations | `compdiam`, `comphidexp` modes 2–5 | Current-source equation and three-fraction native sorting/armoring qualification passes |
| Stratigraphy | Constant or qualified water-depth-scaled active layer plus 1–32 ordered substrate layers and a finite updated base | Extensive Eulerian/Lagrangian/coarse/exchange-layer model | Paired armoring, buried-pulse and thickness sensitivities pass; vertical diffusion fails closed in the locked Delft3D runtime and is excluded; variable porosity remains a separately gated future option |
| Sediment boundaries | Prescribed total load or equilibrium capacity, per fraction, with ledgers | Delft3D equilibrium/prescribed sediment boundaries | Requires supply-step, rotation and restart qualification |
| Wet/dry and arbitrary boundaries | Cut-cell faces and dry thresholds | Structured/unstructured wetting and drying | Particularly important for bars; current whole-event oblique evidence is insufficient |
| AMR morphology | Conservative transfer/reflux paths; physical boundaries remain finest | No direct one-to-one numerical counterpart | Compare BRAIDS AMR to BRAIDS fixed fine grid, and both physical results to Delft3D |
| Morphological acceleration | Not implemented | Constant/time-varying `MorFac`, `updmorfac` | Add only after `MorFac=1` is qualified; then validate acceleration error |
| Secondary/spiral flow | Qualified equilibrium curvature/spiral steering for magnitude-based bed-load laws; direction only, with `BetaSpiral=0` | `get_curvature`, `equili_spiralintensity` and the direction correction in `eqtran` | All 15 frozen mild-bend/bifurcation gates pass against locked current-source D-Flow FM; momentum forcing and Coriolis remain explicit non-goals |
| Non-equilibrium bed load | No adaptation-length model | Delft3D supports more advanced transport handling | Needed where steep, rapidly varying flow invalidates local equilibrium |
| Avalanching | Qualified conservative wet/dry repose redistribution across every retained mobile-bed cell, plus the older separately scoped bank-zone mode | General dune avalanche/repose machinery, including `duneaval` | All 28 frozen paired wet/dry gates pass; this does not represent toe undercutting or cantilever collapse |
| Lateral bank erosion | Conservative wet-cell erosion supplied from adjacent higher dry-bank cells | Delft3D dry-bank redistribution and slope processes | Useful prototype, but not a calibrated undercut/collapse bank model |
| Bedform roughness | Not implemented | Delft3D bedform/roughness options | Later braided-river sensitivity feature |
| Suspended/noncohesive exchange | Implemented and operator-split after transport | Integrated source/sink and transport framework | Deliberately deferred until bed load and Exner pass |
| Cohesive processes | Basic cohesive exchange | Flocculation, fluff, consolidation and burial options | Specialist later phase; not needed for the first gravel-bed braid milestone |

## Evidence that remains useful

The repository already contains substantial validation, not an empty test base.
Historical intercomparisons include bed-load basins, formula combinations,
expansion flumes, a sorting/armoring case, settling, reservoir-scale cases and
Zhu-style migrating-bed tests. Several report strong agreement: formula-scale
capacity errors below one percent in selected cases, high spatial correlations
in the expansion flume, conservative BRAIDS budgets, and a passing two-fraction
sorting case.

Those results demonstrate plausible implementation and useful test designs.
They do not, by themselves, qualify the present code/reference pair. In
particular, the coupled Zhu case is close to its acceptance limits at 70 hours,
the cut-cell qualification covers only a late narrow window, and full
reservoir-scale morphology retains material differences and resolution
sensitivity.

## Blocking findings

### Resolution status

Findings 1–3 are now controlled for the new comparison line: historical
evidence remains immutable, the matching current-source Delft3D4 executable has
been built and cryptographically locked, optional validation dependencies are
declared, and the focused sediment regression set passes on Windows. The old
records remain historical rather than being relabelled. Findings 4–5 remain
active design work, and no sediment-physics qualification claim has yet been
made from the hydrodynamic smoke case.

1. **Source provenance drift.** Of 105 JSON records that refer to Delft3D source
   hashes, 84 disagree with the current local source. Existing preflight and the
   Van Rijn verification refuse to run for this reason. Old evidence must remain
   immutable and be labelled historical; it must not be silently relocked.
2. **Binary/source mismatch.** Some retained cases use the pinned
   `eawag/delft3d-flow` 6.03.00.62434 image, while the audited source is FLOW
   6.04.03 at `c05ed672…`. A result is admissible only when both executable and
   source provenance are recorded and consistent.
3. **Current sediment suite is not green.** In the available Windows environment,
   a focused run produced 88 passes and 4 failures. Three are Windows atomic
   replacement failures in the legacy native exchange-ledger restart test. One
   current comparison contract imports missing `xarray`. Other collections need
   `xarray` or `shapely`. These are partly harness defects, but they block a clean
   qualification statement.
4. **Multiple numerical paths are easy to confuse.** Production cut-cell,
   standalone Exner and legacy WAVE reports are mixed in documentation. Tests
   must name the executable and path they qualify.
5. **The user interface exposes maturity unevenly.** Many transport choices are
   selectable without showing validation status, applicable grain regime,
   required parameters, or the exact mass-budget result. A compact “Sediment
   readiness and diagnostics” panel should show those facts rather than imply
   feature parity.

## Immediate conclusion

Do not add suspended sediment, a new bank model or morphological acceleration
to the production promise yet. First restore reproducibility, qualify each
bed-load prerequisite, and qualify Exner on small cases at `MorFac=1`. The
ordered program is defined in `BRAIDED_RIVER_MORPHODYNAMICS_PLAN.md`.
