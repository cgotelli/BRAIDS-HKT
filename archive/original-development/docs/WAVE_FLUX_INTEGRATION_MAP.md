# WAVE accepted mass-flux integration map

Source audit: 6 September 2026, against the extracted WAVE and modified Clawpack tree recorded in [source_provenance.json](../solver/source_provenance.json). **G3 remains open.** This document identifies integration points; it does not implement or verify numerical inlet flux.

The physical-boundary file in this executable is `solver/wave/bc2amr.f90`, not `bc2.f90`. The [WAVE Makefile][make-wave] substitutes that local routine, while retaining the shared GeoClaw `advanc.f`, `stepgrid.f`, `step2.f90`, and `flux2fw.f`. Check the [actual source list][make-geoclaw] before selecting a similarly named AMRClaw routine: the executable uses GeoClaw's `tick.f` and `update.f90`.

## 1. The actual call chains

```text
GeoClaw tick
  prepare_advanc
    bound, once per patch in deterministic order
      filrecur / interpolation
      WAVE bc2amr → apply_hydraulic_bc
    saveqc → additional bc2amr calls for coarse boundary data
  select_cfl_timestep
    cfl_preflight → par_cfl_preflight
      b4step2 on private copies
      step2_cfl → rpn2_cfl             [may repeat; no physical update]
  advanc_prepared                     [accepted shared level timestep]
    par_advanc, parallel over patches
      b4step2 on live state
      qad, if fine/coarse registers exist
      stepgrid
        step2
          x and y sweeps → flux2 → rpn2 / rpt2
          combine normal + transverse corrections
          optional positivity relimiting
        update live q from fm/fp/gm/gp
        clip negative depth; zero dry momentum
        src2                          [optional shoreline source, friction, etc.]
      fluxsv / fluxad, if registers exist
      advance the patch clock
  [after all required fine substeps] update → upbnd + fine-to-coarse restriction
```

The accepted sequence starts in [tick][tick-accept]; live patch execution is in [par_advanc][par-advanc]. The [boundary-fill implementation][bound] explicitly calls `bc2amr` after filling four ghost strips; recursive [filpatch][filpatch] and [saveqc][saveqc] can also call it. `bc2amr` invocation count is therefore not a timestep or water-volume counter.

**A `stepgrid` call alone also does not prove physical acceptance.** Initialization uses [setgrd → advanc][setgrd], and Richardson/error-estimation paths call [prepregstep][prepregstep] and [prepbigstep][prepbigstep] → `stepgrid`. Arm a ledger context only around the accepted `tick → advanc_prepared` call. Keep it off during initialization, refinement probes, and CFL trials. `actualstep=.true.` in `b4step2` is insufficient: the CFL probe deliberately uses that value on private copies.

## 2. Arrays and the mass-flux reconstruction

`q(1)=h`, `q(2)=hu`, `q(3)=hv`; `aux(1)=bed elevation`. Positive `hu` and `hv` point toward increasing x and y. In [stepgrid][stepgrid-update], the enlarged arrays use indices `1:mitot,1:mjtot`; interior cells are

```text
iL = mbc+1       iR = mitot-mbc
jB = mbc+1       jT = mjtot-mbc
```

The same memory is indexed inside `step2` as `1-mbc:mx+mbc`, with interior `1:mx,1:my`. Face `i` is between cells `i-1` and `i`; face `j` is between rows `j-1` and `j`.

### The important distinction in this WAVE build

WAVE leaves `use_fwave_positivity_limiter=.false.` at its [module default][geo-flags]. On this path, the arrays called “fluxes” contain **fluctuations plus higher-order corrections**, not absolute interface mass fluxes. [flux2fw][flux2-normal] starts with `faddm=amdq` and `faddp=-apdq`; both receive the same normal correction. [step2][step2-sweeps] adds the transverse contributions from both sweeps.

For the Cartesian case (`coordinate_system=1`, `mcapa=0`), the physical numerical mass flux at an x-face is reconstructed as

```text
Fhat_left  = fm(1,i,j) + q_pre(2,i-1,j)
Fhat_right = fp(1,i,j) + q_pre(2,i,j)
```

and at a y-face as

```text
Ghat_below = gm(1,i,j) + q_pre(3,i,j-1)
Ghat_above = gp(1,i,j) + q_pre(3,i,j)
```

Here `q_pre` is the **live state after `b4step2`, before the hyperbolic update**. These equations follow from `Fhat=f(q_left)+A⁻Δq=f(q_right)−A⁺Δq`, including the accumulated corrections. The augmented Riemann solver retains the mass jump `delhu` in its [decomposition][riemann-decomposition]; bed source terms enter other components. Check the two reconstructed values against one another rather than assuming equality under every dry-wall branch or roundoff condition.

A constant uniform throughflow can have `fm=fp=gm=gp=0` while `hu` is nonzero. Reporting the raw arrays would incorrectly report zero inlet discharge. Reconstructing from the **post-update** momentum would also be wrong.

For an exterior face, use the reconstruction from the adjacent interior cell:

| Physical side / existing side number | Positive inward normal | Accepted inward rate per unit face length | Face length |
|---|---|---|---|
| West / 1 | (+1,0) | `fp(1,iL,j) + q_pre(2,iL,j)` | `dy` |
| East / 2 | (−1,0) | `−fm(1,iR+1,j) − q_pre(2,iR,j)` | `dy` |
| South / 3 | (0,+1) | `gp(1,i,jB) + q_pre(3,i,jB)` | `dx` |
| North / 4 | (0,−1) | `−gm(1,i,jT+1) − q_pre(3,i,jT)` | `dx` |

Sum only the side's interior transverse indices. In SI Cartesian coordinates these quantities are m²/s; multiply by face length and the accepted `dt` to obtain m³. Negative values represent water leaving through an inlet face and must remain signed.

If the positivity limiter is enabled in a later change, [flux2fw][flux2-normal] already converts component 1 to a true interface flux, and [step2's final relimiting][step2-relimit] modifies it. On that path, read the final component-1 arrays directly, with the side signs above, **without adding momentum again**. Do not change the numerical path merely to simplify diagnostics. Geographic/capacity cases require the existing metric factors and capacity-weighted volume; they are outside the first implementation slice.

## 3. Where to observe and commit

The smallest useful observer belongs **immediately after `step2` returns in [stepgrid][stepgrid-call], before the update loop**. At that point all normal, transverse, limiter and timestep-dependent corrections are assembled, and live pre-update momentum is still available. Observing inside `rpn2` or `flux2` is too early to capture the final multidimensional update.

With the accepted-step context armed by `tick`, collect patch-local pending records containing:

```text
accepted_step_id, t_start, dt, level, patch_id,
axis, global_face_index, transverse_cell_index, bc_id,
face_length, inward_mass_flux_per_length, volume_increment
```

Also capture interior water volume just before and immediately after the flux update; capture changes from depth clipping and `src2` separately. `b4step2` can clear negative depth and cap momentum, so the broader budget should eventually record its live-state changes too. [stepgrid's cleanup][stepgrid-cleanup] explicitly acknowledges that negative-depth clipping can add water. WAVE [src2][src2] can add depth through the optional internal shoreline source and spherical terms; fixed-bed Cartesian G3 cases should disable those sources and still report a source delta.

Commit patch records in deterministic patch/face order after the accepted `advanc_prepared` level call returns. Its [OpenMP patch loops][advanc-parallel] use dynamic scheduling: avoid unsynchronized shared counters and worker file writes. A read-only level context plus one pending buffer per patch is a small, reviewable design. Commit zero-flux accepted patches too, so an empty patch is not confused with an omitted observation.

For the first fixed-grid case, verify independently:

```text
ΔV_hyperbolic = dt × Σ(exterior inward numerical flux × face length)
ΔV_total     = ΔV_hyperbolic + ΔV_clipping + ΔV_source + ΔV_preparation
```

The first identity should be checked before cleanup masks its residual. Interior reconstructed left/right flux mismatches and same-level patch-seam mismatches are diagnostics; do not subtract them silently to manufacture conservation. Once the measurement is proven, compare requested Q, constructed ghost/state Q, and accepted Q as three distinct outputs.

## 4. Face ownership and the existing benchmark boundary

Use integer hierarchy indices to recognize physical faces: node `ndilo==0` for west, `ndihi==iregsz(level)-1` for east, and analogous `ndjlo/ndjhi/jregsz` checks for south/north. These are the [actual domain index conventions][physical-flags]. Exclude periodic boundaries. Cross-check real patch extents against the expected domain edge during development.

On a single level, a physical face belongs to the patch owning its adjacent interior cell. A key such as `(level,axis,global_face_index,transverse_cell_index)` distinguishes it independently of temporary patch numbering. Map that key to exactly one stable `bc_id`; reject conflicting ID claims. Include only interior transverse cells, never ghost rows/columns or a neighboring patch's physical face. Corner points have zero face length. A polyline endpoint rule must partition neighboring face intervals consistently.

The existing [hydraulic_bc_module][hydraulic-bc] is a benchmark helper, not the G2 section adapter:

- It stores four side modes, with no per-polyline ID or per-face ownership table.
- Total-discharge modes compute area over the current patch and apply the supplied total to that patch. Splitting one physical inlet across patches can apply Q repeatedly; the production normalization must use all owned faces for the ID.
- Its prescribed momentum is in positive x/y coordinates. A positive value is not automatically inward on east/north; convert from the explicit physical normal.
- `bc2amr` receives time, but the current helper does not. Hydrograph evaluation needs an explicit time contract before time series can work.
- `b4step2` may change constructed ghost momentum before the Riemann solve. The observer must measure that actual pre-update state.

Start with full axis-aligned faces and one fixed patch. Add two IDs and split patches only after this ownership and normalization rule is implemented. Arbitrary polyline cuts need a separate geometric mapping decision; multiplying a whole-face flux by a polyline fraction does not create a correctly imposed partial-face boundary.

## 5. AMR synchronization is a separate budget layer

The currently linked register path is:

| Point | Stored/applied information |
|---|---|
| `prepare_advanc → saveqc` | Coarse boundary states for subsequent wave correction. |
| `par_advanc → qad`, before stepping | Coarse/fine Riemann correction into the fine-owned register. |
| `par_advanc → fluxsv`, after stepping | Coarse contribution into registers owned by neighboring fine patches; ordered to prevent concurrent writes. |
| `par_advanc → fluxad`, after stepping | Fine substep contributions weighted by fine `dt` and face lengths. |
| `tick → update → upbnd`, at synchronization | Apply register corrections to uncovered coarse neighbors; then restrict fine values into covered coarse cells. |

See [qad][qad], [fluxsv][fluxsv], [fluxad][fluxad], and the linked [GeoClaw update.f90][update]. Register side order is **left, top, right, bottom**, unlike hydraulic side numbering **west, east, south, north**. Reuse the source's explicit mapping, not the numeric labels interchangeably.

An exterior AMR ledger must count the finest active physical face over each accepted substep and suppress the covered coarse-face contribution. Composite storage likewise excludes covered coarse cells. Keep coarse/fine interface correction accounting separate from external inflow; register sums include wave corrections and are not a substitute for per-ID physical boundary measurements. Refresh coverage and ownership at regridding, and checkpoint both accumulated volumes and event identity before claiming restart continuity.

Two existing policies require explicit diagnostics before a strict AMR conservation claim:

- [upbnd][upbnd] skips corrections into dry coarse cells and can zero a small positive depth afterward.
- WAVE leaves `conserve_depth_amr=.false.`. [update.f90][update-restriction] uses the legacy wet-cell surface-elevation restriction on that path, not unconditional conservative depth averaging. Regridding/interpolation has its own terrain and wet/dry effects to qualify.

Measure changes around correction, restriction and regridding separately. A perfect exterior flux ledger cannot by itself certify composite-domain mass conservation.

## 6. Small first implementation and unresolved checks

Implement only an **opt-in Cartesian, fixed-level accepted-step observer** at the points above, leaving the current boundary states unchanged. Begin with a short uniform rectangular throughflow: known nonzero Q, identical wet states, and zero normal/transverse fluctuations make the raw-array mistake immediately visible. Then use a closed box and a transient wet inlet to exercise the volume identity. This provides the measurement needed to judge the later section-based boundary implementation.

Before declaring that slice qualified:

- [ ] Confirm reconstructed left/right mass flux equality on interior wet faces, topographic steps and dry walls; preserve any residual.
- [ ] Confirm stage/velocity snapshots are taken after `b4step2` and before the update, with all higher-order/transverse corrections included.
- [ ] Confirm only accepted `tick` steps commit rows; initialization, repeated boundary fills and rejected CFL trials add no volume.
- [ ] Check all four inward signs, zero-Q and reversal, then one versus multiple OpenMP workers.
- [ ] Verify the hyperbolic volume identity separately from cleanup/source changes.
- [ ] Add global per-ID face mapping and normalization before split-patch or multiple-inlet claims.
- [ ] Leave AMR, restart, arbitrary oblique boundaries and complete G3 qualification open until their separate checks pass.

## 7. Known Thacker mass issue: candidate mechanisms and diagnostic counters

This section concerns the current extracted WAVE reproduction. The [original notebook audit](THACKER_NOTEBOOK_AUDIT.md) found no historical water-volume record. Compare untouched and extracted builds under matched conditions before assigning the gain to pre-existing solver behavior or changing numerical physics specifically to fix it. Following the user's 7 September decision, inlet implementation proceeds while this investigation remains open; accepted-flux reporting supports both tasks.

**Observed, not yet attributed:** [the native-volume diagnostic][thacker-mass] finds water increasing from `0.15707993600000025` to `0.16073902437282672 m³` over three periods: **+0.00365908837282647 m³, or +2.3294435%**. Independent integration of fixed-grid output shows the same gain; its final volume differs from the native result by only `2.59e-8 m³`. This is not explained by fixed-grid output interpolation alone.

The saved [claw.data][thacker-claw], [geoclaw.data][thacker-geo], [amr.data][thacker-amr] and [setprob.data][thacker-source] specify a 100×100 Cartesian grid at 0.04 m, capacity index 0, one AMR level, four reflecting walls, no internal inflow, and no friction or Coriolis forcing. `sphere_source=1` is present, but its mass source is inactive because `coordinate_system=1`. The native initial frame contains one patch. AMR reflux/restriction cannot account for this level-1 result.

| Candidate mechanism | What the source establishes | Counter needed before assigning cause |
|---|---|---|
| Negative depth after the hyperbolic update | [stepgrid cleanup][stepgrid-cleanup] replaces negative `h` by zero, adding exactly the removed negative volume. The positivity relimiter is disabled. This is a direct mechanism with the correct sign, **not a measured explanation yet**. | Immediately before cleanup: interior `min(h)`, negative-cell count, and `Σ max(−h,0) dx dy`; record the first occurrence and cumulative added volume. |
| Live pre-step cleanup | [b4step2][b4step-cleanup] also clips negative `h` before stepping. It retains positive shallow depths and changes their momentum only. | Interior water-volume difference across live `b4step2`; separate negative-depth count from momentum-zeroing and speed-cap counts. Do not count its private CFL-probe copies. |
| Dry Riemann preprocessing | [rpn2][riemann-dry] zeroes negative input states on the copied 1D slice, treats `h<=dry_tolerance` as dry locally, and mirrors/filters waves at a dry topographic wall. This can affect the computed update but does not itself overwrite `qold` in this `step2` path. | Negative-slice-input and dry-wall branch counts, plus mass-fluctuation compatibility residuals, classified by wet/wet, wet/dry and dry/dry faces. |
| Interior mass-flux incompatibility | For each face the two reconstructions in Section 2 should agree. Normal and transverse corrections are added equally to each face's minus/plus arrays; nevertheless the final residual must be measured, especially at dry walls. | Sum signed `dt × face_length × (Fhat_right−Fhat_left)` on x-faces, and the analogous above-minus-below quantity on y-faces; also store maximum absolute mismatch. This is the interior contribution to water gain. |
| Exterior wall leakage | [bc2amr][wave-bc] mirrors depth/bed and reverses normal momentum. That is the intended wall construction; no accepted-flux measurement yet proves zero water exchange. | Reconstructed accepted flux on each of the four physical sides after both sweeps. Exclude all ghost-only faces. |
| Explicit source or momentum handling | [src2][src2] has two depth-changing mechanisms: internal inflow and the spherical mass term. Both are inactive for these saved inputs; speed caps and dry momentum zeroing change momentum, not depth directly. | Interior volume difference across `src2`, plus `inflow_ncells`, coordinate/source flags and speed-cap counts. An unexpected nonzero source delta warrants a separate investigation. |

Do not label retention of `0<h<dry_tolerance` as water creation or deletion: both `stepgrid` and `b4step2` retain that positive depth. Do not infer an absence of post-update negatives from saved nonnegative output, because cleanup has already run. The [existing solver log][thacker-log] has no `Negative input:` messages and records 198 rejected CFL trials; neither observation measures post-update clipping or proves a retry-related mass defect.

The decisive first diagnostic is an accepted-step balance that closes the measured gain:

```text
ΔV = V_wall_flux + V_interior_flux_mismatch
   + V_postupdate_clip + V_prestep_clip + V_source + residual
```

Capture the pre-clip state while it still exists, and report every term independently. Correlate the first gain with shoreline faces, timestep/CFL, and momentum-cap activity only after these counters identify the responsible term. Preserve this baseline and its numerical settings; do not enable the positivity limiter or alter dry tolerances as a substitute for attribution. No new native run or Fortran change was made for this audit.

[make-wave]: /Users/cmgotelli/Desktop/BRAIDS/solver/wave/Makefile:54
[make-geoclaw]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/geoclaw/src/2d/shallow/Makefile.geoclaw:114
[tick-accept]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/geoclaw/src/2d/shallow/tick.f:285
[par-advanc]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/geoclaw/src/2d/shallow/advanc.f:354
[bound]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/amrclaw/src/2d/bound.f90:159
[filpatch]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/geoclaw/src/2d/shallow/filpatch.f90:535
[saveqc]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/amrclaw/src/2d/saveqc.f:98
[setgrd]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/geoclaw/src/2d/shallow/setgrd.f:44
[prepregstep]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/geoclaw/src/2d/shallow/prepregstep.f:31
[prepbigstep]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/geoclaw/src/2d/shallow/prepbigstep.f:96
[stepgrid-call]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/geoclaw/src/2d/shallow/stepgrid.f:182
[stepgrid-update]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/geoclaw/src/2d/shallow/stepgrid.f:203
[stepgrid-cleanup]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/geoclaw/src/2d/shallow/stepgrid.f:234
[geo-flags]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/geoclaw/src/2d/shallow/geoclaw_module.f90:55
[flux2-normal]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/geoclaw/src/2d/shallow/flux2fw.f:146
[step2-sweeps]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/geoclaw/src/2d/shallow/step2.f90:135
[step2-relimit]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/geoclaw/src/2d/shallow/step2.f90:215
[riemann-decomposition]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/riemann/src/geoclaw_riemann_utils.f:204
[advanc-parallel]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/geoclaw/src/2d/shallow/advanc.f:130
[src2]: /Users/cmgotelli/Desktop/BRAIDS/solver/wave/src2.f90:50
[physical-flags]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/amrclaw/src/2d/setPhysBndryFlags.f:35
[hydraulic-bc]: /Users/cmgotelli/Desktop/BRAIDS/solver/wave/hydraulic_bc_module.f90:58
[qad]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/geoclaw/src/2d/shallow/qad.f:137
[fluxsv]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/amrclaw/src/2d/fluxsv.f:24
[fluxad]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/amrclaw/src/2d/fluxad.f:33
[update]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/geoclaw/src/2d/shallow/update.f90:66
[update-restriction]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/geoclaw/src/2d/shallow/update.f90:148
[upbnd]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/geoclaw/src/2d/shallow/upbnd.f:106
[thacker-mass]: /Users/cmgotelli/Desktop/BRAIDS/validation/hydraulics/wave_baseline/reports/thacker_native_mass.json
[thacker-claw]: /Users/cmgotelli/Desktop/BRAIDS/validation/hydraulics/wave_baseline/runs/04_thacker_planar_paraboloid/Wave/claw.data:28
[thacker-geo]: /Users/cmgotelli/Desktop/BRAIDS/validation/hydraulics/wave_baseline/runs/04_thacker_planar_paraboloid/Wave/geoclaw.data:12
[thacker-amr]: /Users/cmgotelli/Desktop/BRAIDS/validation/hydraulics/wave_baseline/runs/04_thacker_planar_paraboloid/Wave/amr.data:9
[thacker-source]: /Users/cmgotelli/Desktop/BRAIDS/validation/hydraulics/wave_baseline/runs/04_thacker_planar_paraboloid/Wave/setprob.data:7
[thacker-log]: /Users/cmgotelli/Desktop/BRAIDS/validation/hydraulics/wave_baseline/runs/04_thacker_planar_paraboloid/Wave/solver.log:59
[b4step-cleanup]: /Users/cmgotelli/Desktop/BRAIDS/solver/wave/b4step2.f90:22
[riemann-dry]: /Users/cmgotelli/Desktop/BRAIDS/solver/clawpack-v5.14.0/riemann/src/rpn2_geoclaw.f:107
[wave-bc]: /Users/cmgotelli/Desktop/BRAIDS/solver/wave/bc2amr.f90:34
