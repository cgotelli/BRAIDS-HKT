# Thacker notebook and historical water-budget evidence

Reviewed: 7 September 2026. This is a read-only review of the original notebook, driver, saved article metrics and current reproduction. No native simulation was rerun and no numerical source or input was changed.

## Finding

The **current extracted WAVE reproduction** gains 2.3294435% water. The available historical notebook and article artifacts do **not** establish whether the earlier WAVE run gained water. Calling this a demonstrated pre-existing solver defect was too strong; its origin remains unassigned.

## What the original notebook verifies

The [vendored notebook](../vendor/avac-qgis/validation/WAVE/04_thacker_planar_paraboloid/thacker_planar_paraboloid.ipynb) at revision `cfbaaf4d2c6a8c4bbb648dcfb5749fe488b30b32` has five code cells, all with null execution counts and no saved outputs. It calls `run_validation.py thacker_planar_paraboloid --solver wave`, displays `results/summary.json`, and shows the final centerline comparison figure. The notebook in the original local AVAC-QGIS checkout is also unexecuted; no historical native Thacker frames were found in that case directory.

The [driver](../vendor/avac-qgis/validation/WAVE/run_validation.py) computes the final centerline depth RMSE and maximum absolute error in `run_thacker`. It does not integrate water volume or assert conservation. The [saved article metrics](../vendor/avac-qgis/docs/article/figures/wave_analytical_verification.json) and the article's Thacker subsection report that profile comparison, without a Thacker water budget.

| Quantity after three periods | Saved article result | Current extracted reproduction |
|---|---:|---:|
| Centerline depth RMSE (m) | 0.001271876882 | 0.001264966237 |
| Maximum centerline depth error (m) | 0.002835532043 | 0.002935929284 |
| Relative water-volume change | Not recorded | +2.3294435% |

Similar centerline errors do not prove equal two-dimensional water volumes. The historical metrics also do not authenticate a historical compiler, executable or complete input/output deck.

## Setup comparison

The extracted reproduction uses the same pinned `run_thacker` equations and numerical settings: 100×100 cells over a 4×4 m domain, 0.04 m spacing, three periods, 60 output intervals, four walls, analytical h/hu/hv initialization, one AMR level, two ghost cells, MC limiters, second-order unsplit f-waves, dry tolerance `1e-12 m`, CFL target/maximum 0.25/0.5, initial timestep 0.002 s, and no friction. The WAVE Makefile is unchanged; the added OpenBLAS path is a platform linker setting.

The [extraction patch](../solver/wave_hydraulic.patch) changes the configuration filename and makes shoreline forcing optional. The original hydraulic preparation supplies an empty shoreline source; the extracted case specifies `none`. Source inspection shows both leave `inflow_ncells=0`, for which `apply_internal_inflow` returns immediately. No configured water source or changed numerical setting was found. This source comparison is evidence about the setup, not a substitute for a controlled native comparison or historical volume outputs.

## What is measured now, and what remains open

[The current mass diagnostic](../validation/hydraulics/wave_baseline/reports/thacker_native_mass.json) integrates every native level-1 cell over the entire 16 m² domain. Volume rises from `0.15707993600000025 m³` to `0.16073902437282672 m³`. Fixed-grid integration independently confirms approximately the same gain. This establishes a water-budget problem in the current reproduction; it does not identify its cause or date of introduction.

Before changing numerical physics specifically to fix this gain, build untouched pinned WAVE in isolation and compare it with the extracted build using matched compiler/options, thread count and numerical inputs, preserving both runs. Measure the same native-cell volume history for both. Historical conservation can only be established if historical full-domain outputs, or the corresponding executable and complete inputs, are recovered. Then use the accepted-step flux/correction ledger to attribute the observed gain. Clipping remains a candidate, not a demonstrated cause.

The user's subsequent 7 September decision is to continue inlet implementation and development runs while this investigation remains open. This comparison is part of assigning the Thacker gain's cause, not a prerequisite for boundary development. The roadmap retains separate discharge and volume measurements and requires evidence before marking conservation checks complete.
