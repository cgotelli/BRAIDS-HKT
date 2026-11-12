# Total-discharge boundary source audit

The supplied FLOW source implements a `T` boundary as a free-water-level,
strong total-discharge condition. It does not prescribe the boundary water
level.

The runtime path is:

1. `inibnd.f90` maps boundary type `T` to internal type 7.
2. `incbc.f90` computes the width-mean adjacent water level `zavg` over valid wet faces. With the
   default `ZAvgQT=true`, it evaluates a common section depth from that level.
3. `incbc.f90` distributes the requested total discharge with
   `B*h_section^(3/2)*C`. For Manning roughness, `C=h_face^(1/6)/n`, so the face weight is
   proportional to `B*h_section^(3/2)*h_face^(1/6)/n`. It reduces to
   `B*h^(5/3)/n` only when the two depths coincide. The default `ThetQT=0` applies no temporal
   relaxation to those weights.
4. `cucbp.f90` sets the boundary-normal velocity to the allocated face
   discharge divided by face width and current face depth. Its scalar equation
   sets zero normal water-level gradient at the discharge boundary.
5. `bccorr.f90` writes the prescribed face discharge back after the solve so
   the accepted section total is exact.

Relevant supplied files:

- `resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/inichk/inibnd.f90`
- `resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/timedep/incbc.f90`
- `resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute/cucbp.f90`
- `resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute/bccorr.f90`
- `resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_io/src/input/rdbcg.f90`

BRAIDS deck v9 adds three opt-in native controls without changing the existing
characteristic inlet:

- `total_discharge` reproduces the free-stage structure above: adjacent
  zero-gradient stage and section conveyance allocation.
- `prescribed_trace` uses the same face allocation but also prescribes a saved
  horizontal water-level series. It is a diagnostic control for isolating
  sediment and morphology differences when two solvers must receive the same
  boundary depth and velocity history.
- `prescribed_depth_trace` replays saved depth directly. This is required when
  the external open-boundary bed is fixed but BRAIDS's adjacent morphodynamic
  control volume evolves; equal water levels would otherwise produce unequal
  depths and velocities.

All three controls apply total discharge once per logical boundary ID across all of
its selected polygon faces. They run inside the native Fortran/OpenMP timestep;
Python only prepares the versioned deck and validation report. The
[compiled fixed-bed control and matched-depth replay](../validation/sediment/delft3d_intercomparison/reservoir_scale/hydraulic_trace_replay/reports/HYDRAULIC_TRACE_REPLAY.md)
record exact accepted discharge, conservation, reservoir sediment budgets and
final bed-change snapshots.

The [10 September implementation audit](../validation/sediment/delft3d_intercomparison/reservoir_scale/implementation_audit/IMPLEMENTATION_AUDIT.md)
corrects the previous mode-6 use of section-average depth for local velocity/pressure,
retains pressure for zero-Q diagnostic traces, and rejects positive-Q dry-depth
replays before stepping. It compares inlet, outlet and sediment boundary source
paths and separates these fixes from the original 32.1% deposition discrepancy.
