# Coupled Manning work depth: v37 design and qualification contract

Status: **superseded, rejected design**. Do not implement this v37 proposal or
use it to configure a QGIS release. Read-only follow-up inspection of stock
FLOW showed that `erosed.f90` passes the floored 0.01 m calculation depth to
transport; it does not reconstruct a transport Chezy at physical depth as the
proposal below assumes. The physical-depth `Ct` construction is therefore a
BRAIDS hypothesis, not stock-Delft3D equivalence.

The retained narrow correction is instead internal to Van Rijn bedload: keep
BRAIDS' qualified physical-depth C1 activation ramp, evaluate the constitutive
closure and its roughness at `H=max(h_capacity, 0.01 m)`, and preserve resolved
velocity. It needs no deck envelope or QGIS setting. Version 36 remains an
independent hydraulics-only experiment. The historical proposal is retained
below to document what was considered and rejected.

## Decision

Version 37 shall be a new, atomic *coupled Manning work-depth* envelope. A
positive configured floor `F` shall affect every conversion from Manning `n`
to hydrodynamic or sediment resistance in the cut-cell runtime. Version 36
shall remain the frozen hydraulics-only experiment used by the current bend
diagnosis. It shall not be silently reinterpreted as a coupled morphology
format.

The stock-Delft3D value is `F = 0.01 m`. Any other positive value is a BRAIDS
generalization and must not be described as an exact stock-Delft3D setting.
The locked Delft3D source and executable remain read-only.

## Reference evidence

The following observations are from read-only inspection of the unmodified
Delft3D checkout:

- `compute/taubot.f90` floors the velocity-point depth to `0.01 m`, computes
  Manning Chezy from that work depth, uses it for current bed stress, and
  derives current roughness `z0` from that same work depth and Chezy.
- `compute_sediment/erosed.f90` reconstructs the Chezy supplied to transport
  from depth and the roughness stored by `taubot`.
- `timedep/incbc.f90` uses the same `0.01 m` velocity-point floor for the
  Chezy part of total-discharge boundary allocation.

These observations explain why replacing only `h^(4/3)` by a floored
hydraulic denominator is sufficient for a fixed-bed hydraulic comparison but
is not a complete morphology implementation.

## Normative definitions

For a wet state with physical water depth `h > h_dry`, Manning coefficient
`n > 0`, gravitational acceleration `g > 0`, von Karman constant
`kappa = 0.41`, and configured work-depth floor `F > h_dry`, define

```text
H  = max(h, F)
C0 = H^(1/6) / n
x  = kappa*C0/sqrt(g)
z0 = H / (e*(exp(x) - 1))
R  = e*z0 = H/(exp(x) - 1)
Ct = sqrt(g)/kappa * ln(1 + h/R)
nt = h^(1/6) / Ct
```

`C0` is the hydrodynamic Chezy. `z0` is the current-related roughness implied
by the floored hydrodynamic resistance. `Ct` is the Chezy seen by sediment
transport at physical depth. `nt` is only an equivalent transport Manning
coefficient for an interface that still accepts Manning rather than Chezy; it
must never be fed back into the momentum source or stored as bed roughness.

For `h >= F`, the implementation shall return `Ct = C0` by an explicit branch,
not by recomputing the algebraic identity through logarithms. For `h < F`,
`C0` and `z0` are constant with physical depth while `Ct` decreases smoothly
with `h`. At `h = F`, the value is continuous but its derivative is generally
kinked; the equality branch owns the point.

The stable `z0` evaluation is

```text
z0 = H*exp(-1)*exp(-x)/(1 - exp(-x))
```

with a cancellation-safe implementation for small `x`. The `Ct` logarithm
likewise needs a small-argument-safe implementation. Invalid, non-finite, dry,
or nonpositive-`n` inputs shall be rejected by the caller's existing validity
contract rather than hidden with a numerical epsilon.

### Hydrodynamic momentum and shear

The semi-implicit momentum coefficient remains

```text
lambda_f = g*n^2*|u| / (h*H^(1/3))
q_new    = q_old / (1 + dt*lambda_f)
```

where the first `h` is physical. It is incorrect to use `H^(4/3)`.

The hydrodynamic current bed stress used by cohesive exchange and the current
bank-toe model is

```text
tau0 = rho*g*|u|^2/C0^2 = rho*g*n^2*|u|^2/H^(1/3).
```

Transport laws that derive shear from total-flow resistance use

```text
taut = rho*g*|u|^2/Ct^2.
```

The distinction is intentional below `F`.

## Pure helper interface

A small dependency-free Fortran module shall own the new arithmetic. The
recommended public pure functions are:

```text
manning_work_depth(h, F) -> H
manning_hydrodynamic_chezy(h, n, F) -> C0
manning_current_z0(h, n, F, g) -> z0
sediment_transport_chezy(h, n, F, g) -> Ct
equivalent_transport_manning(h, n, F, g) -> nt
sediment_transport_chezy_log_derivative(h, n, F, g) -> Dt
```

The helper module shall not inspect global solver state. Kernels shall receive
either `F` or the already evaluated quantity explicitly. Legacy and v36 paths
shall retain their current arithmetic rather than being routed through helpers
with `F=0`; this is required for byte and numerical replay.

## Analytic derivatives

Let `D0 = d(ln C0)/dh`, `Dz = d(ln z0)/dh`, and
`Dt = d(ln Ct)/dh`. Away from the kink:

```text
h < F:
    D0 = 0
    Dz = 0
    Dt = 1 / ((R + h)*ln(1 + h/R))

h > F:
    D0 = 1/(6h)
    Dz = [1 - x/(6*(1 - exp(-x)))]/h
    Dt = 1/(6h)
```

At `h = F`, use the `h >= F` derivatives. For the equivalent coefficient,

```text
d(ln nt)/dh = 1/(6h) - Dt.
```

For roughness-corrected MPM, define

```text
Cg    = 18*log10(12*max(h/d90, 1))
mu    = min((Ct/Cg)^(3/2), 1)
alpha = mu/(Ct^2*relative_density*d50)
Dg    = 0                                      when h <= d90
      = 18/(ln(10)*h*Cg)                       when h > d90
```

Then the required constitutive derivative is

```text
dalpha/dh = -2*alpha*Dt                         when mu >= 1
dalpha/dh = -0.5*alpha*Dt - 1.5*alpha*Dg       when mu < 1.
```

Above `F`, these reduce exactly to the existing formulas
`-alpha/(3h)` and
`-alpha/(12h) - 1.5*alpha*18/(ln(10)*h*Cg)`.

`mpm_mass_jacobian` already has a separate transport minimum-depth ramp. Its
constitutive depth is `heff = max(h, hmin)`. Evaluate `Ct`, `Cg`, `alpha`, and
the formulas above at `heff`. When physical `h < hmin`, retain the existing
contract: the constitutive state is held fixed (`dalpha/dh = 0`), the existing
smooth wetting weight multiplies the Jacobian, and only the existing
wetting-weight derivative is added. The Manning work floor must not introduce
a second ramp.

## v37 deck and writer contract

The physical text layout shall be

```text
37
<64-character configuration fingerprint>
<F in metres>
<base envelope version, 1 through 35>
<unchanged base payload>
```

Version 37 is logically a coupled successor to v36 but shall wrap versions
`1..35` directly. It shall not contain a nested v36 envelope. On reading v37,
the solver sets both the hydraulic work depth and the coupled sediment work
depth to the one value `F`. A second mode bit is deliberately absent: v37 is
all-or-nothing.

The Python writer shall add a new keyword such as
`manning_work_depth_minimum_m=0.0`. The existing
`manning_friction_chezy_minimum_depth_m` remains the v36 replay switch until
v36 evidence is archived. Rules are:

1. Reject booleans, NaN, infinity, and negative values for either option.
2. Reject simultaneous positive v36 and v37 options.
3. When v37 is enabled, require `F > dry_tolerance_m` and strictly positive
   Manning `n` in every retained cell. This avoids undefined Chezy and the
   current boundary fallback that silently omits division when `n=0`.
4. Emit v37 only for a positive coupled option. Zero emits no new envelope.
5. Reject a malformed/nonpositive v37 `F`, a nested version outside `1..35`,
   or any partially supported configuration before time stepping.
6. Include the v37 line and `F` in the existing fingerprint. The envelope
   version remains in the checkpoint signature, and the fingerprint continues
   to protect the resolved inputs. No new prognostic state is required.

Only enabled v37 metadata gains fields, for example:

```text
manning.work_depth_mode = "coupled"
manning.work_depth_minimum_m = F
manning.hydrodynamic_chezy = "max(h,F)^(1/6)/n"
manning.transport_chezy = "sqrt(g)/kappa*ln(1+h/(e*z0))"
manning.boundary_chezy_floor_m = F
```

With the new option at zero, v1-v35 and current v36 deck text, JSON metadata,
fingerprint inputs, binary sidecars, and numerical paths shall remain
byte-for-byte unchanged for identical arguments and supplied source identity.

## v36 migration and the current partial bed-shear behavior

Current v36 correctly changes the semi-implicit momentum source and therefore
remains valid for the fixed-bed straight and bend evidence. It also currently
changes `bed_shear_stress`, while the Manning-derived `z0`, transport Chezy,
MPM closures, secondary-flow correction, bedform loop, and Exner Jacobian are
still legacy. That mixed morphology behavior is not qualified and must not be
promoted.

The implementation sequence shall be:

1. Freeze the v36 fixed-bed reports, input hashes, executable/source hashes,
   and figures before touching this area again.
2. Preserve the v36 momentum branch exactly.
3. Move the floored `bed_shear_stress` selection behind the v37 coupled flag.
   Under v36 it reverts to the legacy physical-depth stress. This cannot alter
   the frozen fixed-bed runs because neither sediment exchange nor bank-toe
   erosion is active in those runs.
4. Make the writer reject a positive v36-only option when bed load, suspended
   exchange, secondary flow, bedform feedback, or bank-toe shear erosion is
   active. The native reader shall issue the same fail-fast error after the
   base configuration is known. This prevents future partial v36 morphology
   decks while retaining fixed-bed replay.
5. Put the floored hydrodynamic shear and every transport conversion behind
   v37 together. No v37 run may fall back to only part of the coupled mapping.
6. Prove that an `F=0.01 m`, fixed-bed v37 run is bitwise equal to its v36
   counterpart before starting morphology qualification.

This is a deliberate reversion of an unqualified v36 side effect, not a
reinterpretation of the fixed-bed v36 result.

## Boundary allocation

The total-discharge modes currently hard-code the stock `0.01 m` Chezy depth
in both the live trace and the duplicated AMR front-speed forecast. Under v37,
all copies shall use configured `F` and the same `C0` arithmetic.

For a face of width `B`, physical face depth `hf`, section depth `hs`, and
positive local Manning `n`, the unnormalised weights are

```text
free-stage mode 6:       W = B*hs^(3/2)*C0(hf,n,F)
prescribed-stage mode 7: W = B*hf^(3/2)*C0(hf,n,F)
prescribed-depth mode 8: W = B*hf^(3/2)*C0(hf,n,F).
```

The physical powers and the final division by physical `hf` do not receive a
floor. Only the Chezy factor does. The target face velocity remains

```text
uf = Q*W / (sum(W)*hf*B).
```

For `F=0.01 m`, v37 shall reproduce the current stock-style boundary weights
bit for bit by retaining the current multiplication/division order. For an
arbitrary `F`, both live traces and AMR forecasts use that arbitrary value; the
old independent `0.01 m` constant must not survive on a v37 path. Versions
through 36 retain their present hard-coded boundary behavior for replay.

Normal-depth section solving uses the physical Manning conveyance relation,
not this velocity-point Chezy allocation, and is outside this change.

## Call-site matrix

| Consumer | Quantity under v37 | Required change | Legacy/v36 behavior |
|---|---|---|---|
| Semi-implicit momentum source in `cutcell_solver.f90` | `C0`, equivalently `h*H^(1/3)` denominator | Reuse the qualified v36 expression under the v37 flag | Exact existing path |
| `bed_shear_stress` for cohesive exchange and bank toe | `tau0` from `C0` | Select floor only for v37; v36 reverts here | Raw physical-depth stress |
| `sediment_z0` for Van Rijn capacity/exchange and current laws | `z0(H,C0)` | Use `manning_current_z0` only for Manning-derived roughness | Exact current expression |
| Scalar `braids_mpm_bedload_roughness` | `Ct` and physical-depth `Cg` | Accept `Ct` explicitly or call shared helper with `F` | Compute `h^(1/6)/n` as now |
| Batch MPM roughness kernel | Same `Ct` as scalar | One common helper result; retain scalar operation order per fraction | Exact current batch path |
| `mpm_mass_jacobian` in both characteristic and face coupling | `Ct`, `Dt`, `Cg`, `Dg` | Implement the analytic derivatives above and preserve the `hmin` ramp | Exact current derivative |
| Current laws 3, 5, 6, and 7 | `taut` from `Ct` | Pass Chezy explicitly or pass `nt`; do not use `C0` below `F` | Raw Manning Chezy |
| Current law 4 (Soulsby) | v37 `z0` | Retain its law-specific drag equation; change only its Manning-derived roughness input | Current fixed/Manning `z0` behavior |
| Secondary-flow transport direction | `Ct` | Use transport Chezy in the spiral coefficient in both production and diagnostic calls | Raw Manning Chezy |
| Bedform equilibrium feedback | `C0` for hydrodynamic resistance; physical geometry remains physical | Replace seed conversion by `C0`; convert returned total Chezy with `n_next=H^(1/6)/C_total`; downstream transport then derives `z0/Ct` from that effective `n` | Existing physical-depth iteration |
| Boundary modes 6/7/8 | `C0` with configured `F` | Change live trace and both AMR forecast copies together | Hard-coded `0.01 m` stays |
| AMR/restart | Scalar `F`, no new state | Fingerprint and envelope protect it; all levels inherit it | Existing state/reflux logic |

The separate classic-grid `braids_momentum_diffusion_module.f90` runtime is
not built into the cut-cell executable and is not part of v37. It requires a
separate format and qualification if it is brought back into scope.

## Staged tests and release gates

No later stage may waive an earlier failure.

### Stage 0: frozen provenance

- Archive the v36 straight and bend deck, geometry, reference, build, source,
  executable, result, and figure hashes.
- Record the locked unmodified Delft3D runtime identity.
- Verify the Delft3D source tree is unchanged; never patch or rebuild a
  modified reference engine.

### Stage 1: serialization and pure formulas

- Writer and native parser accept valid v37 nesting and reject zero, negative,
  non-finite, boolean, double-envelope, and `F <= h_dry` inputs.
- Disabled-option deck and metadata fixtures for v1-v36 are byte-identical.
- Test `C0`, `z0`, `Ct`, and `nt` at `h = 1, 2, 5, 10, 20 mm` for
  `F = 10 mm`, plus at least two non-stock floors.
- Assert explicit `Ct == C0` at and above `F`, continuity of values at `F`,
  positivity/finite values, and monotonic shallow `Ct`.
- Compare `D0`, `Dz`, `Dt`, and `d ln(nt)/dh` with one-sided or centered finite
  differences away from the kink.

### Stage 2: every local consumer

- Momentum one-step results match the exact semi-implicit formula and always
  dissipate kinetic energy.
- `bed_shear_stress` matches `tau0`; transport-law shear matches `taut`.
- Scalar and batch MPM are numerically identical fraction by fraction.
- Laws 3-7, Van Rijn roughness, secondary flow, and bedform feedback each have
  below-floor, equality, and above-floor tests.
- Compare the complete MPM mass-flux Jacobian with finite differences for
  `mu=1`, `mu<1`, `h<F`, `h>F`, `h<hmin`, and `h>hmin`; exclude only an
  explicitly documented epsilon neighbourhood of the two kinks.
- Prove v37 `F=0.01` boundary allocations conserve prescribed `Q` and are
  bitwise equal to current stock-style weights. Prove arbitrary `F` against a
  manual formula and confirm the AMR forecast uses identical weights.
- V36 morphology-dependent configurations fail before stepping; frozen v36
  fixed-bed decks remain accepted.

### Stage 3: deterministic runtime contracts

- Fixed-bed v37 and v36 with the same `F=0.01 m` are bitwise equal.
- One-thread and 8-thread v37 outputs are bitwise equal; repeat at 32 threads
  on the production host when practical.
- Continuous and checkpoint/restart executions are bitwise equal.
- Static-grid and AMR conservation ledgers close to their existing qualified
  tolerances; AMR regrid/reflux does not change the result when no level change
  is requested.
- No NaN, negative depth/inventory, new wet/dry oscillation, or unexplained
  outlet storage spike is allowed.

### Stage 4: matched fixed-bed hydraulics

- Repeat the shallow straight-trapezoid control against stock Delft3D.
- Repeat the 60 s bend screen, then the full 120 s bend only if the predeclared
  screen gates pass.
- Preserve the current depth, width, wet-area, shoreline, discharge,
  conservation, temporal, and velocity-profile gates.
- Produce depth and velocity-section figures, including the first and last
  `0.1 m` at each bank where the original discrepancy was concentrated.

### Stage 5: isolated sediment equivalence

- Use one grain class in a uniform straight reach and prescribed common
  hydraulic state. Test depths below, at, and above `F` against the locked
  Delft3D result for MPM capacity and bed shear.
- Run a short mobile-bed Exner pulse with matched finest cell size, output
  times, sediment feed, porosity, and boundary treatment. Compare transport,
  cumulative outlet mass, and bed change, not pointwise solver identity.
- Require water and sediment budget closure, correct sediment exit, no outlet
  spike, and convergence under halved time step and halved cell size.
- Produce longitudinal transport/bed profiles and difference figures for both
  engines.

### Stage 6: coupled feature qualification

- Add, one at a time, Van Rijn suspended exchange, cohesive shear, secondary
  flow, bedform roughness, and bank-toe erosion. Each feature gets a minimal
  stock-Delft3D comparison where stock supports the feature and an analytic or
  conservation control where it does not.
- Only after all active consumers pass may the bend morphology and later
  meander/braiding cases use v37.
- QGIS exposure and a rebuilt Windows release are last. The release default
  may change to `0.01 m` only after the enabled configuration, documentation,
  provenance, and all required figures are archived.

## Acceptance verdict

**Rejected.** The evidence does not justify a v37 coupled envelope, a
physical-depth transport `Ct`, or a new plugin control. The fixed-bed v36
evidence remains replayable but does not define morphology. Van Rijn's narrow
calculation-depth correction is qualified separately; MPM, suspended exchange,
cohesive transport and bank-toe shear retain their existing paths until each
receives its own physical validation.
