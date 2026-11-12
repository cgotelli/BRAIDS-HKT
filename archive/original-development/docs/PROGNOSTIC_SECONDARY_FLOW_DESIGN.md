# Prognostic secondary-flow design

Status: **implementation contract; pure physics helpers implemented, but no
cut-cell runtime/state coupling yet**.

This design adds a signed, advected-relaxing secondary-flow intensity to the
BRAIDS cut-cell solver.  It deliberately does not require cellwise agreement
with Delft3D.  The qualification target is the reach-scale bend response:
outer-bend erosion, inner-bend deposition, downstream sediment export, and,
after the fixed-bank transport is credible, outward bank retreat and
compensating deposition.

`resources/Delft3D-main/` was inspected read-only.  It must remain read-only.
The formulas below use the unmodified source as implementation evidence, not
as a claim that Delft3D is ground truth.

## 1. Decision

Add one optional signed conserved state

```text
M_I = h I                                      [m2/s]
```

where `h` is water depth and `I` is spiral-flow intensity `[m/s]`.  Evolve it
with

```text
d(M_I)/dt + div(h u I) = h (I_eq - I) / T_A
I_eq = h k_s |u|
```

for the Cartesian, non-rotating BRAIDS model.  Here `k_s` is signed
streamline curvature.  The source is integrated by exact local relaxation;
advection uses the same final, positivity-limited water flux as the shallow
water equations.

The feature has three independently testable effects:

1. `memory only`: advect and relax `I`, write diagnostics, but do not change
   hydraulics or sediment;
2. `near-bed direction`: use prognostic `I` to rotate bed-load direction
   while preserving each law's scalar capacity exactly;
3. `momentum feedback`: optionally add the secondary-flow dispersion-stress
   tensor through one shared conservative face flux.

The first two are the minimum useful implementation.  Momentum feedback is a
separate experimental stage and defaults to zero.

### Production-candidate decision

The verified frozen-toe preflight changes what can count as the first
**production candidate**, even though it does not change the safe coding
order.  At the mapped outer target toe, the maximum current
`tau/tau_critical` is `0.576757`; the slope-side bracket reaches `0.575887`.
Both have zero resolved Van Rijn capacity and exactly zero one-stage Exner
change.  The roughly `9.69e-13 kg m-1 s-1` raw formula value is below the
`2.65e-9 kg m-1 s-1` numerical-support threshold and is not active transport.

A direction rotation preserves magnitude, so it cannot activate this toe.
Therefore:

- memory-only and direction-only stages remain necessary verification and
  ablation stages, but neither is a production bend candidate;
- the first production-candidate capacity evaluation **must include the
  qualified momentum feedback before capacity is evaluated**, or an
  independently justified and qualified near-bed shear-magnitude closure;
- for the implementation specified here, the recommendation is to complete
  and enable the `BetaSpiral` momentum path for that first candidate;
- do not lower critical shear, retain a below-threshold capacity floor, or
  tune a spatial coefficient merely to force activation.

The candidate must demonstrate `tau/tau_critical >= 1` over a nonzero outer
target length, nonzero resolved capacity with outer-target dominance, and an
exactly nonzero erosive Exner response before bank failure is run.  If a
predeclared physically supported global `BetaSpiral` bracket cannot do that,
the result is that momentum feedback is insufficient; the next step is the
separately validated near-bed shear closure, not promotion of direction-only
physics.

This staging does not narrow the required end state.  The intended bend model
contains prognostic memory, direction coupling, a justified magnitude-changing
hydraulic/near-bed pathway, conservative Exner transport, and physical toe
erosion/undercut/collapse.  The early stages isolate faults; they are not
permission to stop before that coupled state exists.

## 2. Current production audit

### 2.1 What exists

- `solver/wave/braids_secondary_flow_module.f90`
  - `braids_streamline_curvature` computes instantaneous streamline
    curvature from velocity gradients.
  - `braids_delft3d_spiral_transport` computes `I_eq=h*k_s*|u|` and rotates
    bed-load direction.  It preserves the incoming transport magnitude.
- `solver/cut_cells/cutcell_solver.f90`
  - the v28 envelope reads `secondary_flag` and `secondary_flow_espir`;
  - `prepare_bedload_capacity` recomputes curvature and immediately applies
    the equilibrium direction correction;
  - `secondary_flow.txt` is diagnostic output only;
  - no secondary-flow state is advected, refluxed, or checkpointed;
  - the production guard accepts bed laws `2, 3, 5, 6, 7`, but rejects the
    Van Rijn bed-load law `1` used by the bend work.
- QGIS exposes a single checked/unchecked equilibrium correction and labels
  it as direction-only.  Old projects store only `enabled` and `espir`.

Consequently the current option has no downstream memory.  Curvature at the
same cell determines the full correction at every capacity evaluation.  It
cannot represent development through a bend or decay after a bend.

### 2.2 Existing state assumptions that must not be violated

The current runtime layout is

```text
q(1)                              h
q(2:3)                            hu, hv
q(3+g)                            suspended inventory, fraction g
q(3+nfrac+(layer-1)*nfrac+g)      active/substrate inventory
```

with

```text
nstate     = 3 + (1 + nbed_layers) * nfrac
nprimitive = 4 + nfrac
nledger    = 1 + 2*nfrac
```

Several paths currently interpret every row `4:nstate` as a non-negative
sediment inventory.  In particular:

- initial, trial, checkpoint, and AMR-checkpoint validity tests;
- the underflow cleanup at the end of `euler`;
- `amr_repair_reflux_tails`;
- the stratigraphy loop in `amr_prolong_node`.

A signed spiral intensity must never enter any of those loops.

The flux arrays, SSPRK combinations, AMR dense time interpolation, flux
registers, reflux update, and checkpoint `q` write are otherwise generic in
`nstate`.  Those are safe to reuse after the physical/signed ranges are made
explicit.

## 3. Runtime indexing contract

Introduce explicit physical and runtime sizes after `nfrac` and
`nbed_layers` are known:

```text
nphysical_state     = 3 + (1 + nbed_layers) * nfrac
nphysical_primitive = 4 + nfrac
nphysical_ledger    = 1 + 2*nfrac

if prognostic secondary flow:
    secondary_state_index     = nphysical_state + 1
    secondary_primitive_index = 4 + nfrac
    secondary_ledger_index    = nphysical_ledger + 1
    nstate                     = nphysical_state + 1
    nprimitive                 = nphysical_primitive + 1
    nledger                    = nphysical_ledger + 1
else:
    nstate                     = nphysical_state
    nprimitive                 = nphysical_primitive
    nledger                    = nphysical_ledger
```

`bed_primitive_index()` remains `nprimitive`, so the prognostic primitive is
inserted immediately before bed elevation.  Existing concentration indices
do not move.  The conserved `M_I` row is appended after every physical bed
inventory, so every public physical-state slice remains contiguous.

Add small index/query functions rather than repeating arithmetic:

```fortran
pure integer function secondary_state_index()
pure integer function secondary_primitive_index()
pure integer function secondary_ledger_index()
pure logical function prognostic_secondary_flow()
```

All non-negativity tests and sediment-tail loops must end at
`nphysical_state`, never `nstate`.  All-state finiteness tests still cover
`1:nstate`.  A dry cell must satisfy `M_I=0`; a wet cell may have either sign.

The following source patterns are a mandatory audit checklist before merge:

```text
q(4:nstate)              -> q(4:nphysical_state)
state(4:nstate)          -> state(4:nphysical_state)
next(4:nstate)           -> next(4:nphysical_state)
do m=4,nstate            -> do m=4,nphysical_state
do j=4+nfrac,nstate      -> do j=4+nfrac,nphysical_state
at_q(4+nfrac:nstate,...) -> at_q(4+nfrac:nphysical_state,...)
```

Do not mechanically change generic full-state flux, RK, restriction,
reflux, or checkpoint operations.

## 4. Physical closure

### 4.1 Equilibrium intensity

Use the existing signed curvature convention without changing it:

```text
U    = sqrt(u^2 + v^2)
I_eq = h k_s U
```

Set `I_eq=0` when the cell is dry or `U < 1e-3 m/s`.  Coriolis remains absent
because BRAIDS is a Cartesian planar model and has no latitude/Coriolis
configuration.

### 4.2 Adaptation time

For wet cells with `U >= 1e-3 m/s`, define

```text
C       = h^(1/6) / n
a_A     = sqrt(g) / (kappa_A max(C, 20 m^0.5/s))
L_A     = (1 - 2 a_A) h / (2 kappa_A^2 a_A)
T_A     = L_A / U
kappa_A = 0.41
```

The Chezy floor keeps `a_A < 0.5` and therefore `T_A > 0`.  Invalid or
non-finite inputs are errors; they are not hidden with an epsilon.  For dry
or slow cells the relaxation source is inactive.  Dry cells still set
`M_I=0` through the scalar wet/dry invariant.

The `0.41` adaptation constant is distinct from the explicit `0.4` in the
existing Delft transport-angle formula.  Do not collapse the two constants.

### 4.3 Exact relaxation

For one source interval `dt_s`, hold the local hydraulic state fixed and use

```text
I_new = I_eq + (I_old - I_eq) exp(-dt_s/T_A)
M_new = h I_new
```

Use `expm1`-equivalent arithmetic for the source increment when
`dt_s/T_A` is small:

```text
Delta M = h (I_eq - I_old) * (-expm1(-dt_s/T_A))
```

This is unconditionally stable, preserves sign changes, and avoids making a
small shoreline adaptation time the global timestep limit.

### 4.4 Near-bed transport direction

When direction coupling is enabled, replace the equilibrium intensity in the
existing rotation by the prognostic value `I=M_I/h`:

```text
C       = h^(1/6) / n
a_dir   = 12.5 Espir (1 - sqrt(g)/(0.4 C)) I/U
t       = (u + a_dir v, v - a_dir u)
q_b,new = |q_b,old| t/|t|
```

This is the meaning of "near-bed coupling" in this milestone.  It changes
direction only.  It does **not** invent a capacity multiplier, change the
reported depth-averaged velocity into a near-bed velocity, or add momentum
forcing unless the separate momentum coefficient is nonzero.

`braids_delft3d_spiral_transport` must retain its existing legacy arithmetic.
Add a separate helper that accepts `I`; do not silently reinterpret v28.

### 4.5 Optional momentum feedback

For a later, separately qualified stage, use

```text
a_M = sqrt(g)/(kappa_A C)
B   = BetaSpiral (5 a_M - 15.6 a_M^2 + 37.5 a_M^3)
b   = B I/U

Txx = -2 h b u v
Txy =  h b (u^2 - v^2)
Tyy = -Txx
```

`BetaSpiral=0` is the default and must bypass the entire path.  Construct one
face tensor from the two traces (one-sided at a boundary) and add the same
face contribution to both adjacent momentum balances.  With BRAIDS' signed
curvature/intensity convention, the qualified additional conservative flux is
`+T.n`.  The frozen pre-implementation gate was decisive: `-T.n` transferred
streamwise momentum outer-to-inner in both an analytic annular sector and the
Jamieson bend, whereas `+T.n` transferred it inner-to-outer while preserving
conservation, rotation covariance, coefficient linearity and grid convergence.
The sign is therefore fixed by the physical bend-orientation oracle rather
than by matching individual Delft cells.

Use the face-normal stress Jacobian bound

```text
c_spiral = 2 |B I|
```

as an added explicit CFL speed initially, and verify it numerically over the
qualified parameter range before enabling nonzero `BetaSpiral` in QGIS.
Internal face forces must cancel to roundoff.  The domain momentum change may
contain the explicitly reported boundary traction.

## 5. Advection and time integration

### 5.1 Signed scalar flux

Add dedicated arrays in `euler`; do not disguise this field as another
sediment fraction:

```text
secondary_low_flux(nf)
secondary_antidiffusive_flux(nf)
secondary_rplus(nc), secondary_rminus(nc)
```

Build the scalar flux only after the conservative water-draining limiter has
finalized `fl(1,f)`:

```text
low flux  = F_h * upwind(I)
high flux = F_h * reconstructed_face(I)
anti flux = high - low
```

Use a signed FCT limiter with local extrema in `I`, expressed as bounds on the
updated conserved state:

```text
M_min = h_next * min(neighbour and applicable boundary I)
M_max = h_next * max(neighbour and applicable boundary I)
```

When `h_next <= dry_tolerance`, both bounds are exactly zero.  The limiter may
bound `I`, but it must not apply sediment non-negativity to `M_I`.  The final
limited flux is common to both sides of an internal face, so advection is
conservative and AMR can reflux it like any other state.

### 5.2 Boundary contract

- Internal face: use the water-flux donor for the low-order intensity and the
  limited left/right reconstruction for the high-order flux.
- Net outflow: use the reconstructed owner intensity.
- Net inflow through any open hydraulic boundary: prescribe the owner-cell
  local equilibrium intensity.  This is `inflow_mode=1` and is the only
  initially supported inflow mode.
- Closed wall/obstacle: `F_h=0`, hence zero intensity flux.
- A locally closed reversal has zero intensity flux.  A genuinely accepted
  reverse inflow uses the same equilibrium-inflow rule.

Boundary classification follows the **accepted face water-flux sign**, not
the boundary name.  This avoids stale inlet/outlet assumptions during a local
reversal.

### 5.3 SSPRK/source composition

Use Strang splitting around the existing SSPRK3 transport step:

```text
base = q
relax(base, dt/2)
SSPRK3 advection and existing physical source, using base as the RK anchor
relax(candidate, dt/2)
apply the existing accepted bank/collapse sequence
commit candidate
```

Only the signed auxiliary row changes during `relax`; therefore physical
states follow the existing arithmetic when transport and momentum coupling
are off.  On a rejected trial, discard both relaxation increments with the
trial state.

For `amr_level_rk`, apply the same two half relaxations only to cells owned by
the current level and use that level's substep `dt`.  Coarse ghost `M_I` is
already time-interpolated by `amr_coarse_ghosts`.  Fine levels consequently
relax at their own subcycled timestep.

Compute curvature twice per accepted level step (before and after transport),
not once per bed-law evaluation.  Prognostic bed-load rotation reads stored
`I` and does not need curvature.  Keep the existing static OpenMP curvature
loop; this is both cheaper than the present per-RK capacity coupling and
deterministic.

## 6. Conservation ledger

Use the appended internal ledger row for cumulative boundary transport of
`M_I`, positive into the domain.  Fill it beside the existing water/sediment
rows in `euler`.  The existing SSPRK weights and recursive AMR ledger sums can
then be reused.  `prescribed_boundary_rates` leaves this row unprescribed, so
the current hydraulic inlet reconciliation does not reinterpret it as a bed
inventory.

Track two additional checkpointed scalars:

```text
secondary_initial_integral
secondary_source_cumulative
```

Every exact relaxation call returns its cellwise `Delta M`.  Sum
`area(c)*Delta M` serially in ascending cell order after the parallel cell
loop.  Accumulate into a trial scalar and commit it only after the root step
is accepted.  This avoids thread-order changes and rejected-step leakage.

At every diagnostic output, report

```text
R_I = integral(M_I) - secondary_initial_integral
      - sum(boundary_secondary_ledger)
      - secondary_source_cumulative
```

`R_I` is the signed-scalar numerical residual.  Relaxation is a physical
source, so `integral(M_I)` alone is not expected to be constant.

## 7. AMR contract

Restriction already area-averages every runtime state and therefore handles
`M_I` correctly.  Flux registers, dense coarse time interpolation, and the
reflux state update are also generic and require no special branch.

### 7.1 Prolongation

`amr_prolong_node` needs one signed, water-weighted branch after child depths
are known:

1. `I_p=M_p/h_p` for a wet parent; otherwise require `M_p=0`.
2. Reconstruct `I_raw(k)=I_p+grad(I).dx(k)` for wet children.
3. Define `I_lo=min(I_p,min(I_raw))` and
   `I_hi=max(I_p,max(I_raw))`; dry children have `M_k=0` and no capacity.
4. Start with `M_k=h_k*clamp(I_raw,I_lo,I_hi)`.
5. The exact target is `A_p*M_p`.  Distribute the remaining signed integral
   among wet children, in fixed child order, using the capacities
   `A_k*h_k*(I_hi-I_k)` or `A_k*h_k*(I_k-I_lo)`.
6. Assert that the remaining residual is within scaled roundoff.

This preserves the parent integral exactly, retains sign, introduces no new
extremum, and is deterministic.  It should be a dedicated helper such as
`amr_signed_water_transfer`; do not reuse `amr_positive_transfer`.

No secondary-flow AMR trigger is added in the first implementation.  Existing
hydraulic/morphodynamic triggers determine the mesh, while the signed state is
conservatively transferred.  That prevents a new state from changing mesh
history before fixed/AMR equivalence is understood.

### 7.2 Reflux repair

`amr_repair_reflux_tails` stops at `nphysical_state`.

If `amr_repair_water_reflux` zeroes a negative coarse water inventory, it must
also:

- save `A_c*M_c`;
- set coarse `M_c=0`;
- distribute that signed integral to the same fine water donors in the same
  deterministic fractions used for momentum.

This preserves the scalar budget while restoring the dry-cell invariant.

## 8. Input, restart, and results compatibility

### 8.1 Deck envelope

Use the next available outer envelope, currently v37.  Its line is

```text
model transport_direction Espir BetaSpiral inflow_mode
2     0|1                 >=0   >=0        1
```

`model=2` means prognostic advect-relax.  The envelope wraps an existing v1 to
v36 deck.  It must not also wrap v28.  The writer emits:

- no secondary envelope when disabled;
- v28 for the frozen equilibrium legacy model;
- v37, and no v28, for the prognostic model.

Move secondary defaults before the v37 parse in `load_case`; the current
default assignment occurs after the v33-v36 envelope block and would erase a
new outer setting.  Reject duplicate nested v37/v28 model declarations.

The v37 fingerprint covers every field.  Add runtime/physical state and
ledger sizes plus the model enum to `initial_signature` for a defensive
restart mismatch check **only for the v37 model**.  Preserve the existing
signature shape for disabled and v28 decks so their checkpoints retain their
current compatibility.

### 8.2 Initial deck state remains physical

Do not change the Python `q` shape or the native binary sidecar layout.
`braids_core/boundaries/cut_cell_io.py::_write_native_binary` continues to
write `nphysical_state` rows.

In Fortran:

```text
allocate q(nstate,nc)
q = 0
read formatted q(1:nphysical_state,c)
read binary    physical_q(nphysical_state,nc)
assign         q(1:nphysical_state,:) = physical_q
compare binary_header(4) with nphysical_state
```

Thus new runs start with `I=0`, consistent with the reference runtime's
default initial spiral intensity.  A hydraulic spin-up checkpoint is the
recommended way to initialize a developed bend field before morphology.

### 8.3 Checkpoint contains runtime state

The private checkpoint writes full runtime `q`, full runtime `ledger`,
`secondary_initial_integral`, and `secondary_source_cumulative`.  AMR
checkpoints likewise retain full `at_q`.  With the v37 fingerprint/signature,
old checkpoints cannot be misread as prognostic checkpoints.

Full-run versus prefix/restart must be bitwise identical for both fixed and
adaptive runs, including the secondary budget scalars.

### 8.4 Public frames remain physical

The authoritative public files remain backward-compatible:

```text
frames.txt / frames.bin:      q(1:nphysical_state,:), ledger(1:nphysical_ledger,:)
boundary_ledger.txt:          ledger(1:nphysical_ledger,:)
binary frame header nstate:   nphysical_state
binary frame header nledger:  nphysical_ledger
```

This is required because `braids_qgis/core/results.py` validates exactly
`3+(1+bed_layers)*nfrac` state rows and slices all remaining rows as substrate.
Appending `M_I` to public frames would silently corrupt the substrate parser.

Write prognostic diagnostics to a separate sidecar with at least

```text
time, cell/node id, x, y, h, curvature, I_eq, I, T_A,
direction coefficient, M_I, cumulative boundary M_I,
cumulative source M_I, budget residual
```

Per-fraction bed-load vectors may remain in the existing secondary diagnostic.
Keep the legacy v28 diagnostic header and arithmetic unchanged.  A later QGIS
map layer can consume a compact dedicated intensity sidecar; it must not force
a public-frame format change in the solver milestone.

## 9. Configuration and QGIS behavior

### Python writer (`braids_core/boundaries/cut_cell_io.py`)

Add keyword arguments with backward-compatible defaults:

```python
secondary_flow_model="equilibrium"     # used only when enabled
secondary_flow_transport_direction=True
secondary_flow_momentum_beta=0.0
```

Old calls using only `secondary_flow=True` retain v28 semantics.  A prognostic
model validates positive Manning everywhere.  It may run with `bedload=0`
when both physical couplings are off, enabling a bed-fixed spin-up.  Direction
coupling requires an active magnitude-based bed-load law.  Bed law 1 is added
only after its explicit magnitude-preservation test passes; the v28 law list
is not broadened retroactively.

Metadata must distinguish

```text
model: equilibrium | prognostic
transport_direction: true | false
momentum_beta: value
qualified: true only for the already qualified legacy scope
```

Do not describe the prognostic model as qualified until the gates in section
12 pass.

### Project model (`braids_qgis/core/project.py`)

Persist the three fields above.  Missing `model` in an old project resolves to
`equilibrium`; missing `momentum_beta` resolves to `0.0`.  Round-trip tests
must prove old JSON is unchanged unless the new model is selected.

### Dock (`braids_qgis/gui/dock.py`)

Keep the master checkbox blank by default.  Under it add:

- model combo:
  - `Equilibrium direction (legacy/qualified)`;
  - `Advected-relaxing intensity (experimental)`;
- `Use intensity for bed-load direction`, checked when a compatible bed law
  is active;
- advanced `Momentum feedback Beta`, default `0.0`, disabled in the first UI
  release until the momentum gate passes.

The explanatory text must say that direction coupling preserves capacity
magnitude and that momentum feedback is independent.  Diagnostic-only
prognostic spin-up remains available without an active sediment law.

### Results (`braids_qgis/core/results.py`)

No phase-1 parser change is required because authoritative frames remain
physical.  A subsequent result-layer increment may parse the separate
intensity sidecar and expose `spiral_intensity_m_s` and
`equilibrium_spiral_intensity_m_s`.  That visualization is not a prerequisite
for the numerical core tests.

## 10. Exact implementation map

### `solver/wave/braids_secondary_flow_module.f90`

Keep both existing public routines.  Add pure, dependency-free routines:

```fortran
braids_spiral_equilibrium(h,u,v,curvature,intensity,active,valid)
braids_spiral_adaptation_time(h,u,v,manning_n,g,time_a,active,valid)
braids_spiral_relax_exact(h,stored_mass,equilibrium,time_a,dt,
                          updated_mass,source_increment,valid)
braids_spiral_transport_from_intensity(h,u,v,manning_n,g,espir,intensity,
                                       qx,qy,alpha,valid)
braids_spiral_stress(h,u,v,manning_n,g,beta_spiral,intensity,
                     tensor,signal_speed,active,valid)
```

Do not change the legacy wrapper's operation order while legacy exactness is a
test requirement.

### `solver/cut_cells/cutcell_solver.f90`

Touch only these named interfaces in the staged implementation:

- module variables and index functions: physical/runtime sizes, model,
  coupling flags, budget scalars;
- `load_case`: v37 parse, physical input reads, validation, allocation,
  signature;
- `primitive`: compute `I=M_I/h` in the new primitive row;
- `euler`: signed scalar FCT, boundary ledger row, optional shared momentum
  stress flux and CFL bound (inserted after the final water-flux draining
  correction and before the CFL/divergence loop);
- `advance` and `amr_level_rk`: transactional Strang half-relaxations;
- `prepare_bedload_capacity`: select legacy equilibrium or stored-intensity
  direction helper, without recomputing curvature in prognostic mode;
- `secondary_flow_curvatures`: retain the current parallel operator; allow the
  relaxation caller to evaluate from its supplied state rather than relying
  on a stale `active_cell` mask;
- `amr_prolong_node`: signed water-weighted transfer;
- `amr_repair_water_reflux` and `amr_repair_reflux_tails`: signed-state rules;
- checkpoint read/write: full runtime state and budget scalars;
- `output_frame`: public physical slices;
- `output_secondary_flow_frame`: model-specific diagnostic values.

Do not add a second advection engine or a second checkpoint file.

Existing fast paths and compatibility guards that currently test the broad
`secondary_flow` boolean must be split by effect.  A memory-only prognostic
run must not disable an otherwise valid bed-load shortcut.  The v28 guard
keeps its current law list; the prognostic direction guard applies only when
`transport_direction` is true, and diagnostic-only/momentum-only runs do not
require an active bed-load law.

### Python/QGIS/build files

- `braids_core/boundaries/cut_cell_io.py`: v37 writer, validation, metadata;
- `braids_qgis/core/project.py`: persistence and backward defaults;
- `braids_qgis/gui/dock.py`: model/coupling controls and truthful status;
- `braids_qgis/core/results.py`: unchanged in the core milestone; optional
  sidecar reader later;
- `tools/build_cutcell.py`: no source-list change if the existing module is
  extended; its source hash must naturally change when implementation lands.

### Test/validation files

- extend `tests/test_secondary_flow.py` for pure closure and fixed-grid tests;
- extend `tests/test_amr_native.py` for signed AMR/restart tests;
- extend `tests/test_cut_cell_io.py` for v28/v37 serialization;
- extend `tests/test_windows_packager.py` for packaged-source/hash coverage;
- add `tests/test_secondary_flow_prognostic.py` if separation keeps the matrix
  readable;
- add validation artifacts under
  `validation/sediment/secondary_flow_prognostic/` and reuse the established
  Jamieson bend zone definitions rather than inventing cell-pair scores.

## 11. Smallest staged implementation

### Milestone A: signed memory, fixed grid, no physical coupling

Implement pure formulas, v37, internal state/ledger, physical input/output
slices, fixed-grid FCT, exact relaxation, diagnostics, checkpoint, and
determinism.  Keep direction coupling off and `BetaSpiral=0`.

Independent success condition: analytical advection-relaxation, scalar budget,
thread exactness, restart exactness, and physical `q` equality with the
coupling-off control all pass.

### Milestone B: AMR memory

Implement prolongation, reflux repair, level-local Strang splitting, and AMR
restart.  Add no refinement trigger.

Independent success condition: regrid conserves signed `M_I`, fixed/AMR bend
integrals converge, and AMR thread/restart tests are exact.

### Milestone C: near-bed direction

Use stored `I` in the direction rotation and qualify bed law 1.  Capacity
magnitude must remain unchanged to roundoff for every supported law.

Independent success condition: the fixed-bank bend has the correct global
erosion/deposition topology, no outlet spike regression, and closed water/bed
budgets.  Delft3D is a peer curve/map, not a cellwise pass oracle.  This is an
ablation/transport-direction gate, not a production-candidate gate, because
the frozen physical toe has zero active capacity to rotate.

### Milestone D: optional momentum feedback

Add the conservative tensor flux and CFL bound behind `BetaSpiral>0`.  Keep
the general QGIS control disabled until manufactured, conservation, stability,
and bend-orientation gates pass.  Once those isolated gates pass, this stage
is **required in the first production-candidate bend run** unless an
independently validated near-bed shear-magnitude closure replaces its role.
Evaluate a predeclared global coefficient bracket; do not spatially tune it to
the toe threshold.

### Milestone E: coupled bank evolution

Only after fixed-bank transport passes **and the outer target has active,
nonzero capacity**, run the physical toe-erosion, undercut, and collapse
model.  Secondary flow may change the toe forcing through
hydraulics/transport; it must not directly tune collapse or move a bank
independently of the existing material-transfer operator.

## 12. Verification matrix

| Layer | Case | Required observation |
|---|---|---|
| Pure | zero/constant/signed curvature | correct `I_eq` magnitude and sign; zero in dry/slow cells |
| Pure | `dt/T` from `1e-6` to `1e6` | exact relaxation finite and bounded between old and equilibrium intensity |
| Pure | straight, 30, 45, 90 degree rotations | coordinate-rotation invariance and correct sign reversal |
| Pure | supplied positive/negative `I` | bed-load magnitude unchanged to scaled roundoff |
| Pure | stress tensor | symmetric, trace zero, finite signal bound |
| Fixed | straight channel, zero curvature | `M_I` remains exactly zero |
| Fixed | periodic signed pulse, relaxation disabled in test helper | integral conserved and refinement reduces error |
| Fixed | 1-D constant-flow relaxation reach | matches `I(x)=Ieq+(I0-Ieq)exp(-x/L_A)`; spatial/time convergence recorded |
| Fixed | S-bend/reversing curvature | downstream memory and sign reversal occur over the predicted adaptation length |
| Boundary | inflow/outflow/reversal/wall | equilibrium inflow, donor outflow, zero wall flux, budget closure |
| Fixed | coupling off versus feature off | every physical state and physical ledger is byte-identical |
| Fixed | 1/8/32 OpenMP threads | full runtime state, ledgers, and diagnostics are byte-identical |
| Restart | full versus prefix/restart | checkpoint continuation is byte-identical, including budget scalars |
| IO | old v28 and old project JSON | identical legacy deck/result behavior |
| IO | v37 public frames | existing QGIS reader accepts physical state; substrate count unchanged |
| AMR | refine then coarsen a signed field | area integral conserved to scaled machine tolerance; no new sign extremum |
| AMR | subcycled analytic reach | expected relaxation/advection and scalar budget at every regrid |
| AMR | 1/8 threads and restart across regrid | byte-identical runtime/checkpoint result |
| Momentum | constant tensor field | zero interior divergence |
| Momentum | manufactured varying tensor | second-order trend on regular cells and correct source sign |
| Momentum | closed/internal faces | pairwise cancellation; global change equals reported boundary traction |
| Stability | beta parameter sweep | no unreported CFL violation; timestep refinement converges |
| Bend A | fixed-bank Jamieson geometry | outer-half net erosion, inner-half net deposition, erosion centroid farther outward |
| Bend A | downstream strip | no domain maximum or unexplained one-cell depositional spike at outlet |
| Bend A | physical outer toe before bank-on | `tau/tau_critical >= 1` on nonzero length, resolved capacity nonzero, outer target dominates controls, Exner erosion nonzero |
| Bend A | timestep/grid/thread/restart | global zone volumes/centroids stable; water and sediment ledgers close |
| Bend B | bank on minus bank off | outer retreat dominates controls, centreline moves outward, compensating deposition appears |
| Bend B | material accounting | donor loss, toe/block deposition, bed storage, and export close |

For the bend, use the revised global-physics decision in
`validation/sediment/jamieson2013_bend/reports/global_physics_target_v1/PROPOSAL.md`.
The earlier fine-cell expected-pair percentage is retained as a diagnostic,
not a hard gate, because alternating cell changes dominate it.  Net zone
signs, lateral centroids, outlet behavior, mass closure, and bankline movement
are the primary decisions.

Compare BRAIDS first with laboratory observations and the published global
trend, and second with the locked unmodified Delft3D run.  Report common maps,
signed zone volumes, lateral centroids, sections, bank contours, and
longitudinal retreat/deposition profiles.  Do not tune to individual Delft3D
cells and do not modify Delft3D source.

## 13. Promotion gates

The prognostic option may be labeled production-ready only when:

1. all legacy secondary-off and v28 tests pass unchanged;
2. signed-scalar fixed/AMR conservation, restart, and 1/8/32-thread gates pass;
3. Van Rijn direction coupling proves exact capacity-magnitude preservation;
4. fixed-bank bend signs, centroids, outlet behavior, and material ledgers pass
   on both declared grids and timesteps;
5. nonzero momentum feedback passes its tensor/CFL gates and the physical-toe
   activation gate, unless a separately qualified near-bed magnitude closure
   fulfills that role;
6. bank-on results satisfy global retreat and material-transfer gates without
   a direct secondary-flow collapse tuning term;
7. QGIS metadata no longer claims qualification beyond the evidence actually
   produced.

Until then, v28 remains the qualified legacy option and the v37 model is
explicitly experimental.

## 14. Reference provenance

The governing structure is supported by read-only inspection of:

- `resources/Delft3D-main/.../compute/equili_spiralintensity.f90`:
  `I_eq=h*k_s*U` (with a Coriolis term not applicable here);
- `resources/Delft3D-main/.../compute_transport/fill_constituents.f90`:
  prognostic constituent source/sink and `T_A=L_A/U`;
- `resources/Delft3D-main/.../compute/get_spiralforce.f90`:
  optional `BetaSpiral` dispersion stresses;
- `resources/Delft3D-main/.../morphology_kernel/src/eqtran.f90`:
  direction-only use of supplied spiral intensity;
- `resources/Delft3D-main/.../flow2d3d_kernel/src/compute/secrhs.f90` and
  `secbou.f90`: classic source/sink and open-boundary behavior.

The adaptation concept originates with Kalkwijk and Booij, *Adaption of
secondary flow in nearly-horizontal flow* (1986),
<https://doi.org/10.1080/00221688609499330>.

These references determine the model family and sign conventions to test.
BRAIDS retains its own finite-volume/cut-cell discretization, conservative AMR
machinery, and independent global validation criteria.
