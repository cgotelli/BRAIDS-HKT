# Optional native horizontal momentum diffusion

Experimental native-deck API:

```python
write_cut_case(
    horizontal_viscosity_m2_s=0.01,
    horizontal_viscosity_formulation="depth_conservative",
)
```

The coefficient is finite, nonnegative, in m2/s, and defaults to **zero**.
The formulation is either `depth_conservative` (the default and original
BRAIDS operator) or `velocity_laplacian` (the explicitly selected,
Delft-compatible continuum form). `velocity_laplacian` requires a positive
coefficient. This is not yet a qualified QGIS release feature.

## Equations and discrete contract

`depth_conservative` applies

`d(h u)/dt = div(nu h grad(u))`, and the same for `v`.

For interior face i--j its symmetric conductance is

`Kij = nu * face_length * harmonic_mean(hi,hj) / |xj-xi|`.

Momentum transfer into i is `Kij * (uj-ui)`; the identical opposite transfer
enters j. It therefore conserves integrated depth-averaged momentum. Its
stability rate is `lambda_i = sum_j(Kij)/(Ai hi)`.

`velocity_laplacian` instead applies

`du/dt = div(nu grad(u))`, and the same for `v`.

On an active face its primary conductance is
`Kij = nu * face_length * activation / dot(xj-xi,nij)`. Its velocity increment
is the gathered face transfer divided by cell area, and its momentum increment
is that velocity increment multiplied by the unchanged local depth. At variable
depth this mode does **not** conserve
depth-weighted momentum; that is an intentional consequence of selecting a
plain velocity Laplacian, not a budget defect. It conserves the area-weighted
mean of each velocity component on every connected active-face region.

The velocity-Laplacian face activation also requires both cell-centre depths to
exceed the hydraulic dry tolerance. For those two wet centres, BRAIDS computes
the current bed `zi` from bedrock plus every active and substrate inventory,
sets `eta_i = zi + hi`, and evaluates the shared-face water column

`hface = (eta_i + eta_j)/2 - max(zi,zj)`.

The base conductance is multiplied by
`clamp((hface - 0.5*hdry)/(0.5*hdry), 0, 1)`. Thus a perched face at or below
half the dry tolerance is closed, a face at or above the dry tolerance retains
the original arithmetic, and the interval between them opens continuously.
The mask is symmetric and nonnegative. It is stateless: AMR and restart derive
the same mask from the synchronized current state rather than storing a wet/dry
history.

## V35 cut-cell face recovery

New `velocity_laplacian` decks use the v35 spatial operator. A face keeps the
two-point primary flux when it is orthogonal, has equal owner/neighbor
centroid-to-face normal distances, and neither cell touches a no-flux face.
A skew, asymmetric, or no-flux-adjacent face instead uses a two-ring Cartesian
quadratic recovery evaluated at the physical face midpoint.

For cell `i`, let `s=sqrt(Ai)` and `rho=(x-xi)/s`. The rotation-orthonormal
polynomial basis is

`[rho_x, rho_y, rho_x^2/sqrt(2), rho_x*rho_y, rho_y^2/sqrt(2)]`.

Each unique active wet cell reachable in at most two face-graph edges supplies
a value-difference row with objective weight `1/|rho|^2`. Every external or
inactive face incident directly on `i` supplies a homogeneous-Neumann row at
its midpoint with objective weight `face_length/s`. The latter makes the fit
invariant to splitting one physical boundary into several collinear fragments.
The equilibrated 5-by-5 normal system is pivoted and its inverse residual is
checked. A rank-deficient, nonfinite, or more-than-256-cell patch is never
truncated: each affected face falls back to its primary two-point flux.

Owner and neighbor recovered gradients are both projected on the
owner-outward normal at the shared physical midpoint and averaged. That one
face flux is gathered with opposite signs, so the area-integrated Cartesian
velocity invariant is retained. This is centroid-based consistency: solver
states are finite-volume averages, and arbitrary cut polygons do not yet store
the second moments needed for strict quadratic reproduction of cell averages.

After each accepted, synchronized hydraulic/morphological step, the isolated
diffusion step holds water depth, bed, and every sediment inventory bitwise
fixed. Each cell gathers its face transfers in a fixed order, without parallel
scatter/races. OpenMP face/cell loops are enabled for at least 4096 entries.
Zero coefficient bypasses the entire call, including work-array allocation.

Faces touching cells at/below the hydraulic dry tolerance have zero transfer.
In velocity-Laplacian mode, a perched dry shared face between two nominally wet
cell centres is also closed or smoothly activated by the rule above. External
walls and open boundaries have zero viscous traction. This implements free
slip, not no slip; it does not replace hydraulic boundary conditions or bed
friction. The shoreline is not a velocity sink.

The depth-conservative v33 operator retains its positive-graph SSPRK2 bound and
componentwise maximum principle. V35 is a multipoint operator and does **not**
claim a componentwise maximum principle. Its timestep proposal uses a
conservative L1 coefficient bound for both recovered traces. In addition, one
rotation-invariant factor blends the recovered correction with the dissipative
primary graph flux whenever needed to retain at least 5% of primary graph
dissipation. Every Euler stage and final SSPRK2 average is checked against
`sum(A |u|^2)/2`; a failing or nonfinite proposal is retried at half the step.
The depth-conservative mode dissipates `sum(A h |u|^2)/2`. The diffusion
restriction remains independent of hydraulic CFL.

Hydraulic/source versus diffusion composition is **first-order Lie splitting**;
SSPRK2 subcycling does not make the complete split method second-order in time.
There is no morphological acceleration of momentum diffusion.

## Geometry, AMR, restart and limitations

The v33 depth-conservative and archived v34 velocity-Laplacian fluxes are
two-point schemes. They are consistent on an orthogonal Cartesian mesh but are
not general nonorthogonal reconstructions. V35 corrects the material cut-cell
centroid/face-location defect with the recovery above. Conservation, rotation,
linear/quadratic manufactured fields, random-field dissipation, refinement,
AMR/restart, and thread determinism are separate qualification requirements;
none alone proves accuracy on every agglomerated topology.

AMR diffusion is performed once on the synchronized active-leaf graph, after
hydraulic reflux and before regridding/checkpoint/output. It does not add a
second interlevel time-subcycling/reflux algorithm. Existing tree
synchronization then carries the changed leaf momentum. Short lake/wet-front/
sediment tests check budgets, thread identity and restart; this is not AMR
convergence qualification.

Positive default-formulation viscosity adds deck envelope v33. Disabled decks
and positive v33 decks retain their existing byte streams. Archived nondefault
`velocity_laplacian` decks have a composable v34 mode-1 envelope around v33 and
retain their original Euclidean two-point operation order. Newly written decks
add v35 geometry mode 1 around that complete v34 payload. The formulation,
geometry mode, and coefficient therefore enter the configuration fingerprint,
so restart cannot silently change them. There is no new prognostic/checkpoint
state. Metadata names the v35 discretization
`boundary_quadratic_lsq_face_gradient_v35`. The performance log reports
diffusion wall time, substeps, minimum recovery factor, recovered faces, and
TPFA fallbacks.

## Relationship to unmodified Delft3D

Read-only inspection of the unmodified reference source shows that
`flow2d3d_kernel/src/compute/uzd.f90` uses an implicitly split horizontal
velocity Laplacian for the default smooth-wall `IROV=0` path.
`velocity_laplacian` matches that continuum choice and the fully wet
orthogonal-grid two-point spatial operator; this is why it is named the
Delft-compatible formulation. The collocated face activation above is a
stateless analogue of the endpoint behaviour of Delft3D's staggered `kfu/kfv`
wet-face masks, not an exact clone of their hysteretic state machine. It does
not claim whole-solver or cut-cell equivalence. Delft3D's implicit integration,
curvilinear metric terms, vertical representation, wet/dry masks and
open-boundary handling can still differ. `vihrov.f90` is the different
rough-wall stress branch and is not selected by the bend benchmark. No
Delft3D reference source was modified, ported or compiled.

At constant depth the two continuum formulations coincide. Their discrete
answers need not be bitwise equal on a v35 recovered face; the archived v34
two-point mode remains the exact compatibility reference. At varying depth,
`depth_conservative` has the additional continuum term
`nu grad(log(h)) dot grad(u)` relative to the plain velocity Laplacian. Its
purpose is conservation of integrated depth-averaged momentum. Users must make
that physical choice explicitly; neither formulation is a numerical alias for
the other.

Small-case validation and the 120 s bend must precede any claim that either
formulation cures observed bank-cell artifacts or is ready for a release.
Secondary-flow coupling and mobile-bed qualification remain separate
milestones.
