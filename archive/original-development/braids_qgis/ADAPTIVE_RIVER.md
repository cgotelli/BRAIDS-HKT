# Semi-implicit adaptive river workspace

Open **General → Open semi-implicit CPU / GPU river workspace**. Both backends
use the same augmented-f-wave midpoint algorithm, including on cropped AMR
cells. Backend choice is explicit; there is no hybrid or CPU fallback.

Sediment transport starts **unchecked**. Prepare and Run work with it unchecked:
this selects hydrodynamics only, with an immobile bed. Check it to select
**Single grain — bedload only**. This adds the general MPM-type bed capacity
and conservative Exner/finite-stock updates, with no secondary currents,
suspension, bank erosion, collapse, viscosity or bedform roughness feedback.
The two bookkeeping layers have the same grain; they are not a grain mixture.

The optional **Combined river bundle** retains the reviewed multi-process setup.
Its JSON controls govern grains, layers, suspension and bank/collapse; equilibrium
bedform feedback is part of that bundle. Do not select it for a bedload-only test.
Simple modes reject nonzero auxiliary controls or incompatible grain inputs
rather than silently ignoring them. The worker validates the JSON again.

## Run and view results

1. Select the domain, terrain DEM and constant discharge/stage boundaries in the
   main dock. Choose the backend in General, mesh sizes, initial stage and clocks.
2. Open the workspace and choose **Use current QGIS domain / DEM**, or load a
   saved setup JSON. Review the setup and select the physics mode explicitly.
3. Prepare, then Run. Stop cancels preparation or execution cooperatively.
4. Select a result variable and use Play or the timeline. Mesh display starts
   off; depth below 1 mm is transparent. Results retain their native mesh epochs.
5. For restart, load the same prepared folder and enter its checkpoint path.
   The backend, physics, inputs and runtime must match. Output goes to a new folder.

Changing the sediment checkbox or mode invalidates preparation. Loading prepared
inputs restores their explicit mode. Saved legacy projects and the historical
native workflow are not silently converted to a different numerical method;
use this General-tab workspace for the shared semi-implicit CPU/GPU solver.

## Scope

Common-clock AMR supports 1–6 nested 2:1 levels, with finest cells at physical
boundaries and MF1. Constant total-discharge and stage boundaries are admitted;
hydrographs, normal/critical boundaries and incompatible inputs are rejected.
Chezy defaults to 45 and inlet sediment feed to zero; review both. Bedload feed
is solid volume per width (m²/s), not total mass or total discharge. Grain and
dry bulk densities are 2650 and 1590 kg/m³. The simple template has one grain
and uniform material; it does not infer sediment properties from a DEM.

## Windows requirements

CPU libraries, CUDA kernels and numerical dependencies are packaged. No compiler,
checkout, old simulation or Delft3D tree is needed. An external Python environment
with NumPy, SciPy, Shapely and GDAL is required; GPU additionally needs CuPy and a
compatible NVIDIA driver. The field inherits the detected interpreter from the
main dock. This PC uses its Miniconda environment; Python is not bundled.

Use a short output path such as Documents/BRAIDS/adaptive_river. Workers isolate
the QGIS Python/GDAL environment, authenticate packaged inputs and runtime files,
and never treat interrupted preparation as complete. Keep prepared inputs and
checkpoints together. Short natural-terrain delivery tests establish integration,
not calibration, equilibrium morphology or field-scale physical validation.
