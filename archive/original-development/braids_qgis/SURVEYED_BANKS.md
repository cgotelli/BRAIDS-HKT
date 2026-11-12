# Surveyed bank erosion in the unified interface

The Bed & banks tab distinguishes FLOW dry-bank sharing from two BRAIDS
surveyed-bank methods. All bank processes are unchecked by default. The
source plugin uses the same semi-implicit CPU/GPU owner; the installed release
must not be described as updated until final package qualification.

| Choice | Computation | Geographic input |
|---|---|---|
| FLOW dry-bank sharing | Redistributes accepted erosion with adjacent dry-bank cells | None |
| Surveyed toe supply | Supplies accepted net toe scour from finite adjacent bank stock | Eligible bank polygons and toe polylines |
| Surveyed toe supply plus retained block release | Adds retained undercut and threshold-triggered finite block release to the same stock | Same polygons and polylines, plus critical undercut in metres |

The surveyed methods currently require bedload-only, noncohesive material and
the finite independent basal rule. Block release can combine with conservative
wet/dry collapse, but not a second FLOW wet-slope flux. Ineligible choices are
disabled; an already selected incompatible combination is reported rather than
silently replaced. These are current integration limits, not universal laws
about which physical processes may coexist.

## Prepare a survey

1. Create polygon and line layers in the calculation domain's projected CRS,
   with metre units. Give each bank zone one unique ID in both layers.
2. Draw the polygon around eligible material and the open polyline along its
   toe. Polygon holes are allowed; use single-part geometries. Source polygons
   must not overlap.
3. In Bed & banks, enable bank erosion, select a surveyed method and choose
   both layers and their matching ID fields. Set the face-mapping tolerance.
4. Press **Use selected survey layers (snapshot)**. Geometry is copied into
   the project. Editing or removing the source layers does not alter that saved
   snapshot. Reimport to apply subsequent geometry or tolerance changes.
5. Set the transfer fraction in (0, 1]. For block release, also set the positive
   critical undercut. Disable the checkbox to turn the process off.
6. Prepare the run and press **Show prepared bank map**. Inspect the actual
   eligible columns and resolved toe faces before running.

The transfer fraction controls the fraction of accepted net scour supplied by
the bank; it is not a second morphological acceleration factor. Existing bed
layers, packing and finite material stock remain authoritative. A survey does
not create a separate material reservoir.

## Interpret the map

Eligible material is selected by actual finest-cell centroids inside the bank
polygon. Toe faces must be internal faces between that material and outside
recipient cells, with midpoints within the explicit line tolerance. Each guide
must resolve to one open connected chain. Walls, branching chains and
disconnected material regions are rejected.

The map is discretized, not an exact geometric copy of the drawn line. The
prepared receipt records polygon area, selected cell area, mapped toe length
and maximum guide distance. The preview distinguishes eligible columns from
toe recipients and overlays the resolved faces in blue.

Survey columns and toe recipients remain at the finest mesh level throughout
the run. The rest of the domain can adapt. A narrow channel can therefore have
little or no coarsening; the protected column count is displayed explicitly.
The toe map stays fixed during the run. This option does not claim automatic
relocation of an arbitrary migrating bank front; physical validation of that
broader behavior is separate from the short integration checks.

Turning sediment or bank erosion off preserves the imported geometry as
inactive editor data, but no bank survey is sent to the numerical owner.
Current projects and checkpoints have strict identities. Older formats are
rejected without migration.
