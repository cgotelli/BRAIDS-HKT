# Windows alpha 77: sloping stage outlet

This release corrects the terrain/water reconstruction and half-cell gravity
balance at a sloping subcritical stage outlet. Flat-bed, dry-face and
supercritical release controls remain in place; critical-depth and weir
outlet modes are unchanged.

The [numerical audit and figures](../validation/sediment/synthetic_expansion/reports/outlet_corrected_v1/AUDIT.md)
document the independent uniform-flow reference, 2 m/1 m comparisons,
corrected Delft3D staggered-face initialization and compiled MPM equivalence.
103 selected tests pass, with one existing archived-baseline test skipped.
The mobile-bed outlet discrepancy and Delft3D oscillations remain unresolved;
this is not a qualification of braided-river morphology.

Package: `dist/windows-x86_64/BRAIDS-0.1.0-alpha.77-QGIS-windows-x86_64.zip`.

Native executable SHA256:
`57059f45e0aa98c9ee52963ca4dda312993adb33005905897cfeed0e0d31736f`.

See [installation evidence](../validation/sediment/synthetic_expansion/reports/outlet_corrected_v1/installation.json)
for the archive hash, verified files and recoverable previous-version backup.
The archive excludes Delft3D source and raw runs. Prepare a new run to use the
fix: saved results and old prepared runtimes do not change retroactively.
