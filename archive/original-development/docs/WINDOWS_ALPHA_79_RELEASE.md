# Windows alpha 79: hydraulically consistent sediment export

Local-equilibrium MPM export at subcritical stage outlets now uses the
hydraulic face trace, not the adjacent cell-centre capacity. This fixes the
artificial outlet sediment deficit without forcing zero deposition or
suppressing physical fixed-stage reflection. Other transport closures,
non-equilibrium options and supercritical-release controls are unchanged.

See the [audit, acceptance gates and comparison figures](../validation/sediment/synthetic_expansion/reports/stage_capacity_refinement_v1/AUDIT.md).
This scoped correction does not qualify full synthetic-expansion morphology
or establish Delft3D morphology equivalence.

Package: `dist/windows-x86_64/BRAIDS-0.1.0-alpha.79-QGIS-windows-x86_64.zip`.

Production executable SHA256:
`1a9fa196c39d0318676789949c264c710e7347e8755cf572c8702b06a502c187`.

The [installation record](../validation/sediment/synthetic_expansion/reports/stage_capacity_refinement_v1/installation.json)
records archive identity, verified installed files and a recoverable previous
plugin backup. Delft3D source and raw simulation runs are excluded from the
ZIP. Restart QGIS and prepare a new run; existing results and previously
prepared runtimes are not modified.
