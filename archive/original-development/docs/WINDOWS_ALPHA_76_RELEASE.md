# Windows alpha 76: coupled MPM bed-wave stability

The source fix replaces sediment-flow-only upwinding with a coupled
shallow-water/Exner characteristic flux for wet interior local-equilibrium
MPM faces. It corrects the large cell-alternating scour/deposition in the
synthetic 1%-slope expansion. It does not smooth results after computation.

See the [numerical audit](../validation/sediment/synthetic_expansion/reports/coupled_fix_v1/AUDIT.md)
for the root cause, rejected alternatives, refinement tests, same-resolution
Delft3D reference and limitations. This is not a new qualification of every
experimental sediment process or the integrated braided-river benchmark.

Verification this turn: 202 selected regression tests passed, one existing
test skipped; 21 native bed-eigenwave cases passed accuracy/amplitude/mass
checks, with decreasing error on refinement. The 900 s, 4 m, MorFac 10
expansion preserves sediment mass within 1.81e-6 kg. Eight-thread/serial
and restart/uninterrupted expansion checks are bitwise identical.

Package:
`dist/windows-x86_64/BRAIDS-0.1.0-alpha.76-QGIS-windows-x86_64.zip`

Archive SHA-256:
`6ff0d5bacf96d8f5096f833f1000ada6a8d7bf27bc0081e5180b5bcdc38e152e`

Native executable SHA-256:
`d5322746a37a516ca0fd04d24a300080f3dfc382e1a2244c3f14385966bac7ca`

Installed in the default Windows QGIS profile. The staged archive and
installed plugin passed verification of all 81 files, including 62 source
files; the prior plugin was preserved in `.braids_release_backups`.
The archive excludes Delft3D source and raw simulation runs.
See [installation evidence](../validation/sediment/synthetic_expansion/reports/coupled_fix_v1/installation.json).

Open QGIS to use alpha 76. Old run results do not change: prepare a new run
from the original initial bed to compare the corrected numerical method.
