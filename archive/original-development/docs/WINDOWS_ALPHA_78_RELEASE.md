# Windows alpha 78: stable subcritical stage-outlet bed waves

The coupled local-equilibrium MPM water/Exner flux remains active beside
subcritical stage outlets. The previous midpoint-only buffer could amplify
an exiting bed wave at MorFac 10. Physical stage reflection is retained and
verified against the linearized coupled equations. Non-stage buffers and
supercritical free-release equivalence remain unchanged.

See the [audit and comparison figures](../validation/sediment/synthetic_expansion/reports/outlet_buffer_v2/AUDIT.md).
This is a scoped numerical correction, not a qualification of the full
synthetic expansion or a claim of Delft3D morphology equivalence.

Package: `dist/windows-x86_64/BRAIDS-0.1.0-alpha.78-QGIS-windows-x86_64.zip`.

Production executable SHA256:
`78a3fe8e58b67ecfa6f1ad1e42a6c9f673fe662af095b556c139d560851930a4`.

The [installation record](../validation/sediment/synthetic_expansion/reports/outlet_buffer_v2/installation.json)
records the archive hash, verified installed files and recoverable previous
plugin backup. Delft3D main source and raw validation runs are excluded from
the ZIP. Restart QGIS and prepare a new run; existing results and previously
prepared native runtimes are not modified.
