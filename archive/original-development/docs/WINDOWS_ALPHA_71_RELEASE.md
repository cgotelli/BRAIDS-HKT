# BRAIDS Windows alpha 71 release

BRAIDS `0.1.0-alpha.71` fixes the evolving live normal-depth section exposed by
the 64,496-cell experimental flume run and makes long simulations recoverable.

The failed alpha-70 run stopped deterministically at `t=40.299284894 s` because
the lowest evolving section endpoint was treated as an open spill edge. In the
polygon model the adjoining unassigned boundary edges are reflective. The
native section now adds their vertical wetted sidewalls, using the adjacent
roughness subsection, and extends the normal-depth search only when the former
endpoint cap cannot carry the requested flow. A direct replay from the retained
40 s state reaches 40.5 s, satisfies the requested Manning discharge, and
preserves every earlier production-section result.

Every completed output frame now refreshes `checkpoint.bin`. A stop request or
a later model failure therefore leaves the last displayed state restartable.
The checkpoint layout is unchanged, and fixed-grid/AMR restart and thread
identity tests remain exact.

The bedload-only native path also resolves its suspended-transport activity
once from the complete forcing schedule and initial/restart inventory. The live
normal-section root search visits branches from low to high and stops at the
first admissible root; parallel work remains in the large cell, face and
fraction kernels where it provides useful speedup.

The repaired 64,496-cell confirmation reaches 60 s in 49,914 accepted steps.
Water closes to `1.08e-15 m3` and every sediment fraction passes its mixed
budget tolerance. The 0.025/0.0125 m pair has 0.0907 mm bed MAE and a 3.15%
difference in native-grid integrated absolute bed-change volume, passing the
frozen 2 mm and 10% limits. The 0.025 m grid is accepted for H2 calibration.

In the exact three-repeat 1/8/16/32-thread screen, the new path is 2.25x faster
for the complete one-core pilot and 15.26x faster in boundary control. Eight
threads are fastest for that compact 4,031-cell case at 1.18x versus one core;
its diagnostic 1.25x target is not met. The 64,496-cell continuation launches
all 32 workers and averages about 14.5 effective cores. Its profile identifies
bedload capacity, Euler setup/reconstruction, and face flux as the next
large-grid optimization targets.

The Windows package contains neither `resources/Delft3D-main` nor raw
validation run directories.

## Verified Windows artifact

- Archive: `dist/windows-x86_64/BRAIDS-0.1.0-alpha.71-QGIS-windows-x86_64.zip`
- SHA-256: `ca68d721cf60d187a440f12e00a968c3b941ff8dd0f4d405964b345a6e694bc7`
- Size: 3,803,104 bytes
- Verified packaged files: 81
- Native executable SHA-256:
  `30958f08ef92eda6bde94fff8efdd5d4816bfed4edd43e390532cf25cd2ac85b`

The staged manifest and installed manifest are identical. Alpha 71 is installed
in the default QGIS profile; the prior alpha-70 directory was moved to the
timestamped sibling `braids_qgis_alpha70_backup_20260924_071622` and remains
recoverable.

The final focused Windows regression contains 149 passing tests and one
intentional skip. It covers the experimental benchmark and ledger recovery,
production sections, multi-fraction and hindered-settling transport, AMR and
restart paths, inlet/outlet and terrain boundaries, result budgets, output
step hints, and archive verification.
