# Windows alpha 61 release verification

BRAIDS `0.1.0-alpha.61` was rebuilt on 64-bit Windows from the recorded native
sources after the Phase 3 qualification was integrated.

## Artifact identities

- Archive: `dist/windows-x86_64/BRAIDS-0.1.0-alpha.61-QGIS-windows-x86_64.zip`
- Archive SHA-256: `f253af685a5661443d34816fe3d337586c54e5a03ede8597393a271919e2d226`
- Native executable SHA-256: `2e3457eb37175dab15467e6e8dbf470adbbbf34c19a9f759c27eddd1667eb8ff`
- Packaged Phase 2 manifest SHA-256: `79096d1250cedddf884bb32f6f8540498894d7624ce0bcb1cb6a352f0d1d01a0`
- Packaged Phase 3 manifest SHA-256: `af445e3c353d78ae98d3097ee8da9c9cbd1cfef5b6d637cbb117f4a1ea3c4494`
- Archive files verified against the internal manifest: 74

## Runtime smoke test

The executable extracted into the release stage ran the qualified deck-v26
water-depth-scaled active-layer case with two OpenMP threads. It returned zero,
created nonempty state frames, checkpoint and boundary ledger, and produced
frame SHA-256
`2662ac1fe70946f7c0735d06d01798a99613159a40f968632f6555c6b68a7315`.
That is byte-identical to the retained one- and eight-thread Phase 3 evidence.

## Installation

The verified stage was installed into the default QGIS 3 profile. All 74 file
hashes, the native executable identity and the packaged 74/74 Phase 3 status
were rechecked after copying. The previous alpha 60 installation was moved to
a timestamped sibling backup. QGIS was open during the disk update, so it must
be restarted before the running process loads alpha 61.

The archive audit rejects `resources/Delft3D-main`, any raw `runs` directory,
stale native sources, an unexpected DLL closure, unpinned license content, and
a non-passing Phase 3 manifest.
