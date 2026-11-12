"""Final delivery record; runtime and child-process checks must already pass."""
import json,zipfile,xml.etree.ElementTree as ET
from stage_collapse222 import ROOT,OUT,BASE,TARGET,FILES,REPORT,sha,save,verify

def main():
    install=json.loads((REPORT/'INSTALL.json').read_text());verify(BASE)
    live=json.loads((OUT/'INSTALLED_LAUNCH.json').read_text())
    assert live['runtime']==install['runtime']==sha(BASE/'RUNTIME.json')
    m=json.loads((OUT/'METRICS.json').read_text());ex=json.loads((OUT/'EXAMPLE.json').read_text())
    comparison=json.loads((OUT/'EXAMPLE_COMPARISON.json').read_text())
    reg=ET.parse(OUT/'regression3.xml').getroot().find('testsuite')
    assert not any(int(reg.get(k)) for k in ('failures','errors','skipped'))
    tests=176+int(reg.get('tests'))
    backend=max(r['max_bed_mm'] for r in m['paired']);temporal=max(r['max_bed_mm'] for r in m['temporal'])
    save(REPORT/'INSTALLED_LAUNCH.json',live)
    report=f'''# Desktop222 — compatible conservative collapse

Completed 2026-10-29. **Desktop preview46 / project v34 installed.**
The two existing collapse choices are available in **05 Morpho evolution →
General → Bank and slope collapse**. Sediment and collapse remain off in a
new project. Ordinary collapse now composes with all supported ordinary
bedload/suspension combinations and positive MF. Jiang–Cao retains its separate
projection and MF1 restriction. No new transport law or hydraulic method was added.

Runtime: `{install['runtime']}`.
Native binaries, physical transport equations, reconstruction, augmented-f-wave
SSPRK22 integration and collapse algorithms are unchanged. Admission and the
explanations in the UI have changed. CPU/GPU choice remains explicit.

## What each model actually does

| Choice | Geometric rule | Material/water accounting | MF |
| --- | --- | --- | --- |
| BRAIDS conservative wet/dry collapse | If an internal neighbour slope exceeds the trigger, redistribute toward equal elevation, bounded by donor stock. One deterministic sweep per accepted update; subsequent updates can activate further pairs. | Moves existing bed mass with actual polygon areas. No exterior transfer, no direct suspended source, unchanged water volume. | Does not multiply collapse by MF. Bedload and wet exchange retain their separate MF clock. |
| Jiang–Cao repose projection | Iterates until both centroid-pair slopes and the full capacity gradient satisfy the bundle repose criterion, with its documented numerical margin. | Coupled bed solid/pore-water accounting; mobile sediment is not duplicated. Insufficient movable stock to meet the target rejects the trial. | MF1 only, within the Jiang–Cao bundle. |

The trigger slope in the ordinary model is **not its final target slope**.
It may flatten an activated pair considerably more than the repose projection.
Neither model is geotechnical undercut/block failure. Do not interpret either
as a measured bank-retreat law or a validated braided-river experiment.

![Different collapse targets](COLLAPSE_METHODS.png)

The algebraic figure uses the same two unequal-area cells and initial bed for
both target rules. It is not a new Delft3D simulation.

## Full supported transport compatibility

Six independent suspension choices in the table below are: **supplied / settling
only; García–Parker 1991; de Leeuw 2020; Van Rijn 1984; Zyserman–Fredsøe 1994;
Camenen–Larson 2008 (current only)**.

| Active bedload | No suspension + ordinary collapse | Any of the six + ordinary collapse | FLOW Van Rijn 1993 suspension + ordinary collapse |
| --- | --- | --- | --- |
| MPM | Yes | Yes | No — select matching Van Rijn 1993 bedload |
| Wong–Parker 2006 | Yes | Yes | No |
| Ashida–Michiue | Yes | Yes | No |
| Wang–Fredsøe | Yes | Yes | No |
| Smart–Jäggi | Yes | Yes | No |
| Ancey–Recking 2023 | Yes | Yes | No |
| Recking 2010 | Yes | Yes | No |
| Camenen–Larson 2005 | Yes | Yes | No |
| Van Rijn 1993 | Yes | Yes | Yes |
| Bedload OFF | — | Yes | Yes |

That is **9 bedload-only + 7 suspended-only + 55 paired = 71** ordinary
configurations, each executed with active collapse on CPU and GPU at MF25.
Each bedload-on case produced nonzero transport. Native algebraic suspended
capacity was not added on top of independently transported suspension.
Required measured diameter descriptors, shear/resistance rules and applicability
limits remain those of the selected formulas; collapse needs bulk density,
finite bed stock and geometry, not an extra diameter or transport closure.

**Jiang–Cao is a separate 72nd supported setup:** its own mobile-bedload/phase
exchange bundle, its own repose projection, MF1, no independent suspension or
ordinary collapse. It was rechecked with current-package CPU/GPU worker,
AMR/restart/cancellation tests and independent repose/stock/rotation oracles.
Compatibility does not extend any empirical formula beyond its published range.

Bank/toe erosion, sorting, extra bedform feedback and the legacy FLOW wet-slope
flux remain outside the newly admitted suspension composition. These are
qualification restrictions, not a claim that the processes are physically
incompatible in general. No model is silently substituted. Headless preparation
and the UI share collapse-owner checks; an inactive Jiang–Cao editor draft no
longer determines the collapse allowed for suspended-only transport.

## Execution audit

1. Apply conservative bedload divergence on the bed clock.
2. Advect suspended mass using accepted physical water flux; calculate the named
   entrainment/deposition source against remaining finite stock divided by MF.
3. Accumulate the explicit MF acceleration ledger and update one unpublished bed
   column. Dry stranding occurs once without MF multiplication.
4. Redistribute that updated bed inventory across internal physical neighbours.
   Do not release the same transferred mass directly into suspension.
5. Rebalance columns, verify local/global transfers and publish bed plus all
   ledgers once. Rejected/cancelled trials roll back together. Remeshing carries
   both cumulative cell transfers and gross totals conservatively.

Read-only source inspection of FLOW `compute_sediment/avalan.f90` confirms its
legacy threshold/equalization rule: `0.25 * dz * (Aa + Ab)`. BRAIDS already uses
`abs(dz) / (1/Aa + 1/Ab)`, equal on equal-area cells and bounded on unequal areas,
plus a finite-stock limit and actual internal adjacency. FLOW's source comments
explicitly flag missing stock/composition constraints and sweep asymmetry.
BRAIDS also retains a deterministic ordered sweep: this milestone does not
claim rotational invariance of an arbitrary many-cell collapse trajectory or
remove that ordering limitation. The source SHA is in `METRICS.json`; Delft3D
was neither modified nor rerun for a new trajectory comparison.

## Verification and limits

**{tests} distinct passing tests:** 176 collapse/admission/delivery checks and
{reg.get('tests')} MF/Jiang–Cao regressions. Short tests include fixed and moving
AMR meshes, cropped walls, partly dry banks, finite/zero initial erodible stock,
MF1/MF25, cancellation, rollback, exact restart and timestep halving.

| Measured check | Maximum |
| --- | --- |
| Bed difference CPU/GPU, 71-case matrix | {backend:.3g} mm |
| Bed difference after halving dt | {temporal:.4f} mm |
| Sediment budget residual | {m['max_sediment_residual_kg']:.3g} kg |
| Water budget residual | {m['max_water_residual_m3']:.3g} m³ |
| CPU/GPU bed difference, portable example | {comparison['cpu_gpu_max_bed_mm']:.3g} mm |

![Backend and timestep checks](QUALIFICATION.png)

Four portable examples and their midpoint restarts also ran through actual
Desktop child processes, then restarted through the installed runtime with
exact final fields. The test matrix is a composition check, not experimental
calibration, universal MF accuracy or proof of long-time bank migration.

Historical test helpers initially imported files absent from the release.
Their failed receipts are retained. Self-contained geometry/material fixtures
replace those imports; every failed node was rerun successfully. The merged
176-test receipt links the originals in `QUALIFICATION_RECEIPT.json`.
The attempted historical JC200 collection also remains as `regression2.xml`;
current-package repose oracles and the actual JC worker provide the passing
regression3 evidence. No historical missing simulation results were invented.

## Openable bank comparison

Open `desktop/examples/collapse222` and choose CPU MF25 collapse, its no-collapse
control, CPU MF1 or GPU MF25. Each folder contains its DEM, parameters, prepared
mesh, results, logs and checkpoints. Convenience CMD launchers are provided.
The short example is a closed, partly dry bank with cropped walls, MPM plus
García–Parker and a 0.3 m/m collapse trigger, run for **0.4 hydraulic seconds**.

![Unsmoothed bank comparison](BANK_COMPARISON.png)

Without collapse the bank changes only slightly over this short interval.
With collapse, high cells supply lower neighbours; the erosion and deposition
balance internally. MF1 and MF25 each redistribute approximately 50.74 kg,
because MF does not multiply collapse. Their small subsequent differences come
from accelerated bedload/exchange and its flow feedback. CPU/GPU curves overlap.
Colors show actual cell values, without smoothing around cropped walls.

![Morpho evolution controls](UI_COLLAPSE.png)

The previous preview45 runtime delta is preserved in
`desktop/qualification/collapse222/preview45_backup`. Existing user projects and
QGIS were not changed. Project v34 rejects previous formats under the agreed
no-backward-compatibility policy; the supplied examples are new v34 projects.

## Next milestone

Qualify the existing bank/toe-erosion routes with the supported transport
formulations, keeping local dry-bank sharing distinct from surveyed toe supply
and retained block failure. Establish finite-stock, clock and suspension
compatibility before exposing additional combinations in Morpho evolution.

## Immediate next step

Trace the current bank owners' supply/release ledgers on a short, closed
erodible-bank case. Start with bedload-only MF1 on CPU/GPU, compare transfer
direction and mass with the read-only FLOW dry-bank source/operator, then add
the supported suspension and MF contrasts without double-counting collapse.
'''
    (REPORT/'REPORT.md').write_text(report,encoding='utf-8')
    for row in ex['cases']:
        case=row['case'];label=case.replace('_',' ')
        target=ROOT/'desktop/examples/collapse222'/('Open '+label+'.cmd')
        target.write_text('@echo off\ncall "%~dp0..\\..\\..\\Launch BRAIDS Desktop.cmd" "%~dp0'+case+'"\n',encoding='utf-8')
    status=ROOT/'validation/RIVER_MORPHODYNAMICS_STATUS.md'
    text=f'''# Desktop222 — conservative collapse delivered (2026-10-29)

Preview46 / project v34 installed. Both collapse choices remain explicit in
Morpho evolution. Ordinary finite-stock collapse is qualified with all 71
ordinary bedload/suspension setups; MF scales transport/exchange, not collapse.
Jiang–Cao keeps its own repose projection, MF1 and no independent suspension.
{tests} checks pass, plus four child-process examples with exact installed restarts.
Largest timestep bed difference: {temporal:.4f} mm. No hydraulic/transport law,
native binary, QGIS or Delft3D changes. Sediment and collapse default OFF.

| Feature | Current status |
| --- | --- |
| Nine ordinary bedload formulas | Available, original formula/diameter rules retained |
| Seven suspension formulas and 55 bedload pairings | Available with named MF acceleration |
| Ordinary conservative collapse | Available with all 71 ordinary transport configurations |
| Jiang–Cao repose | Available only in its coupled bundle, MF1 |
| Bank/toe erosion and retained block failure | Existing components; next compatibility/qualification milestone |
| Sorting, mixtures and other morphology combinations | Deferred; not silently enabled |
| Experimental braided-river validation | Later, after component integration/qualification |

Report: `sediment/river_morphodynamics/reports/desktop_collapse222/REPORT.md`.
Examples: `desktop/examples/collapse222`.
Next milestone: compatible bank/toe erosion and retained-collapse qualification.
Immediate next step: trace finite-stock supply/release on a short closed bank,
starting bedload-only MF1 on both backends and reading the FLOW reference source.

---

'''
    status.write_text(text+status.read_text(encoding='utf-8'),encoding='utf-8')
    readme=ROOT/'desktop/README.md';s=readme.read_text(encoding='utf-8')
    s=s.replace('BRAIDS 0.1.0-preview45','BRAIDS 0.1.0-preview46').replace('new v33 project','new v34 project')
    s=s.replace('and leave new bank/collapse,\nsorting, bedform and slope extras off.', 'and optionally enable conservative collapse in Morpho evolution. Leave bank erosion,\nsorting, bedform and slope extras off.')
    s=s.replace('Bank/collapse combinations\nwith acceleration remain blocked pending the next compatibility qualification.', 'Ordinary collapse is now qualified with positive MF; bank erosion with acceleration remains blocked.')
    s=s.replace('Open `desktop/examples/morfac221/cpu_MF25` (or `gpu_MF25`) as a project folder\nto inspect the short MPM + Garc?a?Parker demonstration and resume its checkpoint.', 'The `desktop/examples/morfac221` MF demonstrations are retained v33 history, not current v34 inputs. Use `desktop/examples/collapse222` in preview46.')
    s=s.replace('New morphodynamic-feature development remains deferred.', 'Bank/toe erosion is the next morphodynamic compatibility milestone; conservative collapse is available now.')
    s=s.replace('project/model contracts are v32/v3', 'project/model contracts are v34/v3')
    s=s.replace('The usual CMD shortcut opens preview44. Open the project folder\n`desktop/examples/supplied220` for a short supplied-sediment deposition example,\nincluding its results, parameters and checkpoints. See the\n[current report and comparison figures]', 'The usual CMD shortcut opens preview46. The older `desktop/examples/supplied220`\nfiles remain historical v32 evidence, not current inputs. See the\n[earlier report and comparison figures]')
    pos=s.index('## Morphological factor')
    s=s[:pos]+'''## Compatible collapse in preview46

In **Morpho evolution → General → Bank and slope collapse**, choose ordinary
conservative wet/dry redistribution with any supported ordinary transport setup,
or Jiang–Cao repose with its own bundle at MF1. The ordinary trigger slope is in
m/m: over-steep pairs redistribute toward equal elevation, bounded by available
bed stock. Collapse is not multiplied by MF and does not inject a second sediment
source into suspension. Bank/toe erosion remains the next qualification step.
Both sediment and collapse are off by default. New projects use v34.

Open `desktop/examples/collapse222` for short cropped-bank comparisons, results
and checkpoints, with CMD launchers. See the full compatibility matrix and figures
in `validation/sediment/river_morphodynamics/reports/desktop_collapse222/REPORT.md`.

'''+s[pos:];readme.write_text(s,encoding='utf-8')
    # Archive source/tests/reports; no vendor reference code or old run outputs.
    archive=OUT/'SOURCE_AND_REPORT.zip'
    sources=[ROOT/p for p in FILES]+[ROOT/'tests'/p for p in ('test_collapse222.py','test_collapse_delivery222.py','test_collapse_repose222.py')]
    sources+=list((ROOT/'desktop').glob('*collapse222.py'))+list(REPORT.iterdir())
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sources:
            if p.is_file():z.write(p,str(p.relative_to(ROOT)))
    save(OUT/'COMPLETE.json',dict(runtime=install['runtime'],version='0.1.0-preview46',distinct_tests=tests,
        installed_child_process_restarts=True,report=str(REPORT/'REPORT.md'),archive_sha256=sha(archive)))
    print('Milestone222 complete:',tests,'passing checks',flush=True)
if __name__=='__main__':main()
