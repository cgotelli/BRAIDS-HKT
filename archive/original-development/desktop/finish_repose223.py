"""Measured release report, source archive and consistent next handoff."""
import json,zipfile
from stage_repose223 import ROOT,OUT,BASE,TARGET,FILES,REPORT,sha,save

def main():
    installed=json.loads((REPORT/'INSTALL.json').read_text())
    live=json.loads((OUT/'INSTALLED_LAUNCH.json').read_text())
    assert live['runtime']==installed['runtime']==sha(BASE/'RUNTIME.json')
    m=json.loads((OUT/'METRICS.json').read_text());ex=json.loads((OUT/'EXAMPLE.json').read_text())
    comparison=json.loads((OUT/'EXAMPLE_COMPARISON.json').read_text())
    tests=sum(r['tests'] for r in installed['tests'].values())
    backend=max(r['max_bed_mm'] for r in m['paired']);temporal=max(r['max_bed_mm'] for r in m['temporal'])
    save(REPORT/'INSTALLED_LAUNCH.json',live)
    save(REPORT/'PUBLISHED_SOURCE.json',dict(paper='chino.pdf',sha256=sha(ROOT/'chino.pdf'),
        doi='10.1016/j.ijsrc.2026.08.002',pages_used=[3],
        supplement='sup.pdf',supplement_sha256=sha(ROOT/'sup.pdf'),
        interpretation='geometric repose criterion; BRAIDS finite-volume realization, not author code'))
    report=f'''# Desktop223 — repose collapse across transport formulas

Completed 2026-10-29. **Desktop preview47 / project v35 installed.**

Yes: the Jiang–Cao **geometric repose criterion** can accompany the other
transport formulas. The restriction stated in milestone222 was too broad.
It described the coupling that had been qualified, not a necessary dependence
of the slope-collapse criterion on the Jiang–Cao transport law.

## Source and method identity

Read-only source: supplied `chino.pdf`, Jiang and Cao (2026), article page 3,
doi:10.1016/j.ijsrc.2026.08.002 (publisher PII S1001627926000673).
The text describes downslope sediment movement when local slope exceeds the
submerged repose angle and cites Jang and Shimizu (2005). Its model M4 removes
the first three coupling-source components while retaining collapse; M5 also
removes secondary flow and collapse. This supports separating the geometric
criterion from the fully coupled phase model. No unique diameter-to-angle law
is supplied there, so BRAIDS asks for an explicit angle and does not derive it
from D50, D84 or any other transport diameter.

BRAIDS retains its named finite-volume realization, not a claim to reproduce
the authors' unavailable code. The projection uses physical shared faces,
actual cell areas and centroids, finite donor stock, centroid-pair slopes and
the full vector least-squares slope. The ordinary adapter uses **unique physical
neighbours** in that gradient, avoiding duplicate weighting at merged faces.
Native OpenMP/CUDA kernels are unchanged. Rotation, unequal-area and full-vector
tests check that a north–south mesh does not become an axis-only repose test.

| Selected context | Resolved owner | Inventory, water and MF |
| --- | --- | --- |
| Ordinary bedload, independent suspension, or their supported pairing | `redistribution.jiang_cao_repose_exner223_v1` | One finite Exner bed inventory after transport/exchange; water volume and momentum are unchanged by collapse; no direct suspended source; collapse is not multiplied by MF. |
| Original Jiang–Cao coupled transport bundle | `redistribution.jiang_cao_repose200_v1` | Original mobile/bed solid and pore-water coupling, original strict repose requirement and MF1 restriction remain. |
| Existing BRAIDS conservative wet/dry choice | Existing ordinary owner | Threshold-triggered neighbour equalization, not a projection to a nonzero repose slope. Retained unchanged. |

The ordinary repose adapter allows an immobile substrate to retain a steep
slope after movable stock is exhausted. This is recorded in `DIAGNOSTICS.json`
under `repose.last_projection`; a fresh mesh/initial state has no previous
projection entry. It does not erode rock to satisfy a geometric target.
One prescribed angle is used for wet and dry cells; distinct wet/dry angles,
cohesion, pore-pressure failure and kinetic block collapse are not represented.

## Compatibility and UI

In **05 Morpho evolution → General → Bank and slope collapse**, enable
redistribution, select **Repose projection (Jiang–Cao criterion)** and enter
the repose angle in degrees. The UI disables the unused trigger-slope field.
For the original coupled Jiang–Cao bundle its Bedload repose parameter remains
the owner, and the ordinary Exner angle field is disabled. **Resolved setup**
explains the selected inventory coupling, assumptions and restrictions.
Sediment and collapse both remain OFF in a new project.

![Explicit repose controls](UI_COLLAPSE.png)

All **71 ordinary configurations** are covered on both backends:

| Bedload | Alone with repose | With each of the six independent suspension choices | With FLOW Van Rijn 1993 suspension |
| --- | --- | --- | --- |
| MPM | Yes | Yes | Not a valid transport pairing |
| Wong–Parker (2006) | Yes | Yes | Not a valid transport pairing |
| Generalized Ashida–Michiue | Yes | Yes | Not a valid transport pairing |
| Wang–Fredsøe | Yes | Yes | Not a valid transport pairing |
| Smart–Jäggi | Yes | Yes | Not a valid transport pairing |
| Ancey–Recking (2023) | Yes | Yes | Not a valid transport pairing |
| Recking (2010) | Yes | Yes | Not a valid transport pairing |
| Camenen–Larson (2005) | Yes | Yes | Not a valid transport pairing |
| Van Rijn (1993) | Yes | Yes | Yes |
| No bedload | — | Yes | Yes |

The six independent suspension choices are supplied/settling-only,
García–Parker 1991, de Leeuw 2020, Van Rijn 1984, Zyserman–Fredsøe 1994 and
Camenen–Larson 2008 (current only). Formula diameters, shear, roughness,
settling and MF acceleration rules are unchanged. Adding collapse does not
extend a transport formula's experimental applicability.

The new Exner coupling currently requires one bulk material, a finite fully
mixed bed and dry-bulk-density packing. Layered/mixture and bank/toe combinations
are explicitly rejected before mesh preparation as qualification gaps.
The geometric rule itself is not inherently incompatible with those features.

## Checks and measured differences

**{tests} passing final checks**, with no failures or skips in the final
receipts. These include the 142 coupled formula/backend runs, finite-stock
and density oracles, rotation/cropped geometry, MF1/MF25, dynamic AMR,
cancellation/rollback, exact restart, timestep halving, original Jiang–Cao
regressions and the first FLOW dry-bank operator audit.

| Check | Result |
| --- | --- |
| CPU/GPU maximum bed difference over 71 configurations | {backend:.6g} mm |
| Maximum bed difference when halving timestep | {temporal:.6g} mm |
| Maximum water residual | {m['max_water_residual_m3']:.6g} m³ |
| Maximum combined bed/suspended sediment residual | {m['max_sediment_residual_kg']:.6g} kg |
| Restart/rollback cases in the new suite | {m['exact_restart_cases']} |
| Portable examples through actual Desktop child processes | 5, each with exact candidate and installed restart |

![CPU/GPU and timestep checks](QUALIFICATION.png)

The zero-erodible-bed test exposed negative subnormal stock after repeated
donor exhaustion (about −5e−324 to −4e−323 m of solid-volume depth).
The new ordinary adapter retains donors below the representable relative-margin
range (`float64.tiny / epsilon`, about 1e−292) rather than sending rounded
transfers that can overdraw them. It deletes no stock and clips no bed elevation.
This explicit ordinary finite-stock policy leaves the original strict coupled
Jiang–Cao path unchanged. Failed exploratory receipts and the captured stock/flux
trace remain in `desktop/qualification/repose223`.

## Why the bank shapes differ

Five portable projects in `desktop/examples/repose223` start from the same
partly dry, finite raised bank and cropped domain. They compare no collapse,
ordinary equalization, repose at MF1, and repose at MF25 on CPU and GPU, using
MPM plus García–Parker suspension for 0.4 hydraulic seconds. Equalization uses
the same 20° trigger (converted to slope) as the repose target for a fair
activation comparison. These are deliberately short numerical checks, not
an experimental bank-retreat validation or equilibrium river simulation.

![Actual cell bank changes and cross-sections](BANK_COMPARISON.png)

The two collapse choices intentionally differ: an activated equalization pair
is driven toward a flat bed, whereas repose retains a finite stable slope.
MF does not multiply either redistribution; MF1/MF25 differences arise from
their different transport/exchange clocks and subsequent flow feedback.
CPU/GPU example maximum bed difference is {comparison['cpu_gpu_max_bed_mm']:.6g} mm.
Figures retain raw cell outlines and unsmoothed bed values near cropped walls.

![Independent unequal-area target comparison](COLLAPSE_METHODS.png)

## The bank/toe milestone has started

Read-only inspection of FLOW `bott3d.f90` lines 870–990 shows erosion sharing
only for **net total erosion** in a sediment-active wet cell. It draws from
higher inactive neighbours, weighted by available composition, and uses the
minimum participating cell area. Its bed delta already contains `dt × MF`
(line 826); sharing must not multiply by MF again.

BRAIDS `RiverPolygonBankSolver.column_delta` follows that net-erosion inventory
logic on actual internal polygon neighbours. Twelve CPU/GPU operator checks
cover equal/unequal/tiny areas, rotation, finite donors, deposition exclusion
and wall exclusion against independently evaluated FLOW arithmetic.
This is source/operator agreement, **not a new FLOW simulation**.

Surveyed physical-toe supply is a different BRAIDS method: it consumes accepted
net Exner demand on explicitly mapped internal bank/toe faces, limits withdrawal
by stock and geometry, and can feed retained block release before parent-owned
redistribution. Its current installation guard excludes independent suspension.
The shared column transaction must remain the sole owner of all these transfers.
Detailed audit and bounded continuation are in [BANK_AUDIT.md](BANK_AUDIT.md).

Runtime SHA256: `{installed['runtime']}`. The preview46 delta is retained at
`desktop/qualification/repose223/preview46_backup`. Old user projects were not
modified; v35 rejects older project/checkpoint identities. The supplied v35
examples are ready to open. No QGIS or Delft3D source/runtime changes.

## Next milestone

Qualify bank/toe erosion with the supported transport formulations and both
collapse choices, distinguishing FLOW dry-bank sharing from surveyed toe supply
and retained block failure. Require finite stock, single-counted exchange/MF,
CPU/GPU agreement, restart and diagnostic figures before public admission.

## Immediate next step

Run a short closed, erodible-bank MF1 bedload-only case through the actual
general worker on CPU/GPU. Trace accepted Exner demand → bank supply → optional
collapse → column commit. Compare the sharing operator with the locked FLOW
source first; then add suspension and MF contrasts only after the ledger closes.
'''
    (REPORT/'REPORT.md').write_text(report,encoding='utf-8')
    bank='''# Bank/toe source audit and continuation

Read-only reference: `resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute_sediment/bott3d.f90`.
Its digest is recorded in METRICS.json; no Delft3D source or executable was changed.

| Stage | Current BRAIDS owner | FLOW comparison / next check |
| --- | --- | --- |
| Accepted bed demand | river_suspension.py: process_bed_delta adds suspended exchange to class_delta after bedload | bott3d forms net bed change from transport divergence and exchange, with dtmor. Do not use gross entrainment or theoretical capacity. |
| Dry-bank sharing | RiverPolygonBankSolver.column_delta; graph_bank_propose/scale/gather | Same-input one-grain oracle passes on CPU/GPU for min-area scaling, higher dry donors, net erosion and finite stock. Full suspension/MF composition is still to qualify. |
| Surveyed toe supply | PhysicalToeSupply.apply after transport columns | Explicit internal-face map, accepted net demand, stock and geometric limits; not claimed to be FLOW dry-bank erosion. Current suspension guard remains. |
| Retained block release | physical-toe failure proposal, history/count trial | Must not supply the same stock twice; prove clock/history and conservation before exposure. |
| Slope redistribution | Parent finish: choose equalization OR repose | Uses remaining stock; no second MF multiplier. New repose + bank owner currently rejected pending combined qualification. |
| Commit / rollback / AMR | One column transaction and existing process ledgers | Extend actual worker restart and regrid checks; no synthetic capacity-only pass can admit coupled bank physics. |

The current ordinary suspension path already merges net suspended exchange into
class_delta before bank sharing; the surveyed toe path explicitly rejects it.
This is a qualification/integration distinction, not proof that suspension and
bank erosion are physically incompatible. Begin with bedload-only MF1, then
settling-only/entraining contrasts, then MF and both collapse choices. No long
river or braided-experiment run is needed for this first ledger audit.

The operator oracle supplies identical wet/dry flags; it does not establish
identical hydraulic activation. BRAIDS measures wet activity on actual polygons
using its reconstructed bed and sediment-depth cutoff. FLOW uses kfs/kfsed and
staggered open-face masks. Compare those masks and accepted net demand separately
in the first coupled case, especially around cropped cells and walls.

## Next milestone

Qualified compatible bank/toe composition on the shared CPU/GPU worker.

## Immediate next step

Capture and compare each accepted-demand/supply/redistribution ledger in a short
closed erodible-bank run, beginning with MPM at MF1 and collapse off.
'''
    (REPORT/'BANK_AUDIT.md').write_text(bank,encoding='utf-8')
    for case in (ROOT/'desktop/examples/repose223').iterdir():
        if case.is_dir():
            (case.parent/('Open '+case.name+'.cmd')).write_text('@echo off\ncall "%~dp0..\\..\\..\\Launch BRAIDS Desktop.cmd" "%~dp0'+case.name+'"\n',encoding='utf-8')
    status=ROOT/'validation/RIVER_MORPHODYNAMICS_STATUS.md'
    heading=f'''# Desktop223 — repose available across ordinary transport (2026-10-29)

Preview47 / project v35 installed. Repose collapse is independent of the bedload
capacity law: all 71 ordinary bedload/suspension configurations are qualified
on CPU/GPU with the explicit Exner inventory adapter and positive MF.
The original coupled Jiang–Cao bundle retains MF1. Both collapse choices and
sediment remain OFF by default. {tests} final checks pass plus five portable
examples with exact installed restart. No QGIS or Delft3D changes.

| Feature | Status |
| --- | --- |
| Nine ordinary bedload formulas / seven suspension formulas | Available; all existing compatibility/diameter rules retained |
| Ordinary equalization collapse | Available with all 71 ordinary configurations |
| Jiang–Cao repose criterion | Also available with all 71 ordinary configurations via finite Exner adapter |
| Original coupled Jiang–Cao model | Separate inventory coupling; MF1 unchanged |
| Bank/toe erosion and retained failure | Source/operator audit started; combined qualification is next |
| Mixtures/layers and experimental braided river | Deferred |

Report and figures: `sediment/river_morphodynamics/reports/desktop_repose223/REPORT.md`.
Examples: `desktop/examples/repose223`.
Next milestone: compatible bank/toe and collapse composition.
Immediate next step: MF1 bedload-only closed-bank ledger trace on CPU/GPU,
then suspension/MF contrasts. See BANK_AUDIT.md for source comparison.

---

'''
    status.write_text(heading+status.read_text(encoding='utf-8'),encoding='utf-8')
    readme=ROOT/'desktop/README.md';text=readme.read_text(encoding='utf-8').replace('BRAIDS 0.1.0-preview46 is','BRAIDS 0.1.0-preview47 is')
    start=text.index('## Compatible collapse in preview46');end=text.index('## Morphological factor',start)
    text=text[:start]+'''## Compatible collapse in preview47

In **Morpho evolution → General**, choose conservative neighbour equalization
or **Repose projection (Jiang–Cao criterion)**. Both work with all 71 supported
ordinary transport combinations and positive MF; collapse itself is not
multiplied by MF. Repose uses an explicit angle and preserves finite bed stock.
The original coupled Jiang–Cao transport bundle keeps its separate MF1 restriction.
See Resolved setup for the exact inventory/water assumptions. Sediment and
collapse default OFF. New project format v35 rejects older projects/checkpoints.

Open `desktop/examples/repose223` using its CMD shortcuts for short cropped-bank
comparisons, results and checkpoints. The old collapse222/MF221 projects are
historical evidence, not supported current inputs. Report and figures:
`validation/sediment/river_morphodynamics/reports/desktop_repose223/REPORT.md`.
Next: bank/toe compatibility; first trace accepted demand, supply, collapse and
commit in a short MF1 bedload-only case before adding suspension/MF contrasts.

'''+text[end:];readme.write_text(text,encoding='utf-8')
    archive=REPORT/'SOURCE_AND_REPORT.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for n in FILES:z.write(ROOT/n,n)
        for p in list((ROOT/'desktop').glob('*repose223*.py'))+list((ROOT/'tests').glob('*repose223*.py'))+[ROOT/'tests/test_bank_audit223.py']:
            z.write(p,str(p.relative_to(ROOT)))
        for p in REPORT.iterdir():
            if p.is_file() and p!=archive:z.write(p,'report/'+p.name)
    print('Desktop223 delivered; report, figures, source archive and next bank handoff recorded')
if __name__=='__main__':main()
