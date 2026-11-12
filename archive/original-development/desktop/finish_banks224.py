"""Publish measured milestone224 evidence and the next bounded handoff."""
import json,zipfile
from stage_banks224 import ROOT,OUT,TARGET,BASE,REPORT,FILES,sha


def main():
    m=json.loads((OUT/'METRICS.json').read_text());e=json.loads((OUT/'EXAMPLE_COMPARISON.json').read_text())
    receipt=json.loads((REPORT/'INSTALL.json').read_text());installed=json.loads((OUT/'INSTALLED_LAUNCH.json').read_text())
    assert m['runtime']==sha(BASE/'RUNTIME.json')==installed['runtime']
    tests=sum(x['tests'] for x in receipt['tests'].values())
    table='\n'.join(f"| {name} | {v['min_change_mm']:.6g} | {v['max_change_mm']:.6g} | {v['bank_kg']:.6g} | {v['collapse_kg']:.6g} | {v['block_count']} |" for name,v in e['cases'].items())
    text=f'''# Desktop224 — compatible bank supply, toe erosion and collapse

Completed and installed: **BRAIDS Desktop 0.1.0-preview48**, project **v36**.
One maintained explicit SSPRK22 augmented-f-wave CPU/GPU route; backend remains
the user's explicit choice. Sediment, bank erosion and collapse default OFF.
No QGIS or Delft3D source/runtime changes. The previous preview47 delta is
retained at `desktop/qualification/banks224/preview47_backup`.

## What changed

Morpho evolution → General now exposes **Bank erosion and toe supply** independently
from **Bank and slope collapse**. Three bank choices are available: FLOW-style
local dry-bank sharing, surveyed toe supply, and surveyed toe supply with retained
undercut/block release. Either existing collapse choice may follow. Resolved setup
explains ownership, clocks and limitations. No capacity law or native numerical
kernel was changed; this closes the combined admission/qualification gap.

The actual execution order is accepted bedload divergence + bounded suspended
exchange → net bed demand → bank/toe supply → optional retained block release →
one selected slope redistribution → one column commit. Finite stock is shared
by all stages. Bank and collapse material enters the bed, without a duplicate
suspended source or added water volume. MF is already present in accepted demand;
there is no second MF multiplier in bank supply, undercut history or collapse.
Retained block histories, event counts and transferred class mass are now saved
with results and verified on restart.

The retained-block AMR screen found a missing CPU package binding for the
existing integer `bank_counts` history-remap routine. The release now includes
that routine, compiled from its unchanged source. Existing libraries and loader
bindings are byte-identical; the affected AMR group and remaining checks were
run on the corrected package. Unaffected test receipts retain their original
runtime identity in `QUALIFICATION_PARTS.json`; they are not relabelled as reruns.

## Formula compatibility

All nine ordinary bedload formulas remain available: MPM, Wong–Parker 2006,
generalized Ashida–Michiue, Wang–Fredsøe, Smart–Jäggi, Ancey–Recking 2023,
Recking 2010, Camenen–Larson 2005 and Van Rijn 1993.

| Suspended selection | Bedload OFF | Any of the nine ordinary bedload formulas |
| --- | --- | --- |
| OFF | Bank supply needs a transport process; independent collapse remains available | Yes |
| Supplied suspension / settling only | Yes; settling alone cannot request erosional bank supply | Yes |
| García–Parker 1991 | Yes | Yes |
| de Leeuw 2020 | Yes | Yes |
| Van Rijn 1984 | Yes | Yes |
| Zyserman–Fredsøe 1994 | Yes | Yes |
| Camenen–Larson 2008, current only | Yes | Yes |
| Van Rijn 1993 FLOW 2DH | Yes | Matching Van Rijn 1993 bedload only |

These are the existing **71 ordinary transport combinations**. Each can select
one bank owner and collapse OFF, ordinary equalization, or the Jiang–Cao-criterion
Exner repose projection. The bank feature does not relax diameter, shear,
roughness or paired-formula requirements. Material is one bulk sediment, finite
fully mixed bed, dry-bulk-density packing; no sorting or transported mixtures.
Sediment start must be zero. Positive MF remains allowed; this milestone tests
MF1 and MF25, not accuracy for arbitrary large acceleration factors.

The **full coupled Jiang–Cao transport bundle** remains separate: MF1 and its
own repose coupling. Ordinary bank supply is rejected with an explicit integration
message because the mobile-bed/water/density inventory needs a different coupling.
This is a support gap, not a claim that bank erosion is physically impossible
with that model. No hidden formula or backend substitution occurs.

## Source audit and numerical checks

[SOURCE_AUDIT.md](SOURCE_AUDIT.md) traces the read-only FLOW `bott3d.f90` chain.
Its dry-bank operator uses net total erosion, minimum participating area and
donor composition. Independent CPU/GPU oracles check that arithmetic with common
wet/dry flags, unequal areas, rotation, finite donors, no deposition-triggered
withdrawal and no transfer through exterior walls. BRAIDS uses actual polygon
wet activity and a finite-stock limiter; identical FLOW masks/fixfac behavior
are not assumed. Surveyed toe supply and retained block release are distinct
BRAIDS extensions, not renamed FLOW bank physics.

**{tests} distinct qualification/regression tests passed**, plus eight portable Desktop examples and exact
installed-worker restarts. Coverage includes:

- All 71 ordinary transport combinations executed with bank sharing on CPU/GPU.
- Each of nine bedload and seven suspension formulas executed with both surveyed
  owners on CPU/GPU; coupled MPM + García–Parker exercises both collapse options
  and MF1/MF25. This is component/interaction coverage, not a full Cartesian run
  of every bank × formula × collapse × mesh × MF combination.
- Closed and open boundaries, cropped walls, AMR with protected survey corridors,
  finite/zero stock, actual block release, rollback after rejection/cancellation,
  exact checkpoint continuation, current UI round-trip and old-format rejection.
- Maximum CPU/GPU bed difference in the 71-case matrix:
  **{max(p['max_bed_mm'] for p in m['paired']):.6g} mm**.
- Timestep-halving maximum: **{max(p['max_bed_mm'] for p in m['temporal']):.6g} mm**.
- Maximum CPU/GPU difference across cropped-wall and AMR controls:
  **{max(p['max_bed_mm'] for p in m['adaptive_pairs']):.6g} mm**;
  retained block event counts match exactly on both meshes.
- Maximum water residual **{m['max_water_residual_m3']:.6g} m³**;
  sediment residual **{m['max_sediment_residual_kg']:.6g} kg**;
  local transfer residual **{m['max_local_transfer_residual_kg']:.6g} kg**.
  The established MF virtual exchange ledger is included in sediment closure.

![Numerical qualification](QUALIFICATION.png)
![Actual retained releases in the forced-trigger control](RETAINED_BLOCK_TEST.png)

Earlier failed/aborted screens are retained under `desktop/qualification/banks224`.
The first AMR survey protected the entire dry bank, correctly preventing regrids;
the corrected test leaves an unprotected dry region. A later large in-process
test harness aborted during setup; final qualification uses bounded fresh
processes with complete logs, XML and successful process-exit receipts. A stale
four-formula UI assertion was updated to the current ten explicit method IDs.
The pre-fix CPU retained-block AMR failure is also retained, with the successful
corrected-package rerun recorded separately.
None of those failures is counted as a pass.

## Short matched bank comparison

The eight projects under `desktop/examples/banks224` share exactly identical
initial water, bed, sediment stocks and cropped physical cells. MPM +
García–Parker, finite stock 0.2 m, a 0.16 m raised bank and 0.15 m initial stage;
0.4 s hydraulic time, MF25 except the named MF1 contrast. This intentionally
short closed-bank demonstration is **not experimental validation** or a new
FLOW trajectory. The retained-block demo threshold is 0.0001 m; numerical tests
also use 1e-8 m to stress event/history ownership, not as a calibrated bank law.

| Case | Minimum bed change (mm) | Maximum bed change (mm) | Gross bank supply (kg) | Gross collapse (kg) | Block events |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

![Bank methods on actual cells](BANK_METHODS.png)
![Bank contribution relative to OFF](BANK_EFFECT.png)
![Collapse choices](COLLAPSE_COMPOSITION.png)
![Raw section and transfer histories](BANK_HISTORY.png)

The bank methods intentionally differ. Local sharing selects eligible higher
dry neighbours dynamically; surveyed supply acts only on the selected internal
toe faces and applies submerged-height/geometric bounds. Retained blocks add
finite releases only after accumulated actual toe supply crosses the threshold.
Equalization and repose also have different targets: one levels triggered pairs;
the other projects slopes toward the specified angle. Their different bed
patterns are not an unexplained CPU/GPU discrepancy. The matched CPU/GPU
repose example differs by **{e['cpu_gpu_max_bed_mm']:.6g} mm**.
The 0.0001 m block threshold is not reached in these 0.4 s examples, so toe-only
and retained-block beds are identical. The separate forced-trigger control above
does activate block release and checks CPU/GPU agreement, conservation and history.

Surveyed toe corridors remain mapped and finest; they do **not** automatically
migrate with an unconstrained bank line. Local sharing can follow changing
wet/dry neighbours. Neither collapse law supplies cohesive/geotechnical bank
strength. Short conservation/backend tests do not establish long-term river
migration rates. The following physical benchmark is the next acceptance step.

Runtime SHA256: `{m['runtime']}`. Existing user projects were not modified.
Project v36 rejects earlier identities; supplied examples are current and include
inputs, results, history and checkpoints. Open their CMD shortcuts to inspect them.

## Next milestone

Run a short matched open-channel widening comparison against locked, unmodified
FLOW 2DH. Compare dry-bank-sharing ON/OFF trends, wet/bank outlines, raw bed
sections and budgets; keep BRAIDS surveyed toe/block extensions separately named.

## Immediate next step

Freeze identical initial bed, water and finite sediment stocks for an MPM MF1
case. Compare a few accepted transfers first, then a bounded widening run and
an MF25 contrast. No long braided-river simulation yet.
'''
    (REPORT/'REPORT.md').write_text(text,encoding='utf8')
    (REPORT/'INSTALLED_LAUNCH.json').write_text(json.dumps(installed,indent=2))
    for case in (ROOT/'desktop/examples/banks224').iterdir():
        if case.is_dir():
            (case.parent/('Open '+case.name+'.cmd')).write_text('@echo off\ncall "%~dp0..\\..\\..\\Launch BRAIDS Desktop.cmd" "%~dp0'+case.name+'"\n',encoding='utf8')
    status=ROOT/'validation/RIVER_MORPHODYNAMICS_STATUS.md'
    heading=f'''# Desktop224 — ordinary bank/toe composition delivered

Preview48 / project v36 installed. All 71 ordinary transport configurations
can select FLOW-style sharing or surveyed toe supply / retained blocks, and
either existing collapse choice. One finite bulk bed; MF1/MF25 qualification,
all existing measured-diameter and formula-pair requirements preserved.
Full coupled Jiang–Cao remains MF1 with its own repose, without ordinary bank
supply. No QGIS or Delft3D modifications. {tests} final tests pass plus eight
portable examples with exact installed restart.

| Feature | Status |
| --- | --- |
| Ordinary bedload and independent suspension | Existing 71 combinations retained |
| Both collapse choices | Compatible with all ordinary transport combinations |
| Local dry-bank sharing | Delivered with bedload/suspension, positive MF |
| Surveyed toe supply + optional retained blocks | Delivered; fixed mapped toe corridor, finite stock |
| Experimental bank-retreat accuracy | Not established by these short component tests |
| Mixtures/layers and braided experiment | Deferred |

Report/figures: `sediment/river_morphodynamics/reports/desktop_banks224/REPORT.md`.
Examples: `desktop/examples/banks224`.
Next milestone: short matched FLOW 2DH channel widening, keeping separate bank owners.
Immediate next step: identical MPM MF1 initial fields, bank OFF/ON transfer traces,
then bounded evolution and MF25 contrast.

---

'''
    status.write_text(heading+status.read_text(encoding='utf8'),encoding='utf8')
    p=ROOT/'desktop/README.md';s=p.read_text(encoding='utf8').replace('BRAIDS 0.1.0-preview47 is','BRAIDS 0.1.0-preview48 is')
    start=s.index('## Compatible collapse');end=s.index('## Morphological factor',start)
    s=s[:start]+'''## Bank erosion and collapse in preview48

Morpho evolution → General exposes three bank methods: local FLOW-style dry-bank
sharing, surveyed toe supply, and surveyed toe supply with retained block release.
Either conservative equalization or Jiang–Cao-criterion repose may follow.
All 71 ordinary bedload/suspension combinations retain their measured-diameter
and paired-formula restrictions. Bank supply uses accepted net erosion once;
MF is not multiplied again. One finite bulk bed, sediment start=0. The mapped
toe corridor does not automatically migrate. Resolved setup explains these limits.
The full coupled Jiang–Cao transport bundle still uses MF1 and its own repose;
ordinary Exner bank supply is not admitted to that separate inventory coupling.
Sediment, banks and collapse are OFF by default. Project v36 rejects old formats.

Open the CMD shortcuts in `desktop/examples/banks224` for matched short examples,
results and checkpoints. Earlier examples remain historical evidence. Report:
`validation/sediment/river_morphodynamics/reports/desktop_banks224/REPORT.md`.
Next: a short matched FLOW 2DH channel-widening comparison, first at MF1.

'''+s[end:]
    s=s.replace('## Morphological factor and suspended transport in preview45',
                '## Morphological factor and suspended transport')
    s=s.replace('create a **new v34 project**', 'create a **new v36 project**')
    s=s.replace('General, choose a **positive morphological factor and sediment start 0**, and optionally enable conservative collapse in Morpho evolution. Leave bank erosion,\nsorting, bedform and slope extras off.',
                'General, choose a **positive morphological factor and sediment start 0**. Bank supply and collapse are optional in Morpho evolution; review their finite-stock requirements in Resolved setup. Leave sorting, bedform and slope extras off.')
    s=s.replace('Ordinary collapse is now qualified with positive MF; bank erosion with acceleration remains blocked.',
                'Ordinary bank supply and both collapse choices are qualified with positive MF within the one-material finite-bed scope described above.')
    s=s.replace('not inputs for preview46. Use the new `desktop/examples/collapse222` projects.',
                'not current inputs. Use the new `desktop/examples/banks224` projects.')
    s=s.replace('Bank/toe erosion is the next morphodynamic compatibility milestone; conservative collapse is available now.',
                'Bank/toe erosion and both collapse choices are available within the qualified scope above; physical channel-widening comparison is next.')
    s=s.replace('The sediment editor is format 13; project/model contracts are v34/v3.',
                'The sediment editor is format 13; project/model contracts are v36/v3.')
    s=s.replace('The usual CMD shortcut opens preview46.', 'The usual CMD shortcut opens preview48.')
    s=s.replace('Create a new v31 project:', 'Create a new v36 project:')
    s=s.replace('for their respective formula families. No sediment or collapse is activated',
                'with the ordinary-formula extension described above. No sediment or collapse is activated')
    a=s.rindex('## Next milestone')
    s=s[:a]+'''## Next milestone

Run a short matched open-channel widening comparison against locked unmodified
FLOW 2DH. Compare bank sharing OFF/ON, raw bed sections, bank/wet outlines and
budgets; keep surveyed toe and retained-block extensions separately named.

## Immediate next step

Freeze identical initial bed, water and finite sediment stocks for MPM MF1;
compare a few accepted transfers before a bounded widening run and MF25 contrast.
For local testing now, use `desktop/examples/banks224` and its CMD launchers.
A clean-Windows installer remains separate work.
'''
    p.write_text(s,encoding='utf8')
    with zipfile.ZipFile(REPORT/'SOURCE_AND_REPORT.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n in FILES:z.write(ROOT/n,n)
        for n in ('braids_core/sediment/river_bank_remap.h','braids_core/sediment/river_bank_remap.py'):
            z.write(ROOT/n,n)
        for n in ('RUNTIME.json','PRECOMPILED.json'):
            z.write(BASE/n,'runtime/'+n)
        for p in list((ROOT/'desktop').glob('*banks224*.py'))+list((ROOT/'tests').glob('*banks224*.py'))+[ROOT/'tests/test_bank_operators224.py',ROOT/'braids_desktop/tests/test_forms.py',ROOT/'desktop/bind_bank_counts224.py']:
            z.write(p,str(p.relative_to(ROOT)))
        for p in REPORT.iterdir():
            if p.is_file() and p.suffix!='.zip':z.write(p,'report/'+p.name)
    print('Desktop224 report and next handoff published',flush=True)

if __name__=='__main__':main()
