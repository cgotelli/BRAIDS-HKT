"""Publish only measured qualification and installed-package evidence."""
import json,shutil,xml.etree.ElementTree as ET
from stage_morfac221 import ROOT,OUT,REPORT,BASE,TARGET,sha,save

def main():
    m=json.loads((OUT/'METRICS.json').read_text())
    install=json.loads((REPORT/'INSTALL.json').read_text())
    launch=json.loads((OUT/'INSTALLED_LAUNCH.json').read_text())
    assert m['runtime']==install['runtime']==launch['runtime']==sha(BASE/'RUNTIME.json')==sha(TARGET/'RUNTIME.json')
    rows=json.loads((OUT/'EXAMPLE_COMPARISON.json').read_text())
    backend=max(r['max_abs']['z'] for r in m['paired_mf25'])*1000
    temporal=max(r['errors']['z'] for r in m['time_step_halving'])*1000
    counts={k:int(ET.parse(OUT/(k+'.xml')).getroot().find('testsuite').attrib['tests']) for k in ('qualification5','regression5')}
    table='\n'.join(f"| {r['backend'].upper()} | {r['mf']} | {r['bed_kg']:.9g} | {r['suspended_kg']:.9g} | {r['acceleration_kg']:.9g} |" for r in rows)
    text=f'''# Desktop221 — morphological acceleration

Preview45 is installed in BRAIDS Desktop. Project v33, explicit SSPRK22
augmented-f-wave CPU/GPU, sediment off and MF1 by default. No QGIS changes.
Runtime manifest: `{m['runtime']}`.

## Delivered scope

All seven ordinary suspension choices accept positive MF, independently or
with their existing compatible bedload choices. Tested source factors:
0.5, 1, 2, 5, 10 and 25. Subnormal-stock arithmetic was also checked at 0.001
and 1,000,000; those arithmetic checks do not qualify extreme coupled MF.
Bedload-only acceleration is retained. No formula or grain descriptor is
silently changed. The public formula compatibility matrix is unchanged:
six suspension choices pair with the nine ordinary bedload formulas; the
FLOW Van Rijn 1993 suspension pairs only with Van Rijn 1993 bedload, or runs
alone. All seven can run suspended-only.

Jiang–Cao remains MF1: its mobile sediment, water/pore-water and density feedback
need a distinct coupled acceleration formulation. Suspension still requires
start time zero, a single bulk material and a finite mixed bed. Bank/collapse,
sorting and other previously unqualified suspension combinations stay blocked.
New morphology was deliberately not added in this milestone.

## What MF does

For physical suspended-mass change dm (positive for entrainment), wet-bed
exchange is -MF × dm. Suspended advection and source/sink relaxation use physical
dt; bedload divergence uses MF × dt. Entrainment is capped by the available bed
stock remaining after bedload, divided by MF. Dry stranding transfers its
remaining physical inventory once, without MF. This is a named BRAIDS
`morphology.bed_clock_morfac_v1` numerical approximation, not an altered
entrainment or settling formula.

The additional inventory is recorded as an acceleration source:
deltaA = deltaBed + deltaSuspended. The checked budget is
Bed + Suspended = initial + bed-boundary contribution + physical
suspended-boundary contribution + acceleration ledger + basal source.
The bed-boundary contribution already includes MF. Diagnostics expose
`morphological_factor` and `suspended_acceleration_kg`; result frames and
checkpoints retain the spatial acceleration ledger. It is incorrect to call
the extra bed mass physical sediment supplied during the shorter hydraulic run.

MF approximates slowly changing morphology under repeated or near-steady
forcing. It is not exact time compression of an unsteady flood or finite pulse.
A larger factor can change the solution and trigger a smaller accepted step.
The existing 5% relative bed-increment rejection remains in force. No factor
is claimed to be universally accurate or to give an MF-fold speedup.

## Source audit and correction

Read-only Delft3D4 FLOW source audit: `erosed.f90` constructs dtmor=dt×morfac;
`bott3d.f90` applies it to bedload divergence plus deposition minus entrainment;
`red_soursin.f90` accounts for morfac when limiting source/sink updates.
Their SHA256 identities are in METRICS.json. Sources and reference engines
were not modified or rebuilt. This is a source-level comparison, not a new
coupled Delft3D execution or experimental validation.

BRAIDS already had a physical suspension clock and acceleration ledger. The
new independent entrainment closures had MF1 admission guards. They now use
one stock/MF reservation and wet-bed scaling adapter, shared by CPU and GPU
and also used by the FLOW exchange route. Existing native binaries and
published equilibrium, shear, settling and profile laws are unchanged.

An initial finite-stock trial failed after erosion reached about 1e-321 kg/m²:
division/multiplication by MF rounded upward, while a relative floating-point
reserve became ineffective. The fix rounds only subnormal quotients toward
zero by one representable value. It introduces no physical cutoff, mass
clipping or relaxed stock tolerance. Failed receipts are preserved under
`desktop/qualification/morfac221`.

## Verification

- **{counts['qualification5']} new MF checks pass** on the final packaged runtime.
- **{counts['regression5']} retained regression checks pass** on that same runtime.
- Fourteen MF1 CPU/GPU runs have **byte-identical output arrays** to installed
  preview44, including water, bed, inventories and boundary ledgers.
- Fifty-six MF25 worker runs cover seven formulas, both backends, open flow,
  cropped oblique walls, wet/dry remeshing and finite-bed exhaustion. Their
  exact restart and injected rejection/cancellation rollback checks pass.
- Largest paired bed difference: **{backend:.6g} mm** across those cases.
- Fourteen independent MF25 timestep-halving checks: largest bed difference
  **{temporal:.6g} mm** (0.002 versus 0.001 s).
- Maximum tested sediment-budget residual **{m['max_budget_kg']:.6g} kg**;
  water residual **{m['max_water_residual_m3']:.6g} m³**. These include the explicit
  MF acceleration source. Closed walls have zero water and sediment flux.
- All 62 existing suspended formula configurations accept the tested positive
  factors in the real editor/resolver. Invalid MF, missing sizes, old project
  formats, incompatible formulas and changed-MF restart are rejected.
- A real Qt render at 390 px checks General and Resolved setup. Four actual
  Desktop child-process prepare/run/restart demonstrations pass; their
  candidate checkpoints restart exactly with the installed package.

![Numerical checks](QUALIFICATION.png)

## Short demonstration

The self-contained folders in `desktop/examples/morfac221` use MPM +
García–Parker, measured D50=0.2 mm, D90=0.4 mm, Ds=0.1 mm, initially zero
erodible stock and zero suspended concentration, with inlet supply 0.4 kg/m³.
Each runs 4 hydraulic seconds and saves a checkpoint at 2 seconds. The nominal
bed clocks are 4, 20 and 100 seconds for MF1, MF5 and MF25. This is a synthetic
numerical demonstration, not an experiment or a 100-second MF1 reproduction.

| Backend | MF | Bed mass (kg) | Suspended mass (kg) | MF acceleration ledger (kg) |
| --- | ---: | ---: | ---: | ---: |
{table}

![MF evolution](MF_EVOLUTION.png)

Bed accumulation grows approximately with MF because the flow changes little
over this short deposition example. Suspended inventory remains close across
factors because its transport and settling clock is unchanged. Small departures
from linear scaling reflect the updated bed feeding back into flow and source
evaluation. CPU/GPU overlap supports implementation agreement, not universal
accuracy of accelerated morphology.

Open `cpu_MF25` or `gpu_MF25` through Desktop's project-folder command. The
folder contains input data, parameters, prepared geometry, runs, logs and
checkpoints. Preview44 projects/runs remain preserved but are not silently
reinterpreted under v33. Existing user projects were not modified.

## Next milestone

Qualify the existing morphodynamic features with an explicit compatibility
matrix: ordinary conservative slope/collapse redistribution, the named
Jiang–Cao repose projection within its own bundle, then bank/toe erosion and
lateral migration. Resolve physical versus morphological clocks, finite stock,
and suspension interactions before exposing any new combination. Avoid
double-counting collapse or sediment release.

## Immediate next step

Trace the two existing collapse owners and their bed/water/mobile-sediment
transfers. Define a small closed-bank test with fixed total sediment inventory,
then test each admissible CPU/GPU combination and MF sensitivity, with bank
cross-sections and mass-budget figures, before changing the Desktop admission
rules. Jiang–Cao acceleration remains a separate unresolved item.
'''
    (REPORT/'REPORT.md').write_text(text,encoding='utf-8')
    shutil.copy2(OUT/'INSTALLED_LAUNCH.json',REPORT/'INSTALLED_LAUNCH.json')
    status=ROOT/'validation/RIVER_MORPHODYNAMICS_STATUS.md'
    existing=status.read_text(encoding='utf-8')
    entry=f'''# Desktop221 — MF acceleration delivered (2026-10-29)

Preview45 / project v33 installed. Seven ordinary suspension choices now
support positive MF in their existing combinations; MF1 remains default.
Physical suspended clock, finite stock/MF, unscaled dry stranding and an
explicit acceleration ledger are shared by CPU/GPU. MF1 arrays unchanged;
{counts['qualification5']} MF checks and {counts['regression5']} regression checks pass.
Jiang–Cao remains MF1; bank/collapse acceleration and suspension morphology
remain explicitly gated. No QGIS or Delft3D changes.

Report and figures: `sediment/river_morphodynamics/reports/desktop_morfac221/REPORT.md`.
Openable examples: `desktop/examples/morfac221` (MF1, MF5, MF25; CPU/GPU).

Next milestone: compatible collapse and bank-migration integration.
Immediate next step: audit the two existing collapse owners and qualify closed
finite-stock bank tests, including clocks and suspended-material transfers.

---

'''
    assert not existing.startswith('# Desktop221')
    status.write_text(entry+existing,encoding='utf-8')
    save(OUT/'COMPLETE.json',dict(runtime=m['runtime'],version='0.1.0-preview45',tests=counts,
        report=str(REPORT/'REPORT.md'),installed_launch_sha256=sha(OUT/'INSTALLED_LAUNCH.json')))
    print('Published MF report and updated feature status',flush=True)
if __name__=='__main__':main()
