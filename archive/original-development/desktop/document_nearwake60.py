"""Document the completed audit and make the local projects understandable."""
from nearwake60 import *


def profiles(name):
    from braids_desktop import services,workspace
    records=read(HOME/'PROJECTS.json');r=records[name];root=Path(r['folder'])/'Project_data'
    output=Path(read(REPORT/name/'STARTED.json')['output']);progress=read(output/'PROGRESS.json')
    assert progress['status'] in ('complete','cancelled','failed'),'Wait for the case to finish before changing its editor state'
    oldhome=Path.home()/'Desktop/BRAIDS_Turbine_Validation59';old=read(oldhome/'PROJECTS.json')['nash_gpu']
    measurement=services.project_session(Path(old['folder'])/'Project_data')['desktop']['measurements']
    assert len(measurement['profiles'])==6
    before={str(p):sha(p) for p in [Path(r['project']),root/'braids_project.json',output/'PROJECT.json']}
    session=services.project_session(root);session.setdefault('desktop',{})['measurements']=measurement
    session['desktop'].update(model_step='Hydrokinetic turbines',result_step='Turbines')
    services.update_project_session(root,desktop=session['desktop'])
    state=workspace.read_editor(root)
    if state:
        state['session']['desktop']=session['desktop']
        workspace.save_editor(root,state['document'],state['ui'],state['session'],has_domain=state['has_domain'],
            has_terrain=state['has_terrain'],issue=state['issue'])
    assert all(sha(p)==h for p,h in before.items())
    save(REPORT/name/'PROFILE_DELIVERY.json',dict(status='passed',six_profiles=True,physical_project_bytes_unchanged=before))
    print('Added original comparison profiles without changing physical inputs:',name)


def project_notes():
    records=read(HOME/'PROJECTS.json')
    for name,r in records.items():
        if name.startswith('zero'):r['configuration']['ct']=0.
        spec=Path(r['folder'])/'AUDIT_SPEC.json';d=read(spec)
        d.update(runtime=r.get('runtime_sha256',read(REPORT/'LOCKED_SPEC.json')['runtime_sha256']),
            configuration=r['configuration'],
            purpose='Controlled rotor-only versus rotor-and-wake resolution; zero-thrust and bounded CPU controls are explicitly identified.',
            refinement='Native rotor footprint pins only.' if name=='rotor4_gpu' else 'Mesh-only axial breaklines through rotor and wake; these are not solid barriers.')
        save(spec,d)
    save(HOME/'PROJECTS.json',records)
    locked=read(REPORT/'LOCKED_SPEC.json');locked['projects']=records;save(REPORT/'LOCKED_SPEC.json',locked)
    (HOME/'README.md').write_text('''# Near-wake validation projects

Open **Open projects.cmd**, or the Desktop shortcut **Open BRAIDS Near Wake Validation.cmd**.

1. **wake8_gpu**: eight cells across the rotor and a refined downstream corridor; full 2000 s comparison.
2. **wake4_gpu**: four cells across the rotor and the same refined corridor; lower-cost full 2000 s comparison.
3. **wake4_cpu**: the identical four-cell mesh, run to 100 s for a paired backend check.

The projects contain the DEM, parameters, prepared mesh, results, checkpoints and history.
Select Results, then velocity. The saved Profiles include the centreline and transverse sections.
The Turbines tab exports histories to CSV. Comparisons and the detailed audit are in **comparisons**.
For the map, zoom around x=500 m, y=295 m; velocity limits 0.14–0.205 m/s reveal the wake.

The axial breaklines only retain mesh resolution. They do not impose walls or alter the bed.
The negative rotor-only control is stopped at 660 s. Zero-thrust controls are retained separately;
the original eight-cell control stopped on a cumulative floating-point accounting check, and
**zero8_corrected** is its independent cold rerun with the repaired diagnostic.

The hydraulic equations, turbine force, Ct and viscosity are unchanged. Refining the footprint
alone is insufficient: retain a resolved downstream corridor. These results qualify this case,
not a universal eight-cell accuracy guarantee for every river and turbine.

This is a passive-disc wake comparison: equivalent rotor/support Ct=0.90313 is an input;
Cp=0 and useful power=0 are intentional. It does not independently validate Cp or turbine blades.
The published 2D result and the experimental depth mean are separate references. The remaining
experimental near-wake error and reconstruction assumptions are stated in the report.

Preview59 changes cumulative water accounting only. Older preparations/checkpoints remain
readable as evidence but cannot be executed under a changed runtime. Use Prepare and run from
the saved initial condition; no automatic restart or checkpoint conversion is performed.
The current-runtime zero8_corrected checkpoint remains eligible for an explicit continuation.
''',encoding='utf-8')
    print('Project notes and control metadata updated')


def clear_stale_preparation_selections():
    from braids_desktop import services
    current=sha(RUNTIME/'RUNTIME.json');rows=[]
    for name,r in read(HOME/'PROJECTS.json').items():
        if r.get('runtime_sha256')==current:continue
        root=Path(r['folder'])/'Project_data';session=services.project_session(root)
        previous=session.get('prepared');output=session.get('output')
        services.update_project_session(root,prepared='')
        after=services.project_session(root);assert not after.get('prepared') and after.get('output')==output
        rows.append(dict(case=name,cleared_selection=previous,results_preserved=output,
            policy='Only deselect stale preparation; retain files/history. Explicit Run will prepare fresh with the installed runtime.'))
    save(REPORT/'PROJECT_EXECUTION_SELECTIONS.json',dict(status='passed',cases=rows))
    print('Deselected old-runtime preparations; all historical files and results retained')


def finish_report():
    m=read(REPORT/'METRICS.json');p=REPORT/'REPORT.md';s=p.read_text(encoding='utf-8')
    a=s.index('1 November 2026.');b=s.index('## What was audited')
    old=m['cases']['D/2'];new=m['cases']['D/8'];ref='published_2d_model'
    gain=100*(1-new['comparison'][ref]['near_2p5_to4D']['RMS_pp']/old['comparison'][ref]['near_2p5_to4D']['RMS_pp'])
    s=s[:a]+f'''1 November 2026. **Completed: near-wake resolution correction, full 2000 s GPU
comparisons, paired 100 s CPU check, and Desktop preview59 delivery.**

The original mesh conserved turbine thrust but spread it over too few cells.
The corrected projects refine both the rotor and downstream wake. Near-wake RMS
difference from Nash's published 2D result falls by **{gain:.1f}%**, while far-wake
agreement is retained. The apparent truncated peak was two almost equal central
cell values, not a velocity cap. No smoothing or fitted drag/mixing coefficient
is used to hide it.

**This does not close the experimental near-wake discrepancy.** The published
2D-model comparison passes the declared near/far 3 pp RMS aim; the experimental
near-wake aim remains failed. Numerical verification, published-model agreement
and experimental validation are reported separately below.

Open `C:\\Users\\STREEM\\Desktop\\Open BRAIDS Near Wake Validation.cmd`.
Projects and figures are in `C:\\Users\\STREEM\\Desktop\\BRAIDS_Turbine_NearWake60`.
The wake correction is a mesh configuration change; preview59 additionally fixes
an independently exposed cumulative water-budget arithmetic check. The physical
hydraulic/turbine implementation and native kernels are unchanged.

'''+s[b:]
    s=s.replace('| wake8_gpu | D/8 rotor and wake; 48,564 cells | Running to 2000 s |','| wake8_gpu | D/8 rotor and wake; 48,564 cells | Complete 2000 s |')
    s=s.replace('| zero4_gpu / zero8_gpu | Same refined meshes, Ct=0 | Running bounded 500 s controls |',
        '| zero4_gpu / zero8_corrected | Same refined meshes, Ct=0 | Complete 500 s controls |\n| zero8_gpu | Original cumulative-budget diagnostic | Failed at 250 s; retained |')
    s=s.replace(' Our impermeable\nwalls also omit its separate no-slip viscous wall condition.','')
    s=s.replace('not reproduced; the paper does not report the PML coefficient.',
        'not reproduced; the paper does not report the PML coefficient. Our impermeable\nwalls also omit its separate no-slip viscous wall condition.')
    s=s[:s.index('## Completed comparison') if '## Completed comparison' in s else s.index('## Next milestone')]
    s+='''## Completed comparison

Deficit is `1 - speed_with_turbine / speed_zero_thrust`. The wake averages use
1500–2000 s for every resolution. Each new mesh has its own 400–500 s zero-load
normalization; the historical control uses 1500–2000 s. pp means absolute
percentage points of velocity deficit, not relative percentage error.

| Reference | Rotor/wake resolution | Near 2.5–4D RMS (pp) | Far ≥4D RMS (pp) | Far maximum (pp) |
|---|---|---:|---:|---:|
'''
    for ref,label in [('published_2d_model','Published 2D model'),('experiment_depth_average','Experimental depth mean')]:
        for key,r in m['cases'].items():
            c=r['comparison'][ref]
            s+=f"| {label} | {key} | {c['near_2p5_to4D']['RMS_pp']:.3f} | {c['far_ge4D']['RMS_pp']:.3f} | {c['far_ge4D']['max_pp']:.3f} |\n"
    s+='''
![Completed near/far comparisons](NEAR_FAR_COMPARISON.png)

The original ≥4D interval is preserved, including exactly 4D. Its experimental
maximum-error screen is now passed; the old failed result remains in validation59.
The near experimental target is still too strong for this configuration. It cannot
be declared solved by moving the boundary of the far-wake interval or by comparing
only with another numerical model. Unreported PML settings, different loading and
tidal formulations, and three-dimensional rotor/wake physics remain limitations.

![Raw transverse cell values and published 2D profiles](LATERAL_RAW_CELLS.png)

The central unresolved span shrinks from 0.5D to 0.125D. Dots show actual CV means;
lines use linear interpolation. The map below uses a common scale that covers the
actual maximum deficit, so its upper colors do not clip the computed maximum.

![Unsmoothed maps on all three meshes](WAKE_MAPS.png)

| Resolution | 20D half-width (D) | Published half-width (D) | Peak deficit (%) | Published peak (%) |
|---|---:|---:|---:|---:|
'''
    for key,r in m['cases'].items():
        a=r['lateral']['20.0']
        def fmt(x):return f'{x:.3f}' if x is not None else 'outside sampled range'
        s+=f"| {key} | {fmt(a['prediction_halfwidth_D'])} | {fmt(a['reference_halfwidth_D'])} | {a['prediction_peak_percent']:.3f} | {a['reference_peak_percent']:.3f} |\n"
    s+='''
The load *law and coefficient* are identical; coupled force need not be identical
because the resolved local velocity and reconstructed approach speed change:

| Resolution | Mean thrust (N) | Reconstructed approach (m/s) | Ct using nominal inlet speed |
|---|---:|---:|---:|
'''
    for key,r in m['cases'].items():
        s+=f"| {key} | {r['mean_thrust_N']:.2f} | {r['mean_reconstructed_approach_m_s']:.6f} | {r['Ct_against_nominal_inlet']:.6f} |\n"
    s+='''
The uniform-state force integral is exact at every resolution, but that does not
prove grid-independent approach reconstruction on an evolved nonuniform wake.
Thrust decreases about 4.6% from D/2 to D/8, and about 1.1% from D/4 to D/8.
The stronger near deficit therefore does not result from increasing the prescribed
coefficient or applied force. This remaining loading sensitivity is retained;
recovering Ct from the reconstructed speed is algebraic, not independent validation.

Width remains a separate limitation: a similar centreline deficit does not prove
an identical transverse wake. Eight cells are a tested choice for this bundle/case,
not a universal mesh-independent requirement for every turbine or river.

### Time averaging and controls

The stringent predeclared 0.02 pp change screen for the shortened zero-load controls
is retained in `CONTROL_CHECK.json`, including failures. They are not labelled
perfectly stationary. The following sensitivity to the completed historical steady
control bounds their effect on the reported centreline comparison:

| Resolution | Change using historical control (pp) | Wake half-window change (pp) |
|---|---:|---:|
'''
    for key,r in m['cases'].items():
        s+=f"| {key} | {r['maximum_normalization_sensitivity_pp']:.4f} | {r['maximum_centreline_halfwindow_change_pp']:.4f} |\n"
    s+='''
These sensitivity checks are not a replacement normalization chosen to improve fit.
Small persistent oscillations remain (about 0.15 pp difference between the wake
half-window means). They are much smaller than the near-wake discrepancy and do
not explain it away as a startup transient. No unlimited-time stationarity claim
is made from this finite calculation.
Raw values, reference digitization identities, signed bias and full error metrics
are in `METRICS.json` and `COMPARISON.csv`. Published stroke extraction is distinct
from experimental uncertainty; raw laboratory records remain unavailable.

![Wake development and the retained averaging window](TEMPORAL_DEFICIT.png)

## Cumulative water-accounting defect and repair

The original D/8 zero-thrust control stopped at 250 s. Its water residual was
3.331e-7 m³ in an 18,000,000 m³ domain: all local conservative trial checks passed,
but the worker compared 50 accepted updates with a single-reduction allowance.
Long-double replay confirms the residual is already accumulated in state/ledger
arithmetic, not merely the final GPU reduction. It is unrelated to the wake shape.

Preview59 uses a step-count/inventory-scaled cumulative floating-point allowance,
`gamma_(64*(accepted_updates+1)) * inventory_or_boundary_work`, retaining the 1e-9 m³
floor. This is a conservative engineering roundoff allowance for separately checked
updates, not a proof for arbitrary cumulative leakage or a larger physical mass-error
tolerance. The strict per-trial guard, wall-transfer and sediment-stock checks are
unchanged. Significant injected water loss and any nonzero wall transfer still fail.

The corrected zero-load case runs independently from the initial state. Its saved
hydraulic fields are **exactly identical** to the original run through the 250 s
failure, and it completes 500 s. No changed-runtime checkpoint was reinterpreted.

Qualification: 66 focused tests pass; four CPU/GPU hydraulic/MPM controls regrid,
restart exactly, and retain byte-identical accepted fields against preview58.
The matched refined-wake CPU/GPU comparison at 100 s differs by only 2.565e-10 m
in depth and 2.242e-10 m/s in velocity. That is not a full 2000 s CPU qualification.
All physical laws, native payloads, f-wave reconstruction, CFL and timestep control
are unchanged. Failed trials and prior validation58/59 manifests remain intact.

### Desktop delivery and scope

The current-runtime project opens, the UI smoke passes, saved comparison profiles
reopen and turbine CSV exports reproduce the stored data exactly. The automated
Prepare-to-Run harness exposed a nonfatal asynchronous mesh-preview race
(`Another operation is in progress`) before launch. Solver runs and later project
inspection succeed; this trace is retained rather than described as error-free
interactive preparation. No runtime/UI concurrency fix is claimed in this milestone.

The historical wake outputs remain viewable in preview59. Their old preparations
and checkpoints are rejected for execution under the changed runtime: Prepare
explicitly for a fresh run. The delivered projects deselect their old preparation,
so pressing Run also triggers fresh preparation rather than attempting the stale one.
The corrected zero-load preparation/checkpoint has the
current identity. QGIS is untouched. No simulations from this audit remain active.

## Next milestone

Qualify the resolved turbine source with one bedload closure in a short straight
channel: paired turbine-off/on runs, identical initial fields, closed water/sediment
budgets, CPU/GPU agreement and figures of erosion/deposition. Keep experimental
near-wake and three-dimensional local-scour limits explicit; do not call the first
coupling test experimental scour validation.

## Immediate next step

Freeze a single-grain MPM channel with the qualified rotor-and-wake refinement.
Check the turbine/source → velocity/shear → bedload → Exner chain on shared states
before a bounded mobile-bed run. Preserve the present hydrodynamic evidence.
'''
    p.write_text(s,encoding='utf-8');print(p)


def handoff():
    m=read(REPORT/'METRICS.json');r=m['cases']['D/8']['comparison']
    text=f'''## Near-wake correction and preview59

Open **Open BRAIDS Near Wake Validation.cmd** on the Desktop. The projects in
`C:\\Users\\STREEM\\Desktop\\BRAIDS_Turbine_NearWake60` retain the original turbine
law and loading, with explicit rotor-and-wake refinement. D/4 is the practical
starting point; D/8 is the finer confirmation. Refining the rotor alone failed
to improve the near wake and is preserved as a negative control.

The D/8 published-model RMS deficit differences are
{r['published_2d_model']['near_2p5_to4D']['RMS_pp']:.3f} pp at 2.5–4D and
{r['published_2d_model']['far_ge4D']['RMS_pp']:.3f} pp at ≥4D. Experimental near-wake
RMS remains {r['experiment_depth_average']['near_2p5_to4D']['RMS_pp']:.3f} pp: this
target is not closed. The apparent capped peak came from two unresolved central
cells, not a velocity limit. Raw profiles, maps and time histories are retained in
[the detailed audit](../validation/hydrokinetic_turbines/reports/nearwake60/REPORT.md).

Preview59 repairs only the cumulative floating-point water-budget diagnostic.
The per-step conservation guard and native physics remain unchanged. Sixty-six
focused checks and four exact-field CPU/GPU hydro/MPM AMR/restart checks pass.
Old results remain viewable; old preparations/checkpoints cannot execute under
the new runtime. Prepare explicitly. Project format v39 is unchanged.

'''
    p=ROOT/'desktop/README.md';s=p.read_text(encoding='utf-8');assert '## Near-wake correction and preview59' not in s
    s=s.replace('BRAIDS 0.1.0-preview58 is','BRAIDS 0.1.0-preview59 is',1)
    i=s.index('## Direct experimental wake comparison');s=s[:i]+text+s[i:];p.write_text(s,encoding='utf-8')
    p=ROOT/'validation/hydrokinetic_turbines/PLAN.md';s=p.read_text(encoding='utf-8');i=s.rindex('## Next milestone')
    s=s[:i]+text.replace('../validation/hydrokinetic_turbines/','')+'''## Next milestone

Qualify resolved-turbine coupling to single-grain MPM bedload in a bounded straight
channel. Compare turbine-off/on, paired CPU/GPU fields and water/sediment budgets.
Retain the experimental near-wake limitation; do not equate coupling verification
with experimentally validated rotor-scale scour. Yaw and rotor-generated turbulence
remain separate physical qualification work.

## Immediate next step

Freeze identical initial water/bed/stocks and the resolved mesh; audit source,
velocity/shear, transport and Exner on shared states before the short coupled run.
''';p.write_text(s,encoding='utf-8')
    p=ROOT/'validation/RIVER_MORPHODYNAMICS_STATUS.md';s=p.read_text(encoding='utf-8');i=s.find('\n')+1
    s=s[:i]+'''
## Turbine hydraulic follow-up — 1 November 2026

Near-wake60 qualifies explicit rotor-and-wake refinement against the published
2D wake, while retaining the remaining experimental near-wake error. Preview59
changes cumulative water-budget arithmetic only; four CPU/GPU hydraulic/MPM
AMR/restart controls retain exact fields. Turbine–bedload coupling is the next
separate test; this does not qualify turbine-induced sediment scour or change
any accepted morphological law. See
`hydrokinetic_turbines/reports/nearwake60/REPORT.md` for figures and the handoff.

'''+s[i:];p.write_text(s,encoding='utf-8')
    print('Updated Desktop notes, turbine plan and maintained feature status')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['project_notes','finish_report','handoff','profiles','clear_stale_preparation_selections'],nargs='?',default='project_notes');p.add_argument('--case');a=p.parse_args()
    if a.action=='profiles':profiles(a.case)
    else:globals()[a.action]()
