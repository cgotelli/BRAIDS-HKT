"""Seal the bounded power audit, including the failed physical acceptance."""
import csv, hashlib, json, shutil
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/hydrokinetic_turbines/reports/power_array62'
HOME=Path.home()/'Desktop/BRAIDS_Turbine_Power62'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

summary=read(OUT/'POWER_SUMMARY.json')
comparison=read(OUT/'EXPERIMENT_COMPARISON.json')
fast=next(r for r in summary if r['case']=='single_fast')
nominal_tsr=3.67*fast['mean_approach_m_s']/1.
with (ROOT/'validation/hydrokinetic_turbines/measurements/mycek_2014_figures6a_7a.csv').open() as f:raw=list(csv.DictReader(f))
held={}
curves={field:sorted((float(r['TSR']),float(r['coefficient'])) for r in raw if r['field']==field and float(r['inflow_m_s'])==.8 and float(r['TI_percent'])==3) for field in ['Ct','Cp']}
# Reproduce the actual table: Cp is sampled onto the Ct knots when authoring
# the input, then both columns are linearly interpolated by the solver.
knots=np.array([x for x,v in curves['Ct'] if x>0])
ct=np.array([v for x,v in curves['Ct'] if x>0]);cp=np.interp(knots,*np.array(curves['Cp']).T)
inversions=[]
for row in sorted(comparison,key=lambda r:r['spacing_D']):
    if row['turbine']!='Downstream rotor':continue
    def coefficient(values,u):return np.interp(4*.8/u,knots,values)
    up=brentq(lambda u:coefficient(cp,u)*(u/.8)**3-row['Cp_nominal_measured'],.4,.8)
    ut=brentq(lambda u:coefficient(ct,u)*(u/.8)**2-row['Ct_nominal_measured'],.4,.8)
    inversions.append(dict(spacing_D=row['spacing_D'],power_implied_uniform_inflow_m_s=up,thrust_implied_uniform_inflow_m_s=ut,
        thrust_error_if_power_matched_percent=100*(coefficient(ct,up)*(up/.8)**2/row['Ct_nominal_measured']-1),
        note='Diagnostic inversion of measured loads using the exact input lookup table, not predictive calibration or measured approach velocity'))
save(OUT/'INCOMING_VELOCITY_INVERSION.json',inversions)
for field,quantity,power in [('Cp','mean_power_W',3),('Ct','mean_thrust_N',2)]:
    points=sorted((float(r['TSR']),float(r['coefficient'])) for r in raw if r['field']==field and float(r['inflow_m_s'])==1 and float(r['TI_percent'])==3)
    target=float(np.interp(nominal_tsr,*np.array(points).T));value=fast[quantity]/(.5*1000*np.pi*.7**2/4)
    held[field]=dict(measured=target,predicted_nominal=value,relative_error_percent=100*(value/target-1))
save(OUT/'HELD_OUT_SPEED.json',dict(inflow_m_s=1.,local_TSR=3.67,mean_nominal_TSR=nominal_tsr,training_curve_inflow_m_s=.8,values=held,
    limitation='Single-rotor coefficient transfer only. Fixed viscosity; not a varying-discharge or downstream-rotor qualification.'))
records=[]
for folder in sorted(HOME.iterdir()):
    if not (folder/'DONE.json').exists():continue
    done=read(folder/'DONE.json');assert done['result']['status']=='complete' and done['result']['time_s']==60
    diag=read(folder/'run/DIAGNOSTICS.json')
    record=dict(case=folder.name,final_time_s=60.,maximum_water_residual_m3=max(abs(r['water_residual_m3']) for r in diag),
        budget_pass=all(abs(r['water_residual_m3'])<=r['water_roundoff_limit_m3'] for r in diag),
        files={str(p.relative_to(folder)):sha(p) for p in [folder/'SPEC.json',folder/'Project_data/braids_project.json',folder/'DONE.json',folder/'run/TURBINE_PERFORMANCE.json',folder/'run/DIAGNOSTICS.json']})
    assert record['budget_pass'];records.append(record)
assert len(records)==6
stage=read(OUT/'STAGE.json')
assert sha(ROOT/'desktop/runtime/RUNTIME.json')==stage['parent_runtime_sha256']
assert sha(ROOT/'desktop/qualification/power_array62/candidate/RUNTIME.json')==stage['runtime_sha256']
save(OUT/'RUN_RECEIPTS.json',records)
worst=max(r['maximum_water_residual_m3'] for r in records)
window=max(r['temporal_power_range_percent'] for r in summary)
table='\n'.join(f"| {r['spacing_D']}D | {r['power_W_measured']:.2f} | {r['power_W_braids']:.2f} | {r['power_relative_error_percent']:+.1f}% | {r['thrust_relative_error_percent']:+.1f}% |" for r in sorted(comparison,key=lambda r:r['spacing_D']) if r['turbine']=='Downstream rotor')
report=f'''# Power and far-wake audit 62

## Decision

The audit is complete; **wake-exposed turbine power is not qualified**. The existing
depth-averaged far-wake comparison is encouraging, but it does not establish
the velocity entering the rotor disk or downstream mechanical power. Do not use
the present array results as validated Swiss-river energy estimates.

Six bounded 60 s GPU cases and separate short CPU/GPU/restart checks finished.
No audit simulation remains running. Existing user sediment runs were untouched.
Installed Desktop preview60 is unchanged. Two source guard corrections are tested
in an isolated candidate, not installed as a new release. No hydraulic kernel,
viscosity calibration, performance curve or sediment law was changed.

## Independent downstream power comparison

The target is Mycek et al., Part II (2014), Figures 4a and 5a, with nominal inflow
0.8 m/s, ambient turbulence intensity 3%, upstream nominal TSR 4, and downstream
nominal TSR 4. Both model rotors have fixed RPM derived from that nominal TSR;
the model resolves their actual local TSR. Rotor diameter is 0.7 m and water
depth 2 m. Measured mean curves are extracted from the original PDF vectors.
The 0.001 coefficient extraction allowance is **not** experimental uncertainty.

| Separation | Measured power (W) | BRAIDS power (W) | Power error | Thrust error |
|---|---:|---:|---:|---:|
{table}

BRAIDS values are time averages from 40–60 s. Experimental values are converted
using water density 1000 kg/m3. Comparisons of dimensionless coefficients do not
depend on this common density choice. The largest power variation within this
averaging window among all six cases is {window:.3f}% of mean power; waiting
longer is not a credible remedy for the observed factor-of-three 6D discrepancy.
This is a bounded stationarity screen, not a claim of statistical convergence of
all turbulent fluctuations.

The upstream rotor yields about 41.02 W versus the isolated experimental curve's
41.92 W (−2.15%). This is principally a check of reproducing a supplied performance
curve, **not independent validation of blade aerodynamics**. At held-out inflow
1.0 m/s, using the unchanged 0.8 m/s curve and comparing at the corresponding
mean nominal TSR {nominal_tsr:.4f} (prescribed local TSR 3.67), the single-rotor nominal Cp error is
{held['Cp']['relative_error_percent']:+.2f}% and Ct error
{held['Ct']['relative_error_percent']:+.2f}%. That supports coefficient transfer
for this isolated rotor over these two speeds, not array or hydrograph validity.

![Measured and predicted downstream coefficients](MEASURED_ARRAY_POWER.png)

The experimental tandem power is independent of the isolated Ct/Cp curve used
as model input. By contrast, the additional prescribed-local-TSR 5D and 10D
diagnostics are not the experimental operating protocol. They predict downstream
power approximately 77% and 83% of isolated power respectively and establish
that interaction exists, not that it is quantitatively correct. Do not mix them
with fixed-RPM experimental results.

## Why a good 2D wake does not guarantee correct power

The retained Nash/Bahaj porous-disc depth-average comparison at 5–10D gives
**0.952 percentage-point RMS deficit error**, maximum 1.446 pp, on the D/8 mesh.
Agreement with the published 2D model is 2.207 pp RMS. This evidence is for a
single porous disc with its documented support/loading, not the rotating Mycek
turbine or an independent downstream-power measurement.

![Far wake and power sensitivity](FAR_WAKE_POWER_SENSITIVITY.png)

The cubic centreline proxy in the figure is only a sensitivity calculation.
It is neither rotor-integrated power nor a confidence interval for array power.

In the actual source path, BRAIDS takes a horizontally weighted depth-mean normal
velocity at the turbine. With A=pi D²/4, local depth H, and supplied Ct it uses
beta=A Ct/(D H), alpha_num=(1+sqrt(1-beta))/2 and U0=u_local/alpha_num.
This existing correction removes the turbine's own depth-averaged numerical
induction. It does **not** reconstruct the vertically nonuniform wake arriving
from an upstream rotor. Force and mechanical shaft power are then computed as
T=0.5 rho A Ct U0² and P=0.5 rho A Cp U0³. For fixed RPM, Ct and Cp are interpolated
at lambda=omega R/U0 and the operating point is solved consistently.

The shallow-water velocity includes water passing above and below the upstream
rotor. Treating that depth average as incoming rotor velocity overestimates
wake-exposed power. Mycek's measured hub-plane profiles and an explicitly assumed
axisymmetric depth projection illustrate this distinction:

![Hub-plane and depth-average distinction](VERTICAL_WAKE_AUDIT.png)

The radial-symmetry projection is a diagnostic assumption, **not a direct
measurement of the depth average**. It suggests a rotor mean U/U_inf of about
0.695 at 6D and 0.810 at 10D, while the planar BRAIDS projection gives about 0.928
and 0.941. These single-rotor wake data are at TSR 3.67; the tandem performance
test uses upstream nominal TSR 4. The paper itself discusses this small protocol
difference. Do not claim the inferred vertical discrepancy is the sole quantified
cause: wake mixing, velocity profile and rotor operating-range effects remain.

There is a second important limitation. Inverting the *isolated* curve to match
the measured downstream power would require uniform approach speeds
{inversions[0]['power_implied_uniform_inflow_m_s']:.3f} m/s (6D) and
{inversions[1]['power_implied_uniform_inflow_m_s']:.3f} m/s (10D). Matching thrust
instead requires {inversions[0]['thrust_implied_uniform_inflow_m_s']:.3f} and
{inversions[1]['thrust_implied_uniform_inflow_m_s']:.3f} m/s.
Thus a scalar velocity adjustment cannot match both quantities simultaneously.
The power-implied 6D speed is below the isolated turbine's approximately
0.6 m/s operating-range threshold discussed by the authors. These inversions
are diagnostics, not inferred measured velocities or calibration constants.
Mycek Part II, section 3, also reports reduced downstream efficiency after
normalization using disk-integrated velocity cubed, especially at low turbulence.
It explicitly distinguishes mean(U³) from mean(U)³. A corrected array model must
respect nonuniform inflow and performance applicability, not merely lower U
until the power curve fits.

## What was checked and corrected

1. Verified the source-to-output chain: shared hydraulic state for all turbines,
   summed momentum sinks, own-induction correction, local TSR, Ct/Cp lookup,
   density conversion, shutdown/reactivation and exported power values.
2. Fixed inconsistent reverse-flow handling: constant-Ct and prescribed-TSR
   controls now switch off for reversed/absent normal inflow, as the fixed-RPM
   control already did. This is the existing directed, zero-yaw rotor scope.
3. Reject constant/TSR curve pairs that imply more mechanical power than the
   current unblocked ideal-actuator envelope Cp <= Ct(1+sqrt(1-Ct))/2, including
   positive power at zero thrust. Fixed-RPM tables already checked this.
   Blockage-dependent coefficient conventions need their own identified model;
   this guard does not assert a universal bound for every confined turbine.
4. Sixty-five current-format focused tests passed, including eleven new checks.
   Two old UI fixtures were deselected after their recorded failure to load
   projects missing the required transition_layers field. Their failed run is
   retained in TESTS_FULL.xml; no legacy execution route was added.
5. Actual 7200-cell two-turbine CPU/GPU trajectories to 0.8 s agree to
   2.154e-14 in conserved fields. Split runs at 0.4 s restart exactly on each
   backend, including turbine diagnostics. This is numerical verification only.
6. All six longer runs pass their water-budget guards. Maximum absolute reported
   water residual across their saved diagnostics is {worst:.3g} m3.

The new guards do not change the forward-flow experimental runs and do not
resolve the physical downstream-power failure. The old installed runtime and
candidate hashes are recorded in STAGE.json. Native libraries are unchanged.

## Reproducibility and scope

Editable projects and results: `C:/Users/STREEM/Desktop/BRAIDS_Turbine_Power62`.
Each completed case contains Project.braids.json, Project_data, DEM.tif, SPEC.json,
prepared, run and DONE.json. Source, inputs, run receipts, extracted curves,
figures and test identities are retained with SHA256 values.

The constructed flume is 18 by 4 m; rotors are centered laterally, upstream rotor
at x=5 m, hub height 0.9 m, depth 2 m, D=0.7 m. The 0.1 m fixed mesh has 7200 cells.
Initial flow is uniform, inlet discharge 6.4 m3/s (8 for the fast case), outlet
stage 2 m, Chezy 1000, zero sediment and no secondary flow. The 60 s end time,
2 s output interval, CFL 0.5 and 0.2 s outer cap are identical. Rotor support drag
is not separately represented. Horizontal viscosity is fixed at approximately
0.006025 m2/s, inherited from the earlier nondimensional Mycek diagnostic.
It is **not** a validated conversion from the experimental 3% turbulence intensity.
Published dimensions and rotor operating points are matched; turbulence and
vertical flow are not an exact reproduction of the laboratory setup.

The failed first 6D deck is retained: equivalent decimal coordinates x=9.2 and
9.200000000000001 triggered a face-incidence preparation failure. The v2 synthetic
deck uses its exact declared lattice coordinate (one ULP difference); the
production geometry issue remains open. Other harness/deck failures and old
fixture failures are listed in RETAINED_FAILURES.json.

Scripts: desktop/power_array62.py (frozen candidate and runs),
reference_power62.py (original vector-curve extraction), analyze_power62.py
(figures and comparisons), check_power62.py (paired execution/restart), and
finish_power62.py (receipts/report). Do not overwrite completed case directories
when rerunning. Wall times are not a clean performance benchmark because some
qualification jobs overlapped.

![Completed diagnostic time histories](TANDEM_POWER.png)

## Implications for Swiss-river energy estimates

The main target is mechanical power per turbine under a discharge history,
including wake interactions at 5–10D. Electrical energy additionally requires
generator/drivetrain efficiency and a rated-power controller. Current CSV power
is instantaneous sampled mechanical power, and the reported energy here uses
trapezoidal integration at the 2 s output interval; it is not a solver-integrated
energy ledger. Fixed inlet speed tests do not qualify a changing hydrograph.

The existing fixed viscosity cannot be assumed to transfer across discharge,
depth, ambient turbulence and bank proximity. Abolghasemi et al.'s 2D/3D
assessment demonstrates this limitation for depth-averaged models, but its fitted
viscosity coefficients are scenario-specific. Do not copy them as a universal
Swiss-river law. Future rotor-inflow reconstruction must also account for bed
and free-surface proximity and avoid double-counting a wake already represented
in the shallow-water momentum field. Yaw and strongly nonuniform inflow need
separate evidence; straight aligned tests do not qualify them.

## Sources

- Mycek et al. (2014), Part II, experimental interaction/performance:
  https://doi.org/10.1016/j.renene.2013.12.048 ; author PDF
  https://archimer.ifremer.fr/doc/00175/28644/27110.pdf . Figures 4a/5a and section 3.
- Mycek et al. (2014), Part I, isolated Ct/Cp and wake measurements:
  https://doi.org/10.1016/j.renene.2013.12.036 . Retained source and digitization
  under validation/hydrokinetic_turbines/vendor and measurements.
- Nash et al. (2015), depth-averaged wake comparison:
  https://doi.org/10.3390/en81212380 . The retained nearwake60 inputs/results
  remain unchanged; power_array62 reanalyzes only their 5–10D interval.
- Abolghasemi et al., depth-averaged/3D array assessment:
  https://arxiv.org/abs/1810.09827 . Own-induction correction and flow-dependent
  mixing are distinct from reconstructing an upstream rotor's vertical wake.

## Next milestone

Qualify an explicitly identified rotor-inflow/wake model for downstream thrust
and mechanical power at 5–10D, while retaining the verified depth-mean momentum
balance. Use published equations, experimental disk/wake measurements and
held-out spacings/speeds. No turbine-bedload coupling or national energy forecast
is admitted by this failed array-power comparison.

## Immediate next step

Reproduce the paper's rotor-area velocity-square and velocity-cube integrals and
downstream efficiency normalization from the existing measured wake data.
Separate missing vertical wake structure from performance loss under nonuniform
or out-of-range inflow. Then test a named reconstruction with a consistent
momentum sink on the frozen 6D/10D cases, without retuning Ct/Cp or independently
fitting each spacing. Retain both force and power as acceptance outputs before
advancing to a varying-discharge energy test.
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
(HOME/'README.md').write_text('''# Turbine power audit 62

These are completed research comparisons, not validated array-energy forecasts.
Open a case's Project.braids.json in BRAIDS Desktop to inspect its configuration.
Its results are in the same case's run directory; select that directory when
loading results. The input DEM and project data are retained alongside it.
The research runs used an isolated candidate. Installed preview60 is unchanged;
do not treat candidate checkpoints as exact-restart checkpoints for a different
runtime. No long run was launched.

Read the audit report in the BRAIDS repository at
validation/hydrokinetic_turbines/reports/power_array62/REPORT.md.
The matched 6D and 10D tandem cases significantly overpredict measured downstream
power. The constant-local-TSR tandem cases are separate controlled diagnostics.
Failed preparation directories are retained as failure evidence.
''',encoding='utf-8')
source_names=['braids_core/river_turbines.py','braids_core/sediment/river_turbine_flow.py',
    *['desktop/'+n+'.py' for n in ['power_array62','analyze_power62','reference_power62','check_power62','finish_power62']],
    'tests/test_turbine_power62.py','tests/test_turbine_power62_ui.py']
for name in source_names:
    dest=OUT/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
for name in stage['changed']:
    dest=OUT/'installed_reference'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'desktop/runtime'/name,dest)
save(OUT/'MANIFEST.json',dict(files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='MANIFEST.json'},
    installed_unchanged=True,physical_array_power_pass=False,all_six_runs_complete=True))
print(json.dumps(dict(runs=len(records),maximum_water_residual_m3=worst,maximum_averaging_window_variation_percent=window,held_out=held),indent=2))
