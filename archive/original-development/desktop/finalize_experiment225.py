"""Seal the experiment handoff after a confirmed, independently supervised launch."""
from pathlib import Path
import json,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1]
CASE=ROOT.parent/'BRAIDS_JiangCao_Experiment225'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_experiment225'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    clean=read(ROOT/'desktop/cleanup225_receipt.json');audit=read(CASE/'INPUT_AUDIT.json')
    bench=read(CASE/'CORE_BENCHMARK.json');backend_bench=read(CASE/'BACKEND_BENCHMARK.json')
    selection=read(CASE/'RUN_SELECTION.json');state=read(CASE/'RUN_STATE.json');process=read(CASE/'RUN_PROCESS.json')
    assert len(clean['records'])==8 and all(r['all_file_sha256_equal'] for r in clean['records'])
    assert state['status'] in ('running','complete') and state['time_s']>0,state
    assert sha(CASE/'runtime/RUNTIME.json')==audit['runtime']
    import numpy as np
    frames=CASE/'runs/experiment225/frames';ix=read(frames/'INDEX.json')
    with np.load(frames/ix['frames'][0]['file']) as f:
        assert f['state'].shape==(7,15000)
        assert not np.any(f['q']) and not np.any(f['mobile_bedload_solid_m']) and not np.any(f['secondary_hI_m2_s'])
        initial=f['z'].copy()
    with np.load(frames/ix['frames'][0]['epoch']) as mesh:
        v=mesh['vertices'];o=mesh['offsets'];x=np.array([v[o[i]:o[i+1],0].mean() for i in range(len(o)-1)])
    # Remove the actual terrain-to-mesh linear plane before measuring the seeded
    # perturbation.  The sampler introduces a constant sub-pixel intercept offset;
    # counting it as bed noise would falsely inflate the documented amplitude.
    plane=np.polyfit(x,initial,1);assert abs(plane[0]+.01)<1e-5,plane
    perturb=initial-np.polyval(plane,x)
    assert np.max(np.abs(perturb))<=.000100001
    table='\n'.join(f"| {r['label']} | {r['threads']} | {r['step_s']:.4f} | {r['warm_step_median_s']:.6f} | {r['rejections']} |" for r in bench['rows'])
    freed=(clean['freeNow']-clean['freeBefore'])/2**30
    assert process['backend']==selection['backend']=='cpu'
    doc=f'''# Jiang-Cao experiment225 — MF1 launch

Current-format project: `{CASE}`. The selected {selection['backend'].upper()} worker is confirmed advancing at the handoff;
latest live state is in the project `RUN_STATE.json`, not this static report.
The independent supervisor remains running after this chat ends.

## Disk cleanup

Reclaimed **{freed:.2f} GiB** on C: by relocating eight historical build/performance
folders to `{clean['archive']}`. Original paths are NTFS directory links. Every
copied file was checked with SHA256 before removal of its C: copy. Long paths
needed corrected Windows verification/removal handling; completed copy receipts
were retained and revalidated during recovery. Sources, installed Desktop,
current examples, papers and `resources/Delft3D-main` stayed in place. Archives
contain the earlier failed/mixed trials as well as passing results.

## Reconstructed experiment

Jiang-Cao Case 1 Run 1, based on Garcia Lugo Run 27: 25 m x 1.5 m, slope 0.01,
water supply 0.0025 m³/s, solid sediment feed 0.00286 kg/s, uniform 1 mm sediment,
nominal mesh length 0.05 m. Target is **18000 s / 300 min**, with **MF1**.
The supplied paper uses triangles; BRAIDS uses 15000 fixed Cartesian squares.
Published values are checked against Table 1 and section 2.2.1 of `chino.pdf`.
The paper shows planform evolution at 50–300 min; startup alone cannot establish
reproduction of those patterns.

The user approved a fixed-seed bed perturbation: seed 225, maximum amplitude
0.1 mm, Gaussian correlation scale 0.1 m and a 0.5 m inlet/outlet taper. The actual
prepared cell-bed perturbation is at most {np.max(np.abs(perturb))*1000:.6g} mm.
This is an explicitly recorded reconstruction assumption, not author-provided
initial topography. Other assumptions: initially dry bed, zero mobile sediment
and secondary intensity, 0.2 m erodible stock, porosity 0.4, repose 30 degrees,
full-width normal-discharge inlet and critical-export outlet. D50 and D90 are
both explicitly set to 1 mm for the uniform numerical idealization; no measured
grading curve was available and no percentile is silently derived. The earlier
short fixture's D90=1.5 mm is not reused for this uniform-grain experiment.

The full seven-field Jiang-Cao reconstruction owns resistance, phase exchange,
bedload direction, density/water feedback and secondary flow, with its own repose
collapse. Separate ordinary bank/toe supply and independent suspension are OFF.
BRAIDS retains augmented f-wave hydraulics and its named shallow-front and
finite-adjustment extensions; this is not the original HLLC implementation.
See `INPUT_AUDIT.json` for the exact distinctions and source hashes.

Hydraulic inlet Q includes the small solid volume: {audit['conversion']['mixture_discharge_m3_s']:.12g} m³/s.
The saved solid/water ratio is {audit['conversion']['solid_water_ratio']:.12g}.
Maximum dt is 0.005 s; existing CFL, positivity and rejection controls remain
active. Results are saved every 60 simulated seconds; checkpoints every 300 s,
plus the accepted final/cancelled state. All project assets, settings, history,
results, logs, checkpoints and one frozen solver copy are in the case folder.

## Execution benchmark

Ryzen 9 9950X3D: 16 physical cores / 32 logical processors. Selected
**{bench['selected_threads']} CPU threads** after sequential 1/2/4/8/16/32 trials
and reverse-order repeats of the best two. These short trials use this actual
full mesh and cold-start physics; construction/output are excluded from the
step timings. The best count may change once more of the flume becomes wet.
All trial outputs were compared against one thread: maximum bed difference
{max(r['bed_mm'] for r in bench['field_differences']):.6g} mm, depth difference
{max(r['depth_mm'] for r in bench['field_differences']):.6g} mm.

| Trial | Threads | Total stepping (s) | Warm median step (s) | Rejections |
| --- | ---: | ---: | ---: | ---: |
{table}

![CPU core benchmark](CORE_BENCHMARK.png)

The paired GPU diagnostic used the same mesh, initial state and 50 nominal steps.
Its median stepping time was **{backend_bench['gpu_step_s']:.4f} s**, versus
**{backend_bench['cpu_step_s']:.4f} s** for the selected 8-thread CPU route.
The CPU was therefore **{selection['cpu_over_gpu_speed_ratio']:.2f}× faster** for
this case and was retained explicitly. The maximum CPU/GPU bed and depth
differences were below 6e-14 mm. No automatic backend selector was introduced.

## Monitor and completion

In the project folder:

- **Open progress.cmd**: browser dashboard, refresh every 10 seconds.
- **Check progress.cmd**: console progress; signals completion/failure.
- **Open experiment.cmd**: frozen Desktop interface with the project.
- **Stop safely.cmd**: requests cancellation and an accepted-state checkpoint.

`FINISHED.txt` and `COMPLETION.json` are created when the worker exits, including
failure or cancellation. The dashboard reports errors rather than claiming a
failed run finished successfully. ETA is estimated from recent accepted advance
and changes as wetting, transport and collapse evolve. The image updates only
when a result frame is published. Worker/supervisor PIDs at launch:
{process['worker_pid']} / {process['supervisor_pid']}. Never start a second worker
into the same output folder.

Runtime SHA256: `{audit['runtime']}`. This frozen runtime is independent of later
Desktop solver edits. Delft3D source/runtime were not used or modified here.

## Next milestone

After reviewing this MF1 evolution, add and qualify MF different from 1 for the
full coupled Jiang-Cao bundle on CPU and GPU. Preserve water/solid/pore-water
accounting and distinguish physical advection from accelerated bed exchange.

## Immediate next step

Let this baseline advance. Check its status with either monitor; when sufficient
morphology is available, compare planform and bed sections at x=7 and 12.5 m with
the paper before selecting a bounded accelerated counterpart. Do not modify the
running baseline's parameters or its frozen runtime.
'''
    (REPORT/'REPORT.md').write_text(doc,encoding='utf8')
    (CASE/'README.md').write_text(doc,encoding='utf8')
    for name in ('INPUT_AUDIT.json','CORE_BENCHMARK.json','CORE_BENCHMARK.png','BACKEND_BENCHMARK.json','RUN_SELECTION.json','RUN_PROCESS.json'):
        shutil.copy2(CASE/name,REPORT/name)
    shutil.copy2(ROOT/'desktop/cleanup225_receipt.json',REPORT/'CLEANUP.json')
    heading=f'''# Desktop225 — Jiang-Cao MF1 experiment running

Project: `{CASE}`. Frozen preview48 / project v36; MF1, one uniform material,
5 cm fixed mesh, full Jiang-Cao transport/secondary/repose bundle. User-approved
0.1 mm fixed-seed perturbation, documented assumptions and published inputs.
Execution selected by measured trials: {selection['backend'].upper()} with {bench['selected_threads']} CPU threads.
Disk cleanup reclaimed {freed:.2f} GiB; originals are verified on D: with directory links.
Progress/completion controls are supplied in the project. Running is not a
claim of experimental reproduction. Ordinary formula compatibility remains
as qualified in Desktop224. Full Jiang-Cao MF>1 remains the next development milestone.

Report: `sediment/river_morphodynamics/reports/desktop_experiment225/REPORT.md`.
Immediate next step: inspect the advancing MF1 morphology, then qualify accelerated
phase/bed/water coupling on both backends without modifying this baseline.

---

'''
    p=ROOT/'validation/RIVER_MORPHODYNAMICS_STATUS.md';p.write_text(heading+p.read_text(encoding='utf8'),encoding='utf8')
    print('Experiment launch and monitoring handoff recorded',flush=True)

if __name__=='__main__':main()
