"""Measured milestone129 timing summary; originals are read-only inputs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'validation/_inlet129'
REPORT = ROOT / 'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/inlet129'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    rows = []
    cases = [
        ('2 m GPU, fixed mesh, 100 s', 'run_20261022_204331_367153', 'clean_timing2m/2m_critical_gpu'),
        ('5 m CPU, dynamic AMR, 60 s', 'run_20261022_204720_f34397', 'clean_timing5m/5m_critical_cpu'),
    ]
    for label, old_name, new_name in cases:
        old_path = Path('C:/Users/STREEM/Desktop/aaa/runs') / old_name / 'STATUS.json'
        new_path = EVIDENCE / new_name / 'QUALIFICATION.json'
        old, result = read(old_path), read(new_path)
        new = result['status']
        assert old['status'] == new['status'] == 'complete'
        assert old['time_s'] == new['time_s']
        project = read(new_path.parent / 'PROJECT.json')
        rows.append(dict(
            label=label, original_status=str(old_path), corrected_qualification=str(new_path),
            runtime=result['runtime'], cpu_threads=project['settings']['controls']['threads'],
            simulation_s=new['time_s'], worker_s=[old['wall_s'], new['wall_s']],
            corrected_outer_call_s=result['wall_s'], corrected_preparation_s=result['preparation_s'],
            steps=[old['hydraulic']['accepted_substeps'], new['hydraulic']['accepted_substeps']],
            retries=[old['hydraulic']['cfl_retries'], new['hydraulic']['cfl_retries']],
            next_dt_s=[old['hydraulic']['next_substep_hint_s'], new['hydraulic']['next_substep_hint_s']],
            mean_dt_s=[s['time_s']/s['hydraulic']['accepted_substeps'] for s in (old,new)],
            worker_speedup=old['wall_s']/new['wall_s'],
            worker_fraction_saved=1-new['wall_s']/old['wall_s'],
            step_reduction_factor=old['hydraulic']['accepted_substeps']/new['hydraulic']['accepted_substeps']))
    amr = read(EVIDENCE / 'AMR_SPEED.json')
    summary = dict(
        scope='Original user timings versus isolated corrected runs; same status clock, not a numerical-equivalence A/B',
        caveats=[
            'Both corrected cases explicitly realign open sections and correct boundary fluxes.',
            '2m keeps CFL .5 and dt_max .5; 5m changes CFL .225 to .5 and dt_max .1 to .5.',
            'Worker status time excludes preparation; new sampling instrumentation remains enabled.',
            'One new isolated timing per case, warmed native cache; not a statistically repeated benchmark.',
            'Old and new hydraulic fields are not expected to match because the old inlet is defective.',
            'Iber is not a matched mesh/CFL/timing benchmark; the remaining speed gap is not closed.'
        ], cases=rows, isolated_equivalent_amr=amr['improvement'])
    (EVIDENCE/'DELIVERY_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    fig, axs = plt.subplots(1,3,figsize=(12,4),layout='constrained')
    colors=['#9b9b9b','#16738a']
    for ax, row in zip(axs[:2],rows):
        bars=ax.bar(['Original','Corrected'],row['worker_s'],color=colors,width=.6)
        ax.bar_label(bars,fmt='%.2f s',padding=3)
        ax.set(title=row['label'],ylabel='Recorded worker time (s)',ylim=(0,max(row['worker_s'])*1.2))
        ax.text(.5,.88,f"{row['worker_speedup']:.2f}× faster",ha='center',transform=ax.transAxes)
    values=[v['amr'] for v in amr['timings_s']]
    bars=axs[2].bar(['Reference','Optimized'],values,color=colors,width=.6)
    axs[2].bar_label(bars,fmt='%.2f s',padding=3)
    axs[2].set(title='Isolated equivalent CPU AMR A/B',ylabel='AMR rebuilding time (s)',ylim=(0,max(values)*1.2))
    axs[2].text(.5,.88,'39.2% less AMR time',ha='center',transform=axs[2].transAxes)
    fig.suptitle('Measured runtime improvement — left: corrected workflow; right: identical numerical results')
    fig.savefig(REPORT/'PERFORMANCE.png',dpi=170)
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
