"""Compare the isolated source-lease/final-constructor performance gate."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/_inlet129'


def main():
    dirs=[OUT/'amr_reference_clean2',OUT/'amr_optimized_clean2']
    profiles=[json.loads((p/'PROFILE_SUMMARY.json').read_text()) for p in dirs]
    indices=[json.loads((p/'run/frames/INDEX.json').read_text()) for p in dirs]
    assert indices[0]==indices[1], 'Mesh/frame receipts differ'
    fields=[]
    for row in indices[0]['frames']:
        for name in (row['file'],row['epoch']):
            expected=row['sha256'] if name==row['file'] else indices[0]['epochs'][name]
            for path in dirs:
                assert hashlib.sha256((path/'run/frames'/name).read_bytes()).hexdigest()==expected
        with np.load(dirs[0]/'run/frames'/row['file'],allow_pickle=False) as a, np.load(dirs[1]/'run/frames'/row['file'],allow_pickle=False) as b:
            assert set(a.files)==set(b.files)
            for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
            fields.append(dict(time_s=row['time_s'],fields=a.files,all_array_bytes_identical=True))
    for key in ('hydraulic','accepted_steps','integrator','time_s','rejected_trials'):
        assert profiles[0]['result'][key]==profiles[1]['result'][key],key
    identities=[json.loads((p/'LEASE.json').read_text()) for p in dirs]
    phase_names=['load','initialization','amr','stepping','output','checkpoint','other']
    timings=[{k:p['phases'][k] for k in phase_names} for p in profiles]
    gain=dict(amr_fraction_saved=1-timings[1]['amr']/timings[0]['amr'],
              worker_fraction_saved=1-profiles[1]['worker_call_s']/profiles[0]['worker_call_s'],
              worker_speedup=profiles[0]['worker_call_s']/profiles[1]['worker_call_s'])
    payload=dict(scope='Isolated immutable-lease/final-constructor changes on installed alpha127; boundary and timestep arithmetic unchanged',
        directories=[str(p.relative_to(ROOT)) for p in dirs],
        runtime_identity=profiles[0]['runtime_identity'],source_project_sha256=profiles[0]['source_project_sha256'],
        cpu_threads=json.loads((dirs[0]/'PROJECT.json').read_text())['settings']['controls']['threads'],
        timings_s=timings,worker_call_s=[p['worker_call_s'] for p in profiles],
        preparation_s=[p['preparation_s'] for p in profiles],improvement=gain,
        native_and_lease=identities,fields=fields,hydraulic=profiles[0]['result']['hydraulic'],
        frame_and_geometry_sha256_identical=True,
        excluded='amr_reference_clean overlaps another agent test; lease_timing overlaps diagnostics; not used for performance claim',
        qualification='Same physical run, geometry, outputs, CPU thread count and all numerical operators. No changed physics or relaxed integrity tests.',
        limitations='Single sequential clean pair; residual AMR cost remains; no claim of Iber equivalence or whole GPU speedup')
    (OUT/'AMR_SPEED.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4),gridspec_kw=dict(width_ratios=[2,1]))
    colors=['#c5cee4','#8599c7','#e89244','#4f9f99','#ba97ba','#bcbd73','#b8b8b8']
    left=np.zeros(2)
    for name,color in zip(phase_names,colors):
        values=np.array([p[name] for p in timings]);axes[0].barh([0,1],values,left=left,color=color,label=name);left+=values
    axes[0].set_yticks([0,1],['Before','After']);axes[0].invert_yaxis();axes[0].set_xlabel('Worker elapsed time [s]')
    axes[0].set_title('Same 60 s dynamic-AMR CPU river run')
    axes[0].legend(ncol=4,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.17))
    axes[1].bar(['Before','After'],[i['native_constructions'] for i in identities],color=['#c5cee4','#4f9f99'])
    axes[1].set_ylabel('Native hydraulic constructions');axes[1].set_title('26 owners; one final operator each')
    for i,row in enumerate(identities):axes[1].text(i,row['native_constructions']+1,str(row['native_constructions']),ha='center')
    fig.suptitle(f"AMR elapsed cost {100*gain['amr_fraction_saved']:.1f}% lower; full worker {100*gain['worker_fraction_saved']:.1f}% lower — identical frames",fontsize=12)
    fig.tight_layout();fig.savefig(OUT/'AMR_SPEED.png',dpi=160,bbox_inches='tight');plt.close(fig)
    print(json.dumps(dict(**gain,all_frame_and_mesh_hashes_identical=True,frames=len(fields)),indent=2))


if __name__=='__main__':main()
