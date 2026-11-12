"""Final evidence receipt; verify all preserved runtime payloads read-only."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_inlet187'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_inlet187'
sys.path[:0]=[str(OUT/'prescribed_candidate'),str(ROOT),str(ROOT/'desktop')]
import numpy as np
from braids_desktop.results import ResultsStore
from stage_secondary185 import save,sha
from qualify_inlet187 import checkpoint_arrays

def main():
    runtimes={}
    for label,folder,expected in [
        ('installed',ROOT/'desktop/runtime','fddbf4c03a89f4d19aa72e89dc3fb02967470289d0621c1c47d7ab9a73f71ae3'),
        ('v6',ROOT/'validation/_secondary185/candidate','1b078961ad95f0038ba07da0a4aca9c3dd15a7f3cbffff2ec413986d27b4723b'),
        ('prescribed',OUT/'prescribed_candidate','8b27294c161aec91332a74c89b9de99c9c91cd78a94630369bc0557ae8081015')]:
        assert sha(folder/'RUNTIME.json')==expected
        doc=json.loads((folder/'RUNTIME.json').read_text())
        for n,h in doc['files'].items():assert sha(folder/n)==h,(label,n)
        runtimes[label]=dict(manifest=expected,payloads_verified=len(doc['files']))
    checks={};restarts={}
    for name in ('wet_prescribed_cpu','wet_prescribed_gpu','dry_prescribed_cpu','dry_prescribed_gpu'):
        checks[name]=json.loads((OUT/name/'CHECK.json').read_text())
        restarts[name]=json.loads((OUT/name/'restart/CHECK.json').read_text())
        assert all(restarts[name]['frame_exact'].values()) and all(restarts[name]['checkpoint_arrays_exact'].values())
    smoke=OUT/'smoke';a=ResultsStore(smoke/'prescribed/smart_jaggi_cpu/full').read(-1)
    b=ResultsStore(smoke/'prescribed/smart_jaggi_gpu/full').read(-1)
    amr=dict(max_bed_mm=float(abs(a.z-b.z).max()*1000),max_depth_mm=float(abs(a.q[0]-b.q[0]).max()*1000))
    for backend in ('cpu','gpu'):
        path=smoke/f'prescribed/smart_jaggi_{backend}'
        a0=checkpoint_arrays(path/'full/checkpoint.npz');b0=checkpoint_arrays(path/'resumed/checkpoint.npz')
        exact={k:bool(np.array_equal(a0[k],b0[k])) for k in a0}
        assert all(exact.values()),(backend,[k for k,v in exact.items() if not v])
        amr[backend]=dict(check=json.loads((path/'CHECK.json').read_text()),checkpoint_arrays_exact=exact)
    a=ResultsStore(smoke/'advanced_off/smart_jaggi_cpu/full').read(-1)
    b=ResultsStore(ROOT/'validation/_secondary185/smoke/release_v6/smart_jaggi_cpu/full').read(-1)
    inactive={k:bool(np.array_equal(a.arrays[k],b.arrays[k])) for k in ('q','z','W','mass','water','bed_boundary')}
    assert all(inactive.values())
    replays={}
    for mode in ('common','continuous','outlet'):
        with np.load(OUT/f'replay/{mode}_cpu/FIELDS.npz') as a,np.load(OUT/f'replay/{mode}_gpu/FIELDS.npz') as b:
            replays[mode]=dict(max_bed_mm=float(abs(a['final_z']-b['final_z']).max()*1000),
                max_depth_mm=float(abs(a['final_q'][0]-b['final_q'][0]).max()*1000))
            if mode=='common':
                keys=[k for k in a if k.startswith('refresh_secondary_momentum_00')]
                replays[mode]['secondary_first_state_max']={k:float(abs(a[k]-b[k]).max()) for k in keys}
                replays[mode]['hydraulic_first_rate_max']=float(abs(a['hydraulic_00_r']-b['hydraulic_00_r']).max())
    summary=json.loads((REPORT/'COMPARISON.json').read_text())
    inlet=all(summary[s]['inlet_max_bed_difference_mm']<1. and summary[s]['inlet_max_depth_difference_mm']<1. for s in ('wet','dry'))
    domain=all(summary[s]['max_bed_difference_mm']<1. and summary[s]['max_depth_difference_mm']<1. for s in ('wet','dry'))
    assert inlet and not domain
    save(REPORT/'QUALIFICATION.json',dict(runtimes=runtimes,cold=checks,restart=restarts,
        dynamic_amr=amr,advanced_off_exact=inactive,replays=replays,inlet_screen_passed=inlet,
        domain_screen_passed=domain,installed=False,test_report=str(OUT/'PRESCRIBED_TESTS.xml')))
    print('Receipt verified: inlet PASS, domain screen FAIL, installation held.',flush=True)

if __name__=='__main__':main()
