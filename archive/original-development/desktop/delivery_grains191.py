"""Final installed receipt; no package edits or solver execution."""
import json
from stage_grains191 import ROOT,OUT,TARGET,FILES,sha,verify
from install_grains191 import REPORT,suite

def main():
    runtime=ROOT/'desktop/runtime';manifest=verify(runtime)
    assert sha(runtime/'RUNTIME.json')==sha(TARGET/'RUNTIME.json')
    for n in FILES:assert sha(ROOT/n)==sha(runtime/n),n
    suite(OUT/'CONTRACT_INSTALLED.xml',82)
    suite(ROOT/'validation/_formula190/GRAINS191_INSTALLED.xml',180)
    checks={}
    for backend in ('cpu','gpu'):
        row=json.loads((OUT/'installed'/backend/'CHECK.json').read_text())
        assert row['runtime']==sha(runtime/'RUNTIME.json') and len(row['checks'])==6
        for law,check in row['checks'].items():
            assert all(check['restart_exact'].values()) and all(check['previous_release_fields_exact'].values())
        checks[backend]=row
    result=dict(version='0.1.0-preview32',runtime=sha(runtime/'RUNTIME.json'),
        payloads_verified=len(manifest['files']),installed_tests=262,
        installed_coupled_cases=12,checks=checks)
    (REPORT/'DELIVERY.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Delivery verified:',result['installed_tests'],'tests and',result['installed_coupled_cases'],'coupled cases')
if __name__=='__main__':main()
