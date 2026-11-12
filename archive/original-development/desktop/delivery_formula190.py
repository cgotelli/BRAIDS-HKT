"""Read back the installed preview and its post-install execution receipts."""
import json,xml.etree.ElementTree as ET
from stage_formula190 import ROOT,OUT
from stage_secondary185 import save,sha
from release_formula190 import RELEASE,verify
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_transport190'

def main():
    base=ROOT/'desktop/runtime';manifest=verify(base)
    assert sha(base/'RUNTIME.json')==sha(RELEASE/'RUNTIME.json')
    suite=ET.parse(OUT/'REGRESSION_INSTALLED.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==180 and int(suite.attrib['failures'])==int(suite.attrib['errors'])==0
    checks={}
    for backend in ('cpu','gpu'):
        row=json.loads((OUT/'installed_smoke'/(backend+'_CHECK.json')).read_text())
        assert row['runtime']==sha(base/'RUNTIME.json') and row['version']=='0.1.0-preview31'
        assert row['compiler_free'] and row['sediment_default_off'] and row['old_checkpoint_rejected']
        assert all(all(v.values()) for v in row['fresh_prepare_and_6_to_8s_restart_exact'].values())
        checks[backend]=row
    installation=json.loads((OUT/'INSTALL.json').read_text())
    for name in installation['changed']:
        if name.endswith(('.py','.h')):assert sha(ROOT/name)==sha(base/name),name
    assert sha(ROOT/'braids_core/sediment/river_sediment_clock.py')==sha(base/'braids_core/sediment/river_sediment_clock.py')
    save(REPORT/'DELIVERY.json',dict(version='0.1.0-preview31',runtime=sha(base/'RUNTIME.json'),
        payloads_verified=len(manifest['files']),installed_regression_tests=180,checks=checks,
        installation=installation,checkout_changed_sources_match_installed=True,
        examples=json.loads((OUT/'EXAMPLES.json').read_text())))
    print('DELIVERY VERIFIED',len(manifest['files']),'payloads; 180 installed tests; both backends')
if __name__=='__main__':main()
