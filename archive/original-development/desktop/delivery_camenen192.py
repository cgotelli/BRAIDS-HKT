"""Check the installed release and make a new self-contained groin example."""
import json,sys
from stage_camenen192 import ROOT,OUT,TARGET,FILES,sha,verify,save
from install_camenen192 import REPORT,suite
def main():
    runtime=ROOT/'desktop/runtime';manifest=verify(runtime);identity=sha(runtime/'RUNTIME.json')
    assert identity==sha(TARGET/'RUNTIME.json')
    for n in FILES:assert sha(ROOT/n)==sha(runtime/n),n
    suite(OUT/'INSTALLED_REGRESSION.xml')
    checks={}
    for backend in ('cpu','gpu'):
        for start in ('wet','dry'):
            path=OUT/'installed/runs'/start/('camenen_larson_2005_'+backend)
            c=json.loads((path/'CHECK.json').read_text());assert c['runtime']==identity
            assert all(c['restart_exact'].values()) and c['inlet_capacity_oracle'] and c['compiler_free']
            checks[start+'_'+backend]=c
    sys.path[:0]=[str(runtime),str(ROOT)]
    import numpy as np
    from braids_desktop import services
    from braids_desktop.results import ResultsStore
    for name in checks:
        start,backend=name.split('_')
        a=ResultsStore(OUT/'installed/runs'/start/('camenen_larson_2005_'+backend)/'full').read(-1)
        b=ResultsStore(OUT/'release/runs'/start/('camenen_larson_2005_'+backend)/'full').read(-1)
        exact={k:bool(np.array_equal(a.arrays[k],b.arrays[k])) for k in ('q','z','mass','water','bed_boundary')}
        assert all(exact.values()),exact
        checks[name]['release_exact']=exact
    source=OUT/'installed/runs/wet/camenen_larson_2005_gpu/project'
    example=ROOT/'desktop/examples/bedload192/camenen_larson'
    assert not example.exists()
    doc=services.load_project_folder(source)
    doc['name']='Camenen & Larson - single-grain groin'
    services.save_project_folder(example,doc,source_base=source)
    readback=services.load_project_folder(example)
    assert readback['model']['physics']['sediment']['capacity']['method']=='capacity.camenen_larson_2005'
    save(REPORT/'DELIVERY.json',dict(version='0.1.0-preview33',runtime=identity,tests=296,
        payloads_verified=len(manifest['files']),checks=checks,example=str(example/'braids_project.json')))
    print('Installed tests and four full/restart cases passed; example:',example,flush=True)
if __name__=='__main__':main()
