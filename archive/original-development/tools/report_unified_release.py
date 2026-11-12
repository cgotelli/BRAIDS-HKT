"""Collect deployment evidence without changing a solver or running a case."""
import argparse
import json
from pathlib import Path
import numpy as np
from build_unified_qgis import verify,sha,write


def frame(folder):
    root=Path(folder)/'frames';doc=json.loads((root/'INDEX.json').read_text())
    row=doc['frames'][-1]
    with np.load(root/row['file'],allow_pickle=False) as f:state={k:f[k] for k in f.files}
    with np.load(root/row['epoch'],allow_pickle=False) as f:leaves=f['leaves'].copy()
    return state,leaves


def compare(a,b):
    left,li=frame(a);right,ri=frame(b);np.testing.assert_array_equal(li,ri)
    return {k:float(np.max(np.abs(left[k]-right[k]))) for k in ('q','z','mass','W')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();e=a.evidence.resolve();out=a.report.resolve();out.mkdir(parents=True,exist_ok=True)
    install=json.loads((e/'INSTALLATION.json').read_text());manifest=verify(Path(install['destination']))
    checks={}
    for name in ('staged_checks','qgis_staged','qgis_installed','installed_bank_checks'):
        doc=json.loads((e/name/'CHECKS.json').read_text());assert all(doc['checks'].values())
        checks[name]=dict(assertions=len(doc['checks']),all_passed=True)
    staged=json.loads((e/'qgis_staged/CHECKS.json').read_text())['results']
    installed=json.loads((e/'qgis_installed/CHECKS.json').read_text())['results']
    pairs={mode:compare(installed['cpu_'+mode],installed['gpu_'+mode]) for mode in ('hydro','bedload')}
    unchanged={key:compare(staged[key],installed[key]) for key in installed}
    banks=json.loads((e/'installed_bank_checks/CHECKS.json').read_text())['runs']
    pairs['surveyed_block_and_repose']=compare(*banks.values())
    record=json.loads((e/'RECORD.json').read_text())
    metrics=dict(version=install['version'],installed=install,checks=checks,cpu_gpu_maximum_absolute=pairs,
                 staged_installed_maximum_absolute=unchanged,native_binding_count=len(json.loads((e/'BINDINGS.json').read_text())),
                 recorded_projects=len(record['projects']),manifest_sha256=sha(Path(install['destination'])/'MANIFEST.json'),
                 source_identity=manifest['source_identity'],recorded_source_identity=manifest['recorded_source_identity'],
                 loader_only_update=manifest['loader_only_update'],cohesive_in_release=False)
    write(out/'METRICS.json',metrics)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
