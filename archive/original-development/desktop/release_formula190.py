"""Seal preview31 without changing the qualified numerical payload."""
import json,shutil
from stage_formula190 import ROOT,TARGET,OUT
from stage_secondary185 import save,sha
EXPECTED='4246ab4675696c6cb53a4c607ed560eda5e3363c82f6f0cb56e1ed5ff0c7f0dc'
RELEASE=OUT/'release_explicit'

def verify(root):
    manifest=json.loads((root/'RUNTIME.json').read_text())
    for name,h in manifest['files'].items():
        p=(root/name).resolve();assert p.is_relative_to(root.resolve()) and sha(p)==h,name
    return manifest

def main():
    assert sha(TARGET/'RUNTIME.json')==EXPECTED
    manifest=verify(TARGET)
    shutil.copytree(TARGET,RELEASE,ignore=shutil.ignore_patterns('__pycache__'))
    names=['braids_core/sediment_method_catalog.py','braids_desktop/__init__.py']
    assert '__version__ = "0.1.0-preview31"' in (ROOT/names[1]).read_text()
    for name in names:
        shutil.copy2(ROOT/name,RELEASE/name);manifest['files'][name]=sha(RELEASE/name)
    save(RELEASE/'RUNTIME.json',manifest);verify(RELEASE)
    delta=[n for n in manifest['files'] if sha(TARGET/n)!=sha(RELEASE/n)]
    assert sorted(delta)==names,delta
    save(OUT/'RELEASE.json',dict(candidate=EXPECTED,release=sha(RELEASE/'RUNTIME.json'),
        changed_payloads=delta,numerical_payload_identical=True,
        changes='Version and qualified route catalog metadata only',payloads=len(manifest['files'])))
    print('SEALED preview31',sha(RELEASE/'RUNTIME.json'),len(manifest['files']))
if __name__=='__main__':main()
