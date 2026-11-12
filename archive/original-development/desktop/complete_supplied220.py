"""Combine identified runs without rewriting the interrupted original receipt."""
import os,sys,json,copy,xml.etree.ElementTree as ET
from stage_supplied220 import ROOT,OUT,TARGET,sha,save
sys.path[:0]=[str(TARGET),str(ROOT),str(ROOT/'tests')]
os.environ['QT_QPA_PLATFORM']='offscreen';os.environ['OMP_NUM_THREADS']='4'
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
verify();install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET)
import importlib
from pathlib import Path
for name in ('validation','validation.sediment','validation.sediment.river_morphodynamics'):
    importlib.import_module(name).__path__.append(str(ROOT/Path(*name.split('.'))))
TESTS=['test_supplied220','test_suspended_classic219','test_suspended_dl218',
    'test_suspended_gp217','test_suspended_vr216','test_suspended_setup214',
    'test_suspended_core215','test_suspended_exchange215','test_river_vr1993',
    'test_river_vr1993_binding','test_bulk213','test_jiang_cao207','test_jiang_cao208',
    'test_jiang_cao209','test_jiang_cao210','test_jiang_cao206_adjustment',
    'test_river_roughness','test_jiang_cao199']

def main():
    import pytest
    class Inventory:
        def pytest_collection_finish(self,session):
            self.keys={('tests.'+Path(i.path).stem,i.name) for i in session.items}
    inventory=Inventory()
    assert pytest.main(['--collect-only','-q',*['tests/'+n+'.py' for n in TESTS]],plugins=[inventory])==0
    assert len(inventory.keys)==1158
    inputs=['REGRESSION.xml','rebuilt_complete3.xml','recovered_supplied4.xml','rebuilt_vr5.xml']
    passed={};records=[];original_failures=[]
    for index,name in enumerate(inputs):
        path=OUT/name;tree=ET.parse(path);suite=tree.getroot().find('testsuite')
        failures=[t for t in suite.findall('testcase') if any(t.find(tag) is not None for tag in ('error','failure','skipped'))]
        if index==0:
            assert len(failures)==5 and int(suite.get('tests'))==943
            original_failures=[(t.get('classname'),t.get('name')) for t in failures]
        else:assert not failures,suite.attrib
        for t in suite.findall('testcase'):
            key=t.get('classname'),t.get('name')
            if t not in failures:passed[key]=(copy.deepcopy(t),name)
        records.append(dict(file=name,sha256=sha(path),summary=suite.attrib))
    assert set(passed)==inventory.keys,(inventory.keys-set(passed),set(passed)-inventory.keys)
    assert all(passed[key][1]!='REGRESSION.xml' for key in original_failures)
    combined=ET.Element('testsuites');suite=ET.SubElement(combined,'testsuite',
        name='supplied220_combined_distinct_cases',tests=str(len(passed)),failures='0',errors='0',skipped='0',
        time=str(sum(float(v[0].get('time','0')) for v in passed.values())))
    for key,(case,source) in sorted(passed.items()):
        props=case.find('properties')
        if props is None:props=ET.SubElement(case,'properties')
        ET.SubElement(props,'property',name='source_receipt',value=source);suite.append(case)
    ET.ElementTree(combined).write(OUT/'REGRESSION_COMPLETE.xml',encoding='utf-8',xml_declaration=True)
    sourcefiles=['tests/'+n+'.py' for n in TESTS]+['tests/rebuilt_suspension220_case.py','tests/rebuilt_vr1993_inputs220.py']
    save(OUT/'COMPLETION.json',dict(runtime=sha(TARGET/'RUNTIME.json'),distinct_passes=len(passed),
        inputs=records,combined_sha256=sha(OUT/'REGRESSION_COMPLETE.xml'),
        original_failures_retained=True,rebuilt_fixture='tests/rebuilt_suspension220_case.py',
        historical_fixtures_recovered=False,assertions_or_tolerances_relaxed=False,
        retained_native_replays=dict(count=3,receipt='REGRESSION.xml',runtime_unchanged=True,
            source_and_input_currently_unavailable=True,rerun=False),
        source_sha256={n:sha(ROOT/n) for n in sourcefiles}))
    print('Complete: 1158 distinct passing checks across explicitly identified runs.')

if __name__=='__main__':main()
