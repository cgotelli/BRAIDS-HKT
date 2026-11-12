"""Bounded fresh-process groups; retain every complete process log and XML."""
import os,sys,json,subprocess,xml.etree.ElementTree as ET
from copy import deepcopy
from stage_banks224 import ROOT,OUT,TARGET,save,sha

def main():
    fix=json.loads((OUT/'BINDING_FIX.json').read_text())
    assert fix['runtime']==sha(TARGET/'RUNTIME.json')
    parent=OUT/'candidate_release'
    before=json.loads((parent/'RUNTIME.json').read_text())['files']
    after=json.loads((TARGET/'RUNTIME.json').read_text())['files']
    assert all(after[n]==h for n,h in before.items() if n!='PRECOMPILED.json')
    old_bindings=json.loads((parent/'PRECOMPILED.json').read_text())
    new_bindings=json.loads((TARGET/'PRECOMPILED.json').read_text())
    assert all(new_bindings[k]==v for k,v in old_bindings.items())
    assert set(new_bindings)-set(old_bindings)=={fix['new_binding']}
    nodes=json.loads((OUT/'COLLECTION.json').read_text())
    matrix=next(n for n in nodes if n.endswith('::test_matrix_backend_agreement'))
    nodes.remove(matrix)
    # Main-process Qt editors and native geometry are deliberately not kept
    # alive across hundreds of independent projects in one Python process.
    groups=[nodes[i:i+16] for i in range(0,len(nodes),16)]+[[matrix]]
    old_root=OUT/'release_checks'
    root=OUT/'packaged_checks';root.mkdir(exist_ok=True)
    receipts=[]
    for i,group in enumerate(groups,1):
        name=f'part_{i:02d}'
        part_root=old_root if i<=15 else root
        executed_runtime=fix['parent_runtime'] if i<=15 else fix['runtime']
        xml=part_root/(name+'.xml');log=part_root/(name+'.log')
        if xml.exists():
            suite=ET.parse(xml).getroot().find('testsuite')
            assert all(int(suite.attrib[k])==0 for k in ('failures','errors','skipped'))
            assert (part_root/(name+'.exit.json')).exists(), 'Completed process receipt required'
            receipt=json.loads((part_root/(name+'.exit.json')).read_text())
            assert receipt['exit_code']==0 and receipt['runtime']==executed_runtime
        else:
            assert i>15
            env=dict(os.environ,BRAIDS_COLLAPSE_CHECK='packaged_checks/'+name,
                BRAIDS_BANK_MATRIX_ROOT=str(old_root))
            with log.open('x',encoding='utf8') as stream:
                result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'desktop/check_banks224.py'),*group],
                    cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
            save(root/(name+'.exit.json'),dict(exit_code=result.returncode,runtime=sha(TARGET/'RUNTIME.json')))
            if result.returncode:raise RuntimeError(f'{name} failed; inspect {log}')
            suite=ET.parse(xml).getroot().find('testsuite')
            assert all(int(suite.attrib[k])==0 for k in ('failures','errors','skipped'))
        receipts.append(dict(part=name,tests=int(suite.attrib['tests']),xml_sha256=sha(xml),log_sha256=sha(log),
            path=str(part_root),executed_runtime=executed_runtime,
            carried_over_unchanged_operators=i<=15))
        print(f'{name}: {suite.attrib["tests"]} passed ({i}/{len(groups)} processes)',flush=True)
    suites=ET.Element('testsuites');combined=ET.SubElement(suites,'testsuite',name='Banks224 isolated release qualification',
        tests=str(sum(r['tests'] for r in receipts)),failures='0',errors='0',skipped='0',time='0')
    for r in receipts:
        from pathlib import Path
        for case in ET.parse(Path(r['path'])/(r['part']+'.xml')).getroot().find('testsuite').findall('testcase'):
            combined.append(deepcopy(case))
    ET.ElementTree(suites).write(OUT/'qualification_release.xml',encoding='utf8',xml_declaration=True)
    save(OUT/'QUALIFICATION_PARTS.json',dict(runtime=sha(TARGET/'RUNTIME.json'),parts=receipts,
        additive_binding_fix=fix,regression_runtime=fix['parent_runtime'],
        regression_carryover='All existing source, binaries and loader bindings are byte-identical; only the previously absent bank_counts CPU binding is added. Affected AMR and remaining checks run on the final package.'))
    print('Isolated release qualification complete',flush=True)

if __name__=='__main__':main()
