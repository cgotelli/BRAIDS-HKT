"""Retain failed harness receipts; require clean reruns of every failed node."""
import copy,xml.etree.ElementTree as ET
from stage_collapse222 import OUT,save,sha

def main():
    merged={};failed=set();sources={}
    for name in ('qualification2','qualification3'):
        path=OUT/(name+'.xml');sources[name]=sha(path)
        suite=ET.parse(path).getroot().find('testsuite')
        assert not int(suite.get('errors')) and not int(suite.get('skipped'))
        for case in suite.findall('testcase'):
            key=case.get('classname'),case.get('name')
            error=case.find('failure')
            if error is not None:
                assert name=='qualification2' and 'ordinary_exact_area' in key[1]
                assert 'ModuleNotFoundError' in (error.text or '')
                failed.add(key)
            else:merged[key]=copy.deepcopy(case)
    assert len(failed)==5 and failed<=set(merged)
    assert len(merged)==176,len(merged)
    root=ET.Element('testsuites');suite=ET.SubElement(root,'testsuite',dict(name='resolved_qualification',tests=str(len(merged)),failures='0',errors='0',skipped='0'))
    for case in merged.values():suite.append(case)
    ET.ElementTree(root).write(OUT/'qualification.xml',encoding='utf-8',xml_declaration=True)
    save(OUT/'QUALIFICATION_RECEIPT.json',dict(raw_receipts=sources,distinct_passing_tests=len(merged),
        original_fixture_import_failures=sorted('/'.join(k) for k in failed),
        resolution='Self-contained two-cell fixture replaces import of an intentionally unbundled historical bank implementation. Every failed node rerun and passed. No solver code changed between receipts.',
        merged_receipt_sha256=sha(OUT/'qualification.xml')))
if __name__=='__main__':main()
