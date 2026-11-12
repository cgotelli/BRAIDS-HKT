"""Publish installed receipts only after real launch, restart and editor checks."""
import json,shutil
from stage_supplied220 import ROOT,OUT,BASE,TARGET,REPORT,sha,save,verify

def main():
    verify(BASE);runtime=sha(BASE/'RUNTIME.json');assert runtime==sha(TARGET/'RUNTIME.json')
    for name in ('INSTALLED_EDITOR.json','INSTALLED_LAUNCH.json'):
        receipt=json.loads((OUT/name).read_text());assert receipt['runtime']==runtime
        if name=='INSTALLED_LAUNCH.json':
            assert {r['backend'] for r in receipt['cases']}=={'cpu','gpu'}
            assert all(r['prepare'] and r['run'] and r['exact_restart'] for r in receipt['cases'])
            assert receipt['example_restart']['candidate_checkpoint_to_installed']
        else:assert all(r['roundtrip'] and r['tabs']==4 for r in receipt['cases'])
        shutil.copy2(OUT/name,REPORT/name)
    recovery=ROOT/'desktop/qualification/supplied220'
    shutil.copy2(recovery/'RECOVERY.json',REPORT/'RECOVERY.json')
    for name in ('COMPLETION.json','REGRESSION_COMPLETE.xml','INSTALL.json','INSTALLED_EDITOR.json','INSTALLED_LAUNCH.json'):
        shutil.copy2(REPORT/name,recovery/name)
    save(recovery/'CURRENT.json',dict(status='installed and qualified',version='0.1.0-preview44',runtime=runtime,
        report=str(REPORT/'REPORT.md'),project=str(ROOT/'desktop/examples/supplied220'),
        distinct_passing_checks=1158,retained_native_replays_not_rerun=3,
        restored_missing_source_files=817,historical_tree_fully_recovered=False,
        next_milestone='Choose measured noncohesive suspension benchmark together',
        immediate_next_step='User tests preview44 and supplied220 folder'))
    print('Installed qualification recorded:',recovery/'CURRENT.json')

if __name__=='__main__':main()
