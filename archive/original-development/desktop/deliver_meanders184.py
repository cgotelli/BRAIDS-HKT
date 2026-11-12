"""Persist ordinary Desktop editor/session state and verify project round trips."""
from copy import deepcopy
import json
from check_meanders184 import ROOT,RUNTIME,OUT,REPORT,save
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from braids_desktop import services,project_file,workspace,editor_state
from braids_desktop.forms import SettingsPanel
from braids_core.river_project import identity


def main():
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(ROOT/'validation/_meanders184_qt'))
    app=QApplication.instance() or QApplication([])
    receipts={}
    for key in ('advanced_off','advanced_on'):
        root=OUT/key;doc=services.load_project_folder(root)
        panel=SettingsPanel();panel.load_document(doc)
        # This example has an explicit wet initial state in the current JSON.
        # The Desktop draft stores its initial-stage value separately from the
        # default dry-start editor. Preserve it in the ordinary draft snapshot.
        panel._dry_stage_m=doc['initial']['stage_m']
        check=panel.document();check.pop('editor_state',None)
        original=deepcopy(doc);original.pop('editor_state',None)
        assert identity(check)==identity(original),'UI round trip changed physical settings'
        assert panel.secondary.isChecked()==(key=='advanced_on')
        assert panel.turbulence.isChecked()==(key=='advanced_on')
        assert panel.physics.capacity.currentData()=='capacity.smart_jaggi'
        ui=editor_state.capture(panel,root)
        session=services.project_session(root)
        session['desktop']=dict(advanced_hydrodynamics=key=='advanced_on',live_results=True,
                                model_step='Sediment transport',result_step='Probes')
        from braids_core import river_folder
        river_folder.update_session(root,session)
        workspace.save_editor(root,doc,ui,session,has_domain=True,has_terrain=True)
        project_file.write(project_file.default_file(root),root,doc)
        folder,loaded,draft,changed=project_file.read(project_file.default_file(root))
        assert folder==root and not changed and draft is not None
        assert identity(loaded)==identity(doc)
        workspace.check_runnable(root,json.loads((root/'braids_project.json').read_text()))
        history=river_folder.list_history(root)
        assert len(history)>=5
        from braids_core.river_launcher import verify
        verify()
        panel.close()
        receipts[key]=dict(entry=str(project_file.default_file(root)),same_physics=True,
            editor_reopened_without_changes=True,history_records=len(history),runtime_verified=True,
            results=session['output'],prepared=session['prepared'])
    save(REPORT/'DELIVERY.json',receipts)
    print(json.dumps(receipts,indent=2))


if __name__=='__main__':main()
