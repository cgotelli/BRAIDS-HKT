"""New current-schema paired examples, without migrating any user project."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_wall188';TARGET=OUT/'desktop_candidate'
os.environ['QT_QPA_PLATFORM']='offscreen';sys.path[:0]=[str(TARGET),str(ROOT)]
from stage_secondary185 import save,sha


def main():
    from PySide6.QtWidgets import QApplication
    from braids_desktop import services,project_file
    from braids_desktop.forms import SettingsPanel
    app=QApplication.instance() or QApplication([])
    base=services.load_project_folder(OUT/'release/wet_prescribed_gpu/project')
    rows={};destination=ROOT/'desktop/examples/two_meanders188'
    assert not destination.exists()
    for enabled,label in ((True,'advanced_on'),(False,'advanced_off')):
        panel=SettingsPanel();panel.load_document(base)
        panel.secondary.setChecked(enabled);panel.turbulence.setChecked(enabled)
        panel.name.setText('Two-meander Smart & Jäggi — advanced '+('ON' if enabled else 'OFF'))
        # Retain the qualified 200 s warmup, 400 s requested end and MF5. The
        # two new examples differ only in the explicitly labelled flow bundle.
        doc=panel.document();panel.close();doc.pop('editor_state',None)
        assert doc['model']['physics']['sediment']['bed_slope']=='bed_slope.off'
        folder=destination/label;saved=services.save_project_folder(folder,doc)
        path=project_file.default_file(folder);project_file.write(path,folder,saved)
        services.load_project_folder(folder)
        rows[label]=dict(project=str(path),sha256=sha(folder/'braids_project.json'),
            backend='gpu',new_configuration=True,simulated_to_equilibrium=False)
    save(OUT/'EXAMPLES.json',rows);print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
