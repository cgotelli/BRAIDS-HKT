"""Review saved short runs and render the actual desktop forms, without rerunning."""
import os
import sys
import json
from io import BytesIO
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT)]
os.environ['QT_QPA_PLATFORM']='offscreen'
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase
from braids_desktop._avac.theme import apply_theme
from braids_desktop.forms import SettingsPanel
from braids_desktop.services import load_project_folder


def main():
    root=Path(sys.argv[1]).resolve();app=QApplication([])
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf');apply_theme(app,'light')
    doc=load_project_folder(root/'combined_cpu');form=SettingsPanel();form.load_document(doc)
    form.resize(420,850);form.show()
    for step,tabs in (('Sediment transport',form.sediment_tabs),('Morpho evolution',form.morphology_tabs),('Sediment transport BC',None)):
        form.set_step(step)
        for index in range(tabs.count() if tabs else 1):
            if tabs:tabs.setCurrentIndex(index)
            app.processEvents();app.processEvents()
            name=step+('_'+tabs.tabText(index) if tabs else '')
            form.grab().save(str(root/(name.replace(' ','_')+'_review.png')))
            assert form.stack.currentWidget().horizontalScrollBar().maximum()==0,name
    review={}
    for folder in root.iterdir():
        path=folder/'check_run/internal/checkpoint.npz'
        if not path.is_file():continue
        with np.load(path) as outer:
            with np.load(BytesIO(outer['payload'].tobytes())) as adaptive:
                with np.load(BytesIO(adaptive['owner'].tobytes())) as columns:
                    with np.load(BytesIO(columns['river'].tobytes())) as state:
                        fields={name:float(np.max(np.abs(state[name]))) for name in
                            ('class_flux','class_boundary_ledger','suspended_exchange_cumulative','class_bank_gross','class_collapse_gross')}
        if folder.name.startswith(('suspended','collapse')):
            assert fields['class_flux']==fields['class_boundary_ledger']==0.
        review[folder.name]=fields
    (root/'CHANNEL_REVIEW.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
    form.close();print(json.dumps(review,indent=2))


if __name__=='__main__':main()
