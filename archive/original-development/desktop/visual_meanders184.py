"""Open both delivered projects in the actual offscreen Desktop shell."""
import time
from check_meanders184 import ROOT,RUNTIME,OUT,REPORT,save
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFontDatabase
from braids_desktop._avac.theme import apply_theme
from braids_desktop.shell import MainWindow


def main():
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(ROOT/'validation/_meanders184_qt_shell'))
    app=QApplication([]);app.setOrganizationName('BRAIDS-test184');app.setApplicationName('BRAIDS-test184')
    for font in ('segoeui.ttf','segoeuib.ttf','seguisb.ttf'):
        QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+font)
    apply_theme(app,'light');receipts={}
    for key in ('advanced_off','advanced_on'):
        window=MainWindow(runtime=RUNTIME);errors=[];window.error=lambda exc:errors.append(str(exc))
        window.show();window.open_path(OUT/key/(key+'.braids.json'),confirmed=True)
        def wait_until(test):
            deadline=time.perf_counter()+25
            while not test() and time.perf_counter()<deadline and not errors:
                app.processEvents();time.sleep(.01)
            assert not errors,errors
            assert test(),'Timed out opening Desktop project/results'
        wait_until(lambda:window.project_root==OUT/key and not window._tasks)
        assert not window.dirty
        assert window.forms.document()['initial']['stage_m']==1.6
        assert window.forms.secondary.isChecked()==(key=='advanced_on')
        assert window.prepared==OUT/key/'prepared/base'
        assert window._results_output==OUT/key/'runs/to_400'
        window.set_mode('analyze')
        wait_until(lambda:window.results.store is not None and not window._tasks)
        window.results.variable.setCurrentIndex(window.results.variable.findData('bed_change_m'))
        window.results.slider.setValue(window.results.slider.maximum())
        window.layers['Mesh'].setChecked(False)
        for _ in range(30):app.processEvents();time.sleep(.01)
        assert not errors,errors
        assert window.grab().save(str(REPORT/(key.upper()+'_DESKTOP.png')))
        receipts[key]=dict(opened=True,physics_preserved=True,prepared_restored=True,
            results_opened=True,last_time_s=float(window.results.store.times[-1]),errors=errors)
        window.dirty=False;window.close();app.processEvents()
    save(REPORT/'SHELL_CHECK.json',receipts)
    print(receipts)


if __name__=='__main__':main()
