"""Standalone desktop entry; Qt is independent of QGIS and native solver state."""
from pathlib import Path
import argparse
import sys
from PySide6.QtWidgets import QApplication, QMessageBox
from ._avac.theme import apply_theme
from ._avac.i18n import configure
from .shell import MainWindow


def main(*, runtime):
    parser = argparse.ArgumentParser(description='BRAIDS Desktop')
    parser.add_argument('project', nargs='?', help='Current BRAIDS project folder or JSON')
    parser.add_argument('--smoke', action='store_true', help='Construct the UI and quit after event processing')
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setOrganizationName('BRAIDS'); app.setApplicationName('BRAIDS');app.setApplicationDisplayName('BRAIDS')
    if sys.platform=='win32':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('BRAIDS.Desktop')
    from PySide6.QtGui import QIcon
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parents[1]/'BRAIDS_logo.png')))
    configure(app); apply_theme(app)
    window = MainWindow(runtime=Path(runtime))
    def exception_hook(kind, value, tb):
        import traceback
        traceback.print_exception(kind, value, tb)
        window.error(value)
    sys.excepthook = exception_hook
    window.show()
    if args.project:
        from PySide6.QtCore import QTimer
        # A CLI project opens into a fresh window, which has no user work to
        # replace. Deferred widget signals must not trigger an empty-save prompt.
        QTimer.singleShot(0, lambda: window.open_path(Path(args.project),confirmed=True))
    if args.smoke:
        from PySide6.QtCore import QTimer
        def finish_smoke():
            if window._tasks:
                QTimer.singleShot(100,finish_smoke);return
            window.dirty=False
            app.quit()
        QTimer.singleShot(800, finish_smoke)
    return app.exec()
