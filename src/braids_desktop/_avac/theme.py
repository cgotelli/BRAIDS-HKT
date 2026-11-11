"""One consistent native Widgets theme."""
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

SURFACE_BLUE = "#456b8a"
SURFACE_HOVER = "#4b7394"
LEGEND_BLUE = "#41637e"


def apply_theme(app: QApplication, mode: str | None = None) -> None:
    if mode is None:
        from .preferences import appearance
        mode = appearance()
    if mode not in ('light','dark'):mode='light'
    # Do not inherit Windows' dark color scheme for a light BRAIDS session.
    app.styleHints().setColorScheme(Qt.ColorScheme.Light if mode=='light' else Qt.ColorScheme.Dark)
    app.setProperty("appearance", mode)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    palette = QPalette()
    for role, value in ((QPalette.Window, "#101722"), (QPalette.WindowText, "#e4edf5"),
                        (QPalette.Base, "#111b28"), (QPalette.AlternateBase, "#182332"),
                        (QPalette.Text, "#e4edf5"), (QPalette.Button, SURFACE_BLUE),
                        (QPalette.ButtonText, "#e4edf5"), (QPalette.Highlight, "#218dce"),
                        (QPalette.HighlightedText, "#ffffff"), (QPalette.ToolTipBase, "#24354a"),
                        (QPalette.ToolTipText, "#ffffff")):
        palette.setColor(role, QColor(value))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#6a7b8e"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#6a7b8e"))
    palette.setColor(QPalette.PlaceholderText, QColor("#8296aa"))
    if mode == "light":
        for role, value in ((QPalette.Window, "#ffffff"), (QPalette.WindowText, "#182f43"),
                            (QPalette.Base, "#ffffff"), (QPalette.AlternateBase, "#edf2f7"),
                            (QPalette.Text, "#182f43"), (QPalette.Button, SURFACE_BLUE),
                            (QPalette.ButtonText, "#ffffff"), (QPalette.PlaceholderText, "#667e91")):
            palette.setColor(role, QColor(value))
    app.setPalette(palette)
    stylesheet = """
    QMainWindow { background: #0e1621; border: 1px solid #808080; border-radius: 0px; }
    QFrame#TitleBar { background: #121f2d; color: #e4edf5; }
    QPushButton#WindowControl, QPushButton#CloseWindow { background: transparent; border: none; border-radius: 0; padding: 0; font-size: 17px; }
    QPushButton#WindowControl:hover { background: #4b7394; }
    QPushButton#CloseWindow:hover { background: #b83e4c; color: white; }
    QToolButton#MapControl { padding: 0; font-size: 21px; font-weight: 600; border-radius: 4px; }
    QToolButton#MapControl:checked { background: #356b94; color: #edf8ff; border-color: #72b6e1; }
    QToolTip { color: #edf6ff; background: #41637e; border: 1px solid #799ab4; padding: 8px; }
    QFrame#Header { background: #121f2d; border-bottom: 1px solid #2a3d50; }
    QLabel#Brand { font-size: 24px; font-weight: 700; letter-spacing: 1px; }
    QLabel#Eyebrow { color: #6db9e8; font-size: 10px; font-weight: 700; letter-spacing: 2px; }
    QLabel#Title { font-size: 19px; font-weight: 600; }
    QLabel#Muted { color: #8fa4b8; }
    QLabel#Status { color: #d3eaff; background: #345873; border: 1px solid #6385a0; border-radius: 5px; padding: 5px 10px; }
    QFrame#Panel { background: #131e2b; border: 1px solid #29384a; border-radius: 6px; }
    QFrame#Sidebar { background: #111c29; border-right: 1px solid #253547; }
    QPushButton, QToolButton { border: 1px solid #6f91ad; border-radius: 5px; background: #456b8a; padding: 7px 12px; }
    QPushButton:hover, QToolButton:hover { background: #4b7394; border-color: #8db4d1; }
    QPushButton:focus, QToolButton:focus { border: 1px solid #62ccff; }
    QPushButton:checked { background: #356b94; color: #edf8ff; border-color: #72b6e1; }
    QPushButton#Primary { background: #137ec0; color: white; border-color: #299ee3; font-weight: 600; padding: 9px 14px; }
    QPushButton#Primary:hover { background: #2696d7; }
    QPushButton#Primary:disabled { background: #304b63; color: #95aabd; border-color: #476780; }
    QPushButton:disabled { background: #304b63; color: #95aabd; border-color: #476780; }
    QListWidget { background: transparent; border: none; outline: none; padding: 7px; }
    QListWidget::item { padding: 13px 10px; border-radius: 5px; margin: 3px 0; }
    QListWidget::item:selected { background: #456f90; color: #f1f8ff; border-left: 3px solid #72c6f2; }
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTableWidget { background: #0f1925; border: 1px solid #34485b; border-radius: 4px; padding: 5px; selection-background-color: #22638f; }
    QComboBox::drop-down { border: none; width: 22px; }
    QGroupBox { border: 1px solid #2c4054; border-radius: 5px; margin-top: 15px; padding: 14px 10px 10px; font-weight: 600; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #83c8ed; padding: 0 4px; }
    QCheckBox { spacing: 8px; padding: 3px; }
    QSplitter::handle { background: #7291aa; }
    QSplitter::handle:horizontal { width: 3px; }
    QSplitter::handle:vertical { height: 3px; }
    QScrollArea { border: none; background: transparent; }
    QStatusBar { background: #101c29; color: #94aabd; border-top: 1px solid #2a3d50; }
    QMenuBar, QMenu { background: #121d2a; color: #cddbe7; }
    QMenuBar::item:selected, QMenu::item:selected { background: #4b7394; }
    QTabWidget::pane { border: 1px solid #293b4d; }
    QTabBar::tab { background: #172534; color: #96adbf; padding: 6px 15px; }
    QTabBar::tab:selected { background: #456f90; color: #f1f8ff; }
    QSlider::groove:horizontal { height: 4px; background: #34495e; }
    QSlider::handle:horizontal { width: 13px; background: #58c5fa; border-radius: 6px; margin: -5px 0; }
    QPushButton#Stop:enabled { background: #b93646; color: white; border-color: #d35462; }
    QPushButton#Stop:hover:enabled { background: #d13f50; }
    """
    if mode == "light":
        colors = {"#0e1621":"#ffffff", "#121f2d":"#ffffff", "#e4edf5":"#182f43",
                  "#111c29":"#ffffff", "#0f1925":"#ffffff", "#8fa4b8":"#4b667d",
                  "#6db9e8":"#22618b", "#83c8ed":"#22618b", "#2a3d50":"#c4d2df",
                  "#293b4d":"#c4d2df", "#2c4054":"#c4d2df", "#34485b":"#bccbd8",
                  "#121d2a":"#ffffff", "#cddbe7":"#182f43", "#101c29":"#e7eef5",
                  "#94aabd":"#36556c", "#172534":"#dde7f0", "#96adbf":"#36556c",
                  "#131e2b":"#ffffff", "#29384a":"#c4d2df", "#253547":"#c4d2df"}
        colors.update({"#304b63":"#dce6ee", "#95aabd":"#6b8295", "#476780":"#bdcedb",
                       "#345873":"#dceaf5", "#d3eaff":"#365b78", "#6385a0":"#b3cadd"})
        for old, new in colors.items():
            stylesheet = stylesheet.replace(old, new)
        stylesheet += """
        QPushButton, QToolButton { color: #f2f7fb; }
        QPushButton#WindowControl, QPushButton#CloseWindow { color: #182f43; }
        QMenu::item:selected, QMenuBar::item:selected { color: white; }
        """
    resources = Path(__file__).resolve().parent/"resources"
    up_arrow = (resources/"spin-up.svg").as_posix()
    down_arrow = (resources/"spin-down.svg").as_posix()
    stylesheet += """
    QSpinBox, QDoubleSpinBox { padding-right: 27px; }
    QSpinBox::up-button, QDoubleSpinBox::up-button {
        subcontrol-origin: border; subcontrol-position: top right; width: 22px;
        background: #456b8a; border: 1px solid #6f91ad; border-top-right-radius: 3px;
    }
    QSpinBox::down-button, QDoubleSpinBox::down-button {
        subcontrol-origin: border; subcontrol-position: bottom right; width: 22px;
        background: #456b8a; border: 1px solid #6f91ad; border-bottom-right-radius: 3px;
    }
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: #4b7394; }
    QSpinBox::up-button:disabled, QDoubleSpinBox::up-button:disabled,
    QSpinBox::down-button:disabled, QDoubleSpinBox::down-button:disabled { background: #7894a9; }
    """
    stylesheet += f"""
    QComboBox {{ padding-right: 30px; }}
    QComboBox::drop-down {{
        subcontrol-origin: border; subcontrol-position: top right; width: 24px;
        background: #456b8a; border-left: 1px solid #6f91ad;
        border-top-right-radius: 3px; border-bottom-right-radius: 3px;
    }}
    QComboBox::drop-down:hover {{ background: #4b7394; }}
    QComboBox::down-arrow {{ image: url("{down_arrow}"); width: 10px; height: 7px; }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url("{up_arrow}"); width: 10px; height: 7px; }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url("{down_arrow}"); width: 10px; height: 7px; }}
    """
    app.setStyleSheet(stylesheet)
