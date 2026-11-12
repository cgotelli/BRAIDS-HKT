"""QGIS plugin entry point for BRAIDS."""
from pathlib import Path

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from .gui.river_dock import BraidsDockWidget


ICON_PATH = Path(__file__).resolve().parent / "BRAIDS_logo.png"


class BraidsQgisPlugin:
    def __init__(self,iface):self.iface=iface;self.action=None;self.dock=None;self.icon=QIcon(str(ICON_PATH))

    def initGui(self):  # noqa: N802
        self.action=QAction(self.icon,"BRAIDS",self.iface.mainWindow());self.action.setObjectName("braidsQgisAction")
        self.action.triggered.connect(self.show_dock);self.iface.addPluginToMenu("&BRAIDS",self.action);self.iface.addToolBarIcon(self.action)

    def unload(self):
        if self.dock is not None:
            self.dock.shutdown();self.iface.removeDockWidget(self.dock)
            self.dock.dispose_when_idle();self.dock=None
        if self.action is not None:
            self.iface.removePluginMenu("&BRAIDS",self.action);self.iface.removeToolBarIcon(self.action);self.action.deleteLater();self.action=None

    def show_dock(self):
        if self.dock is None:
            self.dock=BraidsDockWidget(self.iface.mainWindow(),self.iface);self.dock.setWindowIcon(self.icon);self.iface.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,self.dock)
        self.dock.show();self.dock.raise_()
