"""Explicit, embedded device data for the qualified Dou power workflow."""
from copy import deepcopy
import json
from pathlib import Path
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget,QVBoxLayout,QLabel,QPushButton,QFileDialog,QMessageBox
from braids_core.sediment.river_turbine_device_yaw import validate


class DeviceYawInput(QWidget):
    changed=Signal()
    def __init__(self,parent=None):
        super().__init__(parent);self._value=None
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        self.import_button=QPushButton('Import device yaw-performance JSON');layout.addWidget(self.import_button)
        self.clear_button=QPushButton('Clear device yaw data');layout.addWidget(self.clear_button)
        self.summary=QLabel();self.summary.setWordWrap(True);layout.addWidget(self.summary)
        self.import_button.clicked.connect(self._import);self.clear_button.clicked.connect(self._clear)
        self.set_value(None)

    def value(self):return deepcopy(self._value)

    def set_value(self,value):
        self._value=deepcopy(value)
        self.clear_button.setEnabled(value is not None)
        if value is None:
            self.summary.setText('No device yaw data. Import ratios P(yaw)/P(0) for the actual rotor and controller. The data are saved inside the project; no external file is needed afterwards.')
            return
        self.summary.setText(f"{value['model_name']}\nDiameter {value['diameter_m']:g} m; fixed RPM {value['rpm']:g}. "
            f"Incoming speed {value['speed_m_s'][0]:g}-{value['speed_m_s'][-1]:g} m/s; |yaw| 0-{value['yaw_deg'][-1]:g} degrees.\n"
            f"Source: {value['provenance']}\nScope: {value['physical_scope']}\n"
            'One shared device/controller table applies to every fixed-axis rotor. It must match their diameter and zero-yaw Ct/Cp data. Below the supported speed, rotors stop; unsupported high speeds or yaw angles remain explicit errors.')

    def import_file(self,path):
        value=validate(json.loads(Path(path).read_text(encoding='utf-8-sig')))
        self.set_value(value);self.changed.emit()

    def _import(self):
        path,_=QFileDialog.getOpenFileName(self,'Import matching device yaw data','','JSON (*.json)')
        if not path:return
        try:self.import_file(path)
        except (ValueError,TypeError,KeyError,OSError) as exc:
            QMessageBox.warning(self,'Device yaw data',str(exc)+'\nThe existing device data were kept. Import a complete matching device table.')

    def _clear(self):
        self.set_value(None);self.changed.emit()
