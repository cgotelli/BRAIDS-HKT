"""One mesh-epoch viewer for live and completed unified river runs."""
import json
from pathlib import Path

import numpy as np
from braids_core.windows_paths import filesystem_path
from qgis.PyQt.QtCore import Qt, QTimer, QVariant
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel,
    QPushButton, QSlider, QVBoxLayout, QWidget)
from qgis.core import (QgsCoordinateReferenceSystem, QgsFeature, QgsField, QgsFillSymbol,
    QgsGeometry, QgsGraduatedSymbolRenderer, QgsPointXY, QgsProject, QgsRendererRange,
    QgsStyle, QgsVectorLayer)

from ..core.adaptive_results import read_frame, read_index
from .river_physics import combo, section


VARIABLES = (("Water depth", "depth_m"), ("Flow speed", "velocity_m_s"),
             ("Water surface", "stage_m"), ("Bed elevation", "bed_m"))


class RiverResults(QWidget):
    def __init__(self, parent=None, iface=None):
        super().__init__(parent)
        self.iface, self.output, self.layer = iface, None, None
        self._epoch = None
        self._geometry = {}
        self.mesh_layer = None
        self._displayed_mesh = None
        layout = QVBoxLayout(self)
        button = QPushButton("Load results…")
        button.clicked.connect(self.choose_output)
        layout.addWidget(button)
        self.variable = combo(VARIABLES)
        layout.addWidget(self.variable)
        self.mesh = QCheckBox("Show computational mesh")
        self.mesh.setToolTip("Keep the first saved computational mesh fixed while viewing results.")
        layout.addWidget(self.mesh)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        layout.addWidget(self.slider)
        self.caption = QLabel("No results loaded")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        self.animation_box, form = section(layout, "Animation")
        self.animate_mesh = QCheckBox("Animate adaptive mesh")
        form.addRow(self.animate_mesh)
        self.play = QPushButton("Play")
        self.play.clicked.connect(self.toggle_animation)
        form.addRow(self.play)
        note = QLabel("Mesh epochs are stored with the results and reused when toggled.")
        note.setWordWrap(True)
        form.addRow(note)
        layout.addStretch()
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(lambda: self.slider.setValue(
            (self.slider.value()+1) % (self.slider.maximum()+1)))
        self.slider.valueChanged.connect(self.show_frame)
        self.variable.currentIndexChanged.connect(self.show_frame)
        self.mesh.toggled.connect(lambda on: self._mesh_changed(self.animate_mesh, on))
        self.animate_mesh.toggled.connect(lambda on: self._mesh_changed(self.mesh, on))

    def _mesh_changed(self, other, checked):
        if checked:
            other.setChecked(False)
        self.show_frame()

    def choose_output(self):
        name = QFileDialog.getExistingDirectory(self, "Unified BRAIDS results")
        if name:
            try:
                self.load_output(name)
            except Exception as exc:
                self.caption.setText(str(exc))

    def load_output(self, output, *, latest=True):
        output = filesystem_path(output).resolve()
        context = json.loads((output/"RUN_CONTEXT.json").read_text(encoding="utf-8"))
        if context.get("schema") != "braids.river-results.v1":
            raise ValueError("Select results from the new unified solver")
        if "project_file" in context:
            # Folder-managed runs carry their immutable parameter snapshot.
            # Never show changed parameters as the provenance of valid frames.
            from braids_core.river_project import load_project, identity
            from braids_core.sediment.river_leaf_io import sha
            if (context["project_file"] != "PROJECT.json" or
                    not (output/"PROJECT.json").is_file() or
                    sha(output/"PROJECT.json") != context.get("project_sha256")):
                raise ValueError("Saved run parameters are missing or changed")
            project = load_project(output/"PROJECT.json")
            if identity(project) != context.get("project_id") or project["model"] != context.get("model"):
                raise ValueError("Saved run parameters differ from the results identity")
        # Fetch only the publication list here. show_frame authenticates and
        # decodes the selected solution once, including when latest=False.
        doc = read_index(output)
        if self.output != output:
            self._epoch = None
            self._displayed_mesh = None
            self._geometry.clear()
        self.output = output
        self.crs = context["crs"]
        self.slider.blockSignals(True)
        self.slider.setMaximum(len(doc["frames"])-1)
        if latest:
            self.slider.setValue(self.slider.maximum())
        self.slider.blockSignals(False)
        self.show_frame()

    def toggle_animation(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play.setText("Play")
        elif self.output is not None:
            self.timer.start()
            self.play.setText("Pause")

    def show_frame(self, *_):
        if self.output is None:
            return
        try:
            doc, row, mesh, values = read_frame(self.output, self.slider.value())
            key = self.variable.currentData()
            v = values[key]
            if self.layer is None or QgsProject.instance().mapLayer(self.layer.id()) is None:
                self.layer = QgsVectorLayer("Polygon", "BRAIDS results", "memory")
                self.layer.dataProvider().addAttributes([QgsField("value", QVariant.Double)])
                self.layer.updateFields()
                QgsProject.instance().addMapLayer(self.layer)
                self._epoch = None
            self.layer.setCrs(QgsCoordinateReferenceSystem(self.crs))
            epoch = row["epoch"]
            provider = self.layer.dataProvider()
            if epoch != self._epoch:
                if epoch not in self._geometry:
                    self._geometry[epoch] = [QgsGeometry.fromPolygonXY([[QgsPointXY(float(x), float(y))
                        for x, y in mesh["vertices"][mesh["offsets"][i]:mesh["offsets"][i+1]]]])
                        for i in range(len(v))]
                features = []
                for geometry, value in zip(self._geometry[epoch], v):
                    f = QgsFeature(self.layer.fields())
                    f.setGeometry(geometry)
                    f.setAttributes([float(value)])
                    features.append(f)
                provider.truncate()
                okay, added = provider.addFeatures(features)
                if not okay:
                    raise ValueError("Could not publish result mesh")
                self._fids = [f.id() for f in added]
                self._epoch = epoch
                self.layer.updateExtents()
            else:
                provider.changeAttributeValues({fid: {0: float(value)} for fid, value in zip(self._fids, v)})
            lo = .001 if key == "depth_m" else 0. if key == "velocity_m_s" else float(np.min(v))
            hi = max(lo+1e-9, float(np.max(v)))
            ramp = QgsStyle.defaultStyle().colorRamp(dict(
                depth_m="Blues", velocity_m_s="Reds", stage_m="Viridis", bed_m="BrBG")[key])
            ranges = []
            for j in range(9):
                a, b = lo+(hi-lo)*j/9, lo+(hi-lo)*(j+1)/9
                color = ramp.color((j+.5)/9) if ramp else QColor.fromHsvF(.6-j*.06, .6, .8)
                symbol = QgsFillSymbol.createSimple({"color": color.name(), "outline_style": "no"})
                ranges.append(QgsRendererRange(a, b, symbol, f"{a:.4g} – {b:.4g}"))
            if key == "depth_m":
                symbol = QgsFillSymbol.createSimple({"color": "0,0,0,0", "outline_style": "no"})
                ranges.insert(0, QgsRendererRange(-1e100, np.nextafter(.001, -np.inf), symbol, "Dry / < 1 mm"))
            self.layer.setRenderer(QgsGraduatedSymbolRenderer("value", ranges))
            self.layer.setName("BRAIDS "+self.variable.currentText()+" — "+self.output.name)
            self.layer.triggerRepaint()
            self.show_mesh(doc, row)
            self.caption.setText(f"t = {row['time_s']:.6g} s · {len(v)} cells · "
                                 f"frame {self.slider.value()+1}/{len(doc['frames'])}")
            if self.iface:
                self.iface.mapCanvas().refresh()
        except Exception as exc:
            self.caption.setText("Results error: "+str(exc))
            return False
        return True

    def show_mesh(self, doc, row):
        show = self.mesh.isChecked() or self.animate_mesh.isChecked()
        if not show:
            if self.mesh_layer is not None:
                node = QgsProject.instance().layerTreeRoot().findLayer(self.mesh_layer.id())
                if node:
                    node.setItemVisibilityChecked(False)
            return
        epoch = row["epoch"] if self.animate_mesh.isChecked() else doc["frames"][0]["epoch"]
        if epoch not in self._geometry:
            _, _, mesh, values = read_frame(self.output, 0)
            self._geometry[epoch] = [QgsGeometry.fromPolygonXY([[QgsPointXY(float(x), float(y))
                for x, y in mesh["vertices"][mesh["offsets"][i]:mesh["offsets"][i+1]]]])
                for i in range(len(values["depth_m"]))]
        if self.mesh_layer is None or not QgsProject.instance().mapLayer(self.mesh_layer.id()):
            self.mesh_layer = QgsVectorLayer("Polygon", "BRAIDS computational mesh", "memory")
            self.mesh_layer.renderer().setSymbol(QgsFillSymbol.createSimple({
                "color": "0,0,0,0", "outline_style": "solid", "outline_color": "#555555", "outline_width": ".12"}))
            QgsProject.instance().addMapLayer(self.mesh_layer)
            self._displayed_mesh = None
        self.mesh_layer.setCrs(QgsCoordinateReferenceSystem(self.crs))
        if self._displayed_mesh != epoch:
            features = []
            for geometry in self._geometry[epoch]:
                f = QgsFeature()
                f.setGeometry(geometry)
                features.append(f)
            provider = self.mesh_layer.dataProvider()
            provider.truncate()
            provider.addFeatures(features)
            self.mesh_layer.updateExtents()
            self._displayed_mesh = epoch
        title = "Adaptive mesh" if self.animate_mesh.isChecked() else "Initial computational mesh"
        self.mesh_layer.setName("BRAIDS "+title+" — "+self.output.name)
        QgsProject.instance().layerTreeRoot().findLayer(self.mesh_layer.id()).setItemVisibilityChecked(True)
        self.mesh_layer.triggerRepaint()
