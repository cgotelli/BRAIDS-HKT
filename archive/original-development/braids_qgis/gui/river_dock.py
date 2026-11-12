"""Tab-based UI: one project, one numerical method, one worker lifecycle."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
from uuid import uuid4

from qgis.PyQt.QtCore import Qt
from qgis.PyQt import sip
from qgis.PyQt.QtWidgets import (QApplication, QMessageBox, QProgressDialog, QCheckBox, QComboBox, QDockWidget, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget, QHeaderView)
from qgis.core import (QgsMapLayerProxyModel, QgsProject, QgsRasterLayer, QgsVectorLayer,
    QgsVectorFileWriter)
from qgis.gui import QgsMapLayerComboBox

from braids_core.boundaries.domain import PolygonDomain, BoundaryAssignment
from braids_core.boundaries.inlet_io import NativeGrid
from braids_core.river_project import (SCHEMA, SLOPE_PARAMETERS, BEDFORM_METHODS,
    load_project, validate_project, read_json, identity, checkpoint_editor_state)
from braids_core.river_folder import (save_folder, load_folder, folder_info,
    list_history, record_event, update_session)
from braids_core.windows_paths import filesystem_path
from braids_core.river_model import SURVEY_BANKS
from braids_core.sediment.current_river_solver import CurrentRiverControls
from braids_core.sediment.bed_columns_v101 import ColumnMethodsV101
from braids_core.sediment.river_field_policy import FieldSettings
from braids_gpu.atomic_output import write_json_atomic
from ..core.gpu_discovery import discover_gpu_pythons
from ..core.river_runner import RiverRunner
from .river_physics import RiverPhysicsEditor, combo, number, section, set_choice
from .river_geometry import BoundaryEdges, domain_from_layer, layer_from_domain
from .river_results import RiverResults, VARIABLES
from .river_forcing import RiverForcingEditor
from .river_boundaries import RiverBoundaryEditor
from .river_setup_tools import RiverSetupTools


def button(layout, title, callback):
    result = QPushButton(title)
    result.clicked.connect(callback)
    layout.addWidget(result)
    return result


class BraidsDockWidget(QDockWidget):
    def __init__(self, parent=None, iface=None, *, development_runtime=None):
        super().__init__("BRAIDS", parent)
        self.setObjectName("BraidsUnifiedRiverDock")
        self.iface = iface
        self.runtime = (Path(development_runtime).resolve() if development_runtime else
                        Path(__file__).resolve().parents[1]/"river_runtime")
        self.runner = RiverRunner(self, development_runtime=development_runtime is not None)
        self.edges = BoundaryEdges()
        self.prepared = self.output = None
        self._loading = True
        self._loaded_project = None
        self._dirty = True
        self._grid_override = None
        self._geometry_review = None
        self._dispose = False
        self.project_path = None
        self.project_root = None
        self._active_event = None
        content = QWidget()
        outer = QVBoxLayout(content)
        toolbar = QHBoxLayout()
        self.save_button = button(toolbar, "Save project folder", self.save_dialog)
        self.load_button = button(toolbar, "Open project folder…", self.load_dialog)
        outer.addLayout(toolbar)
        folder = QHBoxLayout()
        folder.addWidget(QLabel('Project folder'))
        self.workspace = QLineEdit(str(Path.home()/"Documents/BRAIDS"))
        self.workspace.setToolTip('One folder contains the saved model, copied inputs, preparation, runs, results and history. Open this folder to continue later.')
        folder.addWidget(self.workspace,1)
        self.folder_button = button(folder,'Choose folder…',self.choose_project_folder)
        outer.addLayout(folder)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)
        self.setWidget(content)
        self.general = self.page("General")
        form = QFormLayout()
        self.general_form = form
        self.general.addLayout(form)
        self.name = QLineEdit("River")
        self.domain = QgsMapLayerComboBox()
        self.domain.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        self.domain.setAllowEmptyLayer(True);self.domain.setLayer(None)
        self.dem = QgsMapLayerComboBox()
        self.dem.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.backend = combo([("CPU / OpenMP", "cpu"), ("GPU / CUDA", "gpu")])
        self.python = QLineEdit()
        self.mode = combo([('Hydrodynamics only','hydro'),('Hydrodynamics + sediment transport','sediment')])
        self.roughness = combo([("Chezy", "roughness.chezy"), ("Manning", "roughness.manning"),
                                ("White–Colebrook", "roughness.white_colebrook")])
        self.chezy = number(45., 1.e-6, 1e4, 6)
        self.manning = number(.03, .0001, 10., 8)
        self.nikuradse = number(.003, .000001, 1e3, 8)
        self.white_relation = combo([("FLOW log layer (ChzK2d=false)", "flow_log_layer"),
                                     ("Classical 2D (ChzK2d=true)", "classical_2d")])
        self.secondary = QCheckBox("Enable secondary flow")
        self.betac = number(1., 0., 1., 6)
        self.secondary_diffusion = number(0.)
        self.turbulence = QCheckBox("Enable horizontal turbulent viscosity")
        self.viscosity = number(0.)
        self.setup = RiverSetupTools(self)
        outer.insertWidget(2, self.setup.session_widget)
        for label, field in (("Project name", self.name), ('Simulation mode',self.mode), ("Domain polygon", self.setup.domain_field()),
            ("Terrain DEM", self.setup.dem_field()),
            ("Inlet guides (drafting only)", self.setup.guide_field()), ("", self.setup.snap),
            ("Compute backend", self.backend),
            ("Hydraulic roughness", self.roughness), ("Chezy C (m½/s)", self.chezy),
            ("Manning n (s/m⅓)", self.manning), ("Roughness height ks (m)", self.nikuradse),
            ("White–Colebrook relation", self.white_relation)):
            form.addRow(label, field)
        self.advanced,advanced = section(self.general,'Advanced hydraulic and runtime settings')
        advanced.addRow('External Python',self.setup.field(self.python,[('Browse…',lambda:self.setup.browse(self.python,'Python executable')),
            ('Find automatically',self.discover)]))
        for label,field in (('',self.secondary),('Secondary momentum coefficient',self.betac),
                ('Secondary diffusivity (m²/s)',self.secondary_diffusion),('',self.turbulence),('Eddy viscosity (m²/s)',self.viscosity)):
            advanced.addRow(label,field)
        note = QLabel('Choose the project folder above, then set the domain/DEM, boundary conditions and mesh. '
                      'Use Run to prepare and start. Hydrodynamics only keeps the bed fixed and disables all sediment transport. '
                      'CPU and GPU use the same explicit SSPRK22 f-wave solver.')
        note.setWordWrap(True)
        self.general.addWidget(note)
        self.general.addStretch()

        bc = self.page("BC & IC")
        button(bc, "Create / replace boundary faces", self.create_edges)
        bc.addWidget(self.setup.row([('Select faces', lambda:self.setup.digitizing.select(self.edges.layer)),
            ('Zoom to faces', lambda:self.setup.digitizing.zoom(self.edges.layer)),
            ('Clear named boundary', self.clear_boundary)]))
        note = QLabel('Edit the domain polygon to change the exterior shape, then recreate boundary faces. '
            'Creating faces replaces the previous generated layer and clears its assignments. '
            'Click a boundary below to select its faces; reapplying its name replaces the old assignment.')
        note.setWordWrap(True);bc.addWidget(note)
        form = QFormLayout()
        self.bc_kind = combo([("Inlet", "inlet"), ("Outlet", "outlet"), ("Wall", "wall")])
        self.bc_id = QLineEdit("inlet")
        form.addRow("Boundary type", self.bc_kind)
        form.addRow("Boundary name", self.bc_id)
        bc.addLayout(form)
        button(bc, "Apply to selected faces", self.apply_boundary)
        self.boundaries = QTableWidget(0, 2)
        self.boundaries.setHorizontalHeaderLabels(["Assigned boundary", "Type"])
        self.boundaries.setFixedHeight(115)
        self.boundaries.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        bc.addWidget(self.boundaries)
        self.boundary_editor = RiverBoundaryEditor(self)
        bc.addWidget(self.boundary_editor)
        self.boundaries.cellClicked.connect(self.select_boundary)
        form = QFormLayout()
        self.stage = number(0., -1e7, 1e7)
        self.u = number(0., -1e4, 1e4)
        self.v = number(0., -1e4, 1e4)
        form.addRow("Initial water-surface elevation (m)", self.stage)
        form.addRow("Initial east velocity (m/s)", self.u)
        form.addRow("Initial north velocity (m/s)", self.v)
        bc.addLayout(form)
        self.boundary_note = QLabel("Boundary methods above use the same CPU/GPU owner. "
            "Sediment flux forcing below is separate from exterior bed/composition; it is uniform along each named boundary and specified per fraction.")
        self.boundary_note.setWordWrap(True)
        bc.addWidget(self.boundary_note)
        self.forcing = RiverForcingEditor(self)
        bc.addWidget(self.forcing)
        bc.addStretch()

        mesh = self.page("Mesh")
        form = QFormLayout()
        self.minimum_size = number(5., .0001, 1e6, 6)
        self.levels = QSpinBox()
        self.levels.setRange(1, 6)
        self.levels.setValue(3)
        self.mesh_mode = QComboBox()
        self.mesh_mode.addItem("Dynamic AMR — water depth only", "dynamic_depth")
        self.mesh_mode.addItem("Static mesh — terrain slope breaks only", "static_terrain")
        form.addRow("Smallest cell size (m)", self.minimum_size)
        form.addRow("Number of 2:1 levels", self.levels)
        form.addRow("Mesh strategy", self.mesh_mode)
        mesh.addLayout(form)
        label = QLabel("Dynamic: refine wet cells and their front buffer; no slope criterion. Static: refine initial DEM slope breaks once, including dry levees; no runtime AMR assessments or remeshing. Only inlets/outlets and explicitly surveyed bank corridors stay finest in both modes. Walls do not force refinement; they follow the selected depth/terrain criterion and 2:1 grading. Static cells may remain coarse even when wet. Cropped wall cells can be smaller than the nominal selected size. The hydraulic timestep is controlled separately.")
        label.setWordWrap(True)
        mesh.addWidget(label)
        self.align_button=button(mesh, "Align inlet/outlet sections…", self.align_domain)
        self.align_button.setToolTip('Review alignment to the actual finest grid lines, not merely the coordinate axes. Repeat after changing cell size, levels or domain geometry. Review changed section widths, positions and area before accepting. Original domain is retained; nothing is snapped automatically during preparation. Walls remain cropped.')
        _, form = section(mesh, "Mesh refinement settings")
        self.adaptation = {}
        for key, value in asdict(FieldSettings()).items():
            if key == 'terrain_enabled':continue
            w = QSpinBox() if type(value) is int else number(value, 1e-9, 1e7)
            if type(value) is int:
                w.setRange(1, 10000)
                w.setValue(value)
            self.adaptation[key] = w
            if key == 'interval':
                w.setToolTip("Physical seconds between mesh assessments, not the hydraulic timestep. Default: 1 s. You may use 2 s or more with an adequate wet-front buffer. Saved project values are retained.")
            elif key == 'coarsen_delay':
                w.setToolTip("Consecutive dry assessments before coarsening. With interval 1 s and delay 2, dry coarsening is delayed across two assessments.")
            elif key == 'buffer_cells':
                w.setToolTip("Face-neighbour rings protecting wet cells and open-boundary entrances. Enlarge this for faster wet fronts or longer assessment intervals.")
            elif key == 'terrain_angle_deg':
                w.setMaximum(89.999)
                w.setToolTip("Minimum directional change in terrain slope. A constant plane, even a steep one, does not trigger refinement. Both angle and vertical-detail thresholds must pass.")
            elif key == 'terrain_detail_m':
                w.setToolTip("Minimum resolved departure from a local plane, filtering small DEM noise. Features narrower than the selected finest spacing cannot be guaranteed resolved.")
            form.addRow({'interval':'Assessment interval (s)', 'coarsen_delay':'Dry assessments before coarsening', 'buffer_cells':'Wet-front buffer (neighbour rings)',
                'terrain_angle_deg':'Terrain slope change (degrees)', 'terrain_detail_m':'Terrain detail threshold (m)'}[key], w)
        def mesh_controls():
            dynamic=self.mesh_mode.currentData()=='dynamic_depth'
            for key,w in self.adaptation.items():
                w.setEnabled((not dynamic) if key.startswith('terrain_') else dynamic)
        self.mesh_mode.currentIndexChanged.connect(mesh_controls)
        mesh_controls()
        mesh.addStretch()

        self.physics = RiverPhysicsEditor(self)
        self.mode.currentIndexChanged.connect(lambda:self.physics.enabled.setChecked(self.mode.currentData()=='sediment'))
        self.setup.bind_bank(self.physics.bank_survey)
        self.physics.bank_survey.domain_crs=lambda:domain_from_layer(self.domain.currentLayer()).crs_wkt
        self.physics.bank_survey.preview_button.clicked.connect(self.show_bank_preview)
        self.add_page("Sediment", self.physics.sediment_page)
        self.add_page("Bed & banks", self.physics.bed_page)
        run = self.page("Run")
        form = QFormLayout()
        self.run_form = form
        self.end_time = number(60., .000001, 1e12)
        self.dt_max = number(2., .000001, 1e9)
        self.dt_max.setToolTip('Maximum outer transaction, not a guaranteed hydraulic step. The solver still substeps for stage CFL and positivity, and stops at output, forcing and AMR times. Saved project values are preserved.')
        self.output_interval = number(1., .000001, 1e12)
        self.output_interval.setToolTip('Save solution frames for viewing at these simulation times. This clock may shorten a numerical step to reach an exact output time.')
        self.checkpoint_interval = number(60., .000001, 1e12)
        self.checkpoint_interval.setToolTip('Save a restart checkpoint at the first accepted state at or after each interval in simulation seconds. This does not shorten numerical steps. Initial, final and cancelled accepted states are always saved. An abrupt crash can lose work since the last checkpoint.')
        self.checkpoint_duration = QCheckBox('Use simulation duration')
        self.checkpoint_duration.setChecked(True)
        self.checkpoint_duration.setToolTip('For new projects, the checkpoint interval follows End time. Initial, final and cancelled states are still saved. Uncheck to save periodic recovery checkpoints; the last custom interval is retained. With this default, a crash before the end may lose the entire run since its initial checkpoint.')
        self._checkpoint_custom_interval = self.checkpoint_interval.value()
        checkpoint_row = QWidget()
        checkpoint_layout = QHBoxLayout(checkpoint_row)
        checkpoint_layout.setContentsMargins(0,0,0,0)
        checkpoint_layout.addWidget(self.checkpoint_interval)
        checkpoint_layout.addWidget(self.checkpoint_duration)
        self.end_time.valueChanged.connect(self.checkpoint_end_changed)
        self.checkpoint_interval.valueChanged.connect(self.checkpoint_custom_changed)
        self.checkpoint_duration.toggled.connect(self.checkpoint_mode_changed)
        self.threads = QSpinBox()
        self.threads.setRange(1, max(256, os.cpu_count() or 1))
        self.threads.setValue(min(8, os.cpu_count() or 1))
        self.morfac = number(1., .001, 1e6)
        self.restart = QLineEdit()
        for label, field in (("End time (s)", self.end_time), ("Maximum outer step (s)", self.dt_max),
            ("Solution output interval (s)", self.output_interval), ("Restart checkpoint interval (s)", checkpoint_row), ("CPU threads", self.threads),
            ("Morphological factor", self.morfac), ("New-format checkpoint (optional)", self.restart)):
            form.addRow(label, self.setup.field(field, [('Browse…', lambda:self.setup.browse(self.restart,'checkpoint'))])
                if field is self.restart else field)
        run.addLayout(form)
        _,cfl_form=section(run, "Hydraulic timestep settings")
        self.cfl=number(1., .001, 1., 6)
        self.cfl.setToolTip('Explicit face-sum CFL: dt * sum(face length * wave speed) / actual cell area. Fresh pure-hydrodynamic default 1.0; sediment, secondary flow and turbulence require 0.5 or less. CFL 1 is faster but resolves impulsive wetting-front velocities less accurately. Both SSP stages enforce the limit; maximum timestep is only an upper limit. Saved values are preserved and incompatible combinations are rejected, never silently changed.')
        cfl_form.addRow("CFL (face-sum)",self.cfl)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(150)
        _,details = section(run,'Resolved methods / technical details')
        details.addRow(self.summary)
        self.admission = QLabel()
        self.admission.setWordWrap(True)
        run.addWidget(self.admission)
        self.preview = QCheckBox("Live preview")
        self.preview.setChecked(True)
        self.preview_variable = combo(VARIABLES)
        run.addWidget(self.preview)
        run.addWidget(self.preview_variable)
        actions = QHBoxLayout()
        self.prepare_button = button(actions, "Prepare Run", self.prepare)
        self.run_button = button(actions, "Run", self.run)
        self.stop_button = button(actions, "Stop", self.stop)
        run.addLayout(actions)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        run.addWidget(self.progress)
        self.status = QLabel("Create a new project; old projects cannot run.")
        self.status.setWordWrap(True)
        run.addWidget(self.status)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        run.addWidget(self.log)
        history_row = QHBoxLayout()
        self.history = QComboBox()
        self.history.setMinimumContentsLength(24)
        self.history.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.history.setToolTip('Select saved parameters, a preparation or a run. Runs with saved frames also open in Results; worker failures retain their logs.')
        history_row.addWidget(QLabel('Project history'))
        history_row.addWidget(self.history, 1)
        self.history_refresh = button(history_row, 'Refresh', self.refresh_history)
        run.addLayout(history_row)
        self.checkpoint_button = button(run, 'Use selected run checkpoint', self.use_history_checkpoint)
        self.checkpoint_button.setToolTip('Select a compatible checkpoint explicitly. Opening a folder never starts or resumes a run.')
        self.restore_parameters_button = button(run, 'Restore selected parameters', self.restore_history_parameters)
        self.restore_parameters_button.setToolTip('Save current edits, then restore the selected saved parameters as a new revision. Historical results and checkpoints stay intact.')
        self.history.currentIndexChanged.connect(self.select_history)
        self.results = RiverResults(self, iface)
        self.add_page("Results", self.results)

        self.name.textChanged.connect(self.changed)
        # The interpreter is a session preference, not a numerical parameter.
        # Its selection must not rebuild or silently alter the saved document.
        for w in (self.backend, self.roughness, self.white_relation, self.mesh_mode):
            w.currentIndexChanged.connect(self.changed)
        for w in (self.chezy, self.manning, self.nikuradse, self.betac, self.secondary_diffusion, self.viscosity, self.stage, self.u,
                  self.v, self.end_time, self.dt_max, self.output_interval, self.checkpoint_interval, self.threads, self.morfac,
                  *self.adaptation.values(),self.cfl):
            (w.toggled if isinstance(w,QCheckBox) else w.valueChanged).connect(self.changed)
        for w in (self.secondary, self.turbulence):
            w.toggled.connect(self.changed)
        self.minimum_size.valueChanged.connect(self.mesh_changed)
        self.levels.valueChanged.connect(self.mesh_changed)
        self._watched_domain = None
        self.domain.layerChanged.connect(self.domain_changed)
        self.dem.layerChanged.connect(self.changed)
        self.boundary_editor.changed.connect(self.changed)
        self.physics.changed.connect(self.changed)
        self.forcing.changed.connect(self.changed)
        self.runner.log.connect(self.log.appendPlainText)
        self.runner.progress.connect(self.on_progress)
        self.runner.busy_changed.connect(self.controls)
        self.runner.finished.connect(self.finished)
        self.runner.live_frame.connect(self.live_frame)
        self._loading = False
        self.workspace.textChanged.connect(self.controls)
        self.discover()
        self.changed()

    def page(self, name):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.add_page(name, page)
        return layout

    def add_page(self, name, widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        self.tabs.addTab(scroll, name)

    def discover(self):
        found = discover_gpu_pythons()
        if found:
            self.python.setText(str(found[0]))
        elif hasattr(self, "status"):
            self.status.setText("No external Python found. Choose an interpreter with NumPy/GDAL and CuPy for GPU.")

    def mesh_changed(self, *_):
        if not self._loading:
            self._grid_override = None
            self.invalidate_geometry_review()
        self.changed()

    def checkpoint_end_changed(self, *_):
        if not self._loading and self.checkpoint_duration.isChecked():
            blocked=self.checkpoint_interval.blockSignals(True)
            self.checkpoint_interval.setValue(self.end_time.value())
            self.checkpoint_interval.blockSignals(blocked)

    def checkpoint_custom_changed(self, *_):
        if not self._loading and not self.checkpoint_duration.isChecked():
            self._checkpoint_custom_interval=self.checkpoint_interval.value()

    def checkpoint_mode_changed(self, automatic):
        if self._loading:return
        if automatic:self._checkpoint_custom_interval=self.checkpoint_interval.value()
        blocked=self.checkpoint_interval.blockSignals(True)
        self.checkpoint_interval.setValue(self.end_time.value() if automatic else self._checkpoint_custom_interval)
        self.checkpoint_interval.blockSignals(blocked)
        self.changed()

    def invalidate_geometry_review(self):
        """A review belongs to one grid/domain/assignment, never a future edit."""
        if self._geometry_review is not None:
            self._geometry_review = None
            layer=self.domain.currentLayer()
            if layer is not None:layer.removeCustomProperty('braids/axis_review')
            message=('Mesh or boundary geometry changed: the previous inlet/outlet alignment review is no longer valid. '
                'Use Mesh > Align inlet/outlet sections to review the current grid. Preparation will reject off-grid open sections.')
            self.status.setText(message)
            self.log.appendPlainText(message)

    def domain_changed(self, layer):
        if self._watched_domain is not None:
            try:
                for signal in (self._watched_domain.geometryChanged, self._watched_domain.featureAdded,
                               self._watched_domain.featureDeleted, self._watched_domain.selectionChanged,
                               self._watched_domain.crsChanged):
                    try:
                        signal.disconnect(self.mesh_changed)
                    except (RuntimeError, TypeError):
                        pass
            except RuntimeError:
                pass  # QGIS may already have removed the former layer.
        self._watched_domain = layer
        if layer is not None:
            for signal in (layer.geometryChanged, layer.featureAdded, layer.featureDeleted,
                           layer.selectionChanged, layer.crsChanged):
                signal.connect(self.mesh_changed)
        self.mesh_changed()

    def changed(self, *_):
        if self._loading:
            return
        roughness = self.roughness.currentData()
        self.chezy.setEnabled(roughness == "roughness.chezy")
        self.manning.setEnabled(roughness == "roughness.manning")
        self.nikuradse.setEnabled(roughness == "roughness.white_colebrook")
        self.white_relation.setEnabled(roughness == "roughness.white_colebrook")
        for field,show in ((self.chezy,roughness=='roughness.chezy'),(self.manning,roughness=='roughness.manning'),
                (self.nikuradse,roughness=='roughness.white_colebrook'),(self.white_relation,roughness=='roughness.white_colebrook')):
            field.setVisible(show);self.general_form.labelForField(field).setVisible(show)
        sediment=self.physics.enabled.isChecked()
        self.mode.blockSignals(True);set_choice(self.mode,'sediment' if sediment else 'hydro');self.mode.blockSignals(False)
        self.tabs.setTabVisible(3,sediment);self.tabs.setTabVisible(4,sediment)
        self.forcing.setVisible(sediment);self.boundary_note.setVisible(sediment)
        self.boundary_editor.bed_box.setVisible(sediment)
        self.morfac.setVisible(sediment);self.run_form.labelForField(self.morfac).setVisible(sediment)
        self._dirty = True
        self.prepared = None
        self.betac.setEnabled(self.secondary.isChecked())
        self.secondary_diffusion.setEnabled(self.secondary.isChecked())
        self.viscosity.setEnabled(self.turbulence.isChecked())
        self.sync_forcing()
        summary = None
        try:
            model = self.physics.model(self.backend.currentData(), self.roughness.currentData(),
                                       self.secondary.isChecked(), self.turbulence.isChecked())
            summary = dict(physics=model["physics"], decisions=model["decisions"],
                           physics_id=model["physics_id"])
            self.summary.setPlainText(json.dumps(summary, indent=2))
        except Exception as exc:
            self.summary.setPlainText(str(exc))
        try:
            document = self.document()
            if summary is not None:
                # Column rules are explicit project identities, not inferred
                # from the transport law. Do not list the inert hydro carrier
                # as active sediment physics.
                summary["column_methods"] = (document["column_methods"]
                    if model["physics"]["sediment"] is not None else None)
                summary["boundary_methods"] = document["boundary_resolution"]
                summary['bank_survey']=document['bank_survey']
                summary["skin_shear_parameters"] = (document["skin_shear_parameters"]
                    if model["physics"]["sediment"] is not None and
                       model["physics"]["sediment"]["shear"]=="shear.soulsby_clarke_skin" else None)
                summary["wet_avalanche_parameters"] = (document["wet_avalanche_parameters"]
                    if model["physics"]["sediment"] is not None and
                       model["physics"]["sediment"]["avalanche"]=="redistribution.avalanche" else None)
                summary["sediment_forcing"] = (dict(fraction_ids=document["settings"]["columns"]["fraction_ids"],
                    boundaries=document["settings"]["sediment_boundaries"],
                    schedules=document["sediment_schedules"],
                    initial_concentration_kg_m3=document["initial"]["concentration_kg_m3"])
                    if model["physics"]["sediment"] is not None else None)
                self.summary.setPlainText(json.dumps(summary, indent=2))
            self.admission.setText("Configuration admitted by the shared CPU/GPU owner.")
        except Exception as exc:
            category = getattr(exc, "category", "input_required").replace("_", " ").capitalize()
            self.admission.setText(category+": "+str(exc))
        self.controls()

    def controls(self, *_):
        busy = self.runner.is_running
        editing = self.setup.digitizing.active
        for i in range(5):
            self.tabs.widget(i).setEnabled(not busy)
        for w in (self.save_button, self.load_button, self.workspace, self.folder_button,self.end_time, self.dt_max, self.output_interval, self.checkpoint_duration,
                  self.threads, self.morfac, self.restart):
            w.setEnabled(not busy and not editing)
        self.checkpoint_interval.setEnabled(not busy and not editing and not self.checkpoint_duration.isChecked())
        self.prepare_button.setEnabled(not busy and not editing)
        bound=self.project_root is not None and filesystem_path(self.workspace.text()).resolve()==self.project_root
        self.run_button.setEnabled(not busy and not editing and bound and self.prepared is not None)
        self.stop_button.setEnabled(busy and not self.runner.cancel_requested)
        self.physics.bank_survey.preview_button.setEnabled(not busy and self.prepared is not None)
        self.history.setEnabled(not busy and not editing)
        self.history_refresh.setEnabled(not busy and not editing)
        selected=getattr(self,'_history_rows',{}).get(self.history.currentData(),{})
        self.checkpoint_button.setEnabled(not busy and not editing and selected.get('action')=='run')
        self.restore_parameters_button.setEnabled(not busy and not editing and self.history.currentData() is not None)

    def error(self, exc):
        self.status.setText(str(exc))
        self.log.appendPlainText(str(exc))

    def create_edges(self):
        try:
            self.require_finished_geometry()
            layer = self.edges.create(domain_from_layer(self.domain.currentLayer()))
            self.invalidate_geometry_review()
            self.refresh_boundaries()
            if self.iface:
                self.iface.setActiveLayer(layer)
                self.iface.actionSelect().trigger()
            self.changed()
            self.setup.hide_guides()
        except Exception as exc:
            self.error(exc)

    def apply_boundary(self):
        try:
            self.require_finished_geometry()
            self.edges.apply(domain_from_layer(self.domain.currentLayer()), self.bc_kind.currentData(), self.bc_id.text())
            self.invalidate_geometry_review()
            self.refresh_boundaries()
            self.changed()
        except Exception as exc:
            self.error(exc)

    def clear_boundary(self):
        self.require_finished_geometry()
        self.edges.clear(domain_from_layer(self.domain.currentLayer()), self.bc_id.text().strip())
        self.invalidate_geometry_review()
        self.refresh_boundaries();self.changed()

    def require_finished_geometry(self):
        if self.setup.digitizing.active:
            raise ValueError('Finish or cancel the current geometry drawing/edit session first')
        bank=self.physics.bank_survey
        for layer in (self.domain.currentLayer(),self.setup.guides.currentLayer(),
                bank.polygons.currentLayer(),bank.lines.currentLayer()):
            if layer is not None and layer.isEditable():
                raise ValueError('Finish the model layer’s QGIS edit session first: '+layer.name())

    def geometry_grid(self,domain):
        if self._grid_override is not None:return deepcopy(self._grid_override)
        size=self.minimum_size.value()*2**(self.levels.value()-1)
        xs,ys=zip(*domain.vertices_xy)
        x,y=math.floor(min(xs)/size)*size,math.floor(min(ys)/size)*size
        return asdict(NativeGrid(math.ceil((max(xs)-x)/size),math.ceil((max(ys)-y)/size),x,y,size,size))

    def align_domain(self):
        """Explicit geometry replacement after review, never a preparation flag."""
        candidate=None;progress=None
        try:
            self.require_finished_geometry()
            if self.runner.is_running:raise ValueError('Stop the worker before changing geometry')
            from braids_core.boundaries.axis_aligned import derive_axis_aligned,finest_grid
            domain=domain_from_layer(self.domain.currentLayer());assignments=self.edges.assignments(domain)
            grid=self.geometry_grid(domain);g=NativeGrid(**grid)
            fine=finest_grid(g,self.levels.value())
            progress=QProgressDialog('Aligning inlet/outlet sections; retaining polygonal walls…','Cancel',0,0,self)
            progress.setWindowModality(Qt.WindowModal);progress.setMinimumDuration(0);progress.show()
            def cancelled():
                QApplication.processEvents();return progress.wasCanceled()
            new,mapped,audit=derive_axis_aligned(domain,assignments,fine,cancelled=cancelled)
            progress.close();progress=None
            # Fail before changing either selected geometry or its assignments.
            if len({a.edge_id for a in mapped})!=len(mapped):
                raise ValueError('This alignment creates multiple assignments on one edge; edit the sections first')
            candidate=layer_from_domain(new,'BRAIDS proposed inlet/outlet alignment')
            if self.iface:self.setup.digitizing.zoom(candidate)
            rows=[f"{name}: width {s['original_width_m']:.3f} → {s['aligned_width_m']:.3f} m; maximum endpoint shift {s['maximum_endpoint_shift_m']:.3f} m"
                for name,s in audit['sections'].items()]
            rows.append(f"Domain area: {audit['original_area_m2']:.3f} → {audit['derived_area_m2']:.3f} m²")
            message=QMessageBox(self);message.setWindowTitle('Review changed calculation geometry')
            message.setText('Use these aligned inlet/outlet sections and recreate the boundary faces?')
            message.setInformativeText('\n'.join(rows)+'\n\nWalls are not rasterized or forced to the finest level. Wall vertices away from open sections are unchanged; adjacent wall ends reconnect to moved section ends. Cropped wall cells remain allowed. Original layer and saved project remain untouched. Section names and hydraulic settings are retained, but widths and positions change.')
            audit['original_domain']=dict(domain_id=domain.domain_id,vertices_xy=domain.vertices_xy,spatial_reference=domain.crs_wkt)
            audit['original_assignments']=[asdict(a) for a in assignments]
            audit['approved_domain']=dict(domain_id=new.domain_id,vertices_xy=new.vertices_xy,spatial_reference=new.crs_wkt)
            audit['approved_grid']=deepcopy(grid)
            audit['approved_levels']=self.levels.value()
            audit['approved_assignments']=[asdict(a) for a in mapped]
            message.setDetailedText(json.dumps(audit,indent=2))
            message.setStandardButtons(QMessageBox.Yes|QMessageBox.No);message.setDefaultButton(QMessageBox.No)
            if message.exec_()!=QMessageBox.Yes:
                QgsProject.instance().removeMapLayer(candidate.id());candidate=None;return
            self._loading=True
            try:
                candidate.setName('BRAIDS domain with aligned inlets/outlets')
                self.domain.setLayer(candidate);self.edges.create(new,mapped);self._grid_override=grid
                self.refresh_boundaries();self.name.setText(self.name.text()+' axis-aligned')
                self._geometry_review=audit
                candidate.setCustomProperty('braids/axis_review',json.dumps(audit))
            finally:self._loading=False
            self.project_path=None;self.changed()
            self.log.appendPlainText('Accepted geometry review:\n'+json.dumps(audit,indent=2))
            self.status.setText('Inlet/outlet alignment accepted; cropped walls retained. Save as a new project, then prepare.')
        except Exception as exc:
            if candidate is not None and self.domain.currentLayer()!=candidate:
                QgsProject.instance().removeMapLayer(candidate.id())
            self.error(exc)
        finally:
            if progress is not None:progress.close()

    def boundary_kinds(self):
        return {self.boundaries.item(r,0).text():self.boundaries.item(r,1).text()
                for r in range(self.boundaries.rowCount()) if self.boundaries.item(r,1).text()!='wall'}

    def select_boundary(self,row,column):
        name=self.boundaries.item(row,0).text()
        self.bc_id.setText(name);set_choice(self.bc_kind,self.boundaries.item(row,1).text())
        if self.edges.layer is not None:
            self.edges.layer.selectByIds([f.id() for f in self.edges.layer.getFeatures() if str(f['bc_id'])==name])
        if self.boundary_editor.boundary.findData(name)>=0:
            set_choice(self.boundary_editor.boundary,name)
            self.boundary_editor.box.setExpanded(True)

    def sync_forcing(self):
        p=self.physics
        grains=[tuple(p.grains.item(r,c).text() if p.grains.item(r,c) else '' for c in (0,1))
                for r in range(p.grains.rowCount())]
        ids=[g for g,_ in grains]
        valid=all(ids) and len(set(ids))==len(ids)
        self.forcing.setEnabled(valid)
        self.boundary_editor.sync(ids if valid else [],self.boundary_kinds(),enabled=p.enabled.isChecked())
        if valid:
            self.forcing.sync(grains,self.boundary_kinds(),enabled=p.enabled.isChecked(),
                suspended=p.suspension_on.isChecked() and p.route.currentData()=='split')

    def refresh_boundaries(self):
        assignments = self.edges.assignments(self.edges.domain)
        kinds = {a.bc_id: a.kind for a in assignments}
        self.boundaries.blockSignals(True)
        self.boundaries.setRowCount(len(kinds))
        for r, (name, kind) in enumerate(sorted(kinds.items())):
            for c, text in enumerate((name, kind)):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.boundaries.setItem(r, c, item)
        self.boundaries.blockSignals(False)

    def document(self):
        domain = domain_from_layer(self.domain.currentLayer())
        assignments = self.edges.assignments(domain)
        if not self._dirty and self._loaded_project is not None:
            return deepcopy(self._loaded_project)
        dem = self.dem.currentLayer()
        if dem is None or not dem.isValid():
            raise ValueError("Select a valid terrain DEM")
        model = self.physics.model(self.backend.currentData(), self.roughness.currentData(),
                                   self.secondary.isChecked(), self.turbulence.isChecked())
        selected = model["physics"]["sediment"]
        if selected:
            grains = selected["grains"]
            thickness, fractions = self.physics.layer_values()
        else:
            # An inert numerical carrier, not a hidden sediment calculation.
            grains = [dict(id="inert", diameter_m=.0005, solid_density_kg_m3=2650., fraction=1.)]
            thickness, fractions = [.02, .2], [[1.], [1.]]
        controls = asdict(CurrentRiverControls(self.chezy.value(), grains[0]["diameter_m"],
            self.morfac.value(), self.betac.value() if self.secondary.isChecked() else 0.,
            self.viscosity.value() if self.turbulence.isChecked() else 0.,
            self.secondary_diffusion.value() if self.secondary.isChecked() else 0., threads=self.threads.value(), cfl=self.cfl.value()))
        if self._loaded_project:
            for key in ("nonlinear_atol", "nonlinear_rtol"):
                controls[key] = self._loaded_project["settings"]["controls"][key]
        p = self.physics
        bed = selected["bed"] if selected else None
        from braids_core.sediment.river_column_methods import TOP_METHODS, BASE_METHODS, POROSITY_METHODS, method_ids
        layered = bool(bed and bed["composition"] == "bed.layered")
        columns = asdict(ColumnMethodsV101(tuple(g["id"] for g in grains),
            tuple(g["solid_density_kg_m3"] for g in grains),
            tuple(p.dry_densities()) if selected else (1590.,),
            tuple(g["diameter_m"] for g in grains),
            bed="mixed" if bed and bed["composition"] == "bed.fully_mixed" else "layered",
            transport_layer="depth" if bed and bed["transport_layer"] == "transport_layer.water_depth" else "constant",
            transport_thickness_m=p.active_thickness.value() if selected else .02,
            depth_fraction=p.depth_fraction.value() if selected else .01,
            minimum_thickness_m=p.minimum_thickness.value() if selected else .001,
            underlayer_thickness_m=p.substrate_thickness.value() if selected else .2,
            eulerian_thickness_m=p.substrate_thickness.value() if layered else None,
            top_update=TOP_METHODS[p.top_update.currentData()] if layered else 1,
            base_update=BASE_METHODS[p.base_update.currentData()] if layered else 1,
            porosity=POROSITY_METHODS[p.porosity.currentData()] if layered else 0,
            solid_fraction=1.-p.porosity_value.value() if layered else .6,
            sand_solid_fraction=1.-p.sand_porosity.value() if layered else .6,
            mud_solid_fraction=1.-p.mud_porosity.value() if layered else .4,
            phi_std=tuple(p.phi_spreads()) if layered and p.porosity.currentData() in
                ("porosity.frings", "porosity.beard_weyl") else (),
            exchange_thickness_m=p.exchange_thickness.value() if layered and p.exchange_on.isChecked() else 0.,
            lagrangian_underlayers=p.lagrangian_count.value() if layered else 0,
            lagrangian_thickness_m=p.lagrangian_thickness.value() if layered and p.lagrangian_count.value() else None))
        grid=self.geometry_grid(domain)
        hydraulics,boundary_states,boundary_resolution = self.boundary_editor.values(list(columns['fraction_ids']),
            {a.bc_id:a.kind for a in assignments if a.kind!='wall'},sediment=selected,end_time=self.end_time.value())
        suspended = selected is not None and selected["suspension"] != "off"
        sediment_boundaries,initial_conc=self.forcing.values(list(columns['fraction_ids']),
            {a.bc_id:a.kind for a in assignments if a.kind!='wall'},
            sediment=selected is not None,suspended=suspended)
        sediment_schedules=self.forcing.schedule_values(list(columns['fraction_ids']),sediment_boundaries,
            sediment=selected is not None,suspended=suspended)
        p = self.physics
        settings = dict(domain=dict(domain_id=domain.domain_id, vertices_xy=domain.vertices_xy,
            spatial_reference=domain.crs_wkt), assignments=[asdict(a) for a in assignments], grid=grid,
            levels=self.levels.value(), controls=controls, columns=columns, hydraulics=hydraulics,
            sediment_boundaries=sediment_boundaries,
            bank=dict(bank_theta=p.bank_theta.value() if selected and selected['banks']=='redistribution.dry_bank' else 0.,
                collapse=bool(selected and selected["avalanche"] == "redistribution.braids_conservative_wet_dry_avalanche"),
                collapse_slope=p.collapse_slope.value()),
            suspension=dict(bed_factor=1., sus_factor=p.sus_factor.value() if suspended and
                            p.suspension.currentData()!="suspension.settling_only" else 0.,
                diffusivity=p.sus_diffusion.value() if suspended else 0.,
                temperature=p.temperature.value(), cref=p.reference_concentration.value()),
            adaptation=dict({k:w.value() for k,w in self.adaptation.items()},
                terrain_enabled=self.mesh_mode.currentData()=='static_terrain'),
            adaptive=self.mesh_mode.currentData()=='dynamic_depth',
            end_time_s=self.end_time.value(), dt_max_s=self.dt_max.value(), output_interval_s=self.output_interval.value(),
            checkpoint_interval_s=self.checkpoint_interval.value())
        doc = dict(schema=SCHEMA, name=self.name.text(), model=model, settings=settings,
            bank_survey=(p.bank_survey.value(domain.crs_wkt,p.bank_theta.value(),
                p.bank_critical.value() if selected['banks']==SURVEY_BANKS[1] else None)
                if selected and selected['banks'] in SURVEY_BANKS else None),
            sediment_schedules=sediment_schedules,
            boundary_states=boundary_states,boundary_resolution=boundary_resolution,
            terrain=dict(dem_path=dem.source().split("|")[0]),
            initial=dict(stage_m=self.stage.value(), velocity_m_s=[self.u.value(), self.v.value()],
                layer_thickness_m=thickness, layer_fractions=fractions, concentration_kg_m3=initial_conc),
            slope_parameters={k: w.value() for k, w in p.slope_parameters.items()},
            hiding_parameters={channel: {k: w.value() for k, w in controls.items()}
                               for channel, controls in p.hiding_parameters.items()},
            roughness_parameters=dict(chezy_m_half_s=self.chezy.value(),
                manning_s_m_third=self.manning.value(), nikuradse_m=self.nikuradse.value(),
                white_colebrook_relation=self.white_relation.currentData()),
            bedform_methods=deepcopy(BEDFORM_METHODS),
            van_rijn_reference_parameters={k:w.value() for k,w in p.vr_reference_parameters.items()},
            skin_shear_parameters={k:w.value() for k,w in p.skin_shear_parameters.items()},
            wet_avalanche_parameters=dict(wet_slope=p.wet_slope.value(), relaxation_time_s=p.avalanche_time.value()),
            column_methods=method_ids(ColumnMethodsV101(**columns)),
            dependent_layer_parameters=dict(
                {k: w.value() for k, w in p.dependent_parameters.items()},
                hiding_scope=p.hiding_scope.currentData()),
            editor_state=dict(schema=6, sediment=p.editor_state(),bank_survey=p.bank_survey.draft(),forcing=self.forcing.editor_state(),
                schedules=self.forcing.schedule_state(),boundaries=self.boundary_editor.editor_state(),
                checkpoint_clock=dict(mode='end_time' if self.checkpoint_duration.isChecked() else 'custom',
                    custom_interval_s=self._checkpoint_custom_interval)))
        return validate_project(doc)

    def save_project(self, path):
        self.require_finished_geometry()
        doc = self.document()
        write_json_atomic(Path(path), doc)
        if self._geometry_review is not None:
            # Geometry provenance is a sidecar, not an implicit input override.
            if (json.dumps(doc['settings']['domain'],sort_keys=True)==json.dumps(self._geometry_review['approved_domain'],sort_keys=True)
                    and doc['settings']['grid']==self._geometry_review.get('approved_grid')
                    and doc['settings']['levels']==self._geometry_review.get('approved_levels')):
                write_json_atomic(Path(path).with_suffix('.geometry-review.json'),self._geometry_review)
        self._loaded_project, self._dirty = doc, False
        self.project_path = Path(path).resolve()
        self.project_root = None
        self.workspace.setText(str(self.project_path.parent))
        return doc

    def save_dialog(self):
        try:
            self.save_project_folder(self.workspace.text())
            self.status.setText('Project folder saved, including inputs, settings and session selections.')
        except Exception as exc:
            self.error(exc)

    def load_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "Open BRAIDS project folder", self.workspace.text())
        if path:
            try:
                self.load_project_folder(path)
            except Exception as exc:
                self.error(exc)

    def load_project(self, path):
        """Current-schema document helper; the public workflow opens folders."""
        self.require_finished_geometry()
        doc = load_project(path)  # No UI changes before admission.
        self._load_document(doc, path)
        self.project_root = None
        self.refresh_history()

    def _load_document(self, doc, path):
        # Validate raster and domain before replacing any selected model state.
        domain = PolygonDomain(**doc['settings']['domain'])
        assignments=[BoundaryAssignment(**a) for a in doc['settings']['assignments']]
        if len({a.edge_id for a in assignments})!=len(assignments):
            raise ValueError('The boundary-face editor cannot load multiple partial assignments on one edge')
        dem_path = filesystem_path(doc['terrain']['dem_path'])
        if not dem_path.is_absolute():
            dem_path = filesystem_path(path).resolve().parent/dem_path
        dem = QgsRasterLayer(str(dem_path), 'BRAIDS terrain')
        if not dem.isValid():
            raise ValueError('Saved DEM is unavailable: '+str(dem_path))
        self._geometry_review=None
        self._loading = True
        try:
            p, h = doc["settings"], doc["model"]["physics"]["hydrodynamics"]
            self.setup.guides.setLayer(None)
            self.physics.bank_survey.polygons.setLayer(None)
            self.physics.bank_survey.lines.setLayer(None)
            self.domain.setLayer(layer_from_domain(domain))
            self.edges.create(domain, assignments)
            QgsProject.instance().addMapLayer(dem)
            self.dem.setLayer(dem)
            self.name.setText(doc["name"])
            set_choice(self.backend, doc["model"]["request"]["execution"]["backend"])
            set_choice(self.roughness, h["roughness"])
            self.secondary.setChecked(h["secondary_flow"] != "secondary.off")
            self.turbulence.setChecked(h["turbulence"] != "off")
            for w, key in ((self.chezy, "chezy"), (self.betac, "betac"), (self.viscosity, "viscosity"),
                (self.secondary_diffusion, "diffusivity"), (self.threads, "threads"), (self.morfac, "morfac")):
                w.setValue(p["controls"][key])
            self.cfl.setValue(p['controls']['cfl'])
            self.manning.setValue(doc["roughness_parameters"]["manning_s_m_third"])
            self.nikuradse.setValue(doc["roughness_parameters"]["nikuradse_m"])
            set_choice(self.white_relation, doc["roughness_parameters"]["white_colebrook_relation"])
            self.levels.setValue(p["levels"])
            self.minimum_size.setValue(p["grid"]["dx_m"]/2**(p["levels"]-1))
            self._grid_override = deepcopy(p["grid"])
            for key, w in self.adaptation.items():
                (w.setChecked if isinstance(w,QCheckBox) else w.setValue)(p["adaptation"][key])
            self.mesh_mode.setCurrentIndex(self.mesh_mode.findData('dynamic_depth' if p['adaptive'] else 'static_terrain'))
            self.stage.setValue(doc["initial"]["stage_m"])
            self.u.setValue(doc["initial"]["velocity_m_s"][0])
            self.v.setValue(doc["initial"]["velocity_m_s"][1])
            for w, key in ((self.end_time, "end_time_s"), (self.dt_max, "dt_max_s"),
                           (self.output_interval, "output_interval_s"), (self.checkpoint_interval, "checkpoint_interval_s")):
                w.setValue(p[key])
            checkpoint=checkpoint_editor_state(doc)
            self._checkpoint_custom_interval=checkpoint['custom_interval_s']
            self.checkpoint_duration.setChecked(checkpoint['mode']=='end_time')
            self.physics.load_project(doc)
            self.refresh_boundaries()
            self.boundary_editor.load_project(doc)
            self.forcing.load_project(doc)
            doc["terrain"]["dem_path"] = str(dem_path.resolve())
        finally:
            self._loading = False
        self.changed()
        self._loaded_project, self._dirty = doc, False
        self.project_path=Path(path).resolve();self.workspace.setText(str(self.project_path.parent))
        self.prepared = self.output = None
        self.restart.clear()
        self.results.timer.stop()
        self.results.output = None
        for layer in (self.results.layer, self.results.mesh_layer):
            if layer is not None and not sip.isdeleted(layer) and QgsProject.instance().mapLayer(layer.id()):
                QgsProject.instance().removeMapLayer(layer.id())
        self.results.layer = self.results.mesh_layer = None
        self.results.caption.setText('No results loaded')
        self.log.clear()
        self.controls()
        self.status.setText("New-format project loaded. Prepare to run.")

    def choose_project_folder(self):
        path=QFileDialog.getExistingDirectory(self,'Choose BRAIDS project folder',self.workspace.text())
        if path:
            try:
                if (filesystem_path(path)/'BRAIDS_FOLDER.json').is_file():
                    self.load_project_folder(path)
                else:
                    self.workspace.setText(path)
            except Exception as exc:self.error(exc)

    def folder_path(self, value):
        if not value or self.project_root is None:return None
        path=(self.project_root/str(value)).resolve()
        if not path.is_relative_to(self.project_root):
            raise ValueError('Saved selection is outside the project folder')
        return path

    def relative_path(self, value):
        if not value or self.project_root is None:return ''
        path=filesystem_path(value).resolve()
        return path.relative_to(self.project_root).as_posix() if path.is_relative_to(self.project_root) else ''

    def session_state(self):
        return dict(worker_python=self.python.text().strip(), prepared=self.relative_path(self.prepared),
            output=self.relative_path(self.output),restart=self.relative_path(self.restart.text().strip()),
            preview=self.preview.isChecked(),preview_variable=self.preview_variable.currentData(),
            results_variable=self.results.variable.currentData(),results_frame=self.results.slider.value(),
            results_mesh=self.results.mesh.isChecked(),results_animate_mesh=self.results.animate_mesh.isChecked(),
            selected_tab=self.tabs.currentIndex(),selected_history=self.history.currentData(),
            snap=self.setup.snap.isChecked(),boundary_name=self.bc_id.text(),boundary_kind=self.bc_kind.currentData(),
            bank_id=self.setup.bank_id.text(),hydraulic_draft=dict(betac=self.betac.value(),
                diffusivity=self.secondary_diffusion.value(),viscosity=self.viscosity.value()),
            model_layers=self.source_layer_state())

    def source_layer_state(self):
        bank=self.physics.bank_survey;result={}
        for name,combo,field in (('guides',self.setup.guides,None),
                ('bank_polygons',bank.polygons,bank.polygon_id),('toe_lines',bank.lines,bank.line_id)):
            layer=combo.currentLayer()
            if layer is None:continue
            source,*options=layer.source().split('|',1)
            relative=self.relative_path(source)
            if not relative:continue
            result[name]=dict(path=relative,options=options[0] if options else '',name=layer.name(),
                role=layer.customProperty('braids/input_role',''),field=field.currentField() if field else '')
        return result

    def restore_source_layers(self, session, *, clear_missing=True):
        bank=self.physics.bank_survey
        for name,combo,field in (('guides',self.setup.guides,None),
                ('bank_polygons',bank.polygons,bank.polygon_id),('toe_lines',bank.lines,bank.line_id)):
            row=session.get('model_layers',{}).get(name)
            if not row:
                if clear_missing:combo.setLayer(None)
                continue
            combo.setLayer(None)
            path=self.folder_path(row['path'])
            source=str(path)+('|' + row['options'] if row.get('options') else '')
            layer=QgsVectorLayer(source,row.get('name',name),'ogr')
            if not layer.isValid():
                self.log.appendPlainText('Saved drafting layer unavailable: '+str(path));continue
            layer.setCustomProperty('braids/input_role',row.get('role',''))
            QgsProject.instance().addMapLayer(layer);combo.setLayer(layer)
            if field:field.setField(row.get('field',''))

    def retain_source_layers(self):
        """Keep selected drafting/survey inputs local without changing snapshots."""
        bank=self.physics.bank_survey
        for role,combo,field in (('inlet_guides',self.setup.guides,None),
                ('bank_zones',bank.polygons,bank.polygon_id),('toe_lines',bank.lines,bank.line_id)):
            layer=combo.currentLayer()
            if layer is None:continue
            source=layer.source().split('|')[0]
            if layer.providerType()!='memory' and self.relative_path(source):continue
            chosen_field=field.currentField() if field else ''
            directory=self.project_root/'model_layers';directory.mkdir(exist_ok=True)
            target=directory/(role+'_'+uuid4().hex[:10]+'.gpkg')
            options=QgsVectorFileWriter.SaveVectorOptions();options.driverName='GPKG';options.layerName=role
            result=QgsVectorFileWriter.writeAsVectorFormatV3(layer,str(target),QgsProject.instance().transformContext(),options)
            if result[0]!=QgsVectorFileWriter.NoError:raise ValueError('Could not retain selected model layer: '+str(result))
            retained=QgsVectorLayer(str(target)+'|layername='+role,layer.name(),'ogr')
            if not retained.isValid():raise ValueError('Could not reopen copied model layer: '+str(target))
            retained.setCustomProperty('braids/input_role',role)
            QgsProject.instance().addMapLayer(retained);combo.setLayer(retained)
            if field:field.setField(chosen_field)

    def persist_session(self):
        if self.project_root is not None:
            update_session(self.project_root,self.session_state())

    def save_project_folder(self, path):
        self.require_finished_geometry()
        if self.runner.is_running:raise ValueError('Wait for the active worker before saving the model')
        root=filesystem_path(path).expanduser().resolve()
        if (root/'BRAIDS_FOLDER.json').exists() and root!=self.project_root:
            raise ValueError('Open this existing project folder before saving changes to it')
        doc=self.document()
        previous_root=self.project_root
        previous_session=self.session_state()
        if previous_root is not None and root!=previous_root:
            from braids_core.river_folder import copy_folder
            copy_folder(previous_root,root)
        saved=save_folder(root,doc,source_base=self.project_path.parent if self.project_path else None,
            session=previous_session,
            geometry_review=self._geometry_review)
        self.project_root=root;self.project_path=root/'braids_project.json';self.workspace.setText(str(root))
        if root!=previous_root:
            self.prepared=self.folder_path(previous_session.get('prepared'))
            self.output=self.folder_path(previous_session.get('output'))
            if previous_session.get('restart'):self.restart.setText(str(self.folder_path(previous_session['restart'])))
            if previous_root is not None:self.restore_source_layers(previous_session,clear_missing=False)
        # Keep the editor's runtime path local too, so the source can be removed.
        self._loaded_project=load_folder(root);self._dirty=False
        dem_path=self._loaded_project['terrain']['dem_path']
        if self.dem.currentLayer().source().split('|')[0]!=dem_path:
            self._loading=True
            try:
                dem=QgsRasterLayer(dem_path,'BRAIDS terrain');QgsProject.instance().addMapLayer(dem);self.dem.setLayer(dem)
            finally:self._loading=False
        restart=self.restart.text().strip()
        if restart and not filesystem_path(restart).resolve().is_relative_to(root):
            from braids_core.sediment.river_leaf_io import sha
            source=filesystem_path(restart).resolve()
            digest=sha(source)
            target=root/'inputs/checkpoints'/(digest+'.npz')
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                temporary=target.with_suffix('.copy_'+uuid4().hex)
                try:
                    shutil.copy2(source,temporary)
                    if sha(temporary)!=digest or sha(source)!=digest:
                        raise ValueError('Checkpoint changed while copying into the project folder')
                    os.rename(temporary,target)
                finally:temporary.unlink(missing_ok=True)
            elif sha(target)!=digest:raise ValueError('Stored checkpoint differs from its content identity')
            self.restart.setText(str(target))
        self.retain_source_layers()
        self.persist_session();self.refresh_history()
        if self.output is not None and root!=previous_root:
            try:self.results.load_output(self.output,latest=False)
            except Exception as exc:self.log.appendPlainText('Copied results: '+str(exc))
        return saved

    def _current_prepared(self, path):
        from braids_core.river_worker import load_prepared
        project,receipt=load_prepared(path)
        current=load_project(self.project_root/'braids_project.json')
        if identity(project)!=identity(current):
            raise ValueError('This preparation belongs to different saved settings. Prepare the current project.')
        return receipt

    def validate_session(self, session):
        for key in ('worker_python','boundary_name','bank_id'):
            if key in session and not isinstance(session[key],str):raise ValueError('Invalid saved session '+key)
        for key in ('preview','results_mesh','results_animate_mesh','snap'):
            if key in session and type(session[key]) is not bool:raise ValueError('Invalid saved session '+key)
        for key,maximum in (('selected_tab',self.tabs.count()-1),('results_frame',None)):
            if key in session and (type(session[key]) is not int or session[key]<0 or
                    maximum is not None and session[key]>maximum):raise ValueError('Invalid saved session '+key)
        for key,widget in (('preview_variable',self.preview_variable),('results_variable',self.results.variable),('boundary_kind',self.bc_kind)):
            if key in session and widget.findData(session[key])<0:raise ValueError('Unknown saved session '+key)
        draft=session.get('hydraulic_draft',{})
        if not isinstance(draft,dict):raise ValueError('Invalid saved hydraulic editor draft')
        for key,widget in (('betac',self.betac),('diffusivity',self.secondary_diffusion),('viscosity',self.viscosity)):
            if key in draft and (type(draft[key]) not in (int,float) or not math.isfinite(draft[key]) or
                    not widget.minimum()<=draft[key]<=widget.maximum()):raise ValueError('Invalid saved hydraulic draft '+key)

    def load_project_folder(self, path, *, preserve_current=True):
        self.require_finished_geometry()
        if self.runner.is_running:raise ValueError('Wait for the active worker before opening a project')
        root=filesystem_path(path).resolve()
        doc=load_folder(root);info=folder_info(root)
        self.validate_session(info.get('session',{}))
        if self.project_root is None and self._dirty and self.domain.currentLayer() is not None:
            raise ValueError('Save the current model to a project folder before opening another folder; its unsaved edits are still in the editor')
        # Source-layer and session edits need preserving even when the numerical
        # document is clean. Saving does not invalidate a matching preparation.
        if self.project_root is not None and preserve_current:
            self.save_project_folder(self.project_root)
            if root==self.project_root:doc=load_folder(root);info=folder_info(root)
        self._load_document(doc,root/'braids_project.json')
        self.project_root=root
        self._geometry_review=info.get('geometry_review')
        if self._geometry_review is not None:
            self.domain.currentLayer().setCustomProperty('braids/axis_review',json.dumps(self._geometry_review))
        session=info.get('session',{})
        self._loading=True
        try:
            self.python.setText(session.get('worker_python',''))
            self.preview.setChecked(session.get('preview',True))
            set_choice(self.preview_variable,session.get('preview_variable','depth_m'))
            set_choice(self.results.variable,session.get('results_variable','depth_m'))
            self.results.mesh.setChecked(session.get('results_mesh',False))
            self.results.animate_mesh.setChecked(session.get('results_animate_mesh',False))
            self.tabs.setCurrentIndex(session.get('selected_tab',0))
            self.setup.snap.setChecked(session.get('snap',True))
            self.bc_id.setText(session.get('boundary_name','inlet'))
            set_choice(self.bc_kind,session.get('boundary_kind','inlet'))
            self.setup.bank_id.setText(session.get('bank_id','bank_1'))
            draft=session.get('hydraulic_draft',{})
            if not self.secondary.isChecked():
                self.betac.setValue(draft.get('betac',1.))
                self.secondary_diffusion.setValue(draft.get('diffusivity',0.))
            if not self.turbulence.isChecked():self.viscosity.setValue(draft.get('viscosity',0.))
        finally:self._loading=False
        warnings=[]
        prepared=self.folder_path(session.get('prepared'))
        candidates=[prepared] if prepared else []
        candidates += [self.folder_path(row.get('output')) for row in reversed(list_history(root))
            if row.get('action')=='prepare' and row.get('status')=='complete']
        for candidate in dict.fromkeys(candidates):
            try:
                self._current_prepared(candidate);self.prepared=candidate;break
            except Exception as exc:
                if candidate==prepared:warnings.append('Saved preparation: '+str(exc))
        output=self.folder_path(session.get('output'))
        outputs=[output] if output else []
        outputs += [self.folder_path(row.get('output')) for row in reversed(list_history(root))
            if row.get('action')=='run']
        for candidate in dict.fromkeys(outputs):
            try:
                self.results.load_output(candidate,latest=candidate!=output);self.output=candidate
                if candidate==output:self.results.slider.setValue(min(session.get('results_frame',0),self.results.slider.maximum()))
                break
            except Exception as exc:
                if candidate==output:warnings.append('Saved results: '+str(exc))
        restart=self.folder_path(session.get('restart'))
        if restart:
            try:
                self.validate_checkpoint(restart);self.restart.setText(str(restart))
            except Exception as exc:warnings.append('Saved checkpoint: '+str(exc))
        self.refresh_history(session.get('selected_history'))
        selected=self._history_rows.get(self.history.currentData(),{})
        log=self.folder_path(selected.get('log'))
        if log and log.is_file():self.log.setPlainText(log.read_text(encoding='utf-8',errors='replace'))
        self.restore_source_layers(session)
        self._dirty=False;self.controls()
        self.status.setText('Project folder opened. '+('Prepared model ready; Run starts only when selected.' if self.prepared else 'Prepare the current settings to run.'))
        for message in warnings:self.log.appendPlainText(message)

    def refresh_history(self, selected=None):
        if isinstance(selected,bool):selected=None
        selected=selected or self.history.currentData()
        self.history.blockSignals(True);self.history.clear()
        self.history.addItem('Select saved preparation / run',None)
        rows={}
        if self.project_root is not None:
            for row in list_history(self.project_root):
                if row.get('kind')=='saved':
                    rows[row['revision']]=row
                elif row.get('action') in ('prepare','run') and row.get('output'):
                    rows[row['output']]=row
        self._history_rows=rows
        for value,row in reversed(list(rows.items())):
            label=(f"Saved parameters · {row.get('name','')}" if row.get('kind')=='saved' else
                f"{row['action']} · {row.get('status','unknown')} · {Path(value).name}")
            self.history.addItem(f"{row.get('timestamp','')} · {label}",value)
        index=self.history.findData(selected)
        self.history.setCurrentIndex(max(index,0));self.history.blockSignals(False)
        self.controls()

    def select_history(self, *_):
        value=self.history.currentData()
        if not value or self.project_root is None:return
        row=self._history_rows[value]
        self.log.clear()
        log=self.folder_path(row.get('log'))
        if log and log.is_file():self.log.setPlainText(log.read_text(encoding='utf-8',errors='replace'))
        if row.get('kind')=='saved':
            self.status.setText('Saved parameters: '+row.get('name',''))
            self.log.setPlainText(json.dumps(row,indent=2))
        else:self.status.setText(row.get('message') or f"{row['action']}: {row.get('status','unknown')}")
        output=self.folder_path(value)
        if row.get('action')=='run' and (output/'frames/INDEX.json').is_file():
            try:self.results.load_output(output);self.output=output
            except Exception as exc:self.error(exc)
        self.persist_session();self.controls()

    def validate_checkpoint(self, path):
        from braids_core.river_worker import load_prepared,read_checkpoint,checkpoint_identity
        if self.prepared is None:raise ValueError('A compatible current preparation is required to select a checkpoint')
        project,receipt=load_prepared(self.prepared)
        read_checkpoint(path,checkpoint_identity(project,receipt))

    def use_history_checkpoint(self):
        try:
            row=self._history_rows.get(self.history.currentData(),{})
            if row.get('action')!='run':raise ValueError('Select a run with an accepted checkpoint')
            path=self.folder_path(row['output'])/'checkpoint.npz'
            self.validate_checkpoint(path);self.restart.setText(str(path));self.persist_session()
            self.status.setText('Checkpoint selected. Run creates a new run folder when explicitly started.')
        except Exception as exc:self.error(exc)

    def restore_history_parameters(self):
        try:
            from braids_core.river_folder import load_revision,revision_info
            self.require_finished_geometry()
            if self.runner.is_running:raise ValueError('Wait for the active worker before restoring parameters')
            row=self._history_rows.get(self.history.currentData(),{})
            if not row.get('revision'):raise ValueError('Select saved parameters, a preparation or a run')
            root=self.project_root;revision=row['revision']
            doc=load_revision(root,revision);record=revision_info(root,revision)
            self.validate_session(record.get('session',{}))
            self.save_project_folder(root)
            current=self.session_state();session=deepcopy(record.get('session',{}))
            session.update({key:current[key] for key in ('output','results_variable','results_frame',
                'results_mesh','results_animate_mesh','selected_tab')})
            session.update(prepared='',restart='',selected_history=revision)
            save_folder(root,doc,session=session,geometry_review=record.get('geometry_review'))
            record_event(root,'restored',source_revision=revision,message='Restored selected saved parameters as a new revision')
            self.load_project_folder(root,preserve_current=False)
            self.status.setText('Selected parameters restored as a new revision. Historical runs are retained; no execution has started.')
        except Exception as exc:self.error(exc)

    def new_folder(self, prefix):
        root = self.project_root/('prepared' if prefix=='prepare' else 'runs')
        return root/(prefix+"_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")+"_"+uuid4().hex[:6])

    def begin_action(self, action, target):
        log=self.project_root/'history'/(target.name+'.log')
        self._active_event=dict(action=action,output=self.relative_path(target),log=self.relative_path(log),
            prepared=self.relative_path(self.prepared),project_id=identity(load_project(self.project_path)))
        record_event(self.project_root,'worker',status='started',**self._active_event)
        self.log.clear();self.refresh_history()
        return log

    def finish_action(self, status, message):
        if self._active_event is not None:
            log=self.folder_path(self._active_event['log'])
            if not log.exists():
                log.write_text(message+'\n',encoding='utf-8')
            record_event(self.project_root,'worker',status=status,message=message,**self._active_event)
            selected=self._active_event['output'];self._active_event=None
            self.refresh_history(selected);self.persist_session()

    def prepare(self):
        try:
            self.require_finished_geometry()
            root=filesystem_path(self.workspace.text()).expanduser().resolve()
            self.save_project_folder(root)
            target = self.new_folder("prepare")
            self.prepared = None
            log=self.begin_action('prepare',target)
            self.persist_session()
            self.runner.prepare(self.project_path, target, python=self.python.text(), runtime=self.runtime,log_path=log)
        except Exception as exc:
            self.finish_action('failed',str(exc))
            self.error(exc)

    def run(self):
        try:
            self.require_finished_geometry()
            if self.project_root is None or filesystem_path(self.workspace.text()).resolve()!=self.project_root:
                raise ValueError('Save or open the selected project folder before running')
            if self.prepared is None:
                raise ValueError("Prepare the current project first")
            # Recheck live domain edits even if they were made outside this dock.
            self.document()
            self.save_project_folder(self.project_root)
            self._current_prepared(self.prepared)
            if self.restart.text().strip():self.validate_checkpoint(self.restart.text().strip())
            target = self.new_folder("run")
            log=self.begin_action('run',target)
            self.runner.run(self.prepared, target, python=self.python.text(), runtime=self.runtime,
                            restart=self.restart.text().strip() or None,log_path=log)
            self.output = target
            self.persist_session()
        except Exception as exc:
            self.finish_action('failed',str(exc))
            self.error(exc)

    def stop(self):
        self.runner.stop()
        self.status.setText("Stopping at a safe preparation/step boundary…")
        self.controls()

    def show_bank_preview(self):
        try:
            if self.prepared is None:raise ValueError('Prepare the current project first')
            from braids_core.river_worker import load_prepared
            load_prepared(self.prepared)
            survey=read_json((self.prepared/'BANK_SURVEY.json').read_text(encoding='utf-8'))
            if survey is None:raise ValueError('The prepared project has no active bank survey')
            self.physics.bank_survey.show_prepared(survey)
        except Exception as exc:self.error(exc)

    def on_progress(self, percent, phase):
        self.progress.setValue(round(percent))
        self.status.setText(f"{phase}: {percent:.1f}%")

    def live_frame(self, output):
        if self.preview.isChecked():
            try:
                self.results.variable.blockSignals(True)
                set_choice(self.results.variable, self.preview_variable.currentData())
                self.results.variable.blockSignals(False)
                self.results.load_output(output)
            except Exception as exc:
                self.log.appendPlainText("Preview: "+str(exc))

    def finished(self, action, okay, message):
        if action == "prepare" and okay:
            self.prepared = self.runner.output
        if action == "run" and (self.runner.output/"RUN_CONTEXT.json").is_file() and (self.runner.output/'frames/INDEX.json').is_file():
            try:
                self.results.load_output(self.runner.output)
            except Exception as exc:
                self.log.appendPlainText("Results: "+str(exc))
        self.status.setText(message)
        self.finish_action('complete' if okay else 'cancelled' if self.runner.cancel_requested else 'failed',message)
        self.controls()
        if self._dispose:
            self.deleteLater()

    def shutdown(self):
        self.persist_session()
        self.setup.digitizing.cancel()
        self.results.timer.stop()
        self.runner.stop()

    def dispose_when_idle(self):
        self._dispose = True
        if not self.runner.is_running:
            self.deleteLater()
