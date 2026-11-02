"""Tab-workflow shell, portable project lifecycle and map editing integration."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
import sys
from pathlib import Path
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal, QUrl,QFileSystemWatcher
from PySide6.QtGui import QAction, QActionGroup, QIcon, QDesktopServices, QKeySequence, QColor, QPainterPath, QPen
from PySide6.QtWidgets import (QMainWindow, QWidget, QFrame, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QSplitter, QListWidget, QListWidgetItem, QStackedWidget,
    QPlainTextEdit, QProgressBar, QCheckBox, QTreeWidget, QTreeWidgetItem, QFileDialog,
    QMessageBox, QInputDialog, QDialog, QDialogButtonBox, QApplication, QButtonGroup, QComboBox, QAbstractButton,QMenuBar)
from PySide6.QtWidgets import QScrollArea, QSizePolicy
from braids_core.boundaries.domain import PolygonDomain, BoundaryAssignment
from braids_core.river_project import load_project, identity
from . import __version__, services,workspace,editor_state,project_file
from .execution import DesktopRunner
from .forms import SettingsPanel
from .map_view import MapView
from .results import ResultsPanel
from ._avac.geo import read_terrain, read_polygons, geometry_transform
from ._avac.preferences import settings,geometry_color,snap_enabled,adaptive_mesh
from ._avac.i18n import tr,apply_separator
from ._avac.window_chrome import TitleBar,ResizeEdges


class TaskSignals(QObject):
    done = Signal(object)
    failed = Signal(str)


class Task(QRunnable):
    def __init__(self, work):
        super().__init__(); self.work = work; self.signals = TaskSignals()

    def run(self):
        try: self.signals.done.emit(self.work())
        except Exception as exc: self.signals.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, *, runtime):
        super().__init__()
        self.runtime = Path(runtime)
        self.project_root = None
        self.project_file=None;self._project_file_sha=None;self._pending_project_file=None
        self.prepared = None
        self._preview_key=None;self._preview_attempt=None;self._mesh_load_scheduled=False
        self.document_base = None
        self.has_domain = False
        self.dirty = False
        self._loading = True
        self._tasks = set()
        self._chain_run = False
        self._geometry_audits = []
        self.mode='model_run';self._live_run_allowed=None;self._results_output=None
        self._model_step='Domain';self._result_step='Probes';self.advanced_enabled=False;self._preview_after_prepare=False
        self.pool = QThreadPool(self)
        self.setMinimumSize(1100, 720); self.resize(1550, 960)
        self.setWindowFlags(Qt.Window|Qt.FramelessWindowHint|Qt.WindowMinMaxButtonsHint|Qt.WindowSystemMenuHint)
        self.setContentsMargins(1,1,1,1)
        self.setWindowTitle('BRAIDS-HKT');self.setWindowIcon(QIcon(str(Path(__file__).resolve().parents[1]/'BRAIDS_logo.png')))
        self.runner = DesktopRunner(self, runtime=runtime)
        self._build()
        from . import state_initialization
        self.forms.checkpoint_selected.connect(lambda path:state_initialization.import_checkpoint(self,path))
        self.forms.saved_state_check.connect(lambda mapping:state_initialization.check(self,mapping))
        self.resize_edges=ResizeEdges(self)
        self._menus()
        self.project_watcher=QFileSystemWatcher(self);self.project_watcher.fileChanged.connect(self._schedule_project_reload);self.project_watcher.directoryChanged.connect(self._schedule_project_reload)
        self._project_reload_timer=QTimer(self);self._project_reload_timer.setSingleShot(True)
        self._project_reload_timer.timeout.connect(self._external_project_changed)
        self._project_read_retries=0
        self._translate_widgets()
        self.runner.log.connect(self.log.appendPlainText)
        self.runner.progress.connect(self._progress)
        self.runner.busy_changed.connect(self._busy)
        self.runner.finished.connect(self._worker_finished)
        self.runner.live_frame.connect(self._live_frame)
        self.new_project(confirm=False)
        self._loading = False

    def _build(self):
        outer = QWidget(); box = QVBoxLayout(outer); box.setContentsMargins(0,0,0,0); box.setSpacing(0)
        self.setCentralWidget(outer)
        self.title_bar=TitleBar(self)
        chrome=QWidget();chrome_box=QVBoxLayout(chrome);chrome_box.setContentsMargins(0,0,0,0);chrome_box.setSpacing(0)
        self.app_menu=QMenuBar(chrome)
        chrome_box.addWidget(self.title_bar);chrome_box.addWidget(self.app_menu);self.setMenuWidget(chrome)
        header = QFrame(); header.setObjectName('Header'); row = QHBoxLayout(header)
        brand = QLabel('BRAIDS'); brand.setObjectName('Brand'); row.addWidget(brand)
        row.addStretch()
        self.project_label = QLabel('Unsaved project',self);self.project_label.hide()
        self.mode_group=QButtonGroup(self);self.mode_group.setExclusive(True)
        self.model_button=QPushButton(tr('Model && Run'));self.analyze_button=QPushButton(tr('Results'))
        for button,mode in [(self.model_button,'model_run'),(self.analyze_button,'analyze')]:
            button.setCheckable(True);button.setMinimumWidth(130);button.clicked.connect(lambda _,m=mode:self.set_mode(m))
            self.mode_group.addButton(button);row.addWidget(button)
        box.addWidget(header)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        left = QFrame(); left.setObjectName('Sidebar'); left.setMinimumWidth(180); left.setMaximumWidth(320)
        nav = QVBoxLayout(left); nav.setContentsMargins(10,14,10,10)
        self.workflow_title = QLabel(tr('MODEL WORKFLOW')); self.workflow_title.setObjectName('Eyebrow'); nav.addWidget(self.workflow_title)
        self.workflow = QListWidget();self.workflow.setWordWrap(True);self.workflow.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff);nav.addWidget(self.workflow, 1)
        self.workflow.currentRowChanged.connect(self._step)
        from .accordion import AccordionSection
        self.layer_section=AccordionSection('Map layers',expanded=True);nav.addWidget(self.layer_section)
        layer_layout=self.layer_section.body_layout
        self.layers = {}
        for name in ('Terrain', 'Domain', 'Boundaries','Guide lines','Breaklines','Solids','Turbines','Roughness','Erodible depth','Mesh','Results','Profiles','Probes'):
            check = QCheckBox(name); check.setChecked(True); check.toggled.connect(self._layer_visibility); layer_layout.addWidget(check); self.layers[name] = check
        self.splitter.addWidget(left)
        center = QWidget(); middle = QVBoxLayout(center); middle.setContentsMargins(0,0,0,0); middle.setSpacing(0)
        center.setMinimumWidth(320)
        self.map_hint = QLabel('Import a DEM, draw or import a domain, then work through the steps. Right-click exits drawing/editing.')
        self.map_hint.setWordWrap(True); self.map_hint.setContentsMargins(12,9,12,9)
        self.map_hint.hide()
        map_header=QWidget();self.map_header_layout=map_header_layout=QHBoxLayout(map_header);map_header_layout.setContentsMargins(6,4,8,4)
        map_header_layout.addStretch()
        map_header_layout.addWidget(QLabel(tr('Background')));self.map_background=QComboBox()
        for label,key in [('None','none'),('OpenStreetMap','osm'),('Satellite','satellite')]:self.map_background.addItem(tr(label),key)
        self.map_background.currentIndexChanged.connect(lambda _:self.set_background(self.map_background.currentData()))
        map_header_layout.addWidget(self.map_background);middle.addWidget(map_header)
        self.map = MapView(); middle.addWidget(self.map, 1)
        self.map.dem_dropped.connect(self._drop_dem)
        self.map.boundary_edit_requested.connect(self.edit_boundary)
        self.map.zoom_domain_requested.connect(lambda:self.geometry_action('zoom_domain'))
        self.coordinate = QLabel('Projected coordinates in metres'); self.coordinate.setContentsMargins(10,5,10,5); middle.addWidget(self.coordinate)
        self.map.coordinate.connect(self.coordinate.setText)
        self.map.geometry_changed.connect(self._geometry_changed)
        self.map.edge_selection_changed.connect(lambda ids: self.map_hint.setText(f'{len(ids)} edge(s) selected. Ctrl-click adds/removes edges; Apply assigns the section.'))
        for name,key in (('Scale','scale'),('North arrow','north'),('Legends','legends'),('Time','time')):
            check=QCheckBox(name);check.setChecked(True);check.toggled.connect(lambda enabled,k=key:self._decoration(k,enabled));layer_layout.addWidget(check);self.layers[name]=check
        self.splitter.addWidget(center)
        right = QWidget(); right.setMinimumWidth(380); rbox = QVBoxLayout(right); rbox.setContentsMargins(8,12,8,8)
        self.step_title = QLabel('Domain'); self.step_title.setObjectName('Title'); rbox.addWidget(self.step_title)
        self.right_stack = QStackedWidget(); rbox.addWidget(self.right_stack, 1)
        self.forms = SettingsPanel(); self.right_stack.addWidget(self.forms)
        self.forms.localize_source=self._local_input
        self.forms.physics.bank_survey.localize_source=self._local_input
        self.forms.changed.connect(self._changed); self.forms.geometry_action.connect(self.geometry_action)
        self.forms.physics.bank_survey.draw_requested.connect(self._begin_bank_drawing)
        self.map.captured_geometry.connect(self._bank_geometry_drawn)
        self.map.breakline_edited.connect(self._breakline_edited)
        self.map.solid_edited.connect(self._solid_edited)
        self.map.zone_edited.connect(self._zone_edited)
        self.results = ResultsPanel(map_view=self.map, parent=self)
        self.map.results_panel=self.results;self.map.styles_changed.connect(self._appearance_changed)
        self.map_header_layout.insertWidget(0,self.results.map_controls,1);self.results.map_controls.hide()
        results_scroll=QScrollArea();results_scroll.setWidgetResizable(True);results_scroll.setWidget(self.results)
        self.right_stack.addWidget(results_scroll)
        self.results.error.connect(self.log_error)
        self.results.frame_changed.connect(lambda _:self._layer_visibility())
        self.results.measurements_changed.connect(lambda _:self._changed())
        self.results.probe_requested.connect(lambda: self.map.set_tool('gauge'))
        self.results.transect_requested.connect(lambda: self.map.set_tool('profile'))
        self.layers['Results'].hide()
        self.layers['Mesh'].toggled.connect(self.results.mesh.setChecked)
        self.history_tree=QTreeWidget(self);self.history_tree.hide()
        self.run_actions=QWidget();run_actions=QVBoxLayout(self.run_actions);run_actions.setContentsMargins(0,0,0,0)
        self.live = QCheckBox('Show results while running'); self.live.setChecked(True);run_actions.addWidget(self.live)
        self.live.toggled.connect(self._live_changed)
        actions = QHBoxLayout(); self.prepare_button = QPushButton('Prepare'); self.run_button = QPushButton('Run'); self.stop_button = QPushButton('Stop')
        self.run_button.setObjectName('Primary');self.stop_button.setObjectName('Stop'); self.stop_button.setEnabled(False)
        for button in (self.prepare_button,self.run_button): actions.addWidget(button)
        self.prepare_button.clicked.connect(self.prepare); self.run_button.clicked.connect(self.run); self.stop_button.clicked.connect(self.runner.stop)
        run_actions.addLayout(actions)
        self.forms.layouts['Run'].insertWidget(0,self.run_actions)
        self.splitter.addWidget(right); self.splitter.setSizes([205,850,490]); box.addWidget(self.splitter, 1)
        footer = QWidget();footer.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Fixed)
        bottom = QVBoxLayout(footer); bottom.setContentsMargins(10,5,10,6)
        progress_row = QHBoxLayout(); self.status = QLabel('Ready Â· hydrodynamics only'); progress_row.addWidget(self.status,1)
        self.progress = QProgressBar(); self.progress.setRange(0,1000); self.progress.setValue(0); self.progress.setMaximumWidth(300); progress_row.addWidget(self.progress)
        self.log_toggle = QPushButton('Execution log'); self.log_toggle.setCheckable(True); self.log_toggle.toggled.connect(lambda checked: self.log.setVisible(checked)); progress_row.addWidget(self.log_toggle)
        progress_row.addWidget(self.stop_button)
        bottom.addLayout(progress_row)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(8000); self.log.setMaximumHeight(190); self.log.hide(); bottom.addWidget(self.log)
        self.log.setContextMenuPolicy(Qt.CustomContextMenu);self.log.customContextMenuRequested.connect(self._log_menu)
        box.addWidget(footer)

    def _menus(self):
        def action(menu, label, callback, shortcut=None):
            a = QAction(tr(label.replace('â€¦','').replace('...','')),self); a.triggered.connect(callback)
            if shortcut: a.setShortcut(shortcut)
            menu.addAction(a); return a
        file = self.app_menu.addMenu(tr('File'))
        action(file,'New project',self.new_project,QKeySequence.New)
        action(file,'Open project',self.open_folder,QKeySequence.Open)
        self.recent_menu=file.addMenu(tr('Open recent'));self.recent_menu.aboutToShow.connect(self._recent_projects)
        action(file,'Import current project JSONâ€¦',self.import_project)
        action(file,'Save project',self.save,QKeySequence.Save)
        action(file,'Save project as',lambda:self.save(as_new=True),QKeySequence.SaveAs)
        action(file,'Edit project JSON',self.edit_project_json)
        action(file,'Reload project JSON',self.reload_project_json)
        file.addSeparator(); action(file,'Exit',self.close,QKeySequence.Quit)
        tools = self.app_menu.addMenu(tr('Geometry'))
        for label,name in [('Import DEM','import_dem'),('Import domain','import_domain'),('Draw domain','draw_domain'),('Draw guide line','draw_guide'),('Import guide lines','import_guides'),('Edit domain vertices','edit_domain'),('Select boundary edges','select_edges'),('Export domain GeoJSON','export_domain')]:
            action(tools,label,lambda _,n=name:self.geometry_action(n))
        self.extensions_menu=self.app_menu.addMenu(tr('Advanced'))
        self.hydrodynamics_action=QAction(tr('Advanced hydrodynamics'),self);self.hydrodynamics_action.setCheckable(True)
        self.hydrodynamics_action.toggled.connect(self._advanced_changed);self.extensions_menu.addAction(self.hydrodynamics_action)
        self.sediment_action=QAction(tr('Sediment transport'),self);self.sediment_action.setCheckable(True)
        self.sediment_action.toggled.connect(self.forms.sediment_enabled.setChecked);self.extensions_menu.addAction(self.sediment_action)
        self.forms.physics.enabled.toggled.connect(self._extension_changed)
        self.turbine_action=QAction(tr('Hydrokinetic turbines'),self);self.turbine_action.setCheckable(True)
        self.turbine_action.toggled.connect(self.forms.turbines.enabled.setChecked);self.extensions_menu.addAction(self.turbine_action)
        self.forms.turbines.enabled.toggled.connect(self._turbine_extension_changed)
        view = self.app_menu.addMenu(tr('View'))
        action(view,'Fit terrain',self.map.fit_domain)
        action(view,'Fit domain',lambda:self.geometry_action('zoom_domain'))
        self.extrema_actions={}
        for key,label in (('minimum','Show result minimum'),('maximum','Show result maximum')):
            a=QAction(tr(label),self);a.setCheckable(True)
            a.setToolTip('Show the raw visible-cell value and its location for the current field and frame.')
            a.toggled.connect(lambda enabled,k=key:self.results.set_extremum(k,enabled))
            view.addAction(a);self.extrema_actions[key]=a
        self.smooth_action=QAction(tr('Smooth results'),self);self.smooth_action.setCheckable(True)
        self.smooth_action.setToolTip('Join nearby wet-cell tips with straight shoreline segments and use linear colors. Solid and domain walls keep their exact mesh shape. Display only: probes and CSV always use raw results.')
        self.smooth_action.setChecked(settings().value('map/smooth_results',True,type=bool));self.smooth_action.toggled.connect(self._smooth_changed);view.addAction(self.smooth_action)
        self.flow_lines_action=QAction(tr('Flow particle tracks'),self);self.flow_lines_action.setCheckable(True)
        self.flow_lines_action.toggled.connect(self.results.set_particles);view.addAction(self.flow_lines_action)
        background=view.addMenu(tr('Map background'));self.background_group=QActionGroup(self);self.background_actions={}
        for label,key in [('None','none'),('OpenStreetMap','osm'),('Satellite','satellite')]:
            item=QAction(tr(label),self);item.setCheckable(True);self.background_group.addAction(item);background.addAction(item)
            item.triggered.connect(lambda _,v=key:self.set_background(v));self.background_actions[key]=item
        preferences=self.app_menu.addMenu(tr('Settings'));action(preferences,'Preferences',self.edit_settings)
        help_menu = self.app_menu.addMenu(tr('Help'))
        action(help_menu,'Open demo',self.open_demo)
        action(help_menu,'Quick start',self.quick_start)
        action(help_menu,'About BRAIDS',lambda:QMessageBox.information(self,'BRAIDS',
            f'BRAIDS {__version__}\nQualified explicit CPU/GPU solver.'))

    def _step(self,index):
        item=self.workflow.item(index)
        if item is None:return
        name=item.data(Qt.UserRole);self.step_title.setText(tr(name))
        if name in SettingsPanel.step_names:
            self._model_step=name;self.right_stack.setCurrentIndex(0);self.forms.set_step(name)
        else:
            self._result_step=name;self.right_stack.setCurrentIndex(1);self.results.set_page(name)

    def _refresh_navigation(self):
        selected=self._model_step if self.mode=='model_run' else self._result_step
        names=list(SettingsPanel.step_names) if self.mode=='model_run' else list(self.results.pages)
        turbines=((self.results.store.context.get('turbines') or (self.results.store.project or {}).get('power_plan')) if self.results.store is not None else
                  self.forms.turbines.enabled.isChecked() and bool(self.forms.turbines.rows))
        if self.mode=='analyze' and not turbines:names=[n for n in names if n!='Turbines']
        if not self.advanced_enabled:names=[n for n in names if n!='Advanced hydrodynamics']
        if self.mode=='model_run' and not self.forms.turbines.enabled.isChecked():names=[n for n in names if n!='Hydrokinetic turbines']
        from .sediment_workflow import STEPS
        if not self.forms.physics.enabled.isChecked():names=[n for n in names if n not in STEPS]
        self.workflow.blockSignals(True);self.workflow.clear()
        number=0
        for name in names:
            if name!='Advanced hydrodynamics':number+=1
            text=(('03a  ' if name=='Advanced hydrodynamics' else f'{number:02}  ') if self.mode=='model_run' else '')+tr(name)
            if name=='Advanced hydrodynamics':text=text.replace('Advanced ','Advanced\n       ',1)
            item=QListWidgetItem(text);item.setToolTip(tr(name));item.setData(Qt.UserRole,name);self.workflow.addItem(item)
        self.workflow.setCurrentRow(names.index(selected) if selected in names else 0);self.workflow.blockSignals(False)
        self._step(self.workflow.currentRow())

    def set_mode(self,mode):
        self.mode=mode;analyze=mode=='analyze'
        self.model_button.setChecked(not analyze);self.analyze_button.setChecked(analyze)
        self.workflow_title.setText(tr('RESULTS' if analyze else 'MODEL WORKFLOW'))
        self.results.map_controls.setVisible(analyze);self.layers['Results'].setVisible(analyze)
        self.map.set_tool('pan');self.map.set_model_tools(not analyze and not self.runner.is_running and not self._tasks)
        allowed=self._can_view_results()
        self.results.setEnabled(allowed)
        self.results.map_controls.setEnabled(allowed)
        self.results._results_active=analyze and allowed;self.results._update_particles()
        for item in (self.results._field_item,self.results._mesh_item):
            if item is not None:
                from shiboken6 import isValid
                if isValid(item):item.setVisible(analyze and allowed)
        self.map.result_metadata=None;self.map.time_text=''
        if analyze and allowed:
            if self._results_output and self.results.store is None:self.results.load_output(self._results_output)
            else:self.results.show_frame()
        elif analyze:self.results.caption.setText(tr('Live results were disabled before this run. Results become available when the run finishes.'))
        self._refresh_navigation()
        self._layer_visibility();self.map.viewport().update()

    def _select_step(self,name):
        self.set_mode('model_run')
        for i in range(self.workflow.count()):
            if self.workflow.item(i).data(Qt.UserRole)==name:self.workflow.setCurrentRow(i);return

    def _extension_changed(self,enabled):
        if hasattr(self,'sediment_action'):
            self.sediment_action.blockSignals(True);self.sediment_action.setChecked(enabled);self.sediment_action.blockSignals(False)
        self._refresh_navigation()

    def _turbine_extension_changed(self,enabled):
        self.turbine_action.blockSignals(True);self.turbine_action.setChecked(enabled);self.turbine_action.blockSignals(False)
        self._refresh_navigation();self._changed()

    def _advanced_changed(self,enabled):
        self.advanced_enabled=enabled
        if not enabled:
            self.forms.secondary.setChecked(False);self.forms.turbulence.setChecked(False)
        self._refresh_navigation();self._changed()

    def _can_view_results(self):
        if self.results.store is not None:return True
        if self._chain_run and self._live_run_allowed is False:return False
        if self.runner.is_running:return self._live_run_allowed is True
        return self.live.isChecked() or self._results_output is not None

    def _live_changed(self,*_):
        if self.runner.is_running:return
        self._changed()
        self.results.setEnabled(self._can_view_results())
        if self.mode=='analyze':self.set_mode('analyze')

    def _translate_widgets(self):
        for widget in self.findChildren(QWidget):
            if not isinstance(widget,(QLabel,QAbstractButton)):continue
            text=widget.text();widget.setText(tr(text.replace('â€¦','').replace('...','').replace('_',' ')))
        for widget in self.findChildren(QComboBox):
            if widget in (self.forms.boundary_editor.boundary,self.results.boundary):continue
            for i in range(widget.count()):widget.setItemText(i,tr(widget.itemText(i).replace('_',' ')))

    def _changed(self):
        if not self._loading:
            try:self.map.set_turbines(self.forms.turbines.value())
            except (ValueError,TypeError,KeyError):pass
            self._preview_current=self._preview_key is not None and self._preview_key==self._mesh_key()
            self._layer_visibility()
            self.dirty = True; self._title()

    def _mesh_key(self,document=None):
        """Initial mesh dependencies, not playback, probes or timestep controls."""
        try:
            doc=document if document is not None else self._collect(validate=False)
            s=doc['settings'];dem=Path(doc['terrain']['dem_path'])
            if not dem.is_absolute():dem=(self.project_root or Path.cwd())/dem
            return identity(dict(terrain=str(dem.resolve()),initial=doc['initial'],bank_survey=doc['bank_survey'],turbines=doc['turbines'],power_plan=doc.get('power_plan'),
                **{k:s[k] for k in ('domain','assignments','grid','levels','breaklines','solids','adaptive','adaptation')}))
        except (KeyError,TypeError,ValueError):return None

    def _title(self):
        name=(self.project_file.name[:-len(project_file.SUFFIX)] if self.project_file else self.project_root.name if self.project_root else 'Unsaved project')
        self.setWindowTitle('BRAIDS â€” '+name)
        self.title_bar.title.setText('BRAIDS â€” '+name+(' *' if self.dirty else ''))
        self.results.project_name=name
        self.project_label.setText(name)
        self.project_label.setToolTip(str(self.project_root or 'Choose a folder when saving'))

    def _log_menu(self,position):
        menu=self.log.createStandardContextMenu();menu.addSeparator()
        action=menu.addAction(tr('Clear log'));action.triggered.connect(self.log.clear)
        menu.exec(self.log.mapToGlobal(position))

    def _recent_projects(self):
        self.recent_menu.clear()
        for filename in settings().value('projects/recent',[],type=list):
            path=Path(filename);action=self.recent_menu.addAction(path.name+' â€” '+str(path.parent))
            action.setEnabled(path.is_file());action.triggered.connect(lambda _,p=path:self.open_path(p))
        if self.recent_menu.isEmpty():self.recent_menu.addAction('No recent projects').setEnabled(False)
        else:
            self.recent_menu.addSeparator();clear=self.recent_menu.addAction('Clear recent list')
            clear.triggered.connect(lambda:settings().remove('projects/recent'))

    def _remember_project(self,filename):
        filename=str(Path(filename).resolve());recent=settings().value('projects/recent',[],type=list)
        settings().setValue('projects/recent',[filename,*[p for p in recent if p!=filename]][:10])

    def _watch_project(self):
        if not hasattr(self,'project_watcher'):return
        paths=self.project_watcher.files()+self.project_watcher.directories()
        if paths:self.project_watcher.removePaths(paths)
        self._project_file_sha=None
        if self.project_file and self.project_file.is_file():
            self._project_file_sha=workspace.digest(self.project_file)
            self.project_watcher.addPaths([str(self.project_file),str(self.project_file.parent)])

    def _schedule_project_reload(self,*_):
        # A Windows watcher can fire while our own atomic writer still owns
        # the file. Coalesce notifications; never read from this signal handler.
        if self.project_file:self._project_reload_timer.start(300)

    def _external_project_changed(self):
        if not self.project_file:return
        if self.runner.is_running or self._tasks:
            self._project_reload_timer.start(1000);return
        try:stamp=workspace.digest(self.project_file)
        except (PermissionError,FileNotFoundError) as exc:
            self._project_read_retries+=1
            if self._project_read_retries<=10:self._project_reload_timer.start(500)
            else:
                self.log.appendPlainText('Project file remains unavailable; use Reload project JSON when it is accessible: '+str(exc))
            return
        self._project_read_retries=0
        # Atomic replacement can remove the file watch. The directory watch
        # remains, but re-arm both even when this was our own completed save.
        if str(self.project_file) not in self.project_watcher.files():self.project_watcher.addPath(str(self.project_file))
        if stamp==self._project_file_sha:return
        if self.dirty:
            # Keep the file as the explicit pending version if the user declines.
            if getattr(self,'_declined_external_sha',None)==stamp:return
            if QMessageBox.question(self,'Project JSON changed','Reload the changed project JSON? Unsaved UI changes will be discarded.',QMessageBox.Yes|QMessageBox.No,QMessageBox.No)!=QMessageBox.Yes:
                self._declined_external_sha=stamp;return
        self.open_path(self.project_file,confirmed=True)

    def reload_project_json(self):
        if not self.project_file or not self.project_file.is_file():self.error('Save the project first.');return
        if self.runner.is_running or self._tasks:self.error('Finish the current operation before reloading the JSON.');return
        if self.dirty and QMessageBox.question(self,'Reload project JSON','Discard unsaved UI changes and reload the project JSON?',QMessageBox.Yes|QMessageBox.No,QMessageBox.No)!=QMessageBox.Yes:return
        self.open_path(self.project_file,confirmed=True)

    def edit_project_json(self):
        if self.runner.is_running or self._tasks:self.error('Finish the current operation before editing JSON.');return
        if not self.project_file or not self.project_file.is_file():
            self.save(then=self.edit_project_json);return
        if self.dirty:
            self.save(then=self.edit_project_json);return
        dialog=QDialog(self);dialog.setWindowTitle('Edit project JSON');dialog.resize(900,720);layout=QVBoxLayout(dialog)
        note=QLabel('Edit the project settings below. Data paths are relative to data_folder. Save reloads the UI; Prepare/Run still require valid physical settings.');note.setWordWrap(True);layout.addWidget(note)
        from .json_editor import JsonEditor
        editor=JsonEditor(self.project_file.read_text(encoding='utf-8'));layout.addWidget(editor)
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel);layout.addWidget(buttons);buttons.rejected.connect(dialog.reject)
        def accept():
            try:
                from braids_core.river_folder import read_json,_atomic_json
                value=editor.value()
                if value.get('schema')!=project_file.SCHEMA:raise ValueError('Keep the current desktop-project schema.')
                _atomic_json(self.project_file,value);dialog.accept()
            except Exception as exc:QMessageBox.warning(dialog,'Project JSON',str(exc))
        buttons.accepted.connect(accept)
        if dialog.exec()==QDialog.Accepted:self.open_path(self.project_file,confirmed=True)

    def _confirm_replace(self):
        if self.runner.is_running or self._tasks:
            self.error('Wait for the current operation, or stop the solver, before changing projects.'); return False
        if self.dirty:
            return QMessageBox.question(self,'Unsaved settings','Discard unsaved changes?',QMessageBox.Discard|QMessageBox.Cancel,QMessageBox.Cancel)==QMessageBox.Discard
        return True

    def new_project(self, checked=False, *, confirm=True):
        if confirm and not self._confirm_replace():return
        self._clear_results()
        self.forms.restart.clear()
        self.project_root = self.prepared = None; self.has_domain = False; self._geometry_audits=[]
        self._preview_key=None;self._preview_attempt=None;self._preview_current=False
        self.project_file=None;self._project_file_sha=None;self._pending_project_file=None;self._watch_project()
        self.advanced_enabled=False;self._model_step='Domain';self._result_step='Probes'
        self.history_tree.clear()
        doc = services.default_document('SELECT_DEM.tif',[[0,0],[100,0],[100,50],[0,50]],'EPSG:2056')
        doc['initial']['stage_m']=-1e7
        self.map.styles={};self.map.clear();self.forms._dry_stage_m=-1e7
        self._load_document(doc);self.forms.checkpoint_duration.setChecked(True)
        self._live_run_allowed=None;self.live.setEnabled(True);self.set_mode('model_run')
        self.dirty=False; self.workflow.setCurrentRow(0); self.progress.setValue(0); self.status.setText('Import terrain to begin'); self._title()

    def _load_document(self, doc):
        self._loading=True
        try:
            self.document_base=deepcopy(doc); self.forms.load_document(doc)
            self.advanced_enabled=self.advanced_enabled or self.forms.secondary.isChecked() or self.forms.turbulence.isChecked()
            self.hydrodynamics_action.blockSignals(True);self.hydrodynamics_action.setChecked(self.advanced_enabled);self.hydrodynamics_action.blockSignals(False)
            self._display_domain(doc)
            self._extension_changed(self.forms.physics.enabled.isChecked())
            self._turbine_extension_changed(self.forms.turbines.enabled.isChecked())
            self.set_background(str(settings().value('map/background','none')),persist=False)
        finally:self._loading=False
        self._title(); self._layer_visibility()

    def _display_domain(self, doc):
        if self.map.terrain and self.has_domain:
            domain=PolygonDomain(**doc['settings']['domain']);self.map.domain_edges=list(domain.edges)
            self.map.set_release([(np.asarray(domain.vertices_xy),[])])
            self.map.set_boundaries(doc['settings']['assignments'],selected=[])
            self.map.set_breaklines(doc['settings']['breaklines'])
            self.map.set_solids(doc['settings']['solids'])
            self.map.set_zones(doc['material_zones'])
            self.map.set_turbines(doc.get('power_plan',{}).get('layout',doc['turbines']))

    def _clear_results(self):
        self.results.clear();self.results._desktop_output=None
        self._results_output=None
        self.map.result_metadata=None;self.map.time_text=''

    def _collect(self, *, ready=False,validate=True):
        if ready and (not self.has_domain or not self.map.terrain):
            raise ValueError('Import a DEM and draw or import the calculation domain first.')
        return self.forms.document(self.document_base,validate=validate)

    def _new_folder(self):
        selected,_=QFileDialog.getSaveFileName(self,'Save BRAIDS project',str(self.project_file or (self.project_root.parent if self.project_root else Path.home()/'Documents')/('River'+project_file.SUFFIX)),'BRAIDS project (*.braids.json)')
        if not selected:return None
        filename=Path(selected).resolve()
        if not filename.name.endswith(project_file.SUFFIX):filename=filename.with_name(filename.name+project_file.SUFFIX)
        root=project_file.data_directory(filename)
        if filename.exists() or (root.exists() and any(root.iterdir())):raise ValueError('Choose a new project filename. Use Open project to resume an existing file.')
        root.mkdir(parents=True,exist_ok=True);self._pending_project_file=filename
        return root

    def _local_input(self,source,kind='file'):
        if self.project_root is None:
            root=self._new_folder()
            if root is None:raise InterruptedError('Import cancelled: no project folder selected.')
            self.project_root=root;self._title()
            self.project_file=self._pending_project_file;self._title()
        return workspace.import_asset(self.project_root,source,kind=kind)

    def _background(self, work, done, label):
        if self.runner.is_running or self._tasks:raise ValueError('Another operation is in progress.')
        task = Task(work); self._tasks.add(task); self._busy(True); self.progress.setRange(0,0); self.status.setText(label)
        def complete(value=None,error=None):
            self._tasks.discard(task); self.progress.setRange(0,1000); self._busy(False)
            if error:self.error(error)
            else:
                try:done(value)
                except Exception as exc:self.error(exc)
        task.signals.done.connect(lambda value:complete(value)); task.signals.failed.connect(lambda error:complete(error=error)); self.pool.start(task)

    def error(self, error):
        self.status.setText('Action needs attention'); self.log.appendPlainText('ERROR: '+str(error))
        # Validation can fail before a worker/log file exists. Keep those
        # actionable messages alongside the project as well as in the panel.
        try:
            folder=(self.project_root/'logs') if self.project_root is not None else self.runtime.parent/'evidence'
            folder.mkdir(parents=True,exist_ok=True)
            with (folder/'desktop_errors.log').open('a',encoding='utf-8') as stream:
                stream.write(datetime.now().astimezone().isoformat()+' ERROR: '+str(error)+'\n')
        except OSError as exc:self.log.appendPlainText('Could not persist this error to disk: '+str(exc))
        self.log_toggle.setChecked(True)

    def log_error(self, error):
        self.error('Results: '+str(error))

    def open_folder(self):
        if not self._confirm_replace():return
        filename,_=QFileDialog.getOpenFileName(self,'Open BRAIDS project',str(self.project_file or self.project_root or Path.home()),'BRAIDS project (*.braids.json braids_project.json)')
        if filename:self.open_path(Path(filename),confirmed=True)

    def import_project(self):
        if not self._confirm_replace():return
        filename,_=QFileDialog.getOpenFileName(self,'Import current BRAIDS project JSON',str(Path.home()),'BRAIDS project (*.json)')
        if filename:
            try:
                root=self._new_folder()
                if root:self.open_path(Path(filename),confirmed=True,destination=root)
            except Exception as exc:self.error(exc)

    def open_path(self,path,*,confirmed=False,destination=None):
        if not confirmed and not self._confirm_replace():return
        path=Path(path).resolve()
        def read():
            public=path.name.endswith(project_file.SUFFIX) and path.is_file();changed=False
            from braids_core.river_folder import MANIFEST_FILE,PROJECT_FILE
            root=path if path.is_dir() else path.parent if path.name==PROJECT_FILE and (path.parent/MANIFEST_FILE).is_file() and not destination else None
            if public:root,doc,draft,changed=project_file.read(path)
            else:
                draft=workspace.read_editor(root) if root else None
                doc=draft['document'] if draft else services.load_project_folder(root) if root else load_project(path)
            terrain=None
            if (draft is None and doc['terrain']['dem_path']) or (draft and draft['has_terrain']):
                dem=Path(doc['terrain']['dem_path'])
                if not dem.is_absolute():dem=(path if path.is_dir() else path.parent)/dem
                if destination:
                    workspace.import_asset(destination,path)
                    dem=workspace.import_asset(destination,dem,kind='raster')
                doc['terrain']['dem_path']=str(dem.resolve());terrain=read_terrain(dem)
            session=services.project_session(root) if root else {}
            if draft:session['desktop']=draft['session'].get('desktop',{})
            return doc,terrain,session,draft,root,public,changed
        def loaded(value):
            self._clear_results();self.forms.restart.clear()
            doc,terrain,session,draft,root,public,changed=value
            self.project_root=Path(destination) if destination else root; self.prepared=None; self.has_domain=draft['has_domain'] if draft else True
            self._preview_key=None;self._preview_attempt=None;self._preview_current=False
            self.project_file=self._pending_project_file if destination else path if public else project_file.default_file(root) if root else None
            self._pending_project_file=None
            desktop_state=session.get('desktop',{})
            self.map.styles=deepcopy(desktop_state.get('map_styles',{}))
            self.advanced_enabled=bool(desktop_state.get('advanced_hydrodynamics',False))
            self._model_step=desktop_state.get('model_step','Domain');self._result_step=desktop_state.get('result_step','Probes')
            if terrain:self.map.set_terrain(terrain,[])
            else:self.map.clear()
            self._load_document(doc)
            if draft:
                self._loading=True
                try:editor_state.restore(self.forms,draft['ui'],self.project_root)
                finally:self._loading=False
                self._extension_changed(self.forms.physics.enabled.isChecked())
            if not changed and session.get('prepared') and self.project_root:self.prepared=self.project_root/session['prepared']
            desktop_state=session.get('desktop',{})
            self.map.set_sketch_lines(desktop_state.get('guides',[]))
            self._geometry_audits=deepcopy(desktop_state.get('geometry_reviews',[]))
            # Opening a project is never consent to resume an earlier run.
            self.forms.restart.clear()
            self._live_run_allowed=None;self.set_mode('model_run')
            self.live.setChecked(bool(desktop_state.get('live_results',True)))
            self.dirty=not path.is_dir(); self._title()
            self.status.setText('Project loaded Â· review settings, then Prepare')
            self.map_hint.setText('Left: workflow and layers. Right: settings. Right-click exits drawing/editing. Ctrl-click selects multiple boundary edges.')
            if session.get('output') and self.project_root:
                output=self.project_root/session['output']
                if (output/'frames/INDEX.json').is_file():self._results_output=output
            if self._results_output is None and self.project_root:
                self._results_output=next((p for p in sorted((self.project_root/'runs').glob('*'),reverse=True) if (p/'frames/INDEX.json').is_file()),None)
            measurements=desktop_state.get('measurements')
            if measurements and measurements.get('schema')==2:self.results.set_measurement_state(measurements)
            self.dirty=changed or self.project_root is None;self._title();self._watch_project()
            if public:self._remember_project(path)
            if draft and draft['issue']:self.status.setText('Draft loaded Â· settings will be checked on Prepare')
            if destination:self.save()
        try:self._background(read,loaded,'Opening project and terrainâ€¦')
        except Exception as exc:self.error(exc)

    def save(self, checked=False, *, as_new=False, then=None):
        try:
            if self.runner.is_running:raise ValueError('Stop execution before saving changes.')
            if not as_new and self.project_file and self.project_file.is_file() and self._project_file_sha and workspace.digest(self.project_file)!=self._project_file_sha:
                self.reload_project_json();return False
            root=self.project_root if not as_new else None
            if root is None:root=self._new_folder()
            if root is None:return False
            if self.map.terrain:
                local=workspace.import_asset(root,self.forms.dem_path.text(),kind='raster')
                self.forms.dem_path.setText(str(local))
            for field in (self.forms.physics.bank_survey.polygons,self.forms.physics.bank_survey.lines):
                if field.text() and Path(field.text()).is_file():field.setText(str(workspace.import_asset(root,field.text(),kind='vector')))
            if root!=self.project_root:self.forms.restart.clear()
            elif self.forms.restart.text() and Path(self.forms.restart.text()).is_file():
                self.forms.restart.setText(str(workspace.import_asset(root,self.forms.restart.text())))
            issue=None
            try:doc=self._collect(validate=False)
            except Exception as exc:
                from .sediment_workflow import first_error
                doc=deepcopy(self.document_base);issue=first_error(exc)
            filename=self._pending_project_file if self._pending_project_file and project_file.data_directory(self._pending_project_file)==root else self.project_file if root==self.project_root and self.project_file else project_file.default_file(root)
            doc['terrain']['dem_path']=self.forms.dem_path.text();doc['name']=filename.name[:-len(project_file.SUFFIX)]
            if issue is None:
                try:
                    if not self.map.terrain or not self.has_domain:raise ValueError('Import a DEM and define the calculation domain before Prepare or Run.')
                    services.validate_document(doc)
                except Exception as exc:
                    from .sediment_workflow import first_error
                    issue=first_error(exc)
            ui=editor_state.capture(self.forms,root)
            session=services.project_session(self.project_root) if self.project_root and not as_new else {}
            restart=self.forms.restart.text().strip()
            session['restart']=Path(restart).resolve().relative_to(root).as_posix() if restart and Path(restart).is_file() else ''
            session['desktop']=dict(guides=[np.asarray(line).tolist() for line in self.map.sketch_lines],
                geometry_reviews=deepcopy(self._geometry_audits),measurements=self.results.measurement_state(),live_results=self.live.isChecked(),
                advanced_hydrodynamics=self.advanced_enabled,model_step=self._model_step,result_step=self._result_step,
                map_styles=deepcopy(self.map.styles))
            has_terrain=self.map.terrain is not None;has_domain=self.has_domain
            def write():
                result=services.save_project_folder(root,doc,session=session) if issue is None else doc
                value=workspace.save_editor(root,result,ui,session,has_domain=has_domain,has_terrain=has_terrain,issue=issue)
                project_file.write(filename,root,value['document']);return value
            def saved(value):
                if self.project_root!=root:
                    self.prepared=None;self.forms.restart.clear();self._clear_results()
                self.project_root=root;self.document_base=deepcopy(value['document'])
                self.project_file=filename;self._pending_project_file=None
                if has_terrain:
                    self.document_base['terrain']['dem_path']=str(workspace.member(root,value['document']['terrain']['dem_path']))
                    self.forms.dem_path.setText(self.document_base['terrain']['dem_path'])
                self.forms._base=deepcopy(self.document_base);self.dirty=False;self._title();self._watch_project();self._remember_project(filename)
                self.status.setText('Draft saved Â· settings will be checked on Prepare' if issue else 'Project, inputs and settings saved')
                if then:
                    if issue:
                        self._chain_run=False;self._preview_after_prepare=False;self._live_run_allowed=None
                        self.error('Draft saved. Correct this before Prepare or Run:\n'+issue)
                    else:then()
            self._background(write,saved,'Saving project and local inputsâ€¦')
            return True
        except Exception as exc:self.error(exc);return False

    def prepare(self):
        self._chain_run=False
        self._preview_after_prepare=True
        self.save(then=self._start_prepare)

    def _start_prepare(self):
        # Also guard direct calls: never begin geometry with unresolved physics.
        try:
            doc=self._collect(ready=True)
            self.log.appendPlainText('Pre-mesh configuration check passed.\n'+self.forms.method_summary(doc['model']))
        except Exception as exc:
            from .sediment_workflow import first_error
            self._chain_run=False;self._preview_after_prepare=False;self._live_run_allowed=None
            self.error(first_error(exc));return
        self.prepared=None
        self.runner.prepare(self.project_root)

    def preview_mesh(self):
        self._chain_run=False
        def saved():
            if self.prepared and (self.prepared/'PREPARED.json').is_file():
                receipt=json.loads((self.prepared/'PREPARED.json').read_text())
                if receipt.get('project_id')==identity(load_project(self.project_root/'braids_project.json')):
                    self._show_prepared_mesh();return
            self._preview_after_prepare=True;self._start_prepare()
        if not self.save(then=saved):self._preview_after_prepare=False

    def _show_prepared_mesh(self):
        if self.prepared is None or not (self.prepared/'PREPARED.json').is_file():return
        key=self._mesh_key();root=self.project_root
        def shown(value):
            if root!=self.project_root or key!=self._mesh_key():return
            polygons,faces=value;self.map.show_preview_mesh(polygons)
            self._preview_key=key;self._preview_current=True;self._layer_visibility()
            message=f'Mesh preview: {len(polygons):,} cells / {faces:,} faces. Source: {prepared.name}. This preparation is reused by Run.'
            self.log.appendPlainText(message)
            if not self.runner.is_running:self.status.setText(message)
        prepared=self.prepared
        self._mesh_read(lambda:services.prepared_mesh(prepared),shown,'Loading prepared mesh previewâ€¦')

    def _mesh_read(self,work,done,label):
        if not self.runner.is_running and not self._tasks:return self._background(work,done,label)
        # Read-only display loading may overlap the separate solver process.
        # Do not change its backend, progress, execution state or cancellation.
        task=Task(work);self._tasks.add(task)
        def complete(value=None,error=None):
            self._tasks.discard(task)
            if not self.runner.is_running and not self._tasks:self._busy(False)
            if error:self.log_error(error)
            else:
                try:done(value)
                except Exception as exc:self.log_error(exc)
        task.signals.done.connect(lambda value:complete(value));task.signals.failed.connect(lambda error:complete(error=error));self.pool.start(task)

    def _ensure_model_mesh(self):
        self._mesh_load_scheduled=False
        if self._loading or self.mode!='model_run' or not self.layers['Mesh'].isChecked() or self._tasks:return
        key=self._mesh_key()
        if key is None:return
        if self.map.preview_mesh_item is not None and self._preview_key==key:
            self._preview_current=True;self.map.preview_mesh_item.show();return
        attempt=(key,str(self.prepared),str(self._results_output))
        if self._preview_attempt==attempt:return
        self._preview_attempt=attempt
        # Prefer the already authenticated preparation. This reads saved display
        # geometry; checking the box never constructs or prepares a new mesh.
        source=None if self.prepared is None else self.prepared/'PROJECT.json'
        if source is not None and (self.prepared/'PREPARED.json').is_file() and source.is_file() and self._mesh_key(json.loads(source.read_text(encoding='utf-8')))==key:
            self._show_prepared_mesh();return
        store=self.results.store
        if store is not None and store.project is not None and self._mesh_key(store.project)==key:
            frame=store.read(0)
            self.map.show_preview_mesh([list(poly.exterior.coords) for poly in frame.polygons])
            self._preview_key=key;self._preview_current=True;self._layer_visibility();return
        # Results may exist on disk without having entered the Results mode.
        if self._results_output is not None:
            output=self._results_output;root=self.project_root
            def read():
                from .results import ResultsStore
                saved=ResultsStore(output)
                if saved.project is None or self._mesh_key(saved.project)!=key:return None
                return [list(poly.exterior.coords) for poly in saved.read(0).polygons]
            def shown(polygons):
                if polygons is not None and root==self.project_root and key==self._mesh_key():
                    self.map.show_preview_mesh(polygons);self._preview_key=key;self._preview_current=True;self._layer_visibility()
            self._mesh_read(read,shown,'Loading saved initial meshâ€¦')

    def run(self):
        from .state_initialization import is_source,import_checkpoint
        selected=self.forms.restart.text().strip()
        if selected and self.forms.restart_mode.currentData()=='initialize' and not is_source(selected):
            import_checkpoint(self,selected,then=self.run);return
        self._live_run_allowed=self.live.isChecked()
        self._chain_run=True
        def saved():
            try:
                if self.forms.restart.text().strip() and self.forms.restart_mode.currentData()=='exact':self._start_run();return
                if self.prepared and (self.prepared/'PREPARED.json').is_file():
                    receipt=json.loads((self.prepared/'PREPARED.json').read_text())
                    raw=load_project(self.project_root/'braids_project.json')
                    if receipt.get('project_id')==identity(raw):self._start_run(); return
                self._start_prepare()
            except Exception as exc:self.error(exc)
        if not self.save(then=saved):self._chain_run=False;self._live_run_allowed=None

    def _start_run(self):
        if self._live_run_allowed is None:self._live_run_allowed=self.live.isChecked()
        if self.forms.restart.text().strip() and self.forms.restart_mode.currentData()=='initialize':
            from .state_initialization import check
            check(self,then_run=True);return
        self._launch_run(restart=self.forms.restart.text().strip() or None)

    def _launch_run(self,*,restart=None,initial_state=None):
        try:
            self.runner.run(self.project_root,self.prepared,restart=restart,initial_state=initial_state)
            measurements=self.results.measurement_state();self._clear_results();self.results.set_measurement_state(measurements)
            self.live.setEnabled(False);self._chain_run=False
        except Exception as exc:
            self._chain_run=False;self._live_run_allowed=None;self.live.setEnabled(True);self.error(exc)

    def _busy(self,busy):
        self.results.set_live_output(self.runner.output if self.runner.action=='run' else None)
        for name,page in self.forms.pages.items():
            if name!='Run':page.setEnabled(not busy)
        for i in range(self.forms.run_form.count()):
            widget=self.forms.run_form.itemAt(i).widget()
            if widget:widget.setEnabled(not busy)
        for widget in (self.prepare_button,self.run_button,self.live,self.extensions_menu):widget.setEnabled(not busy)
        if self._chain_run:self.live.setEnabled(False)
        self.results.setEnabled(self._can_view_results())
        self.results.map_controls.setEnabled(self._can_view_results())
        if not busy:self.forms.threads.setEnabled(self.forms.backend.currentData()=='cpu')
        self.map.set_model_tools(not busy and self.mode=='model_run')
        if busy:self.map.set_tool('pan')
        self.stop_button.setEnabled(self.runner.is_running)

    def _progress(self,value,label):
        self.progress.setRange(0,1000); self.progress.setValue(int(max(0,min(100,value))*10)); self.status.setText(label)
        self.stop_button.setEnabled(self.runner.is_running)

    def _worker_finished(self,action,success,message):
        self.results.set_live_output(None)
        self.status.setText(message)
        if action=='prepare' and not success:self.prepared=None
        if action=='prepare' and success:
            self.prepared=self.runner.output
            if self._chain_run:QTimer.singleShot(0,self._start_run)
            elif self._preview_after_prepare:QTimer.singleShot(0,self._show_prepared_mesh)
        else:self._chain_run=False
        self._preview_after_prepare=False
        self.live.setEnabled(not self.runner.is_running and not self._tasks and not self._chain_run)
        if action=='run' and self.runner.output and (self.runner.output/'frames/INDEX.json').is_file():
            self._results_output=self.runner.output;self.results.setEnabled(True)
            self.results._results_active=self.mode=='analyze'
            if self.mode=='analyze':self.results.load_output(self.runner.output);self._refresh_navigation()
        if not success and not self.runner.cancel_requested:self.error(message)
        self._layer_visibility()

    def _live_frame(self,path):
        self._results_output=path
        self.results.set_live_output(path if self.runner.action=='run' else None)
        if self._live_run_allowed is True and self.mode=='analyze':
            try:
                if getattr(self.results,'_desktop_output',None)!=path:
                    self.results.load_output(path); self.results._desktop_output=path;self._refresh_navigation()
                else:self.results.refresh()
            except Exception as exc:self.log.appendPlainText('Live result: '+str(exc))

    def open_results(self):
        if not self._can_view_results():return
        directory=QFileDialog.getExistingDirectory(self,'Select a BRAIDS run folder',str(self.project_root or Path.home()))
        if not directory:return
        try:self._results_output=Path(directory);self.results.load_output(directory);self.set_mode('analyze')
        except Exception as exc:self.error(exc)

    def refresh_history(self):
        self.history_tree.clear()
        if not self.project_root:return
        try:
            for record in reversed(services.project_history(self.project_root)):
                item=QTreeWidgetItem([record.get('timestamp','')[:19]+'  '+record.get('kind',''),record.get('status','')]); item.setData(0,Qt.UserRole,record)
                self.history_tree.addTopLevelItem(item); item.setToolTip(0,json.dumps(record,indent=2))
            for path in sorted((self.project_root/'runs').glob('*'),reverse=True):
                if (path/'frames/INDEX.json').is_file():
                    item=QTreeWidgetItem([path.name,'Open results']); item.setData(0,Qt.UserRole,dict(output=str(path))); self.history_tree.addTopLevelItem(item)
        except Exception as exc:self.log.appendPlainText('History: '+str(exc))

    def _history_open(self,item,column):
        record=item.data(0,Qt.UserRole) or {}
        if record.get('output'):
            path=Path(record['output']); path=path if path.is_absolute() else self.project_root/path
            if (path/'frames/INDEX.json').is_file():
                try:self.results.load_output(path); self.set_mode('analyze')
                except Exception as exc:self.error(exc)

    def _selected_history(self):
        if self.runner.is_running or self._tasks:raise ValueError('Finish or stop the current operation first.')
        item=self.history_tree.currentItem()
        if item is None or self.project_root is None:raise ValueError('Select a saved revision or run in History first.')
        return item.data(0,Qt.UserRole) or {}

    def restore_history_parameters(self):
        try:
            from braids_core.river_folder import load_revision,revision_info
            row=self._selected_history();revision=row.get('revision')
            if not revision:raise ValueError('Select a saved parameter revision or its execution event.')
            answer=QMessageBox.question(self,'Restore saved parameters',
                'Load the selected revision into the editor? Unsaved editor changes will be discarded. '
                'The saved project and all history remain unchanged until you explicitly Save.',
                QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
            if answer!=QMessageBox.Yes:return
            root=self.project_root
            def read():
                doc=load_revision(root,revision);record=revision_info(root,revision)
                return doc,read_terrain(doc['terrain']['dem_path']),record
            def restored(value):
                doc,terrain,record=value;self._clear_results();self.prepared=None;self.forms.restart.clear();self.has_domain=True
                self.map.set_terrain(terrain,[]);self._load_document(doc)
                session=record.get('session',{}).get('desktop',{})
                self.map.set_sketch_lines(session.get('guides',[]));self._geometry_audits=session.get('geometry_reviews',[])
                self.dirty=True;self._title();self.workflow.setCurrentRow(0)
                self.status.setText('Saved parameters restored into editor. Save creates a new revision; earlier history remains intact.')
            self._background(read,restored,'Reading authenticated saved revisionâ€¦')
        except Exception as exc:self.error(exc)

    def use_history_checkpoint(self):
        try:
            row=self._selected_history()
            if not row.get('output'):raise ValueError('Select a completed or cancelled run with a checkpoint.')
            path=services.project_member(self.project_root,Path(row['output'])/'checkpoint.npz')
            if not path.is_file():raise ValueError('This selected item has no accepted checkpoint.')
            if self.forms.restart_mode.currentData()=='initialize':
                from .state_initialization import import_checkpoint
                self._select_step('Run');import_checkpoint(self,path);return
            if self.prepared is None:raise ValueError('Prepare the intended parameters before selecting a restart.')
            from braids_core.river_worker import load_prepared,read_checkpoint,checkpoint_identity
            project,receipt=load_prepared(self.prepared)
            read_checkpoint(path,checkpoint_identity(project,receipt))
            self.forms.restart.setText(str(path));self._select_step('Run')
            self.status.setText('Authenticated checkpoint selected. Run starts a separate output folder when pressed.')
        except Exception as exc:self.error(exc)

    def view_history_log(self):
        try:
            row=self._selected_history()
            if not row.get('log'):raise ValueError('Select an execution event with a log.')
            path=services.project_member(self.project_root,row['log'])
            self.log.setPlainText(path.read_text(encoding='utf-8',errors='replace'));self.log_toggle.setChecked(True)
        except Exception as exc:self.error(exc)

    def reveal_folder(self):
        if self.project_root:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.project_root)))
        else:self.status.setText('Save the project to choose its folder first.')

    def open_demo(self):
        if not self._confirm_replace():return
        from .demo import create_river_demo
        root=self.runtime.parent/'test_projects'/('demo_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        def done(_):self.dirty=False; self.open_path(root,confirmed=True)
        try:self._background(lambda:create_river_demo(root),done,'Copying the natural-river demonstration inputsâ€¦')
        except Exception as exc:self.error(exc)

    def _layer_visibility(self):
        if not hasattr(self,'map'):return
        for label,item in [('Terrain',self.map.terrain_item),('Domain',self.map.release_item),('Guide lines',self.map.sketch_item)]:
            if item is not None:item.setVisible(self.layers[label].isChecked())
        for item in self.map.boundary_items:item.setVisible(self.layers['Boundaries'].isChecked())
        for item in self.map.breakline_items:item.setVisible(self.layers['Breaklines'].isChecked())
        for item in self.map.solid_items:item.setVisible(self.layers['Solids'].isChecked())
        for item in self.map.turbine_items:item.setVisible(self.layers['Turbines'].isChecked())
        for kind,name in (('roughness','Roughness'),('erodible','Erodible depth')):
            for item in self.map.zone_items[kind]:item.setVisible(self.layers[name].isChecked())
        self.results.measurements_visible={'profile':self.layers['Profiles'].isChecked(),'gauge':self.layers['Probes'].isChecked()}
        self.results._draw_measurements()
        from shiboken6 import isValid
        analyze=self.mode=='analyze' and self._can_view_results()
        if self.mode=='model_run' and not self._loading:
            item=self.map.preview_mesh_item
            self._preview_current=item is not None and isValid(item) and self._preview_key==self._mesh_key()
        if self.map.preview_mesh_item is not None and isValid(self.map.preview_mesh_item):
            self.map.preview_mesh_item.setVisible(not analyze and self.layers['Mesh'].isChecked() and getattr(self,'_preview_current',False))
        smooth_ready=getattr(getattr(self,'results',None),'_smooth_ready',False)
        for key,item in (('Results',getattr(getattr(self,'results',None),'_field_item',None)),('Mesh',getattr(getattr(self,'results',None),'_mesh_item',None))):
            if item is not None and isValid(item):item.setVisible(analyze and self.layers[key].isChecked())
        if self.results._field_item is not None and isValid(self.results._field_item) and smooth_ready:self.results._field_item.hide()
        if self.results._smooth_item is not None and isValid(self.results._smooth_item):self.results._smooth_item.setVisible(analyze and self.layers['Results'].isChecked() and smooth_ready)
        self.map.result_visible=analyze and self.layers['Results'].isChecked()
        self.results.set_extrema_visible(self.map.result_visible)
        if self.map.terrain_item:self.map.terrain_item.setOpacity(self.map.styles.get('terrain',{}).get('opacity',1.))
        self.map.viewport().update()
        if (self.mode=='model_run' and self.layers['Mesh'].isChecked() and not self._loading and not self._mesh_load_scheduled
                and not self._tasks and not getattr(self,'_preview_current',False)
                and (self.prepared is not None or self._results_output is not None or self.results.store is not None)):
            self._mesh_load_scheduled=True;QTimer.singleShot(0,self._ensure_model_mesh)

    def _appearance_changed(self):
        self._layer_visibility()
        if not self._loading:self.dirty=True;self._title()

    def _decoration(self,key,enabled):
        self.map.decorations[key]=enabled;self.map.viewport().update()

    def _smooth_changed(self,enabled):
        settings().setValue('map/smooth_results',enabled);self.results.smooth=enabled;self.results.show_frame()

    def geometry_action(self,action):
        try:
            # Navigation reads display geometry and must never acquire the
            # model-editing lock used by preparation and execution.
            if action=='zoom_domain':
                if self.has_domain and self.map.release_item is not None:
                    self.map.fitInView(self.map.release_item.path().boundingRect(),Qt.KeepAspectRatio)
                else:self.map.fit_domain()
                return
            if self.runner.is_running or self._tasks:raise ValueError('Finish or stop the current operation before editing geometry.')
            if action=='import_dem':self._import_dem();return
            if not self.map.terrain:raise ValueError('Import a terrain DEM first.')
            if action in ('add_turbine','edit_turbine'):
                self.forms.turbines.begin_placement(edit=action=='edit_turbine')
                self.map.turbine_preview_diameter=self.forms.turbines.diameter.value();self.layers['Turbines'].setChecked(True)
                self.map.begin_capture('turbine');self.status.setText('Click the turbine centre, then a point downstream along the local river. Rotor stays perpendicular.');return
            if action=='remove_turbine':
                self.forms.turbines.remove_selected();self.map.set_tool('pan');return
            if '_zone:' in action:
                operation,kind=action.split(':');doc=self._collect(validate=False)
                self.layers['Roughness' if kind=='roughness' else 'Erodible depth'].setChecked(True)
                if operation=='add_zone':self.map.begin_capture('zone:'+kind);return
                name=self.forms.zones[kind].selected();rows=doc['material_zones'][kind];row=next(r for r in rows if r['id']==name)
                if operation=='edit_zone':self.map.begin_zone_edit(kind,row);return
                if operation=='remove_zone':
                    self.map.set_tool('pan');doc['material_zones'][kind]=[r for r in rows if r['id']!=name]
                    self._load_document(doc);self._changed();return
            if action=='import_domain':self._import_domain();return
            if action=='edit_sediment_boundary':
                name=self.forms.forcing.boundary.currentData()
                if not name:raise ValueError('Assign an inlet or outlet in BC & IC first.')
                self.forms.boundary_editor.bed_box.setExpanded(True)
                self.edit_boundary(name);return
            if action=='draw_domain':
                self.map.set_tool('polygon'); self.map_hint.setText('Draw one domain polygon. Double-click / Enter finishes; right-click exits. A new polygon replaces the previous domain after confirmation.');return
            if action in ('edit_domain','edit_guides'):self.map.set_tool('edit');self.map_hint.setText('Select a polygon or guide, then drag its vertex handles. Right-click exits editing. Changed domain edges must be reassigned.');return
            if action=='draw_guide':self.map.set_tool('line');return
            if action=='add_solid':
                if not self.has_domain:raise ValueError('Create the calculation domain first.')
                self.layers['Solids'].setChecked(True)
                shape=self.forms.solid_shape.currentData();self.map.begin_capture('solid_'+shape)
                self.status.setText('Click two opposite corners of the ellipse.' if shape=='ellipse' else 'Draw the solid polygon; Enter or double-click finishes.');return
            if action in ('edit_solid','remove_solid'):
                name=self.forms.solid_list.currentData();doc=self._collect(validate=False)
                row=next((r for r in doc['settings']['solids'] if r['id']==name),None)
                if row is None:raise ValueError('Add or select a solid first.')
                if action=='edit_solid':
                    self.layers['Solids'].setChecked(True);self.map.begin_solid_edit(row)
                    self.status.setText('Drag solid vertices; right-click ends editing.');return
                self.map.set_tool('pan');doc['settings']['solids']=[r for r in doc['settings']['solids'] if r['id']!=name]
                self._load_document(doc);self._changed();return
            if action=='add_breakline':
                if not self.has_domain:raise ValueError('Create the calculation domain first.')
                self.layers['Breaklines'].setChecked(True);self.map.begin_capture('breakline_line')
                self.status.setText('Draw the breakline polyline. Two or more points; Enter or double-click finishes without closing it. Right-click finishes and exits.');return
            if action in ('edit_breakline','remove_breakline'):
                name=self.forms.breakline_list.currentData();doc=self._collect(validate=False)
                row=next((r for r in doc['settings']['breaklines'] if r['id']==name),None)
                if row is None:raise ValueError('Add or select a breakline first.')
                if action=='edit_breakline':
                    self.layers['Breaklines'].setChecked(True);self.map.begin_breakline_edit(row)
                    self.status.setText('Drag breakline vertices; right-click ends editing.');return
                self.map.set_tool('pan');doc['settings']['breaklines']=[r for r in doc['settings']['breaklines'] if r['id']!=name]
                self._load_document(doc);self._changed();return
            if action=='import_guides':self._import_guides();return
            if action=='select_edges':self.map.set_tool('boundary');self.map_hint.setText('Select one exterior edge, enter its boundary name/type, then Apply.');return
            if action=='create_edges':
                self._load_document(self._collect(validate=False));self.map.set_tool('boundary');self.status.setText('Current domain edges are ready; old changed-edge assignments are not reused.');return
            if action.startswith('select_boundary:'):
                name=action.split(':',1)[1]; doc=self._collect(validate=False); ids={a['edge_id'] for a in doc['settings']['assignments'] if a['bc_id']==name}
                self.map.set_boundaries(doc['settings']['assignments'],selected=[i for i,e in enumerate(self.map.domain_edges) if e.edge_id in ids]);return
            if action.startswith('edit_boundary:'):self.edit_boundary(action.split(':',1)[1]);return
            if action=='clear_boundary':
                doc=services.clear_boundary(self._collect(validate=False),self.forms.boundary_name.text().strip(),for_editing=True);self._load_document(doc);self._changed();return
            if action=='assign_boundary':self._assign_boundary();return
            if action=='align_boundaries':self._align_boundaries();return
            if action=='preview_mesh':self.preview_mesh();return
            if action=='export_domain':self._export_domain();return
            if action=='bank_preview':self._preview_survey();return
        except Exception as exc:self.error(exc)

    def _drop_dem(self,filename):
        try:
            if self.runner.is_running or self._tasks:raise ValueError('Finish or stop the current process before replacing the DEM. Map navigation remains available.')
            self._import_dem(filename)
        except Exception as exc:self.error(exc)

    def _import_dem(self,filename=None):
        if filename is None:filename,_=QFileDialog.getOpenFileName(self,'Import terrain DEM','','Elevation raster (*.tif *.tiff *.asc *.vrt);;All files (*)')
        if not filename:return
        filename=self._local_input(filename,'raster');doc=self._collect(validate=False)
        def done(terrain):
            crs=terrain.wkt
            if not crs:raise ValueError('Assign a projected CRS in metres to the DEM before importing it.')
            old_crs=PolygonDomain(**doc['settings']['domain']).crs_wkt
            from ._avac.geo import spatial_reference
            if self.has_domain and not spatial_reference(old_crs).IsSame(spatial_reference(crs)):
                raise ValueError('The new DEM has a different CRS. Start a new project and import/reproject the domain explicitly.')
            updated=deepcopy(doc);updated['terrain']['dem_path']=str(terrain.source)
            if not self.has_domain:
                m=terrain.raster.metadata
                updated=services.replace_domain(updated,[[m['xmin'],m['ymin']],[m['xmax'],m['ymin']],[m['xmax'],m['ymax']],[m['xmin'],m['ymax']]],crs,for_editing=True)
                minimum=float(np.nanmin(terrain.raster.z))
                updated['initial'].update(stage_m=minimum-max(1.,abs(minimum)*1e-6),velocity_m_s=[0.,0.])
            self.map.set_terrain(terrain,[]);self._load_document(updated);self._changed()
            self.status.setText('DEM loaded Â· draw or import the domain' if not self.has_domain else 'DEM replaced Â· prepare again')
        self._background(lambda:read_terrain(filename),done,'Reading native DEM for zoom-dependent displayâ€¦')

    def _import_domain(self):
        filename,_=QFileDialog.getOpenFileName(self,'Import domain polygon','','Vector polygon (*.gpkg *.geojson *.json *.shp);;All files (*)')
        if not filename:return
        filename=self._local_input(filename,'vector');rings=read_polygons(filename,self.map.terrain.wkt)
        if len(rings)!=1 or rings[0][1]:raise ValueError('The current solver requires one simple domain polygon without holes. Select or export one polygon explicitly.')
        self._accept_domain(rings[0][0])

    def _accept_domain(self,vertices):
        doc=self._collect(validate=False)
        if self.has_domain and doc['settings']['assignments']:
            answer=QMessageBox.question(self,'Replace domain geometry','Changing the domain clears old boundary assignments. Apply this geometry and reassign boundaries?',QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
            if answer!=QMessageBox.Yes:self._load_document(doc);return
        updated=services.replace_domain(doc,vertices,self.map.terrain.wkt,for_editing=True);self.has_domain=True
        self._load_document(updated);self._changed();self.status.setText('Domain updated Â· select inlet/outlet edges and apply boundary conditions')

    def _geometry_changed(self,rings,lines):
        try:
            if self.runner.is_running or self._tasks:raise ValueError('Finish the current operation before editing geometry.')
            if rings:
                candidate=rings[-1][0]
                same=self.has_domain and np.array_equal(np.asarray(self.document_base['settings']['domain']['vertices_xy']),np.asarray(candidate))
                if not same:self._accept_domain(candidate)
            elif self.has_domain:
                self.has_domain=False;self.map.domain_edges=[];self.map.set_boundaries([],selected=[]);self.status.setText('Domain removed; draw or import a new domain.')
            self.map.set_sketch_lines(lines);self._changed()
        except Exception as exc:
            self._display_domain(self.document_base);self.error(exc)

    def _selected_ids(self):
        if not self.has_domain:raise ValueError('Define a calculation domain first.')
        if not self.map.selected_edges:raise ValueError('Select boundary edges on the map first.')
        if len(self.map.selected_edges)!=1:raise ValueError('Select exactly one exterior edge for each boundary condition.')
        domain=PolygonDomain(**self.document_base['settings']['domain'])
        return [domain.edges[i].edge_id for i in sorted(self.map.selected_edges)]

    def _assign_boundary(self):
        doc=self._collect(validate=False);ids=self._selected_ids();name=self.forms.boundary_name.text().strip();kind=self.forms.boundary_kind.currentData()
        updated=services.assign_boundary(doc,ids,name,kind,for_editing=True)
        self._load_document(updated);self._changed();self.status.setText(f'Boundary {name} assigned. Double-click it to edit its conditions.')

    def _align_boundaries(self,assignment=None):
        from braids_core.boundaries.axis_aligned import derive_axis_aligned,finest_grid
        from braids_core.boundaries.inlet_io import NativeGrid
        doc=self._collect(validate=False);domain=PolygonDomain(**doc['settings']['domain']);rows=[BoundaryAssignment(**a) for a in doc['settings']['assignments']]
        if assignment:
            ids,name,kind=assignment
            rows=[a for a in rows if a.bc_id!=name and a.edge_id not in ids]+[BoundaryAssignment(e,name,kind) for e in ids]
        if not any(a.kind!='wall' for a in rows):raise ValueError('Select inlet/outlet edges and a section name/type before alignment.')
        grid=deepcopy(doc['settings']['grid']);fine=finest_grid(NativeGrid(**grid),doc['settings']['levels'])
        new,mapped,audit=derive_axis_aligned(domain,rows,fine)
        updated=services.aligned_document(doc,new,mapped,grid,new_assignment=assignment,for_editing=True)
        details=[f"{n}: width {a['original_width_m']:.3f} â†’ {a['aligned_width_m']:.3f} m; endpoint shift â‰¤ {a['maximum_endpoint_shift_m']:.3f} m" for n,a in audit['sections'].items()]
        details.append(f"Area {audit['original_area_m2']:.3f} â†’ {audit['derived_area_m2']:.3f} mÂ². Walls remain cropped.")
        self.map.set_release([(np.asarray(new.vertices_xy),[])])
        message=QMessageBox(self);message.setWindowTitle('Review inlet/outlet alignment');message.setText('Use the proposed boundary geometry shown on the map?')
        message.setInformativeText('\n'.join(details));message.setDetailedText(json.dumps(audit,indent=2));message.setStandardButtons(QMessageBox.Yes|QMessageBox.No);message.setDefaultButton(QMessageBox.No)
        if message.exec()!=QMessageBox.Yes:self._load_document(doc);return
        audit['original_domain']=doc['settings']['domain'];audit['approved_domain']=updated['settings']['domain'];self._geometry_audits.append(audit)
        self._load_document(updated);self._changed();self.status.setText('Alignment accepted and recorded. Review hydraulic values, then save and prepare.')

    def _import_guides(self):
        from osgeo import ogr
        filename,_=QFileDialog.getOpenFileName(self,'Import reference lines','','Vector lines (*.gpkg *.geojson *.json *.shp)')
        if not filename:return
        filename=self._local_input(filename,'vector');dataset=ogr.Open(str(filename),0)
        if dataset is None:raise ValueError('Cannot open line dataset.')
        if dataset.GetLayerCount()!=1:raise ValueError('Export one line layer before importing.')
        layer=dataset.GetLayer(0);transform=geometry_transform(layer.GetSpatialRef(),self.map.terrain.wkt);lines=[]
        for feature in layer:
            geom=feature.GetGeometryRef()
            if geom is None:continue
            geom=geom.Clone()
            if transform:geom.Transform(transform)
            kind=ogr.GT_Flatten(geom.GetGeometryType())
            segments=[geom] if kind==ogr.wkbLineString else [geom.GetGeometryRef(i) for i in range(geom.GetGeometryCount())] if kind==ogr.wkbMultiLineString else []
            if not segments:raise ValueError('Every feature must be a LineString or MultiLineString.')
            lines.extend([[[float(x),float(y)] for x,y,*_ in segment.GetPoints()] for segment in segments])
        self.map.set_sketch_lines(self.map.sketch_lines+lines);self._changed()

    def _export_domain(self):
        from osgeo import ogr
        from ._avac.geo import spatial_reference
        domain=PolygonDomain(**self._collect(validate=False)['settings']['domain'])
        path,_=QFileDialog.getSaveFileName(self,'Export domain geometry','','GeoJSON (*.geojson)')
        if not path:return
        if Path(path).exists():raise ValueError('Choose a new export filename; existing geometry is not overwritten.')
        driver=ogr.GetDriverByName('GeoJSON');ds=driver.CreateDataSource(path);layer=ds.CreateLayer('BRAIDS domain',srs=spatial_reference(domain.crs_wkt),geom_type=ogr.wkbPolygon)
        ring=ogr.Geometry(ogr.wkbLinearRing)
        for x,y in domain.vertices_xy:ring.AddPoint_2D(x,y)
        poly=ogr.Geometry(ogr.wkbPolygon);poly.AddGeometry(ring);feature=ogr.Feature(layer.GetLayerDefn());feature.SetGeometry(poly);layer.CreateFeature(feature);ds=None
        self.status.setText('Domain exported')

    def _preview_survey(self):
        survey=self.forms.physics.bank_survey.draft()
        if not survey:raise ValueError('Import or draw a bank polygon and matching toe first.')
        from shiboken6 import isValid
        for item in getattr(self,'_bank_items',[]):
            if isValid(item) and item.scene() is not None:self.map.scene().removeItem(item)
        self._bank_items=[]
        for zone in survey['zones']:
            path=QPainterPath();path.setFillRule(Qt.OddEvenFill)
            for ring in [zone['shell_xy'],*zone['holes_xy']]:
                path.moveTo(self.map.to_scene(*ring[0]))
                for point in ring[1:]:path.lineTo(self.map.to_scene(*point))
                path.closeSubpath()
            pen=QPen(QColor(geometry_color()),2);pen.setCosmetic(True)
            item=self.map.scene().addPath(path,pen);item.setZValue(28);item.setToolTip('Bank '+zone['id']);self._bank_items.append(item)
            toe=zone['toe_xy'];path=QPainterPath(self.map.to_scene(*toe[0]))
            for point in toe[1:]:path.lineTo(self.map.to_scene(*point))
            pen=QPen(QColor(geometry_color()),3);pen.setCosmetic(True)
            item=self.map.scene().addPath(path,pen);item.setZValue(29);item.setToolTip('Toe '+zone['id']);self._bank_items.append(item)
        self.status.setText(f"Survey preview: {len(survey['zones'])} bank zone(s). The prepared face mapping is created during Prepare.")

    def _begin_bank_drawing(self,kind):
        if self.runner.is_running or self._tasks:self.error('Finish the current operation before editing.');return
        if not self.map.terrain:self.error('Import the DEM first.');return
        self.map.begin_capture(kind)
        self.map_hint.setText('Draw the bank polygon.' if kind=='bank_polygon' else 'Draw its matching physical toe line.')

    def _bank_geometry_drawn(self,kind,points):
        try:
            if self.runner.is_running or self._tasks:raise ValueError('Finish the current operation before editing geometry.')
            if kind=='turbine':self.forms.turbines.place(points);return
            if kind.startswith('zone:'):
                key=kind.split(':')[1];doc=self._collect(validate=False);rows=doc['material_zones'][key];names={r['id'] for r in rows};i=1
                while f'{key} {i}' in names:i+=1
                value=({'roughness.chezy':self.forms.chezy,'roughness.manning':self.forms.manning,'roughness.white_colebrook':self.forms.nikuradse}[self.forms.roughness.currentData()].value()
                       if key=='roughness' else float(self.forms.physics.layers.item(0,0).text()))
                rows.append(dict(id=f'{key} {i}',value=value,points_xy=np.asarray(points,float).tolist()))
                from braids_core.river_material_zones import validate
                validate(doc['material_zones']);self._load_document(doc);self._changed();return
            if kind in ('solid_polygon','solid_ellipse'):
                from braids_core.boundaries.solids import validate_solids
                doc=self._collect(validate=False);rows=doc['settings']['solids'];names={r['id'] for r in rows};i=1
                while f'Solid {i}' in names:i+=1
                rows.append(dict(id=f'Solid {i}',points_xy=np.asarray(points,float).tolist()))
                validate_solids(rows,PolygonDomain(**doc['settings']['domain']))
                self._load_document(doc);self._changed();self.status.setText('Solid added. Generate the mesh to verify cropped wall cells.');return
            if kind.startswith('breakline_'):
                if kind!='breakline_line':raise ValueError('Breaklines must be drawn as polylines, not polygons.')
                doc=self._collect(validate=False);rows=doc['settings']['breaklines'];names={r['id'] for r in rows};i=1
                while f'Breakline {i}' in names:i+=1
                rows.append(dict(id=f'Breakline {i}',closed=False,points_xy=np.asarray(points,float).tolist()))
                from braids_core.boundaries.breaklines import validate_breaklines
                validate_breaklines(rows,PolygonDomain(**doc['settings']['domain']))
                self._load_document(doc);self._changed();self.status.setText('Breakline added. Generate the mesh to verify minimum areas.');return
            result=self.forms.physics.bank_survey.receive_drawn(kind,points)
            if result:self._preview_survey()
            else:self.status.setText('First geometry captured. Draw the matching bank polygon / toe line to complete the zone.')
        except Exception as exc:self.error(exc)

    def _zone_edited(self,kind,name,points):
        try:
            if self.runner.is_running or self._tasks:raise ValueError('Finish the current operation before editing material zones')
            doc=self._collect(validate=False)
            for row in doc['material_zones'][kind]:
                if row['id']==name:row['points_xy']=np.asarray(points,float).tolist()
            from braids_core.river_material_zones import validate
            validate(doc['material_zones'])
            self.forms.zones[kind].load(doc['material_zones'][kind]);self.document_base['material_zones']=deepcopy(doc['material_zones']);self._changed()
        except Exception as exc:
            self.map.end_zone_edit();self.map.set_zones(self._collect(validate=False)['material_zones']);self.error(exc)

    def _breakline_edited(self,name,points):
        doc=self._collect(validate=False);before=deepcopy(doc['settings']['breaklines'])
        try:
            if self.runner.is_running or self._tasks:raise ValueError('Finish the current operation before editing breaklines.')
            for row in doc['settings']['breaklines']:
                if row['id']==name:row['points_xy']=np.asarray(points,float).tolist()
            from braids_core.boundaries.breaklines import validate_breaklines
            validate_breaklines(doc['settings']['breaklines'],PolygonDomain(**doc['settings']['domain']))
            self.document_base['settings']['breaklines']=deepcopy(doc['settings']['breaklines']);self._changed()
        except Exception as exc:
            self.map.end_breakline_edit();self.map.set_breaklines(before);self._layer_visibility();self.error(exc)

    def _solid_edited(self,name,points):
        doc=self._collect(validate=False);before=deepcopy(doc['settings']['solids'])
        try:
            if self.runner.is_running or self._tasks:raise ValueError('Finish the current operation before editing solids.')
            for row in doc['settings']['solids']:
                if row['id']==name:row['points_xy']=np.asarray(points,float).tolist()
            from braids_core.boundaries.solids import validate_solids
            validate_solids(doc['settings']['solids'],PolygonDomain(**doc['settings']['domain']))
            self.document_base['settings']['solids']=deepcopy(doc['settings']['solids']);self._changed()
        except Exception as exc:
            self.map.end_solid_edit();self.map.set_solids(before);self._layer_visibility();self.error(exc)

    def edit_boundary(self,name):
        if self.runner.is_running or self._tasks or self.mode!='model_run':return
        try:
            original=self._collect(validate=False);editor=self.forms.boundary_editor
            if editor.boundary.findData(name)<0:return
            from .river_physics import set_choice
            set_choice(editor.boundary,name);self._select_step('BC & IC')
            dirty=self.dirty
            dialog=QDialog(self);dialog.setWindowTitle(tr('Boundary conditions')+' â€” '+name);dialog.resize(700,680)
            layout=QVBoxLayout(dialog);scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(editor);layout.addWidget(scroll)
            editor.show();editor.box.setExpanded(True);editor.boundary.setEnabled(False)
            buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel);layout.addWidget(buttons)
            def commit():
                try:self._collect(validate=False);dialog.accept()
                except Exception as exc:QMessageBox.warning(dialog,tr('Boundary conditions'),str(exc))
            buttons.accepted.connect(commit);buttons.rejected.connect(dialog.reject)
            accepted=dialog.exec()==QDialog.Accepted
            scroll.takeWidget();editor.setParent(self.forms);editor.hide();editor.boundary.setEnabled(True)
            if accepted:self._changed()
            else:
                editor.load_project(original);self.forms._sync_boundaries()
                self.dirty=dirty;self._title()
        except Exception as exc:self.error(exc)

    def set_background(self,provider,*,persist=True):
        provider=provider if provider in ('none','osm','satellite') else 'none'
        self.map.basemap.set_provider(provider)
        if hasattr(self,'background_actions'):
            self.background_actions[provider].setChecked(True)
        self.map_background.blockSignals(True);self.map_background.setCurrentIndex(self.map_background.findData(provider));self.map_background.blockSignals(False)
        if persist:settings().setValue('map/background',provider)

    def edit_settings(self):
        from .settings_dialog import SettingsDialog
        from ._avac.theme import apply_theme
        dialog=SettingsDialog(self)
        if dialog.exec()!=QDialog.Accepted:return
        dialog.save_preferences();apply_separator(QApplication.instance(),dialog.separator.currentData())
        from .presentation import NumericLineEdit
        for widget in self.findChildren(NumericLineEdit):widget.setText(widget.canonical_text())
        apply_theme(QApplication.instance(),dialog.theme.currentData());self.map.apply_appearance(dialog.theme.currentData())
        self.map.refresh_geometry_color();self.map.refresh_mesh_color();self.map.viewport().update();self.set_background(dialog.background.currentData())
        self.map.snap_enabled=snap_enabled();self.results.animate_mesh.setChecked(adaptive_mesh());self.results._draw_measurements()
        if self.results.particles:self.results.particles.clear();self.results._update_particles()

    def toggle_theme(self):
        from ._avac.theme import apply_theme
        mode='light' if QApplication.instance().property('appearance')=='dark' else 'dark'
        settings().setValue('interface/appearance',mode);apply_theme(QApplication.instance(),mode);self.map.apply_appearance(mode)

    def quick_start(self):
        QMessageBox.information(self,'Quick start',
            '1. Open demo, or import a DEM and draw/import one domain polygon.\n'
            '2. Select domain edges. Assign named inlets/outlets and their hydraulic conditions. All other edges are walls.\n'
            '3. Choose mesh strategy, then CPU or GPU and duration in Run. Fresh runs start dry.\n'
            '4. Save a .braids.json project file; its data folder is created alongside. Run prepares automatically when needed.\n'
            '5. Results appear on the map. Use Results for animation, probes, transects and export.\n\n'
            'Sediment is optional and off by default. Advanced > Sediment transport opens 04 Transport, 05 Morpho evolution and 06 Sediment transport BC. '
            'Select bedload, suspended mass or both explicitly in 04 / General. Prepare checks combinations before building the mesh and shows one correction at a time.\n'
            'Right-click exits map editing. Ctrl-click selects several edges. All outputs and history remain in the project folder.')

    def closeEvent(self,event):
        if self.runner.is_running:
            answer=QMessageBox.question(self,'Solver is running','Request cancellation and keep this window open until the worker stops?',QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)
            if answer==QMessageBox.Yes:self.runner.stop()
            event.ignore();return
        if self._tasks:event.ignore();self.status.setText('Please wait for the current file operation to finish.');return
        if self.dirty and QMessageBox.question(self,'Unsaved changes','Close without saving changes?',QMessageBox.Discard|QMessageBox.Cancel,QMessageBox.Cancel)!=QMessageBox.Discard:event.ignore();return
        event.accept()
