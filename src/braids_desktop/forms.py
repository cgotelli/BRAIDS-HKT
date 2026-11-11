"""Complete standalone model forms backed by the existing strict river resolver.

Map/project lifecycle belongs to the shell. This widget owns model controls,
method-aware editors and their inactive drafts, never a second solver route.
"""
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QStackedWidget,
    QScrollArea,QLabel,QLineEdit,QCheckBox,QSpinBox,QPushButton,QTableWidget,
    QTableWidgetItem,QHeaderView,QTabWidget,QPlainTextEdit,QFileDialog,QComboBox,QSizePolicy)
from braids_core.boundaries.domain import PolygonDomain, BoundaryAssignment
from braids_core.boundaries.inlet_io import NativeGrid
from braids_core.river_project import (SCHEMA,BEDFORM_METHODS,validate_project,checkpoint_editor_state,
    DEFAULT_OUTPUT_FIELDS,OUTPUT_FIELD_IDS,output_fields)
from braids_core.river_model import SURVEY_BANKS
from braids_core.sediment.current_river_solver import CurrentRiverControls
from braids_core.sediment.bed_columns_v101 import ColumnMethodsV101
from braids_core.sediment.river_field_policy import FieldSettings
from .river_physics import RiverPhysicsEditor,WrappedNote,combo,number,section,set_choice
from .river_boundaries import RiverBoundaryEditor
from .river_forcing import RiverForcingEditor
from ._avac.i18n import tr
from ._avac.number_widgets import NumericDelegate
from .presentation import display_values
from . import sediment_workflow


class SettingsPanel(QWidget):
    changed = Signal()
    checkpoint_selected=Signal(str)
    saved_state_check=Signal(bool)
    geometry_action = Signal(str)
    step_names = ('Domain','Mesh','BC & IC','Hydraulic structures','Advanced hydrodynamics','Hydrokinetic turbines',*sediment_workflow.STEPS,'Run')

    def __init__(self,parent=None):
        super().__init__(parent)
        self._loading=True;self._base=None;self._grid_override=None;self._dry_stage_m=-1e7
        self.output_fields=set(DEFAULT_OUTPUT_FIELDS)
        self.localize_source=lambda path,kind='file':path
        self.stack=QStackedWidget();outer=QVBoxLayout(self);outer.setContentsMargins(0,0,0,0)
        outer.addWidget(self.stack);self.pages={};self.layouts={}
        for name in self.step_names:
            page=QWidget();layout=QVBoxLayout(page)
            scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(page)
            self.stack.addWidget(scroll);self.pages[name]=page;self.layouts[name]=layout
        general=self.layouts['Domain']
        self.actions(general,[('Import DEM','import_dem'),('Import domain','import_domain')])
        self.general_form=QFormLayout();general.addLayout(self.general_form)
        self.name=QLineEdit('BRAIDS river',self);self.name.hide()
        self.dem_path=QLineEdit(self);self.dem_path.setReadOnly(True);self.dem_path.hide()
        self.crs=QLineEdit(self);self.crs.setReadOnly(True);self.crs.hide()
        self.backend=combo([('CPU / OpenMP','cpu'),('GPU / CUDA','gpu')])
        self.sediment_enabled=QCheckBox('Enable sediment transport',self);self.sediment_enabled.hide()
        self.sediment_enabled.setToolTip('Optional extension; unchecked for hydrodynamics only.')
        self.sediment_enabled.setChecked(False)
        self.roughness=combo([('Manning','roughness.manning'),('Chézy','roughness.chezy'),
                              ('White–Colebrook','roughness.white_colebrook')])
        from braids_core.river_jiang_cao import ROUGHNESS as JC_ROUGHNESS
        self.roughness.addItem('Automatic Sun/van Rijn (Jiang–Cao)', JC_ROUGHNESS)
        self._prescribed_roughness = 'roughness.manning'
        self.roughness_note=QLabel();self.roughness_note.setWordWrap(True)
        self.chezy=number(45.,1e-6,1e4,6);self.manning=number(.03,.0001,10.)
        self.nikuradse=number(.003,.000001,1e3)
        self.white_relation=combo([('FLOW log layer','flow_log_layer'),('Classical 2D','classical_2d')])
        for label,w in [('Hydraulic roughness',self.roughness),
                ('Chézy C (m½/s)',self.chezy),('Manning n (s/m⅓)',self.manning),
                ('Roughness height ks (m)',self.nikuradse),('White–Colebrook relation',self.white_relation)]:
            self.general_form.addRow(label,w)
        self.general_form.addRow(self.roughness_note)
        advanced=QFormLayout();self.layouts['Advanced hydrodynamics'].addLayout(advanced)
        self.secondary=QCheckBox('Enable secondary flow');self.betac=number(1.,0.,1.,6)
        self.secondary_diffusion=number(0.);self.turbulence=QCheckBox('Enable horizontal turbulent viscosity')
        self.viscosity=number(0.)
        for label,w in [('',self.secondary),('Secondary momentum coefficient',self.betac),
                ('Secondary diffusivity (m²/s)',self.secondary_diffusion),('',self.turbulence),('Eddy viscosity (m²/s)',self.viscosity)]:
            advanced.addRow(label,w)
        self.note(general,'CPU and GPU offer the same hydraulic methods, selected in Run. '
            'Hydrodynamics-only fixes the bed and executes no sediment processes. Backend selection is always explicit.')
        from .material_zones import MaterialZones
        self.zones={};self.zone_boxes={}
        for kind in ('roughness','erodible'):
            box,zoneform=section(general,'Roughness polygons' if kind=='roughness' else 'Erodible zones',expanded=False)
            self.zone_boxes[kind]=box
            widget=MaterialZones(kind);zoneform.addRow(widget);self.zones[kind]=widget
            widget.changed.connect(self._changed);widget.action.connect(self.geometry_action.emit)
        general.addStretch()

        mesh=self.layouts['Mesh'];form=QFormLayout();mesh.addLayout(form)
        self.minimum_size=number(5.,.0001,1e6,6);self.levels=QSpinBox();self.levels.setRange(1,6);self.levels.setValue(3)
        self.mesh_mode=combo([('Static mesh — manual refinement','static_manual'),
                              ('Static mesh — terrain slope breaks','static_terrain'),
                              ('Dynamic AMR — water depth only','dynamic_depth')])
        self.mesh_mode.setToolTip('Manual: fixed coarse background with breakline refinement and optional domain-border and inlet/outlet refinement; no slope or depth refinement. Solids, breaklines and enabled turbines retain their required local refinement.')
        for label,w in [('Mesh strategy',self.mesh_mode),('Smallest cell size (m)',self.minimum_size),('Number of 2:1 levels',self.levels)]:form.addRow(label,w)
        self.level_sizes=QLabel();self.level_sizes.setObjectName('note');self.level_sizes.setWordWrap(True)
        form.addRow('Resolved level sizes',self.level_sizes)
        self.actions(mesh,[('Add breakline','add_breakline'),('Edit breakline','edit_breakline')])
        self.actions(mesh,[('Import breaklines','import_breaklines')])
        self.breakline_list=QComboBox();mesh.addWidget(self.breakline_list)
        self.actions(mesh,[('Remove breakline','remove_breakline')])
        self.refinement_box,form=section(mesh,'Refinement controls',expanded=False)
        self.domain_border=QCheckBox('Refine domain border to smallest size');self.domain_border.setChecked(False)
        self.domain_border.setToolTip('Keep all outer domain-border cells at the finest level, even when dry. The inlet/outlet switch below controls their separate protection. Solids and breaklines keep their required refinement.')
        form.addRow(self.domain_border)
        self.open_boundary_refinement=QCheckBox('Refine inlets/outlets to smallest size');self.open_boundary_refinement.setChecked(True)
        self.open_boundary_refinement.setToolTip('On: keep inlet/outlet cells at the smallest river size, including dry entrances. Off: boundary cells follow the selected mesh strategy and may remain coarse. Wet-depth AMR, domain-border refinement and local constraints can still refine them. Hydraulic and sediment boundary laws are unchanged.')
        form.addRow(self.open_boundary_refinement)
        self.adaptation={}
        labels=dict(interval='Assessment interval (s)',coarsen_delay='Dry assessments before coarsening',
            buffer_cells='Wet-front buffer (neighbour rings)',transition_layers='Transition rows per refinement level',
            terrain_angle_deg='Slope change (degrees)',terrain_detail_m='Terrain detail threshold (m)')
        for key,value in asdict(FieldSettings()).items():
            if key in ('terrain_enabled','domain_border_enabled','open_boundary_enabled'):continue
            w=QSpinBox() if type(value)is int else number(value,1e-9,1e7)
            if type(value)is int:w.setRange(1,10000);w.setValue(value)
            if key=='terrain_angle_deg':w.setMaximum(89.999)
            if key=='transition_layers':
                w.setMaximum(32)
                w.setToolTip('Minimum face-neighbour rows at each level before a coarser band. 1 retains the existing 2:1 balance. Applies to static and dynamic meshes; more rows add cells. Domain edges and nearby constraints can truncate/overlap bands. This is separate from the wet-front buffer.')
            self.adaptation[key]=w;form.addRow(labels[key],w)
        mesh.addStretch()

        boundaries=self.layouts['BC & IC']
        title=QLabel('Boundaries');title.setObjectName('Title');boundaries.addWidget(title)
        form=QFormLayout();boundaries.addLayout(form)
        self.boundary_name=QLineEdit('inlet');self.boundary_kind=combo([('Inlet','inlet'),('Outlet','outlet'),('Wall','wall')])
        form.addRow('Boundary name',self.boundary_name);form.addRow('Boundary type',self.boundary_kind)
        self.actions(boundaries,[('Select edge','select_boundary_edge'),('Clear named boundary','clear_boundary')])
        self.boundary_table=QTableWidget(0,3);self.boundary_table.setHorizontalHeaderLabels(['Assigned boundary','Type','Condition'])
        header=self.boundary_table.horizontalHeader()
        header.setSectionResizeMode(0,QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1,QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2,QHeaderView.Stretch)
        self.boundary_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.boundary_table.verticalHeader().setDefaultSectionSize(28)
        self.boundary_table.verticalHeader().sectionCountChanged.connect(lambda *_:self._resize_boundary_table())
        self._resize_boundary_table();boundaries.addWidget(self.boundary_table)
        structures=self.layouts['Hydraulic structures']
        title=QLabel('Internal conditions');title.setObjectName('Title');structures.addWidget(title)
        self.solid_shape=combo([('Polygon','polygon'),('Ellipse','ellipse')]);structures.addWidget(self.solid_shape)
        self.solid_shape.setToolTip('Ellipses are stored as 64-sided wall polygons. Click two opposite corners to draw one.')
        self.actions(structures,[('Add solid','add_solid'),('Edit solid','edit_solid')])
        self.solid_list=QComboBox();structures.addWidget(self.solid_list)
        self.actions(structures,[('Remove solid','remove_solid')]);structures.addStretch()
        self.boundary_table.cellClicked.connect(self._select_boundary)
        self.boundary_table.cellDoubleClicked.connect(lambda row,col:self.geometry_action.emit('edit_boundary:'+self.boundary_table.item(row,0).text()))
        self.boundary_editor=RiverBoundaryEditor(self);self.boundary_editor.hide()
        title=QLabel('Initial conditions');title.setObjectName('Title');boundaries.addWidget(title)
        form=QFormLayout();boundaries.addLayout(form)
        self.initial_form=form
        self.initial_mode=combo([('Dry bed','dry'),('Uniform water elevation and velocity','stage'),('Uniform water depth and velocity','depth')])
        self.stage=number(0.,-1e12,1e12);self.u=number(0.,-1e4,1e4);self.v=number(0.,-1e4,1e4)
        self.initial_depth=number(0.,0.,1e7,6)
        form.addRow('Initial water',self.initial_mode)
        form.addRow('Water elevation (m, DEM datum)',self.stage)
        form.addRow('Uniform water depth (m)',self.initial_depth)
        form.addRow('East / X velocity (m/s)',self.u);form.addRow('North / Y velocity (m/s)',self.v)
        self.note(boundaries,'Uniform elevation gives depth = max(elevation − local bed, 0). Uniform depth uses the same depth in every cell. Velocity applies to wet cells. A checkpoint supplies its saved fields instead.')
        self.preview_mesh_button=QPushButton('Generate and Visualize mesh');self.preview_mesh_button.setObjectName('Primary')
        self.preview_mesh_button.clicked.connect(lambda:self.geometry_action.emit('preview_mesh'));boundaries.addWidget(self.preview_mesh_button)
        boundaries.addStretch()

        self.physics=RiverPhysicsEditor(self)
        self.physics.simple_bedload=True
        self.physics.enabled.hide()
        # The core permits an inherited or independent fixed-layer target.
        # Keep that explicit identity when opening a current-schema project.
        self.eulerian_override=QCheckBox('Separate fixed-layer thickness target')
        self.eulerian_thickness=number(.2,1e-9,1e6)
        inventory_form=self.physics.inventory_box.body_layout.itemAt(0).layout()
        inventory_form.addRow(self.eulerian_override)
        inventory_form.addRow('Fixed-layer target (m)',self.eulerian_thickness)
        inventory_form.labelForField(self.physics.substrate_thickness).setText('Substrate target (m)')
        self.physics.bank_survey.domain_crs=lambda:self.crs.text()
        self.physics.bank_survey.preview_requested.connect(lambda:self.geometry_action.emit('bank_preview'))
        sediment_workflow.build(self)
        self.sediment_enabled.toggled.connect(self.physics.enabled.setChecked)
        self.forcing=RiverForcingEditor(self);self.layouts['Sediment transport BC'].addWidget(self.forcing)
        self.forcing.initial_box.hide()
        self.layouts['Sediment transport BC'].addStretch()
        from .turbines import TurbineEditor
        self.turbines=TurbineEditor(self);self.layouts['Hydrokinetic turbines'].addWidget(self.turbines)
        self.turbines.action.connect(self.geometry_action.emit);self.turbines.changed.connect(self._changed)

        run=self.layouts['Run'];self.run_form=QFormLayout();run.addLayout(self.run_form)
        from braids_core.river_model import INTEGRATOR,HLLC_INTEGRATOR
        self.hydraulic_method=combo([('GeoClaw augmented f-wave',INTEGRATOR),('Fast HLLC (2nd order)',HLLC_INTEGRATOR)])
        self.hydraulic_method.setToolTip('Fast HLLC uses compact MUSCL reconstruction and an explicit MUSCL-Hancock half-time predictor. Ordinary bedload and suspended formulas share the conservative sediment transaction; their existing physical compatibility rules still apply. The coupled Jiang–Cao mixture model requires f-wave. Hydraulic/bed splitting is first order in time.')
        self.end_time=number(60.,.000001,1e12);self.dt_max=number(2.,.000001,1e9)
        self.output_interval=number(1.,.000001,1e12);self.checkpoint_interval=number(60.,.000001,1e12)
        self.checkpoint_duration=QCheckBox('Use simulation duration');self.checkpoint_duration.setChecked(True)
        self._checkpoint_custom_interval=60.
        checkpoint_row=QWidget();layout=QVBoxLayout(checkpoint_row);layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.checkpoint_interval);layout.addWidget(self.checkpoint_duration)
        self.threads=QSpinBox();self.threads.setRange(1,max(256,os.cpu_count()or 1));self.threads.setValue(min(8,os.cpu_count()or 1))
        self.morfac=number(1.,.001,1e6);self.cfl=number(1.,.001,1.,6)
        self.sediment_general_form.addRow('Morphological factor',self.morfac)
        self.morfac.setToolTip('Multiplies bedload divergence and wet bed/suspension exchange. Water and suspended transport remain on hydraulic time; dry stranding is unscaled. Extra bed mass is ledgered explicitly. Jiang-Cao requires MF=1. Large factors can change the coupled solution; compare with a smaller factor.')
        self.restart=QLineEdit();restart_row=QWidget();layout=QHBoxLayout(restart_row);layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.restart);browse=QPushButton('Browse…');layout.addWidget(browse);browse.clicked.connect(self._browse_restart)
        self.restart_mode=combo([('Load useful fields (changed setup allowed)','initialize'),('Exact continuation (same setup / backend)','exact')])
        state_row=QWidget();state_layout=QVBoxLayout(state_row);state_layout.setContentsMargins(0,0,0,0)
        self.inspect_state=QPushButton('Check saved state against prepared mesh');self.map_state=QPushButton('Map saved state to prepared mesh');self.map_state.setEnabled(False)
        self.state_summary=QLabel('Saved-state initialization retains the checkpoint time. Water and bed are loaded; compatible active fields are retained. A changed mesh requires an explicit mapping step.');self.state_summary.setWordWrap(True)
        for w in (self.inspect_state,self.map_state,self.state_summary):state_layout.addWidget(w)
        self.inspect_state.clicked.connect(lambda:self.saved_state_check.emit(False));self.map_state.clicked.connect(lambda:self.saved_state_check.emit(True))
        self.restart.textChanged.connect(self._restart_changed)
        for label,w in [('Hydraulic method',self.hydraulic_method),('Compute backend',self.backend),('End time (s)',self.end_time),('Maximum outer step (s)',self.dt_max),
                ('Solution output interval (s)',self.output_interval),('Restart checkpoint interval (s)',checkpoint_row),
                ('CPU threads',self.threads),('CFL (face-sum)',self.cfl),
                ('Saved checkpoint / state (optional)',restart_row),('Checkpoint use',self.restart_mode),('Saved-state transfer',state_row)]:self.run_form.addRow(label,w)
        self.dt_max.setToolTip('Upper bound only. Hydraulic CFL, positivity and physical forcing may shorten accepted steps.')
        self.checkpoint_interval.setToolTip('Saves at accepted states; does not shorten numerical steps. Initial, final and cancelled states are always saved.')
        self.checkpoint_duration.setToolTip('Initial/final/cancelled recovery only by default. An abrupt crash before the end can lose the whole run.')
        self.cfl.setToolTip('Explicit hydraulic face-sum CFL ≤ 1 on both backends. Positivity, diffusion and sediment bed-change guards can still shorten steps.')
        self.admission=QLabel();self.admission.setObjectName('note');self.admission.setWordWrap(True);run.addWidget(self.admission)
        self.summary=QPlainTextEdit(self);self.summary.setReadOnly(True);self.summary.hide()
        run.addStretch()

        self.end_time.valueChanged.connect(self._end_changed)
        self.checkpoint_interval.valueChanged.connect(self._custom_checkpoint_changed)
        self.checkpoint_duration.toggled.connect(self._checkpoint_mode_changed)
        self.minimum_size.valueChanged.connect(self._mesh_changed);self.levels.valueChanged.connect(self._mesh_changed)
        for w in [self.hydraulic_method,self.backend,self.roughness,self.white_relation,self.mesh_mode,self.initial_mode]:w.currentIndexChanged.connect(self._changed)
        for w in [self.chezy,self.manning,self.nikuradse,self.betac,self.secondary_diffusion,self.viscosity,
                  self.stage,self.initial_depth,self.u,self.v,self.end_time,self.dt_max,self.output_interval,self.checkpoint_interval,
                  self.threads,self.morfac,self.cfl,*self.adaptation.values()]:w.valueChanged.connect(self._changed)
        for w in [self.secondary,self.turbulence,self.domain_border,self.open_boundary_refinement]:w.toggled.connect(self._changed)
        self.eulerian_override.toggled.connect(self._changed);self.eulerian_thickness.valueChanged.connect(self._changed)
        self.name.textChanged.connect(self._changed)
        self.physics.changed.connect(self._changed);self.boundary_editor.changed.connect(self._changed);self.forcing.changed.connect(self._changed)
        from .accordion import initialize_sections
        for page in self.pages.values():initialize_sections(page)
        for tabs in (self.sediment_tabs,self.morphology_tabs):
            for i in range(tabs.count()):initialize_sections(tabs.widget(i))
        self.refinement_box.setExpanded(False);self.physics.material_box.setExpanded(False)
        for box in self.zone_boxes.values():box.setExpanded(False)
        self._loading=False;self._mesh_changed();self._restart_changed(self.restart.text())

    def note(self,layout,text):
        label=QLabel(text);label.setObjectName('note');label.setWordWrap(True);layout.addWidget(label);return label

    def actions(self,layout,entries):
        short={'Select domain edges':'Select edge',
               'Apply to selected edges':'Apply selected','Clear named boundary':'Delete boundary',
               'Review inlet/outlet alignment…':'Review boundary alignment'}
        for start in range(0,len(entries),2):
            row=QHBoxLayout()
            for title,action in entries[start:start+2]:
                button=QPushButton(short.get(title,title).replace('…','').replace('...','').replace('_',' '));button.setToolTip(title)
                button.clicked.connect(lambda _,name=action:self.geometry_action.emit(name));row.addWidget(button)
            layout.addLayout(row)

    def _resize_boundary_table(self):
        table=self.boundary_table;rows=table.rowCount();default=table.verticalHeader().defaultSectionSize()
        body=sum(table.rowHeight(i) for i in range(rows))+max(0,2-rows)*default
        table.setFixedHeight(table.horizontalHeader().sizeHint().height()+body+2*table.frameWidth()+4)

    def _compact_layouts(self):
        # The standalone sidebar is narrower than the former QGIS dock. Keep
        # complete labels above controls instead of pushing fields offscreen.
        # Reapply after dynamic method editors create their parameter rows.
        for form in self.findChildren(QFormLayout):
            form.setRowWrapPolicy(QFormLayout.WrapAllRows)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        for label in self.findChildren(QLabel):
            label.setWordWrap(True)
            if not label.property('preserveMethodIds'):
                label.setText(tr(label.text().replace('_',' ')))
        for form in self.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                item=form.itemAt(row,QFormLayout.LabelRole)
                if item and isinstance(item.widget(),QLabel):item.widget().setWordWrap(False)
        for field in self.findChildren(QComboBox):
            field.setMinimumContentsLength(8)
            field.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            policy=field.sizePolicy();policy.setHorizontalPolicy(QSizePolicy.Ignored);field.setSizePolicy(policy)
        for table in self.findChildren(QTableWidget):
            if table is self.boundary_table:continue
            if not table.property('numericPresentation'):
                table.setItemDelegate(NumericDelegate(table));table.setProperty('numericPresentation',True)
            table.verticalHeader().setDefaultSectionSize(max(32,table.verticalHeader().defaultSectionSize()))
            for row in range(table.rowCount()):table.setRowHeight(row,max(32,table.rowHeight(row)))

    def set_step(self,index):
        self.stack.setCurrentIndex(self.step_names.index(index) if isinstance(index,str) else index)

    def _browse_restart(self):
        initial=self.restart.text()
        if not initial:
            root=getattr(self.window(),'project_root',None)
            if root is not None:initial=str(root/'checkpoints')
        filename,_=QFileDialog.getOpenFileName(self,'Select a sediment/hydraulic checkpoint to continue',initial,'Checkpoint (*.npz)')
        if filename:
            if self.restart_mode.currentData()=='initialize':self.checkpoint_selected.emit(filename)
            else:self.restart.setText(str(self.localize_source(filename)))

    def _mesh_changed(self,*_):
        if not self._loading:self._grid_override=None
        sizes=[self.minimum_size.value()*(2**level) for level in range(self.levels.value())]
        self.level_sizes.setText('Finest to coarsest: '+', '.join(f'{value:g} m' for value in sizes)+'.')
        self._changed()

    def _restart_changed(self,text):
        self.inspect_state.setEnabled(bool(text.strip()))
        if not text.strip():self.map_state.setEnabled(False)

    def _end_changed(self,*_):
        if not self._loading and self.checkpoint_duration.isChecked():
            self.checkpoint_interval.setValue(self.end_time.value())

    def _custom_checkpoint_changed(self,*_):
        if not self._loading and not self.checkpoint_duration.isChecked():
            self._checkpoint_custom_interval=self.checkpoint_interval.value()

    def _checkpoint_mode_changed(self,automatic):
        if self._loading:return
        if automatic:self._checkpoint_custom_interval=self.checkpoint_interval.value()
        self.checkpoint_interval.setValue(self.end_time.value() if automatic else self._checkpoint_custom_interval)
        self._changed()

    def boundary_kinds(self):
        return {a['bc_id']:a['kind'] for a in (self._base or {}).get('settings',{}).get('assignments',[]) if a['kind']!='wall'}

    def _select_boundary(self,row,*_):
        name=self.boundary_table.item(row,0).text();self.boundary_name.setText(name)
        set_choice(self.boundary_kind,self.boundary_table.item(row,1).text())
        if self.boundary_editor.boundary.findData(name)>=0:set_choice(self.boundary_editor.boundary,name)
        self.geometry_action.emit('select_boundary:'+name)

    @staticmethod
    def _boundary_condition_summary(doc,name,kind):
        """One short, read-only description of the resolved user condition."""
        if kind=='wall':return 'Wall'
        state=doc.get('boundary_states',{}).get(name,{})
        schedule=state.get('hydraulic_schedule',{})
        if schedule.get('method')!='forcing.constant':return 'Hydrograph'
        hydraulic=doc.get('settings',{}).get('hydraulics',{}).get(name,{})
        method=hydraulic.get('method','')
        def value(key):
            raw=hydraulic.get(key)
            if raw is None:return None
            try:return f'{float(raw):g}'
            except (TypeError,ValueError):return str(raw)
        discharge=value('discharge_m3_s')
        if discharge is not None:return f'Discharge = {discharge} m³/s'
        if method=='outlet.rating_curve':return 'Q–H relation'
        if method=='outlet.normal_depth':
            slope=value('bed_slope')
            return 'Normal depth' if slope is None else f'Normal depth (S = {slope})'
        if method=='outlet.stage':
            stage=value('stage_m')
            return 'Water surface' if stage is None else f'Water surface = {stage} m'
        if method=='outlet.critical':return 'Critical depth'
        if method=='outlet.supercritical':return 'Supercritical outflow'
        return method.removeprefix('inlet.').removeprefix('outlet.').replace('_',' ').capitalize() or 'Configured'

    def _sync_boundaries(self):
        p=self.physics
        grains=[tuple(p.grains.item(r,c).text() if p.grains.item(r,c) else '' for c in (0,1)) for r in range(p.grains.rowCount())]
        # Hydro uses one explicitly inert carrier, not the inactive grain draft.
        if not p.active:grains=[('inert','0.5')]
        ids=[g for g,_ in grains];valid=all(ids) and len(set(ids))==len(ids)
        self.forcing.setEnabled(valid)
        self.boundary_editor.sync(ids if valid else [],self.boundary_kinds(),enabled=p.active)
        if valid:self.forcing.sync(grains,self.boundary_kinds(),enabled=p.active,
            suspended=p.suspension_on.isChecked(),bedload=p.bedload_on.isChecked())

    def _changed(self,*_):
        if self._loading:return
        self._loading=True
        try:
            mode=self.initial_mode.currentData()
            self.stage.setVisible(mode=='stage');self.initial_depth.setVisible(mode=='depth')
            self.initial_form.labelForField(self.stage).setVisible(mode=='stage')
            self.initial_form.labelForField(self.initial_depth).setVisible(mode=='depth')
            for w in (self.u,self.v):
                w.setVisible(mode in ('stage','depth'))
                self.initial_form.labelForField(w).setVisible(mode in ('stage','depth'))
            sediment=self.physics.active;self.sediment_enabled.setChecked(self.physics.enabled.isChecked())
            layered=sediment and self.physics.bed_mode.currentData()=='bed.layered'
            self.eulerian_override.setEnabled(layered)
            self.eulerian_thickness.setEnabled(layered and self.eulerian_override.isChecked())
            self.morfac.setEnabled(sediment);self.boundary_editor.bed_box.hide()
            self.sediment_start.setEnabled(sediment)
            self.physics.layers.setEnabled(sediment)
            self.collapse_box.setEnabled(sediment)
            self.sediment_bc_button.setEnabled(sediment and bool(self.boundary_kinds()))
            self.threads.setEnabled(self.backend.currentData()=='cpu')
            self.betac.setEnabled(self.secondary.isChecked());self.secondary_diffusion.setEnabled(self.secondary.isChecked())
            self.boundary_editor.secondary_intensity.setVisible(self.secondary.isChecked())
            self.boundary_editor.secondary_label.setVisible(self.secondary.isChecked())
            self.viscosity.setEnabled(self.turbulence.isChecked());self.checkpoint_interval.setEnabled(not self.checkpoint_duration.isChecked())
            roughness=self.roughness.currentData()
            from braids_core.river_jiang_cao import METHOD as JC_METHOD, ROUGHNESS as JC_ROUGHNESS
            jc=sediment and self.physics.bedload_on.isChecked() and self.physics.capacity.currentData()==JC_METHOD
            if jc:
                if roughness != JC_ROUGHNESS:self._prescribed_roughness = roughness
                set_choice(self.roughness, JC_ROUGHNESS)
            elif roughness == JC_ROUGHNESS:
                set_choice(self.roughness, self._prescribed_roughness)
            else:self._prescribed_roughness = roughness
            roughness=self.roughness.currentData()
            self.roughness.model().item(self.roughness.findData(JC_ROUGHNESS)).setEnabled(jc)
            self.roughness_note.setText('Required by Jiang–Cao: resistance is computed from your D50, D90 and bedforms. Prescribed values below are inactive; roughness polygons are incompatible.' if jc else
                'This resistance choice is retained. Transport formulas use the corresponding local Chézy coefficient when needed; Manning is not replaced by a constant Chézy value.')
            self.secondary.setText('Enable finite-adjustment secondary flow (required by Jiang–Cao bundle)' if jc else 'Enable secondary flow')
            for control in (self.roughness,self.chezy,self.manning,self.nikuradse,self.white_relation,self.zones['roughness']):
                control.setEnabled(not jc)
            self.betac.setToolTip('Set to 0 for the Jiang–Cao bundle; its own momentum feedback is active.' if jc else '')
            label={'roughness.chezy':'Chézy C (m½/s)','roughness.manning':'Manning n (s/m⅓)','roughness.white_colebrook':'Roughness height ks (m)', JC_ROUGHNESS:'automatic Sun/van Rijn resistance (polygons unavailable)'}[roughness]
            self.zones['roughness'].label.setText('Default above; polygons override '+label+'. Edit values in the table. Overlapping polygons are rejected.')
            self.zones['erodible'].label.setText('Default in 05 / Morphological evolution; erodible zones override initial thickness (m). Zero is a non-erodible base that can receive deposits.')
            for field,show in [(self.chezy,roughness=='roughness.chezy'),(self.manning,roughness=='roughness.manning'),
                              (self.nikuradse,roughness=='roughness.white_colebrook'),(self.white_relation,roughness=='roughness.white_colebrook')]:
                field.setVisible(show);self.general_form.labelForField(field).setVisible(show)
            dynamic=self.mesh_mode.currentData()=='dynamic_depth'
            terrain=self.mesh_mode.currentData()=='static_terrain'
            for key,w in self.adaptation.items():w.setEnabled(key=='transition_layers' or (terrain if key.startswith('terrain_') else dynamic))
            self._sync_boundaries()
            try:
                model=self.hydraulic_model()
                self.summary.setPlainText(json.dumps(display_values(dict(physics=model['physics'],decisions=model['decisions'],physics_id=model['physics_id'])),indent=2))
                summary=self.method_summary(model)
                if self.physics.enabled.isChecked() and self.physics.bedload_on.isChecked() and self.physics.capacity_note.text() not in summary:
                    summary+='\n\nBedload formula details:\n'+self.physics.capacity_note.text()
                self.resolved_setup.setText(summary)
                self.admission.clear()
            except Exception as exc:
                self.admission.clear()
                self.resolved_setup.setText(self.physics.transport_setup_text() if
                    self.physics.enabled.isChecked() and self.physics.suspension_on.isChecked() else
                    'Configuration needs attention (first error):\n\n'+str(exc))
                self.summary.clear()
        finally:self._loading=False
        self._compact_layouts()
        self.changed.emit()

    def hydraulic_model(self):
        from braids_core.river_model import resolve_model
        model=self.physics.model(self.backend.currentData(),self.roughness.currentData(),self.secondary.isChecked(),self.turbulence.isChecked())
        request=deepcopy(model['request']);request['execution']['integrator']=self.hydraulic_method.currentData()
        return resolve_model(request).document()

    def method_summary(self,model=None):
        from braids_core.sediment.collapse_setup import summary as collapse_summary
        if self.physics.enabled.isChecked() and self.physics.suspension_on.isChecked():
            p=self.physics
            if p.suspended_formula.currentData()!='suspended.van_rijn_1993_flow2dh':
                return (p.transport_setup_text()+'\n\nRun controls: MF = '+str(self.morfac.value())+
                    '; sediment starts at '+str(self.sediment_start.value())+' s. Bed clock = MF × hydraulic time while sediment is active.')
            return (p.transport_setup_text()+'\n\nRun controls: MF = '+str(self.morfac.value())+
                '; sediment starts at '+str(self.sediment_start.value())+' s.\nHydraulic resistance: '+self.roughness.currentText()+
                '.\nWater temperature: '+str(p.temperature.value())+' °C; horizontal sediment diffusion: '+str(p.sus_diffusion.value())+
                ' m²/s; hindering reference: '+str(p.reference_concentration.value())+' kg/m³.\n'+
                'Van Rijn S3 resolves IopSus=0, IopKCW=1 (current hydraulic z0), AksFac=1, iTauCr=1, FactCr=1 and molecular Schmidt=1. '
                'Current-only coefficients and reference controls are saved explicitly; no wave calculation is activated.')
        if model is None:model=self.physics.model(self.backend.currentData(),self.roughness.currentData(),self.secondary.isChecked(),self.turbulence.isChecked())
        s=model['physics']['sediment']
        if s is None:return 'Hydrodynamics only. No sediment process is selected; opening this extension adds no physics.'
        p=self.physics
        if 'bundle' in s:
            b=s['bundle']
            return ('Jiang–Cao 3/2 research bundle · one bulk material, measured D50 and D90\n'
                'Non-equilibrium mobile bedload + conservative phase exchange, NOT capacity/Exner.\n'
                'Automatic resistance: '+b['roughness']+'\nSecondary: '+b['secondary']+
                '\nShallow front: '+b['shallow']+'\nCollapse: '+b['collapse']+
                '\nMF 1; initial secondary intensity 0; inlet qs/q is converted to mobile solid fraction.\n'
                'Hydraulic Q is MIXTURE volume flux; water flux = Q/(1 + qs/q).\n'
                'Equilibrium van Rijn bedform resistance is included; no separate bedform switch.\n'
                'Prescribed hydraulic roughness and classic FLOW Betac are not used by this bundle.')
        if getattr(p,'simple_bedload',False):
            sizes=', '.join(f'{k.upper() if k!="dm" else "Dm"}={v*1000:g} mm' for k,v in s['capacity']['grain_sizes_m'].items())
            from braids_core.river_model import HLLC_INTEGRATOR
            clock=('\nHLLC coupling: one accepted-endpoint conservative sediment update per outer step. '
                   'First-order hydraulic/bed time splitting; second-order spatial reconstruction. '
                   'Formula-specific compatibility rules remain active.\n'
                   if self.hydraulic_method.currentData()==HLLC_INTEGRATOR else '')
            stress=('Fixed 5 / 0.055 / 1.5; physical velocity and cell-depth hydraulic resistance, no extra form drag'
                    if s['capacity']['method']=='capacity.mpm_braids_5_055' else
                    'Local hydraulic bed shear, without MPM form-drag reduction'
                    if s['capacity']['method']=='capacity.wong_parker_2006' else
                    'FLOW effective sediment velocity/Chezy, without an additional MPM form-drag factor'
                    if s['capacity']['method']=='capacity.ashida_michiue_generalized' else
                    'FLOW legacy grain-friction threshold with independent characteristic-shear amplitude; suspended residual inactive'
                    if s['capacity']['method']=='capacity.wang_fredsoe' else
                    'FLOW effective sediment shear with MPM form-drag reduction'
                    if s['capacity']['method']=='capacity.mpm' else 'Formula-specific shear (see assumptions below)')
            coefficients=('\nSelected coefficients: '+', '.join(f'{k}={v:g}' for k,v in s['capacity']['parameters'].items())
                          if s['capacity']['method'] in ('capacity.ashida_michiue_generalized','capacity.wang_fredsoe') else '')
            return (f'{p.capacity.currentText()} bedload · one bulk material · {sizes}\nStarts at {self.sediment_start.value():g} s (hydraulics only beforehand)\n'
                    f'{stress}; Exner bed evolution; morphological factor {self.morfac.value():g}.{coefficients}\n'
                    'No suspension, bank erosion, sorting or bedform feedback.\n'+collapse_summary(s['avalanche'],slope=p.collapse_slope.value(),angle=p.repose_angle_deg.value())+'\n'
                    'Hydraulic secondary flow: '+model['physics']['hydrodynamics']['secondary_flow']+
                    '\n\nMethod assumptions and limitations:\n'+p.capacity_note.text()+
                    '\n\nMultiple diameter descriptors are not transported classes. No sorting or cohesive sediment. '+
                    'Suspension is OFF; the suspended formula draft does not affect this calculation.'+clock)
        names=[]
        if p.bedload_on.isChecked():names.append('algebraic total load' if s['transport']=='total_load' else 'bedload')
        if p.suspension_on.isChecked():names.append('suspended mass')
        lines=['Resolved processes: '+(' + '.join(names) or 'morphological redistribution only'),
            'Bedload factor: '+('1 (on)' if p.bedload_on.isChecked() else '0 (off; boundary feeds inactive)'),
            'Capacity: '+s['capacity']['method'],
            'Exchange: '+s['suspension'], 'Bed: '+s['bed']['composition'],
            'Banks: '+s['banks'], 'Collapse / avalanche: '+s['avalanche'],
            'Bedform feedback: '+s['bedforms'], 'Slope: '+s['bed_slope'],
            'Shear: '+s['shear'], 'Morphological factor: '+str(self.morfac.value())]
        lines.extend(d['path']+': '+str(d['resolved'])+' — '+d['reason'] for d in model['decisions'])
        return '\n'.join(lines)

    def geometry_grid(self,domain):
        if self._grid_override is not None:return deepcopy(self._grid_override)
        size=self.minimum_size.value()*2**(self.levels.value()-1);xs,ys=zip(*domain.vertices_xy)
        x,y=math.floor(min(xs)/size)*size,math.floor(min(ys)/size)*size
        return asdict(NativeGrid(math.ceil((max(xs)-x)/size),math.ceil((max(ys)-y)/size),x,y,size,size))

    def load_document(self,doc):
        self._loading=True
        try:
            self._base=deepcopy(doc);p,h=doc['settings'],doc['model']['request']['hydrodynamics']
            # Opening an older project in the editor adopts the current
            # explicit default when it is next saved; untouched historic runs
            # remain readable with their original full field inventory.
            self.output_fields=set(output_fields(p,legacy=False))
            self.turbines.load(doc.get('power_plan',{}).get('layout',doc['turbines']),doc.get('editor_state',{}).get('turbines'))
            self.turbines.power_planning.setChecked(False)
            for kind,widget in self.zones.items():widget.load(doc['material_zones'][kind])
            self.name.setText(doc['name']);self.dem_path.setText(doc['terrain']['dem_path'])
            crs=p['domain']['spatial_reference'];display=crs
            from osgeo import osr
            reference=osr.SpatialReference()
            if reference.SetFromUserInput(crs)==0:
                authority,code=reference.GetAuthorityName(None),reference.GetAuthorityCode(None)
                if authority and code:display=f'{authority}:{code}'
            self.crs.setText(display);self.crs.setToolTip(crs)
            set_choice(self.backend,doc['model']['request']['execution']['backend']);set_choice(self.roughness,h['roughness'])
            set_choice(self.hydraulic_method,doc['model']['request']['execution']['integrator'])
            self.secondary.setChecked(h['secondary_flow']!='secondary.off');self.turbulence.setChecked(h['turbulence']!='off')
            for w,key in [(self.chezy,'chezy'),(self.betac,'betac'),(self.viscosity,'viscosity'),(self.secondary_diffusion,'diffusivity'),
                          (self.threads,'threads'),(self.morfac,'morfac'),(self.cfl,'cfl')]:w.setValue(p['controls'][key])
            self.manning.setValue(doc['roughness_parameters']['manning_s_m_third']);self.nikuradse.setValue(doc['roughness_parameters']['nikuradse_m'])
            set_choice(self.white_relation,doc['roughness_parameters']['white_colebrook_relation'])
            self.levels.setValue(p['levels']);self.minimum_size.setValue(p['grid']['dx_m']/2**(p['levels']-1));self._grid_override=deepcopy(p['grid'])
            self.breakline_list.clear()
            for row in p['breaklines']:self.breakline_list.addItem(row['id'],row['id'])
            self.solid_list.clear()
            for row in p['solids']:self.solid_list.addItem(row['id'],row['id'])
            for key,w in self.adaptation.items():w.setValue(p['adaptation'][key])
            self.domain_border.setChecked(p['adaptation']['domain_border_enabled'])
            self.open_boundary_refinement.setChecked(p['adaptation'].get('open_boundary_enabled',True))
            set_choice(self.mesh_mode,'dynamic_depth' if p['adaptive'] else
                       'static_terrain' if p['adaptation']['terrain_enabled'] else 'static_manual')
            self.stage.setValue(doc['initial']['stage_m']);self.u.setValue(doc['initial']['velocity_m_s'][0]);self.v.setValue(doc['initial']['velocity_m_s'][1])
            self.initial_depth.setValue(doc['initial'].get('depth_m',0.))
            set_choice(self.initial_mode,'depth' if 'depth_m' in doc['initial'] else 'dry' if doc['initial']['stage_m']==-1e7 else 'stage')
            # Retain the existing explicit dry-stage document convention.
            self._dry_stage_m=float(doc['initial']['stage_m'])
            for w,key in [(self.end_time,'end_time_s'),(self.dt_max,'dt_max_s'),(self.output_interval,'output_interval_s'),(self.checkpoint_interval,'checkpoint_interval_s'),(self.sediment_start,'sediment_start_time_s')]:w.setValue(p[key])
            checkpoint=checkpoint_editor_state(doc);self._checkpoint_custom_interval=checkpoint['custom_interval_s']
            self.checkpoint_duration.setChecked(checkpoint['mode']=='end_time')
            self.physics.load_project(doc);self.boundary_editor.load_project(doc);self.forcing.load_project(doc)
            self.physics.substrate_thickness.setValue(p['columns']['underlayer_thickness_m'])
            self.eulerian_override.setChecked(p['columns']['eulerian_thickness_m'] is not None)
            self.eulerian_thickness.setValue(p['columns']['eulerian_thickness_m'] or p['columns']['underlayer_thickness_m'])
            kinds={a['bc_id']:a['kind'] for a in p['assignments']};self.boundary_table.setRowCount(len(kinds))
            for row,(name,kind) in enumerate(sorted(kinds.items())):
                for col,text in enumerate((name,kind,self._boundary_condition_summary(doc,name,kind))):
                    item=QTableWidgetItem(text);item.setFlags(item.flags()&~Qt.ItemIsEditable);self.boundary_table.setItem(row,col,item)
            self._resize_boundary_table()
        finally:self._loading=False
        self._changed()

    def document(self,base_doc=None,*,validate=True):
        base=deepcopy(base_doc if base_doc is not None else self._base)
        if base is None:raise ValueError('Import a DEM and define the domain first')
        domain=PolygonDomain(**base['settings']['domain'])
        assignments=[BoundaryAssignment(**a) for a in base['settings']['assignments']]
        model=self.hydraulic_model()
        selected=model['physics']['sediment'];p=self.physics
        if selected:grains=selected['grains'];thickness,fractions=p.layer_values()
        else:grains=[dict(id='inert',diameter_m=.0005,solid_density_kg_m3=2650.,fraction=1.)];thickness,fractions=[.02,.2],[[1.],[1.]]
        controls=asdict(CurrentRiverControls(self.chezy.value(),grains[0]['diameter_m'],self.morfac.value(),
            self.betac.value() if self.secondary.isChecked() else 0.,self.viscosity.value() if self.turbulence.isChecked() else 0.,
            self.secondary_diffusion.value() if self.secondary.isChecked() else 0.,threads=self.threads.value(),cfl=self.cfl.value()))
        for key in ('nonlinear_atol','nonlinear_rtol'):controls[key]=base['settings']['controls'][key]
        from braids_core.sediment.river_column_methods import TOP_METHODS,BASE_METHODS,POROSITY_METHODS,method_ids
        bed=selected['bed'] if selected else None;layered=bool(bed and bed['composition']=='bed.layered')
        columns=asdict(ColumnMethodsV101(tuple(g['id'] for g in grains),tuple(g['solid_density_kg_m3'] for g in grains),
            tuple(p.dry_densities()) if selected else (1590.,),tuple(g['diameter_m'] for g in grains),
            bed='mixed' if bed and bed['composition']=='bed.fully_mixed' else 'layered',
            transport_layer='depth' if bed and bed['transport_layer']=='transport_layer.water_depth' else 'constant',
            transport_thickness_m=p.active_thickness.value() if selected else .02,depth_fraction=p.depth_fraction.value() if selected else .01,
            minimum_thickness_m=p.minimum_thickness.value() if selected else .001,underlayer_thickness_m=p.substrate_thickness.value() if selected else .2,
            eulerian_thickness_m=self.eulerian_thickness.value() if layered and self.eulerian_override.isChecked() else None,top_update=TOP_METHODS[p.top_update.currentData()] if layered else 1,
            base_update=BASE_METHODS[p.base_update.currentData()] if layered else 1,porosity=POROSITY_METHODS[p.porosity.currentData()] if layered else 0,
            solid_fraction=1.-p.porosity_value.value() if layered else .6,sand_solid_fraction=1.-p.sand_porosity.value() if layered else .6,
            mud_solid_fraction=1.-p.mud_porosity.value() if layered else .4,
            phi_std=tuple(p.phi_spreads()) if layered and p.porosity.currentData() in ('porosity.frings','porosity.beard_weyl') else (),
            exchange_thickness_m=p.exchange_thickness.value() if layered and p.exchange_on.isChecked() else 0.,
            lagrangian_underlayers=p.lagrangian_count.value() if layered else 0,
            lagrangian_thickness_m=p.lagrangian_thickness.value() if layered and p.lagrangian_count.value() else None))
        kinds={a.bc_id:a.kind for a in assignments if a.kind!='wall'}
        hydraulics,boundary_states,boundary_resolution=self.boundary_editor.values(list(columns['fraction_ids']),kinds,sediment=None,end_time=self.end_time.value())
        suspended=selected is not None and selected['suspension']!='off'
        bedload=selected is not None and p.bedload_on.isChecked()
        sediment_boundaries,initial_conc=self.forcing.values(list(columns['fraction_ids']),kinds,sediment=selected is not None,suspended=suspended,bedload=bedload)
        sediment_schedules=self.forcing.schedule_values(list(columns['fraction_ids']),sediment_boundaries,sediment=selected is not None,suspended=suspended,bedload=bedload)
        # Exterior constitutive state is automatic, explicitly saved, never an
        # interior bed clamp. Supplied single grain is independent of local stock.
        from braids_core.river_open_boundary import resolve as resolve_boundaries
        for name,kind in kinds.items():
            state=boundary_states[name]
            state['bed']=dict(method='bed.free' if selected else 'bed.fixed',initial=dict(method='initial.prepared_bed'))
            state['bed_schedule']=dict(method='forcing.constant')
            state['composition']=(dict(method='composition.reference_volume_fraction',fractions=[1.]) if kind=='inlet' else dict(method='composition.free')) if selected else None
            state['composition_schedule']=dict(method='forcing.constant') if selected else None
        boundary_resolution=resolve_boundaries(hydraulics,boundary_states,kinds,sediment=selected,ng=len(grains),end_time=self.end_time.value())
        # A project remembers the user's reporting choices so a module can be
        # enabled later without rebuilding its report setup.  A particular
        # run, however, writes only fields whose module is active now.  This
        # prevents zero-valued suspended/bank/bedform arrays from bloating an
        # otherwise ordinary hydraulic or bedload run.
        active_output={'initial_bed'}
        if selected is not None:
            active_output.add('bedrock')
        if bedload:
            active_output.add('bedload')
        if suspended:
            active_output.add('suspended')
        if self.secondary.isChecked() or self.turbulence.isChecked():
            active_output.add('advanced_hydraulics')
        if selected is not None and (selected['banks']!='off' or selected['avalanche']!='off'):
            active_output.add('bank_and_collapse')
        if selected is not None and selected['bedforms']!='off':
            active_output.add('bedforms')
        settings=dict(domain=deepcopy(base['settings']['domain']),assignments=deepcopy(base['settings']['assignments']),
            breaklines=deepcopy(base['settings']['breaklines']),
            solids=deepcopy(base['settings']['solids']),
            grid=self.geometry_grid(domain),levels=self.levels.value(),controls=controls,columns=columns,hydraulics=hydraulics,
            sediment_boundaries=sediment_boundaries,
            bank=dict(bank_theta=p.bank_theta.value() if selected and selected['banks']=='redistribution.dry_bank' else 0.,
                collapse=bool(selected and selected['avalanche'] in ('redistribution.braids_conservative_wet_dry_avalanche','redistribution.jiang_cao_repose_exner223_v1')),collapse_slope=p.collapse_slope.value(),repose_angle_deg=p.repose_angle_deg.value()),
            suspension=dict(bed_factor=1. if bedload or selected is None else 0.,sus_factor=0. if suspended and p.suspended_formula.currentData()=='suspended.settling_only' else 1. if suspended and p.suspended_formula.currentData() in ('suspended.van_rijn_1993_flow2dh','suspended.garcia_parker_1991','suspended.de_leeuw_2020','suspended.van_rijn_1984','suspended.zyserman_fredsoe_1994','suspended.camenen_larson_2008') else p.sus_factor.value() if suspended and p.suspension.currentData()!='suspension.settling_only' else 0.,
                diffusivity=p.sus_diffusion.value() if suspended else 0.,temperature=p.temperature.value(),cref=p.reference_concentration.value()),
            adaptation=dict({k:w.value() for k,w in self.adaptation.items()},terrain_enabled=self.mesh_mode.currentData()=='static_terrain',domain_border_enabled=self.domain_border.isChecked(),open_boundary_enabled=self.open_boundary_refinement.isChecked()),
            adaptive=self.mesh_mode.currentData()=='dynamic_depth',end_time_s=self.end_time.value(),dt_max_s=self.dt_max.value(),
            output_interval_s=self.output_interval.value(),checkpoint_interval_s=self.checkpoint_interval.value(),
            sediment_start_time_s=self.sediment_start.value() if selected is not None else 0.,
            output_fields=sorted(set(self.output_fields)&active_output))
        # Keep the saved boundary draft canonical with the automatic exterior
        # choice; otherwise reopening would change only hidden editor fields.
        self.boundary_editor.draft.load_active(dict(model=model,settings=dict(hydraulics=hydraulics,columns=columns),boundary_states=boundary_states))
        doc=dict(schema=SCHEMA,name=self.name.text(),model=model,settings=settings,turbines=self.turbines.value(),
            material_zones={kind:widget.values() for kind,widget in self.zones.items()},
            bank_survey=p.bank_survey.value(domain.crs_wkt,p.bank_theta.value(),p.bank_critical.value() if selected['banks']==SURVEY_BANKS[1] else None)
                if selected and selected['banks'] in SURVEY_BANKS else None,
            sediment_schedules=sediment_schedules,boundary_states=boundary_states,boundary_resolution=boundary_resolution,
            terrain=dict(dem_path=self.dem_path.text() or base['terrain']['dem_path']),
            initial=dict(stage_m=self.stage.value() if self.initial_mode.currentData() in ('stage','depth') else -1e7,velocity_m_s=[self.u.value(),self.v.value()] if self.initial_mode.currentData() in ('stage','depth') else [0.,0.],layer_thickness_m=thickness,layer_fractions=fractions,concentration_kg_m3=initial_conc,**({'depth_m':self.initial_depth.value()} if self.initial_mode.currentData()=='depth' else {})),
            slope_parameters={k:w.value() for k,w in p.slope_parameters.items()},
            hiding_parameters={channel:{k:w.value() for k,w in widgets.items()} for channel,widgets in p.hiding_parameters.items()},
            roughness_parameters=dict(chezy_m_half_s=self.chezy.value(),manning_s_m_third=self.manning.value(),nikuradse_m=self.nikuradse.value(),white_colebrook_relation=self.white_relation.currentData()),
            bedform_methods=deepcopy(BEDFORM_METHODS),van_rijn_reference_parameters={k:w.value() for k,w in p.vr_reference_parameters.items()},
            skin_shear_parameters={k:w.value() for k,w in p.skin_shear_parameters.items()},
            wet_avalanche_parameters=dict(wet_slope=p.wet_slope.value(),relaxation_time_s=p.avalanche_time.value()),
            column_methods=method_ids(ColumnMethodsV101(**columns)),dependent_layer_parameters=dict({k:w.value() for k,w in p.dependent_parameters.items()},hiding_scope=p.hiding_scope.currentData()),
            editor_state=dict(schema=6,sediment=p.editor_state(),turbines=self.turbines.editor_state(),bank_survey=p.bank_survey.draft(),forcing=self.forcing.editor_state(),
                schedules=self.forcing.schedule_state(),boundaries=self.boundary_editor.editor_state(),
                checkpoint_clock=dict(mode='end_time' if self.checkpoint_duration.isChecked() else 'custom',custom_interval_s=self._checkpoint_custom_interval)))
        if suspended and p.suspended_formula.currentData()=='suspended.van_rijn_1993_flow2dh':
            doc['van_rijn_reference_parameters']['suspended_diameter_m']=p.transport_setup()['diameters_m']['ds']
        if suspended and p.suspended_formula.currentData()=='suspended.garcia_parker_1991':
            from braids_core.sediment.river_garcia_parker import parameters
            doc['garcia_parker_parameters']=parameters(p.transport_setup()['diameters_m']['ds'])
        if suspended and p.suspended_formula.currentData()=='suspended.settling_only':
            from braids_core.sediment.river_supplied_suspension import parameters
            doc['supplied_suspension_parameters']=parameters(p.transport_setup()['diameters_m']['ds'])
        from braids_core.sediment.river_classic_suspension import EDITOR,parameters as classic_parameters
        if suspended and p.suspended_formula.currentData() in EDITOR:
            doc['classic_suspension_parameters']=classic_parameters(p.suspended_formula.currentData().replace('suspended.','suspension.'),p.transport_setup()['diameters_m'])
        if suspended and p.suspended_formula.currentData()=='suspended.de_leeuw_2020':
            from braids_core.sediment.river_de_leeuw import parameters
            sizes=p.transport_setup()['diameters_m']
            doc['de_leeuw_parameters']=parameters(sizes['ds'],sizes['d84'])
        return validate_project(doc) if validate else doc
