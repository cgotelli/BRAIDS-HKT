"""First complete polygon-to-native-to-results BRAIDS dock workflow."""
from __future__ import annotations

import json
import hashlib
import math
import os
import csv
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from qgis.PyQt import sip
from qgis.PyQt.QtCore import QSettings,QStandardPaths,QTimer,Qt
from qgis.PyQt.QtGui import QColor,QTextCursor
from qgis.PyQt.QtWidgets import (QCheckBox,QComboBox,QDockWidget,QDoubleSpinBox,QFileDialog,
    QFormLayout,QGridLayout,QHBoxLayout,QLabel,QLineEdit,QMessageBox,QPlainTextEdit,
    QHeaderView,QProgressBar,QPushButton,QScrollArea,QSlider,QSpinBox,QTabWidget,QTableWidget,
    QTableWidgetItem,QToolButton,QVBoxLayout,QWidget)
from qgis.core import (Qgis,QgsApplication,QgsColorRampShader,QgsCoordinateReferenceSystem,QgsCoordinateTransform,QgsMapLayerProxyModel,QgsTask,
    QgsDefaultValue,QgsEditFormConfig,QgsFeature,QgsFillSymbol,QgsGeometry,QgsLineSymbol,
    QgsMarkerSymbol,QgsPalLayerSettings,QgsPointXY,QgsProject,QgsRasterLayer,QgsRasterShader,QgsRectangle,
    QgsSingleBandPseudoColorRenderer,QgsSingleSymbolRenderer,QgsSnappingConfig,QgsTextFormat,
    QgsVectorLayer,QgsVectorLayerSimpleLabeling)
from qgis.gui import QgsMapLayerComboBox,QgsMapTool,QgsMapToolEmitPoint,QgsRubberBand

from braids_core.boundaries.cut_cell_io import (CutBoundary,CutMorphologicalFactorSeries,
    CutSediment,CutSedimentFraction,resolve_boundary_target,resolve_hydraulic_series)
from braids_core.boundaries.agglomeration import agglomerate_small_cells
from braids_core.boundaries.amr import (AMRSettings,AMR_LEVELS,AMR_RATIO,
                                        AMR_MIN_LEVELS,AMR_MAX_LEVELS)
from braids_core.boundaries.cut_cells import map_cut_cells
from braids_core.boundaries.domain import PolygonDomain,boundary_polyline
from braids_core.boundaries.gis import _RasterSampler,_projected_metres,_section as build_section
from braids_core.sediment.settling import DEFAULT_HINDERED_REFERENCE_CONCENTRATION_KG_M3
from braids_core.sediment.method_selection import resolve_sediment_methods
from braids_core.gpu_capabilities import (INTEGRATED, gpu_integrator_label,
    resolve_project_route, route_for_preparation_schema)
from braids_core.windows_paths import filesystem_path
from braids_qgis.core.bed_inputs import ERODIBLE_LAYER_NAME,create_erodible_dataset
from braids_qgis.core.bank_inputs import BANK_LAYER_NAME,create_bank_dataset
from braids_qgis.core.boundary_diagnostics import available_boundary_ids,boundary_diagnostic
from braids_qgis.core.refinement_inputs import REFINEMENT_LAYER_NAME,create_refinement_dataset
from braids_qgis.core.structure_inputs import (OBSTACLE_LAYER_NAME,
    create_obstacle_dataset,load_solid_obstacles)
from braids_qgis.core.boundary_terrain import section_with_cell_roughness,terrain_inward_slope
from braids_qgis.core.domain_layers import DOMAIN_LAYER_NAME,create_domain_dataset
from braids_qgis.core.guidance import GUIDE_LAYER_NAME,GUIDE_PROPERTY,create_guidance_dataset
from braids_qgis.core.hardware import recommended_openmp_threads
from braids_qgis.core.gpu_worker_identity import inspect_gpu_source_worker
from braids_qgis.core.gpu_runtime_probe import probe_gpu_python
from braids_qgis.core.gpu_discovery import discover_gpu_defaults
from braids_qgis.core.gpu_worker_runner import GpuWorkerRunner
from braids_qgis.core.gpu_live_transport_io import (
    has_public_law11_total_prepared_profile, validate_integrator_scope)
from braids_qgis.core.gpu_prepared_io import read_gpu_prepared
from braids_qgis.core.gpu_on_demand_prepared import inspect_gpu_on_demand_prepared
from braids_qgis.core.gpu_resume import inspect_gpu_resume_checkpoint
from braids_qgis.core.edges import (EDGE_LAYER_NAME,create_edge_dataset,load_assignments,
                                    reassign_boundary_faces)
from braids_qgis.core.inlet_diagnostics import DIAGNOSTICS_FILE,load_inlet_diagnostics
from braids_qgis.core.mesh_visualization import MESH_LAYER_NAME
from braids_qgis.core.project import (RUN_MARKER,create_project,grid_for_domain,
    load_project,reject_raw_gpu_persistent_front,save_project,
    PROGNOSTIC_SECONDARY_PROJECT_METHOD_ID,
    PROGNOSTIC_SECONDARY_PROJECT_PROVENANCE)
from braids_qgis.core.roughness import (MANNING_LAYER_NAME,boundary_manning,
    load_manning_zones,map_manning_to_cells)
from braids_qgis.core.typed_roughness_zones import (
    ROUGHNESS_LAYER_NAME,boundary_typed_roughness,
    create_typed_roughness_dataset,load_typed_roughness_zones,
    map_typed_roughness_to_cells)
from braids_qgis.core.water_body import (WATER_BODY_LAYER_NAME,WaterBodyRaster,
    connected_water_body_mask,create_water_body_dataset,water_body_seed_cell,
    write_connected_water_body)
from braids_qgis.core.runner import BraidsRunner
from braids_qgis.core.results import (integrate_result_transect,materialize_gpu_live_preview,
                                      materialize_live_preview_results,
                                      sample_result_probe)
from braids_qgis.core.sediment_product import morphology_output_choices
from braids_qgis.core.tasks import (PrepareBraidsMeshTask,PrepareBraidsResultsTask,
    PrepareBraidsRunTask,PrepareBraidsGpuRunTask)
from braids_qgis.gui.accordion import AccordionSection
from braids_qgis.gui.boundary_widgets import HydraulicBoundaryCard,SedimentBoundaryCard
from braids_qgis.gui.plots import COLORS,show_plot_dialog
from braids_qgis.runtime import default_native_executable


_ON_DEMAND_AMR_VARIANT = "on_demand_bank_amr_ssp33"
_ON_DEMAND_GENERAL_AMR_VARIANT = "on_demand_transport_amr_ssp33"
_GENERAL_LIVE_TRANSPORT_VARIANT = "live_normal_critical_transport_ssp33"
_GENERAL_LIVE_BANK_VARIANT = "live_normal_critical_bank_ssp33"
_GENERAL_AMR_SLOPE_POLICY = "local_eta_plane_v1"
_SECONDARY_EQUILIBRIUM = "equilibrium"
_SECONDARY_PROGNOSTIC_CLASSIC = "prognostic_classic_flow2dh"
_SECONDARY_PROGNOSTIC_LEGACY = "prognostic_legacy_fm"


def _secondary_project_selection(ui_value):
    """Map an explicit UI choice to its persisted, non-aliasing method."""
    if ui_value==_SECONDARY_EQUILIBRIUM:
        return dict(model="equilibrium",method_id=None,method_provenance=None)
    if ui_value==_SECONDARY_PROGNOSTIC_CLASSIC:
        return dict(model="prognostic",
                    method_id=PROGNOSTIC_SECONDARY_PROJECT_METHOD_ID,
                    method_provenance=PROGNOSTIC_SECONDARY_PROJECT_PROVENANCE)
    if ui_value==_SECONDARY_PROGNOSTIC_LEGACY:
        return dict(model="prognostic",method_id=None,method_provenance=None)
    raise ValueError("Secondary-flow UI selection is invalid")


def _secondary_ui_selection(settings):
    """Preserve historical no-discriminator projects on their legacy route."""
    if not isinstance(settings,dict) or settings.get("model")!="prognostic":
        return _SECONDARY_EQUILIBRIUM
    method_id=settings.get("method_id")
    provenance=settings.get("method_provenance")
    if method_id is None and provenance is None:
        return _SECONDARY_PROGNOSTIC_LEGACY
    if (method_id==PROGNOSTIC_SECONDARY_PROJECT_METHOD_ID and
            provenance==PROGNOSTIC_SECONDARY_PROJECT_PROVENANCE):
        return _SECONDARY_PROGNOSTIC_CLASSIC
    raise ValueError("Saved prognostic secondary-flow method provenance is invalid")


def _spin(minimum,maximum,value,decimals=3,step=1.):
    widget=QDoubleSpinBox();widget.setDecimals(decimals);widget.setRange(minimum,maximum);widget.setValue(value);widget.setSingleStep(step)
    return widget


class _NullableStageSpin(QDoubleSpinBox):
    """Numeric stage editor whose minimum sentinel is rendered as empty."""
    def textFromValue(self,value):
        return "" if value==self.minimum() else super().textFromValue(value)

    def valueFromText(self,text):
        return self.minimum() if not text.strip() else super().valueFromText(text)


def _nullable_stage_spin():
    widget=_NullableStageSpin();widget.setDecimals(3);widget.setRange(-1e12,1e7)
    widget.setSingleStep(1.);widget.setValue(widget.minimum());return widget


def _section(title,layout_class,expanded=False):
    section=AccordionSection(title,expanded=expanded);layout=layout_class();section.body_layout.addLayout(layout)
    return section,layout


def _parametric_obstacle_vertices(kind, points, segments=96):
    """Return a closed polygon for a circle or oriented ellipse."""
    required=2 if kind=="circle" else 3
    if kind not in ("circle","ellipse") or len(points)!=required:
        raise ValueError(f"{kind.title()} obstacle requires {required} map points")
    cx,cy=map(float,points[0]);ax=float(points[1][0])-cx;ay=float(points[1][1])-cy
    major=math.hypot(ax,ay)
    if major<=0:raise ValueError("Obstacle radius must be positive")
    ux,uy=ax/major,ay/major
    if kind=="circle":minor=major
    else:
        bx=float(points[2][0])-cx;by=float(points[2][1])-cy
        minor=abs(-uy*bx+ux*by)
        if minor<=0:raise ValueError("Ellipse minor radius must be positive")
    vertices=[]
    for index in range(segments):
        angle=2*math.pi*index/segments;c=math.cos(angle);s=math.sin(angle)
        vertices.append((cx+major*c*ux-minor*s*uy,cy+major*c*uy+minor*s*ux))
    vertices.append(vertices[0]);return vertices


class _ObstacleShapeMapTool(QgsMapTool):
    """Two-click circle and three-click oriented ellipse capture."""
    def __init__(self,canvas,kind,finished,cancelled,status):
        super().__init__(canvas);self.kind=kind;self.finished=finished;self.cancelled=cancelled
        self.status=status;self.points=[];self.band=QgsRubberBand(canvas,Qgis.GeometryType.Polygon)
        self.band.setStrokeColor(QColor(20,20,20,230));self.band.setFillColor(QColor(70,70,70,90));self.band.setWidth(2)

    @staticmethod
    def _xy(point):return float(point.x()),float(point.y())

    def _preview(self,cursor):
        if not self.points:return
        points=[*self.points,self._xy(cursor)]
        if self.kind=="ellipse" and len(points)==2:
            center,axis=points;dx=axis[0]-center[0];dy=axis[1]-center[1]
            points.append((center[0]-dy*.5,center[1]+dx*.5))
        required=2 if self.kind=="circle" else 3
        if len(points)!=required:return
        try:vertices=_parametric_obstacle_vertices(self.kind,points)
        except ValueError:return
        self.band.setToGeometry(QgsGeometry.fromPolygonXY([[QgsPointXY(*p) for p in vertices]]),None)

    def canvasMoveEvent(self,event):self._preview(event.mapPoint())

    def canvasReleaseEvent(self,event):
        if event.button()==Qt.MouseButton.RightButton:
            self.cancel();return
        if event.button()!=Qt.MouseButton.LeftButton:return
        self.points.append(self._xy(event.mapPoint()))
        required=2 if self.kind=="circle" else 3
        if len(self.points)==required:
            try:self.finished(_parametric_obstacle_vertices(self.kind,self.points))
            except Exception as exc:self.cancelled(str(exc))
            return
        if self.kind=="circle":self.status("Click a point on the circle. Right-click cancels.")
        elif len(self.points)==1:self.status("Click the end of the ellipse major axis. Right-click cancels.")
        else:self.status("Click to set the perpendicular minor radius. Right-click cancels.")

    def cancel(self):
        self.band.reset(Qgis.GeometryType.Polygon);self.cancelled(None)

    def clear(self):self.band.reset(Qgis.GeometryType.Polygon)


class _ResultPolylineMapTool(QgsMapTool):
    """Simple left-click polyline capture; right-click completes it."""
    def __init__(self,canvas,finished,cancelled):
        super().__init__(canvas);self.finished=finished;self.cancelled=cancelled;self.points=[]
        self.band=QgsRubberBand(canvas,Qgis.GeometryType.Line);self.band.setColor(QColor(30,90,220));self.band.setWidth(3)
    def canvasMoveEvent(self,event):
        if self.points:self.band.setToGeometry(QgsGeometry.fromPolylineXY([*self.points,event.mapPoint()]),None)
    def canvasReleaseEvent(self,event):
        if event.button()==Qt.MouseButton.RightButton:
            if len(self.points)>=2:self.finished([(point.x(),point.y()) for point in self.points])
            else:self.cancelled();return
            return
        if event.button()!=Qt.MouseButton.LeftButton:return
        self.points.append(event.mapPoint());self.band.setToGeometry(QgsGeometry.fromPolylineXY(self.points),None)
    def clear(self):self.band.reset(Qgis.GeometryType.Line)


class _ResultProbeMapTool(QgsMapTool):
    def __init__(self,canvas,finished,cancelled):super().__init__(canvas);self.finished=finished;self.cancelled=cancelled
    def canvasReleaseEvent(self,event):
        if event.button()==Qt.MouseButton.LeftButton:self.finished(float(event.mapPoint().x()),float(event.mapPoint().y()))
        elif event.button()==Qt.MouseButton.RightButton:self.cancelled()


class _GpuLivePreviewTask(QgsTask):
    """Authenticate/rasterize one committed GPU frame away from the QGIS UI."""

    def __init__(self,run_root,prepared_root,variable,after_index,cache):
        super().__init__("BRAIDS GPU live preview",QgsTask.Flag.CanCancel)
        self.run_root=Path(run_root);self.prepared_root=Path(prepared_root)
        self.variable=variable;self.after_index=after_index;self.cache=cache
        self.product=None;self.error=None

    def run(self):
        try:
            self.product=materialize_gpu_live_preview(self.run_root,self.variable,
                prepared_root=self.prepared_root,after_index=self.after_index,
                cache=self.cache,cancelled=self.isCanceled)
            return not self.isCanceled()
        except InterruptedError:
            return False
        except Exception as exc:
            self.error=RuntimeError(str(exc));return False


class BraidsDockWidget(QDockWidget):
    PROJECT_KEY="project_path"
    RUN_KEY="run_root"
    BACKEND_KEY="execution_backend"
    GPU_WORKER_SETTINGS_KEY="BRAIDS/gpu_worker_folder"
    GPU_PYTHON_SETTINGS_KEY="BRAIDS/gpu_python_executable"
    BOUNDARY_CELL_AREA_FRACTION=.5
    RECONSTRUCTION_SCHEME=5

    def __init__(self,parent=None,iface=None):
        super().__init__("BRAIDS",parent);self.setObjectName("braidsDockWidget");self.iface=iface
        self.run_root=None;self.result_run_root=None;self._manifest=None;self._task=None;self._preparation_tasks={};self._capture_connection=None
        self._prepared_controls_armed=False;self._prepared_controls_dirty=False
        self._external_result_run_roots=[]
        # These advanced terrain controls have no separate widgets. Preserve
        # a loaded project's choices rather than recreating them as defaults.
        self._terrain_settings=dict(interpretation="bilinear_surface",
            section_vertical_tolerance_m=0.,section_dem_path=None)
        self._minimum_boundary_cell_area_fraction=self.BOUNDARY_CELL_AREA_FRACTION
        self._gpu_prepared_root=None;self._gpu_output_root=None
        self._gpu_manifest_pin=None;self._gpu_python_pin=None;self._gpu_capability=None
        self._gpu_selector_policy_supported=False
        self._gpu_selector_user_choice=False
        self._capture_started_editing=False
        self._water_body_map_tool=None;self._water_body_previous_map_tool=None
        self._obstacle_map_tool=None;self._obstacle_previous_map_tool=None
        self._result_layers={};self._result_mesh_layers={};self._initial_mesh_layer=None;self._live_preview_layer=None;self._live_preview_frame=None
        self._gpu_preview_task=None;self._gpu_preview_cursor=-1;self._gpu_preview_variable=None
        self._gpu_preview_cache=None;self._gpu_preview_manifest_mtime=None
        self._gpu_preview_manifest_count=0;self._gpu_preview_final_pending=False
        self._gpu_preview_timer=QTimer(self);self._gpu_preview_timer.setInterval(750)
        self._gpu_preview_timer.timeout.connect(self._poll_gpu_live_preview)
        self._result_times=[];self._result_frame_index=0
        self._result_probe_tool=None;self._result_transect_tool=None;self._result_probe_layer=None;self._result_transect_layer=None;self._probe_series=None;self._transect_series=None;self._pending_mesh_mode=None
        self.boundary_cards={};self.sediment_cards={};self._updating_boundary_cards=False;self._plot_windows=[]
        self._legacy_lake_seed_xy=None
        self.runner=BraidsRunner(self);self.runner.stdout.connect(self._log);self.runner.stderr.connect(self._log)
        self.runner.started.connect(lambda:self._set_status("Native solver running",Qgis.Info))
        self.runner.progress.connect(self._run_progress);self.runner.live_frame.connect(self._live_frame_available);self.runner.finished.connect(self._run_finished)
        self.gpu_runner=GpuWorkerRunner(self)
        self.gpu_runner.stdout.connect(self._log);self.gpu_runner.stderr.connect(self._log)
        self.gpu_runner.started.connect(lambda:self._set_status("GPU solver running",Qgis.Info))
        self.gpu_runner.progress.connect(self._run_progress)
        self.gpu_runner.finished.connect(self._gpu_run_finished)
        body=QWidget();layout=QVBoxLayout(body);self.tabs=QTabWidget();layout.addWidget(self.tabs);self.setWidget(body)
        self._general_tab();self._bc_ic_tab();self._mesh_tab();self._sediment_tab();self._run_tab();self._results_tab();self._refresh_input_summary();self._restore_project_reference()
        self._connect_prepared_control_invalidation()
        if self.run_root is not None:self._arm_prepared_controls()
        self.tabs.currentChanged.connect(self._condition_tab_selected)
        QgsProject.instance().layersAdded.connect(self._exclude_guidance_from_boundaries)
        self._exclude_guidance_from_boundaries()

    def _general_tab(self):
        content=QWidget();outer=QVBoxLayout(content)
        self.general_layout=outer
        files,form=_section("BRAIDS project",QFormLayout,True);self.project_section=files
        self.workspace=QLineEdit(str(self._default_workspace()));workspace_row=QHBoxLayout();workspace_row.addWidget(self.workspace);button=QPushButton("Browse…");button.clicked.connect(self._browse_workspace);workspace_row.addWidget(button);form.addRow("Project folder",workspace_row)
        self.project_path=QLineEdit();self.project_path.setVisible(False)
        project_actions=QHBoxLayout();self.save_project_button=QPushButton("Save Project");self.save_project_button.clicked.connect(self.save_project_ui);self.load_project_button=QPushButton("Open Existing Project Folder…");self.load_project_button.clicked.connect(self._browse_project);project_actions.addWidget(self.save_project_button);project_actions.addWidget(self.load_project_button);form.addRow(project_actions)
        project_help=QLabel("BRAIDS stores braids_project.json, generated GIS layers, prepared runs and results in this folder. The bundled native solver is selected automatically.");project_help.setWordWrap(True);form.addRow(project_help);outer.addWidget(files)

        layers,grid=_section("Calculation domain and terrain",QGridLayout,False)
        self.layer_section=layers;self.layers_grid=grid
        self.domain_layer=QgsMapLayerComboBox();self.domain_layer.setFilters(QgsMapLayerProxyModel.PolygonLayer);self.domain_layer.setAllowEmptyLayer(True)
        self.dem_layer=QgsMapLayerComboBox();self.dem_layer.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.create_domain_button=QPushButton("Draw Calculation Domain");self.create_domain_button.clicked.connect(self.create_domain)
        self.create_inlet_guide_button=QToolButton();self.create_inlet_guide_button.setIcon(QgsApplication.getThemeIcon("/mActionCaptureLine.svg"))
        self.create_inlet_guide_button.setText("Draw Inlet Guide");self.create_inlet_guide_button.setAccessibleName("Draw Inlet Guide")
        self.create_inlet_guide_button.setToolTip("Draw an inlet guidance polyline. Click vertices, then right-click to finish. Guides help draw the domain and are hidden when the domain is saved or BC assignment begins.")
        self.create_inlet_guide_button.clicked.connect(self.create_inlet_guide)
        grid.addWidget(QLabel("Calculation domain"),0,0);grid.addWidget(self.domain_layer,0,1,1,2);grid.addWidget(self.create_domain_button,0,3)
        grid.addWidget(self.create_inlet_guide_button,0,4)
        grid.addWidget(QLabel("Terrain DEM"),1,0);grid.addWidget(self.dem_layer,1,1,1,4)
        domain_help=QLabel("A loaded polygon layer, including a Shapefile, can be selected directly as the calculation domain. Select one feature when the layer contains several polygons.");domain_help.setWordWrap(True);grid.addWidget(domain_help,2,0,1,5)
        outer.addWidget(layers)

        self.setup_status=QLabel("Select a polygon and terrain DEM, then define conditions in BC & IC.");self.setup_status.setWordWrap(True);outer.addWidget(self.setup_status);outer.addStretch()
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff);scroll.setWidget(content);self.tabs.addTab(scroll,"General")

    def _bc_ic_tab(self):
        content=QWidget();outer=QVBoxLayout(content)
        forcing,vbox=_section("Boundary conditions",QVBoxLayout,True)
        self.boundary_section=forcing
        edge_row=QGridLayout();self.boundary_grid=edge_row
        self.edge_layer=QgsMapLayerComboBox();self.edge_layer.setFilters(QgsMapLayerProxyModel.LineLayer);self.edge_layer.setAllowEmptyLayer(True)
        self.edge_layer.layerChanged.connect(self._style_boundary_layer)
        self.create_edges_button=QPushButton("Create from Domain");self.create_edges_button.clicked.connect(self.create_edges)
        edge_row.addWidget(QLabel("Boundary faces"),0,0);edge_row.addWidget(self.edge_layer,0,1,1,2);edge_row.addWidget(self.create_edges_button,0,3);vbox.addLayout(edge_row)

        assignment=QGridLayout();self.boundary_assignment_layout=assignment
        self.boundary_id=QLineEdit();self.boundary_id.setPlaceholderText("river_1 or outlet_1")
        self.boundary_kind=QComboBox();self.boundary_kind.addItems(["inlet","outlet","wall"])
        self.apply_edges_button=QPushButton("Apply to Selected Face(s)");self.apply_edges_button.clicked.connect(self.apply_selected_edges)
        self.apply_edges_button.setToolTip("Validates the complete cross-section before saving. Reusing an inlet/outlet ID replaces its previous faces.")
        self.select_edges_button=QPushButton("Select");self.select_edges_button.clicked.connect(
            lambda:self._activate_select_tool(self.edge_layer.currentLayer(),"boundary face"))
        assignment.addWidget(QLabel("Boundary ID"),0,0);assignment.addWidget(self.boundary_id,0,1);assignment.addWidget(QLabel("Type"),0,2);assignment.addWidget(self.boundary_kind,0,3)
        assignment.addWidget(self.select_edges_button,1,0);assignment.addWidget(self.apply_edges_button,1,1,1,3);vbox.addLayout(assignment)
        help_label=QLabel("Select exactly one polygon face for each inlet. Walls and outlets may use several contiguous collinear faces. Applying an existing inlet/outlet ID replaces its previous faces. Each active ID has its own controls and a Delete BC button below. Deleting a BC returns all of its faces to the default wall condition.");help_label.setWordWrap(True);vbox.addWidget(help_label)
        self.refresh_boundaries_button=QPushButton("Refresh IDs from Edge Layer");self.refresh_boundaries_button.clicked.connect(self.refresh_boundary_json);vbox.addWidget(self.refresh_boundaries_button)
        self.boundary_cards_widget=QWidget();self.boundary_cards_layout=QVBoxLayout(self.boundary_cards_widget);self.boundary_cards_layout.setContentsMargins(0,0,0,0);vbox.addWidget(self.boundary_cards_widget)
        self.boundary_json=QPlainTextEdit();self.boundary_json.setVisible(False);vbox.addWidget(self.boundary_json);outer.addWidget(forcing)

        initial,grid=_section("Initial conditions",QGridLayout,False);self.initial_section=initial;self.initial_grid=grid
        self.water_body_layer=QgsMapLayerComboBox();self.water_body_layer.setFilters(QgsMapLayerProxyModel.PolygonLayer);self.water_body_layer.setAllowEmptyLayer(True);self.water_body_layer.setLayer(None)
        self.initial_stage=_nullable_stage_spin();self.initial_stage.setToolTip("Leave blank for a dry initial domain. A level is required when an initial-water polygon is selected.")
        self.create_water_body_button=QPushButton("Draw Initial Water Body");self.create_water_body_button.clicked.connect(self.create_water_body)
        self.create_water_body_from_dem_button=QPushButton("Create from DEM Point");self.create_water_body_from_dem_button.clicked.connect(self.start_water_body_from_dem)
        self.create_water_body_from_dem_button.setToolTip("Enter the initial water level, then click inside its basin. BRAIDS follows the connected DEM contour and creates the polygon.")
        grid.addWidget(QLabel("Initial water body (optional)"),0,0);grid.addWidget(self.water_body_layer,0,1,1,2);grid.addWidget(self.create_water_body_button,0,3)
        grid.addWidget(QLabel("Initial water level (m)"),1,0);grid.addWidget(self.initial_stage,1,1);grid.addWidget(self.create_water_body_from_dem_button,1,2,1,2)
        note=QLabel("A blank water level starts dry. Draw a polygon manually, load one, or enter a level and create the connected water body by clicking inside its DEM basin.");note.setWordWrap(True);grid.addWidget(note,2,0,1,4);outer.addWidget(initial)

        internal,vbox=_section("Internal conditions",QVBoxLayout,False);self.internal_section=internal
        obstacle_grid=QGridLayout();vbox.addLayout(obstacle_grid)
        self.obstacle_layer=QgsMapLayerComboBox();self.obstacle_layer.setFilters(QgsMapLayerProxyModel.PolygonLayer);self.obstacle_layer.setAllowEmptyLayer(True);self.obstacle_layer.setLayer(None)
        self.create_obstacle_polygon_button=QPushButton("★ Polygon");self.create_obstacle_polygon_button.clicked.connect(self.create_solid_obstacle)
        self.create_obstacle_circle_button=QPushButton("○ Circle");self.create_obstacle_circle_button.clicked.connect(self.create_solid_obstacle_circle)
        self.create_obstacle_ellipse_button=QPushButton("⬭ Ellipse");self.create_obstacle_ellipse_button.clicked.connect(self.create_solid_obstacle_ellipse)
        self.select_obstacle_button=QPushButton("Select");self.select_obstacle_button.clicked.connect(
            lambda:self._activate_select_tool(self.obstacle_layer.currentLayer(),"solid obstacle"))
        self.delete_obstacle_button=QPushButton("Delete Selected");self.delete_obstacle_button.clicked.connect(self.delete_selected_obstacles)
        self.cancel_obstacle_button=QPushButton("Cancel");self.cancel_obstacle_button.clicked.connect(self.cancel_solid_obstacle_drawing)
        self.create_obstacle_button=self.create_obstacle_polygon_button
        obstacle_grid.addWidget(QLabel("Solid reflective obstacles"),0,0);obstacle_grid.addWidget(self.obstacle_layer,0,1,1,3)
        obstacle_grid.addWidget(self.create_obstacle_polygon_button,1,0);obstacle_grid.addWidget(self.create_obstacle_circle_button,1,1,1,2);obstacle_grid.addWidget(self.create_obstacle_ellipse_button,1,3)
        obstacle_grid.addWidget(self.select_obstacle_button,2,0);obstacle_grid.addWidget(self.delete_obstacle_button,2,1,1,2);obstacle_grid.addWidget(self.cancel_obstacle_button,2,3)
        obstacle_note=QLabel("Draw a free polygon, circle or oriented ellipse. Every object is removed from the flowing area at its exact cut-cell perimeter; its fixed impermeable walls reflect shallow-water waves. A loaded polygon Shapefile or GeoPackage can also be selected, including bridge-pillar footprints.");obstacle_note.setWordWrap(True);vbox.addWidget(obstacle_note)
        outer.addWidget(internal)
        outer.addStretch();scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff);scroll.setWidget(content);self.tabs.addTab(scroll,"BC && IC")

    def _mesh_tab(self):
        page=QWidget();outer=QVBoxLayout(page)
        grid_box,form=_section("Mesh resolution and AMR",QFormLayout,True);self.mesh_resolution_section=grid_box
        self.spacing=_spin(.01,10000,20/AMR_RATIO**(AMR_LEVELS-1),3,1)
        form.addRow("Minimum cell size (m)",self.spacing)
        self.amr_enabled=QCheckBox("Adaptive mesh refinement");self.amr_enabled.setChecked(True)
        form.addRow(self.amr_enabled)
        self.amr_levels=QSpinBox();self.amr_levels.setRange(AMR_MIN_LEVELS,AMR_MAX_LEVELS);self.amr_levels.setValue(AMR_LEVELS)
        self.amr_levels.setToolTip("Number of nested mesh levels, including the coarsest and finest levels.")
        form.addRow("Number of mesh levels",self.amr_levels)
        form.addRow("Refinement ratio",QLabel("2:1 (fixed)"))
        self.amr_subcycling=QCheckBox("Advance finer levels with smaller timesteps")
        self.amr_subcycling.setChecked(AMRSettings().subcycling)
        self.amr_subcycling.setToolTip("Use AMR time subcycling and coarse/fine reflux. Turn off for synchronous levels only when the selected solver supports them.")
        form.addRow(self.amr_subcycling)
        self.amr_sizes=QLabel();self.amr_sizes.setWordWrap(True);form.addRow(self.amr_sizes)
        self.spacing.valueChanged.connect(self._update_amr_sizes);self.amr_enabled.toggled.connect(self._update_amr_sizes)
        self.amr_levels.valueChanged.connect(self._update_amr_sizes)
        outer.addWidget(grid_box)

        self.visualize_initial_mesh_button=QPushButton("Visualize Initial Mesh")
        self.visualize_initial_mesh_button.clicked.connect(self.visualize_initial_mesh)
        self.visualize_initial_mesh_button.setToolTip("Build and display the exact initial cut-cell mesh using the selected domain and cell size.")
        form.addRow(self.visualize_initial_mesh_button)

        adaptive,criteria=_section("Adaptive refinement criteria",QFormLayout,False)
        self.amr_interval=_spin(.001,3600,1.,3,.1);criteria.addRow("Update mesh every (s)",self.amr_interval)
        self.amr_delay=QSpinBox();self.amr_delay.setRange(1,1000);self.amr_delay.setValue(3)
        criteria.addRow("Quiet updates before coarsening",self.amr_delay)
        self.amr_buffer=QSpinBox();self.amr_buffer.setRange(2,32);self.amr_buffer.setValue(3)
        criteria.addRow("Minimum front buffer (fine cells)",self.amr_buffer)
        self.amr_switches={}
        for key,label in (("refine_wet_dry","Wet/dry front and dry banks"),
                          ("refine_walls","Wetted solid walls and obstacles"),
                          ("refine_bathymetry","Submerged terrain details"),
                          ("refine_hydraulics","Water-surface and velocity changes"),
                          ("refine_sediment","Sediment fronts, bed evolution and bank erosion")):
            widget=QCheckBox(label);widget.setChecked(True);criteria.addRow(widget);self.amr_switches[key]=widget
        self.amr_thresholds={}
        defaults=AMRSettings()
        for key,label in (("surface_tolerance_m","Water-surface change (m)"),
                          ("velocity_tolerance_m_s","Velocity change (m/s)"),
                          ("bed_tolerance_m","Unresolved terrain height (m)"),
                          ("concentration_tolerance_kg_m3","Concentration change (kg/m³)"),
                          ("composition_tolerance","Grain volume-fraction change"),
                          ("bed_change_tolerance_m","Bed evolution (m)")):
            widget=_spin(.000001,1 if key=="composition_tolerance" else 10000,getattr(defaults,key),6,.01)
            criteria.addRow(label,widget);self.amr_thresholds[key]=widget
        note=QLabel("Wet areas use at least L1. The finest level follows active fronts and selected criteria, including a buffer ahead of moving water. Each inlet/outlet stays fully resolved. Water and all sediment fractions share one mesh.")
        note.setWordWrap(True);criteria.addRow(note);outer.addWidget(adaptive)

        zones,zone_layout=_section('Minimum refinement zones',QFormLayout,False)
        self.refinement_layer=QgsMapLayerComboBox();self.refinement_layer.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        self.refinement_layer.setAllowEmptyLayer(True);self.refinement_layer.setLayer(None)
        zone_layout.addRow('Refinement polygons',self.refinement_layer)
        self.refinement_level=QComboBox()
        row=QHBoxLayout();row.addWidget(self.refinement_level)
        self.create_refinement_button=QPushButton('Add Refinement Zone');self.create_refinement_button.clicked.connect(self.create_refinement_zone)
        row.addWidget(self.create_refinement_button)
        self.cancel_refinement_button=QPushButton('Cancel');self.cancel_refinement_button.clicked.connect(lambda:self._cancel_polygon_capture())
        row.addWidget(self.cancel_refinement_button);zone_layout.addRow(row)
        zone_layout.addRow(QLabel('Overlapping zones use the highest minimum level.'))
        outer.addWidget(zones)
        self._update_amr_sizes()

        roughness,grid=_section("Hydraulic roughness",QGridLayout,False);self.manning_section=roughness
        self.hydraulic_roughness_mode=QComboBox()
        self.hydraulic_roughness_mode.addItem("Manning n","manning")
        self.hydraulic_roughness_mode.addItem("Chezy C","chezy")
        self.hydraulic_roughness_mode.addItem("White-Colebrook ks","white_colebrook")
        grid.addWidget(QLabel("Resistance law"),0,0);grid.addWidget(self.hydraulic_roughness_mode,0,1,1,3)
        self.manning=_spin(0,1.e6,.05,12,.005);self.hydraulic_roughness_parameter_label=QLabel("Default Manning n")
        grid.addWidget(self.hydraulic_roughness_parameter_label,1,0);grid.addWidget(self.manning,1,1,1,3)
        self.white_colebrook_chzk2d=QCheckBox("Use depth-averaged ChzK2D branch")
        self.white_colebrook_chzk2d.setToolTip("Delft3D White-Colebrook ChzK2D modifies the depth used in the Chezy conversion. Available only for White-Colebrook ks.")
        grid.addWidget(self.white_colebrook_chzk2d,2,0,1,4)
        self.manning_layer=QgsMapLayerComboBox();self.manning_layer.setFilters(QgsMapLayerProxyModel.PolygonLayer);self.manning_layer.setAllowEmptyLayer(True);self.manning_layer.setLayer(None)
        self.roughness_zone_layer_label=QLabel("Roughness zones (optional)")
        grid.addWidget(self.roughness_zone_layer_label,3,0);grid.addWidget(self.manning_layer,3,1,1,3)
        self.create_manning_button=QPushButton("Add New Roughness Zone");self.create_manning_button.clicked.connect(self.create_manning_zones)
        self.manning_zone_value=_spin(0,1.e6,.08,12,.005)
        self.manning_zone_priority=QSpinBox();self.manning_zone_priority.setRange(-1000000000,1000000000);self.manning_zone_priority.setValue(0)
        self.apply_manning_button=QPushButton("Update Selected Zone");self.apply_manning_button.clicked.connect(self.apply_selected_manning)
        self.snap_polygons=QCheckBox("Snap new polygons to vertices and segments");self.snap_polygons.setChecked(True)
        self.manning_zone_label=QLabel("New zone Manning n")
        grid.addWidget(self.manning_zone_label,4,0);grid.addWidget(self.manning_zone_value,4,1);grid.addWidget(self.create_manning_button,4,2,1,2)
        grid.addWidget(QLabel("Zone priority"),5,0);grid.addWidget(self.manning_zone_priority,5,1,1,3)
        grid.addWidget(self.apply_manning_button,6,0,1,4)
        grid.addWidget(self.snap_polygons,7,0,1,4)
        note=QLabel("Spatial zones store the selected raw C, n or ks parameter. Highest integer priority wins where zones compete; equal-priority zones with different values are rejected. No equivalent-Manning conversion is used.");note.setWordWrap(True);grid.addWidget(note,8,0,1,4);outer.addWidget(roughness)
        self.hydraulic_roughness_mode.currentIndexChanged.connect(self._update_hydraulic_roughness_controls)
        self._update_hydraulic_roughness_controls()
        outer.addStretch();self.tabs.addTab(page,"Mesh")

    def _sediment_tab(self):
        content=QWidget();outer=QVBoxLayout(content)
        self.sediment_enabled=QCheckBox("Enable sediment transport")
        self.sediment_enabled.setChecked(False)
        self.sediment_enabled.setToolTip(
            "Sediment physics is opt-in. Enabling this master switch does not "
            "automatically enable suspended load, bed load, banks, secondary "
            "flow, bedforms or avalanching.")
        outer.addWidget(self.sediment_enabled)
        self.adaptive_river_button=QPushButton("Open semi-implicit CPU / GPU river workspace…")
        self.adaptive_river_button.setEnabled(True)
        self.adaptive_river_button.setToolTip("Shared semi-implicit CPU/GPU f-wave river bundle. Separate reviewed setup; MF1, constant discharge/stage boundaries and common-clock AMR.")
        self.adaptive_river_button.clicked.connect(self._open_adaptive_river)
        self.general_layout.insertWidget(self.general_layout.count()-1,self.adaptive_river_button)
        material,form=_section("Sediment material",QFormLayout,True)
        self.grain_density=_spin(1,10000,2650,2,10);self.water_density=_spin(1,10000,1000,2,10);self.porosity=_spin(0,.999999,.4,6,.01)
        self.d50_mm=_spin(.000001,10000,.2,6,.01);self.d90_mm=_spin(.000001,10000,.3,6,.01);self.z0_mm=_spin(.000001,10000,1.,6,.1)
        for label,widget in (("Grain density (kg/m³)",self.grain_density),("Water density (kg/m³)",self.water_density),("Porosity",self.porosity),("D50 (mm)",self.d50_mm),("D90 (mm)",self.d90_mm)):form.addRow(label,widget)
        self.sediment_roughness=QComboBox();self.sediment_roughness.addItem("Fixed roughness length","fixed");self.sediment_roughness.addItem("From local Manning and water depth","manning")
        form.addRow("Sediment roughness",self.sediment_roughness)
        self.z0_label=QLabel("Fixed roughness z0 (mm)");form.addRow(self.z0_label,self.z0_mm)
        self.sediment_roughness_help=QLabel();self.sediment_roughness_help.setWordWrap(True);form.addRow(self.sediment_roughness_help)
        outer.addWidget(material)

        fractions,vbox=_section("Grain fractions",QVBoxLayout,False)
        note=QLabel("Leave this table empty for one noncohesive material. Add at least two rows for independently conserved sand or mud classes. The substrate/default and active-surface volume fractions must each sum to 1. Cohesive parameters are used for mud; roughness and depth thresholds are inherited above.")
        note.setWordWrap(True);vbox.addWidget(note)
        self.fraction_table=QTableWidget(0,13);self.fraction_table.setHorizontalHeaderLabels(
            ["ID","Substrate/default\nvolume fraction","Active volume\nfraction","Type\n(sand/mud)",
             "Density\n(kg/m³)","Porosity","D50\n(mm)","D90\n(mm)","Settling\n(m/s)",
             "Critical erosion\n(Pa)","Critical deposition\n(Pa)",
             "Erosion parameter\n(kg/m²/s)","Erosion\nexponent"])
        fraction_header=self.fraction_table.horizontalHeader()
        fraction_header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        fraction_header.setTextElideMode(Qt.TextElideMode.ElideNone)
        fraction_header.setMinimumHeight(54)
        fraction_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        fraction_header.setStretchLastSection(True);vbox.addWidget(self.fraction_table)
        row=QHBoxLayout();add=QPushButton("Add Fraction");remove=QPushButton("Remove Selected")
        add.clicked.connect(self._add_fraction);remove.clicked.connect(self._remove_fractions)
        row.addWidget(add);row.addWidget(remove);row.addStretch();vbox.addLayout(row);outer.addWidget(fractions)

        suspended,form=_section("Suspended load",QFormLayout,False)
        self.suspended_enabled=QCheckBox("Enable suspended load")
        self.suspended_enabled.setChecked(False);form.addRow(self.suspended_enabled)
        self.exchange=QComboBox();self.exchange.addItem("Van Rijn suspended exchange",2);self.exchange.addItem("Cohesive or mixed sand/mud",3);self.exchange.addItem("Settling only",1);self.exchange.addItem("Off",0)
        self.settling=_spin(0,10,.01,6,.001);self.exchange_depth=_spin(0,10000,.05,4,.01)
        self.settling_mode=QComboBox();self.settling_mode.addItem("Constant settling velocity",0);self.settling_mode.addItem("Concentration-dependent hindered settling",1)
        self.hindered_settling_reference=_spin(.000001,1.e9,DEFAULT_HINDERED_REFERENCE_CONCENTRATION_KG_M3,3,100.)
        self.hindered_settling_reference.setToolTip("Total suspended concentration at which the bounded settling multiplier reaches zero. The multiplier is max(0, 1 - Ctotal / Creference)^5 and uses the sum of all fractions.")
        self.critical_mud_fraction=_spin(0,1,.5,4,.05);self.critical_mud_fraction.setToolTip("Active-surface mud fraction at which erosion changes from noncohesive to cohesive behavior.")
        self.reference_height_factor=_spin(.000001,1000,1.,6,.1)
        self.reference_height_factor.setToolTip("Scales the reference height used for Van Rijn suspended exchange, within the formula's water-depth limits. Dimensionless; default 1.")
        form.addRow("Exchange formulation",self.exchange);form.addRow("Settling treatment",self.settling_mode);form.addRow("Hindered-settling reference concentration (kg/m³)",self.hindered_settling_reference);form.addRow("Critical mud fraction",self.critical_mud_fraction);form.addRow("Reference-height factor",self.reference_height_factor);form.addRow("Settling velocity (m/s)",self.settling);form.addRow("Minimum exchange depth (m)",self.exchange_depth);outer.addWidget(suspended)

        bed,form=_section("Bed load and morphology",QFormLayout,False)
        self.bedload_enabled=QCheckBox("Enable bed load and Exner morphology")
        self.bedload_enabled.setChecked(False);form.addRow(self.bedload_enabled)
        self.bedload=QComboBox();self.bedload.addItem("Van Rijn 1993",1);self.bedload.addItem("Meyer-Peter-Müller",2);self.bedload.addItem("Ashida-Michiue (current)",3);self.bedload.addItem("Soulsby (current)",4);self.bedload.addItem("Wilcock-Crowe",5);self.bedload.addItem("Modified Wilcock-Crowe",6);self.bedload.addItem("Gaeuman et al.",7);self.bedload.addItem("Engelund-Hansen (constant Chezy)",8);self.bedload.addItem("Off",0)
        self.engelund_hansen_chezy=_spin(.000001,1.e6,28.8,6,.1)
        self.engelund_hansen_chezy.setToolTip("Positive constant Chezy coefficient used only by the qualified Engelund-Hansen law 8 closure. It is independent of Mesh Manning roughness.")
        self.transport_calibration=_spin(0,1000,1.,5,.1);self.transport_calibration.setToolTip("Multiplicative calibration applied to the selected current-driven bed-load formula. Default 1 uses the formula's standard coefficients.")
        self.minimum_bedload_depth=_spin(0,10000,.05,4,.01);self.mpm_cf=_spin(0,100,.01,6,.005);self.mpm_critical=_spin(0,100,0.,6,.005)
        self.mpm_friction_mode=QComboBox();self.mpm_friction_mode.addItem("Specified friction coefficient (Zhu)",0);self.mpm_friction_mode.addItem("Manning and grain roughness corrected",1)
        self.mpm_friction_mode.setToolTip("The specified coefficient retains the Zhu benchmark closure. The roughness-corrected option derives MPM mobility from local depth, Mesh Manning and active-mixture D90, with critical Shields fixed to 0.047.")
        self.bedform_roughness=QCheckBox("Enable equilibrium bedform roughness feedback")
        self.bedform_roughness.setToolTip("Qualified, default-off Fredsoe-MPM dune height with Van Rijn 1984 length and roughness. Writes bedform_roughness.txt. The ordinary route excludes other morphology. The exact constant-stage GPU recursive-AMR route may combine it with evolving physical-toe mode 2 or distributed mode 3 banks and one parent avalanche per accepted macro-step; other combinations fail closed.")
        self.bed_slope=QCheckBox("Enable slope correction");self.bed_slope_alfabs=_spin(0,100,1.,4,.1);self.bed_slope_alfabn=_spin(0,100,1.5,4,.1);self.bed_friction_angle=_spin(0,100,30.,4,.1)
        self.bed_slope_method_help=QLabel()
        self.bed_slope_method=QComboBox()
        for name,value in (("Automatic compatible default",None),("Delft3D FLOW Bagnold/Ikeda (ISlope=2)",5),("Delft3D Koch-Flokstra (ISlope=3)",3),("Legacy BRAIDS Bagnold/Ikeda",2)):
            self.bed_slope_method.addItem(name,value)
        self.bed_slope_method.currentIndexChanged.connect(self._update_sediment_controls)
        self.bed_slope_method_help.setWordWrap(True)
        self.bed_slope_alfabs_label=QLabel("Longitudinal slope factor (AlfaBs)")
        self.bed_slope_alfabn_label=QLabel("Transverse slope factor (AlfaBn)")
        self.bed_friction_angle_label=QLabel("Bed friction angle (°)")
        self.secondary_flow=QCheckBox("Enable curvature-driven transverse transport")
        self.secondary_flow.setToolTip("Equilibrium flow steers bed-load direction only. Prognostic flow also transports spiral intensity and can apply momentum feedback.")
        self.secondary_model=QComboBox()
        self.secondary_model.addItem(
            "Equilibrium (direction only)",_SECONDARY_EQUILIBRIUM)
        self.secondary_model.addItem(
            "Delft3D4 classic prognostic (Equili=N)",
            _SECONDARY_PROGNOSTIC_CLASSIC)
        self.secondary_model.addItem(
            "Legacy FM-style prognostic (historical v37)",
            _SECONDARY_PROGNOSTIC_LEGACY)
        self.secondary_transport_direction=QCheckBox("Steer bed-load direction");self.secondary_transport_direction.setChecked(False)
        self.secondary_transport_direction.setToolTip("Equilibrium flow always steers bed load. Prognostic flow can transport spiral intensity without bed-load steering.")
        self.secondary_beta=_spin(0,1.e9,0.,12,.05)
        self.secondary_beta.setToolTip(
            "Legacy FM-style BetaSpiral momentum feedback only. The classic "
            "Delft3D4 Equili=N route uses a distinct Betac control, fixed to "
            "zero in its first admitted scope.")
        self.secondary_flow_espir=_spin(0,100,1.,4,.1)
        self.secondary_flow_espir.setToolTip("Spiral-flow calibration coefficient. Default 1.0; enabled flow requires positive Manning roughness.")
        self.bedload_adaptation=QCheckBox("Enable non-equilibrium bed-load adaptation")
        self.bedload_adaptation.setToolTip("Optional qualified steady streamline lag. It is off by default and recovers the existing local-equilibrium solver when disabled.")
        self.bedload_adaptation_length=_spin(.001,1.e6,100.,3,10.)
        self.bedload_adaptation_length.setToolTip("Adaptation length for each sediment class, in that class's D50 grain diameters. The 100-D50 default is a measured moderate-transport mean and should be calibrated for the river.")
        self.morphological_acceleration=QCheckBox("Enable morphological acceleration (quasi-steady flow only)")
        self.morphological_acceleration.setToolTip("Qualified through MorFac 20 for the quasi-steady subcritical bedform, but MorFac 1 is required for wet/dry hydraulic transients. Suspended exchange, hydraulic boundary series, and bank erosion are intentionally incompatible.")
        self.morphological_factor=_spin(1.,20.,2.,2,1.)
        self.morphological_factor.setToolTip("Constant MorFac from 1 to 20. Transport capacity is unchanged; the conservative bed update uses dt multiplied by this factor.")
        self.morphological_factor_schedule=QLineEdit()
        self.morphological_factor_schedule.setPlaceholderText("optional: 0:1, 3600:2, 7200:5")
        self.morphological_factor_schedule.setToolTip("Optional Delft3D-compatible block time_s:factor knots. Each factor is held until the next knot; at least two strictly increasing knots are required.")
        self.erodible=_spin(0,1000,.001,6,.001);self.erodible_depth=_spin(0,10000,.25,3,.1)
        self.active_layer_thickness=_spin(0,1000,0.,6,.01);self.active_layer_thickness.setToolTip("Zero retains the total-bed fraction model. A positive value enables layered fractions and is the target in constant mode; depth-scaled mode uses the fraction and minimum controls below.")
        self.active_layer_mode=QComboBox();self.active_layer_mode.addItem("Constant thickness","constant");self.active_layer_mode.addItem("Scaled with water depth","water_depth")
        self.active_layer_mode.setToolTip("Constant thickness is the qualified default. Water-depth scaling uses max(depth fraction × local water depth, minimum thickness) and follows Delft3D TTLForm=2.")
        self.active_layer_depth_fraction=_spin(0,10,.008,6,.001)
        self.active_layer_depth_fraction.setToolTip("Dimensionless fraction of local water depth used by the water-depth-scaled active layer. Qualified default: 0.008.")
        self.active_layer_minimum_thickness=_spin(.000001,1000,.002,6,.001)
        self.active_layer_minimum_thickness.setToolTip("Lower bound for the water-depth-scaled active layer. Qualified default: 0.002 m.")
        self.substrate_layer_count=QSpinBox();self.substrate_layer_count.setRange(1,32);self.substrate_layer_count.setValue(4)
        self.substrate_layer_count.setToolTip("Number of ordered substrate layers preserved below the active layer. The deepest layer stores the remaining erodible bed.")
        self.substrate_layer_thickness=_spin(.000001,1000,.1,6,.01)
        self.substrate_layer_thickness.setToolTip("Thickness of each preserved substrate layer except the deepest variable-thickness base layer.")
        self.hiding_exposure=QComboBox();self.hiding_exposure.addItem("None",0);self.hiding_exposure.addItem("Egiazaroff",2);self.hiding_exposure.addItem("Ashida-Michiue",3);self.hiding_exposure.addItem("Parker-Klingeman-McLean",4);self.hiding_exposure.addItem("Wu-Wang-Jia",5)
        self.hiding_exposure.setToolTip("Uses the active-layer grain mixture to correct the MPM critical Shields threshold. Modes and equations follow Delft3D. Requires multiple fractions and a positive active-layer thickness.")
        self.hiding_parker_alpha=_spin(-10,10,.6,5,.05);self.hiding_parker_alpha.setToolTip("Alpha exponent used only by the Parker-Klingeman-McLean relation. Delft3D requires this calibration value explicitly; 0.6 is a starting value, not a universal calibration.")
        self.hiding_wu_exponent=_spin(-10,10,-.6,5,.05);self.hiding_wu_exponent.setToolTip("M exponent used only by the Wu-Wang-Jia relation. Delft3D requires this calibration value explicitly; -0.6 is a starting value, not a universal calibration.")
        self.sediment_shear_mode=QComboBox()
        self.sediment_shear_mode.addItem("Total hydraulic bed shear","total")
        self.sediment_shear_mode.addItem("Soulsby/Clarke skin shear (BsSkin)","soulsby_clarke_skin")
        self.sediment_shear_mode.setToolTip("BsSkin replaces only sediment RP_TAUB consumers. Hydraulic friction, effective velocity and the Wilcock/Gaeuman transport scale retain total roughness.")
        self.skin_kinematic_viscosity=_spin(1.e-12,1.,1.e-6,12,1.e-6)
        self.skin_sand_roughness=_spin(1.e-9,10.,.003,9,.0001)
        self.skin_silt_roughness=_spin(1.e-9,10.,.0003,9,.0001)
        self.skin_mud_low=_spin(0,.999999,.2,6,.05)
        self.skin_mud_high=_spin(.000001,1.,.8,6,.05)
        self.skin_fluff_fraction=_spin(0,1.,0.,6,.05)
        self.skin_shear_widgets=(self.skin_kinematic_viscosity,self.skin_sand_roughness,
            self.skin_silt_roughness,self.skin_mud_low,self.skin_mud_high,self.skin_fluff_fraction)
        for label,widget in (("Bed-load formulation",self.bedload),("Sediment bed shear",self.sediment_shear_mode),("Skin viscosity (m2/s)",self.skin_kinematic_viscosity),("Skin sand ks (m)",self.skin_sand_roughness),("Skin silt ks (m)",self.skin_silt_roughness),("Skin mud transition low",self.skin_mud_low),("Skin mud transition high",self.skin_mud_high),("Skin fluff fraction",self.skin_fluff_fraction),("Engelund-Hansen Chezy",self.engelund_hansen_chezy),("Transport calibration factor",self.transport_calibration),("Minimum bed-load depth (m)",self.minimum_bedload_depth),("MPM shear closure",self.mpm_friction_mode),("MPM friction coefficient",self.mpm_cf),("MPM critical Shields",self.mpm_critical),("Hiding/exposure correction",self.hiding_exposure),("Parker alpha exponent",self.hiding_parker_alpha),("Wu exponent",self.hiding_wu_exponent),("Slope correction",self.bed_slope),(self.bed_slope_alfabs_label,self.bed_slope_alfabs),(self.bed_slope_alfabn_label,self.bed_slope_alfabn),(self.bed_friction_angle_label,self.bed_friction_angle),("Default erodible thickness (m)",self.erodible),("Active layer thickness (m; 0 disables)",self.active_layer_thickness),("Active layer rule",self.active_layer_mode),("Depth fraction",self.active_layer_depth_fraction),("Minimum active thickness (m)",self.active_layer_minimum_thickness),("Preserved substrate layers",self.substrate_layer_count),("Substrate layer thickness (m)",self.substrate_layer_thickness),("Legacy minimum water depth (m)",self.erodible_depth)):form.addRow(label,widget)
        form.insertRow(form.getWidgetPosition(self.bed_slope)[0]+1,"Resolved method",self.bed_slope_method_help)
        form.insertRow(form.getWidgetPosition(self.bed_slope)[0]+1,"Compatible slope method",self.bed_slope_method)
        form.addRow("Secondary-flow transport",self.secondary_flow)
        form.addRow("Secondary-flow model",self.secondary_model)
        form.addRow(self.secondary_transport_direction)
        form.addRow("Legacy FM momentum Beta",self.secondary_beta)
        form.addRow("Spiral coefficient Espir",self.secondary_flow_espir)
        form.addRow("Non-equilibrium bed load",self.bedload_adaptation)
        form.addRow("Adaptation length (D50)",self.bedload_adaptation_length)
        form.addRow("Morphological acceleration",self.morphological_acceleration)
        form.addRow("Constant MorFac",self.morphological_factor)
        form.addRow("MorFac schedule",self.morphological_factor_schedule)
        self.morphological_acceleration_help=QLabel(
            "Qualified maximum: 20 for quasi-steady subcritical flow; 1 for wet/dry transients. "
            "Each run writes morphological_acceleration.txt with both clocks and the active factor.")
        self.morphological_acceleration_help.setWordWrap(True)
        form.addRow("Qualification",self.morphological_acceleration_help)
        form.addRow("Bedform roughness feedback",self.bedform_roughness)
        outer.addWidget(bed)

        spatial,grid=_section("Spatial bed inventory",QGridLayout,False)
        self.bedrock_layer=QgsMapLayerComboBox();self.bedrock_layer.setFilters(QgsMapLayerProxyModel.RasterLayer);self.bedrock_layer.setAllowEmptyLayer(True);self.bedrock_layer.setLayer(None)
        self.erodible_raster_layer=QgsMapLayerComboBox();self.erodible_raster_layer.setFilters(QgsMapLayerProxyModel.RasterLayer);self.erodible_raster_layer.setAllowEmptyLayer(True);self.erodible_raster_layer.setLayer(None)
        self.erodible_zone_layer=QgsMapLayerComboBox();self.erodible_zone_layer.setFilters(QgsMapLayerProxyModel.PolygonLayer);self.erodible_zone_layer.setAllowEmptyLayer(True);self.erodible_zone_layer.setLayer(None)
        self.erodible_zone_value=_spin(0,1000,.01,6,.001);self.create_erodible_button=QPushButton("Add New Erodible Zone");self.create_erodible_button.clicked.connect(self.create_erodible_zone)
        for row,(label,widget) in enumerate((("Bedrock elevation raster (optional)",self.bedrock_layer),("Erodible thickness raster (optional)",self.erodible_raster_layer),("Erodible thickness zones (optional)",self.erodible_zone_layer))):grid.addWidget(QLabel(label),row,0);grid.addWidget(widget,row,1,1,3)
        grid.addWidget(QLabel("New zone thickness (m)"),3,0);grid.addWidget(self.erodible_zone_value,3,1,1,3);grid.addWidget(self.create_erodible_button,4,0,1,4)
        note=QLabel("The uniform thickness above is the legacy route and the background for polygon zones. Raster values cover the full domain. Bedrock and thickness inputs must agree if both are supplied.");note.setWordWrap(True);grid.addWidget(note,5,0,1,4);outer.addWidget(spatial)

        general,general_vbox=_section("Domain-wide repose avalanching",QVBoxLayout,False);self.general_avalanche_section=general
        self.general_avalanche=QCheckBox("Enable wet/dry repose redistribution across the mobile bed")
        general_vbox.addWidget(self.general_avalanche)
        general_grid=QGridLayout();general_vbox.addLayout(general_grid)
        self.general_avalanche_wet_slope=_spin(.0001,10000,10.,4,.1)
        self.general_avalanche_dry_slope=_spin(.0001,10000,1.,4,.1)
        self.general_avalanche_time=_spin(.001,1e9,86400.,3,60.)
        self.general_avalanche_relaxation_length=_spin(.0001,1e6,1.,4,.1)
        self.general_avalanche_wet_depth_switch=_spin(0,10000,.1,4,.01)
        for row,(label,widget) in enumerate((
            ("Maximum wet slope (vertical/horizontal)",self.general_avalanche_wet_slope),
            ("Maximum dry slope (vertical/horizontal)",self.general_avalanche_dry_slope),
            ("Avalanche relaxation time (s)",self.general_avalanche_time),
            ("Relaxation length (m)",self.general_avalanche_relaxation_length),
            ("Wet/dry slope transition depth (m)",self.general_avalanche_wet_depth_switch))):
            general_grid.addWidget(QLabel(label),row,0);general_grid.addWidget(widget,row,1)
        general_note=QLabel("Qualified against the locked current-source D-Flow FM duneaval operator for wet and shallow/dry repose cases. It conserves every grain fraction and leaves calculation-domain walls fixed. This is bar/channel slope redistribution, not bank undercutting or cantilever collapse.")
        general_note.setWordWrap(True);general_vbox.addWidget(general_note)
        self.general_avalanche_controls=(self.general_avalanche_wet_slope,self.general_avalanche_dry_slope,
            self.general_avalanche_time,self.general_avalanche_relaxation_length,self.general_avalanche_wet_depth_switch)
        self.general_avalanche.toggled.connect(self._domain_avalanche_toggled)
        outer.addWidget(general)

        bank,bank_vbox=_section("Erodible banks and slope failure",QVBoxLayout,False);self.bank_section=bank
        self.bank_failure_enabled=QCheckBox("Enable process-separated toe erosion and bank failure");bank_vbox.addWidget(self.bank_failure_enabled)
        self.bank_morphology_enabled=QCheckBox("Enable Delft3D dry-bank supply (legacy)");bank_vbox.addWidget(self.bank_morphology_enabled)
        bank_grid=QGridLayout();bank_vbox.addLayout(bank_grid)
        self.bank_zone_layer=QgsMapLayerComboBox();self.bank_zone_layer.setFilters(QgsMapLayerProxyModel.PolygonLayer);self.bank_zone_layer.setAllowEmptyLayer(True);self.bank_zone_layer.setLayer(None)
        self.create_bank_zone_button=QPushButton("Add New Erodible Bank Zone");self.create_bank_zone_button.clicked.connect(self.create_bank_zone)
        self.bank_toe_mode=QComboBox()
        self.bank_toe_mode.addItem("Cohesive excess-shear retreat","cohesive_excess_shear")
        self.bank_toe_mode.addItem("Noncohesive toe transport capacity","noncohesive_transport_capacity")
        self.bank_toe_mode.addItem("Distributed slope transport","distributed_slope_transport")
        self.bank_erosion_fraction=_spin(0,1,.5,3,.05)
        self.bank_avalanche=QCheckBox("Enable slope-failure redistribution");self.bank_avalanche.setChecked(False)
        self.bank_wet_slope=_spin(.0001,10000,10.,4,.1);self.bank_dry_slope=_spin(.0001,10000,1.,4,.1)
        self.bank_avalanche_time=_spin(.001,1e9,86400.,3,60.)
        self.bank_relaxation_length=_spin(.0001,1e6,1.,4,.1)
        self.bank_wet_depth_switch=_spin(0,10000,.1,4,.01)
        self.bank_toe_erodibility=_spin(1e-12,1.,1.3e-6,9,1e-7)
        self.bank_toe_critical_shear=_spin(0,1e6,32.5,4,.5)
        self.bank_toe_exponent=_spin(.01,10.,1.,4,.1)
        self.bank_toe_capacity_fraction=_spin(0,1,1.,4,.05)
        self.bank_distributed_transport_fraction=_spin(0,1,.1,4,.01)
        self.bank_distributed_critical_slope=_spin(.0001,1000,.35,4,.05)
        self.bank_critical_undercut=_spin(.0001,1e5,.1,4,.01)
        bank_grid.addWidget(QLabel("Erodible bank zones"),0,0);bank_grid.addWidget(self.bank_zone_layer,0,1,1,2);bank_grid.addWidget(self.create_bank_zone_button,0,3)
        bank_grid.addWidget(QLabel("Toe-erosion law"),1,0,1,2);bank_grid.addWidget(self.bank_toe_mode,1,2,1,2)
        bank_grid.addWidget(QLabel("Bank erosion updates bed elevation and flow at every step."),2,0,1,4)
        bank_grid.addWidget(QLabel("Toe transport capacity fraction (0-1)"),3,0,1,2);bank_grid.addWidget(self.bank_toe_capacity_fraction,3,2,1,2)
        bank_grid.addWidget(QLabel("Distributed transport fraction alpha (0-1)"),4,0,1,2);bank_grid.addWidget(self.bank_distributed_transport_fraction,4,2,1,2)
        bank_grid.addWidget(QLabel("Distributed critical slope (vertical/horizontal)"),5,0,1,2);bank_grid.addWidget(self.bank_distributed_critical_slope,5,2,1,2)
        bank_grid.addWidget(QLabel("Toe erodibility kd (m/(Pa s))"),6,0,1,2);bank_grid.addWidget(self.bank_toe_erodibility,6,2,1,2)
        bank_grid.addWidget(QLabel("Critical toe shear stress (Pa)"),7,0,1,2);bank_grid.addWidget(self.bank_toe_critical_shear,7,2,1,2)
        bank_grid.addWidget(QLabel("Excess-shear exponent"),8,0,1,2);bank_grid.addWidget(self.bank_toe_exponent,8,2,1,2)
        bank_grid.addWidget(QLabel("Critical undercut before collapse (m)"),9,0,1,2);bank_grid.addWidget(self.bank_critical_undercut,9,2,1,2)
        bank_grid.addWidget(QLabel("Legacy: channel erosion supplied by dry bank (0-1)"),10,0,1,2);bank_grid.addWidget(self.bank_erosion_fraction,10,2,1,2)
        bank_grid.addWidget(self.bank_avalanche,11,0,1,4)
        bank_grid.addWidget(QLabel("Maximum wet slope (vertical/horizontal)"),12,0,1,2);bank_grid.addWidget(self.bank_wet_slope,12,2,1,2)
        bank_grid.addWidget(QLabel("Maximum dry slope (vertical/horizontal)"),13,0,1,2);bank_grid.addWidget(self.bank_dry_slope,13,2,1,2)
        bank_grid.addWidget(QLabel("Avalanche relaxation time (s)"),14,0,1,2);bank_grid.addWidget(self.bank_avalanche_time,14,2,1,2)
        bank_grid.addWidget(QLabel("Relaxation length (m)"),15,0,1,2);bank_grid.addWidget(self.bank_relaxation_length,15,2,1,2)
        self.bank_relaxation_length.setToolTip("Physical distance relaxed over the configured timescale. The bank diffusivity is L²/(2T), independent of mesh spacing.")
        bank_grid.addWidget(QLabel("Wet/dry slope transition depth (m)"),16,0,1,2);bank_grid.addWidget(self.bank_wet_depth_switch,16,2,1,2)
        self.bank_wet_depth_switch.setToolTip("Depth over which the maximum slope changes continuously from the dry-bank value to the wet-bank value.")
        note=QLabel("Choose the toe law explicitly. Cohesive excess shear uses kd, critical shear and its exponent; noncohesive capacity uses the capacity fraction; distributed slope transport uses alpha and the critical slope. Bank shape evolves through the bed elevation. Enable Domain-wide bed avalanching above for post-collapse repose redistribution. The legacy Delft3D dry-bank option is retained only for comparison.");note.setWordWrap(True);bank_vbox.addWidget(note)
        self.bank_controls=(self.bank_zone_layer,self.create_bank_zone_button,self.bank_erosion_fraction,
            self.bank_avalanche,self.bank_wet_slope,self.bank_dry_slope,self.bank_avalanche_time,
            self.bank_relaxation_length,self.bank_wet_depth_switch,self.bank_toe_mode,
            self.bank_toe_capacity_fraction,
            self.bank_distributed_transport_fraction,self.bank_distributed_critical_slope,
            self.bank_toe_erodibility,self.bank_toe_critical_shear,self.bank_toe_exponent,
            self.bank_critical_undercut)
        self.bank_failure_enabled.toggled.connect(self._bank_failure_toggled)
        self.bank_morphology_enabled.toggled.connect(self._legacy_bank_toggled)
        self.bank_avalanche.toggled.connect(self._bank_avalanche_toggled)
        self.bank_toe_mode.currentIndexChanged.connect(self._update_sediment_controls)
        outer.addWidget(bank)

        supply,vbox=_section("Inlet sediment supply by boundary ID",QVBoxLayout,False)
        self.sediment_supply_section=supply
        note=QLabel("Each inlet ID has constant or piecewise-linear concentration and bed-load controls. Plot Time Series opens the forcing in a separate window.");note.setWordWrap(True);vbox.addWidget(note)
        refresh_supply=QPushButton("Refresh Inlet IDs from General");refresh_supply.clicked.connect(self.refresh_sediment_boundary_json);vbox.addWidget(refresh_supply)
        self.sediment_cards_widget=QWidget();self.sediment_cards_layout=QVBoxLayout(self.sediment_cards_widget);self.sediment_cards_layout.setContentsMargins(0,0,0,0);vbox.addWidget(self.sediment_cards_widget)
        self.sediment_boundary_json=QPlainTextEdit();self.sediment_boundary_json.setPlainText("{}");self.sediment_boundary_json.setVisible(False);vbox.addWidget(self.sediment_boundary_json);outer.addWidget(supply)
        outputs,vbox=_section("Morphology result fields",QVBoxLayout,False)
        output_note=QLabel("Choose additional typed Delft3D-compatible morphology fields to retain in the portable run identity. Unsupported wave/FM-only fields remain visible with their reason and cannot be selected.")
        output_note.setWordWrap(True);vbox.addWidget(output_note)
        self.morphology_output_defaults=QCheckBox("Apply supported Delft3D source defaults")
        self.morphology_output_defaults.setChecked(False);vbox.addWidget(self.morphology_output_defaults)
        self.morphology_output_table=QTableWidget(0,4)
        self.morphology_output_table.setHorizontalHeaderLabels(("Use","FLOW keyword","Value","Product status"))
        self.morphology_output_table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeMode.ResizeToContents)
        self.morphology_output_table.horizontalHeader().setSectionResizeMode(3,QHeaderView.ResizeMode.Stretch)
        self._morphology_output_rows=[]
        for definition in morphology_output_choices():
            row=self.morphology_output_table.rowCount();self.morphology_output_table.insertRow(row)
            enabled=QTableWidgetItem();enabled.setCheckState(Qt.CheckState.Unchecked)
            keyword=QTableWidgetItem(definition["keyword"]);keyword.setFlags(keyword.flags() & ~Qt.ItemFlag.ItemIsEditable)
            status=("Available" if definition["supported"] else
                    "Excluded: "+str(definition["reason"]))
            status_item=QTableWidgetItem(status);status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            editor=None;kind=definition["value_type"]
            if kind=="boolean":
                value=QTableWidgetItem("true when selected");value.setFlags(value.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.morphology_output_table.setItem(row,2,value)
            elif kind=="integer_enum":
                editor=QComboBox()
                for allowed in definition.get("allowed") or ():editor.addItem(str(allowed),int(allowed))
                self.morphology_output_table.setCellWidget(row,2,editor)
            else:
                editor=QLineEdit()
                editor.setPlaceholderText("three numbers" if kind=="number_vector" else
                    "true or percentiles" if kind=="boolean_or_percentile_list" else "selector")
                self.morphology_output_table.setCellWidget(row,2,editor)
            if not definition["supported"]:
                enabled.setFlags(enabled.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                if editor is not None:editor.setEnabled(False)
            self.morphology_output_table.setItem(row,0,enabled)
            self.morphology_output_table.setItem(row,1,keyword)
            self.morphology_output_table.setItem(row,3,status_item)
            self._morphology_output_rows.append((definition,enabled,editor))
        self.morphology_output_table.setMinimumHeight(250);vbox.addWidget(self.morphology_output_table)
        outer.addWidget(outputs)
        summary_section,vbox=_section("Resolved input summary",QVBoxLayout,False)
        refresh=QPushButton("Refresh Resolved Input Summary");refresh.clicked.connect(self._refresh_input_summary);vbox.addWidget(refresh)
        self.input_summary=QPlainTextEdit();self.input_summary.setReadOnly(True);self.input_summary.setMinimumHeight(170);vbox.addWidget(self.input_summary)
        self.product_summary=QPlainTextEdit();self.product_summary.setReadOnly(True);self.product_summary.setMinimumHeight(110)
        self.product_summary.setPlaceholderText("Save the project to resolve its portable CPU/GPU product identity.")
        vbox.addWidget(QLabel("Resolved product identity"));vbox.addWidget(self.product_summary);outer.addWidget(summary_section)
        self.sediment_sections=(material,fractions,suspended,bed,spatial,general,bank,supply,outputs,summary_section)
        self._slope_parameter_law=self.bedload.currentData()
        self._mode2_slope_values=(self.bed_slope_alfabs.value(),self.bed_slope_alfabn.value(),self.bed_friction_angle.value())
        self._mode3_slope_values=(0.,.85,.5)
        self.bedload_enabled.toggled.connect(self._update_sediment_controls)
        self.suspended_enabled.toggled.connect(self._update_sediment_controls)
        self.bedload.currentIndexChanged.connect(self._update_sediment_controls);self.sediment_shear_mode.currentIndexChanged.connect(self._update_sediment_controls);self.mpm_friction_mode.currentIndexChanged.connect(self._update_sediment_controls);self.hiding_exposure.currentIndexChanged.connect(self._update_sediment_controls);self.bed_slope.toggled.connect(self._update_sediment_controls);self.secondary_flow.toggled.connect(self._update_sediment_controls);self.secondary_model.currentIndexChanged.connect(self._update_sediment_controls);self.secondary_transport_direction.toggled.connect(self._update_sediment_controls);self.bedload_adaptation.toggled.connect(self._update_sediment_controls);self.morphological_acceleration.toggled.connect(self._update_sediment_controls);self.bedform_roughness.toggled.connect(self._update_sediment_controls)
        self.exchange.currentIndexChanged.connect(self._update_sediment_controls);self.settling_mode.currentIndexChanged.connect(self._update_sediment_controls);self.sediment_roughness.currentIndexChanged.connect(self._update_sediment_controls)
        self.hydraulic_roughness_mode.currentIndexChanged.connect(self._update_sediment_controls)
        self.active_layer_thickness.valueChanged.connect(self._update_sediment_controls);self.active_layer_mode.currentIndexChanged.connect(self._update_sediment_controls);self.substrate_layer_count.valueChanged.connect(self._update_sediment_controls)
        self.sediment_enabled.toggled.connect(self._update_sediment_controls);self._update_sediment_controls()
        outer.addStretch();scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff);scroll.setWidget(content);self.tabs.addTab(scroll,"Sediment")

    def _open_adaptive_river(self):
        from .adaptive_river import AdaptiveRiverDialog
        if not hasattr(self,'adaptive_river_dialog'):
            self.adaptive_river_dialog=AdaptiveRiverDialog(self)
        self.adaptive_river_dialog.show();self.adaptive_river_dialog.raise_()

    def _run_tab(self):
        page=QWidget();self.run_page=page;layout=QVBoxLayout(page)
        backend_panel=QWidget();backend_layout=QVBoxLayout(backend_panel)
        backend_layout.setContentsMargins(0,0,0,0)
        backend_form=QFormLayout();self.execution_backend=QComboBox()
        self.execution_backend.addItem("CPU (native)","cpu")
        self.execution_backend.addItem("GPU (external implicit f-wave worker)","gpu")
        backend_form.addRow("Compute backend",self.execution_backend);backend_layout.addLayout(backend_form)
        self.gpu_settings,gpu_form=_section("External GPU worker",QFormLayout,False)
        self.gpu_worker_folder=QLineEdit();self.gpu_worker_folder.setPlaceholderText("Extracted GPU source-worker folder")
        row=QHBoxLayout();row.addWidget(self.gpu_worker_folder,1)
        browse=QPushButton("Browse…");browse.clicked.connect(self._browse_gpu_worker);row.addWidget(browse)
        gpu_form.addRow("Worker folder",row)
        self.gpu_python=QLineEdit();self.gpu_python.setPlaceholderText("External Python with CUDA/CuPy (not QGIS Python)")
        row=QHBoxLayout();row.addWidget(self.gpu_python,1)
        browse=QPushButton("Browse…");browse.clicked.connect(self._browse_gpu_python);row.addWidget(browse)
        gpu_form.addRow("GPU Python",row)
        self.gpu_detect_button=QPushButton("Detect Windows GPU setup")
        self.gpu_detect_button.setToolTip(
            "Find the newest authenticated BRAIDS worker and likely external "
            "CUDA Python installed on this PC. Detection fills paths only; "
            "verification still hashes both and runs the FP64 CUDA probe.")
        self.gpu_detect_button.clicked.connect(self._detect_gpu_setup)
        gpu_form.addRow(self.gpu_detect_button)
        self.gpu_verify_button=QPushButton("Verify and pin worker + Python")
        self.gpu_verify_button.clicked.connect(self._verify_gpu_worker)
        self.gpu_identity_status=QLabel("No external worker is verified for this session.")
        self.gpu_identity_status.setWordWrap(True)
        gpu_form.addRow(self.gpu_verify_button);gpu_form.addRow(self.gpu_identity_status)
        self.gpu_variant=QComboBox();self.gpu_variant.addItem("Integrated (selected physics)",INTEGRATED)
        self.gpu_variant.addItem("auto","auto")
        self.gpu_launch_mode=QComboBox();self.gpu_launch_mode.addItem("CUDA graph","graph");self.gpu_launch_mode.addItem("Plain launches","plain")
        self.gpu_integrator=QComboBox();self.gpu_integrator.addItem("SSPRK(4,3)","ssp43");self.gpu_integrator.addItem("SSPRK(3,3)","ssp33")
        self.gpu_integrator.addItem("Implicit f-wave midpoint",
                                    "implicit_fwave_midpoint")
        self.gpu_integrator.setCurrentIndex(
            self.gpu_integrator.findData("implicit_fwave_midpoint"))
        self.gpu_integrator.setToolTip(
            "The implicit midpoint option retains the augmented f-wave spatial solver. "
            "It is selected explicitly, never as a CPU/GPU fallback, and unsupported "
            "boundaries or physical processes are rejected before execution.")
        gpu_form.addRow("Numerical variant",self.gpu_variant)
        self.gpu_selector_policy=QComboBox()
        self.gpu_selector_policy.addItem("Legacy free-surface jump",None)
        self.gpu_selector_policy.addItem("Slope-aware free-surface residual",_GENERAL_AMR_SLOPE_POLICY)
        self.gpu_selector_policy.setEnabled(False)
        self.gpu_selector_policy.setToolTip("For nonbank GPU AMR, with synchronous or recursive stepping. The slope-aware rule avoids refining a smooth water-surface gradient everywhere; it requires a versioned worker and cannot be changed on restart.")
        self.gpu_selector_policy.activated.connect(
            lambda *_: setattr(self,'_gpu_selector_user_choice',True))
        gpu_form.addRow("AMR refinement selector",self.gpu_selector_policy)
        gpu_form.addRow("Launch mode",self.gpu_launch_mode)
        gpu_form.addRow("Time integrator",self.gpu_integrator)
        self.gpu_amr_evolving_banks=QCheckBox("Evolving bank geometry")
        self.gpu_amr_evolving_banks.setChecked(False)
        self.gpu_amr_evolving_banks.setEnabled(False)
        self.gpu_amr_evolving_banks.setToolTip("Explicit experimental on-demand AMR option. Mode 2 rebuilds physical-toe mappings from the evolving bed; mode 3 uses distributed bank erosion without a persistent toe map. Both preserve their own erosion histories during mesh changes. Supports the declared surveyed inlet/critical outlet or constant stage-outlet configuration. This is not persistent subcell bank-front tracking.")
        gpu_form.addRow(self.gpu_amr_evolving_banks)
        self.gpu_scope_note=QLabel()
        self.gpu_scope_note.setWordWrap(True);gpu_form.addRow(self.gpu_scope_note);backend_layout.addWidget(self.gpu_settings)
        self.backend_panel=backend_panel
        self.general_layout.insertWidget(1,backend_panel)
        self.gpu_worker_folder.textChanged.connect(self._invalidate_gpu_identity)
        self.gpu_python.textChanged.connect(self._invalidate_gpu_identity)
        self.gpu_variant.currentIndexChanged.connect(self._gpu_variant_changed)
        self.gpu_integrator.currentIndexChanged.connect(self._gpu_variant_changed)
        self.amr_enabled.toggled.connect(self._gpu_variant_changed)
        self.amr_subcycling.toggled.connect(self._gpu_variant_changed)
        self.bank_failure_enabled.toggled.connect(self._gpu_variant_changed)
        self.gpu_amr_evolving_banks.toggled.connect(self._update_sediment_controls)
        self.amr_enabled.toggled.connect(self._update_sediment_controls)
        self.amr_subcycling.toggled.connect(self._update_sediment_controls)
        self.gpu_variant.currentIndexChanged.connect(self._update_sediment_controls)
        QTimer.singleShot(0,self._populate_gpu_defaults)
        form=QFormLayout()
        self.end_time=_spin(.001,1e9,600,3,10);self.output_interval=_spin(.001,1e9,150,3,10);self.dt_max=_spin(1e-8,1e6,.05,6,.01)
        self.cfl=_spin(.01,.9,.45,3,.05);self.cfl.setToolTip("Maximum accepted Euler-stage finite-volume Courant number; the solver never silently changes this input. Automatic CPU integration selects SSPMS6 for eligible fixed-bed hydraulics, SSPRK(4,3) for eligible fixed-grid MPM/secondary-flow runs, and the legacy method otherwise. The established CFL default remains 0.45. CFL 0.8 was tested in the bend comparison; check accuracy for your case. The selected method is reported in the execution log.")
        hardware_threads=max(1,os.cpu_count() or 1);self.threads=QSpinBox();self.threads.setRange(1,hardware_threads);self.threads.setValue(recommended_openmp_threads())
        self.threads.setToolTip("Threads used by the separate native OpenMP solver. This is independent of QGIS's own thread-pool setting.")
        for label,widget in (("End time (s)",self.end_time),("Output interval (s)",self.output_interval),("Maximum timestep (s)",self.dt_max),("CFL limit",self.cfl),("OpenMP threads",self.threads)):form.addRow(label,widget)
        self.profile_native=QCheckBox("Write native performance profile to execution log")
        self.profile_native.setToolTip("Reports wall time in AMR regridding, hydraulic preparation, Euler updates and source terms. It only observes timing; it does not change the numerical solution.")
        form.addRow("Performance analysis",self.profile_native)
        self.horizontal_viscosity=_spin(0.,1e6,0.,8,.001)
        self.horizontal_viscosity.setToolTip('Explicit eddy viscosity in m2/s. Zero disables momentum diffusion. Integrated GPU transport requires velocity-Laplacian formulation when positive.')
        self.horizontal_viscosity_formulation=QComboBox()
        self.horizontal_viscosity_formulation.addItem('Depth-conservative','depth_conservative')
        self.horizontal_viscosity_formulation.addItem('Velocity Laplacian','velocity_laplacian')
        form.addRow('Horizontal viscosity (m2/s)',self.horizontal_viscosity)
        form.addRow('Viscosity formulation',self.horizontal_viscosity_formulation)
        layout.addLayout(form)
        checkpoint_row=QHBoxLayout();self.checkpoint=QLineEdit();self.checkpoint.setPlaceholderText("Optional checkpoint.bin");checkpoint_row.addWidget(self.checkpoint);button=QPushButton("Browse…");button.clicked.connect(self._browse_checkpoint);checkpoint_row.addWidget(button);layout.addWidget(QLabel("Restart checkpoint"));layout.addLayout(checkpoint_row)
        live_row=QHBoxLayout();self.live_preview=QCheckBox("Show live preview");self.live_preview.setChecked(True)
        self.live_variable=QComboBox();self.live_variable.addItem("Water depth","depth_m");self.live_variable.addItem("Flow speed","velocity_m_s");self.live_variable.addItem("Water level","stage_m")
        live_row.addWidget(self.live_preview);live_row.addWidget(QLabel("Variable"));live_row.addWidget(self.live_variable,1);layout.addLayout(live_row)
        preparation_help=QLabel("Prepare Run saves the project and builds the mesh, terrain, initial fields and native input deck. It does not start the simulation.");preparation_help.setWordWrap(True);layout.addWidget(preparation_help)
        buttons=QHBoxLayout();self.prepare_run_button=QPushButton("Prepare Run");self.prepare_run_button.clicked.connect(self.prepare_run_ui);self.run_button=QPushButton("Run Prepared Case");self.run_button.clicked.connect(self.start_run);self.run_button.setEnabled(False);self.stop_button=QPushButton("Stop");self.stop_button.clicked.connect(self.stop_current_operation);self.stop_button.setEnabled(False);buttons.addWidget(self.prepare_run_button);buttons.addWidget(self.run_button);buttons.addWidget(self.stop_button);layout.addLayout(buttons)
        self.progress=QProgressBar();self.progress.setRange(0,100);layout.addWidget(self.progress)
        self.run_status=QLabel("No prepared run");layout.addWidget(self.run_status)
        self.live_status=QLabel("Live preview uses the native fixed-grid output and updates after each saved time.");self.live_status.setWordWrap(True);layout.addWidget(self.live_status)
        self.live_preview.toggled.connect(self._toggle_live_preview);self.live_variable.currentIndexChanged.connect(self._refresh_live_preview)
        self.log_toggle=QToolButton();self.log_toggle.setText("Show execution log");self.log_toggle.setCheckable(True);layout.addWidget(self.log_toggle)
        self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setVisible(False);self.log.setMinimumHeight(140)
        self.log.setPlaceholderText("Solver output is recorded here and saved as native.log in the prepared run folder.")
        self.log_toggle.toggled.connect(self.log.setVisible);layout.addWidget(self.log,1);layout.addStretch();self.tabs.addTab(page,"Run")
        self.execution_backend.currentIndexChanged.connect(self._backend_changed)
        self._backend_changed(persist=False)

    def _results_tab(self):
        page=QWidget();layout=QVBoxLayout(page);self.results_status=QLabel("Complete a run before loading results.");self.results_status.setWordWrap(True);layout.addWidget(self.results_status)
        runs,vbox=_section("Runs",QVBoxLayout,True);self.results_runs_section=runs
        note=QLabel("Choose a completed project run or open a completed GPU worker output folder. If its prepared input was moved, select the authenticated input folder when preparing results. Rasters from several runs may remain loaded together in QGIS.");note.setWordWrap(True);vbox.addWidget(note)
        row=QHBoxLayout();self.results_run_combo=QComboBox();row.addWidget(self.results_run_combo,1);self.refresh_results_runs_button=QPushButton("Refresh");self.refresh_results_runs_button.clicked.connect(self._refresh_results_runs);row.addWidget(self.refresh_results_runs_button)
        self.open_result_run_button=QPushButton("Open Run Folder…");self.open_result_run_button.clicked.connect(self.open_result_run_folder);row.addWidget(self.open_result_run_button);vbox.addLayout(row);layout.addWidget(runs)
        self.results_run_combo.currentIndexChanged.connect(self._result_run_changed)
        layers,vbox=_section("Raster results",QVBoxLayout,False);self.result_raster_section=layers
        self.prepare_results_button=QPushButton("Prepare Raster Results");self.prepare_results_button.clicked.connect(self.prepare_results_ui);vbox.addWidget(self.prepare_results_button)
        selection=QFormLayout();self.result_variable=QComboBox();self.frame_combo=QComboBox();selection.addRow("Variable",self.result_variable);selection.addRow("Saved time",self.frame_combo);vbox.addLayout(selection)
        self.load_result_button=QPushButton("Load Selected Raster");self.load_result_button.clicked.connect(self.load_selected_frame);vbox.addWidget(self.load_result_button);layout.addWidget(layers)
        self.show_result_mesh=QCheckBox("Show computational mesh")
        self.show_result_mesh.setEnabled(False);self.show_result_mesh.toggled.connect(self._toggle_result_mesh)
        self.show_result_mesh.setToolTip("Overlay the fixed grid or final AMR mesh used by the completed run.")
        vbox.addWidget(self.show_result_mesh)
        animation,vbox=_section("Animation",QVBoxLayout,False);self.animation_section=animation
        animation_help=QLabel("Load the selected variable as a multiband raster, then play or seek the saved simulation times directly on the map.");animation_help.setWordWrap(True);vbox.addWidget(animation_help)
        self.animate_result_mesh=QCheckBox("Animate adaptive mesh (slow)")
        self.animate_result_mesh.setChecked(False);self.animate_result_mesh.setEnabled(False)
        self.animate_result_mesh.toggled.connect(self._toggle_animated_result_mesh)
        self.animate_result_mesh.setToolTip("Build and show every saved AMR mesh. The on-disk mesh cache is retained when this is unchecked, so later loads are immediate.")
        vbox.addWidget(self.animate_result_mesh)
        self.result_mesh_progress=QProgressBar();self.result_mesh_progress.setRange(0,100)
        self.result_mesh_progress.setValue(0);self.result_mesh_progress.setFormat("Adaptive mesh: %p%")
        self.result_mesh_progress.setVisible(False);vbox.addWidget(self.result_mesh_progress)
        self.load_animation_button=QPushButton("Load Time Series");self.load_animation_button.clicked.connect(self.load_result_animation);vbox.addWidget(self.load_animation_button)
        player=QWidget();player_layout=QHBoxLayout(player);player_layout.setContentsMargins(0,0,0,0)
        self.result_frame_slider=QSlider(Qt.Orientation.Horizontal);self.result_frame_slider.setEnabled(False)
        self.result_play_button=QPushButton("Play");self.result_pause_button=QPushButton("Pause");self.result_restart_button=QPushButton("Restart")
        self.result_playback_fps=QSpinBox();self.result_playback_fps.setRange(1,30);self.result_playback_fps.setValue(5);self.result_playback_fps.setSuffix(" fps")
        for control in (self.result_play_button,self.result_pause_button,self.result_restart_button,self.result_playback_fps):control.setEnabled(False)
        for control in (self.result_frame_slider,self.result_play_button,self.result_pause_button,self.result_restart_button,self.result_playback_fps):player_layout.addWidget(control,1 if control is self.result_frame_slider else 0)
        vbox.addWidget(player);self.result_frame_status=QLabel("Load a time series to enable playback.");vbox.addWidget(self.result_frame_status)
        self.result_player_timer=QTimer(self);self.result_player_timer.timeout.connect(self._advance_result_frame)
        self.result_frame_slider.valueChanged.connect(self._set_result_frame);self.result_play_button.clicked.connect(self._start_result_animation)
        self.result_pause_button.clicked.connect(self.result_player_timer.stop);self.result_restart_button.clicked.connect(self._restart_result_animation)
        self.result_playback_fps.valueChanged.connect(self._update_result_animation_rate)
        self.export_result_png_frames_button=QPushButton("Export Animation PNG Frames…");self.export_result_png_frames_button.clicked.connect(self.export_result_png_frames);vbox.addWidget(self.export_result_png_frames_button)
        export_note=QLabel("Exports the currently visible QGIS map, including any displayed computational mesh, once per saved time.");export_note.setWordWrap(True);vbox.addWidget(export_note);layout.addWidget(animation)
        boundary,vbox=_section("Boundary flow",QVBoxLayout,False);self.result_boundary_section=boundary
        boundary_note=QLabel("Uses the solver's accepted face-flux ledger. All plotted and exported values are positive magnitudes; the boundary mode identifies inflow or outflow. This is the authoritative boundary flow, unlike a cell-centred raster transect drawn near an edge.");boundary_note.setWordWrap(True);vbox.addWidget(boundary_note)
        boundary_form=QFormLayout();self.result_boundary_id=QComboBox();boundary_form.addRow("Boundary",self.result_boundary_id);vbox.addLayout(boundary_form)
        boundary_row=QHBoxLayout();self.plot_result_boundary_button=QPushButton("Plot Boundary Flow");self.plot_result_boundary_button.clicked.connect(self.plot_result_boundary);self.export_result_boundary_button=QPushButton("Export Boundary CSV…");self.export_result_boundary_button.clicked.connect(self.export_result_boundary_csv);boundary_row.addWidget(self.plot_result_boundary_button);boundary_row.addWidget(self.export_result_boundary_button);vbox.addLayout(boundary_row);layout.addWidget(boundary)
        analysis,vbox=_section("Result probes and flow transects",QVBoxLayout,False);self.result_analysis_section=analysis
        self.draw_result_probe_button=QPushButton("Draw Point Probe");self.draw_result_probe_button.clicked.connect(self.draw_result_probe);vbox.addWidget(self.draw_result_probe_button)
        self.edit_result_probe_button=QPushButton("Edit Probe");self.edit_result_probe_button.clicked.connect(self.edit_result_probe);vbox.addWidget(self.edit_result_probe_button)
        probe_row=QHBoxLayout();self.plot_result_probe_button=QPushButton("Plot Probe");self.plot_result_probe_button.clicked.connect(self.plot_result_probe);self.export_result_probe_button=QPushButton("Export Probe CSV…");self.export_result_probe_button.clicked.connect(self.export_result_probe_csv);probe_row.addWidget(self.plot_result_probe_button);probe_row.addWidget(self.export_result_probe_button);vbox.addLayout(probe_row)
        self.draw_result_transect_button=QPushButton("Draw Flow Transect");self.draw_result_transect_button.clicked.connect(self.draw_result_transect);vbox.addWidget(self.draw_result_transect_button)
        self.edit_result_transect_button=QPushButton("Edit Transect");self.edit_result_transect_button.clicked.connect(self.edit_result_transect);vbox.addWidget(self.edit_result_transect_button)
        transect_row=QHBoxLayout();self.plot_result_transect_button=QPushButton("Plot Transect Flow");self.plot_result_transect_button.clicked.connect(self.plot_result_transect);self.export_result_transect_button=QPushButton("Export Transect CSV…");self.export_result_transect_button.clicked.connect(self.export_result_transect_csv);transect_row.addWidget(self.plot_result_transect_button);transect_row.addWidget(self.export_result_transect_button);vbox.addLayout(transect_row)
        analysis_note=QLabel("Probe and transect values come from cell-centred result rasters. Transect discharge is signed: positive is toward the left side of the digitized line. Use Boundary Flow above—not a line drawn on the domain edge—to verify an imposed inlet or outlet flux.");analysis_note.setWordWrap(True);vbox.addWidget(analysis_note);layout.addWidget(analysis)
        layout.addStretch();self.tabs.addTab(page,"Results")

    def _message(self,title,text,critical=False):
        (QMessageBox.critical if critical else QMessageBox.information)(self,title,text)

    def _activate_select_tool(self,layer,feature_label):
        """Activate QGIS feature selection for the requested input layer."""
        try:
            if layer is None or not layer.isValid():raise ValueError(f"Choose a {feature_label} layer first")
            if self.iface is None:raise ValueError("Open BRAIDS inside QGIS to select map features")
            if layer is self.edge_layer.currentLayer():self._hide_inlet_guides()
            self.iface.setActiveLayer(layer)
            action_factory=getattr(self.iface,"actionSelect",None)
            if not callable(action_factory):raise ValueError("This QGIS interface does not expose the Select Features tool")
            action_factory().trigger()
            self.setup_status.setText(f"Select the {feature_label}(s) on the map, then use the adjacent action button.")
        except Exception as exc:self._message("Select features",str(exc),True)

    def _select_tab(self,name):
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index).replace("&&","&")==name:self.tabs.setCurrentIndex(index);return

    def _domain_avalanche_toggled(self,checked):
        if checked and self.bank_avalanche.isChecked():
            self.bank_avalanche.setChecked(False)
        self._update_sediment_controls()

    def _bank_avalanche_toggled(self,checked):
        if checked and self.general_avalanche.isChecked():
            self.general_avalanche.setChecked(False)
        self._update_sediment_controls()

    def _bank_failure_toggled(self,checked):
        if checked:
            self.bank_morphology_enabled.setChecked(False)
        self._update_sediment_controls()

    def _legacy_bank_toggled(self,checked):
        if checked:
            self.bank_failure_enabled.setChecked(False)
        self._update_sediment_controls()

    def _sediment_method_selection(self):
        enabled=self.sediment_enabled.isChecked()
        law=self._active_bedload_law()
        # The slope switch is itself a saved physical choice.  An unchecked
        # switch on an active bed-load route therefore means explicit mode 0,
        # not "choose an automatic method".  Keeping ``None`` solely for the
        # checked selector's Automatic entry preserves explicit-off projects
        # across a load/save round trip.
        slope_enabled=enabled and law>0 and self.bed_slope.isChecked()
        slope_method=(self.bed_slope_method.currentData() if slope_enabled else
                      (0 if enabled and law>0 else None))
        return resolve_sediment_methods(
            law,slope_enabled=slope_enabled,slope_method=slope_method,
            exchange_mode=self._active_exchange_mode(),
            mpm_friction_mode=self.mpm_friction_mode.currentData() if enabled and law==2 else 0)

    def _active_bedload_law(self):
        return (self.bedload.currentData() if self.sediment_enabled.isChecked()
                and self.bedload_enabled.isChecked() else 0)

    def _active_exchange_mode(self):
        return (self.exchange.currentData() if self.sediment_enabled.isChecked()
                and self.suspended_enabled.isChecked() else 0)

    def _update_hydraulic_roughness_controls(self):
        mode=self.hydraulic_roughness_mode.currentData()
        label={"manning":"Default Manning n (s/m^(1/3))",
               "chezy":"Default Chezy C (m^0.5/s)",
               "white_colebrook":"Default White-Colebrook ks (m)"}[mode]
        self.hydraulic_roughness_parameter_label.setText(label)
        white=mode=="white_colebrook"
        self.white_colebrook_chzk2d.setEnabled(white)
        if not white:self.white_colebrook_chzk2d.setChecked(False)
        parameter_label={"manning":"Manning n (s/m^(1/3))",
                         "chezy":"Chezy C (m^0.5/s)",
                         "white_colebrook":"Nikuradse ks (m)"}[mode]
        self.manning_zone_label.setText("New zone "+parameter_label)
        self.manning_zone_value.setMinimum(0. if mode=="manning" else 1.e-12)
        self.manning_zone_value.setSingleStep(
            {"manning":.005,"chezy":1.,"white_colebrook":.001}[mode])
        for widget in (self.create_manning_button,self.apply_manning_button,
                       self.manning_zone_value,self.manning_zone_priority,
                       self.snap_polygons,self.manning_zone_label):
            widget.setEnabled(True)
        self.manning_layer.setToolTip(
            "Optional typed spatial roughness zones. Legacy Manning layers remain readable only with Manning selected.")

    def _update_sediment_controls(self):
        sediment_enabled=self.sediment_enabled.isChecked()
        for section in self.sediment_sections:section.setEnabled(sediment_enabled)
        bedload_active_switch=sediment_enabled and self.bedload_enabled.isChecked()
        suspended_active_switch=sediment_enabled and self.suspended_enabled.isChecked()
        bedload_law=self.bedload.currentData() if bedload_active_switch else 0
        exchange_mode=self.exchange.currentData() if suspended_active_switch else 0
        engelund_hansen=bedload_law==8
        if bedload_law not in (2,8) and self.bed_slope_method.currentData() in (3,5):
            blocked=self.bed_slope_method.blockSignals(True)
            self.bed_slope_method.setCurrentIndex(0)
            self.bed_slope_method.blockSignals(blocked)
        for value in (3,5):
            self.bed_slope_method.model().item(self.bed_slope_method.findData(value)).setEnabled(bedload_law in (2,8))
        parameter_mode=self.bed_slope_method.currentData()
        parameter_mode=(3 if engelund_hansen else 2) if parameter_mode is None else parameter_mode
        koch_flokstra=parameter_mode==3
        previous_mode=getattr(self,"_slope_parameter_mode",2)
        if parameter_mode!=previous_mode:
            values=(self.bed_slope_alfabs.value(),self.bed_slope_alfabn.value(),self.bed_friction_angle.value())
            if previous_mode==3:self._mode3_slope_values=values
            else:self._mode2_slope_values=values
            self.bed_friction_angle.setRange(0.,100.) if koch_flokstra else self.bed_friction_angle.setRange(.0001,89.9999)
            target=self._mode3_slope_values if koch_flokstra else self._mode2_slope_values
            self.bed_slope_alfabs.setValue(target[0]);self.bed_slope_alfabn.setValue(target[1]);self.bed_friction_angle.setValue(target[2])
            self._slope_parameter_mode=parameter_mode
        else:
            self.bed_friction_angle.setRange(0.,100.) if koch_flokstra else self.bed_friction_angle.setRange(.0001,89.9999)
        if engelund_hansen and self.suspended_enabled.isChecked():
            self.suspended_enabled.setChecked(False)
            suspended_active_switch=False;exchange_mode=0
        self.suspended_enabled.setEnabled(sediment_enabled and not engelund_hansen)
        self.exchange.setEnabled(suspended_active_switch and not engelund_hansen)
        self.bedload.setEnabled(bedload_active_switch)
        self.engelund_hansen_chezy.setEnabled(bedload_active_switch and engelund_hansen)
        if engelund_hansen:
            self.transport_calibration.setValue(1.)
        if koch_flokstra:
            self.bed_slope_alfabs_label.setText("Koch-Flokstra AlfaBs")
            self.bed_slope_alfabn_label.setText("Koch-Flokstra AShld")
            self.bed_friction_angle_label.setText("Koch-Flokstra BShld")
            self.bed_slope_alfabs.setToolTip("Koch-Flokstra longitudinal coefficient AlfaBs.")
            self.bed_slope_alfabn.setToolTip("Koch-Flokstra transverse coefficient AShld.")
            self.bed_friction_angle.setToolTip("Koch-Flokstra transverse coefficient BShld; this is not an angle in mode 3.")
        else:
            self.bed_slope_alfabs_label.setText("Longitudinal slope factor (AlfaBs)")
            self.bed_slope_alfabn_label.setText("Transverse slope factor (AlfaBn)")
            self.bed_friction_angle_label.setText("Bed friction angle (°)")
            self.bed_slope_alfabs.setToolTip("Bagnold/Ikeda longitudinal slope coefficient AlfaBs.")
            self.bed_slope_alfabn.setToolTip("Bagnold/Ikeda transverse slope coefficient AlfaBn.")
            self.bed_friction_angle.setToolTip("Bagnold/Ikeda bed friction angle in degrees.")
        van_rijn=bedload_law==1;mpm=bedload_law==2
        suspended_vr=exchange_mode in (2,3);mixed=exchange_mode==3;roughness_active=van_rijn or suspended_vr
        cohesive_fraction=any(
            self.fraction_table.cellWidget(row,3) is not None and
            self.fraction_table.cellWidget(row,3).currentData()=="mud"
            for row in range(self.fraction_table.rowCount()))
        self.reference_height_factor.setEnabled(sediment_enabled and suspended_vr);self.sediment_roughness.setEnabled(sediment_enabled and roughness_active)
        settling_active=suspended_active_switch and exchange_mode!=0
        self.settling_mode.setEnabled(settling_active)
        self.hindered_settling_reference.setEnabled(settling_active and self.settling_mode.currentData()==1)
        self.critical_mud_fraction.setEnabled(sediment_enabled and mixed)
        fixed=self.sediment_roughness.currentData()=="fixed"
        self.z0_mm.setEnabled(sediment_enabled and roughness_active and fixed);self.z0_label.setEnabled(sediment_enabled and roughness_active and fixed)
        if not roughness_active:
            text="Roughness applies to Van Rijn suspended exchange and bed load. Settings are retained while those formulations are off."
        elif fixed:
            text="Sediment transport uses the specified fixed z0. Water-flow resistance uses the selected C/M/W law from Mesh."
        else:
            text="Uses the equivalent z0 from the selected C/M/W hydraulic roughness law; fixed z0 is inactive."
        self.sediment_roughness_help.setText(text)
        bedload_active=bedload_law>0
        skin_compatible=(sediment_enabled and
                         (bedload_law in (5,6,7) or (mixed and cohesive_fraction)))
        if not skin_compatible and self.sediment_shear_mode.currentData()!="total":
            self.sediment_shear_mode.setCurrentIndex(self.sediment_shear_mode.findData("total"))
        self.sediment_shear_mode.setEnabled(skin_compatible)
        skin_active=skin_compatible and self.sediment_shear_mode.currentData()=="soulsby_clarke_skin"
        for widget in self.skin_shear_widgets:widget.setEnabled(skin_active)
        source_mpm=self.mpm_friction_mode.currentData()==1
        secondary_choice=self.secondary_model.currentData()
        classic_prognostic=secondary_choice==_SECONDARY_PROGNOSTIC_CLASSIC
        legacy_prognostic=secondary_choice==_SECONDARY_PROGNOSTIC_LEGACY
        prognostic=classic_prognostic or legacy_prognostic
        if classic_prognostic and not self.secondary_transport_direction.isChecked():
            self.secondary_transport_direction.setChecked(True)
        secondary_compatible=(
            classic_prognostic or
            (not prognostic and sediment_enabled and
             bedload_law in (2,3,5,6,7,8)) or
            (legacy_prognostic and (
                not self.secondary_transport_direction.isChecked() or
                (sediment_enabled and bedload_law in (1,2,3,5,6,7,8)))))
        if not secondary_compatible and self.secondary_flow.isChecked():self.secondary_flow.setChecked(False)
        self.secondary_flow.setEnabled(secondary_compatible)
        self.secondary_transport_direction.setEnabled(
            self.secondary_flow.isChecked() and legacy_prognostic)
        self.secondary_beta.setEnabled(
            self.secondary_flow.isChecked() and legacy_prognostic)
        self.secondary_flow_espir.setEnabled(secondary_compatible and self.secondary_flow.isChecked())
        adaptation_compatible=sediment_enabled and bedload_active
        if not adaptation_compatible and self.bedload_adaptation.isChecked():self.bedload_adaptation.setChecked(False)
        self.bedload_adaptation.setEnabled(adaptation_compatible)
        self.bedload_adaptation_length.setEnabled(adaptation_compatible and self.bedload_adaptation.isChecked())
        acceleration_compatible=(sediment_enabled and bedload_active and exchange_mode==0 and
                                 not self.bank_morphology_enabled.isChecked() and
                                 not self.bank_failure_enabled.isChecked())
        if not acceleration_compatible and self.morphological_acceleration.isChecked():
            self.morphological_acceleration.setChecked(False)
        self.morphological_acceleration.setEnabled(acceleration_compatible)
        acceleration_active=acceleration_compatible and self.morphological_acceleration.isChecked()
        self.morphological_factor.setEnabled(acceleration_active)
        self.morphological_factor_schedule.setEnabled(acceleration_active)
        # Project physics are backend-independent.  In particular, loading a
        # valid recursive bank/bedform project into the default CPU view must
        # not destructively clear its bedform switch before the user selects
        # the explicitly required GPU route.
        gpu_bank_composition=(hasattr(self,'gpu_variant') and
            self.gpu_variant.currentData() in (INTEGRATED,_ON_DEMAND_AMR_VARIANT) and
            self.amr_enabled.isChecked() and self.amr_subcycling.isChecked() and
            hasattr(self,'gpu_amr_evolving_banks') and
            self.gpu_amr_evolving_banks.isChecked() and
            self.bank_failure_enabled.isChecked() and
            self.bank_toe_mode.currentData() in (
                'noncohesive_transport_capacity','distributed_slope_transport') and
            self.general_avalanche.isChecked() and
            not self.bank_morphology_enabled.isChecked() and
            not self.bed_slope.isChecked() and self.hiding_exposure.currentData()==0 and
            self.active_layer_mode.currentData()=='constant' and
            self.substrate_layer_count.value()==1)
        bedform_compatible=(sediment_enabled and mpm and source_mpm and
                            self.hydraulic_roughness_mode.currentData()=="manning" and
                            exchange_mode==0 and
                            not self.secondary_flow.isChecked() and not self.bedload_adaptation.isChecked() and
                            not self.morphological_acceleration.isChecked() and
                            not self.bank_morphology_enabled.isChecked() and
                            ((not self.bank_failure_enabled.isChecked() and
                              not self.general_avalanche.isChecked()) or gpu_bank_composition))
        if not bedform_compatible and self.bedform_roughness.isChecked():self.bedform_roughness.setChecked(False)
        self.bedform_roughness.setEnabled(bedform_compatible)
        if source_mpm:self.mpm_critical.setValue(.047)
        self.mpm_friction_mode.setEnabled(sediment_enabled and mpm);self.mpm_cf.setEnabled(sediment_enabled and mpm and not source_mpm);self.mpm_critical.setEnabled(sediment_enabled and mpm and not source_mpm);self.bed_slope.setEnabled(sediment_enabled and bedload_active)
        self.transport_calibration.setEnabled(sediment_enabled and bedload_law in (3,4,5,6,7))
        self.hiding_exposure.setEnabled(sediment_enabled and mpm)
        if not mpm:self.hiding_exposure.setCurrentIndex(self.hiding_exposure.findData(0))
        hiding_mode=self.hiding_exposure.currentData()
        self.hiding_parker_alpha.setEnabled(sediment_enabled and mpm and hiding_mode==4)
        self.hiding_wu_exponent.setEnabled(sediment_enabled and mpm and hiding_mode==5)
        if not bedload_active:self.bed_slope.setChecked(False)
        enabled=sediment_enabled and bedload_active and self.bed_slope.isChecked()
        self.bed_slope_method.setEnabled(enabled)
        for widget in (self.bed_slope_alfabs,self.bed_slope_alfabn,self.bed_friction_angle):widget.setEnabled(enabled)
        selection=self._sediment_method_selection()
        slope_name={0:"off",2:"Legacy BRAIDS Bagnold/Ikeda",
                    3:"Koch-Flokstra (Delft3D ISlope=3)",5:"Bagnold/Ikeda (Delft3D FLOW ISlope=2)"}[selection.bed_slope]
        self.bed_slope_method_help.setText(f"{selection.selection_origin.capitalize()} slope method: {slope_name}")
        self.bed_slope_method_help.setToolTip("\n".join(selection.messages+selection.restrictions))
        layered=sediment_enabled and self.active_layer_thickness.value()>0
        if not layered and self.bank_failure_enabled.isChecked():self.bank_failure_enabled.setChecked(False)
        bank_failure_enabled=sediment_enabled and self.bank_failure_enabled.isChecked()
        bank_legacy_enabled=sediment_enabled and self.bank_morphology_enabled.isChecked()
        bank_enabled=bank_failure_enabled or bank_legacy_enabled
        self.bank_failure_enabled.setEnabled(sediment_enabled and layered)
        self.bank_morphology_enabled.setEnabled(sediment_enabled)
        if not layered and self.general_avalanche.isChecked():self.general_avalanche.setChecked(False)
        self.general_avalanche.setEnabled(layered)
        general_enabled=layered and self.general_avalanche.isChecked()
        for widget in self.general_avalanche_controls:widget.setEnabled(general_enabled)
        self.active_layer_mode.setEnabled(layered)
        dynamic_layer=layered and self.active_layer_mode.currentData()=="water_depth"
        self.active_layer_depth_fraction.setEnabled(dynamic_layer);self.active_layer_minimum_thickness.setEnabled(dynamic_layer)
        self.substrate_layer_count.setEnabled(layered);self.substrate_layer_thickness.setEnabled(layered and self.substrate_layer_count.value()>1)
        for widget in (self.bank_zone_layer,self.create_bank_zone_button):widget.setEnabled(bank_enabled)
        self.bank_toe_mode.setEnabled(bank_failure_enabled)
        toe_mode=self.bank_toe_mode.currentData()
        cohesive_enabled=bank_failure_enabled and toe_mode=="cohesive_excess_shear"
        for widget in (self.bank_toe_erodibility,self.bank_toe_critical_shear,
                       self.bank_toe_exponent):widget.setEnabled(cohesive_enabled)
        self.bank_toe_capacity_fraction.setEnabled(
            bank_failure_enabled and toe_mode=="noncohesive_transport_capacity")
        distributed_enabled=bank_failure_enabled and toe_mode=="distributed_slope_transport"
        self.bank_distributed_transport_fraction.setEnabled(distributed_enabled)
        self.bank_distributed_critical_slope.setEnabled(distributed_enabled)
        self.bank_critical_undercut.setEnabled(bank_failure_enabled)
        for widget in (self.bank_erosion_fraction,self.bank_avalanche):widget.setEnabled(bank_legacy_enabled)
        avalanche_enabled=bank_legacy_enabled and self.bank_avalanche.isChecked()
        for widget in (self.bank_wet_slope,self.bank_dry_slope,self.bank_avalanche_time,self.bank_relaxation_length,
                       self.bank_wet_depth_switch):widget.setEnabled(avalanche_enabled)

    def _sediment_material(self):
        return CutSediment(grain_density_kg_m3=self.grain_density.value(),water_density_kg_m3=self.water_density.value(),
            porosity=self.porosity.value(),d50_m=self.d50_mm.value()*1e-3,d90_m=self.d90_mm.value()*1e-3,
            z0_m=self.z0_mm.value()*1e-3,minimum_exchange_depth_m=self.exchange_depth.value(),
            minimum_bedload_depth_m=self.minimum_bedload_depth.value(),
            reference_height_factor=self.reference_height_factor.value(),roughness_mode=self.sediment_roughness.currentData())

    def _fraction_ids(self):
        if self.fraction_table.rowCount()<2:return ()
        return tuple((self.fraction_table.item(row,0).text().strip() if self.fraction_table.item(row,0) else "")
                     for row in range(self.fraction_table.rowCount()))

    def _add_fraction(self,*_,values=None):
        row=self.fraction_table.rowCount();defaults=(f"fraction_{row+1}",.5,.5,"sand",self.grain_density.value(),
            self.porosity.value(),self.d50_mm.value(),self.d90_mm.value(),self.settling.value(),.2,.1,1e-4,1.)
        source=defaults if values is None else values
        if len(source)==7:source=(source[0],source[1],source[1],"sand",*source[2:],.2,.1,1e-4,1.)
        if len(source)==8:source=(source[0],source[1],source[2],"sand",*source[3:],.2,.1,1e-4,1.)
        self.fraction_table.insertRow(row)
        for column,value in enumerate(source):
            if column==3:
                kind=QComboBox();kind.addItem("Sand / noncohesive","sand");kind.addItem("Mud / cohesive","mud")
                kind.setCurrentIndex(max(0,kind.findData(str(value).lower())));self.fraction_table.setCellWidget(row,column,kind)
                kind.currentIndexChanged.connect(
                    lambda *_args,source=kind:self._prepared_controls_changed(source))
            else:self.fraction_table.setItem(row,column,QTableWidgetItem(str(value)))
        if self.fraction_table.rowCount()>=2:self.refresh_sediment_boundary_json()

    def _remove_fractions(self,*_):
        rows=sorted({item.row() for item in self.fraction_table.selectedItems()},reverse=True)
        for row in rows:self.fraction_table.removeRow(row)
        self.refresh_sediment_boundary_json()

    def _set_fraction_rows(self,rows):
        self.fraction_table.setRowCount(0)
        for value in rows or ():
            active=value.get("initial_active_volume_fraction")
            if active is None:active=value["initial_volume_fraction"]
            self._add_fraction(values=(value["fraction_id"],value["initial_volume_fraction"],
                active,value.get("sediment_type","sand"),
                value["grain_density_kg_m3"],value["porosity"],value["d50_m"]*1000,
                value["d90_m"]*1000,value["settling_velocity_m_s"],
                value.get("critical_erosion_shear_pa",.2),value.get("critical_deposition_shear_pa",.1),
                value.get("erosion_parameter_kg_m2_s",1e-4),value.get("erosion_exponent",1.)))

    def _sediment_fractions(self):
        count=self.fraction_table.rowCount()
        if count==0:return None
        if count<2:raise ValueError("Add at least two grain-fraction rows, or remove the single row")
        common=self._sediment_material();result=[]
        for row in range(count):
            values=[]
            for column in range(13):
                if column==3:
                    widget=self.fraction_table.cellWidget(row,column)
                    if widget is None:raise ValueError(f"Grain fraction row {row+1} has no material type")
                    values.append(widget.currentData());continue
                item=self.fraction_table.item(row,column)
                if item is None or not item.text().strip():raise ValueError(f"Grain fraction row {row+1} is incomplete")
                values.append(item.text().strip())
            try:
                result.append(CutSedimentFraction(fraction_id=values[0],initial_volume_fraction=float(values[1]),
                    initial_active_volume_fraction=float(values[2]),
                    sediment_type=values[3].lower(),grain_density_kg_m3=float(values[4]),porosity=float(values[5]),d50_m=float(values[6])*1e-3,
                    d90_m=float(values[7])*1e-3,settling_velocity_m_s=float(values[8]),
                    critical_erosion_shear_pa=float(values[9]),critical_deposition_shear_pa=float(values[10]),
                    erosion_parameter_kg_m2_s=float(values[11]),erosion_exponent=float(values[12]),
                    water_density_kg_m3=common.water_density_kg_m3,z0_m=common.z0_m,
                    minimum_exchange_depth_m=common.minimum_exchange_depth_m,
                    minimum_bedload_depth_m=common.minimum_bedload_depth_m,
                    reference_height_factor=common.reference_height_factor,roughness_mode=common.roughness_mode))
            except ValueError as exc:raise ValueError(f"Invalid grain fraction row {row+1}: {exc}") from exc
        if not math.isclose(sum(value.initial_volume_fraction for value in result),1.,rel_tol=0,abs_tol=1e-12):
            raise ValueError("Initial substrate/default grain volume fractions must sum to 1")
        if not math.isclose(sum(value.initial_active_volume_fraction for value in result),1.,rel_tol=0,abs_tol=1e-12):
            raise ValueError("Initial active-layer grain volume fractions must sum to 1")
        return tuple(result)

    def _clear_layout(self,layout):
        while layout.count():
            item=layout.takeAt(0);widget=item.widget()
            if widget is not None:
                widget.hide();widget.setParent(None);widget.deleteLater()

    def _rebuild_boundary_cards(self,rows,kinds):
        self._updating_boundary_cards=True;self._clear_layout(self.boundary_cards_layout);self.boundary_cards={}
        for index,name in enumerate(sorted(kinds)):
            card=HydraulicBoundaryCard(name,kinds[name],rows.get(name,{}),self._boundary_cards_changed,
                self._select_boundary_faces,self._preview_boundary,self._plot_hydraulic_card,
                self._delete_boundary,expanded=index==0)
            self.boundary_cards[name]=card;self.boundary_cards_layout.addWidget(card)
        if not self.boundary_cards:
            label=QLabel("No inlet or outlet IDs. Select polygon faces, assign their IDs, then refresh.");label.setWordWrap(True);self.boundary_cards_layout.addWidget(label)
        self._updating_boundary_cards=False;self._boundary_cards_changed()

    def _rebuild_sediment_cards(self,values):
        self._updating_boundary_cards=True;self._clear_layout(self.sediment_cards_layout);self.sediment_cards={}
        for index,name in enumerate(sorted(values)):
            card=SedimentBoundaryCard(name,values[name],self._sediment_cards_changed,self._plot_sediment_card,
                fraction_ids=self._fraction_ids(),expanded=index==0)
            self.sediment_cards[name]=card;self.sediment_cards_layout.addWidget(card)
        if not self.sediment_cards:
            label=QLabel("No inlet IDs are currently defined in General.");self.sediment_cards_layout.addWidget(label)
        self._updating_boundary_cards=False;self._sediment_cards_changed()

    def _hydraulic_values(self):
        if self.boundary_cards:return {name:card.values() for name,card in self.boundary_cards.items()}
        return json.loads(self.boundary_json.toPlainText() or "{}")

    def _sediment_values(self):
        if self.sediment_cards:return {name:card.values() for name,card in self.sediment_cards.items()}
        return json.loads(self.sediment_boundary_json.toPlainText() or "{}")

    def _current_boundary_rows(self):
        hydraulics=self._hydraulic_values();sediment=self._sediment_values();result={}
        empty={"concentration_kg_m3":0.,"bed_input_kg_m_s":0.,"bed_input_kg_s":None,
               "bedload_boundary_mode":"prescribed","sediment_series":None,
               "fraction_concentration_kg_m3":{},"fraction_bed_input_kg_s":{},"fraction_sediment_series":{}}
        for name,row in hydraulics.items():
            inlet=row.get("mode") in ("total_discharge","discharge","inlet_stage","inlet_depth","inlet_normal","inlet_critical")
            result[name]={**row,**(sediment.get(name,empty) if inlet else empty)}
        return result

    def _morphological_factor_series(self):
        text=self.morphological_factor_schedule.text().strip()
        if not text:return None
        times=[];factors=[]
        for item in text.replace(";",",").split(","):
            item=item.strip()
            if not item:continue
            fields=item.split(":",1) if ":" in item else item.split()
            if len(fields)!=2:
                raise ValueError("Each MorFac schedule knot must be time_s:factor")
            times.append(float(fields[0]));factors.append(float(fields[1]))
        return CutMorphologicalFactorSeries(tuple(times),tuple(factors))

    def _project_boundary_rows(self):
        if self.sediment_enabled.isChecked():return self._current_boundary_rows()
        empty={"concentration_kg_m3":0.,"bed_input_kg_m_s":0.,"bed_input_kg_s":None,
               "bedload_boundary_mode":"prescribed","sediment_series":None,
               "fraction_concentration_kg_m3":{},"fraction_bed_input_kg_s":{},
               "fraction_sediment_series":{}}
        return {name:{**row,**empty} for name,row in self._hydraulic_values().items()}

    def _boundary_cards_changed(self,*_):
        if self._updating_boundary_cards:return
        self._prepared_controls_changed(self.boundary_json)
        try:
            self.boundary_json.setPlainText(json.dumps(self._current_boundary_rows(),indent=2))
        except (TypeError,ValueError,json.JSONDecodeError):
            # A partly typed time-series row is validated by Plot or Save.
            pass

    def _sediment_cards_changed(self,*_):
        if self._updating_boundary_cards:return
        try:
            values=self._sediment_values();self.sediment_boundary_json.setPlainText(json.dumps(values,indent=2));self._boundary_cards_changed()
        except (TypeError,ValueError,json.JSONDecodeError):pass

    def refresh_sediment_boundary_json(self,boundary_rows=None):
        try:
            explicit=boundary_rows is not None and not isinstance(boundary_rows,bool)
            if not explicit:
                boundary_rows=self._hydraulic_values()
            try:current=self._sediment_values()
            except (TypeError,ValueError,json.JSONDecodeError):current={}
            fraction_ids=self._fraction_ids()
            defaults={"concentration_kg_m3":0.,"bed_input_kg_m_s":0.,"bed_input_kg_s":None,
                      "bedload_boundary_mode":"prescribed","sediment_series":None,
                      "fraction_concentration_kg_m3":{name:0. for name in fraction_ids},
                      "fraction_bed_input_kg_s":{name:0. for name in fraction_ids},"fraction_sediment_series":{}}
            fields=tuple(defaults)
            result={}
            for name,row in boundary_rows.items():
                if row.get("mode") not in ("total_discharge","discharge","inlet_stage","inlet_depth","inlet_normal","inlet_critical"):continue
                source=row if explicit and any(key in row for key in fields) else current.get(name,row)
                result[name]={key:source.get(key,defaults[key]) for key in fields}
                result[name]["concentration_kg_m3"]=result[name]["concentration_kg_m3"] or 0.
                result[name]["bed_input_kg_m_s"]=result[name]["bed_input_kg_m_s"] or 0.
                if fraction_ids:
                    result[name]["concentration_kg_m3"]=0.;result[name]["bed_input_kg_m_s"]=0.
                    result[name]["bed_input_kg_s"]=None;result[name]["sediment_series"]=None
                    for key in ("fraction_concentration_kg_m3","fraction_bed_input_kg_s"):
                        values=result[name].get(key) or {}
                        result[name][key]={fraction:float(values.get(fraction,0.)) for fraction in fraction_ids}
                else:
                    result[name]["fraction_concentration_kg_m3"]={};result[name]["fraction_bed_input_kg_s"]={};result[name]["fraction_sediment_series"]={}
            self.sediment_boundary_json.setPlainText(json.dumps(result,indent=2));self._rebuild_sediment_cards(result);return result
        except (TypeError,ValueError,json.JSONDecodeError) as exc:
            self._message("Read sediment boundary IDs",f"Boundary configuration is invalid: {exc}",True);return None

    def _refresh_input_summary(self,payload=None):
        try:
            if isinstance(payload,bool):payload=None
            if payload is None:
                selection=self._sediment_method_selection()
                material=self._sediment_material();fractions=self._sediment_fractions() if self.sediment_enabled.isChecked() else None;zone_layer=self.manning_layer.currentLayer()
                bedrock_layer=self.bedrock_layer.currentLayer();thickness_layer=self.erodible_raster_layer.currentLayer();erodible_layer=self.erodible_zone_layer.currentLayer()
                water_layer=self.water_body_layer.currentLayer();bank_layer=self.bank_zone_layer.currentLayer();obstacle_layer=self.obstacle_layer.currentLayer()
                inlet_supply=(json.loads(self.sediment_boundary_json.toPlainText() or "{}")
                              if self.sediment_enabled.isChecked() else {})
                secondary_selection=_secondary_project_selection(
                    self.secondary_model.currentData())
                summary={"mesh":{"minimum_cell_size_m":self.spacing.value(),
                                 "base_cell_size_m":self._base_mesh_size(),
                                 "levels":self.amr_levels.value() if self.amr_enabled.isChecked() else 1,
                                 "refinement_ratio":AMR_RATIO if self.amr_enabled.isChecked() else 1,
                                 "initial_water_level_m":self._initial_stage_value(),
                                 "initial_water_body":None if water_layer is None else water_layer.name()},
                    "roughness":{"mode":self.hydraulic_roughness_mode.currentData(),
                        "default_parameter":self.manning.value(),
                        "parameter_name":{"manning":"manning_n","chezy":"chezy_C",
                            "white_colebrook":"nikuradse_ks"}[self.hydraulic_roughness_mode.currentData()],
                        "parameter_units":{"manning":"s/m^(1/3)","chezy":"m^0.5/s",
                            "white_colebrook":"m"}[self.hydraulic_roughness_mode.currentData()],
                        "white_colebrook_chzk2d":self.white_colebrook_chzk2d.isChecked(),
                        "zones":None if zone_layer is None else zone_layer.name()},
                    "sediment":{"enabled":self.sediment_enabled.isChecked(),"material":{**material.__dict__,"bulk_kg_m3":material.bulk_kg_m3},
                        "fractions":None if fractions is None else [{**value.__dict__,"bulk_kg_m3":value.bulk_kg_m3} for value in fractions],
                        "suspended_exchange_mode":self._active_exchange_mode(),"settling_velocity_m_s":self.settling.value(),
                        "settling_mode":self.settling_mode.currentData(),"hindered_settling_reference_concentration_kg_m3":self.hindered_settling_reference.value(),
                        "bedload_law":self._active_bedload_law(),"engelund_hansen_chezy":self.engelund_hansen_chezy.value(),"mpm_cf":self.mpm_cf.value(),"mpm_friction_mode":self.mpm_friction_mode.currentData(),
                        "sediment_shear":{"mode":self.sediment_shear_mode.currentData(),
                            "kinematic_viscosity_m2_s":self.skin_kinematic_viscosity.value(),
                            "sand_roughness_m":self.skin_sand_roughness.value(),
                            "silt_roughness_m":self.skin_silt_roughness.value(),
                            "critical_mud_fraction_low":self.skin_mud_low.value(),
                            "critical_mud_fraction_high":self.skin_mud_high.value(),
                            "fluff_fraction":self.skin_fluff_fraction.value()},
                        "critical_mud_fraction":self.critical_mud_fraction.value(),"transport_calibration_factor":self.transport_calibration.value(),
                        "mpm_critical_shields":self.mpm_critical.value(),"hiding_exposure":self.hiding_exposure.currentData(),"hiding_parker_alpha":self.hiding_parker_alpha.value(),"hiding_wu_exponent":self.hiding_wu_exponent.value(),"slope_correction_enabled":self.bed_slope.isChecked(),"slope_correction_mode":selection.bed_slope,
                        "method_selection":selection.as_dict(),
                        "longitudinal_slope_factor":self.bed_slope_alfabs.value(),"transverse_slope_factor":self.bed_slope_alfabn.value(),
                        "bed_friction_angle_deg":self.bed_friction_angle.value(),"initial_erodible_thickness_m":self.erodible.value(),
                        "active_layer_thickness_m":self.active_layer_thickness.value(),
                        "active_layer_mode":self.active_layer_mode.currentData(),
                        "active_layer_depth_fraction":self.active_layer_depth_fraction.value(),
                        "active_layer_minimum_thickness_m":self.active_layer_minimum_thickness.value(),
                        "substrate_layer_count":self.substrate_layer_count.value(),
                        "substrate_layer_thickness_m":self.substrate_layer_thickness.value(),
                        "minimum_initial_erodible_water_depth_m":self.erodible_depth.value(),
                        "spatial_bed_inputs":{"bedrock_raster":None if bedrock_layer is None else bedrock_layer.name(),
                            "thickness_raster":None if thickness_layer is None else thickness_layer.name(),
                            "thickness_zones":None if erodible_layer is None else erodible_layer.name()},"inlet_supply":inlet_supply},
                    "internal_conditions":{"solid_obstacles":{"layer":None if obstacle_layer is None else obstacle_layer.name(),
                        "wall_behavior":"fixed impermeable reflective wall"},
                        "general_avalanche":{"enabled":self.general_avalanche.isChecked(),
                        "qualified":True,
                        "wet_maximum_slope":self.general_avalanche_wet_slope.value(),
                        "dry_maximum_slope":self.general_avalanche_dry_slope.value(),
                        "avalanche_time_s":self.general_avalanche_time.value(),
                        "relaxation_length_m":self.general_avalanche_relaxation_length.value(),
                        "wet_depth_switch_m":self.general_avalanche_wet_depth_switch.value()},
                         "secondary_flow":{"enabled":self.secondary_flow.isChecked(),
                        "qualified":secondary_selection["model"]=="equilibrium",
                        "model":secondary_selection["model"],
                        "method_id":secondary_selection["method_id"],
                        "method_provenance":secondary_selection[
                            "method_provenance"],
                        "transport_direction":(
                            True if self.secondary_model.currentData() in (
                                _SECONDARY_EQUILIBRIUM,
                                _SECONDARY_PROGNOSTIC_CLASSIC) else
                            self.secondary_transport_direction.isChecked()),
                        "momentum_beta":(
                            self.secondary_beta.value()
                            if self.secondary_model.currentData()==
                                _SECONDARY_PROGNOSTIC_LEGACY else 0.),
                        "classic_betac":0.,
                        "scalar_diffusivity_m2_s":0.,
                        "espir":self.secondary_flow_espir.value(),
                        "transport_effect":(
                            "bed-load direction only"
                            if secondary_selection["model"]=="equilibrium" else
                            "classic transported signed hI"
                            if self.secondary_model.currentData()==
                                _SECONDARY_PROGNOSTIC_CLASSIC else
                            "legacy FM-style prognostic spiral flow")},
                        "bedload_adaptation":{"enabled":self.bedload_adaptation.isChecked(),
                        "qualified":True,"length_diameters":self.bedload_adaptation_length.value(),
                        "model":"steady first-order streamline relaxation"},
                        "morphological_acceleration":{"enabled":self.morphological_acceleration.isChecked(),
                        "qualified":True,"constant_factor":self.morphological_factor.value(),
                        "series":(None if self._morphological_factor_series() is None else
                                  self._morphological_factor_series().__dict__),
                        "model":"Delft3D-compatible accelerated conservative Exner clock"},
                        "bedform_roughness":{"enabled":self.bedform_roughness.isChecked(),
                        "qualified":True,"model":"Fredsoe-MPM height and Van Rijn 1984 roughness"},
                        "bank_failure":{"enabled":self.bank_failure_enabled.isChecked(),
                        "qualified":self.bank_toe_mode.currentData()=="cohesive_excess_shear",
                        "zones":None if bank_layer is None else bank_layer.name(),
                        "toe_erosion_mode":self.bank_toe_mode.currentData(),
                        "toe_capacity_fraction":self.bank_toe_capacity_fraction.value(),
                        "distributed_transport_fraction":self.bank_distributed_transport_fraction.value(),
                        "distributed_critical_slope":self.bank_distributed_critical_slope.value(),
                        "front_tracking_enabled":False,
                        "toe_erodibility_m_pa_s":self.bank_toe_erodibility.value(),
                        "toe_critical_shear_pa":self.bank_toe_critical_shear.value(),
                        "toe_exponent":self.bank_toe_exponent.value(),
                        "critical_undercut_m":self.bank_critical_undercut.value()},
                        "bank_morphology":{"enabled":self.bank_morphology_enabled.isChecked(),
                        "zones":None if bank_layer is None else bank_layer.name(),
                        "erosion_supply_fraction":self.bank_erosion_fraction.value(),
                        "avalanche_enabled":self.bank_avalanche.isChecked(),"wet_maximum_slope":self.bank_wet_slope.value(),
                        "dry_maximum_slope":self.bank_dry_slope.value(),"avalanche_time_s":self.bank_avalanche_time.value(),
                        "relaxation_length_m":self.bank_relaxation_length.value(),
                        "wet_depth_switch_m":self.bank_wet_depth_switch.value()}}}
            else:
                numerics=payload["numerics"];material=CutSediment(**payload["sediment"])
                summary={"mesh":{"grid":payload["grid"],
                                 "initial_water_level_m":payload["initial"]["stage_m"]},
                    "roughness":payload.get("roughness",{"default_manning_n":numerics["manning"],"zones_path":None}),
                    "sediment":{"enabled":numerics.get("sediment_enabled",True),"material":{**material.__dict__,"bulk_kg_m3":material.bulk_kg_m3},
                        "fractions":payload.get("sediment_fractions"),
                        "suspended_exchange_mode":numerics["exchange"],"settling_velocity_m_s":numerics["settling_velocity_m_s"],
                        "settling_mode":numerics.get("settling_mode",0),"hindered_settling_reference_concentration_kg_m3":numerics.get("hindered_settling_reference_concentration_kg_m3",DEFAULT_HINDERED_REFERENCE_CONCENTRATION_KG_M3),
                        "bedload_law":numerics["bedload"],"engelund_hansen_chezy":numerics.get("engelund_hansen_chezy",28.8),"mpm_cf":numerics.get("mpm_cf",.01),"mpm_friction_mode":numerics.get("mpm_friction_mode",0),
                        "sediment_shear":payload.get("sediment_shear",{"mode":"total"}),
                        "critical_mud_fraction":numerics.get("critical_mud_fraction",.5),"transport_calibration_factor":numerics.get("transport_calibration_factor",1.),
                        "mpm_critical_shields":numerics.get("mpm_critical_shields",0.),"hiding_exposure":numerics.get("hiding_exposure",0),"hiding_parker_alpha":numerics.get("hiding_parker_alpha",.6),"hiding_wu_exponent":numerics.get("hiding_wu_exponent",-.6),"slope_correction_enabled":numerics["bed_slope"]!=0,"slope_correction_mode":numerics["bed_slope"],
                        "method_selection":numerics.get("sediment_method_selection"),
                        "longitudinal_slope_factor":numerics.get("bed_slope_alfabs",1.),"transverse_slope_factor":numerics.get("bed_slope_alfabn",1.5),
                        "bed_friction_angle_deg":numerics.get("bed_friction_angle_deg",30.),"initial_bed":payload["initial"],
                        "spatial_bed_inputs":payload.get("bed_inputs",{}),
                        "inlet_supply":{name:{key:row.get(key) for key in ("concentration_kg_m3","bed_input_kg_m_s","bed_input_kg_s","sediment_series","fraction_concentration_kg_m3","fraction_bed_input_kg_s","fraction_sediment_series")}
                            for name,row in payload["boundaries"].items()
                            if row.get("mode") in ("total_discharge","discharge","inlet_stage","inlet_depth","inlet_normal","inlet_critical")}},
                    "internal_conditions":payload.get("internal_conditions",{})}
            self.input_summary.setPlainText(json.dumps(summary,indent=2,sort_keys=True))
            self._refresh_product_summary(payload)
        except Exception as exc:self.input_summary.setPlainText(f"Invalid sediment settings: {exc}")

    def _morphology_output_settings(self):
        values={}
        for definition,enabled,editor in self._morphology_output_rows:
            if enabled.checkState()!=Qt.CheckState.Checked:continue
            kind=definition["value_type"];entry_id=definition["entry_id"]
            if kind=="boolean":values[entry_id]=True;continue
            if kind=="integer_enum":values[entry_id]=int(editor.currentData());continue
            text=editor.text().strip()
            if not text:raise ValueError(definition["keyword"]+" needs a value")
            if kind=="string_selector":values[entry_id]=text;continue
            if kind=="boolean_or_percentile_list" and text.casefold() in ("true","false"):
                values[entry_id]=text.casefold()=="true";continue
            try:
                parsed=json.loads(text) if text.startswith("[") else [
                    float(value.strip()) for value in text.split(",")]
            except (ValueError,json.JSONDecodeError) as exc:
                raise ValueError(definition["keyword"]+" needs comma-separated numbers") from exc
            values[entry_id]=parsed
        return values

    def _restore_morphology_outputs(self,payload):
        document=payload.get("qgis_product_v109") or {}
        selection=document.get("output_selection") or {}
        configured={row.get("entry_id"):row.get("value")
                    for row in selection.get("settings",[]) if isinstance(row,dict)}
        self.morphology_output_defaults.setChecked(bool(
            selection.get("apply_source_defaults",False)))
        for definition,enabled,editor in self._morphology_output_rows:
            value=configured.get(definition["entry_id"])
            enabled.setCheckState(Qt.CheckState.Checked if
                definition["entry_id"] in configured else Qt.CheckState.Unchecked)
            if editor is None or value is None:continue
            if definition["value_type"]=="integer_enum":
                index=editor.findData(int(value));editor.setCurrentIndex(max(0,index))
            elif isinstance(value,list):editor.setText(", ".join(str(item) for item in value))
            else:editor.setText(str(value).lower() if isinstance(value,bool) else str(value))

    def _refresh_product_summary(self,payload):
        if not hasattr(self,"product_summary"):return
        product=payload.get("qgis_product_v109") if isinstance(payload,dict) else None
        if not isinstance(product,dict):
            self.product_summary.setPlainText(
                "Legacy/experimental project: no portable Q10 product identity is bound.")
            return
        lines=[
            f"Product: {product.get('product_id')}",
            f"Backend: {product.get('backend')} / {product.get('integrator')}",
            f"Fingerprint: {product.get('fingerprint')}",
            "Selected methods: "+", ".join(product.get("selected_method_ids",[])),
        ]
        explanations=product.get("explanations",[])
        if explanations:lines.append("Automatic resolution:\n- "+"\n- ".join(explanations))
        self.product_summary.setPlainText("\n".join(lines))

    def _set_status(self,text,level=Qgis.Info):
        self.run_status.setText(text)
        if self.iface:self.iface.messageBar().pushMessage("BRAIDS",text,level=level,duration=5)

    def _log(self,text):
        if text:self.log.moveCursor(QTextCursor.MoveOperation.End);self.log.insertPlainText(text);self.log.ensureCursorVisible()

    def _load_run_log(self):
        path=None if self.run_root is None else self.run_root/"native.log"
        if path is not None and path.is_file():
            try:self.log.setPlainText(path.read_text(encoding="utf-8",errors="replace"))
            except OSError:self.log.clear()

    def _default_workspace(self):
        project=QgsProject.instance();bases=[]
        if project.fileName():bases.append(Path(project.fileName()).expanduser().resolve().parent)
        if project.homePath():bases.append(Path(project.homePath()).expanduser().resolve())
        location=getattr(QStandardPaths,"DocumentsLocation",None)
        if location is None:location=QStandardPaths.StandardLocation.DocumentsLocation
        documents=QStandardPaths.writableLocation(location)
        if documents:bases.append(Path(documents).expanduser().resolve())
        bases.append(Path.home().resolve())
        root=Path(Path.home().anchor)
        base=next((path for path in bases if path!=root and path.is_dir() and os.access(path,os.W_OK)),Path.home())
        return base/"BRAIDS"

    def _workspace_path(self):
        text=self.workspace.text().strip()
        path=(Path(text).expanduser().resolve() if text else self._default_workspace().resolve())
        if path==Path(path.anchor):
            raise ValueError("Choose a writable BRAIDS project folder; the filesystem root cannot be used")
        try:path.mkdir(parents=True,exist_ok=True)
        except OSError as exc:raise ValueError(f"Cannot create BRAIDS project folder {path}: {exc}") from exc
        if not path.is_dir() or not os.access(path,os.W_OK):
            raise ValueError(f"BRAIDS project folder is not writable: {path}")
        self.workspace.setText(str(path));return path

    def _initial_stage_value(self,required=False):
        value=self.initial_stage.value()
        if value==self.initial_stage.minimum():
            if required:raise ValueError("Enter the initial water level for the selected initial-water polygon")
            return None
        return value

    def _browse_workspace(self):
        path=QFileDialog.getExistingDirectory(self,"Select BRAIDS project folder",self.workspace.text())
        if path:self.workspace.setText(path)

    def _browse_project(self):
        folder=QFileDialog.getExistingDirectory(self,"Open BRAIDS project folder",self.workspace.text())
        if not folder:return
        path=Path(folder)/"braids_project.json"
        if not path.is_file():path=Path(folder)/"braids_project.json"
        if not path.is_file():
            self._message("Open BRAIDS project",f"No braids_project.json or legacy braids_project.json was found in {folder}",True);return
        self.project_path.setText(str(path));self.load_project_ui()

    def _native_executable(self):
        executable=default_native_executable()
        if executable is None:
            raise ValueError("The native BRAIDS solver bundled with this plugin is missing")
        return executable.resolve()

    def _browse_checkpoint(self):
        gpu=self.execution_backend.currentData()=="gpu"
        if gpu and (self.gpu_variant.currentData() in (_ON_DEMAND_AMR_VARIANT,
                    _ON_DEMAND_GENERAL_AMR_VARIANT) or
                    (self.gpu_variant.currentData()==INTEGRATED and self.amr_enabled.isChecked())):
            selected=QFileDialog.getExistingDirectory(self,
                "Select GPU run folder with an accepted checkpoint",
                self.checkpoint.text() or self.workspace.text())
            if not selected:return
            try:
                if self._gpu_prepared_root is None:
                    raise ValueError("Prepare or restore the authenticated on-demand AMR input before selecting a restart")
                candidate=inspect_gpu_resume_checkpoint(selected,self._gpu_prepared_root)
                if not candidate["checkpoint"].is_dir():
                    raise ValueError("This run does not contain an on-demand AMR checkpoint directory")
                self.checkpoint.setText(str(candidate["checkpoint"]))
            except Exception as exc:
                self._message("Select GPU checkpoint",str(exc),True)
            return
        filter_text=("GPU checkpoint (checkpoint.npz);;All files (*)" if gpu else
                     "Native checkpoint (checkpoint.bin);;All files (*)")
        path,_=QFileDialog.getOpenFileName(self,"Select BRAIDS checkpoint",self.checkpoint.text(),filter_text)
        if path:self.checkpoint.setText(path)

    def _browse_gpu_worker(self):
        path=QFileDialog.getExistingDirectory(self,"Select extracted BRAIDS GPU worker",
                                              self.gpu_worker_folder.text() or self.workspace.text())
        if path:self.gpu_worker_folder.setText(path)

    def _browse_gpu_python(self):
        path,_=QFileDialog.getOpenFileName(self,"Select external GPU Python executable",
                                           self.gpu_python.text(),"Python executable (python.exe);;All files (*)")
        if path:self.gpu_python.setText(path)

    def _populate_gpu_defaults(self):
        """Fill host-specific GPU paths without changing the selected backend."""
        settings=QSettings()
        found=discover_gpu_defaults(
            Path(__file__).resolve().parents[1],
            worker_hint=settings.value(self.GPU_WORKER_SETTINGS_KEY,""),
            python_hint=settings.value(self.GPU_PYTHON_SETTINGS_KEY,""))
        worker=found.get("worker_root");python=found.get("python_executable")
        if worker is not None and not self.gpu_worker_folder.text().strip():
            self.gpu_worker_folder.setText(str(worker))
        if python is not None and not self.gpu_python.text().strip():
            self.gpu_python.setText(str(python))
        if worker is not None and python is not None:
            self.gpu_identity_status.setText(
                "Windows GPU paths were detected for this PC. Verify and pin "
                "them before the first GPU preparation.")
        elif worker is not None:
            self.gpu_identity_status.setText(
                "A BRAIDS GPU worker was detected, but no likely external "
                "CUDA Python was found. Browse to python.exe, then verify.")
        else:
            self.gpu_identity_status.setText(
                "No extracted BRAIDS GPU worker was found in the standard "
                "Windows locations. Extract the companion worker ZIP or browse manually.")

    def _detect_gpu_setup(self):
        self.gpu_worker_folder.clear();self.gpu_python.clear()
        self._populate_gpu_defaults()

    def _invalidate_gpu_identity(self,*_):
        self._gpu_manifest_pin=None;self._gpu_python_pin=None;self._gpu_capability=None
        self._gpu_selector_policy_supported=False
        self._gpu_variant_changed()
        self.gpu_identity_status.setText("Worker/Python changed. Verify both before GPU preparation or execution.")

    def _verify_gpu_worker(self):
        try:
            worker=inspect_gpu_source_worker(self.gpu_worker_folder.text().strip())
            python=Path(self.gpu_python.text().strip()).expanduser().resolve()
            if not python.is_file():raise ValueError("Select an existing external Python executable")
            with python.open("rb") as stream:
                python_pin=hashlib.file_digest(stream,"sha256").hexdigest()
            capability=probe_gpu_python(python,worker["root"],
                                        expected_python_sha256=python_pin)
            self._gpu_manifest_pin=worker["manifest_sha256"]
            self._gpu_python_pin=python_pin
            self._gpu_capability=capability
            settings=QSettings()
            settings.setValue(self.GPU_WORKER_SETTINGS_KEY,str(worker["root"]))
            settings.setValue(self.GPU_PYTHON_SETTINGS_KEY,str(python))
            settings.sync()
            scope=worker["scope"]
            policies=scope.get('general_amr_selector_policies_by_variant',{})
            self._gpu_selector_policy_supported=(isinstance(policies,dict) and
                policies.get(_ON_DEMAND_GENERAL_AMR_VARIANT)==[_GENERAL_AMR_SLOPE_POLICY] and
                scope.get('source_archives_by_variant',{}).get(
                    _ON_DEMAND_GENERAL_AMR_VARIANT) in {
                        'braids_gpu/frozen_sources/on_demand_transport_amr_v3',
                        'braids_gpu/frozen_sources/on_demand_transport_amr_v4',
                        'braids_gpu/frozen_sources/on_demand_transport_amr_v5',
                        'braids_gpu/frozen_sources/on_demand_transport_amr_v6',
                        'braids_gpu/frozen_sources/on_demand_transport_amr_v7',
                        'braids_gpu/frozen_sources/on_demand_transport_amr_v8'} and
                _ON_DEMAND_GENERAL_AMR_VARIANT in scope['supported_gpu_variants'] and
                _ON_DEMAND_GENERAL_AMR_VARIANT not in scope['excluded_gpu_variants'])
            previous=self.gpu_variant.currentData()
            # Rebuilding a verified worker's route menu passes through a
            # transient empty selection.  Do not emit that non-route as a
            # physical compatibility change: it can otherwise clear valid
            # saved switches (notably recursive bank/bedform composition)
            # before the prior route is restored.
            variant_signals=self.gpu_variant.blockSignals(True)
            try:
                self.gpu_variant.clear()
                external_only=scope.get("requires_external_prepared_sidecars",{})
                for variant in scope["supported_gpu_variants"]:
                    if variant not in scope["excluded_gpu_variants"] and variant not in external_only:
                        label=("Integrated (selected physics)" if variant==INTEGRATED else
                               "On-demand bank AMR" if variant==_ON_DEMAND_AMR_VARIANT else
                               "On-demand transport AMR" if variant==_ON_DEMAND_GENERAL_AMR_VARIANT else variant)
                        self.gpu_variant.addItem(label,variant)
                index=self.gpu_variant.findData(previous)
                self.gpu_variant.setCurrentIndex(max(0,index))
            finally:
                self.gpu_variant.blockSignals(variant_signals)
            if not self.gpu_variant.count():
                raise ValueError("This source worker has no selectable GPU variants")
            self._gpu_variant_changed()
            self.gpu_identity_status.setText(
                f"Pinned experimental worker ({worker['payload_files']} files) and external Python. "
                f"FP64 CUDA test passed: {capability['device_name']}, CuPy {capability['cupy']}, "
                f"CUDA runtime {capability['cuda_runtime']} / driver {capability['cuda_driver']}. "
                f"Free VRAM {capability['device_free_memory_bytes'] / 2**30:.2f} of "
                f"{capability['device_total_memory_bytes'] / 2**30:.2f} GiB at verification. "
                "Selected physics is checked when the run starts. "
                f"{len(external_only)} external-preparation variants are unavailable in this QGIS menu.")
        except Exception as exc:
            self._invalidate_gpu_identity()
            self._message("Verify GPU worker",str(exc),True)

    def _gpu_worker_config(self):
        if (self._gpu_manifest_pin is None or self._gpu_python_pin is None or
                self._gpu_capability is None):
            raise ValueError("Verify and pin a CUDA-capable external GPU worker and Python first")
        worker=inspect_gpu_source_worker(self.gpu_worker_folder.text().strip(),
            expected_manifest_sha256=self._gpu_manifest_pin)
        python=Path(self.gpu_python.text().strip()).expanduser().resolve()
        if not python.is_file():raise ValueError("The selected external GPU Python is missing")
        with python.open("rb") as stream:
            if hashlib.file_digest(stream,"sha256").hexdigest()!=self._gpu_python_pin:
                raise ValueError("External GPU Python changed since it was pinned")
        return dict(worker_root=worker["root"],expected_manifest_sha256=self._gpu_manifest_pin,
                    python_executable=python,expected_python_sha256=self._gpu_python_pin)

    def _gpu_variant_changed(self,*_):
        variant=self.gpu_variant.currentData()
        if variant==INTEGRATED:
            implicit=self.gpu_integrator.findData('implicit_fwave_midpoint')
            if implicit>=0 and self.gpu_integrator.currentIndex()!=implicit:
                blocked=self.gpu_integrator.blockSignals(True)
                self.gpu_integrator.setCurrentIndex(implicit)
                self.gpu_integrator.blockSignals(blocked)
            self.gpu_integrator.setEnabled(False)
        else:self.gpu_integrator.setEnabled(True)
        fixed_mpm43=(variant in (INTEGRATED,_GENERAL_LIVE_TRANSPORT_VARIANT) and
            not self.amr_enabled.isChecked() and not self.bank_failure_enabled.isChecked() and
            self._active_bedload_law()==2 and self._active_exchange_mode()==0 and
            not self.bedform_roughness.isChecked() and
            not self.bedload_adaptation.isChecked() and not self.general_avalanche.isChecked())
        if variant in (
                INTEGRATED,_ON_DEMAND_AMR_VARIANT,_ON_DEMAND_GENERAL_AMR_VARIANT,
                _GENERAL_LIVE_TRANSPORT_VARIANT,_GENERAL_LIVE_BANK_VARIANT):
            self.gpu_launch_mode.setCurrentIndex(self.gpu_launch_mode.findData('plain'))
        if variant in (
                INTEGRATED,_ON_DEMAND_AMR_VARIANT,_ON_DEMAND_GENERAL_AMR_VARIANT,
                _GENERAL_LIVE_TRANSPORT_VARIANT,_GENERAL_LIVE_BANK_VARIANT) and not fixed_mpm43 and \
                self.gpu_integrator.currentData() != 'implicit_fwave_midpoint':
            self.gpu_integrator.setCurrentIndex(self.gpu_integrator.findData('ssp33'))
        on_demand=(variant==_ON_DEMAND_AMR_VARIANT or
                   variant==INTEGRATED and self.amr_enabled.isChecked())
        self.gpu_amr_evolving_banks.setEnabled(on_demand and self.bank_failure_enabled.isChecked())
        general_amr=(self.amr_enabled.isChecked() and not self.bank_failure_enabled.isChecked() and
            variant in (INTEGRATED,_ON_DEMAND_GENERAL_AMR_VARIANT))
        selector_enabled=(general_amr and
                          self.execution_backend.currentData()=='gpu' and
                          self._gpu_selector_policy_supported)
        was_enabled=self.gpu_selector_policy.isEnabled()
        self.gpu_selector_policy.setEnabled(selector_enabled)
        if (selector_enabled and not was_enabled and self._gpu_prepared_root is None and
                not self._gpu_selector_user_choice):
            self.gpu_selector_policy.setCurrentIndex(
                self.gpu_selector_policy.findData(_GENERAL_AMR_SLOPE_POLICY))
        elif not general_amr and self._gpu_prepared_root is None:
            self.gpu_selector_policy.setCurrentIndex(0)
        any_amr=on_demand or general_amr
        self.gpu_scope_note.setText(
            "Experimental on-demand GPU transport AMR: scheme 6, Plain SSPRK(3,3), "
            "one surveyed inlet and critical outlet, 2:1 dynamic refinement, banks off. "
            "The AMR subcycling checkbox explicitly selects synchronous or recursive stepping; "
            "recursive stepping requires MPM2, exchange 2 and prognostic secondary flow, and "
            "supports optional positive velocity-Laplacian viscosity and conservative domain-wide "
            "avalanching once per accepted macro-step. Viscosity uses the parent-clock full-leaf "
            "SSPRK2 operator before optional avalanching; it preserves pairwise velocity-face "
            "divergence but variable-depth momentum is not exactly conserved. "
            "The separate recursive bedform branch retains its qualified unscheduled mode-1 "
            "discharge/stage boundary, exchange-off, secondary-off, one-substrate MPM scope; "
            "it cannot be combined with viscosity or avalanching. Bank physics remains unsupported here. "
            "The GPU backend is explicitly selected; unsupported physics fails closed."
            if general_amr else
            "Experimental on-demand GPU bank AMR: scheme 6, Plain SSPRK(3,3), "
            "mode-2 physical-toe or mode-3 distributed bank erosion. Live boundaries use "
            "prognostic secondary flow; constant-stage composed modes use secondary flow off. "
            "Recursive constant-stage mode 2 or 3 can combine equilibrium bedform roughness "
            "with one post-bank parent avalanche per accepted macro-step. "
            "Wall, bathymetry and sediment refinement need the explicit evolving-bank option. "
            "The AMR subcycling checkbox explicitly selects synchronous or recursive stepping; "
            "recursive surveyed sections currently require mode 3. Unsupported combinations "
            "fail closed; no CPU solver fallback."
            if on_demand else
            "Experimental source worker: fixed-grid scheme 6. Plain SSPRK(4,3) is available "
            "only for integrated nonbank MPM with suspended exchange, bedform roughness, "
            "bedload adaptation and general avalanche off; other common transport "
            "configurations require Plain SSPRK(3,3). Unsupported physics fails "
            "closed; no CPU fallback. A verified worker is not yet a supported QGIS GPU release.")
        if self.gpu_integrator.currentData() == 'implicit_fwave_midpoint':
            self.gpu_scope_note.setText(self.gpu_scope_note.text() +
                " The explicitly selected implicit midpoint method keeps the augmented f-wave "
                "spatial residual and treats its full acoustic CFL as a diagnostic. Its stricter "
                "fully-wet fixed-grid physics and boundary envelope is checked before launch; "
                "there is no SSPRK or CPU fallback.")
        if general_amr and not self._gpu_selector_policy_supported:
            self.gpu_scope_note.setText(self.gpu_scope_note.text() +
                " Verify a version-2 worker that advertises the slope-aware selector; older workers remain legacy-only.")
        if self.execution_backend.currentData()=="gpu":
            self.checkpoint.setPlaceholderText(
                "Optional authenticated GPU checkpoint directory (from a run folder)"
                if any_amr else "Optional checkpoint.npz (fresh GPU output folder)")

    def _backend_changed(self,*_,persist=True):
        gpu=self.execution_backend.currentData()=="gpu"
        busy=(self.runner.is_running or self.gpu_runner.is_running or
              bool(self._preparation_tasks.get(self.prepare_run_button)))
        self.gpu_settings.setVisible(gpu)
        self.profile_native.setEnabled(not gpu);self.threads.setEnabled(not gpu)
        self.live_preview.setEnabled(True)
        choices=[("Water depth","depth_m"),("Flow speed","velocity_m_s"),
                 ("Water level","stage_m")]
        if gpu:
            choices.extend((("Bed elevation","bed_elevation_m"),
                            ("Bed elevation change","bed_change_m"),
                            ("Total suspended concentration","concentration_kg_m3")))
        if [self.live_variable.itemData(index) for index in range(self.live_variable.count())] != [key for _,key in choices]:
            selected=self.live_variable.currentData()
            blocked=self.live_variable.blockSignals(True)
            try:
                self.live_variable.clear()
                for label,key in choices:self.live_variable.addItem(label,key)
                index=self.live_variable.findData(selected)
                self.live_variable.setCurrentIndex(index if index>=0 else 0)
            finally:
                self.live_variable.blockSignals(blocked)
        if gpu:self._gpu_variant_changed()
        else:self.checkpoint.setPlaceholderText("Optional checkpoint.bin")
        self._update_sediment_controls()
        if gpu:
            self.live_status.setText("GPU preview will show authenticated saved flow, bed or suspended-concentration frames as the solver runs.")
        else:
            self.live_status.setText("Live preview uses native fixed-grid output and updates after each saved time.")
        self.run_button.setEnabled(not busy and not self._prepared_controls_dirty and
                                   bool(self._gpu_prepared_root if gpu else self.run_root))
        if persist:QgsProject.instance().writeEntry("BRAIDS",self.BACKEND_KEY,"gpu" if gpu else "cpu")

    def _connect_prepared_control_invalidation(self):
        """Invalidate a prepared deck when a preparation input is edited.

        End time remains a run-time continuation control for the GPU backend.
        Live-preview, profiling and result-view controls do not alter the
        numerical problem and are intentionally excluded.
        """
        excluded={self.live_preview,self.live_variable,self.profile_native}
        for index in range(min(5,self.tabs.count())):
            page=self.tabs.widget(index)
            for widget in page.findChildren(QDoubleSpinBox):
                if widget not in excluded:
                    widget.valueChanged.connect(
                        lambda *_args,source=widget:self._prepared_controls_changed(source))
            for widget in page.findChildren(QSpinBox):
                if widget not in excluded:
                    widget.valueChanged.connect(
                        lambda *_args,source=widget:self._prepared_controls_changed(source))
            for widget in page.findChildren(QComboBox):
                if widget not in excluded:
                    # QgsMapLayerComboBox renumbers its proxy-model rows when
                    # a live-preview/result layer is added.  Its generic index
                    # signal can therefore fire while the selected numerical
                    # input layer is unchanged.  The semantic QGIS signal
                    # preserves real user layer changes without invalidating a
                    # prepared run merely because preview output appeared.
                    signal=(widget.layerChanged if
                            isinstance(widget,QgsMapLayerComboBox) else
                            widget.currentIndexChanged)
                    signal.connect(
                        lambda *_args,source=widget:self._prepared_controls_changed(source))
            for widget in page.findChildren(QCheckBox):
                if widget not in excluded:
                    widget.toggled.connect(
                        lambda *_args,source=widget:self._prepared_controls_changed(source))
        self.workspace.textChanged.connect(
            lambda *_:self._prepared_controls_changed(self.workspace))
        self.morphological_factor_schedule.textChanged.connect(
            lambda *_:self._prepared_controls_changed(self.morphological_factor_schedule))
        self.fraction_table.cellChanged.connect(
            lambda *_:self._prepared_controls_changed(self.fraction_table))

    def _arm_prepared_controls(self):
        self._prepared_controls_armed=True
        self._prepared_controls_dirty=False

    def _prepared_controls_changed(self,source=None):
        if not self._prepared_controls_armed:return
        # A GPU continuation may choose a later terminal time without changing
        # its prepared numerical method or checkpoint identity.
        if source is self.end_time and self.execution_backend.currentData()=="gpu":return
        self._prepared_controls_dirty=True
        self.run_button.setEnabled(False)
        self.run_status.setText(
            "Preparation inputs changed; press Prepare Run again before execution.")

    def _assert_prepared_controls_current(self):
        if self._prepared_controls_dirty:
            raise ValueError(
                "Preparation inputs changed after Prepare Run; prepare the case again")
        # This also protects restored/legacy QGIS sessions, whose widgets were
        # populated before change tracking was connected.
        try:
            prepared=json.loads((self.run_root/"project.json").read_text(encoding="utf-8"))
            prepared_cfl=float(prepared["numerics"].get("cfl",.45))
        except (OSError,ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:
            raise ValueError(
                "Prepared project identity is missing or invalid; Prepare Run again") from exc
        if not math.isclose(prepared_cfl,self.cfl.value(),rel_tol=0.,abs_tol=1e-12):
            self._prepared_controls_changed(self.cfl)
            raise ValueError(
                "CFL differs from the prepared input deck; Prepare Run again")

    def _domain(self):
        layer=self.domain_layer.currentLayer()
        if layer is None:raise ValueError("Select a polygon domain layer")
        features=layer.selectedFeatures() or list(layer.getFeatures())
        if len(features)!=1:raise ValueError("Select exactly one polygon feature, or use a one-feature domain layer")
        geometry=features[0].geometry()
        if geometry.isMultipart():
            parts=geometry.asMultiPolygon()
            if len(parts)!=1:raise ValueError("The calculation domain must contain one polygon part")
            polygon=parts[0]
        else:polygon=geometry.asPolygon()
        if len(polygon)!=1:raise ValueError("Domain must be one polygon without holes")
        vertices=[];ignored=[]
        for source_index,point in enumerate(polygon[0]):
            coordinate=(float(point.x()),float(point.y()))
            if vertices and coordinate==vertices[-1]:ignored.append(source_index);continue
            vertices.append(coordinate)
        if vertices and vertices[0]!=vertices[-1]:vertices.append(vertices[0])
        layer.setCustomProperty("braids/ignored_consecutive_domain_vertices",json.dumps(ignored))
        domain_id=str(layer.customProperty("braids/domain_id",layer.name())).strip() or "braids_domain"
        return PolygonDomain(domain_id,vertices,layer.crs().toWkt())

    def _source_path(self,layer,label):
        if layer is None:raise ValueError(f"Select {label}")
        path=Path(layer.source().split("|",1)[0]).expanduser().resolve()
        if not path.is_file():raise ValueError(f"{label} must be a saved file-backed layer")
        return path

    def _source_layer_name(self,layer,default):
        for option in layer.source().split("|")[1:]:
            if option.startswith("layername="):return option.split("=",1)[1]
        path=Path(layer.source().split("|",1)[0])
        return path.stem if path.suffix.lower()==".shp" else default

    def _select_boundary_faces(self,card):
        try:
            layer=self.edge_layer.currentLayer()
            if layer is None:raise ValueError("Select the boundary-face layer")
            index=layer.fields().indexFromName("bc_id")
            if index<0:raise ValueError("Selected layer is not a BRAIDS boundary-face layer")
            ids=[feature.id() for feature in layer.getFeatures() if str(feature[index] or "")==card.bc_id]
            if not ids:raise ValueError(f"No polygon faces are assigned to {card.bc_id!r}")
            layer.selectByIds(ids)
            if self.iface:
                self.iface.setActiveLayer(layer)
                canvas=getattr(self.iface,"mapCanvas",lambda:None)()
                if canvas is not None:canvas.zoomToSelected(layer)
            self.setup_status.setText(f"Selected {len(ids)} polygon face(s) for {card.bc_id}.")
            return ids
        except Exception as exc:self._message("Select boundary faces",str(exc),True);return None

    def _boundary_polyline(self,bc_id):
        domain=self._domain();assignments=self._assignments(domain)
        coordinates,normal=boundary_polyline(domain,assignments,bc_id,
            tolerance_m=1e-7*max(1.,max(edge.length_m for edge in domain.edges)))
        return domain,coordinates,normal

    def _selected_roughness_zones(self,domain):
        """Load one physical zone schema without reinterpreting old layers."""
        layer=self.manning_layer.currentLayer()
        if layer is None:return "typed",(),ROUGHNESS_LAYER_NAME
        path=self._source_path(layer,"a roughness-zone polygon layer")
        layer_name=self._source_layer_name(layer,ROUGHNESS_LAYER_NAME)
        mode=str(self.hydraulic_roughness_mode.currentData())
        fields={field.name() for field in layer.fields()}
        if {"zone_id","mode","parameter","priority"}<=fields:
            return "typed",load_typed_roughness_zones(
                path,domain.crs_wkt,mode,layer_name),layer_name
        if {"zone_id","manning_n"}<=fields:
            if mode!="manning":
                raise ValueError(
                    "A legacy Manning-zone layer cannot be used for Chezy or "
                    "White-Colebrook; create a typed roughness-zone layer")
            return "legacy_manning",load_manning_zones(
                path,domain.crs_wkt,layer_name),layer_name
        raise ValueError(
            "Selected polygons are neither BRAIDS typed roughness zones nor a legacy Manning layer")

    def _roughness_field(self,mesh,domain):
        schema,zones,_=self._selected_roughness_zones(domain)
        if schema=="legacy_manning":
            return map_manning_to_cells(mesh,self.manning.value(),zones)
        return map_typed_roughness_to_cells(
            mesh,str(self.hydraulic_roughness_mode.currentData()),
            self.manning.value(),zones)

    def _boundary_roughness(self,coordinates,domain):
        schema,zones,_=self._selected_roughness_zones(domain)
        if schema=="legacy_manning":
            return boundary_manning(coordinates,self.manning.value(),zones)
        return boundary_typed_roughness(
            coordinates,str(self.hydraulic_roughness_mode.currentData()),
            self.manning.value(),zones)

    def _sample_boundary_section(self,bc_id):
        domain,coordinates,normal=self._boundary_polyline(bc_id)
        dem=self._source_path(self.dem_layer.currentLayer(),"a terrain DEM")
        sampler=_RasterSampler(dem,allow_undeclared_metres=True)
        spatial=_projected_metres(domain.crs_wkt,"Boundary")
        if not sampler.spatial.IsSame(spatial):
            raise ValueError("DEM and boundary CRS must match; reproject the data explicitly")
        section=build_section(sampler,coordinates,bc_id=bc_id,inward_normal=normal,
            sample_spacing_m=None,datum_m=None,max_stage_m=None,seed_station_m=None)
        assignments=self._assignments(domain);mesh=map_cut_cells(
            domain,assignments,grid_for_domain(domain,self.spacing.value()))
        if self.BOUNDARY_CELL_AREA_FRACTION>0:
            mesh,_,_=agglomerate_small_cells(mesh,self.BOUNDARY_CELL_AREA_FRACTION)
        field,_=self._roughness_field(mesh,domain)
        card=self.boundary_cards.get(bc_id)
        if card is not None and card.mode.currentData()=="inlet_normal":
            section=section_with_cell_roughness(section,coordinates,mesh,field)
        return domain,coordinates,normal,section

    def _automatic_normal_settings(self,bc_id,settings,domain=None,coordinates=None,normal=None):
        row=dict(settings)
        if row.get("mode")!="inlet_normal":return row,None
        if domain is None or coordinates is None or normal is None:
            domain,coordinates,normal=self._boundary_polyline(bc_id)
        dem=self._source_path(self.dem_layer.currentLayer(),"a terrain DEM")
        slope,slope_audit=terrain_inward_slope(dem,coordinates,normal)
        coefficient,roughness_audit=self._boundary_roughness(coordinates,domain)
        if coefficient<=0:raise ValueError(
            "Normal-depth inlet requires a positive default or boundary-zone roughness parameter")
        roughness_audit["production_control"]=(
            "sampled whole-cell raw roughness field grouped into contiguous section subsections; "
            "the scalar value is retained only for legacy/fallback input")
        row["energy_slope"]=slope;row["boundary_manning_n"]=coefficient
        card=self.boundary_cards.get(bc_id)
        if card is not None:card.set_automatic_normal_inputs(slope,coefficient)
        return row,{"terrain":slope_audit,"roughness":roughness_audit}

    def _resolve_terrain_slopes(self,rows):
        resolved={name:dict(row) for name,row in rows.items()}
        for name,row in resolved.items():
            resolved[name],_=self._automatic_normal_settings(name,row)
        return resolved

    def _point_along(self,coordinates,station):
        remaining=float(station)
        for start,end in zip(coordinates,coordinates[1:]):
            length=math.dist(start,end)
            if remaining<=length or end==coordinates[-1]:
                fraction=min(1.,max(0.,remaining/length));return tuple(a+fraction*(b-a) for a,b in zip(start,end))
            remaining-=length
        return coordinates[-1]

    def _opening_preview_layer(self,bc_id,coordinates,geometry,crs_wkt):
        for layer_id,layer in list(QgsProject.instance().mapLayers().items()):
            if layer.customProperty("braids/boundary_preview_id","")==bc_id:QgsProject.instance().removeMapLayer(layer_id)
        if not geometry.wet_intervals:return None
        layer=QgsVectorLayer("LineString",f"Wetted boundary section · {bc_id}","memory");crs=QgsCoordinateReferenceSystem();crs.createFromWkt(crs_wkt);layer.setCrs(crs)
        features=[]
        for interval in geometry.wet_intervals:
            feature=QgsFeature();feature.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(*self._point_along(coordinates,interval.start_station_m)),QgsPointXY(*self._point_along(coordinates,interval.end_station_m))]));features.append(feature)
        layer.dataProvider().addFeatures(features);layer.setRenderer(QgsSingleSymbolRenderer(QgsLineSymbol.createSimple({"color":"25,118,210,255","width":"1.8"})))
        layer.setCustomProperty("braids/boundary_preview_id",bc_id);QgsProject.instance().addMapLayer(layer);return layer

    def _hydraulic_roughness_options(self):
        """Return the typed resistance identity used by preparation/runtime."""
        mode=str(self.hydraulic_roughness_mode.currentData())
        chzk=bool(self.white_colebrook_chzk2d.isChecked())
        return {
            "default_manning_n":self.manning.value(),
            "hydraulic_roughness_mode":mode,
            "white_colebrook_chzk2d":chzk,
            # Typed resistance requires the same finite work depth used by the
            # native friction/source and live normal-depth resolver.
            "roughness_minimum_work_depth_m":.01 if mode!="manning" or chzk else 0.,
        }

    def _preview_boundary(self,card):
        try:
            if self._select_boundary_faces(card) is None:return
            domain,coordinates,normal,section=self._sample_boundary_section(card.bc_id)
            settings,normal_audit=self._automatic_normal_settings(
                card.bc_id,card.values(),domain,coordinates,normal)
            spec=CutBoundary(**settings)
            roughness_options=self._hydraulic_roughness_options()
            resolved_series=resolve_hydraulic_series(spec,section,**roughness_options)
            target=(resolved_series["knots"][0] if resolved_series is not None else
                    resolve_boundary_target(spec,section,**roughness_options))
            stage=target["target_stage_m"]
            if stage is None or settings["mode"]=="weir":stage=self._initial_stage_value()
            geometry=None if stage is None else section.geometry(stage)
            if geometry is not None:self._opening_preview_layer(card.bc_id,coordinates,geometry,domain.crs_wkt)
            water_x=[];water_y=[]
            if geometry is not None:
                for interval in geometry.wet_intervals:water_x.extend((interval.start_station_m,interval.end_station_m,float("nan")));water_y.extend((stage,stage,float("nan")))
            series=[("Terrain",section.stations_m,section.bed_elevations_m,QColor(55,55,55))]
            if geometry is not None:series.append(("Water line",water_x,water_y,COLORS[0]))
            plots=[{"title":f"Cross section · {card.bc_id}","x_label":"Station (m)","y_label":"Elevation (m)","series":series}]
            show_plot_dialog(self,f"BRAIDS boundary preview · {card.bc_id}",plots)
            if geometry is None:
                card.summary.setText("Terrain section preview. The wetted outlet section will follow the simulated water level; the initial domain is dry.")
                self.setup_status.setText(f"Previewed terrain section for {card.bc_id}; no initial water level is configured.")
                return
            discharge=(target["discharge_m3_s"] if resolved_series is not None else settings["discharge_m3_s"])
            velocity=discharge/geometry.area_m2 if spec.is_inlet and geometry.area_m2>0 else 0.
            qualifier=(f"Resolved {settings['mode']} target: level {stage:.6g} m, depth above datum {stage-section.datum_m:.6g} m. "
                if spec.is_inlet else (f"Weir crest {settings['stage_m']:.6g} m; preview uses initial level {stage:.6g} m. " if settings["mode"]=="weir" else
                ("The critical outlet level is recomputed from the outward discharge during the run. " if settings["mode"]=="critical" else "")))
            automatic=("" if normal_audit is None else
                "Normal depth resolved automatically from the DEM and Mesh Manning settings. ")
            hydraulic=target.get("hydraulic_solution")
            roughness_text=("" if hydraulic is None or hydraulic.get("roughness_mode")!="spatial subsections" else
                f"Manning conveyance uses {len(hydraulic.get('subsection_conveyances_m3_s_per_sqrt_slope',()))} contiguous roughness subsection(s). ")
            card.summary.setText(qualifier+automatic+roughness_text+f"Wetted cross-section area {geometry.area_m2:.4g} m², top width {geometry.top_width_m:.4g} m, wetted perimeter {geometry.perimeter_m:.4g} m, mean inlet velocity {velocity:.4g} m/s. Accepted numerical flux is recorded during the run.")
            self.setup_status.setText(f"Previewed {card.bc_id}: {len(geometry.wet_intervals)} wetted section interval(s) at one water level, {geometry.area_m2:.4g} m².")
        except Exception as exc:self._message("Preview boundary section",str(exc),True)

    def _plot_hydraulic_card(self,card):
        try:
            settings=card.values();times=(0.,self.end_time.value());plots=[];section=None
            if settings["mode"] in ("inlet_stage","inlet_depth","inlet_normal","inlet_critical"):
                domain,coordinates,normal,section=self._sample_boundary_section(card.bc_id)
                settings,normal_audit=self._automatic_normal_settings(
                    card.bc_id,settings,domain,coordinates,normal)
                if normal_audit is not None:self.setup_status.setText(
                    f"{card.bc_id}: normal-depth inputs resolved automatically from the DEM and Mesh Manning settings.")
            spec=CutBoundary(**settings)
            roughness_options=self._hydraulic_roughness_options()
            resolved=resolve_hydraulic_series(spec,section,**roughness_options)
            if resolved is not None:
                knots=resolved["knots"];times=tuple(knot["time_s"] for knot in knots)
                discharge=tuple(knot["discharge_m3_s"] for knot in knots)
                stages=tuple(knot["target_stage_m"] for knot in knots)
            else:
                target=resolve_boundary_target(spec,section,**roughness_options)
                discharge=(settings["discharge_m3_s"],)*2;stages=(target["target_stage_m"],)*2
            if spec.is_inlet:
                plots.append({"title":f"Total discharge · {card.bc_id}","y_label":"Q (m³/s)","series":[("Discharge",times,discharge,COLORS[0])]})
            if stages[0] is not None:
                label="Crest" if settings["mode"]=="weir" else "Water level"
                plots.append({"title":f"Configured {label.lower()} · {card.bc_id}","y_label":"Elevation (m)","series":[(label,times,stages,COLORS[1])]})
            if not plots:raise ValueError("This outlet has no prescribed time series; its level is computed from the live flow state")
            show_plot_dialog(self,f"BRAIDS hydraulic forcing · {card.bc_id}",plots)
        except Exception as exc:self._message("Plot hydraulic forcing",str(exc),True)

    def _plot_sediment_card(self,card):
        try:
            values=card.values()
            if values.get("fraction_sediment_series"):
                plots=[]
                for index,(name,series) in enumerate(values["fraction_sediment_series"].items()):
                    times=series["times_s"]
                    plots.append({"title":f"Suspended concentration · {card.bc_id} · {name}",
                        "y_label":"Concentration (kg/m³)","series":[(name,times,series["concentration_kg_m3"],COLORS[index%len(COLORS)])]})
                    plots.append({"title":f"Bed-load supply · {card.bc_id} · {name}",
                        "y_label":"Total bed load (kg/s)","series":[(name,times,series["bed_input_kg_s"],COLORS[index%len(COLORS)])]})
                show_plot_dialog(self,f"BRAIDS sediment forcing · {card.bc_id}",plots);return
            if values.get("fraction_concentration_kg_m3"):
                times=(0.,self.end_time.value());plots=[]
                for index,(name,concentration) in enumerate(values["fraction_concentration_kg_m3"].items()):
                    plots.append({"title":f"Suspended concentration · {card.bc_id} · {name}",
                        "y_label":"Concentration (kg/m³)","series":[(name,times,(concentration,)*2,COLORS[index%len(COLORS)])]})
                if values.get("bedload_boundary_mode")!="equilibrium_capacity":
                    for index,(name,bed) in enumerate(values["fraction_bed_input_kg_s"].items()):
                        plots.append({"title":f"Bed-load supply · {card.bc_id} · {name}",
                            "y_label":"Total bed load (kg/s)","series":[(name,times,(bed,)*2,COLORS[index%len(COLORS)])]})
                show_plot_dialog(self,f"BRAIDS sediment forcing · {card.bc_id}",plots);return
            if values["sediment_series"]:
                series=values["sediment_series"];times=series["times_s"];concentration=series["concentration_kg_m3"];bed=series["bed_input_kg_s"]
                bed_label="Total bed load (kg/s)"
            else:
                times=(0.,self.end_time.value());concentration=(values["concentration_kg_m3"],)*2
                if values.get("bedload_boundary_mode")=="equilibrium_capacity":bed=None
                else:
                    total=values["bed_input_kg_s"] is not None;bed=(values["bed_input_kg_s"] if total else values["bed_input_kg_m_s"],)*2
                    bed_label="Total bed load (kg/s)" if total else "Bed load (kg/(m·s))"
            plots=[{"title":f"Suspended concentration · {card.bc_id}","y_label":"Concentration (kg/m³)","series":[("Concentration",times,concentration,COLORS[0])]}]
            if bed is not None:plots.append({"title":f"Bed-load supply · {card.bc_id}","y_label":bed_label,"series":[("Bed load",times,bed,COLORS[1])]})
            show_plot_dialog(self,f"BRAIDS sediment forcing · {card.bc_id}",plots)
        except Exception as exc:self._message("Plot sediment forcing",str(exc),True)

    def _drawing_crs(self):
        dem=self.dem_layer.currentLayer()
        crs=dem.crs() if dem is not None else QgsProject.instance().crs()
        if crs is None or not crs.isValid():raise ValueError("Set a valid projected QGIS project CRS or select the DEM first")
        if crs.isGeographic():raise ValueError("BRAIDS drawing layers require a projected CRS with metre units")
        return crs

    def _unique_path(self,stem):
        workspace=self._workspace_path()
        path=workspace/f"{stem}.gpkg";number=2
        while path.exists():path=workspace/f"{stem}_{number}.gpkg";number+=1
        return path

    def _style_polygon_layer(self,layer,fill,outline,label_expression=None):
        symbol=QgsFillSymbol.createSimple({"color":fill,"outline_color":outline,"outline_width":"0.55"})
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))
        if label_expression:
            settings=QgsPalLayerSettings();settings.fieldName=label_expression;settings.isExpression=True
            text_format=QgsTextFormat();text_format.setColor(QColor(35,35,35));text_format.setSize(9);settings.setFormat(text_format)
            layer.setLabeling(QgsVectorLayerSimpleLabeling(settings));layer.setLabelsEnabled(True)
        layer.triggerRepaint()

    def _style_boundary_layer(self,layer):
        if layer is None:return
        symbol=QgsLineSymbol.createSimple({"color":"210,0,0,255","width":"1.2"})
        layer.setRenderer(QgsSingleSymbolRenderer(symbol));layer.triggerRepaint()

    def _enable_polygon_snapping(self):
        if not self.snap_polygons.isChecked():return
        config=QgsProject.instance().snappingConfig();config.setEnabled(True)
        config.setMode(QgsSnappingConfig.SnappingMode.AllLayers)
        config.setTypeFlag(Qgis.SnappingType.Vertex|Qgis.SnappingType.Segment)
        config.setTolerance(12.);config.setUnits(Qgis.MapToolUnit.Pixels);config.setIntersectionSnapping(True)
        QgsProject.instance().setSnappingConfig(config)

    def _next_identifier(self,layer,field,prefix):
        index=layer.fields().indexFromName(field)
        existing={str(value) for value in layer.uniqueValues(index)} if index>=0 else set()
        number=1
        while f"{prefix}_{number}" in existing:number+=1
        return f"{prefix}_{number}"

    def _manning_identifier(self,layer,value,excluded=(),reserved=()):
        index=layer.fields().indexFromName("zone_id");excluded=set(excluded)
        existing={str(feature[index]) for feature in layer.getFeatures()
                  if feature.id() not in excluded and feature[index] is not None}|set(reserved)
        label={"manning":"Manning n","chezy":"Chezy C",
               "white_colebrook":"Nikuradse ks"}.get(
                   str(self.hydraulic_roughness_mode.currentData()),"Roughness")
        base=f"{label} = {float(value):g}";name=base;number=2
        while name in existing:name=f"{base} ({number})";number+=1
        return name

    def _start_polygon_capture(self,layer,attributes,status,completion=None):
        self._start_feature_capture(layer,attributes,status,completion,"polygon")

    def _start_feature_capture(self,layer,attributes,status,completion=None,geometry_kind="polygon"):
        if self.iface is None:raise ValueError("Open the plugin inside QGIS to draw on the map canvas")
        if self._capture_connection is not None:
            self._cancel_polygon_capture(update_status=False)
        config=layer.editFormConfig();config.setSuppress(QgsEditFormConfig.SuppressOn);layer.setEditFormConfig(config)
        for name,value in attributes.items():
            index=layer.fields().indexFromName(name)
            if index<0:raise ValueError(f"Drawing layer is missing required field {name!r}")
            expression=("'"+value.replace("'","''")+"'" if isinstance(value,str)
                        else repr(value) if isinstance(value,int) and not isinstance(value,bool)
                        else repr(float(value)))
            layer.setDefaultValueDefinition(index,QgsDefaultValue(expression))
        self._enable_polygon_snapping();self.iface.setActiveLayer(layer)
        self._capture_started_editing=not layer.isEditable()
        if self._capture_started_editing and not layer.startEditing():
            self._capture_started_editing=False;raise ValueError(f"{geometry_kind.capitalize()} layer cannot be edited")
        self._capture_geometry_kind=geometry_kind

        def captured(feature_id):
            try:layer.featureAdded.disconnect(captured)
            except (TypeError,RuntimeError):pass
            self._capture_connection=None
            QTimer.singleShot(0,lambda:self._finish_feature_capture(layer,feature_id,attributes,status,completion,geometry_kind))

        self._capture_connection=(layer,captured);layer.featureAdded.connect(captured);self.iface.actionAddFeature().trigger()
        ending="close" if geometry_kind=="polygon" else "finish"
        self.setup_status.setText(f"Click {geometry_kind} vertices in the map. Right-click to {ending} and save it."+
            (" Snapping is active." if self.snap_polygons.isChecked() else ""))

    def _finish_feature_capture(self,layer,feature_id,attributes,status,completion,geometry_kind):
        try:
            for name,value in attributes.items():layer.changeAttributeValue(feature_id,layer.fields().indexFromName(name),value)
            if not layer.commitChanges():raise ValueError(f"QGIS could not save the new {geometry_kind}")
            if completion:completion(layer,feature_id)
            layer.selectByIds([feature_id]);layer.triggerRepaint()
            if self.iface:self.iface.actionPan().trigger()
            self._capture_started_editing=False
            self.setup_status.setText(status)
        except Exception as exc:
            if layer.isEditable():layer.rollBack()
            self._capture_started_editing=False
            self._message(f"Save BRAIDS {geometry_kind}",str(exc),True)

    def _cancel_polygon_capture(self,expected_layer=None,update_status=True):
        if self._capture_connection is None:return False
        layer,callback=self._capture_connection
        if expected_layer is not None and layer is not expected_layer:return False
        try:layer.featureAdded.disconnect(callback)
        except (TypeError,RuntimeError):pass
        self._capture_connection=None
        if self.iface is not None:
            action=getattr(self.iface,"actionPan",None)
            if callable(action):action().trigger()
        if self._capture_started_editing:
            try:
                if layer.isEditable():layer.rollBack()
            except RuntimeError:pass
        self._capture_started_editing=False
        if update_status:self.setup_status.setText(f"{getattr(self,'_capture_geometry_kind','polygon').capitalize()} drawing cancelled.")
        return True

    def _inlet_guidance_layers(self):
        return [layer for layer in QgsProject.instance().mapLayers().values()
                if isinstance(layer,QgsVectorLayer) and (layer.customProperty(GUIDE_PROPERTY,False) or
                    (self._source_layer_name(layer,"")==GUIDE_LAYER_NAME and layer.fields().indexFromName("guide_id")>=0))]

    def _exclude_guidance_from_boundaries(self,*_):
        self.edge_layer.setExceptedLayerList(self._inlet_guidance_layers())

    def _hide_inlet_guides(self):
        project=QgsProject.instance();guides=self._inlet_guidance_layers()
        for layer in guides:
            self._cancel_polygon_capture(expected_layer=layer,update_status=False)
            layer.removeSelection();node=project.layerTreeRoot().findLayer(layer.id())
            if node is not None:node.setItemVisibilityChecked(False)
        self._exclude_guidance_from_boundaries()
        if self.iface:
            target=self.edge_layer.currentLayer() or self.domain_layer.currentLayer()
            if target is not None:self.iface.setActiveLayer(target)

    def _condition_tab_selected(self,index):
        if self.tabs.tabText(index).replace("&&","&")=="BC & IC":self._hide_inlet_guides()

    def _style_inlet_guides(self,layer):
        layer.setCustomProperty(GUIDE_PROPERTY,True);layer.setDisplayExpression('"name"')
        layer.setRenderer(QgsSingleSymbolRenderer(QgsLineSymbol.createSimple(
            {"line_color":"115,70,170,255","line_width":"0.7","line_style":"dash"})))
        layer.triggerRepaint()

    def create_inlet_guide(self):
        try:
            path=self._workspace_path()/"braids_inlet_guides.gpkg"
            layer=next((item for item in self._inlet_guidance_layers()
                        if Path(item.source().split('|',1)[0]).resolve()==path),None)
            if layer is None:
                if not path.exists():create_guidance_dataset(path,self._drawing_crs().toWkt())
                layer=QgsVectorLayer(f"{path}|layername={GUIDE_LAYER_NAME}","Inlet Guidance Lines","ogr")
                if not layer.isValid() or any(layer.fields().indexFromName(name)<0 for name in ("guide_id","name")):
                    raise ValueError(f"Could not open a valid inlet-guidance layer in {path}")
                self._style_inlet_guides(layer);QgsProject.instance().addMapLayer(layer)
            node=QgsProject.instance().layerTreeRoot().findLayer(layer.id())
            if node is not None:node.setItemVisibilityChecked(True)
            self._exclude_guidance_from_boundaries()
            identifier=self._next_identifier(layer,"guide_id","guide")
            self._start_feature_capture(layer,{"guide_id":identifier,"name":f"Inlet guide {identifier.split('_')[-1]}"},
                "Inlet guide saved. Draw the domain using these guides; they will be hidden when the domain is saved.",geometry_kind="polyline")
        except Exception as exc:self._message("Draw Inlet Guide",str(exc),True)

    def create_domain(self):
        try:
            crs=self._drawing_crs();path=self._unique_path("calculation_domain")
            create_domain_dataset(path,crs.toWkt())
            layer=QgsVectorLayer(f"{path}|layername={DOMAIN_LAYER_NAME}","Calculation Domain","ogr")
            if not layer.isValid():raise ValueError("QGIS could not load the calculation-domain layer")
            layer.setCustomProperty("braids/domain_id","calculation_domain");layer.setDisplayExpression('"name"')
            self._style_polygon_layer(layer,"46,134,193,65","28,86,132,255",'\'Calculation Domain\'')
            QgsProject.instance().addMapLayer(layer);self.domain_layer.setLayer(layer)
            self._start_polygon_capture(layer,{"domain_id":"calculation_domain","name":"Calculation Domain"},
                "Calculation Domain saved; inlet guides hidden. Create its boundary faces, then select inlet/outlet faces.",
                completion=lambda *_:self._hide_inlet_guides())
        except Exception as exc:self._message("Draw Calculation Domain",str(exc),True)

    def create_water_body(self):
        try:
            layer=self.water_body_layer.currentLayer()
            if layer is None:
                domain=self._domain();path=self._unique_path(f"{domain.domain_id}_initial_water")
                create_water_body_dataset(path,domain.crs_wkt)
                layer=QgsVectorLayer(f"{path}|layername={WATER_BODY_LAYER_NAME}","Initial Water Body","ogr")
                if not layer.isValid():raise ValueError("QGIS could not load the generated initial-water layer")
                QgsProject.instance().addMapLayer(layer);self.water_body_layer.setLayer(layer)
            required=(layer.fields().indexFromName("water_id"),layer.fields().indexFromName("name"),layer.fields().indexFromName("stage_m"))
            if min(required)<0:
                raise ValueError("The selected polygon can be used as initial water, but adding features requires a BRAIDS initial-water layer. Clear the layer choice and press this button again.")
            expression='concat(\'Initial water = \', format_number("stage_m", 3), \' m\')';layer.setDisplayExpression(expression)
            self._style_polygon_layer(layer,"41,182,246,80","2,119,189,255",expression)
            water_id=self._next_identifier(layer,"water_id","water");stage=self._initial_stage_value(required=True)
            self._legacy_lake_seed_xy=None
            self._start_polygon_capture(layer,{"water_id":water_id,"name":"Initial Water Body","stage_m":stage},
                f"{water_id} saved at constant water level {stage:g} m. Cells touched by its area will be initialized below that level.")
        except Exception as exc:self._message("Draw Initial Water Body",str(exc),True)

    def start_water_body_from_dem(self):
        """Arm one map click that selects a connected basin below the entered level."""
        try:
            stage=self._initial_stage_value(required=True);terrain=self.dem_layer.currentLayer()
            if terrain is None or not terrain.isValid() or not terrain.crs().isValid():
                raise ValueError("Select a valid terrain DEM before creating the water body")
            if self.iface is None or getattr(self.iface,"mapCanvas",lambda:None)() is None:
                raise ValueError("The QGIS map canvas is unavailable")
            self._finish_water_body_point_capture()
            canvas=self.iface.mapCanvas();self._water_body_previous_map_tool=canvas.mapTool()
            self._water_body_map_tool=QgsMapToolEmitPoint(canvas)
            self._water_body_map_tool.canvasClicked.connect(self._create_water_body_from_dem_point)
            canvas.setMapTool(self._water_body_map_tool)
            self.setup_status.setText(
                f"Click inside the basin. Connected DEM cells at or below {stage:g} m will define the initial water body. Right-click cancels.")
        except Exception as exc:self._message("Create Initial Water Body from DEM",str(exc),True)

    def _finish_water_body_point_capture(self):
        if self._water_body_map_tool is None:return
        canvas=None if self.iface is None else getattr(self.iface,"mapCanvas",lambda:None)()
        if canvas is not None:
            try:self._water_body_map_tool.canvasClicked.disconnect(self._create_water_body_from_dem_point)
            except (TypeError,RuntimeError):pass
            if self._water_body_previous_map_tool is not None:canvas.setMapTool(self._water_body_previous_map_tool)
            else:canvas.unsetMapTool(self._water_body_map_tool)
        self._water_body_map_tool=None;self._water_body_previous_map_tool=None

    def _read_dem_window(self,terrain,left,right,bottom,top,cell_x,cell_y):
        extent=terrain.extent();xmin=float(extent.xMinimum())+left*cell_x
        xmax=float(extent.xMinimum())+right*cell_x;ymin=float(extent.yMinimum())+bottom*cell_y
        ymax=float(extent.yMinimum())+top*cell_y;width=right-left;height=top-bottom
        window=QgsRectangle(xmin,ymin,xmax,ymax);provider=terrain.dataProvider()
        block=provider.block(1,window,width,height)
        if block is None or not block.isValid():raise ValueError("QGIS could not read the selected DEM window")
        data_types={Qgis.DataType.Byte:np.uint8,Qgis.DataType.UInt16:np.uint16,
            Qgis.DataType.Int16:np.int16,Qgis.DataType.UInt32:np.uint32,
            Qgis.DataType.Int32:np.int32,Qgis.DataType.Float32:np.float32,
            Qgis.DataType.Float64:np.float64}
        source_type=data_types.get(block.dataType());raw=bytes(block.data())
        if source_type is not None and len(raw)==width*height*np.dtype(source_type).itemsize:
            values=np.frombuffer(raw,dtype=source_type).astype(float,copy=True).reshape(height,width)
        else:
            values=np.fromiter((block.value(row,column) for row in range(height)
                for column in range(width)),dtype=float,count=width*height).reshape(height,width)
        if provider.sourceHasNoDataValue(1):
            nodata=provider.sourceNoDataValue(1)
            if np.isfinite(nodata):values[np.isclose(values,nodata)]=np.nan
        values[~np.isfinite(values)]=np.nan
        return WaterBodyRaster(values[::-1,:],xmin,xmax,ymin,ymax,cell_x,cell_y,terrain.crs().toWkt())

    def _water_body_raster_window(self,terrain,x,y):
        extent=terrain.extent();columns,rows=int(terrain.width()),int(terrain.height())
        if min(columns,rows)<1:raise ValueError("Terrain DEM has no raster cells")
        cell_x=float(extent.width())/columns;cell_y=float(extent.height())/rows
        if min(cell_x,cell_y)<=0 or not np.isfinite([cell_x,cell_y]).all():
            raise ValueError("Terrain DEM cell sizes must be positive and finite")
        xmin,ymin=float(extent.xMinimum()),float(extent.yMinimum())
        xmax,ymax=float(extent.xMaximum()),float(extent.yMaximum())
        if not (xmin<=x<xmax and ymin<=y<ymax):raise ValueError("The selected point is outside the terrain DEM")
        seed_column=min(columns-1,max(0,int(np.floor((x-xmin)/cell_x))))
        seed_row=min(rows-1,max(0,int(np.floor((y-ymin)/cell_y))))
        half_cells=256
        while True:
            left=max(0,seed_column-half_cells);right=min(columns,seed_column+half_cells+1)
            bottom=max(0,seed_row-half_cells);top=min(rows,seed_row+half_cells+1)
            raster=self._read_dem_window(terrain,left,right,bottom,top,cell_x,cell_y)
            seed=water_body_seed_cell(raster,x,y)
            mask=connected_water_body_mask(raster.z,seed,self._initial_stage_value(required=True),require_closed=False)
            touches=(bool(mask[:,0].any()),bool(mask[:,-1].any()),bool(mask[0,:].any()),bool(mask[-1,:].any()))
            source_edges=(left==0,right==columns,bottom==0,top==rows)
            if any(touch and edge for touch,edge in zip(touches,source_edges)):
                raise ValueError("The connected water body reaches the terrain DEM edge, so its contour is not closed. Use a DEM that encloses the water body or lower the water level")
            if not any(touches):return raster,mask
            if left==0 and right==columns and bottom==0 and top==rows:
                raise ValueError("The connected water body does not form a closed contour inside the terrain DEM")
            half_cells*=2

    def _create_water_body_from_dem_point(self,point,mouse_button):
        if mouse_button==Qt.MouseButton.RightButton:
            self._finish_water_body_point_capture();self.setup_status.setText("Initial-water point selection cancelled.");return
        try:
            terrain=self.dem_layer.currentLayer();stage=self._initial_stage_value(required=True)
            if terrain is None or not terrain.isValid():raise ValueError("Terrain DEM is no longer available")
            canvas_crs=self.iface.mapCanvas().mapSettings().destinationCrs();selected=QgsPointXY(point)
            if canvas_crs!=terrain.crs():
                selected=QgsCoordinateTransform(canvas_crs,terrain.crs(),QgsProject.instance().transformContext()).transform(selected)
            raster,mask=self._water_body_raster_window(terrain,selected.x(),selected.y())
            path=self._unique_path(f"initial_water_{stage:.3f}".replace("-","minus_").replace(".","_"))
            write_connected_water_body(path,raster,mask,water_level_m=stage,
                seed_x=selected.x(),seed_y=selected.y())
            layer=QgsVectorLayer(f"{path}|layername={WATER_BODY_LAYER_NAME}",f"Initial Water at {stage:g} m","ogr")
            if not layer.isValid():raise ValueError("QGIS could not load the derived initial-water polygon")
            expression='concat(\'Initial water = \', format_number("stage_m", 3), \' m\')'
            layer.setDisplayExpression(expression);self._style_polygon_layer(layer,"41,182,246,80","2,119,189,255",expression)
            layer.setCustomProperty("braids/derived_water_body",True);layer.setCustomProperty("braids/water_level_m",stage)
            QgsProject.instance().addMapLayer(layer);self.water_body_layer.setLayer(layer);self._legacy_lake_seed_xy=None
            self.setup_status.setText(
                f"Created the {stage:g} m initial-water polygon from {int(np.count_nonzero(mask))} connected DEM cells.")
        except Exception as exc:self._message("Create Initial Water Body from DEM",str(exc),True)
        finally:self._finish_water_body_point_capture()

    def create_edges(self):
        try:
            previous=self.edge_layer.currentLayer();domain=self._domain();path=create_edge_dataset(domain,self._unique_path(f"{domain.domain_id}_boundary_edges"))
            layer=QgsVectorLayer(f"{path}|layername={EDGE_LAYER_NAME}","BRAIDS boundary edges","ogr")
            if not layer.isValid():raise ValueError("QGIS could not load the generated boundary edge layer")
            self._style_boundary_layer(layer);QgsProject.instance().addMapLayer(layer);self.edge_layer.setLayer(layer)
            # A domain edit invalidates every old canonical edge ID and its
            # assignments. Replace a prior BRAIDS layer so stale faces cannot
            # remain selectable or be written into the next project save.
            previous_fields=set() if previous is None else {field.name() for field in previous.fields()}
            previous_is_braids=(previous is not None and previous is not layer and
                {"edge_id","bc_id","kind","domain_id"}<=previous_fields and
                self._source_layer_name(previous,"")==EDGE_LAYER_NAME)
            if previous_is_braids:
                previous.removeSelection();QgsProject.instance().removeMapLayer(previous.id())
            self._hide_inlet_guides();self.refresh_boundary_json()
            ignored=json.loads(self.domain_layer.currentLayer().customProperty("braids/ignored_consecutive_domain_vertices","[]"))
            suffix=f" Ignored {len(ignored)} redundant consecutive source vertex/vertices without modifying the source layer." if ignored else ""
            self.setup_status.setText("Boundary faces created as walls. Assign inlet/outlet IDs in BC & IC → Boundary conditions."+suffix)
        except Exception as exc:self._message("Create BRAIDS edges",str(exc),True)

    def apply_selected_edges(self):
        try:
            layer=self.edge_layer.currentLayer()
            if layer is None or not layer.selectedFeatureIds():raise ValueError("Select one or more features in the boundary edge layer")
            kind_index=layer.fields().indexFromName("kind");id_index=layer.fields().indexFromName("bc_id")
            edge_index=layer.fields().indexFromName("edge_id")
            if min(kind_index,id_index,edge_index)<0:raise ValueError("Selected layer is not a BRAIDS boundary-face layer")
            selected_fids=set(layer.selectedFeatureIds());features=list(layer.getFeatures())
            selected_edges=[str(feature[edge_index]) for feature in features if feature.id() in selected_fids]
            if len(selected_edges)!=len(selected_fids):raise ValueError("Could not resolve every selected boundary face")
            kind=self.boundary_kind.currentText();bc_id=self.boundary_id.text().strip();domain=self._domain()
            current=load_assignments(self._source_path(layer,"a boundary edge layer"),domain)
            replaced=kind!="wall" and any(value.bc_id==bc_id for value in current)
            # Build and validate the complete candidate before starting an edit.
            # A failed collinearity/connectivity check therefore changes nothing.
            candidate=reassign_boundary_faces(domain,current,selected_edges,kind,bc_id)
            desired={value.edge_id:(value.kind,value.bc_id) for value in candidate}
            started_here=not layer.isEditable()
            if started_here and not layer.startEditing():raise ValueError("Boundary edge layer cannot be edited")
            layer.beginEditCommand("Reassign BRAIDS boundary faces")
            try:
                for feature in features:
                    target_kind,target_id=desired.get(str(feature[edge_index]),("wall",""))
                    if not layer.changeAttributeValue(feature.id(),kind_index,target_kind):raise ValueError("Could not update boundary type")
                    if not layer.changeAttributeValue(feature.id(),id_index,target_id):raise ValueError("Could not update boundary ID")
                layer.endEditCommand()
            except Exception:
                layer.destroyEditCommand()
                if started_here:layer.rollBack()
                raise
            if not layer.commitChanges():
                layer.rollBack();raise ValueError("Could not save boundary edge assignments")
            self.refresh_boundary_json()
            action="Reassigned" if replaced else "Assigned"
            suffix=(f" {bc_id!r}; its previous face assignment was replaced."
                    if kind!="wall" and replaced else f" as {kind} {bc_id!r}.")
            self.setup_status.setText(f"{action} {len(selected_edges)} selected edge(s){suffix}")
        except Exception as exc:self._message("Assign BRAIDS edges",str(exc),True)

    def _delete_boundary(self,card):
        try:
            layer=self.edge_layer.currentLayer()
            if layer is None:raise ValueError("Select the boundary-face layer")
            id_index=layer.fields().indexFromName("bc_id")
            kind_index=layer.fields().indexFromName("kind")
            if min(id_index,kind_index)<0:raise ValueError("Selected layer is not a BRAIDS boundary-face layer")
            boundary_id=card.bc_id
            selected=[feature.id() for feature in layer.getFeatures()
                if str(feature[id_index] or "").strip()==boundary_id]
            if not selected:raise ValueError(f"No polygon faces are assigned to {boundary_id!r}")
            if not layer.isEditable() and not layer.startEditing():raise ValueError("Boundary edge layer cannot be edited")
            for feature_id in selected:
                layer.changeAttributeValue(feature_id,kind_index,"wall");layer.changeAttributeValue(feature_id,id_index,"")
            if not layer.commitChanges():raise ValueError(f"Could not delete boundary condition {boundary_id!r}")
            layer.removeSelection();self.refresh_boundary_json()
            self.setup_status.setText(f"Deleted boundary condition {boundary_id!r}; its {len(selected)} polygon face(s) are walls again.")
        except Exception as exc:self._message("Delete boundary condition",str(exc),True)

    def create_manning_zones(self):
        try:
            layer=self.manning_layer.currentLayer()
            if layer is None:
                domain=self._domain();path=self._unique_path(f"{domain.domain_id}_roughness_zones")
                create_typed_roughness_dataset(path,domain.crs_wkt)
                layer=QgsVectorLayer(f"{path}|layername={ROUGHNESS_LAYER_NAME}","BRAIDS Roughness Zones","ogr")
                if not layer.isValid():raise ValueError("QGIS could not load the generated roughness-zone layer")
                QgsProject.instance().addMapLayer(layer);self.manning_layer.setLayer(layer)
            fields={field.name() for field in layer.fields()}
            typed={"zone_id","mode","parameter","priority"}<=fields
            legacy={"zone_id","manning_n"}<=fields
            mode=str(self.hydraulic_roughness_mode.currentData())
            if not typed and not legacy:
                raise ValueError("Select a BRAIDS roughness-zone layer or clear the layer choice to create one")
            if legacy and mode!="manning":
                raise ValueError("Legacy Manning zones cannot store Chezy C or Nikuradse ks")
            expression=(
                'concat("mode", \' = \', format_number("parameter", 6), \' [p=\', "priority", \']\')'
                if typed else 'concat(\'Manning = \', format_number("manning_n", 6))')
            layer.setDisplayExpression(expression)
            self._style_polygon_layer(layer,"245,166,35,75","185,108,15,255",expression)
            value=self.manning_zone_value.value();zone_id=self._manning_identifier(layer,value)
            attributes=({"zone_id":zone_id,"mode":mode,"parameter":value,
                         "priority":self.manning_zone_priority.value()} if typed else
                        {"zone_id":zone_id,"manning_n":value})
            self._start_polygon_capture(layer,attributes,
                f"{zone_id} saved with priority {self.manning_zone_priority.value()}. Add another zone or edit it with normal QGIS tools.")
        except Exception as exc:self._message("Create roughness zones",str(exc),True)

    def create_refinement_zone(self):
        try:
            layer=self.refinement_layer.currentLayer()
            if layer is None:
                domain=self._domain();path=self._unique_path('refinement_zones')
                create_refinement_dataset(path,domain.crs_wkt)
                layer=QgsVectorLayer(f'{path}|layername={REFINEMENT_LAYER_NAME}','Refinement Zones','ogr')
                if not layer.isValid():raise ValueError('QGIS could not load refinement polygons')
                QgsProject.instance().addMapLayer(layer);self.refinement_layer.setLayer(layer)
            level=int(self.refinement_level.currentData());name=self._next_identifier(layer,'zone_id',f'refine_L{level}')
            expression="concat('Minimum L', \"min_level\")"
            self._style_polygon_layer(layer,'115,75,180,60','90,45,160,255',expression)
            self._start_polygon_capture(layer,{'zone_id':name,'min_level':level},f'{name} saved with minimum refinement level L{level}.')
        except Exception as exc:self._message('Create refinement zone',str(exc),True)

    def create_erodible_zone(self):
        try:
            layer=self.erodible_zone_layer.currentLayer()
            if layer is None:
                domain=self._domain();path=self._unique_path(f"{domain.domain_id}_erodible_zones")
                create_erodible_dataset(path,domain.crs_wkt)
                layer=QgsVectorLayer(f"{path}|layername={ERODIBLE_LAYER_NAME}","Erodible Thickness Zones","ogr")
                if not layer.isValid():raise ValueError("QGIS could not load the generated erodible-zone layer")
                QgsProject.instance().addMapLayer(layer);self.erodible_zone_layer.setLayer(layer)
            if min(layer.fields().indexFromName("zone_id"),layer.fields().indexFromName("thickness_m"))<0:
                raise ValueError("Select a BRAIDS erodible-zone layer or clear the layer choice to create one")
            expression='concat(\'Erodible = \', format_number("thickness_m", 3), \' m\')';layer.setDisplayExpression(expression)
            self._style_polygon_layer(layer,"151,105,55,75","91,62,32,255",expression)
            zone_id=self._next_identifier(layer,"zone_id","erodible");value=self.erodible_zone_value.value()
            self._start_polygon_capture(layer,{"zone_id":zone_id,"thickness_m":value},
                f"{zone_id} saved with erodible thickness {value:g} m. Add another zone or edit it with normal QGIS tools.")
        except Exception as exc:self._message("Create erodible zone",str(exc),True)

    def create_bank_zone(self):
        try:
            layer=self.bank_zone_layer.currentLayer()
            if layer is None:
                domain=self._domain();path=self._unique_path(f"{domain.domain_id}_erodible_bank_zones")
                create_bank_dataset(path,domain.crs_wkt)
                layer=QgsVectorLayer(f"{path}|layername={BANK_LAYER_NAME}","Erodible Bank Zones","ogr")
                if not layer.isValid():raise ValueError("QGIS could not load the generated erodible-bank layer")
                QgsProject.instance().addMapLayer(layer);self.bank_zone_layer.setLayer(layer)
            if layer.fields().indexFromName("zone_id")<0:
                raise ValueError("Select a BRAIDS erodible-bank layer or clear the layer choice to create one")
            expression='concat(\'Erodible bank: \', "zone_id")';layer.setDisplayExpression(expression)
            self._style_polygon_layer(layer,"194,24,91,65","136,14,79,255",expression)
            zone_id=self._next_identifier(layer,"zone_id","bank")
            self._start_polygon_capture(layer,{"zone_id":zone_id},
                f"{zone_id} saved as an erodible internal bank zone. Add another zone or edit it with normal QGIS tools.")
        except Exception as exc:self._message("Create erodible bank zone",str(exc),True)

    def _solid_obstacle_capture(self):
        layer=self.obstacle_layer.currentLayer()
        if layer is None:
            domain=self._domain();path=self._unique_path(f"{domain.domain_id}_solid_obstacles")
            create_obstacle_dataset(path,domain.crs_wkt)
            layer=QgsVectorLayer(f"{path}|layername={OBSTACLE_LAYER_NAME}","Solid Reflective Obstacles","ogr")
            if not layer.isValid():raise ValueError("QGIS could not load the generated solid-obstacle layer")
            QgsProject.instance().addMapLayer(layer);self.obstacle_layer.setLayer(layer)
        if layer.fields().indexFromName("obstacle_id")<0:
            raise ValueError("The selected layer can be used as imported obstacles, but drawing into it requires an obstacle_id field. Clear the layer choice to create a BRAIDS obstacle layer.")
        expression='concat(\'Solid obstacle: \', "obstacle_id")';layer.setDisplayExpression(expression)
        self._style_polygon_layer(layer,"70,70,70,150","20,20,20,255",expression)
        obstacle_id=self._next_identifier(layer,"obstacle_id","obstacle")
        attributes={"obstacle_id":obstacle_id}
        if layer.fields().indexFromName("name")>=0:attributes["name"]=obstacle_id
        return layer,obstacle_id,attributes

    def create_solid_obstacle(self):
        try:
            self._finish_obstacle_shape_capture()
            layer,obstacle_id,attributes=self._solid_obstacle_capture()
            self._start_polygon_capture(layer,attributes,
                f"{obstacle_id} saved as a fixed reflective solid obstacle. Preview the mesh to inspect its cut perimeter.")
        except Exception as exc:self._message("Create solid obstacle",str(exc),True)

    def create_solid_obstacle_circle(self):self._start_solid_obstacle_shape("circle")

    def create_solid_obstacle_ellipse(self):self._start_solid_obstacle_shape("ellipse")

    def _start_solid_obstacle_shape(self,kind):
        try:
            if self.iface is None or getattr(self.iface,"mapCanvas",lambda:None)() is None:
                raise ValueError("Open the plugin inside QGIS to draw on the map canvas")
            self._finish_obstacle_shape_capture();self._cancel_polygon_capture(update_status=False)
            layer,obstacle_id,attributes=self._solid_obstacle_capture()
            canvas=self.iface.mapCanvas();self._obstacle_previous_map_tool=canvas.mapTool()
            self._obstacle_map_tool=_ObstacleShapeMapTool(canvas,kind,
                lambda vertices:self._save_solid_obstacle_shape(layer,obstacle_id,attributes,vertices),
                self._cancel_solid_obstacle_shape,self.setup_status.setText)
            canvas.setMapTool(self._obstacle_map_tool)
            self.setup_status.setText("Click the circle centre. Right-click cancels." if kind=="circle" else
                "Click the ellipse centre. Right-click cancels.")
        except Exception as exc:self._message(f"Draw solid {kind}",str(exc),True)

    def _save_solid_obstacle_shape(self,layer,obstacle_id,attributes,vertices):
        try:
            geometry=QgsGeometry.fromPolygonXY([[QgsPointXY(*point) for point in vertices]])
            canvas=self.iface.mapCanvas();source_crs=canvas.mapSettings().destinationCrs()
            if source_crs!=layer.crs():
                geometry.transform(QgsCoordinateTransform(source_crs,layer.crs(),QgsProject.instance()))
            if geometry.isEmpty() or not geometry.isGeosValid():raise ValueError("Generated obstacle geometry is invalid")
            if not layer.isEditable() and not layer.startEditing():raise ValueError("Solid-obstacle layer cannot be edited")
            feature=QgsFeature(layer.fields());feature.setGeometry(geometry)
            for name,value in attributes.items():feature.setAttribute(name,value)
            if not layer.addFeature(feature):raise ValueError("QGIS could not add the solid obstacle")
            if not layer.commitChanges():raise ValueError("QGIS could not save the solid obstacle")
            identifier_index=layer.fields().indexFromName("obstacle_id")
            selected=[row.id() for row in layer.getFeatures() if str(row[identifier_index])==obstacle_id]
            layer.selectByIds(selected);layer.triggerRepaint();self._finish_obstacle_shape_capture()
            self.setup_status.setText(f"{obstacle_id} saved as a fixed reflective solid obstacle. Preview the mesh to inspect its cut perimeter.")
        except Exception as exc:
            if layer.isEditable():layer.rollBack()
            self._finish_obstacle_shape_capture();self._message("Save solid obstacle",str(exc),True)

    def _cancel_solid_obstacle_shape(self,error=None):
        self._finish_obstacle_shape_capture()
        if error:self._message("Draw solid obstacle",error,True)
        else:self.setup_status.setText("Solid-obstacle drawing cancelled.")

    def cancel_solid_obstacle_drawing(self):
        """Leave either obstacle digitizing mode and discard its unfinished feature."""
        if self._obstacle_map_tool is not None:
            self._cancel_solid_obstacle_shape();return
        layer=self.obstacle_layer.currentLayer()
        if self._cancel_polygon_capture(expected_layer=layer):
            self.setup_status.setText("Solid-obstacle drawing cancelled.");return
        self.setup_status.setText("No solid-obstacle drawing is active.")

    def delete_selected_obstacles(self):
        try:
            layer=self.obstacle_layer.currentLayer()
            if layer is None or not layer.isValid():raise ValueError("Choose the solid-obstacle layer first")
            selected=list(layer.selectedFeatureIds())
            if not selected:raise ValueError("Select one or more solid obstacles to delete")
            if not layer.isEditable() and not layer.startEditing():raise ValueError("Solid-obstacle layer cannot be edited")
            if not layer.deleteFeatures(selected):raise ValueError("QGIS could not delete the selected solid obstacles")
            if not layer.commitChanges():raise ValueError("QGIS could not save the obstacle deletion")
            layer.removeSelection();layer.triggerRepaint()
            self.setup_status.setText(f"Deleted {len(selected)} selected solid obstacle(s).")
        except Exception as exc:
            try:
                if layer is not None and layer.isEditable():layer.rollBack()
            except (NameError,RuntimeError):pass
            self._message("Delete solid obstacles",str(exc),True)

    def _finish_obstacle_shape_capture(self):
        tool=self._obstacle_map_tool
        if tool is None:return
        canvas=None if self.iface is None else getattr(self.iface,"mapCanvas",lambda:None)()
        tool.clear()
        if canvas is not None:
            if self._obstacle_previous_map_tool is not None:canvas.setMapTool(self._obstacle_previous_map_tool)
            else:canvas.unsetMapTool(tool)
        self._obstacle_map_tool=None;self._obstacle_previous_map_tool=None

    def apply_selected_manning(self):
        try:
            layer=self.manning_layer.currentLayer()
            if layer is None or not layer.selectedFeatureIds():raise ValueError("Select one or more roughness polygons")
            if not layer.isEditable() and not layer.startEditing():raise ValueError("Roughness-zone layer cannot be edited")
            fields={field.name() for field in layer.fields()}
            typed={"zone_id","mode","parameter","priority"}<=fields
            legacy={"zone_id","manning_n"}<=fields
            mode=str(self.hydraulic_roughness_mode.currentData())
            if not typed and not legacy:raise ValueError("Selected layer is not a BRAIDS roughness-zone layer")
            if legacy and mode!="manning":raise ValueError("Legacy Manning zones cannot store Chezy C or Nikuradse ks")
            id_index=layer.fields().indexFromName("zone_id")
            value_index=layer.fields().indexFromName("parameter" if typed else "manning_n")
            selected=list(layer.selectedFeatureIds());reserved=[];value=self.manning_zone_value.value()
            for feature_id in selected:
                zone_id=self._manning_identifier(layer,value,selected,reserved);reserved.append(zone_id)
                layer.changeAttributeValue(feature_id,id_index,zone_id)
                layer.changeAttributeValue(feature_id,value_index,value)
                if typed:
                    layer.changeAttributeValue(feature_id,layer.fields().indexFromName("mode"),mode)
                    layer.changeAttributeValue(feature_id,layer.fields().indexFromName("priority"),self.manning_zone_priority.value())
            if not layer.commitChanges():raise ValueError("Could not save roughness-zone values")
            self.setup_status.setText(
                f"Assigned {mode} parameter={value:g}, priority={self.manning_zone_priority.value()} "
                f"to {len(selected)} selected polygon(s).")
        except Exception as exc:self._message("Assign roughness zones",str(exc),True)

    def _assignments(self,domain):
        layer=self.edge_layer.currentLayer();path=self._source_path(layer,"a boundary edge layer")
        assignments=load_assignments(path,domain);inlet_ids={}
        for assignment in assignments:
            if assignment.kind!="inlet":continue
            previous=inlet_ids.setdefault(assignment.bc_id,assignment.edge_id)
            if previous!=assignment.edge_id:
                raise ValueError(f"Inlet ID {assignment.bc_id!r} uses more than one polygon face; keep exactly one face or assign different inlet IDs")
        return assignments

    def refresh_boundary_json(self):
        try:
            domain=self._domain();assignments=self._assignments(domain);kinds={value.bc_id:value.kind for value in assignments}
            try:current=json.loads(self.boundary_json.toPlainText()) if self.boundary_json.toPlainText().strip() else {}
            except json.JSONDecodeError:current={}
            updated={}
            initial_stage=self._initial_stage_value();default_stage=0. if initial_stage is None else initial_stage
            for name in sorted(kinds):
                if name in current:updated[name]=current[name];continue
                if kinds[name]=="inlet":updated[name]={"mode":"inlet_stage","discharge_m3_s":1.0,"stage_m":default_stage,"concentration_kg_m3":0.0,"bed_input_kg_m_s":0.0,"bed_input_kg_s":0.0,"bedload_boundary_mode":"prescribed","sediment_series":None,"hydraulic_series":None,"reversal_policy":"reject"}
                else:updated[name]={"mode":"stage","discharge_m3_s":0.0,"stage_m":default_stage,"concentration_kg_m3":0.0,"bed_input_kg_m_s":0.0,"bed_input_kg_s":None,"bedload_boundary_mode":"prescribed","sediment_series":None,"hydraulic_series":None,"reversal_policy":"reject"}
            self.boundary_json.setPlainText(json.dumps(updated,indent=2));self._rebuild_boundary_cards(updated,kinds);self.refresh_sediment_boundary_json(updated);return self._current_boundary_rows()
        except Exception as exc:self._message("Read BRAIDS boundary IDs",str(exc),True);return None

    def _project_payload(self):
        workspace=self._workspace_path();domain=self._domain();assignments=self._assignments(domain)
        sediment_enabled=self.sediment_enabled.isChecked()
        morfac_series=self._morphological_factor_series()
        boundary_rows=self._resolve_terrain_slopes(self._project_boundary_rows())
        boundaries={name:CutBoundary(**row) for name,row in boundary_rows.items()}
        dem=self._source_path(self.dem_layer.currentLayer(),"a terrain DEM");grid=grid_for_domain(domain,self._base_mesh_size())
        source_layers={"domain":self.domain_layer.currentLayer().source(),"dem":self.dem_layer.currentLayer().source(),"boundaries":self.edge_layer.currentLayer().source()}
        guide_path=self._workspace_path()/"braids_inlet_guides.gpkg"
        guide=next((layer for layer in self._inlet_guidance_layers()
                    if Path(layer.source().split('|',1)[0]).resolve()==guide_path),None)
        if guide is not None:source_layers["inlet_guides"]=guide.source()
        water_layer=self.water_body_layer.currentLayer();water_path=None
        if water_layer is not None:
            water_path=self._source_path(water_layer,"an initial-water polygon layer");source_layers["initial_water_body"]=water_layer.source()
        manning_layer=self.manning_layer.currentLayer();manning_path=None
        if manning_layer is not None:
            manning_path=self._source_path(manning_layer,"a roughness-zone polygon layer");source_layers["roughness_zones"]=manning_layer.source()
        bedrock_layer=self.bedrock_layer.currentLayer() if sediment_enabled else None;bedrock_path=None
        if bedrock_layer is not None:
            bedrock_path=self._source_path(bedrock_layer,"a bedrock-elevation raster");source_layers["bedrock_raster"]=bedrock_layer.source()
        thickness_layer=self.erodible_raster_layer.currentLayer() if sediment_enabled else None;thickness_path=None
        if thickness_layer is not None:
            thickness_path=self._source_path(thickness_layer,"an erodible-thickness raster");source_layers["erodible_thickness_raster"]=thickness_layer.source()
        erodible_layer=self.erodible_zone_layer.currentLayer() if sediment_enabled else None;erodible_path=None
        if erodible_layer is not None:
            erodible_path=self._source_path(erodible_layer,"an erodible-thickness polygon layer");source_layers["erodible_zones"]=erodible_layer.source()
        bank_layer=self.bank_zone_layer.currentLayer() if sediment_enabled else None;bank_path=None
        if bank_layer is not None:
            bank_path=self._source_path(bank_layer,"an erodible-bank polygon layer");source_layers["erodible_bank_zones"]=bank_layer.source()
        obstacle_layer=self.obstacle_layer.currentLayer();obstacle_path=None
        if obstacle_layer is not None:
            obstacle_path=self._source_path(obstacle_layer,"a solid-obstacle polygon layer");source_layers["solid_obstacles"]=obstacle_layer.source()
        fractions=self._sediment_fractions() if sediment_enabled else None
        initial_stage=self._initial_stage_value(required=water_path is not None or self._legacy_lake_seed_xy is not None)
        selection=self._sediment_method_selection()
        secondary_selection=_secondary_project_selection(
            self.secondary_model.currentData())
        backend=self.execution_backend.currentData()
        product_backend=(backend if backend=="cpu" or
            (self.gpu_variant.currentData()==INTEGRATED and
             self.gpu_integrator.currentData()=="implicit_fwave_midpoint") else None)
        if (self.execution_backend.currentData()=="gpu" and
                self.sediment_shear_mode.currentData()!="total"):
            raise ValueError(
                "The current public GPU execution profiles do not yet admit BsSkin. "
                "Select total sediment shear or CPU; no CPU fallback is applied.")
        payload=create_project(workspace=workspace,executable=self._native_executable(),domain=domain,assignments=assignments,
            boundaries=boundaries,dem_path=dem,grid=grid,initial_stage_m=initial_stage,
            terrain_interpretation=self._terrain_settings["interpretation"],
            section_vertical_tolerance_m=self._terrain_settings["section_vertical_tolerance_m"],
            section_dem_path=self._terrain_settings["section_dem_path"],
            lake_seed_xy=None if water_path is not None else self._legacy_lake_seed_xy,
            water_body_path=water_path,
            water_body_layer=WATER_BODY_LAYER_NAME if water_layer is None else self._source_layer_name(water_layer,WATER_BODY_LAYER_NAME),
            erodible_thickness_m=self.erodible.value() if sediment_enabled else 0.,erodible_minimum_depth_m=self.erodible_depth.value(),
            end_time_s=self.end_time.value(),output_interval_s=self.output_interval.value(),dt_max_s=self.dt_max.value(),cfl=self.cfl.value(),threads=self.threads.value(),
            sediment=None if fractions is not None else self._sediment_material(),sediment_fractions=fractions,
            active_layer_thickness_m=self.active_layer_thickness.value() if sediment_enabled else 0.,
            active_layer_mode=self.active_layer_mode.currentData() if sediment_enabled else "constant",
            active_layer_depth_fraction=self.active_layer_depth_fraction.value(),
            active_layer_minimum_thickness_m=self.active_layer_minimum_thickness.value(),
            substrate_layer_count=self.substrate_layer_count.value(),
            substrate_layer_thickness_m=self.substrate_layer_thickness.value(),
            scheme=(6 if self.execution_backend.currentData()=="gpu" else self.RECONSTRUCTION_SCHEME),
            bedload=self._active_bedload_law(),
            exchange=self._active_exchange_mode(),
            engelund_hansen_chezy=self.engelund_hansen_chezy.value(),
            settling_velocity_m_s=self.settling.value(),manning=self.manning.value(),
            hydraulic_roughness_mode=self.hydraulic_roughness_mode.currentData(),
            white_colebrook_chzk2d=self.white_colebrook_chzk2d.isChecked(),
            sediment_shear_mode=(self.sediment_shear_mode.currentData() if sediment_enabled else "total"),
            skin_kinematic_viscosity_m2_s=self.skin_kinematic_viscosity.value(),
            skin_sand_roughness_m=self.skin_sand_roughness.value(),
            skin_silt_roughness_m=self.skin_silt_roughness.value(),
            skin_critical_mud_fraction_low=self.skin_mud_low.value(),
            skin_critical_mud_fraction_high=self.skin_mud_high.value(),
            skin_fluff_fraction=self.skin_fluff_fraction.value(),
            horizontal_viscosity_m2_s=self.horizontal_viscosity.value(),
            horizontal_viscosity_formulation=self.horizontal_viscosity_formulation.currentData(),
            bed_slope=selection.bed_slope,
            mpm_cf=self.mpm_cf.value(),mpm_critical_shields=self.mpm_critical.value(),mpm_friction_mode=self.mpm_friction_mode.currentData(),
            hiding_exposure=self.hiding_exposure.currentData() if sediment_enabled else 0,
            hiding_parker_alpha=self.hiding_parker_alpha.value(),hiding_wu_exponent=self.hiding_wu_exponent.value(),
            critical_mud_fraction=self.critical_mud_fraction.value(),transport_calibration_factor=self.transport_calibration.value(),
            settling_mode=self.settling_mode.currentData() if sediment_enabled else 0,
            hindered_settling_reference_concentration_kg_m3=self.hindered_settling_reference.value(),
            bed_slope_alfabs=self.bed_slope_alfabs.value(),bed_slope_alfabn=self.bed_slope_alfabn.value(),
            bed_friction_angle_deg=self.bed_friction_angle.value(),
            secondary_flow_enabled=self.secondary_flow.isChecked(),
            secondary_flow_model=secondary_selection["model"],
            secondary_flow_method_id=secondary_selection["method_id"],
            secondary_flow_method_provenance=secondary_selection[
                "method_provenance"],
            secondary_flow_transport_direction=(
                True if self.secondary_model.currentData() in (
                    _SECONDARY_EQUILIBRIUM,
                    _SECONDARY_PROGNOSTIC_CLASSIC) else
                self.secondary_transport_direction.isChecked()),
            secondary_flow_momentum_beta=(
                self.secondary_beta.value()
                if self.secondary_model.currentData()==
                    _SECONDARY_PROGNOSTIC_LEGACY else 0.),
            secondary_flow_classic_betac=0.,
            secondary_flow_scalar_diffusivity_m2_s=0.,
            secondary_flow_espir=self.secondary_flow_espir.value(),
            bedload_adaptation_enabled=sediment_enabled and self.bedload_adaptation.isChecked(),
            bedload_adaptation_length_diameters=self.bedload_adaptation_length.value(),
            morphological_acceleration_enabled=sediment_enabled and self.morphological_acceleration.isChecked(),
            morphological_factor=self.morphological_factor.value(),
            morphological_factor_series=morfac_series,
            bedform_roughness_enabled=sediment_enabled and self.bedform_roughness.isChecked(),
            minimum_boundary_cell_area_fraction=self._minimum_boundary_cell_area_fraction,
            manning_zones_path=manning_path,
            manning_zones_layer=ROUGHNESS_LAYER_NAME if manning_layer is None else self._source_layer_name(manning_layer,ROUGHNESS_LAYER_NAME),
            bedrock_raster_path=bedrock_path,erodible_thickness_raster_path=thickness_path,
            erodible_zones_path=erodible_path,
            erodible_zones_layer=ERODIBLE_LAYER_NAME if erodible_layer is None else self._source_layer_name(erodible_layer,ERODIBLE_LAYER_NAME),
            bank_morphology_enabled=sediment_enabled and self.bank_morphology_enabled.isChecked(),
            bank_zones_path=bank_path,
            bank_zones_layer=BANK_LAYER_NAME if bank_layer is None else self._source_layer_name(bank_layer,BANK_LAYER_NAME),
            bank_erosion_fraction=self.bank_erosion_fraction.value(),bank_avalanche=self.bank_avalanche.isChecked(),
            bank_wet_slope=self.bank_wet_slope.value(),bank_dry_slope=self.bank_dry_slope.value(),
            bank_avalanche_time_s=self.bank_avalanche_time.value(),
            bank_relaxation_length_m=self.bank_relaxation_length.value(),
            bank_wet_depth_switch_m=self.bank_wet_depth_switch.value(),
            bank_failure_enabled=sediment_enabled and self.bank_failure_enabled.isChecked(),
            bank_toe_erodibility_m_pa_s=self.bank_toe_erodibility.value(),
            bank_toe_critical_shear_pa=self.bank_toe_critical_shear.value(),
            bank_toe_exponent=self.bank_toe_exponent.value(),
            bank_critical_undercut_m=self.bank_critical_undercut.value(),
            bank_toe_erosion_mode=self.bank_toe_mode.currentData(),
            bank_toe_capacity_fraction=self.bank_toe_capacity_fraction.value(),
            bank_distributed_transport_fraction=self.bank_distributed_transport_fraction.value(),
            bank_distributed_critical_slope=self.bank_distributed_critical_slope.value(),
            general_avalanche_enabled=sediment_enabled and self.general_avalanche.isChecked(),
            general_avalanche_wet_slope=self.general_avalanche_wet_slope.value(),
            general_avalanche_dry_slope=self.general_avalanche_dry_slope.value(),
            general_avalanche_time_s=self.general_avalanche_time.value(),
            general_avalanche_relaxation_length_m=self.general_avalanche_relaxation_length.value(),
            general_avalanche_wet_depth_switch_m=self.general_avalanche_wet_depth_switch.value(),
            solid_obstacles_path=obstacle_path,
            solid_obstacles_layer=OBSTACLE_LAYER_NAME if obstacle_layer is None else self._source_layer_name(obstacle_layer,OBSTACLE_LAYER_NAME),
            compact_sidecars=self.execution_backend.currentData()=="gpu",
            source_layers=source_layers,sediment_enabled=sediment_enabled,amr=self._amr_settings(),
            refinement_zones_path=(None if self.refinement_layer.currentLayer() is None else
                self._source_path(self.refinement_layer.currentLayer(),'a refinement polygon layer')),
            refinement_zones_layer=(REFINEMENT_LAYER_NAME if self.refinement_layer.currentLayer() is None else
                self._source_layer_name(self.refinement_layer.currentLayer(),REFINEMENT_LAYER_NAME)),
            execution_backend=product_backend,
            morphology_output_settings=self._morphology_output_settings(),
            apply_morphology_output_defaults=self.morphology_output_defaults.isChecked())
        payload["numerics"]["sediment_method_selection"]=selection.as_dict()
        return payload

    def _base_mesh_size(self):
        if not self.amr_enabled.isChecked():return self.spacing.value()
        return self.spacing.value()*AMR_RATIO**(self.amr_levels.value()-1)

    def _update_amr_sizes(self,*_):
        enabled=self.amr_enabled.isChecked();levels=self.amr_levels.value();ratio=AMR_RATIO
        self.amr_levels.setEnabled(enabled)
        self.amr_subcycling.setEnabled(enabled)
        minimum=self.spacing.value();finest=levels-1
        if enabled:
            self.amr_sizes.setText("   ·   ".join(
                f"L{level}: {minimum*ratio**(finest-level):g} m" for level in range(levels))+
                f"   ·   inlet/outlet boundaries fixed at L{finest} ({minimum:g} m)")
        else:self.amr_sizes.setText(f"Fixed mesh: {minimum:g} m")
        if hasattr(self,'refinement_level'):
            selected=self.refinement_level.currentData()
            if selected is None:selected=finest
            self.refinement_level.blockSignals(True);self.refinement_level.clear()
            for level in range(levels):self.refinement_level.addItem(f'L{level}',level)
            self.refinement_level.setCurrentIndex(self.refinement_level.findData(min(int(selected),finest)))
            self.refinement_level.setEnabled(enabled);self.refinement_level.blockSignals(False)

    def _amr_settings(self):
        return AMRSettings(enabled=self.amr_enabled.isChecked(),
            levels=self.amr_levels.value(),refinement_ratio=AMR_RATIO,
            subcycling=self.amr_subcycling.isChecked(),
            regrid_interval_s=self.amr_interval.value(),coarsen_delay=self.amr_delay.value(),
            buffer_cells=self.amr_buffer.value(),
            **{key:widget.isChecked() for key,widget in self.amr_switches.items()},
            **{key:widget.value() for key,widget in self.amr_thresholds.items()})

    def save_project_ui(self):
        try:
            if self.execution_backend.currentData()=="gpu":
                target=Path(self.workspace.text()).expanduser().resolve()/"braids_project.json"
                if target.is_file():reject_raw_gpu_persistent_front(target)
            payload=self._project_payload();path=Path(payload["workspace"])/"braids_project.json"
            path=save_project(path,payload);self.project_path.setText(str(path));self._refresh_input_summary(payload);QgsProject.instance().writeEntry("BRAIDS",self.PROJECT_KEY,str(path));self.setup_status.setText(f"Saved {path}");return path
        except Exception as exc:self._message("Save BRAIDS project",str(exc),True);return None

    def load_project_ui(self):
        path=self.project_path.text().strip()
        if not path:
            path,_=QFileDialog.getOpenFileName(self,"Open BRAIDS project",self.workspace.text(),"BRAIDS project (*.json)")
        if not path:return
        try:
            if self.execution_backend.currentData()=="gpu":
                reject_raw_gpu_persistent_front(path)
            payload=load_project(path);resolved_path=Path(path).resolve();self.project_path.setText(str(resolved_path));self.workspace.setText(str(resolved_path.parent))
            product=payload.get("qgis_product_v109")
            if isinstance(product,dict):
                backend=("gpu" if product.get("backend")==
                         "implicit_fwave_midpoint_gpu" else "cpu")
                index=self.execution_backend.findData(backend)
                if index>=0:self.execution_backend.setCurrentIndex(index)
                if backend=="gpu":
                    variant=self.gpu_variant.findData(INTEGRATED)
                    if variant>=0:self.gpu_variant.setCurrentIndex(variant)
                    integrator=self.gpu_integrator.findData("implicit_fwave_midpoint")
                    if integrator>=0:self.gpu_integrator.setCurrentIndex(integrator)
                self._restore_morphology_outputs(payload)
            terrain=payload.get("terrain",{})
            self._terrain_settings=dict(interpretation=terrain.get("interpretation","bilinear_surface"),
                section_vertical_tolerance_m=terrain.get("section_vertical_tolerance_m",0.),
                section_dem_path=terrain.get("section_dem_path"))
            for compatibility_note in payload.get("compatibility_notes",[]):
                self.log.appendPlainText(compatibility_note)
            initial=payload["initial"];numerics=payload["numerics"];sediment=payload["sediment"]
            self._minimum_boundary_cell_area_fraction=numerics.get("minimum_boundary_cell_area_fraction",0.)
            self.initial_stage.setValue(self.initial_stage.minimum() if initial["stage_m"] is None else initial["stage_m"]);self.erodible.setValue(initial["erodible_thickness_m"]);self.erodible_depth.setValue(initial["erodible_minimum_depth_m"])
            self.active_layer_thickness.setValue(initial.get("active_layer_thickness_m",0.))
            self.active_layer_mode.setCurrentIndex(max(0,self.active_layer_mode.findData(initial.get("active_layer_mode","constant"))))
            self.active_layer_depth_fraction.setValue(initial.get("active_layer_depth_fraction",.008))
            self.active_layer_minimum_thickness.setValue(initial.get("active_layer_minimum_thickness_m",max(initial.get("active_layer_thickness_m",0.),.002)))
            self.substrate_layer_count.setValue(initial.get("substrate_layer_count",1));self.substrate_layer_thickness.setValue(initial.get("substrate_layer_thickness_m",max(initial.get("active_layer_thickness_m",0.),.1)))
            self.water_body_layer.setLayer(None);self._legacy_lake_seed_xy=(initial.get("lake_seed_xy") if initial.get("water_body_path") is None else None)
            adaptive=AMRSettings(**numerics.get('amr',{}));self.amr_levels.setValue(adaptive.levels);self.amr_enabled.setChecked(adaptive.enabled)
            self.amr_subcycling.setChecked(adaptive.subcycling)
            minimum=payload["grid"]["dx_m"]/(adaptive.refinement_ratio**(adaptive.levels-1) if adaptive.enabled else 1)
            roughness=payload.get("roughness",{})
            self.hydraulic_roughness_mode.setCurrentIndex(max(0,self.hydraulic_roughness_mode.findData(
                roughness.get("mode","manning"))))
            self.white_colebrook_chzk2d.setChecked(roughness.get("white_colebrook_chzk2d",False))
            self.spacing.setValue(minimum);self.end_time.setValue(numerics["end_time_s"]);self.output_interval.setValue(numerics["output_interval_s"]);self.dt_max.setValue(numerics["dt_max_s"]);self.cfl.setValue(numerics.get("cfl",.45));self.threads.setValue(numerics["threads"]);self.manning.setValue(roughness.get("default_parameter",roughness.get("default_manning_n",numerics["manning"])));self.settling.setValue(numerics["settling_velocity_m_s"])
            self.amr_interval.setValue(adaptive.regrid_interval_s);self.amr_delay.setValue(adaptive.coarsen_delay);self.amr_buffer.setValue(adaptive.buffer_cells)
            for key,widget in self.amr_switches.items():widget.setChecked(getattr(adaptive,key))
            for key,widget in self.amr_thresholds.items():widget.setValue(getattr(adaptive,key))
            self.grain_density.setValue(sediment["grain_density_kg_m3"]);self.water_density.setValue(sediment["water_density_kg_m3"]);self.porosity.setValue(sediment["porosity"]);self.d50_mm.setValue(sediment["d50_m"]*1000);self.d90_mm.setValue(sediment["d90_m"]*1000);self.z0_mm.setValue(sediment["z0_m"]*1000);self.exchange_depth.setValue(sediment["minimum_exchange_depth_m"]);self.minimum_bedload_depth.setValue(sediment["minimum_bedload_depth_m"])
            self.reference_height_factor.setValue(sediment.get("reference_height_factor",1.))
            self.sediment_roughness.setCurrentIndex(self.sediment_roughness.findData(sediment.get("roughness_mode","fixed")))
            self._set_fraction_rows(payload.get("sediment_fractions",[]))
            loaded_bedload=numerics.get("bedload",0)
            loaded_exchange=numerics.get("exchange",0)
            self.bedload_enabled.setChecked(loaded_bedload != 0)
            self.suspended_enabled.setChecked(loaded_exchange != 0)
            for combo,value,preferred in ((self.bedload,loaded_bedload,2),
                                          (self.exchange,loaded_exchange,2)):
                index=combo.findData(value if value != 0 else preferred)
                combo.setCurrentIndex(max(0,index))
            self.mpm_cf.setValue(numerics.get("mpm_cf",.01));self.mpm_friction_mode.setCurrentIndex(self.mpm_friction_mode.findData(numerics.get("mpm_friction_mode",0)));self.mpm_critical.setValue(numerics.get("mpm_critical_shields",0.));self.engelund_hansen_chezy.setValue(numerics.get("engelund_hansen_chezy",28.8));self.bed_slope_alfabs.setValue(numerics.get("bed_slope_alfabs",1.));self.bed_slope_alfabn.setValue(numerics.get("bed_slope_alfabn",1.5));self.bed_friction_angle.setValue(numerics.get("bed_friction_angle_deg",30.))
            self.settling_mode.setCurrentIndex(self.settling_mode.findData(numerics.get("settling_mode",0)));self.hindered_settling_reference.setValue(numerics.get("hindered_settling_reference_concentration_kg_m3",DEFAULT_HINDERED_REFERENCE_CONCENTRATION_KG_M3))
            self.horizontal_viscosity.setValue(numerics.get('horizontal_viscosity_m2_s',0.))
            self.horizontal_viscosity_formulation.setCurrentIndex(self.horizontal_viscosity_formulation.findData(numerics.get('horizontal_viscosity_formulation','depth_conservative')))
            self.critical_mud_fraction.setValue(numerics.get("critical_mud_fraction",.5));self.transport_calibration.setValue(numerics.get("transport_calibration_factor",1.))
            shear=payload.get("sediment_shear",{})
            self.sediment_shear_mode.setCurrentIndex(max(0,self.sediment_shear_mode.findData(shear.get("mode","total"))))
            self.skin_kinematic_viscosity.setValue(shear.get("kinematic_viscosity_m2_s",1.e-6))
            self.skin_sand_roughness.setValue(shear.get("sand_roughness_m",.003))
            self.skin_silt_roughness.setValue(shear.get("silt_roughness_m",.0003))
            self.skin_mud_low.setValue(shear.get("critical_mud_fraction_low",.2))
            self.skin_mud_high.setValue(shear.get("critical_mud_fraction_high",.8))
            self.skin_fluff_fraction.setValue(shear.get("fluff_fraction",0.))
            # New projects deliberately start with every sediment process OFF,
            # but a saved project must restore its master switch before any
            # component flags.  Otherwise the compatibility refreshes below
            # correctly see sediment as disabled and destructively clear saved
            # bank, avalanche, secondary-flow and bedform selections.
            self.sediment_enabled.setChecked(
                numerics.get("sediment_enabled",True))
            self._update_sediment_controls()
            bank=payload.get("internal_conditions",{}).get("bank_morphology",{})
            self.bank_morphology_enabled.setChecked(bank.get("enabled",False));self.bank_erosion_fraction.setValue(bank.get("erosion_supply_fraction",.5))
            self.bank_avalanche.setChecked(bank.get("avalanche_enabled",False));self.bank_wet_slope.setValue(bank.get("wet_maximum_slope",10.))
            self.bank_dry_slope.setValue(bank.get("dry_maximum_slope",1.));self.bank_avalanche_time.setValue(bank.get("avalanche_time_s",86400.))
            self.bank_relaxation_length.setValue(bank.get("relaxation_length_m",1.))
            self.bank_wet_depth_switch.setValue(bank.get("wet_depth_switch_m",.1))
            bank_failure=payload.get("internal_conditions",{}).get("bank_failure",{})
            self.bank_toe_mode.setCurrentIndex(max(0,self.bank_toe_mode.findData(
                bank_failure.get("toe_erosion_mode","cohesive_excess_shear"))))
            self.bank_toe_capacity_fraction.setValue(bank_failure.get("toe_capacity_fraction",1.))
            self.bank_distributed_transport_fraction.setValue(
                bank_failure.get("distributed_transport_fraction",.1))
            self.bank_distributed_critical_slope.setValue(
                bank_failure.get("distributed_critical_slope",.35))
            self.bank_toe_erodibility.setValue(bank_failure.get("toe_erodibility_m_pa_s",1.3e-6))
            self.bank_toe_critical_shear.setValue(bank_failure.get("toe_critical_shear_pa",32.5))
            self.bank_toe_exponent.setValue(bank_failure.get("toe_exponent",1.))
            self.bank_critical_undercut.setValue(bank_failure.get("critical_undercut_m",.1))
            self.bank_failure_enabled.setChecked(bank_failure.get("enabled",False))
            general=payload.get("internal_conditions",{}).get("general_avalanche",{})
            self.general_avalanche_wet_slope.setValue(general.get("wet_maximum_slope",10.))
            self.general_avalanche_dry_slope.setValue(general.get("dry_maximum_slope",1.))
            self.general_avalanche_time.setValue(general.get("avalanche_time_s",86400.))
            self.general_avalanche_relaxation_length.setValue(general.get("relaxation_length_m",1.))
            self.general_avalanche_wet_depth_switch.setValue(general.get("wet_depth_switch_m",.1))
            # Loading validated settings is not a user's mutually-exclusive
            # toggle: an inactive legacy bank-avalanche flag may remain ON.
            blocked=self.general_avalanche.blockSignals(True)
            self.general_avalanche.setChecked(general.get("enabled",False))
            self.general_avalanche.blockSignals(blocked)
            secondary=payload.get("internal_conditions",{}).get("secondary_flow",{})
            self.secondary_flow.setChecked(False)
            secondary_model=_secondary_ui_selection(secondary)
            self.secondary_model.setCurrentIndex(
                self.secondary_model.findData(secondary_model))
            self.secondary_transport_direction.setChecked(secondary.get("transport_direction",True))
            self.secondary_beta.setValue(
                secondary.get("momentum_beta",0.)
                if secondary_model==_SECONDARY_PROGNOSTIC_LEGACY else 0.)
            self.secondary_flow_espir.setValue(secondary.get("espir",1.))
            self.secondary_flow.setChecked(secondary.get("enabled",False))
            adaptation=payload.get("internal_conditions",{}).get("bedload_adaptation",{})
            self.bedload_adaptation_length.setValue(adaptation.get("length_diameters",100.))
            self.bedload_adaptation.setChecked(adaptation.get("enabled",False))
            acceleration=payload.get("internal_conditions",{}).get("morphological_acceleration",{})
            self.morphological_factor.setValue(acceleration.get("constant_factor",2.))
            series=acceleration.get("series")
            self.morphological_factor_schedule.setText("" if series is None else ", ".join(
                f"{time_s:g}:{factor:g}" for time_s,factor in zip(series["times_s"],series["factors"])))
            self.morphological_acceleration.setChecked(acceleration.get("enabled",False))
            bedform=payload.get("internal_conditions",{}).get("bedform_roughness",{})
            if (bedform.get("enabled",False) and adaptive.enabled and adaptive.subcycling and
                    bank_failure.get("enabled",False) and
                    bank_failure.get("toe_erosion_mode") in (
                        "noncohesive_transport_capacity","distributed_slope_transport") and
                    general.get("enabled",False)):
                self.gpu_amr_evolving_banks.setChecked(True)
            self.bedform_roughness.setChecked(bedform.get("enabled",False))
            self.hiding_exposure.setCurrentIndex(self.hiding_exposure.findData(numerics.get("hiding_exposure",0)))
            self.hiding_parker_alpha.setValue(numerics.get("hiding_parker_alpha",.6));self.hiding_wu_exponent.setValue(numerics.get("hiding_wu_exponent",-.6))
            self.bed_slope.setChecked(numerics["bed_slope"] in (2,3,5))
            provenance=numerics.get("sediment_method_selection",{})
            automatic=(provenance.get("selection_origin")=="automatic" and provenance.get("bed_slope")==numerics["bed_slope"])
            blocked=self.bed_slope_method.blockSignals(True)
            self.bed_slope_method.setCurrentIndex(self.bed_slope_method.findData(None if automatic or numerics["bed_slope"]==0 else numerics["bed_slope"]))
            self.bed_slope_method.blockSignals(blocked)
            self._slope_parameter_mode=numerics["bed_slope"] if numerics["bed_slope"] else (3 if numerics["bedload"]==8 else 2)
            self._update_sediment_controls()
            self.boundary_json.setPlainText(json.dumps(payload["boundaries"],indent=2));self._restore_source_layers(payload)
            kinds={row["bc_id"]:row["kind"] for row in payload["assignments"]};self._rebuild_boundary_cards(payload["boundaries"],kinds);self.refresh_sediment_boundary_json(payload["boundaries"]);self._restore_morphology_outputs(payload);self._refresh_input_summary(payload)
            QgsProject.instance().writeEntry("BRAIDS",self.PROJECT_KEY,str(Path(path).resolve()));self.setup_status.setText("BRAIDS project and its recorded source layers are loaded.");self._refresh_results_runs()
        except Exception as exc:self._message("Load BRAIDS project",str(exc),True)

    def _restore_source_layers(self,payload):
        """Find or reopen the file-backed layers recorded by a BRAIDS project."""
        sources=dict(payload.get("source_layers",{}))
        if sources.get("manning_zones") and not sources.get("roughness_zones"):
            # Project-v1 compatibility: retain the original datasource while
            # exposing a generic public source key after load.
            sources["roughness_zones"]=sources["manning_zones"]
        refinement=payload.get('mesh_refinement',{})
        self.refinement_layer.setLayer(None)
        if refinement.get('zones_path'):
            sources['refinement_zones']=refinement['zones_path']+'|layername='+refinement.get('zones_layer',REFINEMENT_LAYER_NAME)
        if payload["initial"].get("water_body_path") and not sources.get("initial_water_body"):
            sources["initial_water_body"]=payload["initial"]["water_body_path"]
        roughness=payload.get("roughness",{})
        bed=payload.get("bed_inputs",{})
        internal=payload.get("internal_conditions",{})
        for key,path,layer_name in (
                ("roughness_zones",roughness.get("zones_path"),roughness.get("zones_layer",MANNING_LAYER_NAME)),
                ("erodible_zones",bed.get("zones_path"),bed.get("zones_layer",ERODIBLE_LAYER_NAME)),
                ("erodible_bank_zones",internal.get("bank_failure",{}).get("zones_path") or
                 internal.get("bank_morphology",{}).get("zones_path"),
                 internal.get("bank_failure",{}).get("zones_layer",BANK_LAYER_NAME)),
                ("solid_obstacles",internal.get("solid_obstacles",{}).get("path"),
                 internal.get("solid_obstacles",{}).get("layer",OBSTACLE_LAYER_NAME))):
            if path and not sources.get(key):
                sources[key]=str(path)+"|layername="+str(layer_name)
        for key,path in (("bedrock_raster",bed.get("bedrock_raster_path")),
                         ("erodible_thickness_raster",bed.get("thickness_raster_path"))):
            if path and not sources.get(key):sources[key]=str(path)
        rows=(("domain",self.domain_layer,"BRAIDS domain","vector"),
              ("dem",self.dem_layer,"BRAIDS terrain","raster"),
              ("boundaries",self.edge_layer,"BRAIDS boundary edges","vector"),
              ("initial_water_body",self.water_body_layer,"BRAIDS initial water body","vector"),
              ("roughness_zones",self.manning_layer,"BRAIDS roughness zones","vector"),
              ('refinement_zones',self.refinement_layer,'BRAIDS refinement zones','vector'),
              ("bedrock_raster",self.bedrock_layer,"BRAIDS bedrock elevation","raster"),
              ("erodible_thickness_raster",self.erodible_raster_layer,"BRAIDS erodible thickness","raster"),
              ("erodible_zones",self.erodible_zone_layer,"BRAIDS erodible thickness zones","vector"),
              ("erodible_bank_zones",self.bank_zone_layer,"BRAIDS erodible bank zones","vector"),
              ("solid_obstacles",self.obstacle_layer,"BRAIDS solid reflective obstacles","vector"))
        for key,combo,name,kind in rows:
            source=sources.get(key)
            if not source:
                if key not in ("domain","dem","boundaries"):combo.setLayer(None)
                continue
            normalized=str(Path(source.split("|",1)[0]).expanduser().resolve())
            requested_layer=next((part.split('=',1)[1] for part in source.split('|')[1:]
                                  if part.startswith('layername=')),None)
            layer=next((item for item in QgsProject.instance().mapLayers().values()
                        if str(Path(item.source().split("|",1)[0]).expanduser().resolve())==normalized
                        and (requested_layer is None or self._source_layer_name(item,'')==requested_layer)),None)
            if layer is None:
                layer=QgsRasterLayer(source,name) if kind=="raster" else QgsVectorLayer(source,name,"ogr")
                if not layer.isValid():raise ValueError(f"Could not reopen recorded {key} layer: {source}")
                QgsProject.instance().addMapLayer(layer)
            combo.setLayer(layer)
            if key=="boundaries":self._style_boundary_layer(layer)
            if key=="erodible_bank_zones":
                expression='concat(\'Erodible bank: \', "zone_id")';layer.setDisplayExpression(expression)
                self._style_polygon_layer(layer,"194,24,91,65","136,14,79,255",expression)
            if key=="solid_obstacles":
                expression=('concat(\'Solid obstacle: \', "obstacle_id")'
                            if layer.fields().indexFromName("obstacle_id")>=0 else "")
                if expression:layer.setDisplayExpression(expression)
                self._style_polygon_layer(layer,"70,70,70,150","20,20,20,255",expression or None)
        domain_layer=self.domain_layer.currentLayer()
        if domain_layer is not None:domain_layer.setCustomProperty("braids/domain_id",payload["domain"]["domain_id"])
        source=sources.get("inlet_guides")
        if source:
            guide=next((layer for layer in self._inlet_guidance_layers() if layer.source()==source),None)
            if guide is None:
                guide=QgsVectorLayer(source,"Inlet Guidance Lines","ogr")
                if guide.isValid():
                    self._style_inlet_guides(guide);QgsProject.instance().addMapLayer(guide)
            self._hide_inlet_guides()

    def _restore_project_reference(self):
        backend,backend_ok=QgsProject.instance().readEntry("BRAIDS",self.BACKEND_KEY,"cpu")
        if backend_ok and backend=="gpu":
            self.execution_backend.setCurrentIndex(self.execution_backend.findData("gpu"))
        path,ok=QgsProject.instance().readEntry("BRAIDS",self.PROJECT_KEY,"")
        if not (ok and path):path,ok=QgsProject.instance().readEntry("BRAIDS",self.PROJECT_KEY,"")
        if ok and path:
            resolved=Path(path).expanduser().resolve();self.project_path.setText(str(resolved));self.workspace.setText(str(resolved.parent))
            if resolved.is_file():self.load_project_ui()
        run,ok=QgsProject.instance().readEntry("BRAIDS",self.RUN_KEY,"")
        if not (ok and run):run,ok=QgsProject.instance().readEntry("BRAIDS",self.RUN_KEY,"")
        if ok and run and Path(run).is_dir():
            self.run_root=Path(run);self.run_status.setText(f"Previous run: {run}")
            prepared=self.run_root/"gpu_prepared"
            if (prepared/"preparation_receipt.json").is_file():
                self._gpu_prepared_root=prepared
                self._sync_prepared_gpu_selector()
            self._load_inlet_diagnostics();self._load_run_log()
            self._backend_changed(persist=False)
        self._refresh_results_runs(self.run_root)

    def _completed_project_runs(self):
        text=self.workspace.text().strip()
        workspace=Path(text).expanduser() if text else None
        markers=[]
        if workspace is not None:
            if (workspace/RUN_MARKER).is_file():markers.append(workspace/RUN_MARKER)
            markers.extend((workspace/"braids_runs").glob(f"*/{RUN_MARKER}"))
            markers.extend(workspace.glob(f"*/{RUN_MARKER}"))
        rows=[];seen=set()
        for marker_path in markers:
            root=marker_path.parent.resolve()
            if root in seen:continue
            seen.add(root)
            try:marker=json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError,json.JSONDecodeError):continue
            if marker.get("status")!="completed" or not ((root/"frames.txt").is_file() or (root/"live_preview"/"frames.csv").is_file()):continue
            rows.append((str(marker.get("updated_at") or marker.get("created_at") or ""),root))
        gpu_roots=list(self._external_result_run_roots)
        if workspace is not None:
            gpu_roots.extend([workspace,*(workspace/"braids_runs").glob("*"),
                              *workspace.glob("*")])
        for root in gpu_roots:
            root=filesystem_path(root).resolve()
            if root in seen or not self._completed_gpu_run(root):continue
            seen.add(root)
            modified=datetime.fromtimestamp((root/"run.json").stat().st_mtime,
                                            timezone.utc).isoformat()
            rows.append((modified,root))
        return [root for _,root in sorted(rows,key=lambda row:(row[0],str(row[1])),reverse=True)]

    @staticmethod
    def _completed_gpu_run(root):
        try:
            run=json.loads((root/"run.json").read_text(encoding="utf-8"))
            status=json.loads((root/"status.json").read_text(encoding="utf-8"))
            fixed=(run.get("schema")==1 and (root/"geometry.npz").is_file())
            adaptive=(run.get("schema")==2 and
                run.get("format")=="braids-gpu-amr-run-v1" and
                run.get("input_kind")=="amr-prepared")
            on_demand=(run.get("schema")==3 and
                run.get("format")=="braids-gpu-amr-run-v2" and
                run.get("input_kind") in ("amr-on-demand", "amr-on-demand-general"))
            return (run.get("backend")=="gpu" and
                    run.get("status")==status.get("status")=="completed" and
                    bool(run.get("frames")) and (fixed or adaptive or on_demand))
        except (OSError,ValueError,TypeError,json.JSONDecodeError):
            return False

    def open_result_run_folder(self):
        initial=(str(self.result_run_root) if self.result_run_root is not None else
                 self.workspace.text())
        selected=QFileDialog.getExistingDirectory(self,"Open completed BRAIDS run",initial)
        if not selected:return
        root=filesystem_path(selected).resolve()
        if not self._completed_gpu_run(root):
            try:marker=json.loads((root/RUN_MARKER).read_text(encoding="utf-8"))
            except (OSError,ValueError,json.JSONDecodeError):marker={}
            native=(marker.get("status")=="completed" and
                    ((root/"frames.txt").is_file() or (root/"live_preview"/"frames.csv").is_file()))
            if not native:
                self._message("Open BRAIDS results","Select a completed native or GPU BRAIDS run folder.",True)
                return
        if root not in self._external_result_run_roots:
            self._external_result_run_roots.append(root)
        self._refresh_results_runs(root)

    def _refresh_results_runs(self,select=None):
        if not hasattr(self,"results_run_combo"):return
        if isinstance(select,bool):select=None
        selected=(filesystem_path(select).resolve() if select is not None
                  else self.result_run_root)
        rows=self._completed_project_runs()
        workspace=Path(self.workspace.text()).expanduser().resolve() if self.workspace.text().strip() else None
        if (selected is not None and selected.is_dir() and selected not in rows and workspace is not None
                and selected.is_relative_to(workspace)):
            try:marker=json.loads((selected/RUN_MARKER).read_text(encoding="utf-8"))
            except (OSError,json.JSONDecodeError):marker={}
            if marker.get("status")=="completed" and ((selected/"frames.txt").is_file() or (selected/"live_preview"/"frames.csv").is_file()):rows.insert(0,selected)
        if selected is not None and selected not in rows and self._completed_gpu_run(selected):
            rows.insert(0,selected)
        self.results_run_combo.blockSignals(True);self.results_run_combo.clear()
        for root in rows:self.results_run_combo.addItem(root.name,str(root))
        index=self.results_run_combo.findData(str(selected)) if selected is not None else 0
        self.results_run_combo.setCurrentIndex(max(0,index))
        self.results_run_combo.blockSignals(False);self._result_run_changed()

    def _result_run_changed(self,*_):
        value=self.results_run_combo.currentData() if hasattr(self,"results_run_combo") else None
        self.result_run_root=(None if not value else
                              filesystem_path(value).resolve());self._manifest=None
        self.show_result_mesh.blockSignals(True);self.show_result_mesh.setChecked(False);self.show_result_mesh.setEnabled(False);self.show_result_mesh.blockSignals(False)
        self.animate_result_mesh.blockSignals(True);self.animate_result_mesh.setChecked(False);self.animate_result_mesh.setEnabled(False);self.animate_result_mesh.blockSignals(False)
        self.result_mesh_progress.setVisible(False);self.result_mesh_progress.setValue(0);self.result_mesh_progress.setFormat("Adaptive mesh: %p%")
        self.result_variable.clear();self.frame_combo.clear();self._result_times=[];self._reset_result_player()
        self.result_boundary_id.clear()
        if self.result_run_root is None:
            self.results_status.setText("No completed runs were found in this BRAIDS project folder.");return
        manifest_path=self.result_run_root/"qgis_results"/"results.json"
        if manifest_path.is_file():
            try:self._set_results_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
            except (OSError,json.JSONDecodeError,KeyError,ValueError):self.results_status.setText(f"Selected run: {self.result_run_root.name}. Prepare its raster results.")
        else:self.results_status.setText(f"Selected run: {self.result_run_root.name}. Prepare its raster results.")

    def _load_inlet_diagnostics(self):
        if self.execution_backend.currentData()=="gpu":
            self.run_button.setEnabled(self._gpu_prepared_root is not None and
                                       not self._prepared_controls_dirty)
            return
        ready=False
        if self.run_root is not None and (self.run_root/DIAGNOSTICS_FILE).is_file():
            try:ready=not load_inlet_diagnostics(self.run_root)["errors"]
            except (OSError,ValueError,KeyError,TypeError):ready=False
        self.run_button.setEnabled(ready and not self._prepared_controls_dirty)

    def _start_preparation_task(self,button,create_task,completed,failed,progress_bar=None):
        """Restore the initiating action on submission, worker or callback failure."""
        if button in self._preparation_tasks:return
        button.setEnabled(False);task=None
        def finish(succeeded):
            if self._preparation_tasks.get(button) is not task:return
            self._preparation_tasks.pop(button,None);button.setEnabled(True)
            try:
                if succeeded:completed(task)
                else:failed(task)
            except Exception as exc:
                failed(SimpleNamespace(error=exc,run_root=getattr(task,"run_root",None)))
        try:
            task=create_task();self._task=task;self._preparation_tasks[button]=task
            target_progress=self.progress if progress_bar is None else progress_bar
            task.progressChanged.connect(lambda value:target_progress.setValue(round(value)))
            task.taskCompleted.connect(lambda:finish(True))
            task.taskTerminated.connect(lambda:finish(False))
            if not QgsApplication.taskManager().addTask(task):
                raise RuntimeError("QGIS could not queue the preparation task. Please try again.")
        except Exception as exc:
            self._preparation_tasks.pop(button,None);button.setEnabled(True)
            failed(SimpleNamespace(error=exc,run_root=getattr(task,"run_root",None)))

    def prepare_run_ui(self):
        if self.runner.is_running or self.gpu_runner.is_running or self._preparation_tasks.get(self.prepare_run_button):
            self._message("Prepare BRAIDS run","A solver or run preparation is already active.",True);return
        gpu=self.execution_backend.currentData()=="gpu"
        config=None
        if gpu:
            try:
                variant=self.gpu_variant.currentData()
                if self.amr_enabled.isChecked() and variant not in (
                        _ON_DEMAND_AMR_VARIANT,_ON_DEMAND_GENERAL_AMR_VARIANT,INTEGRATED):
                    raise ValueError("GPU AMR requires a matching on-demand AMR numerical variant")
                if variant in (_ON_DEMAND_AMR_VARIANT,_ON_DEMAND_GENERAL_AMR_VARIANT) and not self.amr_enabled.isChecked():
                    raise ValueError("On-demand GPU AMR requires AMR enabled")
                target=Path(self.workspace.text()).expanduser().resolve()/"braids_project.json"
                if target.is_file():reject_raw_gpu_persistent_front(target)
                config=self._gpu_worker_config()
                config["gpu_integrator"]=self.gpu_integrator.currentData()
                config["gpu_launch_mode"]=self.gpu_launch_mode.currentData()
                config["gpu_numerical_variant"]=self.gpu_variant.currentData()
                config["gpu_amr_evolving_banks"]=(variant in (_ON_DEMAND_AMR_VARIANT,INTEGRATED) and
                    self.amr_enabled.isChecked() and
                    self.gpu_amr_evolving_banks.isChecked())
                general_amr=(self.amr_enabled.isChecked() and not self.bank_failure_enabled.isChecked() and
                    variant in (INTEGRATED,_ON_DEMAND_GENERAL_AMR_VARIANT))
                config['gpu_amr_selector_policy']=(self.gpu_selector_policy.currentData()
                    if general_amr else None)
            except Exception as exc:
                self._message("Prepare BRAIDS GPU run",str(exc),True);return
        path=self.save_project_ui()
        if path is None:return
        if gpu and config['gpu_numerical_variant']==INTEGRATED:
            try:config['gpu_numerical_variant']=resolve_project_route(load_project(path))
            except Exception as exc:
                self._message('Prepare BRAIDS GPU run',str(exc),True);return
        self.run_button.setEnabled(False);self._gpu_prepared_root=None;self._gpu_output_root=None
        # Keep tracking while the background task consumes the just-saved
        # project.  An edit made during preparation must not be blessed when
        # the task later completes.
        self._prepared_controls_armed=True;self._prepared_controls_dirty=False
        self.checkpoint.clear()
        self.execution_backend.setEnabled(False)
        self.progress.setValue(0);self.stop_button.setEnabled(True)
        if gpu:
            self.run_status.setText("Preparing scheme-6 deck and GPU input…")
            self.setup_status.setText("Preparing fixed-grid cut cells and GPU state…")
            self._start_preparation_task(self.prepare_run_button,
                lambda:PrepareBraidsGpuRunTask(str(path),**config),
                self._prepared_gpu,lambda task:self._task_failed(task,"GPU run preparation"))
            if config["gpu_numerical_variant"] in (_ON_DEMAND_AMR_VARIANT,
                                                    _ON_DEMAND_GENERAL_AMR_VARIANT):
                self.run_status.setText("Preparing scheme-6 deck and on-demand GPU AMR input...")
                self.setup_status.setText("Preparing native AMR hierarchy and zero-step GPU initialization...")
        else:
            self.run_status.setText("Preparing mesh, terrain, initial state and native input deck…")
            self.setup_status.setText("Preparing cut cells, terrain, initial state, and native deck…")
            self._start_preparation_task(self.prepare_run_button,lambda:PrepareBraidsRunTask(str(path)),
                self._prepared,lambda task:self._task_failed(task,"Run preparation"))

    def _prepared(self,task):
        changed_during_preparation=self._prepared_controls_dirty
        self.prepare_run_button.setEnabled(True);self.stop_button.setEnabled(False);self.execution_backend.setEnabled(True);self.run_root=Path(task.run_root);self.log.clear();QgsProject.instance().writeEntry("BRAIDS",self.RUN_KEY,str(self.run_root))
        try:cached=bool(json.loads((self.run_root/RUN_MARKER).read_text(encoding="utf-8")).get("preparation_cache"))
        except (OSError,ValueError,json.JSONDecodeError):cached=False
        prefix="Prepared from cache" if cached else "Prepared"
        self.run_status.setText(f"{prefix}: {self.run_root}");self.live_status.setText("Ready for live preview. It will appear as soon as the solver saves its first frame.");self.setup_status.setText(f"Native run prepared: {self.run_root}");self._select_tab("Run")
        self._arm_prepared_controls()
        if changed_during_preparation:self._prepared_controls_changed()
        else:self.run_button.setEnabled(True)

    def _sync_prepared_gpu_selector(self):
        """A reopened prepared job retains its recorded selector, including legacy absence."""
        if self._gpu_prepared_root is None:
            return
        try:
            receipt=json.loads((self._gpu_prepared_root/'preparation_receipt.json').read_text(encoding='utf-8'))
            policy=(receipt.get('selector_policy') if
                    receipt.get('variant')==_ON_DEMAND_GENERAL_AMR_VARIANT else None)
        except (OSError,ValueError,TypeError):
            policy=None  # Strict prepared-input admission reports the malformed receipt.
        self.gpu_selector_policy.setCurrentIndex(self.gpu_selector_policy.findData(policy))
        self._gpu_selector_user_choice=True

    def _prepared_gpu(self,task):
        changed_during_preparation=self._prepared_controls_dirty
        self.prepare_run_button.setEnabled(True);self.stop_button.setEnabled(False);self.execution_backend.setEnabled(True)
        self.run_root=Path(task.run_root);self._gpu_prepared_root=Path(task.prepared_root)
        self._sync_prepared_gpu_selector()
        self.log.clear();QgsProject.instance().writeEntry("BRAIDS",self.RUN_KEY,str(self.run_root))
        self.run_status.setText(f"Experimental GPU input prepared: {self._gpu_prepared_root}")
        try:
            receipt=json.loads((self._gpu_prepared_root/"preparation_receipt.json").read_text(encoding="utf-8"))
            on_demand=(receipt.get("schema")==6 and receipt.get("variant") in
                (_ON_DEMAND_AMR_VARIANT,_ON_DEMAND_GENERAL_AMR_VARIANT))
        except (OSError,ValueError,json.JSONDecodeError):on_demand=False
        self.setup_status.setText("On-demand GPU AMR input prepared; choose Run Prepared Case."
                                  if on_demand else "Scheme-6 GPU input prepared; choose Run Prepared Case.")
        self.live_status.setText("GPU live preview is ready; it will show each authenticated saved frame while the solver runs.")
        self._select_tab("Run");self._arm_prepared_controls()
        if changed_during_preparation:self._prepared_controls_changed()
        else:self.run_button.setEnabled(True)

    def _task_failed(self,task,label):
        outcome="failed" if getattr(task,"error",None) else "cancelled"
        if label in ("Run preparation","GPU run preparation"):
            self._prepared_controls_armed=False
            self.prepare_run_button.setEnabled(True);self.stop_button.setEnabled(False);self.execution_backend.setEnabled(True);self.run_status.setText(f"Run preparation {outcome}.")
            self.setup_status.setText(f"Run preparation {outcome}. Correct any reported input error and prepare again.")
        if label in ("Result preparation","Adaptive mesh preparation"):
            self.prepare_results_button.setEnabled(True);self.results_status.setText(f"Result preparation {outcome}. You can prepare again.")
        if label=="Result preparation" and getattr(task,"run_root",None) is not None and self._completed_gpu_run(Path(task.run_root)):
            self.result_mesh_progress.setVisible(True)
            self.result_mesh_progress.setFormat(f"GPU result preparation {outcome}: %p%")
        if label=="Adaptive mesh preparation":
            self.animate_result_mesh.setEnabled(True)
            self.animate_result_mesh.blockSignals(True);self.animate_result_mesh.setChecked(False);self.animate_result_mesh.blockSignals(False)
            self.result_mesh_progress.setFormat(f"Adaptive mesh {outcome} — %p%")
        if label in ("Run preparation","GPU run preparation") and getattr(task,"run_root",None) is not None:
            self.run_root=Path(task.run_root);QgsProject.instance().writeEntry("BRAIDS",self.RUN_KEY,str(self.run_root))
            self.run_button.setEnabled(False);self.run_status.setText(f"Preparation {outcome}: {self.run_root}")
            self.setup_status.setText("Run preparation cancelled. Partial inputs were preserved; prepare again to retry."
                                      if outcome=="cancelled" else
                                      "Run preparation failed. Correct the reported input or geometry error and prepare again.")
        if task.error:self._message(label,str(task.error),True)

    def start_run(self):
        try:
            if self.runner.is_running or self.gpu_runner.is_running or self._preparation_tasks.get(self.prepare_run_button):
                raise RuntimeError("A solver or run preparation is already active")
            gpu=self.execution_backend.currentData()=="gpu"
            if self.run_root is None:raise ValueError("Prepare a run first")
            self._assert_prepared_controls_current()
            checkpoint=self.checkpoint.text().strip() or None;self.log.clear()
            if gpu:
                if self._gpu_prepared_root is None:
                    raise ValueError("Prepare an authenticated GPU input first")
                config=self._gpu_worker_config()
                launch_mode=self.gpu_launch_mode.currentData()
                integrator=self.gpu_integrator.currentData()
                if integrator=="ssp33" and launch_mode!="plain":
                    raise ValueError("GPU SSPRK(3,3) requires Plain launches")
                receipt=json.loads((self._gpu_prepared_root/"preparation_receipt.json").read_text(encoding="utf-8"))
                variant=self.gpu_variant.currentData()
                if variant==INTEGRATED:
                    variant=route_for_preparation_schema(receipt.get('schema'),receipt.get('variant'))
                on_demand=(receipt.get("schema")==6 and receipt.get('variant') in
                    (_ON_DEMAND_AMR_VARIANT,_ON_DEMAND_GENERAL_AMR_VARIANT))
                selector_policy=(self.gpu_selector_policy.currentData() if
                    variant==_ON_DEMAND_GENERAL_AMR_VARIANT else None)
                if on_demand:
                    if variant!=receipt['variant']:
                        raise ValueError("Select the on-demand AMR variant used to prepare this input")
                    prepared=inspect_gpu_on_demand_prepared(self._gpu_prepared_root)
                    if prepared["variant"]!=variant:
                        raise ValueError("Selected GPU AMR input has a different numerical variant")
                    if prepared.get('selector_policy') != selector_policy:
                        raise ValueError('Selected GPU AMR refinement selector differs from the prepared input')
                else:
                    if variant in (_ON_DEMAND_AMR_VARIANT,_ON_DEMAND_GENERAL_AMR_VARIANT):
                        raise ValueError("Prepare schema-6 on-demand AMR input for this variant")
                    prepared=read_gpu_prepared(self._gpu_prepared_root)
                    if variant==_GENERAL_LIVE_TRANSPORT_VARIANT:
                        validate_integrator_scope(
                            bedlaw=int(prepared["bedlaw"]),
                            exchange_mode=int(prepared["exchange_mode"]),
                            integrator=gpu_integrator_label(integrator),
                            launch_mode=launch_mode,
                            bedform_roughness=int(prepared.get("bedform_roughness",0)),
                            bedload_adaptation=int(prepared.get("bedload_adaptation",0)),
                            general_avalanche=int(prepared.get("general_avalanche",0)),
                            schema_v2_law11=
                                has_public_law11_total_prepared_profile(prepared))
                    elif int(prepared["bedlaw"])==0 and (integrator!="ssp33" or launch_mode!="plain"):
                        raise ValueError("GPU no-bedload mode requires Plain launches and SSPRK(3,3)")
                    if int(prepared["exchange_mode"]) in (1,2):
                        if variant!=_GENERAL_LIVE_TRANSPORT_VARIANT and (integrator!="ssp33" or launch_mode!="plain"):
                            raise ValueError("GPU suspended exchange modes 1/2 require Plain launches and SSPRK(3,3)")
                        if receipt.get('variant')!='live_normal_critical_transport_ssp33' and receipt.get("native_integrator_requested")!="legacy":
                            raise ValueError("GPU suspended exchange input requires legacy native initialization; prepare again with the current worker")
                if checkpoint is not None:
                    selected=filesystem_path(Path(checkpoint).expanduser()).resolve()
                    if on_demand:
                        run_folder=next((parent for parent in selected.parents
                                         if (parent/"run.json").is_file()),None)
                        if run_folder is None:
                            raise ValueError("Select a checkpoint directory inside an authenticated GPU AMR run")
                    else:run_folder=selected.parent
                    candidate=inspect_gpu_resume_checkpoint(run_folder,self._gpu_prepared_root)
                    if selected!=filesystem_path(candidate["checkpoint"]).resolve():
                        raise ValueError("Select the manifest-selected authenticated GPU checkpoint")
                    if self.end_time.value()<=candidate["time_s"]:
                        raise ValueError("GPU restart End time must be later than its checkpoint time")
                    method=candidate["method"]
                    expected_integrator=gpu_integrator_label(integrator)
                    if (method.get("integrator")!=expected_integrator or
                            method.get("launch_mode")!=launch_mode or
                            (on_demand and method.get('selector_policy')!=selector_policy) or
                            ((on_demand or variant!="auto") and method.get("variant")!=variant) or
                            not math.isclose(float(method.get("dtmax_s",float("nan"))),
                                             self.dt_max.value(),rel_tol=0,abs_tol=1e-12)):
                        raise ValueError("GPU restart method differs; retain its integrator, launch mode, variant and maximum timestep")
                    if on_demand and not math.isclose(float(candidate["output_every_s"]),
                            self.output_interval.value(),rel_tol=0,abs_tol=1e-12):
                        raise ValueError("GPU AMR restart output interval differs from its accepted run")
                output=self._next_gpu_output_root()
                self.gpu_runner.start(**config,prepared_root=self._gpu_prepared_root,
                    output_root=output,gpu_launch_mode=launch_mode,
                    gpu_integrator=integrator,
                    gpu_numerical_variant=variant,
                    gpu_amr_selector_policy=selector_policy,
                    until=self.end_time.value(),output_every=self.output_interval.value(),
                    dt_max=self.dt_max.value(),resume=checkpoint)
                self._gpu_output_root=output
                self._begin_gpu_live_preview(output)
                self.run_status.setText(f"GPU run starting: {output}")
            else:
                self.runner.start(self.run_root,checkpoint,self.profile_native.isChecked())
            self.run_button.setEnabled(False);self.prepare_run_button.setEnabled(False)
            self.execution_backend.setEnabled(False);self.stop_button.setEnabled(True)
        except Exception as exc:
            running=self.runner.is_running or self.gpu_runner.is_running
            self.run_button.setEnabled(not running and not self._prepared_controls_dirty)
            self.stop_button.setEnabled(running)
            self._message("Run BRAIDS",str(exc),True)

    def _next_gpu_output_root(self):
        parent=self.run_root.parent
        stem="braids_gpu_"+datetime.now().strftime("%Y%m%d_%H%M%S")
        output=parent/stem;counter=1
        while output.exists():
            counter+=1;output=parent/f"{stem}_{counter:02d}"
        return output

    def stop_current_operation(self):
        """Cancel either the native solver or the currently preparing run."""
        if self.gpu_runner.is_running:
            self.run_status.setText("Requesting GPU stop at an accepted step…")
            self.gpu_runner.stop();return
        if self.runner.is_running:
            self.runner.stop();return
        task=self._preparation_tasks.get(self.prepare_run_button)
        if task is not None:
            self.run_status.setText("Cancelling run preparation…")
            self.setup_status.setText("Cancelling run preparation…")
            task.cancel()
            return
        self.stop_button.setEnabled(False)

    def _run_progress(self,value):self.progress.setValue(round(value))

    def _begin_gpu_live_preview(self,output):
        self._gpu_preview_timer.stop()
        if self._gpu_preview_task is not None:self._gpu_preview_task.cancel()
        self._gpu_preview_cursor=-1;self._gpu_preview_variable=None
        self._gpu_preview_cache=None;self._gpu_preview_manifest_mtime=None
        self._gpu_preview_manifest_count=0;self._gpu_preview_final_pending=False
        if self.live_preview.isChecked():
            self.live_status.setText("Waiting for the first committed GPU preview frame…")
        self._gpu_preview_timer.start()

    def _poll_gpu_live_preview(self):
        """Only a tiny manifest check runs on the UI thread; pixels use a task."""
        if (not self.live_preview.isChecked() or self._gpu_output_root is None or
                self._gpu_prepared_root is None or self._gpu_preview_task is not None):
            return
        index=self._gpu_output_root/"run.json"
        try:
            stamp=index.stat().st_mtime_ns
            if stamp!=self._gpu_preview_manifest_mtime:
                manifest=json.loads(index.read_text(encoding="utf-8"))
                frames=manifest.get("frames",[])
                if not isinstance(frames,list):raise ValueError("Invalid GPU frame inventory")
                self._gpu_preview_manifest_count=len(frames)
                self._gpu_preview_manifest_mtime=stamp
        except FileNotFoundError:
            return
        except (OSError,ValueError,json.JSONDecodeError) as exc:
            self.live_status.setText(f"Waiting for a complete GPU preview manifest: {exc}")
            return
        variable=self.live_variable.currentData()
        after=(self._gpu_preview_cursor if variable==self._gpu_preview_variable else -1)
        if self._gpu_preview_manifest_count<=after+1:return
        task=_GpuLivePreviewTask(self._gpu_output_root,self._gpu_prepared_root,
                                 variable,after,self._gpu_preview_cache)
        self._gpu_preview_task=task
        task.taskCompleted.connect(lambda:self._gpu_live_preview_finished(task,True))
        task.taskTerminated.connect(lambda:self._gpu_live_preview_finished(task,False))
        if not QgsApplication.taskManager().addTask(task):
            self._gpu_preview_task=None
            self.live_status.setText("QGIS could not queue the GPU preview task.")

    def _gpu_live_preview_finished(self,task,succeeded):
        if task is not self._gpu_preview_task:return
        self._gpu_preview_task=None
        if succeeded and task.product is not None:
            self._gpu_preview_cache=task.product["cache"]
            if (self.live_preview.isChecked() and
                    self._gpu_output_root==task.run_root and
                    self.live_variable.currentData()==task.variable):
                self._gpu_preview_cursor=int(task.product["frame_index"])
                self._gpu_preview_variable=task.variable
                self._show_gpu_live_preview(task.product)
            elif self.live_preview.isChecked():
                QTimer.singleShot(0,self._poll_gpu_live_preview)
        elif task.error is not None and self.live_preview.isChecked():
            self.live_status.setText(f"GPU live preview unavailable: {task.error}")
        if self._gpu_preview_final_pending:
            self._gpu_preview_final_pending=False
            QTimer.singleShot(0,self._poll_gpu_live_preview)

    def _show_gpu_live_preview(self,product):
        try:
            path=Path(product["path"])
            variable=product["variable"]
            label=f"BRAIDS Live GPU {self.live_variable.currentText()} · {self._gpu_output_root.name}"
            layer=self._live_result_layer(self._live_preview_layer)
            if layer is None:
                options=QgsRasterLayer.LayerOptions();options.loadDefaultStyle=False
                layer=QgsRasterLayer(str(path),label,"gdal",options)
                if not layer.isValid():raise ValueError(f"QGIS could not load {path}")
                QgsProject.instance().addMapLayer(layer);self._live_preview_layer=layer
                if self.iface:self.iface.mapCanvas().setExtent(layer.extent())
            elif Path(layer.source().split('|',1)[0]).resolve()!=path.resolve():
                layer.setDataSource(str(path),label,"gdal")
                if not layer.isValid():raise ValueError(f"QGIS could not load {path}")
            crs=QgsCoordinateReferenceSystem();crs.createFromWkt(product["crs_wkt"])
            layer.setCrs(crs)
            self._style_result_raster(layer,variable,product["range"],product["unit"])
            layer.setCustomProperty("braids/live_preview",True)
            layer.setCustomProperty("braids/live_preview_backend","gpu")
            layer.setCustomProperty("braids/time_s",float(product["time_s"]))
            layer.triggerRepaint()
            if self.iface:self.iface.mapCanvas().refresh()
            self.live_status.setText(
                f"Showing authenticated GPU {self.live_variable.currentText().lower()} "
                f"at t = {product['time_s']:g} s (frame {product['frame_index']}).")
        except Exception as exc:
            self.live_status.setText(f"GPU live preview unavailable: {exc}")

    def _toggle_live_preview(self,enabled):
        layer=self._live_result_layer(self._live_preview_layer)
        if layer is not None:
            node=QgsProject.instance().layerTreeRoot().findLayer(layer.id())
            if node is not None:node.setItemVisibilityChecked(bool(enabled))
        if not enabled and self._gpu_preview_task is not None:self._gpu_preview_task.cancel()
        if enabled:self._refresh_live_preview()

    def _refresh_live_preview(self,*_):
        if self.execution_backend.currentData()=="gpu":
            self._poll_gpu_live_preview();return
        if self._live_preview_frame is None or not self.live_preview.isChecked() or self.run_root is None:return
        self._show_live_preview(*self._live_preview_frame)

    def _live_frame_available(self,frame,time_s):
        self._live_preview_frame=(int(frame),float(time_s))
        self.live_status.setText(f"Native preview saved at t = {float(time_s):g} s. Updating QGIS live layer…")
        if self.live_preview.isChecked():self._show_live_preview(int(frame),float(time_s))

    def _show_live_preview(self,frame,time_s):
        if self.run_root is None:return
        try:
            manifest=materialize_live_preview_results(self.run_root)
            variable=self.live_variable.currentData()
            if not manifest or variable not in manifest["rasters"]:
                self.live_status.setText("Waiting for a complete native live-preview frame.");return
            product=manifest["rasters"][variable];path=self.run_root/"qgis_results"/product["path"]
            layer=self._live_result_layer(self._live_preview_layer)
            label=f"BRAIDS Live {self.live_variable.currentText()} · {self.run_root.name}"
            if layer is None:
                options=QgsRasterLayer.LayerOptions();options.loadDefaultStyle=False
                layer=QgsRasterLayer(str(path),label,"gdal",options)
                if not layer.isValid():raise ValueError(f"QGIS could not load live preview {path}")
                crs=QgsCoordinateReferenceSystem();crs.createFromWkt(manifest["crs_wkt"]);layer.setCrs(crs)
                QgsProject.instance().addMapLayer(layer);self._live_preview_layer=layer
                if self.iface:self.iface.mapCanvas().setExtent(layer.extent())
            elif Path(layer.source().split('|',1)[0]).resolve()!=path.resolve():
                layer.setDataSource(str(path),label,"gdal")
            else:
                # A VRT gains one band per native output.  Reload so GDAL sees
                # the updated band count instead of retaining its first view.
                layer.reload()
            self._style_result_raster(layer,variable,product["range"],product["unit"])
            renderer=layer.renderer()
            if isinstance(renderer,QgsSingleBandPseudoColorRenderer):renderer.setInputBand(max(1,int(product["band_count"])))
            layer.setCustomProperty("braids/live_preview",True);layer.setCustomProperty("braids/time_s",float(time_s));layer.triggerRepaint()
            if self.iface:self.iface.mapCanvas().refresh()
            self.live_status.setText(f"Showing native-grid {self.live_variable.currentText().lower()} at t = {float(time_s):g} s (frame {int(frame)}).")
        except Exception as exc:
            self.live_status.setText(f"Live preview unavailable: {exc}")

    def _run_finished(self,code,succeeded):
        self.run_button.setEnabled(not self._prepared_controls_dirty);self.prepare_run_button.setEnabled(True)
        self.execution_backend.setEnabled(True);self.stop_button.setEnabled(False)
        self._set_status("BRAIDS run completed" if succeeded else f"BRAIDS run failed or stopped (code {code})",Qgis.Success if succeeded else Qgis.Critical)
        if succeeded:self._refresh_results_runs(self.run_root);self._select_tab("Results");self.results_status.setText(f"Completed run: {self.run_root}")

    def _gpu_run_finished(self,code,succeeded):
        self._gpu_preview_timer.stop()
        self._gpu_preview_final_pending=True
        if self._gpu_preview_task is None:QTimer.singleShot(0,self._poll_gpu_live_preview)
        self.run_button.setEnabled(self._gpu_prepared_root is not None and
                                   not self._prepared_controls_dirty)
        self.prepare_run_button.setEnabled(True);self.execution_backend.setEnabled(True)
        self.stop_button.setEnabled(False)
        output=self._gpu_output_root
        if succeeded:
            self.checkpoint.clear()
            self._set_status("BRAIDS GPU run completed",Qgis.Success)
            self.run_status.setText(f"GPU run completed: {output}")
            self._refresh_results_runs(output);self._select_tab("Results")
            self.results_status.setText(f"Completed GPU run: {output}")
        else:
            self._set_status("BRAIDS GPU run stopped or failed",Qgis.Warning)
            try:
                if output is None or self._gpu_prepared_root is None:
                    raise ValueError("No GPU run output or prepared input is selected")
                candidate=inspect_gpu_resume_checkpoint(output,self._gpu_prepared_root)
                if candidate["status"]=="cancelled":
                    self.checkpoint.setText(str(candidate["checkpoint"]))
                    self.run_status.setText(
                        f"GPU stopped at {candidate['time_s']:.6g} s after "
                        f"{candidate['steps']} accepted steps. Set a later End time "
                        "and press Run Prepared Case to resume in a new folder.")
                else:
                    self.run_status.setText(
                        f"GPU run ended ({candidate['status']}); authenticated checkpoint "
                        f"available at {candidate['checkpoint']}.")
            except Exception as exc:
                self.run_status.setText(
                    f"GPU run stopped or failed (code {code}); no restart was "
                    f"selected automatically: {exc}")

    def prepare_results_ui(self,animated_mesh=None):
        selected=self.result_run_root or self.run_root
        if selected is None:self._message("BRAIDS results","Complete or select a run first",True);return
        prepared_root=None
        gpu_results=False
        gpu_amr_results=False
        selected=filesystem_path(selected).resolve()
        worker_manifest=selected/"run.json"
        if worker_manifest.is_file():
            try:worker=json.loads(worker_manifest.read_text(encoding="utf-8"))
            except (OSError,ValueError,json.JSONDecodeError):worker={}
            if worker.get("backend")=="gpu":
                gpu_results=True
                on_demand=(worker.get("schema")==3 and
                    worker.get("format")=="braids-gpu-amr-run-v2" and
                    worker.get("input_kind") in ("amr-on-demand", "amr-on-demand-general"))
                gpu_amr_results=on_demand or (worker.get("schema")==2 and
                    worker.get("input_kind")=="amr-prepared")
                if not on_demand:
                    declared=worker.get("input_folder")
                    if not isinstance(declared,str) or not declared:
                        self._message("BRAIDS GPU results","This run has no prepared-input folder identity.",True)
                        return
                    try:prepared_available=filesystem_path(
                        Path(declared).expanduser()).is_dir()
                    except (OSError,ValueError):prepared_available=False
                    if not prepared_available:
                        prepared_root=QFileDialog.getExistingDirectory(self,
                            "Locate the authenticated GPU prepared-input folder",
                            str(Path(selected).parent))
                        if not prepared_root:return
        animated=self.animate_result_mesh.isChecked() if animated_mesh is None else bool(animated_mesh)
        if animated and gpu_results and not gpu_amr_results:
            self._message("BRAIDS GPU results","Adaptive GPU mesh animation is not supported yet.",True)
            return
        progress_bar=None;failure_label="Result preparation"
        if animated:
            self.animate_result_mesh.setEnabled(False);self.result_mesh_progress.setVisible(True)
            self.result_mesh_progress.setValue(0);self.result_mesh_progress.setFormat("Building adaptive mesh: %p%")
            self.results_status.setText("Preparing the adaptive mesh animation…")
            progress_bar=self.result_mesh_progress;failure_label="Adaptive mesh preparation"
        elif gpu_results:
            self.result_mesh_progress.setVisible(True);self.result_mesh_progress.setValue(0)
            self.result_mesh_progress.setFormat("Preparing GPU results: %p%")
            self.results_status.setText("Preparing authenticated GPU raster results...")
            progress_bar=self.result_mesh_progress
        else:self.results_status.setText("Preparing multiband raster results…")
        self._start_preparation_task(self.prepare_results_button,lambda:PrepareBraidsResultsTask(
            str(selected),animated_mesh=animated,prepared_root=prepared_root),
            self._results_prepared,lambda task:self._task_failed(task,failure_label),progress_bar)

    def _results_prepared(self,task):
        same_run=bool(self._manifest and self._manifest.get("source_run")==task.manifest.get("source_run"))
        previous_frame=self._result_frame_index if same_run else 0
        restore_player=same_run and self.result_frame_slider.isEnabled()
        self.prepare_results_button.setEnabled(True);self._set_results_manifest(task.manifest)
        result_format=task.manifest.get("format")
        gpu_results=result_format in ("braids-qgis-gpu-results-v1",
                              "braids-qgis-gpu-amr-results-v1")
        self.animate_result_mesh.setEnabled(result_format!="braids-qgis-gpu-results-v1")
        mode="dynamic" if bool(task.manifest.get("mesh",{}).get("animated",False)) else "static"
        if task.manifest.get("mesh",{}).get("path") and self._pending_mesh_mode==mode:
            control=self.animate_result_mesh if mode=="dynamic" else self.show_result_mesh
            other=self.show_result_mesh if mode=="dynamic" else self.animate_result_mesh
            other.blockSignals(True);other.setChecked(False);other.blockSignals(False)
            control.blockSignals(True);control.setChecked(True);control.blockSignals(False)
            self._toggle_mesh_visibility(True)
        if mode=="dynamic":
            self.animate_result_mesh.setEnabled(True);self.result_mesh_progress.setVisible(True)
            self.result_mesh_progress.setValue(100);self.result_mesh_progress.setFormat("Adaptive mesh ready — %p%")
        elif gpu_results:
            self.result_mesh_progress.setVisible(True);self.result_mesh_progress.setValue(100)
            self.result_mesh_progress.setFormat("GPU results ready: %p%")
        elif self._pending_mesh_mode=="dynamic":
            self.animate_result_mesh.blockSignals(True);self.animate_result_mesh.setChecked(False);self.animate_result_mesh.blockSignals(False)
            self.result_mesh_progress.setVisible(True);self.result_mesh_progress.setValue(100)
            self.result_mesh_progress.setFormat("No adaptive mesh in this run — %p%")
        if restore_player or (mode=="dynamic" and self.animate_result_mesh.isChecked()):
            self._enable_result_player(previous_frame)
        self._pending_mesh_mode=None

    def _toggle_animated_result_mesh(self,checked):
        if self._manifest is None:return
        current=bool(self._manifest.get("mesh",{}).get("animated",False))
        if not checked:
            self._toggle_mesh_visibility(False)
            if current:
                self.result_mesh_progress.setVisible(True);self.result_mesh_progress.setValue(100)
                self.result_mesh_progress.setFormat("Adaptive mesh cached — %p%")
            return
        self.show_result_mesh.blockSignals(True);self.show_result_mesh.setChecked(False);self.show_result_mesh.blockSignals(False)
        self._toggle_mesh_visibility(False)
        if current:
            self.result_mesh_progress.setVisible(True);self.result_mesh_progress.setValue(100)
            self.result_mesh_progress.setFormat("Adaptive mesh ready — %p%")
            self._toggle_mesh_visibility(True)
            if self.animate_result_mesh.isChecked():self._enable_result_player(self._result_frame_index)
            return
        self._pending_mesh_mode="dynamic";self.prepare_results_ui(animated_mesh=True)

    def _set_results_manifest(self,manifest):
        self._manifest=manifest;self.result_run_root=filesystem_path(
            manifest["source_run"]).resolve();self.frame_combo.clear();self.result_variable.clear();self.result_boundary_id.clear()
        gpu_results=manifest.get("format")=="braids-qgis-gpu-results-v1"
        priority=("depth_m","velocity_m_s","stage_m","concentration_kg_m3","bed_elevation_m","bed_change_m",
                  "erodible_remaining_m","bedrock_elevation_m","manning_n")
        ordered=sorted(self._manifest["variables"],key=lambda name:(priority.index(name) if name in priority else len(priority),name))
        labels={"depth_m":"Water depth","velocity_m_s":"Flow speed","stage_m":"Water level",
            "manning_n":"Manning coefficient","concentration_kg_m3":"Total suspended concentration",
            "bed_elevation_m":"Bed elevation","bed_change_m":"Bed elevation change",
            "bedrock_elevation_m":"Bedrock elevation","erodible_remaining_m":"Remaining erodible thickness",
            "secondary_hI_m2_s":"Secondary-flow hI",
            "bank_undercut_m":"Bank undercut",
            "bank_toe_removed_depth_m":"Cumulative bank-toe removal (equivalent depth)",
            "bank_collapse_depth_m":"Cumulative bank collapse (equivalent depth)",
            "bank_collapse_count":"Bank collapse count"}
        for variable in ordered:
            unit=self._manifest["variables"][variable]
            label=labels.get(variable,variable.replace("_kg_m3","").replace("_kg_m2","").replace("_"," ").title())
            display_unit=unit.replace("kg/m3","kg/m³").replace("kg/m2","kg/m²")
            self.result_variable.addItem(f"{label} [{display_unit}]",variable)
        for frame in self._manifest["frames"]:self.frame_combo.addItem(f't = {frame["time_s"]:g} s',int(frame["band"])-1)
        if self.frame_combo.count():self.frame_combo.setCurrentIndex(self.frame_combo.count()-1)
        self._result_times=[float(frame["time_s"]) for frame in self._manifest["frames"]]
        try:
            for boundary_id in available_boundary_ids(self.result_run_root):
                self.result_boundary_id.addItem(boundary_id,boundary_id)
        except (OSError,ValueError,KeyError):
            pass
        self._reset_result_player()
        mesh_available=bool(self._manifest.get("mesh",{}).get("path"))
        self.show_result_mesh.blockSignals(True);self.show_result_mesh.setEnabled(mesh_available)
        self.show_result_mesh.setChecked(False);self.show_result_mesh.blockSignals(False)
        self.animate_result_mesh.blockSignals(True);self.animate_result_mesh.setChecked(False);self.animate_result_mesh.blockSignals(False)
        self.animate_result_mesh.setEnabled(not gpu_results)
        result_folder=self.result_run_root/"qgis_results"
        warning=""
        if manifest.get("format")=="braids-qgis-gpu-amr-results-v1":
            try:
                worker=json.loads((self.result_run_root/"run.json").read_text(encoding="utf-8"))
                if worker.get("original_dynamic_physics_gate_passed") is False:
                    warning=" Warning: this diagnostic AMR run failed its dynamic secondary-flow validation."
            except (OSError,ValueError,TypeError,json.JSONDecodeError):
                pass
        self.results_status.setText(f'Run {self.result_run_root.name}: {len(self._manifest["rasters"])} multiband rasters with {self.frame_combo.count()} saved times. Saved in {result_folder}.{warning}')

    def _selected_boundary_diagnostic(self):
        if self.result_run_root is None:raise ValueError("Select a completed BRAIDS run first")
        boundary_id=self.result_boundary_id.currentData()
        if not boundary_id:raise ValueError("This run has no available boundary ledger")
        return boundary_diagnostic(self.result_run_root,str(boundary_id))

    def plot_result_boundary(self):
        try:diagnostic=self._selected_boundary_diagnostic()
        except Exception as exc:self._message("BRAIDS boundary flow",str(exc),True);return
        water=diagnostic["quantities"]["water"]
        series=[("Accepted flow magnitude",diagnostic["interval_midpoints_s"],water["accepted_interval_average"],COLORS[0])]
        if water["requested_interval_average"] is not None:
            series.append(("Requested inlet flow",diagnostic["interval_midpoints_s"],water["requested_interval_average"],COLORS[1]))
        caption=(f'Boundary {diagnostic["bc_id"]} · {diagnostic["mode"]} · {diagnostic["direction"]}. '
                 f'{diagnostic["accepted_sign_convention"]}. {diagnostic["note"]}')
        show_plot_dialog(self,f'BRAIDS boundary flow · {diagnostic["bc_id"]}',
            [{"title":"Accepted boundary water flux","y_label":"m³/s","series":series,"include_zero":True}],caption)

    def export_result_boundary_csv(self):
        try:diagnostic=self._selected_boundary_diagnostic()
        except Exception as exc:self._message("BRAIDS boundary flow",str(exc),True);return
        selected,_=QFileDialog.getSaveFileName(self,"Export BRAIDS boundary CSV",
            f'braids_boundary_{diagnostic["bc_id"]}.csv',"CSV (*.csv)")
        if not selected:return
        path=Path(selected).with_suffix(".csv");keys=("water","suspended","bedload")
        with path.open("w",newline="",encoding="utf-8") as handle:
            writer=csv.writer(handle);writer.writerow(["interval_start_s","interval_end_s",
                "accepted_water_m3_s","requested_water_m3_s","accepted_suspended_kg_s",
                "requested_suspended_kg_s","accepted_bedload_kg_s","requested_bedload_kg_s",
                "boundary_direction"])
            for index,(start,end) in enumerate(zip(diagnostic["interval_start_s"],diagnostic["interval_end_s"])):
                row=[start,end]
                for key in keys:
                    quantity=diagnostic["quantities"][key];requested=quantity["requested_interval_average"]
                    row.extend((quantity["accepted_interval_average"][index],"" if requested is None else requested[index]))
                row.append(diagnostic["direction"]);writer.writerow(row)
        self.results_status.setText(f"Exported authoritative boundary-flow CSV: {path}")

    @staticmethod
    def _style_result_raster(layer,variable,limits,unit,alpha_band=0):
        """Apply stable, physically legible defaults for each result field."""
        minimum,maximum=map(float,limits);maximum=max(maximum,minimum+1e-9)
        palettes={
            "velocity_m_s":("Reds",("#fff5f0","#fcbba1","#fb6a4a","#cb181d","#67000d")),
            "stage_m":("Teal",("#f0fdfa","#99f6e4","#2dd4bf","#0f766e","#134e4a")),
            "concentration_kg_m3":("Purples",("#fcfbfd","#dadaeb","#9e9ac8","#756bb1","#54278f")),
            "bed_elevation_m":("Greens",("#f7fcf5","#c7e9c0","#74c476","#238b45","#00441b")),
            "erodible_remaining_m":("YlGn",("#ffffe5","#d9f0a3","#addd8e","#41ab5d","#005a32")),
            "manning_n":("Oranges",("#fff5eb","#fdd0a2","#fdae6b","#e6550d","#7f2704")),
            "bank_undercut_m":("Oranges",("#fff5eb","#fdd0a2","#fdae6b","#e6550d","#7f2704")),
            "bank_toe_removed_depth_m":("Reds",("#fff5f0","#fcbba1","#fb6a4a","#cb181d","#67000d")),
            "bank_collapse_depth_m":("Purples",("#fcfbfd","#dadaeb","#9e9ac8","#756bb1","#54278f")),
            "bank_collapse_count":("Purples",("#fcfbfd","#dadaeb","#9e9ac8","#756bb1","#54278f")),
            "secondary_hI_m2_s":("Teal",("#f0fdfa","#99f6e4","#2dd4bf","#0f766e","#134e4a")),
        }
        shader=QgsRasterShader();ramp=QgsColorRampShader();ramp.setColorRampType(QgsColorRampShader.Interpolated)
        ramp.setClassificationMode(Qgis.ShaderClassificationMethod.Continuous)
        if variable=="depth_m":
            # Keep a dry cell invisible.  The first opaque blue is deliberately
            # 1 mm, so very shallow water remains visually distinguishable from
            # exact zero without tinting the dry domain.
            lower=0.;upper=max(maximum,.001)
            values=[0.,.001]
            if upper>.001+1e-12:values.extend([.001+(upper-.001)/3,.001+2*(upper-.001)/3,upper])
            colors=[QColor(8,48,107,0),QColor("#eff3ff"),QColor("#bdd7e7"),QColor("#6baed6"),QColor("#08519c")]
            items=[QgsColorRampShader.ColorRampItem(0.,colors[0],"0.0 m (transparent)")]
            items.extend(QgsColorRampShader.ColorRampItem(value,colors[index+1],f"{value:g} {unit}")
                         for index,value in enumerate(values[1:]))
            ramp.setMinimumValue(lower);ramp.setMaximumValue(upper)
        elif variable=="bed_change_m":
            lower=min(minimum,0.);upper=max(maximum,0.)
            if upper==lower:lower-=.5;upper+=.5
            items=[QgsColorRampShader.ColorRampItem(lower,QColor("#2166ac"),f"{lower:g} {unit}"),
                   QgsColorRampShader.ColorRampItem(0.,QColor("#f7f7f7"),"0"),
                   QgsColorRampShader.ColorRampItem(upper,QColor("#b2182b"),f"{upper:g} {unit}")]
            ramp.setMinimumValue(lower);ramp.setMaximumValue(upper)
        else:
            _,colors=palettes.get(variable,("Viridis",("#440154","#3b528b","#21918c","#5ec962","#fde725")))
            lower=minimum;upper=maximum
            values=[lower+(upper-lower)*index/4 for index in range(5)]
            items=[QgsColorRampShader.ColorRampItem(value,QColor(color),f"{value:g} {unit}")
                   for value,color in zip(values,colors)]
            ramp.setMinimumValue(lower);ramp.setMaximumValue(upper)
        ramp.setColorRampItemList(items)
        shader.setRasterShaderFunction(ramp);renderer=QgsSingleBandPseudoColorRenderer(layer.dataProvider(),1,shader)
        if alpha_band:renderer.setAlphaBand(int(alpha_band))
        renderer.setClassificationMin(minimum);renderer.setClassificationMax(maximum);layer.setRenderer(renderer)

    @staticmethod
    def _live_result_layer(layer):
        """Return a cached raster only while its QGIS C++ object still exists."""
        if layer is None or sip.isdeleted(layer):return None
        try:
            if not layer.isValid() or QgsProject.instance().mapLayer(layer.id()) is None:return None
        except RuntimeError:
            return None
        return layer

    def _load_result_raster(self,index):
        if self._manifest is None:raise ValueError("Prepare raster results first")
        variable=self.result_variable.currentData()
        if not variable:raise ValueError("Select a result variable")
        product=self._manifest["rasters"].get(variable)
        if not product:raise ValueError(f"No raster was prepared for {variable}")
        result=Path(self._manifest["source_run"])/"qgis_results";path=result/product["path"]
        key=(self._manifest["source_run"],variable);layer=self._live_result_layer(self._result_layers.get(key))
        if layer is None:
            self._result_layers.pop(key,None)
            options=QgsRasterLayer.LayerOptions();options.loadDefaultStyle=False
            label=self.result_variable.currentText().rsplit(" [",1)[0]
            run_name=Path(self._manifest["source_run"]).name
            layer=QgsRasterLayer(str(path),f"BRAIDS {label} · {run_name}","gdal",options)
            if not layer.isValid():raise ValueError(f"QGIS could not load {path}")
            crs=QgsCoordinateReferenceSystem();crs.createFromWkt(self._manifest["crs_wkt"]);layer.setCrs(crs)
            self._style_result_raster(layer,variable,product["range"],product["unit"],product.get("alpha_band",0))
            layer.setCustomProperty("braids/run_root",self._manifest["source_run"]);layer.setCustomProperty("braids/result_variable",variable)
            layer.setCustomProperty("braids/simulation_times_seconds",json.dumps(self._result_times));QgsProject.instance().addMapLayer(layer)
            self._result_layers[key]=layer
        expected_bands=len(self._result_times)+(1 if product.get("alpha_band") else 0)
        if layer.bandCount()!=expected_bands:raise ValueError("Result raster does not have the expected saved-time and alpha bands")
        self._enable_result_player(index);return layer

    @staticmethod
    def _style_mesh_layer(layer):
        symbol=QgsFillSymbol.createSimple({"color":"0,0,0,0","outline_color":"35,35,35,185","outline_width":"0.18"})
        layer.setRenderer(QgsSingleSymbolRenderer(symbol));layer.triggerRepaint()

    def _load_result_mesh(self):
        if self._manifest is None:raise ValueError("Prepare raster results first")
        product=self._manifest.get("mesh") or {}
        if not product.get("path"):raise ValueError("This prepared result does not contain a computational mesh")
        run_root=self._manifest["source_run"]
        path=Path(run_root)/"qgis_results"/product["path"]
        layer=self._live_result_layer(self._result_mesh_layers.get(run_root))
        if layer is not None and Path(layer.source().split('|',1)[0]).resolve()!=path.resolve():
            layer.setDataSource(f'{path}|layername={product.get("layer",MESH_LAYER_NAME)}',layer.name(),'ogr')
            if not layer.isValid():raise ValueError(f"QGIS could not reload {path}")
            self._style_mesh_layer(layer)
        if layer is None:
            self._result_mesh_layers.pop(run_root,None)
            layer=QgsVectorLayer(f'{path}|layername={product.get("layer",MESH_LAYER_NAME)}',
                                 f'BRAIDS Computational Mesh · {Path(run_root).name}',"ogr")
            if not layer.isValid():raise ValueError(f"QGIS could not load {path}")
            crs=QgsCoordinateReferenceSystem();crs.createFromWkt(self._manifest["crs_wkt"]);layer.setCrs(crs)
            self._style_mesh_layer(layer);layer.setCustomProperty("braids/run_root",run_root)
            layer.setCustomProperty("braids/result_mesh",True);QgsProject.instance().addMapLayer(layer)
            self._result_mesh_layers[run_root]=layer
        return layer

    def _set_result_mesh_frame(self,index):
        if self._manifest is None or not (self.show_result_mesh.isChecked() or self.animate_result_mesh.isChecked()):return
        product=self._manifest.get("mesh") or {};ids=product.get("frame_mesh_ids") or []
        if not ids:return
        layer=self._load_result_mesh();mesh_id=int(ids[min(max(int(index),0),len(ids)-1)])
        expression=f'"mesh_id" = {mesh_id}'
        if layer.subsetString()!=expression:layer.setSubsetString(expression)
        layer.setCustomProperty("braids/mesh_id",mesh_id)
        if self._result_times:layer.setCustomProperty("braids/time_s",self._result_times[index])
        layer.triggerRepaint()

    def _toggle_result_mesh(self,checked):
        if self._manifest is None:return
        if not checked:
            self._toggle_mesh_visibility(False);return
        self.animate_result_mesh.blockSignals(True);self.animate_result_mesh.setChecked(False);self.animate_result_mesh.blockSignals(False)
        if bool(self._manifest.get("mesh",{}).get("animated",False)):
            self._toggle_mesh_visibility(False)
            self._pending_mesh_mode="static";self.prepare_results_ui(animated_mesh=False);return
        self._toggle_mesh_visibility(True)

    def _toggle_mesh_visibility(self,visible):
        try:
            if self._manifest is None:return
            layer=self._load_result_mesh() if visible else self._live_result_layer(
                self._result_mesh_layers.get(self._manifest["source_run"]))
            if layer is None:return
            node=QgsProject.instance().layerTreeRoot().findLayer(layer.id())
            if node is not None:node.setItemVisibilityChecked(bool(visible))
            if visible:self._set_result_mesh_frame(self._result_frame_index)
            if self.iface:self.iface.mapCanvas().refresh()
        except Exception as exc:
            self.show_result_mesh.blockSignals(True);self.show_result_mesh.setChecked(False);self.show_result_mesh.blockSignals(False)
            self.animate_result_mesh.blockSignals(True);self.animate_result_mesh.setChecked(False);self.animate_result_mesh.blockSignals(False)
            self._message("Load BRAIDS mesh",str(exc),True)

    def _reset_result_player(self):
        self.result_player_timer.stop();self._result_frame_index=0;count=len(self._result_times)
        self.result_frame_slider.blockSignals(True);self.result_frame_slider.setRange(0,max(0,count-1));self.result_frame_slider.setValue(0);self.result_frame_slider.blockSignals(False)
        for control in (self.result_frame_slider,self.result_play_button,self.result_pause_button,self.result_restart_button,self.result_playback_fps):control.setEnabled(False)
        self.result_frame_status.setText("Load a time series to enable playback.")

    def _enable_result_player(self,index=0):
        if not self._result_times:return
        for control in (self.result_frame_slider,self.result_play_button,self.result_pause_button,self.result_restart_button,self.result_playback_fps):control.setEnabled(True)
        self._set_result_frame(index)

    def _set_result_frame(self,index):
        if not self._result_times:return
        index=min(max(int(index),0),len(self._result_times)-1);self._result_frame_index=index
        for key,cached_layer in list(self._result_layers.items()):
            layer=self._live_result_layer(cached_layer)
            if layer is None:
                self._result_layers.pop(key,None);continue
            if str(layer.customProperty("braids/run_root",""))!=self._manifest["source_run"]:continue
            renderer=layer.renderer()
            if not isinstance(renderer,QgsSingleBandPseudoColorRenderer):continue
            renderer.setInputBand(index+1);layer.setCustomProperty("braids/time_s",self._result_times[index]);layer.triggerRepaint()
        self._set_result_mesh_frame(index)
        self.result_frame_slider.blockSignals(True);self.result_frame_slider.setValue(index);self.result_frame_slider.blockSignals(False)
        self.frame_combo.blockSignals(True);self.frame_combo.setCurrentIndex(index);self.frame_combo.blockSignals(False)
        self.result_frame_status.setText(f"Frame {index+1} / {len(self._result_times)} — simulation time {self._result_times[index]:g} s")
        if self.iface:self.iface.mapCanvas().refresh()

    def _start_result_animation(self):
        if not self._result_times:return
        if self._result_frame_index>=len(self._result_times)-1:self._set_result_frame(0)
        self.result_player_timer.start(max(1,round(1000/self.result_playback_fps.value())))

    def _restart_result_animation(self):self.result_player_timer.stop();self._set_result_frame(0)

    def _update_result_animation_rate(self,*_):
        if self.result_player_timer.isActive():self._start_result_animation()

    def _advance_result_frame(self):
        index=self._result_frame_index+1
        if index>=len(self._result_times):self.result_player_timer.stop();return
        self._set_result_frame(index)

    def load_result_animation(self):
        try:
            layer=self._load_result_raster(0);self.results_status.setText(f"Loaded {layer.name()} with {len(self._result_times)} saved frames. Use Animation to play or seek.")
        except Exception as exc:self._message("Load BRAIDS animation",str(exc),True)

    def load_selected_frame(self):
        try:
            index=self.frame_combo.currentData()
            if index is None:raise ValueError("Select a saved result time")
            layer=self._load_result_raster(int(index))
            self.results_status.setText(f"Loaded {layer.name()} at {self.frame_combo.currentText()}. The raster remains available for animation.")
        except Exception as exc:self._message("Load BRAIDS results",str(exc),True)

    def export_result_png_frames(self):
        try:
            if self.iface is None:raise ValueError("Open BRAIDS inside QGIS to export map frames")
            self._load_result_raster(0)
            directory=QFileDialog.getExistingDirectory(self,"Export BRAIDS animation PNG frames")
            if not directory:return
            directory=Path(directory);variable=str(self.result_variable.currentData())
            original=self._result_frame_index;canvas=self.iface.mapCanvas()
            for index,time_s in enumerate(self._result_times):
                self._set_result_frame(index);QgsApplication.processEvents()
                target=directory/f"braids_{variable}_frame_{index+1:05d}_t{time_s:g}s.png"
                canvas.saveAsImage(str(target));QgsApplication.processEvents()
            self._set_result_frame(original)
            self.results_status.setText(f"Exported {len(self._result_times)} PNG animation frames to {directory}.")
        except Exception as exc:self._message("Export BRAIDS animation",str(exc),True)

    def _clear_result_map_tool(self,kind):
        tool=getattr(self,kind,None)
        if tool is not None:
            tool.clear() if hasattr(tool,"clear") else None
            if self.iface and self.iface.mapCanvas().mapTool() is tool:self.iface.mapCanvas().unsetMapTool(tool)
        setattr(self,kind,None)

    def _replace_result_analysis_layer(self,kind,geometry_type,label):
        """Replace the single editable probe/transect layer for this dock."""
        for layer in list(QgsProject.instance().mapLayers().values()):
            if layer.customProperty("braids/result_analysis","")==kind:
                QgsProject.instance().removeMapLayer(layer.id())
        crs=QgsCoordinateReferenceSystem();crs.createFromWkt(self._manifest["crs_wkt"])
        name=f"BRAIDS {label} · {Path(self._manifest['source_run']).name}"
        layer=QgsVectorLayer(geometry_type,name,"memory")
        if not layer.isValid():raise ValueError(f"Could not create {name} layer")
        layer.setCrs(crs);layer.setCustomProperty("braids/result_analysis",kind)
        layer.setCustomProperty("braids/run_root",self._manifest["source_run"])
        if kind=="transect":
            layer.setRenderer(QgsSingleSymbolRenderer(QgsLineSymbol.createSimple(
                {"line_color":"30,90,220,255","line_width":"1.4","line_width_unit":"MM"})))
        else:
            layer.setRenderer(QgsSingleSymbolRenderer(QgsMarkerSymbol.createSimple(
                {"name":"circle","color":"30,90,220,255","outline_color":"255,255,255,255","size":"4.5"})))
        QgsProject.instance().addMapLayer(layer)
        setattr(self,f"_result_{kind}_layer",layer)
        return layer

    def _analysis_layer(self,kind):
        layer=self._live_result_layer(getattr(self,f"_result_{kind}_layer",None))
        if layer is None:
            for candidate in QgsProject.instance().mapLayers().values():
                if candidate.customProperty("braids/result_analysis","")==kind:
                    layer=candidate;setattr(self,f"_result_{kind}_layer",layer);break
        if layer is None:raise ValueError(f"Draw a {kind} first.")
        return layer

    def _store_analysis_geometry(self,kind,geometry):
        layer=self._analysis_layer(kind)
        if not layer.isEditable():layer.startEditing()
        for feature in layer.getFeatures():layer.deleteFeature(feature.id())
        feature=QgsFeature(layer.fields());feature.setGeometry(geometry)
        if not layer.addFeature(feature):raise RuntimeError(f"Could not save {kind} geometry")
        if not layer.commitChanges():raise RuntimeError(f"Could not finish {kind} drawing")
        layer.triggerRepaint()

    def _analysis_geometry(self,kind):
        layer=self._analysis_layer(kind)
        feature=next(layer.getFeatures(),None)
        if feature is None or feature.geometry() is None or feature.geometry().isEmpty():
            raise ValueError(f"Draw or edit the {kind} geometry first.")
        return feature.geometry()

    def _edit_result_analysis(self,kind,label):
        try:
            if self.iface is None:raise ValueError("Open BRAIDS inside QGIS to edit result geometry")
            layer=self._analysis_layer(kind)
            self.iface.setActiveLayer(layer)
            if not layer.isEditable():layer.startEditing()
            action=getattr(self.iface,"actionVertexTool",None)
            if callable(action):action().trigger()
            self.results_status.setText(f"Edit {label} vertices on the map, then Save Layer Edits. Plot and CSV export use the edited geometry.")
        except Exception as exc:self._message(f"Edit BRAIDS {label}",str(exc),True)

    def edit_result_probe(self):self._edit_result_analysis("probe","point probe")

    def edit_result_transect(self):self._edit_result_analysis("transect","flow transect")

    def draw_result_probe(self):
        try:
            if self._manifest is None:raise ValueError("Prepare results before drawing a probe")
            if self.iface is None:raise ValueError("Open BRAIDS inside QGIS to draw a probe")
            self._clear_result_map_tool("_result_probe_tool")
            self._replace_result_analysis_layer("probe","Point","Point Probe");self._probe_series=None
            canvas=self.iface.mapCanvas();self._result_probe_tool=_ResultProbeMapTool(
                canvas,self._result_probe_captured,lambda:self._clear_result_map_tool("_result_probe_tool"))
            canvas.setMapTool(self._result_probe_tool);self.results_status.setText("Created BRAIDS Result Probe layer. Click the map to place its point; right-click cancels.")
        except Exception as exc:self._message("BRAIDS point probe",str(exc),True)

    def _result_probe_captured(self,x,y):
        try:
            self._store_analysis_geometry("probe",QgsGeometry.fromPointXY(QgsPointXY(x,y)))
            self._probe_series=sample_result_probe(self._manifest,x,y)
            self.results_status.setText(f"Probe recorded at ({x:.3f}, {y:.3f}) with {len(self._result_times)} saved values.")
        except Exception as exc:self._message("BRAIDS point probe",str(exc),True)
        finally:self._clear_result_map_tool("_result_probe_tool")

    def plot_result_probe(self):
        try:
            point=self._analysis_geometry("probe").asPoint();self._probe_series=sample_result_probe(self._manifest,point.x(),point.y())
        except Exception as exc:self._message("BRAIDS point probe",str(exc),True);return
        plots=[]
        for variable,row in self._probe_series["series"].items():
            plots.append({"title":variable.replace("_"," ").title(),"y_label":row["unit"],
                          "series":[("Probe",self._probe_series["time_s"],row["values"],COLORS[0])],
                          "include_zero":variable in ("depth_m","velocity_m_s")})
        show_plot_dialog(self,"BRAIDS point probe",plots,"Values are sampled from the prepared native-grid result bands.")

    def export_result_probe_csv(self):
        try:
            point=self._analysis_geometry("probe").asPoint();self._probe_series=sample_result_probe(self._manifest,point.x(),point.y())
        except Exception as exc:self._message("BRAIDS point probe",str(exc),True);return
        selected,_=QFileDialog.getSaveFileName(self,"Export BRAIDS probe CSV","braids_probe.csv","CSV (*.csv)")
        if not selected:return
        path=Path(selected).with_suffix(".csv");series=self._probe_series["series"]
        with path.open("w",newline="",encoding="utf-8") as handle:
            writer=csv.writer(handle);writer.writerow(["time_s",*[f"{name}_{row['unit']}" for name,row in series.items()]])
            for index,time_s in enumerate(self._probe_series["time_s"]):writer.writerow([time_s,*[row["values"][index] for row in series.values()]])
        self.results_status.setText(f"Exported probe CSV: {path}")

    def draw_result_transect(self):
        try:
            if self._manifest is None:raise ValueError("Prepare results before drawing a flow transect")
            if "discharge_x_m2_s" not in self._manifest.get("rasters",{}):raise ValueError("This run has no directional discharge preview. Prepare and run it with BRAIDS alpha.44 or newer.")
            if self.iface is None:raise ValueError("Open BRAIDS inside QGIS to draw a flow transect")
            self._clear_result_map_tool("_result_transect_tool");self._replace_result_analysis_layer("transect","LineString","Flow Transect");self._transect_series=None;canvas=self.iface.mapCanvas()
            self._result_transect_tool=_ResultPolylineMapTool(canvas,self._result_transect_captured,
                lambda:self._clear_result_map_tool("_result_transect_tool"))
            canvas.setMapTool(self._result_transect_tool);self.results_status.setText("Created BRAIDS Flow Transect layer. Click vertices; right-click completes the line.")
        except Exception as exc:self._message("BRAIDS flow transect",str(exc),True)

    def _result_transect_captured(self,points):
        try:
            self._store_analysis_geometry("transect",QgsGeometry.fromPolylineXY([QgsPointXY(x,y) for x,y in points]))
            self._transect_series=integrate_result_transect(self._manifest,points)
            self.results_status.setText(f"Integrated flow across {len(points)-1} transect segment(s) for {len(self._result_times)} saved times.")
        except Exception as exc:self._message("BRAIDS flow transect",str(exc),True)
        finally:self._clear_result_map_tool("_result_transect_tool")

    def plot_result_transect(self):
        try:
            line=self._analysis_geometry("transect").asPolyline();self._transect_series=integrate_result_transect(
                self._manifest,[(point.x(),point.y()) for point in line])
        except Exception as exc:self._message("BRAIDS flow transect",str(exc),True);return
        show_plot_dialog(self,"BRAIDS flow transect",[{"title":"Signed discharge","y_label":"m³/s",
            "series":[("Flow",self._transect_series["time_s"],self._transect_series["flow_m3_s"],COLORS[0])]}],
            self._transect_series["sign_convention"])

    def export_result_transect_csv(self):
        try:
            line=self._analysis_geometry("transect").asPolyline();self._transect_series=integrate_result_transect(
                self._manifest,[(point.x(),point.y()) for point in line])
        except Exception as exc:self._message("BRAIDS flow transect",str(exc),True);return
        selected,_=QFileDialog.getSaveFileName(self,"Export BRAIDS transect CSV","braids_transect_flow.csv","CSV (*.csv)")
        if not selected:return
        path=Path(selected).with_suffix(".csv")
        with path.open("w",newline="",encoding="utf-8") as handle:
            writer=csv.writer(handle);writer.writerow(["time_s","flow_m3_s"])
            writer.writerows(zip(self._transect_series["time_s"],self._transect_series["flow_m3_s"]))
        self.results_status.setText(f"Exported transect-flow CSV: {path}")

    def visualize_initial_mesh(self):
        try:
            domain=self._domain();grid=grid_for_domain(domain,self._base_mesh_size());workspace=self._workspace_path()
            old=self._live_result_layer(self._initial_mesh_layer)
            if old is not None:QgsProject.instance().removeMapLayer(old.id())
            for layer_id,layer in list(QgsProject.instance().mapLayers().items()):
                if layer.customProperty("braids/initial_mesh",False):QgsProject.instance().removeMapLayer(layer_id)
            path=workspace/"braids_initial_mesh.gpkg"
            domain_data={"domain_id":domain.domain_id,"vertices_xy":domain.vertices_xy,"crs_wkt":domain.crs_wkt}
            grid_data={name:getattr(grid,name) for name in ("nx","ny","xlower_m","ylower_m","dx_m","dy_m")}
            obstacle_layer=self.obstacle_layer.currentLayer();obstacles=()
            if obstacle_layer is not None:
                obstacle_path=self._source_path(obstacle_layer,"a solid-obstacle polygon layer")
                obstacles=load_solid_obstacles(obstacle_path,domain.crs_wkt,
                    self._source_layer_name(obstacle_layer,OBSTACLE_LAYER_NAME))
            obstacle_data=[{"obstacle_id":item.obstacle_id,"vertices_xy":item.vertices_xy} for item in obstacles]
            preview_project=self._project_payload() if self.amr_enabled.isChecked() else None
            self.setup_status.setText("Building the initial cut-cell mesh preview…")
            self._start_preparation_task(self.visualize_initial_mesh_button,
                lambda:PrepareBraidsMeshTask(domain_data,grid_data,self.BOUNDARY_CELL_AREA_FRACTION,str(path),
                    obstacle_data,adaptive_project=preview_project),self._initial_mesh_ready,self._initial_mesh_failed)
        except Exception as exc:self._initial_mesh_failed(SimpleNamespace(error=exc))

    def _initial_mesh_ready(self,task):
        self.visualize_initial_mesh_button.setEnabled(True);metadata=task.metadata;path=Path(metadata["path"])
        layer=QgsVectorLayer(f'{path}|layername={metadata["layer"]}',"BRAIDS Initial Mesh","ogr")
        if not layer.isValid():
            self._initial_mesh_failed(type("Task",(),{"error":ValueError(f"QGIS could not load {path}")})());return
        self._style_mesh_layer(layer);layer.setCustomProperty("braids/initial_mesh",True)
        layer.setCustomProperty("braids/cell_count",metadata["cell_count"]);QgsProject.instance().addMapLayer(layer)
        self._initial_mesh_layer=layer
        self.setup_status.setText(f'Initial mesh displayed: {metadata["cell_count"]:,} cells ({metadata["partial_cell_count"]:,} partial; {metadata["merged_cell_count"]:,} merged), with {metadata.get("obstacle_count",0)} solid obstacle(s).')
        if self.iface:self.iface.mapCanvas().setExtent(layer.extent());self.iface.mapCanvas().refresh()

    def _initial_mesh_failed(self,task):
        self.visualize_initial_mesh_button.setEnabled(True);self.setup_status.setText("Initial mesh preview failed.")
        if getattr(task,"error",None):self._message("Visualize Initial Mesh",str(task.error),True)

    def shutdown(self):
        if hasattr(self,'adaptive_river_dialog'):
            self.adaptive_river_dialog.stop();self.adaptive_river_dialog.animation.stop()
        try:QgsProject.instance().layersAdded.disconnect(self._exclude_guidance_from_boundaries)
        except (TypeError,RuntimeError):pass
        for task in tuple(self._preparation_tasks.values()):task.cancel()
        if self.gpu_runner.is_running:self.gpu_runner.stop()
        self._gpu_preview_timer.stop()
        if self._gpu_preview_task is not None:self._gpu_preview_task.cancel()
        if self.runner.is_running:self.runner.stop()
        self._finish_water_body_point_capture()
        self._finish_obstacle_shape_capture()
        self._cancel_polygon_capture(update_status=False)
        self.result_player_timer.stop()
        for dialog in list(self._plot_windows):dialog.close()

    def dispose_when_idle(self):
        """Do not destroy a QProcess or task callback during cooperative stop."""
        if (self.runner.is_running or self.gpu_runner.is_running or
                (hasattr(self,'adaptive_river_dialog') and self.adaptive_river_dialog.job is not None) or
                self._gpu_preview_task is not None or
                self._preparation_tasks):
            if not hasattr(self,"_shutdown_timer"):
                self._shutdown_timer=QTimer(self)
                self._shutdown_timer.setInterval(250)
                self._shutdown_timer.timeout.connect(self.dispose_when_idle)
                self._shutdown_timer.start()
            return
        if hasattr(self,"_shutdown_timer"):self._shutdown_timer.stop()
        self.deleteLater()


# Compatibility for automation written against alpha 27 and earlier.
BraidsDockWidget=BraidsDockWidget
