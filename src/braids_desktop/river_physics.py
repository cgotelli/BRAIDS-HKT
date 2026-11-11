# Standalone PySide6 port of the maintained BRAIDS configuration editor.
"""Dynamic sediment and bed/bank editors using the shared physical resolver."""
from copy import deepcopy
import json
import math

from PySide6.QtCore import QObject, Signal, QEvent, QTimer, Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QHeaderView, QLabel, QLayout, QLineEdit, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget)

from braids_core.river_model import (CAPACITIES, HIDING, INTRINSIC, SLOPES,
    new_bedload_selection, new_model, resolve_model, SURVEY_BANKS, ModelError)
from braids_core.river_project import (GENERAL_PARAMETERS, SLOPE_PARAMETERS, HIDING_PARAMETERS,
                                      DEPENDENT_LAYER_PARAMETERS, VAN_RIJN_REFERENCE_PARAMETERS,
                                      SKIN_SHEAR_PARAMETERS)
from braids_core.sediment.capacity_parameters import FORMULA_CONTRACTS
from braids_core.sediment.grain_diameter_contract import (diameter_contract, require_diameter_definition,
    REQUIRED, LABELS, validate_grain_sizes)
from braids_core.sediment.transport_setup import (SUSPENSION, DIAMETER_LABELS,
    required_diameters, resolve_setup, require_runnable, summary as setup_summary)
from .accordion import AccordionSection
from .presentation import DisplaySpinBox


def combo(rows):
    result = QComboBox()
    for label, value in rows:
        result.addItem(label, value)
    return result


def number(value, minimum=0., maximum=1e9, decimals=8):
    result = DisplaySpinBox()
    result.setRange(minimum, maximum)
    result.setValue(value)
    return result


class RiverSection(AccordionSection):
    """Propagate wrapped form height to the enclosing scroll page."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._height_pending=False
        self.content.installEventFilter(self)
        self.layout().setSizeConstraint(QLayout.SetMinimumSize)

    def eventFilter(self,watched,event):
        if watched is self.content and event.type() in (QEvent.LayoutRequest,QEvent.Resize):
            if not self._height_pending:
                self._height_pending=True
                QTimer.singleShot(0,self._fit_content)
        return super().eventFilter(watched,event)

    def _fit_content(self):
        self._height_pending=False
        # Qt's generic scroll-page minimum misses a wrapping QLabel's actual
        # height-for-width. Measure the complete body, including dynamic rows.
        width=self.content.contentsRect().width()
        height=max(self.body_layout.minimumSize().height(),
                   self.body_layout.heightForWidth(width))
        height+=2*self.content.frameWidth()
        if height!=self.content.minimumHeight():
            self.content.setMinimumHeight(height)
            self.updateGeometry()


class WrappedNote(QLabel):
    """Keep a full-width form row's height tied to its displayed text width."""
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setWordWrap(True)

    def setText(self,text):
        super().setText(text)
        self._fit_text()

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self._fit_text()

    def _fit_text(self):
        # QLabel.heightForWidth may return its previous fixed height after a
        # hidden form is shown. Measure the text directly so it can shrink.
        width=max(1,self.width()-2*self.margin())
        bounds=self.fontMetrics().boundingRect(0,0,width,100000,int(Qt.TextWordWrap),self.text())
        self.setFixedHeight(max(0,bounds.height()+2*self.margin()+4))


def section(layout, title, *, expanded=False):
    widget = RiverSection(title.replace('_',' '), expanded=expanded)
    form = QFormLayout()
    widget.body_layout.addLayout(form)
    layout.addWidget(widget)
    return widget, form


def set_choice(widget, value):
    index = widget.findData(value)
    if index < 0:
        raise ValueError("Unknown displayed method: "+str(value))
    widget.setCurrentIndex(index)


def _text(table, row, col):
    item = table.item(row, col)
    return "" if item is None else item.text()


class RiverPhysicsEditor(QObject):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sediment_page, self.bed_page = QWidget(), QWidget()
        self._loading = False
        self._parameters = {}
        self._current_capacity = None
        self._grain_memory = {}
        self._diameter_values = {}  # measured values in mm, keyed by meaning; no inferred sizes
        self._diameter_rows = ()
        self._layer_memory = {}
        self._layer_mode = "bed.layered"
        self._layer_drafts = {}
        self.widgets = {}
        layout = QVBoxLayout(self.sediment_page)
        self.enabled = self._register("enabled", QCheckBox("Enable sediment transport"))
        self.enabled.setChecked(False)
        layout.addWidget(self.enabled)
        self.bedload_on = self._register("bedload_on", QCheckBox("Enable bedload transport"))
        self.transport_box, form = section(layout, "Transport", expanded=True)
        self.capacity = self._register("capacity", combo(
            [(name.removeprefix("capacity.").replace("_", " ").title(), name) for name in CAPACITIES]))
        set_choice(self.capacity, "capacity.mpm_braids_5_055")
        self.route = self._register("route", combo([
            ("Bedload channel only", "bedload"),
            ("Algebraic total load", "total_load")]))
        form.addRow("Transport formula", self.capacity)
        form.addRow("Transport route", self.route)
        self.route_note=WrappedNote()
        form.addRow(self.route_note)
        self.temperature = self._register("temperature", number(20., -19., 80., 2))
        form.addRow("Water temperature (°C)", self.temperature)
        # A widget owns the changing form so minimum-size changes propagate
        # through the accordion and scroll area after a formula switch.
        # A bare nested QFormLayout can retain the previous one-row height.
        self.parameter_panel = QWidget()
        self.parameter_form = QFormLayout(self.parameter_panel)
        self.parameter_form.setContentsMargins(0, 0, 0, 0)
        self.parameter_form.setSizeConstraint(QLayout.SetMinimumSize)
        form.addRow(self.parameter_panel)
        self.parameters = {}
        self.capacity.currentIndexChanged.connect(self._capacity_changed)
        self._capacity_changed()

        self.material_box, form = section(layout, "Grain sizes and bulk material", expanded=True)
        self.count = self._register("fractions", QSpinBox())
        self.count.setRange(1, 32)
        self.basis = self._register("basis", combo([
            ("Solid volume", "fraction_basis.solid_volume"), ("Mass", "fraction_basis.mass")]))
        form.addRow("Number of fractions", self.count)
        form.addRow("Input fraction units", self.basis)
        self.grain_preset = WrappedNote()
        form.addRow(self.grain_preset)
        self.grain_sizes = QTableWidget(0, 2)
        self.grain_sizes.setHorizontalHeaderLabels(['Diameter / percentile', 'Measured size (mm)'])
        self.grain_sizes.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.grain_sizes.itemChanged.connect(self._grain_sizes_changed)
        form.addRow(self.grain_sizes)
        # The first seven columns are the unchanged saved editor data. The last
        # is a read-only view, recomputed rather than trusted on project load.
        self.grains = QTableWidget(0, 8)
        self.grains.setHorizontalHeaderLabels(["ID", "Size (mm)", "Solid\ndensity\n(kg/m³)", "Surface fraction",
                                               "Dry bulk\ndensity\n(kg/m³)", "Within-class\nspread (phi)", "Percentile\n/ basis", "Derived\nsizes\n(mm)"])
        self.grains.horizontalHeaderItem(7).setToolTip('Read-only sizes resolved from the selected formula; not extra sediment fractions.')
        self.grains.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.grains.horizontalHeader().moveSection(6, 1)
        # Stretch widths settle after the form is shown/resized. Refit wrapped
        # cell text then as well, rather than eliding the first formula's row.
        self.grains.horizontalHeader().sectionResized.connect(lambda *_:self.grains.resizeRowsToContents())
        self.grains.setMinimumHeight(125)
        form.addRow(self.grains)
        self.diameter_note = WrappedNote()
        form.addRow(self.diameter_note)
        self.grain_note = QLabel()
        self.grain_note.setWordWrap(True)
        form.addRow(self.grain_note)
        self.composition_note = QLabel()
        self.composition_note.setWordWrap(True)
        form.addRow(self.composition_note)
        self.count.valueChanged.connect(self._resize_grains)
        self.grains.itemChanged.connect(self._changed)

        self.correction_box, form = section(layout, "Bed slope and hiding")
        self.slope_on = self._register("slope_on", QCheckBox("Enable bed-slope correction"))
        self.slope = self._register("slope", combo(
            [(v.removeprefix("bed_slope.delft_").replace("_", " ").title(), v) for v in SLOPES[1:]]))
        set_choice(self.slope, "bed_slope.delft_koch_flokstra")
        self.hiding_on = self._register("hiding_on", QCheckBox("Enable optional capacity hiding"))
        self.hiding = self._register("hiding", combo(
            [(v.removeprefix("hiding.").replace("_", " ").title(), v) for v in HIDING[1:]]))
        self.intrinsic_note = QLabel()
        self.intrinsic_note.setWordWrap(True)
        self.slope_hiding = self._register("slope_hiding", combo(
            [(v.removeprefix("hiding.").replace("_", " ").title(), v) for v in HIDING]))
        self.shear = self._register("shear", combo([
            ("Total hydraulic shear", "shear.total_roughness"),
            ("Soulsby–Clarke skin stress", "shear.soulsby_clarke_skin")]))
        for label, widget in (("", self.slope_on), ("Slope method", self.slope),
            ("", self.hiding_on), ("Capacity hiding", self.hiding), ("", self.intrinsic_note),
            ("Parker–Andrews slope hiding", self.slope_hiding), ("Sediment shear", self.shear)):
            form.addRow(label, widget)
        self.skin_shear_parameters = {}
        for key, default in SKIN_SHEAR_PARAMETERS.items():
            w = self._register("skin_"+key, number(default, 0. if key=="sand_roughness_m" else 1e-12, 1., 12))
            self.skin_shear_parameters[key] = w
            form.addRow("Skin sand roughness (m)" if key=="sand_roughness_m" else "Skin molecular viscosity (m²/s)", w)
        self.skin_note = WrappedNote()
        form.addRow(self.skin_note)
        self.slope_parameters = {}
        for key, default in SLOPE_PARAMETERS.items():
            self.slope_parameters[key] = self._register("slope_"+key,
                number(default, 0. if key == "thcrpa" else -100., 100., 8))
            form.addRow(key, self.slope_parameters[key])
        self.hiding_parameters = {}
        for channel, defaults in HIDING_PARAMETERS.items():
            self.hiding_parameters[channel] = {}
            for key, default in defaults.items():
                w = self._register("hiding_"+channel+"_"+key, number(default, -100., 100., 8))
                self.hiding_parameters[channel][key] = w
                form.addRow(channel.title()+" "+key.replace("_", " "), w)
        note = QLabel("Hiding coefficients are explicit inputs: choose values appropriate to the "
                      "selected formulation. Capacity and slope hiding are applied separately.")
        note.setWordWrap(True)
        form.addRow(note)
        self.vr_reference_box, form = section(layout, "Van Rijn reference conditions")
        self.vr_reference_parameters = {}
        labels = dict(critical_stress_factor="Critical shear multiplier (FactCr)",
            molecular_schmidt="Molecular Schmidt number",
            maximum_volume_concentration="Maximum reference volume concentration",
            bed_critical_stress_pa="Bed-property critical stress (Pa)",
            suspended_diameter_m="Measured suspended diameter Ds (m)")
        for key,default in VAN_RIJN_REFERENCE_PARAMETERS.items():
            widget=self._register("vr_reference_"+key,number(default,1e-8,1. if key=="maximum_volume_concentration" else 1e5,8))
            self.vr_reference_parameters[key]=widget
            form.addRow(labels[key],widget)

        self.suspension_box, form = section(layout, "Suspended transport")
        self.suspension_on = self._register("suspension_on", QCheckBox("Enable suspended-mass transport"))
        self.suspended_formula = self._register('suspended_formula', combo(
            [(row['label'], key) for key, row in SUSPENSION.items()]))
        self.suspended_formula.setCurrentIndex(self.suspended_formula.findData('suspended.garcia_parker_1991'))
        self.suspension = self._register("suspension", combo([
            ("Galappatti equilibrium exchange", "suspension.van_rijn_exchange"),
            ("Settling of supplied suspension only", "suspension.settling_only"),
            ("GP1991 independent entrainment", "suspension.garcia_parker_1991"),
            ("de Leeuw 2020 independent entrainment", "suspension.de_leeuw_2020"),
            ("Van Rijn 1984 independent suspension", "suspension.van_rijn_1984"),
            ("Zyserman-Fredsoe 1994 independent suspension", "suspension.zyserman_fredsoe_1994"),
            ("Camenen-Larson 2008 current suspension", "suspension.camenen_larson_2008")]))
        self.sus_factor = self._register("sus_factor", number(1., maximum=1., decimals=0))
        self.sus_diffusion = self._register("sus_diffusion", number(0.))
        self.reference_concentration = self._register("cref", number(1600., .001, 1e5, 3))
        for label, widget in (("", self.suspension_on), ("Exchange", self.suspension),
            ("Entrainment factor", self.sus_factor), ("Diffusivity (m²/s)", self.sus_diffusion),
            ("Reference concentration (kg/m³)", self.reference_concentration)):
            form.addRow(label, widget)
        self.suspension_note=WrappedNote()
        form.addRow(self.suspension_note)
        layout.addStretch()

        bed_layout = QVBoxLayout(self.bed_page)
        self.inventory_box, form = section(bed_layout, "Bed inventory", expanded=True)
        self.bed_mode = self._register("bed_mode", combo([
            ("Active layer + ordered substrate", "bed.layered"), ("Fully mixed finite bed", "bed.fully_mixed")]))
        self.transport_layer = self._register("transport_layer", combo([
            ("Constant thickness", "transport_layer.constant"),
            ("Water-depth scaled", "transport_layer.water_depth"),
            ("Dune-height scaled", "transport_layer.dune_height")]))
        self.layer_count = self._register("layer_count", QSpinBox())
        self.layer_count.setRange(2, 32)
        self.layer_count.setValue(2)
        form.addRow("Bed representation", self.bed_mode)
        form.addRow("Active-layer thickness method", self.transport_layer)
        form.addRow("Number of initial layers", self.layer_count)
        self.active_thickness = self._register("active_thickness", number(.02, 1e-9, 1e6))
        self.depth_fraction = self._register("depth_fraction", number(.01, 1e-9, 1e3))
        self.minimum_thickness = self._register("minimum_thickness", number(.001, 1e-9, 1e6))
        self.substrate_thickness = self._register("substrate_thickness", number(.2, 1e-9, 1e6))
        form.addRow("Constant active-layer target (m)", self.active_thickness)
        self.thickness_scale_label = QLabel("Active thickness / water depth")
        form.addRow(self.thickness_scale_label, self.depth_fraction)
        form.addRow("Minimum active thickness (m)", self.minimum_thickness)
        form.addRow("Eulerian substrate target (m)", self.substrate_thickness)
        self.inventory_note = QLabel("The initial bed table defines the starting stock. Target thickness "
            "redistributes it; a nonfinite basal rule may also replace material, as explained below. Changing bed representation "
            "keeps the other representation's table as an inactive draft.")
        self.inventory_note.setWordWrap(True)
        form.addRow(self.inventory_note)
        self.layers = QTableWidget()
        self.layers.setMinimumHeight(130)
        self.layers.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        form.addRow(self.layers)
        self.coarse_on = self._register("coarse_on", QCheckBox("Enable coarse layer"))
        self.exchange_on = self._register("exchange_on", QCheckBox("Enable exchange layer"))
        self.exchange_thickness = self._register("exchange_thickness", number(.02, 1e-9, 1e6))
        self.lagrangian_count = self._register("lagrangian_count", QSpinBox())
        self.lagrangian_count.setRange(0, 28)
        self.lagrangian_thickness = self._register("lagrangian_thickness", number(.02, 1e-9, 1e6))
        self.top_update = self._register("top_update", combo([
            ("Recompute top porosity from mixture", "column_top.recompute_porosity"),
            ("Mix existing and newly added porosity", "column_top.mix_added_porosity")]))
        self.base_update = self._register("base_update", combo([
            ("Finite independent base", "column_base.finite_independent"),
            ("Fixed basal composition", "column_base.fixed_composition"),
            ("Copy upper composition", "column_base.copy_upper_composition"),
            ("Fixed composition and thickness", "column_base.fixed_composition_and_thickness")]))
        self.porosity = self._register("porosity", combo([
            ("Packing included in dry bulk density", "porosity.dry_bulk_density"),
            ("Frings grain-distribution relation", "porosity.frings"),
            ("Weltje relation to Beard–Weyl data", "porosity.beard_weyl"),
            ("Constant porosity", "porosity.constant"),
            ("Sand/mud-weighted porosity", "porosity.sand_mud_weighted"),
            ("FLOW CDRYB flag — zero porosity", "porosity.flow_cdryb_zero")]))
        self.porosity_value = self._register("porosity_value", number(.4, 0., .999, 6))
        self.sand_porosity = self._register("sand_porosity", number(.4, 0., .999, 6))
        self.mud_porosity = self._register("mud_porosity", number(.6, 0., .999, 6))
        self.sorting_on = self._register("sorting_on", QCheckBox("Enable vertical sorting"))
        self.sorting = self._register("sorting", combo([
            ("Discrete Shields", "sorting.discrete_shields"),
            ("Continuous Shields", "sorting.continuous_shields"),
            ("Wilcock–McArdell", "sorting.wilcock_mcardell"),
            ("Transport hiding", "sorting.transport_hiding")]))
        form.addRow(self.coarse_on)
        form.addRow(self.exchange_on)
        form.addRow("Exchange-layer target (m)", self.exchange_thickness)
        form.addRow("Moving substrate layers", self.lagrangian_count)
        form.addRow("Moving substrate target (m)", self.lagrangian_thickness)
        form.addRow("Top porosity update", self.top_update)
        form.addRow("Basal-layer rule", self.base_update)
        form.addRow("Porosity method", self.porosity)
        form.addRow("Constant porosity (void fraction)", self.porosity_value)
        form.addRow("Sand porosity (void fraction)", self.sand_porosity)
        form.addRow("Mud porosity (void fraction)", self.mud_porosity)
        self.packing_note = QLabel()
        self.packing_note.setWordWrap(True)
        form.addRow(self.packing_note)
        basal_note = QLabel("Only the finite independent base preserves all basal material. "
            "Other rules may replace class mass; this is recorded separately from transport. "
            "Fixed composition and thickness is a replenishing basal inventory, not bedrock.")
        basal_note.setWordWrap(True)
        form.addRow(basal_note)
        self.sorting_box, form = section(bed_layout, "Coarse-layer sorting")
        form.addRow(self.sorting_on)
        form.addRow("Sorting method", self.sorting)
        self.dependent_parameters = {}
        for key, label in (
            ("coarse_thickness_m", "Coarse-layer target (m)"),
            ("a_max", "Maximum sorting length factor"),
            ("sink_fraction_max", "Maximum sorting fraction per update"),
            ("asfm", "Wilcock–McArdell coefficient"),
            ("bsfm", "Wilcock–McArdell exponent"),
            ("sigma_sfm", "Continuous mobility log spread"),
            ("viscosity_m2_s", "Sorting kinematic viscosity (m²/s)")):
            w = number(DEPENDENT_LAYER_PARAMETERS[key], 0. if key == "sink_fraction_max" else 1e-12,
                       1. if key == "sink_fraction_max" else 1e6, 12)
            if key == "a_max":
                w.setMinimum(1.)
            self.dependent_parameters[key] = self._register("dependent_"+key, w)
            form.addRow(label, w)
        self.hiding_scope = self._register("dependent_hiding_scope", combo([
            ("Active layer only", "active"), ("Active + coarse layers", "active_coarse")]))
        form.addRow("Material used for hiding", self.hiding_scope)
        self.layer_count.valueChanged.connect(self._resize_layers)
        self.layers.itemChanged.connect(self._changed)
        self.bed_mode.currentIndexChanged.connect(self._bed_mode_changed)
        self.coarse_on.toggled.connect(self._bed_mode_changed)
        self.exchange_on.toggled.connect(self._bed_mode_changed)
        self.lagrangian_count.valueChanged.connect(self._bed_mode_changed)
        self.bedform_box, form = section(bed_layout, "Bedforms")
        self.bedforms_on = self._register("bedforms_on", QCheckBox("Enable bedform roughness feedback"))
        form.addRow(self.bedforms_on)
        form.addRow(QLabel("Current bundle: Van Rijn 1984 height, length and roughness; equilibrium evolution."))
        self.bank_box, form = section(bed_layout, "Bank erosion and avalanching")
        self.banks_on = self._register("banks_on", QCheckBox("Enable bank erosion"))
        self.bank_method = self._register('bank_method',combo([
            ('FLOW dry-bank sharing','redistribution.dry_bank'),
            ('Surveyed toe supply',SURVEY_BANKS[0]),('Surveyed toe supply + retained block release',SURVEY_BANKS[1])]))
        self.bank_theta = self._register("bank_theta", number(.5, 0., 1., 4))
        self.bank_critical = self._register('bank_critical',number(.02,1e-9,1e5,9))
        from .river_bank_survey import RiverBankSurveyEditor
        self.bank_survey=RiverBankSurveyEditor();self.bank_survey.changed.connect(self._changed)
        self.bank_note=WrappedNote()
        self.avalanche_on = self._register("avalanche_on", QCheckBox("Enable slope redistribution"))
        self.avalanche = self._register("avalanche", combo([
            ("Conservative wet/dry collapse", "redistribution.braids_conservative_wet_dry_avalanche"),
            ("Repose projection (Jiang–Cao criterion)", "redistribution.jiang_cao_repose200_v1"),
            ("Classic FLOW wet-slope flux", "redistribution.avalanche")]))
        self.collapse_slope = self._register("collapse_slope", number(.5, .001, 100., 4))
        self.repose_angle_deg = self._register("repose_angle_deg", number(32., .01, 89.9, 2))
        self.repose_angle_deg.setToolTip("Measured noncohesive submerged repose angle. Not inferred from grain size. The coupled Jiang–Cao bundle uses its own Bedload repose parameter.")
        self.wet_slope = self._register("wet_slope", number(10., .000001, 100., 6))
        self.avalanche_time = self._register("avalanche_time", number(86400., .000001, 1.e9, 6))
        for label, widget in (("", self.banks_on),('Bank method',self.bank_method),
            ("Bank transfer fraction", self.bank_theta),('Critical undercut (m)',self.bank_critical),
            ('',self.bank_note),('',self.bank_survey),
            ("", self.avalanche_on), ("Redistribution method", self.avalanche),
            ("Trigger slope (m/m)", self.collapse_slope),
            ("Repose angle (degrees)", self.repose_angle_deg),
            ("FLOW wet slope (m/m)", self.wet_slope),
            ("FLOW relaxation time (s)", self.avalanche_time)):
            form.addRow(label, widget)
        self.avalanche_time.setToolTip("Numerical relaxation time, applied in the morphological clock. "
            "FLOW default: one day. This flux does not also run conservative collapse.")
        bed_layout.addStretch()
        self._resize_grains()
        set_choice(self.bed_mode,"bed.fully_mixed")
        self._refresh()

    def _register(self, name, widget):
        self.widgets[name] = widget
        widget.setObjectName("river_"+name)
        if isinstance(widget, QCheckBox):
            widget.toggled.connect(self._changed)
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(self._changed)
        else:
            widget.valueChanged.connect(self._changed)
        return widget

    def _capacity_changed(self, *_):
        if self._current_capacity is not None:
            self._parameters[self._current_capacity] = {k: w.text() for k, w in self.parameters.items()}
        while self.parameter_form.rowCount():
            self.parameter_form.removeRow(0)
        method = self.capacity.currentData()
        remembered = self._parameters.get(method, {})
        proposed = {"ACal": 1.}
        if method == "capacity.general":
            proposed = GENERAL_PARAMETERS
        elif method == "capacity.wang_fredsoe":
            proposed = {"VicMol": 1e-6, "ACal": 1.}
        from .bedload_catalog import CATALOG
        if getattr(self,'simple_bedload',False) and method in CATALOG:
            proposed=CATALOG[method]['parameters']
        self.parameters = {}
        for specification in FORMULA_CONTRACTS[method].parameters:
            if specification.keyword in remembered:
                text = remembered[specification.keyword]
            else:
                default = proposed.get(specification.keyword, specification.default)
                text = "" if default is None else str(default)
            field = QLineEdit(text)
            field.setMinimumHeight(field.minimumSizeHint().height())
            field.setPlaceholderText("Required coefficient" if specification.required else "Source default")
            field.textChanged.connect(self._changed)
            field.setToolTip(specification.current_role+" coefficient from the source formula")
            self.parameters[specification.keyword] = field
            self.parameter_form.addRow('MPM multiplier (standard = 1)' if method=='capacity.mpm' and specification.keyword=='ACal' else specification.keyword, field)
            if getattr(self,'simple_bedload',False) and method in CATALOG:
                self.parameter_form.setRowVisible(field,specification.keyword=='ACal' or method in ('capacity.wang_fredsoe','capacity.smart_jaggi','capacity.recking_2010','capacity.camenen_larson_2005','capacity.mpm_jc2026_p15_reconstruction_v1'))
                if method=='capacity.mpm_jc2026_p15_reconstruction_v1':
                    self.parameter_form.labelForField(field).setText(dict(VicMol='Water kinematic viscosity (m²/s)',
                        ReposeAngle_deg='Repose angle (degrees)',ProtectInitialDryBank='Protect initial dry-bank patch (0 = off, 1 = on)')[specification.keyword])
                if specification.keyword=='ACal':
                    self.parameter_form.labelForField(field).setText('Coefficient A (standard = 17)' if method=='capacity.ashida_michiue_generalized' else 'Transport multiplier (standard = 1)')
                if method=='capacity.smart_jaggi':
                    self.parameter_form.labelForField(field).setText(
                        'Coefficient A (standard = 4)' if specification.keyword=='ACal' else
                        'Critical Shields θcr (standard = 0.05)')
                    field.setToolTip('Dimensionless. Saved explicitly; no hidden grading multiplier.' if specification.keyword=='ACal' else
                                     'Dimensionless threshold for sediment motion; not an angle or a bed slope.')
                if specification.keyword=='ReachSlope':
                    self.parameter_form.labelForField(field).setText('Reach slope S (m/m; 0.01 = 1%)')
                    field.setToolTip('Required representative positive reach slope for Recking (2010). Not a local DEM gradient or instantaneous friction slope.')
                if method=='capacity.camenen_larson_2005' and specification.keyword=='VicMol':
                    self.parameter_form.labelForField(field).setText('Water kinematic viscosity (m²/s)')
                    field.setToolTip('Positive molecular water viscosity, typically 1e-6 m²/s at 20 °C. NOT eddy viscosity. Used to calculate critical Shields from D50; not a freely fitted threshold.')
                if method=='capacity.wang_fredsoe' and specification.keyword=='VicMol':
                    self.parameter_form.labelForField(field).setText('Water kinematic viscosity (m²/s)')
                    field.setToolTip('Positive water molecular kinematic viscosity, standard1e-6m²/s. Computes Dstar and FLOW Shields threshold. Not eddy viscosity; enter a measured/temperature-based value.')
        if hasattr(self,'capacity_note') and method in CATALOG:self.capacity_note.setText(CATALOG[method]['note'])
        if hasattr(self, 'grains') and not self._loading:
            self._preset_diameter()
        self._current_capacity = method
        if hasattr(self, "slope"):
            self._changed()

    def _resize_grains(self, *_):
        previous_loading, self._loading = self._loading, True
        for r in range(self.grains.rowCount()):
            self._grain_memory[r] = [_text(self.grains, r, c) for c in range(7)]
        self.grains.setRowCount(self.count.value())
        for r in range(self.count.value()):
            contract = diameter_contract(self.capacity.currentData(), self.count.value())
            row = self._grain_memory.get(r, [f"class_{r+1}", "0.5", "2650", "1" if r == 0 else "0", "1590", "0", contract.label])
            for c, value in enumerate(row):
                self.grains.setItem(r, c, QTableWidgetItem(value))
            self.grains.item(r, 6).setFlags(self.grains.item(r, 6).flags() & ~Qt.ItemIsEditable)
        if not previous_loading:
            self._preset_diameter()
        if hasattr(self, "layers"):
            self._resize_layers()
        self._loading = previous_loading
        self._changed()

    def _preset_diameter(self):
        """A changed meaning invalidates the number, never silently converts it."""
        previous, self._loading = self._loading, True
        contract = diameter_contract(self.capacity.currentData(), self.count.value())
        try:
            for r in range(self.grains.rowCount()):
                if _text(self.grains, r, 6) != contract.label:
                    self.grains.item(r, 1).setText('')
                    self.grains.item(r, 6).setText(contract.label)
                if self.capacity.currentData() in REQUIRED and self.count.value() == 1:
                    self.grains.item(r, 1).setText(self._diameter_values.get(contract.definition, ''))
                self.grains.item(r, 1).setToolTip('Enter '+contract.label+' in mm. Changing diameter basis clears this value; no conversion is inferred.')
                self.grains.item(r, 6).setToolTip(contract.note)
        finally:
            self._loading = previous

    def _grain_sizes_changed(self, *_):
        if self._loading:return
        for r, name in enumerate(self._diameter_rows):
            self._diameter_values[name] = _text(self.grain_sizes, r, 1)
        self._preset_diameter()
        self._changed()

    def _diameter_definition(self, row):
        contract = diameter_contract(self.capacity.currentData(), self.count.value())
        label = _text(self.grains, row, 6)
        if label != contract.label:
            raise ModelError('incompatible_grain_diameter', f'sediment.grains[{row}].diameter_definition',
                f'{self.capacity.currentText()} requires {contract.label}; received {label!r}. '
                'Choose the intended formula and enter its required diameter in Grain fractions.')
        require_diameter_definition(self.capacity.currentData(), self.count.value(), contract.definition)
        return contract.definition

    def _refresh_grain_table(self):
        """Show required independent bulk descriptors, never derived values."""
        from .bedload_catalog import CATALOG
        method = self.capacity.currentData()
        keys = REQUIRED.get(method, ()) if self.count.value() == 1 else ()
        planning = getattr(self, 'simple_bedload', False) and self.suspension_on.isChecked()
        if planning:
            keys = required_diameters(method if self.bedload_on.isChecked() else None,
                                      self.suspended_formula.currentData())
        self.grain_sizes.setVisible(bool(keys))
        self.grain_preset.setText(('Combined bedload + suspended setup' if self.bedload_on.isChecked() else 'Suspended-only setup') +
            ' — independent measurements for the active formulas; no sorting.' if planning else
            CATALOG.get(method, {}).get('label', self.capacity.currentText()) +
            ' — one bulk material, one transported flux. Required measured sizes below; no sorting.')
        if self._diameter_rows != keys:
            blocked = self.grain_sizes.blockSignals(True)
            self.grain_sizes.setRowCount(len(keys))
            for r, key in enumerate(keys):
                label = QTableWidgetItem('Ds (suspended)' if key == 'ds' else DIAMETER_LABELS[key])
                label.setToolTip(DIAMETER_LABELS[key])
                label.setFlags(label.flags() & ~Qt.ItemIsEditable)
                self.grain_sizes.setItem(r, 0, label)
                self.grain_sizes.setItem(r, 1, QTableWidgetItem(self._diameter_values.get(key, '')))
            self._diameter_rows = keys
            self.grain_sizes.blockSignals(blocked)
        self.grain_sizes.setFixedHeight(38 + 32*len(keys))
        self.grain_sizes.resizeRowsToContents()
        self.grains.setColumnHidden(7, True)
        for col in (1, 6):
            self.grains.setColumnHidden(col, bool(keys))
        self.grains.resizeRowsToContents()

    def _resize_layers(self, *_):
        previous_loading, self._loading = self._loading, True
        for r in range(self.layers.rowCount()):
            for c in range(self.layers.columnCount()):
                self._layer_memory[r, c] = _text(self.layers, r, c)
        self.layers.setRowCount(self.layer_count.value())
        self.layers.setColumnCount(self.count.value()+1)
        self.layers.setHorizontalHeaderLabels(["Thickness (m)"]+[
            _text(self.grains, r, 0) for r in range(self.count.value())])
        coarse = self.bed_mode.currentData() == "bed.layered" and self.coarse_on.isChecked()
        labels = ["Mixed bed"]
        if self.bed_mode.currentData() == "bed.layered":
            labels = (["Active"] + (["Coarse"] if coarse else []) +
                      (["Exchange"] if self.exchange_on.isChecked() else []) +
                      [f"Moving {j+1}" for j in range(self.lagrangian_count.value())])
            labels += [f"Fixed {j+1}" for j in range(self.layers.rowCount()-len(labels)-1)] + ["Base"]
        self.layers.setVerticalHeaderLabels(labels)
        for r in range(self.layers.rowCount()):
            for c in range(self.layers.columnCount()):
                default = (".02" if r == 0 else ".2") if c == 0 else ("1" if c == 1 else "0")
                self.layers.setItem(r, c, QTableWidgetItem(self._layer_memory.get((r, c), default)))
        self._loading = previous_loading
        self._changed()

    def _remember_layers(self):
        self._layer_drafts[self._layer_mode] = dict(count=self.layer_count.value(),
            rows=[[_text(self.layers, r, c) for c in range(self.layers.columnCount())]
                  for r in range(self.layers.rowCount())])

    def _topology_key(self):
        mode = self.bed_mode.currentData()
        if mode == "bed.fully_mixed":
            return mode
        return (mode + (".coarse" if self.coarse_on.isChecked() else "") +
                (".exchange" if self.exchange_on.isChecked() else "") +
                (f".lag{self.lagrangian_count.value()}" if self.lagrangian_count.value() else ""))

    def _minimum_layers(self):
        if self.bed_mode.currentData() == "bed.fully_mixed":
            return 1
        return 2+int(self.coarse_on.isChecked())+int(self.exchange_on.isChecked())+self.lagrangian_count.value()

    def _bed_mode_changed(self, *_):
        if self._loading:
            return
        self._remember_layers()
        self._layer_mode = self._topology_key()
        mixed = self._layer_mode == "bed.fully_mixed"
        coarse = not mixed and self.coarse_on.isChecked()
        exchange = not mixed and self.exchange_on.isChecked()
        nlag = 0 if mixed else self.lagrangian_count.value()
        draft = self._layer_drafts.get(self._layer_mode)
        if draft is None:
            composition = [_text(self.grains, r, 3) for r in range(self.count.value())]
            draft = dict(count=self._minimum_layers(),
                         rows=[[".22", *composition]] if mixed else
                              [[".02", *composition]] +
                              [[".02", *composition] for _ in range(int(coarse)+int(exchange)+nlag)] +
                              [[".2", *composition]])
        self._loading = True
        try:
            self.layers.setRowCount(0)
            self._layer_memory = {(r, c): text for r, row in enumerate(draft["rows"]) for c, text in enumerate(row)}
            self.layer_count.setRange(1, 1) if mixed else self.layer_count.setRange(self._minimum_layers(), 32)
            self.layer_count.setValue(draft["count"])
            self._resize_layers()
        finally:
            self._loading = False
        self._changed()

    def _changed(self, *_):
        if self._loading:
            return
        if hasattr(self, "bank_box"):
            self._refresh()
        self.changed.emit()

    def _refresh(self):
        on = self.enabled.isChecked()
        for widget in (self.transport_box, self.material_box, self.correction_box, self.suspension_box,
                       self.inventory_box, self.sorting_box, self.bedform_box, self.bank_box):
            widget.setEnabled(on)
        method = self.capacity.currentData()
        self.vr_reference_box.setVisible(method=="capacity.van_rijn_1993")
        self.vr_reference_box.setEnabled(on)
        selected_criterion=self.parameters.get("iTauCr")
        try:
            bed_criterion=selected_criterion is not None and float(selected_criterion.text())==2.
        except ValueError:
            bed_criterion=False  # Invalid input is still rejected by the resolver.
        self.vr_reference_parameters["bed_critical_stress_pa"].setEnabled(bed_criterion)
        self.vr_reference_parameters["critical_stress_factor"].setEnabled(not bed_criterion)
        total=self.bedload_on.isChecked() and self.route.currentData()=="total_load"
        vr_total=total and method=="capacity.van_rijn_1993"
        settling_only=self.suspension_on.isChecked() and self.suspended_formula.currentData()=="suspended.settling_only"
        vr_inactive=vr_total or (method=="capacity.van_rijn_1993" and settling_only)
        self.vr_reference_parameters["molecular_schmidt"].setEnabled(not vr_inactive)
        self.vr_reference_parameters["suspended_diameter_m"].setEnabled(False)
        if "IopSus" in self.parameters:self.parameters["IopSus"].setEnabled(not vr_inactive)
        self.route_note.setText(
            "FLOW algebraic total load: bedload plus equilibrium suspended flux, with no transported "
            "suspended inventory. Bed and suspended contributions retain their own directions."
            + (" Van Rijn 1993 uses FLOW's nonsuspended reference: zero fall speed, Schmidt number 1 "
               "and nominal diameter. DSS evolution is inactive. Choose suspended-mass transport "
               "for physical settling and an evolving suspended diameter." if vr_total else "")
            if total else
            ("Suspended mass has conservative advection and bed exchange; bedload is "
             + ("also active." if self.bedload_on.isChecked() else "OFF (factor zero).")
             if self.suspension_on.isChecked() else
             "Only the bedload channel is used. Equilibrium suspended capacity is not added."))
        intrinsic = method in INTRINSIC
        skin=self.shear.currentData()=="shear.soulsby_clarke_skin"
        self.shear.model().item(self.shear.findData("shear.soulsby_clarke_skin")).setEnabled(intrinsic)
        for w in self.skin_shear_parameters.values():
            w.setEnabled(skin and intrinsic)
        self.skin_note.setText(
            "Skin stress changes intrinsic transport mobility only; hydraulic drag, the transport "
            "prefactor and bed-slope stress are unchanged. Sand roughness is explicit (zero means "
            "smooth bed); viscosity is independent of settling temperature. Current-only, no mud or waves."
            if skin and intrinsic else
            ("This capacity cannot consume skin stress. Select total hydraulic shear; no automatic replacement was made."
             if skin else "Total hydraulic shear supplies intrinsic mobility. Skin-stress inputs are inactive."))
        self.intrinsic_note.setText("Required: this formula includes its own mixture/hiding dependence."
                                    if intrinsic else "")
        self.hiding.setEnabled(self.hiding_on.isChecked() and FORMULA_CONTRACTS[method].generic_hiding)
        self.slope.setEnabled(self.slope_on.isChecked())
        self.slope_hiding.setEnabled(self.slope_on.isChecked() and
                                     self.slope.currentData() == "bed_slope.delft_parker_andrews")
        keys = {"bed_slope.delft_bagnold_ikeda": ("alfabs", "alfabn"),
                "bed_slope.delft_koch_flokstra": ("alfabs", "ashld", "bshld", "cshld", "dshld"),
                "bed_slope.delft_parker_andrews": ("alfabs", "alfpa", "thcrpa")}[self.slope.currentData()]
        for key, w in self.slope_parameters.items():
            w.setEnabled(self.slope_on.isChecked() and key in keys)
        for channel, widgets in self.hiding_parameters.items():
            method = self.hiding.currentData() if channel == "capacity" else self.slope_hiding.currentData()
            enabled = (self.hiding_on.isChecked() and FORMULA_CONTRACTS[self.capacity.currentData()].generic_hiding
                       if channel == "capacity" else self.slope_hiding.isEnabled())
            for key, w in widgets.items():
                w.setEnabled(enabled and method == ("hiding.parker_klm" if key == "parker_alpha" else "hiding.wu_wang_jia"))
        self.suspension_on.setEnabled(on)
        for w in (self.suspension, self.sus_factor, self.sus_diffusion, self.reference_concentration):
            w.setEnabled(self.suspension_on.isChecked())
        self.sus_factor.setEnabled(self.suspension_on.isChecked() and not settling_only)
        self.suspension_note.setText(
            "BRAIDS deposition-only extension: supplied suspension advects and settles; no entrainment. "
            "Measured Ds sets clear-water physical settling speed; hindering is inactive. "
            "No equilibrium concentration or evolving suspended diameter is used. "
            "The inactive entrainment input is retained; the resolved factor is zero."
            if settling_only else
            "FLOW Galappatti exchange relaxes toward the selected equilibrium concentration. "
            "An entrainment factor of zero does not select deposition-only exponential settling.")
        self.temperature.setEnabled(self.suspension_on.isChecked() or self.capacity.currentData() in
            ("capacity.van_rijn_1993","capacity.van_rijn_1984_modified",
             "capacity.soulsby_van_rijn","capacity.soulsby"))
        layered = self.bed_mode.currentData() == "bed.layered"
        self.composition_note.setText("Automatically resolved physical weighting: " +
            ("reference-column volume fractions (FLOW layered bed)." if layered else
             "mass fractions (FLOW fully mixed bed).") + " This is independent of input fraction units.")
        self.transport_layer.setEnabled(layered)
        self.layer_count.setEnabled(layered)
        self.coarse_on.setEnabled(layered)
        self.exchange_on.setEnabled(layered)
        self.exchange_thickness.setEnabled(layered and self.exchange_on.isChecked())
        self.lagrangian_count.setEnabled(layered)
        self.lagrangian_thickness.setEnabled(layered and self.lagrangian_count.value() > 0)
        self.top_update.setEnabled(layered)
        self.base_update.setEnabled(layered)
        self.porosity.setEnabled(layered)
        porosity = self.porosity.currentData() if layered else "porosity.dry_bulk_density"
        self.porosity_value.setEnabled(layered and porosity == "porosity.constant")
        self.sand_porosity.setEnabled(layered and porosity == "porosity.sand_mud_weighted")
        # Current material catalog is noncohesive. Retain the inactive value;
        # never reinterpret fine sand as cohesive mud by its grain size alone.
        self.mud_porosity.setEnabled(False)
        self.grains.setColumnHidden(5, porosity not in ("porosity.frings", "porosity.beard_weyl"))
        self.grains.setColumnHidden(4, porosity != "porosity.dry_bulk_density")
        self.grain_note.setText(
            "Fractions must sum to 1. Changing input units does not convert the entered numbers. " +
            ("Dry bulk density includes pore space and must not exceed solid density."
             if porosity == "porosity.dry_bulk_density" else
             "Dry bulk density is inactive: bed volume uses solid density and the chosen porosity rule."))
        self.diameter_note.setText(diameter_contract(self.capacity.currentData(), self.count.value()).note +
            ' Values are retained only under the same diameter label; enter any missing measurements before Prepare.')
        self._refresh_grain_table()
        if getattr(self, 'simple_bedload', False) and self.suspension_on.isChecked():
            self.diameter_note.setText('The table is the union of active formula requirements. Ds is an independently '
                'specified representative suspended diameter, not another transported class or a bed percentile. '
                'No D50, D84, D90 or Dm is silently substituted for Ds. Check Summary for execution support.')
        self.packing_note.setText(
            "Packing is already in dry density; both top update rules are equivalent."
            if porosity == "porosity.dry_bulk_density" else
            "This locked FLOW flag returns zero porosity, not compaction. Dry density is not used."
            if porosity == "porosity.flow_cdryb_zero" else
            "Only noncohesive fractions are admitted here, so the sand value applies. "
            "Mud's inactive value is retained; dry density is not used."
            if porosity == "porosity.sand_mud_weighted" else
            "Explicit porosity uses solid density. Top rules either recompute mixture packing "
            "or mix existing and added packing. Grain-distribution rules use the entered "
            "within-class phi spread (zero means a single diameter).")
        if porosity in ("porosity.frings", "porosity.beard_weyl"):
            self.packing_note.setText(self.packing_note.text() +
                " For packing only, grains at or below 0.1 mm use FLOW's approximately 0.065 mm "
                "reference diameter; transport and settling retain the entered size.")
        self.sorting_on.setEnabled(layered)
        self.sorting.setEnabled(layered and self.sorting_on.isChecked())
        depth = self.transport_layer.currentData() != "transport_layer.constant"
        self.thickness_scale_label.setText("Active thickness / dune height" if
            self.transport_layer.currentData() == "transport_layer.dune_height" else "Active thickness / water depth")
        self.active_thickness.setEnabled(layered and self.transport_layer.currentData() == "transport_layer.constant")
        self.depth_fraction.setEnabled(layered and depth)
        self.minimum_thickness.setEnabled(layered and depth)
        self.substrate_thickness.setEnabled(layered)
        coarse = layered and self.coarse_on.isChecked()
        sorting = coarse and self.sorting_on.isChecked()
        for key, w in self.dependent_parameters.items():
            enabled = coarse if key == "coarse_thickness_m" else sorting
            if key in ("asfm", "bsfm"):
                enabled &= self.sorting.currentData() == "sorting.wilcock_mcardell"
            if key == "sigma_sfm":
                enabled &= self.sorting.currentData() in ("sorting.continuous_shields", "sorting.wilcock_mcardell")
            if key == "viscosity_m2_s":
                enabled &= self.sorting.currentData() in ("sorting.discrete_shields", "sorting.continuous_shields")
            w.setEnabled(enabled)
        self.hiding_scope.setEnabled(coarse)
        self.bank_theta.setEnabled(self.banks_on.isChecked())
        bank_active=self.banks_on.isChecked();survey=self.bank_method.currentData() in SURVEY_BANKS
        self.bank_method.setEnabled(bank_active)
        finite=(self.bed_mode.currentData()=='bed.fully_mixed' or
                self.base_update.currentData()=='column_base.finite_independent')
        for method in SURVEY_BANKS:
            eligible=self.route.currentData()=='bedload' and finite and not (self.bedload_on.isChecked() and self.capacity.currentData()=='capacity.mpm_jc2026_p15_reconstruction_v1')
            eligible &= not (method==SURVEY_BANKS[1] and self.avalanche_on.isChecked() and
                             self.avalanche.currentData()=='redistribution.avalanche')
            self.bank_method.model().item(self.bank_method.findData(method)).setEnabled(eligible)
        self.bank_survey.setVisible(survey);self.bank_survey.setEnabled(bank_active)
        self.bank_critical.setEnabled(bank_active and self.bank_method.currentData()==SURVEY_BANKS[1])
        self.bank_note.setText('Surveyed banks use net bed erosion from bedload and/or suspended exchange, with finite stock. Eligible columns and toe recipients '
            'remain finest; the rest of the mesh adapts. Retained blocks support conservative wet/dry collapse, '
            'not a second FLOW wet-slope flux. Existing incompatible choices are reported, never reset.' if survey else
            'FLOW dry-bank sharing uses local adjacent dry cells. No geographic toe survey or retained undercut is used.')
        self.avalanche.setEnabled(self.avalanche_on.isChecked())
        classic = self.avalanche_on.isChecked() and self.avalanche.currentData() == "redistribution.avalanche"
        self.collapse_slope.setEnabled(self.avalanche_on.isChecked() and not classic and self.avalanche.currentData()!='redistribution.jiang_cao_repose200_v1')
        self.wet_slope.setEnabled(classic)
        self.avalanche_time.setEnabled(classic)
        self.repose_angle_deg.setEnabled(self.avalanche_on.isChecked() and self.avalanche.currentData()=="redistribution.jiang_cao_repose200_v1" and not (self.bedload_on.isChecked() and self.capacity.currentData()=="capacity.mpm_jc2026_p15_reconstruction_v1"))
        # Extension visibility is not a physical process switch. Processes
        # remain independent; dependencies are checked, never silently reset.
        active = self.active
        self.bedload_on.setEnabled(on)
        for box in (self.transport_box,self.material_box,self.vr_reference_box,self.inventory_box):
            box.setEnabled(active)
        self.correction_box.setEnabled(on and (self.bedload_on.isChecked() or self.slope_on.isChecked()))
        self.slope_on.setEnabled(on and (self.bedload_on.isChecked() or self.slope_on.isChecked()))
        self.route.setEnabled(on and self.bedload_on.isChecked())
        self.suspension_box.setEnabled(on and self.suspension_on.isChecked())
        if hasattr(self, 'suspended_plan_box'):
            self.suspended_plan_box.setEnabled(on and self.suspension_on.isChecked())
            row = SUSPENSION[self.suspended_formula.currentData()]
            self.suspended_plan_note.setText(row['closure'] + '\n\n' + row['auxiliaries'] +
                ('\n\nRunnable bundle: current only; measured Ds retained; positive MF; start=0; conservative collapse is available; accepted-demand bank supply is available.'
                 if self.suspended_formula.currentData() in ('suspended.settling_only','suspended.van_rijn_1993_flow2dh','suspended.garcia_parker_1991','suspended.de_leeuw_2020','suspended.van_rijn_1984','suspended.zyserman_fredsoe_1994','suspended.camenen_larson_2008') else
                 '\n\nConfiguration only — execution qualification pending at '+row['stage']+'.'))
            self.reference_concentration.setEnabled(on and self.suspension_on.isChecked() and self.suspended_formula.currentData() not in ('suspended.settling_only','suspended.garcia_parker_1991','suspended.de_leeuw_2020','suspended.van_rijn_1984','suspended.zyserman_fredsoe_1994','suspended.camenen_larson_2008'))
            self.transport_box.setEnabled(on and self.bedload_on.isChecked())
        self.sorting_box.setEnabled(active or (on and self.sorting_on.isChecked()))

    @property
    def active(self):
        return self.enabled.isChecked() and any(w.isChecked() for w in
            (self.bedload_on,self.suspension_on,self.banks_on,self.avalanche_on,self.bedforms_on))

    def validate_choices(self):
        if not self.enabled.isChecked():return
        def fail(path,message):raise ModelError('incompatible_physics','sediment.'+path,message)
        bed=self.bedload_on.isChecked();sus=self.suspension_on.isChecked()
        if getattr(self,'simple_bedload',False):
            from .bedload_catalog import CATALOG
            if sus:
                # Validate the independent proposal before the generic legacy
                # guard. Never let it execute the old shared capacity channel.
                require_runnable(self.transport_setup())
            if any(w.isChecked() for w in (self.bedforms_on,self.sorting_on,self.slope_on,self.hiding_on,self.coarse_on,self.exchange_on)):
                fail('workflow','This Desktop release supports one bulk material with qualified bedload/suspension. Turn off slope, hiding, bedforms and sorting extras; hidden processes cannot execute.')
            from braids_core.sediment.collapse_setup import validate as validate_collapse
            # An inactive bedload draft must not decide a suspended-only model.
            validate_collapse(self.avalanche.currentData() if self.avalanche_on.isChecked() else 'off',
                self.capacity.currentData() if bed else None,
                self.suspended_formula.currentData() if sus else 'off')
            from braids_core.river_jiang_cao import METHOD as JC_METHOD
            if bed and (self.capacity.currentData() not in CATALOG or self.route.currentData()!='bedload' or self.count.value()!=1
                        or self.bed_mode.currentData()!='bed.fully_mixed' or self.shear.currentData()!='shear.total_roughness'
                        or self.porosity.currentData()!='porosity.dry_bulk_density' or self.base_update.currentData()!='column_base.finite_independent'):
                fail('workflow','Choose a listed bedload formula, one grain, total hydraulic shear and a fully mixed finite bed with packing from dry bulk density. Create a new bedload project.')
            if bed:
                for key,value in CATALOG[self.capacity.currentData()]['parameters'].items():
                    try:actual=float(self.parameters[key].text())
                    except ValueError:actual=None
                    if key!='ACal' and self.capacity.currentData() not in ('capacity.wang_fredsoe','capacity.smart_jaggi','capacity.recking_2010','capacity.camenen_larson_2005',JC_METHOD) and actual!=value:
                        fail('capacity.parameters.'+key,f'The simple bedload setup requires {key}={value}. Restore this documented value in the project JSON or create a new project; it is not silently reset.')
        if bed and sus and self.route.currentData()=='total_load':
            fail('transport','Algebraic total load already includes suspended capacity. Choose Bedload channel only in 04 / Bedload to use independently transported suspension.')
        from braids_core.sediment.bank_setup import validate as validate_bank
        validate_bank(self.bank_method.currentData() if self.banks_on.isChecked() else 'off',
            self.capacity.currentData() if bed else None,
            self.suspended_formula.currentData() if sus else 'off',bedload=bed)
        if self.slope_on.isChecked() and not bed:
            fail('bed_slope','Bed-slope correction acts on bedload. Enable bedload in 04 / General, or turn off the correction in 04 / Bedload.')
        if self.sorting_on.isChecked() and not (bed or sus):
            fail('bed.sorting','Sorting needs sediment transport. Enable bedload or suspension in 04 / General, or disable sorting in 05 / Bedload.')

    def transport_setup(self):
        """Backend-independent S1 proposal. This does not construct a run model."""
        on = self.enabled.isChecked()
        bed = self.capacity.currentData() if on and self.bedload_on.isChecked() else None
        sus = self.suspended_formula.currentData() if on and self.suspension_on.isChecked() else None
        sizes = {}
        for key in required_diameters(bed, sus):
            raw=self._diameter_values.get(key, '')
            try:sizes[key] = float(str(raw).strip()) / 1000.
            except (TypeError,ValueError):sizes[key] = None
        return resolve_setup(bedload=bed, suspended=sus, diameters_m=sizes)

    def transport_setup_snapshot(self):
        """Save readable assumptions/admission even for a not-yet-runnable draft."""
        try:return self.transport_setup()
        except ModelError as exc:
            return dict(status='invalid', category=exc.category, path=exc.path, message=str(exc))

    def transport_setup_text(self):
        try:
            from braids_core.sediment.collapse_setup import summary, resolve
            from braids_core.sediment.bank_setup import summary as bank_summary
            return setup_summary(self.transport_setup())+'\n\n'+bank_summary(self.bank_method.currentData() if self.enabled.isChecked() and self.banks_on.isChecked() else 'off')+'\n\n'+summary(
                resolve(self.avalanche.currentData() if self.enabled.isChecked() and self.avalanche_on.isChecked() else 'off', self.capacity.currentData() if self.bedload_on.isChecked() else None),
                slope=self.collapse_slope.value(),angle=self.repose_angle_deg.value())
        except ModelError as exc:
            return 'Configuration needs attention (first error):\n\n' + str(exc) + '\n\nNo physics has been silently changed. Correct this input and check again.'

    def _numeric_table(self,table,row,col,label):
        import math
        try:value=float(_text(table,row,col))
        except ValueError:value=float('nan')
        if not math.isfinite(value):
            raise ModelError('invalid_input','sediment.'+label,
                f'Row {row+1}, {table.horizontalHeaderItem(col).text()}: enter a finite number.')
        return value

    def selection(self):
        self.validate_choices()
        vr_suspended=self.suspension_on.isChecked() and self.suspended_formula.currentData()=='suspended.van_rijn_1993_flow2dh'
        gp_suspended=self.suspension_on.isChecked() and self.suspended_formula.currentData()=='suspended.garcia_parker_1991'
        dl_suspended=self.suspension_on.isChecked() and self.suspended_formula.currentData()=='suspended.de_leeuw_2020'
        from braids_core.sediment.river_classic_suspension import EDITOR
        classic_suspended=self.suspension_on.isChecked() and self.suspended_formula.currentData() in EDITOR
        supplied_suspended=self.suspension_on.isChecked() and self.suspended_formula.currentData()=='suspended.settling_only'
        independent_suspended=gp_suspended or dl_suspended or classic_suspended or supplied_suspended
        suspended_only=(vr_suspended or independent_suspended) and not self.bedload_on.isChecked()
        method=('capacity.general' if independent_suspended else 'capacity.van_rijn_1993') if suspended_only else self.capacity.currentData()
        from braids_core.sediment.river_vr1993 import DESKTOP_PARAMETERS
        parameters={}
        for key,w in (() if suspended_only else self.parameters.items()):
            try:value=float(w.text());valid=__import__('math').isfinite(value)
            except ValueError:valid=False
            if not valid:raise ModelError('invalid_input','sediment.capacity.parameters.'+key,
                'Enter a finite coefficient for '+key+' in 04 / General / Transport.')
            spec=next((p for p in FORMULA_CONTRACTS[self.capacity.currentData()].parameters if p.keyword==key),None)
            if spec is None:
                # A stale editor draft must never surface as an opaque
                # StopIteration.  The current formula editor normally cannot
                # create this state, so this is a recoverable configuration
                # diagnostic for old/corrupt project state.
                raise ModelError('invalid_input','sediment.capacity.parameters.'+key,
                    f'{self.capacity.currentText()} does not define parameter {key!r}. Re-select the formula, then enter only its displayed parameters.')
            if spec.value_kind=='integer' and value!=int(value):
                raise ModelError('invalid_input','sediment.capacity.parameters.'+key,
                    'Enter an integer for '+key+' in 04 / General / Transport.')
            parameters[key]=int(value) if spec.value_kind=='integer' else value
        s = new_bedload_selection()
        sizes = {}
        if suspended_only:
            from braids_core.sediment.river_garcia_parker import ZERO_CAPACITY
            parameters=dict(ZERO_CAPACITY if independent_suspended else DESKTOP_PARAMETERS)
        for name in REQUIRED.get(method, ()):
            try:value = float(str(self._diameter_values.get(name, '')).strip())/1000.
            except (TypeError,ValueError):
                raise ModelError('invalid_input', 'sediment.capacity.grain_sizes_m.'+name,
                    'Enter measured '+LABELS[name]+' in mm in 04 / General / Grain sizes. No diameter is inferred.')
            sizes[name] = value
        s["capacity"] = dict(method=method,parameters=parameters,grain_sizes_m=sizes)
        try:validate_grain_sizes(s['capacity'])
        except ValueError as exc:raise ModelError('incompatible_grain_diameter','sediment.capacity.grain_sizes_m',str(exc)) from exc
        s.update(transport='split' if self.suspension_on.isChecked() else
                     self.route.currentData() if self.bedload_on.isChecked() else 'bedload', fraction_basis=self.basis.currentData(),
                 grains=[dict(id=_text(self.grains, r, 0), diameter_definition=('class_representative' if independent_suspended else 'd50') if suspended_only else self._diameter_definition(r),
                              diameter_m=(self.transport_setup()['diameters_m']['ds'] if independent_suspended else sizes['d50']) if suspended_only else self._numeric_table(self.grains,r,1,'grains')/1000,
                              solid_density_kg_m3=self._numeric_table(self.grains,r,2,'grains'),
                              fraction=self._numeric_table(self.grains,r,3,'grains')) for r in range(self.count.value())],
                 capacity_hiding=self.hiding.currentData() if self.hiding_on.isChecked() else "auto",
                 slope_hiding=self.slope_hiding.currentData() if self.slope_on.isChecked() and
                     self.slope.currentData() == "bed_slope.delft_parker_andrews" else "hiding.none",
                 bed_slope=self.slope.currentData() if self.slope_on.isChecked() else "bed_slope.off",
                 shear=self.shear.currentData(),
                 suspension='suspension.settling_only' if supplied_suspended else self.suspended_formula.currentData().replace('suspended.','suspension.') if classic_suspended else 'suspension.de_leeuw_2020' if dl_suspended else 'suspension.garcia_parker_1991' if gp_suspended else 'suspension.van_rijn_exchange' if vr_suspended else self.suspension.currentData() if self.suspension_on.isChecked() else "off",
                 bedforms="bedform.enabled" if self.bedforms_on.isChecked() else "off",
                 banks=self.bank_method.currentData() if self.banks_on.isChecked() else "off",
                 avalanche=self.avalanche.currentData() if self.avalanche_on.isChecked() else "off")
        s["bed"] = dict(composition=self.bed_mode.currentData(),
                        transport_layer=self.transport_layer.currentData() if self.bed_mode.currentData() == "bed.layered" else "off",
                        coarse_layer=self.bed_mode.currentData() == "bed.layered" and self.coarse_on.isChecked(),
                        sorting=self.sorting.currentData() if self.bed_mode.currentData() == "bed.layered" and self.sorting_on.isChecked() else "sorting.off")
        return s

    def model(self, backend, roughness, secondary, turbulence):
        self.validate_choices()
        model = new_model(backend=backend)
        model["hydrodynamics"] = dict(roughness=roughness,
            secondary_flow="secondary.prognostic" if secondary else "secondary.off",
            turbulence="constant_eddy_viscosity" if turbulence else "off")
        model["sediment"] = dict(enabled=self.active,
                                  selection=self.selection() if self.active else None)
        return resolve_model(model).document()

    def layer_values(self):
        thickness = [self._numeric_table(self.layers,r,0,'bed.layers') for r in range(self.layers.rowCount())]
        fractions = [[self._numeric_table(self.layers,r,c+1,'bed.layers') for c in range(self.count.value())]
                     for r in range(self.layers.rowCount())]
        return thickness, fractions

    def dry_densities(self):
        return [self._numeric_table(self.grains,r,4,'grains') for r in range(self.count.value())]

    def phi_spreads(self):
        return [self._numeric_table(self.grains,r,5,'grains') for r in range(self.count.value())]

    def editor_state(self):
        """Inactive drafts are UI data, never interpreted by the worker."""
        self._parameters[self._current_capacity] = {k: w.text() for k, w in self.parameters.items()}
        values = {}
        for key, widget in self.widgets.items():
            values[key] = (widget.isChecked() if isinstance(widget, QCheckBox) else
                           widget.currentData() if isinstance(widget, QComboBox) else widget.value())
        self._remember_layers()
        return dict(schema=13, widgets=values, parameters=deepcopy(self._parameters),
            transport_setup=self.transport_setup_snapshot(),
            grain_sizes=deepcopy(self._diameter_values),
            layer_drafts=deepcopy(self._layer_drafts),
            grains=[[_text(self.grains, r, c) for c in range(7)] for r in range(self.grains.rowCount())],
            layers=[[_text(self.layers, r, c) for c in range(self.layers.columnCount())]
                    for r in range(self.layers.rowCount())])

    def restore_editor(self, state, *, enabled):
        if state.get("schema") != 13 or set(state) != {"schema", "widgets", "parameters", "grains", "layers", "layer_drafts", "grain_sizes", "transport_setup"} or 'bedload_on' not in state.get('widgets',{}):
            raise ValueError("Current sediment workflow required. Create a new project; old editor formats are not migrated.")
        if any(len(row) != 7 for row in state["grains"]):
            raise ValueError("Explicit grain diameter basis, densities and spread required; old drafts are not migrated")
        if set(state["widgets"]) != set(self.widgets):
            raise ValueError("Sediment editor controls differ")
        self._loading = True
        try:
            if not isinstance(state['grain_sizes'], dict) or any(k not in DIAMETER_LABELS or not isinstance(v,str) for k,v in state['grain_sizes'].items()):
                raise ValueError('Explicit measured grain-size editor values required')
            self._diameter_values = deepcopy(state['grain_sizes'])
            self._diameter_rows = ()
            for key, widget in self.widgets.items():
                if key in ("capacity", "fractions", "layer_count", "enabled"):
                    continue
                value = state["widgets"][key]
                if isinstance(widget, QCheckBox):
                    if type(value) is not bool:
                        raise ValueError("Invalid checkbox state")
                    widget.setChecked(value)
                elif isinstance(widget, QComboBox):
                    set_choice(widget, value)
                else:
                    widget.setValue(value)
            self._grain_memory = {r: row for r, row in enumerate(state["grains"])}
            self.grains.setRowCount(0)
            self.count.setValue(state["widgets"]["fractions"])
            self._resize_grains()
            self._loading = True
            self._layer_mode = self._topology_key()
            self._layer_drafts = deepcopy(state["layer_drafts"])
            self._layer_memory = {(r, c): value for r, row in enumerate(state["layers"]) for c, value in enumerate(row)}
            self.layers.setRowCount(0)
            self.layer_count.setRange(1, 1) if self._layer_mode == "bed.fully_mixed" else self.layer_count.setRange(
                self._minimum_layers(), 32)
            self.layer_count.setValue(state["widgets"]["layer_count"])
            self._resize_layers()
            # Drop the previous method before loading, so its signal cannot
            # overwrite the saved values with those of the preceding project.
            self._current_capacity = None
            self._parameters = deepcopy(state["parameters"])
            self.capacity.blockSignals(True)
            set_choice(self.capacity, state["widgets"]["capacity"])
            self.capacity.blockSignals(False)
            self._capacity_changed()
            self.enabled.setChecked(enabled)
        finally:
            self._loading = False
        self._refresh()
        self.changed.emit()

    def load_project(self, project):
        """Populate from authoritative physics; drafts never override it."""
        s = project["model"]["physics"]["sediment"]
        self.bank_survey.restore(project['bank_survey'] if project['bank_survey'] is not None else
            project.get('editor_state',{}).get('bank_survey'))
        if "editor_state" in project:
            saved=project["editor_state"]["sediment"]
            self.restore_editor(saved, enabled=s is not None or saved['widgets']['enabled'])
            if s is None:
                return
        elif s is None:
            self.enabled.setChecked(False)
            for w in (self.bedload_on,self.suspension_on,self.banks_on,self.avalanche_on,self.bedforms_on,
                      self.slope_on,self.hiding_on,self.sorting_on,self.coarse_on,self.exchange_on):w.setChecked(False)
            return
        state = self.editor_state()
        w = state["widgets"]
        inactive_capacity=w['capacity']
        w.update(capacity=s["capacity"]["method"], route='bedload' if s['transport']=='split' else s['transport'],
            bedload_on=bool(project['settings']['suspension']['bed_factor']), fractions=len(s["grains"]),
            basis=s["fraction_basis"], slope_on=s["bed_slope"] != "bed_slope.off",
            hiding_on=s["capacity_hiding"] not in ("hiding.none", "hiding.intrinsic"),
            slope_hiding=s["slope_hiding"], shear=s["shear"], suspension_on=s["suspension"] != "off",
            bed_mode=s["bed"]["composition"], coarse_on=s["bed"]["coarse_layer"],
            sorting_on=s["bed"]["sorting"] != "sorting.off", bedforms_on=s["bedforms"] != "off",
            banks_on=s["banks"] != "off", avalanche_on=s["avalanche"] != "off",
            layer_count=len(project["initial"]["layer_thickness_m"]))
        w.update({"slope_"+k: v for k, v in project["slope_parameters"].items()})
        w.update({"vr_reference_"+k: v for k,v in project["van_rijn_reference_parameters"].items()})
        w.update({"skin_"+k: v for k,v in project["skin_shear_parameters"].items()})
        w.update(wet_slope=project["wet_avalanche_parameters"]["wet_slope"],
                 avalanche_time=project["wet_avalanche_parameters"]["relaxation_time_s"])
        w.update({"dependent_"+k: v for k, v in project["dependent_layer_parameters"].items()})
        columns = project["settings"]["columns"]
        w.update(top_update=project["column_methods"]["top"], base_update=project["column_methods"]["base"],
                 porosity=project["column_methods"]["porosity"],
                 porosity_value=1.-columns["solid_fraction"],
                 sand_porosity=1.-columns["sand_solid_fraction"], mud_porosity=1.-columns["mud_solid_fraction"],
                 exchange_on=columns["exchange_thickness_m"] > 0,
                 exchange_thickness=columns["exchange_thickness_m"] or w["exchange_thickness"],
                 lagrangian_count=columns["lagrangian_underlayers"],
                 lagrangian_thickness=columns["lagrangian_thickness_m"] or w['lagrangian_thickness'])
        w.update(active_thickness=columns["transport_thickness_m"], depth_fraction=columns["depth_fraction"],
                 minimum_thickness=columns["minimum_thickness_m"],
                 substrate_thickness=columns["eulerian_thickness_m"] or columns["underlayer_thickness_m"])
        for channel, parameters in project["hiding_parameters"].items():
            w.update({"hiding_"+channel+"_"+k: v for k, v in parameters.items()})
        for flag, name, value in ((w["slope_on"], "slope", s["bed_slope"]),
            (w["hiding_on"], "hiding", s["capacity_hiding"]),
            (w["suspension_on"], "suspension", s["suspension"]),
            (w["sorting_on"], "sorting", s["bed"]["sorting"]),
            (s["bed"]["transport_layer"] != "off", "transport_layer", s["bed"]["transport_layer"]),
            (w["avalanche_on"], "avalanche", "redistribution.jiang_cao_repose200_v1" if s["avalanche"]=="redistribution.jiang_cao_repose_exner223_v1" else s["avalanche"])):
            if flag:
                w[name] = value
        if w['banks_on']:w['bank_method']=s['banks']
        if project['bank_survey'] is not None:
            w['bank_theta']=project['bank_survey']['fraction']
            if project['bank_survey']['critical_undercut_m'] is not None:
                w['bank_critical']=project['bank_survey']['critical_undercut_m']
        state["parameters"][w["capacity"]] = {k: str(v) for k, v in s["capacity"]["parameters"].items()}
        from braids_core.sediment.river_classic_suspension import METHODS,FIELDS
        if s['suspension'] in METHODS:
            w['suspended_formula']=s['suspension'].replace('suspension.','suspended.')
            if not w['bedload_on']:w['capacity']=inactive_capacity
            for key,field in FIELDS.items():
                if field in project['classic_suspension_parameters']:
                    state['grain_sizes'][key]=str(project['classic_suspension_parameters'][field]*1000.)
        elif s['suspension']=='suspension.settling_only':
            w['suspended_formula']='suspended.settling_only'
            if not w['bedload_on']:w['capacity']=inactive_capacity
            state['grain_sizes']['ds']=str(project['supplied_suspension_parameters']['suspended_diameter_m']*1000.)
        elif s['suspension']=='suspension.garcia_parker_1991':
            w['suspended_formula']='suspended.garcia_parker_1991'
            if not w['bedload_on']:w['capacity']=inactive_capacity
            state['grain_sizes']['ds']=str(project['garcia_parker_parameters']['suspended_diameter_m']*1000.)
        elif s['suspension']=='suspension.de_leeuw_2020':
            w['suspended_formula']='suspended.de_leeuw_2020'
            if not w['bedload_on']:w['capacity']=inactive_capacity
            for key,field in (('ds','suspended_diameter_m'),('d84','bed_d84_m')):
                state['grain_sizes'][key]=str(project['de_leeuw_parameters'][field]*1000.)
        elif w['capacity']=='capacity.van_rijn_1993' and w['suspension_on']:
            w['suspended_formula']='suspended.van_rijn_1993_flow2dh'
            state['grain_sizes']['ds']=str(project['van_rijn_reference_parameters']['suspended_diameter_m']*1000.)
        for key,value in s['capacity']['grain_sizes_m'].items():
            try:equal = float(state['grain_sizes'].get(key,'')) == value*1000
            except ValueError:equal = False
            if not equal:state['grain_sizes'][key] = str(value*1000)
        old_grains,old_layers=state['grains'],state['layers']
        state["grains"] = [[g["id"], str(g["diameter_m"]*1000), str(g["solid_density_kg_m3"]), str(g["fraction"]),
                            str(columns["dry_density_kg_m3"][k]),
                            str(columns["phi_std"][k] if columns["phi_std"] else 0.),
                            diameter_contract(s['capacity']['method'], len(s['grains'])).label
                            if g['diameter_definition'] == diameter_contract(s['capacity']['method'], len(s['grains'])).definition
                            else g['diameter_definition']] for k, g in enumerate(s["grains"])]
        state["layers"] = [[str(t), *map(str, f)] for t, f in
                           zip(project["initial"]["layer_thickness_m"], project["initial"]["layer_fractions"])]
        # Preserve valid exact editor text; changed active physics still wins.
        for old,rows in ((old_grains,state['grains']),(old_layers,state['layers'])):
            for r,row in enumerate(rows):
                for c,value in enumerate(row):
                    try:
                        if float(old[r][c])==float(value):rows[r][c]=old[r][c]
                    except (ValueError,IndexError):pass
        bank, sus = project["settings"]["bank"], project["settings"]["suspension"]
        w.update(bank_theta=bank["bank_theta"] if s['banks']=='redistribution.dry_bank' else w["bank_theta"],
            collapse_slope=bank["collapse_slope"], repose_angle_deg=bank["repose_angle_deg"], temperature=sus["temperature"],
            cref=sus["cref"])
        if w["suspension_on"]:
            w.update(sus_factor=sus["sus_factor"], sus_diffusion=sus["diffusivity"])
        self.restore_editor(state, enabled=True)

