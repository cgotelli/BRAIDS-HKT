"""Desktop layout and first-error guidance; the runtime remains admission authority."""
from PySide6.QtCore import Qt,QTimer
from PySide6.QtWidgets import QWidget,QVBoxLayout,QFormLayout,QLabel,QTabWidget,QPushButton,QSizePolicy
from .river_physics import section


STEPS=('Sediment transport','Morphological evolution','Sediment transport BC')


def note(layout,text):
    label=QLabel(text);label.setWordWrap(True);label.setObjectName('note');layout.addWidget(label)
    return label


def move_row(source,widget,target,label=None):
    row=source.takeRow(widget)
    old=row.labelItem.widget() if row.labelItem else None
    if old is not None:
        if label is None:label=old.text()
        old.deleteLater()
    if label:target.addRow(label,widget)
    else:target.addRow(widget)


def build(forms):
    """One bulk-material workflow; S1 suspended proposals are not run admission."""
    from .river_physics import number
    p=forms.physics
    def tabs(step,names):
        widget=QTabWidget();widget.setDocumentMode(True)
        forms.layouts[step].addWidget(widget);layouts={}
        for name in names:
            page=QWidget();layouts[name]=QVBoxLayout(page);widget.addTab(page,name)
        return widget,layouts
    forms.sediment_tabs,transport=tabs(STEPS[0],('General','Bedload','Suspended','Summary'))
    forms.sediment_tabs.setStyleSheet('QTabBar::tab { padding: 6px 7px; }')
    forms.morphology_tabs,morph=tabs(STEPS[1],('Erodible bed','Bank erosion','Bank and slope collapse'))
    # The unused editors remain private, inactive draft storage, not pages or
    # selectable physical routes. validate_choices rejects active hidden physics.
    p.sediment_page.hide();p.bed_page.hide()
    general=transport['General'];general.addWidget(p.bedload_on)
    general.addWidget(p.suspension_on)
    p.suspension_on.setText('Enable suspended transport')
    start_form=QFormLayout();general.addLayout(start_form)
    forms.sediment_general_form=start_form
    forms.sediment_start=number(0.,0.,1e12)
    start_form.addRow('Start sediment transport at (s)',forms.sediment_start)
    forms.sediment_start.valueChanged.connect(forms._changed)
    note(general,'Before this physical time, water flows over a fixed bed. At this time bedload and '
         'conservative bed evolution start. Zero starts both together.')
    general.addWidget(p.material_box)
    p.material_box.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Maximum)
    def fit_material_tables():
        for table in (p.grain_sizes,p.grains):
            header=table.horizontalHeader().sizeHint().height()
            height=header+sum(table.rowHeight(i) for i in range(table.rowCount()) if not table.isRowHidden(i))+2*table.frameWidth()+4
            table.setFixedHeight(min(200,max(header+4,height)))
    def refit_material_notes():
        for label in (p.grain_preset,p.diameter_note):label._fit_text()
        fit_material_tables()
        p.material_box.content.updateGeometry();p.material_box.updateGeometry()
    p.material_box.toggle.toggled.connect(lambda expanded:QTimer.singleShot(0,refit_material_notes) if expanded else None)
    for table in (p.grain_sizes,p.grains):
        table.model().rowsInserted.connect(lambda *_:QTimer.singleShot(0,fit_material_tables))
        table.model().rowsRemoved.connect(lambda *_:QTimer.singleShot(0,fit_material_tables))
    material=p.material_box.body_layout.itemAt(0).layout()
    for w in (p.count,p.basis,p.grain_note,p.composition_note):material.setRowVisible(w,False)
    for col in (0,3,5):p.grains.setColumnHidden(col,True)
    fit_material_tables()
    note(general,'One bulk sediment material. Each required diameter is an independent measurement, not a transported class. Dry bulk density = solid density × (1 − porosity); '
         '2650 and 1590 kg/m³ correspond to porosity 0.40.')
    bed=transport['Bedload'];bed.addWidget(p.transport_box)
    source=p.transport_box.body_layout.itemAt(0).layout()
    for w in (p.route,p.route_note,p.temperature):source.setRowVisible(w,False)
    from .bedload_catalog import CATALOG
    p.capacity.blockSignals(True);p.capacity.clear()
    for method,row in CATALOG.items():p.capacity.addItem(row['label'],method)
    p.capacity.blockSignals(False)
    p.capacity_note=note(bed,CATALOG['capacity.mpm_braids_5_055']['note']);p.capacity_note.hide();p._capacity_changed()
    bed_note=note(bed,'Bedload coefficients apply only to bedload. Independent suspension is selected in Suspended. Ordinary formulas use capacity/Exner; '
         'the named Jiang–Cao research choice owns mobile bedload and its auxiliary methods. '
         'Read Summary before preparation. Collapse remains an explicit choice in Morpho evolution.');bed_note.hide()
    suspended=transport['Suspended']
    p.suspended_plan_box,susp=section(suspended,'Suspended formulation',expanded=True)
    susp.addRow('Formula',p.suspended_formula)
    p.suspended_plan_note=QLabel();p.suspended_plan_note.setWordWrap(True)
    susp.addRow(p.suspended_plan_note)
    move_row(source,p.temperature,susp,'Water temperature (°C)')
    p.temperature.show()
    source=p.suspension_box.body_layout.itemAt(0).layout()
    move_row(source,p.sus_diffusion,susp,'Horizontal sediment diffusivity (m²/s)')
    move_row(source,p.reference_concentration,susp,'Hindering reference concentration (kg/m³)')
    note(suspended,'Van Rijn 1993 FLOW 2DH runs alone or with matching Van Rijn bedload. '
         'Supplied suspension (no entrainment), García–Parker 1991, de Leeuw 2020, Van Rijn 1984, Zyserman–Fredsøe 1994 and current-only Camenen–Larson 2008 run alone or with any ordinary bedload formula (not Jiang–Cao). '
         'Positive morphological factors are supported; sediment start=0. MF scales bed exchange, not the suspended clock. Independent closures use clear-water settling; the hindering field is inactive. '
         'Set initial and inlet concentration in Sediment transport BC. '
         'Conservative wet/dry collapse is available with these formulas; accepted-demand bank/toe supply is available.')
    forms.resolved_box,resolved=section(transport['Summary'],'Sediment setup',expanded=True)
    forms.resolved_setup=QLabel();forms.resolved_setup.setWordWrap(True)
    forms.resolved_setup.setProperty('preserveMethodIds',True)
    forms.resolved_setup.setTextInteractionFlags(Qt.TextSelectableByMouse);resolved.addRow(forms.resolved_setup)
    # These are already separate tabs.  A second accordion inside each tab
    # only wastes space and makes the active physical choice ambiguous.
    note(morph['Erodible bed'],'Initial thickness limits the available sediment above the non-erodible base. '
         'Bedload transport divergence updates the bed automatically. One bulk material, one fully mixed finite bed; '
         'no substrate or sorting.')
    stock=QFormLayout();morph['Erodible bed'].addLayout(stock)
    move_row(p.inventory_box.body_layout.itemAt(0).layout(),p.layers,stock)
    p.layers.setColumnHidden(1,True)
    p.layers.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
    def fit_erodible_table():
        header=p.layers.horizontalHeader().sizeHint().height()
        rows=sum(max(32,p.layers.rowHeight(row)) for row in range(p.layers.rowCount()))
        p.layers.setFixedHeight(header+rows+2*p.layers.frameWidth()+4)
    fit_erodible_table()
    p.layers.model().rowsInserted.connect(lambda *_:QTimer.singleShot(0,fit_erodible_table))
    p.layers.model().rowsRemoved.connect(lambda *_:QTimer.singleShot(0,fit_erodible_table))
    forms.bank_erosion_box=QWidget();morph['Bank erosion'].addWidget(forms.bank_erosion_box)
    banks=QFormLayout(forms.bank_erosion_box)
    source=p.bank_box.body_layout.itemAt(0).layout()
    for w in (p.banks_on,p.bank_method,p.bank_theta,p.bank_critical,p.bank_note,p.bank_survey):move_row(source,w,banks)
    forms.collapse_box=QWidget();morph['Bank and slope collapse'].addWidget(forms.collapse_box)
    collapse=QFormLayout(forms.collapse_box)
    source=p.bank_box.body_layout.itemAt(0).layout()
    for w in (p.avalanche_on,p.avalanche,p.collapse_slope,p.repose_angle_deg):move_row(source,w,collapse)
    p.avalanche.removeItem(p.avalanche.findData('redistribution.avalanche'))
    note(morph['Bank and slope collapse'],'BRAIDS conservative wet/dry redistribution uses the collapse slope (m/m) '
        'with ordinary bedload and/or suspension. MF scales transport and wet exchange, not collapse. Repose projection also supports ordinary formulas with the angle above. The coupled Jiang–Cao version uses the bundle’s '
        'repose angle (degrees), conserving mobile/bed solid and pore water. These are different '
        'algorithms, not two corrections to apply together. Ordinary collapse equalizes a triggered neighbour pair; Jiang-Cao projects to repose. Neither is a geotechnical block-failure model.')
    forms.morphology_general_layout=morph['Erodible bed']
    p._refresh()
    for layout in (*transport.values(),*morph.values()):layout.addStretch()
    bc=forms.layouts[STEPS[2]]
    button=QPushButton('Edit exterior bed',forms);button.hide()
    forms.sediment_bc_button=button
    note(bc,'Set sediment supply only at inlets. Outlets export arriving bedload and never import it. '
         'Exterior bed/composition are resolved automatically and saved in JSON. Sediment boundaries activate after warm-up. '
         'Jiang–Cao research bundle: use constant qs/q only; hydraulic Q denotes mixture volume flux, '
         'with water flux Q/(1 + qs/q). It starts at time zero.').setText(
             'Set sediment supply only at inlets. Outlets export arriving bedload and never import it.')


def first_error(exc):
    """One runtime exception, one location and a concrete corrective action."""
    text=str(exc);key=(str(getattr(exc,'path',''))+' '+text).lower()
    rules=(
        (('execution.integrator','fast hllc','hllc bedload'),'Run / Hydraulic method','HLLC supports ordinary bedload and suspended formulas under their physical compatibility rules. The separate Jiang–Cao mixture model requires GeoClaw augmented f-wave. No selected feature is disabled automatically.'),
        (('turbine','rotor','near wake','near-wake'),'Hydrokinetic turbines','Correct the named turbine height, clearance, placement or device/wake setting as explained above. Rotor overlap, force-footprint overlap and the wake model spacing limit are different checks.'),
        (('initial.stage','initial.velocity'),'BC & IC / Initial conditions','Choose Dry bed or a finite water elevation and X/Y velocity. Elevation uses the DEM vertical datum; velocity applies only where initially wet.'),
        (('sediment.setup',),'Sediment transport / General, Suspended and Summary',
         'Correct the named measurement or combination. Van Rijn 1993 runs alone or with matching bedload; the other suspended choices allow ordinary bedload. Positive MF and start=0; no formula is replaced automatically.'),
        (('grain_sizes_m',),'Sediment transport / General / Grain sizes','Enter every required measured diameter in mm. Keep percentile sizes ordered; Dm is a mean, not a percentile. Do not add extra transport fractions.'),
        (('jiang_cao',),'Sediment transport / Bedload and Morpho evolution','Follow the named bundle requirement above. Keep MF 1, start time 0, explicit D50 and D90, constant qs/q inlet feed and zero FLOW Betac/diffusion/eddy viscosity. Select automatic Sun/van Rijn resistance; do not add external slope or roughness corrections.'),
        (('sediment_start_time',),'Sediment transport / General','Set the start time between 0 and the total simulation time in Run. Use 0 to start sediment immediately.'),
        (('cfl',),'Run','Set the hydraulic face-sum CFL between 0 and 1. Physical stability checks may shorten steps.'),
        (('sus_factor','entrainment factor'),'Morpho evolution / Suspended','Set the entrainment switch to 0 or 1. Settling-only resolves it to zero and retains the exchange draft inactive.'),
        (('hydraulics','energy_slope','inlet.','outlet.'),'BC & IC','Open the named inlet/outlet and correct the reported hydraulic value. For normal flow, review the signed inward bed slope and boundary geometry.'),
        (('survey','sediment.banks'),'Morpho evolution / General','Use ordinary bedload and/or suspended transport with a finite mixed bed. Complete the bank polygon and toe survey when selected. The coupled Jiang–Cao bundle cannot use this Exner bank supply.'),
        (('sediment.bed.sorting','sorting requires'),'Morpho evolution / Bedload','Enable a layered bed and its coarse layer in General, or disable sorting.'),
        (('dune','bedforms'),'Morpho evolution','Enable bedform feedback for dune-scaled thickness, or choose constant/water-depth thickness. Bedform fractions must have equal solid densities.'),
        (('suspension','transport route','sediment.transport'),'Sediment transport / General','Choose bedload, suspended mass, or both. Independent suspension cannot accompany algebraic total load. For suspended exchange, select a capacity with a suspended channel, or use supplied-suspension settling.'),
        (('bed_slope','slope_hiding'),'Sediment transport / Bedload','Enable bedload for bed-slope correction. Parker–Andrews alone consumes separate slope hiding; choose a compatible slope/hiding combination.'),
        (('hiding','intrinsic'),'Sediment transport / General','Disable optional external hiding for a formula with intrinsic mixture/hiding dependence; keep its required intrinsic correction.'),
        (('shear','skin'),'Sediment transport / General','Select total hydraulic shear, or an intrinsic formula that supports Soulsby–Clarke skin stress.'),
        (('grains','grain','fraction','density'),'Sediment transport / General','Use one bulk material and enter every measured diameter requested by the formula. Percentiles must be ordered; Dm is a mean. No sizes are inferred and no extra transport fractions are needed. Enter positive densities; dry bulk density must not exceed solid density.'),
        (('capacity','coefficient','parameter'),'Sediment transport / General','Enter the named coefficient with the required finite value/range in Transport. Keep the selected formula’s required physical combination.'),
        (('boundary','feed','concentration','schedule'),'Sediment transport BC','Correct the named boundary value/table. Use nonnegative feeds/concentrations, strictly increasing times starting at 0 and covering the end time. Hydraulic conditions remain in BC & IC.'),
        (('column','layer','thickness','porosity'),'Morpho evolution / General','Correct the named bed/layer input: positive thicknesses, normalized layer fractions, and a topology compatible with the chosen bed representation.'),
    )
    for tokens,where,solution in rules:
        if any(token in key for token in tokens):break
    else:
        where='Model settings';solution='Correct the named field using the requirement above, then press Prepare again.'
    category='Current integration restriction.\n' if getattr(exc,'category',None)=='integration_pending' else ''
    return f'Preparation stopped before mesh construction.\n\n{category}{text}\n\nWhere: {where}\nHow to fix: {solution}\n\nOnly the first error is shown. Correct it and press Prepare again.'
