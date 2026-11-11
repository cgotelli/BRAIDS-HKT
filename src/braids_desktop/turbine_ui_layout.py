"""Compact turbine editor presentation; project and numerical contracts unchanged."""
import math
from pathlib import Path
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QTabWidget,QPushButton,QCheckBox,QSizePolicy

GUIDE=Path(__file__).parent/'docs/hydrokinetic_turbines.html'


class CompactTabs(QTabWidget):
    """Size the editor for the selected page, rather than its tallest page."""
    def sizeHint(self):
        size=super().sizeHint();page=self.currentWidget()
        if page is not None:size.setHeight(page.sizeHint().height()+self.tabBar().sizeHint().height()+8)
        return size

    def minimumSizeHint(self):return self.sizeHint()

def install(w,root,old_form,note,wake_note):
    note.hide();wake_note.hide()
    w.summary.setText('Set the parameters, then Add turbine. First click: rotor centre. Second click: DOWNSTREAM (towards the outlet), not upstream.')
    root.removeWidget(w.add_turbine)
    w.add_turbine.setObjectName('Primary')
    index=next(i for i in range(root.count()) if root.itemAt(i).widget() is w.edit_turbine)
    root.removeWidget(w.edit_turbine);root.removeWidget(w.remove_turbine)
    row=QHBoxLayout();row.addWidget(w.edit_turbine);row.addWidget(w.remove_turbine);root.insertLayout(index,row)
    root.insertWidget(index+1,w.add_turbine)
    w.edit_turbine.setText('Edit position');w.remove_turbine.setText('Remove')
    w.add_turbine.setToolTip('Place another turbine using the current settings. Parameters are retained after placement.')
    w.guide_button=QPushButton('How turbines work / parameter guide')
    w.guide_button.clicked.connect(lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(str(GUIDE))))
    w.guide_button.hide()  # The same guide is available from Help.
    w.detail_tabs=CompactTabs();w.detail_tabs.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Maximum)
    w.detail_tabs.currentChanged.connect(lambda _:w.detail_tabs.updateGeometry());layouts={};forms={}
    for name in ['Placement','Device','Wake & mesh']:
        page=QWidget();layouts[name]=QVBoxLayout(page);forms[name]=QFormLayout()
        forms[name].setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        forms[name].setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        layouts[name].addLayout(forms[name]);w.detail_tabs.addTab(page,name.replace('&','&&'))
    device={w.control,w.performance_model,w.ct,w.cp,w.device_yaw}
    wake={w.nu,w.wake_enabled,w.wake_length,w.power_mode,w.power_growth,w.power_width,w.power_spacing,w.power_provenance}
    renames={w.yaw:'Axis offset from reference (°)',w.orientation:'Rotor orientation',
        w.hub:'Hub height above bed (m)',w.bed_clearance:'Minimum bed clearance (m)',
        w.surface_clearance:'Minimum surface clearance (m)',w.wake_length:'Refined wake length (D)',
        w.nu:'Ambient eddy viscosity (m²/s)',w.power_mode:'Power calculation',
        w.power_growth:'Wake growth k',w.power_width:'Initial width ε (σ / D)',
        w.power_spacing:'Minimum model spacing (D)',w.power_provenance:'Wake parameter source'}
    while old_form.rowCount():
        row=old_form.takeRow(0);field=row.fieldItem.widget();label=row.labelItem.widget() if row.labelItem else None
        if isinstance(field,QLabel):
            field.hide()
            if label:label.hide()
            continue
        group='Device' if field in device else 'Wake & mesh' if field in wake else 'Placement'
        if label:
            if field in renames:label.setText(renames[field])
            if field is w.performance_model:w._model_label=label
            forms[group].addRow(label,field)
        else:forms[group].addRow(field)
    index=next(i for i in range(root.count()) if root.itemAt(i).layout() is old_form)
    root.removeItem(old_form);root.insertWidget(index,w.detail_tabs)
    for widget in [w.table,w.import_table,w.add_curve_row,w.remove_curve_row,w.curves]:
        root.removeWidget(widget);layouts['Device'].addWidget(widget)
    w.orientation.setItemText(0,'Follows the flow (self-aligning)')
    w.orientation.setItemText(1,'Fixed axis')
    w.power_mode.setItemText(0,'Resolved flow (Thetis)')
    w.power_mode.setItemText(1,'Legacy aligned wake correction')
    w.power_mode.setItemText(2,'Dou wake correction (aligned / yawed)')
    w.compatibility=QCheckBox('Show legacy power option');layouts['Wake & mesh'].addWidget(w.compatibility)
    w.compatibility.toggled.connect(lambda:refresh(w))
    w.use_advanced_mixing=QPushButton('Use viscosity from Advanced hydrodynamics')
    w.use_advanced_mixing.setToolTip('Sets turbine ambient viscosity to zero to avoid adding mixing twice. Enable horizontal turbulent viscosity and set its value in Advanced hydrodynamics.')
    w.use_advanced_mixing.clicked.connect(lambda:w.nu.setValue(0.))
    layouts['Wake & mesh'].addWidget(w.use_advanced_mixing)
    mixing_note=QLabel('Secondary flow is compatible. Set viscosity in only one extension; the button above selects the Advanced hydrodynamics input.')
    mixing_note.setWordWrap(True);layouts['Wake & mesh'].addWidget(mixing_note)
    w.direction_status=QLabel();w.direction_status.setWordWrap(True);layouts['Placement'].addWidget(w.direction_status)
    w.wake_status=QLabel();w.wake_status.setWordWrap(True);layouts['Wake & mesh'].addWidget(w.wake_status)
    for layout in layouts.values():layout.addStretch()
    w.yaw.setToolTip('Fixed mounting offset from the downstream reference you draw. Positive rotates the rotor axis counterclockwise. This is not operating yaw: yaw is computed from the simulated incident flow.')
    w.power_growth.setToolTip('Dimensionless wake spreading rate in sigma = epsilon D + k x. Use independent wake data; it is not derived automatically from eddy viscosity.')
    w.power_width.setToolTip('Initial analytical wake width divided by rotor diameter. Must be supported together with k by independent wake data.')

def refresh(w):
    if not hasattr(w,'detail_tabs'):return
    for widget in [w.resolved,w.power_note,w.curve_note,w.wake_summary]:widget.hide()
    fixed=w.orientation.currentData()=='fixed_axis'
    if w.rows:
        row=w.rows[w.current if w.current>=0 else 0]
        a=math.degrees(math.atan2(row['river_direction_xy'][1],row['river_direction_xy'][0]))%360
        w.direction_status.setText(f'Downstream reference: {a:.1f}° from east. '+
            (f'Fixed rotor axis: {(a+w.yaw.value())%360:.1f}°. Operating yaw is computed from the flow.' if fixed else 'Rotor follows the local flow.'))
    else:w.direction_status.setText('Operating yaw is computed from the flow; a fixed rotor needs its mounting direction.')
    legacy=w.power_mode.currentData()=='power.gaussian_upstream_reference.v1'
    w.power_mode.view().setRowHidden(1,not (legacy or w.compatibility.isChecked()))
    dou=w.power_mode.currentData()=='power.dou2019_device_yaw.v1'
    w.wake_status.setText('Upstream wakes: detected automatically from placement and flow. '+
        ('Yawed-array experimental validation is pending.' if dou else 'See the guide for the selected power model.'))
    value=w.device_yaw.value()
    lookup=w.control.currentData()!='constant_ct'
    w.performance_model.setVisible(lookup);w._model_label.setVisible(lookup)
    w.power_provenance.setToolTip(w.power_provenance.text())
    if value:
        w.device_yaw.summary.setToolTip(f"{value['provenance']}\n{value['physical_scope']}")
        w.device_yaw.summary.setText(f"{value['model_name']}\n{value['diameter_m']:g} m · {value['rpm']:g} RPM · {value['speed_m_s'][0]:g}–{value['speed_m_s'][-1]:g} m/s · |yaw| ≤ {value['yaw_deg'][-1]:g}°")
    else:w.device_yaw.summary.setText('Import data for the actual rotor and controller. Stored with this project.')
