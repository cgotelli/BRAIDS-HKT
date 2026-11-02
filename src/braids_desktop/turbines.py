"""Point-and-river-direction placement; the rotor is always cross-stream."""
from copy import deepcopy
import csv,math
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget,QVBoxLayout,QFormLayout,QCheckBox,QLineEdit,QPushButton,QTableWidget,QTableWidgetItem,QHeaderView,QLabel,QFileDialog,QComboBox,QMessageBox
from .river_physics import number,combo,set_choice
from braids_core import river_turbines as model


class TurbineEditor(QWidget):
    changed=Signal();action=Signal(str)
    def __init__(self,parent=None):
        super().__init__(parent);self.rows=[];self.loading=False;self.current=-1;self._capture_edit=None
        self.enabled=QCheckBox('Enable hydrokinetic turbines',self);self.enabled.hide()
        root=QVBoxLayout(self);root.setContentsMargins(0,0,0,0)
        self.title=QLabel('Hydrokinetic turbines');self.title.setObjectName('Title');root.addWidget(self.title)
        self.power_planning=QCheckBox('Estimates only - no turbine forces in the flow');self.power_planning.setChecked(False);root.addWidget(self.power_planning)
        note=QLabel('Default: turbines exert thrust on the hydraulic flow, so wakes appear in the velocity map and affect downstream turbines. Power and thrust use the selected device data. The optional estimates-only mode disables those forces; it is a separate reference-flow calculation, not a turbine simulation.');note.setWordWrap(True);root.addWidget(note)
        self.power_planning.toggled.connect(self._changed)
        self.summary=QLabel('Add rotors, then select a turbine to edit its individual settings. Click its centre, then point downstream; the rotor stays perpendicular to that river direction.');self.summary.setWordWrap(True);root.addWidget(self.summary)
        self.selection=QComboBox();root.addWidget(self.selection);self.selection.currentIndexChanged.connect(self._select)
        for label,key in [('Add turbine','add_turbine'),('Reposition / orient turbine','edit_turbine'),('Remove turbine','remove_turbine')]:
            button=QPushButton(label);button.clicked.connect(lambda checked=False,k=key:self.action.emit(k));root.addWidget(button);setattr(self,key,button)
        form=QFormLayout();root.addLayout(form)
        self.name=QLineEdit('Turbine 1');self.diameter=number(.7,.00001,1e4,6);self.hub=number(1.,.00001,1e6,6);self.nu=number(.006,0.,1e4,8)
        self.bed_clearance=number(.1,0.,1e6,6);self.surface_clearance=number(.1,0.,1e6,6)
        self.wake_enabled=QCheckBox('Refine downstream wake');self.wake_enabled.setChecked(True)
        self.wake_length=number(10.,.01,1000.,2)
        self.wake_length.setToolTip('Local cells no larger than half the rotor diameter, or the river minimum if finer, over this many diameters downstream. Five-diameter-wide corridor clipped to the fluid domain; uses the placed downstream direction.')
        form.addRow(self.wake_enabled);form.addRow('Selected turbine wake length (rotor diameters)',self.wake_length)
        self.wake_summary=QLabel();self.wake_summary.setWordWrap(True);form.addRow(self.wake_summary)
        wake_note=QLabel('Automatic local mesh: at least two cells across each rotor and its enabled wake. If the river minimum is too coarse, only these areas and their transitions receive smaller 2:1 cells. Your river mesh size stays unchanged. Two cells is a minimum, not an accuracy guarantee.');wake_note.setWordWrap(True);form.addRow(wake_note)
        self.control=combo([('Prescribed constant Ct and Cp','constant_ct'),('Prescribed TSR + Ct/Cp table','prescribed_tsr_table'),('Turbine model + fixed RPM','fixed_rpm_table'),('Device speed / TSR surface + rotor inflow','fixed_rpm_surface')]);self.ct=number(.8,0.,.999999,6);self.cp=number(.4,0.,16/27,6);self.tsr=number(3.67,.0001,100.,6)
        self.performance_model=QLineEdit('Demonstration curve — replace with turbine data');self.rpm=number(90.,0.,1e5,6)
        for label,w in [('Turbine name',self.name),('Rotor diameter (m)',self.diameter),('Hub height above prepared bed (m)',self.hub),('Minimum rotor–bed clearance (m)',self.bed_clearance),('Minimum water cover above rotor (m)',self.surface_clearance),('Shared ambient eddy viscosity (m²/s)',self.nu),('Performance input',self.control),('Performance model / curve source',self.performance_model),('Rotor speed (RPM)',self.rpm),('Ct',self.ct),('Mechanical Cp',self.cp),('Prescribed TSR',self.tsr)]:form.addRow(label,w)
        self.table=QTableWidget(3,3);self.table.setHorizontalHeaderLabels(['TSR','Ct','Cp']);self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch);self.table.setMaximumHeight(170)
        for i,values in enumerate([(2.,.8,.3),(3.67,.8,.4),(6.,.7,.3)]):
            for j,x in enumerate(values):self.table.setItem(i,j,QTableWidgetItem(str(x)))
        root.addWidget(self.table);self.import_table=QPushButton('Import TSR / Ct / Cp CSV');root.addWidget(self.import_table);self.import_table.clicked.connect(self._import)
        from .turbine_surface import SurfaceEditor
        self.surface=SurfaceEditor();root.addWidget(self.surface);self.surface.changed.connect(self._changed)
        self.resolved=QLabel();self.resolved.setWordWrap(True);root.addWidget(self.resolved);root.addStretch()
        for w in [self.diameter,self.hub,self.bed_clearance,self.surface_clearance,self.nu,self.ct,self.cp,self.tsr,self.rpm]:w.valueChanged.connect(self._changed)
        self.performance_model.textChanged.connect(self._changed)
        self.name.textChanged.connect(self._changed);self.control.currentIndexChanged.connect(self._changed);self.table.cellChanged.connect(self._changed);self.enabled.toggled.connect(self._changed);self._changed()
        self.wake_enabled.toggled.connect(self._changed);self.wake_length.valueChanged.connect(self._changed)

    def _changed(self,*args):
        if self.loading:return
        if self.power_planning.isChecked() and self.enabled.isChecked() and self.control.currentData()!='fixed_rpm_surface':
            set_choice(self.control,'fixed_rpm_surface')
        self.wake_length.setEnabled(self.wake_enabled.isChecked())
        wake=(f'{self.name.text()}: {self.wake_length.value():g}D = {self.wake_length.value()*self.diameter.value():g} m downstream; corridor width {model.WAKE_WIDTH_DIAMETERS*self.diameter.value():g} m.' if self.wake_enabled.isChecked() else f'{self.name.text()}: wake refinement is off.')
        self.wake_summary.setText(wake+' This setting applies only to the selected turbine. Generate and Visualize mesh to rebuild and inspect it in Model & Run. Results retains the mesh of its saved run.')
        mode=self.control.currentData();surface=mode=='fixed_rpm_surface';lookup=mode!='constant_ct';rpm=mode in ('fixed_rpm_table','fixed_rpm_surface');self.surface.setVisible(surface);self.table.setVisible(lookup and not surface);self.import_table.setVisible(lookup and not surface);self.tsr.setEnabled(mode=='prescribed_tsr_table');self.ct.setEnabled(not lookup);self.cp.setEnabled(not lookup);self.rpm.setEnabled(rpm);self.performance_model.setEnabled(rpm)
        if self.rows:
            r=self.rows[self.current if 0<=self.current<len(self.rows) else 0];angle=math.degrees(math.atan2(*r['river_direction_xy'][::-1]));placement=f'River direction {angle:.1f}° from east; rotor {angle+90:.1f}°.'
        else:placement='No turbine placed.'
        position=f'\nAutomatic local mesh size cap: {self.diameter.value()/2:g} m (2:1 subdivision; existing finer river cells are retained).\nMounted bottom above reference bed: {self.hub.value()-self.diameter.value()/2:g} m. Minimum local depth at the reference bed for this mounting/cover: {self.hub.value()+self.diameter.value()/2+self.surface_clearance.value():g} m.'
        performance='RPM mode computes TSR = angular speed × radius / incoming velocity, then solves Ct, Cp and the velocity correction together. No extrapolation: outside the curve, parked, reversed or uncovered rotors produce no power. Import a curve for your actual turbine; initial table values are demonstrations.' if rpm else 'Ct and Cp are prescribed inputs. Fixed TSR selects fixed coefficients; thrust and power still change with flow.'
        if not surface:performance+=' These controls do not reconstruct the vertical wake for power; do not use them as validated wake-exposed yield estimates.'
        if surface:performance='Speed-surface mode computes Ct/Cp from rotor inflow and fixed RPM with a source-consistent, zero-depth-mean vertical wake correction and linear overlap. The resolved velocity already contains the horizontal wake. Up to three rotors and one upstream row; no topology-specific coefficient tuning. Missing speed/TSR coverage stops the run. Supply explicit device and wake data; see operating limits above. No automatic laboratory calibration.'
        if self.power_planning.isChecked():
            self.nu.setEnabled(False)
            performance='POWER PLANNING: no actuator drag in the hydraulic run. Cp/Ct use wake-corrected incoming rotor speed; power uses the rotor mean of velocity cubed. Linear overlap, finite-section normalization, no own-induction correction. Use explicit device speed/TSR surfaces for all rotors. Operating speed limits apply to incoming rotor speed in this mode. Electrical efficiency is optional. Results are estimates, not hydraulic work removed.'
        else:self.nu.setEnabled(True)
        self.resolved.setText(placement+position+'\n'+(('Resolved: OTF smooth actuators + current-depth Kramer–Piggott correction; common-bed balanced pressure; one shared ambient mixing field. ' if not self.power_planning.isChecked() else '')+'Hydraulic-only, zero-yaw assumption; individual orientations stay fixed. Each rotor independently switches off when either clearance limit fails and restarts when both are met. ')+performance+' Power is rotor mechanical shaft power. An electrical estimate is available only with an explicitly supplied conversion efficiency. No blade-geometry prediction, support drag or rotor-generated turbulence.')
        if 0<=self.current<self.selection.count():self.selection.setItemText(self.current,self.name.text())
        self.add_turbine.setEnabled(True);self.edit_turbine.setEnabled(bool(self.rows));self.remove_turbine.setEnabled(bool(self.rows));self.changed.emit()

    def control_value(self):
        if self.control.currentData()=='fixed_rpm_surface':return self.surface.value(self.performance_model.text().strip(),self.rpm.value())
        if self.control.currentData()=='constant_ct':return dict(mode='constant_ct',thrust_coefficient=self.ct.value(),power_coefficient=self.cp.value())
        rows=[]
        for i in range(self.table.rowCount()):
            try:values=[float(self.table.item(i,j).text()) for j in range(3)]
            except (ValueError,AttributeError):raise ValueError('Fill every TSR, Ct and Cp table cell with a number, or import a complete CSV.')
            rows.append(dict(zip(['tip_speed_ratio','thrust_coefficient','power_coefficient'],values)))
        if self.control.currentData()=='fixed_rpm_table':return dict(mode='fixed_rpm_table',model_name=self.performance_model.text().strip(),rotational_speed_rpm=self.rpm.value(),performance=rows,outside_curve='off')
        return dict(mode='prescribed_tsr_table',tip_speed_ratio=self.tsr.value(),performance=rows)

    def draft(self):
        rows=deepcopy(self.rows)
        if rows:rows[self.current if 0<=self.current<len(rows) else 0].update(id=self.name.text().strip(),rotor_diameter_m=self.diameter.value(),hub_height_m=self.hub.value(),minimum_bed_clearance_m=self.bed_clearance.value(),minimum_surface_clearance_m=self.surface_clearance.value(),control=self.control_value(),wake_refinement=dict(enabled=self.wake_enabled.isChecked(),length_diameters=self.wake_length.value()))
        return dict(method=model.METHOD,turbines=rows,ambient_viscosity_m2_s=self.nu.value())

    def value(self):return self.draft() if self.enabled.isChecked() else None

    def editor_state(self):
        return dict(schema=4,surface=self.surface.state(),rows=deepcopy(self.rows),parameters=dict(name=self.name.text(),diameter=self.diameter.value(),hub=self.hub.value(),bed_clearance=self.bed_clearance.value(),surface_clearance=self.surface_clearance.value(),nu=self.nu.value(),wake_enabled=self.wake_enabled.isChecked(),wake_length=self.wake_length.value(),
            mode=self.control.currentData(),ct=self.ct.value(),cp=self.cp.value(),tsr=self.tsr.value(),rpm=self.rpm.value(),performance_model=self.performance_model.text(),selected_index=self.current),
            table=[[self.table.item(i,j).text() if self.table.item(i,j) else '' for j in range(3)] for i in range(self.table.rowCount())])

    def place(self,points):
        import numpy as np
        points=np.asarray(points,float);direction=points[1]-points[0]
        row=model.definition(points[0],direction,name=self.name.text().strip(),diameter=self.diameter.value(),hub_height=self.hub.value(),minimum_bed_clearance=self.bed_clearance.value(),minimum_surface_clearance=self.surface_clearance.value());row['control']=self.control_value()
        row['wake_refinement']=dict(enabled=self.wake_enabled.isChecked(),length_diameters=self.wake_length.value())
        rows=self.draft()['turbines'];index=self._capture_edit
        if index is None:
            used={r['id'] for r in rows};i=1
            if row['id'] in used:
                while 'Turbine '+str(i) in used:i+=1
                row['id']='Turbine '+str(i)
            index=len(rows);rows.append(row)
        else:rows[index]=row
        if self.power_planning.isChecked():
            for r in rows:model.validate(dict(method=model.METHOD,turbines=[r],ambient_viscosity_m2_s=self.nu.value()))
        else:model.validate(dict(method=model.METHOD,turbines=rows,ambient_viscosity_m2_s=self.nu.value()))
        self.rows=rows;self.current=index;self._capture_edit=None;self._show_row();self._choices();self._changed()

    def begin_placement(self,edit=False):
        self.rows=self.draft()['turbines']
        if edit and not self.rows:raise ValueError('Select a turbine to reposition.')
        self._capture_edit=self.current if edit else None

    def remove_selected(self):
        if not self.rows:return
        self.rows=self.draft()['turbines'];self.rows.pop(self.current);self.current=min(self.current,len(self.rows)-1)
        self._show_row();self._choices();self._changed()

    def _choices(self):
        self.selection.blockSignals(True);self.selection.clear()
        self.selection.addItems([r['id'] for r in self.rows]);self.selection.setCurrentIndex(self.current);self.selection.blockSignals(False)

    def _select(self,index):
        if self.loading or index<0 or index==self.current:return
        try:self.rows=self.draft()['turbines']
        except ValueError as exc:
            self.selection.blockSignals(True);self.selection.setCurrentIndex(self.current);self.selection.blockSignals(False)
            QMessageBox.warning(self,'Turbine settings',str(exc)+'\nComplete this turbine before selecting another.');return
        self.current=index;self._show_row();self._changed()

    def _show_row(self):
        if not 0<=self.current<len(self.rows):return
        was=self.loading;self.loading=True
        try:
            r=self.rows[self.current];self.name.setText(r['id']);self.diameter.setValue(r['rotor_diameter_m']);self.hub.setValue(r['hub_height_m']);self.bed_clearance.setValue(r['minimum_bed_clearance_m']);self.surface_clearance.setValue(r['minimum_surface_clearance_m']);set_choice(self.control,r['control']['mode']);c=r['control']
            self.wake_enabled.setChecked(r['wake_refinement']['enabled']);self.wake_length.setValue(r['wake_refinement']['length_diameters'])
            if c['mode']=='constant_ct':self.ct.setValue(c['thrust_coefficient']);self.cp.setValue(c['power_coefficient'])
            elif c['mode']=='fixed_rpm_surface':self.rpm.setValue(c['rotational_speed_rpm']);self.performance_model.setText(c['model_name']);self.surface.load(c)
            elif c['mode']=='fixed_rpm_table':self.rpm.setValue(c['rotational_speed_rpm']);self.performance_model.setText(c['model_name']);self._set_table(c['performance'])
            else:self.tsr.setValue(c['tip_speed_ratio']);self._set_table(c['performance'])
        finally:self.loading=was

    def load(self,value,draft=None):
        self.loading=True
        try:
            raw=draft if value is None and draft is not None else None
            doc=value or model.default();self.rows=deepcopy(raw['rows'] if raw else doc['turbines']);self.enabled.setChecked(value is not None);self.nu.setValue(doc['ambient_viscosity_m2_s'])
            selection=(draft or {}).get('parameters',{}).get('selected_index',0)
            self.current=max(-1,min(len(self.rows)-1,selection));self._show_row()
            if raw:
                p=raw['parameters'];self.name.setText(p['name']);self.diameter.setValue(p['diameter']);self.hub.setValue(p['hub']);self.bed_clearance.setValue(p['bed_clearance']);self.surface_clearance.setValue(p['surface_clearance']);self.nu.setValue(p['nu']);set_choice(self.control,p['mode']);self.ct.setValue(p['ct']);self.cp.setValue(p['cp']);self.tsr.setValue(p['tsr'])
                self.wake_enabled.setChecked(p['wake_enabled']);self.wake_length.setValue(p['wake_length'])
                if p['mode']=='fixed_rpm_surface':self.surface.restore(raw['surface'])
                if p['mode'] in ('fixed_rpm_table','fixed_rpm_surface'):self.rpm.setValue(p['rpm']);self.performance_model.setText(p['performance_model'])
                self.table.setRowCount(len(raw['table']))
                for i,row in enumerate(raw['table']):
                    for j,text in enumerate(row):self.table.setItem(i,j,QTableWidgetItem(text))
        finally:self.loading=False
        self._choices();self._changed()

    def _set_table(self,rows):
        self.table.setRowCount(len(rows))
        for i,r in enumerate(rows):
            for j,k in enumerate(['tip_speed_ratio','thrust_coefficient','power_coefficient']):self.table.setItem(i,j,QTableWidgetItem(str(r[k])))

    def _import(self):
        p,_=QFileDialog.getOpenFileName(self,'Import performance table','','CSV (*.csv)')
        if not p:return
        try:
            with open(p,encoding='utf-8-sig',newline='') as f:
                reader=csv.DictReader(f);rows=[dict(tip_speed_ratio=float(r['TSR']),thrust_coefficient=float(r['Ct']),power_coefficient=float(r['Cp'])) for r in reader]
            self.loading=True;self._set_table(rows)
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self,'Performance table',str(e)+'\nUse headers TSR,Ct,Cp.')
        finally:self.loading=False
        self._changed()
