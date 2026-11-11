"""Point-and-river-direction placement, with explicit fixed-axis rotor yaw."""
from copy import deepcopy
import csv,math
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget,QVBoxLayout,QFormLayout,QCheckBox,QLineEdit,QPushButton,QTableWidget,QTableWidgetItem,QHeaderView,QLabel,QFileDialog,QComboBox,QMessageBox,QTabWidget
from .river_physics import number,combo,set_choice
from braids_core import river_turbines as model


class TurbineEditor(QWidget):
    changed=Signal();action=Signal(str)
    def __init__(self,parent=None):
        super().__init__(parent);self.rows=[];self.loading=False;self.current=-1;self._capture_edit=None;self._numeric_original={}
        self.enabled=QCheckBox('Enable hydrokinetic turbines',self);self.enabled.hide()
        root=QVBoxLayout(self);root.setContentsMargins(0,0,0,0)
        self.title=QLabel('Hydrokinetic turbines');self.title.setObjectName('Title');root.addWidget(self.title)
        self.power_planning=QCheckBox('Estimates only - no turbine forces in the flow');self.power_planning.setChecked(False);self.power_planning.hide()
        note=QLabel('Thetis depth-averaged turbines: every rotor acts on the water. Power can additionally account for vertical wake structure while preserving that hydraulic response.');note.setWordWrap(True);root.addWidget(note)
        self.power_planning.toggled.connect(self._changed)
        self.summary=QLabel('Add rotors, then select a turbine to edit its individual settings. Click its centre, then point downstream. A fixed rotor axis can be rotated with the yaw control below.');self.summary.setWordWrap(True);root.addWidget(self.summary)
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
        self.control=combo([('Constant Ct / Cp','constant_ct'),('Device curves: separate incoming-speed Ct and Cp','incident_speed_curves'),('Shared incoming-speed table: Ct / Cp','incident_speed_table'),('Stock Thetis table: local Ct / incoming Cp','thetis_speed_table')]);self.ct=number(.8,0.,.999999,6);self.cp=number(.4,0.,1.,6);self.tsr=number(3.67,.0001,100.,6)
        self.performance_model=QLineEdit('Demonstration curve — replace with turbine data');self.rpm=number(90.,0.,1e5,6)
        self.orientation=combo([('Flow-following (stock Thetis)','flow_following'),('Fixed rotor axis (yaw)','fixed_axis')])
        self.yaw=number(0.,-90.,90.,4)
        self.orientation.setToolTip('Fixed-axis mode applies zero-yaw device curves to the incoming normal component. Do not supply coefficients already corrected for yaw. Curves must cover local induction-corrected speeds, not only the inlet or section-average speed.')
        self.yaw.setToolTip('Rotate the rotor normal from the placed downstream river direction. Positive is counterclockwise. This is a fixed mechanical orientation; the actual yaw relative to local water velocity is exported.')
        form.addRow('Rotor loading',self.orientation);form.addRow('Rotor yaw (degrees)',self.yaw)
        for label,w in [('Turbine name',self.name),('Rotor diameter (m)',self.diameter),('Hub height above prepared bed (m)',self.hub),('Minimum rotor–bed clearance (m)',self.bed_clearance),('Minimum water cover above rotor (m)',self.surface_clearance),('Shared ambient eddy viscosity (m²/s)',self.nu),('Performance input',self.control),('Performance model / curve source',self.performance_model),('Ct',self.ct),('Mechanical Cp',self.cp)]:form.addRow(label,w)
        self.power_mode=combo([('Stock Thetis depth-mean power','thetis'),('Non-yawed array power (upstream reference)','power.gaussian_upstream_reference.v1'),('Dou (2019): aligned / yawed array power','power.dou2019_device_yaw.v1')])
        self.power_growth=number(0.,0.,1.,8);self.power_width=number(0.,0.,2.,8);self.power_spacing=number(3.,3.,100.,2);self.power_provenance=QLineEdit()
        self.power_provenance.setPlaceholderText('Source of isolated-wake width/growth or site calibration')
        for label,w in [('Shared power estimator',self.power_mode),('Wake growth k (sigma = epsilon D + k x)',self.power_growth),('Initial width epsilon (sigma / D)',self.power_width),('Minimum far-wake spacing (D)',self.power_spacing),('Wake-parameter source',self.power_provenance)]:form.addRow(label,w)
        self.power_note=QLabel();self.power_note.setWordWrap(True);form.addRow(self.power_note)
        from .turbine_yaw_input import DeviceYawInput
        self.device_yaw=DeviceYawInput();form.addRow(self.device_yaw);self.device_yaw.changed.connect(self._changed)
        self.table=QTableWidget(3,3);self.table.setHorizontalHeaderLabels(['Speed (m/s)','Ct','Cp']);self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch);self.table.setMaximumHeight(170)
        for i,values in enumerate([(0.,0.,0.),(1.,.8,.4),(3.,.7,.3)]):
            for j,x in enumerate(values):self.table.setItem(i,j,QTableWidgetItem(str(x)))
        root.addWidget(self.table);self.import_table=QPushButton('Import Speed / Ct / Cp CSV');root.addWidget(self.import_table);self.import_table.clicked.connect(self._import)
        self.cp.setToolTip('Thetis convention: zero selects ideal actuator-disk Cp = Ct*(1+sqrt(1-Ct))/2. Enter measured Cp for shaft-power estimates.')
        self.add_curve_row=QPushButton('Add speed row');root.addWidget(self.add_curve_row);self.add_curve_row.clicked.connect(lambda:self.table.insertRow(self.table.rowCount()))
        self.remove_curve_row=QPushButton('Remove speed row');root.addWidget(self.remove_curve_row);self.remove_curve_row.clicked.connect(lambda:self.table.removeRow(self.table.currentRow() if self.table.currentRow()>=0 else self.table.rowCount()-1))
        self.curves=QTabWidget();root.addWidget(self.curves)
        self.curve_tables={}
        for field,label in [('thrust_coefficient','Ct'),('power_coefficient','Cp')]:
            panel=QWidget();layout=QVBoxLayout(panel);table=QTableWidget(3,2)
            table.setHorizontalHeaderLabels(['Incoming speed (m/s)',label]);table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch);table.setMaximumHeight(170)
            self.curve_tables[field]=table;layout.addWidget(table)
            for i,(u,c) in enumerate([(0.,0.),(1.,.8 if label=='Ct' else .4),(3.,.7 if label=='Ct' else .3)]):
                for j,value in enumerate((u,c)):table.setItem(i,j,QTableWidgetItem(str(value)))
            imp=QPushButton(f'Import Speed / {label} CSV');layout.addWidget(imp);imp.clicked.connect(lambda checked=False,f=field:self._import_curve(f))
            add=QPushButton(f'Add {label} row');layout.addWidget(add);add.clicked.connect(lambda checked=False,t=table:t.insertRow(t.rowCount()))
            remove=QPushButton(f'Remove {label} row');layout.addWidget(remove);remove.clicked.connect(lambda checked=False,t=table:t.removeRow(t.currentRow() if t.currentRow()>=0 else t.rowCount()-1))
            table.cellChanged.connect(self._changed);self.curves.addTab(panel,label+' curve')
        self.curve_note=QLabel('Ct controls the turbine operating range. Below its lowest speed, the rotor stops with zero power; it restarts when the whole source support is above cut-in. One warning is printed at completion if this occurred. Cp must cover the Ct range and may extend beyond it for wake-affected rotor power. Both use incoming speed in m/s; different knots and row counts are allowed. Supply device data, not extrapolated values.');self.curve_note.setWordWrap(True);root.addWidget(self.curve_note)
        self.resolved=QLabel();self.resolved.setWordWrap(True);root.addWidget(self.resolved);root.addStretch()
        from .turbine_ui_layout import install
        install(self,root,form,note,wake_note)
        self.title.hide()  # The workflow panel already supplies the title.
        for w in [self.diameter,self.hub,self.bed_clearance,self.surface_clearance,self.nu,self.ct,self.cp,self.tsr,self.rpm]:w.valueChanged.connect(self._changed)
        self.performance_model.textChanged.connect(self._changed)
        self.name.textChanged.connect(self._changed);self.control.currentIndexChanged.connect(self._control_changed);self.table.cellChanged.connect(self._changed);self.enabled.toggled.connect(self._changed);self._changed()
        self.wake_enabled.toggled.connect(self._changed);self.wake_length.valueChanged.connect(self._changed)
        self.power_mode.currentIndexChanged.connect(self._changed);self.power_provenance.textChanged.connect(self._changed)
        for w in [self.power_growth,self.power_width,self.power_spacing]:w.valueChanged.connect(self._changed)
        self.orientation.currentIndexChanged.connect(self._changed);self.yaw.valueChanged.connect(self._changed)

    def _changed(self,*args):
        if self.loading:return
        self._last_control_mode=self.control.currentData()
        self.wake_length.setEnabled(self.wake_enabled.isChecked())
        wake=(f'{self.name.text()}: {self.wake_length.value():g}D = {self.wake_length.value()*self.diameter.value():g} m downstream; corridor width {model.WAKE_WIDTH_DIAMETERS*self.diameter.value():g} m.' if self.wake_enabled.isChecked() else f'{self.name.text()}: wake refinement is off.')
        self.wake_summary.setText(wake+' This setting applies only to the selected turbine. Generate and Visualize mesh to rebuild and inspect it in Model & Run. Results retains the mesh of its saved run.')
        lookup=self.control.currentData()!='constant_ct';independent=self.control.currentData()=='incident_speed_curves';shared=lookup and not independent
        self.table.setVisible(shared);self.import_table.setVisible(shared);self.add_curve_row.setVisible(shared);self.remove_curve_row.setVisible(shared);self.ct.setEnabled(not lookup);self.cp.setEnabled(not lookup);self.performance_model.setEnabled(lookup);self.curves.setVisible(independent);self.curve_note.setVisible(independent)
        profile=self.power_mode.currentData()!='thetis'
        dou=self.power_mode.currentData()=='power.dou2019_device_yaw.v1'
        self.orientation.setToolTip('With Dou power, fixed-axis curves use zero-yaw incoming speed magnitude and the imported device/controller yaw-power ratio. With stock Thetis power, fixed-axis curves use incoming normal speed. Do not supply coefficients already corrected for yaw; curves must cover local induction-corrected speeds.')
        self.device_yaw.setVisible(dou)
        fixed=self.orientation.currentData()=='fixed_axis';self.yaw.setEnabled(fixed)
        for w in [self.power_growth,self.power_width,self.power_spacing,self.power_provenance]:w.setEnabled(profile)
        self.power_note.setText('Power only: uses the first-row incoming speed and published Gaussian wake losses. Requires independently supported wake growth/width and mixing; straight non-yawed quasi-uniform ambient flow; depth/D at least 2 for receivers. It rejects >10% reference-speed variation. No additional deficit enters the water field. Mesh wake length does not terminate physical wakes.' if profile else 'Stock Thetis power uses the resolved depth mean. No vertical wake correction is applied.')
        if dou:
            self.power_note.setText('Dou (2019) shifts and skews the analytical wake with local yaw. Overlapping upstream wakes contribute to rotor power once; turbines still apply momentum forces to the water. Wake growth and width need an independent source. Straight, quasi-uniform reaches; receiver depth/D at least 2 and minimum spacing 3D. The mesh wake length is not a physical cutoff. Yawed hydrokinetic array power remains a documented extrapolation pending validation.')
        if self.rows:
            r=self.rows[self.current if 0<=self.current<len(self.rows) else 0];angle=math.degrees(math.atan2(*r['river_direction_xy'][::-1]));placement=f'Placed river direction {angle:.1f}° from east; rotor line {angle+90+(self.yaw.value() if fixed else 0):.1f}°.'
        else:placement='No turbine placed.'
        position=f'\nAutomatic local mesh size cap: {self.diameter.value()/2:g} m (2:1 subdivision; existing finer river cells are retained).\nMounted bottom above reference bed: {self.hub.value()-self.diameter.value()/2:g} m. Minimum local depth at the reference bed for this mounting/cover: {self.hub.value()+self.diameter.value()/2+self.surface_clearance.value():g} m.'
        convention=('Device curves use incoming speed for both Ct and Cp; local-to-incoming induction is solved at current depth.' if self.control.currentData() in ('incident_speed_table','incident_speed_curves') else 'Stock convention: Ct uses local speed; Cp uses induction-corrected speed.')
        self.resolved.setText(placement+position+'\nFlow-directed Thetis drag uses both velocity components and current water depth. '+convention+' Tables define their operating range; no extrapolation. Full rotor coverage and bed-clearance limits remain active. Power is mechanical; generator losses are excluded. '+('The selected power correction requires alignment within 10 degrees; it does not model yaw.' if profile else 'No Gaussian wake overlay or hub-height map is active.'))
        if fixed:
            self.resolved.setText(placement+position+f'\nFixed rotor yaw: {self.yaw.value():g} degrees. Both velocity components are projected onto the rotor normal. Momentum removal and power use this normal component, with the Thetis depth correction. Supply incoming-speed Ct/Cp data for the actual device/controller/RPM. The measured single-rotor check covers 15 degrees; other yaw angles and yawed-array power remain outside that experimental evidence. Select resolved-field Thetis power: the non-yawed Gaussian correction is incompatible. Reverse flow switches the rotor off; mixed flow across it is an error.')
        if dou:
            self.resolved.setText(placement+position+'\nDou power uses the existing hydraulic response, a yaw-dependent analytical wake and circular rotor integration. Fixed-axis Ct/Cp curves must use zero-yaw incoming speed magnitude at the same device/RPM as the imported yaw table; its ratio replaces the yaw power reduction once. Use Fixed rotor axis and the yaw control for fixed turbines. No extra analytical deficit is added to the water velocity field. The unchanged coverage, reverse-flow and low-speed shutdown rules apply. Power is mechanical; generator losses are excluded.')
        if 0<=self.current<self.selection.count():self.selection.setItemText(self.current,self.name.text())
        self.add_turbine.setEnabled(True);self.edit_turbine.setEnabled(bool(self.rows));self.remove_turbine.setEnabled(bool(self.rows))
        from .turbine_ui_layout import refresh
        refresh(self);self.changed.emit()

    def _control_changed(self,*args):
        if self.loading:return
        if self.control.currentData()=='incident_speed_curves':
            self.loading=True
            try:
                if getattr(self,'_last_control_mode',None)=='incident_speed_table':
                    for field,column in [('thrust_coefficient',1),('power_coefficient',2)]:
                        table=self.curve_tables[field];table.setRowCount(self.table.rowCount())
                        for i in range(table.rowCount()):
                            for j,src in enumerate((0,column)):
                                item=self.table.item(i,src);table.setItem(i,j,QTableWidgetItem(item.text() if item else ''))
                else:
                    # Local-speed Ct cannot be relabelled as incoming-speed Ct.
                    # Require actual device data instead of unrelated cached rows.
                    self.performance_model.clear()
                    for table in self.curve_tables.values():
                        table.clearContents();table.setRowCount(2)
            finally:self.loading=False
        elif getattr(self,'_last_control_mode',None)=='incident_speed_curves' and self.control.currentData()!='constant_ct':
            self.loading=True
            try:
                # Do not silently discard independent coverage, or reuse another
                # turbine's hidden shared table when changing the input contract.
                self.table.clearContents();self.table.setRowCount(2);self.performance_model.clear()
            finally:self.loading=False
        self._changed()

    def control_value(self):
        if self.control.currentData()=='constant_ct':return dict(mode='constant_ct',thrust_coefficient=self._number_value('ct',self.ct),power_coefficient=self._number_value('cp',self.cp))
        if self.control.currentData()=='incident_speed_curves':
            curves={}
            for field,key in [('thrust_coefficient','thrust_curve'),('power_coefficient','power_curve')]:
                table=self.curve_tables[field];rows=[]
                for i in range(table.rowCount()):
                    try:u=float(table.item(i,0).text());value=float(table.item(i,1).text())
                    except (ValueError,AttributeError):raise ValueError(f'Fill every speed and coefficient cell in the {key.replace("_"," ")}, or remove the incomplete row.')
                    rows.append({'speed_m_s':u,field:value})
                curves[key]=rows
            return dict(mode='incident_speed_curves',model_name=self.performance_model.text().strip(),**curves)
        rows=[]
        for i in range(self.table.rowCount()):
            try:values=[float(self.table.item(i,j).text()) for j in range(3)]
            except (ValueError,AttributeError):raise ValueError('Fill every speed, Ct and Cp cell, or import a complete CSV with headers Speed,Ct,Cp.')
            rows.append(dict(zip(['speed_m_s','thrust_coefficient','power_coefficient'],values)))
        return dict(mode=self.control.currentData(),model_name=self.performance_model.text().strip(),performance=rows)

    def power_value(self):
        mode=self.power_mode.currentData()
        if mode=='thetis':return dict(method=mode)
        wake=self.wake_power_value()
        return dict(method=mode,wake=wake,device=self.device_yaw.value()) if mode=='power.dou2019_device_yaw.v1' else wake

    def wake_power_value(self):
        return dict(method='power.gaussian_upstream_reference.v1',growth=self._number_value('growth',self.power_growth),initial_width=self._number_value('initial_width',self.power_width),minimum_distance_D=self._number_value('minimum_distance_D',self.power_spacing),provenance=self.power_provenance.text().strip())

    def orientation_value(self):
        return {'mode':'flow_following'} if self.orientation.currentData()=='flow_following' else dict(mode='fixed_axis',yaw_deg=self.yaw.value())

    def _number_value(self,key,widget):
        original=self._numeric_original.get(key)
        return original if original is not None and widget.value()==round(original,widget.decimals()) else widget.value()

    def _load_power(self,power):
        power=power or {'method':'thetis'};set_choice(self.power_mode,power['method'])
        self.device_yaw.set_value(power.get('device'))
        power=power.get('wake',power)
        for key in ['growth','initial_width','minimum_distance_D']:self._numeric_original[key]=power.get(key)
        self.power_growth.setValue(power.get('growth',0.));self.power_width.setValue(power.get('initial_width',0.));self.power_spacing.setValue(power.get('minimum_distance_D',3.));self.power_provenance.setText(power.get('provenance',''))

    def draft(self):
        rows=deepcopy(self.rows)
        if rows:rows[self.current if 0<=self.current<len(rows) else 0].update(id=self.name.text().strip(),rotor_diameter_m=self.diameter.value(),hub_height_m=self.hub.value(),minimum_bed_clearance_m=self.bed_clearance.value(),minimum_surface_clearance_m=self.surface_clearance.value(),control=self.control_value(),wake_refinement=dict(enabled=self.wake_enabled.isChecked(),length_diameters=self.wake_length.value()))
        if rows:
            row=rows[self.current if 0<=self.current<len(rows) else 0]
            if self.orientation.currentData()=='fixed_axis':row['orientation']=self.orientation_value()
            else:row.pop('orientation',None)
        value=dict(method=model.METHOD,turbines=rows,ambient_viscosity_m2_s=self._number_value('nu',self.nu))
        if self.power_mode.currentData()!='thetis':value['power_estimation']=self.power_value()
        return value

    def value(self):return self.draft() if self.enabled.isChecked() else None

    def editor_state(self):
        return dict(schema=7,rows=deepcopy(self.rows),parameters=dict(orientation=self.orientation_value(),name=self.name.text(),diameter=self.diameter.value(),hub=self.hub.value(),bed_clearance=self.bed_clearance.value(),surface_clearance=self.surface_clearance.value(),nu=self._number_value('nu',self.nu),wake_enabled=self.wake_enabled.isChecked(),wake_length=self.wake_length.value(),
            mode=self.control.currentData(),ct=self.ct.value(),cp=self.cp.value(),tsr=self.tsr.value(),rpm=self.rpm.value(),performance_model=self.performance_model.text(),selected_index=self.current,power_estimation=self.power_value(),device_yaw_draft=self.device_yaw.value(),wake_power_draft=self.wake_power_value()),
            table=[[self.table.item(i,j).text() if self.table.item(i,j) else '' for j in range(3)] for i in range(self.table.rowCount())],
            curves={field:[[table.item(i,j).text() if table.item(i,j) else '' for j in range(2)] for i in range(table.rowCount())] for field,table in self.curve_tables.items()})

    def place(self,points):
        import numpy as np
        points=np.asarray(points,float);direction=points[1]-points[0]
        row=model.definition(points[0],direction,name=self.name.text().strip(),diameter=self.diameter.value(),hub_height=self.hub.value(),minimum_bed_clearance=self.bed_clearance.value(),minimum_surface_clearance=self.surface_clearance.value());row['control']=self.control_value()
        row['wake_refinement']=dict(enabled=self.wake_enabled.isChecked(),length_diameters=self.wake_length.value())
        if self.orientation.currentData()=='fixed_axis':row['orientation']=self.orientation_value()
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
            o=model.orientation(r);set_choice(self.orientation,o['mode']);self.yaw.setValue(o.get('yaw_deg',0.))
            if c['mode']=='constant_ct':
                self.ct.setValue(c['thrust_coefficient']);self.cp.setValue(c['power_coefficient'] or 0.)
                self._numeric_original.update(ct=c['thrust_coefficient'],cp=c['power_coefficient'] or 0.)
            elif c['mode'] in ('thetis_speed_table','incident_speed_table'):self.performance_model.setText(c['model_name']);self._set_table(c['performance'])
            elif c['mode']=='incident_speed_curves':
                self.performance_model.setText(c['model_name'])
                for field,key in [('thrust_coefficient','thrust_curve'),('power_coefficient','power_curve')]:self._set_curve(field,c[key])
            else:raise ValueError('Reconfigure old turbine controls as Thetis constant coefficients or a speed table.')
        finally:self.loading=was

    def load(self,value,draft=None):
        self.loading=True
        try:
            raw=draft if value is None and draft is not None else None
            doc=value or model.default();self.rows=deepcopy(raw['rows'] if raw else doc['turbines']);self.enabled.setChecked(value is not None);self.nu.setValue(doc['ambient_viscosity_m2_s'])
            self._numeric_original['nu']=raw['parameters']['nu'] if raw else doc['ambient_viscosity_m2_s']
            self._load_power(raw['parameters'].get('power_estimation') if raw else doc.get('power_estimation'))
            saved=(raw or draft or {}).get('parameters',{})
            if self.power_mode.currentData()=='thetis' and 'wake_power_draft' in saved:
                self._load_power(saved['wake_power_draft']);set_choice(self.power_mode,'thetis')
            if 'device_yaw_draft' in saved and self.power_mode.currentData()!='power.dou2019_device_yaw.v1':self.device_yaw.set_value(saved['device_yaw_draft'])
            selection=(draft or {}).get('parameters',{}).get('selected_index',0)
            self.current=max(-1,min(len(self.rows)-1,selection));self._show_row()
            if raw:
                p=raw['parameters'];self.name.setText(p['name']);self.diameter.setValue(p['diameter']);self.hub.setValue(p['hub']);self.bed_clearance.setValue(p['bed_clearance']);self.surface_clearance.setValue(p['surface_clearance']);self.nu.setValue(p['nu']);set_choice(self.control,p['mode']);self.ct.setValue(p['ct']);self.cp.setValue(p['cp']);self.tsr.setValue(p['tsr'])
                self.wake_enabled.setChecked(p['wake_enabled']);self.wake_length.setValue(p['wake_length'])
                o=p.get('orientation',{'mode':'flow_following'});set_choice(self.orientation,o['mode']);self.yaw.setValue(o.get('yaw_deg',0.))
                self.performance_model.setText(p['performance_model'])
                self.table.setRowCount(len(raw['table']))
                for i,row in enumerate(raw['table']):
                    for j,text in enumerate(row):self.table.setItem(i,j,QTableWidgetItem(text))
                for field,rows in raw.get('curves',{}).items():
                    table=self.curve_tables[field];table.setRowCount(len(rows))
                    for i,row in enumerate(rows):
                        for j,text in enumerate(row):table.setItem(i,j,QTableWidgetItem(text))
        finally:self.loading=False
        self._choices();self._changed()

    def _set_table(self,rows):
        self.table.setRowCount(len(rows))
        for i,r in enumerate(rows):
            for j,k in enumerate(['speed_m_s','thrust_coefficient','power_coefficient']):self.table.setItem(i,j,QTableWidgetItem(str(r[k])))

    def _import(self):
        p,_=QFileDialog.getOpenFileName(self,'Import performance table','','CSV (*.csv)')
        if not p:return
        try:
            with open(p,encoding='utf-8-sig',newline='') as f:
                reader=csv.DictReader(f);rows=[dict(speed_m_s=float(r['Speed']),thrust_coefficient=float(r['Ct']),power_coefficient=float(r['Cp'])) for r in reader]
            self.loading=True;self._set_table(rows)
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self,'Performance table',str(e)+'\nUse headers Speed,Ct,Cp.')
        finally:self.loading=False
        self._changed()

    def _set_curve(self,field,rows):
        table=self.curve_tables[field];table.setRowCount(len(rows))
        for i,row in enumerate(rows):
            for j,key in enumerate(('speed_m_s',field)):table.setItem(i,j,QTableWidgetItem(str(row[key])))

    def _import_curve(self,field):
        label='Ct' if field=='thrust_coefficient' else 'Cp'
        path,_=QFileDialog.getOpenFileName(self,f'Import {label} curve','','CSV (*.csv)')
        if not path:return
        try:
            with open(path,encoding='utf-8-sig',newline='') as stream:
                rows=[{'speed_m_s':float(r['Speed']),field:float(r[label])} for r in csv.DictReader(stream)]
            self.loading=True;self._set_curve(field,rows)
        except Exception as exc:QMessageBox.warning(self,'Device curve',str(exc)+f'\nUse headers Speed,{label}.')
        finally:self.loading=False
        self._changed()
