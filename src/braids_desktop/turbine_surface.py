"""Explicit device data and applicability for the rotor-inflow power model."""
import csv
from PySide6.QtWidgets import QWidget,QVBoxLayout,QFormLayout,QLabel,QLineEdit,QTableWidget,QTableWidgetItem,QHeaderView,QPushButton,QHBoxLayout,QCheckBox,QFileDialog,QMessageBox
from PySide6.QtCore import Signal
from .river_physics import number


class SurfaceEditor(QWidget):
    changed=Signal()
    def __init__(self,parent=None):
        super().__init__(parent);self.loading=False;self.wake_method='gaussian_zero_mean_linear_overlap'
        root=QVBoxLayout(self);root.setContentsMargins(0,0,0,0)
        note=QLabel('Device Cp/Ct versus incoming speed and TSR. Supply at least two speed slices, with at least two TSR rows each. Missing data stops the run; it is never extrapolated or silently replaced by zero power.');note.setWordWrap(True);root.addWidget(note)
        self.table=QTableWidget(0,4);self.table.setHorizontalHeaderLabels(['U (m/s)','TSR','Ct','Mechanical Cp']);self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch);self.table.setMaximumHeight(210);root.addWidget(self.table)
        buttons=QHBoxLayout();root.addLayout(buttons)
        for text,fn in [('Add row',self.add_row),('Remove row',self.remove_row),('Import CSV',self.import_csv),('Export CSV',self.export_csv)]:
            b=QPushButton(text);b.clicked.connect(fn);buttons.addWidget(b)
        form=QFormLayout();root.addLayout(form)
        self.growth=number(0.,0.,10.,9);self.width=number(0.,0.,10.,9);self.distance=number(3.,3.,1000.,3)
        self.provenance=QLineEdit();self.provenance.setPlaceholderText('Source, calibration conditions and applicability of wake/mixing')
        for name,w in [('Wake growth k',self.growth),('Initial wake width / D',self.width),('Minimum downstream model spacing (upstream D)',self.distance),('Wake and mixing provenance',self.provenance)]:form.addRow(name,w)
        self.distance.setToolTip('Far-wake model validity limit, not rotor clearance or mesh length. An exposed downstream rotor must be at least this many upstream rotor diameters away along the upstream river direction. 3D means 2.1 m for D=0.7 m. Do not reduce this limit without validating a near-wake model.')
        note=QLabel('Near-wake exclusion: the velocity/power correction is not qualified immediately behind a rotor. The minimum downstream spacing applies only when another rotor intersects its potential wake envelope. It is separate from physical rotor overlap and the default 10D mesh refinement. Side-by-side rotors are allowed when their rotor/force footprints are separate.');note.setWordWrap(True);form.addRow(note)
        note=QLabel('Wake width: sigma/D = k x/D + initial width/D; independent of the mesh refinement length (normally 10D), with no physical cutoff there. Coupled mode retains actuator forces and supports single, tandem and three-rotor staggered layouts using linear zero-depth-mean overlap. Power planning supports one upstream row with linear overlap in straight rectangular channels. Both use a fixed mesh. Low-turbulence power remains uncertain; supply site-appropriate wake data.');note.setWordWrap(True);form.addRow(note)
        self.operation=QCheckBox('Use local-speed shutdown limits and electrical-efficiency estimate');form.addRow(self.operation)
        self.cut_in=number(0.,0.,100.,6);self.cut_out=number(10.,0.,100.,6);self.efficiency=number(.9,0.,1.,6);self.operating_source=QLineEdit()
        self.operating_source.setPlaceholderText('Controller assumption and generator/drivetrain efficiency source')
        for name,w in [('Cut-in speed (m/s)',self.cut_in),('Cut-out speed (m/s)',self.cut_out),('Electrical / shaft efficiency',self.efficiency),('Operating rule provenance',self.operating_source)]:form.addRow(name,w)
        note=QLabel('Fixed commanded RPM. Coupled mode checks local rotor velocity before own-induction correction; power planning checks wake-corrected incoming rotor mean velocity. Cut-in is inclusive, cut-out exclusive; provide thresholds for the stated convention. Immediate reactivation, no hysteresis or parked drag. Efficiency changes electrical output only. Rated-power/pitch and variable-RPM control are not implemented.');note.setWordWrap(True);form.addRow(note)
        for w in [self.growth,self.width,self.distance,self.cut_in,self.cut_out,self.efficiency]:w.valueChanged.connect(self._changed)
        for w in [self.provenance,self.operating_source]:w.textChanged.connect(self._changed)
        self.table.cellChanged.connect(self._changed);self.operation.toggled.connect(self._changed);self._changed()

    def _changed(self,*args):
        for w in [self.cut_in,self.cut_out,self.efficiency,self.operating_source]:w.setEnabled(self.operation.isChecked())
        if not self.loading:self.changed.emit()

    def add_row(self):
        self.table.insertRow(self.table.rowCount());self._changed()

    def remove_row(self):
        i=self.table.currentRow()
        if i>=0:self.table.removeRow(i);self._changed()

    def state(self):
        return dict(rows=[[self.table.item(i,j).text() if self.table.item(i,j) else '' for j in range(4)] for i in range(self.table.rowCount())],growth=self.growth.value(),width=self.width.value(),distance=self.distance.value(),provenance=self.provenance.text(),operation=self.operation.isChecked(),cut_in=self.cut_in.value(),cut_out=self.cut_out.value(),efficiency=self.efficiency.value(),operating_source=self.operating_source.text())

    def restore(self,s):
        self.loading=True
        try:
            self.table.setRowCount(len(s['rows']))
            for i,row in enumerate(s['rows']):
                for j,text in enumerate(row):self.table.setItem(i,j,QTableWidgetItem(str(text)))
            for key,w in [('growth',self.growth),('width',self.width),('distance',self.distance),('cut_in',self.cut_in),('cut_out',self.cut_out),('efficiency',self.efficiency)]:w.setValue(s[key])
            self.provenance.setText(s['provenance']);self.operating_source.setText(s['operating_source']);self.operation.setChecked(s['operation'])
        finally:self.loading=False
        self._changed()

    def value(self,name,rpm):
        tables={}
        for i in range(self.table.rowCount()):
            try:u,tsr,ct,cp=[float(self.table.item(i,j).text()) for j in range(4)]
            except (ValueError,AttributeError):raise ValueError('Complete every speed-surface row: U_m_s, TSR, Ct, Cp; remove unfinished rows.')
            tables.setdefault(u,[]).append(dict(tip_speed_ratio=tsr,thrust_coefficient=ct,power_coefficient=cp))
        c=dict(mode='fixed_rpm_surface',model_name=name,rotational_speed_rpm=rpm,outside_surface='error',performance=[dict(inflow_m_s=u,curve=rows) for u,rows in sorted(tables.items())],wake=dict(method=self.wake_method,growth=self.growth.value(),initial_width=self.width.value(),minimum_distance_D=self.distance.value(),provenance=self.provenance.text().strip()))
        if self.operation.isChecked():c['operation']=dict(method='local_normal_speed_window',cut_in_m_s=self.cut_in.value(),cut_out_m_s=self.cut_out.value(),electrical_efficiency=self.efficiency.value(),provenance=self.operating_source.text().strip())
        return c

    def load(self,c):
        self.wake_method=c['wake']['method'] if c['wake']['method']=='gaussian_zero_mean_single_wake' else 'gaussian_zero_mean_linear_overlap'
        s=self.state();w=c['wake'];op=c.get('operation')
        s.update(rows=[[t['inflow_m_s'],r['tip_speed_ratio'],r['thrust_coefficient'],r['power_coefficient']] for t in c['performance'] for r in t['curve']],growth=w['growth'],width=w['initial_width'],distance=w['minimum_distance_D'],provenance=w['provenance'],operation=op is not None)
        if op:s.update(cut_in=op['cut_in_m_s'],cut_out=op['cut_out_m_s'],efficiency=op['electrical_efficiency'],operating_source=op['provenance'])
        self.restore(s)

    def read_csv(self,path):
        with open(path,encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            if reader.fieldnames!=['U_m_s','TSR','Ct','Cp']:raise ValueError('Use CSV headers U_m_s,TSR,Ct,Cp, with one speed/TSR point per row.')
            rows=[[str(float(r[k])) for k in reader.fieldnames] for r in reader]
        if not rows:raise ValueError('Performance CSV is empty.')
        s=self.state();s['rows']=rows;self.restore(s)

    def import_csv(self):
        path,_=QFileDialog.getOpenFileName(self,'Import device speed / TSR surface','','CSV (*.csv)')
        if path:
            try:self.read_csv(path)
            except Exception as e:QMessageBox.warning(self,'Device surface',str(e))

    def export_csv(self):
        path,_=QFileDialog.getSaveFileName(self,'Export device speed / TSR surface','turbine_speed_TSR_Ct_Cp.csv','CSV (*.csv)')
        if path:
            try:
                with open(path,'w',encoding='utf-8',newline='') as f:
                    w=csv.writer(f);w.writerow(['U_m_s','TSR','Ct','Cp']);w.writerows(self.state()['rows'])
            except Exception as e:QMessageBox.warning(self,'Device surface',str(e))
