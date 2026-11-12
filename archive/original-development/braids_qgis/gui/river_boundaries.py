"""One method-aware boundary editor; no alternate execution workspace."""
from qgis.PyQt.QtCore import Qt,pyqtSignal
from qgis.PyQt.QtWidgets import (QWidget,QVBoxLayout,QFormLayout,QLineEdit,QTableWidget,
    QTableWidgetItem,QHeaderView,QHBoxLayout,QPushButton)
from braids_core.river_boundary_draft import BoundaryDraft
from braids_core.river_open_boundary import HYDRAULICS,PARAMETERS,BED_METHODS,COMPOSITION_METHODS,STAGE_POLICIES,INITIALS,APPLICABILITY
from braids_core.river_exterior_schedule import BED_FIELDS,COMPOSITION_FIELDS
from braids_core.river_schedule import CONSTANT,LINEAR
from .river_physics import combo,set_choice,section,WrappedNote


class BoundaryValues(QWidget):
    """Text-preserving values and a physical-time table for one named law."""
    changed=pyqtSignal()

    def __init__(self,parent=None):
        super().__init__(parent);self._busy=False;self.source=None;self.fields=[];self.table_fields=[];self.inputs={}
        self.form=QFormLayout(self);self.form.setContentsMargins(0,0,0,0)
        self.constants=QWidget();self.value_form=QFormLayout(self.constants);self.value_form.setContentsMargins(0,0,0,0)
        self.form.addRow(self.constants)
        self.temporal=combo([('Constant',CONSTANT),('Linear table (physical seconds)',LINEAR)])
        self.form.addRow('Time dependence',self.temporal)
        self.series=QTableWidget(0,1);self.series.setMinimumHeight(130)
        self.series.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.form.addRow(self.series)
        self.buttons=QWidget();layout=QHBoxLayout(self.buttons);layout.setContentsMargins(0,0,0,0)
        self.add_time=QPushButton('Add time row');self.remove_time=QPushButton('Remove selected row')
        layout.addWidget(self.add_time);layout.addWidget(self.remove_time);self.form.addRow(self.buttons)
        self.note=WrappedNote();self.form.addRow(self.note)
        self.temporal.currentIndexChanged.connect(self._mode)
        self.series.itemChanged.connect(self._edit)
        self.add_time.clicked.connect(self._add);self.remove_time.clicked.connect(self._remove)

    def load(self,source,fields,table_fields,labels):
        self._busy=True
        try:
            self.source=source;self.fields=list(fields);self.table_fields=list(table_fields);self.inputs={}
            while self.value_form.rowCount():self.value_form.removeRow(0)
            for key in fields:
                w=QLineEdit(source['values'][key]);self.inputs[key]=w;self.value_form.addRow(labels[key],w)
                if key=='energy_slope':
                    w.setReadOnly(True);w.setToolTip('Automatically fitted from the signed inward fall of the prepared finest mesh bed. Normal flow assumes energy slope equals bed slope; the resolved value is recorded during preparation.')
                w.textChanged.connect(self._edit)
            set_choice(self.temporal,source['schedule']['method'])
            self.series.setColumnCount(1+len(table_fields))
            self.series.setHorizontalHeaderLabels(['Physical time (s)']+[labels[k] for k in table_fields])
            rows=source['schedule']['rows'];self.series.setRowCount(len(rows))
            for i,row in enumerate(rows):
                for j,value in enumerate([row['time_s']]+[row['values'][k] for k in table_fields]):
                    self.series.setItem(i,j,QTableWidgetItem(value))
        finally:self._busy=False
        self.availability()

    def remember(self):
        if self.source is None:return
        for key,w in self.inputs.items():self.source['values'][key]=w.text()
        rows=self.source['schedule']['rows']
        for i in range(self.series.rowCount()):
            rows[i]['time_s']=self.series.item(i,0).text()
            for j,key in enumerate(self.table_fields):rows[i]['values'][key]=self.series.item(i,j+1).text()

    def _edit(self,*_):
        if self._busy:return
        self.remember();self.changed.emit()

    def _mode(self,*_):
        if self._busy or self.source is None:return
        self.remember();self.source['schedule']['method']=self.temporal.currentData()
        self.availability();self.changed.emit()

    def _add(self):
        if self.source is None:return
        self.remember();self.source['schedule']['rows'].append(dict(time_s='',values={k:self.source['values'][k] for k in self.table_fields}))
        self._refresh_rows();self.changed.emit()

    def _remove(self):
        i=self.series.currentRow()
        if i<0 or self.source is None:return
        self.remember();del self.source['schedule']['rows'][i];self._refresh_rows();self.changed.emit()

    def _refresh_rows(self):
        labels={k:self.series.horizontalHeaderItem(j+1).text() for j,k in enumerate(self.table_fields)}
        labels.update({k:self.value_form.labelForField(w).text() for k,w in self.inputs.items()})
        self.load(self.source,self.fields,self.table_fields,labels)

    def availability(self):
        linear=self.temporal.currentData()==LINEAR
        self.temporal.setEnabled(bool(self.table_fields))
        self.series.setVisible(linear);self.buttons.setVisible(linear)
        for key,w in self.inputs.items():
            visible=not linear or key not in self.table_fields
            w.setVisible(visible);self.value_form.labelForField(w).setVisible(visible)
        self.note.setVisible(linear)
        self.note.setText('The table replaces the corresponding constants. Its first row is the explicit initial forcing. '
            'Start at 0 s, use strictly increasing times and cover the run end. No extrapolation.')


class RiverBoundaryEditor(QWidget):
    changed=pyqtSignal()

    def __init__(self,parent=None):
        super().__init__(parent);self.draft=BoundaryDraft();self._busy=False;self._shown=None;self._context=None
        self._kinds={};self._ids=[];self._enabled=False
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        self.box,form=section(layout,'Open-boundary conditions',expanded=True)
        self.boundary=combo([]);self.method=combo([]);self.policy=combo([(v,k) for k,v in STAGE_POLICIES.items()])
        form.addRow('Boundary',self.boundary);form.addRow('Hydraulic method',self.method);form.addRow('Stage reversal policy',self.policy)
        self.policy_label=form.labelForField(self.policy)
        self.hydraulic_values=BoundaryValues();form.addRow(self.hydraulic_values)
        self.hydraulic_note=WrappedNote();form.addRow(self.hydraulic_note)
        self.bed_box,form=section(layout,'Exterior bed and composition')
        self.bed_method=combo([(v,k) for k,v in BED_METHODS.items()])
        self.initial=combo([(v,k) for k,v in INITIALS.items()]);self.initial_elevation=QLineEdit('0')
        form.addRow('Exterior bed method',self.bed_method);form.addRow('Initial exterior bed',self.initial)
        form.addRow('Initial exterior elevation (m, DEM datum)',self.initial_elevation)
        self.initial_label=form.labelForField(self.initial);self.elevation_label=form.labelForField(self.initial_elevation)
        self.bed_values=BoundaryValues();form.addRow(self.bed_values)
        self.composition_method=combo([(v,k) for k,v in COMPOSITION_METHODS.items()])
        form.addRow('Exterior composition method',self.composition_method)
        self.composition_values=BoundaryValues();form.addRow(self.composition_values)
        self.note=WrappedNote();form.addRow(self.note)
        self.boundary.currentIndexChanged.connect(self._choose)
        for widget,key in ((self.method,'hydraulic'),(self.bed_method,'bed'),(self.composition_method,'composition')):
            widget.currentIndexChanged.connect(lambda _,w=widget,k=key:self._method(k,w.currentData()))
        self.policy.currentIndexChanged.connect(self._edit)
        self.initial.currentIndexChanged.connect(self._edit);self.initial_elevation.textChanged.connect(self._edit)
        for w in (self.hydraulic_values,self.bed_values,self.composition_values):w.changed.connect(self._edit)

    def row(self):
        return self.draft.data['boundaries'][self._shown][self._shown_role] if self._shown is not None else None

    def _remember(self):
        row=self.row()
        if row is None:return
        for w in (self.hydraulic_values,self.bed_values,self.composition_values):w.remember()
        row['stage_policy']=self.policy.currentData()
        b=row['beds'][row['bed']];b['initial']=self.initial.currentData();b['initial_elevation_m']=self.initial_elevation.text()

    def _edit(self,*_):
        if self._busy:return
        self._remember();self._availability();self.changed.emit()

    def _method(self,key,value):
        if self._busy or self._shown is None:return
        self._remember();self.row()[key]=value;self._render();self.changed.emit()

    def _choose(self,*_):
        if self._busy:return
        self._remember();self._shown=self.boundary.currentData()
        self._shown_role=self._kinds.get(self._shown);self._render()

    def sync(self,ids,kinds,*,enabled):
        self._remember();self.draft.ensure(ids,kinds)
        context=(tuple(ids),tuple(sorted(kinds.items())))
        self._ids=list(ids);self._kinds=dict(kinds);self._enabled=enabled
        if context!=self._context:
            self._busy=True
            try:
                current=self.boundary.currentData();self.boundary.clear()
                for name,kind in sorted(kinds.items()):self.boundary.addItem(name+' — '+kind,name)
                if self.boundary.findData(current)>=0:set_choice(self.boundary,current)
                self._shown=self.boundary.currentData();self._shown_role=kinds.get(self._shown);self._context=context
                self._render()
            finally:self._busy=False
        self._availability()

    def _render(self):
        busy=self._busy;self._busy=True
        try:
            row=self.row()
            if row is None:return
            self.method.clear()
            for m,(label,role,_,_) in HYDRAULICS.items():
                if role==self._shown_role:self.method.addItem(label,m)
            set_choice(self.method,row['hydraulic']);set_choice(self.policy,row['stage_policy'])
            m=row['hydraulic'];self.hydraulic_values.load(row['controls'][m],HYDRAULICS[m][2],HYDRAULICS[m][3],
                                                       {k:v[0] for k,v in PARAMETERS.items()})
            m=row['bed'];set_choice(self.bed_method,m);b=row['beds'][m]
            set_choice(self.initial,b['initial']);self.initial_elevation.setText(b['initial_elevation_m'])
            fields=[BED_FIELDS[m]] if BED_FIELDS[m] else []
            self.bed_values.load(b,fields,fields,{k:v[0] for k,v in PARAMETERS.items()})
            m=row['composition'];set_choice(self.composition_method,m)
            fields=self._ids if COMPOSITION_FIELDS[m] else []
            basis='mass' if m=='composition.mass_fraction' else 'reference volume'
            self.composition_values.load(row['compositions'][m],fields,fields,{g:g+' — '+basis+' fraction' for g in fields})
        finally:self._busy=busy
        self._availability()

    def _availability(self):
        self.box.setEnabled(self._shown is not None);self.bed_box.setEnabled(self._shown is not None and self._enabled)
        stage=self.method.currentData()=='outlet.stage'
        self.policy.setVisible(stage);self.policy_label.setVisible(stage)
        self.hydraulic_note.setText(APPLICABILITY.get(self.method.currentData(),''))
        prescribed=self.bed_method.currentData()=='bed.elevation'
        self.initial.setVisible(not prescribed);self.initial_label.setVisible(not prescribed)
        explicit=not prescribed and self.initial.currentData()=='initial.elevation'
        self.initial_elevation.setVisible(explicit);self.elevation_label.setVisible(explicit)
        self.note.setText(('Sediment is OFF: fixed prepared exterior bed; no composition forcing. Drafts below are inactive. ' if not self._enabled else
            'These are zero-area exterior states, not fixed or overwritten interior cells. '
            'Bed elevation is positive upward. Bed rates use morphological seconds; tables use physical seconds. '
            'Prescribed elevation also sets the explicit initial exterior elevation. Composition fractions must sum to 1. '
            'Reference-volume fractions use the selected bed density/porosity model, not necessarily solid volume. '
            'With a coarse layer, fixed composition resolves to copying the adjacent full column (shown in Run). ')+
            'Changing a method keeps its old values only as an inactive draft; no automatic conversion or normalization.')

    def values(self,ids,kinds,*,sediment,end_time):
        self._remember();return self.draft.active(ids,kinds,sediment=sediment,end_time=end_time)

    def editor_state(self):
        self._remember();return self.draft.snapshot()

    def load_project(self,project):
        self.draft=BoundaryDraft(project.get('editor_state',{}).get('boundaries'));self.draft.load_active(project)
        self._shown=None;self._context=None
