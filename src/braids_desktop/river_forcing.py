# Standalone PySide6 port of the maintained BRAIDS configuration editor.
"""Per-boundary and per-fraction editors for the common river worker."""
from PySide6.QtCore import Qt,Signal
from PySide6.QtWidgets import QWidget,QVBoxLayout,QFormLayout,QLabel,QTableWidget,QTableWidgetItem,QHeaderView,QHBoxLayout,QPushButton,QSizePolicy
from braids_core.river_forcing import ForcingDraft
from braids_core.river_schedule import ScheduleDraft,CONSTANT,LINEAR
from braids_core.river_boundary import FEED_FIELDS,feed_field,FLOW_SOLID,FLOW_BULK,FLOW_MASS,FLOW_FREE,NATIVE,TOTAL,RATIO
from .river_physics import combo,set_choice,section,WrappedNote
from copy import deepcopy


class RiverForcingEditor(QWidget):
    changed=Signal()

    def __init__(self,parent=None):
        super().__init__(parent)
        self.draft=ForcingDraft();self.schedules=ScheduleDraft();self._busy=False;self._context=None;self._shown=None;self._grains=[]
        self._enabled=False;self._suspended=False;self._bedload=False
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        # This step contains only sediment boundary controls, so a second
        # collapsible heading wastes vertical space and hides the only editor.
        self.boundary_box=None;form=QFormLayout();layout.addLayout(form)
        self.boundary_form=form
        self.boundary=combo([])
        self.method=combo([('Capacity (selected bedload formula)',FLOW_FREE),('Total sediment (solid m³/s)',TOTAL),('Sediment / water ratio (qs/q)',RATIO)])
        form.addRow('Inlet',self.boundary);form.addRow('Bed load supply method',self.method)
        self.table=QTableWidget(0,4)
        self.table.setHorizontalHeaderLabels(['Fraction ID','Size (mm)','Inward feed\n(m²/s solid)','Incoming concentration\n(kg/m³)'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.table.horizontalHeader().setStretchLastSection(True);form.addRow(self.table)
        self.temporal=combo([('Constant values above',CONSTANT),('Linear time table below',LINEAR)])
        form.addRow('Time dependence',self.temporal)
        # Initial suspended mass belongs alongside inlet timing rather than
        # below optional schedules and explanatory text.
        self.initial_box,initial_form=section(layout,'Initial suspended sediment')
        self.initial_box.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Maximum)
        self.initial_box.content.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Maximum)
        self.initial=QTableWidget(0,3)
        self.initial.setHorizontalHeaderLabels(['Fraction ID','Size (mm)','Initial concentration (kg/m³)'])
        self.initial.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.initial.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.initial.horizontalHeader().setStretchLastSection(True);initial_form.addRow(self.initial)
        initial_note=WrappedNote();initial_note.setText('Uniform initial concentration in wet cells; dry cells start with zero suspended mass. '
            'Values follow fraction IDs, not row positions. Removed or renamed fractions and boundaries keep inactive drafts; '
            'they are never automatically reassigned.')
        initial_form.addRow(initial_note)
        schedule_form=QFormLayout();layout.addLayout(schedule_form)
        self.series=QTableWidget(0,1)
        self.series.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.series.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        schedule_form.addRow(self.series)
        self.series_buttons=QWidget();buttons=QHBoxLayout(self.series_buttons);buttons.setContentsMargins(0,0,0,0)
        self.add_time=QPushButton('Add time row');self.remove_time=QPushButton('Remove selected row')
        buttons.addWidget(self.add_time);buttons.addWidget(self.remove_time);schedule_form.addRow(self.series_buttons)
        self.note=WrappedNote();schedule_form.addRow(self.note)
        self.boundary.currentIndexChanged.connect(self._choose)
        self.method.currentIndexChanged.connect(self._edit_method)
        self.table.itemChanged.connect(self._edit);self.initial.itemChanged.connect(self._edit)
        self.series.itemChanged.connect(self._edit);self.temporal.currentIndexChanged.connect(self._edit_temporal)
        self.add_time.clicked.connect(self._add_time);self.remove_time.clicked.connect(self._remove_time)
        layout.addStretch()

    @staticmethod
    def _fit_table(table):
        """Keep boundary tables no taller than their actual records."""
        header=table.horizontalHeader().sizeHint().height()
        rows=sum(max(32,table.rowHeight(row)) for row in range(table.rowCount()) if not table.isRowHidden(row))
        table.setFixedHeight(max(header+2*table.frameWidth()+4,header+rows+2*table.frameWidth()+4))

    def _fit_tables(self):
        for table in (self.table,self.initial,self.series):self._fit_table(table)
        self.initial_box.content.updateGeometry();self.initial_box.updateGeometry()

    def _remember(self):
        if self._shown is not None:
            row=self.draft.data['boundaries'][self._shown]
            key=feed_field(row['kind']) or 'solid_flux_m2_s'
            for r,(grain,_) in enumerate(self._grains):
                row['fractions'][grain][key]=self.table.item(r,2).text()
                row['fractions'][grain]['concentration_kg_m3']=self.table.item(r,3).text()
            records=self.schedules.data['boundaries'][self._shown]['rows']
            for r in range(self.series.rowCount()):
                records[r]['time_s']=self.series.item(r,0).text()
                for k,(grain,_) in enumerate(self._grains):
                    records[r]['fractions'][grain][key]=self.series.item(r,1+2*k).text()
                    records[r]['fractions'][grain]['concentration_kg_m3']=self.series.item(r,2+2*k).text()
        for r,(grain,_) in enumerate(self._grains):
            self.draft.data['initial'][grain]=self.initial.item(r,2).text()

    def _edit(self,*_):
        if self._busy:return
        self._remember();self.changed.emit()

    def _edit_method(self,*_):
        if self._busy or self._shown is None:return
        self._remember();self.draft.data['boundaries'][self._shown]['kind']=self.method.currentData()
        self._render();self.changed.emit()

    def _edit_temporal(self,*_):
        if self._busy or self._shown is None:return
        self._remember();self.schedules.data['boundaries'][self._shown]['method']=self.temporal.currentData()
        self._availability();self.changed.emit()

    def _add_time(self):
        self._remember();rows=self.schedules.data['boundaries'][self._shown]['rows']
        rows.append(dict(time_s='',fractions={g:{key:'0' for key in (*FEED_FIELDS,'concentration_kg_m3')} for g,_ in self._grains}))
        self._render();self.changed.emit()

    def _remove_time(self):
        r=self.series.currentRow()
        if r<0:return
        self._remember();del self.schedules.data['boundaries'][self._shown]['rows'][r]
        self._render();self.changed.emit()

    def _choose(self,*_):
        if self._busy:return
        self._remember();self._shown=self.boundary.currentData();self._render()

    def sync(self,grains,boundaries,*,enabled,suspended,bedload=True):
        # Read raw display sizes so an unfinished inactive material draft does
        # not prevent an otherwise valid hydrodynamics-only project.
        context=(tuple(grains),tuple(sorted(boundaries.items())))
        self._remember()
        previous=set(self.draft.data['boundaries'])
        self.draft.ensure([g for g,_ in grains],boundaries)
        for name,kind in boundaries.items():
            if kind=='outlet':self.draft.data['boundaries'][name]['kind']='outflow'
            elif name not in previous:self.draft.data['boundaries'][name]['kind']=FLOW_FREE
        self.schedules.ensure([g for g,_ in grains],boundaries)
        self._enabled=enabled;self._suspended=suspended;self._bedload=bedload
        if context!=self._context:
            self._busy=True
            try:
                current=self.boundary.currentData();self.boundary.clear()
                for name,kind in sorted(boundaries.items()):
                    if kind=='inlet':self.boundary.addItem(name,name)
                if self.boundary.findData(current)>=0:set_choice(self.boundary,current)
                self._grains=list(grains);self._context=context;self._shown=self.boundary.currentData()
                self._render()
            finally:self._busy=False
        self._availability()

    def _render(self):
        busy=self._busy;self._busy=True
        try:
            self.table.setRowCount(len(self._grains) if self._shown is not None else 0)
            self.initial.setRowCount(len(self._grains))
            row=self.draft.data['boundaries'].get(self._shown)
            if row is not None:set_choice(self.method,FLOW_FREE if row['kind']=='outflow' else row['kind'])
            key=feed_field(self.method.currentData()) or 'solid_flux_m2_s'
            units={'solid_flux_m2_s':'m²/s solid','bulk_flux_m2_s':'m²/s bulk','mass_flux_kg_m_s':'kg/(m s)',
                   'solid_discharge_m3_s':'m³/s solid','solid_water_ratio':'m³ solid / m³ water'}[key]
            self.table.horizontalHeaderItem(2).setText('Bed load sediment supply\nInward feed ('+units+')')
            scheduled=self.schedules.data['boundaries'].get(self._shown)
            self.series.setColumnCount(1+2*len(self._grains))
            self.series.setHorizontalHeaderLabels(['Time (s)']+[label for grain,_ in self._grains
                for label in (grain+' feed\n('+units+')',grain+' concentration\n(kg/m³)')])
            self.series.setRowCount(len(scheduled['rows']) if scheduled else 0)
            if scheduled:
                set_choice(self.temporal,scheduled['method'])
                for r,record in enumerate(scheduled['rows']):
                    values=[record['time_s']]+[record['fractions'][g][field] for g,_ in self._grains
                        for field in (key,'concentration_kg_m3')]
                    for c,value in enumerate(values):self.series.setItem(r,c,QTableWidgetItem(value))
            for r,(grain,size) in enumerate(self._grains):
                for table,values in ((self.initial,[grain,size,self.draft.data['initial'][grain]]),
                    (self.table,None if row is None else [grain,size,row['fractions'][grain][key],
                                                         row['fractions'][grain]['concentration_kg_m3']])):
                    if values is None:continue
                    for c,value in enumerate(values):
                        item=QTableWidgetItem(value)
                        if c<2:item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                        table.setItem(r,c,item)
        finally:self._busy=busy
        self._fit_tables()
        self._availability()

    def _availability(self):
        self.initial_box.setVisible(self._enabled and self._suspended)
        self.boundary_form.labelForField(self.method).setText('Bed load supply method')
        self.table.horizontalHeaderItem(0).setText('Material')
        self.initial.horizontalHeaderItem(0).setText('Material')
        # Each physical route identifies its own editable column; the table
        # does not imply that bedload and suspension are one combined supply.
        self.table.horizontalHeaderItem(3).setText('Suspended sediment supply\nIncoming concentration (kg/m³)')
        self.table.setColumnHidden(1,True)
        self.initial.setColumnHidden(1,True)
        self.table.setColumnHidden(2,not self._bedload or feed_field(self.method.currentData()) is None)
        for col in range(1,self.series.columnCount(),2):
            self.series.setColumnHidden(col,not self._bedload or feed_field(self.method.currentData()) is None)
        self.table.setColumnHidden(3,not self._suspended)
        for col in range(2,self.series.columnCount(),2):self.series.setColumnHidden(col,not self._suspended)
        self.boundary.setEnabled(self._enabled and self.boundary.count()>0)
        self.method.setEnabled(self._enabled and self._bedload and self._shown is not None)
        self.boundary_form.setRowVisible(self.method,self._bedload)
        scheduled=self.temporal.currentData()==LINEAR
        self.temporal.setEnabled(self._enabled and self._shown is not None)
        self.table.setEnabled(self._enabled and self._shown is not None and not scheduled)
        self.series.setVisible(scheduled);self.series_buttons.setVisible(scheduled)
        self.series.setEnabled(self._enabled and self._shown is not None)
        self.series_buttons.setEnabled(self._enabled and self._shown is not None)
        self.initial.setEnabled(self._enabled and self._suspended)
        busy=self._busy;self._busy=True
        try:
            for r in range(self.table.rowCount()):
                for c,active in ((2,self._enabled and self._bedload and feed_field(self.method.currentData()) is not None),
                                 (3,self._enabled and self._suspended)):
                    item=self.table.item(r,c)
                    item.setFlags((item.flags() | Qt.ItemIsEditable | Qt.ItemIsEnabled) if active
                                  else (item.flags() & ~Qt.ItemIsEditable & ~Qt.ItemIsEnabled))
            for r in range(self.series.rowCount()):
                for c in range(1,self.series.columnCount()):
                    active=self._enabled and (self._bedload and feed_field(self.method.currentData()) is not None if c%2 else self._suspended)
                    item=self.series.item(r,c)
                    item.setFlags((item.flags() | Qt.ItemIsEditable | Qt.ItemIsEnabled) if active
                        else (item.flags() & ~Qt.ItemIsEditable & ~Qt.ItemIsEnabled))
        finally:self._busy=busy
        # One-grain bedload workflow: equilibrium has no editable feed table.
        prescribed=feed_field(self.method.currentData()) is not None or self._suspended
        self.boundary_form.setRowVisible(self.table,prescribed)
        self.boundary_form.setRowVisible(self.temporal,prescribed)
        if not prescribed:self.series.hide();self.series_buttons.hide()
        if not self._enabled:
            self.note.setText('Enable bedload or suspended transport in 04 / General to set sediment boundary conditions.')
        elif self._suspended and not self._bedload:
            self.note.setText('Supply suspended concentration in kg/m³. No bedload is supplied or transported. Outlets export arriving suspension; reversal uses zero incoming concentration.')
        elif self.method.currentData()==FLOW_FREE:
            self.note.setText('')
        elif self.method.currentData()=='outflow':
            self.note.setText('Exports local bedload when water leaves. No sediment is supplied when water enters. Active after the sediment start time.')
        else:
            self.note.setText(('Total solid volume per second, distributed over incoming faces in proportion to their accepted water discharge. '
                if self.method.currentData()==TOTAL else 'Solid-volume ratio qs/q, multiplied by local inward water discharge. ')+
                'Zero gives clear-water inflow. No supply without incoming water. Time tables use physical seconds and must cover the end time; warm-up has no sediment feed.')
        self.note.setVisible(bool(self.note.text()))
        self._fit_tables()

    def values(self,ids,boundaries,*,sediment,suspended,bedload=True):
        self._remember()
        inactive=[name for name,row in self.schedules.data['boundaries'].items() if row['method']==LINEAR]
        draft=self.draft
        if not bedload:
            data=deepcopy(draft.data)
            for row in data['boundaries'].values():row['kind']='outflow'
            draft=ForcingDraft(data)
        rows,initial=draft.active(ids,boundaries,sediment=sediment,suspended=suspended,inactive_boundaries=inactive)
        for name,kind in boundaries.items():
            if kind=='outlet':rows[name]=dict(kind='outflow',concentration_kg_m3=[0.]*len(ids))
            elif sediment and bedload and rows[name]['kind'] not in (FLOW_FREE,TOTAL,RATIO):raise ValueError(name+': choose Capacity, Total sediment, or Sediment/water ratio')
        return rows,initial

    def editor_state(self):
        self._remember();return self.draft.snapshot()

    def schedule_values(self,ids,boundaries,*,sediment,suspended,bedload=True):
        self._remember()
        inlets={self.boundary.itemData(i) for i in range(self.boundary.count())}
        for name,row in boundaries.items():
            if row['kind']=='outflow' and (not suspended or name not in inlets):
                self.schedules.data['boundaries'][name]['method']=CONSTANT
        return self.schedules.active(ids,boundaries,sediment=sediment,suspended=suspended)

    def schedule_state(self):
        self._remember();return self.schedules.snapshot()

    def load_project(self,project):
        state=project.get('editor_state',{}).get('forcing')
        self.draft=ForcingDraft(state)
        self.draft.load_active(project,inactive_boundaries=[name for name,row in project['sediment_schedules'].items()
            if row['method']==LINEAR])
        if state and not project['settings']['suspension']['bed_factor']:
            # The execution document contains zero bedload outflow, not a new
            # user choice. Keep inactive feed methods/values across reopening.
            for name,row in state['boundaries'].items():
                target=self.draft.data['boundaries'][name];target['kind']=row['kind']
                for grain,fields in row['fractions'].items():
                    for key in FEED_FIELDS:target['fractions'][grain][key]=fields[key]
        self.schedules=ScheduleDraft(project.get('editor_state',{}).get('schedules'));self.schedules.load_active(project)
        self._context=None;self._shown=None;self._grains=[]

