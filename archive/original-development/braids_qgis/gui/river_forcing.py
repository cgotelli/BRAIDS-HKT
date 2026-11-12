"""Per-boundary and per-fraction editors for the common river worker."""
from qgis.PyQt.QtCore import Qt,pyqtSignal
from qgis.PyQt.QtWidgets import QWidget,QVBoxLayout,QFormLayout,QLabel,QTableWidget,QTableWidgetItem,QHeaderView,QHBoxLayout,QPushButton
from braids_core.river_forcing import ForcingDraft
from braids_core.river_schedule import ScheduleDraft,CONSTANT,LINEAR
from braids_core.river_boundary import FEED_FIELDS,feed_field,FLOW_SOLID,FLOW_BULK,FLOW_MASS,FLOW_FREE,NATIVE
from .river_physics import combo,set_choice,section,WrappedNote


class RiverForcingEditor(QWidget):
    changed=pyqtSignal()

    def __init__(self,parent=None):
        super().__init__(parent)
        self.draft=ForcingDraft();self.schedules=ScheduleDraft();self._busy=False;self._context=None;self._shown=None;self._grains=[]
        self._enabled=False;self._suspended=False
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        self.boundary_box,form=section(layout,'Sediment boundary forcing')
        self.boundary=combo([])
        self.method=combo([('Imposed inward solid-volume flux','feed'),('Capacity outflow; no prescribed bedload inflow','outflow'),
            ('FLOW free / equilibrium transport; computed inflow and outflow',FLOW_FREE),
            ('FLOW prescribed solid volume; free on reversal',FLOW_SOLID),
            ('FLOW prescribed bulk volume; free on reversal',FLOW_BULK),
            ('FLOW prescribed mass; free on reversal',FLOW_MASS)])
        form.addRow('Open boundary',self.boundary);form.addRow('Bedload boundary method',self.method)
        self.table=QTableWidget(0,4)
        self.table.setHorizontalHeaderLabels(['Fraction ID','Size (mm)','Inward feed\n(m²/s solid)','Incoming concentration\n(kg/m³)'])
        self.table.setMinimumHeight(150);self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True);form.addRow(self.table)
        self.temporal=combo([('Constant values above',CONSTANT),('Linear time table below',LINEAR)])
        form.addRow('Time dependence',self.temporal)
        self.series=QTableWidget(0,1);self.series.setMinimumHeight(150)
        self.series.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        form.addRow(self.series)
        self.series_buttons=QWidget();buttons=QHBoxLayout(self.series_buttons);buttons.setContentsMargins(0,0,0,0)
        self.add_time=QPushButton('Add time row');self.remove_time=QPushButton('Remove selected row')
        buttons.addWidget(self.add_time);buttons.addWidget(self.remove_time);form.addRow(self.series_buttons)
        self.note=WrappedNote();form.addRow(self.note)
        self.initial_box,form=section(layout,'Initial suspended sediment')
        self.initial=QTableWidget(0,3)
        self.initial.setHorizontalHeaderLabels(['Fraction ID','Size (mm)','Initial concentration (kg/m³)'])
        self.initial.setMinimumHeight(130);self.initial.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.initial.horizontalHeader().setStretchLastSection(True);form.addRow(self.initial)
        note=WrappedNote();note.setText('Uniform initial concentration in wet cells; dry cells start with zero suspended mass. '
            'Values follow fraction IDs, not row positions. Removed or renamed fractions and boundaries keep inactive drafts; '
            'they are never automatically reassigned.')
        form.addRow(note)
        self.boundary.currentIndexChanged.connect(self._choose)
        self.method.currentIndexChanged.connect(self._edit_method)
        self.table.itemChanged.connect(self._edit);self.initial.itemChanged.connect(self._edit)
        self.series.itemChanged.connect(self._edit);self.temporal.currentIndexChanged.connect(self._edit_temporal)
        self.add_time.clicked.connect(self._add_time);self.remove_time.clicked.connect(self._remove_time)

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

    def sync(self,grains,boundaries,*,enabled,suspended):
        # Read raw display sizes so an unfinished inactive material draft does
        # not prevent an otherwise valid hydrodynamics-only project.
        context=(tuple(grains),tuple(sorted(boundaries.items())))
        self._remember()
        self.draft.ensure([g for g,_ in grains],boundaries)
        self.schedules.ensure([g for g,_ in grains],boundaries)
        self._enabled=enabled;self._suspended=suspended
        if context!=self._context:
            self._busy=True
            try:
                current=self.boundary.currentData();self.boundary.clear()
                for name,kind in sorted(boundaries.items()):self.boundary.addItem(name+' — '+kind,name)
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
            if row is not None:set_choice(self.method,row['kind'])
            key=feed_field(self.method.currentData()) or 'solid_flux_m2_s'
            units={'solid_flux_m2_s':'m²/s solid','bulk_flux_m2_s':'m²/s bulk','mass_flux_kg_m_s':'kg/(m s)'}[key]
            self.table.horizontalHeaderItem(2).setText('Inward feed\n('+units+')')
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
        self._availability()

    def _availability(self):
        self.boundary.setEnabled(self._enabled and self.boundary.count()>0)
        self.method.setEnabled(self._enabled and self._shown is not None)
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
                for c,active in ((2,self._enabled and feed_field(self.method.currentData()) is not None),
                                 (3,self._enabled and self._suspended)):
                    item=self.table.item(r,c)
                    item.setFlags((item.flags() | Qt.ItemIsEditable | Qt.ItemIsEnabled) if active
                                  else (item.flags() & ~Qt.ItemIsEditable & ~Qt.ItemIsEnabled))
            for r in range(self.series.rowCount()):
                for c in range(1,self.series.columnCount()):
                    active=self._enabled and (feed_field(self.method.currentData()) is not None if c%2 else self._suspended)
                    item=self.series.item(r,c)
                    item.setFlags((item.flags() | Qt.ItemIsEditable | Qt.ItemIsEnabled) if active
                        else (item.flags() & ~Qt.ItemIsEditable & ~Qt.ItemIsEnabled))
        finally:self._busy=busy
        if not self._enabled:
            text='Sediment is off: all feed and concentrations are inactive and execute as zero.'
        else:
            text=('Feed is per unit boundary width and per fraction, in the displayed units; it is not a fraction of total feed. '
                  'Capacity outflow exports local bedload and blocks unprescribed bedload inflow. ')
            if self.method.currentData() in NATIVE:
                text+='FLOW policy prescribes only on non-outward, hydraulically open faces; local outward flow keeps computed transport. '
                text+='Bulk volume converts with dry bulk / solid density; mass converts with solid density. '
            elif self.method.currentData()==FLOW_FREE:
                text+='Free / equilibrium transport uses the computed signed capacity in either direction, with interior-extrapolated bed and composition. '
                text+='It does not fix the bed, impose a feed or guarantee a steady morphological state. '
            text+=('Incoming concentration is used only when water enters, including outlet backflow. '
                   'Outflow carries interior concentration.' if self._suspended else
                   'Suspension is off: concentration drafts are inactive and execute as zero.')
        self.note.setText(text+' Each unit basis retains separate values; changing units does not convert or reuse the old input. '
            'Disabled values are retained as editor drafts only. '
            'Linear tables use physical seconds, start at zero and must cover the run end. '
            'They replace the constant values; each interval is sampled at its midpoint and steps end at table breakpoints.')

    def values(self,ids,boundaries,*,sediment,suspended):
        self._remember()
        inactive=[name for name,row in self.schedules.data['boundaries'].items() if row['method']==LINEAR]
        return self.draft.active(ids,boundaries,sediment=sediment,suspended=suspended,inactive_boundaries=inactive)

    def editor_state(self):
        self._remember();return self.draft.snapshot()

    def schedule_values(self,ids,boundaries,*,sediment,suspended):
        self._remember();return self.schedules.active(ids,boundaries,sediment=sediment,suspended=suspended)

    def schedule_state(self):
        self._remember();return self.schedules.snapshot()

    def load_project(self,project):
        state=project.get('editor_state',{}).get('forcing')
        self.draft=ForcingDraft(state)
        self.draft.load_active(project,inactive_boundaries=[name for name,row in project['sediment_schedules'].items()
            if row['method']==LINEAR])
        self.schedules=ScheduleDraft(project.get('editor_state',{}).get('schedules'));self.schedules.load_active(project)
        self._context=None;self._shown=None;self._grains=[]
