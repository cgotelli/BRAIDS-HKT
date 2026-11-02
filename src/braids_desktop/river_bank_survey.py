"""Standalone geographic survey snapshots; no QGIS dependency or hidden CRS conversion."""
from copy import deepcopy
import json
from pathlib import Path
import math
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QWidget, QFormLayout, QHBoxLayout, QPushButton,
    QLabel, QDoubleSpinBox, QLineEdit, QFileDialog,QComboBox,QDialog,
    QVBoxLayout,QPlainTextEdit,QDialogButtonBox)
from braids_core.river_bank_survey import SCHEMA, POLICIES, validate
from .presentation import DisplaySpinBox


class RiverBankSurveyEditor(QWidget):
    changed = Signal()
    preview_requested = Signal()
    draw_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._draft = None
        self._pending = {}
        self._drawing = None
        self.domain_crs = lambda: None
        self.localize_source=lambda path,kind='file':Path(path)
        form = QFormLayout(self);form.setContentsMargins(0,0,0,0)
        self.zone_id=QLineEdit('bank_1');self.zones=QComboBox()
        form.addRow('Bank zone ID',self.zone_id);form.addRow('Existing zone',self.zones)
        self.zones.currentTextChanged.connect(lambda name:self.zone_id.setText(name) if name else None)
        drawing=QWidget();buttons=QHBoxLayout(drawing);buttons.setContentsMargins(0,0,0,0)
        for label,kind in [('Draw bank','bank_polygon'),('Draw toe','bank_toe')]:
            button=QPushButton(label);button.clicked.connect(lambda _,k=kind:self.request_draw(k));buttons.addWidget(button)
        form.addRow(drawing)
        self.edit_button=QPushButton('Edit zone coordinates…');self.edit_button.clicked.connect(self.edit_coordinates)
        self.remove_button=QPushButton('Remove selected zone');self.remove_button.clicked.connect(self.remove_zone)
        form.addRow(self.edit_button);form.addRow(self.remove_button)
        self.polygons, self.lines = QLineEdit(), QLineEdit()
        self.polygon_id, self.line_id = QLineEdit('id'), QLineEdit('id')
        self.polygon_layer, self.line_layer = QLineEdit(), QLineEdit()
        for label, path in [('Bank polygons',self.polygons),('Toe polylines',self.lines)]:
            row=QWidget();layout=QHBoxLayout(row);layout.setContentsMargins(0,0,0,0)
            layout.addWidget(path);button=QPushButton('Browse…');layout.addWidget(button)
            button.clicked.connect(lambda _,w=path:self._browse(w))
            form.addRow(label,row)
        for label,w in [('Polygon layer (blank = first)',self.polygon_layer),
                        ('Polygon ID field',self.polygon_id),
                        ('Polyline layer (blank = first)',self.line_layer),('Polyline ID field',self.line_id)]:
            form.addRow(label,w)
        self.tolerance=DisplaySpinBox()
        self.tolerance.setRange(.000001,1e6);self.tolerance.setValue(.1)
        form.addRow('Face mapping tolerance (m)',self.tolerance)
        note=QLabel('Import matching bank polygons and toe polylines by unique ID. '
            'Their declared CRS must already match the projected calculation CRS in metres. '
            'The project stores a geometry snapshot; source edits require reimport. No automatic reprojection.')
        note.setWordWrap(True);form.addRow(note)
        self.import_button=QPushButton('Import survey geometry snapshot')
        self.load_snapshot=QPushButton('Open existing survey snapshot JSON…')
        self.preview_button=QPushButton('Preview survey geometry');self.preview_button.setEnabled(False)
        self.status=QLabel('No survey imported.');self.status.setWordWrap(True)
        for w in [self.import_button,self.load_snapshot,self.preview_button,self.status]:form.addRow(w)
        self.import_button.clicked.connect(self.import_layers)
        self.load_snapshot.clicked.connect(self._load_snapshot)
        self.preview_button.clicked.connect(self.preview_requested)

    def _zone_name(self):
        name=self.zone_id.text().strip()
        if not name:raise ValueError('Enter a unique bank zone ID before drawing or editing.')
        return name

    def request_draw(self,kind):
        if kind not in ('bank_polygon','bank_toe'):raise ValueError('Unknown bank geometry tool')
        try:
            self._drawing=(kind,self._zone_name())
            self.status.setText('Draw the bank polygon.' if kind=='bank_polygon' else 'Draw the open toe polyline.')
            self.draw_requested.emit(kind)
        except ValueError as exc:self.status.setText(str(exc))

    def _definition(self):
        if self._draft is not None:return self.draft()
        crs=self.domain_crs()
        if not crs:raise ValueError('Define a calculation domain CRS before drawing banks.')
        return dict(schema=SCHEMA,spatial_reference=crs,policies=dict(POLICIES),fraction=.5,
                    critical_undercut_m=None,zones=[])

    def _replace_zone(self,zone):
        doc=self._definition()
        doc['zones']=[z for z in doc['zones'] if z['id']!=zone['id']]+[deepcopy(zone)]
        validated=validate(doc,self.domain_crs())
        self.restore(validated);self.zone_id.setText(zone['id']);self.changed.emit()
        return self.draft()

    def receive_drawn(self,kind,points):
        """Commit complete valid survey zones, never an incomplete live snapshot."""
        try:
            if kind not in ('bank_polygon','bank_toe'):raise ValueError('Unknown bank geometry tool')
            name=self._drawing[1] if self._drawing and self._drawing[0]==kind else self._zone_name()
            coordinates=[[float(x),float(y)] for x,y in points]
            if any(not math.isfinite(v) for p in coordinates for v in p):raise ValueError('Bank coordinates must be finite.')
            from shapely.geometry import Polygon,LineString
            if kind=='bank_polygon':
                if len(coordinates)<3:raise ValueError('A bank polygon requires at least three vertices.')
                geometry=Polygon(coordinates)
                if not geometry.is_valid or not geometry.is_simple or geometry.area<=0:raise ValueError('Draw a simple bank polygon with positive area.')
            else:
                if len(coordinates)<2:raise ValueError('A toe polyline requires at least two vertices.')
                geometry=LineString(coordinates)
                if not geometry.is_simple or geometry.is_ring or geometry.length<=0:raise ValueError('Draw one simple open toe polyline.')
            existing=next((z for z in (self._draft or {}).get('zones',[]) if z['id']==name),None)
            zone=deepcopy(self._pending.get(name,existing or dict(id=name,holes_xy=[],tolerance_m=self.tolerance.value())))
            zone['shell_xy' if kind=='bank_polygon' else 'toe_xy']=coordinates
            if kind=='bank_polygon':zone['holes_xy']=[]
            zone['tolerance_m']=self.tolerance.value()
            if 'shell_xy' not in zone or 'toe_xy' not in zone:
                self._pending[name]=zone;self._drawing=None
                missing='polygon' if 'shell_xy' not in zone else 'toe line'
                self.status.setText(f'{name}: geometry held; draw the {missing} to complete this zone.')
                return None
            pending={k:v for k,v in self._pending.items() if k!=name}
            result=self._replace_zone(zone);self._pending=pending;self._drawing=None
            return result
        except Exception as exc:
            self.status.setText('Survey not replaced: '+str(exc));raise

    def edit_coordinates(self):
        try:
            name=self._zone_name()
            zone=next((z for z in (self._draft or {}).get('zones',[]) if z['id']==name),None)
            if zone is None:raise ValueError('Select a completed bank zone first.')
            dialog=QDialog(self);dialog.setWindowTitle('Edit bank polygon and toe coordinates')
            layout=QVBoxLayout(dialog);editor=QPlainTextEdit(json.dumps(zone,indent=2));layout.addWidget(editor)
            status=QLabel('Coordinates use the projected domain CRS in metres.');status.setWordWrap(True);layout.addWidget(status)
            controls=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel);layout.addWidget(controls)
            controls.rejected.connect(dialog.reject)
            def save():
                try:
                    candidate=json.loads(editor.toPlainText())
                    if candidate.get('id')!=name:raise ValueError('Keep the existing ID when editing; draw a new ID to add another zone.')
                    self._replace_zone(candidate);dialog.accept()
                except Exception as exc:status.setText('Not saved: '+str(exc))
            controls.accepted.connect(save);dialog.resize(550,600);dialog.exec()
        except Exception as exc:self.status.setText(str(exc))

    def remove_zone(self):
        try:
            name=self._zone_name();doc=self._definition()
            if not any(z['id']==name for z in doc['zones']):raise ValueError('Select an existing completed bank zone first.')
            doc['zones']=[z for z in doc['zones'] if z['id']!=name]
            self.restore(doc if doc['zones'] else None);self.changed.emit()
        except Exception as exc:self.status.setText(str(exc))

    def _browse(self,field):
        filename,_=QFileDialog.getOpenFileName(self,'Survey vector layer',field.text(),
            'Vector data (*.gpkg *.geojson *.json *.shp);;All files (*)')
        if filename:
            try:field.setText(str(self.localize_source(filename,'vector')))
            except Exception as exc:self.status.setText(str(exc))

    def _load_snapshot(self):
        filename,_=QFileDialog.getOpenFileName(self,'Bank survey snapshot','','JSON (*.json)')
        if not filename:return
        try:
            filename=self.localize_source(filename)
            value=json.loads(Path(filename).read_text(encoding='utf-8'))
            self.restore(validate(value,self.domain_crs()))
            self.changed.emit()
        except Exception as exc:self.status.setText('Survey not replaced: '+str(exc))

    def import_layers(self):
        try:self.capture()
        except Exception as exc:self.status.setText('Survey not replaced: '+str(exc))

    def capture(self):
        from osgeo import ogr,osr
        crs=self.domain_crs()
        if not crs:raise ValueError('Define the calculation domain CRS first')
        for field in (self.polygons,self.lines):field.setText(str(self.localize_source(field.text(),'vector')))
        target=osr.SpatialReference()
        if target.SetFromUserInput(crs)!=0:raise ValueError('Invalid calculation CRS')
        rows=[]
        for filename,layer_name,field,kind in [
            (self.polygons.text(),self.polygon_layer.text(),self.polygon_id.text(),'polygon'),
            (self.lines.text(),self.line_layer.text(),self.line_id.text(),'line')]:
            dataset=ogr.Open(filename,0)
            if dataset is None:raise ValueError('Cannot open '+kind+' dataset: '+filename)
            layer=dataset.GetLayerByName(layer_name) if layer_name else dataset.GetLayer(0)
            if layer is None or layer.GetLayerDefn().GetFieldIndex(field)<0:
                raise ValueError('Choose an existing '+kind+' layer and ID field')
            reference=layer.GetSpatialRef()
            if reference is None or not reference.IsSame(target):
                raise ValueError('Survey CRS differs from domain; reproject explicitly before import')
            found={}
            for feature in layer:
                raw=feature.GetField(field)
                if raw is None:raise ValueError('Survey IDs cannot be null')
                name=str(raw).strip();geometry=feature.GetGeometryRef()
                if not name or name in found:raise ValueError('Unique nonempty IDs required in each survey layer')
                if geometry is None or geometry.IsEmpty():raise ValueError('Survey geometry cannot be empty')
                value=json.loads(geometry.ExportToJson())
                expected='Polygon' if kind=='polygon' else 'LineString'
                if value['type']!=expected:raise ValueError('One single-part '+expected+' required per ID')
                coordinates=value['coordinates']
                if kind=='polygon':
                    found[name]=dict(shell_xy=[p[:2] for p in coordinates[0]],
                                    holes_xy=[[p[:2] for p in ring] for ring in coordinates[1:]])
                else:found[name]=[p[:2] for p in coordinates]
            rows.append(found)
        if set(rows[0])!=set(rows[1]):raise ValueError('Polygon and toe-polyline IDs must match exactly')
        doc=dict(schema=SCHEMA,spatial_reference=crs,policies=dict(POLICIES),fraction=.5,
            critical_undercut_m=None,zones=[dict(id=name,**rows[0][name],toe_xy=rows[1][name],
                tolerance_m=self.tolerance.value()) for name in sorted(rows[0])])
        self.restore(validate(doc,crs));self.changed.emit();return self.draft()

    def draft(self):return deepcopy(self._draft)

    def restore(self,doc):
        self._draft=None if doc is None else validate(doc,doc['spatial_reference'])
        self._pending={};self._drawing=None
        previous=self.zone_id.text().strip();ids=[z['id'] for z in (doc or {}).get('zones',[])]
        self.zones.blockSignals(True);self.zones.clear();self.zones.addItems(ids)
        if ids:
            selected=previous if previous in ids else ids[0]
            self.zones.setCurrentText(selected);self.zone_id.setText(selected)
        else:self.zone_id.setText('bank_1')
        self.zones.blockSignals(False)
        self.preview_button.setEnabled(doc is not None)
        self.edit_button.setEnabled(doc is not None);self.remove_button.setEnabled(doc is not None)
        self.status.setText('No survey imported.' if doc is None else
            f"Saved snapshot: {len(doc['zones'])} bank zones. Draw or edit a selected ID to replace its geometry.")
        if doc is not None:self.tolerance.setValue(doc['zones'][0]['tolerance_m'])

    def value(self,crs,fraction,critical):
        if self._draft is None:raise ValueError('Import bank polygon and toe-polyline snapshots first')
        doc=self.draft();doc.update(fraction=fraction,critical_undercut_m=critical)
        return validate(doc,crs)
