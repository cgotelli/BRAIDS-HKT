"""Geographic layer snapshots for the unified Bed & banks tab."""
from copy import deepcopy
from qgis.PyQt.QtCore import pyqtSignal, QVariant
from qgis.PyQt.QtWidgets import QWidget,QFormLayout,QPushButton,QLabel,QDoubleSpinBox
from qgis.core import (QgsMapLayerProxyModel,QgsProject,QgsGeometry,QgsPointXY,
    QgsCoordinateReferenceSystem,QgsVectorLayer,QgsField,QgsFeature,QgsLineSymbol)
from qgis.gui import QgsMapLayerComboBox,QgsFieldComboBox
from braids_core.river_bank_survey import SCHEMA,POLICIES,validate


class RiverBankSurveyEditor(QWidget):
    changed=pyqtSignal()

    def __init__(self,parent=None):
        super().__init__(parent);self._draft=None;self.domain_crs=lambda:None;self._preview=[]
        form=QFormLayout(self);form.setContentsMargins(0,0,0,0)
        self.polygons=QgsMapLayerComboBox();self.polygons.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        self.lines=QgsMapLayerComboBox();self.lines.setFilters(QgsMapLayerProxyModel.LineLayer)
        self.polygon_id=QgsFieldComboBox();self.line_id=QgsFieldComboBox()
        for combo,field in ((self.polygons,self.polygon_id),(self.lines,self.line_id)):
            combo.setAllowEmptyLayer(True);combo.setLayer(None)
            combo.layerChanged.connect(field.setLayer)
        self.tolerance=QDoubleSpinBox();self.tolerance.setDecimals(6);self.tolerance.setRange(.000001,1e6);self.tolerance.setValue(.1)
        self.import_button=QPushButton('Use selected survey layers (snapshot)')
        self.preview_button=QPushButton('Show prepared bank map');self.preview_button.setEnabled(False)
        self.note=QLabel('Select bank polygons and toe lines with matching unique ID fields. '
            'Layers must use the domain CRS in metres. Import copies their geometry into the project; '
            'reimport after editing the source layers. The finest-grid footprint is shown after preparation.')
        self.note.setWordWrap(True)
        self.status=QLabel('No survey imported.');self.status.setWordWrap(True)
        for title,w in (('Eligible bank polygons',self.polygons),('Polygon ID field',self.polygon_id),
                ('Toe polylines',self.lines),('Line ID field',self.line_id),('Face mapping tolerance (m)',self.tolerance)):
            form.addRow(title,w)
        for w in (self.note,self.import_button,self.status,self.preview_button):form.addRow(w)
        self.import_button.clicked.connect(self.import_layers)

    def import_layers(self):
        try:self.capture()
        except Exception as exc:self.status.setText('Survey not replaced: '+str(exc))

    def capture(self):
        crs=self.domain_crs()
        if not crs:raise ValueError('Select a calculation domain first')
        target=QgsCoordinateReferenceSystem(crs)
        rows=[]
        for layer,field,kind in ((self.polygons.currentLayer(),self.polygon_id.currentField(),'polygon'),
                                 (self.lines.currentLayer(),self.line_id.currentField(),'line')):
            if layer is None or not layer.isValid() or layer.fields().indexFromName(field)<0:
                raise ValueError('Choose a valid '+kind+' layer and ID field')
            if layer.crs()!=target:raise ValueError('Survey layers must already use the domain CRS; reproject explicitly')
            found={}
            for f in layer.getFeatures():
                if f[field] is None or f[field]==QVariant():raise ValueError('Survey ID cannot be null')
                name=str(f[field]).strip();geom=f.geometry()
                if not name or name in found:raise ValueError('Unique nonempty IDs required in each layer')
                if geom.isEmpty() or geom.isMultipart():raise ValueError('One single-part geometry per survey ID required')
                if kind=='polygon':
                    rings=geom.asPolygon()
                    if not rings:raise ValueError('Polygon geometry required')
                    found[name]=dict(shell_xy=[[p.x(),p.y()] for p in rings[0]],
                                     holes_xy=[[[p.x(),p.y()] for p in r] for r in rings[1:]])
                else:
                    found[name]=[[p.x(),p.y()] for p in geom.asPolyline()]
            rows.append(found)
        if set(rows[0])!=set(rows[1]):raise ValueError('Polygon and line IDs must match exactly')
        doc=dict(schema=SCHEMA,spatial_reference=crs,policies=dict(POLICIES),fraction=.5,critical_undercut_m=None,
            zones=[dict(id=k,**rows[0][k],toe_xy=rows[1][k],tolerance_m=self.tolerance.value()) for k in sorted(rows[0])])
        self.restore(validate(doc,crs));self.changed.emit()
        return self.draft()

    def draft(self):return deepcopy(self._draft)

    def restore(self,doc):
        self._draft=None if doc is None else validate(doc,doc['spatial_reference'])
        self.status.setText('No survey imported.' if doc is None else
            f"Saved snapshot: {len(doc['zones'])} bank zones. Reimport to apply source-layer or tolerance edits.")
        if doc is not None:self.tolerance.setValue(doc['zones'][0]['tolerance_m'])

    def value(self,crs,fraction,critical):
        if self._draft is None:raise ValueError('Import geographic bank polygons and toe lines first')
        doc=self.draft();doc.update(fraction=fraction,critical_undercut_m=critical)
        return validate(doc,crs)

    def clear_preview(self):
        for layer in self._preview:
            if QgsProject.instance().mapLayer(layer.id()):QgsProject.instance().removeMapLayer(layer.id())
        self._preview=[]

    def show_prepared(self,doc):
        """Display authenticated compiled geometry, never reclassify live layers."""
        from braids_core.river_bank_survey import BankSurveyMap
        survey=BankSurveyMap.from_document(doc,doc['definition']);mapping=survey.mapping
        crs=QgsCoordinateReferenceSystem(survey.definition['spatial_reference'])
        polygon=QgsVectorLayer('Polygon','BRAIDS protected bank corridor','memory');polygon.setCrs(crs)
        polygon.dataProvider().addAttributes([QgsField('node',QVariant.Int),QgsField('role',QVariant.String)])
        polygon.updateFields();eligible={v for row in mapping['groups'].values() for v in row};features=[]
        for row in mapping['cells']:
            f=QgsFeature(polygon.fields());f.setAttributes([row['node'],'eligible bank' if row['node'] in eligible else 'toe recipient'])
            f.setGeometry(QgsGeometry.fromPolygonXY([[QgsPointXY(*p) for p in ring]
                for ring in [row['vertices_xy'],*row['holes_xy']]]));features.append(f)
        polygon.dataProvider().addFeatures(features);polygon.updateExtents();polygon.setOpacity(.35)
        line=QgsVectorLayer('LineString','BRAIDS resolved bank toes','memory');line.setCrs(crs)
        line.dataProvider().addAttributes([QgsField('zone',QVariant.String),QgsField('bank_node',QVariant.Int)])
        line.updateFields();features=[]
        for row in mapping['faces']:
            f=QgsFeature(line.fields());f.setAttributes([row['zone'],row['bank']])
            f.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(*p) for p in row['endpoints']]));features.append(f)
        line.dataProvider().addFeatures(features);line.updateExtents()
        line.renderer().setSymbol(QgsLineSymbol.createSimple({'line_color':'#2369b2','line_width':'.8'}))
        self.clear_preview();self._preview=[polygon,line]
        for layer in self._preview:QgsProject.instance().addMapLayer(layer)
        self.status.setText(f"Prepared map: {len(eligible)} bank columns, {len(mapping['faces'])} toe faces; "
            f"{len(mapping['required_leaves'])} finest columns protected. No automatic corridor coarsening.")
        return self._preview
