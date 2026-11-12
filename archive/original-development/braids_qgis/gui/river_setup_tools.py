"""Map-facing model configuration, kept separate from numerical and Results UI."""
from uuid import uuid4
from pathlib import Path

from qgis.PyQt.QtCore import QObject
from qgis.PyQt.QtWidgets import QWidget,QHBoxLayout,QVBoxLayout,QPushButton,QLabel,QLineEdit,QCheckBox,QFileDialog
from qgis.core import (QgsProject,QgsVectorLayer,QgsRasterLayer,QgsMapLayerProxyModel,
    QgsCoordinateReferenceSystem,QgsFeature,QgsGeometry,QgsPointXY)
from qgis.gui import QgsMapLayerComboBox

from .river_digitizing import RiverDigitizing,validate_geometry
from .river_geometry import domain_from_layer
from braids_core.boundaries.gis import _projected_metres


class RiverSetupTools(QObject):
    def __init__(self,dock):
        super().__init__(dock);self.dock=dock
        self.note=QLabel();self.note.setWordWrap(True)
        self.digitizing=RiverDigitizing(dock.iface,self,self.message)
        self.digitizing.changed.connect(self.session_changed)
        self.snap=QCheckBox('Snap to vertices and segments while drawing / editing');self.snap.setChecked(True)
        self.guides=QgsMapLayerComboBox();self.guides.setFilters(QgsMapLayerProxyModel.LineLayer)
        self.guides.setAllowEmptyLayer(True);self.guides.setLayer(None)
        QgsProject.instance().layersAdded.connect(self.filter_guides)
        self.filter_guides()
        self.bank_id=QLineEdit('bank_1')
        self.session_widget=QWidget();row=QHBoxLayout(self.session_widget);row.setContentsMargins(0,0,0,0)
        row.addWidget(self.note,1)
        self.finish_button=self.add(row,'Finish edits',self.digitizing.finish_edit)
        self.cancel_button=self.add(row,'Cancel tool',self.digitizing.cancel)
        self.finish_button.setEnabled(False);self.cancel_button.setEnabled(False)

    def message(self,text):self.note.setText(text)

    def filter_guides(self,*_):
        self.guides.setExceptedLayerList([layer for layer in QgsProject.instance().mapLayers().values()
            if layer.customProperty('braids/input_role','')!='inlet_guides'])

    def call(self,callback):
        try:callback()
        except Exception as exc:
            self.message(str(exc))
            if hasattr(self.dock,'status'):self.dock.error(exc)

    def add(self,layout,title,callback):
        button=QPushButton(title);button.clicked.connect(lambda checked=False:self.call(callback));layout.addWidget(button)
        return button

    def row(self,actions):
        widget=QWidget();layout=QVBoxLayout(widget);layout.setContentsMargins(0,0,0,0)
        for start in range(0,len(actions),3):
            line=QHBoxLayout();layout.addLayout(line)
            for title,callback in actions[start:start+3]:self.add(line,title,callback)
        return widget

    def field(self,control,actions):
        widget=QWidget();layout=QVBoxLayout(widget);layout.setContentsMargins(0,0,0,0)
        layout.addWidget(control);layout.addWidget(self.row(actions));return widget

    def domain_field(self):
        return self.field(self.dock.domain,[('Load…',lambda:self.load_vector(self.dock.domain,'polygon')),
            ('Draw polygon',self.draw_domain),('Select',lambda:self.digitizing.select(self.dock.domain.currentLayer())),
            ('Edit vertices',self.edit_domain),('Zoom',lambda:self.digitizing.zoom(self.dock.domain.currentLayer()))])

    def dem_field(self):return self.field(self.dock.dem,[('Load DEM…',self.load_dem),
        ('Zoom',lambda:self.digitizing.zoom(self.dock.dem.currentLayer()))])

    def guide_field(self):
        return self.field(self.guides,[('Load…',lambda:self.load_vector(self.guides,'line')),
            ('Draw line',self.draw_guide),('Select',lambda:self.digitizing.select(self.guides.currentLayer())),
            ('Edit vertices',lambda:self.edit_geometry(self.guides.currentLayer(),'line')),
            ('Zoom',lambda:self.digitizing.zoom(self.guides.currentLayer()))])

    def assert_idle(self):
        self.digitizing.available()
        if self.dock.runner.is_running:raise ValueError('Stop the running worker before changing model geometry')

    def session_changed(self,active):
        self.finish_button.setEnabled(active and self.digitizing.mode=='edit');self.cancel_button.setEnabled(active)
        if hasattr(self.dock,'prepare_button'):self.dock.controls()

    def drawing_crs(self):
        layer=self.dock.dem.currentLayer()
        crs=layer.crs() if layer is not None and layer.isValid() else QgsProject.instance().crs()
        _projected_metres(crs.toWkt(),'Model geometry');return crs

    def new(self,kind,title,role,fields,crs=None):
        self.assert_idle()
        return self.digitizing.new_layer(self.dock.workspace.text(),kind,'BRAIDS '+title,role,
            crs or self.drawing_crs(),fields)

    def draw_domain(self):
        layer=self.new('polygon','calculation domain','domain',('domain_id','name'))
        identity='domain_'+uuid4().hex[:10];layer.setCustomProperty('braids/domain_id',identity)
        def valid(geometry):
            validate_geometry(geometry,'polygon')
            if len(geometry.asPolygon())!=1:raise ValueError('The domain must have no holes')
        def done(layer):
            self.dock.domain.setLayer(layer);self.dock.domain_changed(layer);self.hide_guides()
        self.digitizing.draw(layer,'polygon',dict(domain_id=identity,name=self.dock.name.text()),
            validator=valid,completed=done,snap=self.snap.isChecked())

    def edit_domain(self):
        self.assert_idle();layer=self.dock.domain.currentLayer();domain_from_layer(layer)
        self.digitizing.edit(layer,completed=lambda layer:self.dock.mesh_changed(),
            validator=domain_from_layer,snap=self.snap.isChecked())

    def draw_guide(self):
        layer=self.new('line','inlet guide (drafting only)','inlet_guides',('guide_id','name'))
        self.digitizing.draw(layer,'line',dict(guide_id=uuid4().hex,name='Inlet guide'),
            completed=lambda layer:self.guides.setLayer(layer),snap=self.snap.isChecked())

    def hide_guides(self):
        for layer in QgsProject.instance().mapLayers().values():
            if layer.customProperty('braids/input_role','')=='inlet_guides':
                node=QgsProject.instance().layerTreeRoot().findLayer(layer.id())
                if node:node.setItemVisibilityChecked(False)

    def edit_geometry(self,layer,kind):
        self.assert_idle()
        def valid(target):
            for feature in target.getFeatures():validate_geometry(feature.geometry(),kind)
        self.digitizing.edit(layer,validator=valid,completed=lambda layer:None,snap=self.snap.isChecked())

    def load_vector(self,combo,kind):
        self.assert_idle()
        path,_=QFileDialog.getOpenFileName(self.dock,'Load model '+kind,'','Vector data (*.gpkg *.shp *.geojson);;All files (*)')
        if not path:return
        layer=QgsVectorLayer(path,Path(path).stem,'ogr')
        expected=2 if kind=='polygon' else 1
        if not layer.isValid() or layer.geometryType()!=expected:raise ValueError('Choose a '+kind+' dataset')
        _projected_metres(layer.crs().toWkt(),'Model geometry')
        if combo is self.guides:layer.setCustomProperty('braids/input_role','inlet_guides')
        QgsProject.instance().addMapLayer(layer);combo.setLayer(layer)
        self.message('Layer loaded. For multi-layer GeoPackages use QGIS Data Source Manager to choose the required sublayer.')

    def load_dem(self):
        self.assert_idle()
        path,_=QFileDialog.getOpenFileName(self.dock,'Load terrain DEM','','Raster data (*.tif *.tiff *.asc *.vrt);;All files (*)')
        if not path:return
        layer=QgsRasterLayer(path,Path(path).stem)
        if not layer.isValid():raise ValueError('Choose a valid raster DEM')
        _projected_metres(layer.crs().toWkt(),'DEM');QgsProject.instance().addMapLayer(layer);self.dock.dem.setLayer(layer)

    def browse(self,edit,kind):
        if kind=='folder':path=QFileDialog.getExistingDirectory(self.dock,'Choose output folder',edit.text())
        else:path,_=QFileDialog.getOpenFileName(self.dock,'Choose '+kind,edit.text(),'All files (*)')
        if path:edit.setText(path)

    def bind_bank(self,editor):
        self.bank=editor;form=editor.layout()
        form.insertRow(0,'ID for new polygon / toe',self.bank_id)
        for kind,combo,field in (('polygon',editor.polygons,editor.polygon_id),('line',editor.lines,editor.line_id)):
            row=self.row([('Load…',lambda c=combo,k=kind:self.load_vector(c,k)),
                ('New + draw',lambda k=kind:self.draw_bank(k,True)),('Add geometry',lambda k=kind:self.draw_bank(k,False)),
                ('Select',lambda c=combo:self.digitizing.select(c.currentLayer())),
                ('Edit vertices',lambda c=combo,k=kind:self.edit_geometry(c.currentLayer(),k)),
                ('Zoom',lambda c=combo:self.digitizing.zoom(c.currentLayer()))])
            index=form.getWidgetPosition(combo)[0];form.insertRow(index+1,row)
        editor.import_button.setToolTip('Copy the current layers into the simulation input. Editing source layers does not silently alter the saved survey.')
        form.insertRow(form.getWidgetPosition(editor.import_button)[0],self.row([
            ('Create editable layers from saved survey',self.restore_survey_layers)]))

    def restore_survey_layers(self):
        self.assert_idle();doc=self.bank.draft()
        if not doc:raise ValueError('There is no saved bank survey to restore')
        crs=QgsCoordinateReferenceSystem(doc['spatial_reference'])
        for kind,combo,field,role,title in (('polygon',self.bank.polygons,self.bank.polygon_id,'bank_zones','bank zones'),
                ('line',self.bank.lines,self.bank.line_id,'toe_lines','physical toe lines')):
            layer=self.new(kind,title,role,('bank_id',),crs);features=[]
            for zone in doc['zones']:
                f=QgsFeature(layer.fields());f['bank_id']=zone['id']
                f.setGeometry(QgsGeometry.fromPolygonXY([[QgsPointXY(*p) for p in ring]
                    for ring in [zone['shell_xy'],*zone['holes_xy']]]) if kind=='polygon' else
                    QgsGeometry.fromPolylineXY([QgsPointXY(*p) for p in zone['toe_xy']]))
                features.append(f)
            if not layer.dataProvider().addFeatures(features)[0]:raise ValueError('Could not restore survey geometry')
            layer.updateExtents();layer.triggerRepaint();combo.setLayer(layer);field.setField('bank_id')
        self.bank.status.setText('Editable source copies created. Reimport explicitly after editing; saved simulation input is unchanged.')

    def draw_bank(self,kind,new=False):
        self.assert_idle();name=self.bank_id.text().strip()
        if not name:raise ValueError('Enter a bank/toe ID; matching polygons and lines must share it')
        editor=self.bank
        combo,field=(editor.polygons,editor.polygon_id) if kind=='polygon' else (editor.lines,editor.line_id)
        layer=combo.currentLayer();field_name=field.currentField()
        crs=QgsCoordinateReferenceSystem(domain_from_layer(self.dock.domain.currentLayer()).crs_wkt)
        if new or layer is None:
            layer=self.new(kind,'bank zones' if kind=='polygon' else 'physical toe lines',
                'bank_zones' if kind=='polygon' else 'toe_lines',('bank_id',),crs)
            combo.setLayer(layer);field.setField('bank_id');field_name='bank_id'
        if layer.crs()!=crs:raise ValueError('Survey layers must use the calculation-domain CRS')
        if not field_name:raise ValueError('Choose the ID field first')
        if any(str(f[field_name])==name for f in layer.getFeatures()):raise ValueError('This ID already exists; edit its geometry or choose a new ID')
        def done(layer):editor.status.setText('Geometry saved. Use selected survey layers to update the simulation snapshot.')
        self.digitizing.draw(layer,kind,{field_name:name},completed=done,snap=self.snap.isChecked())
