"""Model-input capture and explicitly owned QGIS edit sessions.

No solver or Results dependencies. Existing unrelated edit buffers are never
committed or rolled back by these tools. New input datasets are persistent GPKGs.
"""
from pathlib import Path
from uuid import uuid4

from qgis.PyQt.QtCore import QObject,QEvent,Qt,QVariant,pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.core import (Qgis,QgsProject,QgsSnappingConfig,QgsVectorLayer,QgsField,
    QgsFeature,QgsGeometry,QgsPointXY,QgsFillSymbol,QgsLineSymbol,QgsVectorFileWriter,
    QgsCoordinateReferenceSystem,QgsCoordinateTransform,QgsWkbTypes)
from qgis.gui import QgsMapTool,QgsRubberBand

from braids_core.boundaries.gis import _projected_metres


def validate_geometry(geometry,kind):
    expected=QgsWkbTypes.PolygonGeometry if kind=='polygon' else QgsWkbTypes.LineGeometry
    if (geometry.isNull() or geometry.isEmpty() or geometry.isMultipart() or
            geometry.type()!=expected or not geometry.isGeosValid()):
        raise ValueError('A valid single-part '+kind+' is required')
    if kind=='polygon' and geometry.area()<=0:raise ValueError('The polygon must have positive area')
    if kind=='line' and geometry.length()<=0:raise ValueError('The line must have positive length')


class CaptureTool(QgsMapTool):
    """Snapped left-click vertices; right-click publishes, Escape discards."""
    def __init__(self,canvas,layer,kind,completed,cancelled,message):
        super().__init__(canvas)
        self.layer,self.kind=layer,kind
        self.completed,self.cancelled,self.message=completed,cancelled,message
        self.points=[];self.closed=False
        self.band=QgsRubberBand(canvas,QgsWkbTypes.PolygonGeometry if kind=='polygon' else QgsWkbTypes.LineGeometry)
        self.band.setColor(QColor(35,105,178,110));self.band.setWidth(2)

    def point(self,event):
        match=self.canvas().snappingUtils().snapToMap(event.pos())
        return QgsPointXY(match.point() if match.isValid() else self.toMapCoordinates(event.pos()))

    def redraw(self,hover=None):
        points=self.points+([] if hover is None else [hover])
        self.band.reset(QgsWkbTypes.PolygonGeometry if self.kind=='polygon' else QgsWkbTypes.LineGeometry)
        for index,point in enumerate(points):self.band.addPoint(point,index==len(points)-1)
        self.band.show()

    def canvasReleaseEvent(self,event):
        if event.button()==Qt.RightButton:
            required=3 if self.kind=='polygon' else 2
            if len(self.points)<required:
                self.message('Add at least '+str(required)+' vertices, or press Escape to cancel.');return
            try:
                transform=QgsCoordinateTransform(self.canvas().mapSettings().destinationCrs(),self.layer.crs(),QgsProject.instance())
                points=[transform.transform(p) for p in self.points]
                geometry=(QgsGeometry.fromPolygonXY([points+[points[0]]]) if self.kind=='polygon'
                          else QgsGeometry.fromPolylineXY(points))
                validate_geometry(geometry,self.kind)
                self.completed(geometry)
            except Exception as exc:self.message(str(exc)+' Use Backspace to correct vertices.')
        elif event.button()==Qt.LeftButton:
            point=self.point(event)
            if not self.points or point!=self.points[-1]:self.points.append(point)
            self.redraw()

    def canvasMoveEvent(self,event):self.redraw(self.point(event))

    def keyPressEvent(self,event):
        if event.key()==Qt.Key_Escape:self.cancelled()
        elif event.key() in (Qt.Key_Backspace,Qt.Key_Delete):
            if self.points:self.points.pop()
            self.redraw()

    def deactivate(self):
        super().deactivate()
        if not self.closed:self.cancelled(False)

    def close(self):
        self.closed=True
        self.canvas().scene().removeItem(self.band)


class RiverDigitizing(QObject):
    changed=pyqtSignal(bool)

    def __init__(self,iface,parent,message):
        super().__init__(parent);self.iface=iface;self.message=message
        self.layer=self.tool=None;self.mode=None;self._finishing=False
        self._original_snapping=self._temporary_snapping=None
        QgsProject.instance().layersWillBeRemoved.connect(self.layers_removed)

    def layers_removed(self,ids):
        if self.active and self.layer is not None and self.layer.id() in ids:self.cancel()

    @property
    def active(self):return self.mode is not None

    def canvas(self):
        if self.iface is None:raise ValueError('Open BRAIDS inside QGIS to use the map tools')
        return self.iface.mapCanvas()

    def available(self):
        self.canvas()
        if self.active:raise ValueError('Finish or cancel the current drawing/edit session first')

    def check_layer(self,layer):
        self.canvas()
        if layer is None or not layer.isValid() or QgsProject.instance().mapLayer(layer.id()) is None:
            raise ValueError('Choose a loaded, valid layer first')

    def select(self,layer):
        self.available();self.check_layer(layer)
        self.iface.setActiveLayer(layer);self.iface.actionSelect().trigger()
        self.message('Select features on the map. Hold Shift to add or remove selections.')

    def zoom(self,layer):
        self.check_layer(layer)
        extent=layer.boundingBoxOfSelected() if isinstance(layer,QgsVectorLayer) and layer.selectedFeatureCount() else layer.extent()
        if isinstance(layer,QgsVectorLayer) and layer.featureCount()==0:raise ValueError('The layer has no extent to zoom to')
        if extent.isEmpty():extent.grow(max(extent.width(),extent.height(),1.)*.05)
        transform=QgsCoordinateTransform(layer.crs(),self.canvas().mapSettings().destinationCrs(),QgsProject.instance())
        extent=transform.transformBoundingBox(extent);extent.scale(1.1)
        self.canvas().setExtent(extent);self.canvas().refresh()

    def new_layer(self,folder,kind,title,role,crs,fields):
        self.available();_projected_metres(crs.toWkt(),'Model geometry')
        folder=Path(folder)/'model_layers';folder.mkdir(parents=True,exist_ok=True)
        path=folder/(role+'_'+uuid4().hex[:10]+'.gpkg')
        memory=QgsVectorLayer('Polygon' if kind=='polygon' else 'LineString',title,'memory');memory.setCrs(crs)
        memory.dataProvider().addAttributes([QgsField(name,QVariant.String) for name in fields]);memory.updateFields()
        options=QgsVectorFileWriter.SaveVectorOptions();options.driverName='GPKG';options.layerName=role
        result=QgsVectorFileWriter.writeAsVectorFormatV3(memory,str(path),QgsProject.instance().transformContext(),options)
        if result[0]!=QgsVectorFileWriter.NoError:raise ValueError('Could not create input GeoPackage: '+str(result))
        layer=QgsVectorLayer(str(path)+'|layername='+role,title,'ogr')
        if not layer.isValid():raise ValueError('Could not load the new input layer')
        layer.setCustomProperty('braids/input_role',role)
        if kind=='polygon':
            symbol=QgsFillSymbol.createSimple({'color':'46,134,193,35','outline_color':'#2369b2','outline_width':'.6'})
        else:symbol=QgsLineSymbol.createSimple({'line_color':'#7546aa','line_width':'.8','line_style':'dash' if role=='inlet_guides' else 'solid'})
        layer.renderer().setSymbol(symbol);QgsProject.instance().addMapLayer(layer)
        return layer

    def snapping(self,enabled):
        project=QgsProject.instance();self._original_snapping=QgsSnappingConfig(project.snappingConfig())
        config=QgsSnappingConfig(project.snappingConfig());config.setEnabled(enabled)
        config.setMode(QgsSnappingConfig.AllLayers);config.setTypeFlag(Qgis.SnappingType.Vertex|Qgis.SnappingType.Segment)
        config.setTolerance(12.);config.setUnits(Qgis.MapToolUnit.Pixels);config.setIntersectionSnapping(True)
        self._temporary_snapping=QgsSnappingConfig(config);project.setSnappingConfig(config)

    def draw(self,layer,kind,attributes,*,completed,validator=None,snap=True):
        self.available();self.check_layer(layer)
        if layer.customProperty('braids/input_role','')=='boundary_faces':
            raise ValueError('Edit the domain and recreate generated boundary faces; do not draw into this layer')
        if layer.isEditable():raise ValueError('Finish this layer’s existing QGIS edit session before drawing with BRAIDS')
        if any(layer.fields().indexFromName(k)<0 for k in attributes):raise ValueError('Choose a layer with the required ID field')
        self.layer,self.mode=layer,'draw';self.validator=validator;self.completed=completed
        def publish(geometry):
            if self.validator:self.validator(geometry)
            if layer.isEditable():raise ValueError('Another edit session started on this layer; cancel drawing first')
            if not layer.startEditing():raise ValueError('This layer cannot be edited')
            try:
                feature=QgsFeature(layer.fields());feature.setGeometry(geometry)
                for key,value in attributes.items():feature[key]=value
                if not layer.addFeature(feature):raise ValueError('Could not add the geometry')
                if not layer.commitChanges():raise ValueError('Could not save the geometry: '+'; '.join(layer.commitErrors()))
            except BaseException:
                if layer.isEditable():layer.rollBack()
                raise
            layer.updateExtents();layer.triggerRepaint()
            self.finish_session()
            completed(layer)
            self.message('Geometry saved. Drawing has ended; use Edit vertices to change it.')
        try:
            self.snapping(snap);self.iface.setActiveLayer(layer)
            self.tool=CaptureTool(self.canvas(),layer,kind,publish,self.cancel,self.message)
            self.canvas().setMapTool(self.tool);self.changed.emit(True)
        except Exception:
            self.cancel();raise
        self.message('Click vertices; right-click to save and finish. Backspace removes a vertex; Escape cancels.')

    def edit(self,layer,*,completed,validator=None,snap=True):
        self.available();self.check_layer(layer)
        if layer.customProperty('braids/input_role','')=='boundary_faces':
            raise ValueError('Edit the domain and recreate generated boundary faces; do not edit this layer independently')
        if layer.isEditable():raise ValueError('Finish this layer’s existing QGIS edit session first; BRAIDS will not commit unrelated edits')
        action=getattr(self.iface,'actionVertexToolActiveLayer',None) or getattr(self.iface,'actionVertexTool',None)
        if action is None:raise ValueError('The QGIS vertex tool is unavailable')
        if not layer.startEditing():raise ValueError('This layer cannot be edited')
        self.layer,self.mode,self.validator,self.completed=layer,'edit',validator,completed
        try:
            self.snapping(snap);self.iface.setActiveLayer(layer)
            self.canvas().viewport().installEventFilter(self);self.canvas().installEventFilter(self)
            action().trigger();self.changed.emit(True)
        except Exception:
            self.cancel();raise
        self.message('Edit vertices. Right-click or Finish edits to save and exit; Escape or Cancel discards this edit session.')

    def finish_edit(self):
        if self.mode!='edit':raise ValueError('Finish drawing with a right-click on the map')
        layer,completed=self.layer,self.completed
        if self.validator:self.validator(layer)
        if not layer.commitChanges():raise ValueError('Could not save edits: '+'; '.join(layer.commitErrors()))
        layer.updateExtents();layer.triggerRepaint();self.finish_session();completed(layer)
        self.message('Edits saved. Vertex editing is off.')

    def eventFilter(self,obj,event):
        if self.mode=='edit':
            if event.type()==QEvent.MouseButtonPress and event.button()==Qt.RightButton:
                try:self.finish_edit()
                except Exception as exc:self.message(str(exc)+' Correct the geometry or Cancel.')
                return True
            if event.type()==QEvent.KeyPress and event.key()==Qt.Key_Escape:
                self.cancel();return True
        return False

    def finish_session(self,pan=True):
        if self._finishing:return
        self._finishing=True
        try:
            canvas=self.canvas()
            if self.tool is not None:
                self.tool.close()
                if canvas.mapTool() is self.tool:canvas.unsetMapTool(self.tool)
                self.tool.deleteLater();self.tool=None
            canvas.viewport().removeEventFilter(self);canvas.removeEventFilter(self)
            self.layer=None;self.mode=None
            if self._original_snapping is not None:
                project=QgsProject.instance()
                if project.snappingConfig()==self._temporary_snapping:project.setSnappingConfig(self._original_snapping)
            self._original_snapping=self._temporary_snapping=None
            if pan:self.iface.actionPan().trigger()
            self.changed.emit(False)
        finally:self._finishing=False

    def cancel(self,pan=True):
        if not self.active or self._finishing:return
        if self.mode=='edit' and self.layer.isEditable():self.layer.rollBack()
        self.finish_session(pan);self.message('Drawing/edit cancelled; previously saved geometry is unchanged.')
