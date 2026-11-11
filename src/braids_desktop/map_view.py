"""BRAIDS editing tools over the retained AVAC map workspace."""
from copy import deepcopy
import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF,QPointF,QSize
from PySide6.QtGui import QColor, QPainterPath, QPen, QLinearGradient,QIcon,QPixmap,QPainter
from PySide6.QtWidgets import QToolButton,QGraphicsView,QGraphicsEllipseItem,QGraphicsItem
from ._avac.map_view import MapView as AVACMapView
from ._avac.preferences import geometry_color,mesh_color,snap_enabled
from .map_tools import TemporaryRuler,TemporaryBedProfile,tool_icon


class MeasurementHandle(QGraphicsEllipseItem):
    def __init__(self,view,index,point):
        super().__init__(-5,-5,10,10);self.view=view;self.index=index
        self.setPen(QPen(QColor('#152638')));self.setBrush(QColor('#ff8030'))
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations);self.setZValue(40)
        self.setPos(view.to_scene(*point));self.setFlags(self.flags()|QGraphicsItem.ItemIsMovable|QGraphicsItem.ItemSendsGeometryChanges)

    def itemChange(self,change,value):
        if change==QGraphicsItem.ItemPositionHasChanged:self.view._move_measurement_vertex(self.index,value)
        return super().itemChange(change,value)

    def mouseReleaseEvent(self,event):
        super().mouseReleaseEvent(event);self.view._commit_measurement_edit()


class BreaklineHandle(MeasurementHandle):
    def itemChange(self,change,value):
        if change==QGraphicsItem.ItemPositionHasChanged:self.view._move_breakline_vertex(self.index,value)
        return QGraphicsEllipseItem.itemChange(self,change,value)

    def mouseReleaseEvent(self,event):
        QGraphicsEllipseItem.mouseReleaseEvent(self,event)
        if self.view._breakline_edit is not None:
            name,points=self.view._breakline_edit;self.view.breakline_edited.emit(name,points.copy())
from .presentation import display_number


class SolidHandle(MeasurementHandle):
    def itemChange(self,change,value):
        if change==QGraphicsItem.ItemPositionHasChanged:self.view._move_solid_vertex(self.index,value)
        return QGraphicsEllipseItem.itemChange(self,change,value)

    def mouseReleaseEvent(self,event):
        QGraphicsEllipseItem.mouseReleaseEvent(self,event)
        if self.view._solid_edit is not None:
            name,points=self.view._solid_edit;self.view.solid_edited.emit(name,points.copy())


class ZoneHandle(MeasurementHandle):
    def itemChange(self,change,value):
        if change==QGraphicsItem.ItemPositionHasChanged:self.view._move_zone_vertex(self.index,value)
        return QGraphicsEllipseItem.itemChange(self,change,value)
    def mouseReleaseEvent(self,event):
        QGraphicsEllipseItem.mouseReleaseEvent(self,event)
        if self.view._zone_edit is not None:
            kind,name,points=self.view._zone_edit;self.view.zone_edited.emit(kind,name,points.copy())


class RiverMapView(AVACMapView):
    edge_selection_changed = Signal(object)
    captured_geometry = Signal(str, object)
    boundary_edit_requested = Signal(str)
    zoom_domain_requested = Signal()
    measurement_selected=Signal(str,str)
    measurement_edited=Signal(str,str,object)
    breakline_edited=Signal(str,object)
    solid_edited=Signal(str,object)
    dem_dropped=Signal(str)
    zone_edited=Signal(str,str,object)
    inspect_result=Signal(object)
    bed_profile_drawn=Signal(object)
    feature_delete_requested=Signal(str,str)

    def __init__(self, parent=None):
        self.boundary_items = []
        self.selected_edges = set()
        self.assignments = []
        self.domain_edges = []
        self.boundary_reference_rings=[]
        self.result_metadata = None
        self.capture_mode = None
        self.turbine_items=[];self.turbines=None;self.turbine_preview_diameter=.7
        self.snap_enabled = snap_enabled()
        self._measurement_handles=[];self._measurement_edit=None;self._measurement_features=[]
        self._draft_cursor=None
        self.preview_mesh_item=None
        self.ruler=TemporaryRuler();self.bed_profile=TemporaryBedProfile();self._exporting_map=False
        self.breakline_items=[];self.breaklines=[];self._breakline_handles=[];self._breakline_edit=None
        self.solid_items=[];self.solids=[];self._solid_handles=[];self._solid_edit=None
        self.zone_items={'roughness':[],'erodible':[]};self.zones={'roughness':[],'erodible':[]};self._zone_handles=[];self._zone_edit=None
        super().__init__(parent)
        self._selected_breakline=None
        self.tool_buttons['delete'].setCheckable(True)
        self.tool_buttons['delete'].setToolTip('Delete a feature: click its line, point or polygon. Right-click cancels.')
        self.setAcceptDrops(True)
        self._middle_pan=None
        self.tool_buttons['polygon'].setToolTip('Draw domain · click vertices; double-click or Enter to finish')
        self.tool_buttons['line'].setToolTip('Draw a guide line · double-click or Enter to finish')
        fit=QToolButton(self.navigation_controls);fit.setObjectName('MapControl');fit.setText('⌖')
        fit.setFixedSize(36,36);fit.setToolTip('Zoom to domain');fit.clicked.connect(self.zoom_domain_requested.emit)
        self.navigation_controls.layout().insertWidget(0,fit);self.tool_buttons['fit_domain']=fit
        ruler=QToolButton(self.navigation_controls);ruler.setObjectName('MapControl');ruler.setFixedSize(36,36)
        ruler.setCheckable(True);ruler.setToolTip('Measure distance and terrain slope · positive slope means elevation falls from the first point to the last; right-click clears')
        ruler.setAccessibleName('Measure distance and terrain slope');ruler.clicked.connect(lambda:self.set_tool('ruler'))
        navigation=self.navigation_controls.layout();navigation.insertWidget(navigation.indexOf(self.tool_buttons['pan'])+1,ruler)
        self.tool_buttons['ruler']=ruler
        bed_profile=QToolButton(self.navigation_controls);bed_profile.setObjectName('MapControl');bed_profile.setFixedSize(36,36)
        # A native Unicode mountain is reliable in the Windows UI font and
        # communicates terrain/elevation more directly than the prior glyph.
        bed_profile.setCheckable(True);bed_profile.setToolTip('Draw bed profile · click points, right-click to plot');bed_profile.setAccessibleName('Draw bed profile')
        bed_profile.clicked.connect(lambda:self.set_tool('bed_profile'))
        navigation.insertWidget(navigation.indexOf(ruler)+1,bed_profile);self.tool_buttons['bed_profile']=bed_profile
        inspect=QToolButton(self.navigation_controls);inspect.setObjectName('MapControl');inspect.setText('i');inspect.setCheckable(True)
        inspect.setFixedSize(36,36);inspect.setToolTip('Identify result value at a cell');inspect.clicked.connect(lambda:self.set_tool('identify'))
        navigation.addWidget(inspect);self.tool_buttons['identify']=inspect
        for key in ('pan','ruler','bed_profile'):
            button=self.tool_buttons[key];button.setText('');button.setIcon(tool_icon(key));button.setIconSize(QSize(24,24))
        for key in ('fit_domain','zoom_in','zoom_out'):
            pix=QPixmap(48,48);pix.fill(Qt.transparent);p=QPainter(pix);p.setRenderHint(QPainter.Antialiasing)
            p.scale(2,2);p.setPen(QPen(QColor('#ffffff'),1.8));p.drawEllipse(QRectF(3,3,12,12));p.drawLine(QPointF(13,13),QPointF(20,20))
            if key!='fit_domain':
                p.drawLine(QPointF(6,9),QPointF(12,9))
                if key=='zoom_in':p.drawLine(QPointF(9,6),QPointF(9,12))
            p.end();button=self.tool_buttons[key];button.setText('');button.setIcon(QIcon(pix));button.setIconSize(QSize(24,24))

    def clear(self):
        self.boundary_reference_rings=[]
        self.turbine_items=[];self.turbines=None
        self.zone_items={'roughness':[],'erodible':[]};self.zones={'roughness':[],'erodible':[]};self._zone_handles=[];self._zone_edit=None
        self.solid_items=[];self.solids=[];self._solid_handles=[];self._solid_edit=None
        self.breakline_items=[];self.breaklines=[];self._breakline_handles=[];self._breakline_edit=None
        self._measurement_handles=[];self._measurement_edit=None;self._measurement_features=[]
        self.preview_mesh_item=None
        self.boundary_items = []
        self.domain_edges = []
        self.assignments = []
        self.selected_edges = set()
        self.result_metadata = None
        self.time_text = ''
        super().clear()

    def set_result_metadata(self, payload):
        self.result_metadata = {key: payload[key] for key in ('label', 'limits', 'palette', 'time_s','variable')}
        self.time_text = f"t = {display_number(payload['time_s'])} s"
        self.viewport().update()

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if self.tool=='ruler' and not self._exporting_map:self.ruler.draw(painter,self)
        if self.tool=='bed_profile' and not self._exporting_map:self.bed_profile.draw(painter,self)
        if not self.result_metadata or not self.terrain or not getattr(self,'result_visible',True) or not self.decorations.get('legends',True):return
        if not self.styles.get('result:'+self.result_metadata['variable'],{}).get('legend_visible',True):return
        painter.save(); painter.resetTransform()
        from matplotlib import colormaps
        label = self.result_metadata['label']
        low, high = self.result_metadata['limits']
        metrics = painter.fontMetrics()
        width = max(210, metrics.horizontalAdvance(label)+20)
        top = self.viewport().height()-156
        panel = QRectF(14, top, width, 64)
        painter.setPen(Qt.NoPen);painter.setBrush(QColor('#41637e'));painter.drawRoundedRect(panel,5,5)
        painter.setPen(QColor('#ffffff'));painter.drawText(23,top+20,label)
        ramp=QLinearGradient(23,0,23+width-18,0)
        cmap=colormaps.get_cmap(self.result_metadata['palette'])
        for i in range(17):
            color=cmap(i/16);ramp.setColorAt(i/16,QColor.fromRgbF(*color))
        painter.fillRect(QRectF(23,top+28,width-18,8),ramp)
        painter.drawText(23,top+53,display_number(low))
        text=display_number(high);painter.drawText(int(14+width-9-metrics.horizontalAdvance(text)),top+53,text)
        painter.restore()

    def appearance_layers(self):
        layers=super().appearance_layers();panel=getattr(self,'results_panel',None)
        if panel is not None and panel.store is not None:
            for i in range(panel.variable.count()):layers.append(('Results — '+panel.variable.itemText(i),'result:'+panel.variable.itemData(i)))
        return layers

    def visible_range_for(self,key):
        if self.terrain is None:raise ValueError('Load terrain first.')
        corners=self.mapToScene(self.viewport().rect()).boundingRect()
        x0,y0=self.from_scene(corners.topLeft());x1,y1=self.from_scene(corners.bottomRight())
        xmin,xmax=sorted((x0,x1));ymin,ymax=sorted((y0,y1))
        if key.startswith('result:'):
            panel=self.results_panel
            if panel.store is None:raise ValueError('Load a result frame first.')
            frame=panel.store.read(panel.slider.value());values=panel.store.display_values(frame,key[7:])
            from shapely.geometry import box
            region=box(xmin,ymin,xmax,ymax)
            selected=np.array([poly.intersects(region) for poly in frame.polygons],dtype=bool)
            data=np.asarray(values)[selected]
        else:
            raster=self.terrain.raster if key=='terrain' else self.additional_fields.get(key,self.field_raster)
            if raster is None:raise ValueError('This layer has no visible data.')
            xs=(raster.x>=xmin)&(raster.x<=xmax);ys=(raster.y>=ymin)&(raster.y<=ymax)
            data=np.asarray(raster.z)[np.ix_(ys,xs)]
        data=np.asarray(data);data=data[np.isfinite(data)]
        if not data.size:raise ValueError('No finite values are visible for this layer.')
        low,high=float(data.min()),float(data.max())
        if high<=low:raise ValueError('Visible values are constant; enter a nonzero range manually.')
        return low,high

    def default_palette(self,key):
        if key.startswith('result:'):
            from .results import variable_info
            return variable_info(key[7:])[2]
        return super().default_palette(key)

    def appearance_palettes(self,key):
        if key.startswith('result:'):
            return list(dict.fromkeys([self.default_palette(key),'Blues','Reds','viridis','plasma','cividis','turbo','coolwarm','RdBu_r','BrBG','Purples','YlOrBr','gray']))
        return super().appearance_palettes(key)

    def range_for(self,key):
        if key.startswith('result:'):
            from .results import colour_values
            panel=self.results_panel;frame=panel.store.read(panel.slider.value());variable=key[7:]
            return colour_values(panel.store.display_values(frame,variable),variable)[1]
        return super().range_for(key)

    def refresh_appearance(self,key):
        if key.startswith('result:'):self.results_panel.show_frame()
        else:super().refresh_appearance(key)

    def set_release(self, rings):
        if rings:self.boundary_reference_rings=deepcopy(rings)
        super().set_release(rings)
        if self.release_item is not None:
            self.release_item.setData(10,'domain');self.release_item.setData(11,'domain')
        self.release_item.setBrush(Qt.NoBrush)
        pen = QPen(QColor(geometry_color()), 2)
        pen.setCosmetic(True)
        self.release_item.setPen(pen)
        self.release_item.setZValue(29)

    def set_boundaries(self, assignments, selected=None):
        self.assignments = deepcopy(assignments)
        if selected is not None:
            self.selected_edges = set(selected)
        for item in self.boundary_items:
            if item.scene() is self.scene():
                self.scene().removeItem(item)
        self.boundary_items = []
        rings=self.rings or self.boundary_reference_rings
        if not rings:
            return
        vertices = rings[0][0]
        if np.array_equal(vertices[0], vertices[-1]):
            vertices = vertices[:-1]
        for index, start in enumerate(vertices):
            end = vertices[(index+1) % len(vertices)]
            edge_id = self.domain_edges[index].edge_id if index < len(self.domain_edges) else str(index)
            edge_rows=[a for a in assignments if edge_id==a.get('edge_id')]
            if index in self.selected_edges:edge_rows.append(dict(bc_id='Selected',kind='selected'))
            length=float(np.linalg.norm(end-start))
            for assignment in edge_rows:
                color='#ffbc30' if assignment['kind']=='selected' else geometry_color()
                begin=assignment.get('start_m',0.)/length
                finish=(length if assignment.get('end_m') is None else assignment['end_m'])/length
                first=start+begin*(end-start);last=start+finish*(end-start)
                path=QPainterPath(self.to_scene(*first));path.lineTo(self.to_scene(*last))
                pen=QPen(QColor(color),4);pen.setCosmetic(True)
                item=self.scene().addPath(path,pen);item.setZValue(30)
                item.setToolTip(f"Edge {index+1}: {assignment['bc_id']}")
                item.setData(0,assignment['bc_id'] if assignment['kind']!='selected' else None)
                self.boundary_items.append(item)

    def begin_capture(self, kind):
        self.capture_mode=kind
        self.set_tool('ellipse' if kind=='solid_ellipse' else 'polygon' if kind in ('bank_polygon','solid_polygon') or kind.startswith('zone:') else 'line')

    def set_tool(self, tool):
        self.ruler.clear()
        self.bed_profile.clear()
        if tool!='measurement_edit':self.end_measurement_edit()
        if tool!='breakline_edit':self.end_breakline_edit()
        if tool!='solid_edit':self.end_solid_edit()
        if tool!='zone_edit':self.end_zone_edit()
        self._draft_cursor=None
        if tool=='pan':self.capture_mode=None
        super().set_tool(tool)
        self.tool_buttons['delete'].setChecked(tool=='delete')
        if 'ruler' in self.tool_buttons:self.tool_buttons['ruler'].setChecked(tool=='ruler')
        if 'bed_profile' in self.tool_buttons:self.tool_buttons['bed_profile'].setChecked(tool=='bed_profile')
        if 'identify' in self.tool_buttons:self.tool_buttons['identify'].setChecked(tool=='identify')
        self.viewport().update()

    def delete_selection(self):
        self.set_tool('delete')

    def delete_feature_at(self,position):
        """Hit-test editable GIS paths in viewport pixels, independent of zoom."""
        from PySide6.QtGui import QPainterPathStroker
        stroke=QPainterPathStroker();stroke.setWidth(16)
        point=QPointF(position);interiors=[]
        for item in self.scene().items():
            kind=item.data(10)
            if not kind or not item.isVisible() or not hasattr(item,'path'):continue
            path=item.deviceTransform(self.viewportTransform()).map(item.path())
            if stroke.createStroke(path).contains(point):
                hit=(kind,str(item.data(11)));self.feature_delete_requested.emit(*hit);return hit
            if path.contains(point):interiors.append((kind,str(item.data(11))))
        if self.sketch_item is not None and self.sketch_item.isVisible():
            for index,line in enumerate(self.sketch_lines):
                path=QPainterPath(self.mapFromScene(self.to_scene(*line[0])))
                for vertex in line[1:]:path.lineTo(self.mapFromScene(self.to_scene(*vertex)))
                if stroke.createStroke(path).contains(point):
                    hit=('guide',str(index));self.feature_delete_requested.emit(*hit);return hit
        if interiors:
            hit=next((p for p in interiors if p[0]!='domain'),interiors[0])
            self.feature_delete_requested.emit(*hit);return hit
        return None

    def set_breaklines(self,rows):
        from shiboken6 import isValid
        for item in self.breakline_items:
            if isValid(item) and item.scene() is not None:self.scene().removeItem(item)
        self.breaklines=deepcopy(rows);self.breakline_items=[]
        for row in rows:
            points=row['points_xy'];path=QPainterPath(self.to_scene(*points[0]))
            for p in points[1:]:path.lineTo(self.to_scene(*p))
            if row['closed']:path.closeSubpath()
            pen=QPen(QColor('#c43fad'),2.5);pen.setCosmetic(True)
            item=self.scene().addPath(path,pen);item.setZValue(31);item.setToolTip(row['id'])
            item.setData(10,'breakline');item.setData(11,row['id'])
            self.breakline_items.append(item)
        self.highlight_breakline(getattr(self,'_selected_breakline',None))

    def highlight_breakline(self,name):
        self._selected_breakline=name
        for item in self.breakline_items:
            selected=item.data(11)==name
            pen=QPen(QColor('#ff8030' if selected else '#c43fad'),4. if selected else 2.5);pen.setCosmetic(True)
            item.setPen(pen);item.setZValue(34 if selected else 31)

    def begin_breakline_edit(self,row):
        self.set_tool('breakline_edit');points=np.asarray(row['points_xy'],float)
        if row['closed'] and len(points)>1 and np.array_equal(points[0],points[-1]):points=points[:-1]
        self._breakline_edit=(row['id'],points.copy())
        for i,point in enumerate(points):
            handle=BreaklineHandle(self,i,point);self.scene().addItem(handle);self._breakline_handles.append(handle)

    def end_breakline_edit(self):
        from shiboken6 import isValid
        for handle in self._breakline_handles:
            if isValid(handle) and handle.scene() is not None:self.scene().removeItem(handle)
        self._breakline_handles=[];self._breakline_edit=None

    def _move_breakline_vertex(self,index,position):
        if self._breakline_edit is None:return
        name,points=self._breakline_edit;points[index]=AVACMapView.from_scene(self,position)
        rows=deepcopy(self.breaklines)
        for row in rows:
            if row['id']==name:row['points_xy']=points.tolist()
        self.set_breaklines(rows)

    def export_image(self):
        self._exporting_map=True
        try:return super().export_image()
        finally:self._exporting_map=False

    def set_solids(self,rows):
        from shiboken6 import isValid
        for item in self.solid_items:
            if isValid(item) and item.scene() is not None:self.scene().removeItem(item)
        self.solids=deepcopy(rows);self.solid_items=[]
        for row in rows:
            points=row['points_xy'];path=QPainterPath(self.to_scene(*points[0]))
            for p in points[1:]:path.lineTo(self.to_scene(*p))
            path.closeSubpath();pen=QPen(QColor('#515964'),2);pen.setCosmetic(True)
            item=self.scene().addPath(path,pen,QColor('#a4aab2'));item.setZValue(31);item.setToolTip(row['id']+' — impermeable solid')
            item.setData(10,'solid');item.setData(11,row['id'])
            self.solid_items.append(item)

    def begin_solid_edit(self,row):
        self.set_tool('solid_edit');points=np.asarray(row['points_xy'],float)
        if np.array_equal(points[0],points[-1]):points=points[:-1]
        self._solid_edit=(row['id'],points.copy())
        for i,point in enumerate(points):
            handle=SolidHandle(self,i,point);self.scene().addItem(handle);self._solid_handles.append(handle)

    def end_solid_edit(self):
        from shiboken6 import isValid
        for handle in self._solid_handles:
            if isValid(handle) and handle.scene() is not None:self.scene().removeItem(handle)
        self._solid_handles=[];self._solid_edit=None

    def _move_solid_vertex(self,index,position):
        if self._solid_edit is None:return
        name,points=self._solid_edit;points[index]=AVACMapView.from_scene(self,position)
        rows=deepcopy(self.solids)
        for row in rows:
            if row['id']==name:row['points_xy']=points.tolist()
        self.set_solids(rows)

    def set_analysis_features(self,features):
        self._measurement_features=deepcopy(features);super().set_analysis_features(features)
        from ._avac.preferences import measurement_color
        for item,feature in zip(self.analysis_items,features):
            pen=QPen(QColor('#ff8030' if feature.get('selected') else measurement_color()),3.5);pen.setCosmetic(True)
            item.setPen(pen);item.setZValue(32);item.setData(1,feature.get('id'))
            item.setData(10,feature['kind']);item.setData(11,feature.get('id'))

    def set_zones(self,zones):
        from shiboken6 import isValid
        for items in self.zone_items.values():
            for item in items:
                if isValid(item) and item.scene() is not None:self.scene().removeItem(item)
        self.zones=deepcopy(zones);self.zone_items={kind:[] for kind in zones}
        for kind,rows in zones.items():
            color=QColor('#b27a24' if kind=='roughness' else '#9656ba');fill=QColor(color);fill.setAlpha(45)
            for row in rows:
                points=row['points_xy'];path=QPainterPath(self.to_scene(*points[0]))
                for p in points[1:]:path.lineTo(self.to_scene(*p))
                path.closeSubpath();pen=QPen(color,2);pen.setCosmetic(True)
                item=self.scene().addPath(path,pen,fill);item.setZValue(29);item.setToolTip(f"{row['id']}: {row['value']}")
                item.setData(10,'zone:'+kind);item.setData(11,row['id'])
                self.zone_items[kind].append(item)

    def begin_zone_edit(self,kind,row):
        self.set_tool('zone_edit');points=np.asarray(row['points_xy'],float)
        if np.array_equal(points[0],points[-1]):points=points[:-1]
        self._zone_edit=(kind,row['id'],points.copy())
        for i,point in enumerate(points):
            handle=ZoneHandle(self,i,point);self.scene().addItem(handle);self._zone_handles.append(handle)

    def end_zone_edit(self):
        from shiboken6 import isValid
        for handle in self._zone_handles:
            if isValid(handle) and handle.scene() is not None:self.scene().removeItem(handle)
        self._zone_handles=[];self._zone_edit=None

    def _move_zone_vertex(self,index,position):
        if self._zone_edit is None:return
        kind,name,points=self._zone_edit;points[index]=AVACMapView.from_scene(self,position);zones=deepcopy(self.zones)
        for row in zones[kind]:
            if row['id']==name:row['points_xy']=points.tolist()
        self.set_zones(zones)

    def begin_measurement_edit(self,kind,identifier,points):
        self.set_tool('measurement_edit');self._measurement_edit=(kind,identifier,np.asarray(points,float).copy())
        for i,point in enumerate(points):
            handle=MeasurementHandle(self,i,point);self.scene().addItem(handle);self._measurement_handles.append(handle)

    def end_measurement_edit(self):
        from shiboken6 import isValid
        for handle in self._measurement_handles:
            if isValid(handle) and handle.scene() is not None:self.scene().removeItem(handle)
        self._measurement_handles=[];self._measurement_edit=None

    def _move_measurement_vertex(self,index,position):
        if self._measurement_edit is None:return
        kind,identifier,points=self._measurement_edit
        points[index]=AVACMapView.from_scene(self,position)
        features=deepcopy(self._measurement_features)
        for feature in features:
            if feature.get('id')==identifier and feature['kind']==kind:feature['coords']=points.tolist()
        self.set_analysis_features(features)

    def _commit_measurement_edit(self):
        if self._measurement_edit is not None:
            kind,identifier,points=self._measurement_edit;self.measurement_edited.emit(kind,identifier,points.copy())

    def set_model_tools(self,enabled):
        super().set_model_tools(enabled)
        self.tool_buttons['delete'].show()
        if not enabled and self.tool in ('solid_edit','breakline_edit','zone_edit','ellipse'):self.set_tool('pan')
        self._position_controls()

    def _position_controls(self):
        super()._position_controls()
        # Keep the trash control accessible for result probes/profiles even
        # when all model drawing controls are hidden by the shell.
        self.map_controls.show()

    def zoom(self,factor):
        center=self.mapToScene(self.viewport().rect().center());anchor=self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        try:super().zoom(factor);self.centerOn(center)
        finally:self.setTransformationAnchor(anchor)
        self._save_extent()

    def wheelEvent(self,event):
        AVACMapView.zoom(self,1.2 if event.angleDelta().y()>0 else 1/1.2)
        if self.tool in ('ruler','bed_profile') and self.terrain:
            (self.ruler if self.tool=='ruler' else self.bed_profile).cursor=AVACMapView.from_scene(self,self.mapToScene(event.position().toPoint()));self.viewport().update()
        event.accept()

    def show_preview_mesh(self,polygons):
        if self.preview_mesh_item is not None:self.scene().removeItem(self.preview_mesh_item)
        path=QPainterPath()
        for vertices in polygons:
            path.moveTo(self.to_scene(*vertices[0]))
            for point in vertices[1:]:path.lineTo(self.to_scene(*point))
            path.closeSubpath()
        pen=QPen(QColor(mesh_color()),.8);pen.setCosmetic(True)
        self.preview_mesh_item=self.scene().addPath(path,pen);self.preview_mesh_item.setZValue(28)

    def refresh_mesh_color(self):
        if self.preview_mesh_item is not None:
            pen=self.preview_mesh_item.pen();pen.setColor(QColor(mesh_color()));self.preview_mesh_item.setPen(pen)

    def set_sketch_lines(self,lines):
        super().set_sketch_lines(lines)
        if self.sketch_item:
            pen=QPen(QColor(geometry_color()),2);pen.setCosmetic(True);self.sketch_item.setPen(pen)

    def refresh_geometry_color(self):
        self.set_release(self.rings);self.set_sketch_lines(self.sketch_lines);self.set_boundaries(self.assignments)
        if self.points:self._draw_draft()

    def _draw_draft(self):
        if self.line_item is not None:self.scene().removeItem(self.line_item);self.line_item=None
        if not self.points:return
        path=QPainterPath(self.to_scene(*self.points[0]))
        if self.capture_mode=='turbine' and self._draft_cursor is not None:
            c=np.array(self.points[0]);d=np.array(self._draft_cursor)-c
            if np.linalg.norm(d)>1e-10:
                tangent=np.array([-d[1],d[0]])/np.linalg.norm(d)*self.turbine_preview_diameter/2
                path.moveTo(self.to_scene(*(c-tangent)));path.lineTo(self.to_scene(*(c+tangent)));path.moveTo(self.to_scene(*c))
        if self.tool=='ellipse' and self._draft_cursor is not None:
            path=QPainterPath();path.addEllipse(QRectF(self.to_scene(*self.points[0]),self.to_scene(*self._draft_cursor)).normalized())
        else:
            for point in self.points[1:]:path.lineTo(self.to_scene(*point))
            if self._draft_cursor is not None:path.lineTo(self.to_scene(*self._draft_cursor))
        if self.tool=='polygon' and len(self.points)>=2:path.closeSubpath()
        pen=QPen(QColor(geometry_color()),2);pen.setCosmetic(True)
        self.line_item=self.scene().addPath(path,pen);self.line_item.setBrush(Qt.NoBrush);self.line_item.setZValue(31)

    def mouseMoveEvent(self,event):
        if self._middle_pan is not None:
            pos=event.position().toPoint();delta=pos-self._middle_pan
            self._middle_pan=pos
            for bar,shift in ((self.horizontalScrollBar(),delta.x()),(self.verticalScrollBar(),delta.y())):
                bar.setValue(bar.value()-shift)
            self._expand_canvas();event.accept();return
        super().mouseMoveEvent(event)
        if self.tool in ('ruler','bed_profile') and self.terrain:
            (self.ruler if self.tool=='ruler' else self.bed_profile).cursor=AVACMapView.from_scene(self,self.mapToScene(event.position().toPoint()));self.viewport().update()
        if self.tool in ('polygon','line','profile','ellipse') and self.points:
            self._draft_cursor=self.from_scene(self.mapToScene(event.position().toPoint()));self._draw_draft()

    def mouseDoubleClickEvent(self,event):
        if self.tool in ('ruler','bed_profile'):event.accept();return
        if event.button()==Qt.LeftButton and self.tool not in ('polygon','line','profile','ellipse'):
            point=self.mapToScene(event.position().toPoint())
            for item in self.items(event.position().toPoint()):
                if item in self.boundary_items and item.data(0):
                    self.boundary_edit_requested.emit(item.data(0));event.accept();return
        super().mouseDoubleClickEvent(event)

    def finish_drawing(self):
        if self.capture_mode:
            minimum=3 if self.tool=='polygon' else 2
            if len(self.points)<minimum:return
            kind=self.capture_mode;points=deepcopy(self.points);self.capture_mode=None
            if kind=='solid_ellipse':
                first,last=np.asarray(points[:2],float);center=(first+last)/2;radius=np.abs(last-first)/2
                angles=np.arange(64)*2*np.pi/64
                points=(center+np.column_stack((np.cos(angles),np.sin(angles)))*radius).tolist()
            self.set_tool('pan');self.captured_geometry.emit(kind,points)
            return
        super().finish_drawing()

    def from_scene(self, point):
        x,y=super().from_scene(point)
        if not self.snap_enabled or self.tool not in ('polygon','line','edit'):return x,y
        candidates=[];p=np.array([x,y])
        contours=[ring for shell,holes in self.rings for ring in (shell,*holes)]+self.sketch_lines
        for contour in contours:
            contour=np.asarray(contour,float)
            if not len(contour):continue
            candidates.extend(contour)
            if self.tool!='edit':
                for a,b in zip(contour[:-1],contour[1:]):
                    d=b-a;t=np.clip(np.dot(p-a,d)/max(np.dot(d,d),1e-30),0,1);candidates.append(a+t*d)
        if candidates:
            rows=np.asarray(candidates);distance=np.linalg.norm(rows-p,axis=1);i=int(np.argmin(distance))
            if distance[i]*abs(self.transform().m11())<=8:return tuple(rows[i])
        return x,y

    def mousePressEvent(self, event):
        if self.tool=='delete' and event.button()==Qt.LeftButton:
            self.delete_feature_at(event.position().toPoint());event.accept();return
        if self.capture_mode=='turbine' and self.tool=='line' and self.terrain and event.button()==Qt.LeftButton:
            self.points.append(list(AVACMapView.from_scene(self,self.mapToScene(event.position().toPoint()))))
            if len(self.points)==2:self.finish_drawing()
            else:self._draw_draft()
            event.accept();return
        if event.button()==Qt.MiddleButton:
            self._middle_pan=event.position().toPoint()
            self._pan_cursor=self.cursor();self.setCursor(Qt.ClosedHandCursor)
            event.accept();return
        if event.button() == Qt.RightButton:
            if self.tool=='bed_profile':
                points=self.bed_profile.vertices()
                if len(points)>=2:self.bed_profile_drawn.emit(points)
                self.set_tool('pan');event.accept();return
            if self.tool in ('polygon', 'line', 'profile') and self.points:
                self.finish_drawing()
            self.set_tool('pan')
            event.accept()
            return
        if self.tool=='identify' and self.terrain and event.button()==Qt.LeftButton:
            self.inspect_result.emit(AVACMapView.from_scene(self,self.mapToScene(event.position().toPoint())));event.accept();return
        if self.tool=='ruler' and self.terrain and event.button()==Qt.LeftButton:
            self.ruler.add(AVACMapView.from_scene(self,self.mapToScene(event.position().toPoint())))
            self.viewport().update();event.accept();return
        if self.tool=='bed_profile' and self.terrain and event.button()==Qt.LeftButton:
            self.bed_profile.add(AVACMapView.from_scene(self,self.mapToScene(event.position().toPoint())))
            self.viewport().update();event.accept();return
        if event.button()==Qt.LeftButton and self.tool=='pan':
            for item in self.items(event.position().toPoint()):
                if item in self.analysis_items and item.data(1):
                    self.measurement_selected.emit(item.data(0),item.data(1));event.accept();return
        if self.tool=='ellipse' and self.terrain and event.button()==Qt.LeftButton:
            self.points.append(list(AVACMapView.from_scene(self,self.mapToScene(event.position().toPoint()))))
            if len(self.points)==2:self.finish_drawing()
            else:self._draw_draft()
            event.accept();return
        if self.tool in ('measurement_edit','breakline_edit','solid_edit','zone_edit'):QGraphicsView.mousePressEvent(self,event);return
        if self.tool == 'boundary' and event.button() == Qt.LeftButton and self.rings:
            point = np.asarray(self.from_scene(self.mapToScene(event.position().toPoint())))
            vertices = np.asarray(self.rings[0][0])
            if np.array_equal(vertices[0], vertices[-1]):
                vertices = vertices[:-1]
            ends = np.roll(vertices, -1, axis=0)
            delta = ends-vertices
            t = np.clip(np.einsum('ij,ij->i', point-vertices, delta)/np.maximum(np.sum(delta*delta, axis=1), 1e-30), 0, 1)
            distances = np.linalg.norm(point-vertices-t[:, None]*delta, axis=1)
            index = int(np.argmin(distances))
            if distances[index]*abs(self.transform().m11()) <= 16:
                self.selected_edges=set() if index in self.selected_edges else {index}
                self.set_boundaries(self.assignments)
                self.edge_selection_changed.emit(sorted(self.selected_edges))
            event.accept()
            return
        super().mousePressEvent(event)

    def set_turbines(self,value):
        from braids_core.river_turbines import rotor_line
        from shiboken6 import isValid
        for item in self.turbine_items:
            if isValid(item) and item.scene() is not None:self.scene().removeItem(item)
        self.turbine_items=[];self.turbines=deepcopy(value)
        for row in (value or {}).get('turbines',[]):
            a,b=rotor_line(row);path=QPainterPath(self.to_scene(*a));path.lineTo(self.to_scene(*b));pen=QPen(QColor('#00a3a8'),4);pen.setCosmetic(True)
            item=self.scene().addPath(path,pen);item.setZValue(33);item.setToolTip(row['id']+' · rotor perpendicular to local river direction');self.turbine_items.append(item)

        for item,row in zip(self.turbine_items,(value or {}).get('turbines',[])):
            item.setData(10,'turbine');item.setData(11,row['id'])

    def mouseReleaseEvent(self,event):
        if event.button()==Qt.MiddleButton and self._middle_pan is not None:
            self._middle_pan=None;self.setCursor(self._pan_cursor)
            self._save_extent();event.accept();return
        super().mouseReleaseEvent(event)

    @staticmethod
    def _dem_drop_path(event):
        from pathlib import Path
        urls=event.mimeData().urls()
        if len(urls)==1 and urls[0].isLocalFile():
            path=Path(urls[0].toLocalFile())
            if path.suffix.lower() in ('.tif','.tiff','.asc','.vrt') and path.is_file():return str(path)
        return None

    def dragEnterEvent(self,event):
        if self._dem_drop_path(event):event.acceptProposedAction()
        else:event.ignore()

    def dragMoveEvent(self,event):
        self.dragEnterEvent(event)

    def dropEvent(self,event):
        path=self._dem_drop_path(event)
        if path:event.acceptProposedAction();self.dem_dropped.emit(path)
        else:event.ignore()


MapView = RiverMapView
