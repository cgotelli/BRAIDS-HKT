"""Continuous vector colour shading clipped to the actual display footprint."""
import numpy as np
import shapely
from collections import OrderedDict
from scipy.spatial import Delaunay, cKDTree, QhullError
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainterPath, QLinearGradient, QColor, QBrush, QPainter
from PySide6.QtWidgets import QGraphicsItem


def painter_path(geometry, to_scene):
    path=QPainterPath();path.setFillRule(Qt.OddEvenFill)
    pieces=(geometry,) if geometry.geom_type=='Polygon' else geometry.geoms
    for piece in pieces:
        if piece.geom_type!='Polygon':continue
        for ring in (piece.exterior,*piece.interiors):
            points=np.asarray(ring.coords);path.moveTo(to_scene(*points[0]))
            for point in points[1:]:path.lineTo(to_scene(*point))
            path.closeSubpath()
    return path


class InterpolatedFieldItem(QGraphicsItem):
    """Linear scalar interpolation drawn as vector gradients, without pixels.

    Cell centres retain their saved values. Boundary nodes extend the nearest
    cell value to the edge. Every triangle is clipped by a single exact path,
    so neither its colour nor its footprint can escape the calculation domain.
    """
    def __init__(self,to_scene):
        super().__init__();self.to_scene=to_scene;self.parts=[];self.clip=QPainterPath();self._bounds=QRectF()
        self._topologies=OrderedDict();self._paint_bounds=np.empty((0,4))
        self.setZValue(20.2);self.setFlag(QGraphicsItem.ItemUsesExtendedStyleOption,True)

    def boundingRect(self):return self._bounds

    def set_field(self,frame,values,variable,limits,palette,*,domain=None,terrain=False,zero_transparent=True):
        from .visual_fields import _display_geometry,terrain_wet_edges
        from matplotlib import colormaps
        polygons,centres,_,mesh_domain=_display_geometry(frame)
        valid=np.isfinite(values)
        # Dry hydraulic zero cells are deliberately included when the user
        # requests opaque zeros. Terrain-aware geometry only relocates water.
        hydraulic=variable not in ('bed_m','bed_change_m','initial_bed_m','bedrock_m','erodible_depth_m','sediment_stock_kg_m2')
        if hydraulic and zero_transparent:valid&=frame.q[0]>1e-6
        if zero_transparent:valid&=np.asarray(values)!=0.
        self.prepareGeometryChange();self.parts=[];self.clip=QPainterPath();self._bounds=QRectF()
        if not np.any(valid):self.hide();return
        geometry_key=(frame.epoch,id(frame.polygons),np.packbits(valid).tobytes(),
                      domain.wkb if domain is not None else None,
                      (frame.q[0].tobytes(),frame.z.tobytes()) if terrain and hydraulic and zero_transparent else None,
                      self.to_scene(0,0),self.to_scene(1,1))
        # QPointF is not hashable on all supported Qt builds.
        geometry_key=(*geometry_key[:-2],geometry_key[-2].x(),geometry_key[-2].y(),geometry_key[-1].x(),geometry_key[-1].y())
        cached=self._topologies.get(geometry_key)
        lo,hi=map(float,limits);cmap=colormaps[palette]
        stops=[(float(t),QColor.fromRgbF(*cmap(t))) for t in np.linspace(0,1,65)]
        if cached is not None:
            self._topologies.move_to_end(geometry_key)
            self.clip,self._bounds,prepared=cached
            self._shade(prepared,values,limits,cmap,stops)
            return
        footprint=mesh_domain if np.all(valid) else shapely.union_all(polygons[valid])
        if terrain and hydraulic and zero_transparent:
            rows,wedges=terrain_wet_edges(frame)
            if len(rows):
                original=shapely.union_all(polygons[rows])
                footprint=footprint.difference(original).union(shapely.union_all(wedges))
        if domain is not None:footprint=footprint.intersection(domain)
        self.clip=painter_path(footprint,self.to_scene);self._bounds=self.clip.boundingRect()
        pieces=list(footprint.geoms) if footprint.geom_type=='MultiPolygon' else [footprint]
        prepared=[]
        for piece in pieces:
            ids=np.flatnonzero(valid & (shapely.intersects(polygons,piece) if terrain else
                                        shapely.intersects_xy(piece,centres[:,0],centres[:,1])))
            if not len(ids):continue
            boundary=np.concatenate([np.asarray(r.coords)[:-1] for r in (piece.exterior,*piece.interiors)])
            boundary=np.unique(boundary,axis=0)
            xy=np.vstack((centres[ids],boundary))
            owners=np.concatenate((ids,ids[cKDTree(centres[ids]).query(boundary)[1]]))
            try:tri=Delaunay(xy)
            except QhullError:continue
            triangles=xy[tri.simplices]
            # Reject triangles spanning holes or an inward bend. Boundary
            # intersections are clipped exactly by the painter as well.
            shapes=shapely.polygons(triangles)
            allowed=shapely.covers(piece,shapes)
            for vertices,owners3 in zip(triangles[allowed],owners[tri.simplices[allowed]]):
                points=[self.to_scene(*p) for p in vertices]
                scene=np.array([[p.x(),p.y()] for p in points])
                path=QPainterPath();path.moveTo(points[0]);path.lineTo(points[1]);path.lineTo(points[2]);path.closeSubpath()
                try:inverse=np.linalg.inv(scene[1:]-scene[0])
                except np.linalg.LinAlgError:continue
                prepared.append((path.boundingRect(),path,scene[0],inverse,owners3))
        self._topologies[geometry_key]=(self.clip,self._bounds,prepared)
        while len(self._topologies)>3:self._topologies.popitem(last=False)
        self._shade(prepared,values,limits,cmap,stops)

    def _shade(self,prepared,values,limits,cmap,stops):
        lo,hi=map(float,limits);self.parts=[]
        if not prepared:self.hide();return
        states=np.asarray(values)[np.asarray([p[4] for p in prepared])]
        inverse=np.asarray([p[3] for p in prepared])
        gradients=np.einsum('ijk,ik->ij',inverse,states[:,1:]-states[:,:1])
        for (bounds,path,origin,_,_),state,gradient in zip(prepared,states,gradients):
                minimum,maximum=float(state.min()),float(state.max())
                if maximum-minimum<1e-12:
                    brush=QBrush(QColor.fromRgbF(*cmap(np.clip((minimum-lo)/(hi-lo),0,1))))
                else:
                    norm=float(gradient@gradient)
                    start=origin+gradient*(lo-state[0])/norm
                    end=origin+gradient*(hi-state[0])/norm
                    ramp=QLinearGradient(QPointF(*start),QPointF(*end))
                    ramp.setStops(stops)
                    brush=QBrush(ramp)
                self.parts.append((bounds,path,brush))
        self._paint_bounds=np.asarray([[b.left(),b.top(),b.right(),b.bottom()] for b,_,_ in self.parts])
        self.show();self.update()

    def paint(self,painter,option,widget=None):
        painter.save();painter.setClipPath(self.clip,Qt.IntersectClip)
        painter.setRenderHint(QPainter.Antialiasing,False);painter.setPen(Qt.NoPen)
        r=option.exposedRect;b=self._paint_bounds
        selected=np.flatnonzero((b[:,0]<=r.right())&(b[:,2]>=r.left())&(b[:,1]<=r.bottom())&(b[:,3]>=r.top()))
        for i in selected:
            _,path,brush=self.parts[i];painter.fillPath(path,brush)
        painter.restore()
