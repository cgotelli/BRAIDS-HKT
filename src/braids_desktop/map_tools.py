"""Code-drawn map icons and a transient terrain distance/slope ruler."""
import numpy as np
from PySide6.QtCore import Qt,QPointF,QRectF
from PySide6.QtGui import QIcon,QPixmap,QPainter,QPainterPath,QPen,QColor
from .presentation import display_number,numeric_locale


def terrain_elevation(raster,point):
    """Bilinear elevation on the native DEM; never interpolate across NoData."""
    x,y=map(float,point);m=raster.metadata
    if not np.isfinite([x,y]).all() or not (m['xmin']<=x<m['xmax'] and m['ymin']<=y<m['ymax']):return None
    nx=len(raster.x);ny=len(raster.y)
    if not nx or not ny:return None
    dx=float(m['cellsize']);dy=float(m.get('cellsize_y',dx))
    ix=float(np.clip((x-float(raster.x[0]))/dx,0,nx-1));iy=float(np.clip((y-float(raster.y[0]))/dy,0,ny-1))
    x0=int(np.floor(ix));y0=int(np.floor(iy));x1=min(x0+1,nx-1);y1=min(y0+1,ny-1)
    tx=ix-x0;ty=iy-y0;value=0.
    for row,col,weight in ((y0,x0,(1-tx)*(1-ty)),(y0,x1,tx*(1-ty)),(y1,x0,(1-tx)*ty),(y1,x1,tx*ty)):
        if weight<=0.:continue
        z=float(raster.z[row,col])
        if not np.isfinite(z):return None
        value+=weight*z
    return value


def slope_label(slope):
    if slope is None:return 'slope n/a'
    return 'slope '+format(slope,'+.6g').replace('.',numeric_locale().decimalPoint())+' m/m'


def tool_icon(kind):
    pixmap=QPixmap(48,48);pixmap.fill(Qt.transparent);p=QPainter(pixmap)
    p.setRenderHint(QPainter.Antialiasing);p.scale(2,2)
    p.setPen(QPen(QColor('white'),1.5,Qt.SolidLine,Qt.RoundCap,Qt.RoundJoin))
    if kind=='select':
        path=QPainterPath(QPointF(5,2))
        for x,y in ((5,20),(10,15),(13,22),(16,20.5),(12.5,14),(20,14)):path.lineTo(x,y)
        path.closeSubpath();p.setBrush(QColor('white'));p.setPen(QPen(QColor('#26445b'),1.))
        p.drawPath(path)
    elif kind=='pan':
        path=QPainterPath(QPointF(7,12));path.lineTo(7,6);path.cubicTo(7,3.5,10,3.5,10,6)
        path.lineTo(10,10);path.lineTo(10,3.5);path.cubicTo(10,1,13,1,13,3.5)
        path.lineTo(13,10);path.lineTo(13,5);path.cubicTo(13,2.5,16,2.5,16,5)
        path.lineTo(16,11);path.lineTo(16,8);path.cubicTo(16,5.5,19,5.5,19,8)
        path.lineTo(19,15);path.cubicTo(19,18,17,20,16,22);path.lineTo(9,22)
        path.cubicTo(7,20,5,17,3,14);path.cubicTo(1,11,3.5,9.5,5,11.5)
        path.lineTo(7,14);p.drawPath(path)
    elif kind=='ruler':
        p.translate(12,12);p.rotate(-35);p.drawRect(QRectF(-10,-4,20,8))
        for i,x in enumerate(range(-7,9,3)):p.drawLine(QPointF(x,-4),QPointF(x,0 if i%2==0 else -1.5))
    elif kind=='bed_profile':
        # Match the monochrome, hand-drawn toolbar language of pan/ruler.
        path=QPainterPath(QPointF(2,18));path.lineTo(8,8);path.lineTo(12,13)
        path.lineTo(16,5);path.lineTo(22,18);p.drawPath(path)
        p.drawLine(QPointF(3,20),QPointF(22,20))
    else:raise ValueError('Unknown map tool icon')
    p.end();return QIcon(pixmap)


class TemporaryRuler:
    """Only viewport paint/state: never a scene layer or saved model geometry."""
    def __init__(self):self.clear()

    def clear(self):self.points=[];self.cursor=None

    def add(self,point):
        point=tuple(map(float,point))
        if not self.points or np.linalg.norm(np.subtract(self.points[-1],point))>1e-10:self.points.append(point)
        self.cursor=point

    def vertices(self):
        points=list(self.points)
        if points and self.cursor is not None and np.linalg.norm(np.subtract(points[-1],self.cursor))>1e-10:points.append(self.cursor)
        return points

    def lengths(self):
        return [float(np.linalg.norm(np.subtract(b,a))) for a,b in zip(self.vertices()[:-1],self.vertices()[1:])]

    def slopes(self,raster):
        """Positive means elevation falls in the order the points were drawn."""
        points=self.vertices();elevations=[terrain_elevation(raster,p) for p in points]
        lengths=self.lengths();segments=[]
        for a,b,length in zip(elevations[:-1],elevations[1:],lengths):
            segments.append((a-b)/length if a is not None and b is not None and length>0 else None)
        total=(elevations[0]-elevations[-1])/sum(lengths) if lengths and all(v is not None for v in elevations) and sum(lengths)>0 else None
        return segments,total

    def draw(self,painter,view):
        if not self.points:return
        world=self.vertices();points=[QPointF(view.mapFromScene(view.to_scene(*p))) for p in world]
        painter.save();painter.resetTransform();painter.setRenderHint(QPainter.Antialiasing)
        path=QPainterPath(points[0])
        for point in points[1:]:path.lineTo(point)
        for color,width in (('#152638',4.),('#ffe978',2.)):
            painter.setPen(QPen(QColor(color),width,Qt.SolidLine,Qt.RoundCap,Qt.RoundJoin));painter.drawPath(path)
        painter.setBrush(QColor('#ffe978'));painter.setPen(QPen(QColor('#152638'),1.))
        for point in points[:len(self.points)]:painter.drawEllipse(point,3.,3.)
        def label(position,text):
            metrics=painter.fontMetrics();rect=QRectF(position.x(),position.y(),metrics.horizontalAdvance(text)+12,metrics.height()+6)
            rect.moveLeft(max(2.,min(rect.left(),view.viewport().width()-rect.width()-2)))
            rect.moveTop(max(2.,min(rect.top(),view.viewport().height()-rect.height()-2)))
            painter.setPen(Qt.NoPen);painter.setBrush(QColor(25,46,64,225));painter.drawRoundedRect(rect,3,3)
            painter.setPen(QColor('white'));painter.drawText(rect.adjusted(6,3,-6,-3),Qt.AlignCenter,text)
        lengths=self.lengths();slopes,total_slope=self.slopes(view.terrain.raster)
        for a,b,length,slope in zip(points[:-1],points[1:],lengths,slopes):
            delta=b-a;norm=max(np.hypot(delta.x(),delta.y()),1.);middle=(a+b)*.5
            label(middle+QPointF(-delta.y()/norm*12,delta.x()/norm*12),display_number(length)+' m | '+slope_label(slope))
        mouse=QPointF(view.mapFromScene(view.to_scene(*(self.cursor or self.points[-1]))))
        label(mouse+QPointF(14,18),'Total: '+display_number(sum(lengths))+' m | '+slope_label(total_slope))
        painter.restore()


class TemporaryBedProfile(TemporaryRuler):
    """A transient brown terrain transect with no on-map numeric labels."""
    def draw(self,painter,view):
        if not self.points:return
        world=self.vertices();points=[QPointF(view.mapFromScene(view.to_scene(*p))) for p in world]
        painter.save();painter.resetTransform();painter.setRenderHint(QPainter.Antialiasing)
        path=QPainterPath(points[0])
        for point in points[1:]:path.lineTo(point)
        painter.setPen(QPen(QColor('#7a4a21'),3.,Qt.SolidLine,Qt.RoundCap,Qt.RoundJoin));painter.drawPath(path)
        painter.restore()

    def sampled(self,raster):
        """Return chain distance and DEM elevation with no interpolation over NoData."""
        points=self.vertices()
        if len(points)<2:return np.empty(0),np.empty(0),np.empty((0,2))
        cell=float(raster.metadata['cellsize'])
        distance=[];elevation=[];locations=[];total=0.
        for first,last in zip(points[:-1],points[1:]):
            length=float(np.linalg.norm(np.subtract(last,first)))
            count=max(2,int(np.ceil(length/max(cell*.5,1.e-9)))+1)
            for i in range(count if not distance else count-1):
                fraction=i/(count-1);point=np.asarray(first)*(1-fraction)+np.asarray(last)*fraction
                distance.append(total+fraction*length);elevation.append(terrain_elevation(raster,point));locations.append(point)
            total+=length
        return (np.asarray(distance,float),np.asarray([np.nan if z is None else z for z in elevation],float),
                np.asarray(locations,float))

    def samples(self,raster):
        distance,elevation,_=self.sampled(raster)
        return distance,elevation
