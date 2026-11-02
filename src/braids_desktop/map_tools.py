"""Code-drawn map icons and a transient, non-model distance ruler."""
import numpy as np
from PySide6.QtCore import Qt,QPointF,QRectF
from PySide6.QtGui import QIcon,QPixmap,QPainter,QPainterPath,QPen,QColor
from .presentation import display_number


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
        lengths=self.lengths()
        for a,b,length in zip(points[:-1],points[1:],lengths):
            delta=b-a;norm=max(np.hypot(delta.x(),delta.y()),1.);middle=(a+b)*.5
            label(middle+QPointF(-delta.y()/norm*12,delta.x()/norm*12),display_number(length)+' m')
        mouse=QPointF(view.mapFromScene(view.to_scene(*(self.cursor or self.points[-1]))))
        label(mouse+QPointF(14,18),'Total: '+display_number(sum(lengths))+' m')
        painter.restore()
