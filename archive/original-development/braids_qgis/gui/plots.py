"""Native Qt plots for boundary forcing and sampled cross sections."""
from __future__ import annotations

import math

from qgis.PyQt.QtCore import QPointF,QRectF,Qt
from qgis.PyQt.QtGui import QColor,QFontMetricsF,QPainter,QPainterPath,QPen,QPolygonF
from qgis.PyQt.QtWidgets import QDialog,QLabel,QScrollArea,QVBoxLayout,QWidget


COLORS=(QColor(30,112,190),QColor(216,83,53),QColor(47,143,74),QColor(120,75,160))


class LinePlot(QWidget):
    def __init__(self,title,x_label,y_label,series,parent=None,include_zero=False):
        super().__init__(parent);self.title=title;self.x_label=x_label;self.y_label=y_label
        self.include_zero=bool(include_zero)
        self.series=[(name,tuple(map(float,x)),tuple(map(float,y)),color) for name,x,y,color in series]
        self.setMinimumHeight(240)

    def paintEvent(self,event):
        painter=QPainter(self);painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(),QColor("white"));bounds=QRectF(65,30,max(20,self.width()-90),max(20,self.height()-82))
        points=[(x,y) for _,xs,ys,_ in self.series for x,y in zip(xs,ys) if math.isfinite(x) and math.isfinite(y)]
        if not points:return
        xmin,xmax=min(x for x,_ in points),max(x for x,_ in points);ymin,ymax=min(y for _,y in points),max(y for _,y in points)
        if self.include_zero:ymin=min(ymin,0.);ymax=max(ymax,0.)
        if xmax==xmin:xmin-=.5;xmax+=.5
        if ymax==ymin:ymin-=max(1.,abs(ymin)*.05);ymax+=max(1.,abs(ymax)*.05)
        ypad=.08*(ymax-ymin);ymin-=ypad;ymax+=ypad
        # Size the margin from the actual font: fixed 58 px labels can clip
        # a minus sign or elevation digit on high-DPI QGIS displays.
        metrics=QFontMetricsF(painter.font())
        tick_width=max(metrics.horizontalAdvance(format(y,".5g")) for y in (ymin,ymax))
        left=max(65.,tick_width+30.);bounds=QRectF(left,30,max(20,self.width()-left-25),max(20,self.height()-82))
        def xy(x,y):return QPointF(bounds.left()+(x-xmin)/(xmax-xmin)*bounds.width(),bounds.bottom()-(y-ymin)/(ymax-ymin)*bounds.height())
        if self.include_zero:
            zero=xy(xmin,0.).y();painter.setPen(QPen(QColor(110,110,110),1,Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(bounds.left(),zero),QPointF(bounds.right(),zero))
            painter.setPen(QPen(QColor(75,75,75),1));painter.drawText(QRectF(24,zero-9,left-31,18),Qt.AlignmentFlag.AlignRight,"0")
        painter.setPen(QPen(QColor(75,75,75),1));painter.drawRect(bounds)
        painter.drawText(QRectF(0,3,self.width(),22),Qt.AlignmentFlag.AlignCenter,self.title)
        painter.drawText(QRectF(bounds.left(),bounds.bottom()+24,bounds.width(),20),Qt.AlignmentFlag.AlignCenter,self.x_label)
        painter.save();painter.translate(17,bounds.center().y());painter.rotate(-90);painter.drawText(QRectF(-bounds.height()/2,-10,bounds.height(),20),Qt.AlignmentFlag.AlignCenter,self.y_label);painter.restore()
        painter.drawText(QRectF(24,bounds.top()-7,left-31,18),Qt.AlignmentFlag.AlignRight,format(ymax,".5g"));painter.drawText(QRectF(24,bounds.bottom()-9,left-31,18),Qt.AlignmentFlag.AlignRight,format(ymin,".5g"))
        for fraction in (0.,.25,.5,.75,1.):
            value=xmin+fraction*(xmax-xmin);position=bounds.left()+fraction*bounds.width()
            painter.drawLine(QPointF(position,bounds.bottom()),QPointF(position,bounds.bottom()+3))
            width=metrics.horizontalAdvance(format(value,".5g"))+6
            label_x=min(self.width()-width,max(0.,position-width/2))
            painter.drawText(QRectF(label_x,bounds.bottom()+4,width,18),Qt.AlignmentFlag.AlignCenter,format(value,".5g"))
        legend_x=bounds.right()-4
        for index,(name,xs,ys,color) in enumerate(self.series):
            path=QPainterPath();valid=False
            for x,y in zip(xs,ys):
                if not math.isfinite(x) or not math.isfinite(y):valid=False;continue
                point=xy(x,y)
                if valid:path.lineTo(point)
                else:path.moveTo(point);valid=True
            painter.setPen(QPen(color,2));painter.drawPath(path)
            width=QFontMetricsF(painter.font()).horizontalAdvance(name);legend_x-=width+26
            painter.drawLine(QPointF(legend_x,bounds.top()+13),QPointF(legend_x+16,bounds.top()+13));painter.setPen(QColor(45,45,45));painter.drawText(QPointF(legend_x+20,bounds.top()+17),name)


def show_plot_dialog(parent,title,plots,caption=None):
    dialog=QDialog(parent);dialog.setWindowTitle(title);dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose,True)
    layout=QVBoxLayout(dialog)
    if caption:
        label=QLabel(caption);label.setWordWrap(True);label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse);layout.addWidget(label)
    if len(plots)>3:
        scroll=QScrollArea(dialog);scroll.setWidgetResizable(True);content=QWidget(scroll);content_layout=QVBoxLayout(content)
        for plot in plots:content_layout.addWidget(LinePlot(plot["title"],plot.get("x_label","Time (s)"),plot["y_label"],plot["series"],content,plot.get("include_zero",False)))
        scroll.setWidget(content);layout.addWidget(scroll)
    else:
        for plot in plots:layout.addWidget(LinePlot(plot["title"],plot.get("x_label","Time (s)"),plot["y_label"],plot["series"],dialog,plot.get("include_zero",False)))
        if not plots:layout.addWidget(QLabel("No values to plot."))
    dialog.resize(820,min(850,90+270*max(1,len(plots))))
    windows=getattr(parent,"_plot_windows",[]);windows.append(dialog);parent._plot_windows=windows
    dialog.destroyed.connect(lambda *_:windows.remove(dialog) if dialog in windows else None);dialog.show();return dialog
