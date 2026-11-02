"""Current-frame extrema and animated raw-cell spatial profiles, display only."""
from pathlib import Path
import re
import numpy as np
from PySide6.QtCore import Qt,QPointF,QRectF
from PySide6.QtGui import QColor,QPainter,QPen,QFont,QFontMetricsF
from PySide6.QtWidgets import QGraphicsItem,QDialog,QVBoxLayout,QHBoxLayout,QCheckBox,QPushButton,QLabel,QFileDialog
from .presentation import display_number


def extrema(payload):
    values=np.asarray(payload['values']);visible=np.isfinite(values)&(np.asarray(payload['rgba'])[:,3]>0)
    cells=np.flatnonzero(visible)
    if not len(cells):return {}
    rows={}
    for kind,choose in (('minimum',np.argmin),('maximum',np.argmax)):
        i=int(cells[choose(values[cells])]);poly=payload['polygons'][i];point=poly.centroid
        if not poly.covers(point):point=poly.representative_point()
        rows[kind]=dict(value=float(values[i]),xy=(point.x,point.y),cell=i,
            leaf=int(payload['frame'].mesh['leaves'][i]))
    return rows


class ExtremumItem(QGraphicsItem):
    def __init__(self,kind,row,label,to_scene):
        super().__init__();self.kind=kind;self.row=row
        self.text=f"{'Min' if kind=='minimum' else 'Max'}: {display_number(row['value'])} · {label}\nE {display_number(row['xy'][0])}   N {display_number(row['xy'][1])}"
        self.font=QFont();self.font.setPointSizeF(9.)
        metrics=QFontMetricsF(self.font);width=max(metrics.horizontalAdvance(line) for line in self.text.splitlines())+14
        height=metrics.height()*2+10;self.box=QRectF(12,-height-8 if kind=='maximum' else 8,width,height)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations);self.setAcceptedMouseButtons(Qt.NoButton)
        self.setZValue(39);self.setPos(to_scene(*row['xy']));self.setToolTip(f"Raw cell value · cell {row['leaf']} · ties use first stored cell")

    def boundingRect(self):return self.box.united(QRectF(-7,-7,14,14)).adjusted(-2,-2,2,2)

    def paint(self,painter,option,widget=None):
        painter.setRenderHint(QPainter.Antialiasing);color=QColor('#b32b35' if self.kind=='maximum' else '#245b9e')
        painter.setPen(QPen(color,2));painter.drawLine(QPointF(-6,0),QPointF(6,0));painter.drawLine(QPointF(0,-6),QPointF(0,6))
        painter.setBrush(QColor(255,255,255,235));painter.setPen(QPen(color,1));painter.drawRect(self.box)
        painter.setPen(QColor('#162b3e'));painter.setFont(self.font);painter.drawText(self.box.adjusted(7,5,-7,-5),Qt.AlignLeft|Qt.AlignVCenter,self.text)


def draw_extrema(panel,payload):
    from shiboken6 import isValid
    for item in panel._extrema_items:
        if isValid(item) and item.scene() is not None:item.scene().removeItem(item)
    panel._extrema_items=[]
    view=panel.map_view
    if view is None:return
    mapping=getattr(view,'to_scene',lambda x,y:QPointF(x,-y))
    for kind,row in extrema(payload).items():
        if panel.extrema_enabled[kind]:
            item=ExtremumItem(kind,row,payload['label'],mapping);view.scene().addItem(item);panel._extrema_items.append(item)
    panel.set_extrema_visible(getattr(view,'result_visible',panel._results_active))


def filename(name,profile):
    raw=f'{name}_{profile.variable}_t{profile.time_s:g}s'
    return re.sub(r'\s+','_',re.sub(r'[<>:"/\\|?*\x00-\x1f]','_',raw)).strip(' ._')


def export_profile(parent,profile,name):
    path,_=QFileDialog.getSaveFileName(parent,'Export spatial profile',filename(name,profile)+'.csv','CSV (*.csv)')
    if path:profile.export_csv(path)


class SpatialProfileDialog(QDialog):
    def __init__(self,panel,points,name,*,follow=True):
        super().__init__(panel);self.panel=panel;self.store=panel.store;self.points=points;self.name=name
        self.resize(850,540);layout=QVBoxLayout(self)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.figure=Figure(constrained_layout=True);self.canvas=FigureCanvasQTAgg(self.figure);layout.addWidget(self.canvas)
        self.ax=self.figure.subplots();self.line,=self.ax.plot([],[],color='#bf3038')
        row=QHBoxLayout();self.follow=QCheckBox('Follow map frame and field');self.follow.setChecked(follow);row.addWidget(self.follow)
        play=QPushButton('Play / pause map');play.clicked.connect(panel.toggle_animation);row.addWidget(play);layout.addLayout(row)
        self.interpolated=QCheckBox('Interpolate cell values');self.interpolated.setChecked(True);layout.addWidget(self.interpolated)
        self.interpolated.toggled.connect(lambda _:self.refresh())
        note=QLabel('Linear interpolation joins cell-intersection centres without overshoot. Uncheck to inspect raw cell steps. Solids and locations outside the mesh remain gaps.');note.setWordWrap(True);layout.addWidget(note)
        row=QHBoxLayout();png=QPushButton('Save plot as PNG');csv=QPushButton('Export profile CSV');row.addWidget(png);row.addWidget(csv);layout.addLayout(row)
        png.clicked.connect(self.save_png);csv.clicked.connect(self.save_csv)
        self.follow.toggled.connect(lambda checked:self.refresh() if checked else None)
        panel.frame_changed.connect(self.frame_changed);self.finished.connect(self.detach)
        self.refresh()

    def detach(self,*_):
        try:self.panel.frame_changed.disconnect(self.frame_changed)
        except (RuntimeError,TypeError):pass

    def frame_changed(self,payload):
        if self.panel.store is not self.store:self.close();return
        if self.follow.isChecked():
            try:self.refresh()
            except Exception as exc:self.panel._fail(exc);self.close()

    def refresh(self):
        self.profile=self.store.spatial_profile(self.points,self.panel.variable.currentData(),self.panel.slider.value(),interpolate=self.interpolated.isChecked())
        p=self.profile;self.line.set_data(p.distance_m,p.values);self.ax.relim();self.ax.autoscale_view()
        self.ax.set(xlabel='Distance along profile [m]',ylabel=f'{p.label} [{p.unit}]',title=f'{self.name} · t = {display_number(p.time_s)} s')
        finite=p.values[np.isfinite(p.values)]
        if len(finite):
            lo=min(0.,float(finite.min()));hi=max(0.,float(finite.max()));span=max(hi-lo,1e-9)
            self.ax.set_ylim(lo-.05*span if lo<0 else 0.,hi+.1*span)
        self.ax.grid(alpha=.25);self.setWindowTitle(f'{self.name} — {p.label}');self.canvas.draw_idle()

    def save_png(self):
        path,_=QFileDialog.getSaveFileName(self,'Export spatial profile plot',filename(self.name+'_'+self.panel.project_name,self.profile)+'.png','PNG (*.png)')
        if path:
            try:
                if Path(path).exists():raise FileExistsError('Choose a new filename; existing plots are not overwritten.')
                self.figure.savefig(path,dpi=180)
            except Exception as exc:self.panel._fail(exc)

    def save_csv(self):
        try:export_profile(self,self.profile,self.name+'_'+self.panel.project_name)
        except Exception as exc:self.panel._fail(exc)
