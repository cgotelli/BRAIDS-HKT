"""PNG frame and MP4 range export with a closed-file atomic publication."""
from pathlib import Path
import os
from uuid import uuid4
import math
import numpy as np
from PySide6.QtCore import Qt,QRectF
from PySide6.QtGui import QImage,QPainter
from PySide6.QtWidgets import QDialog,QVBoxLayout,QFormLayout,QComboBox,QSpinBox,QCheckBox,QDialogButtonBox,QLabel


class ExportDialog(QDialog):
    def __init__(self,times,current,view_size,parent=None):
        super().__init__(parent);self.setWindowTitle('Export results');self.resize(450,400)
        root=QVBoxLayout(self);form=QFormLayout();root.addLayout(form)
        self.kind=QComboBox();self.kind.addItem('MP4 video','mp4');self.kind.addItem('One PNG time frame','png');form.addRow('Format',self.kind)
        self.frame=QComboBox();self.first=QComboBox();self.last=QComboBox()
        for i,time in enumerate(times):
            for combo in (self.frame,self.first,self.last):combo.addItem(f'{i+1:04d} — t = {time:.6g} s',i)
        self.frame.setCurrentIndex(current);self.last.setCurrentIndex(len(times)-1)
        self.all=QCheckBox('All frames');self.all.setChecked(True);form.addRow(self.all)
        for name,w in [('Time frame',self.frame),('First frame',self.first),('Last frame',self.last)]:form.addRow(name,w)
        self.fps=QSpinBox();self.fps.setRange(1,120);self.fps.setValue(20);form.addRow('Frames per second',self.fps)
        self.resolution=QComboBox()
        width,height=view_size
        for label,size in [('Current view',(max(2,width//2*2),max(2,height//2*2))),('1280 × 720',(1280,720)),('1920 × 1080',(1920,1080)),('3840 × 2160',(3840,2160)),('Custom',None)]:self.resolution.addItem(label,size)
        form.addRow('Resolution',self.resolution)
        self.width=QSpinBox();self.height=QSpinBox()
        for field,value in ((self.width,width),(self.height,height)):field.setRange(2,8192);field.setSingleStep(2);field.setValue(max(2,value//2*2))
        form.addRow('Width (pixels)',self.width);form.addRow('Height (pixels)',self.height)
        note=QLabel('The current map extent, visible layers and decorations are exported. Change their checkboxes before exporting. Paused particle trails are omitted. Video uses a fixed color range and equal frame durations.');note.setWordWrap(True);root.addWidget(note)
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel);buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);root.addWidget(buttons)
        self.kind.currentIndexChanged.connect(self._changed);self.all.toggled.connect(self._changed);self.resolution.currentIndexChanged.connect(self._changed);self._changed()

    def _changed(self,*_):
        video=self.kind.currentData()=='mp4';self.all.setEnabled(video);self.fps.setEnabled(video);self.frame.setEnabled(not video)
        self.first.setEnabled(video and not self.all.isChecked());self.last.setEnabled(video and not self.all.isChecked())
        custom=self.resolution.currentData() is None;self.width.setEnabled(custom);self.height.setEnabled(custom)

    def selection(self):
        kind=self.kind.currentData()
        indices=[self.frame.currentData()] if kind=='png' else list(range(self.frame.count())) if self.all.isChecked() else list(range(self.first.currentData(),self.last.currentData()+1))
        if not indices:raise ValueError('Last frame must not precede first frame.')
        size=self.resolution.currentData() or (self.width.value(),self.height.value())
        return dict(kind=kind,indices=indices,fps=self.fps.value(),size=tuple(size))


def export_media(store,target,*,kind,indices,fps,size,variable,mesh_mode,render,progress=None,cancelled=None,limits=None,palette=None):
    from .results import colour_values
    target=Path(target);indices=list(indices);width,height=map(int,size)
    if target.exists():raise FileExistsError('Choose a new export filename; existing files are not overwritten.')
    if kind not in ('png','mp4') or not indices or (kind=='png' and len(indices)!=1):raise ValueError('Choose one PNG frame or a nonempty MP4 frame range.')
    if not math.isfinite(fps) or fps<=0 or min(width,height)<2:raise ValueError('Positive FPS and a valid resolution are required.')
    if kind=='mp4' and (width%2 or height%2):raise ValueError('MP4 width and height must be even numbers.')
    for index in indices:
        if not 0<=index<len(store.times):raise ValueError('Export frame is outside the saved history.')
    def stop():
        if cancelled and cancelled():raise InterruptedError('Export cancelled; no incomplete output published.')
    temporary=target.with_name('.'+target.stem+'.'+uuid4().hex+'.partial.'+kind);writer=None
    try:
        if limits is None:
            low,high=math.inf,-math.inf
            for j,index in enumerate(indices):
                stop();values=store.display_values(store.read(index),variable);finite=values[np.isfinite(values)]
                if len(finite):low=min(low,float(finite.min()));high=max(high,float(finite.max()))
                if progress:progress(.15*(j+1)/len(indices))
            _,limits=colour_values([low,high],variable)
        if kind=='mp4':
            import imageio_ffmpeg
            writer=imageio_ffmpeg.write_frames(str(temporary),(width,height),fps=fps,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','20','-preset','fast','-movflags','+faststart'])
            writer.send(None)
        for j,index in enumerate(indices):
            stop();payload=store.payload(index,variable,mesh_mode=mesh_mode,limits=limits,palette=palette);source=render(payload)
            image=QImage(width,height,QImage.Format_RGBA8888);image.fill(Qt.white if kind=='mp4' else Qt.transparent)
            painter=QPainter(image);painter.setRenderHint(QPainter.SmoothPixmapTransform)
            fitted=source.size().scaled(width,height,Qt.KeepAspectRatio)
            rectangle=QRectF((width-fitted.width())/2,(height-fitted.height())/2,fitted.width(),fitted.height())
            painter.drawImage(rectangle,source);painter.end()
            if kind=='png':
                if not image.save(str(temporary),'PNG'):raise OSError('Could not save PNG frame.')
            else:
                rgb=image.convertToFormat(QImage.Format_RGB888);data=np.frombuffer(rgb.bits(),dtype=np.uint8).reshape(height,rgb.bytesPerLine())[:,:width*3]
                writer.send(np.ascontiguousarray(data).tobytes())
            if progress:progress(.15+.85*(j+1)/len(indices))
        if writer is not None:writer.close();writer=None
        stop()
        if not temporary.is_file() or temporary.stat().st_size==0:raise OSError('The encoder did not produce a complete file.')
        # Do not replace an export created by another session during encoding.
        if target.exists():raise FileExistsError(target)
        os.rename(temporary,target);return target
    finally:
        if writer is not None:writer.close()
        temporary.unlink(missing_ok=True)
