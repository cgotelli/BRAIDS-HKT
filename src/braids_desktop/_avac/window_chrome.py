"""Themed native-move window controls adapted from AVAC Desktop, AGPL-3.0."""
from PySide6.QtCore import QEvent,Qt,QObject,QRect
from PySide6.QtWidgets import QFrame,QHBoxLayout,QLabel,QPushButton,QWidget
from .i18n import tr


class TitleBar(QFrame):
    def __init__(self,window):
        super().__init__(window);self.window=window;self.setObjectName('TitleBar')
        layout=QHBoxLayout(self);layout.setContentsMargins(12,0,0,0)
        self.logo=QLabel();self.logo.setPixmap(window.windowIcon().pixmap(22,22));layout.addWidget(self.logo)
        self.title=QLabel('BRAIDS');self.title.setAttribute(Qt.WA_TransparentForMouseEvents);layout.addWidget(self.title);layout.addStretch()
        for text,hint,callback in [('−','Minimize',window.showMinimized),('□','Maximize / restore',self.toggle_maximized),('×','Close',window.close)]:
            button=QPushButton(text);button.setObjectName('CloseWindow' if hint=='Close' else 'WindowControl')
            button.setToolTip(tr(hint));button.setFixedSize(46,34);button.clicked.connect(callback);layout.addWidget(button)
            if hint=='Maximize / restore':self.maximize_button=button
        window.installEventFilter(self)

    def eventFilter(self,obj,event):
        if obj is self.window and event.type() in (QEvent.Show,QEvent.WinIdChange):windows_taskbar_controls(self.window)
        if obj is self.window and event.type()==QEvent.WindowStateChange:
            self.maximize_button.setText('❐' if self.window.isMaximized() else '□')
        return super().eventFilter(obj,event)

    def toggle_maximized(self):
        self.window.showNormal() if self.window.isMaximized() else self.window.showMaximized()

    def mousePressEvent(self,event):
        if event.button()==Qt.LeftButton and self.window.windowHandle():
            self.window.windowHandle().startSystemMove();event.accept()
        else:super().mousePressEvent(event)

    def mouseDoubleClickEvent(self,event):
        if event.button()==Qt.LeftButton:self.toggle_maximized();event.accept()


def windows_taskbar_controls(window):
    """Keep taskbar controls, but no Windows-drawn frame or rounded corners."""
    import sys
    if sys.platform!='win32':return
    import ctypes
    user32=ctypes.windll.user32;hwnd=ctypes.c_void_p(int(window.winId()))
    required=0x00020000|0x00010000|0x00080000  # MINIMIZEBOX, MAXIMIZEBOX, SYSMENU
    style=user32.GetWindowLongW(hwnd,-16)
    updated=(style&~(0x00040000|0x00c00000))|required  # No THICKFRAME, BORDER or DLGFRAME.
    if style!=updated:
        user32.SetWindowLongW(hwnd,-16,updated)
        user32.SetWindowPos(hwnd,None,0,0,0,0,0x0001|0x0002|0x0004|0x0010|0x0020)
    # Windows 11 attributes; older systems return an unsupported HRESULT and
    # already have square frameless windows. The sole visible border is Qt's.
    dwm=ctypes.windll.dwmapi
    for attribute,value in ((33,1),(34,0xfffffffe)):
        setting=ctypes.c_uint(value)
        dwm.DwmSetWindowAttribute(hwnd,attribute,ctypes.byref(setting),ctypes.sizeof(setting))


class ResizeEdge(QWidget):
    """Invisible client-side hit area; never adds a native/painted frame."""
    def __init__(self,window,edges,cursor):
        super().__init__(window);self.edges=edges;self.origin=None;self.start=None
        self.setObjectName('WindowResizeEdge');self.setCursor(cursor)
        self.setAttribute(Qt.WA_NoSystemBackground);self.setStyleSheet('background: transparent; border: none;')

    def mousePressEvent(self,event):
        if event.button()==Qt.LeftButton:
            self.origin=event.globalPosition().toPoint();self.start=QRect(self.window().geometry());event.accept()
        else:super().mousePressEvent(event)

    def mouseMoveEvent(self,event):
        if self.origin is None:return super().mouseMoveEvent(event)
        window=self.window();delta=event.globalPosition().toPoint()-self.origin;rect=QRect(self.start)
        if 'left' in self.edges:rect.setLeft(min(rect.right()-window.minimumWidth()+1,max(rect.right()-window.maximumWidth()+1,rect.left()+delta.x())))
        if 'right' in self.edges:rect.setRight(max(rect.left()+window.minimumWidth()-1,min(rect.left()+window.maximumWidth()-1,rect.right()+delta.x())))
        if 'top' in self.edges:rect.setTop(min(rect.bottom()-window.minimumHeight()+1,max(rect.bottom()-window.maximumHeight()+1,rect.top()+delta.y())))
        if 'bottom' in self.edges:rect.setBottom(max(rect.top()+window.minimumHeight()-1,min(rect.top()+window.maximumHeight()-1,rect.bottom()+delta.y())))
        window.setGeometry(rect);event.accept()

    def mouseReleaseEvent(self,event):
        if event.button()==Qt.LeftButton:self.origin=None;self.start=None;event.accept()
        else:super().mouseReleaseEvent(event)


class ResizeEdges(QObject):
    """Preserve edge/corner resizing without Windows' thick resize border."""
    def __init__(self,window):
        super().__init__(window);self.window=window;self.handles={}
        for edges,cursor in ((('left',),Qt.SizeHorCursor),(('right',),Qt.SizeHorCursor),
                (('top',),Qt.SizeVerCursor),(('bottom',),Qt.SizeVerCursor),
                (('top','left'),Qt.SizeFDiagCursor),(('top','right'),Qt.SizeBDiagCursor),
                (('bottom','left'),Qt.SizeBDiagCursor),(('bottom','right'),Qt.SizeFDiagCursor)):
            self.handles[edges]=ResizeEdge(window,edges,cursor)
        window.installEventFilter(self)

    def eventFilter(self,obj,event):
        if obj is self.window and event.type() in (QEvent.Resize,QEvent.Show,QEvent.WindowStateChange):self.update()
        return super().eventFilter(obj,event)

    def update(self):
        w,h=self.window.width(),self.window.height();edge=5;corner=8
        rectangles={('left',):(0,corner,edge,h-2*corner),('right',):(w-edge,corner,edge,h-2*corner),
            ('top',):(corner,0,w-2*corner,edge),('bottom',):(corner,h-edge,w-2*corner,edge),
            ('top','left'):(0,0,corner,corner),('top','right'):(w-corner,0,corner,corner),
            ('bottom','left'):(0,h-corner,corner,corner),('bottom','right'):(w-corner,h-corner,corner,corner)}
        visible=not (self.window.isMaximized() or self.window.isFullScreen() or self.window.isMinimized())
        for edges,handle in self.handles.items():
            handle.setGeometry(*rectangles[edges]);handle.setVisible(visible);handle.raise_()
