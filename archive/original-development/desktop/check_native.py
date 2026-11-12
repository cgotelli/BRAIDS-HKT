"""Native Windows construction and taskbar-style/minimize qualification."""
import json
from pathlib import Path
import ctypes
import launch
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer,QSettings
from braids_desktop.shell import MainWindow
from braids_desktop._avac.theme import apply_theme

base=Path(__file__).resolve().parent
(base/'evidence/chrome143').mkdir(parents=True,exist_ok=True)
app=QApplication([]);QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(base/'evidence/chrome143/native_settings'))
apply_theme(app);window=MainWindow(runtime=base/'runtime');window.show();result={}

def check():
    try:
        hwnd=ctypes.c_void_p(int(window.winId()));style=ctypes.windll.user32.GetWindowLongW(hwnd,-16)
        required=0x20000|0x10000|0x80000
        assert style&required==required
        assert not style&(0x40000|0xc00000)
        window.showMinimized();app.processEvents();assert window.isMinimized()
        window.showNormal();app.processEvents();assert not window.isMinimized()
        from ctypes import wintypes
        rect=wintypes.RECT();ctypes.windll.user32.GetWindowRect(hwnd,ctypes.byref(rect))
        client=wintypes.RECT();ctypes.windll.user32.GetClientRect(hwnd,ctypes.byref(client))
        assert rect.right-rect.left==client.right and rect.bottom-rect.top==client.bottom
        from PySide6.QtCore import QPoint,QPointF,QEvent,QCoreApplication,Qt
        from PySide6.QtGui import QMouseEvent
        from PySide6.QtTest import QTest
        handle=window.resize_edges.handles[('bottom','right')]
        local=handle.rect().center();position=handle.mapToGlobal(local);before=window.size();delta=QPoint(40,30)
        QTest.mousePress(handle,Qt.LeftButton,Qt.NoModifier,local)
        QCoreApplication.sendEvent(handle,QMouseEvent(QEvent.MouseMove,QPointF(local+delta),QPointF(position+delta),Qt.NoButton,Qt.LeftButton,Qt.NoModifier))
        QTest.mouseRelease(handle,Qt.LeftButton,Qt.NoModifier,handle.rect().center());app.processEvents()
        assert window.width()==before.width()+40 and window.height()==before.height()+30
        attributes={}
        for attribute,expected in ((33,1),(34,0xfffffffe)):
            value=ctypes.c_uint()
            code=ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd,attribute,ctypes.byref(value),ctypes.sizeof(value))
            attributes[str(attribute)]=dict(hresult=code,value=value.value)
            if code==0:assert value.value==expected
        result.update(native_window=True,native_minimize_style=True,minimize_restore=True,
                      native_resize_frame=False,no_nonclient_border=True,client_resize_drag=True,dwm=attributes,
                      appearance=app.property('appearance'),qt_color_scheme=app.styleHints().colorScheme().name)
        assert app.property('appearance')=='light' and app.styleHints().colorScheme()==Qt.ColorScheme.Light
        image=window.grab().toImage();w,h=image.width(),image.height()
        assert all(image.pixelColor(x,y).name()=='#808080' for x,y in ((0,0),(w-1,0),(0,h-1),(w-1,h-1),(w//2,0),(w//2,h-1),(0,h//2),(w-1,h//2)))
        result['square_gray_border_pixels']=True
        assert image.save(str(base/'evidence/chrome143/WINDOW.png'))
    except Exception as exc:result['error']=str(exc)
    finally:
        folder=base/'evidence/chrome143';folder.mkdir(parents=True,exist_ok=True)
        (folder/'NATIVE.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        window.dirty=False;window.close();app.quit()

QTimer.singleShot(250,check);app.exec()
raise SystemExit(1 if 'error' in result else 0)
