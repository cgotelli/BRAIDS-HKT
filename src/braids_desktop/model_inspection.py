"""Read-only model inspection while a worker owns the simulation settings."""
from PySide6.QtCore import QObject,QEvent,Qt
from PySide6.QtWidgets import (QApplication,QWidget,QAbstractButton,QAbstractSpinBox,
    QComboBox,QSlider,QDial,QLineEdit,QTextEdit,QPlainTextEdit,QAbstractItemView,
    QTabBar,QScrollBar,QTableWidget,QGroupBox)
from shiboken6 import isValid
from .accordion import AccordionSection


class ModelInspectionLock(QObject):
    def __init__(self,root):
        super().__init__(root);self.root=root;self.active=False;self.saved={}
        self.selectors=set()
        if hasattr(root,'breakline_list'):self.selectors.add(root.breakline_list)
        for owner,name in (('boundary_editor','boundary'),('forcing','boundary'),('turbines','selection')):
            widget=getattr(getattr(root,owner,None),name,None)
            if widget is not None:self.selectors.add(widget)

    def _prepare(self,w):
        if w in self.saved:return
        if isinstance(w,(QLineEdit,QTextEdit,QPlainTextEdit)):
            self.saved[w]=('text',w.isReadOnly());w.setReadOnly(True)
        elif isinstance(w,QAbstractItemView):
            self.saved[w]=('view',w.editTriggers());w.setEditTriggers(QAbstractItemView.NoEditTriggers)

    def set_locked(self,active):
        if active==self.active:return
        self.active=active
        if active:
            for w in self.root.findChildren(QWidget):self._prepare(w)
            QApplication.instance().installEventFilter(self)
        else:
            QApplication.instance().removeEventFilter(self)
            for w,(kind,value) in self.saved.items():
                if isValid(w):
                    if kind=='text':w.setReadOnly(value)
                    else:w.setEditTriggers(value)
            self.saved.clear()

    def eventFilter(self,w,e):
        if not self.active or not isinstance(w,QWidget) or not self.root.isAncestorOf(w):return False
        self._prepare(w)
        t=e.type()
        # Tabs, section toggles, scrolling and selecting rows are inspection.
        chain=[];p=w
        while p is not None and p is not self.root:
            chain.append(p);p=p.parentWidget()
        if any(p in self.selectors or isinstance(p,(QTabBar,QScrollBar)) for p in chain):return False
        for p in chain:
            if isinstance(p,QAbstractButton) and isinstance(p.parentWidget(),AccordionSection) and p is p.parentWidget().toggle:return False
        controls=(QAbstractButton,QAbstractSpinBox,QComboBox,QSlider,QDial)
        if any(isinstance(p,controls) or (isinstance(p,QGroupBox) and p.isCheckable()) for p in chain):
            if t in (QEvent.KeyPress,QEvent.KeyRelease) and (e.key() in (Qt.Key_Tab,Qt.Key_Backtab) or (e.modifiers() & Qt.ControlModifier and e.key() in (Qt.Key_C,Qt.Key_A))):return False
            return t in (QEvent.MouseButtonPress,QEvent.MouseButtonRelease,QEvent.MouseButtonDblClick,QEvent.Wheel,QEvent.KeyPress,QEvent.KeyRelease,QEvent.Shortcut,QEvent.ContextMenu,QEvent.InputMethod,QEvent.Drop)
        view=next((p for p in chain if isinstance(p,QAbstractItemView)),None)
        if view is not None:
            if t in (QEvent.ContextMenu,QEvent.Drop,QEvent.InputMethod):return True
            if t==QEvent.KeyPress:
                return e.key() not in (Qt.Key_Left,Qt.Key_Right,Qt.Key_Up,Qt.Key_Down,Qt.Key_PageUp,Qt.Key_PageDown,Qt.Key_Home,Qt.Key_End,Qt.Key_Tab,Qt.Key_Backtab,Qt.Key_Escape,Qt.Key_Control,Qt.Key_Shift) and not (e.key()==Qt.Key_C and e.modifiers() & Qt.ControlModifier)
            if t in (QEvent.MouseButtonPress,QEvent.MouseButtonDblClick,QEvent.MouseButtonRelease) :
                point=view.viewport().mapFrom(w,e.position().toPoint());index=view.indexAt(point)
                if index.isValid() and index.data(Qt.CheckStateRole) is not None:return True
        return False
