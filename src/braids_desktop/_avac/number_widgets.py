"""Locale-aware presentation with numeric model data, never localized storage."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDoubleSpinBox, QStyledItemDelegate, QTableWidgetItem
import pyqtgraph as pg
from .i18n import number, numeric_locale


def numeric_item(value):
    item = QTableWidgetItem()
    item.setData(Qt.EditRole, float(value))
    return item


class NumericDelegate(QStyledItemDelegate):
    def displayText(self, value, locale):
        try:return number(float(value))
        except (TypeError,ValueError):return str(value)

    def createEditor(self, parent, option, index):
        try:float(index.data(Qt.EditRole))
        except (TypeError,ValueError):return super().createEditor(parent,option,index)
        from ..presentation import DisplaySpinBox
        editor = DisplaySpinBox(parent)
        editor.setLocale(numeric_locale())
        editor.setRange(-1e15, 1e15)
        editor.setKeyboardTracking(False)
        return editor

    def setEditorData(self, editor, index):
        if isinstance(editor,QDoubleSpinBox):editor.setValue(float(index.data(Qt.EditRole)))
        else:super().setEditorData(editor,index)

    def setModelData(self, editor, model, index):
        if isinstance(editor,QDoubleSpinBox):
            editor.interpretText();model.setData(index,editor.value(),Qt.EditRole)
        else:super().setModelData(editor,model,index)


class NumericAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        texts = super().tickStrings(values, scale, spacing)
        if numeric_locale().decimalPoint() == ',':
            return [text.replace('.', ',') for text in texts]
        return texts
