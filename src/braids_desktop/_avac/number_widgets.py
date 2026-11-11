"""Locale-aware presentation with numeric model data, never localized storage."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit, QStyledItemDelegate, QTableWidgetItem, QTableView
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
        # Table cells are plain numeric text: arrow steppers steal width and
        # clip digits in narrow configuration tables.
        editor = QLineEdit(parent)
        editor.setMinimumHeight(max(28,editor.minimumSizeHint().height()+4))
        # Table views otherwise retain the compact display-row height, which
        # clips the plain-text editor on high-DPI displays.
        view=parent.parent() if parent is not None else None
        if isinstance(view,QTableView):view.setRowHeight(index.row(),max(32,view.rowHeight(index.row())))
        editor.setText(number(float(index.data(Qt.EditRole))))
        return editor

    def setEditorData(self, editor, index):
        if isinstance(editor,QLineEdit):editor.setText(number(float(index.data(Qt.EditRole))))
        else:super().setEditorData(editor,index)

    def setModelData(self, editor, model, index):
        if isinstance(editor,QLineEdit):
            try:model.setData(index,float(editor.text().replace(',', '.')),Qt.EditRole)
            except ValueError:return
        else:super().setModelData(editor,model,index)


class NumericAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        texts = super().tickStrings(values, scale, spacing)
        if numeric_locale().decimalPoint() == ',':
            return [text.replace('.', ',') for text in texts]
        return texts
