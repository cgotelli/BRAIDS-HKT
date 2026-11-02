"""Small shared polygon-override editor for roughness and erodible depth."""
from copy import deepcopy
from PySide6.QtCore import Qt,Signal
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QLabel,QTableWidget,QTableWidgetItem,QHeaderView,QPushButton


class MaterialZones(QWidget):
    changed=Signal()
    action=Signal(str)
    def __init__(self,kind,parent=None):
        super().__init__(parent);self.kind=kind;layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        self.label=QLabel();self.label.setWordWrap(True);layout.addWidget(self.label)
        self.table=QTableWidget(0,2);self.table.setHorizontalHeaderLabels(['Polygon','Value'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch);self.table.setMaximumHeight(150);layout.addWidget(self.table)
        row=QHBoxLayout();layout.addLayout(row)
        for text,key in [('Add polygon','add'),('Edit shape','edit'),('Remove','remove')]:
            button=QPushButton(text);button.clicked.connect(lambda _,k=key:self.action.emit(k+'_zone:'+self.kind));row.addWidget(button)
        self.table.itemChanged.connect(lambda *_:self.changed.emit())
    def selected(self):
        row=self.table.currentRow()
        if row<0:raise ValueError('Select a material polygon first')
        return self.table.item(row,0).text()
    def load(self,rows):
        self.table.blockSignals(True);self.table.setRowCount(len(rows))
        for i,row in enumerate(rows):
            name=QTableWidgetItem(row['id']);name.setFlags(name.flags()&~Qt.ItemIsEditable);name.setData(Qt.UserRole,deepcopy(row['points_xy']))
            self.table.setItem(i,0,name);self.table.setItem(i,1,QTableWidgetItem(str(row['value'])))
        self.table.blockSignals(False)
    def values(self):
        return [dict(id=self.table.item(i,0).text(),points_xy=deepcopy(self.table.item(i,0).data(Qt.UserRole)),
                     value=float(self.table.item(i,1).text().replace(',','.'))) for i in range(self.table.rowCount())]
