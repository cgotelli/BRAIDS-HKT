"""Presentation preferences, following the AVAC Desktop settings workflow."""
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QDialog,QVBoxLayout,QFormLayout,QComboBox,QPushButton,QColorDialog,QDialogButtonBox,QLabel,QCheckBox,QSpinBox
from ._avac.preferences import appearance,geometry_color,mesh_color,measurement_color,snap_enabled,adaptive_mesh,settings
from ._avac.i18n import LANGUAGES,saved_language,saved_separator,save_interface,tr


class SettingsDialog(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent);self.setWindowTitle(tr('Settings'));self.resize(500,330)
        root=QVBoxLayout(self);form=QFormLayout();root.addLayout(form)
        self.language=QComboBox()
        for code,title in LANGUAGES.items():self.language.addItem(title,code)
        self.language.setCurrentIndex(self.language.findData(saved_language()))
        self.theme=QComboBox()
        for name,key in [('Light','light'),('Dark','dark')]:self.theme.addItem(tr(name),key)
        self.theme.setCurrentIndex(self.theme.findData(appearance()))
        self.separator=QComboBox();self.separator.addItem(tr('Point (1.25)'),'.');self.separator.addItem(tr('Comma (1,25)'),',')
        self.separator.setCurrentIndex(self.separator.findData(saved_separator()))
        self.background=QComboBox()
        for name,key in [('None','none'),('OpenStreetMap','osm'),('Satellite','satellite')]:self.background.addItem(tr(name),key)
        self.background.setCurrentIndex(max(0,self.background.findData(settings().value('map/background','none'))))
        self.color=geometry_color();self.color_button=QPushButton();self._show_color();self.color_button.clicked.connect(self._choose_color)
        self.mesh_color=mesh_color();self.mesh_color_button=QPushButton(self.mesh_color)
        self.mesh_color_button.setStyleSheet('background:'+self.mesh_color+';color:black');self.mesh_color_button.clicked.connect(self._choose_mesh_color)
        for label,widget in [('Language',self.language),('Appearance',self.theme),('Decimal separator',self.separator),('Map background',self.background),('Polygon and line color',self.color_button)]:form.addRow(tr(label),widget)
        form.addRow(tr('Mesh color'),self.mesh_color_button)
        self.measurement_color=measurement_color();self.measurement_color_button=QPushButton(self.measurement_color)
        self.measurement_color_button.setStyleSheet('background:'+self.measurement_color+';color:black')
        def choose():
            color=QColorDialog.getColor(QColor(self.measurement_color),self,tr('Probe and profile color'))
            if color.isValid():
                self.measurement_color=color.name();self.measurement_color_button.setText(self.measurement_color)
                self.measurement_color_button.setStyleSheet('background:'+self.measurement_color+';color:black')
        self.measurement_color_button.clicked.connect(choose);form.addRow(tr('Probe and profile color'),self.measurement_color_button)
        self.snap=QCheckBox(tr('Snap while editing'));self.snap.setChecked(snap_enabled());form.addRow(self.snap)
        self.adaptive_mesh=QCheckBox(tr('Animate adaptive mesh'));self.adaptive_mesh.setChecked(adaptive_mesh());form.addRow(self.adaptive_mesh)
        self.particle_density=QSpinBox();self.particle_density.setRange(50,10000);self.particle_density.setSingleStep(100)
        self.particle_density.setValue(settings().value('map/particle_density',1000,type=int));form.addRow('Flow particle count',self.particle_density)
        note=QLabel(tr('Language changes apply after restarting BRAIDS. Other presentation choices apply immediately.'))
        note.setWordWrap(True);root.addWidget(note)
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel);buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);root.addWidget(buttons)

    def _show_color(self):
        self.color_button.setText(self.color);self.color_button.setStyleSheet('background:'+self.color+';color:white')

    def _choose_color(self):
        color=QColorDialog.getColor(QColor(self.color),self,tr('Polygon and line color'))
        if color.isValid():self.color=color.name();self._show_color()

    def save_preferences(self):
        save_interface(self.language.currentData(),self.separator.currentData())
        store=settings();store.setValue('interface/appearance',self.theme.currentData())
        store.setValue('interface/geometry_color',self.color);store.setValue('map/background',self.background.currentData());store.sync()
        store.setValue('interface/mesh_color',self.mesh_color);store.sync()
        store.setValue('interface/measurement_color',self.measurement_color)
        store.setValue('map/snap',self.snap.isChecked());store.setValue('map/adaptive_mesh',self.adaptive_mesh.isChecked());store.sync()
        store.setValue('map/particle_density',self.particle_density.value());store.sync()

    def _choose_mesh_color(self):
        color=QColorDialog.getColor(QColor(self.mesh_color),self,tr('Mesh color'))
        if color.isValid():
            self.mesh_color=color.name();self.mesh_color_button.setText(self.mesh_color)
            self.mesh_color_button.setStyleSheet('background:'+self.mesh_color+';color:black')
