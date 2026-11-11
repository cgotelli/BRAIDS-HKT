"""Display-only colour ramps and raster style editor."""
from .i18n import tr, trf, tr_list, number, parse_number
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox)

PALETTES = {
    "Terrain": ["#354e49", "#6b7e64", "#a29e7a", "#b9b49d", "#e4e8e3"],
    "Ocean to fire": ["#1e5fa8", "#16b9cf", "#83dc97", "#fad050", "#f05e36"],
    "Viridis": ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
    "Blue–red": ["#2767b5", "#d8e1e4", "#e7763c"],
    "Grayscale": ["#20242a", "#f5f7fa"],
}


class RasterStyleDialog(QDialog):
    def __init__(self, view):
        super().__init__(view)
        self.view = view
        self.setWindowTitle(tr("Layer appearance"))
        form = QFormLayout(self)
        self.layer = QComboBox()
        for label,key in view.appearance_layers():self.layer.addItem(tr(label),key)
        form.addRow(tr("Layer"), self.layer)
        self.palette = QComboBox()
        form.addRow(tr("Palette"), self.palette)
        self.automatic = QCheckBox(tr("Use data range"))
        range_row=QHBoxLayout();range_row.addWidget(self.automatic)
        self.visible_range=QPushButton(tr('Use visible range'));range_row.addWidget(self.visible_range)
        form.addRow(range_row);self.visible_range.clicked.connect(self.use_visible_range)
        from ..presentation import DisplaySpinBox
        self.low, self.high = DisplaySpinBox(), DisplaySpinBox()
        for spin in (self.low, self.high):
            spin.setRange(-1e15, 1e15)
        form.addRow(tr("Minimum"), self.low)
        form.addRow(tr("Maximum"), self.high)
        self.zero = QCheckBox(tr("Exactly zero is transparent"))
        form.addRow(tr(self.zero))
        self.legend = QCheckBox(tr("Show this layer's legend"))
        form.addRow(tr(self.legend))
        self.opacity = QSpinBox()
        self.opacity.setRange(0, 100)
        self.opacity.setSuffix(tr(" %"))
        form.addRow(tr("Layer opacity"), self.opacity)
        self.error = QLabel()
        self.error.setWordWrap(True)
        form.addRow(tr(self.error))
        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)
        buttons.rejected.connect(self.reject)
        form.addRow(tr(buttons))
        self.automatic.toggled.connect(lambda checked: (self.low.setEnabled(not checked), self.high.setEnabled(not checked)))
        self.layer.currentIndexChanged.connect(self.load)
        panel=getattr(view,'results_panel',None)
        if panel is not None and panel.store is not None and panel._results_active:
            self.layer.setCurrentIndex(self.layer.findData('result:'+panel.variable.currentData()))
        self.load()

    def load(self):
        key = self.layer.currentData()
        style = self.view.styles.get(key, {})
        self.palette.clear()
        for palette in self.view.appearance_palettes(key):self.palette.addItem(tr(palette),palette)
        self.palette.setCurrentIndex(self.palette.findData(style.get("palette", self.view.default_palette(key))))
        low, high = self.view.range_for(key)
        limits = style.get("limits")
        self.automatic.setChecked(limits is None)
        self.low.setValue(limits[0] if limits else low)
        self.high.setValue(limits[1] if limits else high)
        self.low.setEnabled(limits is not None)
        self.high.setEnabled(limits is not None)
        self.zero.setChecked(style.get("zero_transparent", self.view.default_zero_transparent(key)))
        # This is a rendering-only choice.  In particular, it must remain
        # adjustable while live results are being viewed; it never modifies a
        # model setting, saved result, solver state or output field.
        self.zero.setEnabled(True)
        self.zero.setToolTip(tr('Display only: this does not change stored results or the running simulation.'))
        self.legend.setChecked(style.get("legend_visible", True))
        self.opacity.setValue(round(style.get("opacity", 1.)*100))

    def use_visible_range(self):
        try:
            low,high=self.view.visible_range_for(self.layer.currentData())
            self.automatic.setChecked(False);self.low.setValue(low);self.high.setValue(high)
            self.error.setText(tr('Visible range selected. Press Apply to keep it when zooming.'))
        except ValueError as exc:self.error.setText(str(exc))

    def apply(self):
        if not self.automatic.isChecked() and self.high.value() <= self.low.value():
            self.error.setText(tr("Maximum must be greater than minimum."))
            return
        self.view.styles[self.layer.currentData()] = dict(palette=self.palette.currentData(),
            limits=None if self.automatic.isChecked() else [self.low.value(), self.high.value()],
            zero_transparent=self.zero.isChecked(), opacity=self.opacity.value()/100, legend_visible=self.legend.isChecked())
        self.view.refresh_appearance(self.layer.currentData())
        self.view.styles_changed.emit()
        self.error.setText(tr("Applied. Scientific raster values are unchanged."))
