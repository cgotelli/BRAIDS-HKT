"""Short numeric presentation without rounding stored scientific inputs."""
import math
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QDoubleSpinBox,QLineEdit
from ._avac.i18n import numeric_locale


def display_number(value):
    value=float(value)
    if value and abs(value)<.001:
        mantissa,exponent=f'{value:.3e}'.split('e')
        text=mantissa.rstrip('0').rstrip('.')+'e'+str(int(exponent))
    else:
        text=f'{value:.3f}'.rstrip('0').rstrip('.')
    if text=='-0':text='0'
    return text.replace('.',numeric_locale().decimalPoint())


def display_values(value):
    """Readable keys/numbers for the resolved-method viewer, never for storage."""
    if isinstance(value,dict):return {k.replace('_',' '):display_values(v) for k,v in value.items()}
    if isinstance(value,list):return [display_values(v) for v in value]
    if isinstance(value,float):return display_number(value)
    return value


class DisplaySpinBox(QDoubleSpinBox):
    """Keep native precision; show at most three decimal places."""
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setDecimals(12);self.setKeyboardTracking(False)

    def textFromValue(self,value):return display_number(value)

    def valueFromText(self,text):
        cleaned=text.strip()
        if cleaned==self.textFromValue(self.value()):return self.value()
        value,valid=self.locale().toDouble(cleaned)
        return value if valid else self.value()

    def validate(self,text,pos):
        if text.strip() in ('','-','+','. ','.',','):return QValidator.Intermediate,text,pos
        value,valid=self.locale().toDouble(text.strip())
        if valid and math.isfinite(value):return QValidator.Acceptable,text,pos
        if text.lower().endswith(('e','e-','e+')):return QValidator.Intermediate,text,pos
        return QValidator.Invalid,text,pos


class NumericLineEdit(QLineEdit):
    """A localized numeric view of an unchanged canonical boundary draft."""
    def __init__(self,text='',parent=None):
        super().__init__(parent);self._canonical='';self.setText(text)
        self.textEdited.connect(self._edited)

    def setText(self,text):
        self._canonical=str(text)
        try:shown=display_number(float(self._canonical))
        except ValueError:shown=self._canonical
        super().setText(shown)

    def _edited(self,text):self._canonical=text.replace(numeric_locale().decimalPoint(),'.')

    def canonical_text(self):return self._canonical
