"""Independent BRAIDS presentation preferences; never scientific parameters."""
from PySide6.QtCore import QSettings


def settings():
    return QSettings(QSettings.defaultFormat(),QSettings.UserScope,"BRAIDS", "BRAIDS Desktop")


def appearance():
    value = settings().value("interface/appearance", "light")
    return value if value in ("dark", "light") else "light"


def geometry_color():
    from PySide6.QtGui import QColor
    value=str(settings().value('interface/geometry_color','#df3038'))
    return value if QColor(value).isValid() else '#df3038'


def mesh_color():
    from PySide6.QtGui import QColor
    value=str(settings().value('interface/mesh_color','#00ff00'))
    return value if QColor(value).isValid() else '#00ff00'


def measurement_color():
    from PySide6.QtGui import QColor
    value=str(settings().value('interface/measurement_color','#ffff00'))
    return value if QColor(value).isValid() else '#ffff00'


def snap_enabled():return settings().value('map/snap',True,type=bool)
def adaptive_mesh():return settings().value('map/adaptive_mesh',True,type=bool)

def smooth_level():
    """Display-only smoothing strength, with bounded, persistent settings."""
    return max(0,min(100,settings().value('map/smooth_level',50,type=int)))
