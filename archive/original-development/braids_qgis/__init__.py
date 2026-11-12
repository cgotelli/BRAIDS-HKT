"""BRAIDS hydrodynamic and morphology plugin for QGIS."""

from pathlib import Path
import sys


# A release ZIP carries ``braids_core`` inside the plugin directory so QGIS can
# import it without a separate Python installation.  A development checkout
# keeps the same package beside ``braids_qgis``.  Establish the matching import
# root before QGIS imports plugin.py and the dock modules.
_PLUGIN_ROOT = Path(__file__).resolve().parent
for _candidate in (_PLUGIN_ROOT / "river_runtime", _PLUGIN_ROOT, _PLUGIN_ROOT.parent):
    if (_candidate / "braids_core" / "__init__.py").is_file():
        _loaded = sys.modules.get("braids_core")
        _loaded_path = None if _loaded is None or not getattr(_loaded,"__file__",None) else Path(_loaded.__file__).resolve().parent
        _expected_path = (_candidate / "braids_core").resolve()
        if _loaded is not None and (_loaded_path != _expected_path or getattr(_loaded,"API_VERSION",None) != 22):
            for _name in tuple(sys.modules):
                if _name == "braids_core" or _name.startswith("braids_core."):
                    del sys.modules[_name]
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break


def classFactory(iface):  # noqa: N802 - QGIS plugin API spelling
    from .plugin import BraidsQgisPlugin
    return BraidsQgisPlugin(iface)
