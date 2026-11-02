"""Interface translations and numbers, independent of scientific file formats.

Language is selected at startup. A change in Settings applies on the next
launch; numeric presentation can change immediately without rebuilding a run.
Never call Python/C setlocale: native controls and exports retain their formats.
"""
from __future__ import annotations

import json
import numbers
from pathlib import Path
import string

from PySide6.QtCore import QLocale, QTranslator, QLibraryInfo
from PySide6.QtWidgets import QWidget

LANGUAGES = {'en': 'English', 'fr': 'Français', 'es': 'Español', 'it': 'Italiano', 'de': 'Deutsch'}
_language = 'en'
_separator = '.'
_catalog = {}
_qt_translator = None


def saved_language():
    from .preferences import settings
    value = settings().value('interface/language', 'en')
    return value if value in LANGUAGES else 'en'


def saved_separator():
    from .preferences import settings
    value = settings().value('interface/decimal_separator', '.')
    return value if value in ('.', ',') else '.'


def language():
    return _language


def configure(app, code=None, separator=None):
    global _language, _catalog, _qt_translator
    code = saved_language() if code is None else code
    if code not in LANGUAGES:
        raise ValueError('Unsupported interface language')
    _language = code
    path = Path(__file__).with_name('resources')/'translations'/f'{code}.json'
    _catalog = json.loads(path.read_text(encoding='utf-8')) if code != 'en' else {}
    from ..translations import LABELS
    _catalog.update(LABELS.get(code,{}))
    if _qt_translator is not None:
        app.removeTranslator(_qt_translator)
    _qt_translator = QTranslator(app)
    # Qt's standard buttons/file dialogs use its distributed translations.
    translations = Path(__file__).with_name('resources')/'translations'
    if not _qt_translator.load(f'qtbase_{code}', str(translations)):
        _qt_translator.load(f'qtbase_{code}', QLibraryInfo.path(QLibraryInfo.TranslationsPath))
    app.installTranslator(_qt_translator)
    apply_separator(app, saved_separator() if separator is None else separator)


def numeric_locale():
    locale = QLocale('de_DE' if _separator == ',' else 'en_US')
    locale.setNumberOptions(QLocale.OmitGroupSeparator | QLocale.RejectGroupSeparator)
    return locale


def apply_separator(app, separator):
    global _separator
    if separator not in ('.', ','):
        raise ValueError('Unsupported decimal separator')
    _separator = separator
    locale = numeric_locale()
    QLocale.setDefault(locale)
    for widget in app.allWidgets():
        widget.setLocale(locale)
        QWidget.update(widget)


def save_interface(language_code, separator):
    if language_code not in LANGUAGES or separator not in ('.', ','):
        raise ValueError('Invalid interface preference')
    from .preferences import settings
    store = settings()
    store.setValue('interface/language', language_code)
    store.setValue('interface/decimal_separator', separator)
    store.sync()


def tr(text):
    """Translate display text only; identifiers and user data stay unchanged."""
    return _catalog.get(text, text) if isinstance(text, str) else text


def tr_list(texts):
    return [tr(text) for text in texts]


def number(value, spec='g'):
    # Grouping is intentionally omitted to avoid ambiguous input such as 1,234.
    from ..presentation import display_number
    return display_number(value)


def parse_number(text):
    value, valid = numeric_locale().toDouble(text.strip())
    if not valid:
        raise ValueError(tr('Enter a number using the selected decimal separator.'))
    return value


class _Formatter(string.Formatter):
    def format_field(self, value, spec):
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            return number(value, spec)
        return super().format_field(value, spec)


def trf(template, *values):
    return _Formatter().format(tr(template), *values)
