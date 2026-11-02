# Standalone PySide6 port of the maintained BRAIDS configuration editor.
"""Small dependency-free accordion section used by the BRAIDS dock."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QToolButton, QVBoxLayout, QWidget


class AccordionSection(QWidget):
    def __init__(self, title: str, *, expanded: bool = False, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)
        self.toggle = QToolButton(text=title, checkable=True, checked=expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.toggle.setStyleSheet(
            "QToolButton { border: none; font-size: 10pt; font-weight: 600; padding: 6px 3px; }")
        self.content = QFrame(); self.content.setFrameShape(QFrame.Shape.StyledPanel)
        self.body_layout = QVBoxLayout(self.content); self.body_layout.setContentsMargins(9, 7, 9, 9)
        self.content.setVisible(expanded)
        self.toggle.toggled.connect(self._set_expanded)
        layout.addWidget(self.toggle); layout.addWidget(self.content)

    def _set_expanded(self, expanded: bool):
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.content.setVisible(expanded)

    def isExpanded(self) -> bool:
        return self.toggle.isChecked()

    def setExpanded(self, expanded: bool) -> None:
        self.toggle.setChecked(expanded)


def initialize_sections(page):
    """Open the first visible top-level section once; preserve later user choices."""
    sections=[]
    for section in page.findChildren(AccordionSection):
        parent=section.parentWidget();hidden=section.isHidden();nested=False
        while parent is not None and parent is not page:
            hidden=hidden or parent.isHidden();nested=nested or isinstance(parent,AccordionSection)
            parent=parent.parentWidget()
        if not hidden and not nested:sections.append(section)
    for i,section in enumerate(sections):section.setExpanded(i==0)

