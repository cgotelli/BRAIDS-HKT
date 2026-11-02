"""Readable project tree plus a strict, explicitly editable JSON source view."""
import json,re
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase,QSyntaxHighlighter,QTextCharFormat,QColor
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QTabWidget,QTreeWidget,QTreeWidgetItem,QPlainTextEdit,QLineEdit,QPushButton,QLabel,QHeaderView


class JsonHighlighter(QSyntaxHighlighter):
    def highlightBlock(self,text):
        for match in re.finditer(r'"(?:[^"\\]|\\.)*"|\b(?:true|false|null)\b|(?<!\w)-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?',text):
            token=match.group();style=QTextCharFormat()
            key=token.startswith('"') and text[match.end():].lstrip().startswith(':')
            style.setForeground(QColor('#287bb5' if key else '#a45b18' if token.startswith('"') else '#8b5dc2'))
            if key:style.setFontWeight(600)
            self.setFormat(match.start(),len(token),style)


class JsonEditor(QWidget):
    def __init__(self,text,parent=None):
        super().__init__(parent);layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        self.tabs=QTabWidget();layout.addWidget(self.tabs)
        page=QWidget();box=QVBoxLayout(page);bar=QHBoxLayout()
        self.search=QLineEdit();self.search.setPlaceholderText('Find a setting, value or JSON path…');bar.addWidget(self.search)
        expand=QPushButton('Expand all');collapse=QPushButton('Collapse all');bar.addWidget(expand);bar.addWidget(collapse);box.addLayout(bar)
        self.tree=QTreeWidget();self.tree.setHeaderLabels(['Setting','Value','Type']);self.tree.setAlternatingRowColors(True)
        self.tree.header().setSectionResizeMode(0,QHeaderView.ResizeToContents);self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(1,QHeaderView.Stretch);box.addWidget(self.tree)
        self.path=QLineEdit();self.path.setReadOnly(True);self.path.setPlaceholderText('Select a setting to see its JSON path');box.addWidget(self.path)
        self.tabs.addTab(page,'Browse settings')
        raw=QWidget();r=QVBoxLayout(raw);row=QHBoxLayout();format_button=QPushButton('Format JSON');row.addWidget(format_button)
        self.raw_search=QLineEdit();self.raw_search.setPlaceholderText('Find in JSON (Enter for next)');row.addWidget(self.raw_search);r.addLayout(row)
        self.source=QPlainTextEdit();self.source.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.source.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont));self.source.setTabStopDistance(28)
        self.source.setPlainText(text);self.highlighter=JsonHighlighter(self.source.document());r.addWidget(self.source)
        self.position=QLabel();r.addWidget(self.position);self.tabs.addTab(raw,'Edit JSON')
        self.message=QLabel('Browse the hierarchy; use Edit JSON to change values.');self.message.setWordWrap(True);layout.addWidget(self.message)
        self.tree.currentItemChanged.connect(lambda item,_:self.path.setText(item.data(0,Qt.UserRole) if item else ''))
        self.search.textChanged.connect(self.filter);expand.clicked.connect(self.tree.expandAll);collapse.clicked.connect(self.tree.collapseAll)
        format_button.clicked.connect(self.format);self.tabs.currentChanged.connect(lambda i:self.refresh_tree() if i==0 else None)
        self.raw_search.returnPressed.connect(self.find_next);self.source.cursorPositionChanged.connect(self.show_position)
        self.refresh_tree();self.show_position()

    def value(self):
        from braids_core.river_folder import read_json
        value=read_json(self.source.toPlainText())
        if not isinstance(value,dict):raise ValueError('The project must be a JSON object.')
        return value

    def refresh_tree(self):
        try:value=self.value()
        except Exception as exc:self.message.setText('JSON error: '+str(exc));return False
        self.tree.clear()
        def add(parent,key,value,path):
            kind='object' if isinstance(value,dict) else 'array' if isinstance(value,list) else 'null' if value is None else 'boolean' if isinstance(value,bool) else 'number' if isinstance(value,(int,float)) else 'string'
            text=f'{len(value)} entries' if isinstance(value,(dict,list)) else json.dumps(value,ensure_ascii=False) if not isinstance(value,str) else value
            item=QTreeWidgetItem(parent,[str(key),text,kind]);item.setData(0,Qt.UserRole,path);item.setToolTip(1,text)
            if isinstance(value,dict):
                for k,v in value.items():add(item,k,v,path+'['+json.dumps(k,ensure_ascii=False)+']')
            elif isinstance(value,list):
                for i,v in enumerate(value):add(item,f'[{i}]',v,f'{path}[{i}]')
        for key,v in value.items():add(self.tree,key,v,'$['+json.dumps(key,ensure_ascii=False)+']')
        self.tree.expandToDepth(1);self.filter(self.search.text());self.message.setText('Browse the hierarchy; use Edit JSON to change values.');return True

    def filter(self,text):
        needle=text.casefold().strip()
        def visit(item,inherited=False):
            own=not needle or any(needle in item.text(i).casefold() for i in range(3)) or needle in item.data(0,Qt.UserRole).casefold()
            child_match=False
            for i in range(item.childCount()):child_match=visit(item.child(i),inherited or own) or child_match
            visible=inherited or own or child_match;item.setHidden(not visible)
            if needle and child_match:item.setExpanded(True)
            return visible
        for i in range(self.tree.topLevelItemCount()):visit(self.tree.topLevelItem(i))

    def format(self):
        try:self.source.setPlainText(json.dumps(self.value(),indent=2,ensure_ascii=False,allow_nan=False)+'\n');self.message.setText('JSON is valid and formatted.')
        except Exception as exc:self.message.setText('JSON error: '+str(exc))

    def show_position(self):
        c=self.source.textCursor();self.position.setText(f'Line {c.blockNumber()+1}, column {c.positionInBlock()+1}')

    def find_next(self):
        if not self.source.find(self.raw_search.text()):
            cursor=self.source.textCursor();cursor.setPosition(0);self.source.setTextCursor(cursor);self.source.find(self.raw_search.text())
