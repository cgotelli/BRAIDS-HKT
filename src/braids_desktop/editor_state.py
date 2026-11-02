"""Exact editable text and inactive choices, independent of run admission."""
from copy import deepcopy
from pathlib import Path
from PySide6.QtWidgets import QCheckBox,QComboBox,QLineEdit,QSpinBox,QDoubleSpinBox
from .presentation import NumericLineEdit
from .river_physics import set_choice


def capture_widgets(owner,root=None):
    result={}
    for name,w in vars(owner).items():
        if isinstance(w,QCheckBox):row=dict(kind='check',value=w.isChecked())
        elif isinstance(w,QComboBox):row=dict(kind='choice',value=w.currentData())
        elif isinstance(w,(QSpinBox,QDoubleSpinBox)):row=dict(kind='number',value=w.value())
        elif isinstance(w,QLineEdit):
            value=w.canonical_text() if isinstance(w,NumericLineEdit) else w.text();row=dict(kind='text',value=value)
            if root and value and Path(value).is_absolute() and Path(value).resolve().is_relative_to(Path(root).resolve()):
                row.update(value=Path(value).resolve().relative_to(Path(root).resolve()).as_posix(),relative=True)
        else:continue
        result[name]=row
    return result


def restore_widgets(owner,values,root=None):
    for name,row in values.items():
        w=getattr(owner,name);v=row['value']
        if row['kind']=='check':w.setChecked(v)
        elif row['kind']=='choice':
            if v is not None:set_choice(w,v)
        elif row['kind']=='number':w.setValue(v)
        else:w.setText(str(Path(root)/v) if row.get('relative') else v)


def capture(forms,root=None):
    return dict(schema=1,controls=capture_widgets(forms,root),adaptation={k:w.value() for k,w in forms.adaptation.items()},
        physics=forms.physics.editor_state(),boundaries=forms.boundary_editor.editor_state(),
        forcing=forms.forcing.editor_state(),schedules=forms.forcing.schedule_state(),bank=forms.physics.bank_survey.draft(),
        bank_controls=capture_widgets(forms.physics.bank_survey,root),bank_pending=deepcopy(forms.physics.bank_survey._pending),
        boundary_selection=forms.boundary_editor.boundary.currentData(),forcing_selection=forms.forcing.boundary.currentData(),
        grid=deepcopy(forms._grid_override),checkpoint_custom=forms._checkpoint_custom_interval,dry_stage=forms._dry_stage_m)


def restore(forms,value,root=None):
    if value.get('schema')!=1:raise ValueError('Current desktop editor snapshot required')
    from braids_core.river_boundary_draft import BoundaryDraft
    from braids_core.river_forcing import ForcingDraft
    from braids_core.river_schedule import ScheduleDraft
    forms._loading=True
    try:
        restore_widgets(forms,value['controls'],root)
        for k,v in value['adaptation'].items():forms.adaptation[k].setValue(v)
        forms.physics.restore_editor(value['physics'],enabled=value['physics']['widgets']['enabled'])
        forms.boundary_editor.draft=BoundaryDraft(value['boundaries']);forms.boundary_editor._shown=None;forms.boundary_editor._context=None
        forms.forcing.draft=ForcingDraft(value['forcing']);forms.forcing.schedules=ScheduleDraft(value['schedules'])
        forms.forcing._shown=None;forms.forcing._context=None
        forms.physics.bank_survey.restore(value['bank']);restore_widgets(forms.physics.bank_survey,value['bank_controls'],root)
        forms.physics.bank_survey._pending=deepcopy(value['bank_pending'])
        forms._grid_override=deepcopy(value['grid']);forms._checkpoint_custom_interval=value['checkpoint_custom'];forms._dry_stage_m=value['dry_stage']
    finally:forms._loading=False
    forms._changed()
    for editor,name in ((forms.boundary_editor,value['boundary_selection']),(forms.forcing,value['forcing_selection'])):
        if name is not None and editor.boundary.findData(name)>=0:set_choice(editor.boundary,name)
