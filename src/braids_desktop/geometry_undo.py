"""Ten reversible geometry edits; model parameters outside geometry stay put."""
from collections import deque
from copy import deepcopy


class GeometryUndo:
    def __init__(self,window):self.window=window;self.states=deque(maxlen=10);self.restoring=False

    def capture(self,*,guides=None):
        w=self.window;doc=w._collect(validate=False)
        return dict(has_domain=w.has_domain,
            settings={key:deepcopy(doc['settings'][key]) for key in ('domain','grid','assignments','hydraulics','sediment_boundaries','breaklines','solids')},
            extra={key:deepcopy(doc.get(key)) for key in ('boundary_states','sediment_schedules','material_zones','turbines','bank_survey')},
            guides=deepcopy(w.map.sketch_lines if guides is None else guides),measurements=w.results.measurement_state())

    def remember(self,state=None):
        if self.restoring:return
        self.states.append(state if state is not None else self.capture());self._action()

    def clear(self):self.states.clear();self._action()

    def _action(self):
        action=getattr(self.window,'undo_geometry_action',None)
        if action:action.setEnabled(bool(self.states))

    def undo(self):
        w=self.window
        if not self.states:return
        if w.runner.is_running or w._tasks:
            w.status.setText('Stop the current operation before undoing geometry edits.');return
        state=self.states[-1];current=w._collect(validate=False)
        current['settings'].update(deepcopy(state['settings']));current.update(deepcopy(state['extra']))
        self.restoring=True
        try:
            w.map.set_tool('pan');w.has_domain=state['has_domain'];w._load_document(current)
            w.map.set_sketch_lines(state['guides']);w._last_geometry_guides=deepcopy(state['guides'])
            w.results.set_measurement_state(state['measurements']);w._changed()
            self.states.pop();w.status.setText(f'Geometry edit undone ({len(self.states)} earlier edits available).')
        finally:self.restoring=False;self._action()
