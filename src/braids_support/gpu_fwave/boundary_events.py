"""Immutable forcing-knot scheduling for the isolated GPU schedule port.

Matches the frozen native next_boundary_knot 32-epsilon exclusion rule. A
sorted union is prepared once; trials/retries never advance a mutable cursor.
No hydraulic fields are moved and no production runner is changed here.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
import numpy as np

GROUPS = (('nseries', 'series_time'), ('nhydro_series', 'hydro_series_time'),
          ('nfraction_series', 'fraction_series_time'))
HUGE = np.finfo(np.float64).max


def tolerance(current):
    current = float(current)
    if not math.isfinite(current): raise ValueError('Finite event clock required')
    return 32*np.finfo(np.float64).eps*max(1., abs(current))


def active_knots(data):
    """Extract active native table rows, not unused allocated padding."""
    nb = np.asarray(data['mode']).size
    fraction = np.asarray(data['fraction_bed_input'])
    if fraction.ndim != 2 or fraction.shape[1] != nb:
        raise ValueError('Expected fraction-by-boundary input shape')
    expected = {'nseries': (nb,), 'nhydro_series': (nb,), 'nfraction_series': fraction.shape}
    result = []
    for count_name, time_name in GROUPS:
        counts = np.asarray(data[count_name])
        if counts.shape != expected[count_name] or counts.dtype.kind not in 'iu' or np.any(counts < 0):
            raise ValueError('Invalid event counts: '+count_name)
        if np.any(counts == 1):
            raise ValueError('Native schedules require zero or at least two rows: '+count_name)
        if time_name not in data:
            if np.any(counts): raise ValueError('Missing active event table: '+time_name)
            continue
        times = np.asarray(data[time_name])
        if times.ndim != counts.ndim+1 or times.shape[1:] != counts.shape or times.dtype.kind != 'f':
            raise ValueError('Invalid event table shape/type: '+time_name)
        if np.any(counts > times.shape[0]): raise ValueError('Event count exceeds storage')
        for index in np.ndindex(counts.shape):
            values = np.asarray(times[(slice(0, int(counts[index])), *index)], dtype=np.float64)
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(np.diff(values) <= 0):
                raise ValueError('Active knot times must be finite, nonnegative and strictly increasing')
            result.extend(values.tolist())
    return np.unique(np.asarray(result, dtype=np.float64))


@dataclass(frozen=True)
class TrialClock:
    current: float
    dt: float
    unclipped_hint: float
    boundary_knot: float
    limit: float
    scheduled_clip: bool

    def stage_times(self, retry_dt=None):
        dt = self.dt if retry_dt is None else float(retry_dt)
        if not math.isfinite(dt) or not 0 < dt <= self.dt or self.current+dt <= self.current:
            raise ValueError('Retry must be a positive advancing step within the original event limit')
        return np.asarray([self.current, self.current+.5*dt, self.current+dt, self.current+.5*dt])

    def accepted_time(self, retry_dt=None):
        # Same arithmetic as native advance; no clock advancement on rejection.
        return float(self.stage_times(retry_dt)[2])


@dataclass(frozen=True, init=False)
class BoundaryEvents:
    knots: np.ndarray
    identity: str

    def __init__(self, knots):
        values = np.asarray(knots, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Finite nonnegative one-dimensional knots required')
        values = np.unique(values)
        # Immutable bytes backing prevents callers turning WRITEABLE back on.
        object.__setattr__(self, 'knots', np.frombuffer(values.astype('<f8').tobytes(), dtype='<f8'))
        object.__setattr__(self, 'identity', hashlib.sha256(
            b'BRAIDS boundary event clock v1\0'+self.knots.tobytes()).hexdigest())

    @classmethod
    def from_tables(cls, data):
        return cls(active_knots(data))

    def next_after(self, current):
        current = float(current); eps = tolerance(current)
        index = int(np.searchsorted(self.knots, current+eps, side='right'))
        return float(self.knots[index]) if index < len(self.knots) else float(HUGE)

    def trial(self, current, hint, *, output, end):
        current, hint, output, end = map(float, (current, hint, output, end))
        if not np.isfinite([current, hint, output, end]).all() or current < 0 or hint <= 0:
            raise ValueError('Finite nonnegative clock and positive hint required')
        if min(output, end) <= current:
            raise ValueError('Process due outputs/end before proposing another trial')
        knot = self.next_after(current); limit = min(knot, output, end)
        dt = min(hint, limit-current)
        if not math.isfinite(dt) or dt <= 0 or current+dt <= current:
            raise ValueError('Nonadvancing event-clipped timestep')
        return TrialClock(current, dt, hint, knot, limit, dt < hint)

    def checkpoint_clock(self, current):
        current = float(current)
        if current < 0: raise ValueError('Negative checkpoint clock')
        return dict(time=current, event_identity=self.identity, next_boundary_knot=self.next_after(current))

    def validate_checkpoint_clock(self, value):
        if value.get('event_identity') != self.identity:
            raise ValueError('Checkpoint event times changed')
        current = float(value['time'])
        if current < 0 or value.get('next_boundary_knot') != self.next_after(current):
            raise ValueError('Checkpoint next event does not follow its accepted clock')
        # Only the event-clock identity is checked here. The integrated runner
        # must also validate forcing VALUES, physical state, source and method.
        return current
