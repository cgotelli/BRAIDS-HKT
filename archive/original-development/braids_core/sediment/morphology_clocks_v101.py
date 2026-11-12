"""Independent FLOW-style activation clocks and extensible, named timestep limits.

Scalar event administration is not a cell/face numerical loop. Morphological
mass increments are produced by the native/resident Q5 owner, never by this API.
"""
from dataclasses import dataclass, asdict
import math
from .bed_columns_v101 import digest


@dataclass(frozen=True)
class MorphologyClockMethodsV101:
    sediment_start_s: float = 0.
    bed_start_s: float = 0.
    composition_start_s: float = 0.
    sediment_enabled: bool = True
    bed_enabled: bool = True
    composition_enabled: bool = True
    update_fractions: tuple[bool, ...] = ()
    morfac_times_s: tuple[float, ...] = (0.,)
    morfac_values: tuple[float, ...] = (1.,)
    interpolation: str = 'linear'
    multi: bool = False

    def __post_init__(self):
        if self.multi is not False: raise ValueError('Multi is external MORmerge orchestration, not a single-run method')
        if any(type(x) is not bool for x in (self.sediment_enabled, self.bed_enabled, self.composition_enabled)):
            raise ValueError('Clock switches must be explicit booleans')
        if any(type(x) is not bool for x in self.update_fractions):
            raise ValueError('Per-fraction composition switches must be explicit booleans')
        values = (*self.morfac_times_s, *self.morfac_values, self.sediment_start_s,
                  self.bed_start_s, self.composition_start_s)
        if not values or not all(math.isfinite(x) and x >= 0 for x in values):
            raise ValueError('Finite nonnegative clock/schedule values required')
        if (not self.morfac_times_s or self.morfac_times_s[0] != 0.
                or len(self.morfac_times_s) != len(self.morfac_values)
                or any(b <= a for a, b in zip(self.morfac_times_s, self.morfac_times_s[1:]))
                or self.interpolation not in ('linear', 'step')):
            raise ValueError('Invalid MorFac schedule')
        if self.sediment_start_s > min(self.bed_start_s, self.composition_start_s):
            raise ValueError('FLOW sediment computation must start no later than bed/composition updates')
        if self.bed_enabled and self.composition_start_s > self.bed_start_s:
            raise ValueError('FLOW composition clock must start no later than bed updates')

    @property
    def fingerprint(self): return digest({'scope': 'braids-q6a-clocks-v101', **asdict(self)})

    @property
    def resolved_switches(self):
        # rdmor: when composition updating was requested but no class is
        # enabled, disable both composition and bed updating, not transport.
        none = self.composition_enabled and bool(self.update_fractions) and not any(self.update_fractions)
        return {'composition_enabled': self.composition_enabled and not none,
                'bed_enabled': self.bed_enabled and not none,
                'reason': 'no_fraction_updates' if none else 'saved_switches'}

    def factor(self, t):
        i = 0
        while i+1 < len(self.morfac_times_s) and self.morfac_times_s[i+1] <= t: i += 1
        if self.interpolation == 'step' or i+1 == len(self.morfac_times_s): return self.morfac_values[i]
        a, b = self.morfac_times_s[i:i+2]
        return self.morfac_values[i]+(t-a)/(b-a)*(self.morfac_values[i+1]-self.morfac_values[i])

    def next_event(self, t):
        events = (*self.morfac_times_s, self.sediment_start_s, self.bed_start_s, self.composition_start_s)
        return min((e for e in events if e > t), default=math.inf)


@dataclass(frozen=True)
class TimestepLimitV101:
    name: str
    dt_s: float
    producer: str

    def __post_init__(self):
        if not self.name or not self.producer or math.isnan(self.dt_s) or self.dt_s <= 0:
            raise ValueError('Limiter must name its implemented producer and positive bound')


def arbitrate_timestep_v101(requested_dt, limits, *, methods, time_s, end_time_s):
    if not all(math.isfinite(x) for x in (requested_dt, time_s, end_time_s)) or requested_dt <= 0 or not 0 <= time_s < end_time_s:
        raise ValueError('Invalid arbiter interval')
    rows = list(limits)
    if len({r.name for r in rows}) != len(rows): raise ValueError('Duplicate timestep limiter owner')
    # The API accepts future operator receipts but never invents their limits.
    rows += [TimestepLimitV101('requested', requested_dt, 'user'),
             TimestepLimitV101('end_time', end_time_s-time_s, 'hydraulic_clock')]
    event = methods.next_event(time_s)
    if math.isfinite(event): rows.append(TimestepLimitV101('schedule_event', event-time_s, methods.fingerprint))
    active = min(rows, key=lambda r: r.dt_s)
    return {'dt_s': active.dt_s, 'active_limiter': active.name,
            'implemented_limits': [asdict(r) for r in rows], 'clock_methods': methods.fingerprint}


class MorphologyClockOwnerV101:
    def __init__(self, methods, *, time_s=0.):
        if not math.isfinite(time_s) or time_s < 0: raise ValueError('Invalid accepted clock')
        self.methods = methods
        self.time_s = float(time_s); self.sediment_time_s = 0.; self.bed_time_s = 0.
        self.composition_time_s = 0.; self.morphological_time_s = 0.
        self.updates = 0; self.cursor = sum(t <= time_s for t in methods.morfac_times_s)-1
        self.last_limiter = 'initial'; self.pending = None; self.pending_sha256 = None

    def begin(self, arbiter):
        if self.pending is not None: raise ValueError('Clock trial is already open')
        if self.updates >= 12: raise ValueError('Q6a clock gate permits at most twelve accepted updates')
        if arbiter['clock_methods'] != self.methods.fingerprint: raise ValueError('Arbiter method identity differs')
        dt = arbiter['dt_s']; t = self.time_s; m = self.methods
        if not math.isfinite(dt) or dt <= 0 or t+dt > m.next_event(t)+1e-12:
            raise ValueError('Step crosses an unresolved clock event')
        factor = m.factor(t+.5*dt) # average of linear table; step tables sampled after event
        sed = m.sediment_enabled and t >= m.sediment_start_s
        resolved = m.resolved_switches
        comp = resolved['composition_enabled'] and t >= m.composition_start_s
        bed = resolved['bed_enabled'] and t >= m.bed_start_s
        self.pending = {'dt_s': dt, 'morfac': factor, 'sediment_enabled': sed,
            'composition_enabled': comp, 'bed_enabled': bed,
            'sediment_dt_s': dt if sed else 0., 'composition_dt_s': dt*factor if comp else 0.,
            'bed_dt_s': dt*factor if bed else 0.,
            'morphological_dt_s': dt*factor if t >= m.bed_start_s else 0.,
            'resolved_switches': resolved, 'active_limiter': arbiter['active_limiter']}
        self.pending_sha256 = digest(self.pending)
        return self.pending.copy()

    def validate_pending(self):
        if self.pending is None or digest(self.pending) != self.pending_sha256:
            raise ValueError('Clock trial was mutated or is missing')

    def commit(self):
        if self.pending is None: raise ValueError('No clock trial')
        self.validate_pending()
        p = self.pending
        self.time_s += p['dt_s']; self.sediment_time_s += p['sediment_dt_s']
        self.composition_time_s += p['composition_dt_s']; self.bed_time_s += p['bed_dt_s']
        self.morphological_time_s += p['morphological_dt_s']
        self.updates += 1; self.last_limiter = p['active_limiter']
        self.cursor = sum(t <= self.time_s for t in self.methods.morfac_times_s)-1
        self.pending = None; self.pending_sha256 = None

    def rollback(self): self.pending = None; self.pending_sha256 = None

    def checkpoint(self):
        if self.pending is not None: raise ValueError('Checkpoint requires accepted clock')
        d = {'schema': 1, 'methods': self.methods.fingerprint,
             **{k: getattr(self, k) for k in ('time_s', 'sediment_time_s', 'bed_time_s',
                'composition_time_s', 'morphological_time_s', 'updates', 'cursor', 'last_limiter')}}
        return {**d, 'sha256': digest(d)}

    def restore(self, d):
        raw = {k: v for k, v in d.items() if k != 'sha256'}
        if self.pending is not None or digest(raw) != d['sha256'] or d['methods'] != self.methods.fingerprint or d['schema'] != 1:
            raise ValueError('Clock restart identity differs')
        keys = ('time_s', 'sediment_time_s', 'bed_time_s', 'composition_time_s',
                'morphological_time_s', 'updates', 'cursor', 'last_limiter')
        c = MorphologyClockOwnerV101(self.methods)
        for k in keys: setattr(c, k, d[k])
        if (not all(math.isfinite(getattr(c, k)) and getattr(c, k) >= 0 for k in keys[:5])
                or type(c.updates) is not int or not 0 <= c.updates <= 12
                or c.cursor != sum(t <= c.time_s for t in self.methods.morfac_times_s)-1):
            raise ValueError('Clock restart domain differs')
        self.__dict__.update(c.__dict__)
