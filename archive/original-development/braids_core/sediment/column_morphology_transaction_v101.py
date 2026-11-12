"""One Q5a/Q6a accepted transaction for bed inventory, geometry and clocks.

The input is an already-owned transport/exchange rate, not another transport
model. Bed-only or composition-only FLOW switches are explicit geometric
diagnostic paths, and are not labelled conserved coupled morphodynamics.
"""
import math
import numpy as np
from .bed_columns_v101 import host, digest, array_sha
from .morphology_clocks_v101 import TimestepLimitV101, arbitrate_timestep_v101


class ColumnMorphologyTransactionV101:
    def __init__(self, columns, clocks):
        if columns.time_s != clocks.time_s: raise ValueError('Column and hydraulic clocks differ')
        if clocks.methods.update_fractions and len(clocks.methods.update_fractions) != len(columns.runtime.methods.fraction_ids):
            raise ValueError('Per-fraction switches must match the saved grain inventory')
        self.columns, self.clocks = columns, clocks
        self.geometry_bed = columns.bed.copy()
        self.geometric_only_ledger_m = columns.xp.zeros(columns.runtime.nc)
        self.pending = None; self.pending_sha256 = None

    def begin(self, *, bed_rate, water_mass, water_rate, depth, requested_dt,
              end_time_s, limits=(), max_bed_change_m=0., max_concentration_kg_m3=0.,
              entrainment_fraction=0., cancelled=False):
        if self.pending is not None: raise ValueError('Composed morphology trial is open')
        if cancelled: raise InterruptedError('Morphology cancelled before preparation')
        c, clock = self.columns, self.clocks
        if c.time_s != clock.time_s: raise ValueError('Accepted column/clock mismatch')
        # Conservative upper schedule envelope protects finite stock even for
        # linearly increasing MorFac. An event is never crossed in one update.
        endpoint = min(clock.time_s+requested_dt, clock.methods.next_event(clock.time_s), end_time_s)
        mf = max(clock.methods.factor(clock.time_s), clock.methods.factor(endpoint))
        sed = clock.methods.sediment_enabled and clock.time_s >= clock.methods.sediment_start_s
        resolved = clock.methods.resolved_switches
        comp = resolved['composition_enabled'] and clock.time_s >= clock.methods.composition_start_s
        bed = resolved['bed_enabled'] and clock.time_s >= clock.methods.bed_start_s
        r = bed_rate if sed else c.xp.zeros_like(bed_rate)
        wr = water_rate if sed else c.xp.zeros_like(water_rate)
        limited_rate = c.runtime.integrate_rate(r, 1., clock.methods.update_fractions) if comp else r
        numbers = c.runtime.limits(c.available, water_mass, depth, limited_rate, wr, morfac=mf,
            max_bed_change_m=max_bed_change_m, max_concentration_kg_m3=max_concentration_kg_m3,
            entrainment_fraction=entrainment_fraction)
        bounds = host(numbers.min(axis=1))
        names = ('finite_active_stock', 'concentration', 'entrainment', 'bed_change')
        rows = list(limits)
        for name, bound in zip(names, bounds):
            if (name == 'finite_active_stock' and not comp) or (name == 'bed_change' and not bed): continue
            if bound <= 0: raise ValueError('A zero admissible timestep requires transport limiting: '+name)
            if math.isfinite(bound): rows.append(TimestepLimitV101(name, float(bound), c.runtime.identity))
        arbiter = arbitrate_timestep_v101(requested_dt, rows, methods=clock.methods,
                                        time_s=clock.time_s, end_time_s=end_time_s)
        p = clock.begin(arbiter)
        try:
            d = c.runtime.integrate_rate(r, p['composition_dt_s'], clock.methods.update_fractions)
            if p['composition_enabled']:
                c.begin(d, depth, p['composition_dt_s'], cancelled=cancelled)
            else:
                c.validate()
                zero = np.zeros(len(c.runtime.methods.fraction_ids))
                c.pending = (c.mass.copy(), c.sv.copy(), zero.copy(), zero.copy(), 0., zero.copy())
                c.seal_pending()
            out, sv = c.pending[:2]
            new_inventory_bed = c.rock+((out/c.xp.asarray(c.runtime.rho)[None, :, None]).sum(axis=1)/sv).sum(axis=0)
            inventory_dz = new_inventory_bed-c.bed
            if p['bed_enabled']:
                # FLOW's CmpUpd=false diagnostic geometry path does not
                # apply CmpUpd fraction masks (bott3d 1045 versus 1130).
                virtual = c.runtime.integrate_rate(r, p['bed_dt_s'])
                dz = inventory_dz if p['composition_enabled'] else (virtual/c.xp.asarray(c.runtime.rho)[:, None]).sum(axis=0)
            else:
                dz = c.xp.zeros_like(inventory_dz)
            self.pending = (arbiter, p, self.geometry_bed+dz, dz-inventory_dz)
            self.pending_sha256 = self._pending_token()
        except BaseException:
            c.rollback(); clock.rollback(); raise
        return arbiter

    def _pending_token(self):
        a, p, bed, geometric = self.pending
        return digest({'arbiter': a, 'clocks': p, 'geometry': array_sha(host(bed), host(geometric))})

    def commit(self, *, cancelled=False, before_publish=None):
        if self.pending is None: raise ValueError('No composed morphology trial')
        if cancelled:
            self.rollback(); raise InterruptedError('Morphology cancelled before publication')
        arbiter, p, bed, geometric = self.pending
        # Validate all producers before any accepted state is changed.
        self.columns.validate()
        self.columns.validate_pending()
        self.clocks.validate_pending()
        if self.pending_sha256 != self._pending_token(): raise ValueError('Morphology trial was mutated')
        if before_publish is not None:
            try: before_publish()
            except BaseException: self.rollback(); raise
        old = self.checkpoint_accepted()
        try:
            self.columns.commit()
            # The column's dt is morphological, whereas accepted state time is hydraulic.
            self.columns.time_s = self.clocks.time_s+p['dt_s']; self.columns._seal()
            self.clocks.commit()
            self.geometry_bed = bed; self.geometric_only_ledger_m += geometric
            self.pending = None; self.pending_sha256 = None
        except BaseException:
            self.columns.rollback(); self.clocks.rollback(); self.pending = None
            self.restore(old); raise
        return arbiter

    def rollback(self):
        self.columns.rollback(); self.clocks.rollback(); self.pending = None; self.pending_sha256 = None

    def checkpoint_accepted(self):
        # Accepted data only; temporarily detach unpublished handles for serialization.
        cp, kp = self.columns.pending, self.clocks.pending
        try:
            self.columns.pending = self.clocks.pending = None
            d = {'scope': 'braids-q5a-q6a-transaction-v101', 'columns': self.columns.checkpoint(),
                 'clocks': self.clocks.checkpoint(), 'bed': host(self.geometry_bed).tolist(),
                 'geometric_only_ledger_m': host(self.geometric_only_ledger_m).tolist()}
            return {**d, 'sha256': digest(d)}
        finally:
            self.columns.pending, self.clocks.pending = cp, kp

    def checkpoint(self):
        if self.pending is not None: raise ValueError('Checkpoint requires accepted morphology')
        return self.checkpoint_accepted()

    def restore(self, d):
        if self.pending is not None: raise ValueError('Restore requires accepted morphology')
        raw = {k: v for k, v in d.items() if k != 'sha256'}
        if digest(raw) != d['sha256'] or d['scope'] != 'braids-q5a-q6a-transaction-v101':
            raise ValueError('Composed morphology restart identity differs')
        old = self.checkpoint_accepted()
        try:
            self.columns.restore(d['columns']); self.clocks.restore(d['clocks'])
            if self.columns.time_s != self.clocks.time_s: raise ValueError('Restart clocks differ')
            bed, geo = (np.asarray(d[k], dtype=float) for k in ('bed', 'geometric_only_ledger_m'))
            if any(a.shape != (self.columns.runtime.nc,) or not np.isfinite(a).all() for a in (bed, geo)):
                raise ValueError('Restart geometry differs')
            self.geometry_bed, self.geometric_only_ledger_m = (self.columns.xp.asarray(a) for a in (bed, geo))
        except BaseException:
            self.columns.restore(old['columns']); self.clocks.restore(old['clocks']); raise
