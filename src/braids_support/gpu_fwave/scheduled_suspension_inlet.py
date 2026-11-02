"""Isolated GPU-resident SSPRK33 boundary records for scheduled suspension.

The existing four-slot GPU table kernel is reused without a CPU forcing
evaluation. Native SSPRK33 queries Euler boundaries at t, t+dt, t+dt/2;
these are resident slots 0, 2, 1 respectively. This component is deliberately
not wired into the stock coupled solver or runner. `feed` means prescribed
bed-load mass per unit time, whereas Q * concentration supplies suspended mass.
"""
from __future__ import annotations

from types import MappingProxyType
import numpy as np

from .boundary_events import BoundaryEvents, tolerance
from .boundary_schedule import ResidentBoundarySchedule, require, stage_times, validate_tables

SSP33_SLOTS = (0, 2, 1)
SSP33_FRACTIONS = (0., 1., .5)
SSP33_LEDGER_WEIGHTS = (1./6., 1./6., 2./3.)
STOCK_MODES = (2, 3, 6, 8)


def validate_input(data):
    """Fail closed on native per-class/stock-GPU scope before initializing CUDA."""
    version = np.asarray(data.get('deck_version', -1))
    require(version.shape == () and version.dtype.kind in 'iu' and int(version) >= 15,
            'Per-class inlet schedules require native deck version >=15')
    fractions = np.asarray(data['fraction_concentration'])
    require(fractions.ndim == 2 and 2 <= fractions.shape[0] <= 32,
            'Native multi-fraction schedules require 2 to 32 classes')
    require(np.isin(np.asarray(data['mode']), STOCK_MODES).all(),
            'Scheduled inlet component supports only stock GPU boundary modes')
    require(not np.any(data.get('nseries', 0)),
            'Native multi-fraction decks forbid legacy scalar sediment series')
    require(not np.any(data.get('nmorfac', 0)),
            'Time-dependent MorFac is outside this schedule component')
    tables = validate_tables(data)
    bed_mode = np.asarray(data.get('bed_boundary_mode', np.zeros(len(tables['mode']), np.int32)))
    require(bed_mode.shape == tables['mode'].shape and bed_mode.dtype.kind in 'iu' and
            np.isin(bed_mode, (0, 1)).all(), 'Invalid bed-load boundary mode')
    for boundary in range(len(tables['mode'])):
        has_feed = (np.any(tables['fraction_bed_input'][:, boundary] > 0) or
            np.any(tables['fraction_series_bed_input'][:, :, boundary] > 0))
        require(not (bed_mode[boundary] == 1 and has_feed),
                'Equilibrium bed-load inlet conflicts with prescribed feed')
        if has_feed:
            require(max(tables['forcing'][0, boundary],
                np.max(tables['hydro_series_q'][:, boundary], initial=0.)) > 0,
                'Positive bed feed requires positive available inlet discharge')
    return tables


class ResidentScheduledInletForcing:
    """Upload native tables once; expose stable device views per Euler stage.

    `begin_trial` only uploads two clock scalars and launches the existing GPU
    interpolation kernel. It does not synchronize, copy stage fields to host,
    or commit anything. The coupled caller must merge each stage's device
    status with its trial status and must use a knot-clipped timestep.
    """

    def __init__(self, data, *, source_path=None):
        validate_input(data)
        self.schedule = ResidentBoundarySchedule(data, source_path=source_path)
        self.events = BoundaryEvents.from_tables(self.schedule.host)
        outputs = self.schedule.outputs
        self.records = tuple(MappingProxyType(dict(
            stage_time=outputs['stage_time'][slot:slot+1],
            hydraulic=outputs['hydraulic'][slot],
            concentration=outputs['concentration'][slot],
            feed=outputs['feed'][slot],
            status=outputs['status'][slot])) for slot in SSP33_SLOTS)

    def begin_trial(self, current, dt):
        stage_times(current, dt)
        require(current >= 0, 'Nonnegative native clock required')
        knot = self.events.next_after(current)
        require(current+dt <= knot+tolerance(current),
                'Trial crosses a prescribed boundary knot')
        self.schedule.evaluate(current, dt)
        return self.records

    def verify_tables(self):
        return self.schedule.verify_tables()

    def receipt(self):
        return dict(self.schedule.receipt(), native_integrator='SSPRK33',
            euler_record_indices=list(SSP33_SLOTS),
            euler_stage_fractions=list(SSP33_FRACTIONS),
            ledger_weights=list(SSP33_LEDGER_WEIGHTS),
            boundary_event_identity=self.events.identity,
            per_class_concentration_supported_by_component=True,
            feed_interpretation='prescribed bed-load kg/s, not suspended mass directly',
            suspended_inlet_flux='water discharge times boundary concentration',
            coupled_integration_qualified=False,
            four_slot_kernel_reused_for_three_stages=True)
