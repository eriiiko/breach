"""PhysicsRecorder — ring buffer of GameMap snapshots that dumps on blowup or F8.

Lifted **verbatim** from ``game.py:PhysicsRecorder`` (lines 179-286).
Active debugging workflow — the ``.npz`` files in the repo root
(``debug_blowup_*.npz``) are loaded by the offline physics analysis
scripts. Do not change field names or per-snapshot shapes without
coordinating: the on-disk format is what offline tools depend on.

.npz schema (frozen):

    Per-tick grids (``capacity``, fh, fw):
        wave_p, wave_v, atmosphere, smoke, fire (float32)
        obstacles                              (bool)
    Per-tick scalars (``capacity``,):
        tick_ids   (int32)   game tick number
        tick_times (float64) wall-clock seconds since round start
    Per-tick unit snapshots (capacity, n_units):
        unit_fx, unit_fy, unit_hp (int32)
        unit_alive                (bool)
    Per-dump scalars:
        unit_names (str array, length n_units, taken from the first snapshot)

Filename: ``debug_{reason}_{YYYYMMDD_HHMMSS}.npz``. ``reason`` is one of
``"manual"`` (F8 dump) or ``"blowup"`` (auto-trigger when
``max |wave_p| > BLOWUP_THRESHOLD``).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np


class PhysicsRecorder:
    """Ring buffer that records physics state each tick.

    Keeps last ``capacity`` snapshots in memory. Dumps to .npz on:
      - blowup detected (``max |wave_p|`` > ``BLOWUP_THRESHOLD``)
      - manual trigger (F8 key)

    Which fields to record is configurable per session via ``fields``.
    """

    # Default fields to record (must match GameMap array attribute names).
    DEFAULT_FIELDS = ('wave_p', 'wave_v', 'atmosphere', 'smoke', 'fire',
                      'obstacles')
    BLOWUP_THRESHOLD = 50.0  # max |wave_p| that triggers auto-dump

    def __init__(self, fh, fw, capacity=1200, fields=None):
        self.fh = fh
        self.fw = fw
        self.capacity = capacity
        self.fields = list(fields or self.DEFAULT_FIELDS)
        self.index = 0       # next write position
        self.count = 0       # total snapshots written (whether buffer wrapped)
        self.dumped = False  # prevent repeated auto-dumps for same blowup

        # Pre-allocate ring buffers
        self.buffers = {}
        for name in self.fields:
            dtype = np.bool_ if name == 'obstacles' else np.float32
            self.buffers[name] = np.zeros((capacity, fh, fw), dtype=dtype)

        # Tick metadata (tick number, real_time, etc.)
        self.tick_ids = np.zeros(capacity, dtype=np.int32)
        self.tick_times = np.zeros(capacity, dtype=np.float64)

        # Unit state per tick: list of dicts, ring buffer style
        self.unit_snapshots = [None] * capacity

        print(f"[recorder] Ring buffer: {capacity} slots, fields={self.fields}, "
              f"~{self._mem_mb():.0f} MB")

    def _mem_mb(self):
        total = 0
        for buf in self.buffers.values():
            total += buf.nbytes
        total += self.tick_ids.nbytes + self.tick_times.nbytes
        return total / (1024 * 1024)

    def record(self, gmap, tick, real_time, units):
        """Snapshot current state into ring buffer."""
        i = self.index % self.capacity
        for name in self.fields:
            arr = getattr(gmap, name)
            # S2a/S2b/S2c: wave_p / wave_v / wave_source / smoke (S2a/S2b) AND
            # atmosphere / wind_x / wind_y (S2c) are now int32 Q16.16 — DEQUANTIZE
            # to real units (/65536) at the recorder boundary so the float32 ring
            # buffer (and the BLOWUP_THRESHOLD compare below) stays in meaningful
            # units, not raw counts. Render/debug only — not part of the synced
            # state. (`smoke` is the BLACK_SMOKE int32 view; all share the 2^16
            # scale.) The dtype guard makes this a no-op for any field that stays
            # float, so the same code is safe across the migration.
            if name in ("wave_p", "wave_v", "wave_source", "smoke",
                        "atmosphere", "wind_x", "wind_y") and \
                    arr.dtype == np.int32:
                arr = arr.astype(np.float64) / 65536.0
            self.buffers[name][i] = arr
        self.tick_ids[i] = tick
        self.tick_times[i] = real_time

        # Unit state snapshot (lightweight dict per unit)
        self.unit_snapshots[i] = [
            {'name': u.name, 'team': u.team, 'x': u.x, 'y': u.y,
             'hp': u.current_hp, 'alive': u.alive}
            for u in units
        ]

        self.index += 1
        self.count += 1

        # Auto-dump on blowup
        if 'wave_p' in self.buffers:
            max_wave = np.max(np.abs(self.buffers['wave_p'][i]))
            if max_wave > self.BLOWUP_THRESHOLD and not self.dumped:
                print(f"[recorder] BLOWUP DETECTED: max |wave_p| = {max_wave:.1f}")
                self.dump("blowup")
                self.dumped = True

    def dump(self, reason="manual"):
        """Write ring buffer contents to timestamped .npz file."""
        n = min(self.count, self.capacity)
        if n == 0:
            print("[recorder] Nothing to dump.")
            return

        # Unroll ring buffer into chronological order
        if self.count <= self.capacity:
            slc = slice(0, n)
        else:
            # Buffer has wrapped — reorder so oldest is first
            start = self.index % self.capacity
            order = np.roll(np.arange(self.capacity), -start)
            slc = order

        data = {}
        for name in self.fields:
            data[name] = self.buffers[name][slc]
        data['tick_ids'] = self.tick_ids[slc]
        data['tick_times'] = self.tick_times[slc]

        # Pack unit snapshots into structured arrays
        unit_snaps = [self.unit_snapshots[i]
                      for i in (range(n) if isinstance(slc, slice) else slc)]
        # Store as: unit_fx[tick, unit_idx], unit_fy, unit_hp, unit_alive
        if unit_snaps[0] is not None:
            n_units = len(unit_snaps[0])
            data['unit_fx'] = np.array(
                [[u['fx'] for u in snap] for snap in unit_snaps], dtype=np.int32)
            data['unit_fy'] = np.array(
                [[u['fy'] for u in snap] for snap in unit_snaps], dtype=np.int32)
            data['unit_hp'] = np.array(
                [[u['hp'] for u in snap] for snap in unit_snaps], dtype=np.int32)
            data['unit_alive'] = np.array(
                [[u['alive'] for u in snap] for snap in unit_snaps],
                dtype=np.bool_)
            data['unit_names'] = np.array([u['name'] for u in unit_snaps[0]])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_{reason}_{timestamp}.npz"
        np.savez_compressed(filename, **data)
        print(f"[recorder] Dumped {n} snapshots to {filename}")
        return filename
