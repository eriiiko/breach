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
    A4 (ADDITIVE, presence-gated — an entity-free level's .npz is
    byte-identical to the frozen schema above):
        entity_state         (bytes array, ``capacity``,) per-tick
                             ENTITY_SECT_V1 payload from THE one serializer
                             (``simulation.entities.serialize.
                             serialize_entity_state`` — the same bytes the
                             tick digest hashes, so an offline tool can
                             locate an entity divergence per instance)
        entity_registry_hash (0-d str) registry_content_hash() — entity
                             digests are only comparable at equal hash

Filename: ``debug_{reason}_{YYYYMMDD_HHMMSS}.npz``. ``reason`` is one of
``"manual"`` (F8 dump) or ``"blowup"`` (auto-trigger when
``max |wave_p| > BLOWUP_THRESHOLD``).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from simulation.entities.registry import registry_content_hash
from simulation.entities.serialize import serialize_entity_state


class PhysicsRecorder:
    """Ring buffer that records physics state each tick.

    Keeps last ``capacity`` snapshots in memory. Dumps to .npz on:
      - blowup detected (``max |wave_p|`` > ``BLOWUP_THRESHOLD``)
      - manual trigger (F8 key)

    Which fields to record is configurable per session via ``fields``.
    """

    # Default fields to record (must match GameMap array attribute names).
    # EOS refactor P3 (design §6 recorder row): `wave_p` (now the P_prev
    # buffer) / `wave_v` (retired) drop out; `atmosphere` IS the derived P;
    # `temperature` + the O2 plane join (the new solver's primary state).
    # NOTE: `gas_o2` is resolved specially in record() (a slice of gmap.gas,
    # not a named attribute).
    DEFAULT_FIELDS = ('atmosphere', 'temperature', 'gas_o2', 'smoke', 'fire',
                      'obstacles')
    # EOS P3: the blowup trigger re-keys on the per-tick pressure TRANSIENT
    # |P - P_prev| (design §6) — a standing dome is not a blowup; a runaway
    # per-tick change is.
    BLOWUP_THRESHOLD = 50.0  # max |P - P_prev| (atm/tick) that triggers auto-dump

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

        # A4: per-tick serialized entity state (ENTITY_SECT_V1 bytes), ring
        # buffer style. Stays all-None for an entity-free level, so dump()
        # emits no entity keys and the .npz is byte-identical to pre-A4.
        self.entity_snapshots = [None] * capacity

        print(f"[recorder] Ring buffer: {capacity} slots, fields={self.fields}, "
              f"~{self._mem_mb():.0f} MB")

    def _mem_mb(self):
        total = 0
        for buf in self.buffers.values():
            total += buf.nbytes
        total += self.tick_ids.nbytes + self.tick_times.nbytes
        return total / (1024 * 1024)

    def record(self, gmap, tick, real_time, units, entities=None):
        """Snapshot current state into ring buffer.

        ``entities`` (A4, additive): the sim's runtime entity list — Arc A
        passes the level's parsed ``EntityInstance`` objects. Serialized
        per tick through THE one canonical serializer under the presence
        rule (None/empty records nothing, keeping entity-free dumps
        byte-identical).
        """
        i = self.index % self.capacity
        for name in self.fields:
            # EOS P3: `gas_o2` names the O2 slice of the (N,h,w) gas array.
            if name == 'gas_o2':
                from simulation.gases import O2
                arr = gmap.gas[O2]
            else:
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
                        "atmosphere", "wind_x", "wind_y", "fire",
                        "temperature", "gas_o2") and \
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

        # A4: entity state snapshot — the same ENTITY_SECT_V1 bytes the tick
        # digest hashes (one serializer). Presence-gated: None when the
        # level carries no entities.
        self.entity_snapshots[i] = (
            serialize_entity_state(entities) if entities else None)

        self.index += 1
        self.count += 1

        # Auto-dump on blowup — EOS P3: keyed on the per-tick pressure
        # TRANSIENT |P - P_prev| (gmap.atmosphere is P; gmap.wave_p is the
        # repurposed P_prev buffer), NOT the raw field level: a standing
        # pressure dome is legitimate physics now; a runaway per-tick change
        # is not.
        if 'atmosphere' in self.buffers:
            transient = np.abs(gmap.atmosphere.astype(np.float64)
                               - gmap.wave_p.astype(np.float64)) / 65536.0
            max_transient = float(transient.max())
            if max_transient > self.BLOWUP_THRESHOLD and not self.dumped:
                print(f"[recorder] BLOWUP DETECTED: max |P - P_prev| = {max_transient:.1f}")
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
            # record() stores the unit position under keys 'x'/'y' (the Q16.16
            # fixed-point coords u.x/u.y); the on-disk schema names them
            # unit_fx/unit_fy. Read the keys record() actually wrote — the old
            # 'fx'/'fy' keys never existed, so every blowup/F8 dump crashed with
            # KeyError: 'fx' (reproduced on test_level: air-vs-vacuum venting
            # trips the blowup dump at tick 70).
            data['unit_fx'] = np.array(
                [[u['x'] for u in snap] for snap in unit_snaps], dtype=np.int32)
            data['unit_fy'] = np.array(
                [[u['y'] for u in snap] for snap in unit_snaps], dtype=np.int32)
            data['unit_hp'] = np.array(
                [[u['hp'] for u in snap] for snap in unit_snaps], dtype=np.int32)
            data['unit_alive'] = np.array(
                [[u['alive'] for u in snap] for snap in unit_snaps],
                dtype=np.bool_)
            data['unit_names'] = np.array([u['name'] for u in unit_snaps[0]])

        # A4: entity payload — ADDITIVE and presence-gated, so an
        # entity-free level's dump carries exactly the frozen key set (and
        # byte-identical content) it did before A4. Serialized payloads end
        # with the '\n' record/preamble terminator, so the S-dtype's
        # trailing-NUL stripping can never truncate them.
        ent_snaps = [self.entity_snapshots[i]
                     for i in (range(n) if isinstance(slc, slice) else slc)]
        if any(s is not None for s in ent_snaps):
            data['entity_state'] = np.array(
                [s if s is not None else b"" for s in ent_snaps],
                dtype=np.bytes_)
            data['entity_registry_hash'] = np.array(registry_content_hash())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_{reason}_{timestamp}.npz"
        np.savez_compressed(filename, **data)
        print(f"[recorder] Dumped {n} snapshots to {filename}")
        return filename
