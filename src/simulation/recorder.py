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
    Arc #63 P2 (ADDITIVE, presence-gated — a swarm-free session's .npz is
    byte-identical to the frozen schema above; engine/17 §6/§7):
        swarm_<species>_<col> (T, wmax) at the ROSTER dtype VERBATIM (no
                             dequantize, no cast), env 0 only, wmax =
                             min(swarm_units_cap, max recorded high_water)
        swarm_<species>_high_water (T,) int32
        swarm_next_unit_id   (T,) int32 — shared across species
        swarm_species_hash   (T,) int64 — the species table hash's first 8
                             bytes, per tick (so a hot reload inside the
                             ring window is visible at its tick); 0 == no
                             swarm that tick

Filename: ``debug_{reason}_{YYYYMMDD_HHMMSS}.npz``. ``reason`` is one of
``"manual"`` (F8 dump) or ``"blowup"`` (auto-trigger when
``max |wave_p| > BLOWUP_THRESHOLD``).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from simulation.entities.registry import registry_content_hash
from simulation.entities.serialize import serialize_entity_state
from simulation.swarm import (  # arc #63 P2
    FIRST_UNIT_ID, NEXT_UNIT_ID_ATTR, ROSTER, SPECIES_HASH_NPZ_KEY,
    SWARM_SPECIES, high_water_attr, store_attr,
)


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
    # ENERGY-BOOKS ARC, P-E5 (2026-08-17): `wind_x`/`wind_y` and `inert_n2`
    # joined the default set. Rationale — the arc closed the THERMAL books and
    # in-game dumps confirm temperature is now well behaved, but pressure
    # transients remain (98 atm at a normal ~700 game-T => ~29x ambient density
    # in one cell: a MASS/MOMENTUM event, not a thermal one). Every remaining
    # candidate is a momentum story, and wind is NOT recoverable from the
    # pressure field: the gradient gives the per-tick ACCELERATION, while u is
    # its accumulated history (the two run ~90 deg out of phase in the
    # Helmholtz mode). The storm audit named this exact gap
    # (docs/storm_audit_2026-08-14.md §1: "a momentum ledger needs a recorder
    # session that adds wind_x/wind_y to `fields`"), and design §9 already
    # required the bulk pair for P-E5 validation recordings. `gas_o2` +
    # `inert_n2` together give N, so p* = C*N*T_abs can be decomposed offline.
    # GAS-ENERGY CONSERVATION ARC, P-G0 (2026-08-29, design §5/§Systems): +
    # `gas_energy` -- the new int64 conserved-energy field (design §2.2). It
    # rides the TRUE int64 ring branch below (`_INT64_FIELDS`), not the
    # float32 one: a float32 cast would drop bits SB's closure-identity gate
    # asserts on exactly (Recorder rule amendment, CLAUDE.md).
    DEFAULT_FIELDS = ('atmosphere', 'temperature', 'gas_o2', 'inert_n2',
                      'wind_x', 'wind_y', 'smoke', 'fire', 'obstacles',
                      'gas_energy')
    # SYNCED bool planes: recorded at bool dtype (not the float32 ring) when a
    # session lists them in `fields`. `ignition_armed` is the edge-trigger arm
    # (combat.apply_temperature_ignition); mirrors the `obstacles` bool handling.
    _BOOL_FIELDS = ('obstacles', 'ignition_armed')
    # GAS-ENERGY CONSERVATION ARC, P-G0: SYNCED int64 planes, recorded at
    # int64 dtype -- NO CAST (a float64 ring is only exact to 2^53 and would
    # drop the LSBs the arc's closure-identity gates assert on; the frozen
    # .npz contract is extended additively by *dtype class*, not just
    # membership -- the Recorder rule amendment this arc makes, mirroring
    # `_BOOL_FIELDS`'s own precedent).
    _INT64_FIELDS = ('gas_energy',)
    # EOS P3: the blowup trigger re-keys on the per-tick pressure TRANSIENT
    # |P - P_prev| (design §6) — a standing dome is not a blowup; a runaway
    # per-tick change is.
    BLOWUP_THRESHOLD = 50.0  # max |P - P_prev| (atm/tick) that triggers auto-dump

    def __init__(self, fh, fw, capacity=1200, fields=None, swarm_units_cap=1024):
        self.fh = fh
        self.fw = fw
        self.capacity = capacity
        self.fields = list(fields or self.DEFAULT_FIELDS)
        self.index = 0       # next write position
        self.count = 0       # total snapshots written (whether buffer wrapped)
        self.dumped = False  # prevent repeated auto-dumps for same blowup

        # arc #63 P2 (engine/17 §6/§7): swarm per-unit ring family — a
        # SEPARATE ring family from the grid `fields` rings above, allocated
        # LAZILY at the first tick a swarm is present (env 0 only). Staying
        # None keeps a swarm-free session's dump byte-identical to pre-P2
        # (the entity/signal precedent, §7).
        self.swarm_units_cap = swarm_units_cap
        self._swarm_rings = None        # {species: {col_name: ndarray}}
        self._swarm_hw_rings = None     # {species: ndarray (capacity,) int32}
        self._swarm_nid_ring = None     # ndarray (capacity,) int32
        self._swarm_shash_ring = None   # ndarray (capacity,) int64
        self._swarm_truncated_printed = False   # one truncation line, ever

        # Pre-allocate ring buffers. BOOL synced planes (`obstacles`, and the
        # edge-trigger `ignition_armed`) record at bool dtype; everything else is
        # the float32 (dequantized) ring. Named here so any bool plane a session
        # adds to `fields` is stored losslessly instead of cast to float32.
        self.buffers = {}
        for name in self.fields:
            if name in self._BOOL_FIELDS:
                dtype = np.bool_
            elif name in self._INT64_FIELDS:
                dtype = np.int64
            else:
                dtype = np.float32
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

        # Arc B: per-tick serialized __signals__ bytes (SIGNAL_SECT_V1), ring
        # buffer style. Stays all-None on a wire-free level, so dump() emits no
        # signal key and the .npz is byte-identical to pre-Arc-B.
        self.signal_snapshots = [None] * capacity

        print(f"[recorder] Ring buffer: {capacity} slots, fields={self.fields}, "
              f"~{self._mem_mb():.0f} MB")

    def _mem_mb(self):
        total = 0
        for buf in self.buffers.values():
            total += buf.nbytes
        total += self.tick_ids.nbytes + self.tick_times.nbytes
        return total / (1024 * 1024)

    def _alloc_swarm_rings(self):
        """Lazy allocation at the first present tick (§7). Env 0 only, width
        ``swarm_units_cap`` — a real swarm may run at ``max_units`` far wider
        than this cap; capacity moves no digest byte and the recorder trims
        the same way (``[0, high_water)``, capped)."""
        W = self.swarm_units_cap
        self._swarm_rings = {
            sp: {c.name: np.zeros((self.capacity, W), dtype=c.dtype)
                 for c in ROSTER}
            for sp in SWARM_SPECIES
        }
        self._swarm_hw_rings = {
            sp: np.zeros((self.capacity,), dtype=np.int32)
            for sp in SWARM_SPECIES
        }
        # "earlier rows are truthfully 'nothing spawned'" (§7) — FIRST_UNIT_ID,
        # not 0, is the correct pre-spawn value of next_unit_id.
        self._swarm_nid_ring = np.full((self.capacity,), FIRST_UNIT_ID,
                                       dtype=np.int32)
        self._swarm_shash_ring = np.zeros((self.capacity,), dtype=np.int64)
        total = sum(a.nbytes for cols in self._swarm_rings.values()
                    for a in cols.values())
        total += sum(a.nbytes for a in self._swarm_hw_rings.values())
        total += self._swarm_nid_ring.nbytes + self._swarm_shash_ring.nbytes
        print(f"[recorder] Swarm ring buffer: {self.capacity} slots x "
              f"{W} units/species, ~{total / (1024 * 1024):.0f} MB")

    @staticmethod
    def _species_hash_int64(species_hash: str) -> int:
        """The species-hash ring's int64 value (R13): the hash's first 8
        bytes, little-endian signed. 0 == no swarm this tick — an existing
        ring dtype class (``_INT64_FIELDS``)."""
        if not species_hash:
            return 0
        return int.from_bytes(bytes.fromhex(species_hash)[:8], "little",
                              signed=True)

    def record(self, gmap, tick, real_time, units, entities=None, signals=(),
              swarm=None):
        """Snapshot current state into ring buffer.

        ``entities`` (A4, additive): the sim's runtime entity list — Arc A
        passes the level's parsed ``EntityInstance`` objects. Serialized
        per tick through THE one canonical serializer under the presence
        rule (None/empty records nothing, keeping entity-free dumps
        byte-identical).

        ``signals`` (Arc B, additive): the SignalBus ``__signals__`` rows
        ``(ordinal, name, value)`` — non-empty ONLY when the level declares
        wires (a bus exists). A dormant (wire-free) level passes ``()`` →
        nothing is recorded and the .npz stays byte-identical to Arc A (§8);
        ``serialize_entity_state`` is NOT extended to fold signals (that would
        change every door level's recorded bytes) — signals ride their own
        additive key.

        ``swarm`` (arc #63 P2, additive): the §5.2 carrier
        (``simulation.swarm.swarm_carrier``) — the digest and the recorder
        consume the SAME carrier structure. ``None`` (a swarm-free session,
        or a caller that never passes it — the ``_FakeGmap`` recorder tests)
        is treated as absent; the per-unit ring family is allocated lazily
        at the first present tick and stays unallocated (no ``swarm_*`` dump
        key at all) for a swarm-free run.
        """
        i = self.index % self.capacity
        for name in self.fields:
            # EOS P3: `gas_o2` names the O2 slice of the (N,h,w) gas array.
            # P-E5: `inert_n2` is the same idiom for the other bulk plane —
            # together they are N, which is what makes p* = C*N*T_abs
            # decomposable offline (a 98 atm cell at a normal ~700 game-T is a
            # DENSITY event, and without this plane that is not provable).
            if name == 'gas_o2':
                from simulation.gases import O2
                arr = gmap.gas[O2]
            elif name == 'inert_n2':
                from simulation.gases import INERT_N2
                arr = gmap.gas[INERT_N2]
            else:
                arr = getattr(gmap, name)
            # S2a/S2b/S2c: wave_p / wave_v / wave_source / smoke (S2a/S2b) AND
            # atmosphere / wind_x / wind_y (S2c) are now int32 Q16.16 — DEQUANTIZE
            # to real units (/65536) at the recorder boundary so the float32 ring
            # buffer (and the BLOWUP_THRESHOLD compare below) stays in meaningful
            # units, not raw counts. Render/debug only — not part of the synced
            # state. (`smoke` is the SMOKE int32 view; all share the 2^16
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

        # Arc B: __signals__ bytes — additive + presence-gated (None when no
        # bus / no wires), so a dormant level's dump is byte-identical.
        if signals:
            from simulation.entities.serialize import serialize_signal_state
            self.signal_snapshots[i] = serialize_signal_state(signals)
        else:
            self.signal_snapshots[i] = None

        # arc #63 P2 (engine/17 §6/§7): the swarm per-unit ring family — a
        # SEPARATE ring family from the grid `fields` rings, allocated
        # lazily at the first present tick, env 0 only. Values are raw, at
        # the ROSTER dtype (no dequantize, no cast).
        if swarm is not None and swarm.get("present"):
            if self._swarm_rings is None:
                self._alloc_swarm_rings()
            W = self.swarm_units_cap
            for sp in SWARM_SPECIES:
                block = swarm["blocks"].get(sp)
                ring = self._swarm_rings[sp]
                if block is None:
                    # A species absent this tick writes zeros and hw 0.
                    for c in ROSTER:
                        ring[c.name][i] = 0
                    self._swarm_hw_rings[sp][i] = 0
                    continue
                hw = int(block["high_water"])
                w = min(hw, W)
                for c in ROSTER:
                    col = block["columns"][c.name]
                    ring[c.name][i, :w] = col[:w]
                    ring[c.name][i, w:] = 0
                self._swarm_hw_rings[sp][i] = hw
                if hw > W and not self._swarm_truncated_printed:
                    print(f"[recorder] swarm {sp} high_water {hw} > "
                          f"swarm_units_cap {W}: recording truncated to "
                          f"the first {W} slots")
                    self._swarm_truncated_printed = True
            self._swarm_nid_ring[i] = int(swarm["next_unit_id"])
            self._swarm_shash_ring[i] = self._species_hash_int64(
                swarm.get("species_hash", ""))
        elif self._swarm_rings is not None:
            # Rings exist from an earlier present tick, but this record()
            # call's swarm is absent/None — the "nothing" row (defensive:
            # unreachable via Simulation, whose presence is monotone within
            # one recorder's lifetime, but correct for a direct caller).
            for sp in SWARM_SPECIES:
                for c in ROSTER:
                    self._swarm_rings[sp][c.name][i] = 0
                self._swarm_hw_rings[sp][i] = 0
            self._swarm_nid_ring[i] = FIRST_UNIT_ID
            self._swarm_shash_ring[i] = 0

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

        # Arc B: __signals__ payload — ADDITIVE + presence-gated, so a
        # wire-free level's dump carries exactly the pre-Arc-B key set.
        sig_snaps = [self.signal_snapshots[i]
                     for i in (range(n) if isinstance(slc, slice) else slc)]
        if any(s is not None for s in sig_snaps):
            data['signal_state'] = np.array(
                [s if s is not None else b"" for s in sig_snaps],
                dtype=np.bytes_)

        # arc #63 P2 (engine/17 §7): swarm per-unit rings — ADDITIVE +
        # presence-gated: no swarm_* key at all when no ring was ever
        # allocated (identical key set to pre-P2). `wmax` bounds each
        # column ring to [0, max recorded high_water) — the same
        # [0, high_water) trim the digest applies, offline.
        if self._swarm_rings is not None:
            for sp in SWARM_SPECIES:
                hw_ring = self._swarm_hw_rings[sp][slc]
                wmax = min(self.swarm_units_cap,
                          int(hw_ring.max()) if hw_ring.size else 0)
                for c in ROSTER:
                    data[store_attr(sp, c.name)] = \
                        self._swarm_rings[sp][c.name][slc][:, :wmax]
                data[high_water_attr(sp)] = hw_ring
            data[NEXT_UNIT_ID_ATTR] = self._swarm_nid_ring[slc]
            data[SPECIES_HASH_NPZ_KEY] = self._swarm_shash_ring[slc]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_{reason}_{timestamp}.npz"
        np.savez_compressed(filename, **data)
        print(f"[recorder] Dumped {n} snapshots to {filename}")
        return filename
