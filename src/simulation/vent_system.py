"""Vent/duct RUNTIME — the slot-9e(d) circulation-N-feed sweep (vent system
PATCH 1, issue #48).

Design: docs/vent_system_design_2026-08-23.md (v2, post-adversarial-critique),
§3 (mechanism) / §4 (what flows) / §5 (plenum) / §7 (patch 1 scope). This is
the sim-side runtime of the ``duct``/``vent`` entities (schema in
:mod:`simulation.entities.vents`, import-light) — a SIBLING of
:mod:`simulation.pump_system`: same 9e(d) actuator slot, same 2-tick
field-effect contract, same build-at-reset pattern, same integer-native
GameMap gas-N primitives (extended here — :meth:`GameMap.extract_gas_n_vec` /
:meth:`GameMap.inject_gas_n_vec`, §3's "the write path is the pump
primitives, not FieldEdit").

UNLIKE pumps, vents carry no wired inputs in patch 1 (no makeup controller,
no pressure sensing at all — see the entities.vents module doc for the
explicit out-of-scope list), so :func:`build_vents` runs UNCONDITIONALLY at
reset (the door precedent), never gated on ``sim._signal_bus``.

Patch-1 mechanism (§3), per duct per tick, in ENTITY-ORDINAL order (pinned,
machine-identical — the serializer's own order, §3):

1. Every RETURN vent extracts up to its own per-tick Bresenham-accrued flux
   quantum (``extract_gas_n_vec`` — zero-clamped by the aperture's actual
   content). The MEASURED per-slice withdrawal vector credits the plenum:
   bulk (o2/n2) verbatim, trace species through the duct's filter (§4:
   scrubbed mass -> the counted sink). The ENERGY credit is BULK-ONLY —
   ``(removed[O2]+removed[INERT_N2]) * t_tile`` — per the engine's own P-T0
   convention (cpp/src/eos_solver.cpp:719-729: ``n_total == n_bulk``, trace
   planes carry no engine-side energy). REVIEW FIX (2026-08-23, orchestrator-
   ruled): the v1 patch credited the FULL measured withdrawal (bulk + trace),
   which manufactured energy out of nothing whenever trace mass was present
   (a scrubber-becomes-heater bug) — the "scrubbed smoke keeps its heat"
   framing in the design doc's §4 is retired; trace transport is mass-only,
   full stop, and the energy books close exactly WITH trace present.
2. The duct then distributes ``min(N_plenum_bulk, sum of supply vents' own
   accrued quanta)`` across its supply vents — draining CIRCULATING CREDIT
   ONLY (patch 3's reserve, when it lands, is a SEPARATE additive ENTITY_SECT
   row pair on the duct, never seeded into ``o2_raw``/``n2_raw``, or
   circulation would drain the reserve by construction) — each vent's own
   accrued quantum serving as its FIXED INTEGER WEIGHT (§3) — floor-
   proportional per vent, remainder simply never leaves the plenum (the
   "Bresenham remainder carried plenum-side" idiom: nothing is force-
   cascaded to sum exactly, so the dust just stays in the ledger for next
   tick). Each supply vent's bulk share is split via
   :meth:`GameMap.gas_proportional_split` against the plenum's REMAINING
   ``[o2, n2]`` pool, DEDUCTED as each vent is processed (REVIEW FIX: the v1
   patch used the plenum's ORIGINAL ratio + an "exact complement" trick for
   EVERY vent independently, which is only safe for a single split — across
   several supply vents on a skewed ratio it could inflate the summed minor
   species past its actual holdings, going negative). Trace rides along at
   the SAME plenum ratio (well-mixed assumption) — safe already, by floor
   subadditivity, without the remaining-pool bookkeeping (see
   ``_duct_sweep``'s comment). Every deposit lands via ``inject_gas_n_vec``,
   which also mass-weighted-mixes the tile's temperature toward the plenum's
   own ``T_dep = E_plenum // N_plenum_bulk`` (§4).

RUNTIME APERTURE GUARDS (§4, checked EVERY tick, not just at load): a vent
whose aperture is solid, thermal_solid, vacuum, ambient, or flooded is a
COUNTED no-op for the WHOLE tick — its accumulator is left untouched (frozen,
not manufactured/lost) and it contributes nothing to either side of the
ledger, so a guard hit can never corrupt conservation.

Determinism: integer-only (Q16.16 for flux quanta, int64 raw N*T for the
energy ledger); ordinal-order sweep; rates quantized ONCE at load
(build_vents); no RNG, no float, no dequantize in the sweep itself.
``e_plenum``/``e_wipe`` are plain Python ints (no overflow in the sim layer
itself) but are PACKED as signed int64 at the ENTITY_SECT digest boundary
(entities/serialize.py's ``_pack_i64``) — a pathological-scale run that
overflows int64 there fails LOUD (an ``OverflowError`` naming the row), which
is accepted: no shipped scenario gets remotely close (accumulated N*T at
Q16.16 scale needs ~2^63 raw to overflow).
"""
from __future__ import annotations

from config import CFG
from simulation import gas_fixed as _gas_fx
from simulation.ambient import DEFAULT_C, DEFAULT_T_AMB_K
from simulation.entities.vents import resolve_aperture_base
from simulation.gases import (
    FUEL_GAS, INERT_N2, N_GASES, O2, POISON, SMOKE, STEAM, TEARGAS,
)

# Trace gas ids in filter-column order (config.toml [filters.<name>] keys,
# entities.vents' trace_i/sink_i digest-row index order) — the ONE place this
# mapping is spelled out; entities/vents.py stays import-light and cannot
# import simulation.gases (it pulls numpy).
_TRACE_IDS = (STEAM, SMOKE, POISON, TEARGAS, FUEL_GAS)
_TRACE_NAMES = ("steam", "smoke", "poison", "teargas", "fuel_gas")
N_TRACE = len(_TRACE_IDS)

# The near-empty-plenum divide guard (§4's "N_EPS-style floor"): below this
# many raw Q16.16 units of BULK holdings, E_plenum/N_plenum is not trusted as
# a temperature (a near-zero denominator would amplify quantization noise
# into an absurd T) — the residual energy is wiped into the counted `e_wipe`
# channel and deposits (if any could even occur — N_bulk this small means
# `total_avail` is at most N_EPS too) go out at ambient (T_dep = 0). Pinned,
# not a config dial — patch 1 ships no tuning (design orchestration scope).
N_EPS = 64


def _duct_filter_efficiency_q(duct_inst_fields, filters_cfg, duct_id):
    """Resolve ``duct.filter`` against ``[filters.<name>]`` (§4) — a
    CONFIG-integrity error (not the generic dangling-entity-ref warning:
    this name addresses config.toml, not another ``[[entity]]``). Returns a
    length-``N_TRACE`` list of Q16 fractions in TRACE-id order.

    REVIEW FIX (minor): each quantized efficiency is validated
    ``0 <= eff_q <= FP_ONE`` (65536) — an authored value outside ``[0, 1]``
    would scrub MORE than the removed mass (or a negative amount), a silent
    conservation break at the intake site rather than a loud one here."""
    name = duct_inst_fields["filter"]
    row = getattr(filters_cfg, name, None) if filters_cfg is not None else None
    if row is None:
        known = sorted(vars(filters_cfg).keys()) if filters_cfg is not None else []
        raise ValueError(
            f"duct '{duct_id}': filter '{name}' has no [filters.{name}] row "
            f"in config.toml (vent design §4 — a filter is a table row); "
            f"declared rows: {known}")
    eff_q = []
    for gname in _TRACE_NAMES:
        q = _gas_fx.quantize_scalar(float(getattr(row, gname)))
        if not (0 <= q <= _gas_fx.FP_ONE):
            raise ValueError(
                f"duct '{duct_id}': [filters.{name}].{gname} = "
                f"{getattr(row, gname)!r} quantizes to {q} raw, outside "
                f"[0, {_gas_fx.FP_ONE}] (a filter efficiency is a [0,1] "
                f"fraction of removed mass — vent design §4)")
        eff_q.append(q)
    return eff_q


class DuctRuntime:
    """Sim-side runtime object for one ``duct`` (§5) — the plenum ledger.

    Doubles as the SERIALIZER runtime object (duck-typed ordinal/id/
    class_name/fields + ``alive``, the PumpRuntime/DoorRuntime pattern) so
    the ``duct`` class's ``runtime_digest_rows`` reads the ledger straight
    off the sim's entity list.
    """

    __slots__ = ("inst", "filter_eff_q", "o2_raw", "n2_raw", "trace_raw",
                 "e_plenum", "sink", "e_wipe", "rail_lo_hits", "rail_hi_hits",
                 "alive")

    def __init__(self, inst, filter_eff_q):
        self.inst = inst
        self.filter_eff_q = list(filter_eff_q)
        self.o2_raw = 0
        self.n2_raw = 0
        self.trace_raw = [0] * N_TRACE
        self.e_plenum = 0
        self.sink = [0] * N_TRACE
        self.e_wipe = 0
        self.rail_lo_hits = 0
        self.rail_hi_hits = 0
        self.alive = True

    # --- serializer duck-type -------------------------------------------
    @property
    def ordinal(self):
        return self.inst.ordinal

    @property
    def id(self):
        return self.inst.id

    @property
    def class_name(self):
        return self.inst.class_name

    @property
    def fields(self):
        return self.inst.fields


class VentRuntime:
    """Sim-side runtime object for one ``vent`` (§2/§3).

    ``aperture_y``/``aperture_x`` are the GMAP-resolution tile the vent
    edits (§2's single aperture mechanism — floor: own tile, wall: the tile
    in front of ``facing``). ``duct`` is the resolved :class:`DuctRuntime`
    (``None`` when the ``duct`` ref is empty or dangling — an unwired vent
    builds but never sweeps, the bus-free-pump fail-safe precedent).
    ``rate_raw_per_s`` is ``q_circ`` quantized ONCE at load (door-2/pump
    idiom) into a raw-Q16.16-per-SECOND accrual rate; ``accum`` is the
    Bresenham flux-error accumulator (§3 Quantization) — the ONLY vent
    runtime row in patch 1.
    """

    __slots__ = ("inst", "aperture_y", "aperture_x", "role", "duct",
                 "rate_raw_per_s", "tps", "accum", "guard_skips", "alive")

    def __init__(self, inst, aperture_y, aperture_x, role, duct_rt,
                 rate_raw_per_s, tps):
        self.inst = inst
        self.aperture_y = int(aperture_y)
        self.aperture_x = int(aperture_x)
        self.role = role
        self.duct = duct_rt
        self.rate_raw_per_s = int(rate_raw_per_s)
        self.tps = int(tps)
        self.accum = 0                     # the synced §3 accumulator row
        self.guard_skips = 0               # diagnostic-only, NOT digested
        self.alive = True

    # --- serializer duck-type -------------------------------------------
    @property
    def ordinal(self):
        return self.inst.ordinal

    @property
    def id(self):
        return self.inst.id

    @property
    def class_name(self):
        return self.inst.class_name

    @property
    def fields(self):
        return self.inst.fields

    def accrue(self) -> int:
        """Bresenham accrual (§3 Quantization): add this tick's fixed
        per-second raw rate, emit the whole quanta crossed, keep the
        remainder — ``accum`` stays in ``[0, tps)``. Never manufactures flux
        below the rate (unlike a naive ``quantize(rate/tps)`` per-tick
        quantum, which can floor to 0 forever for a slow vent — the "tiny-
        flux underflow cliff" this accumulator exists to kill)."""
        self.accum += self.rate_raw_per_s
        emit = self.accum // self.tps
        self.accum -= emit * self.tps
        return emit


def _aperture_blocked(gmap, fy, fx) -> bool:
    """§4 runtime guards, checked EVERY tick (a load-time lint alone misses
    battle damage): solid / thermal_solid / vacuum / ambient / flooded. The
    flood check mirrors ``seal_tiles``' own invariant exactly (gamemap.py's
    water rule v1: ``water_depth != 0`` — no separate threshold dial)."""
    if gmap.solid[fy, fx]:
        return True
    if gmap.thermal_solid[fy, fx]:
        return True
    if gmap.is_vacuum[fy, fx]:
        return True
    if gmap.is_ambient[fy, fx]:
        return True
    if int(gmap.water_depth[fy, fx]) != 0:
        return True
    return False


def build_vents(sim):
    """Build the ordinal-ordered ``(ducts, vents)`` runtime lists for the
    9e(d) sweep and REPLACE each ``duct``/``vent`` instance in
    ``sim.entities`` with its runtime wrapper (so the plenum ledger / flux
    accumulator serialize, the door/pump §8 pattern).

    UNCONDITIONAL (unlike ``build_pumps``): vents carry no wired inputs in
    patch 1, so there is no SignalBus gate — a duct/vent-free level builds
    empty lists and the 9e(d) sweep call is a single ``if self._vents``
    check (dormancy, design §2/§7 "dormant-by-default").

    Ducts build FIRST (vents resolve their ``duct`` ref against the id map);
    each duct's ``filter`` field is resolved against ``[filters.<name>]``
    NOW (a config-integrity hard error, §4) — patch 1 ships no tuning, so
    this is validation, not a dial. Each vent's aperture is resolved via
    :func:`simulation.entities.vents.resolve_aperture_base` at BASE
    resolution, scaled by ``res_factor`` to the gmap grid (the door/pump/
    sensor S1 pattern) and bounds-checked hard.
    """
    level = sim.level
    gmap = sim.gmap
    rf = int(getattr(level, "res_factor", 1) or 1)
    tps = int(sim._tps)
    filters_cfg = getattr(CFG, "filters", None)

    duct_insts = sorted((e for e in sim.entities if e.class_name == "duct"),
                        key=lambda e: int(e.ordinal))
    # T-mix rails (§4), cached ONCE here (not re-read every tick, REVIEW FIX)
    # — a reset (F5/Ctrl+R) already rebuilds the whole vent runtime from a
    # fresh CFG, so this stays live across a hot-reload exactly like every
    # other build-time-bound constant in this module. Gated on ducts actually
    # existing: a duct-free level must stay a TRUE no-op (never even risk the
    # loud missing-config-key error below) — dormancy, §2/§7.
    sim._vent_t_min_q = sim._vent_t_max_phys_q = None
    if duct_insts:
        sim._vent_t_min_q, sim._vent_t_max_phys_q = _t_rails_q()

    ducts: list = []
    duct_by_id: dict = {}
    for e in duct_insts:
        eff_q = _duct_filter_efficiency_q(e.fields, filters_cfg, e.id)
        rt = DuctRuntime(e, eff_q)
        ducts.append(rt)
        duct_by_id[e.id] = rt
        _replace_entity(sim.entities, e, rt)

    # q_circ -> a raw-per-SECOND accrual rate (§3 Quantization): the SAME
    # C*T_amb calibration pump_system._delta_n_quantum uses (a gas-mass
    # quantum and a pressure-rate quantum coincide numerically at the
    # calibrated C*T_amb == 1), quantized ONCE here — never per-tick, never
    # raw in the editor (door-2/pump `rate` idiom).
    ct = float(DEFAULT_C) * float(DEFAULT_T_AMB_K)

    vent_insts = sorted((e for e in sim.entities if e.class_name == "vent"),
                        key=lambda e: int(e.ordinal))
    h, w = gmap.solid.shape
    vents: list = []
    for e in vent_insts:
        by, bx = resolve_aperture_base(e.fields)
        ay, ax = rf * by, rf * bx
        if not (0 <= ay < h and 0 <= ax < w):
            raise ValueError(
                f"vent '{e.id}': aperture tile ({ay}, {ax}) is out of the "
                f"{h}x{w} grid — check x/y + mount/facing (base tiles, "
                f"scaled by res_factor {rf})")
        duct_rt = duct_by_id.get(e.fields["duct"]) or None
        q_circ = float(e.fields["q_circ"])
        rate_per_s = q_circ / ct if ct > 0 else q_circ
        rate_raw_per_s = _gas_fx.quantize_scalar(rate_per_s)
        rt = VentRuntime(e, ay, ax, e.fields["role"], duct_rt,
                         rate_raw_per_s, tps)
        vents.append(rt)
        _replace_entity(sim.entities, e, rt)

    return ducts, vents


def _replace_entity(entities, old, new) -> None:
    """Swap ``old`` for its runtime wrapper ``new`` in the sim entity list,
    preserving ordinal position — mirrors
    :func:`simulation.pump_system._replace_entity` /
    :func:`simulation.door_system.build_runtime_entities`."""
    for i, e in enumerate(entities):
        if e is old:
            entities[i] = new
            return
    raise ValueError(                     # pragma: no cover - defensive
        f"vent-system entity {getattr(old, 'id', old)!r} not found in the "
        f"sim entity list")


def _t_rails_q():
    """``(T_MIN, T_MAX_PHYS)`` quantized ONCE — read at ``build_vents`` time
    and cached on the sim (REVIEW FIX: no longer re-read every tick; a reset
    — the F5/Ctrl+R reload path — already rebuilds the whole vent runtime
    from a fresh ``CFG``, so a per-tick re-read bought nothing but cost).

    LOUD on a missing config key (REVIEW FIX: was a silent -289.0/16000.0
    fallback) — ``[physics.eos].T_MIN`` / ``[physics.thermal].T_MAX_PHYS``
    are load-bearing rails for every OTHER T writer in the engine
    (physics_runner.py); a config that has dropped them is broken in a way
    that should fail here just as loudly as it would fail the EOS solver's
    own bind."""
    eos_cfg = getattr(CFG.physics, "eos", None)
    thermal_cfg = getattr(CFG.physics, "thermal", None)
    if eos_cfg is None or not hasattr(eos_cfg, "T_MIN"):
        raise ValueError(
            "vent design §4: [physics.eos].T_MIN is missing from config.toml "
            "— the vent T-mix rail has no fallback (it must agree with the "
            "EOS solver's own T_MIN bind, physics_runner.py)")
    if thermal_cfg is None or not hasattr(thermal_cfg, "T_MAX_PHYS"):
        raise ValueError(
            "vent design §4: [physics.thermal].T_MAX_PHYS is missing from "
            "config.toml — the vent T-mix rail has no fallback (it must "
            "agree with every other T writer's T_MAX_PHYS rail)")
    t_min = float(eos_cfg.T_MIN)
    t_max = float(thermal_cfg.T_MAX_PHYS)
    return _gas_fx.quantize_scalar(t_min), _gas_fx.quantize_scalar(t_max)


def _duct_sweep(gmap, duct, return_vents, supply_vents, t_min_q, t_max_phys_q):
    """One duct's per-tick circulation (§3): intakes first, then distribute.
    ``return_vents``/``supply_vents`` are this duct's member vents, already
    ordinal-sorted."""
    # --- 1. RETURN vents extract (ordinal order) ------------------------
    for v in return_vents:
        if not v.alive:
            continue
        fy, fx = v.aperture_y, v.aperture_x
        if _aperture_blocked(gmap, fy, fx):
            v.guard_skips += 1              # counted no-op; accum FROZEN
            continue
        emit = v.accrue()
        if emit <= 0:
            continue
        t_tile = int(gmap.temperature[fy, fx])       # honest pre-edit read
        removed = gmap.extract_gas_n_vec(fy, fx, emit)
        total_removed = sum(removed)
        if total_removed == 0:
            continue                        # aperture was empty — nothing credited
        duct.o2_raw += removed[O2]
        duct.n2_raw += removed[INERT_N2]
        for i, gid in enumerate(_TRACE_IDS):
            r = removed[gid]
            if r == 0:
                continue
            scrub = (r * duct.filter_eff_q[i]) >> 16    # truncating (mul_q16 idiom)
            passed = r - scrub
            duct.trace_raw[i] += passed
            duct.sink[i] += scrub
        # E_plenum credit is BULK-ONLY (REVIEW FIX, orchestrator-ruled): the
        # engine's own P-T0 convention is n_total == n_bulk (trace planes
        # carry no engine-side energy, cpp/src/eos_solver.cpp:719-729), so
        # crediting the trace share too (the v1 patch's "scrubbed smoke
        # keeps its heat" framing) manufactured energy out of nothing —
        # a scrubber-becomes-heater bug. Trace transport is mass-only.
        duct.e_plenum += (removed[O2] + removed[INERT_N2]) * t_tile

    # --- 2. SUPPLY vents accrue their own weight (ordinal order) --------
    ready = []                              # [(vent, w_i), ...]
    for v in supply_vents:
        if not v.alive:
            continue
        fy, fx = v.aperture_y, v.aperture_x
        if _aperture_blocked(gmap, fy, fx):
            v.guard_skips += 1
            continue
        w_i = v.accrue()
        if w_i > 0:
            ready.append((v, w_i))

    n_bulk = duct.o2_raw + duct.n2_raw
    if n_bulk < N_EPS:
        # Near-empty plenum (§4 N_EPS floor): the E/N divide is untrustworthy
        # — wipe the residual energy into the counted channel rather than
        # let a near-zero denominator amplify it into an absurd T. No
        # distribution this tick (there is essentially nothing to give).
        if duct.e_plenum != 0:
            duct.e_wipe += duct.e_plenum
            duct.e_plenum = 0
        return

    sum_w = sum(w for _, w in ready)
    total_avail = n_bulk if n_bulk < sum_w else sum_w
    if total_avail <= 0:
        return

    t_dep = duct.e_plenum // n_bulk          # floor(-inf) — Python `//` on ints IS
                                              # floordiv_q for a positive divisor
    total_o2_out = 0
    total_n2_out = 0
    total_trace_out = [0] * N_TRACE
    deposits = []                            # [(vent, comp[7]), ...]
    # REVIEW FIX (MAJOR): the bulk O2/N2 split for each vent's share is drawn
    # against the plenum's REMAINING [o2_rem, n2_rem] pool via
    # gas_proportional_split, DEDUCTED as each vent is processed — the ORIGINAL
    # per-vent "floor(share*o2_raw/n_bulk) + exact complement" trick is only
    # safe for a SINGLE split: `n2_i = share_i - o2_i` is share_i's leftover,
    # not a floor-bounded quantity, so across several supply vents on a
    # skewed ratio the summed n2_i can inflate past n2_raw (concrete:
    # o2_raw=99, n2_raw=1, two shares of 50 -> n2_raw goes to -1). Each
    # `gas_proportional_split` call, by contrast, is individually exact
    # (sums to share_i) AND bounded by what it's handed, so decrementing the
    # remaining pool after each call keeps every later split inside the
    # true remaining holdings — never negative.
    #
    # Trace does NOT need this treatment: `trace_i[g] = floor(share_i *
    # trace_raw[g] / n_bulk)`, computed independently per vent against the
    # ORIGINAL (undecremented) n_bulk/trace_raw, is already safe by floor
    # subadditivity — floor(a/n) + floor(b/n) <= floor((a+b)/n) for any
    # nonnegative a, b, n>0 — so summed across ALL ready vents,
    # Sum(trace_i[g]) <= floor(Sum(share_i) * trace_raw[g] / n_bulk) <=
    # trace_raw[g] (since Sum(share_i) <= total_avail <= n_bulk). No
    # remaining-pool bookkeeping needed there.
    o2_rem, n2_rem = duct.o2_raw, duct.n2_raw
    for v, w_i in ready:
        share_i = (total_avail * w_i) // sum_w
        if share_i <= 0:
            continue
        o2_i, n2_i = gmap.gas_proportional_split([o2_rem, n2_rem], share_i)
        o2_rem -= o2_i
        n2_rem -= n2_i
        trace_i = [(share_i * duct.trace_raw[i]) // n_bulk for i in range(N_TRACE)]
        comp = [0] * N_GASES
        for i, gid in enumerate(_TRACE_IDS):
            comp[gid] = trace_i[i]
        comp[O2] = o2_i
        comp[INERT_N2] = n2_i
        deposits.append((v, comp))
        total_o2_out += o2_i
        total_n2_out += n2_i
        for i in range(N_TRACE):
            total_trace_out[i] += trace_i[i]

    if not deposits:
        return                               # every share floored to 0 — nothing moves

    duct.o2_raw -= total_o2_out
    duct.n2_raw -= total_n2_out
    for i in range(N_TRACE):
        duct.trace_raw[i] -= total_trace_out[i]

    # Debit the plenum by exactly what the deposit HANDED OVER: `ΔN·t_dep`,
    # in the plenum's own RELATIVE currency.
    #
    # arc #54 P-G1b (design §2.7 pump row) REPLACES the old "measured tile-side
    # change `(N_old+ΔN)*T_new − N_old*T_old`" rule, and the review finding
    # that rule answered is now answered structurally instead of by
    # measurement. The old rule existed because `inject_gas_n_vec` MIXED
    # temperatures — a mass-weighted average with a floor-division remainder
    # and a rail clamp, neither visible from here — so the only safe debit was
    # to measure the result. There is no mix any more: the primitive converts
    # at the seam (`E = e + n·T_AMB`) and adds `ΔN·(t_dep + T_AMB)` to the
    # cell's stored energy, EXACTLY. So the honest relative debit is
    # `ΔN·t_dep`, exactly, with no remainder to bank back.
    #
    # A RAIL CLAMP IS NO LONGER THE PLENUM'S PROBLEM either: it is a counted
    # DESTRUCTION at the deposit site (GameMap books it to `pump_rail`), not
    # energy that stayed behind in the duct. Charging the plenum less because
    # the tile railed would have quietly re-created the destroyed energy in the
    # duct — the exact class of leak this arc exists to close. The hit counters
    # below still fire, so the diagnostic is unchanged.
    total_debit = 0
    for v, comp in deposits:
        fy, fx = v.aperture_y, v.aperture_x
        delta_n_bulk = comp[O2] + comp[INERT_N2]
        _t_new, rail_hit = gmap.inject_gas_n_vec(
            fy, fx, comp, t_dep, t_min_q, t_max_phys_q)
        total_debit += delta_n_bulk * t_dep
        if rail_hit < 0:
            duct.rail_lo_hits += 1
        elif rail_hit > 0:
            duct.rail_hi_hits += 1
    duct.e_plenum -= total_debit


def sweep_vents(sim) -> None:
    """9e(d): sweep every duct's circulation, sibling of ``sweep_pumps`` —
    same actuator slot, BEFORE the door structural sweep, independent of the
    SignalBus (§2: vents carry no wired inputs in patch 1)."""
    if not sim._ducts:
        return
    gmap = sim.gmap
    t_min_q, t_max_phys_q = sim._vent_t_min_q, sim._vent_t_max_phys_q
    by_duct: dict = {id(d): ([], []) for d in sim._ducts}
    for v in sim._vents:
        if v.duct is None:
            continue                         # unwired/dangling — inert (fail-safe)
        ret, sup = by_duct[id(v.duct)]
        (ret if v.role == "return" else sup).append(v)
    for duct in sim._ducts:                  # entity-ordinal order (§3)
        ret, sup = by_duct[id(duct)]
        _duct_sweep(gmap, duct, ret, sup, t_min_q, t_max_phys_q)
