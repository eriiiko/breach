"""L0 logic-node classes — Arc B patch B2 (impl doc §5, node set).

Design: docs/arc_b_impl_2026-07-21.md §5 (v2, D5 folded). This module carries
the SCHEMA (fields / signals / inputs) for the pure-logic node classes, plus
THE exact filter time-constant snap (§5 D5 — the iron-rule fix). The runtime
evaluation (prev-read/next-write over the SignalBus) lives sim-side in
:mod:`simulation.logic_nodes`; this package stays IMPORT-LIGHT (stdlib only —
design §3b, CI-tested in tests/test_entities_import_light.py). The only stdlib
import here beyond the schema is :mod:`fractions`, exactly like
:mod:`simulation.entities.door`'s span quantization.

The node set (§5):

- ``decider`` — compares a single input against a threshold (6 comparators);
  optional ``require_alive`` fail-passive gating on the source entity's life.
- ``gate_and`` / ``gate_or`` — many-wire AND / OR reductions.
- ``gate_not`` — logical negation of a single input.
- ``filter`` — an integer EMA low-pass with the exact ``k``-snap below.

All are ``LOGIC_NODE = True``: their ``out`` signal is a prev-read/next-write
node output the SignalBus SWAPS at 9e(e) (one tick per hop, §2c). Nodes are
pure logic — they carry NO ``x``/``y`` and never occupy a tile — so the class
default is ``INTANGIBLE = True`` (the sensible node default, §5): a decider is
not a physical object. Runtime state (the filter EMA accumulator) rides
``runtime_digest_rows`` (the A4 mechanism) — hashed / recorded, ABSENT (zero
bytes) when the class is not instantiated (dormancy, §8).
"""
from __future__ import annotations

from fractions import Fraction

from simulation.entities.schema import (
    Entity, Field, INPUT_AND, INPUT_HELD, INPUT_SINGLE, InputDecl, KIND_BOOL,
    KIND_ENUM, KIND_INT, KIND_LENGTH_M, KIND_Q16, Signal, register,
)

# The six comparator tokens (decider `comparator` enum, §5). The runtime
# evaluation (logic_nodes.py) maps each to an integer predicate.
COMPARATORS = ("gt", "ge", "lt", "le", "eq", "ne")


# ---------------------------------------------------------------------------
# THE exact filter time-constant snap (§5, D5 — the iron-rule fix).
# ---------------------------------------------------------------------------

def snap_filter_k(tau_s, ticks_per_second) -> int:
    """§5 (D5): the EMA right-shift ``k`` = the integer NEAREST
    ``log2(tau_s * tps)``, computed by EXACT ``Fraction`` arithmetic — NEVER
    float ``log2``/``round`` (a cross-machine ULP flip in ``log2`` would enter
    synced state). Mirrors :func:`simulation.entities.door.quantize_span_tiles`:
    no float in the load path feeding synced sim state.

    ``tau_s`` ingresses as ``Fraction(str(float(tau_s)))`` — the DECIMAL the
    author typed, not the binary float's expansion (the N10 pin, shared with
    the door span quantizer). ``k`` is clamped ``>= 0`` (a sub-tick time
    constant degenerates to no smoothing, ``alpha = 1``).

    The snap is round-half-UP in log space, decided by an EXACT integer
    comparison: ``k`` rounds up past a level iff ``x >= 2^(k+1/2)`` iff
    ``x^2 >= 2^(2k+1)`` — pure integer/Fraction, no transcendental, so it is
    bit-identical on every machine. Example (24 tps): ``tau_s`` from 0 up to
    ``sqrt(2)/24 ≈ 0.0589 s`` snaps to ``k=0``; the exact boundary
    ``x = sqrt(2)`` (``x^2 == 2``) rounds UP to ``k=1`` where a naive float
    ``round(log2(1.41421356...))`` could land either side.
    """
    x = Fraction(str(float(tau_s))) * Fraction(int(ticks_per_second))
    if x <= 0:
        return 0                          # sub-tick / zero tau => no smoothing
    x2 = x * x                            # exact; compare against 2^(2k+1)
    k = 0
    while x2 >= (1 << (2 * k + 1)):
        k += 1
    return k


# ---------------------------------------------------------------------------
# The node classes (§5).
# ---------------------------------------------------------------------------

@register
class decider(Entity):
    """Compare a single input against a threshold (§5).

    ``out`` = ``cmp(in, threshold)`` AND (``require_alive`` => source alive).
    ``in`` is ``SINGLE`` (the driving wire's Q16.16 value verbatim, arity 1).
    """

    INTANGIBLE = True    # pure logic, no tile (§5 — the sensible node default)
    LOGIC_NODE = True

    FIELDS = (
        Field("comparator", KIND_ENUM, default="gt", choices=COMPARATORS,
              doc="the predicate cmp(in, threshold): gt|ge|lt|le|eq|ne"),
        Field("threshold", KIND_Q16, default=0,
              doc="Q16.16 threshold in the input's physical unit (quantized "
                  "once at load by the meters-first rule)"),
        Field("require_alive", KIND_BOOL, default=False,
              doc="when true the `out` is ANDed with the source entity's live "
                  "state — a dead source zeroes the decider this tick "
                  "(fail-passive, §2d). Default false = fail-deadly."),
    )
    SIGNALS = (Signal("out", "1 iff cmp(in, threshold) [and source alive]"),)
    INPUTS = (InputDecl("in", INPUT_SINGLE,
                        "the compared value (single driving wire, Q16.16)"),)


@register
class gate_and(Entity):
    """Many-wire AND: ``out`` = 1 iff EVERY driving wire != 0 (empty => 0)."""

    INTANGIBLE = True
    LOGIC_NODE = True

    SIGNALS = (Signal("out", "1 iff all driving inputs != 0 (empty => 0)"),)
    INPUTS = (InputDecl("in", INPUT_AND, "the AND'd inputs (many wires)"),)


@register
class gate_or(Entity):
    """Many-wire OR: ``out`` = 1 iff ANY driving wire != 0."""

    INTANGIBLE = True
    LOGIC_NODE = True

    SIGNALS = (Signal("out", "1 iff any driving input != 0"),)
    INPUTS = (InputDecl("in", INPUT_HELD, "the OR'd inputs (many wires)"),)


@register
class gate_not(Entity):
    """Logical negation: ``out`` = ``1 - (in != 0)`` (single input)."""

    INTANGIBLE = True
    LOGIC_NODE = True

    SIGNALS = (Signal("out", "1 - (in != 0)"),)
    INPUTS = (InputDecl("in", INPUT_SINGLE, "the negated value (single wire)"),)


@register
class filter(Entity):
    """Integer EMA low-pass over a single input (§5, D5).

    ``out`` (Q16.16) is the exponential moving average of ``in`` with smoothing
    ``alpha = 1/2^k``, ``k`` snapped once at load by :func:`snap_filter_k`. The
    runtime carries the accumulator at ``k`` guard bits and rounds-to-nearest
    before the shift (kills the truncation-park bug, §5) — see
    :class:`simulation.logic_nodes.FilterRuntime`.
    """

    INTANGIBLE = True
    LOGIC_NODE = True

    FIELDS = (
        Field("tau_s", KIND_LENGTH_M, default=1.0, minimum=0.0,
              doc="EMA time constant in seconds (length_m-style authoring "
                  "number); k = round(log2(tau_s*tps)) is snapped once at "
                  "load by EXACT Fraction arithmetic (§5 D5). The editor "
                  "shows the snapped effective tau."),
    )
    SIGNALS = (Signal("out", "Q16.16 EMA of in"),)
    INPUTS = (InputDecl("in", INPUT_SINGLE, "the filtered value (single wire, "
                        "Q16.16)"),)

    @classmethod
    def runtime_digest_rows(cls, entity) -> tuple:
        """§5/§8: the EMA accumulator (WITH guard bits) is a synced runtime
        row. One ``ema`` row off the runtime object (a bare EntityInstance has
        no ``ema`` and raises AttributeError — loud; digests only come from
        constructed sims, mirroring the door). ABSENT when no ``filter`` is
        instantiated (no runtime object => no row => dormancy, §8)."""
        return (("ema", int(entity.ema)),)


# ---------------------------------------------------------------------------
# The pump — an N-feed actuator (§6, D10/D11). NOT a logic node (it edits the
# gmap gas field at 9e(d), unlike the pure-SignalBus nodes) and NOT a sensor;
# marked ``PUMP`` so :mod:`simulation.pump_system` (the sim-side runtime) can
# find it. The gas-N primitive it drives lives on GameMap
# (``inject_gas_n``/``extract_gas_n``); the per-tick ΔN quantum + the D11
# band-skip assert are computed once at load in pump_system.build_pumps.
# ---------------------------------------------------------------------------

# The default anti-chatter hysteresis half-band, ±0.05 atm, as Q16.16
# (quantize(0.05) == 3277). Declared here so the schema default and the D11
# load-time assert (ΔN_per_tick_atm < 2·band) read the same constant.
PUMP_DEFAULT_BAND_Q16 = 3277


@register
class pump(Entity):
    """Gas-mass (N) feed actuator (§6b, post-EOS). Drives the port tile's gas
    up (``inject``) or down (``extract``) by a fixed per-tick quantum ΔN toward
    ``target_atm``, and emits ``at_target`` (hysteresis-banded, anti-chatter).

    The PORT tile (``port_dx``/``port_dy`` offset from the mount ``x``/``y``) is
    DISTINCT from the solid body it mounts on — the edit lands on open air, not
    on the wall the skip-solid mask would veto (§6). ``inject``/``extract`` are
    OR/held; both-active resolves EXTRACT-BEATS-INJECT (the safe depressurize
    state — the close-beats-open analogue), pinned via ``INPUT_PRIORITY``.
    """

    INTANGIBLE = False   # a placed actuator with a mount tile (like a sensor)
    PUMP = True          # the pump_system marker (parallels LOGIC_NODE/SENSOR)

    FIELDS = (
        Field("x", KIND_INT, default=None, minimum=0,
              doc="mount tile COL at base resolution — REQUIRED"),
        Field("y", KIND_INT, default=None, minimum=0,
              doc="mount tile ROW at base resolution — REQUIRED"),
        Field("port_dx", KIND_INT, default=0,
              doc="PORT tile COL offset from the mount (base tiles): the open-"
                  "air tile the pump edits, distinct from the solid body (§6)"),
        Field("port_dy", KIND_INT, default=0,
              doc="PORT tile ROW offset from the mount (base tiles)"),
        Field("rate", KIND_LENGTH_M, default=1.0, minimum=0.0,
              doc="feed rate in atm/s (authoring number, quantized once at "
                  "load into the per-tick ΔN quantum — the filter tau_s idiom); "
                  "must satisfy rate/tps < 2·band or load hard-errors (D11)"),
        Field("target_atm", KIND_Q16, default=65536,
              doc="target pressure at the port tile (Q16.16 atm; 65536 == 1.0 "
                  "atm) — at_target latches when the port P reaches this band"),
        Field("hysteresis_band", KIND_Q16, default=PUMP_DEFAULT_BAND_Q16,
              doc="at_target half-band in Q16.16 atm (default 3277 == ±0.05 "
                  "atm) — the anti-chatter Schmitt band; ΔN_per_tick_atm must "
                  "be < 2·band or the port tile can jump the band and "
                  "at_target never latches (airlock deadlock, D11)"),
    )
    SIGNALS = (Signal("at_target",
                      "1 while the port pressure is within the hysteresis band "
                      "of target_atm (latched Schmitt, anti-chatter)"),)
    INPUTS = (InputDecl("inject", INPUT_HELD, "held: raise the port tile's N"),
              InputDecl("extract", INPUT_HELD, "held: lower the port tile's N"))
    INPUT_PRIORITY = (("extract", "inject"),)   # extract beats inject (safe)

    @classmethod
    def runtime_digest_rows(cls, entity) -> tuple:
        """§8: the latched ``at_target`` hysteresis state is a synced runtime
        row (path-dependent — it cannot be re-derived from P alone). Read off
        the :class:`simulation.pump_system.PumpRuntime` wrapper.

        Tolerant ``getattr`` (default 0) UNLIKE the door/filter loud path: a
        pump is swept ONLY when a SignalBus exists (it needs inject/extract
        wires), and a pump needs wires to be useful — so a bus-free level's
        pump is a bare, never-swept EntityInstance whose at_target is
        definitionally 0 (inert by construction, not a missing-runtime bug)."""
        return (("at_target", int(getattr(entity, "at_target", 0))),)
