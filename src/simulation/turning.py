"""The turning law (gameplay pack v1: G1p's turn-rate cap, G5p-a1's angular
inertia; ``docs/unit_impulse_design_2026-11-10.md`` §5.1 and §13).

THE ONLY WRITER OF A UNIT'S FACING. A unit's facing is the integer
``Unit.facing_q`` (Q16.16 RADIANS, the trig kit's own currency: the range
``atan2_q16`` returns, ``(-pi, pi]`` as ``(-PI_Q16, PI_Q16]``,
``fixed_point.h``); ``Unit.facing`` is a read-only float view of it for the
renderer, the pose layer and every other reader. Every reason to face
somewhere -- the live aim, the travel direction of a unit holding no aim, the
zombie's walk, the attack resolve's facing snap, the legacy WEGO and onephase
plans -- is a TARGET handed over through :func:`set_target`; the target
persists until something replaces it. Once per tick, in conductor slot 1a
(``unit_impulse.integrate_all``, after each unit's linear impulse step), every
living unit turns toward its target (the live aim's angle while a non-zero
live aim is held) through :func:`step`. Nothing else
writes ``_facing_q`` except :func:`place`, the one constructor-time door
(spawn: ``Unit.__init__``), which sets the facing AND the target at rest.

The law (§5.1), per tick, in integers, with ``a`` the unit's own angular
acceleration per tick^2 and ``w`` its own top speed per tick::

    e     = wrap(target - facing)
    omega = clamp(sign(e) * s(|e|),  omega - a,  omega + a)
    facing += omega

where ``s(e)`` is the largest speed in ``[0, w]`` whose braking distance
``B(x)`` (braking at ``a`` per tick from ``x``, this tick included) is at most
``e`` (:func:`top_speed_within`: the closed form per braking segment, exact
integers, at most ``ceil(w / a)`` trials -- bounded at the door). It reaches a
fixed target exactly, never overshoots and then stays (critique A, M3: once
``|e| <= a`` the next step equals ``e``). The top speed bounds the unit's OWN
torque only: an incoming ``omega`` above it (an outside spin, the impulse
law's ``omega += dw``, G5p-a2) decays at ``a`` per tick; the incoming
``omega`` is clamped below half a turn per tick so a spin cannot alias.

At ``a >= 2 w`` the clamp never binds and the law is G1p's bare cap,
:func:`turn_toward_q16`, bit for bit (``tests/test_turning_law.py``).

The predictor :func:`predict` is the pure evaluation of exactly :func:`step`
for one unit (``Simulation.facing_after_turn_q16``, the bots' "fire only when
this tick's turn lands within the spread" rule) -- never a copy of the law.

The wrap period is ``TWO_PI_Q16 = 2 * PI_Q16``, so the kit's two ends of the
range name the same angle (as atan2's do): the period is 0.8 LSB short of the
true 2 pi in Q16, far below the kit's own 9e-6 rad accuracy floor.

Settings: ``[marine] turn_rate_deg_per_s`` (the top speed) and
``[marine] turn_accel_rad_per_s2`` (the acceleration), PLACEHOLDERS Erik
tunes, read by :func:`turn_rate_q16` / :func:`turn_accel_q16` -- refusing
doors, quantized once at sim construction. A marine row today and a per-unit
stat later (dexterity, armour weight), so the sim reads them per unit
(``Simulation.unit_turn_rate_q16`` / ``unit_turn_accel_q16``); a zombie reads
the same two (it is a copy of the unit it infected).
"""
from __future__ import annotations

import math

from simulation.unit_fixed import atan2_q16, quantize_scalar

#: quantize(pi) -- the kit's pinned ``atan2_q16(0, x < 0)`` (fixed_point.h).
PI_Q16 = 205887
#: The wrap period: both range ends name the same angle (see the module doc).
TWO_PI_Q16 = 2 * PI_Q16
#: The largest incoming |omega|: below half a turn per tick, so a spin never
#: aliases (critique A, M5).
OMEGA_MAX_Q16 = PI_Q16 - 1
#: The door's bound on the trials of :func:`top_speed_within`,
#: ``ceil(min(w, PI_Q16) / a)`` (the device loop's fixed length; the shipped
#: dials need 6).
TURN_TRIALS_MAX = 32
#: The spawn facing: north, ``quantize(pi / 2)`` (the legacy "N" default).
FACING_NORTH_Q16 = quantize_scalar(math.pi / 2)


def wrap_q16(d: int) -> int:
    """An angle difference folded into ``(-PI_Q16, PI_Q16]`` -- the SHORTER
    way round; an exact half turn goes positive (counter-clockwise), so the
    choice is deterministic."""
    d = int(d) % TWO_PI_Q16          # Python: [0, TWO_PI_Q16)
    return d - TWO_PI_Q16 if d > PI_Q16 else d


def normalize_q16(a: int) -> int:
    """An angle folded into ``(-PI_Q16, PI_Q16]`` (the atan2 range, its two
    ends identified)."""
    return wrap_q16(a)


def turn_toward_q16(cur_q: int, target_q: int, cap_q: int) -> int:
    """G1p's bare cap: ``cur_q`` turned toward ``target_q`` by at most
    ``cap_q`` (all Q16.16 radians), the shorter way round. Within the cap it
    returns ``target_q`` ITSELF (no overshoot, no residual); beyond it it
    steps exactly ``cap_q``. ``cap_q`` must be > 0. Kept as the law's
    ``a >= 2 w`` special case (the gate :func:`step` is held to)."""
    d = wrap_q16(int(target_q) - int(cur_q))
    if -cap_q <= d <= cap_q:
        return int(target_q)
    return normalize_q16(int(cur_q) + (cap_q if d > 0 else -cap_q))


# ---------------------------------------------------------------------------
# The law
# ---------------------------------------------------------------------------
def braking_distance_q16(x: int, a: int) -> int:
    """``B(x)``: the angle covered braking from speed ``x >= 0`` at ``a`` per
    tick, this tick included: ``x + (x - a) + (x - 2a) + ...`` over the
    positive terms, ``(n + 1) x - a n (n + 1) / 2`` with
    ``n = ceil(x / a) - 1``. 0 at rest. Exact integers."""
    x, a = int(x), int(a)
    if x <= 0:
        return 0
    n = -(-x // a) - 1
    return (n + 1) * x - a * (n * (n + 1) // 2)


def top_speed_within(e: int, a: int, w: int) -> int:
    """``s(e)``: the largest speed in ``[0, w]`` whose braking distance is at
    most ``e >= 0``. On the braking segment ``x in (n a, (n + 1) a]`` the
    distance is ``(n + 1) x - a n (n + 1) / 2``, so the segment's largest
    admissible speed is ``floor((e + a n (n + 1) / 2) / (n + 1))``, capped at
    the segment's top and at ``w``; the answer is the highest segment that
    admits one. Tried for ``n = 0 .. ceil(w / a) - 1`` (the door bounds the
    count): exact integers, no search beyond that."""
    e, a, w = int(e), int(a), int(w)
    best = 0
    n = 0
    while n * a < w:
        lo = n * a
        hi = min((n + 1) * a, w)
        x = min(hi, (e + a * (n * (n + 1) // 2)) // (n + 1))
        if x > lo:
            best = x
        else:
            break                     # B is increasing: no higher segment fits
        n += 1
    return best


def law_q16(facing_q: int, omega_q: int, target_q: int, a: int, w: int):
    """One tick of the turning law (§5.1) as a pure function:
    ``(facing', omega')`` from the facing, the incoming angular velocity (an
    outside spin may be in it), the target and the unit's own ``a`` / ``w``
    (Q16.16 radians per tick^2 / per tick, both > 0). The incoming omega is
    clamped below half a turn per tick; the top speed bounds the own torque
    only (``s`` never exceeds ``min(w, PI_Q16)``), so a faster incoming omega
    decays at ``a`` per tick."""
    omega = max(-OMEGA_MAX_Q16, min(OMEGA_MAX_Q16, int(omega_q)))
    w = min(int(w), PI_Q16)
    a = int(a)
    e = wrap_q16(int(target_q) - int(facing_q))
    s = top_speed_within(abs(e), a, w)
    want = s if e >= 0 else -s
    omega = max(omega - a, min(omega + a, want))
    return normalize_q16(int(facing_q) + omega), omega


# ---------------------------------------------------------------------------
# The unit's facing state (the one writer)
# ---------------------------------------------------------------------------
def place(unit, angle_q: int) -> None:
    """THE constructor-time door: the facing AND its target set to
    ``angle_q`` (normalized), at rest (omega 0). Called by ``Unit.__init__``
    (spawn: north); a test fixture may call it to stand a unit facing
    somewhere before the first tick. Never a turn: nothing in the sim's tick
    calls it."""
    q = normalize_q16(angle_q)
    unit._facing_q = q
    unit.facing_target_q = q
    unit.omega_q = 0


def set_target(unit, angle_q: int) -> None:
    """The facing ``unit`` should turn to: every site that used to write the
    facing hands its angle (Q16.16 radians) here. The target persists until
    replaced; slot 1a turns toward it (:func:`step`)."""
    unit.facing_target_q = normalize_q16(angle_q)


def settle(unit) -> None:
    """The turn comes to rest where the unit faces: target = facing,
    omega 0 (a conversion: a revived corpse inherits no stale turn)."""
    unit.facing_target_q = unit._facing_q
    unit.omega_q = 0


def aim_target_q16(unit):
    """The live aim's angle (Q16.16 radians) while ``unit`` holds a non-zero
    live aim (a living non-zombie driven through ``Simulation.set_aim``),
    else None. World Y runs down and facing is math-style, hence ``-dy``."""
    if getattr(unit, "is_zombie", False):
        return None
    aim = getattr(unit, "live_aim", None)
    if aim is None or (aim.dx_q == 0 and aim.dy_q == 0):
        return None
    return atan2_q16(-int(aim.dy_q), int(aim.dx_q))


def predict(unit, target_q: int, a: int, w: int, dw: int = 0) -> int:
    """The facing ``unit`` has after this tick's turn toward ``target_q``:
    exactly what :func:`step` would write, without writing it (the bots'
    trigger rule, ``Simulation.facing_after_turn_q16``). ``dw`` is the
    outside spin slot 1a adds first (``unit_impulse.pending_dw``)."""
    return law_q16(unit._facing_q, unit.omega_q + int(dw), target_q, a, w)[0]


def ticks_to_within(unit, target_q: int, a: int, w: int, dw: int,
                    tol_q: int, cap: int) -> int:
    """The ticks until ``abs(wrap(target_q - facing)) <= tol_q`` for
    ``unit`` turning toward ``target_q`` under its own ``a`` / ``w``, from
    its facing and ``omega_q + dw`` (the outside spin its next slot 1a adds
    first) -- by ITERATING :func:`law_q16`, exactly the steps slot 1a would
    take, never a copy of the law. 0 when already within; ``cap`` when not
    reached in ``cap`` ticks (the observation's trigger aid, #195 1g: the
    entry's ``ticks_to_on``). A pure read."""
    facing, omega = int(unit._facing_q), int(unit.omega_q) + int(dw)
    target, tol, cap = int(target_q), int(tol_q), int(cap)
    n = 0
    while n < cap:
        if abs(wrap_q16(target - facing)) <= tol:
            return n
        facing, omega = law_q16(facing, omega, target, a, w)
        n += 1
    return cap


def step(unit, a: int, w: int, dw: int = 0) -> None:
    """One tick of the law for ``unit`` (slot 1a only, called by
    ``unit_impulse.integrate_all`` after the unit's linear integration):
    ``omega += dw`` (the outside spin the impulse law integrated, G5p-a2),
    the live aim, if one is held, becomes the target; then the facing and
    omega advance."""
    aim = aim_target_q16(unit)
    if aim is not None:
        set_target(unit, aim)
    unit._facing_q, unit.omega_q = law_q16(
        unit._facing_q, unit.omega_q + int(dw), unit.facing_target_q, a, w)


# ---------------------------------------------------------------------------
# The doors
# ---------------------------------------------------------------------------
def _marine_number(cfg, key: str):
    section = getattr(cfg, "marine", None)
    if section is None or not hasattr(section, key):
        raise ValueError(f"[marine] is missing {key!r}")
    v = getattr(section, key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
        raise ValueError(f"[marine] {key} must be a number > 0, got {v!r}")
    return float(v)


def turn_rate_q16(cfg) -> int:
    """``[marine] turn_rate_deg_per_s`` of a loaded config (``CFG``) as the
    per-TICK top speed in Q16.16 radians: ``quantize(radians(rate) / tps)``,
    quantized once. Refused (ValueError naming the key) when the section or
    key is missing, not a number, not > 0, or so small it quantizes to 0."""
    v = _marine_number(cfg, "turn_rate_deg_per_s")
    tps = int(cfg.clock.ticks_per_second)
    cap = quantize_scalar(math.radians(v) / tps)
    if cap < 1:
        raise ValueError(f"[marine] turn_rate_deg_per_s = {v!r} is below one "
                         f"Q16 radian count per tick at {tps} tps")
    return cap


def turn_accel_q16(cfg) -> int:
    """``[marine] turn_accel_rad_per_s2`` as the per-TICK^2 acceleration in
    Q16.16 radians: ``quantize(accel / tps^2)``, quantized once. Refused
    (naming the key) when missing, not a number, not > 0, below one count, or
    so small against the top speed that :func:`top_speed_within` would need
    more than :data:`TURN_TRIALS_MAX` trials (``ceil(min(w, pi) / a)``)."""
    v = _marine_number(cfg, "turn_accel_rad_per_s2")
    tps = int(cfg.clock.ticks_per_second)
    a = quantize_scalar(v / (tps * tps))
    if a < 1:
        raise ValueError(f"[marine] turn_accel_rad_per_s2 = {v!r} is below one "
                         f"Q16 radian count per tick^2 at {tps} tps")
    w = min(turn_rate_q16(cfg), PI_Q16)
    trials = -(-w // a)
    if trials > TURN_TRIALS_MAX:
        raise ValueError(
            f"[marine] turn_accel_rad_per_s2 = {v!r} is too small for the top "
            f"speed: the turning law would need {trials} trials a tick "
            f"(at most {TURN_TRIALS_MAX})")
    return a


__all__ = ["PI_Q16", "TWO_PI_Q16", "OMEGA_MAX_Q16", "TURN_TRIALS_MAX",
           "FACING_NORTH_Q16", "wrap_q16", "normalize_q16", "turn_toward_q16",
           "braking_distance_q16", "top_speed_within", "law_q16", "place",
           "set_target", "settle", "aim_target_q16", "predict",
           "ticks_to_within", "step", "turn_rate_q16", "turn_accel_q16"]
