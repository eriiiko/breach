"""Per-tick continuous intents — the direct-control order vocabulary (P3).

Design: ``docs/control_modularity_design_2026-07-22.md`` §3c. WEGO keeps its
tile-targeted *queued* orders (:mod:`simulation.orders`); direct control adds
these *continuous* intents, consumed the tick they are issued and never
queued. A :class:`~simulation.ruleset.ContinuousRealtime` sim reads them each
tick; a :class:`~control_source.ControlSource` (``GamepadDirect``) or an
``AgentPolicy`` produces them.

Determinism (IRON RULE): an intent is a SYNCED input, so every direction and
angle it carries lives in the sim as **Q16.16 integers only** — no floats, no
libm in the sim path. The control layer (gamepad sticks, mouse) quantizes its
float axes to fixed-point **at the control/facade seam, before the intent
enters the sim** (see ``control_source.quantize_stick_direction``); the sim
consumes fixed-point and dequantizes with the exact power-of-two divide
``q / 65536.0`` only where it feeds already-float, non-transcendental position
math (``Unit.x``/``y``) or the deterministic integer trig kit
(``simulation.unit_fixed`` — ``atan2_rad``/``sin_rad``/``cos_rad``). No
``math.sin``/``cos``/``atan2`` ever runs on an intent inside the sim.

The ``dx_q``/``dy_q`` pair of :class:`MoveDirIntent` / :class:`AimIntent` /
:class:`ThrowIntent` is a Q16.16 **unit vector** (the control seam normalizes
before quantizing), so a common scale factor never leaks a speed dependency
into direction. ``speed_mode`` is one of the movement order-type ids
(``ORDER_MOVE_ATTACK`` / ``ORDER_MOVE_COVER`` / ``ORDER_SPRINT``) so the
continuous move branch reuses the exact same per-order base cadence WEGO uses.
"""
from __future__ import annotations

from dataclasses import dataclass

# Q16.16 fixed-point scale — the same shift used everywhere in the sim
# (fire_fixed / gas_fixed / unit_fixed). Kept local so this module has no
# import-time dependency on the compiled physics kit.
FP_SHIFT = 16
FP_ONE = 1 << FP_SHIFT          # 65536
FP_ONE_F = float(FP_ONE)


def dequantize(q: int) -> float:
    """Q16.16 int -> float. Exact: ``n / 65536`` is a power-of-two divide, so
    the result is an exact double, IEEE-identical on every machine (no libm).
    The ONLY dequantize the sim's continuous-intent consumers use."""
    return q / FP_ONE_F


@dataclass
class MoveDirIntent:
    """Move the possessed unit by velocity this tick (§3c, the new mechanic).

    ``dx_q``/``dy_q`` — Q16.16 unit-vector direction (normalized at the control
    seam). ``speed_mode`` — a movement order-type id whose base ticks-per-tile
    cadence the continuous branch scales by the footprint's area-average
    mobility, exactly as the WEGO A* replay does.
    """

    dx_q: int
    dy_q: int
    speed_mode: int


@dataclass
class AimIntent:
    """Continuous facing (§3c). ``dx_q``/``dy_q`` — Q16.16 unit-vector aim
    direction. The sim turns it into ``Unit.facing`` through the deterministic
    integer ``atan2`` kit (``unit_fixed.atan2_rad``), never ``math.atan2`` —
    facing is synced state hashed by the lockstep digest."""

    dx_q: int
    dy_q: int


@dataclass
class ThrowIntent:
    """Lob a grenade along ``dx_q``/``dy_q`` (Q16.16 unit vector), fusing after
    ``fuse_seconds`` (§3c THROW). Edge-triggered: latched by the facade until
    the next tick consumes it, so a button tap between two ticks is never
    dropped and never double-fired."""

    dx_q: int
    dy_q: int
    fuse_seconds: float


__all__ = [
    "FP_SHIFT", "FP_ONE", "FP_ONE_F", "dequantize",
    "MoveDirIntent", "AimIntent", "ThrowIntent",
]
