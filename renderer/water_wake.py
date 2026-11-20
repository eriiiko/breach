"""Things in the water: a unit's RING and WAKE (water look arc #97, patch W3;
docs/water_look_design_2026-11-19.md W3 + its critique §9, M13 and the W3
minors). Pure numpy, pyray-free: the twin of what the renderer hands W2's
foam for the units standing or wading in water.

WHAT IT IS. A unit whose centre stands on a wet tile adds ONE foam splat
(``water_foam.splat_rate``'s capsule source) per frame:

    a = its centre at the previous sim tick, b = its centre now (tiles)
    r_c = wake_body_m / tile_m          (the ring round its body)
    hw  = wake_band_m / tile_m          (the ring's half-width)
    rate = wake_ring_rate + wake_gain x speed^2      (amount per second)

  * STANDING (``a == b``, speed 0): the RING alone, a thin band of foam round
    the body at the surface (Erik's ruling, 2026-11-20: the waterline is the
    body seen through the water plus a foam ring at the surface).
  * WALKING: the band sweeps the tick's path (the capsule), so the foam it
    leaves behind is the WAKE, carried and decayed by W2's step like any
    foam. Its extra rate grows with the speed SQUARED: a cell beside the path
    takes the band for ``~2 r_c / speed`` seconds, so the foam a wake leaves
    per metre of path grows LINEARLY with the speed. A still unit makes no
    wake; a dry tile makes nothing.
  * THE SPEED is the unit's CENTRE's displacement per sim tick (the sim's own
    positions, kept per unit id here on the render side: never ``PoseState``,
    never a velocity on ``Unit``), held between ticks, divided by the ticks
    between two seen positions, and forgotten when the tick goes BACK (a
    replay seek: the first tick after it is a standing start).

Never a render-side ripple field, never a second wave source (critique M13):
the wake is foam, one more source term of THE foam. Picture state only.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .water_foam import SPLAT_COLS, SPLAT_MAX


def ring_rate(speed_mps, s) -> np.ndarray:
    """The splat's rate (amount per second) for a unit moving at
    ``speed_mps``: ``wake_ring_rate + wake_gain x speed^2`` (the ring, plus
    the wake that grows with the speed; ``s`` a ``WaterSettings``)."""
    v = np.asarray(speed_mps, np.float64)
    return s.wake_ring_rate + s.wake_gain * v * v


def wake_rate(speed_mps, s) -> np.ndarray:
    """The WAKE's share of :func:`ring_rate`: ``wake_gain x speed^2`` --
    exactly 0 for a still unit."""
    v = np.asarray(speed_mps, np.float64)
    return s.wake_gain * v * v


def unit_splat(a: Tuple[float, float], b: Tuple[float, float],
               speed_mps: float, depth_m: float, tile_m: float,
               s) -> Optional[np.ndarray]:
    """One unit's splat row ``(ax, ay, bx, by, r_c, hw, rate)`` (tiles,
    amount/s), or None when its centre's tile is dry (``depth_m <= 0``)."""
    if not depth_m > 0.0:
        return None
    t = float(tile_m)
    return np.array([a[0], a[1], b[0], b[1], s.wake_body_m / t,
                     s.wake_band_m / t, float(ring_rate(speed_mps, s))])


def landing_dabs(points, tile_m: float, s) -> np.ndarray:
    """ONE foam dab per landed spray droplet (``points`` (k, 2) tiles, the
    spray's ``landings_last_frame``): ``(x, y, x, y, 0, splash_radius_m /
    tile, splash_foam)`` -- a tent of peak ``splash_foam`` (an AMOUNT,
    spread over a frame's foam steps by ``water_foam.frame_splats``)."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    out = np.zeros((pts.shape[0], SPLAT_COLS))
    out[:, 0] = out[:, 2] = pts[:, 0]
    out[:, 1] = out[:, 3] = pts[:, 1]
    out[:, 5] = s.splash_radius_m / float(tile_m)
    out[:, 6] = s.splash_foam
    return out


def wet_near(depth: Optional[np.ndarray], x: float, y: float) -> bool:
    """Is the tile under ``(x, y)`` (tiles) or one of its eight neighbours
    wet? (A round that stopped on a wall beside the water splashes.)"""
    if depth is None:
        return False
    h, w = depth.shape
    ix, iy = int(np.floor(x)), int(np.floor(y))
    y0, y1 = max(iy - 1, 0), min(iy + 2, h)
    x0, x1 = max(ix - 1, 0), min(ix + 2, w)
    if y1 <= y0 or x1 <= x0:
        return False
    return bool((np.asarray(depth[y0:y1, x0:x1]) > 0).any())


class WakeTracker:
    """Per unit id (render side), the centre at the last sim tick it was
    seen and the tick before's: the tick's path and the unit's speed.
    :meth:`splats` is called once per frame with the drawn units."""

    def __init__(self) -> None:
        # uid -> (tick, (x, y) now, (x, y) the tick before, speed m/s)
        self._seen: Dict[int, tuple] = {}
        self._tick: Optional[int] = None

    def reset(self) -> None:
        self._seen.clear()
        self._tick = None

    def observe(self, uid: int, centre: Tuple[float, float], tick: int,
                tile_m: float, tps: float) -> Tuple[tuple, tuple, float]:
        """Record ``uid``'s centre (tiles) at sim ``tick``; return the tick's
        path ``(a, b)`` and the speed (m/s). A first sighting, or a sighting
        after the unit was missed for more than one tick, is a standing
        start (a == b, speed 0)."""
        tick = int(tick)
        c = (float(centre[0]), float(centre[1]))
        prev = self._seen.get(uid)
        if prev is None:
            rec = (tick, c, c, 0.0)
        elif prev[0] == tick:
            rec = prev                       # the same tick: hold
        elif tick - prev[0] == 1:
            d = float(np.hypot(c[0] - prev[1][0], c[1] - prev[1][1]))
            rec = (tick, c, prev[1], d * float(tile_m) * float(tps))
        else:
            rec = (tick, c, c, 0.0)          # a gap: no path to sweep
        self._seen[uid] = rec
        return rec[2], rec[1], rec[3]

    def splats(self, units: Iterable[Tuple[int, float, float]], tick: int,
               depth: Optional[np.ndarray], tile_m: float, tps: float,
               s) -> np.ndarray:
        """This frame's splats ((n, 7), at most ``water_foam.SPLAT_MAX``) for
        ``units`` -- ``(uid, centre x, centre y)`` in tiles, the living units
        drawn -- over the packed per-tile ``depth`` (m; None = a dry map:
        nothing). A tick that went BACK forgets every unit (a seek)."""
        tick = int(tick)
        if self._tick is not None and tick < self._tick:
            self.reset()
        self._tick = tick
        rows = []
        live = set()
        for uid, cx, cy in units:
            live.add(uid)
            a, b, v = self.observe(uid, (cx, cy), tick, tile_m, tps)
            if depth is None or len(rows) >= SPLAT_MAX:
                continue
            h, w = depth.shape
            iy, ix = int(np.floor(cy)), int(np.floor(cx))
            if not (0 <= iy < h and 0 <= ix < w):
                continue
            row = unit_splat(a, b, v, float(depth[iy, ix]), tile_m, s)
            if row is not None:
                rows.append(row)
        for uid in [u for u in self._seen if u not in live]:
            del self._seen[uid]              # gone (dead, despawned)
        if not rows:
            return np.zeros((0, SPLAT_COLS))
        return np.stack(rows)


__all__ = ["ring_rate", "wake_rate", "unit_splat", "landing_dabs",
           "wet_near", "WakeTracker"]
