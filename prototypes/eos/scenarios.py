"""
The 5 scenario builders (S1-S5) for the EOS Phase-1.2 visual prototype, per
docs/eos_research_brief.md §8.

Each `build_S*(width, height, seed) -> (State, schedule)` returns an
initialized `State` plus a `schedule`: a list of `(kind, tick, *args)` tuples.
`apply_event` is the generic dispatch the runner calls once per scheduled
event, at its tick, regardless of which `--scheme` is selected -- events are
solver-agnostic writes into `state` (mirroring how the real engine treats
explosions/breaches/doors as staged sources between solver steps, not
something living inside the solver: engine/07 §2.4).

Grid convention: every builder starts by filling `state.solid[:, :] = True`
(seal everything) and then carves open space inward, always leaving at least
`MARGIN` tiles of solid hull at the array's outer edge. Every roll-based
stencil in this codebase (diffusion, the water gradient) wraps periodically
at the array boundary; keeping that boundary permanently solid means the
wraparound neighbor is always a wall, so the existing wall-mirror handling
takes care of the domain edge for free -- no separate boundary-condition code
needed anywhere.
"""

import numpy as np

from state import State

MARGIN = 3


# ---------------------------------------------------------------------------
# Shared geometry/field helpers
# ---------------------------------------------------------------------------

def _lumpy_smoke(shape: tuple[int, int], region: tuple[int, int, int, int],
                  rng: np.random.Generator, n_blobs: int = 10,
                  amp_range: tuple[float, float] = (0.4, 1.0),
                  radius_frac: tuple[float, float] = (0.06, 0.16)) -> np.ndarray:
    """Sum of soft Gaussian blobs inside `region` (y0, y1, x0, x1) -- a patchy,
    non-uniform smoke cloud. Required by S4/S5: a flat blob shows nothing
    being pushed or shaped."""
    y0, y1, x0, x1 = region
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    smoke = np.zeros(shape, dtype=np.float32)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    span = min(x1 - x0, y1 - y0)
    for _ in range(n_blobs):
        bx = cx + rng.uniform(-0.35, 0.35) * (x1 - x0)
        by = cy + rng.uniform(-0.35, 0.35) * (y1 - y0)
        radius = rng.uniform(*radius_frac) * span
        amp = rng.uniform(*amp_range)
        r2 = (xx - bx) ** 2 + (yy - by) ** 2
        smoke += (amp * np.exp(-r2 / (2.0 * radius ** 2))).astype(np.float32)
    return smoke


def _radial_falloff(shape: tuple[int, int], x: int, y: int, radius: float):
    """Smooth radial falloff centred at tile (x, y), plus the unit direction
    field and raw distance -- shared by detonate/ignite."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    dxr, dyr = xx - x, yy - y
    r = np.sqrt(dxr * dxr + dyr * dyr)
    falloff = np.clip(1.0 - r / max(radius, 1e-3), 0.0, 1.0) ** 2
    return falloff, dxr, dyr, r


# ---------------------------------------------------------------------------
# S1 -- corridor blast
# ---------------------------------------------------------------------------

def build_s1(width: int, height: int, seed: int = 0) -> tuple[State, list[tuple]]:
    """Long walled corridor; detonation at the near end. Expect a planar
    front, reflection at the far (closed) door, then venting once it opens."""
    state = State(width, height)
    state.solid[:, :] = True

    half_h = max(height // 10, 2)
    cy0, cy1 = height // 2 - half_h, height // 2 + half_h + 1
    cx0 = MARGIN
    cx1 = width - MARGIN - 3   # end wall + door column + room for the vacuum stub
    state.solid[cy0:cy1, cx0:cx1] = False

    door_y0 = (cy0 + cy1) // 2 - 1
    door_y1 = door_y0 + 2
    state.door[door_y0:door_y1, cx1] = True   # closed (solid) until opened

    stub0, stub1 = cx1 + 1, width - MARGIN
    state.solid[cy0:cy1, stub0:stub1] = False
    state.vacuum[cy0:cy1, stub0:stub1] = True

    schedule = [
        ("detonate", 2, (cx0 + 3, (cy0 + cy1) // 2), 8.0),
        ("open_door", 45),
    ]
    return state, schedule


# ---------------------------------------------------------------------------
# S2 -- room + door jet
# ---------------------------------------------------------------------------

def build_s2(width: int, height: int, seed: int = 0) -> tuple[State, list[tuple]]:
    """Sealed room, blast trapped inside; a door opens later onto a short
    vent stub. Expect overpressure, then a jet through the doorway."""
    state = State(width, height)
    state.solid[:, :] = True

    y0, y1 = MARGIN, height - MARGIN
    x0 = MARGIN
    x1 = width - MARGIN - 6   # room's east wall, leaving room for the vent stub
    state.solid[y0:y1, x0:x1] = False

    door_y0 = (y0 + y1) // 2 - 1
    door_y1 = door_y0 + 2
    state.door[door_y0:door_y1, x1] = True   # closed (solid) until opened

    stub0, stub1 = x1 + 1, width - MARGIN
    state.solid[door_y0:door_y1, stub0:stub1] = False
    state.vacuum[door_y0:door_y1, stub0:stub1] = True

    schedule = [
        ("detonate", 2, (x0 + 4, (y0 + y1) // 2), 6.0),
        ("open_door", 30),
    ]
    return state, schedule


# ---------------------------------------------------------------------------
# S3 -- breach to vacuum
# ---------------------------------------------------------------------------

def build_s3(width: int, height: int, seed: int = 0) -> tuple[State, list[tuple]]:
    """Smoke-filled compartment next to a pre-seeded vacuum strip (space);
    a hull breach opens a hole between them. Expect sustained outrush."""
    state = State(width, height)
    state.solid[:, :] = True

    y0, y1 = MARGIN, height - MARGIN
    x0 = MARGIN
    x1 = width - MARGIN - 4   # compartment's east wall, room for the vacuum strip
    state.solid[y0:y1, x0:x1] = False
    state.smoke[y0:y1, x0:x1] = 0.8

    vx0, vx1 = x1 + 2, width - MARGIN
    state.solid[y0:y1, vx0:vx1] = False
    state.vacuum[y0:y1, vx0:vx1] = True

    breach_x, breach_y = x1, (y0 + y1) // 2
    schedule = [("breach", 15, (breach_x, breach_y))]
    return state, schedule


# ---------------------------------------------------------------------------
# S4 -- fireball over standing smoke
# ---------------------------------------------------------------------------

def build_s4(width: int, height: int, seed: int = 0) -> tuple[State, list[tuple]]:
    """Open bay with a patchy, non-uniform smoke cloud; ignition inside it.
    Expect the expansion to visibly push/shape the smoke."""
    rng = np.random.default_rng(seed)
    state = State(width, height)
    state.solid[:, :] = True

    y0, y1 = MARGIN, height - MARGIN
    x0, x1 = MARGIN, width - MARGIN
    state.solid[y0:y1, x0:x1] = False

    state.smoke = _lumpy_smoke(state.shape, (y0, y1, x0, x1), rng, n_blobs=10)
    state.smoke[state.solid] = 0.0
    np.clip(state.smoke, 0.0, 1.2, out=state.smoke)

    ignite_xy = (int((x0 + x1) / 2), int((y0 + y1) / 2))
    schedule = [("ignite", 10, ignite_xy, 10.0)]
    return state, schedule


# ---------------------------------------------------------------------------
# S5 -- water displacement pushes smoke (tilted ship)
# ---------------------------------------------------------------------------

def build_s5(width: int, height: int, seed: int = 0) -> tuple[State, list[tuple]]:
    """Tilted ship compartment split by a doorway; a water tank on the
    uphill (east) side bursts and floods downhill across the doorway into a
    patchy smoke cloud on the low (west) side."""
    rng = np.random.default_rng(seed + 1)
    state = State(width, height)
    state.solid[:, :] = True

    y0, y1 = MARGIN, height - MARGIN
    x0, x1 = MARGIN, width - MARGIN
    state.solid[y0:y1, x0:x1] = False

    mid_x = (x0 + x1) // 2
    state.solid[y0:y1, mid_x] = True
    gap0 = (y0 + y1) // 2 - 2
    gap1 = gap0 + 4
    state.solid[gap0:gap1, mid_x] = False   # doorway between the two halves

    # Ship tilt (engine/07 §2.3): positive x-tilt means surface rises with x,
    # so water accelerates toward -x (west). Tank sits east (high side),
    # smoke sits west (low side) -- the flood crosses the doorway into it.
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dx = state.tile_size_m
    tilt = (np.tan(np.radians(4.0)) * (xx - cx) * dx
            + np.tan(np.radians(1.5)) * (yy - cy) * dx)
    tilt[state.solid] = 0.0
    state.tilt = tilt.astype(np.float32)

    west_region = (y0, y1, x0, mid_x)
    state.smoke = _lumpy_smoke(state.shape, west_region, rng, n_blobs=8)
    state.smoke[state.solid] = 0.0
    np.clip(state.smoke, 0.0, 1.2, out=state.smoke)

    # Tank sits right beside the doorway (same row band as the gap) so the
    # dominant x-tilt drives the burst straight through it -- a tank placed
    # far from the doorway's row band would need the much weaker y-tilt to
    # walk it there first, which (verified numerically) does not happen in
    # a 120-tick run.
    tank = (gap0 - 1, gap1 + 1, mid_x + 2, min(mid_x + 7, x1 - 1))
    schedule = [("release_water", 15, tank, 0.6)]
    return state, schedule


SCENARIOS = {
    "S1": build_s1,
    "S2": build_s2,
    "S3": build_s3,
    "S4": build_s4,
    "S5": build_s5,
}


# ---------------------------------------------------------------------------
# Event application -- generic dispatch, solver-agnostic
# ---------------------------------------------------------------------------

def _apply_detonate(state: State, x: int, y: int, strength: float) -> None:
    """Stand-in for the engine's real event -> field staging (heat_dep,
    wave_p impulse). Bumps T/smoke and kicks velocity radially outward with
    a smooth falloff -- "energy staged for the solver", per engine/07 §2.4's
    pattern of sources being writes between solver steps, not solver-internal
    code."""
    radius = 6.0 + 0.4 * strength
    falloff, dxr, dyr, r = _radial_falloff(state.shape, x, y, radius)
    falloff = np.where(state.open_air, falloff, 0.0).astype(np.float32)

    state.T += strength * 400.0 * falloff
    state.smoke += strength * 0.6 * falloff
    ux = np.where(r > 1e-3, dxr / np.maximum(r, 1e-3), 0.0)
    uy = np.where(r > 1e-3, dyr / np.maximum(r, 1e-3), 0.0)
    state.vx += strength * 2.0 * falloff * ux
    state.vy += strength * 2.0 * falloff * uy


def _apply_breach(state: State, x: int, y: int, radius: int = 1) -> None:
    """Hull breach: knock a hole at (x, y), exposing whatever the scenario
    already placed on the other side (usually a pre-seeded vacuum strip).
    The hole itself is also marked vacuum so it reads instantly."""
    y0, y1 = max(y - radius, 0), y + radius + 1
    x0, x1 = max(x - radius, 0), x + radius + 1
    state.solid[y0:y1, x0:x1] = False
    state.vacuum[y0:y1, x0:x1] = True


def _apply_ignite(state: State, x: int, y: int, strength: float = 1.0) -> None:
    """Combustion event: strong local heating, a little smoke consumed
    (approximating soot conversion), and a gentle expansion kick -- enough
    for the hot air to visibly push the surrounding smoke."""
    radius = 4.0 + 0.3 * strength
    falloff, dxr, dyr, r = _radial_falloff(state.shape, x, y, radius)
    falloff = np.where(state.open_air, falloff, 0.0).astype(np.float32)

    state.T += strength * 900.0 * falloff
    state.smoke -= strength * 0.1 * falloff
    np.clip(state.smoke, 0.0, None, out=state.smoke)
    ux = np.where(r > 1e-3, dxr / np.maximum(r, 1e-3), 0.0)
    uy = np.where(r > 1e-3, dyr / np.maximum(r, 1e-3), 0.0)
    state.vx += strength * 0.8 * falloff * ux
    state.vy += strength * 0.8 * falloff * uy


def _apply_release_water(state: State, region: tuple[int, int, int, int],
                          amount: float) -> None:
    """Aquarium/tank burst (engine/07 §2.4): a one-shot deposit; the
    released column then flows under its own gradient. `region` is
    (y0, y1, x0, x1) in array-slice order."""
    y0, y1, x0, x1 = region
    state.water_depth[y0:y1, x0:x1] = np.maximum(state.water_depth[y0:y1, x0:x1], amount)


def _apply_open_door(state: State) -> None:
    """Opens every door tile the scenario placed (one door per scenario in P0)."""
    state.solid[state.door] = False


def apply_event(state: State, event: tuple) -> str:
    """Generic dispatch: mutate `state` for one scheduled event. Solver-
    agnostic -- run.py calls this once per scheduled event, at its tick,
    before that tick's solver.step, regardless of --scheme. Returns the
    event kind (for logging)."""
    kind = event[0]
    if kind == "detonate":
        _, _tick, (x, y), strength = event
        _apply_detonate(state, x, y, strength)
    elif kind == "breach":
        _, _tick, (x, y) = event
        _apply_breach(state, x, y)
    elif kind == "ignite":
        _, _tick, (x, y), *rest = event
        _apply_ignite(state, x, y, rest[0] if rest else 1.0)
    elif kind == "release_water":
        _, _tick, region, amount = event
        _apply_release_water(state, region, amount)
    elif kind == "open_door":
        _apply_open_door(state)
    else:
        raise ValueError(f"unknown event kind: {kind!r}")
    return kind
